#!/usr/bin/env python3
"""Two claims about the tiers, both measured on disassembly instead of remembered.

**Claim 1 -- `autovec`.** `autovec` is not a code tier: the same scalar bodies are the
implementation, and the only thing that can be true or false is "the shipped flags actually
vectorised them". Measured on this tree (2026-11-26), `-O2` produced zero packed instructions for
add/relu/scale/fc while `-O3` produced 5/5/8/7 -- so the earlier claim that "the kernel ABI blocks
vectorisation" (gaps G-48) was wrong: what blocks it is the optimisation level, and
`restrict`/`ivdep` change nothing at `-O3`.

**Claim 2 -- the hand-written tiers.** `x86`, `arm_neon`, `rvv` and `arm_m` claim to be vector (or
DSP) code, and nothing used to check that. A body labelled `x86` that is really a scalar loop is
green everywhere else in this tree: the parity tests compare *values*, and a scalar loop produces
the right values. The one place the difference shows is the instruction stream.

The sharp part is the optimisation level, and the two claims sit on opposite sides of it on
purpose:

  * claim 1 is measured at `-O3`, because it is a statement about what the *compiler* does;
  * claim 2 is measured at `-O0`, where GCC's auto-vectoriser is off -- so a packed instruction
    can only come from an explicit intrinsic. `-O0` therefore separates "this body uses the ISA"
    from "this body happens to be vectorisable", and a scalar loop cannot pass it by accident.

Rules:

  I1 every file listed under `expect` compiles with the declared flags and its disassembly
     contains at least `min_packed` packed (SIMD) instructions
  I2 every file listed under `degraded` carries a non-empty reason -- a degradation that is
     not written down is indistinguishable from a claim
  I3 on a host without SSE2 there is nothing to measure: SKIP with the reason, never pass
     silently
  I4 every `src/kernel/<handwritten tier>/*.c` file is either measured (`handwritten.expect`) or
     excluded by name with a reason -- a new body that nobody measured is red, because the
     failure mode is a file that is labelled a tier and is not one
  I5 every measured body reaches `min_packed` at `-O0` under its own toolchain and ISA mnemonics
  I6 a tier whose compiler cannot be found is SKIPped **with the reason and the search paths it
     tried**; if nothing at all was measured or skipped the gate refuses to print OK
  I7 an `expect` entry whose file no longer exists is red (a stale measurement claims coverage
     that expired)

`--selftest` proves each rule can fail on a body it should reject and pass on one it should accept.
"""

import argparse
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA = ROOT / ".github/project/isa_expectations.json"

# SSE2-era mnemonics that only appear when a loop was packed. Deliberately generous: the gate
# asks "was this body vectorised", not "was it vectorised exactly the way I expected".
PACKED_RE = re.compile(
    r"\b(addps|subps|mulps|divps|maxps|minps|andps|orps|xorps|movups|movaps|unpcklps|unpckhps|"
    r"shufps|movhlps|movlhps|cvtdq2ps|cvtps2dq|paddd|psubd|pmullw|pmulld|pmaddwd|packssdw|"
    r"packuswb|pshufd|punpckldq|punpckhdq)\b"
)

# Same instruction family, but the wider set the hand-written x86 bodies use (int64 accumulate,
# byte compares, unaligned moves). Kept separate so `autovec`'s thresholds are not silently
# loosened by everything the intrinsics tier needs.
X86_RE = re.compile(
    r"\b(addps|subps|mulps|divps|maxps|minps|andps|orps|xorps|movups|movaps|movdqu|movdqa|"
    r"unpcklps|unpckhps|shufps|movhlps|movlhps|cvtdq2ps|cvtps2dq|paddd|psubd|paddq|pmullw|"
    r"pmulld|pmaddwd|psrad|packssdw|packuswb|pshufd|punpckldq|punpckhdq|punpcklwd|punpckhwd|"
    r"pcmpgtd|pcmpgtb|pand|por|pxor)\b"
)

# AArch64/NEON: a vector register operand carrying a lane size only exists in SIMD instructions
# (`v0.4s`, `v3.16b`, ...). Scalar FP code writes `s0`/`d0`, which this does not match, so the
# count cannot be inflated by ordinary floating point.
NEON_RE = re.compile(r"\bv[0-9]+\.(4s|2d|16b|8b|8h|4h|2s|1d)\b")

# RVV: vector instructions are the ones with a `.vv`/`.vx`/`.vi`/`.vf`/`.vs` suffix (or a
# whole-vector memory op / configuration instruction). No scalar instruction has those forms.
RVV_RE = re.compile(
    r"\bv(setvli|setivli|setvl|le[0-9]+\.v|se[0-9]+\.v|fmv\.|mv\.v\.x|mv\.x\.v)\b"
    r"|\b[a-z0-9]+\.(vv|vx|vi|vf|vs|wv|wf)\b"
)

# Cortex-M DSP: the multiply-accumulate forms the arm_m int8/int16 dot products exist for.
ARMM_RE = re.compile(r"\b(smlad|smladx|smlald|smlaldx|smlal|smlabb|smlabt|usad8|sadd16)\b")

# Which regular expression answers "is this body using its ISA" for each tier.
TIER_RE = {"x86": X86_RE, "arm_neon": NEON_RE, "rvv": RVV_RE, "arm_m": ARMM_RE}


def _run(cmd, **kwargs):
    # stdin is closed: `cc -dM -E -` reads standard input, and inheriting the caller's
    # pipe makes the gate hang forever instead of reporting a missing compiler.
    kwargs.setdefault("stdin", subprocess.DEVNULL)
    return subprocess.run(cmd, capture_output=True, text=True, **kwargs)


def count_packed(path, compiler, flags, workdir, objdump="objdump", pattern=PACKED_RE, root=None):
    """Compile one translation unit and count packed instructions in its .text.

    `root` exists so the self-test can point the same rule at a synthetic tree; resolving the
    fixture against the real repository was the first version of this bug, and it showed up as
    "did not build" rather than as a wrong answer.
    """
    base = pathlib.Path(root) if root is not None else ROOT
    obj = str(pathlib.Path(workdir) / (pathlib.Path(path).stem + ".o"))
    done = _run([compiler, *flags, "-I", str(base / "include"), "-c", str(base / path), "-o", obj])
    if done.returncode != 0:
        return None, (done.stderr.strip().splitlines() or ["compile failed"])[-1]
    dis = _run([objdump, "-d", obj])
    if dis.returncode != 0 or not re.search(r"^\s+[0-9a-f]+:", dis.stdout, re.M):
        return None, (dis.stderr.strip().splitlines() or ["objdump produced no disassembly"])[-1]
    return len(pattern.findall(dis.stdout)), None


def resolve_tool(spec):
    """Find a compiler named by the data file.

    `host` means the machine's own `cc`. A house toolchain is looked for in three places, in
    order, and the caller records which one answered -- "it built on my machine" is not a fact
    about the repository unless the gate says where the compiler came from.
    """
    if spec == "host":
        found = shutil.which("cc")
        return (found, "objdump", "PATH") if found else (None, None, "PATH (cc not found)")
    binary = spec.split("house:", 1)[1] if spec.startswith("house:") else spec
    env_key = "INFR_ISA_" + re.sub(r"[^A-Z0-9]+", "_", binary.upper()) + "_CC"
    if os.environ.get(env_key):
        cc = os.environ[env_key]
        return cc, cc.replace("-gcc", "-objdump"), f"${env_key}"
    on_path = shutil.which(binary)
    if on_path:
        return on_path, on_path.replace("-gcc", "-objdump"), "PATH"
    cache = pathlib.Path.home() / ".conan2/p"
    for pattern in (f"b/*/p/bin/{binary}", f"*/p/bin/{binary}"):
        for hit in sorted(cache.glob(pattern)):
            return str(hit), str(hit).replace("-gcc", "-objdump"), "~/.conan2 cache"
    return None, None, f"PATH, ${{{env_key}}}, ~/.conan2 cache"


def check_autovec(data, workdir, compiler="cc"):
    problems = []
    reported = []
    for path, spec in sorted(data.get("expect", {}).items()):
        minimum = spec.get("min_packed")
        if not isinstance(minimum, int) or minimum < 1:
            problems.append(f"isa: '{path}' must declare a positive min_packed")
            continue
        count, error = count_packed(path, compiler, data["flags"], workdir)
        if error:
            problems.append(f"isa: '{path}' did not build: {error}")
            continue
        if count < minimum:
            problems.append(
                f"isa: '{path}' has {count} packed instruction(s) but the autovec claim needs "
                f"at least {minimum} -- the body is being built as scalar"
            )
        else:
            reported.append(f"{path}: {count} >= {minimum}")
    for path, reason in sorted(data.get("degraded", {}).items()):
        if not isinstance(reason, str) or not reason.strip():
            problems.append(f"isa: '{path}' is declared degraded without a reason")
        else:
            reported.append(f"{path}: degraded ({reason[:60]}...)")
    return problems, reported


def check_handwritten(data, workdir, root=None):
    """Rules I4-I7: the hand-written tiers really are hand-written."""
    root = pathlib.Path(root) if root is not None else ROOT
    problems = []
    reported = []
    skipped = []
    hw = data.get("handwritten")
    if not isinstance(hw, dict):
        return ["isa: the data file has no 'handwritten' section, so no tier body is checked"], [], []

    tiers = hw.get("tiers", {})
    expect = hw.get("expect", {})
    excluded = hw.get("excluded", {})

    # I7 -- a measurement that points at a file nobody has any more.
    for path in sorted(expect):
        if not (root / path).exists():
            problems.append(
                f"isa: handwritten.expect lists '{path}' but the file does not exist -- a stale "
                f"measurement claims coverage that expired"
            )

    # I4 -- every body in a tier directory is accounted for.
    for tier in sorted(tiers):
        tier_dir = root / "src/kernel" / tier
        if not tier_dir.is_dir():
            problems.append(f"isa: tier '{tier}' is declared but src/kernel/{tier}/ does not exist")
            continue
        for source in sorted(tier_dir.glob("*.c")):
            key = str(source.relative_to(root)).replace(os.sep, "/")
            if key in expect:
                continue
            reason = excluded.get(key)
            if isinstance(reason, str) and reason.strip():
                reported.append(f"{key}: excluded ({reason[:50]}...)")
                continue
            problems.append(
                f"isa: '{key}' is a body in the {tier} tier but is neither measured nor excluded "
                f"-- a file labelled a tier that nobody checked is the failure mode this rule is for"
            )

    # I5 -- measure each declared body with its own toolchain and mnemonics.
    for tier in sorted(tiers):
        spec = tiers[tier]
        compiler, objdump, where = resolve_tool(spec.get("cc", "host"))
        members = [p for p, s in expect.items() if s.get("tier") == tier]
        if compiler is None:
            # I6 -- not silently green: name the tier, the compiler and where it looked.
            skipped.append(
                f"{tier}: compiler '{spec.get('cc')}' not found (tried {where}); "
                f"{len(members)} body/bodies unmeasured on this host"
            )
            continue
        pattern = TIER_RE.get(tier)
        if pattern is None:
            problems.append(f"isa: tier '{tier}' has no mnemonic set in the gate, so it cannot be judged")
            continue
        for path in sorted(members):
            entry = expect[path]
            minimum = entry.get("min_packed")
            if not isinstance(minimum, int) or minimum < 1:
                problems.append(f"isa: '{path}' must declare a positive min_packed")
                continue
            count, error = count_packed(
                path, compiler, spec["flags"], workdir, objdump, pattern, root=root
            )
            if error:
                problems.append(f"isa: '{path}' did not build with {pathlib.Path(compiler).name}: {error}")
                continue
            if count < minimum:
                problems.append(
                    f"isa: '{path}' has {count} {tier} instruction(s) at -O0 but needs at least "
                    f"{minimum} -- the body does not use its ISA (a scalar loop produces the same "
                    f"values, which is why the parity tests cannot catch this)"
                )
            else:
                measured = entry.get("measured")
                delta = "" if measured in (None, count) else f", was {measured} when measured"
                reported.append(
                    f"{tier}/{pathlib.Path(path).name}: {count} >= {minimum} @-O0 "
                    f"({pathlib.Path(compiler).name} from {where}{delta})"
                )

    for path, reason in sorted(excluded.items()):
        if not isinstance(reason, str) or not reason.strip():
            problems.append(f"isa: '{path}' is excluded without a reason")

    if not reported and not problems and not skipped:
        problems.append("isa: the handwritten section declared nothing at all")
    return problems, reported, skipped


def selftest():
    failures = 0
    workdir = tempfile.mkdtemp(prefix="isa-selftest-")
    try:
        vectorisable = pathlib.Path(workdir) / "packed.c"
        opaque = pathlib.Path(workdir) / "scalar.c"
        vectorisable.write_text(
            "void f(const float *a, const float *b, float *y, unsigned n) {\n"
            "    for (unsigned i = 0; i < n; ++i) { y[i] = a[i] + b[i]; }\n"
            "}\n",
            encoding="utf-8",
        )
        opaque.write_text(
            "float g(float);\n"
            "void f(const float *a, float *y, unsigned n) {\n"
            "    for (unsigned i = 0; i < n; ++i) { y[i] = g(a[i]); }\n"
            "}\n",
            encoding="utf-8",
        )
        intrinsics = pathlib.Path(workdir) / "intrinsic.c"
        intrinsics.write_text(
            "#include <emmintrin.h>\n"
            "void f(const float *a, const float *b, float *y) {\n"
            "    __m128 x = _mm_loadu_ps(a);\n"
            "    __m128 z = _mm_loadu_ps(b);\n"
            "    _mm_storeu_ps(y, _mm_add_ps(x, z));\n"
            "}\n",
            encoding="utf-8",
        )
        flags = ["-O3", "-std=c11"]
        for name, source, minimum, want_problem in (
            ("I1 a loop that packs is green", vectorisable, 1, False),
            ("I1 a loop that cannot pack is red", opaque, 1, True),
        ):
            count, error = count_packed(str(source), "cc", flags, workdir)
            if error:
                failures += 1
                print(f"[isa:selftest] FAIL {name}: {error}")
                continue
            if (count < minimum) == want_problem:
                print(f"[isa:selftest] PASS {name} ({count} packed)")
            else:
                failures += 1
                print(f"[isa:selftest] FAIL {name}: {count} packed, unexpected")

        problems, _ = check_autovec(
            {"flags": flags, "expect": {str(opaque): {"min_packed": 1}}, "degraded": {}},
            workdir,
        )
        if any("at least 1" in p for p in problems):
            print("[isa:selftest] PASS I1 the rule itself reports the shortfall")
        else:
            failures += 1
            print(f"[isa:selftest] FAIL I1 expected a shortfall report, got {problems}")

        problems, _ = check_autovec(
            {"flags": flags, "expect": {}, "degraded": {str(opaque): "  "}}, workdir
        )
        if any("without a reason" in p for p in problems):
            print("[isa:selftest] PASS I2 a degradation with no reason is red")
        else:
            failures += 1
            print(f"[isa:selftest] FAIL I2 expected a missing-reason report, got {problems}")

        # --- I5's premise: the claim that -O0 separates intrinsics from a scalar loop. This is
        # the fixture that matters, because the whole second claim rests on it: if -O0 vectorised
        # the scalar loop, "packed at -O0" would not mean "written with intrinsics".
        zero = ["-O0", "-std=c11", "-msse2"]
        count_intr, error_intr = count_packed(str(intrinsics), "cc", zero, workdir, "objdump", X86_RE)
        count_scalar, error_flat = count_packed(str(opaque), "cc", zero, workdir, "objdump", X86_RE)
        if error_intr or error_flat:
            failures += 1
            print(f"[isa:selftest] FAIL I5 fixtures did not build: {error_intr or error_flat}")
        elif count_intr > 0 and count_scalar == 0:
            print(
                f"[isa:selftest] PASS I5 -O0 keeps intrinsics ({count_intr} packed) and drops the "
                f"scalar loop ({count_scalar})"
            )
        else:
            failures += 1
            print(
                f"[isa:selftest] FAIL I5 -O0 does not separate the two bodies (intrinsics "
                f"{count_intr}, scalar {count_scalar}) -- the rule would be blind"
            )

        # --- I4-I7 on a synthetic tier tree, so the fixtures do not depend on the real one.
        fake = pathlib.Path(workdir) / "tree"
        (fake / "src/kernel/x86").mkdir(parents=True, exist_ok=True)
        (fake / "src/kernel/arm_neon").mkdir(parents=True, exist_ok=True)
        (fake / "src/kernel/x86/body.c").write_text(
            intrinsics.read_text(encoding="utf-8"), encoding="utf-8"
        )
        (fake / "src/kernel/x86/stray.c").write_text(
            opaque.read_text(encoding="utf-8"), encoding="utf-8"
        )
        base = {
            "handwritten": {
                "tiers": {"x86": {"cc": "host", "flags": zero}},
                "expect": {"src/kernel/x86/body.c": {"tier": "x86", "min_packed": 2}},
                "excluded": {},
            }
        }
        problems, _, _ = check_handwritten(base, workdir, root=fake)
        if any("neither measured nor excluded" in p and "stray.c" in p for p in problems):
            print("[isa:selftest] PASS I4 an undeclared body in a tier directory is red")
        else:
            failures += 1
            print(f"[isa:selftest] FAIL I4 expected the undeclared body to be named, got {problems}")

        excluded = json.loads(json.dumps(base))
        excluded["handwritten"]["excluded"] = {"src/kernel/x86/stray.c": "fixture: no kernel body"}
        problems, _, _ = check_handwritten(excluded, workdir, root=fake)
        if not problems:
            print("[isa:selftest] PASS I4 excluding it by name with a reason is green")
        else:
            failures += 1
            print(f"[isa:selftest] FAIL I4 expected a clean run once excluded, got {problems}")

        blank_reason = json.loads(json.dumps(excluded))
        blank_reason["handwritten"]["excluded"] = {"src/kernel/x86/stray.c": "  "}
        problems, _, _ = check_handwritten(blank_reason, workdir, root=fake)
        if any("excluded without a reason" in p for p in problems):
            print("[isa:selftest] PASS I4 an exclusion without a reason is red")
        else:
            failures += 1
            print(f"[isa:selftest] FAIL I4 expected a missing-reason report, got {problems}")

        # I5 on the real rule path: declare a body that cannot reach the floor.
        short = json.loads(json.dumps(base))
        short["handwritten"]["expect"]["src/kernel/x86/stray.c"] = {"tier": "x86", "min_packed": 2}
        problems, _, _ = check_handwritten(short, workdir, root=fake)
        if any("does not use its ISA" in p and "stray.c" in p for p in problems):
            print("[isa:selftest] PASS I5 a declared body with no ISA instructions is red")
        else:
            failures += 1
            print(f"[isa:selftest] FAIL I5 expected the shortfall report, got {problems}")

        # I6: an absent compiler must be named, never turned into a pass.
        absent = {
            "handwritten": {
                "tiers": {"arm_neon": {"cc": "house:no-such-gcc", "flags": zero}},
                "expect": {"src/kernel/x86/body.c": {"tier": "arm_neon", "min_packed": 1}},
                "excluded": {"src/kernel/x86/stray.c": "fixture"},
            }
        }
        problems, _, skipped = check_handwritten(absent, workdir, root=fake)
        if not problems and any("not found" in s for s in skipped):
            print("[isa:selftest] PASS I6 a missing compiler is a named SKIP, not a silent pass")
        else:
            failures += 1
            print(
                f"[isa:selftest] FAIL I6 expected a named skip, got problems={problems} "
                f"skipped={skipped}"
            )

        # I7: a measurement pointing at a file that is gone.
        stale = json.loads(json.dumps(excluded))
        stale["handwritten"]["expect"]["src/kernel/x86/gone.c"] = {"tier": "x86", "min_packed": 1}
        problems, _, _ = check_handwritten(stale, workdir, root=fake)
        if any("stale measurement" in p for p in problems):
            print("[isa:selftest] PASS I7 a stale measurement is red")
        else:
            failures += 1
            print(f"[isa:selftest] FAIL I7 expected a stale-measurement report, got {problems}")
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    if failures:
        print(f"[isa:selftest] FAIL ({failures} fixture(s))", file=sys.stderr)
        return 1
    print("[isa:selftest] OK")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default=str(DATA), help="expectations file")
    parser.add_argument("--selftest", action="store_true", help="prove the rules can fail")
    args = parser.parse_args()

    if args.selftest:
        return selftest()

    data = json.loads(pathlib.Path(args.data).read_text(encoding="utf-8"))
    workdir = tempfile.mkdtemp(prefix="isa-gate-")
    try:
        probe = _run(["cc", "-dM", "-E", "-"])
        has_sse2 = "__SSE2__" in probe.stdout and probe.returncode == 0
        problems = []
        reported = []
        skipped = []
        if has_sse2:
            autovec_problems, autovec_reported = check_autovec(data, workdir)
            problems += autovec_problems
            reported += autovec_reported
        else:
            # I3 -- the host cannot answer claim 1; say so instead of passing quietly.
            skipped.append(
                "autovec: the host compiler has no SSE2, so the packed-instruction count is not "
                "meaningful here (the tier-vs-scalar parity tests cover behaviour separately)"
            )
        hw_problems, hw_reported, hw_skipped = check_handwritten(data, workdir)
        problems += hw_problems
        reported += hw_reported
        skipped += hw_skipped
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    if problems:
        for problem in problems:
            print(f"[isa] {problem}", file=sys.stderr)
        print(f"[isa] FAIL ({len(problems)} problem(s))", file=sys.stderr)
        return 1
    for line in skipped:
        print(f"[isa] SKIP {line}")
    if not reported:
        print(
            "[isa] FAIL (nothing was measured and nothing was skipped with a reason)",
            file=sys.stderr,
        )
        return 1
    print(f"[isa] OK ({len(reported)} measurement(s), {len(skipped)} skip(s))")
    for line in reported:
        print(f"[isa]   {line}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
