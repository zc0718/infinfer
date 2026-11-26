#!/usr/bin/env python3
"""The `autovec` claim is about the compiler, so it has to be checked on the compiler's output.

`autovec` is not a code tier: the same scalar bodies are the implementation, and the only thing
that can be true or false is "the shipped flags actually vectorised them". Measured on this
tree (2026-11-26), `-O2` produced zero packed instructions for add/relu/scale/fc while `-O3`
produced 5/5/8/7 -- so the earlier claim that "the kernel ABI blocks vectorisation" (gaps G-48)
was wrong: what blocks it is the optimisation level, and `restrict`/`ivdep` change nothing at
`-O3`. This gate keeps that sentence honest by re-measuring it.

Rules:

  I1 every file listed under `expect` compiles with the declared flags and its disassembly
     contains at least `min_packed` packed (SIMD) instructions
  I2 every file listed under `degraded` carries a non-empty reason -- a degradation that is
     not written down is indistinguishable from a claim
  I3 on a host without SSE2 there is nothing to measure: SKIP with the reason, never pass
     silently

`--selftest` proves I1 can fail on a body the compiler cannot vectorise, and pass on one it can.
"""

import argparse
import json
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


def _run(cmd, **kwargs):
    # stdin is closed: `cc -dM -E -` reads standard input, and inheriting the caller's
    # pipe makes the gate hang forever instead of reporting a missing compiler.
    kwargs.setdefault("stdin", subprocess.DEVNULL)
    return subprocess.run(cmd, capture_output=True, text=True, **kwargs)


def count_packed(path, compiler, flags, workdir):
    """Compile one translation unit and count packed instructions in its .text."""
    obj = str(pathlib.Path(workdir) / (pathlib.Path(path).stem + ".o"))
    done = _run([compiler, *flags, "-I", str(ROOT / "include"), "-c", str(ROOT / path), "-o", obj])
    if done.returncode != 0:
        return None, (done.stderr.strip().splitlines() or ["compile failed"])[-1]
    dis = _run(["objdump", "-d", obj])
    if dis.returncode != 0:
        return None, (dis.stderr.strip().splitlines() or ["objdump failed"])[-1]
    return len(PACKED_RE.findall(dis.stdout)), None


def check(data, workdir, compiler="cc"):
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

        problems, _ = check(
            {"flags": flags, "expect": {str(opaque): {"min_packed": 1}}, "degraded": {}},
            workdir,
        )
        if any("at least 1" in p for p in problems):
            print("[isa:selftest] PASS I1 the rule itself reports the shortfall")
        else:
            failures += 1
            print(f"[isa:selftest] FAIL I1 expected a shortfall report, got {problems}")

        problems, _ = check(
            {"flags": flags, "expect": {}, "degraded": {str(opaque): "  "}},
            workdir,
        )
        if any("without a reason" in p for p in problems):
            print("[isa:selftest] PASS I2 a degradation with no reason is red")
        else:
            failures += 1
            print(f"[isa:selftest] FAIL I2 expected a missing-reason report, got {problems}")
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
    probe = _run(["cc", "-dM", "-E", "-"])
    if "__SSE2__" not in probe.stdout and probe.returncode == 0:
        print("[isa] SKIP: the host compiler has no SSE2, so the packed-instruction count is not "
              "meaningful here (the tier-vs-scalar parity tests cover behaviour separately)")
        return 0

    workdir = tempfile.mkdtemp(prefix="isa-gate-")
    try:
        problems, reported = check(data, workdir)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    if problems:
        for problem in problems:
            print(f"[isa] {problem}", file=sys.stderr)
        print(f"[isa] FAIL ({len(problems)} problem(s))", file=sys.stderr)
        return 1
    print(f"[isa] OK ({len(reported)} file(s): {data.get('tier', 'autovec')} is real on this host)")
    for line in reported:
        print(f"[isa]   {line}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
