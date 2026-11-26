#!/usr/bin/env python3
"""AddressSanitizer + UBSan over the library and a driver that drives its edges.

**Why this gate exists.** Two defects in one week were invisible to the compiler
and to the tests: a positional aggregate initialiser that shifted silently when a
field was inserted, and a requantisation path that converted an out-of-range
double to an integer. Both are the class of bug a sanitizer finds by construction
rather than by argument, and both were found by something else being stricter
than expected. This gate makes that strictness routine.

**What it builds.** Every `.c` and `.cpp` under `src/` (discovered, not listed --
a hand-written source list is a second truth that goes stale the day a file is
added) plus `.github/project/sanitize/smoke.c`, which drives the public ABI at its
edges: NULL arguments for every registered row, shape rejections, a workspace
that refuses to over-allocate or to rewind past its mark, a graph operator on an
arena that cannot hold its scratch, and a few hundred deterministic geometries
through conv2d.

**Why not the conan build.** The `project-gates` job in CI has a checkout and
`bash ci.sh`, nothing else: no conan, no gtest. A driver that only this gate
builds keeps the check honest instead of turning it into a skipped step, and it
takes about a second.

**What this gate cannot know.** It proves nothing about behaviour beyond what the
driver asserts -- the unit suite is still the behavioural oracle. It also cannot
see code paths the driver never enters; that limit is why the driver leans on the
hostile inputs rather than on volume.

Usage:

    python3 .github/project/sanitize_gate.py              # build, run, judge
    python3 .github/project/sanitize_gate.py --selftest   # prove it can go red
    python3 .github/project/sanitize_gate.py --keep       # keep the build dir
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DRIVER = Path(__file__).resolve().with_name("sanitize") / "smoke.c"

# The inference core, declared as roots rather than filtered as files: the template's
# own demo units (`src/cpptest.cpp`, `src/etl.cpp`) pull Eigen and ETL from the conan
# cache, which a CI runner does not have, and neither carries inference code. A file
# list would go stale the day someone adds a kernel; a root list does not.
SOURCE_ROOTS = ("src/kernel", "src/runtime", "src/abi")

SAN_FLAGS = ["-g", "-O1", "-fno-omit-frame-pointer", "-fsanitize=address,undefined", "-fno-sanitize-recover=all"]
C_FLAGS = ["-std=c11", "-I", str(ROOT / "include")]
CXX_FLAGS = ["-std=c++17", "-I", str(ROOT / "include")]
LINK_FLAGS = ["-fsanitize=address,undefined", "-lm"]

# The markers a report starts with. Checking for them *as well as* the exit code is
# deliberate: `halt_on_error` is what makes UBSan abort, and a future flag change
# must not turn a printed report into a green run.
MARKERS = (
    "ERROR: AddressSanitizer",
    "ERROR: LeakSanitizer",
    "runtime error:",
    "SUMMARY: UndefinedBehaviorSanitizer",
)

ENV_PROBE = """
int main(void) { int a[2]; a[0] = 1; return a[0] - 1; }
"""


def compiler(env_name, fallback):
    return os.environ.get(env_name, fallback)


def run(command, **kwargs):
    return subprocess.run(command, capture_output=True, text=True, **kwargs)


def supports_sanitizers(cc, workdir):
    """A compiler without the sanitizer runtime is an environment gap, said out loud."""
    probe = Path(workdir) / "probe.c"
    probe.write_text(ENV_PROBE, encoding="utf-8")
    binary = Path(workdir) / "probe"
    result = run([cc, *C_FLAGS, *SAN_FLAGS, str(probe), "-o", str(binary)])
    if result.returncode != 0:
        return False, (result.stderr or result.stdout).strip().splitlines()[-1:]
    return True, []


def sources():
    """Every translation unit under the declared roots, discovered rather than listed."""
    found = []
    for root in SOURCE_ROOTS:
        base = ROOT / root
        if not base.exists():
            print(f"[sanitize] skip {root}/ (absent)")
            continue
        found.extend(sorted(list(base.rglob("*.c")) + list(base.rglob("*.cpp"))))
    return found


def object_name(path):
    """A collision-free object name: `scalar/register.c` and `arm_m/register.c` differ."""
    return path.relative_to(ROOT).as_posix().replace("/", "_").replace(".", "_") + ".o"


def build(cc, cxx, workdir, driver=DRIVER):
    """Compile every source plus the driver; returns (objects, problems)."""
    problems = []
    objects = []
    units = sources()
    if not units:
        return [], ["sanitize: no sources found under " + ", ".join(SOURCE_ROOTS)]
    for path in units:
        obj = Path(workdir) / object_name(path)
        flags = C_FLAGS if path.suffix == ".c" else CXX_FLAGS
        tool = cc if path.suffix == ".c" else cxx
        result = run([tool, *flags, *SAN_FLAGS, "-c", str(path), "-o", str(obj)])
        if result.returncode != 0:
            problems.append(f"sanitize: {path.relative_to(ROOT)} does not compile:\n{result.stderr.strip()}")
            continue
        objects.append(obj)
    driver_obj = Path(workdir) / "sanitize_smoke.o"
    result = run([cc, *C_FLAGS, *SAN_FLAGS, "-c", str(driver), "-o", str(driver_obj)])
    if result.returncode != 0:
        problems.append(f"sanitize: the driver does not compile:\n{result.stderr.strip()}")
    else:
        objects.append(driver_obj)
    return objects, problems


def link_and_run(cxx, workdir, objects, name="sanitize_smoke"):
    """Link, run, and return (problems, transcript)."""
    binary = Path(workdir) / name
    result = run([cxx, *LINK_FLAGS, *[str(obj) for obj in objects], "-o", str(binary)])
    if result.returncode != 0:
        return [f"sanitize: link failed:\n{result.stderr.strip()}"], ""
    env = dict(os.environ)
    env["ASAN_OPTIONS"] = env.get("ASAN_OPTIONS", "detect_leaks=1:abort_on_error=1")
    env["UBSAN_OPTIONS"] = env.get("UBSAN_OPTIONS", "print_stacktrace=1:halt_on_error=1")
    result = run([str(binary)], env=env)
    transcript = (result.stdout or "") + (result.stderr or "")
    problems = []
    hits = sorted({marker for marker in MARKERS if marker in transcript})
    if hits:
        problems.append(f"sanitize: the run reported {', '.join(hits)}:\n{transcript.strip()}")
    if result.returncode != 0 and not hits:
        problems.append(f"sanitize: the driver exited {result.returncode}:\n{transcript.strip()}")
    return problems, transcript


def selftest(cc, cxx):
    """Prove the detector is live: a clean fixture is green, a real bug is caught.

    Both halves matter. A gate that only shows a red would also be satisfied by a
    detector that fires on everything.
    """
    clean = """
#include <stdlib.h>
int main(void) {
    int *p = malloc(4U * sizeof(int));
    int i;
    if (p == 0) { return 2; }
    for (i = 0; i < 4; ++i) { p[i] = i; }
    free(p);
    return 0;
}
"""
    heap_overflow = """
#include <stdlib.h>
int main(void) {
    const int n = 4;
    int *p = malloc((size_t)n * sizeof(int));
    if (p == 0) { return 2; }
    p[n] = 1;  /* one past the end */
    free(p);
    return 0;
}
"""
    signed_overflow = """
#include <limits.h>
int main(void) {
    int big = INT_MAX;
    volatile int one = 1;   /* volatile: the UB must survive to run time */
    big += one;
    return big == 0;
}
"""
    fixtures = [
        ("a clean fixture", clean, False),
        ("a heap overflow", heap_overflow, True),
        ("signed overflow", signed_overflow, True),
    ]
    failures = 0
    with tempfile.TemporaryDirectory(prefix="sanitize-selftest-") as workdir:
        base = Path(workdir)
        for index, (label, text, must_go_red) in enumerate(fixtures):
            source = base / f"case{index}.c"
            source.write_text(text, encoding="utf-8")
            obj = base / f"case{index}.o"
            build_result = run([cc, *C_FLAGS, *SAN_FLAGS, "-c", str(source), "-o", str(obj)])
            if build_result.returncode != 0:
                failures += 1
                print(f"[sanitize:selftest] FAIL {label}: does not compile: {build_result.stderr.strip()}")
                continue
            problems, _ = link_and_run(cxx, base, [obj], name=f"case{index}")
            got_red = bool(problems)
            if got_red == must_go_red:
                verdict = "red as it must" if must_go_red else "green as it must"
                print(f"[sanitize:selftest] PASS {label} -> {verdict}")
            else:
                failures += 1
                print(f"[sanitize:selftest] FAIL {label}: expected red={must_go_red}, got {problems}")
    if failures:
        print(f"[sanitize:selftest] FAIL ({failures} fixture(s))", file=sys.stderr)
        return 1
    print("[sanitize:selftest] OK")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--selftest", action="store_true", help="prove both sanitizers can catch a real bug")
    parser.add_argument("--keep", action="store_true", help="keep the build directory for inspection")
    args = parser.parse_args()

    cc = compiler("CC", "cc")
    cxx = compiler("CXX", "c++")

    if args.selftest:
        return selftest(cc, cxx)

    workdir = tempfile.mkdtemp(prefix="sanitize-")
    try:
        supported, detail = supports_sanitizers(cc, workdir)
        if not supported:
            print(f"[sanitize] SKIP {cc} cannot build with -fsanitize=address,undefined: {detail}")
            return 0
        started = time.monotonic()
        objects, problems = build(cc, cxx, workdir)
        if not problems:
            problems, transcript = link_and_run(cxx, workdir, objects)
            ok_line = [line for line in transcript.splitlines() if line.startswith("sanitize smoke:")]
            if not problems and not ok_line:
                problems.append("sanitize: the driver produced no verdict line; the gate cannot tell a "
                                "silent run from a passing one")
        elapsed = time.monotonic() - started
        if problems:
            for problem in problems:
                print(f"[sanitize] {problem}", file=sys.stderr)
            print(f"[sanitize] FAIL ({len(problems)} problem(s))", flush=True)
            return 1
        print(f"[sanitize] OK ({len(objects)} unit(s), {elapsed:.1f}s, {cc})", flush=True)
        return 0
    finally:
        if args.keep:
            print(f"[sanitize] build directory kept at {workdir}")
        else:
            shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
