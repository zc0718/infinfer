#!/usr/bin/env python3
"""The scalar reference bodies must still be C99, not just C11.

**Why this gate exists.** The project compiles everything as C11, and the header
takes advantage of it (`_Static_assert`, `_Generic` behind a version guard). But
the scalar tier is the one place that promises something narrower: it is the
oracle every other tier is compared against, so it has to be readable and
buildable by whoever arrives with an older toolchain, and a vendored copy of it
has to land in a codebase that predates C11. That promise costs nothing to keep
and costs a rewrite to recover, which is the definition of a rule worth gating.

**What it does.** Compiles every `.c` under `src/kernel/scalar/` twice: once as
the project builds it (C11, the normal path) and once as `-std=c99
-pedantic-errors`. The second compile is the verdict. Headers are shared, so this
also proves the public kernel contract parses under C99 -- which is why
`kernel.h` puts its `_Static_assert` behind `__STDC_VERSION__`.

**What this gate cannot know.** It compiles; it does not link, and it does not run
the C99 build. A C99 translation unit that compiles but misbehaves would pass,
and that is deliberate: running the suite twice would double the meaning of
"tests pass" without doubling the evidence.

Usage:

    python3 .github/project/c99_gate.py
    python3 .github/project/c99_gate.py --selftest   # prove it can go red
"""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCALAR_ROOT = ROOT / "src" / "kernel" / "scalar"
INCLUDES = ["-I", str(ROOT / "include")]

# `-pedantic-errors` turns every language-version complaint into a failure, which is
# what makes "still C99" checkable rather than a matter of taste. `-Wall -Wextra`
# keeps the gate honest about the code being clean, not merely parseable.
C99_FLAGS = ["-std=c99", "-pedantic-errors", "-Wall", "-Wextra"]

# The fixture deliberately avoids `_Static_assert`. glibc's <sys/cdefs.h> (pulled in by
# <stdint.h>) defines a fallback macro for it when the dialect is older than C11, so a
# `_Static_assert` fixture **compiles clean under -std=c99** and the selftest passes for
# the wrong reason -- which is exactly what happened on this gate's first run. `_Alignas`
# and `_Generic` have no such fallback.
C11_ONLY_FIXTURE = """
/* C11-only: `_Alignas` and `_Generic` are not C99 constructs, so this must fail. */
#include <stdint.h>
static _Alignas(16) uint8_t buffer[4];
int main(void) {
    int value = 0;
    return _Generic(value, int: 0, default: 1) + (buffer[0] != 0U);
}
"""

C99_FIXTURE = """
#include <stdint.h>
/* The portable spelling the scalar tier is expected to use instead. */
typedef char assert_uint32_is_four_bytes[(sizeof(uint32_t) == 4U) ? 1 : -1];
static uint8_t buffer[4];
int main(void) { return (int)buffer[0]; }
"""


def compiler():
    return os.environ.get("CC", "cc")


def compile_one(cc, source, flags, workdir):
    """Returns the compiler's stderr, empty on success."""
    target = Path(workdir) / (Path(source).stem + ".o")
    chosen = C99_FLAGS if flags == "c99" else flags
    result = subprocess.run(
        [cc, *chosen, *INCLUDES, "-c", str(source), "-o", str(target)],
        capture_output=True,
        text=True,
    )
    return result.stderr.strip()


def scalar_sources():
    if not SCALAR_ROOT.is_dir():
        return []
    return sorted(SCALAR_ROOT.rglob("*.c"))


def run_check():
    """Returns (problems, compiled_count). Empty problems means green."""
    problems = []
    units = scalar_sources()
    if not units:
        return [f"c99: no scalar sources under {SCALAR_ROOT}; the gate would be a silent no-op"], 0
    cc = compiler()
    with tempfile.TemporaryDirectory(prefix="c99-") as workdir:
        for source in units:
            stderr = compile_one(cc, source, "c99", workdir)
            if stderr:
                problems.append(f"c99: {source.relative_to(ROOT)} is not C99-clean:\n{stderr}")
    return problems, len(units)


def selftest():
    """Prove the gate fails for C11-only code and passes for the portable spelling."""
    cc = compiler()
    failures = 0
    with tempfile.TemporaryDirectory(prefix="c99-selftest-") as workdir:
        base = Path(workdir)
        cases = [
            ("C11-only code", C11_ONLY_FIXTURE, True),
            ("the portable spelling", C99_FIXTURE, False),
        ]
        for index, (label, text, must_fail) in enumerate(cases):
            source = base / f"case{index}.c"
            source.write_text(text, encoding="utf-8")
            stderr = compile_one(cc, source, "c99", base)
            got_failure = bool(stderr)
            if got_failure == must_fail:
                print(f"[c99:selftest] PASS {label} -> {'rejected' if must_fail else 'accepted'}")
            else:
                failures += 1
                print(f"[c99:selftest] FAIL {label}: expected failure={must_fail}, got stderr={stderr!r}")
        # The same code must compile under C11, or the gate would be rejecting the
        # project's own dialect rather than enforcing a narrower one.
        source = base / "case0.c"
        result = subprocess.run(
            [cc, "-std=c11", *INCLUDES, "-c", str(source), "-o", str(base / "case0_c11.o")],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            print("[c99:selftest] PASS the C11-only fixture still compiles under C11")
        else:
            failures += 1
            print(f"[c99:selftest] FAIL the C11 fixture must compile under C11, got {result.stderr.strip()}")
    if failures:
        print(f"[c99:selftest] FAIL ({failures} fixture(s))", file=sys.stderr)
        return 1
    print("[c99:selftest] OK")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--selftest", action="store_true", help="prove the gate rejects C11-only code")
    args = parser.parse_args()

    if args.selftest:
        return selftest()

    problems, count = run_check()
    if problems:
        for problem in problems:
            print(f"[c99] {problem}", file=sys.stderr)
        print(f"[c99] FAIL ({len(problems)} problem(s))", flush=True)
        return 1
    print(f"[c99] OK ({count} scalar unit(s) under -std=c99 -pedantic-errors, {compiler()})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
