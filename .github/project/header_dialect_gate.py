#!/usr/bin/env python3
"""The installed headers must stay includable from a C99 consumer; the code is C11.

**What changed and why.** This gate used to require that `src/kernel/scalar/**/*.c`
compile as C99. That requirement was retracted (gaps G-56): nobody compiles our
reference bodies -- a consumer includes the header and links the library -- and the
construct it claimed to protect was invisible to it anyway. `-std=c99
-pedantic-errors` cannot see `_Static_assert` in this project's translation units,
because every one of them reaches glibc's <sys/cdefs.h> through <stdint.h>, and that
header defines a fallback macro for the keyword. Measured, not assumed: a file that
includes kernel.h and writes `_Static_assert(...)` compiles clean under
`-std=c99 -pedantic-errors`. So the old gate banned three constructs nobody used
(`_Generic`, `_Alignas`, `_Atomic`) and was blind to the one that is used.

**What the property is worth.** Everything in this project is built as C11
(`metadata.build_cstd`), and that is the single code standard. But `include/` is the
ABI: a consumer compiles *that*, with whatever compiler it has. Keeping the installed
headers parseable as C99 costs one `#if __STDC_VERSION__` around the one C11
construct they use, and it is the only compatibility claim here with a real consumer.

  H1 every installed header parses under `-std=c99 -pedantic-errors`, so a C99
     consumer can include it. That is what the `__STDC_VERSION__` guard in
     kernel.h exists for, and this gate is what keeps the guard honest.
  H2 the same headers also parse under `-std=c11`. Without this the gate could
     drift into rejecting the project's own dialect, which would be the opposite
     of portability.
  H3 there is no C99 build of the library. The `.c` files are C11 only, with no
     dialect guards; a guard in a body would be policy with no consumer.

**Known blind spot, kept executable.** `_Static_assert` in a header passes H1 for the
glibc reason above. The selftest asserts that it *does* pass under C99 -- not because
that is desirable, but so the day a toolchain stops masking it, the selftest goes red
and we learn the gate became stronger instead of quietly assuming it did not.

Usage:

    python3 .github/project/header_dialect_gate.py
    python3 .github/project/header_dialect_gate.py --selftest
"""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INCLUDE_ROOT = ROOT / "include"
FLAGS = ["-pedantic-errors", "-Wall", "-Wextra"]

CONSUMER_DIALECT = "c99"
PROJECT_DIALECT = "c11"


def compiler():
    return os.environ.get("CC", "cc")


def compile_header(cc, include_root, header, dialect, workdir):
    """Include one header from a scratch TU; returns the compiler's stderr."""
    include_root = Path(include_root)
    rel = Path(header).relative_to(include_root).as_posix()
    source = Path(workdir) / f"{rel.replace('/', '_')}.{dialect}.c"
    source.write_text(f"#include <{rel}>\nint main(void) {{ return 0; }}\n", encoding="utf-8")
    result = subprocess.run(
        [cc, f"-std={dialect}", *FLAGS, "-I", str(include_root), "-c", str(source), "-o", str(source.with_suffix(".o"))],
        capture_output=True,
        text=True,
    )
    return result.stderr.strip()


def headers(include_root):
    root = Path(include_root)
    if not root.is_dir():
        return []
    return sorted(root.rglob("*.h"))


def run_check(include_root):
    """Returns (problems, header_count). Empty problems means green."""
    found = headers(include_root)
    if not found:
        return [f"header-dialect: no headers under {include_root}; the gate would be a silent no-op"], 0
    problems = []
    cc = compiler()
    with tempfile.TemporaryDirectory(prefix="hdr-") as workdir:
        for header in found:
            rel = Path(header).relative_to(Path(include_root)).as_posix()
            stderr = compile_header(cc, include_root, header, CONSUMER_DIALECT, workdir)
            if stderr:
                problems.append(f"header-dialect: {rel} is not includable from a C99 consumer:\n{stderr}")
            stderr = compile_header(cc, include_root, header, PROJECT_DIALECT, workdir)
            if stderr:
                problems.append(f"header-dialect: {rel} does not compile as {PROJECT_DIALECT} either:\n{stderr}")
    return problems, len(found)


def selftest():
    """Prove both directions, and pin the blind spot rather than pretending it is not there."""
    clean = """#pragma once
#include <stdint.h>
typedef struct probe { uint32_t a; float b; } probe_t;
#define PROBE_LIMIT 7U
"""
    c11_only = """#pragma once
#include <stdint.h>
extern _Alignas(16) uint8_t probe_buffer[16];
"""
    static_assert_in_expansion = """#pragma once
#include <stdint.h>
_Static_assert(sizeof(uint32_t) == 4U, "masked by glibc under C99");
"""
    cases = [
        ("a clean header", clean, None),
        ("a C11-only header", c11_only, "not includable from a C99 consumer"),
        # The blind spot: accepted under C99 because glibc masks the keyword. If a
        # toolchain ever stops masking it, this case fails and says the gate got
        # stronger -- which is information, not noise.
        ("_Static_assert is masked (known blind spot)", static_assert_in_expansion, None),
    ]
    failures = 0
    with tempfile.TemporaryDirectory(prefix="hdr-selftest-") as workdir:
        base = Path(workdir)
        for index, (label, text, expected) in enumerate(cases):
            root = base / f"case{index}" / "include"
            root.mkdir(parents=True)
            (root / "probe.h").write_text(text, encoding="utf-8")
            problems, count = run_check(root)
            if count != 1:
                failures += 1
                print(f"[header-dialect:selftest] FAIL {label}: expected 1 header, got {count}")
                continue
            if expected is None and not problems:
                print(f"[header-dialect:selftest] PASS {label} -> accepted in both dialects")
            elif expected is not None and any(expected in problem for problem in problems):
                print(f"[header-dialect:selftest] PASS {label} -> rejected as it must")
            else:
                failures += 1
                print(f"[header-dialect:selftest] FAIL {label}: expected {expected!r}, got {problems}")
        # H3: a body is not held to C99 any more, so the gate must not look at one.
        body = base / "case0" / "src" / "body.c"
        body.parent.mkdir(parents=True)
        body.write_text("_Static_assert(1, \"C11 in a body is fine now\");\nint main(void) { return 0; }\n",
                        encoding="utf-8")
        problems, count = run_check(base / "case0" / "include")
        if not problems and count == 1:
            print("[header-dialect:selftest] PASS H3 a C11-only body next to the headers is ignored")
        else:
            failures += 1
            print(f"[header-dialect:selftest] FAIL H3 the gate looked at a body: {problems}")
    if failures:
        print(f"[header-dialect:selftest] FAIL ({failures} fixture(s))", file=sys.stderr)
        return 1
    print("[header-dialect:selftest] OK")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--include", default=str(INCLUDE_ROOT), help="installed-header root")
    parser.add_argument("--selftest", action="store_true", help="prove both dialects are judged")
    args = parser.parse_args()

    if args.selftest:
        return selftest()

    problems, count = run_check(args.include)
    if problems:
        for problem in problems:
            print(f"[header-dialect] {problem}", file=sys.stderr)
        print(f"[header-dialect] FAIL ({len(problems)} problem(s))", flush=True)
        return 1
    print(f"[header-dialect] OK ({count} header(s) includable from C99 and C11, {compiler()})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
