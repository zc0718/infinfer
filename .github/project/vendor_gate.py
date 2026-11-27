#!/usr/bin/env python3
"""V1/V2: a vendor backend is present when switched on and absent when switched off.

Why this gate exists
--------------------
A vendor backend is optional by construction: the library has to build without it, and the
*only* way to know the option really removed it -- rather than merely disabling a call site --
is to look at the object file's symbol table. A build that "supports" an option but still
carries the vendor's code is the failure this checks for, and it is invisible to every other
gate in the tree because the build is green either way.

What it checks
--------------
V1  with the option off, the object defines none of the backend's symbols
V2  with the option on, the object defines them

V1 is the rule that matters; V2 exists so that V1 cannot pass by the source being empty.

The stub header
---------------
V2 needs `arm_nnfunctions.h`, which only exists once the vendor's recipe does. The gate writes
its own stub into a temporary directory and compiles against that. The stub never enters the
repository, and the gate says so in its own output: what is proven here is the *guard*, not the
vendor library. Claiming otherwise would be the fabrication this project's rules forbid.

`--selftest` gives each rule its own fixture, and asserts on that rule's own wording, because an
assertion satisfied by a neighbouring fixture proves nothing (gaps G-115).
"""
import argparse
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[2]
SOURCE = ROOT / "src" / "backend" / "cmsis_nn.c"

#: Symbols the switched-on backend is expected to define. Kept as a prefix so adding a body
#: without adding it here is caught rather than silently unchecked.
EXPECTED_ON = "infr_cmsis_nn_backend_present"
PREFIX = "infr_cmsis_nn"

STUB_HEADER = """/* Written by vendor_gate.py -- not part of the repository. It exists so the
 * switched-on branch can be compiled without the vendor package, which proves the guard in
 * src/backend/cmsis_nn.c and nothing about the vendor library itself. */
#ifndef ARM_NNFUNCTIONS_H
#define ARM_NNFUNCTIONS_H
#endif
"""


def _run(cmd, **kwargs):
    return subprocess.run(cmd, capture_output=True, text=True, check=False, **kwargs)


def compile_object(source, workdir, defines=(), include_dirs=(), compiler="cc"):
    """Compile one translation unit to an object and return its path, or an error string."""
    obj = pathlib.Path(workdir) / (pathlib.Path(source).stem + ".o")
    cmd = [compiler, "-std=c11", "-Wall", "-Wextra", "-Werror", "-c", str(source), "-o", str(obj)]
    for d in defines:
        cmd.append(f"-D{d}")
    for inc in include_dirs:
        cmd.append(f"-I{inc}")
    proc = _run(cmd)
    if proc.returncode != 0:
        return None, f"could not compile {pathlib.Path(source).name}: {proc.stderr.strip()[:400]}"
    return obj, None


def symbols(obj):
    """Defined symbols of an object, via nm."""
    nm = shutil.which("nm")
    if nm is None:
        return None, "nm is not installed, so the symbol table cannot be read"
    proc = _run([nm, "--defined-only", str(obj)])
    if proc.returncode != 0:
        return None, f"nm failed: {proc.stderr.strip()[:200]}"
    found = []
    for line in proc.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 3:
            found.append(parts[2])
    return found, None


def check_off(source, workdir):
    """V1: with no define, the object must carry none of the backend's symbols."""
    obj, error = compile_object(source, workdir)
    if error:
        return [error]
    found, error = symbols(obj)
    if error:
        return [error]
    leftover = [s for s in found if s.startswith(PREFIX)]
    if leftover:
        return [
            f"V1 {pathlib.Path(source).relative_to(ROOT) if pathlib.Path(source).is_relative_to(ROOT) else pathlib.Path(source).name}: "
            f"the option is off but the object still defines {', '.join(leftover)} -- "
            "switching a backend off has to remove it from the binary, not just stop calling it"
        ]
    return []


def check_on(source, workdir):
    """V2: with the define and a stub header, the backend's symbol must be defined."""
    stub = pathlib.Path(workdir) / "stub"
    stub.mkdir(parents=True, exist_ok=True)
    (stub / "arm_nnfunctions.h").write_text(STUB_HEADER)
    obj, error = compile_object(source, workdir, defines=("INFR_WITH_CMSIS_NN",), include_dirs=(stub,))
    if error:
        return [error]
    found, error = symbols(obj)
    if error:
        return [error]
    if EXPECTED_ON not in found:
        return [
            f"V2 {pathlib.Path(source).name}: the option is on but the object does not define "
            f"{EXPECTED_ON} -- V1 would then pass for the wrong reason, because there would be "
            "nothing to find either way"
        ]
    return []


def check_tree():
    if not SOURCE.is_file():
        return [
            f"vendor: {SOURCE.relative_to(ROOT)} is missing, so the option has nothing to switch "
            "off and this gate would pass while checking nothing"
        ]
    problems = []
    with tempfile.TemporaryDirectory() as workdir:
        off_dir = pathlib.Path(workdir) / "off"
        on_dir = pathlib.Path(workdir) / "on"
        off_dir.mkdir()
        on_dir.mkdir()
        problems.extend(check_off(SOURCE, off_dir))
        problems.extend(check_on(SOURCE, on_dir))
    return problems


def selftest():
    """Each rule gets a fixture that makes only that rule fail."""
    failures = 0
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = pathlib.Path(tmp)

        # V1's own failure: a source that defines the symbol unconditionally.
        unguarded = tmpdir / "unguarded.c"
        unguarded.write_text(f'int {EXPECTED_ON}(void) {{ return 1; }}\n')
        work = tmpdir / "a"
        work.mkdir()
        problems = check_off(unguarded, work)
        if problems and problems[0].startswith("V1"):
            failures += 1
            print(f"[vendor:selftest] PASS V1 an unguarded definition is red ({problems[0][:60]}...)")
        else:
            print(f"[vendor:selftest] FAIL V1 an unguarded definition should be red, got {problems}")

        # V2's own failure: a source that never defines it.
        silent = tmpdir / "silent.c"
        silent.write_text("typedef int not_the_backend_t;\n")
        work = tmpdir / "b"
        work.mkdir()
        problems = check_on(silent, work)
        if problems and problems[0].startswith("V2"):
            failures += 1
            print(f"[vendor:selftest] PASS V2 a branch that defines nothing is red ({problems[0][:60]}...)")
        else:
            print(f"[vendor:selftest] FAIL V2 a branch that defines nothing should be red, got {problems}")

        # And the tree itself has to be well formed, or the two rules above prove nothing
        # about the real source.
        if not SOURCE.is_file():
            print(f"[vendor:selftest] FAIL the guarded source {SOURCE.relative_to(ROOT)} is missing")
        else:
            with tempfile.TemporaryDirectory() as workdir:
                off_dir = pathlib.Path(workdir) / "off"
                on_dir = pathlib.Path(workdir) / "on"
                off_dir.mkdir()
                on_dir.mkdir()
                problems = check_off(SOURCE, off_dir) + check_on(SOURCE, on_dir)
            if problems:
                print(f"[vendor:selftest] FAIL the real source does not satisfy V1/V2: {problems}")
            else:
                failures += 1
                print("[vendor:selftest] PASS the real source satisfies V1 and V2")

    if failures == 3:
        print("[vendor:selftest] OK")
        return 0
    print("[vendor:selftest] FAIL")
    return 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        return selftest()

    problems = check_tree()
    if problems:
        for problem in problems:
            print(f"[vendor] {problem}")
        print(f"[vendor] FAIL ({len(problems)} problem(s))")
        return 1
    print(f"[vendor] OK (2 check(s): the switch removes the backend's symbols, and the on branch defines them)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
