#!/usr/bin/env python3
"""Every declaration in the installed headers must be callable from C *and* from C++ under one name.

Gaps G-40 recorded the failure this rule exists for: `tensor.h` and `graph.h` were missing
`extern "C"`, the whole C++ test suite stayed green (every consumer was C++, and the recipe's
`_make_c_compatible()` re-wrapped the *exported* headers), and the first C consumer -- the
bare-metal KAT image -- failed at link time. Nothing in the tree notices the difference, because
a C compiler does not mangle and a C++ one does.

So the check is done by looking at what a **C++** consumer references. Each probe takes the
address of every declared function; the object then carries one undefined symbol per function,
and those symbols must be unmangled. If a header forgot `extern "C"`, the C++ probe emits
`_Z...` and the rule goes red.

Rules:

  A1 the header scan finds a plausible number of declarations (an empty set is red, not green)
  A2 the C probe compiles and every referenced symbol is unmangled
  A3 the C++ probe compiles and every referenced symbol is unmangled -- the G-40 catch
  A4 both probes reference the same names, so neither language can drift from the other

`--selftest` builds a header without `extern "C"` and proves A3 fails on it, and one with it and
proves it passes.
"""

import argparse
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[2]
INCLUDE = ROOT / "include"
DOWNSTREAM_SCAN = ("include",)
BANNED = ("include/cpptest.hpp", "include/ctest.h", "include/doctest.hpp", "include/etl.hpp")

DECL_RE = re.compile(r"\b(infr_[a-z0-9_]+)\s*\(")
BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.S)
LINE_COMMENT = re.compile(r"//[^\n]*")
SYMBOL_RE = re.compile(r"^\s+U (\S+)$", re.M)


def headers():
    out = []
    for path in sorted((ROOT / "include").rglob("*.h")):
        relative = path.relative_to(ROOT).as_posix()
        if relative in BANNED:
            continue  # template-owned test/utility headers, not the product ABI
        out.append(path)
    return out


def declared(paths):
    """Function names declared in the installed headers.

    Two kinds of false positive have to be removed, and both were caught by the gate failing
    on its first run rather than by reading the regex: `typedef infr_status (*infr_kernel_fn)(...)`
    matches `infr_status (` as if it were a call, and the typedef's own name matches too. So a
    candidate is dropped when its line says `typedef`, and when it appears inside a
    function-pointer declarator.
    """
    names = set()
    fnptr = set()
    for path in paths:
        text = path.read_text(encoding="utf-8")
        text = BLOCK_COMMENT.sub(" ", text)
        text = LINE_COMMENT.sub(" ", text)
        fnptr.update(re.findall(r"\(\s*\*\s*(infr_[a-z0-9_]+)\s*\)", text))
        for match in DECL_RE.finditer(text):
            line_start = text.rfind("\n", 0, match.start()) + 1
            line = text[line_start:match.start()]
            if "typedef" in line:
                continue
            names.add(match.group(1))
    return sorted(names - fnptr)


def probe(names, language):
    lines = ["#include <infinfer/infinfer.h>", ""]
    if language == "cxx":
        lines.append("extern \"C\" {")
    lines.append("void *const infr_probe_table[] = {")
    for name in names:
        lines.append(f"    (void *)&{name},")
    lines.append("};")
    if language == "cxx":
        lines.append("}")
    lines.append("")
    return "\n".join(lines)


def compile_probe(source, language, workdir):
    suffix = ".cpp" if language == "cxx" else ".c"
    src = pathlib.Path(workdir) / ("probe" + suffix)
    src.write_text(source, encoding="utf-8")
    obj = pathlib.Path(workdir) / ("probe" + suffix + ".o")
    compiler = "c++" if language == "cxx" else "cc"
    std = "-std=c++17" if language == "cxx" else "-std=c11"
    done = subprocess.run(
        [compiler, std, "-I", str(INCLUDE), "-c", str(src), "-o", str(obj)],
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
    )
    if done.returncode != 0:
        return None, (done.stderr.strip().splitlines() or ["compile failed"])[-1]
    nm = subprocess.run(
        ["nm", "-u", str(obj)], capture_output=True, text=True, stdin=subprocess.DEVNULL
    )
    if nm.returncode != 0:
        return None, "nm failed on the probe object"
    return sorted(set(SYMBOL_RE.findall(nm.stdout))), None


def check(names, workdir):
    problems = []
    if len(names) < 10:
        problems.append(
            f"api: only {len(names)} declaration(s) found under include/ -- the scan is broken, "
            "and a rule that finds nothing must not be green"
        )
        return problems

    seen = {}
    for language in ("c", "cxx"):
        symbols, error = compile_probe(probe(names, language), language, workdir)
        if error:
            problems.append(f"api: the {language} consumer does not compile: {error}")
            continue
        seen[language] = set(symbols)
        mangled = [s for s in symbols if not s.startswith("infr_")]
        if mangled:
            problems.append(
                f"api: the {language} consumer references mangled symbol(s) {mangled[:4]} -- a "
                "declaration is missing extern \"C\" (see gaps G-40)"
            )
        missing = [n for n in names if f"__imp_{n}" not in symbols and n not in symbols]
        if missing:
            problems.append(
                f"api: the {language} consumer does not reference {missing[:4]} -- they were "
                "declared in a header but the linker would not find them"
            )
    if "c" in seen and "cxx" in seen and seen["c"] != seen["cxx"]:
        only_c = sorted(seen["c"] - seen["cxx"])[:4]
        only_cxx = sorted(seen["cxx"] - seen["c"])[:4]
        problems.append(
            f"api: the two consumers disagree on names (only C: {only_c}, only C++: {only_cxx})"
        )
    return problems


def selftest():
    failures = 0
    workdir = tempfile.mkdtemp(prefix="api-selftest-")
    try:
        with_marker = pathlib.Path(workdir) / "with.h"
        without_marker = pathlib.Path(workdir) / "without.h"
        with_marker.write_text(
            '#ifdef __cplusplus\nextern "C" {\n#endif\n'
            "void infr_alpha(void);\n"
            '#ifdef __cplusplus\n}\n#endif\n',
            encoding="utf-8",
        )
        without_marker.write_text("void infr_alpha(void);\n", encoding="utf-8")

        for label, header, want_red in (
            ("A3 a header with extern \"C\" is green", with_marker, False),
            ("A3 a header without extern \"C\" is red", without_marker, True),
        ):
            src = pathlib.Path(workdir) / "probe.cpp"
            src.write_text(
                f'#include "{header}"\n'
                "void *const infr_probe_table[] = {(void *)&infr_alpha};\n",
                encoding="utf-8",
            )
            obj = pathlib.Path(workdir) / "probe.o"
            done = subprocess.run(
                ["c++", "-std=c++17", "-c", str(src), "-o", str(obj)],
                capture_output=True,
                text=True,
                stdin=subprocess.DEVNULL,
            )
            if done.returncode != 0:
                failures += 1
                print(f"[api:selftest] FAIL {label}: probe did not compile")
                continue
            nm = subprocess.run(
                ["nm", "-u", str(obj)], capture_output=True, text=True, stdin=subprocess.DEVNULL
            )
            symbols = SYMBOL_RE.findall(nm.stdout)
            red = any(not s.startswith("infr_") for s in symbols)
            if red == want_red:
                print(f"[api:selftest] PASS {label} (symbols: {symbols})")
            else:
                failures += 1
                print(f"[api:selftest] FAIL {label}: symbols were {symbols}")

        problems = check(["infr_only_a", "infr_only_b"], workdir)
        if any("scan is broken" in p for p in problems):
            print("[api:selftest] PASS A1 an empty scan is red")
        else:
            failures += 1
            print(f"[api:selftest] FAIL A1 expected an empty-scan report, got {problems}")
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    if failures:
        print(f"[api:selftest] FAIL ({failures} fixture(s))", file=sys.stderr)
        return 1
    print("[api:selftest] OK")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--selftest", action="store_true", help="prove the rules can fail")
    args = parser.parse_args()
    if args.selftest:
        return selftest()

    names = declared(headers())
    workdir = tempfile.mkdtemp(prefix="api-gate-")
    try:
        problems = check(names, workdir)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    if problems:
        for problem in problems:
            print(f"[api] {problem}", file=sys.stderr)
        print(f"[api] FAIL ({len(problems)} problem(s))", file=sys.stderr)
        return 1
    print(f"[api] OK ({len(names)} declaration(s) reachable from C and C++ under one name)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
