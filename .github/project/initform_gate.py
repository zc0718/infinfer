#!/usr/bin/env python3
"""Aggregate-initialiser form for the project's public structs.

**Why this gate exists.** `infr_quant` gained a `requant_multiplier` field between
`zero_point` and `channel_scales`. One consumer in `workspace/` initialised that
struct positionally, so every field after the insertion point shifted by one:
`channel_zero_points` was built from `1u` and `channel_count` from `0u`. The ARM
legs compiled it with a warning and the tests still passed, because the per-tensor
path never reads those fields; only the RISC-V legs turned it into a build failure,
and only because that toolchain treats `int-conversion` as an error. A public
struct is allowed to grow; a consumer that names its fields cannot be broken by it.

  I1 every aggregate initialiser of a public struct names its fields
     (`.field = value`), or is empty (`{}`) or all zeros (`{0}`). Arrays of public
     structs are checked element by element, and a C compound literal
     (`(infr_quant){...}`) is an initialiser like any other.
  I2 the subject set is read from `include/infinfer/**/*.h` at run time. A
     hand-written type list here would be a second source of the same fact, and it
     would silently shrink to nothing when a struct is renamed.
  I3 the scanned roots are declared below, and a root that is missing is reported
     as skipped rather than dropped: a clean clone has no `workspace/`, and a gate
     that quietly scans less is a gate that quietly passes (INV-6).
  I4 third-party sources are not scanned. The list is `toolchains.json:skip_dirs`,
     because "which directories hold somebody else's code" is one fact and it is
     already declared there.

**What this gate cannot know.** It does not type-check: an initialiser whose first
element happens to be a designator is accepted on sight, and a struct declared
outside `include/infinfer/` is out of scope by I2 rather than by judgement. The
compiler still has the last word; this gate is here so the compiler is not the
*first* word on a code path (`__ARM_FEATURE_DSP` off, per-tensor only) where the
damage is invisible.

Usage:

    python3 .github/project/initform_gate.py               # check the tree
    python3 .github/project/initform_gate.py --selftest    # prove each rule fails
    python3 .github/project/initform_gate.py --include DIR --roots DIR [DIR...]
"""

import argparse
import json
import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INCLUDE_ROOT = ROOT / "include" / "infinfer"
SKIP_CONFIG = Path(__file__).resolve().with_name("toolchains.json")

# I3: the roots this gate owns. `workspace/` is absent in a clean clone on purpose.
ROOTS = ("src", "include", "test_package", "api", "benchmark", "workspace")
SUFFIXES = (".c", ".h", ".cpp", ".hpp")

# Declared exemptions, each with the reason it exists and what would remove it. An
# exemption without an exit is a hole nobody reopens.
EXEMPT = (
    {
        "path": "test_package/test",
        "why": "C++17 has no designated initialisers; requiring them here would either "
               "rely on a C++20 extension (clang warns, and the tidy gate is "
               "warnings-as-errors) or force assignment-style setup on every case. The risk "
               "is also different in kind: a test whose struct shifted computes a different "
               "value and fails, while a shipped or evidence-producing consumer keeps "
               "passing -- which is what happened to the KAT chain",
        "exit": "when the project moves to C++20, drop this entry and convert the sites",
    },
)

TYPEDEF = re.compile(r"typedef\s+struct\s+\w*\s*\{")

# Between a type name and the `)` that closes it: only qualifiers may appear. An
# identifier there is a declarator (`pub_t v`), which makes it a declaration.
DECLARATOR_FREE = re.compile(r"^[\s*&\[\]0-9]*$")

# Characters that can precede a `(` which starts an expression. A function
# declarator's `(` follows a name instead, which is what keeps `f(pub_t v) {`
# from being read as a compound literal whose body is the function body.
EXPRESSION_STARTERS = "=(,:?{;"


def mask(text):
    """Blank out comments and string/char literals *in place*.

    Length and newlines are preserved, so offsets and line numbers computed on the
    masked text are valid for the original. Doing it this way (rather than
    stripping) is what keeps `// {1, 2}` and `"{"` from being read as code.
    """
    out = list(text)
    i = 0
    n = len(text)
    state = None  # None | 'line' | 'block' | 'str' | 'chr'
    while i < n:
        char = text[i]
        nxt = text[i + 1] if i + 1 < n else ""
        if state is None:
            if char == "/" and nxt == "/":
                state = "line"
                out[i] = out[i + 1] = " "
                i += 2
            elif char == "/" and nxt == "*":
                state = "block"
                out[i] = out[i + 1] = " "
                i += 2
            elif char == '"':
                state = "str"
                out[i] = " "
                i += 1
            elif char == "'":
                state = "chr"
                out[i] = " "
                i += 1
            else:
                i += 1
            continue
        if state == "line":
            if char == "\n":
                state = None
            else:
                out[i] = " "
            i += 1
        elif state == "block":
            if char == "*" and nxt == "/":
                out[i] = out[i + 1] = " "
                i += 2
                state = None
            else:
                if char != "\n":
                    out[i] = " "
                i += 1
        elif char == "\\":
            out[i] = " "
            if i + 1 < n and text[i + 1] != "\n":
                out[i + 1] = " "
            i += 2
        elif (state == "str" and char == '"') or (state == "chr" and char == "'"):
            out[i] = " "
            state = None
            i += 1
        else:
            if char != "\n":
                out[i] = " "
            i += 1
    return "".join(out)


def public_structs(include_root):
    """I2: every struct typedef declared under the include root, name -> header."""
    names = {}
    root = Path(include_root)
    if not root.is_dir():
        return names
    for header in sorted(root.rglob("*.h")):
        text = mask(header.read_text(encoding="utf-8", errors="replace"))
        for match in TYPEDEF.finditer(text):
            tail = text[match.end():]
            close = re.search(r"\}\s*(\w+)\s*;", tail)
            if close:
                names[close.group(1)] = str(header)
    return names


def elements(masked, open_pos):
    """Split the brace group at `open_pos` into top-level element spans."""
    spans = []
    depth = 0
    start = open_pos + 1
    i = open_pos
    while i < len(masked):
        char = masked[i]
        if char in "{(":
            depth += 1
        elif char in "})":
            depth -= 1
            if depth == 0:
                if masked[start:i].strip():
                    spans.append((start, i))
                return spans
        elif char == "," and depth == 1:
            if masked[start:i].strip():
                spans.append((start, i))
            start = i + 1
        i += 1
    return spans


def is_all_zero(fragment):
    return all(token.strip() == "0" for token in fragment.split(",") if token.strip())


def first_token(fragment):
    stripped = fragment.lstrip()
    return stripped[:1]


def first_offence(masked, open_pos, depth=0):
    """I1: the first positionally initialised element of one initialiser, or None.

    One problem per initialiser, not per element: a row is fixed as a row, and a
    gate that prints fifty lines for thirteen rows is a gate people scroll past.
    """
    if depth > 8:  # pathological nesting: stop rather than hang
        return None
    for start, end in elements(masked, open_pos):
        fragment = masked[start:end]
        token = first_token(fragment)
        if token == "{":
            nested = first_offence(masked, masked.index("{", start), depth + 1)
            if nested is not None:
                return nested
            continue
        if token == "." or is_all_zero(fragment):
            continue
        return start, fragment
    return None


def compound_literal_brace(masked, after_type, close_paren):
    """Return the `{` position of a `(T){...}` compound literal, or None.

    Two conditions, both necessary: nothing but qualifiers between the type name and
    the `)` (an identifier there is a declarator), and a `(` that can start an
    expression. Without the second one, the body brace of `void f(pub_t v) {` would
    be read as an initialiser -- measured, in this gate's own first selftest run.
    """
    if not DECLARATOR_FREE.match(masked[after_type:close_paren]):
        return None
    depth = 0
    # Scan back to the matching `(`; the type name sits *inside* the parentheses,
    # so the search must be allowed to pass after_type (the first version bounded it
    # there, which made this whole branch unreachable -- caught by the fixture).
    for index in range(close_paren, -1, -1):
        if masked[index] == ")":
            depth += 1
        elif masked[index] == "(":
            depth -= 1
            if depth == 0:
                before = index - 1
                while before >= 0 and masked[before].isspace():
                    before -= 1
                if before >= 0 and masked[before] not in EXPRESSION_STARTERS:
                    return None
                rest = masked[close_paren + 1:]
                stripped = rest.lstrip()
                if not stripped.startswith("{"):
                    return None
                return close_paren + 1 + (len(rest) - len(stripped))
    return None


def is_declaration(masked, start):
    """True when the type name is `struct <name>` rather than a use of the typedef.

    Without this, every struct *definition* in `include/` reads as an initialiser whose
    body is the field list -- seven false positives on the first real run. The cost is a
    documented miss: an elaborated `struct infr_quant v = {...}` is not checked, and no
    such site exists here.
    """
    index = start - 1
    while index >= 0 and masked[index].isspace():
        index -= 1
    end = index + 1
    while index >= 0 and (masked[index].isalnum() or masked[index] == "_"):
        index -= 1
    return masked[index + 1:end] == "struct"


def scan_file(path, names, problems):
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    masked = mask(text)
    if not names:
        return
    pattern = re.compile(r"\b(" + "|".join(sorted(names)) + r")\b")
    for match in pattern.finditer(masked):
        if is_declaration(masked, match.start()):
            continue
        i = match.end()
        open_pos = None
        while i < len(masked):
            char = masked[i]
            if char == "{":
                open_pos = i
                break
            if char == ")":
                open_pos = compound_literal_brace(masked, match.end(), i)
                break
            if char in ";(":
                break  # declaration without an initialiser, or a call
            i += 1
        if open_pos is None:
            continue
        offence = first_offence(masked, open_pos)
        if offence is None:
            continue
        start, fragment = offence
        line = masked.count("\n", 0, start) + 1
        first = fragment.strip().split(",")[0].strip()
        problems.append(
            f"initform: {path}:{line}: a public struct is initialised positionally "
            f"(element starting with '{first}'); name the fields ('.field = value') so a "
            f"field inserted into the struct cannot shift the rest"
        )


def code_files(roots, skip_dirs):
    """I3/I4: yield code files under the declared roots, honouring declared skips."""
    scanned = 0
    exempted = 0
    for root in roots:
        base = ROOT / root
        if not base.exists():
            print(f"[initform] skip {root}/ (absent: a clean clone has no {root}/)")
            continue
        for path in sorted(base.rglob("*")):
            if path.suffix not in SUFFIXES or not path.is_file():
                continue
            rel = path.relative_to(ROOT).as_posix()
            if any(rel == skip or rel.startswith(skip + "/") for skip in skip_dirs):
                continue
            if any(rel == item["path"] or rel.startswith(item["path"] + "/") for item in EXEMPT):
                exempted += 1
                continue
            scanned += 1
            yield path
    print(f"[initform] scanned {scanned} code file(s) under {', '.join(roots)}")
    if exempted:
        print(f"[initform] {exempted} file(s) exempt by declaration: " + ", ".join(i["path"] for i in EXEMPT))


def skip_dirs():
    """I4: the third-party skip list, read from the one file that declares it."""
    try:
        data = json.loads(SKIP_CONFIG.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SystemExit(f"initform: cannot read {SKIP_CONFIG}: {exc}")
    entries = data.get("skip_dirs", [])
    for entry in entries:
        if not str(entry.get("why", "")).strip():
            raise SystemExit(f"initform: skip entry {entry.get('path')!r} has no 'why'")
    return [entry["path"] for entry in entries]


def run_check(include_root, roots):
    names = public_structs(include_root)
    if not names:
        return [f"initform: no struct typedef found under {include_root}; the subject "
                f"set cannot be empty without the rule becoming a no-op"]
    print(f"[initform] {len(names)} public struct(s) from {Path(include_root).name}/: "
          f"{', '.join(sorted(names))}")
    problems = []
    for path in code_files(roots, skip_dirs()):
        scan_file(path, names, problems)
    return problems


def selftest():
    """Prove each rule can fail, and that the three zero-risk forms stay green."""
    good_header = """
typedef struct pub { int a; float b; } pub_t;
typedef struct other { int a; } other_t;
"""
    # A red case names the line and the element it must report. Asserting only on a
    # phrase would let a *different* initialiser in the same file satisfy it -- the
    # first version of this selftest passed its compound-literal case that way, off
    # the function body it misinterpreted.
    cases = [
        ("I1 positional", "void f(void) { pub_t v = {1, 2.0f}; }", (1, "1")),
        ("I1 designated", "void f(void) { pub_t v = {.a = 1, .b = 2.0f}; }", None),
        ("I1 zero fill", "void f(void) { pub_t v = {0}; }", None),
        ("I1 empty braces", "void f(void) { pub_t v = {}; }", None),
        (
            "I1 array of structs",
            "static const pub_t rows[] = {\n    {1, 2.0f},\n    {.a = 3, .b = 4.0f},\n};",
            (2, "1"),
        ),
        (
            "I1 array, all designated",
            "static const pub_t rows[] = {\n    {.a = 1, .b = 2.0f},\n    {.a = 3, .b = 4.0f},\n};",
            None,
        ),
        ("I1 partial zero fill", "void f(void) { pub_t v = {0, 1}; }", (1, "1")),
        ("I1 compound literal", "void f(pub_t *p) { *p = (pub_t){1, 2.0f}; }", (1, "1")),
        (
            "I2 a local struct is out of scope",
            "typedef struct loc { int a; } loc_t;\nvoid f(void) { loc_t v = {1}; }",
            None,
        ),
        (
            "I3 a signature brace is not an initialiser",
            "void f(pub_t v) {\n    (void)v;\n}",
            None,
        ),
        (
            "I3 a signature brace with an unnamed parameter is not an initialiser",
            "void f(pub_t) {\n}",
            None,
        ),
        (
            "I4 braces inside comments and strings are not code",
            '/* pub_t v = {1, 2.0f}; */\nvoid f(void) { const char *s = "{1, 2}"; (void)s; }',
            None,
        ),
        (
            "a `return (T){...}` stays undetected (the documented miss)",
            "pub_t f(void) { return (pub_t){1, 2.0f}; }",
            None,
        ),
    ]
    failures = 0
    with tempfile.TemporaryDirectory(prefix="initform-selftest-") as workdir:
        base = Path(workdir)
        (base / "include" / "probe").mkdir(parents=True)
        (base / "include" / "probe" / "probe.h").write_text(good_header, encoding="utf-8")
        names = public_structs(base / "include")
        if sorted(names) != ["other_t", "pub_t"]:
            failures += 1
            print(f"[initform:selftest] FAIL I2 subject set came back as {sorted(names)}")
        else:
            print("[initform:selftest] PASS I2 the subject set is read from the include root")

        for index, (label, body, expected) in enumerate(cases):
            source = base / f"src/case{index}.c"
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_text(body + "\n", encoding="utf-8")
            problems = []
            scan_file(source, names, problems)
            if expected is None:
                if not problems:
                    print(f"[initform:selftest] PASS {label} -> green")
                else:
                    failures += 1
                    print(f"[initform:selftest] FAIL {label}: expected green, got {problems}")
                continue
            line, token = expected
            hit = [p for p in problems if f":{line}:" in p and f"starting with '{token}'" in p]
            if hit:
                print(f"[initform:selftest] PASS {label} -> line {line}, element '{token}'")
            else:
                failures += 1
                print(f"[initform:selftest] FAIL {label}: expected line {line}/element {token!r}, got {problems}")

        empty = run_check(base / "does-not-exist", ())
        if any("subject set cannot be empty" in p for p in empty):
            print("[initform:selftest] PASS I2 an empty subject set is red, not a silent no-op")
        else:
            failures += 1
            print(f"[initform:selftest] FAIL an empty subject set should be red, got {empty}")

    if failures:
        print(f"[initform:selftest] FAIL ({failures} fixture(s))", file=sys.stderr)
        return 1
    print("[initform:selftest] OK")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--include", default=str(INCLUDE_ROOT), help="root holding the public headers")
    parser.add_argument("--roots", nargs="*", default=list(ROOTS), help="directories to scan")
    parser.add_argument("--selftest", action="store_true", help="prove each rule can fail")
    args = parser.parse_args()

    if args.selftest:
        return selftest()

    problems = run_check(args.include, tuple(args.roots))
    if problems:
        for problem in problems:
            print(f"[initform] {problem}", file=sys.stderr)
        print(f"[initform] FAIL ({len(problems)} problem(s))", flush=True)
        return 1
    print("[initform] OK", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
