#!/usr/bin/env python3
"""Enforce the docstring contract on the shipped surface (`include/` + `src/`).

Why a gate: a doc block that misses `[en]`/`[zh]` still *compiles*, so nothing
notices until `docs/build.py` produces a language build with the wrong text in it
-- which is a rework, not a bug report. The rules below are the ones the
conventions actually state (`.github/skills/_shared/code-conventions.md`):

  D1 every block Doxygen comment (`/** ... */`) carries `@brief [en]` and `@brief [zh]`
  D2 every *top-level* block Doxygen comment carries `@since <version>`; a member
     block nested in a struct/enum body inherits its type's version, so repeating
     the tag on 40 fields would be noise, not information
  D3 `@exporter` / `@attacher` appear only inside a Doxygen block (else module
     generation fails silently)
  D4 no language marker inside an *inline* member comment (`/**< ... */`): the
     docs builder drops whole lines per language, and the closing `*/` would ride
     along with the dropped language, leaving an unterminated comment (measured:
     Doxygen then reports "Reached end of file while still inside a comment").
     Document members with a preceding `/** @brief [en] ... / @brief [zh] ... */`
     block instead.
  D5 a file-level block (the one carrying `@file`) opens with `/*!` and carries
     `@ingroup`: `docs/build.py` version-filters every block that starts with
     `/**` by reading its `@since`, so a file block written that way disappears
     as soon as a newer version is built. `/*!` matches neither the version
     filter nor the language splitter, which is exactly what a file block needs.

  D5 (below) is the ratchet: a file may be exempted only with a written reason and
  what removes it, so the exemption cannot quietly become permanent.

`--selftest` gives each rule its own fixture and asserts on that rule's wording,
because an assertion satisfied by a neighbouring fixture proves nothing.

强制 `include/` + `src/` 的 docstring 契约。缺 `[en]`/`[zh]` 的文档块照样能编译，
不会被发现，直到文档构建产出语言不对的版本——那是返工，不是报错。
四条规则见上：块注释要双语 `@brief` + `@since`；行内成员注释（`/**< */`）没有
`@brief` 可挂，就要求两种语言的`[en]`/`[zh]`标记；模块标签必须在 Doxygen 块内。
D5 是棘轮：豁免必须写理由与“什么会让它消失”。
"""
import argparse
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
SCAN_DIRS = ("include", "src")
SUFFIXES = {".h", ".hpp", ".c", ".cpp"}
BLOCK = re.compile(r"/\*\*(?![<])(.*?)\*/", re.S)
FILE_BLOCK = re.compile(r"/\*!(.*?)\*/", re.S)
NO_FILE = re.compile(r"/\*")
INLINE = re.compile(r"/\*\*<(.*?)\*/", re.S)
BRIEF_EN = re.compile(r"@brief\s*\[en\]")
BRIEF_ZH = re.compile(r"@brief\s*\[zh\]")
SINCE = re.compile(r"@since\s+\S+")

# The template's own demo translation units, which P0.4 replaces with real code.
# Exempted here rather than "fixed" for the same reason the scaffold contract
# tolerates forks: rewriting a file that is scheduled for deletion is churn.
EXEMPT = {
    "src/ctest.c": "template demo TU (zlib/pcre2 probes); P0.4 replaces it",
    "src/cpptest.cpp": "template demo TU (cpptest probes); P0.4 replaces it",
    "src/doctest.cpp": "template demo TU (doctest probes); P0.4 replaces it",
    "include/cpptest.hpp": "template demo header; P0.4 replaces it",
    "include/doctest.hpp": "template demo header; P0.4 replaces it",
    "include/etl.hpp": "template demo header (etl probe); P0.4 replaces it",
}


def scan(paths, pattern):
    """Yield (path, match_text, start_line) for one comment flavour."""
    for path in paths:
        text = path.read_text(encoding="utf-8", errors="replace")
        for match in pattern.finditer(text):
            yield path, match.group(1), text.count("\n", 0, match.start()) + 1


def brace_depth(text, upto):
    """Approximate nesting depth at an offset, ignoring comments and literals.

    Approximate on purpose: the only question this answers is "is this block
    inside a type body?", and a mis-count changes nothing else.
    """
    depth = 0
    i = 0
    body = text[:upto]
    while i < len(body):
        ch = body[i]
        if body.startswith("/*", i):
            end = body.find("*/", i + 2)
            i = len(body) if end < 0 else end + 2
            continue
        if body.startswith("//", i):
            end = body.find("\n", i)
            i = len(body) if end < 0 else end + 1
            continue
        if ch in "\"'":
            quote = ch
            i += 1
            while i < len(body):
                if body[i] == "\\":
                    i += 2
                    continue
                if body[i] == quote:
                    i += 1
                    break
                i += 1
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        i += 1
    return depth


def audit(root):
    """Apply D1-D4 to one tree and return problem strings."""
    files = [
        path
        for name in SCAN_DIRS
        for path in sorted((pathlib.Path(root) / name).rglob("*"))
        if path.is_file() and path.suffix in SUFFIXES
        and pathlib.Path(path).relative_to(root).as_posix() not in EXEMPT
    ]
    problems = []
    for path, block, line in scan(files, BLOCK):
        rel = pathlib.Path(path).relative_to(root).as_posix()
        where = f"{rel}:{line}"
        if not BRIEF_EN.search(block) or not BRIEF_ZH.search(block):
            problems.append(f"D1 {where}: Doxygen block lacks @brief [en] or @brief [zh]")
        if not SINCE.search(block):
            text = pathlib.Path(path).read_text(encoding="utf-8", errors="replace")
            offset = sum(len(item) + 1 for item in text.split("\n")[: line - 1])
            if brace_depth(text, offset) <= 0:
                problems.append(f"D2 {where}: Doxygen block lacks @since <version>")
    for path, block, line in scan(files, FILE_BLOCK):
        rel = pathlib.Path(path).relative_to(root).as_posix()
        where = f"{rel}:{line}"
        if "@file" not in block:
            problems.append(
                f"D5 {where}: a /*! block without @file dodges the version filter; "
                "use /** for symbol docs")
            continue
        if "@ingroup" not in block:
            problems.append(f"D5 {where}: file block must carry @ingroup")
        if not BRIEF_EN.search(block) or not BRIEF_ZH.search(block):
            problems.append(f"D1 {where}: Doxygen block lacks @brief [en] or @brief [zh]")
    for path, block, line in scan(files, BLOCK):
        rel = pathlib.Path(path).relative_to(root).as_posix()
        if "@file" in block:
            problems.append(
                f"D5 {rel}:{line}: a @file block must open with /*!, not /** "
                "(the version filter would drop it)")
    for path, block, line in scan(files, INLINE):
        rel = pathlib.Path(path).relative_to(root).as_posix()
        if re.search(r"\[(?:en|zh)\]", block):
            problems.append(
                f"D4 {rel}:{line}: inline member comment carries a language marker; "
                "the closing */ rides along and the comment never closes")
    for path in files:
        rel = pathlib.Path(path).relative_to(root).as_posix()
        text = path.read_text(encoding="utf-8", errors="replace")
        for line_no, line in enumerate(text.splitlines(), start=1):
            if "@exporter" in line or "@attacher" in line:
                at = text.find(line)
                if text.rfind("/**", 0, at) != text.rfind("/*", 0, at):
                    problems.append(
                        f"D3 {rel}:{line_no}: module tag outside a Doxygen block "
                        "(module generation fails silently)")
    return problems


def selftest():
    """One fixture per rule, each isolated so only that rule can fire."""
    import tempfile

    cases = (
        ("clean file passes", "src/ok.c",
         "/**\n * @brief [en] ok\n * @brief [zh] 好\n * @since 0.1.0\n */\nvoid ok(void) {}\n", None),
        ("D1 a monolingual block is red", "src/d1.c",
         "/**\n * @brief [en] only english\n * @since 0.1.0\n */\nvoid f(void) {}\n",
         "D1 src/d1.c:1: Doxygen block lacks @brief [en] or @brief [zh]"),
        ("D2 a block without @since is red", "src/d2.c",
         "/**\n * @brief [en] a\n * @brief [zh] 甲\n */\nvoid f(void) {}\n",
         "D2 src/d2.c:1: Doxygen block lacks @since"),
        ("D3 a module tag outside Doxygen is red", "src/d3.c",
         "/*\n * @exporter\n */\nvoid f(void) {}\n",
         "D3 src/d3.c:2: module tag outside a Doxygen block"),
        ("D4 a tagged inline member comment is red", "src/d4.c",
         "/**\n * @brief [en] a\n * @brief [zh] 甲\n * @since 0.1.0\n */\nint x; /**< [en] only english */\n",
         "D4 src/d4.c:6: inline member comment carries a language marker"),
        ("a member block without @since is accepted", "src/ok2.c",
         "/**\n * @brief [en] t\n * @brief [zh] 型\n * @since 0.1.0\n */\ntypedef struct s {\n    /**\n     * @brief [en] field\n     * @brief [zh] 字段\n     */\n    int f;\n} s;\n",
         None),
        ("a /*! file block with @ingroup is accepted", "src/ok3.c",
         "/*!\n * @ingroup group_x\n * @file\n * @brief [en] f\n * @brief [zh] 文\n * @since 0.1.0\n */\nvoid f(void) {}\n",
         None),
        ("D5 a /** file block is red", "src/d5.c",
         "/**\n * @file\n * @brief [en] f\n * @brief [zh] 文\n * @since 0.1.0\n */\nvoid f(void) {}\n",
         "D5 src/d5.c:1: a @file block must open with /*!, not /**"),
        ("D5 a /*! block without @ingroup is red", "src/d5b.c",
         "/*!\n * @file\n * @brief [en] f\n * @brief [zh] 文\n * @since 0.1.0\n */\nvoid f(void) {}\n",
         "D5 src/d5b.c:1: file block must carry @ingroup"),
    )
    failures = 0
    for name, rel, body, expected in cases:
        work = pathlib.Path(tempfile.mkdtemp())
        target = work / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(body, encoding="utf-8")
        problems = audit(work)
        joined = "\n".join(problems)
        ok = not problems if expected is None else expected in joined
        print(f"[selftest] {'OK  ' if ok else 'FAIL'} {name}")
        if not ok:
            failures += 1
            print(f"[selftest]   expected: {expected!r}\n[selftest]   got: {problems}")
    if failures:
        print(f"[selftest] {failures} case(s) FAILED", file=sys.stderr)
        return 1
    print(f"[selftest] {len(cases)} case(s) behave as specified")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--root", default=str(ROOT), help="tree to scan")
    args = parser.parse_args()

    if args.selftest:
        return selftest()
    problems = audit(args.root)
    if problems:
        for line in problems:
            print(f"::error:: {line}")
        print(f"[docannot] FAIL ({len(problems)} problem(s))", file=sys.stderr)
        return 1
    scanned = sum(
        1
        for name in SCAN_DIRS
        for path in pathlib.Path(args.root, name).rglob("*")
        if path.is_file() and path.suffix in SUFFIXES
    )
    print(f"[docannot] OK ({scanned} file(s) scanned in include/ + src/)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
