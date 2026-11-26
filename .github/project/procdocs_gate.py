#!/usr/bin/env python3
"""The process documents must never be published, so nothing may publish them by accident.

Two independent locks, because "I will remember" is not one:

  * `.gitignore` keeps them out of `git add` and `git add -A` (the convenient paths);
  * this gate fails when one of them is in the index at all (the `git add -f` path, and any
    future edit to the ignore file that quietly drops a rule).

The owner's instruction (2026-11-26) is that the working papers -- the gap ledger and the plan --
cannot become public **by any route**, while the rest of the execution subset stays tracked. That
makes this gate a safety property rather than a style rule: it is the difference between "we did
not push them yet" and "they cannot be pushed".

Rules:

  P1 no process document is tracked: the index (`git ls-files`) contains no path matching a
     protected pattern
  P2 the ignore file still declares every protected pattern -- a lock that is only in a gate can be
     removed by the next person who finds it inconvenient, and a lock that is only in .gitignore is
     bypassed by `git add -f`
  P3 the documents still exist on disk: they are working papers, so deleting them is not the way to
     make P1 pass. "Self-discipline" here means kept but never shipped.
  P4 every table in a listed Markdown document is well formed: a block of consecutive `|` lines has a
     delimiter row under its header and the same number of **unescaped** pipes on every line. A
     Markdown table ends at the first blank line, so a row separated by a blank one is not a row any
     more -- it renders as bare `|` text, which is how the ledger became unreadable (2026-11-26).
  P5 ids are unique: two records claiming the same `G-<n>` means one of them lost its number.

Usage:

    python3 .github/project/procdocs_gate.py             # check
    python3 .github/project/procdocs_gate.py --selftest  # prove each rule can fail
"""

import argparse
import json
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA = ROOT / ".github/project/procdocs.json"


def load_data(path):
    data = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    if not isinstance(data.get("protected"), list) or not data["protected"]:
        raise SystemExit("procdocs: 'protected' must be a non-empty list of glob patterns")
    return data


def tracked_files(root):
    """Paths in the git index, relative to `root`."""
    done = subprocess.run(["git", "-C", str(root), "ls-files"], capture_output=True, text=True)
    if done.returncode != 0:
        raise SystemExit(f"procdocs: git ls-files failed: {done.stderr.strip()}")
    return [line for line in done.stdout.splitlines() if line.strip()]


def matches(path, pattern):
    """Glob match where `**` means "anything below", including the directory itself.

    `pathlib`'s `match` would treat `workspace/audit/**` as needing at least one more component,
    which would let `workspace/audit` itself (a symlink, or a file named like the directory) slip
    through. The rule here is the one a reader expects: the pattern's prefix is protected.
    """
    if pattern.endswith("/**"):
        prefix = pattern[: -len("/**")]
        return path == prefix or path.startswith(prefix + "/")
    if pattern.endswith("/*"):
        prefix = pattern[: -len("/*")]
        return path.startswith(prefix + "/") and "/" not in path[len(prefix) + 1 :]
    import fnmatch

    return fnmatch.fnmatchcase(path, pattern)


def normalized_ignore_target(rule):
    """What an ignore rule targets, comparable across the two spellings.

    `.gitignore` writes `/workspace/audit/` (leading and trailing slash) while the data file writes
    `workspace/audit/**` (a glob). Demanding the two be character-identical would make the data file
    hostage to the ignore file's formatting, so both sides are reduced to the same prefix here.
    """
    text = rule.strip()
    if text.startswith("/"):
        text = text[1:]
    if text.endswith("/**"):
        text = text[: -len("/**")]
    if text.endswith("/"):
        text = text[:-1]
    return text


def on_disk(root, patterns):
    found = {}
    for pattern in patterns:
        hits = [p for p in root.rglob("*") if p.is_file() and matches(str(p.relative_to(root)), pattern)]
        found[pattern] = hits
    return found


def unescaped_pipes(line):
    """Count `|` that mark a cell boundary. `\\|` is a literal pipe inside a cell (GFM)."""
    return len(re.findall(r"(?<!\\)\|", line))


def table_blocks(lines):
    """Maximal runs of lines that start with `|`, as (start, end) index pairs."""
    blocks = []
    start = None
    for index, line in enumerate(lines):
        if line.startswith("|"):
            if start is None:
                start = index
        elif start is not None:
            blocks.append((start, index))
            start = None
    if start is not None:
        blocks.append((start, len(lines)))
    return blocks


def check_markdown_tables(root, patterns):
    """P4 + P5 over the listed documents."""
    problems = []
    reported = []
    for pattern in patterns:
        for path in sorted(p for p in root.rglob("*") if p.is_file() and matches(str(p.relative_to(root)), pattern)):
            text = path.read_text(encoding="utf-8")
            lines = text.split("\n")
            blocks = table_blocks(lines)
            shaped = 0
            for start, end in blocks:
                rows = lines[start:end]
                where = f"{path.relative_to(root)}:{start + 1}"
                if len(rows) < 2 or not re.fullmatch(r"\|(:?-+:?\|)+\s*", rows[1].replace(" ", "")):
                    problems.append(
                        f"procdocs: {where} -- this table has no header + delimiter row, so it renders "
                        f"as bare `|` text (a blank line ends a table: keep rows consecutive)"
                    )
                    continue
                counts = {unescaped_pipes(row) for row in rows}
                if len(counts) != 1:
                    offenders = [
                        f"line {start + 1 + offset} ({unescaped_pipes(row)} pipes)"
                        for offset, row in enumerate(rows)
                        if unescaped_pipes(row) != unescaped_pipes(rows[0])
                    ]
                    problems.append(
                        f"procdocs: {where} -- rows disagree on the column count: {', '.join(offenders[:3])}. "
                        f"A `|` inside a cell (usually a code sample) must be written `\\|`."
                    )
                    continue
                shaped += 1
            ids = re.findall(r"^\|\s*(G-\d+)\s*\|", text, flags=re.MULTILINE)
            duplicates = sorted({gap for gap in ids if ids.count(gap) > 1})
            if duplicates:
                problems.append(
                    f"procdocs: {path.relative_to(root)} -- duplicated id(s): {', '.join(duplicates)}; "
                    f"one of them lost its number"
                )
            reported.append(
                f"{path.relative_to(root)}: {shaped} table(s) well formed, {len(ids)} id(s) unique"
            )
    return problems, reported


def check(root, data):
    problems = []
    reported = []
    patterns = data["protected"]

    # P1 -- the index. This is the rule that actually prevents publication.
    indexed = tracked_files(root)
    for pattern in patterns:
        hits = sorted(path for path in indexed if matches(path, pattern))
        if hits:
            problems.append(
                f"procdocs: '{pattern}' is a process document but is tracked: {', '.join(hits[:3])}"
                f"{' ...' if len(hits) > 3 else ''} -- it would be published with the next push. "
                f"Untrack it with `git rm --cached` (the file stays on disk)."
            )
        else:
            reported.append(f"{pattern}: not tracked")

    # P2 -- the ignore file, so `git add -A` cannot pick it up either.
    ignore_path = root / data.get("ignore_file", ".gitignore")
    try:
        ignore_text = ignore_path.read_text(encoding="utf-8")
    except OSError as exc:
        problems.append(f"procdocs: cannot read {ignore_path.name}: {exc}")
        ignore_text = ""
    rules = [line.strip() for line in ignore_text.splitlines()]
    ignore_targets = {normalized_ignore_target(rule) for rule in rules
                      if rule and not rule.startswith("#")}
    for pattern in patterns:
        if normalized_ignore_target(pattern) not in ignore_targets:
            problems.append(
                f"procdocs: '{pattern}' is not ignored in {ignore_path.name} -- the rule must be "
                f"written there too, because this gate only runs when someone runs it"
            )
        else:
            reported.append(f"{pattern}: ignored by {ignore_path.name}")

    # P3 -- the papers exist. Making P1 pass by deleting the ledger would be the wrong fix.
    for pattern, hits in on_disk(root, patterns).items():
        if not hits:
            problems.append(
                f"procdocs: '{pattern}' matches no file on disk -- the working papers are meant to "
                f"be kept locally; deleting them is not how this rule is satisfied"
            )
        else:
            reported.append(f"{pattern}: {len(hits)} file(s) kept locally")

    # P4 + P5 -- the papers have to stay readable and unnumbered-twice.
    markdown_problems, markdown_reported = check_markdown_tables(root, data.get("markdown", []))
    problems.extend(markdown_problems)
    reported.extend(markdown_reported)
    return problems, reported


def selftest():
    import tempfile

    failures = 0
    with tempfile.TemporaryDirectory(prefix="procdocs-selftest-") as workdir:
        root = pathlib.Path(workdir)
        subprocess.run(["git", "init", "-q", str(root)], check=True)
        subprocess.run(["git", "-C", str(root), "config", "user.email", "t@t"], check=True)
        subprocess.run(["git", "-C", str(root), "config", "user.name", "t"], check=True)
        (root / "workspace/audit").mkdir(parents=True)
        (root / "workspace/audit/gaps.md").write_text("ledger\n", encoding="utf-8")
        (root / "workspace/PLAN-x.md").write_text("plan\n", encoding="utf-8")
        (root / "src").mkdir()
        (root / "src/code.c").write_text("int x;\n", encoding="utf-8")
        data = {
            "protected": ["workspace/audit/**", "workspace/PLAN-*.md"],
            "ignore_file": ".gitignore",
        }

        def write_ignore(body):
            (root / ".gitignore").write_text(body, encoding="utf-8")

        def commit_all():
            subprocess.run(["git", "-C", str(root), "add", "-A"], check=True)
            subprocess.run(["git", "-C", str(root), "commit", "-qm", "x"], check=True)

        # P1: a tracked process document is the failure this whole gate exists for. `git add -f` is
        # how it happens in practice -- exactly the route the ignore file cannot close.
        write_ignore("/workspace/audit/\n/workspace/PLAN-*.md\n")
        commit_all()
        subprocess.run(["git", "-C", str(root), "add", "-f", "workspace/audit/gaps.md",
                        "workspace/PLAN-x.md"], check=True)
        subprocess.run(["git", "-C", str(root), "commit", "-qm", "forced"], check=True)
        problems, _ = check(root, data)
        if any("is tracked" in p for p in problems):
            print("[procdocs:selftest] PASS P1 a tracked process document is red")
        else:
            failures += 1
            print(f"[procdocs:selftest] FAIL P1 expected a tracked-document report, got {problems}")

        # Untracked, ignored, still on disk: green.
        subprocess.run(["git", "-C", str(root), "rm", "-q", "--cached", "workspace/audit/gaps.md",
                        "workspace/PLAN-x.md"], check=True)
        commit_all()
        problems, _ = check(root, data)
        if not problems:
            print("[procdocs:selftest] PASS P1 untracked + ignored + kept locally is green")
        else:
            failures += 1
            print(f"[procdocs:selftest] FAIL P1 expected a clean run, got {problems}")

        # P2: the ignore rules removed while the gate is the only lock.
        write_ignore("")
        problems, _ = check(root, data)
        if sum("is not ignored" in p for p in problems) == len(data["protected"]):
            print("[procdocs:selftest] PASS P2 dropping the ignore rule is red (both patterns)")
        else:
            failures += 1
            print(f"[procdocs:selftest] FAIL P2 expected every pattern reported, got {problems}")

        # P3: deleting the working papers must not be a way to pass.
        write_ignore("/workspace/audit/\n/workspace/PLAN-*.md\n")
        (root / "workspace/audit/gaps.md").unlink()
        (root / "workspace/PLAN-x.md").unlink()
        problems, _ = check(root, data)
        if sum("matches no file on disk" in p for p in problems) == len(data["protected"]):
            print("[procdocs:selftest] PASS P3 deleting the working papers is red")
        else:
            failures += 1
            print(f"[procdocs:selftest] FAIL P3 expected every pattern reported, got {problems}")

        # A pattern that is too broad would protect ordinary code: `workspace/**` must not be used.
        problems, _ = check(root, {"protected": ["src/**"], "ignore_file": ".gitignore"})
        if any("is tracked" in p for p in problems):
            print("[procdocs:selftest] PASS the rule fires on any protected path, not just the ledger")
        else:
            failures += 1
            print(f"[procdocs:selftest] FAIL expected src/code.c to be reported, got {problems}")

        # P4/P5 fixtures run on their own file so the P1..P3 fixtures stay untouched.
        doc = root / "workspace/audit/ledger.md"
        table_data = {"protected": ["workspace/audit/**"], "ignore_file": ".gitignore",
                      "markdown": ["workspace/audit/ledger.md"]}
        head = "| # | what |\n|---|---|\n"

        def tables(body):
            doc.write_text(body, encoding="utf-8")
            return check(root, table_data)[0]

        # A row after a blank line is not a row: the table ended, and the row renders as bare text.
        problems = tables(head + "| G-1 | a |\n" + "\n" + "| G-2 | b |\n")
        if any("no header + delimiter row" in p for p in problems):
            print("[procdocs:selftest] PASS P4 a blank line inside a table is red")
        else:
            failures += 1
            print(f"[procdocs:selftest] FAIL P4 expected the blank-line report, got {problems}")

        # An unescaped pipe inside a cell adds a column.
        problems = tables(head + "| G-1 | `a | b` |\n")
        if any("disagree on the column count" in p for p in problems):
            print("[procdocs:selftest] PASS P4 a `|` in a cell is red")
        else:
            failures += 1
            print(f"[procdocs:selftest] FAIL P4 expected the column-count report, got {problems}")

        # ...and the GFM escape is green: this is the twin that stops the rule from banning code samples.
        problems = tables(head + "| G-1 | `a \\| b` |\n| G-2 | c |\n")
        if not problems:
            print("[procdocs:selftest] PASS P4 an escaped `\\|` in a cell is green")
        else:
            failures += 1
            print(f"[procdocs:selftest] FAIL P4 expected a clean run, got {problems}")

        # Two records with the same number: one of them has lost its identity.
        problems = tables(head + "| G-1 | a |\n| G-1 | b |\n")
        if any("duplicated id" in p for p in problems):
            print("[procdocs:selftest] PASS P5 a duplicated id is red")
        else:
            failures += 1
            print(f"[procdocs:selftest] FAIL P5 expected the duplicate report, got {problems}")
        doc.unlink()

    if failures:
        print(f"[procdocs:selftest] FAIL ({failures} fixture(s))", file=sys.stderr)
        return 1
    print("[procdocs:selftest] OK")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default=str(DATA), help="protected-path data")
    parser.add_argument("--selftest", action="store_true", help="prove each rule can fail")
    args = parser.parse_args()

    if args.selftest:
        return selftest()

    problems, reported = check(ROOT, load_data(args.data))
    if problems:
        for problem in problems:
            print(f"[procdocs] {problem}", file=sys.stderr)
        print(f"[procdocs] FAIL ({len(problems)} problem(s))", file=sys.stderr)
        return 1
    print(f"[procdocs] OK ({len(reported)} check(s): process documents cannot be published)")
    for line in reported:
        print(f"[procdocs]   {line}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
