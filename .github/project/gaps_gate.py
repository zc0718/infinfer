#!/usr/bin/env python3
"""Gap ids cited from the tree must exist in the ledger -- when the ledger is here.

The gap ledger (`workspace/audit/gaps.md`) is a **local-only** process document (gaps G-76), so it
is deliberately not in git. That has a consequence worth enforcing rather than remembering: every
`G-<n>` written into a source comment, the matrix or a gate's data is a *dangling label* in any
clone, and a label can also be wrong here -- a typo, or an id cited before it was written.

So the rule is conditional, and the condition is printed instead of hidden:

  * ledger present  -> every cited id must exist in it (this is the local-machine check)
  * ledger absent   -> SKIP with the reason and the number of citations that went unchecked, which
                       is what a fresh clone gets, and which is honest rather than green-washed

Rules:

  G1 every cited `G-<n>` exists in the ledger
  G2 the ledger parses to a non-empty id set -- a ledger the parser cannot read must not silently
     turn G1 into "no citations are wrong"
  G3 the citation scan is not empty: a run that found nothing to check is reported as a skip, not
     as a pass

Usage:

    python3 .github/project/gaps_gate.py             # check
    python3 .github/project/gaps_gate.py --selftest  # prove each rule can fail
"""

import argparse
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
LEDGER = "workspace/audit/gaps.md"
CITATION = re.compile(r"\bG-(\d{1,3})\b")
ROW = re.compile(r"^\|\s*(G-\d{1,3})\s*\|", re.MULTILINE)
SKIP = "SKIP"


def tracked_text_files(root, suffixes):
    done = subprocess.run(["git", "-C", str(root), "ls-files"], capture_output=True, text=True)
    if done.returncode != 0:
        raise SystemExit(f"gaps: git ls-files failed: {done.stderr.strip()}")
    return [path for path in done.stdout.splitlines() if path.endswith(tuple(suffixes))]


def scan(root, files):
    """Every cited id, and where it came from."""
    found = {}
    for rel in files:
        path = root / rel
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for match in CITATION.finditer(text):
            found.setdefault(f"G-{int(match.group(1))}", set()).add(rel)
    return found


def ledger_ids(root, ledger=None):
    path = pathlib.Path(ledger) if ledger is not None else root / LEDGER
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8")
    return {f"G-{int(m.group(1)[2:])}" for m in ROW.finditer(text)}


def check(root, ledger=None, suffixes=(".c", ".h", ".cpp", ".hpp", ".yml", ".yaml", ".json", ".py")):
    problems = []
    reported = []
    cited = scan(root, tracked_text_files(root, suffixes))
    ids = ledger_ids(root, ledger)

    if ids is None:
        # G3: the honest answer is a skip with the count, not a pass.
        print(f"[gaps] {SKIP} the ledger '{LEDGER}' is not in this tree (it is a local-only "
              f"process document, gaps G-76), so {len(cited)} cited id(s) across "
              f"{len(set().union(*cited.values())) if cited else 0} file(s) could not be checked")
        return None, reported

    if not ids:
        problems.append(f"gaps: '{LEDGER}' parsed to no id at all -- G1 would pass vacuously")
    if not cited:
        problems.append("gaps: no G-<n> citation was found in any tracked file -- the scan is not "
                        "doing what it claims (G3)")

    dangling = {name: files for name, files in cited.items() if ids and name not in ids}
    for name in sorted(dangling, key=lambda x: int(x[2:])):
        where = ", ".join(sorted(dangling[name])[:3])
        problems.append(f"gaps: {name} is cited from {where} but no record with that number exists "
                        f"in the ledger")
    reported.append(f"{len(cited)} cited id(s) across {len(set().union(*cited.values())) if cited else 0} "
                    f"file(s), {len(ids)} record(s) in the ledger")
    return problems, reported


def selftest():
    import tempfile

    failures = 0
    with tempfile.TemporaryDirectory(prefix="gaps-selftest-") as workdir:
        root = pathlib.Path(workdir)
        subprocess.run(["git", "init", "-q", str(root)], check=True)
        subprocess.run(["git", "-C", str(root), "config", "user.email", "t@t"], check=True)
        subprocess.run(["git", "-C", str(root), "config", "user.name", "t"], check=True)
        (root / "src").mkdir()
        (root / "workspace/audit").mkdir(parents=True)
        ledger = root / LEDGER
        ledger.write_text(
            "| # | what |\n|---|---|\n| G-1 | a |\n| G-2 | b |\n", encoding="utf-8"
        )
        code = root / "src/code.c"
        code.write_text("/* see G-1 */\nint x;\n", encoding="utf-8")
        subprocess.run(["git", "-C", str(root), "add", "src/code.c", "src"], check=True)
        subprocess.run(["git", "-C", str(root), "commit", "-qm", "fixture"], check=True)

        def run():
            return check(root, ledger=ledger)[0]

        problems = run()
        if not problems:
            print("[gaps:selftest] PASS G1 a citation of a real record is green")
        else:
            failures += 1
            print(f"[gaps:selftest] FAIL G1 expected a clean run, got {problems}")

        code.write_text("/* see G-1 and G-9 */\nint x;\n", encoding="utf-8")
        subprocess.run(["git", "-C", str(root), "add", "src/code.c"], check=True)
        subprocess.run(["git", "-C", str(root), "commit", "-qm", "cite"], check=True)
        problems = run()
        if any("G-9 is cited" in p for p in problems):
            print("[gaps:selftest] PASS G1 a citation of a missing record is red")
        else:
            failures += 1
            print(f"[gaps:selftest] FAIL G1 expected the dangling report, got {problems}")

        ledger.write_text("nothing here\n", encoding="utf-8")
        problems = run()
        if any("parsed to no id" in p for p in problems):
            print("[gaps:selftest] PASS G2 an unparsable ledger is red")
        else:
            failures += 1
            print(f"[gaps:selftest] FAIL G2 expected the empty-ledger report, got {problems}")

        ledger.write_text("| # | what |\n|---|---|\n| G-1 | a |\n", encoding="utf-8")
        code.write_text("int x;\n", encoding="utf-8")
        subprocess.run(["git", "-C", str(root), "add", "src/code.c"], check=True)
        subprocess.run(["git", "-C", str(root), "commit", "-qm", "no citation"], check=True)
        problems = run()
        if any("no G-<n> citation was found" in p for p in problems):
            print("[gaps:selftest] PASS G3 scanning nothing is red, not green")
        else:
            failures += 1
            print(f"[gaps:selftest] FAIL G3 expected the empty-scan report, got {problems}")

        # The ledger is local-only: a clone without it must skip with a reason, not fail.
        ledger.unlink()
        import io
        import contextlib

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            result = check(root, ledger=ledger)[0]
        printed = buffer.getvalue()
        if result is None and SKIP in printed and "G-76" in printed:
            print("[gaps:selftest] PASS the local-only ledger turns the check into a printed skip")
        else:
            failures += 1
            print(f"[gaps:selftest] FAIL expected a printed skip, got {result} / {printed!r}")

    if failures:
        print(f"[gaps:selftest] FAIL ({failures} fixture(s))", file=sys.stderr)
        return 1
    print("[gaps:selftest] OK")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--selftest", action="store_true", help="prove each rule can fail")
    parser.add_argument("--ledger", default=None, help="override the ledger path")
    args = parser.parse_args()

    if args.selftest:
        return selftest()

    ledger = pathlib.Path(args.ledger) if args.ledger else None
    problems, reported = check(ROOT, ledger=ledger)
    if problems is None:
        return 0
    if problems:
        for problem in problems:
            print(f"[gaps] {problem}", file=sys.stderr)
        print(f"[gaps] FAIL ({len(problems)} problem(s))", file=sys.stderr)
        return 1
    print(f"[gaps] OK ({len(reported)} check(s): every cited gap id exists)")
    for line in reported:
        print(f"[gaps]   {line}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
