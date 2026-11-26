#!/usr/bin/env python3
"""D3 · no development-domain vocabulary in the shipped surface.

The contract this enforces: a comment in `include/` or `src/` describes the code as
it is, for someone who has never read this project's plan. Plan-stage identifiers
(`INV-3`, `P1.3`), internal counters (`MVP-7`), gate names (`census gate`,
`gate/census`), issue numbers and the planning documents themselves are vocabulary
of *how* the code was built, not of *what* it does. A reader outside the team has no
way to resolve any of them, and once one is copied into a header it reads as a
broken cross-reference.

What is deliberately NOT flagged:

  - `Cortex-M3` / `M7`: hardware names, not milestones. A rule on `\\bM\\d\\b` would
    fight the domain it is meant to protect.
  - references to real paths such as `src/kernel/matrix.yml`: those are navigable by
    anyone with the tree.

Usage:

    python3 .github/project/docvoice_gate.py            # check
    python3 .github/project/docvoice_gate.py --selftest  # prove each rule can fail
"""

import argparse
import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCAN_DIRS = ("include", "src")
SUFFIXES = (".h", ".hpp", ".c", ".cpp", ".ixx", ".cppm")

# One entry per family of forbidden vocabulary, each with the reason it is forbidden
# so a future contributor can tell "this is a real ban" from "this was convenient".
RULES = (
    ("plan-identifier", re.compile(r"\bINV-\d+\b"), "plan invariant number; state the rule instead"),
    ("plan-identifier", re.compile(r"\bMVP-\d+\b"), "validation-experiment number; not resolvable outside the plan"),
    ("plan-phase", re.compile(r"\bP\d\.\d\b"), "plan task number"),
    ("plan-document", re.compile(r"\bPLAN\b"), "reference to the planning document"),
    ("gate-name", re.compile(r"\bgate/"), "gate identifier as a path"),
    ("gate-name", re.compile(r"census[ _]gate|census 门禁", re.ASCII), "gate name; describe the check instead"),
    ("gate-name", re.compile(r"\bgaps\.md\b"), "reference to the local gap register"),
    ("issue-number", re.compile(r"\b(?:Issue|issue|#)\s?\d{2,}\b"), "issue number"),
)


def scan_text(text, rules=RULES):
    """Return [(line_no, rule, fragment, reason)] for every forbidden hit."""
    hits = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        for rule, pattern, reason in rules:
            match = pattern.search(line)
            if match:
                hits.append((line_no, rule, match.group(0), reason))
    return hits


def scan_tree(root=ROOT):
    """Scan the shipped surface; returns [(path, hits)] for files with at least one hit."""
    findings = []
    for name in SCAN_DIRS:
        base = Path(root) / name
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*")):
            if path.suffix not in SUFFIXES or not path.is_file():
                continue
            hits = scan_text(path.read_text(encoding="utf-8", errors="replace"))
            if hits:
                findings.append((path.relative_to(root), hits))
    return findings


def selftest():
    """Prove every rule can fire, and that the hardware-name exemption holds."""
    failures = 0
    text = "\n".join(
        [
            "/* fine: Cortex-M3 and Cortex-M7 are hardware, not milestones */",
            "/* fine: see src/kernel/matrix.yml for the declared pairs */",
            "/* bad: INV-7 says every board must be built */",
            "/* bad: this lands in MVP-10 */",
            "/* bad: P1.3 generates the golden */",
            "/* bad: PLAN section 3.4 decides this */",
            "/* bad: see gate/apicontract for the rule */",
            "/* bad: the census gate rejects orphans */",
            "/* bad: recorded in gaps.md */",
            "/* bad: fixed by Issue 42 */",
        ]
    )
    hits = scan_text(text)
    found_lines = {line_no for line_no, _, _, _ in hits}

    # Every forbidden line must be caught ...
    for line_no in (3, 4, 5, 6, 7, 8, 9, 10):
        if line_no not in found_lines:
            failures += 1
            print(f"[docvoice:selftest] FAIL line {line_no} should have been flagged")
    # ... and the two exempt lines must not be.
    for line_no in (1, 2):
        if line_no in found_lines:
            failures += 1
            print(f"[docvoice:selftest] FAIL line {line_no} is legitimate: {text.splitlines()[line_no - 1]}")

    # The tree scan must actually walk the surface, not an empty set.
    with tempfile.TemporaryDirectory(prefix="docvoice-selftest-") as workdir:
        fake = Path(workdir)
        (fake / "include").mkdir()
        (fake / "include" / "bad.h").write_text("/* INV-1 forbids this */\n", encoding="utf-8")
        (fake / "src").mkdir()
        (fake / "src" / "good.c").write_text("/* no plan vocabulary here */\n", encoding="utf-8")
        findings = scan_tree(fake)
        if [str(path) for path, _ in findings] != ["include/bad.h"]:
            failures += 1
            print(f"[docvoice:selftest] FAIL the tree scan should report only include/bad.h, got {findings}")

    if failures:
        print(f"[docvoice:selftest] FAIL ({failures})", file=sys.stderr)
        return 1
    print("[docvoice:selftest] OK")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--selftest", action="store_true", help="prove every rule can fail")
    args = parser.parse_args()
    if args.selftest:
        return selftest()

    findings = scan_tree()
    if findings:
        for path, hits in findings:
            for line_no, rule, fragment, reason in hits:
                print(f"[docvoice] {path}:{line_no}: {rule} '{fragment}' -- {reason}", file=sys.stderr)
        total = sum(len(hits) for _, hits in findings)
        print(f"[docvoice] FAIL ({total} hit(s) in {len(findings)} file(s))", flush=True)
        return 1
    scanned = sum(
        1
        for name in SCAN_DIRS
        for path in (ROOT / name).rglob("*")
        if path.suffix in SUFFIXES and path.is_file()
    )
    print(f"[docvoice] OK ({scanned} file(s) scanned, no development-domain vocabulary)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
