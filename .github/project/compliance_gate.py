#!/usr/bin/env python3
"""The standards ledger has to stay a ledger, and the patent ledger has to stay empty.

`workspace/compliance/` holds the material behind the claims: `metadata/standards.json` (the
standards and specifications each claim leans on), its `schema.json`, and
`patents/INDEX.md` -- a ledger that is deliberately empty until a real search fills it.

Why a gate rather than care:

  * a `status: have` entry with a `local_file` that does not resolve is a citation to a document
    nobody can open. The pointer is relative to `workspace/compliance/` (`../referential_repos/...`),
    and `..` drifts silently when directories move;
  * a fabricated patent number is worse than no ledger at all, so the rule is mechanical: no
    patent-number-shaped token may appear until a retrieval is recorded;
  * the schema is the record's contract; a field that exists in the data but not in the schema means
    the schema stopped describing the data.

Rules:

  K1 the ledger parses to a non-empty list; every entry carries id / kind / title / status /
     source_url / verified_at; ids are unique and shaped like `S-nnn`
  K2 `status` is one of the vocabulary the schema declares; `have` requires a `local_file` that
     resolves under `workspace/compliance/`
  K3 `patents/INDEX.md` says what it is (not a legal opinion, deliberately empty) and contains no
     patent-number-shaped token
  K4 every field an entry uses is declared in `schema.json` -- no field outside the contract

Usage:

    python3 .github/project/compliance_gate.py             # check
    python3 .github/project/compliance_gate.py --selftest  # prove each rule can fail
"""

import argparse
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
LEDGER = "workspace/compliance/metadata/standards.json"
SCHEMA = "workspace/compliance/metadata/schema.json"
PATENTS = "workspace/compliance/patents/INDEX.md"
REQUIRED = ("id", "kind", "title", "status", "source_url", "verified_at")
ID_SHAPE = re.compile(r"^S-\d{3}$")
PATENT_NUMBER = re.compile(r"\b(?:CN|US|EP|JP|KR|WO)\s?\d{6,}[A-Z]?\d?\b")


def check(root):
    problems = []
    reported = []
    root = pathlib.Path(root)
    ledger_path = root / LEDGER
    schema_path = root / SCHEMA
    patents_path = root / PATENTS

    if not ledger_path.is_file():
        problems.append(f"compliance: '{LEDGER}' is missing -- the claims have no citation ledger")
        return problems, reported
    if not schema_path.is_file():
        problems.append(f"compliance: '{SCHEMA}' is missing, so K2/K4 cannot be checked")
        return problems, reported

    entries = json.loads(ledger_path.read_text(encoding="utf-8"))
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    props = schema.get("properties") or {}
    statuses = set((props.get("status") or {}).get("enum") or [])

    # K1 -- shape of the ledger itself.
    if not isinstance(entries, list) or not entries:
        problems.append("compliance: the standards ledger must be a non-empty list")
        return problems, reported

    seen = set()
    for index, entry in enumerate(entries):
        missing = [field for field in REQUIRED if not str(entry.get(field, "")).strip()]
        if missing:
            problems.append(
                f"compliance: standards[{index}] is missing {', '.join(missing)}"
            )
            continue
        ident = str(entry["id"])
        if not ID_SHAPE.match(ident):
            problems.append(f"compliance: standards[{index}] id '{ident}' is not shaped S-nnn")
        if ident in seen:
            problems.append(f"compliance: id '{ident}' appears twice")
        seen.add(ident)

    # K2 -- the vocabulary, and the pointers that have to resolve.
    if not statuses:
        problems.append("compliance: schema.json declares no status enum, so K2 would pass vacuously")
    for entry in entries:
        if not isinstance(entry, dict) or "status" not in entry:
            continue
        status = entry["status"]
        if statuses and status not in statuses:
            problems.append(
                f"compliance: '{entry.get('id')}' has status '{status}', which the schema does not "
                f"declare (it allows {sorted(statuses)})"
            )
        if status == "have":
            local = str(entry.get("local_file", "")).strip()
            if not local:
                problems.append(
                    f"compliance: '{entry.get('id')}' claims status 'have' without a local_file"
                )
            elif not (root / "workspace/compliance" / local).exists():
                problems.append(
                    f"compliance: '{entry.get('id')}' points at '{local}', which does not resolve "
                    f"under workspace/compliance/ -- the citation cannot be opened"
                )

    # K3 -- the patent ledger stays a ledger.
    if not patents_path.is_file():
        problems.append(f"compliance: '{PATENTS}' is missing -- the empty ledger is still a document")
    else:
        text = patents_path.read_text(encoding="utf-8")
        lowered = text.lower()
        if "法律意见" not in text and "legal opinion" not in lowered:
            problems.append(
                "compliance: the patent ledger must say it is not a legal opinion; a reader who "
                "takes it for one will rely on it"
            )
        if "留空" not in text and "deliberately empty" not in lowered:
            problems.append(
                "compliance: the patent ledger must say it is deliberately empty until a real search "
                "fills it"
            )
        found = PATENT_NUMBER.findall(text)
        if found:
            problems.append(
                f"compliance: the patent ledger contains {len(found)} patent-number-shaped token(s) "
                f"({', '.join(found[:3])}); a number must come with a retrieval record, not from memory"
            )

    # K4 -- the schema describes the data.
    for entry in entries:
        if not isinstance(entry, dict):
            problems.append("compliance: every ledger entry must be an object")
            continue
        undeclared = sorted(set(entry) - set(props))
        if undeclared:
            problems.append(
                f"compliance: '{entry.get('id')}' uses {', '.join(undeclared)}, which schema.json does "
                f"not declare -- the schema stopped describing the data"
            )

    reported.append(f"{len(entries)} standard(s), {len(seen)} unique id(s), statuses {sorted(statuses)}")
    return problems, reported


def selftest():
    import tempfile

    failures = 0
    with tempfile.TemporaryDirectory(prefix="compliance-selftest-") as workdir:
        root = pathlib.Path(workdir)
        (root / "workspace/compliance/metadata").mkdir(parents=True)
        (root / "workspace/compliance/patents").mkdir(parents=True)
        # `local_file` is relative to workspace/compliance, so `..` lands in workspace/
        (root / "workspace/referential_repos/thing").mkdir(parents=True)

        def write(entries, patents=None, schema_props=None):
            (root / LEDGER).write_text(json.dumps(entries), encoding="utf-8")
            (root / SCHEMA).write_text(json.dumps({
                "properties": schema_props or {
                    "id": {"type": "string"}, "kind": {"enum": ["standard", "spec"]},
                    "title": {"type": "string"}, "status": {"enum": ["have", "to-collect"]},
                    "source_url": {"type": "string"}, "verified_at": {"type": "string"},
                    "local_file": {"type": "string"},
                }
            }), encoding="utf-8")
            body = patents if patents is not None else (
                "# 专利检索台账\n\n> 这不是法律意见。本表刻意留空。\n")
            (root / PATENTS).write_text(body, encoding="utf-8")
            return check(root)[0]

        def entry(**over):
            base = {"id": "S-001", "kind": "standard", "title": "ISO/IEC 9899:2018",
                    "status": "have", "source_url": "https://example.org", "verified_at": "2026-11-26",
                    "local_file": "../referential_repos/thing"}
            base.update(over)
            return base

        if not write([entry()]):
            print("[compliance:selftest] PASS K1..K4 a coherent ledger is green")
        else:
            failures += 1
            print(f"[compliance:selftest] FAIL expected a clean run, got {write([entry()])}")

        problems = write([entry(verified_at="")])
        if any("is missing verified_at" in p for p in problems):
            print("[compliance:selftest] PASS K1 a missing field is red")
        else:
            failures += 1
            print(f"[compliance:selftest] FAIL K1 expected the missing-field report, got {problems}")

        problems = write([entry(status="have", local_file="../referential_repos/gone")])
        if any("does not resolve" in p for p in problems):
            print("[compliance:selftest] PASS K2 a `have` pointer that does not resolve is red")
        else:
            failures += 1
            print(f"[compliance:selftest] FAIL K2 expected the unresolved-pointer report, got {problems}")

        problems = write([entry(status="wishful")])
        if any("which the schema does not declare" in p for p in problems):
            print("[compliance:selftest] PASS K2 a status outside the schema is red")
        else:
            failures += 1
            print(f"[compliance:selftest] FAIL K2 expected the status-vocabulary report, got {problems}")

        problems = write([entry()], patents="# 台账\n\n> 这不是法律意见。本表刻意留空。\nCN114123456A\n")
        if any("patent-number-shaped token" in p for p in problems):
            print("[compliance:selftest] PASS K3 an invented patent number is red")
        else:
            failures += 1
            print(f"[compliance:selftest] FAIL K3 expected the patent-number report, got {problems}")

        problems = write([entry()], patents="# 台账\n\n| a | b |\n")
        if any("not a legal opinion" in p for p in problems):
            print("[compliance:selftest] PASS K3 a ledger without its disclaimer is red")
        else:
            failures += 1
            print(f"[compliance:selftest] FAIL K3 expected the disclaimer report, got {problems}")

        problems = write([entry(notes="a field the schema does not know")])
        if any("schema.json does not declare" in p for p in problems):
            print("[compliance:selftest] PASS K4 a field outside the schema is red")
        else:
            failures += 1
            print(f"[compliance:selftest] FAIL K4 expected the undeclared-field report, got {problems}")

    if failures:
        print(f"[compliance:selftest] FAIL ({failures} fixture(s))", file=sys.stderr)
        return 1
    print("[compliance:selftest] OK")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--selftest", action="store_true", help="prove each rule can fail")
    args = parser.parse_args()

    if args.selftest:
        return selftest()

    problems, reported = check(ROOT)
    if problems:
        for problem in problems:
            print(f"[compliance] {problem}", file=sys.stderr)
        print(f"[compliance] FAIL ({len(problems)} problem(s))", file=sys.stderr)
        return 1
    print(f"[compliance] OK ({len(reported)} check(s): the ledgers are ledgers)")
    for line in reported:
        print(f"[compliance]   {line}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
