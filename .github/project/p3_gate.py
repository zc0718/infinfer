#!/usr/bin/env python3
"""P3 has no third state: a gate is either built, or consciously deferred with a trigger.

PLAN P3 lists the evidence-and-gate surface it wants. Fourteen of those names were still
open, and "open" hides two very different things -- a gate nobody got to yet, and a gate
whose subject does not exist (`vendor` before any vendor backend, `oracle` before any
reference pipeline). Left as prose, both look identical and the list never closes.

`p3-gates.json` is that list as data. Every name is either

  covered   -- built somewhere in this repository, naming the file and the rule
  deferred  -- not built, with *why* and the *trigger* that makes it buildable

and this gate holds the two ends together:

  Q1 ids are unique
  Q2 every entry is `covered` or `deferred`
  Q3 a covered entry names a file that exists **and** that `ci.sh` runs, and says what it
     checks -- a covered entry pointing at nothing is the same lie as an unbuilt gate
  Q4 a deferred entry carries both `why` and `trigger`; "not yet" without a trigger is how
     a deferral becomes permanent
  Q5 every `*_gate.py` in this directory is named by some covered entry: a gate that runs
     but is not in the inventory is the third state coming back in through the side door
  Q6 `plan_targets` is exactly the set of ids marked `from: plan`, so a name cannot be
     dropped from the list by deleting its entry

`--selftest` gives each rule its own fixture and asserts on that rule's wording.

Usage:

    python3 .github/project/p3_gate.py             # check
    python3 .github/project/p3_gate.py --selftest  # prove each rule can fail
"""

import argparse
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA = ROOT / ".github/project/p3-gates.json"
STATUSES = ("covered", "deferred")


def load_json(path):
    return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))


def ci_gate_files(root):
    """Gate file names `ci.sh` invokes, in the order it invokes them."""
    ci = pathlib.Path(root) / ".github/project/ci.sh"
    if not ci.is_file():
        return []
    return re.findall(r"([a-z0-9_]+_gate\.py)", ci.read_text(encoding="utf-8"))


def check(root, data_path=None):
    problems = []
    reported = []
    data = load_json(data_path or pathlib.Path(root) / ".github/project/p3-gates.json")
    entries = data.get("entries") or []
    if not entries:
        problems.append("p3: the inventory is empty, so every rule below would pass vacuously")

    invoked = set(ci_gate_files(root))
    if not invoked:
        problems.append("p3: could not read any gate name out of ci.sh")

    # Q1 / Q2 -- shape.
    seen = set()
    for entry in entries:
        name = entry.get("id", "<no id>")
        if name in seen:
            problems.append(f"p3: '{name}' appears twice in the inventory")
        seen.add(name)
        if entry.get("status") not in STATUSES:
            problems.append(
                f"p3: '{name}' has status '{entry.get('status')}', which is neither "
                f"{STATUSES[0]} nor {STATUSES[1]} -- there is no third state"
            )

    # Q3 -- covered means built, named and run.
    for entry in entries:
        if entry.get("status") != "covered":
            continue
        name = entry.get("id")
        by = str(entry.get("by", ""))
        gate_file = by.split(":")[0]
        if not gate_file.endswith("_gate.py"):
            problems.append(f"p3: '{name}' is covered by '{by}', which is not a gate file")
            continue
        if not (pathlib.Path(root) / ".github/project" / gate_file).is_file():
            problems.append(f"p3: '{name}' is covered by '{gate_file}', which does not exist")
        elif gate_file not in invoked:
            problems.append(
                f"p3: '{name}' is covered by '{gate_file}', which `ci.sh` never runs -- built "
                f"but not wired is not covered"
            )
        if not str(entry.get("what", "")).strip():
            problems.append(f"p3: '{name}' is covered but does not say what it checks")

    # Q4 -- deferred means decided.
    for entry in entries:
        if entry.get("status") != "deferred":
            continue
        name = entry.get("id")
        for field in ("why", "trigger"):
            if not str(entry.get(field, "")).strip():
                problems.append(
                    f"p3: '{name}' is deferred without '{field}' -- 'not yet' and 'never' look "
                    f"the same unless the trigger is written down"
                )

    # Q5 -- no undeclared gate file.
    declared = {
        str(entry.get("by", "")).split(":")[0]
        for entry in entries
        if entry.get("status") == "covered"
    }
    for path in sorted((pathlib.Path(root) / ".github/project").glob("*_gate.py")):
        if path.name not in declared:
            problems.append(
                f"p3: '{path.name}' runs but no inventory entry names it -- that is the third "
                f"state coming back through the side door"
            )

    # Q6 -- the plan's list cannot shrink by deletion.
    plan_ids = {entry.get("id") for entry in entries if entry.get("from") == "plan"}
    declared_plan = set(data.get("plan_targets") or [])
    missing = sorted(declared_plan - plan_ids)
    extra = sorted(plan_ids - declared_plan)
    if missing or extra:
        problems.append(
            f"p3: plan_targets and the entries disagree (missing {missing or 'none'}, "
            f"undeclared {extra or 'none'})"
        )

    counts = {status: sum(1 for e in entries if e.get("status") == status) for status in STATUSES}
    reported.append(
        f"{len(entries)} inventoried name(s): {counts['covered']} covered, "
        f"{counts['deferred']} deferred with a trigger"
    )
    return problems, reported


def selftest():
    import tempfile

    failures = 0
    with tempfile.TemporaryDirectory(prefix="p3-selftest-") as workdir:
        root = pathlib.Path(workdir)
        (root / ".github/project").mkdir(parents=True)
        (root / ".github/project/ci.sh").write_text(
            "for gate in real_gate.py; do python3 \"$gate\"; done\n", encoding="utf-8"
        )
        (root / ".github/project/real_gate.py").write_text("# gate\n", encoding="utf-8")

        def touch_stray():
            """A gate file ci.sh does not run: Q3 calls it not-wired, Q5 calls it undeclared."""
            (root / ".github/project/stray_gate.py").write_text("# never invoked\n", encoding="utf-8")

        def run(entries, plan_targets):
            data = {"schema": 1, "plan_targets": plan_targets, "entries": entries}
            path = root / "fixture.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            return check(root, data_path=path)[0]

        def covered(**over):
            base = {"id": "ports", "from": "plan", "status": "covered", "by": "real_gate.py:R1",
                    "what": "something checked"}
            base.update(over)
            return base

        def deferred(**over):
            base = {"id": "oracle", "from": "plan", "status": "deferred",
                    "why": "no reference pipeline yet", "trigger": "P5 lands the blob importer"}
            base.update(over)
            return base

        plan = ["ports", "oracle"]
        problems = run([covered(), deferred()], plan)
        if not problems:
            print("[p3:selftest] PASS Q1..Q6 a complete inventory is green")
        else:
            failures += 1
            print(f"[p3:selftest] FAIL expected a clean run, got {problems}")

        problems = run([covered(), covered(id="ports")], plan)
        if any("appears twice" in p for p in problems):
            print("[p3:selftest] PASS Q1 a duplicated id is red")
        else:
            failures += 1
            print(f"[p3:selftest] FAIL Q1 expected the duplicate report, got {problems}")

        problems = run([covered(), deferred(status="pending")], plan)
        if any("neither covered nor deferred" in p for p in problems):
            print("[p3:selftest] PASS Q2 a third status is red")
        else:
            failures += 1
            print(f"[p3:selftest] FAIL Q2 expected the status report, got {problems}")

        problems = run([covered(by="ghost_gate.py"), deferred()], plan)
        if any("which does not exist" in p for p in problems):
            print("[p3:selftest] PASS Q3 a covered entry naming a missing gate is red")
        else:
            failures += 1
            print(f"[p3:selftest] FAIL Q3 expected the missing-gate report, got {problems}")

        touch_stray()
        problems = run([covered(by="stray_gate.py"), deferred()], plan)
        if any("which `ci.sh` never runs" in p for p in problems):
            print("[p3:selftest] PASS Q3 a gate that never runs is not covered")
        else:
            failures += 1
            print(f"[p3:selftest] FAIL Q3 expected the not-wired report, got {problems}")

        problems = run([covered(), deferred(trigger="")], plan)
        if any("without 'trigger'" in p for p in problems):
            print("[p3:selftest] PASS Q4 a deferral without a trigger is red")
        else:
            failures += 1
            print(f"[p3:selftest] FAIL Q4 expected the trigger report, got {problems}")

        # Q5: stray_gate.py is not invoked by ci.sh, so only Q5 can name it.
        touch_stray()
        problems = run([covered(), deferred()], plan)
        if any("stray_gate.py' runs but no inventory entry" in p for p in problems):
            print("[p3:selftest] PASS Q5 an undeclared gate file is red")
        else:
            failures += 1
            print(f"[p3:selftest] FAIL Q5 expected the undeclared-gate report, got {problems}")

        problems = run([covered(), deferred()], ["ports"])
        if any("plan_targets and the entries disagree" in p for p in problems):
            print("[p3:selftest] PASS Q6 a plan target dropped from the entries is red")
        else:
            failures += 1
            print(f"[p3:selftest] FAIL Q6 expected the target-list report, got {problems}")

    if failures:
        print(f"[p3:selftest] FAIL ({failures} fixture(s))", file=sys.stderr)
        return 1
    print("[p3:selftest] OK")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default=str(DATA), help="gate inventory")
    parser.add_argument("--selftest", action="store_true", help="prove each rule can fail")
    args = parser.parse_args()

    if args.selftest:
        return selftest()

    problems, reported = check(ROOT, data_path=args.data)
    if problems:
        for problem in problems:
            print(f"[p3] {problem}", file=sys.stderr)
        print(f"[p3] FAIL ({len(problems)} problem(s))", file=sys.stderr)
        return 1
    print(f"[p3] OK ({len(reported)} check(s): every gate is built or consciously deferred)")
    for line in reported:
        print(f"[p3]   {line}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
