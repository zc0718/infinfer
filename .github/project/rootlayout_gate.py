#!/usr/bin/env python3
"""INV-11: the repository root is an inventory, not a dumping ground.

Adding a tool is cheapest as a new top-level directory, and a dozen of those is how a repository
becomes unreadable. The rule is that every root entry belongs to one of three declared groups --
template-owned, project content, tool state -- and that the inventory has not drifted stale.

Rules:

  L1 every entry in the root is declared in `rootlayout.json` (an undeclared directory is red)
  L2 every declared entry exists (a stale inventory would make L1 pass while describing a tree
     that no longer exists)
  L3 no entry is declared twice, and no group is empty -- an empty group means the inventory was
     gutted

`--selftest` proves L1 and L2 can each fail on their own fixture.
"""

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA = ROOT / ".github/project/rootlayout.json"
GROUPS = ("template", "project", "toolchain")


def declared(data):
    out = {}
    problems = []
    for group in GROUPS:
        entries = data.get(group)
        if not isinstance(entries, list) or not entries:
            problems.append(f"rootlayout: group '{group}' must be a non-empty list")
            continue
        for entry in entries:
            if entry in out:
                problems.append(
                    f"rootlayout: '{entry}' is declared twice ('{out[entry]}' and '{group}')"
                )
                continue
            out[entry] = group
    return out, problems


def check(root, data):
    entries, problems = declared(data)
    actual = {path.name for path in pathlib.Path(root).iterdir()}
    for name in sorted(actual - set(entries)):
        problems.append(
            f"rootlayout: '{name}' is in the repository root but not declared in rootlayout.json "
            "(INV-11: new tools belong under workspace/ unless CI must run them)"
        )
    for name in sorted(set(entries) - actual):
        # A directory declared local-only is *absent* on a clean clone, which is the state CI is
        # in. That is the documented state, not a stale inventory -- saying so beats a red that
        # nobody can act on, while the rule still holds for every non-local entry.
        if name in (data.get("local_only") or ()):
            print(f"[rootlayout] {name}: absent locally (declared local-only) -- skipped")
            continue
        problems.append(
            f"rootlayout: '{name}' is declared as {entries[name]} but does not exist -- the "
            "inventory is stale, so the rule above is describing a tree that is gone"
        )
    return problems, len(actual)


def selftest():
    failures = 0
    import tempfile

    data = {
        "template": ["CMakeLists.txt", "README.md"],
        "project": ["src"],
        "toolchain": [".git"],
    }
    with tempfile.TemporaryDirectory(prefix="rootlayout-selftest-") as workdir:
        root = pathlib.Path(workdir)
        for name in ("CMakeLists.txt", "README.md"):
            (root / name).write_text("", encoding="utf-8")
        (root / "src").mkdir()
        (root / ".git").mkdir()

        problems, _ = check(root, data)
        if problems:
            failures += 1
            print(f"[rootlayout:selftest] FAIL a matching root must be green, got {problems}")
        else:
            print("[rootlayout:selftest] PASS a matching root is green")

        (root / "tools").mkdir()
        problems, _ = check(root, data)
        if any("'tools' is in the repository root but not declared" in p for p in problems):
            print("[rootlayout:selftest] PASS L1 an undeclared root entry is red")
        else:
            failures += 1
            print(f"[rootlayout:selftest] FAIL L1 expected an undeclared-entry report, got {problems}")
        (root / "tools").rmdir()

        (root / "README.md").unlink()
        problems, _ = check(root, data)
        if any("but does not exist" in p for p in problems):
            print("[rootlayout:selftest] PASS L2 a stale inventory entry is red")
        else:
            failures += 1
            print(f"[rootlayout:selftest] FAIL L2 expected a stale-entry report, got {problems}")

        _, problems = declared({"template": ["a", "a"], "project": ["b"], "toolchain": ["c"]})
        if any("declared twice" in p for p in problems):
            print("[rootlayout:selftest] PASS L3 a duplicate declaration is red")
        else:
            failures += 1
            print(f"[rootlayout:selftest] FAIL L3 expected a duplicate report, got {problems}")

    if failures:
        print(f"[rootlayout:selftest] FAIL ({failures} fixture(s))", file=sys.stderr)
        return 1
    print("[rootlayout:selftest] OK")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default=str(DATA), help="inventory file")
    parser.add_argument("--root", default=str(ROOT), help="tree whose root is checked")
    parser.add_argument("--selftest", action="store_true", help="prove the rules can fail")
    args = parser.parse_args()
    if args.selftest:
        return selftest()

    data = json.loads(pathlib.Path(args.data).read_text(encoding="utf-8"))
    problems, count = check(args.root, data)
    if problems:
        for problem in problems:
            print(f"[rootlayout] {problem}", file=sys.stderr)
        print(f"[rootlayout] FAIL ({len(problems)} problem(s))", file=sys.stderr)
        return 1
    print(f"[rootlayout] OK ({count} root entries, all declared and present)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
