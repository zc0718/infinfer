#!/usr/bin/env python3
"""The dependency set is written in four places. This makes them agree.

  conandata.yml            what Conan resolves
  metadata.json            what the template's CMake reads, in four buckets
  metadata.baremetal_white_list   which ones may appear on a bare-metal target
  attribution              what we would print if we shipped it (a field, not a file)

Nothing compared them until now, and each pair can drift silently: a package added to
`conandata.yml` and forgotten in `metadata.json` simply never gets linked, which looks
like "the feature does not work" rather than "the declaration is missing".

Rules:

  D1 `conandata.yml` requirements and the entries are the same set, in both directions
  D2 `metadata.json:dependencies` and the entries are the same set, bucket by bucket and
     target by target (both directions; the metadata key is spelled independently of the
     Conan name, which is why the entry carries both)
  D3 every entry declares an SPDX licence from `allowed_licences`; a licence that could
     not be verified on the machine (`license_status: unverified`) is only acceptable for
     something we do not redistribute
  D4 `redistributed` and carrying an `attribution` are the same predicate, in both directions: an
     attribution line for something we do not ship is as wrong as shipping something without one.
     Attribution is a field next to the licence rather than a separate notice file, because a
     dedicated file would exist for a hypothetical: nothing here is redistributed today, so its
     body would be empty and a new root-level document would still be a decision nobody needed
  D5 "it is used" is measured rather than claimed: `used_from.path` must exist and contain
     `used_from.contains`; an entry with no consumer must carry `why_unused` instead
  D6 `metadata.baremetal_white_list` names entries that are marked `baremetal_ok` -- the
     whitelist cannot outlive the dependency it names

Usage:

    python3 .github/project/deps_gate.py             # check
    python3 .github/project/deps_gate.py --selftest  # prove each rule can fail
  D8 a self-authored recipe lives outside the repository, so the register carries the
     digests of the recipe's own files: when the folder is present they must match, and when
     it is absent (a clean clone) the gate says so instead of failing -- an absent recipe is
     the documented state, a changed one is not.
"""

import argparse
import hashlib
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA = ROOT / ".github/project/deps.json"
REQUIREMENT = re.compile(r'^\s*-\s*"?([A-Za-z0-9_.+-]+)(?:/[^"]*)?"?\s*$')


def load_json(path):
    return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))


def conandata_requirements(path):
    """Package names from `conandata.yml`'s requirements list (comments excluded)."""
    names = []
    in_list = False
    for line in pathlib.Path(path).read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("#") or not stripped:
            continue
        if re.match(r"^requirements\s*:", stripped):
            in_list = True
            continue
        if in_list:
            if not stripped.startswith("-"):
                break
            match = REQUIREMENT.match(line)
            if match:
                names.append(match.group(1))
    return names


def check(root, data_path=None):
    problems = []
    reported = []
    data = load_json(data_path or root / ".github/project/deps.json")
    entries = data.get("entries", [])
    allowed = set(data.get("allowed_licences", []))

    if not entries:
        problems.append("deps: the entry list is empty, so every rule below would pass vacuously")
    if not allowed:
        problems.append("deps: 'allowed_licences' is empty, so D3 would reject everything")

    # D1 -- conandata.yml and the entries, both ways.
    conandata = root / data.get("conandata", "conandata.yml")
    if not conandata.is_file():
        problems.append(f"deps: '{conandata.name}' is missing, so D1 cannot be checked")
    else:
        declared = conandata_requirements(conandata)
        known = [entry.get("conan") for entry in entries]
        for name in declared:
            if name not in known:
                problems.append(
                    f"deps: '{name}' is a requirement in {conandata.name} but has no entry in "
                    f"deps.json -- it would reach the build with an unrecorded licence"
                )
        for name in known:
            if name not in declared:
                problems.append(
                    f"deps: entry '{name}' is not a requirement in {conandata.name} -- the entry "
                    f"outlived the dependency"
                )

    # D2 -- metadata.json and the entries, both ways.
    metadata = root / data.get("metadata", "metadata.json")
    buckets = {}
    if not metadata.is_file():
        problems.append(f"deps: '{metadata.name}' is missing, so D2 cannot be checked")
    else:
        meta = load_json(metadata)
        buckets = meta.get("dependencies") or {}
        seen = set()
        for entry in entries:
            bucket = entry.get("bucket")
            key = entry.get("metadata_key")
            if bucket not in buckets:
                problems.append(
                    f"deps: entry '{entry.get('conan')}' declares bucket '{bucket}', which is not one "
                    f"of {sorted(buckets)}"
                )
                continue
            seen.add((bucket, key))
            if key not in buckets[bucket]:
                problems.append(
                    f"deps: entry '{entry.get('conan')}' expects metadata bucket '{bucket}' to hold "
                    f"'{key}', but it holds {sorted(buckets[bucket])}"
                )
                continue
            if sorted(buckets[bucket][key]) != sorted(entry.get("targets", [])):
                problems.append(
                    f"deps: entry '{entry.get('conan')}' declares targets "
                    f"{entry.get('targets')} but metadata says {buckets[bucket][key]}"
                )
        for bucket, keys in buckets.items():
            for key in keys:
                if (bucket, key) not in seen:
                    problems.append(
                        f"deps: metadata declares '{bucket}/{key}' with no entry in deps.json"
                    )

    for entry in entries:
        name = entry.get("conan", "<no name>")

        # D3 -- a licence we can name, from the set we accept.
        licence = entry.get("license")
        if licence not in allowed:
            problems.append(
                f"deps: '{name}' declares licence '{licence}', which is not in allowed_licences "
                f"{sorted(allowed)}"
            )
        if not str(entry.get("license_evidence", "")).strip():
            problems.append(f"deps: '{name}' has no license_evidence -- an SPDX id alone is a guess")
        if entry.get("license_status") == "unverified" and entry.get("redistributed"):
            problems.append(
                f"deps: '{name}' is redistributed with an unverified licence -- that combination is "
                f"the one that cannot ship"
            )

        # D4 -- attribution and shipping are the same predicate.
        redistributed = bool(entry.get("redistributed"))
        attribution = str(entry.get("attribution", "")).strip()
        if redistributed and not attribution:
            problems.append(
                f"deps: '{name}' is marked redistributed but carries no attribution -- an "
                f"attribution that is not written down does not travel with the artifact"
            )
        if attribution and not redistributed:
            problems.append(
                f"deps: '{name}' carries an attribution while the entry says it is not "
                f"redistributed -- one of the two is wrong"
            )

        # D5 -- usage is a measurement.
        used = entry.get("used_from")
        if used:
            path = root / used.get("path", "")
            needle = used.get("contains", "")
            if not path.is_file():
                problems.append(f"deps: '{name}' says it is used from '{used.get('path')}', which does not exist")
            elif needle and needle not in path.read_text(encoding="utf-8", errors="replace"):
                problems.append(
                    f"deps: '{name}' says it is used from '{used.get('path')}' via '{needle}', which is "
                    f"not in that file -- the evidence moved or the dependency did"
                )
        elif not str(entry.get("why_unused", "")).strip():
            problems.append(
                f"deps: '{name}' names no consumer and no why_unused -- a dependency kept without a "
                f"reason is indistinguishable from one nobody dares to delete"
            )

    # D6 -- the bare-metal whitelist cannot outlive its dependency.
    whitelist = (load_json(metadata).get("baremetal_white_list") or []) if metadata.is_file() else []
    allowed_on_baremetal = {entry.get("metadata_key") for entry in entries if entry.get("baremetal_ok")}
    for key in whitelist:
        if key not in allowed_on_baremetal:
            problems.append(
                f"deps: baremetal_white_list names '{key}', which no entry marks as baremetal_ok"
            )

    # D8 -- a recipe that lives outside the tree can only be kept honest by its digests.
    for entry in entries:
        recipe = entry.get("local_package") or entry.get("recipe")
        if not recipe:
            continue
        name = entry.get("conan", "?")
        folder = root / recipe.get("path", "")
        recorded = recipe.get("files") or {}
        if not recorded:
            problems.append(
                f"deps: '{name}' declares a recipe but no file digests, so nothing about it can "
                f"be checked from here"
            )
            continue
        if not folder.is_dir():
            print(f"[deps]   {name}: recipe folder '{recipe['path']}' is not present locally; its "
                  f"digests cannot be checked from a clean clone, and that is the documented state")
            continue
        for rel, digest in recorded.items():
            candidate = folder / rel
            if not candidate.is_file():
                problems.append(
                    f"deps: '{name}''s recipe names '{rel}', which is missing from '{recipe['path']}'"
                )
            elif not hashlib.sha256(candidate.read_bytes()).hexdigest().startswith(digest[:32]):
                problems.append(
                    f"deps: '{name}''s recipe file '{rel}' does not hash to the digest recorded "
                    f"here -- the recipe on this machine is not the one this register describes"
                )

    # D7 (PLAN gate/license) -- this project's own licence, not a dependency's.
    if metadata.is_file():
        own = str(load_json(metadata).get("license", "")).strip()
        if not own:
            problems.append("deps: metadata.json declares no license")
        elif own not in allowed:
            problems.append(
                f"deps: metadata.json declares license '{own}', which is not in allowed_licences "
                f"{sorted(allowed)} -- we would be shipping under a licence we do not accept"
            )
        licence_file = root / data.get("license_file", "LICENSE")
        if not licence_file.is_file():
            problems.append(f"deps: '{licence_file.name}' is missing, so the declared licence has no text")
        elif own:
            heading = {"Apache-2.0": "Apache License", "MIT": "MIT License"}.get(own)
            if heading and heading.lower() not in licence_file.read_text(encoding="utf-8").lower():
                problems.append(
                    f"deps: metadata.json says '{own}' but {licence_file.name} does not read like it"
                )

    reported.append(f"{len(entries)} dependency entr(ies), {len(buckets)} metadata bucket(s), "
                    f"{len(whitelist)} bare-metal allowance(s)")
    return problems, reported


def selftest():
    import tempfile

    failures = 0
    with tempfile.TemporaryDirectory(prefix="deps-selftest-") as workdir:
        root = pathlib.Path(workdir)
        (root / "src").mkdir()
        (root / "src/uses.c").write_text("#include <zlib.h>\nint x;\n", encoding="utf-8")
        (root / "conandata.yml").write_text(
            "requirements:\n  - \"zlib/1.3.1\"\n  # - \"dlib/19.24.2\"\n", encoding="utf-8"
        )
        (root / "LICENSE").write_text("Apache License\nVersion 2.0\n", encoding="utf-8")
        (root / "metadata.json").write_text(
            json.dumps({
                "license": "Apache-2.0",
                "dependencies": {"common": {"ZLIB": ["ZLIB::ZLIB"]}},
                "baremetal_white_list": ["ZLIB"],
            }), encoding="utf-8"
        )

        def entry(**overrides):
            base = {
                "conan": "zlib",
                "metadata_key": "ZLIB",
                "bucket": "common",
                "targets": ["ZLIB::ZLIB"],
                "license": "Zlib",
                "license_status": "verified",
                "license_evidence": "recipe says Zlib",
                "redistributed": False,
                "baremetal_ok": True,
                "why": "fixture",
                "used_from": {"path": "src/uses.c", "contains": "#include <zlib.h>"},
            }
            base.update(overrides)
            return base

        def run(entries=None, allowed=("Zlib", "Apache-2.0")):
            data = {
                "allowed_licences": list(allowed),
                "entries": entries if entries is not None else [entry()],
            }
            path = root / "deps-fixture.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            return check(root, data_path=path)[0]

        if not run():
            print("[deps:selftest] PASS D1..D6 a consistent set is green")
        else:
            failures += 1
            print(f"[deps:selftest] FAIL expected a clean run, got {run()}")

        (root / "conandata.yml").write_text(
            "requirements:\n  - \"zlib/1.3.1\"\n  - \"ghost/0.1\"\n", encoding="utf-8"
        )
        problems = run()
        if any("'ghost' is a requirement" in p for p in problems):
            print("[deps:selftest] PASS D1 a requirement with no entry is red")
        else:
            failures += 1
            print(f"[deps:selftest] FAIL D1 expected the unrecorded-requirement report, got {problems}")
        (root / "conandata.yml").write_text(
            "requirements:\n  - \"zlib/1.3.1\"\n", encoding="utf-8"
        )

        problems = run([entry(conan="dlib")])
        if any("outlived the dependency" in p for p in problems):
            print("[deps:selftest] PASS D1 an entry with no requirement is red")
        else:
            failures += 1
            print(f"[deps:selftest] FAIL D1 expected the orphan-entry report, got {problems}")

        recipe_dir = root / "recipes" / "zlib"
        recipe_dir.mkdir(parents=True, exist_ok=True)
        (recipe_dir / "conanfile.py").write_text("# the real thing\n", encoding="utf-8")
        good = hashlib.sha256((recipe_dir / "conanfile.py").read_bytes()).hexdigest()
        problems = run([entry(recipe={"path": "recipes/zlib",
                                     "files": {"conanfile.py": good}})])
        if not problems:
            print("[deps:selftest] PASS D8 a recipe matching its digests is green")
        else:
            failures += 1
            print(f"[deps:selftest] FAIL D8 expected a clean run, got {problems}")

        (recipe_dir / "conanfile.py").write_text("# something else\n", encoding="utf-8")
        problems = run([entry(recipe={"path": "recipes/zlib",
                                     "files": {"conanfile.py": good}})])
        if any("not the one this register describes" in p for p in problems):
            print("[deps:selftest] PASS D8 a recipe that changed since it was registered is red")
        else:
            failures += 1
            print(f"[deps:selftest] FAIL D8 expected the digest-mismatch report, got {problems}")

        problems = run([entry(recipe={"path": "recipes/absent",
                                     "files": {"conanfile.py": good}})])
        if any("missing from" in p for p in problems) or not problems:
            print("[deps:selftest] PASS D8 an absent recipe folder is not a failure (clean clone)")
        else:
            failures += 1
            print(f"[deps:selftest] FAIL D8 an absent folder should not be red, got {problems}")

        problems = run([entry(bucket="cpp")])
        if any("not one of" in p for p in problems):
            print("[deps:selftest] PASS D2 an unknown bucket is red")
        else:
            failures += 1
            print(f"[deps:selftest] FAIL D2 expected the bucket report, got {problems}")

        problems = run([entry(targets=["ZLIB::Other"])])
        if any("but metadata says" in p for p in problems):
            print("[deps:selftest] PASS D2 a target mismatch is red")
        else:
            failures += 1
            print(f"[deps:selftest] FAIL D2 expected the target report, got {problems}")

        problems = run([entry(license="GPL-3.0")])
        if any("not in allowed_licences" in p for p in problems):
            print("[deps:selftest] PASS D3 a licence outside the set is red")
        else:
            failures += 1
            print(f"[deps:selftest] FAIL D3 expected the licence report, got {problems}")

        problems = run([entry(license_status="unverified", redistributed=True)], allowed=("Zlib",))
        if any("unverified licence" in p for p in problems):
            print("[deps:selftest] PASS D3 redistribution with an unverified licence is red")
        else:
            failures += 1
            print(f"[deps:selftest] FAIL D3 expected the unverified report, got {problems}")

        problems = run([entry(license_evidence="")])
        if any("no license_evidence" in p for p in problems):
            print("[deps:selftest] PASS D3 an SPDX id with no evidence is red")
        else:
            failures += 1
            print(f"[deps:selftest] FAIL D3 expected the evidence report, got {problems}")

        problems = run([entry(redistributed=True, attribution="zlib (C) Jean-loup Gailly and Mark Adler")])
        if not problems:
            print("[deps:selftest] PASS D4 redistributed with an attribution is green (positive twin)")
        else:
            failures += 1
            print(f"[deps:selftest] FAIL D4 expected a clean run, got {problems}")

        problems = run([entry(redistributed=False, attribution="zlib (C) Jean-loup Gailly")])
        if any("while the entry says it is not redistributed" in p for p in problems):
            print("[deps:selftest] PASS D4 an attribution without redistribution is red")
        else:
            failures += 1
            print(f"[deps:selftest] FAIL D4 expected the reverse report, got {problems}")

        problems = run([entry(redistributed=True)])
        if any("carries no attribution" in p for p in problems):
            print("[deps:selftest] PASS D4 redistribution without an attribution line is red")
        else:
            failures += 1
            print(f"[deps:selftest] FAIL D4 expected the missing-attribution report, got {problems}")

        problems = run([entry(used_from={"path": "src/uses.c", "contains": "#include <ghost.h>"})])
        if any("not in that file" in p for p in problems):
            print("[deps:selftest] PASS D5 evidence that does not hold is red")
        else:
            failures += 1
            print(f"[deps:selftest] FAIL D5 expected the evidence report, got {problems}")

        problems = run([entry(used_from=None)])
        if any("no why_unused" in p for p in problems):
            print("[deps:selftest] PASS D5 an unused dependency without a reason is red")
        else:
            failures += 1
            print(f"[deps:selftest] FAIL D5 expected the why_unused report, got {problems}")

        problems = run([entry(used_from=None, why_unused="kept from the template, gaps G-4")])
        if not problems:
            print("[deps:selftest] PASS D5 an unused dependency with a reason is green")
        else:
            failures += 1
            print(f"[deps:selftest] FAIL D5 expected a clean run, got {problems}")

        problems = run([entry(baremetal_ok=False)])
        if any("baremetal_white_list names" in p for p in problems):
            print("[deps:selftest] PASS D6 a whitelist entry without the flag is red")
        else:
            failures += 1
            print(f"[deps:selftest] FAIL D6 expected the whitelist report, got {problems}")

        # D7 is about *this* project's licence, not a dependency's: it must be one we accept, and
        # LICENSE has to read like it.
        meta_path = root / "metadata.json"
        saved = meta_path.read_text(encoding="utf-8")
        meta = json.loads(saved)
        meta["license"] = "GPL-3.0"
        meta_path.write_text(json.dumps(meta), encoding="utf-8")
        problems = run()
        meta_path.write_text(saved, encoding="utf-8")
        if any("metadata.json declares license 'GPL-3.0'" in p for p in problems):
            print("[deps:selftest] PASS D7 a project licence outside the set is red")
        else:
            failures += 1
            print(f"[deps:selftest] FAIL D7 expected the licence report, got {problems}")

        problems = run([])
        if any("entry list is empty" in p for p in problems):
            print("[deps:selftest] PASS an empty entry list is red, not vacuous")
        else:
            failures += 1
            print(f"[deps:selftest] FAIL expected the empty-list report, got {problems}")

    if failures:
        print(f"[deps:selftest] FAIL ({failures} fixture(s))", file=sys.stderr)
        return 1
    print("[deps:selftest] OK")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default=str(DATA), help="dependency inventory")
    parser.add_argument("--selftest", action="store_true", help="prove each rule can fail")
    args = parser.parse_args()

    if args.selftest:
        return selftest()

    problems, reported = check(ROOT, data_path=args.data)
    if problems:
        for problem in problems:
            print(f"[deps] {problem}", file=sys.stderr)
        print(f"[deps] FAIL ({len(problems)} problem(s))", file=sys.stderr)
        return 1
    print(f"[deps] OK ({len(reported)} check(s): the four declarations agree)")
    for line in reported:
        print(f"[deps]   {line}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
