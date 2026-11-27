#!/usr/bin/env python3
"""Every claim in the evidence index must still have its footing.

`claim-basis.json` is where the repository says "this was measured, here is how you check
it". Written by hand and read by nobody, it decays exactly like the `grad` field did (gaps
G-72): the numbers drift as the tree moves, and prose that nobody validates reads the same
whether it is true or false.

The rules, all derived from something else in the tree rather than from the file itself:

  C1 every claim carries the fields that make it checkable at all: id, claim, gate, commit,
     host, date
  C2 the commit exists in this repository **and is an ancestor of HEAD** -- a claim whose
     basis commit is unreachable has lost its foundation, and a sha that never existed is
     indistinguishable from one that was force-pushed away
  C3 the `gate` names a check that exists: a gate in `ci.sh`, or an id in
     `toolchains.json:heavy_checks` for the ones that need hardware/qemu and are declared as
     not-in-CI on purpose
  C4 ids are unique
  C5 `local_only: true` requires a `reproduce` line: "only reproducible on my machine" is
     acceptable when it is written down, not when it is implied
  C6 declared `cells`/`builds` counts must equal what `src/kernel/matrix.yml` says today --
     this is the rule that catches a claim going stale when cells are added, which is exactly
     what happened between the int8 rounds and this gate's first run
  C7 the index is not empty: an evidence file with no evidence is not evidence

Usage:

    python3 .github/project/claim_gate.py             # check
    python3 .github/project/claim_gate.py --selftest  # prove each rule can fail
"""

import argparse
import importlib.util
import json
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA = ROOT / ".github/project/claim-basis.json"
CI = ROOT / ".github/project/ci.sh"
TOOLCHAINS = ROOT / ".github/project/toolchains.json"
MATRIX = ROOT / "src/kernel/matrix.yml"
EMITTER = ROOT / ".github/project/qa/prep/emit_k3_cells.py"
REQUIRED = ("id", "claim", "gate", "commit", "host", "date")


def load_data(path):
    data = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    if "claims" not in data:
        raise SystemExit("claim: the data file has no 'claims' list")
    return data


def git(root, *args):
    return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)


def known_checks(root, ci_path=None, toolchains_path=None):
    """Names a claim may cite: gate files `ci.sh` runs, plus declared heavy checks."""
    names = set()
    ci = pathlib.Path(ci_path) if ci_path is not None else root / ".github/project/ci.sh"
    if ci.is_file():
        names.update(re.findall(r"([a-z_]+)_gate\.py", ci.read_text(encoding="utf-8")))
    toolchains = pathlib.Path(toolchains_path) if toolchains_path is not None else (
        root / ".github/project/toolchains.json")
    if toolchains.is_file():
        declared = json.loads(toolchains.read_text(encoding="utf-8"))
        for check in declared.get("heavy_checks", []):
            if isinstance(check, dict) and check.get("id"):
                names.add(check["id"])
    return names


def matrix_cells(root, emitter_path=None):
    """Per-tier `done` cell counts as the k3 chain derives them, or None if unavailable.

    Imported from the emitter instead of re-implemented: a second copy of the rule would
    drift from the chain that actually runs, which is the failure mode this gate exists for.
    """
    path = pathlib.Path(emitter_path) if emitter_path is not None else root / ".github/project/qa/prep/emit_k3_cells.py"
    if not path.is_file():
        return None
    spec = importlib.util.spec_from_file_location("emit_k3_cells", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    matrix = module.load_yaml(root / "src/kernel/matrix.yml")
    return {tier: len(module.collect(matrix, tier)) for tier in ("arm_neon", "arm_m", "rvv")}


def check(root, data, ci_path=None, toolchains_path=None, emitter_path=None):
    problems = []
    reported = []
    claims = data.get("claims", [])

    # C7 -- an empty index is not evidence.
    if not claims:
        problems.append("claim: the index holds no claim at all -- an empty evidence file is not evidence")

    checks = known_checks(root, ci_path, toolchains_path)
    if not checks:
        problems.append("claim: could not read any gate name; C3 would pass vacuously")
    derived = matrix_cells(root, emitter_path)

    seen = set()
    for claim in claims:
        name = claim.get("id", "<no id>")
        where = f"claim '{name}'"

        # C1 -- the fields that make it checkable.
        missing = [field for field in REQUIRED if not str(claim.get(field, "")).strip()]
        if missing:
            problems.append(f"claim: {where} is missing {', '.join(missing)}")
            continue

        # C4 -- one number, one claim.
        if name in seen:
            problems.append(f"claim: {where} appears twice; one of them will be read as the other")
        seen.add(name)

        # C2 -- the basis commit has to be real and still reachable.
        commit = str(claim["commit"])
        if git(root, "cat-file", "-e", f"{commit}^{{commit}}").returncode != 0:
            problems.append(
                f"claim: {where} cites commit '{commit}', which does not exist in this repository -- "
                f"a sha that resolves to nothing is not a basis"
            )
        elif git(root, "merge-base", "--is-ancestor", commit, "HEAD").returncode != 0:
            problems.append(
                f"claim: {where} cites commit '{commit}', which is not an ancestor of HEAD -- the "
                f"evidence was recorded on a line of history this tree no longer contains"
            )

        # C3 -- the claimed gate has to exist, or the claim points at nothing.
        if checks and claim["gate"] not in checks:
            problems.append(
                f"claim: {where} says it is checked by '{claim['gate']}', which is neither a gate in "
                f"ci.sh nor an id in toolchains.json:heavy_checks"
            )

        # C5 -- local-only is allowed, silently local-only is not.
        if claim.get("local_only") and not str(claim.get("reproduce", "")).strip():
            problems.append(f"claim: {where} is local_only without a reproduce line -- that is unverifiable")

        # C6 -- the numbers must match the matrix this tree has today.
        declared = claim.get("cells")
        if isinstance(declared, dict):
            if derived is None:
                reported.append(f"{name}: cell counts not compared (the k3 emitter is absent)")
            else:
                for tier, count in declared.items():
                    if tier.startswith("_"):
                        continue
                    actual = derived.get(tier)
                    if actual != count:
                        problems.append(
                            f"claim: {where} declares {count} cell(s) for {tier} but the matrix has "
                            f"{actual} -- the claim went stale when the matrix moved"
                        )
                builds = claim.get("builds", {})
                total = declared.get("_cell_comparisons")
                if isinstance(total, int) and builds:
                    computed = sum(
                        count * builds[tier] for tier, count in declared.items() if not tier.startswith("_")
                    )
                    if computed != total:
                        problems.append(
                            f"claim: {where} declares {total} cell comparisons but "
                            f"sum(cells x builds) is {computed}"
                        )
        reported.append(f"{name}: fields, commit, gate and counts checked")

    return problems, reported


def selftest():
    import tempfile

    failures = 0
    with tempfile.TemporaryDirectory(prefix="claim-selftest-") as workdir:
        root = pathlib.Path(workdir)
        subprocess.run(["git", "init", "-q", str(root)], check=True)
        subprocess.run(["git", "-C", str(root), "config", "user.email", "t@t"], check=True)
        subprocess.run(["git", "-C", str(root), "config", "user.name", "t"], check=True)
        (root / "src/kernel").mkdir(parents=True)
        # The emitter lives under `.github/project/qa/prep` now, which creates `.github/project`
        # on the way -- so that parent has to accept already existing.
        (root / ".github/project/qa/prep").mkdir(parents=True)
        (root / ".github/project").mkdir(parents=True, exist_ok=True)
        (root / "src/kernel/matrix.yml").write_text(
            "kernels:\n"
            "  add:\n"
            "    dtype_quant: [[fp32, none]]\n"
            "    tiers: { scalar: done, arm_m: done, arm_neon: na, rvv: na }\n"
            "  relu:\n"
            "    dtype_quant: [[fp32, none]]\n"
            "    tiers: { scalar: done, arm_m: done, arm_neon: na, rvv: na }\n",
            encoding="utf-8",
        )
        (root / ".github/project/ci.sh").write_text(
            "for gate in census_gate.py; do python3 \"$gate\"; done\n", encoding="utf-8"
        )
        (root / ".github/project/toolchains.json").write_text(
            json.dumps({"heavy_checks": [{"id": "tier-parity"}]}), encoding="utf-8"
        )
        (root / ".github/project/qa/prep/emit_k3_cells.py").write_text(
            "import pathlib\n"
            "def load_yaml(path):\n"
            "    import yaml\n"
            "    return yaml.safe_load(pathlib.Path(path).read_text(encoding='utf-8'))\n"
            "MODES = {'add': ('BINARY', 'BITWISE'), 'relu': ('UNARY', 'BITWISE')}\n"
            "SUPPORTED = {('fp32', 'none')}\n"
            "def status_of(spec, dtype, quant):\n"
            "    if isinstance(spec, str):\n"
            "        return spec\n"
            "    if isinstance(spec, dict):\n"
            "        return spec.get(f'{dtype}/{quant}')\n"
            "    return None\n"
            "def collect(matrix, tier):\n"
            "    out = []\n"
            "    for name, entry in matrix['kernels'].items():\n"
            "        if name not in MODES:\n"
            "            continue\n"
            "        for dtype, quant in entry['dtype_quant']:\n"
            "            if (dtype, quant) not in SUPPORTED:\n"
            "                continue\n"
            "            if status_of(entry['tiers'].get(tier), dtype, quant) != 'done':\n"
            "                continue\n"
            "            out.append(name)\n"
            "    return out\n",
            encoding="utf-8",
        )
        subprocess.run(["git", "-C", str(root), "add", "-A"], check=True)
        subprocess.run(["git", "-C", str(root), "commit", "-qm", "fixture"], check=True)
        head = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--short", "HEAD"], capture_output=True, text=True
        ).stdout.strip()

        ci = root / ".github/project/ci.sh"
        toolchains = root / ".github/project/toolchains.json"
        emitter = root / ".github/project/qa/prep/emit_k3_cells.py"

        def run(claims):
            data = {"schema": 1, "claims": claims}
            return check(root, data, ci_path=ci, toolchains_path=toolchains, emitter_path=emitter)[0]

        def claim(**overrides):
            base = {
                "id": "c1",
                "claim": "something measured",
                "gate": "census",
                "commit": head,
                "host": "fixture",
                "date": "2026-11-26",
                "local_only": True,
                "reproduce": "python3 something.py",
            }
            base.update(overrides)
            return base

        if not run([claim()]):
            print("[claim:selftest] PASS C1..C6 a complete claim is green")
        else:
            failures += 1
            print(f"[claim:selftest] FAIL expected a clean run, got {run([claim()])}")

        problems = run([claim(date="")])
        if any("is missing date" in p for p in problems):
            print("[claim:selftest] PASS C1 a claim without a date is red")
        else:
            failures += 1
            print(f"[claim:selftest] FAIL C1 expected the missing-field report, got {problems}")

        problems = run([claim(commit="deadbee")])
        if any("does not exist in this repository" in p for p in problems):
            print("[claim:selftest] PASS C2 an unreachable commit is red")
        else:
            failures += 1
            print(f"[claim:selftest] FAIL C2 expected the missing-commit report, got {problems}")

        # A commit that exists but is not an ancestor: build one on a side branch.
        subprocess.run(["git", "-C", str(root), "checkout", "-q", "-b", "side"], check=True)
        (root / "side.txt").write_text("x\n", encoding="utf-8")
        subprocess.run(["git", "-C", str(root), "add", "-A"], check=True)
        subprocess.run(["git", "-C", str(root), "commit", "-qm", "side"], check=True)
        side = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--short", "HEAD"], capture_output=True, text=True
        ).stdout.strip()
        subprocess.run(["git", "-C", str(root), "checkout", "-q", "-"], check=True)
        problems = run([claim(commit=side)])
        if any("not an ancestor of HEAD" in p for p in problems):
            print("[claim:selftest] PASS C2 a commit outside this history is red")
        else:
            failures += 1
            print(f"[claim:selftest] FAIL C2 expected the ancestor report, got {problems}")

        problems = run([claim(gate="no-such-check")])
        if any("neither a gate in ci.sh" in p for p in problems):
            print("[claim:selftest] PASS C3 an unknown gate name is red")
        else:
            failures += 1
            print(f"[claim:selftest] FAIL C3 expected the unknown-gate report, got {problems}")

        problems = run([claim(gate="tier-parity")])
        if not problems:
            print("[claim:selftest] PASS C3 a declared heavy check is accepted")
        else:
            failures += 1
            print(f"[claim:selftest] FAIL C3 expected the heavy check to be accepted, got {problems}")

        problems = run([claim(), claim()])
        if any("appears twice" in p for p in problems):
            print("[claim:selftest] PASS C4 a duplicated id is red")
        else:
            failures += 1
            print(f"[claim:selftest] FAIL C4 expected the duplicate report, got {problems}")

        problems = run([claim(reproduce="")])
        if any("without a reproduce line" in p for p in problems):
            print("[claim:selftest] PASS C5 local_only without reproduce is red")
        else:
            failures += 1
            print(f"[claim:selftest] FAIL C5 expected the reproduce report, got {problems}")

        # C6: the fixture matrix has two `done` cells for arm_m, none for the other tiers.
        problems = run([claim(cells={"arm_m": 2, "_cell_comparisons": 2}, builds={"arm_m": 1})])
        if not problems:
            print("[claim:selftest] PASS C6 counts that match the matrix are green")
        else:
            failures += 1
            print(f"[claim:selftest] FAIL C6 expected a clean run, got {problems}")

        problems = run([claim(cells={"arm_m": 3, "_cell_comparisons": 3}, builds={"arm_m": 1})])
        if any("the claim went stale" in p for p in problems):
            print("[claim:selftest] PASS C6 a stale cell count is red")
        else:
            failures += 1
            print(f"[claim:selftest] FAIL C6 expected the stale-count report, got {problems}")

        problems = run([claim(cells={"arm_m": 2, "_cell_comparisons": 9}, builds={"arm_m": 1})])
        if any("cell comparisons but" in p for p in problems):
            print("[claim:selftest] PASS C6 a wrong comparison total is red")
        else:
            failures += 1
            print(f"[claim:selftest] FAIL C6 expected the comparison-total report, got {problems}")

        problems = run([])
        if any("holds no claim" in p for p in problems):
            print("[claim:selftest] PASS C7 an empty index is red")
        else:
            failures += 1
            print(f"[claim:selftest] FAIL C7 expected the empty-index report, got {problems}")

    if failures:
        print(f"[claim:selftest] FAIL ({failures} fixture(s))", file=sys.stderr)
        return 1
    print("[claim:selftest] OK")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default=str(DATA), help="evidence index")
    parser.add_argument("--selftest", action="store_true", help="prove each rule can fail")
    args = parser.parse_args()

    if args.selftest:
        return selftest()

    problems, reported = check(ROOT, load_data(args.data))
    if problems:
        for problem in problems:
            print(f"[claim] {problem}", file=sys.stderr)
        print(f"[claim] FAIL ({len(problems)} problem(s))", file=sys.stderr)
        return 1
    print(f"[claim] OK ({len(reported)} claim(s) still have their footing)")
    for line in reported:
        print(f"[claim]   {line}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
