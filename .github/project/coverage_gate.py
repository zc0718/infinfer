#!/usr/bin/env python3
"""Judge the coverage summary against the floors declared in `metadata.json`.

**Why this gate exists.** The floors are already enforced in CI -- inside
`full-test-automation.yml`, which is template-owned. That makes the policy real
in CI and unreachable everywhere else: a developer cannot answer "am I still
above the floor?" without pushing, and `preflight` cannot include a rule that
only exists as YAML. This file is the same rule, in the place a local run can
reach. The duplication is forced by the scaffold boundary and is recorded as debt
in `workspace/audit/gaps.md` (G-55) rather than hidden.

**The rule, mirrored from the workflow line for line.**

  C1 rates come from the counts (`lines.hit / lines.found`), never from the
     rounded `percent` field: a rounding difference is not a verdict. Same for a
     core rate, which is summed over `per_file` entries under `core_scopes`.
  C2 the floor set is `coverage.floors[<leg>]`, falling back to
     `coverage.default_floor`, where `<leg>` is the runner OS (`Linux`, `macOS`).
  C3 with no declared floor the rule degrades to "coverage is not zero" and says
     so, because a missing declaration is not a licence to pass silently.
  C4 a declared core scope that matches no instrumented file is a warning and core
     is not judged -- a scope nobody wrote code for is not a regression.
  C5 the artifact is a build product, so an absent summary is a **skip with a
     reason**, not a pass and not a failure: `conan create` is what writes it.

Usage:

    python3 .github/project/coverage_gate.py                     # judge the tree's artifact
    python3 .github/project/coverage_gate.py --selftest          # prove each rule can fail
    python3 .github/project/coverage_gate.py --summary X --metadata Y --leg Linux
"""

import argparse
import json
import platform
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SUMMARY = ROOT / "test_package" / "test" / "export" / "coverage" / "coverage_summary.json"
METADATA = ROOT / "metadata.json"

# RUNNER_OS names, so a local run and a CI run are judged on the same floor.
LEGS = {"Linux": "Linux", "Darwin": "macOS", "Windows": "Windows"}


def local_leg():
    return LEGS.get(platform.system(), platform.system())


def load_json(path, what):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except OSError as exc:
        raise SystemExit(f"coverage: cannot read the {what}: {exc}")
    except ValueError as exc:
        raise SystemExit(f"coverage: the {what} is not valid JSON: {exc}")


def rates(data, scopes):
    """C1: (rates, warnings) with every rate computed from counts."""
    warnings = []
    lines = data.get("lines") or {}
    hit = int(lines.get("hit", 0))
    found = int(lines.get("found", 0))
    all_rate = (100.0 * hit / found) if found else 0.0
    out = {"all": (all_rate, hit, found)}

    core_hit = 0
    core_found = 0
    for path, counts in (data.get("per_file") or {}).items():
        if any(path.startswith(scope) for scope in scopes):
            core_hit += int(counts.get("lines_hit", 0))
            core_found += int(counts.get("lines_found", 0))
    if scopes and core_found:
        out["core"] = (100.0 * core_hit / core_found, core_hit, core_found)
    elif scopes:
        # C4
        warnings.append("a core scope is declared but matched no instrumented file; core not judged")
    return out, warnings


def judge(summary_path, metadata_path, leg):
    """Returns (problems, warnings, report_lines). Empty problems means green."""
    coverage = (load_json(metadata_path, "metadata") or {}).get("coverage") or {}
    scopes = tuple(coverage.get("core_scopes") or ())
    floors = (coverage.get("floors") or {}).get(leg) or (coverage.get("default_floor") or {})
    data = load_json(summary_path, "coverage summary")

    computed, warnings = rates(data, scopes)
    report = []
    problems = []
    overall = computed["all"]

    if not floors:
        # C3
        warnings.append("metadata.json has no coverage floors; judging only that the report is not empty")
        if not overall[1]:
            problems.append(f"coverage: line coverage is 0 ({overall[1]} of {overall[2]} lines)")
        return problems, warnings, report

    for name in ("all", "core"):
        if name not in computed:
            continue
        rate, hit, found = computed[name]
        floor = float(floors.get(name, 0.0))
        ok = rate + 1e-9 >= floor
        report.append(f"  {'ok  ' if ok else 'FAIL'} {name:4s} {rate:6.2f}% ({hit}/{found}) floor {floor:.2f}%")
        if not ok:
            problems.append(f"coverage: {name} {rate:.2f}% is below the {leg} floor {floor:.2f}%")
    return problems, warnings, report


def selftest():
    """Prove every rule can fail, and that the three exemptions stay green."""
    metadata = {
        "coverage": {
            "core_scopes": ["src/runtime/"],
            "floors": {"Linux": {"core": 90.0, "all": 85.0}},
            "default_floor": {"core": 80.0, "all": 70.0},
        }
    }
    no_floors = {"coverage": {"core_scopes": ["src/runtime/"]}}
    per_file = {
        "src/runtime/conv2d.cpp": {"lines_hit": 95, "lines_found": 100},
        "src/kernel/scalar/fc.c": {"lines_hit": 50, "lines_found": 100},
    }

    def summary(lines_hit, lines_found, files=None, percent=0.0):
        return {
            "lines": {"hit": lines_hit, "found": lines_found, "percent": percent},
            "functions": {"hit": 0, "found": 0, "percent": 0.0},
            "per_file": per_file if files is None else files,
        }

    cases = [
        # 195/200 lines overall, 95/100 in core: above both floors.
        ("above both floors", metadata, summary(195, 200), "Linux", 0, None),
        # 170/200 overall (85%, passes) with 80/100 in core (80%, misses): the case
        # that exists because core is judged separately from all. The first version
        # of this fixture reused the shared per_file and therefore had 95% core --
        # it asserted the wrong thing and passed for the wrong reason.
        (
            "core below floor while all passes",
            metadata,
            summary(
                170,
                200,
                files={
                    "src/runtime/conv2d.cpp": {"lines_hit": 80, "lines_found": 100},
                    "src/kernel/scalar/fc.c": {"lines_hit": 90, "lines_found": 100},
                },
            ),
            "Linux",
            1,
            "core 80.00% is below the Linux floor 90.00%",
        ),
        # 160/200 = 80% overall.
        ("all below floor", metadata, summary(160, 200), "Linux", 1, "all 80.00% is below the Linux floor 85.00%"),
        # C1: the rounded percent lies, the counts decide.
        (
            "percent contradicts the counts",
            metadata,
            summary(160, 200, percent=99.9),
            "Linux",
            1,
            "all 80.00% is below the Linux floor 85.00%",
        ),
        # C2: an unlisted leg falls back to default_floor (70/80), so the same
        # numbers that fail on Linux pass on a runner nobody declared.
        ("default floor for an undeclared leg", metadata, summary(160, 200), "macOS", 0, None),
        # C3: no floors -> only "not zero" is judged.
        ("no floors, some coverage", no_floors, summary(1, 200), "Linux", 0, None),
        ("no floors, zero coverage", no_floors, summary(0, 200), "Linux", 1, "line coverage is 0"),
        # C4: a core scope with no instrumented file is a warning, not a failure.
        (
            "core scope matched nothing",
            metadata,
            summary(195, 200, files={"src/kernel/scalar/fc.c": {"lines_hit": 50, "lines_found": 100}}),
            "Linux",
            0,
            None,
        ),
    ]

    failures = 0
    with tempfile.TemporaryDirectory(prefix="coverage-selftest-") as workdir:
        base = Path(workdir)
        meta_path = base / "metadata.json"
        meta_path.write_text(json.dumps(metadata), encoding="utf-8")
        no_floors_path = base / "metadata-no-floors.json"
        no_floors_path.write_text(json.dumps(no_floors), encoding="utf-8")
        for index, (label, meta, data, leg, expected_problems, phrase) in enumerate(cases):
            summary_path = base / f"summary{index}.json"
            summary_path.write_text(json.dumps(data), encoding="utf-8")
            meta_used = meta_path if meta is metadata else no_floors_path
            problems, warnings, _report = judge(summary_path, meta_used, leg)
            if len(problems) != expected_problems:
                failures += 1
                print(f"[coverage:selftest] FAIL {label}: expected {expected_problems} problem(s), got {problems}")
                continue
            if phrase is not None and not any(phrase in problem for problem in problems):
                failures += 1
                print(f"[coverage:selftest] FAIL {label}: expected '{phrase}', got {problems}")
                continue
            if label == "core scope matched nothing" and not any("matched no instrumented file" in w for w in warnings):
                failures += 1
                print(f"[coverage:selftest] FAIL {label}: the warning is missing, got {warnings}")
                continue
            print(f"[coverage:selftest] PASS {label} -> {expected_problems} problem(s)")

    if failures:
        print(f"[coverage:selftest] FAIL ({failures} fixture(s))", file=sys.stderr)
        return 1
    print("[coverage:selftest] OK")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--summary", default=str(SUMMARY), help="coverage_summary.json to judge")
    parser.add_argument("--metadata", default=str(METADATA), help="metadata.json holding the floors")
    parser.add_argument("--leg", default=local_leg(), help="runner OS whose floor applies (Linux / macOS)")
    parser.add_argument("--selftest", action="store_true", help="prove each rule can fail")
    args = parser.parse_args()

    if args.selftest:
        return selftest()

    if not Path(args.summary).is_file():
        print(f"[coverage] SKIP no summary at {args.summary}: it is a build product, written by "
              f"`conan create` (INV-6: a clean clone has none). CI judges the same floors inside "
              f"full-test-automation.yml.")
        return 0

    problems, warnings, report = judge(args.summary, args.metadata, args.leg)
    for warning in warnings:
        print(f"[coverage] note: {warning}")
    for line in report:
        print(f"[coverage] {line}")
    if problems:
        for problem in problems:
            print(f"[coverage] {problem}", file=sys.stderr)
        print(f"[coverage] FAIL ({len(problems)} problem(s))", flush=True)
        return 1
    if not report:
        print(f"[coverage] OK ({args.leg}: no floor judged, see the note above)", flush=True)
        return 0
    print(f"[coverage] OK ({args.leg})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
