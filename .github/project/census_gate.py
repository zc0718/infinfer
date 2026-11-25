#!/usr/bin/env python3
"""The kernel census: every legal (kernel, dtype, quant, tier) cell declared in
`src/kernel/matrix.yml` must name at least one test case in
`.github/project/census.json`.

**Why this gate exists.** A matrix that lists combinations is a claim; a census
that names the test pinning each cell turns the claim into something a machine
can check. The reference defect this repository was derived from had the opposite
property -- a platform table that said "supported" everywhere while no test named
anything -- which is why PLAN P1.2 makes the two files mutually accountable:

  R1 matrix.yml parses, declares `schema: 1`, and has a non-empty `kernels` map.
  R2 every kernel declares tflite / onnx / dtype_quant / tiers / grad, and every
     token comes from the closed sets at the top of this file.
  R3 a kernel does not declare the same (dtype, quant) pair twice.
  R4 an ONNX op claimed by more than one kernel must carry `where` on every
     claiming entry (ONNX reuses `Conv` for conv and dwconv; see PLAN §3.7).
  R5 every matrix cell has a census key, and every census key is a matrix cell.
  R6 a census entry is a non-empty list of non-empty test names; a declared cell
     with `[]` is "declared but unproven" and is red, not absent.

**What this gate cannot know yet.** It does not run the named tests and does not
check that the name exists in some file; that cross-check belongs to the parity
harness (PLAN P1.1) once the first tier lands. The gap is stated here so nobody
mistakes a green census for a green kernel.

Usage:

    python3 .github/project/census_gate.py                    # check
    python3 .github/project/census_gate.py --emit-census      # print the skeleton
    python3 .github/project/census_gate.py --emit-parity CELL # print a parity stub
    python3 .github/project/census_gate.py --selftest         # prove each rule fails
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MATRIX = ROOT / "src" / "kernel" / "matrix.yml"
CENSUS = Path(__file__).resolve().with_name("census.json")

DTYPES = ("fp32", "int8", "int16", "u8", "i32")
QUANTS = ("none", "tensor", "channel")
TIERS = ("scalar", "autovec", "arm_neon", "arm_m", "rvv")
STATUSES = ("todo", "done", "na")
GRADS = ("todo", "done", "no-grad")
KERNEL_FIELDS = ("tflite", "onnx", "dtype_quant", "tiers", "grad")


def load_yaml(path):
    """Parse a YAML file, turning a missing PyYAML into one actionable line."""
    try:
        import yaml
    except ImportError:
        raise SystemExit("census: PyYAML is required (`pip install pyyaml`)")
    try:
        return yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except OSError as exc:
        raise SystemExit(f"census: cannot read {path}: {exc}")
    except yaml.YAMLError as exc:
        raise SystemExit(f"census: {path} is not valid YAML: {exc}")


def check_matrix(data):
    """R1-R4: validate the matrix shape and return {kernel: [(dtype, quant, tier)]}."""
    problems = []
    if not isinstance(data, dict):
        return ["matrix: root must be a mapping"], {}
    if data.get("schema") != 1:
        problems.append("matrix: schema must be 1")
    kernels = data.get("kernels")
    if not isinstance(kernels, dict) or not kernels:
        problems.append("matrix: kernels must be a non-empty mapping")
        return problems, {}

    claims = {}
    cells = {}
    for name, entry in sorted(kernels.items()):
        if not isinstance(entry, dict):
            problems.append(f"matrix: kernel '{name}' must be a mapping")
            continue
        missing = [field for field in KERNEL_FIELDS if field not in entry]
        if missing:
            problems.append(f"matrix: kernel '{name}' is missing {', '.join(missing)}")
            continue

        pairs = entry["dtype_quant"]
        seen = set()
        legal_pairs = []
        for pair in pairs if isinstance(pairs, list) else []:
            if not isinstance(pair, list) or len(pair) != 2:
                problems.append(f"matrix: kernel '{name}' dtype_quant entries must be [dtype, quant]")
                continue
            dtype, quant = pair
            if dtype not in DTYPES:
                problems.append(f"matrix: kernel '{name}' has unknown dtype token '{dtype}'")
                continue
            if quant not in QUANTS:
                problems.append(f"matrix: kernel '{name}' has unknown quant token '{quant}'")
                continue
            if (dtype, quant) in seen:
                problems.append(f"matrix: kernel '{name}' declares ({dtype}, {quant}) twice")
                continue
            seen.add((dtype, quant))
            legal_pairs.append((dtype, quant))
        if not legal_pairs:
            problems.append(f"matrix: kernel '{name}' declares no legal dtype_quant pair")
            continue

        tiers = entry["tiers"]
        if not isinstance(tiers, dict) or not tiers:
            problems.append(f"matrix: kernel '{name}' tiers must be a non-empty mapping")
            continue
        for tier, status in sorted(tiers.items()):
            if tier not in TIERS:
                problems.append(f"matrix: kernel '{name}' has unknown tier '{tier}'")
            if status not in STATUSES:
                problems.append(f"matrix: kernel '{name}' tier '{tier}' has unknown status '{status}'")
        if entry["grad"] not in GRADS:
            problems.append(f"matrix: kernel '{name}' has unknown grad '{entry['grad']}'")

        legal_tiers = [tier for tier in tiers if tier in TIERS]
        cells[name] = [(dtype, quant, tier) for (dtype, quant) in legal_pairs for tier in legal_tiers]

        for op in entry["onnx"] if isinstance(entry["onnx"], list) else []:
            if not isinstance(op, dict) or "op" not in op:
                problems.append(f"matrix: kernel '{name}' onnx entries must carry an op name")
                continue
            claims.setdefault(op["op"], []).append((name, op))

    for op, claimants in sorted(claims.items()):
        if len(claimants) < 2:
            continue
        owners = ", ".join(name for name, _ in claimants)
        for name, entry in claimants:
            if not str(entry.get("where", "")).strip():
                problems.append(
                    f"matrix: ONNX op '{op}' is claimed by {owners}; kernel '{name}' must carry a non-empty 'where'"
                )
    return problems, cells


def cell_key(kernel, dtype, quant, tier):
    """The one spelling of a cell, shared by the matrix and the census."""
    return f"{kernel}/{dtype}/{quant}/{tier}"


def check_census(census, cells):
    """R5-R6: cross-check the census against the matrix."""
    problems = []
    if not isinstance(census, dict):
        return ["census: root must be a mapping"]
    if census.get("schema") != 1:
        problems.append("census: schema must be 1")
    entries = census.get("cells")
    if not isinstance(entries, dict):
        return problems + ["census: cells must be a mapping"]

    declared = {cell_key(name, dtype, quant, tier) for name, rows in cells.items() for dtype, quant, tier in rows}
    for key in sorted(declared - set(entries)):
        problems.append(f"census: cell '{key}' is declared in matrix.yml but missing from census.json")
    for key in sorted(set(entries) - declared):
        problems.append(f"census: key '{key}' is not declared in matrix.yml")

    for key, names in sorted(entries.items()):
        if not isinstance(names, list):
            problems.append(f"census: cell '{key}' entry must be a list of test names")
            continue
        if not names:
            problems.append(f"census: cell '{key}' is declared but has no test case")
            continue
        for index, name in enumerate(names):
            if not isinstance(name, str) or not name.strip():
                problems.append(f"census: cell '{key}' entry {index} is not a non-empty test name")
    return problems


def run_check(matrix_path, census_path):
    """Full check; returns the list of problems (empty means green)."""
    data = load_yaml(matrix_path)
    problems, cells = check_matrix(data)
    if problems:
        return problems
    try:
        census = json.loads(Path(census_path).read_text(encoding="utf-8"))
    except OSError as exc:
        return [f"census: cannot read {census_path}: {exc}"]
    except ValueError as exc:
        return [f"census: {census_path} is not valid JSON: {exc}"]
    return check_census(census, cells)


def emit_census(matrix_path):
    """Print the census skeleton for every matrix cell (generator, not a verdict)."""
    data = load_yaml(matrix_path)
    problems, cells = check_matrix(data)
    if problems:
        for problem in problems:
            print(problem, file=sys.stderr)
        return 1
    skeleton = {
        "schema": 1,
        "cells": {cell_key(name, dtype, quant, tier): [] for name, rows in cells.items() for dtype, quant, tier in rows},
    }
    print(json.dumps(skeleton, indent=2))
    return 0


def emit_parity(matrix_path, cell):
    """Print a GTest stub for one cell; the harness decides where the file lives."""
    data = load_yaml(matrix_path)
    problems, cells = check_matrix(data)
    if problems:
        for problem in problems:
            print(problem, file=sys.stderr)
        return 1
    keys = {cell_key(name, dtype, quant, tier) for name, rows in cells.items() for dtype, quant, tier in rows}
    if cell not in keys:
        print(f"census: '{cell}' is not a declared cell; run --emit-census for the list", file=sys.stderr)
        return 1
    test_name = cell.replace("/", "_").replace("-", "_")
    print("// Generated stub: replace the skip with the scalar-vs-tier comparison (PLAN P1.1).")
    print("#include <gtest/gtest.h>")
    print("")
    print(f"TEST(kernel_parity, {test_name}) {{")
    print(f'    GTEST_SKIP() << "census stub: {cell} has no parity test yet";')
    print("}")
    return 0


def selftest():
    """Prove each rule can fail, with a fixture that triggers only that rule."""
    good_matrix = {
        "schema": 1,
        "kernels": {
            "relu": {
                "tflite": ["RELU"],
                "onnx": [{"op": "Relu"}],
                "dtype_quant": [["fp32", "none"]],
                "tiers": {"scalar": "todo"},
                "grad": "todo",
            }
        },
    }
    good_census = {"schema": 1, "cells": {"relu/fp32/none/scalar": ["kernel_parity/relu_scalar_fp32"]}}

    def mutate_matrix(**changes):
        data = json.loads(json.dumps(good_matrix))
        data["kernels"]["relu"].update(changes)
        return data

    fixtures = [
        (
            "R1 schema",
            {"schema": 2, "kernels": good_matrix["kernels"]},
            good_census,
            "schema must be 1",
        ),
        (
            "R2 quant token",
            mutate_matrix(dtype_quant=[["fp32", "per_tensor"]]),
            good_census,
            "unknown quant token 'per_tensor'",
        ),
        (
            "R2 tier token",
            mutate_matrix(tiers={"avx512": "todo"}),
            {"schema": 1, "cells": {}},
            "unknown tier 'avx512'",
        ),
        (
            "R3 duplicate pair",
            mutate_matrix(dtype_quant=[["fp32", "none"], ["fp32", "none"]]),
            good_census,
            "declares (fp32, none) twice",
        ),
        (
            "R4 ambiguous onnx op",
            {
                "schema": 1,
                "kernels": {
                    "conv2d": {
                        "tflite": ["CONV_2D"],
                        "onnx": [{"op": "Conv"}],
                        "dtype_quant": [["fp32", "none"]],
                        "tiers": {"scalar": "todo"},
                        "grad": "no-grad",
                    },
                    "dwconv2d": {
                        "tflite": ["DEPTHWISE_CONV_2D"],
                        "onnx": [{"op": "Conv", "where": "group == in_channels"}],
                        "dtype_quant": [["fp32", "none"]],
                        "tiers": {"scalar": "todo"},
                        "grad": "no-grad",
                    },
                },
            },
            {"schema": 1, "cells": {}},
            "must carry a non-empty 'where'",
        ),
        (
            "R5 missing cell",
            good_matrix,
            {"schema": 1, "cells": {}},
            "is declared in matrix.yml but missing from census.json",
        ),
        (
            "R5 orphan key",
            good_matrix,
            {"schema": 1, "cells": {**good_census["cells"], "conv2d/fp32/none/scalar": ["x"]}},
            "is not declared in matrix.yml",
        ),
        (
            "R6 empty entry",
            good_matrix,
            {"schema": 1, "cells": {"relu/fp32/none/scalar": []}},
            "is declared but has no test case",
        ),
    ]

    failures = 0
    with tempfile.TemporaryDirectory(prefix="census-selftest-") as workdir:
        def write(name, payload):
            path = Path(workdir) / name
            if name.endswith(".yml"):
                try:
                    import yaml
                except ImportError:
                    raise SystemExit("census: PyYAML is required (`pip install pyyaml`)")
                path.write_text(yaml.safe_dump(payload), encoding="utf-8")
            else:
                path.write_text(json.dumps(payload), encoding="utf-8")
            return path

        baseline = run_check(write("good.yml", good_matrix), write("good.json", good_census))
        if baseline:
            failures += 1
            print(f"[census:selftest] FAIL the good fixture reported: {baseline}")
        else:
            print("[census:selftest] PASS the good fixture is green")

        for index, (label, matrix, census, phrase) in enumerate(fixtures):
            matrix_path = write(f"matrix-{index}.yml", matrix)
            census_path = write(f"census-{index}.json", census)
            problems = run_check(matrix_path, census_path)
            hit = [p for p in problems if phrase in p]
            if hit:
                print(f"[census:selftest] PASS {label} -> {hit[0]}")
            else:
                failures += 1
                print(f"[census:selftest] FAIL {label}: expected '{phrase}', got {problems}")

    if failures:
        print(f"[census:selftest] FAIL ({failures} fixture(s))", file=sys.stderr)
        return 1
    print("[census:selftest] OK")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--matrix", default=str(MATRIX), help="matrix.yml to read")
    parser.add_argument("--census", default=str(CENSUS), help="census.json to read")
    parser.add_argument("--emit-census", action="store_true", help="print the census skeleton for the matrix")
    parser.add_argument("--emit-parity", metavar="CELL", help="print a parity stub for one cell")
    parser.add_argument("--selftest", action="store_true", help="prove every rule can fail")
    args = parser.parse_args()

    if args.selftest:
        return selftest()
    if args.emit_census:
        return emit_census(args.matrix)
    if args.emit_parity:
        return emit_parity(args.matrix, args.emit_parity)

    problems = run_check(args.matrix, args.census)
    if problems:
        for problem in problems:
            print(f"[census] {problem}", file=sys.stderr)
        print(f"[census] FAIL ({len(problems)} problem(s))", flush=True)
        return 1
    print("[census] OK", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
