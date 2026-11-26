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
  R2a a `tiers` entry is a status token (a kernel-level claim: it applies to every
     declared pair) or a mapping from `dtype/quant` to a status token (a claim
     about that pair only). The second form exists because a tier can be real for
     one pair and absent for another -- arm_m is a DSP dot product, so int8 has
     one and fp32 has no FPU on the Cortex-M profile (gaps G-48/G-51).
  R2b a per-pair mapping states exactly the kernel's declared pairs: a key that is
     not a declared pair is red (a typo must not silently become a different
     claim) and a declared pair left out is red (silence is not a status, or the
     difference between "we have it" and "we do not" would be an omission).
  R3 a kernel does not declare the same (dtype, quant) pair twice.
  R4 an ONNX op claimed by more than one row (kernel or composed) must carry
     `where` on every claiming entry (ONNX reuses `Conv` for conv and dwconv; see
     PLAN §3.7).
  R5 every cell whose tier is `done` has a census key with a test name; every
       census key is a matrix cell. Cells whose tier is `todo` are *pending*,
       not red -- a red cell has to mean "we claim this and cannot prove it", not
       "we have not written this tier yet" (that is PLAN P2.2's work list, and a
       permanently red gate stops being read). Cells whose tier is `na` are
       *decided*, not pending: `todo` is work nobody has done, `na` is a written
       decision that this tier is not a code tier for this kernel (for example
       `autovec`, which is the same bodies built with vectorising flags and is
       proven by `isa_gate.py` instead). Reporting both as pending would let a
       decision look like a backlog, and a backlog look like a decision.
  R6 a census entry is a non-empty list of non-empty test names; a declared cell
     with `[]` is "declared but unproven" and is red, not absent.
  R7 `composed` rows are operators the graph layer assembles from kernels: they
     declare tflite / onnx / uses / kat, every `uses` names an existing kernel,
     and no name appears in both sections (a convolution that claims to be a
     kernel body is exactly the defect this section exists to prevent).

**What this gate cannot know yet.** It does not run the named tests and does not
check that the name exists in some file; that cross-check belongs to the parity
harness (PLAN P1.1) once the first non-scalar tier lands. The gap is stated here
so nobody mistakes a green census for a green kernel.

Usage:

    python3 .github/project/census_gate.py                    # check
    python3 .github/project/census_gate.py --strict           # pending cells are red too
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
# The order mirrors infr_tier in include/infinfer/kernel/kernel.h, so a row and the enum
# read the same way. A tier token the header does not define is not a legal claim.
TIERS = ("scalar", "autovec", "x86", "arm_neon", "arm_m", "rvv")
STATUSES = ("todo", "done", "na")
GRADS = ("todo", "done", "no-grad")
KERNEL_FIELDS = ("tflite", "onnx", "dtype_quant", "tiers", "grad")
COMPOSED_FIELDS = ("tflite", "onnx", "uses", "kat")


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


def collect_onnx_claims(name, entry, claims, problems, section):
    """Record one row's ONNX claims so R4 can look at them across sections."""
    ops = entry.get("onnx")
    for op in ops if isinstance(ops, list) else []:
        if not isinstance(op, dict) or "op" not in op:
            problems.append(f"matrix: {section} '{name}' onnx entries must carry an op name")
            continue
        claims.setdefault(op["op"], []).append((name, op))


def check_onnx_claims(claims, problems):
    """R4: an ONNX op with more than one claimant needs `where` on every claim."""
    for op, claimants in sorted(claims.items()):
        if len(claimants) < 2:
            continue
        owners = ", ".join(name for name, _ in claimants)
        for name, entry in claimants:
            if not str(entry.get("where", "")).strip():
                problems.append(
                    f"matrix: ONNX op '{op}' is claimed by {owners}; row '{name}' must carry a non-empty 'where'"
                )


def check_composed(composed, kernel_names, claims, problems):
    """R7: composed operators must be assembled from kernels that exist."""
    if composed is None:
        return
    if not isinstance(composed, dict):
        problems.append("matrix: composed must be a mapping")
        return
    for name, entry in sorted(composed.items()):
        if name in kernel_names:
            problems.append(
                f"matrix: '{name}' is declared as both a kernel and a composed operator; "
                "only one of them can be the truth"
            )
        if not isinstance(entry, dict):
            problems.append(f"matrix: composed '{name}' must be a mapping")
            continue
        missing = [field for field in COMPOSED_FIELDS if field not in entry]
        if missing:
            problems.append(f"matrix: composed '{name}' is missing {', '.join(missing)}")
            continue
        uses = entry["uses"]
        if not isinstance(uses, list) or not uses:
            problems.append(f"matrix: composed '{name}' uses must be a non-empty list of kernel names")
        else:
            for used in uses:
                if used not in kernel_names:
                    problems.append(f"matrix: composed '{name}' uses '{used}', which is not a kernel in this file")
        if not str(entry["kat"]).strip():
            problems.append(f"matrix: composed '{name}' kat must be 'pending' or the KAT that judges it")
        collect_onnx_claims(name, entry, claims, problems, "composed")


def pair_key(dtype, quant):
    """The one spelling of a (dtype, quant) pair inside a `tiers` mapping."""
    return f"{dtype}/{quant}"


def resolve_tier(kernel, tier, spec, legal_pairs, problems):
    """R2a/R2b: turn one `tiers` entry into {pair_key: status}, or None when red.

    A status token is the original kernel-level claim and fans out to every
    declared pair. A mapping is a per-pair claim and has to name every declared
    pair, so the reader never has to guess whether an absent key means "todo" or
    was forgotten.
    """
    if isinstance(spec, str):
        if spec not in STATUSES:
            problems.append(f"matrix: kernel '{kernel}' tier '{tier}' has unknown status '{spec}'")
            return None
        return {pair_key(dtype, quant): spec for dtype, quant in legal_pairs}
    if not isinstance(spec, dict):
        problems.append(
            f"matrix: kernel '{kernel}' tier '{tier}' must be a status token or a (dtype, quant) mapping"
        )
        return None

    declared = {pair_key(dtype, quant) for dtype, quant in legal_pairs}
    resolved = {}
    for key, status in spec.items():
        if key not in declared:
            problems.append(
                f"matrix: kernel '{kernel}' tier '{tier}' names '{key}', which is not a declared pair of this kernel"
            )
            continue
        if status not in STATUSES:
            problems.append(
                f"matrix: kernel '{kernel}' tier '{tier}' names '{key}' with unknown status '{status}'"
            )
            continue
        resolved[key] = status

    missing = sorted(declared - set(resolved))
    if missing:
        problems.append(
            f"matrix: kernel '{kernel}' tier '{tier}' must state every declared (dtype, quant) pair; "
            f"missing {', '.join(missing)}"
        )
    return resolved if len(resolved) == len(declared) else None


def check_matrix(data):
    """R1-R4, R7.

    Returns (problems, cells, status_by_key, kernel_names): `cells` maps a kernel
    name to its declared (dtype, quant, tier) rows, and `status_by_key` maps a
    cell key to that tier's status so the census can tell "unproven claim" from
    "not built yet".
    """
    problems = []
    if not isinstance(data, dict):
        return ["matrix: root must be a mapping"], {}, {}, set()
    if data.get("schema") != 1:
        problems.append("matrix: schema must be 1")
    kernels = data.get("kernels")
    if not isinstance(kernels, dict) or not kernels:
        problems.append("matrix: kernels must be a non-empty mapping")
        return problems, {}, {}, set()

    claims = {}
    cells = {}
    status_by_key = {}
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
        resolved_tiers = {}
        for tier, spec in sorted(tiers.items()):
            if tier not in TIERS:
                problems.append(f"matrix: kernel '{name}' has unknown tier '{tier}'")
                continue
            resolved = resolve_tier(name, tier, spec, legal_pairs, problems)
            if resolved is not None:
                resolved_tiers[tier] = resolved
        if entry["grad"] not in GRADS:
            problems.append(f"matrix: kernel '{name}' has unknown grad '{entry['grad']}'")

        legal_tiers = [tier for tier in TIERS if tier in resolved_tiers]
        cells[name] = [(dtype, quant, tier) for (dtype, quant) in legal_pairs for tier in legal_tiers]
        for dtype, quant, tier in cells[name]:
            status_by_key[cell_key(name, dtype, quant, tier)] = resolved_tiers[tier][pair_key(dtype, quant)]

        collect_onnx_claims(name, entry, claims, problems, "kernel")

    check_onnx_claims(claims, problems)
    check_composed(data.get("composed"), set(kernels), claims, problems)
    check_onnx_claims(claims, problems)
    return problems, cells, status_by_key, set(kernels)


def cell_key(kernel, dtype, quant, tier):
    """The one spelling of a cell, shared by the matrix and the census."""
    return f"{kernel}/{dtype}/{quant}/{tier}"


def check_census(census, cells, status_by_key):
    """R5-R6: cross-check the census against the matrix; returns (problems, pending)."""
    problems = []
    pending = []
    if not isinstance(census, dict):
        return ["census: root must be a mapping"], pending
    if census.get("schema") != 1:
        problems.append("census: schema must be 1")
    entries = census.get("cells")
    if not isinstance(entries, dict):
        return problems + ["census: cells must be a mapping"], pending

    declared = {cell_key(name, dtype, quant, tier) for name, rows in cells.items() for dtype, quant, tier in rows}
    for key in sorted(declared - set(entries)):
        status = status_by_key.get(key)
        if status == "done":
            problems.append(f"census: cell '{key}' is declared in matrix.yml but missing from census.json")
        elif status == "na":
            # Decided, not pending: this tier is not a code tier for this kernel and the matrix
            # says so. Counting it as backlog would make a decision look like unfinished work.
            pass
        else:
            pending.append(key)
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
    return problems, pending


def run_check(matrix_path, census_path):
    """Full check; returns (problems, pending). Empty problems means green."""
    data = load_yaml(matrix_path)
    problems, cells, status_by_key, _ = check_matrix(data)
    if problems:
        return problems, []
    try:
        census = json.loads(Path(census_path).read_text(encoding="utf-8"))
    except OSError as exc:
        return [f"census: cannot read {census_path}: {exc}"], []
    except ValueError as exc:
        return [f"census: {census_path} is not valid JSON: {exc}"], []
    return check_census(census, cells, status_by_key)


def emit_census(matrix_path):
    """Print the census skeleton for every matrix cell (generator, not a verdict)."""
    data = load_yaml(matrix_path)
    problems, cells, _, _ = check_matrix(data)
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
    problems, cells, _, _ = check_matrix(data)
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
                "tiers": {"scalar": "done"},
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
            # The positive twin of the case above: 'x86' must be a legal token, otherwise the
            # matrix cannot express the tier the header already defines (INFR_TIER_X86), and a
            # tier nobody can declare reads exactly like a tier nobody has implemented.
            # `phrase=None` means "this fixture must produce no problem at all". The scalar claim
            # stays because `good_census` names its cell -- dropping it would raise an orphan-key
            # problem and the fixture would then be red for a reason that has nothing to do with
            # the token under test.
            "R2 x86 is a legal tier token",
            mutate_matrix(tiers={"scalar": "done", "x86": "todo"}),
            good_census,
            None,
        ),
        (
            "R2a per-pair status token",
            mutate_matrix(tiers={"scalar": {"fp32/none": "maybe"}}),
            good_census,
            "names 'fp32/none' with unknown status 'maybe'",
        ),
        (
            "R2a tier value shape",
            mutate_matrix(tiers={"scalar": ["done"]}),
            good_census,
            "tier 'scalar' must be a status token or a (dtype, quant) mapping",
        ),
        (
            "R2b per-pair tier names an undeclared pair",
            mutate_matrix(tiers={"scalar": {"int8/none": "done"}}),
            good_census,
            "tier 'scalar' names 'int8/none', which is not a declared pair of this kernel",
        ),
        (
            "R2b per-pair tier must state every declared pair",
            mutate_matrix(
                dtype_quant=[["fp32", "none"], ["int8", "tensor"]],
                tiers={"scalar": {"fp32/none": "done"}},
            ),
            good_census,
            "must state every declared (dtype, quant) pair; missing int8/tensor",
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
        (
            "R4 ambiguity spans kernels and composed",
            {
                "schema": 1,
                "kernels": {
                    "dwconv2d": {
                        "tflite": ["DEPTHWISE_CONV_2D"],
                        "onnx": [{"op": "Conv", "where": "group == in_channels"}],
                        "dtype_quant": [["fp32", "none"]],
                        "tiers": {"scalar": "todo"},
                        "grad": "no-grad",
                    }
                },
                "composed": {
                    "conv2d": {
                        "tflite": ["CONV_2D"],
                        "onnx": [{"op": "Conv"}],
                        "uses": ["dwconv2d"],
                        "kat": "pending",
                    }
                },
            },
            {"schema": 1, "cells": {}},
            "row 'conv2d' must carry a non-empty 'where'",
        ),
        (
            "R7 uses an unknown kernel",
            {
                "schema": 1,
                "kernels": good_matrix["kernels"],
                "composed": {
                    "conv2d": {
                        "tflite": ["CONV_2D"],
                        "onnx": [{"op": "Conv"}],
                        "uses": ["imaginary_dot"],
                        "kat": "pending",
                    }
                },
            },
            good_census,
            "which is not a kernel in this file",
        ),
        (
            "R7 name in both sections",
            {
                "schema": 1,
                "kernels": good_matrix["kernels"],
                "composed": {
                    "relu": {
                        "tflite": ["RELU"],
                        "onnx": [{"op": "Relu", "where": "composed"}],
                        "uses": ["relu"],
                        "kat": "pending",
                    }
                },
            },
            good_census,
            "declared as both a kernel and a composed operator",
        ),
        (
            "R7 blank kat",
            {
                "schema": 1,
                "kernels": good_matrix["kernels"],
                "composed": {
                    "conv2d": {
                        "tflite": ["CONV_2D"],
                        "onnx": [{"op": "Conv"}],
                        "uses": ["relu"],
                        "kat": "  ",
                    }
                },
            },
            good_census,
            "kat must be 'pending' or the KAT that judges it",
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

        baseline, baseline_pending = run_check(write("good.yml", good_matrix), write("good.json", good_census))
        if baseline:
            failures += 1
            print(f"[census:selftest] FAIL the good fixture reported: {baseline}")
        elif baseline_pending:
            failures += 1
            print(f"[census:selftest] FAIL the good fixture has pending cells: {baseline_pending}")
        else:
            print("[census:selftest] PASS the good fixture is green")

        # A `todo` tier with no census entry is the backlog, not a lie: pending.
        pending_matrix = mutate_matrix(tiers={"scalar": "done", "autovec": "todo"})
        pending_problems, pending_keys = run_check(
            write("pending.yml", pending_matrix), write("pending.json", good_census)
        )
        if pending_problems:
            failures += 1
            print(f"[census:selftest] FAIL an unbuilt tier should be pending, got {pending_problems}")
        elif pending_keys != ["relu/fp32/none/autovec"]:
            failures += 1
            print(f"[census:selftest] FAIL expected exactly the autovec cell to be pending, got {pending_keys}")
        else:
            print("[census:selftest] PASS an unbuilt tier is pending, not red")

        # G-51: `tiers` has to be able to carry a claim per (dtype, quant), because
        # the same tier is real for one pair and absent for another -- arm_m is a DSP
        # dot product, int8 has one, fp32 has no FPU on this profile. Both halves are
        # proven, because the risk is not that the new form is too small but that the
        # old one quietly stops meaning "every pair":
        #   - the per-pair form lets fc/int8/tensor/arm_m be `done` while its fp32
        #     sibling stays pending (a wrong key cannot hide there either: R2b)
        #   - the plain form still fans out to every pair, so it cannot be used to
        #     dodge a cell that has no implementation.
        per_pair_matrix = {
            "schema": 1,
            "kernels": {
                "fc": {
                    "tflite": ["FULLY_CONNECTED"],
                    "onnx": [{"op": "Gemm"}],
                    "dtype_quant": [["fp32", "none"], ["int8", "tensor"]],
                    "tiers": {
                        "scalar": "done",
                        "arm_m": {"fp32/none": "todo", "int8/tensor": "done"},
                    },
                    "grad": "todo",
                }
            },
        }
        per_pair_census = {
            "schema": 1,
            "cells": {
                "fc/fp32/none/scalar": ["kernel_parity/fc_scalar_fp32"],
                "fc/int8/tensor/scalar": ["kernel_parity/fc_int8_tensor"],
                "fc/int8/tensor/arm_m": ["kernel_parity/fc_arm_m_int8"],
            },
        }
        narrow_problems, narrow_pending = run_check(
            write("pair.yml", per_pair_matrix), write("pair.json", per_pair_census)
        )
        if narrow_problems:
            failures += 1
            print(f"[census:selftest] FAIL a per-pair tier should isolate the done cell, got {narrow_problems}")
        elif narrow_pending != ["fc/fp32/none/arm_m"]:
            failures += 1
            print(f"[census:selftest] FAIL expected only fc/fp32/none/arm_m pending, got {narrow_pending}")
        else:
            print("[census:selftest] PASS a per-pair tier claims one pair without claiming its sibling")

        widened = json.loads(json.dumps(per_pair_matrix))
        widened["kernels"]["fc"]["tiers"]["arm_m"] = "done"
        wide_problems, _ = run_check(write("wide.yml", widened), write("wide.json", per_pair_census))
        if [p for p in wide_problems if "fc/fp32/none/arm_m" in p and "missing from census.json" in p]:
            print("[census:selftest] PASS the plain form still fans out to every pair")
        else:
            failures += 1
            print(f"[census:selftest] FAIL a kernel-level 'done' must demand every pair, got {wide_problems}")

        for index, (label, matrix, census, phrase) in enumerate(fixtures):
            matrix_path = write(f"matrix-{index}.yml", matrix)
            census_path = write(f"census-{index}.json", census)
            problems, _ = run_check(matrix_path, census_path)
            if phrase is None:
                # A green fixture: the rule must accept it outright. Without one of these,
                # tightening a token list could reject a legal value and no fixture would say so.
                if problems:
                    failures += 1
                    print(f"[census:selftest] FAIL {label}: expected no problem, got {problems}")
                else:
                    print(f"[census:selftest] PASS {label}")
                continue
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
    parser.add_argument("--strict", action="store_true", help="treat pending (todo-tier) cells as problems")
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

    problems, pending = run_check(args.matrix, args.census)
    if args.strict:
        problems += [f"census: cell '{key}' is declared but not implemented (--strict)" for key in pending]
        pending = []
    if problems:
        for problem in problems:
            print(f"[census] {problem}", file=sys.stderr)
        print(f"[census] FAIL ({len(problems)} problem(s))", flush=True)
        return 1
    if pending:
        print(f"[census] OK (pending: {len(pending)} unbuilt cell(s), first = {pending[0]})", flush=True)
        return 0
    print("[census] OK", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
