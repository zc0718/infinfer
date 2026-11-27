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
  R8 every test name in a census entry is a real GTest id in `test_package/test/`
     (`Suite/Name`, parsed from `TEST`/`TEST_F`/`TEST_P`). R5 only asked whether a
     cell had a name; this asks whether the name is real. When the rule was added
     (gaps G-70) 20 of the 36 entries named tests that did not exist: the cells
     were implemented, but the index pointed at nothing, so "which test proves
     this cell" had no answer -- and a rename could have produced that state with
     every other gate green.
  R9 `grad` is a claim with consequences: `no-grad` needs a written `grad_why`
     (otherwise it is indistinguishable from `todo`), `done` needs a `grad_tests`
     list naming tests that exist, and `grad_tests` on a non-`done` row is red
     (a test named for an unfinished claim is decoration).
  R13 the ONNX importer's vocabulary and the matrix's `onnx:` claims must name
     each other (gaps G-104): R13a a kernel claims an op the importer neither
     translates nor defers, R13b a deferral that outlives the work or carries no
     reason, R13c an importer target that the frozen blob tables do not have.
     The importer's tables are hand-written, so "what it accepts is what the
     matrix declares" is an assumption until something checks it.

**What this gate cannot know yet.** It checks that the named test is *written*, not that it is
*green* or that it really compares the tier against the reference -- that is what running the
suite (and the parity harness, PLAN P1.1) is for. The gap is stated here so nobody mistakes a
green census for a green kernel.

Usage:

    python3 .github/project/census_gate.py                    # check
    python3 .github/project/census_gate.py --strict           # pending cells are red too
    python3 .github/project/census_gate.py --emit-census      # print the skeleton
    python3 .github/project/census_gate.py --emit-parity CELL # print a parity stub
    python3 .github/project/census_gate.py --selftest         # prove each rule fails
"""

import argparse
import itertools
import json
import re
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


def check_grad(name, entry, problems):
    """R9: `grad` is a claim, so it has to cost something.

    The field existed from the start and only R2 looked at it -- a token from a closed set. A
    token is not a decision, and the two ways this field can lie are opposite:

      R9a `no-grad` is a decision that the derivative is not part of this kernel's job (argmax
          has no unique one; requantisation needs a straight-through estimator, which is a
          graph-layer choice). Without a written reason it reads exactly like "nobody thought
          about it yet", which is what `todo` already means.
      R9b `done` is a claim that a gradient exists, so it must name the test that pins it. No
          kernel claims `done` today, which is the point of adding the rule now: the first
          gradient kernel (PLAN P6) meets the rule before it can be claimed, not after.
      R9c the reverse -- naming gradient tests on a row that does not claim `done` -- is red too,
          because a test listed for an unfinished claim is decoration, not evidence.
    """
    grad = entry.get("grad")
    why = entry.get("grad_why")
    tests = entry.get("grad_tests")
    if "grad_why" in entry and not (isinstance(why, str) and why.strip()):
        problems.append(f"matrix: kernel '{name}' has a blank grad_why")
    if grad == "no-grad" and not (isinstance(why, str) and why.strip()):
        problems.append(
            f"matrix: kernel '{name}' claims grad: no-grad without a grad_why -- the difference "
            f"between 'not differentiable' and 'not considered' is the whole value of the token"
        )
    if grad == "done":
        if not isinstance(tests, list) or not [t for t in tests if isinstance(t, str) and t.strip()]:
            problems.append(f"matrix: kernel '{name}' claims grad: done without naming a grad test")
    elif tests is not None:
        problems.append(
            f"matrix: kernel '{name}' lists grad_tests while grad is '{grad}' -- a test named for "
            f"an unfinished claim is decoration"
        )


def check_grad_evidence(data, tests, problems):
    """R9b, second half: a named gradient test has to exist.

    Same reasoning as R8 -- a name is not evidence. Kept separate from `check_grad` because only
    this half needs the parsed test ids, and `check_matrix` is also called by the emitters, which
    must work without reading the test tree.
    """
    for name, entry in sorted((data.get("kernels") or {}).items()):
        if not isinstance(entry, dict) or entry.get("grad") != "done":
            continue
        for test in entry.get("grad_tests") or []:
            if isinstance(test, str) and test.strip() and test not in tests:
                problems.append(
                    f"matrix: kernel '{name}' names grad test '{test}', which does not exist in "
                    f"test_package/test/"
                )


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
        check_grad(name, entry, problems)

        legal_tiers = [tier for tier in TIERS if tier in resolved_tiers]
        cells[name] = [(dtype, quant, tier) for (dtype, quant) in legal_pairs for tier in legal_tiers]
        for dtype, quant, tier in cells[name]:
            status_by_key[cell_key(name, dtype, quant, tier)] = resolved_tiers[tier][pair_key(dtype, quant)]

        collect_onnx_claims(name, entry, claims, problems, "kernel")

    check_onnx_claims(claims, problems)
    check_composed(data.get("composed"), set(kernels), claims, problems)
    check_onnx_claims(claims, problems)
    check_scalar_reference(status_by_key, problems)
    check_tier_chain_coverage(data, status_by_key, problems)
    return problems, cells, status_by_key, set(kernels)


# The tiers `.github/project/qa/kat/run_kat.py` builds a chain for. A `done` cell on one of them
# is supposed to be compared against the scalar row **on the target**; a cell the cell
# generator cannot express is never compared anywhere but on the host.
TIER_CHAIN_TIERS = ("arm_neon", "arm_m", "rvv")


def check_tier_chain_coverage(data, status_by_key, problems):
    """R11: a `done` cell on a chain tier must be derivable by `emit_k3_cells`.

    Why this rule exists: the generator keeps a hand-written `MODES` map from kernel name
    to the call/comparison shape k3 uses. A kernel added to `matrix.yml` and marked `done`
    on `rvv` is therefore *silently* absent from the on-target chain unless somebody also
    edits that map -- and `claim_gate`'s count check cannot see it either, because both
    sides of that check derive from the same filtered list. The result would be an
    `arm_neon`/`rvv` cell whose only evidence is a host test, presented as if it had
    cross-target evidence. `mul` was added exactly that way (gaps G-94).
    """
    prep = ROOT / ".github" / "project" / "qa" / "prep"
    if not (prep / "emit_k3_cells.py").is_file():
        problems.append(
            "R11: .github/project/qa/prep/emit_k3_cells.py is missing, so no vector-tier cell can "
            "be checked against the chain that is supposed to prove it"
        )
        return
    if str(prep) not in sys.path:
        sys.path.insert(0, str(prep))
    import emit_k3_cells  # noqa: E402  (path is set up above, on purpose)

    for tier in TIER_CHAIN_TIERS:
        derived = {(name, dtype, quant) for name, dtype, quant, _call, _cmp in emit_k3_cells.collect(data, tier)}
        for key, status in sorted(status_by_key.items()):
            kernel, dtype, quant, cell_tier = key.split("/")
            if cell_tier != tier or status != "done":
                continue
            if (kernel, dtype, quant) not in derived:
                problems.append(
                    f"matrix: '{key}' is done but the k3 cell generator cannot express it -- a "
                    f"kernel reaches the on-target parity chain only through "
                    f"emit_k3_cells.MODES, so this cell would never be compared off the host"
                )


def check_scalar_reference(status_by_key, problems):
    """R10 (PLAN `gate/ports`): a tier cell is only a cell if the scalar reference covers it.

    The parity story of this repository is "compare the tier against the scalar row on the same
    target" -- k3 does exactly that. A `done` cell in a vector/DSP tier whose scalar counterpart is
    not `done` has nothing to be compared with: the claim would rest on a reference that does not
    exist, and the chain would either skip it silently or fail for the wrong reason.
    """
    for key, status in sorted(status_by_key.items()):
        kernel, dtype, quant, tier = key.split("/")
        if tier == "scalar" or status != "done":
            continue
        reference = status_by_key.get(cell_key(kernel, dtype, quant, "scalar"))
        if reference != "done":
            problems.append(
                f"matrix: '{key}' is done but the scalar reference is '{reference}' -- a tier cell "
                f"with no reference has nothing to be parity-checked against"
            )


def check_blob_activation_table(problems, header_text=None, act_map=None):
    """R12: the blob format's frozen activation table must mirror `kernel.h`.

    `.github/project/qa/prep/blobfmt.py` freezes the activation codes on purpose: a blob already
    in the tree must keep its meaning even if somebody renumbers the enum. The price of
    freezing is a second copy, and a second copy drifts. It did: `mul` landed with
    hard-sigmoid in the header while the blob table still stopped at hard-swish, so the
    first generated MobileNetV3 blob died on `KeyError: 'hard_sigmoid'` (gaps G-96). The
    check is two-way -- a code the header declares and the table lacks, an extra code the
    table invents, and a code both have at different numbers are all red.
    """
    if header_text is None or act_map is None:
        prep = ROOT / ".github" / "project" / "qa" / "prep"
        header = ROOT / "include" / "infinfer" / "kernel" / "kernel.h"
        if not (prep / "blobfmt.py").is_file() or not header.is_file():
            problems.append(
                "R12: .github/project/qa/prep/blobfmt.py or include/infinfer/kernel/kernel.h is "
                "missing, so the blob's activation table cannot be checked against the header"
            )
            return
        if str(prep) not in sys.path:
            sys.path.insert(0, str(prep))
        import blobfmt  # noqa: E402  (path is set up above, on purpose)

        header_text = header.read_text(encoding="utf-8")
        act_map = blobfmt.ACT

    declared = {}
    for name, code in re.findall(r"INFR_ACT_([A-Z0-9_]+)\s*=\s*(\d+)", header_text):
        if name == "COUNT":
            continue
        declared[name.lower()] = int(code)
    if not declared:
        problems.append(
            "R12: no INFR_ACT_<NAME> = <n> rows found in the header, so this rule would pass "
            "by scanning nothing"
        )
        return
    if declared != dict(act_map):
        missing = sorted(set(declared) - set(act_map))
        extra = sorted(set(act_map) - set(declared))
        renumbered = sorted(name for name in set(declared) & set(act_map) if declared[name] != act_map[name])
        problems.append(
            f"R12: blobfmt.ACT and kernel.h disagree (missing {missing}, extra {extra}, "
            f"renumbered {renumbered}) -- the blob codes are frozen, but frozen does not mean "
            f"allowed to lag: a code the format does not know is a blob nobody can emit"
        )


def matrix_onnx_ops(data):
    """Every ONNX op the matrix claims, as `{op: [kernel, ...]}`."""
    ops = {}
    for name, entry in sorted((data.get("kernels") or {}).items()):
        for item in entry.get("onnx") or []:
            op = item.get("op") if isinstance(item, dict) else None
            if op:
                ops.setdefault(op, []).append(name)
    return ops


def importer_tables():
    """The importer's vocabulary as plain data, so a fixture can stand in for it."""
    prep = ROOT / ".github" / "project" / "qa" / "prep"
    if str(prep) not in sys.path:
        sys.path.insert(0, str(prep))
    import blobfmt  # noqa: E402  (path is set up above, on purpose)
    import onnx_subset  # noqa: E402

    return {
        "translated": (set(onnx_subset.NODE_OPS) | set(onnx_subset.PASSTHROUGH)
                       | set(onnx_subset.ACTIVATIONS)),
        "deferred": dict(onnx_subset.DEFERRED),
        "node_targets": set(onnx_subset.NODE_OPS.values()),
        "act_targets": set(onnx_subset.ACTIVATIONS.values()),
        "op_codes": set(blobfmt.OP),
        "act_codes": set(blobfmt.ACT),
    }


def check_importer_vocabulary(problems, data=None, tables=None):
    """R13: the importer's vocabulary and the matrix's `onnx:` claims must name each other.

    The importer's tables are hand-written, so "what the importer accepts is what the matrix
    declares" was an assumption nobody checked (gaps G-104). Three failures are red: a kernel
    claiming an op the importer neither translates nor defers; a deferral that outlives the
    work (or carries no reason, which is the same disease as a debt entry without an exit);
    and an importer target that the frozen blob tables do not contain -- that last one would
    translate a model into a blob nobody can execute, and it is the only one of the three that
    would fail at run time instead of at import time.
    """
    if data is None or tables is None:
        prep = ROOT / ".github" / "project" / "qa" / "prep"
        if not (prep / "onnx_subset.py").is_file() or not MATRIX.is_file():
            problems.append(
                "R13: .github/project/qa/prep/onnx_subset.py or src/kernel/matrix.yml is "
                "missing, so the importer's vocabulary cannot be checked against the matrix"
            )
            return
        data = load_yaml(MATRIX)
        tables = importer_tables()

    claims = matrix_onnx_ops(data)
    if not claims:
        problems.append(
            "R13: the matrix claims no ONNX op at all, so this rule would pass by scanning "
            "nothing -- R4 has the same shape of failure and is red for the same reason"
        )
        return

    translated = tables["translated"]
    deferred = tables["deferred"]
    for op, kernels in sorted(claims.items()):
        if op not in translated and op not in deferred:
            problems.append(
                f"R13a: {', '.join(kernels)} claims ONNX '{op}', which the importer neither "
                f"translates nor defers -- the matrix says we serve it and the only path from "
                f"a model into this library cannot; translate it or say why in "
                f"onnx_subset.DEFERRED"
            )
    for op in sorted(deferred):
        if op in translated:
            problems.append(
                f"R13b: onnx_subset.DEFERRED lists '{op}', which the importer translates -- a "
                f"backlog entry that outlives the work reads as current while it is not"
            )
        if not str(deferred[op]).strip():
            problems.append(
                f"R13b: onnx_subset.DEFERRED['{op}'] has no reason, which is how a deferral "
                f"becomes folklore"
            )
    for target in sorted(tables["node_targets"] - tables["op_codes"]):
        problems.append(
            f"R13c: the importer maps a node to op '{target}', which is not in blobfmt.OP -- "
            f"the translation would emit a blob no executor can read"
        )
    for target in sorted(tables["act_targets"] - tables["act_codes"]):
        problems.append(
            f"R13c: the importer maps an activation to '{target}', which is not in blobfmt.ACT"
        )


def cell_key(kernel, dtype, quant, tier):
    """The one spelling of a cell, shared by the matrix and the census."""
    return f"{kernel}/{dtype}/{quant}/{tier}"

def collect_tests(root):
    """Map every GTest id in `test_package/test/` to the file that defines it.

    R5 asks whether a `done` cell has a *name*; this is what asks whether the name is *real*. The
    difference is not academic: when this rule was written, 20 of the 36 census entries named
    tests that did not exist anywhere in the tree -- the cells were honestly implemented, but the
    index pointed at nothing, so "which test proves this cell" had no answer.

    `TEST_F`/`TEST_P` are included because a future cell may be proven by a fixture; the id is the
    same `Suite/Name` shape in all three forms.
    """
    found = {}
    pattern = re.compile(r"\bTEST(?:_F|_P)?\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*,\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)")
    tests_root = Path(root) / "test_package" / "test"
    if not tests_root.is_dir():
        return found
    for path in sorted(itertools.chain(tests_root.rglob("*.cpp"), tests_root.rglob("*.cc"))):
        for suite, name in pattern.findall(path.read_text(encoding="utf-8", errors="replace")):
            found[f"{suite}/{name}"] = str(path.relative_to(Path(root)))
    return found


def check_census(census, cells, status_by_key, tests):
    """R5-R6-R8: cross-check the census against the matrix and against the test sources.

    `tests` is required rather than optional: an optional argument would let a caller skip R8 by
    forgetting it, which is exactly the "silently green" shape this gate exists to avoid.
    """
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
                continue
            if name not in tests:
                problems.append(
                    f"census: cell '{key}' names test '{name}', which does not exist in "
                    f"test_package/test/ -- the cell may be implemented, but the index points at "
                    f"nothing, so 'which test proves this cell' has no answer"
                )
    return problems, pending


def run_check(matrix_path, census_path, tests=None):
    """Full check; returns (problems, pending). Empty problems means green."""
    data = load_yaml(matrix_path)
    problems, cells, status_by_key, _ = check_matrix(data)
    if problems:
        return problems, []
    check_blob_activation_table(problems)
    if problems:
        return problems, []
    check_importer_vocabulary(problems, data)
    if problems:
        return problems, []
    if tests is None:
        # Production callers read the tree. The self-test passes a fixture set, and passing
        # `tests` explicitly is the only way to run the rules without one.
        tests = collect_tests(ROOT)
    check_grad_evidence(data, tests, problems)
    if problems:
        return problems, []
    try:
        census = json.loads(Path(census_path).read_text(encoding="utf-8"))
    except OSError as exc:
        return [f"census: cannot read {census_path}: {exc}"], []
    except ValueError as exc:
        return [f"census: {census_path} is not valid JSON: {exc}"], []
    return check_census(census, cells, status_by_key, tests)


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

    # R8 needs a test-id set. The fixture uses its own, so the rules can be exercised without
    # depending on which tests happen to exist in the tree today -- but it is passed explicitly
    # (never defaulted) so a caller cannot skip R8 by forgetting it.
    fixture_tests = {
        "kernel_parity/relu_scalar_fp32": "fixture/relu.cpp",
        "kernel_parity/fc_scalar_fp32": "fixture/fc.cpp",
        "kernel_parity/fc_int8_tensor": "fixture/fc_int8.cpp",
        "kernel_parity/fc_arm_m_int8": "fixture/fc_arm_m.cpp",
        "kernel_parity/relu_x86_fp32": "fixture/relu_x86.cpp",
    }

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
            # R11: a `done` cell on a chain tier that the k3 generator cannot express has no
            # cross-target evidence, and nothing else in the tree notices.
            "R11 a done vector-tier cell the chain cannot derive",
            {
                "schema": 1,
                "kernels": {
                    "cos": {
                        "tflite": ["COS"],
                        "onnx": [{"op": "Cos"}],
                        "dtype_quant": [["fp32", "none"]],
                        "tiers": {"scalar": "done", "rvv": "done"},
                        "grad": "todo",
                    }
                },
            },
            {
                "schema": 1,
                "cells": {
                    "cos/fp32/none/scalar": ["KernelParityMul/ScalarFp32MatchesDoubleReference"],
                    "cos/fp32/none/rvv": ["KernelParityRvv/MulFp32MatchesScalarBitwise"],
                },
            },
            "the k3 cell generator cannot express it",
        ),
        (
            # R10 (PLAN gate/ports): a tier cell is only a cell if the scalar reference covers it.
            "R10 a tier cell without a scalar reference",
            mutate_matrix(tiers={"scalar": "todo", "x86": "done"}),
            {"schema": 1, "cells": {"relu/fp32/none/x86": ["kernel_parity/relu_x86_fp32"]}},
            "but the scalar reference is 'todo'",
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
        (
            # R9a: `no-grad` is a decision, and without a reason it reads as `todo`.
            "R9a no-grad without a reason",
            mutate_matrix(grad="no-grad"),
            good_census,
            "claims grad: no-grad without a grad_why",
        ),
        (
            # The positive twin: the same token is accepted once the decision is written down.
            # Without this, tightening R9a could reject every legal `no-grad` row and no fixture
            # would notice.
            "R9a no-grad with a reason is green",
            mutate_matrix(grad="no-grad", grad_why="output is an index, not a value"),
            good_census,
            None,
        ),
        (
            # R9b: `done` claims a derivative exists, so it has to name what proves it.
            "R9b done without a grad test",
            mutate_matrix(grad="done"),
            good_census,
            "claims grad: done without naming a grad test",
        ),
        (
            "R9b a grad test that is not a test",
            mutate_matrix(grad="done", grad_tests=["KernelParityRelu/Imaginary"]),
            good_census,
            "which does not exist in test_package/test/",
        ),
        (
            "R9b done with a real grad test is green",
            mutate_matrix(grad="done", grad_tests=["kernel_parity/relu_scalar_fp32"]),
            good_census,
            None,
        ),
        (
            # R9c: the reverse -- a test listed for a claim nobody makes is decoration.
            "R9c grad tests on an unfinished claim",
            mutate_matrix(grad="todo", grad_tests=["kernel_parity/relu_scalar_fp32"]),
            good_census,
            "a test named for an unfinished claim is decoration",
        ),
        (
            # R8: the cell is implemented and R5 is satisfied by the presence of a name, but the
            # name is not a test -- the state the tree was actually in for 20 of 36 cells.
            "R8 a census name that is not a test",
            good_matrix,
            {"schema": 1, "cells": {"relu/fp32/none/scalar": ["kernel_parity/imaginary_relu"]}},
            "which does not exist in test_package/test/",
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

        baseline, baseline_pending = run_check(
            write("good.yml", good_matrix), write("good.json", good_census), tests=fixture_tests
        )
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
            write("pending.yml", pending_matrix), write("pending.json", good_census), tests=fixture_tests
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
            write("pair.yml", per_pair_matrix), write("pair.json", per_pair_census), tests=fixture_tests
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
        wide_problems, _ = run_check(
            write("wide.yml", widened), write("wide.json", per_pair_census), tests=fixture_tests
        )
        if [p for p in wide_problems if "fc/fp32/none/arm_m" in p and "missing from census.json" in p]:
            print("[census:selftest] PASS the plain form still fans out to every pair")
        else:
            failures += 1
            print(f"[census:selftest] FAIL a kernel-level 'done' must demand every pair, got {wide_problems}")

        for index, (label, matrix, census, phrase) in enumerate(fixtures):
            matrix_path = write(f"matrix-{index}.yml", matrix)
            census_path = write(f"census-{index}.json", census)
            problems, _ = run_check(matrix_path, census_path, tests=fixture_tests)
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

    # R12 is not matrix-shaped: it compares the frozen blob table with the header text, so it
    # gets its two fixtures here rather than in the tuple list above.
    header = "INFR_ACT_NONE = 0,\nINFR_ACT_RELU = 1,\nINFR_ACT_COUNT = 2"
    coherent = []
    check_blob_activation_table(coherent, header_text=header, act_map={"none": 0, "relu": 1})
    if coherent:
        failures += 1
        print(f"[census:selftest] FAIL R12 a table that matches the header: {coherent}")
    else:
        print("[census:selftest] PASS R12 a table that matches the header")
    lagging = []
    check_blob_activation_table(lagging, header_text=header, act_map={"none": 0})
    if not any("R12" in problem and "disagree" in problem for problem in lagging):
        failures += 1
        print(f"[census:selftest] FAIL R12 a table that lags the header, got {lagging}")
    else:
        print(f"[census:selftest] PASS R12 a table that lags the header -> {lagging[0]}")

    # R13, four fixtures: "each rule gets a fixture that only it can make red" is the reason
    # there are four rather than one -- each failure mode needs to be shown on its own.
    def r13(matrix_ops, **tables):
        base = {"translated": {"Relu"}, "deferred": {}, "node_targets": {"relu"},
                "act_targets": {"relu"}, "op_codes": {"relu"}, "act_codes": {"relu"}}
        base.update(tables)
        found = []
        check_importer_vocabulary(
            found, data={"kernels": {"relu": {"onnx": [{"op": op} for op in matrix_ops]}}},
            tables=base)
        return found

    if r13(["Relu"]):
        failures += 1
        print(f"[census:selftest] FAIL R13 a vocabulary that names each other: {r13(['Relu'])}")
    else:
        print("[census:selftest] PASS R13a a vocabulary that names each other")
    problems = r13(["MatMul"])
    if any("R13a" in p and "MatMul" in p for p in problems):
        print("[census:selftest] PASS R13a a matrix claim the importer cannot serve")
    else:
        failures += 1
        print(f"[census:selftest] FAIL R13a expected the unserved-claim report, got {problems}")
    problems = r13(["Relu"], deferred={"Relu": "already done"})
    if any("R13b" in p and "translates" in p for p in problems):
        print("[census:selftest] PASS R13b a deferral that outlives the work")
    else:
        failures += 1
        print(f"[census:selftest] FAIL R13b expected the stale-deferral report, got {problems}")
    problems = r13(["Relu"], node_targets={"no_such_op"})
    if any("R13c" in p and "no_such_op" in p for p in problems):
        print("[census:selftest] PASS R13c a target the blob format does not have")
    else:
        failures += 1
        print(f"[census:selftest] FAIL R13c expected the missing-target report, got {problems}")

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
