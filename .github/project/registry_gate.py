#!/usr/bin/env python3
"""The kernel registry must say exactly what the matrix claims, and nothing else.

`src/kernel/matrix.yml` is the single source of legal combinations (INV-10). `census_gate.py`
ties it to `census.json` (does a `done` cell have a named test?), but the **third** list --
the `builtins[]` table inside each tier's `register.c` -- was tied to nothing. That gap has a
silent failure mode rather than a loud one: a parity test asks the registry for a row, gets
`NULL`, and calls `GTEST_SKIP()`. Drop one row from a registry and the suite still reports
100% passing with a skip nobody reads.

Rules:

  R1 the matrix parses and declares `kernels` with `dtype_quant` and `tiers`
  R2 every `done` cell has a registry row with that (name, dtype, quant, tier)
  R3 every registry row corresponds to a `done` cell -- a row nobody claims is drift too
  R4 a row's `.fn` is `infr_<tier>_<kernel>_<dtype-suffix>`, the convention every tier
     already follows; a mismatch means the row points at some other kernel's body
  R5 no two rows in one tier share a `.fn` -- copy-paste in a table is invisible otherwise

`--emit <tier>` prints the rows the matrix implies, so the table can be regenerated rather
than retyped. `--selftest` proves each rule can fail on its own fixture.

**为什么它值得存在**：矩阵与 census 之间有门禁，但每个档位 `register.c` 里的 `builtins[]`
第三份清单没有任何东西在管。它的失配是静默的——parity 测试拿不到行就 `GTEST_SKIP()`，
套件照报 100% 通过，只是多了一条没人读的跳过。
"""

import argparse
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
MATRIX = ROOT / "src/kernel/matrix.yml"

TIER_DIRS = {
    "scalar": "src/kernel/scalar",
    "x86": "src/kernel/x86",
    "arm_neon": "src/kernel/arm_neon",
    "arm_m": "src/kernel/arm_m",
    "rvv": "src/kernel/rvv",
}
TIER_ENUM = {
    "scalar": "INFR_TIER_SCALAR",
    "autovec": "INFR_TIER_AUTOVEC",
    "x86": "INFR_TIER_X86",
    "arm_neon": "INFR_TIER_ARM_NEON",
    "arm_m": "INFR_TIER_ARM_M",
    "rvv": "INFR_TIER_RVV",
}
DTYPE_ENUM = {"fp32": "INFR_DTYPE_F32", "int8": "INFR_DTYPE_I8", "int16": "INFR_DTYPE_I16"}
QUANT_ENUM = {
    "none": "INFR_QUANT_NONE",
    "tensor": "INFR_QUANT_PER_TENSOR",
    "channel": "INFR_QUANT_PER_CHANNEL",
}
DTYPE_SUFFIX = {"fp32": "f32", "int8": "i8", "int16": "i16"}

ROW_RE = re.compile(
    r"\{\.name = \"([a-z0-9_]+)\"\s*,\s*"
    r"\.dtype = (INFR_DTYPE_[A-Z0-9_]+)\s*,\s*"
    r"\.quant = (INFR_QUANT_[A-Z0-9_]+)\s*,\s*"
    r"\.tier = (INFR_TIER_[A-Z0-9_]+)\s*,\s*"
    r"\.fn = ([A-Za-z0-9_]+)\s*,?\s*\}"
)


def load_yaml(path):
    try:
        import yaml
    except ImportError:
        raise SystemExit("registry_gate: PyYAML is missing (pip install pyyaml)")
    return yaml.safe_load(pathlib.Path(path).read_text(encoding="utf-8"))


def done_cells(matrix):
    """{tier: {(kernel, dtype, quant), ...}} for every `done` claim."""
    out = {}
    for name, entry in matrix["kernels"].items():
        pairs = [tuple(pair) for pair in entry["dtype_quant"]]
        for tier, claim in entry["tiers"].items():
            if isinstance(claim, str):
                resolved = {pair: claim for pair in pairs}
            else:
                resolved = {tuple(key.split("/")): value for key, value in claim.items()}
            for pair, status in resolved.items():
                if status == "done":
                    out.setdefault(tier, set()).add((name, pair[0], pair[1]))
    return out


def parse_rows(text):
    return [
        {
            "name": name,
            "dtype": dtype,
            "quant": quant,
            "tier": tier,
            "fn": fn,
        }
        for name, dtype, quant, tier, fn in ROW_RE.findall(text)
    ]


def check(matrix, registry_texts):
    """R1-R5 over a matrix and {tier: register.c text}. Returns a problem list."""
    problems = []
    if not isinstance(matrix, dict) or matrix.get("schema") != 1:
        return ["registry: matrix.yml must declare `schema: 1`"]
    if not isinstance(matrix.get("kernels"), dict) or not matrix["kernels"]:
        return ["registry: matrix.yml has no `kernels` map"]

    claims = done_cells(matrix)

    for tier in sorted(claims):
        if tier not in TIER_DIRS:
            problems.append(f"registry: tier '{tier}' claims done cells but has no registry directory")
        elif tier not in registry_texts:
            problems.append(f"registry: tier '{tier}' claims done cells but has no register.c")

    # R4/R5 first, so a broken row is reported before the set comparisons use it.
    for tier, text in sorted(registry_texts.items()):
        rows = parse_rows(text)
        seen = {}
        for row in rows:
            if row["tier"] != TIER_ENUM.get(tier):
                problems.append(
                    f"registry: {tier}/register.c has a row for '{row['name']}' whose tier is "
                    f"{row['tier']} (expected {TIER_ENUM.get(tier)})"
                )
            want = f"infr_{tier}_{row['name']}_"
            if not row["fn"].startswith(want):
                problems.append(
                    f"registry: {tier}/register.c wires '{row['name']}' to {row['fn']}, which does "
                    f"not follow the infr_{tier}_{row['name']}_<dtype> convention"
                )
            if row["fn"] in seen:
                problems.append(
                    f"registry: {tier}/register.c uses {row['fn']} twice "
                    f"('{seen[row['fn']]}' and '{row['name']}')"
                )
            seen[row["fn"]] = row["name"]

    # R2: every claim is registered.
    for tier, cells in sorted(claims.items()):
        if tier not in registry_texts:
            continue
        rows = parse_rows(registry_texts[tier])
        have = {(r["name"], r["dtype"], r["quant"]) for r in rows if r["tier"] == TIER_ENUM.get(tier)}
        for name, dtype, quant in sorted(cells):
            key = (name, DTYPE_ENUM.get(dtype, dtype), QUANT_ENUM.get(quant, quant))
            if key not in have:
                problems.append(
                    f"registry: {tier}/register.c is missing a row for the done cell "
                    f"'{name}/{dtype}/{quant}' -- a parity test would silently skip it"
                )

    # R3: every row is claimed.
    for tier, text in sorted(registry_texts.items()):
        tier_cells = claims.get(tier, set())
        for row in parse_rows(text):
            if row["tier"] != TIER_ENUM.get(tier):
                continue
            match = [
                (name, dtype, quant)
                for name, dtype, quant in tier_cells
                if name == row["name"]
                and DTYPE_ENUM.get(dtype, dtype) == row["dtype"]
                and QUANT_ENUM.get(quant, quant) == row["quant"]
            ]
            if not match:
                problems.append(
                    f"registry: {tier}/register.c registers '{row['name']}' for {row['dtype']}/"
                    f"{row['quant']} but matrix.yml does not claim that cell as done"
                )
    return problems


def read_registries(root):
    texts = {}
    for tier, relative in TIER_DIRS.items():
        path = pathlib.Path(root) / relative / "register.c"
        if path.is_file():
            texts[tier] = path.read_text(encoding="utf-8")
    return texts


def emit(matrix, tier):
    """The rows the matrix implies for one tier, ready to paste into register.c."""
    if tier not in TIER_DIRS:
        raise SystemExit(f"registry_gate: unknown tier '{tier}'")
    out = []
    for name, dtype, quant in sorted(done_cells(matrix).get(tier, set())):
        out.append(
            "        {.name = \"%s\",\n"
            "         .dtype = %s,\n"
            "         .quant = %s,\n"
            "         .tier = %s,\n"
            "         .fn = infr_%s_%s_%s},\n"
            % (name, DTYPE_ENUM[dtype], QUANT_ENUM[quant], TIER_ENUM[tier], tier, name, DTYPE_SUFFIX[dtype])
        )
    print("".join(out), end="")


def _matrix(fixtures):
    return {"schema": 1, "kernels": fixtures}


def selftest():
    failures = 0

    good_matrix = _matrix(
        {
            "relu": {
                "dtype_quant": [["fp32", "none"]],
                "tiers": {"scalar": "done"},
            }
        }
    )
    good_text = (
        "static const infr_kernel_desc builtins[] = {\n"
        '    {.name = "relu",\n     .dtype = INFR_DTYPE_F32,\n     .quant = INFR_QUANT_NONE,\n'
        "     .tier = INFR_TIER_SCALAR,\n     .fn = infr_scalar_relu_f32},\n};\n"
    )

    if check(good_matrix, {"scalar": good_text}):
        failures += 1
        print(f"[registry:selftest] FAIL a matching registry must be green, got {check(good_matrix, {'scalar': good_text})}")
    else:
        print("[registry:selftest] PASS a matching registry is green")

    cases = [
        (
            "R2 a done cell with no row is red (the silent skip)",
            good_matrix,
            {"scalar": "static const infr_kernel_desc builtins[] = {\n};\n"},
            "silently skip it",
        ),
        (
            "R3 a row nobody claims is red",
            _matrix({"relu": {"dtype_quant": [["fp32", "none"]], "tiers": {"scalar": "todo"}}}),
            {"scalar": good_text},
            "does not claim that cell as done",
        ),
        (
            "R4 a row wired to another kernel's body is red",
            good_matrix,
            {"scalar": good_text.replace("infr_scalar_relu_f32", "infr_scalar_fc_f32")},
            "does not follow the infr_scalar_relu_<dtype> convention",
        ),
        (
            "R5 the same body registered twice is red",
            good_matrix,
            {
                "scalar": good_text.replace(
                    "};\n",
                    '    {.name = "relu",\n     .dtype = INFR_DTYPE_I8,\n     .quant = INFR_QUANT_NONE,\n'
                    "     .tier = INFR_TIER_SCALAR,\n     .fn = infr_scalar_relu_f32},\n};\n",
                )
            },
            "uses infr_scalar_relu_f32 twice",
        ),
        (
            "R4 a row whose tier enum disagrees with its file is red",
            good_matrix,
            {"scalar": good_text.replace("INFR_TIER_SCALAR", "INFR_TIER_X86")},
            "(expected INFR_TIER_SCALAR)",
        ),
        (
            "R1 a matrix without kernels is red",
            {"schema": 1},
            {},
            "has no `kernels` map",
        ),
    ]

    for label, matrix, texts, phrase in cases:
        problems = check(matrix, texts)
        hit = [p for p in problems if phrase in p]
        if hit:
            print(f"[registry:selftest] PASS {label} -> {hit[0]}")
        else:
            failures += 1
            print(f"[registry:selftest] FAIL {label}: expected '{phrase}', got {problems}")

    if failures:
        print(f"[registry:selftest] FAIL ({failures} fixture(s))", file=sys.stderr)
        return 1
    print("[registry:selftest] OK")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--matrix", default=str(MATRIX), help="matrix.yml to read")
    parser.add_argument("--root", default=str(ROOT), help="tree that holds src/kernel/*/register.c")
    parser.add_argument("--emit", metavar="TIER", help="print the rows the matrix implies for one tier")
    parser.add_argument("--selftest", action="store_true", help="prove every rule can fail")
    args = parser.parse_args()

    if args.selftest:
        return selftest()

    matrix = load_yaml(args.matrix)
    if args.emit:
        emit(matrix, args.emit)
        return 0

    problems = check(matrix, read_registries(args.root))
    if problems:
        for problem in problems:
            print(f"[registry] {problem}", file=sys.stderr)
        print(f"[registry] FAIL ({len(problems)} problem(s))", file=sys.stderr)
        return 1
    total = sum(len(cells) for cells in done_cells(matrix).values())
    print(f"[registry] OK ({total} done cell(s) all registered, no orphan row)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
