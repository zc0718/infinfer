#!/usr/bin/env python3
"""Emit the cell list a tier-parity KAT build has to compare, derived from matrix.yml.

`k3` is self-comparing: it runs the scalar row and the tier row of the same cell in one process
and compares them, so there is no baked expectation and nothing to hand-write. What it does need
is the list of cells -- and that list already exists in exactly one place, `src/kernel/matrix.yml`.
This script turns the `done` cells of one tier into a C table so the chain cannot claim a cell the
matrix does not, and cannot silently skip one the matrix does.

Deriving rather than listing is the point (PLAN P2.9): a hand-written table in the chain would go
stale the first time a cell was flipped, and the failure would look like a passing test.

    python3 emit_k3_cells.py <matrix.yml> <tier> <out.h>

`tier` is a token from the matrix (`x86`, `arm_neon`, `arm_m`, `rvv`). The output is a header with
`kat_k3_cells[]` and `KAT_K3_TIER`; it is written to workspace/qa/kat/generated/ and included by
`kat_chain_k3.c`.
"""
import json
import pathlib
import sys

# How a kernel is called, and how its outputs are compared. Both are properties of the kernel, not
# of the tier, so they are decided here once instead of in every cell row.
#
#   UNARY        in[0] -> out[0], one output per input
#   BINARY       in[0], in[1] -> out[0] (the second addend is a real operand, not a factor)
#   SCALE        in[0], in[1] = one factor broadcast over the input
#   REDUCE       in[0] -> out[0] = one reduced value
#   DOT          in[0], in[1] = two vectors, out[0] = one dot product
#   DOT_Q        like DOT, plus a resolved infr_quant
#   QUANTIZE     fp32 in -> int8 out, with a resolved infr_quant
#   DEQUANTIZE   int8 in -> fp32 out, with a resolved infr_quant
#
# BITWISE vs WINDOW is the criterion the parity harness already uses, restated in C:
#   * elementwise and integer cells do one IEEE operation (or exact integer arithmetic) per
#     element, so the verdict is bit equality;
#   * `sum` and fp32 `fc` accumulate, and the accumulator split differs between the reference and
#     a vector body, so the verdict is a relative window. Measured worst cases on the targets this
#     runs on (gaps G-57, G-60): fc 1.024e-06, sum 1.756e-06. The window below is 4e-06 -- above
#     the measured worst, not a wish.
MODES = {
    "add": ("BINARY", "BITWISE"),
    "relu": ("UNARY", "BITWISE"),
    "scale": ("SCALE", "BITWISE"),
    "max": ("REDUCE", "BITWISE"),
    "sum": ("REDUCE", "WINDOW"),
    "fc": ("DOT", "WINDOW"),  # overridden to DOT_Q/BITWISE for integer dtypes
    # Conversions: the input dtype differs from the output dtype, and both are exact (an integer
    # result from rounding, or an integer subtract followed by one IEEE multiply), so the verdict is
    # bit equality in both directions.
    "quantize": ("QUANTIZE", "BITWISE"),
    "dequantize": ("DEQUANTIZE", "BITWISE"),
}
# Keys are the matrix's own tokens (`fp32`, not `f32`): a near-miss here is silent -- it makes every
# cell look unsupported, which reads as "this tier has nothing to compare" rather than as a typo.
SUPPORTED = {
    ("fp32", "none"): "INFR_DTYPE_F32, INFR_QUANT_NONE",
    # fp32 with per-tensor quantisation is the quantize cell: the dtype is still fp32, the
    # quantisation describes where the output lands.
    ("fp32", "tensor"): "INFR_DTYPE_F32, INFR_QUANT_PER_TENSOR",
    # int8 with quant `none` is the 8-bit elementwise cells (relu): the dtype is quantised in the
    # graph but the kernel itself takes no requantisation.
    ("int8", "none"): "INFR_DTYPE_I8, INFR_QUANT_NONE",
    ("int8", "tensor"): "INFR_DTYPE_I8, INFR_QUANT_PER_TENSOR",
    ("int16", "tensor"): "INFR_DTYPE_I16, INFR_QUANT_PER_TENSOR",
}


def load_yaml(path):
    try:
        import yaml
    except ImportError:
        raise SystemExit("emit_k3_cells: PyYAML is required (`pip install pyyaml`)")
    return yaml.safe_load(pathlib.Path(path).read_text(encoding="utf-8"))


def status_of(spec, dtype, quant):
    """One tier entry, resolved for one (dtype, quant) pair.

    The matrix allows a per-pair mapping (G-51) or a kernel-level token; both mean "this tier is
    real for this pair" here, and anything else (todo, na) is not a cell this chain should claim.
    """
    if isinstance(spec, str):
        return spec
    if isinstance(spec, dict):
        return spec.get(f"{dtype}/{quant}")
    return None


def collect(matrix, tier):
    cells = []
    for name, entry in matrix["kernels"].items():
        if name not in MODES:
            continue
        for dtype, quant in entry["dtype_quant"]:
            pair = (dtype, quant)
            if pair not in SUPPORTED:
                continue
            if status_of(entry["tiers"].get(tier), dtype, quant) != "done":
                continue
            call, cmp_mode = MODES[name]
            if name == "fc" and dtype != "fp32":
                # Integer dot products accumulate exactly, so bit equality is the right verdict and
                # the quantised entry point is the one to call.
                call, cmp_mode = "DOT_Q", "BITWISE"
            cells.append((name, dtype, quant, call, cmp_mode))
    return cells


def main(argv):
    if len(argv) != 4:
        print(__doc__, file=sys.stderr)
        return 2
    matrix_path, tier, out_path = argv[1], argv[2], argv[3]
    matrix = load_yaml(matrix_path)
    cells = collect(matrix, tier)
    if not cells:
        print(f"emit_k3_cells: tier '{tier}' has no done cell this chain can compare",
              file=sys.stderr)
        return 1

    tier_enum = "INFR_TIER_" + tier.upper()
    lines = [
        "// Conan::ImportStart",
        "/* Generated by workspace/qa/prep/emit_k3_cells.py from src/kernel/matrix.yml.",
        " * Do not edit: change the matrix and regenerate. The tier below is the only tier this",
        " * image compares, and every row is a cell the matrix calls `done` for it.",
        " *",
        " * 由 emit_k3_cells.py 从 matrix.yml 生成，不要手改：改矩阵再重生成。下面这行是",
        " * 本镜像比较的唯一档位，每一行都是矩阵为它声明 `done` 的格。 */",
        f"#define KAT_K3_TIER {tier_enum}",
        f'#define KAT_K3_TIER_NAME "{tier}"',
        # The register function of this tier only: the other tiers' sources are not linked into
        # this image, so calling all of them would not link.
        f"#define KAT_K3_REGISTER infr_kernel_register_{tier}",
        f"#define KAT_K3_CELL_COUNT {len(cells)}u",
        "",
        "static const kat_k3_cell kat_k3_cells[KAT_K3_CELL_COUNT] = {",
    ]
    for name, dtype, quant, call, cmp_mode in cells:
        lines.append(
            f'    {{"{name}", {SUPPORTED[(dtype, quant)]}, '
            f'KAT_K3_CALL_{call}, KAT_K3_CMP_{cmp_mode}}},'
        )
    lines.append("};")
    lines.append("// Conan::ImportEnd")
    lines.append("")

    out = pathlib.Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"[emit_k3_cells] {tier}: {len(cells)} cell(s) -> {out}")
    print("  " + json.dumps([f"{n}/{d}/{q}" for n, d, q, _, _ in cells]))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
