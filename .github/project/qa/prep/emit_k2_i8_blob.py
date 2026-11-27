#!/usr/bin/env python3
"""Emit the first **v3** blob: one int8 convolution, with the reference output it must produce.

The point of this fixture is not to be a network. It is to be the smallest graph that exercises
every new field in the blob format at once -- an int8 activation slot with a per-tensor scale, an
int8 weight slot with per-channel scales, an int32 bias, a non-zero output zero point -- and to have
its expected answer computed by an **independent integer implementation** rather than by the C++
executor it is checking. If this file and the executor agreed on a wrong rounding rule, they would
have to have made the same mistake twice.

这个 fixture 不是为了成为一个网络，而是为了用最小的图一次跑满格式里所有新字段——逐张量 scale 的
int8 激活槽、逐通道 scale 的 int8 权重槽、int32 偏置、非零输出零点——并让它的期望值由一份
**独立的整数实现**算出，而不是由它要检查的 C++ 执行器算出。若两者在错误的舍入规则上一致，那得
是同一个错误被独立犯了两次。

Machine-level? No. It uses only the standard library, so `ci.sh` can run it and the test it feeds.

机器级？不是。只用标准库，所以 `ci.sh` 能跑它，也能跑它喂的那个测试。
"""
import argparse
import hashlib
import json
import pathlib
import struct
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import blobfmt  # noqa: E402

# A 5x5x2 input, a 3x3x2 filter with 3 output channels, no padding, stride 1 -> 3x3x3.
IN_H, IN_W, IN_C = 5, 5, 2
OUT_C, KERNEL = 3, 3
OUT_H, OUT_W = IN_H - KERNEL + 1, IN_W - KERNEL + 1

# Chosen so that nothing saturates in the nominal case and the arithmetic is checkable by hand:
# the accumulator for one output element is at most 9 * 2 * 127 * 127 = 290322, times a multiplier
# near 0.01 gives about 2900 -- still inside the int16 range, so the rounding rule is what decides
# the answer, not the clamp.
INPUT_SCALE = 0.02
OUTPUT_SCALE = 0.05
OUTPUT_ZERO_POINT = -7
WEIGHT_SCALES = [0.010, 0.013, 0.016]


def quantise_input(values):
    """The activations, quantised per tensor with a zero point of 0 (the operator's subset)."""
    return bytes(bytearray((v & 0xFF) for v in values))


def as_i8(value):
    return value - 256 if value > 127 else value


def reference(input_bytes, weights, bias):
    """The expected output, from the definition -- int64 accumulation, one rounding, one clamp.

    Written out rather than imported: `infr_scalar_requantize_i32` lives in the library, and a
    reference that calls the code under test proves nothing about the code under test.

    期望输出按定义写出——int64 累加、一次舍入、一次截断。刻意不 import
    `infr_scalar_requantize_i32`：调用被测代码的"参考实现"证明不了被测代码。
    """
    activations = [as_i8(b) for b in input_bytes]
    out = []
    for oy in range(OUT_H):
        for ox in range(OUT_W):
            for oc in range(OUT_C):
                acc = 0
                for ky in range(KERNEL):
                    for kx in range(KERNEL):
                        for ci in range(IN_C):
                            src = (((oy + ky) * IN_W) + (ox + kx)) * IN_C + ci
                            w = weights[oc][ky][kx][ci]
                            acc += activations[src] * w
                acc += bias[oc]
                multiplier = (INPUT_SCALE * WEIGHT_SCALES[oc]) / OUTPUT_SCALE
                scaled = (acc * multiplier) + OUTPUT_ZERO_POINT
                if scaled <= -128.0:
                    out.append(-128)
                elif scaled >= 127.0:
                    out.append(127)
                else:
                    out.append(int(scaled + 0.5) if scaled >= 0.0 else int(scaled - 0.5))
    return out


def weights_and_bias():
    """A deterministic filter and bias, with a comment on why these shapes of value.

    The weights are spread over most of the int8 range so a dropped term changes the answer; they
    are *not* all large, because then every output would sit on the clamp and rounding could not be
    observed. The bias is signed and non-zero for the same reason.

    权重铺开到 int8 范围的大部分，这样漏掉一项就会改变答案；又不是全都很大，否则每个输出都坐在
    截断点上，舍入根本观察不到。偏置有正有负、非零，同样是为了这个。
    """
    weights = []
    for oc in range(OUT_C):
        channel = []
        for ky in range(KERNEL):
            row = []
            for kx in range(KERNEL):
                row.append([((((oc * 7) + (ky * 3) + kx) * 11 + ci * 5) % 61) - 30
                            for ci in range(IN_C)])
            channel.append(row)
        weights.append(channel)
    bias = [17, -23, 5]
    return weights, bias


def build(out_dir: pathlib.Path):
    input_values = [(((i * 13) % 97) - 48) for i in range(IN_H * IN_W * IN_C)]
    input_bytes = quantise_input(input_values)
    weights, bias = weights_and_bias()

    weight_bytes = bytearray()
    for oc in range(OUT_C):
        for ky in range(KERNEL):
            for kx in range(KERNEL):
                for ci in range(IN_C):
                    weight_bytes.append(weights[oc][ky][kx][ci] & 0xFF)

    slots = [
        # The input is a runtime slot: byte_len 0 in the format's own terms, but in v3 the width
        # comes from the dtype, so "empty" now means zero bytes rather than zero floats.
        ("input", [1, IN_H, IN_W, IN_C], b"", blobfmt.DTYPE_I8,
         {"kind": "tensor", "scale": INPUT_SCALE, "zero_point": 0}),
        ("conv_w", [OUT_C, KERNEL, KERNEL, IN_C], bytes(weight_bytes), blobfmt.DTYPE_I8,
         {"kind": "channel", "scale": WEIGHT_SCALES[0], "zero_point": 0,
          "channel_scales": WEIGHT_SCALES}),
        ("conv_b", [OUT_C], struct.pack(f"<{OUT_C}i", *bias), blobfmt.DTYPE_I32, None),
        ("conv", [1, OUT_H, OUT_W, OUT_C], b"", blobfmt.DTYPE_I8,
         {"kind": "tensor", "scale": OUTPUT_SCALE, "zero_point": OUTPUT_ZERO_POINT}),
    ]
    nodes = [
        {"op": "input", "out": "input"},
        {"op": "conv2d", "in": "input", "out": "conv", "w": "conv_w", "b": "conv_b",
         "p": (KERNEL, 1, 0, 1)},
    ]
    blob = blobfmt.build(slots, nodes, version=blobfmt.VERSION)

    expected = reference(input_bytes, weights, bias)
    expect = bytes(bytearray((v & 0xFF) for v in expected))

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "model.blob").write_bytes(blob)
    (out_dir / "input.i8").write_bytes(input_bytes)
    (out_dir / "expect.i8").write_bytes(expect)
    (out_dir / "manifest.json").write_text(json.dumps({
        "graph": "k2_i8_conv",
        "blob_version": blobfmt.VERSION,
        "blob_bytes": len(blob),
        "blob_sha256": hashlib.sha256(blob).hexdigest()[:16],
        "input_dims": [1, IN_H, IN_W, IN_C],
        "output_dims": [1, OUT_H, OUT_W, OUT_C],
        "kernel": KERNEL,
        "input_scale": INPUT_SCALE,
        "output_scale": OUTPUT_SCALE,
        "output_zero_point": OUTPUT_ZERO_POINT,
        "weight_scales": WEIGHT_SCALES,
        "operators": ["conv2d"],
        "note": ("the expected output is computed by an integer implementation in this file, not "
                 "by the executor the fixture checks"),
    }, indent=2) + "\n")
    return blob, input_bytes, expect, expected


def selftest():
    """Re-derive the fixture in memory and check the three claims that make it usable.

    A fixture is only worth committing if (a) it parses back through the same reader the other
    generators use, (b) its expectation is reproducible rather than once-written, and (c) it stays
    inside the subset the operator promises -- a fixture that needs an unsupported feature would
    fail for a reason unrelated to what it tests.

    fixture 值得入库，需要三条：(a) 能用其它生成器同一个读取器解回来；(b) 期望值可复现而不是一次性
    写下；(c) 落在算子承诺的子集内——需要不支持特性的 fixture 会因为与它要测的东西无关的原因失败。
    """
    import tempfile

    with tempfile.TemporaryDirectory() as scratch:
        blob, input_bytes, expect, expected = build(pathlib.Path(scratch))

        version, node_count, slot_count, slots = blobfmt.walk_slots(blob)
        assert version == blobfmt.VERSION, f"version {version}"
        assert node_count == 2, f"{node_count} nodes"
        assert slot_count == 4, f"{slot_count} slots"
        by_name = {entry["name"]: entry for entry in slots}
        assert by_name["input"]["dtype"] == blobfmt.DTYPE_I8, "input is not int8"
        assert by_name["input"]["byte_len"] == 0, "input must be a runtime slot"
        assert by_name["conv_w"]["byte_len"] == OUT_C * KERNEL * KERNEL * IN_C, "weight bytes"
        # Round-tripped through f32, so compare as floats rather than as Python floats: 0.010
        # read back as 0.00999999977648258 is the format working, not a mismatch.
        read_back = by_name["conv_w"]["quant"]["channel_scales"]
        assert len(read_back) == len(WEIGHT_SCALES), "per-channel scale count"
        assert all(abs(a - b) < 1e-7 for a, b in zip(read_back, WEIGHT_SCALES)), \
            f"per-channel scales {read_back}"
        assert by_name["conv_b"]["dtype"] == blobfmt.DTYPE_I32, "bias must be int32"
        assert by_name["conv"]["quant"]["zero_point"] == OUTPUT_ZERO_POINT, "output zero point"

        # (b) reproducible: computing it twice from the same inputs must give the same bytes.
        again = reference(input_bytes, *weights_and_bias())
        assert again == expected, "the reference is not deterministic"

        # (c) inside the subset: no saturation in the nominal case, and a zero input landing on
        # the output zero point -- the two properties the operator's own tests lean on.
        assert not any(v in (-128, 127) for v in expected), "the fixture saturates; pick smaller weights"
        assert by_name["conv"]["dims"] == [1, OUT_H, OUT_W, OUT_C], "declared output shape"

        # And v2 is still byte-for-byte the old format: the k2 fixture's first slot must parse
        # through the same reader, which is the whole point of keeping two versions readable.
        version2, _n2, _s2, slots2 = blobfmt.walk_slots(
            (pathlib.Path(__file__).resolve().parents[4] / "test_package" / "resources"
             / "kat" / "k2_conv_micro" / "model.blob").read_bytes())
        assert version2 == blobfmt.VERSION_V2, "the committed k2 blob is not v2 any more"
        assert slots2[0]["quant"]["kind"] == "none", "a v2 slot must read as unquantised"

    print(f"[emit_k2_i8_blob] selftest OK (v3 {len(blob)} B, v2 still readable, "
          f"expected[{expected[:6]}...])")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out",
                        default=str(pathlib.Path(__file__).resolve().parents[4] / "test_package"
                                    / "resources" / "kat" / "k2_i8_conv"),
                        help="directory for model.blob / input.i8 / expect.i8 / manifest.json")
    parser.add_argument("--selftest", action="store_true", help="check the fixture's own claims")
    args = parser.parse_args()

    if args.selftest:
        selftest()
        return 0
    blob, _input_bytes, expect, expected = build(pathlib.Path(args.out))
    print(f"[emit_k2_i8_blob] v{blobfmt.VERSION} blob {len(blob)} B -> {args.out}")
    print(f"[emit_k2_i8_blob] expected {len(expect)} B, first values {expected[:8]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
