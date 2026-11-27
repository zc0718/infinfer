#!/usr/bin/env python3
"""Import an ONNX model into an `INFRKAT` v2 blob, or check whether it is in the subset.

Two modes, and the second one is the reason this file exists:

  emit_onnx_blob.py --check  model.onnx        # what would refuse, and why
  emit_onnx_blob.py         model.onnx out.blob
  emit_onnx_blob.py --selftest                  # no model needed

`--selftest` builds a small ONNX graph with `onnx.helper`, imports it, and compares the
result **byte for byte** against the same blob written out by hand -- including the
NCHW -> NHWC filter transposition, which is the only place the two conventions disagree
and therefore the only place a silent error can live. If the importer transposed the
wrong axis, or forgot a bias slot, or numbered a code differently, the two files differ
and the check says where.

The self-check also holds the two refusals that matter: an op outside the subset must be
**named**, and ONNX's default `HardSigmoid` (alpha = 0.2) must be refused rather than
silently treated as the `relu6(x+3)/6` this library implements.

Requires the `onnx` package (import + selftest only; `blobfmt` stays dependency-free).
"""
import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import blobfmt  # noqa: E402
import onnx_subset  # noqa: E402


def _build_reference_model():
    """A graph that touches every translation rule in one place.

    input (1,2,9,7) -> Conv(3 filters, 3x3, stride 2, pad 1) -> Clip(0,6)
                    -> Mul(gate 1x1xC) -> GlobalAveragePool -> Flatten -> Gemm(2 out)

    The input is deliberately **non-square**: the conv lands on an (H,W,C) = (5,4,3) map,
    so H, W and C are all different. A fixture whose axes are all the same number cannot
    tell an axis swap from a correct implementation, and the per-channel gate below is
    exactly the kind of code where such a swap hides (it did: the gate was applied along
    the wrong stride while every extent was equal).

    输入故意是**非方形**：卷积输出 (H,W,C) = (5,4,3)，三个轴互不相同。三个轴都相等的夹具
    分不出“轴换错了”和“算对了”，而下面的逐通道门正是最容易藏住这种错的地方（确实藏过）。
    """
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    def tensor(name, array):
        return numpy_helper.from_array(np.asarray(array, dtype=np.float32), name=name)

    conv_w = np.arange(3 * 2 * 3 * 3, dtype=np.float32).reshape(3, 2, 3, 3) / 32.0
    conv_b = np.array([0.25, -0.5, 0.75], dtype=np.float32)
    gate = np.array([0.5, 1.5, -2.0], dtype=np.float32).reshape(1, 3, 1, 1)
    fc_w = np.arange(2 * 3, dtype=np.float32).reshape(2, 3) / 8.0
    fc_b = np.array([0.125, -0.25], dtype=np.float32)

    nodes = [
        helper.make_node("Conv", ["x", "conv_w", "conv_b"], ["c"], name="stem",
                         kernel_shape=[3, 3], strides=[2, 2], pads=[1, 1, 1, 1], group=1),
        # opset 13 Clip: the bounds are *inputs*, not attributes.
        helper.make_node("Clip", ["c", "clip_min", "clip_max"], ["c6"], name="clip6"),
        helper.make_node("Mul", ["c6", "gate"], ["g"], name="se_gate"),
        helper.make_node("GlobalAveragePool", ["g"], ["p"], name="pool"),
        helper.make_node("Flatten", ["p"], ["f"], name="flatten"),
        helper.make_node("Gemm", ["f", "fc_w", "fc_b"], ["y"], name="classifier", transB=1),
    ]
    graph = helper.make_graph(
        nodes, "k2o_onnx_subset",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 9, 7])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2])],
        [tensor("conv_w", conv_w), tensor("conv_b", conv_b), tensor("gate", gate),
         tensor("fc_w", fc_w), tensor("fc_b", fc_b),
         tensor("clip_min", np.array(0.0)), tensor("clip_max", np.array(6.0))])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


def _expected_blob(model):
    """The same graph, written out by hand: this is the independent statement the importer
    is checked against. Every transposition below is written as an explicit axis order so
    that reading it next to `onnx_subset.translate` shows whether they agree.

    同一张图，手写一遍。下面每个转置都用显式轴序写出来，好让它和
    `onnx_subset.translate` 并排一读就能看出是否一致。
    """
    import numpy as np

    weights = onnx_subset.initializers(model)
    conv_w = np.transpose(weights["conv_w"], (0, 2, 3, 1))          # (oc,ic,kh,kw) -> (oc,kh,kw,ic)
    gate = np.transpose(weights["gate"], (0, 2, 3, 1))              # (1,3,1,1)       -> (1,1,1,3)
    raw = lambda array: np.ascontiguousarray(array, dtype="<f4").tobytes()  # noqa: E731

    slots = [
        ("x", [1, 9, 7, 2], b""),                                     # input, NHWC
        ("c", [1, 5, 4, 3], b""),                                     # conv out
        ("conv_w", [3, 3, 3, 2], raw(conv_w)),
        ("conv_b", [3], raw(weights["conv_b"])),
        ("g", [1, 5, 4, 3], b""),                                     # after the SE gate
        ("gate", [1, 1, 1, 3], raw(gate)),
        ("p", [1, 1, 1, 3], b""),                                     # global average pool
        ("y", [1, 1, 1, 2], b""),                                     # classifier output
        ("fc_w", [2, 3], raw(weights["fc_w"])),
        ("fc_b", [2], raw(weights["fc_b"])),
    ]
    nodes = [
        {"op": "input", "out": "x"},
        {"op": "conv2d", "in": "x", "out": "c", "w": "conv_w", "b": "conv_b",
         "p": (3, 2, 1, 1), "act": "relu6"},
        {"op": "mul", "in": "c", "out": "g", "w": "gate", "p": (1, 0, 0, 0)},
        {"op": "mean_hw", "in": "g", "out": "p"},
        {"op": "fc", "in": "p", "out": "y", "w": "fc_w", "b": "fc_b"},
    ]
    return blobfmt.build(slots, nodes)


def reference_forward(model, x):
    """The reference model's arithmetic, written from the ONNX semantics in NCHW numpy.

    This is deliberately a *second* implementation: the point of baking its output as a
    fixture is that the executor agreeing with it means two independent statements of the
    same graph agree, not that one implementation agrees with itself.

    参考前向：按 ONNX 语义在 NCHW 的 numpy 里重写一遍。故意做成**第二份**实现——把它的输出
    烘成夹具，执行器与它一致才说明"同一张图的两种独立表述一致"，而不是一个实现自证。
    """
    import numpy as np

    weights = onnx_subset.initializers(model)
    state = {"x": np.asarray(x, dtype=np.float32)}

    def value(name):
        return state[name] if name in state else weights[name]

    for node in model.graph.node:
        kind = node.op_type
        if kind in onnx_subset.PASSTHROUGH:
            state[node.output[0]] = value(node.input[0]).reshape(
                [d if d else -1 for d in _reshape_dims(node, weights)]
            ) if kind == "Reshape" else value(node.input[0]).reshape(-1)
            continue
        if kind == "Conv":
            weight = weights[node.input[1]]
            bias = weights[node.input[2]] if len(node.input) > 2 and node.input[2] else None
            attrs = {attr.name: attr for attr in node.attribute}
            stride = list(attrs["strides"].ints)[0] if "strides" in attrs else 1
            pads = list(attrs["pads"].ints) if "pads" in attrs else [0, 0, 0, 0]
            groups = int(attrs["group"].i) if "group" in attrs else 1
            src = value(node.input[0])
            n, c, h, w = src.shape
            oc, icg, kh, kw = weight.shape
            oh = (h + pads[0] + pads[2] - kh) // stride + 1
            ow = (w + pads[1] + pads[3] - kw) // stride + 1
            padded = np.pad(src, ((0, 0), (0, 0), (pads[0], pads[2]), (pads[1], pads[3])))
            out = np.zeros((n, oc, oh, ow), dtype=np.float64)
            per_group = oc // groups
            for g in range(groups):
                src_g = padded[:, g * icg:(g + 1) * icg]
                for o in range(g * per_group, (g + 1) * per_group):
                    acc = np.zeros((n, oh, ow), dtype=np.float64)
                    for ky in range(kh):
                        for kx in range(kw):
                            acc += np.einsum("nchw,c->nhw", src_g[:, :, ky:ky + stride * oh:stride,
                                                              kx:kx + stride * ow:stride],
                                             weight[o, :, ky, kx].astype(np.float64))
                    out[:, o] = acc
            if bias is not None:
                out += bias.reshape(1, oc, 1, 1)
            state[node.output[0]] = out.astype(np.float32)
            continue
        if kind == "GlobalAveragePool":
            state[node.output[0]] = value(node.input[0]).mean(axis=(2, 3), keepdims=True,
                                                              dtype=np.float32)
            continue
        if kind == "Gemm":
            src = value(node.input[0]).reshape(-1).astype(np.float32)
            weight = weights[node.input[1]]
            out = (weight.astype(np.float32) @ src).reshape(1, -1)
            if len(node.input) > 2 and node.input[2]:
                out = out + weights[node.input[2]].reshape(1, -1)
            state[node.output[0]] = out
            continue
        if kind == "Mul":
            state[node.output[0]] = (value(node.input[0]) * value(node.input[1])).astype(np.float32)
            continue
        if kind == "Add":
            state[node.output[0]] = (value(node.input[0]) + value(node.input[1])).astype(np.float32)
            continue
        if kind in onnx_subset.ACTIVATIONS:
            src = value(node.input[0])
            if kind == "Relu":
                state[node.output[0]] = np.maximum(src, 0.0).astype(np.float32)
            elif kind == "Clip":
                state[node.output[0]] = np.clip(src, 0.0, 6.0).astype(np.float32)
            elif kind == "HardSigmoid":
                state[node.output[0]] = np.clip(src / 6.0 + 0.5, 0.0, 1.0).astype(np.float32)
            else:  # HardSwish
                state[node.output[0]] = (src * np.clip(src / 6.0 + 0.5, 0.0, 1.0)).astype(np.float32)
            continue
        raise SystemExit(f"reference_forward: op {kind} has no reference implementation")
    return state[model.graph.output[0].name]


def _reshape_dims(node, weights):
    import numpy as np

    return [int(round(float(v))) for v in np.asarray(weights[node.input[1]]).ravel()]


def fixed_input(model):
    """The input the fixture is baked for: a deterministic ramp, exactly representable."""
    import numpy as np

    source = [entry for entry in model.graph.input
              if entry.name not in onnx_subset.initializers(model)][0]
    dims = [axis.dim_value for axis in source.type.tensor_type.shape.dim]
    count = int(np.prod(dims))
    return (np.arange(count, dtype=np.float32) / 16.0 - 1.0).reshape(dims)


def selftest() -> int:
    import onnx

    failures = 0
    model = _build_reference_model()

    # 1. The subset check must accept the reference model.
    problems = onnx_subset.check(model)
    if problems:
        print(f"[emit_onnx_blob] FAIL the reference model was refused: {problems}", file=sys.stderr)
        failures += 1
    else:
        print("[emit_onnx_blob] PASS the reference model is inside the subset")

    # 2. The translation must equal the hand-written blob, byte for byte.
    translated = onnx_subset.translate(model)
    expected = _expected_blob(model)
    if translated == expected:
        print(f"[emit_onnx_blob] PASS the translation matches the hand-written blob ({len(expected)} B)")
    else:
        failures += 1
        first = next((i for i, (a, b) in enumerate(zip(translated, expected)) if a != b),
                     min(len(translated), len(expected)))
        print(f"[emit_onnx_blob] FAIL the translation differs at byte {first} "
              f"({len(translated)} B vs {len(expected)} B)", file=sys.stderr)

    # 2b. The reference forward says what the graph computes; the fixture bakes it so the
    #     executor can be held to it from C++ (see graph_blob_onnx_subset.cpp).
    import numpy as np

    out = reference_forward(model, fixed_input(model))
    print(f"[emit_onnx_blob] the reference forward gives {np.asarray(out).ravel().tolist()}")

    # 3. An op outside the subset must be named, not swallowed.
    outside = onnx.ModelProto()
    outside.CopyFrom(model)
    extra = outside.graph.node.add()
    extra.op_type = "Sigmoid"
    extra.name = "not_in_subset"
    extra.input.append("g")
    extra.output.append("sigmoid_out")
    outside.graph.output[0].name = "sigmoid_out"
    problems = onnx_subset.check(outside)
    hit = [reason for name, reason in problems if name == "not_in_subset"]
    if hit and "Sigmoid" in hit[0]:
        print(f"[emit_onnx_blob] PASS an unsupported op is named: {hit[0]}")
    else:
        failures += 1
        print(f"[emit_onnx_blob] FAIL an unsupported op was not named: {problems}", file=sys.stderr)

    # 4. ONNX's default HardSigmoid is a different function and must be refused.
    shifted = onnx.ModelProto()
    shifted.CopyFrom(model)
    node = shifted.graph.node.add()
    node.op_type = "HardSigmoid"      # no attributes: ONNX default alpha=0.2, beta=0.5
    node.name = "default_hard_sigmoid"
    node.input.append("g")
    node.output.append("hs_out")
    shifted.graph.output[0].name = "hs_out"
    problems = onnx_subset.check(shifted)
    hit = [reason for name, reason in problems if name == "default_hard_sigmoid"]
    if hit and "alpha=0.2" in hit[0]:
        print(f"[emit_onnx_blob] PASS a default HardSigmoid is refused: {hit[0]}")
    else:
        failures += 1
        print(f"[emit_onnx_blob] FAIL a default HardSigmoid was accepted: {problems}", file=sys.stderr)

    if failures:
        print(f"[emit_onnx_blob] FAIL ({failures} check(s))", file=sys.stderr)
        return 1
    print("[emit_onnx_blob] selftest OK")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", nargs="?", help="the .onnx file to import")
    ap.add_argument("out", nargs="?", help="where to write the blob")
    ap.add_argument("--check", action="store_true", help="report refusals and exit; write nothing")
    ap.add_argument("--selftest", action="store_true", help="build, import and compare; needs no model")
    ap.add_argument("--emit-fixture", metavar="DIR",
                    help="write the reference model's blob, input and expected output into DIR, so a "
                         "C++ test can hold the executor to the reference forward")
    args = ap.parse_args()

    if args.emit_fixture:
        import numpy as np

        model = _build_reference_model()
        out_dir = pathlib.Path(args.emit_fixture)
        out_dir.mkdir(parents=True, exist_ok=True)
        x = fixed_input(model)
        blob = onnx_subset.translate(model)
        counts, refs, _ops = blobfmt.layout(blob)
        in_floats, out_floats = blobfmt.endpoints(counts, refs, _ops)
        # The blob's input slot is NHWC, so the fixture carries the tensor in *that* layout.
        # Writing ONNX's NCHW here would still give the right element count and the right
        # file sizes; the only symptom would be a numeric mismatch nobody could localise.
        body = onnx_subset.nchw_to_nhwc_payload(x)
        if body.size != in_floats:
            raise SystemExit(f"[emit_onnx_blob] input has {body.size} floats, "
                             f"the blob declares {in_floats}")
        want = np.ascontiguousarray(reference_forward(model, x), dtype="<f4").ravel()
        if want.size != out_floats:
            raise SystemExit(f"[emit_onnx_blob] expectation has {want.size} floats, "
                             f"the blob declares {out_floats}")
        (out_dir / "model.blob").write_bytes(blob)
        (out_dir / "input.bin").write_bytes(body.tobytes())
        (out_dir / "expect.bin").write_bytes(want.tobytes())
        print(f"[emit_onnx_blob] wrote {out_dir} (model.blob, input.bin in blob order, "
              f"expect.bin: {in_floats} in float -> {out_floats} out float)")
        return 0
    if args.selftest:
        return selftest()
    if not args.model:
        ap.error("a model is required unless --selftest is given")

    import onnx

    model = onnx.load(args.model)
    problems = onnx_subset.check(model)
    for name, reason in problems:
        print(f"[emit_onnx_blob] refused {name}: {reason}", file=sys.stderr)
    if problems:
        return 1
    if args.check:
        print(f"[emit_onnx_blob] {args.model}: inside the subset ({len(model.graph.node)} node(s))")
        return 0
    if not args.out:
        ap.error("an output path is required to write a blob")
    blob = onnx_subset.translate(model)
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.out).write_bytes(blob)
    counts, refs, _ops = blobfmt.layout(blob)
    print(f"[emit_onnx_blob] wrote {args.out} ({len(blob)} B, {len(refs)} node(s), "
          f"arena {blobfmt.plan_arena(counts, refs)} float)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
