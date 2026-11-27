#!/usr/bin/env python3
"""Which ONNX operators do real models actually need?

Why this exists: the "missing operators" table in the plan was derived by reading
architecture descriptions, and the owner pointed out it is short. Guessing at an
operator set is how a library ends up adding `concat` for one model family and then
discovering `split`, `slice` and `sub` for the next one -- so this script derives the
set from **exported graphs** instead, with random weights (no download, same ops).

The models are chosen to cover the plan's architecture axis, one family each:

  mobilenet_v3_small      MobileNetV3 (DWConv + SE)       image classification
  retinanet / fcos        detection head + FPN            object detection
  transformer encoder     attention                       text / sequence
  lstm                    recurrence                      sequence / keyword

在别的机器上重跑：`python3 workspace/qa/op_census.py`（需要 torch + torchvision + onnx）。
"""

import collections
import pathlib
import tempfile

import onnx
import torch
import torch.nn as nn
import torchvision

# The operators this library implements, named the way the plan's matrix names them. The
# mapping below turns an ONNX op into one of these keys, or into `None` for graph-level
# bookkeeping that no kernel should own.
# What this library can actually express, read from the single source of truth rather
# than from memory: the kernel matrix plus the fused activation codes in the header. The
# first version of this table was written from belief and claimed `softmax`, which the
# matrix does not contain -- exactly the mistake this repository forbids.
#
# 本库实际能表达什么，从唯一事实源读取而不是凭印象：内核矩阵 + 头文件里的融合激活码。
# 这张表的第一版是凭印象写的，声称支持 `softmax`，而矩阵里没有——正是本仓明令禁止的错误。
MATRIX = pathlib.Path(__file__).resolve().parents[2] / "src" / "kernel" / "matrix.yml"
HEADER = pathlib.Path(__file__).resolve().parents[2] / "include" / "infinfer" / "kernel" / "kernel.h"
import re as _re
import yaml as _yaml

_m = _yaml.safe_load(MATRIX.read_text())
HAVE = set(_m["kernels"]) | set(_m["composed"])
ACTIVATIONS = set(_re.findall(r"INFR_ACT_([A-Z0-9_]+)", HEADER.read_text()))

# ONNX op -> the kernel that would serve it. `None` means "no kernel covers this".
SERVED_BY = {
    "Conv": "conv2d",
    "Gemm": "fc",
    "MatMul": "fc",
    "Add": "add",
    "Mul": "mul",
    "Relu": "relu",
    "MaxPool": "max_pool2d",
    "ReduceMean": "mean",
    "GlobalAveragePool": "mean",
    "ReduceSum": "sum",
    "GlobalMaxPool": "max",
    "Softmax": "softmax",
    "QuantizeLinear": "quantize",
    "DequantizeLinear": "dequantize",
}
# Ops a fused activation already covers, checked against the header's codes.
SERVED_BY_ACTIVATION = {
    "HardSwish": "HARD_SWISH",
    "HardSigmoid": "HARD_SIGMOID",
    "Tanh": "TANH",
}

# Graph-level plumbing: shape arithmetic and layout bookkeeping that the L2 layer does
# while it walks the graph, not a kernel.
PLUMBING = {
    "Constant",
    "Shape",
    "Gather",
    "Unsqueeze",
    "Squeeze",
    "Reshape",
    "Cast",
    "ConstantOfShape",
    "Range",
    "Expand",
    "Identity",
    "Flatten",
    "NonZero",
}


def export(model, dummy, name):
    """Export to ONNX and return the set of op types actually used."""
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / f"{name}.onnx"
        try:
            torch.onnx.export(model, dummy, path, opset_version=17, do_constant_folding=True)
        except Exception as exc:  # noqa: BLE001 - the point is to report, not to crash
            print(f"  {name}: export failed ({type(exc).__name__}: {str(exc)[:90]})")
            return None
        graph = onnx.load(str(path))
    kinds = collections.Counter(node.op_type for node in graph.graph.node)
    return kinds


def main():
    torch.manual_seed(0)
    results = {}

    x = torch.randn(1, 3, 224, 224)
    results["mobilenet_v3_small"] = export(
        torchvision.models.mobilenet_v3_small(weights=None).eval(), x, "mobilenet_v3_small")

    # Detection: the backbone plus FPN plus head is the operator family YOLO sits in. These
    # exports often fail on the post-processing (NMS), which is not a kernel anyway -- the
    # measurement is what the export actually produced, not what it was supposed to.
    for name, factory in (
        ("retinanet", lambda: torchvision.models.detection.retinanet_resnet50_fpn_v2(weights=None, weights_backbone=None)),
        ("fcos", lambda: torchvision.models.detection.fcos_resnet50_fpn(weights=None, weights_backbone=None)),
        ("fasterrcnn_mbv3", lambda: torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(weights=None, weights_backbone=None)),
    ):
        try:
            model = factory().eval()
        except Exception as exc:  # noqa: BLE001
            print(f"  {name}: build failed ({type(exc).__name__}: {str(exc)[:90]})")
            continue
        results[name] = export(model, [torch.randn(3, 320, 320)], name)

    # Dense prediction: an FPN-shaped decoder, which is where `Resize` and `Concat` live.
    # YOLO's own head is the same two operators plus a sigmoid; the full detectors above
    # refuse to export because of their NMS post-processing, which is not a kernel.
    results["deeplabv3_mbv3"] = export(
        torchvision.models.segmentation.deeplabv3_mobilenet_v3_large(weights=None, weights_backbone=None).eval(),
        x, "deeplabv3_mbv3")

    encoder_layer = nn.TransformerEncoderLayer(d_model=64, nhead=4, batch_first=True)
    results["transformer"] = export(nn.TransformerEncoder(encoder_layer, num_layers=2).eval(),
                                    torch.randn(1, 16, 64), "transformer")

    results["lstm"] = export(nn.LSTM(input_size=32, hidden_size=64, num_layers=2, batch_first=True).eval(),
                             torch.randn(1, 12, 32), "lstm")

    used = collections.Counter()
    for kinds in results.values():
        if kinds:
            used.update(kinds)
    models = [name for name, kinds in results.items() if kinds]

    print(f"\nexported: {models}\n")
    print(f"{'op':34s} {'models':6s} {'ours':22s}")
    print("-" * 66)
    missing = []
    for op, count in sorted(used.items(), key=lambda kv: (-kv[1], kv[0])):
        who = sum(1 for kinds in results.values() if kinds and op in kinds)
        target = SERVED_BY.get(op)
        activation = SERVED_BY_ACTIVATION.get(op)
        if activation is not None:
            mark = f"have: fused {activation}" if activation in ACTIVATIONS else "-- MISSING"
        elif target is not None:
            mark = f"have: {target}" if target in HAVE else f"-- MISSING ({target})"
        else:
            mark = "plumbing" if op in PLUMBING else "-- MISSING"
        if mark.startswith("-- MISSING"):
            missing.append(op)
        print(f"{op:34s} {who:^6d} {mark:22s}")
    print("\nmissing kernels:", ", ".join(sorted(missing)) if missing else "none")


if __name__ == "__main__":
    main()
