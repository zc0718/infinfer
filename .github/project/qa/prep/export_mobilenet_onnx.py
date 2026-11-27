#!/usr/bin/env python3
"""Export a real MobileNetV3 from PyTorch to ONNX, with the reference tensors the blob is
checked against.

This is the other half of `emit_onnx_blob.py`: that one takes an ONNX file and produces a blob,
this one produces an ONNX file from a model somebody else wrote. Together they answer a question
the synthetic fixture cannot -- "does this library run a network we did not design" -- and the
reference output comes from **PyTorch**, not from our own arithmetic, so agreement means two
independent implementations agree.

**Machine-level dependency, on purpose.** `torch` and `torchvision` are not in `conanfile.py` and
never will be: they exist to *author* a model, and what the repository ships is the resulting
ONNX file and the tensors. A clone without torch can still run every test; it just cannot
re-export. The requirement is recorded in `toolchains.json:heavy_checks` like every other check
CI cannot run.

    机器级依赖，故意的。`torch` / `torchvision` 不进 `conanfile.py`，也永远不会：它们的用途是
    **生产**模型，而仓库交付的是产出的 ONNX 与张量。没有 torch 的 clone 照样能跑全部测试，只是
    不能重新导出。这条依赖与其它 CI 跑不了的重活一样，登记在 `toolchains.json:heavy_checks`。

What it writes, and why each file exists:

    model.onnx    the graph, opset 13, static shapes (the blob format has no dynamic dims)
    input.bin     the fixed input, **in blob order (NHWC)** -- the same tensor the executor gets
    expect.bin    PyTorch's own output for that input, float32 little-endian
    export.json   what was exported: model, weights, resolution, shapes, and torch's versions

Weights default to **seeded random**: the shapes, and therefore every size the memory budget
cares about, are identical to pretrained, and a random model needs no download. `--pretrained`
switches to the real checkpoints for accuracy work, and `export.json` records which one was used
so a number can never be traced to the wrong weights.

Usage:

    python3 .github/project/qa/prep/export_mobilenet_onnx.py --out /tmp/mbv3
    python3 .github/project/qa/prep/export_mobilenet_onnx.py --pretrained --res 96 --out /tmp/mbv3
"""
import argparse
import json
import pathlib
import sys


def torch_version_report():
    import torch
    import torchvision

    return {"torch": torch.__version__, "torchvision": torchvision.__version__}


def fixed_input(shape, seed=0):
    """A deterministic ramp, exactly representable in float32.

    Not an image: the point of this export is that two implementations agree on *the same
    numbers*, and a ramp makes a transposition or an off-by-one channel visible instead of
    plausible.
    """
    import torch

    count = 1
    for axis in shape:
        count *= axis
    ramp = (torch.arange(count, dtype=torch.float32) / 16.0 - 1.0) * (seed * 0.0 + 1.0)
    return ramp.reshape(*shape)


def build_model(name, pretrained):
    import torch
    import torchvision

    factory = getattr(torchvision.models, name, None)
    if factory is None:
        raise SystemExit(f"export_mobilenet_onnx: torchvision has no model '{name}'")
    weights = "DEFAULT" if pretrained else None
    model = factory(weights=weights)
    model.eval()
    return model


def export(args):
    import numpy as np
    import torch

    if args.seed is not None:
        torch.manual_seed(args.seed)

    model = build_model(args.model, args.pretrained)
    shape = [1, 3, args.res, args.res]
    x = fixed_input(shape)
    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    onnx_path = out_dir / "model.onnx"
    torch.onnx.export(
        model, x, str(onnx_path),
        opset_version=13,
        input_names=["x"], output_names=["y"],
        do_constant_folding=True,
        dynamic_axes=None,
        # Static shapes: the blob format has no dynamic dims, so an export that leaves a
        # symbolic axis would be refused later by the importer for a reason that belongs here.
        export_params=True,
    )

    with torch.no_grad():
        reference = model(x)
    # The executor reads NHWC, the same convention `emit_onnx_blob.py --emit-fixture` writes.
    (out_dir / "input.bin").write_bytes(
        np.ascontiguousarray(x.permute(0, 2, 3, 1).numpy(), dtype="<f4").tobytes())
    (out_dir / "expect.bin").write_bytes(
        np.ascontiguousarray(reference.numpy(), dtype="<f4").tobytes())

    report = {
        "model": args.model,
        "weights": "pretrained" if args.pretrained else f"random(seed={args.seed})",
        "resolution": args.res,
        "input_nchw": shape,
        "input_order_in_file": "NHWC",
        "output": list(reference.shape),
        "params": sum(p.numel() for p in model.parameters()),
        "onnx_bytes": onnx_path.stat().st_size,
        "versions": torch_version_report(),
    }
    (out_dir / "export.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"[export] {args.model} {shape} -> {onnx_path} ({report['onnx_bytes']} B, "
          f"{report['params']} params, {report['weights']})")
    print(f"[export] torch reference: {reference.flatten()[:4].tolist()} ...")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="mobilenet_v3_small")
    parser.add_argument("--res", type=int, default=96,
                        help="square input resolution; the graph is fully convolutional")
    parser.add_argument("--seed", type=int, default=0,
                        help="seed for the random weights (ignored with --pretrained)")
    parser.add_argument("--pretrained", action="store_true",
                        help="download the real checkpoints; needs the network and is not "
                             "required for any numeric check")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    try:
        import torch  # noqa: F401
    except ImportError:
        print("[export] torch is not installed. This is a machine-level dependency on purpose "
              "(see this file's header): pip install --index-url "
              "https://download.pytorch.org/whl/cpu torch torchvision", file=sys.stderr)
        return 1
    return export(args)


if __name__ == "__main__":
    sys.exit(main())
