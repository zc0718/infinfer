#!/usr/bin/env python3
"""Measure what int8 post-training quantisation buys, and what it costs, before writing any of it
into this library.

The question PTQ has to answer here is not "can a quantiser run". It is: **how much smaller, and
how much worse, on the model this project actually ships?** Those two numbers decide whether an
int8 `conv2d` is worth building, and they are cheap to get: `onnxruntime.quantization` already
produces a QDQ model, so this script quantises the same ONNX the importer accepts, runs both
through onnxruntime on the same input, and reports the difference.

It deliberately does **not** translate the quantised model into a blob. A QDQ graph is full of
`QuantizeLinear` / `DequantizeLinear` / `ConvInteger` nodes, and this library's importer refuses
them by name -- that refusal is the finding, not a bug to work around: an int8 path needs an int8
convolution with per-channel requantisation, which the runtime does not have (gaps G-110).

Machine-level, like the exporter: `onnx` and `onnxruntime` are not dependencies of the library.

    python3 .github/project/qa/prep/survey_ptq.py --onnx /tmp/real/model.onnx --input /tmp/real/input.bin
"""
import argparse
import json
import pathlib
import sys
import tempfile


def reference(session, feed):
    import numpy as np

    name = session.get_inputs()[0].name
    (out,) = session.run(None, {name: feed})
    return np.asarray(out, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--onnx", required=True, help="the fp32 model the importer accepts")
    parser.add_argument("--input", required=True,
                        help="fixed input in blob order (NHWC); it is transposed back for ORT")
    parser.add_argument("--mode", choices=("dynamic", "static"), default="dynamic",
                        help="static needs calibration data; the fixed input doubles as it")
    parser.add_argument("--calib", type=int, default=1,
                        help="static only: how many synthetic samples to calibrate on. 1 is what a "
                             "survey does when it has no data; the error that produces is a "
                             "statement about the calibration set, not about int8")
    parser.add_argument("--images",
                        help="comma-separated real images: calibrate on all but the last, evaluate "
                             "on the last. Real photographs are the only way this survey can say "
                             "anything about int8 accuracy; a ramp cannot")
    parser.add_argument("--size", type=int, default=None,
                        help="side of the square the model wants; defaults to the model's own height")
    args = parser.parse_args()

    try:
        import numpy as np
        import onnx
        import onnxruntime as ort
        from onnxruntime.quantization import (CalibrationDataReader, QuantFormat, QuantType,
                                              quantize_dynamic, quantize_static)
    except ImportError as error:
        print(f"[ptq] needs onnx + onnxruntime ({error})", file=sys.stderr)
        return 1

    model = onnx.load(args.onnx)
    n, c, h, w = [d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim]
    blob_order = np.fromfile(args.input, dtype="<f4")
    if blob_order.size != n * c * h * w:
        print(f"[ptq] {args.input} holds {blob_order.size} float, the model wants {n*c*h*w}",
              file=sys.stderr)
        return 1
    # The file is NHWC (the blob's layout); onnxruntime wants the model's own NCHW.
    feed = np.transpose(blob_order.reshape(n, h, w, c), (0, 3, 1, 2)).copy()
    name = model.graph.input[0].name

    fp32 = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    want = reference(fp32, feed)

    out_dir = pathlib.Path(tempfile.mkdtemp(prefix="ptq-"))
    quantised = out_dir / f"model_{args.mode}_u8.onnx"

    def sample(seed):
        """A ramp like the fixture's, offset per seed. Held out from the evaluation sample: a
        model calibrated on the data it is measured on reports its own memory."""
        ramp = (np.arange(feed.size, dtype=np.float32) / 16.0 - 1.0) * (1.0 + 0.125 * seed)
        return np.transpose(ramp.reshape(n, h, w, c), (0, 3, 1, 2)).copy()

    def image_sample(path):
        """A real photograph in the model's input layout.

        Grayscale is replicated across the three channels when the model wants three and the file
        has one: that keeps the *content* real (edges, scale, contrast) even though the colour is
        not. Values are mapped to [-1, 1) per channel, which is the same convention the synthetic
        fixture uses.
        """
        from PIL import Image

        side = args.size or h
        with Image.open(path) as handle:
            picture = handle.convert("L" if c == 1 else "RGB").resize((side, side))
        pixels = np.asarray(picture, dtype=np.float32) / 127.5 - 1.0
        if c == 1:
            array = pixels[:, :, None]
        elif pixels.ndim == 3:
            array = pixels                       # already RGB: repeating it would give 9 channels
        else:
            array = np.repeat(pixels[:, :, None], c, axis=2)
        return np.transpose(array, (2, 0, 1))[None, ...].copy()

    if args.images:
        paths = [entry.strip() for entry in args.images.split(",") if entry.strip()]
        if len(paths) < 2:
            print("[ptq] --images needs at least two files: one to calibrate on, one held out",
                  file=sys.stderr)
            return 1
        calibration = [image_sample(entry) for entry in paths[:-1]]
        feed = image_sample(paths[-1])
        print(f"[ptq] calibrating on {len(calibration)} image(s), evaluating on {paths[-1]}")
    else:
        calibration = [sample(seed + 1) for seed in range(args.calib)]

    # The fp32 baseline is taken *after* the input is final: an image run and a ramp run must not
    # share a reference, or a file-order mistake looks like a quantisation error.
    # fp32 基线在输入定下来之后才算：图像运行与斜坡运行不能共用一份参考，否则一个文件顺序的
    # 错误会看起来像量化误差。
    want = reference(ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"]), feed)

    if args.mode == "dynamic":
        quantize_dynamic(args.onnx, str(quantised), weight_type=QuantType.QUInt8)
    else:
        class Calibration(CalibrationDataReader):
            """`--calib` synthetic samples, none of which is the evaluation input."""

            def __init__(self):
                self._queue = list(calibration)

            def get_next(self):
                return None if not self._queue else {name: self._queue.pop(0)}

        quantize_static(args.onnx, str(quantised), Calibration(),
                        quant_format=QuantFormat.QDQ, per_channel=True,
                        activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8)

    q = ort.InferenceSession(str(quantised), providers=["CPUExecutionProvider"])
    got = reference(q, feed)

    # Both sides are flattened first: ORT returns the model's own shape (1, 1024) for a feature
    # model, and a 2-D dot product is not the cosine of two vectors.
    want = want.reshape(-1)
    got = got.reshape(-1)
    diff = np.abs(got.astype(np.float64) - want.astype(np.float64))
    scale = np.abs(want.astype(np.float64)).max() + 1e-12
    # A feature vector is mostly near-zero with a few large channels, so correlation alone is a
    # misleading summary: it is dominated by the near-zero ones. Cosine similarity asks whether the
    # *direction* survived, which is what a downstream layer actually consumes.
    # 特征向量里大多数通道接近 0、少数很大，所以单看相关系数会误导：它被近零的那批主导。余弦相似度
    # 问的是“方向”有没有保住，而那正是下一层真正消费的东西。
    cosine = float(
        np.dot(want.astype(np.float64), got.astype(np.float64))
        / (np.linalg.norm(want.astype(np.float64)) * np.linalg.norm(got.astype(np.float64)) + 1e-12))
    fp32_bytes = pathlib.Path(args.onnx).stat().st_size
    # torch's exporter keeps the weights in a sidecar `.data` file; both halves are the model.
    sidecar = pathlib.Path(args.onnx).with_suffix(".onnx.data")
    if sidecar.is_file():
        fp32_bytes += sidecar.stat().st_size
    q_bytes = quantised.stat().st_size + (quantised.with_suffix(".onnx.data").stat().st_size
                                          if quantised.with_suffix(".onnx.data").is_file() else 0)

    report = {
        "mode": args.mode,
        "calibration": "images" if args.images else f"ramps({args.calib})",
        "calibration_samples": len(calibration) if args.mode == "static" else 0,
        "fp32_bytes": fp32_bytes,
        "int8_bytes": q_bytes,
        "ratio": round(fp32_bytes / q_bytes, 2) if q_bytes else None,
        "output_elements": int(want.size),
        "fp32_absmax": float(np.abs(want).max()),
        "fp32_rms": float(np.sqrt(np.mean(np.square(want.astype(np.float64))))),
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "max_rel_to_range": float(diff.max() / scale),
        "cosine": cosine,
        "correlation": float(np.corrcoef(want.astype(np.float64), got.astype(np.float64))[0, 1]),
        "caveat": (
            "the SIZES are facts about the model and can be quoted. The ERROR is only an accuracy "
            "claim when it was calibrated and evaluated on real photographs -- and even then it "
            "is one held-out sample, so it bounds the mechanism rather than measuring the model. "
            "With the synthetic ramp it is a statement about the ramp: a large error says the "
            "calibration set is wrong, not that int8 is (gaps G-110)"),
    }
    print(json.dumps(report, indent=2))

    # And the part that decides the next increment: this model is not in the subset.
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    import onnx_subset

    problems = onnx_subset.check(onnx.load(str(quantised)))
    named = sorted({reason.split(" is not in the subset")[0] for _node, reason in problems})
    print(f"[ptq] the quantised graph is {'INSIDE' if not problems else 'OUTSIDE'} the subset "
          f"({len(problems)} refusal(s))")
    for line in named[:6]:
        print(f"[ptq]   {line}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
