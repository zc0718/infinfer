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

    if args.mode == "dynamic":
        quantize_dynamic(args.onnx, str(quantised), weight_type=QuantType.QUInt8)
    else:
        class Calibration(CalibrationDataReader):
            """`--calib` synthetic samples, none of which is the evaluation input."""

            def __init__(self):
                self._queue = [sample(seed + 1) for seed in range(args.calib)]

            def get_next(self):
                return None if not self._queue else {name: self._queue.pop(0)}

        quantize_static(args.onnx, str(quantised), Calibration(),
                        quant_format=QuantFormat.QDQ, per_channel=True,
                        activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8)

    q = ort.InferenceSession(str(quantised), providers=["CPUExecutionProvider"])
    got = reference(q, feed)

    diff = np.abs(got.astype(np.float64) - want.astype(np.float64))
    scale = np.abs(want.astype(np.float64)).max() + 1e-12
    fp32_bytes = pathlib.Path(args.onnx).stat().st_size
    # torch's exporter keeps the weights in a sidecar `.data` file; both halves are the model.
    sidecar = pathlib.Path(args.onnx).with_suffix(".onnx.data")
    if sidecar.is_file():
        fp32_bytes += sidecar.stat().st_size
    q_bytes = quantised.stat().st_size + (quantised.with_suffix(".onnx.data").stat().st_size
                                          if quantised.with_suffix(".onnx.data").is_file() else 0)

    report = {
        "mode": args.mode,
        "calibration_samples": args.calib if args.mode == "static" else 0,
        "fp32_bytes": fp32_bytes,
        "int8_bytes": q_bytes,
        "ratio": round(fp32_bytes / q_bytes, 2) if q_bytes else None,
        "output_elements": int(want.size),
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "max_rel_to_range": float(diff.max() / scale),
        "correlation": float(np.corrcoef(want.astype(np.float64), got.astype(np.float64))[0, 1]),
        "caveat": (
            "the SIZES are facts about the model and can be quoted. The ERROR is not an accuracy "
            "claim: calibration here is a handful of synthetic ramps, and int8 activation ranges "
            "estimated from a ramp have nothing to do with the ranges of real images. A large "
            "error says the calibration set is wrong, not that int8 is; a small one would have "
            "said nothing either. Measuring int8 accuracy needs calibration data and a held-out "
            "set, which this repository does not have (gaps G-110)"),
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
