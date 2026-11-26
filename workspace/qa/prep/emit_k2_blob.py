#!/usr/bin/env python3
"""Emit the `k2_conv_micro` KAT blob, its input and its expectation (PLAN P1.3).

The graph is the plan's smallest one that touches every `composed` operator:

    input  [1,16,16,1]
    conv3x3(8)   pad=1 stride=1   -> [1,16,16,8]
    relu
    maxpool 2/2                   -> [1,8,8,8]
    dwconv3x3(16) pad=1 groups=8  -> [1,8,8,16]     (channel multiplier 2)
    relu
    mean over H,W                 -> [1,1,1,16]     (global average pool)
    fc 16 -> 10                   -> [1,10]

Generation lives here and verification lives in
`test_package/test/unit/graph_kat_k2_conv_micro.cpp` (PLAN §3.10: generation is a
maintainer step, CI only reads the bytes).

The expectation is computed **independently of the C code** -- the definition is
transcribed in float64 with explicit loops, i.e. neither the same formulas as the
kernels nor the same summation order -- so agreement means the operators are
right, not that the code equals itself.

Blob format v1 (little endian), identical to `k1_mlp3`:

    "INFRKAT\\0"          8 bytes
    u32 version = 1
    u32 tensor_count
    per tensor:
        char name[16]     NUL padded
        u32 dtype         1 = F32 (mirrors infr_dtype)
        u32 byte_len
        bytes payload
"""
import argparse
import hashlib
import json
import pathlib
import struct
import sys

import numpy as np

SEED = 0x2C0B
MAGIC = b"INFRKAT\0"
VERSION = 1
DTYPE_F32 = 1

IN_H = IN_W = 16
IN_C = 1
CONV_K = 3
CONV_OC = 8
POOL_K = 2
POOL_STRIDE = 2
DW_K = 3
DW_OC = 16
FC_IN = 16
FC_OC = 10


def lcg_value(seed: int) -> np.float32:
    """Same integer stream as the other KAT generators; one exact int->float division."""
    state = (seed * 1664525 + 1013904223) & 0xFFFFFFFF
    centred = ((state >> 8) % 2001) - 1000
    return np.float32(centred) / np.float32(1000.0)


def make_weights():
    def block(name_seed: int, count: int) -> np.ndarray:
        return np.array([lcg_value(name_seed + i) for i in range(count)], dtype=np.float32)

    return {
        "conv_w": block(SEED + 1, CONV_OC * CONV_K * CONV_K * IN_C).reshape(CONV_OC, CONV_K, CONV_K, IN_C),
        "conv_b": block(SEED + 500, CONV_OC),
        "dw_w": block(SEED + 1000, DW_OC * DW_K * DW_K).reshape(DW_OC, DW_K, DW_K, 1),
        "dw_b": block(SEED + 2000, DW_OC),
        "fc_w": block(SEED + 3000, FC_OC * FC_IN).reshape(FC_OC, FC_IN),
        "fc_b": block(SEED + 4000, FC_OC),
    }


def conv2d(x, w, b, pad: int, stride: int, groups: int):
    """Definition: sum over (ky, kx, ci) of x[iy, ix, ci] * w[oc, ky, kx, ci], zero outside."""
    h, wd, c = x.shape
    oc_all, kh, kw, cg = w.shape
    oh = (h + 2 * pad - kh) // stride + 1
    ow = (wd + 2 * pad - kw) // stride + 1
    out = np.zeros((oh, ow, oc_all), dtype=np.float64)
    per_group = oc_all // groups
    for oy in range(oh):
        for ox in range(ow):
            for oc in range(oc_all):
                group = oc // per_group
                acc = 0.0
                for ky in range(kh):
                    iy = oy * stride + ky - pad
                    if iy < 0 or iy >= h:
                        continue
                    for kx in range(kw):
                        ix = ox * stride + kx - pad
                        if ix < 0 or ix >= wd:
                            continue
                        for ci in range(cg):
                            acc += float(x[iy, ix, group * cg + ci]) * float(w[oc, ky, kx, ci])
                out[oy, ox, oc] = acc + float(b[oc])
    return out


def relu(x):
    return np.maximum(x, 0.0)


def max_pool(x, k: int, stride: int):
    h, wd, c = x.shape
    oh = (h - k) // stride + 1
    ow = (wd - k) // stride + 1
    out = np.full((oh, ow, c), -np.inf, dtype=np.float64)
    for oy in range(oh):
        for ox in range(ow):
            window = x[oy * stride:oy * stride + k, ox * stride:ox * stride + k, :]
            out[oy, ox, :] = window.max(axis=(0, 1))
    return out


def mean_hw(x):
    return x.mean(axis=(0, 1), keepdims=True)


def fully_connected(x, w, b):
    flat = x.reshape(-1).astype(np.float64)
    return (w.astype(np.float64) @ flat) + b.astype(np.float64)


def tensor(name: str, dtype: int, payload: bytes) -> bytes:
    raw = name.encode("ascii")
    if len(raw) > 15:
        raise SystemExit(f"emit_k2_blob: tensor name too long: {name}")
    return raw.ljust(16, b"\0") + struct.pack("<II", dtype, len(payload)) + payload


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out", help="output directory (test_package/resources/kat/k2_conv_micro)")
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    weights = make_weights()
    x = np.array([lcg_value(SEED + 9000 + i) for i in range(IN_H * IN_W * IN_C)], dtype=np.float32)
    x = x.reshape(1, IN_H, IN_W, IN_C)

    # Explicit pipeline, in float64, from the definition of each operator.
    h1 = relu(conv2d(x[0], weights["conv_w"], weights["conv_b"], pad=1, stride=1, groups=1))
    h2 = max_pool(h1, POOL_K, POOL_STRIDE)
    h3 = relu(conv2d(h2, weights["dw_w"], weights["dw_b"], pad=1, stride=1, groups=CONV_OC))
    h4 = mean_hw(h3)
    y = fully_connected(h4, weights["fc_w"], weights["fc_b"]).astype(np.float32)

    order = ("conv_w", "conv_b", "dw_w", "dw_b", "fc_w", "fc_b")
    payloads = [tensor(name, DTYPE_F32, weights[name].tobytes()) for name in order]
    blob = MAGIC + struct.pack("<II", VERSION, len(payloads)) + b"".join(payloads)

    files = {
        "model.blob": blob,
        "input.bin": x.tobytes(),
        "expect_fp32.bin": y.tobytes(),
    }
    for name, data in files.items():
        (out / name).write_bytes(data)

    manifest = {
        "id": "k2_conv_micro",
        "source": "hand-written integer-LCG weights (no closed-source tool)",
        "built_with": f"numpy {np.__version__} float64 reference",
        "subset": ["conv2d", "relu", "max_pool2d", "dwconv2d", "mean", "fc"],
        "graph": [
            {"op": "conv2d", "kernel": 3, "stride": 1, "pad": 1, "out_channels": CONV_OC, "groups": 1},
            {"op": "relu"},
            {"op": "max_pool2d", "kernel": POOL_K, "stride": POOL_STRIDE, "pad": 0},
            {"op": "conv2d", "kernel": 3, "stride": 1, "pad": 1, "out_channels": DW_OC, "groups": CONV_OC},
            {"op": "relu"},
            {"op": "mean"},
            {"op": "fc", "out_channels": FC_OC},
        ],
        "weights": {"order": list(order), "dtype": "fp32"},
        "input": {"file": "input.bin", "dtype": "fp32", "shape": [1, IN_H, IN_W, IN_C],
                  "sha256": hashlib.sha256(files["input.bin"]).hexdigest()},
        "blob": {"file": "model.blob", "sha256": hashlib.sha256(blob).hexdigest()},
        "expect": {"file": "expect_fp32.bin", "dtype": "fp32", "shape": [1, FC_OC],
                   "sha256": hashlib.sha256(files["expect_fp32.bin"]).hexdigest()},
        "tolerance": {"kind": "float", "max_rel": 1e-5, "max_abs": 1e-6},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"[emit_k2_blob] wrote {out} (blob {len(blob)} B, expect[0] = {y.tobytes()[:4].hex()})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
