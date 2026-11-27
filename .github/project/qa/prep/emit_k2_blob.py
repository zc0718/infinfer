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

Graph blob format v2 (little endian):

    "INFRKAT\\0"          8 bytes
    u32 version = 2
    u32 node_count
    u32 slot_count
    u32 reserved = 0
    per node (36 bytes):
        u8  op            0=input 1=conv2d 2=max_pool2d 3=mean_hw 4=fc
        u8  act           infr_activation applied to the output, 0 = none
        u16 reserved = 0
        u32 in_slot, out_slot, w_slot, b_slot     (0xFFFFFFFF = none)
        u32 p[4]          conv2d {kernel, stride, pad, groups},
                          max_pool2d {kernel, stride, pad, -}
    per slot:
        char name[16]     NUL padded
        u32 dtype         1 = F32 (mirrors infr_dtype)
        u32 rank
        u32 dims[4]
        u32 byte_len      0 = runtime buffer, >0 = constant payload
        bytes payload

v1 carried only the tensor table, so the graph had to be spelled out a second time in
C++ (and a third time in prose in `manifest.json`, which nothing read). With
`k2_conv_micro` promoted from fixture to consumer case (PLAN §2.2.1 / P4.5) the graph
has to be **data**, otherwise every new architecture means another hand-written
runner. The v1 reader is dropped on purpose: two versions of one format drift.
"""
import argparse
import hashlib
import json
import pathlib
import sys

import numpy as np

import blobfmt

SEED = 0x2C0B

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


# The graph, as data. Slot order is the on-disk order and is also what `manifest.json`
# is generated from, so the blob and the prose cannot disagree.
SLOTS = (
    ("input", (1, IN_H, IN_W, IN_C), False),
    ("conv", (1, IN_H, IN_W, CONV_OC), False),
    ("pool", (1, IN_H // 2, IN_W // 2, CONV_OC), False),
    ("dw", (1, IN_H // 2, IN_W // 2, DW_OC), False),
    ("gap", (1, 1, 1, DW_OC), False),
    ("out", (1, 1, 1, FC_OC), False),
    ("conv_w", (CONV_OC, CONV_K, CONV_K, IN_C), True),
    ("conv_b", (CONV_OC,), True),
    ("dw_w", (DW_OC, DW_K, DW_K, 1), True),
    ("dw_b", (DW_OC,), True),
    ("fc_w", (FC_OC, FC_IN), True),
    ("fc_b", (FC_OC,), True),
)
NODES = (
    {"op": "input", "out": "input"},
    {"op": "conv2d", "in": "input", "out": "conv", "w": "conv_w", "b": "conv_b",
     "p": (CONV_K, 1, 1, 1), "act": "relu"},
    {"op": "max_pool2d", "in": "conv", "out": "pool", "p": (POOL_K, POOL_STRIDE, 0, 0)},
    {"op": "conv2d", "in": "pool", "out": "dw", "w": "dw_w", "b": "dw_b",
     "p": (DW_K, 1, 1, CONV_OC), "act": "relu"},
    {"op": "mean_hw", "in": "dw", "out": "gap"},
    {"op": "fc", "in": "gap", "out": "out", "w": "fc_w", "b": "fc_b"},
)


def describe(node) -> dict:
    """The same node, in the words a human reads in `manifest.json`."""
    if node["op"] == "conv2d":
        kernel, stride, pad, groups = node["p"]
        return {"op": "conv2d", "kernel": kernel, "stride": stride, "pad": pad,
                "groups": groups, "act": node.get("act", "none")}
    if node["op"] == "max_pool2d":
        kernel, stride, pad, _ = node["p"]
        return {"op": "max_pool2d", "kernel": kernel, "stride": stride, "pad": pad}
    if node["op"] == "fc":
        return {"op": "fc", "out_channels": FC_OC}
    return {"op": node["op"]}


def subset_of(nodes) -> list:
    """The operator names this model needs, with `conv2d` split into dense/depthwise."""
    names = []
    for node in nodes:
        if node["op"] == "conv2d":
            names.append("dwconv2d" if node["p"][3] > 1 else "conv2d")
        elif node["op"] == "mean_hw":
            names.append("mean")
        else:
            names.append(node["op"])
        act = node.get("act", "none")
        if act != "none" and act not in names:
            names.append(act)
    return names


def slot(name: str, dims, payload: bytes) -> bytes:
    return blobfmt.slot(name, dims, payload)





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

    slot_specs = [(name, dims, weights[name].tobytes() if is_weight else b"")
                  for name, dims, is_weight in SLOTS]
    blob = blobfmt.build(slot_specs, NODES)

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
        "subset": subset_of(NODES),
        "graph": [describe(node) for node in NODES],
        "weights": {"order": [name for name, _, is_weight in SLOTS if is_weight], "dtype": "fp32"},
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
