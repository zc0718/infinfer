#!/usr/bin/env python3
"""Emit the `k1_mlp3` KAT blob, its input and its expectation (PLAN P1.3).

Generation lives here, verification lives in `test_package/test/unit/
graph_kat_k1_mlp3.cpp` (PLAN §3.10: generation is a maintainer step, CI only
reads the bytes). The weights are the same integer-LCG stream the QEMU KAT chain
compiles in, so the host blob and the cross-target chain describe one model.

The expectation is computed **independently of the C code** -- numpy float32
matmul, a different summation order -- so agreement means the kernel is right,
not that the code equals itself. Bitwise cross-target equality is a separate
judgement (see `.github/project/qa/kat/run_kat.py`); here the tolerance is the plan's
FP32 window.

The blob is the `INFRKAT` **v2 graph blob**: weights *and* the graph live in the file,
and the format has exactly one definition (`blobfmt.py`), imported by every producer
and every Python consumer. This model has no runtime slots at all -- its input is a
constant slot, which is the one shape the executor has to support without an `input`
node (a graph that needs no caller-provided data).
"""
import argparse
import hashlib
import json
import pathlib
import sys

import numpy as np

import blobfmt

SEED = 0x5EED
IN, HIDDEN, OUT = 16, 32, 4

# Slot order is the on-disk order and the source of `manifest.json`'s graph, so the
# blob and the prose cannot disagree.
SLOTS = (
    ("x", (1, IN), True),
    ("h", (1, HIDDEN), False),
    ("y", (1, OUT), False),
    ("w1", (HIDDEN, IN), True),
    ("w2", (OUT, HIDDEN), True),
)
NODES = (
    {"op": "fc", "in": "x", "out": "h", "w": "w1", "act": "relu"},
    {"op": "fc", "in": "h", "out": "y", "w": "w2"},
)


def describe(node) -> dict:
    """The same node, in the words a human reads in `manifest.json`."""
    entry = {"op": node["op"]}
    if "act" in node:
        entry["act"] = node["act"]
    return entry


def lcg_value(seed: int) -> np.float32:
    """Same integer stream as kat_chain_k1.c; one exact int->float division."""
    state = (seed * 1664525 + 1013904223) & 0xFFFFFFFF
    centred = ((state >> 8) % 2001) - 1000
    return np.float32(centred) / np.float32(1000.0)


def weights():
    x = np.array([lcg_value(SEED + i) for i in range(IN)], dtype=np.float32)
    w1 = np.array(
        [[lcg_value(SEED + 100 + row * IN + i) for i in range(IN)] for row in range(HIDDEN)],
        dtype=np.float32,
    )
    w2 = np.array(
        [[lcg_value(SEED + 2000 + row * HIDDEN + i) for i in range(HIDDEN)] for row in range(OUT)],
        dtype=np.float32,
    )
    return x, w1, w2


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out", help="output directory (test_package/resources/kat/k1_mlp3)")
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    x, w1, w2 = weights()
    hidden = np.maximum(x @ w1.T, np.float32(0.0)).astype(np.float32)
    y2 = (hidden @ w2.T).astype(np.float32)

    payloads = {"x": x.tobytes(), "w1": w1.tobytes(), "w2": w2.tobytes()}
    slot_specs = [(name, dims, payloads[name] if is_constant else b"")
                  for name, dims, is_constant in SLOTS]
    blob = blobfmt.build(slot_specs, NODES)

    files = {
        "model.blob": blob,
        "input.bin": x.tobytes(),
        "expect_fp32.bin": y2.tobytes(),
    }
    for name, data in files.items():
        (out / name).write_bytes(data)

    manifest = {
        "id": "k1_mlp3",
        "source": "hand-written integer-LCG weights (no closed-source tool)",
        "built_with": f"numpy {np.__version__} float32",
        "subset": ["fc", "relu"],
        "graph": [describe(node) for node in NODES],
        "input": {"file": "input.bin", "dtype": "fp32", "shape": [1, IN],
                  "sha256": hashlib.sha256(files["input.bin"]).hexdigest()},
        "blob": {"file": "model.blob",
                 "sha256": hashlib.sha256(blob).hexdigest()},
        "expect": {"file": "expect_fp32.bin", "dtype": "fp32", "shape": [1, OUT],
                   "sha256": hashlib.sha256(files["expect_fp32.bin"]).hexdigest()},
        "tolerance": {"kind": "float", "max_rel": 1e-5, "max_abs": 1e-6},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"[emit_k1_blob] wrote {out} (blob {len(blob)} B, expect {y2.tobytes()[:4].hex()}...)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
