#!/usr/bin/env python3
"""Emit the `k1_mlp3` KAT blob, its input and its expectation (PLAN P1.3).

Generation lives here, verification lives in `test_package/test/unit/
graph_kat_k1_mlp3.cpp` (PLAN §3.10: generation is a maintainer step, CI only
reads the bytes). The weights are the same integer-LCG stream the QEMU KAT chain
compiles in, so the host blob and the cross-target chain describe one model.

The expectation is computed **independently of the C code** -- numpy float32
matmul, a different summation order -- so agreement means the kernel is right,
not that the code equals itself. Bitwise cross-target equality is a separate
judgement (see `workspace/qa/kat/run_kat.py`); here the tolerance is the plan's
FP32 window.

Blob format v1 (little endian), documented so the reader is not reverse
engineered:

    "INFRKAT\\0"          8 bytes
    u32 version = 1
    u32 tensor_count
    per tensor:
        char name[16]     NUL padded
        u32 dtype         1 = F32, 2 = I8 (mirrors infr_dtype)
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

SEED = 0x5EED
IN, HIDDEN, OUT = 16, 32, 4
MAGIC = b"INFRKAT\0"
VERSION = 1
DTYPE_F32 = 1


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


def tensor(name: str, dtype: int, payload: bytes) -> bytes:
    raw = name.encode("ascii")
    if len(raw) > 15:
        raise SystemExit(f"emit_k1_blob: tensor name too long: {name}")
    return raw.ljust(16, b"\0") + struct.pack("<II", dtype, len(payload)) + payload


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out", help="output directory (test_package/resources/kat/k1_mlp3)")
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    x, w1, w2 = weights()
    hidden = np.maximum(x @ w1.T, np.float32(0.0)).astype(np.float32)
    y2 = (hidden @ w2.T).astype(np.float32)

    tensors = [
        tensor("x", DTYPE_F32, x.tobytes()),
        tensor("w1", DTYPE_F32, w1.tobytes()),
        tensor("w2", DTYPE_F32, w2.tobytes()),
    ]
    blob = MAGIC + struct.pack("<II", VERSION, len(tensors)) + b"".join(tensors)

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
