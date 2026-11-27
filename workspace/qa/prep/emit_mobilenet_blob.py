#!/usr/bin/env python3
"""Build a MobileNetV3-Small **graph blob** and report its memory budget.

Why this exists: §2.2.1 says the MobileNetV3 cell is the next coverage win, and the
owner pointed out that IE already ran a MobileNetV3 on an MCU -- in f32, with the
weights resident in one block. So the question is not "can the operators express it"
(they can: `conv2d` with `groups` covers dense and depthwise, `mean_hw` is the global
pool, `fc` is a 1x1 conv on a 1x1 map, `mul` + `add` are the SE gate and the residual,
and `hard_swish`/`hard_sigmoid` are already fused activation codes), but **does it fit**.

So this script answers the fitting question first, before anything is compiled:

  * `weights_bytes` -- the blob's constant payload, which on a target lives in flash
    and is read in place (the executor never copies it into RAM);
  * `arena_naive`  -- every intermediate gets its own slot, i.e. the sum of all
    feature maps. This is what a slot table says literally, and it is what a naive
    executor would allocate;
  * `arena_liveness` -- the same activations with **first-fit reuse over live ranges**,
    which is what a real engine does with one memory block.

The gap between the last two is the whole point: it decides whether "MCU-scale" needs
tiling or just an allocator.

Weights are the integer-LCG stream the other KAT generators use, so the artifact is
deterministic and free of any closed-source tool; the **shape** of the network is the
paper's MobileNetV3-Small block list, the values are not a trained model. Swapping in
real weights later is P5.3's job and does not change the blob layout.

Usage:
    emit_mobilenet_blob.py --width 0.5 --res 160 --out workspace/qa/kat/build/mobilenet
"""
import argparse
import json
import pathlib
import sys

import numpy as np

import blobfmt

ROOT = pathlib.Path(__file__).resolve().parents[3]

# MobileNetV3-Small, the paper's block list: (kernel, expand, out, squeeze-excite, stride).
BLOCKS = (
    (3, 16, 16, True, 2),
    (3, 72, 24, False, 2),
    (3, 88, 24, False, 1),
    (5, 96, 40, True, 2),
    (5, 240, 40, True, 1),
    (5, 240, 40, True, 1),
    (5, 120, 48, True, 1),
    (5, 144, 48, True, 1),
    (5, 288, 96, True, 2),
    (5, 576, 96, True, 1),
    (5, 576, 96, True, 1),
)
STEM_OUT = 16
HEAD_OUT = 576
CLASSIFIER_OUT = 1024
SE_REDUCE = 4


def lcg_block(count: int, seed: int) -> np.ndarray:
    """The same integer stream as the other KAT generators, vectorised."""
    index = np.arange(count, dtype=np.int64)
    state = ((index + seed) * 1664525 + 1013904223) & 0xFFFFFFFF
    centred = ((state >> 8) % 2001) - 1000
    return (centred.astype(np.float32) / np.float32(1000.0))


class Builder:
    """Accumulates slots and nodes; every name is resolved to an index on emit."""

    def __init__(self):
        self.slots = []
        self.index = {}
        self.nodes = []
        self.seed = 1

    def runtime(self, name, dims):
        self.index[name] = len(self.slots)
        self.slots.append((name, dims, b""))
        return name

    def constant(self, name, dims):
        count = 1
        for axis in dims:
            count *= axis
        payload = lcg_block(count, self.seed).tobytes()
        self.seed += count
        self.index[name] = len(self.slots)
        self.slots.append((name, dims, payload))
        return name

    def node(self, op, out, src=None, w=None, b=None, params=(0, 0, 0, 0), act="none"):
        self.nodes.append({"op": op, "in": src, "out": out, "w": w, "b": b,
                           "p": params, "act": act})

    # -- layer helpers -------------------------------------------------------
    def conv(self, name, src, h, w, cin, cout, kernel, stride, act, groups=1):
        pad = kernel // 2
        oh = (h + 2 * pad - kernel) // stride + 1
        ow = (w + 2 * pad - kernel) // stride + 1
        weight = self.constant(f"{name}_w", (cout, kernel, kernel, cin // groups))
        bias = self.constant(f"{name}_b", (cout,))
        self.runtime(f"{name}_o", (1, oh, ow, cout))
        self.node("conv2d", f"{name}_o", src, weight, bias, (kernel, stride, pad, groups), act)
        return f"{name}_o", oh, ow, cout

    def global_pool(self, name, src, c):
        self.runtime(f"{name}_o", (1, 1, 1, c))
        self.node("mean_hw", f"{name}_o", src)
        return f"{name}_o"

    def dense(self, name, src, cin, cout, act):
        weight = self.constant(f"{name}_w", (cout, cin))
        bias = self.constant(f"{name}_b", (cout,))
        self.runtime(f"{name}_o", (1, 1, 1, cout))
        self.node("fc", f"{name}_o", src, weight, bias, act=act)
        return f"{name}_o"

    def binary(self, op, name, lhs, rhs, dims):
        self.runtime(f"{name}_o", dims)
        # A binary node carries its two operands in the slots a weightful node would use
        # for filter and bias (see blob_graph.hpp).
        self.node(op, f"{name}_o", lhs, rhs)
        return f"{name}_o"


def scaled(channels: int, width: float) -> int:
    """Width multiplier with the paper's rounding (nearest multiple of 8)."""
    return max(8, int(round(channels * width / 8.0)) * 8)


def build_linear(width: float, res: int, classes: int):
    """The block list, written straight through, keeping every block's input by name."""
    b = Builder()
    b.runtime("input", (1, res, res, 3))
    b.node("input", "input")

    stem_out = scaled(STEM_OUT, width)
    cur, h, w, c = b.conv("stem", "input", res, res, 3, stem_out, 3, 2, "hard_swish")

    for number, (kernel, expand, out, se, stride) in enumerate(BLOCKS):
        name = f"b{number}"
        exp_c = scaled(expand, width)
        out_c = scaled(out, width)
        block_in, block_h, block_w, block_c = cur, h, w, c

        if exp_c != c:
            cur, h, w, c = b.conv(f"{name}_exp", cur, h, w, c, exp_c, 1, 1, "hard_swish")
        cur, h2, w2, c = b.conv(f"{name}_dw", cur, h, w, c, c, kernel, stride, "hard_swish", groups=c)
        if se:
            reduced = max(8, c // SE_REDUCE)
            gate = b.global_pool(f"{name}_se_pool", cur, c)
            gate = b.dense(f"{name}_se_red", gate, c, reduced, "relu")
            gate = b.dense(f"{name}_se_exp", gate, reduced, c, "hard_sigmoid")
            cur = b.binary("mul", f"{name}_se_mul", cur, gate, (1, h2, w2, c))
        cur, h, w, c = b.conv(f"{name}_proj", cur, h2, w2, c, out_c, 1, 1, "none")
        if stride == 1 and block_c == c and block_h == h and block_w == w:
            cur = b.binary("add", f"{name}_res", cur, block_in, (1, h, w, c))

    head_in = scaled(HEAD_OUT, width)
    cur, h, w, c = b.conv("head", cur, h, w, c, head_in, 1, 1, "hard_swish")
    cur = b.global_pool("head_pool", cur, c)
    cur = b.dense("head_fc", cur, c, scaled(CLASSIFIER_OUT, width), "hard_swish")
    cur = b.dense("logits", cur, scaled(CLASSIFIER_OUT, width), classes, "none")
    return b, h, w


def plan_arena(slots, nodes):
    """Two numbers: the literal sum of runtime slots, and first-fit reuse over live ranges.

    `nodes` must already carry **indices**, not names -- the same encoding `blobfmt.build`
    consumes -- so the plan and the artifact cannot disagree about which slot a node uses.
    """
    runtime = [i for i, (_name, _dims, payload) in enumerate(slots) if not payload]
    size = {}
    for i, (_name, dims, _payload) in enumerate(slots):
        count = 1
        for axis in dims:
            count *= axis
        size[i] = count

    last_use = {}
    for position, node in enumerate(nodes):
        for ref in (node["in"], node["w"], node["b"]):
            if ref is not None and not slots[ref][2]:
                last_use[ref] = position
        last_use[node["out"]] = position

    free = [(0, None)]  # (offset, length) with None meaning "to the end"
    offsets = {}
    high_water = 0

    def take(count):
        nonlocal high_water
        for position, (offset, length) in enumerate(free):
            if length is None:
                # The tail block: always fits, and stays a tail block after the cut.
                free[position] = (offset + count, None)
                high_water = max(high_water, offset + count)
                return offset
            if length >= count:
                if length > count:
                    free[position] = (offset + count, length - count)
                else:
                    free.pop(position)
                high_water = max(high_water, offset + count)
                return offset
        raise SystemExit("emit_mobilenet_blob: allocator ran out of free list")

    for position, node in enumerate(nodes):
        offsets[node["out"]] = take(size[node["out"]])
        for index in sorted(i for i in runtime if last_use.get(i) == position):
            free.append((offsets[index], size[index]))
            free.sort()
    return sum(size[i] for i in runtime), high_water


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--width", type=float, default=0.5)
    ap.add_argument("--res", type=int, default=160)
    ap.add_argument("--classes", type=int, default=1000)
    ap.add_argument("--out", default="workspace/qa/kat/build/mobilenet",
                    help="output directory, relative to the repository root")
    ap.add_argument("--write", action="store_true", help="write model.blob and manifest.json")
    args = ap.parse_args()

    builder, h, w = build_linear(args.width, args.res, args.classes)
    nodes = [{**node,
              "in": None if node["in"] is None else builder.index[node["in"]],
              "out": builder.index[node["out"]],
              "w": None if node["w"] is None else builder.index[node["w"]],
              "b": None if node["b"] is None else builder.index[node["b"]]}
             for node in builder.nodes]

    weights = sum(len(payload) for _name, _dims, payload in builder.slots if payload)
    naive, reused = plan_arena(builder.slots, nodes)
    report = {
        "config": {"width": args.width, "res": args.res, "classes": args.classes},
        "nodes": len(builder.nodes),
        "slots": len(builder.slots),
        "weights_bytes": weights,
        "arena_naive_bytes": naive * 4,
        "arena_liveness_bytes": reused * 4,
        "arena_saved_ratio": round(naive / reused, 2) if reused else None,
        "output": [1, 1, 1, args.classes],
    }
    print(json.dumps(report, indent=2))

    if args.write:
        blob = blobfmt.build(builder.slots, builder.nodes)
        out = ROOT / args.out
        out.mkdir(parents=True, exist_ok=True)
        (out / "model.blob").write_bytes(blob)
        (out / "manifest.json").write_text(json.dumps({**report, "blob_bytes": len(blob)}, indent=2) + "\n",
                                           encoding="utf-8")
        print(f"[emit_mobilenet_blob] wrote {out / 'model.blob'} ({len(blob)} B)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
