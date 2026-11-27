#!/usr/bin/env python3
"""The ONNX **subset** this library can run, and the translation into a v2 blob.

Why a subset with a named refusal instead of "best effort": a graph one op outside the
subset does not fail at import, it fails as a wrong number three layers later, and the
question "why is my model not supported" then has no answer. Every refusal carries the
node name, the op and the reason.

What is translated, and what that costs:

  * **Layout.** ONNX is NCHW, the blob is NHWC. Conv weights are transposed
    `(oc, ic/g, kh, kw) -> (oc, kh, kw, ic/g)` and shapes `(n, c, h, w) -> (n, h, w, c)`.
    This is the one place the two conventions disagree, so it is the one place a silent
    error can live -- which is why the self-check compares the result byte for byte
    against a blob written out by hand.
  * **Shape-only ops are dropped.** `Flatten` / `Reshape` that preserve the element count
    emit no node (`fc` reads a slot's element count), but the count is checked first.
  * **Activations ride on the producing node**, because that is where the blob carries
    them (the `act` byte). An activation on a graph input, or one whose output feeds more
    than one consumer, is refused rather than silently mis-fused.
  * **`HardSigmoid`** is accepted only with `alpha = 1/6, beta = 0.5` (i.e.
    `relu6(x + 3) / 6`). ONNX's default is `alpha = 0.2`, a **different function**, so
    accepting it would be a wrong answer that looks right.

`nchw_to_nhwc_payload` is exported because the *fixture* writer needs the same permutation:
an input tensor handed to the executor has to be in the layout the blob declares, and both
layouts have the same element count, so getting it wrong is invisible to every shape check.
The subset, its refusals and its honest boundaries are recorded as gaps G-104.

`nchw_to_nhwc_payload` 对外导出，是因为**夹具**写入方需要同一个置换：交给执行器的输入张量必须
是 blob 声明的布局，而两种布局元素个数相同，换错了任何形状检查都看不见。这个子集、它的拒绝规则
和诚实的边界记录在 gaps G-104。
"""
import numpy as np

import blobfmt

# ONNX op_type -> the blob op name (a `blobfmt.OP` key).
NODE_OPS = {
    "Conv": "conv2d",
    "MaxPool": "max_pool2d",
    "GlobalAveragePool": "mean_hw",
    "Gemm": "fc",
    "Mul": "mul",
    "Add": "add",
}
# Ops that change how a tensor is spelled, never its numbers.
PASSTHROUGH = {"Flatten", "Reshape", "Identity"}
# ONNX op_type -> the blob activation name (a `blobfmt.ACT` key).
ACTIVATIONS = {
    "Relu": "relu",
    "Clip": "relu6",          # the (0, 6) form only; checked in `check()`
    "HardSigmoid": "hard_sigmoid",
    "HardSwish": "hard_swish",
}
HARD_SIGMOID_ALPHA = 1.0 / 6.0
HARD_SIGMOID_BETA = 0.5
MAX_OPSET = 20


class Unsupported(Exception):
    """One node is outside the subset; the message names it and says why."""


def _attrs(node):
    return {attr.name: attr for attr in node.attribute}


def _ints(attr, default):
    return list(default) if attr is None else list(attr.ints)


def _scalar(attr, default):
    return default if attr is None else float(attr.f)


def initializers(model):
    """Every static weight, as `{name: ndarray}`.

    A model whose weights are not all static cannot run where there is no allocator, so a
    missing one is a refusal rather than a TODO.
    """
    arrays = {}
    for tensor in model.graph.initializer:
        if tensor.data_type != 1:  # FLOAT
            raise Unsupported(f"initializer {tensor.name!r} is not FLOAT "
                              f"(data_type={tensor.data_type})")
        dims = list(tensor.dims)
        if tensor.raw_data:
            values = np.frombuffer(tensor.raw_data, dtype="<f4")
        else:
            values = np.asarray(tensor.float_data, dtype="<f4")
        arrays[tensor.name] = np.asarray(values, dtype=np.float32).reshape(dims)
    return arrays


def check(model):
    """`[(node_name, reason)]` for everything outside the subset.

    Nothing is executed and nothing is translated: this answers "will this model work
    here" without running it.
    """
    problems = []
    opset = max((entry.version for entry in model.opset_import), default=0)
    if opset > MAX_OPSET:
        problems.append(("<model>", f"opset {opset} is newer than this importer knows (<={MAX_OPSET})"))
    try:
        weights = initializers(model)
    except Unsupported as exc:
        return problems + [("<model>", str(exc))]
    known = set(NODE_OPS) | PASSTHROUGH | set(ACTIVATIONS) | {"Constant"}
    consumers = {}
    for node in model.graph.node:
        for name in node.input:
            consumers.setdefault(name, []).append(node)
    graph_inputs = {entry.name for entry in model.graph.input}
    for node in model.graph.node:
        label = node.name or node.op_type
        if node.op_type not in known:
            problems.append((label, f"op {node.op_type} is not in the subset"))
            continue
        if node.op_type == "Constant":
            problems.append((label, "a Constant node is not supported; use an initializer"))
        if node.op_type == "HardSigmoid":
            attrs = _attrs(node)
            alpha = _scalar(attrs.get("alpha"), 0.2)
            beta = _scalar(attrs.get("beta"), 0.5)
            if abs(alpha - HARD_SIGMOID_ALPHA) > 1e-12 or abs(beta - HARD_SIGMOID_BETA) > 1e-12:
                problems.append((label, f"alpha={alpha}, beta={beta}; this library implements "
                                        "alpha=1/6, beta=0.5 (relu6(x+3)/6)"))
        if node.op_type == "Clip":
            # opset <= 10 carries min/max as attributes, 11+ as inputs. Both forms are read
            # here so the answer does not depend on which opset the exporter chose.
            #
            # opset <= 10 把 min/max 当属性，11+ 当输入。两种都读，免得答案取决于导出器选的 opset。
            attrs = _attrs(node)
            if "min" in attrs or "max" in attrs:
                low = _scalar(attrs.get("min"), 0.0)
                high = _scalar(attrs.get("max"), 6.0)
            else:
                bounds = []
                for slot, default in ((1, 0.0), (2, 6.0)):
                    if len(node.input) > slot and node.input[slot] in weights:
                        bounds.append(float(np.asarray(weights[node.input[slot]]).ravel()[0]))
                    elif len(node.input) > slot and node.input[slot]:
                        problems.append((label, "Clip bounds must be static initializers"))
                        bounds.append(default)
                    else:
                        bounds.append(default)
                low, high = bounds
            if low != 0.0 or high != 6.0:
                problems.append((label, f"Clip({low}, {high}) is not relu6; only (0, 6) is in the subset"))
        if node.op_type == "Gemm":
            attrs = _attrs(node)
            if _scalar(attrs.get("alpha"), 1.0) != 1.0 or _scalar(attrs.get("beta"), 1.0) != 1.0:
                problems.append((label, "alpha/beta scaling has no representation in a blob node"))
            if "transB" not in attrs or attrs["transB"].i != 1:
                problems.append((label, "only transB=1 is accepted (a blob stores the weight as (out, in))"))
        if node.op_type in ACTIVATIONS and node.input:
            source = node.input[0]
            if source in graph_inputs:
                problems.append((label, "an activation on a graph input cannot ride on a node"))
            elif len(consumers.get(source, [])) > 1:
                problems.append((label, f"{source!r} feeds {len(consumers[source])} nodes; fusing "
                                        "the activation would change the others"))
        if node.op_type == "Reshape":
            if len(node.input) < 2 or node.input[1] not in weights:
                problems.append((label, "only a static Reshape target is supported"))
    return problems


def _nchw_to_nhwc(dims):
    n, c, h, w = dims
    return [n, h, w, c]


def nchw_to_nhwc_payload(array):
    """A rank-4 NCHW array as the contiguous NHWC bytes a blob slot wants.

    Exported because the *fixture* writer needs the same permutation: a test that reads an
    input tensor has to put it in the layout the blob declares, and both layouts have the
    same element count, so a mistake here is invisible to any shape check.

    对外导出是因为**夹具**写入方需要同一个置换：读入一个输入张量并对准 blob 声明的布局时，
    两种布局元素个数相同，任何形状检查都看不见换错布局这种错。
    """
    if array.ndim != 4:
        raise Unsupported(f"a rank-{array.ndim} tensor has no NCHW layout to change")
    return np.ascontiguousarray(np.transpose(array, (0, 2, 3, 1)), dtype="<f4")


def translate(model):
    """ONNX model -> v2 blob bytes. Raises `Unsupported`, naming the node."""
    problems = check(model)
    if problems:
        name, reason = problems[0]
        raise Unsupported(f"{name}: {reason}")
    weights = initializers(model)
    opset = max((entry.version for entry in model.opset_import), default=0)
    if opset == 0:
        raise Unsupported("<model>: no opset_import")

    slots = []          # [(name, dims, payload)] in blob order
    slot_of = {}
    shape_of = {}
    alias_of = {}
    nodes = []          # {"op": <OP key>, "in"/"out"/"w"/"b": names, "p": tuple, "act": key}

    def declare(name, dims, payload):
        if name in slot_of:
            raise Unsupported(f"<model>: tensor {name!r} is produced twice")
        slot_of[name] = len(slots)
        slots.append((name, dims, payload))

    def edge(name):
        """Follow Flatten/Reshape/activation aliases to the tensor that actually holds bytes."""
        seen = set()
        while name in alias_of:
            if name in seen:
                raise Unsupported(f"<model>: alias cycle at {name!r}")
            seen.add(name)
            name = alias_of[name]
        return name

    def stamp_act(tensor_name, act_key):
        for entry in nodes:
            if entry.get("out") == tensor_name:
                entry["act"] = act_key
                return
        raise Unsupported(f"<model>: an activation on {tensor_name!r} has no producing node")

    graph_inputs = [entry for entry in model.graph.input if entry.name not in weights]
    if len(graph_inputs) != 1:
        raise Unsupported(f"<model>: exactly one graph input is supported, found {len(graph_inputs)}")
    source = graph_inputs[0]
    dims = [axis.dim_value for axis in source.type.tensor_type.shape.dim]
    if len(dims) != 4:
        raise Unsupported(f"{source.name}: only rank-4 NCHW inputs are supported, got rank {len(dims)}")
    shape_of[source.name] = _nchw_to_nhwc(dims)
    declare(source.name, shape_of[source.name], b"")
    nodes.append({"op": "input", "out": source.name})

    for node in model.graph.node:
        label = node.name or node.op_type
        kind = node.op_type

        if kind in PASSTHROUGH:
            src = edge(node.input[0])
            if src not in shape_of:
                raise Unsupported(f"{label}: {src!r} is read before it is produced")
            if kind == "Reshape":
                requested = [int(round(float(v))) for v in np.asarray(weights[node.input[1]]).ravel()]
                after = int(np.prod([v for v in requested if v != 0]))
                before = int(np.prod(shape_of[src]))
                if after != before:
                    raise Unsupported(f"{label}: Reshape changes the element count ({before} -> "
                                      f"{after}); only shape-only reshapes are dropped")
            alias_of[node.output[0]] = src
            shape_of[node.output[0]] = list(shape_of[src])
            continue

        if kind in ACTIVATIONS:
            src = edge(node.input[0])
            stamp_act(src, ACTIVATIONS[kind])
            alias_of[node.output[0]] = src
            shape_of[node.output[0]] = list(shape_of[src])
            continue

        src = edge(node.input[0])

        if kind == "Conv":
            weight = weights.get(node.input[1])
            if weight is None:
                raise Unsupported(f"{label}: a non-static filter is not supported")
            bias_name = node.input[2] if len(node.input) > 2 and node.input[2] else None
            bias = weights.get(bias_name) if bias_name else None
            attrs = _attrs(node)
            groups = int(attrs["group"].i) if "group" in attrs else 1
            strides = _ints(attrs.get("strides"), [1, 1])
            pads = _ints(attrs.get("pads"), [0, 0, 0, 0])
            kh, kw = int(weight.shape[2]), int(weight.shape[3])
            if strides[0] != strides[1] or kh != kw:
                raise Unsupported(f"{label}: only square kernels and equal strides are supported")
            if pads[0] != pads[2] or pads[1] != pads[3] or pads[0] != kh // 2 or pads[1] != kw // 2:
                raise Unsupported(f"{label}: padding {pads} is not the kernel//2 form the "
                                  "executor derives")
            src_shape = shape_of[src]
            out_c = int(weight.shape[0])
            if src_shape[3] != int(weight.shape[1]) * groups or out_c % groups:
                raise Unsupported(f"{label}: {weight.shape} does not divide the {src_shape[3]}-channel "
                                  f"input into {groups} group(s)")
            stride, pad = strides[0], pads[0]
            oh = (src_shape[1] + 2 * pad - kh) // stride + 1
            ow = (src_shape[2] + 2 * pad - kw) // stride + 1
            shape_of[node.output[0]] = [src_shape[0], oh, ow, out_c]
            declare(node.output[0], shape_of[node.output[0]], b"")
            payload = nchw_to_nhwc_payload(weight)
            declare(node.input[1], _nchw_to_nhwc(list(weight.shape)), payload.tobytes())
            if bias is not None:
                declare(bias_name, list(bias.shape), np.ascontiguousarray(bias, dtype="<f4").tobytes())
            nodes.append({"op": "conv2d", "in": src, "out": node.output[0],
                          "w": node.input[1], "b": bias_name,
                          "p": (kh, stride, pad, groups)})
            continue

        if kind == "MaxPool":
            attrs = _attrs(node)
            kernels = _ints(attrs.get("kernel_shape"), [1, 1])
            strides = _ints(attrs.get("strides"), [1, 1])
            pads = _ints(attrs.get("pads"), [0, 0, 0, 0])
            if kernels[0] != kernels[1] or strides[0] != strides[1]:
                raise Unsupported(f"{label}: only square kernels and equal strides are supported")
            if pads[0] != pads[2] or pads[1] != pads[3]:
                raise Unsupported(f"{label}: only symmetric padding is supported")
            src_shape = shape_of[src]
            k, stride, pad = kernels[0], strides[0], pads[0]
            oh = (src_shape[1] + 2 * pad - k) // stride + 1
            ow = (src_shape[2] + 2 * pad - k) // stride + 1
            shape_of[node.output[0]] = [src_shape[0], oh, ow, src_shape[3]]
            declare(node.output[0], shape_of[node.output[0]], b"")
            nodes.append({"op": "max_pool2d", "in": src, "out": node.output[0],
                          "p": (k, stride, pad, 0)})
            continue

        if kind == "GlobalAveragePool":
            src_shape = shape_of[src]
            shape_of[node.output[0]] = [src_shape[0], 1, 1, src_shape[3]]
            declare(node.output[0], shape_of[node.output[0]], b"")
            nodes.append({"op": "mean_hw", "in": src, "out": node.output[0]})
            continue

        if kind == "Gemm":
            weight = weights.get(node.input[1])
            if weight is None or weight.ndim != 2:
                raise Unsupported(f"{label}: a non-static or non-2-D Gemm weight is not supported")
            bias_name = node.input[2] if len(node.input) > 2 and node.input[2] else None
            bias = weights.get(bias_name) if bias_name else None
            if int(np.prod(shape_of[src])) != int(weight.shape[1]):
                raise Unsupported(f"{label}: Gemm expects {weight.shape[1]} inputs, the tensor "
                                  f"holds {int(np.prod(shape_of[src]))}")
            shape_of[node.output[0]] = [1, 1, 1, int(weight.shape[0])]
            declare(node.output[0], shape_of[node.output[0]], b"")
            declare(node.input[1], list(weight.shape),
                    np.ascontiguousarray(weight, dtype="<f4").tobytes())
            if bias is not None:
                declare(bias_name, list(bias.shape), np.ascontiguousarray(bias, dtype="<f4").tobytes())
            nodes.append({"op": "fc", "in": src, "out": node.output[0],
                          "w": node.input[1], "b": bias_name})
            continue

        if kind in ("Mul", "Add"):
            lhs_shape = shape_of[src]
            gate_name = node.input[1]
            rhs = weights.get(gate_name)
            if rhs is None:
                raise Unsupported(f"{label}: only a constant right operand is supported")
            rhs_nhwc = _nchw_to_nhwc(list(rhs.shape)) if rhs.ndim == 4 else list(rhs.shape)
            if rhs_nhwc == lhs_shape:
                payload = np.ascontiguousarray(rhs, dtype="<f4").tobytes()
                params = (0, 0, 0, 0)
            elif kind == "Mul" and rhs.ndim == 4 and rhs_nhwc[:3] == [1, 1, 1] \
                    and rhs_nhwc[3] == lhs_shape[3]:
                # The squeeze-and-excite gate: one factor per output channel.
                payload = np.ascontiguousarray(np.transpose(rhs, (0, 2, 3, 1)),
                                               dtype="<f4").tobytes()
                params = (1, 0, 0, 0)
            else:
                raise Unsupported(f"{label}: {list(rhs.shape)} against {lhs_shape} is not a "
                                  "broadcast this executor performs")
            shape_of[node.output[0]] = list(lhs_shape)
            # Output first, then its operand: every other branch declares in that order, and a
            # slot table whose order depends on which op produced it is a needless difference
            # between two blobs that mean the same thing.
            # 先声明输出、再声明操作数：其它分支都是这个顺序。让槽表顺序取决于算子种类，只会
            # 给"意思相同的两个 blob"制造无谓的差异。
            declare(node.output[0], shape_of[node.output[0]], b"")
            declare(gate_name, rhs_nhwc, payload)
            nodes.append({"op": NODE_OPS[kind], "in": src, "out": node.output[0],
                          "w": gate_name, "p": params})
            continue

        raise Unsupported(f"{label}: op {kind} is not in the subset")

    graph_outputs = [entry.name for entry in model.graph.output]
    if len(graph_outputs) != 1:
        raise Unsupported(f"<model>: exactly one graph output is supported, found {len(graph_outputs)}")
    produced = {entry["out"] for entry in nodes}
    if edge(graph_outputs[0]) not in produced:
        raise Unsupported(f"<model>: output {graph_outputs[0]!r} is not produced by a translated node")
    if nodes[-1]["out"] != edge(graph_outputs[0]):
        raise Unsupported("<model>: a blob's output is its last node's, so the graph output must "
                          f"be produced last (it is {graph_outputs[0]!r})")
    return blobfmt.build(slots, nodes)
