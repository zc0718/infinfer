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
import math
import numpy as np

import blobfmt

# ONNX op_type -> the blob op name (a `blobfmt.OP` key).
#
# `ReduceMean` is in here with a condition (axes = the two spatial ones, keepdims = 1) which
# `check()` enforces: that is how a modern exporter spells a global average pool, and refusing
# it was a mistake of *reason* rather than of taste -- the first real MobileNetV3 export arrived
# as 10 `ReduceMean` nodes where the synthetic fixture had `GlobalAveragePool`.
#
# `ReduceMean` 带条件（轴是两个空间轴、keepdims=1），条件在 `check()` 里查：这正是新导出器表达
# global average pool 的方式。当初拒绝它是个**理由写错**，不是口味问题——第一份真 MobileNetV3
# 导出里是 10 个 `ReduceMean`，而合成夹具里是 `GlobalAveragePool`。
NODE_OPS = {
    "Conv": "conv2d",
    "MaxPool": "max_pool2d",
    "GlobalAveragePool": "mean_hw",
    "ReduceMean": "mean_hw",
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
# Ops a kernel in `src/kernel/matrix.yml` claims, which the importer deliberately does not
# translate yet, with the reason. Silence was the old behaviour and it was indistinguishable
# from "we serve this": `census_gate` R13a now refuses a matrix claim that is neither
# translated nor listed here, and R13b refuses an entry that outlives the work.
#
# 内核在 `matrix.yml` 里声称、而导入器**故意**还没翻译的 ONNX 算子，连带理由。旧行为是“沉默”，
# 而沉默与“我们支持它”无法区分：`census_gate` 的 R13a 现在拒绝“既没翻译也没登记”的矩阵声称，
# R13b 拒绝“工作已经完成却还留着”的条目。
DEFERRED = {
    "MatMul": "`fc` serves it at the library level, but the importer only translates `Gemm`, "
              "which carries the bias and transB handling; a bare MatMul would need those "
              "rules restated here, and MobileNetV3 has none",
    "QuantizeLinear": "quantisation is the PTQ step's output (PLAN P5.3), which emits the blob "
                       "directly; a model that still carries Q/DQ pairs is not translated yet",
    "DequantizeLinear": "the other half of the same pair; see QuantizeLinear",
    "ReduceSum": "the blob has `mean_hw` and no general reduce -- a reduction needs an axis "
                 "and a shape rule the format does not carry",
    "ReduceMax": "same as ReduceSum; `max` is a kernel, not a graph node in this format",
}
HARD_SIGMOID_ALPHA = 1.0 / 6.0
HARD_SIGMOID_BETA = 0.5
MAX_OPSET = 20
# The only initializer families this importer accepts: float weights, integer shape constants.
# One spelling, so `initializers` and `shape_constants` cannot disagree about what 7 means.
# 本导入器只接受的两类初始化器：浮点权重与整数形状常量。只在此处写一次，
# 好让 `initializers` 与 `shape_constants` 对“7 是什么”没有第二种说法。
_DTYPES = {"float32": 1, "int32": 6, "int64": 7}


class Unsupported(Exception):
    """One node is outside the subset; the message names it and says why."""


def _attrs(node):
    return {attr.name: attr for attr in node.attribute}


def _ints(attr, default):
    return list(default) if attr is None else list(attr.ints)


def _scalar(attr, default):
    return default if attr is None else float(attr.f)


def initializers(model):
    """Every static **float** weight, as `{name: ndarray}`.

    A model whose weights are not all static cannot run where there is no allocator, so a
    missing one is a refusal rather than a TODO. Integer initializers are not weights -- they
    are the shape constants a `Reshape` carries -- so they are read by `shape_constants`
    instead of being refused here, and only those two families are accepted at all.

    每个静态**浮点**权重。权重不静态的模型在无分配器的目标上跑不了，所以缺一个就是拒绝而不是 TODO。
    整数初始化器不是权重（它们是 `Reshape` 携带的形状常量），由 `shape_constants` 读取；只接受
    这两类，其它类型仍然拒绝。
    """
    arrays = {}
    for tensor in model.graph.initializer:
        if tensor.data_type == _DTYPES["int64"] or tensor.data_type == _DTYPES["int32"]:
            continue
        if tensor.data_type != _DTYPES["float32"]:
            raise Unsupported(f"initializer {tensor.name!r} has data_type={tensor.data_type}, "
                              "which is neither FLOAT nor an integer shape constant")
        dims = list(tensor.dims)
        if tensor.raw_data:
            values = np.frombuffer(tensor.raw_data, dtype="<f4")
        else:
            values = np.asarray(tensor.float_data, dtype="<f4")
        arrays[tensor.name] = np.asarray(values, dtype=np.float32).reshape(dims)
    return arrays


def shape_constants(model):
    """Integer initializers, as `{name: [int, ...]}` -- `Reshape` targets and the like.

    Kept apart from `initializers` so that "every weight is fp32" stays an invariant the rest
    of the module can rely on: a blob slot is fp32 and an integer array has no place in one.

    与 `initializers` 分开，好让“所有权重都是 fp32”这条不变量继续成立：blob 槽是 fp32，
    整数数组在那里没有位置。
    """
    shapes = {}
    for tensor in model.graph.initializer:
        if tensor.data_type not in (_DTYPES["int64"], _DTYPES["int32"]):
            continue
        if tensor.raw_data:
            dtype = "<i8" if tensor.data_type == _DTYPES["int64"] else "<i4"
            values = np.frombuffer(tensor.raw_data, dtype=dtype)
        else:
            values = np.asarray(
                tensor.int64_data if tensor.data_type == _DTYPES["int64"] else tensor.int32_data,
                dtype=np.int64)
        shapes[tensor.name] = [int(v) for v in np.asarray(values).ravel()]
    return shapes


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
    shapes = shape_constants(model)
    static = set(weights) | set(shapes)
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
            # The attributes are float32 on the wire, so an exporter that means 1/6 hands us
            # 0.16666667163372040 -- a strict == against the Python double is a check of the
            # fixture's precision, not of the semantics. It was written that way and it refused
            # the first real MobileNetV3 export for exactly this reason.
            #
            # 属性在线上是 float32，所以导出器想写 1/6 时交给我们的是 0.16666667163372040；
            # 拿 Python double 去严格相等，量的是夹具的精度而不是语义。原实现就是这么写的，
            # 于是它拒绝了第一份真 MobileNetV3 导出，理由正是这个。
            if (not math.isclose(alpha, HARD_SIGMOID_ALPHA, rel_tol=1e-6, abs_tol=1e-9)
                    or not math.isclose(beta, HARD_SIGMOID_BETA, rel_tol=1e-6, abs_tol=1e-9)):
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
            if len(node.input) < 2 or node.input[1] not in static:
                problems.append((label, "only a static Reshape target is supported"))
        if node.op_type == "ReduceMean":
            # Only the global-pool form. A ReduceMean over anything else would need an axis and
            # a shape rule the blob format does not carry, and accepting it because the numbers
            # happen to match for one shape is how a wrong answer looks right.
            #
            # Two spellings, like `Clip`: opset < 18 carries `axes` as an attribute, opset 18
            # moved it to input[1]. Reading only the attribute reports `axes=[]` for a modern
            # export, which is what happened first.
            #
            # 只接受 global-pool 形式；其余 ReduceMean 需要 blob 格式并不携带的轴与形状规则。
            # 两种写法（与 `Clip` 同理）：opset < 18 把 `axes` 当属性，opset 18 挪到了 input[1]。
            # 只读属性会对新导出报 `axes=[]`——第一次就是这么撞上的。
            attrs = _attrs(node)
            axes = _ints(attrs.get("axes"), [])
            if not axes and len(node.input) > 1 and node.input[1]:
                axes = list(shapes.get(node.input[1]) or [])
                if not axes:
                    problems.append((label, "ReduceMean axes must be a static integer "
                                            "initializer when given as an input"))
            keepdims = int(attrs["keepdims"].i) if "keepdims" in attrs else 1
            if sorted(axis % 4 for axis in axes) != [2, 3] or keepdims != 1:
                problems.append((label, f"ReduceMean(axes={axes}, keepdims={keepdims}) is not a "
                                        "global average pool; only axes=[2,3] with keepdims=1 "
                                        "is in the subset"))
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
    shapes = shape_constants(model)
    static = set(weights) | set(shapes)
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

    graph_inputs = [entry for entry in model.graph.input if entry.name not in static]
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
                target = shapes.get(node.input[1])
                if target is None:
                    raise Unsupported(f"{label}: Reshape target {node.input[1]!r} is not a static "
                                      "integer initializer")
                requested = [int(axis) for axis in target]
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

        if kind in ("GlobalAveragePool", "ReduceMean"):
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
            lhs_name, rhs_name = node.input[0], node.input[1]
            lhs, rhs = edge(lhs_name), edge(rhs_name)
            lhs_shape, rhs_shape = shape_of.get(lhs), shape_of.get(rhs)

            # A **runtime** per-channel gate: real squeeze-and-excite computes its gate, so the
            # multiply has an (1,1,1,C) tensor against an (N,H,W,C) map and neither side is a
            # constant. The constant form below was written first and it is what made the
            # synthetic MobileNet's "SE" block not an SE block at all (gaps G-107).
            #
            # **运行时**的逐通道门：真的 squeeze-and-excite 是算出门的，所以乘法一边是
            # (1,1,1,C) 张量、另一边是 (N,H,W,C) 特征图，两侧都不是常量。下面那种常量形式是先
            # 写的，而它正是让合成 MobileNet 的"SE"块根本不是 SE 的原因（gaps G-107）。
            if kind == "Mul" and lhs_shape and rhs_shape:
                pair = None
                for feat, feat_shape, gate, gate_shape in (
                        (lhs, lhs_shape, rhs, rhs_shape), (rhs, rhs_shape, lhs, lhs_shape)):
                    if (gate_shape[:3] == [1, 1, 1] and gate_shape[3] == feat_shape[3]
                            and feat_shape[1:3] != [1, 1]):
                        pair = (feat, feat_shape, gate)
                        break
                if pair is not None:
                    feat, feat_shape, gate = pair
                    shape_of[node.output[0]] = list(feat_shape)
                    declare(node.output[0], shape_of[node.output[0]], b"")
                    nodes.append({"op": "mul", "in": feat, "out": node.output[0],
                                  "w": gate, "p": (1, 0, 0, 0)})
                    continue

            lhs_shape = shape_of[src]
            gate_name = node.input[1]
            rhs = weights.get(gate_name)
            if rhs is None:
                # A runtime right operand of the same extent: the residual connection, which is
                # `Add(feature, skip)` with both sides computed. The blob stores both as slots
                # and the binary kernel runs over the whole map.
                #
                # 等长的运行期右操作数：残差连接，`Add(feature, skip)` 两边都是算出来的。blob 把
                # 两者都存成槽，二元核在整张图上跑。
                runtime_rhs = edge(gate_name)
                if kind in ("Mul", "Add") and shape_of.get(runtime_rhs) == lhs_shape:
                    shape_of[node.output[0]] = list(lhs_shape)
                    declare(node.output[0], shape_of[node.output[0]], b"")
                    nodes.append({"op": NODE_OPS[kind], "in": src, "out": node.output[0],
                                  "w": runtime_rhs, "p": (0, 0, 0, 0)})
                    continue
                raise Unsupported(f"{label}: only a constant or equal-extent right operand is "
                                  "supported")
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
