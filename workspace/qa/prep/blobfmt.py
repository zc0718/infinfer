#!/usr/bin/env python3
"""The `INFRKAT` **graph blob** format, in one place (writer + constant reader).

Every producer and consumer on the Python side imports this module, so the 24-byte
header, the 36-byte node record and the 44-byte slot header exist exactly once. The
reader used by CI lives in C++ (`test_package/test/unit/blob_graph.hpp`) on purpose:
the tests must decode the bytes without Python. Those two implementations are kept
honest by the blob's own self-checks (declared dims vs `byte_len`, no trailing bytes,
no writing into a constant slot) rather than by sharing code -- a shared parser would
only prove that one implementation agrees with itself.

`INFRKAT` 图 blob 格式的唯一 Python 定义。CI 侧的读取器故意用 C++ 重写一份
（`test_package/test/unit/blob_graph.hpp`），因为测试必须不依赖 Python 也能解字节；两者
靠 blob 自身的自洽检查（声明形状与 byte_len 相符、无尾随字节、不许写常量槽）互相钉住，
而不是共享同一份解析代码——共享只能证明"一个实现和它自己一致"。

Layout (little endian):

    "INFRKAT\\0"   8 B
    u32 version = 2
    u32 node_count
    u32 slot_count
    u32 reserved = 0

    per node, 36 B:
        u8  op, u8 act, u16 reserved = 0
        u32 in_slot, out_slot, w_slot, b_slot      (NO_SLOT = absent)
        u32 p[4]

    per slot:
        char name[16], u32 dtype, u32 rank, u32 dims[4], u32 byte_len, payload

Op and activation codes are part of the **file format**, not of the C header: they are
frozen here so that renumbering an enum cannot silently reinterpret a blob that is
already in the tree.
"""
import struct

MAGIC = b"INFRKAT\0"
VERSION = 2
DTYPE_F32 = 1
NO_SLOT = 0xFFFFFFFF
HEADER_SIZE = 24
NODE_SIZE = 36
SLOT_HEADER_SIZE = 44

OP = {"input": 0, "conv2d": 1, "max_pool2d": 2, "mean_hw": 3, "fc": 4, "mul": 5, "add": 6}
ACT = {"none": 0, "relu": 1, "relu_n1_to_1": 2, "relu6": 3, "tanh": 4, "sign_bit": 5,
       "hard_swish": 6, "hard_sigmoid": 7}


def slot(name: str, dims, payload: bytes) -> bytes:
    """One slot record. `payload` empty means a runtime buffer (the graph writes it)."""
    raw = name.encode("ascii")
    if len(raw) > 15:
        raise SystemExit(f"blobfmt: slot name too long: {name}")
    if len(dims) > 4:
        raise SystemExit(f"blobfmt: slot {name} has rank {len(dims)}, max 4")
    count = 1
    for axis in dims:
        count *= axis
    if payload and len(payload) != count * 4:
        raise SystemExit(f"blobfmt: slot {name} payload is {len(payload)} B, dims need {count * 4} B")
    padded = list(dims) + [0] * (4 - len(dims))
    return (raw.ljust(16, b"\0") +
            struct.pack("<II4II", DTYPE_F32, len(dims), *padded, len(payload)) + payload)


def encode_node(index_of, node) -> bytes:
    """One node record. Slot references are names here and indices on disk."""
    def ref(key):
        name = node.get(key)
        return NO_SLOT if name is None else index_of[name]

    return (struct.pack("<BBH", OP[node["op"]], ACT[node.get("act", "none")], 0) +
            struct.pack("<IIII", ref("in"), ref("out"), ref("w"), ref("b")) +
            struct.pack("<4I", *node.get("p", (0, 0, 0, 0))))


def build(slot_specs, nodes) -> bytes:
    """Assemble a blob. `slot_specs` is an ordered list of `(name, dims, payload)`."""
    index_of = {name: i for i, (name, _dims, _payload) in enumerate(slot_specs)}
    slot_bytes = b"".join(slot(name, dims, payload) for name, dims, payload in slot_specs)
    node_bytes = b"".join(encode_node(index_of, node) for node in nodes)
    return (MAGIC + struct.pack("<IIII", VERSION, len(nodes), len(slot_specs), 0) +
            node_bytes + slot_bytes)


def parse_constants(data: bytes) -> dict:
    """The constant slots of a v2 blob, as `{name: (float, ...)}`.

    The nodes are skipped: callers of this function want the baked weights, not the
    graph. A v1 blob is rejected rather than misparsed, because the two layouts only
    differ by a header size -- exactly the mistake that would look like working code.
    """
    if len(data) < HEADER_SIZE or data[:8] != MAGIC:
        raise SystemExit("blobfmt: not an INFRKAT blob")
    version, node_count, slot_count, reserved = struct.unpack_from("<IIII", data, 8)
    if version != VERSION:
        raise SystemExit(f"blobfmt: unsupported version {version} (expected {VERSION})")
    if reserved != 0:
        raise SystemExit("blobfmt: header reserved field is not zero")
    offset = HEADER_SIZE + node_count * NODE_SIZE
    if offset > len(data):
        raise SystemExit("blobfmt: node table runs past the end of the blob")
    constants = {}
    for _ in range(slot_count):
        if offset + SLOT_HEADER_SIZE > len(data):
            raise SystemExit("blobfmt: slot header runs past the end of the blob")
        name = data[offset:offset + 16].split(b"\0", 1)[0].decode("ascii")
        dtype, rank = struct.unpack_from("<II", data, offset + 16)
        dims = struct.unpack_from("<4I", data, offset + 24)
        (byte_len,) = struct.unpack_from("<I", data, offset + 40)
        offset += SLOT_HEADER_SIZE
        if dtype != DTYPE_F32 or byte_len % 4:
            raise SystemExit(f"blobfmt: slot {name} is not fp32")
        if offset + byte_len > len(data):
            raise SystemExit(f"blobfmt: slot {name} payload runs past the end of the blob")
        if byte_len == 0:
            continue  # runtime buffer: nothing to bake
        declared = 1
        for axis in range(rank):
            declared *= dims[axis]
        if declared * 4 != byte_len:
            raise SystemExit(f"blobfmt: slot {name} byte_len disagrees with its dims")
        constants[name] = struct.unpack_from(f"<{byte_len // 4}f", data, offset)
        offset += byte_len
    if offset != len(data):
        raise SystemExit("blobfmt: trailing bytes after the last slot")
    return constants
