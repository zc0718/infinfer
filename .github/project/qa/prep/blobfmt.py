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

For a binary node (`mul` / `add`) `in_slot` and `w_slot` are the two operands and `p[0]`
says how they combine: 0 means element-wise over equal extents, 1 means the right operand
is **one factor per output channel** (a `(C,)` constant against an `(H,W,C)` map). The
second form is MobileNet's squeeze-and-excite gate; the executors run it as one `scale`
call per channel rather than through a kernel of its own.

二元节点（`mul` / `add`）的 `in_slot` 与 `w_slot` 是两个操作数，`p[0]` 说明怎么合成：
0 表示等长逐元素；1 表示右操作数是**每个输出通道一个因子**（`(C,)` 常量对 `(H,W,C)`
特征图）。后者正是 MobileNet 的 SE 门，执行时按通道各调一次 `scale`，而不是另立一个核。

Op and activation codes are part of the **file format**, not of the C header: they are
frozen here so that renumbering an enum cannot silently reinterpret a blob that is
already in the tree.
"""
import hashlib
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


def short_name(name: str) -> str:
    """A slot name that fits the 16-byte field, still distinguishable.

    The field is diagnostic: the node table refers to slots by index, so nothing looks a name up
    at run time. But a real export's names are framework paths (`features.0.0.weight`), and a
    blob whose slots are all `features_0_0` is a blob nobody can debug. So the rule is: keep as
    much of the front as fits, then a `~` and four hex digits of the name's sha1 -- the same
    name always shortens to the same string, and two names that collide in their first ten
    characters almost certainly differ in the hash.

    Names that already fit are returned unchanged, which is why every blob written before this
    rule existed is byte-identical afterwards.

    字段是给人看的：节点表按**下标**引用槽，运行期没有谁按名字查找。但真导出的名字是框架路径
    （`features.0.0.weight`），若全部压成 `features_0_0`，这个 blob 就没人能调了。规则是：能放
    多少放多少前缀，再补一个 `~` 和名字 sha1 的四位十六进制——同名总是缩成同一串，前十个字符
    撞车的两个名字几乎一定在哈希上分开。

    本来就放得下的名字原样返回，所以这条规则之前写下的每个 blob 都逐字节不变。
    """
    if len(name.encode("ascii")) <= 15:
        return name
    digest = hashlib.sha1(name.encode("utf-8")).hexdigest()[:4]
    return name[:10] + "~" + digest


def slot(name: str, dims, payload: bytes) -> bytes:
    """One slot record. `payload` empty means a runtime buffer (the graph writes it)."""
    name = short_name(name)
    raw = name.encode("ascii")
    if len(raw) > 15:
        raise SystemExit(f"blobfmt: slot name too long even after shortening: {name}")
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


def layout(data: bytes):
    """`(slot_counts, node_refs, ops)` -- everything a memory plan needs and nothing else.

    `slot_counts[i]` is the slot's element count, or None when the slot is a constant
    (its payload lives in the blob, i.e. it costs flash and no arena). `node_refs[j]` is
    the node's `(in, out, w, b)` with None for an absent reference; `ops[j]` is its op
    code, so a caller can find the graph's endpoints without parsing the blob twice.

    内存计划需要且只需要这两样：每个槽的元素数（常量槽为 None，它花 flash 不花 arena），
    以及每个节点的四个槽引用（缺省为 None）；`ops` 带上算子码，调用方找端点时不必再解析
    一遍。
    """
    if len(data) < HEADER_SIZE or data[:8] != MAGIC:
        raise SystemExit("blobfmt: not an INFRKAT blob")
    version, node_count, slot_count, reserved = struct.unpack_from("<IIII", data, 8)
    if version != VERSION:
        raise SystemExit(f"blobfmt: unsupported version {version} (expected {VERSION})")
    if reserved != 0:
        raise SystemExit("blobfmt: header reserved field is not zero")
    offset = HEADER_SIZE
    if offset + node_count * NODE_SIZE > len(data):
        raise SystemExit("blobfmt: node table runs past the end of the blob")
    node_refs = []
    ops = []
    for index in range(node_count):
        base = HEADER_SIZE + index * NODE_SIZE
        ops.append(data[base])
        refs = struct.unpack_from("<IIII", data, base + 4)
        node_refs.append(tuple(None if ref == NO_SLOT else ref for ref in refs))
    offset += node_count * NODE_SIZE
    slot_counts = []
    for _ in range(slot_count):
        if offset + SLOT_HEADER_SIZE > len(data):
            raise SystemExit("blobfmt: slot header runs past the end of the blob")
        name = data[offset:offset + 16].split(b"\0", 1)[0].decode("ascii")
        dtype, rank = struct.unpack_from("<II", data, offset + 16)
        dims = struct.unpack_from("<4I", data, offset + 24)
        (byte_len,) = struct.unpack_from("<I", data, offset + 40)
        offset += SLOT_HEADER_SIZE
        if dtype != DTYPE_F32 or rank > 4:
            raise SystemExit(f"blobfmt: slot {name} is not a rank<=4 fp32 slot")
        count = 1
        for axis in range(rank):
            count *= dims[axis]
        if byte_len != 0 and byte_len != count * 4:
            raise SystemExit(f"blobfmt: slot {name} byte_len disagrees with its dims")
        if offset + byte_len > len(data):
            raise SystemExit(f"blobfmt: slot {name} payload runs past the end of the blob")
        slot_counts.append(None if byte_len else count)
        offset += byte_len
    if offset != len(data):
        raise SystemExit("blobfmt: trailing bytes after the last slot")
    return slot_counts, node_refs, ops


def endpoints(slot_counts, node_refs, ops):
    """`(in_floats, out_floats)` -- the extents a caller has to bring and gets back.

    A chain sizes its buffers from these instead of from constants of its own: the model
    file says how wide the input is, so the C side must not say it a second time.

    调用方要带多少个输入、拿回多少个输出。链用它来定缓冲区，而不是自己再写一遍常数：
    模型文件已经说了输入多宽，C 侧不该再说一次。
    """
    in_floats = 0
    for op, refs in zip(ops, node_refs):
        if op == OP["input"]:
            in_floats = slot_counts[refs[1]]
            break
    out_floats = slot_counts[node_refs[-1][1]] if node_refs else 0
    return in_floats, out_floats


def plan_arena(slot_counts, node_refs) -> int:
    """Floats a first-fit allocator needs for those live ranges -- the high-water mark.

    A slot is released at the end of the node that last touches it, so node i's output
    may land on a slot whose last use was node i-1, while two slots that are live at the
    same time never overlap. This is the **plan**; the executors derive the same number
    themselves (independently, in C and C++) and the KAT asserts the two agree on every
    target, because a plan nobody executes is just a number in a document.

    槽在最后一个碰它的节点结束时才释放，所以第 i 个节点的输出可以落在第 i-1 个节点最后
    用过的槽上，而同时活跃的两个槽永不重叠。这是**计划**；两份执行器（C 与 C++，彼此
    独立）各自算出同一个数，KAT 在每个目标上断言两者相等——因为没人执行的计划只是文档里
    的一个数字。
    """
    runtime = [index for index, count in enumerate(slot_counts) if count is not None]
    written = set()
    last_use = {}
    for position, (ins, out, weight, bias) in enumerate(node_refs):
        for ref in (ins, weight, bias):
            if ref is not None and slot_counts[ref] is not None:
                last_use[ref] = position
        last_use[out] = position
        written.add(out)
    orphans = [index for index in runtime if index not in written]
    if orphans:
        raise SystemExit(f"blobfmt: runtime slot(s) {orphans} are never written")

    free = [(0, None)]  # (offset, length); None means "the tail", so it always fits
    offsets = {}
    high_water = 0

    def take(count):
        nonlocal high_water
        for index, (offset, length) in enumerate(free):
            if length is None:
                free[index] = (offset + count, None)
                high_water = max(high_water, offset + count)
                return offset
            if length >= count:
                if length > count:
                    free[index] = (offset + count, length - count)
                else:
                    free.pop(index)
                high_water = max(high_water, offset + count)
                return offset
        raise SystemExit("blobfmt: allocator ran out of free list")

    for position, (_ins, out, _weight, _bias) in enumerate(node_refs):
        count = slot_counts[out]  # `out` is runtime by the format's own rule
        if count is None:
            raise SystemExit("blobfmt: a node writes into a constant slot")
        offsets[out] = take(count)
        for index in sorted(i for i in runtime if last_use.get(i) == position):
            free.append((offsets[index], slot_counts[index]))
            free.sort()
    return high_water
