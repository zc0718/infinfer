// Graph blob executor: read a `INFRKAT` v2 blob (graph nodes + tensor slots) and run
// it through the public L2 operators, so the *model* owns the graph and the runner
// does not have to be rewritten per model.
//
// 图 blob 执行器：读入 `INFRKAT` v2 blob（图节点 + 张量槽），通过公开的 L2 算子执行，
// 让**模型**拥有图，执行器不必按模型重写一遍。
//
// Why this exists：KAT 从 k0 到 k2 一直是"权重在 blob 里、图写死在 C/C++ 里"。写死一张图
// 是可行的，但覆盖面（PLAN §2.2.1）要求同一份代码跑 MobileNetV3 / YOLO / 时序网，那就
// 只能把图变成数据。v1 blob 只有张量表，v2 加了一张节点表，本文件是它唯一的执行者。
//
// Format v2 (little endian), header 24 B:
//
//   "INFRKAT\0"        8 B
//   u32 version = 2
//   u32 node_count
//   u32 slot_count
//   u32 reserved = 0
//
// nodes[node_count], 36 B each:
//
//   u8  op              0=input 1=conv2d 2=max_pool2d 3=mean_hw 4=fc
//   u8  act             `infr_activation` applied in place to the output, 0 = none
//   u16 reserved = 0
//   u32 in_slot         kNoSlot for `input`
//   u32 out_slot
//   u32 w_slot          kNoSlot when the op has no weight
//   u32 b_slot          kNoSlot when the op has no bias
//   u32 p[4]            op-specific: conv2d {kernel, stride, pad, groups},
//                       max_pool2d {kernel, stride, pad, -}, fc {-, -, -, -}
//
// slots[slot_count]:
//
//   char name[16]       NUL padded
//   u32  dtype          1 = F32 (mirrors infr_dtype)
//   u32  rank
//   u32  dims[4]
//   u32  byte_len       payload bytes; 0 means "runtime buffer", read/written by run()
//   bytes payload       only when byte_len > 0 (weights, biases)
//
// 两条设计约束：① 每个算子都**重新推导**输出形状并与槽里声明的形状比对——生成器把形状
// 写错时这里红，而不是把垃圾浮点数喂给下游；② 所有非恒定槽都落在一块**调用方给的**
// arena 上，图跑完 arena 用量不变，这正是 INV-8 要的性质在最外层被观测到。
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include <infinfer/infinfer.h>

namespace infrb {

constexpr std::uint32_t kNoSlot = 0xFFFFFFFFU;
constexpr std::uint8_t kOpInput = 0;
constexpr std::uint8_t kOpConv2d = 1;
constexpr std::uint8_t kOpMaxPool2d = 2;
constexpr std::uint8_t kOpMeanHw = 3;
constexpr std::uint8_t kOpFc = 4;
constexpr std::uint8_t kOpMul = 5;
constexpr std::uint8_t kOpAdd = 6;
constexpr std::uint32_t kDtypeF32 = 1;

struct Slot {
    std::string name;
    std::uint32_t dtype = 0;
    std::uint32_t rank = 0;
    std::uint32_t dims[4] = {0, 0, 0, 0};
    std::uint32_t byte_len = 0;
    std::size_t payload = 0;  // byte offset into Blob::bytes
    std::size_t arena = 0;    // float offset into the caller's arena
};

struct Node {
    std::uint8_t op = 0;
    std::uint8_t act = 0;
    std::uint32_t in_slot = kNoSlot;
    std::uint32_t out_slot = kNoSlot;
    std::uint32_t w_slot = kNoSlot;
    std::uint32_t b_slot = kNoSlot;
    std::uint32_t p[4] = {0, 0, 0, 0};
};

struct Blob {
    std::uint32_t version = 0;
    std::vector<Node> nodes;
    std::vector<Slot> slots;
    std::vector<unsigned char> bytes;  // owns the payloads
    std::size_t arena_floats = 0;      // required arena length in floats
};

namespace detail {

inline std::uint32_t read_u32(const unsigned char* p) {
    return static_cast<std::uint32_t>(p[0]) | (static_cast<std::uint32_t>(p[1]) << 8) |
           (static_cast<std::uint32_t>(p[2]) << 16) | (static_cast<std::uint32_t>(p[3]) << 24);
}

inline std::size_t slot_floats(const Slot& slot) {
    std::size_t count = 1;
    for (std::uint32_t axis = 0; axis < slot.rank; ++axis) {
        count *= slot.dims[axis];
    }
    return count;
}

inline bool same_dims(const Slot& slot, const infr_shape& shape) {
    if (slot.rank != shape.rank) {
        return false;
    }
    for (std::uint32_t axis = 0; axis < slot.rank; ++axis) {
        if (slot.dims[axis] != shape.dims[axis]) {
            return false;
        }
    }
    return true;
}

}  // namespace detail

// Parse a v2 blob. Every field is bounds-checked and every reference between a node
// and a slot is resolved here, so `run()` can assume a well-formed graph.
//
// 解析 v2 blob：逐字段做边界检查，并在这一步解析掉节点与槽之间的所有引用，于是 run()
// 可以假定图是完好的。返回 false 时 `err` 指出**第一个**出问题的字段。
inline bool parse(std::vector<unsigned char> bytes, Blob* out, std::string* err) {
    static const char kMagic[8] = {'I', 'N', 'F', 'R', 'K', 'A', 'T', '\0'};
    if (bytes.size() < 24) {
        *err = "blob shorter than the 24-byte header";
        return false;
    }
    if (std::memcmp(bytes.data(), kMagic, sizeof kMagic) != 0) {
        *err = "bad magic (expected INFRKAT\\0)";
        return false;
    }
    out->version = detail::read_u32(bytes.data() + 8);
    if (out->version != 2) {
        *err = "unsupported blob version " + std::to_string(out->version) + " (this reader is v2)";
        return false;
    }
    const std::uint32_t node_count = detail::read_u32(bytes.data() + 12);
    const std::uint32_t slot_count = detail::read_u32(bytes.data() + 16);
    const std::uint32_t reserved = detail::read_u32(bytes.data() + 20);
    if (reserved != 0) {
        *err = "header reserved field is not zero";
        return false;
    }

    std::size_t offset = 24;
    if (bytes.size() < offset + (static_cast<std::size_t>(node_count) * 36U)) {
        *err = "node table runs past the end of the blob";
        return false;
    }
    out->nodes.resize(node_count);
    for (std::uint32_t i = 0; i < node_count; ++i) {
        const unsigned char* p = bytes.data() + offset;
        Node& node = out->nodes[i];
        node.op = p[0];
        node.act = p[1];
        const std::uint16_t node_reserved = static_cast<std::uint16_t>(p[2] | (p[3] << 8));
        if (node_reserved != 0) {
            *err = "node " + std::to_string(i) + " reserved field is not zero";
            return false;
        }
        node.in_slot = detail::read_u32(p + 4);
        node.out_slot = detail::read_u32(p + 8);
        node.w_slot = detail::read_u32(p + 12);
        node.b_slot = detail::read_u32(p + 16);
        for (std::uint32_t k = 0; k < 4; ++k) {
            node.p[k] = detail::read_u32(p + 20 + (4U * k));
        }
        offset += 36;
    }

    out->slots.resize(slot_count);
    for (std::uint32_t i = 0; i < slot_count; ++i) {
        if (bytes.size() < offset + 44U) {
            *err = "slot " + std::to_string(i) + " header runs past the end of the blob";
            return false;
        }
        const unsigned char* p = bytes.data() + offset;
        Slot& slot = out->slots[i];
        slot.name.assign(reinterpret_cast<const char*>(p));
        slot.dtype = detail::read_u32(p + 16);
        slot.rank = detail::read_u32(p + 20);
        for (std::uint32_t axis = 0; axis < 4; ++axis) {
            slot.dims[axis] = detail::read_u32(p + 24 + (4U * axis));
        }
        slot.byte_len = detail::read_u32(p + 40);
        offset += 44;
        if (slot.dtype != kDtypeF32) {
            *err = "slot " + slot.name + " dtype " + std::to_string(slot.dtype) + " is not F32";
            return false;
        }
        if (slot.rank > 4) {
            *err = "slot " + slot.name + " rank " + std::to_string(slot.rank) + " exceeds 4";
            return false;
        }
        if (slot.byte_len % sizeof(float) != 0) {
            *err = "slot " + slot.name + " byte_len is not a multiple of 4";
            return false;
        }
        if (slot.byte_len != 0 &&
            slot.byte_len != detail::slot_floats(slot) * static_cast<std::size_t>(sizeof(float))) {
            *err = "slot " + slot.name + " byte_len disagrees with its declared dims";
            return false;
        }
        if (bytes.size() < offset + slot.byte_len) {
            *err = "slot " + slot.name + " payload runs past the end of the blob";
            return false;
        }
        slot.payload = offset;
        offset += slot.byte_len;
    }
    if (offset != bytes.size()) {
        *err = "trailing bytes after the declared slots";
        return false;
    }

    // Every node's slots must exist, and the output of a node must not be a constant:
    // writing over a weight would make the next run of the same blob disagree.
    std::vector<bool> is_constant(out->slots.size(), false);
    for (std::size_t i = 0; i < out->slots.size(); ++i) {
        is_constant[i] = out->slots[i].byte_len != 0;
    }
    for (std::uint32_t i = 0; i < node_count; ++i) {
        const Node& node = out->nodes[i];
        const std::uint32_t refs[4] = {node.in_slot, node.out_slot, node.w_slot, node.b_slot};
        for (std::uint32_t k = 0; k < 4; ++k) {
            if (refs[k] != kNoSlot && refs[k] >= out->slots.size()) {
                *err = "node " + std::to_string(i) + " references slot " + std::to_string(refs[k]) +
                       " but only " + std::to_string(out->slots.size()) + " exist";
                return false;
            }
        }
        if (node.out_slot == kNoSlot) {
            *err = "node " + std::to_string(i) + " has no output slot";
            return false;
        }
        if (is_constant[node.out_slot]) {
            *err = "node " + std::to_string(i) + " writes into constant slot " +
                   out->slots[node.out_slot].name;
            return false;
        }
        if (node.op == kOpInput) {
            continue;
        }
        if (node.in_slot == kNoSlot) {
            *err = "node " + std::to_string(i) + " is not an input but has no input slot";
            return false;
        }
    }

    // Arena layout: **first-fit reuse over live ranges**, not the literal sum of every
    // runtime slot. A slot is released into the free list at the end of the node that
    // last touches it, so node i's output may land on a slot whose last use was node
    // i-1; two slots that are live at the same time never overlap, and a node never
    // lands on its own input (the release happens *after* the node runs).
    //
    // 布局是**按活跃区间首次适配复用**，不是"每个中间量各占一段再求和"。槽在其最后一个
    // 使用它的节点执行完之后才回到空闲表，所以第 i 个节点的输出可以落在第 i-1 个节点最后
    // 用过的槽上；同时活跃的两个槽永不重叠，节点也永不会覆盖自己的输入（释放发生在节点
    // 执行之后）。
    const std::size_t count = out->slots.size();
    std::vector<bool> is_runtime(count, false);
    std::vector<bool> is_written(count, false);
    std::vector<std::size_t> last_use(count, 0);
    for (std::size_t index = 0; index < count; ++index) {
        is_runtime[index] = out->slots[index].byte_len == 0;
    }
    for (std::size_t position = 0; position < out->nodes.size(); ++position) {
        const Node& node = out->nodes[position];
        const std::uint32_t refs[3] = {node.in_slot, node.w_slot, node.b_slot};
        for (std::uint32_t ref : refs) {
            if (ref != kNoSlot && is_runtime[ref]) {
                last_use[ref] = position;
            }
        }
        last_use[node.out_slot] = position;
        is_written[node.out_slot] = true;
    }
    // A runtime slot nobody writes has no offset to start from: the layout would be
    // built on an uninitialised value, so it is refused instead of guessed.
    for (std::size_t index = 0; index < count; ++index) {
        if (is_runtime[index] && !is_written[index]) {
            *err = "runtime slot " + out->slots[index].name + " is never written";
            return false;
        }
    }

    struct Free {
        std::size_t offset;
        std::size_t length;
    };
    std::vector<Free> free_blocks;
    std::size_t tail = 0;
    for (std::size_t position = 0; position < out->nodes.size(); ++position) {
        Slot& produced = out->slots[out->nodes[position].out_slot];
        const std::size_t need = detail::slot_floats(produced);
        bool placed = false;
        for (std::size_t block = 0; block < free_blocks.size(); ++block) {
            if (free_blocks[block].length >= need) {
                produced.arena = free_blocks[block].offset;
                if (free_blocks[block].length > need) {
                    free_blocks[block].offset += need;
                    free_blocks[block].length -= need;
                } else {
                    free_blocks.erase(free_blocks.begin() + static_cast<std::ptrdiff_t>(block));
                }
                placed = true;
                break;
            }
        }
        if (!placed) {
            produced.arena = tail;
            tail += need;
        }
        for (std::size_t index = 0; index < count; ++index) {
            if (is_runtime[index] && last_use[index] == position) {
                free_blocks.push_back(Free{out->slots[index].arena, detail::slot_floats(out->slots[index])});
            }
        }
        std::sort(free_blocks.begin(), free_blocks.end(),
                  [](const Free& lhs, const Free& rhs) { return lhs.offset < rhs.offset; });
    }
    out->arena_floats = tail;
    out->bytes = std::move(bytes);
    return true;
}

// The slot the caller must fill before `run()`: the output of the first `input` node.
inline std::uint32_t input_slot(const Blob& blob) {
    for (const Node& node : blob.nodes) {
        if (node.op == kOpInput) {
            return node.out_slot;
        }
    }
    return kNoSlot;
}

// The slot `run()` produces: the output of the last node.
inline std::uint32_t output_slot(const Blob& blob) {
    return blob.nodes.empty() ? kNoSlot : blob.nodes.back().out_slot;
}

namespace detail {

inline float* slot_data(Blob& blob, float* arena, std::uint32_t index) {
    Slot& slot = blob.slots[index];
    if (slot.byte_len != 0) {
        return reinterpret_cast<float*>(blob.bytes.data() + slot.payload);
    }
    return arena + slot.arena;
}

inline const float* slot_const(const Blob& blob, std::uint32_t index) {
    const Slot& slot = blob.slots[index];
    if (slot.byte_len != 0) {
        return reinterpret_cast<const float*>(blob.bytes.data() + slot.payload);
    }
    return nullptr;  // runtime slots are only reachable through slot_data()
}

inline bool bind(Blob& blob, float* arena, std::uint32_t index, infr_tensor* tensor, std::string* err) {
    const Slot& slot = blob.slots[index];
    infr_shape shape{};
    const infr_status status = infr_shape_init(&shape, static_cast<std::uint8_t>(slot.rank), slot.dims);
    if (status != INFR_OK) {
        *err = "slot " + slot.name + ": infr_shape_init failed: " + infr_status_str(status);
        return false;
    }
    const infr_status bound = infr_tensor_init(tensor, slot_data(blob, arena, index), INFR_DTYPE_F32, &shape);
    if (bound != INFR_OK) {
        *err = "slot " + slot.name + ": infr_tensor_init failed: " + infr_status_str(bound);
        return false;
    }
    return true;
}

}  // namespace detail

// Execute the graph. `arena` must hold at least `blob.arena_floats` floats and the
// caller fills `input_slot()` before calling. The result lands in `output_slot()`.
//
// 执行图。`arena` 至少要有 `blob.arena_floats` 个 float，调用前填好 `input_slot()`，
// 结果落在 `output_slot()`。形状由每个算子**自己重新推导**并与槽声明比对，所以生成器
// 把形状写错时在这里红，而不是让垃圾浮点数流到下游。
inline bool run(Blob& blob, float* arena, std::size_t arena_len, infr_workspace* workspace, std::string* err) {
    if (arena_len < blob.arena_floats) {
        *err = "arena too small: need " + std::to_string(blob.arena_floats) + " floats";
        return false;
    }
    if (infr_kernel_register_scalar() != INFR_OK) {
        *err = "infr_kernel_register_scalar failed";
        return false;
    }

    for (std::uint32_t i = 0; i < blob.nodes.size(); ++i) {
        const Node& node = blob.nodes[i];
        const std::string where = "node " + std::to_string(i);
        infr_tensor out_t{};
        if (node.op == kOpInput) {
            continue;
        }
        if (!detail::bind(blob, arena, node.out_slot, &out_t, err)) {
            *err = where + ": " + *err;
            return false;
        }
        const Slot& out_slot = blob.slots[node.out_slot];
        infr_status status = INFR_OK;

        if (node.op == kOpConv2d) {
            infr_tensor in_t{};
            infr_tensor w_t{};
            if (!detail::bind(blob, arena, node.in_slot, &in_t, err) ||
                !detail::bind(blob, arena, node.w_slot, &w_t, err)) {
                *err = where + ": " + *err;
                return false;
            }
            const Slot& w_slot = blob.slots[node.w_slot];
            if (w_slot.rank != 4 || in_t.shape.dims[3] % node.p[3] != 0) {
                *err = where + ": filter rank or groups do not divide the input channels";
                return false;
            }
            infr_conv2d_params params{};
            params.kernel_h = node.p[0];
            params.kernel_w = node.p[0];
            params.stride_h = node.p[1];
            params.stride_w = node.p[1];
            params.pad_h = node.p[2];
            params.pad_w = node.p[2];
            params.in_channels = in_t.shape.dims[3];
            params.out_channels = w_slot.dims[0];
            params.groups = node.p[3];
            if (w_slot.dims[1] != node.p[0] || w_slot.dims[2] != node.p[0] ||
                w_slot.dims[3] != params.in_channels / params.groups) {
                *err = where + ": filter dims are not [oc, kernel, kernel, in/groups]";
                return false;
            }
            infr_shape derived{};
            status = infr_conv2d_output_shape(&params, &in_t.shape, &derived);
            if (status != INFR_OK) {
                *err = where + ": infr_conv2d_output_shape: " + infr_status_str(status);
                return false;
            }
            if (!detail::same_dims(out_slot, derived)) {
                *err = where + ": declared output dims disagree with the convolution geometry";
                return false;
            }
            const float* bias = node.b_slot == kNoSlot ? nullptr : detail::slot_const(blob, node.b_slot);
            status = infr_conv2d_f32(&in_t, &w_t, bias, &params, &out_t, workspace);
        } else if (node.op == kOpMaxPool2d) {
            infr_tensor in_t{};
            if (!detail::bind(blob, arena, node.in_slot, &in_t, err)) {
                *err = where + ": " + *err;
                return false;
            }
            infr_pool_params params{};
            params.kernel_h = node.p[0];
            params.kernel_w = node.p[0];
            params.stride_h = node.p[1];
            params.stride_w = node.p[1];
            params.pad_h = node.p[2];
            params.pad_w = node.p[2];
            params.channels = in_t.shape.dims[3];
            infr_shape derived{};
            status = infr_max_pool2d_output_shape(&params, &in_t.shape, &derived);
            if (status != INFR_OK) {
                *err = where + ": infr_max_pool2d_output_shape: " + infr_status_str(status);
                return false;
            }
            if (!detail::same_dims(out_slot, derived)) {
                *err = where + ": declared output dims disagree with the pooling geometry";
                return false;
            }
            status = infr_max_pool2d_f32(&in_t, &params, &out_t, workspace);
        } else if (node.op == kOpMeanHw) {
            infr_tensor in_t{};
            if (!detail::bind(blob, arena, node.in_slot, &in_t, err)) {
                *err = where + ": " + *err;
                return false;
            }
            status = infr_mean_hw_f32(&in_t, &out_t, workspace);
        } else if (node.op == kOpFc) {
            infr_tensor in_t{};
            infr_tensor w_t{};
            if (!detail::bind(blob, arena, node.in_slot, &in_t, err) ||
                !detail::bind(blob, arena, node.w_slot, &w_t, err)) {
                *err = where + ": " + *err;
                return false;
            }
            const Slot& w_slot = blob.slots[node.w_slot];
            if (w_slot.rank != 2 || detail::slot_floats(blob.slots[node.in_slot]) != w_slot.dims[1]) {
                *err = where + ": fc weight is not [out, in] or its input extent disagrees";
                return false;
            }
            const infr_kernel_desc* fc =
                infr_kernel_find("fc", INFR_DTYPE_F32, INFR_QUANT_NONE, INFR_TIER_SCALAR);
            if (fc == nullptr) {
                *err = where + ": the fc kernel is not registered";
                return false;
            }
            float* out_data = detail::slot_data(blob, arena, node.out_slot);
            const float* in_data = detail::slot_data(blob, arena, node.in_slot);
            const float* w_data = detail::slot_const(blob, node.w_slot);
            const float* b_data = node.b_slot == kNoSlot ? nullptr : detail::slot_const(blob, node.b_slot);
            for (std::uint32_t oc = 0; oc < w_slot.dims[0]; ++oc) {
                float acc = 0.0F;
                const void* dot_inputs[2] = {in_data, w_data + (static_cast<std::size_t>(oc) * w_slot.dims[1])};
                void* dot_output[1] = {&acc};
                const infr_status dot = fc->fn(dot_inputs, dot_output, w_slot.dims[1], nullptr);
                if (dot != INFR_OK) {
                    *err = where + ": fc kernel failed: " + infr_status_str(dot);
                    return false;
                }
                out_data[oc] = acc + (b_data == nullptr ? 0.0F : b_data[oc]);
            }
        } else if (node.op == kOpMul || node.op == kOpAdd) {
            /* A binary node reuses the two slot references a weightful node already has:
             * `in_slot` and `w_slot` are the operands, `b_slot` stays empty. The slot table
             * keeps working because a runtime slot carries no payload either way. `add` and
             * `mul` differ only in which registered kernel they call. */
            infr_tensor a_t{};
            infr_tensor b_t{};
            if (!detail::bind(blob, arena, node.in_slot, &a_t, err) ||
                !detail::bind(blob, arena, node.w_slot, &b_t, err)) {
                *err = where + ": " + *err;
                return false;
            }
            const std::size_t count = detail::slot_floats(out_slot);
            /* `p[0] == 1`: the right operand is one factor per output channel, not a tensor
             * of the same shape. MobileNet's squeeze-and-excite gate is this shape (a
             * 1x1xC vector times an HxWxC map); spelling it as a full-size constant would
             * cost 1.7 MB of weights to say 96 numbers. It is carried out by the registered
             * `scale` kernel -- `y[i] = a[i] * factor` with the factor in a one-element
             * buffer -- called once per channel, so no arithmetic is re-implemented here.
             *
             * `p[0] == 1`：右操作数是每个输出通道一个因子，而不是同形状张量。MobileNet 的
             * SE 门就是这个形状；写成整尺寸常量等于花 1.7 MB 权重去表达 96 个数。计算交给
             * 已注册的 `scale` 核（`y[i] = a[i] * factor`，因子放在单元素缓冲里）按通道各调
             * 一次，这里不重写任何算术。 */
            if (node.p[0] == 1) {
                const std::size_t channels = out_slot.dims[3];
                if (node.op != kOpMul || channels == 0 || count == 0 || count % channels != 0 ||
                    detail::slot_floats(blob.slots[node.in_slot]) != count ||
                    detail::slot_floats(blob.slots[node.w_slot]) != channels) {
                    *err = where + ": a per-channel scale needs (H,W,C) times (C,)";
                    return false;
                }
                const std::size_t plane = count / channels;
                const infr_kernel_desc* scale =
                    infr_kernel_find("scale", INFR_DTYPE_F32, INFR_QUANT_NONE, INFR_TIER_SCALAR);
                if (scale == nullptr) {
                    *err = where + ": the scale kernel is not registered";
                    return false;
                }
                float* out_data = detail::slot_data(blob, arena, node.out_slot);
                const float* in_data = detail::slot_data(blob, arena, node.in_slot);
                const float* factors = detail::slot_data(blob, arena, node.w_slot);
                for (std::size_t channel = 0; channel < channels; ++channel) {
                    const void* scaled_in[2] = {in_data + channel * plane, factors + channel};
                    void* scaled_out[1] = {out_data + channel * plane};
                    status = scale->fn(scaled_in, scaled_out, static_cast<std::uint32_t>(plane), nullptr);
                    if (status != INFR_OK) {
                        break;
                    }
                }
                if (status != INFR_OK) {
                    *err = where + ": scale (per channel) failed: " + infr_status_str(status);
                    return false;
                }
                if (node.act != 0) {
                    const infr_act_f32_fn act = infr_activation_f32(static_cast<infr_activation>(node.act));
                    if (act == nullptr) {
                        *err = where + ": activation " + std::to_string(node.act) + " is not implemented";
                        return false;
                    }
                    act(out_data, out_data, static_cast<std::uint32_t>(count));
                }
                continue;
            }
            if (detail::slot_floats(blob.slots[node.in_slot]) != count ||
                detail::slot_floats(blob.slots[node.w_slot]) != count) {
                *err = where + ": binary operands disagree with the output extent";
                return false;
            }
            const char* kernel = node.op == kOpMul ? "mul" : "add";
            const infr_kernel_desc* binary =
                infr_kernel_find(kernel, INFR_DTYPE_F32, INFR_QUANT_NONE, INFR_TIER_SCALAR);
            if (binary == nullptr) {
                *err = where + ": the " + kernel + " kernel is not registered";
                return false;
            }
            const void* operands[2] = {detail::slot_data(blob, arena, node.in_slot),
                                      detail::slot_data(blob, arena, node.w_slot)};
            void* product[1] = {detail::slot_data(blob, arena, node.out_slot)};
            status = binary->fn(operands, product, static_cast<std::uint32_t>(count), nullptr);
        } else {
            *err = where + ": unknown op " + std::to_string(node.op);
            return false;
        }

        if (status != INFR_OK) {
            *err = where + ": operator failed: " + infr_status_str(status);
            return false;
        }
        if (node.act != 0) {
            const infr_act_f32_fn act = infr_activation_f32(static_cast<infr_activation>(node.act));
            if (act == nullptr) {
                *err = where + ": activation " + std::to_string(node.act) + " is not implemented";
                return false;
            }
            float* out_data = detail::slot_data(blob, arena, node.out_slot);
            act(out_data, out_data, static_cast<std::uint32_t>(detail::slot_floats(out_slot)));
        }
    }
    return true;
}

}  // namespace infrb
