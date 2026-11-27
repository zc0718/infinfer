// Conan::ImportStart
#include <stdint.h>
#include <string.h>

#include <infinfer/infinfer.h>

#include "blob_graph.h"
// Conan::ImportEnd

/*!
 * @brief [en] The C executor for `INFRKAT` v2 blobs (see `blob_graph.h`).
 * @brief [zh] `INFRKAT` v2 blob 的 C 执行器（见 `blob_graph.h`）。
 * @note [en] Ordered the way the format is: header, node table, slot table. The slot
 *            table is walked first because it is the payload that tells us where the
 *            weights are, and the nodes then only carry indices.
 * @note [zh] 顺序与格式一致：头、节点表、槽表。先走槽表，因为权重的位置写在槽里；节点只带
 *            索引。
 * @since 0.1.0
 */

#define KAT_BLOB_HEADER 24U
#define KAT_BLOB_NODE 36U
#define KAT_BLOB_SLOT 44U

#define KAT_BLOB_MAX_NODES 256U
#define KAT_BLOB_MAX_SLOTS 256U

#define KAT_BLOB_NO_SLOT 0xFFFFFFFFU
#define KAT_BLOB_OP_INPUT 0U
#define KAT_BLOB_OP_CONV2D 1U
#define KAT_BLOB_OP_MAX_POOL2D 2U
#define KAT_BLOB_OP_MEAN_HW 3U
#define KAT_BLOB_OP_FC 4U
#define KAT_BLOB_OP_MUL 5U
#define KAT_BLOB_OP_ADD 6U

typedef struct kat_blob_slot {
    uint32_t rank;
    uint32_t dims[4];
    uint32_t byte_len;
    uint32_t offset;
    uint32_t arena;
} kat_blob_slot;

typedef struct kat_blob_node {
    uint8_t op;
    uint8_t act;
    uint32_t in_slot;
    uint32_t out_slot;
    uint32_t w_slot;
    uint32_t b_slot;
    uint32_t p[4];
} kat_blob_node;

static uint32_t rd_u32(const unsigned char *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static uint32_t slot_floats(const kat_blob_slot *slot) {
    uint32_t count = 1U;
    uint32_t axis;
    for (axis = 0U; axis < slot->rank; ++axis) {
        count *= slot->dims[axis];
    }
    return count;
}

static int same_dims(const kat_blob_slot *slot, const infr_shape *shape) {
    uint32_t axis;
    if (slot->rank != (uint32_t)shape->rank) {
        return 0;
    }
    for (axis = 0U; axis < slot->rank; ++axis) {
        if (slot->dims[axis] != shape->dims[axis]) {
            return 0;
        }
    }
    return 1;
}

/* A tensor view over a slot. Constants live in the blob, runtime slots in the arena;
 * the cast drops const only because `infr_tensor_init` has no const form. */
static int bind_slot(const unsigned char *bytes, const kat_blob_slot *slot, float *arena, infr_tensor *tensor) {
    infr_shape shape;
    void *data;
    if (infr_shape_init(&shape, (uint8_t)slot->rank, slot->dims) != INFR_OK) {
        return 0;
    }
    data =
        slot->byte_len != 0U ? (void *)(uintptr_t)(const void *)(bytes + slot->offset) : (void *)(arena + slot->arena);
    return infr_tensor_init(tensor, data, INFR_DTYPE_F32, &shape) == INFR_OK;
}

static const float *constant_of(const unsigned char *bytes, const kat_blob_slot *slot) {
    return (const float *)(const void *)(bytes + slot->offset);
}

/* A slot's numbers, wherever they live: constants in the blob's flash image, runtime slots in
 * the arena. The per-channel gate needs both cases -- a stored factor vector, and one a previous
 * node computed (a real squeeze-and-excite block computes its gate; a constant one is not SE).
 *
 * 一个槽的数字，不管它住在哪里：常量在 blob 的 flash 映像里，运行期槽在 arena 里。逐通道门
 * 两种都需要——存起来的因子向量，以及上一个节点算出来的那个（真的 SE 块是算出来的；常量门
 * 不是 SE）。 */
static const float *floats_of(const unsigned char *bytes, const kat_blob_slot *slot, const float *arena) {
    return slot->byte_len != 0U ? constant_of(bytes, slot) : (arena + slot->arena);
}

int kat_blob_run(const unsigned char *bytes, uint32_t byte_len, const float *input, uint32_t input_len, float *arena,
                 uint32_t arena_len, infr_workspace *workspace, kat_blob_view *view) {
    static const unsigned char magic[8] = {'I', 'N', 'F', 'R', 'K', 'A', 'T', '\0'};
    /* `static`, not automatic: at 256 slots these tables are ~18 KB, which is a normal
     * .bss reservation on an MCU but a stack overflow waiting to happen. The price is
     * that `kat_blob_run` is **not reentrant** -- one graph at a time, which is what a
     * single-counter inference job is. 256 is not a guess either: the MobileNetV3-Small
     * blob is 80 nodes / 188 slots, so 32 (the old ceiling) could not express it and the
     * network was therefore never actually executed.
     *
     * 用 `static` 而不是自动变量：256 个槽时这两张表约 18 KB，在 MCU 上是一块正常的 .bss，
     * 放在栈上就是等着溢出。代价是 `kat_blob_run` **不可重入**——一次跑一张图，而单片机的
     * 推理作业本来就是这样。256 也不是拍的：MobileNetV3-Small 的 blob 是 80 个节点 / 188
     * 个槽，旧上限 32 根本表达不了它，那张网络因此从未被真正执行过。 */
    static kat_blob_slot slots[KAT_BLOB_MAX_SLOTS];
    static kat_blob_node nodes[KAT_BLOB_MAX_NODES];
    static uint32_t last_use[KAT_BLOB_MAX_SLOTS];
    static uint8_t written[KAT_BLOB_MAX_SLOTS];
    /* One extra: while node i's output is being placed, every slot it releases is
     * inserted into a list that may already hold all the others.
     *
     * 多一格：给第 i 个节点定位输出时，它释放的槽要插进一个"可能已经装了其余全部"的表里。 */
    static uint32_t free_off[KAT_BLOB_MAX_SLOTS + 1U];
    static uint32_t free_len[KAT_BLOB_MAX_SLOTS + 1U];
    uint32_t free_count;
    uint32_t node_count;
    uint32_t slot_count;
    uint32_t offset;
    uint32_t cursor = 0U;
    uint32_t input_slot = KAT_BLOB_NO_SLOT;
    uint32_t i;
    uint32_t k;

    if (bytes == 0 || view == 0 || workspace == 0 || arena == 0) {
        return KAT_BLOB_ERR_HEADER;
    }
    if (byte_len < KAT_BLOB_HEADER || memcmp(bytes, magic, sizeof magic) != 0) {
        return KAT_BLOB_ERR_HEADER;
    }
    if (rd_u32(bytes + 8) != 2U || rd_u32(bytes + 20) != 0U) {
        return KAT_BLOB_ERR_VERSION;
    }
    node_count = rd_u32(bytes + 12);
    slot_count = rd_u32(bytes + 16);
    if (node_count > KAT_BLOB_MAX_NODES || slot_count > KAT_BLOB_MAX_SLOTS) {
        return KAT_BLOB_ERR_TABLE;
    }
    offset = KAT_BLOB_HEADER + (node_count * KAT_BLOB_NODE);
    if (offset > byte_len) {
        return KAT_BLOB_ERR_TABLE;
    }
    for (i = 0U; i < node_count; ++i) {
        const unsigned char *p = bytes + KAT_BLOB_HEADER + (i * KAT_BLOB_NODE);
        if ((uint32_t)p[2] != 0U || (uint32_t)p[3] != 0U) {
            return KAT_BLOB_ERR_TABLE;
        }
        nodes[i].op = p[0];
        nodes[i].act = p[1];
        nodes[i].in_slot = rd_u32(p + 4);
        nodes[i].out_slot = rd_u32(p + 8);
        nodes[i].w_slot = rd_u32(p + 12);
        nodes[i].b_slot = rd_u32(p + 16);
        for (k = 0U; k < 4U; ++k) {
            nodes[i].p[k] = rd_u32(p + 20 + (4U * k));
        }
    }

    /* Slot table: payload offsets first, arena layout second. */
    for (i = 0U; i < slot_count; ++i) {
        const unsigned char *p;
        if (offset + KAT_BLOB_SLOT > byte_len) {
            return KAT_BLOB_ERR_SLOT;
        }
        p = bytes + offset;
        slots[i].rank = rd_u32(p + 20);
        for (k = 0U; k < 4U; ++k) {
            slots[i].dims[k] = rd_u32(p + 24 + (4U * k));
        }
        slots[i].byte_len = rd_u32(p + 40);
        if (rd_u32(p + 16) != 1U || slots[i].rank > 4U) {
            return KAT_BLOB_ERR_SLOT;
        }
        offset += KAT_BLOB_SLOT;
        if (slots[i].byte_len != 0U) {
            if (slots[i].byte_len != slot_floats(&slots[i]) * (uint32_t)sizeof(float)) {
                return KAT_BLOB_ERR_SLOT;
            }
            if (offset + slots[i].byte_len > byte_len) {
                return KAT_BLOB_ERR_SLOT;
            }
            slots[i].offset = offset;
            slots[i].arena = 0U;
            offset += slots[i].byte_len;
        } else {
            slots[i].offset = 0U;
            slots[i].arena = 0U; /* assigned by the layout pass below */
        }
    }
    if (offset != byte_len) {
        return KAT_BLOB_ERR_SLOT;
    }

    /* Layout: first-fit reuse over live ranges. A runtime slot goes back into the free
     * list after the node that last touches it, so node i's output may land on a slot
     * whose last use was node i-1. Slots that are live at the same time never overlap,
     * and a node never lands on its own input (the release happens after the node).
     *
     * 布局：按活跃区间首次适配复用。槽在最后一个碰它的节点之后才回到空闲表，所以第 i 个
     * 节点的输出可以落在第 i-1 个节点最后用过的槽上。同时活跃的槽永不重叠，节点也永不会
     * 覆盖自己的输入（释放发生在节点之后）。 */
    for (i = 0U; i < slot_count; ++i) {
        last_use[i] = 0U;
        written[i] = 0;
    }
    for (i = 0U; i < node_count; ++i) {
        const uint32_t refs[3] = {nodes[i].in_slot, nodes[i].w_slot, nodes[i].b_slot};
        for (k = 0U; k < 3U; ++k) {
            if (refs[k] != KAT_BLOB_NO_SLOT && slots[refs[k]].byte_len == 0U) {
                last_use[refs[k]] = i;
            }
        }
        last_use[nodes[i].out_slot] = i;
        written[nodes[i].out_slot] = 1;
    }
    for (i = 0U; i < slot_count; ++i) {
        if (slots[i].byte_len == 0U && written[i] == 0) {
            return KAT_BLOB_ERR_SLOT; /* no writer: there is no offset to reuse */
        }
    }
    cursor = 0U;
    free_count = 0U;
    for (i = 0U; i < node_count; ++i) {
        uint32_t need = slot_floats(&slots[nodes[i].out_slot]);
        uint32_t placed = 0U;
        int found = 0;
        for (k = 0U; k < free_count; ++k) {
            if (free_len[k] >= need) {
                placed = free_off[k];
                if (free_len[k] > need) {
                    free_off[k] += need;
                    free_len[k] -= need;
                } else {
                    uint32_t move;
                    for (move = k; move + 1U < free_count; ++move) {
                        free_off[move] = free_off[move + 1U];
                        free_len[move] = free_len[move + 1U];
                    }
                    --free_count;
                }
                found = 1;
                break;
            }
        }
        if (found == 0) {
            placed = cursor;
            cursor += need;
        }
        slots[nodes[i].out_slot].arena = placed;
        /* Release every slot whose last use was this node, keeping the list sorted by
         * offset so the scan above is first-fit. */
        for (k = 0U; k < slot_count; ++k) {
            if (slots[k].byte_len == 0U && last_use[k] == i && written[k] != 0) {
                uint32_t at = free_count;
                while (at > 0U && free_off[at - 1U] > slots[k].arena) {
                    free_off[at] = free_off[at - 1U];
                    free_len[at] = free_len[at - 1U];
                    --at;
                }
                free_off[at] = slots[k].arena;
                free_len[at] = slot_floats(&slots[k]);
                ++free_count;
            }
        }
    }
    if (cursor > arena_len) {
        return KAT_BLOB_ERR_ARENA;
    }
    view->arena_floats = cursor;

    /* Every reference has to resolve, and no node may write into a constant. */
    for (i = 0U; i < node_count; ++i) {
        const uint32_t refs[4] = {nodes[i].in_slot, nodes[i].out_slot, nodes[i].w_slot, nodes[i].b_slot};
        for (k = 0U; k < 4U; ++k) {
            if (refs[k] != KAT_BLOB_NO_SLOT && refs[k] >= slot_count) {
                return KAT_BLOB_ERR_TABLE;
            }
        }
        if (nodes[i].out_slot == KAT_BLOB_NO_SLOT || slots[nodes[i].out_slot].byte_len != 0U) {
            return KAT_BLOB_ERR_TABLE;
        }
        if (nodes[i].op == KAT_BLOB_OP_INPUT) {
            if (input_slot == KAT_BLOB_NO_SLOT) {
                input_slot = nodes[i].out_slot;
            }
        } else if (nodes[i].in_slot == KAT_BLOB_NO_SLOT) {
            return KAT_BLOB_ERR_TABLE;
        }
    }

    if (input_slot != KAT_BLOB_NO_SLOT) {
        if (input == 0 || input_len != slot_floats(&slots[input_slot])) {
            return KAT_BLOB_ERR_INPUT;
        }
        memcpy(arena + slots[input_slot].arena, input, (size_t)input_len * sizeof(float));
    } else if (input != 0) {
        return KAT_BLOB_ERR_INPUT;
    }

    for (i = 0U; i < node_count; ++i) {
        const kat_blob_node *node = &nodes[i];
        const kat_blob_slot *out_slot;
        infr_tensor out_t;
        infr_shape derived;
        infr_status status;
        if (node->op == KAT_BLOB_OP_INPUT) {
            continue;
        }
        out_slot = &slots[node->out_slot];
        if (!bind_slot(bytes, out_slot, arena, &out_t)) {
            return KAT_BLOB_ERR_SLOT;
        }
        if (node->op == KAT_BLOB_OP_CONV2D) {
            const kat_blob_slot *w_slot = &slots[node->w_slot];
            infr_tensor in_t;
            infr_tensor w_t;
            infr_conv2d_params params;
            if (!bind_slot(bytes, &slots[node->in_slot], arena, &in_t) || !bind_slot(bytes, w_slot, arena, &w_t)) {
                return KAT_BLOB_ERR_SLOT;
            }
            if (w_slot->rank != 4U || node->p[3] == 0U || in_t.shape.dims[3] % node->p[3] != 0U) {
                return KAT_BLOB_ERR_SHAPE;
            }
            params.kernel_h = node->p[0];
            params.kernel_w = node->p[0];
            params.stride_h = node->p[1];
            params.stride_w = node->p[1];
            params.pad_h = node->p[2];
            params.pad_w = node->p[2];
            params.in_channels = in_t.shape.dims[3];
            params.out_channels = w_slot->dims[0];
            params.groups = node->p[3];
            if (w_slot->dims[1] != node->p[0] || w_slot->dims[2] != node->p[0] ||
                w_slot->dims[3] != params.in_channels / params.groups) {
                return KAT_BLOB_ERR_SHAPE;
            }
            if (infr_conv2d_output_shape(&params, &in_t.shape, &derived) != INFR_OK) {
                return KAT_BLOB_ERR_SHAPE;
            }
            if (!same_dims(out_slot, &derived)) {
                return KAT_BLOB_ERR_SHAPE;
            }
            status = infr_conv2d_f32(&in_t, &w_t,
                                     node->b_slot == KAT_BLOB_NO_SLOT ? 0 : constant_of(bytes, &slots[node->b_slot]),
                                     &params, &out_t, workspace);
        } else if (node->op == KAT_BLOB_OP_MAX_POOL2D) {
            infr_tensor in_t;
            infr_pool_params params;
            if (!bind_slot(bytes, &slots[node->in_slot], arena, &in_t)) {
                return KAT_BLOB_ERR_SLOT;
            }
            params.kernel_h = node->p[0];
            params.kernel_w = node->p[0];
            params.stride_h = node->p[1];
            params.stride_w = node->p[1];
            params.pad_h = node->p[2];
            params.pad_w = node->p[2];
            params.channels = in_t.shape.dims[3];
            if (infr_max_pool2d_output_shape(&params, &in_t.shape, &derived) != INFR_OK) {
                return KAT_BLOB_ERR_SHAPE;
            }
            if (!same_dims(out_slot, &derived)) {
                return KAT_BLOB_ERR_SHAPE;
            }
            status = infr_max_pool2d_f32(&in_t, &params, &out_t, workspace);
        } else if (node->op == KAT_BLOB_OP_MEAN_HW) {
            infr_tensor in_t;
            if (!bind_slot(bytes, &slots[node->in_slot], arena, &in_t)) {
                return KAT_BLOB_ERR_SLOT;
            }
            status = infr_mean_hw_f32(&in_t, &out_t, workspace);
        } else if (node->op == KAT_BLOB_OP_FC) {
            const kat_blob_slot *w_slot = &slots[node->w_slot];
            const infr_kernel_desc *fc;
            infr_tensor in_t;
            infr_tensor w_t;
            const float *w_data;
            const float *b_data;
            float *out_data = arena + out_slot->arena;
            const float *in_data = arena + slots[node->in_slot].arena;
            uint32_t oc;
            if (!bind_slot(bytes, &slots[node->in_slot], arena, &in_t) || !bind_slot(bytes, w_slot, arena, &w_t)) {
                return KAT_BLOB_ERR_SLOT;
            }
            if (w_slot->rank != 2U || slot_floats(&slots[node->in_slot]) != w_slot->dims[1] ||
                out_slot->dims[3] != w_slot->dims[0]) {
                return KAT_BLOB_ERR_SHAPE;
            }
            if (infr_kernel_register_scalar() != INFR_OK) {
                return INFR_ERR_NOT_IMPLEMENTED;
            }
            fc = infr_kernel_find("fc", INFR_DTYPE_F32, INFR_QUANT_NONE, INFR_TIER_SCALAR);
            if (fc == 0) {
                return INFR_ERR_NOT_IMPLEMENTED;
            }
            w_data = constant_of(bytes, w_slot);
            b_data = node->b_slot == KAT_BLOB_NO_SLOT ? 0 : constant_of(bytes, &slots[node->b_slot]);
            status = INFR_OK;
            for (oc = 0U; oc < w_slot->dims[0]; ++oc) {
                const void *inputs[2];
                void *outputs[1];
                float acc = 0.0F;
                inputs[0] = in_data;
                inputs[1] = &w_data[oc * w_slot->dims[1]];
                outputs[0] = &acc;
                status = fc->fn(inputs, outputs, w_slot->dims[1], 0);
                if (status != INFR_OK) {
                    break;
                }
                out_data[oc] = acc + (b_data == 0 ? 0.0F : b_data[oc]);
            }
        } else if (node->op == KAT_BLOB_OP_MUL || node->op == KAT_BLOB_OP_ADD) {
            /* Binary node: `in_slot` and `w_slot` are the two operands (the field a
             * weightful node uses for its filter), `b_slot` stays empty. */
            const kat_blob_slot *a_slot = &slots[node->in_slot];
            const kat_blob_slot *b_slot = &slots[node->w_slot];
            const infr_kernel_desc *binary;
            const char *kernel = node->op == KAT_BLOB_OP_MUL ? "mul" : "add";
            const uint32_t count = slot_floats(out_slot);
            const void *operands[2];
            void *product[1];
            if (infr_kernel_register_scalar() != INFR_OK) {
                return INFR_ERR_NOT_IMPLEMENTED;
            }
            /* `p[0] == 1`: the right operand is **one factor per output channel** rather
             * than a tensor of the same shape -- MobileNet's squeeze-and-excite gate is
             * exactly that (a 1x1xC vector times an HxWxC feature map), and writing it as
             * a full-size constant would cost 1.7 MB of weights to say 96 numbers.
             *
             * The map is NHWC, so the elements that share a factor sit at `pixel * C + c`
             * for a fixed `c`: they are **C apart**, not adjacent. `scale` multiplies one
             * contiguous run by one factor, so calling it once per channel would take
             * channel 0's first element, channel 1's first element, ... as one run -- the
             * wrong factors, on a map with more than one pixel. `mul` over one pixel's C
             * elements is the same arithmetic with the access pattern the layout really
             * has: one call per pixel, no new claim, and no materialised gate.
             *
             * `p[0] == 1`：右操作数是**每个输出通道一个因子**而不是同形状张量——MobileNet
             * 的 SE 门正是这个形状，写成整尺寸常量等于花 1.7 MB 权重去表达 96 个数。特征图
             * 是 NHWC，共享同一因子的元素下标为 `pixel * C + c`，**相隔 C** 而不是相邻；
             * `scale` 是“一段连续元素乘同一个因子”，按通道各调一次等于把各通道首元素当成
             * 同一通道。改为对一个像素的 C 个元素调一次 `mul`：算术相同、访问顺序正确，
             * 不需要新 claim，也不需要物化门向量。旧写法乘错因子这件事见 gaps G-103。 */
            if (node->p[0] == 1U) {
                const uint32_t channels = out_slot->dims[3];
                const uint32_t pixels = channels == 0U ? 0U : count / channels;
                const infr_kernel_desc *gate;
                float *out_data = arena + out_slot->arena;
                const float *in_data = arena + a_slot->arena;
                const float *factors = floats_of(bytes, b_slot, arena);
                uint32_t pixel;
                if (node->op != KAT_BLOB_OP_MUL || count == 0U || channels == 0U || pixels * channels != count ||
                    slot_floats(a_slot) != count || slot_floats(b_slot) != channels) {
                    return KAT_BLOB_ERR_SHAPE;
                }
                gate = infr_kernel_find("mul", INFR_DTYPE_F32, INFR_QUANT_NONE, INFR_TIER_SCALAR);
                if (gate == 0) {
                    return INFR_ERR_NOT_IMPLEMENTED;
                }
                status = INFR_OK;
                for (pixel = 0U; pixel < pixels; ++pixel) {
                    const void *gate_in[2];
                    void *gate_out[1];
                    gate_in[0] = in_data + ((size_t)pixel * channels);
                    gate_in[1] = factors;
                    gate_out[0] = out_data + ((size_t)pixel * channels);
                    status = gate->fn(gate_in, gate_out, channels, 0);
                    if (status != INFR_OK) {
                        break;
                    }
                }
                if (status != INFR_OK) {
                    return (int)status;
                }
                if (node->act != 0U) {
                    const infr_act_f32_fn act = infr_activation_f32((infr_activation)node->act);
                    if (act == 0) {
                        return KAT_BLOB_ERR_OP;
                    }
                    act(out_data, out_data, count);
                }
                continue;
            }
            if (slot_floats(a_slot) != count || slot_floats(b_slot) != count) {
                return KAT_BLOB_ERR_SHAPE;
            }
            binary = infr_kernel_find(kernel, INFR_DTYPE_F32, INFR_QUANT_NONE, INFR_TIER_SCALAR);
            if (binary == 0) {
                return INFR_ERR_NOT_IMPLEMENTED;
            }
            /* Both operands through `floats_of`: a residual `add` has two runtime slots and an
             * equal-extent `mul` against a stored tensor has one of each. Reading the arena for
             * a constant (or the flash image for a runtime slot) is a wrong pointer either way.
             *
             * 两个操作数都走 `floats_of`：残差 `add` 两侧都是运行期槽，而等长的 `mul` 对着存好的
             * 常量时一侧是常量。对常量读 arena（或对运行期槽读 flash 映像）都是错指针。 */
            operands[0] = floats_of(bytes, a_slot, arena);
            operands[1] = floats_of(bytes, b_slot, arena);
            product[0] = arena + out_slot->arena;
            status = binary->fn(operands, product, count, 0);
        } else {
            return KAT_BLOB_ERR_OP;
        }
        if (status != INFR_OK) {
            return (int)status;
        }
        if (node->act != 0U) {
            const infr_act_f32_fn act = infr_activation_f32((infr_activation)node->act);
            if (act == 0) {
                return KAT_BLOB_ERR_OP;
            }
            act(arena + out_slot->arena, arena + out_slot->arena, slot_floats(out_slot));
        }
    }

    if (node_count == 0U) {
        return KAT_BLOB_ERR_TABLE;
    }
    view->out = arena + slots[nodes[node_count - 1U].out_slot].arena;
    view->out_len = slot_floats(&slots[nodes[node_count - 1U].out_slot]);
    return KAT_BLOB_OK;
}

/* ---------------------------------------------------------------------------------------------
 * v3, int8. One operator, byte-addressed arena, and the field offsets the format froze.
 *
 * The offsets are written out as `+16`, `+44` and so on rather than reached through a packed
 * struct: a struct would make the compiler pick the padding, and the format's whole promise is
 * that the bytes are where the specification says. The C++ twin uses the same numbers, and the
 * host test that compares them is what keeps the two from drifting.
 *
 * 字段偏移写成 `+16`、`+44` 这样的字面量而不是走 packed struct：struct 会让编译器决定填充，
 * 而格式的全部承诺就是"字节在规范说的位置"。C++ 那份用的是同一组数字，比对两者的宿主测试
 * 就是它们不漂移的保证。
 * ------------------------------------------------------------------------------------------- */

#define KAT_I8_SLOT_HEADER 60U
#define NO_SLOT_I8 0xFFFFFFFFU
#define KAT_I8_MAX_SLOTS 8U
#define KAT_I8_MAX_NODES 4U
#define KAT_I8_DTYPE_F32 1U
#define KAT_I8_DTYPE_I8 2U
#define KAT_I8_DTYPE_I32 3U

typedef struct kat_i8_slot {
    uint32_t dtype;
    uint32_t rank;
    uint32_t dims[4];
    uint32_t byte_len;
    uint32_t payload;
    uint8_t quant_kind;
    float scale;
    int32_t zero_point;
    uint32_t scales_at;
    uint32_t scale_count;
} kat_i8_slot;

static float rd_f32(const unsigned char *p) {
    float value = 0.0F;
    memcpy(&value, p, sizeof value);
    return value;
}

static int32_t rd_i32(const unsigned char *p) { return (int32_t)rd_u32(p); }

int kat_blob_run_i8(const unsigned char *bytes, uint32_t byte_len, const int8_t *input, uint32_t input_len,
                    unsigned char *arena, uint32_t arena_len, infr_workspace *workspace, int8_t *out,
                    uint32_t out_len) {
    static const unsigned char magic[8] = {'I', 'N', 'F', 'R', 'K', 'A', 'T', '\0'};
    kat_i8_slot slots[KAT_I8_MAX_SLOTS];
    uint32_t node_out_slot[KAT_I8_MAX_NODES];
    uint32_t node_op[KAT_I8_MAX_NODES];
    uint32_t node_in_slot[KAT_I8_MAX_NODES];
    uint32_t node_w_slot[KAT_I8_MAX_NODES];
    uint32_t node_b_slot[KAT_I8_MAX_NODES];
    uint32_t node_p[KAT_I8_MAX_NODES][4];
    uint32_t node_count;
    uint32_t slot_count;
    uint32_t offset;
    uint32_t index;
    uint32_t input_slot = NO_SLOT_I8;
    uint32_t conv = NO_SLOT_I8;
    int32_t bias_local[32];
    float scales_local[32];
    infr_shape in_shape;
    infr_shape w_shape;
    infr_shape out_shape;
    infr_tensor in_tensor;
    infr_tensor w_tensor;
    infr_tensor out_tensor;
    infr_conv2d_params params;
    infr_status status;
    uint32_t in_count = 1U;
    uint32_t out_count = 1U;
    uint32_t axis;
    const kat_i8_slot *w_slot;
    const kat_i8_slot *b_slot;
    const kat_i8_slot *out_slot;

    if (bytes == NULL || input == NULL || arena == NULL || workspace == NULL || out == NULL) {
        return KAT_BLOB_ERR_INPUT;
    }
    if (byte_len < 24U || memcmp(bytes, magic, sizeof magic) != 0) {
        return KAT_BLOB_ERR_HEADER;
    }
    if (rd_u32(bytes + 8) != 3U) {
        return KAT_BLOB_ERR_VERSION;
    }
    node_count = rd_u32(bytes + 12);
    slot_count = rd_u32(bytes + 16);
    if (node_count > KAT_I8_MAX_NODES || slot_count > KAT_I8_MAX_SLOTS) {
        return KAT_BLOB_ERR_TABLE;
    }
    offset = 24U;
    if (byte_len < offset + (node_count * 36U)) {
        return KAT_BLOB_ERR_TABLE;
    }
    for (index = 0U; index < node_count; ++index) {
        const unsigned char *p = bytes + offset + (index * 36U);
        uint32_t k;
        node_op[index] = (uint32_t)p[0];
        node_in_slot[index] = rd_u32(p + 4);
        node_out_slot[index] = rd_u32(p + 8);
        node_w_slot[index] = rd_u32(p + 12);
        node_b_slot[index] = rd_u32(p + 16);
        for (k = 0U; k < 4U; ++k) {
            node_p[index][k] = rd_u32(p + 20U + (4U * k));
        }
    }
    offset += node_count * 36U;
    for (index = 0U; index < slot_count; ++index) {
        const unsigned char *p = bytes + offset;
        uint32_t k;
        if (byte_len < offset + KAT_I8_SLOT_HEADER) {
            return KAT_BLOB_ERR_SLOT;
        }
        slots[index].dtype = rd_u32(p + 16);
        slots[index].rank = rd_u32(p + 20);
        for (k = 0U; k < 4U; ++k) {
            slots[index].dims[k] = rd_u32(p + 24U + (4U * k));
        }
        slots[index].byte_len = rd_u32(p + 40);
        slots[index].quant_kind = p[44];
        slots[index].scale = rd_f32(p + 48);
        slots[index].zero_point = rd_i32(p + 52);
        slots[index].scale_count = rd_u32(p + 56);
        offset += KAT_I8_SLOT_HEADER;
        slots[index].payload = offset;
        if (byte_len < offset + slots[index].byte_len) {
            return KAT_BLOB_ERR_SLOT;
        }
        offset += slots[index].byte_len;
        slots[index].scales_at = offset;
        if (byte_len < offset + (slots[index].scale_count * 4U)) {
            return KAT_BLOB_ERR_SLOT;
        }
        offset += slots[index].scale_count * 4U;
        if (slots[index].dtype != KAT_I8_DTYPE_F32 && slots[index].dtype != KAT_I8_DTYPE_I8 &&
            slots[index].dtype != KAT_I8_DTYPE_I32) {
            return KAT_BLOB_ERR_SLOT;
        }
    }
    if (offset != byte_len) {
        return KAT_BLOB_ERR_SLOT;
    }

    for (index = 0U; index < node_count; ++index) {
        if (node_op[index] == 0U) { /* input */
            input_slot = node_out_slot[index];
        } else if (node_op[index] == 1U) { /* conv2d */
            if (conv != NO_SLOT_I8) {
                return KAT_BLOB_ERR_OP;
            }
            conv = index;
        } else {
            return KAT_BLOB_ERR_OP;
        }
    }
    if (input_slot == NO_SLOT_I8 || conv == NO_SLOT_I8) {
        return KAT_BLOB_ERR_OP;
    }
    w_slot = &slots[node_w_slot[conv]];
    b_slot = &slots[node_b_slot[conv]];
    out_slot = &slots[node_out_slot[conv]];

    for (axis = 0U; axis < slots[input_slot].rank; ++axis) {
        in_count *= slots[input_slot].dims[axis];
    }
    for (axis = 0U; axis < out_slot->rank; ++axis) {
        out_count *= out_slot->dims[axis];
    }
    if (in_count != input_len || out_count != out_len || arena_len < out_count) {
        return KAT_BLOB_ERR_ARENA;
    }
    if (w_slot->scale_count > 32U || b_slot->dims[0] > 32U) {
        return KAT_BLOB_ERR_SLOT;
    }
    if (infr_shape_init(&in_shape, (uint8_t)slots[input_slot].rank, slots[input_slot].dims) != INFR_OK ||
        infr_shape_init(&w_shape, (uint8_t)w_slot->rank, w_slot->dims) != INFR_OK ||
        infr_shape_init(&out_shape, (uint8_t)out_slot->rank, out_slot->dims) != INFR_OK) {
        return KAT_BLOB_ERR_SHAPE;
    }
    if (infr_tensor_init(&in_tensor, (void *)(uintptr_t)(const void *)input, INFR_DTYPE_I8, &in_shape) != INFR_OK ||
        infr_tensor_init(&w_tensor, (void *)(uintptr_t)(const void *)(bytes + w_slot->payload), INFR_DTYPE_I8,
                         &w_shape) != INFR_OK ||
        infr_tensor_init(&out_tensor, arena, INFR_DTYPE_I8, &out_shape) != INFR_OK) {
        return KAT_BLOB_ERR_SLOT;
    }
    in_tensor.quant.kind = INFR_QUANT_PER_TENSOR;
    in_tensor.quant.scale = slots[input_slot].scale;
    in_tensor.quant.zero_point = slots[input_slot].zero_point;
    w_tensor.quant.kind = w_slot->quant_kind == 2U ? INFR_QUANT_PER_CHANNEL : INFR_QUANT_PER_TENSOR;
    w_tensor.quant.scale = w_slot->scale;
    w_tensor.quant.zero_point = w_slot->zero_point;
    for (index = 0U; index < w_slot->scale_count; ++index) {
        scales_local[index] = rd_f32(bytes + w_slot->scales_at + (4U * index));
    }
    if (w_slot->scale_count != 0U) {
        w_tensor.quant.channel_scales = scales_local;
    }
    out_tensor.quant.kind = INFR_QUANT_PER_TENSOR;
    out_tensor.quant.scale = out_slot->scale;
    out_tensor.quant.zero_point = out_slot->zero_point;

    for (index = 0U; index < b_slot->dims[0]; ++index) {
        bias_local[index] = rd_i32(bytes + b_slot->payload + (4U * index));
    }

    params.kernel_h = node_p[conv][0];
    params.kernel_w = node_p[conv][0];
    params.stride_h = node_p[conv][1];
    params.stride_w = node_p[conv][1];
    params.pad_h = node_p[conv][2];
    params.pad_w = node_p[conv][2];
    params.in_channels = slots[input_slot].dims[3];
    params.out_channels = out_slot->dims[3];
    params.groups = node_p[conv][3] != 0U ? node_p[conv][3] : 1U;

    status = infr_conv2d_i8(&in_tensor, &w_tensor, bias_local, &params, &out_tensor, workspace);
    if (status != INFR_OK) {
        return KAT_BLOB_ERR_OP;
    }
    memcpy(out, arena, out_count);
    return KAT_BLOB_OK;
}
