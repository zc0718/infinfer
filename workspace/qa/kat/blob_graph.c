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

#define KAT_BLOB_MAX_NODES 32U
#define KAT_BLOB_MAX_SLOTS 32U

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
    data = slot->byte_len != 0U ? (void *)(uintptr_t)(const void *)(bytes + slot->offset)
                                : (void *)(arena + slot->arena);
    return infr_tensor_init(tensor, data, INFR_DTYPE_F32, &shape) == INFR_OK;
}

static const float *constant_of(const unsigned char *bytes, const kat_blob_slot *slot) {
    return (const float *)(const void *)(bytes + slot->offset);
}

int kat_blob_run(const unsigned char *bytes, uint32_t byte_len, const float *input, uint32_t input_len,
                 float *arena, uint32_t arena_len, infr_workspace *workspace, kat_blob_view *view) {
    static const unsigned char magic[8] = {'I', 'N', 'F', 'R', 'K', 'A', 'T', '\0'};
    kat_blob_slot slots[KAT_BLOB_MAX_SLOTS];
    kat_blob_node nodes[KAT_BLOB_MAX_NODES];
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
            slots[i].arena = cursor;
            cursor += slot_floats(&slots[i]);
        }
    }
    if (offset != byte_len) {
        return KAT_BLOB_ERR_SLOT;
    }
    if (cursor > arena_len) {
        return KAT_BLOB_ERR_ARENA;
    }

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
            if (!bind_slot(bytes, &slots[node->in_slot], arena, &in_t) ||
                !bind_slot(bytes, w_slot, arena, &w_t)) {
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
            if (slot_floats(a_slot) != count || slot_floats(b_slot) != count) {
                return KAT_BLOB_ERR_SHAPE;
            }
            if (infr_kernel_register_scalar() != INFR_OK) {
                return INFR_ERR_NOT_IMPLEMENTED;
            }
            binary = infr_kernel_find(kernel, INFR_DTYPE_F32, INFR_QUANT_NONE, INFR_TIER_SCALAR);
            if (binary == 0) {
                return INFR_ERR_NOT_IMPLEMENTED;
            }
            operands[0] = arena + a_slot->arena;
            operands[1] = arena + b_slot->arena;
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
