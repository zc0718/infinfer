// Conan::ImportStart
#include <infinfer/runtime/tensor.h>
// Conan::ImportEnd

/*!
 * @ingroup group_runtime
 * @file
 * @brief [en] Tensor descriptor: derived strides, checked indexing, byte sizes.
 * @brief [zh] 张量描述符：步长推导、带检查的索引、字节数。
 * @since 0.1.0
 */

/* Nothing here allocates, and nothing here stores a stride: `elem_count` and the
 * offsets are both derived from `dims`, so there is exactly one description of the
 * layout in memory. The dtype sizes are spelled out rather than computed from a
 * table so that adding a dtype without a test is a compile-time omission instead of
 * a silent zero size.
 *
 * 这里不分配内存，也不保存步长：`elem_count` 与偏移都由 `dims` 推导，因此内存布局
 * 只有一份描述。各 dtype 的宽度是逐个写出来的，而不是查表算出来的——这样"加了
 * dtype 却没写测试"会变成一处显眼的遗漏，而不是一个静默的 0。 */

/**
 * @brief [en] Bytes per element for a dtype; 0 for an unknown dtype.
 * @brief [zh] 某 dtype 每个元素的字节数；未知 dtype 返回 0。
 * @since 0.1.0
 */
size_t infr_dtype_size(infr_dtype dtype) {
    switch (dtype) {
    case INFR_DTYPE_F32:
        return sizeof(float);
    case INFR_DTYPE_I8:
    case INFR_DTYPE_U8:
        return 1U;
    case INFR_DTYPE_I16:
        return 2U;
    case INFR_DTYPE_I32:
        return 4U;
    default:
        return 0U;
    }
}

/**
 * @brief [en] Fill a shape from a rank and its dims, deriving `elem_count`.
 * @brief [zh] 用阶数和各维长度填写 shape，并推导 `elem_count`。
 * @since 0.1.0
 */
infr_status infr_shape_init(infr_shape* shape, uint8_t rank, const uint32_t* dims) {
    if (shape == nullptr || dims == nullptr) {
        return INFR_ERR_INVALID_ARGUMENT;
    }
    if (rank == 0U || rank > INFR_TENSOR_MAX_RANK) {
        return INFR_ERR_INVALID_ARGUMENT;
    }

    /* Accumulate in 64 bits: a 4-D fp32 tensor at 2^16 per axis is 2^64 elements,
     * which wraps a 32-bit product to a small, plausible-looking count. */
    uint64_t count = 1U;
    for (uint8_t axis = 0U; axis < INFR_TENSOR_MAX_RANK; ++axis) {
        if (axis >= rank) {
            shape->dims[axis] = 0U;
            continue;
        }
        if (dims[axis] == 0U) {
            return INFR_ERR_INVALID_ARGUMENT;
        }
        count *= dims[axis];
        if (count > UINT32_MAX) {
            return INFR_ERR_INVALID_ARGUMENT;
        }
        shape->dims[axis] = dims[axis];
    }
    shape->rank = rank;
    shape->elem_count = static_cast<uint32_t>(count);
    return INFR_OK;
}

/**
 * @brief [en] Row-major element offset of a multi-dimensional index.
 * @brief [zh] 多维索引对应的行主序元素偏移。
 * @since 0.1.0
 */
infr_status infr_shape_offset(const infr_shape* shape, const uint32_t* index, uint32_t* out_offset) {
    if (shape == nullptr || index == nullptr || out_offset == nullptr) {
        return INFR_ERR_INVALID_ARGUMENT;
    }
    if (shape->rank == 0U || shape->rank > INFR_TENSOR_MAX_RANK) {
        return INFR_ERR_INVALID_ARGUMENT;
    }

    /* Walk from the innermost axis outwards, so each step multiplies by the extent
     * of the axis just visited: offset += index[axis] * stride, stride starts at 1
     * and becomes stride *= dims[axis] after every step. */
    uint32_t offset = 0U;
    uint32_t stride = 1U;
    for (uint8_t axis = shape->rank; axis-- > 0U;) {
        if (index[axis] >= shape->dims[axis]) {
            return INFR_ERR_INVALID_ARGUMENT;
        }
        offset += index[axis] * stride;
        stride *= shape->dims[axis];
    }
    *out_offset = offset;
    return INFR_OK;
}

/**
 * @brief [en] Bind a buffer, dtype and shape into a tensor view.
 * @brief [zh] 把缓冲区、dtype 与 shape 绑成一个张量视图。
 * @since 0.1.0
 */
infr_status infr_tensor_init(infr_tensor* tensor, void* data, infr_dtype dtype, const infr_shape* shape) {
    if (tensor == nullptr || data == nullptr || shape == nullptr) {
        return INFR_ERR_INVALID_ARGUMENT;
    }
    if (shape->rank == 0U || shape->rank > INFR_TENSOR_MAX_RANK || shape->elem_count == 0U) {
        return INFR_ERR_INVALID_ARGUMENT;
    }
    if (infr_dtype_size(dtype) == 0U) {
        return INFR_ERR_INVALID_ARGUMENT;
    }
    tensor->data = data;
    tensor->shape = *shape;
    tensor->dtype = dtype;
    tensor->quant.kind = INFR_QUANT_NONE;
    tensor->quant.scale = 1.0F;
    tensor->quant.zero_point = 0;
    /* Zero means "this tensor does not requantize": only dot-product kernels read it,
     * and they refuse a non-positive value instead of zeroing their output.
     *
     * 0 表示“这张张量不做再量化”：只有点积内核读它，且它们会拒绝非正值而不是把输出清零。 */
    tensor->quant.requant_multiplier = 0.0F;
    tensor->quant.channel_scales = nullptr;
    tensor->quant.channel_zero_points = nullptr;
    tensor->quant.channel_count = 0U;
    tensor->quant.axis = 0U;
    return INFR_OK;
}

/**
 * @brief [en] Byte size of the data a tensor view spans.
 * @brief [zh] 张量视图所跨数据的字节数。
 * @since 0.1.0
 */
size_t infr_tensor_bytes(const infr_tensor* tensor) {
    if (tensor == nullptr) {
        return 0U;
    }
    const size_t element_size = infr_dtype_size(tensor->dtype);
    if (element_size == 0U) {
        return 0U;
    }
    return static_cast<size_t>(tensor->shape.elem_count) * element_size;
}
