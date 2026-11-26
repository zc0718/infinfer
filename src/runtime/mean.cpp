// Conan::ImportStart
#include <infinfer/runtime/graph.h>

#include "shape_internal.hpp"
// Conan::ImportEnd

/*!
 * @ingroup group_runtime
 * @file
 * @brief [en] Global average pooling as an L2 composition: `sum`, then `scale`.
 * @brief [zh] 全局平均池化作为 L2 组合：先 `sum`，再 `scale`。
 * @since 0.1.0
 */

/* The plane is gathered into scratch first because the plane of one channel is not
 * contiguous in NHWC: the channel stride is `channels`. Gathering is the only
 * shape-dependent work; the reduction and the multiply both come from the registry,
 * so this operator contains no arithmetic of its own.
 *
 * 先把一个通道的平面 gather 进临时缓冲，因为 NHWC 下单个通道的平面并不连续（通道步长
 * 是 `channels`）。gather 是唯一依赖 shape 的活；归约与乘法都来自注册表，所以本算子
 * 自己没有任何算术。 */

/**
 * @brief [en] Global average pooling: mean over H and W, per channel.
 * @brief [zh] 全局平均池化：对 H、W 求均值，逐通道。
 * @since 0.1.0
 */
infr_status infr_mean_hw_f32(const infr_tensor* input, infr_tensor* output, infr_workspace* workspace) {
    if (input == nullptr || output == nullptr || workspace == nullptr || input->data == nullptr ||
        output->data == nullptr) {
        return INFR_ERR_INVALID_ARGUMENT;
    }
    if (input->shape.rank != 4U || output->shape.rank != 4U) {
        return INFR_ERR_UNSUPPORTED;
    }
    if (!infr_detail::is_f32(input) || !infr_detail::is_f32(output)) {
        return INFR_ERR_UNSUPPORTED;
    }
    if (input->shape.dims[0] != 1U || output->shape.dims[0] != 1U) {
        return INFR_ERR_UNSUPPORTED;
    }
    const uint32_t in_h = input->shape.dims[1];
    const uint32_t in_w = input->shape.dims[2];
    const uint32_t channels = input->shape.dims[3];
    if (output->shape.dims[1] != 1U || output->shape.dims[2] != 1U || output->shape.dims[3] != channels) {
        return INFR_ERR_INVALID_ARGUMENT;
    }

    infr_status status = infr_kernel_register_scalar();
    if (status != INFR_OK) {
        return status;
    }
    const infr_kernel_desc* reduce = infr_kernel_find("sum", INFR_DTYPE_F32, INFR_QUANT_NONE, INFR_TIER_SCALAR);
    const infr_kernel_desc* scale = infr_kernel_find("scale", INFR_DTYPE_F32, INFR_QUANT_NONE, INFR_TIER_SCALAR);
    if (reduce == nullptr || scale == nullptr) {
        return INFR_ERR_NOT_IMPLEMENTED;
    }

    const size_t mark = infr_workspace_mark(workspace);
    const uint32_t plane_len = in_h * in_w;
    void* plane_raw = nullptr;
    status =
        infr_workspace_alloc(workspace, static_cast<size_t>(plane_len) * sizeof(float), alignof(float), &plane_raw);
    if (status != INFR_OK) {
        (void)infr_workspace_release(workspace, mark);
        return status;
    }

    auto* plane = static_cast<float*>(plane_raw);
    const auto* in = static_cast<const float*>(input->data);
    auto* out = static_cast<float*>(output->data);
    /* One division per channel, not one per element: the reciprocal is computed once
     * and both kernels then see only a multiply. */
    const float factor = 1.0F / static_cast<float>(plane_len);

    for (uint32_t channel = 0U; channel < channels; ++channel) {
        for (uint32_t i = 0U; i < plane_len; ++i) {
            plane[i] = in[(static_cast<size_t>(i) * channels) + channel];
        }

        float total = 0.0F;
        const void* sum_inputs[1] = {plane};
        void* sum_output[1] = {&total};
        status = reduce->fn(sum_inputs, sum_output, plane_len, nullptr);
        if (status != INFR_OK) {
            (void)infr_workspace_release(workspace, mark);
            return status;
        }

        const void* scale_inputs[2] = {&total, &factor};
        void* scale_output[1] = {&out[channel]};
        status = scale->fn(scale_inputs, scale_output, 1U, nullptr);
        if (status != INFR_OK) {
            (void)infr_workspace_release(workspace, mark);
            return status;
        }
    }

    return infr_workspace_release(workspace, mark);
}
