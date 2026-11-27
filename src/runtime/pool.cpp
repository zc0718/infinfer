// Conan::ImportStart
#include <cmath>
#include <limits>

#include <infinfer/runtime/graph.h>

#include "shape_internal.hpp"
// Conan::ImportEnd

/*!
 * @ingroup group_runtime
 * @file
 * @brief [en] Max pooling as an L2 composition: gather a window, then the registered `max`.
 * @brief [zh] 最大池化作为 L2 组合：gather 一个窗口，再交给已注册的 `max`。
 * @since 0.1.0
 */

/* Two things here are easy to get wrong and are therefore written down where they
 * are implemented rather than only in the header:
 *
 *  1. padding contributes -infinity. A zero pad would beat every negative value in
 *     the window, so a row of zeros would leak into the result and max pooling would
 *     quietly become "max(x, 0)".
 *  2. padding must be smaller than the window, so no output position is entirely
 *     padding -- otherwise the honest answer is -infinity, which is almost never
 *     what the graph meant.
 *
 * 有两件事最容易写错，所以写在实现旁边而不是只在头里：
 *  1. 填充贡献负无穷。填 0 会赢过窗口里所有负值，于是结果里混进一行 0，最大池化就
 *     悄悄变成了 max(x, 0)。
 *  2. 填充必须小于窗口，保证没有哪个输出位置完全落在填充里——否则正确答案是负无穷，
 *     而那几乎从来不是图想要的。 */

namespace {

struct pool_plan {
    uint32_t in_h;
    uint32_t in_w;
    uint32_t channels;
    uint32_t out_h;
    uint32_t out_w;
    uint32_t kh;
    uint32_t kw;
    uint32_t stride_h;
    uint32_t stride_w;
    uint32_t pad_h;
    uint32_t pad_w;
};

infr_status build_plan(const infr_tensor* input, const infr_pool_params* params, const infr_tensor* output,
                       pool_plan* plan) {
    if (input->shape.rank != 4U || output->shape.rank != 4U) {
        return INFR_ERR_UNSUPPORTED;
    }
    if (!infr_detail::is_f32(input) || !infr_detail::is_f32(output)) {
        return INFR_ERR_UNSUPPORTED;
    }
    if (input->shape.dims[0] != 1U || output->shape.dims[0] != 1U) {
        return INFR_ERR_UNSUPPORTED;
    }

    plan->in_h = input->shape.dims[1];
    plan->in_w = input->shape.dims[2];
    plan->channels = input->shape.dims[3];
    plan->kh = params->kernel_h;
    plan->kw = params->kernel_w;
    plan->stride_h = params->stride_h;
    plan->stride_w = params->stride_w;
    plan->pad_h = params->pad_h;
    plan->pad_w = params->pad_w;

    if (params->channels != plan->channels || output->shape.dims[3] != plan->channels) {
        return INFR_ERR_INVALID_ARGUMENT;
    }
    /* No output position may be entirely padding. */
    if (plan->kh == 0U || plan->kw == 0U || plan->pad_h >= plan->kh || plan->pad_w >= plan->kw) {
        return INFR_ERR_INVALID_ARGUMENT;
    }

    uint32_t out_h = 0U;
    uint32_t out_w = 0U;
    infr_status status = infr_detail::out_extent(plan->in_h, plan->kh, plan->pad_h, plan->stride_h, &out_h);
    if (status != INFR_OK) {
        return status;
    }
    status = infr_detail::out_extent(plan->in_w, plan->kw, plan->pad_w, plan->stride_w, &out_w);
    if (status != INFR_OK) {
        return status;
    }
    if (output->shape.dims[1] != out_h || output->shape.dims[2] != out_w) {
        return INFR_ERR_INVALID_ARGUMENT;
    }
    plan->out_h = out_h;
    plan->out_w = out_w;
    return INFR_OK;
}

} // namespace

/**
 * @brief [en] Output shape of a pooling operator.
 * @brief [zh] 池化算子的输出形状。
 * @since 0.1.0
 */
infr_status infr_max_pool2d_output_shape(const infr_pool_params* params, const infr_shape* input_shape,
                                         infr_shape* out_shape) {
    if (params == nullptr || input_shape == nullptr || out_shape == nullptr) {
        return INFR_ERR_INVALID_ARGUMENT;
    }
    if (input_shape->rank != 4U) {
        return INFR_ERR_INVALID_ARGUMENT;
    }
    if (params->kernel_h == 0U || params->kernel_w == 0U || params->pad_h >= params->kernel_h ||
        params->pad_w >= params->kernel_w) {
        return INFR_ERR_INVALID_ARGUMENT;
    }
    uint32_t out_h = 0U;
    uint32_t out_w = 0U;
    infr_status status =
        infr_detail::out_extent(input_shape->dims[1], params->kernel_h, params->pad_h, params->stride_h, &out_h);
    if (status != INFR_OK) {
        return status;
    }
    status = infr_detail::out_extent(input_shape->dims[2], params->kernel_w, params->pad_w, params->stride_w, &out_w);
    if (status != INFR_OK) {
        return status;
    }
    const uint32_t dims[4] = {1U, out_h, out_w, params->channels};
    return infr_shape_init(out_shape, 4U, dims);
}

/**
 * @brief [en] fp32 max pooling over an NHWC tensor.
 * @brief [zh] 在 NHWC 张量上做 fp32 最大池化。
 * @since 0.1.0
 */
infr_status infr_max_pool2d_f32(const infr_tensor* input, const infr_pool_params* params, infr_tensor* output,
                                infr_workspace* workspace) {
    if (input == nullptr || params == nullptr || output == nullptr || workspace == nullptr || input->data == nullptr ||
        output->data == nullptr) {
        return INFR_ERR_INVALID_ARGUMENT;
    }

    pool_plan plan{};
    infr_status status = build_plan(input, params, output, &plan);
    if (status != INFR_OK) {
        return status;
    }

    status = infr_kernel_init();
    if (status != INFR_OK) {
        return status;
    }
    const infr_kernel_desc* reduce = infr_kernel_find_best("max", INFR_DTYPE_F32, INFR_QUANT_NONE);
    if (reduce == nullptr) {
        return INFR_ERR_NOT_IMPLEMENTED;
    }

    const size_t mark = infr_workspace_mark(workspace);
    const size_t window_len = static_cast<size_t>(plan.kh) * plan.kw;
    void* window_raw = nullptr;
    status = infr_workspace_alloc(workspace, window_len * sizeof(float), alignof(float), &window_raw);
    if (status != INFR_OK) {
        (void)infr_workspace_release(workspace, mark);
        return status;
    }

    auto* window = static_cast<float*>(window_raw);
    const auto* in = static_cast<const float*>(input->data);
    auto* out = static_cast<float*>(output->data);
    constexpr float kPad = -std::numeric_limits<float>::infinity();

    for (uint32_t oy = 0U; oy < plan.out_h; ++oy) {
        const int32_t base_y = static_cast<int32_t>(oy * plan.stride_h) - static_cast<int32_t>(plan.pad_h);
        for (uint32_t ox = 0U; ox < plan.out_w; ++ox) {
            const int32_t base_x = static_cast<int32_t>(ox * plan.stride_w) - static_cast<int32_t>(plan.pad_w);
            float* pixel = out + ((static_cast<size_t>(oy) * plan.out_w) + ox) * plan.channels;

            for (uint32_t channel = 0U; channel < plan.channels; ++channel) {
                for (uint32_t ky = 0U; ky < plan.kh; ++ky) {
                    const int32_t iy = base_y + static_cast<int32_t>(ky);
                    for (uint32_t kx = 0U; kx < plan.kw; ++kx) {
                        const int32_t ix = base_x + static_cast<int32_t>(kx);
                        float value = kPad;
                        if (iy >= 0 && ix >= 0 && iy < static_cast<int32_t>(plan.in_h) &&
                            ix < static_cast<int32_t>(plan.in_w)) {
                            const size_t pixel_index = (static_cast<size_t>(iy) * plan.in_w) + static_cast<size_t>(ix);
                            value = in[(pixel_index * plan.channels) + channel];
                        }
                        window[(static_cast<size_t>(ky) * plan.kw) + kx] = value;
                    }
                }
                const void* reduce_inputs[1] = {window};
                void* reduce_output[1] = {&pixel[channel]};
                status = reduce->fn(reduce_inputs, reduce_output, static_cast<uint32_t>(window_len), nullptr);
                if (status != INFR_OK) {
                    (void)infr_workspace_release(workspace, mark);
                    return status;
                }
            }
        }
    }

    return infr_workspace_release(workspace, mark);
}
