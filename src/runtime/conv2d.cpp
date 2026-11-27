// Conan::ImportStart
#include <infinfer/runtime/graph.h>

#include "shape_internal.hpp"
// Conan::ImportEnd

/*!
 * @ingroup group_runtime
 * @file
 * @brief [en] Convolution as an L2 composition: gather, dot product, bias add.
 * @brief [zh] 卷积作为 L2 组合：gather、点积、加偏置。
 * @since 0.1.0
 */

/* Why this file looks like three operators bolted together: it is three operators
 * bolted together, deliberately. The arithmetic is not written here at all -- it is
 * a call to the registered `fc` for the dot product and a call to the registered
 * `add` for the bias. What is written here is only the part that needs shape: which
 * input elements belong to which output pixel, and how much scratch that takes.
 *
 * 本文件看起来像三个算子拼在一起，因为它就是故意的三段拼装：算术一行都不在这里
 * ——点积是对已注册 `fc` 的调用，偏置是对已注册 `add` 的调用。这里写的只有真正需要
 * shape 的那部分：哪些输入元素属于哪个输出像素，以及为此要借多少临时空间。 */

namespace {

/* Everything derived from the tensors and the params, computed once and then only
 * read. Keeping it in one struct is what makes the validation impossible to skip:
 * the loop cannot run without a plan, and a plan cannot exist unless every check
 * passed. */
struct conv_plan {
    uint32_t in_h;
    uint32_t in_w;
    uint32_t in_c;
    uint32_t out_h;
    uint32_t out_w;
    uint32_t out_c;
    uint32_t kh;
    uint32_t kw;
    uint32_t stride_h;
    uint32_t stride_w;
    uint32_t pad_h;
    uint32_t pad_w;
    uint32_t groups;
    uint32_t in_per_group;
    uint32_t out_per_group;
    uint32_t window_len; /* kh * kw * in_per_group: the dot-product length */
};

infr_status build_plan(const infr_tensor* input, const infr_tensor* filter, const infr_conv2d_params* params,
                       const infr_tensor* output, conv_plan* plan) {
    if (input->shape.rank != 4U || filter->shape.rank != 4U || output->shape.rank != 4U) {
        return INFR_ERR_UNSUPPORTED; /* batch-first 3-D and NCHW are out of the subset */
    }
    if (input->shape.dims[0] != 1U || output->shape.dims[0] != 1U) {
        return INFR_ERR_UNSUPPORTED; /* batch 1 only */
    }

    plan->in_h = input->shape.dims[1];
    plan->in_w = input->shape.dims[2];
    plan->in_c = input->shape.dims[3];
    plan->out_c = output->shape.dims[3];
    plan->kh = params->kernel_h;
    plan->kw = params->kernel_w;
    plan->stride_h = params->stride_h;
    plan->stride_w = params->stride_w;
    plan->pad_h = params->pad_h;
    plan->pad_w = params->pad_w;
    plan->groups = params->groups;

    if (params->in_channels != plan->in_c || params->out_channels != plan->out_c) {
        return INFR_ERR_INVALID_ARGUMENT;
    }
    if (plan->groups == 0U || (plan->in_c % plan->groups) != 0U || (plan->out_c % plan->groups) != 0U) {
        return INFR_ERR_INVALID_ARGUMENT;
    }
    /* The subset the matrix declares: dense (1) and depthwise (in_channels). A
     * general grouped convolution is expressible but not promised, so it is refused
     * rather than half-supported. */
    if (plan->groups != 1U && plan->groups != plan->in_c) {
        return INFR_ERR_UNSUPPORTED;
    }
    plan->in_per_group = plan->in_c / plan->groups;
    plan->out_per_group = plan->out_c / plan->groups;

    /* The filter must agree with the params on every axis it can disagree about. */
    if (filter->shape.dims[0] != plan->out_c || filter->shape.dims[1] != plan->kh ||
        filter->shape.dims[2] != plan->kw || filter->shape.dims[3] != plan->in_per_group) {
        return INFR_ERR_INVALID_ARGUMENT;
    }

    uint32_t derived_h = 0U;
    uint32_t derived_w = 0U;
    infr_status status = infr_detail::out_extent(plan->in_h, plan->kh, plan->pad_h, plan->stride_h, &derived_h);
    if (status != INFR_OK) {
        return status;
    }
    status = infr_detail::out_extent(plan->in_w, plan->kw, plan->pad_w, plan->stride_w, &derived_w);
    if (status != INFR_OK) {
        return status;
    }
    /* The caller's output descriptor has to match what the geometry says. A mismatch
     * is a graph bug, and writing anyway would silently shift the whole feature map. */
    if (output->shape.dims[1] != derived_h || output->shape.dims[2] != derived_w) {
        return INFR_ERR_INVALID_ARGUMENT;
    }
    plan->out_h = derived_h;
    plan->out_w = derived_w;
    plan->window_len = plan->kh * plan->kw * plan->in_per_group;
    return INFR_OK;
}

/* Read one input element, or 0.0 when the coordinate falls in the padding: the pad
 * is zeros, so "outside" and "zero" are the same contribution and the dot product
 * never needs to know which it saw. */
float gather_one(const float* in, const conv_plan& plan, uint32_t channel, int32_t iy, int32_t ix) {
    if (iy < 0 || ix < 0 || iy >= static_cast<int32_t>(plan.in_h) || ix >= static_cast<int32_t>(plan.in_w)) {
        return 0.0F;
    }
    const uint32_t row = static_cast<uint32_t>(iy) * plan.in_w;
    const uint32_t pixel = row + static_cast<uint32_t>(ix);
    return in[(pixel * plan.in_c) + channel];
}

/* The int8 equivalent, and the reason padding is worth a comment here: in a quantised
 * tensor the value that means "zero" is the zero point, not the byte 0. The operator
 * only admits a zero point of 0 (see the header), so the two coincide and this returns
 * 0; if asymmetric inputs are ever admitted, this is the line that has to change.
 *
 * int8 的对应物，也是填充在这里值得写一条注释的原因：在量化张量里表示“零”的是零点，
 * 而不是字节 0。本算子只接受零点为 0（见头文件），两者因而重合、这里返回 0；将来若
 * 接纳非对称输入，就是这一行要改。 */
int8_t gather_one_i8(const int8_t* in, const conv_plan& plan, uint32_t channel, int32_t iy, int32_t ix) {
    if (iy < 0 || ix < 0 || iy >= static_cast<int32_t>(plan.in_h) || ix >= static_cast<int32_t>(plan.in_w)) {
        return INT8_C(0);
    }
    const uint32_t row = static_cast<uint32_t>(iy) * plan.in_w;
    const uint32_t pixel = row + static_cast<uint32_t>(ix);
    return in[(static_cast<size_t>(pixel) * plan.in_c) + channel];
}

/* Per-output-channel requantization multiplier, `(input.scale * weight.scale) / output.scale`.
 * A per-tensor filter is the degenerate case and is read through the same helper, so the
 * inner loop never learns which scheme was declared. */
bool channel_multiplier(const infr_tensor& filter, const infr_tensor& input, const infr_tensor& output,
                        uint32_t channel, float* multiplier) {
    const float weight_scale =
        filter.quant.kind == INFR_QUANT_PER_CHANNEL ? filter.quant.channel_scales[channel] : filter.quant.scale;
    if (!(weight_scale > 0.0F)) {
        return false;
    }
    *multiplier = (input.quant.scale * weight_scale) / output.quant.scale;
    return *multiplier > 0.0F;
}

} // namespace

/**
 * @brief [en] Output shape of a convolution, so callers do not re-derive it.
 * @brief [zh] 卷积的输出形状，免得每个调用方自己再推一遍。
 * @since 0.1.0
 */
infr_status infr_conv2d_output_shape(const infr_conv2d_params* params, const infr_shape* input_shape,
                                     infr_shape* out_shape) {
    if (params == nullptr || input_shape == nullptr || out_shape == nullptr) {
        return INFR_ERR_INVALID_ARGUMENT;
    }
    if (input_shape->rank != 4U) {
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
    const uint32_t dims[4] = {1U, out_h, out_w, params->out_channels};
    return infr_shape_init(out_shape, 4U, dims);
}

/**
 * @brief [en] fp32 convolution (dense or depthwise) over an NHWC tensor.
 * @brief [zh] 在 NHWC 张量上做 fp32 卷积（普通或逐通道）。
 * @since 0.1.0
 */
infr_status infr_conv2d_f32(const infr_tensor* input, const infr_tensor* filter, const float* bias,
                            const infr_conv2d_params* params, infr_tensor* output, infr_workspace* workspace) {
    if (input == nullptr || filter == nullptr || params == nullptr || output == nullptr || workspace == nullptr ||
        input->data == nullptr || filter->data == nullptr || output->data == nullptr) {
        return INFR_ERR_INVALID_ARGUMENT;
    }

    /* The dtype gate is checked before the geometry on purpose: the documented contract says an
     * unsupported dtype answers UNSUPPORTED, and a caller whose tensor is the wrong type should not
     * first be told its filter shape is wrong. That order is a promise the f32 tests already pin. */
    if (!infr_detail::is_f32(input) || !infr_detail::is_f32(filter) || !infr_detail::is_f32(output)) {
        return INFR_ERR_UNSUPPORTED;
    }

    conv_plan plan{};
    infr_status status = build_plan(input, filter, params, output, &plan);
    if (status != INFR_OK) {
        return status;
    }

    /* The composition is the point: both halves come from the registry, so the tier
     * that wins for `fc`/`add` is the tier this operator runs at. */
    status = infr_kernel_register_scalar();
    if (status != INFR_OK) {
        return status;
    }
    const infr_kernel_desc* dot = infr_kernel_find("fc", INFR_DTYPE_F32, INFR_QUANT_NONE, INFR_TIER_SCALAR);
    if (dot == nullptr) {
        return INFR_ERR_NOT_IMPLEMENTED;
    }
    const infr_kernel_desc* sum_vec = nullptr;
    if (bias != nullptr) {
        sum_vec = infr_kernel_find("add", INFR_DTYPE_F32, INFR_QUANT_NONE, INFR_TIER_SCALAR);
        if (sum_vec == nullptr) {
            return INFR_ERR_NOT_IMPLEMENTED;
        }
    }

    /* Borrow scratch and hand it back on every path: `EVAL` must not grow with the
     * number of nodes executed. */
    const size_t mark = infr_workspace_mark(workspace);
    void* window_raw = nullptr;
    void* acc_raw = nullptr;
    status = infr_workspace_alloc(workspace, static_cast<size_t>(plan.window_len) * sizeof(float), alignof(float),
                                  &window_raw);
    if (status == INFR_OK) {
        status =
            infr_workspace_alloc(workspace, static_cast<size_t>(plan.out_c) * sizeof(float), alignof(float), &acc_raw);
    }
    if (status != INFR_OK) {
        (void)infr_workspace_release(workspace, mark);
        return status;
    }

    auto* window = static_cast<float*>(window_raw);
    auto* accum = static_cast<float*>(acc_raw);
    const auto* in = static_cast<const float*>(input->data);
    const auto* weights = static_cast<const float*>(filter->data);
    auto* out = static_cast<float*>(output->data);

    for (uint32_t oy = 0U; oy < plan.out_h; ++oy) {
        const int32_t base_y = static_cast<int32_t>(oy * plan.stride_h) - static_cast<int32_t>(plan.pad_h);
        for (uint32_t ox = 0U; ox < plan.out_w; ++ox) {
            const int32_t base_x = static_cast<int32_t>(ox * plan.stride_w) - static_cast<int32_t>(plan.pad_w);

            for (uint32_t group = 0U; group < plan.groups; ++group) {
                /* Gather this group's window once, then feed every output channel of
                 * the group from it: the gather is the only shape-dependent work. */
                const uint32_t channel_base = group * plan.in_per_group;
                for (uint32_t ky = 0U; ky < plan.kh; ++ky) {
                    for (uint32_t kx = 0U; kx < plan.kw; ++kx) {
                        const int32_t iy = base_y + static_cast<int32_t>(ky);
                        const int32_t ix = base_x + static_cast<int32_t>(kx);
                        const uint32_t slot = ((ky * plan.kw) + kx) * plan.in_per_group;
                        for (uint32_t ci = 0U; ci < plan.in_per_group; ++ci) {
                            window[slot + ci] = gather_one(in, plan, channel_base + ci, iy, ix);
                        }
                    }
                }

                for (uint32_t k = 0U; k < plan.out_per_group; ++k) {
                    const uint32_t oc = (group * plan.out_per_group) + k;
                    const float* row = weights + (static_cast<size_t>(oc) * plan.window_len);
                    const void* dot_inputs[2] = {window, row};
                    /* Index the accumulator by the GLOBAL output channel, not by the
                     * position inside the group: with `out_per_group == 1` (depthwise)
                     * a per-group index would make every group overwrite slot 0. */
                    void* dot_output[1] = {&accum[oc]};
                    status = dot->fn(dot_inputs, dot_output, plan.window_len, nullptr);
                    if (status != INFR_OK) {
                        (void)infr_workspace_release(workspace, mark);
                        return status;
                    }
                }
            }

            float* pixel = out + ((static_cast<size_t>(oy) * plan.out_w) + ox) * plan.out_c;
            if (bias != nullptr) {
                const void* add_inputs[2] = {accum, bias};
                void* add_output[1] = {pixel};
                status = sum_vec->fn(add_inputs, add_output, plan.out_c, nullptr);
                if (status != INFR_OK) {
                    (void)infr_workspace_release(workspace, mark);
                    return status;
                }
            } else {
                for (uint32_t oc = 0U; oc < plan.out_c; ++oc) {
                    pixel[oc] = accum[oc];
                }
            }
        }
    }

    return infr_workspace_release(workspace, mark);
}

/* The quantised counterpart of the operator above. The structure is the same on purpose --
 * gather, dot product per output channel, offset -- and only two things differ: the gather
 * reads bytes and the last step is a requantization rather than a float add.
 *
 * 上方算子的量化版本。结构刻意保持一致——gather、逐输出通道点积、加偏移——只有两处不同：
 * gather 读的是字节，最后一步是再量化而不是浮点加法。 */

/**
 * @brief [en] int8 convolution (dense or depthwise) over an NHWC tensor, with an int32 bias and one
 *            requantization.
 * @brief [zh] 在 NHWC 张量上做 int8 卷积（普通或逐通道），偏置为 int32，全程只再量化一次。
 * @since 0.2.0
 */
infr_status infr_conv2d_i8(const infr_tensor* input, const infr_tensor* filter, const int32_t* bias,
                           const infr_conv2d_params* params, infr_tensor* output, infr_workspace* workspace) {
    if (input == nullptr || filter == nullptr || params == nullptr || output == nullptr || workspace == nullptr ||
        input->data == nullptr || filter->data == nullptr || output->data == nullptr) {
        return INFR_ERR_INVALID_ARGUMENT;
    }

    /* The subset this operator promises, checked before the geometry for the same reason the f32
     * operator checks its dtype first: an unsupported dtype answers UNSUPPORTED. Asymmetric input
     * is refused rather than approximated -- the padding value would be wrong, and a wrong padding
     * value is invisible in the output. */
    const bool input_ok = input->dtype == INFR_DTYPE_I8 && input->quant.kind == INFR_QUANT_PER_TENSOR &&
                          input->quant.scale > 0.0F && input->quant.zero_point == 0;
    const bool filter_ok = filter->dtype == INFR_DTYPE_I8 &&
                           ((filter->quant.kind == INFR_QUANT_PER_TENSOR && filter->quant.scale > 0.0F) ||
                            (filter->quant.kind == INFR_QUANT_PER_CHANNEL && filter->quant.channel_scales != nullptr));
    const bool output_ok =
        output->dtype == INFR_DTYPE_I8 && output->quant.kind == INFR_QUANT_PER_TENSOR && output->quant.scale > 0.0F;
    if (!input_ok || !filter_ok || !output_ok) {
        return INFR_ERR_UNSUPPORTED;
    }

    conv_plan plan{};
    infr_status status = build_plan(input, filter, params, output, &plan);
    if (status != INFR_OK) {
        return status;
    }

    status = infr_kernel_register_scalar();
    if (status != INFR_OK) {
        return status;
    }
    const infr_kernel_desc* dot = infr_kernel_find("dot", INFR_DTYPE_I8, INFR_QUANT_PER_TENSOR, INFR_TIER_SCALAR);
    const infr_kernel_desc* requant =
        infr_kernel_find("requant", INFR_DTYPE_I32, INFR_QUANT_PER_TENSOR, INFR_TIER_SCALAR);
    if (dot == nullptr || requant == nullptr) {
        return INFR_ERR_NOT_IMPLEMENTED;
    }

    const size_t mark = infr_workspace_mark(workspace);
    void* window_raw = nullptr;
    status = infr_workspace_alloc(workspace, static_cast<size_t>(plan.window_len) * sizeof(int8_t), alignof(int8_t),
                                  &window_raw);
    if (status != INFR_OK) {
        return status;
    }

    auto* window = static_cast<int8_t*>(window_raw);
    const auto* in = static_cast<const int8_t*>(input->data);
    const auto* weights = static_cast<const int8_t*>(filter->data);
    auto* out = static_cast<int8_t*>(output->data);

    for (uint32_t oy = 0U; oy < plan.out_h && status == INFR_OK; ++oy) {
        const int32_t base_y = static_cast<int32_t>(oy * plan.stride_h) - static_cast<int32_t>(plan.pad_h);
        for (uint32_t ox = 0U; ox < plan.out_w && status == INFR_OK; ++ox) {
            const int32_t base_x = static_cast<int32_t>(ox * plan.stride_w) - static_cast<int32_t>(plan.pad_w);
            int8_t* pixel = out + ((static_cast<size_t>(oy) * plan.out_w) + ox) * plan.out_c;

            for (uint32_t group = 0U; group < plan.groups; ++group) {
                const uint32_t channel_base = group * plan.in_per_group;
                for (uint32_t ky = 0U; ky < plan.kh; ++ky) {
                    for (uint32_t kx = 0U; kx < plan.kw; ++kx) {
                        const int32_t iy = base_y + static_cast<int32_t>(ky);
                        const int32_t ix = base_x + static_cast<int32_t>(kx);
                        const uint32_t slot = ((ky * plan.kw) + kx) * plan.in_per_group;
                        for (uint32_t ci = 0U; ci < plan.in_per_group; ++ci) {
                            window[slot + ci] = gather_one_i8(in, plan, channel_base + ci, iy, ix);
                        }
                    }
                }

                for (uint32_t k = 0U; k < plan.out_per_group; ++k) {
                    const uint32_t oc = (group * plan.out_per_group) + k;
                    const int8_t* row = weights + (static_cast<size_t>(oc) * plan.window_len);
                    int32_t accumulator = 0;
                    const void* dot_inputs[2] = {window, row};
                    void* dot_output[1] = {&accumulator};
                    status = dot->fn(dot_inputs, dot_output, plan.window_len, nullptr);
                    if (status != INFR_OK) {
                        break;
                    }

                    /* One channel at a time, because the multiplier is per output channel: the
                     * graph layer resolves the channel's parameters and calls the kernel once per
                     * slice, which is the rule quant.c states. */
                    infr_quant quant{};
                    quant.kind = INFR_QUANT_PER_TENSOR;
                    quant.zero_point = output->quant.zero_point;
                    if (!channel_multiplier(*filter, *input, *output, oc, &quant.requant_multiplier)) {
                        status = INFR_ERR_INVALID_ARGUMENT;
                        break;
                    }
                    const int32_t channel_bias = bias != nullptr ? bias[oc] : 0;
                    const int32_t with_bias = accumulator + channel_bias;
                    const void* requant_inputs[2] = {&with_bias, nullptr};
                    void* requant_outputs[1] = {&pixel[oc]};
                    status = requant->fn(requant_inputs, requant_outputs, 1U, &quant);
                    if (status != INFR_OK) {
                        break;
                    }
                }
            }
        }
    }

    const infr_status released = infr_workspace_release(workspace, mark);
    return status == INFR_OK ? released : status;
}
