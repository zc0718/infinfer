// Conan::ImportStart
#include <cstddef>

#include <infinfer/runtime/graph.h>

#include "shape_internal.hpp"
// Conan::ImportEnd

/*!
 * @ingroup group_runtime
 * @file
 * @brief [en] Slice as an L2 composition: a strided view, moved with `copy` runs.
 * @brief [zh] 切片作为 L2 组合：一个带步长的视图，用若干段 `copy` 搬出去。
 * @since 0.2.0
 */

/* The one decision that shapes this operator: how far does a contiguous run reach?
 *
 * The innermost axis decides it. When that axis is walked with step 1, each output run of
 * `out_dims[rank-1]` floats is contiguous in the input, so the operator issues one `copy` per
 * position of the outer axes -- and the pattern the real models actually emit (slicing the
 * channel axis of an NHWC tensor, which leaves the innermost axis full and contiguous) is
 * exactly this case. When the innermost axis is strided, no contiguous run exists in output
 * order, so the values are gathered one at a time into the workspace and handed to a single
 * `copy`, the same fallback `transpose` uses.
 *
 * Clamping, not validating: a start past the end or an end past the axis are both normal in
 * exported graphs (an `INT64_MAX` end is how ONNX says "to the end"), so the numbers are
 * clamped and a zero-length result is returned as a legal empty tensor. Refusing those would
 * make the importer responsible for arithmetic this operator can do in three lines.
 *
 * 决定这个算子形状的只有一个问题：一段连续能有多长？
 *
 * 由末轴决定。末轴以步长 1 遍历时，每条 `out_dims[rank-1]` 长的输出段在输入里都是连续的，
 * 于是外层每个位置一次 `copy`——而真实模型实际生成的那种形态（切 NHWC 的通道轴，末轴完整
 * 且连续）正是这一档。末轴带步长时，输出顺序里不存在连续段，于是逐元素 gather 进工作区再
 * 交给一次 `copy`，与 `transpose` 同一个兜底。
 *
 * 是夹取而不是校验：起点超过末尾、终点超过轴长，在导出的图里都是正常现象（ONNX 用
 * `INT64_MAX` 表示"到末尾"），所以数字被夹取，长度 0 作为合法的空张量返回。拒绝这些会让
 * 导入器承担三行就能算完的算术。
 */

namespace {

/**
 * @brief [en] A slice resolved against a concrete input shape.
 * @brief [zh] 针对具体输入形状解析后的切片。
 */
struct slice_plan {
    /**
     * @brief [en] Number of axes in the input.
     * @brief [zh] 输入的轴数。
     */
    uint32_t rank;
    /**
     * @brief [en] Output extent of each axis.
     * @brief [zh] 每条轴的输出长度。
     */
    uint32_t out_dims[INFR_TENSOR_MAX_RANK];
    /**
     * @brief [en] Input offset contributed by one step along each output axis.
     * @brief [zh] 沿每条输出轴走一步所贡献的输入偏移。
     */
    ptrdiff_t in_delta[INFR_TENSOR_MAX_RANK];
    /**
     * @brief [en] Input offset of the first output element.
     * @brief [zh] 第一个输出元素的输入偏移。
     */
    ptrdiff_t base;
    /**
     * @brief [en] Total number of output elements.
     * @brief [zh] 输出元素总数。
     */
    size_t total;
};

/**
 * @brief [en] Resolve `params` against `in_shape` and derive the plan.
 * @brief [zh] 用 `in_shape` 解析 `params` 并推导 plan。
 * @since 0.2.0
 */
infr_status resolve_plan(const infr_shape& in_shape, const infr_slice_params* params, slice_plan* plan) {
    const uint32_t rank = in_shape.rank;
    if (rank == 0U || rank > INFR_TENSOR_MAX_RANK || params->axis_count == 0U ||
        params->axis_count > INFR_TENSOR_MAX_RANK) {
        return INFR_ERR_INVALID_ARGUMENT;
    }

    /* Per-axis window, defaulting to the whole axis. Two arrays rather than one struct so the
     * arithmetic below reads the way the ONNX operator is defined.
     *
     * 逐轴窗口，默认整条轴。用两个数组而不是结构体，是为了让下面的算术读起来与 ONNX 的定义
     * 一致。 */
    int64_t first[INFR_TENSOR_MAX_RANK];
    int64_t stop[INFR_TENSOR_MAX_RANK];
    int64_t step[INFR_TENSOR_MAX_RANK];
    for (uint32_t a = 0U; a < rank; ++a) {
        first[a] = 0;
        stop[a] = static_cast<int64_t>(in_shape.dims[a]);
        step[a] = 1;
    }
    for (uint32_t i = 0U; i < params->axis_count; ++i) {
        const uint32_t a = params->axis[i];
        if (a >= rank) {
            return INFR_ERR_INVALID_ARGUMENT;
        }
        for (uint32_t j = 0U; j < i; ++j) {
            if (params->axis[j] == a) {
                return INFR_ERR_INVALID_ARGUMENT;
            }
        }
        first[a] = params->start[i];
        stop[a] = params->end[i];
        step[a] = params->step[i];
        if (step[a] < 1) {
            return INFR_ERR_UNSUPPORTED;
        }
    }

    ptrdiff_t base = 0;
    plan->total = 1U;
    for (uint32_t a = 0U; a < rank; ++a) {
        const int64_t dim = static_cast<int64_t>(in_shape.dims[a]);
        /* Negative indices count from the end, then everything is clamped into the axis.
         *
         * 负下标从末尾数，之后一律夹进轴内。 */
        int64_t s = (first[a] < 0) ? (dim + first[a]) : first[a];
        int64_t e = (stop[a] < 0) ? (dim + stop[a]) : stop[a];
        if (s < 0) {
            s = 0;
        }
        if (s > dim) {
            s = dim;
        }
        if (e < 0) {
            e = 0;
        }
        if (e > dim) {
            e = dim;
        }
        const int64_t length = (e > s) ? (((e - s) + step[a] - 1) / step[a]) : 0;
        plan->out_dims[a] = static_cast<uint32_t>(length);

        ptrdiff_t stride = 1;
        for (uint32_t k = a + 1U; k < rank; ++k) {
            stride *= static_cast<ptrdiff_t>(in_shape.dims[k]);
        }
        plan->in_delta[a] = static_cast<ptrdiff_t>(step[a]) * stride;
        base += static_cast<ptrdiff_t>(s) * stride;
        plan->total *= static_cast<size_t>(length);
    }
    plan->rank = rank;
    plan->base = base;
    return INFR_OK;
}

} // namespace

/**
 * @brief [en] Shape of the slice `params` cuts out of `input_shape`.
 * @brief [zh] `params` 从 `input_shape` 里切出的形状。
 * @since 0.2.0
 */
infr_status infr_slice_output_shape(const infr_shape* input_shape, const infr_slice_params* params,
                                    infr_shape* out_shape) {
    if (input_shape == nullptr || params == nullptr || out_shape == nullptr) {
        return INFR_ERR_INVALID_ARGUMENT;
    }
    slice_plan plan{};
    const infr_status status = resolve_plan(*input_shape, params, &plan);
    if (status != INFR_OK) {
        return status;
    }
    out_shape->rank = static_cast<uint8_t>(plan.rank);
    out_shape->elem_count = static_cast<uint32_t>(plan.total);
    for (uint32_t a = 0U; a < plan.rank; ++a) {
        out_shape->dims[a] = plan.out_dims[a];
    }
    for (uint32_t a = plan.rank; a < INFR_TENSOR_MAX_RANK; ++a) {
        out_shape->dims[a] = 0U;
    }
    return INFR_OK;
}

/**
 * @brief [en] fp32 slice: cut a strided view out of a tensor, moving values with `copy`.
 * @brief [zh] fp32 切片：从张量里切出一个带步长的视图，用 `copy` 搬值。
 * @since 0.2.0
 */
infr_status infr_slice_f32(const infr_tensor* input, const infr_slice_params* params, infr_tensor* output,
                           infr_workspace* workspace) {
    if (input == nullptr || params == nullptr || output == nullptr || workspace == nullptr || input->data == nullptr ||
        output->data == nullptr) {
        return INFR_ERR_INVALID_ARGUMENT;
    }
    if (!infr_detail::is_f32(input) || !infr_detail::is_f32(output)) {
        return INFR_ERR_UNSUPPORTED;
    }

    slice_plan plan{};
    infr_status status = resolve_plan(input->shape, params, &plan);
    if (status != INFR_OK) {
        return status;
    }
    if (output->shape.rank != plan.rank || static_cast<size_t>(output->shape.elem_count) != plan.total) {
        return INFR_ERR_INVALID_ARGUMENT;
    }
    for (uint32_t a = 0U; a < plan.rank; ++a) {
        if (output->shape.dims[a] != plan.out_dims[a]) {
            return INFR_ERR_INVALID_ARGUMENT;
        }
    }

    /* An empty cut is a legal result and needs no work at all -- and saying so here keeps the
     * run loop below free of a zero-length special case.
     *
     * 空切片是合法结果、不需要任何搬运——把它写在这里，下面的段循环就不必再处理零长度。 */
    if (plan.total == 0U) {
        return INFR_OK;
    }
    if (plan.total > static_cast<size_t>(0xFFFFFFFFU)) {
        return INFR_ERR_UNSUPPORTED;
    }

    status = infr_kernel_init();
    if (status != INFR_OK) {
        return status;
    }
    const infr_kernel_desc* copy = infr_kernel_find_best("copy", INFR_DTYPE_F32, INFR_QUANT_NONE);
    if (copy == nullptr) {
        return INFR_ERR_NOT_IMPLEMENTED;
    }

    const auto* src = static_cast<const float*>(input->data);
    auto* dst = static_cast<float*>(output->data);
    const uint32_t rank = plan.rank;
    const uint32_t last = rank - 1U;

    /* Path A: the innermost axis has step 1, so every output run is contiguous. */
    if (plan.in_delta[last] == 1) {
        const uint32_t run = plan.out_dims[last];
        if (rank == 1U) {
            const void* in1[1] = {src + plan.base};
            void* out1[1] = {dst};
            return copy->fn(in1, out1, run, nullptr);
        }
        uint32_t idx[INFR_TENSOR_MAX_RANK] = {0U, 0U, 0U, 0U};
        ptrdiff_t offset = plan.base;
        size_t written = 0U;
        for (;;) {
            const void* in1[1] = {src + offset};
            void* out1[1] = {dst + written};
            status = copy->fn(in1, out1, run, nullptr);
            if (status != INFR_OK) {
                return status;
            }
            written += run;
            if (!infr_detail::advance_odometer(idx, plan.out_dims, plan.in_delta, rank - 1U, &offset)) {
                break;
            }
        }
        return INFR_OK;
    }

    /* Path B: the innermost axis is strided, so gather in output order and copy once. */
    const size_t mark = infr_workspace_mark(workspace);
    void* scratch_raw = nullptr;
    status = infr_workspace_alloc(workspace, plan.total * sizeof(float), alignof(float), &scratch_raw);
    if (status != INFR_OK) {
        (void)infr_workspace_release(workspace, mark);
        return status;
    }
    auto* scratch = static_cast<float*>(scratch_raw);
    {
        uint32_t idx[INFR_TENSOR_MAX_RANK] = {0U, 0U, 0U, 0U};
        ptrdiff_t offset = plan.base;
        for (size_t p = 0U; p < plan.total; ++p) {
            scratch[p] = src[offset];
            (void)infr_detail::advance_odometer(idx, plan.out_dims, plan.in_delta, rank, &offset);
        }
    }
    const void* in1[1] = {scratch};
    void* out1[1] = {dst};
    status = copy->fn(in1, out1, static_cast<uint32_t>(plan.total), nullptr);
    (void)infr_workspace_release(workspace, mark);
    return status;
}
