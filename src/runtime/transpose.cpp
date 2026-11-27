// Conan::ImportStart
#include <cstddef>

#include <infinfer/runtime/graph.h>

#include "shape_internal.hpp"
// Conan::ImportEnd

/*!
 * @ingroup group_runtime
 * @file
 * @brief [en] Transpose as an L2 composition: a permutation of axes, built out of `copy` runs.
 * @brief [zh] 转置作为 L2 组合：轴的置换，由若干段 `copy` 拼成。
 * @since 0.2.0
 */

/* The shape of this operator is dictated by one question: which of its three possible loops
 * can be expressed as contiguous copies?
 *
 *  1. identity permutation -> one `copy` of the whole tensor (1 call).
 *  2. the last output axis maps to the last input axis -> every output run of
 *     `dims[rank-1]` floats sits contiguously in the input, so one `copy` per position of the
 *     outer axes. This covers the permutations that only shuffle the leading axes, which is
 *     most of what a folded graph produces.
 *  3. anything else (the NHWC -> NCHW case) -> no contiguous run exists in the output order,
 *     so the values are gathered one at a time into the workspace and handed to a single
 *     `copy`. One kernel call, and the gather is the only scalar loop in the operator.
 *
 * Case 3 is not a rare corner: it is what every PyTorch export of `permute(0,3,1,2)` becomes,
 * so it is the case worth measuring rather than apologising for. The workspace requirement is
 * therefore part of the contract, and it is stated in the header.
 *
 * 这个算子的形状由一个问题决定：它可能的三层循环里，哪一层能表达成连续拷贝？
 *
 *  1. 恒等置换 -> 整个张量一次 `copy`（1 次调用）。
 *  2. 输出末轴映射到输入末轴 -> 每条 `dims[rank-1]` 长的输出段在输入里都是连续的，于是
 *     外层轴每个位置一次 `copy`。这一档覆盖了"只重排前面的轴"的置换，也是折过的图里
 *     最常见的形态。
 *  3. 其余情况（NHWC -> NCHW 就是）-> 在输出顺序里不存在连续段，于是逐元素 gather 到
 *     工作区，再交给一次 `copy`。一次内核调用，而 gather 是本算子唯一的标量循环。
 *
 * 第 3 档不是罕见角落：每个 PyTorch 的 `permute(0,3,1,2)` 导出都是它，所以它值得被**测量**
 * 而不是被道歉。工作区需求因此是契约的一部分，写在头文件里。
 */

namespace {

/**
 * @brief [en] A validated permutation plus the strides it implies.
 * @brief [zh] 一个已校验的置换，以及它蕴含的步长。
 */
struct transpose_plan {
    /**
     * @brief [en] Number of axes; 1..`INFR_TENSOR_MAX_RANK`.
     * @brief [zh] 轴数；1..`INFR_TENSOR_MAX_RANK`。
     */
    uint32_t rank;
    /**
     * @brief [en] Output extent of each axis.
     * @brief [zh] 每条轴的输出长度。
     */
    uint32_t out_dims[INFR_TENSOR_MAX_RANK];
    /**
     * @brief [en] Input offset contributed by one step along output axis `a`.
     * @brief [zh] 沿输出轴 `a` 走一步所贡献的输入偏移。
     */
    ptrdiff_t in_delta[INFR_TENSOR_MAX_RANK];
    /**
     * @brief [en] Total number of elements in the output.
     * @brief [zh] 输出元素总数。
     */
    size_t total;
};

/**
 * @brief [en] Resolve `params` against the input rank and check that it is a permutation.
 * @brief [zh] 用输入秩解析 `params`，并检查它确实是一个置换。
 * @note [en] A duplicated or out-of-range entry is rejected rather than clamped. Clamping would
 *            produce a legal-looking transpose of the wrong thing, and the caller's bug would
 *            then surface as a wrong number two layers later.
 * @note [zh] 重复或越界的项被拒绝而不是夹取。夹取会产出一个"看起来合法"的错转置，然后把
 *            调用方的缺陷推迟到两层之后以错误数字的形式出现。
 * @since 0.2.0
 */
infr_status resolve_plan(const infr_shape& in_shape, const infr_transpose_params* params, transpose_plan* plan) {
    const uint32_t rank = in_shape.rank;
    if (rank == 0U || rank > INFR_TENSOR_MAX_RANK) {
        return INFR_ERR_INVALID_ARGUMENT;
    }
    if (params->rank != 0U && params->rank != rank) {
        return INFR_ERR_INVALID_ARGUMENT;
    }

    bool seen[INFR_TENSOR_MAX_RANK] = {false, false, false, false};
    for (uint32_t a = 0U; a < rank; ++a) {
        /* rank == 0 means "no perm given": reverse the axes, which is the front-end default.
         *
         * rank == 0 表示"没给置换"：反转各轴，这是前端的默认行为。 */
        const uint32_t src_axis = (params->rank == 0U) ? (rank - 1U - a) : params->perm[a];
        if (src_axis >= rank || seen[src_axis]) {
            return INFR_ERR_INVALID_ARGUMENT;
        }
        seen[src_axis] = true;

        ptrdiff_t stride = 1;
        for (uint32_t k = src_axis + 1U; k < rank; ++k) {
            stride *= static_cast<ptrdiff_t>(in_shape.dims[k]);
        }
        plan->in_delta[a] = stride;
        plan->out_dims[a] = in_shape.dims[src_axis];
    }

    plan->rank = rank;
    plan->total = 1U;
    for (uint32_t a = 0U; a < rank; ++a) {
        plan->total *= static_cast<size_t>(plan->out_dims[a]);
    }
    return INFR_OK;
}

/**
 * @brief [en] Advance a mixed-radix odometer and keep `offset` in step with it.
 * @brief [zh] 推进混合进制里程计，并让 `offset` 跟随它。
 * @param idx [en] Digits; modified in place.
 * @param idx [zh] 各位数字；就地修改。
 * @param dim [en] Radix of each digit.
 * @param dim [zh] 各位的进制。
 * @param delta [en] Offset contributed by one step of each digit.
 * @param delta [zh] 各位走一步贡献的偏移。
 * @param count [en] Number of digits.
 * @param count [zh] 位数。
 * @param offset [en] Running offset; updated in place.
 * @param offset [zh] 运行中的偏移；就地更新。
 * @return [en] False once the odometer has run past its last position.
 * @return [zh] 里程计走完最后一个位置后返回 false。
 * @note [en] The wrap correction subtracts `dim * delta` after having just added `delta`, which
 *            leaves the digit at 0 with its contribution fully removed -- the arithmetic is
 *            spelled out here because an off-by-one-delta wrap is exactly the kind of bug that
 *            only shows up as one wrong element per row.
 * @note [zh] 进位修正是在刚加过 `delta` 之后再减 `dim * delta`，于是该位回到 0 且贡献被
 *            完全移除——这里把算术写明白，因为进位少减或多减一个 `delta` 正是那种"每行错
 *            一个元素"的缺陷。
 * @since 0.2.0
 */
bool advance_odometer(uint32_t* idx, const uint32_t* dim, const ptrdiff_t* delta, uint32_t count, ptrdiff_t* offset) {
    for (uint32_t a = count; a-- > 0U;) {
        ++idx[a];
        *offset += delta[a];
        if (idx[a] < dim[a]) {
            return true;
        }
        idx[a] = 0U;
        *offset -= static_cast<ptrdiff_t>(dim[a]) * delta[a];
    }
    return false;
}

} // namespace

/**
 * @brief [en] Shape produced by transposing `input_shape` with `params`.
 * @brief [zh] 用 `params` 对 `input_shape` 转置后得到的形状。
 * @since 0.2.0
 */
infr_status infr_transpose_output_shape(const infr_shape* input_shape, const infr_transpose_params* params,
                                        infr_shape* out_shape) {
    if (input_shape == nullptr || params == nullptr || out_shape == nullptr) {
        return INFR_ERR_INVALID_ARGUMENT;
    }
    transpose_plan plan{};
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
 * @brief [en] fp32 transpose: permute the axes, moving values with `copy`.
 * @brief [zh] fp32 转置：置换各轴，用 `copy` 搬值。
 * @since 0.2.0
 */
infr_status infr_transpose_f32(const infr_tensor* input, const infr_transpose_params* params, infr_tensor* output,
                               infr_workspace* workspace) {
    if (input == nullptr || params == nullptr || output == nullptr || workspace == nullptr || input->data == nullptr ||
        output->data == nullptr) {
        return INFR_ERR_INVALID_ARGUMENT;
    }
    if (!infr_detail::is_f32(input) || !infr_detail::is_f32(output)) {
        return INFR_ERR_UNSUPPORTED;
    }

    transpose_plan plan{};
    infr_status status = resolve_plan(input->shape, params, &plan);
    if (status != INFR_OK) {
        return status;
    }
    /* The output must already be the shape this operator produces: overwriting a buffer whose
     * extent disagrees with the plan would write past it.
     *
     * 输出必须已经是本算子会产出的形状：覆盖一个与 plan 不一致的缓冲区会写到它外面。 */
    if (output->shape.rank != plan.rank || static_cast<size_t>(output->shape.elem_count) != plan.total) {
        return INFR_ERR_INVALID_ARGUMENT;
    }
    for (uint32_t a = 0U; a < plan.rank; ++a) {
        if (output->shape.dims[a] != plan.out_dims[a]) {
            return INFR_ERR_INVALID_ARGUMENT;
        }
    }

    /* `copy` takes a `uint32_t` element count on every path, so the ceiling is checked once
     * here rather than three times below.
     *
     * `copy` 的每条路径都收 `uint32_t` 元素数，所以上限在这里查一次，而不是在下面查三次。 */
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

    /* Case 1: identity -- one copy of everything. Identity is checked by walking the axes
     * from the innermost outwards and demanding the row-major stride of the output, which is
     * the only permutation whose runs tile the buffer in order.
     *
     * 情形 1：恒等——整个张量一次拷贝。恒等的判定是从最内层轴向外走、要求正好是输出的行
     * 主序步长：只有这种置换的各段才按顺序无缝铺满缓冲区。 */
    bool identity = true;
    {
        ptrdiff_t expect = 1;
        for (uint32_t a = rank; a-- > 0U;) {
            if (plan.in_delta[a] != expect) {
                identity = false;
                break;
            }
            expect *= static_cast<ptrdiff_t>(plan.out_dims[a]);
        }
    }
    if (identity) {
        const void* in1[1] = {src};
        void* out1[1] = {dst};
        return copy->fn(in1, out1, static_cast<uint32_t>(plan.total), nullptr);
    }

    /* Case 2: the last output axis is the last input axis, so runs are contiguous. */
    if (plan.in_delta[last] == 1) {
        const uint32_t run = plan.out_dims[last];
        if (rank == 1U) {
            const void* in1[1] = {src};
            void* out1[1] = {dst};
            return copy->fn(in1, out1, run, nullptr);
        }
        uint32_t idx[INFR_TENSOR_MAX_RANK] = {0U, 0U, 0U, 0U};
        ptrdiff_t offset = 0;
        size_t written = 0U;
        for (;;) {
            const void* in1[1] = {src + offset};
            void* out1[1] = {dst + written};
            status = copy->fn(in1, out1, run, nullptr);
            if (status != INFR_OK) {
                return status;
            }
            written += run;
            if (!advance_odometer(idx, plan.out_dims, plan.in_delta, rank - 1U, &offset)) {
                break;
            }
        }
        return INFR_OK;
    }

    /* Case 3: gather in output order, then one copy. This is the NHWC -> NCHW case. */
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
        ptrdiff_t offset = 0;
        for (size_t p = 0U; p < plan.total; ++p) {
            scratch[p] = src[offset];
            (void)advance_odometer(idx, plan.out_dims, plan.in_delta, rank, &offset);
        }
    }
    const void* in1[1] = {scratch};
    void* out1[1] = {dst};
    status = copy->fn(in1, out1, static_cast<uint32_t>(plan.total), nullptr);
    (void)infr_workspace_release(workspace, mark);
    return status;
}
