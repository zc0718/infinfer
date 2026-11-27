// Conan::ImportStart
#include <cstddef>

#include <infinfer/runtime/graph.h>

#include "shape_internal.hpp"
// Conan::ImportEnd

/*!
 * @ingroup group_runtime
 * @file
 * @brief [en] Softmax as an L2 composition: gather a slice, subtract its max, `exp`, `sum`, `scale`.
 * @brief [zh] softmax 作为 L2 组合：取出一个切片，减最大值，`exp`、`sum`、`scale`。
 * @since 0.2.0
 */

/* Three decisions, in the order they cost something.
 *
 * The slice is gathered first because the axis being normalised is not contiguous: elements along
 * it are `inner` apart. This is the same shape `mean` has, and it is the only shape-dependent work
 * in the operator -- every arithmetic step below comes from the registry, so this file contains no
 * arithmetic of its own.
 *
 * The max is subtracted before `exp`, because that is what makes the result finite for logits
 * above ~88: without it `expf` overflows to infinity and the division yields NaN. It is one scalar
 * subtraction per element, spent as a `shift` kernel rather than as a loop here, so the reason the
 * kernel exists is visible at the call site.
 *
 * The divide is `scale` with the reciprocal, computed once per slice: one division instead of one
 * per element, and the same kernel `mean` already uses.
 *
 * 三个决定，按代价从大到小。
 *
 * 先 gather 出切片，因为被归一化的那条轴并不连续：沿它的元素相隔 `inner` 个。这与 `mean`
 * 是同一形状，也是本算子唯一依赖 shape 的活——下面的每一步算术都来自注册表，所以这个文件
 * 自己没有任何算术。
 *
 * 在 `exp` 之前减去最大值，是为了让 logit 超过约 88 时结果仍然有限：不减的话 `expf` 溢出成
 * 无穷、除法得到 NaN。代价是每元素一次标量减法，做成 `shift` 内核而不是这里的循环，这样
 * 那个内核为什么存在，在调用处就能看见。
 *
 * 除法用 `scale` 乘倒数，每个切片只算一次：一次除法而不是每元素一次，且与 `mean` 用的是同
 * 一个内核。
 */

/**
 * @brief [en] Softmax over one axis: `exp(x - max(x)) / sum(exp(x - max(x)))`.
 * @brief [zh] 对一条轴做 softmax：`exp(x - max(x)) / sum(exp(x - max(x)))`。
 * @param input [en] fp32 tensor of rank 1..`INFR_TENSOR_MAX_RANK`.
 * @param input [zh] 秩为 1..`INFR_TENSOR_MAX_RANK` 的 fp32 张量。
 * @param params [en] The axis to normalise over.
 * @param params [zh] 被归一化的那条轴。
 * @param output [en] fp32 output with the input's shape; the data is overwritten.
 * @param output [zh] 形状与输入相同的 fp32 输出；数据会被覆盖。
 * @param workspace [en] Scratch source; must hold `dim(axis) + 2` floats.
 * @param workspace [zh] 临时空间来源；需能放下 `dim(axis) + 2` 个 float。
 * @return INFR_OK; INFR_ERR_INVALID_ARGUMENT for NULL arguments, an axis out of range or a shape
 *         mismatch; INFR_ERR_UNSUPPORTED for a non-fp32 tensor; INFR_ERR_BUFFER_TOO_SMALL when the
 *         workspace cannot hold one slice plus two scalars; INFR_ERR_NOT_IMPLEMENTED when a
 *         required kernel is missing.
 * @since 0.2.0
 */
infr_status infr_softmax_f32(const infr_tensor* input, const infr_softmax_params* params, infr_tensor* output,
                             infr_workspace* workspace) {
    if (input == nullptr || params == nullptr || output == nullptr || workspace == nullptr || input->data == nullptr ||
        output->data == nullptr) {
        return INFR_ERR_INVALID_ARGUMENT;
    }
    if (!infr_detail::is_f32(input) || !infr_detail::is_f32(output)) {
        return INFR_ERR_UNSUPPORTED;
    }
    const uint32_t rank = input->shape.rank;
    if (rank == 0U || rank > INFR_TENSOR_MAX_RANK || params->axis >= rank) {
        return INFR_ERR_INVALID_ARGUMENT;
    }
    if (output->shape.rank != rank) {
        return INFR_ERR_INVALID_ARGUMENT;
    }
    for (uint32_t a = 0U; a < rank; ++a) {
        if (output->shape.dims[a] != input->shape.dims[a]) {
            return INFR_ERR_INVALID_ARGUMENT;
        }
    }

    size_t outer = 1U;
    size_t inner = 1U;
    for (uint32_t a = 0U; a < params->axis; ++a) {
        outer *= input->shape.dims[a];
    }
    for (uint32_t a = params->axis + 1U; a < rank; ++a) {
        inner *= input->shape.dims[a];
    }
    const uint32_t length = input->shape.dims[params->axis];
    if (length == 0U || outer == 0U || inner == 0U) {
        return INFR_OK; /* An empty tensor has no slice to normalise. */
    }

    const infr_status init = infr_kernel_init();
    if (init != INFR_OK) {
        return init;
    }
    const infr_kernel_desc* max = infr_kernel_find_best("max", INFR_DTYPE_F32, INFR_QUANT_NONE);
    const infr_kernel_desc* shift = infr_kernel_find_best("shift", INFR_DTYPE_F32, INFR_QUANT_NONE);
    const infr_kernel_desc* exp = infr_kernel_find_best("exp", INFR_DTYPE_F32, INFR_QUANT_NONE);
    const infr_kernel_desc* sum = infr_kernel_find_best("sum", INFR_DTYPE_F32, INFR_QUANT_NONE);
    const infr_kernel_desc* scale = infr_kernel_find_best("scale", INFR_DTYPE_F32, INFR_QUANT_NONE);
    if (max == nullptr || shift == nullptr || exp == nullptr || sum == nullptr || scale == nullptr) {
        return INFR_ERR_NOT_IMPLEMENTED;
    }

    const size_t mark = infr_workspace_mark(workspace);
    void* slice_raw = nullptr;
    infr_status status =
        infr_workspace_alloc(workspace, (static_cast<size_t>(length) + 2U) * sizeof(float), alignof(float), &slice_raw);
    if (status != INFR_OK) {
        (void)infr_workspace_release(workspace, mark);
        return status;
    }

    auto* slice = static_cast<float*>(slice_raw);
    auto* peak = slice + length;        /* one reduced scalar */
    auto* factor = slice + length + 1U; /* the reciprocal of the sum */
    const auto* in = static_cast<const float*>(input->data);
    auto* out = static_cast<float*>(output->data);

    for (size_t o = 0U; o < outer; ++o) {
        for (size_t i = 0U; i < inner; ++i) {
            const size_t base = (o * static_cast<size_t>(length) * inner) + i;
            for (uint32_t k = 0U; k < length; ++k) {
                slice[k] = in[base + (static_cast<size_t>(k) * inner)];
            }

            const void* reduce_inputs[1] = {slice};
            void* reduce_output[2] = {peak, peak};
            status = max->fn(reduce_inputs, reduce_output, length, nullptr);
            if (status != INFR_OK) {
                (void)infr_workspace_release(workspace, mark);
                return status;
            }

            const float offset = -*peak;
            const void* shift_inputs[2] = {slice, &offset};
            void* shift_output[1] = {slice};
            status = shift->fn(shift_inputs, shift_output, length, nullptr);
            if (status != INFR_OK) {
                (void)infr_workspace_release(workspace, mark);
                return status;
            }

            status = exp->fn(reduce_inputs, shift_output, length, nullptr);
            if (status != INFR_OK) {
                (void)infr_workspace_release(workspace, mark);
                return status;
            }

            status = sum->fn(reduce_inputs, reduce_output, length, nullptr);
            if (status != INFR_OK) {
                (void)infr_workspace_release(workspace, mark);
                return status;
            }

            *factor = 1.0F / *peak;
            const void* scale_inputs[2] = {slice, factor};
            void* scale_output[1] = {slice};
            status = scale->fn(scale_inputs, scale_output, length, nullptr);
            if (status != INFR_OK) {
                (void)infr_workspace_release(workspace, mark);
                return status;
            }

            for (uint32_t k = 0U; k < length; ++k) {
                out[base + (static_cast<size_t>(k) * inner)] = slice[k];
            }
        }
    }

    return infr_workspace_release(workspace, mark);
}
