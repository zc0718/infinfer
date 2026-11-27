// Conan::ImportStart
#include <cstddef>
#include <cstdint>

#include <infinfer/runtime/graph.h>

#include "shape_internal.hpp"
// Conan::ImportEnd

/*!
 * @ingroup group_runtime
 * @file
 * @brief [en] Nearest resize as an L2 composition: the input walked in output
 * order, one `copy` per run.
 * @brief [zh] 最近邻缩放作为 L2 组合：按输出顺序走输入，每一段一次 `copy`。
 * @since 0.2.0
 */

/* Two decisions shape this operator, and both are about arithmetic rather than
 * about layout.
 *
 * First, the mapping is integer: `src = (dst * in) / out`. Scaling by a float
 * factor and flooring gives a different answer once extents grow, and it is not
 * what any exporter computes, so the integer form is both exact and the
 * compatible one. It also makes the second decision provable: floor division is
 * monotone and rises by 0 or 1 per output step, so consecutive outputs either
 * repeat one input element or walk forward. Any run of consecutive source
 * indices is therefore contiguous in the input and moves with a single `copy`.
 * A 2x integer upscale costs one call per two outputs; a downscale degenerates
 * to one call per element, which is stated rather than hidden -- those outputs
 * really do read non-adjacent inputs, and inventing a gather for them would
 * only add a workspace to reach the same number of reads.
 *
 * Second, no workspace: unlike `transpose` and the strided `slice` path,
 * nearest never needs a gather, because the mapping is monotone. The output is
 * written once, in order, straight from the input.
 *
 * Bilinear is deliberately absent. It reads four neighbours per output and sums
 * them with weights, which is arithmetic over a window -- the shape `conv2d`
 * has (`fc` + `add`), not the shape a `copy` has. Folding it in here would make
 * this file two operators and would hide that it needs a different kernel set.
 *
 * 决定这个算子的有两点，且都不是布局问题而是算术问题。
 *
 * 第一，映射是整数：`src = (dst * in) /
 * out`。用浮点比例再取整，在尺寸变大后会给出不同答案，
 * 也不是任何导出器的算法，所以整数形式既精确又是兼容的那一种。它同时让第二个决定可以被证明：
 * 向下取整单调、每个输出步只增 0 或
 * 1，所以相邻输出要么重复同一个输入元素、要么向前走一步。
 * 于是任何一段源下标连续的输出，在输入里都是连续的，用一次 `copy`
 * 就能搬完。整数 2 倍上采样
 * 每两个输出一次调用；降采样退化成每元素一次——这一点写出来而不是藏起来：那些输出读的确实
 * 是不同邻的输入，为它们造一个 gather 只会多出一个工作区、读到同样多的位置。
 *
 * 第二，不需要工作区：与 `transpose` 和带步长的 `slice`
 * 路径不同，最近邻永远不需要 gather，
 * 因为映射单调。输出按顺序、直接从输入写出一次。
 *
 * 双线性是刻意不在的。它每个输出读四个邻点做带权求和，属于窗口算术——`conv2d`
 * 那种形状 （`fc` + `add`），不是 `copy`
 * 那种。并进来会让这个文件变成两个算子，也会掩盖它需要的是 另一套内核。
 */

namespace {

/**
 * @brief [en] A resize resolved against a concrete input shape.
 * @brief [zh] 针对具体输入形状解析后的缩放。
 */
struct resize_plan {
    /**
   * @brief [en] Number of axes in both tensors.
   * @brief [zh] 两个张量共同的轴数。
   */
    uint32_t rank;
    /**
   * @brief [en] Extent of each input axis.
   * @brief [zh] 每条输入轴的长度。
   */
    uint32_t in_dims[INFR_TENSOR_MAX_RANK];
    /**
   * @brief [en] Extent of each output axis.
   * @brief [zh] 每条输出轴的长度。
   */
    uint32_t out_dims[INFR_TENSOR_MAX_RANK];
    /**
   * @brief [en] Element stride of each input axis.
   * @brief [zh] 每条输入轴的元素步长。
   */
    ptrdiff_t in_stride[INFR_TENSOR_MAX_RANK];
    /**
   * @brief [en] Element stride of each output axis.
   * @brief [zh] 每条输出轴的元素步长。
   */
    ptrdiff_t out_stride[INFR_TENSOR_MAX_RANK];
};

/**
 * @brief [en] Validate the request and fill in the strides.
 * @brief [zh] 校验请求并算好各轴步长。
 * @param input_shape [en] Concrete input shape.
 * @param input_shape [zh] 具体的输入形状。
 * @param params [en] Target extents and rule.
 * @param params [zh] 目标长度与规则。
 * @param plan [en] Receives the resolved plan; untouched on failure.
 * @param plan [zh] 接收解析结果；失败时不改动。
 * @return INFR_OK or an error code; the same set the public entry points
 * document.
 */
infr_status resolve_plan(const infr_shape& input_shape, const infr_resize_params& params, resize_plan* plan) {
    if (params.rank == 0U || params.rank > INFR_TENSOR_MAX_RANK) {
        return INFR_ERR_INVALID_ARGUMENT;
    }
    if (input_shape.rank != params.rank) {
        return INFR_ERR_INVALID_ARGUMENT;
    }
    if (params.mode != INFR_RESIZE_NEAREST) {
        return INFR_ERR_UNSUPPORTED;
    }

    resize_plan resolved{};
    resolved.rank = params.rank;
    for (uint32_t a = 0U; a < resolved.rank; ++a) {
        if (input_shape.dims[a] == 0U || params.out_dims[a] < 1 ||
            params.out_dims[a] > static_cast<int64_t>(0xFFFFFFFFU)) {
            return INFR_ERR_INVALID_ARGUMENT;
        }
        resolved.in_dims[a] = input_shape.dims[a];
        resolved.out_dims[a] = static_cast<uint32_t>(params.out_dims[a]);
    }

    ptrdiff_t in_step = 1;
    ptrdiff_t out_step = 1;
    for (uint32_t a = resolved.rank; a-- > 0U;) {
        resolved.in_stride[a] = in_step;
        resolved.out_stride[a] = out_step;
        in_step *= static_cast<ptrdiff_t>(resolved.in_dims[a]);
        out_step *= static_cast<ptrdiff_t>(resolved.out_dims[a]);
    }
    *plan = resolved;
    return INFR_OK;
}

/**
 * @brief [en] Source index of one output index along an axis of the given
 * extents.
 * @brief [zh] 给定轴长时，某个输出下标对应的源下标。
 */
uint32_t source_index(uint32_t out_index, uint32_t in_extent, uint32_t out_extent) {
    return static_cast<uint32_t>((static_cast<uint64_t>(out_index) * in_extent) / out_extent);
}

} // namespace

/**
 * @brief [en] Shape `params` resizes `input_shape` to.
 * @brief [zh] `params` 把 `input_shape` 缩放成的形状。
 * @since 0.2.0
 */
infr_status infr_resize_output_shape(const infr_shape* input_shape, const infr_resize_params* params,
                                     infr_shape* out_shape) {
    if (input_shape == nullptr || params == nullptr || out_shape == nullptr) {
        return INFR_ERR_INVALID_ARGUMENT;
    }
    resize_plan plan{};
    const infr_status status = resolve_plan(*input_shape, *params, &plan);
    if (status != INFR_OK) {
        return status;
    }
    out_shape->rank = static_cast<uint8_t>(plan.rank);
    out_shape->elem_count = 1U;
    for (uint32_t a = 0U; a < plan.rank; ++a) {
        out_shape->dims[a] = plan.out_dims[a];
        out_shape->elem_count *= plan.out_dims[a];
    }
    for (uint32_t a = plan.rank; a < INFR_TENSOR_MAX_RANK; ++a) {
        out_shape->dims[a] = 0U;
    }
    return INFR_OK;
}

/**
 * @brief [en] fp32 nearest resize: the input walked in output order, one `copy`
 * per run.
 * @brief [zh] fp32 最近邻缩放：按输出顺序走输入，每一段一次 `copy`。
 * @since 0.2.0
 */
infr_status infr_resize_f32(const infr_tensor* input, const infr_resize_params* params, infr_tensor* output) {
    if (input == nullptr || params == nullptr || output == nullptr || input->data == nullptr ||
        output->data == nullptr) {
        return INFR_ERR_INVALID_ARGUMENT;
    }
    if (!infr_detail::is_f32(input) || !infr_detail::is_f32(output)) {
        return INFR_ERR_UNSUPPORTED;
    }

    resize_plan plan{};
    const infr_status status = resolve_plan(input->shape, *params, &plan);
    if (status != INFR_OK) {
        return status;
    }
    if (output->shape.rank != plan.rank) {
        return INFR_ERR_INVALID_ARGUMENT;
    }
    for (uint32_t a = 0U; a < plan.rank; ++a) {
        if (output->shape.dims[a] != plan.out_dims[a]) {
            return INFR_ERR_INVALID_ARGUMENT;
        }
    }
    if (output->shape.elem_count > 0xFFFFFFFFU) {
        return INFR_ERR_UNSUPPORTED;
    }
    if (output->shape.elem_count == 0U) {
        return INFR_OK;
    }

    const infr_status init = infr_kernel_init();
    if (init != INFR_OK) {
        return init;
    }
    const infr_kernel_desc* copy = infr_kernel_find_best("copy", INFR_DTYPE_F32, INFR_QUANT_NONE);
    if (copy == nullptr) {
        return INFR_ERR_NOT_IMPLEMENTED;
    }

    const auto* src = static_cast<const float*>(input->data);
    auto* dst = static_cast<float*>(output->data);
    const size_t total = static_cast<size_t>(output->shape.elem_count);

    // Coalescing happens over the flat output, not over the innermost axis. Restricting it to the
    // innermost axis looked right and was measured wrong: with one channel per pixel (the shape a
    // grayscale model emits) that axis has extent 1, so an upscale degenerated to one call per
    // element. Walking the output linearly and extending a run whenever the source advances by
    // exactly 1 finds the real runs -- consecutive pixels of an upscaled row read consecutive
    // input pixels -- without assuming which axis they live on.
    //
    // 段合并发生在扁平输出上，而不是沿末轴。只沿末轴看着合理，实测是错的：每个像素只有一个
    // 通道时（灰度模型就是这种形状），末轴长度就是 1，于是上采样退化成每元素一次调用。按扁平
    // 输出顺序走、只要源偏移恰好加 1 就延长当前段，才能找到真正的段——被放大的行里相邻像素
    // 读的是相邻的输入像素——而且不必假设它们落在哪条轴上。

    uint32_t index[INFR_TENSOR_MAX_RANK] = {};
    ptrdiff_t source_offset = 0;
    ptrdiff_t run_source = 0;
    size_t run_start = 0;
    uint32_t run_length = 0U;
    for (size_t o = 0U; o < total; ++o) {
        if (run_length > 0U && source_offset == run_source + static_cast<ptrdiff_t>(run_length)) {
            ++run_length;
        } else {
            if (run_length > 0U) {
                const void* from = src + run_source;
                void* to = dst + static_cast<ptrdiff_t>(run_start);
                const infr_status copied = copy->fn(&from, &to, run_length, nullptr);
                if (copied != INFR_OK) {
                    return copied;
                }
            }
            run_start = o;
            run_source = source_offset;
            run_length = 1U;
        }

        if (o + 1U == total) {
            break;
        }
        // Advance the odometer and recompute the source offset: a resized axis does not move by a
        // constant stride, so the offset is a function of every index.
        //
        // 推进里程表并重算源偏移：被缩放的轴不是按常数步长前进，所以偏移是所有下标的函数。
        uint32_t a = plan.rank;
        while (a-- > 0U) {
            ++index[a];
            if (index[a] < plan.out_dims[a]) {
                break;
            }
            index[a] = 0U;
        }
        source_offset = 0;
        for (uint32_t b = 0U; b < plan.rank; ++b) {
            source_offset +=
                static_cast<ptrdiff_t>(source_index(index[b], plan.in_dims[b], plan.out_dims[b])) * plan.in_stride[b];
        }
    }
    if (run_length > 0U) {
        const void* from = src + run_source;
        void* to = dst + static_cast<ptrdiff_t>(run_start);
        return copy->fn(&from, &to, run_length, nullptr);
    }
    return INFR_OK;
}
