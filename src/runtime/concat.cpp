// Conan::ImportStart
#include <limits>

#include <infinfer/runtime/graph.h>

#include "shape_internal.hpp"
// Conan::ImportEnd

/*!
 * @ingroup group_runtime
 * @file
 * @brief [en] Concatenation as an L2 composition: every input is a run of copies into a strided window of the output.
 * @brief [zh] 拼接作为 L2 组合：每个输入都是往输出里一个带步长的窗口做若干段拷贝。
 * @since 0.2.0
 */

/* Why this operator is shaped the way it is.
 *
 * Concatenation touches no arithmetic at all, which makes it tempting to write as one
 * hand-rolled loop with `memcpy` in the middle. It is not, for one reason: the composed
 * layer names the kernels it uses, and a gate checks those names, so a data-movement
 * operator has to be built out of a declared primitive. `copy` is that primitive.
 *
 * The loop is over *runs*, not over elements: with the concatenated axis innermost (the
 * NHWC channel case, which is what the real models do) each run is one pixel's channel
 * block, so a 56x56 concat of two 64-channel maps costs 2 x 3136 calls for 401 408
 * outputs -- 0.016 calls per output element, three orders of magnitude below the
 * one-call-per-element line the call-count metric draws. Getting that wrong is not a
 * hypothetical: it is the shape of three separate incidents (G-129, G-132).
 *
 * 为什么这个算子是现在这个形状。
 *
 * 拼接完全不碰算术，所以很容易写成中间一段 `memcpy` 的手工循环。之所以不那么写，原因
 * 只有一个：组合层要点名它用的内核，而门禁会核对这些名字，所以搬运类算子必须由被声明
 * 过的原语拼出来。`copy` 就是这个原语。
 *
 * 循环是按**段**而不是按元素：当被拼的轴在最内层（NHWC 的通道情形，也是真实模型的情形）
 * 每段就是"一个像素的通道块"，于是 56x56 上拼两张 64 通道图是 2 x 3136 次调用对应
 * 401 408 个输出——每输出元素 0.016 次调用，比「每元素一次」那条线低三个数量级。写错
 * 不是假设：那个形状已经造成过三次事故（G-129、G-132）。
 */

namespace {

/**
 * @brief [en] The three numbers a concat needs: how many runs, how long each is, how wide the output is.
 * @brief [zh] 拼接需要的三个数：多少段、每段多长、输出有多宽。
 */
struct concat_plan {
    /**
     * @brief [en] Runs per input: the product of the axes before the concatenated one.
     * @brief [zh] 每个输入的段数：被拼那条轴之前各轴长度之积。
     */
    uint32_t outer;
    /**
     * @brief [en] Elements after the concatenated axis, per unit of that axis.
     * @brief [zh] 被拼那条轴之后、每单位该轴的元素数。
     */
    uint32_t inner;
    /**
     * @brief [en] Sum of the axis extents across inputs.
     * @brief [zh] 各输入在该轴上的长度之和。
     */
    uint32_t total_axis;
};

/**
 * @brief [en] Validate the inputs against each other and against the output, and derive the plan.
 * @brief [zh] 校验各输入之间以及与输出之间的一致性，并推导 plan。
 * @note [en] Every axis except the concatenated one must agree exactly. That is not pedantry:
 *            a mismatched `W` would copy rows of the wrong length into the output and produce a
 *            tensor that is well-formed and wrong, which no downstream shape check catches.
 * @note [zh] 除被拼的那条轴外，每条轴都必须严格相等。这不是抠字眼：`W` 不一致会把长度
 *            不对的行拷进输出，产出一个形状合法但内容错误的张量，下游任何形状检查都发现不了。
 * @since 0.2.0
 */
infr_status build_concat_plan(const infr_tensor* const* inputs, const infr_concat_params* params,
                              const infr_tensor* output, concat_plan* plan) {
    if (params->input_count == 0U) {
        return INFR_ERR_INVALID_ARGUMENT;
    }
    const infr_shape& first = inputs[0]->shape;
    const uint32_t rank = first.rank;
    if (rank == 0U || rank > INFR_TENSOR_MAX_RANK || params->axis >= rank) {
        return INFR_ERR_INVALID_ARGUMENT;
    }

    uint32_t outer = 1U;
    for (uint32_t a = 0U; a < params->axis; ++a) {
        outer *= first.dims[a];
    }
    uint32_t inner = 1U;
    for (uint32_t a = params->axis + 1U; a < rank; ++a) {
        inner *= first.dims[a];
    }

    uint32_t total_axis = 0U;
    for (uint32_t i = 0U; i < params->input_count; ++i) {
        const infr_tensor* in = inputs[i];
        if (in == nullptr || in->data == nullptr) {
            return INFR_ERR_INVALID_ARGUMENT;
        }
        if (!infr_detail::is_f32(in)) {
            return INFR_ERR_UNSUPPORTED;
        }
        if (in->shape.rank != rank) {
            return INFR_ERR_INVALID_ARGUMENT;
        }
        for (uint32_t a = 0U; a < rank; ++a) {
            if (a != params->axis && in->shape.dims[a] != first.dims[a]) {
                return INFR_ERR_INVALID_ARGUMENT;
            }
        }
        total_axis += in->shape.dims[params->axis];
    }

    if (output->shape.rank != rank) {
        return INFR_ERR_INVALID_ARGUMENT;
    }
    for (uint32_t a = 0U; a < rank; ++a) {
        const uint32_t want = (a == params->axis) ? total_axis : first.dims[a];
        if (output->shape.dims[a] != want) {
            return INFR_ERR_INVALID_ARGUMENT;
        }
    }

    plan->outer = outer;
    plan->inner = inner;
    plan->total_axis = total_axis;
    return INFR_OK;
}

} // namespace

/**
 * @brief [en] Shape of the concatenation of `inputs` along `params->axis`.
 * @brief [zh] 各输入沿 `params->axis` 拼接后的形状。
 * @since 0.2.0
 */
infr_status infr_concat_output_shape(const infr_tensor* const* inputs, const infr_concat_params* params,
                                     infr_shape* out_shape) {
    if (inputs == nullptr || params == nullptr || out_shape == nullptr || params->input_count == 0U) {
        return INFR_ERR_INVALID_ARGUMENT;
    }
    if (inputs[0] == nullptr) {
        return INFR_ERR_INVALID_ARGUMENT;
    }
    const infr_shape& first = inputs[0]->shape;
    const uint32_t rank = first.rank;
    if (rank == 0U || rank > INFR_TENSOR_MAX_RANK || params->axis >= rank) {
        return INFR_ERR_INVALID_ARGUMENT;
    }

    *out_shape = first;
    uint32_t total_axis = 0U;
    for (uint32_t i = 0U; i < params->input_count; ++i) {
        if (inputs[i] == nullptr || inputs[i]->shape.rank != rank) {
            return INFR_ERR_INVALID_ARGUMENT;
        }
        for (uint32_t a = 0U; a < rank; ++a) {
            if (a != params->axis && inputs[i]->shape.dims[a] != first.dims[a]) {
                return INFR_ERR_INVALID_ARGUMENT;
            }
        }
        total_axis += inputs[i]->shape.dims[params->axis];
    }
    out_shape->dims[params->axis] = total_axis;
    return INFR_OK;
}

/**
 * @brief [en] fp32 concatenation of `params->input_count` tensors along one axis.
 * @brief [zh] 把 `params->input_count` 个张量沿一条轴做 fp32 拼接。
 * @since 0.2.0
 */
infr_status infr_concat_f32(const infr_tensor* const* inputs, const infr_concat_params* params, infr_tensor* output,
                            infr_workspace* workspace) {
    /* Concatenation allocates nothing; the parameter exists so the operator table stays
     * uniform. It is still required to be non-null, for the same reason as its siblings.
     *
     * 拼接不分配任何东西；这个参数存在只是为了算子表保持一致。它仍然必须非空，理由与
     * 同层的其它算子相同。 */
    (void)workspace;
    if (inputs == nullptr || params == nullptr || output == nullptr || workspace == nullptr ||
        output->data == nullptr) {
        return INFR_ERR_INVALID_ARGUMENT;
    }

    concat_plan plan{};
    infr_status status = build_concat_plan(inputs, params, output, &plan);
    if (status != INFR_OK) {
        return status;
    }

    status = infr_kernel_init();
    if (status != INFR_OK) {
        return status;
    }
    const infr_kernel_desc* copy = infr_kernel_find_best("copy", INFR_DTYPE_F32, INFR_QUANT_NONE);
    if (copy == nullptr) {
        return INFR_ERR_NOT_IMPLEMENTED;
    }

    auto* dst = static_cast<float*>(output->data);
    uint32_t axis_offset = 0U;
    for (uint32_t i = 0U; i < params->input_count; ++i) {
        const auto* src = static_cast<const float*>(inputs[i]->data);
        const uint32_t width = inputs[i]->shape.dims[params->axis];
        const size_t run = static_cast<size_t>(plan.inner) * width;
        if (run > std::numeric_limits<uint32_t>::max()) {
            /* A single run wider than a `uint32_t` count cannot be handed to the kernel, and
             * splitting it would mean inventing a second copy path. Refuse instead.
             *
             * 单段宽度超过 `uint32_t` 计数时无法交给内核，而拆段意味着再造一条拷贝路径。
             * 这里选择拒绝。 */
            return INFR_ERR_UNSUPPORTED;
        }
        for (uint32_t o = 0U; o < plan.outer; ++o) {
            const void* in1[1] = {src + (static_cast<size_t>(o) * run)};
            void* out1[1] = {dst + ((static_cast<size_t>(o) * plan.total_axis) + axis_offset) * plan.inner};
            status = copy->fn(in1, out1, static_cast<uint32_t>(run), nullptr);
            if (status != INFR_OK) {
                return status;
            }
        }
        axis_offset += width;
    }
    return INFR_OK;
}
