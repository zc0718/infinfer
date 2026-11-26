// Conan::ImportStart
#include "scalar_internal.h"
// Conan::ImportEnd

/*!
 * @ingroup group_scalar
 * @file
 * @brief [en] Scalar reductions (sum, max): the reference bodies for the reduction cells.
 * @brief [zh] 标量归约（求和、取最大）：归约格的参考实现。
 * @since 0.1.0
 */

/* A reduction collapses `element_count` elements into exactly one value in
 * `outputs[0]`, which is what makes it shape-free: pooling, global average and
 * bias accumulation all reach it through (pointer, length) alone.
 *
 * The accumulation order is ascending and fixed; that order *is* the reference
 * definition, and any other tier has to reproduce it.
 *
 * 归约把 `element_count` 个元素压成 `outputs[0]` 里的**一个**值，这正是它不依赖
 * shape 的原因：池化、全局平均、偏置累加都只用（指针、长度）到达它。
 * 累加顺序固定为升序；该顺序**就是**参考定义，其它档位必须复现。
 */

/**
 * @brief [en] Sum `element_count` floats into `outputs[0]`; a zero-length sum is 0.0.
 * @brief [zh] 把 `element_count` 个 float 求和写入 `outputs[0]`；零长度求和为 0.0。
 * @since 0.1.0
 */
infr_status infr_scalar_sum_f32(const void *const *inputs, void *const *outputs, uint32_t element_count,
                                const infr_quant *quant) {
    const float *in;
    float acc = 0.0F;
    uint32_t i;

    (void)quant;
    if (inputs == NULL || outputs == NULL || inputs[0] == NULL || outputs[0] == NULL) {
        return INFR_ERR_INVALID_ARGUMENT;
    }
    in = (const float *)inputs[0];
    for (i = 0; i < element_count; ++i) {
        acc += in[i];
    }
    ((float *)outputs[0])[0] = acc;
    return INFR_OK;
}

/**
 * @brief [en] Maximum of `element_count` floats into `outputs[0]`; a zero-length max is rejected.
 * @brief [zh] 把 `element_count` 个 float 的最大值写入 `outputs[0]`；零长度取最大被拒绝。
 * @note [en] A max over an empty set has no value, so it is an argument error rather than a silent
 *            -infinity that would poison a pooling layer downstream.
 * @note [zh] 空集合的最大值不存在，因此报参数错误，而不是悄悄返回 -inf 去污染下游池化层。
 * @since 0.1.0
 */
infr_status infr_scalar_max_f32(const void *const *inputs, void *const *outputs, uint32_t element_count,
                                const infr_quant *quant) {
    const float *in;
    float best;
    uint32_t i;

    (void)quant;
    if (inputs == NULL || outputs == NULL || inputs[0] == NULL || outputs[0] == NULL) {
        return INFR_ERR_INVALID_ARGUMENT;
    }
    if (element_count == 0) {
        return INFR_ERR_INVALID_ARGUMENT;
    }
    in = (const float *)inputs[0];
    best = in[0];
    for (i = 1; i < element_count; ++i) {
        /* `>` and not `>=`: the first occurrence of a maximum wins, so two equal
         * maxima cannot make the result depend on the scan order. */
        if (in[i] > best) {
            best = in[i];
        }
    }
    ((float *)outputs[0])[0] = best;
    return INFR_OK;
}
