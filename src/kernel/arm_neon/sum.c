// Conan::ImportStart
#include <stdint.h>

#include "arm_neon_internal.h"
// Conan::ImportEnd

/*!
 * @ingroup group_arm_neon
 * @file
 * @brief [en] NEON implementation of the fp32 population sum.
 * @brief [zh] fp32 全量求和的 NEON 实现。
 * @since 0.1.0
 */

#if defined(__ARM_NEON)
#include <arm_neon.h>

/**
 * @brief [en] fp32 sum into one value: four lanes plus a scalar tail, judged with a float window.
 * @brief [zh] fp32 求和到单值：四车道 + 标量尾，用浮点窗口判定。
 * @note [en] This is the same class as the dot product and not the same class as `add`: the
 *            reference accumulates in **one** `float` left to right, so four lanes change the
 *            grouping of the partial sums and the answer stops being bit-identical. The pairing
 *            is deliberately simple (four running lanes, then a two-step fold, then the scalar
 *            remainder) because the criterion is a ULP window, not an equality: a cleverer tree
 *            would not buy correctness, only a different rounding.
 * @note [zh] 与本档位的点积同类，而与 `add` 不同类：参考实现用**一个** `float` 从左往右累加，
 *            四车道改变了部分和的合并顺序，结果因此不再逐位相同。配对方式刻意保持简单（四条
 *            流水车道、再两级折叠、最后标量余项），因为判据是 ULP 窗口而不是相等：更花哨的
 *            归约树买不到正确性，只买到另一种舍入。
 * @since 0.1.0
 */
infr_status infr_arm_neon_sum_f32(const void *const *inputs, void *const *outputs, uint32_t element_count,
                                  const infr_quant *quant) {
    const float *in;
    float total;
    float32x4_t acc = vdupq_n_f32(0.0F);
    uint32_t i = 0U;

    (void)quant; /* Unquantized instance: parameters carry no meaning here. */
    if (inputs == NULL || outputs == NULL || inputs[0] == NULL || outputs[0] == NULL) {
        return INFR_ERR_INVALID_ARGUMENT;
    }
    in = (const float *)inputs[0];

    for (; i + 4U <= element_count; i += 4U) {
        acc = vaddq_f32(acc, vld1q_f32(&in[i]));
    }
    {
        const float32x2_t pair = vadd_f32(vget_low_f32(acc), vget_high_f32(acc));
        total = vget_lane_f32(pair, 0) + vget_lane_f32(pair, 1);
    }
    for (; i < element_count; ++i) {
        total += in[i];
    }

    /* One store, like the reference: an empty input is a zero sum, not a refusal. */
    ((float *)outputs[0])[0] = total;
    return INFR_OK;
}

#endif /* __ARM_NEON */
