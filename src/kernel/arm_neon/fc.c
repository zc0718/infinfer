// Conan::ImportStart
#include <stdint.h>

#include "arm_neon_internal.h"
// Conan::ImportEnd

/*!
 * @ingroup group_arm_neon
 * @file
 * @brief [en] NEON implementation of the fp32 dot product.
 * @brief [zh] fp32 点积的 NEON 实现。
 * @since 0.1.0
 */

#if defined(__ARM_NEON)
#include <arm_neon.h>

/**
 * @brief [en] fp32 dot product over four lanes, then a horizontal sum of the lanes and a scalar tail.
 * @brief [zh] 四车道 fp32 点积，随后对车道做水平求和，末端的余数走标量。
 * @note [en] This is the first tier whose answer is **not bit-identical** to the reference, and the
 *            contract says so rather than leaving it to be discovered: integer tiers match bit for bit
 *            (integer addition is associative), while a float dot product depends on the order the
 *            partial sums are combined in. Four lanes plus a horizontal tail is a different grouping
 *            from the reference's single running sum, and `vmlaq_f32` fuses the multiply into the
 *            accumulate. The parity harness judges fp32 with a ULP/relative window because of exactly
 *            this; a tier that claimed bitwise equality here would be claiming that float addition is
 *            associative.
 * @note [zh] 这是第一个答案**不逐位相同**的档位，契约把这件事写明，而不是留给别人去发现：整数档位逐位
 *            相同（整数加法可结合），而浮点点积取决于部分和的合并顺序。四车道 + 末端水平求和与参考实现的
 *            单条累加和是不同的分组，`vmlaq_f32` 又把乘法融进累加。parity 骨架之所以对 fp32 用 ULP/相对
 *            窗口判定，正是因为这个；在这里声称逐位相等，等于声称浮点加法可结合。
 * @note [en] The tail is scalar on purpose: a masked load would need a scratch buffer and would trade a
 *            branch for a copy, which costs more than the three multiplies it saves.
 * @note [zh] 末端刻意走标量：掩码加载需要临时缓冲区，用一个拷贝换掉一个分支，比它省下的三次乘法更贵。
 * @since 0.1.0
 */
infr_status infr_arm_neon_fc_f32(const void *const *inputs, void *const *outputs, uint32_t element_count,
                                 const infr_quant *quant) {
    const float *x;
    const float *w;
    float *out;
    float32x4_t acc;
    float lanes[4];
    float sum;
    uint32_t i = 0U;

    (void)quant; /* Unquantized instance: parameters carry no meaning here. */
    if (inputs == NULL || outputs == NULL || inputs[0] == NULL || inputs[1] == NULL || outputs[0] == NULL) {
        return INFR_ERR_INVALID_ARGUMENT;
    }
    out = (float *)outputs[0];
    x = (const float *)inputs[0];
    w = (const float *)inputs[1];

    if (element_count == 0U) {
        /* Same zero-length reduction as the reference: one output, value 0. */
        out[0] = 0.0F;
        return INFR_OK;
    }

    acc = vdupq_n_f32(0.0F);
    for (; i + 4U <= element_count; i += 4U) {
        acc = vmlaq_f32(acc, vld1q_f32(&x[i]), vld1q_f32(&w[i]));
    }
    vst1q_f32(lanes, acc);
    sum = (lanes[0] + lanes[1]) + (lanes[2] + lanes[3]);
    for (; i < element_count; ++i) {
        sum += x[i] * w[i];
    }
    out[0] = sum;
    return INFR_OK;
}

#endif /* __ARM_NEON */
