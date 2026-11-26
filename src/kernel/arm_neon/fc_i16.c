// Conan::ImportStart
#include <stdint.h>

#include "../scalar/scalar_internal.h"
#include "arm_neon_internal.h"
// Conan::ImportEnd

/*!
 * @ingroup group_arm_neon
 * @file
 * @brief [en] arm_neon implementation of the int16 dot product.
 * @brief [zh] int16 点积的 arm_neon 实现。
 * @since 0.1.0
 */

#if defined(__ARM_NEON)
#include <arm_neon.h>

/**
 * @brief [en] int16 dot product with an int64 accumulator, bitwise identical to the reference.
 * @brief [zh] int64 累加的 int16 点积，与参考实现逐位相同。
 * @note [en] `vmlal_s16` accumulates into int32 lanes, and the reference accumulates in int64: with
 *            257 terms of magnitude up to 2^30 the sum exceeds int32 by an order of magnitude, so
 *            an int32 accumulator would wrap on inputs the reference handles. Each product is
 *            therefore widened all the way to int64 before it is added -- `vmull_s16` gives the
 *            exact int32 product, `vmovl_s32` sign-extends it, and two int64 lanes take the sum.
 *            The cost is a few extra instructions per four elements; the accumulator width is part
 *            of the contract (gaps G-52), and this tier is the reference implementation on ARM
 *            rather than a performance target.
 * @note [zh] `vmlal_s16` 累加进 int32 车道，而参考实现累加的是 int64：257 项、单项最大 2^30 时和
 *            会超 int32 一个数量级，用 int32 累加器会在参考实现能处理的输入上回绕。所以每个乘积在相加
 *            前一路扩到 int64——`vmull_s16` 给出精确的 int32 乘积，`vmovl_s32` 符号扩展，两个 int64
 *            车道承接。代价是每四个元素多几条指令；累加器宽度属于契约（gaps G-52），而本档在 ARM 上
 *            是参照实现而不是性能靶。
 * @note [en] Integer addition is associative, so the lane split and the horizontal fold produce the
 *            same int64 total as the reference's single left-to-right sum: the verdict is bit
 *            equality, and it holds by arithmetic rather than by a tolerance.
 * @note [zh] 整数加法可结合，所以车道拆分与水平折叠给出的 int64 总和与参考实现的单条左到右求和相同：
 *            判据是位相等，靠算术成立而不是靠容差。
 * @since 0.1.0
 */
infr_status infr_arm_neon_fc_i16(const void *const *inputs, void *const *outputs, uint32_t element_count,
                                 const infr_quant *quant) {
    const int16_t *x;
    const int16_t *w;
    int16_t *out;
    int64x2_t acc_lo;
    int64x2_t acc_hi;
    int64_t lanes[2];
    int64_t total;
    uint32_t i = 0U;

    if (inputs == NULL || outputs == NULL || inputs[0] == NULL || inputs[1] == NULL || outputs[0] == NULL) {
        return INFR_ERR_INVALID_ARGUMENT;
    }
    if (quant == NULL || quant->kind == INFR_QUANT_NONE || !(quant->requant_multiplier > 0.0F)) {
        return INFR_ERR_INVALID_ARGUMENT;
    }
    out = (int16_t *)outputs[0];
    x = (const int16_t *)inputs[0];
    w = (const int16_t *)inputs[1];

    acc_lo = vdupq_n_s64(0);
    acc_hi = vdupq_n_s64(0);
    for (; i + 8U <= element_count; i += 8U) {
        const int16x8_t vx = vld1q_s16(&x[i]);
        const int16x8_t vw = vld1q_s16(&w[i]);
        /* `vmull_*` is the widening multiply: int16 x int16 -> int32, exact. */
        const int32x4_t p_lo = vmull_s16(vget_low_s16(vx), vget_low_s16(vw));
        const int32x4_t p_hi = vmull_s16(vget_high_s16(vx), vget_high_s16(vw));
        acc_lo = vaddq_s64(acc_lo, vmovl_s32(vget_low_s32(p_lo)));
        acc_hi = vaddq_s64(acc_hi, vmovl_s32(vget_high_s32(p_lo)));
        acc_lo = vaddq_s64(acc_lo, vmovl_s32(vget_low_s32(p_hi)));
        acc_hi = vaddq_s64(acc_hi, vmovl_s32(vget_high_s32(p_hi)));
    }
    vst1q_s64(lanes, acc_lo);
    total = lanes[0] + lanes[1];
    vst1q_s64(lanes, acc_hi);
    total += lanes[0] + lanes[1];

    for (; i < element_count; ++i) {
        total += (int64_t)x[i] * (int64_t)w[i];
    }

    /* The reference's own requantizer, so rounding and saturation have one implementation rather
     * than two that should agree. */
    out[0] = infr_scalar_requantize_i16(total, quant);
    return INFR_OK;
}

#endif /* __ARM_NEON */
