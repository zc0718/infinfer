// Conan::ImportStart
#include <stdint.h>

#include "../scalar/scalar_internal.h"
#include "arm_neon_internal.h"
// Conan::ImportEnd

/*!
 * @ingroup group_arm_neon
 * @file
 * @brief [en] arm_neon implementation of the int8 dot product.
 * @brief [zh] int8 点积的 arm_neon 实现。
 * @since 0.1.0
 */

#if defined(__ARM_NEON)
#include <arm_neon.h>

/**
 * @brief [en] int8 dot product with an int32 accumulator, bitwise identical to the reference.
 * @brief [zh] int32 累加的 int8 点积，与参考实现逐位相同。
 * @note [en] AArch64 has a real dot-product instruction (`vdotq_s32`) and AArch32 does not, which
 *            is the same split that keeps `quantize` off this tier. Here it does **not** block the
 *            claim: `vmull_s8` gives the exact 16-bit product of two int8 values (|a*b| <= 2^14),
 *            and `vpadalq_s16` adds adjacent pairs into int32 lanes -- the same exact integer sum
 *            the reference computes, reached with instructions this baseline does have. The
 *            reference's accumulator is int32, so the width matches too.
 * @note [zh] AArch64 有真正的点积指令（`vdotq_s32`），AArch32 没有——这正是让 `quantize` 离开本档的
 *            同一处分裂。但在这里它**不阻塞声称**：`vmull_s8` 给出两个 int8 的精确 16 位乘积
 *            （|a*b| <= 2^14），`vpadalq_s16` 把相邻成对加进 int32 车道——得到的就是参考实现算的那
 *            个精确整数和，用的却是本基线确实有的指令。参考实现的累加器是 int32，宽度也对得上。
 * @note [en] Overflow is outside the contract on both sides, exactly as the x86 body states: the
 *            reference accumulates in int32 and says nothing about the case where that wraps.
 * @note [zh] 溢出在两侧都在契约之外，与 x86 那份的表述一致：参考实现用 int32 累加，对回绕的情形不
 *            作任何声称。
 * @since 0.1.0
 */
infr_status infr_arm_neon_fc_i8(const void *const *inputs, void *const *outputs, uint32_t element_count,
                                const infr_quant *quant) {
    const int8_t *x;
    const int8_t *w;
    int8_t *out;
    int32x4_t acc;
    int32_t lanes[4];
    int32_t total;
    double scaled;
    int32_t rounded;
    uint32_t i = 0U;

    if (inputs == NULL || outputs == NULL || inputs[0] == NULL || inputs[1] == NULL || outputs[0] == NULL) {
        return INFR_ERR_INVALID_ARGUMENT;
    }
    if (quant == NULL || quant->kind == INFR_QUANT_NONE || !(quant->requant_multiplier > 0.0F)) {
        return INFR_ERR_INVALID_ARGUMENT;
    }
    out = (int8_t *)outputs[0];
    x = (const int8_t *)inputs[0];
    w = (const int8_t *)inputs[1];

    acc = vdupq_n_s32(0);
    for (; i + 16U <= element_count; i += 16U) {
        const int8x16_t vx = vld1q_s8(&x[i]);
        const int8x16_t vw = vld1q_s8(&w[i]);
        /* Two exact 16-bit products, then pairwise-add them into the int32 accumulator. */
        acc = vpadalq_s16(acc, vmull_s8(vget_low_s8(vx), vget_low_s8(vw)));
        acc = vpadalq_s16(acc, vmull_s8(vget_high_s8(vx), vget_high_s8(vw)));
    }
    vst1q_s32(lanes, acc);
    total = (lanes[0] + lanes[1]) + (lanes[2] + lanes[3]);
    for (; i < element_count; ++i) {
        total += (int32_t)x[i] * (int32_t)w[i];
    }

    /* The same three lines the reference writes, calling the reference's own clamp. */
    scaled = ((double)total * (double)quant->requant_multiplier) + (double)quant->zero_point;
    rounded = (int32_t)(scaled >= 0.0 ? scaled + 0.5 : scaled - 0.5);
    out[0] = infr_scalar_clamp_i8(rounded);
    return INFR_OK;
}

#endif /* __ARM_NEON */
