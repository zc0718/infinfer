// Conan::ImportStart
#include <stdint.h>

#include "../scalar/scalar_internal.h"
#include "arm_neon_internal.h"
// Conan::ImportEnd

#if defined(__ARM_NEON)
#include <arm_neon.h>

/*!
 * @ingroup group_arm_neon
 * @file
 * @brief [en] NEON int8 dot product that stops at the int32 accumulator.
 * @brief [zh] 停在 int32 累加器上的 NEON int8 点积。
 * @since 0.2.0
 */

/**
 * @brief [en] int8 dot product into an int32 accumulator, bitwise equal to the scalar body.
 * @brief [zh] int8 点积写进 int32 累加器，与 scalar 体逐位相同。
 * @note [en] Same two exact int16 products per half as `infr_arm_neon_fc_i8`, pairwise-added into
 *            int32 by `vpadalq`: `vmull_s8` is exact for int8 inputs, and `vpadalq_s16` widens
 *            before adding, so no intermediate can wrap. Sum order differs from the scalar loop,
 *            sum does not.
 * @note [zh] 与 `infr_arm_neon_fc_i8` 一样每半各算两个精确的 int16 乘积，再由 `vpadalq` 成对加进
 *            int32：`vmull_s8` 对 int8 输入精确，`vpadalq_s16` 先加宽再相加，因此中间结果不会回绕。
 *            求和顺序与标量循环不同，和相同。
 * @since 0.2.0
 */
infr_status infr_arm_neon_dot_i8(const void *const *inputs, void *const *outputs, uint32_t element_count,
                                 const infr_quant *quant) {
    const int8_t *x;
    const int8_t *w;
    int32_t lanes[4];
    int32_t total;
    int32x4_t acc;
    uint32_t i = 0U;

    (void)quant;
    if (inputs == NULL || outputs == NULL || inputs[0] == NULL || inputs[1] == NULL || outputs[0] == NULL) {
        return INFR_ERR_INVALID_ARGUMENT;
    }
    x = (const int8_t *)inputs[0];
    w = (const int8_t *)inputs[1];

    acc = vdupq_n_s32(0);
    for (; i + 16U <= element_count; i += 16U) {
        const int8x16_t vx = vld1q_s8(&x[i]);
        const int8x16_t vw = vld1q_s8(&w[i]);
        acc = vpadalq_s16(acc, vmull_s8(vget_low_s8(vx), vget_low_s8(vw)));
        acc = vpadalq_s16(acc, vmull_s8(vget_high_s8(vx), vget_high_s8(vw)));
    }
    vst1q_s32(lanes, acc);
    total = (lanes[0] + lanes[1]) + (lanes[2] + lanes[3]);
    for (; i < element_count; ++i) {
        total += (int32_t)x[i] * (int32_t)w[i];
    }
    *(int32_t *)outputs[0] = total;
    return INFR_OK;
}

#endif /* __ARM_NEON */
