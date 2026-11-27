// Conan::ImportStart
#include <stdint.h>

#include "../scalar/scalar_internal.h"
#include "arm_m_internal.h"
// Conan::ImportEnd

#if defined(__ARM_FEATURE_DSP)
#include <arm_acle.h>

/*!
 * @ingroup group_arm_m
 * @file
 * @brief [en] Cortex-M DSP int8 dot product that stops at the int32 accumulator.
 * @brief [zh] 停在 int32 累加器上的 Cortex-M DSP int8 点积。
 * @since 0.2.0
 */

/**
 * @brief [en] int8 dot product into an int32 accumulator, bitwise equal to the scalar body.
 * @brief [zh] int8 点积写进 int32 累加器，与 scalar 体逐位相同。
 * @note [en] Same pairing as `infr_arm_m_fc_i8`: each half of the packed word is a signed 16-bit
 *            value carrying one int8, which is the shape `SMLAD` expects, and it accumulates
 *            `a0*b0 + a1*b1` into one int32 register. Integer addition is associative, so the
 *            result equals the scalar loop's rather than merely approximating it.
 * @note [zh] 与 `infr_arm_m_fc_i8` 同样的打包：打包字的每一半都是承载一个 int8 的有符号 16 位值，
 *            正是 `SMLAD` 期望的形状，它把 `a0*b0 + a1*b1` 累加进一个 int32 寄存器。整数加法可
 *            结合，因此结果与标量循环**相等**而不是接近。
 * @since 0.2.0
 */
infr_status infr_arm_m_dot_i8(const void *const *inputs, void *const *outputs, uint32_t element_count,
                              const infr_quant *quant) {
    const int8_t *x;
    const int8_t *w;
    int32_t acc = 0;
    uint32_t i = 0U;

    (void)quant;
    if (inputs == NULL || outputs == NULL || inputs[0] == NULL || inputs[1] == NULL || outputs[0] == NULL) {
        return INFR_ERR_INVALID_ARGUMENT;
    }
    x = (const int8_t *)inputs[0];
    w = (const int8_t *)inputs[1];

    for (; i + 1U < element_count; i += 2U) {
        const int32_t packed_x =
            (int32_t)((uint32_t)(uint16_t)(int16_t)x[i] | ((uint32_t)(uint16_t)(int16_t)x[i + 1U] << 16));
        const int32_t packed_w =
            (int32_t)((uint32_t)(uint16_t)(int16_t)w[i] | ((uint32_t)(uint16_t)(int16_t)w[i + 1U] << 16));
        acc = __smlad(packed_x, packed_w, acc);
    }
    /* Odd tail: one lane left, so the paired instruction cannot help. */
    if (i < element_count) {
        acc += (int32_t)x[i] * (int32_t)w[i];
    }
    *(int32_t *)outputs[0] = acc;
    return INFR_OK;
}

#endif /* __ARM_FEATURE_DSP */
