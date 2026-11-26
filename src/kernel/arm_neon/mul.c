// Conan::ImportStart
#include <stdint.h>

#include "arm_neon_internal.h"
// Conan::ImportEnd

/*!
 * @ingroup group_arm_neon
 * @file
 * @brief [en] NEON implementation of the fp32 elementwise multiply.
 * @brief [zh] fp32 逐元素乘法的 NEON 实现。
 * @since 0.1.0
 */

#if defined(__ARM_NEON)
#include <arm_neon.h>

/**
 * @brief [en] fp32 multiply over four lanes with a scalar tail: `y[i] = a[i] * b[i]`, bitwise.
 * @brief [zh] 四车道 fp32 乘法 + 标量尾：`y[i] = a[i] * b[i]`，逐位相同。
 * @note [en] Bitwise because each element is one IEEE multiplication and nothing
 *            accumulates across elements, so the lane width cannot change the result.
 * @note [zh] 逐位相同，因为每元素只有一次 IEEE 乘法、没有跨元素的累加，车道宽度改变不了
 *            结果。
 * @since 0.1.0
 */
infr_status infr_arm_neon_mul_f32(const void *const *inputs, void *const *outputs, uint32_t element_count,
                                  const infr_quant *quant) {
    const float *a;
    const float *b;
    float *y;
    uint32_t i = 0U;

    (void)quant;
    if (inputs == NULL || outputs == NULL || inputs[0] == NULL || inputs[1] == NULL || outputs[0] == NULL) {
        return INFR_ERR_INVALID_ARGUMENT;
    }
    a = (const float *)inputs[0];
    b = (const float *)inputs[1];
    y = (float *)outputs[0];

    for (; i + 4U <= element_count; i += 4U) {
        vst1q_f32(&y[i], vmulq_f32(vld1q_f32(&a[i]), vld1q_f32(&b[i])));
    }
    for (; i < element_count; ++i) {
        y[i] = a[i] * b[i];
    }
    return INFR_OK;
}

#endif /* __ARM_NEON */
