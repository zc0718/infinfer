// Conan::ImportStart
#include <stdint.h>

#include "x86_internal.h"
// Conan::ImportEnd

/*!
 * @ingroup group_x86
 * @file
 * @brief [en] x86 implementation of fp32 elementwise multiply.
 * @brief [zh] fp32 逐元素乘法的 x86 实现。
 * @since 0.1.0
 */

#if defined(__SSE2__)
#include <emmintrin.h>

/**
 * @brief [en] fp32 elementwise multiply, bitwise identical to the reference.
 * @brief [zh] fp32 逐元素乘法，与参考实现逐位相同。
 * @note [en] One IEEE multiplication per element and nothing accumulates across
 *            elements, so the lane width cannot change the answer: the verdict is bit
 *            equality, not a window.
 * @note [zh] 每元素一次 IEEE 乘法、没有跨元素的累加，车道宽度改变不了结果：判据是位相等，
 *            不是窗口。
 * @since 0.1.0
 */
infr_status infr_x86_mul_f32(const void *const *inputs, void *const *outputs, uint32_t element_count,
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
        _mm_storeu_ps(&y[i], _mm_mul_ps(_mm_loadu_ps(&a[i]), _mm_loadu_ps(&b[i])));
    }
    for (; i < element_count; ++i) {
        y[i] = a[i] * b[i];
    }
    return INFR_OK;
}

#endif /* __SSE2__ */
