// Conan::ImportStart
#include <stdint.h>

#include "x86_internal.h"
// Conan::ImportEnd

/*!
 * @ingroup group_x86
 * @file
 * @brief [en] x86 implementation of the broadcast scalar multiply.
 * @brief [zh] 按标量广播乘法的 x86 实现。
 * @since 0.1.0
 */

#if defined(__SSE2__)
#include <emmintrin.h>

/**
 * @brief [en] fp32 multiply by one broadcast factor, bitwise identical to the reference.
 * @brief [zh] 用一个广播因子的 fp32 乘法，与参考实现逐位相同。
 * @note [en] Same class as `add`: one IEEE multiply per element, so the verdict is bit equality.
 * @note [zh] 与 `add` 同类：每元素一次 IEEE 乘法，判据是位相等。
 * @since 0.1.0
 */
infr_status infr_x86_scale_f32(const void *const *inputs, void *const *outputs, uint32_t element_count,
                               const infr_quant *quant) {
    const float *in;
    float *out;
    float scalar;
    __m128 factor;
    uint32_t i = 0U;

    (void)quant;
    if (inputs == NULL || outputs == NULL || inputs[0] == NULL || inputs[1] == NULL || outputs[0] == NULL) {
        return INFR_ERR_INVALID_ARGUMENT;
    }
    in = (const float *)inputs[0];
    out = (float *)outputs[0];
    /* Read after the guard, never in an initialiser: the reference returns a status for a missing
     * factor, and a dereference here would crash on the same call. */
    scalar = ((const float *)inputs[1])[0];
    factor = _mm_set1_ps(scalar);

    for (; i + 4U <= element_count; i += 4U) {
        _mm_storeu_ps(&out[i], _mm_mul_ps(_mm_loadu_ps(&in[i]), factor));
    }
    for (; i < element_count; ++i) {
        out[i] = in[i] * scalar;
    }
    return INFR_OK;
}

#endif /* __SSE2__ */
