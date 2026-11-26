// Conan::ImportStart
#include <stdint.h>

#include "../scalar/scalar_internal.h"
#include "x86_internal.h"
// Conan::ImportEnd

/*!
 * @ingroup group_x86
 * @file
 * @brief [en] x86 implementation of the int16 dot product.
 * @brief [zh] int16 点积的 x86 实现。
 * @since 0.1.0
 */

#if defined(__SSE2__)
#include <emmintrin.h>

/**
 * @brief [en] int16 dot product with an int64 accumulator, then the reference's own requantizer.
 * @brief [zh] int64 累加的 int16 点积，尾部用参考实现自己的再量化器。
 * @note [en] `_mm_madd_epi16` is the obvious instruction here and it is the wrong one: it adds two
 *            products into one int32, and `(-32768)*(-32768) + (-32768)*(-32768)` is exactly
 *            2^31, which wraps. The reference accumulates in int64 and does not have that corner,
 *            so this body spends one extra instruction per element to keep it: each int16 is
 *            interleaved with a zero before the multiply, so every `_mm_madd_epi16` yields **one**
 *            product (`v*w + 0*0`) in an int32 lane that cannot overflow, and those lanes are then
 *            sign-extended and accumulated in int64.
 * @note [zh] 这里显而易见该用的指令是 `_mm_madd_epi16`，而它恰恰是错的：它把两个乘积加进一个
 *            int32，而 `(-32768)*(-32768) + (-32768)*(-32768)` 正好是 2^31，会回绕。参考实现用
 *            int64 累加、没有这个角，所以本实现每元素多花一条指令把它保住：每个 int16 在相乘前
 *            与零交错，于是每次 `_mm_madd_epi16` 只得到**一个**乘积（`v*w + 0*0`），落在不可能
 *            溢出的 int32 车道里，随后符号扩展、按 int64 累加。
 * @note [en] The cost of the safe form is a few instructions per eight elements, and this tier is
 *            the reference implementation on x86 rather than a performance target. Correctness of
 *            the accumulator type is part of the contract (gaps G-52); speed is not.
 * @note [zh] 安全形式的代价是每八个元素多几条指令，而 x86 档在本项目里是**参照实现**、不是性能
 *            靶。累加器类型属于契约（gaps G-52）；速度不属于。
 * @note [en] The tail is `infr_scalar_requantize_i16()`, the reference's own function, so the
 *            rounding and saturation have one implementation rather than two that should agree.
 * @note [zh] 尾部是参考实现自己的 `infr_scalar_requantize_i16()`，因此舍入与饱和只有一处实现，
 *            而不是两处"应该一致"的实现。
 * @since 0.1.0
 */
infr_status infr_x86_fc_i16(const void *const *inputs, void *const *outputs, uint32_t element_count,
                            const infr_quant *quant) {
    const int16_t *x;
    const int16_t *w;
    int16_t *out;
    int64_t total;
    __m128i acc0;
    __m128i acc1;
    const __m128i zero = _mm_setzero_si128();
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

    acc0 = _mm_setzero_si128();
    acc1 = _mm_setzero_si128();
    for (; i + 8U <= element_count; i += 8U) {
        const __m128i vx = _mm_loadu_si128((const __m128i *)(const void *)&x[i]);
        const __m128i vw = _mm_loadu_si128((const __m128i *)(const void *)&w[i]);
        /* Interleave with zero so each lane holds one product, never a pair sum. */
        const __m128i p_lo = _mm_madd_epi16(_mm_unpacklo_epi16(vx, zero), _mm_unpacklo_epi16(vw, zero));
        const __m128i p_hi = _mm_madd_epi16(_mm_unpackhi_epi16(vx, zero), _mm_unpackhi_epi16(vw, zero));
        /* Widen to int64 before adding: two products can sum past int32 in theory, and the
         * reference's accumulator is int64 for exactly that reason. */
        acc0 = _mm_add_epi64(acc0, _mm_unpacklo_epi32(p_lo, _mm_srai_epi32(p_lo, 31)));
        acc1 = _mm_add_epi64(acc1, _mm_unpackhi_epi32(p_lo, _mm_srai_epi32(p_lo, 31)));
        acc0 = _mm_add_epi64(acc0, _mm_unpacklo_epi32(p_hi, _mm_srai_epi32(p_hi, 31)));
        acc1 = _mm_add_epi64(acc1, _mm_unpackhi_epi32(p_hi, _mm_srai_epi32(p_hi, 31)));
    }
    {
        int64_t lanes0[2];
        int64_t lanes1[2];
        _mm_storeu_si128((__m128i *)(void *)lanes0, acc0);
        _mm_storeu_si128((__m128i *)(void *)lanes1, acc1);
        total = ((lanes0[0] + lanes0[1]) + (lanes1[0] + lanes1[1]));
    }
    for (; i < element_count; ++i) {
        total += (int64_t)x[i] * (int64_t)w[i];
    }

    out[0] = infr_scalar_requantize_i16(total, quant);
    return INFR_OK;
}

#endif /* __SSE2__ */
