// Conan::ImportStart
#include <stdint.h>

#include "../scalar/scalar_internal.h"
#include "x86_internal.h"
// Conan::ImportEnd

/*!
 * @ingroup group_x86
 * @file
 * @brief [en] x86 implementation of the int8 dot product.
 * @brief [zh] int8 点积的 x86 实现。
 * @since 0.1.0
 */

#if defined(__SSE2__)
#include <emmintrin.h>

/**
 * @brief [en] int8 dot product with an int32 accumulator, then the reference's own requantizer.
 * @brief [zh] int32 累加的 int8 点积，尾部用参考实现自己的再量化器。
 * @note [en] There is no 8-bit dot product in SSE2, so the bytes are sign-extended to int16 and
 *            `_mm_madd_epi16` does the paired multiply-add into int32 lanes -- which is exactly
 *            the accumulator type the reference uses, so the two bodies are **bitwise** equal
 *            even though the summation order differs: integer addition is associative, which is
 *            the same argument the DSP tier relies on (gaps G-51).
 * @note [zh] SSE2 没有 8 位点积，所以先把字节符号扩展成 int16，再用 `_mm_madd_epi16` 成对乘加
 *            成 int32 车道——正是参考实现用的累加器类型，因此两个体**逐位**相等，即使求和顺序
 *            不同：整数加法可结合，这与 DSP 档位依赖的是同一条理由（gaps G-51）。
 * @note [en] The tail calls `infr_scalar_clamp_i8()`, the reference's own clamp, rather than
 *            repeating the rounding and saturation here. Two floating-point roundings that "should"
 *            agree are two things that can drift; one function cannot drift from itself.
 * @note [zh] 尾部调用参考实现自己的 `infr_scalar_clamp_i8()`，而不是在这里把舍入与饱和再写一遍。
 *            两处"应该一致"的浮点舍入就是两处会漂移的东西；一个函数无法与自己漂移。
 * @note [en] The accumulator is int32 like the reference, so a sequence whose true sum overflows
 *            int32 is undefined for both -- see the contract boundary in `matrix.yml`.
 * @note [zh] 累加器与参考实现一样是 int32，因此真和超出 int32 的序列对两者都是未定义的——边界
 *            见 `matrix.yml` 的契约说明。
 * @since 0.1.0
 */
infr_status infr_x86_fc_i8(const void *const *inputs, void *const *outputs, uint32_t element_count,
                           const infr_quant *quant) {
    const int8_t *x;
    const int8_t *w;
    int8_t *out;
    int32_t total;
    double scaled;
    int32_t rounded;
    __m128i acc;
    const __m128i zero = _mm_setzero_si128();
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

    acc = _mm_setzero_si128();
    for (; i + 16U <= element_count; i += 16U) {
        const __m128i vx = _mm_loadu_si128((const __m128i *)(const void *)&x[i]);
        const __m128i vw = _mm_loadu_si128((const __m128i *)(const void *)&w[i]);
        /* 0xFF where the byte is negative, which is what makes `unpack` sign-extend. */
        const __m128i sx = _mm_cmpgt_epi8(zero, vx);
        const __m128i sw = _mm_cmpgt_epi8(zero, vw);
        acc = _mm_add_epi32(acc, _mm_madd_epi16(_mm_unpacklo_epi8(vx, sx), _mm_unpacklo_epi8(vw, sw)));
        acc = _mm_add_epi32(acc, _mm_madd_epi16(_mm_unpackhi_epi8(vx, sx), _mm_unpackhi_epi8(vw, sw)));
    }
    {
        int32_t lanes[4];
        _mm_storeu_si128((__m128i *)(void *)lanes, acc);
        total = (lanes[0] + lanes[1]) + (lanes[2] + lanes[3]);
    }
    for (; i < element_count; ++i) {
        total += (int32_t)x[i] * (int32_t)w[i];
    }

    /* The same three lines the reference writes, calling the reference's own clamp. */
    scaled = ((double)total * (double)quant->requant_multiplier) + (double)quant->zero_point;
    rounded = (int32_t)(scaled >= 0.0 ? scaled + 0.5 : scaled - 0.5);
    out[0] = infr_scalar_clamp_i8(rounded);
    return INFR_OK;
}

#endif /* __SSE2__ */
