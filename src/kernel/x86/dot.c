// Conan::ImportStart
#include <stdint.h>

#include "../scalar/scalar_internal.h"
#include "x86_internal.h"
// Conan::ImportEnd

/*!
 * @ingroup group_x86
 * @file
 * @brief [en] x86 int8 dot product that stops at the int32 accumulator.
 * @brief [zh] 停在 int32 累加器上的 x86 int8 点积。
 * @since 0.2.0
 */

/* Why this exists beside `fc_i8` instead of inside it: `fc` answers "dot product *and* requantize"
 * in one call, and a quantized convolution needs the accumulator on its own so the bias can be
 * added before the single rounding step. The loop below is the same one `fc_i8` runs; the copy is
 * what it costs to leave `fc`'s contract alone rather than growing it a mode flag.
 *
 * 为什么放在 `fc_i8` 旁边而不在里面：`fc` 一次调用就回答"点积**并且**再量化"，而量化卷积需要单独
 * 拿到累加器，好让偏置在那唯一一次舍入之前加进去。下面的循环与 `fc_i8` 跑的是同一个；这份拷贝换来
 * 的是 `fc` 的契约不必长出一个模式开关。 */

#if defined(__SSE2__)
#include <emmintrin.h>

/**
 * @brief [en] int8 dot product into an int32 accumulator, bitwise equal to the scalar body.
 * @brief [zh] int8 点积写进 int32 累加器，与 scalar 体逐位相同。
 * @note [en] Same widening as `infr_x86_fc_i8`: neither operand is small enough to multiply in
 *            8-bit lanes (two int8 values already reach 2^14), so the bytes are sign-extended to
 *            int16 and `_mm_madd_epi16` accumulates paired products into int32 lanes. Different
 *            summation order, identical sum -- integer addition is associative, which is the same
 *            argument the DSP tier relies on.
 * @note [zh] 与 `infr_x86_fc_i8` 同一套加宽：两个操作数都不小到能在 8 位车道上相乘（两个 int8 相乘
 *            已达 2^14），所以字节符号扩展成 int16，再用 `_mm_madd_epi16` 把成对乘积累加进 int32
 *            车道。求和顺序不同、和相同——整数加法可结合，与 DSP 档位依赖的是同一条理由。
 * @note [en] The tail is a plain scalar loop, not a masked load: an inner dimension that is not a
 *            multiple of 16 is ordinary (a 3x3x3 window is 27), and a masked tail is the usual place
 *            where a vector body stops agreeing with the reference.
 * @note [zh] 尾部是普通标量循环、不做掩码加载：内维不是 16 的倍数很常见（3×3×3 的窗口是 27），而
 *            掩码尾部恰恰是向量体最容易与参考不一致的地方。
 * @since 0.2.0
 */
infr_status infr_x86_dot_i8(const void *const *inputs, void *const *outputs, uint32_t element_count,
                            const infr_quant *quant) {
    const int8_t *x;
    const int8_t *w;
    int32_t total;
    __m128i acc;
    const __m128i zero = _mm_setzero_si128();
    uint32_t i = 0U;

    (void)quant;
    if (inputs == NULL || outputs == NULL || inputs[0] == NULL || inputs[1] == NULL || outputs[0] == NULL) {
        return INFR_ERR_INVALID_ARGUMENT;
    }
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
    *(int32_t *)outputs[0] = total;
    return INFR_OK;
}

#endif /* __SSE2__ */
