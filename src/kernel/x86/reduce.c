// Conan::ImportStart
#include <stdint.h>

#include "x86_internal.h"
// Conan::ImportEnd

/*!
 * @ingroup group_x86
 * @file
 * @brief [en] x86 implementation of the fp32 maximum reduction.
 * @brief [zh] fp32 最大值归约的 x86 实现。
 * @since 0.1.0
 */

#if defined(__SSE2__)
#include <emmintrin.h>

/**
 * @brief [en] fp32 maximum reduction seeded with the first element, first occurrence wins.
 * @brief [zh] fp32 最大值归约，以第一个元素为种子，首次出现者胜。
 * @note [en] Lane-wise compare-and-select rather than `_mm_max_ps`, for the reason the relu row
 *            states: a NaN must never win, and MAXPS writes its second operand on NaN, so one of
 *            the two operand orders would let a NaN through. The lane fold and the scalar tail
 *            both use `>`, which keeps the earliest of equal values.
 * @note [zh] 逐车道比较+选择而不是 `_mm_max_ps`，理由与 relu 行相同：NaN 绝不能赢，而 MAXPS 在
 *            NaN 时写出第二个操作数，两种操作数顺序里总有一种会放过 NaN。车道折叠与标量尾都用
 *            `>`，因此相等值保留最早出现的那个。
 * @since 0.1.0
 */
infr_status infr_x86_max_f32(const void *const *inputs, void *const *outputs, uint32_t element_count,
                             const infr_quant *quant) {
    const float *in;
    __m128 acc;
    float best;
    uint32_t i;
    uint32_t k;

    (void)quant;
    if (inputs == NULL || outputs == NULL || inputs[0] == NULL || outputs[0] == NULL) {
        return INFR_ERR_INVALID_ARGUMENT;
    }
    if (element_count == 0U) {
        return INFR_ERR_INVALID_ARGUMENT; /* An empty maximum has no value to report. */
    }
    in = (const float *)inputs[0];
    best = in[0];
    i = 1U;

    /* The seed is the reference's first element replicated; each lane keeps the earliest winner
     * of its own subsequence, and the fold below keeps the earliest lane on a tie. */
    acc = _mm_set1_ps(best);
    for (; i + 4U <= element_count; i += 4U) {
        const __m128 v = _mm_loadu_ps(&in[i]);
        const __m128 take = _mm_cmpgt_ps(v, acc);
        acc = _mm_or_ps(_mm_and_ps(take, v), _mm_andnot_ps(take, acc));
    }
    {
        float lanes[4];
        _mm_storeu_ps(lanes, acc);
        for (k = 0U; k < 4U; ++k) {
            if (lanes[k] > best) {
                best = lanes[k];
            }
        }
    }
    for (; i < element_count; ++i) {
        if (in[i] > best) {
            best = in[i];
        }
    }
    ((float *)outputs[0])[0] = best;
    return INFR_OK;
}

#endif /* __SSE2__ */
