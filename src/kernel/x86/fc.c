// Conan::ImportStart
#include <stdint.h>

#include "x86_internal.h"
// Conan::ImportEnd

/*!
 * @ingroup group_x86
 * @file
 * @brief [en] x86 implementation of the fp32 fully connected inner product.
 * @brief [zh] x86 版本的 fp32 全连接内积。
 * @since 0.1.0
 */

#if defined(__SSE2__)
#include <emmintrin.h>

/**
 * @brief [en] fp32 dot product over `element_count` pairs, four lanes at a time.
 * @brief [zh] 在 `element_count` 对上做 fp32 点积，一次四车道。
 * @note [en] Unaligned loads: the kernel contract promises a pointer and a length, never an
 *            alignment. An aligned load would be faster and would be a promise this layer cannot
 *            make; the scalar tail covers the remainder.
 * @note [zh] 用未对齐加载：内核契约只承诺一个指针和一个长度，从不承诺对齐。对齐加载会更快，但那
 *            是本层无法作出的承诺；余项由标量尾覆盖。
 * @note [en] Not bitwise to the reference: one running float became four partial sums, so the
 *            verdict is the parity harness's float window.
 * @note [zh] 与参考实现**不**逐位相同：一条流水 float 变成四个部分和，因此判据是 parity 骨架的
 *            浮点窗口。
 * @since 0.1.0
 */
infr_status infr_x86_fc_f32(const void *const *inputs, void *const *outputs, uint32_t element_count,
                            const infr_quant *quant) {
    const float *x;
    const float *w;
    float *out;
    float sum;
    __m128 acc;
    uint32_t i = 0U;

    (void)quant; /* Unquantized instance: parameters carry no meaning here. */
    if (inputs == NULL || outputs == NULL || inputs[0] == NULL || inputs[1] == NULL || outputs[0] == NULL) {
        return INFR_ERR_INVALID_ARGUMENT;
    }
    out = (float *)outputs[0];
    if (element_count == 0U) {
        out[0] = 0.0F; /* An empty inner dimension is a zero-length reduction, not an error. */
        return INFR_OK;
    }
    x = (const float *)inputs[0];
    w = (const float *)inputs[1];

    acc = _mm_setzero_ps();
    for (; i + 4U <= element_count; i += 4U) {
        acc = _mm_add_ps(acc, _mm_mul_ps(_mm_loadu_ps(&x[i]), _mm_loadu_ps(&w[i])));
    }
    {
        float lanes[4];
        _mm_storeu_ps(lanes, acc);
        sum = (lanes[0] + lanes[1]) + (lanes[2] + lanes[3]);
    }
    for (; i < element_count; ++i) {
        sum += x[i] * w[i];
    }

    out[0] = sum;
    return INFR_OK;
}

#endif /* __SSE2__ */
