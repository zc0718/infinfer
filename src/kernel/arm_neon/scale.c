// Conan::ImportStart
#include <stdint.h>

#include "arm_neon_internal.h"
// Conan::ImportEnd

/*!
 * @ingroup group_arm_neon
 * @file
 * @brief [en] NEON implementation of the fp32 broadcast multiply (`scale`).
 * @brief [zh] fp32 广播乘法（`scale`）的 NEON 实现。
 * @since 0.1.0
 */

#if defined(__ARM_NEON)
#include <arm_neon.h>

/**
 * @brief [en] fp32 multiply by the factor in `inputs[1][0]`, four lanes at a
 * time, bitwise.
 * @brief [zh] 乘以 `inputs[1][0]` 承载的因子，一次四个车道，逐位相同。
 * @note [en] The factor is read from the first element of the second input and
 * broadcast, so the arithmetic is one multiply per element -- bitwise for the
 * same reason as the add. The guard still requires `inputs[1]` even when
 * `element_count` is zero: the reference reads the factor before the loop, and
 * a tier that skipped that read would accept a call the reference refuses.
 * Matching a refusal is part of matching the contract.
 * @note [zh]
 * 因子取自第二个输入的第一个元素并广播，因此算术是"每元素一次乘法"——与加法同样逐位相同。
 *            即使 `element_count` 为零，守卫仍要求 `inputs[1]`
 * 非空：参考实现在循环之前就读这个因子，
 *            省掉这次读的档位会接受一个参考实现会拒绝的调用。**拒绝也算契约的一部分。**
 * @since 0.1.0
 */
infr_status infr_arm_neon_scale_f32(const void *const *inputs, void *const *outputs, uint32_t element_count,
                                    const infr_quant *quant) {
    const float *in;
    float *out;
    float scalar;
    float32x4_t factor;
    uint32_t i = 0U;

    (void)quant;
    if (inputs == NULL || outputs == NULL || inputs[0] == NULL || inputs[1] == NULL || outputs[0] == NULL) {
        return INFR_ERR_INVALID_ARGUMENT;
    }
    in = (const float *)inputs[0];
    out = (float *)outputs[0];
    /* Read after the guard, never in an initialiser: the first version of this
   * body read the factor on the line that declared it, which dereferenced
   * `inputs[1]` before the NULL check and would have crashed where the
   * reference returns a status. Refusing the same calls is part of matching the
   * contract. */
    scalar = ((const float *)inputs[1])[0];
    factor = vdupq_n_f32(scalar);

    for (; i + 4U <= element_count; i += 4U) {
        vst1q_f32(&out[i], vmulq_f32(vld1q_f32(&in[i]), factor));
    }
    for (; i < element_count; ++i) {
        out[i] = in[i] * scalar;
    }
    return INFR_OK;
}

#endif /* __ARM_NEON */
