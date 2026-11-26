// Conan::ImportStart
#include <stdint.h>

#include "arm_neon_internal.h"
// Conan::ImportEnd

/*!
 * @ingroup group_arm_neon
 * @file
 * @brief [en] NEON implementation of the int8 relu.
 * @brief [zh] int8 relu 的 NEON 实现。
 * @since 0.1.0
 */

#if defined(__ARM_NEON)
#include <arm_neon.h>

/**
 * @brief [en] int8 relu over sixteen bytes at a time, bitwise identical to the reference.
 * @brief [zh] 一次 十六字节 的 int8 relu，与参考实现逐位相同。
 * @note [en] Plain integer maximum, unlike the fp32 row in this tier: `v > 0 ? v : 0` on two's
 *            complement int8 has no NaN and no negative zero, so max and compare-and-select are
 *            the same function here. That is exactly why the fp32 row had to avoid `max` -- the
 *            restriction came from IEEE semantics, not from the instruction.
 * @note [zh] 用普通的整数取大，与本档位的 fp32 行不同：二进制补码的 int8 上 `v > 0 ? v : 0`
 *            没有 NaN、也没有负零，因此"取大"与"比较+选择"在这里是同一个函数。fp32 行之所以
 *            必须避开 `max`，原因在 IEEE 语义而不在指令。
 * @since 0.1.0
 */
infr_status infr_arm_neon_relu_i8(const void *const *inputs, void *const *outputs, uint32_t element_count,
                                  const infr_quant *quant) {
    const int8_t *in;
    int8_t *out;
    const int8x16_t zero = vdupq_n_s8(0);
    uint32_t i = 0U;

    (void)quant;
    if (inputs == NULL || outputs == NULL || inputs[0] == NULL || outputs[0] == NULL) {
        return INFR_ERR_INVALID_ARGUMENT;
    }
    in = (const int8_t *)inputs[0];
    out = (int8_t *)outputs[0];

    for (; i + 16U <= element_count; i += 16U) {
        vst1q_s8(&out[i], vmaxq_s8(vld1q_s8(&in[i]), zero));
    }
    for (; i < element_count; ++i) {
        out[i] = in[i] > 0 ? in[i] : 0;
    }
    return INFR_OK;
}

#endif /* __ARM_NEON */
