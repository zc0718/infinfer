// Conan::ImportStart
#include <stdint.h>

#include "rvv_internal.h"
// Conan::ImportEnd

/*!
 * @ingroup group_rvv
 * @file
 * @brief [en] RVV implementation of fp32 relu.
 * @brief [zh] fp32 relu 的 RVV 实现。
 * @since 0.1.0
 */

#if defined(__riscv_vector)
#include <riscv_vector.h>

/**
 * @brief [en] fp32 relu as a vector maximum against zero, bitwise identical to the reference.
 * @brief [zh] fp32 relu 写成对零的向量取大，与参考实现逐位相同。
 * @note [en] `vfmax` is IEEE 754 `maximumNumber`, and that is exactly the reference's
 *            `in[i] > 0.0F ? in[i] : 0.0F`: a NaN input yields the other operand (the zero),
 *            and `max(-0.0, +0.0)` is `+0.0`. The bit patterns are asserted on target rather
 *            than argued, because "maximum" is the one spelling that cannot say NaN.
 * @note [zh] `vfmax` 是 IEEE 754 的 `maximumNumber`，正是参考实现的
 *            `in[i] > 0.0F ? in[i] : 0.0F`：NaN 输入返回另一个操作数（零），
 *            `max(-0.0, +0.0)` 是 `+0.0`。位模式在目标上**断言**而不是靠论证——"最大"恰好是
 *            表达不了 NaN 的那一种写法。
 * @since 0.1.0
 */
infr_status infr_rvv_relu_f32(const void *const *inputs, void *const *outputs, uint32_t element_count,
                              const infr_quant *quant) {
    const float *in;
    float *out;
    uint32_t i = 0U;

    (void)quant;
    if (inputs == NULL || outputs == NULL || inputs[0] == NULL || outputs[0] == NULL) {
        return INFR_ERR_INVALID_ARGUMENT;
    }
    in = (const float *)inputs[0];
    out = (float *)outputs[0];

    while (i < element_count) {
        const size_t vl = __riscv_vsetvl_e32m1((size_t)(element_count - i));
        const vfloat32m1_t v = __riscv_vle32_v_f32m1(&in[i], vl);
        __riscv_vse32_v_f32m1(&out[i], __riscv_vfmax_vf_f32m1(v, 0.0F, vl), vl);
        i += (uint32_t)vl;
    }
    return INFR_OK;
}

#endif /* __riscv_vector */
