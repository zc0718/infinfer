// Conan::ImportStart
#include <stdint.h>

#include "arm_neon_internal.h"
// Conan::ImportEnd

/*!
 * @ingroup group_arm_neon
 * @file
 * @brief [en] NEON implementations of the fp32 elementwise kernels.
 * @brief [zh] fp32 逐元素内核的 NEON 实现。
 * @since 0.1.0
 */

#if defined(__ARM_NEON)
#include <arm_neon.h>

/**
 * @brief [en] fp32 relu over four lanes with a scalar tail: `x > 0 ? x : 0`, bitwise.
 * @brief [zh] 四车道 fp32 relu + 标量尾：`x > 0 ? x : 0`，逐位相同。
 * @note [en] Compare-and-select rather than `vmaxq_f32`, and that choice is the whole point: the
 *            reference writes `in[i] > 0.0F ? in[i] : 0.0F`, so NaN and negative zero both become
 *            `+0.0`. `VMAX` propagates a NaN instead, which would make this tier differ from the
 *            oracle on the one input class a "max" spelling cannot express. `vcgtq_f32` returns false
 *            for NaN and `vbslq_f32` then picks the zero, which is the reference's answer exactly.
 * @note [zh] 用比较+选择而不是 `vmaxq_f32`，这正是关键：参考实现写的是
 *            `in[i] > 0.0F ? in[i] : 0.0F`，因此 NaN 与负零都变成 `+0.0`。`VMAX` 会把 NaN 传播出去，
 *            于是这一个输入类上本档位会与 oracle 不同——而这正是"max"写法表达不了的那一类。
 *            `vcgtq_f32` 对 NaN 返回 false，再由 `vbslq_f32` 选出零，恰好就是参考实现的答案。
 * @since 0.1.0
 */
infr_status infr_arm_neon_relu_f32(const void *const *inputs, void *const *outputs, uint32_t element_count,
                                   const infr_quant *quant) {
    const float *in;
    float *out;
    const float32x4_t zero = vdupq_n_f32(0.0F);
    float32x4_t v;
    uint32_t i = 0U;

    (void)quant; /* Unquantized instance: parameters carry no meaning here. */
    if (inputs == NULL || outputs == NULL || inputs[0] == NULL || outputs[0] == NULL) {
        return INFR_ERR_INVALID_ARGUMENT;
    }
    in = (const float *)inputs[0];
    out = (float *)outputs[0];

    for (; i + 4U <= element_count; i += 4U) {
        v = vld1q_f32(&in[i]);
        vst1q_f32(&out[i], vbslq_f32(vcgtq_f32(v, zero), v, zero));
    }
    for (; i < element_count; ++i) {
        out[i] = in[i] > 0.0F ? in[i] : 0.0F;
    }
    return INFR_OK;
}

#endif /* __ARM_NEON */
