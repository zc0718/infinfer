// Conan::ImportStart
#include <stdint.h>

#include "rvv_internal.h"
// Conan::ImportEnd

/*!
 * @ingroup group_rvv
 * @file
 * @brief [en] RVV implementation of the broadcast scalar multiply.
 * @brief [zh] 按标量广播乘法的 RVV 实现。
 * @since 0.1.0
 */

#if defined(__riscv_vector)
#include <riscv_vector.h>

/**
 * @brief [en] fp32 multiply by one broadcast factor, bitwise identical to the reference.
 * @brief [zh] 用一个广播因子的 fp32 乘法，与参考实现逐位相同。
 * @note [en] Same class as `add`: one IEEE multiply per element, so the verdict is bit equality.
 * @note [zh] 与 `add` 同类：每元素一次 IEEE 乘法，判据是位相等。
 * @since 0.1.0
 */
infr_status infr_rvv_scale_f32(const void *const *inputs, void *const *outputs, uint32_t element_count,
                               const infr_quant *quant) {
    const float *in;
    float *out;
    float scalar;
    uint32_t i = 0U;

    (void)quant;
    if (inputs == NULL || outputs == NULL || inputs[0] == NULL || inputs[1] == NULL || outputs[0] == NULL) {
        return INFR_ERR_INVALID_ARGUMENT;
    }
    in = (const float *)inputs[0];
    out = (float *)outputs[0];
    /* Read after the guard, never in an initialiser: the reference returns a status for a
     * missing factor, and a dereference here would crash on the same call. */
    scalar = ((const float *)inputs[1])[0];

    while (i < element_count) {
        const size_t vl = __riscv_vsetvl_e32m1((size_t)(element_count - i));
        const vfloat32m1_t v = __riscv_vle32_v_f32m1(&in[i], vl);
        __riscv_vse32_v_f32m1(&out[i], __riscv_vfmul_vf_f32m1(v, scalar, vl), vl);
        i += (uint32_t)vl;
    }
    return INFR_OK;
}

#endif /* __riscv_vector */
