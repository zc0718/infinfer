// Conan::ImportStart
#include <stdint.h>

#include "rvv_internal.h"
// Conan::ImportEnd

/*!
 * @ingroup group_rvv
 * @file
 * @brief [en] RVV implementation of the int8 relu.
 * @brief [zh] int8 relu 的 RVV 实现。
 * @since 0.1.0
 */

#if defined(__riscv_vector)
#include <riscv_vector.h>

/**
 * @brief [en] int8 relu over one vector block at a time, bitwise identical to the reference.
 * @brief [zh] 一次 一个向量块 的 int8 relu，与参考实现逐位相同。
 * @note [en] Plain integer maximum, unlike the fp32 row in this tier: `v > 0 ? v : 0` on two's
 *            complement int8 has no NaN and no negative zero, so max and compare-and-select are
 *            the same function here. That is exactly why the fp32 row had to avoid `max` -- the
 *            restriction came from IEEE semantics, not from the instruction.
 * @note [zh] 用普通的整数取大，与本档位的 fp32 行不同：二进制补码的 int8 上 `v > 0 ? v : 0`
 *            没有 NaN、也没有负零，因此"取大"与"比较+选择"在这里是同一个函数。fp32 行之所以
 *            必须避开 `max`，原因在 IEEE 语义而不在指令。
 * @since 0.1.0
 */
infr_status infr_rvv_relu_i8(const void *const *inputs, void *const *outputs, uint32_t element_count,
                             const infr_quant *quant) {
    const int8_t *in;
    int8_t *out;
    uint32_t i = 0U;

    (void)quant;
    if (inputs == NULL || outputs == NULL || inputs[0] == NULL || outputs[0] == NULL) {
        return INFR_ERR_INVALID_ARGUMENT;
    }
    in = (const int8_t *)inputs[0];
    out = (int8_t *)outputs[0];

    while (i < element_count) {
        const size_t vl = __riscv_vsetvl_e8m1((size_t)(element_count - i));
        const vint8m1_t v = __riscv_vle8_v_i8m1(&in[i], vl);
        __riscv_vse8_v_i8m1(&out[i], __riscv_vmax_vx_i8m1(v, 0, vl), vl);
        i += (uint32_t)vl;
    }
    return INFR_OK;
}

#endif /* __riscv_vector */
