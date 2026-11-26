// Conan::ImportStart
#include <stdint.h>

#include "../scalar/scalar_internal.h"
#include "rvv_internal.h"
// Conan::ImportEnd

/*!
 * @ingroup group_rvv
 * @file
 * @brief [en] RVV implementation of the int8 dot product.
 * @brief [zh] int8 点积的 RVV 实现。
 * @since 0.1.0
 */

#if defined(__riscv_vector)
#include <riscv_vector.h>

/**
 * @brief [en] int8 dot product with an int32 accumulator, bitwise identical to the reference.
 * @brief [zh] int32 累加的 int8 点积，与参考实现逐位相同。
 * @note [en] The tempting instruction is `vmacc.vv` on int8 lanes, which would wrap almost
 *            immediately (two int8 values already reach 2^14). The chain here widens first:
 *            i8m1 -> i16m2 for both operands (their product is exact in int16), then
 *            `vwmacc` accumulates those int16 products into int32 lanes -- the reference's
 *            accumulator width, reached without any intermediate that can overflow.
 * @note [zh] 最顺手的写法是在 int8 车道上直接 `vmacc.vv`，那几乎立刻就会回绕（两个 int8 相乘已达
 *            2^14）。这里的链路先加宽：两个操作数都 i8m1 -> i16m2（其乘积在 int16 里精确），再用
 *            `vwmacc` 把 int16 乘积累加进 int32 车道——正是参考实现的累加器宽度，中间不经过任何
 *            会溢出的宽度。
 * @note [en] Overflow is outside the contract on both sides, exactly as the x86 body states.
 * @note [zh] 溢出在两侧都在契约之外，与 x86 那份的表述一致。
 * @since 0.1.0
 */
infr_status infr_rvv_fc_i8(const void *const *inputs, void *const *outputs, uint32_t element_count,
                           const infr_quant *quant) {
    const int8_t *x;
    const int8_t *w;
    int8_t *out;
    int32_t total = 0;
    double scaled;
    int32_t rounded;
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

    if (element_count > 0U) {
        const size_t vlmax = __riscv_vsetvl_e32m4((size_t)element_count);
        vint32m4_t acc = __riscv_vmv_v_x_i32m4(0, vlmax);

        while (i < element_count) {
            /* e8m1 -> e16m2 doubles the register group, so one `vl` serves both widths. */
            const size_t vl = __riscv_vsetvl_e8m1((size_t)(element_count - i));
            const vint16m2_t wx = __riscv_vsext_vf2_i16m2(__riscv_vle8_v_i8m1(&x[i], vl), vl);
            const vint16m2_t ww = __riscv_vsext_vf2_i16m2(__riscv_vle8_v_i8m1(&w[i], vl), vl);
            acc = __riscv_vwmacc_vv_i32m4(acc, wx, ww, vl);
            i += (uint32_t)vl;
        }
        {
            const vint32m1_t zero = __riscv_vmv_v_x_i32m1(0, vlmax);
            const vint32m1_t folded = __riscv_vredsum_vs_i32m4_i32m1(acc, zero, vlmax);
            total = (int32_t)__riscv_vmv_x_s_i32m1_i32(folded);
        }
    }

    /* The same three lines the reference writes, calling the reference's own clamp. */
    scaled = ((double)total * (double)quant->requant_multiplier) + (double)quant->zero_point;
    rounded = (int32_t)(scaled >= 0.0 ? scaled + 0.5 : scaled - 0.5);
    out[0] = infr_scalar_clamp_i8(rounded);
    return INFR_OK;
}

#endif /* __riscv_vector */
