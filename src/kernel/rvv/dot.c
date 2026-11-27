// Conan::ImportStart
#include <stdint.h>

#include "../scalar/scalar_internal.h"
#include "rvv_internal.h"
// Conan::ImportEnd

/*!
 * @ingroup group_rvv
 * @file
 * @brief [en] RVV int8 dot product that stops at the int32 accumulator.
 * @brief [zh] 停在 int32 累加器上的 RVV int8 点积。
 * @since 0.2.0
 */

#if defined(__riscv_vector)
#include <riscv_vector.h>

/**
 * @brief [en] int8 dot product into an int32 accumulator, bitwise equal to the scalar body.
 * @brief [zh] int8 点积写进 int32 累加器，与 scalar 体逐位相同。
 * @note [en] Same chain as `infr_rvv_fc_i8`: widening first, because multiplying in int8 lanes turns
 *            over almost immediately (two int8 values already reach 2^14) -- e8m1 to e16m2 for both
 *            operands, whose product is exact in int16, then `vwmacc` into int32 lanes, the
 *            reference's own accumulator width.
 * @note [zh] 与 `infr_rvv_fc_i8` 同一条链路：先加宽，因为在 int8 车道上相乘几乎立刻回绕（两个 int8
 *            相乘已达 2^14）——两个操作数都 e8m1 到 e16m2，其乘积在 int16 里精确，再用 `vwmacc`
 *            累加进 int32 车道，正是参考自己的累加器宽度。
 * @since 0.2.0
 */
infr_status infr_rvv_dot_i8(const void *const *inputs, void *const *outputs, uint32_t element_count,
                            const infr_quant *quant) {
    const int8_t *x;
    const int8_t *w;
    int32_t total = 0;
    uint32_t i = 0U;

    (void)quant;
    if (inputs == NULL || outputs == NULL || inputs[0] == NULL || inputs[1] == NULL || outputs[0] == NULL) {
        return INFR_ERR_INVALID_ARGUMENT;
    }
    x = (const int8_t *)inputs[0];
    w = (const int8_t *)inputs[1];

    if (element_count != 0U) {
        /* Tail undisturbed: `vsetvl` may return a shorter `vl` than `vlmax`, and the accumulator
         * lanes beyond it were initialized to zero, so the tail cannot be read as garbage.
         *
         * 尾部不受扰动：`vsetvl` 可能返回比 `vlmax` 更短的 `vl`，而它之外的车道已初始化为零，
         * 因此尾部不会被当作垃圾读进来。 */
        const size_t vlmax = __riscv_vsetvl_e32m4((size_t)element_count);
        vint32m4_t acc = __riscv_vmv_v_x_i32m4(0, vlmax);

        while (i < element_count) {
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
    *(int32_t *)outputs[0] = total;
    return INFR_OK;
}

#endif /* __riscv_vector */
