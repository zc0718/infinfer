// Conan::ImportStart
#include <stdint.h>

#include "../scalar/scalar_internal.h"
#include "rvv_internal.h"
// Conan::ImportEnd

/*!
 * @ingroup group_rvv
 * @file
 * @brief [en] RVV implementation of the int16 dot product.
 * @brief [zh] int16 点积的 RVV 实现。
 * @since 0.1.0
 */

#if defined(__riscv_vector)
#include <riscv_vector.h>

/**
 * @brief [en] int16 dot product with an int64 accumulator, bitwise identical to the reference.
 * @brief [zh] int64 累加的 int16 点积，与参考实现逐位相同。
 * @note [en] The accumulator width is part of the contract (gaps G-52): 257 terms of magnitude up
 *            to 2^30 exceed int32 by an order of magnitude, so an int32 accumulator would wrap on
 *            inputs the reference handles. The chain is therefore i16 -> i32 (multiply) -> i64
 *            (accumulate): `vmacc` into int64 lanes needs the widening move to happen before the
 *            add, and RVV's LMUL rule -- halve the group when the element width doubles -- is what
 *            makes e16m1 -> e32m2 -> e64m4 the consistent chain.
 * @note [zh] 累加器宽度属于契约（gaps G-52）：257 项、单项最大 2^30 时和超 int32 一个数量级，用
 *            int32 累加器会在参考实现能处理的输入上回绕。所以链路是 i16 -> i32（乘）-> i64（累加）：
 *            往 int64 车道里 `vmacc` 需要在相加前完成加宽，而 RVV 的 LMUL 规则——元素宽度翻倍时
 *            寄存器组跟着翻倍——正是让 e16m1 -> e32m2 -> e64m4 成为一致链路的原因。
 * @note [en] Integer addition is associative, so the lane split and the fold give the same int64
 *            total as the reference's single left-to-right sum: bit equality holds by arithmetic.
 * @note [zh] 整数加法可结合，所以车道拆分与折叠给出的 int64 总和与参考实现的单条左到右求和相同：
 *            位相等靠算术成立。
 * @since 0.1.0
 */
infr_status infr_rvv_fc_i16(const void *const *inputs, void *const *outputs, uint32_t element_count,
                            const infr_quant *quant) {
    const int16_t *x;
    const int16_t *w;
    int16_t *out;
    int64_t total = 0;
    uint32_t i = 0U;

    if (inputs == NULL || outputs == NULL || inputs[0] == NULL || inputs[1] == NULL || outputs[0] == NULL) {
        return INFR_ERR_INVALID_ARGUMENT;
    }
    if (quant == NULL || quant->kind == INFR_QUANT_NONE || !(quant->requant_multiplier > 0.0F)) {
        return INFR_ERR_INVALID_ARGUMENT;
    }
    out = (int16_t *)outputs[0];
    x = (const int16_t *)inputs[0];
    w = (const int16_t *)inputs[1];

    if (element_count > 0U) {
        /* One accumulator group for the whole reduction: e64m4 holds 4 int64 values on a 128-bit
         * VLEN, which is enough to carry every partial sum exactly. */
        const size_t vlmax = __riscv_vsetvl_e64m4((size_t)element_count);
        vint64m4_t acc = __riscv_vmv_v_x_i64m4(0, vlmax);

        while (i < element_count) {
            const size_t vl = __riscv_vsetvl_e16m1((size_t)(element_count - i));
            const vint16m1_t vx = __riscv_vle16_v_i16m1(&x[i], vl);
            const vint16m1_t vw = __riscv_vle16_v_i16m1(&w[i], vl);
            /* Widen both operands, multiply in int32 (exact: |a*b| <= 2^30), then widen the product
             * into int64 and accumulate. Widening multiplication would be one instruction less but
             * `vwmacc` accumulates in int32, which is the width this kernel exists to avoid. */
            const vint32m2_t wx = __riscv_vsext_vf2_i32m2(vx, vl);
            const vint32m2_t ww = __riscv_vsext_vf2_i32m2(vw, vl);
            const vint32m2_t prod = __riscv_vmul_vv_i32m2(wx, ww, vl);
            const vint64m4_t wide = __riscv_vsext_vf2_i64m4(prod, vl);
            acc = __riscv_vadd_vv_i64m4_tu(acc, acc, wide, vl);
            i += (uint32_t)vl;
        }
        /* Fold the accumulator to one value. The fold order is fixed, and the total is exact, so
         * the result does not depend on it. */
        {
            const vint64m1_t zero = __riscv_vmv_v_x_i64m1(0, vlmax);
            const vint64m1_t folded = __riscv_vredsum_vs_i64m4_i64m1(acc, zero, vlmax);
            total = (int64_t)__riscv_vmv_x_s_i64m1_i64(folded);
        }
    }

    /* The reference's own requantizer, so rounding and saturation have one implementation. */
    out[0] = infr_scalar_requantize_i16(total, quant);
    return INFR_OK;
}

#endif /* __riscv_vector */
