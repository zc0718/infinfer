// Conan::ImportStart
#include <stdint.h>

#include "../scalar/scalar_internal.h"
#include "rvv_internal.h"
// Conan::ImportEnd

/*!
 * @ingroup group_rvv
 * @file
 * @brief [en] RVV implementation of the int8 quantize and dequantize pair.
 * @brief [zh] int8 量化与反量化对的 RVV 实现。
 * @note [en] Unlike the arm_neon file, this one carries **both** directions: RVV has an IEEE
 *            float divide (`vfdiv`), so `x / scale` can be written as a division rather than as a
 *            multiply by a reciprocal. That is what makes `quantize` exact here and absent there
 *            (gaps G-77).
 * @note [zh] 与 arm_neon 那份不同，本文件**两个方向都有**：RVV 有 IEEE 浮点除法（`vfdiv`），
 *            可以把 `x / scale` 写成除法而不是"乘倒数"。这正是 `quantize` 在这里精确、在那里
 *            缺席的原因（gaps G-77）。
 * @note [en] NaN input is outside the contract of both directions, because the reference casts a
 *            float to int32 (undefined for NaN) before clamping. The bodies do not paper over it;
 *            they simply do not claim anything about it.
 * @note [zh] NaN 输入在两个方向上都在契约之外：参考实现先把浮点转成 int32（NaN 是未定义）再钳位。
 *            两个实现都不掩饰这一点，只是对它不做任何声称。
 * @since 0.1.0
 */

#if defined(__riscv_vector)
#include <riscv_vector.h>

/**
 * @brief [en] fp32 to int8 quantize: divide, add the zero point, round half away from zero, clamp.
 * @brief [zh] fp32 到 int8 量化：除法、加零点、半远离零舍入、钳位。
 * @note [en] Every step mirrors the reference line for line: `scaled = in[i] / scale + zp`, then
 *            `(int32_t)(scaled >= 0 ? scaled + 0.5F : scaled - 0.5F)`, then clamp. The divide is a
 *            real division because IEEE division and multiplication by a reciprocal do not agree
 *            in the last bit, and this kernel's output is an integer produced by rounding.
 * @note [zh] 每一步都逐行对齐参考实现：`scaled = in[i] / scale + zp`，然后
 *            `(int32_t)(scaled >= 0 ? scaled + 0.5F : scaled - 0.5F)`，最后钳位。这里用真除法，
 *            因为 IEEE 除法与"乘倒数"在最后一位上并不相等，而本内核的输出是舍入后的整数。
 * @note [en] `vfsgnj` copies the sign of `scaled` onto +0.5, which is the reference's
 *            `scaled >= 0 ? +0.5F : -0.5F` for every finite input: the two differ only at `-0.0`,
 *            where both arms truncate to zero anyway.
 * @note [zh] `vfsgnj` 把 `scaled` 的符号复制到 +0.5 上，对一切有限输入等同参考的
 *            `scaled >= 0 ? +0.5F : -0.5F`：两者只在 `-0.0` 处不同，而那里两侧截断后都是零。
 * @since 0.1.0
 */
infr_status infr_rvv_quantize_f32(const void *const *inputs, void *const *outputs, uint32_t element_count,
                                  const infr_quant *quant) {
    const float *in;
    int8_t *out;
    float scale;
    float zp;
    uint32_t i = 0U;

    if (inputs == NULL || outputs == NULL || inputs[0] == NULL || outputs[0] == NULL) {
        return INFR_ERR_INVALID_ARGUMENT;
    }
    if (quant == NULL || quant->kind == INFR_QUANT_NONE || !(quant->scale > 0.0F)) {
        return INFR_ERR_INVALID_ARGUMENT;
    }
    in = (const float *)inputs[0];
    out = (int8_t *)outputs[0];
    scale = quant->scale;
    /* The reference adds `(float)zero_point`, so the integer is converted once, here, instead of
     * per element. */
    zp = (float)quant->zero_point;

    while (i < element_count) {
        /* LMUL 4 for the arithmetic, then two narrowing steps to reach int8: a narrowing move in
         * RVV 1.0 halves the element width while doubling the register group, so e32m4 -> e16m2 ->
         * e8m1 is the chain that keeps the same `vl`. Asking for a single-step narrow from e32m1
         * was the first version; the compiler rejected it on types, which is the good failure.
         *
         * 算术用 LMUL 4，再用两次窄化降到 int8：RVV 1.0 的窄化把元素宽度减半、同时把寄存器组加倍，
         * 所以 e32m4 -> e16m2 -> e8m1 是保持同一 `vl` 的链路。第一版想从 e32m1 一步窄化，编译器在
         * 类型上直接拒绝——那是好的失败方式。 */
        const size_t vl = __riscv_vsetvl_e32m4((size_t)(element_count - i));
        const vfloat32m4_t x = __riscv_vle32_v_f32m4(&in[i], vl);
        const vfloat32m4_t scaled = __riscv_vfadd_vf_f32m4(__riscv_vfdiv_vf_f32m4(x, scale, vl), zp, vl);
        /* `scaled >= 0 ? +0.5F : -0.5F`, written as compare-and-merge rather than with `vfsgnj`.
         * The first version used `vfsgnj(scaled, 0.5F)` on the belief that it copies the sign of
         * the vector onto +0.5; on this toolchain it produced `scaled` itself, which turned the
         * kernel into `2 * trunc(scaled) + 1` and was caught by the target probe (gaps G-77).
         * The two-operand semantics of that instruction are exactly the kind of thing that reads
         * right and is not, so the transcription below spells out the reference instead.
         *
         * 参考写的是 `scaled >= 0 ? +0.5F : -0.5F`，这里用比较+合并直译，而不是 `vfsgnj`。第一版
         * 以为 `vfsgnj(scaled, 0.5F)` 就是把向量的符号盖到 +0.5 上；本工具链上它返回的是 `scaled`
         * 本身，于是内核变成了 `2 * trunc(scaled) + 1`，由目标探针抓出（gaps G-77）。这条指令的
         * 双操作数语义正是那种"读起来对、其实不对"的东西，所以下面直接逐字写出参考的行为。 */
        const vbool8_t non_negative = __riscv_vmfge_vf_f32m4_b8(scaled, 0.0F, vl);
        const vfloat32m4_t half = __riscv_vfmerge_vfm_f32m4(__riscv_vfmv_v_f_f32m4(-0.5F, vl), 0.5F, non_negative, vl);
        const vfloat32m4_t shifted = __riscv_vfadd_vv_f32m4(scaled, half, vl);
        /* Round toward zero, which is what a C cast to int32 does for finite values. */
        const vint32m4_t rounded = __riscv_vfcvt_rtz_x_f_v_i32m4(shifted, vl);
        const vint32m4_t clamped = __riscv_vmin_vx_i32m4(__riscv_vmax_vx_i32m4(rounded, INT8_MIN, vl), INT8_MAX, vl);
        const vint16m2_t narrowed = __riscv_vncvt_x_x_w_i16m2(clamped, vl);
        __riscv_vse8_v_i8m1(&out[i], __riscv_vncvt_x_x_w_i8m1(narrowed, vl), vl);
        i += (uint32_t)vl;
    }
    return INFR_OK;
}

/**
 * @brief [en] int8 to fp32 dequantize: integer subtraction, then one IEEE multiply.
 * @brief [zh] int8 到 fp32 反量化：整数减法，然后一次 IEEE 乘法。
 * @note [en] Widened to int32 lanes before subtracting: `zero_point` is an int32 and a 16-bit
 *            difference could clip it, which would be a silent disagreement with the reference.
 * @note [zh] 减法前扩到 int32 车道：`zero_point` 是 int32，16 位差值可能截断它，那会变成与参考
 *            实现的静默分歧。
 * @since 0.1.0
 */
infr_status infr_rvv_dequantize_i8(const void *const *inputs, void *const *outputs, uint32_t element_count,
                                   const infr_quant *quant) {
    const int8_t *in;
    float *out;
    float scale;
    int32_t zero_point;
    uint32_t i = 0U;

    if (inputs == NULL || outputs == NULL || inputs[0] == NULL || outputs[0] == NULL) {
        return INFR_ERR_INVALID_ARGUMENT;
    }
    if (quant == NULL || quant->kind == INFR_QUANT_NONE || !(quant->scale > 0.0F)) {
        return INFR_ERR_INVALID_ARGUMENT;
    }
    in = (const int8_t *)inputs[0];
    out = (float *)outputs[0];
    scale = quant->scale;
    zero_point = quant->zero_point;

    while (i < element_count) {
        /* The mirror of the quantize chain: e8m1 -> e16m2 -> e32m4, so the zero-point subtraction
         * happens on int32 lanes and a zero point outside the int8 range is not clipped.
         * 与量化链互逆：e8m1 -> e16m2 -> e32m4，零点减法因此发生在 int32 车道上，int8 范围之外的
         * 零点不会被截断。 */
        const size_t vl = __riscv_vsetvl_e8m1((size_t)(element_count - i));
        const vint8m1_t v = __riscv_vle8_v_i8m1(&in[i], vl);
        const vint16m2_t wide16 = __riscv_vsext_vf2_i16m2(v, vl);
        const vint32m4_t wide32 = __riscv_vsext_vf2_i32m4(wide16, vl);
        const vint32m4_t centred = __riscv_vsub_vx_i32m4(wide32, zero_point, vl);
        const vfloat32m4_t f = __riscv_vfcvt_f_x_v_f32m4(centred, vl);
        __riscv_vse32_v_f32m4(&out[i], __riscv_vfmul_vf_f32m4(f, scale, vl), vl);
        i += (uint32_t)vl;
    }
    return INFR_OK;
}

#endif /* __riscv_vector */
