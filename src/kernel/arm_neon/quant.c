// Conan::ImportStart
#include <stdint.h>

#include "../scalar/scalar_internal.h"
#include "arm_neon_internal.h"
// Conan::ImportEnd

/*!
 * @ingroup group_arm_neon
 * @file
 * @brief [en] arm_neon implementation of the int8 dequantize.
 * @brief [zh] int8 反量化的 arm_neon 实现。
 * @note [en] `quantize` is deliberately absent from this file. The reference divides
 *            (`in[i] / scale`), and AArch32 NEON has no IEEE float divide: a body built on
 *            `vrecpe`/`vrecps` multiplies by a reciprocal, which is a different function. Claiming
 *            the cell for the whole tier would mean claiming bit equality on the armv7hf build
 *            too, so the cell stays `todo` and this file holds only the direction that is exact.
 *            On AArch64 `vdivq_f32` would make it clean, but the claim is per tier, not per build
 *            (gaps G-77).
 * @note [zh] 本文件**故意不含** `quantize`。参考实现做除法（`in[i] / scale`），而 AArch32 NEON
 *            没有 IEEE 浮点除法：用 `vrecpe`/`vrecps` 写出来的是"乘倒数"，是另一个函数。按档位
 *            声明这一格就意味着 armv7hf 构建也要声称逐位相等，所以该格保持 `todo`，本文件只放
 *            精确的那个方向。AArch64 上 `vdivq_f32` 可以让它成立，但声称是按档位而不是按构建
 *            （gaps G-77）。
 * @since 0.1.0
 */

#if defined(__ARM_NEON)
#include <arm_neon.h>

/**
 * @brief [en] int8 to fp32 dequantize, bitwise identical to the reference.
 * @brief [zh] int8 到 fp32 反量化，与参考实现逐位相同。
 * @note [en] The reference computes `(float)((int32_t)in[i] - zero_point) * scale`: an **integer**
 *            subtraction, then one IEEE multiply. Both steps are exact, so there is no window to
 *            argue about -- the verdict is bit equality, and the lanes are widened to int32 before
 *            the subtraction because `zero_point` is an int32 and a 16-bit difference could clip
 *            it.
 * @note [zh] 参考实现算的是 `(float)((int32_t)in[i] - zero_point) * scale`：先做**整数**减法，
 *            再做一次 IEEE 乘法。两步都精确，所以没有容差可谈——判据是位相等；减法前把车道
 *            扩到 int32，因为 `zero_point` 是 int32，16 位差值可能截断它。
 * @since 0.1.0
 */
infr_status infr_arm_neon_dequantize_i8(const void *const *inputs, void *const *outputs, uint32_t element_count,
                                        const infr_quant *quant) {
    const int8_t *in;
    float *out;
    float scale;
    int32_t zero_point;
    int32x4_t vzero;
    uint32_t i = 0U;

    if (inputs == NULL || outputs == NULL || inputs[0] == NULL || outputs[0] == NULL) {
        return INFR_ERR_INVALID_ARGUMENT;
    }
    /* Same order as the reference: pointers first, then the quantisation parameters. A NaN scale
     * fails the comparison too, which is the intent. */
    if (quant == NULL || quant->kind == INFR_QUANT_NONE || !(quant->scale > 0.0F)) {
        return INFR_ERR_INVALID_ARGUMENT;
    }
    in = (const int8_t *)inputs[0];
    out = (float *)outputs[0];
    scale = quant->scale;
    zero_point = quant->zero_point;
    vzero = vdupq_n_s32(zero_point);

    for (; i + 8U <= element_count; i += 8U) {
        /* i8 -> i16 -> i32 in two steps: NEON has no direct 8-to-32 widening move, and the
         * intermediate int16 form is what `vmovl_s16` takes. */
        const int16x8_t wide = vmovl_s8(vld1_s8(&in[i]));
        const int32x4_t lo = vsubq_s32(vmovl_s16(vget_low_s16(wide)), vzero);
        const int32x4_t hi = vsubq_s32(vmovl_s16(vget_high_s16(wide)), vzero);
        vst1q_f32(&out[i], vmulq_n_f32(vcvtq_f32_s32(lo), scale));
        vst1q_f32(&out[i + 4U], vmulq_n_f32(vcvtq_f32_s32(hi), scale));
    }
    for (; i < element_count; ++i) {
        out[i] = (float)((int32_t)in[i] - zero_point) * scale;
    }
    return INFR_OK;
}

#endif /* __ARM_NEON */
