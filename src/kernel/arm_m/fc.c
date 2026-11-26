// Conan::ImportStart
#include <stdint.h>

#include "../scalar/scalar_internal.h"
#include "arm_m_internal.h"
// Conan::ImportEnd

/*!
 * @ingroup group_arm_m
 * @file
 * @brief [en] Cortex-M DSP (`__SMLAD`) implementation of the int8 dot product.
 * @brief [zh] int8 点积的 Cortex-M DSP（`__SMLAD`）实现。
 * @since 0.1.0
 */

#if defined(__ARM_FEATURE_DSP)
#include <arm_acle.h>

/**
 * @brief [en] int8 dot product on the DSP extension: two int8 lanes packed into one int16 pair, accumulated by `SMLAD`.
 * @brief [zh] DSP 扩展上的 int8 点积：两个 int8 车道打包成一个 int16 对，由 `SMLAD` 一次累加。
 * @note [en] The result is bit-identical to the scalar body, not merely close: `SMLAD`
 *            adds `a0*b0 + a1*b1` into the same int32 accumulator the reference adds
 *            into one product at a time, and integer addition is exact as long as the
 *            accumulator does not overflow (int8 operands keep the worst case far
 *            below 2^31 for any realistic inner dimension). Requantization is shared
 *            with the scalar body, so both read the same resolved multiplier.
 * @note [zh] 结果与标量体**逐位相同**而不只是接近：`SMLAD` 把 `a0*b0 + a1*b1` 累加进同一个
 *            int32 累加器，而参考实现只是逐个乘积累加；只要累加器不溢出，整数加法就是精确的
 *            （int8 操作数在现实内维下距离 2^31 很远）。再量化与标量体共用，两者读同一个已解析乘子。
 * @note [en] Known cost, recorded rather than hidden: on ARMv7E-M there is no int8 dot
 *            product (`SDOT` arrives with ARMv8.1-M), so the packing into int16 pairs
 *            costs a shift/or per lane pair. Native int16 data would skip that step.
 * @note [zh] 已知代价，如实记录而不掩饰：ARMv7E-M 没有 int8 点积指令（`SDOT` 属
 *            ARMv8.1-M），因此打包成 int16 对每个车道对要多一次移位/或。原生 int16 数据
 *            可以省掉这一步。
 * @since 0.1.0
 */
infr_status infr_arm_m_fc_i8(const void *const *inputs, void *const *outputs, uint32_t element_count,
                             const infr_quant *quant) {
    const int8_t *x;
    const int8_t *w;
    int8_t *out;
    int32_t acc = 0;
    uint32_t i = 0U;
    double scaled;
    int32_t rounded;

    if (inputs == NULL || outputs == NULL || inputs[0] == NULL || inputs[1] == NULL || outputs[0] == NULL) {
        return INFR_ERR_INVALID_ARGUMENT;
    }
    if (quant == NULL || quant->kind == INFR_QUANT_NONE || !(quant->requant_multiplier > 0.0F)) {
        return INFR_ERR_INVALID_ARGUMENT;
    }
    out = (int8_t *)outputs[0];
    x = (const int8_t *)inputs[0];
    w = (const int8_t *)inputs[1];

    /* Paired lane: each half of the packed word is a signed 16-bit value that carries
     * one int8, which is what SMLAD expects.
     *
     * 成对车道：打包字的每一半都是承载一个 int8 的有符号 16 位值，正是 SMLAD 期望的形状。 */
    for (; i + 1U < element_count; i += 2U) {
        const int32_t packed_x =
            (int32_t)((uint32_t)(uint16_t)(int16_t)x[i] | ((uint32_t)(uint16_t)(int16_t)x[i + 1U] << 16));
        const int32_t packed_w =
            (int32_t)((uint32_t)(uint16_t)(int16_t)w[i] | ((uint32_t)(uint16_t)(int16_t)w[i + 1U] << 16));
        acc = __smlad(packed_x, packed_w, acc);
    }
    /* Odd tail: one lane left, so the paired instruction cannot help. */
    if (i < element_count) {
        acc += (int32_t)x[i] * (int32_t)w[i];
    }

    scaled = ((double)acc * (double)quant->requant_multiplier) + (double)quant->zero_point;
    rounded = (int32_t)(scaled >= 0.0 ? scaled + 0.5 : scaled - 0.5);
    out[0] = infr_scalar_clamp_i8(rounded);
    return INFR_OK;
}

#endif /* __ARM_FEATURE_DSP */
