// Conan::ImportStart
#include <stdint.h>

#include "rvv_internal.h"
// Conan::ImportEnd

/*!
 * @ingroup group_rvv
 * @file
 * @brief [en] RVV implementation of the fp32 maximum reduction.
 * @brief [zh] fp32 最大值归约的 RVV 实现。
 * @since 0.1.0
 */

#if defined(__riscv_vector)
#include <riscv_vector.h>

/**
 * @brief [en] fp32 maximum reduction, seeded with the first element.
 * @brief [zh] fp32 最大值归约，以第一个元素为种子。
 * @note [en] The seed is the reference's `best = in[0]`, and the lane-wise maximum uses
 *            `maximumNumber`, so a NaN element never wins -- the same rule the NEON row
 *            implements with compare-and-select. Only the ±0 tie could differ bitwise,
 *            which is the exception G-58 already recorded rather than a new one.
 * @note [zh] 种子就是参考实现的 `best = in[0]`，逐车道取最大用 `maximumNumber`，因此 NaN
 *            元素永远不赢——与 NEON 行用比较+选择实现的是同一条规则。唯一可能逐位不同的只有
 *            ±0 打平，那是 G-58 已经登记过的例外，不是新例外。
 * @since 0.1.0
 */
infr_status infr_rvv_max_f32(const void *const *inputs, void *const *outputs, uint32_t element_count,
                             const infr_quant *quant) {
    const float *in;
    float best;
    uint32_t i;

    (void)quant;
    if (inputs == NULL || outputs == NULL || inputs[0] == NULL || outputs[0] == NULL) {
        return INFR_ERR_INVALID_ARGUMENT;
    }
    if (element_count == 0U) {
        return INFR_ERR_INVALID_ARGUMENT; /* An empty maximum has no value to report. */
    }
    in = (const float *)inputs[0];
    best = in[0];
    i = 1U;

    if (element_count > 1U) {
        const size_t vlmax = __riscv_vsetvl_e32m1((size_t)(element_count - 1U));
        vfloat32m1_t acc = __riscv_vfmv_v_f_f32m1(best, vlmax);
        vfloat32m1_t seed;
        vfloat32m1_t folded;

        while (i < element_count) {
            const size_t vl = __riscv_vsetvl_e32m1((size_t)(element_count - i));
            const vfloat32m1_t v = __riscv_vle32_v_f32m1(&in[i], vl);
            acc = __riscv_vfmax_vv_f32m1(acc, v, vl);
            i += (uint32_t)vl;
        }
        seed = __riscv_vfmv_v_f_f32m1(best, 1U);
        folded = __riscv_vfredmax_vs_f32m1_f32m1(acc, seed, vlmax);
        best = __riscv_vfmv_f_s_f32m1_f32(folded);
    }

    ((float *)outputs[0])[0] = best;
    return INFR_OK;
}

#endif /* __riscv_vector */
