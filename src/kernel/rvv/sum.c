// Conan::ImportStart
#include <stdint.h>

#include "rvv_internal.h"
// Conan::ImportEnd

/*!
 * @ingroup group_rvv
 * @file
 * @brief [en] RVV implementation of the fp32 population sum.
 * @brief [zh] fp32 全量求和的 RVV 实现。
 * @since 0.1.0
 */

#if defined(__riscv_vector)
#include <riscv_vector.h>

/**
 * @brief [en] fp32 sum into one value: vector partial sums, one ordered fold.
 * @brief [zh] fp32 求和到单值：向量部分和、一次有序折叠。
 * @note [en] Not bitwise -- the same class as the dot product and the same reason: the
 *            reference carries one running sum, this carries one per lane.
 * @note [zh] 非逐位——与点积同类同因：参考只有一条流水和，这里每个车道一条。
 * @since 0.1.0
 */
infr_status infr_rvv_sum_f32(const void *const *inputs, void *const *outputs, uint32_t element_count,
                             const infr_quant *quant) {
    const float *in;
    float total;
    uint32_t i = 0U;

    (void)quant;
    if (inputs == NULL || outputs == NULL || inputs[0] == NULL || outputs[0] == NULL) {
        return INFR_ERR_INVALID_ARGUMENT;
    }
    in = (const float *)inputs[0];

    if (element_count == 0U) {
        ((float *)outputs[0])[0] = 0.0F; /* Empty input is a zero sum, like the reference. */
        return INFR_OK;
    }

    {
        const size_t vlmax = __riscv_vsetvl_e32m1(element_count);
        vfloat32m1_t acc = __riscv_vfmv_v_f_f32m1(0.0F, vlmax);
        vfloat32m1_t seed;
        vfloat32m1_t folded;

        while (i < element_count) {
            const size_t vl = __riscv_vsetvl_e32m1((size_t)(element_count - i));
            acc = __riscv_vfadd_vv_f32m1(acc, __riscv_vle32_v_f32m1(&in[i], vl), vl);
            i += (uint32_t)vl;
        }
        seed = __riscv_vfmv_v_f_f32m1(0.0F, 1U);
        folded = __riscv_vfredosum_vs_f32m1_f32m1(acc, seed, vlmax);
        total = __riscv_vfmv_f_s_f32m1_f32(folded);
    }

    ((float *)outputs[0])[0] = total;
    return INFR_OK;
}

#endif /* __riscv_vector */
