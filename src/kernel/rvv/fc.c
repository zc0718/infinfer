// Conan::ImportStart
#include <stdint.h>

#include "rvv_internal.h"
// Conan::ImportEnd

/*!
 * @ingroup group_rvv
 * @file
 * @brief [en] RVV implementation of the fp32 fully connected inner product.
 * @brief [zh] RVV 版本的 fp32 全连接内积。
 * @since 0.1.0
 */

#if defined(__riscv_vector)
#include <riscv_vector.h>

/**
 * @brief [en] fp32 dot product over `element_count` pairs, one lane block at a time.
 * @brief [zh] 在 `element_count` 对上做 fp32 点积，一次一个车道块。
 * @note [en] The accumulator keeps one partial sum per lane and is reduced **once** by
 *            `vfredosum` (ordered), which is the closest thing RVV has to "left to right".
 *            It still is not the reference's order: the reference has a single running sum,
 *            so the verdict is the float window, exactly as for the NEON row.
 * @note [zh] 累加器每个车道保留一个部分和，最后用 `vfredosum`（有序）归约**一次**，这是 RVV
 *            里最接近"从左往右"的东西。但它仍不是参考实现的顺序：参考只有一条流水和，因此
 *            判据是浮点窗口，与 NEON 行完全一致。
 * @since 0.1.0
 */
infr_status infr_rvv_fc_f32(const void *const *inputs, void *const *outputs, uint32_t element_count,
                            const infr_quant *quant) {
    const float *x;
    const float *w;
    float *out;
    float sum;
    uint32_t i = 0U;

    (void)quant; /* Unquantized instance: parameters carry no meaning here. */
    if (inputs == NULL || outputs == NULL || inputs[0] == NULL || inputs[1] == NULL || outputs[0] == NULL) {
        return INFR_ERR_INVALID_ARGUMENT;
    }
    out = (float *)outputs[0];
    if (element_count == 0U) {
        /* An empty inner dimension is a legal zero-length reduction, not an error. */
        out[0] = 0.0F;
        return INFR_OK;
    }
    x = (const float *)inputs[0];
    w = (const float *)inputs[1];

    {
        /* One VLMAX-sized accumulator; the tail updates only the lanes it owns, so the
         * reduction runs at VLMAX and never at the tail's shorter vl. */
        const size_t vlmax = __riscv_vsetvl_e32m1(element_count);
        vfloat32m1_t acc = __riscv_vfmv_v_f_f32m1(0.0F, vlmax);
        vfloat32m1_t seed;
        vfloat32m1_t folded;

        while (i < element_count) {
            const size_t vl = __riscv_vsetvl_e32m1((size_t)(element_count - i));
            const vfloat32m1_t vx = __riscv_vle32_v_f32m1(&x[i], vl);
            const vfloat32m1_t vw = __riscv_vle32_v_f32m1(&w[i], vl);
            acc = __riscv_vfmacc_vv_f32m1(acc, vx, vw, vl);
            i += (uint32_t)vl;
        }
        seed = __riscv_vfmv_v_f_f32m1(0.0F, 1U);
        folded = __riscv_vfredosum_vs_f32m1_f32m1(acc, seed, vlmax);
        sum = __riscv_vfmv_f_s_f32m1_f32(folded);
    }

    out[0] = sum;
    return INFR_OK;
}

#endif /* __riscv_vector */
