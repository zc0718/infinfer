// Conan::ImportStart
#include <stdint.h>

#include "rvv_internal.h"
// Conan::ImportEnd

/*!
 * @ingroup group_rvv
 * @file
 * @brief [en] RVV implementation of fp32 elementwise add.
 * @brief [zh] fp32 逐元素加法的 RVV 实现。
 * @since 0.1.0
 */

#if defined(__riscv_vector)
#include <riscv_vector.h>

/**
 * @brief [en] fp32 elementwise add over vector blocks, bitwise identical to the reference.
 * @brief [zh] 按向量块做 fp32 逐元素加法，与参考实现逐位相同。
 * @note [en] One IEEE addition per element with no accumulator crossing elements, so the
 *            lane width cannot change the answer: the verdict is bit equality, not a window.
 * @note [zh] 每元素一次 IEEE 加法、没有跨元素的累加器，车道宽度改变不了结果：判据是位相等，
 *            不是窗口。
 * @since 0.1.0
 */
infr_status infr_rvv_add_f32(const void *const *inputs, void *const *outputs, uint32_t element_count,
                             const infr_quant *quant) {
    const float *a;
    const float *b;
    float *y;
    uint32_t i = 0U;

    (void)quant;
    if (inputs == NULL || outputs == NULL || inputs[0] == NULL || inputs[1] == NULL || outputs[0] == NULL) {
        return INFR_ERR_INVALID_ARGUMENT;
    }
    a = (const float *)inputs[0];
    b = (const float *)inputs[1];
    y = (float *)outputs[0];

    while (i < element_count) {
        const size_t vl = __riscv_vsetvl_e32m1((size_t)(element_count - i));
        const vfloat32m1_t va = __riscv_vle32_v_f32m1(&a[i], vl);
        const vfloat32m1_t vb = __riscv_vle32_v_f32m1(&b[i], vl);
        __riscv_vse32_v_f32m1(&y[i], __riscv_vfadd_vv_f32m1(va, vb, vl), vl);
        i += (uint32_t)vl;
    }
    return INFR_OK;
}

#endif /* __riscv_vector */
