// Conan::ImportStart
#include <stdint.h>

#include "arm_neon_internal.h"
// Conan::ImportEnd

/*!
 * @ingroup group_arm_neon
 * @file
 * @brief [en] NEON implementation of the fp32 elementwise add.
 * @brief [zh] fp32 逐元素加法的 NEON 实现。
 * @since 0.1.0
 */

#if defined(__ARM_NEON)
#include <arm_neon.h>

/**
 * @brief [en] fp32 add over four lanes with a scalar tail: `y[i] = a[i] +
 * b[i]`, bitwise.
 * @brief [zh] 四车道 fp32 加法 + 标量尾：`y[i] = a[i] + b[i]`，逐位相同。
 * @note [en] Bitwise because each element is one IEEE addition and there is
 * nothing to reorder: no accumulator crosses elements, so the lane width cannot
 * change the result. That is the opposite of the dot product in this same tier
 * (gaps G-57), and the two cases together are the reason the parity harness
 * carries both an exact comparison and a ULP window.
 * @note [zh] 逐位相同，因为每个元素只是一次 IEEE
 * 加法、没有任何可重排的东西：没有跨元素的累加器，
 *            车道宽度改变不了结果。这与同一档位里的点积正好相反（gaps
 * G-57）；两个用例合起来， 正是 parity 骨架同时带"精确比较"与"ULP
 * 窗口"两种判据的原因。
 * @since 0.1.0
 */
infr_status infr_arm_neon_add_f32(const void *const *inputs, void *const *outputs, uint32_t element_count,
                                  const infr_quant *quant) {
    const float *a;
    const float *b;
    float *y;
    uint32_t i = 0U;

    (void)quant; /* Unquantized instance: parameters carry no meaning here. */
    if (inputs == NULL || outputs == NULL || inputs[0] == NULL || inputs[1] == NULL || outputs[0] == NULL) {
        return INFR_ERR_INVALID_ARGUMENT;
    }
    a = (const float *)inputs[0];
    b = (const float *)inputs[1];
    y = (float *)outputs[0];

    for (; i + 4U <= element_count; i += 4U) {
        vst1q_f32(&y[i], vaddq_f32(vld1q_f32(&a[i]), vld1q_f32(&b[i])));
    }
    for (; i < element_count; ++i) {
        y[i] = a[i] + b[i];
    }
    return INFR_OK;
}

#endif /* __ARM_NEON */
