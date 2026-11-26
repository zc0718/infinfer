// Conan::ImportStart
#include <stdint.h>

#include "x86_internal.h"
// Conan::ImportEnd

/*!
 * @ingroup group_x86
 * @file
 * @brief [en] x86 implementation of fp32 relu.
 * @brief [zh] fp32 relu 的 x86 实现。
 * @since 0.1.0
 *
 * @note [en] The file name carries the ISA because the coverage collector in
 *            `test_package/conanfile.py` copies every `.gcda`/`.gcno` **by basename** into one
 *            flat directory, so two sources called `fc.c` overwrite each other's counts and the
 *            second one silently disappears from the report. Until that collector is fixed
 *            (gaps G-62), the six x86 bodies keep unique stems; renaming them back is the whole
 *            diff of the follow-up commit.
 * @note [zh] 文件名带 ISA 前缀，是因为 `test_package/conanfile.py` 里的覆盖率收集**按 basename**
 *            把每个 `.gcda`/`.gcno` 平铺复制到同一个目录：两个都叫 `fc.c` 的源文件会互相覆盖计数，
 *            后处理的那个从报告里静默消失。在该收集器修好之前（gaps G-62），六个 x86 实现体保持唯一
 *            的文件名主干；后续那条提交的全部 diff 就是把它们改回去。
 */

#if defined(__SSE2__)
#include <emmintrin.h>

/**
 * @brief [en] fp32 relu as compare-then-select, bitwise identical to the reference.
 * @brief [zh] fp32 relu 写成比较再选择，与参考实现逐位相同。
 * @note [en] Not `_mm_max_ps`, and the reason is the same one the NEON row records: MAXPS writes
 *            its **second** operand whenever either input is NaN, so `max(x, 0)` happens to give
 *            the reference's answer for a NaN input but `max(0, x)` does not -- a spelling whose
 *            meaning depends on operand order is not a contract. `_mm_cmpgt_ps` is false for
 *            NaN and for -0.0, and the select then yields `+0.0`, which is exactly
 *            `in[i] > 0.0F ? in[i] : 0.0F`.
 * @note [zh] 不用 `_mm_max_ps`，理由与 NEON 行记录的是同一条：任一输入为 NaN 时 MAXPS 写出它的
 *            **第二个**操作数，于是 `max(x, 0)` 恰好给出参考实现的答案而 `max(0, x)` 不会——
 *            一个含义取决于操作数顺序的写法不是契约。`_mm_cmpgt_ps` 对 NaN 与 -0.0 都返回
 *            false，选择后得到 `+0.0`，正是 `in[i] > 0.0F ? in[i] : 0.0F`。
 * @since 0.1.0
 */
infr_status infr_x86_relu_f32(const void *const *inputs, void *const *outputs, uint32_t element_count,
                              const infr_quant *quant) {
    const float *in;
    float *out;
    const __m128 zero = _mm_setzero_ps();
    uint32_t i = 0U;

    (void)quant;
    if (inputs == NULL || outputs == NULL || inputs[0] == NULL || outputs[0] == NULL) {
        return INFR_ERR_INVALID_ARGUMENT;
    }
    in = (const float *)inputs[0];
    out = (float *)outputs[0];

    for (; i + 4U <= element_count; i += 4U) {
        const __m128 v = _mm_loadu_ps(&in[i]);
        _mm_storeu_ps(&out[i], _mm_and_ps(_mm_cmpgt_ps(v, zero), v));
    }
    for (; i < element_count; ++i) {
        out[i] = in[i] > 0.0F ? in[i] : 0.0F;
    }
    return INFR_OK;
}

#endif /* __SSE2__ */
