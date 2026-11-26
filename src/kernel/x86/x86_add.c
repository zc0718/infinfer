// Conan::ImportStart
#include <stdint.h>

#include "x86_internal.h"
// Conan::ImportEnd

/*!
 * @ingroup group_x86
 * @file
 * @brief [en] x86 implementation of fp32 elementwise add.
 * @brief [zh] fp32 逐元素加法的 x86 实现。
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
 * @brief [en] fp32 elementwise add, bitwise identical to the reference.
 * @brief [zh] fp32 逐元素加法，与参考实现逐位相同。
 * @note [en] One IEEE addition per element and nothing accumulates across elements, so the lane
 *            width cannot change the answer: the verdict is bit equality, not a window.
 * @note [zh] 每元素一次 IEEE 加法、没有跨元素的累加，车道宽度改变不了结果：判据是位相等，不是窗口。
 * @since 0.1.0
 */
infr_status infr_x86_add_f32(const void *const *inputs, void *const *outputs, uint32_t element_count,
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

    for (; i + 4U <= element_count; i += 4U) {
        _mm_storeu_ps(&y[i], _mm_add_ps(_mm_loadu_ps(&a[i]), _mm_loadu_ps(&b[i])));
    }
    for (; i < element_count; ++i) {
        y[i] = a[i] + b[i];
    }
    return INFR_OK;
}

#endif /* __SSE2__ */
