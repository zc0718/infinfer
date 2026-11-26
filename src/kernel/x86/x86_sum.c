// Conan::ImportStart
#include <stdint.h>

#include "x86_internal.h"
// Conan::ImportEnd

/*!
 * @ingroup group_x86
 * @file
 * @brief [en] x86 implementation of the fp32 population sum.
 * @brief [zh] fp32 全量求和的 x86 实现。
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
 * @brief [en] fp32 sum into one value: four lanes, horizontal fold, scalar tail.
 * @brief [zh] fp32 求和到单值：四车道、水平折叠、标量尾。
 * @note [en] Not bitwise -- the reference carries one running float, this carries four. Judged
 *            with the float window, the same call the dot product makes.
 * @note [zh] 非逐位——参考只有一条流水 float，这里有四条。用浮点窗口判定，与点积同一条调用。
 * @since 0.1.0
 */
infr_status infr_x86_sum_f32(const void *const *inputs, void *const *outputs, uint32_t element_count,
                             const infr_quant *quant) {
    const float *in;
    float total;
    __m128 acc;
    uint32_t i = 0U;

    (void)quant;
    if (inputs == NULL || outputs == NULL || inputs[0] == NULL || outputs[0] == NULL) {
        return INFR_ERR_INVALID_ARGUMENT;
    }
    in = (const float *)inputs[0];

    acc = _mm_setzero_ps();
    for (; i + 4U <= element_count; i += 4U) {
        acc = _mm_add_ps(acc, _mm_loadu_ps(&in[i]));
    }
    {
        float lanes[4];
        _mm_storeu_ps(lanes, acc);
        total = (lanes[0] + lanes[1]) + (lanes[2] + lanes[3]);
    }
    for (; i < element_count; ++i) {
        total += in[i];
    }

    /* One store, like the reference: an empty input is a zero sum, not a refusal. */
    ((float *)outputs[0])[0] = total;
    return INFR_OK;
}

#endif /* __SSE2__ */
