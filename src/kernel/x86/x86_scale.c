// Conan::ImportStart
#include <stdint.h>

#include "x86_internal.h"
// Conan::ImportEnd

/*!
 * @ingroup group_x86
 * @file
 * @brief [en] x86 implementation of the broadcast scalar multiply.
 * @brief [zh] 按标量广播乘法的 x86 实现。
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
 * @brief [en] fp32 multiply by one broadcast factor, bitwise identical to the reference.
 * @brief [zh] 用一个广播因子的 fp32 乘法，与参考实现逐位相同。
 * @note [en] Same class as `add`: one IEEE multiply per element, so the verdict is bit equality.
 * @note [zh] 与 `add` 同类：每元素一次 IEEE 乘法，判据是位相等。
 * @since 0.1.0
 */
infr_status infr_x86_scale_f32(const void *const *inputs, void *const *outputs, uint32_t element_count,
                               const infr_quant *quant) {
    const float *in;
    float *out;
    float scalar;
    __m128 factor;
    uint32_t i = 0U;

    (void)quant;
    if (inputs == NULL || outputs == NULL || inputs[0] == NULL || inputs[1] == NULL || outputs[0] == NULL) {
        return INFR_ERR_INVALID_ARGUMENT;
    }
    in = (const float *)inputs[0];
    out = (float *)outputs[0];
    /* Read after the guard, never in an initialiser: the reference returns a status for a missing
     * factor, and a dereference here would crash on the same call. */
    scalar = ((const float *)inputs[1])[0];
    factor = _mm_set1_ps(scalar);

    for (; i + 4U <= element_count; i += 4U) {
        _mm_storeu_ps(&out[i], _mm_mul_ps(_mm_loadu_ps(&in[i]), factor));
    }
    for (; i < element_count; ++i) {
        out[i] = in[i] * scalar;
    }
    return INFR_OK;
}

#endif /* __SSE2__ */
