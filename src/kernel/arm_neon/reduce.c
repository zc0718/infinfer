// Conan::ImportStart
#include <stdint.h>

#include "arm_neon_internal.h"
// Conan::ImportEnd

/*!
 * @ingroup group_arm_neon
 * @file
 * @brief [en] NEON implementation of the fp32 max reduction.
 * @brief [zh] fp32 max 归约的 NEON 实现。
 * @since 0.1.0
 */

#if defined(__ARM_NEON)
#include <arm_neon.h>

/**
 * @brief [en] fp32 maximum over four lanes, then a lane fold and a scalar tail.
 * @brief [zh] 四车道 fp32 最大值，随后折叠车道并处理标量余数。
 * @note [en] The reference keeps the **first** occurrence of a maximum (`>` and not `>=`), so the
 *            result cannot depend on where the scan happens to start. This tier reproduces that rule
 *            twice: lane-wise with `vcgtq_f32` + `vbslq_f32` (which also declines a NaN, like the
 *            reference), then across the four lanes in index order. Equal maxima that differ only in
 *            the sign of zero are the one input class where "first" means "first in lane order"
 *            rather than "first in linear order": `-0.0` and `+0.0` compare equal, so both bodies
 *            return one of them and the difference is not observable as a number. Recorded in gaps
 *            G-58 rather than papered over.
 * @note [zh] 参考实现保留最大值的**首次出现**（用 `>` 而非 `>=`），因此结果不会取决于扫描从哪开始。
 *            本档位把这条规则复现两次：车道内用 `vcgtq_f32` + `vbslq_f32`（NaN 同样不被选中），
 *            再按车道索引顺序折叠四个车道。只有"仅相差零的符号"的相等最大值属于例外：那里"首次"
 *            指的是"车道序首次"而不是"线性序首次"——`-0.0` 与 `+0.0` 相等，两个体各返回其中一个，
 *            差异不作为数值可观察。这一点记进 gaps G-58，而不是抹平。
 * @note [en] A zero-length reduction is an error here, exactly as in the reference: unlike the
 *            dot product, "the maximum of nothing" has no value to report, so it is refused rather
 *            than invented.
 * @note [zh] 零长度归约在这里是错误，与参考实现一致：与点积不同，"空集的最大值"没有可报告的值，
 *            因此拒绝而不是编一个。
 * @since 0.1.0
 */
infr_status infr_arm_neon_max_f32(const void *const *inputs, void *const *outputs, uint32_t element_count,
                                  const infr_quant *quant) {
    const float *in;
    float32x4_t acc;
    float32x4_t v;
    float lanes[4];
    float best;
    uint32_t i;
    uint32_t k;

    (void)quant;
    if (inputs == NULL || outputs == NULL || inputs[0] == NULL || outputs[0] == NULL) {
        return INFR_ERR_INVALID_ARGUMENT;
    }
    if (element_count == 0U) {
        return INFR_ERR_INVALID_ARGUMENT;
    }
    in = (const float *)inputs[0];

    best = in[0];
    i = 1U;
    /* `acc` starts as the first element replicated in all four lanes; the loop then folds the
     * elements from index 1 on. Each lane therefore keeps the earliest winner of its own
     * subsequence, and the lane fold below keeps the earliest lane on a tie. */
    acc = vdupq_n_f32(best);
    for (; i + 4U <= element_count; i += 4U) {
        v = vld1q_f32(&in[i]);
        acc = vbslq_f32(vcgtq_f32(v, acc), v, acc);
    }
    vst1q_f32(lanes, acc);
    for (k = 0U; k < 4U; ++k) {
        if (lanes[k] > best) {
            best = lanes[k];
        }
    }
    for (; i < element_count; ++i) {
        if (in[i] > best) {
            best = in[i];
        }
    }
    ((float *)outputs[0])[0] = best;
    return INFR_OK;
}

#endif /* __ARM_NEON */
