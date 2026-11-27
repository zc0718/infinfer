// Conan::ImportStart
#include <math.h>

#include "scalar_internal.h"
// Conan::ImportEnd

/*!
 * @ingroup group_scalar
 * @file
 * @brief [en] Scalar exp body: the reference definition of correctness for the `exp` cell.
 * @brief [zh] 标量 exp 体：`exp` 格的正确性定义。
 * @since 0.2.0
 */

/* Declared scalar-only on purpose. `expf` is a libm call, so a hand-written vector body would
 * either be the same call in a loop or a polynomial with its own accuracy contract; the second
 * is real work that belongs to a vendor backend (ADR-007), not to a reference cell. Saying that
 * here is cheaper than a vectorised body nobody measured.
 *
 * 刻意只声明 scalar 档。`expf` 是一次 libm 调用，手写向量体要么是同一个调用套循环、要么是
 * 自带精度契约的多项式；后者属于厂商后端（ADR-007），不是参考格该做的事。把这句话写在这里，
 * 比交一个没人量过的向量体便宜。
 */

/**
 * @brief [en] Unquantized fp32 exp over `element_count` floats.
 * @brief [zh] 对 `element_count` 个 float 做非量化 fp32 exp。
 * @note [en] In-place is safe: element `i` is read before it is written and no element depends on
 *            another.
 * @note [zh] 就地安全：第 `i` 个元素先读后写，且元素之间没有依赖。
 * @since 0.2.0
 */
infr_status infr_scalar_exp_f32(const void *const *inputs, void *const *outputs, uint32_t element_count,
                                const infr_quant *quant) {
    const float *in;
    float *out;
    uint32_t i;

    (void)quant; /* Unquantized instance: parameters carry no meaning here. */
    if (inputs == NULL || outputs == NULL || inputs[0] == NULL || outputs[0] == NULL) {
        return INFR_ERR_INVALID_ARGUMENT;
    }
    in = (const float *)inputs[0];
    out = (float *)outputs[0];
    for (i = 0; i < element_count; ++i) {
        out[i] = expf(in[i]);
    }
    return INFR_OK;
}
