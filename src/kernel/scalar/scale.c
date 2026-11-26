// Conan::ImportStart
#include "scalar_internal.h"
// Conan::ImportEnd

/*!
 * @ingroup group_scalar
 * @file
 * @brief [en] Scalar scaling: multiply a vector by one factor carried in a 1-element buffer.
 * @brief [zh] 标量缩放：把向量乘以一个由单元素缓冲区承载的因子。
 * @since 0.1.0
 */

/* Why the factor rides in `inputs[1][0]` instead of a function parameter: the kernel
 * contract is (inputs, outputs, length, quant) with no room for a scalar, and widening
 * it for one operator would change every tier's signature. A one-element input tensor
 * is the smallest thing that fits the existing contract, and it composes with the same
 * pointer-length rule everything else uses.
 *
 * 为什么因子放在 `inputs[1][0]` 而不是函数参数：kernel 契约是
 * （inputs, outputs, length, quant），没有标量的位置；为了一个算子去拓宽它，等于改
 * 所有档位的签名。单元素的输入张量是能塞进既有契约的最小表示，且与其它 kernel 一样
 * 遵守"指针 + 长度"这条规则。 */

/**
 * @brief [en] y[i] = a[i] * factor, where factor is `inputs[1][0]`.
 * @brief [zh] y[i] = a[i] * factor，其中 factor 取自 `inputs[1][0]`。
 * @note [en] The factor is read once, before the loop: a caller that aliases
 *            `inputs[1]` with part of `outputs[0]` still gets the value it passed in,
 *            not a value already overwritten by this call.
 * @note [zh] 因子在循环外只读一次：调用方即使让 `inputs[1]` 与 `outputs[0]` 部分重叠，
 *            读到的也是传入那一刻的值，而不是被本次调用覆盖后的值。
 * @since 0.1.0
 */
infr_status infr_scalar_scale_f32(const void *const *inputs, void *const *outputs, uint32_t element_count,
                                  const infr_quant *quant) {
    const float *in;
    float *out;
    float factor;
    uint32_t i;

    (void)quant;
    if (inputs == NULL || outputs == NULL || inputs[0] == NULL || inputs[1] == NULL || outputs[0] == NULL) {
        return INFR_ERR_INVALID_ARGUMENT;
    }
    in = (const float *)inputs[0];
    out = (float *)outputs[0];
    factor = ((const float *)inputs[1])[0];

    for (i = 0; i < element_count; ++i) {
        out[i] = in[i] * factor;
    }
    return INFR_OK;
}
