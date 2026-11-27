// Conan::ImportStart
#include "scalar_internal.h"
// Conan::ImportEnd

/*!
 * @ingroup group_scalar
 * @file
 * @brief [en] Scalar shift body: y[i] = x[i] + offset, offset read from `inputs[1][0]`.
 * @brief [zh] 标量平移体：y[i] = x[i] + offset，offset 取自 `inputs[1][0]`。
 * @since 0.2.0
 */

/* Why the offset rides in `inputs[1][0]` rather than in a function parameter: the kernel contract
 * is (inputs, outputs, length, quant) and has no slot for a scalar. `scale` already settled this
 * question the same way, and a single-element input tensor is the smallest thing that fits -- the
 * alternative is widening every tier's signature for one operator.
 *
 * Why a separate kernel instead of a fused act: subtraction by a constant is what numerically
 * stable softmax and layer normalisation both need, and neither has an activation to hang it on.
 * Keeping it here means the composed operators contain no arithmetic of their own.
 *
 * 为什么 offset 放在 `inputs[1][0]` 而不是函数参数：kernel 契约是
 * （inputs, outputs, length, quant），没有标量的位置。`scale` 已经用同样的方式解决了这个问题，
 * 单元素输入张量是能塞进既有契约的最小表示——另一条路是为了一个算子拓宽所有档位的签名。
 *
 * 为什么单独一个内核而不是融合进激活：减去一个常数，是数值稳定的 softmax 与 layer normalization
 * 都需要的一步，而两者都没有可挂靠的激活。放在这里，组合层就能不含任何自己的算术。
 */

/**
 * @brief [en] Unquantized fp32 shift over `element_count` floats.
 * @brief [zh] 对 `element_count` 个 float 做非量化 fp32 平移。
 * @note [en] The offset is read once, before the loop, so aliasing `inputs[1]` with part of
 *            `outputs[0]` still gets the value the caller passed in.
 * @note [zh] offset 在循环外只读一次，所以即使 `inputs[1]` 与 `outputs[0]` 部分重叠，读到的也是
 *            调用方传入那一刻的值。
 * @since 0.2.0
 */
infr_status infr_scalar_shift_f32(const void *const *inputs, void *const *outputs, uint32_t element_count,
                                  const infr_quant *quant) {
    const float *in;
    float *out;
    float offset;
    uint32_t i;

    (void)quant; /* Unquantized instance: parameters carry no meaning here. */
    if (inputs == NULL || outputs == NULL || inputs[0] == NULL || inputs[1] == NULL || outputs[0] == NULL) {
        return INFR_ERR_INVALID_ARGUMENT;
    }
    in = (const float *)inputs[0];
    out = (float *)outputs[0];
    offset = *(const float *)inputs[1];
    for (i = 0; i < element_count; ++i) {
        out[i] = in[i] + offset;
    }
    return INFR_OK;
}
