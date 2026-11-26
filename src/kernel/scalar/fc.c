// Conan::ImportStart
#include "scalar_internal.h"
// Conan::ImportEnd

/*!
 * @ingroup group_scalar
 * @file
 * @brief [en] Scalar fully-connected body: one dot product per call, the reference for the `fc` cell.
 * @brief [zh] 标量全连接体：一次调用一次点积，`fc` 格的正确性定义。
 * @since 0.1.0
 */

/* What this body may see is fixed: two buffers and an element count, never a
 * shape. So one call produces **one** output element -- the dot product of the
 * input vector with one weight row -- and the graph layer (L2) loops over rows.
 * Kernel call count is an L2 decision; keeping shape out of the signature is what
 * lets the same body serve a 16-wide MLP row and a 512-wide one unchanged.
 *
 * Bias and activation are not folded in: they are separate cells the graph layer
 * sequences (relu already exists; add arrives with the first model that needs
 * it). A fused variant would have to duplicate both definitions.
 *
 * 本体只能看到两个缓冲与一个元素个数、绝无 shape：因此一次调用**只
 * 产出一个**输出元素（输入向量与一行权重的点积），逐行循环属于图层（L2）的职责。
 * 把 shape 挡在签名外，才能让同一个本体既服务 16 宽的行、也服务 512 宽的行。
 *
 * 偏置与激活不融合：它们是图层负责排序的独立格子（relu 已有；add 随第一个需要
 * 它的模型一起落地）。融合版本等于把两个定义各复制一份。 */

/**
 * @brief [en] Dot product of `inputs[0]` and `inputs[1]`; writes exactly one float to `outputs[0]`.
 * @brief [zh] 计算 `inputs[0]` 与 `inputs[1]` 的点积，向 `outputs[0]` 写恰好一个 float。
 * @since 0.1.0
 */
infr_status infr_scalar_fc_f32(const void *const *inputs, void *const *outputs, uint32_t element_count,
                               const infr_quant *quant) {
    const float *x;
    const float *w;
    float *out;
    float acc = 0.0F;
    uint32_t i;

    (void)quant; /* Unquantized instance: parameters carry no meaning here. */
    if (inputs == NULL || outputs == NULL || inputs[0] == NULL || inputs[1] == NULL || outputs[0] == NULL) {
        return INFR_ERR_INVALID_ARGUMENT;
    }
    out = (float *)outputs[0];
    if (element_count == 0) {
        /* An empty inner dimension is a legal zero-length reduction, not an
         * error: the graph layer may hand us a zero-width row while probing. */
        out[0] = 0.0F;
        return INFR_OK;
    }
    x = (const float *)inputs[0];
    w = (const float *)inputs[1];
    /* Fixed accumulation order (i ascending): the reference is defined by this
     * order, and every other tier has to reproduce it. */
    for (i = 0; i < element_count; ++i) {
        acc += x[i] * w[i];
    }
    out[0] = acc;
    return INFR_OK;
}
