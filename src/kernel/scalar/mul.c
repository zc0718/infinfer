// Conan::ImportStart
#include "scalar_internal.h"
// Conan::ImportEnd

/*!
 * @ingroup group_scalar
 * @file
 * @brief [en] Scalar elementwise multiply: the reference for the `mul` cell.
 * @brief [zh] 标量逐元素乘法：`mul` 格的参考实现。
 * @since 0.1.0
 */

/* Two inputs, one output, `element_count` elements; input and output may alias.
 * `mul` is the first kernel MobileNetV3's squeeze-and-excite block cannot be built
 * without, and it is a kernel rather than a graph trick for the same reason `add` is:
 * the tier rows get to speed it up without the graph knowing.
 *
 * 两个输入、一个输出、`element_count` 个元素；输入输出可同一缓冲区。`mul` 是
 * MobileNetV3 的 squeeze-and-excite 块**绕不过去**的第一个内核，而它做成 kernel 而不是
 * 图层的小把戏，理由与 `add` 相同：各档可以各自加速，图层不必知道。 */

/**
 * @brief [en] y[i] = a[i] * b[i] over `element_count` floats.
 * @brief [zh] 对 `element_count` 个 float 计算 y[i] = a[i] * b[i]。
 * @since 0.1.0
 */
infr_status infr_scalar_mul_f32(const void *const *inputs, void *const *outputs, uint32_t element_count,
                                const infr_quant *quant) {
    const float *a;
    const float *b;
    float *y;
    uint32_t i;

    (void)quant; /* Unquantized instance: parameters carry no meaning here. */
    if (inputs == NULL || outputs == NULL || inputs[0] == NULL || inputs[1] == NULL || outputs[0] == NULL) {
        return INFR_ERR_INVALID_ARGUMENT;
    }
    a = (const float *)inputs[0];
    b = (const float *)inputs[1];
    y = (float *)outputs[0];
    for (i = 0; i < element_count; ++i) {
        y[i] = a[i] * b[i];
    }
    return INFR_OK;
}
