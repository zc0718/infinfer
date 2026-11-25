// Conan::ImportStart
#include "scalar_internal.h"
// Conan::ImportEnd

/*!
 * @ingroup group_scalar
 * @file
 * @brief [en] Scalar elementwise binary add: the reference for the `add` cell (INV-2), used for bias.
 * @brief [zh] 标量逐元素二元加法：`add` 格的参考实现（INV-2），偏置相加用它。
 * @since 0.1.0
 */

/* Two inputs, one output, `element_count` elements; input and output may alias,
 * because the graph layer reuses the workspace in place for bias adds.
 *
 * 两个输入、一个输出、`element_count` 个元素；输入输出可同一缓冲区，因为图层
 * 做偏置相加时会就地复用工作区。 */

/**
 * @brief [en] y[i] = a[i] + b[i] over `element_count` floats.
 * @brief [zh] 对 `element_count` 个 float 计算 y[i] = a[i] + b[i]。
 * @since 0.1.0
 */
infr_status infr_scalar_add_f32(const void *const *inputs, void *const *outputs, uint32_t element_count,
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
        y[i] = a[i] + b[i];
    }
    return INFR_OK;
}
