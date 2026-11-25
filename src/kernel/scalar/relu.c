// Conan::ImportStart
#include "scalar_internal.h"
// Conan::ImportEnd

/*!
 * @ingroup group_scalar
 * @file
 * @brief [en] Scalar relu bodies: the reference definition of correctness for the `relu` cell (INV-2).
 * @brief [zh] 标量 relu 体：`relu` 格的正确性定义（INV-2）。
 * @since 0.1.0
 */

/* Both instances follow the generic kernel signature from kernel.h: one input
 * buffer, one output buffer, an element count, no shape or stride.
 *
 * 两个实例都遵循 kernel.h 的通用内核签名：一个输入、一个输出、元素个数，没有
 * shape/stride。 */

/**
 * @brief [en] Unquantized fp32 relu over `element_count` floats.
 * @brief [zh] 对 `element_count` 个 float 做非量化 fp32 relu。
 * @since 0.1.0
 */
infr_status infr_scalar_relu_f32(const void *const *inputs, void *const *outputs, uint32_t element_count,
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
        out[i] = in[i] > 0.0F ? in[i] : 0.0F;
    }
    return INFR_OK;
}

infr_status infr_scalar_relu_i8(const void *const *inputs, void *const *outputs, uint32_t element_count,
                                const infr_quant *quant) {
    const int8_t *in;
    int8_t *out;
    uint32_t i;

    (void)quant;
    if (inputs == NULL || outputs == NULL || inputs[0] == NULL || outputs[0] == NULL) {
        return INFR_ERR_INVALID_ARGUMENT;
    }
    in = (const int8_t *)inputs[0];
    out = (int8_t *)outputs[0];
    for (i = 0; i < element_count; ++i) {
        out[i] = in[i] > 0 ? in[i] : 0;
    }
    return INFR_OK;
}
