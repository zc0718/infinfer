// Conan::ImportStart
#include <infinfer/kernel/kernel.h>
// Conan::ImportEnd

/*!
 * @ingroup group_scalar
 * @file
 * @brief [en] Scalar quantize/dequantize: the reference definition of correctness.
 * @brief [zh] 标量量化/反量化：正确性的参考定义。
 * @since 0.1.0
 */

/* Semantics pinned by the contract:
 *   quantize:   q = round(x / scale) + zero_point, clamped to the int8 range
 *   dequantize: x = (q - zero_point) * scale
 * Rounding is half away from zero; a request with INFR_QUANT_NONE or a
 * non-positive scale is rejected instead of writing garbage.
 *
 * For per-channel tensors the graph layer resolves the channel's parameters
 * into `scale`/`zero_point` and calls these once per slice.
 *
 * 语义由契约钉死：量化 `q = round(x/scale) + zero_point` 并截断到 int8 范围；
 * 反量化 `x = (q - zero_point) * scale`。舍入为“半远离零”；声明 INFR_QUANT_NONE
 * 或 scale 非正时直接拒绝，而不是写入垃圾值。逐通道张量由图层把该通道的参数
 * 解析到 `scale`/`zero_point` 后逐片调用。 */

static int quant_args_ok(const infr_quant *quant) {
    if (quant == NULL) {
        return 0;
    }
    if (quant->kind == INFR_QUANT_NONE) {
        return 0;
    }
    /* A NaN scale fails this comparison too, which is the intent. */
    if (!(quant->scale > 0.0F)) {
        return 0;
    }
    return 1;
}

static int8_t clamp_i8(int32_t value) {
    if (value < -128) {
        return (int8_t)-128;
    }
    if (value > 127) {
        return (int8_t)127;
    }
    return (int8_t)value;
}

/**
 * @brief [en] Quantize `n` floats to int8 with the given per-tensor parameters.
 * @brief [zh] 用给定逐张量参数把 `n` 个 float 量化为 int8。
 * @since 0.1.0
 */
infr_status infr_quantize_i8(const float *in, int8_t *out, uint32_t n, const infr_quant *quant) {
    uint32_t i;
    if (in == NULL || out == NULL || !quant_args_ok(quant)) {
        return INFR_ERR_INVALID_ARGUMENT;
    }
    for (i = 0; i < n; ++i) {
        float scaled = in[i] / quant->scale + (float)quant->zero_point;
        int32_t rounded = (int32_t)(scaled >= 0.0F ? scaled + 0.5F : scaled - 0.5F);
        out[i] = clamp_i8(rounded);
    }
    return INFR_OK;
}

/**
 * @brief [en] Dequantize `n` int8 values back to float with the given per-tensor parameters.
 * @brief [zh] 用给定逐张量参数把 `n` 个 int8 反量化回 float。
 * @since 0.1.0
 */
infr_status infr_dequantize_i8(const int8_t *in, float *out, uint32_t n, const infr_quant *quant) {
    uint32_t i;
    if (in == NULL || out == NULL || !quant_args_ok(quant)) {
        return INFR_ERR_INVALID_ARGUMENT;
    }
    for (i = 0; i < n; ++i) {
        out[i] = (float)((int32_t)in[i] - quant->zero_point) * quant->scale;
    }
    return INFR_OK;
}

/* Registry-shaped adapters. The public pair above takes typed pointers because
 * that is what a caller wants; the registry ABI is pointer arrays because that is
 * what every other body speaks. Keeping both means neither side has to lie about
 * its shape, and the arithmetic still lives in exactly one place.
 *
 * 注册表形状的适配层。上面的公开函数用类型化指针（调用者想要的就是这个），而注册表
 * ABI 用指针数组（其它所有体都是这个形状）。两者并存让两侧都不必伪装自己的形状，
 * 而算术仍然只有一处。
 */

/**
 * @brief [en] Registry-shaped quantize: reads `inputs[0]` as floats, writes `outputs[0]` as int8.
 * @brief [zh] 注册表形状的量化：`inputs[0]` 按 float 读，`outputs[0]` 按 int8 写。
 * @since 0.1.0
 */
infr_status infr_scalar_quantize_f32(const void *const *inputs, void *const *outputs, uint32_t element_count,
                                     const infr_quant *quant) {
    if (inputs == NULL || outputs == NULL || inputs[0] == NULL || outputs[0] == NULL) {
        return INFR_ERR_INVALID_ARGUMENT;
    }
    return infr_quantize_i8((const float *)inputs[0], (int8_t *)outputs[0], element_count, quant);
}

/**
 * @brief [en] Registry-shaped dequantize: reads `inputs[0]` as int8, writes `outputs[0]` as floats.
 * @brief [zh] 注册表形状的反量化：`inputs[0]` 按 int8 读，`outputs[0]` 按 float 写。
 * @since 0.1.0
 */
infr_status infr_scalar_dequantize_i8(const void *const *inputs, void *const *outputs, uint32_t element_count,
                                      const infr_quant *quant) {
    if (inputs == NULL || outputs == NULL || inputs[0] == NULL || outputs[0] == NULL) {
        return INFR_ERR_INVALID_ARGUMENT;
    }
    return infr_dequantize_i8((const int8_t *)inputs[0], (float *)outputs[0], element_count, quant);
}
