// Conan::ImportStart
#include <infinfer/kernel/kernel.h>
// Conan::ImportEnd

/*
 * Scalar quantize/dequantize: the reference definition of correctness (INV-2).
 *
 * Semantics pinned by the contract:
 *   quantize:   q = round(x / scale) + zero_point, clamped to the int8 range
 *   dequantize: x = (q - zero_point) * scale
 * Rounding is half away from zero; a request with INFR_QUANT_NONE or a
 * non-positive scale is rejected instead of writing garbage.
 *
 * For per-channel tensors the graph layer resolves the channel's parameters
 * into `scale`/`zero_point` and calls these once per slice (INV-3: no shape in
 * kernel signatures).
 *
 * 量化/反量化的 scalar 参考实现：正确性的唯一定义（INV-2）。
 */

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
