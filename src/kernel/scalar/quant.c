// Conan::ImportStart
#include <infinfer/kernel/kernel.h>
// Conan::ImportEnd

/*!
 * @ingroup group_scalar
 * @file
 * @brief [en] Scalar quantize/dequantize: the reference definition of
 * correctness.
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

/**
 * @brief [en] Clamp an int32 to the int8 range, saturating at both ends.
 * @brief [zh] 把 int32 截断到 int8 范围，两端饱和。
 * @since 0.1.0
 */
int8_t infr_scalar_clamp_i8(int32_t value) {
    if (value < -128) {
        return (int8_t)-128;
    }
    if (value > 127) {
        return (int8_t)127;
    }
    return (int8_t)value;
}

/**
 * @brief [en] Turn an int64 dot-product accumulator into one int16: multiply by
 * the resolved multiplier, add the zero point, round half away from zero,
 * saturate.
 * @brief [zh] 把 int64 点积累加器变成一个 int16：乘已解析乘子、加 zero
 * point、半远离零舍入、饱和。
 * @note [en] One definition for both int16 tiers on purpose: the DSP body calls
 * this too, so the two cannot drift into "bit-identical except on the value
 * that rounds at a boundary".
 * @note [zh] 两个 int16 档位共用这一处定义是刻意的：DSP
 * 体也调它，两者就不可能漂移成 “除了边界舍入那个值之外逐位相同”。
 * @note [en] Saturation happens in the double domain *before* the conversion,
 * unlike the int8 tail: an int64 accumulator times any multiplier can exceed
 * every integer type, and a float-to-int conversion of an out-of-range value is
 * undefined behaviour, not a clamp.
 * @note [zh] 与 int8 尾部不同，饱和在 double 域内、**转换之前**完成：int64
 * 累加器乘任何乘子都
 *            可能超出所有整数类型，而越界的浮点→整数转换是未定义行为，不会自己截断。
 * @since 0.1.0
 */
int16_t infr_scalar_requantize_i16(int64_t acc, const infr_quant *quant) {
    double scaled = ((double)acc * (double)quant->requant_multiplier) + (double)quant->zero_point;
    if (scaled <= (double)INT16_MIN) {
        return (int16_t)INT16_MIN;
    }
    if (scaled >= (double)INT16_MAX) {
        return (int16_t)INT16_MAX;
    }
    return (int16_t)(scaled >= 0.0 ? scaled + 0.5 : scaled - 0.5);
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
        out[i] = infr_scalar_clamp_i8(rounded);
    }
    return INFR_OK;
}

/**
 * @brief [en] Dequantize `n` int8 values back to float with the given
 * per-tensor parameters.
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
 * that is what a caller wants; the registry ABI is pointer arrays because that
 * is what every other body speaks. Keeping both means neither side has to lie
 * about its shape, and the arithmetic still lives in exactly one place.
 *
 * 注册表形状的适配层。上面的公开函数用类型化指针（调用者想要的就是这个），而注册表
 * ABI
 * 用指针数组（其它所有体都是这个形状）。两者并存让两侧都不必伪装自己的形状，
 * 而算术仍然只有一处。
 */

/**
 * @brief [en] Registry-shaped quantize: reads `inputs[0]` as floats, writes
 * `outputs[0]` as int8.
 * @brief [zh] 注册表形状的量化：`inputs[0]` 按 float 读，`outputs[0]` 按 int8
 * 写。
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
 * @brief [en] Registry-shaped dequantize: reads `inputs[0]` as int8, writes
 * `outputs[0]` as floats.
 * @brief [zh] 注册表形状的反量化：`inputs[0]` 按 int8 读，`outputs[0]` 按 float
 * 写。
 * @since 0.1.0
 */
infr_status infr_scalar_dequantize_i8(const void *const *inputs, void *const *outputs, uint32_t element_count,
                                      const infr_quant *quant) {
    if (inputs == NULL || outputs == NULL || inputs[0] == NULL || outputs[0] == NULL) {
        return INFR_ERR_INVALID_ARGUMENT;
    }
    return infr_dequantize_i8((const int8_t *)inputs[0], (float *)outputs[0], element_count, quant);
}

/**
 * @brief [en] Turn an int32 accumulator into one int8: multiply by the resolved
 * multiplier, add the zero point, round half away from zero, saturate.
 * @brief [zh] 把 int32 累加器变成一个 int8：乘已解析乘子、加 zero
 * point、半远离零舍入、饱和。
 * @note [en] Shared by the scalar body and by any tier that fuses the tail, for
 * the same reason as `infr_scalar_requantize_i16()`: a second copy of "round
 * and clamp" is a second definition, and the tiers would drift exactly on the
 * boundary values.
 * @note [zh] scalar 体与任何融合尾部的档位共用，理由与
 * `infr_scalar_requantize_i16()` 相同：
 *            "舍入加截断"的第二份拷贝就是第二个定义，档位会恰好在边界值上漂移。
 * @note [en] Saturation happens in the double domain before the integer
 * conversion, because an int32 accumulator times any multiplier can leave every
 * 8-bit range and an out-of-range float-to-integer conversion is undefined
 * behaviour, not a clamp.
 * @note [zh] 饱和在 double 域内、整数转换之前完成：int32
 * 累加器乘任何乘子都可能离开 8 位范围，
 *            而越界的浮点→整数转换是未定义行为，不会自己截断。
 * @since 0.1.0
 */
int8_t infr_scalar_requantize_i32(int32_t acc, const infr_quant *quant) {
    const double scaled = ((double)acc * (double)quant->requant_multiplier) + (double)quant->zero_point;
    if (scaled <= (double)INT8_MIN) {
        return (int8_t)INT8_MIN;
    }
    if (scaled >= (double)INT8_MAX) {
        return (int8_t)INT8_MAX;
    }
    return (int8_t)(scaled >= 0.0 ? scaled + 0.5 : scaled - 0.5);
}

/**
 * @brief [en] Registry-shaped bias add plus requantization: `n` int32
 * accumulators to `n` int8.
 * @brief [zh] 注册表形状的偏置相加加再量化：`n` 个 int32 累加器变成 `n` 个
 * int8。
 * @note [en] `inputs[1]` is an optional int32 bias. It is folded into this cell
 * rather than given a row of its own because the accumulator's only producer is
 * a dot product: a separate int32 add row would exist purely to be called
 * between two other cells, and it would have to be re-summed with the
 * requantization anyway.
 * @note [zh] `inputs[1]` 是可选的 int32
 * 偏置。它被折进本格、而不是单独成行，因为累加器唯一的 来源就是点积：单独一个
 * int32 加法行存在的唯一用途就是被夹在另外两格之间调用，
 *            而且最终还得与再量化合并成一次运算。
 * @since 0.1.0
 */
infr_status infr_scalar_requant_i32(const void *const *inputs, void *const *outputs, uint32_t element_count,
                                    const infr_quant *quant) {
    const int32_t *src;
    const int32_t *bias;
    int8_t *dst;
    uint32_t i;

    if (inputs == NULL || outputs == NULL || inputs[0] == NULL || outputs[0] == NULL) {
        return INFR_ERR_INVALID_ARGUMENT;
    }
    if (quant == NULL || quant->kind == INFR_QUANT_NONE || !(quant->requant_multiplier > 0.0F)) {
        return INFR_ERR_INVALID_ARGUMENT;
    }
    src = (const int32_t *)inputs[0];
    bias = (const int32_t *)inputs[1];
    dst = (int8_t *)outputs[0];
    for (i = 0; i < element_count; ++i) {
        const int32_t acc = src[i] + (bias != NULL ? bias[i] : 0);
        dst[i] = infr_scalar_requantize_i32(acc, quant);
    }
    return INFR_OK;
}
