// Conan::ImportStart
#include "scalar_internal.h"
// Conan::ImportEnd

/*!
 * @ingroup group_scalar
 * @file
 * @brief [en] Scalar int8 dot product that stops at the int32 accumulator.
 * @brief [zh] 标量 int8 点积，停在 int32 累加器上。
 * @since 0.1.0
 */

/* Why this is a cell of its own instead of a flag on `fc`:
 *
 * A quantized convolution has to add its bias to the int32 accumulator and
 * requantize afterwards -- that is the order that keeps the bias from being
 * rounded away before it is used. `fc`'s int8 body requantizes inside the same
 * call, so a graph built out of `fc` alone cannot express a convolution with a
 * bias. Splitting the accumulator out gives the graph layer exactly one place
 * to insert the bias, and leaves one definition of rounding (in `requant`)
 * rather than two.
 *
 * The cost is honest and worth stating: the two cells mean two indirect calls
 * per output element where `fc` used one. That is the same call structure the
 * fp32 convolution already has (one `fc` per output channel per pixel), so the
 * quantized path is not worse than the float one it replaces.
 *
 * 为什么单独成一格，而不是给 `fc` 加一个开关：
 *
 * 量化卷积必须把偏置加到 int32 累加器上、**之后**再再量化——只有这个顺序，
 * 偏置才不会在用到之前就被舍入掉。`fc` 的 int8 体在同一次调用里就再量化了，
 * 所以只由 `fc` 拼出来的图**表达不了带偏置的卷积**。把累加器单独拿出来，图层
 * 就有了唯一一个插偏置的位置，而舍入的定义只剩一处（在 `requant` 里）。
 *
 * 代价要说清楚：两个格意味着每个输出元素两次间接调用，而 `fc` 是一次。这与
 * fp32 卷积已有的调用结构相同（每个输出通道、每个像素一次 `fc`），所以量化路径
 * 并不比它所替代的浮点路径更差。 */

/**
 * @brief [en] `outputs[0]` gets `inputs[0] · inputs[1]`, both int8, accumulated
 * in int32.
 * @brief [zh] `outputs[0]` 得到 `inputs[0] · inputs[1]`，两侧均为 int8，以
 * int32 累加。
 * @param inputs [en] `inputs[0]` activations, `inputs[1]` one weight row;
 * neither may be NULL.
 * @param inputs [zh] `inputs[0]` 为激活、`inputs[1]` 为一行权重；都不得为
 * NULL。
 * @param outputs [en] `outputs[0]` receives one int32 accumulator.
 * @param outputs [zh] `outputs[0]` 接收一个 int32 累加器。
 * @param element_count [en] Inner dimension: how many products to accumulate.
 * @param element_count [zh] 内维长度：累加多少个乘积。
 * @param quant [en] Unused: this cell does not requantize. NULL is accepted on
 * purpose, so a caller that only wants the accumulator does not have to invent
 * parameters.
 * @param quant [zh] 不使用：本格不做再量化。**故意**接受
 * NULL，只想要累加器的调用方 不必为此编造参数。
 * @note [en] The accumulation order is `i` ascending, the same rule the fp32
 * and int8 `fc` bodies follow, so a candidate tier has one unambiguous order to
 * match.
 * @note [zh] 累加顺序为 `i` 递增，与 fp32 和 int8 的 `fc`
 * 体同一规则，候选档位只有一种 无歧义的顺序可对齐。
 * @note [en] The range is the caller's to keep: a row of `n` products each at
 * most `127 * 127` stays inside int32 up to `n = 133144` (`2147483647 /
 * 16129`). A convolution inner dimension that large is not represented in this
 * library (rank-4 NHWC, batch 1), and signed overflow would be undefined
 * behaviour rather than a documented wrap, so it is not handled here.
 * @note [zh] 范围由调用方保证：一行 `n` 个乘积、每个至多 `127 * 127` 时，`n =
 * 133144` （`2147483647 / 16129`）以内都装得下
 * int32。本库不表达这么大的卷积内维 （rank-4 NHWC、batch
 * 1），而带符号溢出是未定义行为、不是有文档的回绕，因此这里不处理。
 * @since 0.1.0
 */
infr_status infr_scalar_dot_i8(const void *const *inputs, void *const *outputs, uint32_t element_count,
                               const infr_quant *quant) {
    const int8_t *x;
    const int8_t *w;
    int32_t acc = 0;
    uint32_t i;

    (void)quant;
    if (inputs == NULL || outputs == NULL || inputs[0] == NULL || inputs[1] == NULL || outputs[0] == NULL) {
        return INFR_ERR_INVALID_ARGUMENT;
    }
    x = (const int8_t *)inputs[0];
    w = (const int8_t *)inputs[1];
    for (i = 0; i < element_count; ++i) {
        acc += (int32_t)x[i] * (int32_t)w[i];
    }
    *(int32_t *)outputs[0] = acc;
    return INFR_OK;
}
