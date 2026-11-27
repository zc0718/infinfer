// Conan::ImportStart
#include "scalar_internal.h"
// Conan::ImportEnd

/*!
 * @ingroup group_scalar
 * @file
 * @brief [en] Fused 3x3 depthwise convolution: one call for a whole pixel, not one call per channel.
 * @brief [zh] 融合的 3×3 深卷积：一次调用算完一个像素，而不是每通道一次调用。
 * @note [en] Why this kernel exists at all. A depthwise convolution used to be expressed
 *            as one `fc` per output element, so a 56x56x128 layer made 373k kernel calls
 *            of nine multiplications each, and the call cost dominated: measured 18.6 ns
 *            of fixed cost per call against 36.7 ns total, i.e. half the time was the
 *            call (gaps G-129). Batching the channels of one output pixel into a single
 *            call removes 127 of every 128 calls without needing a shape in the kernel:
 *            `element_count` is the number of output elements, and the inner dimension is
 *            fixed by the kernel identity, so the signature still carries no shape or
 *            stride.
 * @note [zh] 这个内核为什么存在。深卷积原来被表达成「每个输出元素一次 `fc`」，于是
 *            56×56×128 的一层要发 37.3 万次内核调用、每次只有 9 次乘加，调用成本成了
 *            大头：实测每次调用 18.6 ns 固定开销、而每次总共只有 36.7 ns，即一半时间花
 *            在调用上（gaps G-129）。把一个输出像素的所有通道批进一次调用，就能去掉
 *            128 次里的 127 次，而且内核里仍然不需要形状——`element_count` 是输出元素
 *            数，内维由内核身份固定——签名里仍然没有形状、没有 stride。
 * @since 0.2.0
 */

/**
 * @brief [en] y[r] = sum over the 9 taps of window[r*9 + j] * weights[r*9 + j].
 * @brief [zh] y[r] = window[r*9 + j] * weights[r*9 + j] 在 9 个抽头上的和。
 * @note [en] `inputs[0]` is the gathered window plane, row-major, 9 floats per output
 *            element; `inputs[1]` is the filter in the same layout, which is what the
 *            NHWC filter with one input channel per group already looks like. `quant` is
 *            unused: an unquantized instance carries no parameters. The nine products are
 *            written out instead of looped so the accumulation order is spelled in the
 *            source and the parity test can be read against it.
 * @note [zh] `inputs[0]` 是已经 gather 好的窗口平面，行主序、每个输出元素 9 个 float；
 *            `inputs[1]` 是同布局的滤波器——每分组一个输入通道的 NHWC 滤波器本来就是这个
 *            形状。`quant` 不使用：未量化实例不携带参数。九个乘积写开而不是循环，是为了
 *            让累加顺序直接写在源码里，parity 测试可以逐项对照。
 * @since 0.2.0
 */
infr_status infr_scalar_dwconv3x3_f32(const void *const *inputs, void *const *outputs, uint32_t element_count,
                                      const infr_quant *quant) {
    const float *window;
    const float *weights;
    float *y;
    uint32_t r;

    (void)quant; /* Unquantized instance: parameters carry no meaning here. */
    if (inputs == NULL || outputs == NULL || inputs[0] == NULL || inputs[1] == NULL || outputs[0] == NULL) {
        return INFR_ERR_INVALID_ARGUMENT;
    }
    window = (const float *)inputs[0];
    weights = (const float *)inputs[1];
    y = (float *)outputs[0];
    for (r = 0U; r < element_count; ++r) {
        const float *w9 = window + ((size_t)r * 9U);
        const float *k9 = weights + ((size_t)r * 9U);
        y[r] = ((((((((w9[0] * k9[0]) + (w9[1] * k9[1])) + (w9[2] * k9[2])) + (w9[3] * k9[3])) + (w9[4] * k9[4])) +
                  (w9[5] * k9[5])) +
                 (w9[6] * k9[6])) +
                (w9[7] * k9[7])) +
               (w9[8] * k9[8]);
    }
    return INFR_OK;
}
