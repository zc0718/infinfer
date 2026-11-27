// Conan::ImportStart
#pragma once
#include <infinfer/kernel/kernel.h>
// Conan::ImportEnd

/*!
 * @ingroup group_scalar
 * @file
 * @brief [en] Internal contract between the scalar reference bodies and the
 * registry: not public API.
 * @brief [zh] scalar 参考体与注册表之间的内部契约，不是公开 API。
 * @since 0.1.0
 */

/* Consumers reach these symbols only through infr_kernel_register_scalar() +
 * infr_kernel_find(); nothing outside src/kernel/scalar links against them.
 *
 * 外部消费者只能经由 infr_kernel_register_scalar() + infr_kernel_find() 访问
 * 这些符号；src/kernel/scalar 之外没有任何东西链接它们。 */

/**
 * @brief [en] Scalar fp32 relu over a buffer of `element_count` floats.
 * @brief [zh] 对 `element_count` 个 float 的缓冲做标量 fp32 relu。
 * @since 0.1.0
 */
infr_status infr_scalar_relu_f32(const void *const *inputs, void *const *outputs, uint32_t element_count,
                                 const infr_quant *quant);

/**
 * @brief [en] Scalar int8 relu over a buffer of `element_count` bytes.
 * @brief [zh] 对 `element_count` 字节的缓冲做标量 int8 relu。
 * @since 0.1.0
 */
infr_status infr_scalar_relu_i8(const void *const *inputs, void *const *outputs, uint32_t element_count,
                                const infr_quant *quant);

/**
 * @brief [en] Registry-shaped fp32-to-int8 quantize: `outputs[0]` gets
 * `round(x/scale)+zero_point`.
 * @brief [zh] 注册表形状的 fp32→int8 量化：`outputs[0]` 得到
 * `round(x/scale)+zero_point`。
 * @note [en] The direction lives in the name, so one ABI covers both halves and
 * the registry can hand the graph a quantize node without a second signature.
 * @note [zh] 方向写在名字里，因此一套 ABI
 * 同时覆盖两个方向，注册表可以把量化节点作为 普通内核交给图层，不必新增签名。
 * @since 0.1.0
 */
infr_status infr_scalar_quantize_f32(const void *const *inputs, void *const *outputs, uint32_t element_count,
                                     const infr_quant *quant);

/**
 * @brief [en] Registry-shaped int8-to-fp32 dequantize: `outputs[0]` gets `(q -
 * zero_point) * scale`.
 * @brief [zh] 注册表形状的 int8→fp32 反量化：`outputs[0]` 得到 `(q -
 * zero_point) * scale`。
 * @since 0.1.0
 */
infr_status infr_scalar_dequantize_i8(const void *const *inputs, void *const *outputs, uint32_t element_count,
                                      const infr_quant *quant);

/**
 * @brief [en] Clamp an int32 to the int8 range: the shared tail of every
 * quantizing body.
 * @brief [zh] 把 int32 截断到 int8 范围：每个量化体的共用尾部。
 * @since 0.1.0
 */
int8_t infr_scalar_clamp_i8(int32_t value);

/**
 * @brief [en] Requantize an int64 accumulator to int16: shared by the scalar
 * and DSP int16 bodies.
 * @brief [zh] 把 int64 累加器再量化为 int16：scalar 与 DSP 两个 int16 体共用。
 * @note [en] Sharing the whole tail (not just a clamp) is what makes
 * "bit-identical across tiers" a property of the design: the rounding and
 * saturation have exactly one definition.
 * @note [zh] 共享整个尾部（而不只是一个
 * clamp）才让“跨档位逐位相同”成为设计属性：舍入与饱和 只有一处定义。
 * @since 0.1.0
 */
int16_t infr_scalar_requantize_i16(int64_t acc, const infr_quant *quant);

/**
 * @brief [en] One int8 dot product: zero-centred operands accumulated in int32,
 * requantized to int8.
 * @brief [zh] 一次 int8 点积：零中心化的操作数以 int32 累加，再再量化回 int8。
 * @note [en] Requantization reads `quant->requant_multiplier` and
 * `quant->zero_point`; the multiplier is the already-resolved `(scale_a *
 * scale_b) / scale_out`, so the body never divides. Asymmetric activation
 * offsets are not modelled here: the graph layer routes them through the
 * quantize/dequantize kernels instead.
 * @note [zh] 再量化读 `quant->requant_multiplier` 与
 * `quant->zero_point`；乘子是已解析的
 *            `(scale_a * scale_b) /
 * scale_out`，因此本体不做除法。这里不建模非对称激活偏移： 图层把它们走
 * quantize/dequantize 内核。
 * @since 0.1.0
 */
infr_status infr_scalar_fc_i8(const void *const *inputs, void *const *outputs, uint32_t element_count,
                              const infr_quant *quant);

/**
 * @brief [en] One int16 dot product: operands accumulated in int64, requantized
 * and saturated to int16.
 * @brief [zh] 一次 int16 点积：操作数以 int64 累加，再量化并饱和回 int16。
 * @note [en] The accumulator is int64 because int16 products are not small: one
 * term can be about 2^30, so int32 holds a single worst-case term, loses
 * precision on the second and **overflows on the third** -- a 512-wide row,
 * which is an ordinary inner dimension, overshoots the int32 range by about
 * 256x. int32 would not be "tighter", it would be wrong.
 * @note [zh] 累加器是 int64，因为 int16 乘积并不小：单项可达 2^30，因此 int32
 * 只能稳稳装下
 *            **一项**最坏情况，第二项已丢精度，**第三项就溢出**——512
 * 宽的行（正常内维）会 超出 int32 上限约 256 倍。int32
 * 不是“精度紧一点”，而是会算错。
 * @since 0.1.0
 */
infr_status infr_scalar_fc_i16(const void *const *inputs, void *const *outputs, uint32_t element_count,
                               const infr_quant *quant);

/**
 * @brief [en] One dot product: activations against one weight row,
 * `element_count` = inner dimension.
 * @brief [zh] 一次点积：激活对一行权重，`element_count` 为内维长度。
 * @since 0.1.0
 */
infr_status infr_scalar_fc_f32(const void *const *inputs, void *const *outputs, uint32_t element_count,
                               const infr_quant *quant);

/**
 * @brief [en] Elementwise binary add over `element_count` floats (bias add).
 * @brief [zh] 对 `element_count` 个 float 做逐元素二元加法（偏置相加）。
 * @since 0.1.0
 */
infr_status infr_scalar_add_f32(const void *const *inputs, void *const *outputs, uint32_t element_count,
                                const infr_quant *quant);

/**
 * @brief [en] Elementwise binary multiply over `element_count` floats.
 * @brief [zh] 对 `element_count` 个 float 做逐元素二元乘法。
 * @since 0.1.0
 */
infr_status infr_scalar_mul_f32(const void *const *inputs, void *const *outputs, uint32_t element_count,
                                const infr_quant *quant);

/**
 * @brief [en] Sum of `element_count` floats written to `outputs[0]`.
 * @brief [zh] 把 `element_count` 个 float 求和写入 `outputs[0]`。
 * @since 0.1.0
 */
infr_status infr_scalar_sum_f32(const void *const *inputs, void *const *outputs, uint32_t element_count,
                                const infr_quant *quant);

/**
 * @brief [en] Maximum of `element_count` floats written to `outputs[0]`.
 * @brief [zh] 把 `element_count` 个 float 的最大值写入 `outputs[0]`。
 * @since 0.1.0
 */
infr_status infr_scalar_max_f32(const void *const *inputs, void *const *outputs, uint32_t element_count,
                                const infr_quant *quant);

/**
 * @brief [en] Multiply by the factor carried in `inputs[1][0]` (broadcast
 * scalar).
 * @brief [zh] 乘以 `inputs[1][0]` 承载的因子（广播标量）。
 * @since 0.1.0
 */
infr_status infr_scalar_scale_f32(const void *const *inputs, void *const *outputs, uint32_t element_count,
                                  const infr_quant *quant);

/* The fc contract in one line: `inputs[0]` against `inputs[1]`, one float into
 * `outputs[0]`; the graph layer loops over rows, because shape stays outside
 * of this signature.
 *
 * fc 的契约一句话：`inputs[0]` 对 `inputs[1]`，一个 float 写进 `outputs[0]`；
 * 逐行循环在图层，因为 shape 被挡在签名之外。 */

/**
 * @brief [en] int8 dot product that stops at the int32 accumulator; `quant` is
 * ignored (NULL is legal).
 * @brief [zh] 停在 int32 累加器上的 int8 点积；`quant` 被忽略（NULL 合法）。
 * @note [en] A cell of its own so the graph layer can add a bias *before*
 * requantizing; see the file comment in dot.c for why a flag on `fc` would not
 * do.
 * @note [zh] 单独成一格，是为了让图层能在再量化**之前**加偏置；为什么给 `fc`
 * 加开关不行， 见 dot.c 的文件注释。
 * @since 0.1.0
 */
infr_status infr_scalar_dot_i8(const void *const *inputs, void *const *outputs, uint32_t element_count,
                               const infr_quant *quant);

/**
 * @brief [en] Requantize one int32 accumulator to int8: the shared tail of
 * every 8-bit requantizing body.
 * @brief [zh] 把一个 int32 累加器再量化为 int8：每个 8 位再量化体的共用尾部。
 * @since 0.1.0
 */
int8_t infr_scalar_requantize_i32(int32_t acc, const infr_quant *quant);

/**
 * @brief [en] Bias add plus requantization over `n` accumulators; `inputs[1]`
 * is an optional int32 bias.
 * @brief [zh] 对 `n` 个累加器做偏置相加与再量化；`inputs[1]` 是可选的 int32
 * 偏置。
 * @since 0.1.0
 */
infr_status infr_scalar_requant_i32(const void *const *inputs, void *const *outputs, uint32_t element_count,
                                    const infr_quant *quant);
