// Conan::ImportStart
#pragma once
#include <infinfer/kernel/kernel.h>
// Conan::ImportEnd

/*!
 * @ingroup group_scalar
 * @file
 * @brief [en] Internal contract between the scalar reference bodies and the registry: not public API.
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
 * @brief [en] Registry-shaped fp32-to-int8 quantize: `outputs[0]` gets `round(x/scale)+zero_point`.
 * @brief [zh] 注册表形状的 fp32→int8 量化：`outputs[0]` 得到 `round(x/scale)+zero_point`。
 * @note [en] The direction lives in the name, so one ABI covers both halves and the
 *            registry can hand the graph a quantize node without a second signature.
 * @note [zh] 方向写在名字里，因此一套 ABI 同时覆盖两个方向，注册表可以把量化节点作为
 *            普通内核交给图层，不必新增签名。
 * @since 0.1.0
 */
infr_status infr_scalar_quantize_f32(const void *const *inputs, void *const *outputs, uint32_t element_count,
                                     const infr_quant *quant);

/**
 * @brief [en] Registry-shaped int8-to-fp32 dequantize: `outputs[0]` gets `(q - zero_point) * scale`.
 * @brief [zh] 注册表形状的 int8→fp32 反量化：`outputs[0]` 得到 `(q - zero_point) * scale`。
 * @since 0.1.0
 */
infr_status infr_scalar_dequantize_i8(const void *const *inputs, void *const *outputs, uint32_t element_count,
                                      const infr_quant *quant);

/**
 * @brief [en] One dot product: activations against one weight row, `element_count` = inner dimension.
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
 * @brief [en] Multiply by the factor carried in `inputs[1][0]` (broadcast scalar).
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
