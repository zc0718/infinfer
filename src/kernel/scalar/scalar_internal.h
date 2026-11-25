// Conan::ImportStart
#pragma once
#include <infinfer/kernel/kernel.h>
// Conan::ImportEnd

/**
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
 * @brief [en] One dot product: activations against one weight row, `element_count` = inner dimension.
 * @brief [zh] 一次点积：激活对一行权重，`element_count` 为内维长度。
 * @since 0.1.0
 */
infr_status infr_scalar_fc_f32(const void *const *inputs, void *const *outputs, uint32_t element_count,
                               const infr_quant *quant);

/* The fc contract in one line: `inputs[0]` against `inputs[1]`, one float into
 * `outputs[0]`; the graph layer loops over rows, because INV-3 keeps shape out
 * of this signature.
 *
 * fc 的契约一句话：`inputs[0]` 对 `inputs[1]`，一个 float 写进 `outputs[0]`；
 * 逐行循环在图层，因为 INV-3 把 shape 挡在签名之外。 */
