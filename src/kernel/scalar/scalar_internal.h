// Conan::ImportStart
#pragma once
#include <infinfer/kernel/kernel.h>
// Conan::ImportEnd

/*
 * Internal contract between the scalar reference bodies and register.c.
 * These symbols are not public API: consumers reach them through
 * infr_kernel_register_scalar() + infr_kernel_find() only.
 *
 * scalar 参考体与 register.c 之间的内部契约；这些符号不是公开 API，消费者只能
 * 经由 infr_kernel_register_scalar() + infr_kernel_find() 访问。
 */

infr_status infr_scalar_relu_f32(const void *const *inputs, void *const *outputs, uint32_t element_count,
                                 const infr_quant *quant);

infr_status infr_scalar_relu_i8(const void *const *inputs, void *const *outputs, uint32_t element_count,
                                const infr_quant *quant);
