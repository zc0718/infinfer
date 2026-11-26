// Conan::ImportStart
#pragma once
#include <infinfer/kernel/kernel.h>
// Conan::ImportEnd

/*!
 * @ingroup group_arm_m
 * @file
 * @brief [en] Internal contract of the Cortex-M DSP tier: not public API.
 * @brief [zh] Cortex-M DSP 档位的内部契约，不是公开 API。
 * @since 0.1.0
 */

/**
 * @brief [en] int8 dot product using the DSP extension's paired multiply-accumulate.
 * @brief [zh] 用 DSP 扩展的成对乘加指令完成 int8 点积。
 * @note [en] Only declared when the translation unit is compiled for a core with
 *            `__ARM_FEATURE_DSP`; `infr_kernel_register_arm_m()` is what decides
 *            whether the registry ever sees this row.
 * @note [zh] 只在编译目标带 `__ARM_FEATURE_DSP` 时存在；是否进入注册表由
 *            `infr_kernel_register_arm_m()` 决定。
 * @since 0.1.0
 */
infr_status infr_arm_m_fc_i8(const void *const *inputs, void *const *outputs, uint32_t element_count,
                             const infr_quant *quant);

/**
 * @brief [en] int16 dot product using the DSP extension's dual multiply-accumulate into 64 bits.
 * @brief [zh] 用 DSP 扩展的成对乘加（64 位累加）完成 int16 点积。
 * @note [en] Declared under the same `__ARM_FEATURE_DSP` condition as the int8 row; the registry
 *            decides whether it is ever visible.
 * @note [zh] 与 int8 行同一个 `__ARM_FEATURE_DSP` 条件；是否可见由注册表决定。
 * @since 0.1.0
 */
infr_status infr_arm_m_fc_i16(const void *const *inputs, void *const *outputs, uint32_t element_count,
                              const infr_quant *quant);

/**
 * @brief [en] int8 ReLU using the SIMD32 half of the DSP extension (`SADD8` + `SEL`).
 * @brief [zh] 用 DSP 扩展的 SIMD32 部分（`SADD8` + `SEL`）完成 int8 的 ReLU。
 * @note [en] A different feature macro from the dot-product rows: byte-parallel arithmetic needs
 *            `__ARM_FEATURE_SIMD32`, and the registry row is conditional on it exactly as the
 *            other two are conditional on `__ARM_FEATURE_DSP`.
 * @note [zh] 与点积行的特性宏不同：按字节并行运算需要 `__ARM_FEATURE_SIMD32`，注册行以它为条件，
 *            就像另两行以 `__ARM_FEATURE_DSP` 为条件一样。
 * @since 0.1.0
 */
infr_status infr_arm_m_relu_i8(const void *const *inputs, void *const *outputs, uint32_t element_count,
                               const infr_quant *quant);
