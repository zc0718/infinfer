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
