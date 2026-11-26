// Conan::ImportStart
#pragma once
#include <infinfer/kernel/kernel.h>
// Conan::ImportEnd

/*!
 * @ingroup group_arm_neon
 * @file
 * @brief [en] Internal contract of the AArch32/AArch64 NEON tier: not public API.
 * @brief [zh] AArch32/AArch64 NEON 档位的内部契约，不是公开 API。
 * @since 0.1.0
 */

/**
 * @brief [en] fp32 dot product using four float lanes and a horizontal tail.
 * @brief [zh] 用四个 float 车道加速的 fp32 点积，末端做一次水平求和。
 * @note [en] Only declared when this translation unit is compiled with `__ARM_NEON`;
 *            `infr_kernel_register_arm_neon()` decides whether the registry sees it.
 * @note [zh] 仅在本编译单元带 `__ARM_NEON` 时存在；是否进入注册表由
 *            `infr_kernel_register_arm_neon()` 决定。
 * @since 0.1.0
 */
infr_status infr_arm_neon_fc_f32(const void *const *inputs, void *const *outputs, uint32_t element_count,
                                 const infr_quant *quant);
