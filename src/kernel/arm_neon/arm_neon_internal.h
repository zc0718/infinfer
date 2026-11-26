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

/**
 * @brief [en] fp32 relu over four lanes: `x > 0 ? x : 0`, bitwise identical to the reference.
 * @brief [zh] 四车道 fp32 relu：`x > 0 ? x : 0`，与参考实现逐位相同。
 * @since 0.1.0
 */
infr_status infr_arm_neon_relu_f32(const void *const *inputs, void *const *outputs, uint32_t element_count,
                                   const infr_quant *quant);

/**
 * @brief [en] fp32 maximum over four lanes: first occurrence wins, NaN declined, empty refused.
 * @brief [zh] 四车道 fp32 最大值：首次出现者胜、NaN 不被选中、空集拒绝。
 * @since 0.1.0
 */
infr_status infr_arm_neon_max_f32(const void *const *inputs, void *const *outputs, uint32_t element_count,
                                  const infr_quant *quant);

/**
 * @brief [en] fp32 elementwise add over four lanes: `y[i] = a[i] + b[i]`, bitwise identical.
 * @brief [zh] 四车道 fp32 逐元素加法：`y[i] = a[i] + b[i]`，逐位相同。
 * @since 0.1.0
 */
infr_status infr_arm_neon_add_f32(const void *const *inputs, void *const *outputs, uint32_t element_count,
                                  const infr_quant *quant);

/**
 * @brief [en] fp32 broadcast multiply by `inputs[1][0]`, bitwise identical.
 * @brief [zh] 按 `inputs[1][0]` 的 fp32 广播乘法，逐位相同。
 * @since 0.1.0
 */
infr_status infr_arm_neon_scale_f32(const void *const *inputs, void *const *outputs, uint32_t element_count,
                                    const infr_quant *quant);

/**
 * @brief [en] fp32 sum into one value: four running lanes, then a two-step fold, then a tail.
 * @brief [zh] fp32 求和到单值：四条流水车道、两级折叠、再加标量尾。
 * @note [en] Not bitwise: the reference accumulates in one float, so the grouping differs.
 * @note [zh] 非逐位：参考实现用一个 float 累加，分组不同。
 * @since 0.1.0
 */
infr_status infr_arm_neon_sum_f32(const void *const *inputs, void *const *outputs, uint32_t element_count,
                                  const infr_quant *quant);

/**
 * @brief [en] int8 relu: `v > 0 ? v : 0` over one vector block, bitwise identical.
 * @brief [zh] int8 relu：一个向量块上的 `v > 0 ? v : 0`，逐位相同。
 * @note [en] Integer maximum, unlike the fp32 row in this tier (see `relu.c`).
 * @note [zh] 整数取大，与本档位的 fp32 行不同（见 `relu.c`）。
 * @since 0.1.0
 */
infr_status infr_arm_neon_relu_i8(const void *const *inputs, void *const *outputs, uint32_t element_count,
                                  const infr_quant *quant);
