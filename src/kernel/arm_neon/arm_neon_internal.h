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
 * @brief [en] fp32 elementwise multiply over four lanes: `y[i] = a[i] * b[i]`, bitwise identical.
 * @brief [zh] 四车道 fp32 逐元素乘法：`y[i] = a[i] * b[i]`，逐位相同。
 * @since 0.1.0
 */
infr_status infr_arm_neon_mul_f32(const void *const *inputs, void *const *outputs, uint32_t element_count,
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

/**
 * @brief [en] int8 to fp32 dequantize: integer subtraction then one IEEE multiply, bitwise equal.
 * @brief [zh] int8 到 fp32 反量化：整数减法后一次 IEEE 乘法，逐位相等。
 * @note [en] The quantize direction is deliberately absent (gaps G-77): the reference divides and
 *            AArch32 NEON has no IEEE divide, so a reciprocal body would be a different function.
 * @note [zh] 量化方向**故意缺席**（gaps G-77）：参考实现做除法，而 AArch32 NEON 没有 IEEE 除法，
 *            用倒数写成的体会是另一个函数。
 * @since 0.1.0
 */
infr_status infr_arm_neon_dequantize_i8(const void *const *inputs, void *const *outputs, uint32_t element_count,
                                        const infr_quant *quant);

/**
 * @brief [en] int16 dot product with an int64 accumulator, bitwise identical to the reference.
 * @brief [zh] int64 累加的 int16 点积，与参考实现逐位相同。
 * @note [en] `vmlal_s16` accumulates in int32 and would wrap where the reference does not, so every
 *            product is widened to int64 before the add (gaps G-52).
 * @note [zh] `vmlal_s16` 累加进 int32，会在参考实现不回绕的输入上回绕，所以每个乘积在相加前扩到
 *            int64（gaps G-52）。
 * @since 0.1.0
 */
infr_status infr_arm_neon_fc_i16(const void *const *inputs, void *const *outputs, uint32_t element_count,
                                 const infr_quant *quant);

/**
 * @brief [en] int8 dot product with an int32 accumulator, bitwise identical to the reference.
 * @brief [zh] int32 累加的 int8 点积，与参考实现逐位相同。
 * @note [en] No dot-product instruction is needed: `vmull_s8` + `vpadalq_s16` reach the same exact
 *            integer sum on both AArch32 and AArch64, so unlike `quantize` this cell can be claimed
 *            for the whole tier.
 * @note [zh] 不需要点积指令：`vmull_s8` + `vpadalq_s16` 在 AArch32 与 AArch64 上都能得到同一个
 *            精确整数和，所以与 `quantize` 不同，这一格可以按整个档位声称。
 * @since 0.1.0
 */
infr_status infr_arm_neon_fc_i8(const void *const *inputs, void *const *outputs, uint32_t element_count,
                                const infr_quant *quant);
