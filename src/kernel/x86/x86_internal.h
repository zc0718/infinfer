// Conan::ImportStart
#include <stdint.h>

#include <infinfer/kernel/kernel.h>
// Conan::ImportEnd

/*!
 * @ingroup group_x86
 * @file
 * @brief [en] Internal declarations of the x86 (SSE2) tier bodies.
 * @brief [zh] x86（SSE2）档位实现体的内部声明。
 * @note [en] Four lanes of `__m128`, the baseline every x86-64 machine has: no ISA flag has to be
 *            negotiated for this tier to exist, which is why it is the only vector tier that can
 *            be exercised by the host test run itself instead of by a probe on another machine.
 * @note [zh] `__m128` 的四车道，是每台 x86-64 都具备的基线：本档位不需要协商任何 ISA 开关就能存在，
 *            因此它是**唯一能在宿主测试里被真正执行**的向量档位，而不是靠另一台机器上的探针。
 * @since 0.1.0
 */

#ifndef INFR_SRC_KERNEL_X86_INTERNAL_H
#define INFR_SRC_KERNEL_X86_INTERNAL_H

/**
 * @brief [en] fp32 dot product: four-lane partial sums, horizontal fold, scalar tail.
 * @brief [zh] fp32 点积：四车道部分和、水平折叠、标量尾。
 * @note [en] Not bitwise: the reference carries one running float, this carries four.
 * @note [zh] 非逐位：参考只有一条流水和，这里有四条。
 * @since 0.1.0
 */
infr_status infr_x86_fc_f32(const void *const *inputs, void *const *outputs, uint32_t element_count,
                            const infr_quant *quant);

/**
 * @brief [en] fp32 relu as a comparison plus a select, bitwise identical to the reference.
 * @brief [zh] fp32 relu 写成比较加选择，与参考实现逐位相同。
 * @since 0.1.0
 */
infr_status infr_x86_relu_f32(const void *const *inputs, void *const *outputs, uint32_t element_count,
                              const infr_quant *quant);

/**
 * @brief [en] fp32 maximum reduction, first occurrence wins, bitwise identical.
 * @brief [zh] fp32 最大值归约，首次出现者胜，逐位相同。
 * @since 0.1.0
 */
infr_status infr_x86_max_f32(const void *const *inputs, void *const *outputs, uint32_t element_count,
                             const infr_quant *quant);

/**
 * @brief [en] fp32 elementwise add, bitwise identical to the reference.
 * @brief [zh] fp32 逐元素加法，与参考实现逐位相同。
 * @since 0.1.0
 */
infr_status infr_x86_add_f32(const void *const *inputs, void *const *outputs, uint32_t element_count,
                             const infr_quant *quant);

/**
 * @brief [en] fp32 broadcast multiply by `inputs[1][0]`, bitwise identical.
 * @brief [zh] 按 `inputs[1][0]` 的 fp32 广播乘法，逐位相同。
 * @since 0.1.0
 */
infr_status infr_x86_scale_f32(const void *const *inputs, void *const *outputs, uint32_t element_count,
                               const infr_quant *quant);

/**
 * @brief [en] fp32 sum into one value: four-lane partial sums, horizontal fold, scalar tail.
 * @brief [zh] fp32 求和到单值：四车道部分和、水平折叠、标量尾。
 * @note [en] Not bitwise: same reason as the dot product.
 * @note [zh] 非逐位：与点积同一原因。
 * @since 0.1.0
 */
infr_status infr_x86_sum_f32(const void *const *inputs, void *const *outputs, uint32_t element_count,
                             const infr_quant *quant);

/**
 * @brief [en] int8 relu: `v > 0 ? v : 0` over one vector block, bitwise identical.
 * @brief [zh] int8 relu：一个向量块上的 `v > 0 ? v : 0`，逐位相同。
 * @note [en] Integer maximum, unlike the fp32 row in this tier (see `relu.c`).
 * @note [zh] 整数取大，与本档位的 fp32 行不同（见 `relu.c`）。
 * @since 0.1.0
 */
infr_status infr_x86_relu_i8(const void *const *inputs, void *const *outputs, uint32_t element_count,
                             const infr_quant *quant);

#endif /* INFR_SRC_KERNEL_X86_INTERNAL_H */
