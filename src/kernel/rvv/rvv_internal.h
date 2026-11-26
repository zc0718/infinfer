// Conan::ImportStart
#include <stdint.h>

#include <infinfer/kernel/kernel.h>
// Conan::ImportEnd

/*!
 * @ingroup group_rvv
 * @file
 * @brief [en] Internal declarations of the RISC-V Vector tier bodies.
 * @brief [zh] RISC-V Vector 档位实现体的内部声明。
 * @note [en] Every declaration here is defined only when `__riscv_vector` is set; the
 *            registration entry point exists in every build and registers nothing without it.
 * @note [zh] 这里的每个声明都只在 `__riscv_vector` 存在时才被定义；注册入口在所有构建里都存在，
 *            但没有该扩展时它什么都不登记。
 * @since 0.1.0
 */

#ifndef INFR_SRC_KERNEL_RVV_INTERNAL_H
#define INFR_SRC_KERNEL_RVV_INTERNAL_H

/**
 * @brief [en] fp32 dot product: vector partial sums, one ordered fold, scalar tail.
 * @brief [zh] fp32 点积：向量部分和、一次有序折叠、标量尾。
 * @note [en] Not bitwise: the reference accumulates in one float, so the grouping differs.
 * @note [zh] 非逐位：参考实现用一个 float 累加，分组不同。
 * @since 0.1.0
 */
infr_status infr_rvv_fc_f32(const void *const *inputs, void *const *outputs, uint32_t element_count,
                            const infr_quant *quant);

/**
 * @brief [en] fp32 relu as `maximum_number(x, 0)`, which is the reference's `x > 0 ? x : 0`.
 * @brief [zh] fp32 relu，写成 `maximum_number(x, 0)`，正是参考实现的 `x > 0 ? x : 0`。
 * @since 0.1.0
 */
infr_status infr_rvv_relu_f32(const void *const *inputs, void *const *outputs, uint32_t element_count,
                              const infr_quant *quant);

/**
 * @brief [en] fp32 maximum reduction: lane-wise maximum, then one ordered fold.
 * @brief [zh] fp32 最大值归约：逐车道取最大，再一次有序折叠。
 * @since 0.1.0
 */
infr_status infr_rvv_max_f32(const void *const *inputs, void *const *outputs, uint32_t element_count,
                             const infr_quant *quant);

/**
 * @brief [en] fp32 elementwise add, bitwise identical to the reference.
 * @brief [zh] fp32 逐元素加法，与参考实现逐位相同。
 * @since 0.1.0
 */
infr_status infr_rvv_add_f32(const void *const *inputs, void *const *outputs, uint32_t element_count,
                             const infr_quant *quant);

/**
 * @brief [en] fp32 broadcast multiply by `inputs[1][0]`, bitwise identical.
 * @brief [zh] 按 `inputs[1][0]` 的 fp32 广播乘法，逐位相同。
 * @since 0.1.0
 */
infr_status infr_rvv_scale_f32(const void *const *inputs, void *const *outputs, uint32_t element_count,
                               const infr_quant *quant);

/**
 * @brief [en] fp32 sum into one value: vector partial sums, one ordered fold, scalar tail.
 * @brief [zh] fp32 求和到单值：向量部分和、一次有序折叠、标量尾。
 * @note [en] Not bitwise: same reason as the dot product.
 * @note [zh] 非逐位：与点积同一原因。
 * @since 0.1.0
 */
infr_status infr_rvv_sum_f32(const void *const *inputs, void *const *outputs, uint32_t element_count,
                             const infr_quant *quant);

/**
 * @brief [en] int8 relu: `v > 0 ? v : 0` over one vector block, bitwise identical.
 * @brief [zh] int8 relu：一个向量块上的 `v > 0 ? v : 0`，逐位相同。
 * @note [en] Integer maximum, unlike the fp32 row in this tier (see `relu.c`).
 * @note [zh] 整数取大，与本档位的 fp32 行不同（见 `relu.c`）。
 * @since 0.1.0
 */
infr_status infr_rvv_relu_i8(const void *const *inputs, void *const *outputs, uint32_t element_count,
                             const infr_quant *quant);

/**
 * @brief [en] fp32 to int8 quantize: IEEE divide, add zero point, round half away from zero, clamp.
 * @brief [zh] fp32 到 int8 量化：IEEE 除法、加零点、半远离零舍入、钳位。
 * @note [en] The division is a real division: IEEE divide and multiply-by-reciprocal differ in the
 *            last bit, and this kernel's answer is an integer produced by rounding.
 * @note [zh] 这里用真除法：IEEE 除法与"乘倒数"在末位上并不相等，而本内核的答案是舍入后的整数。
 * @since 0.1.0
 */
infr_status infr_rvv_quantize_f32(const void *const *inputs, void *const *outputs, uint32_t element_count,
                                  const infr_quant *quant);

/**
 * @brief [en] int8 to fp32 dequantize, bitwise equal to the reference.
 * @brief [zh] int8 到 fp32 反量化，与参考实现逐位相等。
 * @since 0.1.0
 */
infr_status infr_rvv_dequantize_i8(const void *const *inputs, void *const *outputs, uint32_t element_count,
                                   const infr_quant *quant);

/**
 * @brief [en] int16 dot product with an int64 accumulator, bitwise identical to the reference.
 * @brief [zh] int64 累加的 int16 点积，与参考实现逐位相同。
 * @note [en] i16 -> i32 (multiply) -> i64 (accumulate): `vwmacc` would accumulate in int32, the
 *            width this kernel exists to avoid (gaps G-52).
 * @note [zh] i16 -> i32（乘）-> i64（累加）：`vwmacc` 会累加进 int32，而那正是本内核要避开的宽度
 *            （gaps G-52）。
 * @since 0.1.0
 */
infr_status infr_rvv_fc_i16(const void *const *inputs, void *const *outputs, uint32_t element_count,
                            const infr_quant *quant);

/**
 * @brief [en] int8 dot product with an int32 accumulator (widen to int16, then `vwmacc`).
 * @brief [zh] int32 累加的 int8 点积（先加宽到 int16，再 `vwmacc`）。
 * @note [en] `vmacc.vv` on int8 lanes would wrap immediately; the product of two int8 values is
 *            exact in int16, which is what makes the widening form both correct and cheap.
 * @note [zh] 在 int8 车道上 `vmacc.vv` 会立刻回绕；两个 int8 的乘积在 int16 里是精确的，这正是
 *            加宽写法既正确又便宜的原因。
 * @since 0.1.0
 */
infr_status infr_rvv_fc_i8(const void *const *inputs, void *const *outputs, uint32_t element_count,
                           const infr_quant *quant);

#endif /* INFR_SRC_KERNEL_RVV_INTERNAL_H */
