// Conan::ImportStart
#include <stdint.h>

#include "arm_m_internal.h"
// Conan::ImportEnd

/*!
 * @ingroup group_arm_m
 * @file
 * @brief [en] Registration of the Cortex-M DSP tier, empty on cores without the extension.
 * @brief [zh] Cortex-M DSP 档位的注册；没有该扩展的核上为空。
 * @since 0.1.0
 */

#if defined(__ARM_FEATURE_DSP)
/**
 * @brief [en] Register every row of one table, stopping at the first conflict.
 * @brief [zh] 登记一张表里的每一行，遇到第一个冲突就停下。
 * @param rows [en] Table of rows to register.
 * @param rows [zh] 要登记的行表。
 * @param count [en] Number of rows in `rows`.
 * @param count [zh] `rows` 里的行数。
 * @return [en] INFR_OK, or the status of the first registration that failed.
 * @return [zh] INFR_OK，或第一个失败登记的返回码。
 * @note [en] A table and a loop instead of one flat initialiser: the elementwise row is guarded
 *            by its own feature macro, and a preprocessor conditional *inside* an initialiser is
 *            both awkward to read and reported by the `initform` gate as a positional element.
 *            The helper itself lives inside the same guard as its only caller, so a build
 *            without the extension has no dead function to leave uncovered.
 * @note [zh] 用「一张表 + 一个循环」而不是一个扁平初始化：逐元素那一行由它自己的特性宏守卫，而
 *            把预处理条件写进初始化列表里既难读，也会被 `initform` 门禁当成位置式元素。
 *            辅助函数与它唯一的调用者放在同一个守卫里，于是没有扩展的构建不会留下未被覆盖的死函数。
 * @since 0.1.0
 */
static infr_status register_rows(const infr_kernel_desc *rows, uint32_t count) {
    uint32_t i;

    for (i = 0U; i < count; ++i) {
        infr_status status = infr_kernel_register(&rows[i]);
        if (status != INFR_OK) {
            return status;
        }
    }
    return INFR_OK;
}
#endif

/**
 * @brief [en] Register every DSP-tier row this build contains (none without `__ARM_FEATURE_DSP`).
 * @brief [zh] 注册本次构建包含的全部 DSP 档位行（无 `__ARM_FEATURE_DSP` 时为空）。
 * @return [en] INFR_OK, including the empty case; a registration error otherwise.
 * @return [zh] INFR_OK（空集也算），或注册过程中的错误。
 * @since 0.1.0
 */
infr_status infr_kernel_register_arm_m(void) {
#if defined(__ARM_FEATURE_DSP)
    /* One row per (name, dtype, quant) the tier implements. The two dot products need the
     * paired multiply (`SMLAD`/`SMLALD`); the fp32 cells have no DSP path on this profile (the
     * FPU is a build flag, not an instruction set this tier can select), which is why the matrix
     * keeps fp32/arm_m as an unavailable cell rather than an open one.
     *
     * 这里每登记一行 = 该档位实现的一个 (名字, dtype, quant)。两个点积需要成对乘法
     * （`SMLAD`/`SMLALD`）；fp32 格在本档位没有 DSP 路径（FPU 是编译选项，不是这个档位
     * 能选择的指令集），所以矩阵里 fp32/arm_m 是「不可能」而不是「待办」。 */
    static const infr_kernel_desc dot_rows[] = {
        {.name = "fc",
         .dtype = INFR_DTYPE_I8,
         .quant = INFR_QUANT_PER_TENSOR,
         .tier = INFR_TIER_ARM_M,
         .fn = infr_arm_m_fc_i8},
        /* The accumulator half of a quantized convolution; see scalar/dot.c for why
         * `fc` cannot serve. */
        {.name = "dot",
         .dtype = INFR_DTYPE_I8,
         .quant = INFR_QUANT_PER_TENSOR,
         .tier = INFR_TIER_ARM_M,
         .fn = infr_arm_m_dot_i8},
        {.name = "fc",
         .dtype = INFR_DTYPE_I16,
         .quant = INFR_QUANT_PER_TENSOR,
         .tier = INFR_TIER_ARM_M,
         .fn = infr_arm_m_fc_i16},
    };
#if defined(__ARM_FEATURE_SIMD32)
    /* The elementwise row needs the byte-parallel half of the extension, not just the paired
     * multiply: a core with DSP but without SIMD32 keeps the two dot products and loses this
     * one, which is why the guard is per table rather than per tier.
     *
     * 逐元素这一行需要扩展的「按字节并行」那一半，而不只是成对乘法：只有 DSP 没有 SIMD32
     * 的核保留两个点积、丢掉这一行，这正是守卫按表而非按档的原因。 */
    static const infr_kernel_desc simd32_rows[] = {
        {.name = "relu",
         .dtype = INFR_DTYPE_I8,
         .quant = INFR_QUANT_NONE,
         .tier = INFR_TIER_ARM_M,
         .fn = infr_arm_m_relu_i8},
    };
#endif
    infr_status status = register_rows(dot_rows, (uint32_t)(sizeof(dot_rows) / sizeof(dot_rows[0])));

    if (status != INFR_OK) {
        return status;
    }
#if defined(__ARM_FEATURE_SIMD32)
    status = register_rows(simd32_rows, (uint32_t)(sizeof(simd32_rows) / sizeof(simd32_rows[0])));
    if (status != INFR_OK) {
        return status;
    }
#endif
#endif
    return INFR_OK;
}
