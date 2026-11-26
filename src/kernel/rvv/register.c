// Conan::ImportStart
#include <stdint.h>

#include "rvv_internal.h"
// Conan::ImportEnd

/*!
 * @ingroup group_rvv
 * @file
 * @brief [en] Registration of the RVV tier, empty on targets that do not select it.
 * @brief [zh] RVV 档位的注册；未选择该指令集的目标上为空。
 * @since 0.1.0
 */

/**
 * @brief [en] Register every RVV-tier row this build contains (none without `__riscv_vector`).
 * @brief [zh] 注册本次构建包含的全部 RVV 档位行（无 `__riscv_vector` 时为空）。
 * @return [en] INFR_OK, including the empty case; a registration error otherwise.
 * @return [zh] INFR_OK（空集也算），或注册过程中的错误。
 * @note [en] Same posture as the other tiers: the tier is a capability of the build. GCC
 *            defines `__riscv_vector` only for `-march=...v...`, so a build that cannot run
 *            the code cannot advertise it either, and `infr_kernel_find` reports the absence.
 * @note [zh] 与其他档位同一姿态：档位是构建能力。GCC 只在 `-march=...v...` 时定义
 *            `__riscv_vector`，因此跑不了这段代码的构建也登记不了，缺席由
 *            `infr_kernel_find` 如实报出。
 * @note [en] fp32 rows here are **not** bit-identical to the reference in the two reductions
 *            (the dot product and the sum) and **are** bit-identical in the four elementwise
 *            kernels. Judge them with the parity harness, which already draws that line.
 * @note [zh] 这里的 fp32 行在两个归约（点积与求和）上**不是**逐位相同，在四个逐元素内核上
 *            **是**逐位相同。判定用 parity 骨架——它本来就把这条界线画好了。
 * @since 0.1.0
 */
infr_status infr_kernel_register_rvv(void) {
#if defined(__riscv_vector)
    static const infr_kernel_desc builtins[] = {
        {.name = "fc", .dtype = INFR_DTYPE_F32, .quant = INFR_QUANT_NONE, .tier = INFR_TIER_RVV, .fn = infr_rvv_fc_f32},
        {.name = "relu",
         .dtype = INFR_DTYPE_F32,
         .quant = INFR_QUANT_NONE,
         .tier = INFR_TIER_RVV,
         .fn = infr_rvv_relu_f32},
        {.name = "relu",
         .dtype = INFR_DTYPE_I8,
         .quant = INFR_QUANT_NONE,
         .tier = INFR_TIER_RVV,
         .fn = infr_rvv_relu_i8},
        {.name = "quantize",
         .dtype = INFR_DTYPE_F32,
         .quant = INFR_QUANT_PER_TENSOR,
         .tier = INFR_TIER_RVV,
         .fn = infr_rvv_quantize_f32},
        {.name = "dequantize",
         .dtype = INFR_DTYPE_I8,
         .quant = INFR_QUANT_PER_TENSOR,
         .tier = INFR_TIER_RVV,
         .fn = infr_rvv_dequantize_i8},
        {.name = "fc",
         .dtype = INFR_DTYPE_I16,
         .quant = INFR_QUANT_PER_TENSOR,
         .tier = INFR_TIER_RVV,
         .fn = infr_rvv_fc_i16},
        {.name = "fc",
         .dtype = INFR_DTYPE_I8,
         .quant = INFR_QUANT_PER_TENSOR,
         .tier = INFR_TIER_RVV,
         .fn = infr_rvv_fc_i8},
        {.name = "max",
         .dtype = INFR_DTYPE_F32,
         .quant = INFR_QUANT_NONE,
         .tier = INFR_TIER_RVV,
         .fn = infr_rvv_max_f32},
        {.name = "add",
         .dtype = INFR_DTYPE_F32,
         .quant = INFR_QUANT_NONE,
         .tier = INFR_TIER_RVV,
         .fn = infr_rvv_add_f32},
        {.name = "scale",
         .dtype = INFR_DTYPE_F32,
         .quant = INFR_QUANT_NONE,
         .tier = INFR_TIER_RVV,
         .fn = infr_rvv_scale_f32},
        {.name = "sum",
         .dtype = INFR_DTYPE_F32,
         .quant = INFR_QUANT_NONE,
         .tier = INFR_TIER_RVV,
         .fn = infr_rvv_sum_f32},
    };
    uint32_t i;

    for (i = 0U; i < (uint32_t)(sizeof(builtins) / sizeof(builtins[0])); ++i) {
        infr_status status = infr_kernel_register(&builtins[i]);
        if (status != INFR_OK) {
            return status;
        }
    }
#endif
    return INFR_OK;
}
