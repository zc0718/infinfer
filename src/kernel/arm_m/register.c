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

/**
 * @brief [en] Register every DSP-tier row this build contains (none without `__ARM_FEATURE_DSP`).
 * @brief [zh] 注册本次构建包含的全部 DSP 档位行（无 `__ARM_FEATURE_DSP` 时为空）。
 * @return [en] INFR_OK, including the empty case; a registration error otherwise.
 * @return [zh] INFR_OK（空集也算），或注册过程中的错误。
 * @since 0.1.0
 */
infr_status infr_kernel_register_arm_m(void) {
#if defined(__ARM_FEATURE_DSP)
    /* One row per (name, dtype, quant) the tier implements. Today only the int8 dot
     * product: the fp32 cells have no DSP path on this profile (the FPU is a build
     * flag, not an instruction set this tier can select), which is why the matrix
     * keeps fp32/arm_m as an open cell rather than a claim.
     *
     * 这里每登记一行 = 该档位实现的一个 (名字, dtype, quant)。目前只有 int8 点积：
     * fp32 格在本档位没有 DSP 路径（FPU 是编译选项，不是这个档位能选择的指令集），
     * 所以矩阵里 fp32/arm_m 仍是待办格而不是声称。 */
    static const infr_kernel_desc builtins[] = {
        {"fc", INFR_DTYPE_I8, INFR_QUANT_PER_TENSOR, INFR_TIER_ARM_M, infr_arm_m_fc_i8},
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
