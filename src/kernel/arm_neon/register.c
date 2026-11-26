// Conan::ImportStart
#include <stdint.h>

#include "arm_neon_internal.h"
// Conan::ImportEnd

/*!
 * @ingroup group_arm_neon
 * @file
 * @brief [en] Registration of the NEON tier, empty on targets that do not select it.
 * @brief [zh] NEON 档位的注册；未选择该指令集的目标上为空。
 * @since 0.1.0
 */

/**
 * @brief [en] Register every NEON-tier row this build contains (none without `__ARM_NEON`).
 * @brief [zh] 注册本次构建包含的全部 NEON 档位行（无 `__ARM_NEON` 时为空）。
 * @return [en] INFR_OK, including the empty case; a registration error otherwise.
 * @return [zh] INFR_OK（空集也算），或注册过程中的错误。
 * @note [en] AArch64 always defines `__ARM_NEON`; AArch32 defines it only when the chosen FPU has NEON
 *            (the `armv7hf-linux` board's `neon-vfpv4` variant is the case that exercises this). The
 *            guard is the compiler's answer, so a build that cannot run the code also cannot
 *            advertise it.
 * @note [zh] AArch64 总是定义 `__ARM_NEON`；AArch32 只在所选 FPU 带 NEON 时定义（`armv7hf-linux`
 *            的 `neon-vfpv4` 变体正是覆盖这一点的用例）。守卫由编译器回答，因此跑不了的构建也登记不了。
 * @since 0.1.0
 */
infr_status infr_kernel_register_arm_neon(void) {
#if defined(__ARM_NEON)
    static const infr_kernel_desc builtins[] = {
        {.name = "fc",
         .dtype = INFR_DTYPE_F32,
         .quant = INFR_QUANT_NONE,
         .tier = INFR_TIER_ARM_NEON,
         .fn = infr_arm_neon_fc_f32},
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
