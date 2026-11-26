// Conan::ImportStart
#include <stdint.h>

#include "x86_internal.h"
// Conan::ImportEnd

/*!
 * @ingroup group_x86
 * @file
 * @brief [en] Registration of the x86 tier, empty on targets that do not select it.
 * @brief [zh] x86 档位的注册；未选择该指令集的目标上为空。
 * @since 0.1.0
 */

/**
 * @brief [en] Register every x86-tier row this build contains (none without `__SSE2__`).
 * @brief [zh] 注册本次构建包含的全部 x86 档位行（无 `__SSE2__` 时为空）。
 * @return [en] INFR_OK, including the empty case; a registration error otherwise.
 * @return [zh] INFR_OK（空集也算），或注册过程中的错误。
 * @note [en] Same posture as the other tiers: the tier is a capability of the build. SSE2 is
 *            baseline on x86-64 but not on 32-bit x86, so the guard is still the compiler's
 *            answer rather than an assumption about the host.
 * @note [zh] 与其他档位同一姿态：档位是构建能力。SSE2 在 x86-64 上是基线、在 32 位 x86 上不是，
 *            所以守卫仍由编译器回答，而不是对宿主作假设。
 * @since 0.1.0
 */
infr_status infr_kernel_register_x86(void) {
#if defined(__SSE2__)
    static const infr_kernel_desc builtins[] = {
        {.name = "fc", .dtype = INFR_DTYPE_F32, .quant = INFR_QUANT_NONE, .tier = INFR_TIER_X86, .fn = infr_x86_fc_f32},
        {.name = "fc",
         .dtype = INFR_DTYPE_I8,
         .quant = INFR_QUANT_PER_TENSOR,
         .tier = INFR_TIER_X86,
         .fn = infr_x86_fc_i8},
        {.name = "relu",
         .dtype = INFR_DTYPE_F32,
         .quant = INFR_QUANT_NONE,
         .tier = INFR_TIER_X86,
         .fn = infr_x86_relu_f32},
        {.name = "relu",
         .dtype = INFR_DTYPE_I8,
         .quant = INFR_QUANT_NONE,
         .tier = INFR_TIER_X86,
         .fn = infr_x86_relu_i8},
        {.name = "max",
         .dtype = INFR_DTYPE_F32,
         .quant = INFR_QUANT_NONE,
         .tier = INFR_TIER_X86,
         .fn = infr_x86_max_f32},
        {.name = "add",
         .dtype = INFR_DTYPE_F32,
         .quant = INFR_QUANT_NONE,
         .tier = INFR_TIER_X86,
         .fn = infr_x86_add_f32},
        {.name = "scale",
         .dtype = INFR_DTYPE_F32,
         .quant = INFR_QUANT_NONE,
         .tier = INFR_TIER_X86,
         .fn = infr_x86_scale_f32},
        {.name = "sum",
         .dtype = INFR_DTYPE_F32,
         .quant = INFR_QUANT_NONE,
         .tier = INFR_TIER_X86,
         .fn = infr_x86_sum_f32},
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
