// Conan::ImportStart
#include <infinfer/kernel/kernel.h>
// Conan::ImportEnd

/*!
 * @ingroup group_contract
 * @file
 * @brief [en] Tier selection: one place decides which tier the operators ask for.
 * @brief [zh] 档位选择：由这一处决定组合算子向注册表要哪一档。
 * @note [en] The composed operators used to name `INFR_TIER_SCALAR` at every call
 *            site, which made every hand-written vector body unreachable from the
 *            shipping path (gaps G-122). The tier is a property of the platform,
 *            so it belongs in one place.
 * @note [zh] 组合算子此前在每个调用点都写死 `INFR_TIER_SCALAR`，于是所有手写向量体
 *            在出货路径上都不可达（gaps G-122）。档位是平台的属性，只该有一处决定。
 * @since 0.2.0
 */

/*
 * The default tier is chosen by the same guard its bodies use, not by a CPU
 * probe. Every tier is covered by the baseline that compiles it: SSE2 is the
 * x86-64 baseline, NEON is the ARMv8-A baseline, `-mcpu=cortex-m4` implies DSP,
 * and RISC-V defines `__riscv_vector` only when V is asked for. So "the guard is
 * on" and "the bodies are in" are the same fact, and a build that turns the ISA
 * off registers no row and falls back per cell.
 *
 * 默认档用的是它自己那套守卫，而不是 CPU 探测。每一档都被编译它的基线覆盖：
 * SSE2 是 x86-64 基线、NEON 是 ARMv8-A 基线、`-mcpu=cortex-m4` 蕴含 DSP、
 * RISC-V 只有显式要求 V 才定义 `__riscv_vector`。所以「守卫开着」与「体在库里」
 * 是同一件事，关掉该 ISA 的构建登记不到行，就按格回退。
 */
static infr_tier platform_tier(void) {
#if defined(__SSE2__)
    return INFR_TIER_X86;
#elif defined(__ARM_NEON)
    return INFR_TIER_ARM_NEON;
#elif defined(__ARM_FEATURE_DSP)
    return INFR_TIER_ARM_M;
#elif defined(__riscv_vector)
    return INFR_TIER_RVV;
#else
    return INFR_TIER_SCALAR;
#endif
}

/*
 * The policy starts at the reference tier, not at the platform tier: a consumer
 * that registers only the scalar rows must keep getting the reference, and the
 * KAT images depend on that for their bit-exact goldens.
 *
 * 策略初值是参考档，不是平台档：只注册 scalar 行的消费者必须继续拿到参考实现，
 * KAT 镜像的逐位 golden 依赖这一点。
 */
static infr_tier g_tier_policy = INFR_TIER_SCALAR;

/*
 * Set by the first `infr_kernel_set_tier`; from then on `infr_kernel_init` stops
 * applying the platform default. Without this an operator calling `init` on
 * every entry would undo a consumer's decision, and the KAT images, which pin
 * the reference tier, would change tier on their first operator call.
 *
 * 第一次 `infr_kernel_set_tier` 之后置位；此后 `infr_kernel_init` 不再套用平台
 * 默认。否则每个入口都调 `init` 的算子会撤销消费者的决定，而钉住参考档的 KAT
 * 镜像会在第一次调算子时换档。
 */
static int g_tier_explicit = 0;

/**
 * @brief [en] Register every tier this build has, then prefer the best of them.
 * @brief [zh] 注册本构建拥有的全部档位，然后优先使用其中最好的。
 * @since 0.2.0
 */
infr_status infr_kernel_init(void) {
    /* Every register_* is a no-op returning INFR_OK when its ISA guard is off,
     * so the five calls are unconditional and the registry stays the only thing
     * that knows which rows exist.
     *
     * 每个 register_* 在自身 ISA 守卫关闭时都是返回 INFR_OK 的空操作，所以五次
     * 调用无条件写出，而「哪些行存在」仍然只有注册表知道。 */
    const infr_status statuses[] = {
        infr_kernel_register_scalar(), infr_kernel_register_x86(), infr_kernel_register_arm_neon(),
        infr_kernel_register_arm_m(),  infr_kernel_register_rvv(),
    };
    const uint32_t count = (uint32_t)(sizeof(statuses) / sizeof(statuses[0]));
    for (uint32_t index = 0U; index < count; ++index) {
        if (statuses[index] != INFR_OK) {
            return statuses[index];
        }
    }
    if (g_tier_explicit == 0) {
        g_tier_policy = platform_tier();
    }
    return INFR_OK;
}

/**
 * @brief [en] Set the tier policy.
 * @brief [zh] 设置档位策略。
 * @since 0.2.0
 */
infr_status infr_kernel_set_tier(infr_tier tier) {
    /* The enum is the range: `INFR_TIER_VENDOR` is the last member today, so a
     * value past it is a caller bug, not a tier this build lacks.
     *
     * 枚举本身就是范围：今天是 `INFR_TIER_VENDOR` 收尾，越过它的值属于调用方
     * 的错误，而不是本构建缺少的档位。 */
    if ((uint32_t)tier > (uint32_t)INFR_TIER_VENDOR) {
        return INFR_ERR_INVALID_ARGUMENT;
    }
    g_tier_policy = tier;
    g_tier_explicit = 1;
    return INFR_OK;
}

/**
 * @brief [en] Report the tier policy in force.
 * @brief [zh] 报告当前生效的档位策略。
 * @since 0.2.0
 */
infr_tier infr_kernel_tier(void) { return g_tier_policy; }

/**
 * @brief [en] Ask the registry for the policy's row, else the scalar one.
 * @brief [zh] 向注册表要策略档的行，取不到再要 scalar 的。
 * @since 0.2.0
 */
const infr_kernel_desc *infr_kernel_find_best(const char *name, infr_dtype dtype, infr_quant_kind quant) {
    if (name == NULL) {
        return NULL;
    }
    /* Two lookups, not a table walk: the scalar tier carries every cell, so
     * "policy, then reference" is a complete chain and the caller never has to
     * know which tiers this build has.
     *
     * 两次查找而不是遍历一堆档：scalar 档承载每一格，所以「策略档、再参考档」
     * 是一条完整的链，调用方不必知道本构建有哪些档。 */
    if (g_tier_policy != INFR_TIER_SCALAR) {
        const infr_kernel_desc *preferred = infr_kernel_find(name, dtype, quant, g_tier_policy);
        if (preferred != NULL) {
            return preferred;
        }
    }
    return infr_kernel_find(name, dtype, quant, INFR_TIER_SCALAR);
}
