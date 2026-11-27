// Conan::ImportStart
#include <gtest/gtest.h>

#include <cstdint>

#include <infinfer/infinfer.h>
// Conan::ImportEnd

/*!
 * @ingroup group_contract
 * @file
 * @brief [en] Tier selection: the policy decides, and a missing cell falls back
 * instead of failing.
 * @brief [zh] 档位选择：策略说了算，缺格时回退而不是失败。
 * @note [en] Before this, the composed operators named `INFR_TIER_SCALAR` at
 * seven call sites, so every hand-written vector body was unreachable from the
 * shipping path (gaps G-122). The cases below pin the two properties that make
 * the single decision point safe: the policy is what picks, and a cell the
 * policy's tier does not carry still resolves.
 * @note [zh] 在此之前，组合算子在七处调用点写死
 * `INFR_TIER_SCALAR`，于是所有手写向量体在出货 路径上都不可达（gaps
 * G-122）。下面这些用例钉住「单一决策点」得以安全的两条性质：
 *            由策略决定档位；策略档没有的格仍然能解析出行。
 * @since 0.2.0
 */

namespace {

/* The registry and the policy are process-wide, so each case saves the policy it
 * found and puts it back: an explicit choice outranks the platform default, which
 * means `init` cannot be used as a reset.
 *
 * 注册表与策略是进程级的，所以每个用例先记下进场时的策略、退出时恢复：显式选择优先于
 * 平台默认，也就意味着 `init` 不能当重置用。 */
class KernelDispatch : public ::testing::Test {
protected:
    void SetUp() override {
        EXPECT_EQ(INFR_OK, infr_kernel_init());
        saved_ = infr_kernel_tier();
    }
    void TearDown() override { EXPECT_EQ(INFR_OK, infr_kernel_set_tier(saved_)); }

    infr_tier saved_ = INFR_TIER_SCALAR;
};

/* What this build's default is, computed here from the same guards the tier
 * bodies use rather than asked of the library -- a check that asks the code
 * under test what to expect proves nothing.
 *
 * 这个构建的默认值在这里按各档内核体自己的守卫算出来，而不是问库——让被测代码告诉
 * 自己该期望什么，什么也证不了。 */
infr_tier expected_platform_tier() {
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

TEST_F(KernelDispatch, InitPrefersThisBuildsPlatformTier) {
    const infr_kernel_desc* row = infr_kernel_find_best("fc", INFR_DTYPE_F32, INFR_QUANT_NONE);
    ASSERT_NE(nullptr, row);
    EXPECT_EQ(expected_platform_tier(), row->tier) << "fc/fp32 resolved to the wrong tier";
}

TEST_F(KernelDispatch, ThePolicyIsWhatPicksTheTier) {
    ASSERT_EQ(INFR_OK, infr_kernel_set_tier(INFR_TIER_SCALAR));
    const infr_kernel_desc* reference = infr_kernel_find_best("fc", INFR_DTYPE_F32, INFR_QUANT_NONE);
    ASSERT_NE(nullptr, reference);
    EXPECT_EQ(INFR_TIER_SCALAR, reference->tier);

    /* The platform tier, when this build has one, must come back through the
   * same call once the policy says so. Skipped rather than faked on a build
   * whose only tier is the reference.
   *
   * 本构建有平台档时，策略一改，同一次调用就必须返回它。只有参考档的构建上跳过，
   * 而不是假装通过。 */
    const infr_tier platform = expected_platform_tier();
    if (platform == INFR_TIER_SCALAR) {
        GTEST_SKIP() << "this build carries no vector tier, so there is no second "
                        "policy to test";
    }
    ASSERT_EQ(INFR_OK, infr_kernel_set_tier(platform));
    const infr_kernel_desc* preferred = infr_kernel_find_best("fc", INFR_DTYPE_F32, INFR_QUANT_NONE);
    ASSERT_NE(nullptr, preferred);
    EXPECT_EQ(platform, preferred->tier);
}

TEST_F(KernelDispatch, ACellTheTierDoesNotCarryResolvesToTheReference) {
    /* `requant` is the load-bearing case and not a made-up one: the matrix has it
   * `done` on scalar and `na` on every vector tier (gaps G-121 measured why),
   * so a quantised convolution resolves `dot` to the platform tier and
   * `requant` to the reference **in the same node**. If this fell back to NULL
   * instead, every int8 graph would stop with INFR_ERR_NOT_IMPLEMENTED the
   * moment the policy stopped being scalar.
   *
   * `requant` 是真正承重的例子，不是编的：矩阵里它在 scalar 上是
   * `done`、在所有向量档 上是 `na`（理由见 G-121
   * 的实测），所以一个量化卷积会在同一个节点里把 `dot` 解析到 平台档、把
   * `requant` 解析到参考档。若这里退成 NULL，那么策略一旦不是 scalar，每张 int8
   * 图都会以 INFR_ERR_NOT_IMPLEMENTED 停下。 */
    ASSERT_EQ(INFR_OK, infr_kernel_set_tier(expected_platform_tier()));
    const infr_kernel_desc* row = infr_kernel_find_best("requant", INFR_DTYPE_I32, INFR_QUANT_PER_TENSOR);
    ASSERT_NE(nullptr, row) << "the fallback chain must end at the reference, not at NULL";
    EXPECT_EQ(INFR_TIER_SCALAR, row->tier);
}

TEST_F(KernelDispatch, ATierThisBuildNeverRegistersFallsBackPerCell) {
    /* `INFR_TIER_VENDOR` is declared by the contract and registered by nothing
   * yet. Asking for it must degrade to the reference for every cell rather
   * than fail -- that is what makes a vendor backend additive.
   *
   * `INFR_TIER_VENDOR`
   * 是契约已声明、但今天没人登记的档。要它必须按格退到参考实现，
   * 而不是失败——这才让厂商后端是可加的。 */
    ASSERT_EQ(INFR_OK, infr_kernel_set_tier(INFR_TIER_VENDOR));
    struct Cell {
        const char* name;
        infr_dtype dtype;
        infr_quant_kind quant;
    };
    const Cell cells[] = {
        {"fc", INFR_DTYPE_F32, INFR_QUANT_NONE},
        {"add", INFR_DTYPE_F32, INFR_QUANT_NONE},
        {"sum", INFR_DTYPE_F32, INFR_QUANT_NONE},
        {"max", INFR_DTYPE_F32, INFR_QUANT_NONE},
        {"scale", INFR_DTYPE_F32, INFR_QUANT_NONE},
        {"dot", INFR_DTYPE_I8, INFR_QUANT_PER_TENSOR},
        {"requant", INFR_DTYPE_I32, INFR_QUANT_PER_TENSOR},
    };
    for (const Cell& cell : cells) {
        const infr_kernel_desc* row = infr_kernel_find_best(cell.name, cell.dtype, cell.quant);
        ASSERT_NE(nullptr, row) << cell.name << " lost its reference row";
        EXPECT_EQ(INFR_TIER_SCALAR, row->tier) << cell.name << " did not fall back";
    }
}

TEST_F(KernelDispatch, RefusesATierOutsideTheEnum) {
    const infr_kernel_desc* before = infr_kernel_find_best("fc", INFR_DTYPE_F32, INFR_QUANT_NONE);
    ASSERT_NE(nullptr, before);
    const infr_tier policy_before = before->tier;

    EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, infr_kernel_set_tier(static_cast<infr_tier>(7)));
    const infr_kernel_desc* after = infr_kernel_find_best("fc", INFR_DTYPE_F32, INFR_QUANT_NONE);
    ASSERT_NE(nullptr, after);
    EXPECT_EQ(policy_before, after->tier) << "a rejected policy must not take effect";
}

TEST_F(KernelDispatch, AnAbsentKernelIsNullAndNotAReferenceRow) {
    EXPECT_EQ(nullptr, infr_kernel_find_best("nosuchkernel", INFR_DTYPE_F32, INFR_QUANT_NONE));
    EXPECT_EQ(nullptr, infr_kernel_find_best(nullptr, INFR_DTYPE_F32, INFR_QUANT_NONE));
}

TEST_F(KernelDispatch, InitIsIdempotent) {
    const infr_tier before = infr_kernel_tier();
    EXPECT_EQ(INFR_OK, infr_kernel_init());
    EXPECT_EQ(INFR_OK, infr_kernel_init());
    EXPECT_EQ(before, infr_kernel_tier()) << "init must not change a settled policy";
}

TEST_F(KernelDispatch, AnExplicitPolicySurvivesInit) {
    /* This is the property the KAT images rely on: they pin the reference tier
     * and then call an operator that registers on every entry.
     *
     * 这是 KAT 镜像依赖的性质：它们钉住参考档，然后去调一个每次入口都会注册的算子。 */
    ASSERT_EQ(INFR_OK, infr_kernel_set_tier(INFR_TIER_SCALAR));
    ASSERT_EQ(INFR_OK, infr_kernel_init());
    EXPECT_EQ(INFR_TIER_SCALAR, infr_kernel_tier()) << "init overrode an explicit choice";
    const infr_kernel_desc* row = infr_kernel_find_best("fc", INFR_DTYPE_F32, INFR_QUANT_NONE);
    ASSERT_NE(nullptr, row);
    EXPECT_EQ(INFR_TIER_SCALAR, row->tier);
}

} // namespace
