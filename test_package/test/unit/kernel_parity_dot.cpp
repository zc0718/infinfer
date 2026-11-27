// Census cell: dot/int8/tensor/scalar.
//
// The accumulator is the interface between a quantized convolution's arithmetic
// and its rounding, so it is pinned on its own (INV-2): through the registry,
// against a reference accumulated in a wider type. Two things are asserted that
// a "compare against the same loop" test would not catch: the accumulator is
// exact (an int64 reference agrees bit for bit, so no int32 wrap happened), and
// `quant` being NULL is accepted rather than dereferenced -- this cell does not
// requantize, and a caller that only wants the accumulator must not have to
// invent parameters.
#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <vector>

#include <infinfer/infinfer.h>

namespace {

const infr_kernel_desc* scalar_dot() {
    EXPECT_EQ(INFR_OK, infr_kernel_register_scalar());
    return infr_kernel_find("dot", INFR_DTYPE_I8, INFR_QUANT_PER_TENSOR, INFR_TIER_SCALAR);
}

// The reference accumulates in int64 on purpose: if the body wrapped in int32,
// the two answers would differ, and that difference is the only way "the
// accumulator is exact" can fail.
std::int64_t reference_dot(const std::vector<std::int8_t>& x, const std::vector<std::int8_t>& w) {
    std::int64_t total = 0;
    for (std::size_t i = 0; i < x.size(); ++i) {
        total += static_cast<std::int64_t>(x[i]) * static_cast<std::int64_t>(w[i]);
    }
    return total;
}

void expect_dot(const std::vector<std::int8_t>& x, const std::vector<std::int8_t>& w) {
    const infr_kernel_desc* kernel = scalar_dot();
    ASSERT_NE(nullptr, kernel);
    std::int32_t got = 0;
    // `vector::data()` on an empty vector may be NULL, and a NULL operand is
    // (correctly) rejected, so the zero-length case is fed a valid pointer with a
    // count of zero instead.
    const std::int8_t no_terms = 0;
    const std::int8_t no_weights = 0;
    const void* inputs[2] = {x.empty() ? &no_terms : x.data(), w.empty() ? &no_weights : w.data()};
    void* outputs[1] = {&got};
    ASSERT_EQ(INFR_OK, kernel->fn(inputs, outputs, static_cast<std::uint32_t>(x.size()), nullptr));
    const std::int64_t want = reference_dot(x, w);
    ASSERT_GE(want, INT32_MIN);
    ASSERT_LE(want, INT32_MAX);
    EXPECT_EQ(static_cast<std::int32_t>(want), got) << "inner dimension " << x.size();
}

} // namespace

TEST(KernelParityDot, ScalarI8MatchesInt32Reference) {
    // Length 0 is a legal inner dimension and must yield an empty sum, not an
    // error.
    expect_dot({}, {});

    // One product, both signs: the case where a sign bug shows up as a whole
    // result.
    expect_dot({3}, {-5});
    expect_dot({-128}, {-128}); // 16384: the largest single int8 product
    expect_dot({127}, {-127});

    // A short row, where every term is visible in the answer.
    expect_dot({1, 2, 3, 4}, {5, 6, 7, 8});

    // Alternating signs: the two half-sums cancel, so a missing negation cannot
    // hide inside a total that happens to look plausible.
    expect_dot({1, -1, 1, -1, 1, -1}, {100, 100, 100, 100, 100, 100});

    // A realistic convolution window: depthwise 3x3 on int8, and the expand width
    // of a mobile block. Values come from a fixed pattern, not a generator, so a
    // failure is reproducible from the test source alone.
    for (std::uint32_t n : {9U, 144U, 576U}) {
        std::vector<std::int8_t> x(n);
        std::vector<std::int8_t> w(n);
        for (std::uint32_t i = 0; i < n; ++i) {
            x[i] = static_cast<std::int8_t>((i * 37U) % 255U - 127);
            w[i] = static_cast<std::int8_t>((i * 53U) % 255U - 127);
        }
        expect_dot(x, w);
    }

    // The documented range limit, approached rather than crossed: a full 127*127
    // row of 133144 terms still fits int32 (2147483647 / 16129 = 133144). The
    // test sits exactly on that bound, so it proves the headroom claim in dot.c
    // instead of merely asserting it.
    constexpr std::uint32_t kHeadroom = 133144U;
    std::vector<std::int8_t> big_x(kHeadroom, static_cast<std::int8_t>(127));
    std::vector<std::int8_t> big_w(kHeadroom, static_cast<std::int8_t>(127));
    expect_dot(big_x, big_w);
}

TEST(KernelParityDot, ScalarRejectsNullArguments) {
    const infr_kernel_desc* kernel = scalar_dot();
    ASSERT_NE(nullptr, kernel);
    std::int8_t x[4] = {1, 2, 3, 4};
    std::int8_t w[4] = {1, 2, 3, 4};
    std::int32_t acc = 0;
    const void* inputs_ok[2] = {x, w};
    const void* inputs_no_x[2] = {nullptr, w};
    const void* inputs_no_w[2] = {x, nullptr};
    void* outputs_ok[1] = {&acc};

    EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, kernel->fn(nullptr, outputs_ok, 4U, nullptr));
    EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, kernel->fn(inputs_ok, nullptr, 4U, nullptr));
    EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, kernel->fn(inputs_no_x, outputs_ok, 4U, nullptr));
    EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, kernel->fn(inputs_no_w, outputs_ok, 4U, nullptr));
    const void* inputs_ok2[2] = {x, w};
    void* outputs_null[1] = {nullptr};
    EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, kernel->fn(inputs_ok2, outputs_null, 4U, nullptr));
}

/* ---------------------------------------------------------------------------------------------
 * The three non-host tiers of the same cell: dot/int8/tensor/{arm_neon,arm_m,rvv}.
 *
 * Each compares that tier against the scalar row **through the registry** over inner dimensions
 * that include the awkward ones -- 9 and 27 are ordinary convolution windows and neither is a
 * multiple of the vector width, which is where a widening loop and a scalar tail usually part
 * company -- and each also checks the answer against an int64 reference, so two tiers agreeing
 * cannot mean two tiers wrapping identically.
 *
 * On a host build all three skip: the rows are absent by design, and the message says so. Their
 * real runs are where the tier exists: the cross-compiled KAT and the per-tier ISA measurements.
 *
 * 同一格的三个非宿主档位。每个都通过注册表与 scalar 行比对，内维里包含不整齐的那些——9 和 27 是
 * 普通卷积窗口、都不是向量宽度的倍数，而向量宽度与标量尾部通常正是在那里分手——并且结果还要对
 * int64 参考相等，所以两个档位一致不能是"一致地回绕"。宿主构建上三者都跳过：行是按设计缺席的，
 * 消息里会说明；它们真正的运行发生在档位存在的地方。
 * ------------------------------------------------------------------------------------------- */
namespace {

void expect_dot_tier(infr_status (*register_tier)(void), infr_tier tier, const char* name) {
    EXPECT_EQ(INFR_OK, infr_kernel_register_scalar());
    EXPECT_EQ(INFR_OK, register_tier());
    const infr_kernel_desc* scalar = infr_kernel_find("dot", INFR_DTYPE_I8, INFR_QUANT_PER_TENSOR, INFR_TIER_SCALAR);
    ASSERT_NE(nullptr, scalar);
    const infr_kernel_desc* candidate = infr_kernel_find("dot", INFR_DTYPE_I8, INFR_QUANT_PER_TENSOR, tier);
    if (candidate == nullptr) {
        GTEST_SKIP() << "this build has no " << name << " tier: the row is absent by design";
    }

    for (std::uint32_t n : {0U, 1U, 9U, 16U, 17U, 27U, 31U, 144U}) {
        const std::uint32_t storage = n == 0U ? 1U : n;
        std::vector<std::int8_t> x(storage, 0);
        std::vector<std::int8_t> w(storage, 0);
        for (std::uint32_t i = 0; i < n; ++i) {
            x[i] = static_cast<std::int8_t>((i * 37U) % 255U - 127);
            w[i] = static_cast<std::int8_t>((i * 53U) % 255U - 127);
        }
        std::int32_t want = 0;
        std::int32_t got = 0;
        const void* inputs[2] = {x.data(), w.data()};
        void* want_out[1] = {&want};
        void* got_out[1] = {&got};
        ASSERT_EQ(INFR_OK, scalar->fn(inputs, want_out, n, nullptr));
        ASSERT_EQ(INFR_OK, candidate->fn(inputs, got_out, n, nullptr));
        EXPECT_EQ(want, got) << name << " element_count " << n;

        std::int64_t exact = 0;
        for (std::uint32_t i = 0; i < n; ++i) {
            exact += static_cast<std::int64_t>(x[i]) * static_cast<std::int64_t>(w[i]);
        }
        ASSERT_GE(exact, INT32_MIN);
        ASSERT_LE(exact, INT32_MAX);
        EXPECT_EQ(static_cast<std::int32_t>(exact), got) << name << " element_count " << n;
    }
}

} // namespace

TEST(KernelParityDotNeon, Int8MatchesScalarBitwise) {
    expect_dot_tier(infr_kernel_register_arm_neon, INFR_TIER_ARM_NEON, "arm_neon");
}

TEST(KernelParityDotArmM, Int8MatchesScalarBitwise) {
    expect_dot_tier(infr_kernel_register_arm_m, INFR_TIER_ARM_M, "arm_m");
}

TEST(KernelParityRvv, DotInt8MatchesScalarBitwise) { expect_dot_tier(infr_kernel_register_rvv, INFR_TIER_RVV, "rvv"); }
