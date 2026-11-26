// Conan::ImportStart
#include <gtest/gtest.h>

#include <climits>
#include <cstdint>
#include <vector>

#include <infinfer/infinfer.h>

#include "parity_harness.hpp"
// Conan::ImportEnd

/*!
 * @ingroup group_x86
 * @file
 * @brief [en] Parity for the whole x86 (SSE2) tier, one case per census cell.
 * @brief [zh] x86（SSE2）整档的 parity，一个 census 格一个用例。
 * @note [en] Unlike the NEON and RVV tiers, these cases actually run in `conan create`:
 *            SSE2 is baseline on x86-64, so the host build carries the tier. That is the point
 *            of adding it -- the two earlier vector tiers are only ever verified by a probe on
 *            another machine, and a tier verified elsewhere is a tier this suite cannot vouch for. The per-kernel criterion still differs
 *            inside the file, and it is stated per case: four cells are bitwise (one IEEE
 *            operation per element, nothing accumulates across elements) and two are judged
 *            with the float window (a reduction carries one running sum in the reference and
 *            one partial sum per lane here).
 * @note [zh] 整档一个文件而不是一内核一个文件：六格是一次增量填的，且共享同一个问题——向量体与
 *            参考实现在同一目标上是否一致。判据在文件内部仍按格分别写明：四格逐位相等（每元素一次
 *            IEEE 运算、没有跨元素的累加），两格用浮点窗口（归约在参考里只有一条流水和，这里每
 *            车道一条部分和）。
 * @since 0.1.0
 */

namespace {

constexpr std::uint32_t kMaxLength = 257U;

// Deterministic fill: an integer LCG keeps the values identical on every target.
void fill_corpus(std::vector<float>& a, std::vector<float>& b) {
    a.resize(kMaxLength);
    b.resize(kMaxLength);
    for (std::size_t i = 0; i < a.size(); ++i) {
        const std::uint32_t state = (static_cast<std::uint32_t>(i) * 1664525U) + 1013904223U;
        a[i] = static_cast<float>(static_cast<std::int32_t>((state >> 8) % 20001U) - 10000) / 100.0F;
        b[i] = -a[i] / 3.0F;
    }
}

const infr_kernel_desc* tier_row(const char* name, infr_tier tier) {
    return infr_kernel_find(name, INFR_DTYPE_F32, INFR_QUANT_NONE, tier);
}

// Registers both tiers and returns the pair, or nullptr with a skip when this build has no
// vector extension -- the RVV row is a capability of the build, so its absence is a skip
// and not a failure.
struct Pair {
    const infr_kernel_desc* scalar;
    const infr_kernel_desc* rvv;
};

// Reports whether this build has the tier, and fills both rows. The skip itself has to be
// issued by the caller: `GTEST_SKIP()` expands to a `return`, so it only compiles inside a
// void function -- a value-returning helper cannot carry it.
bool resolve_pair(const char* name, Pair* out) {
    EXPECT_EQ(INFR_OK, infr_kernel_register_scalar());
    EXPECT_EQ(INFR_OK, infr_kernel_register_x86());
    out->scalar = tier_row(name, INFR_TIER_SCALAR);
    out->rvv = tier_row(name, INFR_TIER_X86);
    EXPECT_NE(nullptr, out->scalar) << name << "/fp32/none/scalar is not registered";
    return out->rvv != nullptr;
}

constexpr const char* kNoVector = "no SSE2 in this build: the x86 row is absent by design";

const std::uint32_t kLengths[] = {0U, 1U, 3U, 4U, 5U, 7U, 16U, 17U, 64U, 257U};

}  // namespace

TEST(kernel_parity, add_x86_fp32) {
    Pair pair = {};
    if (!resolve_pair("add", &pair)) {
        GTEST_SKIP() << kNoVector;
    }
    std::vector<float> a;
    std::vector<float> b;
    fill_corpus(a, b);
    a[0] = std::numeric_limits<float>::infinity(); /* infinity absorbs a finite addend */
    b[0] = -1.0F;
    a[1] = -0.0F; /* -0.0 + +0.0 must be +0.0 */
    b[1] = 0.0F;

    for (std::uint32_t n : kLengths) {
        // At least one slot: `std::vector<float>(0).data()` is nullptr, the kernel then
        // refuses the call by contract, and the assertion would fail for a reason that has
        // nothing to do with the tier under test. The length passed stays n.
        std::vector<float> want(n == 0U ? 1U : n, 0.0F);
        std::vector<float> got(n == 0U ? 1U : n, 0.0F);
        const void* inputs[2] = {a.data(), b.data()};
        void* want_out[1] = {want.data()};
        void* got_out[1] = {got.data()};
        ASSERT_EQ(INFR_OK, pair.scalar->fn(inputs, want_out, n, nullptr));
        ASSERT_EQ(INFR_OK, pair.rvv->fn(inputs, got_out, n, nullptr));
        for (std::uint32_t i = 0; i < n; ++i) {
            EXPECT_EQ(infr_test::float_bits(want[i]), infr_test::float_bits(got[i]))
                << "element " << i << " of " << n;
        }
    }
}

TEST(kernel_parity, scale_x86_fp32) {
    Pair pair = {};
    if (!resolve_pair("scale", &pair)) {
        GTEST_SKIP() << kNoVector;
    }
    std::vector<float> a;
    std::vector<float> b;
    fill_corpus(a, b);
    a[0] = std::numeric_limits<float>::infinity(); /* zero factor must give NaN, not 0 */
    a[1] = -0.0F;

    const float factors[] = {2.5F, 0.0F, -1.0F};
    for (float factor : factors) {
        for (std::uint32_t n : kLengths) {
            std::vector<float> want(n == 0U ? 1U : n, 0.0F);
            std::vector<float> got(n == 0U ? 1U : n, 0.0F);
            const void* inputs[2] = {a.data(), &factor};
            void* want_out[1] = {want.data()};
            void* got_out[1] = {got.data()};
            ASSERT_EQ(INFR_OK, pair.scalar->fn(inputs, want_out, n, nullptr));
            ASSERT_EQ(INFR_OK, pair.rvv->fn(inputs, got_out, n, nullptr));
            for (std::uint32_t i = 0; i < n; ++i) {
                EXPECT_EQ(infr_test::float_bits(want[i]), infr_test::float_bits(got[i]))
                    << "element " << i << " of " << n << " with factor " << factor;
            }
        }
    }

    // Both bodies owe the caller the same refusal for a missing factor.
    std::vector<float> out(4, 0.0F);
    const void* no_factor[2] = {a.data(), nullptr};
    void* out_ptr[1] = {out.data()};
    EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, pair.rvv->fn(no_factor, out_ptr, 4U, nullptr));
    EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, pair.scalar->fn(no_factor, out_ptr, 4U, nullptr));
}

TEST(kernel_parity, relu_x86_fp32) {
    Pair pair = {};
    if (!resolve_pair("relu", &pair)) {
        GTEST_SKIP() << kNoVector;
    }
    std::vector<float> a;
    std::vector<float> b;
    fill_corpus(a, b);
    // The one input class a "maximum" spelling cannot express: NaN and -0.0 must both
    // become +0.0, exactly as the reference's `x > 0 ? x : 0` does.
    a[0] = std::numeric_limits<float>::quiet_NaN();
    a[1] = -0.0F;
    a[2] = std::numeric_limits<float>::infinity();

    for (std::uint32_t n : kLengths) {
        std::vector<float> want(n == 0U ? 1U : n, 0.0F);
        std::vector<float> got(n == 0U ? 1U : n, 0.0F);
        const void* inputs[1] = {a.data()};
        void* want_out[1] = {want.data()};
        void* got_out[1] = {got.data()};
        ASSERT_EQ(INFR_OK, pair.scalar->fn(inputs, want_out, n, nullptr));
        ASSERT_EQ(INFR_OK, pair.rvv->fn(inputs, got_out, n, nullptr));
        for (std::uint32_t i = 0; i < n; ++i) {
            EXPECT_EQ(infr_test::float_bits(want[i]), infr_test::float_bits(got[i]))
                << "element " << i << " of " << n;
        }
    }
}

TEST(kernel_parity, max_x86_fp32) {
    Pair pair = {};
    if (!resolve_pair("max", &pair)) {
        GTEST_SKIP() << kNoVector;
    }
    std::vector<float> a;
    std::vector<float> b;
    fill_corpus(a, b);
    a[0] = std::numeric_limits<float>::quiet_NaN(); /* a NaN never wins */

    for (std::uint32_t n : kLengths) {
        if (n == 0U) {
            continue; /* the empty case is the refusal below, not a value */
        }
        float want = 0.0F;
        float got = 0.0F;
        const void* inputs[1] = {a.data()};
        void* want_out[1] = {&want};
        void* got_out[1] = {&got};
        ASSERT_EQ(INFR_OK, pair.scalar->fn(inputs, want_out, n, nullptr));
        ASSERT_EQ(INFR_OK, pair.rvv->fn(inputs, got_out, n, nullptr));
        EXPECT_EQ(infr_test::float_bits(want), infr_test::float_bits(got)) << "length " << n;
    }

    float out = 0.0F;
    const void* inputs[1] = {a.data()};
    void* out_ptr[1] = {&out};
    EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, pair.rvv->fn(inputs, out_ptr, 0U, nullptr));
    EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, pair.scalar->fn(inputs, out_ptr, 0U, nullptr));
}

TEST(kernel_parity, fc_x86_fp32_window) {
    Pair pair = {};
    if (!resolve_pair("fc", &pair)) {
        GTEST_SKIP() << kNoVector;
    }
    std::vector<float> a;
    std::vector<float> b;
    fill_corpus(a, b);

    for (std::uint32_t n : kLengths) {
        float want = 1234.0F;
        float got = 1234.0F;
        const void* inputs[2] = {a.data(), b.data()};
        void* want_out[1] = {&want};
        void* got_out[1] = {&got};
        ASSERT_EQ(INFR_OK, pair.scalar->fn(inputs, want_out, n, nullptr));
        ASSERT_EQ(INFR_OK, pair.rvv->fn(inputs, got_out, n, nullptr));
        if (n == 0U) {
            EXPECT_EQ(infr_test::float_bits(0.0F), infr_test::float_bits(got));
        }
        infr_test::expect_float_buffers_near(&want, &got, 1U, "fc");
    }
}

TEST(kernel_parity, sum_x86_fp32_window) {
    Pair pair = {};
    if (!resolve_pair("sum", &pair)) {
        GTEST_SKIP() << kNoVector;
    }
    std::vector<float> a;
    std::vector<float> b;
    fill_corpus(a, b);

    for (std::uint32_t n : kLengths) {
        float want = 1234.0F;
        float got = 1234.0F;
        const void* inputs[1] = {a.data()};
        void* want_out[1] = {&want};
        void* got_out[1] = {&got};
        ASSERT_EQ(INFR_OK, pair.scalar->fn(inputs, want_out, n, nullptr));
        ASSERT_EQ(INFR_OK, pair.rvv->fn(inputs, got_out, n, nullptr));
        if (n == 0U) {
            EXPECT_EQ(infr_test::float_bits(0.0F), infr_test::float_bits(got));
        }
        infr_test::expect_float_buffers_near(&want, &got, 1U, "sum");
    }
}

TEST(kernel_parity, relu_x86_i8) {
    EXPECT_EQ(INFR_OK, infr_kernel_register_scalar());
    EXPECT_EQ(INFR_OK, infr_kernel_register_x86());
    const infr_kernel_desc* scalar =
        infr_kernel_find("relu", INFR_DTYPE_I8, INFR_QUANT_NONE, INFR_TIER_SCALAR);
    const infr_kernel_desc* row =
        infr_kernel_find("relu", INFR_DTYPE_I8, INFR_QUANT_NONE, INFR_TIER_X86);
    ASSERT_NE(nullptr, scalar) << "relu/int8/none/scalar is not registered";
    if (row == nullptr) {
        GTEST_SKIP() << kNoVector;
    }

    // int8 relu is a pure integer maximum: no NaN, no negative zero, so this is judged
    // bitwise. The edges worth pinning are the ones a signed-byte comparison can get
    // wrong: INT8_MIN (the value that overflows if a body widens to int and negates),
    // -1 (must become 0, not stay), 0 (stays 0) and INT8_MAX.
    const std::uint32_t lengths[] = {0U, 1U, 3U, 15U, 16U, 17U, 32U, 33U, 64U};
    std::vector<std::int8_t> in(64);
    for (std::size_t i = 0; i < in.size(); ++i) {
        in[i] = static_cast<std::int8_t>(static_cast<int>((i * 37U) % 255U) - 127);
    }
    in[0] = INT8_MIN;
    in[1] = -1;
    in[2] = 0;
    in[3] = INT8_MAX;

    for (std::uint32_t n : lengths) {
        std::vector<std::int8_t> want(n == 0U ? 1U : n, 0);
        std::vector<std::int8_t> got(n == 0U ? 1U : n, 0);
        const void* inputs[1] = {in.data()};
        void* want_out[1] = {want.data()};
        void* got_out[1] = {got.data()};
        ASSERT_EQ(INFR_OK, scalar->fn(inputs, want_out, n, nullptr));
        ASSERT_EQ(INFR_OK, row->fn(inputs, got_out, n, nullptr));
        for (std::uint32_t i = 0; i < n; ++i) {
            EXPECT_EQ(static_cast<int>(want[i]), static_cast<int>(got[i]))
                << "element " << i << " of " << n;
        }
    }

    std::int8_t out = 0;
    const void* no_input[1] = {nullptr};
    void* out_ptr[1] = {&out};
    EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, row->fn(no_input, out_ptr, 4U, nullptr));
    EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, scalar->fn(no_input, out_ptr, 4U, nullptr));
}

TEST(kernel_parity, fc_x86_i8) {
    EXPECT_EQ(INFR_OK, infr_kernel_register_scalar());
    EXPECT_EQ(INFR_OK, infr_kernel_register_x86());
    const infr_kernel_desc* scalar =
        infr_kernel_find("fc", INFR_DTYPE_I8, INFR_QUANT_PER_TENSOR, INFR_TIER_SCALAR);
    const infr_kernel_desc* row =
        infr_kernel_find("fc", INFR_DTYPE_I8, INFR_QUANT_PER_TENSOR, INFR_TIER_X86);
    ASSERT_NE(nullptr, scalar) << "fc/int8/tensor/scalar is not registered";
    if (row == nullptr) {
        GTEST_SKIP() << kNoVector;
    }

    // Bitwise, and the reason is worth stating because the summation ORDER differs: the
    // reference accumulates in one int32 from left to right, this widens bytes to int16 and
    // folds with _mm_madd_epi16 into four int32 lanes. Integer addition is associative, so the
    // totals are equal exactly rather than approximately -- the same argument the DSP tier
    // rests on. The requantisation tail is not re-implemented either: both call the reference's
    // own clamp, so a rounding drift has nowhere to come from.
    const std::uint32_t lengths[] = {0U, 1U, 3U, 15U, 16U, 17U, 32U, 33U, 64U, 255U};
    std::vector<std::int8_t> x(255);
    std::vector<std::int8_t> w(255);
    for (std::size_t i = 0; i < x.size(); ++i) {
        x[i] = static_cast<std::int8_t>(static_cast<int>((i * 37U) % 255U) - 127);
        w[i] = static_cast<std::int8_t>(static_cast<int>((i * 53U + 11U) % 255U) - 127);
    }
    x[0] = INT8_MIN; /* the value that breaks a widening done with a plain zero-fill */
    w[0] = INT8_MIN;
    x[1] = -1;
    w[1] = 127;
    x[2] = 0;
    w[2] = INT8_MIN;

    infr_quant quant{};
    quant.kind = INFR_QUANT_PER_TENSOR;
    quant.scale = 1.0F;
    quant.zero_point = -7;
    quant.requant_multiplier = 0.0035F; /* non-trivial: exercises the double rounding path */

    const void* inputs[2] = {x.data(), w.data()};
    for (std::uint32_t n : lengths) {
        std::int8_t want = 42;
        std::int8_t got = 42;
        void* want_out[1] = {&want};
        void* got_out[1] = {&got};
        ASSERT_EQ(INFR_OK, scalar->fn(inputs, want_out, n, &quant));
        ASSERT_EQ(INFR_OK, row->fn(inputs, got_out, n, &quant));
        EXPECT_EQ(static_cast<int>(want), static_cast<int>(got)) << "length " << n;
    }

    // Both bodies owe the caller the same refusals: no quant at all, and an unresolved one.
    std::int8_t out = 0;
    void* out_ptr[1] = {&out};
    EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, row->fn(inputs, out_ptr, 4U, nullptr));
    EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, scalar->fn(inputs, out_ptr, 4U, nullptr));
    infr_quant unresolved{};
    unresolved.requant_multiplier = 1.0F; /* kind still NONE */
    EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, row->fn(inputs, out_ptr, 4U, &unresolved));
    EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, scalar->fn(inputs, out_ptr, 4U, &unresolved));
}

TEST(kernel_parity, fc_x86_i16) {
    EXPECT_EQ(INFR_OK, infr_kernel_register_scalar());
    EXPECT_EQ(INFR_OK, infr_kernel_register_x86());
    const infr_kernel_desc* scalar =
        infr_kernel_find("fc", INFR_DTYPE_I16, INFR_QUANT_PER_TENSOR, INFR_TIER_SCALAR);
    const infr_kernel_desc* row =
        infr_kernel_find("fc", INFR_DTYPE_I16, INFR_QUANT_PER_TENSOR, INFR_TIER_X86);
    ASSERT_NE(nullptr, scalar) << "fc/int16/tensor/scalar is not registered";
    if (row == nullptr) {
        GTEST_SKIP() << kNoVector;
    }

    // The corpus exists to hit the one corner the obvious SSE2 form gets wrong: two lanes that
    // are both INT16_MIN squared sum to exactly 2^31, which wraps in `_mm_madd_epi16`'s paired
    // int32 result but not in the reference's int64 accumulator. If a future body switches to the
    // paired form for speed, this case is what fails.
    const std::uint32_t lengths[] = {0U, 1U, 3U, 7U, 8U, 9U, 16U, 17U, 64U, 511U};
    std::vector<std::int16_t> x(511);
    std::vector<std::int16_t> w(511);
    for (std::size_t i = 0; i < x.size(); ++i) {
        x[i] = static_cast<std::int16_t>(static_cast<int>((i * 733U) % 65535U) - 32767);
        w[i] = static_cast<std::int16_t>(static_cast<int>((i * 977U + 13U) % 65535U) - 32767);
    }
    x[0] = INT16_MIN; /* INT16_MIN * INT16_MIN twice is the wrap corner */
    w[0] = INT16_MIN;
    x[1] = INT16_MIN;
    w[1] = INT16_MIN;
    x[2] = INT16_MAX;
    w[2] = INT16_MIN;
    x[3] = -1;
    w[3] = 1;

    infr_quant quant{};
    quant.kind = INFR_QUANT_PER_TENSOR;
    quant.scale = 1.0F;
    quant.zero_point = 5;
    quant.requant_multiplier = 1e-6F; /* keeps the sum inside int16 after requantisation */

    const void* inputs[2] = {x.data(), w.data()};
    for (std::uint32_t n : lengths) {
        std::int16_t want = 42;
        std::int16_t got = 42;
        void* want_out[1] = {&want};
        void* got_out[1] = {&got};
        ASSERT_EQ(INFR_OK, scalar->fn(inputs, want_out, n, &quant));
        ASSERT_EQ(INFR_OK, row->fn(inputs, got_out, n, &quant));
        EXPECT_EQ(static_cast<int>(want), static_cast<int>(got)) << "length " << n;
    }

    std::int16_t out = 0;
    void* out_ptr[1] = {&out};
    EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, row->fn(inputs, out_ptr, 4U, nullptr));
    EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, scalar->fn(inputs, out_ptr, 4U, nullptr));
    infr_quant unresolved{};
    unresolved.requant_multiplier = 1.0F;
    EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, row->fn(inputs, out_ptr, 4U, &unresolved));
    EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, scalar->fn(inputs, out_ptr, 4U, &unresolved));
}
