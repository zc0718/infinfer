// Conan::ImportStart
#include <gtest/gtest.h>

#include <cstdint>

#include <infinfer/infinfer.h>

#include "parity_harness.hpp"
// Conan::ImportEnd

/*!
 * @brief [en] The `max` cell (census key `max/fp32/none/scalar`): the reduction a max-pool window uses.
 * @brief [zh] `max` 格（census 键 `max/fp32/none/scalar`）：最大池化窗口所用的归约。
 * @since 0.1.0
 */

namespace {

const infr_kernel_desc* scalar_max() {
    EXPECT_EQ(INFR_OK, infr_kernel_register_scalar());
    return infr_kernel_find("max", INFR_DTYPE_F32, INFR_QUANT_NONE, INFR_TIER_SCALAR);
}

}  // namespace

TEST(KernelParityMax, ScalarFp32MatchesDefinition) {
    const infr_kernel_desc* row = scalar_max();
    ASSERT_NE(nullptr, row) << "max/fp32/none/scalar is not registered";

    const float in[6] = {-3.0F, -1.5F, -7.0F, -1.5F, -9.0F, -2.0F};
    const float want = -1.5F;
    float got = 0.0F; /* Poisoned: 0.0 is not in the set, so a no-op cannot pass. */
    const void* inputs[1] = {in};
    void* outputs[1] = {&got};

    ASSERT_EQ(INFR_OK, row->fn(inputs, outputs, 6, nullptr));
    infr_test::expect_float_buffers_near(&want, &got, 1, "max fp32");
}

TEST(KernelParityMax, FirstOccurrenceWinsOnTies) {
    // Two equal maxima must not make the result depend on scan direction; the
    // reference takes the first one, so a NaN-free tie is stable.
    const infr_kernel_desc* row = scalar_max();
    ASSERT_NE(nullptr, row);
    const float in[4] = {5.0F, 5.0F, -1.0F, 2.0F};
    float got = 0.0F;
    const void* inputs[1] = {in};
    void* outputs[1] = {&got};

    ASSERT_EQ(INFR_OK, row->fn(inputs, outputs, 4, nullptr));
    EXPECT_EQ(5.0F, got);
}

TEST(KernelParityMax, RejectsEmptyWindowAndMissingBuffers) {
    const infr_kernel_desc* row = scalar_max();
    ASSERT_NE(nullptr, row);
    const float in[2] = {1.0F, 2.0F};
    const void* inputs[1] = {in};
    float got = 0.0F;
    void* outputs[1] = {&got};

    // A max over an empty set has no value: an error, not a silent -infinity that
    // would poison the next layer.
    EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, row->fn(inputs, outputs, 0, nullptr));
    EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, row->fn(nullptr, outputs, 2, nullptr));
    const void* no_in[1] = {nullptr};
    EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, row->fn(no_in, outputs, 2, nullptr));
    void* no_output[1] = {nullptr};
    EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, row->fn(inputs, no_output, 2, nullptr));
}

TEST(KernelParityMaxNeon, Fp32MatchesScalarBitwise) {
    EXPECT_EQ(INFR_OK, infr_kernel_register_scalar());
    EXPECT_EQ(INFR_OK, infr_kernel_register_arm_neon());
    const infr_kernel_desc* scalar = scalar_max();
    const infr_kernel_desc* neon =
        infr_kernel_find("max", INFR_DTYPE_F32, INFR_QUANT_NONE, INFR_TIER_ARM_NEON);
    ASSERT_NE(nullptr, scalar) << "max/fp32/none/scalar is not registered";
    if (neon == nullptr) {
        GTEST_SKIP() << "no NEON in this build: the arm_neon row is absent by design";
    }

    // Bitwise: a maximum moves bits, it does not compute with them. Lengths 0 and 1
    // are the two the vector loop never enters, and 5/7/16/17 push the tail and the
    // lane fold through both parities of the remainder.
    const std::uint32_t lengths[] = {1U, 2U, 3U, 4U, 5U, 7U, 16U, 17U, 64U, 257U};
    std::vector<float> in(257);
    for (std::size_t i = 0; i < in.size(); ++i) {
        const std::uint32_t state = (static_cast<std::uint32_t>(i) * 1664525U) + 1013904223U;
        in[i] = static_cast<float>(static_cast<std::int32_t>((state >> 8) % 2001U) - 1000) / 100.0F;
    }
    // A tie at the maximum: both bodies must return the same one, and a NaN must
    // never win (the reference compares with `>`, which is false for NaN).
    in[2] = 9.5F;
    in[9] = 9.5F;
    in[1] = std::numeric_limits<float>::quiet_NaN();

    for (std::uint32_t length : lengths) {
        const void* inputs[1] = {in.data()};
        float want = 0.0F;
        float got = 0.0F;
        void* want_out[1] = {&want};
        void* got_out[1] = {&got};
        ASSERT_EQ(INFR_OK, scalar->fn(inputs, want_out, length, nullptr));
        ASSERT_EQ(INFR_OK, neon->fn(inputs, got_out, length, nullptr));
        EXPECT_EQ(infr_test::float_bits(want), infr_test::float_bits(got)) << "length " << length;
    }

    // An empty reduction is refused by both, not invented.
    const void* inputs[1] = {in.data()};
    float got = 0.0F;
    void* got_out[1] = {&got};
    EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, neon->fn(inputs, got_out, 0U, nullptr));
    EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, scalar->fn(inputs, got_out, 0U, nullptr));
}
