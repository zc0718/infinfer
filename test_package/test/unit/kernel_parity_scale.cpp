// Conan::ImportStart
#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <vector>

#include <infinfer/infinfer.h>

#include "parity_harness.hpp"
// Conan::ImportEnd

/*!
 * @brief [en] The `scale` cell (census key `scale/fp32/none/scalar`): broadcast multiply by one factor.
 * @brief [zh] `scale` 格（census 键 `scale/fp32/none/scalar`）：乘以单个因子的广播缩放。
 * @since 0.1.0
 */

namespace {

const infr_kernel_desc* scalar_scale() {
    EXPECT_EQ(INFR_OK, infr_kernel_register_scalar());
    return infr_kernel_find("scale", INFR_DTYPE_F32, INFR_QUANT_NONE, INFR_TIER_SCALAR);
}

}  // namespace

TEST(KernelParityScale, ScalarFp32MatchesDoubleReference) {
    const infr_kernel_desc* row = scalar_scale();
    ASSERT_NE(nullptr, row) << "scale/fp32/none/scalar is not registered";

    const float in[5] = {-2.0F, -0.5F, 0.0F, 1.5F, 7.25F};
    const float factor = -0.375F;
    float want[5];
    for (int i = 0; i < 5; ++i) {
        want[i] = static_cast<float>(static_cast<double>(in[i]) * static_cast<double>(factor));
    }
    float got[5] = {};
    const void* inputs[2] = {in, &factor};
    void* outputs[1] = {got};

    ASSERT_EQ(INFR_OK, row->fn(inputs, outputs, 5, nullptr));
    infr_test::expect_float_buffers_near(want, got, 5, "scale fp32");
}

TEST(KernelParityScale, FactorIsReadOnceSoAliasingIsSafe) {
    /* A mean over a buffer that also holds the factor must not see the factor change
     * underneath it: the value read before the loop is the value used. */
    infr_kernel_desc row{};
    ASSERT_NE(nullptr, scalar_scale());
    row = *scalar_scale();

    float buffer[3] = {1.0F, 2.0F, 0.5F}; /* buffer[2] is the factor */
    const void* inputs[2] = {buffer, &buffer[2]};
    void* outputs[1] = {buffer};

    ASSERT_EQ(INFR_OK, row.fn(inputs, outputs, 3, nullptr));
    const float want[3] = {0.5F, 1.0F, 0.25F};
    infr_test::expect_float_buffers_near(want, buffer, 3, "scale aliased");
}

TEST(KernelParityScale, RejectsMissingBuffers) {
    const infr_kernel_desc* row = scalar_scale();
    ASSERT_NE(nullptr, row);
    const float in[2] = {1.0F, 2.0F};
    const float factor = 2.0F;
    float got[2] = {};
    void* outputs[1] = {got};

    EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, row->fn(nullptr, outputs, 2, nullptr));
    const void* no_factor[2] = {in, nullptr};
    EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, row->fn(no_factor, outputs, 2, nullptr));
    const void* inputs[2] = {in, &factor};
    void* no_output[1] = {nullptr};
    EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, row->fn(inputs, no_output, 2, nullptr));
}

TEST(KernelParityScaleNeon, Fp32MatchesScalarBitwise) {
    EXPECT_EQ(INFR_OK, infr_kernel_register_scalar());
    EXPECT_EQ(INFR_OK, infr_kernel_register_arm_neon());
    const infr_kernel_desc* scalar = scalar_scale();
    const infr_kernel_desc* neon =
        infr_kernel_find("scale", INFR_DTYPE_F32, INFR_QUANT_NONE, INFR_TIER_ARM_NEON);
    ASSERT_NE(nullptr, scalar) << "scale/fp32/none/scalar is not registered";
    if (neon == nullptr) {
        GTEST_SKIP() << "no NEON in this build: the arm_neon row is absent by design";
    }

    const std::uint32_t lengths[] = {0U, 1U, 3U, 4U, 5U, 7U, 16U, 17U, 64U};
    std::vector<float> in(64);
    for (std::size_t i = 0; i < in.size(); ++i) {
        const std::uint32_t state = (static_cast<std::uint32_t>(i) * 22695477U) + 1U;
        in[i] = static_cast<float>(static_cast<std::int32_t>((state >> 8) % 2001U) - 1000) / 100.0F;
    }
    in[0] = -0.0F;
    in[1] = std::numeric_limits<float>::infinity();

    // Three factors worth pinning: a plain value, zero (which must turn the infinity
    // into a NaN in both bodies, not into 0), and a negative one.
    const float factors[] = {2.5F, 0.0F, -1.0F};
    for (float factor : factors) {
        for (std::uint32_t length : lengths) {
            std::vector<float> want(length, 0.0F);
            std::vector<float> got(length, 0.0F);
            const void* inputs[2] = {in.data(), &factor};
            void* want_out[1] = {want.data()};
            void* got_out[1] = {got.data()};
            ASSERT_EQ(INFR_OK, scalar->fn(inputs, want_out, length, nullptr));
            ASSERT_EQ(INFR_OK, neon->fn(inputs, got_out, length, nullptr));
            for (std::uint32_t i = 0; i < length; ++i) {
                EXPECT_EQ(infr_test::float_bits(want[i]), infr_test::float_bits(got[i]))
                    << "element " << i << " of " << length << " with factor " << factor;
            }
        }
    }

    // The refusal both bodies owe a caller: no factor at all is an argument error,
    // even for an empty buffer.
    std::vector<float> out(4, 0.0F);
    const void* no_factor[2] = {in.data(), nullptr};
    void* out_ptr[1] = {out.data()};
    EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, neon->fn(no_factor, out_ptr, 4U, nullptr));
    EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, scalar->fn(no_factor, out_ptr, 4U, nullptr));
}
