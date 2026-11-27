// Conan::ImportStart
#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <vector>

#include <infinfer/infinfer.h>

#include "parity_harness.hpp"
// Conan::ImportEnd

/*!
 * @brief [en] The `sum` cell (census key `sum/fp32/none/scalar`): a shape-free reduction to one value.
 * @brief [zh] `sum` 格（census 键 `sum/fp32/none/scalar`）：不依赖 shape 的求和归约。
 * @since 0.1.0
 */

namespace {

const infr_kernel_desc* scalar_sum() {
    EXPECT_EQ(INFR_OK, infr_kernel_register_scalar());
    return infr_kernel_find("sum", INFR_DTYPE_F32, INFR_QUANT_NONE, INFR_TIER_SCALAR);
}

}  // namespace

TEST(KernelParitySum, ScalarFp32MatchesDoubleReference) {
    const infr_kernel_desc* row = scalar_sum();
    ASSERT_NE(nullptr, row) << "sum/fp32/none/scalar is not registered";

    const float in[6] = {0.5F, -1.25F, 2.0F, 1e-3F, -1e-3F, 4.0F};
    double want = 0.0;
    for (float value : in) {
        want += static_cast<double>(value);
    }
    const float want_f = static_cast<float>(want);

    float got = 1234.0F; /* Poisoned: a no-op body cannot pass. */
    const void* inputs[1] = {in};
    void* outputs[1] = {&got};

    ASSERT_EQ(INFR_OK, row->fn(inputs, outputs, 6, nullptr));
    infr_test::expect_float_buffers_near(&want_f, &got, 1, "sum fp32");
}

TEST(KernelParitySum, ZeroLengthIsAZeroSum) {
    // Global average pooling divides this by the length, so the empty case has to
    // be a defined 0.0 rather than an error the layer would have to special-case.
    const infr_kernel_desc* row = scalar_sum();
    ASSERT_NE(nullptr, row);
    const float in[1] = {7.0F};
    const void* inputs[1] = {in};
    float got = 99.0F;
    void* outputs[1] = {&got};

    ASSERT_EQ(INFR_OK, row->fn(inputs, outputs, 0, nullptr));
    EXPECT_EQ(0.0F, got);
}

TEST(KernelParitySum, RejectsMissingBuffers) {
    const infr_kernel_desc* row = scalar_sum();
    ASSERT_NE(nullptr, row);
    float got = 0.0F;
    void* outputs[1] = {&got};

    EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, row->fn(nullptr, outputs, 1, nullptr));
    const void* no_in[1] = {nullptr};
    EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, row->fn(no_in, outputs, 1, nullptr));
    const float in[1] = {1.0F};
    const void* inputs[1] = {in};
    void* no_output[1] = {nullptr};
    EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, row->fn(inputs, no_output, 1, nullptr));
}

TEST(KernelParitySumNeon, Fp32WithinTheFloatWindow) {
    EXPECT_EQ(INFR_OK, infr_kernel_register_scalar());
    EXPECT_EQ(INFR_OK, infr_kernel_register_arm_neon());
    const infr_kernel_desc* scalar = scalar_sum();
    const infr_kernel_desc* neon =
        infr_kernel_find("sum", INFR_DTYPE_F32, INFR_QUANT_NONE, INFR_TIER_ARM_NEON);
    ASSERT_NE(nullptr, scalar) << "sum/fp32/none/scalar is not registered";
    if (neon == nullptr) {
        GTEST_SKIP() << "no NEON in this build: the arm_neon row is absent by design";
    }

    // Not bitwise, and deliberately so: the reference accumulates in ONE float from
    // left to right, while four lanes keep four partial sums. The two bodies therefore
    // round differently by construction, so this is judged with the float window the
    // parity harness uses for every fp32 reduction -- the same call G-57 made for the
    // dot product, and the opposite of what the `add`/`scale`/`relu`/`max` rows claim.
    const std::uint32_t lengths[] = {0U, 1U, 3U, 4U, 5U, 7U, 64U, 257U};
    std::vector<float> in(257);
    for (std::size_t i = 0; i < in.size(); ++i) {
        const std::uint32_t state = (static_cast<std::uint32_t>(i) * 1103515245U) + 12345U;
        in[i] = static_cast<float>(static_cast<std::int32_t>((state >> 9) % 20001U) - 10000) / 100.0F;
    }

    for (std::uint32_t length : lengths) {
        float want = 1234.0F;
        float got = 1234.0F;
        const void* inputs[1] = {in.data()};
        void* want_out[1] = {&want};
        void* got_out[1] = {&got};
        ASSERT_EQ(INFR_OK, scalar->fn(inputs, want_out, length, nullptr));
        ASSERT_EQ(INFR_OK, neon->fn(inputs, got_out, length, nullptr));
        // Zero length is a zero sum in both bodies, never a refusal.
        if (length == 0U) {
            EXPECT_EQ(infr_test::float_bits(0.0F), infr_test::float_bits(got));
        }
        infr_test::expect_float_buffers_near(&want, &got, 1U, "sum");
    }

    // The refusal both bodies owe a caller, at the same call.
    float out = 0.0F;
    const void* no_input[1] = {nullptr};
    void* out_ptr[1] = {&out};
    EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, neon->fn(no_input, out_ptr, 4U, nullptr));
    EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, scalar->fn(no_input, out_ptr, 4U, nullptr));
}


int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
