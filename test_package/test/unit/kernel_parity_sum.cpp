// Conan::ImportStart
#include <gtest/gtest.h>

#include <cstdint>

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
