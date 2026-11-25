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
