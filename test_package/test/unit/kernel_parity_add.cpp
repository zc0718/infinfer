// Conan::ImportStart
#include <gtest/gtest.h>

#include <cstdint>

#include <infinfer/infinfer.h>

#include "parity_harness.hpp"
// Conan::ImportEnd

/*!
 * @brief [en] The `add` cell (census key `add/fp32/none/scalar`): elementwise binary add.
 * @brief [zh] `add` 格（census 键 `add/fp32/none/scalar`）：逐元素二元加法。
 * @since 0.1.0
 */

namespace {

const infr_kernel_desc* scalar_add() {
    EXPECT_EQ(INFR_OK, infr_kernel_register_scalar());
    return infr_kernel_find("add", INFR_DTYPE_F32, INFR_QUANT_NONE, INFR_TIER_SCALAR);
}

}  // namespace

TEST(KernelParityAdd, ScalarFp32MatchesDoubleReference) {
    const infr_kernel_desc* row = scalar_add();
    ASSERT_NE(nullptr, row) << "add/fp32/none/scalar is not registered";

    const float a[5] = {1.5F, -2.25F, 0.0F, 3.75F, -1e-3F};
    const float b[5] = {-0.5F, 0.25F, 0.0F, -3.75F, 1e-3F};
    float want[5];
    for (int i = 0; i < 5; ++i) {
        want[i] = static_cast<float>(static_cast<double>(a[i]) + static_cast<double>(b[i]));
    }
    float got[5] = {};
    const void* inputs[2] = {a, b};
    void* outputs[1] = {got};

    ASSERT_EQ(INFR_OK, row->fn(inputs, outputs, 5, nullptr));
    infr_test::expect_float_buffers_near(want, got, 5, "add fp32");
}

TEST(KernelParityAdd, AliasingOutputOverInputIsAllowed) {
    // The graph layer does in-place bias adds on the workspace, so y == a has to
    // be a defined input, not an accident that happens to work.
    const infr_kernel_desc* row = scalar_add();
    ASSERT_NE(nullptr, row);
    float a[3] = {1.0F, 2.0F, 3.0F};
    const float b[3] = {0.5F, 0.5F, 0.5F};
    const void* inputs[2] = {a, b};
    void* outputs[1] = {a};

    ASSERT_EQ(INFR_OK, row->fn(inputs, outputs, 3, nullptr));
    const float want[3] = {1.5F, 2.5F, 3.5F};
    infr_test::expect_float_buffers_near(want, a, 3, "add aliased");
}

TEST(KernelParityAdd, RejectsMissingBuffers) {
    const infr_kernel_desc* row = scalar_add();
    ASSERT_NE(nullptr, row);
    const float a[2] = {1.0F, 2.0F};
    float got[2] = {};
    void* outputs[1] = {got};

    EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, row->fn(nullptr, outputs, 2, nullptr));
    const void* no_b[2] = {a, nullptr};
    EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, row->fn(no_b, outputs, 2, nullptr));
    const void* inputs[2] = {a, a};
    void* no_output[1] = {nullptr};
    EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, row->fn(inputs, no_output, 2, nullptr));
}
