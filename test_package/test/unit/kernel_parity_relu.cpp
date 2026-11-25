// Census cells: relu/fp32/none/scalar and relu/int8/none/scalar.
//
// The scalar body is the definition of correctness (INV-2); this test pins it to
// the mathematical definition through the registry, and it is the fixture a later
// tier (autovec/neon/...) plugs into as `scalar vs candidate`.
#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>

#include <infinfer/infinfer.h>

#include "parity_harness.hpp"

namespace {

const infr_kernel_desc* scalar_relu(infr_dtype dtype) {
    EXPECT_EQ(INFR_OK, infr_kernel_register_scalar());
    return infr_kernel_find("relu", dtype, INFR_QUANT_NONE, INFR_TIER_SCALAR);
}

} // namespace

TEST(KernelParityRelu, ScalarFp32MatchesDefinition) {
    const infr_kernel_desc* row = scalar_relu(INFR_DTYPE_F32);
    ASSERT_NE(nullptr, row) << "relu/fp32/none/scalar is not registered";

    const float in[6] = {-2.0F, -0.5F, 0.0F, 0.5F, 3.0F, 1000.0F};
    const float want[6] = {0.0F, 0.0F, 0.0F, 0.5F, 3.0F, 1000.0F};
    float got[6] = {};
    const void* inputs[1] = {in};
    void* outputs[1] = {got};

    ASSERT_EQ(INFR_OK, row->fn(inputs, outputs, 6, nullptr));
    infr_test::expect_float_buffers_near(want, got, 6, "relu fp32");
}

TEST(KernelParityRelu, ScalarInt8MatchesDefinition) {
    const infr_kernel_desc* row = scalar_relu(INFR_DTYPE_I8);
    ASSERT_NE(nullptr, row) << "relu/int8/none/scalar is not registered";

    const int8_t in[5] = {-128, -1, 0, 1, 127};
    const int8_t want[5] = {0, 0, 0, 1, 127};
    int8_t got[5] = {1, 1, 1, 1, 1}; /* Poisoned so a no-op body cannot pass. */
    const void* inputs[1] = {in};
    void* outputs[1] = {got};

    ASSERT_EQ(INFR_OK, row->fn(inputs, outputs, 5, nullptr));
    infr_test::expect_i8_buffers_equal(want, got, 5, "relu int8");
}

TEST(KernelParityRelu, RejectsMissingBuffers) {
    const infr_kernel_desc* row = scalar_relu(INFR_DTYPE_F32);
    ASSERT_NE(nullptr, row);
    void* outputs[1] = {nullptr};
    EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, row->fn(nullptr, outputs, 1, nullptr));
    const void* inputs[1] = {nullptr};
    EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, row->fn(inputs, outputs, 1, nullptr));
}
