// Census cells: relu/fp32/none/scalar and relu/int8/none/scalar.
//
// The scalar body is the definition of correctness (INV-2); this test pins it to
// the mathematical definition through the registry, and it is the fixture a later
// tier (autovec/neon/...) plugs into as `scalar vs candidate`.
#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include <gtest/gtest-spi.h> /* EXPECT_NONFATAL_FAILURE: the harness must be provably able to fail. */

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

// Cross-implementation differential: the kernel row and the fused-activation row
// are written in different TUs from the same definition; the parity harness must
// find them agreeing. This is the first true "candidate vs reference" comparison.
TEST(KernelParityRelu, KernelAgreesWithFusedActivation) {
    const infr_kernel_desc* row = scalar_relu(INFR_DTYPE_F32);
    ASSERT_NE(nullptr, row);
    const infr_act_f32_fn act = infr_activation_f32(INFR_ACT_RELU);
    ASSERT_NE(nullptr, act);

    const float in[9] = {-3.5F, -1.0F, -0.0F, 0.0F, 0.25F, 1.0F, 7.5F, -128.0F, 1000000.0F};
    float kernel_out[9] = {};
    float act_out[9] = {};
    const void* inputs[1] = {in};
    void* outputs[1] = {kernel_out};

    ASSERT_EQ(INFR_OK, row->fn(inputs, outputs, 9, nullptr));
    act(in, act_out, 9);
    infr_test::expect_float_buffers_near(act_out, kernel_out, 9, "relu kernel vs fused activation");
}

// The harness must be able to fail, or a green parity run means nothing. Both
// comparators are exercised against a deliberately drifted buffer and the
// failure message is matched by its own wording.
TEST(KernelParityRelu, HarnessDetectsDrift) {
    const float want_f[3] = {1.0F, 2.0F, 3.0F};
    const float same_f[3] = {1.0F, 2.0F, 3.0F};
    infr_test::expect_float_buffers_near(want_f, same_f, 3, "identical float");

    const float drifted_f[3] = {1.0F, 2.0F, 3.0009765625F}; /* 3 + 2^-10: far beyond 4 ULP. */
    EXPECT_NONFATAL_FAILURE(infr_test::expect_float_buffers_near(want_f, drifted_f, 3, "drifted float"),
                            "[drifted float] element 2");

    const int8_t want_i[3] = {1, 2, 3};
    const int8_t same_i[3] = {1, 2, 3};
    infr_test::expect_i8_buffers_equal(want_i, same_i, 3, "identical int8");

    const int8_t drifted_i[3] = {1, 2, 4};
    EXPECT_NONFATAL_FAILURE(infr_test::expect_i8_buffers_equal(want_i, drifted_i, 3, "drifted int8"),
                            "[drifted int8] element 2");
}

TEST(KernelParityReluNeon, Fp32MatchesScalarBitwise) {
    EXPECT_EQ(INFR_OK, infr_kernel_register_scalar());
    EXPECT_EQ(INFR_OK, infr_kernel_register_arm_neon());
    const infr_kernel_desc* scalar =
        infr_kernel_find("relu", INFR_DTYPE_F32, INFR_QUANT_NONE, INFR_TIER_SCALAR);
    const infr_kernel_desc* neon =
        infr_kernel_find("relu", INFR_DTYPE_F32, INFR_QUANT_NONE, INFR_TIER_ARM_NEON);
    ASSERT_NE(nullptr, scalar) << "relu/fp32/none/scalar is not registered";
    if (neon == nullptr) {
        GTEST_SKIP() << "no NEON in this build: the arm_neon row is absent by design";
    }

    // Bitwise, unlike the fp32 dot product: a select writes the same bits it read,
    // so the tolerance question does not arise. The values include the two edges a
    // `max` spelling would get wrong -- NaN (must become 0.0) and -0.0 (must become
    // +0.0, since `-0.0 > 0` is false).
    const std::uint32_t lengths[] = {0U, 1U, 3U, 4U, 5U, 7U, 16U, 17U, 64U};
    std::vector<float> in(64);
    for (std::size_t i = 0; i < in.size(); ++i) {
        // Deterministic LCG: the same bytes on every target, no library RNG.
        const std::uint32_t state = (static_cast<std::uint32_t>(i) * 1664525U) + 1013904223U;
        in[i] = static_cast<float>(static_cast<std::int32_t>((state >> 8) % 2001U) - 1000) / 100.0F;
    }
    in[0] = -0.0F;
    in[1] = std::numeric_limits<float>::quiet_NaN();
    in[2] = -1.5F;
    in[3] = 0.0F;

    for (std::uint32_t length : lengths) {
        // At least one slot: `std::vector<float>(0).data()` is nullptr, the kernel then
        // refuses the call by contract, and the assertion would fail for a reason that has
        // nothing to do with the tier under test. The length passed stays n.
        std::vector<float> want(length == 0U ? 1U : length, 0.0F);
        std::vector<float> got(length == 0U ? 1U : length, 0.0F);
        const void* inputs[1] = {in.data()};
        void* want_out[1] = {want.data()};
        void* got_out[1] = {got.data()};
        ASSERT_EQ(INFR_OK, scalar->fn(inputs, want_out, length, nullptr));
        ASSERT_EQ(INFR_OK, neon->fn(inputs, got_out, length, nullptr));
        for (std::uint32_t i = 0; i < length; ++i) {
            EXPECT_EQ(infr_test::float_bits(want[i]), infr_test::float_bits(got[i]))
                << "element " << i << " of " << length;
        }
    }
}
