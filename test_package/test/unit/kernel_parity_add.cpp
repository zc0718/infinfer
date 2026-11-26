// Conan::ImportStart
#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <vector>

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

TEST(KernelParityAddNeon, Fp32MatchesScalarBitwise) {
    EXPECT_EQ(INFR_OK, infr_kernel_register_scalar());
    EXPECT_EQ(INFR_OK, infr_kernel_register_arm_neon());
    const infr_kernel_desc* scalar = scalar_add();
    const infr_kernel_desc* neon =
        infr_kernel_find("add", INFR_DTYPE_F32, INFR_QUANT_NONE, INFR_TIER_ARM_NEON);
    ASSERT_NE(nullptr, scalar) << "add/fp32/none/scalar is not registered";
    if (neon == nullptr) {
        GTEST_SKIP() << "no NEON in this build: the arm_neon row is absent by design";
    }

    // Bitwise: one IEEE addition per element, no accumulator crossing elements, so
    // the lane width cannot change the answer. Lengths pick the vector loop's two
    // parities of the remainder, plus the two it never enters.
    const std::uint32_t lengths[] = {0U, 1U, 3U, 4U, 5U, 7U, 16U, 17U, 64U};
    std::vector<float> a(64);
    std::vector<float> b(64);
    for (std::size_t i = 0; i < a.size(); ++i) {
        const std::uint32_t state = (static_cast<std::uint32_t>(i) * 1664525U) + 1013904223U;
        a[i] = static_cast<float>(static_cast<std::int32_t>((state >> 8) % 2001U) - 1000) / 100.0F;
        b[i] = -a[i] / 3.0F;
    }
    // The edges: an infinity absorbing a finite value, and -0.0 + +0.0 which must be
    // +0.0 (IEEE round-to-nearest), not -0.0.
    a[0] = std::numeric_limits<float>::infinity();
    b[0] = -1.0F;
    a[1] = -0.0F;
    b[1] = 0.0F;

    for (std::uint32_t length : lengths) {
        // At least one slot: `std::vector<float>(0).data()` is nullptr, the kernel then
        // refuses the call by contract, and the assertion would fail for a reason that has
        // nothing to do with the tier under test. The length passed stays n.
        std::vector<float> want(length == 0U ? 1U : length, 0.0F);
        std::vector<float> got(length == 0U ? 1U : length, 0.0F);
        const void* inputs[2] = {a.data(), b.data()};
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
