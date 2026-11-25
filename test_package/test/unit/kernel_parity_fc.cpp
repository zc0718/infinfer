// Conan::ImportStart
#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>

#include <infinfer/infinfer.h>

#include "parity_harness.hpp"
// Conan::ImportEnd

/*
 * The `fc` cell (census key `fc/fp32/none/scalar`). One kernel call is one dot
 * product because INV-3 keeps shape out of the signature; the row loop is the
 * graph layer's job, so these cases exercise exactly what L2 will see.
 *
 * `fc` 格（census 键 `fc/fp32/none/scalar`）。因为 INV-3 把 shape 挡在签名外，
 * 一次内核调用就是一次点积；逐行循环属于图层——所以这几个用例覆盖的正是 L2 会看到
 * 的东西。
 */

namespace {

// Deterministic fill: an integer LCG keeps the values identical on every target
// and avoids depending on any library's random generator.
float sample(std::uint32_t seed) {
    const std::uint32_t state = seed * 1664525U + 1013904223U;
    const std::int32_t centred = static_cast<std::int32_t>((state >> 8) % 2001U) - 1000;
    return static_cast<float>(centred) / 100.0F;
}

void fill(float* buffer, std::size_t count, std::uint32_t seed) {
    for (std::size_t i = 0; i < count; ++i) {
        buffer[i] = sample(seed + static_cast<std::uint32_t>(i));
    }
}

const infr_kernel_desc* scalar_fc() {
    EXPECT_EQ(INFR_OK, infr_kernel_register_scalar());
    return infr_kernel_find("fc", INFR_DTYPE_F32, INFR_QUANT_NONE, INFR_TIER_SCALAR);
}

}  // namespace

TEST(KernelParityFc, ScalarFp32MatchesDoubleReference) {
    const std::uint32_t lengths[] = {1U, 3U, 17U, 64U};
    const infr_kernel_desc* row = scalar_fc();
    ASSERT_NE(nullptr, row) << "fc/fp32/none/scalar is not registered";

    for (std::uint32_t length : lengths) {
        float x[64];
        float w[64];
        fill(x, length, 7U + length);
        fill(w, length, 91U + length);

        const void* inputs[2] = {x, w};
        float got = 0.0F;
        void* outputs[1] = {&got};
        ASSERT_EQ(INFR_OK, row->fn(inputs, outputs, length, nullptr));

        // The reference accumulates in double on purpose: comparing against the
        // same float loop would only prove the code equals itself.
        double want = 0.0;
        for (std::uint32_t i = 0; i < length; ++i) {
            want += static_cast<double>(x[i]) * static_cast<double>(w[i]);
        }
        const float want_f = static_cast<float>(want);
        infr_test::expect_float_buffers_near(&want_f, &got, 1, "fc dot product");
    }
}

TEST(KernelParityFc, RejectsMissingBuffers) {
    const infr_kernel_desc* row = scalar_fc();
    ASSERT_NE(nullptr, row);

    const float x[2] = {1.0F, 2.0F};
    const float w[2] = {3.0F, 4.0F};
    float got = 0.0F;
    void* outputs[1] = {&got};

    EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, row->fn(nullptr, outputs, 2, nullptr));
    const void* no_activation[2] = {nullptr, w};
    EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, row->fn(no_activation, outputs, 2, nullptr));
    const void* no_weights[2] = {x, nullptr};
    EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, row->fn(no_weights, outputs, 2, nullptr));
    const void* inputs[2] = {x, w};
    void* no_output[1] = {nullptr};
    EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, row->fn(inputs, no_output, 2, nullptr));
}

TEST(KernelParityFc, ZeroLengthIsAZeroReduction) {
    // A zero-width row is a legal reduction, not an error: the graph layer must
    // be able to hand one over while probing a model.
    const infr_kernel_desc* row = scalar_fc();
    ASSERT_NE(nullptr, row);

    const float x[1] = {1.0F};
    const float w[1] = {1.0F};
    const void* inputs[2] = {x, w};
    float got = 42.0F; /* Poisoned: a body that skips the store would keep it. */
    void* outputs[1] = {&got};

    ASSERT_EQ(INFR_OK, row->fn(inputs, outputs, 0, nullptr));
    EXPECT_EQ(0.0F, got);
}
