// Census cells: quantize/fp32/tensor/scalar and dequantize/int8/tensor/scalar.
//
// The pair is the gateway between the float world and the int8 world, so it is
// pinned the same way every other scalar body is (INV-2): through the registry,
// against the written definition, in both directions. Two details decide whether
// a quantized graph is reproducible, and both are asserted here rather than left
// to the comment in quant.c: the rounding rule (half away from zero) and the
// clamp to the int8 range. A round trip is also checked, because "quantize then
// dequantize" is the operation the graph layer actually performs when it feeds a
// float tensor to a quantized kernel.
#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <vector>

#include <infinfer/infinfer.h>

#include "parity_harness.hpp"

namespace {

const infr_kernel_desc* scalar_quantize() {
    EXPECT_EQ(INFR_OK, infr_kernel_register_scalar());
    return infr_kernel_find("quantize", INFR_DTYPE_F32, INFR_QUANT_PER_TENSOR, INFR_TIER_SCALAR);
}

const infr_kernel_desc* scalar_dequantize() {
    EXPECT_EQ(INFR_OK, infr_kernel_register_scalar());
    return infr_kernel_find("dequantize", INFR_DTYPE_I8, INFR_QUANT_PER_TENSOR, INFR_TIER_SCALAR);
}

infr_quant per_tensor(float scale, int32_t zero_point) {
    infr_quant quant{};
    quant.kind = INFR_QUANT_PER_TENSOR;
    quant.scale = scale;
    quant.zero_point = zero_point;
    quant.channel_count = 1U;
    return quant;
}

} // namespace

TEST(KernelParityQuant, ScalarQuantizeMatchesDefinition) {
    const infr_kernel_desc* row = scalar_quantize();
    ASSERT_NE(nullptr, row) << "quantize/fp32/tensor/scalar is not registered";

    // scale 0.5, zero point -128: q = round(x / 0.5) - 128, clamped to [-128, 127].
    const float in[6] = {0.0F, 1.0F, -1.0F, 2.5F, -2.5F, 100.0F};
    const int8_t want[6] = {-128, -126, -128, -123, -128, 72};
    int8_t got[6] = {};
    const infr_quant quant = per_tensor(0.5F, -128);
    const void* inputs[1] = {in};
    void* outputs[1] = {got};

    ASSERT_EQ(INFR_OK, row->fn(inputs, outputs, 6, &quant));
    for (int i = 0; i < 6; ++i) {
        EXPECT_EQ(static_cast<int>(want[i]), static_cast<int>(got[i])) << "element " << i;
    }
}

TEST(KernelParityQuant, ScalarDequantizeMatchesDefinition) {
    const infr_kernel_desc* row = scalar_dequantize();
    ASSERT_NE(nullptr, row) << "dequantize/int8/tensor/scalar is not registered";

    // x = (q - (-128)) * 0.5
    const int8_t in[4] = {-128, -126, 0, 72};
    const float want[4] = {0.0F, 1.0F, 64.0F, 100.0F};
    float got[4] = {};
    const infr_quant quant = per_tensor(0.5F, -128);
    const void* inputs[1] = {in};
    void* outputs[1] = {got};

    ASSERT_EQ(INFR_OK, row->fn(inputs, outputs, 4, &quant));
    for (int i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ(want[i], got[i]) << "element " << i;
    }
}

TEST(KernelParityQuant, RoundTripStaysWithinHalfAStep) {
    const infr_kernel_desc* quantize_row = scalar_quantize();
    const infr_kernel_desc* dequantize_row = scalar_dequantize();
    ASSERT_NE(nullptr, quantize_row);
    ASSERT_NE(nullptr, dequantize_row);

    const float scale = 0.25F;
    const infr_quant quant = per_tensor(scale, 8);
    const float in[5] = {0.0F, 0.3F, -0.7F, 1.9F, -2.2F};
    int8_t middle[5] = {};
    float back[5] = {};
    const void* q_in[1] = {in};
    void* q_out[1] = {middle};
    const void* d_in[1] = {middle};
    void* d_out[1] = {back};

    ASSERT_EQ(INFR_OK, quantize_row->fn(q_in, q_out, 5, &quant));
    ASSERT_EQ(INFR_OK, dequantize_row->fn(d_in, d_out, 5, &quant));
    for (int i = 0; i < 5; ++i) {
        // Half a quantization step is the most a round trip may lose.
        EXPECT_LE(static_cast<double>(std::abs(back[i] - in[i])), static_cast<double>(scale) / 2.0 + 1e-6)
            << "element " << i;
    }
}

TEST(KernelParityQuant, RejectsNullBuffersAndNonPositiveScale) {
    const infr_kernel_desc* quantize_row = scalar_quantize();
    ASSERT_NE(nullptr, quantize_row);

    float in[2] = {0.0F, 1.0F};
    int8_t out[2] = {};
    const infr_quant good = per_tensor(1.0F, 0);
    const infr_quant bad_scale = per_tensor(0.0F, 0);
    infr_quant unquantized = per_tensor(1.0F, 0);
    unquantized.kind = INFR_QUANT_NONE;
    const void* null_input[1] = {nullptr};
    const void* inputs[1] = {in};
    void* null_output[1] = {nullptr};
    void* outputs[1] = {out};

    EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, quantize_row->fn(null_input, outputs, 2, &good));
    EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, quantize_row->fn(inputs, null_output, 2, &good));
    EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, quantize_row->fn(inputs, outputs, 2, &bad_scale));
    // A NULL parameter block is the branch no realistic call reaches, and the one
    // that would dereference first if the guard were ever reordered.
    EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, quantize_row->fn(inputs, outputs, 2, nullptr));
    EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, quantize_row->fn(inputs, outputs, 2, &unquantized));

    const infr_kernel_desc* dequantize_row = scalar_dequantize();
    ASSERT_NE(nullptr, dequantize_row);
    const void* quantized_in[1] = {out};
    float restored[2] = {};
    void* restored_out[1] = {restored};
    EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, dequantize_row->fn(null_input, restored_out, 2, &good));
    EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, dequantize_row->fn(quantized_in, null_output, 2, &good));
    EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, dequantize_row->fn(quantized_in, restored_out, 2, nullptr));
    EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, dequantize_row->fn(quantized_in, restored_out, 2, &bad_scale));
    EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, dequantize_row->fn(quantized_in, restored_out, 2, &unquantized));
    EXPECT_EQ(INFR_OK, dequantize_row->fn(quantized_in, restored_out, 2, &good));
}

// Census cell: dequantize/int8/tensor/arm_neon.
//
// Same criterion as the scalar pair above and as every other NEON row in this
// tree: the reference does an integer subtraction and then one IEEE multiply, so
// both steps are exact and the verdict is bit equality rather than a window.
// Runs on the host only where NEON exists (AArch64 CI legs, or a native ARM
// machine); elsewhere the row is absent and the case says so instead of passing.
TEST(KernelParityQuantNeon, DequantizeInt8MatchesScalarBitwise) {
    EXPECT_EQ(INFR_OK, infr_kernel_register_scalar());
    EXPECT_EQ(INFR_OK, infr_kernel_register_arm_neon());
    const infr_kernel_desc* scalar = scalar_dequantize();
    const infr_kernel_desc* neon =
        infr_kernel_find("dequantize", INFR_DTYPE_I8, INFR_QUANT_PER_TENSOR, INFR_TIER_ARM_NEON);
    ASSERT_NE(nullptr, scalar) << "dequantize/int8/tensor/scalar is not registered";
    if (neon == nullptr) {
        GTEST_SKIP() << "no NEON in this build: the arm_neon row is absent by design";
    }

    const std::uint32_t lengths[] = {0U, 1U, 3U, 7U, 8U, 9U, 15U, 16U, 17U, 64U};
    std::vector<std::int8_t> q(64);
    for (std::size_t i = 0; i < q.size(); ++i) {
        q[i] = static_cast<std::int8_t>(static_cast<int>((i * 61U) % 256U) - 128);
    }
    q[0] = INT8_MIN; /* the ends of the range, where a widening mistake shows up */
    q[1] = INT8_MAX;
    q[2] = 0;

    /* A zero point outside the int8 range on purpose: the reference subtracts it as an int32, so
     * a body that subtracts in 16 bits would clip it here and nowhere else. */
    const struct {
        float scale;
        std::int32_t zero_point;
    } params[] = {{0.5F, 0}, {0.125F, 3}, {2.0F, 70000}, {0.25F, -256}};

    const void* inputs[1] = {q.data()};
    for (std::uint32_t n : lengths) {
        for (const auto& p : params) {
            infr_quant quant = per_tensor(p.scale, p.zero_point);
            std::vector<float> want(n == 0U ? 1U : n, 0.0F);
            std::vector<float> got(n == 0U ? 1U : n, 0.0F);
            void* want_out[1] = {want.data()};
            void* got_out[1] = {got.data()};
            ASSERT_EQ(INFR_OK, scalar->fn(inputs, want_out, n, &quant));
            ASSERT_EQ(INFR_OK, neon->fn(inputs, got_out, n, &quant));
            for (std::uint32_t i = 0; i < n; ++i) {
                EXPECT_EQ(infr_test::float_bits(want[i]), infr_test::float_bits(got[i]))
                    << "length " << n << " element " << i << " zero_point " << p.zero_point;
            }
        }
    }

    float out = 0.0F;
    void* out_ptr[1] = {&out};
    infr_quant good = per_tensor(0.5F, 0);
    EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, neon->fn(inputs, out_ptr, 4U, nullptr));
    EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, scalar->fn(inputs, out_ptr, 4U, nullptr));
}
