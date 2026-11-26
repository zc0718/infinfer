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

#include <infinfer/infinfer.h>

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
