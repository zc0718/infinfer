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

TEST(KernelParityFc, RejectsMissingBuffers) {    const infr_kernel_desc* row = scalar_fc();
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

TEST(KernelParityFcInt8, ScalarInt8MatchesDefinition) {
    EXPECT_EQ(INFR_OK, infr_kernel_register_scalar());
    const infr_kernel_desc* row =
        infr_kernel_find("fc", INFR_DTYPE_I8, INFR_QUANT_PER_TENSOR, INFR_TIER_SCALAR);
    ASSERT_NE(nullptr, row) << "fc/int8/tensor/scalar is not registered";

    // 1*4 + 2*5 + 3*6 = 32, then 32 * 1.0 + 0 = 32.
    const int8_t x[3] = {1, 2, 3};
    const int8_t w[3] = {4, 5, 6};
    int8_t got = -1;
    infr_quant quant{};
    quant.kind = INFR_QUANT_PER_TENSOR;
    quant.scale = 1.0F;
    quant.zero_point = 0;
    quant.requant_multiplier = 1.0F;
    const void* inputs[2] = {x, w};
    void* outputs[1] = {&got};

    ASSERT_EQ(INFR_OK, row->fn(inputs, outputs, 3, &quant));
    EXPECT_EQ(32, static_cast<int>(got));
}

TEST(KernelParityFcInt8, AppliesMultiplierAndZeroPointThenClamps) {
    EXPECT_EQ(INFR_OK, infr_kernel_register_scalar());
    const infr_kernel_desc* row =
        infr_kernel_find("fc", INFR_DTYPE_I8, INFR_QUANT_PER_TENSOR, INFR_TIER_SCALAR);
    ASSERT_NE(nullptr, row);

    const int8_t x[3] = {1, 2, 3};
    const int8_t w[3] = {4, 5, 6};  // acc = 32
    int8_t got = 0;
    infr_quant quant{};
    quant.kind = INFR_QUANT_PER_TENSOR;
    quant.scale = 2.0F;
    quant.zero_point = -10;
    quant.requant_multiplier = 0.5F;  // 32 * 0.5 - 10 = 6
    const void* inputs[2] = {x, w};
    void* outputs[1] = {&got};
    ASSERT_EQ(INFR_OK, row->fn(inputs, outputs, 3, &quant));
    EXPECT_EQ(6, static_cast<int>(got));

    // Saturation: 127*127 * 0.01 + 0 = 161.29 -> 161 -> clamped to 127.
    const int8_t big_x[1] = {127};
    const int8_t big_w[1] = {127};
    quant.zero_point = 0;
    quant.requant_multiplier = 0.01F;
    const void* big_inputs[2] = {big_x, big_w};
    ASSERT_EQ(INFR_OK, row->fn(big_inputs, outputs, 1, &quant));
    EXPECT_EQ(127, static_cast<int>(got));
}

TEST(KernelParityFcInt8, ZeroLengthReducesToTheZeroPoint) {
    EXPECT_EQ(INFR_OK, infr_kernel_register_scalar());
    const infr_kernel_desc* row =
        infr_kernel_find("fc", INFR_DTYPE_I8, INFR_QUANT_PER_TENSOR, INFR_TIER_SCALAR);
    ASSERT_NE(nullptr, row);

    const int8_t x[1] = {7};
    const int8_t w[1] = {9};
    int8_t got = 0;
    infr_quant quant{};
    quant.kind = INFR_QUANT_PER_TENSOR;
    quant.scale = 1.0F;
    quant.zero_point = -3;
    quant.requant_multiplier = 1.0F;
    const void* inputs[2] = {x, w};
    void* outputs[1] = {&got};
    ASSERT_EQ(INFR_OK, row->fn(inputs, outputs, 0, &quant));
    EXPECT_EQ(-3, static_cast<int>(got));
}

TEST(KernelParityFcInt8, RejectsUnresolvedQuantization) {
    EXPECT_EQ(INFR_OK, infr_kernel_register_scalar());
    const infr_kernel_desc* row =
        infr_kernel_find("fc", INFR_DTYPE_I8, INFR_QUANT_PER_TENSOR, INFR_TIER_SCALAR);
    ASSERT_NE(nullptr, row);

    const int8_t x[1] = {1};
    const int8_t w[1] = {1};
    int8_t got = 0;
    const void* inputs[2] = {x, w};
    void* outputs[1] = {&got};

    infr_quant unresolved{};
    unresolved.kind = INFR_QUANT_PER_TENSOR;  // multiplier left at 0
    EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, row->fn(inputs, outputs, 1, &unresolved));
    EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, row->fn(inputs, outputs, 1, nullptr));

    infr_quant unquantized{};
    unquantized.kind = INFR_QUANT_NONE;
    unquantized.requant_multiplier = 1.0F;
    EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, row->fn(inputs, outputs, 1, &unquantized));

    // The pointer guard runs before the quantization guard, so a caller that got both
    // wrong gets the pointer verdict -- pinned because "which error wins" is part of
    // the contract a consumer branches on.
    const void* no_input[2] = {nullptr, w};
    const void* no_weights[2] = {x, nullptr};
    void* no_output[1] = {nullptr};
    EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, row->fn(nullptr, outputs, 1, &unquantized));
    EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, row->fn(no_input, outputs, 1, &unquantized));
    EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, row->fn(no_weights, outputs, 1, &unquantized));
    EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, row->fn(inputs, no_output, 1, &unquantized));
}

/*
 * The int16 cell (`fc/int16/tensor/{scalar,arm_m}`). It is not "int8 with a wider
 * type": the accumulator has to be int64, because `32767 * 32767` is about 2^30
 * and int32 wraps on the *second* worst-case term. One case below pins exactly
 * that -- it is the case that fails if anyone ever "simplifies" the accumulator.
 *
 * int16 格。它不是“更宽的 int8”：累加器必须是 int64，因为 `32767 * 32767` 约为 2^30，
 * int32 在最坏情况下的**第二项**就会回绕。下面有一条用例专钉这件事——谁把累加器
 * “简化”回去，它就先红。
 */

namespace {

const infr_kernel_desc* scalar_fc_i16() {
    EXPECT_EQ(INFR_OK, infr_kernel_register_scalar());
    return infr_kernel_find("fc", INFR_DTYPE_I16, INFR_QUANT_PER_TENSOR, INFR_TIER_SCALAR);
}

}  // namespace

TEST(KernelParityFcInt16, ScalarInt16MatchesDefinition) {
    const infr_kernel_desc* row = scalar_fc_i16();
    ASSERT_NE(nullptr, row) << "fc/int16/tensor/scalar is not registered";

    // 1*4 + 2*5 + 3*6 = 32, then 32 * 1.0 + 0 = 32.
    const std::int16_t x[3] = {1, 2, 3};
    const std::int16_t w[3] = {4, 5, 6};
    std::int16_t got = -1;
    infr_quant quant{};
    quant.kind = INFR_QUANT_PER_TENSOR;
    quant.scale = 1.0F;
    quant.zero_point = 0;
    quant.requant_multiplier = 1.0F;
    const void* inputs[2] = {x, w};
    void* outputs[1] = {&got};

    ASSERT_EQ(INFR_OK, row->fn(inputs, outputs, 3, &quant));
    EXPECT_EQ(32, static_cast<int>(got));
}

TEST(KernelParityFcInt16, AccumulatesIn64BitsNot32) {
    const infr_kernel_desc* row = scalar_fc_i16();
    ASSERT_NE(nullptr, row);

    // 64 terms of 16384*16384 = 2^28 put the accumulator at 2^34, eight times INT32_MAX,
    // and 2^34 wraps to exactly 0 in 32 bits -- so a body that accumulated in int32 would
    // answer 0 here, not "something slightly off". The multiplier is 2^-21, which makes
    // the exact answer 2^13 = 8192 (both powers of two, so no rounding is involved).
    std::int16_t x[64];
    std::int16_t w[64];
    for (int i = 0; i < 64; ++i) {
        x[i] = 16384;
        w[i] = 16384;
    }
    std::int16_t got = 0;
    infr_quant quant{};
    quant.kind = INFR_QUANT_PER_TENSOR;
    quant.scale = 1.0F;
    quant.zero_point = 0;
    quant.requant_multiplier = 1.0F / 2097152.0F;
    const void* inputs[2] = {x, w};
    void* outputs[1] = {&got};

    ASSERT_EQ(INFR_OK, row->fn(inputs, outputs, 64, &quant));
    std::int64_t want = 0;
    for (int i = 0; i < 64; ++i) {
        want += static_cast<std::int64_t>(x[i]) * static_cast<std::int64_t>(w[i]);
    }
    ASSERT_EQ(17179869184LL, static_cast<long long>(want));  // 2^34, the accumulator itself
    EXPECT_EQ(8192, static_cast<int>(got));
}

TEST(KernelParityFcInt16, SaturatesInsteadOfWrapping) {
    const infr_kernel_desc* row = scalar_fc_i16();
    ASSERT_NE(nullptr, row);

    infr_quant quant{};
    quant.kind = INFR_QUANT_PER_TENSOR;
    quant.scale = 1.0F;
    quant.zero_point = 0;
    quant.requant_multiplier = 1.0F;

    // 32767*32767 = 1073676289, far outside int16: saturating is the contract.
    const std::int16_t big_x[1] = {32767};
    const std::int16_t big_w[1] = {32767};
    std::int16_t got = 0;
    const void* inputs[2] = {big_x, big_w};
    void* outputs[1] = {&got};
    ASSERT_EQ(INFR_OK, row->fn(inputs, outputs, 1, &quant));
    EXPECT_EQ(32767, static_cast<int>(got));

    // -32768*32767 = -1073709056: the other end, and the value that catches a
    // clamp written as `value > 32767 ? ... : (int16_t)value` (which would wrap).
    const std::int16_t low_x[1] = {-32768};
    const std::int16_t low_w[1] = {32767};
    const void* low_inputs[2] = {low_x, low_w};
    ASSERT_EQ(INFR_OK, row->fn(low_inputs, outputs, 1, &quant));
    EXPECT_EQ(-32768, static_cast<int>(got));
}

TEST(KernelParityFcInt16, ZeroLengthReducesToTheZeroPoint) {
    const infr_kernel_desc* row = scalar_fc_i16();
    ASSERT_NE(nullptr, row);

    const std::int16_t x[1] = {7};
    const std::int16_t w[1] = {9};
    std::int16_t got = 0;
    infr_quant quant{};
    quant.kind = INFR_QUANT_PER_TENSOR;
    quant.scale = 1.0F;
    quant.zero_point = -5;
    quant.requant_multiplier = 1.0F;
    const void* inputs[2] = {x, w};
    void* outputs[1] = {&got};
    ASSERT_EQ(INFR_OK, row->fn(inputs, outputs, 0, &quant));
    EXPECT_EQ(-5, static_cast<int>(got));
}

TEST(KernelParityFcInt16, RejectsUnresolvedQuantization) {
    const infr_kernel_desc* row = scalar_fc_i16();
    ASSERT_NE(nullptr, row);

    const std::int16_t x[1] = {1};
    const std::int16_t w[1] = {1};
    std::int16_t got = 0;
    const void* inputs[2] = {x, w};
    void* outputs[1] = {&got};

    infr_quant unresolved{};
    unresolved.kind = INFR_QUANT_PER_TENSOR;  // multiplier left at 0
    EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, row->fn(inputs, outputs, 1, &unresolved));
    EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, row->fn(inputs, outputs, 1, nullptr));

    infr_quant unquantized{};
    unquantized.kind = INFR_QUANT_NONE;
    unquantized.requant_multiplier = 1.0F;
    EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, row->fn(inputs, outputs, 1, &unquantized));

    // Same pointer-before-quantization ordering as the int8 row: a consumer that
    // branches on the status must see the same verdict from both quantized cells.
    const void* no_input[2] = {nullptr, w};
    const void* no_weights[2] = {x, nullptr};
    void* no_output[1] = {nullptr};
    EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, row->fn(nullptr, outputs, 1, &unquantized));
    EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, row->fn(no_input, outputs, 1, &unquantized));
    EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, row->fn(no_weights, outputs, 1, &unquantized));
    EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, row->fn(inputs, no_output, 1, &unquantized));
}

TEST(KernelParityFcArmM, Int8MatchesScalarBitwise) {
    EXPECT_EQ(INFR_OK, infr_kernel_register_scalar());
    EXPECT_EQ(INFR_OK, infr_kernel_register_arm_m());
    const infr_kernel_desc* scalar =
        infr_kernel_find("fc", INFR_DTYPE_I8, INFR_QUANT_PER_TENSOR, INFR_TIER_SCALAR);
    const infr_kernel_desc* dsp =
        infr_kernel_find("fc", INFR_DTYPE_I8, INFR_QUANT_PER_TENSOR, INFR_TIER_ARM_M);
    ASSERT_NE(nullptr, scalar) << "fc/int8/tensor/scalar is not registered";
    if (dsp == nullptr) {
        GTEST_SKIP() << "no DSP extension in this build: the arm_m row is absent by design";
    }

    infr_quant quant{};
    quant.kind = INFR_QUANT_PER_TENSOR;
    quant.scale = 0.5F;
    quant.zero_point = -7;
    quant.requant_multiplier = 0.125F;

    // Odd lengths exercise the scalar tail; negative values exercise the signed lane
    // packing, which is where a hand-written DSP path usually breaks.
    std::uint8_t x[17];
    std::uint8_t w[17];
    for (int i = 0; i < 17; ++i) {
        x[i] = static_cast<std::uint8_t>((i * 37) % 256);
        w[i] = static_cast<std::uint8_t>((i * 91 + 13) % 256);
    }
    for (std::uint32_t n = 1U; n <= 17U; ++n) {
        std::int8_t want = 0;
        std::int8_t got = 0;
        const void* inputs[2] = {x, w};
        void* want_out[1] = {&want};
        void* got_out[1] = {&got};
        ASSERT_EQ(INFR_OK, scalar->fn(inputs, want_out, n, &quant));
        ASSERT_EQ(INFR_OK, dsp->fn(inputs, got_out, n, &quant));
        EXPECT_EQ(static_cast<int>(want), static_cast<int>(got)) << "inner dimension " << n;
    }
}

TEST(KernelParityFcArmM, Int16MatchesScalarBitwise) {
    EXPECT_EQ(INFR_OK, infr_kernel_register_scalar());
    EXPECT_EQ(INFR_OK, infr_kernel_register_arm_m());
    const infr_kernel_desc* scalar =
        infr_kernel_find("fc", INFR_DTYPE_I16, INFR_QUANT_PER_TENSOR, INFR_TIER_SCALAR);
    const infr_kernel_desc* dsp =
        infr_kernel_find("fc", INFR_DTYPE_I16, INFR_QUANT_PER_TENSOR, INFR_TIER_ARM_M);
    ASSERT_NE(nullptr, scalar) << "fc/int16/tensor/scalar is not registered";
    if (dsp == nullptr) {
        GTEST_SKIP() << "no DSP extension in this build: the arm_m row is absent by design";
    }

    infr_quant quant{};
    quant.kind = INFR_QUANT_PER_TENSOR;
    quant.scale = 0.5F;
    quant.zero_point = -1234;
    quant.requant_multiplier = 1.0F / 1024.0F;

    // Signed values across the whole int16 range, so the pair lanes are exercised in
    // both halves -- a DSP body that mixes lanes up still gets the product *sum*
    // right, which is why the comparison is on the output value and not on lanes.
    std::int16_t x[64];
    std::int16_t w[64];
    for (int i = 0; i < 64; ++i) {
        x[i] = static_cast<std::int16_t>(((i * 4001) % 65536) - 32768);
        w[i] = static_cast<std::int16_t>(((i * 7919 + 12345) % 65536) - 32768);
    }

    // 1..17 covers the odd tail and single-SMLALD lengths; 64 is 2^34-scale accumulation
    // (see the scalar case), which is the int64 claim under test.
    const std::uint32_t lengths[] = {1U, 2U, 3U, 5U, 16U, 17U, 64U};
    for (std::uint32_t n : lengths) {
        std::int16_t want = 0;
        std::int16_t got = 0;
        const void* inputs[2] = {x, w};
        void* want_out[1] = {&want};
        void* got_out[1] = {&got};
        ASSERT_EQ(INFR_OK, scalar->fn(inputs, want_out, n, &quant));
        ASSERT_EQ(INFR_OK, dsp->fn(inputs, got_out, n, &quant));
        EXPECT_EQ(static_cast<int>(want), static_cast<int>(got)) << "inner dimension " << n;
    }
}

TEST(KernelParityFcNeon, Fp32MatchesScalarWithinAFloatWindow) {
    EXPECT_EQ(INFR_OK, infr_kernel_register_scalar());
    EXPECT_EQ(INFR_OK, infr_kernel_register_arm_neon());
    const infr_kernel_desc* scalar =
        infr_kernel_find("fc", INFR_DTYPE_F32, INFR_QUANT_NONE, INFR_TIER_SCALAR);
    const infr_kernel_desc* neon =
        infr_kernel_find("fc", INFR_DTYPE_F32, INFR_QUANT_NONE, INFR_TIER_ARM_NEON);
    ASSERT_NE(nullptr, scalar) << "fc/fp32/none/scalar is not registered";
    if (neon == nullptr) {
        GTEST_SKIP() << "no NEON in this build: the arm_neon row is absent by design";
    }

    // The first tier that is not bit-identical to the reference: four lanes change the
    // grouping of the partial sums, so this is judged with the float window the parity
    // harness exists for. Integer tiers keep their zero-tolerance comparison.
    const std::uint32_t lengths[] = {1U, 2U, 3U, 4U, 5U, 7U, 16U, 17U, 64U, 257U};
    float x[257];
    float w[257];
    fill(x, sizeof(x) / sizeof(x[0]), 11U);
    fill(w, sizeof(w) / sizeof(w[0]), 29U);
    for (std::uint32_t length : lengths) {
        const void* inputs[2] = {x, w};
        float want = 0.0F;
        float got = 0.0F;
        void* want_out[1] = {&want};
        void* got_out[1] = {&got};
        ASSERT_EQ(INFR_OK, scalar->fn(inputs, want_out, length, nullptr));
        ASSERT_EQ(INFR_OK, neon->fn(inputs, got_out, length, nullptr));
        infr_test::expect_float_buffers_near(&want, &got, 1, "fc fp32 neon vs scalar");
    }

    // A zero-length row is a legal reduction in both tiers, not an error.
    const void* inputs[2] = {x, w};
    float got = 42.0F;
    void* got_out[1] = {&got};
    ASSERT_EQ(INFR_OK, neon->fn(inputs, got_out, 0U, nullptr));
    EXPECT_EQ(0.0F, got);
}

// Census cell: fc/int16/tensor/arm_neon.
//
// The criterion is bit equality even though the tier is a dot product: the
// reference accumulates integers in an int64 and integer addition is
// associative, so a different lane split still has to produce the same total.
// The corpus is built so that an int32 accumulator cannot pass -- 64 terms of
// 16384*16384 put the total at 2^34, which wraps to exactly 0 in 32 bits.
TEST(KernelParityFcNeon, Int16MatchesScalarBitwise) {
    EXPECT_EQ(INFR_OK, infr_kernel_register_scalar());
    EXPECT_EQ(INFR_OK, infr_kernel_register_arm_neon());
    const infr_kernel_desc* scalar = scalar_fc_i16();
    const infr_kernel_desc* neon =
        infr_kernel_find("fc", INFR_DTYPE_I16, INFR_QUANT_PER_TENSOR, INFR_TIER_ARM_NEON);
    ASSERT_NE(nullptr, scalar) << "fc/int16/tensor/scalar is not registered";
    if (neon == nullptr) {
        GTEST_SKIP() << "no NEON in this build: the arm_neon row is absent by design";
    }

    const std::uint32_t lengths[] = {0U, 1U, 3U, 7U, 8U, 9U, 15U, 16U, 17U, 64U};
    std::vector<std::int16_t> x(64);
    std::vector<std::int16_t> w(64);
    for (std::size_t i = 0; i < x.size(); ++i) {
        x[i] = static_cast<std::int16_t>(static_cast<int>((i * 733U) % 65533U) - 32766);
        w[i] = static_cast<std::int16_t>(static_cast<int>((i * 911U + 7U) % 65533U) - 32766);
    }
    /* The int32-wrap corner lives at the front, so the short lengths hit it too. */
    for (int i = 0; i < 64; ++i) {
        x[i] = 16384;
        w[i] = 16384;
    }
    x[0] = INT16_MIN; /* -32768^2 is the other end a widening mistake trips on */
    w[0] = INT16_MIN;

    infr_quant quant{};
    quant.kind = INFR_QUANT_PER_TENSOR;
    quant.scale = 1.0F;
    quant.zero_point = 0;
    quant.requant_multiplier = 1.0F / 2097152.0F; /* 2^-21: the exact total 2^34 lands on 8192 */
    const void* inputs[2] = {x.data(), w.data()};
    for (std::uint32_t n : lengths) {
        std::int16_t want = 0;
        std::int16_t got = 0;
        void* want_out[1] = {&want};
        void* got_out[1] = {&got};
        ASSERT_EQ(INFR_OK, scalar->fn(inputs, want_out, n, &quant));
        ASSERT_EQ(INFR_OK, neon->fn(inputs, got_out, n, &quant));
        EXPECT_EQ(static_cast<int>(want), static_cast<int>(got)) << "length " << n;
    }

    std::int16_t out = 0;
    void* out_ptr[1] = {&out};
    EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, neon->fn(inputs, out_ptr, 4U, nullptr));
    EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, scalar->fn(inputs, out_ptr, 4U, nullptr));
}

// Census cell: fc/int8/tensor/arm_neon.
//
// Bit equality, for the reason the x86 body states: the reference accumulates in
// an int32 and the tier reaches the same exact integer sum with widening
// multiplies, so the lane split cannot change the answer. Overflow is outside
// the contract on both sides.
TEST(KernelParityFcNeon, Int8MatchesScalarBitwise) {
    EXPECT_EQ(INFR_OK, infr_kernel_register_scalar());
    EXPECT_EQ(INFR_OK, infr_kernel_register_arm_neon());
    const infr_kernel_desc* scalar =
        infr_kernel_find("fc", INFR_DTYPE_I8, INFR_QUANT_PER_TENSOR, INFR_TIER_SCALAR);
    const infr_kernel_desc* neon =
        infr_kernel_find("fc", INFR_DTYPE_I8, INFR_QUANT_PER_TENSOR, INFR_TIER_ARM_NEON);
    ASSERT_NE(nullptr, scalar) << "fc/int8/tensor/scalar is not registered";
    if (neon == nullptr) {
        GTEST_SKIP() << "no NEON in this build: the arm_neon row is absent by design";
    }

    const std::uint32_t lengths[] = {0U, 1U, 3U, 7U, 8U, 9U, 15U, 16U, 17U, 64U, 257U};
    std::vector<std::int8_t> x(257);
    std::vector<std::int8_t> w(257);
    for (std::size_t i = 0; i < x.size(); ++i) {
        x[i] = static_cast<std::int8_t>(static_cast<int>((i * 37U) % 255U) - 127);
        w[i] = static_cast<std::int8_t>(static_cast<int>((i * 91U + 7U) % 255U) - 127);
    }
    /* The corners a signed-byte widening mistake trips on. */
    x[0] = INT8_MIN;
    w[0] = INT8_MIN;
    x[1] = INT8_MAX;
    w[1] = INT8_MIN;
    x[2] = -1;
    w[2] = INT8_MIN;

    infr_quant quant{};
    quant.kind = INFR_QUANT_PER_TENSOR;
    quant.scale = 1.0F;
    quant.zero_point = 3;
    quant.requant_multiplier = 1.0e-3F; /* the corpus accumulates to ~1e5, so 1e-3 keeps the result in range while staying sensitive */
    const void* inputs[2] = {x.data(), w.data()};
    for (std::uint32_t n : lengths) {
        std::int8_t want = 0;
        std::int8_t got = 0;
        void* want_out[1] = {&want};
        void* got_out[1] = {&got};
        ASSERT_EQ(INFR_OK, scalar->fn(inputs, want_out, n, &quant));
        ASSERT_EQ(INFR_OK, neon->fn(inputs, got_out, n, &quant));
        EXPECT_EQ(static_cast<int>(want), static_cast<int>(got)) << "length " << n;
    }

    std::int8_t out = 0;
    void* out_ptr[1] = {&out};
    EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, neon->fn(inputs, out_ptr, 4U, nullptr));
    EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, scalar->fn(inputs, out_ptr, 4U, nullptr));
}
