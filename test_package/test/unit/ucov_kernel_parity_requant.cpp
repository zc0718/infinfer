// Census cell: requant/i32/tensor/scalar.
//
// This is where a quantized convolution's rounding and saturation live, so it
// is pinned against the written rule (INV-2) rather than against a second copy
// of the body: the reference below is the rule from quant.c written out
// independently, in double, with the boundaries spelled explicitly. Four things
// decide whether a quantized graph is reproducible, and each has a case that
// fails if it is violated:
//
//   1. rounding is half away from zero, not half to even    -> 3 * 0.5 and -3 *
//   0.5
//   2. the bias is added to the accumulator, before rounding -> 10 and -10 with
//   a 0.5 multiplier
//   3. saturation clamps to the int8 range, both ends        -> +1e6 and -1e6
//   4. the zero point is an output offset, added last        -> zero point 40
//   on a known result
#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include <infinfer/infinfer.h>

namespace {

const infr_kernel_desc* scalar_requant() {
    EXPECT_EQ(INFR_OK, infr_kernel_register_scalar());
    return infr_kernel_find("requant", INFR_DTYPE_I32, INFR_QUANT_PER_TENSOR, INFR_TIER_SCALAR);
}

// The written rule, transcribed independently: saturate in the double domain
// before converting, because an out-of-range float-to-integer conversion is
// undefined behaviour rather than a clamp.
std::int8_t reference_requant(std::int32_t acc, double multiplier, std::int32_t zero_point) {
    const double scaled = (static_cast<double>(acc) * multiplier) + static_cast<double>(zero_point);
    if (scaled <= static_cast<double>(std::numeric_limits<std::int8_t>::min())) {
        return std::numeric_limits<std::int8_t>::min();
    }
    if (scaled >= static_cast<double>(std::numeric_limits<std::int8_t>::max())) {
        return std::numeric_limits<std::int8_t>::max();
    }
    return static_cast<std::int8_t>(scaled >= 0.0 ? scaled + 0.5 : scaled - 0.5);
}

std::int8_t empty_out = 0;

void expect_requant(const std::vector<std::int32_t>& accumulators, const std::vector<std::int32_t>* bias,
                    float multiplier, std::int32_t zero_point) {
    const infr_kernel_desc* kernel = scalar_requant();
    ASSERT_NE(nullptr, kernel);
    infr_quant quant{};
    quant.kind = INFR_QUANT_PER_TENSOR;
    quant.scale = 1.0F;
    quant.zero_point = zero_point;
    quant.requant_multiplier = multiplier;

    std::vector<std::int8_t> got(accumulators.size(), 0);
    // `vector::data()` on an empty vector may be NULL, and a NULL accumulator is
    // (correctly) rejected, so the empty case is fed a valid pointer with a count
    // of zero.
    const std::int32_t nothing = 0;
    const void* inputs[2] = {accumulators.empty() ? &nothing : accumulators.data(),
                             bias != nullptr ? bias->data() : nullptr};
    void* outputs[1] = {got.empty() ? static_cast<void*>(&empty_out) : got.data()};
    ASSERT_EQ(INFR_OK, kernel->fn(inputs, outputs, static_cast<std::uint32_t>(accumulators.size()), &quant));
    for (std::size_t i = 0; i < accumulators.size(); ++i) {
        const std::int32_t acc = accumulators[i] + (bias != nullptr ? (*bias)[i] : 0);
        EXPECT_EQ(reference_requant(acc, multiplier, zero_point), got[i])
            << "element " << i << " of " << accumulators.size();
    }
}

} // namespace

TEST(KernelParityRequant, ScalarI32MatchesWrittenRule) {
    // Rounding, half away from zero. With a 0.5 multiplier the odd accumulators
    // land exactly on .5: half-to-even would answer 2 and -2, this rule answers 1
    // and -1.
    expect_requant({1, 3, 5, -1, -3, -5, 0, 2, -2}, nullptr, 0.5F, 0);

    // The same boundaries with a multiplier that is not a power of two, where a
    // body that rounds by truncation or by adding 0.5F as a float would part
    // company from the double definition.
    expect_requant({7, -7, 21, -21, 100, -100}, nullptr, 0.01F, 0);

    // Saturation, both ends, and exactly at the ends: -128 and 127 are
    // representable and must not be clamped away by an off-by-one in the guard.
    expect_requant({-128, 127, -129, 128, -1000000, 1000000}, nullptr, 1.0F, 0);

    // The bias is added to the accumulator *before* rounding, which is the whole
    // reason this cell exists: with a 0.5 multiplier, (10 - 10) * 0.5 = 0,
    // whereas adding a bias of -10 after the requantization would answer -10.
    expect_requant({10, -10, 11, -11}, nullptr, 0.5F, 0);
    expect_requant({10, -10, 11, -11}, new std::vector<std::int32_t>{-10, 10, -10, 10}, 0.5F, 0);

    // A bias alone can saturate, without the accumulator contributing anything.
    expect_requant({0, 0}, new std::vector<std::int32_t>{500, -500}, 1.0F, 0);

    // The zero point offsets the output, and is applied after the multiplication:
    // 60 * 0.5 + 40 is 70, not (60 + 40) * 0.5 = 50.
    expect_requant({60, -60, 0}, nullptr, 0.5F, 40);

    // An empty range is legal and touches nothing.
    expect_requant({}, nullptr, 1.0F, 0);
}

TEST(KernelParityRequant, ScalarRejectsMissingParameters) {
    const infr_kernel_desc* kernel = scalar_requant();
    ASSERT_NE(nullptr, kernel);
    std::int32_t acc[2] = {1, 2};
    std::int8_t out[2] = {0, 0};
    infr_quant quant{};
    quant.kind = INFR_QUANT_PER_TENSOR;
    quant.scale = 1.0F;
    quant.zero_point = 0;
    quant.requant_multiplier = 1.0F;
    const void* inputs[2] = {acc, nullptr};
    void* outputs[1] = {out};

    // No parameters at all: an accumulator cannot be turned into an int8 without
    // a multiplier, and guessing one would be worse than refusing.
    EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, kernel->fn(inputs, outputs, 2U, nullptr));

    // A declared-but-absent scheme, and a multiplier that is zero or negative:
    // the graph layer resolves `(scale_a * scale_b) / scale_out`, so a
    // non-positive value means it did not.
    infr_quant none = quant;
    none.kind = INFR_QUANT_NONE;
    EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, kernel->fn(inputs, outputs, 2U, &none));
    infr_quant zero = quant;
    zero.requant_multiplier = 0.0F;
    EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, kernel->fn(inputs, outputs, 2U, &zero));
    infr_quant negative = quant;
    negative.requant_multiplier = -1.0F;
    EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, kernel->fn(inputs, outputs, 2U, &negative));

    // Pointers, including a NULL source even though a NULL bias is legal.
    const void* inputs_no_acc[2] = {nullptr, nullptr};
    EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, kernel->fn(nullptr, outputs, 2U, &quant));
    EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, kernel->fn(inputs_no_acc, outputs, 2U, &quant));
    void* outputs_null[1] = {nullptr};
    EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, kernel->fn(inputs, outputs_null, 2U, &quant));
}


int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
