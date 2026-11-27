// P0.5 case 3 (deliberately red): quantize/dequantize must do arithmetic or say no.
//
// The reference defect this pins: `quant`/`dequant` stubs that returned success
// while writing nothing. Values here are chosen so the expected int8 result is
// exact (no rounding-mode ambiguity): x / 0.5 + (-8).
//
// Expected state at 0.1.0: RED -- the implementation returns
// INFR_ERR_NOT_IMPLEMENTED instead of producing bytes.
#include <gtest/gtest.h>

#include <cstdint>

#include <infinfer/infinfer.h>

TEST(ContractQuant, RoundTripDoesArithmeticOrReportsNotImplemented) {
    infr_quant quant{};
    quant.kind = INFR_QUANT_PER_TENSOR;
    quant.scale = 0.5F;
    quant.zero_point = -8;

    // 1) float -> int8: every byte must be written.
    const float floats[3] = {1.0F, 2.0F, -1.0F};
    int8_t bytes[3] = {0, 0, 0};
    EXPECT_EQ(INFR_OK, infr_quantize_i8(floats, bytes, 3, &quant));
    EXPECT_EQ(static_cast<int8_t>(-6), bytes[0]);  /* round(1.0 / 0.5) - 8 */
    EXPECT_EQ(static_cast<int8_t>(-4), bytes[1]);  /* round(2.0 / 0.5) - 8 */
    EXPECT_EQ(static_cast<int8_t>(-10), bytes[2]); /* round(-1.0 / 0.5) - 8 */

    // 2) int8 -> float: the same parameters restore the inputs.
    const int8_t stored[3] = {-6, -4, -10};
    float restored[3] = {0.0F, 0.0F, 0.0F};
    EXPECT_EQ(INFR_OK, infr_dequantize_i8(stored, restored, 3, &quant));
    EXPECT_FLOAT_EQ(1.0F, restored[0]); /* (-6 + 8) * 0.5 */
    EXPECT_FLOAT_EQ(2.0F, restored[1]); /* (-4 + 8) * 0.5 */
    EXPECT_FLOAT_EQ(-1.0F, restored[2]);

    // 3) The guard that already holds: an empty implementation must not claim success,
    //    and an unquantized request is not a quantization.
    infr_quant none{};
    none.kind = INFR_QUANT_NONE;
    EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, infr_quantize_i8(floats, bytes, 1, &none));
    EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, infr_quantize_i8(nullptr, bytes, 1, &none));
}


int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
