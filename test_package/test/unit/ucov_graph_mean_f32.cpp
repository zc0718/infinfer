// Conan::ImportStart
#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include <infinfer/infinfer.h>

#include "parity_harness.hpp"
// Conan::ImportEnd

/*!
 * @brief [en] `infr_mean_hw_f32`: global average pooling judged against a double-precision mean.
 * @brief [zh] `infr_mean_hw_f32`：按 double 精度均值判定的全局平均池化。
 * @since 0.1.0
 */

namespace {

/* The reference sums in double, in the same ascending order the operator uses, then
 * divides once. Reproducing the summation order matters: a different order is a
 * different rounding, and the operator promises one specific answer.
 *
 * 参考实现用 double 按与算子相同的升序累加，最后只除一次。复现求和顺序是必要的：
 * 换顺序就是换舍入，而算子承诺的是同一个确定答案。 */
std::vector<double> reference_global_mean(const std::vector<float>& in, uint32_t in_h, uint32_t in_w,
                                         uint32_t channels) {
    const uint32_t plane = in_h * in_w;
    std::vector<double> out(channels, 0.0);
    for (uint32_t c = 0; c < channels; ++c) {
        double total = 0.0;
        for (uint32_t i = 0; i < plane; ++i) {
            total += static_cast<double>(in[(static_cast<size_t>(i) * channels) + c]);
        }
        out[c] = total / static_cast<double>(plane);
    }
    return out;
}

constexpr size_t kWorkspaceBytes = 4096U;

}  // namespace

TEST(GraphMeanHw, MatchesDoubleReference) {
    const uint32_t in_dims[4] = {1U, 3U, 2U, 3U}; /* 6 positions per channel */
    const uint32_t out_dims[4] = {1U, 1U, 1U, 3U};

    std::vector<float> input_data = {1.0F, 10.0F, -1.0F, 2.0F,  20.0F, -2.0F, 3.0F,  30.0F, -3.0F,
                                     4.0F, 40.0F, -4.0F, 5.0F,  50.0F, -5.0F, 6.0F,  60.0F, -6.0F};
    std::vector<float> output_data(3U, 0.0F);

    infr_shape in_shape{};
    infr_shape out_shape{};
    ASSERT_EQ(INFR_OK, infr_shape_init(&in_shape, 4U, in_dims));
    ASSERT_EQ(INFR_OK, infr_shape_init(&out_shape, 4U, out_dims));
    infr_tensor input{};
    infr_tensor output{};
    ASSERT_EQ(INFR_OK, infr_tensor_init(&input, input_data.data(), INFR_DTYPE_F32, &in_shape));
    ASSERT_EQ(INFR_OK, infr_tensor_init(&output, output_data.data(), INFR_DTYPE_F32, &out_shape));

    alignas(8) uint8_t workspace_bytes[kWorkspaceBytes] = {};
    infr_workspace workspace{};
    ASSERT_EQ(INFR_OK, infr_workspace_init(&workspace, workspace_bytes, sizeof(workspace_bytes)));
    ASSERT_EQ(INFR_OK, infr_mean_hw_f32(&input, &output, &workspace));

    /* Channel means are 3.5, 35.0, -3.5 by hand: 1..6 / 10..60 / -1..-6. */
    const auto want = reference_global_mean(input_data, 3U, 2U, 3U);
    const std::vector<float> want_f(want.begin(), want.end());
    infr_test::expect_float_buffers_near(want_f.data(), output_data.data(), want_f.size(), "mean hw");
    EXPECT_NEAR(3.5F, output_data[0], 1e-6F);
    EXPECT_NEAR(35.0F, output_data[1], 1e-6F);
    EXPECT_NEAR(-3.5F, output_data[2], 1e-6F);

    /* The scratch is given back, so a graph full of GAP layers does not grow. */
    EXPECT_EQ(0U, infr_workspace_used(&workspace));
}

TEST(GraphMeanHw, SinglePositionKeepsTheValue) {
    const uint32_t in_dims[4] = {1U, 1U, 1U, 2U};
    const uint32_t out_dims[4] = {1U, 1U, 1U, 2U};

    std::vector<float> input_data = {0.25F, -8.0F};
    std::vector<float> output_data(2U, 0.0F);

    infr_shape in_shape{};
    infr_shape out_shape{};
    ASSERT_EQ(INFR_OK, infr_shape_init(&in_shape, 4U, in_dims));
    ASSERT_EQ(INFR_OK, infr_shape_init(&out_shape, 4U, out_dims));
    infr_tensor input{};
    infr_tensor output{};
    ASSERT_EQ(INFR_OK, infr_tensor_init(&input, input_data.data(), INFR_DTYPE_F32, &in_shape));
    ASSERT_EQ(INFR_OK, infr_tensor_init(&output, output_data.data(), INFR_DTYPE_F32, &out_shape));

    alignas(8) uint8_t workspace_bytes[kWorkspaceBytes] = {};
    infr_workspace workspace{};
    ASSERT_EQ(INFR_OK, infr_workspace_init(&workspace, workspace_bytes, sizeof(workspace_bytes)));
    ASSERT_EQ(INFR_OK, infr_mean_hw_f32(&input, &output, &workspace));

    /* Dividing by 1 must not perturb the value: this pins the reciprocal path. */
    infr_test::expect_float_buffers_near(input_data.data(), output_data.data(), 2U, "mean 1x1");
}

TEST(GraphMeanHw, RejectsWrongOutputShapeAndOutsideSubset) {
    const uint32_t in_dims[4] = {1U, 2U, 2U, 2U};
    const uint32_t bad_out_dims[4] = {1U, 2U, 1U, 2U};

    std::vector<float> input_data(8U, 1.0F);
    std::vector<float> output_data(4U, 0.0F);

    infr_shape in_shape{};
    infr_shape bad_out_shape{};
    ASSERT_EQ(INFR_OK, infr_shape_init(&in_shape, 4U, in_dims));
    ASSERT_EQ(INFR_OK, infr_shape_init(&bad_out_shape, 4U, bad_out_dims));
    infr_tensor input{};
    infr_tensor bad_output{};
    ASSERT_EQ(INFR_OK, infr_tensor_init(&input, input_data.data(), INFR_DTYPE_F32, &in_shape));
    ASSERT_EQ(INFR_OK, infr_tensor_init(&bad_output, output_data.data(), INFR_DTYPE_F32, &bad_out_shape));

    alignas(8) uint8_t workspace_bytes[kWorkspaceBytes] = {};
    infr_workspace workspace{};
    ASSERT_EQ(INFR_OK, infr_workspace_init(&workspace, workspace_bytes, sizeof(workspace_bytes)));

    /* `[1,2,1,C]` is not a global mean; writing into it would place the result at the
     * wrong stride, so it is refused. */
    EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, infr_mean_hw_f32(&input, &bad_output, &workspace));
    EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, infr_mean_hw_f32(nullptr, &bad_output, &workspace));

    infr_tensor quantized = input;
    quantized.quant.kind = INFR_QUANT_PER_TENSOR;
    quantized.quant.scale = 0.5F;
    EXPECT_EQ(INFR_ERR_UNSUPPORTED, infr_mean_hw_f32(&quantized, &bad_output, &workspace));

    /* Too small for one 2x2 plane (4 floats = 16 bytes). */
    alignas(8) uint8_t tiny[8] = {};
    infr_workspace small_ws{};
    ASSERT_EQ(INFR_OK, infr_workspace_init(&small_ws, tiny, sizeof(tiny)));
    const uint32_t good_out_dims[4] = {1U, 1U, 1U, 2U};
    infr_shape good_out_shape{};
    ASSERT_EQ(INFR_OK, infr_shape_init(&good_out_shape, 4U, good_out_dims));
    infr_tensor good_output{};
    ASSERT_EQ(INFR_OK, infr_tensor_init(&good_output, output_data.data(), INFR_DTYPE_F32, &good_out_shape));
    EXPECT_EQ(INFR_ERR_BUFFER_TOO_SMALL, infr_mean_hw_f32(&input, &good_output, &small_ws));
    EXPECT_EQ(0U, infr_workspace_used(&small_ws));
}


int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
