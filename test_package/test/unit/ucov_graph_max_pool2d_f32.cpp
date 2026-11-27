// Conan::ImportStart
#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <vector>

#include <infinfer/infinfer.h>

#include "parity_harness.hpp"
// Conan::ImportEnd

/*!
 * @brief [en] `infr_max_pool2d_f32`: padding must behave like -infinity, not like zero.
 * @brief [zh] `infr_max_pool2d_f32`：填充必须表现得像负无穷，而不是零。
 * @since 0.1.0
 */

namespace {

/* Reference written from the definition: the maximum over the in-bounds part of the
 * window. Padding is skipped rather than compared, which is the same thing as -inf
 * only because the operator forbids all-padding windows.
 *
 * 参考实现照定义写：窗口内落在有效区域的元素取最大。填充被跳过而不是参与比较——它
 * 与"填负无穷"等价，前提是算子禁止全填充窗口，而那正是它的约束。 */
std::vector<float> reference_max_pool(const std::vector<float>& in, uint32_t in_h, uint32_t in_w, uint32_t channels,
                                      uint32_t out_h, uint32_t out_w, uint32_t kh, uint32_t kw, uint32_t stride_h,
                                      uint32_t stride_w, uint32_t pad_h, uint32_t pad_w) {
    std::vector<float> out(static_cast<size_t>(out_h) * out_w * channels, 0.0F);
    for (uint32_t oy = 0; oy < out_h; ++oy) {
        for (uint32_t ox = 0; ox < out_w; ++ox) {
            for (uint32_t c = 0; c < channels; ++c) {
                float best = -std::numeric_limits<float>::infinity();
                for (uint32_t ky = 0; ky < kh; ++ky) {
                    for (uint32_t kx = 0; kx < kw; ++kx) {
                        const int64_t iy = static_cast<int64_t>(oy) * stride_h + ky - pad_h;
                        const int64_t ix = static_cast<int64_t>(ox) * stride_w + kx - pad_w;
                        if (iy < 0 || ix < 0 || iy >= static_cast<int64_t>(in_h) || ix >= static_cast<int64_t>(in_w)) {
                            continue;
                        }
                        const size_t index =
                            ((static_cast<size_t>(iy) * in_w) + static_cast<size_t>(ix)) * channels + c;
                        best = in[index] > best ? in[index] : best;
                    }
                }
                out[(static_cast<size_t>(oy) * out_w + ox) * channels + c] = best;
            }
        }
    }
    return out;
}

constexpr size_t kWorkspaceBytes = 1024U;

struct pool_fixture {
    infr_tensor input{};
    infr_tensor output{};
    infr_pool_params params{};
};

void bind(pool_fixture* fixture, std::vector<float>* input_data, std::vector<float>* output_data,
          const uint32_t in_dims[4], const uint32_t out_dims[4]) {
    infr_shape in_shape{};
    infr_shape out_shape{};
    EXPECT_EQ(INFR_OK, infr_shape_init(&in_shape, 4U, in_dims));
    EXPECT_EQ(INFR_OK, infr_shape_init(&out_shape, 4U, out_dims));
    EXPECT_EQ(INFR_OK, infr_tensor_init(&fixture->input, input_data->data(), INFR_DTYPE_F32, &in_shape));
    EXPECT_EQ(INFR_OK, infr_tensor_init(&fixture->output, output_data->data(), INFR_DTYPE_F32, &out_shape));
}

}  // namespace

TEST(GraphMaxPool2d, OutputShapeMatchesHandComputedExtents) {
    const uint32_t in_dims[4] = {1U, 6U, 6U, 2U};
    infr_shape in_shape{};
    ASSERT_EQ(INFR_OK, infr_shape_init(&in_shape, 4U, in_dims));

    infr_pool_params params{2U, 2U, 2U, 2U, 0U, 0U, 2U};
    infr_shape out_shape{};
    ASSERT_EQ(INFR_OK, infr_max_pool2d_output_shape(&params, &in_shape, &out_shape));
    EXPECT_EQ(3U, out_shape.dims[1]); /* (6-2)/2 + 1 */
    EXPECT_EQ(3U, out_shape.dims[2]);
    EXPECT_EQ(2U, out_shape.dims[3]);

    params.stride_h = 1U;
    params.stride_w = 1U;
    ASSERT_EQ(INFR_OK, infr_max_pool2d_output_shape(&params, &in_shape, &out_shape));
    EXPECT_EQ(5U, out_shape.dims[1]);

    params.pad_h = 2U; /* padding may not swallow a whole window */
    EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, infr_max_pool2d_output_shape(&params, &in_shape, &out_shape));
}

TEST(GraphMaxPool2d, MatchesReferenceWithoutPadding) {
    const uint32_t in_dims[4] = {1U, 4U, 4U, 1U};
    const uint32_t out_dims[4] = {1U, 2U, 2U, 1U};

    std::vector<float> input_data = {1.0F, 2.0F, 3.0F, 4.0F,  5.0F,  6.0F,  7.0F,  8.0F,
                                     9.0F, 10.0F, 11.0F, 12.0F, 13.0F, 14.0F, 15.0F, 16.0F};
    std::vector<float> output_data(4U, 0.0F);

    pool_fixture fixture{};
    bind(&fixture, &input_data, &output_data, in_dims, out_dims);
    fixture.params = infr_pool_params{2U, 2U, 2U, 2U, 0U, 0U, 1U};

    alignas(8) uint8_t workspace_bytes[kWorkspaceBytes] = {};
    infr_workspace workspace{};
    ASSERT_EQ(INFR_OK, infr_workspace_init(&workspace, workspace_bytes, sizeof(workspace_bytes)));
    ASSERT_EQ(INFR_OK,
              infr_max_pool2d_f32(&fixture.input, &fixture.params, &fixture.output, &workspace));

    /* 2x2 non-overlapping windows: the bottom-right element of each window. */
    const std::vector<float> want = {6.0F, 8.0F, 14.0F, 16.0F};
    infr_test::expect_float_buffers_near(want.data(), output_data.data(), want.size(), "maxpool 2x2/2");
    EXPECT_EQ(0U, infr_workspace_used(&workspace));
}

TEST(GraphMaxPool2d, PaddingDoesNotActLikeZero) {
    /* Every value is negative, so a zero pad would win the maximum at every border
     * position. This is the case that separates "pad = -inf" from "pad = 0". */
    const uint32_t in_dims[4] = {1U, 3U, 3U, 1U};
    const uint32_t out_dims[4] = {1U, 3U, 3U, 1U};

    std::vector<float> input_data = {-9.0F, -8.0F, -7.0F, -6.0F, -5.0F, -4.0F, -3.0F, -2.0F, -1.0F};
    std::vector<float> output_data(9U, 0.0F);

    pool_fixture fixture{};
    bind(&fixture, &input_data, &output_data, in_dims, out_dims);
    fixture.params = infr_pool_params{3U, 3U, 1U, 1U, 1U, 1U, 1U};

    alignas(8) uint8_t workspace_bytes[kWorkspaceBytes] = {};
    infr_workspace workspace{};
    ASSERT_EQ(INFR_OK, infr_workspace_init(&workspace, workspace_bytes, sizeof(workspace_bytes)));
    ASSERT_EQ(INFR_OK,
              infr_max_pool2d_f32(&fixture.input, &fixture.params, &fixture.output, &workspace));

    const auto want = reference_max_pool(input_data, 3U, 3U, 1U, 3U, 3U, 3U, 3U, 1U, 1U, 1U, 1U);
    infr_test::expect_float_buffers_near(want.data(), output_data.data(), want.size(), "maxpool padded");

    /* The corner sees only four of the nine inputs; if padding were zero, this would
     * be 0.0 -- the whole feature map would be shifted up to zero. */
    EXPECT_LT(output_data[0], 0.0F);
    EXPECT_FLOAT_EQ(-5.0F, output_data[0]);
}

TEST(GraphMaxPool2d, ChannelsAreReducedIndependently) {
    const uint32_t in_dims[4] = {1U, 2U, 2U, 2U};
    const uint32_t out_dims[4] = {1U, 1U, 1U, 2U};

    std::vector<float> input_data = {1.0F, -1.0F, 2.0F, -2.0F, 3.0F, -3.0F, 4.0F, -4.0F};
    std::vector<float> output_data(2U, 0.0F);

    pool_fixture fixture{};
    bind(&fixture, &input_data, &output_data, in_dims, out_dims);
    fixture.params = infr_pool_params{2U, 2U, 2U, 2U, 0U, 0U, 2U};

    alignas(8) uint8_t workspace_bytes[kWorkspaceBytes] = {};
    infr_workspace workspace{};
    ASSERT_EQ(INFR_OK, infr_workspace_init(&workspace, workspace_bytes, sizeof(workspace_bytes)));
    ASSERT_EQ(INFR_OK,
              infr_max_pool2d_f32(&fixture.input, &fixture.params, &fixture.output, &workspace));

    EXPECT_FLOAT_EQ(4.0F, output_data[0]);
    EXPECT_FLOAT_EQ(-1.0F, output_data[1]);
}

TEST(GraphMaxPool2d, RejectsArgumentsOutsideTheStatedSubset) {
    const uint32_t in_dims[4] = {1U, 4U, 4U, 1U};
    const uint32_t out_dims[4] = {1U, 2U, 2U, 1U};

    std::vector<float> input_data(16U, 1.0F);
    std::vector<float> output_data(4U, 0.0F);

    pool_fixture fixture{};
    bind(&fixture, &input_data, &output_data, in_dims, out_dims);
    fixture.params = infr_pool_params{2U, 2U, 2U, 2U, 0U, 0U, 1U};

    alignas(8) uint8_t workspace_bytes[kWorkspaceBytes] = {};
    infr_workspace workspace{};
    ASSERT_EQ(INFR_OK, infr_workspace_init(&workspace, workspace_bytes, sizeof(workspace_bytes)));

    EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT,
              infr_max_pool2d_f32(nullptr, &fixture.params, &fixture.output, &workspace));
    EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT,
              infr_max_pool2d_f32(&fixture.input, nullptr, &fixture.output, &workspace));
    EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, infr_max_pool2d_f32(&fixture.input, &fixture.params, &fixture.output, nullptr));

    /* The window is larger than the input gives room for. */
    fixture.params = infr_pool_params{9U, 9U, 1U, 1U, 0U, 0U, 1U};
    EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT,
              infr_max_pool2d_f32(&fixture.input, &fixture.params, &fixture.output, &workspace));

    /* Quantized tensors are outside the subset this operator declares. */
    fixture.params = infr_pool_params{2U, 2U, 2U, 2U, 0U, 0U, 1U};
    infr_tensor quantized = fixture.input;
    quantized.quant.kind = INFR_QUANT_PER_TENSOR;
    quantized.quant.scale = 0.25F;
    EXPECT_EQ(INFR_ERR_UNSUPPORTED,
              infr_max_pool2d_f32(&quantized, &fixture.params, &fixture.output, &workspace));
    EXPECT_EQ(0U, infr_workspace_used(&workspace));
}


int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
