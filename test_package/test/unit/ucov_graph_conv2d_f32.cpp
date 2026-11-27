// Conan::ImportStart
#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include <infinfer/infinfer.h>

#include "parity_harness.hpp"
// Conan::ImportEnd

/*!
 * @brief [en] `infr_conv2d_f32`: the L2 composition judged against the mathematical definition.
 * @brief [zh] `infr_conv2d_f32`：按数学定义判定的 L2 组合。
 * @since 0.1.0
 */

namespace {

/* The reference below is written straight from the definition of a 2-D convolution
 * (sum over kernel position and input channel, zero outside the padding), in double
 * precision, and it is handed the output extents as literals. It deliberately does
 * NOT call `infr_conv2d_output_shape`, so a wrong extent formula shows up as a failed
 * EXPECT on the shape instead of being reproduced by the reference.
 *
 * 下面的参考实现直接照二维卷积的定义写（对核位置与输入通道求和，填充区取零），用
 * double 精度，且输出尺寸由字面量传入。它**不**调用 `infr_conv2d_output_shape`，
 * 所以尺寸公式写错会表现为 shape 断言失败，而不是被参考实现一起复现。 */
std::vector<double> reference_conv(const std::vector<float>& in, uint32_t in_h, uint32_t in_w, uint32_t in_c,
                                   const std::vector<float>& weights, uint32_t out_h, uint32_t out_w, uint32_t out_c,
                                   uint32_t kh, uint32_t kw, uint32_t stride_h, uint32_t stride_w, uint32_t pad_h,
                                   uint32_t pad_w, uint32_t groups, const std::vector<float>* bias) {
    const uint32_t in_per_group = in_c / groups;
    const uint32_t out_per_group = out_c / groups;
    std::vector<double> out(static_cast<size_t>(out_h) * out_w * out_c, 0.0);

    for (uint32_t oy = 0; oy < out_h; ++oy) {
        for (uint32_t ox = 0; ox < out_w; ++ox) {
            for (uint32_t oc = 0; oc < out_c; ++oc) {
                const uint32_t group = oc / out_per_group;
                double acc = 0.0;
                for (uint32_t ky = 0; ky < kh; ++ky) {
                    for (uint32_t kx = 0; kx < kw; ++kx) {
                        for (uint32_t ci = 0; ci < in_per_group; ++ci) {
                            const int64_t iy = static_cast<int64_t>(oy) * stride_h + ky - pad_h;
                            const int64_t ix = static_cast<int64_t>(ox) * stride_w + kx - pad_w;
                            if (iy < 0 || ix < 0 || iy >= static_cast<int64_t>(in_h) ||
                                ix >= static_cast<int64_t>(in_w)) {
                                continue;
                            }
                            const size_t in_index =
                                ((static_cast<size_t>(iy) * in_w + static_cast<size_t>(ix)) * in_c +
                                 (static_cast<size_t>(group) * in_per_group) + ci);
                            const size_t w_index =
                                ((static_cast<size_t>(oc) * kh + ky) * kw + kx) * in_per_group + ci;
                            acc += static_cast<double>(in[in_index]) * static_cast<double>(weights[w_index]);
                        }
                    }
                }
                if (bias != nullptr) {
                    acc += static_cast<double>((*bias)[oc]);
                }
                out[(static_cast<size_t>(oy) * out_w + ox) * out_c + oc] = acc;
            }
        }
    }
    return out;
}

std::vector<float> ramp(size_t count, float step, float origin) {
    std::vector<float> values(count);
    for (size_t i = 0; i < count; ++i) {
        values[i] = origin + (step * static_cast<float>(i));
    }
    return values;
}

struct conv_fixture {
    infr_tensor input{};
    infr_tensor filter{};
    infr_tensor output{};
    infr_conv2d_params params{};
};

/* Build the tensors for one case and return the shapes it used, so the assertions
 * can talk about literals instead of indices. */
void bind(conv_fixture* fixture, std::vector<float>* input_data, std::vector<float>* filter_data,
          std::vector<float>* output_data, const uint32_t in_dims[4], const uint32_t filter_dims[4],
          const uint32_t out_dims[4]) {
    infr_shape in_shape{};
    infr_shape filter_shape{};
    infr_shape out_shape{};
    EXPECT_EQ(INFR_OK, infr_shape_init(&in_shape, 4U, in_dims));
    EXPECT_EQ(INFR_OK, infr_shape_init(&filter_shape, 4U, filter_dims));
    EXPECT_EQ(INFR_OK, infr_shape_init(&out_shape, 4U, out_dims));
    EXPECT_EQ(INFR_OK, infr_tensor_init(&fixture->input, input_data->data(), INFR_DTYPE_F32, &in_shape));
    EXPECT_EQ(INFR_OK, infr_tensor_init(&fixture->filter, filter_data->data(), INFR_DTYPE_F32, &filter_shape));
    EXPECT_EQ(INFR_OK, infr_tensor_init(&fixture->output, output_data->data(), INFR_DTYPE_F32, &out_shape));
}

constexpr size_t kWorkspaceBytes = 4096U;

}  // namespace

TEST(GraphConv2d, OutputShapeMatchesHandComputedExtents) {
    const uint32_t in_dims[4] = {1U, 5U, 5U, 3U};
    infr_shape in_shape{};
    ASSERT_EQ(INFR_OK, infr_shape_init(&in_shape, 4U, in_dims));

    infr_conv2d_params params{};
    params.kernel_h = 3U;
    params.kernel_w = 3U;
    params.stride_h = 1U;
    params.stride_w = 1U;
    params.pad_h = 1U;
    params.pad_w = 1U;
    params.in_channels = 3U;
    params.out_channels = 4U;
    params.groups = 1U;

    infr_shape out_shape{};
    ASSERT_EQ(INFR_OK, infr_conv2d_output_shape(&params, &in_shape, &out_shape));
    EXPECT_EQ(4U, out_shape.rank);
    EXPECT_EQ(1U, out_shape.dims[0]);
    EXPECT_EQ(5U, out_shape.dims[1]); /* (5 + 2 - 3)/1 + 1 */
    EXPECT_EQ(5U, out_shape.dims[2]);
    EXPECT_EQ(4U, out_shape.dims[3]);

    params.pad_h = 0U;
    params.pad_w = 0U;
    params.stride_h = 2U;
    params.stride_w = 2U;
    ASSERT_EQ(INFR_OK, infr_conv2d_output_shape(&params, &in_shape, &out_shape));
    EXPECT_EQ(2U, out_shape.dims[1]); /* (5 - 3)/2 + 1 */

    params.kernel_h = 7U; /* larger than the padded input */
    EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, infr_conv2d_output_shape(&params, &in_shape, &out_shape));
    params.kernel_h = 3U;
    params.stride_h = 0U;
    EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, infr_conv2d_output_shape(&params, &in_shape, &out_shape));
}

TEST(GraphConv2d, DenseMatchesDoubleReference) {
    const uint32_t in_dims[4] = {1U, 4U, 4U, 1U};
    const uint32_t filter_dims[4] = {2U, 3U, 3U, 1U};
    const uint32_t out_dims[4] = {1U, 2U, 2U, 2U}; /* (4-3)/1+1 = 2 */

    auto input_data = ramp(16U, 0.25F, -1.5F);
    auto filter_data = ramp(18U, 0.125F, -0.5F);
    std::vector<float> output_data(8U, 0.0F);

    conv_fixture fixture{};
    bind(&fixture, &input_data, &filter_data, &output_data, in_dims, filter_dims, out_dims);
    fixture.params = infr_conv2d_params{3U, 3U, 1U, 1U, 0U, 0U, 1U, 2U, 1U};

    alignas(8) uint8_t workspace_bytes[kWorkspaceBytes] = {};
    infr_workspace workspace{};
    ASSERT_EQ(INFR_OK, infr_workspace_init(&workspace, workspace_bytes, sizeof(workspace_bytes)));

    ASSERT_EQ(INFR_OK, infr_conv2d_f32(&fixture.input, &fixture.filter, nullptr, &fixture.params, &fixture.output,
                                       &workspace));

    const auto want = reference_conv(input_data, 4U, 4U, 1U, filter_data, 2U, 2U, 2U, 3U, 3U, 1U, 1U, 0U, 0U, 1U,
                                     nullptr);
    const std::vector<float> want_f(want.begin(), want.end());
    infr_test::expect_float_buffers_near(want_f.data(), output_data.data(), want_f.size(), "conv2d dense");

    /* The scratch is given back: a graph that runs N convolutions must not need N
     * times the workspace. */
    EXPECT_EQ(0U, infr_workspace_used(&workspace));
}

TEST(GraphConv2d, SamePaddingKeepsSpatialExtent) {
    const uint32_t in_dims[4] = {1U, 3U, 3U, 2U};
    const uint32_t filter_dims[4] = {1U, 3U, 3U, 2U};
    const uint32_t out_dims[4] = {1U, 3U, 3U, 1U};

    auto input_data = ramp(18U, 0.5F, -4.0F);
    auto filter_data = ramp(18U, -0.25F, 1.0F);
    std::vector<float> output_data(9U, 0.0F);

    conv_fixture fixture{};
    bind(&fixture, &input_data, &filter_data, &output_data, in_dims, filter_dims, out_dims);
    fixture.params = infr_conv2d_params{3U, 3U, 1U, 1U, 1U, 1U, 2U, 1U, 1U};

    alignas(8) uint8_t workspace_bytes[kWorkspaceBytes] = {};
    infr_workspace workspace{};
    ASSERT_EQ(INFR_OK, infr_workspace_init(&workspace, workspace_bytes, sizeof(workspace_bytes)));
    ASSERT_EQ(INFR_OK, infr_conv2d_f32(&fixture.input, &fixture.filter, nullptr, &fixture.params, &fixture.output,
                                       &workspace));

    const auto want = reference_conv(input_data, 3U, 3U, 2U, filter_data, 3U, 3U, 1U, 3U, 3U, 1U, 1U, 1U, 1U, 1U,
                                     nullptr);
    const std::vector<float> want_f(want.begin(), want.end());
    infr_test::expect_float_buffers_near(want_f.data(), output_data.data(), want_f.size(), "conv2d same-pad");
}

TEST(GraphConv2d, StridedMatchesDoubleReference) {
    const uint32_t in_dims[4] = {1U, 5U, 5U, 1U};
    const uint32_t filter_dims[4] = {1U, 3U, 3U, 1U};
    const uint32_t out_dims[4] = {1U, 2U, 2U, 1U}; /* (5-3)/2+1 = 2 */

    auto input_data = ramp(25U, 0.1F, -1.0F);
    auto filter_data = ramp(9U, 0.2F, -0.5F);
    std::vector<float> output_data(4U, 0.0F);

    conv_fixture fixture{};
    bind(&fixture, &input_data, &filter_data, &output_data, in_dims, filter_dims, out_dims);
    fixture.params = infr_conv2d_params{3U, 3U, 2U, 2U, 0U, 0U, 1U, 1U, 1U};

    alignas(8) uint8_t workspace_bytes[kWorkspaceBytes] = {};
    infr_workspace workspace{};
    ASSERT_EQ(INFR_OK, infr_workspace_init(&workspace, workspace_bytes, sizeof(workspace_bytes)));
    ASSERT_EQ(INFR_OK, infr_conv2d_f32(&fixture.input, &fixture.filter, nullptr, &fixture.params, &fixture.output,
                                       &workspace));

    const auto want = reference_conv(input_data, 5U, 5U, 1U, filter_data, 2U, 2U, 1U, 3U, 3U, 2U, 2U, 0U, 0U, 1U,
                                     nullptr);
    const std::vector<float> want_f(want.begin(), want.end());
    infr_test::expect_float_buffers_near(want_f.data(), output_data.data(), want_f.size(), "conv2d stride 2");
}

TEST(GraphConv2d, DepthwiseUsesOneFilterRowPerChannel) {
    const uint32_t in_dims[4] = {1U, 4U, 4U, 3U};
    const uint32_t filter_dims[4] = {3U, 3U, 3U, 1U}; /* groups == in_channels */
    const uint32_t out_dims[4] = {1U, 4U, 4U, 3U};     /* pad 1 keeps the extent */

    auto input_data = ramp(48U, 0.05F, -0.7F);
    auto filter_data = ramp(27U, 0.1F, -0.9F);
    std::vector<float> output_data(48U, 0.0F);

    conv_fixture fixture{};
    bind(&fixture, &input_data, &filter_data, &output_data, in_dims, filter_dims, out_dims);
    fixture.params = infr_conv2d_params{3U, 3U, 1U, 1U, 1U, 1U, 3U, 3U, 3U};

    alignas(8) uint8_t workspace_bytes[kWorkspaceBytes] = {};
    infr_workspace workspace{};
    ASSERT_EQ(INFR_OK, infr_workspace_init(&workspace, workspace_bytes, sizeof(workspace_bytes)));
    ASSERT_EQ(INFR_OK, infr_conv2d_f32(&fixture.input, &fixture.filter, nullptr, &fixture.params, &fixture.output,
                                       &workspace));

    const auto want = reference_conv(input_data, 4U, 4U, 3U, filter_data, 4U, 4U, 3U, 3U, 3U, 1U, 1U, 1U, 1U, 3U,
                                     nullptr);
    const std::vector<float> want_f(want.begin(), want.end());
    infr_test::expect_float_buffers_near(want_f.data(), output_data.data(), want_f.size(), "conv2d depthwise");
}

TEST(GraphConv2d, BiasIsAddedPerOutputChannel) {
    const uint32_t in_dims[4] = {1U, 3U, 3U, 1U};
    const uint32_t filter_dims[4] = {2U, 3U, 3U, 1U};
    const uint32_t out_dims[4] = {1U, 1U, 1U, 2U};

    auto input_data = ramp(9U, 0.3F, 0.5F);
    auto filter_data = ramp(18U, -0.1F, 0.25F);
    const std::vector<float> bias = {0.75F, -1.25F};

    conv_fixture with_bias{};
    std::vector<float> out_with_bias(2U, 0.0F);
    bind(&with_bias, &input_data, &filter_data, &out_with_bias, in_dims, filter_dims, out_dims);
    with_bias.params = infr_conv2d_params{3U, 3U, 1U, 1U, 0U, 0U, 1U, 2U, 1U};

    conv_fixture without_bias{};
    std::vector<float> out_plain(2U, 0.0F);
    bind(&without_bias, &input_data, &filter_data, &out_plain, in_dims, filter_dims, out_dims);
    without_bias.params = with_bias.params;

    alignas(8) uint8_t workspace_bytes[kWorkspaceBytes] = {};
    infr_workspace workspace{};
    ASSERT_EQ(INFR_OK, infr_workspace_init(&workspace, workspace_bytes, sizeof(workspace_bytes)));
    ASSERT_EQ(INFR_OK, infr_conv2d_f32(&with_bias.input, &with_bias.filter, bias.data(), &with_bias.params,
                                       &with_bias.output, &workspace));
    ASSERT_EQ(INFR_OK, infr_conv2d_f32(&without_bias.input, &without_bias.filter, nullptr, &without_bias.params,
                                       &without_bias.output, &workspace));

    for (size_t oc = 0; oc < bias.size(); ++oc) {
        EXPECT_NEAR(out_plain[oc] + bias[oc], out_with_bias[oc], 1e-6F) << "channel " << oc;
    }

    const auto want = reference_conv(input_data, 3U, 3U, 1U, filter_data, 1U, 1U, 2U, 3U, 3U, 1U, 1U, 0U, 0U, 1U,
                                     &bias);
    const std::vector<float> want_f(want.begin(), want.end());
    infr_test::expect_float_buffers_near(want_f.data(), out_with_bias.data(), want_f.size(), "conv2d bias");
}

TEST(GraphConv2d, SmallWorkspaceIsReportedAndLeavesTheCursorAlone) {
    const uint32_t in_dims[4] = {1U, 8U, 8U, 4U};
    const uint32_t filter_dims[4] = {2U, 3U, 3U, 4U};
    const uint32_t out_dims[4] = {1U, 6U, 6U, 2U};

    auto input_data = ramp(256U, 0.01F, 0.0F);
    auto filter_data = ramp(72U, 0.01F, 0.0F);
    std::vector<float> output_data(72U, 0.0F);

    conv_fixture fixture{};
    bind(&fixture, &input_data, &filter_data, &output_data, in_dims, filter_dims, out_dims);
    fixture.params = infr_conv2d_params{3U, 3U, 1U, 1U, 0U, 0U, 4U, 2U, 1U};

    /* 3*3*4 + 2 = 38 floats are needed; 16 bytes cannot hold them. */
    alignas(8) uint8_t workspace_bytes[16] = {};
    infr_workspace workspace{};
    ASSERT_EQ(INFR_OK, infr_workspace_init(&workspace, workspace_bytes, sizeof(workspace_bytes)));
    EXPECT_EQ(INFR_ERR_BUFFER_TOO_SMALL, infr_conv2d_f32(&fixture.input, &fixture.filter, nullptr, &fixture.params,
                                                         &fixture.output, &workspace));
    EXPECT_EQ(0U, infr_workspace_used(&workspace));
}

TEST(GraphConv2d, RejectsArgumentsOutsideTheStatedSubset) {
    const uint32_t in_dims[4] = {1U, 4U, 4U, 4U};
    const uint32_t filter_dims[4] = {4U, 3U, 3U, 2U};
    const uint32_t out_dims[4] = {1U, 2U, 2U, 4U};

    auto input_data = ramp(64U, 0.01F, 0.0F);
    auto filter_data = ramp(72U, 0.01F, 0.0F);
    std::vector<float> output_data(16U, 0.0F);

    conv_fixture fixture{};
    bind(&fixture, &input_data, &filter_data, &output_data, in_dims, filter_dims, out_dims);
    fixture.params = infr_conv2d_params{3U, 3U, 1U, 1U, 0U, 0U, 4U, 4U, 2U};

    alignas(8) uint8_t workspace_bytes[kWorkspaceBytes] = {};
    infr_workspace workspace{};
    ASSERT_EQ(INFR_OK, infr_workspace_init(&workspace, workspace_bytes, sizeof(workspace_bytes)));

    /* General grouped convolution is expressible but not promised. */
    EXPECT_EQ(INFR_ERR_UNSUPPORTED, infr_conv2d_f32(&fixture.input, &fixture.filter, nullptr, &fixture.params,
                                                    &fixture.output, &workspace));

    /* NULL anywhere is an argument error, not a crash. */
    EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT,
              infr_conv2d_f32(nullptr, &fixture.filter, nullptr, &fixture.params, &fixture.output, &workspace));
    EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, infr_conv2d_f32(&fixture.input, &fixture.filter, nullptr, nullptr,
                                                         &fixture.output, &workspace));
    EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, infr_conv2d_f32(&fixture.input, &fixture.filter, nullptr, &fixture.params,
                                                         &fixture.output, nullptr));

    /* A quantized tensor is outside the subset this operator declares. */
    infr_tensor quantized = fixture.input;
    quantized.quant.kind = INFR_QUANT_PER_TENSOR;
    quantized.quant.scale = 0.5F;
    fixture.params.groups = 1U;
    EXPECT_EQ(INFR_ERR_UNSUPPORTED, infr_conv2d_f32(&quantized, &fixture.filter, nullptr, &fixture.params,
                                                    &fixture.output, &workspace));
}


int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
