// Conan::ImportEnd
//
// `infr_conv2d_i8`: the quantised L2 composition judged the same way `infr_conv2d_f32` is -- against
// the definition, written out independently -- but the definition now has three parts that a float
// version does not have, and each one gets a case that fails if it is wrong:
//
//   1. the multiplier is `(input.scale * weight.scale[oc]) / output.scale`, resolved per channel
//   2. the int32 bias is added to the accumulator *before* the single rounding step
//   3. the output zero point is an offset applied after that rounding
//
// The reference accumulates in int64 and applies the written rounding rule in double, so it shares
// no code with the operator or with the kernels it calls.
#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include <infinfer/infinfer.h>

namespace {

constexpr size_t kWorkspaceBytes = 4096U;

struct i8_fixture {
    infr_tensor input{};
    infr_tensor filter{};
    infr_tensor output{};
    infr_conv2d_params params{};
};

void bind_i8(i8_fixture* fixture, std::vector<std::int8_t>* input_data, std::vector<std::int8_t>* filter_data,
             std::vector<std::int8_t>* output_data, const uint32_t in_dims[4], const uint32_t filter_dims[4],
             const uint32_t out_dims[4]) {
    infr_shape in_shape{};
    infr_shape filter_shape{};
    infr_shape out_shape{};
    EXPECT_EQ(INFR_OK, infr_shape_init(&in_shape, 4U, in_dims));
    EXPECT_EQ(INFR_OK, infr_shape_init(&filter_shape, 4U, filter_dims));
    EXPECT_EQ(INFR_OK, infr_shape_init(&out_shape, 4U, out_dims));
    EXPECT_EQ(INFR_OK, infr_tensor_init(&fixture->input, input_data->data(), INFR_DTYPE_I8, &in_shape));
    EXPECT_EQ(INFR_OK, infr_tensor_init(&fixture->filter, filter_data->data(), INFR_DTYPE_I8, &filter_shape));
    EXPECT_EQ(INFR_OK, infr_tensor_init(&fixture->output, output_data->data(), INFR_DTYPE_I8, &out_shape));
}

// The written rule, transcribed: `(accumulator + bias) * multiplier + zero_point`, saturated in the
// double domain, rounded half away from zero. Nothing here calls the library.
std::int8_t reference_requant(std::int64_t accumulator, std::int32_t bias, double multiplier, std::int32_t zero_point) {
    const double scaled = (static_cast<double>(accumulator + bias) * multiplier) + static_cast<double>(zero_point);
    if (scaled <= static_cast<double>(std::numeric_limits<std::int8_t>::min())) {
        return std::numeric_limits<std::int8_t>::min();
    }
    if (scaled >= static_cast<double>(std::numeric_limits<std::int8_t>::max())) {
        return std::numeric_limits<std::int8_t>::max();
    }
    return static_cast<std::int8_t>(scaled >= 0.0 ? scaled + 0.5 : scaled - 0.5);
}

std::vector<std::int8_t> reference_conv_i8(const std::vector<std::int8_t>& in, uint32_t in_h, uint32_t in_w,
                                           uint32_t in_c, const std::vector<std::int8_t>& weights, uint32_t out_h,
                                           uint32_t out_w, uint32_t out_c, uint32_t kh, uint32_t kw, uint32_t stride_h,
                                           uint32_t stride_w, uint32_t pad_h, uint32_t pad_w, uint32_t groups,
                                           const std::vector<std::int32_t>* bias,
                                           const std::vector<float>& weight_scales, float input_scale,
                                           float output_scale, std::int32_t output_zero_point) {
    const uint32_t in_per_group = in_c / groups;
    const uint32_t out_per_group = out_c / groups;
    std::vector<std::int8_t> out(static_cast<size_t>(out_h) * out_w * out_c, 0);

    for (uint32_t oy = 0; oy < out_h; ++oy) {
        for (uint32_t ox = 0; ox < out_w; ++ox) {
            for (uint32_t oc = 0; oc < out_c; ++oc) {
                const uint32_t group = oc / out_per_group;
                std::int64_t acc = 0;
                for (uint32_t ky = 0; ky < kh; ++ky) {
                    for (uint32_t kx = 0; kx < kw; ++kx) {
                        for (uint32_t ci = 0; ci < in_per_group; ++ci) {
                            const std::int64_t iy = static_cast<std::int64_t>(oy) * stride_h + ky - pad_h;
                            const std::int64_t ix = static_cast<std::int64_t>(ox) * stride_w + kx - pad_w;
                            if (iy < 0 || ix < 0 || iy >= static_cast<std::int64_t>(in_h) ||
                                ix >= static_cast<std::int64_t>(in_w)) {
                                continue;
                            }
                            const size_t in_index = ((static_cast<size_t>(iy) * in_w + static_cast<size_t>(ix)) * in_c +
                                                     (static_cast<size_t>(group) * in_per_group) + ci);
                            const size_t w_index = ((static_cast<size_t>(oc) * kh + ky) * kw + kx) * in_per_group + ci;
                            acc +=
                                static_cast<std::int64_t>(in[in_index]) * static_cast<std::int64_t>(weights[w_index]);
                        }
                    }
                }
                const std::int32_t channel_bias = bias != nullptr ? (*bias)[oc] : 0;
                const double multiplier = (static_cast<double>(input_scale) * weight_scales[oc]) / output_scale;
                out[(static_cast<size_t>(oy) * out_w + ox) * out_c + oc] =
                    reference_requant(acc, channel_bias, multiplier, output_zero_point);
            }
        }
    }
    return out;
}

// A deterministic int8 pattern in [-64, 63]: wide enough that a wrong sign or a dropped term shows
// up, narrow enough that a 3x3 window cannot overflow anything.
std::vector<std::int8_t> pattern(size_t count, int step) {
    std::vector<std::int8_t> values(count);
    for (size_t i = 0; i < count; ++i) {
        values[i] = static_cast<std::int8_t>(((static_cast<int>(i) * step) % 128) - 64);
    }
    return values;
}

/* One case, run end to end: build the tensors, quantise the descriptions, call the operator, and
 * compare against the reference byte by byte. `weight_scales` has one entry per output channel. */
void expect_conv_i8(uint32_t in_h, uint32_t in_w, uint32_t in_c, uint32_t out_c, uint32_t kh, uint32_t kw,
                    uint32_t stride, uint32_t pad, uint32_t groups, float input_scale, float output_scale,
                    std::int32_t output_zero_point, bool with_bias, bool per_channel) {
    const uint32_t out_h = ((in_h + (2U * pad) - kh) / stride) + 1U;
    const uint32_t out_w = ((in_w + (2U * pad) - kw) / stride) + 1U;
    const uint32_t in_dims[4] = {1U, in_h, in_w, in_c};
    const uint32_t filter_dims[4] = {out_c, kh, kw, in_c / groups};
    const uint32_t out_dims[4] = {1U, out_h, out_w, out_c};

    auto input_data = pattern(static_cast<size_t>(in_h) * in_w * in_c, 7);
    auto filter_data = pattern(static_cast<size_t>(out_c) * kh * kw * (in_c / groups), 11);
    std::vector<std::int8_t> output_data(static_cast<size_t>(out_h) * out_w * out_c, 0);
    std::vector<std::int32_t> bias_data;
    if (with_bias) {
        for (uint32_t oc = 0; oc < out_c; ++oc) {
            bias_data.push_back(static_cast<std::int32_t>(oc) * 17 - 9);
        }
    }

    std::vector<float> weight_scales(out_c, 0.0F);
    for (uint32_t oc = 0; oc < out_c; ++oc) {
        // Per-channel: a different scale per channel, all positive and distinct, so reading the
        // wrong channel's scale cannot pass.
        weight_scales[oc] = per_channel ? (0.010F + (0.002F * static_cast<float>(oc))) : 0.010F;
    }

    i8_fixture fixture{};
    bind_i8(&fixture, &input_data, &filter_data, &output_data, in_dims, filter_dims, out_dims);
    fixture.input.quant.kind = INFR_QUANT_PER_TENSOR;
    fixture.input.quant.scale = input_scale;
    fixture.input.quant.zero_point = 0;
    if (per_channel) {
        fixture.filter.quant.kind = INFR_QUANT_PER_CHANNEL;
        fixture.filter.quant.channel_scales = weight_scales.data();
    } else {
        fixture.filter.quant.kind = INFR_QUANT_PER_TENSOR;
        fixture.filter.quant.scale = weight_scales[0];
    }
    fixture.output.quant.kind = INFR_QUANT_PER_TENSOR;
    fixture.output.quant.scale = output_scale;
    fixture.output.quant.zero_point = output_zero_point;

    fixture.params.kernel_h = kh;
    fixture.params.kernel_w = kw;
    fixture.params.stride_h = stride;
    fixture.params.stride_w = stride;
    fixture.params.pad_h = pad;
    fixture.params.pad_w = pad;
    fixture.params.in_channels = in_c;
    fixture.params.out_channels = out_c;
    fixture.params.groups = groups;

    alignas(8) uint8_t workspace_bytes[kWorkspaceBytes] = {};
    infr_workspace workspace{};
    ASSERT_EQ(INFR_OK, infr_workspace_init(&workspace, workspace_bytes, sizeof(workspace_bytes)));
    ASSERT_EQ(INFR_OK, infr_conv2d_i8(&fixture.input, &fixture.filter, with_bias ? bias_data.data() : nullptr,
                                      &fixture.params, &fixture.output, &workspace));
    // The operator promises the scratch is handed back on every path, including success.
    EXPECT_EQ(0U, infr_workspace_used(&workspace));

    const auto want = reference_conv_i8(input_data, in_h, in_w, in_c, filter_data, out_h, out_w, out_c, kh, kw, stride,
                                        stride, pad, pad, groups, with_bias ? &bias_data : nullptr, weight_scales,
                                        input_scale, output_scale, output_zero_point);
    for (size_t i = 0; i < want.size(); ++i) {
        EXPECT_EQ(static_cast<int>(want[i]), static_cast<int>(output_data[i])) << "element " << i;
    }
}

} // namespace

TEST(GraphConv2dI8, DenseMatchesIntegerReference) {
    expect_conv_i8(4U, 4U, 1U, 2U, 3U, 3U, 1U, 0U, 1U, 0.02F, 0.05F, 0, true, false);
}

TEST(GraphConv2dI8, DepthwiseWithPaddingMatchesIntegerReference) {
    expect_conv_i8(5U, 5U, 4U, 4U, 3U, 3U, 1U, 1U, 4U, 0.02F, 0.05F, 0, true, false);
}

TEST(GraphConv2dI8, StrideTwoMatchesIntegerReference) {
    expect_conv_i8(6U, 6U, 3U, 4U, 3U, 3U, 2U, 1U, 1U, 0.015F, 0.04F, 0, false, false);
}

TEST(GraphConv2dI8, PerChannelScalesAndZeroPointAreBothApplied) {
    // A non-zero output zero point on purpose: an offset applied before the multiply would answer a
    // different vector, so this case pins the order as well as the per-channel scales.
    expect_conv_i8(4U, 4U, 2U, 3U, 3U, 3U, 1U, 1U, 1U, 0.02F, 0.05F, 40, true, true);
}

TEST(GraphConv2dI8, SaturatesAtBothEnds) {
    // A multiplier large enough that every accumulator leaves the 8-bit range: the result has to be
    // ±127, not a wrapped byte.
    expect_conv_i8(3U, 3U, 1U, 2U, 3U, 3U, 1U, 0U, 1U, 0.02F, 0.0005F, 0, true, false);
}

TEST(GraphConv2dI8, ZeroInputGivesTheOutputZeroPoint) {
    const uint32_t in_dims[4] = {1U, 3U, 3U, 2U};
    const uint32_t filter_dims[4] = {2U, 3U, 3U, 2U};
    const uint32_t out_dims[4] = {1U, 1U, 1U, 2U};
    std::vector<std::int8_t> input_data(18U, 0);
    auto filter_data = pattern(36U, 5);
    std::vector<std::int8_t> output_data(2U, 0);
    const float scales[2] = {0.01F, 0.01F};

    i8_fixture fixture{};
    bind_i8(&fixture, &input_data, &filter_data, &output_data, in_dims, filter_dims, out_dims);
    fixture.input.quant.kind = INFR_QUANT_PER_TENSOR;
    fixture.input.quant.scale = 0.02F;
    fixture.input.quant.zero_point = 0;
    fixture.filter.quant.kind = INFR_QUANT_PER_CHANNEL;
    fixture.filter.quant.channel_scales = scales;
    fixture.output.quant.kind = INFR_QUANT_PER_TENSOR;
    fixture.output.quant.scale = 0.05F;
    fixture.output.quant.zero_point = -12;
    fixture.params.kernel_h = 3U;
    fixture.params.kernel_w = 3U;
    fixture.params.stride_h = 3U;
    fixture.params.stride_w = 3U;
    fixture.params.in_channels = 2U;
    fixture.params.out_channels = 2U;
    fixture.params.groups = 1U;

    alignas(8) uint8_t workspace_bytes[kWorkspaceBytes] = {};
    infr_workspace workspace{};
    ASSERT_EQ(INFR_OK, infr_workspace_init(&workspace, workspace_bytes, sizeof(workspace_bytes)));
    ASSERT_EQ(INFR_OK,
              infr_conv2d_i8(&fixture.input, &fixture.filter, nullptr, &fixture.params, &fixture.output, &workspace));
    // A zero accumulator must land exactly on the zero point: that is what "the zero point is the
    // quantised zero" means, and it is the one value a mistake in the offset cannot hide.
    EXPECT_EQ(-12, static_cast<int>(output_data[0]));
    EXPECT_EQ(-12, static_cast<int>(output_data[1]));
}

TEST(GraphConv2dI8, RejectsSubsetsItDoesNotPromise) {
    const uint32_t in_dims[4] = {1U, 3U, 3U, 1U};
    const uint32_t filter_dims[4] = {1U, 3U, 3U, 1U};
    const uint32_t out_dims[4] = {1U, 1U, 1U, 1U};
    std::vector<std::int8_t> input_data(9U, 1);
    std::vector<std::int8_t> filter_data(9U, 1);
    std::vector<std::int8_t> output_data(1U, 0);
    const float one_scale = 0.01F;

    i8_fixture fixture{};
    bind_i8(&fixture, &input_data, &filter_data, &output_data, in_dims, filter_dims, out_dims);
    fixture.input.quant.kind = INFR_QUANT_PER_TENSOR;
    fixture.input.quant.scale = 0.02F;
    fixture.input.quant.zero_point = 0;
    fixture.filter.quant.kind = INFR_QUANT_PER_CHANNEL;
    fixture.filter.quant.channel_scales = &one_scale;
    fixture.output.quant.kind = INFR_QUANT_PER_TENSOR;
    fixture.output.quant.scale = 0.05F;
    fixture.params.kernel_h = 3U;
    fixture.params.kernel_w = 3U;
    fixture.params.stride_h = 1U;
    fixture.params.stride_w = 1U;
    fixture.params.in_channels = 1U;
    fixture.params.out_channels = 1U;
    fixture.params.groups = 1U;

    alignas(8) uint8_t workspace_bytes[kWorkspaceBytes] = {};
    infr_workspace workspace{};
    ASSERT_EQ(INFR_OK, infr_workspace_init(&workspace, workspace_bytes, sizeof(workspace_bytes)));

    // Asymmetric input: the padding byte would be wrong, so it is refused rather than approximated.
    fixture.input.quant.zero_point = -128;
    EXPECT_EQ(INFR_ERR_UNSUPPORTED,
              infr_conv2d_i8(&fixture.input, &fixture.filter, nullptr, &fixture.params, &fixture.output, &workspace));
    fixture.input.quant.zero_point = 0;

    // A filter with no scales at all, and an fp32 tensor where an int8 one is promised.
    fixture.filter.quant.channel_scales = nullptr;
    EXPECT_EQ(INFR_ERR_UNSUPPORTED,
              infr_conv2d_i8(&fixture.input, &fixture.filter, nullptr, &fixture.params, &fixture.output, &workspace));
    fixture.filter.quant.channel_scales = &one_scale;
    fixture.output.dtype = INFR_DTYPE_F32;
    EXPECT_EQ(INFR_ERR_UNSUPPORTED,
              infr_conv2d_i8(&fixture.input, &fixture.filter, nullptr, &fixture.params, &fixture.output, &workspace));

    // Missing arguments, including the workspace, which is easy to forget because the f32 operator
    // needs it for two buffers and this one only for the window.
    fixture.output.dtype = INFR_DTYPE_I8;
    EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT,
              infr_conv2d_i8(nullptr, &fixture.filter, nullptr, &fixture.params, &fixture.output, &workspace));
    EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT,
              infr_conv2d_i8(&fixture.input, &fixture.filter, nullptr, &fixture.params, &fixture.output, nullptr));
}
