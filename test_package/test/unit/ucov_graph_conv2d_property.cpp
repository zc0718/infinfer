// Conan::ImportStart
#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include <infinfer/infinfer.h>

#include "parity_harness.hpp"
// Conan::ImportEnd

/*
 * Property test for the two composed window operators (`conv2d`, `max_pool2d`).
 * The hand-written cases pin the geometries somebody thought of; this one walks a
 * deterministic lattice of geometries nobody thought of, and judges each one three
 * ways that the hand-written cases cannot:
 *
 *   P1 the shape function's extent agrees with the formula derived here from scratch
 *      (`(in + 2*pad - kernel) / stride + 1`, refused when the padded input is
 *      narrower than the kernel) -- one derivation in the test, one in the library;
 *   P2 an accepted geometry computes every output element equal to a naive
 *      reference written here, so a drift in the gather/accumulate order or the
 *      filter layout shows up as a value mismatch rather than as a crash;
 *   P3 the operator and the shape function agree on what they refuse: whatever one
 *      rejects the other must reject too, so the two paths cannot diverge into
 *      "shape says yes, op says no";
 *   P4 the workspace is borrowed, not taken: after a successful call the cursor is
 *      back where it started (INV-8 -- an `EVAL` loop may not grow).
 *
 * There is no stride dimension to randomise: INV-3 keeps strides out of the kernel
 * and of the graph tensor, so "shape" is the whole geometry this API has.
 *
 * 两个组合窗口算子的 property test。手写用例钉的是有人想到的几何，这里走的是确定性
 * 格点上**没人特意想到**的几何，并从三个手写用例做不到的角度判：形状公式与独立推导
 * 一致、每个输出元素与独立写的朴素参照相等、算子与形状函数对"拒绝什么"意见一致、
 * 以及工作区是**借来的不是拿走的**（INV-8：EVAL 循环不许越长越胖）。
 * 本 API 没有 stride 可随机化：INV-3 把 stride 挡在内核与图张量之外。
 */

namespace {

// Deterministic LCG: the same geometries on every target, so a failure is
// reproducible by round number rather than by seed hunting.
std::uint32_t lcg(std::uint32_t* state) {
    *state = (*state * 1664525U) + 1013904223U;
    return *state >> 8;
}

std::uint32_t pick(std::uint32_t* state, std::uint32_t low, std::uint32_t high) {
    return low + (lcg(state) % ((high - low) + 1U));
}

float value_at(std::uint32_t seed) {
    const std::int32_t centred = static_cast<std::int32_t>((seed >> 8) % 2001U) - 1000;
    return static_cast<float>(centred) / 500.0F;
}

constexpr std::size_t kArenaBytes = 64U * 1024U;

// The reference extent, derived from the contract rather than from the library:
// every kernel position must fit inside the padded input.
bool expected_extent(std::uint32_t in, std::uint32_t kernel, std::uint32_t pad, std::uint32_t stride,
                     std::uint32_t* out) {
    if (stride == 0U || kernel == 0U) {
        return false;
    }
    const std::uint32_t padded = in + (2U * pad);
    if (padded < kernel) {
        return false;
    }
    *out = ((padded - kernel) / stride) + 1U;
    return true;
}

// Filter layout is [out_c, kh, kw, in_per_group]; output is [1, out_h, out_w, out_c].
void reference_conv2d(const std::vector<float>& input, const std::vector<float>& filter, std::uint32_t in_h,
                      std::uint32_t in_w, std::uint32_t in_c, std::uint32_t out_c, std::uint32_t kernel,
                      std::uint32_t stride, std::uint32_t pad, std::uint32_t groups, std::uint32_t out_h,
                      std::uint32_t out_w, std::vector<float>* want) {
    const std::uint32_t in_per_group = in_c / groups;
    const std::uint32_t out_per_group = out_c / groups;
    want->assign(static_cast<std::size_t>(out_h) * out_w * out_c, 0.0F);
    for (std::uint32_t oy = 0; oy < out_h; ++oy) {
        for (std::uint32_t ox = 0; ox < out_w; ++ox) {
            for (std::uint32_t oc = 0; oc < out_c; ++oc) {
                const std::uint32_t group = oc / out_per_group;
                float acc = 0.0F;
                for (std::uint32_t ky = 0; ky < kernel; ++ky) {
                    for (std::uint32_t kx = 0; kx < kernel; ++kx) {
                        const std::int32_t iy = static_cast<std::int32_t>(oy * stride + ky) - static_cast<std::int32_t>(pad);
                        const std::int32_t ix = static_cast<std::int32_t>(ox * stride + kx) - static_cast<std::int32_t>(pad);
                        if (iy < 0 || ix < 0 || iy >= static_cast<std::int32_t>(in_h) ||
                            ix >= static_cast<std::int32_t>(in_w)) {
                            continue;  // zero padding contributes nothing
                        }
                        for (std::uint32_t ic = 0; ic < in_per_group; ++ic) {
                            const std::size_t in_index =
                                ((static_cast<std::size_t>(iy) * in_w) + ix) * in_c + (group * in_per_group + ic);
                            const std::size_t w_index =
                                (((static_cast<std::size_t>(oc) * kernel) + ky) * kernel + kx) * in_per_group + ic;
                            acc += input[in_index] * filter[w_index];
                        }
                    }
                }
                (*want)[((static_cast<std::size_t>(oy) * out_w) + ox) * out_c + oc] = acc;
            }
        }
    }
}

void reference_max_pool(const std::vector<float>& input, std::uint32_t in_h, std::uint32_t in_w, std::uint32_t channels,
                        std::uint32_t kernel, std::uint32_t stride, std::uint32_t pad, std::uint32_t out_h,
                        std::uint32_t out_w, std::vector<float>* want) {
    const float padded = -std::numeric_limits<float>::infinity();
    want->assign(static_cast<std::size_t>(out_h) * out_w * channels, padded);
    for (std::uint32_t oy = 0; oy < out_h; ++oy) {
        for (std::uint32_t ox = 0; ox < out_w; ++ox) {
            for (std::uint32_t c = 0; c < channels; ++c) {
                float best = padded;
                for (std::uint32_t ky = 0; ky < kernel; ++ky) {
                    for (std::uint32_t kx = 0; kx < kernel; ++kx) {
                        const std::int32_t iy = static_cast<std::int32_t>(oy * stride + ky) - static_cast<std::int32_t>(pad);
                        const std::int32_t ix = static_cast<std::int32_t>(ox * stride + kx) - static_cast<std::int32_t>(pad);
                        if (iy < 0 || ix < 0 || iy >= static_cast<std::int32_t>(in_h) ||
                            ix >= static_cast<std::int32_t>(in_w)) {
                            continue;
                        }
                        const float v = input[((static_cast<std::size_t>(iy) * in_w) + ix) * channels + c];
                        best = v > best ? v : best;
                    }
                }
                (*want)[((static_cast<std::size_t>(oy) * out_w) + ox) * channels + c] = best;
            }
        }
    }
}

}  // namespace

TEST(GraphConv2dProperty, GeometryLatticeAgreesWithIndependentShapeAndReference) {
    std::uint32_t state = 20261126U;
    std::uint32_t accepted = 0U;
    std::uint32_t refused = 0U;

    for (std::uint32_t round = 0U; round < 400U; ++round) {
        const std::uint32_t in_h = pick(&state, 1U, 8U);
        const std::uint32_t in_w = pick(&state, 1U, 8U);
        const std::uint32_t in_c = pick(&state, 1U, 4U);
        const std::uint32_t out_c = (lcg(&state) % 4U == 0U) ? in_c : pick(&state, 1U, 4U);
        const std::uint32_t kernel = pick(&state, 1U, 3U);
        const std::uint32_t stride = pick(&state, 1U, 2U);
        const std::uint32_t pad = pick(&state, 0U, 1U);
        const std::uint32_t groups = (in_c == out_c && (lcg(&state) % 4U == 0U)) ? in_c : 1U;

        std::uint32_t want_h = 0U;
        std::uint32_t want_w = 0U;
        const bool geometry_legal = expected_extent(in_h, kernel, pad, stride, &want_h) &&
                                    expected_extent(in_w, kernel, pad, stride, &want_w);

        infr_conv2d_params params{};
        params.kernel_h = kernel;
        params.kernel_w = kernel;
        params.stride_h = stride;
        params.stride_w = stride;
        params.pad_h = pad;
        params.pad_w = pad;
        params.in_channels = in_c;
        params.out_channels = out_c;
        params.groups = groups;

        const std::uint32_t in_dims[4] = {1U, in_h, in_w, in_c};
        infr_shape in_shape{};
        ASSERT_EQ(INFR_OK, infr_shape_init(&in_shape, 4U, in_dims));
        infr_shape out_shape{};
        const infr_status shape_status = infr_conv2d_output_shape(&params, &in_shape, &out_shape);

        // P1: the library's extent agrees with the one derived here.
        if (!geometry_legal) {
            EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, shape_status) << "round " << round;
        } else {
            ASSERT_EQ(INFR_OK, shape_status) << "round " << round;
            EXPECT_EQ(want_h, out_shape.dims[1]) << "round " << round;
            EXPECT_EQ(want_w, out_shape.dims[2]) << "round " << round;
            EXPECT_EQ(out_c, out_shape.dims[3]) << "round " << round;
        }

        const std::size_t in_count = static_cast<std::size_t>(in_h) * in_w * in_c;
        const std::size_t w_count = static_cast<std::size_t>(out_c) * kernel * kernel * (in_c / groups);
        const std::size_t out_count = static_cast<std::size_t>(want_h) * want_w * out_c;
        std::vector<float> input_data(in_count);
        std::vector<float> filter_data(w_count);
        std::vector<float> output_data(out_count + 8U, 0.0F);  // slack: a tail write is caught below
        for (std::size_t i = 0; i < in_count; ++i) {
            input_data[i] = value_at(static_cast<std::uint32_t>(i) * 7U + round);
        }
        for (std::size_t i = 0; i < w_count; ++i) {
            filter_data[i] = value_at(static_cast<std::uint32_t>(i) * 13U + 3U + round);
        }

        infr_tensor input{};
        infr_tensor filter{};
        infr_tensor output{};
        ASSERT_EQ(INFR_OK, infr_tensor_init(&input, input_data.data(), INFR_DTYPE_F32, &in_shape));
        const std::uint32_t w_dims[4] = {out_c, kernel, kernel, in_c / groups};
        infr_shape w_shape{};
        ASSERT_EQ(INFR_OK, infr_shape_init(&w_shape, 4U, w_dims));
        ASSERT_EQ(INFR_OK, infr_tensor_init(&filter, filter_data.data(), INFR_DTYPE_F32, &w_shape));

        alignas(8) static std::uint8_t arena[kArenaBytes];
        infr_workspace workspace{};
        ASSERT_EQ(INFR_OK, infr_workspace_init(&workspace, arena, sizeof(arena)));
        const std::size_t mark = infr_workspace_mark(&workspace);

        if (!geometry_legal) {
            // P3: the operator refuses what the shape function refuses. The output
            // tensor is bound to the (stale) shape on purpose: a refusal must not
            // depend on the caller having given up already.
            infr_shape shaped{};
            const std::uint32_t ok_dims[4] = {1U, in_h, in_w, out_c};
            ASSERT_EQ(INFR_OK, infr_shape_init(&shaped, 4U, ok_dims));
            ASSERT_EQ(INFR_OK, infr_tensor_init(&output, output_data.data(), INFR_DTYPE_F32, &shaped));
            EXPECT_NE(INFR_OK, infr_conv2d_f32(&input, &filter, nullptr, &params, &output, &workspace))
                << "round " << round;
            ++refused;
            continue;
        }

        ASSERT_EQ(INFR_OK, infr_tensor_init(&output, output_data.data(), INFR_DTYPE_F32, &out_shape));
        ASSERT_EQ(INFR_OK, infr_conv2d_f32(&input, &filter, nullptr, &params, &output, &workspace))
            << "round " << round;

        // P2: every element equals the independent reference.
        std::vector<float> want;
        reference_conv2d(input_data, filter_data, in_h, in_w, in_c, out_c, kernel, stride, pad, groups, want_h, want_w,
                         &want);
        infr_test::expect_float_buffers_near(want.data(), output_data.data(), want.size(), "conv2d property");

        // P4: the scratch is returned, and nothing was written past the output.
        EXPECT_EQ(mark, infr_workspace_mark(&workspace)) << "round " << round;
        for (std::size_t i = out_count; i < output_data.size(); ++i) {
            EXPECT_EQ(0.0F, output_data[i]) << "a write past the output at offset " << (i - out_count);
        }
        ++accepted;
    }

    // A lattice that only produced one of the two verdicts would prove nothing
    // about the other branch.
    EXPECT_GT(accepted, 0U);
    EXPECT_GT(refused, 0U);
}

TEST(GraphMaxPool2dProperty, GeometryLatticeAgreesWithIndependentShapeAndReference) {
    std::uint32_t state = 7717U;
    std::uint32_t accepted = 0U;
    std::uint32_t refused = 0U;

    for (std::uint32_t round = 0U; round < 400U; ++round) {
        const std::uint32_t in_h = pick(&state, 1U, 8U);
        const std::uint32_t in_w = pick(&state, 1U, 8U);
        const std::uint32_t channels = pick(&state, 1U, 4U);
        const std::uint32_t kernel = pick(&state, 1U, 3U);
        const std::uint32_t stride = pick(&state, 1U, 2U);
        // Padding may not swallow a whole window: that is the documented refusal
        // this lattice has to reach sometimes.
        const std::uint32_t pad = pick(&state, 0U, kernel);

        std::uint32_t want_h = 0U;
        std::uint32_t want_w = 0U;
        const bool geometry_legal = pad < kernel && expected_extent(in_h, kernel, pad, stride, &want_h) &&
                                    expected_extent(in_w, kernel, pad, stride, &want_w);

        infr_pool_params params{};
        params.kernel_h = kernel;
        params.kernel_w = kernel;
        params.stride_h = stride;
        params.stride_w = stride;
        params.pad_h = pad;
        params.pad_w = pad;
        params.channels = channels;

        const std::uint32_t in_dims[4] = {1U, in_h, in_w, channels};
        infr_shape in_shape{};
        ASSERT_EQ(INFR_OK, infr_shape_init(&in_shape, 4U, in_dims));
        infr_shape out_shape{};
        const infr_status shape_status = infr_max_pool2d_output_shape(&params, &in_shape, &out_shape);

        if (!geometry_legal) {
            EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, shape_status) << "round " << round;
            ++refused;
            continue;
        }
        ASSERT_EQ(INFR_OK, shape_status) << "round " << round;
        EXPECT_EQ(want_h, out_shape.dims[1]) << "round " << round;
        EXPECT_EQ(want_w, out_shape.dims[2]) << "round " << round;

        const std::size_t in_count = static_cast<std::size_t>(in_h) * in_w * channels;
        std::vector<float> input_data(in_count);
        for (std::size_t i = 0; i < in_count; ++i) {
            input_data[i] = value_at(static_cast<std::uint32_t>(i) * 11U + round);
        }
        std::vector<float> output_data(static_cast<std::size_t>(want_h) * want_w * channels, 0.0F);

        infr_tensor input{};
        infr_tensor output{};
        ASSERT_EQ(INFR_OK, infr_tensor_init(&input, input_data.data(), INFR_DTYPE_F32, &in_shape));
        ASSERT_EQ(INFR_OK, infr_tensor_init(&output, output_data.data(), INFR_DTYPE_F32, &out_shape));

        alignas(8) static std::uint8_t arena[kArenaBytes];
        infr_workspace workspace{};
        ASSERT_EQ(INFR_OK, infr_workspace_init(&workspace, arena, sizeof(arena)));
        const std::size_t mark = infr_workspace_mark(&workspace);
        ASSERT_EQ(INFR_OK, infr_max_pool2d_f32(&input, &params, &output, &workspace)) << "round " << round;

        std::vector<float> want;
        reference_max_pool(input_data, in_h, in_w, channels, kernel, stride, pad, want_h, want_w, &want);
        // A maximum involves no arithmetic, so the comparison is exact.
        EXPECT_EQ(want, output_data) << "round " << round;
        EXPECT_EQ(mark, infr_workspace_mark(&workspace)) << "round " << round;
        ++accepted;
    }

    EXPECT_GT(accepted, 0U);
    EXPECT_GT(refused, 0U);
}


int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
