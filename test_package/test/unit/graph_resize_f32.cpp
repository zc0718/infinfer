// Conan::ImportStart
#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include <infinfer/infinfer.h>
// Conan::ImportEnd

/*!
 * @ingroup group_runtime
 * @file
 * @brief [en] `infr_resize_f32`: the case that matters is the integer upscale real models emit.
 * @brief [zh] `infr_resize_f32`：要紧的是真实模型实际用的整数上采样。
 * @note [en] Two things are asserted beyond the values. The integer mapping is checked against
 *            hand-computed source indices, because a float-scaled mapping agrees on small extents
 *            and would pass a sloppier test. The cost is checked by counting `copy` calls: an
 *            upscale of two must cost fewer than one call per output, and the count is stated
 *            exactly so a regression that splits runs shows up as a number, not as "still fast".
 * @note [zh] 除了数值，还断言两件事。整数映射要对着手算的源下标检查，因为浮点比例映射在小
 *            尺寸上与它一致，写成松一点的测试就会通过。代价靠数 `copy` 调用：2 倍上采样必须
 *            低于每输出一次，而且把次数写死，这样把段拆碎的回退会显示成一个数字，
 *            而不是"看起来还快"。
 * @since 0.2.0
 */

namespace {

infr_tensor make_tensor(float* data, uint32_t rank, const uint32_t* dims) {
    infr_tensor t{};
    t.data = data;
    t.dtype = INFR_DTYPE_F32;
    t.quant.kind = INFR_QUANT_NONE;
    t.shape.rank = static_cast<uint8_t>(rank);
    t.shape.elem_count = 1U;
    for (uint32_t a = 0U; a < rank; ++a) {
        t.shape.dims[a] = dims[a];
        t.shape.elem_count *= dims[a];
    }
    return t;
}

/* Value at (h, w, c) of an NHWC fixture: unique per coordinate and readable by eye. */
float nhwc_value(uint32_t h, uint32_t w, uint32_t c) {
    return 100.0F * static_cast<float>(h) + 10.0F * static_cast<float>(w) + static_cast<float>(c);
}

std::vector<float> make_nhwc(uint32_t h, uint32_t w, uint32_t c) {
    std::vector<float> v(static_cast<size_t>(h) * w * c);
    size_t p = 0U;
    for (uint32_t y = 0U; y < h; ++y) {
        for (uint32_t x = 0U; x < w; ++x) {
            for (uint32_t ch = 0U; ch < c; ++ch) {
                v[p++] = nhwc_value(y, x, ch);
            }
        }
    }
    return v;
}

infr_resize_params nearest_params(uint32_t rank, const int64_t* out_dims) {
    infr_resize_params p{};
    p.rank = rank;
    p.mode = INFR_RESIZE_NEAREST;
    for (uint32_t a = 0U; a < rank; ++a) {
        p.out_dims[a] = out_dims[a];
    }
    return p;
}

/* ---- the cost assertion: fewer calls than output elements ---- */

std::uint32_t g_copy_calls = 0U;

infr_status counting_copy(const void* const* inputs, void* const* outputs, std::uint32_t element_count,
                          const infr_quant* quant) {
    ++g_copy_calls;
    const infr_kernel_desc* row = infr_kernel_find("copy", INFR_DTYPE_F32, INFR_QUANT_NONE, INFR_TIER_SCALAR);
    return row == nullptr ? INFR_ERR_NOT_IMPLEMENTED : row->fn(inputs, outputs, element_count, quant);
}

void install_counting_copy() {
    const infr_kernel_desc shim_row = [] {
        infr_kernel_desc d{};
        d.name = "copy";
        d.dtype = INFR_DTYPE_F32;
        d.quant = INFR_QUANT_NONE;
        d.tier = INFR_TIER_VENDOR;
        d.fn = counting_copy;
        return d;
    }();
    ASSERT_EQ(INFR_OK, infr_kernel_register(&shim_row));
    ASSERT_EQ(INFR_OK, infr_kernel_set_tier(INFR_TIER_VENDOR));
}

} // namespace

TEST(GraphResizeF32, UpscaleByTwoRepeatsEachSourcePixel) {
    ASSERT_EQ(INFR_OK, infr_kernel_init());
    const uint32_t in_dims[4] = {1U, 2U, 2U, 1U};
    std::vector<float> in = make_nhwc(2U, 2U, 1U);
    infr_tensor input = make_tensor(in.data(), 4U, in_dims);

    const int64_t out_dims[4] = {1, 4, 4, 1};
    const infr_resize_params params = nearest_params(4U, out_dims);

    infr_shape shape{};
    ASSERT_EQ(INFR_OK, infr_resize_output_shape(&input.shape, &params, &shape));
    ASSERT_EQ(4U, shape.rank);
    EXPECT_EQ(1U, shape.dims[0]);
    EXPECT_EQ(4U, shape.dims[1]);
    EXPECT_EQ(4U, shape.dims[2]);
    EXPECT_EQ(1U, shape.dims[3]);
    EXPECT_EQ(16U, shape.elem_count);

    std::vector<float> out(16U, -1.0F);
    const uint32_t out_d[4] = {1U, 4U, 4U, 1U};
    infr_tensor output = make_tensor(out.data(), 4U, out_d);
    ASSERT_EQ(INFR_OK, infr_resize_f32(&input, &params, &output));

    /* `src = (dst * in) / out` on both spatial axes: [0, 0, 1, 1]. */
    const uint32_t src[4] = {0U, 0U, 1U, 1U};
    size_t p = 0U;
    for (uint32_t y = 0U; y < 4U; ++y) {
        for (uint32_t x = 0U; x < 4U; ++x) {
            EXPECT_FLOAT_EQ(nhwc_value(src[y], src[x], 0U), out[p++]) << "y=" << y << " x=" << x;
        }
    }
}

TEST(GraphResizeF32, DownscalePicksTheTopLeftOfEachBlock) {
    ASSERT_EQ(INFR_OK, infr_kernel_init());
    const uint32_t in_dims[4] = {1U, 4U, 4U, 1U};
    std::vector<float> in = make_nhwc(4U, 4U, 1U);
    infr_tensor input = make_tensor(in.data(), 4U, in_dims);

    const int64_t out_dims[4] = {1, 2, 2, 1};
    const infr_resize_params params = nearest_params(4U, out_dims);
    std::vector<float> out(4U, -1.0F);
    const uint32_t out_d[4] = {1U, 2U, 2U, 1U};
    infr_tensor output = make_tensor(out.data(), 4U, out_d);
    ASSERT_EQ(INFR_OK, infr_resize_f32(&input, &params, &output));

    /* Sources are [0, 2] on both axes: the top-left neighbour, which is what floor division says
     * and what the exporters' asymmetric mapping means.
     *
     * 两条轴的源下标都是 [0, 2]：左上邻点，这就是向下取整的说法，也是导出器 asymmetric 映射
     * 的含义。 */
    EXPECT_FLOAT_EQ(nhwc_value(0U, 0U, 0U), out[0]);
    EXPECT_FLOAT_EQ(nhwc_value(0U, 2U, 0U), out[1]);
    EXPECT_FLOAT_EQ(nhwc_value(2U, 0U, 0U), out[2]);
    EXPECT_FLOAT_EQ(nhwc_value(2U, 2U, 0U), out[3]);
}

TEST(GraphResizeF32, IdentityIsBitwiseEqual) {
    ASSERT_EQ(INFR_OK, infr_kernel_init());
    const uint32_t dims[4] = {1U, 3U, 5U, 4U};
    std::vector<float> in = make_nhwc(3U, 5U, 4U);
    infr_tensor input = make_tensor(in.data(), 4U, dims);

    const int64_t out_dims[4] = {1, 3, 5, 4};
    const infr_resize_params params = nearest_params(4U, out_dims);
    std::vector<float> out(in.size(), -1.0F);
    infr_tensor output = make_tensor(out.data(), 4U, dims);
    ASSERT_EQ(INFR_OK, infr_resize_f32(&input, &params, &output));
    EXPECT_EQ(in, out);
}

TEST(GraphResizeF32, RankOneResizesAlongTheOnlyAxis) {
    ASSERT_EQ(INFR_OK, infr_kernel_init());
    const uint32_t in_dims[1] = {5U};
    std::vector<float> in = {0.0F, 1.0F, 2.0F, 3.0F, 4.0F};
    infr_tensor input = make_tensor(in.data(), 1U, in_dims);

    const int64_t out_dims[1] = {3};
    const infr_resize_params params = nearest_params(1U, out_dims);
    std::vector<float> out(3U, -1.0F);
    const uint32_t out_d[1] = {3U};
    infr_tensor output = make_tensor(out.data(), 1U, out_d);
    ASSERT_EQ(INFR_OK, infr_resize_f32(&input, &params, &output));

    /* `(dst * 5) / 3` = [0, 1, 3]. */
    EXPECT_FLOAT_EQ(0.0F, out[0]);
    EXPECT_FLOAT_EQ(1.0F, out[1]);
    EXPECT_FLOAT_EQ(3.0F, out[2]);
}

TEST(GraphResizeF32, RefusesWhatItCannotHonour) {
    ASSERT_EQ(INFR_OK, infr_kernel_init());
    const uint32_t dims[4] = {1U, 2U, 2U, 1U};
    std::vector<float> in(4U, 1.0F);
    infr_tensor input = make_tensor(in.data(), 4U, dims);
    std::vector<float> out(16U, 0.0F);
    const uint32_t out_d[4] = {1U, 4U, 4U, 1U};
    infr_tensor output = make_tensor(out.data(), 4U, out_d);
    infr_shape shape{};

    const int64_t good[4] = {1, 4, 4, 1};
    infr_resize_params params = nearest_params(4U, good);
    EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, infr_resize_output_shape(nullptr, &params, &shape));
    EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, infr_resize_output_shape(&input.shape, nullptr, &shape));
    EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, infr_resize_output_shape(&input.shape, &params, nullptr));

    infr_resize_params zero = nearest_params(4U, good);
    zero.out_dims[2] = 0;
    EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, infr_resize_output_shape(&input.shape, &zero, &shape));

    infr_resize_params wrong_rank = nearest_params(3U, good);
    EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, infr_resize_output_shape(&input.shape, &wrong_rank, &shape));

    infr_resize_params unknown_mode = nearest_params(4U, good);
    unknown_mode.mode = static_cast<infr_resize_mode>(7);
    EXPECT_EQ(INFR_ERR_UNSUPPORTED, infr_resize_output_shape(&input.shape, &unknown_mode, &shape));

    /* A shape that disagrees with the request is refused rather than resized into. */
    const uint32_t wrong_out[4] = {1U, 2U, 4U, 1U};
    infr_tensor mismatched = make_tensor(out.data(), 4U, wrong_out);
    EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, infr_resize_f32(&input, &params, &mismatched));

    infr_tensor non_f32 = make_tensor(in.data(), 4U, dims);
    non_f32.dtype = INFR_DTYPE_I8;
    EXPECT_EQ(INFR_ERR_UNSUPPORTED, infr_resize_f32(&non_f32, &params, &output));

    infr_tensor non_f32_out = make_tensor(out.data(), 4U, out_d);
    non_f32_out.dtype = INFR_DTYPE_I8;
    EXPECT_EQ(INFR_ERR_UNSUPPORTED, infr_resize_f32(&input, &params, &non_f32_out));

    /* A rank-3 output is refused on rank before any extent is compared. */
    const uint32_t out_rank3[3] = {1U, 4U, 4U};
    std::vector<float> out3(16U, 0.0F);
    infr_tensor wrong_rank_out = make_tensor(out3.data(), 3U, out_rank3);
    EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, infr_resize_f32(&input, &params, &wrong_rank_out));
}

TEST(GraphResizeF32, CostsFewerCallsThanOutputElements) {
    ASSERT_EQ(INFR_OK, infr_kernel_init());
    install_counting_copy();

    const uint32_t in_dims[4] = {1U, 2U, 2U, 1U};
    std::vector<float> in = make_nhwc(2U, 2U, 1U);
    infr_tensor input = make_tensor(in.data(), 4U, in_dims);
    const int64_t out_dims[4] = {1, 4, 4, 1};
    const infr_resize_params params = nearest_params(4U, out_dims);
    std::vector<float> out(16U, 0.0F);
    const uint32_t out_d[4] = {1U, 4U, 4U, 1U};
    infr_tensor output = make_tensor(out.data(), 4U, out_d);

    g_copy_calls = 0U;
    ASSERT_EQ(INFR_OK, infr_resize_f32(&input, &params, &output));
    EXPECT_EQ(11U, g_copy_calls)
        << "measured, not estimated: the flat source sequence of a 2x2 -> 4x4 upscale splits into "
           "runs of 1, 2, 1, 1, 2, 2, 2, 1, 1, 2, 1 because a repeated source stops a run";
    const double per_element = static_cast<double>(g_copy_calls) / static_cast<double>(out.size());
    EXPECT_LT(per_element, 1.0) << "an upscale must not cost one call per element";
    EXPECT_DOUBLE_EQ(0.6875, per_element);
}
