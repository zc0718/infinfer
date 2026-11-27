// Conan::ImportStart
#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <vector>

#include <infinfer/infinfer.h>
// Conan::ImportEnd

/*!
 * @ingroup group_runtime
 * @file
 * @brief [en] `infr_slice_f32`: a strided view, so the tests are about the two paths and the clamps.
 * @brief [zh] `infr_slice_f32`：一个带步长的视图，所以测试关心的是两条路径与夹取。
 * @note [en] The case that matters commercially is the one the real models emit: cut the channel
 *            axis of an NHWC tensor for a channel shuffle, which leaves the innermost axis full
 *            and contiguous. The case that breaks the fast path is a stride on the innermost
 *            axis. Both are here, plus the clamps -- an `INT64_MAX` end and a start past the
 *            axis are how exported graphs ask for "to the end" and "nothing", and a zero-length
 *            result has to be a legal empty tensor rather than an error.
 * @note [zh] 商业上最要紧的是真实模型实际生成的那种：切 NHWC 的通道轴做 channel shuffle，
 *            末轴完整且连续。破坏快路径的是末轴带步长。两种都在这里，另外还有夹取——
 *            `INT64_MAX` 终点和超过轴长的起点，是导出图表达"到末尾"和"什么都不要"的方式，
 *            而长度为 0 的结果必须是合法的空张量而不是错误。
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

TEST(GraphSliceF32, ChannelCutKeepsThePixelOrder) {
    ASSERT_EQ(INFR_OK, infr_kernel_init());
    /* The channel-shuffle case: 1x2x2x4 cut to channels [1, 3) leaves the innermost axis
     * contiguous, so this runs on the run path.
     *
     * channel shuffle 那种情形：1x2x2x4 切出通道 [1, 3)，末轴仍连续，走段路径。 */
    const uint32_t in_dims[4] = {1U, 2U, 2U, 4U};
    const uint32_t out_dims[4] = {1U, 2U, 2U, 2U};
    std::vector<float> src = make_nhwc(2U, 2U, 4U);
    infr_tensor in = make_tensor(src.data(), 4U, in_dims);
    std::vector<float> dst(8U, -1.0F);
    infr_tensor out = make_tensor(dst.data(), 4U, out_dims);

    infr_slice_params params{};
    params.axis_count = 1U;
    params.axis[0] = 3U;
    params.start[0] = 1;
    params.end[0] = 3;
    params.step[0] = 1;

    alignas(8) uint8_t bytes[256] = {};
    infr_workspace ws{};
    ASSERT_EQ(INFR_OK, infr_workspace_init(&ws, bytes, sizeof(bytes)));
    ASSERT_EQ(INFR_OK, infr_slice_f32(&in, &params, &out, &ws));
    size_t p = 0U;
    for (uint32_t y = 0U; y < 2U; ++y) {
        for (uint32_t x = 0U; x < 2U; ++x) {
            for (uint32_t ch = 1U; ch < 3U; ++ch) {
                EXPECT_EQ(nhwc_value(y, x, ch), dst[p++]) << "(" << y << "," << x << "," << ch << ")";
            }
        }
    }
}

TEST(GraphSliceF32, Int64MaxMeansToTheEnd) {
    ASSERT_EQ(INFR_OK, infr_kernel_init());
    const uint32_t in_dims[4] = {1U, 1U, 1U, 6U};
    const uint32_t out_dims[4] = {1U, 1U, 1U, 4U};
    float src[6] = {0.0F, 1.0F, 2.0F, 3.0F, 4.0F, 5.0F};
    infr_tensor in = make_tensor(src, 4U, in_dims);
    float dst[4] = {-1.0F, -1.0F, -1.0F, -1.0F};
    infr_tensor out = make_tensor(dst, 4U, out_dims);

    infr_slice_params params{};
    params.axis_count = 1U;
    params.axis[0] = 3U;
    params.start[0] = 2;
    params.end[0] = std::numeric_limits<int64_t>::max();
    params.step[0] = 1;

    alignas(8) uint8_t bytes[64] = {};
    infr_workspace ws{};
    ASSERT_EQ(INFR_OK, infr_workspace_init(&ws, bytes, sizeof(bytes)));
    ASSERT_EQ(INFR_OK, infr_slice_f32(&in, &params, &out, &ws));
    EXPECT_EQ(2.0F, dst[0]);
    EXPECT_EQ(5.0F, dst[3]);
}

TEST(GraphSliceF32, NegativeIndicesCountFromTheEnd) {
    ASSERT_EQ(INFR_OK, infr_kernel_init());
    const uint32_t dims[2] = {1U, 5U};
    const uint32_t out_dims[2] = {1U, 2U};
    float src[5] = {10.0F, 11.0F, 12.0F, 13.0F, 14.0F};
    infr_tensor in = make_tensor(src, 2U, dims);
    float dst[2] = {-1.0F, -1.0F};
    infr_tensor out = make_tensor(dst, 2U, out_dims);

    infr_slice_params params{};
    params.axis_count = 1U;
    params.axis[0] = 1U;
    params.start[0] = -2; /* the last two elements */
    params.end[0] = std::numeric_limits<int64_t>::max();
    params.step[0] = 1;

    alignas(8) uint8_t bytes[64] = {};
    infr_workspace ws{};
    ASSERT_EQ(INFR_OK, infr_workspace_init(&ws, bytes, sizeof(bytes)));
    ASSERT_EQ(INFR_OK, infr_slice_f32(&in, &params, &out, &ws));
    EXPECT_EQ(13.0F, dst[0]);
    EXPECT_EQ(14.0F, dst[1]);
}

TEST(GraphSliceF32, StridedInnerAxisTakesTheGatherPath) {
    ASSERT_EQ(INFR_OK, infr_kernel_init());
    /* Step 2 on the innermost axis: nothing is contiguous in output order.
     *
     * 末轴步长 2：输出顺序里没有连续段。 */
    const uint32_t dims[4] = {1U, 1U, 1U, 6U};
    const uint32_t out_dims[4] = {1U, 1U, 1U, 3U};
    float src[6] = {0.0F, 1.0F, 2.0F, 3.0F, 4.0F, 5.0F};
    infr_tensor in = make_tensor(src, 4U, dims);
    float dst[3] = {-1.0F, -1.0F, -1.0F};
    infr_tensor out = make_tensor(dst, 4U, out_dims);

    infr_slice_params params{};
    params.axis_count = 1U;
    params.axis[0] = 3U;
    params.start[0] = 1;
    params.end[0] = std::numeric_limits<int64_t>::max();
    params.step[0] = 2;

    alignas(8) uint8_t bytes[64] = {};
    infr_workspace ws{};
    ASSERT_EQ(INFR_OK, infr_workspace_init(&ws, bytes, sizeof(bytes)));
    ASSERT_EQ(INFR_OK, infr_slice_f32(&in, &params, &out, &ws));
    EXPECT_EQ(1.0F, dst[0]);
    EXPECT_EQ(3.0F, dst[1]);
    EXPECT_EQ(5.0F, dst[2]);
}

TEST(GraphSliceF32, StriderOnAnOuterAxisStaysOnTheRunPath) {
    ASSERT_EQ(INFR_OK, infr_kernel_init());
    /* Step 2 on the height axis of 1x4x2x2, keeping both inner axes whole: every output run is
     * two pixels' worth of contiguous data.
     *
     * 在 1x4x2x2 的高度轴上步长 2、两个内层轴整条保留：每条输出段是两个像素的连续数据。 */
    const uint32_t dims[4] = {1U, 4U, 2U, 2U};
    const uint32_t out_dims[4] = {1U, 2U, 2U, 2U};
    std::vector<float> src = make_nhwc(4U, 2U, 2U);
    infr_tensor in = make_tensor(src.data(), 4U, dims);
    std::vector<float> dst(8U, -1.0F);
    infr_tensor out = make_tensor(dst.data(), 4U, out_dims);

    infr_slice_params params{};
    params.axis_count = 1U;
    params.axis[0] = 1U;
    params.start[0] = 0;
    params.end[0] = std::numeric_limits<int64_t>::max();
    params.step[0] = 2;

    alignas(8) uint8_t bytes[64] = {};
    infr_workspace ws{};
    ASSERT_EQ(INFR_OK, infr_workspace_init(&ws, bytes, sizeof(bytes)));
    ASSERT_EQ(INFR_OK, infr_slice_f32(&in, &params, &out, &ws));
    size_t p = 0U;
    for (uint32_t y = 0U; y < 4U; y += 2U) {
        for (uint32_t x = 0U; x < 2U; ++x) {
            for (uint32_t ch = 0U; ch < 2U; ++ch) {
                EXPECT_EQ(nhwc_value(y, x, ch), dst[p++]) << "(" << y << "," << x << "," << ch << ")";
            }
        }
    }
}

TEST(GraphSliceF32, EmptyCutIsALegalEmptyTensor) {
    ASSERT_EQ(INFR_OK, infr_kernel_init());
    const uint32_t dims[2] = {1U, 4U};
    const uint32_t out_dims[2] = {1U, 0U};
    float src[4] = {1.0F, 2.0F, 3.0F, 4.0F};
    infr_tensor in = make_tensor(src, 2U, dims);
    float dst[1] = {-1.0F};
    infr_tensor out = make_tensor(dst, 2U, out_dims);

    infr_slice_params params{};
    params.axis_count = 1U;
    params.axis[0] = 1U;
    params.start[0] = 2;
    params.end[0] = 2; /* start == end: nothing */
    params.step[0] = 1;

    infr_shape shape{};
    ASSERT_EQ(INFR_OK, infr_slice_output_shape(&in.shape, &params, &shape));
    EXPECT_EQ(0U, shape.dims[1]);

    alignas(8) uint8_t bytes[64] = {};
    infr_workspace ws{};
    ASSERT_EQ(INFR_OK, infr_workspace_init(&ws, bytes, sizeof(bytes)));
    EXPECT_EQ(INFR_OK, infr_slice_f32(&in, &params, &out, &ws));
    EXPECT_EQ(-1.0F, dst[0]) << "an empty cut must not touch the output buffer";
}

TEST(GraphSliceF32, RefusesWhatItCannotHonour) {
    ASSERT_EQ(INFR_OK, infr_kernel_init());
    const uint32_t dims[2] = {2U, 4U};
    float src[8] = {0.0F};
    infr_tensor in = make_tensor(src, 2U, dims);
    float dst[8] = {0.0F};
    infr_tensor out = make_tensor(dst, 2U, dims);
    alignas(8) uint8_t bytes[64] = {};
    infr_workspace ws{};
    ASSERT_EQ(INFR_OK, infr_workspace_init(&ws, bytes, sizeof(bytes)));

    infr_slice_params params{};
    params.axis_count = 1U;
    params.axis[0] = 2U; /* out of range for rank 2 */
    params.end[0] = std::numeric_limits<int64_t>::max();
    params.step[0] = 1;
    EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, infr_slice_f32(&in, &params, &out, &ws));

    params.axis[0] = 1U;
    params.axis_count = 2U; /* the same axis twice */
    params.axis[1] = 1U;
    EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, infr_slice_f32(&in, &params, &out, &ws));

    params.axis_count = 1U;
    params.step[0] = 0; /* a step of 0 has no meaning */
    EXPECT_EQ(INFR_ERR_UNSUPPORTED, infr_slice_f32(&in, &params, &out, &ws));

    params.step[0] = -1; /* reversal is another operator's job */
    EXPECT_EQ(INFR_ERR_UNSUPPORTED, infr_slice_f32(&in, &params, &out, &ws));

    params.step[0] = 1;
    params.start[0] = 0;
    params.end[0] = 2;
    /* Output that is not the shape this cut produces. */
    {
        const uint32_t wrong[2] = {2U, 4U};
        float other[8] = {0.0F};
        infr_tensor bad_out = make_tensor(other, 2U, wrong);
        EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, infr_slice_f32(&in, &params, &bad_out, &ws));
    }
    EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, infr_slice_f32(nullptr, &params, &out, &ws));
    EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, infr_slice_f32(&in, nullptr, &out, &ws));
    EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, infr_slice_f32(&in, &params, nullptr, &ws));
    EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, infr_slice_f32(&in, &params, &out, nullptr));
}

TEST(GraphSliceF32, ClampsOutOfRangeIndicesInsteadOfRefusing) {
    ASSERT_EQ(INFR_OK, infr_kernel_init());
    const uint32_t dims[2] = {1U, 4U};
    float src[4] = {1.0F, 2.0F, 3.0F, 4.0F};
    infr_tensor in = make_tensor(src, 2U, dims);
    alignas(8) uint8_t bytes[64] = {};
    infr_workspace ws{};
    ASSERT_EQ(INFR_OK, infr_workspace_init(&ws, bytes, sizeof(bytes)));

    /* A start past the axis clamps to the axis, so the cut is empty rather than an error: this
     * is how an exported graph says nothing is wanted, and refusing it would push the clamp
     * into every importer.
     *
     * 起点超过轴长时夹到轴长，于是切片为空而不是错误：导出图用它表达"什么都不要"，拒绝它
     * 等于把夹取推给每一个导入器。 */
    {
        const uint32_t empty_dims[2] = {1U, 0U};
        float dst[1] = {-1.0F};
        infr_tensor out = make_tensor(dst, 2U, empty_dims);
        infr_slice_params params{};
        params.axis_count = 1U;
        params.axis[0] = 1U;
        params.start[0] = 99;
        params.end[0] = std::numeric_limits<int64_t>::max();
        params.step[0] = 1;
        EXPECT_EQ(INFR_OK, infr_slice_f32(&in, &params, &out, &ws));
        EXPECT_EQ(-1.0F, dst[0]);
    }
    /* A start below zero clamps to zero; a negative end counts from the end and then clamps.
     * Both clamp to 0 here, so the output axis is 0 -- the first version of this fixture
     * declared a 1-wide output and the operator refused it, which was correct: the shape must
     * be the shape the cut produces.
     *
     * 起点为负夹到 0；负终点先从末尾数再夹取。这里两者都夹到 0，所以输出轴长为 0——这个
     * fixture 的第一版声明了宽度 1 的输出而被算子拒绝，那是对的：形状必须就是切出来的形状。 */
    {
        const uint32_t out_dims[2] = {1U, 0U};
        float dst[1] = {-1.0F};
        infr_tensor out = make_tensor(dst, 2U, out_dims);
        infr_slice_params params{};
        params.axis_count = 1U;
        params.axis[0] = 1U;
        params.start[0] = -99;
        params.end[0] = -99;
        params.step[0] = 1;
        EXPECT_EQ(INFR_OK, infr_slice_f32(&in, &params, &out, &ws));
        EXPECT_EQ(-1.0F, dst[0]) << "both clamp to 0, so the cut is empty";
    }
}

TEST(GraphSliceF32, RankOneAndNonFp32AndTightWorkspace) {
    ASSERT_EQ(INFR_OK, infr_kernel_init());
    alignas(8) uint8_t bytes[64] = {};
    infr_workspace ws{};
    ASSERT_EQ(INFR_OK, infr_workspace_init(&ws, bytes, sizeof(bytes)));

    /* Rank 1 takes the single-copy path. */
    {
        const uint32_t dims[1] = {5U};
        const uint32_t out_dims[1] = {3U};
        float src[5] = {10.0F, 11.0F, 12.0F, 13.0F, 14.0F};
        infr_tensor in = make_tensor(src, 1U, dims);
        float dst[3] = {-1.0F, -1.0F, -1.0F};
        infr_tensor out = make_tensor(dst, 1U, out_dims);
        infr_slice_params params{};
        params.axis_count = 1U;
        params.axis[0] = 0U;
        params.start[0] = 1;
        params.end[0] = 4;
        params.step[0] = 1;
        ASSERT_EQ(INFR_OK, infr_slice_f32(&in, &params, &out, &ws));
        EXPECT_EQ(11.0F, dst[0]);
        EXPECT_EQ(13.0F, dst[2]);
    }
    /* A non-fp32 tensor on either side is out of the promised subset. */
    {
        const uint32_t dims[1] = {4U};
        float src[4] = {0.0F};
        infr_tensor in = make_tensor(src, 1U, dims);
        float dst[4] = {0.0F};
        infr_tensor out = make_tensor(dst, 1U, dims);
        infr_slice_params params{};
        params.axis_count = 1U;
        params.axis[0] = 0U;
        params.start[0] = 0;
        params.end[0] = 4;
        params.step[0] = 1;
        infr_tensor in_i8 = in;
        in_i8.dtype = INFR_DTYPE_I8;
        EXPECT_EQ(INFR_ERR_UNSUPPORTED, infr_slice_f32(&in_i8, &params, &out, &ws));
        infr_tensor out_i8 = out;
        out_i8.dtype = INFR_DTYPE_I8;
        EXPECT_EQ(INFR_ERR_UNSUPPORTED, infr_slice_f32(&in, &params, &out_i8, &ws));
    }
    /* The gather path needs `elem_count` floats; a workspace that cannot hold them must say so
     * rather than fall back to a second copy path.
     *
     * gather 路径需要 `elem_count` 个 float；放不下时必须如实返回，而不是退化成第二条
     * 拷贝路径。 */
    {
        const uint32_t dims[2] = {1U, 8U};
        const uint32_t out_dims[2] = {1U, 4U};
        float src[8] = {0.0F, 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F};
        infr_tensor in = make_tensor(src, 2U, dims);
        float dst[4] = {-1.0F, -1.0F, -1.0F, -1.0F};
        infr_tensor out = make_tensor(dst, 2U, out_dims);
        infr_slice_params params{};
        params.axis_count = 1U;
        params.axis[0] = 1U;
        params.start[0] = 0;
        params.end[0] = 8;
        params.step[0] = 2; /* strided innermost axis -> gather */
        alignas(8) uint8_t tiny[4] = {};
        infr_workspace small_ws{};
        ASSERT_EQ(INFR_OK, infr_workspace_init(&small_ws, tiny, sizeof(tiny)));
        EXPECT_EQ(INFR_ERR_BUFFER_TOO_SMALL, infr_slice_f32(&in, &params, &out, &small_ws));
    }
}

/* ---- the cost assertion ---- */

std::uint32_t g_copy_calls = 0U;

infr_status counting_copy(const void* const* inputs, void* const* outputs, std::uint32_t element_count,
                          const infr_quant* quant) {
    ++g_copy_calls;
    const infr_kernel_desc* row = infr_kernel_find("copy", INFR_DTYPE_F32, INFR_QUANT_NONE, INFR_TIER_SCALAR);
    return row == nullptr ? INFR_ERR_NOT_IMPLEMENTED : row->fn(inputs, outputs, element_count, quant);
}

TEST(GraphSliceF32, CostsAtMostOneCallPerOutputElement) {
    ASSERT_EQ(INFR_OK, infr_kernel_init());
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

    /* 1x4x4x8 cut to 8 channels: 16 outer positions, one copy each, for 128 outputs. */
    const uint32_t dims[4] = {1U, 4U, 4U, 16U};
    const uint32_t half[4] = {1U, 4U, 4U, 8U};
    std::vector<float> src(static_cast<size_t>(4U) * 4U * 16U, 1.0F);
    infr_tensor in = make_tensor(src.data(), 4U, dims);
    std::vector<float> dst(128U, 0.0F);
    infr_tensor out = make_tensor(dst.data(), 4U, half);

    infr_slice_params params{};
    params.axis_count = 1U;
    params.axis[0] = 3U;
    params.start[0] = 0;
    params.end[0] = 8;
    params.step[0] = 1;

    alignas(8) uint8_t bytes[1024] = {};
    infr_workspace ws{};
    ASSERT_EQ(INFR_OK, infr_workspace_init(&ws, bytes, sizeof(bytes)));

    g_copy_calls = 0U;
    ASSERT_EQ(INFR_OK, infr_slice_f32(&in, &params, &out, &ws));
    EXPECT_EQ(16U, g_copy_calls) << "one copy per pixel, not per element";
    const double per_element = static_cast<double>(g_copy_calls) / static_cast<double>(dst.size());
    EXPECT_LE(per_element, 1.0);

    /* Strided innermost axis: one call for the whole gather. */
    {
        const uint32_t strided[4] = {1U, 4U, 4U, 8U};
        std::vector<float> dst2(128U, 0.0F);
        infr_tensor out2 = make_tensor(dst2.data(), 4U, strided);
        infr_slice_params p2{};
        p2.axis_count = 1U;
        p2.axis[0] = 3U;
        p2.start[0] = 0;
        p2.end[0] = 16;
        p2.step[0] = 2;
        g_copy_calls = 0U;
        ASSERT_EQ(INFR_OK, infr_slice_f32(&in, &p2, &out2, &ws));
        EXPECT_EQ(1U, g_copy_calls) << "the gather path issues one copy";
        const double per_element = static_cast<double>(g_copy_calls) / static_cast<double>(dst2.size());
        EXPECT_LE(per_element, 1.0);
    }

    ASSERT_EQ(INFR_OK, infr_kernel_set_tier(INFR_TIER_SCALAR));
}

} // namespace
