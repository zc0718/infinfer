// Conan::ImportStart
#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include <infinfer/infinfer.h>
// Conan::ImportEnd

/*!
 * @ingroup group_runtime
 * @file
 * @brief [en] `infr_transpose_f32`: three paths, one contract -- where each value ends up.
 * @brief [zh] `infr_transpose_f32`：三条路径、一个契约——每个值落在哪里。
 * @note [en] The operator picks one of three loops depending on whether a contiguous run
 *            exists, so the tests are aimed at the *seams* between them: identity, last axis
 *            in place, and the full gather (the NHWC to NCHW case every PyTorch export
 *            produces). Each fixture uses values that encode their own coordinates, so a value
 *            written at the wrong place shows as a wrong coordinate rather than a wrong number.
 * @note [zh] 这个算子按"是否存在连续段"在三条循环里选一条，所以测试瞄的是它们之间的**接缝**：
 *            恒等、末轴不动、完全 gather（每个 PyTorch 导出都会产生的 NHWC 转 NCHW）。每个
 *            fixture 的值都编码了自己的坐标，于是写错位置会表现为坐标错，而不是数字错。
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

/* Value at (n, h, w, c) of a 4-D fixture: unique per coordinate and readable by eye. */
float nhwc_value(uint32_t h, uint32_t w, uint32_t c) {
    return 100.0F * static_cast<float>(h) + 10.0F * static_cast<float>(w) + static_cast<float>(c);
}

std::vector<float> make_nhwc(uint32_t n, uint32_t h, uint32_t w, uint32_t c) {
    std::vector<float> v(static_cast<size_t>(n) * h * w * c);
    size_t p = 0U;
    for (uint32_t i = 0; i < n; ++i) {
        (void)i;
        for (uint32_t y = 0; y < h; ++y) {
            for (uint32_t x = 0; x < w; ++x) {
                for (uint32_t ch = 0; ch < c; ++ch) {
                    v[p++] = nhwc_value(y, x, ch);
                }
            }
        }
    }
    return v;
}

TEST(GraphTransposeF32, IdentityIsTheIdentity) {
    ASSERT_EQ(INFR_OK, infr_kernel_init());
    const uint32_t dims[4] = {1U, 2U, 3U, 2U};
    std::vector<float> src = make_nhwc(1U, 2U, 3U, 2U);
    infr_tensor in = make_tensor(src.data(), 4U, dims);
    std::vector<float> dst(src.size(), -1.0F);
    infr_tensor out = make_tensor(dst.data(), 4U, dims);

    infr_transpose_params params{};
    params.rank = 4U;
    for (uint32_t a = 0U; a < 4U; ++a) {
        params.perm[a] = a;
    }

    alignas(8) uint8_t bytes[256] = {};
    infr_workspace ws{};
    ASSERT_EQ(INFR_OK, infr_workspace_init(&ws, bytes, sizeof(bytes)));
    ASSERT_EQ(INFR_OK, infr_transpose_f32(&in, &params, &out, &ws));
    for (size_t p = 0U; p < src.size(); ++p) {
        EXPECT_EQ(src[p], dst[p]) << "element " << p;
    }
}

TEST(GraphTransposeF32, SwappingTheTwoInnerAxesIsExact) {
    ASSERT_EQ(INFR_OK, infr_kernel_init());
    /* 2x3 -> 3x2 with values that are the coordinate itself. */
    const uint32_t in_dims[2] = {2U, 3U};
    const uint32_t out_dims[2] = {3U, 2U};
    float src[6];
    for (uint32_t r = 0U; r < 2U; ++r) {
        for (uint32_t c = 0U; c < 3U; ++c) {
            src[(r * 3U) + c] = static_cast<float>((r * 10U) + c);
        }
    }
    infr_tensor in = make_tensor(src, 2U, in_dims);
    float dst[6] = {0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F};
    infr_tensor out = make_tensor(dst, 2U, out_dims);

    infr_transpose_params params{};
    params.rank = 2U;
    params.perm[0] = 1U;
    params.perm[1] = 0U;

    alignas(8) uint8_t bytes[256] = {};
    infr_workspace ws{};
    ASSERT_EQ(INFR_OK, infr_workspace_init(&ws, bytes, sizeof(bytes)));
    ASSERT_EQ(INFR_OK, infr_transpose_f32(&in, &params, &out, &ws));
    for (uint32_t r = 0U; r < 2U; ++r) {
        for (uint32_t c = 0U; c < 3U; ++c) {
            EXPECT_EQ(src[(r * 3U) + c], dst[(c * 2U) + r]) << "(" << r << "," << c << ")";
        }
    }
}

TEST(GraphTransposeF32, NhwcToNchwTakesTheGatherPath) {
    ASSERT_EQ(INFR_OK, infr_kernel_init());
    const uint32_t h = 2U;
    const uint32_t w = 3U;
    const uint32_t c = 2U;
    const uint32_t in_dims[4] = {1U, h, w, c};
    const uint32_t out_dims[4] = {1U, c, h, w};
    std::vector<float> src = make_nhwc(1U, h, w, c);
    infr_tensor in = make_tensor(src.data(), 4U, in_dims);

    std::vector<float> dst(static_cast<size_t>(c) * h * w, -1.0F);
    infr_tensor out = make_tensor(dst.data(), 4U, out_dims);

    infr_transpose_params params{};
    params.rank = 4U;
    params.perm[0] = 0U;
    params.perm[1] = 3U;
    params.perm[2] = 1U;
    params.perm[3] = 2U;

    /* The gather path needs `count` floats of workspace. */
    alignas(8) uint8_t bytes[64] = {};
    infr_workspace ws{};
    ASSERT_EQ(INFR_OK, infr_workspace_init(&ws, bytes, sizeof(bytes)));
    ASSERT_EQ(INFR_OK, infr_transpose_f32(&in, &params, &out, &ws));

    for (uint32_t y = 0U; y < h; ++y) {
        for (uint32_t x = 0U; x < w; ++x) {
            for (uint32_t ch = 0U; ch < c; ++ch) {
                const size_t o = ((static_cast<size_t>(ch) * h) + y) * w + x;
                EXPECT_EQ(nhwc_value(y, x, ch), dst[o]) << "NCHW(" << ch << "," << y << "," << x << ")";
            }
        }
    }
}

TEST(GraphTransposeF32, KeepsTheLeadingAxesAndTheRunPath) {
    ASSERT_EQ(INFR_OK, infr_kernel_init());
    /* perm keeps the last axis in place, so runs are contiguous and the operator takes the
     * middle path instead of the gather.
     *
     * 该 perm 保持末轴不动，所以各段连续，算子走中间那条路径而不是 gather。 */
    const uint32_t in_dims[4] = {1U, 2U, 3U, 2U};
    const uint32_t out_dims[4] = {1U, 3U, 2U, 2U};
    std::vector<float> src = make_nhwc(1U, 2U, 3U, 2U);
    infr_tensor in = make_tensor(src.data(), 4U, in_dims);
    std::vector<float> dst(12U, -1.0F);
    infr_tensor out = make_tensor(dst.data(), 4U, out_dims);

    infr_transpose_params params{};
    params.rank = 4U;
    params.perm[0] = 0U;
    params.perm[1] = 2U;
    params.perm[2] = 1U;
    params.perm[3] = 3U;

    alignas(8) uint8_t bytes[64] = {};
    infr_workspace ws{};
    ASSERT_EQ(INFR_OK, infr_workspace_init(&ws, bytes, sizeof(bytes)));
    ASSERT_EQ(INFR_OK, infr_transpose_f32(&in, &params, &out, &ws));
    for (uint32_t y = 0U; y < 2U; ++y) {
        for (uint32_t x = 0U; x < 3U; ++x) {
            for (uint32_t ch = 0U; ch < 2U; ++ch) {
                const size_t o = ((static_cast<size_t>(x) * 2U) + y) * 2U + ch;
                EXPECT_EQ(nhwc_value(y, x, ch), dst[o]) << "HW->WH(" << y << "," << x << "," << ch << ")";
            }
        }
    }
}

TEST(GraphTransposeF32, RankZeroReversesTheAxes) {
    ASSERT_EQ(INFR_OK, infr_kernel_init());
    const uint32_t in_dims[2] = {2U, 3U};
    const uint32_t out_dims[2] = {3U, 2U};
    float src[6];
    for (uint32_t i = 0U; i < 6U; ++i) {
        src[i] = static_cast<float>(i);
    }
    infr_tensor in = make_tensor(src, 2U, in_dims);
    float dst[6] = {0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F};
    infr_tensor out = make_tensor(dst, 2U, out_dims);

    infr_transpose_params params{};
    params.rank = 0U; /* no perm given: reverse */

    infr_shape shape{};
    ASSERT_EQ(INFR_OK, infr_transpose_output_shape(&in.shape, &params, &shape));
    EXPECT_EQ(2U, shape.rank);
    EXPECT_EQ(3U, shape.dims[0]);
    EXPECT_EQ(2U, shape.dims[1]);

    alignas(8) uint8_t bytes[256] = {};
    infr_workspace ws{};
    ASSERT_EQ(INFR_OK, infr_workspace_init(&ws, bytes, sizeof(bytes)));
    ASSERT_EQ(INFR_OK, infr_transpose_f32(&in, &params, &out, &ws));
    EXPECT_EQ(0.0F, dst[0]);
    EXPECT_EQ(3.0F, dst[1]);
    EXPECT_EQ(1.0F, dst[2]);
    EXPECT_EQ(4.0F, dst[3]);
    EXPECT_EQ(2.0F, dst[4]);
    EXPECT_EQ(5.0F, dst[5]);
}

TEST(GraphTransposeF32, RefusesAnythingThatIsNotAPermutation) {
    ASSERT_EQ(INFR_OK, infr_kernel_init());
    const uint32_t dims[2] = {2U, 3U};
    float src[6] = {0.0F, 1.0F, 2.0F, 3.0F, 4.0F, 5.0F};
    infr_tensor in = make_tensor(src, 2U, dims);
    float dst[6] = {0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F};
    infr_tensor out = make_tensor(dst, 2U, dims);

    alignas(8) uint8_t bytes[256] = {};
    infr_workspace ws{};
    ASSERT_EQ(INFR_OK, infr_workspace_init(&ws, bytes, sizeof(bytes)));

    infr_transpose_params params{};
    params.rank = 2U;
    params.perm[0] = 0U;
    params.perm[1] = 0U; /* an axis used twice */
    EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, infr_transpose_f32(&in, &params, &out, &ws));

    params.perm[1] = 2U; /* out of range for rank 2 */
    EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, infr_transpose_f32(&in, &params, &out, &ws));

    params.perm[1] = 1U; /* legal */
    params.rank = 3U;    /* but a rank that disagrees with the tensor */
    EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, infr_transpose_f32(&in, &params, &out, &ws));

    params.rank = 2U;
    /* Output that is not the shape this permutation produces. */
    {
        const uint32_t wrong[2] = {3U, 3U};
        float other[9] = {0.0F};
        infr_tensor bad_out = make_tensor(other, 2U, wrong);
        EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, infr_transpose_f32(&in, &params, &bad_out, &ws));
    }
    /* NULL arguments. */
    EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, infr_transpose_f32(nullptr, &params, &out, &ws));
    EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, infr_transpose_f32(&in, nullptr, &out, &ws));
    EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, infr_transpose_f32(&in, &params, nullptr, &ws));
    EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, infr_transpose_f32(&in, &params, &out, nullptr));
}

/* ---- the cost assertion: never more than one kernel call per output element ---- */

std::uint32_t g_copy_calls = 0U;

infr_status counting_copy(const void* const* inputs, void* const* outputs, std::uint32_t element_count,
                          const infr_quant* quant) {
    ++g_copy_calls;
    const infr_kernel_desc* row = infr_kernel_find("copy", INFR_DTYPE_F32, INFR_QUANT_NONE, INFR_TIER_SCALAR);
    return row == nullptr ? INFR_ERR_NOT_IMPLEMENTED : row->fn(inputs, outputs, element_count, quant);
}

infr_status install_counting_copy() {
    const infr_kernel_desc shim_row = [] {
        infr_kernel_desc d{};
        d.name = "copy";
        d.dtype = INFR_DTYPE_F32;
        d.quant = INFR_QUANT_NONE;
        d.tier = INFR_TIER_VENDOR;
        d.fn = counting_copy;
        return d;
    }();
    const infr_status status = infr_kernel_register(&shim_row);
    return status == INFR_OK ? infr_kernel_set_tier(INFR_TIER_VENDOR) : status;
}

TEST(GraphTransposeF32, CostsAtMostOneCallPerOutputElementOnEveryPath) {
    ASSERT_EQ(INFR_OK, infr_kernel_init());
    ASSERT_EQ(INFR_OK, install_counting_copy());

    const uint32_t in_dims[4] = {1U, 4U, 4U, 8U};
    const uint32_t nchw[4] = {1U, 8U, 4U, 4U};
    std::vector<float> src = make_nhwc(1U, 4U, 4U, 8U);
    infr_tensor in = make_tensor(src.data(), 4U, in_dims);

    alignas(8) uint8_t bytes[4096] = {};
    infr_workspace ws{};
    ASSERT_EQ(INFR_OK, infr_workspace_init(&ws, bytes, sizeof(bytes)));

    /* Identity: one call for 128 elements. */
    {
        std::vector<float> dst(src.size(), 0.0F);
        infr_tensor out = make_tensor(dst.data(), 4U, in_dims);
        infr_transpose_params params{};
        params.rank = 4U;
        for (uint32_t a = 0U; a < 4U; ++a) {
            params.perm[a] = a;
        }
        g_copy_calls = 0U;
        ASSERT_EQ(INFR_OK, infr_transpose_f32(&in, &params, &out, &ws));
        EXPECT_EQ(1U, g_copy_calls) << "identity should be a single copy";
    }
    /* Gather (NHWC -> NCHW): still one call, after the scalar gather. */
    {
        std::vector<float> dst(src.size(), 0.0F);
        infr_tensor out = make_tensor(dst.data(), 4U, nchw);
        infr_transpose_params params{};
        params.rank = 4U;
        params.perm[0] = 0U;
        params.perm[1] = 3U;
        params.perm[2] = 1U;
        params.perm[3] = 2U;
        g_copy_calls = 0U;
        ASSERT_EQ(INFR_OK, infr_transpose_f32(&in, &params, &out, &ws));
        EXPECT_EQ(1U, g_copy_calls) << "the gather path should still issue one copy";
        const double per_element = static_cast<double>(g_copy_calls) / static_cast<double>(src.size());
        EXPECT_LE(per_element, 1.0);
    }
    /* Run path: a permutation that moves axes 1 and 2 but keeps axis 3 in place, so every
     * run of 8 floats is contiguous and the cost is one call per outer position -- 16 of them
     * here (1 x 4 x 4).
     *
     * 段路径：把轴 1、2 换位但保持轴 3 不动的置换，于是每条 8 个 float 的段都连续，代价是
     * 外层每个位置一次调用——这里是 16 次（1 x 4 x 4）。 */
    {
        const uint32_t swapped[4] = {1U, 4U, 4U, 8U};
        std::vector<float> dst(src.size(), 0.0F);
        infr_tensor out = make_tensor(dst.data(), 4U, swapped);
        infr_transpose_params params{};
        params.rank = 4U;
        params.perm[0] = 0U;
        params.perm[1] = 2U;
        params.perm[2] = 1U;
        params.perm[3] = 3U;
        g_copy_calls = 0U;
        ASSERT_EQ(INFR_OK, infr_transpose_f32(&in, &params, &out, &ws));
        EXPECT_EQ(16U, g_copy_calls) << "16 outer positions, one copy each";
        const double per_element = static_cast<double>(g_copy_calls) / static_cast<double>(src.size());
        EXPECT_LE(per_element, 1.0);
    }

    ASSERT_EQ(INFR_OK, infr_kernel_set_tier(INFR_TIER_SCALAR));
}

} // namespace
