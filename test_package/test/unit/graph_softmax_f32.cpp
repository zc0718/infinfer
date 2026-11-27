// Conan::ImportStart
#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <vector>

#include <infinfer/infinfer.h>
// Conan::ImportEnd

/*!
 * @ingroup group_runtime
 * @file
 * @brief [en] `infr_softmax_f32`: the normalisation itself, its numerical guard, and the axis.
 * @brief [zh] `infr_softmax_f32`：归一化本身、它的数值护栏，以及轴。
 * @note [en] Three things are worth a test here. The values against a hand computation, because a
 *            softmax that forgets to divide still looks monotonic. The large-logit case, because
 *            the whole reason `shift` exists is that `expf` overflows above ~88 and a fixture with
 *            small numbers would never notice its absence. And an axis that is not the last one,
 *            because the gather stride is the only shape-dependent line in the operator.
 * @note [zh] 这里值得测三件事。数值要对着手算，因为一个忘了除的 softmax 依然单调。大 logit 的
 *            用例要有，因为 `shift` 存在的全部理由是 `expf` 超过约 88 会溢出，而小数字的
 *            夹具永远发现不了它缺席。轴不能只测末轴，因为 gather 的步长是这个算子里唯一
 *            依赖 shape 的一行。
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

} // namespace

TEST(GraphSoftmaxF32, NormalisesTheLastAxis) {
    ASSERT_EQ(INFR_OK, infr_kernel_init());
    const uint32_t dims[2] = {2U, 3U};
    std::vector<float> in = {1.0F, 2.0F, 3.0F, 10.0F, 10.0F, 10.0F};
    infr_tensor input = make_tensor(in.data(), 2U, dims);
    std::vector<float> out(6U, -1.0F);
    infr_tensor output = make_tensor(out.data(), 2U, dims);
    infr_softmax_params params{};
    params.axis = 1U;

    float bytes[8] = {};
    infr_workspace ws{};
    ASSERT_EQ(INFR_OK, infr_workspace_init(&ws, bytes, sizeof(bytes)));
    ASSERT_EQ(INFR_OK, infr_softmax_f32(&input, &params, &output, &ws));

    /* Hand computation: exp(x - max) / sum, with max = 3 for the first row. */
    const float e0 = std::exp(-2.0F);
    const float e1 = std::exp(-1.0F);
    const float e2 = 1.0F;
    const float denom = e0 + e1 + e2;
    EXPECT_NEAR(e0 / denom, out[0], 1e-6F);
    EXPECT_NEAR(e1 / denom, out[1], 1e-6F);
    EXPECT_NEAR(e2 / denom, out[2], 1e-6F);
    /* A constant row is uniform, which is the case a missing subtraction would still pass -- so it
     * is here to pin the second row, not as the only evidence. */
    EXPECT_NEAR(1.0F / 3.0F, out[3], 1e-6F);
    EXPECT_NEAR(1.0F / 3.0F, out[4], 1e-6F);
    EXPECT_NEAR(1.0F / 3.0F, out[5], 1e-6F);
}

TEST(GraphSoftmaxF32, LargeLogitsStayFinite) {
    ASSERT_EQ(INFR_OK, infr_kernel_init());
    const uint32_t dims[2] = {1U, 2U};
    std::vector<float> in = {1000.0F, 1001.0F};
    infr_tensor input = make_tensor(in.data(), 2U, dims);
    std::vector<float> out(2U, -1.0F);
    infr_tensor output = make_tensor(out.data(), 2U, dims);
    infr_softmax_params params{};
    params.axis = 1U;

    float bytes[8] = {};
    infr_workspace ws{};
    ASSERT_EQ(INFR_OK, infr_workspace_init(&ws, bytes, sizeof(bytes)));
    ASSERT_EQ(INFR_OK, infr_softmax_f32(&input, &params, &output, &ws));

    /* Without the max shift this is inf/inf = NaN. */
    ASSERT_TRUE(std::isfinite(out[0]));
    ASSERT_TRUE(std::isfinite(out[1]));
    EXPECT_NEAR(1.0F / (1.0F + std::exp(1.0F)), out[0], 1e-6F);
    EXPECT_NEAR(std::exp(1.0F) / (1.0F + std::exp(1.0F)), out[1], 1e-6F);
}

TEST(GraphSoftmaxF32, NormalisesAnAxisThatIsNotTheLast) {
    ASSERT_EQ(INFR_OK, infr_kernel_init());
    const uint32_t dims[2] = {2U, 3U};
    /* Columns: {0, 0}, {1, 1}, {2, 2} -- so every column is uniform and any stride mistake that
     * mixes columns shows up as a wrong value rather than as a plausible one. */
    std::vector<float> in = {0.0F, 1.0F, 2.0F, 0.0F, 1.0F, 2.0F};
    infr_tensor input = make_tensor(in.data(), 2U, dims);
    std::vector<float> out(6U, -1.0F);
    infr_tensor output = make_tensor(out.data(), 2U, dims);
    infr_softmax_params params{};
    params.axis = 0U;

    float bytes[8] = {};
    infr_workspace ws{};
    ASSERT_EQ(INFR_OK, infr_workspace_init(&ws, bytes, sizeof(bytes)));
    ASSERT_EQ(INFR_OK, infr_softmax_f32(&input, &params, &output, &ws));

    for (size_t i = 0U; i < out.size(); ++i) {
        EXPECT_NEAR(0.5F, out[i], 1e-6F) << "index " << i;
    }
}

TEST(GraphSoftmaxF32, RefusesWhatItCannotHonour) {
    ASSERT_EQ(INFR_OK, infr_kernel_init());
    const uint32_t dims[2] = {1U, 3U};
    std::vector<float> in = {1.0F, 2.0F, 3.0F};
    infr_tensor input = make_tensor(in.data(), 2U, dims);
    std::vector<float> out(3U, 0.0F);
    infr_tensor output = make_tensor(out.data(), 2U, dims);
    infr_softmax_params params{};
    params.axis = 1U;

    float bytes[8] = {};
    infr_workspace ws{};
    ASSERT_EQ(INFR_OK, infr_workspace_init(&ws, bytes, sizeof(bytes)));

    EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, infr_softmax_f32(nullptr, &params, &output, &ws));
    EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, infr_softmax_f32(&input, nullptr, &output, &ws));
    EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, infr_softmax_f32(&input, &params, nullptr, &ws));
    EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, infr_softmax_f32(&input, &params, &output, nullptr));

    infr_softmax_params bad_axis = params;
    bad_axis.axis = 2U;
    EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, infr_softmax_f32(&input, &bad_axis, &output, &ws));

    const uint32_t wrong[2] = {3U, 1U};
    infr_tensor mismatched = make_tensor(out.data(), 2U, wrong);
    EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, infr_softmax_f32(&input, &params, &mismatched, &ws));

    infr_tensor non_f32 = make_tensor(in.data(), 2U, dims);
    non_f32.dtype = INFR_DTYPE_I8;
    EXPECT_EQ(INFR_ERR_UNSUPPORTED, infr_softmax_f32(&non_f32, &params, &output, &ws));

    infr_tensor non_f32_out = make_tensor(out.data(), 2U, dims);
    non_f32_out.dtype = INFR_DTYPE_I8;
    EXPECT_EQ(INFR_ERR_UNSUPPORTED, infr_softmax_f32(&input, &params, &non_f32_out, &ws));

    float tiny[2] = {};
    infr_workspace small{};
    ASSERT_EQ(INFR_OK, infr_workspace_init(&small, tiny, sizeof(tiny)));
    EXPECT_EQ(INFR_ERR_BUFFER_TOO_SMALL, infr_softmax_f32(&input, &params, &output, &small));
}
