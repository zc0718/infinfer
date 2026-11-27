// Conan::ImportStart
#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include <infinfer/infinfer.h>
// Conan::ImportEnd

/*!
 * @ingroup group_contract
 * @file
 * @brief [en] How many kernel calls an operator makes per output element.
 * @brief [zh] 一个算子每个输出元素要发多少次内核调用。
 * @note [en] This is the guard rail for a mistake that was made three times in one week:
 *            a composed operator calling a kernel once per output element, so the call cost
 *            dominates the arithmetic. Measured on a depthwise layer it was 18.6 ns of fixed
 *            cost against 36.7 ns total (gaps G-129), and the same shape showed up again in
 *            the gather bounds test (G-132). Nothing in the tree noticed, because a
 *            per-element call is correct -- it is only slow.
 * @note [en] The instrument is the tier policy itself: the shims below register at
 *            `INFR_TIER_VENDOR`, which nothing else claims, so switching the policy to that
 *            tier makes every kernel the operator asks for a counting wrapper that forwards
 *            to the scalar body. No production code changes, and the count is exact.
 * @note [zh] 这是一周内犯过三次的那个错的护栏：组合算子每个输出元素调用一次内核，于是调用
 *            成本盖过算术。深卷积上实测是每次 18.6 ns 固定开销、总共只有 36.7 ns
 *            （gaps G-129），同样的形状又在 gather 的边界检查里出现（G-132）。树里没有
 *            任何东西发现它，因为「每元素一次调用」是**正确**的——只是慢。
 * @note [zh] 仪器就是档位策略本身：下面的 shim 登记在 `INFR_TIER_VENDOR`（无人认领的那
 *            一档），把策略切过去之后，算子要的每个内核都变成计数包装、再转发给 scalar
 *            体。生产代码一行不改，计数是精确的。
 * @since 0.2.0
 */

namespace {

/* Counts, one per kernel name this file cares about. */
std::uint32_t g_fc_calls = 0U;
std::uint32_t g_add_calls = 0U;
std::uint32_t g_dwconv_calls = 0U;

/* The bodies are not public, so each shim reaches the reference through the registry with
 * an explicit scalar tier -- that cannot recurse, because the policy is vendor and this
 * lookup names scalar.
 *
 * 那些体不是公开的，所以每个 shim 都显式按 scalar 档从注册表取参考实现——这不会递归，因为
 * 策略是 vendor 而这次查找点名 scalar。
 */
infr_kernel_fn scalar_body(const char* name) {
    const infr_kernel_desc* row = infr_kernel_find(name, INFR_DTYPE_F32, INFR_QUANT_NONE, INFR_TIER_SCALAR);
    return row == nullptr ? nullptr : row->fn;
}

infr_status counting_fc(const void* const* inputs, void* const* outputs, std::uint32_t element_count,
                        const infr_quant* quant) {
    ++g_fc_calls;
    return scalar_body("fc")(inputs, outputs, element_count, quant);
}

infr_status counting_add(const void* const* inputs, void* const* outputs, std::uint32_t element_count,
                         const infr_quant* quant) {
    ++g_add_calls;
    return scalar_body("add")(inputs, outputs, element_count, quant);
}

infr_status counting_dwconv(const void* const* inputs, void* const* outputs, std::uint32_t element_count,
                            const infr_quant* quant) {
    ++g_dwconv_calls;
    return scalar_body("dwconv3x3")(inputs, outputs, element_count, quant);
}

struct Fixture {
    std::vector<float> in;
    std::vector<float> w;
    std::vector<float> out;
    std::vector<unsigned char> arena;
    infr_tensor input{};
    infr_tensor filter{};
    infr_tensor output{};
    infr_workspace ws{};
    infr_conv2d_params params{};
};

/* A layer whose every number is small enough to read in a failure message. */
void build(Fixture* f, std::uint32_t in_h, std::uint32_t in_w, std::uint32_t channels, std::uint32_t groups) {
    const std::uint32_t out_h = in_h - 2U;
    const std::uint32_t out_w = in_w - 2U;
    const std::uint32_t k_in = channels / groups;
    f->in.assign(static_cast<std::size_t>(in_h) * in_w * channels, 0.5F);
    f->w.assign(static_cast<std::size_t>(channels) * 9U * k_in, 0.25F);
    f->out.assign(static_cast<std::size_t>(out_h) * out_w * channels, 0.0F);
    f->arena.assign(1U << 16, 0U);

    infr_shape in_shape{};
    in_shape.rank = 4;
    in_shape.dims[0] = 1;
    in_shape.dims[1] = in_h;
    in_shape.dims[2] = in_w;
    in_shape.dims[3] = channels;
    in_shape.elem_count = 1U * in_h * in_w * channels;
    infr_shape filter_shape{};
    filter_shape.rank = 4;
    filter_shape.dims[0] = channels;
    filter_shape.dims[1] = 3;
    filter_shape.dims[2] = 3;
    filter_shape.dims[3] = k_in;
    filter_shape.elem_count = channels * 9U * k_in;
    infr_shape out_shape{};
    out_shape.rank = 4;
    out_shape.dims[0] = 1;
    out_shape.dims[1] = out_h;
    out_shape.dims[2] = out_w;
    out_shape.dims[3] = channels;
    out_shape.elem_count = 1U * out_h * out_w * channels;

    EXPECT_EQ(INFR_OK, infr_tensor_init(&f->input, f->in.data(), INFR_DTYPE_F32, &in_shape));
    EXPECT_EQ(INFR_OK, infr_tensor_init(&f->filter, f->w.data(), INFR_DTYPE_F32, &filter_shape));
    EXPECT_EQ(INFR_OK, infr_tensor_init(&f->output, f->out.data(), INFR_DTYPE_F32, &out_shape));
    EXPECT_EQ(INFR_OK, infr_workspace_init(&f->ws, f->arena.data(), f->arena.size()));

    f->params.kernel_h = 3;
    f->params.kernel_w = 3;
    f->params.stride_h = 1;
    f->params.stride_w = 1;
    f->params.in_channels = channels;
    f->params.out_channels = channels;
    f->params.groups = groups;
}

class CallCount : public ::testing::Test {
  protected:
    void SetUp() override {
        EXPECT_EQ(INFR_OK, infr_kernel_init());
        g_fc_calls = 0U;
        g_add_calls = 0U;
        g_dwconv_calls = 0U;

        /* One shim per kernel the operators below ask for. Registered at the vendor tier so
         * the policy can select them without disturbing anything else.
         *
         * 下面这些算子要的内核各一个 shim，登记在厂商档，这样策略能把它们选中而不打扰
         * 别的东西。 */
        static const infr_kernel_desc rows[] = {
            {.name = "fc",
             .dtype = INFR_DTYPE_F32,
             .quant = INFR_QUANT_NONE,
             .tier = INFR_TIER_VENDOR,
             .fn = counting_fc},
            {.name = "add",
             .dtype = INFR_DTYPE_F32,
             .quant = INFR_QUANT_NONE,
             .tier = INFR_TIER_VENDOR,
             .fn = counting_add},
            {.name = "dwconv3x3",
             .dtype = INFR_DTYPE_F32,
             .quant = INFR_QUANT_NONE,
             .tier = INFR_TIER_VENDOR,
             .fn = counting_dwconv},
        };
        for (const infr_kernel_desc& row : rows) {
            EXPECT_EQ(INFR_OK, infr_kernel_register(&row));
        }
        ASSERT_EQ(INFR_OK, infr_kernel_set_tier(INFR_TIER_VENDOR));
    }
    void TearDown() override { EXPECT_EQ(INFR_OK, infr_kernel_set_tier(INFR_TIER_SCALAR)); }
};

TEST_F(CallCount, ADepthwisePixelCostsOneCallNotOnePerChannel) {
    Fixture f;
    build(&f, 16U, 16U, 32U, /*groups=*/32U);
    ASSERT_EQ(INFR_OK, infr_conv2d_f32(&f.input, &f.filter, nullptr, &f.params, &f.output, &f.ws));

    const std::uint32_t out_h = 14U;
    const std::uint32_t out_w = 14U;
    const std::uint32_t pixels = out_h * out_w;
    const std::uint32_t outputs = pixels * 32U;

    /* One call per output pixel, and the fused cell also covers the bias-free path's copy
     * (there is no `add` here because the bias is NULL).
     *
     * 一个输出像素一次调用；无偏置路径里没有 `add`（bias 为 NULL）。 */
    EXPECT_EQ(pixels, g_dwconv_calls) << "depthwise must batch a pixel, not a channel";
    EXPECT_EQ(0U, g_fc_calls) << "the fused cell should have replaced every per-channel dot";
    EXPECT_GE(outputs, g_dwconv_calls) << "sanity: fewer calls than outputs is the whole point";
}

TEST_F(CallCount, ADensePixelStillCostsOneCallPerChannel) {
    Fixture f;
    build(&f, 16U, 16U, 8U, /*groups=*/1U);
    ASSERT_EQ(INFR_OK, infr_conv2d_f32(&f.input, &f.filter, nullptr, &f.params, &f.output, &f.ws));

    const std::uint32_t pixels = 14U * 14U;
    /* Dense is not batched yet, so this is exactly one call per output element -- recorded
     * rather than required. When a batched dense cell lands this number must fall, and the
     * case failing is how anyone would notice it landed.
     *
     * 稠密还没批处理，所以正好是每个输出元素一次——这是**记录**而不是要求的。等批处理版
     * 稠密落地，这个数必须下降，而这个用例变红就是别人知道它落地的途径。 */
    EXPECT_EQ(pixels * 8U, g_fc_calls) << "dense is one dot per output element today";
}

} // namespace


int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
