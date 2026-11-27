// Conan::ImportStart
#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include <infinfer/infinfer.h>
// Conan::ImportEnd

/*!
 * @ingroup group_runtime
 * @file
 * @brief [en] `infr_concat_f32`: the layout is the whole contract, so the tests are about where the bytes land.
 * @brief [zh] `infr_concat_f32`：布局就是全部契约，所以测试关心的是字节落在哪里。
 * @note [en] A concat has no arithmetic to be approximately right about. Either every element
 *            of every input appears at exactly the right offset -- or the result is a
 *            well-formed tensor that silently feeds wrong features to the next layer. So the
 *            fixtures below are chosen so that a misplaced run lands on a *distinguishable*
 *            value, and the expected output is written out by hand rather than computed.
 * @note [en] The last case is the one that generalises: with the concatenated axis innermost
 *            the cost is one `copy` per position of the outer axes, and the assertion is the
 *            same one the call-count metric makes of every operator -- at most one kernel call
 *            per output element.
 * @note [zh] 拼接没有可以被"大致算对"的算术。要么每个输入的每个元素都恰好落在正确偏移上，
 *            要么得到一个形状合法、却把错特征喂给下一层的张量。所以下面的 fixture 都让
 *            放错的段**落在能分辨出来的值**上，并且期望输出是手写出来的，不是算出来的。
 * @note [zh] 最后一个用例才是可推广的那个：被拼的轴在最内层时，代价是外层轴每个位置一次
 *            `copy`，而断言就是调用计数指标对所有算子的那条——每输出元素最多一次内核调用。
 * @since 0.2.0
 */

namespace {

/* Builds a rank-4 NHWC tensor descriptor over a caller-owned buffer. */
infr_tensor make_tensor(float* data, uint32_t n, uint32_t h, uint32_t w, uint32_t c) {
    infr_tensor t{};
    t.data = data;
    /* A zero-initialised descriptor has no dtype, and the operator refuses it. Set both
     * fields explicitly instead of relying on a default: the first version of this fixture
     * left them zeroed, so every call came back INFR_ERR_UNSUPPORTED and the failures looked
     * like a layout bug in the operator.
     *
     * 零初始化的描述符没有 dtype，算子会拒绝它。两个字段都显式写上，不要依赖默认值：
     * 这个 fixture 的第一版把它们留成 0，于是每次调用都返回 INFR_ERR_UNSUPPORTED，
     * 而失败的样子像是算子里的布局缺陷。 */
    t.dtype = INFR_DTYPE_F32;
    t.quant.kind = INFR_QUANT_NONE;
    t.shape.rank = 4U;
    t.shape.dims[0] = n;
    t.shape.dims[1] = h;
    t.shape.dims[2] = w;
    t.shape.dims[3] = c;
    return t;
}

TEST(GraphConcatF32, ChannelsLandOnTheirOwnStride) {
    ASSERT_EQ(INFR_OK, infr_kernel_init());
    /* 1x2x2 spatial, 3 then 2 channels. The values are distinct per (pixel, channel) so a
     * run written at the wrong pixel or the wrong channel is visible in the diff.
     *
     * 1x2x2 空间，先 3 通道再 2 通道。每个 (像素, 通道) 的值都不同，所以写错像素或写错
     * 通道都会在 diff 里现形。 */
    float a[2 * 2 * 3];
    float b[2 * 2 * 2];
    for (uint32_t p = 0; p < 4U; ++p) {
        for (uint32_t c = 0; c < 3U; ++c) {
            a[(p * 3U) + c] = 10.0F + static_cast<float>((p * 3U) + c);
        }
        for (uint32_t c = 0; c < 2U; ++c) {
            b[(p * 2U) + c] = 100.0F + static_cast<float>((p * 2U) + c);
        }
    }
    infr_tensor in_a = make_tensor(a, 1, 2, 2, 3);
    infr_tensor in_b = make_tensor(b, 1, 2, 2, 2);

    infr_concat_params params{};
    params.axis = 3U;
    params.input_count = 2U;

    /* The signature takes an array of pointers, not a pointer to a tensor. Casting `&in_a`
     * into one compiles and then reads the second entry out of whatever follows on the stack,
     * which is how this test failed the first time it ran.
     *
     * 签名要的是指针数组，不是指向张量的指针。把 `&in_a` 强转成前者能编过，然后从栈上
     * 紧邻的位置读出第二个输入——这个测试第一次运行失败就是这么来的。 */
    const infr_tensor* concat_inputs[2] = {&in_a, &in_b};
    infr_shape out_shape{};
    ASSERT_EQ(INFR_OK, infr_concat_output_shape(concat_inputs, &params, &out_shape));
    EXPECT_EQ(4U, out_shape.rank);
    EXPECT_EQ(1U, out_shape.dims[0]);
    EXPECT_EQ(2U, out_shape.dims[1]);
    EXPECT_EQ(2U, out_shape.dims[2]);
    EXPECT_EQ(5U, out_shape.dims[3]);

    float out[2 * 2 * 5];
    for (float& v : out) {
        v = -1.0F;
    }
    infr_tensor output = make_tensor(out, 1, 2, 2, 5);

    alignas(8) uint8_t workspace_bytes[64] = {};
    infr_workspace ws{};
    ASSERT_EQ(INFR_OK, infr_workspace_init(&ws, workspace_bytes, sizeof(workspace_bytes)));
    const infr_tensor* inputs[2] = {&in_a, &in_b};
    ASSERT_EQ(INFR_OK, infr_concat_f32(inputs, &params, &output, &ws));

    for (uint32_t p = 0; p < 4U; ++p) {
        for (uint32_t c = 0; c < 3U; ++c) {
            EXPECT_EQ(a[(p * 3U) + c], out[(p * 5U) + c]) << "pixel " << p << " channel " << c;
        }
        for (uint32_t c = 0; c < 2U; ++c) {
            EXPECT_EQ(b[(p * 2U) + c], out[(p * 5U) + 3U + c]) << "pixel " << p << " channel " << 3U + c;
        }
    }
}

TEST(GraphConcatF32, RowsLandOnTheRowStride) {
    ASSERT_EQ(INFR_OK, infr_kernel_init());
    /* axis = 1: whole rows move, so a run is `w * c` long and the stride between runs is the
     * output row. A body that treated the axis as channels would interleave them instead.
     *
     * axis = 1：整行搬移，所以一段长 `w * c`、段间步长是输出的一行。把轴当成通道处理的
     * 实现会改成交错。 */
    float top[1 * 2 * 2 * 2];
    float bottom[1 * 1 * 2 * 2];
    for (uint32_t i = 0; i < 8U; ++i) {
        top[i] = static_cast<float>(i);
    }
    for (uint32_t i = 0; i < 4U; ++i) {
        bottom[i] = 50.0F + static_cast<float>(i);
    }
    infr_tensor in_top = make_tensor(top, 1, 2, 2, 2);
    infr_tensor in_bottom = make_tensor(bottom, 1, 1, 2, 2);

    infr_concat_params params{};
    params.axis = 1U;
    params.input_count = 2U;

    float out[1 * 3 * 2 * 2];
    infr_tensor output = make_tensor(out, 1, 3, 2, 2);
    alignas(8) uint8_t workspace_bytes[64] = {};
    infr_workspace ws{};
    ASSERT_EQ(INFR_OK, infr_workspace_init(&ws, workspace_bytes, sizeof(workspace_bytes)));
    const infr_tensor* inputs[2] = {&in_top, &in_bottom};
    ASSERT_EQ(INFR_OK, infr_concat_f32(inputs, &params, &output, &ws));

    for (uint32_t i = 0; i < 8U; ++i) {
        EXPECT_EQ(top[i], out[i]) << "top " << i;
    }
    for (uint32_t i = 0; i < 4U; ++i) {
        EXPECT_EQ(bottom[i], out[8U + i]) << "bottom " << i;
    }
}

TEST(GraphConcatF32, OneInputIsTheIdentity) {
    ASSERT_EQ(INFR_OK, infr_kernel_init());
    float a[1 * 1 * 3 * 2] = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
    infr_tensor in = make_tensor(a, 1, 1, 3, 2);

    infr_concat_params params{};
    params.axis = 3U;
    params.input_count = 1U;

    float out[6] = {0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F};
    infr_tensor output = make_tensor(out, 1, 1, 3, 2);
    alignas(8) uint8_t workspace_bytes[64] = {};
    infr_workspace ws{};
    ASSERT_EQ(INFR_OK, infr_workspace_init(&ws, workspace_bytes, sizeof(workspace_bytes)));
    const infr_tensor* inputs[1] = {&in};
    ASSERT_EQ(INFR_OK, infr_concat_f32(inputs, &params, &output, &ws));
    for (uint32_t i = 0; i < 6U; ++i) {
        EXPECT_EQ(a[i], out[i]) << "element " << i;
    }
}

TEST(GraphConcatF32, RefusesShapesThatWouldCopyTheWrongLength) {
    ASSERT_EQ(INFR_OK, infr_kernel_init());
    float a[1 * 2 * 2 * 3] = {0.0F};
    float b[1 * 2 * 2 * 2] = {0.0F};
    infr_tensor in_a = make_tensor(a, 1, 2, 2, 3);
    infr_tensor in_b = make_tensor(b, 1, 2, 2, 2);

    infr_concat_params params{};
    params.axis = 3U;
    params.input_count = 2U;
    const infr_tensor* inputs[2] = {&in_a, &in_b};

    float out[1 * 2 * 2 * 5] = {0.0F};
    infr_tensor good_output = make_tensor(out, 1, 2, 2, 5);
    alignas(8) uint8_t workspace_bytes[64] = {};
    infr_workspace ws{};
    ASSERT_EQ(INFR_OK, infr_workspace_init(&ws, workspace_bytes, sizeof(workspace_bytes)));

    /* A height mismatch would copy rows of the wrong length into a correctly shaped output:
     * well formed, wrong, and invisible to any downstream shape check.
     *
     * 高度不一致会把长度不对的行拷进形状正确的输出：形式合法、内容是错的，下游任何形状
     * 检查都看不见。 */
    infr_tensor wrong_h = make_tensor(b, 1, 3, 2, 2);
    {
        const infr_tensor* bad[2] = {&in_a, &wrong_h};
        EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, infr_concat_f32(bad, &params, &good_output, &ws));
    }
    /* Output that does not already hold the summed axis. */
    {
        infr_tensor small_output = make_tensor(out, 1, 2, 2, 4);
        EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, infr_concat_f32(inputs, &params, &small_output, &ws));
    }
    /* Rank disagreement. */
    {
        float rank3_data[1 * 2 * 2] = {0.0F};
        infr_tensor rank3{};
        rank3.data = rank3_data;
        rank3.dtype = INFR_DTYPE_F32;
        rank3.quant.kind = INFR_QUANT_NONE;
        rank3.shape.rank = 3U;
        rank3.shape.dims[0] = 1U;
        rank3.shape.dims[1] = 2U;
        rank3.shape.dims[2] = 2U;
        const infr_tensor* bad[2] = {&in_a, &rank3};
        EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, infr_concat_f32(bad, &params, &good_output, &ws));
    }
    /* Axis outside the rank, and a zero input count. */
    {
        infr_concat_params bad_axis = params;
        bad_axis.axis = 4U;
        EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, infr_concat_f32(inputs, &bad_axis, &good_output, &ws));
        infr_concat_params no_inputs = params;
        no_inputs.input_count = 0U;
        EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, infr_concat_f32(inputs, &no_inputs, &good_output, &ws));
    }
    /* NULL arguments. */
    EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, infr_concat_f32(nullptr, &params, &good_output, &ws));
    EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, infr_concat_f32(inputs, nullptr, &good_output, &ws));
    EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, infr_concat_f32(inputs, &params, nullptr, &ws));
    EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, infr_concat_f32(inputs, &params, &good_output, nullptr));
    /* The workspace is unused by concat, but it is still required, like its siblings. */
    {
        const infr_tensor* null_input[2] = {&in_a, nullptr};
        EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, infr_concat_f32(null_input, &params, &good_output, &ws));
    }
}

/* ---- the cost assertion: one kernel call per position of the outer axes ---- */

std::uint32_t g_copy_calls = 0U;

infr_status counting_copy(const void* const* inputs, void* const* outputs, std::uint32_t element_count,
                          const infr_quant* quant) {
    ++g_copy_calls;
    /* Reach the reference through the registry with an explicit scalar tier: the policy is
     * vendor here, so this lookup cannot come back to the shim.
     *
     * 显式按 scalar 档从注册表取参考实现：此处策略是 vendor，所以这次查找不会回到 shim。 */
    const infr_kernel_desc* row = infr_kernel_find("copy", INFR_DTYPE_F32, INFR_QUANT_NONE, INFR_TIER_SCALAR);
    return row == nullptr ? INFR_ERR_NOT_IMPLEMENTED : row->fn(inputs, outputs, element_count, quant);
}

TEST(GraphConcatF32, CostsOneCallPerOuterPositionNotPerElement) {
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

    /* 4x4 = 16 outer positions, 2 inputs, so 32 calls for 512 output elements. One call per
     * output element would be 512, sixteen times more, so the assertion cannot pass by
     * accident. (The first version of this comment said 32 positions and expected 64 calls;
     * the operator was right and the arithmetic in the test was wrong -- which is the same
     * class of mistake the test exists to catch, so it is recorded rather than quietly
     * corrected.)
     *
     * 4x4 = 16 个外层位置、2 个输入，所以是 32 次调用对应 512 个输出元素。每元素一次会是
     * 512 次，多 16 倍，断言不可能碰巧通过。（这段注释的第一版写成 32 个位置、期望 64 次；
     * **算子是对的、测试里的算术是错的**——而这类错正是本测试要抓的，所以记下来而不是
     * 悄悄改掉。） */
    std::vector<float> a(4U * 4U * 8U, 1.5F);
    std::vector<float> b(4U * 4U * 8U, -2.5F);
    infr_tensor in_a = make_tensor(a.data(), 1, 4, 4, 8);
    infr_tensor in_b = make_tensor(b.data(), 1, 4, 4, 8);
    std::vector<float> out(4U * 4U * 16U, 0.0F);
    infr_tensor output = make_tensor(out.data(), 1, 4, 4, 16);

    infr_concat_params params{};
    params.axis = 3U;
    params.input_count = 2U;
    const infr_tensor* inputs[2] = {&in_a, &in_b};
    alignas(8) uint8_t workspace_bytes[64] = {};
    infr_workspace ws{};
    ASSERT_EQ(INFR_OK, infr_workspace_init(&ws, workspace_bytes, sizeof(workspace_bytes)));

    g_copy_calls = 0U;
    ASSERT_EQ(INFR_OK, infr_concat_f32(inputs, &params, &output, &ws));

    EXPECT_EQ(32U, g_copy_calls) << "calls should be outer positions (16) x inputs (2)";
    const double per_element = static_cast<double>(g_copy_calls) / static_cast<double>(out.size());
    EXPECT_LE(per_element, 1.0) << "concat must stay under one kernel call per output element";

    /* And the values still have to be right with the shim in place. */
    EXPECT_EQ(1.5F, out[0]);
    EXPECT_EQ(1.5F, out[7]);
    EXPECT_EQ(-2.5F, out[8]);
    EXPECT_EQ(-2.5F, out[15]);

    ASSERT_EQ(INFR_OK, infr_kernel_set_tier(INFR_TIER_SCALAR));
}

} // namespace


int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
