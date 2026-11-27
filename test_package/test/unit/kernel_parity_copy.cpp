// Conan::ImportStart
#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include <infinfer/infinfer.h>
// Conan::ImportEnd

/*!
 * @ingroup group_scalar
 * @file
 * @brief [en] Parity for `copy`: the layout operators' primitive.
 * @brief [zh] `copy` 的 parity：布局类算子的原语。
 * @note [en] A copy is easy to get right and easy to get *almost* right, and the second kind is
 *            the dangerous one: an off-by-one here surfaces later as a wrong pixel in a
 *            concatenated feature map, which no arithmetic test will notice. So the cases below
 *            check the exact bytes, that nothing outside `element_count` is touched, and that a
 *            zero-length call is a no-op rather than an error -- the layout operators do produce
 *            empty runs when a dimension is zero.
 * @note [zh] 拷贝很容易写对，也很容易写得**几乎对**，而后者才危险：这里差一个元素，之后会
 *            表现为拼接后特征图里一个错的像素，任何算术测试都发现不了。所以下面这些用例检查
 *            精确字节、检查 `element_count` 之外一个字节都没被动过，以及零长度调用是空操作而
 *            不是错误——某个维度为 0 时布局算子确实会产出空段。
 * @since 0.2.0
 */

namespace {

const infr_kernel_desc* copy_row() {
    return infr_kernel_find("copy", INFR_DTYPE_F32, INFR_QUANT_NONE, INFR_TIER_SCALAR);
}

TEST(KernelParityCopy, CopiesTheRunExactly) {
    EXPECT_EQ(INFR_OK, infr_kernel_register_scalar());
    const infr_kernel_desc* row = copy_row();
    ASSERT_NE(nullptr, row) << "copy/fp32/none/scalar is not registered";

    for (std::uint32_t n : {1U, 3U, 4U, 7U, 64U, 257U}) {
        std::vector<float> src(n);
        for (std::uint32_t i = 0; i < n; ++i) {
            src[i] = static_cast<float>(i) * 0.5F - 3.0F;
        }
        /* The destination is pre-filled with a value the source can never contain, and one
         * sentinel float follows the run: a body that copied `n + 1` elements would overwrite
         * it, and a body that copied `n - 1` would leave a sentinel value inside the run.
         *
         * 目标缓冲区预填成源里不可能出现的值，并在区间后留一个哨兵：多拷一个会覆盖哨兵，
         * 少拷一个会把哨兵值留在区间内。 */
        std::vector<float> dst(n + 1U, 1234.5F);
        const void* inputs[1] = {src.data()};
        void* outputs[1] = {dst.data()};
        ASSERT_EQ(INFR_OK, row->fn(inputs, outputs, n, nullptr)) << "n " << n;
        for (std::uint32_t i = 0; i < n; ++i) {
            EXPECT_EQ(src[i], dst[i]) << "element " << i << " of " << n;
        }
        EXPECT_EQ(1234.5F, dst[n]) << "the body wrote past element_count, n " << n;
    }
}

TEST(KernelParityCopy, CountsOnlyWhatItIsTold) {
    EXPECT_EQ(INFR_OK, infr_kernel_register_scalar());
    const infr_kernel_desc* row = copy_row();
    ASSERT_NE(nullptr, row);

    float src[4] = {1.0F, 2.0F, 3.0F, 4.0F};
    float dst[4] = {0.0F, 0.0F, 0.0F, 0.0F};
    const void* inputs[1] = {src};
    void* outputs[1] = {dst};

    /* Zero elements is a no-op, not an error: the layout operators produce empty runs.
     *
     * 零个元素是空操作而不是错误：布局算子会产出空段。 */
    EXPECT_EQ(INFR_OK, row->fn(inputs, outputs, 0U, nullptr));
    EXPECT_EQ(0.0F, dst[0]);

    EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, row->fn(nullptr, outputs, 4U, nullptr));
    EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, row->fn(inputs, nullptr, 4U, nullptr));
    {
        const void* null_source[1] = {nullptr};
        EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, row->fn(null_source, outputs, 4U, nullptr));
    }
    {
        void* null_dest[1] = {nullptr};
        EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, row->fn(inputs, null_dest, 4U, nullptr));
    }
}

} // namespace
