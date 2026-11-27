// Conan::ImportStart
#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include <infinfer/infinfer.h>

#include "parity_harness.hpp"
// Conan::ImportEnd

/*!
 * @ingroup group_scalar
 * @file
 * @brief [en] Parity for `dwconv3x3`: the 9-tap window sum against the definition in double.
 * @brief [zh] `dwconv3x3` 的 parity：九抽头窗口求和对照 double 域的定义。
 * @note [en] The cell exists because a depthwise convolution used to cost one kernel call
 *            per output element, and half of its measured time was that call (gaps G-129).
 *            The kernel is therefore judged on two things and not one: that it computes
 *            the 9-tap sum the operator means, and that it really does so for **many**
 *            output elements in one call, because that is the entire reason it exists --
 *            a body that only handled `element_count == 1` would pass a value check and
 *            buy nothing.
 * @note [zh] 这一格存在的原因是：深卷积原来每个输出元素一次内核调用，而实测有一半时间
 *            花在调用上（gaps G-129）。所以这个内核要判两件事而不是一件：它算的确实是
 *            算子所想的九抽头和；它确实在**一次调用**里处理了很多输出元素——否则一个只
 *            支持 `element_count == 1` 的体会通过数值检查，却一分收益都没有。
 * @since 0.2.0
 */

namespace {

/* Deterministic corpus: an integer LCG keeps the values identical on every target. */
void fill(std::vector<float>& v, std::size_t n, std::uint32_t seed) {
    v.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
        const std::uint32_t state = (static_cast<std::uint32_t>(i) * 1664525U) + 1013904223U + seed;
        v[i] = static_cast<float>(static_cast<std::int32_t>((state >> 8) % 20001U) - 10000) / 100.0F;
    }
}

TEST(KernelParityDwconv3x3, ScalarFp32MatchesDoubleReference) {
    EXPECT_EQ(INFR_OK, infr_kernel_register_scalar());
    const infr_kernel_desc* row = infr_kernel_find("dwconv3x3", INFR_DTYPE_F32, INFR_QUANT_NONE, INFR_TIER_SCALAR);
    ASSERT_NE(nullptr, row) << "dwconv3x3/fp32/none/scalar is not registered";

    /* One call, many output elements: 1, 3, 8, 64 and 257 cover the empty-ish, the
     * short, the vector-width, the cache-line and the ragged cases.
     *
     * 一次调用、多个输出元素：1、3、8、64、257 覆盖空、短、向量宽度、缓存行与不规则
     * 长度这几种情形。 */
    for (std::uint32_t rows : {1U, 3U, 8U, 64U, 257U}) {
        std::vector<float> window;
        std::vector<float> weights;
        std::vector<float> got(rows, 0.0F);
        fill(window, static_cast<std::size_t>(rows) * 9U, 7U);
        fill(weights, static_cast<std::size_t>(rows) * 9U, 11U);

        const void* inputs[2] = {window.data(), weights.data()};
        void* outputs[1] = {got.data()};
        ASSERT_EQ(INFR_OK, row->fn(inputs, outputs, rows, nullptr)) << "rows " << rows;

        for (std::uint32_t r = 0; r < rows; ++r) {
            /* The reference sums in double and in the order the source spells out, so a
             * reordered body shows up as a difference rather than being absorbed.
             *
             * 参考实现在 double 域按源码写出的顺序求和，所以换了顺序的体会表现为差异，
             * 而不是被吸收掉。 */
            double want = 0.0;
            for (std::uint32_t j = 0; j < 9U; ++j) {
                want += static_cast<double>(window[(static_cast<std::size_t>(r) * 9U) + j]) *
                        static_cast<double>(weights[(static_cast<std::size_t>(r) * 9U) + j]);
            }
            EXPECT_NEAR(want, static_cast<double>(got[r]), 1e-5 * (1.0 + std::abs(want)))
                << "row " << r << " of " << rows;
        }
    }
}

TEST(KernelParityDwconv3x3, RefusesAMissingPointer) {
    EXPECT_EQ(INFR_OK, infr_kernel_register_scalar());
    const infr_kernel_desc* row = infr_kernel_find("dwconv3x3", INFR_DTYPE_F32, INFR_QUANT_NONE, INFR_TIER_SCALAR);
    ASSERT_NE(nullptr, row);

    float window[9] = {0.0F};
    float weights[9] = {0.0F};
    float out = 0.0F;
    const void* with_null_input[2] = {nullptr, weights};
    const void* with_null_weights[2] = {window, nullptr};
    const void* good[2] = {window, weights};
    void* outputs[1] = {&out};

    EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, row->fn(with_null_input, outputs, 1U, nullptr));
    EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, row->fn(with_null_weights, outputs, 1U, nullptr));
    EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, row->fn(good, nullptr, 1U, nullptr));
    EXPECT_EQ(INFR_OK, row->fn(good, outputs, 0U, nullptr)) << "zero elements is not an error";
}

} // namespace
