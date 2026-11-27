// Conan::ImportStart
#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <vector>

#include <infinfer/infinfer.h>
// Conan::ImportEnd

/*!
 * @ingroup group_kernel
 * @file
 * @brief [en] Parity for the two primitives stable softmax is built from: `exp` and `shift`.
 * @brief [zh] 稳定 softmax 所依赖的两个原语的 parity：`exp` 与 `shift`。
 * @note [en] `exp` is compared against `std::exp` in double and allowed one float rounding, because
 *            the reference and `expf` are different implementations of the same function -- that is
 *            the accuracy contract this cell claims, and stating it here is cheaper than implying
 *            bitwise equality nobody promised.
 * @note [zh] `exp` 与 double 的 `std::exp` 比较、允许一个 float 的舍入，因为参考实现与 `expf`
 *            是同一函数的两种实现——这就是本格主张的精度契约，写在这里比暗示"逐位相等"便宜。
 * @since 0.2.0
 */

namespace {

infr_status run_kernel(const char* name, const void* const* inputs, void* const* outputs, uint32_t count) {
    const infr_kernel_desc* row = infr_kernel_find(name, INFR_DTYPE_F32, INFR_QUANT_NONE, INFR_TIER_SCALAR);
    if (row == nullptr) {
        return INFR_ERR_NOT_IMPLEMENTED;
    }
    return row->fn(inputs, outputs, count, nullptr);
}

} // namespace

TEST(KernelParityExp, ScalarFp32MatchesDoubleReference) {
    ASSERT_EQ(INFR_OK, infr_kernel_init());
    const std::vector<float> in = {-3.5F, -1.0F, 0.0F, 0.5F, 2.0F, 7.25F};
    std::vector<float> out(in.size(), 0.0F);
    const void* inputs[1] = {in.data()};
    void* outputs[1] = {out.data()};
    ASSERT_EQ(INFR_OK, run_kernel("exp", inputs, outputs, static_cast<uint32_t>(in.size())));

    for (size_t i = 0U; i < in.size(); ++i) {
        const double expected = std::exp(static_cast<double>(in[i]));
        const double tolerance = std::max(1e-7, std::fabs(expected) * 1e-6);
        EXPECT_NEAR(expected, static_cast<double>(out[i]), tolerance) << "index " << i;
    }
}

TEST(KernelParityShift, ScalarFp32MatchesDoubleReference) {
    ASSERT_EQ(INFR_OK, infr_kernel_init());
    const std::vector<float> in = {-2.0F, -0.5F, 0.0F, 1.5F, 4.0F};
    std::vector<float> out(in.size(), 0.0F);
    std::vector<float> offset = {-1.25F};
    const void* inputs[2] = {in.data(), offset.data()};
    void* outputs[1] = {out.data()};
    ASSERT_EQ(INFR_OK, run_kernel("shift", inputs, outputs, static_cast<uint32_t>(in.size())));

    for (size_t i = 0U; i < in.size(); ++i) {
        EXPECT_FLOAT_EQ(static_cast<float>(static_cast<double>(in[i]) + static_cast<double>(offset[0])), out[i])
            << "index " << i;
    }
}

TEST(KernelParityExp, RefusesMissingBuffers) {
    ASSERT_EQ(INFR_OK, infr_kernel_init());
    EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, run_kernel("exp", nullptr, nullptr, 1U));
}

TEST(KernelParityShift, RefusesMissingBuffers) {
    ASSERT_EQ(INFR_OK, infr_kernel_init());
    EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, run_kernel("shift", nullptr, nullptr, 1U));
}
