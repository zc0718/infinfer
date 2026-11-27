// Conan::ImportStart
#pragma once
#include <cstddef>
#include <stdint.h>

#include <infinfer/runtime/tensor.h>
// Conan::ImportEnd

/*!
 * @ingroup group_runtime
 * @file shape_internal.hpp
 *
 * @brief [en] Shape arithmetic shared by the L2 operators; not part of the public API.
 * @brief [zh] L2 算子共用的尺寸算术；不属于公开 API。
 * @note [en] There is exactly one spelling of "how many outputs fit along one axis",
 *            because two spellings drift. It was about to be copied into the pooling
 *            operator as well, which is the point at which a helper is cheaper than a
 *            second copy.
 * @note [zh] "一条轴能放下几个输出"只有一种写法，因为两种写法必然漂移。它本来快要
 *            被复制进池化算子，而那正是"抽一个 helper 比复制第二份更便宜"的临界点。
 * @since 0.1.0
 */

namespace infr_detail {

/**
 * @brief [en] Output extent for a strided window with symmetric zero padding (floor).
 * @brief [zh] 带对称零填充的滑动窗口输出长度（向下取整）。
 * @param in [en] Input extent along this axis.
 * @param in [zh] 该轴上的输入长度。
 * @param kernel [en] Window extent; 0 is rejected.
 * @param kernel [zh] 窗口长度；0 被拒绝。
 * @param pad [en] Padding on each side.
 * @param pad [zh] 每侧的填充量。
 * @param stride [en] Step between windows; 0 is rejected.
 * @param stride [zh] 相邻窗口的步长；0 被拒绝。
 * @param out [en] Receives the extent; untouched on failure.
 * @param out [zh] 接收输出长度；失败时不改动。
 * @return INFR_OK, or INFR_ERR_INVALID_ARGUMENT when the window cannot fit at all.
 * @note [en] The window may hang off the *end* (floor drops the partial one) but may
 *            not be longer than the padded input: that case has no output at all and
 *            is an argument error rather than an empty result.
 * @note [zh] 窗口可以从**末端**溢出（向下取整丢掉不完整的那一个），但不能比补齐后的
 *            输入还长：那种情况一个输出都没有，属于参数错误而不是空结果。
 * @since 0.1.0
 */
inline infr_status out_extent(uint32_t in, uint32_t kernel, uint32_t pad, uint32_t stride, uint32_t* out) {
    if (stride == 0U || kernel == 0U) {
        return INFR_ERR_INVALID_ARGUMENT;
    }
    const uint32_t padded = in + (2U * pad);
    if (padded < kernel) {
        return INFR_ERR_INVALID_ARGUMENT;
    }
    *out = ((padded - kernel) / stride) + 1U;
    return INFR_OK;
}

/**
 * @brief [en] Whether a tensor is fp32 and unquantized -- the only dtype the L2
 *            operators currently promise.
 * @brief [zh] 张量是否为 fp32 且未量化——这是 L2 算子目前唯一承诺的类型。
 * @param tensor [en] Tensor to inspect; must not be NULL.
 * @param tensor [zh] 要检查的张量；不得为 NULL。
 * @return True when `dtype == INFR_DTYPE_F32` and `quant.kind == INFR_QUANT_NONE`.
 * @since 0.1.0
 */
inline bool is_f32(const infr_tensor* tensor) {
    return tensor->dtype == INFR_DTYPE_F32 && tensor->quant.kind == INFR_QUANT_NONE;
}

/**
 * @brief [en] Advance a mixed-radix odometer and keep an offset in step with it.
 * @brief [zh] 推进混合进制里程计，并让偏移跟随它。
 * @param idx [en] Digits; modified in place.
 * @param idx [zh] 各位数字；就地修改。
 * @param dim [en] Radix of each digit.
 * @param dim [zh] 各位的进制。
 * @param delta [en] Offset contributed by one step of each digit.
 * @param delta [zh] 各位走一步贡献的偏移。
 * @param count [en] Number of digits, at least 1.
 * @param count [zh] 位数，至少为 1。
 * @param offset [en] Running offset; updated in place.
 * @param offset [zh] 运行中的偏移；就地更新。
 * @return [en] False once the odometer has run past its last position.
 * @return [zh] 里程计走完最后一个位置后返回 false。
 * @note [en] This sits here for the same reason `out_extent` does: `transpose` and `slice` both
 *            walk a strided view, and a second spelling of the carry correction would drift. The
 *            correction subtracts `dim * delta` after having just added `delta`, which leaves
 *            the digit at 0 with its contribution removed -- an off-by-one-delta carry is the
 *            classic "one wrong element per row" bug, so the arithmetic is written out.
 * @note [zh] 它放在这里的理由和 `out_extent` 相同：`transpose` 与 `slice` 都要走带步长的视图，
 *            第二份进位修正的写法必然漂移。修正是刚加过 `delta` 之后再减 `dim * delta`，于是
 *            该位回到 0 且贡献被移除——少见/多减一个 `delta` 正是经典的「每行错一个元素」，
 *            所以把算术写明白。
 * @since 0.2.0
 */
inline bool advance_odometer(uint32_t* idx, const uint32_t* dim, const ptrdiff_t* delta, uint32_t count,
                             ptrdiff_t* offset) {
    for (uint32_t a = count; a-- > 0U;) {
        ++idx[a];
        *offset += delta[a];
        if (idx[a] < dim[a]) {
            return true;
        }
        idx[a] = 0U;
        *offset -= static_cast<ptrdiff_t>(dim[a]) * delta[a];
    }
    return false;
}

} // namespace infr_detail
