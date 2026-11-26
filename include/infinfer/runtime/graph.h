// Conan::ImportStart
#pragma once
#include <stdint.h>

#include <infinfer/kernel/kernel.h>
#include <infinfer/runtime/tensor.h>
#include <infinfer/runtime/workspace.h>
// Conan::ImportEnd

// C linkage, because this header is included from C and from the C++ core.
/*!
 * @ingroup group_runtime
 * @file graph.h
 *
 * @brief [en] L2 operator semantics: the operators the graph layer assembles out
 *            of registered kernels.
 * @brief [zh] L2 算子语义：图层用已注册的 kernel 组合出来的算子。
 * @note [en] A convolution is not a kernel. INV-3 forbids shape in a kernel
 *            signature, and a convolution needs kernel extent, stride and padding
 *            before it can do any arithmetic -- so it lives here, where shape is
 *            allowed, and is built from the registered `fc` dot product plus the
 *            registered `add` for the bias. `matrix.yml` records that dependency
 *            in its `composed` section, and that is the only place the composition
 *            is written down.
 * @note [zh] 卷积不是 kernel。INV-3 禁止 kernel 签名里出现 shape，而卷积必须先知道
 *            核大小、步长与填充才能算第一个乘加——所以它住在这里（这里允许 shape），
 *            由已注册的 `fc` 点积和已注册的 `add`（加偏置）组合而成。`matrix.yml`
 *            的 `composed` 段记录了这个依赖，那是这份组合唯一的书面来源。
 * @note [en] Every operator here is shape-explicit and allocation-free: the caller
 *            brings the tensors and a workspace, and the operator returns the
 *            workspace cursor to where it found it. That is what keeps `EVAL` from
 *            growing as a graph executes (INV-8).
 * @note [zh] 这里的每个算子都显式吃 shape、且不分配内存：调用方带来张量与工作区，
 *            算子在返回前把工作区游标退回原处。这正是 `EVAL` 不会随图的执行而长胖
 *            的原因（INV-8）。
 * @note [en] Supported subset, stated so it fails loudly instead of silently
 *            degrading: fp32 only, unquantized only, batch 1, rank 4 NHWC input and
 *            output, filter in OHWI order, `groups` either 1 or `in_channels`.
 *            Anything else returns INFR_ERR_UNSUPPORTED.
 * @note [zh] 支持的子集写在这里，好让它**响亮失败**而不是悄悄降级：仅 fp32、仅非量化、
 *            batch 为 1、输入输出为 4 维 NHWC、滤波器按 OHWI 排列、`groups` 只允许 1
 *            或 `in_channels`。其余一律返回 INFR_ERR_UNSUPPORTED。
 * @since 0.1.0
 */

/**
 * @brief [en] Geometry of a convolution: what the shape-dependent part needs.
 * @brief [zh] 卷积的几何参数：依赖 shape 的那部分需要的东西。
 * @since 0.1.0
 */
typedef struct infr_conv2d_params {
    /**
     * @brief [en] Filter height in pixels.
     * @brief [zh] 滤波器的高度（像素）。
     */
    uint32_t kernel_h;
    /**
     * @brief [en] Filter width in pixels.
     * @brief [zh] 滤波器的宽度（像素）。
     */
    uint32_t kernel_w;
    /**
     * @brief [en] Vertical step between output pixels; must be at least 1.
     * @brief [zh] 相邻输出像素的纵向步长；至少为 1。
     */
    uint32_t stride_h;
    /**
     * @brief [en] Horizontal step between output pixels; must be at least 1.
     * @brief [zh] 相邻输出像素的横向步长；至少为 1。
     */
    uint32_t stride_w;
    /**
     * @brief [en] Zero padding added on each side vertically.
     * @brief [zh] 上下各加的零填充。
     */
    uint32_t pad_h;
    /**
     * @brief [en] Zero padding added on each side horizontally.
     * @brief [zh] 左右各加的零填充。
     */
    uint32_t pad_w;
    /**
     * @brief [en] Input channels; must equal the input tensor's innermost extent.
     * @brief [zh] 输入通道数；必须等于输入张量最内轴长度。
     */
    uint32_t in_channels;
    /**
     * @brief [en] Output channels; must equal the filter's outermost extent.
     * @brief [zh] 输出通道数；必须等于滤波器最外轴长度。
     */
    uint32_t out_channels;
    /**
     * @brief [en] 1 for a dense convolution, `in_channels` for a depthwise one.
     * @brief [zh] 1 表示普通卷积，`in_channels` 表示逐通道（depthwise）卷积。
     */
    uint32_t groups;
} infr_conv2d_params;

/**
 * @brief [en] Output shape of a convolution, so callers do not re-derive it.
 * @brief [zh] 卷积的输出形状，免得每个调用方自己再推一遍。
 * @param [in] params Geometry; must not be NULL.
 * @param [in] input_shape Rank-4 NHWC input descriptor; must not be NULL.
 * @param [out] out_shape Receives `[1, out_h, out_w, out_channels]`.
 * @return INFR_OK; INFR_ERR_INVALID_ARGUMENT for a NULL argument, a rank other
 *         than 4, a zero stride, or a filter larger than the padded input.
 * @note [en] The extent arithmetic lives in one place because getting it wrong is
 *            silent: an off-by-one output height still produces plausible numbers,
 *            just shifted.
 * @note [zh] 尺寸算术只写一处，因为算错是**无声的**：输出高度差一格，结果照样是一堆
 *            看起来合理的数字，只是整体错位。
 * @since 0.1.0
 */
infr_status infr_conv2d_output_shape(const infr_conv2d_params *params, const infr_shape *input_shape,
                                     infr_shape *out_shape);

/**
 * @brief [en] fp32 convolution (dense or depthwise) over an NHWC tensor.
 * @brief [zh] 在 NHWC 张量上做 fp32 卷积（普通或逐通道）。
 * @param [in] input Rank-4 NHWC input, `[1, in_h, in_w, in_channels]`.
 * @param [in] filter OHWI filter, `[out_channels, kernel_h, kernel_w, in_channels/groups]`.
 * @param [in] bias `out_channels` per-channel offsets, or NULL for none.
 * @param [in] params Geometry; `in_channels`/`out_channels` must match the tensors.
 * @param [in,out] output Rank-4 NHWC output whose shape must already equal
 *                 `infr_conv2d_output_shape`; the data is overwritten.
 * @param [in,out] workspace Scratch source; rewound before returning, on every path.
 * @return INFR_OK; INFR_ERR_INVALID_ARGUMENT for NULL arguments, mismatched shapes,
 *         or a filter larger than the padded input; INFR_ERR_UNSUPPORTED for a dtype,
 *         quantisation or grouping outside the stated subset;
 *         INFR_ERR_BUFFER_TOO_SMALL when the workspace cannot hold the scratch
 *         (`kernel_h * kernel_w * in_channels/groups + out_channels` floats).
 * @note [en] The inner loop is a call to the registered `fc` (dot product) and the
 *            bias step is a call to the registered `add`, so a faster tier of those
 *            two kernels speeds this operator up without touching this file. The
 *            gather is the only shape-dependent work, and it is scalar here.
 * @note [zh] 内层循环是一次对已注册 `fc`（点积）的调用，偏置步是一次对已注册 `add`
 *            的调用——所以那两个 kernel 换了更快的档位，本算子自动变快，无需改这里。
 *            只有 gather 是依赖 shape 的活，它在本档是标量实现。
 * @since 0.1.0
 */
infr_status infr_conv2d_f32(const infr_tensor *input, const infr_tensor *filter, const float *bias,
                            const infr_conv2d_params *params, infr_tensor *output, infr_workspace *workspace);
