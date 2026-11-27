// Conan::ImportStart
#pragma once
#include <stdint.h>

#include <infinfer/kernel/kernel.h>
#include <infinfer/runtime/tensor.h>
#include <infinfer/runtime/workspace.h>
// Conan::ImportEnd

// C linkage, because this header is included from C and from the C++ core.
#ifdef __cplusplus
extern "C" {
#endif

/*!
 * @ingroup group_runtime
 * @file graph.h
 *
 * @brief [en] L2 operator semantics: the operators the graph layer assembles out
 *            of registered kernels.
 * @brief [zh] L2 算子语义：图层用已注册的 kernel 组合出来的算子。
 * @note [en] A convolution is not a kernel: shape has no place in a kernel
 *            signature, and a convolution needs kernel extent, stride and padding
 *            before it can do any arithmetic -- so it lives here, where shape is
 *            allowed, and is built from the registered `fc` dot product plus the
 *            registered `add` for the bias. `matrix.yml` records that dependency
 *            in its `composed` section, and that is the only place the composition
 *            is written down.
 * @note [zh] 卷积不是 kernel：kernel 签名里不出现 shape，而卷积必须先知道
 *            核大小、步长与填充才能算第一个乘加——所以它住在这里（这里允许 shape），
 *            由已注册的 `fc` 点积和已注册的 `add`（加偏置）组合而成。`matrix.yml`
 *            的 `composed` 段记录了这个依赖，那是这份组合唯一的书面来源。
 * @note [en] Every operator here is shape-explicit and allocation-free: the caller
 *            brings the tensors and a workspace, and the operator returns the
 *            workspace cursor to where it found it. That is what keeps `EVAL` from
 *            growing as a graph executes.
 * @note [zh] 这里的每个算子都显式吃 shape、且不分配内存：调用方带来张量与工作区，
 *            算子在返回前把工作区游标退回原处。这正是 `EVAL` 不会随图的执行而长胖
 *            的原因。
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
 * @param params [en] Geometry; must not be NULL.
 * @param params [zh] 几何参数；不得为 NULL。
 * @param input_shape [en] Rank-4 NHWC input descriptor; must not be NULL.
 * @param input_shape [zh] 4 维 NHWC 输入描述符；不得为 NULL。
 * @param out_shape [en] Receives `[1, out_h, out_w, out_channels]`.
 * @param out_shape [zh] 接收 `[1, out_h, out_w, out_channels]`。
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
 * @param input [en] Rank-4 NHWC input, `[1, in_h, in_w, in_channels]`.
 * @param input [zh] 4 维 NHWC 输入，`[1, in_h, in_w, in_channels]`。
 * @param filter [en] OHWI filter, `[out_channels, kernel_h, kernel_w, in_channels/groups]`.
 * @param filter [zh] OHWI 滤波器，`[out_channels, kernel_h, kernel_w, in_channels/groups]`。
 * @param bias [en] `out_channels` per-channel offsets, or NULL for none.
 * @param bias [zh] `out_channels` 个逐通道偏移；不使用时为 NULL。
 * @param params [en] Geometry; `in_channels`/`out_channels` must match the tensors.
 * @param params [zh] 几何参数；`in_channels`/`out_channels` 必须与张量一致。
 * @param output [en] Rank-4 NHWC output whose shape must already equal `infr_conv2d_output_shape`; the data is overwritten.
 * @param output [zh] 4 维 NHWC 输出，其形状必须已等于 `infr_conv2d_output_shape`；数据会被覆盖。
 * @param workspace [en] Scratch source; rewound before returning, on every path.
 * @param workspace [zh] 临时空间来源；任何返回路径上都会先回退。
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

/**
 * @brief [en] int8 convolution (dense or depthwise) over an NHWC tensor, with an int32 bias and one
 *            requantization.
 * @brief [zh] 在 NHWC 张量上做 int8 卷积（普通或逐通道），偏置为 int32，全程只再量化一次。
 * @param input [en] Rank-4 NHWC int8 input, `[1, in_h, in_w, in_channels]`, quantised per tensor
 *            with a zero point of 0. Padding is the quantised zero, so an asymmetric (u8-style)
 *            input is refused rather than silently padded with the wrong value.
 * @param input [zh] 4 维 NHWC int8 输入，`[1, in_h, in_w, in_channels]`，按张量量化且零点为 0。
 *            填充用的是量化后的零，所以非对称（u8 风格）输入会被**拒绝**，而不是被填错值。
 * @param filter [en] OHWI int8 filter, `[out_channels, kernel_h, kernel_w, in_channels/groups]`,
 *            quantised per channel (or per tensor); scales must be positive.
 * @param filter [zh] OHWI int8 滤波器，`[out_channels, kernel_h, kernel_w, in_channels/groups]`，
 *            逐通道（或逐张量）量化；scale 必须为正。
 * @param bias [en] `out_channels` int32 offsets, or NULL for none. Added to the accumulator
 *            *before* the requantization, which is the reason this operator is its own function.
 * @param bias [zh] `out_channels` 个 int32 偏移，不用时为 NULL。它加在**再量化之前**的累加器上
 *            ——这正是本算子自成一个函数的理由。
 * @param params [en] Geometry; `in_channels`/`out_channels` must match the tensors.
 * @param params [zh] 几何参数；`in_channels`/`out_channels` 必须与张量一致。
 * @param output [en] Rank-4 NHWC int8 output whose shape must already equal
 *            `infr_conv2d_output_shape`; quantised per tensor, and its zero point may be anything.
 * @param output [zh] 4 维 NHWC int8 输出，其形状必须已等于 `infr_conv2d_output_shape`；按张量
 *            量化，零点可以任意。
 * @param workspace [en] Scratch source; rewound before returning, on every path. Needs
 *            `kernel_h * kernel_w * in_channels/groups` **bytes**, not floats: the accumulator for
 *            one output element lives in a local, so there is no second buffer to borrow.
 * @param workspace [zh] 临时空间来源；任何返回路径上都会先回退。需要
 *            `kernel_h * kernel_w * in_channels/groups` 个**字节**而非 float：单个输出元素的
 *            累加器在局部变量里，因此没有第二块缓冲区要借。
 * @return INFR_OK; INFR_ERR_INVALID_ARGUMENT for NULL arguments, mismatched shapes, a
 *         non-positive scale, or a filter larger than the padded input; INFR_ERR_UNSUPPORTED for a
 *         dtype, quantisation scheme or grouping outside the stated subset;
 *         INFR_ERR_BUFFER_TOO_SMALL when the workspace cannot hold the window;
 *         INFR_ERR_NOT_IMPLEMENTED when no registered tier supplies `dot` or `requant`.
 * @note [en] The multiplier is resolved once per output channel as
 *            `(input.scale * weight.scale[oc]) / output.scale` and handed to `requant`, so the
 *            division never appears in the inner loop and rounding has exactly one definition.
 * @note [zh] 乘子按输出通道解析一次为 `(input.scale * weight.scale[oc]) / output.scale` 后交给
 *            `requant`，因此内层循环里不出现除法，而舍入只有一处定义。
 * @note [en] Two indirect calls per output element (`dot`, then `requant`). That is the call
 *            structure the fp32 operator above already has, and it is deliberate: a faster tier of
 *            either kernel speeds this operator up without a line changing here.
 * @note [zh] 每个输出元素两次间接调用（先 `dot`，再 `requant`）。这与上面的 fp32 算子调用结构
 *            相同，而且是刻意的：那两个内核哪一档变快，本算子就随之变快，这里一行都不用改。
 * @since 0.2.0
 */
infr_status infr_conv2d_i8(const infr_tensor *input, const infr_tensor *filter, const int32_t *bias,
                           const infr_conv2d_params *params, infr_tensor *output, infr_workspace *workspace);

/**
 * @brief [en] Window geometry of a pooling operator.
 * @brief [zh] 池化算子的窗口几何。
 * @since 0.1.0
 */
typedef struct infr_pool_params {
    /**
     * @brief [en] Window height in pixels.
     * @brief [zh] 窗口高度（像素）。
     */
    uint32_t kernel_h;
    /**
     * @brief [en] Window width in pixels.
     * @brief [zh] 窗口宽度（像素）。
     */
    uint32_t kernel_w;
    /**
     * @brief [en] Vertical step between windows; must be at least 1.
     * @brief [zh] 相邻窗口的纵向步长；至少为 1。
     */
    uint32_t stride_h;
    /**
     * @brief [en] Horizontal step between windows; must be at least 1.
     * @brief [zh] 相邻窗口的横向步长；至少为 1。
     */
    uint32_t stride_w;
    /**
     * @brief [en] Padding on each side vertically; must be smaller than `kernel_h`.
     * @brief [zh] 上下各加的填充；必须小于 `kernel_h`。
     */
    uint32_t pad_h;
    /**
     * @brief [en] Padding on each side horizontally; must be smaller than `kernel_w`.
     * @brief [zh] 左右各加的填充；必须小于 `kernel_w`。
     */
    uint32_t pad_w;
    /**
     * @brief [en] Channel count; must equal the input's innermost extent.
     * @brief [zh] 通道数；必须等于输入最内轴长度。
     */
    uint32_t channels;
} infr_pool_params;

/**
 * @brief [en] Output shape of a pooling operator.
 * @brief [zh] 池化算子的输出形状。
 * @param params [en] Window geometry; must not be NULL.
 * @param params [zh] 窗口几何；不得为 NULL。
 * @param input_shape [en] Rank-4 NHWC input descriptor; must not be NULL.
 * @param input_shape [zh] 4 维 NHWC 输入描述符；不得为 NULL。
 * @param out_shape [en] Receives `[1, out_h, out_w, channels]`.
 * @param out_shape [zh] 接收 `[1, out_h, out_w, channels]`。
 * @return INFR_OK; INFR_ERR_INVALID_ARGUMENT for a NULL argument, rank other than 4,
 *         a zero stride or window, or a window larger than the padded input.
 * @since 0.1.0
 */
infr_status infr_max_pool2d_output_shape(const infr_pool_params *params, const infr_shape *input_shape,
                                         infr_shape *out_shape);

/**
 * @brief [en] fp32 max pooling over an NHWC tensor.
 * @brief [zh] 在 NHWC 张量上做 fp32 最大池化。
 * @param input [en] Rank-4 NHWC input, `[1, in_h, in_w, channels]`.
 * @param input [zh] 4 维 NHWC 输入，`[1, in_h, in_w, channels]`。
 * @param params [en] Window geometry; `channels` must match the input.
 * @param params [zh] 窗口几何；`channels` 必须与输入一致。
 * @param output [en] Rank-4 NHWC output whose shape must already equal `infr_max_pool2d_output_shape`; the data is overwritten.
 * @param output [zh] 4 维 NHWC 输出，其形状必须已等于 `infr_max_pool2d_output_shape`；数据会被覆盖。
 * @param workspace [en] Scratch source; rewound before returning, on every path.
 * @param workspace [zh] 临时空间来源；任何返回路径上都会先回退。
 * @return INFR_OK; INFR_ERR_INVALID_ARGUMENT for NULL arguments, a mismatched channel
 *         count, a stride of 0, or padding not smaller than the window;
 *         INFR_ERR_UNSUPPORTED for a dtype or quantisation outside the stated subset;
 *         INFR_ERR_BUFFER_TOO_SMALL when the workspace cannot hold `kernel_h*kernel_w`
 *         floats.
 * @note [en] The reduction is a call to the registered `max` kernel over a gathered
 *            window, so this operator has no comparison of its own.
 * @note [zh] 归约是对已注册 `max` kernel 的一次调用（作用在 gather 出来的窗口上），
 *            所以本算子自己没有任何比较运算。
 * @note [en] Padding contributes negative infinity, not zero: a zero pad would win the
 *            maximum on an all-negative window and silently turn max pooling into
 *            "max of the feature map and 0". Padding must also be smaller than the
 *            window, so no output position can be entirely padding.
 * @note [zh] 填充贡献的是负无穷而不是 0：填 0 会在全负窗口里赢下最大值，把最大池化
 *            悄悄变成"特征图与 0 取最大"。填充还必须小于窗口，这样任何输出位置都不会
 *            完全落在填充里。
 * @since 0.1.0
 */
infr_status infr_max_pool2d_f32(const infr_tensor *input, const infr_pool_params *params, infr_tensor *output,
                                infr_workspace *workspace);

/**
 * @brief [en] Global average pooling: mean over H and W, per channel.
 * @brief [zh] 全局平均池化：对 H、W 求均值，逐通道。
 * @param input [en] Rank-4 NHWC input, `[1, in_h, in_w, channels]`.
 * @param input [zh] 4 维 NHWC 输入，`[1, in_h, in_w, channels]`。
 * @param output [en] Rank-4 NHWC output that must already be `[1, 1, 1, channels]`.
 * @param output [zh] 4 维 NHWC 输出，必须已等于 `[1, 1, 1, channels]`。
 * @param workspace [en] Scratch source; rewound before returning, on every path.
 * @param workspace [zh] 临时空间来源；任何返回路径上都会先回退。
 * @return INFR_OK; INFR_ERR_INVALID_ARGUMENT for NULL arguments, a non-rank-4 tensor,
 *         a channel mismatch, or an output that is not `[1,1,1,C]`;
 *         INFR_ERR_UNSUPPORTED for a dtype or quantisation outside the stated subset;
 *         INFR_ERR_BUFFER_TOO_SMALL when the workspace cannot hold `in_h*in_w` floats.
 * @note [en] Composed as `sum` then `scale`, both from the registry: the mean is not a
 *            kernel, it is a reduction plus a multiply by `1/(H*W)`. The factor lives
 *            in a one-element buffer because the kernel contract has no scalar slot.
 * @note [zh] 由注册表里的 `sum` 与 `scale` 组合而成：均值不是 kernel，它是一次归约
 *            再乘 `1/(H*W)`。因子放在单元素缓冲区里，因为 kernel 契约没有标量位置。
 * @note [en] The division happens once per channel, not once per element: `sum` is
 *            exact in ascending order and the single multiply is what makes the result
 *            reproducible across targets.
 * @note [zh] 除法按通道只做一次、不是逐元素做：`sum` 按升序累加是确定的，而只乘一次
 *            正是跨目标可复现的原因。
 * @since 0.1.0
 */
infr_status infr_mean_hw_f32(const infr_tensor *input, infr_tensor *output, infr_workspace *workspace);

#ifdef __cplusplus
}
#endif
