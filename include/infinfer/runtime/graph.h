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
 *         (`max(kernel_h * kernel_w * in_channels/groups, out_channels * kernel_h * kernel_w) +
 *         out_channels` floats).
 * @note [en] The inner loop is a call to the registered `fc` (dot product) and the
 *            bias step is a call to the registered `add`, so a faster tier of those
 *            two kernels speeds this operator up without touching this file. The
 *            gather is the only shape-dependent work, and it is scalar here.
 * @note [en] A 3x3 window with one input channel per group takes the registered
 *            `dwconv3x3` instead: one call for a whole output pixel rather than one call
 *            per output channel, because a call cost more than the nine multiplications
 *            it performed (measured, gaps G-129). That path gathers one window per
 *            output channel, which is why the scratch requirement above is the larger of
 *            the two shapes. When the cell is absent the operator falls back to `fc`.
 * @note [zh] 内层循环是一次对已注册 `fc`（点积）的调用，偏置步是一次对已注册 `add`
 *            的调用——所以那两个 kernel 换了更快的档位，本算子自动变快，无需改这里。
 *            只有 gather 是依赖 shape 的活，它在本档是标量实现。
 * @note [zh] 每分组一个输入通道且窗口为 3×3 时，改走已注册的 `dwconv3x3`：一个输出像素
 *            一次调用，而不是每个输出通道一次——因为一次调用的成本高于它执行的九次乘法
 *            （实测，gaps G-129）。那条路径为**每个输出通道** gather 一个窗口，所以上面的
 *            scratch 需求取两种形状里较大的那个。该格缺席时本算子退回 `fc`。
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
 *            `kernel_h * kernel_w * in_channels/groups` **bytes** for the window plus
 *            `out_channels * 4` bytes for the pixel's accumulators. The second buffer exists so the
 *            requantisation can be one call over a whole pixel instead of one call per channel: a
 *            call costs more than the arithmetic it performs (measured, gaps G-121).
 * @param workspace [zh] 临时空间来源；任何返回路径上都会先回退。需要
 *            `kernel_h * kernel_w * in_channels/groups` 个**字节**装窗口，另加
 *            `out_channels * 4` 字节装一个像素的累加器。第二块缓冲区是为了让再量化能按整像素调用一次、
 *            而不是每通道一次：一次调用的成本高于它执行的算术（实测，gaps G-121）。
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

/**
 * @brief [en] Which axis is being concatenated, and how many inputs there are.
 * @brief [zh] 沿哪条轴拼接，以及有多少个输入。
 * @since 0.2.0
 */
typedef struct infr_concat_params {
    /**
     * @brief [en] Zero-based axis index; every input must agree on all the others.
     * @brief [zh] 从 0 开始的轴序号；所有输入必须在其它的轴上完全一致。
     */
    uint32_t axis;
    /**
     * @brief [en] Number of entries in the `inputs` array; at least 1.
     * @brief [zh] `inputs` 数组的项数；至少为 1。
     */
    uint32_t input_count;
} infr_concat_params;

/**
 * @brief [en] Shape of the concatenation of `inputs` along `params->axis`.
 * @brief [zh] 各输入沿 `params->axis` 拼接后的形状。
 * @param inputs [en] At least `params->input_count` non-NULL tensors.
 * @param inputs [zh] 至少 `params->input_count` 个非空张量。
 * @param params [en] Axis and input count.
 * @param params [zh] 轴序号与输入数。
 * @param out_shape [en] Receives the result; untouched on failure.
 * @param out_shape [zh] 接收结果；失败时不改动。
 * @return INFR_OK; INFR_ERR_INVALID_ARGUMENT for NULL arguments, a zero input count,
 *         an out-of-range axis, or inputs that disagree off the concatenated axis.
 * @since 0.2.0
 */
infr_status infr_concat_output_shape(const infr_tensor *const *inputs, const infr_concat_params *params,
                                     infr_shape *out_shape);

/**
 * @brief [en] fp32 concatenation of `params->input_count` tensors along one axis.
 * @brief [zh] 把 `params->input_count` 个张量沿一条轴做 fp32 拼接。
 * @param inputs [en] At least `params->input_count` non-NULL fp32 tensors.
 * @param inputs [zh] 至少 `params->input_count` 个非空 fp32 张量。
 * @param params [en] Axis and input count.
 * @param params [zh] 轴序号与输入数。
 * @param output [en] Rank-4-compatible output whose shape must already equal
 *               `infr_concat_output_shape`; the data is overwritten.
 * @param output [zh] 形状必须已等于 `infr_concat_output_shape` 的输出；数据会被覆盖。
 * @param workspace [en] Required to be non-NULL for uniformity with its siblings; concat
 *                  allocates nothing and never touches it.
 * @param workspace [zh] 为与同层算子保持一致而要求非空；拼接不做任何分配，也不会碰它。
 * @return INFR_OK; INFR_ERR_INVALID_ARGUMENT for NULL arguments or inconsistent shapes;
 *         INFR_ERR_UNSUPPORTED for a non-fp32 input or a single run wider than `uint32_t`;
 *         INFR_ERR_NOT_IMPLEMENTED when the `copy` primitive is not registered.
 * @note [en] There is no arithmetic here: each input is a run of `copy` calls into a
 *            strided window of the output, one run per position of the outer axes.
 * @note [zh] 这里没有任何算术：每个输入是若干次 `copy` 调用，写进输出里一个带步长的
 *            窗口，每个外层轴位置一段。
 * @note [en] `input_count == 1` is legal and is the identity, which is what a graph
 *            importer produces after folding a single-input concat -- it is not a special
 *            case in the code, it falls out of the same loop.
 * @note [zh] `input_count == 1` 合法且就是恒等，正是图导入器折掉单输入拼接后得到的东西
 *            ——它在代码里不是特例，而是同一个循环自然给出的结果。
 * @since 0.2.0
 */
infr_status infr_concat_f32(const infr_tensor *const *inputs, const infr_concat_params *params, infr_tensor *output,
                            infr_workspace *workspace);

/**
 * @brief [en] Which input axis becomes which output axis.
 * @brief [zh] 哪条输入轴变成哪条输出轴。
 * @since 0.2.0
 */
typedef struct infr_transpose_params {
    /**
     * @brief [en] Number of axes; 0 means "no permutation given", and the axes are reversed.
     * @brief [zh] 轴数；0 表示"没给置换"，各轴反转。
     */
    uint32_t rank;
    /**
     * @brief [en] `perm[a]` is the input axis that becomes output axis `a`, as in ONNX
     *            `Transpose`; entries above `rank` are ignored.
     * @brief [zh] `perm[a]` 是变成输出轴 `a` 的输入轴，与 ONNX `Transpose` 一致；
     *            `rank` 以上的项被忽略。
     */
    uint32_t perm[INFR_TENSOR_MAX_RANK];
} infr_transpose_params;

/**
 * @brief [en] Shape produced by transposing `input_shape` with `params`.
 * @brief [zh] 用 `params` 对 `input_shape` 转置后得到的形状。
 * @param input_shape [en] Rank 1..`INFR_TENSOR_MAX_RANK`.
 * @param input_shape [zh] 秩为 1..`INFR_TENSOR_MAX_RANK`。
 * @param params [en] Permutation; `params->rank` must be 0 or equal to `input_shape->rank`.
 * @param params [zh] 置换；`params->rank` 必须是 0 或等于 `input_shape->rank`。
 * @param out_shape [en] Receives the result; untouched on failure.
 * @param out_shape [zh] 接收结果；失败时不改动。
 * @return INFR_OK; INFR_ERR_INVALID_ARGUMENT for NULL arguments, a rank mismatch, or a
 *         `perm` that is not a permutation (an entry out of range or appearing twice).
 * @note [en] A repeated or out-of-range entry is refused rather than clamped: clamping would
 *            produce a legal-looking transpose of the wrong thing.
 * @note [zh] 重复或越界的项被拒绝而不是夹取：夹取会产出一个"看起来合法"的错转置。
 * @since 0.2.0
 */
infr_status infr_transpose_output_shape(const infr_shape *input_shape, const infr_transpose_params *params,
                                        infr_shape *out_shape);

/**
 * @brief [en] fp32 transpose: permute the axes, moving values with the `copy` kernel.
 * @brief [zh] fp32 转置：置换各轴，用 `copy` 内核搬值。
 * @param input [en] fp32 tensor of rank 1..`INFR_TENSOR_MAX_RANK`.
 * @param input [zh] 秩为 1..`INFR_TENSOR_MAX_RANK` 的 fp32 张量。
 * @param params [en] Permutation; `params->rank` must be 0 or equal to the input rank.
 * @param params [zh] 置换；`params->rank` 必须是 0 或等于输入的秩。
 * @param output [en] fp32 output whose shape must already equal `infr_transpose_output_shape`;
 *               the data is overwritten.
 * @param output [zh] 形状必须已等于 `infr_transpose_output_shape` 的 fp32 输出；数据会被覆盖。
 * @param workspace [en] Scratch source. Needed only when no contiguous run exists in output
 *                  order (the NHWC to NCHW case), where it must hold `output->shape.count`
 *                  floats; rewound before returning, on every path.
 * @param workspace [zh] 临时空间来源。只有在输出顺序里不存在连续段时（NHWC 转 NCHW 那种）
 *                  才用到，此时需能放下 `output->shape.count` 个 float；任何返回路径上都会
 *                  先回退。
 * @return INFR_OK; INFR_ERR_INVALID_ARGUMENT for NULL arguments, a rank or shape mismatch;
 *         INFR_ERR_UNSUPPORTED for a non-fp32 tensor or more than `UINT32_MAX` elements;
 *         INFR_ERR_BUFFER_TOO_SMALL when the workspace cannot hold the gathered output;
 *         INFR_ERR_NOT_IMPLEMENTED when `copy` is not registered.
 * @note [en] Three paths, chosen by what can be a contiguous copy: identity is one call; a
 *            permutation that keeps the last axis in place is one call per position of the
 *            outer axes; everything else gathers into the workspace and issues one call. So
 *            the kernel-call cost is never above one per output element, which is the line the
 *            call-count metric draws.
 * @note [zh] 三条路径，按"什么能成为连续拷贝"来选：恒等是一次调用；保持末轴不动的置换是外层
 *            轴每个位置一次调用；其余情况 gather 进工作区后发一次调用。所以内核调用代价永远
 *            不超过每输出元素一次，也就是调用计数指标画的那条线。
 * @note [en] `perm[0]` is the input axis that becomes output axis 0. A `params->rank` of 0
 *            means the front-end omitted `perm`, and the axes are reversed, which is the ONNX
 *            default.
 * @note [zh] `perm[0]` 是变成输出轴 0 的输入轴。`params->rank` 为 0 表示前端省略了
 *            `perm`，此时各轴反转，这是 ONNX 的默认行为。
 * @since 0.2.0
 */
infr_status infr_transpose_f32(const infr_tensor *input, const infr_transpose_params *params, infr_tensor *output,
                               infr_workspace *workspace);

/**
 * @brief [en] Which axes to cut, and where.
 * @brief [zh] 切哪几条轴，切在哪里。
 * @since 0.2.0
 */
typedef struct infr_slice_params {
    /**
     * @brief [en] Number of entries used in the arrays below; 0 is rejected.
     * @brief [zh] 下面各数组实际使用的项数；0 被拒绝。
     */
    uint32_t axis_count;
    /**
     * @brief [en] Axis each entry applies to; an axis may appear only once.
     * @brief [zh] 每项作用的轴；同一条轴只能出现一次。
     */
    uint32_t axis[INFR_TENSOR_MAX_RANK];
    /**
     * @brief [en] Start index, inclusive; negative counts from the end.
     * @brief [zh] 起始下标（含）；负数表示从末尾数。
     */
    int64_t start[INFR_TENSOR_MAX_RANK];
    /**
     * @brief [en] End index, exclusive; negative counts from the end, and `INT64_MAX`
     *            means to the end of that axis.
     * @brief [zh] 结束下标（不含）；负数表示从末尾数，`INT64_MAX` 表示到该轴末尾。
     */
    int64_t end[INFR_TENSOR_MAX_RANK];
    /**
     * @brief [en] Step along that axis; must be at least 1.
     * @brief [zh] 沿该轴步长；至少为 1。
     */
    int64_t step[INFR_TENSOR_MAX_RANK];
} infr_slice_params;

/**
 * @brief [en] Shape of the slice `params` cuts out of `input_shape`.
 * @brief [zh] `params` 从 `input_shape` 里切出的形状。
 * @param input_shape [en] Rank 1..`INFR_TENSOR_MAX_RANK`.
 * @param input_shape [zh] 秩为 1..`INFR_TENSOR_MAX_RANK`。
 * @param params [en] Axes, start, end and step; axes not listed are kept whole.
 * @param params [zh] 轴、起止与步长；未列出的轴整条保留。
 * @param out_shape [en] Receives the result; untouched on failure.
 * @param out_shape [zh] 接收结果；失败时不改动。
 * @return INFR_OK; INFR_ERR_INVALID_ARGUMENT for NULL arguments, an axis out of range or
 *         repeated, or a step below 1.
 * @note [en] Start and end are clamped to the axis, so `INT64_MAX` and a negative `-dim - 5`
 *            are both legal and mean the obvious thing. A start at or past the end yields a
 *            zero-length axis, which is a legal result, not an error.
 * @note [zh] 起止被夹到轴长之内，所以 `INT64_MAX` 和 `-dim - 5` 都合法且含义如常。起点等于或
 *            超过终点会得到长度 0 的轴，那是合法结果而不是错误。
 * @since 0.2.0
 */
infr_status infr_slice_output_shape(const infr_shape *input_shape, const infr_slice_params *params,
                                    infr_shape *out_shape);

/**
 * @brief [en] fp32 slice: cut a strided view out of a tensor, moving values with `copy`.
 * @brief [zh] fp32 切片：从张量里切出一个带步长的视图，用 `copy` 搬值。
 * @param input [en] fp32 tensor of rank 1..`INFR_TENSOR_MAX_RANK`.
 * @param input [zh] 秩为 1..`INFR_TENSOR_MAX_RANK` 的 fp32 张量。
 * @param params [en] Axes, start, end and step.
 * @param params [zh] 轴、起止与步长。
 * @param output [en] fp32 output whose shape must already equal `infr_slice_output_shape`;
 *               the data is overwritten.
 * @param output [zh] 形状必须已等于 `infr_slice_output_shape` 的 fp32 输出；数据会被覆盖。
 * @param workspace [en] Scratch source, used only when the innermost axis is strided, where it
 *                  must hold `output->shape.elem_count` floats; rewound on every path.
 * @param workspace [zh] 临时空间来源，仅在末轴带步长时使用，此时需能放下
 *                  `output->shape.elem_count` 个 float；任何路径都会回退。
 * @return INFR_OK; INFR_ERR_INVALID_ARGUMENT for NULL arguments, an axis out of range or
 *         repeated, or a shape mismatch; INFR_ERR_UNSUPPORTED for a non-fp32 tensor, a step
 *         below 1 (a reversal), or more than `UINT32_MAX` elements;
 *         INFR_ERR_BUFFER_TOO_SMALL when the workspace cannot hold the gathered output;
 *         INFR_ERR_NOT_IMPLEMENTED when `copy` is not registered.
 * @note [en] Two paths, the same rule as `transpose`: when the innermost axis has step 1 every
 *            output run is contiguous and costs one `copy` per position of the outer axes;
 *            otherwise the values are gathered and one `copy` is issued. So the kernel-call
 *            cost is never above one per output element.
 * @note [zh] 两条路径，规则与 `transpose` 相同：末轴步长为 1 时每条输出段都连续，外层每个
 *            位置一次 `copy`；否则 gather 后发一次 `copy`。所以内核调用代价永不超过每输出
 *            元素一次。
 * @note [en] A step below 1 is refused rather than supported: a reversal is another operator's
 *            job (the models this library targets do not emit it), and quietly accepting it
 *            would make the contiguous-run test wrong by omission.
 * @note [zh] 步长小于 1 是拒绝而不是支持：反向切片是另一个算子的职责（本库面向的模型也不生成
 *            它），而默默接受会让「段是否连续」的判断因为漏掉这一种而失准。
 * @since 0.2.0
 */
infr_status infr_slice_f32(const infr_tensor *input, const infr_slice_params *params, infr_tensor *output,
                           infr_workspace *workspace);

/**
 * @brief [en] How a resized axis chooses the input element it copies from.
 * @brief [zh] 缩放后的轴从哪个输入元素取值。
 * @since 0.2.0
 */
typedef enum infr_resize_mode {
    /**
     * @brief [en] Nearest neighbour, `src = (dst * in) / out` in integer arithmetic -- the
     *            asymmetric mapping ONNX and TFLite exporters emit, and the default upsample of
     *            the detection models this library targets.
     * @brief [zh] 最近邻，`src = (dst * in) / out` 全程整数——即 ONNX 与 TFLite 导出器产生的
     *            asymmetric 映射，也是本库面向的检测模型默认的上采样方式。
     */
    INFR_RESIZE_NEAREST = 0
} infr_resize_mode;

/**
 * @brief [en] Target shape and interpolation rule of a resize.
 * @brief [zh] 缩放的目标形状与插值规则。
 * @since 0.2.0
 */
typedef struct infr_resize_params {
    /**
     * @brief [en] Rank shared by input and output; 1..`INFR_TENSOR_MAX_RANK`.
     * @brief [zh] 输入与输出共同的秩；1..`INFR_TENSOR_MAX_RANK`。
     */
    uint32_t rank;
    /**
     * @brief [en] Extent of each output axis, in element order; every entry must be at least 1.
     * @brief [zh] 每条输出轴的长度，按元素顺序排列；每项至少为 1。
     */
    int64_t out_dims[INFR_TENSOR_MAX_RANK];
    /**
     * @brief [en] Interpolation rule.
     * @brief [zh] 插值规则。
     */
    infr_resize_mode mode;
} infr_resize_params;

/**
 * @brief [en] Shape `params` resizes `input_shape` to.
 * @brief [zh] `params` 把 `input_shape` 缩放成的形状。
 * @param input_shape [en] Rank 1..`INFR_TENSOR_MAX_RANK`, every extent at least 1.
 * @param input_shape [zh] 秩为 1..`INFR_TENSOR_MAX_RANK`，每维长度至少为 1。
 * @param params [en] Target extents and the interpolation rule.
 * @param params [zh] 目标长度与插值规则。
 * @param out_shape [en] Receives the result; untouched on failure.
 * @param out_shape [zh] 接收结果；失败时不改动。
 * @return INFR_OK; INFR_ERR_INVALID_ARGUMENT for NULL arguments, a rank of 0, a rank that
 *         differs from the input's, or an extent below 1; INFR_ERR_UNSUPPORTED for a mode this
 *         build does not implement.
 * @since 0.2.0
 */
infr_status infr_resize_output_shape(const infr_shape *input_shape, const infr_resize_params *params,
                                     infr_shape *out_shape);

/**
 * @brief [en] fp32 nearest resize: the input walked in output order, one `copy` per run.
 * @brief [zh] fp32 最近邻缩放：按输出顺序走输入，每一段一次 `copy`。
 * @param input [en] fp32 tensor of rank 1..`INFR_TENSOR_MAX_RANK`.
 * @param input [zh] 秩为 1..`INFR_TENSOR_MAX_RANK` 的 fp32 张量。
 * @param params [en] Target extents and the interpolation rule.
 * @param params [zh] 目标长度与插值规则。
 * @param output [en] fp32 output whose shape must already equal `infr_resize_output_shape`; the
 *               data is overwritten.
 * @param output [zh] 形状必须已等于 `infr_resize_output_shape` 的 fp32 输出；数据会被覆盖。
 * @return INFR_OK; INFR_ERR_INVALID_ARGUMENT for NULL arguments, a rank or shape mismatch, or an
 *         extent below 1; INFR_ERR_UNSUPPORTED for a non-fp32 tensor, an unimplemented mode, or
 *         more than `UINT32_MAX` outputs; INFR_ERR_NOT_IMPLEMENTED when `copy` is not registered.
 * @note [en] No workspace: the integer mapping is monotone, so the output is written straight
 *            from the input in order. An integer upscale of `s` costs one call per `s` outputs;
 *            a downscale costs one call per output, and that is stated rather than optimised
 *            away, because those outputs really do read non-adjacent inputs.
 * @note [zh] 不需要工作区：整数映射单调，输出按顺序直接从输入写出。整数上采样 `s` 倍时
 *            每 `s` 个输出一次调用；降采样每输出一次，这一点写出来而不是「优化掉」，
 *            因为那些输出读的确实是不同邻的输入。
 * @note [en] Bilinear is a separate row, not a flag here: it reads four neighbours per output
 *            and needs arithmetic over a window, which is a different kernel set.
 * @note [zh] 双线性是单独一行，而不是这里的一个开关：它每个输出读四个邻点并做带权求和，
 *            需要的是另一套内核。
 * @since 0.2.0
 */
infr_status infr_resize_f32(const infr_tensor *input, const infr_resize_params *params, infr_tensor *output);

/**
 * @brief [en] Which axis a softmax normalises over.
 * @brief [zh] softmax 在哪条轴上归一化。
 * @since 0.2.0
 */
typedef struct infr_softmax_params {
    /**
     * @brief [en] Axis index in `0..rank-1`; the shape is not changed by this operator.
     * @brief [zh] 轴下标，取值 `0..rank-1`；本算子不改变形状。
     */
    uint32_t axis;
} infr_softmax_params;

/**
 * @brief [en] fp32 softmax over one axis: subtract the slice max, `exp`, `sum`, `scale`.
 * @brief [zh] 对一条轴做 fp32 softmax：减切片最大值、`exp`、`sum`、`scale`。
 * @param input [en] fp32 tensor of rank 1..`INFR_TENSOR_MAX_RANK`.
 * @param input [zh] 秩为 1..`INFR_TENSOR_MAX_RANK` 的 fp32 张量。
 * @param params [en] The axis to normalise over.
 * @param params [zh] 被归一化的那条轴。
 * @param output [en] fp32 output with the input's shape; the data is overwritten.
 * @param output [zh] 形状与输入相同的 fp32 输出；数据会被覆盖。
 * @param workspace [en] Scratch source; must hold `dim(axis) + 2` floats.
 * @param workspace [zh] 临时空间来源；需能放下 `dim(axis) + 2` 个 float。
 * @return INFR_OK; INFR_ERR_INVALID_ARGUMENT for NULL arguments, an axis out of range or a shape
 *         mismatch; INFR_ERR_UNSUPPORTED for a non-fp32 tensor; INFR_ERR_BUFFER_TOO_SMALL when the
 *         workspace cannot hold one slice plus two scalars; INFR_ERR_NOT_IMPLEMENTED when a
 *         required kernel is missing.
 * @note [en] The max is subtracted before `exp`, so logits above ~88 still produce a finite
 *            result; without it `expf` overflows and the division yields NaN. There is no shape
 *            function because the operator never changes the shape.
 * @note [zh] 先减去最大值再 `exp`，所以 logit 超过约 88 时结果仍然有限；不减的话 `expf` 溢出、
 *            除法得到 NaN。没有 shape 函数，因为本算子从不改变形状。
 * @since 0.2.0
 */
infr_status infr_softmax_f32(const infr_tensor *input, const infr_softmax_params *params, infr_tensor *output,
                             infr_workspace *workspace);

#ifdef __cplusplus
}
#endif
