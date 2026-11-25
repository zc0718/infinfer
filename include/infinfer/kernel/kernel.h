// Conan::ImportStart
#pragma once
#include <stddef.h>
#include <stdint.h>
// Conan::ImportEnd

// C linkage, because a C header has to survive being included from C++.
//
// The recipe's `_make_c_compatible()` wraps every installed `.h`, so a consumer
// of the *package* sees an `extern "C"` guard; a consumer of the *tree* (and
// every in-tree gate that builds) does not. The guard here makes both worlds
// agree; it nests inside the packager's own wrapper, which is legal.
//
// 一个 C 头文件必须能从 C++ 里被包含而不出问题：打包器会给装机头文件套
// `extern "C"`，但树内消费方看不到那层包裹。保护放在这里，两个世界就一致。
// 对 C 调用方没有任何变化：`__cplusplus` 未定义时这两行在预处理阶段就没了。
#ifdef __cplusplus
extern "C" {
#endif

/**
 * @file kernel.h
 *
 * @brief [en] Kernel contract: a kernel is a pure function over (pointer, length, dtype, quant);
 *            shape and stride belong to the graph layer, never to this interface.
 * @brief [zh] 内核契约：内核是只认（指针、长度、dtype、quant）的纯函数；
 *            shape 与 stride 属于图层的职责，不得出现在本接口里。
 * @since 0.1.0
 */

/**
 * @brief [en] Status codes; numeric values are stable across releases.
 * @brief [zh] 状态码；数值在各版本间保持稳定。
 * @since 0.1.0
 */
typedef enum infr_status {
    INFR_OK = 0,                   /**< [en] Success. [zh] 成功。 */
    INFR_ERR_INVALID_ARGUMENT = 1, /**< [en] A pointer, length or parameter is invalid. [zh] 指针、长度或参数非法。 */
    INFR_ERR_UNSUPPORTED =
        2, /**< [en] The requested combination is not declared in the matrix. [zh] 请求的组合未在矩阵中声明。 */
    INFR_ERR_BUFFER_TOO_SMALL = 3, /**< [en] The destination cannot hold the result. [zh] 目标缓冲区容不下结果。 */
    INFR_ERR_NOT_IMPLEMENTED =
        4 /**< [en] The contract exists but the implementation does not. [zh] 契约存在但实现缺失。 */
} infr_status;

/**
 * @brief [en] Element types a kernel may be instantiated for.
 * @brief [zh] 内核可被实例化的元素类型。
 * @note [en] The set is closed on purpose: the legal dtype x quant pairs live in
 *            `src/kernel/matrix.yml`, and the census gate rejects a pair declared
 *            there without a test case.
 * @note [zh] 该集合刻意封闭：合法的 dtype x quant 组合只声明在
 *            `src/kernel/matrix.yml`，census 门禁会拒绝没有用例的声明。
 * @since 0.1.0
 */
typedef enum infr_dtype {
    INFR_DTYPE_F32 = 1, /**< [en] IEEE-754 binary32. [zh] IEEE-754 binary32。 */
    INFR_DTYPE_I8 = 2,  /**< [en] Signed 8-bit integer. [zh] 有符号 8 位整数。 */
    INFR_DTYPE_U8 = 3,  /**< [en] Unsigned 8-bit integer. [zh] 无符号 8 位整数。 */
    INFR_DTYPE_I16 = 4, /**< [en] Signed 16-bit integer. [zh] 有符号 16 位整数。 */
    INFR_DTYPE_I32 = 5  /**< [en] Signed 32-bit integer. [zh] 有符号 32 位整数。 */
} infr_dtype;

/**
 * @brief [en] Quantization scheme a kernel is instantiated for.
 * @brief [zh] 内核被实例化时的量化方案。
 * @since 0.1.0
 */
typedef enum infr_quant_kind {
    INFR_QUANT_NONE = 0, /**< [en] Unquantized (float) arithmetic. [zh] 非量化（浮点）运算。 */
    INFR_QUANT_PER_TENSOR =
        1, /**< [en] One scale/zero point for the whole tensor. [zh] 整张量共用一组 scale/zero point。 */
    INFR_QUANT_PER_CHANNEL = 2 /**< [en] One scale/zero point per channel. [zh] 每个通道各一组 scale/zero point。 */
} infr_quant_kind;

/**
 * @brief [en] Resolved quantization parameters for one kernel invocation.
 * @brief [zh] 一次内核调用的已解析量化参数。
 * @note [en] `scale` and `zero_point` are what the kernel reads; the per-channel
 *            arrays are the declarations the graph layer resolves from and are
 *            never indexed by the kernel itself. Splitting a per-channel tensor
 *            into slices is the graph layer's job, which is why no shape appears
 *            in any kernel signature.
 * @note [zh] 内核只读 `scale` 与 `zero_point`；`channel_*` 数组是图层解析用的
 *            声明，内核永不索引。把逐通道张量切成片是图层的职责——这也正是
 *            内核签名里没有 shape 的原因。
 * @since 0.1.0
 */
typedef struct infr_quant {
    infr_quant_kind kind;        /**< [en] Declared scheme. [zh] 声明的量化方案。 */
    float scale;                 /**< [en] Resolved scale. [zh] 已解析的 scale。 */
    int32_t zero_point;          /**< [en] Resolved zero point. [zh] 已解析的 zero point。 */
    const float *channel_scales; /**< [en] Per-channel scales, or NULL. [zh] 逐通道 scale，可为 NULL。 */
    const int32_t *
        channel_zero_points; /**< [en] Per-channel zeros, or NULL to reuse `zero_point`. [zh] 逐通道零点；NULL 表示复用 `zero_point`。 */
    uint32_t channel_count; /**< [en] Declared channel count; 1 for per-tensor. [zh] 声明的通道数；逐张量时为 1。 */
    uint32_t axis; /**< [en] Declared quantized axis; ignored for per-tensor. [zh] 声明的量化轴；逐张量时忽略。 */
} infr_quant;

/**
 * @brief [en] Implementation tier a kernel belongs to (matrix.yml `tiers` keys).
 * @brief [zh] 内核所属的实现档位（对应 matrix.yml 的 `tiers` 键）。
 * @note [en] INV-2: `scalar` is the only definition of correctness; every other
 *            tier is proven equivalent by the parity harness, never by review.
 * @note [zh] INV-2：只有 `scalar` 是正确性的定义；其它档位一律由 parity
 *            骨架证明等价，不靠人工评审。
 * @since 0.1.0
 */
typedef enum infr_tier {
    INFR_TIER_SCALAR = 0,   /**< [en] Reference implementation. [zh] 参考实现。 */
    INFR_TIER_AUTOVEC = 1,  /**< [en] Same C, left to the auto-vectorizer. [zh] 同源 C，交给自动向量化。 */
    INFR_TIER_X86 = 2,      /**< [en] x86 SSE2/AVX2 intrinsics. [zh] x86 SSE2/AVX2 intrinsics。 */
    INFR_TIER_ARM_NEON = 3, /**< [en] AArch32/AArch64 NEON. [zh] AArch32/AArch64 NEON。 */
    INFR_TIER_ARM_M = 4,    /**< [en] Cortex-M DSP / MVE. [zh] Cortex-M DSP / MVE。 */
    INFR_TIER_RVV = 5,      /**< [en] RISC-V Vector 1.0. [zh] RISC-V Vector 1.0。 */
    INFR_TIER_VENDOR = 6 /**< [en] Optional vendor backend (CMSIS-NN and peers). [zh] 可选的厂商后端（CMSIS-NN 等）。 */
} infr_tier;

/**
 * @brief [en] Fused-activation codes.
 * @brief [zh] 融合激活码。
 * @note [en] 0..5 mirror TFLite `ActivationFunctionType`; 6 is this project's
 *            extension for the standalone hard-swish op mapped into the fused
 *            domain. The numeric values are contract: the reference defect this
 *            header exists to prevent was a table whose rows had drifted away
 *            from the codes they claimed to name.
 * @note [zh] 0..5 与 TFLite `ActivationFunctionType` 对齐；6 是本工程把独立
 *            hard-swish 算子映射进融合域时的扩展。数值即契约：本头文件要防的
 *            正是"表格行与它声称的码值漂移"这一类缺陷。
 * @since 0.1.0
 */
typedef enum infr_activation {
    INFR_ACT_NONE = 0,         /**< [en] Identity. [zh] 恒等。 */
    INFR_ACT_RELU = 1,         /**< [en] max(x, 0). [zh] max(x, 0)。 */
    INFR_ACT_RELU_N1_TO_1 = 2, /**< [en] Clip to [-1, 1]. [zh] 截断到 [-1, 1]。 */
    INFR_ACT_RELU6 = 3,        /**< [en] Clip to [0, 6]. [zh] 截断到 [0, 6]。 */
    INFR_ACT_TANH = 4,         /**< [en] Hyperbolic tangent. [zh] 双曲正切。 */
    INFR_ACT_SIGN_BIT = 5,     /**< [en] Sign bit as float (TFLite sign-bit). [zh] 符号位转浮点（TFLite sign-bit）。 */
    INFR_ACT_HARD_SWISH = 6,   /**< [en] x * relu6(x + 3) / 6. [zh] x * relu6(x + 3) / 6。 */
    INFR_ACT_COUNT = 7         /**< [en] Number of codes (not an activation). [zh] 码值个数（不是激活）。 */
} infr_activation;

#if defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 201112L)
_Static_assert(INFR_ACT_COUNT == 7, "activation codes are contract; extend the enum and the switch together");
#endif

/**
 * @brief [en] Elementwise float32 activation over n elements (in and out may alias).
 * @brief [zh] 对 n 个元素做逐元素 float32 激活（输入输出可同一缓冲区）。
 * @since 0.1.0
 */
typedef void (*infr_act_f32_fn)(const float *in, float *out, uint32_t n);

/**
 * @brief [en] Look up the float32 implementation of a fused activation.
 * @brief [zh] 查询某个融合激活的 float32 实现。
 * @param [in] act Activation code.
 * @return [en] A function pointer, or NULL when the tier has no implementation yet.
 * @return [zh] 函数指针；该档位尚无实现时返回 NULL。
 * @note [en] A NULL return is a fact the caller must handle, not an error to
 *            paper over: dispatching a fused activation to the wrong row is the
 *            defect this lookup exists to make testable.
 * @note [zh] 返回 NULL 是调用方必须处理的事实，不是可以掩盖的错误：把融合激活
 *            派发到错误的行，正是本查询要让其可被测试发现的缺陷。
 * @since 0.1.0
 */
infr_act_f32_fn infr_activation_f32(infr_activation act);

/**
 * @brief [en] Validate and convert a raw activation code to the enum.
 * @brief [zh] 校验原始激活码并转换为枚举值。
 * @param [in] code Raw code from a model blob.
 * @param [out] out Receives the enum value on success; untouched on failure.
 * @return INFR_OK, INFR_ERR_INVALID_ARGUMENT for a NULL `out`, or
 *         INFR_ERR_UNSUPPORTED for a code outside `infr_activation`.
 * @since 0.1.0
 */
infr_status infr_activation_from_code(uint32_t code, infr_activation *out);

/**
 * @brief [en] Quantize n floats to int8 using resolved parameters.
 * @brief [zh] 用已解析参数把 n 个 float 量化为 int8。
 * @param [in] in Source floats.
 * @param [out] out Destination int8 buffer.
 * @param [in] n Element count.
 * @param [in] quant Resolved parameters; `kind` must not be INFR_QUANT_NONE.
 * @return INFR_OK on success; INFR_ERR_* otherwise.
 * @since 0.1.0
 */
infr_status infr_quantize_i8(const float *in, int8_t *out, uint32_t n, const infr_quant *quant);

/**
 * @brief [en] Dequantize n int8 values back to float using resolved parameters.
 * @brief [zh] 用已解析参数把 n 个 int8 反量化为 float。
 * @param [in] in Source int8 values.
 * @param [out] out Destination floats.
 * @param [in] n Element count.
 * @param [in] quant Resolved parameters; `kind` must not be INFR_QUANT_NONE.
 * @return INFR_OK on success; INFR_ERR_* otherwise.
 * @since 0.1.0
 */
infr_status infr_dequantize_i8(const int8_t *in, float *out, uint32_t n, const infr_quant *quant);

/**
 * @brief [en] A kernel body: arrays of input/output pointers plus an element count.
 * @brief [zh] 内核主体：输入/输出指针数组加元素个数。
 * @note [en] No shape, no stride, no batch: layout is the graph layer's
 *            responsibility (INV-3). A kernel that needs a channel index gets it
 *            by being called once per channel slice.
 * @note [zh] 没有 shape、没有 stride、没有 batch：布局是图层的职责（INV-3）。
 *            需要通道下标的内核由图层按通道切片多次调用。
 * @since 0.1.0
 */
typedef infr_status (*infr_kernel_fn)(const void *const *inputs, void *const *outputs, uint32_t element_count,
                                      const infr_quant *quant);

/**
 * @brief [en] One row of the kernel registry: name x dtype x quant x tier -> body.
 * @brief [zh] 内核注册表的一行：名字 x dtype x quant x tier -> 实现。
 * @note [en] `name` must match a kernel key in `src/kernel/matrix.yml`; the
 *            census gate cross-checks the two and fails on either side's orphans.
 * @note [zh] `name` 必须与 `src/kernel/matrix.yml` 的 kernel 键一致；census
 *            门禁双向核对，任一侧出现孤儿行都会报红。
 * @since 0.1.0
 */
typedef struct infr_kernel_desc {
    const char *name;      /**< [en] Matrix kernel key, e.g. "conv2d". [zh] 矩阵中的 kernel 键，如 "conv2d"。 */
    infr_dtype dtype;      /**< [en] Element type. [zh] 元素类型。 */
    infr_quant_kind quant; /**< [en] Quantization scheme. [zh] 量化方案。 */
    infr_tier tier;        /**< [en] Implementation tier. [zh] 实现档位。 */
    infr_kernel_fn fn;     /**< [en] Kernel body; must not be NULL. [zh] 内核主体，不得为 NULL。 */
} infr_kernel_desc;

/**
 * @brief [en] Register one kernel row.
 * @brief [zh] 注册一行内核。
 * @param [in] kernel Descriptor; the registry copies it, the caller keeps ownership of `name`.
 * @return INFR_OK, INFR_ERR_INVALID_ARGUMENT, or INFR_ERR_NOT_IMPLEMENTED until the
 *         registry lands (PLAN P2.1).
 * @since 0.1.0
 */
infr_status infr_kernel_register(const infr_kernel_desc *kernel);

/**
 * @brief [en] Find a registered kernel row.
 * @brief [zh] 查找已注册的内核行。
 * @param [in] name Matrix kernel key.
 * @param [in] dtype Element type.
 * @param [in] quant Quantization scheme.
 * @param [in] tier Implementation tier.
 * @return [en] The row, or NULL when absent; a NULL body is never registered.
 * @return [zh] 命中行；不存在时为 NULL（NULL 实现永不入库）。
 * @since 0.1.0
 */
const infr_kernel_desc *infr_kernel_find(const char *name, infr_dtype dtype, infr_quant_kind quant, infr_tier tier);

/**
 * @brief [en] Stable, non-NULL name of a status code (safe in log format strings).
 * @brief [zh] 状态码的稳定非空名称（可安全用于日志格式串）。
 * @param [in] status Status code.
 * @return [en] A static string; unknown values map to "INFR_STATUS_UNKNOWN".
 * @return [zh] 静态字符串；未知值映射为 "INFR_STATUS_UNKNOWN"。
 * @since 0.1.0
 */
const char *infr_status_str(infr_status status);

#ifdef __cplusplus
}
#endif
