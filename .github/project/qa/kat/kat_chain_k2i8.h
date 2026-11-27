// Conan::ImportStart
#pragma once
// Conan::ImportEnd

/*!
 * @brief [en] `k2_i8_conv` chain: the v3 blob through the C executor, on the target.
 * @brief [zh] `k2_i8_conv` 链：v3 blob 经 C 执行器在目标端跑通。
 * @note [en] The graph is one int8 convolution, which is the smallest thing that exercises every
 *            field v3 added: an int8 activation with a per-tensor scale, an int8 weight with
 *            per-channel scales, an int32 bias, and a non-zero output zero point.
 * @note [zh] 图里只有一个 int8 卷积——这是能把 v3 新增字段一次跑满的最小东西：逐张量 scale 的
 *            int8 激活、逐通道 scale 的 int8 权重、int32 偏置、非零输出零点。
 * @note [en] The expectation is **not** produced by this chain or by the host executor: it is the
 *            integer reference in `emit_k2_i8_blob.py`, computed before the blob existed. A target
 *            run that matched a host run of the same code would only prove the two agree; this
 *            agrees with a separate implementation of the written rule.
 * @note [zh] 期望**不是**本链或宿主执行器产出的，而是 `emit_k2_i8_blob.py` 里那份整数参考——
 *            它在 blob 存在之前就算好了。与"同一份代码的宿主运行"相符只能证明两者一致；这里
 *            相符的是**那份书面规则的另一份实现**。
 * @since 0.2.0
 */

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief [en] Buffers the chain fills; the checked tensor is the whole output.
 * @brief [zh] 链填写的缓冲区；被校验的张量就是整个输出。
 * @since 0.2.0
 */
typedef struct kat_k2i8_result {
    /**
     * @brief [en] The convolution's output bytes, one int8 per element.
     * @brief [zh] 卷积输出的字节，每元素一个 int8。
     */
    signed char out[27];
} kat_k2i8_result;

/**
 * @brief [en] Run the graph; the blob executor's status, not a boolean.
 * @brief [zh] 运行整张图；返回 blob 执行器的状态码而不是布尔值。
 * @since 0.2.0
 */
int kat_k2i8_run(kat_k2i8_result *result);

/**
 * @brief [en] Length in bytes of the checked tensor.
 * @brief [zh] 被校验张量的字节数。
 * @since 0.2.0
 */
unsigned kat_k2i8_expect_len(void);

/**
 * @brief [en] Copy the checked tensor as raw bytes, for a memcmp downstream.
 * @brief [zh] 把被校验张量按裸字节拷出，供下游 memcmp。
 * @since 0.2.0
 */
void kat_k2i8_expect(const kat_k2i8_result *result, unsigned char *out);

#ifdef __cplusplus
}
#endif
