// Conan::ImportStart
#pragma once
// Conan::ImportEnd

/*!
 * @brief [en] `k2_conv_micro` chain: run the graph through the public C API.
 * @brief [zh] `k2_conv_micro` 链：通过公开 C API 跑通整张图。
 * @note [en] Same graph as the host KAT (`graph_kat_k2_conv_micro.cpp`): conv, relu,
 *            max pool, depthwise conv, relu, global mean, fully connected.
 * @note [zh] 与宿主 KAT（`graph_kat_k2_conv_micro.cpp`）同一张图：卷积、relu、最大
 *            池化、逐通道卷积、relu、全局均值、全连接。
 * @since 0.1.0
 */

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief [en] Buffers the chain fills; only the checked tensor is compared.
 * @brief [zh] 链填写的缓冲区；只有被校验的那个张量参与比对。
 * @since 0.1.0
 */
typedef struct kat_k2_result {
    /**
     * @brief [en] Final 10 logits, the tensor the check compares.
     * @brief [zh] 最终 10 个输出，即校验比对的张量。
     */
    float out[10];
} kat_k2_result;

/**
 * @brief [en] Run the graph; INFR_OK when every operator returned INFR_OK.
 * @brief [zh] 运行整张图；每个算子都返回 INFR_OK 时返回 INFR_OK。
 * @since 0.1.0
 */
int kat_k2_run(kat_k2_result *result);

/**
 * @brief [en] Length in bytes of the checked tensor.
 * @brief [zh] 被校验张量的字节数。
 * @since 0.1.0
 */
unsigned kat_k2_expect_len(void);

/**
 * @brief [en] Copy the checked tensor as raw bytes, for a memcmp downstream.
 * @brief [zh] 把被校验张量按裸字节拷出，供下游 memcmp。
 * @since 0.1.0
 */
void kat_k2_expect(const kat_k2_result *result, unsigned char *out);

#ifdef __cplusplus
}
#endif
