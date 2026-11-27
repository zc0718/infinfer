// Conan::ImportStart
#pragma once

#include "generated/kat_k2m_blob.h"
// Conan::ImportEnd

/*!
 * @brief [en] `k2m_mobilenet`: a whole MobileNetV3-Small graph, from the blob, on the target.
 * @brief [zh] `k2m_mobilenet`：整张 MobileNetV3-Small，从 blob 出发，在目标端跑。
 * @note [en] k2 proves the *executor* with a six-node graph. This chain proves it with the
 *            graph it was built for: 80 nodes, 188 slots, 108 constants, 1.4 MB of weights
 *            that never leave flash. The operators are the same ones -- what changes is
 *            that nobody wrote this network down in C, and that the arena question (does it
 *            fit?) is answered by the executor rather than by a generator's promise.
 * @note [zh] k2 用六个节点证明**执行器**；本条链用它本来要跑的那张图来证明：80 个节点、
 *            188 个槽、108 个常量，1.4 MB 权重始终待在 flash 里。算子还是那些算子——不同的
 *            是：这张网络没有人用 C 写过一遍，以及"arena 够不够"这个问题由执行器回答，
 *            而不是由生成器的承诺回答。
 * @note [en] The weights are a reproducible LCG stream, so this is a **shape and memory**
 *            claim, not an accuracy claim: there is no reference model here to be right
 *            about. What it does establish, and k2 cannot, is that a 80-node blob parses,
 *            lays out and executes inside the arena the planner predicted.
 * @note [zh] 权重是可复现的 LCG 流，所以这是**形状与内存**的主张，不是精度主张：这里没有
 *            可以参考的模型。它能确立而 k2 不能确立的是：80 个节点的 blob 能解析、能排布局、
 *            并且正好跑在计划预测的 arena 里。
 * @since 0.1.0
 */

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief [en] What the check compares: the classifier's logits.
 * @brief [zh] 校验比对的东西：分类器的 logits。
 * @since 0.1.0
 */
typedef struct kat_k2m_result {
    float out[KAT_K2M_BLOB_OUT_FLOATS];
} kat_k2m_result;

/**
 * @brief [en] Parse the baked blob, lay it out, run it; INFR_OK when all of that held.
 * @brief [zh] 解析烘进来的 blob、排布局、跑完；三者都成立时返回 INFR_OK。
 * @since 0.1.0
 */
int kat_k2m_run(kat_k2m_result *result);

/**
 * @brief [en] Length in bytes of the checked tensor.
 * @brief [zh] 被校验张量的字节数。
 * @since 0.1.0
 */
unsigned kat_k2m_expect_len(void);

/**
 * @brief [en] Copy the checked tensor as raw bytes, for a memcmp downstream.
 * @brief [zh] 把被校验张量按裸字节拷出，供下游 memcmp。
 * @since 0.1.0
 */
void kat_k2m_expect(const kat_k2m_result *result, unsigned char *out);

#ifdef __cplusplus
}
#endif
