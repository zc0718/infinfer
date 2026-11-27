// Conan::ImportStart
#pragma once

#include <stdint.h>

#include <infinfer/infinfer.h>
// Conan::ImportEnd

/*!
 * @brief [en] Run an `INFRKAT` v2 graph blob through the public L2 operators, on the target.
 * @brief [zh] 在目标端把 `INFRKAT` v2 图 blob 通过公开 L2 算子跑起来。
 * @note [en] This is the C twin of `test_package/test/unit/blob_graph.hpp`. Two
 *            implementations on purpose: the C++ one is checked against numpy on the
 *            host, this one proves the *same bytes* execute the same way on a
 *            Cortex-M, and a shared parser would only prove one implementation agrees
 *            with itself. Both re-derive every output shape from the operator's own
 *            arithmetic and compare it against what the slot declares, so a generator
 *            that writes a wrong shape fails here instead of feeding garbage floats to
 *            the next layer.
 * @note [zh] 这是 `test_package/test/unit/blob_graph.hpp` 的 C 版。故意两份实现：C++ 那份
 *            在宿主上对 numpy 校验，这份证明**同一批字节**在 Cortex-M 上走同一条路；共享
 *            解析器只能证明一个实现和它自己一致。两边都会用算子自己的算术重推每个输出
 *            形状并与槽声明比对，所以生成器把形状写错时是在这里失败，而不是把垃圾浮点数
 *            喂给下一层。
 * @note [en] No heap and no libc beyond `memcpy`: the slot table lives in a fixed
 *            array, and the caller owns both the arena and the workspace. "It ran on
 *            the target" therefore does not secretly mean "a malloc succeeded".
 * @note [zh] 不用堆、除 `memcpy` 外不依赖 libc：槽表放在固定数组里，arena 与 workspace
 *            都由调用方提供。于是"它在目标端跑通了"不会悄悄等于"某次 malloc 成功了"。
 * @since 0.1.0
 */

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief [en] Error codes below `infr_status`: distinct so a QEMU failure names its stage.
 * @brief [zh] `infr_status` 之外的错误码：彼此可区分，好让 QEMU 上的失败直接点名阶段。
 * @since 0.1.0
 */
#define KAT_BLOB_OK 0
#define KAT_BLOB_ERR_HEADER (-40)
#define KAT_BLOB_ERR_VERSION (-41)
#define KAT_BLOB_ERR_TABLE (-42)
#define KAT_BLOB_ERR_SLOT (-43)
#define KAT_BLOB_ERR_SHAPE (-44)
#define KAT_BLOB_ERR_OP (-45)
#define KAT_BLOB_ERR_ARENA (-46)
#define KAT_BLOB_ERR_INPUT (-47)
#define KAT_BLOB_ERR_MISMATCH (-48)

/**
 * @brief [en] What the caller reads after a successful run.
 * @brief [zh] 运行成功后调用方要读的东西。
 * @since 0.1.0
 */
typedef struct kat_blob_view {
    /**
     * @brief [en] First element of the last node's output slot, inside the arena.
     * @brief [zh] 最后一个节点输出槽的首元素，位于 arena 内。
     */
    const float *out;
    /**
     * @brief [en] Element count of that slot.
     * @brief [zh] 该槽的元素个数。
     */
    uint32_t out_len;
    /**
     * @brief [en] Floats the executor's own layout actually needed, i.e. the first-fit
     *        reuse high-water mark. Reported so a caller can size the arena from the
     *        graph instead of trusting a number written in a document.
     * @brief [zh] 执行器自己排出的布局真正需要的 float 个数，即首次适配复用的高水位。
     *        报出来是为了让调用方能按图算 arena 大小，而不是相信文档里的数字。
     */
    uint32_t arena_floats;
} kat_blob_view;

/**
 * @brief [en] Parse and execute a v2 graph blob.
 * @brief [zh] 解析并执行一份 v2 图 blob。
 * @param bytes [en] Blob bytes; must stay alive for the duration of the call.
 * @param bytes [zh] blob 字节；调用期间必须保持有效。
 * @param byte_len [en] Length of `bytes`.
 * @param byte_len [zh] `bytes` 的长度。
 * @param input [en] Data for the input slot, or NULL when the graph has no `input` node.
 * @param input [zh] 输入槽的数据；图里没有 `input` 节点时为 NULL。
 * @param input_len [en] Element count of `input`; must equal the input slot's extent.
 * @param input_len [zh] `input` 的元素个数；必须等于输入槽的容量。
 * @param arena [en] Scratch for every runtime slot. The size it needs is the
 *              first-fit high-water mark of the slots' live ranges (not the sum of
 *              their extents: a slot is reused once the node that last reads it has
 *              run), and `KAT_BLOB_ERR_ARENA` is returned when it is short.
 * @param arena [zh] 所有运行期槽的临时空间。所需大小是各槽活跃区间按首次适配排出的高水位
 *              （不是容量之和：最后一个读它的节点跑完之后这个槽就可以被复用），不够时返回
 *              `KAT_BLOB_ERR_ARENA`。
 * @param arena_len [en] Element count of `arena`.
 * @param arena_len [zh] `arena` 的元素个数。
 * @param workspace [en] Scratch for the operators themselves (rewound by each of them).
 * @param workspace [zh] 算子自身要的临时空间（每个算子返回前都会回退）。
 * @param view [en] Receives the output pointer and length; must not be NULL.
 * @param view [zh] 接收输出指针与长度；不得为 NULL。
 * @return KAT_BLOB_OK, or one of the KAT_BLOB_ERR_* codes, or a non-OK `infr_status`
 *         passed through from the operator that failed.
 * @note [en] Weights are read straight out of `bytes`: the slot payload is handed to
 *            `infr_tensor_init` as a writable pointer because the contract has no
 *            const form, and the L2 operators never write a filter. A constant slot is
 *            never a node's output, which is checked while walking the table.
 * @note [zh] 权重直接从 `bytes` 里读：槽的负载以可写指针交给 `infr_tensor_init`，因为契约
 *            没有 const 形式，而 L2 算子从不写滤波器。常量槽永不作为节点的输出，这一点在
 *            遍历槽表时就会检查。
 * @since 0.1.0
 */
int kat_blob_run(const unsigned char *bytes, uint32_t byte_len, const float *input, uint32_t input_len,
                 float *arena, uint32_t arena_len, infr_workspace *workspace, kat_blob_view *view);

#ifdef __cplusplus
}
#endif
