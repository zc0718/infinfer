// Conan::ImportStart
#include <stdint.h>
#include <string.h>

#include <infinfer/infinfer.h>

#include "kat_chain_k2m.h"
#include "blob_graph.h"
// Conan::ImportEnd

/*!
 * @brief [en] `k2m_mobilenet` on the target: the blob, the arena, and nothing hand-written.
 * @brief [zh] 目标端的 `k2m_mobilenet`：blob、arena，没有别的手写东西。
 * @note [en] Every size below comes from the generated header, which derives it from the
 *            blob. That is deliberate: `K2M_IN` is 27648 because the model says so, and a
 *            C constant repeating it would be a second place to be wrong.
 * @note [zh] 下面的每个尺寸都来自生成的头部，而生成的头部是从 blob 推出来的。这是有意的：
 *            `K2M_IN` 是 27648 是因为模型这么说，在 C 里再写一遍就是第二个可以说错的地方。
 * @since 0.1.0
 */

/* The arena the executor asked for, to the float. Handing it exactly `_ARENA` is half the
 * test: if the layout ever needs more, `kat_blob_run` returns KAT_BLOB_ERR_ARENA and the
 * chain says so by exit code instead of silently reading past a buffer.
 *
 * 执行器要多少就给多少，一个 float 都不多。这正是测试的一半：布局若需要更多，
 * `kat_blob_run` 会返回 KAT_BLOB_ERR_ARENA，链用退出码说出来，而不是悄悄越界读。 */
static float g_arena[KAT_K2M_BLOB_ARENA];

static float g_input[KAT_K2M_BLOB_IN_FLOATS];

/* Conv2d allocates its innermost window from the workspace; the widest map in this model
 * is 96x96, so one such plane plus slack covers it. Asserted back to zero afterwards.
 *
 * 卷积从 workspace 里分配最内层的窗口；这张模型最宽的特征图是 96×96，一个平面加余量足够。
 * 末尾断言回到 0。 */
static unsigned char g_workspace[96U * 96U * 4U + 4096U] __attribute__((aligned(8)));

/* Deterministic input: a plain LCG, so "the host and the target saw the same numbers"
 * needs no file and no float literal. The values are in [-1, 1) with a 1/1024 step, which
 * keeps them exactly representable and therefore identical everywhere.
 *
 * 确定性输入：一个朴素的 LCG，所以"宿主与目标看到同一批数"既不需要文件也不需要浮点字面量。
 * 取值在 [-1, 1) 内、步长 1/1024，因而可精确表示，在任何平台上都相同。 */
static void fill_input(void)
{
    uint32_t state = 20261127U;
    uint32_t index;
    for (index = 0U; index < KAT_K2M_BLOB_IN_FLOATS; ++index) {
        int32_t value;
        state = (state * 1664525U) + 1013904223U;
        value = (int32_t)((state >> 12) & 0x7FFU) - 1024; /* [-1024, 1023] */
        g_input[index] = (float)value / 1024.0F;
    }
}

int kat_k2m_run(kat_k2m_result *result)
{
    static kat_blob_view view;
    infr_workspace workspace;
    int status;

    if (result == 0) {
        return INFR_ERR_INVALID_ARGUMENT;
    }
    if (infr_workspace_init(&workspace, g_workspace, sizeof g_workspace) != INFR_OK) {
        return INFR_ERR_INVALID_ARGUMENT;
    }
    fill_input();

    status = kat_blob_run(kat_k2m_blob, KAT_K2M_BLOB_LEN, g_input, KAT_K2M_BLOB_IN_FLOATS,
                          g_arena, KAT_K2M_BLOB_ARENA, &workspace, &view);
    if (status != KAT_BLOB_OK) {
        return status;
    }
    /* Two implementations of the same plan have to agree -- same check as k2, now on a
     * graph with 80 nodes instead of six.
     *
     * 同一份计划的两份实现必须一致——与 k2 同一条检查，只是图从六个节点变成八十个。 */
    if (view.arena_floats != KAT_K2M_BLOB_ARENA) {
        return KAT_BLOB_ERR_ARENA;
    }
    if (view.out_len != KAT_K2M_BLOB_OUT_FLOATS || view.out == 0) {
        return KAT_BLOB_ERR_MISMATCH;
    }
    if (infr_workspace_used(&workspace) != 0U) {
        /* No dedicated code exists for "an operator did not rewind the workspace", and
         * inventing one here would be a status the header does not document. It is a
         * contract violation, so it reports as one. */
        return INFR_ERR_INVALID_ARGUMENT;
    }
    memcpy(result->out, view.out, sizeof result->out);

    /* The blob itself is hashed, not the weights: if the .incbin ever picked up a stale
     * file the length would agree while the bytes would not, and every target would then
     * agree with each other about the wrong model.
     *
     * 校验的是 blob 的摘要而不是权重：若 `.incbin` 拿到过期文件，长度会一致而字节不会，
     * 于是所有目标都会一致地跑错模型。 */
    {
        uint32_t hash = 2166136261U; /* FNV-1a over the blob, folded to 8 bits */
        uint32_t index;
        for (index = 0U; index < KAT_K2M_BLOB_LEN; ++index) {
            hash = (hash ^ (uint32_t)kat_k2m_blob[index]) * 16777619U;
        }
        if ((hash & 0xFFU) != (KAT_K2M_BLOB_HASH8)) {
            return KAT_BLOB_ERR_HEADER;
        }
    }
    return INFR_OK;
}

unsigned kat_k2m_expect_len(void)
{
    return (unsigned)(sizeof(kat_k2m_result));
}

void kat_k2m_expect(const kat_k2m_result *result, unsigned char *out)
{
    if (result == 0 || out == 0) {
        return;
    }
    memcpy(out, result->out, sizeof result->out);
}
