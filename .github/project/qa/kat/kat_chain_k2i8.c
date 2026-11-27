// Conan::ImportStart
#include <stdint.h>
#include <string.h>

#include <infinfer/infinfer.h>

#include "blob_graph.h"
#include "kat_chain_k2i8.h"
#include "generated/kat_k2i8_blob.h"
#include "generated/kat_k2i8_expect.h"
#include "generated/kat_k2i8_input.h"
// Conan::ImportEnd

/*!
 * @brief [en] `k2_i8_conv` on the target: one int8 convolution, driven by the blob's own fields.
 * @brief [zh] 目标端的 `k2_i8_conv`：一个 int8 卷积，由 blob 自己的字段驱动。
 * @note [en] Three byte arrays arrive from one place -- the fixture under
 *            `test_package/resources/kat/k2_i8_conv` -- so the thing this chain runs is the same
 *            artifact the host test reads and the same one the generator wrote. Nothing here is
 *            transcribed by hand.
 * @note [zh] 三个字节数组来自**同一处**——`test_package/resources/kat/k2_i8_conv` 夹具——所以
 *            本链跑的东西、宿主测试读的东西、生成器写的东西是同一个产物。这里没有一个字节是
 *            手抄的。
 * @since 0.2.0
 */

/* The workspace is one scratch call deep: `infr_conv2d_i8` borrows the window
 * (kernel*kernel*in_per_group bytes) and nothing else, because the accumulator for one output
 * element lives in a local. 64 bytes covers a 3x3 window over 2 channels (18 B) with room to
 * spare; the chain asserts the workspace comes back empty, so a future operator that quietly
 * needs more fails here rather than corrupting the next node.
 *
 * 工作区只有一次借用深度：`infr_conv2d_i8` 只借窗口（kernel*kernel*in_per_group 字节），因为
 * 单个输出元素的累加器在局部变量里。64 字节足够 3x3 窗口跨 2 通道（18 B）并留有余量；链在末尾
 * 断言工作区归零，于是将来某个算子悄悄要得更多时是在这里失败，而不是污染下一个节点。 */
/* One buffer per runtime slot: the fixture has two, the intermediate between the two
 * convolutions and the output. The executor refuses rather than overrunning when this is too
 * small, so the figure is checkable rather than remembered.
 *
 * 每个运行期槽一块：(夹具里有两个——两卷积之间的中间结果与输出)。这个数太小时执行器是拒绝
 * 而不是越界，所以这个数字是可核对的，不是靠记。 */
#define K2I8_ARENA_BYTES (2 * KAT_K2I8_EXPECT_BYTES_LEN)
#define K2I8_WORKSPACE_BYTES 64U

int kat_k2i8_run(kat_k2i8_result *result) {
    /* static, not automatic: a Cortex-M chain runs with a few kilobytes of stack and these are
     * three fixed buffers whose lifetime is the whole call anyway.
     *
     * 用 static 而不是自动变量：Cortex-M 链的栈只有几 KB，而这三块缓冲区的生命周期本来就是
     * 整个调用。 */
    static unsigned char arena[K2I8_ARENA_BYTES];
    static uint8_t workspace_bytes[K2I8_WORKSPACE_BYTES];
    infr_workspace workspace;
    infr_status status;
    int blob_status;

    if (result == NULL) {
        return KAT_BLOB_ERR_INPUT;
    }
    status = infr_workspace_init(&workspace, workspace_bytes, sizeof workspace_bytes);
    if (status != INFR_OK) {
        return KAT_BLOB_ERR_ARENA;
    }
    blob_status = kat_blob_run_i8(kat_k2i8_blob, KAT_K2I8_BLOB_LEN, (const int8_t *)(const void *)kat_k2i8_input_bytes,
                                  KAT_K2I8_INPUT_BYTES_LEN, arena, (uint32_t)sizeof arena, &workspace,
                                  (int8_t *)(void *)result->out, (uint32_t)sizeof result->out);
    if (blob_status != KAT_BLOB_OK) {
        return blob_status;
    }
    /* The workspace must be back where it started, or the next node in a longer graph would be
     * handed a buffer that is already spoken for.
     *
     * 工作区必须回到起点，否则更长图里的下一个节点会拿到一块已经被占用的缓冲区。 */
    if (infr_workspace_used(&workspace) != 0U) {
        return KAT_BLOB_ERR_ARENA;
    }
    /* And the arithmetic, against the generator's integer reference rather than against a host run
     * of this same code. Byte equality is the whole claim: both sides are integers and the rounding
     * rule is written down, so a tolerance here would hide exactly the disagreement the fixture
     * exists to expose.
     *
     * 还有算术本身：对齐的是生成器那份整数参考，而不是本代码的一次宿主运行。逐字节相等就是全部
     * 主张：两侧都是整数、舍入规则是写下来的，所以在这里留容差恰好会盖住夹具存在的意义。 */
    if (memcmp(result->out, kat_k2i8_expect_bytes, KAT_K2I8_EXPECT_BYTES_LEN) != 0) {
        return KAT_BLOB_ERR_MISMATCH;
    }
    return KAT_BLOB_OK;
}

unsigned kat_k2i8_expect_len(void) { return (unsigned)sizeof(((kat_k2i8_result *)0)->out); }

void kat_k2i8_expect(const kat_k2i8_result *result, unsigned char *out) {
    if (result == NULL || out == NULL) {
        return;
    }
    memcpy(out, result->out, sizeof result->out);
}
