// Conan::ImportStart
#include <gtest/gtest.h>

#include <cstdint>

#include <infinfer/infinfer.h>
// Conan::ImportEnd

/*!
 * @brief [en] Workspace rewind: an operator borrows scratch and gives it back before the next node.
 * @brief [zh] 工作区回退：算子借临时缓冲，并在下一个节点前还回去。
 * @since 0.1.0
 */

namespace {

constexpr size_t kScratch = 32U;

}  // namespace

TEST(ContractWorkspaceRewind, ReleaseGivesBackScratchAndTheSameBytesAreReused) {
    alignas(8) uint8_t buffer[64] = {};
    infr_workspace ws{};
    ASSERT_EQ(INFR_OK, infr_workspace_init(&ws, buffer, sizeof(buffer)));

    void* first = nullptr;
    ASSERT_EQ(INFR_OK, infr_workspace_alloc(&ws, kScratch, alignof(float), &first));
    const size_t mark = infr_workspace_mark(&ws);
    EXPECT_EQ(kScratch, mark);
    EXPECT_EQ(mark, infr_workspace_used(&ws));

    void* scratch = nullptr;
    ASSERT_EQ(INFR_OK, infr_workspace_alloc(&ws, kScratch, alignof(float), &scratch));
    EXPECT_EQ(2U * kScratch, infr_workspace_used(&ws));

    ASSERT_EQ(INFR_OK, infr_workspace_release(&ws, mark));
    EXPECT_EQ(mark, infr_workspace_used(&ws));

    // The released bytes are handed out again, at the same address: a graph that
    // borrows the same scratch for every node must not grow the workspace.
    void* reused = nullptr;
    ASSERT_EQ(INFR_OK, infr_workspace_alloc(&ws, kScratch, alignof(float), &reused));
    EXPECT_EQ(scratch, reused);
}

TEST(ContractWorkspaceRewind, ReleaseIsLifo) {
    alignas(8) uint8_t buffer[64] = {};
    infr_workspace ws{};
    ASSERT_EQ(INFR_OK, infr_workspace_init(&ws, buffer, sizeof(buffer)));

    void* outer = nullptr;
    ASSERT_EQ(INFR_OK, infr_workspace_alloc(&ws, 8U, 1U, &outer));
    const size_t outer_mark = infr_workspace_mark(&ws);
    void* middle = nullptr;
    ASSERT_EQ(INFR_OK, infr_workspace_alloc(&ws, 8U, 1U, &middle));
    void* inner = nullptr;
    ASSERT_EQ(INFR_OK, infr_workspace_alloc(&ws, 8U, 1U, &inner));

    // Out of order: releasing the outer mark first drops the middle reservation too,
    // which is what a nested operator that forgot to release its own mark would do.
    ASSERT_EQ(INFR_OK, infr_workspace_release(&ws, outer_mark));
    EXPECT_EQ(outer_mark, infr_workspace_used(&ws));

    // The middle reservation's bytes are handed out again at the same address.
    void* again = nullptr;
    ASSERT_EQ(INFR_OK, infr_workspace_alloc(&ws, 8U, 1U, &again));
    EXPECT_EQ(middle, again);
}

TEST(ContractWorkspaceRewind, ReleaseRefusesAMarkPastTheCursor) {
    alignas(8) uint8_t buffer[16] = {};
    infr_workspace ws{};
    ASSERT_EQ(INFR_OK, infr_workspace_init(&ws, buffer, sizeof(buffer)));
    void* one = nullptr;
    ASSERT_EQ(INFR_OK, infr_workspace_alloc(&ws, 8U, 1U, &one));
    const size_t used = infr_workspace_used(&ws);

    // Moving the cursor forwards would hand out bytes that were already handed out.
    EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, infr_workspace_release(&ws, used + 1U));
    EXPECT_EQ(used, infr_workspace_used(&ws));

    EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, infr_workspace_release(nullptr, 0U));
    EXPECT_EQ(0U, infr_workspace_mark(nullptr));
}
