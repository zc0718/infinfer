// Conan::ImportStart
#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>

#include <infinfer/infinfer.h>
// Conan::ImportEnd

/*
 * The workspace's guard clauses, which the happy-path cases never reach.
 *
 * These are the branches a coverage floor exists to point at: they only run when
 * a caller is already wrong, so no amount of realistic use exercises them, and a
 * guard that was never run is an assumption rather than a contract. The empty
 * workspace is the interesting one -- `infr_workspace_init(&ws, nullptr, 0)` is a
 * legal state whose documented behaviour is "a zero-byte request succeeds and
 * hands back NULL, anything else is refused".
 *
 * 工作区的**守卫分支**——happy path 永远走不到的那些。覆盖率地板要指的正是这类：只在
 * 调用方已经出错时才执行，正常使用永远不会跑到，而**没被执行过的守卫只是假设**。
 * 其中最有意思的是空工作区：`infr_workspace_init(&ws, nullptr, 0)` 是合法状态，其
 * 文档化行为是「零字节请求成功并返回 NULL，其余一律拒绝」。
 */

namespace {

constexpr std::size_t kArenaBytes = 64U;

}  // namespace

TEST(ContractWorkspaceGuards, RejectsNullWorkspaceAndNullOut) {
    std::uint8_t arena[kArenaBytes] = {};
    void* block = nullptr;

    EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, infr_workspace_alloc(nullptr, 8U, 1U, &block));
    EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, infr_workspace_alloc(nullptr, 0U, 1U, &block));

    infr_workspace workspace;
    ASSERT_EQ(INFR_OK, infr_workspace_init(&workspace, arena, kArenaBytes));
    EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, infr_workspace_alloc(&workspace, 8U, 1U, nullptr));
    EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, infr_workspace_alloc(&workspace, 8U, 0U, &block));
    EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, infr_workspace_alloc(&workspace, 8U, 3U, &block));
}

TEST(ContractWorkspaceGuards, AnEmptyWorkspaceOnlySatisfiesAZeroByteRequest) {
    infr_workspace workspace;
    void* block = reinterpret_cast<void*>(0x1);

    ASSERT_EQ(INFR_OK, infr_workspace_init(&workspace, nullptr, 0U));
    EXPECT_EQ(INFR_OK, infr_workspace_alloc(&workspace, 0U, 1U, &block));
    EXPECT_EQ(nullptr, block) << "a zero-byte request from an empty workspace hands back nothing";

    block = reinterpret_cast<void*>(0x1);
    EXPECT_EQ(INFR_ERR_BUFFER_TOO_SMALL, infr_workspace_alloc(&workspace, 1U, 1U, &block));
    EXPECT_EQ(reinterpret_cast<void*>(0x1), block) << "a refused request must not write the out pointer";
}

TEST(ContractWorkspaceGuards, AlignmentPaddingIsCountedAgainstTheArena) {
    // Four bytes of arena cannot pay for aligning to sixteen: the refusal has to
    // come from the padding check, not from running off the end and finding out.
    std::uint8_t arena[4] = {};
    infr_workspace workspace;
    void* block = nullptr;

    ASSERT_EQ(INFR_OK, infr_workspace_init(&workspace, arena, sizeof(arena)));
    EXPECT_EQ(INFR_ERR_BUFFER_TOO_SMALL, infr_workspace_alloc(&workspace, 1U, 16U, &block));
    EXPECT_EQ(0U, infr_workspace_used(&workspace)) << "a refused allocation reserves nothing";
}

TEST(ContractWorkspaceGuards, RejectsABufferWithoutASize) {
    infr_workspace workspace;

    EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, infr_workspace_init(nullptr, nullptr, 0U));
    EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, infr_workspace_init(&workspace, nullptr, 8U));
}


int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
