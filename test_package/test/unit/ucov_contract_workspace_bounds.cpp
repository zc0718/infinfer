// P0.5 case 2 (deliberately red): the static memory boundary contract.
//
// The reference defect this pins: an offset table computed in words while the
// buffer was measured in bytes (8 floats fit in 32 bytes, not 8). The assertions
// below are byte offsets taken from workspace.h's documented contract.
//
// Expected state at 0.1.0: RED -- the bump allocator is not implemented, so the
// first allocation returns INFR_ERR_NOT_IMPLEMENTED instead of a reservation.
#include <gtest/gtest.h>

#include <cstdint>

#include <infinfer/infinfer.h>

TEST(ContractWorkspace, ByteOffsetsAreBytesNotWords) {
    // 1) Consecutive reservations are placed back to back in BYTES.
    uint8_t buffer[16] = {};
    infr_workspace ws{};
    ASSERT_EQ(INFR_OK, infr_workspace_init(&ws, buffer, sizeof buffer));
    void* first = nullptr;
    void* second = nullptr;
    void* third = nullptr;
    EXPECT_EQ(INFR_OK, infr_workspace_alloc(&ws, 8, 1, &first));
    EXPECT_EQ(static_cast<void*>(buffer), first);
    EXPECT_EQ(INFR_OK, infr_workspace_alloc(&ws, 8, 1, &second));
    EXPECT_EQ(static_cast<void*>(buffer + 8), second)
        << "an offset table in words puts the second reservation at buffer+32";
    EXPECT_EQ(INFR_ERR_BUFFER_TOO_SMALL, infr_workspace_alloc(&ws, 1, 1, &third));
    EXPECT_EQ(sizeof buffer, infr_workspace_used(&ws));

    // 2) Alignment is honoured from a deliberately misaligned base.
    uint8_t raw[17] = {};
    infr_workspace aligned_ws{};
    ASSERT_EQ(INFR_OK, infr_workspace_init(&aligned_ws, raw + 1, sizeof raw - 1));
    void* aligned = nullptr;
    EXPECT_EQ(INFR_OK, infr_workspace_alloc(&aligned_ws, 1, 8, &aligned));
    EXPECT_EQ(0U, reinterpret_cast<uintptr_t>(aligned) % 8U);
    EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, infr_workspace_alloc(&aligned_ws, 1, 3, &aligned));

    // 3) Argument validation that holds regardless of the allocator.
    EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, infr_workspace_init(nullptr, nullptr, 0));
    EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, infr_workspace_init(&ws, nullptr, 4));
    EXPECT_EQ(INFR_OK, infr_workspace_init(&ws, nullptr, 0));
    EXPECT_EQ(0U, infr_workspace_used(nullptr));
}


int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
