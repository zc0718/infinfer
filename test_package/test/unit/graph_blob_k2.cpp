// Run the `k2_conv_micro` blob **the way the blob says to**, through the generic
// graph executor, and check it against the same numpy expectation the hand-written
// chain uses.
//
// 按 blob 自己说的方式跑 `k2_conv_micro`：通过通用图执行器，对准与手写链同一份 numpy
// 期望。
//
// 与 `graph_kat_k2_conv_micro.cpp` 的分工（两个测试都要留着）：
//  · 那个测试把图**在 C 里重写一遍**——所以它是格式的独立第二意见，blob 里的节点表写错
//    它不会跟着错；
//  · 本测试**服从节点表**——所以它是"图是数据"这件事的证明，也是 P4.5 的第一条案例：
//    同一份执行器将来跑 MobileNetV3 / YOLO，不必再写一遍 C。
// 只有两条同时绿，才说明"blob 说的是对的"且"执行器照做是对的"。
#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include <infinfer/infinfer.h>

#include "blob_graph.hpp"
#include "parity_harness.hpp"

namespace {

std::vector<unsigned char> read_file(const std::string& path) {
    std::ifstream stream(path, std::ios::binary);
    EXPECT_TRUE(stream.good()) << "cannot open " << path;
    return std::vector<unsigned char>(std::istreambuf_iterator<char>(stream),
                                      std::istreambuf_iterator<char>());
}

std::string k2_dir() {
    return std::string(RESOURCES_PATH) + "/kat/k2_conv_micro";
}

/* The graph and the arena both come out of the blob, so nothing here names a shape. */
TEST(GraphBlobK2, TheBlobOwnsTheGraph) {
    infrb::Blob blob;
    std::string err;
    ASSERT_TRUE(infrb::parse(read_file(k2_dir() + "/model.blob"), &blob, &err)) << err;
    EXPECT_EQ(2U, blob.version);
    EXPECT_EQ(6U, blob.nodes.size());
    EXPECT_EQ(infrb::kOpInput, blob.nodes.front().op);
    EXPECT_EQ(infrb::kOpFc, blob.nodes.back().op);
    /* Every stage is a runtime buffer except the six constants, and the arena is
     * exactly the sum of their extents -- so the caller's buffer size is derivable
     * from the blob instead of being guessed. */
    std::size_t runtime_slots = 0;
    std::size_t expected_arena = 0;
    for (const auto& slot : blob.slots) {
        if (slot.byte_len != 0) {
            continue;
        }
        ++runtime_slots;
        std::size_t count = 1;
        for (std::uint32_t axis = 0; axis < slot.rank; ++axis) {
            count *= slot.dims[axis];
        }
        expected_arena += count;
    }
    EXPECT_EQ(6U, runtime_slots);
    EXPECT_EQ(expected_arena, blob.arena_floats);
    EXPECT_NE(infrb::kNoSlot, infrb::input_slot(blob));
    EXPECT_NE(infrb::kNoSlot, infrb::output_slot(blob));
}

TEST(GraphBlobK2, ReproducesTheNumpyExpectation) {
    infrb::Blob blob;
    std::string err;
    ASSERT_TRUE(infrb::parse(read_file(k2_dir() + "/model.blob"), &blob, &err)) << err;

    const auto input_bytes = read_file(k2_dir() + "/input.bin");
    const auto expect_bytes = read_file(k2_dir() + "/expect_fp32.bin");
    const auto& input_slot = blob.slots[infrb::input_slot(blob)];
    const auto& output_slot = blob.slots[infrb::output_slot(blob)];
    ASSERT_EQ(input_bytes.size(), static_cast<std::size_t>(input_slot.dims[0] * input_slot.dims[1] *
                                                           input_slot.dims[2] * input_slot.dims[3] * 4U));
    ASSERT_EQ(expect_bytes.size(), static_cast<std::size_t>(output_slot.dims[0] * output_slot.dims[1] *
                                                            output_slot.dims[2] * output_slot.dims[3] * 4U));

    std::vector<float> arena(blob.arena_floats);
    std::memcpy(arena.data() + input_slot.arena, input_bytes.data(), input_bytes.size());

    alignas(8) std::uint8_t workspace_bytes[4096] = {};
    infr_workspace workspace{};
    ASSERT_EQ(INFR_OK, infr_workspace_init(&workspace, workspace_bytes, sizeof(workspace_bytes)));
    ASSERT_TRUE(infrb::run(blob, arena.data(), arena.size(), &workspace, &err)) << err;

    std::vector<float> expect(output_slot.dims[3]);
    std::memcpy(expect.data(), expect_bytes.data(), expect_bytes.size());
    infr_test::expect_float_buffers_near(expect.data(), arena.data() + output_slot.arena, output_slot.dims[3],
                                         "k2_conv_micro (blob executor)");

    /* The graph ran in a constant workspace: every operator gave its scratch back
     * (INV-8), observed at the outermost level. */
    EXPECT_EQ(0U, infr_workspace_used(&workspace));
}

/* The executor is only worth having if a wrong blob is rejected instead of being
 * executed. Both mutations are applied to the parsed-free byte buffer, so they are
 * the exact class of mistake a generator makes. */
TEST(GraphBlobK2, RejectsAMalformedBlob) {
    const auto good = read_file(k2_dir() + "/model.blob");

    {
        std::vector<unsigned char> bad = good;
        ASSERT_GT(bad.size(), std::size_t{8});
        bad[0] = 'X';
        infrb::Blob blob;
        std::string err;
        EXPECT_FALSE(infrb::parse(std::move(bad), &blob, &err));
        EXPECT_NE(std::string::npos, err.find("magic")) << err;
    }
    {
        std::vector<unsigned char> bad = good;
        bad.resize(good.size() - 16);
        infrb::Blob blob;
        std::string err;
        EXPECT_FALSE(infrb::parse(std::move(bad), &blob, &err)) << "a truncated blob parsed";
    }
    {
        /* Declare the first convolution's output one row too short. The slot table
         * stays self-consistent (byte_len is 0), so only the executor's own
         * re-derivation of the output shape can catch it. */
        std::vector<unsigned char> bad = good;
        const std::string name = "conv";
        std::size_t where = std::string::npos;
        for (std::size_t i = 0; i + name.size() < bad.size(); ++i) {
            if (std::memcmp(bad.data() + i, name.c_str(), name.size()) == 0 && bad[i + name.size()] == 0) {
                where = i;
                break;
            }
        }
        ASSERT_NE(std::string::npos, where) << "the slot named 'conv' vanished from the blob";
        bad[where + 28] = 15U; /* dims[1]: 16 -> 15 */
        infrb::Blob blob;
        std::string err;
        ASSERT_TRUE(infrb::parse(std::move(bad), &blob, &err)) << err;
        std::vector<float> arena(blob.arena_floats);
        alignas(8) std::uint8_t workspace_bytes[4096] = {};
        infr_workspace workspace{};
        ASSERT_EQ(INFR_OK, infr_workspace_init(&workspace, workspace_bytes, sizeof(workspace_bytes)));
        EXPECT_FALSE(infrb::run(blob, arena.data(), arena.size(), &workspace, &err));
        EXPECT_NE(std::string::npos, err.find("disagree")) << err;
    }
}

}  // namespace
