// The v3 blob, run through the executor **the way the blob says to**.
//
// Three things are checked here, and each one can fail on its own:
//
//   1. a v3 blob is a file this repository can still produce and read -- the fixture under
//      `resources/kat/k2_i8_conv` is committed, so "the format exists" is a fact in the tree rather
//      than a claim in a document;
//   2. the executor drives `infr_conv2d_i8` from the blob's own fields -- kernel, stride, padding,
//      groups, per-channel scales, int32 bias, output zero point -- so a generator that writes the
//      wrong field fails here instead of producing plausible garbage;
//   3. the answer matches a reference computed by an **independent integer implementation** in
//      `emit_k2_i8_blob.py`, which never calls the library. Agreement means two implementations of
//      the same written rule agree, which is the only kind of agreement worth having.
//
// The fp32 blobs are also re-read in the same test file's sibling (`graph_blob_k2.cpp`); if this
// reader had broken v2, that test is the one that would say so, because nothing here touches it.
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

std::vector<unsigned char> read_bytes(const std::string& path) {
    std::ifstream stream(path, std::ios::binary);
    EXPECT_TRUE(stream.good()) << "cannot open " << path;
    return std::vector<unsigned char>(std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>());
}

std::string fixture_dir() { return std::string(RESOURCES_PATH) + "/kat/k2_i8_conv"; }

constexpr std::size_t kWorkspaceBytes = 4096U;

} // namespace

TEST(GraphBlobI8, RunsTheCommittedV3Fixture) {
    const auto blob = read_bytes(fixture_dir() + "/model.blob");
    const auto input = read_bytes(fixture_dir() + "/input.i8");
    const auto expected = read_bytes(fixture_dir() + "/expect.i8");
    ASSERT_FALSE(blob.empty());
    ASSERT_FALSE(input.empty());
    ASSERT_FALSE(expected.empty());

    // The magic and the version are what the older reader keys on, so they are asserted here rather
    // than assumed: this fixture is the tree's only v3 blob and its header is the claim.
    EXPECT_EQ(0, std::memcmp(blob.data(), "INFRKAT\0", 8));
    std::uint32_t version = 0;
    std::memcpy(&version, blob.data() + 8, sizeof version);
    EXPECT_EQ(3U, version);

    // One arena buffer per runtime slot, and the fixture has two (the intermediate between the two
    // convolutions, and the output). The executor refuses rather than overrunning when this is too
    // small, which is how the figure was found.
    //
    // 每个运行期槽一块 arena 缓冲区，夹具里有两个（两卷积之间的中间结果，以及输出）。这个数太小
    // 时执行器是拒绝而不是越界——这个数字就是这么被发现的。
    std::vector<std::int8_t> got(expected.size(), 0);
    std::vector<unsigned char> arena(expected.size() * 2U, 0);
    alignas(8) std::uint8_t workspace_bytes[kWorkspaceBytes] = {};
    infr_workspace workspace{};
    ASSERT_EQ(INFR_OK, infr_workspace_init(&workspace, workspace_bytes, sizeof(workspace_bytes)));

    std::string err;
    ASSERT_TRUE(infrb::run_i8(blob, reinterpret_cast<const std::int8_t*>(input.data()), input.size(), got.data(),
                              got.size(), &arena, &workspace, &err))
        << err;

    // Byte equality, not a tolerance: both sides are integers and the rule is written down. A
    // tolerance here would hide exactly the rounding differences this format exists to expose.
    for (std::size_t i = 0; i < expected.size(); ++i) {
        EXPECT_EQ(static_cast<int>(static_cast<std::int8_t>(expected[i])), static_cast<int>(got[i])) << "element " << i;
    }
    // And the scratch came back, which is the operator's promise and the reason a chain can run
    // several graphs through one workspace.
    EXPECT_EQ(0U, infr_workspace_used(&workspace));
}

TEST(GraphBlobI8, RefusesAVersionItDoesNotRead) {
    // The fixture, relabelled as version 2. The v3 reader must say so by name instead of reading on:
    // "the old runtime refuses the new blob" and "the new runtime refuses to guess" are the same
    // property seen from two sides, and this is the second side.
    auto blob = read_bytes(fixture_dir() + "/model.blob");
    ASSERT_GE(blob.size(), 12U);
    const std::uint32_t version = 2;
    std::memcpy(blob.data() + 8, &version, sizeof version);

    std::vector<std::int8_t> got(27, 0);
    std::vector<unsigned char> arena(27, 0);
    alignas(8) std::uint8_t workspace_bytes[kWorkspaceBytes] = {};
    infr_workspace workspace{};
    ASSERT_EQ(INFR_OK, infr_workspace_init(&workspace, workspace_bytes, sizeof(workspace_bytes)));
    std::string err;
    EXPECT_FALSE(infrb::run_i8(blob, got.data(), 50, got.data(), got.size(), &arena, &workspace, &err));
    EXPECT_NE(std::string::npos, err.find("v3")) << err;
}

TEST(GraphBlobI8, RefusesAnOperatorItDoesNotRun) {
    // `p[3] = groups` is the last field the executor reads from the node, so flipping the op code
    // is the cheapest way to prove the node table is actually consulted rather than a single-conv
    // shortcut being taken. Node records start at byte 24 and the second node is the conv.
    auto blob = read_bytes(fixture_dir() + "/model.blob");
    ASSERT_GE(blob.size(), 24U + 36U + 36U);
    blob[24U + 36U] = 5U; // op 5 is `mul` in this format
    std::vector<std::int8_t> got(27, 0);
    std::vector<std::int8_t> input(50, 0);
    std::vector<unsigned char> arena(27, 0);
    alignas(8) std::uint8_t workspace_bytes[kWorkspaceBytes] = {};
    infr_workspace workspace{};
    ASSERT_EQ(INFR_OK, infr_workspace_init(&workspace, workspace_bytes, sizeof(workspace_bytes)));
    std::string err;
    EXPECT_FALSE(infrb::run_i8(blob, input.data(), input.size(), got.data(), got.size(), &arena, &workspace, &err));
    EXPECT_NE(std::string::npos, err.find("conv2d")) << err;
}
