// Run a blob that **came out of an ONNX importer**, against the expectation an independent
// numpy forward produced, so the two halves of the pipeline are held to each other.
//
// 跑一个**从 ONNX 导入器出来**的 blob，对准一份独立的 numpy 前向所产生的期望——让流水线的
// 两半互相钉住。
//
// Why the fixture exists: `.github/project/qa/prep/emit_onnx_blob.py --selftest` proves the
// importer's translation is byte-identical to a hand-written blob, but "the bytes are the
// ones I meant" is not the same claim as "the graph computes the right numbers". This test
// is the second claim, and it is the first *independent* numeric reference the blob
// executor has ever had: the moment it was added it disagreed with the executor and named
// a real bug (the per-channel gate multiplied by the wrong factors, gaps G-103).
//
// The map is deliberately non-square -- H, W and C are 5, 4, 3 -- because a fixture whose
// axes are all the same number cannot tell an axis swap from a correct implementation.
//
// 夹具存在的理由：`emit_onnx_blob.py --selftest` 证明导入的翻译与手写 blob 逐字节相同，但
// "字节是我想要的那些"和"这张图算出来的数是对的"不是同一个主张。本测试是第二个主张，也是
// blob 执行器有史以来**第一份独立数值参考**——它一加进来就与执行器不一致，并指出一个真 bug
// （逐通道门乘错了因子，gaps G-103）。特征图故意非方形（H/W/C = 5/4/3）：三个轴都相等的
// 夹具分不出“轴换错了”和“算对了”。
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

std::string fixture_dir() {
    return std::string(RESOURCES_PATH) + "/kat/onnx_subset";
}

std::vector<float> read_floats(const std::string& path) {
    const std::vector<unsigned char> bytes = read_file(path);
    std::vector<float> values(bytes.size() / sizeof(float));
    if (!values.empty()) {
        std::memcpy(values.data(), bytes.data(), values.size() * sizeof(float));
    }
    return values;
}

}  // namespace

/* The imported graph is what the importer said it is: one conv with its relu6 fused, a
 * squeeze-and-excite block (pool + hard-sigmoid fused, then two per-channel multiplies --
 * one with a **computed** gate and one with a stored one), a global pool, and one fully
 * connected node. */
TEST(GraphBlobOnnxSubset, TheImportedGraphHasTheShapeTheImporterDeclared) {
    infrb::Blob blob;
    std::string err;
    ASSERT_TRUE(infrb::parse(read_file(fixture_dir() + "/model.blob"), &blob, &err)) << err;
    ASSERT_EQ(7U, blob.nodes.size());
    ASSERT_EQ(12U, blob.slots.size());

    EXPECT_EQ(infrb::kOpInput, blob.nodes[0].op);
    EXPECT_EQ(infrb::kOpConv2d, blob.nodes[1].op);
    EXPECT_EQ(static_cast<std::uint8_t>(INFR_ACT_RELU6), blob.nodes[1].act)
        << "Clip(0,6) must arrive as relu6 on the conv that produces the tensor";
    EXPECT_EQ(infrb::kOpMeanHw, blob.nodes[2].op) << "ReduceMean([2,3], keepdims=1) is `mean_hw`";
    EXPECT_EQ(static_cast<std::uint8_t>(INFR_ACT_HARD_SIGMOID), blob.nodes[2].act)
        << "the HardSigmoid that squashes the pooled value rides on the pooling node";
    EXPECT_EQ(infrb::kOpMul, blob.nodes[3].op);
    EXPECT_EQ(1U, blob.nodes[3].p[0]) << "the squeeze-and-excite gate is the per-channel form";
    EXPECT_EQ(infrb::kOpMul, blob.nodes[4].op);
    EXPECT_EQ(1U, blob.nodes[4].p[0]) << "the stored gate uses the same form";
    EXPECT_EQ(infrb::kOpMeanHw, blob.nodes[5].op);
    EXPECT_EQ(infrb::kOpFc, blob.nodes[6].op);

    /* The two gates differ in where their numbers come from, which is exactly what a real
     * squeeze-and-excite block needs: the computed one is a runtime slot (`byte_len == 0`),
     * the stored one carries its payload in the blob. Both are per-channel, so an executor
     * that only handled one of them would pass a shape check and produce wrong numbers. */
    const auto& computed_gate = blob.slots[4];  /* `pooled` */
    EXPECT_EQ(0U, computed_gate.byte_len)
        << "the computed gate must be a runtime slot: a real SE block computes it";
    EXPECT_EQ(blob.nodes[3].w_slot, 4U) << "the per-channel mul reads the pooling node's slot";
    const auto& stored_gate = blob.slots[7];  /* `bias`, transposed to (1,1,1,3) */
    EXPECT_NE(0U, stored_gate.byte_len) << "the stored gate must carry its payload";

    /* The NCHW -> NHWC transposition is the one place a silent error can live, so the
     * shape is asserted rather than assumed: ONNX stores (3, 2, 3, 3), the blob (3, 3, 3, 2). */
    const auto& filter = blob.slots[2];
    ASSERT_EQ(4U, filter.rank);
    EXPECT_EQ(3U, filter.dims[0]);
    EXPECT_EQ(3U, filter.dims[1]);
    EXPECT_EQ(3U, filter.dims[2]);
    EXPECT_EQ(2U, filter.dims[3]) << "the input-channel axis must move last";
    ASSERT_EQ(4U, stored_gate.rank);
    EXPECT_EQ(1U, stored_gate.dims[0]);
    EXPECT_EQ(1U, stored_gate.dims[1]);
    EXPECT_EQ(1U, stored_gate.dims[2]);
    EXPECT_EQ(3U, stored_gate.dims[3]);
}

/* And it computes what an independently written numpy forward of the same ONNX graph says.
 * A tolerance rather than bit equality on purpose: the reference is written against the
 * ONNX semantics, not against our kernel's accumulation order, so demanding identical bits
 * would be testing the reference's luck. 1e-6 relative is well below what a transposition
 * error or a wrongly fused activation would produce. */
TEST(GraphBlobOnnxSubset, ReproducesTheNumpyForwardOfTheOnnxGraph) {
    infrb::Blob blob;
    std::string err;
    ASSERT_TRUE(infrb::parse(read_file(fixture_dir() + "/model.blob"), &blob, &err)) << err;

    const std::vector<float> input = read_floats(fixture_dir() + "/input.bin");
    const std::vector<float> expect = read_floats(fixture_dir() + "/expect.bin");
    ASSERT_FALSE(expect.empty());

    std::vector<float> arena(blob.arena_floats);
    const std::uint32_t source = infrb::input_slot(blob);
    ASSERT_NE(infrb::kNoSlot, source);
    ASSERT_EQ(input.size(), static_cast<std::size_t>(blob.slots[source].dims[0] *
                                                     blob.slots[source].dims[1] *
                                                     blob.slots[source].dims[2] *
                                                     blob.slots[source].dims[3]));
    std::memcpy(arena.data() + blob.slots[source].arena, input.data(),
                input.size() * sizeof(float));

    std::vector<unsigned char> workspace_bytes(1U << 20U);
    infr_workspace workspace;
    ASSERT_EQ(INFR_OK, infr_workspace_init(&workspace, workspace_bytes.data(), workspace_bytes.size()));
    ASSERT_TRUE(infrb::run(blob, arena.data(), arena.size(), &workspace, &err)) << err;

    const std::uint32_t target = infrb::output_slot(blob);
    ASSERT_EQ(expect.size(), static_cast<std::size_t>(blob.slots[target].dims[3]));
    infr_test::expect_float_buffers_near(expect.data(), arena.data() + blob.slots[target].arena,
                                        expect.size(), "onnx_subset/classifier");
    EXPECT_EQ(0U, infr_workspace_used(&workspace)) << "the operators must rewind the workspace";
}
