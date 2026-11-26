// Conan::ImportStart
#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include <infinfer/infinfer.h>

#include "parity_harness.hpp"
// Conan::ImportEnd

/*!
 * @brief [en] Graph-level KAT: the `k2_conv_micro` blob must reproduce the expectation
 *            through the four `composed` operators.
 * @brief [zh] 图级 KAT：`k2_conv_micro` blob 必须经四个 `composed` 算子复现期望输出。
 * @note [en] The blob and the expectation are produced by
 *            `workspace/qa/prep/emit_k2_blob.py` in float64 from the definition of each
 *            operator, so agreement means the composition is right, not that the code
 *            equals itself (PLAN §3.10). Bitwise cross-target equality is judged
 *            separately by the QEMU matrix.
 * @note [zh] blob 与期望由 `workspace/qa/prep/emit_k2_blob.py` 按各算子的定义在 float64
 *            下生成，所以"一致"说明组合是对的，而不是代码等于它自己（PLAN §3.10）。
 *            跨目标逐位相等由 QEMU 矩阵单独判定。
 * @since 0.1.0
 */

namespace {

constexpr char kMagic[8] = {'I', 'N', 'F', 'R', 'K', 'A', 'T', '\0'};
constexpr std::uint32_t kVersion = 1;
constexpr std::uint32_t kDtypeF32 = 1;

/* The graph is the KAT's definition, not a parameter: it is spelled out here in the
 * same order as `manifest.json`'s `graph` list, so a change to either has to change
 * both (and the blob). */
constexpr std::uint32_t kInH = 16;
constexpr std::uint32_t kInW = 16;
constexpr std::uint32_t kInC = 1;
constexpr std::uint32_t kConvOc = 8;
constexpr std::uint32_t kDwOc = 16;
constexpr std::uint32_t kFcOc = 10;

std::uint32_t read_u32(const std::vector<unsigned char>& bytes, std::size_t offset) {
    return static_cast<std::uint32_t>(bytes[offset]) |
           (static_cast<std::uint32_t>(bytes[offset + 1]) << 8) |
           (static_cast<std::uint32_t>(bytes[offset + 2]) << 16) |
           (static_cast<std::uint32_t>(bytes[offset + 3]) << 24);
}

std::vector<unsigned char> read_file(const std::string& path) {
    std::ifstream stream(path, std::ios::binary);
    EXPECT_TRUE(stream.good()) << "cannot open " << path;
    return std::vector<unsigned char>(std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>());
}

/* Blob reader for the documented v1 layout; any drift fails loudly instead of being
 * read as garbage floats. */
std::map<std::string, std::vector<float>> parse_blob(const std::vector<unsigned char>& bytes) {
    std::map<std::string, std::vector<float>> tensors;
    EXPECT_GE(bytes.size(), std::size_t{16});
    if (bytes.size() < 16) {
        return tensors;
    }
    EXPECT_EQ(0, std::memcmp(bytes.data(), kMagic, sizeof kMagic));
    EXPECT_EQ(kVersion, read_u32(bytes, 8));
    const std::uint32_t count = read_u32(bytes, 12);
    std::size_t offset = 16;
    for (std::uint32_t i = 0; i < count; ++i) {
        EXPECT_LE(offset + 24, bytes.size());
        if (offset + 24 > bytes.size()) {
            return tensors;
        }
        std::string name(reinterpret_cast<const char*>(bytes.data() + offset));
        const std::uint32_t dtype = read_u32(bytes, offset + 16);
        const std::uint32_t byte_len = read_u32(bytes, offset + 20);
        offset += 24;
        EXPECT_EQ(kDtypeF32, dtype) << "tensor " << name;
        EXPECT_EQ(0U, byte_len % sizeof(float)) << "tensor " << name;
        EXPECT_LE(offset + byte_len, bytes.size()) << "tensor " << name;
        if (offset + byte_len > bytes.size()) {
            return tensors;
        }
        std::vector<float> values(byte_len / sizeof(float));
        std::memcpy(values.data(), bytes.data() + offset, byte_len);
        tensors[name] = values;
        offset += byte_len;
    }
    EXPECT_EQ(bytes.size(), offset) << "trailing bytes after the declared tensors";
    return tensors;
}

/* A view over a caller-owned buffer: the KAT keeps the buffers, the operators borrow
 * them, and nothing here owns memory through the tensor. */
void bind_stage(infr_tensor* value, std::vector<float>* storage, const uint32_t dims[4]) {
    infr_shape shape{};
    EXPECT_EQ(INFR_OK, infr_shape_init(&shape, 4U, dims));
    EXPECT_EQ(INFR_OK, infr_tensor_init(value, storage->data(), INFR_DTYPE_F32, &shape));
}

/* Run relu through the registry: the activation is a kernel, and the graph layer must
 * reach it the same way it reaches the others. */
infr_status run_relu(float* data, uint32_t count) {
    const infr_kernel_desc* relu =
        infr_kernel_find("relu", INFR_DTYPE_F32, INFR_QUANT_NONE, INFR_TIER_SCALAR);
    if (relu == nullptr) {
        return INFR_ERR_NOT_IMPLEMENTED;
    }
    const void* inputs[1] = {data};
    void* outputs[1] = {data};
    return relu->fn(inputs, outputs, count, nullptr);
}

}  // namespace

TEST(GraphKatK2ConvMicro, ReproducesTheNumpyExpectation) {
    const std::string dir = std::string(RESOURCES_PATH) + "/kat/k2_conv_micro";
    const auto blob = read_file(dir + "/model.blob");
    const auto input_bytes = read_file(dir + "/input.bin");
    const auto expect_bytes = read_file(dir + "/expect_fp32.bin");

    const auto tensors = parse_blob(blob);
    for (const char* name : {"conv_w", "conv_b", "dw_w", "dw_b", "fc_w", "fc_b"}) {
        ASSERT_TRUE(tensors.count(name) == 1U) << "blob is missing tensor " << name;
    }
    ASSERT_EQ(static_cast<std::size_t>(kInH) * kInW * kInC * sizeof(float), input_bytes.size());
    ASSERT_EQ(static_cast<std::size_t>(kFcOc) * sizeof(float), expect_bytes.size());
    ASSERT_EQ(INFR_OK, infr_kernel_register_scalar());

    std::vector<float> input_data(kInH * kInW * kInC);
    std::memcpy(input_data.data(), input_bytes.data(), input_bytes.size());

    /* Stage buffers, sized from the graph. */
    std::vector<float> conv_out(kInH * kInW * kConvOc);
    std::vector<float> pool_out((kInH / 2) * (kInW / 2) * kConvOc);
    std::vector<float> dw_out((kInH / 2) * (kInW / 2) * kDwOc);
    std::vector<float> gap_out(kDwOc);
    std::vector<float> fc_out(kFcOc);

    const uint32_t in_dims[4] = {1U, kInH, kInW, kInC};
    const uint32_t conv_dims[4] = {1U, kInH, kInW, kConvOc};
    const uint32_t pool_dims[4] = {1U, kInH / 2U, kInW / 2U, kConvOc};
    const uint32_t dw_dims[4] = {1U, kInH / 2U, kInW / 2U, kDwOc};
    const uint32_t gap_dims[4] = {1U, 1U, 1U, kDwOc};
    const uint32_t fc_dims[4] = {1U, 1U, 1U, kFcOc};

    infr_tensor input_stage{};
    infr_tensor conv_stage{};
    infr_tensor pool_stage{};
    infr_tensor dw_stage{};
    infr_tensor gap_stage{};
    infr_tensor fc_stage{};
    bind_stage(&input_stage, &input_data, in_dims);
    bind_stage(&conv_stage, &conv_out, conv_dims);
    bind_stage(&pool_stage, &pool_out, pool_dims);
    bind_stage(&dw_stage, &dw_out, dw_dims);
    bind_stage(&gap_stage, &gap_out, gap_dims);
    bind_stage(&fc_stage, &fc_out, fc_dims);

    /* Blob payloads are const; the tensor view needs a writable pointer, so the weights
     * are copied once into owned buffers. */
    std::vector<float> conv_w = tensors.at("conv_w");
    std::vector<float> dw_w = tensors.at("dw_w");
    infr_shape conv_filter_shape{};
    const uint32_t conv_filter_dims[4] = {kConvOc, 3U, 3U, kInC};
    ASSERT_EQ(INFR_OK, infr_shape_init(&conv_filter_shape, 4U, conv_filter_dims));
    infr_tensor conv_filter{};
    ASSERT_EQ(INFR_OK, infr_tensor_init(&conv_filter, conv_w.data(), INFR_DTYPE_F32, &conv_filter_shape));
    infr_shape dw_filter_shape{};
    const uint32_t dw_filter_dims[4] = {kDwOc, 3U, 3U, 1U};
    ASSERT_EQ(INFR_OK, infr_shape_init(&dw_filter_shape, 4U, dw_filter_dims));
    infr_tensor dw_filter{};
    ASSERT_EQ(INFR_OK, infr_tensor_init(&dw_filter, dw_w.data(), INFR_DTYPE_F32, &dw_filter_shape));

    /* One workspace for the whole graph: every operator returns the cursor, so the
     * peak is the largest single operator, not the sum (INV-8). */
    alignas(8) uint8_t workspace_bytes[4096] = {};
    infr_workspace workspace{};
    ASSERT_EQ(INFR_OK, infr_workspace_init(&workspace, workspace_bytes, sizeof(workspace_bytes)));

    const infr_conv2d_params conv_params{3U, 3U, 1U, 1U, 1U, 1U, kInC, kConvOc, 1U};
    ASSERT_EQ(INFR_OK, infr_conv2d_f32(&input_stage, &conv_filter, tensors.at("conv_b").data(), &conv_params,
                                       &conv_stage, &workspace));
    ASSERT_EQ(INFR_OK, run_relu(conv_out.data(), kInH * kInW * kConvOc));

    const infr_pool_params pool_params{2U, 2U, 2U, 2U, 0U, 0U, kConvOc};
    ASSERT_EQ(INFR_OK, infr_max_pool2d_f32(&conv_stage, &pool_params, &pool_stage, &workspace));

    /* Depthwise: groups == in_channels, so the same operator serves both matrix rows. */
    const infr_conv2d_params dw_params{3U, 3U, 1U, 1U, 1U, 1U, kConvOc, kDwOc, kConvOc};
    ASSERT_EQ(INFR_OK, infr_conv2d_f32(&pool_stage, &dw_filter, tensors.at("dw_b").data(), &dw_params, &dw_stage,
                                       &workspace));
    ASSERT_EQ(INFR_OK, run_relu(dw_out.data(), (kInH / 2U) * (kInW / 2U) * kDwOc));

    ASSERT_EQ(INFR_OK, infr_mean_hw_f32(&dw_stage, &gap_stage, &workspace));

    /* The KAT's last layer: one dot product per output channel, which is exactly the
     * `fc` kernel. */
    const infr_kernel_desc* fc = infr_kernel_find("fc", INFR_DTYPE_F32, INFR_QUANT_NONE, INFR_TIER_SCALAR);
    ASSERT_NE(nullptr, fc);
    const std::vector<float> fc_w = tensors.at("fc_w");
    const std::vector<float> fc_b = tensors.at("fc_b");
    for (std::uint32_t oc = 0; oc < kFcOc; ++oc) {
        float acc = 0.0F;
        const float* fc_w_row = fc_w.data() + (static_cast<std::size_t>(oc) * kDwOc);
        const void* dot_inputs[2] = {gap_out.data(), fc_w_row};
        void* dot_output[1] = {&acc};
        ASSERT_EQ(INFR_OK, fc->fn(dot_inputs, dot_output, kDwOc, nullptr));
        fc_out[oc] = acc + fc_b[oc];
    }

    std::vector<float> expect(kFcOc);
    std::memcpy(expect.data(), expect_bytes.data(), expect_bytes.size());
    infr_test::expect_float_buffers_near(expect.data(), fc_out.data(), kFcOc, "k2_conv_micro");

    /* Every operator gave its scratch back: the graph ran in a constant workspace. */
    EXPECT_EQ(0U, infr_workspace_used(&workspace));
}
