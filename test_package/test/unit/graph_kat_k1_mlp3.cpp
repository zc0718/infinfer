// Conan::ImportStart
#include <gtest/gtest.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <vector>

#include <infinfer/infinfer.h>

#include "parity_harness.hpp"
// Conan::ImportEnd

/*!
 * @brief [en] Graph-level KAT: the `k1_mlp3` blob (fc 16->32, relu, fc 32->4) must reproduce the expectation.
 * @brief [zh] 图级 KAT：`k1_mlp3` blob（fc 16→32、relu、fc 32→4）必须复现期望输出。
 * @note [en] The blob and the expectation are produced by
 *      `.github/project/qa/prep/emit_k1_blob.py` with numpy, so agreement means the kernels are right,
 *      not that the code equals itself (PLAN §3.10: generation is a maintainer step, CI only
 *      reads the bytes). Bitwise cross-target equality is judged separately by the QEMU matrix.
 * @note [zh] blob 与期望由 `.github/project/qa/prep/emit_k1_blob.py` 用 numpy 生成，所以"一致"说明
 *      内核是对的，而不是代码等于它自己（PLAN §3.10：生成归维护者，CI 只读字节）。跨目标逐位
 *      相等由 QEMU 矩阵单独判定。
 * @since 0.1.0
 */

namespace {

constexpr char kMagic[8] = {'I', 'N', 'F', 'R', 'K', 'A', 'T', '\0'};
constexpr std::uint32_t kVersion = 2;
constexpr std::uint32_t kDtypeF32 = 1;
constexpr std::size_t kIn = 16;
constexpr std::size_t kHidden = 32;
constexpr std::size_t kOut = 4;

struct Tensor {
    std::string name;
    std::vector<float> values;
};

std::uint32_t read_u32(const std::vector<unsigned char>& bytes, std::size_t offset) {
    return static_cast<std::uint32_t>(bytes[offset]) |
           (static_cast<std::uint32_t>(bytes[offset + 1]) << 8) |
           (static_cast<std::uint32_t>(bytes[offset + 2]) << 16) |
           (static_cast<std::uint32_t>(bytes[offset + 3]) << 24);
}

std::vector<unsigned char> read_file(const std::string& path) {
    std::ifstream stream(path, std::ios::binary);
    EXPECT_TRUE(stream.good()) << "cannot open " << path;
    return std::vector<unsigned char>(std::istreambuf_iterator<char>(stream),
                                      std::istreambuf_iterator<char>());
}

// Blob reader for the documented v2 layout. The node table is skipped: this test
// re-spells the two nodes in C, so a wrong node list in the blob cannot hide behind a
// runner that obeys it (`graph_blob_k2.cpp` is the one that obeys the nodes). k1's
// input is a constant slot, which is why nothing here needs `input.bin`.
std::vector<Tensor> parse_blob(const std::vector<unsigned char>& bytes) {
    std::vector<Tensor> tensors;
    EXPECT_GE(bytes.size(), std::size_t{24});
    if (bytes.size() < 24) {
        return tensors;
    }
    EXPECT_EQ(0, std::memcmp(bytes.data(), kMagic, sizeof kMagic));
    EXPECT_EQ(kVersion, read_u32(bytes, 8));
    const std::uint32_t node_count = read_u32(bytes, 12);
    const std::uint32_t slot_count = read_u32(bytes, 16);
    std::size_t offset = 24 + (static_cast<std::size_t>(node_count) * 36U);
    EXPECT_LE(offset, bytes.size()) << "node table runs past the end of the blob";
    if (offset > bytes.size()) {
        return tensors;
    }
    for (std::uint32_t i = 0; i < slot_count; ++i) {
        EXPECT_LE(offset + 44, bytes.size());
        if (offset + 44 > bytes.size()) {
            break;
        }
        Tensor tensor;
        tensor.name.assign(reinterpret_cast<const char*>(bytes.data() + offset));
        const std::uint32_t dtype = read_u32(bytes, offset + 16);
        const std::uint32_t length = read_u32(bytes, offset + 40);
        offset += 44;
        EXPECT_EQ(kDtypeF32, dtype) << "slot " << tensor.name;
        EXPECT_LE(offset + length, bytes.size());
        EXPECT_EQ(0U, length % sizeof(float));
        if (length == 0) {
            continue; /* runtime slot: this test reads only the constants */
        }
        const std::size_t n = length / sizeof(float);
        tensor.values.resize(n);
        std::memcpy(tensor.values.data(), bytes.data() + offset, length);
        offset += length;
        tensors.push_back(std::move(tensor));
    }
    return tensors;
}

const Tensor* find(const std::vector<Tensor>& tensors, const char* name) {
    for (const Tensor& tensor : tensors) {
        if (tensor.name == name) {
            return &tensor;
        }
    }
    return nullptr;
}

const infr_kernel_desc* row(const char* name) {
    EXPECT_EQ(INFR_OK, infr_kernel_register_scalar());
    return infr_kernel_find(name, INFR_DTYPE_F32, INFR_QUANT_NONE, INFR_TIER_SCALAR);
}

infr_status dot(const infr_kernel_desc* fc, const float* x, const float* w, std::size_t length,
                float* out) {
    const void* inputs[2] = {x, w};
    void* outputs[1] = {out};
    return fc->fn(inputs, outputs, static_cast<std::uint32_t>(length), nullptr);
}

}  // namespace

TEST(GraphKatK1Mlp3, ReproducesNumpyExpectation) {
    const std::string root = std::string(RESOURCES_PATH) + "/kat/k1_mlp3/";
    const std::vector<Tensor> tensors = parse_blob(read_file(root + "model.blob"));
    const Tensor* x = find(tensors, "x");
    const Tensor* w1 = find(tensors, "w1");
    const Tensor* w2 = find(tensors, "w2");
    ASSERT_NE(nullptr, x);
    ASSERT_NE(nullptr, w1);
    ASSERT_NE(nullptr, w2);
    ASSERT_EQ(kIn, x->values.size());
    ASSERT_EQ(kHidden * kIn, w1->values.size());
    ASSERT_EQ(kOut * kHidden, w2->values.size());

    // The expectation lives in its own file: a model blob carries the model, and
    // keeping the two apart is what makes tampering with the expectation a real
    // check instead of a no-op.
    const std::vector<unsigned char> expect_bytes = read_file(root + "expect_fp32.bin");
    ASSERT_EQ(kOut * sizeof(float), expect_bytes.size());
    float expect_y2[kOut] = {};
    std::memcpy(expect_y2, expect_bytes.data(), expect_bytes.size());

    const infr_kernel_desc* fc = row("fc");
    const infr_kernel_desc* relu = row("relu");
    ASSERT_NE(nullptr, fc) << "fc/fp32/none/scalar is not registered";
    ASSERT_NE(nullptr, relu) << "relu/fp32/none/scalar is not registered";

    // fc 16->32: one kernel call per output row, which is exactly the L2 loop the
    // model needs (INV-3 keeps shape out of the kernel signature).
    float hidden[kHidden] = {};
    for (std::size_t r = 0; r < kHidden; ++r) {
        ASSERT_EQ(INFR_OK, dot(fc, x->values.data(), w1->values.data() + r * kIn, kIn, &hidden[r]));
    }
    const void* relu_in[1] = {hidden};
    void* relu_out[1] = {hidden};
    ASSERT_EQ(INFR_OK, relu->fn(relu_in, relu_out, static_cast<std::uint32_t>(kHidden), nullptr));

    float y2[kOut] = {};
    for (std::size_t r = 0; r < kOut; ++r) {
        ASSERT_EQ(INFR_OK, dot(fc, hidden, w2->values.data() + r * kHidden, kHidden, &y2[r]));
    }

    infr_test::expect_float_buffers_near(expect_y2, y2, kOut, "k1_mlp3 output",
                                         /*rel_tol=*/1e-5, /*abs_tol=*/1e-6);
}
