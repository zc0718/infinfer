// Conan::ImportStart
#include <gtest/gtest.h>

#include <cstdint>
#include <limits>

#include <infinfer/infinfer.h>
// Conan::ImportEnd

/*!
 * @brief [en] The tensor descriptor contract: derived element counts, checked indexing, byte sizes.
 * @brief [zh] 张量描述符契约：推导的元素个数、带检查的索引、字节数。
 * @since 0.1.0
 */

TEST(ContractTensor, DtypeSizesAreSpelledOut) {
    EXPECT_EQ(sizeof(float), infr_dtype_size(INFR_DTYPE_F32));
    EXPECT_EQ(1U, infr_dtype_size(INFR_DTYPE_I8));
    EXPECT_EQ(1U, infr_dtype_size(INFR_DTYPE_U8));
    EXPECT_EQ(2U, infr_dtype_size(INFR_DTYPE_I16));
    EXPECT_EQ(4U, infr_dtype_size(INFR_DTYPE_I32));
    // An unknown dtype must not look like a 1-byte dtype: 0 is the "not a dtype" value.
    EXPECT_EQ(0U, infr_dtype_size(static_cast<infr_dtype>(99)));
}

TEST(ContractTensor, ShapeDerivesElemCount) {
    const uint32_t nhwc[4] = {2U, 3U, 4U, 5U};
    infr_shape shape{};
    ASSERT_EQ(INFR_OK, infr_shape_init(&shape, 4U, nhwc));
    EXPECT_EQ(4U, shape.rank);
    EXPECT_EQ(2U * 3U * 4U * 5U, shape.elem_count);
    for (uint8_t axis = 0U; axis < 4U; ++axis) {
        EXPECT_EQ(nhwc[axis], shape.dims[axis]);
    }

    // Slots above `rank` are cleared, so a reused descriptor cannot carry stale extents.
    const uint32_t vector[1] = {7U};
    ASSERT_EQ(INFR_OK, infr_shape_init(&shape, 1U, vector));
    EXPECT_EQ(7U, shape.elem_count);
    for (uint8_t axis = 1U; axis < INFR_TENSOR_MAX_RANK; ++axis) {
        EXPECT_EQ(0U, shape.dims[axis]);
    }
}

TEST(ContractTensor, ShapeRejectsWhatWouldBecomeSilentEmptyOps) {
    infr_shape shape{};
    const uint32_t zero_axis[2] = {3U, 0U};
    const uint32_t blank[1] = {0U};
    const uint32_t ok_dims[1] = {1U};

    EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, infr_shape_init(nullptr, 1U, ok_dims));
    EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, infr_shape_init(&shape, 1U, nullptr));
    EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, infr_shape_init(&shape, 0U, blank));
    EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, infr_shape_init(&shape, INFR_TENSOR_MAX_RANK + 1U, ok_dims));
    EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, infr_shape_init(&shape, 2U, zero_axis));
}

TEST(ContractTensor, ShapeRejectsAnElemCountThatWouldWrap) {
    // 65536^4 wraps a 32-bit product to exactly zero, which would look like an empty
    // tensor instead of "this does not fit".
    const uint32_t big[4] = {65536U, 65536U, 65536U, 65536U};
    infr_shape shape{};
    EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, infr_shape_init(&shape, 4U, big));

    // The largest count that still fits is fine, one more column is not.
    const uint32_t fits[3] = {1024U, 1024U, 4095U};
    ASSERT_EQ(INFR_OK, infr_shape_init(&shape, 3U, fits));
    EXPECT_EQ(1024U * 1024U * 4095U, shape.elem_count);
}

TEST(ContractTensor, OffsetIsRowMajorAndBoundsChecked) {
    const uint32_t nhwc[4] = {2U, 3U, 4U, 5U};
    infr_shape shape{};
    ASSERT_EQ(INFR_OK, infr_shape_init(&shape, 4U, nhwc));

    uint32_t offset = 0U;
    const uint32_t first[4] = {0U, 0U, 0U, 0U};
    const uint32_t last[4] = {1U, 2U, 3U, 4U};
    const uint32_t one_row_down[4] = {0U, 1U, 0U, 0U};
    const uint32_t one_channel_over[4] = {0U, 0U, 0U, 1U};
    const uint32_t next_image[4] = {1U, 0U, 0U, 0U};

    ASSERT_EQ(INFR_OK, infr_shape_offset(&shape, first, &offset));
    EXPECT_EQ(0U, offset);
    ASSERT_EQ(INFR_OK, infr_shape_offset(&shape, one_channel_over, &offset));
    EXPECT_EQ(1U, offset);
    // W*C = 20 elements per row, not W: the innermost two axes move together.
    ASSERT_EQ(INFR_OK, infr_shape_offset(&shape, one_row_down, &offset));
    EXPECT_EQ(4U * 5U, offset);
    ASSERT_EQ(INFR_OK, infr_shape_offset(&shape, next_image, &offset));
    EXPECT_EQ(3U * 4U * 5U, offset);
    ASSERT_EQ(INFR_OK, infr_shape_offset(&shape, last, &offset));
    EXPECT_EQ(shape.elem_count - 1U, offset);

    // One past the end of any axis is an error, not the first element of the next one.
    const uint32_t past_c[4] = {0U, 0U, 0U, 5U};
    const uint32_t past_n[4] = {2U, 0U, 0U, 0U};
    EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, infr_shape_offset(&shape, past_c, &offset));
    EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, infr_shape_offset(&shape, past_n, &offset));
    EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, infr_shape_offset(&shape, first, nullptr));
    EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, infr_shape_offset(nullptr, first, &offset));

    // An uninitialised descriptor has no strides to derive.
    const infr_shape unset{};
    EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, infr_shape_offset(&unset, first, &offset));
}

TEST(ContractTensor, TensorViewBindsWithoutTakingOwnership) {
    const uint32_t shape_dims[2] = {3U, 4U};
    infr_shape shape{};
    ASSERT_EQ(INFR_OK, infr_shape_init(&shape, 2U, shape_dims));

    float storage[12] = {};
    infr_tensor tensor{};
    ASSERT_EQ(INFR_OK, infr_tensor_init(&tensor, storage, INFR_DTYPE_F32, &shape));
    EXPECT_EQ(static_cast<void*>(storage), tensor.data);
    EXPECT_EQ(INFR_DTYPE_F32, tensor.dtype);
    EXPECT_EQ(12U, tensor.shape.elem_count);
    EXPECT_EQ(INFR_QUANT_NONE, tensor.quant.kind);
    EXPECT_EQ(1.0F, tensor.quant.scale);
    EXPECT_EQ(12U * sizeof(float), infr_tensor_bytes(&tensor));

    // Re-viewing the same buffer as int8 changes the byte span, not the storage.
    ASSERT_EQ(INFR_OK, infr_tensor_init(&tensor, storage, INFR_DTYPE_I8, &shape));
    EXPECT_EQ(12U, infr_tensor_bytes(&tensor));
    EXPECT_EQ(static_cast<void*>(storage), tensor.data);
}

TEST(ContractTensor, TensorViewRejectsInvalidArgumentsAndReportsNoBytes) {
    const uint32_t shape_dims[1] = {4U};
    infr_shape shape{};
    ASSERT_EQ(INFR_OK, infr_shape_init(&shape, 1U, shape_dims));
    float storage[4] = {};
    infr_tensor tensor{};

    EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, infr_tensor_init(nullptr, storage, INFR_DTYPE_F32, &shape));
    EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, infr_tensor_init(&tensor, nullptr, INFR_DTYPE_F32, &shape));
    EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, infr_tensor_init(&tensor, storage, INFR_DTYPE_F32, nullptr));
    EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, infr_tensor_init(&tensor, storage, static_cast<infr_dtype>(99), &shape));

    const infr_shape unset{};
    EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, infr_tensor_init(&tensor, storage, INFR_DTYPE_F32, &unset));

    EXPECT_EQ(0U, infr_tensor_bytes(nullptr));
}


int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
