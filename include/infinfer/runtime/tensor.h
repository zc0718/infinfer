// Conan::ImportStart
#pragma once
#include <stddef.h>
#include <stdint.h>

#include <infinfer/kernel/kernel.h>
// Conan::ImportEnd

// C linkage, because this header is included from C and from the C++ core.
#ifdef __cplusplus
extern "C" {
#endif

/*!
 * @ingroup group_runtime
 * @file tensor.h
 *
 * @brief [en] The L2 tensor descriptor: rank, dims, dtype and quant in one value,
 *            with the strides derived instead of stored.
 * @brief [zh] L2 张量描述符：把 rank、dims、dtype、quant 放在一起，步长由 dims
 *            推导而不单独存储。
 * @note [en] Shape lives here and nowhere else. Kernels take (pointer, length) only
 *, so the moment an operator needs to know "how wide is a row"
 *            it has to come through this header -- that is what keeps shape out of
 *            the L1 signatures.
 * @note [zh] shape 只活在这里。kernel 只收（指针、长度），所以算子一旦需
 *            要知道"一行有多宽"，就必须经过本头——这正是 L1 签名里没有 shape 的
 *            原因。
 * @note [en] Dims are row-major, outermost first: a 4-D tensor is N,H,W,C and the
 *            innermost axis is the fastest. Strides are never stored, because a
 *            stored stride is a second source of truth for the same fact.
 * @note [zh] dims 行主序、最外层在前：4 维张量是 N,H,W,C，最内轴变化最快。步长
 *            一律不存——存下来的步长就是同一事实的第二份真相。
 * @since 0.1.0
 */

/**
 * @brief [en] Largest rank the descriptor represents (N,H,W,C).
 * @brief [zh] 描述符能表示的最大阶数（N,H,W,C）。
 * @note [en] A macro, because this is a C11 header and the alternatives both fail a
 *            gate: `constexpr` is not C, and an enum whose only job is to hold the
 *            value 4 trips the enum-size check. The NOLINT below records that choice.
 * @note [zh] 用宏，是因为这是 C11 头，两种替代方案各会被一道门禁挡住：`constexpr`
 *            不是 C，而只为了装下 4 的枚举会被 enum-size 检查判红。下面的 NOLINT
 *            记录了这个选择。
 * @since 0.1.0
 */
#define INFR_TENSOR_MAX_RANK 4U /* NOLINT(cppcoreguidelines-macro-usage): C11 header, see the note above */

/**
 * @brief [en] Rank, dims and the element count that follows from them.
 * @brief [zh] 阶数、各维长度，以及由它们推出来的元素个数。
 * @since 0.1.0
 */
typedef struct infr_shape {
    /**
     * @brief [en] Extent of each axis, outermost first; entries above `rank` are 0.
     * @brief [zh] 各轴长度，最外层在前；`rank` 以上的项为 0。
     */
    uint32_t dims[INFR_TENSOR_MAX_RANK];
    /**
     * @brief [en] Number of axes in use, 1..INFR_TENSOR_MAX_RANK; 0 means "not initialised".
     * @brief [zh] 使用的轴数，1..INFR_TENSOR_MAX_RANK；0 表示"未初始化"。
     */
    uint8_t rank;
    /**
     * @brief [en] Product of `dims[0..rank)`, always derived, never assigned directly.
     * @brief [zh] `dims[0..rank)` 的乘积，永远推导得出，绝不直接赋值。
     */
    uint32_t elem_count;
} infr_shape;

/**
 * @brief [en] A tensor as a data pointer plus its descriptor.
 * @brief [zh] 一个张量 = 数据指针 + 描述符。
 * @since 0.1.0
 */
typedef struct infr_tensor {
    /**
     * @brief [en] Element 0 of the buffer; the layout is the `shape` row-major order.
     * @brief [zh] 缓冲区第 0 个元素；布局即 `shape` 的行主序。
     */
    void *data;
    /**
     * @brief [en] Descriptor of the data behind `data`.
     * @brief [zh] `data` 背后数据的描述符。
     */
    infr_shape shape;
    /**
     * @brief [en] Element type.
     * @brief [zh] 元素类型。
     */
    infr_dtype dtype;
    /**
     * @brief [en] Quantisation parameters; `kind == INFR_QUANT_NONE` for unquantized data.
     * @brief [zh] 量化参数；未量化数据为 `kind == INFR_QUANT_NONE`。
     */
    infr_quant quant;
} infr_tensor;

/**
 * @brief [en] Bytes per element for a dtype; 0 for an unknown dtype.
 * @brief [zh] 某 dtype 每个元素的字节数；未知 dtype 返回 0。
 * @param [in] dtype Element type.
 * @return Bytes per element, or 0 when `dtype` is not one of the declared values.
 * @since 0.1.0
 */
size_t infr_dtype_size(infr_dtype dtype);

/**
 * @brief [en] Fill a shape from a rank and its dims, deriving `elem_count`.
 * @brief [zh] 用阶数和各维长度填写 shape，并推导 `elem_count`。
 * @param [out] shape Descriptor to fill; must not be NULL.
 * @param [in] rank Number of axes, 1..INFR_TENSOR_MAX_RANK.
 * @param [in] dims Extent of each axis, outermost first; must not be NULL.
 * @return INFR_OK; INFR_ERR_INVALID_ARGUMENT for a NULL argument, rank 0 or above
 *         INFR_TENSOR_MAX_RANK, a zero-length axis, or an element count that does
 *         not fit in 32 bits.
 * @note [en] Rejecting a zero-length axis is deliberate: an empty tensor is a graph
 *            bug, and letting one through turns it into a silent identity op
 *            somewhere downstream. The one place emptiness is legal is the
 *            reduction *length* passed to a kernel, which is not a shape.
 * @note [zh] 拒绝零长度轴是故意的：空张量是图构造错误，放过去就会在下游变成一次
 *            静默的空操作。唯一允许"空"的地方是传给 kernel 的归约*长度*，那不是
 *            shape。
 * @since 0.1.0
 */
infr_status infr_shape_init(infr_shape *shape, uint8_t rank, const uint32_t *dims);

/**
 * @brief [en] Row-major element offset of a multi-dimensional index.
 * @brief [zh] 多维索引对应的行主序元素偏移。
 * @param [in] shape Descriptor; must not be NULL.
 * @param [in] index `shape->rank` indices, outermost first; must not be NULL.
 * @param [out] out_offset Receives the element offset; untouched on failure.
 * @return INFR_OK; INFR_ERR_INVALID_ARGUMENT for a NULL argument, an uninitialised
 *         shape, or an index equal to or beyond that axis.
 * @note [en] Bounds are checked here rather than at the call site, so a gather that
 *            walks off the end of a tensor is an error return, not a read of the
 *            next allocation.
 * @note [zh] 越界检查放在这里而不是调用点：这样"走出张量末尾"的 gather 会变成一次
 *            错误返回，而不是读到下一块分配。
 * @since 0.1.0
 */
infr_status infr_shape_offset(const infr_shape *shape, const uint32_t *index, uint32_t *out_offset);

/**
 * @brief [en] Bind a buffer, dtype and shape into a tensor view.
 * @brief [zh] 把缓冲区、dtype 与 shape 绑成一个张量视图。
 * @param [out] tensor View to fill; must not be NULL.
 * @param [in] data Buffer holding `shape->elem_count` elements of `dtype`; not NULL.
 * @param [in] dtype Element type.
 * @param [in] shape Descriptor to copy in; must not be NULL and must be initialised.
 * @return INFR_OK; INFR_ERR_INVALID_ARGUMENT for a NULL argument, an uninitialised
 *         shape, or an unknown dtype.
 * @note [en] The view borrows the buffer: it neither copies nor owns it, which is
 *            what lets the same memory be an input and an output of two different
 *            nodes.
 * @note [zh] 视图只借用缓冲区：不复制也不拥有，这正是同一块内存能同时是不同节点的
 *            输入与输出的原因。
 * @since 0.1.0
 */
infr_status infr_tensor_init(infr_tensor *tensor, void *data, infr_dtype dtype, const infr_shape *shape);

/**
 * @brief [en] Byte size of the data a tensor view spans.
 * @brief [zh] 张量视图所跨数据的字节数。
 * @param [in] tensor View; NULL or an unknown dtype yields 0.
 * @return `elem_count * infr_dtype_size(dtype)`, or 0 for an invalid view.
 * @since 0.1.0
 */
size_t infr_tensor_bytes(const infr_tensor *tensor);

#ifdef __cplusplus
}
#endif
