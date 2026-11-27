// Conan::ImportStart
#include <string.h>

#include "scalar_internal.h"
// Conan::ImportEnd

/*!
 * @ingroup group_scalar
 * @file
 * @brief [en] `y[i] = x[i]` over a run of elements: the primitive the layout operators are built on.
 * @brief [zh] 对一段元素做 `y[i] = x[i]`：布局类算子的底座。
 * @note [en] Why a kernel and not a `memcpy` in each operator. The composed operators are not
 *            allowed to reach the registry directly -- they name the kernels they use, and the
 *            gate checks those names -- so a data-movement operator still has to be built out of
 *            something that is declared. `copy` is that something, and it is the smallest kernel
 *            in the tree on purpose.
 * @note [en] The declared cell is `scalar` only, and that is not laziness: a copy is
 *            memory-bound, the reference body is already `memcpy`, and every compiler turns that
 *            into the widest move the target has. A hand-written vector body here would be the
 *            same instruction with more code around it (ADR-007). `scalar` is therefore the
 *            **best** implementation for this cell, not the fallback.
 * @note [zh] 为什么它是个内核而不是各算子自己 `memcpy`。组合算子不允许直接访问注册表——它们
 *            点名自己用的内核，而门禁会核对这些名字——所以一个纯搬运算子也必须由**被声明过**
 *            的东西拼出来。`copy` 就是这个东西，而且它刻意是树里最小的内核。
 * @note [zh] 只声明 `scalar` 一档，不是偷懒：拷贝是内存受限的，参考体本来就是 `memcpy`，任何
 *            编译器都会把它展开成目标上最宽的搬移指令。在这里手写向量体只会得到同一条指令、
 *            外加更多周边代码（ADR-007）。所以 `scalar` 对这一格是**最优**实现，不是后备。
 * @since 0.2.0
 */

/**
 * @brief [en] Copy `element_count` floats from `inputs[0]` to `outputs[0]`.
 * @brief [zh] 把 `element_count` 个 float 从 `inputs[0]` 拷到 `outputs[0]`。
 * @note [en] The two buffers must not overlap. That is stated rather than tested because a
 *            caller that overlaps them has a layout bug, and `memcpy` on overlapping ranges is
 *            undefined -- `memmove` would hide the bug and cost the compiler its widest path.
 * @note [zh] 两个缓冲区不得重叠。这一条是**写下来的约定**而不是运行时检查：调用了重叠区间的
 *            调用方已经有布局缺陷，而 `memcpy` 对重叠区间是未定义行为——改用 `memmove` 只会
 *            掩盖缺陷并让编译器失去最宽的路径。
 * @since 0.2.0
 */
infr_status infr_scalar_copy_f32(const void *const *inputs, void *const *outputs, uint32_t element_count,
                                 const infr_quant *quant) {
    (void)quant; /* Unquantized instance: parameters carry no meaning here. */
    if (inputs == NULL || outputs == NULL || inputs[0] == NULL || outputs[0] == NULL) {
        return INFR_ERR_INVALID_ARGUMENT;
    }
    if (element_count == 0U) {
        return INFR_OK;
    }
    memcpy(outputs[0], inputs[0], (size_t)element_count * sizeof(float));
    return INFR_OK;
}
