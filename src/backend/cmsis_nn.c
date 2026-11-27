// Conan::ImportStart
#include <stdint.h>
// Conan::ImportEnd

/*!
 * @ingroup group_kernel
 * @file
 * @brief [en] The vendor backend slot: compiled only when a vendor library is switched on, and producing nothing at all when it is off.
 * @brief [zh] 厂商后端槽位：只有当某个厂商库被打开时才编译，关闭时**一个符号都不产生**。
 * @note [en] Why an empty translation unit is the right shape for a switched-off backend. The
 *            gate that guards this (`.github/project/vendor_gate.py`) asserts that the archive
 *            contains none of this file's symbols when the option is off; if the file defined
 *            anything unconditionally, that assertion would pass while the object still carried
 *            code -- a check that cannot fail is worse than no check.
 * @note [zh] 关闭状态下为什么应该是一个空翻译单元。守护这一点的门禁
 *            （`.github/project/vendor_gate.py`）断言：选项关闭时归档里不含本文件的任何符号；
 *            如果本文件无条件定义了任何东西，这条断言会通过而目标文件里其实带着代码——
 *            一条不可能失败的自检比没有自检更糟。
 * @note [en] What is deliberately absent: there are no kernel bodies here yet. Registering rows
 *            at `INFR_TIER_VENDOR` without the vendor library would mean claiming a cell the
 *            matrix has not declared, and `registry_gate` refuses an orphan row -- correctly.
 *            So this increment delivers the slot and its proof; the bodies arrive with the
 *            dependency, and that dependency belongs to the first page of the task guides.
 * @note [zh] 这里**故意**还没有内核实现体。在没有厂商库的情况下就向 `INFR_TIER_VENDOR` 登记
 *            行，等于声称一个矩阵并未声明的格子，而 `registry_gate` 会拒绝孤立行——那是对的。
 *            所以这一增量只交付槽位与其证明；实现体随依赖一起到来，而那个依赖属于任务指南的
 *            第一页。
 * @since 0.2.0
 */

#if defined(INFR_WITH_CMSIS_NN)

/* The real header, provided by the CMSIS-NN package once its recipe exists. The gate compiles
 * this branch against a stub of its own making, which proves the guard -- not the library.
 *
 * 真实头文件，等 CMSIS-NN 的配方存在后由包提供。门禁用自己造的桩来编译这一分支，证明的是
 * **守卫**而不是那个库。 */
#include "arm_nnfunctions.h"

/**
 * @brief [en] Whether this build carries the CMSIS-NN backend.
 * @brief [zh] 本次构建是否带有 CMSIS-NN 后端。
 * @return [en] 1 when compiled with `INFR_WITH_CMSIS_NN`.
 * @return [zh] 用 `INFR_WITH_CMSIS_NN` 编译时返回 1。
 * @since 0.2.0
 */
int infr_cmsis_nn_backend_present(void) { return 1; }

#else

/* No symbols, by design. A typedef is the smallest legal way to keep the unit non-empty
 * without emitting anything -- an entirely empty translation unit is not valid C.
 *
 * 按设计不产生符号。typedef 是让翻译单元非空又不发出的最小合法写法——完全空的翻译单元
 * 不是合法的 C。 */
typedef int infr_cmsis_nn_tu_not_empty_t;

#endif
