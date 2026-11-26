// Conan::ImportStart
#pragma once
#include <stddef.h>
#include <stdint.h>

#include <infinfer/kernel/kernel.h>
// Conan::ImportEnd

// C linkage, because this header is included from C and from the C++ core.
// 见 kernel.h 的说明：头文件自带保护，树内与装机两种消费方式一致。
#ifdef __cplusplus
extern "C" {
#endif

/*!
 * @ingroup group_runtime
 * @file workspace.h
 *
 * @brief [en] The reusable tensor region (`EVAL` mode): one caller-owned buffer,
 *            byte-granular bump allocation, no heap.
 * @brief [zh] 可复用张量区（`EVAL` 模式）：调用方持有的单块缓冲区、按字节的
 *            指针式分配、不碰堆。
 * @note [en] The three tensor legs are kept apart: `workspace` (reuse),
 *            `activation` (retain for backward) and `state` (persist). This header
 *            defines the reusable leg only.
 * @note [zh] 三类张量是分开的：`workspace`（复用）、`activation`（为反向
 *            保留）、`state`（持久）。本头只定义"复用"这条腿。
 * @note [en] Offsets are bytes, never words. The reference defect this contract
 *            exists to make testable is an offset table that mixed the two units.
 * @note [zh] 偏移一律按字节、不按字。本契约要让其可测的参考缺陷，正是一张把两种
 *            单位混算的偏移表。
 * @since 0.1.0
 */

/**
 * @brief [en] One reusable region bound to a caller-owned buffer.
 * @brief [zh] 绑定到调用方缓冲区的可复用区域。
 * @since 0.1.0
 */
typedef struct infr_workspace {
    /**
     * @brief [en] Buffer start.
     * @brief [zh] 缓冲区起点。
     */
    uint8_t *base;
    /**
     * @brief [en] Total size in bytes.
     * @brief [zh] 缓冲区总字节数。
     */
    size_t capacity_bytes;
    /**
     * @brief [en] Bytes handed out so far.
     * @brief [zh] 已分配字节数。
     */
    size_t used_bytes;
} infr_workspace;

/**
 * @brief [en] Bind a workspace to a buffer and reset its cursor.
 * @brief [zh] 把工作区绑定到缓冲区并重置游标。
 * @param ws [en] Workspace to initialise; must not be NULL.
 * @param ws [zh] 要初始化的工作区；不得为 NULL。
 * @param buffer [en] Caller-owned storage; NULL is allowed only with size_bytes == 0.
 * @param buffer [zh] 调用方持有的存储；仅当 size_bytes == 0 时允许 NULL。
 * @param size_bytes [en] Capacity in bytes.
 * @param size_bytes [zh] 容量，字节。
 * @return INFR_OK, or INFR_ERR_INVALID_ARGUMENT for a NULL workspace or a NULL
 *         buffer with a non-zero size.
 * @since 0.1.0
 */
infr_status infr_workspace_init(infr_workspace *ws, void *buffer, size_t size_bytes);

/**
 * @brief [en] Reserve size_bytes from the workspace, aligned to `alignment`.
 * @brief [zh] 从工作区预留 size_bytes 字节，按 `alignment` 对齐。
 * @param ws [en] Workspace to allocate from.
 * @param ws [zh] 从中分配的工作区。
 * @param size_bytes [en] Requested size in bytes.
 * @param size_bytes [zh] 请求的字节数。
 * @param alignment [en] Power-of-two alignment in bytes (1, 2, 4, 8, ...).
 * @param alignment [zh] 对齐字节数，必须是 2 的幂（1, 2, 4, 8, ...）。
 * @param out [en] Receives the reservation; untouched on failure.
 * @param out [zh] 接收预留区；失败时不改动。
 * @return INFR_OK; INFR_ERR_INVALID_ARGUMENT for NULL arguments or a non-power-of-two
 *         alignment; INFR_ERR_BUFFER_TOO_SMALL when the capacity cannot hold the
 *         request (including alignment padding).
 * @note [en] Alignment applies to the absolute address of the returned pointer, not
 *            to the offset inside the buffer: a misaligned base is padded to the next
 *            multiple of `alignment`.
 * @note [zh] 对齐针对返回指针的**绝对地址**，不是缓冲区内的偏移：基址不对齐时先补齐
 *            到 `alignment` 的下一个倍数。
 * @note [en] Consecutive allocations are placed back to back in bytes: the second
 *            8-byte reservation from a 16-byte buffer starts at offset 8, not offset 32.
 * @note [zh] 连续分配按字节紧排：16 字节缓冲区里第二次 8 字节预留从偏移 8 开始，
 *            而不是偏移 32。
 * @since 0.1.0
 */
infr_status infr_workspace_alloc(infr_workspace *ws, size_t size_bytes, size_t alignment, void **out);

/**
 * @brief [en] Bytes handed out so far, including alignment padding.
 * @brief [zh] 已分配字节数（含对齐填充）。
 * @param ws [en] Workspace; NULL yields 0.
 * @param ws [zh] 工作区；NULL 返回 0。
 * @return Bytes used, or 0 for a NULL workspace.
 * @since 0.1.0
 */
size_t infr_workspace_used(const infr_workspace *ws);

/**
 * @brief [en] Snapshot of the cursor, for a later `infr_workspace_release`.
 * @brief [zh] 游标快照，供之后的 `infr_workspace_release` 使用。
 * @param ws [en] Workspace; NULL yields 0.
 * @param ws [zh] 工作区；NULL 返回 0。
 * @return The current `used_bytes`, or 0 for a NULL workspace.
 * @note [en] A mark is only meaningful for the same workspace it came from: it is a
 *            byte offset, not a reservation handle.
 * @note [zh] mark 只对产生它的那个工作区有意义：它是一个字节偏移，不是保留区句柄。
 * @since 0.1.0
 */
size_t infr_workspace_mark(const infr_workspace *ws);

/**
 * @brief [en] Give back everything allocated since `mark`, so scratch can be reused.
 * @brief [zh] 归还 `mark` 之后分配的全部空间，以便临时缓冲被复用。
 * @param ws [en] Workspace to rewind.
 * @param ws [zh] 要回退的工作区。
 * @param mark [en] Value previously returned by `infr_workspace_mark` on `ws`.
 * @param mark [zh] 此前由 `infr_workspace_mark` 在 `ws` 上返回的值。
 * @return INFR_OK; INFR_ERR_INVALID_ARGUMENT for a NULL workspace or a mark larger
 *         than the current `used_bytes` (which would move the cursor forwards and
 *         hand out memory that was already handed out).
 * @note [en] This is what lets an L2 operator borrow scratch for a gather buffer and
 *            return it before the next node runs: `EVAL` must not grow the workspace
 *            as the graph executes.
 * @note [zh] 有了它，L2 算子可以借一块临时缓冲做 gather，并在下一个节点前还回去：
 *            `EVAL` 不许随图的执行而长胖。
 * @since 0.1.0
 */
infr_status infr_workspace_release(infr_workspace *ws, size_t mark);

#ifdef __cplusplus
}
#endif
