// Conan::ImportStart
#include <infinfer/runtime/workspace.h>
// Conan::ImportEnd

/**
 * @file
 * @brief [en] Workspace contract: byte-granular bump allocator behind a C ABI, C++17 implementation.
 * @brief [zh] 工作区契约：C ABI 背后的按字节 bump 分配器，C++17 实现。
 * @since 0.1.0
 */

/* Alignment is applied to the ABSOLUTE address of the reservation (a misaligned
 * base is padded), and the padding is counted in `used_bytes`. Offsets are bytes,
 * never words -- that is the defect `contract_workspace_bounds.cpp` pins.
 *
 * 对齐作用于保留区的**绝对地址**（基址不对齐就补齐），补齐字节计入
 * `used_bytes`。偏移一律按字节、绝不按字——这正是 `contract_workspace_bounds.cpp`
 * 钉住的缺陷。 */

/**
 * @brief [en] Bind a workspace to a caller-owned buffer; zero state is valid only after this call.
 * @brief [zh] 把工作区绑定到调用方提供的缓冲；只有调用过它之后状态才算有效。
 * @since 0.1.0
 */
infr_status infr_workspace_init(infr_workspace* ws, void* buffer, size_t size_bytes) {
    if (ws == nullptr) {
        return INFR_ERR_INVALID_ARGUMENT;
    }
    if (buffer == nullptr && size_bytes != 0U) {
        return INFR_ERR_INVALID_ARGUMENT;
    }
    ws->base = static_cast<uint8_t*>(buffer);
    ws->capacity_bytes = size_bytes;
    ws->used_bytes = 0;
    return INFR_OK;
}

/**
 * @brief [en] Reserve `size_bytes` with the requested alignment; never reuses a previous reservation.
 * @brief [zh] 按请求的对齐保留 `size_bytes`；永不复用之前的保留区。
 * @since 0.1.0
 */
infr_status infr_workspace_alloc(infr_workspace* ws, size_t size_bytes, size_t alignment, void** out) {
    if (ws == nullptr || out == nullptr) {
        return INFR_ERR_INVALID_ARGUMENT;
    }
    if (alignment == 0U || (alignment & (alignment - 1U)) != 0U) {
        return INFR_ERR_INVALID_ARGUMENT;
    }
    if (ws->base == nullptr) {
        /* An empty workspace can only satisfy a zero-byte request. */
        if (size_bytes == 0U) {
            *out = nullptr;
            return INFR_OK;
        }
        return INFR_ERR_BUFFER_TOO_SMALL;
    }

    const auto base = reinterpret_cast<uintptr_t>(ws->base);
    const uintptr_t current = base + ws->used_bytes;
    const uintptr_t aligned = (current + alignment - 1U) & ~static_cast<uintptr_t>(alignment - 1U);
    const size_t padding = static_cast<size_t>(aligned - current);
    if (padding > ws->capacity_bytes - ws->used_bytes) {
        return INFR_ERR_BUFFER_TOO_SMALL;
    }
    const size_t consumed = ws->used_bytes + padding;
    if (size_bytes > ws->capacity_bytes - consumed) {
        return INFR_ERR_BUFFER_TOO_SMALL;
    }
    /* Pointer arithmetic from the original pointer: no integer-to-pointer cast. */
    *out = ws->base + consumed;
    ws->used_bytes = consumed + size_bytes;
    return INFR_OK;
}

/**
 * @brief [en] Bytes handed out so far, padding included; zero for a null workspace.
 * @brief [zh] 至今已分配的字节数（含补齐）；空工作区返回 0。
 * @since 0.1.0
 */
size_t infr_workspace_used(const infr_workspace* ws) { return ws == nullptr ? 0 : ws->used_bytes; }
