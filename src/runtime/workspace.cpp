// Conan::ImportStart
#include <infinfer/runtime/workspace.h>
// Conan::ImportEnd

/*
 * Workspace contract, C++17 implementation behind a C ABI (PLAN 3.4: the runtime
 * leg is C++17; the header stays C11).
 *
 * The allocator is a byte-granular bump pointer. Alignment is applied to the
 * ABSOLUTE address of the reservation (a misaligned base is padded), and the
 * padding is counted in `used_bytes`. Offsets are bytes, never words -- that is
 * the defect `contract_workspace_bounds.cpp` pins.
 *
 * 工作区契约：C11 头 + C++17 实现（PLAN 3.4：编排层用 C++17）。
 * 分配器是按字节的 bump 指针；对齐作用于**绝对地址**，补齐字节计入 `used_bytes`。
 * 偏移一律按字节——这正是 `contract_workspace_bounds.cpp` 钉住的缺陷。
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

size_t infr_workspace_used(const infr_workspace* ws) { return ws == nullptr ? 0 : ws->used_bytes; }
