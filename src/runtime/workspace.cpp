// Conan::ImportStart
#include <infinfer/runtime/workspace.h>
// Conan::ImportEnd

/*
 * Workspace contract, C++17 implementation behind a C ABI (PLAN 3.4: the runtime
 * leg is C++17; the header stays C11).
 *
 * As of 0.1.0 the bump allocator is not implemented: `infr_workspace_alloc`
 * reports INFR_ERR_NOT_IMPLEMENTED rather than pretending to succeed. Binding a
 * buffer and reporting usage are exact and safe to use now.
 *
 * 工作区契约：C11 头 + C++17 实现（PLAN 3.4：编排层用 C++17）。
 * 0.1.0 的 bump allocator 尚未实现，`infr_workspace_alloc` 如实返回
 * INFR_ERR_NOT_IMPLEMENTED，不伪装成功。
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
    (void)size_bytes;
    (void)alignment;
    return INFR_ERR_NOT_IMPLEMENTED; /* Lands with the offset table (PLAN P2). */
}

size_t infr_workspace_used(const infr_workspace* ws) { return ws == nullptr ? 0 : ws->used_bytes; }
