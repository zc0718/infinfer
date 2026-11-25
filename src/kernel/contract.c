// Conan::ImportStart
#include <infinfer/kernel/kernel.h>
// Conan::ImportEnd

/*
 * Contract glue that is not a kernel body: status naming, activation-code
 * validation, and the registry stubs (P2.1). Activation, quantize and workspace
 * bodies live in src/kernel/scalar and src/runtime respectively; a body that is
 * not implemented must report INFR_ERR_NOT_IMPLEMENTED, never INFR_OK.
 *
 * 非内核体的契约粘合：状态名、激活码校验、注册表桩（P2.1）。激活、量化与
 * 工作区的实现在 src/kernel/scalar 与 src/runtime；未实现的体一律返回
 * INFR_ERR_NOT_IMPLEMENTED，不返回 INFR_OK。
 */

infr_status infr_activation_from_code(uint32_t code, infr_activation *out) {
    if (out == NULL) {
        return INFR_ERR_INVALID_ARGUMENT;
    }
    if (code >= (uint32_t)INFR_ACT_COUNT) {
        return INFR_ERR_UNSUPPORTED;
    }
    *out = (infr_activation)code;
    return INFR_OK;
}

infr_status infr_kernel_register(const infr_kernel_desc *kernel) {
    if (kernel == NULL || kernel->name == NULL || kernel->fn == NULL) {
        return INFR_ERR_INVALID_ARGUMENT;
    }
    return INFR_ERR_NOT_IMPLEMENTED; /* Registry lands with PLAN P2.1. */
}

const infr_kernel_desc *infr_kernel_find(const char *name, infr_dtype dtype, infr_quant_kind quant, infr_tier tier) {
    (void)name;
    (void)dtype;
    (void)quant;
    (void)tier;
    return NULL; /* Empty registry: nothing can be found, and nothing pretends otherwise. */
}

const char *infr_status_str(infr_status status) {
    switch (status) {
    case INFR_OK:
        return "INFR_OK";
    case INFR_ERR_INVALID_ARGUMENT:
        return "INFR_ERR_INVALID_ARGUMENT";
    case INFR_ERR_UNSUPPORTED:
        return "INFR_ERR_UNSUPPORTED";
    case INFR_ERR_BUFFER_TOO_SMALL:
        return "INFR_ERR_BUFFER_TOO_SMALL";
    case INFR_ERR_NOT_IMPLEMENTED:
        return "INFR_ERR_NOT_IMPLEMENTED";
    default:
        return "INFR_STATUS_UNKNOWN";
    }
}
