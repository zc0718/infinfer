// Conan::ImportStart
#include <infinfer/kernel/kernel.h>
// Conan::ImportEnd

/*
 * Contract stubs. Every function here answers with the truth as of 0.1.0:
 * the contract exists, the implementation does not. Returning INFR_OK from an
 * empty body is the defect this file refuses to reproduce (P0.5 case 3).
 *
 * 契约桩。这里每个函数都如实回答 0.1.0 的事实：契约存在、实现未到。
 * 空实现返回 INFR_OK 正是本文件拒绝复现的缺陷（P0.5 第 3 个用例）。
 */

infr_act_f32_fn infr_activation_f32(infr_activation act) {
    (void)act; /* No tier ships activation bodies yet; NULL is the honest answer. */
    return NULL;
}

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

infr_status infr_quantize_i8(const float *in, int8_t *out, uint32_t n, const infr_quant *quant) {
    if (in == NULL || out == NULL || quant == NULL) {
        return INFR_ERR_INVALID_ARGUMENT;
    }
    if (quant->kind == INFR_QUANT_NONE) {
        return INFR_ERR_INVALID_ARGUMENT;
    }
    (void)n; /* A quantizer that writes nothing must not claim success. */
    return INFR_ERR_NOT_IMPLEMENTED;
}

infr_status infr_dequantize_i8(const int8_t *in, float *out, uint32_t n, const infr_quant *quant) {
    if (in == NULL || out == NULL || quant == NULL) {
        return INFR_ERR_INVALID_ARGUMENT;
    }
    if (quant->kind == INFR_QUANT_NONE) {
        return INFR_ERR_INVALID_ARGUMENT;
    }
    (void)n; /* Same rule as infr_quantize_i8. */
    return INFR_ERR_NOT_IMPLEMENTED;
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
