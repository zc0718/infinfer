// Conan::ImportStart
#include <infinfer/kernel/kernel.h>
// Conan::ImportEnd

#include <string.h>

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

/*
 * Fixed-capacity registry. Static storage only (no dynamic allocation, MCU
 * friendly), linear search over a table small enough that the cost is noise.
 * Registration happens at init time, single-threaded; see kernel.h.
 *
 * 定容量注册表：只用静态存储（不动态分配，适配 MCU），线性查找。
 * 注册在初始化阶段单线程进行；见 kernel.h。
 */

#define INFR_KERNEL_TABLE_CAPACITY 64U

static infr_kernel_desc g_kernel_table[INFR_KERNEL_TABLE_CAPACITY];
static uint32_t g_kernel_count = 0U;

infr_status infr_kernel_register(const infr_kernel_desc *kernel) {
    uint32_t i;

    if (kernel == NULL || kernel->name == NULL || kernel->fn == NULL) {
        return INFR_ERR_INVALID_ARGUMENT;
    }
    for (i = 0U; i < g_kernel_count; ++i) {
        const infr_kernel_desc *row = &g_kernel_table[i];
        if (row->dtype == kernel->dtype && row->quant == kernel->quant && row->tier == kernel->tier &&
            strcmp(row->name, kernel->name) == 0) {
            /* Idempotent re-registration is fine; a conflicting body is not. */
            return row->fn == kernel->fn ? INFR_OK : INFR_ERR_INVALID_ARGUMENT;
        }
    }
    if (g_kernel_count >= INFR_KERNEL_TABLE_CAPACITY) {
        return INFR_ERR_BUFFER_TOO_SMALL;
    }
    g_kernel_table[g_kernel_count] = *kernel;
    ++g_kernel_count;
    return INFR_OK;
}

const infr_kernel_desc *infr_kernel_find(const char *name, infr_dtype dtype, infr_quant_kind quant, infr_tier tier) {
    uint32_t i;

    if (name == NULL) {
        return NULL;
    }
    for (i = 0U; i < g_kernel_count; ++i) {
        const infr_kernel_desc *row = &g_kernel_table[i];
        if (row->dtype == dtype && row->quant == quant && row->tier == tier && strcmp(row->name, name) == 0) {
            return row;
        }
    }
    return NULL; /* Absent is a fact the caller must handle, never an error to paper over. */
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
