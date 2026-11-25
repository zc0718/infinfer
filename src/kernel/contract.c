// Conan::ImportStart
#include <infinfer/kernel/kernel.h>
// Conan::ImportEnd

#include <string.h>

/*!
 * @ingroup group_contract
 * @file
 * @brief [en] Contract glue that is not a kernel body: activation-code validation, the registry, status naming.
 * @brief [zh] 不是内核体的契约粘合：激活码校验、注册表、状态名。
 * @since 0.1.0
 */

/* Activation, quantize and workspace bodies live in src/kernel/scalar and
 * src/runtime respectively; a body that is not implemented must report
 * INFR_ERR_NOT_IMPLEMENTED, never INFR_OK.
 *
 * 激活、量化与工作区的实体分别在 src/kernel/scalar 与 src/runtime；未实现的体
 * 一律返回 INFR_ERR_NOT_IMPLEMENTED，不返回 INFR_OK。 */

/**
 * @brief [en] Validate a fused-activation code and convert it to the enum.
 * @brief [zh] 校验融合激活码并转换为枚举值。
 * @since 0.1.0
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
/* No initializer on purpose: `= 0U` would put this in `.data`, and on a bare-metal
 * image `.data` only holds its values if a startup routine copies it out of FLASH.
 * A counter that happens to be garbage before registration then reads past the
 * table and takes the whole image down (found by the k0 KAT on mps2-an505: the
 * fault landed on `&g_kernel_table[0]`, i.e. "called through a garbage row").
 * `.bss` is the right home: it is the section startup zeroes, and this file keeps
 * *no* mutable initialised globals at all.
 *
 * 刻意不写初始化器：`= 0U` 会把它放进 `.data`，而裸机镜像里的 `.data` 只有在
 * 启动代码把它从 FLASH 搬过去之后才有值。注册前计数是垃圾值 → 越过表尾读 → 整个
 * 镜像挂掉（k0 KAT 在 mps2-an505 上发现的：故障 PC 落在 `&g_kernel_table[0]`，
 * 即“从一个垃圾行里取函数指针调用”）。它的家在 `.bss`：那是启动代码清零的段，
 * 本文件也从此**不**持有任何可变且带初值的全局对象。 */
static uint32_t g_kernel_count;

/**
 * @brief [en] Register one kernel row; idempotent for an identical body, rejects a conflicting one.
 * @brief [zh] 注册一行内核；同一个体重复注册幂等，冲突的体一律拒绝。
 * @since 0.1.0
 */
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

/**
 * @brief [en] Find a registered row by (name, dtype, quant, tier); NULL means absent, not an error.
 * @brief [zh] 按（名字、dtype、quant、tier）查找已注册的行；NULL 表示没有，不是错误。
 * @since 0.1.0
 */
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

/**
 * @brief [en] Stable textual name of a status code, for logs and test failure messages.
 * @brief [zh] 状态码的稳定文本名，供日志与测试失败信息使用。
 * @since 0.1.0
 */
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
