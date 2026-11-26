// Conan::ImportStart
#include "scalar_internal.h"
// Conan::ImportEnd

/*!
 * @ingroup group_scalar
 * @file
 * @brief [en] Single entry point that registers every scalar row compiled into this build.
 * @brief [zh] 注册本构建内全部 scalar 行的唯一入口。
 * @since 0.1.0
 */

/* A trimmed build (a scalar body not compiled) also loses its row here, so the
 * registry can never advertise a body that is not linked.
 *
 * 裁剪掉某个标量体时，对应行也消失，注册表不会宣称一个未链接的实现。 */

/**
 * @brief [en] Register the scalar reference rows; idempotent, safe to call at every init.
 * @brief [zh] 注册 scalar 参考行；幂等，可在每次初始化时调用。
 * @since 0.1.0
 */
infr_status infr_kernel_register_scalar(void) {
    /* Rows name their fields because `infr_kernel_desc` is public and may grow: a
     * positional row would shift silently the moment a field is inserted, which is
     * exactly the defect that stayed invisible on ARM until a stricter toolchain
     * caught it in a consumer. The initform gate holds this file to it.
     *
     * 行里写明字段名，因为 `infr_kernel_desc` 是公开类型、会增长：位置式写法在插入
     * 字段的那一刻就会静默错位——这正是那个在 ARM 上一直看不见、直到更严格的工具链
     * 在消费方抓到它的缺陷。initform 门禁把这条约束加在本文件上。 */
    static const infr_kernel_desc builtins[] = {
        {.name = "add",
         .dtype = INFR_DTYPE_F32,
         .quant = INFR_QUANT_NONE,
         .tier = INFR_TIER_SCALAR,
         .fn = infr_scalar_add_f32},
        {.name = "mul",
         .dtype = INFR_DTYPE_F32,
         .quant = INFR_QUANT_NONE,
         .tier = INFR_TIER_SCALAR,
         .fn = infr_scalar_mul_f32},
        {.name = "fc",
         .dtype = INFR_DTYPE_F32,
         .quant = INFR_QUANT_NONE,
         .tier = INFR_TIER_SCALAR,
         .fn = infr_scalar_fc_f32},
        {.name = "fc",
         .dtype = INFR_DTYPE_I8,
         .quant = INFR_QUANT_PER_TENSOR,
         .tier = INFR_TIER_SCALAR,
         .fn = infr_scalar_fc_i8},
        /* int16 is not a wider int8 with more headroom: its accumulator has to be int64 (one
         * worst-case term is already about 2^30, so int32 overflows from the third), so the two
         * quantized cells are separate rows rather than one body with a cast. The pair is declared
         * together in matrix.yml for the same reason -- what is legal is one fact, and it lives in
         * one place.
         *
         * int16 不是“更宽的 int8”：它的累加器必须是 int64（单项最坏情况已约 2^30，int32 从第三项
         * 起就溢出），所以两个量化格是两行，而不是一个体加一次强制转换。两者在 matrix.yml 里成对
         * 声明也是同一个理由——合法组合是一个事实，只存一处。 */
        {.name = "fc",
         .dtype = INFR_DTYPE_I16,
         .quant = INFR_QUANT_PER_TENSOR,
         .tier = INFR_TIER_SCALAR,
         .fn = infr_scalar_fc_i16},
        {.name = "max",
         .dtype = INFR_DTYPE_F32,
         .quant = INFR_QUANT_NONE,
         .tier = INFR_TIER_SCALAR,
         .fn = infr_scalar_max_f32},
        {.name = "relu",
         .dtype = INFR_DTYPE_F32,
         .quant = INFR_QUANT_NONE,
         .tier = INFR_TIER_SCALAR,
         .fn = infr_scalar_relu_f32},
        {.name = "relu",
         .dtype = INFR_DTYPE_I8,
         .quant = INFR_QUANT_NONE,
         .tier = INFR_TIER_SCALAR,
         .fn = infr_scalar_relu_i8},
        /* Quantize and dequantize are kernels like any other: the graph layer has to be
         * able to ask for them by name, otherwise "quantization is a first-class citizen"
         * only holds for the bodies the KAT happens to call directly.
         *
         * 量化与反量化就是普通内核：图层必须能按名字要到它们，否则“量化是一等公民”
         * 只成立于 KAT 恰好直接调用的那几个体上。 */
        {.name = "quantize",
         .dtype = INFR_DTYPE_F32,
         .quant = INFR_QUANT_PER_TENSOR,
         .tier = INFR_TIER_SCALAR,
         .fn = infr_scalar_quantize_f32},
        {.name = "dequantize",
         .dtype = INFR_DTYPE_I8,
         .quant = INFR_QUANT_PER_TENSOR,
         .tier = INFR_TIER_SCALAR,
         .fn = infr_scalar_dequantize_i8},
        {.name = "scale",
         .dtype = INFR_DTYPE_F32,
         .quant = INFR_QUANT_NONE,
         .tier = INFR_TIER_SCALAR,
         .fn = infr_scalar_scale_f32},
        {.name = "sum",
         .dtype = INFR_DTYPE_F32,
         .quant = INFR_QUANT_NONE,
         .tier = INFR_TIER_SCALAR,
         .fn = infr_scalar_sum_f32},
    };
    uint32_t i;
    infr_status status;

    for (i = 0; i < (uint32_t)(sizeof(builtins) / sizeof(builtins[0])); ++i) {
        status = infr_kernel_register(&builtins[i]);
        if (status != INFR_OK) {
            return status;
        }
    }
    return INFR_OK;
}
