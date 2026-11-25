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
    static const infr_kernel_desc builtins[] = {
        {"add", INFR_DTYPE_F32, INFR_QUANT_NONE, INFR_TIER_SCALAR, infr_scalar_add_f32},
        {"fc", INFR_DTYPE_F32, INFR_QUANT_NONE, INFR_TIER_SCALAR, infr_scalar_fc_f32},
        {"max", INFR_DTYPE_F32, INFR_QUANT_NONE, INFR_TIER_SCALAR, infr_scalar_max_f32},
        {"relu", INFR_DTYPE_F32, INFR_QUANT_NONE, INFR_TIER_SCALAR, infr_scalar_relu_f32},
        {"relu", INFR_DTYPE_I8, INFR_QUANT_NONE, INFR_TIER_SCALAR, infr_scalar_relu_i8},
        {"sum", INFR_DTYPE_F32, INFR_QUANT_NONE, INFR_TIER_SCALAR, infr_scalar_sum_f32},
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
