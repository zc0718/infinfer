// Conan::ImportStart
#include "scalar_internal.h"
// Conan::ImportEnd

/*
 * Single entry point for every scalar reference row compiled into this build.
 * A trimmed build (a scalar body not compiled) also loses its row here, so the
 * registry can never advertise a body that is not linked.
 *
 * 本构建内全部 scalar 参考行的唯一注册入口：裁剪掉某个标量体时，对应行也消失，
 * 注册表不会宣称一个未链接的实现。
 */

infr_status infr_kernel_register_scalar(void) {
    static const infr_kernel_desc builtins[] = {
        {"relu", INFR_DTYPE_F32, INFR_QUANT_NONE, INFR_TIER_SCALAR, infr_scalar_relu_f32},
        {"relu", INFR_DTYPE_I8, INFR_QUANT_NONE, INFR_TIER_SCALAR, infr_scalar_relu_i8},
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
