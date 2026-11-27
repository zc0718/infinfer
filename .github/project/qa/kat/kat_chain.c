// Conan::ImportStart
#include <string.h>

#include "kat_chain.h"
// Conan::ImportEnd
/*
 * The chain itself. Everything goes through the public surface a consumer has:
 * the registry (for relu) and the declared quantize helpers. Nothing here knows
 * what a target it is running on, which is what makes the same bytes reaching
 * every target meaningful.
 *
 * 链本体。全部经由消费方可见的公开面：注册表（取 relu）与声明的量化助手。
 * 这里不出现任何"目标"概念——正因如此，"同一份字节到达每个目标"才有意义。
 */

const float kat_k0_input[KAT_K0_LEN] = {1.5F, -2.25F, 0.0F, 3.75F, -0.5F, 7.125F, -8.0F, 0.25F};

/* Designated initialisers, not positional ones. The positional version was a
 * real defect: when `requant_multiplier` was inserted into `infr_quant`, every
 * field after it shifted by one, so `channel_zero_points` was initialised from
 * `1u` and `channel_count` from `0u`. The ARM legs compiled it with a warning and
 * still "passed" (the per-tensor path never reads those fields); the RISC-V legs
 * turned the same line into `-Wint-conversion` and the build failed. Naming the
 * fields makes the next insertion a non-event instead of a silent reinterpretation.
 *
 * 用指定初始化而非位置式。位置式那一版是**真缺陷**：`infr_quant` 插入
 * `requant_multiplier` 后，其后每个字段都错位一格，`channel_zero_points` 由 `1u`
 * 初始化、`channel_count` 拿到 `0u`。ARM 腿只告警、照样"通过"（逐张量路径不读这些
 * 字段），RISC-V 腿把同一行变成 `-Wint-conversion`，构建直接失败。写出字段名之后，
 * 下一次插入只是无害事件，而不再是一次静默重解释。 */
const infr_quant kat_k0_quant = {
    .kind = INFR_QUANT_PER_TENSOR,
    .scale = 0.05F,
    .zero_point = 3,
    .requant_multiplier = 0.0F,
    .channel_scales = NULL,
    .channel_zero_points = NULL,
    .channel_count = 1u,
    .axis = 0u,
};

/* Stage cut for bisecting a target that faults mid-chain without a debugger.
 * Build with -DKAT_STAGE=n and the chain returns OK right after stage n, so the
 * exit code says how far it got: 11 (expectation mismatch) means the stage was
 * reached, a lockup means it was not. Stepping under QEMU is not an option here:
 * single-stepping this build corrupts the instruction stream, so the only
 * trustworthy bisection axis is "how far did it get before dying".
 *
 * 阶段切点：给"镜像会在链中间出错、但没有可用调试器"的目标做二分用。
 * 加 `-DKAT_STAGE=n` 编译，链在第 n 阶段后直接返回 OK，于是退出码就代表"走到哪"：
 * 11（期望不符）说明该阶段到达了，锁死说明没到。这里不能靠单步——本机单步会破坏
 * 指令流，可信的二分轴只有"死之前走完了多少"。 */
#if defined(KAT_STAGE)
#define KAT_STAGE_CUT(n) do { if ((KAT_STAGE) <= (n)) { return INFR_OK; } } while (0)
#else
#define KAT_STAGE_CUT(n) do { } while (0)
#endif

infr_status kat_k0_run(kat_k0_result *out) {
    const infr_kernel_desc *relu;
    const void *in_ptr[1];
    void *out_ptr[1];
    infr_status status;

    if (out == 0) {
        return INFR_ERR_INVALID_ARGUMENT;
    }
    KAT_STAGE_CUT(0);
    status = infr_kernel_register_scalar();
    if (status != INFR_OK) {
        return status;
    }
    KAT_STAGE_CUT(1);
    relu = infr_kernel_find("relu", INFR_DTYPE_F32, INFR_QUANT_NONE, INFR_TIER_SCALAR);
    if (relu == 0) {
        return INFR_ERR_NOT_IMPLEMENTED;
    }
    KAT_STAGE_CUT(2);

    status = infr_quantize_i8(kat_k0_input, out->q1, KAT_K0_LEN, &kat_k0_quant);
    if (status != INFR_OK) {
        return status;
    }
    KAT_STAGE_CUT(3);
    status = infr_dequantize_i8(out->q1, out->d1, KAT_K0_LEN, &kat_k0_quant);
    if (status != INFR_OK) {
        return status;
    }
    KAT_STAGE_CUT(4);
    in_ptr[0] = out->d1;
    out_ptr[0] = out->relu;
    status = relu->fn(in_ptr, out_ptr, KAT_K0_LEN, 0);
    if (status != INFR_OK) {
        return status;
    }
    KAT_STAGE_CUT(5);
    status = infr_quantize_i8(out->relu, out->q2, KAT_K0_LEN, &kat_k0_quant);
    if (status != INFR_OK) {
        return status;
    }
    KAT_STAGE_CUT(6);
    return infr_dequantize_i8(out->q2, out->d2, KAT_K0_LEN, &kat_k0_quant);
}

unsigned kat_k0_expect_len(void) { return KAT_K0_LEN + (unsigned)sizeof(float) * KAT_K0_LEN; }

void kat_k0_expect(const kat_k0_result *result, unsigned char *dst) {
    memcpy(dst, result->q2, KAT_K0_LEN);
    memcpy(dst + KAT_K0_LEN, result->d2, sizeof(result->d2));
}
