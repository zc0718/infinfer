// Conan::ImportStart
#pragma once
#include <infinfer/kernel/kernel.h>
// Conan::ImportEnd

/*
 * k0_actquant: the smallest chain that exercises the kernels this tree actually
 * ships (quantize -> dequantize -> relu -> quantize). It exists to prove the KAT
 * *pipeline* end to end -- baked input, baked expectation, in-target compare,
 * exit code -- before any graph interpreter exists. The PLAN's k1_mlp3 needs a
 * Gemm cell that has not landed yet; when it does, k0 stays as the cheap smoke.
 *
 * k0_actquant：用本树已有的算子（量化→反量化→relu→量化）串出的最小链。
 * 目的是在网络解释器出现之前，把 KAT 管线本身（烘焙输入/烘焙期望/目标内比对/
 * 退出码）先证明可用。PLAN 的 k1_mlp3 需要尚未落地的 Gemm 格；落地后 k0 仍作为
 * 最便宜的冒烟保留。
 */

#define KAT_K0_LEN 8u

/** Fixed input, shared by the emitter (host) and every target: the two never
 *  disagree about the numbers because they compile the same constants.
 *
 *  固定输入，宿主生成器与所有目标共用同一份常量，二者不可能对不上。 */
extern const float kat_k0_input[KAT_K0_LEN];

/** Per-tensor quantization parameters for both quantize steps.
 *
 *  两步量化共用的逐张量量化参数。 */
extern const infr_quant kat_k0_quant;

/** Result buffers the chain produces; caller owns the storage so the driver can
 *  place them in static memory (no heap on bare metal).
 *
 *  链的输出缓冲；由调用方提供存储，便于裸机放进静态区（不依赖堆）。 */
typedef struct kat_k0_result {
    int8_t q1[KAT_K0_LEN];  /* After the first quantize. 第一次量化后。 */
    float d1[KAT_K0_LEN];   /* After dequantize. 反量化后。 */
    float relu[KAT_K0_LEN]; /* After relu. relu 后。 */
    int8_t q2[KAT_K0_LEN];  /* After the second quantize (the checked tensor). 第二次量化后（被校验的张量）。 */
    float d2[KAT_K0_LEN];   /* After the second dequantize (also checked). 第二次反量化后（同样被校验）。 */
} kat_k0_result;

/** Run the chain. Returns INFR_OK or the first non-OK status from a cell.
 *
 *  运行整条链；返回 INFR_OK 或第一个非 OK 的状态。 */
infr_status kat_k0_run(kat_k0_result *out);

/** Checked bytes of k0: the int8 tensor followed by the float tensor's raw
 *  bytes. Same shape as k1's interface so one driver serves both chains.
 *
 *  k0 的被校验字节：先是 int8 张量，再是浮点张量的裸字节。与 k1 的接口同形，
 *  所以一个驱动可以服务两条链。 */
unsigned kat_k0_expect_len(void);
void kat_k0_expect(const kat_k0_result *result, unsigned char *dst);
