// Conan::ImportStart
#include <stdint.h>

#include "arm_m_internal.h"
// Conan::ImportEnd

/*!
 * @ingroup group_arm_m
 * @file
 * @brief [en] Cortex-M byte-parallel ReLU (`SADD8` + `SEL`) for int8 tensors.
 * @brief [zh] int8 张量的 Cortex-M 按字节并行 ReLU（`SADD8` + `SEL`）。
 * @since 0.1.0
 */

#if defined(__ARM_FEATURE_SIMD32)

/**
 * @brief [en] ReLU over an int8 tensor, four bytes per iteration, on the SIMD32 part of the DSP
 *        extension.
 * @brief [zh] 用 DSP 扩展的 SIMD32 部分做 int8 张量的 ReLU，每轮四个字节。
 * @param inputs [en] `inputs[0]` = source tensor.
 * @param inputs [zh] `inputs[0]` = 源张量。
 * @param outputs [en] `outputs[0]` = destination tensor (may not alias the source).
 * @param outputs [zh] `outputs[0]` = 目的张量（不得与源重叠）。
 * @param element_count [en] Number of int8 elements.
 * @param element_count [zh] int8 元素个数。
 * @param quant [en] Unused: this cell is unquantized, so the parameters carry no meaning.
 * @param quant [zh] 未使用：本格不带量化，参数没有含义。
 * @return [en] INFR_OK, or INFR_ERR_INVALID_ARGUMENT on a null pointer.
 * @return [zh] INFR_OK，指针为空时返回 INFR_ERR_INVALID_ARGUMENT。
 * @note [en] Why this is a different body and not a rewrite of the scalar loop: ARMv7E-M has no
 *            `vmax.s8`, but the DSP extension gives the pair `SADD8` (sets the GE flags per byte
 *            to "the signed result is non-negative") and `SEL` (picks each byte by its GE flag).
 *            With a zero addend that is exactly `x > 0 ? x : 0` for all 256 byte values,
 *            including `INT8_MIN`, and it is bit-identical to the reference rather than close:
 *            the operation selects bytes, it never rounds.
 * @note [zh] 为什么这是一具新实体而不是把标量循环改写：ARMv7E-M 没有 `vmax.s8`，但 DSP 扩展
 *            提供 `SADD8`（按字节把 GE 标志置为「有符号结果非负」）与 `SEL`（按各字节的 GE
 *            标志二选一）。加数为零时，对全部 256 个字节值都恰好是 `x > 0 ? x : 0`（含
 *            `INT8_MIN`），而且与参考实现是**逐位**相同而非接近：它是按字节选择，不做舍入。
 * @note [en] Why the two instructions are one assembly block: the GE flags are the interface
 *            between them, so nothing may run in between. Written as two intrinsics, `-O0` puts
 *            four loads and stores between the `SADD8` and the `SEL` (measured); the block makes
 *            the coupling explicit and keeps the `-O0` ISA measurement honest, which is the
 *            optimisation level `isa_gate` uses for hand-written tiers.
 * @note [zh] 为什么这两条指令放在同一个汇编块里：GE 标志就是它们之间的接口，中间不能插入任何
 *            东西。写成两条 intrinsic 时，`-O0` 会在 `SADD8` 与 `SEL` 之间塞进四条读写
 *            （实测）；写成一块既把耦合写明白，也让 `-O0` 的 ISA 测量保持诚实——那正是
 *            `isa_gate` 量手写档位所用的优化档位。
 * @note [en] Tail handling is scalar on purpose: the reference is element-wise, so four bytes per
 *            word leaves at most three, and a masked path would need a lane count the extension
 *            does not offer.
 * @note [zh] 尾部刻意用标量：参考实现是逐元素的，一个机器字四个字节最多剩三个，而掩码路径
 *            需要扩展并不提供的车道计数。
 * @since 0.1.0
 */
infr_status infr_arm_m_relu_i8(const void *const *inputs, void *const *outputs, uint32_t element_count,
                               const infr_quant *quant) {
    const int8_t *in;
    int8_t *out;
    const uint8_t *bytes_in;
    uint8_t *bytes_out;
    const uint32_t zero = 0U;
    uint32_t i = 0U;

    (void)quant; /* Unquantized instance: parameters carry no meaning here. */
    if (inputs == NULL || outputs == NULL || inputs[0] == NULL || outputs[0] == NULL) {
        return INFR_ERR_INVALID_ARGUMENT;
    }
    in = (const int8_t *)inputs[0];
    out = (int8_t *)outputs[0];

    bytes_in = (const uint8_t *)in;
    bytes_out = (uint8_t *)out;
    /* Four bytes at a time. The loads are assembled byte by byte, so the kernel never asks the
     * caller for an alignment the tensor contract does not promise.
     *
     * 每轮四个字节。四个字节是逐字节装进机器字的，所以内核不会向调用方索要张量契约并未承诺的
     * 对齐。 */
    for (; i + 4U <= element_count; i += 4U) {
        uint32_t word = (uint32_t)bytes_in[i] | ((uint32_t)bytes_in[i + 1U] << 8) | ((uint32_t)bytes_in[i + 2U] << 16) |
                        ((uint32_t)bytes_in[i + 3U] << 24);
        uint32_t masked;

        __asm__ volatile("sadd8 %1, %1, %2\n\tsel %0, %1, %2" : "=r"(masked), "+r"(word) : "r"(zero) : "cc");
        bytes_out[i] = (uint8_t)(masked & 0xFFU);
        bytes_out[i + 1U] = (uint8_t)((masked >> 8) & 0xFFU);
        bytes_out[i + 2U] = (uint8_t)((masked >> 16) & 0xFFU);
        bytes_out[i + 3U] = (uint8_t)((masked >> 24) & 0xFFU);
    }
    /* Tail: at most three elements, and the reference is per element. */
    for (; i < element_count; ++i) {
        out[i] = in[i] > 0 ? in[i] : 0;
    }
    return INFR_OK;
}

#endif /* __ARM_FEATURE_SIMD32 */
