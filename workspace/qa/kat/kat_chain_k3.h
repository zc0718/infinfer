// Conan::ImportStart
#pragma once
#include <infinfer/kernel/kernel.h>
// Conan::ImportEnd

/*
 * k3_tier_parity: every `done` cell of one tier compared against the scalar row **on the target**.
 *
 * Why this chain exists. The tier parity tests in `test_package/test/unit/` are compiled for the
 * target but only run on the host, so on a target they are skipped and the claim "the NEON body
 * agrees with the reference" rested on a probe run by hand on one machine (gaps G-73). This chain
 * closes that hole with the KAT machinery that already runs on eight QEMU targets: the reference
 * and the tier row are called in the same process, on the same input, and compared there.
 *
 * The shape is deliberately different from k0/k1/k2: those bake an expectation on the host and
 * compare bytes on the target, which needs the host to be the reference. Here the reference is
 * the scalar row **on the same target**, so there is no expectation to bake and the answer is a
 * verdict, not a byte string.
 *
 * The cell list is not written here: it is generated from `src/kernel/matrix.yml` for the one tier
 * this image was built for (`generated/kat_k3_cells.h`), so the chain cannot claim a cell the
 * matrix does not and cannot skip one the matrix does.
 *
 * k3_tier_parity：把某一档位的**每个** `done` 格在**目标上**与参考行比较。存在理由见 gaps G-73：
 * 目标侧 parity 用例只在宿主上跑过，目标上一直是 skip，于是"NEON 与参考一致"靠的是某台机器上
 * 手跑的探针。本链复用已跑通 8 个 QEMU 目标的 KAT 机器，把参考行与档位行放在**同一进程、同一
 * 输入**里当场比较。它与 k0/k1/k2 的形状不同：那几条在宿主上烘焙期望、在目标上比字节；这里的
 * 参考是**同目标的 scalar 行**，所以没有期望可烘焙，产物只有判定。格清单不写在这里，由
 * `matrix.yml` 生成为单档位表，链既不能多声称一格，也不能少测一格。
 */

/** How a kernel is called. Generated, one per cell. 内核怎么调，逐格生成。 */
typedef enum kat_k3_call {
    KAT_K3_CALL_UNARY = 0,  /* in[0] -> out[0]: relu */
    KAT_K3_CALL_BINARY = 1, /* in[0], in[1] -> out[0]: add, two real operands */
    KAT_K3_CALL_SCALE = 2,  /* in[0], in[1] = one broadcast factor */
    KAT_K3_CALL_REDUCE = 3, /* in[0] -> out[0] = one reduced value: sum, max */
    KAT_K3_CALL_DOT = 4,    /* in[0], in[1] -> out[0]: fp32 dot product */
    KAT_K3_CALL_DOT_Q = 5   /* in[0], in[1], quant -> out[0]: integer dot product */
} kat_k3_call;

/** The verdict for a cell. 一格的判据。 */
typedef enum kat_k3_cmp {
    KAT_K3_CMP_BITWISE = 0, /* bit equality: one IEEE op per element, or exact integers */
    KAT_K3_CMP_WINDOW = 1   /* relative window: an accumulator split differs from the reference */
} kat_k3_cmp;

/** One generated row: kernel name, its dtype/quant, how to call it, how to judge it. */
typedef struct kat_k3_cell {
    const char *kernel;
    infr_dtype dtype;
    infr_quant_kind quant_kind;
    int call;
    int cmp_mode;
} kat_k3_cell;

/** What the chain found, so the driver can encode a failure in its exit code without printing
 *  (stdout is not safe on every target: semihosting printf blocks with no console attached).
 *
 *  链的结果：驱动据此把失败编进退出码而不打印（不是每个目标都能打印：无控制台时半主机
 *  printf 会阻塞）。 */
typedef struct kat_k3_result {
    unsigned compared;  /* Cells that were registered and compared. 实际比较过的格数。 */
    unsigned missing;   /* Cells the matrix calls done but the tier did not register. 声称 done 却没登记。 */
    unsigned mismatched;/* First disagreeing cell index, or ~0u. 第一个不一致的格下标，无则 ~0u。 */
    unsigned length;    /* Length of the corpus element that disagreed. 出错的语料长度。 */
    unsigned long elements; /* Output elements actually compared. 真正比较过的输出元素个数。 */
} kat_k3_result;

/** Run every generated cell: scalar row vs tier row, same input, same process.
 *
 *  逐格运行：scalar 行与档位行，同一输入、同一进程。 */
infr_status kat_k3_run(kat_k3_result *out);

/** The window used for accumulating cells (KAT_K3_CMP_WINDOW), as a relative bound.
 *
 *  累加类格（KAT_K3_CMP_WINDOW）使用的相对窗口。 */
#define KAT_K3_WINDOW 4.0e-6
