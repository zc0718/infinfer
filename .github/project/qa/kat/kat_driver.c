// Conan::ImportStart
#include <infinfer/kernel/kernel.h>
#include <string.h>

#include "kat_chain.h"
#if defined(KAT_CHAIN_K1)
#include "kat_chain_k1.h"
#endif
#if defined(KAT_CHAIN_K2)
#include "kat_chain_k2.h"
#endif
#if defined(KAT_CHAIN_K2M)
#include "kat_chain_k2m.h"
#endif
#if defined(KAT_CHAIN_K2I8)
#include "kat_chain_k2i8.h"
#endif
#if defined(KAT_CHAIN_K3)
#include "kat_chain_k3.h"
#endif
#if !defined(KAT_EMIT) && !defined(KAT_CHAIN_K3)
#if defined(KAT_CHAIN_K2M)
#include "generated/kat_k2m_image.h"
#elif defined(KAT_CHAIN_K2I8)
#include "generated/kat_k2i8_image.h"
#elif defined(KAT_CHAIN_K2)
#include "generated/kat_k2_image.h"
#elif defined(KAT_CHAIN_K1)
#include "generated/kat_k1_image.h"
#else
#include "generated/kat_k0_image.h"
#endif
#endif
#if defined(KAT_EMIT)
#include <stdio.h>
#endif
// Conan::ImportEnd

/*
 * k0_actquant driver. One source, three worlds:
 *
 *   KAT_EMIT  - host build that runs the chain and writes the expectation blob
 *               (the only place expectations are ever produced).
 *   (default) - target/host *check* build: runs the same chain and compares the
 *               produced bytes against the baked expectation, bit for bit.
 *
 * Exit status is the verdict, never stdout: semihosting `printf` blocks when no
 * console is attached (measured on mps2-an385), so the pass/fail path must not
 * contain a write.
 *
 * k0_actquant 驱动，一份源码三个世界：KAT_EMIT 在宿主上跑链并写出期望字节（唯一
 * 产生期望的地方）；默认是校验构建，在目标上跑同一条链，与烘焙的期望逐位比对。
 * 判定只走退出码，不靠 stdout：无控制台时 semihosting `printf` 会阻塞（mps2-an385
 * 实测），所以 pass/fail 路径上不能有写操作。
 */

#define KAT_EXIT_CHAIN 10  /* A cell returned non-OK. */
#define KAT_EXIT_EXPECT 11 /* The produced bytes differ from the baked ones. */
#define KAT_EXIT_FAULT 13  /* The target took a CPU exception (see kat_fault_handler). */
/* k3 only. 仅 k3 使用。 */
#define KAT_EXIT_TIER_MISSING 12 /* A cell the matrix calls done was not registered. */
#define KAT_EXIT_TIER_DIFF 20    /* Plus the first disagreeing cell index, capped at 7. */

/* One driver, two chains: the only difference between them is which run/expect
 * pair it calls. Everything downstream compares bytes and reports an exit code.
 *
 * 一个驱动、两条链：差别只在调哪一对 run/expect。下游一律比字节、用退出码报告。 */
#if defined(KAT_CHAIN_K3)
/* k3 compares the tier row against the scalar row inside the chain, on this target, so there is no
 * expectation to bake and no blob to compare: the result is a verdict, carried by the exit code
 * alone. Baking a host-produced blob would be wrong here -- the whole point is that the reference
 * lives on the target.
 *
 * k3 在链内部、在本目标上把档位行与 scalar 行比完，所以没有期望要烘焙、没有 blob 要比：产物只有
 * 判定，由退出码携带。这里烘焙宿主 blob 是错的——本链的要义就是参考在目标上。 */
#define KAT_SELFCOMPARING 1
typedef kat_k3_result kat_result;
#define kat_run kat_k3_run
#elif defined(KAT_CHAIN_K2M)
typedef kat_k2m_result kat_result;
#define kat_run kat_k2m_run
#define kat_expect_len kat_k2m_expect_len
#define kat_expect kat_k2m_expect
#elif defined(KAT_CHAIN_K2I8)
typedef kat_k2i8_result kat_result;
#define kat_run kat_k2i8_run
#define kat_expect_len kat_k2i8_expect_len
#define kat_expect kat_k2i8_expect
#elif defined(KAT_CHAIN_K2)
typedef kat_k2_result kat_result;
#define kat_run kat_k2_run
#define kat_expect_len kat_k2_expect_len
#define kat_expect kat_k2_expect
#elif defined(KAT_CHAIN_K1)
typedef kat_k1_result kat_result;
#define kat_run kat_k1_run
#define kat_expect_len kat_k1_expect_len
#define kat_expect kat_k1_expect
#else
typedef kat_k0_result kat_result;
#define kat_run kat_k0_run
#define kat_expect_len kat_k0_expect_len
#define kat_expect kat_k0_expect
#endif

/* The image certifies the **reference** tier, and says so instead of relying on
 * it. The operators register on every entry now, so on a build that carries a
 * vector tier (x86-64, aarch64, armv7hf+neon-vfpv4, rv64gcv) the policy would
 * otherwise become that tier on the first operator call. The emitter and the
 * checker must agree, or the golden is produced under one tier and compared
 * under another: the memcmp at the end of every chain would go red for a reason
 * that has nothing to do with the graph.
 *
 * 本镜像认证的是**参考档**，而且写出来而不是靠默认。算子现在每次入口都注册，所以在
 * 带向量档的构建上（x86-64、aarch64、armv7hf+neon-vfpv4、rv64gcv），策略会在第一次
 * 调算子时变成那一档。发射器与校验器必须一致，否则 golden 在一个档下发射、在另一个档
 * 下比较——每条链末尾的 memcmp 会为了与图无关的理由变红。 */
#if defined(KAT_CHAIN_K2) || defined(KAT_CHAIN_K2M) || defined(KAT_CHAIN_K2I8)
static int kat_pin_reference_tier(void) { return infr_kernel_set_tier(INFR_TIER_SCALAR) == INFR_OK ? 0 : 1; }
#else
/* `k0`/`k1`/`k3` look their kernels up by name and never reach a composed
 * operator, so they do not link the dispatcher and there is no policy to pin.
 *
 * `k0`/`k1`/`k3` 按名字查内核、从不经过组合算子，所以它们不链接分发器，也没有策略
 * 要钉。 */
static int kat_pin_reference_tier(void) { return 0; }
#endif

#if defined(KAT_EMIT)

/* Expectation blob: magic, version, length, then the two checked tensors as raw
 * bytes so the comparison downstream can be a memcmp rather than a float compare.
 *
 * 期望 blob：magic、版本、长度，随后两个张量的裸字节——下游比对因此是 memcmp，
 * 而不是浮点比较。 */
static const char k_blob_magic[4] = {'K', 'A', 'T', 'R'};

int main(int argc, char **argv) {
    kat_result r;
    /* The emitter cannot use KAT_EXPECT_LEN: that macro lives in the baked
     * header, which only the check build includes. A fixed ceiling plus a guard
     * keeps it honest. 8 KiB because the MobileNet chain's checked tensor is 1000
     * floats (4 KB); the smaller chains use tens of bytes, so the ceiling costs
     * nothing on the host and still refuses a chain that grows unreasonably.
     *
     * 发射器用不了 KAT_EXPECT_LEN：那个宏在烘出来的头部里，而只有校验构建才包含它。所以
     * 用一个固定上限加一道守卫。取 8 KiB 是因为 MobileNet 链被校验的张量是 1000 个 float
     * （4 KB）；更小的链只用几十字节，所以这个上限在宿主上不花钱，同时仍能拒绝无节制变大的链。 */
    unsigned char expect[8192];
    unsigned char header[8];
    unsigned v;
    FILE *f;

    if (argc != 2) {
        (void)fprintf(stderr, "usage: %s <out.bin>\n", argv[0]);
        return 2;
    }
    if (kat_pin_reference_tier() != 0) {
        return KAT_EXIT_CHAIN;
    }
    int run_status = kat_run(&r);

    if (run_status != INFR_OK) {
        /* Name the code: `kat_emit` failing with a bare exit 10 has cost a rebuild-and-
         * guess cycle before, and the status is the one thing that separates "the graph
         * is wrong" from "the arena is short".
         *
         * 把状态码说出来：`kat_emit` 只报一个退出码 10，以前就得靠重编加猜，而这个码正是
         * 区分"图不对"与"arena 不够"的唯一信息。 */
        (void)fprintf(stderr, "kat_emit: chain failed (%d)\n", run_status);
        return KAT_EXIT_CHAIN;
    }
    v = kat_expect_len();
    if (v > sizeof expect) {
        (void)fprintf(stderr, "kat_emit: checked tensor exceeds the emitter buffer\n");
        return 2;
    }
    kat_expect(&r, expect);

    header[0] = 'K';
    header[1] = 'A';
    header[2] = 'T';
    header[3] = 'R';
    header[4] = (unsigned char)(v & 0xFFu);
    header[5] = (unsigned char)((v >> 8) & 0xFFu);
    header[6] = (unsigned char)((v >> 16) & 0xFFu);
    header[7] = (unsigned char)((v >> 24) & 0xFFu);

    f = fopen(argv[1], "wb");
    if (f == 0) {
        (void)fprintf(stderr, "kat_emit: cannot open %s\n", argv[1]);
        return 2;
    }
    (void)fwrite(header, 1, sizeof(header), f);
    (void)fwrite(expect, 1, v, f);
    if (fclose(f) != 0) {
        (void)fprintf(stderr, "kat_emit: write failed\n");
        return 2;
    }
    return 0;
}

#else /* check build */

/* Freestanding means "this image owns its own startup": M-profile ARM and
 * bare-metal RISC-V. It must NOT be keyed on `__arm__`, because AArch32 Linux
 * defines that too - measured: the armv7hf Linux build then compiled the bare-metal
 * path, supplied no `main`, and glibc's crt1.o failed with "undefined reference to
 * main". A Linux userspace build has a C runtime and takes `main()` instead.
 *
 * 自带启动代码 = M-profile ARM 与裸机 RISC-V。**不能用 `__arm__` 作判据**：AArch32
 * Linux 同样定义它——实测 armv7hf Linux 构建因此编成了裸机路径、不提供 `main`，
 * glibc 的 crt1.o 报 “undefined reference to main”。Linux 用户态有 C 运行时，走 `main()`。 */
#if (defined(__arm__) && defined(__ARM_ARCH_PROFILE) && (__ARM_ARCH_PROFILE == 'M')) ||                                \
    (defined(__riscv) && !defined(__linux__))
#define KAT_FREESTANDING 1
#else
#define KAT_FREESTANDING 0
#endif

#if KAT_FREESTANDING
/* Section bounds the linker script provides; the ARM vector table also needs
 * `_estack`. Both scripts define the four data/bss symbols.
 *
 * 链接脚本提供的段边界；ARM 的向量表另外需要 `_estack`。两份脚本都定义了
 * 数据段与 bss 段的四个符号。 */
extern unsigned _sidata[];
extern unsigned _sdata[];
extern unsigned _edata[];
extern unsigned _sbss[];
extern unsigned _ebss[];
#if defined(__arm__)
/* Start of the vector table, named by the linker script (see smoke.ld).
 * 向量表起始地址，由链接脚本给出（见 smoke.ld）。 */
extern unsigned _svectors[];
#endif

/* The job every C runtime does before main, and the reason a bare-metal KAT
 * needs it: without the copy, `.data` holds whatever RAM contained (the k0 KAT
 * found this the hard way -- a garbage registry count on mps2-an505), and without
 * the zeroing, `.bss` does too.
 *
 * Byte-wise on purpose, with `volatile`, so the compiler cannot turn it into a
 * libc `memcpy` call: startup must not depend on the library it is about to make
 * usable, and a libc call here is exactly what made the first version of this
 * step unfollowable on mps2-an505.
 *
 * 每个 C 运行时在 main 之前都要做的事，也是裸机 KAT 非做不可的原因：不搬
 * `.data` 它就等于内存残留值（k0 KAT 就是这么撞见的——mps2-an505 上注册表计数
 * 是垃圾），不清零 `.bss` 同理。
 *
 * 刻意写成 volatile 的逐字节循环，让编译器不能把它变成 libc `memcpy` 调用：启动
 * 代码不该依赖它正在初始化其可用性的库，而第一版正是被那个 libc 调用卡住的。 */
void kat_startup_init(void) {
    volatile unsigned char *src = (volatile unsigned char *)_sidata;
    volatile unsigned char *dst = (volatile unsigned char *)_sdata;

#if defined(__arm__)
    /* Point the vector base at the table this image carries, before anything
     * else can fault. A loader that only fills SP and PC (QEMU's generic
     * `-device loader`) leaves VTOR at 0, so an image linked above address 0
     * faults into a table that is not there: measured on mps2-an505, the
     * HardFault fetch reads 0 at 0x0C, loads PC 0 and dies of INVSTATE instead
     * of reaching the handler. On the boards whose table sits at 0x0 this
     * writes the reset value, so one line covers both shapes.
     *
     * 在可能出异常之前，把向量基址指到本镜像真正带的那张表。只填 SP 和 PC 的
     * 加载器（QEMU 的通用 `-device loader`）会把 VTOR 留在 0，于是链接在非 0
     * 地址的镜像会到一张不存在的表里取向量：mps2-an505 实测取 HardFault 向量时
     * 在 0x0C 读到 0、装载 PC=0、以 INVSTATE 死掉，根本到不了处理函数。表本来
     * 就在 0x0 的板子上这一行写的是复位值，所以一条语句覆盖两种形状。 */
    *(volatile unsigned *)0xE000ED08u = (unsigned)_svectors;
#endif

#if !defined(KAT_SKIP_DATA)
#if defined(KAT_COPY_READ_ONLY)
    /* Diagnostic: read the whole source range, write nothing. Isolates "can the
     * CPU read the .data load image" from "can it write the .data run image". */
    {
        volatile unsigned char sink = 0u;
        while (dst < (volatile unsigned char *)_edata) {
            sink = (unsigned char)(sink ^ *src++);
            dst++;
        }
        if (sink == 0xFFu) {
            (void)sink;
        }
    }
#elif defined(KAT_COPY_WRITE_ONLY)
    /* Diagnostic: write the destination range, read nothing, N bytes at most. */
    {
        unsigned written = 0u;
        while (dst < (volatile unsigned char *)_edata && written < (unsigned)(0x550)) {
            *dst++ = 0x5Au;
            ++written;
#if defined(KAT_WRITE_LIMIT)
            if (written >= (unsigned)(KAT_WRITE_LIMIT)) {
                break;
            }
#endif
        }
    }
#elif defined(KAT_COPY_ZEROS)
    /* Diagnostic: same destination, same length, content independent of the
     * load image. Separates "the payload breaks the runtime" from "writing this
     * range at all breaks something". */
    while (dst < (volatile unsigned char *)_edata) {
        *dst++ = 0u;
    }
#else
    while (dst < (volatile unsigned char *)_edata) {
        *dst++ = *src++;
    }
#endif
#endif
#if !defined(KAT_SKIP_BSS)
    for (dst = (volatile unsigned char *)_sbss; dst < (volatile unsigned char *)_ebss; ++dst) {
        *dst = 0u;
    }
#else
    (void)dst;
#endif
}
#endif

#if KAT_FREESTANDING && defined(__arm__)
/* M-profile reset reads SP from offset 0 and PC from offset 4 of the raw image,
 * so the vector table comes from the linker script and this handler is its only
 * entry. `_estack` is the top of RAM it names.
 *
 * M-profile 复位从裸镜像偏移 0 取 SP、偏移 4 取 PC；向量表由链接脚本给出，本
 * 函数是它唯一的入口；`_estack` 就是它写的 RAM 顶。 */
extern unsigned _estack;
#endif

#if defined(__riscv) && defined(__riscv_vector)
/* The vector unit is off after reset, and nothing else turns it on.
 *
 * `mstatus.VS` resets to Off, so the first `vsetvli` of a vector body traps; and with VS on but
 * `FS` still Off, a floating-point vector instruction traps in turn. Both were measured on
 * rv64-virt while writing gaps G-61, and this chain is the first KAT image to contain vector code
 * at all -- which is why the setup lives here rather than in the chain: it belongs to the target's
 * startup, and a chain that had to remember it would forget it eventually.
 *
 * 复位后向量单元是关的，且没有别人会去打开它。`mstatus.VS` 复位为 Off，向量体的第一条
 * `vsetvli` 就会陷入；只开 VS 而 `FS` 仍为 Off 时，浮点向量指令照样陷入。两条都在写 G-61 时于
 * rv64-virt 上实测过。本链是第一个含向量代码的 KAT 镜像，所以这段放在这里而不是链里：它属于目标
 * 的启动，放在链里早晚会被忘掉。 */
static void kat_enable_vector_state(void) {
    unsigned bits = (1u << 9) | (1u << 13); /* mstatus.VS = Initial, mstatus.FS = Initial */
    __asm__ volatile("csrs mstatus, %0" : : "r"(bits) : "memory");
}
#endif

/* External linkage, not `static`: the RISC-V start stub below calls it by name.
 * 外部链接而非 `static`：下面的 RISC-V 启动桩按名字调用它。 */
int kat_check(void) {
    kat_result r;
#if !defined(KAT_SELFCOMPARING)
    unsigned char expect[KAT_EXPECT_LEN];
#endif
#if defined(__riscv) && defined(__riscv_vector)
    /* Before the first vector instruction of the image, which is inside the chain. */
    kat_enable_vector_state();
#endif
    /* The image certifies the **reference** tier, and says so instead of relying
     * on it. The operators register on every entry now, so on a target whose
     * build carries a vector tier (aarch64, armv7hf+neon-vfpv4, rv64gcv) the
     * policy would otherwise become that tier on the first operator call and the
     * goldens, which are emitted by the host's reference run, would stop being
     * bit-comparable -- memcmp at the end of every chain would go red for a
     * reason that has nothing to do with the graph.
     *
     * 本镜像认证的是**参考档**，而且写出来而不是靠默认。算子现在每次入口都注册，
     * 所以在带向量档的构建上（aarch64、armv7hf+neon-vfpv4、rv64gcv），策略会在第一次
     * 调算子时变成那一档，而由宿主参考运行发射的 golden 就不再逐位可比——每条链末尾的
     * memcmp 会为了与图无关的理由变红。 */
    if (kat_pin_reference_tier() != 0) {
        return KAT_EXIT_CHAIN;
    }
    int status = kat_run(&r);

    if (status != INFR_OK) {
#if defined(KAT_STATUS_CODE)
        /* Diagnostic: carry the chain's status out as 60 + status, so a probe can
         * tell INVALID_ARGUMENT (61) from UNSUPPORTED (62), BUFFER_TOO_SMALL (63)
         * and NOT_IMPLEMENTED (64) without a debugger. Default builds keep the
         * single 10, because a verdict must be one number.
         *
         * 诊断：把链返回的状态码以 60 + status 带出来，探针不必调试器就能区分
         * INVALID_ARGUMENT(61) / UNSUPPORTED(62) / BUFFER_TOO_SMALL(63) /
         * NOT_IMPLEMENTED(64)。默认构建仍是单一的 10，因为判定只能是一个数。 */
        return (int)(60 + status);
#else
        return KAT_EXIT_CHAIN;
#endif
    }
#if defined(KAT_STAGE)
    /* Bisect probe. Returning here instead of continuing matters: with the chain
     * cut short the result struct is only partly written, and running the
     * expectation on it turns the run into garbage-shaped float work -- a
     * measurement that changes what it measures. So the probe reports the stage
     * it reached and does nothing else.
     *
     * 二分探针。这里直接返回而不是往下走是有原因的：链被截断后结果结构体只写了一
     * 部分，再拿它跑期望计算，等于用垃圾数据做浮点运算——那是一个会改变被测量对象
     * 的测量。所以探针只报“走到了第几阶段”，别的什么都不做。 */
    return 100 + (KAT_STAGE);
#endif
#if defined(KAT_SELFCOMPARING)
    /* The verdict is the counters the chain filled in. `missing` is a contradiction between the
     * matrix and the registry, not a disagreement, and it gets its own code so the two are never
     * confused in a log. A disagreement carries the cell index (capped) because that is the one
     * piece of information a debugger would otherwise be needed for.
     *
     * 判定来自链填好的计数。`missing` 是矩阵与注册表矛盾、不是不一致，因此单独一个码，免得日志里
     * 混淆。不一致会把格下标带出来（封顶），因为那是唯一本来需要调试器才能拿到的信息。 */
    if (r.missing != 0u) {
        return KAT_EXIT_TIER_MISSING;
    }
    if (r.mismatched != ~0u) {
#if defined(KAT_K3_DIAG)
        /* Diagnostic: name the cell *and* the corpus length in one exit code, 100 + cell*16 +
         * length index. The default build keeps the single documented code, because a verdict must
         * be one number; this exists because a target has no console to print to, and bisecting
         * twelve lengths by rebuild is the alternative.
         *
         * 诊断：一个退出码同时给出格与语料长度，100 + cell*16 + 长度下标。默认构建仍是那个单一
         * 的判定码——判定只能是一个数；这段存在的理由是目标上没有控制台可打印，而替代方案是
         * 重编十二次来二分长度。 */
        return 100 + (int)(r.mismatched * 16u) + (int)r.length_index;
#else
        return KAT_EXIT_TIER_DIFF + (int)(r.mismatched < 8u ? r.mismatched : 8u);
#endif
    }
    if (r.compared == 0u || r.elements == 0ul) {
        /* Nothing compared is not a pass: it would make an image built without the tier sources
         * look green -- or, worse, one whose cells were called with the wrong arguments and
         * agreed only because both rows refused. Both counters have to be non-zero.
         *
         * 「什么都没比」不是通过：否则没编进档位源的镜像会变绿；更糟的是格被用错参数调用，
         * 两个行都拒绝因而"一致"。两个计数都必须非零。 */
        return KAT_EXIT_TIER_MISSING;
    }
    return 0;
#else
    kat_expect(&r, expect);
    if (memcmp(expect, kat_expect_bytes, KAT_EXPECT_LEN) != 0) {
        return KAT_EXIT_EXPECT;
    }
    return 0;
#endif
}

#if KAT_FREESTANDING && defined(__arm__)

static void kat_exit(int code) {
    /* SYS_EXIT_EXTENDED (0x20): a two-word block, reason = ADP_Stopped_
     * ApplicationExit (0x20026) and the subcode, which QEMU returns as the
     * process exit status. Plain SYS_EXIT cannot express a failing code.
     *
     * SYS_EXIT_EXTENDED（0x20）：两字块，reason = ADP_Stopped_ApplicationExit
     * (0x20026) 与 subcode；QEMU 把 subcode 作为进程退出码。朴素 SYS_EXIT
     * 无法表达失败码。 */
    unsigned block[2];
    block[0] = 0x20026u;
    block[1] = (unsigned)code;
    {
        register unsigned r0 __asm__("r0") = 0x20u;
        register unsigned r1 __asm__("r1") = (unsigned)(void *)block;
        __asm__ volatile("bkpt 0xAB" : : "r"(r0), "r"(r1) : "memory");
    }
    for (;;) {
    }
}

/* SYS_WRITE0 is unused by the verdict path and QEMU discards it without a
 * console chardev, so the fault site is recovered from exit codes instead (see
 * KAT_FAULT_SLICE below). Kept out of the binary on purpose.
 *
 * SYS_WRITE0 不在判定路径上、且 QEMU 没接控制台时会丢弃它，所以出错位置改用
 * 退出码切片重建（见下面的 KAT_FAULT_SLICE）。刻意不进二进制。 */

void reset_handler(void) {
    kat_startup_init();
    kat_exit(kat_check());
}

#if defined(KAT_FAULT_SLICE)
#if !defined(KAT_FAULT_REG)
/* Exception frame word to report: 6 = PC (the faulting instruction), 5 = LR.
 * 取哪一帧字：6 = PC（出错指令），5 = LR。 */
#define KAT_FAULT_REG 6
#endif

/* Naked, so the stack pointer still points at the exception frame when it is read;
 * the asm then hands the frame base to an ordinary C function, which is free to
 * push. Kept in the diagnostics build only.
 *
 * naked 入口：读栈指针时它仍指向异常帧；汇编随后把帧基址交给普通 C 函数，那边
 * 随便压栈都不影响。只在诊断构建里存在。 */
__attribute__((naked, noreturn)) void kat_fault_report(void) {
    __asm__ volatile("mrs r0, msp\n\t"
                     "b kat_fault_report_from\n");
}

void kat_fault_report_from(unsigned frame_base) {
    kat_exit((int)((((const unsigned *)frame_base)[KAT_FAULT_REG] >> (8 * KAT_FAULT_SLICE)) & 0xFFu));
}
#endif

#if defined(KAT_FAULT_DUMP)
/* Park the exception frame and the fault status registers in a fixed RAM range so a
 * run with no debugger can be read from outside (QEMU monitor `xp /14xw 0x700000`).
 * Needed because an exit code carries one byte and the frame alone cannot tell an
 * undefined instruction from a stacking failure — CFSR/HFSR can.
 *
 * 把异常帧与故障状态寄存器停到一块固定 RAM，供无调试器的运行从外部读取（QEMU monitor
 * `xp /14xw 0x700000`）。原因：退出码只能带一个字节，而光看帧分不清“未定义指令”和
 * “入栈失败”，CFSR/HFSR 能分清。 */
#define KAT_DUMP_BASE 0x00700000u

__attribute__((naked, noreturn)) void kat_fault_dump(void) {
    __asm__ volatile("mrs r0, msp\n\t"
                     "b kat_fault_dump_from\n");
}

void kat_fault_dump_from(unsigned frame_base) {
    volatile unsigned *out = (volatile unsigned *)KAT_DUMP_BASE;
    unsigned i;

    out[0] = frame_base;
    for (i = 0u; i < 8u; ++i) {
        out[1u + i] = ((const unsigned *)frame_base)[i];
    }
    out[9] = *(volatile unsigned *)0xE000ED28u;  /* CFSR: what kind of fault */
    out[10] = *(volatile unsigned *)0xE000ED2Cu; /* HFSR: forced/escalated? */
    out[11] = *(volatile unsigned *)0xE000ED34u; /* MMFAR */
    out[12] = *(volatile unsigned *)0xE000ED38u; /* BFAR */
    out[13] = 0x5A5A5A5Au;                       /* marker: dump completed */
    for (;;) {
    }
}
#endif

/* Every fault vector points here (see smoke.ld). A fault is a verdict, not a
 * lockup: without this the CPU reads the HardFault vector out of whatever byte
 * follows .text and double-faults, which looks like "qemu died" instead of
 * "this binary faulted". The exit code keeps it distinguishable from a plain
 * expectation mismatch.
 *
 * 所有异常向量都指向这里（见 smoke.ld）。异常也是判定结果而不是挂死：没有它时
 * CPU 会把 .text 后面的字节当 HardFault 向量、直接双异常，看起来像“qemu 挂了”
 * 而不是“这个镜像异常了”。用独立退出码把它与期望不符区分开。 */
void kat_fault_handler(void) {
    /* The M-profile exception frame sits at MSP on entry:
     * [0..3] R0-R3, [4] R12, [5] LR, [6] PC (the faulting instruction), [7] xPSR.
     *
     * With KAT_FAULT_SLICE=n the handler exits with byte n of one of those words
     * instead of the plain 13: QEMU discards semihosting SYS_WRITE0 without a
     * console, so a few runs reconstruct the fault site when nothing else can.
     * KAT_FAULT_REG picks which word (default 6, the PC; 5 is the LR). The default
     * build keeps the plain code, because a verdict must be one number.
     *
     * M-profile 异常帧在入口处就在 MSP 上：[6] 是出错指令地址、[5] 是 LR。带
     * KAT_FAULT_SLICE=n 时退出码变成所选字（KAT_FAULT_REG，默认 6，取 5 则是 LR）
     * 的第 n 个字节——QEMU 在没有控制台时会丢弃 SYS_WRITE0，几次运行就能重建出错
     * 位置；默认构建仍是单纯一个码，因为判定只能是一个数。
     *
     * The MSP must be read before anything is pushed: a normal C prologue moves the
     * stack pointer, and then `frame[6]` reads past the real frame. Measured: it
     * reported four identical zero slices, i.e. "faulted at address 0", which sends
     * the investigation the wrong way.
     *
     * MSP 必须在任何压栈之前读：普通 C 函数序言会移动栈指针，之后 `frame[6]` 读到
     * 真帧之外。实测四个切片全读出 0（"在地址 0 出错"），把排查带偏。 */
#if defined(KAT_FAULT_SLICE)
    kat_fault_report();
    for (;;) {
    }
#elif defined(KAT_FAULT_DUMP)
    kat_fault_dump();
    for (;;) {
    }
#else
    kat_exit(KAT_EXIT_FAULT);
#endif
}

#elif KAT_FREESTANDING

/* External linkage on purpose: the start stub below is written in asm and looks
 * these up by name, so they must not be `static` (a static symbol has no link
 * name the assembler can reach).
 *
 * 刻意用外部链接：下面的启动桩是汇编写的，按名字找这两个符号；`static` 的符号
 * 没有汇编器能找到的链接名。 */
void kat_exit(int code) {
    /* SiFive test finisher: QEMU exits with the 16-bit code written above the
     * 0x5555 magic. RISC-V semihosting is not serviced by QEMU 8.2 (measured).
     *
     * SiFive test finisher：向 0x5555 魔数之上写 16 位码，QEMU 即以该码退出。
     * QEMU 8.2 不服务 RISC-V semihosting（实测）。 */
    *(volatile unsigned *)0x100000u = ((unsigned)code << 16) | 0x5555u;
    for (;;) {
    }
}

__asm__(".section .text.start,\"ax\"\n"
        ".globl _start\n"
        "_start:\n"
        "  li sp, 0x80200000\n"
        "  call kat_startup_init\n"
        "  call kat_check\n"
        "  call kat_exit\n"
        "  tail _start\n");

#else

/* Host and Linux userspace: the C runtime already did the startup, `main` returns
 * the verdict, and the shell forwards it as the exit status.
 *
 * 宿主与 Linux 用户态：启动已由 C 运行时完成，`main` 的返回值就是判定，由 shell
 * 转成退出码。 */
int main(void) { return kat_check(); }

#endif

#endif /* KAT_EMIT */
