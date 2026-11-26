// Conan::ImportStart
#include <stdint.h>
#include <string.h>

#include "kat_chain_k3.h"
// Conan::ImportEnd

/*
 * The cell table is generated from `src/kernel/matrix.yml` for the one tier this image was built
 * for, so the chain cannot claim a cell the matrix does not and cannot skip one the matrix does.
 * `KAT_K3_REGISTER` comes from the same header: the register function of that tier only, because
 * the other tiers' sources are not linked into this image.
 *
 * 格表由 `matrix.yml` 生成，只含本镜像那一个档位，因此链既不能多声称也不能少测。
 * `KAT_K3_REGISTER` 同样来自生成的头，只调该档位的注册函数——其余档位的源不在本镜像里。
 */
#include "generated/kat_k3_cells.h"

/* 257 elements of fp32 in four buffers is ~4 KB of static storage: no heap on bare metal, and the
 * M-profile boards this runs on have megabytes of RAM. 263 is deliberately not a power of two so
 * a body that assumes one shows up.
 *
 * 四个 fp32 缓冲共约 4 KB 静态存储：裸机不用堆，而这些 M 档板有兆级 RAM。257 故意不是 2 的幂，
 * 好让"假设长度是 2 的幂"的实现暴露出来。 */
#define K3_MAX 257u

static const unsigned k3_lengths[] = {0u, 1u, 3u, 4u, 5u, 7u, 8u, 9u, 16u, 17u, 64u, K3_MAX};
#define K3_LENGTH_COUNT (sizeof k3_lengths / sizeof k3_lengths[0])

static float g_x[K3_MAX];
static float g_w[K3_MAX];
static float g_fa[K3_MAX];
static float g_fb[K3_MAX];
static int8_t g_xi[K3_MAX];
static int8_t g_wi[K3_MAX];
static int8_t g_qa[K3_MAX];
static int8_t g_qb[K3_MAX];
static int16_t g_xs[K3_MAX];
static int16_t g_ws[K3_MAX];
static int16_t g_sa[K3_MAX];
static int16_t g_sb[K3_MAX];
/* `scale` takes one broadcast factor as its second input, so it has to live somewhere. */
static float g_factor;

/* How many output elements were actually compared, across the whole run. A cell count alone is not
 * evidence: the first version of this chain reported every cell as compared while comparing nothing
 * inside them, because both rows were handed the same NULL operand. The driver fails the run when
 * this is zero.
 *
 * 真正被比较过的输出元素个数。只看格数不算证据：本链第一版"比较了每一格"却一个元素都没比——两个
 * 行都收到了同一个 NULL 操作数。这个数为 0 时驱动直接判失败。 */
static unsigned long g_elements_compared;

/* One binary32, compared as bits. Elementwise cells do exactly one IEEE operation, so the verdict
 * really is bit equality -- and comparing bits rather than `==` keeps +0.0 and -0.0 distinct,
 * which is the difference the reference's `x > 0 ? x : 0` is there to produce.
 *
 * 按位比较一个 binary32。逐元素格只做一次 IEEE 运算，所以判据就是位相等；用位而不是 `==`
 * 还能把 +0.0 与 -0.0 区分开，而这正是参考里 `x > 0 ? x : 0` 要产生的差别。 */
static int bits_equal(float a, float b) {
    uint32_t x;
    uint32_t y;
    memcpy(&x, &a, sizeof x);
    memcpy(&y, &b, sizeof y);
    return x == y;
}

/* Relative window for the accumulating cells. The measured worst cases on the targets this runs on
 * are fc 1.024e-06 and sum 1.756e-06 (gaps G-57, G-60), so 4e-06 sits above them rather than being
 * a number chosen to make the run pass. A `1.0` floor keeps it from becoming an absolute-zero test
 * on tiny values, where a relative bound says nothing.
 *
 * 累加类格的相对窗口。本链运行的目标上实测最差为 fc 1.024e-06、sum 1.756e-06（gaps G-57/G-60），
 * 4e-06 在它们之上，而不是"调到能过"的数。以 1.0 兜底，避免在极小值上退化成绝对零判据。 */
static int within_window(float a, float b) {
    float d = a > b ? a - b : b - a;
    float m = a > b ? a : b;
    if (m < 0.0f) {
        m = -m;
    }
    if (m < 1.0f) {
        m = 1.0f;
    }
    return d <= (float)KAT_K3_WINDOW * m;
}

/* Deterministic corpus, built the same way on every target from integer maths only: the LCG state
 * is exact, the division by 1000 is exact in binary32 for these magnitudes, so no target can
 * disagree about the input and blame the kernel.
 *
 * 确定性语料：只用整数运算在同一套规则下生成，LCG 状态精确、除以 1000 在这些量级上于 binary32
 * 中精确，因此没有目标会对输入产生分歧、再把责任推给内核。 */
static void build_corpus(void) {
    uint32_t s = 12345u;
    unsigned i;
    for (i = 0; i < K3_MAX; ++i) {
        int32_t v;
        s = s * 1664525u + 1013904223u;
        v = (int32_t)((s >> 9) % 4001u) - 2000;
        g_x[i] = (float)v / 1000.0f;
        s = s * 1664525u + 1013904223u;
        v = (int32_t)((s >> 9) % 4001u) - 2000;
        g_w[i] = (float)v / 1000.0f;
    }
    /* The corners a vector body can get wrong: the int8 extremes and the int16 extremes, plus -1
     * and 0 so a signed compare against zero is exercised.
     *
     * 向量实现容易搞错的角：int8/int16 的极值，加上 -1 与 0，好让"与零的有符号比较"被走到。 */
    g_xi[0] = INT8_MIN;
    g_wi[0] = INT8_MIN;
    g_xi[1] = INT8_MAX;
    g_wi[1] = INT8_MAX;
    g_xi[2] = -1;
    g_wi[2] = 1;
    g_xs[0] = INT16_MIN;
    g_ws[0] = INT16_MIN;
    g_xs[1] = INT16_MAX;
    g_ws[1] = INT16_MAX;
    g_xs[2] = -1;
    g_ws[2] = 1;
    for (i = 3; i < K3_MAX; ++i) {
        g_xi[i] = (int8_t)((int32_t)((i * 37u) % 255u) - 127);
        g_wi[i] = (int8_t)((int32_t)((i * 91u + 7u) % 255u) - 127);
        g_xs[i] = (int16_t)((int32_t)((i * 911u) % 65533u) - 32766);
        g_ws[i] = (int16_t)((int32_t)((i * 131u + 29u) % 65533u) - 32766);
    }
    g_factor = 1.75f;
}

/* The quantised dot product is compared under two parameter sets: one that keeps the result inside
 * the dtype (so the verdict is about arithmetic) and one that saturates it (so the verdict is also
 * about the clamp). The saturation corners themselves are pinned exactly by the host unit tests
 * (`KernelParityFcInt8/AppliesMultiplierAndZeroPointThenClamps`); what this adds is that both tiers
 * saturate **the same way** on the same target.
 *
 * 量化点积用两组参数比较：一组让结果落在 dtype 内（判的是算术），一组直接饱和（判的还有钳位）。
 * 饱和角本身由宿主单测精确钉住（`KernelParityFcInt8/...`）；这里补的是两个档位在**同一目标上**
 * **以同样方式**饱和。 */
static const float k3_multipliers[] = {1.0e-6f, 1.0e6f};
#define K3_MULT_COUNT (sizeof k3_multipliers / sizeof k3_multipliers[0])

static void fill_quant(infr_quant *q, float multiplier) {
    q->kind = INFR_QUANT_PER_TENSOR;
    q->scale = 1.0f;
    q->zero_point = 3;
    q->requant_multiplier = multiplier;
}

/* Compare one cell at one length. Returns 1 when the two rows agree, 0 when they do not. */
static int compare_at(const kat_k3_cell *cell, const infr_kernel_desc *ref, const infr_kernel_desc *row,
                      unsigned n) {
    const void *in_ref[2] = {0, 0};
    const void *in_row[2] = {0, 0};
    void *out_ref[1];
    void *out_row[1];
    infr_quant q;
    infr_status sa;
    infr_status sb;
    unsigned i;
    unsigned count;

    /* Dot products accumulate, so every distinct parameter set is its own comparison. */
    unsigned passes = (cell->call == KAT_K3_CALL_DOT_Q) ? (unsigned)K3_MULT_COUNT : 1u;
    unsigned pass;
    for (pass = 0; pass < passes; ++pass) {
        fill_quant(&q, k3_multipliers[pass]);

        switch (cell->call) {
        case KAT_K3_CALL_BINARY:
            /* Two real operands. Handing `add` a NULL second input made both rows refuse, and two
             * refusals compare equal -- the first version of this chain passed every cell without
             * comparing an element. */
            in_ref[0] = g_x;
            in_ref[1] = g_w;
            in_row[0] = g_x;
            in_row[1] = g_w;
            break;
        case KAT_K3_CALL_SCALE:
            in_ref[0] = g_x;
            in_ref[1] = &g_factor;
            in_row[0] = g_x;
            in_row[1] = &g_factor;
            break;
        case KAT_K3_CALL_DOT:
            in_ref[0] = g_x;
            in_ref[1] = g_w;
            in_row[0] = g_x;
            in_row[1] = g_w;
            break;
        case KAT_K3_CALL_DOT_Q:
            if (cell->dtype == INFR_DTYPE_I8) {
                in_ref[0] = g_xi;
                in_ref[1] = g_wi;
                in_row[0] = g_xi;
                in_row[1] = g_wi;
            } else {
                in_ref[0] = g_xs;
                in_ref[1] = g_ws;
                in_row[0] = g_xs;
                in_row[1] = g_ws;
            }
            break;
        default:
            if (cell->dtype == INFR_DTYPE_I8) {
                in_ref[0] = g_xi;
                in_row[0] = g_xi;
            } else {
                in_ref[0] = g_x;
                in_row[0] = g_x;
            }
            break;
        }

        switch (cell->dtype) {
        case INFR_DTYPE_I8:
            out_ref[0] = g_qa;
            out_row[0] = g_qb;
            break;
        case INFR_DTYPE_I16:
            out_ref[0] = g_sa;
            out_row[0] = g_sb;
            break;
        default:
            out_ref[0] = g_fa;
            out_row[0] = g_fb;
            break;
        }

        {
            const infr_quant *qp = (cell->call == KAT_K3_CALL_DOT_Q) ? &q : 0;
            sa = ref->fn(in_ref, out_ref, n, qp);
            sb = row->fn(in_row, out_row, n, qp);
        }
        /* Both refusing is agreement; one refusing is not. Zero length is the usual case here, and
         * the reference rejects it for the reductions, so the status is part of the contract.
         *
         * A refusal at a non-zero length is different: the harness feeds a valid call, so a refusal
         * there means this chain is not measuring the kernel at all. That is exactly how the first
         * version passed with `add` never compared (both rows got a NULL operand, both refused, and
         * the statuses matched), so it is a failure now rather than a skip. */
        if (sa != sb) {
            return 0;
        }
        if (sa != INFR_OK) {
            if (n != 0u) {
                return 0;
            }
            continue;
        }


        count = (cell->call == KAT_K3_CALL_REDUCE || cell->call == KAT_K3_CALL_DOT ||
                 cell->call == KAT_K3_CALL_DOT_Q)
                    ? 1u
                    : n;
        for (i = 0; i < count; ++i) {
            if (cell->dtype == INFR_DTYPE_I8) {
                if (g_qa[i] != g_qb[i]) {
                    return 0;
                }
            } else if (cell->dtype == INFR_DTYPE_I16) {
                if (g_sa[i] != g_sb[i]) {
                    return 0;
                }
            } else if (cell->cmp_mode == KAT_K3_CMP_BITWISE) {
                if (!bits_equal(g_fa[i], g_fb[i])) {
                    return 0;
                }
            } else if (!within_window(g_fa[i], g_fb[i])) {
                return 0;
            }
            g_elements_compared += 1ul;
        }
    }
    return 1;
}

infr_status kat_k3_run(kat_k3_result *out) {
    infr_status status;
    unsigned c;
    unsigned li;

    if (out == 0) {
        return INFR_ERR_INVALID_ARGUMENT;
    }
    out->compared = 0u;
    out->missing = 0u;
    out->mismatched = ~0u;
    out->length = 0u;
    out->elements = 0ul;
    g_elements_compared = 0ul;

    status = infr_kernel_register_scalar();
    if (status != INFR_OK) {
        return status;
    }
    status = KAT_K3_REGISTER();
    if (status != INFR_OK) {
        return status;
    }
    build_corpus();

    for (c = 0; c < KAT_K3_CELL_COUNT; ++c) {
        const kat_k3_cell *cell = &kat_k3_cells[c];
        const infr_kernel_desc *ref =
            infr_kernel_find(cell->kernel, cell->dtype, cell->quant_kind, INFR_TIER_SCALAR);
        const infr_kernel_desc *row =
            infr_kernel_find(cell->kernel, cell->dtype, cell->quant_kind, (infr_tier)KAT_K3_TIER);
        if (ref == 0) {
            /* The reference is not a capability of the build: a missing scalar row means the image
             * is wrong, not that the tier disagrees. */
            return INFR_ERR_NOT_IMPLEMENTED;
        }
        if (row == 0) {
            /* matrix.yml calls this cell done for this tier, so its absence is a contradiction
             * rather than a skip. The driver turns `missing` into a failure. */
            out->missing += 1u;
            continue;
        }
        for (li = 0; li < (unsigned)K3_LENGTH_COUNT; ++li) {
            if (!compare_at(cell, ref, row, k3_lengths[li])) {
                out->mismatched = c;
                out->length = k3_lengths[li];
                return INFR_OK;
            }
        }
        out->compared += 1u;
    }
    out->elements = g_elements_compared;
    return INFR_OK;
}
