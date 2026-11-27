// Conan::ImportStart
#include <stdint.h>
#include <string.h>

#include <infinfer/infinfer.h>

#include "kat_chain_k2.h"
#include "blob_graph.h"
#include "generated/kat_k2_model.h"
#include "generated/kat_k2_blob.h"
// Conan::ImportEnd

/*!
 * @brief [en] `k2_conv_micro` on the target: the whole L2 chain through the C ABI.
 * @brief [zh] 目标端的 `k2_conv_micro`：通过 C ABI 跑完整条 L2 链。
 * @note [en] This file is the reason the C++ orchestration had to keep C linkage:
 *            every operator below is reached through `infr_*`, from a C translation
 *            unit, on a Cortex-M target.
 * @note [zh] 本文件正是"C++ 编排必须保持 C 链接"的理由：下面每个算子都从 C 编译单元、
 *            在 Cortex-M 目标上通过 `infr_*` 到达。
 * @since 0.1.0
 */

#define K2_IN_H 16U
#define K2_IN_W 16U
#define K2_IN_C 1U
#define K2_CONV_OC 8U
#define K2_DW_OC 16U
#define K2_FC_OC 10U
#define K2_POOLED_H (K2_IN_H / 2U)
#define K2_POOLED_W (K2_IN_W / 2U)

/* Stage cut points, for bisecting a failure that only happens on the target: with
 * K2_STAGE=n the chain returns OK right after the n-th operator instead of running
 * the rest, so "it faulted" becomes "it got this far". Pair it with the driver's
 * KAT_STAGE=n, which then exits with 100+n rather than comparing against an
 * expectation produced by a half-finished graph. Undefined by default, and the
 * generated code is unchanged then.
 *
 * 分阶段裁剪点，用于二分只发生在目标端的故障：定义 K2_STAGE=n 时，链在第 n 个算子之后
 * 就返回 OK，把"它异常了"变成"它走到了第几步"。与驱动的 KAT_STAGE=n 配合使用，此时
 * 驱动以 100+n 退出，而不是拿半张图的结果去比期望。默认不定义，生成的代码与原来一致。 */
#if defined(K2_STAGE)
#define K2_STAGE_CUT(n)       \
    do {                      \
        if ((K2_STAGE) <= (n)) { \
            return INFR_OK;   \
        }                     \
    } while (0)
#else
#define K2_STAGE_CUT(n) \
    do {                \
    } while (0)
#endif

/* One workspace for the whole graph. Sized from the largest operator (the global
 * mean's 16x16 plane) plus slack, and asserted back to zero at the end: a graph that
 * grows the workspace as it executes is the defect this chain would expose.
 *
 * 整张图共用一个工作区，按最大算子（全局均值的 16×16 平面）加余量取尺寸，并在末尾断言
 * 回到 0：随执行而长胖正是本条链要暴露的缺陷。 */
static unsigned char g_workspace[2048] __attribute__((aligned(8)));

static float g_conv_out[K2_IN_H * K2_IN_W * K2_CONV_OC];
static float g_pool_out[K2_POOLED_H * K2_POOLED_W * K2_CONV_OC];
static float g_dw_out[K2_POOLED_H * K2_POOLED_W * K2_DW_OC];
static float g_gap_out[K2_DW_OC];
static float g_input[K2_IN_H * K2_IN_W * K2_IN_C];

static int bind_tensor(infr_tensor *tensor, float *data, unsigned h, unsigned w, unsigned c) {
    uint32_t dims[4];
    infr_shape shape;
    dims[0] = 1U;
    dims[1] = h;
    dims[2] = w;
    dims[3] = c;
    if (infr_shape_init(&shape, 4U, dims) != INFR_OK) {
        return INFR_ERR_INVALID_ARGUMENT;
    }
    return infr_tensor_init(tensor, data, INFR_DTYPE_F32, &shape);
}

static int run_relu(float *data, unsigned count) {
    const infr_kernel_desc *relu;
    const void *inputs[1];
    void *outputs[1];
    if (infr_kernel_register_scalar() != INFR_OK) {
        return INFR_ERR_NOT_IMPLEMENTED;
    }
    relu = infr_kernel_find("relu", INFR_DTYPE_F32, INFR_QUANT_NONE, INFR_TIER_SCALAR);
    if (relu == 0) {
        return INFR_ERR_NOT_IMPLEMENTED;
    }
    inputs[0] = data;
    outputs[0] = data;
    return relu->fn(inputs, outputs, count, 0);
}

int kat_k2_run(kat_k2_result *result) {
    infr_workspace workspace;
    infr_tensor input;
    infr_tensor filter;
    infr_tensor output;
    infr_shape filter_shape;
    uint32_t filter_dims[4];
    infr_conv2d_params conv;
    infr_pool_params pool;
    const infr_kernel_desc *fc;
    unsigned oc;
    int status;

    if (result == 0) {
        return INFR_ERR_INVALID_ARGUMENT;
    }
    K2_STAGE_CUT(0);
    memcpy(g_input, kat_k2_input, sizeof g_input);
    K2_STAGE_CUT(1);

    if (infr_workspace_init(&workspace, g_workspace, sizeof g_workspace) != INFR_OK) {
        return INFR_ERR_INVALID_ARGUMENT;
    }
    K2_STAGE_CUT(2);
    if (bind_tensor(&input, g_input, K2_IN_H, K2_IN_W, K2_IN_C) != INFR_OK ||
        bind_tensor(&output, g_conv_out, K2_IN_H, K2_IN_W, K2_CONV_OC) != INFR_OK) {
        return INFR_ERR_INVALID_ARGUMENT;
    }
    K2_STAGE_CUT(3);

    /* conv 3x3, pad 1, stride 1 */
    filter_dims[0] = K2_CONV_OC;
    filter_dims[1] = 3U;
    filter_dims[2] = 3U;
    filter_dims[3] = K2_IN_C;
    if (infr_shape_init(&filter_shape, 4U, filter_dims) != INFR_OK ||
        infr_tensor_init(&filter, (void *)kat_k2_conv_w, INFR_DTYPE_F32, &filter_shape) != INFR_OK) {
        return INFR_ERR_INVALID_ARGUMENT;
    }
    K2_STAGE_CUT(4);
    conv.kernel_h = 3U;
    conv.kernel_w = 3U;
    conv.stride_h = 1U;
    conv.stride_w = 1U;
    conv.pad_h = 1U;
    conv.pad_w = 1U;
    conv.in_channels = K2_IN_C;
    conv.out_channels = K2_CONV_OC;
    conv.groups = 1U;
    K2_STAGE_CUT(5);

    status = infr_conv2d_f32(&input, &filter, kat_k2_conv_b, &conv, &output, &workspace);
    if (status != INFR_OK) {
        return status;
    }
    K2_STAGE_CUT(6);
    status = run_relu(g_conv_out, K2_IN_H * K2_IN_W * K2_CONV_OC);
    if (status != INFR_OK) {
        return status;
    }
    K2_STAGE_CUT(7);

    /* max pool 2x2 stride 2, reading the convolution output */
    {
        infr_tensor conv_tensor;
        if (bind_tensor(&conv_tensor, g_conv_out, K2_IN_H, K2_IN_W, K2_CONV_OC) != INFR_OK ||
            bind_tensor(&output, g_pool_out, K2_POOLED_H, K2_POOLED_W, K2_CONV_OC) != INFR_OK) {
            return INFR_ERR_INVALID_ARGUMENT;
        }
        pool.kernel_h = 2U;
        pool.kernel_w = 2U;
        pool.stride_h = 2U;
        pool.stride_w = 2U;
        pool.pad_h = 0U;
        pool.pad_w = 0U;
        pool.channels = K2_CONV_OC;
        status = infr_max_pool2d_f32(&conv_tensor, &pool, &output, &workspace);
        if (status != INFR_OK) {
            return status;
        }
    }
    K2_STAGE_CUT(8);

    /* depthwise conv 3x3: same operator, groups == in_channels */
    filter_dims[0] = K2_DW_OC;
    filter_dims[1] = 3U;
    filter_dims[2] = 3U;
    filter_dims[3] = 1U;
    if (infr_shape_init(&filter_shape, 4U, filter_dims) != INFR_OK ||
        infr_tensor_init(&filter, (void *)kat_k2_dw_w, INFR_DTYPE_F32, &filter_shape) != INFR_OK) {
        return INFR_ERR_INVALID_ARGUMENT;
    }
    {
        infr_tensor pooled;
        if (bind_tensor(&pooled, g_pool_out, K2_POOLED_H, K2_POOLED_W, K2_CONV_OC) != INFR_OK) {
            return INFR_ERR_INVALID_ARGUMENT;
        }
        if (bind_tensor(&output, g_dw_out, K2_POOLED_H, K2_POOLED_W, K2_DW_OC) != INFR_OK) {
            return INFR_ERR_INVALID_ARGUMENT;
        }
        conv.in_channels = K2_CONV_OC;
        conv.out_channels = K2_DW_OC;
        conv.groups = K2_CONV_OC;
        status = infr_conv2d_f32(&pooled, &filter, kat_k2_dw_b, &conv, &output, &workspace);
        if (status != INFR_OK) {
            return status;
        }
    }
    K2_STAGE_CUT(9);
    status = run_relu(g_dw_out, K2_POOLED_H * K2_POOLED_W * K2_DW_OC);
    if (status != INFR_OK) {
        return status;
    }
    K2_STAGE_CUT(10);

    /* global mean over H,W */
    {
        infr_tensor dw_tensor;
        infr_tensor gap_tensor;
        if (bind_tensor(&dw_tensor, g_dw_out, K2_POOLED_H, K2_POOLED_W, K2_DW_OC) != INFR_OK ||
            bind_tensor(&gap_tensor, g_gap_out, 1U, 1U, K2_DW_OC) != INFR_OK) {
            return INFR_ERR_INVALID_ARGUMENT;
        }
        status = infr_mean_hw_f32(&dw_tensor, &gap_tensor, &workspace);
        if (status != INFR_OK) {
            return status;
        }
    }
    K2_STAGE_CUT(11);

    /* fully connected: one dot product per output channel */
    if (infr_kernel_register_scalar() != INFR_OK) {
        return INFR_ERR_NOT_IMPLEMENTED;
    }
    fc = infr_kernel_find("fc", INFR_DTYPE_F32, INFR_QUANT_NONE, INFR_TIER_SCALAR);
    if (fc == 0) {
        return INFR_ERR_NOT_IMPLEMENTED;
    }
    for (oc = 0U; oc < K2_FC_OC; ++oc) {
        const void *inputs[2];
        void *outputs[1];
        float acc = 0.0F;
        inputs[0] = g_gap_out;
        inputs[1] = &kat_k2_fc_w[oc * K2_DW_OC];
        outputs[0] = &acc;
        status = fc->fn(inputs, outputs, K2_DW_OC, 0);
        if (status != INFR_OK) {
            return status;
        }
        result->out[oc] = acc + kat_k2_fc_b[oc];
    }
    K2_STAGE_CUT(12);

    /* Second pass: the **same bytes** the host gtest reads, driven by the blob's own
     * node table. The hand-written path above is the independent re-spelling, so
     * requiring the two to be bitwise equal on every target is the whole point of
     * this chain -- and it is the evidence a host-only test cannot give.
     *
     * 第二遍：与宿主 gtest 读的**同一批字节**，由 blob 自己的节点表驱动。上面手写的路径
     * 就是独立重述，所以要求两者在每个目标上逐位相等，正是这条链的意义——也是只跑宿主的
     * 测试给不出的证据。 */
    {
        /* The size the layout needs is not this number: it is `view.arena_floats`,
         * reported back by the executor after it has placed every slot. This buffer is
         * the *capacity* (the earlier per-slot sum was 3866 floats; live-range reuse
         * brought it to 2816), and the check below makes the difference visible
         * instead of leaving a hand-picked constant as the only record.
         *
         * 需要的不是这个数，而是执行器排完布局后报回的 `view.arena_floats`。这块是**容量**
         * （按槽逐段相加曾是 3866 float，活跃区间复用后是 2816），下面的检查让这个差额有
         * 记录，而不是只剩一个手挑的常数。 */
        static float arena[4096];
        kat_blob_view view;
        int blob_status;
        blob_status = kat_blob_run(kat_k2_blob, KAT_K2_BLOB_LEN, kat_k2_input,
                                   (uint32_t)(K2_IN_H * K2_IN_W * K2_IN_C), arena,
                                   (uint32_t)(sizeof arena / sizeof arena[0]), &workspace, &view);
        if (blob_status != KAT_BLOB_OK) {
            return blob_status;
        }
        /* Two implementations of the same plan have to agree, and this is the only
         * place they can be compared: the figure below was derived in Python from the
         * blob at emit time, the one in the view was derived here in C at run time.
         * Disagreeing means one of them is wrong about the same bytes -- which is
         * exactly the "plan nobody executes" this check exists to stop.
         *
         * 同一份计划的两份实现必须一致，而这里是比较它们的唯一地方：宏是生成期由 Python
         * 从 blob 推出的，view 里的数是运行期由这里的 C 算出的。两者不等就说明其中之一把
         * 同一批字节读错了——这正是这条检查要拦住的“没人执行的计划”。 */
        if (view.arena_floats != KAT_K2_BLOB_ARENA) {
            return KAT_BLOB_ERR_ARENA;
        }
        if (view.arena_floats * sizeof(float) >= sizeof arena) {
            return KAT_BLOB_ERR_ARENA;
        }
        if (view.out_len != K2_FC_OC || memcmp(view.out, result->out, K2_FC_OC * sizeof(float)) != 0) {
            return KAT_BLOB_ERR_MISMATCH;
        }
    }

    /* The scratch has to be back where it started: a graph whose operators leak
     * workspace runs out of memory after enough layers, not after one. */
    if (infr_workspace_used(&workspace) != 0U) {
        return INFR_ERR_BUFFER_TOO_SMALL;
    }
    return INFR_OK;
}

unsigned kat_k2_expect_len(void) { return (unsigned)(K2_FC_OC * sizeof(float)); }

void kat_k2_expect(const kat_k2_result *result, unsigned char *out) {
    if (result == 0 || out == 0) {
        return;
    }
    memcpy(out, result->out, sizeof result->out);
}
