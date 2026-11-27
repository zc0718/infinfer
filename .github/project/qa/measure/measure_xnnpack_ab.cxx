/* Operator-level A/B: this library's conv2d against XNNPACK's, on the host.
 *
 * 算子级 A/B：本库的 conv2d 对 XNNPACK 的 conv，跑在宿主上。
 *
 * Why the host: per ADR-008 a magnitude claim needs an instrument calibrated
 * against ground truth. The x86-64 host is the only place in this repo where the
 * instrument *is* the ground truth, so this is the only place a "we are X times
 * faster/slower than the vendor wheel" number is allowed to come from.
 *
 * 为什么用宿主：按 ADR-008，幅度结论要求仪器在有真值的地方标定过。x86-64 宿主是本仓
 * 唯一「仪器即真值」的地方，所以要下「我们比厂商轮子快/慢几倍」这种话，只能在这里量。
 *
 * Fairness, stated rather than assumed:
 *  - XNNPACK runs with **one** thread. This library is single-threaded, so a
 *    multi-threaded XNNPACK would compare a library against a machine.
 *  - XNNPACK packs its weights once, at setup; only `xnn_invoke_runtime` is
 *    timed. That matches how an inference engine amortises a model load, and it
 *    is the arrangement most favourable to XNNPACK -- which is the right way to
 *    ask "is our wheel worse", because a verdict must survive the strongest case
 *    for the other side.
 *  - Both arms are fed the same NHWC buffers and the same OHWI filters, and the
 *    outputs are compared, so a speed number cannot come from doing less work.
 *
 * 公平性写出来而不是假设：XNNPACK 只用 **1 个线程**（本库是单线程，否则是拿库比机器）；
 * XNNPACK 的权重打包在 setup 里做、只计 `xnn_invoke_runtime` 的时间（这正是对 XNNPACK
 * 最有利的排法——判「我们的轮子是不是更差」必须扛得住对方最强的情形）；两臂喂同样的
 * NHWC 输入与 OHWI 权重，并比对输出，避免「少算了活」被当成更快。
 */

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <infinfer/infinfer.h>
#include <pthreadpool.h>
#include <xnnpack.h>

namespace {

struct Shape {
    const char *name;
    unsigned in_h, in_w, in_c, out_c, k;
    unsigned groups; /* 1 = dense, in_c = depthwise */
};

/* This library: one call per output pixel per output channel, the x86 tier. */
double time_ours(const Shape &s, const std::vector<float> &in, const std::vector<float> &w, std::vector<float> *out,
                 double *macs) {
    const unsigned out_h = s.in_h - s.k + 1;
    const unsigned out_w = s.in_w - s.k + 1;
    std::vector<float> result(static_cast<size_t>(out_h) * out_w * s.out_c);

    infr_shape in_shape{};
    in_shape.rank = 4;
    in_shape.dims[0] = 1;
    in_shape.dims[1] = s.in_h;
    in_shape.dims[2] = s.in_w;
    in_shape.dims[3] = s.in_c;
    in_shape.elem_count = 1U * s.in_h * s.in_w * s.in_c;
    infr_shape filter_shape{};
    filter_shape.rank = 4;
    filter_shape.dims[0] = s.out_c;
    filter_shape.dims[1] = s.k;
    filter_shape.dims[2] = s.k;
    filter_shape.dims[3] = s.in_c / s.groups;
    filter_shape.elem_count = s.out_c * s.k * s.k * (s.in_c / s.groups);
    infr_shape out_shape{};
    out_shape.rank = 4;
    out_shape.dims[0] = 1;
    out_shape.dims[1] = out_h;
    out_shape.dims[2] = out_w;
    out_shape.dims[3] = s.out_c;
    out_shape.elem_count = 1U * out_h * out_w * s.out_c;

    infr_tensor input{};
    infr_tensor filter{};
    infr_tensor output{};
    if (infr_tensor_init(&input, const_cast<float *>(in.data()), INFR_DTYPE_F32, &in_shape) != INFR_OK ||
        infr_tensor_init(&filter, const_cast<float *>(w.data()), INFR_DTYPE_F32, &filter_shape) != INFR_OK ||
        infr_tensor_init(&output, result.data(), INFR_DTYPE_F32, &out_shape) != INFR_OK) {
        std::perror("tensor_init");
        std::exit(1);
    }
    infr_conv2d_params params{};
    params.kernel_h = s.k;
    params.kernel_w = s.k;
    params.stride_h = 1;
    params.stride_w = 1;
    params.in_channels = s.in_c;
    params.out_channels = s.out_c;
    params.groups = s.groups;

    std::vector<unsigned char> arena(1U << 20);
    infr_workspace ws{};
    if (infr_workspace_init(&ws, arena.data(), arena.size()) != INFR_OK) {
        std::perror("workspace");
        std::exit(1);
    }
    if (infr_kernel_init() != INFR_OK) {
        std::perror("init");
        std::exit(1);
    }

    (void)infr_conv2d_f32(&input, &filter, nullptr, &params, &output, &ws);
    double best = 0.0;
    const int reps = std::getenv("INFR_MB_REPS") != nullptr ? std::atoi(std::getenv("INFR_MB_REPS")) : 7;
    for (int rep = 0; rep < reps; ++rep) {
        const auto t0 = std::chrono::steady_clock::now();
        const infr_status st = infr_conv2d_f32(&input, &filter, nullptr, &params, &output, &ws);
        const auto t1 = std::chrono::steady_clock::now();
        if (st != INFR_OK) {
            std::perror("conv2d");
            std::exit(1);
        }
        const double ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
        if (best == 0.0 || ns < best) {
            best = ns;
        }
    }
    if (out != nullptr) {
        *out = result;
    }
    *macs = static_cast<double>(out_h) * out_w * s.out_c * s.k * s.k * (s.in_c / s.groups);
    return best;
}

/* XNNPACK: one subgraph, one convolution node, timed at `xnn_invoke_runtime`. */
double time_xnnpack(const Shape &s, const std::vector<float> &in, const std::vector<float> &w,
                    std::vector<float> *out, bool *ok) {
    const unsigned out_h = s.in_h - s.k + 1;
    const unsigned out_w = s.in_w - s.k + 1;
    std::vector<float> result(static_cast<size_t>(out_h) * out_w * s.out_c);
    *ok = false;

    xnn_status status;
    xnn_subgraph_t subgraph = nullptr;
    status = xnn_create_subgraph(/*external_value_ids=*/2, /*flags=*/0, &subgraph);
    if (status != xnn_status_success) {
        std::fprintf(stderr, "xnn_create_subgraph: %d\n", static_cast<int>(status));
        return 0.0;
    }
    const size_t in_dims[4] = {1, s.in_h, s.in_w, s.in_c};
    const size_t filter_dims[4] = {s.out_c, s.k, s.k, s.in_c / s.groups};
    const size_t out_dims[4] = {1, out_h, out_w, s.out_c};
    uint32_t input_id = 0;
    uint32_t filter_id = 0;
    uint32_t output_id = 0;
    /* `flags` declares the role. With 0 the subgraph has no external tensors at
     * all, and the later `setup_runtime_v2(..., 2, external)` answers
     * invalid_parameter -- which is what the first two attempts died on.
     *
     * `flags` 声明角色。传 0 时子图里等于没有外部张量，随后
     * `setup_runtime_v2(..., 2, external)` 就报 invalid_parameter——前两次失败都死在这里。 */
    status = xnn_define_tensor_value(subgraph, xnn_datatype_fp32, 4, in_dims, nullptr, /*external_id=*/0,
                                     XNN_VALUE_FLAG_EXTERNAL_INPUT, &input_id);
    if (status == xnn_status_success) {
        status = xnn_define_tensor_value(subgraph, xnn_datatype_fp32, 4, filter_dims, w.data(),
                                         XNN_INVALID_VALUE_ID, /*flags=*/0, &filter_id);
    }
    if (status == xnn_status_success) {
        status = xnn_define_tensor_value(subgraph, xnn_datatype_fp32, 4, out_dims, nullptr, /*external_id=*/1,
                                         XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id);
    }
    if (status == xnn_status_success) {
        status = xnn_define_convolution_2d(subgraph, /*pad_top=*/0, /*pad_right=*/0, /*pad_bottom=*/0,
                                           /*pad_left=*/0, s.k, s.k, /*sub_height=*/1, /*sub_width=*/1,
                                           /*dilation_height=*/1, /*dilation_width=*/1, s.groups,
                                           /*group_input_channels=*/s.in_c / s.groups,
                                           /*group_output_channels=*/s.out_c / s.groups,
                                           /*output_min=*/-INFINITY, /*output_max=*/INFINITY, input_id, filter_id,
                                           XNN_INVALID_VALUE_ID, output_id, /*flags=*/0);
    }
    if (status != xnn_status_success) {
        std::fprintf(stderr, "xnn_define_*: %d\n", static_cast<int>(status));
        xnn_delete_subgraph(subgraph);
        return 0.0;
    }

    pthreadpool_t pool = pthreadpool_create(1);
    xnn_runtime_t runtime = nullptr;
    status = xnn_create_runtime_v4(subgraph, /*weights_cache=*/nullptr, /*workspace=*/nullptr, pool,
                                   /*flags=*/0, &runtime);
    if (status != xnn_status_success) {
        std::fprintf(stderr, "xnn_create_runtime_v4: %d\n", static_cast<int>(status));
        xnn_delete_subgraph(subgraph);
        pthreadpool_destroy(pool);
        return 0.0;
    }
    xnn_external_value external[2] = {
        {0, const_cast<float *>(in.data())},
        {1, result.data()},
    };
    /* `xnn_setup_runtime` is deprecated in this version and answers
     * invalid_parameter; the supported pair is reshape then setup_v2.
     *
     * `xnn_setup_runtime` 在这个版本里已废弃并返回 invalid_parameter；受支持的组合
     * 是 reshape 之后 setup_v2。 */
    status = xnn_reshape_runtime(runtime);
    std::fprintf(stderr, "  reshape=%d", static_cast<int>(status));
    if (status == xnn_status_success) {
        status = xnn_setup_runtime_v2(runtime, 2, external);
        std::fprintf(stderr, " setup=%d", static_cast<int>(status));
    }
    std::fprintf(stderr, "\n");
    if (status != xnn_status_success) {
        std::fprintf(stderr, "xnn_reshape/setup_runtime: %d\n", static_cast<int>(status));
        xnn_delete_runtime(runtime);
        pthreadpool_destroy(pool);
        return 0.0;
    }

    (void)xnn_invoke_runtime(runtime);
    double best = 0.0;
    const int reps = std::getenv("INFR_MB_REPS") != nullptr ? std::atoi(std::getenv("INFR_MB_REPS")) : 7;
    for (int rep = 0; rep < reps; ++rep) {
        const auto t0 = std::chrono::steady_clock::now();
        status = xnn_invoke_runtime(runtime);
        const auto t1 = std::chrono::steady_clock::now();
        if (status != xnn_status_success) {
            std::fprintf(stderr, "xnn_invoke_runtime: %d\n", static_cast<int>(status));
            break;
        }
        const double ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
        if (best == 0.0 || ns < best) {
            best = ns;
        }
    }
    if (out != nullptr) {
        *out = result;
    }
    xnn_delete_runtime(runtime);
    xnn_delete_subgraph(subgraph);
    pthreadpool_destroy(pool);
    *ok = best != 0.0;
    return best;
}

}  // namespace

int main() {
    /* This XNNPACK build exposes no version accessor; the identity that matters
     * is which package the numbers came from, and that is the commit/recipe
     * recorded next to them, not a string.
     *
     * 这个 XNNPACK 构建没有版本访问接口；真正要紧的不是字符串，而是这些数字来自哪个
     * 包——那由记录在旁边的 recipe 版本说明。 */
    /* XNNPACK refuses to build a subgraph before the library is initialised
     * (status 1 = xnn_status_uninitialized), which is where it detects the host
     * ISA. `nullptr` hardware means "detect it yourself".
     *
     * XNNPACK 在库初始化之前拒绝建子图（status 1 = xnn_status_uninitialized），
     * 而那一步正是它探测宿主 ISA 的地方。`nullptr` 表示「自己去探测」。 */
    if (xnn_initialize(nullptr) != xnn_status_success) {
        std::fprintf(stderr, "xnn_initialize failed\n");
        return 1;
    }
    std::printf("XNNPACK cci.20241203, one thread\n\n");
    std::printf("%-24s %11s %11s %8s %8s %9s\n", "layer", "ours ns", "xnn ns", "ours G/s", "xnn G/s", "ours/xnn");
    const Shape shapes[] = {
        {"5x5x2 -> dense 3x3x2", 5, 5, 2, 3, 3, 1},
        {"56x56x64 -> 64 3x3", 56, 56, 64, 64, 3, 1},
        {"28x28x128 -> 128 3x3", 28, 28, 128, 128, 3, 1},
        {"14x14x256 -> 256 3x3", 14, 14, 256, 256, 3, 1},
        {"dw 56x56x128 3x3", 56, 56, 128, 128, 3, 128},
        {"dw 112x112x32 3x3", 112, 112, 32, 32, 3, 32},
        /* Same call count as the 3x3 depthwise row, 2.8x the inner dimension:
         * if the cost is per-call overhead rather than arithmetic, the time must
         * barely move. That is the control that separates the two explanations.
         *
         * 与 3x3 深卷积同样的调用次数、2.8 倍的内维：若成本是每次调用的固定开销而不是
         * 算术，时间几乎不会变。这条对照正是用来分开这两种解释的。 */
        {"dw 56x56x128 5x5", 56, 56, 128, 128, 5, 128},
    };
    for (const Shape &s : shapes) {
        std::vector<float> in(static_cast<size_t>(s.in_h) * s.in_w * s.in_c);
        std::vector<float> w(static_cast<size_t>(s.out_c) * s.k * s.k * (s.in_c / s.groups));
        for (size_t i = 0; i < in.size(); ++i) {
            in[i] = static_cast<float>((i * 37U) % 251U) / 251.0F - 0.5F;
        }
        for (size_t i = 0; i < w.size(); ++i) {
            w[i] = static_cast<float>((i * 53U) % 199U) / 199.0F - 0.5F;
        }
        std::vector<float> ours;
        std::vector<float> theirs;
        double macs = 0.0;
        const double t_ours = time_ours(s, in, w, &ours, &macs);
        bool ok = false;
        const double t_xnn = time_xnnpack(s, in, w, &theirs, &ok);
        double worst = 0.0;
        if (ok && ours.size() == theirs.size()) {
            for (size_t i = 0; i < ours.size(); ++i) {
                worst = std::fmax(worst, std::fabs(static_cast<double>(ours[i]) - static_cast<double>(theirs[i])));
            }
        }
        if (!ok) {
            std::printf("%-24s %11.0f %11s  (xnnpack refused this shape)\n", s.name, t_ours, "-");
            continue;
        }
        std::printf("%-24s %11.0f %11.0f %8.2f %8.2f %9.2f   max-abs-diff %.2g\n", s.name, t_ours, t_xnn,
                    macs / t_ours, macs / t_xnn, t_ours / t_xnn, worst);
    }
    return 0;
}
