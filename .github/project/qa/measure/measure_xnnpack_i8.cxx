/* int8 convolution, this library against XNNPACK, on the host.
 *
 * int8 卷积：本库对 XNNPACK，跑在宿主上。
 *
 * The fp32 numbers are in measure_xnnpack_ab.cxx (gaps G-128/G-129). This file
 * exists because int8 is the path this repository just closed by hand, and the
 * question "should that path exist at all" is answered by the same yardstick,
 * not by the fact that it passes its parity tests.
 *
 * fp32 的数字在 measure_xnnpack_ab.cxx（gaps G-128/G-129）。这个文件存在是因为
 * int8 正是本仓刚手写闭环的那条路，而「这条路该不该存在」要用同一把尺子回答，
 * 不能因为它的 parity 测试过了就算数。
 *
 * Fairness, same rules as the fp32 harness: XNNPACK on one thread, weights packed
 * at setup, only the invoke timed, identical int8 buffers on both arms, outputs
 * compared.
 *
 * 公平性与 fp32 harness 相同：XNNPACK 单线程、权重在 setup 里打包、只计 invoke、
 * 两臂同一份 int8 缓冲区、比对输出。
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
    unsigned groups;
};

constexpr float kInputScale = 0.02F;
constexpr float kFilterScale = 0.01F;
constexpr float kOutputScale = 0.05F;

double time_ours(const Shape &s, const std::vector<std::int8_t> &in, const std::vector<std::int8_t> &w,
                 std::vector<float> *scales, std::vector<std::int8_t> *out, double *macs) {
    const unsigned out_h = s.in_h - s.k + 1;
    const unsigned out_w = s.in_w - s.k + 1;
    const unsigned k_in = s.in_c / s.groups;
    std::vector<std::int8_t> result(static_cast<size_t>(out_h) * out_w * s.out_c);

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
    filter_shape.dims[3] = k_in;
    filter_shape.elem_count = s.out_c * s.k * s.k * k_in;
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
    if (infr_tensor_init(&input, const_cast<std::int8_t *>(in.data()), INFR_DTYPE_I8, &in_shape) != INFR_OK ||
        infr_tensor_init(&filter, const_cast<std::int8_t *>(w.data()), INFR_DTYPE_I8, &filter_shape) != INFR_OK ||
        infr_tensor_init(&output, result.data(), INFR_DTYPE_I8, &out_shape) != INFR_OK) {
        std::perror("tensor_init");
        std::exit(1);
    }
    input.quant.kind = INFR_QUANT_PER_TENSOR;
    input.quant.scale = kInputScale;
    input.quant.zero_point = 0;
    filter.quant.kind = INFR_QUANT_PER_CHANNEL;
    filter.quant.scale = kFilterScale;
    filter.quant.zero_point = 0;
    filter.quant.channel_scales = scales->data();
    filter.quant.channel_zero_points = nullptr;
    filter.quant.channel_count = s.out_c;
    output.quant.kind = INFR_QUANT_PER_TENSOR;
    output.quant.scale = kOutputScale;
    output.quant.zero_point = 0;

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
    const infr_status first = infr_conv2d_i8(&input, &filter, nullptr, &params, &output, &ws);
    if (first != INFR_OK) {
        std::fprintf(stderr, "ours refused this shape: %d\n", static_cast<int>(first));
        return 0.0;
    }
    double best = 0.0;
    const int reps = std::getenv("INFR_MB_REPS") != nullptr ? std::atoi(std::getenv("INFR_MB_REPS")) : 7;
    for (int rep = 0; rep < reps; ++rep) {
        const auto t0 = std::chrono::steady_clock::now();
        const infr_status st = infr_conv2d_i8(&input, &filter, nullptr, &params, &output, &ws);
        const auto t1 = std::chrono::steady_clock::now();
        if (st != INFR_OK) {
            std::perror("conv2d_i8");
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
    *macs = static_cast<double>(out_h) * out_w * s.out_c * s.k * s.k * k_in;
    return best;
}

double time_xnnpack(const Shape &s, const std::vector<std::int8_t> &in, const std::vector<std::int8_t> &w,
                    std::vector<float> *scales, std::vector<std::int8_t> *out, bool *ok) {
    const unsigned out_h = s.in_h - s.k + 1;
    const unsigned out_w = s.in_w - s.k + 1;
    const unsigned k_in = s.in_c / s.groups;
    std::vector<std::int8_t> result(static_cast<size_t>(out_h) * out_w * s.out_c);
    *ok = false;

    xnn_subgraph_t subgraph = nullptr;
    xnn_status status = xnn_create_subgraph(/*external_value_ids=*/2, /*flags=*/0, &subgraph);
    if (status != xnn_status_success) {
        std::fprintf(stderr, "xnn_create_subgraph: %d\n", static_cast<int>(status));
        return 0.0;
    }
    const size_t in_dims[4] = {1, s.in_h, s.in_w, s.in_c};
    const size_t filter_dims[4] = {s.out_c, s.k, s.k, k_in};
    const size_t out_dims[4] = {1, out_h, out_w, s.out_c};
    uint32_t input_id = 0;
    uint32_t filter_id = 0;
    uint32_t output_id = 0;
    status = xnn_define_quantized_tensor_value(subgraph, xnn_datatype_qint8, /*zero_point=*/0, kInputScale, 4,
                                               in_dims, nullptr, /*external_id=*/0, XNN_VALUE_FLAG_EXTERNAL_INPUT,
                                               &input_id);
    if (status == xnn_status_success) {
        /* Per-channel filter, channel axis 0 (output channels) -- the qc8w layout
         * XNNPACK's fast path expects, and the same thing our filter tensor
         * describes with `channel_scales`.
         *
         * 逐通道滤波器，通道轴 0（输出通道）——XNNPACK 快路径要的 qc8w 布局，也正是
         * 我们那个滤波器张量用 `channel_scales` 描述的东西。 */
        status = xnn_define_channelwise_quantized_tensor_value(subgraph, xnn_datatype_qcint8, scales->data(), 4,
                                                               /*channel_dim=*/0, filter_dims, w.data(),
                                                               XNN_INVALID_VALUE_ID, /*flags=*/0, &filter_id);
    }
    if (status == xnn_status_success) {
        status = xnn_define_quantized_tensor_value(subgraph, xnn_datatype_qint8, /*zero_point=*/0, kOutputScale, 4,
                                                   out_dims, nullptr, /*external_id=*/1,
                                                   XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id);
    }
    if (status == xnn_status_success) {
        status = xnn_define_convolution_2d(subgraph, /*pad_top=*/0, /*pad_right=*/0, /*pad_bottom=*/0,
                                           /*pad_left=*/0, s.k, s.k, /*sub_height=*/1, /*sub_width=*/1,
                                           /*dilation_height=*/1, /*dilation_width=*/1, s.groups, k_in,
                                           s.out_c / s.groups, /*output_min=*/-INFINITY, /*output_max=*/INFINITY,
                                           input_id, filter_id, XNN_INVALID_VALUE_ID, output_id, /*flags=*/0);
    }
    if (status != xnn_status_success) {
        std::fprintf(stderr, "xnn_define_*: %d\n", static_cast<int>(status));
        xnn_delete_subgraph(subgraph);
        return 0.0;
    }

    pthreadpool_t pool = pthreadpool_create(1);
    xnn_runtime_t runtime = nullptr;
    status = xnn_create_runtime_v4(subgraph, nullptr, nullptr, pool, 0, &runtime);
    if (status == xnn_status_success) {
        status = xnn_reshape_runtime(runtime);
    }
    xnn_external_value external[2] = {
        {0, const_cast<std::int8_t *>(in.data())},
        {1, result.data()},
    };
    if (status == xnn_status_success) {
        status = xnn_setup_runtime_v2(runtime, 2, external);
    }
    if (status != xnn_status_success) {
        std::fprintf(stderr, "xnn runtime setup: %d\n", static_cast<int>(status));
        if (runtime != nullptr) {
            xnn_delete_runtime(runtime);
        }
        xnn_delete_subgraph(subgraph);
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
    if (xnn_initialize(nullptr) != xnn_status_success) {
        std::fprintf(stderr, "xnn_initialize failed\n");
        return 1;
    }
    std::printf("int8 convolution: ours (per-channel filter, per-tensor in/out) vs XNNPACK qs8/qc8w, one thread\n\n");
    std::printf("%-24s %11s %11s %8s %8s %9s\n", "layer", "ours ns", "xnn ns", "ours G/s", "xnn G/s", "ours/xnn");
    const Shape shapes[] = {
        {"i8 56x56x64 -> 64 3x3", 56, 56, 64, 64, 3, 1},
        {"i8 28x28x128 -> 128 3x3", 28, 28, 128, 128, 3, 1},
        {"i8 dw 56x56x128 3x3", 56, 56, 128, 128, 3, 128},
        /* Same call count as the 3x3 row, 2.78x the inner dimension: the same
         * control the fp32 harness used to show the cost is per-call overhead and
         * not arithmetic (gaps G-129). If int8 repeats that pattern, one fix
         * covers both dtypes; if it does not, the two need separate treatment.
         *
         * 与 3x3 行同样的调用次数、2.78 倍内维：fp32 harness 用这条对照证明成本是每次
         * 调用的固定开销而不是算术（gaps G-129）。若 int8 重复同一形态，一个修复就能
         * 覆盖两种 dtype；若不重复，两者要分开处理。 */
        {"i8 dw 56x56x128 5x5", 56, 56, 128, 128, 5, 128},
        /* 4x the MACs of the first row: if the time does not scale with the work,
         * the number is overhead, not throughput, and must not be published.
         *
         * 第一行 4 倍的 MAC：若时间不随工作量走，这个数就是开销而不是吞吐，不能发。 */
        {"i8 112x112x64 -> 64 3x3", 112, 112, 64, 64, 3, 1},
    };
    for (const Shape &s : shapes) {
        const unsigned k_in = s.in_c / s.groups;
        std::vector<std::int8_t> in(static_cast<size_t>(s.in_h) * s.in_w * s.in_c);
        std::vector<std::int8_t> w(static_cast<size_t>(s.out_c) * s.k * s.k * k_in);
        std::vector<float> scales(s.out_c, kFilterScale);
        for (size_t i = 0; i < in.size(); ++i) {
            in[i] = static_cast<std::int8_t>(static_cast<int>((i * 37U) % 61U) - 30);
        }
        for (size_t i = 0; i < w.size(); ++i) {
            w[i] = static_cast<std::int8_t>(static_cast<int>((i * 53U) % 51U) - 25);
        }
        std::vector<std::int8_t> ours;
        std::vector<std::int8_t> theirs;
        double macs = 0.0;
        const double t_ours = time_ours(s, in, w, &scales, &ours, &macs);
        bool ok = false;
        const double t_xnn = time_xnnpack(s, in, w, &scales, &theirs, &ok);
        if (t_ours == 0.0 || !ok) {
            std::printf("%-24s %11s %11s  (one arm refused this shape)\n", s.name, t_ours == 0.0 ? "-" : "ok",
                        ok ? "ok" : "-");
            continue;
        }
        /* int8 comparison is exact: both arms round the same int32 accumulator
         * through the same requantisation, so a difference is a difference in
         * the implementation rather than in rounding.
         *
         * int8 比对是逐位精确的：两臂把同一个 int32 累加器过同一套再量化，出现差异
         * 就说明实现不同，而不是舍入不同。 */
        long sum_ours = 0;
        long sum_theirs = 0;
        for (size_t i = 0; i < ours.size(); ++i) {
            sum_ours += ours[i];
        }
        for (size_t i = 0; i < theirs.size(); ++i) {
            sum_theirs += theirs[i];
        }
        long worst = 0;
        if (ours.size() == theirs.size()) {
            for (size_t i = 0; i < ours.size(); ++i) {
                const long d = std::labs(static_cast<long>(ours[i]) - static_cast<long>(theirs[i]));
                if (d > worst) {
                    worst = d;
                }
            }
        }
        std::printf("%-24s %11.0f %11.0f %8.2f %8.2f %9.2f   diff %ld  sums %ld/%ld\n", s.name, t_ours, t_xnn,
                    macs / t_ours, macs / t_xnn, t_ours / t_xnn, worst, sum_ours, sum_theirs);
    }
    return 0;
}
