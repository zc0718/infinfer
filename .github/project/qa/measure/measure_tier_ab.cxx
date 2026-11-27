/* Operator-level A/B of the tier policy, on the host.
 *
 * 宿主上的档位策略算子级 A/B。这个文件在 workspace 下（本地证据，不入库）。
 *
 * Build: g++ -O2 -std=c++17 -Iinclude measure_tier_ab.cxx <libinfinfer_cpp.a>
 * <libinfinfer_c.a>, with both archives taken from the Conan build folder that
 * `conan create` leaves behind.
 */

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <infinfer/infinfer.h>

namespace {

struct Shape {
    const char *name;
    unsigned in_h, in_w, in_c, out_c, k;
};

double time_conv(const Shape &s, infr_tier tier, std::vector<float> *out_snapshot, double *macs_out) {
    const unsigned out_h = s.in_h - s.k + 1;
    const unsigned out_w = s.in_w - s.k + 1;
    const unsigned in_h_count = 1U * s.in_h * s.in_w * s.in_c;
    const unsigned w_count = s.out_c * s.k * s.k * s.in_c;
    const unsigned out_count = 1U * out_h * out_w * s.out_c;
    std::vector<float> in(static_cast<size_t>(s.in_h) * s.in_w * s.in_c);
    std::vector<float> w(static_cast<size_t>(s.out_c) * s.k * s.k * s.in_c);
    std::vector<float> out(static_cast<size_t>(out_h) * out_w * s.out_c);
    for (size_t i = 0; i < in.size(); ++i) {
        in[i] = static_cast<float>((i * 37U) % 251U) / 251.0F - 0.5F;
    }
    for (size_t i = 0; i < w.size(); ++i) {
        w[i] = static_cast<float>((i * 53U) % 199U) / 199.0F - 0.5F;
    }

    infr_tensor input{};
    infr_tensor filter{};
    infr_tensor output{};
    infr_shape in_shape{};
    infr_shape filter_shape{};
    infr_shape out_shape{};
    in_shape.rank = 4;
    in_shape.dims[0] = 1;
    in_shape.dims[1] = s.in_h;
    in_shape.dims[2] = s.in_w;
    in_shape.dims[3] = s.in_c;
    filter_shape.rank = 4;
    filter_shape.dims[0] = s.out_c;
    filter_shape.dims[1] = s.k;
    filter_shape.dims[2] = s.k;
    filter_shape.dims[3] = s.in_c;
    out_shape.rank = 4;
    out_shape.dims[0] = 1;
    out_shape.dims[1] = out_h;
    out_shape.dims[2] = out_w;
    out_shape.dims[3] = s.out_c;
    in_shape.elem_count = in_h_count;
    filter_shape.elem_count = w_count;
    out_shape.elem_count = out_count;
    if (infr_tensor_init(&input, in.data(), INFR_DTYPE_F32, &in_shape) != INFR_OK ||
        infr_tensor_init(&filter, w.data(), INFR_DTYPE_F32, &filter_shape) != INFR_OK ||
        infr_tensor_init(&output, out.data(), INFR_DTYPE_F32, &out_shape) != INFR_OK) {
        std::perror("tensor_init");
        std::exit(1);
    }

    infr_conv2d_params params{};
    params.kernel_h = s.k;
    params.kernel_w = s.k;
    params.stride_h = 1;
    params.stride_w = 1;
    params.pad_h = 0;
    params.pad_w = 0;
    params.in_channels = s.in_c;
    params.out_channels = s.out_c;
    params.groups = 1;

    std::vector<unsigned char> arena(1U << 20);
    infr_workspace ws{};
    if (infr_workspace_init(&ws, arena.data(), arena.size()) != INFR_OK) {
        std::perror("workspace");
        std::exit(1);
    }
    if (infr_kernel_set_tier(tier) != INFR_OK) {
        std::perror("tier");
        std::exit(1);
    }

    /* Warm up, then take the best of several runs: the measurement is a ratio
     * between two policies on one machine, so the minimum is the honest statistic.
     *
     * 先热身再取多轮最小值：这是同一台机器上两种策略的比值，最小值才是诚实统计量。 */
    (void)infr_conv2d_f32(&input, &filter, nullptr, &params, &output, &ws);
    double best = 0.0;
    /* Repetition count from the environment: emulated runs are orders of
     * magnitude slower, and the emulator control only needs a ratio.
     *
     * 重复次数取自环境变量：模拟运行慢几个数量级，而模拟器的对照只需要一个比值。 */
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
    if (out_snapshot != nullptr) {
        *out_snapshot = out;
    }
    *macs_out = static_cast<double>(out_h) * out_w * s.out_c * s.k * s.k * s.in_c;
    return best;
}

}  // namespace

int main() {
    /* The second arm has to be **this build's** platform tier, not x86 by name.
     * The first version named `INFR_TIER_X86` and produced a flat 1.00x on
     * aarch64 with max-abs-diff 0 on every layer: both arms had fallen back to
     * the reference, because the x86 tier has no row in an AArch64 build. A
     * measurement that cannot tell "the vector body is not faster" from "the
     * vector body did not run" is worse than no measurement at all.
     *
     * 第二组必须是**本构建**的平台档，不能按名字写 x86。第一版写的就是
     * `INFR_TIER_X86`，在 aarch64 上得到平坦的 1.00x、每层 max-abs-diff 为 0：
     * 两组都退到了参考实现，因为 AArch64 构建里 x86 档根本没有行。一个分不清
     * 「向量体不快」与「向量体根本没跑」的测量，比没有测量更糟。 */
    if (infr_kernel_init() != INFR_OK) {
        std::perror("init");
        return 1;
    }
    const infr_tier vector_tier = infr_kernel_tier();
    if (vector_tier == INFR_TIER_SCALAR) {
        std::printf("this build carries no vector tier, so there is nothing to compare\n");
        return 0;
    }
    std::printf("vector arm: tier %d (0=scalar 2=x86 3=arm_neon 4=arm_m 5=rvv)\n\n",
                static_cast<int>(vector_tier));
    std::printf("%-26s %10s %10s %9s %9s %8s\n", "layer", "scalar ns", "vector ns", "scalar G/s", "vector G/s",
                "speedup");
    const Shape shapes[] = {
        {"5x5x2 -> dense 3x3x2", 5, 5, 2, 3, 3},
        {"3x3x3 -> dw 3x3 (16x16)", 16, 16, 3, 3, 3},
        {"56x56x64 -> 64 3x3", 56, 56, 64, 64, 3},
        {"28x28x128 -> 128 3x3", 28, 28, 128, 128, 3},
        {"14x14x256 -> 256 3x3", 14, 14, 256, 256, 3},
    };
    double total_scalar = 0.0;
    double total_vector = 0.0;
    for (const Shape &s : shapes) {
        std::vector<float> a;
        std::vector<float> b;
        double macs = 0.0;
        const double t_scalar = time_conv(s, INFR_TIER_SCALAR, &a, &macs);
        const double t_vector = time_conv(s, vector_tier, &b, &macs);
        double worst = 0.0;
        for (size_t i = 0; i < a.size(); ++i) {
            worst = std::fmax(worst, std::fabs(static_cast<double>(a[i]) - static_cast<double>(b[i])));
        }
        std::printf("%-26s %10.0f %10.0f %9.2f %9.2f %8.2fx   max|diff| %.3g\n", s.name, t_scalar, t_vector,
                    macs / t_scalar, macs / t_vector, t_scalar / t_vector, worst);
        total_scalar += t_scalar;
        total_vector += t_vector;
    }
    std::printf("\ntotal  %.0f ns -> %.0f ns   (%.2fx)\n", total_scalar, total_vector, total_scalar / total_vector);
    return 0;
}
