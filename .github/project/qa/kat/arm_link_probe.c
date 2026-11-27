/* Bare-metal link probe: call one real L2 operator through the public C ABI.
 *
 * The point is not the arithmetic (a 1x1x1x1 convolution is trivial) but the linkage:
 * if this image builds and runs, the C++17 orchestration is usable from a C image on
 * a Cortex-M target, and a graph-level KAT can follow. The exit code carries the
 * verdict so the runner needs no console.
 *
 * 裸机链接探针：通过公开 C ABI 调用一个真实的 L2 算子。重点不是算术（1x1x1x1 卷积
 * 平凡得很）而是**链接**：这张镜像能构建并跑起来，就说明 C++17 编排层可以被 Cortex-M
 * 上的 C 镜像使用，图级 KAT 才有落点。判定通过退出码传出，不需要控制台。 */

#include <stdint.h>

#include <infinfer/infinfer.h>

/* Semihosting exit: ARM's documented SYS_EXIT_EXTENDED, which lets QEMU turn the
 * application's status into its own exit code. */
static void arm_exit(uint32_t code) {
    /* The block has to live in memory the debugger can read: a compound literal in
     * the register initialiser is folded into a static one, which is why the first
     * version of this probe would not compile. */
    static uint32_t block[2];
    block[0] = 0x20026U; /* ADP_Stopped_ApplicationExit */
    block[1] = code;
    register uint32_t r0 __asm__("r0") = 0x20U; /* SYS_EXIT_EXTENDED */
    register uint32_t r1 __asm__("r1") = (uint32_t)(uintptr_t)block;
    __asm__ volatile("bkpt 0xAB" : : "r"(r0), "r"(r1) : "memory");
    for (;;) {
    }
}

/* Entry: nothing has set the stack pointer yet, because there is no crt0 here. */
__attribute__((naked)) void _start(void) {
    __asm__ volatile("ldr sp, =0x00400000\n"
                     "bl main\n"
                     "b .\n");
}

int main(void) {
    /* Zero-initialised statics only, values assigned below: an initialised static
     * would live in .data and need a startup copy table, which this probe has no
     * crt0 for. Doing it this way keeps the probe about linkage. */
    static float input_data[1];
    static float filter_data[1];
    static float output_data[1];
    static uint8_t workspace_bytes[256] __attribute__((aligned(8)));
    const float bias = 1.0F;

    input_data[0] = 3.0F;
    filter_data[0] = 2.0F;

    const uint32_t dims[4] = {1U, 1U, 1U, 1U};
    infr_shape shape;
    infr_tensor input;
    infr_tensor filter;
    infr_tensor output;
    infr_workspace workspace;

    if (infr_shape_init(&shape, 4U, dims) != INFR_OK) {
        arm_exit(11U);
    }
    if (infr_tensor_init(&input, input_data, INFR_DTYPE_F32, &shape) != INFR_OK ||
        infr_tensor_init(&filter, filter_data, INFR_DTYPE_F32, &shape) != INFR_OK ||
        infr_tensor_init(&output, output_data, INFR_DTYPE_F32, &shape) != INFR_OK) {
        arm_exit(12U);
    }
    if (infr_workspace_init(&workspace, workspace_bytes, sizeof(workspace_bytes)) != INFR_OK) {
        arm_exit(13U);
    }

    const infr_conv2d_params params = {.kernel_h = 1U,
                                       .kernel_w = 1U,
                                       .stride_h = 1U,
                                       .stride_w = 1U,
                                       .pad_h = 0U,
                                       .pad_w = 0U,
                                       .in_channels = 1U,
                                       .out_channels = 1U,
                                       .groups = 1U};
    if (infr_conv2d_f32(&input, &filter, &bias, &params, &output, &workspace) != INFR_OK) {
        arm_exit(14U);
    }

    /* 3 * 2 + 1 = 7: a wrong number here means the composition ran but computed
     * something else, which is a different failure from a link failure. */
    if (output_data[0] != 7.0F) {
        arm_exit(15U);
    }
    /* The scratch must be back: a graph runs many operators in one workspace. */
    if (infr_workspace_used(&workspace) != 0U) {
        arm_exit(16U);
    }

    arm_exit(0U);
    return 0;
}
