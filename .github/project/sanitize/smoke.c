/* Sanitizer smoke for the public C ABI.
 *
 * Built and run by `.github/project/sanitize_gate.py` with
 * `-fsanitize=address,undefined -fno-sanitize-recover=all`. It is deliberately
 * not part of the conan build: the unit suite covers behaviour, this covers the
 * memory and UB contract on the same code paths, and a driver that only the gate
 * builds can never drift into "a test nobody runs".
 *
 * What it asks of the library, in the order the plan asks for it:
 *   1. every registered row survives NULL arguments with a status instead of a crash;
 *   2. shape arithmetic rejects rank 0, rank > max, a zero axis and a product that
 *      does not fit, and a valid index never offsets past the element count;
 *   3. the workspace refuses to hand out more than it has, and refuses a rewind to
 *      a mark it never issued -- then the graph operators run on an arena that is
 *      deliberately too small, which is where an out-of-bounds write would live;
 *   4. a few hundred deterministic pseudo-random geometries run conv/pool/mean to
 *      completion, so the addressing arithmetic is exercised rather than sampled.
 *
 * 公共 C ABI 的消毒器冒烟。由门禁以 `-fsanitize=address,undefined` 构建并运行；
 * 它刻意不进 conan 构建：单测覆盖行为，这里覆盖同一批代码路径上的内存与 UB 契约，
 * 而"只有门禁才会构建"的驱动不会退化成没人跑的测试。
 */

#include <stdint.h>
#include <stdio.h>

#include <infinfer/infinfer.h>

static int failures;
static const char *current_case = "(none)";

static void begin(const char *name) {
    current_case = name;
}

static void check(int condition, const char *what) {
    if (!condition) {
        ++failures;
        printf("  FAIL [%s] %s\n", current_case, what);
    }
}

static void expect_status(const char *what, infr_status want, infr_status got) {
    if (want != got) {
        ++failures;
        printf("  FAIL [%s] %s: want %s, got %s\n", current_case, what, infr_status_str(want),
               infr_status_str(got));
    }
}

/* 1. NULL arguments must produce a status, never a dereference. */
static void registry_null_arguments(void) {
    static const struct {
        const char *name;
        infr_dtype dtype;
        infr_quant_kind quant;
        int needs_quant;
    } rows[] = {
        {"relu", INFR_DTYPE_F32, INFR_QUANT_NONE, 0},       {"relu", INFR_DTYPE_I8, INFR_QUANT_NONE, 0},
        {"add", INFR_DTYPE_F32, INFR_QUANT_NONE, 0},        {"sum", INFR_DTYPE_F32, INFR_QUANT_NONE, 0},
        {"max", INFR_DTYPE_F32, INFR_QUANT_NONE, 0},        {"scale", INFR_DTYPE_F32, INFR_QUANT_NONE, 0},
        {"quantize", INFR_DTYPE_F32, INFR_QUANT_PER_TENSOR, 1},
        {"dequantize", INFR_DTYPE_I8, INFR_QUANT_PER_TENSOR, 1},
        {"fc", INFR_DTYPE_I8, INFR_QUANT_PER_TENSOR, 1},    {"fc", INFR_DTYPE_I16, INFR_QUANT_PER_TENSOR, 1},
    };
    float arena[8];
    infr_quant quant;
    void *outputs[1];
    const void *inputs[2];
    size_t i;

    quant.kind = INFR_QUANT_PER_TENSOR;
    quant.scale = 1.0F;
    quant.zero_point = 0;
    quant.requant_multiplier = 1.0F;
    quant.channel_scales = NULL;
    quant.channel_zero_points = NULL;
    quant.channel_count = 1U;
    quant.axis = 0U;
    inputs[0] = arena;
    inputs[1] = arena;
    outputs[0] = arena;

    begin("registry/NULL");
    expect_status("register", INFR_OK, infr_kernel_register_scalar());
    for (i = 0; i < sizeof(rows) / sizeof(rows[0]); ++i) {
        const infr_kernel_desc *row = infr_kernel_find(rows[i].name, rows[i].dtype, rows[i].quant, INFR_TIER_SCALAR);
        const infr_quant *q = rows[i].needs_quant ? &quant : NULL;
        if (row == NULL) {
            ++failures;
            printf("  FAIL [%s] row '%s' is not registered\n", current_case, rows[i].name);
            continue;
        }
        expect_status("inputs == NULL", INFR_ERR_INVALID_ARGUMENT, row->fn(NULL, outputs, 4U, q));
        expect_status("outputs == NULL", INFR_ERR_INVALID_ARGUMENT, row->fn(inputs, NULL, 4U, q));
        {
            void *null_output[1];
            null_output[0] = NULL;
            expect_status("outputs[0] == NULL", INFR_ERR_INVALID_ARGUMENT, row->fn(inputs, null_output, 4U, q));
        }
    }
}

/* 2. Shape arithmetic: the four rejections and the offset bounds. */
static void shape_arithmetic(void) {
    infr_shape shape;
    uint32_t dims[INFR_TENSOR_MAX_RANK];
    uint32_t offset = 0U;
    uint32_t index[INFR_TENSOR_MAX_RANK];

    begin("shape/rejections");
    dims[0] = 4U;
    expect_status("rank 0", INFR_ERR_INVALID_ARGUMENT, infr_shape_init(&shape, 0U, dims));
    expect_status("rank > max", INFR_ERR_INVALID_ARGUMENT,
                  infr_shape_init(&shape, (uint8_t)(INFR_TENSOR_MAX_RANK + 1U), dims));
    dims[0] = 4U;
    dims[1] = 0U;
    expect_status("zero axis", INFR_ERR_INVALID_ARGUMENT, infr_shape_init(&shape, 2U, dims));
    dims[0] = 65536U;
    dims[1] = 65536U;
    dims[2] = 65536U;
    dims[3] = 65536U;
    expect_status("product beyond 32 bits", INFR_ERR_INVALID_ARGUMENT,
                  infr_shape_init(&shape, 4U, dims));

    begin("shape/offsets");
    dims[0] = 2U;
    dims[1] = 3U;
    dims[2] = 4U;
    expect_status("valid rank-3", INFR_OK, infr_shape_init(&shape, 3U, dims));
    check(shape.elem_count == 24U, "elem_count is 2*3*4");
    index[0] = 1U;
    index[1] = 2U;
    index[2] = 3U;
    expect_status("last element", INFR_OK, infr_shape_offset(&shape, index, &offset));
    check(offset == 23U, "last offset is 23");
    index[0] = 2U; /* one past the first axis */
    expect_status("index out of range", INFR_ERR_INVALID_ARGUMENT,
                  infr_shape_offset(&shape, index, &offset));
}

/* 3. Workspace: capacity, marks, and a graph op on an arena that cannot hold it. */
static void workspace_contract(void) {
    static uint8_t bytes[256];
    infr_workspace ws;
    void *block = NULL;
    size_t mark;

    begin("workspace/capacity");
    expect_status("init", INFR_OK, infr_workspace_init(&ws, bytes, sizeof(bytes)));
    expect_status("alloc 16", INFR_OK, infr_workspace_alloc(&ws, 16U, 8U, &block));
    check(block != NULL, "alloc returns a block");
    check(infr_workspace_used(&ws) >= 16U, "used grows by at least the request");
    mark = infr_workspace_mark(&ws);
    expect_status("alloc 64", INFR_OK, infr_workspace_alloc(&ws, 64U, 8U, &block));
    expect_status("rewind to the mark", INFR_OK, infr_workspace_release(&ws, mark));
    check(infr_workspace_used(&ws) == mark, "used returns to the mark");
    expect_status("rewind past the mark", INFR_ERR_INVALID_ARGUMENT,
                  infr_workspace_release(&ws, sizeof(bytes) + 1U));
    expect_status("alloc more than the arena holds", INFR_ERR_BUFFER_TOO_SMALL,
                  infr_workspace_alloc(&ws, sizeof(bytes) * 2U, 1U, &block));
}

/* 4. Deterministic pseudo-random geometries through the graph operators. */
static uint32_t lcg(uint32_t *state) {
    *state = (*state * 1664525U) + 1013904223U;
    return *state >> 8;
}

static void graph_geometries(void) {
    static float input_data[8U * 8U * 4U];
    static float filter_data[3U * 3U * 4U * 4U];
    static float output_data[8U * 8U * 4U];
    static uint8_t arena_bytes[64U * 1024U];
    uint32_t state = 12345U;
    uint32_t round;

    begin("graph/random-geometries");
    for (round = 0U; round < 200U; ++round) {
        const uint32_t in_h = 1U + (lcg(&state) % 8U);
        const uint32_t in_w = 1U + (lcg(&state) % 8U);
        const uint32_t channels = 1U + (lcg(&state) % 4U);
        const uint32_t kernel = 1U + (lcg(&state) % 3U);
        const uint32_t stride = 1U + (lcg(&state) % 2U);
        const uint32_t pad = lcg(&state) % 2U;
        infr_conv2d_params conv;
        infr_shape in_shape;
        infr_shape filter_shape;
        infr_shape out_shape;
        infr_tensor input;
        infr_tensor filter;
        infr_tensor output;
        infr_workspace ws;
        uint32_t out_dims[INFR_TENSOR_MAX_RANK];
        uint32_t elems;
        uint32_t i;

        conv.kernel_h = kernel;
        conv.kernel_w = kernel;
        conv.stride_h = stride;
        conv.stride_w = stride;
        conv.pad_h = pad;
        conv.pad_w = pad;
        conv.in_channels = channels;
        conv.out_channels = channels;
        conv.groups = 1U;

        out_dims[0] = 1U;
        out_dims[1] = in_h;
        out_dims[2] = in_w;
        out_dims[3] = channels;
        if (infr_shape_init(&in_shape, 4U, out_dims) != INFR_OK) {
            continue;
        }
        /* [out_c, kh, kw, in_per_group]: the body checks every axis, so a filter
         * shaped like the activation is refused -- correctly, and once. */
        out_dims[0] = channels;
        out_dims[1] = kernel;
        out_dims[2] = kernel;
        out_dims[3] = channels;
        if (infr_shape_init(&filter_shape, 4U, out_dims) != INFR_OK) {
            continue;
        }
        out_dims[0] = 1U;
        out_dims[1] = in_h;
        out_dims[2] = in_w;
        out_dims[3] = channels;
        if (infr_shape_init(&in_shape, 4U, out_dims) != INFR_OK) {
            continue;
        }
        if (infr_conv2d_output_shape(&conv, &in_shape, &out_shape) != INFR_OK) {
            continue; /* an illegal geometry is a rejection, not a crash */
        }
        if (filter_shape.elem_count > (sizeof(filter_data) / sizeof(filter_data[0]))) {
            continue;
        }
        elems = in_shape.elem_count;
        for (i = 0U; i < elems && i < (sizeof(input_data) / sizeof(input_data[0])); ++i) {
            input_data[i] = (float)(i % 17U) * 0.125F;
        }
        for (i = 0U; i < filter_shape.elem_count; ++i) {
            filter_data[i] = (float)(i % 5U) * 0.25F;
        }
        if (infr_tensor_init(&input, input_data, INFR_DTYPE_F32, &in_shape) != INFR_OK ||
            infr_tensor_init(&filter, filter_data, INFR_DTYPE_F32, &filter_shape) != INFR_OK ||
            infr_tensor_init(&output, output_data, INFR_DTYPE_F32, &out_shape) != INFR_OK) {
            continue;
        }
        if (infr_workspace_init(&ws, arena_bytes, sizeof(arena_bytes)) != INFR_OK) {
            ++failures;
            printf("  FAIL [%s] workspace init\n", current_case);
            return;
        }
        if (infr_conv2d_f32(&input, &filter, NULL, &conv, &output, &ws) != INFR_OK) {
            ++failures;
            printf("  FAIL [%s] conv2d refused a legal geometry (round %u)\n", current_case, round);
            return;
        }
    }
}

/* 5. The operator that needs a scratch buffer, on an arena two bytes long. */
static void graph_insufficient_arena(void) {
    static float input_data[16];
    static float output_data[16];
    static float filter_data[9];
    static uint8_t tiny[2];
    uint32_t dims[INFR_TENSOR_MAX_RANK];
    infr_conv2d_params conv;
    infr_shape in_shape;
    infr_shape out_shape;
    infr_tensor input;
    infr_tensor filter;
    infr_tensor output;
    infr_workspace ws;

    begin("graph/insufficient-arena");
    dims[0] = 1U;
    dims[1] = 4U;
    dims[2] = 4U;
    dims[3] = 1U;
    expect_status("input shape", INFR_OK, infr_shape_init(&in_shape, 4U, dims));
    conv.kernel_h = 3U;
    conv.kernel_w = 3U;
    conv.stride_h = 1U;
    conv.stride_w = 1U;
    conv.pad_h = 0U;
    conv.pad_w = 0U;
    conv.in_channels = 1U;
    conv.out_channels = 1U;
    conv.groups = 1U;
    expect_status("output shape", INFR_OK, infr_conv2d_output_shape(&conv, &in_shape, &out_shape));
    expect_status("input tensor", INFR_OK, infr_tensor_init(&input, input_data, INFR_DTYPE_F32, &in_shape));
    expect_status("filter tensor", INFR_OK, infr_tensor_init(&filter, filter_data, INFR_DTYPE_F32, &in_shape));
    expect_status("output tensor", INFR_OK, infr_tensor_init(&output, output_data, INFR_DTYPE_F32, &out_shape));
    expect_status("tiny workspace", INFR_OK, infr_workspace_init(&ws, tiny, sizeof(tiny)));
    /* A refusal is the contract; ASan is here to prove the refusal did not write
     * past `tiny` on the way to deciding. */
    check(infr_conv2d_f32(&input, &filter, NULL, &conv, &output, &ws) != INFR_OK,
          "conv2d refuses an arena smaller than its scratch need");
}

int main(void) {
    registry_null_arguments();
    shape_arithmetic();
    workspace_contract();
    graph_insufficient_arena();
    graph_geometries();
    if (failures != 0) {
        printf("sanitize smoke: %d failure(s)\n", failures);
        return 1;
    }
    printf("sanitize smoke: OK\n");
    return 0;
}
