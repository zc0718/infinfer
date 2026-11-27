// Conan::ImportStart
#include <string.h>

#include "kat_chain_k1.h"
// Conan::ImportEnd

/*
 * The chain. Every numeric step goes through the public surface a consumer has:
 * `infr_kernel_register_scalar()` + `infr_kernel_find()` -- so the KAT exercises
 * the registry on the target too, not just the bodies.
 *
 * Shape lives here, never in the kernel signature (INV-3): this file is the
 * graph layer in miniature -- it loops the rows and hands the kernel one row at a
 * time.
 *
 * 链本体。每个数值步骤都走消费方可见的公开面（注册 + 查找），因此 KAT 在目标上
 * 连注册表一起验证，而不只是算子体。
 *
 * shape 只存在于这里、绝不进内核签名（INV-3）：这个文件就是一个微缩图层——它
 * 逐行循环，一次交给内核一行。
 */

#define KAT_K1_SEED 0x5EEDu

static float lcg_value(unsigned seed) {
    const uint32_t state = (uint32_t)seed * 1664525U + 1013904223U;
    const int32_t centred = (int32_t)((state >> 8) % 2001U) - 1000;
    return (float)centred / 1000.0F;
}

static infr_status dot_row(const infr_kernel_desc *fc, const float *x, const float *row,
                           unsigned length, float *out) {
    const void *inputs[2];
    void *outputs[1];

    inputs[0] = x;
    inputs[1] = row;
    outputs[0] = out;
    return fc->fn(inputs, outputs, length, 0);
}

infr_status kat_k1_run(kat_k1_result *out) {
    const infr_kernel_desc *fc;
    const infr_kernel_desc *relu;
    float w1[KAT_K1_HIDDEN][KAT_K1_IN];
    float w2[KAT_K1_OUT][KAT_K1_HIDDEN];
    const void *relu_in[1];
    void *relu_out[1];
    infr_status status;
    unsigned row;
    unsigned i;

    if (out == 0) {
        return INFR_ERR_INVALID_ARGUMENT;
    }
    status = infr_kernel_register_scalar();
    if (status != INFR_OK) {
        return status;
    }
    fc = infr_kernel_find("fc", INFR_DTYPE_F32, INFR_QUANT_NONE, INFR_TIER_SCALAR);
    relu = infr_kernel_find("relu", INFR_DTYPE_F32, INFR_QUANT_NONE, INFR_TIER_SCALAR);
    if (fc == 0 || relu == 0) {
        return INFR_ERR_NOT_IMPLEMENTED;
    }

    for (i = 0; i < KAT_K1_IN; ++i) {
        out->x[i] = lcg_value(KAT_K1_SEED + i);
    }
    for (row = 0; row < KAT_K1_HIDDEN; ++row) {
        for (i = 0; i < KAT_K1_IN; ++i) {
            w1[row][i] = lcg_value(KAT_K1_SEED + 100u + row * KAT_K1_IN + i);
        }
    }
    for (row = 0; row < KAT_K1_OUT; ++row) {
        for (i = 0; i < KAT_K1_HIDDEN; ++i) {
            w2[row][i] = lcg_value(KAT_K1_SEED + 2000u + row * KAT_K1_HIDDEN + i);
        }
    }

    /* Layer 1: one fc call per output row, then one relu over the whole vector. */
    for (row = 0; row < KAT_K1_HIDDEN; ++row) {
        status = dot_row(fc, out->x, w1[row], KAT_K1_IN, &out->y1[row]);
        if (status != INFR_OK) {
            return status;
        }
    }
    relu_in[0] = out->y1;
    relu_out[0] = out->y1;
    status = relu->fn(relu_in, relu_out, KAT_K1_HIDDEN, 0);
    if (status != INFR_OK) {
        return status;
    }

    /* Layer 2: same shape of loop, no activation on the output layer. */
    for (row = 0; row < KAT_K1_OUT; ++row) {
        status = dot_row(fc, out->y1, w2[row], KAT_K1_HIDDEN, &out->y2[row]);
        if (status != INFR_OK) {
            return status;
        }
    }
    return INFR_OK;
}

unsigned kat_k1_expect_len(void) { return (unsigned)sizeof(float) * KAT_K1_OUT; }

void kat_k1_expect(const kat_k1_result *result, unsigned char *dst) {
    memcpy(dst, result->y2, kat_k1_expect_len());
}
