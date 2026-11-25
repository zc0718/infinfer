// Conan::ImportStart
#include <infinfer/kernel/kernel.h>
// Conan::ImportEnd

#include <math.h>

/*
 * Scalar fused activations: the reference definition of correctness (INV-2).
 * Every other tier must be proven equivalent to these bodies by the parity
 * harness; none of them may redefine what an activation means.
 *
 * 融合激活的 scalar 参考实现：正确性的唯一定义（INV-2）。其它档位只能由 parity
 * 骨架证明等价，不得重定义语义。
 */

static void act_none_f32(const float *in, float *out, uint32_t n) {
    uint32_t i;
    for (i = 0; i < n; ++i) {
        out[i] = in[i];
    }
}

static void act_relu_f32(const float *in, float *out, uint32_t n) {
    uint32_t i;
    for (i = 0; i < n; ++i) {
        out[i] = in[i] > 0.0F ? in[i] : 0.0F;
    }
}

static void act_relu_n1_to_1_f32(const float *in, float *out, uint32_t n) {
    uint32_t i;
    for (i = 0; i < n; ++i) {
        float v = in[i];
        if (v < -1.0F) {
            v = -1.0F;
        } else if (v > 1.0F) {
            v = 1.0F;
        }
        out[i] = v;
    }
}

static void act_relu6_f32(const float *in, float *out, uint32_t n) {
    uint32_t i;
    for (i = 0; i < n; ++i) {
        float v = in[i];
        if (v < 0.0F) {
            v = 0.0F;
        } else if (v > 6.0F) {
            v = 6.0F;
        }
        out[i] = v;
    }
}

static void act_tanh_f32(const float *in, float *out, uint32_t n) {
    uint32_t i;
    for (i = 0; i < n; ++i) {
        out[i] = tanhf(in[i]);
    }
}

static void act_sign_bit_f32(const float *in, float *out, uint32_t n) {
    uint32_t i;
    for (i = 0; i < n; ++i) {
        out[i] = in[i] < 0.0F ? -1.0F : 1.0F;
    }
}

static void act_hard_swish_f32(const float *in, float *out, uint32_t n) {
    uint32_t i;
    for (i = 0; i < n; ++i) {
        float shifted = in[i] + 3.0F;
        if (shifted < 0.0F) {
            shifted = 0.0F;
        } else if (shifted > 6.0F) {
            shifted = 6.0F;
        }
        out[i] = in[i] * shifted / 6.0F;
    }
}

/* Row order is contract: index == infr_activation value (see kernel.h). */
static const infr_act_f32_fn ACT_TABLE[INFR_ACT_COUNT] = {
    act_none_f32,         /* INFR_ACT_NONE */
    act_relu_f32,         /* INFR_ACT_RELU */
    act_relu_n1_to_1_f32, /* INFR_ACT_RELU_N1_TO_1 */
    act_relu6_f32,        /* INFR_ACT_RELU6 */
    act_tanh_f32,         /* INFR_ACT_TANH */
    act_sign_bit_f32,     /* INFR_ACT_SIGN_BIT */
    act_hard_swish_f32,   /* INFR_ACT_HARD_SWISH */
};

#if defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 201112L)
_Static_assert(sizeof(ACT_TABLE) / sizeof(ACT_TABLE[0]) == (size_t)INFR_ACT_COUNT,
               "an activation code without a row must fail the build, not the dispatch");
#endif

infr_act_f32_fn infr_activation_f32(infr_activation act) {
    unsigned index = (unsigned)act;
    if (index >= (unsigned)INFR_ACT_COUNT) {
        return NULL; /* Out-of-range codes have no body; the caller must handle NULL. */
    }
    return ACT_TABLE[index];
}
