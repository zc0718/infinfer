// P0.5 case 1 (deliberately red): the fused-activation dispatch contract.
//
// The reference defect this pins: an op-code table whose rows had drifted away
// from the codes they claimed to name (a fused activation dispatched to the wrong
// body). Every assertion below is a contract statement from kernel.h, not an
// implementation detail.
//
// Expected state at 0.1.0: RED -- no float32 activation body exists yet, so
// `infr_activation_f32` returns NULL and the numeric checks fail loudly.
//
// One TEST per file: the recipe turns this file into a coverage executable whose
// gtest result is one case, so the count in ctest is the count of defect cases.
#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>

#include <infinfer/infinfer.h>

TEST(ContractActivation, DispatchMatchesThePinnedContract) {
    // 1) Raw code -> enum: every declared code maps to itself, nothing beyond it.
    for (uint32_t code = 0; code < static_cast<uint32_t>(INFR_ACT_COUNT); ++code) {
        infr_activation act = INFR_ACT_NONE;
        EXPECT_EQ(INFR_OK, infr_activation_from_code(code, &act));
        EXPECT_EQ(static_cast<infr_activation>(code), act);
    }
    infr_activation act = INFR_ACT_NONE;
    EXPECT_EQ(INFR_ERR_UNSUPPORTED, infr_activation_from_code(static_cast<uint32_t>(INFR_ACT_COUNT), &act));
    EXPECT_EQ(INFR_ERR_INVALID_ARGUMENT, infr_activation_from_code(0, nullptr));

    // 2) Every declared code has a body (INV-4: no silent hole in the dispatch).
    for (uint32_t code = 0; code < static_cast<uint32_t>(INFR_ACT_COUNT); ++code) {
        EXPECT_NE(nullptr, infr_activation_f32(static_cast<infr_activation>(code)))
            << "activation code " << code << " has no float32 body";
    }

    // 3) The bodies compute what the enum promises (per-element, n = 1).
    struct Case {
        infr_activation act;
        float in;
        float want;
    };
    const Case cases[] = {
        {INFR_ACT_RELU, -2.0F, 0.0F},        {INFR_ACT_RELU, 3.0F, 3.0F},         {INFR_ACT_RELU_N1_TO_1, -2.0F, -1.0F},
        {INFR_ACT_RELU_N1_TO_1, 0.5F, 0.5F}, {INFR_ACT_RELU_N1_TO_1, 9.0F, 1.0F}, {INFR_ACT_RELU6, -1.0F, 0.0F},
        {INFR_ACT_RELU6, 10.0F, 6.0F},       {INFR_ACT_SIGN_BIT, -2.0F, -1.0F},   {INFR_ACT_SIGN_BIT, 3.0F, 1.0F},
        {INFR_ACT_HARD_SWISH, 0.0F, 0.0F},   {INFR_ACT_HARD_SWISH, 3.0F, 3.0F},   {INFR_ACT_HARD_SWISH, -4.0F, 0.0F},
    };
    for (const Case& item : cases) {
        const infr_act_f32_fn fn = infr_activation_f32(item.act);
        if (fn == nullptr) {
            continue; /* Already reported by check 2; do not call through NULL. */
        }
        const float src[1] = {item.in};
        float out = 0.0F;
        fn(src, &out, 1);
        EXPECT_NEAR(item.want, out, 1e-6F) << "activation " << static_cast<int>(item.act) << " at " << item.in;
    }

    // 4) Tanh is pinned against the library, not against a hand-computed constant.
    const infr_act_f32_fn tanh_fn = infr_activation_f32(INFR_ACT_TANH);
    if (tanh_fn != nullptr) {
        for (const float in : {-1.5F, -0.25F, 0.0F, 0.25F, 1.5F}) {
            const float src[1] = {in};
            float out = 0.0F;
            tanh_fn(src, &out, 1);
            EXPECT_NEAR(std::tanh(in), out, 1e-6F) << "tanh at " << in;
        }
    }
}
