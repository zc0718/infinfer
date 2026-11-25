// Parity harness (PLAN P1.1): compare a candidate tier against the scalar
// reference with two independent thresholds -- relative/absolute error AND ULP
// distance -- so neither a vanishing value nor a large value can hide a drift.
//
// 差分骨架（PLAN P1.1）：用两套独立阈值（相对/绝对误差 + ULP 距离）比较候选档与
// scalar 参考，避免小值或大值单独把漂移藏起来。
#pragma once

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace infr_test {

constexpr double kFloatRelTol = 1e-5;
constexpr double kFloatAbsTol = 1e-6;
constexpr std::uint32_t kUlpTolerance = 4;

inline std::uint32_t float_bits(float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof bits);
    return bits;
}

// Monotonic integer ordering of IEEE-754 binary32: adjacent floats differ by one.
inline std::uint32_t order_key(float value) {
    const std::uint32_t bits = float_bits(value);
    return (bits & 0x80000000U) != 0U ? ~bits + 1U : bits | 0x80000000U;
}

inline std::uint32_t ulp_distance(float lhs, float rhs) {
    const std::uint32_t a = order_key(lhs);
    const std::uint32_t b = order_key(rhs);
    return a >= b ? a - b : b - a;
}

// Passes when a sample is within the relative/absolute window OR within the ULP
// window. Failure messages carry hex bit patterns, never formatted decimals.
inline void expect_float_buffers_near(const float* want, const float* got, std::size_t count, const char* what,
                                      double rel_tol = kFloatRelTol, double abs_tol = kFloatAbsTol,
                                      std::uint32_t ulp_tol = kUlpTolerance) {
    ASSERT_NE(nullptr, want);
    ASSERT_NE(nullptr, got);
    for (std::size_t i = 0; i < count; ++i) {
        const double diff = std::fabs(static_cast<double>(got[i]) - static_cast<double>(want[i]));
        const bool near_enough = diff <= abs_tol || diff <= rel_tol * std::fabs(static_cast<double>(want[i]));
        const bool ulp_ok = ulp_distance(want[i], got[i]) <= ulp_tol;
        EXPECT_TRUE(near_enough || ulp_ok) << "[" << what << "] element " << i << " want bits 0x" << std::hex
                                           << float_bits(want[i]) << " got bits 0x" << float_bits(got[i]) << std::dec;
    }
}

// Integer tensors are compared exactly: a MAC order is fixed, so any difference
// is a difference in the implementation, not in rounding.
inline void expect_i8_buffers_equal(const std::int8_t* want, const std::int8_t* got, std::size_t count,
                                    const char* what) {
    ASSERT_NE(nullptr, want);
    ASSERT_NE(nullptr, got);
    for (std::size_t i = 0; i < count; ++i) {
        EXPECT_EQ(static_cast<int>(want[i]), static_cast<int>(got[i])) << "[" << what << "] element " << i;
    }
}

} // namespace infr_test
