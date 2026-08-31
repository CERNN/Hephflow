/**
 *  @file constexpr_math.h
 *  @brief Compile-time mathematical functions and utilities
 *  @version 0.4.0
 *  @date 27/12/2025
 */

#ifndef __CONSTEXPR_MATH_H
#define __CONSTEXPR_MATH_H

#include <limits>
#include "var_types.h"  // for dfloat type

/* ====================== COMPILE-TIME MATH FUNCTIONS ====================== */

// Compile-time square root implementation
constexpr dfloat constexprSqrt(dfloat x, dfloat curr, dfloat prev) {
    return (curr == prev) ? curr : constexprSqrt(x, 0.5_df * (curr + x / curr), curr);
}

constexpr dfloat sqrtt(dfloat x) {
    return (x >= 0 && x < std::numeric_limits<dfloat>::infinity())
        ? constexprSqrt(x, x, 0)
        : std::numeric_limits<dfloat>::quiet_NaN();
}

// Compile-time inverse square root implementation
constexpr dfloat invSqrtNewton(dfloat x, dfloat curr, dfloat prev) {
    return (curr == prev) ? curr : invSqrtNewton(x, curr * (1.5_df - 0.5_df * x * curr * curr), curr);
}

constexpr dfloat invSqrtt(dfloat x) {
    return (x > 0 && x < std::numeric_limits<dfloat>::infinity())
        ? invSqrtNewton(x, 1.0_df / x, 0)
        : std::numeric_limits<dfloat>::quiet_NaN();
}

// Compile-time natural logarithm implementation
constexpr dfloat constexprLnHelper(dfloat y, int n, dfloat sum) {
    if (n > 10) {
        return sum;
    }
    dfloat term = y;
    for (int i = 0; i < n - 1; ++i) {
        term *= y * y;
    }
    return constexprLnHelper(y, n + 1, sum + term / (2.0_df * n - 1.0_df));
}

constexpr dfloat constexprLn(dfloat x) {
    if (x <= 0.0_df) {
        return std::numeric_limits<dfloat>::quiet_NaN();
    }
    dfloat y = (x - 1.0_df) / (x + 1.0_df);
    return 2.0_df * constexprLnHelper(y, 1, 0.0_df);
}

#endif //__CONSTEXPR_MATH_H