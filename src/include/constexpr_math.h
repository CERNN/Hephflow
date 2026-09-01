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

constexpr dfloat LN2 = 0.693147180559945309417232121458176568_df;

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
    if (n > 12) return sum;
    dfloat term = y;
    for (int i = 0; i < n - 1; ++i) term *= y * y;
    return constexprLnHelper(y, n + 1, sum + term / static_cast<dfloat>(2 * n - 1));
}

// Fixed constexprLn with range reduction into [0.75, 1.5]
constexpr dfloat constexprLn(dfloat x) {
    if (x <= 0.0_df) return std::numeric_limits<dfloat>::quiet_NaN();
    
    int k = 0;
    dfloat xr = x;
    while (xr > 1.5_df) {
        xr *= 0.5_df;
        ++k;
    }
    while (xr < 0.75_df) {
        xr *= 2.0_df;
        --k;
    }
    dfloat y = (xr - 1.0_df) / (xr + 1.0_df);
    return 2.0_df * constexprLnHelper(y, 1, 0.0_df) + static_cast<dfloat>(k) * LN2;
}

constexpr dfloat constexprCos(dfloat x)
{
    dfloat term = 1.0_df;
    dfloat sum  = 1.0_df;

    for (int n = 1; n < 12; ++n)
    {
        term *= -x * x / ((2*n - 1) * (2*n));
        sum += term;
    }

    return sum;
}


constexpr dfloat constexprExpHelper(dfloat x, int n, dfloat term, dfloat sum) {
    if (n > 25) return sum;
    dfloat next_term = term * x / static_cast<dfloat>(n);
    return constexprExpHelper(x, n + 1, next_term, sum + next_term);
}

constexpr dfloat constexprExp(dfloat x) {
    int k = 0;
    dfloat xr = x;
    while (xr > 1.0_df || xr < -1.0_df) {
        xr *= 0.5_df;
        ++k;
    }
    dfloat result = constexprExpHelper(xr, 1, 1.0_df, 1.0_df);
    for (int i = 0; i < k; ++i) result *= result;
    return result;
}

constexpr dfloat constexprIntPow(dfloat base, long long n) {
    if (n == 0) return 1.0_df;
    if (n < 0) return 1.0_df / constexprIntPow(base, -n);
    dfloat half = constexprIntPow(base, n / 2);
    return (n % 2 == 0) ? (half * half) : (half * half * base);
}

// Updated constexprPow with fast path for square root (exponent == 0.5)
constexpr dfloat constexprPow(dfloat base, dfloat exponent) {
    if (exponent == 0.0_df) return 1.0_df;
    if (base == 1.0_df) return 1.0_df;

    long long n = static_cast<long long>(exponent);
    if (static_cast<dfloat>(n) == exponent) {
        return constexprIntPow(base, n);
    }

    if (base < 0.0_df) return std::numeric_limits<dfloat>::quiet_NaN();
    if (base == 0.0_df) return (exponent > 0.0_df) ? 0.0_df : std::numeric_limits<dfloat>::infinity();

    return constexprExp(exponent * constexprLn(base));
}


constexpr dfloat constexprAbs(dfloat val) {
    return (val < 0.0_df) ? -val : val;
}

// Compile-time assertion helper with scalable epsilon tolerance
constexpr dfloat TEST_TOL = 100.0_df * std::numeric_limits<dfloat>::epsilon();
constexpr bool isClose(dfloat a, dfloat b, dfloat tol = TEST_TOL) {
    return constexprAbs(a - b) <= tol * (1.0_df + constexprAbs(b));
}

#endif //__CONSTEXPR_MATH_H