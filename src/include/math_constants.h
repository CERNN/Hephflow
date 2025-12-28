/**
 *  @file math_constants.h
 *  @brief Mathematical constants and LBM-specific derived constants
 *  @version 0.4.0
 *  @date 27/12/2025
 */

#ifndef __MATH_CONSTANTS_H
#define __MATH_CONSTANTS_H

#include "var_types.h"  // for dfloat type

/* ========================== MATHEMATICAL CONSTANTS ======================= */

// Mathematical constants (only the ones actually used)
#define SQRT_2  (1.41421356237309504880168872420969807856967187537)

/* ========================== LBM-DERIVED CONSTANTS ======================== */

// These depend on TAU which comes from case constants
// So they need to be included after TAU is defined
constexpr dfloat OMEGA = 1.0_df / TAU;        // (tau)^-1

#endif //__MATH_CONSTANTS_H