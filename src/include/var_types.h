/**
 *  @file var_types.h
 *  @brief Type definitions and precision settings
 *  @version 0.4.0
 *  @date 27/12/2025
 */

#ifndef __VAR_TYPES_H
#define __VAR_TYPES_H

#include <builtin_types.h>  // for devices variables

/* ========================= PRECISION DEFINITIONS ========================= */

#define SINGLE_PRECISION 
#ifdef SINGLE_PRECISION
    typedef float dfloat;      // single precision
    #define VTK_DFLOAT_TYPE "float"
    #define POW_FUNCTION powf 
#endif //SINGLE_PRECISION

#ifdef DOUBLE_PRECISION
    typedef double dfloat;      // double precision
    #define VTK_DFLOAT_TYPE "double"
    #define POW_FUNCTION pow
#endif //DOUBLE_PRECISION

// User-defined literal for dfloat type
__host__ __device__ 
constexpr dfloat operator "" _df(long double val) {
    return static_cast<dfloat>(val);
}

#endif //__VAR_TYPES_H