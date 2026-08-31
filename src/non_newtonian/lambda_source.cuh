/**
 *  @file lambda_source.cuh
 *  Contributors history:
 *  @author Marco Aurelio Ferrari (e.marcoferrari@utfpr.edu.br)
 *  @brief Lambda source term computation for thixotropic models
 *  @version 0.1.0
 *  @date 02/01/2026
 */

#ifndef __LAMBDA_SOURCE_CUH
#define __LAMBDA_SOURCE_CUH

#include "nnf_types.h"
#include "../var.h"

/* ============================================================================
 * THIXOTROPIC MODEL SOURCE TERM FUNCTIONS
 * ========================================================================== */

/**
 * @brief Compute lambda source term for Moore 1959 model
 * @param lambda_actual Actual lambda value in [0,1] range
 * @param gammaDot Shear rate
 * @param k1 Break rate constant
 * @param k2 Build rate constant
 * @param lambda_0 Equilibrium structure parameter
 * @return Source term (build - break)
 */
__device__ __forceinline__ 
dfloat calcLambdaSource_Moore1959(dfloat lambda_actual, dfloat gammaDot, 
                                   dfloat k1, dfloat k2, dfloat lambda_0) {
    // f_build = k2 * (lambda0 - lambda)
    // f_break = k1 * gammaDot * lambda
    dfloat buildRate = k2 * (lambda_0 - lambda_actual);
    dfloat breakRate = k1 * lambda_actual * gammaDot;
    return buildRate - breakRate;
}

/**
 * @brief Compute lambda source term for Worrall 1964 model
 * @param lambda_actual Actual lambda value in [0,1] range
 * @param k1 Break rate constant
 * @return Source term (0 - break)
 */
__device__ __forceinline__ 
dfloat calcLambdaSource_Worrall1964(dfloat lambda_actual, dfloat k1) {
    // f_build = k1
    // f_break = 0
    return k1;
}

/**
 * @brief Compute lambda source term for Houska 1980 model
 * @param lambda_actual Actual lambda value in [0,1] range
 * @param gammaDot Shear rate
 * @param k1 Break rate constant
 * @param k2 Build rate constant
 * @param m_exponent Shear rate exponent
 * @return Source term (build - break)
 */
__device__ __forceinline__ 
dfloat calcLambdaSource_Houska1980(dfloat lambda_actual, dfloat gammaDot,
                                    dfloat k1, dfloat k2, dfloat m_exponent) {
    // f_build = k2 * (1 - lambda)
    // f_break = k1 * lambda * gammaDot^m
    dfloat buildRate = k2 * (1.0_df - lambda_actual);
    dfloat breakRate = k1 * lambda_actual * POW_FUNCTION(gammaDot, m_exponent);
    return buildRate - breakRate;
}

/**
 * @brief Compute lambda source term for Toorman 1997 model
 * @param lambda_actual Actual lambda value in [0,1] range
 * @param gammaDot Shear rate
 * @param k1 Break rate constant
 * @param k2 Build rate constant
 * @param a_exponent Lambda exponent for break
 * @param b_exponent Lambda exponent for build
 * @return Source term (build - break)
 */
__device__ __forceinline__ 
dfloat calcLambdaSource_Toorman1997(dfloat lambda_actual, dfloat gammaDot,
                                     dfloat k1, dfloat k2, dfloat a_exponent, dfloat b_exponent) {
    // f_build = k2 * (1 - lambda)^b
    // f_break = k1 * lambda^a * gammaDot
    dfloat buildRate = k2 * POW_FUNCTION(1.0_df - lambda_actual, b_exponent);
    dfloat breakRate = k1 * POW_FUNCTION(lambda_actual, a_exponent) * gammaDot;
    return buildRate - breakRate;
}

/* ============================================================================
 * UNIFIED LAMBDA SOURCE DISPATCHER
 * ========================================================================== */

/**
 * @brief Compute lambda source term based on thixotropic model
 * @param fp Fluid properties struct containing thixotropic model info
 * @param lambdaVar Lambda variable with LAMBDA_ZERO offset
 * @param gammaDot Shear rate
 * @return Source term for lambda evolution
 */
__device__ __forceinline__
dfloat calcLambdaSource(const fluidProps& fp, dfloat lambdaVar, dfloat gammaDot) {
    // Extract actual lambda in [0,1] range
    dfloat lambda_actual = lambdaVar - LAMBDA_ZERO;
    lambda_actual = fmax(0.0_df, fmin(1.0_df, lambda_actual));
    
    // Dispatch to appropriate model
    switch(fp.u.thixo.model) {
        case THIXO_MOORE1959:
            return calcLambdaSource_Moore1959(
                lambda_actual, gammaDot,
                fp.u.thixo.u.moore1959.k1,
                fp.u.thixo.u.moore1959.k2,
                fp.u.thixo.u.moore1959.lambda_0
            );
            
        case THIXO_WORRALL1964:
            return calcLambdaSource_Worrall1964(
                lambda_actual,
                fp.u.thixo.u.worrall1964.k1
            );
            
        case THIXO_HOUSKA1980:
            return calcLambdaSource_Houska1980(
                lambda_actual, gammaDot,
                fp.u.thixo.u.houska1980.k1,
                fp.u.thixo.u.houska1980.k2,
                fp.u.thixo.u.houska1980.m_exponent
            );
            
        case THIXO_TOORMAN1997:
            return calcLambdaSource_Toorman1997(
                lambda_actual, gammaDot,
                fp.u.thixo.u.toorman1997.k1,
                fp.u.thixo.u.toorman1997.k2,
                fp.u.thixo.u.toorman1997.a_exponent,
                fp.u.thixo.u.toorman1997.b_exponent
            );
            
        default:
            // Default fallback to Moore 1959-like behavior
            return calcLambdaSource_Moore1959(lambda_actual, gammaDot, 1.0_df, 1.0_df, 0.5_df);
    }
}

#endif // __LAMBDA_SOURCE_CUH
