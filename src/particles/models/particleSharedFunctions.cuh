/**
 *  @file particleSharedFunctions.cuh
 *  Contributors history:
 *  @author Marco Aurelio Ferrari (e.marcoferrari@utfpr.edu.br)
 *  @brief shared functions for particle simulation
 *  @version 0.1.0
 *  @date 01/09/2025
 */

#ifndef __PARTICLE_SHARED_FUNCTIONS_CUH
#define __PARTICLE_SHARED_FUNCTIONS_CUH

#include "./../../globalStructs.h"
#include "./../../globalFunctions.h"

#ifdef PARTICLE_MODEL

// Stencil distance
#if defined(STENCIL_2)
    #define P_DIST 1
#elif defined(STENCIL_4)
    #define P_DIST 2
#elif defined(STENCIL_COS)
    #define P_DIST 2
#else
    #define P_DIST 1
#endif



/**
 *  @brief Compute the value of the interpolation stencil function based on the distance x.
 *  @param x The distance from the point of interest.
 *  @return The value of the stencil function at distance x.
 */
__device__ __forceinline__  dfloat stencil(dfloat x) {
    dfloat absX = abs(x);
    #if defined STENCIL_2
        if (absX > 1.0_df) {
            return 0.0_df;
        }
        else {
            return (1 - x);
        }
    #elif defined STENCIL_4
        if (absX <= 1) {
            return (3.0_df - 2.0_df*absX + sqrt(1.0_df + 4.0_df * absX - 4.0_df * absX*absX))/8.0_df;
        }
        else if (absX > 1.0_df && absX <= 2.0_df) {
            return (5.0_df - 2.0_df*absX - sqrt(-7.0_df + 12.0_df*absX - 4.0_df*absX*absX))/8.0_df;
        }
        else {
            return 0.0_df;
        }
    #elif defined STENCIL_COS
        if (absX <= 2){
            return (cos(M_PI*absX*0.5_df)+1.0_df)/4.0_df;
        }   
        else{
            return 0.0_df;
        }
    #endif //STENCIL
    return 0.0_df;
}

#endif //PARTICLE_MODEL
#endif //__PARTICLE_MODEL_TRACER_CUH

