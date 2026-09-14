/**
 *  @file propertyInterpolation.cuh
 *  Contributors history:
 *  @author Marco Aurelio Ferrari (e.marcoferrari@utfpr.edu.br)
 *  @brief Compile-time configurable interpolation helpers for material properties
 *  @version 0.1.0
 *  @date 19/02/2026
 */

#ifndef __PROPERTY_INTERPOLATION_CUH
#define __PROPERTY_INTERPOLATION_CUH

#include "../var.h"

// Optional steppiness parameter for sigmoidal interpolation
#ifndef SIGMOID_STEEPNESS
	#define SIGMOID_STEEPNESS 6.0_df
#endif

// Map exp function to the active precision
#ifndef EXP_FUNCTION
	#ifdef SINGLE_PRECISION
		#define EXP_FUNCTION expf
	#else
		#define EXP_FUNCTION exp
	#endif
#endif

#ifndef POW_FUNCTION
    #ifdef SINGLE_PRECISION
        #define POW_FUNCTION powf
    #else
        #define POW_FUNCTION pow
    #endif
#endif

#ifndef ATAN_FUNCTION
    #ifdef SINGLE_PRECISION
        #define ATAN_FUNCTION atanf
    #else
        #define ATAN_FUNCTION atan
    #endif
#endif

#ifndef LOG_FUNCTION
    #ifdef SINGLE_PRECISION
        #define LOG_FUNCTION logf
    #else
        #define LOG_FUNCTION log
    #endif
#endif

#ifndef TANH_FUNCTION
    #ifdef SINGLE_PRECISION
        #define TANH_FUNCTION tanhf
    #else
        #define TANH_FUNCTION tanh
    #endif
#endif

#ifndef POWER_EXPONENT
    #define POWER_EXPONENT 2.0_df
#endif

#ifndef BLEND_ALPHA
    #define BLEND_ALPHA 0.5_df
#endif

#ifndef INTERP_PI
    #define INTERP_PI 3.14159265358979323846_df
#endif

#ifndef INTERP_EPS
    #define INTERP_EPS 1.0e-12_df
#endif


__host__ __device__ __forceinline__
dfloat interpolateProperty(dfloat low, dfloat high, dfloat param)
{
    dfloat t = param;

#ifdef INTERPOLATION_USE_SIGNED_PARAM
    t = 0.5_df * (param + 1.0_df);
#endif

// Clamp
t = fmin(1.0_df, fmax(0.0_df, t));

#if defined(INTERPOLATION_SIGMOIDAL)

    dfloat x = (t - 0.5_df) * SIGMOID_STEEPNESS * 2.0_df;
    t = 1.0_df / (1.0_df + EXP_FUNCTION(-x));
    t = fmin(1.0_df, fmax(0.0_df, t));
    return low + (high - low) * t;

#elif defined(INTERPOLATION_POLYNOMIAL)

    t = t * t * (3.0_df - 2.0_df * t);
    return low + (high - low) * t;

#elif defined(INTERPOLATION_QUINTIC)

    t = t*t*t*(10.0_df + t*(-15.0_df + 6.0_df*t));
    return low + (high - low) * t;

#elif defined(INTERPOLATION_HARMONIC)

    if (fabs(low) < INTERP_EPS || fabs(high) < INTERP_EPS) {
        return 0.0_df;
    }
    dfloat denom = (t / high) + ((1.0_df - t) / low);
    return (denom != 0.0_df) ? (1.0_df / denom) : 0.0_df;

#elif defined(INTERPOLATION_LOG)

    if (low <= 0.0_df || high <= 0.0_df) {
        return low + (high - low) * t; // fallback to linear to avoid NaNs
    }
    return EXP_FUNCTION(
        (1.0_df - t) * LOG_FUNCTION(low) +
        t * LOG_FUNCTION(high)
    );

#elif defined(INTERPOLATION_POWER)

    t = POW_FUNCTION(t, POWER_EXPONENT);
    return low + (high - low) * t;

#elif defined(INTERPOLATION_TANH)

    #ifdef INTERPOLATION_USE_SIGNED_PARAM
        dfloat phi = param; // already signed
    #else
        dfloat phi = 2.0_df * t - 1.0_df;
    #endif

    dfloat shaped = 0.5_df * (1.0_df + TANH_FUNCTION(SIGMOID_STEEPNESS * phi));
    return low + (high - low) * shaped;

#elif defined(INTERPOLATION_ARCTAN)

    dfloat x = (t - 0.5_df) * SIGMOID_STEEPNESS;
    t = ATAN_FUNCTION(x) / INTERP_PI + 0.5_df;
    return low + (high - low) * t;

#elif defined(INTERPOLATION_SHARP)

    t = (t > 0.5_df) ? 1.0_df : 0.0_df;
    return low + (high - low) * t;

#elif defined(INTERPOLATION_BLEND)

    dfloat linear = low + (high - low) * t;
    dfloat harmonic = 0.0_df;
    if (fabs(low) >= INTERP_EPS && fabs(high) >= INTERP_EPS) {
        dfloat denom = (t / high) + ((1.0_df - t) / low);
        harmonic = (denom != 0.0_df) ? (1.0_df / denom) : 0.0_df;
    }
    return BLEND_ALPHA * harmonic + (1.0_df - BLEND_ALPHA) * linear;

#else
    // Default: linear
    return low + (high - low) * t;
#endif

 
}

#endif // __PROPERTY_INTERPOLATION_CUH
