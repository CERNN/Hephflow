/**
 *  @file nnf.h
 *  Contributors history:
 *  @author Marco Aurelio Ferrari (e.marcoferrari@utfpr.edu.br)
 *  @brief Information about non-Newtonian fluids
 *  @version 0.1.0
 *  @date 01/09/2025
 */

#ifndef __NNF_H
#define __NNF_H

#include <math.h>
#include <cmath>
#include "../var.h"
#include "nnf_types.h"

// Forward declarations for calcOmega* functions
__host__ __device__ dfloat __forceinline__ calcOmegaPowerLaw(dfloat k_consistency, dfloat n_index, dfloat omegaOld, dfloat const auxStressMag);
__host__ __device__ dfloat __forceinline__ calcOmegaBingham(dfloat omega_p, dfloat s_y, dfloat auxStressMag);
__host__ __device__ dfloat __forceinline__ calcOmegaHerschelBulkley(dfloat k_consistency, dfloat n_index, dfloat s_y, dfloat omegaOld, dfloat const auxStressMag);
__host__ __device__ dfloat __forceinline__ calcOmegaBiViscosity(dfloat omega_y, dfloat omega_p, dfloat s_y, dfloat visc_ratio, dfloat auxStressMag);
__host__ __device__ dfloat __forceinline__ calcOmegaKeeTurcotee(dfloat s_y, dfloat t1, dfloat eta_0, dfloat omegaOld, dfloat auxStressMag, int step);
__host__ __device__ dfloat __forceinline__ calcOmega_thixo(const fluidProps& fp, dfloat lambda, dfloat gammaDot, dfloat auxStressMag, dfloat rhoVar);
__host__ __device__ dfloat __forceinline__ calcOmega(const fluidProps& fp, dfloat omegaOld, dfloat auxStressMag, dfloat lambdaVar, dfloat gammaDot, dfloat rhoVar, int step){

    dfloat newOmegaVar;
    
    switch (fp.type) {
    case FLUID_POWERLAW:
        newOmegaVar = calcOmegaPowerLaw(fp.u.powerlaw.k_consistency, fp.u.powerlaw.n_index, omegaOld, auxStressMag);
        break;
    case FLUID_BINGHAM: 
        newOmegaVar = calcOmegaBingham(fp.u.bingham.omega_p, fp.u.bingham.s_y, auxStressMag);
        break;
    case FLUID_HERSCHEL_BULKLEY:
        newOmegaVar = calcOmegaHerschelBulkley(fp.u.hb.k_consistency, fp.u.hb.n_index, fp.u.hb.s_y, omegaOld, auxStressMag);
        break;
    case FLUID_BI_VISCOSITY: 
        newOmegaVar = calcOmegaBiViscosity(fp.u.bi.omega_y, fp.u.bi.omega_p, fp.u.bi.s_y, fp.u.bi.visc_ratio, auxStressMag);
        break;
    case FLUID_KEE_TURCOTEE: 
        newOmegaVar = calcOmegaKeeTurcotee(fp.u.kee.s_y, fp.u.kee.t1, fp.u.kee.eta_0, omegaOld, auxStressMag, step);
        break;
    case FLUID_THIXO: 
        newOmegaVar = calcOmega_thixo(fp, lambdaVar, gammaDot, auxStressMag, rhoVar);
        break;
    default: return omegaOld;
    }
    return newOmegaVar;
}

__host__ __device__ 
dfloat __forceinline__ calcOmegaPowerLaw(dfloat k_consistency, dfloat n_index, dfloat omegaOld, dfloat const auxStressMag){

    dfloat omega = omegaOld; //initial guess

    dfloat fx, fx_dx;
    const dfloat cs2 = 1.0_df / 3.0_df;
    const dfloat a = k_consistency * POW_FUNCTION(auxStressMag / (RHO_0 * cs2), n_index);
    const dfloat b = 0.5_df * auxStressMag;
    const dfloat c = -auxStressMag;

    if(auxStressMag < 1e-6_df)
        return 0.0_df;

    for (int i = 0; i < 7; i++){
        fx = a * POW_FUNCTION(omega, n_index) + b * omega + c;
        fx_dx = a * n_index * POW_FUNCTION(omega, n_index - 1.0_df) + b;

        if (fabs(fx / fx_dx) < 1e-6_df){
            break;
        }
            
        omega = omega - fx / fx_dx;
    }
    return omega;
}

__host__ __device__ 
dfloat __forceinline__ calcOmegaBingham(dfloat omega_p, dfloat s_y, dfloat auxStressMag){
    return omega_p * myMax(0.0_df, (1.0_df - s_y / auxStressMag));
}

__host__ __device__ 
dfloat __forceinline__ calcOmegaHerschelBulkley(dfloat k_consistency, dfloat n_index, dfloat s_y, dfloat omegaOld, dfloat const auxStressMag){
    dfloat omega = omegaOld;
    if(auxStressMag < 1e-6_df) return 0.0_df;
    return omega;
}


// NOT TESTED/VALIDATED
__host__ __device__ 
dfloat __forceinline__ calcOmegaBiViscosity(dfloat omega_y, dfloat omega_p, dfloat s_y, dfloat visc_ratio, dfloat auxStressMag){
    return myMax(omega_y, omega_p * (1.0_df - s_y * (1.0_df - visc_ratio) / auxStressMag));
}

// NOT TESTED/VALIDATED https://arxiv.org/abs/2401.02942 has analytical solution
__host__ __device__ 
dfloat __forceinline__ calcOmegaKeeTurcotee(dfloat s_y, dfloat t1, dfloat eta_0, dfloat omegaOld, dfloat auxStressMag, int step){
    const dfloat cs2 = 1.0_df / 3.0_df;
    dfloat omega = omegaOld;
    const dfloat A = auxStressMag / 2.0_df;
    const dfloat B = auxStressMag / (RHO_0 * cs2);
    const dfloat C = B * eta_0;
    const dfloat D = -t1 * B;
    const dfloat E = s_y - auxStressMag;

    if(auxStressMag < 1e-6_df)
        return 0.0_df;
    
    dfloat fx, fx_dx;
    for (int i = 0; i < 7; i++){
        fx = omega * (A + C * expf(D * omega)) + E;
        fx_dx = A + C * expf(D * omega) * (1.0_df + D * omega);

        if (fabs(fx / fx_dx) < 1e-6_df){
            break;
        }
            
        omega = omega - fx / fx_dx;
    }
    return omega;  
}

__host__ __device__ 
dfloat __forceinline__ calcYieldStress_thixo(const fluidProps& fp, dfloat lambda)
{
    switch(fp.u.thixo.model)
    {
        case THIXO_MOORE1959:
            return 0.0_df;
        case THIXO_WORRALL1964:
            return fp.u.thixo.u.worrall1964.s_y_0;
        case THIXO_HOUSKA1980:
            return lambda * (fp.u.thixo.u.houska1980.s_y_0 - fp.u.thixo.u.houska1980.s_y_inf) + fp.u.thixo.u.houska1980.s_y_inf;
        case THIXO_TOORMAN1997:
            return lambda * fp.u.thixo.u.toorman1997.s_y_0;
        default:
            return 0.0_df;
    }
}


__host__ __device__ 
dfloat __forceinline__ calcVisco_thixo(const fluidProps& fp, dfloat lambda, dfloat gammaDot)
{
    // Clamp lambda to [0, 1]
    lambda = fmax(0.0_df, fmin(1.0_df, lambda - LAMBDA_ZERO));
    
    switch(fp.u.thixo.model)
    {
        case THIXO_MOORE1959:
            return lambda * fp.u.thixo.u.moore1959.eta_0;
        case THIXO_WORRALL1964:
            return lambda * fp.u.thixo.u.worrall1964.eta_0;
        case THIXO_HOUSKA1980:
            return lambda * fp.u.thixo.u.houska1980.k_consistency * POW_FUNCTION(fmax(gammaDot, 1e-12_df), fp.u.thixo.u.houska1980.n_index - 1.0_df);
        case THIXO_TOORMAN1997:
            return lambda * fp.u.thixo.u.toorman1997.eta_0;
        default:
            return 0.0_df;
    }
}

__host__ __device__ 
dfloat __forceinline__ calcOmega_thixo(const fluidProps& fp, dfloat lambda, dfloat gammaDot, dfloat auxStressMag, dfloat rhoVar){
    // Extract actual lambda value (remove LAMBDA_ZERO offset)
    dfloat lambda_actual = lambda - LAMBDA_ZERO;
    lambda_actual = fmax(0.0_df, fmin(1.0_df, lambda_actual));
    
    dfloat visc = calcVisco_thixo(fp, lambda_actual, gammaDot);
    dfloat yieldStress = calcYieldStress_thixo(fp, lambda_actual);

    // Convert viscosity to omega: tau = visc / (rho * cs2) + 0.5; omega = 1/tau
    const dfloat cs2 = 1.0_df / 3.0_df;
    dfloat tau_p = visc / (rhoVar * cs2) + 0.5_df;
    dfloat omega_p = 1.0_df / tau_p;

    // Use Bingham approach with yield stress for final omega
    dfloat omega = calcOmegaBingham(omega_p, yieldStress, auxStressMag);
    return omega;
}








#endif // __NNF_H

