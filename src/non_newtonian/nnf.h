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

// Provide PHASE1/PHASE2 aliases when a case only defines CASE_PHASE_PROPS
#ifdef CASE_PHASE_PROPS
#ifndef CASE_PHASE_PROPS_PHASE1
#define CASE_PHASE_PROPS_PHASE1 CASE_PHASE_PROPS
#endif
#ifndef CASE_PHASE_PROPS_PHASE2
#define CASE_PHASE_PROPS_PHASE2 CASE_PHASE_PROPS
#endif
#endif

// Forward declarations for calcOmega* functions
__host__ __device__ dfloat __forceinline__ calcOmegaPowerLaw(dfloat k_consistency, dfloat n_index, dfloat omegaOld, dfloat const auxStressMag);
__host__ __device__ dfloat __forceinline__ calcOmegaBingham(dfloat omega_p, dfloat s_y, dfloat auxStressMag);
__host__ __device__ dfloat __forceinline__ calcOmegaHerschelBulkley(dfloat k_consistency, dfloat n_index, dfloat s_y, dfloat omegaOld, dfloat const auxStressMag);
__host__ __device__ dfloat __forceinline__ calcOmegaBiViscosity(dfloat omega_y, dfloat omega_p, dfloat s_y, dfloat visc_ratio, dfloat auxStressMag);
__host__ __device__ dfloat __forceinline__ calcOmegaKeeTurcotee(dfloat s_y, dfloat t1, dfloat eta_0, dfloat omegaOld, dfloat auxStressMag, int step);
#ifdef LAMBDA_DIST
__host__ __device__ dfloat __forceinline__ calcOmega_thixo(const fluidProps& fp, dfloat lambda, dfloat gammaDot, dfloat auxStressMag, dfloat rhoVar);
#endif
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
    #ifdef LAMBDA_DIST
    case FLUID_THIXO: 
        newOmegaVar = calcOmega_thixo(fp, lambdaVar, gammaDot, auxStressMag, rhoVar);
        break;
    #endif
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
    if (s_y <= 0.0_df)
        return omega_p;
    if (auxStressMag <= 1.0e-12_df)
        return 0.0_df;
    return omega_p * myMax(0.0_df, (1.0_df - s_y / auxStressMag));
}

__host__ __device__ 
dfloat __forceinline__ calcOmegaHerschelBulkley(
    dfloat k_consistency, dfloat n_index, dfloat s_y, 
    dfloat omegaOld, dfloat const auxStressMag)
{
    const dfloat cs2   = 1.0_df / 3.0_df;
    const dfloat rho0  = RHO_0;

    if(auxStressMag * (1.0_df - omegaOld * 0.5_df)  < s_y)
        return 0.0_df;

    const dfloat rhoCs2_n = POW_FUNCTION(rho0 * cs2, n_index);
    const dfloat Pi_n     = POW_FUNCTION(auxStressMag, n_index);

    const dfloat A = (2.0_df * k_consistency / rhoCs2_n) * Pi_n;
    const dfloat B = auxStressMag;
    const dfloat C = 2.0_df * (auxStressMag - s_y);

    // Guard: C must be > 0 after yield check, but clamp for safety
    if (C <= 0.0_df) return 0.0_df;

    // Initial guess: use Newtonian limit (n=1) as warm start
    dfloat omega = (omegaOld > 1e-8_df && omegaOld < 2.0_df) 
                    ? omegaOld 
                    : C / (A + B);

    const dfloat OMEGA_MIN  = 1e-7_df;   // prevent omega -> 0 blowup when n < 1
    const dfloat OMEGA_MAX  = 2.0_df;    // physical upper bound (stability)
    const dfloat TOL        = 1e-5_df;
    const dfloat MAX_ITER   = 20;        // more iterations for n < 1

    for (int i = 0; i < MAX_ITER; i++)
    {
        // Clamp omega before pow to avoid NaN from negative base or zero^negative
        dfloat omegaSafe = fmaxf(omega, OMEGA_MIN);

        dfloat omn   = POW_FUNCTION(omegaSafe, n_index);           // omega^n
        dfloat omn1  = POW_FUNCTION(omegaSafe, n_index - 1.0_df); // omega^(n-1)

        dfloat fx    = A * omn + B * omegaSafe - C;
        dfloat fx_dx = A * n_index * omn1 + B;

        // Guard against degenerate derivative
        if (fabs(fx_dx) < 1e-12_df) break;

        dfloat delta = fx / fx_dx;

        // Residual tolerance check BEFORE update
        if (fabs(delta) < TOL * (1.0_df + fabs(omega))) break;

        // --- Damped Newton step (critical for n < 1) ---
        // Backtracking: halve step if it pushes omega out of bounds or increases |f|
        dfloat omegaNew = omega - delta;
        dfloat alpha    = 1.0_df;

        for (int ls = 0; ls < 8; ls++)   // line search
        {
            omegaNew = omega - alpha * delta;
            if (omegaNew >= OMEGA_MIN && omegaNew <= OMEGA_MAX) break;
            alpha *= 0.5_df;
        }

        omega = fmaxf(fminf(omegaNew, OMEGA_MAX), OMEGA_MIN);
    }

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

#ifdef LAMBDA_DIST
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

#endif



// ============================================================================
// VISCOELASTIC FLUID DISPATCH
// ============================================================================


// Forward declarations
__host__ __device__ veRelaxTerm __forceinline__ calcVeRelaxationTerm_OldroydB(dfloat inv_lambda, dfloat Axx, dfloat Axy, dfloat Axz, dfloat Ayy, dfloat Ayz, dfloat Azz);
__host__ __device__ veRelaxTerm __forceinline__ calcVeRelaxationTerm_FeneP(dfloat L_sq, dfloat inv_lambda, dfloat Axx, dfloat Axy, dfloat Axz, dfloat Ayy, dfloat Ayz, dfloat Azz);
__host__ __device__ veRelaxTerm __forceinline__ calcVeRelaxationTerm_Giesekus(dfloat alpha, dfloat inv_lambda, dfloat Axx, dfloat Axy, dfloat Axz, dfloat Ayy, dfloat Ayz, dfloat Azz);
__host__ __device__ veRelaxTerm __forceinline__ calcVeRelaxationTerm_PttLinear(dfloat epsilon, dfloat inv_lambda, dfloat Axx, dfloat Axy, dfloat Axz, dfloat Ayy, dfloat Ayz, dfloat Azz);
__host__ __device__ veRelaxTerm __forceinline__ calcVeRelaxationTerm_PttExponential(dfloat epsilon, dfloat inv_lambda, dfloat Axx, dfloat Axy, dfloat Axz, dfloat Ayy, dfloat Ayz, dfloat Azz);

__host__ __device__
veFluidProps __forceinline__ blendVeProps(
    const veFluidProps& A, const veFluidProps& B, dfloat h)
{
    h = fmax(0.0_df, fmin(1.0_df, h));
    const dfloat inv_h = 1.0_df - h;
    const dfloat lambda_min = 1.0e-5_df;
    const dfloat phase_eps = 1.0e-4_df;
    veFluidProps vp;

    if (h <= phase_eps) {
        return A;
    }
    if (h >= 1.0_df - phase_eps) {
        return B;
    }

    // 1. Resolve the active model type for the mixture
    if (A.type == B.type) {
        vp.type = A.type;
    } else if (A.type == VE_NEWTONIAN) { // Assuming VE_NEWTONIAN is defined
        vp.type = B.type;
    } else if (B.type == VE_NEWTONIAN) {
        vp.type = A.type;
    } else {
        // Edge Case: Mixing two DIFFERENT viscoelastic models (e.g., FENE-P and PTT).
        //TODO: Need figure out how to implement
    }

    // 2. Interpolate macroscopic properties
    vp.eta_p = inv_h * A.eta_p + h * B.eta_p;

    const dfloat eps = 1.0e-14_df;

    // Interpolate eta and lambda directly, then compute Gmix = eta_mix/lambda_mix.
    const dfloat eta_mix = vp.eta_p;
    const dfloat lambda_mix = inv_h * A.lambda + h * B.lambda;
    const dfloat lambda_mix_safe = fmax(lambda_mix, lambda_min);
    const dfloat Gmix = eta_mix / lambda_mix_safe;

    if (vp.eta_p <= eps || Gmix <= eps) {
        // No polymer contribution in mixture cell: treat it as Newtonian.
        vp.type = VE_NEWTONIAN;
        vp.eta_p = 0.0_df;
        vp.lambda = 0.0_df;
    } else {
        vp.lambda = fmax(vp.eta_p / Gmix, lambda_min);
    }

    // 3. Handle internal structural parameters
    bool same_ve_model = (A.type == B.type && A.type != VE_NEWTONIAN);

    switch (vp.type) {
        case VE_FENE_P:
            vp.u.fenep.L_sq = inv_h * A.u.fenep.L_sq + h * B.u.fenep.L_sq;
            break;
        case VE_GIESEKUS:
            vp.u.giesekus.alpha = inv_h * A.u.giesekus.alpha + h * B.u.giesekus.alpha;
            break;
        case VE_PTT_LINEAR:
        case VE_PTT_EXPONENTIAL:
            vp.u.ptt.epsilon = inv_h * A.u.ptt.epsilon + h * B.u.ptt.epsilon;
            break;
        default: 
            break;
    }

    return vp;
}

__host__ __device__
veRelaxTerm __forceinline__ calcVeRelaxationTerm(
    const veFluidProps& vp, dfloat inv_lambda_local,
    dfloat Axx, dfloat Axy, dfloat Axz,
    dfloat Ayy, dfloat Ayz, dfloat Azz)
{
    switch (vp.type) {
    case VE_NEWTONIAN:       return veRelaxTerm{};
    case VE_OLDROYD_B:       return calcVeRelaxationTerm_OldroydB(inv_lambda_local, Axx, Axy, Axz, Ayy, Ayz, Azz);
    case VE_FENE_P:          return calcVeRelaxationTerm_FeneP(vp.u.fenep.L_sq, inv_lambda_local, Axx, Axy, Axz, Ayy, Ayz, Azz);
    case VE_GIESEKUS:        return calcVeRelaxationTerm_Giesekus(vp.u.giesekus.alpha, inv_lambda_local, Axx, Axy, Axz, Ayy, Ayz, Azz);
    case VE_PTT_LINEAR:      return calcVeRelaxationTerm_PttLinear(vp.u.ptt.epsilon, inv_lambda_local, Axx, Axy, Axz, Ayy, Ayz, Azz);
    case VE_PTT_EXPONENTIAL: return calcVeRelaxationTerm_PttExponential(vp.u.ptt.epsilon, inv_lambda_local, Axx, Axy, Axz, Ayy, Ayz, Azz);
    default: return veRelaxTerm{};
    }
}

// ---- Oldroyd-B ---------------------------------------------------------------
__host__ __device__
veRelaxTerm __forceinline__ calcVeRelaxationTerm_OldroydB(
    dfloat inv_lambda,
    dfloat Axx, dfloat Axy, dfloat Axz,
    dfloat Ayy, dfloat Ayz, dfloat Azz)
{
    veRelaxTerm R;
    R.xx = inv_lambda * (1.0_df - Axx); 
    R.yy = inv_lambda * (1.0_df - Ayy);
    R.zz = inv_lambda * (1.0_df - Azz); 
    R.xy = inv_lambda * (-Axy);
    R.xz = inv_lambda * (-Axz);         
    R.yz = inv_lambda * (-Ayz);
    return R;
}

// ---- FENE-P ------------------------------------------------------------------
// aa = b/(b-trA),  bb = b/(b-3),  b = L_sq. Recovers Oldroyd-B as L_sq->inf.
__host__ __device__
veRelaxTerm __forceinline__ calcVeRelaxationTerm_FeneP(
    dfloat L_sq, dfloat inv_lambda,
    dfloat Axx, dfloat Axy, dfloat Axz,
    dfloat Ayy, dfloat Ayz, dfloat Azz)
{
    dfloat trA = Axx + Ayy + Azz;
    dfloat aa  = 1.0_df / (1.0_df - trA   / L_sq);
    dfloat bb  = 1.0_df / (1.0_df - 3.0_df / L_sq);
    veRelaxTerm R;
    R.xx = inv_lambda * (bb - aa * Axx); 
    R.yy = inv_lambda * (bb - aa * Ayy);
    R.zz = inv_lambda * (bb - aa * Azz); 
    R.xy = inv_lambda * (-aa * Axy);
    R.xz = inv_lambda * (-aa * Axz);     
    R.yz = inv_lambda * (-aa * Ayz);
    return R;
}

// ---- Giesekus ----------------------------------------------------------------
// G = (1/lambda)*(I - A - alpha*(A-I)^2).  alpha=0 -> Oldroyd-B.
__host__ __device__
veRelaxTerm __forceinline__ calcVeRelaxationTerm_Giesekus(
    dfloat alpha, dfloat inv_lambda,
    dfloat Axx, dfloat Axy, dfloat Axz,
    dfloat Ayy, dfloat Ayz, dfloat Azz)
{
    dfloat A2xx = Axx*Axx + Axy*Axy + Axz*Axz;
    dfloat A2yy = Axy*Axy + Ayy*Ayy + Ayz*Ayz;
    dfloat A2zz = Axz*Axz + Ayz*Ayz + Azz*Azz;
    dfloat A2xy = Axx*Axy + Axy*Ayy + Axz*Ayz;
    dfloat A2xz = Axx*Axz + Axy*Ayz + Axz*Azz;
    dfloat A2yz = Axy*Axz + Ayy*Ayz + Ayz*Azz;
    dfloat c1 = 1.0_df - alpha, c2 = 1.0_df - 2.0_df * alpha;
    veRelaxTerm R;
    R.xx = inv_lambda * (c1     - c2*Axx - alpha*A2xx);
    R.yy = inv_lambda * (c1     - c2*Ayy - alpha*A2yy);
    R.zz = inv_lambda * (c1     - c2*Azz - alpha*A2zz);
    R.xy = inv_lambda * (0.0_df - c2*Axy - alpha*A2xy);
    R.xz = inv_lambda * (0.0_df - c2*Axz - alpha*A2xz);
    R.yz = inv_lambda * (0.0_df - c2*Ayz - alpha*A2yz);
    return R;
}

// ---- PTT linear --------------------------------------------------------------
// f(trA) = 1 + epsilon*(trA - 3).  epsilon=0 -> Oldroyd-B.
__host__ __device__
veRelaxTerm __forceinline__ calcVeRelaxationTerm_PttLinear(
    dfloat epsilon, dfloat inv_lambda,
    dfloat Axx, dfloat Axy, dfloat Axz,
    dfloat Ayy, dfloat Ayz, dfloat Azz)
{
    dfloat f = 1.0_df + epsilon * (Axx + Ayy + Azz - 3.0_df);
    veRelaxTerm R;
    R.xx = inv_lambda * f * (1.0_df - Axx);
    R.yy = inv_lambda * f * (1.0_df - Ayy);
    R.zz = inv_lambda * f * (1.0_df - Azz);
    R.xy = inv_lambda * f * (-Axy);
    R.xz = inv_lambda * f * (-Axz);
    R.yz = inv_lambda * f * (-Ayz);
    return R;
}

// ---- PTT exponential ---------------------------------------------------------
// f(trA) = exp(epsilon*(trA - 3)).  epsilon=0 -> Oldroyd-B.
__host__ __device__
veRelaxTerm __forceinline__ calcVeRelaxationTerm_PttExponential(
    dfloat epsilon, dfloat inv_lambda,
    dfloat Axx, dfloat Axy, dfloat Axz,
    dfloat Ayy, dfloat Ayz, dfloat Azz)
{
    dfloat f = EXP_FUNCTION(epsilon * (Axx + Ayy + Azz - 3.0_df));
    veRelaxTerm R;
    R.xx = inv_lambda * f * (1.0_df - Axx); 
    R.yy = inv_lambda * f * (1.0_df - Ayy);
    R.zz = inv_lambda * f * (1.0_df - Azz); 
    R.xy = inv_lambda * f * (-Axy);
    R.xz = inv_lambda * f * (-Axz);    
    R.yz = inv_lambda * f * (-Ayz);
    return R;
}

#endif // __NNF_H

