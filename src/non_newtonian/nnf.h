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
#include "log_conformation_math.cuh"

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
__host__ __device__ dfloat __forceinline__ calcOmegaHerschelBulkley(dfloat k_consistency, dfloat n_index, dfloat s_y, dfloat omegaOld, dfloat const auxStressMag, dfloat rhoVar);
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
        newOmegaVar = calcOmegaHerschelBulkley(fp.u.hb.k_consistency, fp.u.hb.n_index, fp.u.hb.s_y, omegaOld, auxStressMag, rhoVar);
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

    // The two most common indices have closed forms.  Besides being exact,
    // these avoid the iterative solver and all per-iteration pow() calls.
    // NNF_DISABLE_ANALYTIC_FAST_PATHS is retained for performance A/B tests.
#ifndef NNF_DISABLE_ANALYTIC_FAST_PATHS
    if (n_index == 1.0_df)
        return auxStressMag / (a + b);
    if (n_index == 0.5_df) {
        const dfloat discriminant = a * a + 4.0_df * b * auxStressMag;
        const dfloat root = (2.0_df * auxStressMag) /
                            (a + sqrt(discriminant));
        return root * root;
    }
#endif

    for (int i = 0; i < 7; i++){
        // omega is positive for physical LBM relaxation rates.  Reusing
        // omega^n avoids a second, expensive pow() in every Newton step.
        const dfloat omegaPowN = POW_FUNCTION(omega, n_index);
        const dfloat omegaPowNm1 = (omega != 0.0_df)
                                  ? (omegaPowN / omega)
                                  : POW_FUNCTION(omega, n_index - 1.0_df);
        fx = a * omegaPowN + b * omega + c;
        fx_dx = a * n_index * omegaPowNm1 + b;

        const dfloat delta = fx / fx_dx;
        if (fabs(delta) < 1e-6_df){
            break;
        }
            
        omega -= delta;
    }
    return omega;
}

__host__ __device__ 
dfloat __forceinline__ calcOmegaBingham(dfloat omega_p, dfloat s_y, dfloat auxStressMag){
    if (s_y <= 0.0_df)
        return omega_p;
    // The unyielded branch is exactly zero and needs no division.  Testing
    // against s_y also subsumes the near-zero-stress guard for s_y > 0.
    if (auxStressMag <= s_y)
        return 0.0_df;
    return omega_p * (1.0_df - s_y / auxStressMag);
}

__host__ __device__ 
dfloat __forceinline__ calcOmegaHerschelBulkley(
    dfloat k_consistency, dfloat n_index, dfloat s_y, 
    dfloat omegaOld, dfloat const auxStressMag, dfloat rhoVar)
{
    const dfloat cs2 = 1.0_df / 3.0_df;
    const dfloat OMEGA_MIN = 1.0e-7_df;
#ifdef SINGLE_PRECISION
    const dfloat OMEGA_MAX = 2.0_df - 1.0e-6_df;
    const dfloat RES_TOL   = 2.0e-6_df;
    const dfloat STEP_TOL  = 2.0e-6_df;
#else
    const dfloat OMEGA_MAX = 2.0_df - 1.0e-12_df;
    const dfloat RES_TOL   = 1.0e-12_df;
    const dfloat STEP_TOL  = 1.0e-12_df;
#endif
    const int MAX_ITER = 50;

    // Invalid material data must not be allowed to inject NaNs into collision.
    // Keep a finite previous value as a conservative device-side fallback;
    // material parameters should additionally be validated when a case is set up.
    dfloat omegaFallback = OMEGA_MIN;
    if (isfinite(omegaOld))
        omegaFallback = fmax(0.0_df, fmin(omegaOld, OMEGA_MAX));

    if (!isfinite(k_consistency) || !isfinite(n_index) ||
        !isfinite(s_y) || !isfinite(auxStressMag) || !isfinite(rhoVar) ||
        k_consistency < 0.0_df || n_index <= 0.0_df || s_y < 0.0_df ||
        auxStressMag < 0.0_df || rhoVar <= 0.0_df)
        return omegaFallback;

    // auxStressMag is |Pi_neq|.  The exact unyielded branch has omega = 0;
    // a yielded root exists if and only if |Pi_neq| > yield stress.
    if (auxStressMag <= s_y)
        return 0.0_df;

    const dfloat B = auxStressMag;
    const dfloat C = 2.0_df * (auxStressMag - s_y);

    // Computing the ratio before pow avoids separate overflow/underflow in
    // |Pi_neq|^n and (rho*c_s^2)^n.
    const dfloat stressRatio = auxStressMag / (rhoVar * cs2);
    const dfloat A = 2.0_df * k_consistency * POW_FUNCTION(stressRatio, n_index);

    if (!isfinite(A))
        return OMEGA_MIN; // an overflowing consistency term implies omega -> 0

    // K = 0 leaves a linear equation and avoids evaluating an unnecessary
    // omega^(n-1) in the Newton derivative.
    if (A == 0.0_df)
        return fmax(OMEGA_MIN, fmin(C / B, OMEGA_MAX));

    // Closed forms for common rheological indices avoid the safeguarded
    // Newton loop entirely.  The formulas use the positive physical root.
#ifndef NNF_DISABLE_ANALYTIC_FAST_PATHS
    if (n_index == 1.0_df)
        return fmax(OMEGA_MIN, fmin(C / (A + B), OMEGA_MAX));
    if (n_index == 0.5_df) {
        const dfloat discriminant = A * A + 4.0_df * B * C;
        if (!isfinite(discriminant))
            return OMEGA_MIN;
        const dfloat root = (2.0_df * C) / (A + sqrt(discriminant));
        return fmax(OMEGA_MIN, fmin(root * root, OMEGA_MAX));
    }
    if (n_index == 2.0_df) {
        const dfloat discriminant = B * B + 4.0_df * A * C;
        if (!isfinite(discriminant))
            return OMEGA_MIN;
        const dfloat omegaExact = (2.0_df * C) /
                                  (B + sqrt(discriminant));
        return fmax(OMEGA_MIN, fmin(omegaExact, OMEGA_MAX));
    }
#endif

    // For physical parameters f(omega)=A*omega^n+B*omega-C is strictly
    // increasing.  Maintain a sign-changing bracket and accept a Newton step
    // only when it remains inside that bracket; otherwise use bisection.
    dfloat omegaLow  = OMEGA_MIN;
    dfloat omegaHigh = OMEGA_MAX;

    dfloat lowPow = POW_FUNCTION(omegaLow, n_index);
    const dfloat fLow = A * lowPow + B * omegaLow - C;
    if (!isfinite(fLow) || fLow >= 0.0_df)
        return OMEGA_MIN; // the mathematical root is below the configured floor

    dfloat highPow = POW_FUNCTION(omegaHigh, n_index);
    const dfloat fHigh = A * highPow + B * omegaHigh - C;
    if (!isfinite(fHigh) || fHigh <= 0.0_df)
        return OMEGA_MAX; // the root is at/above the admissible LBM limit

    dfloat omega = (omegaOld > omegaLow && omegaOld < omegaHigh && isfinite(omegaOld))
                 ? omegaOld
                 : 0.5_df * (omegaLow + omegaHigh);

    for (int i = 0; i < MAX_ITER; ++i)
    {
        const dfloat omn = POW_FUNCTION(omega, n_index);
        const dfloat fx  = A * omn + B * omega - C;
        const dfloat residualScale = fabs(A * omn) + fabs(B * omega) + fabs(C);

        if (isfinite(fx) && fabs(fx) <= RES_TOL * residualScale)
            return omega;

        if (!isfinite(fx)) {
            omega = 0.5_df * (omegaLow + omegaHigh);
            continue;
        }

        if (fx < 0.0_df) {
            omegaLow = omega;
        } else {
            omegaHigh = omega;
        }

        const dfloat midpoint = 0.5_df * (omegaLow + omegaHigh);
        const dfloat omegaScale = fmax(fabs(midpoint), OMEGA_MIN);
        if ((omegaHigh - omegaLow) <= STEP_TOL * omegaScale)
            return midpoint;

        // omega is bracketed above OMEGA_MIN, so omega^(n-1) can be obtained
        // from the already computed omega^n without another pow().
        const dfloat derivative = A * n_index * (omn / omega) + B;
        dfloat candidate = midpoint;

        if (isfinite(derivative) && derivative > 0.0_df) {
            const dfloat newtonCandidate = omega - fx / derivative;
            const dfloat bracketQuarter = 0.25_df * (omegaHigh - omegaLow);
            if (isfinite(newtonCandidate) &&
                newtonCandidate > omegaLow + bracketQuarter &&
                newtonCandidate < omegaHigh - bracketQuarter)
                candidate = newtonCandidate;
        }

        omega = candidate;
    }

    // Bisection guarantees this is bounded even if the residual tolerance was
    // not reached because of floating-point resolution.
    return 0.5_df * (omegaLow + omegaHigh);
}


// EXPERIMENTAL: algebraically regression-tested, not yet physically validated.
__host__ __device__ 
dfloat __forceinline__ calcOmegaBiViscosity(dfloat omega_y, dfloat omega_p, dfloat s_y, dfloat visc_ratio, dfloat auxStressMag){
    if (s_y <= 0.0_df)
        return myMax(omega_y, omega_p);
    if (auxStressMag <= 1.0e-12_df)
        return omega_y;
    return myMax(omega_y, omega_p * (1.0_df - s_y * (1.0_df - visc_ratio) / auxStressMag));
}

// EXPERIMENTAL: algebraically regression-tested, not yet physically validated.
// https://arxiv.org/abs/2401.02942 has an analytical solution.
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
        const dfloat expTerm = EXP_FUNCTION(D * omega);
        fx = omega * (A + C * expTerm) + E;
        fx_dx = A + C * expTerm * (1.0_df + D * omega);

        const dfloat delta = fx / fx_dx;
        if (fabs(delta) < 1e-6_df){
            break;
        }
            
        omega -= delta;
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
dfloat __forceinline__ calcVisco_thixo(const fluidProps& fp, dfloat lambda_actual, dfloat gammaDot)
{
    // The caller has already removed the distribution offset.
    lambda_actual = fmax(0.0_df, fmin(1.0_df, lambda_actual));
    
    switch(fp.u.thixo.model)
    {
        case THIXO_MOORE1959:
            return lambda_actual * fp.u.thixo.u.moore1959.eta_0;
        case THIXO_WORRALL1964:
            return lambda_actual * fp.u.thixo.u.worrall1964.eta_0;
        case THIXO_HOUSKA1980:
            return lambda_actual * fp.u.thixo.u.houska1980.k_consistency * POW_FUNCTION(fmax(gammaDot, 1e-12_df), fp.u.thixo.u.houska1980.n_index - 1.0_df);
        case THIXO_TOORMAN1997:
            return lambda_actual * fp.u.thixo.u.toorman1997.eta_0;
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

__host__ __device__ __forceinline__ dfloat calcVePolymerModulus(
    const veFluidProps& vp)
{
    return vp.type != VE_NEWTONIAN && vp.lambda > 0.0_df
        ? vp.eta_p / vp.lambda : 0.0_df;
}

__host__ __device__ __forceinline__ bool validateVeFluidProps(
    const veFluidProps& vp)
{
    if (vp.type == VE_NEWTONIAN) return vp.eta_p >= 0.0_df;
    if (vp.eta_p < 0.0_df || vp.lambda <= 0.0_df) return false;
    if ((vp.type == VE_FENE_P || vp.type == VE_FENE_CR) && vp.fene_L2 <= 3.0_df)
        return false;
    if (vp.type == VE_GIESEKUS &&
        (vp.giesekus_alpha < 0.0_df || vp.giesekus_alpha > 0.5_df)) return false;
    if ((vp.type == VE_PTT_LINEAR || vp.type == VE_PTT_EXPONENTIAL) &&
        vp.ptt_epsilon < 0.0_df) return false;
    return true;
}

__host__ __device__ __forceinline__ dfloat calcVePeterlinFactor(
    const veFluidProps& vp, const Sym3& A)
{
    return (vp.fene_L2 - 3.0_df) / (vp.fene_L2 - A.trace());
}

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
        // Unsupported mixed constitutive laws are made inert deterministically;
        // host-side case validation should reject this configuration.
        vp.type = VE_NEWTONIAN;
    }

    // 2. Interpolate macroscopic properties
    vp.eta_p = inv_h * A.eta_p + h * B.eta_p;

    const dfloat eps = 1.0e-14_df;

    // Interpolate eta and lambda directly.  The previous eta/lambda followed
    // by eta/G round trip reduced algebraically to this same lambda and cost
    // two divisions in every diffuse-interface cell.
    const dfloat eta_mix = vp.eta_p;
    const dfloat lambda_mix = inv_h * A.lambda + h * B.lambda;
    const dfloat lambda_mix_safe = fmax(lambda_mix, lambda_min);

    if (eta_mix <= eps || eta_mix <= eps * lambda_mix_safe) {
        // No polymer contribution in mixture cell: treat it as Newtonian.
        vp.type = VE_NEWTONIAN;
        vp.eta_p = 0.0_df;
        vp.lambda = 0.0_df;
    } else {
        vp.lambda = lambda_mix_safe;
    }

    // 3. Handle internal structural parameters
    switch (vp.type) {
        case VE_FENE_P:
        case VE_FENE_CR:
            vp.fene_L2 = A.type == VE_NEWTONIAN ? B.fene_L2
                       : B.type == VE_NEWTONIAN ? A.fene_L2
                       : inv_h*A.fene_L2+h*B.fene_L2;
            break;
        case VE_GIESEKUS:
            vp.giesekus_alpha = A.type == VE_NEWTONIAN ? B.giesekus_alpha
                              : B.type == VE_NEWTONIAN ? A.giesekus_alpha
                              : inv_h*A.giesekus_alpha+h*B.giesekus_alpha;
            break;
        case VE_PTT_LINEAR:
        case VE_PTT_EXPONENTIAL:
            vp.ptt_epsilon = A.type == VE_NEWTONIAN ? B.ptt_epsilon
                           : B.type == VE_NEWTONIAN ? A.ptt_epsilon
                           : inv_h*A.ptt_epsilon+h*B.ptt_epsilon;
            break;
        default: 
            break;
    }

    return vp;
}

__host__ __device__ __forceinline__ Sym3 calcVeRelaxation(
    const veFluidProps& vp, const Sym3& A)
{
    if (vp.type == VE_NEWTONIAN || vp.lambda <= 0.0_df) return Sym3{};
    const dfloat invLambda = 1.0_df / vp.lambda;
    const Sym3 B = A - veIdentity();
    switch (vp.type) {
    case VE_OLDROYD_B:
        return B * (-invLambda);
    case VE_FENE_P: {
        const dfloat f = calcVePeterlinFactor(vp, A);
        return (veIdentity() - A*f) * invLambda;
    }
    case VE_FENE_CR: {
        const dfloat f = calcVePeterlinFactor(vp, A);
        return B * (-f*invLambda);
    }
    case VE_GIESEKUS:
        return (B + veSquareSymmetric(B)*vp.giesekus_alpha) * (-invLambda);
    case VE_PTT_LINEAR: {
        const dfloat f = 1.0_df + vp.ptt_epsilon*(A.trace()-3.0_df);
        return B * (-f*invLambda);
    }
    case VE_PTT_EXPONENTIAL: {
        const dfloat f = EXP_FUNCTION(vp.ptt_epsilon*(A.trace()-3.0_df));
        return B * (-f*invLambda);
    }
    default:
        return Sym3{};
    }
}

__host__ __device__ __forceinline__ Sym3 calcVeStressReduced(
    const veFluidProps& vp, const Sym3& A)
{
    const Sym3 B = A - veIdentity();
    if (vp.type == VE_FENE_P)
        return A*calcVePeterlinFactor(vp, A) - veIdentity();
    if (vp.type == VE_FENE_CR)
        return B*calcVePeterlinFactor(vp, A);
    return vp.type == VE_NEWTONIAN ? Sym3{} : B;
}

__host__ __device__ __forceinline__ Sym3 calcVeStressReducedDifferential(
    const veFluidProps& vp, const Sym3& A, const Sym3& dA)
{
    if (vp.type != VE_FENE_P && vp.type != VE_FENE_CR)
        return vp.type == VE_NEWTONIAN ? Sym3{} : dA;
    const dfloat denominator = vp.fene_L2 - A.trace();
    const dfloat f = calcVePeterlinFactor(vp, A);
    const dfloat df = f*dA.trace()/denominator;
    const Sym3 base = vp.type == VE_FENE_P ? A : (A-veIdentity());
    return dA*f + base*df;
}

#endif // __NNF_H
