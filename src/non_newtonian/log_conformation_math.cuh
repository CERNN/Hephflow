#ifndef __LOG_CONFORMATION_MATH_CUH
#define __LOG_CONFORMATION_MATH_CUH

/**
 * Spectral functions used by the log-conformation representation.
 *
 * For Psi = log(A), veExpSymmetric reconstructs A = exp(Psi), while
 * veFrechetLogFromPsi maps a material derivative in A-space to Psi-space.
 */

#include "conformation_tensor_math.cuh"

// Stable divided difference (exp(pi)-exp(pj))/(pi-pj). The even series around
// pi=pj avoids cancellation when eigenvalues are nearly repeated.
__host__ __device__ __forceinline__ dfloat veExpDividedDifference(
    dfloat pi, dfloat pj)
{
    const dfloat d = pi-pj;
    const dfloat mid = 0.5_df*(pi+pj);
    if (fabs(d) < 1.0e-3_df) {
        const dfloat d2 = d*d;
        return EXP_FUNCTION(mid)*(1.0_df+d2/24.0_df+d2*d2/1920.0_df);
    }
    return EXP_FUNCTION(mid)*(2.0_df*sinh(0.5_df*d)/d);
}

__host__ __device__ __forceinline__ Sym3 veExpSymmetric(
    const Sym3& psi, VeEigenSym3* eigOut = nullptr)
{
    VeEigenSym3 e = veEigenSymmetric(psi);
    const Sym3 diagonal(EXP_FUNCTION(e.value[0]),0.0_df,0.0_df,
                        EXP_FUNCTION(e.value[1]),0.0_df,EXP_FUNCTION(e.value[2]));
    if (eigOut) {
        *eigOut = e;
    }
    return veRotateQSQt(e, diagonal);
}

__host__ __device__ __forceinline__ Sym3 veLogSymmetric(const Sym3& a)
{
    VeEigenSym3 e = veEigenSymmetric(a);
    return veRotateQSQt(e, Sym3(log(e.value[0]),0.0_df,0.0_df,
                                log(e.value[1]),0.0_df,log(e.value[2])));
}

// D exp_Psi[E], evaluated in Psi's eigenbasis with exponential divided
// differences for the off-diagonal components.
__host__ __device__ __forceinline__ Sym3 veFrechetExp(
    const VeEigenSym3& e, const Sym3& direction)
{
    const Sym3 b = veRotateQtSQ(e, direction);
    const dfloat a0 = EXP_FUNCTION(e.value[0]);
    const dfloat a1 = EXP_FUNCTION(e.value[1]);
    const dfloat a2 = EXP_FUNCTION(e.value[2]);
    const Sym3 transformed(
        a0*b.xx, veExpDividedDifference(e.value[0],e.value[1])*b.xy,
        veExpDividedDifference(e.value[0],e.value[2])*b.xz, a1*b.yy,
        veExpDividedDifference(e.value[1],e.value[2])*b.yz, a2*b.zz);
    return veRotateQSQt(e, transformed);
}

// D log_A[E] with A=exp(Psi). This is the inverse of D exp_Psi in the shared
// eigenbasis and supplies the transported source D(Psi)/Dt.
__host__ __device__ __forceinline__ Sym3 veFrechetLogFromPsi(
    const VeEigenSym3& e, const Sym3& directionInA)
{
    const Sym3 b = veRotateQtSQ(e, directionInA);
    const dfloat a0 = EXP_FUNCTION(e.value[0]);
    const dfloat a1 = EXP_FUNCTION(e.value[1]);
    const dfloat a2 = EXP_FUNCTION(e.value[2]);
    const Sym3 transformed(
        b.xx/a0, b.xy/veExpDividedDifference(e.value[0],e.value[1]),
        b.xz/veExpDividedDifference(e.value[0],e.value[2]), b.yy/a1,
        b.yz/veExpDividedDifference(e.value[1],e.value[2]), b.zz/a2);
    return veRotateQSQt(e, transformed);
}

#endif
