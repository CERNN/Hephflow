#ifndef __SQRT_CONFORMATION_MATH_CUH
#define __SQRT_CONFORMATION_MATH_CUH

/**
 * Symmetric square-root conformation mathematics.
 *
 * The transported tensor is the positive root Q, with A=Q^2. Following
 * Balci et al., its source is
 *
 *   DQ/Dt = GQ + QL^T + 1/2 Q^{-1} R(A),
 *
 * where G is antisymmetric and chosen so that DQ/Dt remains symmetric.
 */

#include "conformation_tensor_math.cuh"

enum VeSqrtDiagnostic : unsigned int {
    VE_SQRT_DIAG_NONE = 0u,
    VE_SQRT_DIAG_NON_POSITIVE_Q = 1u << 0,
    VE_SQRT_DIAG_NON_POSITIVE_PIVOT = 1u << 1,
    VE_SQRT_DIAG_NEAR_SINGULAR_Q = 1u << 2,
    VE_SQRT_DIAG_NONFINITE_SOURCE = 1u << 3
};

__host__ __device__ __forceinline__ Sym3 veSqrtSymmetric(const Sym3& a, bool* ok = nullptr)
{
    const VeEigenSym3 eig = veEigenSymmetric(a);
    const bool valid = eig.value[0] > 0.0_df && eig.value[1] > 0.0_df && eig.value[2] > 0.0_df;
    if (ok) {
        *ok = valid;
    }
    if (!valid) {
        const dfloat nanValue = static_cast<dfloat>(NAN);
        return Sym3(nanValue,nanValue,nanValue,nanValue,nanValue,nanValue);
    }
    return veRotateQSQt(eig, Sym3(sqrt(eig.value[0]),0.0_df,0.0_df,
                                  sqrt(eig.value[1]),0.0_df,sqrt(eig.value[2])));
}

// Inverse differential of A=Q^2. For symmetric SPD Q and symmetric H,
// X=D(sqrt)_A[H] is the unique symmetric solution of QX+XQ=H.
__host__ __device__ __forceinline__ Sym3 veSquareRootTangent(
    const Sym3& q, const Sym3& directionInA)
{
    const VeEigenSym3 eig=veEigenSymmetric(q);
    const Sym3 h=veRotateQtSQ(eig,directionInA);
    const Sym3 x(
        h.xx/(2.0_df*eig.value[0]),
        h.xy/(eig.value[0]+eig.value[1]),
        h.xz/(eig.value[0]+eig.value[2]),
        h.yy/(2.0_df*eig.value[1]),
        h.yz/(eig.value[1]+eig.value[2]),
        h.zz/(2.0_df*eig.value[2]));
    return veRotateQSQt(eig,x);
}

// Fixed 3x3 Cholesky solve. The Balci system is SPD whenever Q is SPD; a
// failed pivot is therefore also a useful representation-validity diagnostic.
__host__ __device__ __forceinline__ bool veSolveSpd3(
    const dfloat m[3][3], const dfloat rhs[3], dfloat x[3])
{
    dfloat l[3][3]{};
    dfloat scale=1.0_df;
    for (int i=0; i<3; ++i) {
        scale=fmax(scale,fabs(m[i][i]));
    }

    for (int i=0; i<3; ++i) {
        for (int j=0; j<=i; ++j) {
            dfloat sum=m[i][j];
            for (int k=0; k<j; ++k) {
                sum-=l[i][k]*l[j][k];
            }
            if (i==j) {
                if (!isfinite(sum) || sum<=64.0_df*VE_REP_EPSILON*scale) {
                    return false;
                }
                l[i][j]=sqrt(sum);
            } else {
                l[i][j]=sum/l[j][j];
            }
        }
    }

    dfloat y[3]{};
    for (int i=0; i<3; ++i) {
        dfloat sum=rhs[i];
        for (int k=0; k<i; ++k) {
            sum-=l[i][k]*y[k];
        }
        y[i]=sum/l[i][i];
    }
    for (int i=2; i>=0; --i) {
        dfloat sum=y[i];
        for (int k=i+1; k<3; ++k) {
            sum-=l[k][i]*x[k];
        }
        x[i]=sum/l[i][i];
    }
    return isfinite(x[0]) && isfinite(x[1]) && isfinite(x[2]);
}

// Solve LQ-QL^T=GQ+QG for the three independent entries of antisymmetric G,
// then assemble the Balci square-root source shown in the file-level comment.
__host__ __device__ __forceinline__ Sym3 veSquareRootSource(
    const Sym3& q, const VeMat3& l, const Sym3& relaxation, bool* ok = nullptr,
    unsigned int* diagnostics = nullptr)
{
    unsigned int diagnostic=VE_SQRT_DIAG_NONE;
    const dfloat qScale=fmax(1.0_df,fmax(fabs(q.xx),fmax(fabs(q.yy),fabs(q.zz))));
    const dfloat qMinor2=q.xx*q.yy-q.xy*q.xy;
    const dfloat qDet=q.xx*(q.yy*q.zz-q.yz*q.yz)
        -q.xy*(q.xy*q.zz-q.xz*q.yz)+q.xz*(q.xy*q.yz-q.xz*q.yy);
    const dfloat qTolerance=64.0_df*VE_REP_EPSILON;

    if (!isfinite(q.xx)||!isfinite(qMinor2)||!isfinite(qDet)||q.xx<=0.0_df||
        qMinor2<=0.0_df||qDet<=0.0_df) {
        diagnostic|=VE_SQRT_DIAG_NON_POSITIVE_Q;
    }
    if (!isfinite(qDet)||fabs(qDet)<=qTolerance*qScale*qScale*qScale) {
        diagnostic|=VE_SQRT_DIAG_NEAR_SINGULAR_Q;
    }

    const VeMat3 qm=veSymToMat(q);
    const VeMat3 lm=veMatMul(l,qm);
    const VeMat3 qlt=veMatMul(qm,veTranspose(l));
    VeMat3 c{};
    for (int i=0; i<3; ++i) {
        for (int j=0; j<3; ++j) {
            c.v[i][j]=lm.v[i][j]-qlt.v[i][j];
        }
    }

    const dfloat system[3][3]={{q.xx+q.yy,q.yz,-q.xz},
                               {q.yz,q.xx+q.zz,q.xy},
                               {-q.xz,q.xy,q.yy+q.zz}};
    const dfloat rhs[3]={c.v[0][1],c.v[0][2],c.v[1][2]};
    dfloat gv[3]{};
    Sym3 qInv;
    const bool solveValid=veSolveSpd3(system,rhs,gv);
    if (!solveValid) {
        diagnostic|=VE_SQRT_DIAG_NON_POSITIVE_PIVOT;
    }
    const bool inverseValid=veInverseSymmetric(q,qInv);
    if (!inverseValid) {
        diagnostic|=VE_SQRT_DIAG_NEAR_SINGULAR_Q;
    }

    const bool valid=diagnostic==VE_SQRT_DIAG_NONE && solveValid && inverseValid;
    if (ok) {
        *ok=valid;
    }
    if (!valid) {
        if (diagnostics) {
            *diagnostics=diagnostic;
        }
        const dfloat nanValue=static_cast<dfloat>(NAN);
        return Sym3(nanValue,nanValue,nanValue,nanValue,nanValue,nanValue);
    }

    const VeMat3 g{{{0.0_df,gv[0],gv[1]},
                    {-gv[0],0.0_df,gv[2]},
                    {-gv[1],-gv[2],0.0_df}}};
    const VeMat3 qInvR=veMatMul(veSymToMat(qInv),veSymToMat(relaxation));
    VeMat3 source=veMatMul(g,qm);
    const VeMat3 qltSource=veMatMul(qm,veTranspose(l));
    for (int i=0; i<3; ++i) {
        for (int j=0; j<3; ++j) {
            source.v[i][j]+=qltSource.v[i][j]+0.5_df*qInvR.v[i][j];
        }
    }

    const Sym3 result=veMatToSym(source);
    if (!isfinite(result.xx)||!isfinite(result.xy)||!isfinite(result.xz)||
        !isfinite(result.yy)||!isfinite(result.yz)||!isfinite(result.zz)) {
        diagnostic|=VE_SQRT_DIAG_NONFINITE_SOURCE;
    }
    if (diagnostics) {
        *diagnostics=diagnostic;
    }
    if (ok) {
        *ok=diagnostic==VE_SQRT_DIAG_NONE;
    }
    return result;
}

#endif
