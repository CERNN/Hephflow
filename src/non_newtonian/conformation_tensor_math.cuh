#ifndef __CONFORMATION_TENSOR_MATH_CUH
#define __CONFORMATION_TENSOR_MATH_CUH

/**
 * Representation-independent algebra for symmetric 3x3 conformation tensors.
 *
 * Sym3 stores the six independent entries of a symmetric tensor. VeMat3 is
 * reserved for operations whose result is not necessarily symmetric. The
 * eigensolver returns eigenvectors as columns of VeEigenSym3::vectors.
 */

#include "nnf_types.h"
#include <math.h>

#ifdef SINGLE_PRECISION
constexpr dfloat VE_REP_EPSILON = 1.1920928955078125e-7_df;
#else
constexpr dfloat VE_REP_EPSILON = 2.2204460492503131e-16_df;
#endif

struct VeMat3 {
    dfloat v[3][3];
};

struct VeEigenSym3 {
    dfloat value[3];
    VeMat3 vectors;
};

__host__ __device__ constexpr Sym3 veIdentity()
{
    return Sym3(1.0_df, 0.0_df, 0.0_df, 1.0_df, 0.0_df, 1.0_df);
}

__host__ __device__ __forceinline__ Sym3 veSquareSymmetric(const Sym3& a)
{
    return Sym3(
        a.xx*a.xx + a.xy*a.xy + a.xz*a.xz,
        a.xx*a.xy + a.xy*a.yy + a.xz*a.yz,
        a.xx*a.xz + a.xy*a.yz + a.xz*a.zz,
        a.xy*a.xy + a.yy*a.yy + a.yz*a.yz,
        a.xy*a.xz + a.yy*a.yz + a.yz*a.zz,
        a.xz*a.xz + a.yz*a.yz + a.zz*a.zz);
}

__host__ __device__ __forceinline__ bool veInverseSymmetric(
    const Sym3& a, Sym3& inverse, dfloat relativeTolerance = 64.0_df*VE_REP_EPSILON)
{
    const dfloat c00 = a.yy*a.zz-a.yz*a.yz;
    const dfloat c01 = a.xz*a.yz-a.xy*a.zz;
    const dfloat c02 = a.xy*a.yz-a.xz*a.yy;
    const dfloat c11 = a.xx*a.zz-a.xz*a.xz;
    const dfloat c12 = a.xy*a.xz-a.xx*a.yz;
    const dfloat c22 = a.xx*a.yy-a.xy*a.xy;
    const dfloat det = a.xx*c00+a.xy*c01+a.xz*c02;
    const dfloat scale = fmax(fmax(fabs(a.xx),fabs(a.yy)), fmax(fabs(a.zz),1.0_df));

    if (!isfinite(det) || fabs(det) <= relativeTolerance*scale*scale*scale) {
        return false;
    }

    const dfloat invDet = 1.0_df/det;
    inverse = Sym3(c00*invDet,c01*invDet,c02*invDet,
                   c11*invDet,c12*invDet,c22*invDet);
    return true;
}

__host__ __device__ __forceinline__ VeMat3 veSymToMat(const Sym3& s)
{
    return VeMat3{{{s.xx,s.xy,s.xz},
                   {s.xy,s.yy,s.yz},
                   {s.xz,s.yz,s.zz}}};
}

__host__ __device__ __forceinline__ Sym3 veMatToSym(const VeMat3& a)
{
    return Sym3(a.v[0][0], 0.5_df*(a.v[0][1]+a.v[1][0]),
                0.5_df*(a.v[0][2]+a.v[2][0]), a.v[1][1],
                0.5_df*(a.v[1][2]+a.v[2][1]), a.v[2][2]);
}

__host__ __device__ __forceinline__ VeMat3 veIdentityMat()
{
    return VeMat3{{{1.0_df,0.0_df,0.0_df},
                   {0.0_df,1.0_df,0.0_df},
                   {0.0_df,0.0_df,1.0_df}}};
}

__host__ __device__ __forceinline__ VeMat3 veMatMul(const VeMat3& a, const VeMat3& b)
{
    VeMat3 out{};
    for (int i=0; i<3; ++i) {
        for (int j=0; j<3; ++j) {
            for (int k=0; k<3; ++k) {
                out.v[i][j] += a.v[i][k]*b.v[k][j];
            }
        }
    }
    return out;
}

__host__ __device__ __forceinline__ VeMat3 veTranspose(const VeMat3& a)
{
    VeMat3 out{};
    for (int i=0; i<3; ++i) {
        for (int j=0; j<3; ++j) {
            out.v[i][j] = a.v[j][i];
        }
    }
    return out;
}

// One Jacobi plane rotation. Both the tensor and accumulated eigenvectors are
// updated, preserving the convention A = Q diag(lambda) Q^T.
__host__ __device__ __forceinline__ void veJacobiRotate(VeMat3& a, VeMat3& q, int p, int r)
{
    const dfloat apq = a.v[p][r];
    if (apq == 0.0_df) {
        return;
    }

    const dfloat tau = (a.v[r][r]-a.v[p][p])/(2.0_df*apq);
    const dfloat t = (tau >= 0.0_df ? 1.0_df : -1.0_df)
                   / (fabs(tau)+sqrt(1.0_df+tau*tau));
    const dfloat c = 1.0_df/sqrt(1.0_df+t*t);
    const dfloat s = t*c;
    const dfloat app = a.v[p][p];
    const dfloat arr = a.v[r][r];

    a.v[p][p] = app-t*apq;
    a.v[r][r] = arr+t*apq;
    a.v[p][r] = a.v[r][p] = 0.0_df;

    for (int k=0; k<3; ++k) {
        if (k!=p && k!=r) {
            const dfloat akp = a.v[k][p];
            const dfloat akr = a.v[k][r];
            a.v[k][p] = a.v[p][k] = c*akp-s*akr;
            a.v[k][r] = a.v[r][k] = s*akp+c*akr;
        }
    }

    for (int k=0; k<3; ++k) {
        const dfloat qkp = q.v[k][p];
        const dfloat qkr = q.v[k][r];
        q.v[k][p] = c*qkp-s*qkr;
        q.v[k][r] = s*qkp+c*qkr;
    }
}

// Ten fixed cyclic sweeps avoid convergence-state divergence between GPU
// threads and are sufficient for the small symmetric tensors used here.
__host__ __device__ __forceinline__ VeEigenSym3 veEigenSymmetric(const Sym3& s)
{
    VeMat3 a = veSymToMat(s);
    VeMat3 q = veIdentityMat();
    for (int sweep=0; sweep<10; ++sweep) {
        veJacobiRotate(a,q,0,1);
        veJacobiRotate(a,q,0,2);
        veJacobiRotate(a,q,1,2);
    }
    return VeEigenSym3{{a.v[0][0],a.v[1][1],a.v[2][2]},q};
}

__host__ __device__ __forceinline__ VeMat3 veRotateQtMQ(const VeEigenSym3& e, const VeMat3& m)
{
    VeMat3 out{};
    for (int i=0; i<3; ++i) {
        for (int j=0; j<3; ++j) {
            for (int k=0; k<3; ++k) {
                for (int l=0; l<3; ++l) {
                    out.v[i][j] += e.vectors.v[k][i]*m.v[k][l]*e.vectors.v[l][j];
                }
            }
        }
    }
    return out;
}

__host__ __device__ __forceinline__ Sym3 veRotateQSQt(const VeEigenSym3& e, const Sym3& s)
{
    const VeMat3 m = veSymToMat(s);
    VeMat3 out{};
    for (int i=0; i<3; ++i) {
        for (int j=0; j<3; ++j) {
            for (int k=0; k<3; ++k) {
                for (int l=0; l<3; ++l) {
                    out.v[i][j] += e.vectors.v[i][k]*m.v[k][l]*e.vectors.v[j][l];
                }
            }
        }
    }
    return veMatToSym(out);
}

__host__ __device__ __forceinline__ Sym3 veRotateQtSQ(const VeEigenSym3& e, const Sym3& s)
{
    return veMatToSym(veRotateQtMQ(e, veSymToMat(s)));
}

#endif
