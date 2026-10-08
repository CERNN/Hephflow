#ifndef __CONFORMATION_REPRESENTATION_CUH
#define __CONFORMATION_REPRESENTATION_CUH

/**
 * Conversion boundary between stored and physical conformation tensors.
 *
 * The transport stores six symmetric components, interpreted as A,
 * Psi=log(A), or Q=sqrt(A) according to CONF_REPRESENTATION. Consumers outside
 * the evolution equations should use these helpers rather than inspect the
 * representation macros themselves.
 */

#include "log_conformation_math.cuh"
#include "sqrt_conformation_math.cuh"

inline constexpr const char* veConformationRepresentationName()
{
#if CONF_REPRESENTATION == CONF_REP_LOG
    return "log";
#elif CONF_REPRESENTATION == CONF_REP_SQRT
    return "sqrt";
#else
    return "standard";
#endif
}

inline constexpr const char* veConformationRepresentationDescription()
{
#if CONF_REPRESENTATION == CONF_REP_LOG
    return "log(A)";
#elif CONF_REPRESENTATION == CONF_REP_SQRT
    return "symmetric sqrt(A)";
#else
    return "standard A";
#endif
}

__host__ __device__ __forceinline__ Sym3 vePhysicalAFromStored(const Sym3& stored)
{
#if !defined(CONFORMATION_TENSOR)
    return stored;
#elif CONF_REPRESENTATION == CONF_REP_LOG
    return veExpSymmetric(stored);
#elif CONF_REPRESENTATION == CONF_REP_SQRT
    return veSquareSymmetric(stored);
#else
    return stored-Sym3(CONF_ZERO,CONF_ZERO,CONF_ZERO,
                       CONF_ZERO,CONF_ZERO,CONF_ZERO);
#endif
}

__host__ __device__ __forceinline__ Sym3 veStoredFromPhysicalA(const Sym3& a)
{
#if !defined(CONFORMATION_TENSOR)
    return a;
#elif CONF_REPRESENTATION == CONF_REP_LOG
    return veLogSymmetric(a);
#elif CONF_REPRESENTATION == CONF_REP_SQRT
    return veSqrtSymmetric(a);
#else
    return a+Sym3(CONF_ZERO,CONF_ZERO,CONF_ZERO,
                  CONF_ZERO,CONF_ZERO,CONF_ZERO);
#endif
}

#ifdef CONFORMATION_TENSOR
// Third-order wall continuation performed in physical-A space. Legacy case
// boundary fragments call this only for transformed representations; applying
// the polynomial extrapolation directly to log(A) or sqrt(A) imposes a
// representation-dependent wall condition at high conformation stretch.
__device__ __forceinline__ Sym3 veExtrapolateStoredFromPhysicalAY(
    const dfloat* fMom, int tx, int ty, int tz, int bx, int by, int bz,
    int fluidDirection)
{
    #define LOAD_STORED_AT_Y_OFFSET(offset) Sym3( \
        fMom[idxMom(tx,ty+(offset),tz,A_XX_C_INDEX,bx,by,bz)], \
        fMom[idxMom(tx,ty+(offset),tz,A_XY_C_INDEX,bx,by,bz)], \
        fMom[idxMom(tx,ty+(offset),tz,A_XZ_C_INDEX,bx,by,bz)], \
        fMom[idxMom(tx,ty+(offset),tz,A_YY_C_INDEX,bx,by,bz)], \
        fMom[idxMom(tx,ty+(offset),tz,A_YZ_C_INDEX,bx,by,bz)], \
        fMom[idxMom(tx,ty+(offset),tz,A_ZZ_C_INDEX,bx,by,bz)])

    const Sym3 a1=vePhysicalAFromStored(LOAD_STORED_AT_Y_OFFSET(fluidDirection));
    const Sym3 a2=vePhysicalAFromStored(LOAD_STORED_AT_Y_OFFSET(2*fluidDirection));
    const Sym3 a3=vePhysicalAFromStored(LOAD_STORED_AT_Y_OFFSET(3*fluidDirection));

    #undef LOAD_STORED_AT_Y_OFFSET
    return veStoredFromPhysicalA(a1*3.0_df-a2*3.0_df+a3);
}
#endif // CONFORMATION_TENSOR

#endif
