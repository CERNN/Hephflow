/**
 *  @file curvedBC.cuh
 *  Contributors history:
 *  @author Marco Aurelio Ferrari (e.marcoferrari@utfpr.edu.br)
 *  @brief Functions for curved boundary condition
 *  @version 0.1.0
 *  @date 21/11/2025
 */


#include <builtin_types.h> // for device variables
#include "var.h"
#include "globalStructs.h"
#include "globalFunctions.h"

#ifdef CURVED_BOUNDARY_CONDITION
#ifndef __CURVED_BC_CUH
#define __CURVED_BC_CUH

__host__ __device__
dfloat curvedBoundaryExtrapolation(
    dfloat delta,
    dfloat delta_r,
    dfloat wall_value,
    dfloat pf1_value,
    dfloat pf2_value)
{
    const dfloat inv_delta_r_sq = 1.0_df / (delta_r * delta_r);
    const dfloat wall_weight =
        (2.0_df * delta_r * delta_r - delta * delta + 3.0_df * delta * delta_r)
        * 0.5_df * inv_delta_r_sq;
    const dfloat pf1_weight = delta * (delta - 2.0_df * delta_r) * inv_delta_r_sq;
    const dfloat pf2_weight = -delta * (delta - delta_r) * 0.5_df * inv_delta_r_sq;

    return wall_weight * wall_value + pf1_weight * pf1_value + pf2_weight * pf2_value;
}

__device__ inline
dfloat curvedBC_interp_moment(dfloat3 p, int mom, dfloat* fMom, 
                               bool same_x, bool same_y, bool same_z,
                               int const_x, int const_y, int const_z) {
    if (same_z) {
        // Z is constant: bilinear in XY plane
        return mom_bilinear_interp_xy(p.x, p.y, const_z, mom, fMom);
    } else if (same_y) {
        // Y is constant: bilinear in XZ plane
        return mom_bilinear_interp_xz(p.x, const_y, p.z, mom, fMom);
    } else if (same_x) {
        // X is constant: bilinear in YZ plane
        return mom_bilinear_interp_yz(const_x, p.y, p.z, mom, fMom);
    } else {
        // No constant axis: use trilinear interpolation
        return mom_trilinear_interp(p.x, p.y, p.z, mom, fMom);
    }
}

__device__ inline 
void curvedBoundaryInterpExtrapStore(
    dfloat delta,
    dfloat delta_r,
    dfloat3 pf1, 
    dfloat3 pf2,
    int tx, int ty, int tz,
    int bx, int by, int bz,
    dfloat *fMom,
    CurvedBoundary* tempCBC,
    int idx)
{
    // Check which axes have effectively constant coordinate for pf1 and pf2
    // Use epsilon comparison to determine if bilinear interpolation is appropriate
    // (for axis-aligned ducts, the normal has no component in that axis direction)
    constexpr dfloat EPS = 1e-6_df;
    const bool same_x = (fabs(pf1.x - pf2.x) < EPS);
    const bool same_y = (fabs(pf1.y - pf2.y) < EPS);
    const bool same_z = (fabs(pf1.z - pf2.z) < EPS);
    
    // For bilinear interpolation, use the integer coordinate of the constant axis
    const int const_x = (int)floor(pf1.x);
    const int const_y = (int)floor(pf1.y);
    const int const_z = (int)floor(pf1.z);
    
    dfloat val1, val2;

    // Get scaled velocities from fMom and unscale them
    val1 = curvedBC_interp_moment(pf1, M_UX_INDEX, fMom, same_x, same_y, same_z, const_x, const_y, const_z) / F_M_I_SCALE;
    val2 = curvedBC_interp_moment(pf2, M_UX_INDEX, fMom, same_x, same_y, same_z, const_x, const_y, const_z) / F_M_I_SCALE;
    dfloat ux_e = curvedBoundaryExtrapolation(delta, delta_r, tempCBC->wallVel.x, val1, val2);

    val1 = curvedBC_interp_moment(pf1, M_UY_INDEX, fMom, same_x, same_y, same_z, const_x, const_y, const_z) / F_M_I_SCALE;
    val2 = curvedBC_interp_moment(pf2, M_UY_INDEX, fMom, same_x, same_y, same_z, const_x, const_y, const_z) / F_M_I_SCALE;
    dfloat uy_e = curvedBoundaryExtrapolation(delta, delta_r, tempCBC->wallVel.y, val1, val2);

    val1 = curvedBC_interp_moment(pf1, M_UZ_INDEX, fMom, same_x, same_y, same_z, const_x, const_y, const_z) / F_M_I_SCALE;
    val2 = curvedBC_interp_moment(pf2, M_UZ_INDEX, fMom, same_x, same_y, same_z, const_x, const_y, const_z) / F_M_I_SCALE;
    dfloat uz_e = curvedBoundaryExtrapolation(delta, delta_r, tempCBC->wallVel.z, val1, val2);

    tempCBC->vel = dfloat3(ux_e, uy_e, uz_e);
}


 //can you give a better name for this function?
 __global__
void updateCurvedBoundaryVelocities(
    CurvedBoundary* d_curvedBC_array, 
    dfloat *fMom, 
    unsigned int numberCurvedBoundaryNodes
) {
    const int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (idx >= numberCurvedBoundaryNodes)
        return;
        
    CurvedBoundary* tempCBC = &d_curvedBC_array[idx];
    //FIX: (int) is better than static cast int?
    const int xb = (int)tempCBC->b.x;
    const int yb = (int)tempCBC->b.y;
    const int zb = (int)tempCBC->b.z;

    const int tx = xb % BLOCK_NX;
    const int ty = yb % BLOCK_NY;
    const int tz = zb % BLOCK_NZ;
    const int bx = xb / BLOCK_NX;
    const int by = yb / BLOCK_NY;
    const int bz = zb / BLOCK_NZ;

    const dfloat3 pf1 = tempCBC->pf1; 
    const dfloat3 pf2 = tempCBC->pf2;
    const dfloat delta = tempCBC->delta;
    const dfloat delta_r = tempCBC->delta_r;

    // Perform interpolation, extrapolation, and store result
    curvedBoundaryInterpExtrapStore(delta, delta_r, pf1, pf2, tx, ty, tz, bx, by, bz, fMom, tempCBC, idx);
}

#if defined(CONFORMATION_TENSOR) && defined(D3G19)
__global__
void updateCurvedBoundaryConformation(
    CurvedBoundary* d_curvedBC_array,
    dfloat* fMom,
    unsigned int numberCurvedBoundaryNodes)
{
    const int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= numberCurvedBoundaryNodes) return;

    CurvedBoundary* curvedBC = &d_curvedBC_array[idx];
#ifndef CURVED_CONF_COUPLED_MOMENT_BC
    const dfloat delta = curvedBC->delta;
    const dfloat dr = curvedBC->delta_r;
    const dfloat invTwoDrSq = 0.5_df / (dr * dr);
    const dfloat w1 = (delta - 2.0_df * dr) * (delta - 3.0_df * dr) * invTwoDrSq;
    const dfloat w2 = -(delta - dr) * (delta - 3.0_df * dr) / (dr * dr);
    const dfloat w3 = (delta - dr) * (delta - 2.0_df * dr) * invTwoDrSq;
#endif
    const int z = (int)floor(curvedBC->pf1.z);
    const int scalarMoments[6] = {
        A_XX_C_INDEX, A_XY_C_INDEX, A_XZ_C_INDEX,
        A_YY_C_INDEX, A_YZ_C_INDEX, A_ZZ_C_INDEX
    };
#ifdef CURVED_CONF_COUPLED_MOMENT_BC
    const int fluxMoments[18] = {
        A_XX_CX_INDEX, A_XX_CY_INDEX, A_XX_CZ_INDEX,
        A_XY_CX_INDEX, A_XY_CY_INDEX, A_XY_CZ_INDEX,
        A_XZ_CX_INDEX, A_XZ_CY_INDEX, A_XZ_CZ_INDEX,
        A_YY_CX_INDEX, A_YY_CY_INDEX, A_YY_CZ_INDEX,
        A_YZ_CX_INDEX, A_YZ_CY_INDEX, A_YZ_CZ_INDEX,
        A_ZZ_CX_INDEX, A_ZZ_CY_INDEX, A_ZZ_CZ_INDEX
    };

    curvedBC->conformationInteriorVelocityT30[0] = mom_bilinear_interp_xy(
        curvedBC->pf1.x, curvedBC->pf1.y, z, M_UX_INDEX, fMom);
    curvedBC->conformationInteriorVelocityT30[1] = mom_bilinear_interp_xy(
        curvedBC->pf1.x, curvedBC->pf1.y, z, M_UY_INDEX, fMom);
    curvedBC->conformationInteriorVelocityT30[2] = mom_bilinear_interp_xy(
        curvedBC->pf1.x, curvedBC->pf1.y, z, M_UZ_INDEX, fMom);
#endif

    #pragma unroll
    for (int component = 0; component < 6; ++component) {
        const int moment = scalarMoments[component];
        const dfloat a1 = mom_bilinear_interp_xy(
            curvedBC->pf1.x, curvedBC->pf1.y, z, moment, fMom);
#ifdef CURVED_CONF_COUPLED_MOMENT_BC
        // A nearest-interior, bilinearly interpolated tensor is a convex
        // combination of interior tensors and therefore does not introduce
        // the negative-weight overshoot of quadratic extrapolation.
        curvedBC->conformation[component] = a1;
#else
        const dfloat a2 = mom_bilinear_interp_xy(
            curvedBC->pf2.x, curvedBC->pf2.y, z, moment, fMom);
        const dfloat a3 = mom_bilinear_interp_xy(
            curvedBC->pf3.x, curvedBC->pf3.y, z, moment, fMom);
        curvedBC->conformation[component] = w1 * a1 + w2 * a2 + w3 * a3;
#endif

#ifdef CURVED_CONF_COUPLED_MOMENT_BC
        #pragma unroll
        for (int direction = 0; direction < 3; ++direction) {
            curvedBC->conformationFluxT30[3 * component + direction] =
                mom_bilinear_interp_xy(
                    curvedBC->pf1.x, curvedBC->pf1.y, z,
                    fluxMoments[3 * component + direction], fMom);
        }
#endif
    }
}
#endif


#endif //!__CURVED_BC_CUH
#endif //CURVED_BOUNDARY_CONDITION
