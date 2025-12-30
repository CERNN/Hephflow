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

const int curvedBCBlockSize = 256;
const int curvedBCGridSize = (numberCurvedBoundaryNodes + curvedBCBlockSize - 1) / curvedBCBlockSize;

#ifndef __CURVED_BC_CUH
#define __CURVED_BC_CUH

__host__ __device__
dfloat curvedBoundaryExtrapolation(dfloat delta, dfloat delta_r, dfloat pf1_value, dfloat pf2_value) {
	return (delta * (delta - 2.0_df * delta_r) / (delta_r * delta_r)) * pf1_value 
	     - (delta * (delta - delta_r) / (2.0_df * delta_r * delta_r)) * pf2_value;
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
    dfloat ux_e = curvedBoundaryExtrapolation(delta, delta_r, val1, val2);

    val1 = curvedBC_interp_moment(pf1, M_UY_INDEX, fMom, same_x, same_y, same_z, const_x, const_y, const_z) / F_M_I_SCALE;
    val2 = curvedBC_interp_moment(pf2, M_UY_INDEX, fMom, same_x, same_y, same_z, const_x, const_y, const_z) / F_M_I_SCALE;
    dfloat uy_e = curvedBoundaryExtrapolation(delta, delta_r, val1, val2);

    val1 = curvedBC_interp_moment(pf1, M_UZ_INDEX, fMom, same_x, same_y, same_z, const_x, const_y, const_z) / F_M_I_SCALE;
    val2 = curvedBC_interp_moment(pf2, M_UZ_INDEX, fMom, same_x, same_y, same_z, const_x, const_y, const_z) / F_M_I_SCALE;
    dfloat uz_e = curvedBoundaryExtrapolation(delta, delta_r, val1, val2);

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


#endif //!__CURVED_BC_CUH
#endif //CURVED_BOUNDARY_CONDITION