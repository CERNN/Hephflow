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
dfloat curvedBoundaryExtrapolation(dfloat delta, dfloat pf1_value, dfloat pf2_value) {
	const dfloat delta_r = sqrt(2.0); //TODO: CURRENTLY A MAGIC NUMBNER

	return (delta * (delta - 2.0 * delta_r) / (delta_r * delta_r)) * pf1_value - (delta * (delta - delta_r) / (2.0 * delta_r * delta_r)) * pf2_value;
}

__device__ inline 
void curvedBoundaryInterpExtrapStore(
    dfloat delta,
    dfloat3 pf1, 
    dfloat3 pf2,
    int tx, int ty, int tz,
    int bx, int by, int bz,
    dfloat *fMom,
    CurvedBoundary* tempCBC)
{

    dfloat val1;
    dfloat val2;

    val1 = mom_trilinear_interp(pf1.x, pf1.y, pf1.z, M_UX_INDEX, fMom);
    val2 = mom_trilinear_interp(pf2.x, pf2.y, pf2.z, M_UX_INDEX, fMom);

    dfloat ux_e = curvedBoundaryExtrapolation(delta, val1, val2);

    val1 = mom_trilinear_interp(pf1.x, pf1.y, pf1.z, M_UY_INDEX, fMom);
    val2 = mom_trilinear_interp(pf2.x, pf2.y, pf2.z, M_UY_INDEX, fMom);

    dfloat uy_e = curvedBoundaryExtrapolation(delta, val1, val2);

    val1 = mom_trilinear_interp(pf1.x, pf1.y, pf1.z, M_UZ_INDEX, fMom);
    val2 = mom_trilinear_interp(pf2.x, pf2.y, pf2.z, M_UZ_INDEX, fMom);

    dfloat uz_e = curvedBoundaryExtrapolation(delta, val1, val2);

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

    const int xb = tempCBC->b.x;
    const int yb = tempCBC->b.y;
    const int zb = tempCBC->b.z;

    const int tx = xb % BLOCK_NX;
    const int ty = yb % BLOCK_NY;
    const int tz = zb % BLOCK_NZ;
    const int bx = xb / BLOCK_NX;
    const int by = yb / BLOCK_NY;
    const int bz = zb / BLOCK_NZ;

    const dfloat3 pf1 = tempCBC->pf1; 
    const dfloat3 pf2 = tempCBC->pf2;
    const dfloat delta = tempCBC->delta;

    // Pass the dfloat3 structures to the inline function
    curvedBoundaryInterpExtrapStore(delta, pf1, pf2, tx, ty, tz, bx, by, bz, fMom, tempCBC);
}


#endif //!__CURVED_BC_CUH
#endif //CURVED_BOUNDARY_CONDITION