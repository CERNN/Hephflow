/**
 * @file curvedBC.cuh
 * @brief Functions for curved boundary conditions.
 */
#ifndef __CURVED_BC_CUH
#define __CURVED_BC_CUH

#include <builtin_types.h>

#include "var.h"
#include "globalStructs.h"
#include "globalFunctions.h"

#ifdef CURVED_BOUNDARY_CONDITION

/** @brief Extrapolates a field value from a curved wall and two fluid points. */
__host__ __device__
dfloat curvedBoundaryExtrapolation(dfloat delta, dfloat deltaR, dfloat wallValue, dfloat firstFluidValue, dfloat secondFluidValue);

/** @brief Updates the interpolated velocity stored for each curved-boundary node. */
__global__
void updateCurvedBoundaryVelocities(CurvedBoundary* curvedBoundaryArray, dfloat* moments, unsigned int numberCurvedBoundaryNodes);

#if defined(CONFORMATION_TENSOR) && defined(D3G19)
/** @brief Updates conformation data stored for each curved-boundary node. */
__global__
void updateCurvedBoundaryConformation(CurvedBoundary* curvedBoundaryArray, dfloat* moments, unsigned int numberCurvedBoundaryNodes);
#endif // defined(CONFORMATION_TENSOR) && defined(D3G19)

#endif // CURVED_BOUNDARY_CONDITION
#endif // __CURVED_BC_CUH
