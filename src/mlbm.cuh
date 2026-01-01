/**
 *  @file mlbm.h
 *  Contributors history:
 *  @author Marco Aurelio Ferrari (e.marcoferrari@utfpr.edu.br)
 *  @brief main kernel for moment representation 
 *  @version 0.1.0
 *  @date 01/09/2025
 */

#ifndef __MLBM_H
#define __MLBM_H

#include <string>
#include <math.h>
#include <cuda.h>
#include <curand.h>
#include <cuda_runtime.h>
#include <builtin_types.h>
#include "globalFunctions.h"
#include "var.h"
#include "include/interface.h"
#include "nodeTypeMap.h"
#ifdef OMEGA_FIELD
    #include "include/nnf.h"
#endif //OMEGA_FIELD


/**
 *  @brief Updates macroscopics and then performs collision and streaming
 *  @param params: Structure containing all kernel parameters
 *             - fMom: Pointer to the device array containing the current macroscopic moments.
 *             - ghostInterface: interface block transfer information
 *             - d_mean_rho: mean density, used for density correction
 *             - d_BC_Fx/Fy/Fz: boundary condition forces
 *             - step: Current time step
 *             - save: if is necessary save some data
 *             - d_curvedBC/d_curvedBC_array: curved boundary condition data
 */
__global__
void gpuMomCollisionStream(DeviceKernelParams params);

#ifdef LOCAL_FORCES
/**
 * @brief Resets the macroscopic forces to the predefined values FX, FY, FZ
 * @param fMom Pointer to the device array containing the current macroscopic moments.
 */
__global__
void gpuResetMacroForces(dfloat *fMom);
#endif //LOCAL_FORCES


/**
 * @brief Compute phase field gradients and normals, store to global memory
 * @param fMom Pointer to the device array containing the current macroscopic moments.
 * @param dNodeType Pointer to the device array containing the node type information.
 */
__global__ void gpuComputePhaseNormals(
    dfloat *fMom, 
    unsigned int *dNodeType
);


#endif //__MLBM_H