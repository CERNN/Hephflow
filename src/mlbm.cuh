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
#include "non_newtonian/nnf.h"
#include "non_newtonian/conformation_representation.cuh"
#include "fragments/lambdaTransport/lambda_evolution.cuh"
#include "non_newtonian/propertyInterpolation.cuh"
#include "curvedConformationBC.cuh"
#define CCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING
#include <cuda/barrier>


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

#ifdef CONFORMATION_PHYSICAL_A_HALO
/**
 * @brief Reconstructs physical A and its stored representation on CUDA-block faces for the next collision generation.
 *
 * This is intentionally a post-transport kernel: collision reads generation n, transport produces n+1, and this
 * kernel publishes the face values for generation n+1 after every block has finished writing its conformation moments.
 *
 * @param fMom Current device moment array containing the stored conformation representation.
 * @param halo Face buffers for the AB generation being reconstructed.
 * @param localNZ Number of lattice planes owned by this GPU.
 */
__global__ void gpuFillPolymerStressHalo( const dfloat* fMom, PolymerStressHaloPtrs halo, size_t localNZ);
#endif // CONFORMATION_PHYSICAL_A_HALO

#ifdef LOCAL_FORCES
/**
 * @brief Resets the macroscopic forces to the predefined values FX, FY, FZ
 * @param fMom Pointer to the device array containing the current macroscopic moments.
 */
__global__
void gpuResetMacroForces(dfloat *fMom);
#endif //LOCAL_FORCES


/**
 * @brief Packs bottom and top macroscopic Z halos for exchange.
 *
 * Launch with a 1D plane grid and @c grid.y equal to two (bottom/top).
 * @param fMom Device moment storage for the local slab.
 * @param halos Destination macroscopic halo buffers.
 * @param localNZ Number of physical Z planes owned by this GPU.
 */
__global__ void gpuPackMacroHalos(const dfloat *fMom, macroInterfaceGPUData halos, size_t localNZ);

#endif //__MLBM_H
