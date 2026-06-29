/**
 *  @file main.cuh
 *  Contributors history:
 *  @author Waine Jr. (waine@alunos.utfpr.edu.br)
 *  @author Marco Aurelio Ferrari (e.marcoferrari@utfpr.edu.br)
 *  @author Ricardo de Souza
 *  @brief Main routine
 *  @version 0.4.0
 *  @date 01/09/2025
 */

// main.cuh
#ifndef MAIN_CUH
#define MAIN_CUH

#include <stdio.h>
#include <stdlib.h>

// CUDA INCLUDE
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// FILE INCLUDES
#include "var.h"
#include "globalStructs.h"
#include "auxFunctions.cuh"
#include "treatData.cuh"

#ifdef PARTICLE_MODEL
    #include "./particles/class/Particle.cuh"
    #include "./particles/utils/particlesReport.cuh"
    #include "./particles/models/particleSim.cuh"
    #include "./particles/models/dem/collision/collisionDetection.cuh"
    #include "./particles/models/dem/particleMovement.cuh"
#endif //PARTICLE_MODEL

#ifdef OMEGA_FIELD
    #include "non_newtonian/nnf.h"
#endif //OMEGA_FIELD

#include "include/errorDef.h"
#include "include/cuda_utils.cuh"
//#include "structs.h"
//#include "globalFunctions.h"
#include "lbmInitialization.cuh"
#include "mlbm.cuh"
#include "saveData.cuh"
#include "checkpoint.cuh"

#ifdef CURVED_BOUNDARY_CONDITION
    #include "curvedBC.cuh"
#endif

//TODO: maybe move to inside deviceField?
/**
 *  @brief Swaps the pointers of two dfloat variables.
 *  @param pt1: reference to the first dfloat pointer to be swapped
 *  @param pt2: reference to the second dfloat pointer to be swapped
 */
__host__ __device__
void interfaceSwap(dfloat* &pt1, dfloat* &pt2) {
    dfloat *temp = pt1;
    pt1 = pt2;
    pt2 = temp;
}

/**
 *  @brief Frees the memory allocated for the ghost interface data.
 *  @param ghostInterface: reference to the ghost interface data structure
 */
__host__
void interfaceFree(ghostInterfaceData &ghostInterface)
{
    cudaFree(ghostInterface.pop.X_0);
    cudaFree(ghostInterface.pop.X_1);
    cudaFree(ghostInterface.pop.Y_0);
    cudaFree(ghostInterface.pop.Y_1);
    cudaFree(ghostInterface.pop.Z_0);
    cudaFree(ghostInterface.pop.Z_1);

    #ifdef SECOND_DIST
        cudaFree(ghostInterface.g.X_0);
        cudaFree(ghostInterface.g.X_1);
        cudaFree(ghostInterface.g.Y_0);
        cudaFree(ghostInterface.g.Y_1);
        cudaFree(ghostInterface.g.Z_0);
        cudaFree(ghostInterface.g.Z_1);

    #endif //SECOND_DIST
    #ifdef PHI_DIST
        cudaFree(ghostInterface.phi.X_0);
        cudaFree(ghostInterface.phi.X_1);
        cudaFree(ghostInterface.phi.Y_0);
        cudaFree(ghostInterface.phi.Y_1);
        cudaFree(ghostInterface.phi.Z_0);
        cudaFree(ghostInterface.phi.Z_1);

    #endif //PHI_DIST
    #ifdef LAMBDA_DIST
        cudaFree(ghostInterface.lambda.X_0);
        cudaFree(ghostInterface.lambda.X_1);
        cudaFree(ghostInterface.lambda.Y_0);
        cudaFree(ghostInterface.lambda.Y_1);
        cudaFree(ghostInterface.lambda.Z_0);
        cudaFree(ghostInterface.lambda.Z_1);

    #endif //LAMBDA_DIST
    #ifdef A_XX_DIST
        cudaFree(ghostInterface.Axx.X_0);
        cudaFree(ghostInterface.Axx.X_1);
        cudaFree(ghostInterface.Axx.Y_0);
        cudaFree(ghostInterface.Axx.Y_1);
        cudaFree(ghostInterface.Axx.Z_0);
        cudaFree(ghostInterface.Axx.Z_1);

    #endif //A_XX_DIST
    #ifdef A_XY_DIST
        cudaFree(ghostInterface.Axy.X_0);
        cudaFree(ghostInterface.Axy.X_1);
        cudaFree(ghostInterface.Axy.Y_0);
        cudaFree(ghostInterface.Axy.Y_1);
        cudaFree(ghostInterface.Axy.Z_0);
        cudaFree(ghostInterface.Axy.Z_1);

    #endif //A_XY_DIST
    #ifdef A_XZ_DIST
        cudaFree(ghostInterface.Axz.X_0);
        cudaFree(ghostInterface.Axz.X_1);
        cudaFree(ghostInterface.Axz.Y_0);
        cudaFree(ghostInterface.Axz.Y_1);
        cudaFree(ghostInterface.Axz.Z_0);
        cudaFree(ghostInterface.Axz.Z_1);

    #endif //A_XZ_DIST
    #ifdef A_YY_DIST
        cudaFree(ghostInterface.Ayy.X_0);
        cudaFree(ghostInterface.Ayy.X_1);
        cudaFree(ghostInterface.Ayy.Y_0);
        cudaFree(ghostInterface.Ayy.Y_1);
        cudaFree(ghostInterface.Ayy.Z_0);
        cudaFree(ghostInterface.Ayy.Z_1);

    #endif //A_YY_DIST
    #ifdef A_YZ_DIST
        cudaFree(ghostInterface.Ayz.X_0);
        cudaFree(ghostInterface.Ayz.X_1);
        cudaFree(ghostInterface.Ayz.Y_0);
        cudaFree(ghostInterface.Ayz.Y_1);
        cudaFree(ghostInterface.Ayz.Z_0);
        cudaFree(ghostInterface.Ayz.Z_1);

    #endif //A_YZ_DIST
    #ifdef A_ZZ_DIST
        cudaFree(ghostInterface.Azz.X_0);
        cudaFree(ghostInterface.Azz.X_1);
        cudaFree(ghostInterface.Azz.Y_0);
        cudaFree(ghostInterface.Azz.Y_1);
        cudaFree(ghostInterface.Azz.Z_0);
        cudaFree(ghostInterface.Azz.Z_1);

    #endif //A_ZZ_DIST

    if (LOAD_CHECKPOINT){
        cudaFree(ghostInterface.h_pop.X_0);
        cudaFree(ghostInterface.h_pop.X_1);
        cudaFree(ghostInterface.h_pop.Y_0);
        cudaFree(ghostInterface.h_pop.Y_1);
        cudaFree(ghostInterface.h_pop.Z_0);
        cudaFree(ghostInterface.h_pop.Z_1);
        #ifdef SECOND_DIST
            cudaFree(ghostInterface.h_g.X_0);
            cudaFree(ghostInterface.h_g.X_1);
            cudaFree(ghostInterface.h_g.Y_0);
            cudaFree(ghostInterface.h_g.Y_1);
            cudaFree(ghostInterface.h_g.Z_0);
            cudaFree(ghostInterface.h_g.Z_1);
        #endif //SECOND_DIST
        #ifdef PHI_DIST
            cudaFree(ghostInterface.h_phi.X_0);
            cudaFree(ghostInterface.h_phi.X_1);
            cudaFree(ghostInterface.h_phi.Y_0);
            cudaFree(ghostInterface.h_phi.Y_1);
            cudaFree(ghostInterface.h_phi.Z_0);
            cudaFree(ghostInterface.h_phi.Z_1);
        #endif //PHI_DIST
        #ifdef LAMBDA_DIST
            cudaFree(ghostInterface.h_lambda.X_0);
            cudaFree(ghostInterface.h_lambda.X_1);
            cudaFree(ghostInterface.h_lambda.Y_0);
            cudaFree(ghostInterface.h_lambda.Y_1);
            cudaFree(ghostInterface.h_lambda.Z_0);
            cudaFree(ghostInterface.h_lambda.Z_1);
        #endif //LAMBDA_DIST
        #ifdef A_XX_DIST
            cudaFree(ghostInterface.h_Axx.X_0);
            cudaFree(ghostInterface.h_Axx.X_1);
            cudaFree(ghostInterface.h_Axx.Y_0);
            cudaFree(ghostInterface.h_Axx.Y_1);
            cudaFree(ghostInterface.h_Axx.Z_0);
            cudaFree(ghostInterface.h_Axx.Z_1);
        #endif //A_XX_DIST
        #ifdef A_XY_DIST
            cudaFree(ghostInterface.h_Axy.X_0);
            cudaFree(ghostInterface.h_Axy.X_1);
            cudaFree(ghostInterface.h_Axy.Y_0);
            cudaFree(ghostInterface.h_Axy.Y_1);
            cudaFree(ghostInterface.h_Axy.Z_0);
            cudaFree(ghostInterface.h_Axy.Z_1);
        #endif //A_XY_DIST
        #ifdef A_XZ_DIST
            cudaFree(ghostInterface.h_Axz.X_0);
            cudaFree(ghostInterface.h_Axz.X_1);
            cudaFree(ghostInterface.h_Axz.Y_0);
            cudaFree(ghostInterface.h_Axz.Y_1);
            cudaFree(ghostInterface.h_Axz.Z_0);
            cudaFree(ghostInterface.h_Axz.Z_1);
        #endif //A_XZ_DIST
        #ifdef A_YY_DIST
            cudaFree(ghostInterface.h_Ayy.X_0);
            cudaFree(ghostInterface.h_Ayy.X_1);
            cudaFree(ghostInterface.h_Ayy.Y_0);
            cudaFree(ghostInterface.h_Ayy.Y_1);
            cudaFree(ghostInterface.h_Ayy.Z_0);
            cudaFree(ghostInterface.h_Ayy.Z_1);
        #endif //A_YY_DIST
        #ifdef A_YZ_DIST
            cudaFree(ghostInterface.h_Ayz.X_0);
            cudaFree(ghostInterface.h_Ayz.X_1);
            cudaFree(ghostInterface.h_Ayz.Y_0);
            cudaFree(ghostInterface.h_Ayz.Y_1);
            cudaFree(ghostInterface.h_Ayz.Z_0);
            cudaFree(ghostInterface.h_Ayz.Z_1);
        #endif //A_YZ_DIST
        #ifdef A_ZZ_DIST
            cudaFree(ghostInterface.h_Azz.X_0);
            cudaFree(ghostInterface.h_Azz.X_1);
            cudaFree(ghostInterface.h_Azz.Y_0);
            cudaFree(ghostInterface.h_Azz.Y_1);
            cudaFree(ghostInterface.h_Azz.Z_0);
            cudaFree(ghostInterface.h_Azz.Z_1);
        #endif //A_ZZ_DIST

    }
}

/**
 *  @brief Performs a CUDA memory copy for ghost interface data between source and destination.
 *  @param ghostInterface: reference to the ghost interface data structure
 *  @param dst: destination ghost data structure
 *  @param src: source ghost data structure
 *  @param kind: type of memory copy (e.g., cudaMemcpyHostToDevice)
 *  @param Q: number of quantities in the ghost data that are transfered
 */
__host__
void interfaceCudaMemcpy(GhostInterfaceData& ghostInterface, ghostData& dst, const ghostData& src, cudaMemcpyKind kind, int Q, int g) {
    checkCudaErrors(cudaSetDevice(GPUS_TO_USE[g]));
    struct MemcpyPair {
        dfloat* dst;
        const dfloat* src;
        size_t size;
    };

    MemcpyPair memcpyPairs[] = {
        { dst.X_0, src.X_0, sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * Q},
        { dst.X_1, src.X_1, sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * Q},
        { dst.Y_0, src.Y_0, sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * Q},
        { dst.Y_1, src.Y_1, sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * Q},
        { dst.Z_0, src.Z_0, sizeof(dfloat) * NUMBER_GHOST_FACE_XY_LOCAL * Q},
        { dst.Z_1, src.Z_1, sizeof(dfloat) * NUMBER_GHOST_FACE_XY_LOCAL * Q}
    };

    checkCudaErrors(cudaDeviceSynchronize());
    for (const auto& pair : memcpyPairs) {
        // Skip null pairs (e.g., unused AA ghost buffers)
        if (pair.dst == nullptr || pair.src == nullptr) continue;
        checkCudaErrors(cudaMemcpy(pair.dst, pair.src, pair.size, kind));
    }

}
/**
 *  @brief Swaps the ghost interfaces.
 *  @param ghostInterface: reference to the ghost interface data structure
 */
__host__
void swapGhostInterfaces(GhostInterfaceData& ghostInterface) {
    // Synchronize device before performing swaps
    checkCudaErrors(cudaDeviceSynchronize());

    #ifdef SECOND_DIST
    #endif //SECOND_DIST

    #ifdef PHI_DIST
    #endif //PHI_DIST

    #ifdef LAMBDA_DIST
    #endif //LAMBDA_DIST

    #ifdef A_XX_DIST
    #endif //A_XX_DIST

    #ifdef A_XY_DIST
    #endif //A_XY_DIST

    #ifdef A_XZ_DIST
    #endif //A_XZ_DIST

    #ifdef A_YY_DIST
    #endif //A_YY_DIST

    #ifdef A_YZ_DIST
    #endif //A_YZ_DIST

    #ifdef A_ZZ_DIST
    #endif //A_ZZ_DIST
}

/**
 *  @brief Allocates memory for the ghost interface data.
 *  @param ghostInterface: reference to the ghost interface data structure
 */
__host__
void interfaceMalloc(ghostInterfaceData &ghostInterface)
{
    unsigned int memAllocated = 0;

    cudaMalloc((void **)&(ghostInterface.pop.X_0), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * QF);
    cudaMalloc((void **)&(ghostInterface.pop.X_1), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * QF);
    cudaMalloc((void **)&(ghostInterface.pop.Y_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * QF);
    cudaMalloc((void **)&(ghostInterface.pop.Y_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * QF);
    cudaMalloc((void **)&(ghostInterface.pop.Z_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XY_LOCAL * QF);
    cudaMalloc((void **)&(ghostInterface.pop.Z_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XY_LOCAL * QF);
    cudaMalloc((void **)&(ghostInterface.popAux.Z_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XY_AUX * QF);
    cudaMalloc((void **)&(ghostInterface.popAux.Z_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XY_AUX * QF);

    memAllocated = QF * (NUMBER_GHOST_FACE_YZ + NUMBER_GHOST_FACE_XZ + NUMBER_GHOST_FACE_XY_LOCAL) * sizeof(dfloat);  // AA: ghost only

#ifdef SECOND_DIST
    cudaMalloc((void **)&(ghostInterface.g.X_0), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF);
    cudaMalloc((void **)&(ghostInterface.g.X_1), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF);
    cudaMalloc((void **)&(ghostInterface.g.Y_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF);
    cudaMalloc((void **)&(ghostInterface.g.Y_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF);
    cudaMalloc((void **)&(ghostInterface.g.Z_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XY_LOCAL * GF);
    cudaMalloc((void **)&(ghostInterface.g.Z_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XY_LOCAL * GF);

    memAllocated += 2 * GF * (NUMBER_GHOST_FACE_YZ + NUMBER_GHOST_FACE_XZ + NUMBER_GHOST_FACE_XY_LOCAL) * sizeof(dfloat);
#endif //SECOND_DIST

#ifdef PHI_DIST
    cudaMalloc((void **)&(ghostInterface.phi.X_0), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF);
    cudaMalloc((void **)&(ghostInterface.phi.X_1), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF);
    cudaMalloc((void **)&(ghostInterface.phi.Y_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF);
    cudaMalloc((void **)&(ghostInterface.phi.Y_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF);
    cudaMalloc((void **)&(ghostInterface.phi.Z_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XY_LOCAL * GF);
    cudaMalloc((void **)&(ghostInterface.phi.Z_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XY_LOCAL * GF);

    memAllocated += 2 * GF * (NUMBER_GHOST_FACE_YZ + NUMBER_GHOST_FACE_XZ + NUMBER_GHOST_FACE_XY_LOCAL) * sizeof(dfloat);
#endif //PHI_DIST

#ifdef LAMBDA_DIST
    cudaMalloc((void **)&(ghostInterface.lambda.X_0), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF);
    cudaMalloc((void **)&(ghostInterface.lambda.X_1), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF);
    cudaMalloc((void **)&(ghostInterface.lambda.Y_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF);
    cudaMalloc((void **)&(ghostInterface.lambda.Y_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF);
    cudaMalloc((void **)&(ghostInterface.lambda.Z_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XY_LOCAL * GF);
    cudaMalloc((void **)&(ghostInterface.lambda.Z_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XY_LOCAL * GF);

    memAllocated += 2 * GF * (NUMBER_GHOST_FACE_YZ + NUMBER_GHOST_FACE_XZ + NUMBER_GHOST_FACE_XY_LOCAL) * sizeof(dfloat);
#endif //LAMBDA_DIST

#ifdef A_XX_DIST
    cudaMalloc((void **)&(ghostInterface.Axx.X_0), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF);
    cudaMalloc((void **)&(ghostInterface.Axx.X_1), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF);
    cudaMalloc((void **)&(ghostInterface.Axx.Y_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF);
    cudaMalloc((void **)&(ghostInterface.Axx.Y_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF);
    cudaMalloc((void **)&(ghostInterface.Axx.Z_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XY_LOCAL * GF);
    cudaMalloc((void **)&(ghostInterface.Axx.Z_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XY_LOCAL * GF);

    memAllocated += 2 * GF * (NUMBER_GHOST_FACE_YZ + NUMBER_GHOST_FACE_XZ + NUMBER_GHOST_FACE_XY_LOCAL) * sizeof(dfloat);
#endif //A_XX_DIST

#ifdef A_XY_DIST
    cudaMalloc((void **)&(ghostInterface.Axy.X_0), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF);
    cudaMalloc((void **)&(ghostInterface.Axy.X_1), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF);
    cudaMalloc((void **)&(ghostInterface.Axy.Y_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF);
    cudaMalloc((void **)&(ghostInterface.Axy.Y_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF);
    cudaMalloc((void **)&(ghostInterface.Axy.Z_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XY_LOCAL * GF);
    cudaMalloc((void **)&(ghostInterface.Axy.Z_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XY_LOCAL * GF);

    memAllocated += 2 * GF * (NUMBER_GHOST_FACE_YZ + NUMBER_GHOST_FACE_XZ + NUMBER_GHOST_FACE_XY_LOCAL) * sizeof(dfloat);
#endif //A_XY_DIST

#ifdef A_XZ_DIST
    cudaMalloc((void **)&(ghostInterface.Axz.X_0), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF);
    cudaMalloc((void **)&(ghostInterface.Axz.X_1), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF);
    cudaMalloc((void **)&(ghostInterface.Axz.Y_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF);
    cudaMalloc((void **)&(ghostInterface.Axz.Y_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF);
    cudaMalloc((void **)&(ghostInterface.Axz.Z_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XY_LOCAL * GF);
    cudaMalloc((void **)&(ghostInterface.Axz.Z_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XY_LOCAL * GF);

    memAllocated += 2 * GF * (NUMBER_GHOST_FACE_YZ + NUMBER_GHOST_FACE_XZ + NUMBER_GHOST_FACE_XY_LOCAL) * sizeof(dfloat);
#endif //A_XZ_DIST

#ifdef A_YY_DIST
    cudaMalloc((void **)&(ghostInterface.Ayy.X_0), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF);
    cudaMalloc((void **)&(ghostInterface.Ayy.X_1), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF);
    cudaMalloc((void **)&(ghostInterface.Ayy.Y_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF);
    cudaMalloc((void **)&(ghostInterface.Ayy.Y_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF);
    cudaMalloc((void **)&(ghostInterface.Ayy.Z_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XY_LOCAL * GF);
    cudaMalloc((void **)&(ghostInterface.Ayy.Z_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XY_LOCAL * GF);

    memAllocated += 2 * GF * (NUMBER_GHOST_FACE_YZ + NUMBER_GHOST_FACE_XZ + NUMBER_GHOST_FACE_XY_LOCAL) * sizeof(dfloat);
#endif //A_YY_DIST

#ifdef A_YZ_DIST
    cudaMalloc((void **)&(ghostInterface.Ayz.X_0), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF);
    cudaMalloc((void **)&(ghostInterface.Ayz.X_1), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF);
    cudaMalloc((void **)&(ghostInterface.Ayz.Y_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF);
    cudaMalloc((void **)&(ghostInterface.Ayz.Y_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF);
    cudaMalloc((void **)&(ghostInterface.Ayz.Z_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XY_LOCAL * GF);
    cudaMalloc((void **)&(ghostInterface.Ayz.Z_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XY_LOCAL * GF);

    memAllocated += 2 * GF * (NUMBER_GHOST_FACE_YZ + NUMBER_GHOST_FACE_XZ + NUMBER_GHOST_FACE_XY_LOCAL) * sizeof(dfloat);
#endif //A_YZ_DIST

#ifdef A_ZZ_DIST
    cudaMalloc((void **)&(ghostInterface.Azz.X_0), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF);
    cudaMalloc((void **)&(ghostInterface.Azz.X_1), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF);
    cudaMalloc((void **)&(ghostInterface.Azz.Y_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF);
    cudaMalloc((void **)&(ghostInterface.Azz.Y_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF);
    cudaMalloc((void **)&(ghostInterface.Azz.Z_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XY_LOCAL * GF);
    cudaMalloc((void **)&(ghostInterface.Azz.Z_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XY_LOCAL * GF);

    memAllocated += 2 * GF * (NUMBER_GHOST_FACE_YZ + NUMBER_GHOST_FACE_XZ + NUMBER_GHOST_FACE_XY_LOCAL) * sizeof(dfloat);
#endif //A_ZZ_DIST

    if (LOAD_CHECKPOINT || CHECKPOINT_SAVE)
    {
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.h_pop.X_0), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * QF));
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.h_pop.X_1), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * QF));
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.h_pop.Y_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * QF));
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.h_pop.Y_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * QF));
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.h_pop.Z_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XY_LOCAL * QF));
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.h_pop.Z_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XY_LOCAL * QF));

        memAllocated += QF * (NUMBER_GHOST_FACE_YZ + NUMBER_GHOST_FACE_XZ + NUMBER_GHOST_FACE_XY_LOCAL) * sizeof(dfloat);

        #ifdef SECOND_DIST
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.h_g.X_0), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF));
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.h_g.X_1), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF));
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.h_g.Y_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF));
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.h_g.Y_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF));
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.h_g.Z_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XY_LOCAL * GF));
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.h_g.Z_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XY_LOCAL * GF));

        memAllocated += GF * (NUMBER_GHOST_FACE_YZ + NUMBER_GHOST_FACE_XZ + NUMBER_GHOST_FACE_XY_LOCAL) * sizeof(dfloat);
        #endif //SECOND_DIST

        #ifdef PHI_DIST
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.h_phi.X_0), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF));
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.h_phi.X_1), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF));
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.h_phi.Y_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF));
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.h_phi.Y_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF));
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.h_phi.Z_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XY_LOCAL * GF));
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.h_phi.Z_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XY_LOCAL * GF));

        memAllocated += GF * (NUMBER_GHOST_FACE_YZ + NUMBER_GHOST_FACE_XZ + NUMBER_GHOST_FACE_XY_LOCAL) * sizeof(dfloat);
        #endif //PHI_DIST

        #ifdef A_XX_DIST
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.h_Axx.X_0), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF));
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.h_Axx.X_1), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF));
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.h_Axx.Y_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF));
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.h_Axx.Y_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF));
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.h_Axx.Z_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XY_LOCAL * GF));
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.h_Axx.Z_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XY_LOCAL * GF));

        memAllocated += GF * (NUMBER_GHOST_FACE_YZ + NUMBER_GHOST_FACE_XZ + NUMBER_GHOST_FACE_XY_LOCAL) * sizeof(dfloat);
        #endif //A_XX_DIST
        #ifdef A_XY_DIST
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.h_Axy.X_0), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF));
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.h_Axy.X_1), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF));
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.h_Axy.Y_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF));
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.h_Axy.Y_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF));
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.h_Axy.Z_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XY_LOCAL * GF));
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.h_Axy.Z_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XY_LOCAL * GF));

        memAllocated += GF * (NUMBER_GHOST_FACE_YZ + NUMBER_GHOST_FACE_XZ + NUMBER_GHOST_FACE_XY_LOCAL) * sizeof(dfloat);
        #endif //A_XY_DIST
        #ifdef A_XZ_DIST
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.h_Axz.X_0), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF));
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.h_Axz.X_1), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF));
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.h_Axz.Y_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF));
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.h_Axz.Y_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF));
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.h_Axz.Z_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XY_LOCAL * GF));
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.h_Axz.Z_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XY_LOCAL * GF));

        memAllocated += GF * (NUMBER_GHOST_FACE_YZ + NUMBER_GHOST_FACE_XZ + NUMBER_GHOST_FACE_XY_LOCAL) * sizeof(dfloat);
        #endif //A_XZ_DIST
        #ifdef A_YY_DIST
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.h_Ayy.X_0), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF));
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.h_Ayy.X_1), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF));
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.h_Ayy.Y_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF));
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.h_Ayy.Y_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF));
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.h_Ayy.Z_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XY_LOCAL * GF));
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.h_Ayy.Z_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XY_LOCAL * GF));

        memAllocated += GF * (NUMBER_GHOST_FACE_YZ + NUMBER_GHOST_FACE_XZ + NUMBER_GHOST_FACE_XY_LOCAL) * sizeof(dfloat);
        #endif //A_YY_DIST
        #ifdef A_YZ_DIST
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.h_Ayz.X_0), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF));
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.h_Ayz.X_1), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF));
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.h_Ayz.Y_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF));
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.h_Ayz.Y_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF));
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.h_Ayz.Z_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XY_LOCAL * GF));
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.h_Ayz.Z_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XY_LOCAL * GF));

        memAllocated += GF * (NUMBER_GHOST_FACE_YZ + NUMBER_GHOST_FACE_XZ + NUMBER_GHOST_FACE_XY_LOCAL) * sizeof(dfloat);
        #endif //A_YZ_DIST
        #ifdef A_ZZ_DIST
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.h_Azz.X_0), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF));
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.h_Azz.X_1), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF));
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.h_Azz.Y_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF));
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.h_Azz.Y_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF));
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.h_Azz.Z_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XY_LOCAL * GF));
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.h_Azz.Z_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XY_LOCAL * GF));

        memAllocated += GF * (NUMBER_GHOST_FACE_YZ + NUMBER_GHOST_FACE_XZ + NUMBER_GHOST_FACE_XY_LOCAL) * sizeof(dfloat);
        #endif //A_ZZ_DIST
    }

    printf("Device Memory Allocated for Interface: %.2f MB \n", (float)memAllocated /(1024.0 * 1024.0)); if(console_flush) fflush(stdout);
}

/**
 * @brief Initialize the simulation domain, including random numbers, LBM distributions, node types, and ghost interfaces.
 * @details This function is now inlined in deviceField.cuh:initializeDomainDeviceField()
 */

#endif // MAIN_CUH
