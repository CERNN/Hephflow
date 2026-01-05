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
    cudaFree(ghostInterface.fGhost.X_0);
    cudaFree(ghostInterface.fGhost.X_1);
    cudaFree(ghostInterface.fGhost.Y_0);
    cudaFree(ghostInterface.fGhost.Y_1);
    cudaFree(ghostInterface.fGhost.Z_0);
    cudaFree(ghostInterface.fGhost.Z_1);

    cudaFree(ghostInterface.gGhost.X_0);
    cudaFree(ghostInterface.gGhost.X_1);
    cudaFree(ghostInterface.gGhost.Y_0);
    cudaFree(ghostInterface.gGhost.Y_1);
    cudaFree(ghostInterface.gGhost.Z_0);
    cudaFree(ghostInterface.gGhost.Z_1);

    #ifdef SECOND_DIST
        cudaFree(ghostInterface.g_fGhost.X_0);
        cudaFree(ghostInterface.g_fGhost.X_1);
        cudaFree(ghostInterface.g_fGhost.Y_0);
        cudaFree(ghostInterface.g_fGhost.Y_1);
        cudaFree(ghostInterface.g_fGhost.Z_0);
        cudaFree(ghostInterface.g_fGhost.Z_1);

        cudaFree(ghostInterface.g_gGhost.X_0);
        cudaFree(ghostInterface.g_gGhost.X_1);
        cudaFree(ghostInterface.g_gGhost.Y_0);
        cudaFree(ghostInterface.g_gGhost.Y_1);
        cudaFree(ghostInterface.g_gGhost.Z_0);
        cudaFree(ghostInterface.g_gGhost.Z_1);
    #endif //SECOND_DIST
    #ifdef PHI_DIST
        cudaFree(ghostInterface.phi_fGhost.X_0);
        cudaFree(ghostInterface.phi_fGhost.X_1);
        cudaFree(ghostInterface.phi_fGhost.Y_0);
        cudaFree(ghostInterface.phi_fGhost.Y_1);
        cudaFree(ghostInterface.phi_fGhost.Z_0);
        cudaFree(ghostInterface.phi_fGhost.Z_1);

        cudaFree(ghostInterface.phi_gGhost.X_0);
        cudaFree(ghostInterface.phi_gGhost.X_1);
        cudaFree(ghostInterface.phi_gGhost.Y_0);
        cudaFree(ghostInterface.phi_gGhost.Y_1);
        cudaFree(ghostInterface.phi_gGhost.Z_0);
        cudaFree(ghostInterface.phi_gGhost.Z_1);
    #endif //PHI_DIST
    #ifdef LAMBDA_DIST
        cudaFree(ghostInterface.lambda_fGhost.X_0);
        cudaFree(ghostInterface.lambda_fGhost.X_1);
        cudaFree(ghostInterface.lambda_fGhost.Y_0);
        cudaFree(ghostInterface.lambda_fGhost.Y_1);
        cudaFree(ghostInterface.lambda_fGhost.Z_0);
        cudaFree(ghostInterface.lambda_fGhost.Z_1);

        cudaFree(ghostInterface.lambda_gGhost.X_0);
        cudaFree(ghostInterface.lambda_gGhost.X_1);
        cudaFree(ghostInterface.lambda_gGhost.Y_0);
        cudaFree(ghostInterface.lambda_gGhost.Y_1);
        cudaFree(ghostInterface.lambda_gGhost.Z_0);
        cudaFree(ghostInterface.lambda_gGhost.Z_1);
    #endif //LAMBDA_DIST
    #ifdef A_XX_DIST
        cudaFree(ghostInterface.Axx_fGhost.X_0);
        cudaFree(ghostInterface.Axx_fGhost.X_1);
        cudaFree(ghostInterface.Axx_fGhost.Y_0);
        cudaFree(ghostInterface.Axx_fGhost.Y_1);
        cudaFree(ghostInterface.Axx_fGhost.Z_0);
        cudaFree(ghostInterface.Axx_fGhost.Z_1);

        cudaFree(ghostInterface.Axx_gGhost.X_0);
        cudaFree(ghostInterface.Axx_gGhost.X_1);
        cudaFree(ghostInterface.Axx_gGhost.Y_0);
        cudaFree(ghostInterface.Axx_gGhost.Y_1);
        cudaFree(ghostInterface.Axx_gGhost.Z_0);
        cudaFree(ghostInterface.Axx_gGhost.Z_1);
    #endif //A_XX_DIST
    #ifdef A_XY_DIST
        cudaFree(ghostInterface.Axy_fGhost.X_0);
        cudaFree(ghostInterface.Axy_fGhost.X_1);
        cudaFree(ghostInterface.Axy_fGhost.Y_0);
        cudaFree(ghostInterface.Axy_fGhost.Y_1);
        cudaFree(ghostInterface.Axy_fGhost.Z_0);
        cudaFree(ghostInterface.Axy_fGhost.Z_1);

        cudaFree(ghostInterface.Axy_gGhost.X_0);
        cudaFree(ghostInterface.Axy_gGhost.X_1);
        cudaFree(ghostInterface.Axy_gGhost.Y_0);
        cudaFree(ghostInterface.Axy_gGhost.Y_1);
        cudaFree(ghostInterface.Axy_gGhost.Z_0);
        cudaFree(ghostInterface.Axy_gGhost.Z_1);
    #endif //A_XY_DIST
    #ifdef A_XZ_DIST
        cudaFree(ghostInterface.Axz_fGhost.X_0);
        cudaFree(ghostInterface.Axz_fGhost.X_1);
        cudaFree(ghostInterface.Axz_fGhost.Y_0);
        cudaFree(ghostInterface.Axz_fGhost.Y_1);
        cudaFree(ghostInterface.Axz_fGhost.Z_0);
        cudaFree(ghostInterface.Axz_fGhost.Z_1);

        cudaFree(ghostInterface.Axz_gGhost.X_0);
        cudaFree(ghostInterface.Axz_gGhost.X_1);
        cudaFree(ghostInterface.Axz_gGhost.Y_0);
        cudaFree(ghostInterface.Axz_gGhost.Y_1);
        cudaFree(ghostInterface.Axz_gGhost.Z_0);
        cudaFree(ghostInterface.Axz_gGhost.Z_1);
    #endif //A_XZ_DIST
    #ifdef A_YY_DIST
        cudaFree(ghostInterface.Ayy_fGhost.X_0);
        cudaFree(ghostInterface.Ayy_fGhost.X_1);
        cudaFree(ghostInterface.Ayy_fGhost.Y_0);
        cudaFree(ghostInterface.Ayy_fGhost.Y_1);
        cudaFree(ghostInterface.Ayy_fGhost.Z_0);
        cudaFree(ghostInterface.Ayy_fGhost.Z_1);

        cudaFree(ghostInterface.Ayy_gGhost.X_0);
        cudaFree(ghostInterface.Ayy_gGhost.X_1);
        cudaFree(ghostInterface.Ayy_gGhost.Y_0);
        cudaFree(ghostInterface.Ayy_gGhost.Y_1);
        cudaFree(ghostInterface.Ayy_gGhost.Z_0);
        cudaFree(ghostInterface.Ayy_gGhost.Z_1);
    #endif //A_YY_DIST
    #ifdef A_YZ_DIST
        cudaFree(ghostInterface.Ayz_fGhost.X_0);
        cudaFree(ghostInterface.Ayz_fGhost.X_1);
        cudaFree(ghostInterface.Ayz_fGhost.Y_0);
        cudaFree(ghostInterface.Ayz_fGhost.Y_1);
        cudaFree(ghostInterface.Ayz_fGhost.Z_0);
        cudaFree(ghostInterface.Ayz_fGhost.Z_1);

        cudaFree(ghostInterface.Ayz_gGhost.X_0);
        cudaFree(ghostInterface.Ayz_gGhost.X_1);
        cudaFree(ghostInterface.Ayz_gGhost.Y_0);
        cudaFree(ghostInterface.Ayz_gGhost.Y_1);
        cudaFree(ghostInterface.Ayz_gGhost.Z_0);
        cudaFree(ghostInterface.Ayz_gGhost.Z_1);
    #endif //A_YZ_DIST
    #ifdef A_ZZ_DIST
        cudaFree(ghostInterface.Azz_fGhost.X_0);
        cudaFree(ghostInterface.Azz_fGhost.X_1);
        cudaFree(ghostInterface.Azz_fGhost.Y_0);
        cudaFree(ghostInterface.Azz_fGhost.Y_1);
        cudaFree(ghostInterface.Azz_fGhost.Z_0);
        cudaFree(ghostInterface.Azz_fGhost.Z_1);

        cudaFree(ghostInterface.Azz_gGhost.X_0);
        cudaFree(ghostInterface.Azz_gGhost.X_1);
        cudaFree(ghostInterface.Azz_gGhost.Y_0);
        cudaFree(ghostInterface.Azz_gGhost.Y_1);
        cudaFree(ghostInterface.Azz_gGhost.Z_0);
        cudaFree(ghostInterface.Azz_gGhost.Z_1);
    #endif //A_ZZ_DIST

    if (LOAD_CHECKPOINT){
        cudaFree(ghostInterface.h_fGhost.X_0);
        cudaFree(ghostInterface.h_fGhost.X_1);
        cudaFree(ghostInterface.h_fGhost.Y_0);
        cudaFree(ghostInterface.h_fGhost.Y_1);
        cudaFree(ghostInterface.h_fGhost.Z_0);
        cudaFree(ghostInterface.h_fGhost.Z_1);
        #ifdef SECOND_DIST
            cudaFree(ghostInterface.g_h_fGhost.X_0);
            cudaFree(ghostInterface.g_h_fGhost.X_1);
            cudaFree(ghostInterface.g_h_fGhost.Y_0);
            cudaFree(ghostInterface.g_h_fGhost.Y_1);
            cudaFree(ghostInterface.g_h_fGhost.Z_0);
            cudaFree(ghostInterface.g_h_fGhost.Z_1);
        #endif //SECOND_DIST
        #ifdef PHI_DIST
            cudaFree(ghostInterface.phi_h_fGhost.X_0);
            cudaFree(ghostInterface.phi_h_fGhost.X_1);
            cudaFree(ghostInterface.phi_h_fGhost.Y_0);
            cudaFree(ghostInterface.phi_h_fGhost.Y_1);
            cudaFree(ghostInterface.phi_h_fGhost.Z_0);
            cudaFree(ghostInterface.phi_h_fGhost.Z_1);
        #endif //PHI_DIST
        #ifdef LAMBDA_DIST
            cudaFree(ghostInterface.lambda_h_fGhost.X_0);
            cudaFree(ghostInterface.lambda_h_fGhost.X_1);
            cudaFree(ghostInterface.lambda_h_fGhost.Y_0);
            cudaFree(ghostInterface.lambda_h_fGhost.Y_1);
            cudaFree(ghostInterface.lambda_h_fGhost.Z_0);
            cudaFree(ghostInterface.lambda_h_fGhost.Z_1);
        #endif //LAMBDA_DIST
        #ifdef A_XX_DIST
            cudaFree(ghostInterface.Axx_h_fGhost.X_0);
            cudaFree(ghostInterface.Axx_h_fGhost.X_1);
            cudaFree(ghostInterface.Axx_h_fGhost.Y_0);
            cudaFree(ghostInterface.Axx_h_fGhost.Y_1);
            cudaFree(ghostInterface.Axx_h_fGhost.Z_0);
            cudaFree(ghostInterface.Axx_h_fGhost.Z_1);
        #endif //A_XX_DIST
        #ifdef A_XY_DIST
            cudaFree(ghostInterface.Axy_h_fGhost.X_0);
            cudaFree(ghostInterface.Axy_h_fGhost.X_1);
            cudaFree(ghostInterface.Axy_h_fGhost.Y_0);
            cudaFree(ghostInterface.Axy_h_fGhost.Y_1);
            cudaFree(ghostInterface.Axy_h_fGhost.Z_0);
            cudaFree(ghostInterface.Axy_h_fGhost.Z_1);
        #endif //A_XY_DIST
        #ifdef A_XZ_DIST
            cudaFree(ghostInterface.Axz_h_fGhost.X_0);
            cudaFree(ghostInterface.Axz_h_fGhost.X_1);
            cudaFree(ghostInterface.Axz_h_fGhost.Y_0);
            cudaFree(ghostInterface.Axz_h_fGhost.Y_1);
            cudaFree(ghostInterface.Axz_h_fGhost.Z_0);
            cudaFree(ghostInterface.Axz_h_fGhost.Z_1);
        #endif //A_XZ_DIST
        #ifdef A_YY_DIST
            cudaFree(ghostInterface.Ayy_h_fGhost.X_0);
            cudaFree(ghostInterface.Ayy_h_fGhost.X_1);
            cudaFree(ghostInterface.Ayy_h_fGhost.Y_0);
            cudaFree(ghostInterface.Ayy_h_fGhost.Y_1);
            cudaFree(ghostInterface.Ayy_h_fGhost.Z_0);
            cudaFree(ghostInterface.Ayy_h_fGhost.Z_1);
        #endif //A_YY_DIST
        #ifdef A_YZ_DIST
            cudaFree(ghostInterface.Ayz_h_fGhost.X_0);
            cudaFree(ghostInterface.Ayz_h_fGhost.X_1);
            cudaFree(ghostInterface.Ayz_h_fGhost.Y_0);
            cudaFree(ghostInterface.Ayz_h_fGhost.Y_1);
            cudaFree(ghostInterface.Ayz_h_fGhost.Z_0);
            cudaFree(ghostInterface.Ayz_h_fGhost.Z_1);
        #endif //A_YZ_DIST
        #ifdef A_ZZ_DIST
            cudaFree(ghostInterface.Azz_h_fGhost.X_0);
            cudaFree(ghostInterface.Azz_h_fGhost.X_1);
            cudaFree(ghostInterface.Azz_h_fGhost.Y_0);
            cudaFree(ghostInterface.Azz_h_fGhost.Y_1);
            cudaFree(ghostInterface.Azz_h_fGhost.Z_0);
            cudaFree(ghostInterface.Azz_h_fGhost.Z_1);
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
void interfaceCudaMemcpy(GhostInterfaceData& ghostInterface, ghostData& dst, const ghostData& src, cudaMemcpyKind kind, int Q) {
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
        { dst.Z_0, src.Z_0, sizeof(dfloat) * NUMBER_GHOST_FACE_XY * Q},
        { dst.Z_1, src.Z_1, sizeof(dfloat) * NUMBER_GHOST_FACE_XY * Q}
    };

    checkCudaErrors(cudaDeviceSynchronize());
    for (const auto& pair : memcpyPairs) {
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

    // Swap interface pointers for fGhost and gGhost
    interfaceSwap(ghostInterface.fGhost.X_0, ghostInterface.gGhost.X_0);
    interfaceSwap(ghostInterface.fGhost.X_1, ghostInterface.gGhost.X_1);
    interfaceSwap(ghostInterface.fGhost.Y_0, ghostInterface.gGhost.Y_0);
    interfaceSwap(ghostInterface.fGhost.Y_1, ghostInterface.gGhost.Y_1);
    interfaceSwap(ghostInterface.fGhost.Z_0, ghostInterface.gGhost.Z_0);
    interfaceSwap(ghostInterface.fGhost.Z_1, ghostInterface.gGhost.Z_1);

    #ifdef SECOND_DIST
    interfaceSwap(ghostInterface.g_fGhost.X_0, ghostInterface.g_gGhost.X_0);
    interfaceSwap(ghostInterface.g_fGhost.X_1, ghostInterface.g_gGhost.X_1);
    interfaceSwap(ghostInterface.g_fGhost.Y_0, ghostInterface.g_gGhost.Y_0);
    interfaceSwap(ghostInterface.g_fGhost.Y_1, ghostInterface.g_gGhost.Y_1);
    interfaceSwap(ghostInterface.g_fGhost.Z_0, ghostInterface.g_gGhost.Z_0);
    interfaceSwap(ghostInterface.g_fGhost.Z_1, ghostInterface.g_gGhost.Z_1);
    #endif //SECOND_DIST

    #ifdef PHI_DIST
    interfaceSwap(ghostInterface.phi_fGhost.X_0, ghostInterface.phi_gGhost.X_0);
    interfaceSwap(ghostInterface.phi_fGhost.X_1, ghostInterface.phi_gGhost.X_1);
    interfaceSwap(ghostInterface.phi_fGhost.Y_0, ghostInterface.phi_gGhost.Y_0);
    interfaceSwap(ghostInterface.phi_fGhost.Y_1, ghostInterface.phi_gGhost.Y_1);
    interfaceSwap(ghostInterface.phi_fGhost.Z_0, ghostInterface.phi_gGhost.Z_0);
    interfaceSwap(ghostInterface.phi_fGhost.Z_1, ghostInterface.phi_gGhost.Z_1);
    #endif //PHI_DIST

    #ifdef LAMBDA_DIST
    interfaceSwap(ghostInterface.lambda_fGhost.X_0, ghostInterface.lambda_gGhost.X_0);
    interfaceSwap(ghostInterface.lambda_fGhost.X_1, ghostInterface.lambda_gGhost.X_1);
    interfaceSwap(ghostInterface.lambda_fGhost.Y_0, ghostInterface.lambda_gGhost.Y_0);
    interfaceSwap(ghostInterface.lambda_fGhost.Y_1, ghostInterface.lambda_gGhost.Y_1);
    interfaceSwap(ghostInterface.lambda_fGhost.Z_0, ghostInterface.lambda_gGhost.Z_0);
    interfaceSwap(ghostInterface.lambda_fGhost.Z_1, ghostInterface.lambda_gGhost.Z_1);
    #endif //LAMBDA_DIST


    #ifdef A_XX_DIST
    interfaceSwap(ghostInterface.Axx_fGhost.X_0, ghostInterface.Axx_gGhost.X_0);
    interfaceSwap(ghostInterface.Axx_fGhost.X_1, ghostInterface.Axx_gGhost.X_1);
    interfaceSwap(ghostInterface.Axx_fGhost.Y_0, ghostInterface.Axx_gGhost.Y_0);
    interfaceSwap(ghostInterface.Axx_fGhost.Y_1, ghostInterface.Axx_gGhost.Y_1);
    interfaceSwap(ghostInterface.Axx_fGhost.Z_0, ghostInterface.Axx_gGhost.Z_0);
    interfaceSwap(ghostInterface.Axx_fGhost.Z_1, ghostInterface.Axx_gGhost.Z_1);
    #endif //A_XX_DIST

    #ifdef A_XY_DIST
    interfaceSwap(ghostInterface.Axy_fGhost.X_0, ghostInterface.Axy_gGhost.X_0);
    interfaceSwap(ghostInterface.Axy_fGhost.X_1, ghostInterface.Axy_gGhost.X_1);
    interfaceSwap(ghostInterface.Axy_fGhost.Y_0, ghostInterface.Axy_gGhost.Y_0);
    interfaceSwap(ghostInterface.Axy_fGhost.Y_1, ghostInterface.Axy_gGhost.Y_1);
    interfaceSwap(ghostInterface.Axy_fGhost.Z_0, ghostInterface.Axy_gGhost.Z_0);
    interfaceSwap(ghostInterface.Axy_fGhost.Z_1, ghostInterface.Axy_gGhost.Z_1);
    #endif //A_XY_DIST

    #ifdef A_XZ_DIST
    interfaceSwap(ghostInterface.Axz_fGhost.X_0, ghostInterface.Axz_gGhost.X_0);
    interfaceSwap(ghostInterface.Axz_fGhost.X_1, ghostInterface.Axz_gGhost.X_1);
    interfaceSwap(ghostInterface.Axz_fGhost.Y_0, ghostInterface.Axz_gGhost.Y_0);
    interfaceSwap(ghostInterface.Axz_fGhost.Y_1, ghostInterface.Axz_gGhost.Y_1);
    interfaceSwap(ghostInterface.Axz_fGhost.Z_0, ghostInterface.Axz_gGhost.Z_0);
    interfaceSwap(ghostInterface.Axz_fGhost.Z_1, ghostInterface.Axz_gGhost.Z_1);
    #endif //A_XZ_DIST

    #ifdef A_YY_DIST
    interfaceSwap(ghostInterface.Ayy_fGhost.X_0, ghostInterface.Ayy_gGhost.X_0);
    interfaceSwap(ghostInterface.Ayy_fGhost.X_1, ghostInterface.Ayy_gGhost.X_1);
    interfaceSwap(ghostInterface.Ayy_fGhost.Y_0, ghostInterface.Ayy_gGhost.Y_0);
    interfaceSwap(ghostInterface.Ayy_fGhost.Y_1, ghostInterface.Ayy_gGhost.Y_1);
    interfaceSwap(ghostInterface.Ayy_fGhost.Z_0, ghostInterface.Ayy_gGhost.Z_0);
    interfaceSwap(ghostInterface.Ayy_fGhost.Z_1, ghostInterface.Ayy_gGhost.Z_1);
    #endif //A_YY_DIST

    #ifdef A_YZ_DIST
    interfaceSwap(ghostInterface.Ayz_fGhost.X_0, ghostInterface.Ayz_gGhost.X_0);
    interfaceSwap(ghostInterface.Ayz_fGhost.X_1, ghostInterface.Ayz_gGhost.X_1);
    interfaceSwap(ghostInterface.Ayz_fGhost.Y_0, ghostInterface.Ayz_gGhost.Y_0);
    interfaceSwap(ghostInterface.Ayz_fGhost.Y_1, ghostInterface.Ayz_gGhost.Y_1);
    interfaceSwap(ghostInterface.Ayz_fGhost.Z_0, ghostInterface.Ayz_gGhost.Z_0);
    interfaceSwap(ghostInterface.Ayz_fGhost.Z_1, ghostInterface.Ayz_gGhost.Z_1);
    #endif //A_YZ_DIST

    #ifdef A_ZZ_DIST
    interfaceSwap(ghostInterface.Azz_fGhost.X_0, ghostInterface.Azz_gGhost.X_0);
    interfaceSwap(ghostInterface.Azz_fGhost.X_1, ghostInterface.Azz_gGhost.X_1);
    interfaceSwap(ghostInterface.Azz_fGhost.Y_0, ghostInterface.Azz_gGhost.Y_0);
    interfaceSwap(ghostInterface.Azz_fGhost.Y_1, ghostInterface.Azz_gGhost.Y_1);
    interfaceSwap(ghostInterface.Azz_fGhost.Z_0, ghostInterface.Azz_gGhost.Z_0);
    interfaceSwap(ghostInterface.Azz_fGhost.Z_1, ghostInterface.Azz_gGhost.Z_1);
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

    cudaMalloc((void **)&(ghostInterface.fGhost.X_0), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * QF);
    cudaMalloc((void **)&(ghostInterface.fGhost.X_1), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * QF);
    cudaMalloc((void **)&(ghostInterface.fGhost.Y_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * QF);
    cudaMalloc((void **)&(ghostInterface.fGhost.Y_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * QF);
    cudaMalloc((void **)&(ghostInterface.fGhost.Z_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XY * QF);
    cudaMalloc((void **)&(ghostInterface.fGhost.Z_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XY * QF);

    cudaMalloc((void **)&(ghostInterface.gGhost.X_0), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * QF);
    cudaMalloc((void **)&(ghostInterface.gGhost.X_1), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * QF);
    cudaMalloc((void **)&(ghostInterface.gGhost.Y_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * QF);
    cudaMalloc((void **)&(ghostInterface.gGhost.Y_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * QF);
    cudaMalloc((void **)&(ghostInterface.gGhost.Z_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XY * QF);
    cudaMalloc((void **)&(ghostInterface.gGhost.Z_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XY * QF);

    memAllocated = 2 * QF * (NUMBER_GHOST_FACE_YZ + NUMBER_GHOST_FACE_XZ + NUMBER_GHOST_FACE_XY) * sizeof(dfloat);

#ifdef SECOND_DIST
    cudaMalloc((void **)&(ghostInterface.g_fGhost.X_0), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF);
    cudaMalloc((void **)&(ghostInterface.g_fGhost.X_1), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF);
    cudaMalloc((void **)&(ghostInterface.g_fGhost.Y_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF);
    cudaMalloc((void **)&(ghostInterface.g_fGhost.Y_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF);
    cudaMalloc((void **)&(ghostInterface.g_fGhost.Z_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XY * GF);
    cudaMalloc((void **)&(ghostInterface.g_fGhost.Z_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XY * GF);

    cudaMalloc((void **)&(ghostInterface.g_gGhost.X_0), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF);
    cudaMalloc((void **)&(ghostInterface.g_gGhost.X_1), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF);
    cudaMalloc((void **)&(ghostInterface.g_gGhost.Y_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF);
    cudaMalloc((void **)&(ghostInterface.g_gGhost.Y_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF);
    cudaMalloc((void **)&(ghostInterface.g_gGhost.Z_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XY * GF);
    cudaMalloc((void **)&(ghostInterface.g_gGhost.Z_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XY * GF);

    memAllocated += 2 * GF * (NUMBER_GHOST_FACE_YZ + NUMBER_GHOST_FACE_XZ + NUMBER_GHOST_FACE_XY) * sizeof(dfloat);
#endif //SECOND_DIST

#ifdef PHI_DIST
    cudaMalloc((void **)&(ghostInterface.phi_fGhost.X_0), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF);
    cudaMalloc((void **)&(ghostInterface.phi_fGhost.X_1), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF);
    cudaMalloc((void **)&(ghostInterface.phi_fGhost.Y_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF);
    cudaMalloc((void **)&(ghostInterface.phi_fGhost.Y_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF);
    cudaMalloc((void **)&(ghostInterface.phi_fGhost.Z_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XY * GF);
    cudaMalloc((void **)&(ghostInterface.phi_fGhost.Z_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XY * GF);

    cudaMalloc((void **)&(ghostInterface.phi_gGhost.X_0), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF);
    cudaMalloc((void **)&(ghostInterface.phi_gGhost.X_1), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF);
    cudaMalloc((void **)&(ghostInterface.phi_gGhost.Y_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF);
    cudaMalloc((void **)&(ghostInterface.phi_gGhost.Y_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF);
    cudaMalloc((void **)&(ghostInterface.phi_gGhost.Z_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XY * GF);
    cudaMalloc((void **)&(ghostInterface.phi_gGhost.Z_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XY * GF);

    memAllocated += 2 * GF * (NUMBER_GHOST_FACE_YZ + NUMBER_GHOST_FACE_XZ + NUMBER_GHOST_FACE_XY) * sizeof(dfloat);
#endif //PHI_DIST

#ifdef LAMBDA_DIST
    cudaMalloc((void **)&(ghostInterface.lambda_fGhost.X_0), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF);
    cudaMalloc((void **)&(ghostInterface.lambda_fGhost.X_1), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF);
    cudaMalloc((void **)&(ghostInterface.lambda_fGhost.Y_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF);
    cudaMalloc((void **)&(ghostInterface.lambda_fGhost.Y_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF);
    cudaMalloc((void **)&(ghostInterface.lambda_fGhost.Z_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XY * GF);
    cudaMalloc((void **)&(ghostInterface.lambda_fGhost.Z_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XY * GF);

    cudaMalloc((void **)&(ghostInterface.lambda_gGhost.X_0), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF);
    cudaMalloc((void **)&(ghostInterface.lambda_gGhost.X_1), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF);
    cudaMalloc((void **)&(ghostInterface.lambda_gGhost.Y_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF);
    cudaMalloc((void **)&(ghostInterface.lambda_gGhost.Y_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF);
    cudaMalloc((void **)&(ghostInterface.lambda_gGhost.Z_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XY * GF);
    cudaMalloc((void **)&(ghostInterface.lambda_gGhost.Z_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XY * GF);

    memAllocated += 2 * GF * (NUMBER_GHOST_FACE_YZ + NUMBER_GHOST_FACE_XZ + NUMBER_GHOST_FACE_XY) * sizeof(dfloat);
#endif //LAMBDA_DIST

#ifdef A_XX_DIST
    cudaMalloc((void **)&(ghostInterface.Axx_fGhost.X_0), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF);
    cudaMalloc((void **)&(ghostInterface.Axx_fGhost.X_1), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF);
    cudaMalloc((void **)&(ghostInterface.Axx_fGhost.Y_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF);
    cudaMalloc((void **)&(ghostInterface.Axx_fGhost.Y_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF);
    cudaMalloc((void **)&(ghostInterface.Axx_fGhost.Z_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XY * GF);
    cudaMalloc((void **)&(ghostInterface.Axx_fGhost.Z_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XY * GF);

    cudaMalloc((void **)&(ghostInterface.Axx_gGhost.X_0), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF);
    cudaMalloc((void **)&(ghostInterface.Axx_gGhost.X_1), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF);
    cudaMalloc((void **)&(ghostInterface.Axx_gGhost.Y_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF);
    cudaMalloc((void **)&(ghostInterface.Axx_gGhost.Y_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF);
    cudaMalloc((void **)&(ghostInterface.Axx_gGhost.Z_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XY * GF);
    cudaMalloc((void **)&(ghostInterface.Axx_gGhost.Z_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XY * GF);

    memAllocated += 2 * GF * (NUMBER_GHOST_FACE_YZ + NUMBER_GHOST_FACE_XZ + NUMBER_GHOST_FACE_XY) * sizeof(dfloat);
#endif //A_XX_DIST

#ifdef A_XY_DIST
    cudaMalloc((void **)&(ghostInterface.Axy_fGhost.X_0), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF);
    cudaMalloc((void **)&(ghostInterface.Axy_fGhost.X_1), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF);
    cudaMalloc((void **)&(ghostInterface.Axy_fGhost.Y_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF);
    cudaMalloc((void **)&(ghostInterface.Axy_fGhost.Y_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF);
    cudaMalloc((void **)&(ghostInterface.Axy_fGhost.Z_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XY * GF);
    cudaMalloc((void **)&(ghostInterface.Axy_fGhost.Z_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XY * GF);

    cudaMalloc((void **)&(ghostInterface.Axy_gGhost.X_0), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF);
    cudaMalloc((void **)&(ghostInterface.Axy_gGhost.X_1), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF);
    cudaMalloc((void **)&(ghostInterface.Axy_gGhost.Y_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF);
    cudaMalloc((void **)&(ghostInterface.Axy_gGhost.Y_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF);
    cudaMalloc((void **)&(ghostInterface.Axy_gGhost.Z_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XY * GF);
    cudaMalloc((void **)&(ghostInterface.Axy_gGhost.Z_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XY * GF);

    memAllocated += 2 * GF * (NUMBER_GHOST_FACE_YZ + NUMBER_GHOST_FACE_XZ + NUMBER_GHOST_FACE_XY) * sizeof(dfloat);
#endif //A_XY_DIST

#ifdef A_XZ_DIST
    cudaMalloc((void **)&(ghostInterface.Axz_fGhost.X_0), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF);
    cudaMalloc((void **)&(ghostInterface.Axz_fGhost.X_1), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF);
    cudaMalloc((void **)&(ghostInterface.Axz_fGhost.Y_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF);
    cudaMalloc((void **)&(ghostInterface.Axz_fGhost.Y_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF);
    cudaMalloc((void **)&(ghostInterface.Axz_fGhost.Z_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XY * GF);
    cudaMalloc((void **)&(ghostInterface.Axz_fGhost.Z_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XY * GF);

    cudaMalloc((void **)&(ghostInterface.Axz_gGhost.X_0), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF);
    cudaMalloc((void **)&(ghostInterface.Axz_gGhost.X_1), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF);
    cudaMalloc((void **)&(ghostInterface.Axz_gGhost.Y_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF);
    cudaMalloc((void **)&(ghostInterface.Axz_gGhost.Y_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF);
    cudaMalloc((void **)&(ghostInterface.Axz_gGhost.Z_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XY * GF);
    cudaMalloc((void **)&(ghostInterface.Axz_gGhost.Z_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XY * GF);

    memAllocated += 2 * GF * (NUMBER_GHOST_FACE_YZ + NUMBER_GHOST_FACE_XZ + NUMBER_GHOST_FACE_XY) * sizeof(dfloat);
#endif //A_XZ_DIST

#ifdef A_YY_DIST
    cudaMalloc((void **)&(ghostInterface.Ayy_fGhost.X_0), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF);
    cudaMalloc((void **)&(ghostInterface.Ayy_fGhost.X_1), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF);
    cudaMalloc((void **)&(ghostInterface.Ayy_fGhost.Y_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF);
    cudaMalloc((void **)&(ghostInterface.Ayy_fGhost.Y_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF);
    cudaMalloc((void **)&(ghostInterface.Ayy_fGhost.Z_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XY * GF);
    cudaMalloc((void **)&(ghostInterface.Ayy_fGhost.Z_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XY * GF);

    cudaMalloc((void **)&(ghostInterface.Ayy_gGhost.X_0), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF);
    cudaMalloc((void **)&(ghostInterface.Ayy_gGhost.X_1), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF);
    cudaMalloc((void **)&(ghostInterface.Ayy_gGhost.Y_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF);
    cudaMalloc((void **)&(ghostInterface.Ayy_gGhost.Y_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF);
    cudaMalloc((void **)&(ghostInterface.Ayy_gGhost.Z_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XY * GF);
    cudaMalloc((void **)&(ghostInterface.Ayy_gGhost.Z_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XY * GF);

    memAllocated += 2 * GF * (NUMBER_GHOST_FACE_YZ + NUMBER_GHOST_FACE_XZ + NUMBER_GHOST_FACE_XY) * sizeof(dfloat);
#endif //A_YY_DIST

#ifdef A_YZ_DIST
    cudaMalloc((void **)&(ghostInterface.Ayz_fGhost.X_0), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF);
    cudaMalloc((void **)&(ghostInterface.Ayz_fGhost.X_1), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF);
    cudaMalloc((void **)&(ghostInterface.Ayz_fGhost.Y_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF);
    cudaMalloc((void **)&(ghostInterface.Ayz_fGhost.Y_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF);
    cudaMalloc((void **)&(ghostInterface.Ayz_fGhost.Z_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XY * GF);
    cudaMalloc((void **)&(ghostInterface.Ayz_fGhost.Z_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XY * GF);

    cudaMalloc((void **)&(ghostInterface.Ayz_gGhost.X_0), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF);
    cudaMalloc((void **)&(ghostInterface.Ayz_gGhost.X_1), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF);
    cudaMalloc((void **)&(ghostInterface.Ayz_gGhost.Y_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF);
    cudaMalloc((void **)&(ghostInterface.Ayz_gGhost.Y_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF);
    cudaMalloc((void **)&(ghostInterface.Ayz_gGhost.Z_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XY * GF);
    cudaMalloc((void **)&(ghostInterface.Ayz_gGhost.Z_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XY * GF);

    memAllocated += 2 * GF * (NUMBER_GHOST_FACE_YZ + NUMBER_GHOST_FACE_XZ + NUMBER_GHOST_FACE_XY) * sizeof(dfloat);
#endif //A_YZ_DIST

#ifdef A_ZZ_DIST
    cudaMalloc((void **)&(ghostInterface.Azz_fGhost.X_0), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF);
    cudaMalloc((void **)&(ghostInterface.Azz_fGhost.X_1), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF);
    cudaMalloc((void **)&(ghostInterface.Azz_fGhost.Y_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF);
    cudaMalloc((void **)&(ghostInterface.Azz_fGhost.Y_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF);
    cudaMalloc((void **)&(ghostInterface.Azz_fGhost.Z_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XY * GF);
    cudaMalloc((void **)&(ghostInterface.Azz_fGhost.Z_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XY * GF);

    cudaMalloc((void **)&(ghostInterface.Azz_gGhost.X_0), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF);
    cudaMalloc((void **)&(ghostInterface.Azz_gGhost.X_1), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF);
    cudaMalloc((void **)&(ghostInterface.Azz_gGhost.Y_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF);
    cudaMalloc((void **)&(ghostInterface.Azz_gGhost.Y_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF);
    cudaMalloc((void **)&(ghostInterface.Azz_gGhost.Z_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XY * GF);
    cudaMalloc((void **)&(ghostInterface.Azz_gGhost.Z_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XY * GF);

    memAllocated += 2 * GF * (NUMBER_GHOST_FACE_YZ + NUMBER_GHOST_FACE_XZ + NUMBER_GHOST_FACE_XY) * sizeof(dfloat);
#endif //A_ZZ_DIST

    if (LOAD_CHECKPOINT || CHECKPOINT_SAVE)
    {
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.h_fGhost.X_0), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * QF));
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.h_fGhost.X_1), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * QF));
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.h_fGhost.Y_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * QF));
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.h_fGhost.Y_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * QF));
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.h_fGhost.Z_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XY * QF));
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.h_fGhost.Z_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XY * QF));

        memAllocated += QF * (NUMBER_GHOST_FACE_YZ + NUMBER_GHOST_FACE_XZ + NUMBER_GHOST_FACE_XY) * sizeof(dfloat);

        #ifdef SECOND_DIST
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.g_h_fGhost.X_0), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF));
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.g_h_fGhost.X_1), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF));
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.g_h_fGhost.Y_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF));
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.g_h_fGhost.Y_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF));
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.g_h_fGhost.Z_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XY * GF));
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.g_h_fGhost.Z_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XY * GF));

        memAllocated += GF * (NUMBER_GHOST_FACE_YZ + NUMBER_GHOST_FACE_XZ + NUMBER_GHOST_FACE_XY) * sizeof(dfloat);
        #endif //SECOND_DIST

        #ifdef PHI_DIST
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.phi_h_fGhost.X_0), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF));
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.phi_h_fGhost.X_1), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF));
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.phi_h_fGhost.Y_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF));
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.phi_h_fGhost.Y_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF));
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.phi_h_fGhost.Z_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XY * GF));
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.phi_h_fGhost.Z_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XY * GF));

        memAllocated += GF * (NUMBER_GHOST_FACE_YZ + NUMBER_GHOST_FACE_XZ + NUMBER_GHOST_FACE_XY) * sizeof(dfloat);
        #endif //PHI_DIST

        #ifdef A_XX_DIST
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.Axx_h_fGhost.X_0), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF));
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.Axx_h_fGhost.X_1), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF));
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.Axx_h_fGhost.Y_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF));
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.Axx_h_fGhost.Y_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF));
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.Axx_h_fGhost.Z_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XY * GF));
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.Axx_h_fGhost.Z_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XY * GF));

        memAllocated += GF * (NUMBER_GHOST_FACE_YZ + NUMBER_GHOST_FACE_XZ + NUMBER_GHOST_FACE_XY) * sizeof(dfloat);
        #endif //A_XX_DIST
        #ifdef A_XY_DIST
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.Axy_h_fGhost.X_0), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF));
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.Axy_h_fGhost.X_1), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF));
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.Axy_h_fGhost.Y_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF));
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.Axy_h_fGhost.Y_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF));
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.Axy_h_fGhost.Z_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XY * GF));
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.Axy_h_fGhost.Z_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XY * GF));

        memAllocated += GF * (NUMBER_GHOST_FACE_YZ + NUMBER_GHOST_FACE_XZ + NUMBER_GHOST_FACE_XY) * sizeof(dfloat);
        #endif //A_XY_DIST
        #ifdef A_XZ_DIST
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.Axz_h_fGhost.X_0), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF));
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.Axz_h_fGhost.X_1), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF));
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.Axz_h_fGhost.Y_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF));
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.Axz_h_fGhost.Y_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF));
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.Axz_h_fGhost.Z_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XY * GF));
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.Axz_h_fGhost.Z_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XY * GF));

        memAllocated += GF * (NUMBER_GHOST_FACE_YZ + NUMBER_GHOST_FACE_XZ + NUMBER_GHOST_FACE_XY) * sizeof(dfloat);
        #endif //A_XZ_DIST
        #ifdef A_YY_DIST
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.Ayy_h_fGhost.X_0), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF));
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.Ayy_h_fGhost.X_1), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF));
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.Ayy_h_fGhost.Y_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF));
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.Ayy_h_fGhost.Y_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF));
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.Ayy_h_fGhost.Z_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XY * GF));
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.Ayy_h_fGhost.Z_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XY * GF));

        memAllocated += GF * (NUMBER_GHOST_FACE_YZ + NUMBER_GHOST_FACE_XZ + NUMBER_GHOST_FACE_XY) * sizeof(dfloat);
        #endif //A_YY_DIST
        #ifdef A_YZ_DIST
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.Ayz_h_fGhost.X_0), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF));
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.Ayz_h_fGhost.X_1), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF));
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.Ayz_h_fGhost.Y_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF));
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.Ayz_h_fGhost.Y_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF));
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.Ayz_h_fGhost.Z_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XY * GF));
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.Ayz_h_fGhost.Z_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XY * GF));

        memAllocated += GF * (NUMBER_GHOST_FACE_YZ + NUMBER_GHOST_FACE_XZ + NUMBER_GHOST_FACE_XY) * sizeof(dfloat);
        #endif //A_YZ_DIST
        #ifdef A_ZZ_DIST
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.Azz_h_fGhost.X_0), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF));
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.Azz_h_fGhost.X_1), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF));
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.Azz_h_fGhost.Y_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF));
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.Azz_h_fGhost.Y_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF));
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.Azz_h_fGhost.Z_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XY * GF));
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.Azz_h_fGhost.Z_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XY * GF));

        memAllocated += GF * (NUMBER_GHOST_FACE_YZ + NUMBER_GHOST_FACE_XZ + NUMBER_GHOST_FACE_XY) * sizeof(dfloat);
        #endif //A_ZZ_DIST
    }

    printf("Device Memory Allocated for Interface: %.2f MB \n", (float)memAllocated /(1024.0 * 1024.0)); if(console_flush) fflush(stdout);
}

/**
 * @brief Initialize the simulation domain, including random numbers, LBM distributions, node types, and ghost interfaces.
 * @details This function is now inlined in deviceField.cuh:initializeDomainDeviceField()
 */

#endif // MAIN_CUH
