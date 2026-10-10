/**
 *  @file ghostInterface.cu
 *  Contributors history:
 *  @author Waine Jr. (waine@alunos.utfpr.edu.br)
 *  @author Marco Aurelio Ferrari (e.marcoferrari@utfpr.edu.br)
 *  @author Ricardo de Souza
 *  @brief Ghost-interface memory and copy utilities.
 *  @version 0.4.0
 *  @date 01/09/2025
 */

#include <stdio.h>
#include <stdlib.h>

// CUDA INCLUDE
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// FILE INCLUDES
#include "ghostInterface.cuh"
#include "include/errorDef.h"

//TODO: maybe move to inside deviceField?
/**
 *  @brief Swaps the pointers of two dfloat variables.
 *  @param pt1: reference to the first dfloat pointer to be swapped
 *  @param pt2: reference to the second dfloat pointer to be swapped
 */
__host__ __device__
static void interfaceSwap(dfloat* &pt1, dfloat* &pt2) {
    dfloat *temp = pt1;
    pt1 = pt2;
    pt2 = temp;
}

/**
 *  @brief Frees the memory allocated for the ghost interface data.
 *  @param ghostInterface: reference to the ghost interface data structure
 */
__host__
void interfaceFree(ghostInterfaceData &ghostInterface, ghostInterfaceData &ghostInterfaceCopy, macroInterfaceGPUData &macroInterfaceGPU, macroInterfaceGPUData &macroInterfaceGPUCopy)
{
    cudaFree(ghostInterface.pop.X_0);
    cudaFree(ghostInterface.pop.X_1);
    cudaFree(ghostInterface.pop.Y_0);
    cudaFree(ghostInterface.pop.Y_1);
    cudaFree(ghostInterface.pop.Z_0);
    cudaFree(ghostInterface.pop.Z_1);
    cudaFree(ghostInterface.popAux.Z_0);
    cudaFree(ghostInterface.popAux.Z_1);

    cudaFree(ghostInterfaceCopy.popAux.Z_0);
    cudaFree(ghostInterfaceCopy.popAux.Z_1);

    #ifdef STREAMING_AB
        cudaFree(ghostInterfaceCopy.pop.X_0);
        cudaFree(ghostInterfaceCopy.pop.X_1);
        cudaFree(ghostInterfaceCopy.pop.Y_0);
        cudaFree(ghostInterfaceCopy.pop.Y_1);
        cudaFree(ghostInterfaceCopy.pop.Z_0);
        cudaFree(ghostInterfaceCopy.pop.Z_1);
    #endif // STREAMING_AB

    cudaFree(macroInterfaceGPU.rho.Z_0);
    cudaFree(macroInterfaceGPU.rho.auxZ_0);
    cudaFree(macroInterfaceGPU.rho.Z_1);
    cudaFree(macroInterfaceGPU.rho.auxZ_1);

    cudaFree(macroInterfaceGPUCopy.rho.auxZ_0);
    cudaFree(macroInterfaceGPUCopy.rho.auxZ_1);

    cudaFree(macroInterfaceGPU.ux.Z_0);
    cudaFree(macroInterfaceGPU.ux.auxZ_0);
    cudaFree(macroInterfaceGPU.ux.Z_1);
    cudaFree(macroInterfaceGPU.ux.auxZ_1);

    cudaFree(macroInterfaceGPUCopy.ux.auxZ_0);
    cudaFree(macroInterfaceGPUCopy.ux.auxZ_1);

    cudaFree(macroInterfaceGPU.uy.Z_0);
    cudaFree(macroInterfaceGPU.uy.auxZ_0);
    cudaFree(macroInterfaceGPU.uy.Z_1);
    cudaFree(macroInterfaceGPU.uy.auxZ_1);

    cudaFree(macroInterfaceGPUCopy.uy.auxZ_0);
    cudaFree(macroInterfaceGPUCopy.uy.auxZ_1);

    cudaFree(macroInterfaceGPU.uz.Z_0);
    cudaFree(macroInterfaceGPU.uz.auxZ_0);
    cudaFree(macroInterfaceGPU.uz.Z_1);
    cudaFree(macroInterfaceGPU.uz.auxZ_1);

    cudaFree(macroInterfaceGPUCopy.uz.auxZ_0);
    cudaFree(macroInterfaceGPUCopy.uz.auxZ_1);

    #ifdef SECOND_DIST
        cudaFree(ghostInterface.g.X_0);
        cudaFree(ghostInterface.g.X_1);
        cudaFree(ghostInterface.g.Y_0);
        cudaFree(ghostInterface.g.Y_1);
        cudaFree(ghostInterface.g.Z_0);
        cudaFree(ghostInterface.g.Z_1);
        cudaFree(ghostInterface.gAux.Z_0);
        cudaFree(ghostInterface.gAux.Z_1);

        cudaFree(ghostInterfaceCopy.gAux.Z_0);
        cudaFree(ghostInterfaceCopy.gAux.Z_1);
        
        cudaFree(macroInterfaceGPU.g.Z_0);
        cudaFree(macroInterfaceGPU.g.auxZ_0);
        cudaFree(macroInterfaceGPU.g.Z_1);
        cudaFree(macroInterfaceGPU.g.auxZ_1);

        cudaFree(macroInterfaceGPUCopy.g.auxZ_0);
        cudaFree(macroInterfaceGPUCopy.g.auxZ_1);

    #endif //SECOND_DIST
    #ifdef PHI_DIST
        cudaFree(ghostInterface.phi.X_0);
        cudaFree(ghostInterface.phi.X_1);
        cudaFree(ghostInterface.phi.Y_0);
        cudaFree(ghostInterface.phi.Y_1);
        cudaFree(ghostInterface.phi.Z_0);
        cudaFree(ghostInterface.phi.Z_1);
        cudaFree(ghostInterface.phiAux.Z_0);
        cudaFree(ghostInterface.phiAux.Z_1);

        cudaFree(ghostInterfaceCopy.phiAux.Z_0);
        cudaFree(ghostInterfaceCopy.phiAux.Z_1);

        cudaFree(macroInterfaceGPU.phi.Z_0);
        cudaFree(macroInterfaceGPU.phi.auxZ_0);
        cudaFree(macroInterfaceGPU.phi.Z_1);
        cudaFree(macroInterfaceGPU.phi.auxZ_1);

        cudaFree(macroInterfaceGPUCopy.phi.auxZ_0);
        cudaFree(macroInterfaceGPUCopy.phi.auxZ_1);

        cudaFree(macroInterfaceGPU.nx.Z_0);
        cudaFree(macroInterfaceGPU.nx.auxZ_0);
        cudaFree(macroInterfaceGPU.nx.Z_1);
        cudaFree(macroInterfaceGPU.nx.auxZ_1);

        cudaFree(macroInterfaceGPUCopy.nx.auxZ_0);
        cudaFree(macroInterfaceGPUCopy.nx.auxZ_1);

        cudaFree(macroInterfaceGPU.ny.Z_0);
        cudaFree(macroInterfaceGPU.ny.auxZ_0);
        cudaFree(macroInterfaceGPU.ny.Z_1);
        cudaFree(macroInterfaceGPU.ny.auxZ_1);

        cudaFree(macroInterfaceGPUCopy.ny.auxZ_0);
        cudaFree(macroInterfaceGPUCopy.ny.auxZ_1);

        cudaFree(macroInterfaceGPU.nz.Z_0);
        cudaFree(macroInterfaceGPU.nz.auxZ_0);
        cudaFree(macroInterfaceGPU.nz.Z_1);
        cudaFree(macroInterfaceGPU.nz.auxZ_1);

        cudaFree(macroInterfaceGPUCopy.nz.auxZ_0);
        cudaFree(macroInterfaceGPUCopy.nz.auxZ_1);

    #ifdef PHASE_NEEDS_CHEMICAL_POTENTIAL
            cudaFree(macroInterfaceGPU.mu.Z_0);
            cudaFree(macroInterfaceGPU.mu.auxZ_0);
            cudaFree(macroInterfaceGPU.mu.Z_1);
            cudaFree(macroInterfaceGPU.mu.auxZ_1);

            cudaFree(macroInterfaceGPUCopy.mu.auxZ_0);
            cudaFree(macroInterfaceGPUCopy.mu.auxZ_1);
    #endif // PHASE_NEEDS_CHEMICAL_POTENTIAL

    #endif //PHI_DIST
    #ifdef LAMBDA_DIST
        cudaFree(ghostInterface.lambda.X_0);
        cudaFree(ghostInterface.lambda.X_1);
        cudaFree(ghostInterface.lambda.Y_0);
        cudaFree(ghostInterface.lambda.Y_1);
        cudaFree(ghostInterface.lambda.Z_0);
        cudaFree(ghostInterface.lambda.Z_1);
        cudaFree(ghostInterface.lambdaAux.Z_0);
        cudaFree(ghostInterface.lambdaAux.Z_1);

        cudaFree(ghostInterfaceCopy.lambdaAux.Z_0);
        cudaFree(ghostInterfaceCopy.lambdaAux.Z_1);

        cudaFree(macroInterfaceGPU.lambda.Z_0);
        cudaFree(macroInterfaceGPU.lambda.auxZ_0);
        cudaFree(macroInterfaceGPU.lambda.Z_1);
        cudaFree(macroInterfaceGPU.lambda.auxZ_1);

        cudaFree(macroInterfaceGPUCopy.lambda.auxZ_0);
        cudaFree(macroInterfaceGPUCopy.lambda.auxZ_1);

    #endif //LAMBDA_DIST
    #ifdef A_XX_DIST
        cudaFree(ghostInterface.Axx.X_0);
        cudaFree(ghostInterface.Axx.X_1);
        cudaFree(ghostInterface.Axx.Y_0);
        cudaFree(ghostInterface.Axx.Y_1);
        cudaFree(ghostInterface.Axx.Z_0);
        cudaFree(ghostInterface.Axx.Z_1);
        cudaFree(ghostInterface.AxxAux.Z_0);
        cudaFree(ghostInterface.AxxAux.Z_1);

        cudaFree(ghostInterfaceCopy.AxxAux.Z_0);
        cudaFree(ghostInterfaceCopy.AxxAux.Z_1);

        cudaFree(macroInterfaceGPU.Axx.Z_0);
        cudaFree(macroInterfaceGPU.Axx.auxZ_0);
        cudaFree(macroInterfaceGPU.Axx.Z_1);
        cudaFree(macroInterfaceGPU.Axx.auxZ_1);

        cudaFree(macroInterfaceGPUCopy.Axx.auxZ_0);
        cudaFree(macroInterfaceGPUCopy.Axx.auxZ_1);

    #endif //A_XX_DIST
    #ifdef A_XY_DIST
        cudaFree(ghostInterface.Axy.X_0);
        cudaFree(ghostInterface.Axy.X_1);
        cudaFree(ghostInterface.Axy.Y_0);
        cudaFree(ghostInterface.Axy.Y_1);
        cudaFree(ghostInterface.Axy.Z_0);
        cudaFree(ghostInterface.Axy.Z_1);
        cudaFree(ghostInterface.AxyAux.Z_0);
        cudaFree(ghostInterface.AxyAux.Z_1);

        cudaFree(ghostInterfaceCopy.AxyAux.Z_0);
        cudaFree(ghostInterfaceCopy.AxyAux.Z_1);

        cudaFree(macroInterfaceGPU.Axy.Z_0);
        cudaFree(macroInterfaceGPU.Axy.auxZ_0);
        cudaFree(macroInterfaceGPU.Axy.Z_1);
        cudaFree(macroInterfaceGPU.Axy.auxZ_1);

        cudaFree(macroInterfaceGPUCopy.Axy.auxZ_0);
        cudaFree(macroInterfaceGPUCopy.Axy.auxZ_1);

    #endif //A_XY_DIST
    #ifdef A_XZ_DIST
        cudaFree(ghostInterface.Axz.X_0);
        cudaFree(ghostInterface.Axz.X_1);
        cudaFree(ghostInterface.Axz.Y_0);
        cudaFree(ghostInterface.Axz.Y_1);
        cudaFree(ghostInterface.Axz.Z_0);
        cudaFree(ghostInterface.Axz.Z_1);
        cudaFree(ghostInterface.AxzAux.Z_0);
        cudaFree(ghostInterface.AxzAux.Z_1);

        cudaFree(ghostInterfaceCopy.AxzAux.Z_0);
        cudaFree(ghostInterfaceCopy.AxzAux.Z_1);

        cudaFree(macroInterfaceGPU.Axz.Z_0);
        cudaFree(macroInterfaceGPU.Axz.auxZ_0);
        cudaFree(macroInterfaceGPU.Axz.Z_1);
        cudaFree(macroInterfaceGPU.Axz.auxZ_1);

        cudaFree(macroInterfaceGPUCopy.Axz.auxZ_0);
        cudaFree(macroInterfaceGPUCopy.Axz.auxZ_1);

    #endif //A_XZ_DIST
    #ifdef A_YY_DIST
        cudaFree(ghostInterface.Ayy.X_0);
        cudaFree(ghostInterface.Ayy.X_1);
        cudaFree(ghostInterface.Ayy.Y_0);
        cudaFree(ghostInterface.Ayy.Y_1);
        cudaFree(ghostInterface.Ayy.Z_0);
        cudaFree(ghostInterface.Ayy.Z_1);
        cudaFree(ghostInterface.AyyAux.Z_0);
        cudaFree(ghostInterface.AyyAux.Z_1);

        cudaFree(ghostInterfaceCopy.AyyAux.Z_0);
        cudaFree(ghostInterfaceCopy.AyyAux.Z_1);

        cudaFree(macroInterfaceGPU.Ayy.Z_0);
        cudaFree(macroInterfaceGPU.Ayy.auxZ_0);
        cudaFree(macroInterfaceGPU.Ayy.Z_1);
        cudaFree(macroInterfaceGPU.Ayy.auxZ_1);

        cudaFree(macroInterfaceGPUCopy.Ayy.auxZ_0);
        cudaFree(macroInterfaceGPUCopy.Ayy.auxZ_1);

    #endif //A_YY_DIST
    #ifdef A_YZ_DIST
        cudaFree(ghostInterface.Ayz.X_0);
        cudaFree(ghostInterface.Ayz.X_1);
        cudaFree(ghostInterface.Ayz.Y_0);
        cudaFree(ghostInterface.Ayz.Y_1);
        cudaFree(ghostInterface.Ayz.Z_0);
        cudaFree(ghostInterface.Ayz.Z_1);
        cudaFree(ghostInterface.AyzAux.Z_0);
        cudaFree(ghostInterface.AyzAux.Z_1);

        cudaFree(ghostInterfaceCopy.AyzAux.Z_0);
        cudaFree(ghostInterfaceCopy.AyzAux.Z_1);

        cudaFree(macroInterfaceGPU.Ayz.Z_0);
        cudaFree(macroInterfaceGPU.Ayz.auxZ_0);
        cudaFree(macroInterfaceGPU.Ayz.Z_1);
        cudaFree(macroInterfaceGPU.Ayz.auxZ_1);

        cudaFree(macroInterfaceGPUCopy.Ayz.auxZ_0);
        cudaFree(macroInterfaceGPUCopy.Ayz.auxZ_1);

    #endif //A_YZ_DIST
    #ifdef A_ZZ_DIST
        cudaFree(ghostInterface.Azz.X_0);
        cudaFree(ghostInterface.Azz.X_1);
        cudaFree(ghostInterface.Azz.Y_0);
        cudaFree(ghostInterface.Azz.Y_1);
        cudaFree(ghostInterface.Azz.Z_0);
        cudaFree(ghostInterface.Azz.Z_1);
        cudaFree(ghostInterface.AzzAux.Z_0);
        cudaFree(ghostInterface.AzzAux.Z_1);

        cudaFree(ghostInterfaceCopy.AzzAux.Z_0);
        cudaFree(ghostInterfaceCopy.AzzAux.Z_1);

        cudaFree(macroInterfaceGPU.Azz.Z_0);
        cudaFree(macroInterfaceGPU.Azz.auxZ_0);
        cudaFree(macroInterfaceGPU.Azz.Z_1);
        cudaFree(macroInterfaceGPU.Azz.auxZ_1);

        cudaFree(macroInterfaceGPUCopy.Azz.auxZ_0);
        cudaFree(macroInterfaceGPUCopy.Azz.auxZ_1);

    #endif //A_ZZ_DIST

    // Canonical global checkpoints do not use legacy per-partition host faces.
    if (false && (LOAD_CHECKPOINT || CHECKPOINT_SAVE)){
        cudaFreeHost(ghostInterface.h_pop.X_0);
        cudaFreeHost(ghostInterface.h_pop.X_1);
        cudaFreeHost(ghostInterface.h_pop.Y_0);
        cudaFreeHost(ghostInterface.h_pop.Y_1);
        cudaFreeHost(ghostInterface.h_pop.Z_0);
        cudaFreeHost(ghostInterface.h_pop.Z_1);
        #ifdef SECOND_DIST
            cudaFreeHost(ghostInterface.h_g.X_0);
            cudaFreeHost(ghostInterface.h_g.X_1);
            cudaFreeHost(ghostInterface.h_g.Y_0);
            cudaFreeHost(ghostInterface.h_g.Y_1);
            cudaFreeHost(ghostInterface.h_g.Z_0);
            cudaFreeHost(ghostInterface.h_g.Z_1);
        #endif //SECOND_DIST
        #ifdef PHI_DIST
            cudaFreeHost(ghostInterface.h_phi.X_0);
            cudaFreeHost(ghostInterface.h_phi.X_1);
            cudaFreeHost(ghostInterface.h_phi.Y_0);
            cudaFreeHost(ghostInterface.h_phi.Y_1);
            cudaFreeHost(ghostInterface.h_phi.Z_0);
            cudaFreeHost(ghostInterface.h_phi.Z_1);
        #endif //PHI_DIST
        #ifdef LAMBDA_DIST
            cudaFreeHost(ghostInterface.h_lambda.X_0);
            cudaFreeHost(ghostInterface.h_lambda.X_1);
            cudaFreeHost(ghostInterface.h_lambda.Y_0);
            cudaFreeHost(ghostInterface.h_lambda.Y_1);
            cudaFreeHost(ghostInterface.h_lambda.Z_0);
            cudaFreeHost(ghostInterface.h_lambda.Z_1);
        #endif //LAMBDA_DIST
        #ifdef A_XX_DIST
            cudaFreeHost(ghostInterface.h_Axx.X_0);
            cudaFreeHost(ghostInterface.h_Axx.X_1);
            cudaFreeHost(ghostInterface.h_Axx.Y_0);
            cudaFreeHost(ghostInterface.h_Axx.Y_1);
            cudaFreeHost(ghostInterface.h_Axx.Z_0);
            cudaFreeHost(ghostInterface.h_Axx.Z_1);
        #endif //A_XX_DIST
        #ifdef A_XY_DIST
            cudaFreeHost(ghostInterface.h_Axy.X_0);
            cudaFreeHost(ghostInterface.h_Axy.X_1);
            cudaFreeHost(ghostInterface.h_Axy.Y_0);
            cudaFreeHost(ghostInterface.h_Axy.Y_1);
            cudaFreeHost(ghostInterface.h_Axy.Z_0);
            cudaFreeHost(ghostInterface.h_Axy.Z_1);
        #endif //A_XY_DIST
        #ifdef A_XZ_DIST
            cudaFreeHost(ghostInterface.h_Axz.X_0);
            cudaFreeHost(ghostInterface.h_Axz.X_1);
            cudaFreeHost(ghostInterface.h_Axz.Y_0);
            cudaFreeHost(ghostInterface.h_Axz.Y_1);
            cudaFreeHost(ghostInterface.h_Axz.Z_0);
            cudaFreeHost(ghostInterface.h_Axz.Z_1);
        #endif //A_XZ_DIST
        #ifdef A_YY_DIST
            cudaFreeHost(ghostInterface.h_Ayy.X_0);
            cudaFreeHost(ghostInterface.h_Ayy.X_1);
            cudaFreeHost(ghostInterface.h_Ayy.Y_0);
            cudaFreeHost(ghostInterface.h_Ayy.Y_1);
            cudaFreeHost(ghostInterface.h_Ayy.Z_0);
            cudaFreeHost(ghostInterface.h_Ayy.Z_1);
        #endif //A_YY_DIST
        #ifdef A_YZ_DIST
            cudaFreeHost(ghostInterface.h_Ayz.X_0);
            cudaFreeHost(ghostInterface.h_Ayz.X_1);
            cudaFreeHost(ghostInterface.h_Ayz.Y_0);
            cudaFreeHost(ghostInterface.h_Ayz.Y_1);
            cudaFreeHost(ghostInterface.h_Ayz.Z_0);
            cudaFreeHost(ghostInterface.h_Ayz.Z_1);
        #endif //A_YZ_DIST
        #ifdef A_ZZ_DIST
            cudaFreeHost(ghostInterface.h_Azz.X_0);
            cudaFreeHost(ghostInterface.h_Azz.X_1);
            cudaFreeHost(ghostInterface.h_Azz.Y_0);
            cudaFreeHost(ghostInterface.h_Azz.Y_1);
            cudaFreeHost(ghostInterface.h_Azz.Z_0);
            cudaFreeHost(ghostInterface.h_Azz.Z_1);
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
void interfaceCudaMemcpy(GhostInterfaceData& ghostInterface, ghostData& dst, const ghostData& src,
                         cudaMemcpyKind kind, int Q, int g) {
    checkCudaErrors(cudaSetDevice(GPUS_TO_USE[g]));
    const DomainDecomposition decomposition(N_GPUS);
    const GpuDomainLayout& layout = decomposition[g];
    const size_t NUMBER_GHOST_FACE_YZ = layout.ghostFaceYZCount;
    const size_t NUMBER_GHOST_FACE_XZ = layout.ghostFaceXZCount;
    const size_t NUMBER_GHOST_FACE_XY_LOCAL = layout.ghostFaceXYCount;
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
void swapGhostInterfaces(GhostInterfaceData& ghostInterface, GhostInterfaceData& ghostInterfaceCopy) {
    #ifdef STREAMING_AB
        interfaceSwap(ghostInterface.pop.X_0, ghostInterfaceCopy.pop.X_0);
        interfaceSwap(ghostInterface.pop.X_1, ghostInterfaceCopy.pop.X_1);
        interfaceSwap(ghostInterface.pop.Y_0, ghostInterfaceCopy.pop.Y_0);
        interfaceSwap(ghostInterface.pop.Y_1, ghostInterfaceCopy.pop.Y_1);
        interfaceSwap(ghostInterface.pop.Z_0, ghostInterfaceCopy.pop.Z_0);
        interfaceSwap(ghostInterface.pop.Z_1, ghostInterfaceCopy.pop.Z_1);
        interfaceSwap(ghostInterface.popAux.Z_0, ghostInterfaceCopy.popAux.Z_0);
        interfaceSwap(ghostInterface.popAux.Z_1, ghostInterfaceCopy.popAux.Z_1);
    #else
        (void)ghostInterface;
        (void)ghostInterfaceCopy;
    #endif // STREAMING_AB
}

/**
 *  @brief Allocates memory for the ghost interface data.
 *  @param ghostInterface: reference to the ghost interface data structure
 */
static cudaError_t allocateZeroedInterfaceBuffer(void** pointer, size_t bytes)
{
    cudaError_t error = cudaMalloc(pointer, bytes);
    if (error != cudaSuccess) return error;
    error = cudaMemset(*pointer, 0, bytes);
    if (error != cudaSuccess) {
        cudaFree(*pointer);
        *pointer = nullptr;
    }
    return error;
}

__host__
void interfaceMalloc(ghostInterfaceData &ghostInterface, ghostInterfaceData &ghostInterfaceCopy,
                     macroInterfaceGPUData &macroInterfaceGPU, macroInterfaceGPUData &macroInterfaceGPUCopy,
                     const GpuDomainLayout& layout)
{
#define cudaMalloc(pointer, bytes) allocateZeroedInterfaceBuffer(pointer, bytes)
    size_t memAllocated = 0;
    const size_t NUMBER_GHOST_FACE_YZ = layout.ghostFaceYZCount;
    const size_t NUMBER_GHOST_FACE_XZ = layout.ghostFaceXZCount;
    const size_t NUMBER_GHOST_FACE_XY_LOCAL = layout.ghostFaceXYCount;
    const size_t NUMBER_GHOST_FACE_XY_AUX = layout.exchangePlaneCount;

    size_t MacroBoundarySize = NX * NY * sizeof(dfloat);

    cudaMalloc((void **)&(ghostInterface.pop.X_0), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * QF);
    cudaMalloc((void **)&(ghostInterface.pop.X_1), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * QF);
    cudaMalloc((void **)&(ghostInterface.pop.Y_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * QF);
    cudaMalloc((void **)&(ghostInterface.pop.Y_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * QF);
    cudaMalloc((void **)&(ghostInterface.pop.Z_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XY_LOCAL * QF);
    cudaMalloc((void **)&(ghostInterface.pop.Z_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XY_LOCAL * QF);
    cudaMalloc((void **)&(ghostInterface.popAux.Z_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XY_AUX * QF);
    cudaMalloc((void **)&(ghostInterface.popAux.Z_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XY_AUX * QF);

    cudaMalloc((void **)&(ghostInterfaceCopy.popAux.Z_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XY_AUX * QF);
    cudaMalloc((void **)&(ghostInterfaceCopy.popAux.Z_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XY_AUX * QF);

    #ifdef STREAMING_AB
        cudaMalloc((void **)&(ghostInterfaceCopy.pop.X_0), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * QF);
        cudaMalloc((void **)&(ghostInterfaceCopy.pop.X_1), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * QF);
        cudaMalloc((void **)&(ghostInterfaceCopy.pop.Y_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * QF);
        cudaMalloc((void **)&(ghostInterfaceCopy.pop.Y_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * QF);
        cudaMalloc((void **)&(ghostInterfaceCopy.pop.Z_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XY_LOCAL * QF);
        cudaMalloc((void **)&(ghostInterfaceCopy.pop.Z_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XY_LOCAL * QF);
    #endif // STREAMING_AB

    cudaMalloc((void **)&(macroInterfaceGPU.rho.Z_0), MacroBoundarySize);
    cudaMalloc((void **)&(macroInterfaceGPU.rho.Z_1), MacroBoundarySize);
    cudaMalloc((void **)&(macroInterfaceGPU.rho.auxZ_0), MacroBoundarySize);
    cudaMalloc((void **)&(macroInterfaceGPU.rho.auxZ_1), MacroBoundarySize);

    cudaMalloc((void **)&(macroInterfaceGPUCopy.rho.auxZ_0), MacroBoundarySize);
    cudaMalloc((void **)&(macroInterfaceGPUCopy.rho.auxZ_1), MacroBoundarySize);

    cudaMalloc((void **)&(macroInterfaceGPU.ux.Z_0), MacroBoundarySize);
    cudaMalloc((void **)&(macroInterfaceGPU.ux.Z_1), MacroBoundarySize);
    cudaMalloc((void **)&(macroInterfaceGPU.ux.auxZ_0), MacroBoundarySize);
    cudaMalloc((void **)&(macroInterfaceGPU.ux.auxZ_1), MacroBoundarySize);

    cudaMalloc((void **)&(macroInterfaceGPUCopy.ux.auxZ_0), MacroBoundarySize);
    cudaMalloc((void **)&(macroInterfaceGPUCopy.ux.auxZ_1), MacroBoundarySize);

    cudaMalloc((void **)&(macroInterfaceGPU.uy.Z_0), MacroBoundarySize);
    cudaMalloc((void **)&(macroInterfaceGPU.uy.Z_1), MacroBoundarySize);
    cudaMalloc((void **)&(macroInterfaceGPU.uy.auxZ_0), MacroBoundarySize);
    cudaMalloc((void **)&(macroInterfaceGPU.uy.auxZ_1), MacroBoundarySize);
    
    cudaMalloc((void **)&(macroInterfaceGPUCopy.uy.auxZ_0), MacroBoundarySize);
    cudaMalloc((void **)&(macroInterfaceGPUCopy.uy.auxZ_1), MacroBoundarySize);

    cudaMalloc((void **)&(macroInterfaceGPU.uz.Z_0), MacroBoundarySize);
    cudaMalloc((void **)&(macroInterfaceGPU.uz.Z_1), MacroBoundarySize);
    cudaMalloc((void **)&(macroInterfaceGPU.uz.auxZ_0), MacroBoundarySize);
    cudaMalloc((void **)&(macroInterfaceGPU.uz.auxZ_1), MacroBoundarySize);

    cudaMalloc((void **)&(macroInterfaceGPUCopy.uz.auxZ_0), MacroBoundarySize);
    cudaMalloc((void **)&(macroInterfaceGPUCopy.uz.auxZ_1), MacroBoundarySize);

    #ifdef STREAMING_AB
        memAllocated = 4ULL * QF * (NUMBER_GHOST_FACE_YZ + NUMBER_GHOST_FACE_XZ + NUMBER_GHOST_FACE_XY_LOCAL) * sizeof(dfloat);
    #else
        memAllocated = 2ULL * QF * (NUMBER_GHOST_FACE_YZ + NUMBER_GHOST_FACE_XZ + NUMBER_GHOST_FACE_XY_LOCAL) * sizeof(dfloat);
    #endif // STREAMING_AB

    #ifdef SECOND_DIST
        cudaMalloc((void **)&(ghostInterface.g.X_0), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF);
        cudaMalloc((void **)&(ghostInterface.g.X_1), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF);
        cudaMalloc((void **)&(ghostInterface.g.Y_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF);
        cudaMalloc((void **)&(ghostInterface.g.Y_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF);
        cudaMalloc((void **)&(ghostInterface.g.Z_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XY_LOCAL * GF);
        cudaMalloc((void **)&(ghostInterface.g.Z_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XY_LOCAL * GF);
        cudaMalloc((void **)&(ghostInterface.gAux.Z_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XY_AUX * GF);
        cudaMalloc((void **)&(ghostInterface.gAux.Z_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XY_AUX * GF);

        cudaMalloc((void **)&(ghostInterfaceCopy.gAux.Z_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XY_AUX * GF);
        cudaMalloc((void **)&(ghostInterfaceCopy.gAux.Z_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XY_AUX * GF);

        cudaMalloc((void **)&(macroInterfaceGPU.g.Z_0), MacroBoundarySize);
        cudaMalloc((void **)&(macroInterfaceGPU.g.Z_1), MacroBoundarySize);
        cudaMalloc((void **)&(macroInterfaceGPU.g.auxZ_0), MacroBoundarySize);
        cudaMalloc((void **)&(macroInterfaceGPU.g.auxZ_1), MacroBoundarySize);

        cudaMalloc((void **)&(macroInterfaceGPUCopy.g.auxZ_0), MacroBoundarySize);
        cudaMalloc((void **)&(macroInterfaceGPUCopy.g.auxZ_1), MacroBoundarySize);

        memAllocated += 2 * GF * (NUMBER_GHOST_FACE_YZ + NUMBER_GHOST_FACE_XZ + NUMBER_GHOST_FACE_XY_LOCAL) * sizeof(dfloat);
    #endif //SECOND_DIST

    #ifdef PHI_DIST
        cudaMalloc((void **)&(ghostInterface.phi.X_0), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF);
        cudaMalloc((void **)&(ghostInterface.phi.X_1), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF);
        cudaMalloc((void **)&(ghostInterface.phi.Y_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF);
        cudaMalloc((void **)&(ghostInterface.phi.Y_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF);
        cudaMalloc((void **)&(ghostInterface.phi.Z_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XY_LOCAL * GF);
        cudaMalloc((void **)&(ghostInterface.phi.Z_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XY_LOCAL * GF);
        cudaMalloc((void **)&(ghostInterface.phiAux.Z_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XY_AUX * GF);
        cudaMalloc((void **)&(ghostInterface.phiAux.Z_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XY_AUX * GF);

        cudaMalloc((void **)&(ghostInterfaceCopy.phiAux.Z_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XY_AUX * GF);
        cudaMalloc((void **)&(ghostInterfaceCopy.phiAux.Z_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XY_AUX * GF);

        cudaMalloc((void **)&(macroInterfaceGPU.phi.Z_0), MacroBoundarySize);
        cudaMalloc((void **)&(macroInterfaceGPU.phi.Z_1), MacroBoundarySize);
        cudaMalloc((void **)&(macroInterfaceGPU.phi.auxZ_0), MacroBoundarySize);
        cudaMalloc((void **)&(macroInterfaceGPU.phi.auxZ_1), MacroBoundarySize);

        cudaMalloc((void **)&(macroInterfaceGPUCopy.phi.auxZ_0), MacroBoundarySize);
        cudaMalloc((void **)&(macroInterfaceGPUCopy.phi.auxZ_1), MacroBoundarySize);

        cudaMalloc((void **)&(macroInterfaceGPU.nx.Z_0), MacroBoundarySize);
        cudaMalloc((void **)&(macroInterfaceGPU.nx.Z_1), MacroBoundarySize);
        cudaMalloc((void **)&(macroInterfaceGPU.nx.auxZ_0), MacroBoundarySize);
        cudaMalloc((void **)&(macroInterfaceGPU.nx.auxZ_1), MacroBoundarySize);

        cudaMalloc((void **)&(macroInterfaceGPUCopy.nx.auxZ_0), MacroBoundarySize);
        cudaMalloc((void **)&(macroInterfaceGPUCopy.nx.auxZ_1), MacroBoundarySize);

        cudaMalloc((void **)&(macroInterfaceGPU.ny.Z_0), MacroBoundarySize);
        cudaMalloc((void **)&(macroInterfaceGPU.ny.Z_1), MacroBoundarySize);
        cudaMalloc((void **)&(macroInterfaceGPU.ny.auxZ_0), MacroBoundarySize);
        cudaMalloc((void **)&(macroInterfaceGPU.ny.auxZ_1), MacroBoundarySize);

        cudaMalloc((void **)&(macroInterfaceGPUCopy.ny.auxZ_0), MacroBoundarySize);
        cudaMalloc((void **)&(macroInterfaceGPUCopy.ny.auxZ_1), MacroBoundarySize);

        cudaMalloc((void **)&(macroInterfaceGPU.nz.Z_0), MacroBoundarySize);
        cudaMalloc((void **)&(macroInterfaceGPU.nz.Z_1), MacroBoundarySize);
        cudaMalloc((void **)&(macroInterfaceGPU.nz.auxZ_0), MacroBoundarySize);
        cudaMalloc((void **)&(macroInterfaceGPU.nz.auxZ_1), MacroBoundarySize);

        cudaMalloc((void **)&(macroInterfaceGPUCopy.nz.auxZ_0), MacroBoundarySize);
        cudaMalloc((void **)&(macroInterfaceGPUCopy.nz.auxZ_1), MacroBoundarySize);

    #ifdef PHASE_NEEDS_CHEMICAL_POTENTIAL
            cudaMalloc((void **)&(macroInterfaceGPU.mu.Z_0), MacroBoundarySize);
            cudaMalloc((void **)&(macroInterfaceGPU.mu.Z_1), MacroBoundarySize);
            cudaMalloc((void **)&(macroInterfaceGPU.mu.auxZ_0), MacroBoundarySize);
            cudaMalloc((void **)&(macroInterfaceGPU.mu.auxZ_1), MacroBoundarySize);

            cudaMalloc((void **)&(macroInterfaceGPUCopy.mu.auxZ_0), MacroBoundarySize);
            cudaMalloc((void **)&(macroInterfaceGPUCopy.mu.auxZ_1), MacroBoundarySize);
    #endif // PHASE_NEEDS_CHEMICAL_POTENTIAL


        memAllocated += 2 * GF * (NUMBER_GHOST_FACE_YZ + NUMBER_GHOST_FACE_XZ + NUMBER_GHOST_FACE_XY_LOCAL) * sizeof(dfloat);
    #endif //PHI_DIST

    #ifdef LAMBDA_DIST
        cudaMalloc((void **)&(ghostInterface.lambda.X_0), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF);
        cudaMalloc((void **)&(ghostInterface.lambda.X_1), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF);
        cudaMalloc((void **)&(ghostInterface.lambda.Y_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF);
        cudaMalloc((void **)&(ghostInterface.lambda.Y_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF);
        cudaMalloc((void **)&(ghostInterface.lambda.Z_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XY_LOCAL * GF);
        cudaMalloc((void **)&(ghostInterface.lambda.Z_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XY_LOCAL * GF);
        cudaMalloc((void **)&(ghostInterface.lambdaAux.Z_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XY_AUX * GF);
        cudaMalloc((void **)&(ghostInterface.lambdaAux.Z_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XY_AUX * GF);

        cudaMalloc((void **)&(ghostInterfaceCopy.lambdaAux.Z_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XY_AUX * GF);
        cudaMalloc((void **)&(ghostInterfaceCopy.lambdaAux.Z_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XY_AUX * GF);

        cudaMalloc((void **)&(macroInterfaceGPU.lambda.Z_0), MacroBoundarySize);
        cudaMalloc((void **)&(macroInterfaceGPU.lambda.Z_1), MacroBoundarySize);
        cudaMalloc((void **)&(macroInterfaceGPU.lambda.auxZ_0), MacroBoundarySize);
        cudaMalloc((void **)&(macroInterfaceGPU.lambda.auxZ_1), MacroBoundarySize);

        cudaMalloc((void **)&(macroInterfaceGPUCopy.lambda.auxZ_0), MacroBoundarySize);
        cudaMalloc((void **)&(macroInterfaceGPUCopy.lambda.auxZ_1), MacroBoundarySize);

        memAllocated += 2 * GF * (NUMBER_GHOST_FACE_YZ + NUMBER_GHOST_FACE_XZ + NUMBER_GHOST_FACE_XY_LOCAL) * sizeof(dfloat);
    #endif //LAMBDA_DIST

    #ifdef A_XX_DIST
        cudaMalloc((void **)&(ghostInterface.Axx.X_0), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF);
        cudaMalloc((void **)&(ghostInterface.Axx.X_1), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF);
        cudaMalloc((void **)&(ghostInterface.Axx.Y_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF);
        cudaMalloc((void **)&(ghostInterface.Axx.Y_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF);
        cudaMalloc((void **)&(ghostInterface.Axx.Z_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XY_LOCAL * GF);
        cudaMalloc((void **)&(ghostInterface.Axx.Z_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XY_LOCAL * GF);
        cudaMalloc((void **)&(ghostInterface.AxxAux.Z_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XY_AUX * GF);
        cudaMalloc((void **)&(ghostInterface.AxxAux.Z_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XY_AUX * GF);

        cudaMalloc((void **)&(ghostInterfaceCopy.AxxAux.Z_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XY_AUX * GF);
        cudaMalloc((void **)&(ghostInterfaceCopy.AxxAux.Z_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XY_AUX * GF);

        cudaMalloc((void **)&(macroInterfaceGPU.Axx.Z_0), MacroBoundarySize);
        cudaMalloc((void **)&(macroInterfaceGPU.Axx.Z_1), MacroBoundarySize);
        cudaMalloc((void **)&(macroInterfaceGPU.Axx.auxZ_0), MacroBoundarySize);
        cudaMalloc((void **)&(macroInterfaceGPU.Axx.auxZ_1), MacroBoundarySize);

        cudaMalloc((void **)&(macroInterfaceGPUCopy.Axx.auxZ_0), MacroBoundarySize);
        cudaMalloc((void **)&(macroInterfaceGPUCopy.Axx.auxZ_1), MacroBoundarySize);


        memAllocated += 2 * GF * (NUMBER_GHOST_FACE_YZ + NUMBER_GHOST_FACE_XZ + NUMBER_GHOST_FACE_XY_LOCAL) * sizeof(dfloat);
    #endif //A_XX_DIST

    #ifdef A_XY_DIST
        cudaMalloc((void **)&(ghostInterface.Axy.X_0), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF);
        cudaMalloc((void **)&(ghostInterface.Axy.X_1), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF);
        cudaMalloc((void **)&(ghostInterface.Axy.Y_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF);
        cudaMalloc((void **)&(ghostInterface.Axy.Y_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF);
        cudaMalloc((void **)&(ghostInterface.Axy.Z_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XY_LOCAL * GF);
        cudaMalloc((void **)&(ghostInterface.Axy.Z_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XY_LOCAL * GF);
        cudaMalloc((void **)&(ghostInterface.AxyAux.Z_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XY_AUX * GF);
        cudaMalloc((void **)&(ghostInterface.AxyAux.Z_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XY_AUX * GF);

        cudaMalloc((void **)&(ghostInterfaceCopy.AxyAux.Z_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XY_AUX * GF);
        cudaMalloc((void **)&(ghostInterfaceCopy.AxyAux.Z_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XY_AUX * GF);

        cudaMalloc((void **)&(macroInterfaceGPU.Axy.Z_0), MacroBoundarySize);
        cudaMalloc((void **)&(macroInterfaceGPU.Axy.Z_1), MacroBoundarySize);
        cudaMalloc((void **)&(macroInterfaceGPU.Axy.auxZ_0), MacroBoundarySize);
        cudaMalloc((void **)&(macroInterfaceGPU.Axy.auxZ_1), MacroBoundarySize);

        cudaMalloc((void **)&(macroInterfaceGPUCopy.Axy.auxZ_0), MacroBoundarySize);
        cudaMalloc((void **)&(macroInterfaceGPUCopy.Axy.auxZ_1), MacroBoundarySize);

        memAllocated += 2 * GF * (NUMBER_GHOST_FACE_YZ + NUMBER_GHOST_FACE_XZ + NUMBER_GHOST_FACE_XY_LOCAL) * sizeof(dfloat);
    #endif //A_XY_DIST

    #ifdef A_XZ_DIST
        cudaMalloc((void **)&(ghostInterface.Axz.X_0), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF);
        cudaMalloc((void **)&(ghostInterface.Axz.X_1), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF);
        cudaMalloc((void **)&(ghostInterface.Axz.Y_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF);
        cudaMalloc((void **)&(ghostInterface.Axz.Y_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF);
        cudaMalloc((void **)&(ghostInterface.Axz.Z_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XY_LOCAL * GF);
        cudaMalloc((void **)&(ghostInterface.Axz.Z_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XY_LOCAL * GF);
        cudaMalloc((void **)&(ghostInterface.AxzAux.Z_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XY_AUX * GF);
        cudaMalloc((void **)&(ghostInterface.AxzAux.Z_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XY_AUX * GF);

        cudaMalloc((void **)&(ghostInterfaceCopy.AxzAux.Z_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XY_AUX * GF);
        cudaMalloc((void **)&(ghostInterfaceCopy.AxzAux.Z_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XY_AUX * GF);

        cudaMalloc((void **)&(macroInterfaceGPU.Axz.Z_0), MacroBoundarySize);
        cudaMalloc((void **)&(macroInterfaceGPU.Axz.Z_1), MacroBoundarySize);
        cudaMalloc((void **)&(macroInterfaceGPU.Axz.auxZ_0), MacroBoundarySize);
        cudaMalloc((void **)&(macroInterfaceGPU.Axz.auxZ_1), MacroBoundarySize);

        cudaMalloc((void **)&(macroInterfaceGPUCopy.Axz.auxZ_0), MacroBoundarySize);
        cudaMalloc((void **)&(macroInterfaceGPUCopy.Axz.auxZ_1), MacroBoundarySize);

        memAllocated += 2 * GF * (NUMBER_GHOST_FACE_YZ + NUMBER_GHOST_FACE_XZ + NUMBER_GHOST_FACE_XY_LOCAL) * sizeof(dfloat);
    #endif //A_XZ_DIST

    #ifdef A_YY_DIST
        cudaMalloc((void **)&(ghostInterface.Ayy.X_0), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF);
        cudaMalloc((void **)&(ghostInterface.Ayy.X_1), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF);
        cudaMalloc((void **)&(ghostInterface.Ayy.Y_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF);
        cudaMalloc((void **)&(ghostInterface.Ayy.Y_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF);
        cudaMalloc((void **)&(ghostInterface.Ayy.Z_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XY_LOCAL * GF);
        cudaMalloc((void **)&(ghostInterface.Ayy.Z_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XY_LOCAL * GF);
        cudaMalloc((void **)&(ghostInterface.AyyAux.Z_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XY_AUX * GF);
        cudaMalloc((void **)&(ghostInterface.AyyAux.Z_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XY_AUX * GF);

        cudaMalloc((void **)&(ghostInterfaceCopy.AyyAux.Z_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XY_AUX * GF);
        cudaMalloc((void **)&(ghostInterfaceCopy.AyyAux.Z_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XY_AUX * GF);

        cudaMalloc((void **)&(macroInterfaceGPU.Ayy.Z_0), MacroBoundarySize);
        cudaMalloc((void **)&(macroInterfaceGPU.Ayy.Z_1), MacroBoundarySize);
        cudaMalloc((void **)&(macroInterfaceGPU.Ayy.auxZ_0), MacroBoundarySize);
        cudaMalloc((void **)&(macroInterfaceGPU.Ayy.auxZ_1), MacroBoundarySize);

        cudaMalloc((void **)&(macroInterfaceGPUCopy.Ayy.auxZ_0), MacroBoundarySize);
        cudaMalloc((void **)&(macroInterfaceGPUCopy.Ayy.auxZ_1), MacroBoundarySize);

        memAllocated += 2 * GF * (NUMBER_GHOST_FACE_YZ + NUMBER_GHOST_FACE_XZ + NUMBER_GHOST_FACE_XY_LOCAL) * sizeof(dfloat);
    #endif //A_YY_DIST

    #ifdef A_YZ_DIST
        cudaMalloc((void **)&(ghostInterface.Ayz.X_0), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF);
        cudaMalloc((void **)&(ghostInterface.Ayz.X_1), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF);
        cudaMalloc((void **)&(ghostInterface.Ayz.Y_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF);
        cudaMalloc((void **)&(ghostInterface.Ayz.Y_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF);
        cudaMalloc((void **)&(ghostInterface.Ayz.Z_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XY_LOCAL * GF);
        cudaMalloc((void **)&(ghostInterface.Ayz.Z_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XY_LOCAL * GF);
        cudaMalloc((void **)&(ghostInterface.AyzAux.Z_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XY_AUX * GF);
        cudaMalloc((void **)&(ghostInterface.AyzAux.Z_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XY_AUX * GF);

        cudaMalloc((void **)&(ghostInterfaceCopy.AyzAux.Z_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XY_AUX * GF);
        cudaMalloc((void **)&(ghostInterfaceCopy.AyzAux.Z_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XY_AUX * GF);

        cudaMalloc((void **)&(macroInterfaceGPU.Ayz.Z_0), MacroBoundarySize);
        cudaMalloc((void **)&(macroInterfaceGPU.Ayz.Z_1), MacroBoundarySize);
        cudaMalloc((void **)&(macroInterfaceGPU.Ayz.auxZ_0), MacroBoundarySize);
        cudaMalloc((void **)&(macroInterfaceGPU.Ayz.auxZ_1), MacroBoundarySize);

        cudaMalloc((void **)&(macroInterfaceGPUCopy.Ayz.auxZ_0), MacroBoundarySize);
        cudaMalloc((void **)&(macroInterfaceGPUCopy.Ayz.auxZ_1), MacroBoundarySize);

        memAllocated += 2 * GF * (NUMBER_GHOST_FACE_YZ + NUMBER_GHOST_FACE_XZ + NUMBER_GHOST_FACE_XY_LOCAL) * sizeof(dfloat);
    #endif //A_YZ_DIST

    #ifdef A_ZZ_DIST
        cudaMalloc((void **)&(ghostInterface.Azz.X_0), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF);
        cudaMalloc((void **)&(ghostInterface.Azz.X_1), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF);
        cudaMalloc((void **)&(ghostInterface.Azz.Y_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF);
        cudaMalloc((void **)&(ghostInterface.Azz.Y_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF);
        cudaMalloc((void **)&(ghostInterface.Azz.Z_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XY_LOCAL * GF);
        cudaMalloc((void **)&(ghostInterface.Azz.Z_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XY_LOCAL * GF);
        cudaMalloc((void **)&(ghostInterface.AzzAux.Z_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XY_AUX * GF);
        cudaMalloc((void **)&(ghostInterface.AzzAux.Z_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XY_AUX * GF);

        cudaMalloc((void **)&(ghostInterfaceCopy.AzzAux.Z_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XY_AUX * GF);
        cudaMalloc((void **)&(ghostInterfaceCopy.AzzAux.Z_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XY_AUX * GF);

        cudaMalloc((void **)&(macroInterfaceGPU.Azz.Z_0), MacroBoundarySize);
        cudaMalloc((void **)&(macroInterfaceGPU.Azz.Z_1), MacroBoundarySize);
        cudaMalloc((void **)&(macroInterfaceGPU.Azz.auxZ_0), MacroBoundarySize);
        cudaMalloc((void **)&(macroInterfaceGPU.Azz.auxZ_1), MacroBoundarySize);

        cudaMalloc((void **)&(macroInterfaceGPUCopy.Azz.auxZ_0), MacroBoundarySize);
        cudaMalloc((void **)&(macroInterfaceGPUCopy.Azz.auxZ_1), MacroBoundarySize);

        memAllocated += 2 * GF * (NUMBER_GHOST_FACE_YZ + NUMBER_GHOST_FACE_XZ + NUMBER_GHOST_FACE_XY_LOCAL) * sizeof(dfloat);
    #endif //A_ZZ_DIST

    if (false && (LOAD_CHECKPOINT || CHECKPOINT_SAVE))
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

        #ifdef LAMBDA_DIST
            checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.h_lambda.X_0), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF));
            checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.h_lambda.X_1), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF));
            checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.h_lambda.Y_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF));
            checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.h_lambda.Y_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF));
            checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.h_lambda.Z_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XY_LOCAL * GF));
            checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.h_lambda.Z_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XY_LOCAL * GF));

            memAllocated += GF * (NUMBER_GHOST_FACE_YZ + NUMBER_GHOST_FACE_XZ + NUMBER_GHOST_FACE_XY_LOCAL) * sizeof(dfloat);
        #endif //LAMBDA_DIST

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
#undef cudaMalloc
}

/**
 * @brief Initialize the simulation domain, including random numbers, LBM distributions, node types, and ghost interfaces.
 * @details This function is now inlined in deviceField.cuh:initializeDomainDeviceField()
 */

