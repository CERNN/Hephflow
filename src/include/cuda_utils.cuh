/**
 *  @file cuda_utils.cuh
 *  @brief CUDA utilities, timing functions and block dimension optimization
 *  @version 0.4.0
 *  @date 31/12/2025
 */

#ifndef __CUDA_UTILS_H
#define __CUDA_UTILS_H

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "var_types.h"  // for dfloat type
#include "errorDef.h"

/* ======================= COMPILE-TIME UTILITIES ========================= */

// Compile-time power of 2 checker
constexpr bool isPowerOfTwo(int x) {
    return (x & (x - 1)) == 0 && x != 0;
}

// Helper function to get the closest power of 2 under or equal to `n`
constexpr int closestPowerOfTwo(int n) {
    int power = 1;
    while (power * 2 <= n) {
        power *= 2;
    }
    return power;
}

/* ==================== CUDA BLOCK DIMENSION OPTIMIZATION ================= */

struct BlockDim {
    int x, y, z;
};

// Compute optimal block dimensions for CUDA kernels
constexpr BlockDim findOptimalBlockDimensions(size_t maxElements) {
    int bestX = 1, bestY = 1, bestZ = 1;
    int closestVolume = 0;
    float bestForm = 0.0_df;    
    
    // Iterate over powers of 2 up to `maxElements` to find optimal dimensions
    for (int x = closestPowerOfTwo(maxElements); x >= 1; x /= 2) {
        for (int y = closestPowerOfTwo(maxElements / x); y >= 1; y /= 2) {
            for (int z = closestPowerOfTwo(maxElements / (x * y)); z >= 1; z /= 2) {
                if (x * y * z <= maxElements) {
                    int volume = x * y * z;
                    float form = 1.0_df/(1.0_df/x + 1.0_df/y + 1.0_df/z);
                    if (volume > closestVolume) {
                        bestX = x;
                        bestY = y;
                        bestZ = z;
                        closestVolume = volume;
                        bestForm = form;
                    } else if (volume == closestVolume && form > bestForm) {
                        bestX = x;
                        bestY = y;
                        bestZ = z;
                        bestForm = form;
                    }
                }
            }
        }
    }
    return {bestX, bestY, bestZ};
}

#ifdef DYNAMIC_SHARED_MEMORY
// Configure dynamic shared memory for kernels that opt-in to larger shared memory.
// Returns 0 on success, non-zero on error.
inline int configureDynamicSharedMemory(int gpuIndex)
{
    int maxShared = 0;
    cudaError_t err = cudaDeviceGetAttribute(&maxShared, cudaDevAttrMaxSharedMemoryPerBlockOptin, gpuIndex);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to query device shared memory attribute: %s\n", cudaGetErrorString(err));
        return 1;
    }

    printf("Dynamic shared memory: requesting %d bytes (GPU max opt-in: %d bytes)\n", MAX_SHARED_MEMORY_SIZE, maxShared);
    
    if (MAX_SHARED_MEMORY_SIZE > maxShared) {
        fprintf(stderr, "ERROR: Requested shared memory (%d bytes) exceeds device max (%d bytes)\n", MAX_SHARED_MEMORY_SIZE, maxShared);
        fprintf(stderr, "Try reducing BLOCK_NX/NY/NZ in memory_layout.h\n");
        return 1;
    }

    cudaError_t sharedMemErr = cudaFuncSetAttribute(gpuMomCollisionStream, cudaFuncAttributeMaxDynamicSharedMemorySize, MAX_SHARED_MEMORY_SIZE);
    if (sharedMemErr != cudaSuccess) {
        fprintf(stderr, "ERROR: Failed to set dynamic shared memory attribute: %s\n", cudaGetErrorString(sharedMemErr));
        return 1;
    }

    printf("Successfully configured dynamic shared memory: %d bytes\n", MAX_SHARED_MEMORY_SIZE);
    return 0;
}
#endif //DYNAMIC_SHARED_MEMORY

/* ======================= CUDA TIMING UTILITIES ========================= */

/**
 * @brief Initializes CUDA events for timing.
 * @param start Reference to the start event.
 * @param stop Reference to the stop event.
 * @param start_step Reference to the start event for a step.
 * @param stop_step Reference to the stop event for a step.
 */
inline void initializeCudaEvents(cudaEvent_t &start, cudaEvent_t &stop, cudaEvent_t &start_step, cudaEvent_t &stop_step) {
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    checkCudaErrors(cudaEventCreate(&start_step));
    checkCudaErrors(cudaEventCreate(&stop_step));

    checkCudaErrors(cudaEventRecord(start, 0));
    checkCudaErrors(cudaEventRecord(start_step, 0));
}

/**
 * @brief Records the elapsed time between two CUDA events and calculates MLUPS.
 * @param start_step Reference to the start event for a step.
 * @param stop_step Reference to the stop event for a step.
 * @param step Current simulation step.
 * @param ini_step Initial simulation step.
 * @param numberLbmNodes Number of LBM nodes being updated per step.
 * @return The calculated MLUPS (Mega Lattice Updates Per Second).
 */
inline dfloat recordElapsedTime(cudaEvent_t &start_step, cudaEvent_t &stop_step, int step, int ini_step, size_t numberLbmNodes) {
    checkCudaErrors(cudaEventRecord(stop_step, 0));
    checkCudaErrors(cudaEventSynchronize(stop_step));
    
    float elapsedTime;
    checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start_step, stop_step));
    elapsedTime *= 0.001;

    size_t nodesUpdatedSync = (step - ini_step) * numberLbmNodes;
    dfloat MLUPS = (nodesUpdatedSync / 1e6) / elapsedTime;
    return MLUPS;
}

#endif //__CUDA_UTILS_H