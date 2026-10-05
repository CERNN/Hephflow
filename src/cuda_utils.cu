#include "mlbm.cuh"
#include "include/cuda_utils.cuh"

#ifdef DYNAMIC_SHARED_MEMORY
int configureDynamicSharedMemory(int gpuIndex)
{
    int maxShared = 0;
    cudaError_t error = cudaDeviceGetAttribute(
        &maxShared, cudaDevAttrMaxSharedMemoryPerBlockOptin, gpuIndex);
    if (error != cudaSuccess) {
        fprintf(stderr, "Failed to query device shared memory attribute: %s\n",
                cudaGetErrorString(error));
        return 1;
    }

    printf("Dynamic shared memory: requesting %zu bytes "
           "(GPU max opt-in: %d bytes)\n", MAX_SHARED_MEMORY_SIZE, maxShared);
    if (MAX_SHARED_MEMORY_SIZE > (size_t)maxShared) {
        fprintf(stderr, "ERROR: Requested shared memory (%zu bytes) exceeds "
                "device max (%d bytes)\n", MAX_SHARED_MEMORY_SIZE, maxShared);
        return 1;
    }

    cudaError_t sharedMemoryError = cudaFuncSetAttribute(
        gpuMomCollisionStream, cudaFuncAttributeMaxDynamicSharedMemorySize,
        MAX_SHARED_MEMORY_SIZE);
    if (sharedMemoryError != cudaSuccess) {
        fprintf(stderr, "ERROR: Failed to set dynamic shared memory "
                "attribute: %s\n", cudaGetErrorString(sharedMemoryError));
        return 1;
    }
    return 0;
}
#endif
