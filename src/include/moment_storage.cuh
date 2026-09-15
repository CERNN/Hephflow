/**
 * @file moment_storage.cuh
 * @brief Inlined access to persistent FP32/mixed-precision moment storage.
 */
#ifndef HEPHFLOW_MOMENT_STORAGE_CUH
#define HEPHFLOW_MOMENT_STORAGE_CUH

#include <cuda_fp16.h>
#include "moment_storage_config.h"

__host__ __device__ __forceinline__
size_t momentStoragePlaneOffsetBytes(int moment)
{
    size_t offset = 0;
    for (int plane = 0; plane < moment; ++plane)
        offset += BLOCK_LBM_SIZE * momentStorageElementBytes(plane);
    return offset;
}

__host__ __device__ __forceinline__
size_t momentStorageByteOffset(size_t logicalIndex, int moment)
{
    const size_t blockSpan = BLOCK_LBM_SIZE * NUMBER_MOMENTS;
    const size_t block = logicalIndex / blockSpan;
    const size_t localNode = logicalIndex % BLOCK_LBM_SIZE;
    return block * BLOCK_LBM_SIZE * MOM_STORAGE_BYTES_PER_NODE
         + momentStoragePlaneOffsetBytes(moment)
         + localNode * momentStorageElementBytes(moment);
}

__host__ __device__ __forceinline__
dfloat loadMoment(const dfloat* storage, size_t logicalIndex, int moment)
{
#if !USE_FP16_SECOND_ORDER_MOMENTS
    (void)moment;
    return storage[logicalIndex];
#else
    const unsigned char* bytes = reinterpret_cast<const unsigned char*>(storage);
    const size_t offset = momentStorageByteOffset(logicalIndex, moment);
    if (momentStorageUsesHalf(moment)) {
        const __half encoded = *reinterpret_cast<const __half*>(bytes + offset);
        return static_cast<dfloat>(__half2float(encoded)) / momentStorageScale(moment);
    }
    return *reinterpret_cast<const dfloat*>(bytes + offset);
#endif
}

__host__ __device__ __forceinline__
void storeMoment(dfloat* storage, size_t logicalIndex, int moment, dfloat value)
{
#if !USE_FP16_SECOND_ORDER_MOMENTS
    (void)moment;
    storage[logicalIndex] = value;
#else
    unsigned char* bytes = reinterpret_cast<unsigned char*>(storage);
    const size_t offset = momentStorageByteOffset(logicalIndex, moment);
    if (momentStorageUsesHalf(moment)) {
        *reinterpret_cast<__half*>(bytes + offset) =
            __float2half_rn(static_cast<float>(value * momentStorageScale(moment)));
    } else {
        *reinterpret_cast<dfloat*>(bytes + offset) = value;
    }
#endif
}

// Compact device bytes are copied into the start of an FP32 host buffer.
// Reverse traversal makes the in-place expansion overlap-safe.
inline void decodeMomentStorageInPlace(dfloat* buffer, size_t numberBlocks)
{
#if USE_FP16_SECOND_ORDER_MOMENTS
    const size_t blockSpan = BLOCK_LBM_SIZE * NUMBER_MOMENTS;
    for (size_t block = numberBlocks; block-- > 0;) {
        for (int moment = static_cast<int>(NUMBER_MOMENTS); moment-- > 0;) {
            for (size_t node = BLOCK_LBM_SIZE; node-- > 0;) {
                const size_t logical = block * blockSpan
                    + static_cast<size_t>(moment) * BLOCK_LBM_SIZE + node;
                buffer[logical] = loadMoment(buffer, logical, moment);
            }
        }
    }
#else
    (void)buffer;
    (void)numberBlocks;
#endif
}

#endif
