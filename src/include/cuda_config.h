/**
 *  @file cuda_config.h
 *  @brief CUDA-specific configuration and settings
 *  @version 0.4.0
 *  @date 27/12/2025
 */

#ifndef __CUDA_CONFIG_H
#define __CUDA_CONFIG_H

/* ============================= CUDA SETTINGS ============================= */

const int N_THREADS = (NX%64?((NX%32||(NX<32))?NX:32):64); // NX or 32 or 64 
                                    // multiple of 32 for better performance.
const int CURAND_SEED = 0;          // seed for random numbers for CUDA
constexpr float CURAND_STD_DEV = 0.5_df; // standard deviation for random numbers 
                                    // in normal distribution

/* ======================== DYNAMIC SHARED MEMORY MACROS =================== */

// Enable dynamic shared memory to allow larger block sizes
// Comment out to use statically allocated shared memory (default CUDA limit ~48KB)
#define DYNAMIC_SHARED_MEMORY 

#ifdef DYNAMIC_SHARED_MEMORY
    // Validate that we have enough shared memory for the requested block size
    static_assert(MAX_SHARED_MEMORY_SIZE <= GPU_MAX_SHARED_MEMORY, 
        "MAX_SHARED_MEMORY_SIZE exceeds GPU_MAX_SHARED_MEMORY for this architecture");
    #define DYNAMIC_SHARED_MEMORY_PARAMS ,MAX_SHARED_MEMORY_SIZE
#else
    // Static shared memory - limited to ~48KB
    static_assert(MAX_SHARED_MEMORY_SIZE <= 49152, 
        "MAX_SHARED_MEMORY_SIZE exceeds 48KB static limit. Enable DYNAMIC_SHARED_MEMORY.");
    #define DYNAMIC_SHARED_MEMORY_PARAMS
#endif //DYNAMIC_SHARED_MEMORY

#endif //__CUDA_CONFIG_H