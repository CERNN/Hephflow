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

//FUNCTION DECLARATION MACROS
#ifdef DYNAMIC_SHARED_MEMORY
    #define DYNAMIC_SHARED_MEMORY_PARAMS ,MAX_SHARED_MEMORY_SIZE
#else
    #define DYNAMIC_SHARED_MEMORY_PARAMS
#endif //DYNAMIC_SHARED_MEMORY

#endif //__CUDA_CONFIG_H