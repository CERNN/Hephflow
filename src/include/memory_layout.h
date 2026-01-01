/**
 *  @file memory_layout.h
 *  @brief Memory size calculations and block layout definitions
 *  @version 0.4.0
 *  @date 27/12/2025
 */

#ifndef __MEMORY_LAYOUT_H
#define __MEMORY_LAYOUT_H

#include "var_types.h"    // for dfloat type
#include "cuda_utils.h"   // for BlockDim
#include "utils.h"        // for myMax
#include "../arrayIndex.h"   // for domain size definitions

/* ============================= VELOCITY SETS ============================= */

#ifdef D3Q19
    #include "fragments/velocitySets/D3Q19.inc"
#endif //D3Q19
#ifdef D3Q27
    #include "fragments/velocitySets/D3Q27.inc"
#endif //D3Q27

// #define SECOND_DIST
#ifdef D3G7
    #include "fragments/velocitySets/D3G7.inc"
#endif //D3G7
#ifdef D3G19
    #include "fragments/velocitySets/D3G19.inc"
#endif //D3G19

/* ========================== MEMORY SIZE CONSTANTS ======================== */

constexpr size_t BYTES_PER_GB = (1 << 30);
constexpr size_t BYTES_PER_MB = (1 << 20);

/* ====================== SHARED MEMORY CONFIGURATION ====================== */

// Architecture-specific maximum shared memory sizes (for opt-in dynamic shared memory)
// These are the max bytes available when using cudaFuncAttributeMaxDynamicSharedMemorySize
#if (defined(SM_90) || defined(SM_100) || defined(SM_120))
    constexpr size_t GPU_MAX_SHARED_MEMORY = 232448;  // sm90+
#elif (defined(SM_80)) || (defined(SM_87))
    constexpr size_t GPU_MAX_SHARED_MEMORY = 166912;  // sm80/87
#elif (defined(SM_86) || defined(SM_89))
    constexpr size_t GPU_MAX_SHARED_MEMORY = 101376;  // sm86/89
#else
    constexpr size_t GPU_MAX_SHARED_MEMORY = 49152;   // default (sm75 and below)
#endif

// Note: Q-dependent calculations are in definitions.h after velocity sets are included

constexpr int SHARED_MEMORY_ELEMENT_SIZE = sizeof(dfloat) * (Q - 1);
#ifdef SHARED_MEMORY_LIMIT
    constexpr size_t MAX_ELEMENTS_IN_BLOCK = SHARED_MEMORY_LIMIT / SHARED_MEMORY_ELEMENT_SIZE;
#else
    constexpr int MAX_ELEMENTS_IN_BLOCK = 48128 / SHARED_MEMORY_ELEMENT_SIZE;
#endif //SHARED_MEMORY_LIMIT

constexpr BlockDim optimalBlockDimArray = findOptimalBlockDimensions(MAX_ELEMENTS_IN_BLOCK);

/* =========================== BLOCK DEFINITIONS =========================== */

#define BLOCK_NX 8
#define BLOCK_NY 8
#define BLOCK_NZ 8

#define BLOCK_LBM_SIZE (BLOCK_NX * BLOCK_NY * BLOCK_NZ)

const size_t BLOCK_LBM_SIZE_POP = BLOCK_LBM_SIZE * (Q - 1);

const size_t BLOCK_FACE_XY = BLOCK_NX * BLOCK_NY;
const size_t BLOCK_FACE_XZ = BLOCK_NX * BLOCK_NZ;
const size_t BLOCK_FACE_YZ = BLOCK_NY * BLOCK_NZ;
const size_t BLOCK_GHOST_SIZE = BLOCK_FACE_XY + BLOCK_FACE_XZ + BLOCK_FACE_YZ;

const size_t BLOCK_SIZE = BLOCK_LBM_SIZE + BLOCK_GHOST_SIZE;
//const size_t BLOCK_SIZE = (BLOCK_NX + 1) * (BLOCK_NY + 1) * (BLOCK_NZ + 1);

/* ========================= DOMAIN BLOCK CALCULATIONS ===================== */

//TODO: in order to support domains with size that are not multiple of the block size need fix the index function where the transfer populations occur for periodic domains. 
const size_t NUM_BLOCK_X = (NX / BLOCK_NX) + (NX % BLOCK_NX > 0 ? 1 : 0);
const size_t NUM_BLOCK_Y = (NY / BLOCK_NY) + (NY % BLOCK_NY > 0 ? 1 : 0);
const size_t NUM_BLOCK_Z = (NZ / BLOCK_NZ) + (NZ % BLOCK_NZ > 0 ? 1 : 0);

const size_t NUM_BLOCK = NUM_BLOCK_X * NUM_BLOCK_Y * NUM_BLOCK_Z;

const size_t NUMBER_LBM_NODES = NUM_BLOCK * BLOCK_LBM_SIZE;
const size_t NUMBER_GHOST_FACE_XY = BLOCK_NX*BLOCK_NY*NUM_BLOCK_X*NUM_BLOCK_Y*NUM_BLOCK_Z;
const size_t NUMBER_GHOST_FACE_XZ = BLOCK_NX*BLOCK_NZ*NUM_BLOCK_X*NUM_BLOCK_Y*NUM_BLOCK_Z;
const size_t NUMBER_GHOST_FACE_YZ = BLOCK_NY*BLOCK_NZ*NUM_BLOCK_X*NUM_BLOCK_Y*NUM_BLOCK_Z;

/* ======================== MEMORY ALLOCATION SIZES ======================== */

const size_t MEM_SIZE_BLOCK_LBM = sizeof(dfloat) * BLOCK_LBM_SIZE * NUMBER_MOMENTS;
const size_t MEM_SIZE_BLOCK_GHOST = sizeof(dfloat) * BLOCK_GHOST_SIZE * Q;
const size_t MEM_SIZE_BLOCK_TOTAL = MEM_SIZE_BLOCK_GHOST + MEM_SIZE_BLOCK_LBM;

const size_t NUMBER_LBM_POP_NODES = NX * NY * NZ;

//memory size
const size_t MEM_SIZE_SCALAR = sizeof(dfloat) * NUMBER_LBM_NODES;
const size_t MEM_SIZE_POP = sizeof(dfloat) * NUMBER_LBM_POP_NODES * Q;
const size_t MEM_SIZE_MOM = sizeof(dfloat) * NUMBER_LBM_NODES * NUMBER_MOMENTS;

const size_t MEM_SIZE_MAP_BC = sizeof(uint32_t) * NUMBER_LBM_NODES;

/* ======================== MEMORY ALLOCATION SIZES ======================== */



/* ===================== GRADIENT COMPUTATION BLOCK SIZES ================== */

#ifdef COMPUTE_VEL_GRADIENT_FINITE_DIFFERENCE
    #define HALO_SIZE 1
    const size_t VEL_GRAD_BLOCK_SIZE = (BLOCK_NX + 2 * HALO_SIZE) * (BLOCK_NY + 2 * HALO_SIZE) * (BLOCK_NZ + 2 * HALO_SIZE) * 3;
#else
    const size_t VEL_GRAD_BLOCK_SIZE = 0;
#endif //COMPUTE_VEL_GRADIENT_FINITE_DIFFERENCE

#ifdef COMPUTE_CONF_GRADIENT_FINITE_DIFFERENCE
    #define HALO_SIZE 1
    const size_t CONFORMATION_GRAD_BLOCK_SIZE = (BLOCK_NX + 2 * HALO_SIZE) * (BLOCK_NY + 2 * HALO_SIZE) * (BLOCK_NZ + 2 * HALO_SIZE) * 6;
#else
    const size_t CONFORMATION_GRAD_BLOCK_SIZE = 0;
#endif //COMPUTE_CONF_GRADIENT_FINITE_DIFFERENCE

constexpr int MAX_SHARED_MEMORY_SIZE = myMax(BLOCK_LBM_SIZE_POP, myMax(VEL_GRAD_BLOCK_SIZE, CONFORMATION_GRAD_BLOCK_SIZE))*sizeof(dfloat);

#endif //__MEMORY_LAYOUT_H