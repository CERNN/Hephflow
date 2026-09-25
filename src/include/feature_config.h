/**
 *  @file feature_config.h
 *  @brief Feature flags and model configuration macros
 *  @version 0.4.0
 *  @date 27/12/2025
 */

#ifndef __FEATURE_CONFIG_H
#define __FEATURE_CONFIG_H

/* ====================== POPULATION STREAMING ====================== */

// Select exactly one interface streaming layout at compile time. The build
// script defines one of these; AA remains the fallback for direct builds.
#if defined(STREAMING_AA_STD) || defined(STREAMING_AA_ESOTERIC_PULL) || defined(STREAMING_AA_ESOTERIC_TWIST) || defined(STREAMING_AA_ESOTERIC_PUSH)
    #ifndef STREAMING_AA
        #define STREAMING_AA
    #endif
#endif

#if defined(STREAMING_AA) && defined(STREAMING_AB)
    #error "Select exactly one population streaming scheme: STREAMING_AA or STREAMING_AB"
#elif !defined(STREAMING_AA) && !defined(STREAMING_AB)
    #define STREAMING_AA
#endif

#if (defined(STREAMING_AA_STD) + defined(STREAMING_AA_ESOTERIC_PULL) + defined(STREAMING_AA_ESOTERIC_TWIST) + defined(STREAMING_AA_ESOTERIC_PUSH)) > 1
    #error "Select only one AA streaming method"
#endif

#if defined(STREAMING_AA) && !defined(STREAMING_AA_STD) && !defined(STREAMING_AA_ESOTERIC_PULL) && !defined(STREAMING_AA_ESOTERIC_TWIST) && !defined(STREAMING_AA_ESOTERIC_PUSH)
    #define STREAMING_AA_STD
#endif

#if defined(STREAMING_AB) && (defined(STREAMING_AA_ESOTERIC_PULL) || defined(STREAMING_AA_ESOTERIC_TWIST) || defined(STREAMING_AA_ESOTERIC_PUSH))
    #error "AA esoteric streaming methods cannot be combined with STREAMING_AB"
#endif

#if (defined(STREAMING_AA_ESOTERIC_PULL) || defined(STREAMING_AA_ESOTERIC_TWIST) || defined(STREAMING_AA_ESOTERIC_PUSH)) && !defined(D3Q19)
    #error "AA esoteric streaming methods currently support D3Q19 only"
#endif

#if defined(STREAMING_AB) && defined(OVERLAP_HALO_TRANSFERS)
    #error "STREAMING_AB is not compatible with OVERLAP_HALO_TRANSFERS"
#endif

/* ============================== MODEL MACROS ============================= */

#if defined(POWERLAW) || defined(BINGHAM) || defined(BI_VISCOSITY) || defined(HERSCHEL_BULKLEY) || defined(KEE_TURCOTEE)
    #define OMEGA_FIELD
    #define NON_NEWTONIAN_FLUID
    #define COMPUTE_SHEAR
#endif  //POWERLAW || BINGHAM || BI_VISCOSITY || HERSCHEL_BULKLEY || KEE_TURCOTEE

#if defined(LAMBDA_MODEL)
    #define OMEGA_FIELD
    #define NON_NEWTONIAN_FLUID
    #define COMPUTE_SHEAR
#endif //LAMBDA_MODEL

#if defined(LES_MODEL)
    #define OMEGA_FIELD
    #define COMPUTE_SHEAR
#endif //LES_MODEL

#if defined(PHI_DIST)
    #define COMPUTE_SHEAR
    #define NON_NEWTONIAN_FLUID
    #define OMEGA_FIELD
#endif //PHI_DIST


#if defined(HO_RR) || defined(HOME_LBM)
    #define HIGH_ORDER_COLLISION
#endif // HO_RR || HOME_LBM

/* ======================= GHOST INTERFACE OPTIMIZATION ====================== */

// Cooperative ghost-population load and save selections are independent and
// default off. Select at most one direction set for loads and one for saves.
// Available suffixes: X, Y, Z, XY, XZ, YZ, and ALL.
// #define COOPERATIVE_GHOST_LOAD_X
// #define COOPERATIVE_GHOST_SAVE_X

#if (defined(COOPERATIVE_GHOST_LOAD_X) + defined(COOPERATIVE_GHOST_LOAD_Y) + \
     defined(COOPERATIVE_GHOST_LOAD_Z) + defined(COOPERATIVE_GHOST_LOAD_XY) + \
     defined(COOPERATIVE_GHOST_LOAD_XZ) + defined(COOPERATIVE_GHOST_LOAD_YZ) + \
     defined(COOPERATIVE_GHOST_LOAD_ALL)) > 1
    #error "Select at most one cooperative ghost-load direction set"
#endif

#if (defined(COOPERATIVE_GHOST_SAVE_X) + defined(COOPERATIVE_GHOST_SAVE_Y) + \
     defined(COOPERATIVE_GHOST_SAVE_Z) + defined(COOPERATIVE_GHOST_SAVE_XY) + \
     defined(COOPERATIVE_GHOST_SAVE_XZ) + defined(COOPERATIVE_GHOST_SAVE_YZ) + \
     defined(COOPERATIVE_GHOST_SAVE_ALL)) > 1
    #error "Select at most one cooperative ghost-save direction set"
#endif

#if defined(COOPERATIVE_GHOST_LOAD_X)
    #define COOPERATIVE_GHOST_LOAD_MASK 1
#elif defined(COOPERATIVE_GHOST_LOAD_Y)
    #define COOPERATIVE_GHOST_LOAD_MASK 2
#elif defined(COOPERATIVE_GHOST_LOAD_Z)
    #define COOPERATIVE_GHOST_LOAD_MASK 4
#elif defined(COOPERATIVE_GHOST_LOAD_XY)
    #define COOPERATIVE_GHOST_LOAD_MASK 3
#elif defined(COOPERATIVE_GHOST_LOAD_XZ)
    #define COOPERATIVE_GHOST_LOAD_MASK 5
#elif defined(COOPERATIVE_GHOST_LOAD_YZ)
    #define COOPERATIVE_GHOST_LOAD_MASK 6
#elif defined(COOPERATIVE_GHOST_LOAD_ALL)
    #define COOPERATIVE_GHOST_LOAD_MASK 7
#endif

#if defined(COOPERATIVE_GHOST_SAVE_X)
    #define COOPERATIVE_GHOST_SAVE_MASK 1
#elif defined(COOPERATIVE_GHOST_SAVE_Y)
    #define COOPERATIVE_GHOST_SAVE_MASK 2
#elif defined(COOPERATIVE_GHOST_SAVE_Z)
    #define COOPERATIVE_GHOST_SAVE_MASK 4
#elif defined(COOPERATIVE_GHOST_SAVE_XY)
    #define COOPERATIVE_GHOST_SAVE_MASK 3
#elif defined(COOPERATIVE_GHOST_SAVE_XZ)
    #define COOPERATIVE_GHOST_SAVE_MASK 5
#elif defined(COOPERATIVE_GHOST_SAVE_YZ)
    #define COOPERATIVE_GHOST_SAVE_MASK 6
#elif defined(COOPERATIVE_GHOST_SAVE_ALL)
    #define COOPERATIVE_GHOST_SAVE_MASK 7
#endif

#if defined(COOPERATIVE_GHOST_LOAD_MASK) || defined(COOPERATIVE_GHOST_SAVE_MASK)
    #if !defined(D3Q19)
        #error "Cooperative ghost transfer currently supports D3Q19 only"
    #endif
    #if defined(STREAMING_AA_ESOTERIC_PULL) || defined(STREAMING_AA_ESOTERIC_TWIST) || defined(STREAMING_AA_ESOTERIC_PUSH)
        #error "Cooperative ghost transfer is not compatible with esoteric AA streaming"
    #endif
#endif

// Enable shared memory staging for ghost interface I/O.
// All 512 threads cooperatively load/store ghost face data through s_pop,
// replacing scattered surface-thread-only global memory access with coalesced
// block-wide transfers. Adds 3 extra __syncthreads() per timestep.
// #define USE_SHARED_GHOST_STAGING

#endif //__FEATURE_CONFIG_H
