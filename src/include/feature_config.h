/**
 *  @file feature_config.h
 *  @brief Feature flags and model configuration macros
 *  @version 0.4.0
 *  @date 27/12/2025
 */

#ifndef __FEATURE_CONFIG_H
#define __FEATURE_CONFIG_H

/* ============================== MODEL MACROS ============================= */

#if defined(POWERLAW) || defined(BINGHAM) || defined(BI_VISCOSITY)
    #define OMEGA_FIELD
    #define NON_NEWTONIAN_FLUID
    #define COMPUTE_SHEAR
#endif  //POWERLAW || BINGHAM || BI_VISCOSITY

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

// Enable shared memory staging for ghost interface I/O.
// All 512 threads cooperatively load/store ghost face data through s_pop,
// replacing scattered surface-thread-only global memory access with coalesced
// block-wide transfers. Adds 3 extra __syncthreads() per timestep.
// #define USE_SHARED_GHOST_STAGING

#endif //__FEATURE_CONFIG_H