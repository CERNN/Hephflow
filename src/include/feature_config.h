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

// FENE_P historically enabled the entire conformation subsystem. Keep it as
// a compatibility alias while allowing cases to use the model-neutral name.
#if defined(FENE_P) && !defined(VISCOELASTIC_MODEL)
    #define VISCOELASTIC_MODEL
#endif

#if defined(VISCOELASTIC_MODEL) && !defined(CONFORMATION_TENSOR)
    #define CONFORMATION_TENSOR
#endif

#if defined(CONFORMATION_TENSOR)
    // Canonical conformation representation selector.  The legacy symbolic
    // defines remain accepted because existing cases and validation scripts
    // pass CONFORMATION_LOG on the compiler command line.
    #define CONF_REP_STANDARD 0
    #define CONF_REP_LOG      1
    #define CONF_REP_SQRT     2

    #if (defined(CONFORMATION_STANDARD) + defined(CONFORMATION_LOG) + \
         defined(CONFORMATION_SQRT)) > 1
        #error "Select exactly one conformation representation"
    #endif

    #ifndef CONF_REPRESENTATION
        #if defined(CONFORMATION_LOG)
            #define CONF_REPRESENTATION CONF_REP_LOG
        #elif defined(CONFORMATION_SQRT)
            #define CONF_REPRESENTATION CONF_REP_SQRT
        #else
            #define CONF_REPRESENTATION CONF_REP_STANDARD
        #endif
    #endif

    #if CONF_REPRESENTATION == CONF_REP_STANDARD
        #if defined(CONFORMATION_LOG) || defined(CONFORMATION_SQRT)
            #error "CONF_REPRESENTATION conflicts with a legacy representation define"
        #endif
        #ifndef CONFORMATION_STANDARD
            #define CONFORMATION_STANDARD
        #endif
    #elif CONF_REPRESENTATION == CONF_REP_LOG
        #if defined(CONFORMATION_STANDARD) || defined(CONFORMATION_SQRT)
            #error "CONF_REPRESENTATION conflicts with a legacy representation define"
        #endif
        #ifndef CONFORMATION_LOG
            #define CONFORMATION_LOG
        #endif
    #elif CONF_REPRESENTATION == CONF_REP_SQRT
        #if defined(CONFORMATION_STANDARD) || defined(CONFORMATION_LOG)
            #error "CONF_REPRESENTATION conflicts with a legacy representation define"
        #endif
        #ifndef CONFORMATION_SQRT
            #define CONFORMATION_SQRT
        #endif
    #else
        #error "Invalid CONF_REPRESENTATION"
    #endif

    #define CONF_FORCE_GRAD_EXISTING           0
    #define CONF_FORCE_GRAD_FRECHET            1
    #define CONF_FORCE_GRAD_DIRECT_STRESS_HALO 2

    // Compatibility alias for validation scripts and cases written during
    // development of the direct-halo formulation.
    #define CONF_FORCE_GRAD_DIRECT_PHYSICAL_HALO CONF_FORCE_GRAD_DIRECT_STRESS_HALO

    #ifndef CONF_FORCE_GRADIENT
        #if CONF_REPRESENTATION == CONF_REP_LOG
            #define CONF_FORCE_GRADIENT CONF_FORCE_GRAD_FRECHET
        #elif CONF_REPRESENTATION == CONF_REP_SQRT
            #define CONF_FORCE_GRADIENT CONF_FORCE_GRAD_DIRECT_STRESS_HALO
        #else
            #define CONF_FORCE_GRADIENT CONF_FORCE_GRAD_EXISTING
        #endif
    #endif

    #if CONF_FORCE_GRADIENT < CONF_FORCE_GRAD_EXISTING || \
        CONF_FORCE_GRADIENT > CONF_FORCE_GRAD_DIRECT_STRESS_HALO
        #error "Invalid CONF_FORCE_GRADIENT"
    #endif
    #if CONF_REPRESENTATION == CONF_REP_STANDARD && \
        CONF_FORCE_GRADIENT == CONF_FORCE_GRAD_FRECHET
        #error "Frechet force gradients are only defined for log conformation"
    #endif
    #if CONF_REPRESENTATION == CONF_REP_SQRT && \
        CONF_FORCE_GRADIENT != CONF_FORCE_GRAD_DIRECT_STRESS_HALO
        #error "Square-root conformation requires the direct polymer-stress halo force path"
    #endif

    #if CONF_FORCE_GRADIENT == CONF_FORCE_GRAD_DIRECT_STRESS_HALO
        #define CONFORMATION_DIRECT_STRESS_HALO
    #endif

    // Transformed representations need immutable neighboring physical-A values
    // to map the auxiliary LBM diffusion back to the same physical equation as
    // the standard representation.  The direct force path reuses this halo.
    #if defined(CONFORMATION_LOG) || defined(CONFORMATION_SQRT) || \
        defined(CONFORMATION_DIRECT_STRESS_HALO)
        #define CONFORMATION_PHYSICAL_A_HALO
        #if HEPHFLOW_N_GPUS != 1
            #error "The physical-A halo is currently single-GPU only"
        #endif
        #if defined(PHI_DIST)
            #error "The physical-A halo has not been audited for phase blending"
        #endif
        #if defined(CURVED_BOUNDARY_CONDITION)
            #error "The physical-A halo has not been audited for curved boundaries"
        #endif
        #if defined(OVERLAP_HALO_TRANSFERS)
            #error "The physical-A halo is incompatible with split/overlapped launches"
        #endif
    #endif

    #if defined(VE_FORCE_LEGACY_COEFFICIENT) && defined(VE_FORCE_FULL_DIVERGENCE)
        #error "Select exactly one viscoelastic force policy"
    #elif !defined(VE_FORCE_LEGACY_COEFFICIENT) && !defined(VE_FORCE_FULL_DIVERGENCE)
        #define VE_FORCE_LEGACY_COEFFICIENT
    #endif
#endif

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

    /* Cases select the high-level phase method in model.inc. */
    #if defined(METHOD_CAHN_HILLIARD) && defined(METHOD_ALLEN_CAHN)
        #error "Select only one phase method in model.inc"
    #endif
    #if defined(METHOD_CAHN_HILLIARD) && \
        !defined(PHASE_TRANSPORT_CAHN_HILLIARD) && \
        !defined(PHASE_TRANSPORT_CONSERVATIVE_ALLEN_CAHN)
        #define PHASE_TRANSPORT_CAHN_HILLIARD
    #endif
    #if defined(METHOD_ALLEN_CAHN) && \
        !defined(PHASE_TRANSPORT_CAHN_HILLIARD) && \
        !defined(PHASE_TRANSPORT_CONSERVATIVE_ALLEN_CAHN)
        #define PHASE_TRANSPORT_CONSERVATIVE_ALLEN_CAHN
    #endif

    /* Default force; define PHASE_FORCE_CSF explicitly to override it. */
    #if !defined(PHASE_FORCE_CHEMICAL_POTENTIAL) && \
        !defined(PHASE_FORCE_CSF)
        #define PHASE_FORCE_CHEMICAL_POTENTIAL
    #endif

    #if (defined(PHASE_TRANSPORT_CAHN_HILLIARD) + \
         defined(PHASE_TRANSPORT_CONSERVATIVE_ALLEN_CAHN)) != 1
        #error "PHI_DIST requires exactly one phase transport model"
    #endif
    #if (defined(PHASE_FORCE_CHEMICAL_POTENTIAL) + \
         defined(PHASE_FORCE_CSF)) != 1
        #error "PHI_DIST requires exactly one phase capillary-force model"
    #endif
    #if defined(PHASE_TRANSPORT_CONSERVATIVE_ALLEN_CAHN) && \
        !defined(INTERFACE_SHARPENING)
        #error "Conservative Allen-Cahn requires INTERFACE_SHARPENING"
    #endif

    #if defined(PHASE_TRANSPORT_CONSERVATIVE_ALLEN_CAHN)
        static_assert(M_AC >= 0.0_df, "M_AC must be non-negative");
        static_assert(PHI_INTERFACE_WIDTH > 0.0_df,
                      "PHI_INTERFACE_WIDTH must be positive");
        static_assert(PHI_NORMAL_EPS > 0.0_df,
                      "PHI_NORMAL_EPS must be positive");
    #endif

    #if defined(PHASE_TRANSPORT_CAHN_HILLIARD) || \
        defined(PHASE_FORCE_CHEMICAL_POTENTIAL)
        #define PHASE_NEEDS_CHEMICAL_POTENTIAL
    #endif
    #if defined(PHASE_TRANSPORT_CONSERVATIVE_ALLEN_CAHN) || \
        defined(PHASE_FORCE_CSF)
        #define PHASE_NEEDS_NORMAL_HALOS
    #endif

    #ifndef PHI_CLAMP_ENABLED
        #define PHI_CLAMP_ENABLED 1
    #endif
    #if (PHI_CLAMP_ENABLED != 0) && (PHI_CLAMP_ENABLED != 1)
        #error "PHI_CLAMP_ENABLED must be 0 or 1"
    #endif
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
