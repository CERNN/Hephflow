/**
 *  @file nnf_types.h
 *  Contributors history:
 *  @author Marco Aurelio Ferrari (e.marcoferrari@utfpr.edu.br)
 *  @brief Type definitions for non-Newtonian fluids
 *  @version 0.1.0
 *  @date 02/01/2026
 */

#ifndef __NNF_TYPES_H
#define __NNF_TYPES_H

#include "../include/var_types.h"

enum fluidType { FLUID_POWERLAW, FLUID_BINGHAM, FLUID_BI_VISCOSITY, FLUID_KEE_TURCOTEE, FLUID_HERSCHEL_BULKLEY, FLUID_THIXO, FLUID_NEWTONIAN };
enum thixoModel { THIXO_MOORE1959, THIXO_WORRALL1964, THIXO_HOUSKA1980, THIXO_TOORMAN1997 };

struct fluidProps {
    fluidType type;
    bool hasLambda;
    union {
        struct { dfloat n_index, k_consistency, gamma_0; } powerlaw;
        struct { dfloat s_y, omega_p; } bingham;
        struct { dfloat n_index, k_consistency, gamma_0, s_y; } hb;
        struct { dfloat s_y, visc_ratio, eta_y, tau_y, omega_y, omega_p, gamma_c; } bi;
        struct { dfloat s_y, t1, eta_0; } kee;
        struct { 
            enum thixoModel model;    
            union{
                struct { dfloat k1, k2, lambda_0, eta_0;} moore1959;
                struct { dfloat k1, s_y_0, eta_0;} worrall1964;
                struct { dfloat k1, k2, m_exponent, s_y_0, s_y_inf, k_consistency, n_index;} houska1980;
                struct { dfloat k1, k2, a_exponent, b_exponent, s_y_0, eta_0;} toorman1997;
            } u;
        } thixo;
    } u;
};

// ============================================================================
// VISCOELASTIC FLUID TYPES
// ============================================================================

/**
 * @brief Supported viscoelastic constitutive models.
 *
 * All models evolve the polymer conformation tensor A via:
 *   DA/Dt = R_ij(A, model_params) + (A.du + du^T.A)
 * where R_ij is the model-specific relaxation term dispatched by
 * calcVeRelaxationTerm() in nnf.h.
 *
 * Adding a new model:
 *  1. Add an entry here.
 *  2. Add its parameters to the veFluidProps union below.
 *  3. Add a case to calcVeRelaxationTerm() in nnf.h.
 *  4. Provide a makeCasePhaseProps*() factory in the case's constants.inc.
 */
enum veFluidType {
    VE_NEWTONIAN,           ///< No polymer (eta_p = 0); use for the Newtonian phase in multiphase
    VE_OLDROYD_B,           ///< Oldroyd-B: linear, infinite extensibility
    VE_FENE_P,              ///< FENE-P: finitely extensible nonlinear (Peterlin closure)
    VE_GIESEKUS,            ///< Giesekus: quadratic relaxation, shear-thinning polymers
    VE_PTT_LINEAR,          ///< Phan-Thien-Tanner with linear trace function
    VE_PTT_EXPONENTIAL,     ///< Phan-Thien-Tanner with exponential trace function
};

/**
 * @brief Six independent components of the symmetric 3×3 relaxation term R_ij.
 *
 * Returned by calcVeRelaxationTerm(). The upper-convected derivative contribution
 * (A.du + du^T.A) is added separately by the caller in conformation_evolution.inc.
 */
struct veRelaxTerm {
    dfloat xx, yy, zz, xy, xz, yz;
    __host__ __device__ veRelaxTerm()
        : xx(0.0f), yy(0.0f), zz(0.0f), xy(0.0f), xz(0.0f), yz(0.0f) {}
};

/**
 * @brief Runtime-dispatch viscoelastic fluid properties.
 *
 * Mirrors the design of fluidProps for GNF models. Each simulation case
 * provides a makeCasePhaseProps*() factory (in constants.inc) that fills
 * both the nnf and ve members of fluidPhaseProps.
 *
 * For two-component simulations:
 *   - The Newtonian phase uses type = VE_NEWTONIAN with eta_p = 0.
 *   - The VE phase uses the desired model type with its parameters.
 * blendVeProps() interpolates the two structs across the interface.
 */
struct veFluidProps {
    veFluidType type;   ///< Constitutive model selector
    dfloat      eta_p;  ///< Polymeric viscosity
    dfloat      lambda; ///< Relaxation time

    union {
        struct { dfloat L_sq;    } fenep;     ///< FENE-P: max extensibility squared (L_max^2)
        struct { dfloat alpha;   } giesekus;  ///< Giesekus: mobility parameter (0 < α < 0.5)
        struct { dfloat epsilon; } ptt;       ///< PTT: extensibility parameter (epsilson > 0)
    } u;
};

// ============================================================================
// UNIFIED PHASE FLUID PROPERTIES
// Bundles the viscous (GNF) and viscoelastic components for one fluid phase.
// Either component can be disabled by setting its type to the Newtonian default:
//   .nnf.type = FLUID_NEWTONIAN  -> pure solvent (returns omegaOld unchanged)
//   .ve.type  = VE_NEWTONIAN     -> no elasticity (returns zero relaxation term)
// This allows mixing, e.g. Bingham (phase A) + Oldroyd-B (phase B) via a
// diffuse-interface blending strategy (stress decomposition across the interface).
// ============================================================================
struct fluidPhaseProps {
    fluidProps   nnf;  ///< Viscous / GNF component
    veFluidProps ve;   ///< Viscoelastic / polymer component
};

#endif //__NNF_TYPES_H
