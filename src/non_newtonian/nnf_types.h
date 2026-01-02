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

// Default LAMBDA_ZERO if not defined (when lambda transport not enabled)
#ifndef LAMBDA_ZERO
#define LAMBDA_ZERO 1.0_df
#endif

enum fluidType { FLUID_POWERLAW, FLUID_BINGHAM, FLUID_BI_VISCOSITY, FLUID_KEE_TURCOTEE, FLUID_HERSCHEL_BULKLEY, FLUID_THIXO };
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

#endif //__NNF_TYPES_H
