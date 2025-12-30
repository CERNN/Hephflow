/**
 *  @file function_macros.h
 *  @brief Function parameter declaration macros for conditional compilation
 *  @version 0.4.0
 *  @date 27/12/2025
 */

#ifndef __FUNCTION_MACROS_H
#define __FUNCTION_MACROS_H

/* ===================== SINGLE-POINTER PARAMETER MACROS =================== */

#ifdef DENSITY_CORRECTION
    #define DENSITY_CORRECTION_PARAMS_DECLARATION(PREFIX) dfloat *PREFIX##mean_rho,
    #define DENSITY_CORRECTION_PARAMS(PREFIX) PREFIX##mean_rho,
#else
    #define DENSITY_CORRECTION_PARAMS_DECLARATION(PREFIX)
    #define DENSITY_CORRECTION_PARAMS(PREFIX)
#endif //DENSITY_CORRECTION

#ifdef BC_FORCES
    #define BC_FORCES_PARAMS_DECLARATION(PREFIX) dfloat *PREFIX##BC_Fx, dfloat *PREFIX##BC_Fy, dfloat *PREFIX##BC_Fz,
    #define BC_FORCES_PARAMS(PREFIX) PREFIX##BC_Fx, PREFIX##BC_Fy, PREFIX##BC_Fz,
#else
    #define BC_FORCES_PARAMS_DECLARATION(PREFIX)
    #define BC_FORCES_PARAMS(PREFIX)
#endif //BC_FORCES

#if NODE_TYPE_SAVE
    #define NODE_TYPE_SAVE_PARAMS_DECLARATION unsigned int* nodeTypeSave,
    #define NODE_TYPE_SAVE_PARAMS nodeTypeSave,
#else
    #define NODE_TYPE_SAVE_PARAMS_DECLARATION
    #define NODE_TYPE_SAVE_PARAMS
#endif //NODE_TYPE_SAVE

#ifdef OMEGA_FIELD
    #define OMEGA_FIELD_PARAMS_DECLARATION dfloat *omega,
    #define OMEGA_FIELD_PARAMS omega,
#else
    #define OMEGA_FIELD_PARAMS_DECLARATION
    #define OMEGA_FIELD_PARAMS
#endif //OMEGA_FIELD

/* ===================== DOUBLE-POINTER PARAMETER MACROS =================== */

#ifdef OMEGA_FIELD
    #define OMEGA_FIELD_PARAMS_DECLARATION_PTR ,dfloat** omega
    #define OMEGA_FIELD_PARAMS_PTR ,&omega
#else
    #define OMEGA_FIELD_PARAMS_DECLARATION_PTR
    #define OMEGA_FIELD_PARAMS_PTR
#endif //OMEGA_FIELD

#ifdef SECOND_DIST
    #define SECOND_DIST_PARAMS_DECLARATION_PTR ,dfloat** C
    #define SECOND_DIST_PARAMS_PTR ,&C
#else
    #define SECOND_DIST_PARAMS_DECLARATION_PTR
    #define SECOND_DIST_PARAMS_PTR
#endif //SECOND_DIST

#ifdef PHI_DIST
    #define PHI_DIST_PARAMS_DECLARATION_PTR ,dfloat** phi
    #define PHI_DIST_PARAMS_PTR ,&phi
#else
    #define PHI_DIST_PARAMS_DECLARATION_PTR
    #define PHI_DIST_PARAMS_PTR
#endif //PHI_DIST

/* ==================== AIJ TENSOR PARAMETER MACROS ======================= */

#ifdef A_XX_DIST
    #define A_XX_DIST_PARAMS_DECLARATION_PTR ,dfloat** Axx
    #define A_XX_DIST_PARAMS_PTR ,&Axx
#else
    #define A_XX_DIST_PARAMS_DECLARATION_PTR
    #define A_XX_DIST_PARAMS_PTR
#endif //A_XX_DIST

#ifdef A_XY_DIST
    #define A_XY_DIST_PARAMS_DECLARATION_PTR ,dfloat** Axy
    #define A_XY_DIST_PARAMS_PTR ,&Axy
#else
    #define A_XY_DIST_PARAMS_DECLARATION_PTR
    #define A_XY_DIST_PARAMS_PTR
#endif //A_XY_DIST

#ifdef A_XZ_DIST
    #define A_XZ_DIST_PARAMS_DECLARATION_PTR ,dfloat** Axz
    #define A_XZ_DIST_PARAMS_PTR ,&Axz
#else
    #define A_XZ_DIST_PARAMS_DECLARATION_PTR
    #define A_XZ_DIST_PARAMS_PTR
#endif //A_XZ_DIST

#ifdef A_YY_DIST
    #define A_YY_DIST_PARAMS_DECLARATION_PTR ,dfloat** Ayy
    #define A_YY_DIST_PARAMS_PTR ,&Ayy
#else
    #define A_YY_DIST_PARAMS_DECLARATION_PTR
    #define A_YY_DIST_PARAMS_PTR
#endif //A_YY_DIST

#ifdef A_YZ_DIST
    #define A_YZ_DIST_PARAMS_DECLARATION_PTR ,dfloat** Ayz
    #define A_YZ_DIST_PARAMS_PTR ,&Ayz
#else
    #define A_YZ_DIST_PARAMS_DECLARATION_PTR
    #define A_YZ_DIST_PARAMS_PTR
#endif //A_YZ_DIST

#ifdef A_ZZ_DIST
    #define A_ZZ_DIST_PARAMS_DECLARATION_PTR ,dfloat** Azz
    #define A_ZZ_DIST_PARAMS_PTR ,&Azz
#else
    #define A_ZZ_DIST_PARAMS_DECLARATION_PTR
    #define A_ZZ_DIST_PARAMS_PTR
#endif //A_ZZ_DIST

/* ======================= MEAN FLOW PARAMETER MACROS ===================== */

#if MEAN_FLOW
    #define MEAN_FLOW_PARAMS_DECLARATION_PTR ,dfloat** m_fMom,dfloat** m_rho,dfloat** m_ux,dfloat** m_uy,dfloat** m_uz
    #define MEAN_FLOW_PARAMS_PTR , &m_fMom, &m_rho, &m_ux, &m_uy, &m_uz
    #ifdef SECOND_DIST
        #define MEAN_FLOW_SECOND_DIST_PARAMS_DECLARATION_PTR ,dfloat** m_c
        #define MEAN_FLOW_SECOND_DIST_PARAMS_PTR , &m_c
    #else
        #define MEAN_FLOW_SECOND_DIST_PARAMS_DECLARATION_PTR
        #define MEAN_FLOW_SECOND_DIST_PARAMS_PTR
    #endif //SECOND_DIST
    #ifdef PHI_DIST
        #define MEAN_FLOW_PHI_DIST_PARAMS_DECLARATION_PTR ,dfloat** phi_c
        #define MEAN_FLOW_PHI_DIST_PARAMS_PTR , &phi_c
    #else
        #define MEAN_FLOW_PHI_DIST_PARAMS_DECLARATION_PTR
        #define MEAN_FLOW_PHI_DIST_PARAMS_PTR
    #endif //PHI_DIST
#else
    #define MEAN_FLOW_PARAMS_DECLARATION_PTR
    #define MEAN_FLOW_PARAMS_PTR
    #define MEAN_FLOW_SECOND_DIST_PARAMS_DECLARATION_PTR
    #define MEAN_FLOW_SECOND_DIST_PARAMS_PTR
    #define MEAN_FLOW_PHI_DIST_PARAMS_DECLARATION_PTR
    #define MEAN_FLOW_PHI_DIST_PARAMS_PTR
#endif //MEAN_FLOW

/* ==================== BOUNDARY CONDITION PARAMETER MACROS =============== */

#ifdef BC_FORCES
    #define BC_FORCES_PARAMS_DECLARATION_PTR(PREFIX) ,dfloat** PREFIX##BC_Fx ,dfloat** PREFIX##BC_Fy ,dfloat** PREFIX##BC_Fz
    #define BC_FORCES_PARAMS_PTR(PREFIX) ,&PREFIX##BC_Fx ,&PREFIX##BC_Fy ,&PREFIX##BC_Fz
#else
    #define BC_FORCES_PARAMS_DECLARATION_PTR(PREFIX)
    #define BC_FORCES_PARAMS_PTR(PREFIX)
#endif //BC_FORCES

#ifdef DENSITY_CORRECTION
    #define DENSITY_CORRECTION_PARAMS_DECLARATION_PTR(PREFIX) ,dfloat** PREFIX##mean_rho
    #define DENSITY_CORRECTION_PARAMS_PTR(PREFIX) , &PREFIX##mean_rho
#else
    #define DENSITY_CORRECTION_PARAMS_DECLARATION_PTR
    #define DENSITY_CORRECTION_PARAMS_PTR
#endif //DENSITY_CORRECTION

#ifdef CURVED_BOUNDARY_CONDITION
    #define CURVED_BC_PARAMS_DECLARATION_PTR(PREFIX) CurvedBoundary*** PREFIX##curvedBC,
    #define CURVED_BC_PARAMS_PTR(PREFIX) &PREFIX##curvedBC,
#else
    #define CURVED_BC_PARAMS_DECLARATION_PTR(PREFIX)
    #define CURVED_BC_PARAMS_PTR(PREFIX)
#endif

#ifdef CURVED_BOUNDARY_CONDITION
    #define CURVED_BC_PTRS_DECL(PREFIX) \
        CurvedBoundary** &PREFIX##curvedBC,

    #define CURVED_BC_ARRAY_DECL(PREFIX) \
        CurvedBoundary* &PREFIX##curvedBC_array,

    #define CURVED_BC_PTRS(PREFIX) \
        PREFIX##curvedBC,

    #define CURVED_BC_ARRAY(PREFIX) \
        PREFIX##curvedBC_array,
#else
    #define CURVED_BC_PTRS_DECL(PREFIX)
    #define CURVED_BC_ARRAY_DECL(PREFIX)
    #define CURVED_BC_PTRS(PREFIX)
    #define CURVED_BC_ARRAY(PREFIX)
#endif

#endif //__FUNCTION_MACROS_H