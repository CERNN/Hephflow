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

#if defined(LES_MODEL)
    #define OMEGA_FIELD
    #define COMPUTE_SHEAR
#endif //LES_MODEL

#if defined(HO_RR) || defined(HOME_LBM)
    #define HIGH_ORDER_COLLISION
#endif // HO_RR || HOME_LBM

#endif //__FEATURE_CONFIG_H