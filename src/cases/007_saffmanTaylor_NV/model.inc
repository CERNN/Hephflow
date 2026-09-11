/* ============================================================================
 * MODEL CONFIGURATION
 * Case: Vertical viscoelastic displacement between parallel plates
 * Description: Viscoelastic fluid displacing a Newtonian resident fluid
 * ========================================================================== */

/* --------------------- VELOCITY SET --------------------- */
#define D3Q19

/* --------------------- COLLISION METHOD --------------------- */
#define COLLISION_TYPE MR_LBM_2ND_ORDER
//#define COLLISION_TYPE HO_RR //http://dx.doi.org/10.1063/1.4981227
//#define COLLISION_TYPE HOME_LBM //https://inria.hal.science/hal-04223237/

/* --------------------- LES MODELS --------------------- */
// Uncomment the one to use. Comment all to simulate newtonian fluid
//#define LES_MODEL
//#define MODEL_CONST_SMAGORINSKY //https://doi.org/10.1016/j.jcp.2005.03.022

/* --------------------- PHYSICS MODELS --------------------- */
// Define SAFFMAN_TAYLOR_NEWTONIAN_BASELINE at compile time to exercise the
// same two-phase problem with identical Newtonian fluids and no conformation
// transport. The default remains the intended Newtonian-viscoelastic case.
#ifndef SAFFMAN_TAYLOR_NEWTONIAN_BASELINE
    #define FENE_P
#endif

// Enable phase-dependent viscosity blending via omega field
#define NON_NEWTONIAN_FLUID
// Use Bingham flag to activate OMEGA_FIELD path (yield set to zero in constants)
#define BINGHAM

/* --------------------- OTHER DEFINITIONS --------------------- */
//#define DENSITY_CORRECTION

#define RANDOM_NUMBERS true    // to generate random numbers
                               // (useful for turbulence)


#define TREAT_DATA_INCLUDE     // Include data treatment routines

