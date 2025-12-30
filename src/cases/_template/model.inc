/* ============================================================================
 * MODEL CONFIGURATION
 * Defines velocity set, collision operator, and numerical methods
 * ========================================================================== */

/* --------------------- PHYSICS MODELS --------------------- */
// Uncomment to enable specific physics models
//#define THERMAL_MODEL               // Enable thermal LBM (requires g_bc_definition.inc)
//#define PARTICLE_MODEL              // Enable IBM particles (requires particleCreation.inc)
//#define FENE_P                      // Enable FENE-P viscoelastic model (requires a*_bc_definition.inc files)

/* --------------------- VELOCITY SET --------------------- */
// Choose ONE velocity set (D3Q19 is default, D3Q27 for higher accuracy)
#define D3Q19
//#define D3Q27

/* --------------------- COLLISION METHOD --------------------- */
// Choose ONE collision operator
// See src/colrec/ folder for available collision-reconstruction implementations
#define COLLISION_TYPE MR_LBM_2ND_ORDER
//#define COLLISION_TYPE MR_LBM_3RD_ORDER
//#define COLLISION_TYPE MR_LBM_4TH_ORDER


/* --------------------- LES MODEL --------------------- */
// Uncomment to enable Large Eddy Simulation
//#define LES_MODEL
//#define MODEL_CONST_SMAGORINSKY         // https://doi.org/10.1016/j.jcp.2005.03.022

/* --------------------- NUMERICAL OPTIONS --------------------- */
//#define DENSITY_CORRECTION              // Correct density drift
//#define RANDOM_NUMBERS true             // Enable random number generation

/* --------------------- NON-NEWTONIAN FLUID MODELS --------------------- */
// Uncomment ONE model and set #define OMEGA_FIELD below
// See src/include/nnf.h for implementation details
//#define POWERLAW                        // Power-law model (shear thinning/thickening)
//#define BINGHAM                         // Bingham viscoplastic model
//#define BI_VISCOSITY                    // Bi-viscosity model
//#define KEE_TURCOTEE                    // Kee-Turcotte model (experimental)

// Enable variable viscosity field for non-Newtonian fluids
//#define OMEGA_FIELD
//#define NON_NEWTONIAN_FLUID
