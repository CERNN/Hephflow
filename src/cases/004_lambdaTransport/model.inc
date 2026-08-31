/* ============================================================================
 * MODEL CONFIGURATION
 * Case: 004_lambdaTransport
 * Description: Lambda thixotropic transport between parallel plates
 * ========================================================================== */

/* --------------------- VELOCITY SET --------------------- */
#define D3Q19

/* --------------------- COLLISION METHOD --------------------- */
#define COLLISION_TYPE MR_LBM_2ND_ORDER

/* --------------------- PHYSICS MODELS --------------------- */
// Enable lambda thixotropic transport
#define CONVECTION_DIFFUSION_TRANSPORT
#define LAMBDA_DIST
#define LAMBDA_MODEL
//#define COLLISION_TYPE HO_RR //http://dx.doi.org/10.1063/1.4981227
//#define COLLISION_TYPE HOME_LBM //https://inria.hal.science/hal-04223237/

/* --------------------- PHYSICS MODELS --------------------- */
// Enable specific physics models (uncomment to activate)
//#define THERMAL_MODEL                 // Thermal coupling
//#define PARTICLE_MODEL                // IBM particle simulation
//#define FENE_P                        // FENE-P viscoelastic model

/* --------------------- LES MODELS --------------------- */
// Uncomment the one to use. Comment all to simulate newtonian fluid
//#define LES_MODEL
//#define MODEL_CONST_SMAGORINSKY //https://doi.org/10.1016/j.jcp.2005.03.022

/* --------------------- OTHER DEFINITIONS --------------------- */
//#define DENSITY_CORRECTION

//#define RANDOM_NUMBERS true    // to generate random numbers 
                               // (useful for turbulence)


