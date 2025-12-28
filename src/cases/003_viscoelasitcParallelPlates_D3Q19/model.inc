/* ============================================================================
 * MODEL CONFIGURATION
 * Case: 003_viscoelasitcParallelPlates_D3Q19
 * Description: Viscoelastic flow between parallel plates (FENE-P model)
 * ========================================================================== */

/* --------------------- VELOCITY SET --------------------- */
#define D3Q19

/* --------------------- COLLISION METHOD --------------------- */
#define COLLISION_TYPE MR_LBM_2ND_ORDER
//#define COLLISION_TYPE HO_RR //http://dx.doi.org/10.1063/1.4981227
//#define COLLISION_TYPE HOME_LBM //https://inria.hal.science/hal-04223237/

/* --------------------- PHYSICS MODELS --------------------- */
// Enable specific physics models (uncomment to activate)
#define FENE_P                        // FENE-P viscoelastic model
//#define BINGHAM                       // Bingham viscoplastic model
//#define THERMAL_MODEL                 // Thermal coupling
//#define PARTICLE_MODEL                // IBM particle simulation

/* --------------------- LES MODELS --------------------- */
// Uncomment the one to use. Comment all to simulate newtonian fluid
//#define LES_MODEL
//#define MODEL_CONST_SMAGORINSKY //https://doi.org/10.1016/j.jcp.2005.03.022

/* --------------------- OTHER DEFINITIONS --------------------- */
//#define DENSITY_CORRECTION //WE ENABLE DUE TO FORCE COMPONENT IN THE DIRECTION OF THE WALL, MAKING IT LOSE MASS

//#define RANDOM_NUMBERS true    // to generate random numbers 
                               // (useful for turbulence)



