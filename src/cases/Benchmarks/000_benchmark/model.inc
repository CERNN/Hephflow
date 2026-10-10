/* ============================================================================
 * MODEL CONFIGURATION
 * Defines velocity set, collision operator, and numerical methods
 * ========================================================================== */

/* --------------------- PHYSICS MODELS --------------------- */
// No additional physics models enabled for benchmark

/* --------------------- VELOCITY SET --------------------- */
#define D3Q19

/* --------------------- COLLISION METHOD --------------------- */
#define COLLISION_TYPE MR_LBM_2ND_ORDER

/* --------------------- LES MODEL --------------------- */
//#define LES_MODEL
//#define MODEL_CONST_SMAGORINSKY         // https://doi.org/10.1016/j.jcp.2005.03.022

/* --------------------- NUMERICAL OPTIONS --------------------- */
//#define DENSITY_CORRECTION
#define RANDOM_NUMBERS true             // Generate random numbers for turbulence



