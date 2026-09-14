/* ============================================================================
 * MODEL CONFIGURATION
 * Case: 3D Rayleigh-Taylor Instability verification
 * ========================================================================== */

/* --------------------- VELOCITY SET --------------------- */
#define D3Q19

/* --------------------- COLLISION METHOD --------------------- */
#define COLLISION_TYPE MR_LBM_2ND_ORDER

/* --------------------- LES MODELS --------------------- */
// #define LES_MODEL
// #define MODEL_CONST_SMAGORINSKY

/* --------------------- OTHER DEFINITIONS --------------------- */
// #define DENSITY_CORRECTION

#define RANDOM_NUMBERS false
#define TREAT_DATA_INCLUDE
