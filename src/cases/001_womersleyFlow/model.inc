/* ============================================================================
 * MODEL CONFIGURATION
 * Case: 001_womersleyFlow
 * Description: Oscillatory channel flow (planar Womersley flow)
 * ========================================================================== */

/* --------------------- VELOCITY SET --------------------- */
#define D3Q19

/* --------------------- COLLISION METHOD --------------------- */
#define COLLISION_TYPE MR_LBM_2ND_ORDER

/* --------------------- PHYSICS MODELS --------------------- */
//#define THERMAL_MODEL
//#define PARTICLE_MODEL
//#define FENE_P
//#define BINGHAM

/* --------------------- LES MODELS --------------------- */
//#define LES_MODEL
//#define MODEL_CONST_SMAGORINSKY

/* --------------------- OTHER DEFINITIONS --------------------- */
//#define DENSITY_CORRECTION

/* --------------------- TIME-DEPENDENT FORCE FIELD --------------------- */
#define FORCE_FIELD_INCLUDE             // Enables force_field.inc (oscillatory driving)

/* --------------------- TREAT DATA --------------------- */
#define TREAT_DATA_INCLUDE              // Enables treat_data.inc (velocity profile output)
