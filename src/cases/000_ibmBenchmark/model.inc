/* ============================================================================
 * MODEL CONFIGURATION
 * Defines velocity set, collision operator, and numerical methods
 * ========================================================================== */

/* --------------------- PHYSICS MODELS --------------------- */
#define PARTICLE_MODEL                  // Enable IBM particles
//#define THERMAL_MODEL                 // Enable thermal LBM
//#define FENE_P                        // Enable viscoelastic FENE-P model

/* --------------------- VELOCITY SET --------------------- */
#define D3Q19

/* --------------------- COLLISION METHOD --------------------- */
#define COLLISION_TYPE MR_LBM_2ND_ORDER
//#define COLLISION_TYPE HO_RR            // http://dx.doi.org/10.1063/1.4981227
//#define COLLISION_TYPE HOME_LBM         // https://inria.hal.science/hal-04223237/

/* --------------------- LES MODEL --------------------- */
//#define LES_MODEL
//#define MODEL_CONST_SMAGORINSKY         // https://doi.org/10.1016/j.jcp.2005.03.022

/* --------------------- NUMERICAL OPTIONS --------------------- */
//#define DENSITY_CORRECTION
#define RANDOM_NUMBERS true             // Generate random numbers for turbulence

/* --------------------- IBM CONFIGURATION --------------------- */
#define BREUGEM_PARAMETER (0.0)         // Node location offset (http://dx.doi.org/10.1016/j.jcp.2012.02.026)
#define MESH_SCALE 1.0                  // Mesh scale for IBM nodes
#define MESH_COULOMB 0                  // Coulomb optimization iterations
#define IBM_THICKNESS 1.0               // Assumed boundary thickness
#define STENCIL_4                       // Peskin 4-point stencil
//#define ROTATION_LOCK true            // Lock particle rotation
