/* Turbulent Newtonian flow in an eccentric annulus. */
#define D3Q19
#define COLLISION_TYPE MR_LBM_2ND_ORDER

// #define LES_MODEL
// #define MODEL_CONST_SMAGORINSKY

#define RANDOM_NUMBERS true

#define TREAT_DATA_INCLUDE      // Include four-ray turbulence statistics
#define TURBULENT_TREAT_DATA    // Copy the macroscopic field for treatment
