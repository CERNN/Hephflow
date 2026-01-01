/* ============================================================================
 * CONSTANTS CONFIGURATION
 * Case: Oscillating Droplet
 * Description: Two-phase flow simulation of oscillating droplet
 * ========================================================================== */

/* --------------------- SIMULATION SCALE --------------------- */
constexpr int SCALE = 1;                    // Grid scaling factor

/* --------------------- TIME STEPPING --------------------- */
constexpr int N_STEPS = 5000;               // Total simulation steps
constexpr int INI_STEP = 0;                 // Initial step (for restarts)

/* --------------------- GRID DIMENSIONS --------------------- */
constexpr int N = 128 * SCALE;              // Reference grid size
constexpr int NX = N;                       // Grid size in X
constexpr int NY = N;                       // Grid size in Y
constexpr int NZ = N;                       // Grid size in Z
constexpr int NZ_TOTAL = NZ;                // Total grid size in Z (multi-GPU)

/* --------------------- REFERENCE VALUES --------------------- */
constexpr dfloat U_MAX = 0.1;               // Maximum/reference velocity
constexpr dfloat L = N;                     // Reference length

/* --------------------- BODY FORCES --------------------- */
constexpr dfloat FX = 0.0;                  // Force in X direction
constexpr dfloat FY = 0.0;                  // Force in Y direction
constexpr dfloat FZ = 0.0;                  // Force in Z direction

/* --------------------- INITIAL CONDITIONS --------------------- */
constexpr dfloat RHO_0 = 1.0;               // Initial density
constexpr dfloat U_0_X = 0.0;               // Initial velocity X
constexpr dfloat U_0_Y = 0.0;               // Initial velocity Y
constexpr dfloat U_0_Z = 0.0;               // Initial velocity Z

/* --------------------- BOUNDARY CONDITION VALUES --------------------- */
__device__ const dfloat UX_BC[4] = {0, 0, 0, 0};
__device__ const dfloat UY_BC[4] = {0, 0, 0, 0};
__device__ const dfloat UZ_BC[4] = {0, 0, 0, 0};
__device__ const dfloat RHO_BC[4] = {RHO_0, RHO_0, RHO_0, RHO_0};

/* --------------------- BOUNDARY CONDITION TYPES --------------------- */
#define BC_X_PERIODIC
#define BC_Y_PERIODIC
#define BC_Z_PERIODIC

/* --------------------- FLUID PROPERTIES --------------------- */
constexpr dfloat TAU = 0.55;                // Relaxation time
constexpr dfloat VISC = (TAU - 0.5) / 3;    // Kinematic viscosity
constexpr dfloat MACH_NUMBER = U_MAX / 0.57735026918962; // Mach number

/* --------------------- CHECKPOINT AND INITIALIZATION --------------------- */
#define LOAD_CHECKPOINT false               // Load simulation checkpoint
#define CHECKPOINT_SAVE false               // Save checkpoint frequency
#define LOAD_FIELD false                    // Load field data
#define LOAD_FIELD_STEP 0                   // Field load step
#define SIMULATION_FOLDER_LOAD_CHECKPOINT "TEST" // Checkpoint folder

/* ============================================================================
 * PHASE FIELD MODEL PARAMETERS
 * Requires #define PHASE_MODEL in model.inc
 * Uses second distribution for phase field tracking
 * ========================================================================== */

#define PHASE_MODEL                         // Enable phase field model
#define INTERFACE_SHARPENING               // Enable interface sharpening
#define CONVECTION_DIFFUSION_TRANSPORT     // Transport equation method
#define D3G19                              // 19-velocity phase field (D3Q19 weights, as2=3)
//#define D3G7                              // 7-velocity phase field (D3Q7 weights, as2=4)

/* --------------------- PHASE FIELD DISTRIBUTION --------------------- */
#define PHI_DIST                           // Enable phase field distribution

/* --------------------- PHASE FIELD REFERENCE VALUES --------------------- */
constexpr dfloat PHI_REFERENCE = (dfloat)(0.5);   // Reference phase value
constexpr dfloat PHI_DELTA_PHI = 1;                // Phase field range
constexpr dfloat PHI_ONE = PHI_REFERENCE - PHI_DELTA_PHI/2.0;   // Phase 1 value
constexpr dfloat PHI_TWO = PHI_REFERENCE + PHI_DELTA_PHI/2.0;   // Phase 2 value
constexpr dfloat PHI_ALPHA = 0.05;                 // Regularization parameter
constexpr dfloat PHI_ZERO = (dfloat)(10.0);        // Initial phase value

/* --------------------- SURFACE TENSION AND INTERFACE PARAMETERS --------------------- */
// Model constants
constexpr dfloat phi_sigma = 1e-2;         // Surface tension coefficient
constexpr dfloat phi_del = 3;              // Interface width
constexpr dfloat phi_beta = 12.0*phi_sigma / phi_del;  // Beta parameter
constexpr dfloat phi_kappa = 3.0*phi_sigma*phi_del/2;  // Kappa parameter

// Density coupling constants
constexpr dfloat phi_rho_l = 1.0;          // Liquid density
constexpr dfloat phi_rho_g = 0.9;          // Gas density (10% difference)
constexpr dfloat phi_drho = phi_rho_l - phi_rho_g; // Density difference

// Interface sharpening coefficient (larger = sharper interface)
constexpr dfloat phi_sharp_c = 0.45;       // Sharpening coefficient

/* --------------------- FORCE COUPLING CONTROL FLAGS --------------------- */
#define PHI_FORCE_FS_ENABLED 1             // Surface tension force (DISABLED for test)
#define PHI_FORCE_FP_ENABLED 0             // Pressure gradient force (DISABLED)
#define PHI_FORCE_FN_ENABLED 0             // Viscous stress force (DISABLED)

/* --------------------- PHASE FIELD ADVECTION-DIFFUSION PARAMETERS --------------------- */
constexpr dfloat PHI_DIFFUSIVITY_RATIO = 1; // Diffusivity ratio (like Prandtl number)
constexpr dfloat PHI_DIFFUSIVITY = VISC/PHI_DIFFUSIVITY_RATIO; // Thermal diffusivity

constexpr dfloat PHI_DIFF_REF = (1.0/6);    // Reference diffusion
constexpr dfloat PHI_TAU = PHI_DIFF_REF*3+0.5;    // Phase field relaxation time
constexpr dfloat PHI_OMEGA  = 1.0/PHI_TAU;        // Phase field collision frequency
constexpr dfloat PHI_TT_OMEGA = 1.0-0.5*PHI_OMEGA; // TRT parameter

// Diffusivity fluctuation correction
constexpr dfloat PHI_DIFF_FLUC = PHI_DIFFUSIVITY - PHI_DIFF_REF;
constexpr dfloat PHI_AAA = (3*PHI_DIFF_FLUC*PHI_OMEGA);
constexpr dfloat PHI_DIFF_FLUC_COEF = PHI_AAA/(1.0+PHI_AAA); // Diffusivity correction

/* ----- END CONSTANTS CONFIGURATION ----- */
