/* ============================================================================
 * CONSTANTS CONFIGURATION
 * Case: Oscillating Droplet
 * Description: Two-phase flow simulation of oscillating droplet
 * ========================================================================== */

/* --------------------- SIMULATION SCALE --------------------- */
constexpr int SCALE = 1;                    // Grid scaling factor

/* --------------------- TIME STEPPING --------------------- */
constexpr int N_STEPS = 1000000;               // Total simulation steps
constexpr int INI_STEP = 0;                 // Initial step (for restarts)

/* --------------------- GRID DIMENSIONS --------------------- */
constexpr int N = 128 * SCALE;              // Reference grid size
constexpr int NX = 2*N;                       // Grid size in X
constexpr int NY = 8*N;                       // Grid size in Y
constexpr int NZ = 4;                       // Grid size in Z
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
#define BC_Y_WALL
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
//#define INTERFACE_SHARPENING               // Enable interface sharpening
#define CONVECTION_DIFFUSION_TRANSPORT     // Transport equation method
#define D3G19                             // 19-velocity phase field (D3Q19 weights, as2=3)
//#define D3G7                              // 7-velocity phase field (D3Q7 weights, as2=4)

/* --------------------- PHASE FIELD DISTRIBUTION --------------------- */
#define PHI_DIST                           // Enable phase field distribution
#define METHOD_CAHN_HILLIARD
/* --------------------- PHASE FIELD REFERENCE VALUES --------------------- */
constexpr dfloat PHI_REFERENCE = (dfloat)(0.0);   // Reference phase value
constexpr dfloat PHI_DELTA_PHI = 2.0_df;                // Phase field range
constexpr dfloat PHI_ONE = PHI_REFERENCE - PHI_DELTA_PHI/2.0;   // Phase 1 value
constexpr dfloat PHI_TWO = PHI_REFERENCE + PHI_DELTA_PHI/2.0;   // Phase 2 value
constexpr dfloat PHI_ZERO = (dfloat)(1.1);        // Initial phase value
// Physical CH order parameter mapping
constexpr dfloat PHI_SCALE = 2.0/PHI_DELTA_PHI;

/* --------------------- SURFACE TENSION AND INTERFACE PARAMETERS --------------------- */

// Physical surface tension
constexpr dfloat phi_sigma = 0.0003;
constexpr dfloat phi_del = 4.0;
constexpr dfloat phi_xi = phi_del / 2.94;
constexpr dfloat phi_beta  = 12.0 * phi_sigma / phi_del;
constexpr dfloat phi_kappa = 1.5 * phi_sigma * phi_del;
constexpr dfloat phi_gamma = 1.0 / (2.0 * 1.41421356237309 * phi_xi);

/* --------------------- DENSITY COUPLING --------------------- */

constexpr dfloat phi_rho_l = 1.0;
constexpr dfloat phi_rho_g = 0.9;
constexpr dfloat phi_drho = phi_rho_l - phi_rho_g;

constexpr dfloat g_value = 1.5e-5_df; 
__device__ const dfloat gravity_vector[3] = {0.0, 1.0, 0.0}; // Gravity direction


constexpr dfloat tau_H = TAU;
constexpr dfloat tau_L = TAU;

/* --------------------- FORCE COUPLING CONTROL FLAGS --------------------- */
#define PHI_FORCE_FS_ENABLED 1
#define PHI_FORCE_FP_ENABLED 1
#define PHI_FORCE_FN_ENABLED 1
#define PHI_FORCE_FB_ENABLED 1

/* --------------------- PHASE-FIELD TRANSPORT (LBM) --------------------- */

// Reference diffusion (controls interface mobility)
constexpr dfloat PHI_DIFFUSIVITY_RATIO = 1; // Diffusivity ratio (like Prandtl number) 
constexpr dfloat PHI_DIFFUSIVITY = VISC/PHI_DIFFUSIVITY_RATIO; // Thermal diffusivity
constexpr dfloat PHI_DIFF_REF = 0.0000033333;

// LBM relaxation
constexpr dfloat PHI_TAU   = 0.5 + 3.0 * PHI_DIFF_REF;
constexpr dfloat PHI_OMEGA = 1.0 / PHI_TAU;

/* --------------------- INTERFACE SHARPENING --------------------- */

// Dimensionless sharpening strength (0.3–0.6 recommended)
constexpr dfloat SHARP_FACTOR = 30.0_df;

// Sharpening coefficient (now scale-consistent)
constexpr dfloat phi_sharp_c = SHARP_FACTOR * phi_gamma * PHI_DIFF_REF;

constexpr dfloat M_CH = 0.005_df;
constexpr dfloat kappa_CH = 0.75_df * phi_sigma * phi_del; 
constexpr dfloat A_CH     = (6.0_df * phi_sigma) / phi_del;
/* ----- END CONSTANTS CONFIGURATION ----- */
