/* ============================================================================
 * CONSTANTS CONFIGURATION
 * Case: 004_lambdaTransport
 * Description: Lambda thixotropic transport between parallel plates
 * ========================================================================== */

/* --------------------- SIMULATION SCALE --------------------- */
constexpr int SCALE = 1;                    // Grid scaling factor

/* --------------------- DIMENSIONLESS NUMBERS --------------------- */
constexpr dfloat RE = 10;                   // Reynolds number

/* --------------------- TIME STEPPING --------------------- */
constexpr int N_STEPS = 10000;                 // Total simulation steps
constexpr int INI_STEP = 0;                 // Initial step (for restarts)

/* --------------------- GRID DIMENSIONS --------------------- */
constexpr int N = 128 * SCALE;              // Reference grid size
constexpr int NX = 8;                       // Grid size in X (thin channel)
constexpr int NY = N;                       // Grid size in Y (channel height)
constexpr int NZ = 8;                       // Grid size in Z (thin channel)
constexpr int NZ_TOTAL = NZ;                // Total grid size in Z (multi-GPU)

/* --------------------- REFERENCE VALUES --------------------- */
constexpr dfloat U_MAX = 0.0;               // Maximum/reference velocity
constexpr dfloat L = N;                     // Reference length

/* --------------------- BODY FORCES --------------------- */
constexpr dfloat FX = 0.0;                  // Force in X direction
constexpr dfloat FY = 0.0;                  // Force in Y direction
constexpr dfloat FZ = 1e-6;                 // Force in Z direction (flow driving)

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
constexpr dfloat VISC = 0.8;                // Kinematic viscosity
constexpr dfloat TAU = 0.5 + 3.0*VISC;      // Relaxation time
constexpr dfloat MACH_NUMBER = U_MAX/0.57735026918962; // Mach number

/* ============================================================================
 * LAMBDA THIXOTROPIC MODEL PARAMETERS
 * ========================================================================== */

#define D3G19
/* --------------------- LAMBDA INITIALIZATION --------------------- */
constexpr dfloat LAMBDA_ZERO = 10.0;             // Initial lambda value (fully structured)

/* --------------------- LAMBDA KINETICS --------------------- */
constexpr dfloat LAMBDA_K1 = 1.0;            //  Breakdown-ratemultiplier
constexpr dfloat LAMBDA_K2 = 3e-4;            // Build-rate multiplier

// Build/break timescales (in lattice units)

/* --------------------- LAMBDA TRANSPORT PARAMETERS --------------------- */
constexpr dfloat LAMBDA_DIFFUSIVITY_RATIO = 100.0; // Diffusivity ratio (like Prandtl number)
constexpr dfloat LAMBDA_DIFFUSIVITY = VISC/LAMBDA_DIFFUSIVITY_RATIO;

constexpr dfloat LAMBDA_DIFF_REF = (1.0/6);    // Reference diffusion
constexpr dfloat LAMBDA_TAU = LAMBDA_DIFF_REF*3+0.5;    // Lambda relaxation time
constexpr dfloat LAMBDA_OMEGA  = 1.0/LAMBDA_TAU;        // Lambda collision frequency
constexpr dfloat LAMBDA_TT_OMEGA = 1.0-0.5*LAMBDA_OMEGA; // TRT parameter

// Diffusivity fluctuation correction
constexpr dfloat LAMBDA_DIFF_FLUC = LAMBDA_DIFFUSIVITY - LAMBDA_DIFF_REF;
constexpr dfloat LAMBDA_AAA = (3*LAMBDA_DIFF_FLUC*LAMBDA_OMEGA);
constexpr dfloat LAMBDA_DIFF_FLUC_COEF = LAMBDA_AAA/(1.0+LAMBDA_AAA);

/* ----- END CONSTANTS CONFIGURATION ----- */