/* ============================================================================
 * CONSTANTS CONFIGURATION
 * Case: 000_benchmark
 * Description: Performance benchmark case with periodic boundaries and random
 *              velocity perturbations for turbulence testing
 * 
 * Note: Physics model flags are defined in model.inc
 * ========================================================================== */

/* --------------------- SIMULATION SCALE --------------------- */
constexpr dfloat RE = 500;                  // Reynolds number
constexpr int SCALE = 1;                    // Grid scaling factor

/* --------------------- TIME STEPPING --------------------- */
constexpr int N_STEPS = 10000;              // Total simulation steps
constexpr int INI_STEP = 0;                 // Initial step (for restarts)

/* --------------------- GRID DIMENSIONS --------------------- */
constexpr int N = 256 * SCALE;              // Reference grid size
constexpr int NX = N;                       // Grid size in X
constexpr int NY = N;                       // Grid size in Y
constexpr int NZ = N;                       // Grid size in Z (single GPU)
constexpr int NZ_TOTAL = NZ;                // Total grid size in Z (multi-GPU)

/* --------------------- REFERENCE VALUES --------------------- */
constexpr dfloat U_MAX = 0.0;               // Maximum/reference velocity
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
constexpr dfloat VISC = U_MAX * NX / RE;    // Kinematic viscosity
constexpr dfloat TAU = 0.8;                 // Relaxation time
constexpr dfloat MACH_NUMBER = U_MAX/0.57735026918962;

/* ============================================================================
 * THERMAL MODEL PARAMETERS
 * Requires #define THERMAL_MODEL in model.inc
 * ========================================================================== */

#ifdef THERMAL_MODEL
#define SECOND_DIST
#define D3G19

constexpr bool T_BOUYANCY = true;
constexpr dfloat T_PR_NUMBER = 10;          // Prandtl number
constexpr dfloat T_RA_NUMBER = 1e5;         // Rayleigh number
constexpr dfloat BN_NUMBER = 0.2*8;
constexpr dfloat T_GR_NUMBER = T_RA_NUMBER/T_PR_NUMBER;  // Grashof number

constexpr dfloat T_DELTA_T = 1;             // Temperature difference
constexpr dfloat T_REFERENCE = 1;           // Reference temperature
constexpr dfloat T_COLD = T_REFERENCE - T_DELTA_T/2.0;
constexpr dfloat T_HOT = T_REFERENCE + T_DELTA_T/2.0;
constexpr dfloat C_0 = T_REFERENCE;         // Initial temperature field

constexpr dfloat T_DIFFUSIVITY = VISC/T_PR_NUMBER;
constexpr dfloat T_gravity_t_beta = 0;
constexpr dfloat Ra_conf = 0;

__device__ const dfloat gravity_vector[3] = {1.0, 0.0, 0.0};

constexpr dfloat T_DIFF_REF = (1.0/6);
constexpr dfloat G_TAU = T_DIFF_REF*3+0.5;
constexpr dfloat G_OMEGA = 1.0/G_TAU;
constexpr dfloat G_TT_OMEGA = 1.0-0.5*G_OMEGA;

constexpr dfloat T_DIFF_FLUC = T_DIFFUSIVITY - T_DIFF_REF;
constexpr dfloat G_AAA = (3*T_DIFF_FLUC*G_OMEGA);
constexpr dfloat G_DIFF_FLUC_COEF = G_AAA/(1.0+G_AAA);
#endif
