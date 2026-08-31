/* ============================================================================
 * CONSTANTS CONFIGURATION
 * Case: 001_taylorGreen
 * Description: Taylor-Green vortex decay benchmark
 * 
 * Note: N_STEPS calculated from non-dimensional time (20.0/Ct)
 * ========================================================================== */

/* --------------------- SIMULATION SCALE --------------------- */
constexpr int SCALE = 1;                    // Grid scaling factor

/* --------------------- DIMENSIONLESS NUMBERS --------------------- */
constexpr dfloat RE = 1;                    // Reynolds number

/* --------------------- GRID DIMENSIONS --------------------- */
constexpr int N = 128 * SCALE;              // Reference grid size
constexpr int NX = N;                       // Grid size in X
constexpr int NY = N;                       // Grid size in Y
constexpr int NZ = N;                       // Grid size in Z
constexpr int NZ_TOTAL = NZ;                // Total grid size in Z (multi-GPU)

/* --------------------- REFERENCE VALUES --------------------- */
constexpr dfloat U_MAX = 1.0/(125.0*M_PI);  // Maximum/reference velocity
constexpr dfloat L = (dfloat)N / (2.0 * M_PI); // Reference length

/* --------------------- NON-DIMENSIONAL SCALING --------------------- */
constexpr dfloat Cv = 1.0/U_MAX;            // Velocity scaling
constexpr dfloat Ct = (1.0/L)/(1.0/U_MAX);  // Time scaling
constexpr dfloat Cx = 1.0/L;                // Length scaling
constexpr dfloat Cp = 1.0/(Cx * Ct * Ct);   // Pressure scaling

/* --------------------- TIME STEPPING --------------------- */
constexpr int N_STEPS = (int)(20.0/Ct)+1;   // Full simulation steps (calculated)
constexpr int INI_STEP = 0;                 // Initial step (for restarts)

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
constexpr dfloat VISC = L*U_MAX / RE;       // Kinematic viscosity
constexpr dfloat TAU = 0.5 + 3.0*VISC;      // Relaxation time
constexpr dfloat MACH_NUMBER = U_MAX/0.57735026918962; // Mach number
#define LOAD_FIELD false
#define LOAD_FIELD_STEP 0