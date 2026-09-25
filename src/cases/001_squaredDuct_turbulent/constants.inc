/* ============================================================================
 * CONSTANTS CONFIGURATION
 * Case: 001_squaredDuct_turbulent
 * Description: Turbulent flow through a square duct
 * ========================================================================== */

/* --------------------- DIMENSIONLESS NUMBERS --------------------- */
constexpr dfloat RE_TAU = 180.0;            // Friction Reynolds number

/* --------------------- TIME STEPPING --------------------- */
constexpr int N_STEPS = 1360000;            // Total simulation steps
constexpr int INI_STEP = 0;                 // Initial step (for restarts)

/* --------------------- GRID DIMENSIONS --------------------- */
constexpr int HALF_WIDTH = 96;               // Duct half-width / integral scale
constexpr int N = HALF_WIDTH;                // Reference length retained for utilities
constexpr int NX = 2 * HALF_WIDTH;           // Square-duct width in X
constexpr int NY = 2 * HALF_WIDTH;           // Square-duct width in Y
constexpr int NZ = 16 * HALF_WIDTH;          // Periodic streamwise length
constexpr int NZ_TOTAL = NZ;                // Total grid size in Z (multi-GPU)

/* --------------------- REFERENCE VALUES --------------------- */
constexpr dfloat U_TAU = 0.00579;            // Perimeter-averaged friction velocity
constexpr dfloat U_CENTER_INITIAL = 18.2 * U_TAU;
constexpr dfloat U_MAX = U_CENTER_INITIAL;   // Velocity scale used for Mach reporting
constexpr dfloat L = HALF_WIDTH;

/* --------------------- BODY FORCES --------------------- */
constexpr dfloat FX = 0.0;                  // Force in X direction
constexpr dfloat FY = 0.0;                  // Force in Y direction
// For a square duct, force balance gives FZ = 2*u_tau^2/h, where h is
// the half-width. This differs by a factor of two from a plane channel.
constexpr dfloat FZ = 2.0 * U_TAU * U_TAU / HALF_WIDTH;

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
#define BC_X_WALL
#define BC_Y_WALL
#define BC_Z_PERIODIC

/* --------------------- FLUID PROPERTIES --------------------- */
constexpr dfloat VISC = HALF_WIDTH * U_TAU / RE_TAU;
constexpr dfloat TAU = 0.5 + 3.0*VISC;      // Relaxation time
constexpr dfloat MACH_NUMBER = U_MAX/0.57735026918962; // Mach number

/* --------------------- TURBULENCE CONTROL --------------------- */
constexpr dfloat INITIAL_PERTURBATION = 0.30; // Fraction of initial center velocity
constexpr int TURBULENCE_STATS_START_STEP = 0;
constexpr int TURBULENCE_STATS_SAVE_INTERVAL = 3400;

/* --------------------- INITIALIZATION LOADING DEFINES -------------------- */
#define LOAD_CHECKPOINT false   // loads simulation checkpoint from folder 
// WITHOUT ID_SIM (change it in ID_SIM) AND "/" AT THE END
#define SIMULATION_FOLDER_LOAD_CHECKPOINT "TEST"
// Interval to make checkpoint to save all simulation data and restart from it.
// It must not be very frequent (10000 or more), because it takes a long time
#define CHECKPOINT_SAVE false // the frequency on which the simulation checkpoint is saved
#define LOAD_FIELD false
#define LOAD_FIELD_STEP 0
