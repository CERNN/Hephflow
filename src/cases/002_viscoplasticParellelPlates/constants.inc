/* ============================================================================
 * CONSTANTS CONFIGURATION
 * Case: 002_viscoplasticParellelPlates
 * Description: iscoplastic flow between parallel plates (Bingham)
 * ========================================================================== */

/* --------------------- TIME STEPPING --------------------- */
constexpr int N_STEPS = 10000;            // Total simulation steps
constexpr int INI_STEP = 0;                 // Initial step (for restarts)

/* --------------------- DIMENSIONLESS NUMBERS --------------------- */
constexpr dfloat RE = 180.0;                // Reynolds number
constexpr dfloat Bn = 0.1;                  // Bingham number (0 < Bn < 0.5)

/* --------------------- TURBULENCE PARAMETERS --------------------- */
constexpr dfloat U_TAU = 0.00579;           // Friction velocity

/* --------------------- GRID DIMENSIONS --------------------- */
constexpr unsigned int N = 128;          // Grid size in Y (192)
constexpr unsigned int NX = 8;          // Grid size in X (192)
constexpr unsigned int NY = N;          // Grid size in Y (192)
constexpr unsigned int NZ = 8;         // Grid size in Z (1536)
constexpr int NZ_TOTAL = NZ;                // Total grid size in Z (multi-GPU)

/* --------------------- REFERENCE VALUES --------------------- */
constexpr dfloat U_MAX = U_TAU;             // Maximum/reference velocity
constexpr dfloat RHO_0 = 1.0;               // Initial density
constexpr dfloat L = N;                     // Reference length

/* --------------------- BODY FORCES --------------------- */
constexpr dfloat FX = 0.0;                  // Force in X direction
constexpr dfloat FY = 0.0;                  // Force in Y direction
constexpr dfloat FZ = 2e-6;      // Force in Z direction (flow driving)

/* --------------------- INITIAL CONDITIONS --------------------- */
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
constexpr dfloat VISC = 0.1;      // Relaxation time
constexpr dfloat TAU = 3.0*VISC + 0.5;      // Relaxation time
constexpr dfloat S_Y = 1e-5;      // Yield stress (Bingham)
constexpr dfloat MACH_NUMBER = U_MAX/0.57735026918962; // Mach number

/* --------------------- INITIALIZATION LOADING DEFINES -------------------- */
#define LOAD_CHECKPOINT false   // loads simulation checkpoint from folder 
                                // (folder name defined below)


// Folder with simulation to load data from last checkpoint. 
// WITHOUT ID_SIM (change it in ID_SIM) AND "/" AT THE END
#define SIMULATION_FOLDER_LOAD_CHECKPOINT "TEST"
// Interval to make checkpoint to save all simulation data and restart from it.
// It must not be very frequent (10000 or more), because it takes a long time
#define CHECKPOINT_SAVE false // the frequency on which the simulation checkpoint is saved
#define LOAD_FIELD false
#define LOAD_FIELD_STEP 0