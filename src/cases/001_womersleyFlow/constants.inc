/* ============================================================================
 * CONSTANTS CONFIGURATION
 * Case: 001_womersleyFlow
 * Description: Oscillatory flow between parallel plates (planar Womersley flow)
 *              Driven by a time-harmonic body force:
 *                F_x(t) = G_L * cos(OMEGA_OSC * t)
 *              with no-slip (bounce-back) top/bottom walls and
 *              periodic boundary conditions in X and Z.
 * ========================================================================== */

/* --------------------- DIMENSIONLESS NUMBERS --------------------- */
constexpr dfloat WO = 1.0;                  // Womersley number

/* --------------------- SIMULATION SCALE --------------------- */
constexpr int SCALE = 1;                    // Grid scaling factor

/* --------------------- GRID DIMENSIONS --------------------- */
constexpr int NY = 64 * SCALE;              // Channel height (walls at y=0 and y=NY-1)
constexpr int NX = 8;                       // Thin periodic direction
constexpr int NZ = 8;                       // Thin periodic direction
constexpr int NZ_TOTAL = NZ;                // Total Z grid size (multi-GPU)

/* --------------------- FLUID PROPERTIES --------------------- */
constexpr dfloat TAU  = 0.8_df;           
constexpr dfloat VISC = (TAU - 0.5_df) / 3.0_df;
/* --------------------- WOMERSLEY PARAMETERS --------------------- */
constexpr dfloat H_CHANNEL = (dfloat)NY / 2.0_df; 
constexpr dfloat OMEGA_OSC  = VISC * (WO / H_CHANNEL) * (WO / H_CHANNEL);

constexpr dfloat T_OSC = 2.0_df * M_PI / OMEGA_OSC;

/* --------------------- DRIVING FORCE --------------------- */
constexpr dfloat G_L = 2.0e-5_df;    

/* --------------------- STATIC BODY FORCES (zero - force is time-dependent) ----- */
constexpr dfloat FX = 0.0_df;
constexpr dfloat FY = 0.0_df;
constexpr dfloat FZ = 0.0_df;

/* --------------------- REFERENCE VALUES --------------------- */
constexpr dfloat RHO_0   = 1.0_df;         // Initial density
constexpr dfloat U_MAX   = 0.0_df;         // (no mean flow; used for Mach tag only)
constexpr dfloat L       = (dfloat)NY;     // Reference length

/* --------------------- INITIAL CONDITIONS --------------------- */
constexpr dfloat U_0_X = 0.0_df;
constexpr dfloat U_0_Y = 0.0_df;
constexpr dfloat U_0_Z = 0.0_df;

/* --------------------- TIME STEPPING --------------------- */
// Run for 10 full oscillation periods
constexpr int N_STEPS   = (int)(10.0_df * T_OSC + 0.5_df);
constexpr int INI_STEP  = 0;
constexpr int SAVE_FREQ = (int)(T_OSC / 24.0_df + 0.5_df) < 1
                        ? 1
                        : (int)(T_OSC / 24.0_df + 0.5_df);

/* --------------------- BOUNDARY CONDITION VALUES --------------------- */
__device__ const dfloat UX_BC[4]  = {0, 0, 0, 0};
__device__ const dfloat UY_BC[4]  = {0, 0, 0, 0};
__device__ const dfloat UZ_BC[4]  = {0, 0, 0, 0};
__device__ const dfloat RHO_BC[4] = {RHO_0, RHO_0, RHO_0, RHO_0};

/* --------------------- BOUNDARY CONDITION TYPES --------------------- */
#define BC_X_PERIODIC                       
#define BC_Y_WALL                           
#define BC_Z_PERIODIC                       

/* --------------------- CHECKPOINT / RESTART --------------------- */
#define LOAD_CHECKPOINT false
#define SIMULATION_FOLDER_LOAD_CHECKPOINT "TEST"
#define CHECKPOINT_SAVE false
#define LOAD_FIELD false
#define LOAD_FIELD_STEP 0
