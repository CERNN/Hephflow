/* ============================================================================
 * CONSTANTS CONFIGURATION
 * Case: Herschel-Bulkley Laminar Flow Between Parallel Plates
 * ========================================================================== */

/* --------------------- TIME STEPPING --------------------- */
constexpr int N_STEPS = 100000;              // Total simulation steps
constexpr int INI_STEP = 0;                  // Initial step

/* --------------------- GRID DIMENSIONS --------------------- */
constexpr unsigned int N = 128;              // Domain height (Y direction)
constexpr unsigned int NX = 8;               // Streamwise length (X)
constexpr unsigned int NY = N;               // Channel height (Y)
constexpr unsigned int NZ = 8;               // Spanwise width (Z)
constexpr int NZ_TOTAL = NZ;                 // Total grid size in Z

/* --------------------- REFERENCE VALUES --------------------- */
constexpr dfloat U_MAX = 0.1_df;             // Target plug velocity (limit u <= 0.1)
constexpr dfloat RHO_0 = 1.0_df;             // Initial fluid density
constexpr dfloat L = N;                      // Reference length scale
constexpr dfloat H = (static_cast<dfloat>(NY) - 1.0_df) / 2.0_df;            // Channel half-width (H = 64)

/* --------------------- HERSCHEL-BULKLEY PARAMETERS --------------------- */
constexpr dfloat K_CONSISTENCY = 1e-1_df;    // Consistency index (K)
constexpr dfloat N_INDEX = 0.5_df;          // Flow behavior index (n < 1 shear thinning)
constexpr dfloat YIELD_RATIO = 0.2_df;       // Ratio of yield stress to wall shear stress (a = S_Y / tau_w)

constexpr dfloat VISC = K_CONSISTENCY;
constexpr dfloat TAU  = 3.0 * VISC + 0.5;

/* --------------------- ANALYTICAL FORCE & YIELD CALCULATION --------------------- */
constexpr dfloat PLUG_WIDTH_RATIO = 1.0_df - YIELD_RATIO;

// Analytical wall shear stress required to obtain U_MAX in the plug region
constexpr dfloat TAU_WALL = K_CONSISTENCY * constexprPow(
    (U_MAX * (N_INDEX + 1.0_df) / N_INDEX) / (H * constexprPow(PLUG_WIDTH_RATIO, (N_INDEX + 1.0_df) / N_INDEX)),
    N_INDEX
);

// Driving body force (FZ = dp/dz) and yield stress (S_Y)
constexpr dfloat FZ = TAU_WALL / H;          
constexpr dfloat FX = 0.0_df;                  
constexpr dfloat FY = 0.0_df;                  
constexpr dfloat S_Y = YIELD_RATIO * TAU_WALL;

/* --------------------- INITIAL CONDITIONS --------------------- */
constexpr dfloat U_0_X = 0.0_df;               
constexpr dfloat U_0_Y = 0.0_df;               
constexpr dfloat U_0_Z = 0.0_df;               

/* --------------------- BOUNDARY CONDITIONS --------------------- */
__device__ const dfloat UX_BC[4] = {0, 0, 0, 0};
__device__ const dfloat UY_BC[4] = {0, 0, 0, 0};
__device__ const dfloat UZ_BC[4] = {0, 0, 0, 0};
__device__ const dfloat RHO_BC[4] = {RHO_0, RHO_0, RHO_0, RHO_0};

#define BC_X_PERIODIC
#define BC_Y_WALL
#define BC_Z_PERIODIC

/* --------------------- NON-NEWTONIAN PHASE CONFIGURATION --------------------- */
static inline fluidPhaseProps makeCasePhaseProps(){
    fluidPhaseProps pp{};
    pp.nnf.type = FLUID_HERSCHEL_BULKLEY;
    pp.nnf.hasLambda = false;
    pp.nnf.u.hb.s_y = S_Y;
    pp.nnf.u.hb.n_index = N_INDEX;
    pp.nnf.u.hb.k_consistency = K_CONSISTENCY;
    pp.ve.type = VE_NEWTONIAN;
    return pp;
}
#define CASE_PHASE_PROPS makeCasePhaseProps()

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