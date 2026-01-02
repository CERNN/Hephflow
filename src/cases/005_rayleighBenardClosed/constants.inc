/* ============================================================================
 * CONSTANTS CONFIGURATION
 * Case: Rayleigh-Benard Convection (Closed)
 * Description: Thermal convection in closed domain with Bingham fluid model
 * ========================================================================== */

/* --------------------- SIMULATION SCALE --------------------- */
constexpr int SCALE = 1;                    // Grid scaling factor

/* --------------------- TIME STEPPING --------------------- */
constexpr int N_STEPS = 80000000;           // Total simulation steps
constexpr int INI_STEP = 0;                 // Initial step (for restarts)

/* --------------------- GRID DIMENSIONS --------------------- */
constexpr int N = 256 * SCALE;              // Reference grid size
constexpr int NX = N;                       // Grid size in X
constexpr int NY = N;                       // Grid size in Y
constexpr int NZ = N;                       // Grid size in Z
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
#define BC_X_WALL
#define BC_Y_WALL
#define BC_Z_WALL

/* --------------------- FLUID PROPERTIES --------------------- */
constexpr dfloat VISC = (0.8 - 0.5) / 3;    // Kinematic viscosity
constexpr dfloat TAU = 0.8;                 // Relaxation time
constexpr dfloat MACH_NUMBER = U_MAX / 0.57735026918962; // Mach number

/* ============================================================================
 * THERMAL FLUID PARAMETERS
 * Uncomment the section corresponding to the model enabled in model.inc
 * ========================================================================== */

/* --------------------- THERMAL MODEL DEFINITIONS --------------------- */
#define THERMAL_MODEL
#define SECOND_DIST
#define CONVECTION_DIFFUSION_TRANSPORT
#define D3G19

/* --------------------- BINGHAM MODEL --------------------- */
#define BINGHAM
constexpr dfloat Bn = 0.08;                 // Bingham number
constexpr dfloat S_Y = Bn*L/(U_MAX*VISC); // Yield stress
// Fluid properties macro for this case - expands to initializer list
#define CASE_NNF_PROPS { FLUID_BINGHAM, false, { S_Y, 1.0 / TAU } }

/* --------------------- THERMAL PARAMETERS --------------------- */
constexpr bool T_BOUYANCY = true;           // Enable buoyancy
constexpr dfloat T_PR_NUMBER = 10;          // Prandtl number
constexpr dfloat T_RA_NUMBER = 1e6;         // Rayleigh number
constexpr dfloat T_GR_NUMBER = T_RA_NUMBER / T_PR_NUMBER; // Grashof number

    constexpr dfloat T_DELTA_T = 0.1;         // Temperature difference
    constexpr dfloat T_REFERENCE = 1.0;      // Reference temperature
    constexpr dfloat T_COLD = T_REFERENCE - T_DELTA_T / 2.0;  // Cold wall temperature
    constexpr dfloat T_HOT = T_REFERENCE + T_DELTA_T / 2.0;   // Hot wall temperature
    constexpr dfloat C_0 = T_REFERENCE;      // Initial temperature field

    constexpr dfloat T_DIFFUSIVITY = VISC / T_PR_NUMBER;      // Thermal diffusivity (alpha)

    constexpr dfloat T_gravity_t_beta = T_RA_NUMBER * T_DIFFUSIVITY * VISC / (T_DELTA_T * L * L * L);
    constexpr dfloat Ra_conf = T_gravity_t_beta * T_DELTA_T * L * L * L * T_PR_NUMBER / (VISC * VISC);
    constexpr dfloat T_Q_INTERNAL_D_Cp = 0.0; // Internal heat source

    __device__ const dfloat gravity_vector[3] = {1.0, 0.0, 0.0}; // Gravity direction

    constexpr dfloat T_DIFF_REF = (1.0 / 6);
    constexpr dfloat G_TAU = T_DIFFUSIVITY * 3 + 0.5;
    constexpr dfloat G_OMEGA = 1.0 / G_TAU;
    constexpr dfloat G_TT_OMEGA = 1.0 - 0.5 * G_OMEGA;

    constexpr dfloat T_DIFF_FLUC = T_DIFFUSIVITY - T_DIFF_REF;
    constexpr dfloat G_AAA = (3 * T_DIFF_FLUC * G_OMEGA);
    constexpr dfloat G_DIFF_FLUC_COEF = G_AAA / (1.0 + G_AAA);
#endif //THERMAL_MODEL

/* ============================================================================
 * CHECKPOINT AND INITIALIZATION
 * ========================================================================== */

/* --------------------- CHECKPOINT LOADING --------------------- */
constexpr dfloat NNF_TRIGGER_STEP = 2000000;      // NNF trigger step
constexpr dfloat NNF_TRIGGER_STEP_SIZE = 2000000; // NNF step size
constexpr dfloat NNF_TRIGGER_STEP_END = NNF_TRIGGER_STEP + NNF_TRIGGER_STEP_SIZE; // NNF end step
#define LOAD_CHECKPOINT false                  // Load checkpoint from folder 
                                // (folder name defined below)


// Folder with simulation to load data from last checkpoint. 
// WITHOUT ID_SIM (change it in ID_SIM) AND "/" AT THE END
#define SIMULATION_FOLDER_LOAD_CHECKPOINT "TEST"
// Interval to make checkpoint to save all simulation data and restart from it.
// It must not be very frequent (10000 or more), because it takes a long time
#define CHECKPOINT_SAVE false // the frequency on which the simulation checkpoint is saved
#define LOAD_FIELD false
#define LOAD_FIELD_STEP 0