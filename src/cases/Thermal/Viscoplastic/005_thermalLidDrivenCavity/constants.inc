/* ============================================================================
 * CONSTANTS CONFIGURATION
 * Case: 005_thermalViscoplasticBuoyantLidDrivenCavity
 * Description: 3D buoyant thermal viscoplastic lid-driven cavity
 * ========================================================================== */

/* --------------------- SIMULATION SCALE --------------------- */
constexpr int SCALE = 1;                    // Grid scaling factor

/* --------------------- DIMENSIONLESS NUMBERS --------------------- */
constexpr dfloat Reynolds_Number  = 10;                 // Reynolds number
constexpr dfloat Bingham_Number = 0;                  // Bingham number 
constexpr dfloat Grashof_Number = 10000;
constexpr dfloat Prandtl_Number = 1;
constexpr dfloat Rayleigh_Number = Grashof_Number * Prandtl_Number;


/* --------------------- TIME STEPPING --------------------- */
constexpr int N_STEPS = 500000;             // Total simulation steps
constexpr int INI_STEP = 0;                 // Initial step (for restarts)

/* --------------------- GRID DIMENSIONS --------------------- */
constexpr int N = 256 * SCALE;               // Reference grid size
constexpr int NX = N;                       // Grid size in X
constexpr int NY = N;                       // Grid size in Y
constexpr int NZ = N;                       // Grid size in Z
constexpr int NZ_TOTAL = NZ;                // Total grid size in Z (multi-GPU)

/* --------------------- REFERENCE VALUES --------------------- */
constexpr dfloat U_MAX = 0.0256;            // Maximum/reference velocity
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
constexpr dfloat VISC = U_MAX * NX / Reynolds_Number ;    // Kinematic viscosity
constexpr dfloat TAU = 0.5 + 3.0*VISC;      // Relaxation time
constexpr dfloat MACH_NUMBER = U_MAX/0.57735026918962; // Mach number
constexpr dfloat S_Y = Bingham_Number * VISC * U_MAX / L;      // Yield stress (Bingham)

static inline fluidPhaseProps makeCasePhaseProps(){
        fluidPhaseProps pp{};
        pp.nnf.type = FLUID_BINGHAM;
        pp.nnf.hasLambda = false;
        pp.nnf.u.bingham.s_y = S_Y;
        pp.nnf.u.bingham.omega_p = 1.0 / TAU;
        pp.ve.type = VE_NEWTONIAN;
        return pp;
}
#define CASE_PHASE_PROPS makeCasePhaseProps()

/* ============================================================================
 * THERMAL FLUID PARAMETERS
 * Uncomment the section corresponding to the model enabled in model.inc
 * ========================================================================== */

/* --------------------- THERMAL MODEL DEFINITIONS --------------------- */
#define THERMAL_MODEL
#define SECOND_DIST
#define CONVECTION_DIFFUSION_TRANSPORT
#define D3G19


/* --------------------- THERMAL PARAMETERS --------------------- */
#ifdef THERMAL_MODEL
constexpr bool T_BOUYANCY = true;            // Enable buoyancy
constexpr dfloat T_PR_NUMBER = Prandtl_Number;        // Prandtl number
constexpr dfloat T_RA_NUMBER = Rayleigh_Number;     // Rayleigh number
constexpr dfloat T_GR_NUMBER = T_RA_NUMBER / T_PR_NUMBER; // Grashof number

constexpr dfloat T_DELTA_T = 0.1;           // Temperature difference
constexpr dfloat T_REFERENCE = 1.0;         // Reference temperature
constexpr dfloat T_COLD = T_REFERENCE - T_DELTA_T / 2.0;  // Cold wall temperature
constexpr dfloat T_HOT = T_REFERENCE + T_DELTA_T / 2.0;   // Hot wall temperature
constexpr dfloat C_0 = T_REFERENCE;         // Initial temperature field

constexpr dfloat T_DIFFUSIVITY = VISC / T_PR_NUMBER;      // Thermal diffusivity (alpha)

constexpr dfloat T_gravity_t_beta = 2 * T_RA_NUMBER * T_DIFFUSIVITY * VISC / (T_DELTA_T * L * L * L);
constexpr dfloat Ra_conf = T_gravity_t_beta * T_DELTA_T * L * L * L * T_PR_NUMBER / (VISC * VISC);
constexpr dfloat T_Q_INTERNAL_D_Cp = 0.0; // Internal heat source

__device__ const dfloat gravity_vector[3] = {0.0, 1.0, 0.0}; // Gravity direction

constexpr dfloat G_TAU = T_DIFFUSIVITY * 3 + 0.5;
constexpr dfloat G_OMEGA = 1.0 / G_TAU;
constexpr dfloat G_TT_OMEGA = 1.0 - 0.5 * G_OMEGA;

#endif //THERMAL_MODEL

/* --------------------- CHECKPOINT AND LOAD CONFIGURATION --------------------- */
#define LOAD_CHECKPOINT false               // Load simulation from checkpoint
#define SIMULATION_FOLDER_LOAD_CHECKPOINT "TEST" // Folder to load checkpoint from
#define CHECKPOINT_SAVE false               // Save simulation checkpoints
#define LOAD_FIELD false                    // Load field data
#define LOAD_FIELD_STEP 0                   // Step to load field from
