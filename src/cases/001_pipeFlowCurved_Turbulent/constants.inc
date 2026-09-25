/* ============================================================================
 * CONSTANTS CONFIGURATION
 * Case: 001_pipeFlowCurved_Turbulent
 * Description: Turbulent Newtonian flow through a circular duct
 * ========================================================================== */

/* --------------------- CURVED BOUNDARY CONDITION --------------------- */
#define CURVED_BOUNDARY_CONDITION

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

/* --------------------- DUCT GEOMETRY --------------------- */
constexpr dfloat DUCT_CENTER_X = ((NX-1)/2.0)+0.5;
constexpr dfloat DUCT_CENTER_Y = ((NY-1)/2.0)+0.5;
constexpr dfloat PIPE_RADIUS = NY/2.0-3.5;
constexpr int N_RADIAL_BINS = static_cast<int>(PIPE_RADIUS + 0.5);

/* --------------------- TURBULENCE PARAMETERS --------------------- */
constexpr dfloat RE = 180.0;                 // Friction Reynolds number Re_tau
constexpr dfloat U_TAU = 0.00579;            // Friction velocity
constexpr dfloat U_BULK = 15.6*U_TAU;        // Initial bulk-velocity scale
constexpr dfloat U_CENTERLINE = 18.2*U_TAU;  // Initial centerline velocity

/* --------------------- REFERENCE VALUES --------------------- */
constexpr dfloat U_MAX = U_CENTERLINE;       // Maximum/reference velocity
constexpr dfloat L = PIPE_RADIUS;            // Reference length

/* --------------------- BODY FORCES --------------------- */
constexpr dfloat FX = 0.0;                  // Force in X direction
constexpr dfloat FY = 0.0;                  // Force in Y direction
// For a fully developed pipe, tau_w = rho*FZ*R/2.
constexpr dfloat FZ = 2.0*U_TAU*U_TAU/PIPE_RADIUS;

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
constexpr dfloat VISC = PIPE_RADIUS*U_TAU/RE; // Re_tau = U_tau*R/VISC
constexpr dfloat TAU = 0.5 + 3.0*VISC;      // Relaxation time
constexpr dfloat MACH_NUMBER = U_MAX/0.57735026918962; // Mach number

static inline fluidPhaseProps makeCasePhaseProps(){
    fluidPhaseProps pp{};
    pp.nnf.type = FLUID_NEWTONIAN;
    pp.nnf.hasLambda = false;
    pp.ve.type = VE_NEWTONIAN;
    return pp;
}
#define CASE_PHASE_PROPS makeCasePhaseProps()

/* --------------------- CHECKPOINT AND LOAD CONFIGURATION --------------------- */
#define LOAD_CHECKPOINT false               // Load simulation from checkpoint
#define SIMULATION_FOLDER_LOAD_CHECKPOINT "TEST" // Folder to load checkpoint from
#define CHECKPOINT_SAVE false               // Save simulation checkpoints
#define LOAD_FIELD false                    // Load field data
#define LOAD_FIELD_STEP 0                   // Load field step

/* --------------------- STAIRCASE DUCT CONFIGURATION --------------------- */
#define FORCE_VOXEL_BC_BUILDING
#define EXTERNAL_DUCT_BC

#ifdef EXTERNAL_DUCT_BC
    #define OUTER_RADIUS PIPE_RADIUS
    #define EXTERNAL_DUCT_BC_RADIUS PIPE_RADIUS
    #define OUTER_VELOCITY 0.0
    #define OUTER_ROTATION 0.0
#endif // EXTERNAL_DUCT_BC
