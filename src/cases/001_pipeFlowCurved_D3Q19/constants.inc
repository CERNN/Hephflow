/* ============================================================================
 * CONSTANTS CONFIGURATION
 * Case: 001_pipeFlowCurved_D3Q19
 * Description: Pipe flow with curved boundary condition
 * ========================================================================== */

/* --------------------- CURVED BOUNDARY CONDITION --------------------- */
#define CURVED_BOUNDARY_CONDITION

/* --------------------- SIMULATION SCALE --------------------- */
constexpr int SCALE = 1;                    // Grid scaling factor

/* --------------------- DIMENSIONLESS NUMBERS --------------------- */
constexpr dfloat RE = 10;                   // Reynolds number

/* --------------------- TIME STEPPING --------------------- */
constexpr int N_STEPS = 10000;              // Total simulation steps
constexpr int INI_STEP = 0;                 // Initial step (for restarts)

/* --------------------- GRID DIMENSIONS --------------------- */
constexpr int N = 64 * SCALE;               // Reference grid size
constexpr int NX = N;                       // Grid size in X
constexpr int NY = N;                       // Grid size in Y
constexpr int NZ = 8;                       // Grid size in Z
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
#define BC_X_WALL
#define BC_Y_WALL
#define BC_Z_PERIODIC

/* --------------------- FLUID PROPERTIES --------------------- */
constexpr dfloat VISC = 0.1;                // Direct viscosity definition
constexpr dfloat TAU = 0.5 + 3.0*VISC;      // Relaxation time
constexpr dfloat MACH_NUMBER = U_MAX/0.57735026918962; // Mach number

/* --------------------- CHECKPOINT AND LOAD CONFIGURATION --------------------- */
#define LOAD_CHECKPOINT false               // Load simulation from checkpoint
#define SIMULATION_FOLDER_LOAD_CHECKPOINT "TEST" // Folder to load checkpoint from
#define CHECKPOINT_SAVE false               // Save simulation checkpoints
#define LOAD_FIELD false                    // Load field data
#define LOAD_FIELD_STEP 0                   // Step to load field from

/* --------------------- EXTERNAL DUCT CONFIGURATION --------------------- */
#define FORCE_VOXEL_BC_BUILDING

#define EXTERNAL_DUCT_BC 
#define DUCT_CENTER_X (((NX-1)/2.0)+0.5)
#define DUCT_CENTER_Y (((NY-1)/2.0)+0.5)

#ifdef EXTERNAL_DUCT_BC 
    #define OUTER_RADIUS (NY/2.0-3.5)
    #define EXTERNAL_DUCT_BC_RADIUS OUTER_RADIUS
    #define OUTER_VELOCITY 0.0
    #define OUTER_ROTATION 0.0 //(U_MAX/OUTER_RADIUS)    
#endif //EXTERNAL_DUCT_BC