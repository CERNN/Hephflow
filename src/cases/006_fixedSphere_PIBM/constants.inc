/* ============================================================================
 * CONSTANTS CONFIGURATION
 * Case: 001_mountedCube
 * Description: Flow around a mounted cube on plate
 * ========================================================================== */

/* --------------------- SIMULATION SCALE --------------------- */
constexpr int SCALE = 1;                    // Grid scaling factor

/* --------------------- DIMENSIONLESS NUMBERS --------------------- */
constexpr dfloat RE = 0.005677714;                  // Reynolds number

/* --------------------- TIME STEPPING --------------------- */
constexpr int N_STEPS = 10000;              // Total simulation steps
constexpr int INI_STEP = 0;                 // Initial step (for restarts)

/* --------------------- GRID DIMENSIONS --------------------- */
constexpr int N = 256 * SCALE;              // Reference grid size
constexpr int NX = 32;                       // Grid size in X
constexpr int NY = 32;                       // Grid size in Y
constexpr int NZ = 128;                       // Grid size in Z
constexpr int NZ_TOTAL = NZ;                // Total grid size in Z (multi-GPU)

/* --------------------- REFERENCE VALUES --------------------- */
constexpr dfloat U_MAX = 0.0;              // Maximum/reference velocity
constexpr dfloat L = N;                     // Reference length

/* --------------------- BODY FORCES --------------------- */
constexpr dfloat FX = 0.0;                  // Force in X direction
constexpr dfloat FY = 0.0;                  // Force in Y direction
constexpr dfloat FZ = 1e-5;                  // Force in Z direction

/* --------------------- INITIAL CONDITIONS --------------------- */
constexpr dfloat RHO_0 = 1.0;               // Initial density
constexpr dfloat U_0_X = 0.0;               // Initial velocity X
constexpr dfloat U_0_Y = 0.0;               // Initial velocity Y
constexpr dfloat U_0_Z = 0.0;             // Initial velocity Z

/* --------------------- BOUNDARY CONDITION VALUES --------------------- */
__device__ const dfloat UX_BC[4] = {0, 0, 0, 0};
__device__ const dfloat UY_BC[4] = {0, 0, 0, 0};
__device__ const dfloat UZ_BC[4] = {0, 0, 0, 0};
__device__ const dfloat RHO_BC[4] = {RHO_0, RHO_0, RHO_0, RHO_0};

/* --------------------- BOUNDARY CONDITION TYPES --------------------- */
#define BC_X_PERIODIC                          // Free slip
#define BC_Y_PERIODIC                          // Wall
#define BC_Z_PERIODIC                          // Inlet/outlet

/* --------------------- FLUID PROPERTIES --------------------- */
constexpr dfloat VISC = U_MAX * 1 / RE;    // Kinematic viscosity
constexpr dfloat TAU = 0.5 + 3.0*VISC;      // Relaxation time
constexpr dfloat MACH_NUMBER = U_MAX/0.57735026918962; // Mach number

/* --------------------- CHECKPOINT AND LOAD CONFIGURATION --------------------- */
#define LOAD_CHECKPOINT false               // Load simulation from checkpoint
#define SIMULATION_FOLDER_LOAD_CHECKPOINT "TEST" // Folder to load checkpoint from
#define CHECKPOINT_SAVE false               // Save simulation checkpoints
#define LOAD_FIELD false                    // Load field data
#define LOAD_FIELD_STEP 0                   // Step to load field from

/* ============================================================================
 * PARTICLE/IBM MODEL PARAMETERS
 * Requires #define PARTICLE_MODEL in model.inc
 * Uses Immersed Boundary Method for fluid-solid interaction
 * ========================================================================== */

#define PARTICLE_MODEL                      // Enable particle simulation
#define PIBM_METHOD                          // Immersed Boundary Method
#define DEM_METHOD                          // Discrete Element Method for collisions
#define LOCAL_FORCES                        // Compute local forces on particles

/* --------------------- PARTICLE CONFIGURATION --------------------- */
#define NUM_PARTICLES 1                     // Number of particles

/* --------------------- PARTICLE PIBM CONFIGURATION --------------------- */
// Define Lagrangian lattice intervals, min of 2 nodes
const int LAGRANGIAN_I_MAX = 2; // min 2
const int LAGRANGIAN_J_MAX = 2; // min 2
const int LAGRANGIAN_K_MAX = 2; // min 2

const int FORCE_SPREAD_X_NODES = 2; // min 1
const int FORCE_SPREAD_Y_NODES = 2; // min 1
const int FORCE_SPREAD_Z_NODES = 2; // min 1

/* --------------------- GRAVITY PARAMETERS --------------------- */
constexpr dfloat GX = 0.0;                  // Gravity in X direction
constexpr dfloat GY = 0.0;              // Gravity in Y direction (settling force)
constexpr dfloat GZ = 0.0;                  // Gravity in Z direction

/* --------------------- FLUID PROPERTIES FOR PARTICLES --------------------- */
constexpr dfloat FLUID_DENSITY = 1.0;       // Fluid density (for buoyancy)

/* --------------------- DEM COLLISION PARAMETERS --------------------- */
// Friction coefficients
constexpr dfloat PP_FRICTION_COEF = 0.1;    // Particle-particle friction
constexpr dfloat PW_FRICTION_COEF = 0.1;    // Particle-wall friction

// Restitution coefficients
constexpr dfloat PP_REST_COEF = 0.9;        // Particle-particle restitution
constexpr dfloat PW_REST_COEF = 0.5;        // Particle-wall restitution (lower for settling)
//#define REST_COEF_CORRECTION             // Enable restitution correction

/* --------------------- MATERIAL PROPERTIES --------------------- */
// Particle material properties
constexpr dfloat PARTICLE_YOUNG_MODULUS = 1.0;  // Young's modulus (Pa)
constexpr dfloat PARTICLE_POISSON_RATIO = 0.3; // Poisson's ratio

// Wall material properties
constexpr dfloat WALL_YOUNG_MODULUS = 100.0;    // Young's modulus (Pa)
constexpr dfloat WALL_POISSON_RATIO = 0.3;     // Poisson's ratio

/* --------------------- COLLISION DAMPING CONSTANTS --------------------- */
// Sphere-wall collision constants (compile-time)
constexpr dfloat SPHERE_WALL_DAMPING_CONST = (-2.0 * constexprLn(PW_REST_COEF) / 
    (sqrtt(M_PI * M_PI + constexprLn(PW_REST_COEF) * constexprLn(PW_REST_COEF))));
constexpr dfloat SPHERE_SPHERE_DAMPING_CONST = (-2.0 * constexprLn(PP_REST_COEF) / 
    (sqrtt(M_PI * M_PI + constexprLn(PP_REST_COEF) * constexprLn(PP_REST_COEF))));

/* --------------------- LONG-RANGE INTERACTION PARAMETERS --------------------- */
// Lubrication force parameters
#define LUB_CUTOFF_FACTOR   0.02           // Cutoff factor (stop above this value)
#define LUB_MIN_FACTOR      1e-3            // Minimum factor (stop below for 1/h divergence)

// Repulsive force parameters
#define REP_RANGE_FACTOR    0.1             // Range factor (stop working above this)
#define REP_FORCE_AMPLITUDE 1.0             // Repulsion force amplitude

// Attractive force parameters
#define ATT_RANGE_FACTOR    0.1             // Range factor
#define HAMAKER_CONST 1e-5                  // Hamaker constant (lattice units)

/* ----- END CONSTANTS CONFIGURATION ----- */
