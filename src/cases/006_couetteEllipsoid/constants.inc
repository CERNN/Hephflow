/* ============================================================================
 * CONSTANTS CONFIGURATION
 * Case: Couette Flow with Ellipsoid Particle
 * Description: Particle simulation in couette flow with ellipsoidal geometry
 * ========================================================================== */

/* --------------------- SIMULATION SCALE --------------------- */
constexpr dfloat RE = 10;                   // Reynolds number
constexpr int SCALE = 1;                    // Grid scaling factor

/* --------------------- TIME STEPPING --------------------- */
constexpr int N_STEPS = 20000;              // Total simulation steps
constexpr int INI_STEP = 0;                 // Initial step (for restarts)

/* --------------------- GRID DIMENSIONS --------------------- */
constexpr int N = 64 * SCALE;               // Reference grid size
constexpr int NX = 2 * N;                   // Grid size in X
constexpr int NY = N;                       // Grid size in Y
constexpr int NZ = N;                       // Grid size in Z
constexpr int NZ_TOTAL = NZ;                // Total grid size in Z (multi-GPU)

/* --------------------- REFERENCE VALUES --------------------- */
constexpr dfloat U_MAX = 0.05;              // Maximum/reference velocity
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
#define BC_X_PERIODIC
#define BC_Y_WALL
#define BC_Z_PERIODIC

/* --------------------- FLUID PROPERTIES --------------------- */
constexpr dfloat VISC = NY * U_MAX / RE;    // Kinematic viscosity
constexpr dfloat TAU = 0.9;                 // Relaxation time
constexpr dfloat MACH_NUMBER = U_MAX / 0.57735026918962; // Mach number

/* --------------------- CHECKPOINT AND INITIALIZATION --------------------- */
#define LOAD_CHECKPOINT false               // Load simulation checkpoint
#define CHECKPOINT_SAVE false               // Save checkpoint frequency
#define LOAD_FIELD false                    // Load field data
#define LOAD_FIELD_STEP 0                   // Field load step
#define SIMULATION_FOLDER_LOAD_CHECKPOINT "TEST" // Checkpoint folder
#define WALL_VEL_UX U_MAX                   // Wall velocity in X

/* ============================================================================
 * PARTICLE/IBM MODEL PARAMETERS
 * Requires #define PARTICLE_MODEL in model.inc
 * Uses Immersed Boundary Method for fluid-solid interaction
 * ========================================================================== */

#define PARTICLE_MODEL                      // Enable particle simulation
#define IBM_METHOD                          // Immersed Boundary Method
#define DEM_METHOD                          // Discrete Element Method for collisions
#define LOCAL_FORCES                        // Compute local forces on particles

/* --------------------- PARTICLE CONFIGURATION --------------------- */
#define NUM_PARTICLES 1                     // Number of particles

/* --------------------- ELLIPSOID PARAMETERS --------------------- */
constexpr dfloat a_semi = 25.6;             // Semi-axis a
constexpr dfloat chi = 0.6;                 // Axis ratio
constexpr dfloat b_semi = a_semi*chi;       // Semi-axis b
constexpr dfloat c_semi = a_semi;           // Semi-axis c

/* --------------------- GRAVITY PARAMETERS --------------------- */
constexpr dfloat GX = 0.0;                  // Gravity in X direction
constexpr dfloat GY = 0.0;                  // Gravity in Y direction
constexpr dfloat GZ = 0.0;                  // Gravity in Z direction

/* --------------------- FLUID PROPERTIES FOR PARTICLES --------------------- */
constexpr dfloat FLUID_DENSITY = 1.0;       // Fluid density (for buoyancy)

/* --------------------- DEM COLLISION PARAMETERS --------------------- */
// Friction coefficients
constexpr dfloat PP_FRICTION_COEF = 0.1;    // Particle-particle friction
constexpr dfloat PW_FRICTION_COEF = 0.1;    // Particle-wall friction

// Restitution coefficients
constexpr dfloat PP_REST_COEF = 0.9;        // Particle-particle restitution
constexpr dfloat PW_REST_COEF = 0.9;        // Particle-wall restitution
//#define REST_COEF_CORRECTION             // Enable restitution correction

/* --------------------- MATERIAL PROPERTIES --------------------- */
// Particle material properties
constexpr dfloat PARTICLE_YOUNG_MODULUS = 0.5;  // Young's modulus (Pa)
constexpr dfloat PARTICLE_POISSON_RATIO = 0.3; // Poisson's ratio

// Wall material properties
constexpr dfloat WALL_YOUNG_MODULUS = 100.0;    // Young's modulus (Pa)
constexpr dfloat WALL_POISSON_RATIO = 0.3;     // Poisson's ratio

/* ----- END CONSTANTS CONFIGURATION ----- */