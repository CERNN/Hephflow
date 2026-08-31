/* ============================================================================
 * CONSTANTS CONFIGURATION
 * Case: Single Particle Settling
 * Description: Particle settling under gravity in viscous fluid
 * ========================================================================== */

/* --------------------- SIMULATION SCALE --------------------- */
constexpr dfloat RE = 10;                   // Reynolds number (custom for settling)
constexpr int SCALE = 1;                    // Grid scaling factor

/* --------------------- TIME STEPPING --------------------- */
constexpr int N_STEPS = 100000;             // Total simulation steps
constexpr int INI_STEP = 0;                 // Initial step (for restarts)

/* --------------------- GRID DIMENSIONS --------------------- */
constexpr int N = 200;                      // Reference grid size
constexpr int NX = N;                       // Grid size in X
constexpr int NY = 320 * SCALE;             // Grid size in Y (irregular grid)
constexpr int NZ = N;                       // Grid size in Z
constexpr int NZ_TOTAL = NZ;                // Total grid size in Z (multi-GPU)

/* --------------------- REFERENCE VALUES --------------------- */
constexpr dfloat U_MAX = 0.01;              // Maximum/reference velocity
constexpr dfloat L = NY;                    // Reference length

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
constexpr dfloat VISC = U_MAX * L / RE;     // Kinematic viscosity
constexpr dfloat TAU = 0.5 + 3.0 * VISC;    // Relaxation time
constexpr dfloat MACH_NUMBER = U_MAX / 0.57735026918962; // Mach number

/* --------------------- CHECKPOINT AND INITIALIZATION --------------------- */
#define LOAD_CHECKPOINT false               // Load simulation checkpoint
#define CHECKPOINT_SAVE false               // Save checkpoint frequency
#define LOAD_FIELD false                    // Load field data
#define LOAD_FIELD_STEP 0                   // Field load step
#define SIMULATION_FOLDER_LOAD_CHECKPOINT "TEST" // Checkpoint folder

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

/* --------------------- GRAVITY PARAMETERS --------------------- */
constexpr dfloat GX = 0.0;                  // Gravity in X direction
constexpr dfloat GY = -0.0001;              // Gravity in Y direction (settling force)
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
