/* ============================================================================
 * CONSTANTS CONFIGURATION
 * Case: 000_ibmBenchmark
 * Description: IBM performance benchmark with 1000 particles in periodic domain
 * 
 * Note: Physics model flags are defined in model.inc
 * ========================================================================== */

/* --------------------- SIMULATION SCALE --------------------- */
constexpr dfloat RE = 500;                  // Reynolds number
constexpr int SCALE = 1;                    // Grid scaling factor

/* --------------------- TIME STEPPING --------------------- */
constexpr int N_STEPS = 1000;               // Total simulation steps
constexpr int INI_STEP = 0;                 // Initial step (for restarts)

/* --------------------- GRID DIMENSIONS --------------------- */
constexpr int N = 256 * SCALE;              // Reference grid size
constexpr int NX = N;                       // Grid size in X
constexpr int NY = N;                       // Grid size in Y
constexpr int NZ = N;                       // Grid size in Z (single GPU)
constexpr int NZ_TOTAL = NZ;                // Total grid size in Z (multi-GPU)

/* --------------------- REFERENCE VALUES --------------------- */
constexpr dfloat U_MAX = 0.0;               // Maximum/reference velocity
constexpr dfloat L = N;                     // Reference length

/* --------------------- BODY FORCES --------------------- */
constexpr dfloat FX = 0.0;                  // Force in X direction
constexpr dfloat FY = 0.0;                  // Force in Y direction
constexpr dfloat FZ = 1e-7;                 // Force in Z direction (flow direction)

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
#define BC_Y_PERIODIC
#define BC_Z_PERIODIC

/* --------------------- FLUID PROPERTIES --------------------- */
constexpr dfloat VISC = U_MAX * NX / RE;    // Kinematic viscosity
constexpr dfloat TAU = 0.8;                 // Relaxation time
constexpr dfloat MACH_NUMBER = U_MAX/0.57735026918962;

/* ============================================================================
 * PARTICLE/IBM MODEL PARAMETERS
 * Requires #define PARTICLE_MODEL in model.inc
 * ========================================================================== */

#ifdef PARTICLE_MODEL
#define IBM_METHOD
#define DEM_METHOD
#define LOCAL_FORCES

#define NUM_PARTICLES 1000                  // Number of particles

// Gravity acceleration
constexpr dfloat GX = 0.0;
constexpr dfloat GY = 0.0;
constexpr dfloat GZ = 0.0;

// Fluid density
constexpr dfloat FLUID_DENSITY = 1.0;

#ifdef DEM_METHOD
// Friction coefficients
constexpr dfloat PP_FRICTION_COEF = 0.1;    // Particle-particle friction
constexpr dfloat PW_FRICTION_COEF = 0.1;    // Particle-wall friction

// Restitution coefficients
constexpr dfloat PP_REST_COEF = 0.9;        // Particle-particle restitution
constexpr dfloat PW_REST_COEF = 0.9;        // Particle-wall restitution

// Material properties - Particles
constexpr dfloat PARTICLE_YOUNG_MODULUS = 0.5;
constexpr dfloat PARTICLE_POISSON_RATIO = 0.3;

// Material properties - Walls
constexpr dfloat WALL_YOUNG_MODULUS = 100.0;
constexpr dfloat WALL_POISSON_RATIO = 0.3;

// Damping constants (computed from restitution coefficients)
constexpr dfloat SPHERE_WALL_DAMPING_CONST = (-2.0 * constexprLn(PW_REST_COEF) / 
    (sqrtt(M_PI * M_PI + constexprLn(PW_REST_COEF) * constexprLn(PW_REST_COEF))));
constexpr dfloat SPHERE_SPHERE_DAMPING_CONST = (-2.0 * constexprLn(PP_REST_COEF) / 
    (sqrtt(M_PI * M_PI + constexprLn(PP_REST_COEF) * constexprLn(PP_REST_COEF))));
#endif
#endif
