/* ============================================================================
 * CONSTANTS CONFIGURATION
 * Case: [CASE_NAME]
 * Description: [Brief description of the simulation]
 * ========================================================================== */

/* --------------------- SIMULATION SCALE --------------------- */
constexpr dfloat RE = 100;                  // Reynolds number
constexpr int SCALE = 1;                    // Grid scaling factor

/* --------------------- TIME STEPPING --------------------- */
constexpr int N_STEPS = 10000;              // Total simulation steps
constexpr int INI_STEP = 0;                 // Initial step (for restarts)

/* --------------------- GRID DIMENSIONS --------------------- */
constexpr int N = 64 * SCALE;               // Reference grid size
constexpr int NX = N;                       // Grid size in X
constexpr int NY = N;                       // Grid size in Y
constexpr int NZ = N;                       // Grid size in Z (single GPU)
constexpr int NZ_TOTAL = NZ;                // Total grid size in Z (multi-GPU)

/* --------------------- REFERENCE VALUES --------------------- */
constexpr dfloat U_MAX = 0.01;              // Maximum/reference velocity
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
// Array indices: [0]=X-, [1]=X+, [2]=Y-, [3]=Y+, etc.
__device__ const dfloat UX_BC[4] = {0, 0, 0, 0};
__device__ const dfloat UY_BC[4] = {0, 0, 0, 0};
__device__ const dfloat UZ_BC[4] = {0, 0, 0, 0};
__device__ const dfloat RHO_BC[4] = {RHO_0, RHO_0, RHO_0, RHO_0};

/* --------------------- BOUNDARY CONDITION TYPES --------------------- */
// Options: BC_X_WALL, BC_X_PERIODIC, BC_X_PRESSURE
#define BC_X_WALL
#define BC_Y_WALL
#define BC_Z_WALL

/* --------------------- FLUID PROPERTIES --------------------- */
// NOTE: Always define VISC first, then compute TAU from VISC
constexpr dfloat VISC = U_MAX * L / RE;     // Kinematic viscosity
constexpr dfloat TAU = 0.5 + 3.0 * VISC;    // Relaxation time

/* --------------------- DIMENSIONLESS NUMBERS --------------------- */
constexpr dfloat MACH_NUMBER = U_MAX / 0.57735026918962;

/* ============================================================================
 * NON-NEWTONIAN FLUID PARAMETERS
 * Uncomment the section corresponding to the model enabled in model.inc
 * See PHYSICS_MODELS.md for detailed descriptions
 * ========================================================================== */

/* --------------------- POWER-LAW MODEL --------------------- */
#ifdef POWERLAW
constexpr dfloat N_INDEX = 0.5;                          // Power-law index (n < 1: shear-thinning)
constexpr dfloat K_CONSISTENCY = RHO_0*(TAU-0.5)/3.0;    // Consistency factor
constexpr dfloat GAMMA_0 = 0.0;                          // Truncation (0 = no truncation)
#define OMEGA_LAST_STEP                                  // Uses omega from previous step
static inline fluidProps makeCaseNnfProps(){
    fluidProps fp{};
    fp.type = FLUID_POWERLAW;
    fp.hasLambda = false;
    fp.u.powerlaw.n_index = N_INDEX;
    fp.u.powerlaw.k_consistency = K_CONSISTENCY;
    fp.u.powerlaw.gamma_0 = GAMMA_0;
    return fp;
}
#define CASE_NNF_PROPS makeCaseNnfProps()
#endif

/* --------------------- BINGHAM MODEL --------------------- */
#ifdef BINGHAM
constexpr dfloat Bn = 0.1;                               // Bingham number
constexpr dfloat S_Y = Bn * U_MAX * U_MAX;               // Yield stress
constexpr dfloat OMEGA_P = 1.0 / (3.0*VISC + 0.5);       // Plastic relaxation parameter
static inline fluidProps makeCaseNnfProps(){
    fluidProps fp{};
    fp.type = FLUID_BINGHAM;
    fp.hasLambda = false;
    fp.u.bingham.s_y = S_Y;
    fp.u.bingham.omega_p = OMEGA_P;
    return fp;
}
#define CASE_NNF_PROPS makeCaseNnfProps()
#endif

/* --------------------- BI-VISCOSITY MODEL --------------------- */
#ifdef BI_VISCOSITY
constexpr dfloat Bn = 0.4;                               // Bingham number
constexpr dfloat VISC_RATIO = 1.0/1000.0;                // μ/μ_y ratio
constexpr dfloat ETA_Y = VISC / VISC_RATIO;              // High shear viscosity
constexpr dfloat TAU_Y = 3.0*ETA_Y + 0.5;                // High shear relaxation time
constexpr dfloat OMEGA_Y = 1.0/TAU_Y;                    // High shear omega
constexpr dfloat OMEGA_P = 1.0 / (3.0*VISC + 0.5);       // Low shear omega
// Note: S_Y and GAMMA_C require thermal parameters if using thermal coupling
static inline fluidProps makeCaseNnfProps(){
    fluidProps fp{};
    fp.type = FLUID_BI_VISCOSITY;
    fp.hasLambda = false;
    fp.u.bi.s_y = Bn * U_MAX * U_MAX;
    fp.u.bi.visc_ratio = VISC_RATIO;
    fp.u.bi.eta_y = ETA_Y;
    fp.u.bi.tau_y = TAU_Y;
    fp.u.bi.omega_y = OMEGA_Y;
    fp.u.bi.omega_p = OMEGA_P;
    fp.u.bi.gamma_c = 0.0_df;
    return fp;
}
#define CASE_NNF_PROPS makeCaseNnfProps()
#endif

/* --------------------- KEE-TURCOTTE MODEL (EXPERIMENTAL) --------------------- */
#ifdef KEE_TURCOTEE
constexpr dfloat S_Y = 0.0;                              // Yield stress
constexpr dfloat t1 = 1e-3;                              // Time scale parameter
constexpr dfloat eta_0 = 1e-3;                           // Reference viscosity
static inline fluidProps makeCaseNnfProps(){
    fluidProps fp{};
    fp.type = FLUID_KEE_TURCOTEE;
    fp.hasLambda = false;
    fp.u.kee.s_y = S_Y;
    fp.u.kee.t1 = t1;
    fp.u.kee.eta_0 = eta_0;
    return fp;
}
#define CASE_NNF_PROPS makeCaseNnfProps()
#endif

/* --------------------- THIXOTROPIC (LAMBDA) MODEL --------------------- */
#ifdef LAMBDA_MODEL
// Transported lambda field requires LAMBDA_DIST and LAMBDA_MODEL in model.inc
// Provide build and break rates plus equilibrium structure parameter
constexpr dfloat LAMBDA_K1 = 3.0e-4;                      // Breakdown rate
constexpr dfloat LAMBDA_K2 = 1.0;                        // Build-up rate
constexpr dfloat THIXO_LAMBDA_0 = 1.0;                   // Equilibrium lambda

// Map the thixotropic parameters into fluidProps for host/device consumption
static inline fluidProps makeCaseNnfProps(){
    fluidProps fp{};
    fp.type = FLUID_THIXO;
    fp.hasLambda = true;
    fp.u.thixo.model = THIXO_MOORE1959;
    fp.u.thixo.u.moore1959.k1 = LAMBDA_K1;
    fp.u.thixo.u.moore1959.k2 = LAMBDA_K2;
    fp.u.thixo.u.moore1959.lambda_0 = THIXO_LAMBDA_0;
    fp.u.thixo.u.moore1959.eta_0 = 1.0_df;
    return fp;
}
#define CASE_NNF_PROPS makeCaseNnfProps()
#endif


/* ============================================================================
 * VISCOELASTIC FLUID PARAMETERS (FENE-P)
 * Requires #define FENE_P in model.inc
 * Uses additional distributions for conformation tensor components
 * ========================================================================== */

#ifdef FENE_P
// Model selection
#define OLDROYD_B                                        // Oldroyd-B constitutive model
#define CONVECTION_DIFFUSION_TRANSPORT                   // Transport equation method
#define D3G19                                            // Velocity set for conformation

// Conformation tensor components (6 symmetric components)
#define CONFORMATION_TENSOR
#define A_XX_DIST                                        // Axx component distribution
#define A_XY_DIST                                        // Axy component distribution
#define A_XZ_DIST                                        // Axz component distribution
#define A_YY_DIST                                        // Ayy component distribution
#define A_YZ_DIST                                        // Ayz component distribution
#define A_ZZ_DIST                                        // Azz component distribution

// Gradient computation methods
#define COMPUTE_CONF_GRADIENT_FINITE_DIFFERENCE          // Conformation gradient
#define COMPUTE_VEL_GRADIENT_FINITE_DIFFERENCE           // Velocity gradient

// Viscosity parameters
constexpr dfloat SUM_VISC = (U_MAX * L) / RE;            // Total viscosity (ηs + ηp)
constexpr dfloat BETA = 0.5;                             // Viscosity ratio (ηs/ηtotal)
constexpr dfloat nu_p = SUM_VISC - VISC;                 // Polymer viscosity

// Characteristic times
constexpr dfloat t_c = L / U_MAX;                        // Characteristic time
constexpr dfloat Weissenberg_number = 1.0;               // Weissenberg number (Wi = λ/t_c)
constexpr dfloat LAMBDA = Weissenberg_number * t_c;      // Polymer relaxation time
constexpr dfloat inv_lambda = 1.0 / LAMBDA;              // Inverse relaxation time
constexpr dfloat fenep_re = 10.0;                        // FENE-P extensibility (L²)

// Conformation tensor initialization and transport
constexpr int CONFORMATION_SWITCH = 0;                   // Step to activate conformation
constexpr dfloat CONF_ZERO = 10.0;                       // Initial conformation value

// Conformation diffusion parameters
constexpr dfloat CONF_DIFFUSIVITY_RATIO = 100;           // Schmidt-like number
constexpr dfloat CONF_DIFFUSIVITY = VISC / CONF_DIFFUSIVITY_RATIO;
constexpr dfloat CONF_DIFF_REF = 1.0 / 6.0;              // Reference diffusion
constexpr dfloat CONF_TAU = CONF_DIFF_REF * 3.0 + 0.5;   // Conformation relaxation time
constexpr dfloat CONF_OMEGA = 1.0 / CONF_TAU;            // Conformation collision frequency
constexpr dfloat CONF_TT_OMEGA = 1.0 - 0.5 * CONF_OMEGA; // TRT parameter
#endif


/* ============================================================================
 * THERMAL MODEL PARAMETERS
 * Requires #define THERMAL_MODEL in model.inc
 * Uses second distribution for temperature field
 * ========================================================================== */

#ifdef THERMAL_MODEL
// Distribution setup
#define SECOND_DIST                                      // Enable second distribution
#define CONVECTION_DIFFUSION_TRANSPORT                   // Transport equation method
#define D3G19                                            // Velocity set for temperature

// Buoyancy control
constexpr bool T_BOUYANCY = true;                        // Enable buoyancy force

// Dimensionless numbers
constexpr dfloat T_PR_NUMBER = 0.71;                     // Prandtl number (Pr)
constexpr dfloat T_RA_NUMBER = 1e5;                      // Rayleigh number (Ra)
constexpr dfloat T_GR_NUMBER = T_RA_NUMBER/T_PR_NUMBER;  // Grashof number (Gr = Ra/Pr)

// Temperature scales
constexpr dfloat T_DELTA_T = 1.0;                        // Temperature difference (ΔT)
constexpr dfloat T_REFERENCE = 1.0;                      // Reference temperature (Boussinesq)
constexpr dfloat T_COLD = T_REFERENCE - T_DELTA_T/2.0;   // Cold wall temperature
constexpr dfloat T_HOT = T_REFERENCE + T_DELTA_T/2.0;    // Hot wall temperature
constexpr dfloat C_0 = T_REFERENCE;                      // Initial temperature field

// Thermal properties
constexpr dfloat T_DIFFUSIVITY = VISC/T_PR_NUMBER;       // Thermal diffusivity (α)
constexpr dfloat T_Q_INTERNAL_D_Cp = 0.0;                // Internal heat source/Cp

// Buoyancy force calculation
constexpr dfloat T_gravity_t_beta = T_RA_NUMBER * T_DIFFUSIVITY*VISC/(T_DELTA_T*L*L*L);
constexpr dfloat Ra_conf = T_gravity_t_beta * T_DELTA_T*L*L*L*T_PR_NUMBER/(VISC*VISC);
__device__ const dfloat gravity_vector[3] = {0.0, 1.0, 0.0};  // Gravity direction

// Temperature distribution LBM parameters
constexpr dfloat T_DIFF_REF = 1.0/6.0;                   // Reference diffusion
constexpr dfloat G_TAU = T_DIFF_REF*3.0 + 0.5;           // Temperature relaxation time
constexpr dfloat G_OMEGA = 1.0/G_TAU;                    // Temperature collision frequency
constexpr dfloat G_TT_OMEGA = 1.0 - 0.5*G_OMEGA;         // TRT parameter
constexpr dfloat T_DIFF_FLUC = T_DIFFUSIVITY - T_DIFF_REF;
constexpr dfloat G_AAA = 3.0*T_DIFF_FLUC*G_OMEGA;
constexpr dfloat G_DIFF_FLUC_COEF = G_AAA/(1.0 + G_AAA); // Diffusivity correction
#endif

/* ============================================================================
 * PARTICLE/IBM MODEL PARAMETERS
 * Requires #define PARTICLE_MODEL in model.inc
 * Uses Immersed Boundary Method for fluid-solid interaction
 * ========================================================================== */

#ifdef PARTICLE_MODEL
// Particle methods
#define IBM_METHOD                                       // Immersed Boundary Method
#define DEM_METHOD                                       // Discrete Element Method for collisions
#define LOCAL_FORCES                                     // Compute local forces on particles

// Number of particles (also define NUM_PARTICLES macro)
#define NUM_PARTICLES 1

// Gravity vector
constexpr dfloat GX = 0.0;                               // Gravity in X direction
constexpr dfloat GY = 8.4e-4;                            // Gravity in Y direction
constexpr dfloat GZ = 0.0;                               // Gravity in Z direction

// Fluid properties
constexpr dfloat FLUID_DENSITY = 1.0;                    // Fluid density (for buoyancy)

// DEM collision parameters
#ifdef DEM_METHOD
// Friction coefficients
constexpr dfloat PP_FRICTION_COEF = 0.1;                 // Particle-particle friction
constexpr dfloat PW_FRICTION_COEF = 0.1;                 // Particle-wall friction

// Restitution coefficients
constexpr dfloat PP_REST_COEF = 0.9;                     // Particle-particle restitution
constexpr dfloat PW_REST_COEF = 0.9;                     // Particle-wall restitution
//#define REST_COEF_CORRECTION                           // Enable restitution correction

// Material properties - Particles
constexpr dfloat PARTICLE_YOUNG_MODULUS = 0.5;           // Young's modulus (Pa)
constexpr dfloat PARTICLE_POISSON_RATIO = 0.3;           // Poisson's ratio

// Material properties - Walls
constexpr dfloat WALL_YOUNG_MODULUS = 100.0;             // Young's modulus (Pa)
constexpr dfloat WALL_POISSON_RATIO = 0.3;               // Poisson's ratio

// Collision damping constants (computed from restitution coefficients)
constexpr dfloat SPHERE_WALL_DAMPING_CONST = (-2.0 * constexprLn(PW_REST_COEF) / 
    (sqrtt(M_PI * M_PI + constexprLn(PW_REST_COEF) * constexprLn(PW_REST_COEF))));
constexpr dfloat SPHERE_SPHERE_DAMPING_CONST = (-2.0 * constexprLn(PP_REST_COEF) / 
    (sqrtt(M_PI * M_PI + constexprLn(PP_REST_COEF) * constexprLn(PP_REST_COEF))));
#endif
#endif
