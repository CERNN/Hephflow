/* ============================================================================
 * CONSTANTS CONFIGURATION
 * Case: Oscillating Droplet
 * Description: Two-phase flow simulation of oscillating droplet
 * ========================================================================== */

/* --------------------- SIMULATION SCALE --------------------- */
constexpr int SCALE = 1;                    // Grid scaling factor

/* --------------------- TIME STEPPING --------------------- */
constexpr int N_STEPS = 100000;               // Total simulation steps
constexpr int INI_STEP = 0;                 // Initial step (for restarts)

/* --------------------- GRID DIMENSIONS --------------------- */
constexpr int N = 128 * SCALE;              // Reference grid size
constexpr int NX = 3*N;                       // Grid size in X
constexpr int NY = N;                       // Grid size in Y
constexpr int NZ = N;                       // Grid size in Z
constexpr int NZ_TOTAL = NZ;                // Total grid size in Z (multi-GPU)

/* --------------------- REFERENCE VALUES --------------------- */
constexpr dfloat U_MAX = 0.1;               // Maximum/reference velocity
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
#define BC_Y_PERIODIC
#define BC_Z_PERIODIC

/* --------------------- FLUID PROPERTIES --------------------- */

constexpr dfloat PHI_RHO_PHASE2 = 1.0_df;  // Phase 2 density (heavier phase)
constexpr dfloat PHI_RHO_PHASE1 = 0.9_df;  // Phase 1 density (lighter phase)
constexpr dfloat PHI_DRHO_PHASE12 = PHI_RHO_PHASE2 - PHI_RHO_PHASE1;

constexpr dfloat RHO_MID_REF = (PHI_RHO_PHASE1 + PHI_RHO_PHASE2) / 2.0_df;

// Paper setup uses viscosity ratio = 1 as kinematic viscosity ratio (nu_l/nu_g).
constexpr dfloat KINEMATIC_VISCOSITY_RATIO = 1.0_df;
constexpr dfloat TAU = 0.55; 
constexpr dfloat VISC  = (TAU - 0.5_df) / 3.0_df;
constexpr dfloat NU_PHASE2 = VISC;
constexpr dfloat NU_PHASE1 = VISC;

constexpr dfloat MU_PHASE1 = PHI_RHO_PHASE1 * NU_PHASE1;
constexpr dfloat MU_PHASE2 = PHI_RHO_PHASE2 * NU_PHASE2;

constexpr dfloat DYNAMIC_VISCOSITY_RATIO = MU_PHASE2 / MU_PHASE1;

// Multiphase blending uses h_phi in [0,1]: h_phi=0 -> Phase 1, h_phi=1 -> Phase 2.
constexpr dfloat TAU_PHASE1 = 0.5_df + 3.0_df * NU_PHASE1;
constexpr dfloat TAU_PHASE2 = 0.5_df + 3.0_df * NU_PHASE2;

constexpr dfloat MACH_NUMBER = U_MAX / 0.57735026918962_df;

// Explicit phase properties keep this case compatible with two-phase setup utilities.
static inline fluidPhaseProps makeCasePhasePropsPhase1(){
	fluidPhaseProps pp{};
	pp.nnf.type = FLUID_BINGHAM;
	pp.nnf.hasLambda = false;
	pp.nnf.u.bingham.s_y = 0.0_df;
	pp.nnf.u.bingham.omega_p = 1.0_df / TAU_PHASE1;
	pp.ve.type = VE_NEWTONIAN;
	return pp;
}

static inline fluidPhaseProps makeCasePhasePropsPhase2(){
	fluidPhaseProps pp{};
	pp.nnf.type = FLUID_BINGHAM;
	pp.nnf.hasLambda = false;
	pp.nnf.u.bingham.s_y = 0.0_df;
	pp.nnf.u.bingham.omega_p = 1.0_df / TAU_PHASE2;
	pp.ve.type = VE_NEWTONIAN;
	return pp;
}

#define CASE_PHASE_PROPS_PHASE1 makeCasePhasePropsPhase1()
#define CASE_PHASE_PROPS_PHASE2 makeCasePhasePropsPhase2()

/* --------------------- CHECKPOINT AND INITIALIZATION --------------------- */
#define LOAD_CHECKPOINT false               // Load simulation checkpoint
#define CHECKPOINT_SAVE false               // Save checkpoint frequency
#define LOAD_FIELD false                    // Load field data
#define LOAD_FIELD_STEP 0                   // Field load step
#define SIMULATION_FOLDER_LOAD_CHECKPOINT "TEST" // Checkpoint folder

/* ============================================================================
 * PHASE FIELD MODEL PARAMETERS
 * Requires #define PHASE_MODEL in model.inc
 * Uses second distribution for phase field tracking
 * ========================================================================== */

#define PHASE_MODEL
#define INTERFACE_SHARPENING
#define CONVECTION_DIFFUSION_TRANSPORT
#define D3G19

#define PHI_DIST
#define METHOD_CAHN_HILLIARD


/* --------------------- PHASE FIELD REFERENCE VALUES --------------------- */
constexpr dfloat PHI_REFERENCE = 0.0_df;
constexpr dfloat PHI_DELTA_PHI = 2.0_df;
constexpr dfloat PHI_ONE = PHI_REFERENCE - PHI_DELTA_PHI / 2.0_df;
constexpr dfloat PHI_TWO = PHI_REFERENCE + PHI_DELTA_PHI / 2.0_df;
constexpr dfloat PHI_ZERO = 0.0_df;
constexpr dfloat PHI_SCALE = 2.0_df / PHI_DELTA_PHI;

/* --------------------- SURFACE TENSION AND INTERFACE PARAMETERS --------------------- */
constexpr dfloat phi_del = 4.0_df;  // Interface width (in lattice units)


// Derived transport coefficients from target Ca and Pe.
constexpr dfloat phi_sigma = 0.0003;
constexpr dfloat phi_xi = phi_del / 2.94;
constexpr dfloat M_CH = 1.0_df / (2.0_df * 1.41421356237309_df * phi_xi);
constexpr dfloat kappa_CH =  0.75_df * phi_sigma * phi_del;
constexpr dfloat A_CH = 6.0_df * phi_sigma / phi_del;

/* --------------------- DENSITY COUPLING --------------------- */
// g is derived from U_ref = sqrt(gL).
constexpr dfloat g_value = 0.0_df;
// Downward gravity in y to destabilize heavy-over-light stratification.
__device__ const dfloat gravity_vector[3] = {0.0_df, 0.0_df, 0.0_df};

constexpr dfloat R0 = 20.0_df;

/* --------------------- FORCE COUPLING CONTROL FLAGS --------------------- */
#define PHI_FORCE_FS_ENABLED 1
#define PHI_FORCE_FP_ENABLED 1
#define PHI_FORCE_FN_ENABLED 1
#define PHI_FORCE_FB_ENABLED 1

/* --------------------- PHASE-FIELD TRANSPORT (LBM) --------------------- */
constexpr dfloat PECLET_NUMBER = 10.0_df;
constexpr dfloat PHI_DIFF_REF = VISC / PECLET_NUMBER;
constexpr dfloat PHI_DIFFUSIVITY = VISC / PECLET_NUMBER;
constexpr dfloat PHI_DIFFUSIVITY_RATIO = PECLET_NUMBER;

constexpr dfloat PHI_TAU = 0.5_df + 3.0_df * PHI_DIFF_REF;
constexpr dfloat PHI_OMEGA = 1.0_df / PHI_TAU;

/* ----- END CONSTANTS CONFIGURATION ----- */

