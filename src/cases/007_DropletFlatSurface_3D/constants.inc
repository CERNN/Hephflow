/* ============================================================================
 * CONSTANTS CONFIGURATION
 * Case: 3D Rayleigh-Taylor Instability verification
 * ========================================================================== */

/* --------------------- SIMULATION SCALE --------------------- */
constexpr int SCALE = 1;

/* --------------------- TIME STEPPING --------------------- */
constexpr int N_STEPS = 60000;
constexpr int INI_STEP = 0;

/* --------------------- GRID DIMENSIONS --------------------- */
constexpr int L = 128 * SCALE;            // Characteristic length [lu]
constexpr int NX = 2*L;                     // x in [-L/2, L/2]
constexpr int NY = L;                 // y in [-2L, 2L]
constexpr int NZ = 2*L;                     // z in [-L/2, L/2]
constexpr int NZ_TOTAL = NZ;

/* --------------------- BENCHMARK TARGETS (NON-DIMENSIONAL INPUTS) --------------------- */
constexpr dfloat ATWOOD_NUMBER = 0.5_df;      // At = (rho_l-rho_g)/(rho_l+rho_g)
constexpr dfloat REYNOLDS_NUMBER = 1024.0_df;  // Re = L*sqrt(gL)/nu_l
constexpr dfloat CAPILLARY_NUMBER = 0.2_df;   // Ca = mu_l*sqrt(gL)/sigma
constexpr dfloat PECLET_NUMBER = 744.0_df;    // Pe = L*sqrt(gL)/M
constexpr dfloat INTERFACE_WIDTH = 5.0_df;    // W [lu]

/* --------------------- REFERENCE VALUES --------------------- */
constexpr dfloat U_REF = 0.05_df;          // U_ref = sqrt(gL), kept low for low-Mach stability
constexpr dfloat U_MAX = U_REF;

constexpr dfloat vtk_time = U_REF/L;                   // Physical time per VTK step (if HAS_VTK_TIME defined)

/* --------------------- BODY FORCES --------------------- */
constexpr dfloat FX = 0.0_df;
constexpr dfloat FY = 0.0_df;
constexpr dfloat FZ = 0.0_df;

/* --------------------- INITIAL CONDITIONS --------------------- */
constexpr dfloat RHO_0 = 1.0_df;
constexpr dfloat U_0_X = 0.0_df;
constexpr dfloat U_0_Y = 0.0_df;
constexpr dfloat U_0_Z = 0.0_df;

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

// Atwood-locked density ratio: rho_l/rho_g = (1+At)/(1-At)
constexpr dfloat DENSITY_RATIO = (1.0_df + ATWOOD_NUMBER) / (1.0_df - ATWOOD_NUMBER);

constexpr dfloat PHI_RHO_PHASE2 = 1.5_df;
constexpr dfloat PHI_RHO_PHASE1 = PHI_RHO_PHASE2 / DENSITY_RATIO;
constexpr dfloat PHI_DRHO_PHASE12 = PHI_RHO_PHASE2 - PHI_RHO_PHASE1;

constexpr dfloat RHO_MID_REF = (PHI_RHO_PHASE1 + PHI_RHO_PHASE2) / 2.0_df;

// Paper setup uses viscosity ratio = 1 as kinematic viscosity ratio (nu_l/nu_g).
constexpr dfloat KINEMATIC_VISCOSITY_RATIO = 1.0_df;
constexpr dfloat NU_PHASE2 = ((dfloat)L * U_REF) / REYNOLDS_NUMBER;
constexpr dfloat NU_PHASE1 = NU_PHASE2 / KINEMATIC_VISCOSITY_RATIO;

constexpr dfloat MU_PHASE1 = PHI_RHO_PHASE1 * NU_PHASE1;
constexpr dfloat MU_PHASE2 = PHI_RHO_PHASE2 * NU_PHASE2;

constexpr dfloat DYNAMIC_VISCOSITY_RATIO = MU_PHASE2 / MU_PHASE1;

// Multiphase blending uses h_phi in [0,1]: h_phi=0 -> Phase 1, h_phi=1 -> Phase 2.
constexpr dfloat TAU_PHASE1 = 0.5_df + 3.0_df * NU_PHASE1;
constexpr dfloat TAU_PHASE2 = 0.5_df + 3.0_df * NU_PHASE2;

constexpr dfloat TAU = TAU_PHASE2;
constexpr dfloat VISC = (TAU - 0.5_df) / 3.0_df;
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
#define LOAD_CHECKPOINT false
#define CHECKPOINT_SAVE false
#define LOAD_FIELD false
#define LOAD_FIELD_STEP 0
#define SIMULATION_FOLDER_LOAD_CHECKPOINT "TEST"

/* ============================================================================
 * PHASE FIELD MODEL PARAMETERS
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
constexpr dfloat phi_del = INTERFACE_WIDTH;

/* --------------------- WETTING BOUNDARY CONFIGURATION --------------------- */

constexpr dfloat CONTACT_ANGLE_DEG = 60.0_df;
static __device__ dfloat contact_angle = CONTACT_ANGLE_DEG * ((dfloat)M_PI / 180.0_df);

// Derived transport coefficients from target Ca and Pe.
constexpr dfloat phi_sigma = (MU_PHASE2 * U_REF) / CAPILLARY_NUMBER;
constexpr dfloat M_CH = ((dfloat)L * U_REF) / PECLET_NUMBER;

constexpr dfloat kappa_CH = 0.75_df * phi_sigma * phi_del;
constexpr dfloat A_CH = (6.0_df * phi_sigma) / phi_del;

/* --------------------- DENSITY COUPLING --------------------- */
// g is derived from U_ref = sqrt(gL).
constexpr dfloat g_value = (U_REF * U_REF) / (dfloat)L;
// Downward gravity in y to destabilize heavy-over-light stratification.
__device__ const dfloat gravity_vector[3] = {0.0_df, -0.010_df, 0.0_df};

constexpr dfloat R0 = 20.0_df;

/* --------------------- FORCE COUPLING CONTROL FLAGS --------------------- */
#define PHI_FORCE_FS_ENABLED 1
#define PHI_FORCE_FP_ENABLED 1
#define PHI_FORCE_FN_ENABLED 1
#define PHI_FORCE_FB_ENABLED 1

/* --------------------- PHASE-FIELD TRANSPORT (LBM) --------------------- */
constexpr dfloat PHI_DIFFUSIVITY_RATIO = 1.0_df;
constexpr dfloat PHI_DIFFUSIVITY = VISC / PHI_DIFFUSIVITY_RATIO;
constexpr dfloat PHI_DIFF_REF = ((dfloat)L * U_REF) / PECLET_NUMBER;

constexpr dfloat PHI_TAU = 0.5_df + 3.0_df * PHI_DIFF_REF;
constexpr dfloat PHI_OMEGA = 1.0_df / PHI_TAU;

/* ----- END CONSTANTS CONFIGURATION ----- */
