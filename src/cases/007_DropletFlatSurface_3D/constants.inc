/* ============================================================================
 * CONSTANTS CONFIGURATION
 * Case: 3D droplet on flat surface (wetting verification)
 * ========================================================================== */

/* --------------------- SIMULATION SCALE --------------------- */
constexpr int SCALE = 1;

/* --------------------- TIME STEPPING --------------------- */
constexpr int N_STEPS = 300000;
constexpr int INI_STEP = 0;

/* --------------------- GRID DIMENSIONS --------------------- */
constexpr int NX = 256 * SCALE;
constexpr int NY = 128 * SCALE;
constexpr int NZ = 256 * SCALE;
constexpr int NZ_TOTAL = NZ;

/* --------------------- BENCHMARK INPUT PARAMETERS --------------------- */
constexpr dfloat CONTACT_ANGLE_DEG = 90.0_df;  // Sweep target contact angle by editing this value

constexpr dfloat RHO_PHASE2 = 1000.0_df;
constexpr dfloat DENSITY_RATIO_PHASE2_PHASE1 = 1000.0_df;
constexpr dfloat RHO_PHASE1 = RHO_PHASE2 / DENSITY_RATIO_PHASE2_PHASE1;

constexpr dfloat NU_PHASE2 = 0.01_df;
constexpr dfloat NU_PHASE1 = 0.1_df;

constexpr dfloat SIGMA = 0.2_df;
constexpr dfloat INTERFACE_WIDTH = 5.0_df;
constexpr dfloat MOBILITY = 0.1_df;
constexpr dfloat R0 = 20.0_df;

/* --------------------- REFERENCE VALUES --------------------- */
constexpr dfloat U_MAX = 0.1_df;
constexpr dfloat L = (dfloat)NX;

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
constexpr dfloat TAU_PHASE1 = 0.5_df + 3.0_df * NU_PHASE1;
constexpr dfloat TAU_PHASE2 = 0.5_df + 3.0_df * NU_PHASE2;
constexpr dfloat TAU = TAU_PHASE2;
constexpr dfloat VISC = NU_PHASE2;
constexpr dfloat MACH_NUMBER = U_MAX / 0.57735026918962_df;

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
constexpr dfloat phi_sigma = SIGMA;
constexpr dfloat phi_del = INTERFACE_WIDTH;
constexpr dfloat M_CH = MOBILITY;

constexpr dfloat kappa_CH = 0.75_df * phi_sigma * phi_del;
constexpr dfloat A_CH = (6.0_df * phi_sigma) / phi_del;

/* --------------------- DENSITY COUPLING --------------------- */
constexpr dfloat PHI_RHO_PHASE2 = RHO_PHASE2;
constexpr dfloat PHI_RHO_PHASE1 = RHO_PHASE1;
constexpr dfloat PHI_DRHO_PHASE12 = PHI_RHO_PHASE2 - PHI_RHO_PHASE1;

constexpr dfloat g_value = 0.0_df;
__device__ const dfloat gravity_vector[3] = {0.0_df, 0.0_df, 0.0_df};

/* --------------------- FORCE COUPLING CONTROL FLAGS --------------------- */
#define PHI_FORCE_FS_ENABLED 1
#define PHI_FORCE_FP_ENABLED 1
#define PHI_FORCE_FN_ENABLED 1
#define PHI_FORCE_FB_ENABLED 1

/* --------------------- PHASE-FIELD TRANSPORT (LBM) --------------------- */
constexpr dfloat PHI_DIFFUSIVITY_RATIO = 1.0_df;
constexpr dfloat PHI_DIFFUSIVITY = VISC / PHI_DIFFUSIVITY_RATIO;
constexpr dfloat PHI_DIFF_REF = M_CH;

constexpr dfloat PHI_TAU = 0.5_df + 3.0_df * PHI_DIFF_REF;
constexpr dfloat PHI_OMEGA = 1.0_df / PHI_TAU;

/* --------------------- WETTING PARAMETERS --------------------- */
constexpr dfloat CONTACT_ANGLE_RAD = CONTACT_ANGLE_DEG * (M_PI / 180.0_df);

/* ----- END CONSTANTS CONFIGURATION ----- */
