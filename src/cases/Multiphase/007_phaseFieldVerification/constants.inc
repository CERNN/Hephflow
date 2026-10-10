/* Compact periodic case used by the Allen-Cahn correctness suite. */

#define PHASE_TEST_PLANAR 1
#define PHASE_TEST_TRANSLATION 2
#define PHASE_TEST_DROPLET 3
#define PHASE_TEST_CAPILLARY_DROPLET 4

#ifndef PHASE_TEST_SCENARIO
    #define PHASE_TEST_SCENARIO PHASE_TEST_PLANAR
#endif
#ifndef PHASE_TEST_N
    #define PHASE_TEST_N 32
#endif
#ifndef PHASE_TEST_STEPS
    #define PHASE_TEST_STEPS 1000
#endif
#ifndef PHASE_TEST_WIDTH
    #define PHASE_TEST_WIDTH 6.0_df
#endif
#ifndef PHASE_TEST_MOBILITY
    #define PHASE_TEST_MOBILITY 0.05_df
#endif

constexpr int NX = PHASE_TEST_N;
constexpr int NY = PHASE_TEST_N;
constexpr int NZ = PHASE_TEST_N;
constexpr int NZ_TOTAL = NZ;
constexpr int N_STEPS = PHASE_TEST_STEPS;
constexpr int INI_STEP = 0;

constexpr dfloat RHO_0 = 1.0_df;
constexpr dfloat PHASE_TEST_ADVECT_SPEED = 0.04_df;
#if PHASE_TEST_SCENARIO == PHASE_TEST_TRANSLATION
constexpr dfloat U_0_X = PHASE_TEST_ADVECT_SPEED;
constexpr dfloat U_0_Y = PHASE_TEST_ADVECT_SPEED;
constexpr dfloat U_0_Z = 0.0_df;
#else
constexpr dfloat U_0_X = 0.0_df;
constexpr dfloat U_0_Y = 0.0_df;
constexpr dfloat U_0_Z = 0.0_df;
#endif
constexpr dfloat U_MAX = PHASE_TEST_ADVECT_SPEED;
constexpr dfloat L = (dfloat)PHASE_TEST_N;

constexpr dfloat FX = 0.0_df;
constexpr dfloat FY = 0.0_df;
constexpr dfloat FZ = 0.0_df;
__device__ const dfloat UX_BC[4] = {0.0_df, 0.0_df, 0.0_df, 0.0_df};
__device__ const dfloat UY_BC[4] = {0.0_df, 0.0_df, 0.0_df, 0.0_df};
__device__ const dfloat UZ_BC[4] = {0.0_df, 0.0_df, 0.0_df, 0.0_df};
__device__ const dfloat RHO_BC[4] = {RHO_0, RHO_0, RHO_0, RHO_0};

#define BC_X_PERIODIC
#define BC_Y_PERIODIC
#define BC_Z_PERIODIC

constexpr dfloat TAU = 0.8_df;
constexpr dfloat VISC = (TAU - 0.5_df) / 3.0_df;
constexpr dfloat TAU_PHASE1 = TAU;
constexpr dfloat TAU_PHASE2 = TAU;

static inline fluidPhaseProps makeCasePhasePropsPhase1() {
    fluidPhaseProps pp{};
    pp.nnf.type = FLUID_NEWTONIAN;
    pp.nnf.hasLambda = false;
    pp.ve.type = VE_NEWTONIAN;
    pp.ve.eta_p = 0.0_df;
    pp.ve.lambda = 1.0_df;
    return pp;
}
static inline fluidPhaseProps makeCasePhasePropsPhase2() {
    return makeCasePhasePropsPhase1();
}
#define CASE_PHASE_PROPS_PHASE1 makeCasePhasePropsPhase1()
#define CASE_PHASE_PROPS_PHASE2 makeCasePhasePropsPhase2()

#define LOAD_CHECKPOINT false
#define CHECKPOINT_SAVE false
#define LOAD_FIELD false
#define LOAD_FIELD_STEP 0
#define SIMULATION_FOLDER_LOAD_CHECKPOINT "TEST"

#define PHASE_MODEL
#ifndef PHASE_TEST_NO_SHARPENING
    #define INTERFACE_SHARPENING
#endif
#define CONVECTION_DIFFUSION_TRANSPORT
#define D3G19
#define PHI_DIST

constexpr dfloat PHI_REFERENCE = 0.0_df;
constexpr dfloat PHI_DELTA_PHI = 2.0_df;
constexpr dfloat PHI_ONE = PHI_REFERENCE - 0.5_df * PHI_DELTA_PHI;
constexpr dfloat PHI_TWO = PHI_REFERENCE + 0.5_df * PHI_DELTA_PHI;
constexpr dfloat PHI_ZERO = 0.0_df;
constexpr dfloat PHI_SCALE = 1.0_df;
#ifndef PHASE_TEST_MISSING_INTERFACE_WIDTH
constexpr dfloat PHI_INTERFACE_WIDTH = PHASE_TEST_WIDTH;
#else
constexpr dfloat phi_del = PHASE_TEST_WIDTH;
#endif
#ifndef PHASE_TEST_MISSING_INTERFACE_WIDTH
constexpr dfloat phi_del = PHI_INTERFACE_WIDTH;
#endif
#ifndef PHASE_TEST_MISSING_NORMAL_EPS
constexpr dfloat PHI_NORMAL_EPS = 1.0e-12_df;
#endif
constexpr dfloat CONTACT_ANGLE_DEG = 90.0_df;
static __device__ dfloat contact_angle =
    CONTACT_ANGLE_DEG * ((dfloat)M_PI / 180.0_df);

constexpr dfloat phi_sigma = 1.0e-3_df;
/* Coefficients for phi in [-1,1], tanh(2*x/W), and integral tension sigma. */
constexpr dfloat kappa_CH = 0.375_df * phi_sigma * phi_del;
constexpr dfloat A_CH = 3.0_df * phi_sigma / phi_del;
constexpr dfloat M_CH = 0.05_df;
constexpr dfloat PHI_LBM_DIFF = 0.05_df;
#ifndef PHASE_TEST_MISSING_M_AC
constexpr dfloat M_AC = PHASE_TEST_MOBILITY;
#endif
constexpr dfloat PHI_DIFF_REF = PHI_LBM_DIFF;
constexpr dfloat PHI_DIFFUSIVITY = PHI_LBM_DIFF;
constexpr dfloat PHI_DIFFUSIVITY_RATIO = VISC / PHI_DIFFUSIVITY;
constexpr dfloat PHI_TAU = 0.5_df + 3.0_df * PHI_LBM_DIFF;
constexpr dfloat PHI_OMEGA = 1.0_df / PHI_TAU;

constexpr dfloat PHI_RHO_PHASE1 = RHO_0;
constexpr dfloat PHI_RHO_PHASE2 = RHO_0;
constexpr dfloat PHI_DRHO_PHASE12 = 0.0_df;
constexpr dfloat RHO_MID_REF = RHO_0;
constexpr dfloat g_value = 0.0_df;
__device__ const dfloat gravity_vector[3] = {0.0_df, 0.0_df, 0.0_df};

#if PHASE_TEST_SCENARIO == PHASE_TEST_CAPILLARY_DROPLET
    #define PHI_FORCE_FS_ENABLED 1
#else
    #define PHI_FORCE_FS_ENABLED 0
#endif
#define PHI_FORCE_FP_ENABLED 0
#define PHI_FORCE_FN_ENABLED 0
#define PHI_FORCE_FB_ENABLED 1

#ifndef PHI_CLAMP_ENABLED
    #define PHI_CLAMP_ENABLED 0
#endif

constexpr dfloat PHASE_TEST_RADIUS = 0.25_df * (dfloat)PHASE_TEST_N;
constexpr dfloat PHASE_TEST_SLAB_HALF_WIDTH = 0.25_df * (dfloat)PHASE_TEST_N;

__host__ __device__ static inline dfloat phaseTestPeriodicDistance(
    dfloat coordinate, dfloat center, dfloat period)
{
    dfloat distance = fabs(coordinate - center);
    return fmin(distance, period - distance);
}

__host__ __device__ static inline dfloat phaseTestInitialPhi(
    dfloat x, dfloat y, dfloat z)
{
    const dfloat cx = 0.5_df * (dfloat)(NX - 1);
    const dfloat cy = 0.5_df * (dfloat)(NY - 1);
    const dfloat cz = 0.5_df * (dfloat)(NZ - 1);
    dfloat signedDistance;
#if PHASE_TEST_SCENARIO == PHASE_TEST_PLANAR || \
    PHASE_TEST_SCENARIO == PHASE_TEST_TRANSLATION
    signedDistance = PHASE_TEST_SLAB_HALF_WIDTH
        - phaseTestPeriodicDistance(x, cx, (dfloat)NX);
#else
    const dfloat dx = phaseTestPeriodicDistance(x, cx, (dfloat)NX);
    const dfloat dy = phaseTestPeriodicDistance(y, cy, (dfloat)NY);
    const dfloat dz = phaseTestPeriodicDistance(z, cz, (dfloat)NZ);
    signedDistance = PHASE_TEST_RADIUS - sqrt(dx*dx + dy*dy + dz*dz);
#endif
    const dfloat mid = 0.5_df * (PHI_ONE + PHI_TWO);
    const dfloat half = 0.5_df * (PHI_TWO - PHI_ONE);
    return mid + half * tanh(2.0_df * signedDistance / phi_del);
}
