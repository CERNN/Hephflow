/* Oldroyd-B IBM pipe verification. All quantities are in lattice units. */
constexpr int NX = 96;
constexpr int NY = 96;
constexpr int NZ = 16;
constexpr int NZ_TOTAL = NZ;

constexpr dfloat DUCT_CENTER_X = 0.5_df * (NX - 1);
constexpr dfloat DUCT_CENTER_Y = 0.5_df * (NY - 1);
constexpr dfloat DUCT_RADIUS = 36.0_df;
constexpr dfloat DUCT_RADIUS_A = DUCT_RADIUS;
constexpr dfloat DUCT_RADIUS_B = DUCT_RADIUS;
#ifndef DUCT_STAGGERED_RINGS
#define DUCT_STAGGERED_RINGS true
#endif
constexpr bool DUCT_USE_STAGGERED_RINGS = DUCT_STAGGERED_RINGS;

constexpr dfloat RHO_0 = 1.0_df;
constexpr dfloat BETA = 0.8_df;
constexpr dfloat TAU = 0.8_df;
constexpr dfloat VISC = (TAU - 0.5_df) / 3.0_df;
constexpr dfloat SUM_VISC = VISC / BETA;
constexpr dfloat nu_p = SUM_VISC - VISC;
constexpr dfloat DUCT_DRIVING_FORCE = 1.0e-6_df;
constexpr dfloat U_C = DUCT_DRIVING_FORCE * DUCT_RADIUS * DUCT_RADIUS /
    (4.0_df * SUM_VISC);
constexpr dfloat U_MAX = U_C;
constexpr dfloat L = 2.0_df * DUCT_RADIUS;
constexpr dfloat RE = U_C * L / SUM_VISC;
constexpr dfloat MACH_NUMBER = U_MAX / 0.57735026918962_df;

constexpr dfloat WALL_WEISSENBERG = 0.1_df;
constexpr dfloat LAMBDA = 2.0_df * SUM_VISC * WALL_WEISSENBERG /
    (DUCT_DRIVING_FORCE * DUCT_RADIUS);
constexpr dfloat inv_lambda = 1.0_df / LAMBDA;
constexpr dfloat Weissenberg_number = LAMBDA * U_C / DUCT_RADIUS;
constexpr dfloat t_c = L / U_C;

constexpr dfloat FX = 0.0_df;
constexpr dfloat FY = 0.0_df;
constexpr dfloat FZ = DUCT_DRIVING_FORCE;
constexpr dfloat U_0_X = 0.0_df;
constexpr dfloat U_0_Y = 0.0_df;
constexpr dfloat U_0_Z = 0.0_df;

__device__ const dfloat UX_BC[4] = {0, 0, 0, 0};
__device__ const dfloat UY_BC[4] = {0, 0, 0, 0};
__device__ const dfloat UZ_BC[4] = {0, 0, 0, 0};
__device__ const dfloat RHO_BC[4] = {RHO_0, RHO_0, RHO_0, RHO_0};

#define BC_X_PERIODIC
#define BC_Y_PERIODIC
#define BC_Z_PERIODIC

constexpr int CONFORMATION_SWITCH = 0;
constexpr int N_STEPS = 60000;
constexpr int INI_STEP = 0;
#define LOAD_CHECKPOINT false
#define SIMULATION_FOLDER_LOAD_CHECKPOINT "TEST"
#define CHECKPOINT_SAVE false
#define LOAD_FIELD false
#define LOAD_FIELD_STEP 0

#ifdef PARTICLE_MODEL
#define IBM_METHOD
#define LOCAL_FORCES
#define NUM_PARTICLES 1
constexpr dfloat GX = 0.0_df;
constexpr dfloat GY = 0.0_df;
constexpr dfloat GZ = 0.0_df;
constexpr dfloat FLUID_DENSITY = RHO_0;
// The particle module currently compiles DEM support constants even when this
// fixed wall has all collision flags disabled.
constexpr dfloat PP_FRICTION_COEF = 0.1_df;
constexpr dfloat PW_FRICTION_COEF = 0.1_df;
constexpr dfloat PP_REST_COEF = 0.9_df;
constexpr dfloat PW_REST_COEF = 0.9_df;
constexpr dfloat PARTICLE_YOUNG_MODULUS = 0.5_df;
constexpr dfloat PARTICLE_POISSON_RATIO = 0.3_df;
constexpr dfloat WALL_YOUNG_MODULUS = 100.0_df;
constexpr dfloat WALL_POISSON_RATIO = 0.3_df;
constexpr dfloat SPHERE_WALL_DAMPING_CONST = -2.0_df * constexprLn(PW_REST_COEF) /
    sqrtt(M_PI * M_PI + constexprLn(PW_REST_COEF) * constexprLn(PW_REST_COEF));
constexpr dfloat SPHERE_SPHERE_DAMPING_CONST = -2.0_df * constexprLn(PP_REST_COEF) /
    sqrtt(M_PI * M_PI + constexprLn(PP_REST_COEF) * constexprLn(PP_REST_COEF));
#endif

#ifdef VISCOELASTIC_MODEL
#define OLDROYD_B
#define CONVECTION_DIFFUSION_TRANSPORT
#define D3G19
#define CONFORMATION_TENSOR
#define A_XX_DIST
#define A_XY_DIST
#define A_XZ_DIST
#define A_YY_DIST
#define A_YZ_DIST
#define A_ZZ_DIST
#define COMPUTE_CONF_GRADIENT_FINITE_DIFFERENCE
#define COMPUTE_VEL_GRADIENT_FINITE_DIFFERENCE

constexpr dfloat CONF_ZERO = 1.0_df;
constexpr dfloat fenep_re = 10.0_df;

static inline fluidPhaseProps makeCasePhaseProps(){
    fluidPhaseProps pp{};
    pp.nnf.type = FLUID_BINGHAM;
    pp.nnf.hasLambda = false;
    pp.nnf.u.bingham.s_y = 0.0_df;
    pp.nnf.u.bingham.omega_p = 1.0_df / TAU;
    pp.ve.type = VE_OLDROYD_B;
    pp.ve.eta_p = nu_p;
    pp.ve.lambda = LAMBDA;
    return pp;
}
#define CASE_PHASE_PROPS makeCasePhaseProps()

// The auxiliary transport retains this explicit diffusivity. It is small but
// nonzero and must be included in the verification error audit.
#ifndef DUCT_CONF_DIFFUSIVITY_RATIO
#define DUCT_CONF_DIFFUSIVITY_RATIO 2000.0_df
#endif
constexpr dfloat CONF_DIFFUSIVITY_RATIO = DUCT_CONF_DIFFUSIVITY_RATIO;
constexpr dfloat CONF_DIFFUSIVITY = VISC / CONF_DIFFUSIVITY_RATIO;
constexpr dfloat CONF_DIFF_REF = 1.0_df / 6.0_df;
constexpr dfloat CONF_TAU = 0.5_df + 3.0_df * CONF_DIFFUSIVITY;
constexpr dfloat CONF_OMEGA = 1.0_df / CONF_TAU;
constexpr dfloat CONF_TT_OMEGA = 1.0_df - 0.5_df * CONF_OMEGA;
constexpr dfloat CONF_DIFF_FLUC = CONF_DIFFUSIVITY - CONF_DIFF_REF;
constexpr dfloat CONF_AAA = 3.0_df * CONF_DIFF_FLUC * CONF_OMEGA;
constexpr dfloat CONF_DIFF_FLUC_COEF = CONF_AAA / (1.0_df + CONF_AAA);
#endif
