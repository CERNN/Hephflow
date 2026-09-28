/* Newtonian IBM pipe baseline. All quantities are in lattice units. */
constexpr int NX = 96;
constexpr int NY = 96;
constexpr int NZ = 16;
constexpr int NZ_TOTAL = NZ;

constexpr dfloat DUCT_CENTER_X = 0.5_df * (NX - 1);
constexpr dfloat DUCT_CENTER_Y = 0.5_df * (NY - 1);
constexpr dfloat DUCT_RADIUS = 36.0_df;
constexpr dfloat DUCT_RADIUS_A = DUCT_RADIUS;
constexpr dfloat DUCT_RADIUS_B = DUCT_RADIUS;

constexpr dfloat RHO_0 = 1.0_df;
constexpr dfloat TAU = 0.8_df;
constexpr dfloat VISC = (TAU - 0.5_df) / 3.0_df;
constexpr dfloat DUCT_DRIVING_FORCE = 1.0e-6_df;
constexpr dfloat U_C = DUCT_DRIVING_FORCE * DUCT_RADIUS * DUCT_RADIUS /
    (4.0_df * VISC);
constexpr dfloat U_MAX = U_C;
constexpr dfloat L = 2.0_df * DUCT_RADIUS;
constexpr dfloat RE = U_C * L / VISC;
constexpr dfloat MACH_NUMBER = U_MAX / 0.57735026918962_df;

constexpr dfloat FX = 0.0_df;
constexpr dfloat FY = 0.0_df;
// LOCAL_FORCES needs the base value in its moment field. force_field.inc
// removes it in the fictitious exterior, so only r<R is driven.
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
