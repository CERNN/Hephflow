/* ============================================================================
 * CONSTANTS CONFIGURATION
 * Case: 001_eccentricAnnularFlow_Turbulent
 * Description: Pressure-driven turbulent flow in an eccentric annulus
 * ========================================================================== */

#define CURVED_BOUNDARY_CONDITION

constexpr int N_STEPS = 1360000;
constexpr int INI_STEP = 0;

constexpr int HALF_WIDTH = 96;
constexpr int N = HALF_WIDTH;
constexpr int NX = 2 * HALF_WIDTH;
constexpr int NY = 2 * HALF_WIDTH;
constexpr int NZ = 16 * HALF_WIDTH;
constexpr int NZ_TOTAL = NZ;

/* Static eccentric-annulus geometry, extruded in the z direction. */
constexpr dfloat OUTER_CENTER_X = ((NX - 1) / 2.0) + 0.5;
constexpr dfloat OUTER_CENTER_Y = ((NY - 1) / 2.0) + 0.5;
constexpr dfloat OUTER_RADIUS = NY / 2.0 - 3.5;
constexpr dfloat INNER_RADIUS = 32.0;
constexpr dfloat ECCENTRIC_OFFSET_X = 30.0;
constexpr dfloat ECCENTRIC_OFFSET_Y = 0.0;
constexpr dfloat INNER_CENTER_X = OUTER_CENTER_X + ECCENTRIC_OFFSET_X;
constexpr dfloat INNER_CENTER_Y = OUTER_CENTER_Y + ECCENTRIC_OFFSET_Y;
constexpr dfloat NOMINAL_GAP = OUTER_RADIUS - INNER_RADIUS;
constexpr dfloat HYDRAULIC_RADIUS = 0.5 * NOMINAL_GAP;
constexpr dfloat ECCENTRICITY_SQUARED =
    ECCENTRIC_OFFSET_X * ECCENTRIC_OFFSET_X +
    ECCENTRIC_OFFSET_Y * ECCENTRIC_OFFSET_Y;
constexpr dfloat MINIMUM_INTERPOLATION_CLEARANCE =
    OUTER_RADIUS - INNER_RADIUS - 3.0 * 1.4142135623730951;

static_assert(INNER_RADIUS > 3.0 * 1.4142135623730951,
              "The inner cylinder must accommodate all interpolation points");
static_assert(ECCENTRICITY_SQUARED <
                  MINIMUM_INTERPOLATION_CLEARANCE *
                  MINIMUM_INTERPOLATION_CLEARANCE,
              "The minimum annular gap must accommodate all interpolation points");

/*
 * Called only for a fluid node adjacent to solid. Whichever physical surface
 * is closest determines whether this node belongs to the convex inner wall or
 * the concave outer wall.
 */
#define CURVED_BOUNDARY_IS_CONVEX(x, y, z)                                 \
    ((sqrt(((dfloat)(x) - INNER_CENTER_X) *                                \
           ((dfloat)(x) - INNER_CENTER_X) +                                \
           ((dfloat)(y) - INNER_CENTER_Y) *                                \
           ((dfloat)(y) - INNER_CENTER_Y)) - INNER_RADIUS) <                \
     (OUTER_RADIUS - sqrt(((dfloat)(x) - OUTER_CENTER_X) *                  \
                          ((dfloat)(x) - OUTER_CENTER_X) +                  \
                          ((dfloat)(y) - OUTER_CENTER_Y) *                  \
                          ((dfloat)(y) - OUTER_CENTER_Y))))

/* Independent rigid-wall kinematics. Rotation is about each cylinder's z axis. */
constexpr dfloat OUTER_TRANSLATION_X = 0.0;
constexpr dfloat OUTER_TRANSLATION_Y = 0.0;
constexpr dfloat OUTER_TRANSLATION_Z = 0.0;
constexpr dfloat OUTER_ANGULAR_VELOCITY = 0.0;
constexpr dfloat INNER_TRANSLATION_X = 0.0;
constexpr dfloat INNER_TRANSLATION_Y = 0.0;
constexpr dfloat INNER_TRANSLATION_Z = 0.0;
constexpr dfloat INNER_ANGULAR_VELOCITY = 0.0;

constexpr dfloat RE = 180.0;
constexpr dfloat U_TAU = 0.00579;
constexpr dfloat U_BULK = 15.6 * U_TAU;
constexpr dfloat U_MAX = 18.2 * U_TAU;
constexpr dfloat L = HYDRAULIC_RADIUS;

constexpr dfloat FX = 0.0;
constexpr dfloat FY = 0.0;
// Nominal friction forcing based on the annular hydraulic radius.
constexpr dfloat FZ = U_TAU * U_TAU / HYDRAULIC_RADIUS;

constexpr dfloat RHO_0 = 1.0;
constexpr dfloat U_0_X = 0.0;
constexpr dfloat U_0_Y = 0.0;
constexpr dfloat U_0_Z = 0.0;

__device__ const dfloat UX_BC[4] = {0, 0, 0, 0};
__device__ const dfloat UY_BC[4] = {0, 0, 0, 0};
__device__ const dfloat UZ_BC[4] = {0, 0, 0, 0};
__device__ const dfloat RHO_BC[4] = {RHO_0, RHO_0, RHO_0, RHO_0};

#define BC_X_WALL
#define BC_Y_WALL
#define BC_Z_PERIODIC

constexpr dfloat VISC = HYDRAULIC_RADIUS * U_TAU / RE;
constexpr dfloat TAU = 0.5 + 3.0 * VISC;
constexpr dfloat MACH_NUMBER = U_MAX / 0.57735026918962;

static inline fluidPhaseProps makeCasePhaseProps(){
    fluidPhaseProps pp{};
    pp.nnf.type = FLUID_NEWTONIAN;
    pp.nnf.hasLambda = false;
    pp.ve.type = VE_NEWTONIAN;
    return pp;
}
#define CASE_PHASE_PROPS makeCasePhaseProps()

#define LOAD_CHECKPOINT false
#define SIMULATION_FOLDER_LOAD_CHECKPOINT "TEST"
#define CHECKPOINT_SAVE false
#define LOAD_FIELD false
#define LOAD_FIELD_STEP 0

#define FORCE_VOXEL_BC_BUILDING
#define EXTERNAL_DUCT_BC
#define EXTERNAL_DUCT_BC_RADIUS OUTER_RADIUS
