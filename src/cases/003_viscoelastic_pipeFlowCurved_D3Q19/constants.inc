/* ============================================================================
 * CONSTANTS CONFIGURATION
 * Case: 003_viscoelastic_pipeFlowCurved_D3Q19
 * Description: Oldroyd-B flow through a pipe with a curved wall
 * ========================================================================== */

/* --------------------- CURVED BOUNDARY CONDITION --------------------- */
#define CURVED_BOUNDARY_CONDITION
// #define VISCOELASTIC_STARTUP_RAMP
// #define ANALYTICAL_STEADY_INITIALIZATION
// Diagnostic one-way coupling: evolve A but omit div(tau_p) from momentum.
// #define DISABLE_POLYMER_MOMENTUM_FEEDBACK
// Experimental moment-space closure. It delays loss of positive definiteness,
// but the current formulation reaches NaN earlier than anti-bounce-back.
// #define CURVED_CONF_COUPLED_MOMENT_BC
// Experimental no-diffusive-flux projection. It is intentionally independent
// of the coupled moment closure so the two effects can be tested separately.
// #define CURVED_CONF_ZERO_NORMAL_NON_EQ_FLUX

/* --------------------- SIMULATION SCALE --------------------- */
constexpr int SCALE = 1;                    // Grid scaling factor

/* --------------------- GRID DIMENSIONS --------------------- */
constexpr int N = 128 * SCALE;              // Reference grid size
constexpr int NX = N;                       // Grid size in X
constexpr int NY = N;                       // Grid size in Y
constexpr int NZ = 8;                       // Grid size in Z
constexpr int NZ_TOTAL = NZ;                // Total grid size in Z (multi-GPU)

// Six D3G19 conformation distributions make the fused collision kernel
// register-bound. Keep this case at 256 threads/block on current GPUs.

/* --------------------- EXTERNAL DUCT CONFIGURATION --------------------- */
#define FORCE_VOXEL_BC_BUILDING
#define EXTERNAL_DUCT_BC
#define DUCT_CENTER_X (((NX-1)/2.0)+0.5)
#define DUCT_CENTER_Y (((NY-1)/2.0)+0.5)
#define OUTER_RADIUS (NY/2.0-3.5)
#define EXTERNAL_DUCT_BC_RADIUS OUTER_RADIUS
#define OUTER_VELOCITY 0.0
#define OUTER_ROTATION 0.0

/* --------------------- OLDROYD-B FLOW PARAMETERS --------------------- */
constexpr dfloat RE = 1.0_df;
constexpr dfloat BETA = 0.1_df;             // Solvent/total viscosity ratio
constexpr dfloat TAU = 0.8_df;
constexpr dfloat VISC = (TAU - 0.5_df) / 3.0_df;
constexpr dfloat SUM_VISC = VISC / BETA;
constexpr dfloat L = 2.0_df * OUTER_RADIUS; // Pipe diameter
constexpr dfloat U_C = RE * SUM_VISC / L;   // Analytical centerline velocity
constexpr dfloat U_MAX = U_C;
constexpr dfloat t_c = L / U_C;
constexpr dfloat MACH_NUMBER = U_MAX / 0.57735026918962_df;

/* --------------------- TIME STEPPING --------------------- */
constexpr int CONFORMATION_SWITCH = 0;
constexpr int N_STEPS = 24000;
constexpr int INI_STEP = 0;

/* --------------------- BODY FORCES --------------------- */
constexpr dfloat FX = 0.0;                  // Force in X direction
constexpr dfloat FY = 0.0;                  // Force in Y direction
#ifdef DISABLE_POLYMER_MOMENTUM_FEEDBACK
// With only solvent viscosity active in momentum, this preserves U_C at the
// uncoupled Newtonian steady state and avoids a tenfold velocity overshoot.
constexpr dfloat FZ = 4.0_df * VISC * U_C / (OUTER_RADIUS * OUTER_RADIUS);
#else
constexpr dfloat FZ = 4.0_df * SUM_VISC * U_C / (OUTER_RADIUS * OUTER_RADIUS);
#endif

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
#define BC_Z_PERIODIC

/* --------------------- CHECKPOINT AND LOAD CONFIGURATION --------------------- */
#define LOAD_CHECKPOINT false               // Load simulation from checkpoint
#define SIMULATION_FOLDER_LOAD_CHECKPOINT "TEST" // Folder to load checkpoint from
#define CHECKPOINT_SAVE false               // Save simulation checkpoints
#define LOAD_FIELD false                    // Load field data
#define LOAD_FIELD_STEP 0                   // Step to load field from

/* --------------------- CONFORMATION TRANSPORT --------------------- */
#ifdef FENE_P
    // FENE_P enables the conformation infrastructure; this case selects the
    // Oldroyd-B relaxation model explicitly through CASE_PHASE_PROPS.
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
    constexpr dfloat nu_p = SUM_VISC - VISC;
    constexpr dfloat Weissenberg_number = 1.0_df;
    constexpr dfloat LAMBDA = Weissenberg_number * t_c;
    constexpr dfloat inv_lambda = 1.0_df / LAMBDA;
    constexpr int STARTUP_RAMP_STEPS = (int)(5.0_df * LAMBDA);
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

    constexpr dfloat CONF_DIFFUSIVITY_RATIO = 2000.0_df;
    constexpr dfloat CONF_DIFFUSIVITY = VISC / CONF_DIFFUSIVITY_RATIO;
    constexpr dfloat CONF_DIFF_REF = 1.0_df / 6.0_df;
    constexpr dfloat CONF_TAU = CONF_DIFFUSIVITY * 3.0_df + 0.5_df;
    constexpr dfloat CONF_OMEGA = 1.0_df / CONF_TAU;
    constexpr dfloat CONF_TT_OMEGA = 1.0_df - 0.5_df * CONF_OMEGA;
    constexpr dfloat CONF_DIFF_FLUC = CONF_DIFFUSIVITY - CONF_DIFF_REF;
    constexpr dfloat CONF_AAA = 3.0_df * CONF_DIFF_FLUC * CONF_OMEGA;
    constexpr dfloat CONF_DIFF_FLUC_COEF = CONF_AAA / (1.0_df + CONF_AAA);
#endif // FENE_P
