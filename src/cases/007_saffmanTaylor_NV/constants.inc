/* ============================================================================
 * CONSTANTS CONFIGURATION
 * Case: Vertical viscoelastic displacement between parallel plates
 * Description: Viscoelastic phase injected at z = 0 into a Newtonian resident
 *              phase. Walls at y = 0 and y = NY-1; x is periodic.
 * ========================================================================== */

/* --------------------- SIMULATION SCALE --------------------- */
constexpr int SCALE = 1;                    // Grid scaling factor

/* --------------------- TIME STEPPING --------------------- */
constexpr int N_STEPS = 40000;            // Total simulation steps
constexpr int CONFORMATION_SWITCH = 0;       // Step to switch on conformation
constexpr int INI_STEP = 0;                  // Initial step (for restarts)

/* --------------------- GRID DIMENSIONS --------------------- */
constexpr int N = 32 * SCALE;               // Reference channel resolution
constexpr int NX = N;                        // Periodic spanwise direction, x
constexpr int NY = N;                        // Plate-normal direction, y
constexpr int NZ = 16 * N;                   // Streamwise direction, z
constexpr int NZ_TOTAL = NZ;                 // Total grid size in Z (multi-GPU)

/* --------------------- REFERENCE VALUES --------------------- */
constexpr dfloat H = (dfloat)(NY - 1);       // Plate spacing in lattice units
constexpr dfloat L = H;                      // Compatibility alias: reference length = H
constexpr dfloat U_IN = 0.005_df;             // Mean inlet velocity
constexpr dfloat U_MAX = 1.5_df * U_IN;      // Maximum of planar Poiseuille inlet profile
constexpr dfloat U_C = U_IN;                 // Characteristic velocity
constexpr dfloat t_c = H / U_IN;             // Characteristic advective time H/U_in
constexpr dfloat MACH_NUMBER = U_MAX / 0.57735026918962_df;

// Initial viscoelastic buffer: interface centered one channel height from inlet.
// Phase 2 is viscoelastic and occupies z < INITIAL_INTERFACE_Z initially.
constexpr int MIN_INITIAL_INJECTED_NODES = 10;
constexpr dfloat INITIAL_INTERFACE_Z = H;

// Maintain a pure injected-fluid reservoir next to the inlet. This prevents
// wall/inlet corner transients and CH kinetic modes from contaminating the
// prescribed incoming phase before it enters the physical displacement zone.
#define PHI_INLET_RESERVOIR_Z_NODES MIN_INITIAL_INJECTED_NODES

/* --------------------- VISCOELASTIC PARAMETERS --------------------- */
constexpr dfloat BETA = 0.1;                 // Solvent fraction in viscoelastic phase

/* --------------------- BODY FORCES --------------------- */
// The flow is velocity-driven; no streamwise pressure/body-force driving is imposed.
constexpr dfloat FX = 0.0_df;
constexpr dfloat FY = 0.0_df;
constexpr dfloat FZ = 0.0_df;

/* --------------------- INITIAL CONDITIONS --------------------- */
constexpr dfloat RHO_0 = 1.0_df;
constexpr dfloat U_0_X = 0.0_df;
constexpr dfloat U_0_Y = 0.0_df;
constexpr dfloat U_0_Z = 0.0_df;             // Initially at rest; inlet BC drives the flow

/* --------------------- BOUNDARY CONDITION VALUES --------------------- */
// Wall values remain zero. The z-inlet velocity is spatially varying and will
// be imposed directly by the inlet BC using U_IN.
__device__ const dfloat UX_BC[4] = {0, 0, 0, 0};
__device__ const dfloat UY_BC[4] = {0, 0, 0, 0};
__device__ const dfloat UZ_BC[4] = {0, 0, 0, 0};
__device__ const dfloat RHO_BC[4] = {RHO_0, RHO_0, RHO_0, RHO_0};
constexpr dfloat RHO_OUTLET = RHO_0;

/* --------------------- BOUNDARY CONDITION TYPES --------------------- */
#define BC_X_PERIODIC
#define BC_Y_WALL
#define BC_Z_WALL

/* --------------------- FLUID PROPERTIES --------------------- */
constexpr dfloat TAU = 0.8;                  // Phase 1: Newtonian resident fluid
constexpr dfloat TAU_PHASE1 = TAU;
constexpr dfloat TAU_PHASE2 = 1.1;           // Phase 2: Oldroyd-B total zero-shear viscosity

constexpr dfloat MU_1 = (TAU - 0.5_df) / 3.0_df;
constexpr dfloat MU_2 = (TAU_PHASE2 - 0.5_df) / 3.0_df;
constexpr dfloat SUM_VISC = MU_2;

constexpr dfloat ETA_S_V = BETA * MU_2;
constexpr dfloat ETA_P_V = MU_2 - ETA_S_V;
constexpr dfloat nu_p = ETA_P_V;             // Compatibility alias used in shared code
constexpr dfloat TAU_SOLVENT_PHASE2 = 0.5_df + 3.0_df * ETA_S_V;

constexpr dfloat VISC = MU_1;
constexpr dfloat Weissenberg_number = 1.0;
constexpr dfloat LAMBDA = Weissenberg_number * t_c;
constexpr dfloat inv_lambda = 1.0_df / LAMBDA;
constexpr dfloat fenep_re = 10.0;

// Phase 1: Newtonian resident fluid, phi = PHI_ONE
static inline fluidPhaseProps makeCasePhasePropsPhase1(){
        fluidPhaseProps pp{};
        pp.nnf.type = FLUID_BINGHAM;
        pp.nnf.hasLambda = false;
        pp.nnf.u.bingham.s_y = 0.0_df;
        pp.nnf.u.bingham.omega_p = 1.0_df / TAU;
        pp.ve.type = VE_NEWTONIAN;
        pp.ve.eta_p = 0.0_df;
        pp.ve.lambda = 1.0_df;
        return pp;
}

// Phase 2: Oldroyd-B injected fluid, phi = PHI_TWO
// Keep the existing validated viscoelastic-property implementation unchanged.
static inline fluidPhaseProps makeCasePhasePropsPhase2(){
        fluidPhaseProps pp{};
        pp.nnf.type = FLUID_BINGHAM;
        pp.nnf.hasLambda = false;
        pp.nnf.u.bingham.s_y = 0.0_df;
#ifdef SAFFMAN_TAYLOR_NEWTONIAN_BASELINE
        // Boundary-condition baseline: make phase 2 identical to the
        // Newtonian resident phase while retaining the phase-field interface.
        pp.nnf.u.bingham.omega_p = 1.0_df / TAU;
        pp.ve.type = VE_NEWTONIAN;
        pp.ve.eta_p = 0.0_df;
        pp.ve.lambda = 1.0_df;
#else
        pp.nnf.u.bingham.omega_p = 1.0_df / TAU_SOLVENT_PHASE2;
        pp.ve.type = VE_OLDROYD_B;
        pp.ve.eta_p = ETA_P_V;
        pp.ve.lambda = LAMBDA;
#endif
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

/* --------------------- FENE-P VISCOELASTIC MODEL CONFIGURATION --------------------- */
#ifdef FENE_P
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

        constexpr dfloat CONF_ZERO = (dfloat)(10.0);

        constexpr dfloat CONF_DIFFUSIVITY_RATIO = 100;
        constexpr dfloat CONF_DIFFUSIVITY = VISC / CONF_DIFFUSIVITY_RATIO;

        constexpr dfloat CONF_DIFF_REF = (1.0/6);
        constexpr dfloat CONF_TAU = CONF_DIFF_REF*3+0.5;
        constexpr dfloat CONF_OMEGA  = 1.0/CONF_TAU;
        constexpr dfloat CONF_TT_OMEGA = 1.0-0.5*CONF_OMEGA;

        constexpr dfloat CONF_DIFF_FLUC = CONF_DIFFUSIVITY - CONF_DIFF_REF;
        constexpr dfloat CONF_AAA = (3*CONF_DIFF_FLUC*CONF_OMEGA);
        constexpr dfloat CONF_DIFF_FLUC_COEF = CONF_AAA/(1.0+CONF_AAA);
#endif

/* ============================================================================
 * PHASE FIELD MODEL PARAMETERS
 * Requires #define PHASE_MODEL in model.inc
 * Uses second distribution for phase field tracking
 * ========================================================================== */

#define PHASE_MODEL
#define INTERFACE_SHARPENING
#define CONVECTION_DIFFUSION_TRANSPORT
#define D3G19
//#define D3G7

/* --------------------- PHASE FIELD DISTRIBUTION --------------------- */
#define PHI_DIST
#define METHOD_CAHN_HILLIARD

/* --------------------- PHASE FIELD REFERENCE VALUES --------------------- */
constexpr dfloat PHI_REFERENCE = (dfloat)(0.0);
constexpr dfloat PHI_DELTA_PHI = 2.0_df;
constexpr dfloat PHI_ONE = PHI_REFERENCE - PHI_DELTA_PHI/2.0;   // Newtonian resident phase
constexpr dfloat PHI_TWO = PHI_REFERENCE + PHI_DELTA_PHI/2.0;   // Viscoelastic injected phase
constexpr dfloat PHI_ZERO = (dfloat)(0.0);
constexpr dfloat PHI_SCALE = 2.0/PHI_DELTA_PHI;

/* --------------------- SURFACE TENSION AND INTERFACE PARAMETERS --------------------- */
constexpr dfloat phi_sigma = 0.0003;
constexpr dfloat phi_del = 4.0;
constexpr dfloat M_CH = 0.005_df;

// Keep at least ten lattice nodes of injected fluid before the diffuse
// interface begins. With the current grid, the interface begins near z = 29.
static_assert(INITIAL_INTERFACE_Z - 0.5_df * phi_del >=
              (dfloat)MIN_INITIAL_INJECTED_NODES,
              "Initial viscoelastic inlet buffer must contain at least 10 nodes");

constexpr dfloat kappa_CH = 0.75_df * phi_sigma * phi_del;
constexpr dfloat A_CH = (6.0_df * phi_sigma) / phi_del;

constexpr dfloat CONTACT_ANGLE_DEG = 90.0_df;
static __device__ dfloat contact_angle = CONTACT_ANGLE_DEG * ((dfloat)M_PI / 180.0_df);


// constexpr dfloat phi_xi = phi_del / 2.94;
// Unused in current METHOD_CAHN_HILLIARD path (Allen-Cahn only):
// constexpr dfloat phi_beta  = 12.0 * phi_sigma / phi_del;
// constexpr dfloat phi_kappa = 1.5 * phi_sigma * phi_del;
// constexpr dfloat phi_gamma = 1.0 / (2.0 * 1.41421356237309 * phi_xi);

/* --------------------- DENSITY COUPLING --------------------- */
constexpr dfloat PHI_RHO_PHASE2 = 1.0;
constexpr dfloat PHI_RHO_PHASE1 = 1.0;
constexpr dfloat PHI_DRHO_PHASE12 = PHI_RHO_PHASE2 - PHI_RHO_PHASE1;
constexpr dfloat DELTA_RHO_ABS = (PHI_DRHO_PHASE12 >= 0.0_df) ? PHI_DRHO_PHASE12 : -PHI_DRHO_PHASE12;
constexpr dfloat RHO_MID_REF = 0.5_df * (PHI_RHO_PHASE1 + PHI_RHO_PHASE2);


// Gravity acts downward, i.e. in the -z direction.
// Existing convention is kept: signed magnitude times a unit direction vector.
constexpr dfloat g_value = -1.5e-5_df;
__device__ const dfloat gravity_vector[3] = {0.0, 0.0, 1.0};

/* --------------------- FORCE COUPLING CONTROL FLAGS --------------------- */
#define PHI_FORCE_FS_ENABLED 1
#define PHI_FORCE_FP_ENABLED 1
#define PHI_FORCE_FN_ENABLED 1
#define PHI_FORCE_FB_ENABLED 1

/* --------------------- PHASE-FIELD TRANSPORT (LBM) --------------------- */
constexpr dfloat PHI_DIFFUSIVITY_RATIO = 1;
constexpr dfloat PHI_DIFFUSIVITY = VISC/PHI_DIFFUSIVITY_RATIO;
// Numerical reference diffusivity for the phase-population relaxation. The
// explicit counter-source removes this contribution from the macroscopic CH
// equation, so 1/6 damps kinetic boundary modes without changing M_CH.
constexpr dfloat PHI_DIFF_REF = 1.0_df / 6.0_df;

constexpr dfloat PHI_LBM_DIFF = PHI_DIFF_REF;
constexpr dfloat PHI_TAU = 0.5_df + 3.0_df * PHI_LBM_DIFF;
constexpr dfloat PHI_OMEGA = 1.0_df / PHI_TAU;

/* ----- END CONSTANTS CONFIGURATION ----- */
