/* ============================================================================
 * CONSTANTS CONFIGURATION
 * Case: Oscillating Droplet
 * Description: Two-phase flow simulation of oscillating droplet
 * ========================================================================== */

/* --------------------- SIMULATION SCALE --------------------- */
constexpr int SCALE = 1;                    // Grid scaling factor

/* --------------------- DIMENSIONLESS NUMBERS --------------------- */
constexpr dfloat RE = 1;                    // Reynolds number

/* --------------------- TIME STEPPING --------------------- */
constexpr int N_STEPS = 2500000;               // Total simulation steps (003 reference)
constexpr int CONFORMATION_SWITCH = 0;      // Step to switch on conformation
constexpr int INI_STEP = 0;                 // Initial step (for restarts)

/* --------------------- GRID DIMENSIONS --------------------- */
constexpr int N = 256 * SCALE;              // Reference grid size
constexpr int NX = 8;                       // Grid size in X (thin channel)
constexpr int NY = N;                       // Grid size in Y (channel height)
constexpr int NZ = 8;                       // Grid size in Z (thin channel)
constexpr int NZ_TOTAL = NZ;                // Total grid size in Z (multi-GPU)

/* --------------------- REFERENCE VALUES --------------------- */
constexpr dfloat U_MAX = 0.0;               // Maximum/reference velocity
constexpr dfloat L = N;                     // Reference length

/* --------------------- VISCOELASTIC PARAMETERS --------------------- */
constexpr dfloat U_C = 0.01;                // Characteristic velocity
constexpr dfloat SUM_VISC = (U_C*L)/RE;     // Total viscosity
constexpr dfloat BETA = 0.1;                // Solvent fraction in viscoelastic phase

/* --------------------- BODY FORCES --------------------- */
constexpr dfloat FX = 8.0*SUM_VISC*U_C/NY/NY; // Force in X direction (flow driving)
constexpr dfloat FY = 0.0;                  // Force in Y direction
constexpr dfloat FZ = 0.0;                 // Force in Z direction

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
#define BC_Y_WALL
#define BC_Z_PERIODIC

/* --------------------- FLUID PROPERTIES --------------------- */
constexpr dfloat VISC = BETA * SUM_VISC;                     // Solvent viscosity (003 reference)
constexpr dfloat TAU = 0.5 + 3.0*VISC;                       // Relaxation time from 003 reference
constexpr dfloat TAU_PHASE2 = TAU;                           // Both phases identical in VV debug case

constexpr dfloat MU_1 = VISC;
constexpr dfloat MU_2 = VISC;

constexpr dfloat ETA_S_V = VISC;
constexpr dfloat ETA_P_V = SUM_VISC - ETA_S_V;
constexpr dfloat nu_p = ETA_P_V;                              // Compatibility alias used in shared code
constexpr dfloat TAU_SOLVENT_PHASE2 = TAU;

constexpr dfloat t_c = L/U_C;               // Characteristic time
constexpr dfloat MACH_NUMBER = U_MAX / 0.57735026918962; // Mach number
constexpr dfloat Weissenberg_number = 1.0;
constexpr dfloat LAMBDA = Weissenberg_number*t_c;
constexpr dfloat inv_lambda = 1.0/LAMBDA;
constexpr dfloat fenep_re = 10.0;

// Phase 1: Oldroyd-B (VV debug case)
static inline fluidPhaseProps makeCasePhasePropsPhase1(){
        fluidPhaseProps pp{};
        pp.nnf.type = FLUID_BINGHAM;
        pp.nnf.hasLambda = false;
        pp.nnf.u.bingham.s_y = 0.0_df;
        pp.nnf.u.bingham.omega_p = 1.0_df / TAU;
        pp.ve.type = VE_OLDROYD_B;
        pp.ve.eta_p = ETA_P_V;
        pp.ve.lambda = LAMBDA;
        return pp;
}

// Phase 2: Oldroyd-B (identical to phase 1 in VV debug case)
static inline fluidPhaseProps makeCasePhasePropsPhase2(){
        fluidPhaseProps pp{};
        pp.nnf.type = FLUID_BINGHAM;
        pp.nnf.hasLambda = false;
        pp.nnf.u.bingham.s_y = 0.0_df;
        pp.nnf.u.bingham.omega_p = 1.0_df / TAU_PHASE2;
        pp.ve.type = VE_OLDROYD_B;
        pp.ve.eta_p = ETA_P_V;
        pp.ve.lambda = LAMBDA;
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
        constexpr dfloat CONF_DIFFUSIVITY = VISC/CONF_DIFFUSIVITY_RATIO;

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

#define PHASE_MODEL                         // Enable phase field model
#define INTERFACE_SHARPENING               // Enable interface sharpening
#define CONVECTION_DIFFUSION_TRANSPORT     // Transport equation method
#define D3G19                             // 19-velocity phase field (D3Q19 weights, as2=3)
//#define D3G7                              // 7-velocity phase field (D3Q7 weights, as2=4)

/* --------------------- PHASE FIELD DISTRIBUTION --------------------- */
#define PHI_DIST                           // Enable phase field distribution
#define METHOD_CAHN_HILLIARD
/* --------------------- PHASE FIELD REFERENCE VALUES --------------------- */
constexpr dfloat PHI_REFERENCE = (dfloat)(0.0);   // Reference phase value
constexpr dfloat PHI_DELTA_PHI = 2.0_df;                // Phase field range
constexpr dfloat PHI_ONE = PHI_REFERENCE - PHI_DELTA_PHI/2.0;   // Phase 1 value
constexpr dfloat PHI_TWO = PHI_REFERENCE + PHI_DELTA_PHI/2.0;   // Phase 2 value
constexpr dfloat PHI_ZERO = (dfloat)(0.0);        // No offset: stored phi = physical phi in [-1,+1]
// Physical CH order parameter mapping
constexpr dfloat PHI_SCALE = 2.0/PHI_DELTA_PHI;

/* --------------------- SURFACE TENSION AND INTERFACE PARAMETERS --------------------- */

// Physical surface tension
constexpr dfloat phi_sigma = 0.0003;
constexpr dfloat phi_del = 4.0;
constexpr dfloat M_CH      = 0.005_df;  // Mobility

// --- 2. Cahn-Hilliard Constants ---
constexpr dfloat kappa_CH  = 0.75_df * phi_sigma * phi_del; 
constexpr dfloat A_CH      = (6.0_df * phi_sigma) / phi_del;

// constexpr dfloat phi_xi = phi_del / 2.94;
// Unused in current METHOD_CAHN_HILLIARD path (Allen-Cahn only):
// constexpr dfloat phi_beta  = 12.0 * phi_sigma / phi_del;
// constexpr dfloat phi_kappa = 1.5 * phi_sigma * phi_del;
// constexpr dfloat phi_gamma = 1.0 / (2.0 * 1.41421356237309 * phi_xi);

/* --------------------- DENSITY COUPLING --------------------- */

constexpr dfloat phi_rho_l = 1.0;
constexpr dfloat phi_rho_g = 1.0;
constexpr dfloat phi_drho = phi_rho_l - phi_rho_g;

constexpr dfloat g_value = -1.5e-5_df; 
__device__ const dfloat gravity_vector[3] = {0.0, 0.0, 0.0}; // Gravity direction


constexpr dfloat tau_H = TAU;
constexpr dfloat tau_L = TAU;

/* --------------------- FORCE COUPLING CONTROL FLAGS --------------------- */
#define PHI_FORCE_FS_ENABLED 1
#define PHI_FORCE_FP_ENABLED 1
#define PHI_FORCE_FN_ENABLED 1
#define PHI_FORCE_FB_ENABLED 1

/* --------------------- PHASE-FIELD TRANSPORT (LBM) --------------------- */

// Reference diffusion (controls interface mobility)
constexpr dfloat PHI_DIFFUSIVITY_RATIO = 1; // Diffusivity ratio (like Prandtl number) 
constexpr dfloat PHI_DIFFUSIVITY = VISC/PHI_DIFFUSIVITY_RATIO; // Thermal diffusivity
constexpr dfloat PHI_DIFF_REF = 0.0000033333;

// LBM relaxation
constexpr dfloat PHI_TAU   = 0.5 + 3.0 * PHI_DIFF_REF;
constexpr dfloat PHI_OMEGA = 1.0 / PHI_TAU;

/* ----- END CONSTANTS CONFIGURATION ----- */
