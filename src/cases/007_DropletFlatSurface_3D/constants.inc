/* ============================================================================
 * CONSTANTS CONFIGURATION
 * Case: 3D droplet wetting-angle verification
 * ========================================================================== */

/* --------------------- SIMULATION SCALE --------------------- */
constexpr int SCALE = 1;

/* --------------------- TIME STEPPING --------------------- */
constexpr int N_STEPS = 25000;
constexpr int INI_STEP = 0;

/* --------------------- GRID DIMENSIONS --------------------- */
constexpr int L = 128 * SCALE;
constexpr int NX = 2 * L;
constexpr int NY = L;
constexpr int NZ = 2 * L;
constexpr int NZ_TOTAL = NZ;

constexpr dfloat R0 = 20.0_df;
constexpr dfloat D0 = 2.0_df * R0;
constexpr dfloat INITIAL_DROPLET_GAP = 1.5_df;

/* --------------------- PRIMARY DIMENSIONLESS INPUTS --------------------- */
constexpr dfloat BOND_NUMBER = 0.0_df; // Bo_R = Delta(rho) g R0^2 / sigma 
constexpr dfloat CH_PECLET_NUMBER = 6750.0_df; // Pe_CH = U_sigma R0 / D_CH, D_CH = M_CH A_CH, U_sigma = sqrt(sigma/(rho_2 R0))

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
__device__ const dfloat UX_BC[4] = {0.0_df, 0.0_df, 0.0_df, 0.0_df};
__device__ const dfloat UY_BC[4] = {0.0_df, 0.0_df, 0.0_df, 0.0_df};
__device__ const dfloat UZ_BC[4] = {0.0_df, 0.0_df, 0.0_df, 0.0_df};
__device__ const dfloat RHO_BC[4] = {RHO_0, RHO_0, RHO_0, RHO_0};

/* --------------------- BOUNDARY CONDITION TYPES --------------------- */
#define BC_X_PERIODIC
#define BC_Y_WALL
#define BC_Z_PERIODIC

/* --------------------- FLUID PROPERTIES --------------------- */

constexpr dfloat DENSITY_RATIO = 3.0_df;

constexpr dfloat PHI_RHO_PHASE2 = 1.5_df;
constexpr dfloat PHI_RHO_PHASE1 = PHI_RHO_PHASE2 / DENSITY_RATIO;
constexpr dfloat PHI_DRHO_PHASE12 = PHI_RHO_PHASE2 - PHI_RHO_PHASE1;
constexpr dfloat DELTA_RHO_ABS = (PHI_DRHO_PHASE12 >= 0.0_df) ? PHI_DRHO_PHASE12 : -PHI_DRHO_PHASE12;
constexpr dfloat RHO_MID_REF = 0.5_df * (PHI_RHO_PHASE1 + PHI_RHO_PHASE2);

constexpr dfloat KINEMATIC_VISCOSITY_RATIO = 1.0_df;
constexpr dfloat NU_PHASE2 = 0.00625_df;
constexpr dfloat NU_PHASE1 = NU_PHASE2 / KINEMATIC_VISCOSITY_RATIO;

constexpr dfloat MU_PHASE1 = PHI_RHO_PHASE1 * NU_PHASE1;
constexpr dfloat MU_PHASE2 = PHI_RHO_PHASE2 * NU_PHASE2;

constexpr dfloat DYNAMIC_VISCOSITY_RATIO = MU_PHASE2 / MU_PHASE1;

constexpr dfloat TAU_PHASE1 = 0.5_df + 3.0_df * NU_PHASE1;
constexpr dfloat TAU_PHASE2 = 0.5_df + 3.0_df * NU_PHASE2;

constexpr dfloat TAU = TAU_PHASE2;
constexpr dfloat VISC = (TAU - 0.5_df) / 3.0_df;


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


/* --------------------- FREE ENERGY / CAHN-HILLIARD COEFFICIENTS --------------------- */
constexpr dfloat INTERFACE_WIDTH = 4.0_df;
constexpr dfloat INTERFACE_RATIO = INTERFACE_WIDTH / R0;
constexpr dfloat phi_del = INTERFACE_WIDTH;

constexpr dfloat CONTACT_ANGLE_DEG = 90.0_df;
static __device__ dfloat contact_angle = CONTACT_ANGLE_DEG * ((dfloat)M_PI / 180.0_df);

constexpr dfloat phi_sigma = 0.002_df;
constexpr dfloat kappa_CH = 0.75_df * phi_sigma * phi_del; 						// kappa_CH = 3 sigma W / 4 
constexpr dfloat A_CH = (6.0_df * phi_sigma) / phi_del;                         // A_CH = 6 sigma / W 

constexpr dfloat U_CAPILLARY = sqrtt(phi_sigma / (PHI_RHO_PHASE2 * R0));         //U_sigma = sqrt(sigma/(rho_2 R0)) 
constexpr dfloat T_CAPILLARY = R0 / U_CAPILLARY;                                // t_sigma = R0/U_sigma 
constexpr dfloat CH_DIFFUSIVITY_SCALE = (U_CAPILLARY * R0) / CH_PECLET_NUMBER;  // D_CH = M_CH A_CH = U_sigma R0 / Pe_CH 

constexpr dfloat U_MAX = U_CAPILLARY;
constexpr dfloat M_CH = CH_DIFFUSIVITY_SCALE / A_CH;

/* --------------------- CAPILLARY REFERENCE SCALE --------------------- */

constexpr dfloat MACH_NUMBER = U_CAPILLARY / 0.57735026918962_df;

/* --------------------- DENSITY COUPLING --------------------- */
// g = Bo_R sigma / (Delta(rho) R0^2);
constexpr dfloat g_value = (BOND_NUMBER == 0.0_df) ? 0.0_df : (BOND_NUMBER * phi_sigma) / (DELTA_RHO_ABS * R0 * R0); 
__device__ const dfloat gravity_vector[3] = {0.0_df, -1.0_df, 0.0_df};


/* --------------------- FORCE COUPLING CONTROL FLAGS --------------------- */
#define PHI_FORCE_FS_ENABLED 1
#define PHI_FORCE_FP_ENABLED 1
#define PHI_FORCE_FN_ENABLED 1
#define PHI_FORCE_FB_ENABLED 1

/* --------------------- PHASE-FIELD TRANSPORT (LBM) --------------------- */
constexpr dfloat PHI_LBM_DIFF = 1e-6_df;
constexpr dfloat PHI_TAU = 0.5_df + 3.0_df * PHI_LBM_DIFF;
constexpr dfloat PHI_OMEGA = 1.0_df / PHI_TAU;

