/* ============================================================================
 * CONSTANTS CONFIGURATION
 * Case: 003_viscoelasitcParallelPlates_D3Q19
 * Description: Viscoelastic flow between parallel plates (FENE-P model)
 * ========================================================================== */

/* --------------------- SIMULATION SCALE --------------------- */
constexpr int SCALE = 1;

/* --------------------- GRID DIMENSIONS --------------------- */
constexpr int N = 256;
constexpr int NX = 4;
constexpr int NY = N;
constexpr int NZ = 4;
constexpr int NZ_TOTAL = NZ;

/* --------------------- REFERENCE VALUES --------------------- */
constexpr dfloat U_MAX = 0.0;
constexpr dfloat L = N-1;

/* --------------------- VISCOELASTIC PARAMETERS --------------------- */
constexpr dfloat TAU = 0.8;                // Fixed relaxation time
constexpr dfloat BETA = 0.1;               // Viscosity ratio

constexpr dfloat VISC = (TAU - 0.5)/3.0;   // Solvent viscosity
constexpr dfloat SUM_VISC = VISC/BETA;     // Total viscosity

/* Reynolds number — fixed target, independent of grid size */
constexpr dfloat RE = 1;

/* Centerline velocity derived from Re (adapts to N and tau automatically) */
constexpr dfloat U_C = RE*SUM_VISC/L;

constexpr dfloat t_c = L/U_C;

/* --------------------- TIME STEPPING --------------------- */
constexpr int CONFORMATION_SWITCH = 0;
constexpr int N_STEPS = 20*t_c;
constexpr int INI_STEP = 0;

/* --------------------- BODY FORCES --------------------- */
constexpr dfloat FX = 8.0*SUM_VISC*U_C/(L*L);
constexpr dfloat FY = 0.0;
constexpr dfloat FZ = 0.0;

/* --------------------- INITIAL CONDITIONS --------------------- */
constexpr dfloat U_0_X = 0.0;               // Initial velocity X
constexpr dfloat U_0_Y = 0.0;               // Initial velocity Y
constexpr dfloat U_0_Z = 0.0;               // Initial velocity Z
constexpr dfloat RHO_0 = 1.0;               // Initial density

/* --------------------- BOUNDARY CONDITION VALUES --------------------- */
__device__ const dfloat UX_BC[4] = {0, 0, 0, 0};
__device__ const dfloat UY_BC[4] = {0, 0, 0, 0};
__device__ const dfloat UZ_BC[4] = {0, 0, 0, 0};
__device__ const dfloat RHO_BC[4] = {RHO_0, RHO_0, RHO_0, RHO_0};

/* --------------------- BOUNDARY CONDITION TYPES --------------------- */
#define BC_X_PERIODIC
#define BC_Y_WALL
#define BC_Z_PERIODIC

/* --------------------- INITIALIZATION LOADING DEFINES -------------------- */
#define LOAD_CHECKPOINT false   // loads simulation checkpoint from folder

// Folder with simulation to load data from last checkpoint. 
// WITHOUT ID_SIM (change it in ID_SIM) AND "/" AT THE END
#define SIMULATION_FOLDER_LOAD_CHECKPOINT "TEST"
// Interval to make checkpoint to save all simulation data and restart from it.
// It must not be very frequent (10000 or more), because it takes a long time
#define CHECKPOINT_SAVE false // the frequency on which the simulation checkpoint is saved
#define LOAD_FIELD false
#define LOAD_FIELD_STEP 0

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


    constexpr dfloat CONF_ZERO = (dfloat)(1.0);

    constexpr dfloat nu_p = SUM_VISC - VISC;
    constexpr dfloat Weissenberg_number = 1.0;

    constexpr dfloat LAMBDA = Weissenberg_number*t_c;
    constexpr dfloat inv_lambda = 1.0/LAMBDA;
    constexpr dfloat fenep_re = 10.0;

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

    //ADVECTION-DIFFUSIONS CONSTANTS
    constexpr dfloat CONF_DIFFUSIVITY_RATIO = 2000; //OR Pr in malaspinas
    constexpr dfloat CONF_DIFFUSIVITY = VISC/CONF_DIFFUSIVITY_RATIO; // alpha

    constexpr dfloat CONF_DIFF_REF = (1.0/6);
    constexpr dfloat CONF_TAU = CONF_DIFFUSIVITY*3+0.5;
    constexpr dfloat CONF_OMEGA  = 1.0/CONF_TAU;
    constexpr dfloat CONF_TT_OMEGA = 1.0-0.5*CONF_OMEGA;

    constexpr dfloat CONF_DIFF_FLUC = CONF_DIFFUSIVITY - CONF_DIFF_REF;
    constexpr dfloat CONF_AAA = (3*CONF_DIFF_FLUC*CONF_OMEGA);
    constexpr dfloat CONF_DIFF_FLUC_COEF = CONF_AAA/(1.0+CONF_AAA);
#endif
