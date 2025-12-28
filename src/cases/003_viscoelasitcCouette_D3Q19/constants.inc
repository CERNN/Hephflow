/* ============================================================================
 * CONSTANTS CONFIGURATION
 * Case: 003_viscoelasitcCouette_D3Q19
 * Description: Viscoelastic Couette flow (FENE-P model)
 * ========================================================================== */

/* --------------------- DIMENSIONLESS NUMBERS --------------------- */
constexpr dfloat RE = 1;                    // Reynolds number

/* --------------------- SIMULATION SCALE --------------------- */
constexpr int SCALE = 1;                    // Grid scaling factor

/* --------------------- GRID DIMENSIONS --------------------- */
constexpr int N = 128 * SCALE;              // Reference grid size
constexpr int NX = 4;                       // Grid size in X
constexpr int NY = N;                       // Grid size in Y
constexpr int NZ = 4;                       // Grid size in Z
constexpr int NZ_TOTAL = NZ;                // Total grid size in Z (multi-GPU)

/* --------------------- REFERENCE VALUES --------------------- */
constexpr dfloat U_MAX = 0.0125;            // Maximum/reference velocity
constexpr dfloat L = N;                     // Reference length

/* --------------------- VISCOELASTIC PARAMETERS --------------------- */
constexpr dfloat U_C = 2*U_MAX;             // Characteristic velocity
constexpr dfloat SUM_VISC = (U_C*L)/RE;     // Total viscosity
constexpr dfloat BETA = 0.1;                // Viscosity ratio (R_v)
constexpr dfloat VISC = BETA * SUM_VISC;    // Solvent kinematic viscosity
constexpr dfloat TAU = 0.5 + 3.0*VISC;      // Relaxation time
constexpr dfloat t_c = L/U_C;               // Characteristic time

/* --------------------- TIME STEPPING --------------------- */
constexpr int CONFORMATION_SWITCH = 0;      // Step to switch on conformation
constexpr int N_STEPS = 8*20*t_c;           // Total simulation steps
constexpr int INI_STEP = 0;                 // Initial step (for restarts)

/* --------------------- BODY FORCES --------------------- */
constexpr dfloat FX = 0.0;                  // Force in X direction
constexpr dfloat FY = 0.0;                  // Force in Y direction
constexpr dfloat FZ = 0.0;                  // Force in Z direction

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


    constexpr dfloat CONF_ZERO = (dfloat)(2.0);

    constexpr dfloat nu_p = SUM_VISC - VISC;
    constexpr dfloat Weissenberg_number = 8.0;

    constexpr dfloat LAMBDA = Weissenberg_number*t_c;
    constexpr dfloat inv_lambda = 1.0/LAMBDA;
    constexpr dfloat fenep_re = 10.0;

    //ADVECTION-DIFFUSIONS CONSTANTS
    constexpr dfloat CONF_DIFFUSIVITY_RATIO = 100; //OR Pr in malaspinas
    constexpr dfloat CONF_DIFFUSIVITY = VISC/CONF_DIFFUSIVITY_RATIO; // a0lpha

    constexpr dfloat CONF_DIFF_REF = (1.0/6);
    constexpr dfloat CONF_TAU = CONF_DIFF_REF*3+0.5;
    constexpr dfloat CONF_OMEGA  = 1.0/CONF_TAU;
    constexpr dfloat CONF_TT_OMEGA = 1.0-0.5*CONF_OMEGA;

    constexpr dfloat CONF_DIFF_FLUC = CONF_DIFFUSIVITY - CONF_DIFF_REF;
    constexpr dfloat CONF_AAA = (3*CONF_DIFF_FLUC*CONF_OMEGA);
    constexpr dfloat CONF_DIFF_FLUC_COEF = CONF_AAA/(1.0+CONF_AAA);
#endif
