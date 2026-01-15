/* ============================================================================
 * CONSTANTS CONFIGURATION
 * Case: 003_viscoelasticFourRollMill
 * Description: Viscoelastic four-roll mill flow (FENE-P model)
 * ========================================================================== */

/* --------------------- DIMENSIONLESS NUMBERS --------------------- */
constexpr dfloat RE = 1;                    // Reynolds number
constexpr dfloat BETA = 2.0/3.0;            // Viscosity ratio (Thomases2007)

/* --------------------- SIMULATION SCALE --------------------- */
constexpr int SCALE = 1;                    // Grid scaling factor

/* --------------------- GRID DIMENSIONS --------------------- */
constexpr int N = 256 * SCALE;              // Base grid size
constexpr int NX = N;                       // Grid size in X
constexpr int NY = N;                       // Grid size in Y
constexpr int NZ = 4;                       // Grid size in Z
constexpr int NZ_TOTAL = NZ;                // Total Z grid size

/* --------------------- REFERENCE VALUES --------------------- */
constexpr dfloat K_const = 2.0*M_PI/(dfloat)N; // Wave number constant
constexpr dfloat U_MAX = 0.004908875;       // Maximum velocity
constexpr dfloat SUM_VISC = U_MAX/(RE*K_const); // Total viscosity
constexpr dfloat VISC = BETA * SUM_VISC;    // Solvent viscosity
constexpr dfloat TAU = 0.5 + 3.0*VISC;      // Relaxation time
constexpr dfloat F_0 = 2*U_MAX*VISC*K_const*K_const; // Force magnitude
constexpr dfloat t_c = 2*VISC*K_const/F_0;  // Characteristic time
constexpr dfloat L = N;                     // Characteristic length
constexpr dfloat MACH_NUMBER = U_MAX/0.57735026918962; // Mach number

/* --------------------- TIME STEPPING --------------------- */
constexpr int CONFORMATION_SWITCH = 0*(int)t_c; // Step to switch on conformation
constexpr int N_STEPS = CONFORMATION_SWITCH + 200*(int)t_c; // Total simulation steps
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

/* --------------------- BC VALUES --------------------- */
__device__ const dfloat UX_BC[4] =  {0, 0, 0, 0};
__device__ const dfloat UY_BC[4] =  {0, 0, 0, 0};
__device__ const dfloat UZ_BC[4] =  {0, 0, 0, 0};
__device__ const dfloat RHO_BC[4] = {RHO_0, RHO_0, RHO_0, RHO_0};

/* --------------------- BC TYPES --------------------- */
#define BC_X_PERIODIC                       // Periodic in X
#define BC_Y_PERIODIC                       // Periodic in Y
#define BC_Z_PERIODIC                       // Periodic in Z

/* --------------------- INITIALIZATION LOADING --------------------- */
#define LOAD_CHECKPOINT false               // Load from checkpoint
#define SIMULATION_FOLDER_LOAD_CHECKPOINT "TEST" // Checkpoint folder
#define CHECKPOINT_SAVE false               // Save checkpoints
#define LOAD_FIELD false                    // Load field data
#define LOAD_FIELD_STEP 0                   // Field loading step

/* ========== FENE-P VISCOELASTIC MODEL CONFIGURATION ========== */
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

    constexpr dfloat nu_p = (VISC/BETA) - VISC;


    constexpr dfloat Weissenberg_number = 0.6;



    constexpr dfloat LAMBDA = Weissenberg_number*t_c;
    constexpr dfloat inv_lambda = 1.0/LAMBDA;
    constexpr dfloat fenep_re = 10.0;

    //ADVECTION-DIFFUSIONS CONSTANTS
    constexpr dfloat CONF_DIFFUSIVITY_RATIO = 100; //OR Pr in malaspinas
    constexpr dfloat CONF_DIFFUSIVITY = VISC/CONF_DIFFUSIVITY_RATIO; // alpha

    constexpr dfloat CONF_DIFF_REF = (1.0/6);
    constexpr dfloat CONF_TAU = CONF_DIFF_REF*3+0.5;
    constexpr dfloat CONF_OMEGA  = 1.0/CONF_TAU;
    constexpr dfloat CONF_TT_OMEGA = 1.0-0.5*CONF_OMEGA;

    constexpr dfloat CONF_DIFF_FLUC = CONF_DIFFUSIVITY - CONF_DIFF_REF;
    constexpr dfloat CONF_AAA = (3*CONF_DIFF_FLUC*CONF_OMEGA);
    constexpr dfloat CONF_DIFF_FLUC_COEF = CONF_AAA/(1.0+CONF_AAA);
#endif
