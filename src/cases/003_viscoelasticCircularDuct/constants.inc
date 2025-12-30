/* ============================================================================
 * SIMULATION CONSTANTS
 * Case: 003_viscoelasticCircularDuct
 * Description: Viscoelastic flow through circular duct (FENE-P model)
 * ========================================================================== */

/* --------------------- DIMENSIONLESS NUMBERS --------------------- */
constexpr dfloat RE = 10;                   // Reynolds number
constexpr dfloat BETA = 1;                  // Viscosity ratio (R_v)

/* --------------------- SIMULATION SCALE --------------------- */
constexpr int SCALE = 1;                    // Grid scaling factor

/* --------------------- GRID DIMENSIONS --------------------- */
constexpr int N = 64 * SCALE;               // Base grid size
constexpr int NX = N;                       // Grid size in X
constexpr int NY = N;                       // Grid size in Y
constexpr int NZ = 4;                       // Grid size in Z
constexpr int NZ_TOTAL = NZ;                // Total Z grid size

/* --------------------- REFERENCE VALUES --------------------- */
constexpr dfloat L = N;                     // Characteristic length
constexpr dfloat U_C = 0.01;                // Characteristic velocity
constexpr dfloat U_MAX = 0.0;               // Maximum velocity
constexpr dfloat SUM_VISC = (U_C*L)/RE;     // Total viscosity
constexpr dfloat VISC = BETA * SUM_VISC;    // Solvent viscosity
constexpr dfloat TAU = 0.5 + 3.0*VISC;      // Relaxation time
constexpr dfloat t_c = L/U_C;               // Characteristic time
constexpr dfloat MACH_NUMBER = U_MAX/0.57735026918962; // Mach number

/* --------------------- GEOMETRY PARAMETERS --------------------- */
#define VOXEL_FILENAME "circular_duct.csv"   // Voxel geometry file
#define DUCT_CENTER_X (((NX-1)/2.0))        // Duct center X coordinate
#define DUCT_CENTER_Y (((NY-1)/2.0))        // Duct center Y coordinate

/* --------------------- TIME STEPPING --------------------- */
constexpr int CONFORMATION_SWITCH = 0;      // Step to switch on conformation
constexpr int N_STEPS = 5;                  // Total simulation steps
constexpr int INI_STEP = 0;                 // Initial step (for restarts)

/* --------------------- BODY FORCES --------------------- */
constexpr dfloat FX = 0.0;                  // Force in X direction
constexpr dfloat FY = 0.0;                  // Force in Y direction
constexpr dfloat FZ = 1e-6;                 // Force in Z direction (flow direction)
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
#define BC_Y_WALL                           // Wall BC in Y
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

    constexpr dfloat nu_p = SUM_VISC - VISC;
    constexpr dfloat Weissenberg_number = 0.0;

    constexpr dfloat LAMBDA = Weissenberg_number*t_c;
    constexpr dfloat inv_lambda = (Weissenberg_number == 0) ? 0.0 : 1.0/LAMBDA;
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