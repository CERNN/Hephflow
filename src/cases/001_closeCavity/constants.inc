/* ============================================================================
 * CONSTANTS CONFIGURATION
 * Case: 001_closeCavity
 * Description: 2D closed cavity with thermal effects
 * 
 * Note: Physics model flags (THERMAL_MODEL, etc.) are defined in model.inc
 * ========================================================================== */

/* --------------------- SIMULATION SCALE --------------------- */
constexpr int SCALE = 1;                    // Grid scaling factor

/* --------------------- TIME STEPPING --------------------- */
constexpr int N_STEPS = 10000;              // Total simulation steps
constexpr int INI_STEP = 0;                 // Initial step (for restarts)

/* --------------------- GRID DIMENSIONS --------------------- */
constexpr int N = 256 * SCALE;              // Reference grid size
constexpr int NX = N;                       // Grid size in X
constexpr int NY = N;                       // Grid size in Y
constexpr int NZ = 2;                       // Grid size in Z (2D simulation)
constexpr int NZ_TOTAL = NZ;                // Total grid size in Z (multi-GPU)

/* --------------------- REFERENCE VALUES --------------------- */
constexpr dfloat U_MAX = 0.0;               // Maximum/reference velocity
constexpr dfloat L = N;                     // Reference length

/* --------------------- BODY FORCES --------------------- */
constexpr dfloat FX = 0.0;                  // Force in X direction
constexpr dfloat FY = 0.0;                  // Force in Y direction
constexpr dfloat FZ = 0.0;                  // Force in Z direction

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
#define BC_Z_WALL

/* --------------------- FLUID PROPERTIES --------------------- */
constexpr dfloat TAU = 0.53;                // Relaxation time
constexpr dfloat VISC = (TAU-0.5)/3.0;      // Kinematic viscosity
constexpr dfloat MACH_NUMBER = U_MAX/0.57735026918962; // Mach number


#ifdef THERMAL_MODEL
    #define SECOND_DIST
    #define CONVECTION_DIFFUSION_TRANSPORT

    #define D3G19

    constexpr bool T_BOUYANCY = true;
    constexpr dfloat T_PR_NUMBER = 0.71; //Prandtl Number
    constexpr dfloat T_RA_NUMBER = 10000.0; // Rayleigh Number
    constexpr dfloat T_GR_NUMBER = T_RA_NUMBER/T_PR_NUMBER; //Grashof number

    constexpr dfloat T_DELTA_T = 0.1; //temperature difference
    constexpr dfloat T_REFERENCE  = 1; //better closer to 1
    constexpr dfloat T_COLD = T_REFERENCE - T_DELTA_T/2.0;
    constexpr dfloat T_HOT = T_REFERENCE + T_DELTA_T/2.0;
    constexpr dfloat C_0 = T_REFERENCE; //initial temperature field 

    constexpr dfloat T_DIFFUSIVITY = VISC/T_PR_NUMBER; // alpha

    constexpr dfloat T_gravity_t_beta = 2*T_RA_NUMBER * T_DIFFUSIVITY*VISC/(T_DELTA_T*L*L*L);
    constexpr dfloat Ra_conf = T_gravity_t_beta * T_DELTA_T*L*L*L*T_PR_NUMBER/(VISC*VISC);
    constexpr dfloat T_Q_INTERNAL_D_Cp = 0.0;

    __device__ const dfloat gravity_vector[3] = {1.0, 0.0, 0.0};

    constexpr dfloat T_DIFF_REF = (1.0/6);
    constexpr dfloat G_TAU = T_DIFF_REF*3+0.5;
    constexpr dfloat G_OMEGA  = 1.0/G_TAU;
    constexpr dfloat G_TT_OMEGA = 1.0-0.5*G_OMEGA;

    constexpr dfloat T_DIFF_FLUC = T_DIFFUSIVITY - T_DIFF_REF;
    constexpr dfloat G_AAA = (3*T_DIFF_FLUC*G_OMEGA);
    constexpr dfloat G_DIFF_FLUC_COEF = G_AAA/(1.0+G_AAA);
#endif //THERMAL_MODEL

/* --------------------- THERMAL MODEL PARAMETERS --------------------- */
// Note: These are only used if THERMAL_MODEL is defined in model.inc