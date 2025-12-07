#define PHASE_MODEL 

constexpr int SCALE = 1;
constexpr int N_STEPS = 400000;
constexpr int N = 128 * SCALE;
constexpr int NX = N;        // size x of the grid 
constexpr int NY = N;        // size y of the grid
constexpr int NZ = N;        // size z of the grid in one GPU
constexpr int NZ_TOTAL = NZ;       // size z of the grid

constexpr dfloat U_MAX = 0.1;  
constexpr dfloat L = N;

constexpr dfloat FX = 0.0;        // force in x
constexpr dfloat FY = 0.0;        // force in y
constexpr dfloat FZ = 0.0;        // force in z (flow direction in most cases)

// value for the velocity initial condition in the domain
constexpr dfloat U_0_X = 0.0;
constexpr dfloat U_0_Y = 0.0;
constexpr dfloat U_0_Z = 0.0;
constexpr dfloat RHO_0 = 1.0;

// values options for boundary conditions //not used yet
__device__ const dfloat UX_BC[4] =  {0, 0, 0, 0};
__device__ const dfloat UY_BC[4] =  {0, 0, 0, 0};
__device__ const dfloat UZ_BC[4] =  {0, 0, 0, 0};
__device__ const dfloat RHO_BC[4] = {RHO_0, RHO_0, RHO_0, RHO_0};

#define BC_X_WALL
#define BC_Y_WALL
#define BC_Z_WALL

constexpr dfloat TAU = 0.55;//0.5 + 3.0*VISC;     // relaxation time
constexpr dfloat VISC = (TAU-0.5)/3;
constexpr dfloat MACH_NUMBER = U_MAX/0.57735026918962;


#ifdef PHASE_MODEL
    #define CONVECTION_DIFFUSION_TRANSPORT
    #define D3G19

    #define PHI_DIST

    constexpr dfloat PHI_REFERENCE = (dfloat)(0.5);
    constexpr dfloat PHI_DELTA_PHI = 1;
    constexpr dfloat PHI_ONE = PHI_REFERENCE - PHI_DELTA_PHI/2.0;
    constexpr dfloat PHI_TWO = PHI_REFERENCE + PHI_DELTA_PHI/2.0;
    constexpr dfloat PHI_ALPHA = 0.05;
    constexpr dfloat PHI_ZERO = (dfloat)(10.0);
    
    //model constanst
    constexpr dfloat phi_sigma = 1e-4; //surface tension coeficiet
    constexpr dfloat phi_del = 3; //interface width
    constexpr dfloat phi_beta = 12.0*phi_sigma / phi_del;
    constexpr dfloat phi_kappa = 3.0*phi_sigma*phi_del/2;

    //ADVECTION-DIFFUSIONS CONSTANTS
    constexpr dfloat PHI_DIFFUSIVITY_RATIO = 1000; //OR Pr in malaspinas
    constexpr dfloat PHI_DIFFUSIVITY = VISC/PHI_DIFFUSIVITY_RATIO; // a0lpha

    constexpr dfloat PHI_DIFF_REF = (1.0/6);
    constexpr dfloat PHI_TAU = PHI_DIFF_REF*3+0.5;
    constexpr dfloat PHI_OMEGA  = 1.0/PHI_TAU;
    constexpr dfloat PHI_TT_OMEGA = 1.0-0.5*PHI_OMEGA;

    constexpr dfloat PHI_DIFF_FLUC = PHI_DIFFUSIVITY - PHI_DIFF_REF;
    constexpr dfloat PHI_AAA = (3*PHI_DIFF_FLUC*PHI_OMEGA);
    constexpr dfloat PHI_DIFF_FLUC_COEF = PHI_AAA/(1.0+PHI_AAA);
#endif



/* --------------------- INITIALIZATION LOADING DEFINES -------------------- */
constexpr int INI_STEP = 0; // initial simulation step (0 default)
#define LOAD_CHECKPOINT false   // loads simulation checkpoint from folder 
                                // (folder name defined below)


// Folder with simulation to load data from last checkpoint. 
// WITHOUT ID_SIM (change it in ID_SIM) AND "/" AT THE END
#define SIMULATION_FOLDER_LOAD_CHECKPOINT "TEST"
// Interval to make checkpoint to save all simulation data and restart from it.
// It must not be very frequent (10000 or more), because it takes a long time
#define CHECKPOINT_SAVE false // the frequency on which the simulation checkpoint is saved
#define LOAD_FIELD false
#define LOAD_FIELD_STEP 0