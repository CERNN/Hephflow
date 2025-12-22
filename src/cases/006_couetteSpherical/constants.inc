#define PARTICLE_MODEL 
#define TREAT_DATA_INCLUDE

constexpr dfloat RE = 10;

constexpr int SCALE = 1;
constexpr int N_STEPS = 200000;
constexpr int N = 320 * SCALE;
constexpr int NX = N;        // size x of the grid 
constexpr int NY = N;        // size y of the grid
constexpr int NZ = N;        // size z of the grid in one GPU
constexpr int NZ_TOTAL = NZ;       // size z of the grid

constexpr dfloat U_MAX = 0.0000533333;  
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


#define BC_X_PERIODIC
#define BC_Y_WALL
#define BC_Z_PERIODIC

constexpr dfloat VISC = NY * U_MAX / RE;
constexpr dfloat TAU = 0.9; //0.5 + 3.0*VISC;     // relaxation time
constexpr dfloat MACH_NUMBER = U_MAX/0.57735026918962;

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

#define WALL_VEL_UX U_MAX

#ifdef PARTICLE_MODEL
    #define IBM_METHOD
    #define DEM_METHOD
    #define LOCAL_FORCES
    //number of particles
    #define NUM_PARTICLES 3129 // 391 / 782 / 1565 / 2347 / 3129

    constexpr dfloat GX = 0.0;
    constexpr dfloat GY = 0.0;
    constexpr dfloat GZ = 0.0;

    constexpr dfloat FLUID_DENSITY = 1.0;


    #ifdef DEM_METHOD
        constexpr dfloat PP_FRICTION_COEF = 0.1; // friction coeficient particle particle -- case
        constexpr dfloat PW_FRICTION_COEF = 0.1; // friction coeficient particle wall -- case
        constexpr dfloat PP_REST_COEF = 0.9; // restitution coeficient particle particle -- case
        constexpr dfloat PW_REST_COEF = 0.9; // restitution coeficient particle wall-- case
        //#define REST_COEF_CORRECTION


        //material properties
        constexpr dfloat PARTICLE_YOUNG_MODULUS = 0.3; //-- case
        constexpr dfloat PARTICLE_POISSON_RATIO = 0.3; //-- case


        constexpr dfloat WALL_YOUNG_MODULUS = 10.0; //-- case
        constexpr dfloat WALL_POISSON_RATIO = 0.3; //-- case

        // Sphere-wall collision constants (compile-time)
        constexpr dfloat SPHERE_WALL_DAMPING_CONST = (-2.0 * constexprLn(PW_REST_COEF) / (sqrtt(M_PI * M_PI + constexprLn(PW_REST_COEF) * constexprLn(PW_REST_COEF))));
        constexpr dfloat SPHERE_SPHERE_DAMPING_CONST = (-2.0 * constexprLn(PP_REST_COEF) / (sqrtt(M_PI * M_PI + constexprLn(PP_REST_COEF) * constexprLn(PP_REST_COEF))));

    #endif //DEM_METHOD

    // ---- Lubrication ----
    #define LUB_CUTOFF_FACTOR   0.02 //stop working above this value
    #define LUB_MIN_FACTOR      1e-3 //stop working below this values since the force is proportional to 1/h

    // ---- Repulsion ----
    #define REP_RANGE_FACTOR    0.1 //stops woirking above this value
    #define REP_FORCE_AMPLITUDE  1.0

    // ---- Attraction ----
    #define ATT_RANGE_FACTOR    0.1

    // ---- Van der Waals attraction ----
    #define HAMAKER_CONST 1e-5   // lattice units


#endif //PARTICLE_MODEL
