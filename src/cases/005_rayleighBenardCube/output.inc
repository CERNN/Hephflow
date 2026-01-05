/* ============================================================================
 * OUTPUT CONFIGURATION
 * Case: Rayleigh-Benard Convection (Cube)
 * Description: Thermal convection with cubic domain and Bingham fluid model
 * ========================================================================== */

/* --------------------- OUTPUT FILES --------------------- */
#define ID_SIM "001"                    // Prefix for simulation's files
#define PATH_FILES "RayleighBenardCube" // Path to save simulation's files

/* --------------------- OUTPUT FORMATS --------------------- */
#define ONLY_FINAL_MACRO (false)        // Save only last time step macroscopic
#define MACR_SAVE (1000)                // Save macroscopic fields
#define REPORT_SAVE (200)               // Console report frequency (steps)

/* --------------------- OUTPUT SAVE OPTIONS --------------------- */
#define VTK_SAVE (true)                 // Save VTK format (ParaView)
#define BIN_SAVE (false)                // Save binary format


//#define BC_FORCES //create scalar field to export the reaction forces from BC;
//#define SAVE_BC_FORCES // define if it will export BC force field to bin

