/* ============================================================================
 * OUTPUT CONFIGURATION
 * Case: Rayleigh-Benard Convection (Closed)
 * Description: Thermal convection in closed domain with Bingham fluid model
 * ========================================================================== */

/* --------------------- OUTPUT FILES --------------------- */
#define ID_SIM "000"                    // Prefix for simulation's files
#define PATH_FILES "RayleighBenardClosed" // Path to save simulation's files

/* --------------------- OUTPUT FORMATS --------------------- */
#define VTK_SAVE (true)                 // Save VTK format (ParaView)
#define BIN_SAVE (false)                // Save binary format
#define ONLY_FINAL_MACRO (false)        // Save only last time step macroscopic
#define MACR_SAVE (3000000)             // Save macroscopic fields
#define REPORT_SAVE (200)               // Console report frequency (steps)

/* --------------------- DATA TREATMENT --------------------- */
#define TREATFIELD (true)               // Treat data over entire field
#define TREATPOINT (false)              // Treat data at specific points
#define TREATLINE (false)               // Save macro along a line
#define SAVEDATA (true)                 // Save treated data
#define CONSOLEPRINT (false)            // Print data saving to console
#define MEAN_FLOW (false)               // Store mean flow for turbulence statistics
#define NODE_TYPE_SAVE (false)          // Save BC conditions for visualization


//#define BC_FORCES //create scalar field to export the reaction forces from BC;
//#define SAVE_BC_FORCES // define if it will export BC force field to bin

