/* ============================================================================
 * OUTPUT CONFIGURATION
 * Case: 001_pipeFlowCurved_Turbulent
 * Description: Turbulent Newtonian flow through a circular duct
 * ========================================================================== */

/* --------------------- OUTPUT FILES --------------------- */
#define ID_SIM "002"                    // Prefix for simulation's files
#define PATH_FILES "PipeFlowCurved_Turbulent" // Path to save simulation files
#define ONLY_FINAL_MACRO (false)        // Save only last time step macroscopic

/* --------------------- OUTPUT FORMATS --------------------- */
#define VTK_SAVE (false)                 // Save VTK format (ParaView)
#define BIN_SAVE (false)                // Save binary format
#define MACR_SAVE (34000)               // Save macroscopic fields every N steps
#define REPORT_SAVE (340)               // Sample and save radial statistics
#define NODE_TYPE_SAVE (false)          // Save BC conditions for visualization

/* --------------------- DATA TREATMENT --------------------- */
#define TREATFIELD (false)              // Treat data over entire field
#define TREATPOINT (false)              // Treat data at specific points
#define TREATLINE (false)               // Save macro along a line
#define SAVEDATA (true)                 // Save treated data
#define CONSOLEPRINT (false)            // Print data saving to console
#define MEAN_FLOW (false)               // Temporal averaging is deferred to post-processing

//#define BC_FORCES //create scalar field to export the reaction forces from BC;
//#define SAVE_BC_FORCES // define if it will export BC force field to bin
