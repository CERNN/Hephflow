/* ============================================================================
 * OUTPUT CONFIGURATION
 * Case: 001_pipeFlowStair_Turbulent
 * Description: Turbulent Newtonian flow through a circular duct using the
 *              staircase boundary-condition formulation
 * ========================================================================== */

/* --------------------- OUTPUT FILES --------------------- */
#define ID_SIM "001"                    // Prefix for simulation's files
#define PATH_FILES "PipeFlowStair_Turbulent" // Path to save simulation files
#define ONLY_FINAL_MACRO (false)        // Save only last time step macroscopic

/* --------------------- OUTPUT FORMATS --------------------- */
#define VTK_SAVE (true)                 // Save VTK format (ParaView)
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

/* --------------------- CHECKPOINT CONFIGURATION --------------------- */
#define LOAD_CHECKPOINT false           // Load simulation from checkpoint
#define SIMULATION_FOLDER_LOAD_CHECKPOINT "TEST" // Folder to load checkpoint from
#define CHECKPOINT_SAVE false           // Save simulation checkpoints
#define LOAD_FIELD false                // Load field data
#define LOAD_FIELD_STEP 0               // Load field step

//#define BC_FORCES //create scalar field to export the reaction forces from BC;
//#define SAVE_BC_FORCES // define if it will export BC force field to bin
