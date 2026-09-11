/* ============================================================================
 * OUTPUT CONFIGURATION
 * Case: 003_viscoelastic_pipeFlowCurved_D3Q19
 * ========================================================================== */

/* --------------------- OUTPUT FILES --------------------- */
#define ID_SIM "003"
#define PATH_FILES "003_viscoelastic_pipeFlowCurved_D3Q19"
#define ONLY_FINAL_MACRO (false)        // Save only last time step macroscopic

/* --------------------- OUTPUT FORMATS --------------------- */
#define VTK_SAVE (true)                 // Save VTK format (ParaView)
#define BIN_SAVE (false)                // Save binary format
#define MACR_SAVE (500)
#define REPORT_SAVE ((int)(t_c / 20.0_df))
#define NODE_TYPE_SAVE (false)          // Save BC conditions for visualization

/* --------------------- DATA TREATMENT --------------------- */
#define TREATFIELD (false)              // Treat data over entire field
#define TREATPOINT (false)              // Treat data at specific points
#define TREATLINE (true)                // Save radial centerline profiles
#define SAVEDATA (true)                 // Save validation metrics
#define CONSOLEPRINT (false)            // Print data saving to console
#define MEAN_FLOW (false)               // Store mean flow for turbulence statistics

/* --------------------- CHECKPOINT CONFIGURATION --------------------- */
#define LOAD_CHECKPOINT false           // Load simulation from checkpoint
#define SIMULATION_FOLDER_LOAD_CHECKPOINT "TEST" // Folder to load checkpoint from
#define CHECKPOINT_SAVE false           // Save simulation checkpoints
#define LOAD_FIELD false                // Load field data
#define LOAD_FIELD_STEP 0               // Step to load field from

//#define BC_FORCES //create scalar field to export the reaction forces from BC;
//#define SAVE_BC_FORCES // define if it will export BC force field to bin

