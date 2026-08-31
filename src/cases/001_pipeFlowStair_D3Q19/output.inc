/* ============================================================================
 * OUTPUT CONFIGURATION
 * Case: 001_pipeFlowStair_D3Q19
 * ========================================================================== */

/* --------------------- OUTPUT FILES --------------------- */
#define ID_SIM "001"                    // Prefix for simulation's files
#define PATH_FILES "PipeFlowStaircase"  // Path to save simulation's files
#define ONLY_FINAL_MACRO (false)        // Save only last time step macroscopic

/* --------------------- OUTPUT FORMATS --------------------- */
#define VTK_SAVE (true)                 // Save VTK format (ParaView)
#define BIN_SAVE (false)                // Save binary format
#define MACR_SAVE (100)                 // Save macroscopic fields every N steps
#define REPORT_SAVE (false)             // Save report files
#define NODE_TYPE_SAVE (false)          // Save BC conditions for visualization

/* --------------------- DATA TREATMENT --------------------- */
#define TREATFIELD (false)              // Treat data over entire field
#define TREATPOINT (false)              // Treat data at specific points
#define TREATLINE (false)               // Save macro along a line
#define SAVEDATA (false)                // Save treated data
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

/* --------------------- PARTICLE SAVE CONFIGURATION --------------------- */
#define PARTICLES_SAVE (100)            // Save particles info every N steps
#define PARTICLES_DATA_REPORT (false)   // Report IBM treated data every N steps
#define IBM_PARTICLES_NODES_SAVE true   // Saves particles nodes data

//#define PARTICLE_DEBUG