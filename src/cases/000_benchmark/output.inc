/* ============================================================================
 * OUTPUT CONFIGURATION
 * Case: 000_benchmark
 * ========================================================================== */

/* --------------------- FILE IDENTIFICATION --------------------- */
#define ID_SIM "000"                        // Simulation identifier
#define PATH_FILES "Benchmark"              // Output folder name

/* --------------------- DATA TREATMENT --------------------- */
//#define TREAT_DATA                        // Enable custom data treatment
#define SAVEDATA (false)                    // Save treated data to file
#define CONSOLEPRINT (false)                // Print treated data to console

/* --------------------- FLOW STATISTICS --------------------- */
#define MEAN_FLOW (false)                   // Compute time-averaged flow fields

/* --------------------- OUTPUT CONTROL --------------------- */
#define ONLY_FINAL_MACRO (true)             // Save only final timestep macroscopic
#define MACR_SAVE (false)                   // Macroscopic save frequency
#define REPORT_SAVE (false)                 // Console report frequency
#define VTK_SAVE (true)                     // Enable VTK output
#define BIN_SAVE (false)                    // Enable binary output

/* --------------------- CHECKPOINT CONFIGURATION --------------------- */
#define LOAD_CHECKPOINT false               // Load simulation from checkpoint
#define SIMULATION_FOLDER_LOAD_CHECKPOINT "TEST"
#define CHECKPOINT_SAVE false               // Save checkpoints during simulation
#define LOAD_FIELD false                    // Load initial field from file
#define LOAD_FIELD_STEP 0                   // Timestep to load field from

/* --------------------- BOUNDARY FORCES --------------------- */
//#define BC_FORCES                         // Compute boundary forces
//#define SAVE_BC_FORCES                    // Save boundary forces to file
