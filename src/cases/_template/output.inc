/* ============================================================================
 * OUTPUT CONFIGURATION
 * Defines data saving, reporting, and file management
 * ========================================================================== */

/* --------------------- FILE IDENTIFICATION --------------------- */
#define ID_SIM "000"                        // Simulation identifier
#define PATH_FILES "CaseName"               // Output folder name

/* --------------------- DATA TREATMENT --------------------- */
//#define TREAT_DATA                        // Enable custom data treatment (see treatData.cu)
#define SAVEDATA (false)                    // Save treated data to file
#define CONSOLEPRINT (false)                // Print treated data to console

/* --------------------- FLOW STATISTICS --------------------- */
#define MEAN_FLOW (false)                   // Compute time-averaged flow fields

/* --------------------- OUTPUT CONTROL --------------------- */
#define ONLY_FINAL_MACRO (true)             // Save only final timestep macroscopic
#define MACR_SAVE (false)                   // Macroscopic save frequency (steps), false = disabled
#define REPORT_SAVE (false)                 // Console report frequency (steps), false = disabled
#define VTK_SAVE (true)                     // Enable VTK output for visualization
#define BIN_SAVE (false)                    // Enable binary output

/* --------------------- CHECKPOINT CONFIGURATION --------------------- */
#define LOAD_CHECKPOINT false               // Load simulation from checkpoint
#define SIMULATION_FOLDER_LOAD_CHECKPOINT "checkpoint_folder"
#define CHECKPOINT_SAVE false               // Save checkpoints during simulation
#define LOAD_FIELD false                    // Load initial field from file
#define LOAD_FIELD_STEP 0                   // Timestep to load field from

/* --------------------- BOUNDARY FORCES --------------------- */
//#define BC_FORCES                         // Compute boundary forces
//#define SAVE_BC_FORCES                    // Save boundary forces to file

/* --------------------- PARTICLE OUTPUT --------------------- */
#ifdef PARTICLE_MODEL
#define PARTICLES_SAVE (100)                // Save particles every # steps
#define PARTICLES_DATA_REPORT (false)       // Report IBM data (false = disabled)
#define IBM_PARTICLES_NODES_SAVE true       // Save particle node positions
//#define PARTICLE_DEBUG                    // Enable particle debug output
#endif
