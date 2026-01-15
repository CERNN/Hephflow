/* ============================================================================
 * OUTPUT CONFIGURATION
 * Case: 003_viscoelasitcParallelPlates_D3Q19
 * Description: Viscoelastic flow between parallel plates (FENE-P model)
 * ========================================================================== */

/* --------------------- FILE IDENTIFICATION --------------------- */
#define ID_SIM "000"                        // Simulation identifier
#define PATH_FILES "ViscoelasticParallelPlates"  // Output folder name

/* --------------------- DATA TREATMENT --------------------- */
//#define TREAT_DATA                        // Enable custom data treatment (see treatData.cu)
#define TREATFIELD (false)                  // Treat data over entire field
#define TREATPOINT (false)                  // Treat data at specific points
#define TREATLINE (false)                   // Save macroscopic along a line
#define SAVEDATA (false)                    // Save treated data to file
#define CONSOLEPRINT (false)                // Print treated data to console

/* --------------------- FLOW STATISTICS --------------------- */
#define MEAN_FLOW (false)                   // Compute time-averaged flow fields

/* --------------------- NODE TYPE OUTPUT --------------------- */
#define NODE_TYPE_SAVE (false)              // Save boundary condition node types

/* --------------------- OUTPUT CONTROL --------------------- */
#define ONLY_FINAL_MACRO (false)            // Save only final timestep macroscopic
#define MACR_SAVE (64000)                   // Macroscopic save frequency (steps)
#define REPORT_SAVE (false)                 // Console report frequency (steps), false = disabled
#define VTK_SAVE (true)                     // Enable VTK output for visualization
#define BIN_SAVE (false)                    // Enable binary output

/* --------------------- BOUNDARY FORCES --------------------- */
//#define BC_FORCES                         // Compute boundary forces
//#define SAVE_BC_FORCES                    // Save boundary forces to file
