/* ============================================================================
 * OUTPUT CONFIGURATION
 * Case: Vertical viscoelastic displacement between parallel plates
 * Description: Viscoelastic fluid displacing a Newtonian resident fluid
 * ========================================================================== */

/* --------------------- OUTPUT FILES --------------------- */
#define ID_SIM "000"                      // Prefix for simulation's files
#define PATH_FILES "007_saffmanTaylor_NV"   // Path to save simulation's files

/* --------------------- OUTPUT FORMATS --------------------- */
#define VTK_SAVE (true)                   // Save VTK format (ParaView)
#define BIN_SAVE (false)                  // Save binary format
#define ONLY_FINAL_MACRO (false)          // Save only last time step macroscopic
#define MACR_SAVE (100)                   // Save macroscopic fields
#define REPORT_SAVE (500)               // Console report frequency (steps)

/* --------------------- DATA TREATMENT --------------------- */
#define TREATFIELD (true)                // Treat data over entire field
#define TREATPOINT (false)                // Treat data at specific points
#define TREATLINE (false)                 // Save macro along a line
#define SAVEDATA (true)                  // Save treated data
#define CONSOLEPRINT (false)              // Print data saving to console
#define MEAN_FLOW (false)                 // Store mean flow for turbulence statistics
#define NODE_TYPE_SAVE (false)            // Save BC conditions for visualization

/* --------------------- PARTICLE TRACER --------------------- */
//#define PARTICLE_TRACER                 // Enable massless particle tracing
#define PARTICLE_TRACER_SAVE false        // Save particle tracer data
