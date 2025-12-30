/* ============================================================================
 * OUTPUT CONFIGURATION
 * Case: 002_viscoplasticParellelPlates_Turbulent
 * Description: Turbulent viscoplastic flow between parallel plates (Bingham)
 * ========================================================================== */

/* --------------------- FILE IDENTIFICATION --------------------- */
#define ID_SIM "000"                        // Simulation identifier
#define PATH_FILES "ViscoplasticParallelPlates_Turbulent"  // Output folder name

/* --------------------- DATA TREATMENT --------------------- */
//#define TREAT_DATA                        // Enable custom data treatment (see treatData.cu)
#define TREATFIELD (false)                  // Treat data over entire field
#define TREATPOINT (false)                  // Treat data at specific points
#define TREATLINE (false)                   // Save macroscopic along a line
#define SAVEDATA (true)                     // Save treated data to file
#define CONSOLEPRINT (false)                // Print treated data to console

/* --------------------- FLOW STATISTICS --------------------- */
#define MEAN_FLOW (true)                    // Compute time-averaged flow fields

/* --------------------- NODE TYPE OUTPUT --------------------- */
#define NODE_TYPE_SAVE (false)              // Save boundary condition node types

/* --------------------- OUTPUT CONTROL --------------------- */
#define ONLY_FINAL_MACRO (false)            // Save only final timestep macroscopic
#define MACR_SAVE (34000)                   // Macroscopic save frequency (steps)
#define REPORT_SAVE (340)                   // Console report frequency (steps)
#define VTK_SAVE (true)                     // Enable VTK output for visualization
#define BIN_SAVE (false)                    // Enable binary output

/* --------------------- BOUNDARY FORCES --------------------- */
//#define BC_FORCES                         // Compute boundary forces
//#define SAVE_BC_FORCES                    // Save boundary forces to file

/* --------------------- PARTICLE TRACER --------------------- */
//#define PARTICLE_TRACER                   // Trace massless particles in flow
#define PARTICLE_TRACER_SAVE false          // Save tracer particles

/* --------------------- CUTOFF PARAMETERS --------------------- */
#define OMEGA_CUTOFF (1e-3)                 // Omega cutoff threshold