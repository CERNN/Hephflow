/* ============================================================================
 * OUTPUT CONFIGURATION
 * Case: 002_viscoplasticJet
 * Description: Viscoplastic jet flow (Bingham fluid)
 * ========================================================================== */

/* --------------------- FILE IDENTIFICATION --------------------- */
#define ID_SIM "000"                        // Simulation identifier
#define PATH_FILES "ViscoplasticJet"       // Output folder name

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
#define MACR_SAVE (false)                   // Macroscopic save frequency (steps), false = disabled
#define REPORT_SAVE (false)                 // Console report frequency (steps), false = disabled
#define VTK_SAVE (true)                     // Enable VTK output for visualization
#define BIN_SAVE (false)                    // Enable binary output

/* --------------------- BOUNDARY FORCES --------------------- */
//#define BC_FORCES                         // Compute boundary forces
//#define SAVE_BC_FORCES                    // Save boundary forces to file

/* --------------------- PARTICLE TRACER --------------------- */
//#define PARTICLE_TRACER                   // Trace massless particles in flow
#define PARTICLE_TRACER_SAVE false          // Save tracer particles

