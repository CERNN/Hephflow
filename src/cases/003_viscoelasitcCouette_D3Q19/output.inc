/* ============================================================================
 * OUTPUT CONFIGURATION
 * Case: 003_viscoelasitcCouette_D3Q19
 * Description: Viscoelastic Couette flow (FENE-P model)
 * ========================================================================== */

/* --------------------- FILE IDENTIFICATION --------------------- */
#define ID_SIM "000"                        // Simulation identifier
#define PATH_FILES "ViscoelasticCouette"    // Output folder name

/* --------------------- DATA TREATMENT --------------------- */
//#define TREAT_DATA                        // Enable custom data treatment (see treatData.cu)
#define TREATFIELD (false)                  // Treat data over entire field
#define TREATPOINT (false)                  // Treat data at specific points
#define TREATLINE (false)                   // Save macroscopic along a line
#define SAVEDATA (true)                     // Save treated data to file
#define CONSOLEPRINT (false)                // Print treated data to console

/* --------------------- FLOW STATISTICS --------------------- */
#define MEAN_FLOW (false)                   // Compute time-averaged flow fields

/* --------------------- NODE TYPE OUTPUT --------------------- */
#define NODE_TYPE_SAVE (false)              // Save boundary condition node types

/* --------------------- OUTPUT CONTROL --------------------- */
#define ONLY_FINAL_MACRO (false)            // Save only final timestep macroscopic
#define MACR_SAVE (50000)                   // Macroscopic save frequency (steps)
#define REPORT_SAVE (100)                   // Console report frequency (steps)
#define VTK_SAVE (true)                     // Enable VTK output for visualization
#define BIN_SAVE (false)                    // Enable binary output

/* --------------------- BOUNDARY FORCES --------------------- */
//#define BC_FORCES                         // Compute boundary forces
//#define SAVE_BC_FORCES                    // Save boundary forces to file

/* --------------------- PARTICLE TRACER --------------------- */
//#define PARTICLE_TRACER                   // Trace massless particles in flow
#define PARTICLE_TRACER_SAVE false          // Save tracer particles