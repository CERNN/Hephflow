/* ============================================================================
 * OUTPUT CONFIGURATION
 * Case: Couette Flow with Ellipsoid Particle
 * Description: Particle simulation in couette flow with ellipsoidal geometry
 * ========================================================================== */

/* --------------------- OUTPUT FILES --------------------- */
#define ID_SIM "000"                      // Prefix for simulation's files
#define PATH_FILES "CouetteEllipsoid"     // Path to save simulation's files

/* --------------------- OUTPUT FORMATS --------------------- */
#define VTK_SAVE (true)                   // Save VTK format (ParaView)
#define BIN_SAVE (false)                  // Save binary format
#define ONLY_FINAL_MACRO (false)          // Save only last time step macroscopic
#define MACR_SAVE (500)                   // Save macroscopic fields
#define REPORT_SAVE (false)               // Console report frequency (steps)

/* --------------------- DATA TREATMENT --------------------- */
#define TREATFIELD (false)                // Treat data over entire field
#define TREATPOINT (false)                // Treat data at specific points
#define TREATLINE (false)                 // Save macro along a line
#define SAVEDATA (false)                  // Save treated data
#define CONSOLEPRINT (false)              // Print data saving to console
#define MEAN_FLOW (false)                 // Store mean flow for turbulence statistics
#define NODE_TYPE_SAVE (false)            // Save BC conditions for visualization

/* --------------------- PARTICLE OUTPUT --------------------- */
#define PARTICLES_SAVE (500)              // Save particles every # steps
#define PARTICLES_DATA_REPORT (false)     // Report IBM data
#define IBM_PARTICLES_NODES_SAVE true     // Save particle node positions
//#define PARTICLE_DEBUG                  // Enable particle debug output */