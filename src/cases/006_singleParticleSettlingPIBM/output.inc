/* ============================================================================
 * OUTPUT CONFIGURATION
 * Case: Single Particle Settling pIBM
 * Description: Particle settling under gravity in viscous fluid
 * ========================================================================== */

/* --------------------- OUTPUT FILES --------------------- */
#define ID_SIM "000"                            // Prefix for simulation's files
#define PATH_FILES "SingleParticleSettlingPIBM" // Path to save simulation's files

/* --------------------- OUTPUT FORMATS --------------------- */
#define VTK_SAVE (true)          // Save VTK format (ParaView)
#define BIN_SAVE (false)         // Save binary format
#define ONLY_FINAL_MACRO (false) // Save only last time step macroscopic
#define MACR_SAVE (50)           // Save macroscopic fields every # timesteps
#define REPORT_SAVE (false)      // Console report frequency (steps)

/* --------------------- DATA TREATMENT --------------------- */
#define TREATFIELD (false)     // Treat data over entire field
#define TREATPOINT (false)     // Treat data at specific points
#define TREATLINE (false)      // Save macro along a line
#define SAVEDATA (false)       // Save treated data
#define CONSOLEPRINT (false)   // Print data saving to console
#define MEAN_FLOW (false)      // Store mean flow for turbulence statistics
#define NODE_TYPE_SAVE (false) // Save BC conditions for visualization

/* --------------------- PARTICLE OUTPUT --------------------- */
#define PARTICLES_SAVE (50)           // Save particles info every given steps
#define PARTICLES_DATA_REPORT (false) // Report IBM treated data every given steps
#define IBM_PARTICLES_NODES_SAVE true // Saves particles nodes data

// #define PARTICLE_DEBUG
// #define BC_FORCES                        // Create scalar field for BC reaction forces

/* ----- END OUTPUT CONFIGURATION ----- */