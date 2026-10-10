/* ============================================================================
 * OUTPUT CONFIGURATION
 * Case: Couette Flow with Spherical Particle
 * Description: Particle simulation in spherical couette flow
 * ========================================================================== */

/* --------------------- OUTPUT FILES --------------------- */
#define ID_SIM "000"                      // Prefix for simulation's files
#define PATH_FILES "CouetteSpherical"     // Path to save simulation's files

/* --------------------- OUTPUT FORMATS --------------------- */
#define VTK_SAVE (true)                   // Save VTK format (ParaView)
#define BIN_SAVE (false)                  // Save binary format
#define ONLY_FINAL_MACRO (false)          // Save only last time step macroscopic
#define MACR_SAVE (false)                 // Save macroscopic fields
#define REPORT_SAVE (false)               // Console report frequency (steps)

/* --------------------- DATA TREATMENT --------------------- */
#define TREATFIELD (false)                // Treat data over entire field
#define TREATPOINT (false)                // Treat data at specific points
#define TREATLINE (false)                 // Save macro along a line
#define SAVEDATA (false)                  // Save treated data
#define CONSOLEPRINT (false)              // Print data saving to console
#define MEAN_FLOW (false)                 // Store mean flow for turbulence statistics
#define NODE_TYPE_SAVE (false)            // Save BC conditions for visualization

/* ----- END OUTPUT CONFIGURATION ----- */

//#define BC_FORCES //create scalar field to export the reaction forces from BC;
//#define SAVE_BC_FORCES // define if it will export BC force field to bin

/* ------------------------- TIME AND SAVE DEFINES ------------------------- */

#define PARTICLES_SAVE (5000)               // Save particles info every given steps (0 not report)
#define PARTICLES_DATA_REPORT (false)                   // Report IBM treated data every given steps (0 not report)


#define IBM_PARTICLES_NODES_SAVE true      // Saves particles nodes data

//#define PARTICLE_DEBUG
/* ------------------------------------------------------------------------- */