/* ============================================================================
 * OUTPUT CONFIGURATION
 * Case: Thermal Conduction
 * Description: Pure thermal conduction simulation
 * ========================================================================== */

/* --------------------- OUTPUT FILES --------------------- */
#define ID_SIM "000"                    // Prefix for simulation's files
#define PATH_FILES "ThermalConduction"  // Path to save simulation's files

/* --------------------- OUTPUT FORMATS --------------------- */
#define VTK_SAVE (true)                 // Save VTK format (ParaView)
#define BIN_SAVE (false)                // Save binary format
#define ONLY_FINAL_MACRO (false)        // Save only last time step macroscopic
#define MACR_SAVE (false)               // Save macroscopic fields
#define REPORT_SAVE (false)             // Console report frequency (steps)


//#define BC_FORCES //create scalar field to export the reaction forces from BC;
//#define SAVE_BC_FORCES // define if it will export BC force field to bin

