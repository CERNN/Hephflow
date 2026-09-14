/* ============================================================================
 * OUTPUT CONFIGURATION
 * Case: 001_womersleyFlow
 * Description: Oscillatory channel flow (planar Womersley flow)
 * ========================================================================== */

/* --------------------- FILE IDENTIFICATION --------------------- */
#define ID_SIM "001"
#define PATH_FILES "WomersleyFlow"

/* --------------------- OUTPUT FORMATS --------------------- */
#define VTK_SAVE (false)                // Disable VTK output (enable only for visualisation runs)
#define BIN_SAVE (false)
#define NODE_TYPE_SAVE (false)

/* --------------------- OUTPUT CONTROL --------------------- */
#define MACR_SAVE   (false)         // VTK save frequency [steps]
#define REPORT_SAVE (SAVE_FREQ)         // treat_data save frequency [steps]
#define ONLY_FINAL_MACRO (false)

/* --------------------- DATA TREATMENT --------------------- */
#define TREATFIELD  (false)
#define TREATPOINT  (false)
#define TREATLINE   (false)
#define SAVEDATA    (true)
#define CONSOLEPRINT (false)
#define MEAN_FLOW   (false)

/* --------------------- BOUNDARY FORCES --------------------- */
//#define BC_FORCES
//#define SAVE_BC_FORCES
