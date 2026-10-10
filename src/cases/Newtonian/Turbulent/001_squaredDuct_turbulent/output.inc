/* ============================================================================
 * OUTPUT CONFIGURATION
 * Case: 001_squaredDuct_turbulent
 * ========================================================================== */

#define ID_SIM "000"
#define PATH_FILES "SquaredDuct_Turbulent"

#define TREATFIELD (false)
#define TREATPOINT (false)
#define TREATLINE (false)
#define SAVEDATA (true)
#define CONSOLEPRINT (false)

// Cross-sectional turbulent statistics are accumulated explicitly in
// treat_data.inc, so the full 3D mean-flow allocation is unnecessary.
#define MEAN_FLOW (false)

#define NODE_TYPE_SAVE (false)
#define ONLY_FINAL_MACRO (false)
#define MACR_SAVE (34000)
#define REPORT_SAVE (340)
#define VTK_SAVE (true)
#define BIN_SAVE (false)

//#define BC_FORCES
//#define SAVE_BC_FORCES
