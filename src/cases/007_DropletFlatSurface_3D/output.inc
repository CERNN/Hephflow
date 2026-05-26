/* ============================================================================
 * OUTPUT CONFIGURATION
 * Case: 3D droplet on flat surface (wetting verification)
 * ========================================================================== */

/* --------------------- OUTPUT FILES --------------------- */
#define ID_SIM "000"
#define PATH_FILES "DropletFlatSurface_3D"

/* --------------------- OUTPUT FORMATS --------------------- */
#define VTK_SAVE (true)
#define BIN_SAVE (false)
#define ONLY_FINAL_MACRO (false)
#define MACR_SAVE (2000)
#define REPORT_SAVE (1000)

/* --------------------- DATA TREATMENT --------------------- */
#define TREATFIELD (false)
#define TREATPOINT (false)
#define TREATLINE (false)
#define SAVEDATA (true)
#define CONSOLEPRINT (false)
#define MEAN_FLOW (false)
#define NODE_TYPE_SAVE (false)

/* --------------------- PARTICLE TRACER --------------------- */
#define PARTICLE_TRACER_SAVE false
