/* ============================================================================
 * OUTPUT CONFIGURATION
 * Case: 3D Rayleigh-Taylor Instability verification
 * ========================================================================== */

/* --------------------- OUTPUT FILES --------------------- */
#define ID_SIM "001"
#define PATH_FILES "RayleighTaylor_3D"

/* --------------------- OUTPUT FORMATS --------------------- */
#define VTK_SAVE (true)
#define BIN_SAVE (false)
#define ONLY_FINAL_MACRO (false)
#define HAS_VTK_TIME
#if defined(HAS_VTK_TIME)
#define VTK_SAVE_EVERY_TIME (0.25_df)
#define MACR_SAVE ((int)(VTK_SAVE_EVERY_TIME / vtk_time))
#else
#define MACR_SAVE (1000)
#endif
#define REPORT_SAVE (100)

/* --------------------- DATA TREATMENT --------------------- */
#define TREATFIELD (false)
#define TREATPOINT (false)
#define TREATLINE (false)
#define SAVEDATA (true)
#define CONSOLEPRINT (false)
#define MEAN_FLOW (false)
#define NODE_TYPE_SAVE (false)

/* --------------------- PARTICLE TRACER --------------------- */
// #define PARTICLE_TRACER
#define PARTICLE_TRACER_SAVE false
