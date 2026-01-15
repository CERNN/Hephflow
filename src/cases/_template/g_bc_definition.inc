/* ============================================================================
 * THERMAL BOUNDARY CONDITION DEFINITIONS
 * Mathematical implementation of temperature boundary conditions
 * 
 * This file defines boundary conditions for the thermal field when using
 * #define THERMAL_MODEL in model.inc
 * 
 * Available variables:
 *   cVar             - Temperature variable (assigned at boundary)
 *   qx_t30, qy_t30, qz_t30 - Heat flux moments
 *   gNode[Q]         - Thermal distribution functions
 *   T_HOT, T_COLD    - Temperature constants (from constants.inc)
 *   T_REFERENCE      - Reference temperature
 *   T_DELTA_T        - Temperature difference
 * 
 * Common boundary conditions:
 *   - Dirichlet (fixed temperature): Set cVar = T_value, qx/qy/qz_t30 = 0
 *   - Neumann (adiabatic/flux): Use neighboring moments for flux
 *   - Free-slip: Bounce back thermal populations (e.g., gNode[3] = gNode[4])
 * ========================================================================== */

/* --------------------- LOCAL VARIABLES --------------------- */
dfloat c_I;
dfloat inv_c;

/* --------------------- BOUNDARY SWITCH --------------------- */
switch (nodeType){
    case BULK:
        break;

    case SOUTH:
        // Example: Adiabatic wall (zero normal heat flux)
        // Free-slip BC for thermal IS NOT bounce back
        gNode[3] = gNode[4];    // Reflect populations
        gNode[7] = gNode[13];
        gNode[11] = gNode[18];
        gNode[14] = gNode[8];
        gNode[17] = gNode[12];

        // Compute flux from distributions
        qx_t30 = F_M_I_SCALE*((gNode[1] - gNode[2] + gNode[7] - gNode[14] + gNode[9] - gNode[15] + gNode[8] - gNode[13] + gNode[16] - gNode[10]));
        qy_t30 = F_M_I_SCALE*((gNode[3] - gNode[4] + gNode[7] - gNode[8] + gNode[11] - gNode[12] + gNode[14] - gNode[13] + gNode[17] - gNode[18]));
        qz_t30 = F_M_I_SCALE*((gNode[5] - gNode[6] + gNode[9] - gNode[10] + gNode[11] - gNode[12] + gNode[16] - gNode[15] + gNode[18] - gNode[17]));
        break;

    case NORTH:
        // Example: Fixed temperature (hot wall)
        cVar = T_HOT;
        qx_t30 = 0;
        qy_t30 = 0;
        qz_t30 = 0;
        break;

    case WEST:
        // Example: Fixed temperature (hot wall)
        cVar = T_HOT;
        qx_t30 = fMom[idxMom(threadIdx.x+1, threadIdx.y, threadIdx.z, M2_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
        qy_t30 = 0;
        qz_t30 = 0;
        break;

    case EAST:
        // Example: Fixed temperature (cold wall)
        cVar = T_COLD;
        qx_t30 = fMom[idxMom(threadIdx.x-1, threadIdx.y, threadIdx.z, M2_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
        qy_t30 = 0;
        qz_t30 = 0;
        break;

    /* --------------------- EDGE AND CORNER NODES --------------------- */
    case SOUTH_WEST:
        cVar = T_HOT;
        qx_t30 = 0;
        qy_t30 = 0;
        qz_t30 = 0;
        break;

    case SOUTH_EAST:
        cVar = T_COLD;
        qx_t30 = 0;
        qy_t30 = 0;
        qz_t30 = 0;
        break;

    case NORTH_WEST:
        cVar = T_HOT;
        qx_t30 = 0;
        qy_t30 = 0;
        qz_t30 = 0;
        break;

    case NORTH_EAST:
        cVar = T_COLD;
        qx_t30 = 0;
        qy_t30 = 0;
        qz_t30 = 0;
        break;

    /* --------------------- 3D EDGES AND CORNERS --------------------- */
    // Add 3D edge/corner cases if BC_Z_WALL is used
    // Examples: SOUTH_WEST_BACK, SOUTH_WEST_FRONT, etc.

    default:
        break;
}
