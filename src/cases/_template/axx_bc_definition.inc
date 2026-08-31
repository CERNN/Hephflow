/* ============================================================================
 * VISCOELASTIC BOUNDARY CONDITION DEFINITIONS - A_XX COMPONENT
 * Polymer stress tensor component σ_xx boundary conditions
 * 
 * This file defines boundary conditions for the A_xx stress component when
 * using #define VISCOELASTIC in model.inc (FENE-P model)
 * 
 * Available variables:
 *   AxxVar          - A_xx stress component at boundary
 *   Axx_qx_t30, Axx_qy_t30, Axx_qz_t30 - Stress flux moments
 *   gNode[Q]        - Stress distribution functions (local populations)
 *   
 * Common approach: Compute stress from distribution functions at boundary
 *   - Sum populations weighted appropriately for each component
 *   - Compute flux moments using velocity-weighted sums
 *   - Normal flux = 0 at walls (zero normal derivative)
 * ========================================================================== */

switch (nodeType) {
    case BULK:
        break;

    case NORTH:
        // Wall at y_max: Compute A_xx from stress distributions
        // Sum includes incoming populations (doubled) and tangential populations
        AxxVar = gNode[0] + gNode[1] + gNode[2] + 2*gNode[3] + gNode[5] + gNode[6] 
               + 2*gNode[7] + gNode[9] + gNode[10] + 2*gNode[11] + 2*gNode[14] 
               + gNode[15] + gNode[16] + 2*gNode[17];

        // Flux moments (F_M_I_SCALE = 1/moment_scale factor)
        Axx_qx_t30 = F_M_I_SCALE*((gNode[1] - gNode[2] + 2*gNode[7] + gNode[9] 
                   - gNode[10] - 2*gNode[14] + gNode[15] - gNode[16]));
        Axx_qy_t30 = 0;  // Zero normal flux at wall
        Axx_qz_t30 = F_M_I_SCALE*((gNode[5] - gNode[6] + gNode[9] - gNode[10] 
                   + 2*gNode[11] + gNode[16] - gNode[15] - 2*gNode[17]));
        break;

    case SOUTH:
        // Wall at y_min: Compute A_xx from stress distributions
        AxxVar = gNode[0] + gNode[1] + gNode[2] + 2*gNode[4] + gNode[5] + gNode[6] 
               + 2*gNode[8] + gNode[9] + gNode[10] + 2*gNode[12] + 2*gNode[13] 
               + gNode[15] + gNode[16] + 2*gNode[18];

        Axx_qx_t30 = F_M_I_SCALE*((gNode[1] - gNode[2] - 2*gNode[8] + gNode[9] 
                   - gNode[10] + 2*gNode[13] + gNode[15] - gNode[16]));
        Axx_qy_t30 = 0;  // Zero normal flux at wall
        Axx_qz_t30 = F_M_I_SCALE*((gNode[5] - gNode[6] + gNode[9] - gNode[10] 
                   - 2*gNode[12] + gNode[16] - gNode[15] + 2*gNode[18]));
        break;

    default:
        break;
}
