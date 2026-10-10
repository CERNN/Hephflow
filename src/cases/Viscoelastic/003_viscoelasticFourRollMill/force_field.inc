/* ============================================================================
 * FORCE FIELD DEFINITION
 * Case: 003_viscoelasticFourRollMill
 * Description: Four-roll mill body force field (Kolmogorov-type forcing)
 *              Drives the steady-state four-roll mill flow pattern.
 *              Reference: Thomases & Shelley (2007)
 *
 * Available variables (from kernel scope):
 *   x, y, z  -- global grid coordinates (integers)
 *   L_Fx, L_Fy, L_Fz -- force accumulators (modified in-place)
 * ========================================================================== */

// F_x =  F_0 * sin(K_const*x) * cos(K_const*y)
// F_y = -F_0 * sin(K_const*y) * cos(K_const*x)
L_Fx += F_0 * sin(K_const * x) * cos(K_const * y);
L_Fy += -F_0 * sin(K_const * y) * cos(K_const * x);
L_Fz += 0;