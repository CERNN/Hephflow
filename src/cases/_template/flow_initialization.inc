/* ============================================================================
 * FLOW INITIALIZATION
 * Sets initial velocity and density fields
 * 
 * NOTE: This file is included inside an existing loop over (x, y, z).
 *       Do NOT add for loops here.
 * 
 * Available variables:
 *   x, y, z     - Current node coordinates
 *   rho, ux, uy, uz - Macroscopic fields to initialize
 *   NX, NY, NZ  - Grid dimensions
 * ========================================================================== */

/* --------------------- DEFAULT INITIALIZATION --------------------- */
rho = RHO_0;
ux = U_0_X;
uy = U_0_Y;
uz = U_0_Z;

/* --------------------- OPTIONAL: OMEGA FIELD --------------------- */
#ifdef OMEGA_FIELD
omega = OMEGA;
#endif

/* --------------------- CUSTOM INITIALIZATION --------------------- */
// Add case-specific initialization here

// Example: Parabolic profile for channel flow
// dfloat y_norm = (dfloat)y / (NY - 1);
// uz = 4.0 * U_MAX * y_norm * (1.0 - y_norm);

// Example: Taylor-Green vortex
// dfloat kx = 2.0 * M_PI / NX;
// dfloat ky = 2.0 * M_PI / NY;
// ux =  U_MAX * sin(kx * x) * cos(ky * y);
// uy = -U_MAX * cos(kx * x) * sin(ky * y);

// Example: Random perturbation (requires RANDOM_NUMBERS defined)
// int l = idxScalarGlobal(x, y, z);
// int Nt = NUMBER_LBM_NODES;
// ux += U_MAX * 0.05 * randomNumbers[l + x - Nt*((l + x) / Nt)];
