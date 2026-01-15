/* ============================================================================
 * FLOW INITIALIZATION
 * Case: 000_benchmark
 * 
 * Initializes the flow field with small random perturbations for turbulence
 * development. Uses random number array generated when RANDOM_NUMBERS is enabled.
 * ========================================================================== */

rho = RHO_0;
ux = U_0_X;
uy = U_0_Y;
uz = U_0_Z;

// Add random perturbations for turbulence
dfloat pert = 0.01;
int l = idxScalarGlobal(x, y, z);
int Nt = NUMBER_LBM_NODES;
dfloat xc = (NX-1)/2.0;
dfloat yc = (NY-1)/2.0;

dfloat distX = (x-xc)/(NX-1);
dfloat distY = (y-yc)/(NY-1);

dfloat velX = distY * pert;
dfloat velY = distY * pert;

ux += pert*randomNumbers[l + x - Nt*((l + x) / Nt)];
uy += pert*randomNumbers[l + y - Nt*((l + y) / Nt)];
uz += pert*randomNumbers[l + z - Nt*((l + z) / Nt)];
