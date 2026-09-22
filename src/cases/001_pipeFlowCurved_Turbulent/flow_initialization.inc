// Reuse the parallel-plates turbulent initial field as requested, using the
// distance from the circular duct wall instead of the Cartesian y coordinate.
const dfloat ub_f = 15.6;
const dfloat uc_f = 18.2;
const dfloat dx = static_cast<dfloat>(x) - DUCT_CENTER_X;
const dfloat dy = static_cast<dfloat>(y) - DUCT_CENTER_Y;
const dfloat radius = sqrt(dx*dx + dy*dy);

ux = 0.0;
uy = 0.0;
uz = 0.0;
rho = RHO_0;

if (radius < PIPE_RADIUS) {
    const dfloat pos = PIPE_RADIUS - radius;
    uz = (uc_f*U_TAU)*(pos/PIPE_RADIUS)*(pos/PIPE_RADIUS);

    // Same 10% random perturbation used by 001_parallelPlates_Turbulent.
    const dfloat pert = 0.2;
    const int l = idxScalarGlobal(x, y, z);
    const int Nt = NUMBER_LBM_NODES_LOCAL;
    uz += (ub_f*U_TAU)*pert*randomNumbers[l + z - Nt*((l + z)/Nt)];
    ux += (ub_f*U_TAU)*pert*randomNumbers[l + x - Nt*((l + x)/Nt)];
    uy += (ub_f*U_TAU)*pert*randomNumbers[l + y - Nt*((l + y)/Nt)];
}
