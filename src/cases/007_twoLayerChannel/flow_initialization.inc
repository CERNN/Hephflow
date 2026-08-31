rho = RHO_0;
ux  = U_0_X;
uy  = U_0_Y;
uz  = U_0_Z;

#ifdef PHI_DIST

const dfloat phi_mid  = 0.5 * (PHI_ONE + PHI_TWO);
const dfloat phi_half = 0.5 * (PHI_TWO - PHI_ONE);

const dfloat denom = sqrt(2.0) * phi_xi;

dfloat X = (dfloat)x;
dfloat Y = (dfloat)y;
dfloat Z = (dfloat)z;


dfloat y0 = 0.5 * NY;   // interface height

dfloat delta = Y - y0;

// liquid below, gas above
phiVar = phi_mid + phi_half * tanh(delta / denom);

phiVar += PHI_ZERO;

#endif
