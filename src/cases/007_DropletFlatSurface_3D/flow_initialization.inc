rho = RHO_0;
ux  = U_0_X;
uy  = U_0_Y;
uz  = U_0_Z;

#ifdef PHI_DIST
const dfloat phi_mid = 0.5_df * (PHI_ONE + PHI_TWO);
const dfloat phi_half = 0.5_df * (PHI_TWO - PHI_ONE);

const dfloat xc = 0.5_df * (dfloat)NX;
const dfloat zc = 0.5_df * (dfloat)NZ;

const dfloat X = (dfloat)x;
const dfloat Y = (dfloat)y;
const dfloat Z = (dfloat)z;

const dfloat rr = sqrt((X - xc) * (X - xc) + Y * Y + (Z - zc) * (Z - zc));
const dfloat delta = R0 - rr;

// Hemispherical droplet touching the bottom wall (y=0).
phiVar = phi_mid + phi_half * tanh(2.0_df * delta / phi_del);
phiVar += PHI_ZERO;
#endif
