rho = RHO_0;
ux  = U_0_X;
uy  = U_0_Y;
uz  = U_0_Z;

#ifdef PHI_DIST
const dfloat phi_mid  = 0.5_df * (PHI_ONE + PHI_TWO);
const dfloat phi_half = 0.5_df * (PHI_TWO - PHI_ONE);

// Bubble center in x-z plane
const dfloat xc = 0.5_df * (dfloat)NX;
const dfloat zc = 0.5_df * (dfloat)NZ;

// Distance between the bubble surface and the wall.
// Set to 0 for exact contact.
const dfloat gap = 1.5_df;

// Bubble center height above the wall.
dfloat R0 = 20.0_df; // Initial bubble radius
const dfloat yc = R0 + gap;

const dfloat X = (dfloat)x;
const dfloat Y = (dfloat)y;
const dfloat Z = (dfloat)z;

// Distance from bubble center
const dfloat rr =
    sqrt((X - xc) * (X - xc) +
         (Y - yc) * (Y - yc) +
         (Z - zc) * (Z - zc));

// Signed distance to the bubble interface
const dfloat delta = R0 - rr;

// PHI_TWO inside the bubble, PHI_ONE outside
phiVar = phi_mid + phi_half * tanh(2.0_df * delta / phi_del);
phiVar += PHI_ZERO;
#endif
