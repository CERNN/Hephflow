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

dfloat y0 = 0.35 * NY;   // pool height (adjust as needed)

dfloat delta_pool = Y - y0;

// liquid below, gas above
dfloat phi_pool = phi_mid + phi_half * tanh(-delta_pool / denom);

dfloat xc = 0.5 * NX;     // center in x
dfloat yc = 0.70 * NY;    // above pool
dfloat R  = 20.0;         // droplet radius

dfloat r = sqrt((X - xc)*(X - xc) + (Y - yc)*(Y - yc));

dfloat delta_drop = R - r;

dfloat phi_drop = phi_mid + phi_half * tanh(delta_drop / denom);


// liquid wherever pool OR droplet exists
phiVar = fmax(phi_pool, phi_drop);
phiVar += PHI_ZERO; 
#endif
