rho = RHO_0;
ux = U_0_X;
uy = U_0_Y;
uz = U_0_Z;

#ifdef PHI_DIST

const dfloat phi_mid  = 0.5 * (PHI_ONE + PHI_TWO);
const dfloat phi_half = 0.5 * (PHI_TWO - PHI_ONE);

const dfloat a = 20.0;
const dfloat b = 20.0;
const dfloat c = 20.0;

const dfloat denom = sqrt(2.0) * phi_xi;

dfloat X = (dfloat)x - (NX-1)/2.0;
dfloat Y = (dfloat)y - (NY-1)/2.0;
dfloat Z = (dfloat)z - (NZ-1)/2.0;

dfloat f = X*X/(a*a) + Y*Y/(b*b) + Z*Z/(c*c) - 1.0;

// gradient of f
dfloat gx = 2.0 * X / (a*a);
dfloat gy = 2.0 * Y / (b*b);
dfloat gz = 2.0 * Z / (c*c);

dfloat gnorm2 = gx*gx + gy*gy + gz*gz;
dfloat gnorm = sqrt(gnorm2 + 1e-20);

dfloat delta = f / gnorm;  //signed distance ( >0 outside, <0 inside )

phiVar = phi_mid + phi_half * tanh( -delta / denom );
phiVar += PHI_ZERO;

#endif

