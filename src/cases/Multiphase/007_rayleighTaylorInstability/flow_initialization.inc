rho = RHO_0;
ux = U_0_X;
uy = U_0_Y;
uz = U_0_Z;


#ifdef PHI_DIST

const dfloat phi_mid  = 0.5 * (PHI_ONE + PHI_TWO);
const dfloat phi_half = 0.5 * (PHI_TWO - PHI_ONE);

const dfloat denom = sqrt(2.0) * phi_xi;

dfloat X = (dfloat)x;
dfloat Y = (dfloat)y;
dfloat Z = (dfloat)z;

dfloat y0 = 0.5 * NY;

// small sinusoidal perturbation
dfloat amplitude = 4.0;
dfloat wavelength = NX;

dfloat perturb = amplitude * cos(2.0 * M_PI * X / wavelength);

// signed distance
dfloat delta = Y - (y0 + perturb);

// heavy fluid on top
phiVar = phi_mid + phi_half * tanh( -delta / denom );
phiVar += PHI_ZERO;

#endif
