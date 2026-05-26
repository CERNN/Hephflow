rho = RHO_0;
ux = U_0_X;
uy = U_0_Y;
uz = U_0_Z;

#ifdef PHI_DIST

const dfloat phi_mid = 0.5_df * (PHI_ONE + PHI_TWO);
const dfloat phi_half = 0.5_df * (PHI_TWO - PHI_ONE);
const dfloat halfL = 0.5_df * (dfloat)L;

// Map lattice coordinates to physical x,z in [-L/2, L/2].
const dfloat x_phys = (dfloat)x - halfL;
const dfloat z_phys = (dfloat)z - halfL;

// Single-mode RTI perturbation h(x,z) = 0.05*L*(cos(2*pi*x/L)+cos(2*pi*z/L)).
const dfloat h = 0.05_df * (dfloat)L * (
    cos(2.0_df * M_PI * x_phys / (dfloat)L) +
    cos(2.0_df * M_PI * z_phys / (dfloat)L)
);

const dfloat y0 = 0.5_df * (dfloat)NY;
const dfloat delta = (dfloat)y - (y0 + h);

// Heavy liquid (phi=+1) above, lighter gas (phi=-1) below.
phiVar = phi_mid + phi_half * tanh(2.0_df * delta / phi_del);
phiVar += PHI_ZERO;

#endif
