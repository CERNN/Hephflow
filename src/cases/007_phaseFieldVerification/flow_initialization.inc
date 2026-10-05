rho = RHO_0;
ux = U_0_X;
uy = U_0_Y;
uz = U_0_Z;
#ifdef PHI_DIST
phiVar = phaseTestInitialPhi((dfloat)x, (dfloat)y, (dfloat)z);
phi_qx_t30 = 3.0_df * phiVar * ux;
phi_qy_t30 = 3.0_df * phiVar * uy;
phi_qz_t30 = 3.0_df * phiVar * uz;
#endif

