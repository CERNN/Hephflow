rho = RHO_0;
// Start from the same fully-developed volume flux imposed at the inlet.
// Initializing only x=0 with this profile creates a large artificial
// divergence and leaves the next streamwise node almost at rest.
const dfloat eta_init = ((dfloat)y) / ((dfloat)(NY - 1));
ux  = 6.0_df * U_IN * eta_init * (1.0_df - eta_init);
uy  = U_0_Y;
uz  = U_0_Z;

#ifdef PHI_DIST

const dfloat phi_mid  = 0.5_df * (PHI_ONE + PHI_TWO);
const dfloat phi_half = 0.5_df * (PHI_TWO - PHI_ONE);
const dfloat denom    = 0.5_df * phi_del;

const dfloat X = (dfloat)x;

// Initial configuration:
//   x < INITIAL_INTERFACE_X : viscoelastic injected phase (PHI_TWO)
//   x > INITIAL_INTERFACE_X : Newtonian resident phase   (PHI_ONE)
// The interface is diffuse over the same thickness used by the CH model.
const dfloat delta = X - INITIAL_INTERFACE_X;
phiVar = phi_mid - phi_half * tanh(delta / denom);

// Smooth indicator of the injected (viscoelastic) phase. It is used below
// to initialize its conformation consistently with the inlet profile while
// retaining A=I in the initially Newtonian resident phase.
const dfloat injected_fraction = myMin(myMax(
    (phiVar - PHI_ONE) / (PHI_TWO - PHI_ONE), 0.0_df), 1.0_df);

#endif

#ifdef OMEGA_FIELD
omega = OMEGA;
#endif

// Fully-developed Oldroyd-B conformation in the injected phase, smoothly
// blended to A=I in the Newtonian resident phase.
#if defined(A_XX_DIST) || defined(A_XY_DIST) || defined(A_XZ_DIST) || \
    defined(A_YY_DIST) || defined(A_YZ_DIST) || defined(A_ZZ_DIST)
const dfloat gamma_init = (6.0_df * U_IN / ((dfloat)(NY - 1)))
                        * (1.0_df - 2.0_df * eta_init);
const dfloat Wi_init = LAMBDA * gamma_init;
#endif
#ifdef A_XX_DIST
AxxVar = CONF_ZERO + 1.0_df
       + injected_fraction * 2.0_df * Wi_init * Wi_init;
Axx_qx_t30 = F_M_I_SCALE*AxxVar*ux;
Axx_qy_t30 = F_M_I_SCALE*AxxVar*uy;
Axx_qz_t30 = F_M_I_SCALE*AxxVar*uz;
#endif
#ifdef A_XY_DIST
AxyVar = CONF_ZERO + injected_fraction * Wi_init;
Axy_qx_t30 = F_M_I_SCALE*AxyVar*ux;
Axy_qy_t30 = F_M_I_SCALE*AxyVar*uy;
Axy_qz_t30 = F_M_I_SCALE*AxyVar*uz;
#endif
#ifdef A_XZ_DIST
AxzVar = CONF_ZERO;
Axz_qx_t30 = F_M_I_SCALE*AxzVar*ux;
Axz_qy_t30 = F_M_I_SCALE*AxzVar*uy;
Axz_qz_t30 = F_M_I_SCALE*AxzVar*uz;
#endif
#ifdef A_YY_DIST
AyyVar = CONF_ZERO + 1.0;
Ayy_qx_t30 = F_M_I_SCALE*AyyVar*ux;
Ayy_qy_t30 = F_M_I_SCALE*AyyVar*uy;
Ayy_qz_t30 = F_M_I_SCALE*AyyVar*uz;
#endif
#ifdef A_YZ_DIST
AyzVar = CONF_ZERO;
Ayz_qx_t30 = F_M_I_SCALE*AyzVar*ux;
Ayz_qy_t30 = F_M_I_SCALE*AyzVar*uy;
Ayz_qz_t30 = F_M_I_SCALE*AyzVar*uz;
#endif
#ifdef A_ZZ_DIST
AzzVar = CONF_ZERO + 1.0;
Azz_qx_t30 = F_M_I_SCALE*AzzVar*ux;
Azz_qy_t30 = F_M_I_SCALE*AzzVar*uy;
Azz_qz_t30 = F_M_I_SCALE*AzzVar*uz;
#endif

#ifdef NON_NEWTONIAN_FLUID
omega = OMEGA;
#endif
