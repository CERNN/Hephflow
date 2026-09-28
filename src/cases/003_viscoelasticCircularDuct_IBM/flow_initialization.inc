rho = RHO_0;
ux = U_0_X;
uy = U_0_Y;
uz = U_0_Z;

#ifdef A_XX_DIST
AxxVar = CONF_ZERO + 1.0_df;
Axx_qx_t30 = F_M_I_SCALE * AxxVar * ux;
Axx_qy_t30 = F_M_I_SCALE * AxxVar * uy;
Axx_qz_t30 = F_M_I_SCALE * AxxVar * uz;
#endif
#ifdef A_XY_DIST
AxyVar = CONF_ZERO;
Axy_qx_t30 = Axy_qy_t30 = Axy_qz_t30 = 0.0_df;
#endif
#ifdef A_XZ_DIST
AxzVar = CONF_ZERO;
Axz_qx_t30 = Axz_qy_t30 = Axz_qz_t30 = 0.0_df;
#endif
#ifdef A_YY_DIST
AyyVar = CONF_ZERO + 1.0_df;
Ayy_qx_t30 = F_M_I_SCALE * AyyVar * ux;
Ayy_qy_t30 = F_M_I_SCALE * AyyVar * uy;
Ayy_qz_t30 = F_M_I_SCALE * AyyVar * uz;
#endif
#ifdef A_YZ_DIST
AyzVar = CONF_ZERO;
Ayz_qx_t30 = Ayz_qy_t30 = Ayz_qz_t30 = 0.0_df;
#endif
#ifdef A_ZZ_DIST
AzzVar = CONF_ZERO + 1.0_df;
Azz_qx_t30 = F_M_I_SCALE * AzzVar * ux;
Azz_qy_t30 = F_M_I_SCALE * AzzVar * uy;
Azz_qz_t30 = F_M_I_SCALE * AzzVar * uz;
#endif
