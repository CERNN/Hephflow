#ifdef PHI_DIST
    const dfloat invRhoCol = invRhoPF;
    const dfloat rhoCol = rho_pf;
#else
    const dfloat invRhoCol = invRho;
    const dfloat rhoCol = rhoVar; 

ux_t30 = ux_t30 + 3.0_df*invRhoCol * L_Fx;
uy_t30 = uy_t30 + 3.0_df*invRhoCol * L_Fy;
uz_t30 = uz_t30 + 3.0_df*invRhoCol * L_Fz;


dfloat invRho_mt15 = 3.0_df * invRhoCol / 2.0_df;

m_xx_t45 = (t_omegaVar * m_xx_t45  +   omegaVar_d2 * ux_t30 * ux_t30  + invRho_mt15 * tt_omegaVar * (L_Fx * ux_t30 + L_Fx * ux_t30));
m_yy_t45 = (t_omegaVar * m_yy_t45  +   omegaVar_d2 * uy_t30 * uy_t30  + invRho_mt15 * tt_omegaVar * (L_Fy * uy_t30 + L_Fy * uy_t30));
m_zz_t45 = (t_omegaVar * m_zz_t45  +   omegaVar_d2 * uz_t30 * uz_t30  + invRho_mt15 * tt_omegaVar * (L_Fz * uz_t30 + L_Fz * uz_t30));

m_xy_t90 = (t_omegaVar * m_xy_t90  +   omegaVar * ux_t30 * uy_t30  + tt_omega_t3 * invRhoCol * (L_Fx * uy_t30 + L_Fy * ux_t30));
m_xz_t90 = (t_omegaVar * m_xz_t90  +   omegaVar * ux_t30 * uz_t30  + tt_omega_t3 * invRhoCol * (L_Fx * uz_t30 + L_Fz * ux_t30));
m_yz_t90 = (t_omegaVar * m_yz_t90  +   omegaVar * uy_t30 * uz_t30  + tt_omega_t3 * invRhoCol * (L_Fy * uz_t30 + L_Fz * uy_t30));