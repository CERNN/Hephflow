// rho = rho_g + (rho_l - rho_g)phi
dfloat phi_normalized = (phiVar - PHI_ONE) / (PHI_TWO - PHI_ONE); // just for sake, but should be between 0 and 1 already
phi_normalized = fmaxf(0.0_df, fminf(1.0_df, phi_normalized)); // clamp to [0,1]

dfloat rho_local = phi_rho_g + phi_drho * phi_normalized;

// nabla rho = (rho_l - rho_g)nabla phi/(PHI_TWO - PHI_ONE)
dfloat drho_scale = phi_drho / (PHI_TWO - PHI_ONE);
dfloat drhox = drho_scale * dphidx;
dfloat drhoy = drho_scale * dphidy;
dfloat drhoz = drho_scale * dphidz;

// Fs = sigma * kappa * n|nabla phi|
#if PHI_FORCE_FS_ENABLED
dfloat Fs_x = -phi_sigma * curvature * pnx * mod_grad;
dfloat Fs_y = -phi_sigma * curvature * pny * mod_grad;
dfloat Fs_z = -phi_sigma * curvature * pnz * mod_grad;
#else
dfloat Fs_x = 0.0_df;
dfloat Fs_y = 0.0_df;
dfloat Fs_z = 0.0_df;
#endif

// Fp = -cs2 nabla rho 
#if PHI_FORCE_FP_ENABLED
dfloat Fp_coeff = -cs2;
dfloat Fp_x = Fp_coeff * drhox;
dfloat Fp_y = Fp_coeff * drhoy;
dfloat Fp_z = Fp_coeff * drhoz;
#else
dfloat Fp_x = 0.0_df;
dfloat Fp_y = 0.0_df;
dfloat Fp_z = 0.0_df;
#endif

//Fn = -(1-omega/2) pi_neq . nabla rho
dfloat inv_F_M_II = 1.0_df / F_M_II_SCALE;
dfloat inv_F_M_IJ = 1.0_df / F_M_IJ_SCALE;

dfloat Pi_xx_neq = (m_xx_t45 * inv_F_M_II - ux_t30*ux_t30 / (F_M_I_SCALE*F_M_I_SCALE)) ;
dfloat Pi_yy_neq = (m_yy_t45 * inv_F_M_II - uy_t30*uy_t30 / (F_M_I_SCALE*F_M_I_SCALE)) ;
dfloat Pi_zz_neq = (m_zz_t45 * inv_F_M_II - uz_t30*uz_t30 / (F_M_I_SCALE*F_M_I_SCALE)) ;
dfloat Pi_xy_neq = (m_xy_t90 * inv_F_M_IJ - ux_t30*uy_t30 / (F_M_I_SCALE*F_M_I_SCALE)) ;
dfloat Pi_xz_neq = (m_xz_t90 * inv_F_M_IJ - ux_t30*uz_t30 / (F_M_I_SCALE*F_M_I_SCALE)) ;
dfloat Pi_yz_neq = (m_yz_t90 * inv_F_M_IJ - uy_t30*uz_t30 / (F_M_I_SCALE*F_M_I_SCALE)) ;

#if PHI_FORCE_FN_ENABLED
dfloat Fn_coeff = -tt_omegaVar;
dfloat Fn_x = Fn_coeff * (Pi_xx_neq * drhox + Pi_xy_neq * drhoy + Pi_xz_neq * drhoz);
dfloat Fn_y = Fn_coeff * (Pi_xy_neq * drhox + Pi_yy_neq * drhoy + Pi_yz_neq * drhoz);
dfloat Fn_z = Fn_coeff * (Pi_xz_neq * drhox + Pi_yz_neq * drhoy + Pi_zz_neq * drhoz);
#else
dfloat Fn_x = 0.0_df;
dfloat Fn_y = 0.0_df;
dfloat Fn_z = 0.0_df;
#endif

dfloat F_phase_x = Fs_x + Fp_x + Fn_x;
dfloat F_phase_y = Fs_y + Fp_y + Fn_y;
dfloat F_phase_z = Fs_z + Fp_z + Fn_z;
