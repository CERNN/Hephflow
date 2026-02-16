// 1. Local Density and Gradient Calculation
// rho = rho_g + (rho_l - rho_g) * phi
dfloat phi_normalized = (phiVar - PHI_ONE) / (PHI_TWO - PHI_ONE);
phi_normalized = fmaxf(0.0_df, fminf(1.0_df, phi_normalized)); // clamp to [0,1]

dfloat rho_local = phi_rho_g + phi_drho * phi_normalized;

// nabla rho = (rho_l - rho_g) * nabla phi / (PHI_TWO - PHI_ONE)
dfloat drho_scale = phi_drho / (PHI_TWO - PHI_ONE);
dfloat drhox = drho_scale * dphidx;
dfloat drhoy = drho_scale * dphidy;
dfloat drhoz = drho_scale * dphidz;

// 2. Surface Tension Force (Fs)
dfloat Fs_x = 0.0_df;
dfloat Fs_y = 0.0_df;
dfloat Fs_z = 0.0_df;

#if PHI_FORCE_FS_ENABLED
    #ifdef METHOD_CAHN_HILLIARD
        // -------------------------------------------------
        // Cahn-Hilliard: Korteweg / Potential Force
        // F = mu * grad(phi)
        // -------------------------------------------------
        dfloat mu_val = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M3_MU_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
        
        // In CH, we don't need sigma or curvature directly; they are inside mu
        Fs_x = mu_val * dphidx;
        Fs_y = mu_val * dphidy;
        Fs_z = mu_val * dphidz;
    #else
        // -------------------------------------------------
        // Allen-Cahn: CSF (Continuum Surface Force)
        // F = -sigma * kappa * grad(phi)
        // Note: pnx * mod_grad is equivalent to dphidx
        // -------------------------------------------------
        Fs_x = -phi_sigma * curvature * dphidx;
        Fs_y = -phi_sigma * curvature * dphidy;
        Fs_z = -phi_sigma * curvature * dphidz;
    #endif
#endif

// 3. Pressure Gradient Force (Fp)
// Fp = -cs2 * phi * nabla rho
dfloat Fp_x = 0.0_df;
dfloat Fp_y = 0.0_df;
dfloat Fp_z = 0.0_df;

#if PHI_FORCE_FP_ENABLED
    dfloat p_star = (rhoVar - RHO_0) / (RHO_0); // Non-dimensional pressure
    dfloat Fp_coeff = -cs2 * p_star;
    Fp_x = Fp_coeff * drhox;
    Fp_y = Fp_coeff * drhoy;
    Fp_z = Fp_coeff * drhoz;
#endif

// 4. Non-equilibrium Force (Fn)
// Fn = -(1-omega/2) * Pi_neq . nabla rho
dfloat Fn_x = 0.0_df;
dfloat Fn_y = 0.0_df;
dfloat Fn_z = 0.0_df;

#if PHI_FORCE_FN_ENABLED

    dfloat tau_local = tau_L + (tau_H - tau_L) * phi_normalized;
    dfloat omega_local = 1.0_df / tau_local;
    dfloat tt_omegaVar_local = 1.0_df - 0.5_df * omega_local;

    dfloat inv_F_M_II = 1.0_df / F_M_II_SCALE;
    dfloat inv_F_M_IJ = 1.0_df / F_M_IJ_SCALE;

    // Correcting the stress calculation for local velocity scaling
    dfloat inv_v_scale2 = 1.0_df / (F_M_I_SCALE * F_M_I_SCALE);
    dfloat Pi_xx_neq = (m_xx_t45 * inv_F_M_II - ux_t30 * ux_t30 * inv_v_scale2);
    dfloat Pi_yy_neq = (m_yy_t45 * inv_F_M_II - uy_t30 * uy_t30 * inv_v_scale2);
    dfloat Pi_zz_neq = (m_zz_t45 * inv_F_M_II - uz_t30 * uz_t30 * inv_v_scale2);
    dfloat Pi_xy_neq = (m_xy_t90 * inv_F_M_IJ - ux_t30 * uy_t30 * inv_v_scale2);
    dfloat Pi_xz_neq = (m_xz_t90 * inv_F_M_IJ - ux_t30 * uz_t30 * inv_v_scale2);
    dfloat Pi_yz_neq = (m_yz_t90 * inv_F_M_IJ - uy_t30 * uz_t30 * inv_v_scale2);

    dfloat Fn_coeff = -tt_omegaVar_local;
    Fn_x = Fn_coeff * (Pi_xx_neq * drhox + Pi_xy_neq * drhoy + Pi_xz_neq * drhoz);
    Fn_y = Fn_coeff * (Pi_xy_neq * drhox + Pi_yy_neq * drhoy + Pi_yz_neq * drhoz);
    Fn_z = Fn_coeff * (Pi_xz_neq * drhox + Pi_yz_neq * drhoy + Pi_zz_neq * drhoz);
#endif

// 5. Total Phase Force
dfloat F_phase_x = Fs_x + Fp_x + Fn_x;
dfloat F_phase_y = Fs_y + Fp_y + Fn_y;
dfloat F_phase_z = Fs_z + Fp_z + Fn_z;

#if PHI_FORCE_FB_ENABLED
L_Fx += gravity_vector[0]*g_value * (rho_local - phi_rho_g);;
L_Fy += gravity_vector[1]*g_value * (rho_local - phi_rho_g);;
L_Fz += gravity_vector[2]*g_value * (rho_local - phi_rho_g);;
#endif