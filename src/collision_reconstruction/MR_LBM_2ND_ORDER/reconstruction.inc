#ifdef SHIFTED_HYDRO_POP
#ifndef SHIFTED_HYDRO_POP_ACTIVE
#define SHIFTED_HYDRO_POP_ACTIVE
#endif
{
    // Reconstruct fbar_i directly. Computing f_i first and subtracting the
    // reference equilibrium would reintroduce the FP32 cancellation this
    // representation is intended to avoid.
    const dfloat drho = rhoVar - RHO_0;
    const dfloat pics2_shift = -cs2 * (m_xx_t45 + m_yy_t45 + m_zz_t45);

    pop[ 0] = W0 * (drho + rhoVar * (pics2_shift));
    pop[ 1] = W1 * (drho + rhoVar * (pics2_shift + ux_t30 + m_xx_t45));
    pop[ 2] = W1 * (drho + rhoVar * (pics2_shift - ux_t30 + m_xx_t45));
    pop[ 3] = W1 * (drho + rhoVar * (pics2_shift + uy_t30 + m_yy_t45));
    pop[ 4] = W1 * (drho + rhoVar * (pics2_shift - uy_t30 + m_yy_t45));
    pop[ 5] = W1 * (drho + rhoVar * (pics2_shift + uz_t30 + m_zz_t45));
    pop[ 6] = W1 * (drho + rhoVar * (pics2_shift - uz_t30 + m_zz_t45));
    pop[ 7] = W2 * (drho + rhoVar * (pics2_shift + ux_t30 + uy_t30 + m_xx_t45 + m_yy_t45 + m_xy_t90));
    pop[ 8] = W2 * (drho + rhoVar * (pics2_shift - ux_t30 - uy_t30 + m_xx_t45 + m_yy_t45 + m_xy_t90));
    pop[ 9] = W2 * (drho + rhoVar * (pics2_shift + ux_t30 + uz_t30 + m_xx_t45 + m_zz_t45 + m_xz_t90));
    pop[10] = W2 * (drho + rhoVar * (pics2_shift - ux_t30 - uz_t30 + m_xx_t45 + m_zz_t45 + m_xz_t90));
    pop[11] = W2 * (drho + rhoVar * (pics2_shift + uy_t30 + uz_t30 + m_yy_t45 + m_zz_t45 + m_yz_t90));
    pop[12] = W2 * (drho + rhoVar * (pics2_shift - uy_t30 - uz_t30 + m_yy_t45 + m_zz_t45 + m_yz_t90));
    pop[13] = W2 * (drho + rhoVar * (pics2_shift + ux_t30 - uy_t30 + m_xx_t45 + m_yy_t45 - m_xy_t90));
    pop[14] = W2 * (drho + rhoVar * (pics2_shift - ux_t30 + uy_t30 + m_xx_t45 + m_yy_t45 - m_xy_t90));
    pop[15] = W2 * (drho + rhoVar * (pics2_shift + ux_t30 - uz_t30 + m_xx_t45 + m_zz_t45 - m_xz_t90));
    pop[16] = W2 * (drho + rhoVar * (pics2_shift - ux_t30 + uz_t30 + m_xx_t45 + m_zz_t45 - m_xz_t90));
    pop[17] = W2 * (drho + rhoVar * (pics2_shift + uy_t30 - uz_t30 + m_yy_t45 + m_zz_t45 - m_yz_t90));
    pop[18] = W2 * (drho + rhoVar * (pics2_shift - uy_t30 + uz_t30 + m_yy_t45 + m_zz_t45 - m_yz_t90));
    #ifdef D3Q27
    pop[19] = W3 * (drho + rhoVar * (pics2_shift + ux_t30 + uy_t30 + uz_t30 + m_xx_t45 + m_yy_t45 + m_zz_t45 + (m_xy_t90 + m_xz_t90 + m_yz_t90)));
    pop[20] = W3 * (drho + rhoVar * (pics2_shift - ux_t30 - uy_t30 - uz_t30 + m_xx_t45 + m_yy_t45 + m_zz_t45 + (m_xy_t90 + m_xz_t90 + m_yz_t90)));
    pop[21] = W3 * (drho + rhoVar * (pics2_shift + ux_t30 + uy_t30 - uz_t30 + m_xx_t45 + m_yy_t45 + m_zz_t45 + (m_xy_t90 - m_xz_t90 - m_yz_t90)));
    pop[22] = W3 * (drho + rhoVar * (pics2_shift - ux_t30 - uy_t30 + uz_t30 + m_xx_t45 + m_yy_t45 + m_zz_t45 + (m_xy_t90 - m_xz_t90 - m_yz_t90)));
    pop[23] = W3 * (drho + rhoVar * (pics2_shift + ux_t30 - uy_t30 + uz_t30 + m_xx_t45 + m_yy_t45 + m_zz_t45 - (m_xy_t90 - m_xz_t90 + m_yz_t90)));
    pop[24] = W3 * (drho + rhoVar * (pics2_shift - ux_t30 + uy_t30 - uz_t30 + m_xx_t45 + m_yy_t45 + m_zz_t45 - (m_xy_t90 - m_xz_t90 + m_yz_t90)));
    pop[25] = W3 * (drho + rhoVar * (pics2_shift - ux_t30 + uy_t30 + uz_t30 + m_xx_t45 + m_yy_t45 + m_zz_t45 - (m_xy_t90 + m_xz_t90 - m_yz_t90)));
    pop[26] = W3 * (drho + rhoVar * (pics2_shift + ux_t30 - uy_t30 - uz_t30 + m_xx_t45 + m_yy_t45 + m_zz_t45 - (m_xy_t90 + m_xz_t90 - m_yz_t90)));
    #endif //D3Q27
}
#else
    multiplyTerm = rhoVar * W0;
    pics2 = 1.0_df - cs2 * (m_xx_t45 + m_yy_t45 + m_zz_t45);

    pop[ 0] = multiplyTerm * (pics2);
    multiplyTerm = rhoVar * W1;
    pop[ 1] = multiplyTerm * (pics2 + ux_t30 + m_xx_t45);
    pop[ 2] = multiplyTerm * (pics2 - ux_t30 + m_xx_t45);
    pop[ 3] = multiplyTerm * (pics2 + uy_t30 + m_yy_t45);
    pop[ 4] = multiplyTerm * (pics2 - uy_t30 + m_yy_t45);
    pop[ 5] = multiplyTerm * (pics2 + uz_t30 + m_zz_t45);
    pop[ 6] = multiplyTerm * (pics2 - uz_t30 + m_zz_t45);
    multiplyTerm = rhoVar * W2;
    pop[ 7] = multiplyTerm * (pics2 + ux_t30 + uy_t30 + m_xx_t45 + m_yy_t45 + m_xy_t90);
    pop[ 8] = multiplyTerm * (pics2 - ux_t30 - uy_t30 + m_xx_t45 + m_yy_t45 + m_xy_t90);
    pop[ 9] = multiplyTerm * (pics2 + ux_t30 + uz_t30 + m_xx_t45 + m_zz_t45 + m_xz_t90);
    pop[10] = multiplyTerm * (pics2 - ux_t30 - uz_t30 + m_xx_t45 + m_zz_t45 + m_xz_t90);
    pop[11] = multiplyTerm * (pics2 + uy_t30 + uz_t30 + m_yy_t45 + m_zz_t45 + m_yz_t90);
    pop[12] = multiplyTerm * (pics2 - uy_t30 - uz_t30 + m_yy_t45 + m_zz_t45 + m_yz_t90);
    pop[13] = multiplyTerm * (pics2 + ux_t30 - uy_t30 + m_xx_t45 + m_yy_t45 - m_xy_t90);
    pop[14] = multiplyTerm * (pics2 - ux_t30 + uy_t30 + m_xx_t45 + m_yy_t45 - m_xy_t90);
    pop[15] = multiplyTerm * (pics2 + ux_t30 - uz_t30 + m_xx_t45 + m_zz_t45 - m_xz_t90);
    pop[16] = multiplyTerm * (pics2 - ux_t30 + uz_t30 + m_xx_t45 + m_zz_t45 - m_xz_t90);
    pop[17] = multiplyTerm * (pics2 + uy_t30 - uz_t30 + m_yy_t45 + m_zz_t45 - m_yz_t90);
    pop[18] = multiplyTerm * (pics2 - uy_t30 + uz_t30 + m_yy_t45 + m_zz_t45 - m_yz_t90);
    #ifdef D3Q27
    multiplyTerm = rhoVar * W3;
    pop[19] = multiplyTerm * (pics2 + ux_t30 + uy_t30 + uz_t30 + m_xx_t45 + m_yy_t45 + m_zz_t45 + (m_xy_t90 + m_xz_t90 + m_yz_t90));
    pop[20] = multiplyTerm * (pics2 - ux_t30 - uy_t30 - uz_t30 + m_xx_t45 + m_yy_t45 + m_zz_t45 + (m_xy_t90 + m_xz_t90 + m_yz_t90));
    pop[21] = multiplyTerm * (pics2 + ux_t30 + uy_t30 - uz_t30 + m_xx_t45 + m_yy_t45 + m_zz_t45 + (m_xy_t90 - m_xz_t90 - m_yz_t90));
    pop[22] = multiplyTerm * (pics2 - ux_t30 - uy_t30 + uz_t30 + m_xx_t45 + m_yy_t45 + m_zz_t45 + (m_xy_t90 - m_xz_t90 - m_yz_t90));
    pop[23] = multiplyTerm * (pics2 + ux_t30 - uy_t30 + uz_t30 + m_xx_t45 + m_yy_t45 + m_zz_t45 - (m_xy_t90 - m_xz_t90 + m_yz_t90));
    pop[24] = multiplyTerm * (pics2 - ux_t30 + uy_t30 - uz_t30 + m_xx_t45 + m_yy_t45 + m_zz_t45 - (m_xy_t90 - m_xz_t90 + m_yz_t90));
    pop[25] = multiplyTerm * (pics2 - ux_t30 + uy_t30 + uz_t30 + m_xx_t45 + m_yy_t45 + m_zz_t45 - (m_xy_t90 + m_xz_t90 - m_yz_t90));
    pop[26] = multiplyTerm * (pics2 + ux_t30 - uy_t30 - uz_t30 + m_xx_t45 + m_yy_t45 + m_zz_t45 - (m_xy_t90 + m_xz_t90 - m_yz_t90));
    #endif //D3Q27
#endif //SHIFTED_HYDRO_POP
