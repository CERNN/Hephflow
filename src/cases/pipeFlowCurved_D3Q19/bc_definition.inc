dfloat rho_I;
dfloat inv_rho_I;

dfloat m_xx_I;
dfloat m_xy_I;
dfloat m_xz_I;
dfloat m_yy_I;
dfloat m_yz_I;
dfloat m_zz_I;

dfloat rho;
dfloat inv_rho;

switch (nodeType)
{
case 0b0000010100000001:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = rho_I;

    m_xx_t45 = m_xx_I;
    m_xy_t90 = m_xy_I;
    m_xz_t90 = m_xz_I;
    m_yy_t45 = m_yy_I;
    m_yz_t90 = m_yz_I;
    m_zz_t45 = m_zz_I;

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = 0;
    L_BC_Fy = 0;
    L_BC_Fz = 0;
    #endif //_BC_FORCES 

    break;
case 0b0000010100000010:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = rho_I;

    m_xx_t45 = m_xx_I;
    m_xy_t90 = m_xy_I;
    m_xz_t90 = m_xz_I;
    m_yy_t45 = m_yy_I;
    m_yz_t90 = m_yz_I;
    m_zz_t45 = m_zz_I;

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = 0;
    L_BC_Fy = 0;
    L_BC_Fz = 0;
    #endif //_BC_FORCES 

    break;
case 0b0000010100000011:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (12*(15*rho_I - m_xx_I*rho_I + 2*m_yy_I*rho_I + 6*m_yz_I*rho_I + 2*m_zz_I*rho_I + omegaVar*m_xx_I*rho_I - 2*omegaVar*m_yy_I*rho_I - 6*omegaVar*m_yz_I*rho_I - 2*omegaVar*m_zz_I*rho_I))/(3*omegaVar + 172);

    m_xx_t45 = (792*m_xx_I - 36*m_yy_I - 108*m_yz_I - 36*m_zz_I + 13*omegaVar*m_xx_I + omegaVar*m_yy_I + 3*omegaVar*m_yz_I + omegaVar*m_zz_I - 12)/(54*(2*m_yy_I - m_xx_I + 6*m_yz_I + 2*m_zz_I + omegaVar*m_xx_I - 2*omegaVar*m_yy_I - 6*omegaVar*m_yz_I - 2*omegaVar*m_zz_I + 15));
    m_xy_t90 = (m_xy_I*(3*omegaVar + 172))/(12*(2*m_yy_I - m_xx_I + 6*m_yz_I + 2*m_zz_I + omegaVar*m_xx_I - 2*omegaVar*m_yy_I - 6*omegaVar*m_yz_I - 2*omegaVar*m_zz_I + 15));
    m_xz_t90 = (m_xz_I*(3*omegaVar + 172))/(12*(2*m_yy_I - m_xx_I + 6*m_yz_I + 2*m_zz_I + omegaVar*m_xx_I - 2*omegaVar*m_yy_I - 6*omegaVar*m_yz_I - 2*omegaVar*m_zz_I + 15));
    m_yy_t45 = (1692*m_yy_I - 72*m_xx_I + 432*m_yz_I + 144*m_zz_I + 2*omegaVar*m_xx_I + 23*omegaVar*m_yy_I - 12*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 48)/(108*(2*m_yy_I - m_xx_I + 6*m_yz_I + 2*m_zz_I + omegaVar*m_xx_I - 2*omegaVar*m_yy_I - 6*omegaVar*m_yz_I - 2*omegaVar*m_zz_I + 15));
    m_yz_t90 = (72*m_yy_I - 36*m_xx_I + 732*m_yz_I + 72*m_zz_I + omegaVar*m_xx_I - 2*omegaVar*m_yy_I + 3*omegaVar*m_yz_I - 2*omegaVar*m_zz_I + 24)/(36*(2*m_yy_I - m_xx_I + 6*m_yz_I + 2*m_zz_I + omegaVar*m_xx_I - 2*omegaVar*m_yy_I - 6*omegaVar*m_yz_I - 2*omegaVar*m_zz_I + 15));
    m_zz_t45 = (144*m_yy_I - 72*m_xx_I + 432*m_yz_I + 1692*m_zz_I + 2*omegaVar*m_xx_I - 4*omegaVar*m_yy_I - 12*omegaVar*m_yz_I + 23*omegaVar*m_zz_I + 48)/(108*(2*m_yy_I - m_xx_I + 6*m_yz_I + 2*m_zz_I + omegaVar*m_xx_I - 2*omegaVar*m_yy_I - 6*omegaVar*m_yz_I - 2*omegaVar*m_zz_I + 15));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = 0;
    L_BC_Fy = (m_xx_t45/12 - m_yy_t45/6 - m_yz_t90/2 - m_zz_t45/6 - (omegaVar*m_xx_t45)/24 + (omegaVar*m_yy_t45)/12 + (omegaVar*m_yz_t90)/4 + (omegaVar*m_zz_t45)/12 - 1.0/18.0)*rho + RHO_0/18;
    L_BC_Fz = (m_xx_t45/12 - m_yy_t45/6 - m_yz_t90/2 - m_zz_t45/6 - (omegaVar*m_xx_t45)/24 + (omegaVar*m_yy_t45)/12 + (omegaVar*m_yz_t90)/4 + (omegaVar*m_zz_t45)/12 - 1.0/18.0)*rho + RHO_0/18;
    #endif //_BC_FORCES 

    break;
case 0b0000010100000100:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = rho_I;

    m_xx_t45 = m_xx_I;
    m_xy_t90 = m_xy_I;
    m_xz_t90 = m_xz_I;
    m_yy_t45 = m_yy_I;
    m_yz_t90 = m_yz_I;
    m_zz_t45 = m_zz_I;

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = 0;
    L_BC_Fy = 0;
    L_BC_Fz = 0;
    #endif //_BC_FORCES 

    break;
case 0b0000010100000101:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (12*(15*rho_I + 2*m_xx_I*rho_I + 6*m_xz_I*rho_I - m_yy_I*rho_I + 2*m_zz_I*rho_I - 2*omegaVar*m_xx_I*rho_I - 6*omegaVar*m_xz_I*rho_I + omegaVar*m_yy_I*rho_I - 2*omegaVar*m_zz_I*rho_I))/(3*omegaVar + 172);

    m_xx_t45 = (1692*m_xx_I + 432*m_xz_I - 72*m_yy_I + 144*m_zz_I + 23*omegaVar*m_xx_I - 12*omegaVar*m_xz_I + 2*omegaVar*m_yy_I - 4*omegaVar*m_zz_I + 48)/(108*(2*m_xx_I + 6*m_xz_I - m_yy_I + 2*m_zz_I - 2*omegaVar*m_xx_I - 6*omegaVar*m_xz_I + omegaVar*m_yy_I - 2*omegaVar*m_zz_I + 15));
    m_xy_t90 = (m_xy_I*(3*omegaVar + 172))/(12*(2*m_xx_I + 6*m_xz_I - m_yy_I + 2*m_zz_I - 2*omegaVar*m_xx_I - 6*omegaVar*m_xz_I + omegaVar*m_yy_I - 2*omegaVar*m_zz_I + 15));
    m_xz_t90 = (72*m_xx_I + 732*m_xz_I - 36*m_yy_I + 72*m_zz_I - 2*omegaVar*m_xx_I + 3*omegaVar*m_xz_I + omegaVar*m_yy_I - 2*omegaVar*m_zz_I + 24)/(36*(2*m_xx_I + 6*m_xz_I - m_yy_I + 2*m_zz_I - 2*omegaVar*m_xx_I - 6*omegaVar*m_xz_I + omegaVar*m_yy_I - 2*omegaVar*m_zz_I + 15));
    m_yy_t45 = (792*m_yy_I - 108*m_xz_I - 36*m_xx_I - 36*m_zz_I + omegaVar*m_xx_I + 3*omegaVar*m_xz_I + 13*omegaVar*m_yy_I + omegaVar*m_zz_I - 12)/(54*(2*m_xx_I + 6*m_xz_I - m_yy_I + 2*m_zz_I - 2*omegaVar*m_xx_I - 6*omegaVar*m_xz_I + omegaVar*m_yy_I - 2*omegaVar*m_zz_I + 15));
    m_yz_t90 = (m_yz_I*(3*omegaVar + 172))/(12*(2*m_xx_I + 6*m_xz_I - m_yy_I + 2*m_zz_I - 2*omegaVar*m_xx_I - 6*omegaVar*m_xz_I + omegaVar*m_yy_I - 2*omegaVar*m_zz_I + 15));
    m_zz_t45 = (144*m_xx_I + 432*m_xz_I - 72*m_yy_I + 1692*m_zz_I - 4*omegaVar*m_xx_I - 12*omegaVar*m_xz_I + 2*omegaVar*m_yy_I + 23*omegaVar*m_zz_I + 48)/(108*(2*m_xx_I + 6*m_xz_I - m_yy_I + 2*m_zz_I - 2*omegaVar*m_xx_I - 6*omegaVar*m_xz_I + omegaVar*m_yy_I - 2*omegaVar*m_zz_I + 15));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = (m_yy_t45/12 - m_xz_t90/2 - m_xx_t45/6 - m_zz_t45/6 + (omegaVar*m_xx_t45)/12 + (omegaVar*m_xz_t90)/4 - (omegaVar*m_yy_t45)/24 + (omegaVar*m_zz_t45)/12 - 1.0/18.0)*rho + RHO_0/18;
    L_BC_Fy = 0;
    L_BC_Fz = (m_yy_t45/12 - m_xz_t90/2 - m_xx_t45/6 - m_zz_t45/6 + (omegaVar*m_xx_t45)/12 + (omegaVar*m_xz_t90)/4 - (omegaVar*m_yy_t45)/24 + (omegaVar*m_zz_t45)/12 - 1.0/18.0)*rho + RHO_0/18;
    #endif //_BC_FORCES 

    break;
case 0b0000010100000110:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = rho_I;

    m_xx_t45 = m_xx_I;
    m_xy_t90 = m_xy_I;
    m_xz_t90 = m_xz_I;
    m_yy_t45 = m_yy_I;
    m_yz_t90 = m_yz_I;
    m_zz_t45 = m_zz_I;

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = 0;
    L_BC_Fy = 0;
    L_BC_Fz = 0;
    #endif //_BC_FORCES 

    break;
case 0b0000010100000111:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[10] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (6*(15*rho_I + m_xx_I*rho_I + 6*m_xz_I*rho_I + m_yy_I*rho_I + 6*m_yz_I*rho_I + 4*m_zz_I*rho_I - omegaVar*m_xx_I*rho_I - 6*omegaVar*m_xz_I*rho_I - omegaVar*m_yy_I*rho_I - 6*omegaVar*m_yz_I*rho_I - 4*omegaVar*m_zz_I*rho_I))/(3*omegaVar + 82);

    m_xx_t45 = (2052*m_xx_I + 504*m_xz_I - 162*m_yy_I - 234*m_yz_I + 90*m_zz_I + 73*omegaVar*m_xx_I + 6*omegaVar*m_xz_I - 8*omegaVar*m_yy_I - 21*omegaVar*m_yz_I - 5*omegaVar*m_zz_I + 30)/(135*(m_xx_I + 6*m_xz_I + m_yy_I + 6*m_yz_I + 4*m_zz_I - omegaVar*m_xx_I - 6*omegaVar*m_xz_I - omegaVar*m_yy_I - 6*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 15));
    m_xy_t90 = (m_xy_I*(3*omegaVar + 82))/(6*(m_xx_I + 6*m_xz_I + m_yy_I + 6*m_yz_I + 4*m_zz_I - omegaVar*m_xx_I - 6*omegaVar*m_xz_I - omegaVar*m_yy_I - 6*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 15));
    m_xz_t90 = (168*m_xx_I + 1746*m_xz_I - 78*m_yy_I + 24*m_yz_I + 180*m_zz_I + 2*omegaVar*m_xx_I + 39*omegaVar*m_xz_I - 7*omegaVar*m_yy_I - 24*omegaVar*m_yz_I - 10*omegaVar*m_zz_I + 60)/(90*(m_xx_I + 6*m_xz_I + m_yy_I + 6*m_yz_I + 4*m_zz_I - omegaVar*m_xx_I - 6*omegaVar*m_xz_I - omegaVar*m_yy_I - 6*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 15));
    m_yy_t45 = (2052*m_yy_I - 234*m_xz_I - 162*m_xx_I + 504*m_yz_I + 90*m_zz_I - 8*omegaVar*m_xx_I - 21*omegaVar*m_xz_I + 73*omegaVar*m_yy_I + 6*omegaVar*m_yz_I - 5*omegaVar*m_zz_I + 30)/(135*(m_xx_I + 6*m_xz_I + m_yy_I + 6*m_yz_I + 4*m_zz_I - omegaVar*m_xx_I - 6*omegaVar*m_xz_I - omegaVar*m_yy_I - 6*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 15));
    m_yz_t90 = (24*m_xz_I - 78*m_xx_I + 168*m_yy_I + 1746*m_yz_I + 180*m_zz_I - 7*omegaVar*m_xx_I - 24*omegaVar*m_xz_I + 2*omegaVar*m_yy_I + 39*omegaVar*m_yz_I - 10*omegaVar*m_zz_I + 60)/(90*(m_xx_I + 6*m_xz_I + m_yy_I + 6*m_yz_I + 4*m_zz_I - omegaVar*m_xx_I - 6*omegaVar*m_xz_I - omegaVar*m_yy_I - 6*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 15));
    m_zz_t45 = (36*m_xx_I + 216*m_xz_I + 36*m_yy_I + 216*m_yz_I + 882*m_zz_I - 2*omegaVar*m_xx_I - 12*omegaVar*m_xz_I - 2*omegaVar*m_yy_I - 12*omegaVar*m_yz_I + 19*omegaVar*m_zz_I + 48)/(54*(m_xx_I + 6*m_xz_I + m_yy_I + 6*m_yz_I + 4*m_zz_I - omegaVar*m_xx_I - 6*omegaVar*m_xz_I - omegaVar*m_yy_I - 6*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 15));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = (m_yy_t45/12 - m_xz_t90/2 - m_xx_t45/6 - m_zz_t45/6 + (omegaVar*m_xx_t45)/12 + (omegaVar*m_xz_t90)/4 - (omegaVar*m_yy_t45)/24 + (omegaVar*m_zz_t45)/12 - 1.0/18.0)*rho + RHO_0/18;
    L_BC_Fy = (m_xx_t45/12 - m_yy_t45/6 - m_yz_t90/2 - m_zz_t45/6 - (omegaVar*m_xx_t45)/24 + (omegaVar*m_yy_t45)/12 + (omegaVar*m_yz_t90)/4 + (omegaVar*m_zz_t45)/12 - 1.0/18.0)*rho + RHO_0/18;
    L_BC_Fz = ((omegaVar*m_xx_t45)/24 - m_xz_t90/2 - m_yy_t45/12 - m_yz_t90/2 - m_zz_t45/3 - m_xx_t45/12 + (omegaVar*m_xz_t90)/4 + (omegaVar*m_yy_t45)/24 + (omegaVar*m_yz_t90)/4 + (omegaVar*m_zz_t45)/6 - 1.0/9.0)*rho + RHO_0/9;
    #endif //_BC_FORCES 

    break;
case 0b0000010100001000:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = rho_I;

    m_xx_t45 = m_xx_I;
    m_xy_t90 = m_xy_I;
    m_xz_t90 = m_xz_I;
    m_yy_t45 = m_yy_I;
    m_yz_t90 = m_yz_I;
    m_zz_t45 = m_zz_I;

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = 0;
    L_BC_Fy = 0;
    L_BC_Fz = 0;
    #endif //_BC_FORCES 

    break;
case 0b0000010100001001:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = rho_I;

    m_xx_t45 = m_xx_I;
    m_xy_t90 = m_xy_I;
    m_xz_t90 = m_xz_I;
    m_yy_t45 = m_yy_I;
    m_yz_t90 = m_yz_I;
    m_zz_t45 = m_zz_I;

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = 0;
    L_BC_Fy = 0;
    L_BC_Fz = 0;
    #endif //_BC_FORCES 

    break;
case 0b0000010100001010:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (12*(15*rho_I + 2*m_xx_I*rho_I - 6*m_xz_I*rho_I - m_yy_I*rho_I + 2*m_zz_I*rho_I - 2*omegaVar*m_xx_I*rho_I + 6*omegaVar*m_xz_I*rho_I + omegaVar*m_yy_I*rho_I - 2*omegaVar*m_zz_I*rho_I))/(3*omegaVar + 172);

    m_xx_t45 = (1692*m_xx_I - 432*m_xz_I - 72*m_yy_I + 144*m_zz_I + 23*omegaVar*m_xx_I + 12*omegaVar*m_xz_I + 2*omegaVar*m_yy_I - 4*omegaVar*m_zz_I + 48)/(108*(2*m_xx_I - 6*m_xz_I - m_yy_I + 2*m_zz_I - 2*omegaVar*m_xx_I + 6*omegaVar*m_xz_I + omegaVar*m_yy_I - 2*omegaVar*m_zz_I + 15));
    m_xy_t90 = (m_xy_I*(3*omegaVar + 172))/(12*(2*m_xx_I - 6*m_xz_I - m_yy_I + 2*m_zz_I - 2*omegaVar*m_xx_I + 6*omegaVar*m_xz_I + omegaVar*m_yy_I - 2*omegaVar*m_zz_I + 15));
    m_xz_t90 = (732*m_xz_I - 72*m_xx_I + 36*m_yy_I - 72*m_zz_I + 2*omegaVar*m_xx_I + 3*omegaVar*m_xz_I - omegaVar*m_yy_I + 2*omegaVar*m_zz_I - 24)/(36*(2*m_xx_I - 6*m_xz_I - m_yy_I + 2*m_zz_I - 2*omegaVar*m_xx_I + 6*omegaVar*m_xz_I + omegaVar*m_yy_I - 2*omegaVar*m_zz_I + 15));
    m_yy_t45 = (108*m_xz_I - 36*m_xx_I + 792*m_yy_I - 36*m_zz_I + omegaVar*m_xx_I - 3*omegaVar*m_xz_I + 13*omegaVar*m_yy_I + omegaVar*m_zz_I - 12)/(54*(2*m_xx_I - 6*m_xz_I - m_yy_I + 2*m_zz_I - 2*omegaVar*m_xx_I + 6*omegaVar*m_xz_I + omegaVar*m_yy_I - 2*omegaVar*m_zz_I + 15));
    m_yz_t90 = (m_yz_I*(3*omegaVar + 172))/(12*(2*m_xx_I - 6*m_xz_I - m_yy_I + 2*m_zz_I - 2*omegaVar*m_xx_I + 6*omegaVar*m_xz_I + omegaVar*m_yy_I - 2*omegaVar*m_zz_I + 15));
    m_zz_t45 = (144*m_xx_I - 432*m_xz_I - 72*m_yy_I + 1692*m_zz_I - 4*omegaVar*m_xx_I + 12*omegaVar*m_xz_I + 2*omegaVar*m_yy_I + 23*omegaVar*m_zz_I + 48)/(108*(2*m_xx_I - 6*m_xz_I - m_yy_I + 2*m_zz_I - 2*omegaVar*m_xx_I + 6*omegaVar*m_xz_I + omegaVar*m_yy_I - 2*omegaVar*m_zz_I + 15));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = (m_xx_t45/6 - m_xz_t90/2 - m_yy_t45/12 + m_zz_t45/6 - (omegaVar*m_xx_t45)/12 + (omegaVar*m_xz_t90)/4 + (omegaVar*m_yy_t45)/24 - (omegaVar*m_zz_t45)/12 + 1.0/18.0)*rho - RHO_0/18;
    L_BC_Fy = 0;
    L_BC_Fz = (m_xz_t90/2 - m_xx_t45/6 + m_yy_t45/12 - m_zz_t45/6 + (omegaVar*m_xx_t45)/12 - (omegaVar*m_xz_t90)/4 - (omegaVar*m_yy_t45)/24 + (omegaVar*m_zz_t45)/12 - 1.0/18.0)*rho + RHO_0/18;
    #endif //_BC_FORCES 

    break;
case 0b0000010100001011:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[12] + pop[13] + pop[14] + pop[15] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (6*(15*rho_I + m_xx_I*rho_I - 6*m_xz_I*rho_I + m_yy_I*rho_I + 6*m_yz_I*rho_I + 4*m_zz_I*rho_I - omegaVar*m_xx_I*rho_I + 6*omegaVar*m_xz_I*rho_I - omegaVar*m_yy_I*rho_I - 6*omegaVar*m_yz_I*rho_I - 4*omegaVar*m_zz_I*rho_I))/(3*omegaVar + 82);

    m_xx_t45 = -(504*m_xz_I - 2052*m_xx_I + 162*m_yy_I + 234*m_yz_I - 90*m_zz_I - 73*omegaVar*m_xx_I + 6*omegaVar*m_xz_I + 8*omegaVar*m_yy_I + 21*omegaVar*m_yz_I + 5*omegaVar*m_zz_I - 30)/(135*(m_xx_I - 6*m_xz_I + m_yy_I + 6*m_yz_I + 4*m_zz_I - omegaVar*m_xx_I + 6*omegaVar*m_xz_I - omegaVar*m_yy_I - 6*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 15));
    m_xy_t90 = (m_xy_I*(3*omegaVar + 82))/(6*(m_xx_I - 6*m_xz_I + m_yy_I + 6*m_yz_I + 4*m_zz_I - omegaVar*m_xx_I + 6*omegaVar*m_xz_I - omegaVar*m_yy_I - 6*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 15));
    m_xz_t90 = (1746*m_xz_I - 168*m_xx_I + 78*m_yy_I - 24*m_yz_I - 180*m_zz_I - 2*omegaVar*m_xx_I + 39*omegaVar*m_xz_I + 7*omegaVar*m_yy_I + 24*omegaVar*m_yz_I + 10*omegaVar*m_zz_I - 60)/(90*(m_xx_I - 6*m_xz_I + m_yy_I + 6*m_yz_I + 4*m_zz_I - omegaVar*m_xx_I + 6*omegaVar*m_xz_I - omegaVar*m_yy_I - 6*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 15));
    m_yy_t45 = (234*m_xz_I - 162*m_xx_I + 2052*m_yy_I + 504*m_yz_I + 90*m_zz_I - 8*omegaVar*m_xx_I + 21*omegaVar*m_xz_I + 73*omegaVar*m_yy_I + 6*omegaVar*m_yz_I - 5*omegaVar*m_zz_I + 30)/(135*(m_xx_I - 6*m_xz_I + m_yy_I + 6*m_yz_I + 4*m_zz_I - omegaVar*m_xx_I + 6*omegaVar*m_xz_I - omegaVar*m_yy_I - 6*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 15));
    m_yz_t90 = (168*m_yy_I - 24*m_xz_I - 78*m_xx_I + 1746*m_yz_I + 180*m_zz_I - 7*omegaVar*m_xx_I + 24*omegaVar*m_xz_I + 2*omegaVar*m_yy_I + 39*omegaVar*m_yz_I - 10*omegaVar*m_zz_I + 60)/(90*(m_xx_I - 6*m_xz_I + m_yy_I + 6*m_yz_I + 4*m_zz_I - omegaVar*m_xx_I + 6*omegaVar*m_xz_I - omegaVar*m_yy_I - 6*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 15));
    m_zz_t45 = (36*m_xx_I - 216*m_xz_I + 36*m_yy_I + 216*m_yz_I + 882*m_zz_I - 2*omegaVar*m_xx_I + 12*omegaVar*m_xz_I - 2*omegaVar*m_yy_I - 12*omegaVar*m_yz_I + 19*omegaVar*m_zz_I + 48)/(54*(m_xx_I - 6*m_xz_I + m_yy_I + 6*m_yz_I + 4*m_zz_I - omegaVar*m_xx_I + 6*omegaVar*m_xz_I - omegaVar*m_yy_I - 6*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 15));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = (m_xx_t45/6 - m_xz_t90/2 - m_yy_t45/12 + m_zz_t45/6 - (omegaVar*m_xx_t45)/12 + (omegaVar*m_xz_t90)/4 + (omegaVar*m_yy_t45)/24 - (omegaVar*m_zz_t45)/12 + 1.0/18.0)*rho - RHO_0/18;
    L_BC_Fy = (m_xx_t45/12 - m_yy_t45/6 - m_yz_t90/2 - m_zz_t45/6 - (omegaVar*m_xx_t45)/24 + (omegaVar*m_yy_t45)/12 + (omegaVar*m_yz_t90)/4 + (omegaVar*m_zz_t45)/12 - 1.0/18.0)*rho + RHO_0/18;
    L_BC_Fz = (m_xz_t90/2 - m_xx_t45/12 - m_yy_t45/12 - m_yz_t90/2 - m_zz_t45/3 + (omegaVar*m_xx_t45)/24 - (omegaVar*m_xz_t90)/4 + (omegaVar*m_yy_t45)/24 + (omegaVar*m_yz_t90)/4 + (omegaVar*m_zz_t45)/6 - 1.0/9.0)*rho + RHO_0/9;
    #endif //_BC_FORCES 

    break;
case 0b0000010100001100:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17]);

    rho = (12*(15*rho_I - m_xx_I*rho_I + 2*m_yy_I*rho_I - 6*m_yz_I*rho_I + 2*m_zz_I*rho_I + omegaVar*m_xx_I*rho_I - 2*omegaVar*m_yy_I*rho_I + 6*omegaVar*m_yz_I*rho_I - 2*omegaVar*m_zz_I*rho_I))/(3*omegaVar + 172);

    m_xx_t45 = (792*m_xx_I - 36*m_yy_I + 108*m_yz_I - 36*m_zz_I + 13*omegaVar*m_xx_I + omegaVar*m_yy_I - 3*omegaVar*m_yz_I + omegaVar*m_zz_I - 12)/(54*(2*m_yy_I - m_xx_I - 6*m_yz_I + 2*m_zz_I + omegaVar*m_xx_I - 2*omegaVar*m_yy_I + 6*omegaVar*m_yz_I - 2*omegaVar*m_zz_I + 15));
    m_xy_t90 = (m_xy_I*(3*omegaVar + 172))/(12*(2*m_yy_I - m_xx_I - 6*m_yz_I + 2*m_zz_I + omegaVar*m_xx_I - 2*omegaVar*m_yy_I + 6*omegaVar*m_yz_I - 2*omegaVar*m_zz_I + 15));
    m_xz_t90 = (m_xz_I*(3*omegaVar + 172))/(12*(2*m_yy_I - m_xx_I - 6*m_yz_I + 2*m_zz_I + omegaVar*m_xx_I - 2*omegaVar*m_yy_I + 6*omegaVar*m_yz_I - 2*omegaVar*m_zz_I + 15));
    m_yy_t45 = (1692*m_yy_I - 72*m_xx_I - 432*m_yz_I + 144*m_zz_I + 2*omegaVar*m_xx_I + 23*omegaVar*m_yy_I + 12*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 48)/(108*(2*m_yy_I - m_xx_I - 6*m_yz_I + 2*m_zz_I + omegaVar*m_xx_I - 2*omegaVar*m_yy_I + 6*omegaVar*m_yz_I - 2*omegaVar*m_zz_I + 15));
    m_yz_t90 = (36*m_xx_I - 72*m_yy_I + 732*m_yz_I - 72*m_zz_I - omegaVar*m_xx_I + 2*omegaVar*m_yy_I + 3*omegaVar*m_yz_I + 2*omegaVar*m_zz_I - 24)/(36*(2*m_yy_I - m_xx_I - 6*m_yz_I + 2*m_zz_I + omegaVar*m_xx_I - 2*omegaVar*m_yy_I + 6*omegaVar*m_yz_I - 2*omegaVar*m_zz_I + 15));
    m_zz_t45 = (144*m_yy_I - 72*m_xx_I - 432*m_yz_I + 1692*m_zz_I + 2*omegaVar*m_xx_I - 4*omegaVar*m_yy_I + 12*omegaVar*m_yz_I + 23*omegaVar*m_zz_I + 48)/(108*(2*m_yy_I - m_xx_I - 6*m_yz_I + 2*m_zz_I + omegaVar*m_xx_I - 2*omegaVar*m_yy_I + 6*omegaVar*m_yz_I - 2*omegaVar*m_zz_I + 15));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = 0;
    L_BC_Fy = (m_yy_t45/6 - m_xx_t45/12 - m_yz_t90/2 + m_zz_t45/6 + (omegaVar*m_xx_t45)/24 - (omegaVar*m_yy_t45)/12 + (omegaVar*m_yz_t90)/4 - (omegaVar*m_zz_t45)/12 + 1.0/18.0)*rho - RHO_0/18;
    L_BC_Fz = (m_xx_t45/12 - m_yy_t45/6 + m_yz_t90/2 - m_zz_t45/6 - (omegaVar*m_xx_t45)/24 + (omegaVar*m_yy_t45)/12 - (omegaVar*m_yz_t90)/4 + (omegaVar*m_zz_t45)/12 - 1.0/18.0)*rho + RHO_0/18;
    #endif //_BC_FORCES 

    break;
case 0b0000010100001101:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17]);

    rho = (6*(15*rho_I + m_xx_I*rho_I + 6*m_xz_I*rho_I + m_yy_I*rho_I - 6*m_yz_I*rho_I + 4*m_zz_I*rho_I - omegaVar*m_xx_I*rho_I - 6*omegaVar*m_xz_I*rho_I - omegaVar*m_yy_I*rho_I + 6*omegaVar*m_yz_I*rho_I - 4*omegaVar*m_zz_I*rho_I))/(3*omegaVar + 82);

    m_xx_t45 = (2052*m_xx_I + 504*m_xz_I - 162*m_yy_I + 234*m_yz_I + 90*m_zz_I + 73*omegaVar*m_xx_I + 6*omegaVar*m_xz_I - 8*omegaVar*m_yy_I + 21*omegaVar*m_yz_I - 5*omegaVar*m_zz_I + 30)/(135*(m_xx_I + 6*m_xz_I + m_yy_I - 6*m_yz_I + 4*m_zz_I - omegaVar*m_xx_I - 6*omegaVar*m_xz_I - omegaVar*m_yy_I + 6*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 15));
    m_xy_t90 = (m_xy_I*(3*omegaVar + 82))/(6*(m_xx_I + 6*m_xz_I + m_yy_I - 6*m_yz_I + 4*m_zz_I - omegaVar*m_xx_I - 6*omegaVar*m_xz_I - omegaVar*m_yy_I + 6*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 15));
    m_xz_t90 = (168*m_xx_I + 1746*m_xz_I - 78*m_yy_I - 24*m_yz_I + 180*m_zz_I + 2*omegaVar*m_xx_I + 39*omegaVar*m_xz_I - 7*omegaVar*m_yy_I + 24*omegaVar*m_yz_I - 10*omegaVar*m_zz_I + 60)/(90*(m_xx_I + 6*m_xz_I + m_yy_I - 6*m_yz_I + 4*m_zz_I - omegaVar*m_xx_I - 6*omegaVar*m_xz_I - omegaVar*m_yy_I + 6*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 15));
    m_yy_t45 = -(162*m_xx_I + 234*m_xz_I - 2052*m_yy_I + 504*m_yz_I - 90*m_zz_I + 8*omegaVar*m_xx_I + 21*omegaVar*m_xz_I - 73*omegaVar*m_yy_I + 6*omegaVar*m_yz_I + 5*omegaVar*m_zz_I - 30)/(135*(m_xx_I + 6*m_xz_I + m_yy_I - 6*m_yz_I + 4*m_zz_I - omegaVar*m_xx_I - 6*omegaVar*m_xz_I - omegaVar*m_yy_I + 6*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 15));
    m_yz_t90 = (78*m_xx_I - 24*m_xz_I - 168*m_yy_I + 1746*m_yz_I - 180*m_zz_I + 7*omegaVar*m_xx_I + 24*omegaVar*m_xz_I - 2*omegaVar*m_yy_I + 39*omegaVar*m_yz_I + 10*omegaVar*m_zz_I - 60)/(90*(m_xx_I + 6*m_xz_I + m_yy_I - 6*m_yz_I + 4*m_zz_I - omegaVar*m_xx_I - 6*omegaVar*m_xz_I - omegaVar*m_yy_I + 6*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 15));
    m_zz_t45 = (36*m_xx_I + 216*m_xz_I + 36*m_yy_I - 216*m_yz_I + 882*m_zz_I - 2*omegaVar*m_xx_I - 12*omegaVar*m_xz_I - 2*omegaVar*m_yy_I + 12*omegaVar*m_yz_I + 19*omegaVar*m_zz_I + 48)/(54*(m_xx_I + 6*m_xz_I + m_yy_I - 6*m_yz_I + 4*m_zz_I - omegaVar*m_xx_I - 6*omegaVar*m_xz_I - omegaVar*m_yy_I + 6*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 15));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = (m_yy_t45/12 - m_xz_t90/2 - m_xx_t45/6 - m_zz_t45/6 + (omegaVar*m_xx_t45)/12 + (omegaVar*m_xz_t90)/4 - (omegaVar*m_yy_t45)/24 + (omegaVar*m_zz_t45)/12 - 1.0/18.0)*rho + RHO_0/18;
    L_BC_Fy = (m_yy_t45/6 - m_xx_t45/12 - m_yz_t90/2 + m_zz_t45/6 + (omegaVar*m_xx_t45)/24 - (omegaVar*m_yy_t45)/12 + (omegaVar*m_yz_t90)/4 - (omegaVar*m_zz_t45)/12 + 1.0/18.0)*rho - RHO_0/18;
    L_BC_Fz = (m_yz_t90/2 - m_xz_t90/2 - m_yy_t45/12 - m_xx_t45/12 - m_zz_t45/3 + (omegaVar*m_xx_t45)/24 + (omegaVar*m_xz_t90)/4 + (omegaVar*m_yy_t45)/24 - (omegaVar*m_yz_t90)/4 + (omegaVar*m_zz_t45)/6 - 1.0/9.0)*rho + RHO_0/9;
    #endif //_BC_FORCES 

    break;
case 0b0000010100001110:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[17];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] - (1.0/3.0)*pop[17]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] + (2.0/3.0)*pop[17]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[17]);

    rho = (6*(15*rho_I + m_xx_I*rho_I - 6*m_xz_I*rho_I + m_yy_I*rho_I - 6*m_yz_I*rho_I + 4*m_zz_I*rho_I - omegaVar*m_xx_I*rho_I + 6*omegaVar*m_xz_I*rho_I - omegaVar*m_yy_I*rho_I + 6*omegaVar*m_yz_I*rho_I - 4*omegaVar*m_zz_I*rho_I))/(3*omegaVar + 82);

    m_xx_t45 = (2052*m_xx_I - 504*m_xz_I - 162*m_yy_I + 234*m_yz_I + 90*m_zz_I + 73*omegaVar*m_xx_I - 6*omegaVar*m_xz_I - 8*omegaVar*m_yy_I + 21*omegaVar*m_yz_I - 5*omegaVar*m_zz_I + 30)/(135*(m_xx_I - 6*m_xz_I + m_yy_I - 6*m_yz_I + 4*m_zz_I - omegaVar*m_xx_I + 6*omegaVar*m_xz_I - omegaVar*m_yy_I + 6*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 15));
    m_xy_t90 = (m_xy_I*(3*omegaVar + 82))/(6*(m_xx_I - 6*m_xz_I + m_yy_I - 6*m_yz_I + 4*m_zz_I - omegaVar*m_xx_I + 6*omegaVar*m_xz_I - omegaVar*m_yy_I + 6*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 15));
    m_xz_t90 = (1746*m_xz_I - 168*m_xx_I + 78*m_yy_I + 24*m_yz_I - 180*m_zz_I - 2*omegaVar*m_xx_I + 39*omegaVar*m_xz_I + 7*omegaVar*m_yy_I - 24*omegaVar*m_yz_I + 10*omegaVar*m_zz_I - 60)/(90*(m_xx_I - 6*m_xz_I + m_yy_I - 6*m_yz_I + 4*m_zz_I - omegaVar*m_xx_I + 6*omegaVar*m_xz_I - omegaVar*m_yy_I + 6*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 15));
    m_yy_t45 = (234*m_xz_I - 162*m_xx_I + 2052*m_yy_I - 504*m_yz_I + 90*m_zz_I - 8*omegaVar*m_xx_I + 21*omegaVar*m_xz_I + 73*omegaVar*m_yy_I - 6*omegaVar*m_yz_I - 5*omegaVar*m_zz_I + 30)/(135*(m_xx_I - 6*m_xz_I + m_yy_I - 6*m_yz_I + 4*m_zz_I - omegaVar*m_xx_I + 6*omegaVar*m_xz_I - omegaVar*m_yy_I + 6*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 15));
    m_yz_t90 = (78*m_xx_I + 24*m_xz_I - 168*m_yy_I + 1746*m_yz_I - 180*m_zz_I + 7*omegaVar*m_xx_I - 24*omegaVar*m_xz_I - 2*omegaVar*m_yy_I + 39*omegaVar*m_yz_I + 10*omegaVar*m_zz_I - 60)/(90*(m_xx_I - 6*m_xz_I + m_yy_I - 6*m_yz_I + 4*m_zz_I - omegaVar*m_xx_I + 6*omegaVar*m_xz_I - omegaVar*m_yy_I + 6*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 15));
    m_zz_t45 = (36*m_xx_I - 216*m_xz_I + 36*m_yy_I - 216*m_yz_I + 882*m_zz_I - 2*omegaVar*m_xx_I + 12*omegaVar*m_xz_I - 2*omegaVar*m_yy_I + 12*omegaVar*m_yz_I + 19*omegaVar*m_zz_I + 48)/(54*(m_xx_I - 6*m_xz_I + m_yy_I - 6*m_yz_I + 4*m_zz_I - omegaVar*m_xx_I + 6*omegaVar*m_xz_I - omegaVar*m_yy_I + 6*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 15));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = (m_xx_t45/6 - m_xz_t90/2 - m_yy_t45/12 + m_zz_t45/6 - (omegaVar*m_xx_t45)/12 + (omegaVar*m_xz_t90)/4 + (omegaVar*m_yy_t45)/24 - (omegaVar*m_zz_t45)/12 + 1.0/18.0)*rho - RHO_0/18;
    L_BC_Fy = (m_yy_t45/6 - m_xx_t45/12 - m_yz_t90/2 + m_zz_t45/6 + (omegaVar*m_xx_t45)/24 - (omegaVar*m_yy_t45)/12 + (omegaVar*m_yz_t90)/4 - (omegaVar*m_zz_t45)/12 + 1.0/18.0)*rho - RHO_0/18;
    L_BC_Fz = (m_xz_t90/2 - m_xx_t45/12 - m_yy_t45/12 + m_yz_t90/2 - m_zz_t45/3 + (omegaVar*m_xx_t45)/24 - (omegaVar*m_xz_t90)/4 + (omegaVar*m_yy_t45)/24 - (omegaVar*m_yz_t90)/4 + (omegaVar*m_zz_t45)/6 - 1.0/9.0)*rho + RHO_0/9;
    #endif //_BC_FORCES 

    break;
case 0b0000010100001111:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[6] + pop[7] + pop[8] + pop[10] + pop[12] + pop[13] + pop[14] + pop[15] + pop[17];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] - (1.0/3.0)*pop[17]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[10] - pop[15]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] + (2.0/3.0)*pop[17]);
    m_yz_I = inv_rho_I * (pop[12] - pop[17]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[17]);

    rho = (3*rho_I*(3*m_zz_I - 3*omegaVar*m_zz_I + 4))/(omegaVar + 9);

    m_xx_t45 = (4*(omegaVar + 9)*(10*m_xx_I - m_yy_I))/(99*(3*m_zz_I - 3*omegaVar*m_zz_I + 4));
    m_xy_t90 = (m_xy_I*(omegaVar + 9))/(3*(3*m_zz_I - 3*omegaVar*m_zz_I + 4));
    m_xz_t90 = (2*m_xz_I*(omegaVar + 9))/(3*(3*m_zz_I - 3*omegaVar*m_zz_I + 4));
    m_yy_t45 = -(4*(m_xx_I - 10*m_yy_I)*(omegaVar + 9))/(99*(3*m_zz_I - 3*omegaVar*m_zz_I + 4));
    m_yz_t90 = (2*m_yz_I*(omegaVar + 9))/(3*(3*m_zz_I - 3*omegaVar*m_zz_I + 4));
    m_zz_t45 = (15*m_zz_I + 2)/(3*(3*m_zz_I - 3*omegaVar*m_zz_I + 4));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = (m_xz_t90*rho*(omegaVar - 2))/2;
    L_BC_Fy = (m_yz_t90*rho*(omegaVar - 2))/2;
    L_BC_Fz = ((omegaVar*m_zz_t45)/2 - m_zz_t45 - 1.0/3.0)*rho + RHO_0/3;
    #endif //_BC_FORCES 

    break;
case 0b0000010100010000:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = rho_I;

    m_xx_t45 = m_xx_I;
    m_xy_t90 = m_xy_I;
    m_xz_t90 = m_xz_I;
    m_yy_t45 = m_yy_I;
    m_yz_t90 = m_yz_I;
    m_zz_t45 = m_zz_I;

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = 0;
    L_BC_Fy = 0;
    L_BC_Fz = 0;
    #endif //_BC_FORCES 

    break;
case 0b0000010100010001:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (12*(15*rho_I + 2*m_xx_I*rho_I + 6*m_xy_I*rho_I + 2*m_yy_I*rho_I - m_zz_I*rho_I - 2*omegaVar*m_xx_I*rho_I - 6*omegaVar*m_xy_I*rho_I - 2*omegaVar*m_yy_I*rho_I + omegaVar*m_zz_I*rho_I))/(3*omegaVar + 172);

    m_xx_t45 = (1692*m_xx_I + 432*m_xy_I + 144*m_yy_I - 72*m_zz_I + 23*omegaVar*m_xx_I - 12*omegaVar*m_xy_I - 4*omegaVar*m_yy_I + 2*omegaVar*m_zz_I + 48)/(108*(2*m_xx_I + 6*m_xy_I + 2*m_yy_I - m_zz_I - 2*omegaVar*m_xx_I - 6*omegaVar*m_xy_I - 2*omegaVar*m_yy_I + omegaVar*m_zz_I + 15));
    m_xy_t90 = (72*m_xx_I + 732*m_xy_I + 72*m_yy_I - 36*m_zz_I - 2*omegaVar*m_xx_I + 3*omegaVar*m_xy_I - 2*omegaVar*m_yy_I + omegaVar*m_zz_I + 24)/(36*(2*m_xx_I + 6*m_xy_I + 2*m_yy_I - m_zz_I - 2*omegaVar*m_xx_I - 6*omegaVar*m_xy_I - 2*omegaVar*m_yy_I + omegaVar*m_zz_I + 15));
    m_xz_t90 = (m_xz_I*(3*omegaVar + 172))/(12*(2*m_xx_I + 6*m_xy_I + 2*m_yy_I - m_zz_I - 2*omegaVar*m_xx_I - 6*omegaVar*m_xy_I - 2*omegaVar*m_yy_I + omegaVar*m_zz_I + 15));
    m_yy_t45 = (144*m_xx_I + 432*m_xy_I + 1692*m_yy_I - 72*m_zz_I - 4*omegaVar*m_xx_I - 12*omegaVar*m_xy_I + 23*omegaVar*m_yy_I + 2*omegaVar*m_zz_I + 48)/(108*(2*m_xx_I + 6*m_xy_I + 2*m_yy_I - m_zz_I - 2*omegaVar*m_xx_I - 6*omegaVar*m_xy_I - 2*omegaVar*m_yy_I + omegaVar*m_zz_I + 15));
    m_yz_t90 = (m_yz_I*(3*omegaVar + 172))/(12*(2*m_xx_I + 6*m_xy_I + 2*m_yy_I - m_zz_I - 2*omegaVar*m_xx_I - 6*omegaVar*m_xy_I - 2*omegaVar*m_yy_I + omegaVar*m_zz_I + 15));
    m_zz_t45 = (792*m_zz_I - 108*m_xy_I - 36*m_yy_I - 36*m_xx_I + omegaVar*m_xx_I + 3*omegaVar*m_xy_I + omegaVar*m_yy_I + 13*omegaVar*m_zz_I - 12)/(54*(2*m_xx_I + 6*m_xy_I + 2*m_yy_I - m_zz_I - 2*omegaVar*m_xx_I - 6*omegaVar*m_xy_I - 2*omegaVar*m_yy_I + omegaVar*m_zz_I + 15));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = (m_zz_t45/12 - m_xy_t90/2 - m_yy_t45/6 - m_xx_t45/6 + (omegaVar*m_xx_t45)/12 + (omegaVar*m_xy_t90)/4 + (omegaVar*m_yy_t45)/12 - (omegaVar*m_zz_t45)/24 - 1.0/18.0)*rho + RHO_0/18;
    L_BC_Fy = (m_zz_t45/12 - m_xy_t90/2 - m_yy_t45/6 - m_xx_t45/6 + (omegaVar*m_xx_t45)/12 + (omegaVar*m_xy_t90)/4 + (omegaVar*m_yy_t45)/12 - (omegaVar*m_zz_t45)/24 - 1.0/18.0)*rho + RHO_0/18;
    L_BC_Fz = 0;
    #endif //_BC_FORCES 

    break;
case 0b0000010100010010:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = rho_I;

    m_xx_t45 = m_xx_I;
    m_xy_t90 = m_xy_I;
    m_xz_t90 = m_xz_I;
    m_yy_t45 = m_yy_I;
    m_yz_t90 = m_yz_I;
    m_zz_t45 = m_zz_I;

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = 0;
    L_BC_Fy = 0;
    L_BC_Fz = 0;
    #endif //_BC_FORCES 

    break;
case 0b0000010100010011:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[8] + pop[9] + pop[10] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (6*(15*rho_I + m_xx_I*rho_I + 6*m_xy_I*rho_I + 4*m_yy_I*rho_I + 6*m_yz_I*rho_I + m_zz_I*rho_I - omegaVar*m_xx_I*rho_I - 6*omegaVar*m_xy_I*rho_I - 4*omegaVar*m_yy_I*rho_I - 6*omegaVar*m_yz_I*rho_I - omegaVar*m_zz_I*rho_I))/(3*omegaVar + 82);

    m_xx_t45 = (2052*m_xx_I + 504*m_xy_I + 90*m_yy_I - 234*m_yz_I - 162*m_zz_I + 73*omegaVar*m_xx_I + 6*omegaVar*m_xy_I - 5*omegaVar*m_yy_I - 21*omegaVar*m_yz_I - 8*omegaVar*m_zz_I + 30)/(135*(m_xx_I + 6*m_xy_I + 4*m_yy_I + 6*m_yz_I + m_zz_I - omegaVar*m_xx_I - 6*omegaVar*m_xy_I - 4*omegaVar*m_yy_I - 6*omegaVar*m_yz_I - omegaVar*m_zz_I + 15));
    m_xy_t90 = (168*m_xx_I + 1746*m_xy_I + 180*m_yy_I + 24*m_yz_I - 78*m_zz_I + 2*omegaVar*m_xx_I + 39*omegaVar*m_xy_I - 10*omegaVar*m_yy_I - 24*omegaVar*m_yz_I - 7*omegaVar*m_zz_I + 60)/(90*(m_xx_I + 6*m_xy_I + 4*m_yy_I + 6*m_yz_I + m_zz_I - omegaVar*m_xx_I - 6*omegaVar*m_xy_I - 4*omegaVar*m_yy_I - 6*omegaVar*m_yz_I - omegaVar*m_zz_I + 15));
    m_xz_t90 = (m_xz_I*(3*omegaVar + 82))/(6*(m_xx_I + 6*m_xy_I + 4*m_yy_I + 6*m_yz_I + m_zz_I - omegaVar*m_xx_I - 6*omegaVar*m_xy_I - 4*omegaVar*m_yy_I - 6*omegaVar*m_yz_I - omegaVar*m_zz_I + 15));
    m_yy_t45 = (36*m_xx_I + 216*m_xy_I + 882*m_yy_I + 216*m_yz_I + 36*m_zz_I - 2*omegaVar*m_xx_I - 12*omegaVar*m_xy_I + 19*omegaVar*m_yy_I - 12*omegaVar*m_yz_I - 2*omegaVar*m_zz_I + 48)/(54*(m_xx_I + 6*m_xy_I + 4*m_yy_I + 6*m_yz_I + m_zz_I - omegaVar*m_xx_I - 6*omegaVar*m_xy_I - 4*omegaVar*m_yy_I - 6*omegaVar*m_yz_I - omegaVar*m_zz_I + 15));
    m_yz_t90 = (24*m_xy_I - 78*m_xx_I + 180*m_yy_I + 1746*m_yz_I + 168*m_zz_I - 7*omegaVar*m_xx_I - 24*omegaVar*m_xy_I - 10*omegaVar*m_yy_I + 39*omegaVar*m_yz_I + 2*omegaVar*m_zz_I + 60)/(90*(m_xx_I + 6*m_xy_I + 4*m_yy_I + 6*m_yz_I + m_zz_I - omegaVar*m_xx_I - 6*omegaVar*m_xy_I - 4*omegaVar*m_yy_I - 6*omegaVar*m_yz_I - omegaVar*m_zz_I + 15));
    m_zz_t45 = (90*m_yy_I - 234*m_xy_I - 162*m_xx_I + 504*m_yz_I + 2052*m_zz_I - 8*omegaVar*m_xx_I - 21*omegaVar*m_xy_I - 5*omegaVar*m_yy_I + 6*omegaVar*m_yz_I + 73*omegaVar*m_zz_I + 30)/(135*(m_xx_I + 6*m_xy_I + 4*m_yy_I + 6*m_yz_I + m_zz_I - omegaVar*m_xx_I - 6*omegaVar*m_xy_I - 4*omegaVar*m_yy_I - 6*omegaVar*m_yz_I - omegaVar*m_zz_I + 15));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = (m_zz_t45/12 - m_xy_t90/2 - m_yy_t45/6 - m_xx_t45/6 + (omegaVar*m_xx_t45)/12 + (omegaVar*m_xy_t90)/4 + (omegaVar*m_yy_t45)/12 - (omegaVar*m_zz_t45)/24 - 1.0/18.0)*rho + RHO_0/18;
    L_BC_Fy = ((omegaVar*m_xx_t45)/24 - m_xy_t90/2 - m_yy_t45/3 - m_yz_t90/2 - m_zz_t45/12 - m_xx_t45/12 + (omegaVar*m_xy_t90)/4 + (omegaVar*m_yy_t45)/6 + (omegaVar*m_yz_t90)/4 + (omegaVar*m_zz_t45)/24 - 1.0/9.0)*rho + RHO_0/9;
    L_BC_Fz = (m_xx_t45/12 - m_yy_t45/6 - m_yz_t90/2 - m_zz_t45/6 - (omegaVar*m_xx_t45)/24 + (omegaVar*m_yy_t45)/12 + (omegaVar*m_yz_t90)/4 + (omegaVar*m_zz_t45)/12 - 1.0/18.0)*rho + RHO_0/18;
    #endif //_BC_FORCES 

    break;
case 0b0000010100010100:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = rho_I;

    m_xx_t45 = m_xx_I;
    m_xy_t90 = m_xy_I;
    m_xz_t90 = m_xz_I;
    m_yy_t45 = m_yy_I;
    m_yz_t90 = m_yz_I;
    m_zz_t45 = m_zz_I;

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = 0;
    L_BC_Fy = 0;
    L_BC_Fz = 0;
    #endif //_BC_FORCES 

    break;
case 0b0000010100010101:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[8] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (6*(15*rho_I + 4*m_xx_I*rho_I + 6*m_xy_I*rho_I + 6*m_xz_I*rho_I + m_yy_I*rho_I + m_zz_I*rho_I - 4*omegaVar*m_xx_I*rho_I - 6*omegaVar*m_xy_I*rho_I - 6*omegaVar*m_xz_I*rho_I - omegaVar*m_yy_I*rho_I - omegaVar*m_zz_I*rho_I))/(3*omegaVar + 82);

    m_xx_t45 = (882*m_xx_I + 216*m_xy_I + 216*m_xz_I + 36*m_yy_I + 36*m_zz_I + 19*omegaVar*m_xx_I - 12*omegaVar*m_xy_I - 12*omegaVar*m_xz_I - 2*omegaVar*m_yy_I - 2*omegaVar*m_zz_I + 48)/(54*(4*m_xx_I + 6*m_xy_I + 6*m_xz_I + m_yy_I + m_zz_I - 4*omegaVar*m_xx_I - 6*omegaVar*m_xy_I - 6*omegaVar*m_xz_I - omegaVar*m_yy_I - omegaVar*m_zz_I + 15));
    m_xy_t90 = (180*m_xx_I + 1746*m_xy_I + 24*m_xz_I + 168*m_yy_I - 78*m_zz_I - 10*omegaVar*m_xx_I + 39*omegaVar*m_xy_I - 24*omegaVar*m_xz_I + 2*omegaVar*m_yy_I - 7*omegaVar*m_zz_I + 60)/(90*(4*m_xx_I + 6*m_xy_I + 6*m_xz_I + m_yy_I + m_zz_I - 4*omegaVar*m_xx_I - 6*omegaVar*m_xy_I - 6*omegaVar*m_xz_I - omegaVar*m_yy_I - omegaVar*m_zz_I + 15));
    m_xz_t90 = (180*m_xx_I + 24*m_xy_I + 1746*m_xz_I - 78*m_yy_I + 168*m_zz_I - 10*omegaVar*m_xx_I - 24*omegaVar*m_xy_I + 39*omegaVar*m_xz_I - 7*omegaVar*m_yy_I + 2*omegaVar*m_zz_I + 60)/(90*(4*m_xx_I + 6*m_xy_I + 6*m_xz_I + m_yy_I + m_zz_I - 4*omegaVar*m_xx_I - 6*omegaVar*m_xy_I - 6*omegaVar*m_xz_I - omegaVar*m_yy_I - omegaVar*m_zz_I + 15));
    m_yy_t45 = (90*m_xx_I + 504*m_xy_I - 234*m_xz_I + 2052*m_yy_I - 162*m_zz_I - 5*omegaVar*m_xx_I + 6*omegaVar*m_xy_I - 21*omegaVar*m_xz_I + 73*omegaVar*m_yy_I - 8*omegaVar*m_zz_I + 30)/(135*(4*m_xx_I + 6*m_xy_I + 6*m_xz_I + m_yy_I + m_zz_I - 4*omegaVar*m_xx_I - 6*omegaVar*m_xy_I - 6*omegaVar*m_xz_I - omegaVar*m_yy_I - omegaVar*m_zz_I + 15));
    m_yz_t90 = (m_yz_I*(3*omegaVar + 82))/(6*(4*m_xx_I + 6*m_xy_I + 6*m_xz_I + m_yy_I + m_zz_I - 4*omegaVar*m_xx_I - 6*omegaVar*m_xy_I - 6*omegaVar*m_xz_I - omegaVar*m_yy_I - omegaVar*m_zz_I + 15));
    m_zz_t45 = (90*m_xx_I - 234*m_xy_I + 504*m_xz_I - 162*m_yy_I + 2052*m_zz_I - 5*omegaVar*m_xx_I - 21*omegaVar*m_xy_I + 6*omegaVar*m_xz_I - 8*omegaVar*m_yy_I + 73*omegaVar*m_zz_I + 30)/(135*(4*m_xx_I + 6*m_xy_I + 6*m_xz_I + m_yy_I + m_zz_I - 4*omegaVar*m_xx_I - 6*omegaVar*m_xy_I - 6*omegaVar*m_xz_I - omegaVar*m_yy_I - omegaVar*m_zz_I + 15));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = ((omegaVar*m_xx_t45)/6 - m_xy_t90/2 - m_xz_t90/2 - m_yy_t45/12 - m_zz_t45/12 - m_xx_t45/3 + (omegaVar*m_xy_t90)/4 + (omegaVar*m_xz_t90)/4 + (omegaVar*m_yy_t45)/24 + (omegaVar*m_zz_t45)/24 - 1.0/9.0)*rho + RHO_0/9;
    L_BC_Fy = (m_zz_t45/12 - m_xy_t90/2 - m_yy_t45/6 - m_xx_t45/6 + (omegaVar*m_xx_t45)/12 + (omegaVar*m_xy_t90)/4 + (omegaVar*m_yy_t45)/12 - (omegaVar*m_zz_t45)/24 - 1.0/18.0)*rho + RHO_0/18;
    L_BC_Fz = (m_yy_t45/12 - m_xz_t90/2 - m_xx_t45/6 - m_zz_t45/6 + (omegaVar*m_xx_t45)/12 + (omegaVar*m_xz_t90)/4 - (omegaVar*m_yy_t45)/24 + (omegaVar*m_zz_t45)/12 - 1.0/18.0)*rho + RHO_0/18;
    #endif //_BC_FORCES 

    break;
case 0b0000010100010110:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = rho_I;

    m_xx_t45 = m_xx_I;
    m_xy_t90 = m_xy_I;
    m_xz_t90 = m_xz_I;
    m_yy_t45 = m_yy_I;
    m_yz_t90 = m_yz_I;
    m_zz_t45 = m_zz_I;

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = 0;
    L_BC_Fy = 0;
    L_BC_Fz = 0;
    #endif //_BC_FORCES 

    break;
case 0b0000010100010111:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[8] + pop[10] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (12*(5*rho_I + m_xx_I*rho_I + 2*m_xy_I*rho_I + 2*m_xz_I*rho_I + m_yy_I*rho_I + 2*m_yz_I*rho_I + m_zz_I*rho_I - omegaVar*m_xx_I*rho_I - 2*omegaVar*m_xy_I*rho_I - 2*omegaVar*m_xz_I*rho_I - omegaVar*m_yy_I*rho_I - 2*omegaVar*m_yz_I*rho_I - omegaVar*m_zz_I*rho_I))/(3*omegaVar + 52);

    m_xx_t45 = (472*m_xx_I + 112*m_xy_I + 112*m_xz_I + 4*m_yy_I - 44*m_yz_I + 4*m_zz_I + 23*omegaVar*m_xx_I - 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - 4*omegaVar*m_yy_I - 11*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 20)/(90*(m_xx_I + 2*m_xy_I + 2*m_xz_I + m_yy_I + 2*m_yz_I + m_zz_I - omegaVar*m_xx_I - 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - omegaVar*m_yy_I - 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_xy_t90 = (112*m_xx_I + 1108*m_xy_I + 16*m_xz_I + 112*m_yy_I + 16*m_yz_I - 44*m_zz_I - 2*omegaVar*m_xx_I + 47*omegaVar*m_xy_I - 16*omegaVar*m_xz_I - 2*omegaVar*m_yy_I - 16*omegaVar*m_yz_I - 11*omegaVar*m_zz_I + 40)/(180*(m_xx_I + 2*m_xy_I + 2*m_xz_I + m_yy_I + 2*m_yz_I + m_zz_I - omegaVar*m_xx_I - 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - omegaVar*m_yy_I - 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_xz_t90 = (112*m_xx_I + 16*m_xy_I + 1108*m_xz_I - 44*m_yy_I + 16*m_yz_I + 112*m_zz_I - 2*omegaVar*m_xx_I - 16*omegaVar*m_xy_I + 47*omegaVar*m_xz_I - 11*omegaVar*m_yy_I - 16*omegaVar*m_yz_I - 2*omegaVar*m_zz_I + 40)/(180*(m_xx_I + 2*m_xy_I + 2*m_xz_I + m_yy_I + 2*m_yz_I + m_zz_I - omegaVar*m_xx_I - 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - omegaVar*m_yy_I - 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_yy_t45 = (4*m_xx_I + 112*m_xy_I - 44*m_xz_I + 472*m_yy_I + 112*m_yz_I + 4*m_zz_I - 4*omegaVar*m_xx_I - 2*omegaVar*m_xy_I - 11*omegaVar*m_xz_I + 23*omegaVar*m_yy_I - 2*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 20)/(90*(m_xx_I + 2*m_xy_I + 2*m_xz_I + m_yy_I + 2*m_yz_I + m_zz_I - omegaVar*m_xx_I - 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - omegaVar*m_yy_I - 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_yz_t90 = (16*m_xy_I - 44*m_xx_I + 16*m_xz_I + 112*m_yy_I + 1108*m_yz_I + 112*m_zz_I - 11*omegaVar*m_xx_I - 16*omegaVar*m_xy_I - 16*omegaVar*m_xz_I - 2*omegaVar*m_yy_I + 47*omegaVar*m_yz_I - 2*omegaVar*m_zz_I + 40)/(180*(m_xx_I + 2*m_xy_I + 2*m_xz_I + m_yy_I + 2*m_yz_I + m_zz_I - omegaVar*m_xx_I - 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - omegaVar*m_yy_I - 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_zz_t45 = (4*m_xx_I - 44*m_xy_I + 112*m_xz_I + 4*m_yy_I + 112*m_yz_I + 472*m_zz_I - 4*omegaVar*m_xx_I - 11*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - 4*omegaVar*m_yy_I - 2*omegaVar*m_yz_I + 23*omegaVar*m_zz_I + 20)/(90*(m_xx_I + 2*m_xy_I + 2*m_xz_I + m_yy_I + 2*m_yz_I + m_zz_I - omegaVar*m_xx_I - 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - omegaVar*m_yy_I - 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = ((omegaVar*m_xx_t45)/6 - m_xy_t90/2 - m_xz_t90/2 - m_yy_t45/12 - m_zz_t45/12 - m_xx_t45/3 + (omegaVar*m_xy_t90)/4 + (omegaVar*m_xz_t90)/4 + (omegaVar*m_yy_t45)/24 + (omegaVar*m_zz_t45)/24 - 1.0/9.0)*rho + RHO_0/9;
    L_BC_Fy = ((omegaVar*m_xx_t45)/24 - m_xy_t90/2 - m_yy_t45/3 - m_yz_t90/2 - m_zz_t45/12 - m_xx_t45/12 + (omegaVar*m_xy_t90)/4 + (omegaVar*m_yy_t45)/6 + (omegaVar*m_yz_t90)/4 + (omegaVar*m_zz_t45)/24 - 1.0/9.0)*rho + RHO_0/9;
    L_BC_Fz = ((omegaVar*m_xx_t45)/24 - m_xz_t90/2 - m_yy_t45/12 - m_yz_t90/2 - m_zz_t45/3 - m_xx_t45/12 + (omegaVar*m_xz_t90)/4 + (omegaVar*m_yy_t45)/24 + (omegaVar*m_yz_t90)/4 + (omegaVar*m_zz_t45)/6 - 1.0/9.0)*rho + RHO_0/9;
    #endif //_BC_FORCES 

    break;
case 0b0000010100011000:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = rho_I;

    m_xx_t45 = m_xx_I;
    m_xy_t90 = m_xy_I;
    m_xz_t90 = m_xz_I;
    m_yy_t45 = m_yy_I;
    m_yz_t90 = m_yz_I;
    m_zz_t45 = m_zz_I;

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = 0;
    L_BC_Fy = 0;
    L_BC_Fz = 0;
    #endif //_BC_FORCES 

    break;
case 0b0000010100011001:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (12*(15*rho_I + 2*m_xx_I*rho_I + 6*m_xy_I*rho_I + 2*m_yy_I*rho_I - m_zz_I*rho_I - 2*omegaVar*m_xx_I*rho_I - 6*omegaVar*m_xy_I*rho_I - 2*omegaVar*m_yy_I*rho_I + omegaVar*m_zz_I*rho_I))/(3*omegaVar + 172);

    m_xx_t45 = (1692*m_xx_I + 432*m_xy_I + 144*m_yy_I - 72*m_zz_I + 23*omegaVar*m_xx_I - 12*omegaVar*m_xy_I - 4*omegaVar*m_yy_I + 2*omegaVar*m_zz_I + 48)/(108*(2*m_xx_I + 6*m_xy_I + 2*m_yy_I - m_zz_I - 2*omegaVar*m_xx_I - 6*omegaVar*m_xy_I - 2*omegaVar*m_yy_I + omegaVar*m_zz_I + 15));
    m_xy_t90 = (72*m_xx_I + 732*m_xy_I + 72*m_yy_I - 36*m_zz_I - 2*omegaVar*m_xx_I + 3*omegaVar*m_xy_I - 2*omegaVar*m_yy_I + omegaVar*m_zz_I + 24)/(36*(2*m_xx_I + 6*m_xy_I + 2*m_yy_I - m_zz_I - 2*omegaVar*m_xx_I - 6*omegaVar*m_xy_I - 2*omegaVar*m_yy_I + omegaVar*m_zz_I + 15));
    m_xz_t90 = (m_xz_I*(3*omegaVar + 172))/(12*(2*m_xx_I + 6*m_xy_I + 2*m_yy_I - m_zz_I - 2*omegaVar*m_xx_I - 6*omegaVar*m_xy_I - 2*omegaVar*m_yy_I + omegaVar*m_zz_I + 15));
    m_yy_t45 = (144*m_xx_I + 432*m_xy_I + 1692*m_yy_I - 72*m_zz_I - 4*omegaVar*m_xx_I - 12*omegaVar*m_xy_I + 23*omegaVar*m_yy_I + 2*omegaVar*m_zz_I + 48)/(108*(2*m_xx_I + 6*m_xy_I + 2*m_yy_I - m_zz_I - 2*omegaVar*m_xx_I - 6*omegaVar*m_xy_I - 2*omegaVar*m_yy_I + omegaVar*m_zz_I + 15));
    m_yz_t90 = (m_yz_I*(3*omegaVar + 172))/(12*(2*m_xx_I + 6*m_xy_I + 2*m_yy_I - m_zz_I - 2*omegaVar*m_xx_I - 6*omegaVar*m_xy_I - 2*omegaVar*m_yy_I + omegaVar*m_zz_I + 15));
    m_zz_t45 = (792*m_zz_I - 108*m_xy_I - 36*m_yy_I - 36*m_xx_I + omegaVar*m_xx_I + 3*omegaVar*m_xy_I + omegaVar*m_yy_I + 13*omegaVar*m_zz_I - 12)/(54*(2*m_xx_I + 6*m_xy_I + 2*m_yy_I - m_zz_I - 2*omegaVar*m_xx_I - 6*omegaVar*m_xy_I - 2*omegaVar*m_yy_I + omegaVar*m_zz_I + 15));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = (m_zz_t45/12 - m_xy_t90/2 - m_yy_t45/6 - m_xx_t45/6 + (omegaVar*m_xx_t45)/12 + (omegaVar*m_xy_t90)/4 + (omegaVar*m_yy_t45)/12 - (omegaVar*m_zz_t45)/24 - 1.0/18.0)*rho + RHO_0/18;
    L_BC_Fy = (m_zz_t45/12 - m_xy_t90/2 - m_yy_t45/6 - m_xx_t45/6 + (omegaVar*m_xx_t45)/12 + (omegaVar*m_xy_t90)/4 + (omegaVar*m_yy_t45)/12 - (omegaVar*m_zz_t45)/24 - 1.0/18.0)*rho + RHO_0/18;
    L_BC_Fz = 0;
    #endif //_BC_FORCES 

    break;
case 0b0000010100011010:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (12*(15*rho_I + 2*m_xx_I*rho_I - 6*m_xz_I*rho_I - m_yy_I*rho_I + 2*m_zz_I*rho_I - 2*omegaVar*m_xx_I*rho_I + 6*omegaVar*m_xz_I*rho_I + omegaVar*m_yy_I*rho_I - 2*omegaVar*m_zz_I*rho_I))/(3*omegaVar + 172);

    m_xx_t45 = (1692*m_xx_I - 432*m_xz_I - 72*m_yy_I + 144*m_zz_I + 23*omegaVar*m_xx_I + 12*omegaVar*m_xz_I + 2*omegaVar*m_yy_I - 4*omegaVar*m_zz_I + 48)/(108*(2*m_xx_I - 6*m_xz_I - m_yy_I + 2*m_zz_I - 2*omegaVar*m_xx_I + 6*omegaVar*m_xz_I + omegaVar*m_yy_I - 2*omegaVar*m_zz_I + 15));
    m_xy_t90 = (m_xy_I*(3*omegaVar + 172))/(12*(2*m_xx_I - 6*m_xz_I - m_yy_I + 2*m_zz_I - 2*omegaVar*m_xx_I + 6*omegaVar*m_xz_I + omegaVar*m_yy_I - 2*omegaVar*m_zz_I + 15));
    m_xz_t90 = (732*m_xz_I - 72*m_xx_I + 36*m_yy_I - 72*m_zz_I + 2*omegaVar*m_xx_I + 3*omegaVar*m_xz_I - omegaVar*m_yy_I + 2*omegaVar*m_zz_I - 24)/(36*(2*m_xx_I - 6*m_xz_I - m_yy_I + 2*m_zz_I - 2*omegaVar*m_xx_I + 6*omegaVar*m_xz_I + omegaVar*m_yy_I - 2*omegaVar*m_zz_I + 15));
    m_yy_t45 = (108*m_xz_I - 36*m_xx_I + 792*m_yy_I - 36*m_zz_I + omegaVar*m_xx_I - 3*omegaVar*m_xz_I + 13*omegaVar*m_yy_I + omegaVar*m_zz_I - 12)/(54*(2*m_xx_I - 6*m_xz_I - m_yy_I + 2*m_zz_I - 2*omegaVar*m_xx_I + 6*omegaVar*m_xz_I + omegaVar*m_yy_I - 2*omegaVar*m_zz_I + 15));
    m_yz_t90 = (m_yz_I*(3*omegaVar + 172))/(12*(2*m_xx_I - 6*m_xz_I - m_yy_I + 2*m_zz_I - 2*omegaVar*m_xx_I + 6*omegaVar*m_xz_I + omegaVar*m_yy_I - 2*omegaVar*m_zz_I + 15));
    m_zz_t45 = (144*m_xx_I - 432*m_xz_I - 72*m_yy_I + 1692*m_zz_I - 4*omegaVar*m_xx_I + 12*omegaVar*m_xz_I + 2*omegaVar*m_yy_I + 23*omegaVar*m_zz_I + 48)/(108*(2*m_xx_I - 6*m_xz_I - m_yy_I + 2*m_zz_I - 2*omegaVar*m_xx_I + 6*omegaVar*m_xz_I + omegaVar*m_yy_I - 2*omegaVar*m_zz_I + 15));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = (m_xx_t45/6 - m_xz_t90/2 - m_yy_t45/12 + m_zz_t45/6 - (omegaVar*m_xx_t45)/12 + (omegaVar*m_xz_t90)/4 + (omegaVar*m_yy_t45)/24 - (omegaVar*m_zz_t45)/12 + 1.0/18.0)*rho - RHO_0/18;
    L_BC_Fy = 0;
    L_BC_Fz = (m_xz_t90/2 - m_xx_t45/6 + m_yy_t45/12 - m_zz_t45/6 + (omegaVar*m_xx_t45)/12 - (omegaVar*m_xz_t90)/4 - (omegaVar*m_yy_t45)/24 + (omegaVar*m_zz_t45)/12 - 1.0/18.0)*rho + RHO_0/18;
    #endif //_BC_FORCES 

    break;
case 0b0000010100011011:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[8] + pop[9] + pop[10] + pop[12] + pop[13] + pop[14] + pop[15] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (12*(5*rho_I + m_xx_I*rho_I + 2*m_xy_I*rho_I - 2*m_xz_I*rho_I + m_yy_I*rho_I + 2*m_yz_I*rho_I + m_zz_I*rho_I - omegaVar*m_xx_I*rho_I - 2*omegaVar*m_xy_I*rho_I + 2*omegaVar*m_xz_I*rho_I - omegaVar*m_yy_I*rho_I - 2*omegaVar*m_yz_I*rho_I - omegaVar*m_zz_I*rho_I))/(3*omegaVar + 52);

    m_xx_t45 = (472*m_xx_I + 112*m_xy_I - 112*m_xz_I + 4*m_yy_I - 44*m_yz_I + 4*m_zz_I + 23*omegaVar*m_xx_I - 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - 4*omegaVar*m_yy_I - 11*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 20)/(90*(m_xx_I + 2*m_xy_I - 2*m_xz_I + m_yy_I + 2*m_yz_I + m_zz_I - omegaVar*m_xx_I - 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - omegaVar*m_yy_I - 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_xy_t90 = (112*m_xx_I + 1108*m_xy_I - 16*m_xz_I + 112*m_yy_I + 16*m_yz_I - 44*m_zz_I - 2*omegaVar*m_xx_I + 47*omegaVar*m_xy_I + 16*omegaVar*m_xz_I - 2*omegaVar*m_yy_I - 16*omegaVar*m_yz_I - 11*omegaVar*m_zz_I + 40)/(180*(m_xx_I + 2*m_xy_I - 2*m_xz_I + m_yy_I + 2*m_yz_I + m_zz_I - omegaVar*m_xx_I - 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - omegaVar*m_yy_I - 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_xz_t90 = (1108*m_xz_I - 16*m_xy_I - 112*m_xx_I + 44*m_yy_I - 16*m_yz_I - 112*m_zz_I + 2*omegaVar*m_xx_I + 16*omegaVar*m_xy_I + 47*omegaVar*m_xz_I + 11*omegaVar*m_yy_I + 16*omegaVar*m_yz_I + 2*omegaVar*m_zz_I - 40)/(180*(m_xx_I + 2*m_xy_I - 2*m_xz_I + m_yy_I + 2*m_yz_I + m_zz_I - omegaVar*m_xx_I - 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - omegaVar*m_yy_I - 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_yy_t45 = (4*m_xx_I + 112*m_xy_I + 44*m_xz_I + 472*m_yy_I + 112*m_yz_I + 4*m_zz_I - 4*omegaVar*m_xx_I - 2*omegaVar*m_xy_I + 11*omegaVar*m_xz_I + 23*omegaVar*m_yy_I - 2*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 20)/(90*(m_xx_I + 2*m_xy_I - 2*m_xz_I + m_yy_I + 2*m_yz_I + m_zz_I - omegaVar*m_xx_I - 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - omegaVar*m_yy_I - 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_yz_t90 = (16*m_xy_I - 44*m_xx_I - 16*m_xz_I + 112*m_yy_I + 1108*m_yz_I + 112*m_zz_I - 11*omegaVar*m_xx_I - 16*omegaVar*m_xy_I + 16*omegaVar*m_xz_I - 2*omegaVar*m_yy_I + 47*omegaVar*m_yz_I - 2*omegaVar*m_zz_I + 40)/(180*(m_xx_I + 2*m_xy_I - 2*m_xz_I + m_yy_I + 2*m_yz_I + m_zz_I - omegaVar*m_xx_I - 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - omegaVar*m_yy_I - 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_zz_t45 = (4*m_xx_I - 44*m_xy_I - 112*m_xz_I + 4*m_yy_I + 112*m_yz_I + 472*m_zz_I - 4*omegaVar*m_xx_I - 11*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - 4*omegaVar*m_yy_I - 2*omegaVar*m_yz_I + 23*omegaVar*m_zz_I + 20)/(90*(m_xx_I + 2*m_xy_I - 2*m_xz_I + m_yy_I + 2*m_yz_I + m_zz_I - omegaVar*m_xx_I - 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - omegaVar*m_yy_I - 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = (rho*(omegaVar - 2)*(2*m_xy_t90 + 2*m_xz_t90 + m_yy_t45 - m_zz_t45))/8;
    L_BC_Fy = ((omegaVar*m_xx_t45)/24 - m_xy_t90/2 - m_yy_t45/3 - m_yz_t90/2 - m_zz_t45/12 - m_xx_t45/12 + (omegaVar*m_xy_t90)/4 + (omegaVar*m_yy_t45)/6 + (omegaVar*m_yz_t90)/4 + (omegaVar*m_zz_t45)/24 - 1.0/9.0)*rho + RHO_0/9;
    L_BC_Fz = (m_xz_t90/2 - m_xx_t45/12 - m_yy_t45/12 - m_yz_t90/2 - m_zz_t45/3 + (omegaVar*m_xx_t45)/24 - (omegaVar*m_xz_t90)/4 + (omegaVar*m_yy_t45)/24 + (omegaVar*m_yz_t90)/4 + (omegaVar*m_zz_t45)/6 - 1.0/9.0)*rho + RHO_0/9;
    #endif //_BC_FORCES 

    break;
case 0b0000010100011100:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17]);

    rho = (12*(15*rho_I - m_xx_I*rho_I + 2*m_yy_I*rho_I - 6*m_yz_I*rho_I + 2*m_zz_I*rho_I + omegaVar*m_xx_I*rho_I - 2*omegaVar*m_yy_I*rho_I + 6*omegaVar*m_yz_I*rho_I - 2*omegaVar*m_zz_I*rho_I))/(3*omegaVar + 172);

    m_xx_t45 = (792*m_xx_I - 36*m_yy_I + 108*m_yz_I - 36*m_zz_I + 13*omegaVar*m_xx_I + omegaVar*m_yy_I - 3*omegaVar*m_yz_I + omegaVar*m_zz_I - 12)/(54*(2*m_yy_I - m_xx_I - 6*m_yz_I + 2*m_zz_I + omegaVar*m_xx_I - 2*omegaVar*m_yy_I + 6*omegaVar*m_yz_I - 2*omegaVar*m_zz_I + 15));
    m_xy_t90 = (m_xy_I*(3*omegaVar + 172))/(12*(2*m_yy_I - m_xx_I - 6*m_yz_I + 2*m_zz_I + omegaVar*m_xx_I - 2*omegaVar*m_yy_I + 6*omegaVar*m_yz_I - 2*omegaVar*m_zz_I + 15));
    m_xz_t90 = (m_xz_I*(3*omegaVar + 172))/(12*(2*m_yy_I - m_xx_I - 6*m_yz_I + 2*m_zz_I + omegaVar*m_xx_I - 2*omegaVar*m_yy_I + 6*omegaVar*m_yz_I - 2*omegaVar*m_zz_I + 15));
    m_yy_t45 = (1692*m_yy_I - 72*m_xx_I - 432*m_yz_I + 144*m_zz_I + 2*omegaVar*m_xx_I + 23*omegaVar*m_yy_I + 12*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 48)/(108*(2*m_yy_I - m_xx_I - 6*m_yz_I + 2*m_zz_I + omegaVar*m_xx_I - 2*omegaVar*m_yy_I + 6*omegaVar*m_yz_I - 2*omegaVar*m_zz_I + 15));
    m_yz_t90 = (36*m_xx_I - 72*m_yy_I + 732*m_yz_I - 72*m_zz_I - omegaVar*m_xx_I + 2*omegaVar*m_yy_I + 3*omegaVar*m_yz_I + 2*omegaVar*m_zz_I - 24)/(36*(2*m_yy_I - m_xx_I - 6*m_yz_I + 2*m_zz_I + omegaVar*m_xx_I - 2*omegaVar*m_yy_I + 6*omegaVar*m_yz_I - 2*omegaVar*m_zz_I + 15));
    m_zz_t45 = (144*m_yy_I - 72*m_xx_I - 432*m_yz_I + 1692*m_zz_I + 2*omegaVar*m_xx_I - 4*omegaVar*m_yy_I + 12*omegaVar*m_yz_I + 23*omegaVar*m_zz_I + 48)/(108*(2*m_yy_I - m_xx_I - 6*m_yz_I + 2*m_zz_I + omegaVar*m_xx_I - 2*omegaVar*m_yy_I + 6*omegaVar*m_yz_I - 2*omegaVar*m_zz_I + 15));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = 0;
    L_BC_Fy = (m_yy_t45/6 - m_xx_t45/12 - m_yz_t90/2 + m_zz_t45/6 + (omegaVar*m_xx_t45)/24 - (omegaVar*m_yy_t45)/12 + (omegaVar*m_yz_t90)/4 - (omegaVar*m_zz_t45)/12 + 1.0/18.0)*rho - RHO_0/18;
    L_BC_Fz = (m_xx_t45/12 - m_yy_t45/6 + m_yz_t90/2 - m_zz_t45/6 - (omegaVar*m_xx_t45)/24 + (omegaVar*m_yy_t45)/12 - (omegaVar*m_yz_t90)/4 + (omegaVar*m_zz_t45)/12 - 1.0/18.0)*rho + RHO_0/18;
    #endif //_BC_FORCES 

    break;
case 0b0000010100011101:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[8] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17]);
    m_xy_I = inv_rho_I * (pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17]);

    rho = (12*(5*rho_I + m_xx_I*rho_I + 2*m_xy_I*rho_I + 2*m_xz_I*rho_I + m_yy_I*rho_I - 2*m_yz_I*rho_I + m_zz_I*rho_I - omegaVar*m_xx_I*rho_I - 2*omegaVar*m_xy_I*rho_I - 2*omegaVar*m_xz_I*rho_I - omegaVar*m_yy_I*rho_I + 2*omegaVar*m_yz_I*rho_I - omegaVar*m_zz_I*rho_I))/(3*omegaVar + 52);

    m_xx_t45 = (472*m_xx_I + 112*m_xy_I + 112*m_xz_I + 4*m_yy_I + 44*m_yz_I + 4*m_zz_I + 23*omegaVar*m_xx_I - 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - 4*omegaVar*m_yy_I + 11*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 20)/(90*(m_xx_I + 2*m_xy_I + 2*m_xz_I + m_yy_I - 2*m_yz_I + m_zz_I - omegaVar*m_xx_I - 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - omegaVar*m_yy_I + 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_xy_t90 = (112*m_xx_I + 1108*m_xy_I + 16*m_xz_I + 112*m_yy_I - 16*m_yz_I - 44*m_zz_I - 2*omegaVar*m_xx_I + 47*omegaVar*m_xy_I - 16*omegaVar*m_xz_I - 2*omegaVar*m_yy_I + 16*omegaVar*m_yz_I - 11*omegaVar*m_zz_I + 40)/(180*(m_xx_I + 2*m_xy_I + 2*m_xz_I + m_yy_I - 2*m_yz_I + m_zz_I - omegaVar*m_xx_I - 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - omegaVar*m_yy_I + 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_xz_t90 = (112*m_xx_I + 16*m_xy_I + 1108*m_xz_I - 44*m_yy_I - 16*m_yz_I + 112*m_zz_I - 2*omegaVar*m_xx_I - 16*omegaVar*m_xy_I + 47*omegaVar*m_xz_I - 11*omegaVar*m_yy_I + 16*omegaVar*m_yz_I - 2*omegaVar*m_zz_I + 40)/(180*(m_xx_I + 2*m_xy_I + 2*m_xz_I + m_yy_I - 2*m_yz_I + m_zz_I - omegaVar*m_xx_I - 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - omegaVar*m_yy_I + 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_yy_t45 = (4*m_xx_I + 112*m_xy_I - 44*m_xz_I + 472*m_yy_I - 112*m_yz_I + 4*m_zz_I - 4*omegaVar*m_xx_I - 2*omegaVar*m_xy_I - 11*omegaVar*m_xz_I + 23*omegaVar*m_yy_I + 2*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 20)/(90*(m_xx_I + 2*m_xy_I + 2*m_xz_I + m_yy_I - 2*m_yz_I + m_zz_I - omegaVar*m_xx_I - 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - omegaVar*m_yy_I + 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_yz_t90 = (44*m_xx_I - 16*m_xy_I - 16*m_xz_I - 112*m_yy_I + 1108*m_yz_I - 112*m_zz_I + 11*omegaVar*m_xx_I + 16*omegaVar*m_xy_I + 16*omegaVar*m_xz_I + 2*omegaVar*m_yy_I + 47*omegaVar*m_yz_I + 2*omegaVar*m_zz_I - 40)/(180*(m_xx_I + 2*m_xy_I + 2*m_xz_I + m_yy_I - 2*m_yz_I + m_zz_I - omegaVar*m_xx_I - 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - omegaVar*m_yy_I + 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_zz_t45 = (4*m_xx_I - 44*m_xy_I + 112*m_xz_I + 4*m_yy_I - 112*m_yz_I + 472*m_zz_I - 4*omegaVar*m_xx_I - 11*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - 4*omegaVar*m_yy_I + 2*omegaVar*m_yz_I + 23*omegaVar*m_zz_I + 20)/(90*(m_xx_I + 2*m_xy_I + 2*m_xz_I + m_yy_I - 2*m_yz_I + m_zz_I - omegaVar*m_xx_I - 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - omegaVar*m_yy_I + 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = ((omegaVar*m_xx_t45)/6 - m_xy_t90/2 - m_xz_t90/2 - m_yy_t45/12 - m_zz_t45/12 - m_xx_t45/3 + (omegaVar*m_xy_t90)/4 + (omegaVar*m_xz_t90)/4 + (omegaVar*m_yy_t45)/24 + (omegaVar*m_zz_t45)/24 - 1.0/9.0)*rho + RHO_0/9;
    L_BC_Fy = (rho*(omegaVar - 2)*(m_xx_t45 + 2*m_xy_t90 + 2*m_yz_t90 - m_zz_t45))/8;
    L_BC_Fz = (m_yz_t90/2 - m_xz_t90/2 - m_yy_t45/12 - m_xx_t45/12 - m_zz_t45/3 + (omegaVar*m_xx_t45)/24 + (omegaVar*m_xz_t90)/4 + (omegaVar*m_yy_t45)/24 - (omegaVar*m_yz_t90)/4 + (omegaVar*m_zz_t45)/6 - 1.0/9.0)*rho + RHO_0/9;
    #endif //_BC_FORCES 

    break;
case 0b0000010100011110:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[17];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] - (1.0/3.0)*pop[17]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] + (2.0/3.0)*pop[17]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[17]);

    rho = (6*(15*rho_I + m_xx_I*rho_I - 6*m_xz_I*rho_I + m_yy_I*rho_I - 6*m_yz_I*rho_I + 4*m_zz_I*rho_I - omegaVar*m_xx_I*rho_I + 6*omegaVar*m_xz_I*rho_I - omegaVar*m_yy_I*rho_I + 6*omegaVar*m_yz_I*rho_I - 4*omegaVar*m_zz_I*rho_I))/(3*omegaVar + 82);

    m_xx_t45 = (2052*m_xx_I - 504*m_xz_I - 162*m_yy_I + 234*m_yz_I + 90*m_zz_I + 73*omegaVar*m_xx_I - 6*omegaVar*m_xz_I - 8*omegaVar*m_yy_I + 21*omegaVar*m_yz_I - 5*omegaVar*m_zz_I + 30)/(135*(m_xx_I - 6*m_xz_I + m_yy_I - 6*m_yz_I + 4*m_zz_I - omegaVar*m_xx_I + 6*omegaVar*m_xz_I - omegaVar*m_yy_I + 6*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 15));
    m_xy_t90 = (m_xy_I*(3*omegaVar + 82))/(6*(m_xx_I - 6*m_xz_I + m_yy_I - 6*m_yz_I + 4*m_zz_I - omegaVar*m_xx_I + 6*omegaVar*m_xz_I - omegaVar*m_yy_I + 6*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 15));
    m_xz_t90 = (1746*m_xz_I - 168*m_xx_I + 78*m_yy_I + 24*m_yz_I - 180*m_zz_I - 2*omegaVar*m_xx_I + 39*omegaVar*m_xz_I + 7*omegaVar*m_yy_I - 24*omegaVar*m_yz_I + 10*omegaVar*m_zz_I - 60)/(90*(m_xx_I - 6*m_xz_I + m_yy_I - 6*m_yz_I + 4*m_zz_I - omegaVar*m_xx_I + 6*omegaVar*m_xz_I - omegaVar*m_yy_I + 6*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 15));
    m_yy_t45 = (234*m_xz_I - 162*m_xx_I + 2052*m_yy_I - 504*m_yz_I + 90*m_zz_I - 8*omegaVar*m_xx_I + 21*omegaVar*m_xz_I + 73*omegaVar*m_yy_I - 6*omegaVar*m_yz_I - 5*omegaVar*m_zz_I + 30)/(135*(m_xx_I - 6*m_xz_I + m_yy_I - 6*m_yz_I + 4*m_zz_I - omegaVar*m_xx_I + 6*omegaVar*m_xz_I - omegaVar*m_yy_I + 6*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 15));
    m_yz_t90 = (78*m_xx_I + 24*m_xz_I - 168*m_yy_I + 1746*m_yz_I - 180*m_zz_I + 7*omegaVar*m_xx_I - 24*omegaVar*m_xz_I - 2*omegaVar*m_yy_I + 39*omegaVar*m_yz_I + 10*omegaVar*m_zz_I - 60)/(90*(m_xx_I - 6*m_xz_I + m_yy_I - 6*m_yz_I + 4*m_zz_I - omegaVar*m_xx_I + 6*omegaVar*m_xz_I - omegaVar*m_yy_I + 6*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 15));
    m_zz_t45 = (36*m_xx_I - 216*m_xz_I + 36*m_yy_I - 216*m_yz_I + 882*m_zz_I - 2*omegaVar*m_xx_I + 12*omegaVar*m_xz_I - 2*omegaVar*m_yy_I + 12*omegaVar*m_yz_I + 19*omegaVar*m_zz_I + 48)/(54*(m_xx_I - 6*m_xz_I + m_yy_I - 6*m_yz_I + 4*m_zz_I - omegaVar*m_xx_I + 6*omegaVar*m_xz_I - omegaVar*m_yy_I + 6*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 15));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = (m_xx_t45/6 - m_xz_t90/2 - m_yy_t45/12 + m_zz_t45/6 - (omegaVar*m_xx_t45)/12 + (omegaVar*m_xz_t90)/4 + (omegaVar*m_yy_t45)/24 - (omegaVar*m_zz_t45)/12 + 1.0/18.0)*rho - RHO_0/18;
    L_BC_Fy = (m_yy_t45/6 - m_xx_t45/12 - m_yz_t90/2 + m_zz_t45/6 + (omegaVar*m_xx_t45)/24 - (omegaVar*m_yy_t45)/12 + (omegaVar*m_yz_t90)/4 - (omegaVar*m_zz_t45)/12 + 1.0/18.0)*rho - RHO_0/18;
    L_BC_Fz = (m_xz_t90/2 - m_xx_t45/12 - m_yy_t45/12 + m_yz_t90/2 - m_zz_t45/3 + (omegaVar*m_xx_t45)/24 - (omegaVar*m_xz_t90)/4 + (omegaVar*m_yy_t45)/24 - (omegaVar*m_yz_t90)/4 + (omegaVar*m_zz_t45)/6 - 1.0/9.0)*rho + RHO_0/9;
    #endif //_BC_FORCES 

    break;
case 0b0000010100011111:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[6] + pop[8] + pop[10] + pop[12] + pop[13] + pop[14] + pop[15] + pop[17];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] - (1.0/3.0)*pop[17]);
    m_xy_I = inv_rho_I * (pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[10] - pop[15]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] + (2.0/3.0)*pop[17]);
    m_yz_I = inv_rho_I * (pop[12] - pop[17]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[17]);

    rho = (36*(107*rho_I + 12*m_xx_I*rho_I + 33*m_xy_I*rho_I + 12*m_yy_I*rho_I + 72*m_zz_I*rho_I - 12*omegaVar*m_xx_I*rho_I - 33*omegaVar*m_xy_I*rho_I - 12*omegaVar*m_yy_I*rho_I - 72*omegaVar*m_zz_I*rho_I))/(313*omegaVar + 2790);

    m_xx_t45 = (2772*m_xx_I + 648*m_xy_I - 18*m_yy_I - 108*m_zz_I + 302*omegaVar*m_xx_I + 48*omegaVar*m_xy_I - 11*omegaVar*m_yy_I - 66*omegaVar*m_zz_I + 72)/(27*(12*m_xx_I + 33*m_xy_I + 12*m_yy_I + 72*m_zz_I - 12*omegaVar*m_xx_I - 33*omegaVar*m_xy_I - 12*omegaVar*m_yy_I - 72*omegaVar*m_zz_I + 107));
    m_xy_t90 = (432*m_xx_I + 3978*m_xy_I + 432*m_yy_I - 198*m_zz_I + 32*omegaVar*m_xx_I + 401*omegaVar*m_xy_I + 32*omegaVar*m_yy_I - 121*omegaVar*m_zz_I + 132)/(36*(12*m_xx_I + 33*m_xy_I + 12*m_yy_I + 72*m_zz_I - 12*omegaVar*m_xx_I - 33*omegaVar*m_xy_I - 12*omegaVar*m_yy_I - 72*omegaVar*m_zz_I + 107));
    m_xz_t90 = (m_xz_I*(313*omegaVar + 2790))/(18*(12*m_xx_I + 33*m_xy_I + 12*m_yy_I + 72*m_zz_I - 12*omegaVar*m_xx_I - 33*omegaVar*m_xy_I - 12*omegaVar*m_yy_I - 72*omegaVar*m_zz_I + 107));
    m_yy_t45 = (648*m_xy_I - 18*m_xx_I + 2772*m_yy_I - 108*m_zz_I - 11*omegaVar*m_xx_I + 48*omegaVar*m_xy_I + 302*omegaVar*m_yy_I - 66*omegaVar*m_zz_I + 72)/(27*(12*m_xx_I + 33*m_xy_I + 12*m_yy_I + 72*m_zz_I - 12*omegaVar*m_xx_I - 33*omegaVar*m_xy_I - 12*omegaVar*m_yy_I - 72*omegaVar*m_zz_I + 107));
    m_yz_t90 = (m_yz_I*(313*omegaVar + 2790))/(18*(12*m_xx_I + 33*m_xy_I + 12*m_yy_I + 72*m_zz_I - 12*omegaVar*m_xx_I - 33*omegaVar*m_xy_I - 12*omegaVar*m_yy_I - 72*omegaVar*m_zz_I + 107));
    m_zz_t45 = -(72*m_xx_I + 198*m_xy_I + 72*m_yy_I - 2358*m_zz_I + 44*omegaVar*m_xx_I + 121*omegaVar*m_xy_I + 44*omegaVar*m_yy_I - 49*omegaVar*m_zz_I - 288)/(18*(12*m_xx_I + 33*m_xy_I + 12*m_yy_I + 72*m_zz_I - 12*omegaVar*m_xx_I - 33*omegaVar*m_xy_I - 12*omegaVar*m_yy_I - 72*omegaVar*m_zz_I + 107));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = (m_zz_t45/12 - m_xy_t90/2 - m_xz_t90 - m_yy_t45/6 - m_xx_t45/6 + (omegaVar*m_xx_t45)/12 + (omegaVar*m_xy_t90)/4 + (omegaVar*m_xz_t90)/2 + (omegaVar*m_yy_t45)/12 - (omegaVar*m_zz_t45)/24 - 1.0/18.0)*rho + RHO_0/18;
    L_BC_Fy = (m_zz_t45/12 - m_xy_t90/2 - m_yy_t45/6 - m_yz_t90 - m_xx_t45/6 + (omegaVar*m_xx_t45)/12 + (omegaVar*m_xy_t90)/4 + (omegaVar*m_yy_t45)/12 + (omegaVar*m_yz_t90)/2 - (omegaVar*m_zz_t45)/24 - 1.0/18.0)*rho + RHO_0/18;
    L_BC_Fz = ((omegaVar*m_zz_t45)/2 - m_zz_t45 - 1.0/3.0)*rho + RHO_0/3;
    #endif //_BC_FORCES 

    break;
case 0b0000010100100000:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = rho_I;

    m_xx_t45 = m_xx_I;
    m_xy_t90 = m_xy_I;
    m_xz_t90 = m_xz_I;
    m_yy_t45 = m_yy_I;
    m_yz_t90 = m_yz_I;
    m_zz_t45 = m_zz_I;

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = 0;
    L_BC_Fy = 0;
    L_BC_Fz = 0;
    #endif //_BC_FORCES 

    break;
case 0b0000010100100001:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = rho_I;

    m_xx_t45 = m_xx_I;
    m_xy_t90 = m_xy_I;
    m_xz_t90 = m_xz_I;
    m_yy_t45 = m_yy_I;
    m_yz_t90 = m_yz_I;
    m_zz_t45 = m_zz_I;

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = 0;
    L_BC_Fy = 0;
    L_BC_Fz = 0;
    #endif //_BC_FORCES 

    break;
case 0b0000010100100010:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (12*(15*rho_I + 2*m_xx_I*rho_I - 6*m_xy_I*rho_I + 2*m_yy_I*rho_I - m_zz_I*rho_I - 2*omegaVar*m_xx_I*rho_I + 6*omegaVar*m_xy_I*rho_I - 2*omegaVar*m_yy_I*rho_I + omegaVar*m_zz_I*rho_I))/(3*omegaVar + 172);

    m_xx_t45 = (1692*m_xx_I - 432*m_xy_I + 144*m_yy_I - 72*m_zz_I + 23*omegaVar*m_xx_I + 12*omegaVar*m_xy_I - 4*omegaVar*m_yy_I + 2*omegaVar*m_zz_I + 48)/(108*(2*m_xx_I - 6*m_xy_I + 2*m_yy_I - m_zz_I - 2*omegaVar*m_xx_I + 6*omegaVar*m_xy_I - 2*omegaVar*m_yy_I + omegaVar*m_zz_I + 15));
    m_xy_t90 = (732*m_xy_I - 72*m_xx_I - 72*m_yy_I + 36*m_zz_I + 2*omegaVar*m_xx_I + 3*omegaVar*m_xy_I + 2*omegaVar*m_yy_I - omegaVar*m_zz_I - 24)/(36*(2*m_xx_I - 6*m_xy_I + 2*m_yy_I - m_zz_I - 2*omegaVar*m_xx_I + 6*omegaVar*m_xy_I - 2*omegaVar*m_yy_I + omegaVar*m_zz_I + 15));
    m_xz_t90 = (m_xz_I*(3*omegaVar + 172))/(12*(2*m_xx_I - 6*m_xy_I + 2*m_yy_I - m_zz_I - 2*omegaVar*m_xx_I + 6*omegaVar*m_xy_I - 2*omegaVar*m_yy_I + omegaVar*m_zz_I + 15));
    m_yy_t45 = (144*m_xx_I - 432*m_xy_I + 1692*m_yy_I - 72*m_zz_I - 4*omegaVar*m_xx_I + 12*omegaVar*m_xy_I + 23*omegaVar*m_yy_I + 2*omegaVar*m_zz_I + 48)/(108*(2*m_xx_I - 6*m_xy_I + 2*m_yy_I - m_zz_I - 2*omegaVar*m_xx_I + 6*omegaVar*m_xy_I - 2*omegaVar*m_yy_I + omegaVar*m_zz_I + 15));
    m_yz_t90 = (m_yz_I*(3*omegaVar + 172))/(12*(2*m_xx_I - 6*m_xy_I + 2*m_yy_I - m_zz_I - 2*omegaVar*m_xx_I + 6*omegaVar*m_xy_I - 2*omegaVar*m_yy_I + omegaVar*m_zz_I + 15));
    m_zz_t45 = (108*m_xy_I - 36*m_xx_I - 36*m_yy_I + 792*m_zz_I + omegaVar*m_xx_I - 3*omegaVar*m_xy_I + omegaVar*m_yy_I + 13*omegaVar*m_zz_I - 12)/(54*(2*m_xx_I - 6*m_xy_I + 2*m_yy_I - m_zz_I - 2*omegaVar*m_xx_I + 6*omegaVar*m_xy_I - 2*omegaVar*m_yy_I + omegaVar*m_zz_I + 15));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = (m_xx_t45/6 - m_xy_t90/2 + m_yy_t45/6 - m_zz_t45/12 - (omegaVar*m_xx_t45)/12 + (omegaVar*m_xy_t90)/4 - (omegaVar*m_yy_t45)/12 + (omegaVar*m_zz_t45)/24 + 1.0/18.0)*rho - RHO_0/18;
    L_BC_Fy = (m_xy_t90/2 - m_xx_t45/6 - m_yy_t45/6 + m_zz_t45/12 + (omegaVar*m_xx_t45)/12 - (omegaVar*m_xy_t90)/4 + (omegaVar*m_yy_t45)/12 - (omegaVar*m_zz_t45)/24 - 1.0/18.0)*rho + RHO_0/18;
    L_BC_Fz = 0;
    #endif //_BC_FORCES 

    break;
case 0b0000010100100011:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[12] + pop[13] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (6*(15*rho_I + m_xx_I*rho_I - 6*m_xy_I*rho_I + 4*m_yy_I*rho_I + 6*m_yz_I*rho_I + m_zz_I*rho_I - omegaVar*m_xx_I*rho_I + 6*omegaVar*m_xy_I*rho_I - 4*omegaVar*m_yy_I*rho_I - 6*omegaVar*m_yz_I*rho_I - omegaVar*m_zz_I*rho_I))/(3*omegaVar + 82);

    m_xx_t45 = -(504*m_xy_I - 2052*m_xx_I - 90*m_yy_I + 234*m_yz_I + 162*m_zz_I - 73*omegaVar*m_xx_I + 6*omegaVar*m_xy_I + 5*omegaVar*m_yy_I + 21*omegaVar*m_yz_I + 8*omegaVar*m_zz_I - 30)/(135*(m_xx_I - 6*m_xy_I + 4*m_yy_I + 6*m_yz_I + m_zz_I - omegaVar*m_xx_I + 6*omegaVar*m_xy_I - 4*omegaVar*m_yy_I - 6*omegaVar*m_yz_I - omegaVar*m_zz_I + 15));
    m_xy_t90 = (1746*m_xy_I - 168*m_xx_I - 180*m_yy_I - 24*m_yz_I + 78*m_zz_I - 2*omegaVar*m_xx_I + 39*omegaVar*m_xy_I + 10*omegaVar*m_yy_I + 24*omegaVar*m_yz_I + 7*omegaVar*m_zz_I - 60)/(90*(m_xx_I - 6*m_xy_I + 4*m_yy_I + 6*m_yz_I + m_zz_I - omegaVar*m_xx_I + 6*omegaVar*m_xy_I - 4*omegaVar*m_yy_I - 6*omegaVar*m_yz_I - omegaVar*m_zz_I + 15));
    m_xz_t90 = (m_xz_I*(3*omegaVar + 82))/(6*(m_xx_I - 6*m_xy_I + 4*m_yy_I + 6*m_yz_I + m_zz_I - omegaVar*m_xx_I + 6*omegaVar*m_xy_I - 4*omegaVar*m_yy_I - 6*omegaVar*m_yz_I - omegaVar*m_zz_I + 15));
    m_yy_t45 = (36*m_xx_I - 216*m_xy_I + 882*m_yy_I + 216*m_yz_I + 36*m_zz_I - 2*omegaVar*m_xx_I + 12*omegaVar*m_xy_I + 19*omegaVar*m_yy_I - 12*omegaVar*m_yz_I - 2*omegaVar*m_zz_I + 48)/(54*(m_xx_I - 6*m_xy_I + 4*m_yy_I + 6*m_yz_I + m_zz_I - omegaVar*m_xx_I + 6*omegaVar*m_xy_I - 4*omegaVar*m_yy_I - 6*omegaVar*m_yz_I - omegaVar*m_zz_I + 15));
    m_yz_t90 = (180*m_yy_I - 24*m_xy_I - 78*m_xx_I + 1746*m_yz_I + 168*m_zz_I - 7*omegaVar*m_xx_I + 24*omegaVar*m_xy_I - 10*omegaVar*m_yy_I + 39*omegaVar*m_yz_I + 2*omegaVar*m_zz_I + 60)/(90*(m_xx_I - 6*m_xy_I + 4*m_yy_I + 6*m_yz_I + m_zz_I - omegaVar*m_xx_I + 6*omegaVar*m_xy_I - 4*omegaVar*m_yy_I - 6*omegaVar*m_yz_I - omegaVar*m_zz_I + 15));
    m_zz_t45 = (234*m_xy_I - 162*m_xx_I + 90*m_yy_I + 504*m_yz_I + 2052*m_zz_I - 8*omegaVar*m_xx_I + 21*omegaVar*m_xy_I - 5*omegaVar*m_yy_I + 6*omegaVar*m_yz_I + 73*omegaVar*m_zz_I + 30)/(135*(m_xx_I - 6*m_xy_I + 4*m_yy_I + 6*m_yz_I + m_zz_I - omegaVar*m_xx_I + 6*omegaVar*m_xy_I - 4*omegaVar*m_yy_I - 6*omegaVar*m_yz_I - omegaVar*m_zz_I + 15));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = (m_xx_t45/6 - m_xy_t90/2 + m_yy_t45/6 - m_zz_t45/12 - (omegaVar*m_xx_t45)/12 + (omegaVar*m_xy_t90)/4 - (omegaVar*m_yy_t45)/12 + (omegaVar*m_zz_t45)/24 + 1.0/18.0)*rho - RHO_0/18;
    L_BC_Fy = (m_xy_t90/2 - m_xx_t45/12 - m_yy_t45/3 - m_yz_t90/2 - m_zz_t45/12 + (omegaVar*m_xx_t45)/24 - (omegaVar*m_xy_t90)/4 + (omegaVar*m_yy_t45)/6 + (omegaVar*m_yz_t90)/4 + (omegaVar*m_zz_t45)/24 - 1.0/9.0)*rho + RHO_0/9;
    L_BC_Fz = (m_xx_t45/12 - m_yy_t45/6 - m_yz_t90/2 - m_zz_t45/6 - (omegaVar*m_xx_t45)/24 + (omegaVar*m_yy_t45)/12 + (omegaVar*m_yz_t90)/4 + (omegaVar*m_zz_t45)/12 - 1.0/18.0)*rho + RHO_0/18;
    #endif //_BC_FORCES 

    break;
case 0b0000010100100100:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = rho_I;

    m_xx_t45 = m_xx_I;
    m_xy_t90 = m_xy_I;
    m_xz_t90 = m_xz_I;
    m_yy_t45 = m_yy_I;
    m_yz_t90 = m_yz_I;
    m_zz_t45 = m_zz_I;

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = 0;
    L_BC_Fy = 0;
    L_BC_Fz = 0;
    #endif //_BC_FORCES 

    break;
case 0b0000010100100101:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (12*(15*rho_I + 2*m_xx_I*rho_I + 6*m_xz_I*rho_I - m_yy_I*rho_I + 2*m_zz_I*rho_I - 2*omegaVar*m_xx_I*rho_I - 6*omegaVar*m_xz_I*rho_I + omegaVar*m_yy_I*rho_I - 2*omegaVar*m_zz_I*rho_I))/(3*omegaVar + 172);

    m_xx_t45 = (1692*m_xx_I + 432*m_xz_I - 72*m_yy_I + 144*m_zz_I + 23*omegaVar*m_xx_I - 12*omegaVar*m_xz_I + 2*omegaVar*m_yy_I - 4*omegaVar*m_zz_I + 48)/(108*(2*m_xx_I + 6*m_xz_I - m_yy_I + 2*m_zz_I - 2*omegaVar*m_xx_I - 6*omegaVar*m_xz_I + omegaVar*m_yy_I - 2*omegaVar*m_zz_I + 15));
    m_xy_t90 = (m_xy_I*(3*omegaVar + 172))/(12*(2*m_xx_I + 6*m_xz_I - m_yy_I + 2*m_zz_I - 2*omegaVar*m_xx_I - 6*omegaVar*m_xz_I + omegaVar*m_yy_I - 2*omegaVar*m_zz_I + 15));
    m_xz_t90 = (72*m_xx_I + 732*m_xz_I - 36*m_yy_I + 72*m_zz_I - 2*omegaVar*m_xx_I + 3*omegaVar*m_xz_I + omegaVar*m_yy_I - 2*omegaVar*m_zz_I + 24)/(36*(2*m_xx_I + 6*m_xz_I - m_yy_I + 2*m_zz_I - 2*omegaVar*m_xx_I - 6*omegaVar*m_xz_I + omegaVar*m_yy_I - 2*omegaVar*m_zz_I + 15));
    m_yy_t45 = (792*m_yy_I - 108*m_xz_I - 36*m_xx_I - 36*m_zz_I + omegaVar*m_xx_I + 3*omegaVar*m_xz_I + 13*omegaVar*m_yy_I + omegaVar*m_zz_I - 12)/(54*(2*m_xx_I + 6*m_xz_I - m_yy_I + 2*m_zz_I - 2*omegaVar*m_xx_I - 6*omegaVar*m_xz_I + omegaVar*m_yy_I - 2*omegaVar*m_zz_I + 15));
    m_yz_t90 = (m_yz_I*(3*omegaVar + 172))/(12*(2*m_xx_I + 6*m_xz_I - m_yy_I + 2*m_zz_I - 2*omegaVar*m_xx_I - 6*omegaVar*m_xz_I + omegaVar*m_yy_I - 2*omegaVar*m_zz_I + 15));
    m_zz_t45 = (144*m_xx_I + 432*m_xz_I - 72*m_yy_I + 1692*m_zz_I - 4*omegaVar*m_xx_I - 12*omegaVar*m_xz_I + 2*omegaVar*m_yy_I + 23*omegaVar*m_zz_I + 48)/(108*(2*m_xx_I + 6*m_xz_I - m_yy_I + 2*m_zz_I - 2*omegaVar*m_xx_I - 6*omegaVar*m_xz_I + omegaVar*m_yy_I - 2*omegaVar*m_zz_I + 15));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = (m_yy_t45/12 - m_xz_t90/2 - m_xx_t45/6 - m_zz_t45/6 + (omegaVar*m_xx_t45)/12 + (omegaVar*m_xz_t90)/4 - (omegaVar*m_yy_t45)/24 + (omegaVar*m_zz_t45)/12 - 1.0/18.0)*rho + RHO_0/18;
    L_BC_Fy = 0;
    L_BC_Fz = (m_yy_t45/12 - m_xz_t90/2 - m_xx_t45/6 - m_zz_t45/6 + (omegaVar*m_xx_t45)/12 + (omegaVar*m_xz_t90)/4 - (omegaVar*m_yy_t45)/24 + (omegaVar*m_zz_t45)/12 - 1.0/18.0)*rho + RHO_0/18;
    #endif //_BC_FORCES 

    break;
case 0b0000010100100110:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (12*(15*rho_I + 2*m_xx_I*rho_I - 6*m_xy_I*rho_I + 2*m_yy_I*rho_I - m_zz_I*rho_I - 2*omegaVar*m_xx_I*rho_I + 6*omegaVar*m_xy_I*rho_I - 2*omegaVar*m_yy_I*rho_I + omegaVar*m_zz_I*rho_I))/(3*omegaVar + 172);

    m_xx_t45 = (1692*m_xx_I - 432*m_xy_I + 144*m_yy_I - 72*m_zz_I + 23*omegaVar*m_xx_I + 12*omegaVar*m_xy_I - 4*omegaVar*m_yy_I + 2*omegaVar*m_zz_I + 48)/(108*(2*m_xx_I - 6*m_xy_I + 2*m_yy_I - m_zz_I - 2*omegaVar*m_xx_I + 6*omegaVar*m_xy_I - 2*omegaVar*m_yy_I + omegaVar*m_zz_I + 15));
    m_xy_t90 = (732*m_xy_I - 72*m_xx_I - 72*m_yy_I + 36*m_zz_I + 2*omegaVar*m_xx_I + 3*omegaVar*m_xy_I + 2*omegaVar*m_yy_I - omegaVar*m_zz_I - 24)/(36*(2*m_xx_I - 6*m_xy_I + 2*m_yy_I - m_zz_I - 2*omegaVar*m_xx_I + 6*omegaVar*m_xy_I - 2*omegaVar*m_yy_I + omegaVar*m_zz_I + 15));
    m_xz_t90 = (m_xz_I*(3*omegaVar + 172))/(12*(2*m_xx_I - 6*m_xy_I + 2*m_yy_I - m_zz_I - 2*omegaVar*m_xx_I + 6*omegaVar*m_xy_I - 2*omegaVar*m_yy_I + omegaVar*m_zz_I + 15));
    m_yy_t45 = (144*m_xx_I - 432*m_xy_I + 1692*m_yy_I - 72*m_zz_I - 4*omegaVar*m_xx_I + 12*omegaVar*m_xy_I + 23*omegaVar*m_yy_I + 2*omegaVar*m_zz_I + 48)/(108*(2*m_xx_I - 6*m_xy_I + 2*m_yy_I - m_zz_I - 2*omegaVar*m_xx_I + 6*omegaVar*m_xy_I - 2*omegaVar*m_yy_I + omegaVar*m_zz_I + 15));
    m_yz_t90 = (m_yz_I*(3*omegaVar + 172))/(12*(2*m_xx_I - 6*m_xy_I + 2*m_yy_I - m_zz_I - 2*omegaVar*m_xx_I + 6*omegaVar*m_xy_I - 2*omegaVar*m_yy_I + omegaVar*m_zz_I + 15));
    m_zz_t45 = (108*m_xy_I - 36*m_xx_I - 36*m_yy_I + 792*m_zz_I + omegaVar*m_xx_I - 3*omegaVar*m_xy_I + omegaVar*m_yy_I + 13*omegaVar*m_zz_I - 12)/(54*(2*m_xx_I - 6*m_xy_I + 2*m_yy_I - m_zz_I - 2*omegaVar*m_xx_I + 6*omegaVar*m_xy_I - 2*omegaVar*m_yy_I + omegaVar*m_zz_I + 15));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = (m_xx_t45/6 - m_xy_t90/2 + m_yy_t45/6 - m_zz_t45/12 - (omegaVar*m_xx_t45)/12 + (omegaVar*m_xy_t90)/4 - (omegaVar*m_yy_t45)/12 + (omegaVar*m_zz_t45)/24 + 1.0/18.0)*rho - RHO_0/18;
    L_BC_Fy = (m_xy_t90/2 - m_xx_t45/6 - m_yy_t45/6 + m_zz_t45/12 + (omegaVar*m_xx_t45)/12 - (omegaVar*m_xy_t90)/4 + (omegaVar*m_yy_t45)/12 - (omegaVar*m_zz_t45)/24 - 1.0/18.0)*rho + RHO_0/18;
    L_BC_Fz = 0;
    #endif //_BC_FORCES 

    break;
case 0b0000010100100111:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[10] + pop[12] + pop[13] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13]);
    m_xz_I = inv_rho_I * (pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (12*(5*rho_I + m_xx_I*rho_I - 2*m_xy_I*rho_I + 2*m_xz_I*rho_I + m_yy_I*rho_I + 2*m_yz_I*rho_I + m_zz_I*rho_I - omegaVar*m_xx_I*rho_I + 2*omegaVar*m_xy_I*rho_I - 2*omegaVar*m_xz_I*rho_I - omegaVar*m_yy_I*rho_I - 2*omegaVar*m_yz_I*rho_I - omegaVar*m_zz_I*rho_I))/(3*omegaVar + 52);

    m_xx_t45 = (472*m_xx_I - 112*m_xy_I + 112*m_xz_I + 4*m_yy_I - 44*m_yz_I + 4*m_zz_I + 23*omegaVar*m_xx_I + 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - 4*omegaVar*m_yy_I - 11*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 20)/(90*(m_xx_I - 2*m_xy_I + 2*m_xz_I + m_yy_I + 2*m_yz_I + m_zz_I - omegaVar*m_xx_I + 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - omegaVar*m_yy_I - 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_xy_t90 = (1108*m_xy_I - 112*m_xx_I - 16*m_xz_I - 112*m_yy_I - 16*m_yz_I + 44*m_zz_I + 2*omegaVar*m_xx_I + 47*omegaVar*m_xy_I + 16*omegaVar*m_xz_I + 2*omegaVar*m_yy_I + 16*omegaVar*m_yz_I + 11*omegaVar*m_zz_I - 40)/(180*(m_xx_I - 2*m_xy_I + 2*m_xz_I + m_yy_I + 2*m_yz_I + m_zz_I - omegaVar*m_xx_I + 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - omegaVar*m_yy_I - 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_xz_t90 = (112*m_xx_I - 16*m_xy_I + 1108*m_xz_I - 44*m_yy_I + 16*m_yz_I + 112*m_zz_I - 2*omegaVar*m_xx_I + 16*omegaVar*m_xy_I + 47*omegaVar*m_xz_I - 11*omegaVar*m_yy_I - 16*omegaVar*m_yz_I - 2*omegaVar*m_zz_I + 40)/(180*(m_xx_I - 2*m_xy_I + 2*m_xz_I + m_yy_I + 2*m_yz_I + m_zz_I - omegaVar*m_xx_I + 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - omegaVar*m_yy_I - 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_yy_t45 = (4*m_xx_I - 112*m_xy_I - 44*m_xz_I + 472*m_yy_I + 112*m_yz_I + 4*m_zz_I - 4*omegaVar*m_xx_I + 2*omegaVar*m_xy_I - 11*omegaVar*m_xz_I + 23*omegaVar*m_yy_I - 2*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 20)/(90*(m_xx_I - 2*m_xy_I + 2*m_xz_I + m_yy_I + 2*m_yz_I + m_zz_I - omegaVar*m_xx_I + 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - omegaVar*m_yy_I - 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_yz_t90 = (16*m_xz_I - 16*m_xy_I - 44*m_xx_I + 112*m_yy_I + 1108*m_yz_I + 112*m_zz_I - 11*omegaVar*m_xx_I + 16*omegaVar*m_xy_I - 16*omegaVar*m_xz_I - 2*omegaVar*m_yy_I + 47*omegaVar*m_yz_I - 2*omegaVar*m_zz_I + 40)/(180*(m_xx_I - 2*m_xy_I + 2*m_xz_I + m_yy_I + 2*m_yz_I + m_zz_I - omegaVar*m_xx_I + 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - omegaVar*m_yy_I - 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_zz_t45 = (4*m_xx_I + 44*m_xy_I + 112*m_xz_I + 4*m_yy_I + 112*m_yz_I + 472*m_zz_I - 4*omegaVar*m_xx_I + 11*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - 4*omegaVar*m_yy_I - 2*omegaVar*m_yz_I + 23*omegaVar*m_zz_I + 20)/(90*(m_xx_I - 2*m_xy_I + 2*m_xz_I + m_yy_I + 2*m_yz_I + m_zz_I - omegaVar*m_xx_I + 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - omegaVar*m_yy_I - 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = (rho*(omegaVar - 2)*(2*m_xy_t90 + 2*m_xz_t90 - m_yy_t45 + m_zz_t45))/8;
    L_BC_Fy = (m_xy_t90/2 - m_xx_t45/12 - m_yy_t45/3 - m_yz_t90/2 - m_zz_t45/12 + (omegaVar*m_xx_t45)/24 - (omegaVar*m_xy_t90)/4 + (omegaVar*m_yy_t45)/6 + (omegaVar*m_yz_t90)/4 + (omegaVar*m_zz_t45)/24 - 1.0/9.0)*rho + RHO_0/9;
    L_BC_Fz = ((omegaVar*m_xx_t45)/24 - m_xz_t90/2 - m_yy_t45/12 - m_yz_t90/2 - m_zz_t45/3 - m_xx_t45/12 + (omegaVar*m_xz_t90)/4 + (omegaVar*m_yy_t45)/24 + (omegaVar*m_yz_t90)/4 + (omegaVar*m_zz_t45)/6 - 1.0/9.0)*rho + RHO_0/9;
    #endif //_BC_FORCES 

    break;
case 0b0000010100101000:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = rho_I;

    m_xx_t45 = m_xx_I;
    m_xy_t90 = m_xy_I;
    m_xz_t90 = m_xz_I;
    m_yy_t45 = m_yy_I;
    m_yz_t90 = m_yz_I;
    m_zz_t45 = m_zz_I;

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = 0;
    L_BC_Fy = 0;
    L_BC_Fz = 0;
    #endif //_BC_FORCES 

    break;
case 0b0000010100101001:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = rho_I;

    m_xx_t45 = m_xx_I;
    m_xy_t90 = m_xy_I;
    m_xz_t90 = m_xz_I;
    m_yy_t45 = m_yy_I;
    m_yz_t90 = m_yz_I;
    m_zz_t45 = m_zz_I;

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = 0;
    L_BC_Fy = 0;
    L_BC_Fz = 0;
    #endif //_BC_FORCES 

    break;
case 0b0000010100101010:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[15] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[15] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] - (1.0/3.0)*pop[15] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (6*(15*rho_I + 4*m_xx_I*rho_I - 6*m_xy_I*rho_I - 6*m_xz_I*rho_I + m_yy_I*rho_I + m_zz_I*rho_I - 4*omegaVar*m_xx_I*rho_I + 6*omegaVar*m_xy_I*rho_I + 6*omegaVar*m_xz_I*rho_I - omegaVar*m_yy_I*rho_I - omegaVar*m_zz_I*rho_I))/(3*omegaVar + 82);

    m_xx_t45 = (882*m_xx_I - 216*m_xy_I - 216*m_xz_I + 36*m_yy_I + 36*m_zz_I + 19*omegaVar*m_xx_I + 12*omegaVar*m_xy_I + 12*omegaVar*m_xz_I - 2*omegaVar*m_yy_I - 2*omegaVar*m_zz_I + 48)/(54*(4*m_xx_I - 6*m_xy_I - 6*m_xz_I + m_yy_I + m_zz_I - 4*omegaVar*m_xx_I + 6*omegaVar*m_xy_I + 6*omegaVar*m_xz_I - omegaVar*m_yy_I - omegaVar*m_zz_I + 15));
    m_xy_t90 = (1746*m_xy_I - 180*m_xx_I + 24*m_xz_I - 168*m_yy_I + 78*m_zz_I + 10*omegaVar*m_xx_I + 39*omegaVar*m_xy_I - 24*omegaVar*m_xz_I - 2*omegaVar*m_yy_I + 7*omegaVar*m_zz_I - 60)/(90*(4*m_xx_I - 6*m_xy_I - 6*m_xz_I + m_yy_I + m_zz_I - 4*omegaVar*m_xx_I + 6*omegaVar*m_xy_I + 6*omegaVar*m_xz_I - omegaVar*m_yy_I - omegaVar*m_zz_I + 15));
    m_xz_t90 = (24*m_xy_I - 180*m_xx_I + 1746*m_xz_I + 78*m_yy_I - 168*m_zz_I + 10*omegaVar*m_xx_I - 24*omegaVar*m_xy_I + 39*omegaVar*m_xz_I + 7*omegaVar*m_yy_I - 2*omegaVar*m_zz_I - 60)/(90*(4*m_xx_I - 6*m_xy_I - 6*m_xz_I + m_yy_I + m_zz_I - 4*omegaVar*m_xx_I + 6*omegaVar*m_xy_I + 6*omegaVar*m_xz_I - omegaVar*m_yy_I - omegaVar*m_zz_I + 15));
    m_yy_t45 = (90*m_xx_I - 504*m_xy_I + 234*m_xz_I + 2052*m_yy_I - 162*m_zz_I - 5*omegaVar*m_xx_I - 6*omegaVar*m_xy_I + 21*omegaVar*m_xz_I + 73*omegaVar*m_yy_I - 8*omegaVar*m_zz_I + 30)/(135*(4*m_xx_I - 6*m_xy_I - 6*m_xz_I + m_yy_I + m_zz_I - 4*omegaVar*m_xx_I + 6*omegaVar*m_xy_I + 6*omegaVar*m_xz_I - omegaVar*m_yy_I - omegaVar*m_zz_I + 15));
    m_yz_t90 = (m_yz_I*(3*omegaVar + 82))/(6*(4*m_xx_I - 6*m_xy_I - 6*m_xz_I + m_yy_I + m_zz_I - 4*omegaVar*m_xx_I + 6*omegaVar*m_xy_I + 6*omegaVar*m_xz_I - omegaVar*m_yy_I - omegaVar*m_zz_I + 15));
    m_zz_t45 = (90*m_xx_I + 234*m_xy_I - 504*m_xz_I - 162*m_yy_I + 2052*m_zz_I - 5*omegaVar*m_xx_I + 21*omegaVar*m_xy_I - 6*omegaVar*m_xz_I - 8*omegaVar*m_yy_I + 73*omegaVar*m_zz_I + 30)/(135*(4*m_xx_I - 6*m_xy_I - 6*m_xz_I + m_yy_I + m_zz_I - 4*omegaVar*m_xx_I + 6*omegaVar*m_xy_I + 6*omegaVar*m_xz_I - omegaVar*m_yy_I - omegaVar*m_zz_I + 15));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = (m_xx_t45/3 - m_xy_t90/2 - m_xz_t90/2 + m_yy_t45/12 + m_zz_t45/12 - (omegaVar*m_xx_t45)/6 + (omegaVar*m_xy_t90)/4 + (omegaVar*m_xz_t90)/4 - (omegaVar*m_yy_t45)/24 - (omegaVar*m_zz_t45)/24 + 1.0/9.0)*rho - RHO_0/9;
    L_BC_Fy = (m_xy_t90/2 - m_xx_t45/6 - m_yy_t45/6 + m_zz_t45/12 + (omegaVar*m_xx_t45)/12 - (omegaVar*m_xy_t90)/4 + (omegaVar*m_yy_t45)/12 - (omegaVar*m_zz_t45)/24 - 1.0/18.0)*rho + RHO_0/18;
    L_BC_Fz = (m_xz_t90/2 - m_xx_t45/6 + m_yy_t45/12 - m_zz_t45/6 + (omegaVar*m_xx_t45)/12 - (omegaVar*m_xz_t90)/4 - (omegaVar*m_yy_t45)/24 + (omegaVar*m_zz_t45)/12 - 1.0/18.0)*rho + RHO_0/18;
    #endif //_BC_FORCES 

    break;
case 0b0000010100101011:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[12] + pop[13] + pop[15] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[15] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] - (1.0/3.0)*pop[15] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (12*(5*rho_I + m_xx_I*rho_I - 2*m_xy_I*rho_I - 2*m_xz_I*rho_I + m_yy_I*rho_I + 2*m_yz_I*rho_I + m_zz_I*rho_I - omegaVar*m_xx_I*rho_I + 2*omegaVar*m_xy_I*rho_I + 2*omegaVar*m_xz_I*rho_I - omegaVar*m_yy_I*rho_I - 2*omegaVar*m_yz_I*rho_I - omegaVar*m_zz_I*rho_I))/(3*omegaVar + 52);

    m_xx_t45 = (472*m_xx_I - 112*m_xy_I - 112*m_xz_I + 4*m_yy_I - 44*m_yz_I + 4*m_zz_I + 23*omegaVar*m_xx_I + 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - 4*omegaVar*m_yy_I - 11*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 20)/(90*(m_xx_I - 2*m_xy_I - 2*m_xz_I + m_yy_I + 2*m_yz_I + m_zz_I - omegaVar*m_xx_I + 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - omegaVar*m_yy_I - 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_xy_t90 = (1108*m_xy_I - 112*m_xx_I + 16*m_xz_I - 112*m_yy_I - 16*m_yz_I + 44*m_zz_I + 2*omegaVar*m_xx_I + 47*omegaVar*m_xy_I - 16*omegaVar*m_xz_I + 2*omegaVar*m_yy_I + 16*omegaVar*m_yz_I + 11*omegaVar*m_zz_I - 40)/(180*(m_xx_I - 2*m_xy_I - 2*m_xz_I + m_yy_I + 2*m_yz_I + m_zz_I - omegaVar*m_xx_I + 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - omegaVar*m_yy_I - 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_xz_t90 = (16*m_xy_I - 112*m_xx_I + 1108*m_xz_I + 44*m_yy_I - 16*m_yz_I - 112*m_zz_I + 2*omegaVar*m_xx_I - 16*omegaVar*m_xy_I + 47*omegaVar*m_xz_I + 11*omegaVar*m_yy_I + 16*omegaVar*m_yz_I + 2*omegaVar*m_zz_I - 40)/(180*(m_xx_I - 2*m_xy_I - 2*m_xz_I + m_yy_I + 2*m_yz_I + m_zz_I - omegaVar*m_xx_I + 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - omegaVar*m_yy_I - 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_yy_t45 = (4*m_xx_I - 112*m_xy_I + 44*m_xz_I + 472*m_yy_I + 112*m_yz_I + 4*m_zz_I - 4*omegaVar*m_xx_I + 2*omegaVar*m_xy_I + 11*omegaVar*m_xz_I + 23*omegaVar*m_yy_I - 2*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 20)/(90*(m_xx_I - 2*m_xy_I - 2*m_xz_I + m_yy_I + 2*m_yz_I + m_zz_I - omegaVar*m_xx_I + 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - omegaVar*m_yy_I - 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_yz_t90 = (112*m_yy_I - 16*m_xy_I - 16*m_xz_I - 44*m_xx_I + 1108*m_yz_I + 112*m_zz_I - 11*omegaVar*m_xx_I + 16*omegaVar*m_xy_I + 16*omegaVar*m_xz_I - 2*omegaVar*m_yy_I + 47*omegaVar*m_yz_I - 2*omegaVar*m_zz_I + 40)/(180*(m_xx_I - 2*m_xy_I - 2*m_xz_I + m_yy_I + 2*m_yz_I + m_zz_I - omegaVar*m_xx_I + 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - omegaVar*m_yy_I - 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_zz_t45 = (4*m_xx_I + 44*m_xy_I - 112*m_xz_I + 4*m_yy_I + 112*m_yz_I + 472*m_zz_I - 4*omegaVar*m_xx_I + 11*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - 4*omegaVar*m_yy_I - 2*omegaVar*m_yz_I + 23*omegaVar*m_zz_I + 20)/(90*(m_xx_I - 2*m_xy_I - 2*m_xz_I + m_yy_I + 2*m_yz_I + m_zz_I - omegaVar*m_xx_I + 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - omegaVar*m_yy_I - 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = (m_xx_t45/3 - m_xy_t90/2 - m_xz_t90/2 + m_yy_t45/12 + m_zz_t45/12 - (omegaVar*m_xx_t45)/6 + (omegaVar*m_xy_t90)/4 + (omegaVar*m_xz_t90)/4 - (omegaVar*m_yy_t45)/24 - (omegaVar*m_zz_t45)/24 + 1.0/9.0)*rho - RHO_0/9;
    L_BC_Fy = (m_xy_t90/2 - m_xx_t45/12 - m_yy_t45/3 - m_yz_t90/2 - m_zz_t45/12 + (omegaVar*m_xx_t45)/24 - (omegaVar*m_xy_t90)/4 + (omegaVar*m_yy_t45)/6 + (omegaVar*m_yz_t90)/4 + (omegaVar*m_zz_t45)/24 - 1.0/9.0)*rho + RHO_0/9;
    L_BC_Fz = (m_xz_t90/2 - m_xx_t45/12 - m_yy_t45/12 - m_yz_t90/2 - m_zz_t45/3 + (omegaVar*m_xx_t45)/24 - (omegaVar*m_xz_t90)/4 + (omegaVar*m_yy_t45)/24 + (omegaVar*m_yz_t90)/4 + (omegaVar*m_zz_t45)/6 - 1.0/9.0)*rho + RHO_0/9;
    #endif //_BC_FORCES 

    break;
case 0b0000010100101100:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17]);

    rho = (12*(15*rho_I - m_xx_I*rho_I + 2*m_yy_I*rho_I - 6*m_yz_I*rho_I + 2*m_zz_I*rho_I + omegaVar*m_xx_I*rho_I - 2*omegaVar*m_yy_I*rho_I + 6*omegaVar*m_yz_I*rho_I - 2*omegaVar*m_zz_I*rho_I))/(3*omegaVar + 172);

    m_xx_t45 = (792*m_xx_I - 36*m_yy_I + 108*m_yz_I - 36*m_zz_I + 13*omegaVar*m_xx_I + omegaVar*m_yy_I - 3*omegaVar*m_yz_I + omegaVar*m_zz_I - 12)/(54*(2*m_yy_I - m_xx_I - 6*m_yz_I + 2*m_zz_I + omegaVar*m_xx_I - 2*omegaVar*m_yy_I + 6*omegaVar*m_yz_I - 2*omegaVar*m_zz_I + 15));
    m_xy_t90 = (m_xy_I*(3*omegaVar + 172))/(12*(2*m_yy_I - m_xx_I - 6*m_yz_I + 2*m_zz_I + omegaVar*m_xx_I - 2*omegaVar*m_yy_I + 6*omegaVar*m_yz_I - 2*omegaVar*m_zz_I + 15));
    m_xz_t90 = (m_xz_I*(3*omegaVar + 172))/(12*(2*m_yy_I - m_xx_I - 6*m_yz_I + 2*m_zz_I + omegaVar*m_xx_I - 2*omegaVar*m_yy_I + 6*omegaVar*m_yz_I - 2*omegaVar*m_zz_I + 15));
    m_yy_t45 = (1692*m_yy_I - 72*m_xx_I - 432*m_yz_I + 144*m_zz_I + 2*omegaVar*m_xx_I + 23*omegaVar*m_yy_I + 12*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 48)/(108*(2*m_yy_I - m_xx_I - 6*m_yz_I + 2*m_zz_I + omegaVar*m_xx_I - 2*omegaVar*m_yy_I + 6*omegaVar*m_yz_I - 2*omegaVar*m_zz_I + 15));
    m_yz_t90 = (36*m_xx_I - 72*m_yy_I + 732*m_yz_I - 72*m_zz_I - omegaVar*m_xx_I + 2*omegaVar*m_yy_I + 3*omegaVar*m_yz_I + 2*omegaVar*m_zz_I - 24)/(36*(2*m_yy_I - m_xx_I - 6*m_yz_I + 2*m_zz_I + omegaVar*m_xx_I - 2*omegaVar*m_yy_I + 6*omegaVar*m_yz_I - 2*omegaVar*m_zz_I + 15));
    m_zz_t45 = (144*m_yy_I - 72*m_xx_I - 432*m_yz_I + 1692*m_zz_I + 2*omegaVar*m_xx_I - 4*omegaVar*m_yy_I + 12*omegaVar*m_yz_I + 23*omegaVar*m_zz_I + 48)/(108*(2*m_yy_I - m_xx_I - 6*m_yz_I + 2*m_zz_I + omegaVar*m_xx_I - 2*omegaVar*m_yy_I + 6*omegaVar*m_yz_I - 2*omegaVar*m_zz_I + 15));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = 0;
    L_BC_Fy = (m_yy_t45/6 - m_xx_t45/12 - m_yz_t90/2 + m_zz_t45/6 + (omegaVar*m_xx_t45)/24 - (omegaVar*m_yy_t45)/12 + (omegaVar*m_yz_t90)/4 - (omegaVar*m_zz_t45)/12 + 1.0/18.0)*rho - RHO_0/18;
    L_BC_Fz = (m_xx_t45/12 - m_yy_t45/6 + m_yz_t90/2 - m_zz_t45/6 - (omegaVar*m_xx_t45)/24 + (omegaVar*m_yy_t45)/12 - (omegaVar*m_yz_t90)/4 + (omegaVar*m_zz_t45)/12 - 1.0/18.0)*rho + RHO_0/18;
    #endif //_BC_FORCES 

    break;
case 0b0000010100101101:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17]);

    rho = (6*(15*rho_I + m_xx_I*rho_I + 6*m_xz_I*rho_I + m_yy_I*rho_I - 6*m_yz_I*rho_I + 4*m_zz_I*rho_I - omegaVar*m_xx_I*rho_I - 6*omegaVar*m_xz_I*rho_I - omegaVar*m_yy_I*rho_I + 6*omegaVar*m_yz_I*rho_I - 4*omegaVar*m_zz_I*rho_I))/(3*omegaVar + 82);

    m_xx_t45 = (2052*m_xx_I + 504*m_xz_I - 162*m_yy_I + 234*m_yz_I + 90*m_zz_I + 73*omegaVar*m_xx_I + 6*omegaVar*m_xz_I - 8*omegaVar*m_yy_I + 21*omegaVar*m_yz_I - 5*omegaVar*m_zz_I + 30)/(135*(m_xx_I + 6*m_xz_I + m_yy_I - 6*m_yz_I + 4*m_zz_I - omegaVar*m_xx_I - 6*omegaVar*m_xz_I - omegaVar*m_yy_I + 6*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 15));
    m_xy_t90 = (m_xy_I*(3*omegaVar + 82))/(6*(m_xx_I + 6*m_xz_I + m_yy_I - 6*m_yz_I + 4*m_zz_I - omegaVar*m_xx_I - 6*omegaVar*m_xz_I - omegaVar*m_yy_I + 6*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 15));
    m_xz_t90 = (168*m_xx_I + 1746*m_xz_I - 78*m_yy_I - 24*m_yz_I + 180*m_zz_I + 2*omegaVar*m_xx_I + 39*omegaVar*m_xz_I - 7*omegaVar*m_yy_I + 24*omegaVar*m_yz_I - 10*omegaVar*m_zz_I + 60)/(90*(m_xx_I + 6*m_xz_I + m_yy_I - 6*m_yz_I + 4*m_zz_I - omegaVar*m_xx_I - 6*omegaVar*m_xz_I - omegaVar*m_yy_I + 6*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 15));
    m_yy_t45 = -(162*m_xx_I + 234*m_xz_I - 2052*m_yy_I + 504*m_yz_I - 90*m_zz_I + 8*omegaVar*m_xx_I + 21*omegaVar*m_xz_I - 73*omegaVar*m_yy_I + 6*omegaVar*m_yz_I + 5*omegaVar*m_zz_I - 30)/(135*(m_xx_I + 6*m_xz_I + m_yy_I - 6*m_yz_I + 4*m_zz_I - omegaVar*m_xx_I - 6*omegaVar*m_xz_I - omegaVar*m_yy_I + 6*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 15));
    m_yz_t90 = (78*m_xx_I - 24*m_xz_I - 168*m_yy_I + 1746*m_yz_I - 180*m_zz_I + 7*omegaVar*m_xx_I + 24*omegaVar*m_xz_I - 2*omegaVar*m_yy_I + 39*omegaVar*m_yz_I + 10*omegaVar*m_zz_I - 60)/(90*(m_xx_I + 6*m_xz_I + m_yy_I - 6*m_yz_I + 4*m_zz_I - omegaVar*m_xx_I - 6*omegaVar*m_xz_I - omegaVar*m_yy_I + 6*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 15));
    m_zz_t45 = (36*m_xx_I + 216*m_xz_I + 36*m_yy_I - 216*m_yz_I + 882*m_zz_I - 2*omegaVar*m_xx_I - 12*omegaVar*m_xz_I - 2*omegaVar*m_yy_I + 12*omegaVar*m_yz_I + 19*omegaVar*m_zz_I + 48)/(54*(m_xx_I + 6*m_xz_I + m_yy_I - 6*m_yz_I + 4*m_zz_I - omegaVar*m_xx_I - 6*omegaVar*m_xz_I - omegaVar*m_yy_I + 6*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 15));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = (m_yy_t45/12 - m_xz_t90/2 - m_xx_t45/6 - m_zz_t45/6 + (omegaVar*m_xx_t45)/12 + (omegaVar*m_xz_t90)/4 - (omegaVar*m_yy_t45)/24 + (omegaVar*m_zz_t45)/12 - 1.0/18.0)*rho + RHO_0/18;
    L_BC_Fy = (m_yy_t45/6 - m_xx_t45/12 - m_yz_t90/2 + m_zz_t45/6 + (omegaVar*m_xx_t45)/24 - (omegaVar*m_yy_t45)/12 + (omegaVar*m_yz_t90)/4 - (omegaVar*m_zz_t45)/12 + 1.0/18.0)*rho - RHO_0/18;
    L_BC_Fz = (m_yz_t90/2 - m_xz_t90/2 - m_yy_t45/12 - m_xx_t45/12 - m_zz_t45/3 + (omegaVar*m_xx_t45)/24 + (omegaVar*m_xz_t90)/4 + (omegaVar*m_yy_t45)/24 - (omegaVar*m_yz_t90)/4 + (omegaVar*m_zz_t45)/6 - 1.0/9.0)*rho + RHO_0/9;
    #endif //_BC_FORCES 

    break;
case 0b0000010100101110:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[15] + pop[17];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[15] - (1.0/3.0)*pop[17]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] - (1.0/3.0)*pop[15] + (2.0/3.0)*pop[17]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[17]);

    rho = (12*(5*rho_I + m_xx_I*rho_I - 2*m_xy_I*rho_I - 2*m_xz_I*rho_I + m_yy_I*rho_I - 2*m_yz_I*rho_I + m_zz_I*rho_I - omegaVar*m_xx_I*rho_I + 2*omegaVar*m_xy_I*rho_I + 2*omegaVar*m_xz_I*rho_I - omegaVar*m_yy_I*rho_I + 2*omegaVar*m_yz_I*rho_I - omegaVar*m_zz_I*rho_I))/(3*omegaVar + 52);

    m_xx_t45 = (472*m_xx_I - 112*m_xy_I - 112*m_xz_I + 4*m_yy_I + 44*m_yz_I + 4*m_zz_I + 23*omegaVar*m_xx_I + 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - 4*omegaVar*m_yy_I + 11*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 20)/(90*(m_xx_I - 2*m_xy_I - 2*m_xz_I + m_yy_I - 2*m_yz_I + m_zz_I - omegaVar*m_xx_I + 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - omegaVar*m_yy_I + 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_xy_t90 = (1108*m_xy_I - 112*m_xx_I + 16*m_xz_I - 112*m_yy_I + 16*m_yz_I + 44*m_zz_I + 2*omegaVar*m_xx_I + 47*omegaVar*m_xy_I - 16*omegaVar*m_xz_I + 2*omegaVar*m_yy_I - 16*omegaVar*m_yz_I + 11*omegaVar*m_zz_I - 40)/(180*(m_xx_I - 2*m_xy_I - 2*m_xz_I + m_yy_I - 2*m_yz_I + m_zz_I - omegaVar*m_xx_I + 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - omegaVar*m_yy_I + 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_xz_t90 = (16*m_xy_I - 112*m_xx_I + 1108*m_xz_I + 44*m_yy_I + 16*m_yz_I - 112*m_zz_I + 2*omegaVar*m_xx_I - 16*omegaVar*m_xy_I + 47*omegaVar*m_xz_I + 11*omegaVar*m_yy_I - 16*omegaVar*m_yz_I + 2*omegaVar*m_zz_I - 40)/(180*(m_xx_I - 2*m_xy_I - 2*m_xz_I + m_yy_I - 2*m_yz_I + m_zz_I - omegaVar*m_xx_I + 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - omegaVar*m_yy_I + 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_yy_t45 = (4*m_xx_I - 112*m_xy_I + 44*m_xz_I + 472*m_yy_I - 112*m_yz_I + 4*m_zz_I - 4*omegaVar*m_xx_I + 2*omegaVar*m_xy_I + 11*omegaVar*m_xz_I + 23*omegaVar*m_yy_I + 2*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 20)/(90*(m_xx_I - 2*m_xy_I - 2*m_xz_I + m_yy_I - 2*m_yz_I + m_zz_I - omegaVar*m_xx_I + 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - omegaVar*m_yy_I + 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_yz_t90 = (44*m_xx_I + 16*m_xy_I + 16*m_xz_I - 112*m_yy_I + 1108*m_yz_I - 112*m_zz_I + 11*omegaVar*m_xx_I - 16*omegaVar*m_xy_I - 16*omegaVar*m_xz_I + 2*omegaVar*m_yy_I + 47*omegaVar*m_yz_I + 2*omegaVar*m_zz_I - 40)/(180*(m_xx_I - 2*m_xy_I - 2*m_xz_I + m_yy_I - 2*m_yz_I + m_zz_I - omegaVar*m_xx_I + 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - omegaVar*m_yy_I + 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_zz_t45 = (4*m_xx_I + 44*m_xy_I - 112*m_xz_I + 4*m_yy_I - 112*m_yz_I + 472*m_zz_I - 4*omegaVar*m_xx_I + 11*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - 4*omegaVar*m_yy_I + 2*omegaVar*m_yz_I + 23*omegaVar*m_zz_I + 20)/(90*(m_xx_I - 2*m_xy_I - 2*m_xz_I + m_yy_I - 2*m_yz_I + m_zz_I - omegaVar*m_xx_I + 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - omegaVar*m_yy_I + 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = (m_xx_t45/3 - m_xy_t90/2 - m_xz_t90/2 + m_yy_t45/12 + m_zz_t45/12 - (omegaVar*m_xx_t45)/6 + (omegaVar*m_xy_t90)/4 + (omegaVar*m_xz_t90)/4 - (omegaVar*m_yy_t45)/24 - (omegaVar*m_zz_t45)/24 + 1.0/9.0)*rho - RHO_0/9;
    L_BC_Fy = (rho*(omegaVar - 2)*(m_xx_t45 - 2*m_xy_t90 + 2*m_yz_t90 - m_zz_t45))/8;
    L_BC_Fz = (m_xz_t90/2 - m_xx_t45/12 - m_yy_t45/12 + m_yz_t90/2 - m_zz_t45/3 + (omegaVar*m_xx_t45)/24 - (omegaVar*m_xz_t90)/4 + (omegaVar*m_yy_t45)/24 - (omegaVar*m_yz_t90)/4 + (omegaVar*m_zz_t45)/6 - 1.0/9.0)*rho + RHO_0/9;
    #endif //_BC_FORCES 

    break;
case 0b0000010100101111:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[6] + pop[7] + pop[8] + pop[10] + pop[12] + pop[13] + pop[15] + pop[17];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[15] - (1.0/3.0)*pop[17]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13]);
    m_xz_I = inv_rho_I * (pop[10] - pop[15]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] - (1.0/3.0)*pop[15] + (2.0/3.0)*pop[17]);
    m_yz_I = inv_rho_I * (pop[12] - pop[17]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[17]);

    rho = (36*(107*rho_I + 12*m_xx_I*rho_I - 33*m_xy_I*rho_I + 12*m_yy_I*rho_I + 72*m_zz_I*rho_I - 12*omegaVar*m_xx_I*rho_I + 33*omegaVar*m_xy_I*rho_I - 12*omegaVar*m_yy_I*rho_I - 72*omegaVar*m_zz_I*rho_I))/(313*omegaVar + 2790);

    m_xx_t45 = -(648*m_xy_I - 2772*m_xx_I + 18*m_yy_I + 108*m_zz_I - 302*omegaVar*m_xx_I + 48*omegaVar*m_xy_I + 11*omegaVar*m_yy_I + 66*omegaVar*m_zz_I - 72)/(27*(12*m_xx_I - 33*m_xy_I + 12*m_yy_I + 72*m_zz_I - 12*omegaVar*m_xx_I + 33*omegaVar*m_xy_I - 12*omegaVar*m_yy_I - 72*omegaVar*m_zz_I + 107));
    m_xy_t90 = -(432*m_xx_I - 3978*m_xy_I + 432*m_yy_I - 198*m_zz_I + 32*omegaVar*m_xx_I - 401*omegaVar*m_xy_I + 32*omegaVar*m_yy_I - 121*omegaVar*m_zz_I + 132)/(36*(12*m_xx_I - 33*m_xy_I + 12*m_yy_I + 72*m_zz_I - 12*omegaVar*m_xx_I + 33*omegaVar*m_xy_I - 12*omegaVar*m_yy_I - 72*omegaVar*m_zz_I + 107));
    m_xz_t90 = (m_xz_I*(313*omegaVar + 2790))/(18*(12*m_xx_I - 33*m_xy_I + 12*m_yy_I + 72*m_zz_I - 12*omegaVar*m_xx_I + 33*omegaVar*m_xy_I - 12*omegaVar*m_yy_I - 72*omegaVar*m_zz_I + 107));
    m_yy_t45 = -(18*m_xx_I + 648*m_xy_I - 2772*m_yy_I + 108*m_zz_I + 11*omegaVar*m_xx_I + 48*omegaVar*m_xy_I - 302*omegaVar*m_yy_I + 66*omegaVar*m_zz_I - 72)/(27*(12*m_xx_I - 33*m_xy_I + 12*m_yy_I + 72*m_zz_I - 12*omegaVar*m_xx_I + 33*omegaVar*m_xy_I - 12*omegaVar*m_yy_I - 72*omegaVar*m_zz_I + 107));
    m_yz_t90 = (m_yz_I*(313*omegaVar + 2790))/(18*(12*m_xx_I - 33*m_xy_I + 12*m_yy_I + 72*m_zz_I - 12*omegaVar*m_xx_I + 33*omegaVar*m_xy_I - 12*omegaVar*m_yy_I - 72*omegaVar*m_zz_I + 107));
    m_zz_t45 = (198*m_xy_I - 72*m_xx_I - 72*m_yy_I + 2358*m_zz_I - 44*omegaVar*m_xx_I + 121*omegaVar*m_xy_I - 44*omegaVar*m_yy_I + 49*omegaVar*m_zz_I + 288)/(18*(12*m_xx_I - 33*m_xy_I + 12*m_yy_I + 72*m_zz_I - 12*omegaVar*m_xx_I + 33*omegaVar*m_xy_I - 12*omegaVar*m_yy_I - 72*omegaVar*m_zz_I + 107));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = (m_xx_t45/6 - m_xy_t90/2 - m_xz_t90 + m_yy_t45/6 - m_zz_t45/12 - (omegaVar*m_xx_t45)/12 + (omegaVar*m_xy_t90)/4 + (omegaVar*m_xz_t90)/2 - (omegaVar*m_yy_t45)/12 + (omegaVar*m_zz_t45)/24 + 1.0/18.0)*rho - RHO_0/18;
    L_BC_Fy = (m_xy_t90/2 - m_xx_t45/6 - m_yy_t45/6 - m_yz_t90 + m_zz_t45/12 + (omegaVar*m_xx_t45)/12 - (omegaVar*m_xy_t90)/4 + (omegaVar*m_yy_t45)/12 + (omegaVar*m_yz_t90)/2 - (omegaVar*m_zz_t45)/24 - 1.0/18.0)*rho + RHO_0/18;
    L_BC_Fz = ((omegaVar*m_zz_t45)/2 - m_zz_t45 - 1.0/3.0)*rho + RHO_0/3;
    #endif //_BC_FORCES 

    break;
case 0b0000010100110000:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);

    rho = (12*(15*rho_I - m_xx_I*rho_I + 2*m_yy_I*rho_I - 6*m_yz_I*rho_I + 2*m_zz_I*rho_I + omegaVar*m_xx_I*rho_I - 2*omegaVar*m_yy_I*rho_I + 6*omegaVar*m_yz_I*rho_I - 2*omegaVar*m_zz_I*rho_I))/(3*omegaVar + 172);

    m_xx_t45 = (792*m_xx_I - 36*m_yy_I + 108*m_yz_I - 36*m_zz_I + 13*omegaVar*m_xx_I + omegaVar*m_yy_I - 3*omegaVar*m_yz_I + omegaVar*m_zz_I - 12)/(54*(2*m_yy_I - m_xx_I - 6*m_yz_I + 2*m_zz_I + omegaVar*m_xx_I - 2*omegaVar*m_yy_I + 6*omegaVar*m_yz_I - 2*omegaVar*m_zz_I + 15));
    m_xy_t90 = (m_xy_I*(3*omegaVar + 172))/(12*(2*m_yy_I - m_xx_I - 6*m_yz_I + 2*m_zz_I + omegaVar*m_xx_I - 2*omegaVar*m_yy_I + 6*omegaVar*m_yz_I - 2*omegaVar*m_zz_I + 15));
    m_xz_t90 = (m_xz_I*(3*omegaVar + 172))/(12*(2*m_yy_I - m_xx_I - 6*m_yz_I + 2*m_zz_I + omegaVar*m_xx_I - 2*omegaVar*m_yy_I + 6*omegaVar*m_yz_I - 2*omegaVar*m_zz_I + 15));
    m_yy_t45 = (1692*m_yy_I - 72*m_xx_I - 432*m_yz_I + 144*m_zz_I + 2*omegaVar*m_xx_I + 23*omegaVar*m_yy_I + 12*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 48)/(108*(2*m_yy_I - m_xx_I - 6*m_yz_I + 2*m_zz_I + omegaVar*m_xx_I - 2*omegaVar*m_yy_I + 6*omegaVar*m_yz_I - 2*omegaVar*m_zz_I + 15));
    m_yz_t90 = (36*m_xx_I - 72*m_yy_I + 732*m_yz_I - 72*m_zz_I - omegaVar*m_xx_I + 2*omegaVar*m_yy_I + 3*omegaVar*m_yz_I + 2*omegaVar*m_zz_I - 24)/(36*(2*m_yy_I - m_xx_I - 6*m_yz_I + 2*m_zz_I + omegaVar*m_xx_I - 2*omegaVar*m_yy_I + 6*omegaVar*m_yz_I - 2*omegaVar*m_zz_I + 15));
    m_zz_t45 = (144*m_yy_I - 72*m_xx_I - 432*m_yz_I + 1692*m_zz_I + 2*omegaVar*m_xx_I - 4*omegaVar*m_yy_I + 12*omegaVar*m_yz_I + 23*omegaVar*m_zz_I + 48)/(108*(2*m_yy_I - m_xx_I - 6*m_yz_I + 2*m_zz_I + omegaVar*m_xx_I - 2*omegaVar*m_yy_I + 6*omegaVar*m_yz_I - 2*omegaVar*m_zz_I + 15));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = 0;
    L_BC_Fy = (m_xx_t45/12 - m_yy_t45/6 + m_yz_t90/2 - m_zz_t45/6 - (omegaVar*m_xx_t45)/24 + (omegaVar*m_yy_t45)/12 - (omegaVar*m_yz_t90)/4 + (omegaVar*m_zz_t45)/12 - 1.0/18.0)*rho + RHO_0/18;
    L_BC_Fz = (m_yy_t45/6 - m_xx_t45/12 - m_yz_t90/2 + m_zz_t45/6 + (omegaVar*m_xx_t45)/24 - (omegaVar*m_yy_t45)/12 + (omegaVar*m_yz_t90)/4 - (omegaVar*m_zz_t45)/12 + 1.0/18.0)*rho - RHO_0/18;
    #endif //_BC_FORCES 

    break;
case 0b0000010100110001:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);

    rho = (6*(15*rho_I + m_xx_I*rho_I + 6*m_xy_I*rho_I + 4*m_yy_I*rho_I - 6*m_yz_I*rho_I + m_zz_I*rho_I - omegaVar*m_xx_I*rho_I - 6*omegaVar*m_xy_I*rho_I - 4*omegaVar*m_yy_I*rho_I + 6*omegaVar*m_yz_I*rho_I - omegaVar*m_zz_I*rho_I))/(3*omegaVar + 82);

    m_xx_t45 = (2052*m_xx_I + 504*m_xy_I + 90*m_yy_I + 234*m_yz_I - 162*m_zz_I + 73*omegaVar*m_xx_I + 6*omegaVar*m_xy_I - 5*omegaVar*m_yy_I + 21*omegaVar*m_yz_I - 8*omegaVar*m_zz_I + 30)/(135*(m_xx_I + 6*m_xy_I + 4*m_yy_I - 6*m_yz_I + m_zz_I - omegaVar*m_xx_I - 6*omegaVar*m_xy_I - 4*omegaVar*m_yy_I + 6*omegaVar*m_yz_I - omegaVar*m_zz_I + 15));
    m_xy_t90 = (168*m_xx_I + 1746*m_xy_I + 180*m_yy_I - 24*m_yz_I - 78*m_zz_I + 2*omegaVar*m_xx_I + 39*omegaVar*m_xy_I - 10*omegaVar*m_yy_I + 24*omegaVar*m_yz_I - 7*omegaVar*m_zz_I + 60)/(90*(m_xx_I + 6*m_xy_I + 4*m_yy_I - 6*m_yz_I + m_zz_I - omegaVar*m_xx_I - 6*omegaVar*m_xy_I - 4*omegaVar*m_yy_I + 6*omegaVar*m_yz_I - omegaVar*m_zz_I + 15));
    m_xz_t90 = (m_xz_I*(3*omegaVar + 82))/(6*(m_xx_I + 6*m_xy_I + 4*m_yy_I - 6*m_yz_I + m_zz_I - omegaVar*m_xx_I - 6*omegaVar*m_xy_I - 4*omegaVar*m_yy_I + 6*omegaVar*m_yz_I - omegaVar*m_zz_I + 15));
    m_yy_t45 = (36*m_xx_I + 216*m_xy_I + 882*m_yy_I - 216*m_yz_I + 36*m_zz_I - 2*omegaVar*m_xx_I - 12*omegaVar*m_xy_I + 19*omegaVar*m_yy_I + 12*omegaVar*m_yz_I - 2*omegaVar*m_zz_I + 48)/(54*(m_xx_I + 6*m_xy_I + 4*m_yy_I - 6*m_yz_I + m_zz_I - omegaVar*m_xx_I - 6*omegaVar*m_xy_I - 4*omegaVar*m_yy_I + 6*omegaVar*m_yz_I - omegaVar*m_zz_I + 15));
    m_yz_t90 = (78*m_xx_I - 24*m_xy_I - 180*m_yy_I + 1746*m_yz_I - 168*m_zz_I + 7*omegaVar*m_xx_I + 24*omegaVar*m_xy_I + 10*omegaVar*m_yy_I + 39*omegaVar*m_yz_I - 2*omegaVar*m_zz_I - 60)/(90*(m_xx_I + 6*m_xy_I + 4*m_yy_I - 6*m_yz_I + m_zz_I - omegaVar*m_xx_I - 6*omegaVar*m_xy_I - 4*omegaVar*m_yy_I + 6*omegaVar*m_yz_I - omegaVar*m_zz_I + 15));
    m_zz_t45 = -(162*m_xx_I + 234*m_xy_I - 90*m_yy_I + 504*m_yz_I - 2052*m_zz_I + 8*omegaVar*m_xx_I + 21*omegaVar*m_xy_I + 5*omegaVar*m_yy_I + 6*omegaVar*m_yz_I - 73*omegaVar*m_zz_I - 30)/(135*(m_xx_I + 6*m_xy_I + 4*m_yy_I - 6*m_yz_I + m_zz_I - omegaVar*m_xx_I - 6*omegaVar*m_xy_I - 4*omegaVar*m_yy_I + 6*omegaVar*m_yz_I - omegaVar*m_zz_I + 15));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = (m_zz_t45/12 - m_xy_t90/2 - m_yy_t45/6 - m_xx_t45/6 + (omegaVar*m_xx_t45)/12 + (omegaVar*m_xy_t90)/4 + (omegaVar*m_yy_t45)/12 - (omegaVar*m_zz_t45)/24 - 1.0/18.0)*rho + RHO_0/18;
    L_BC_Fy = (m_yz_t90/2 - m_xy_t90/2 - m_yy_t45/3 - m_xx_t45/12 - m_zz_t45/12 + (omegaVar*m_xx_t45)/24 + (omegaVar*m_xy_t90)/4 + (omegaVar*m_yy_t45)/6 - (omegaVar*m_yz_t90)/4 + (omegaVar*m_zz_t45)/24 - 1.0/9.0)*rho + RHO_0/9;
    L_BC_Fz = (m_yy_t45/6 - m_xx_t45/12 - m_yz_t90/2 + m_zz_t45/6 + (omegaVar*m_xx_t45)/24 - (omegaVar*m_yy_t45)/12 + (omegaVar*m_yz_t90)/4 - (omegaVar*m_zz_t45)/12 + 1.0/18.0)*rho - RHO_0/18;
    #endif //_BC_FORCES 

    break;
case 0b0000010100110010:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[15] + pop[16] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);

    rho = (6*(15*rho_I + m_xx_I*rho_I - 6*m_xy_I*rho_I + 4*m_yy_I*rho_I - 6*m_yz_I*rho_I + m_zz_I*rho_I - omegaVar*m_xx_I*rho_I + 6*omegaVar*m_xy_I*rho_I - 4*omegaVar*m_yy_I*rho_I + 6*omegaVar*m_yz_I*rho_I - omegaVar*m_zz_I*rho_I))/(3*omegaVar + 82);

    m_xx_t45 = (2052*m_xx_I - 504*m_xy_I + 90*m_yy_I + 234*m_yz_I - 162*m_zz_I + 73*omegaVar*m_xx_I - 6*omegaVar*m_xy_I - 5*omegaVar*m_yy_I + 21*omegaVar*m_yz_I - 8*omegaVar*m_zz_I + 30)/(135*(m_xx_I - 6*m_xy_I + 4*m_yy_I - 6*m_yz_I + m_zz_I - omegaVar*m_xx_I + 6*omegaVar*m_xy_I - 4*omegaVar*m_yy_I + 6*omegaVar*m_yz_I - omegaVar*m_zz_I + 15));
    m_xy_t90 = (1746*m_xy_I - 168*m_xx_I - 180*m_yy_I + 24*m_yz_I + 78*m_zz_I - 2*omegaVar*m_xx_I + 39*omegaVar*m_xy_I + 10*omegaVar*m_yy_I - 24*omegaVar*m_yz_I + 7*omegaVar*m_zz_I - 60)/(90*(m_xx_I - 6*m_xy_I + 4*m_yy_I - 6*m_yz_I + m_zz_I - omegaVar*m_xx_I + 6*omegaVar*m_xy_I - 4*omegaVar*m_yy_I + 6*omegaVar*m_yz_I - omegaVar*m_zz_I + 15));
    m_xz_t90 = (m_xz_I*(3*omegaVar + 82))/(6*(m_xx_I - 6*m_xy_I + 4*m_yy_I - 6*m_yz_I + m_zz_I - omegaVar*m_xx_I + 6*omegaVar*m_xy_I - 4*omegaVar*m_yy_I + 6*omegaVar*m_yz_I - omegaVar*m_zz_I + 15));
    m_yy_t45 = (36*m_xx_I - 216*m_xy_I + 882*m_yy_I - 216*m_yz_I + 36*m_zz_I - 2*omegaVar*m_xx_I + 12*omegaVar*m_xy_I + 19*omegaVar*m_yy_I + 12*omegaVar*m_yz_I - 2*omegaVar*m_zz_I + 48)/(54*(m_xx_I - 6*m_xy_I + 4*m_yy_I - 6*m_yz_I + m_zz_I - omegaVar*m_xx_I + 6*omegaVar*m_xy_I - 4*omegaVar*m_yy_I + 6*omegaVar*m_yz_I - omegaVar*m_zz_I + 15));
    m_yz_t90 = (78*m_xx_I + 24*m_xy_I - 180*m_yy_I + 1746*m_yz_I - 168*m_zz_I + 7*omegaVar*m_xx_I - 24*omegaVar*m_xy_I + 10*omegaVar*m_yy_I + 39*omegaVar*m_yz_I - 2*omegaVar*m_zz_I - 60)/(90*(m_xx_I - 6*m_xy_I + 4*m_yy_I - 6*m_yz_I + m_zz_I - omegaVar*m_xx_I + 6*omegaVar*m_xy_I - 4*omegaVar*m_yy_I + 6*omegaVar*m_yz_I - omegaVar*m_zz_I + 15));
    m_zz_t45 = (234*m_xy_I - 162*m_xx_I + 90*m_yy_I - 504*m_yz_I + 2052*m_zz_I - 8*omegaVar*m_xx_I + 21*omegaVar*m_xy_I - 5*omegaVar*m_yy_I - 6*omegaVar*m_yz_I + 73*omegaVar*m_zz_I + 30)/(135*(m_xx_I - 6*m_xy_I + 4*m_yy_I - 6*m_yz_I + m_zz_I - omegaVar*m_xx_I + 6*omegaVar*m_xy_I - 4*omegaVar*m_yy_I + 6*omegaVar*m_yz_I - omegaVar*m_zz_I + 15));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = (m_xx_t45/6 - m_xy_t90/2 + m_yy_t45/6 - m_zz_t45/12 - (omegaVar*m_xx_t45)/12 + (omegaVar*m_xy_t90)/4 - (omegaVar*m_yy_t45)/12 + (omegaVar*m_zz_t45)/24 + 1.0/18.0)*rho - RHO_0/18;
    L_BC_Fy = (m_xy_t90/2 - m_xx_t45/12 - m_yy_t45/3 + m_yz_t90/2 - m_zz_t45/12 + (omegaVar*m_xx_t45)/24 - (omegaVar*m_xy_t90)/4 + (omegaVar*m_yy_t45)/6 - (omegaVar*m_yz_t90)/4 + (omegaVar*m_zz_t45)/24 - 1.0/9.0)*rho + RHO_0/9;
    L_BC_Fz = (m_yy_t45/6 - m_xx_t45/12 - m_yz_t90/2 + m_zz_t45/6 + (omegaVar*m_xx_t45)/24 - (omegaVar*m_yy_t45)/12 + (omegaVar*m_yz_t90)/4 - (omegaVar*m_zz_t45)/12 + 1.0/18.0)*rho - RHO_0/18;
    #endif //_BC_FORCES 

    break;
case 0b0000010100110011:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[4] + pop[5] + pop[6] + pop[8] + pop[9] + pop[10] + pop[12] + pop[13] + pop[15] + pop[16] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[8] - pop[13]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[12] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);

    rho = (3*rho_I*(3*m_yy_I - 3*omegaVar*m_yy_I + 4))/(omegaVar + 9);

    m_xx_t45 = (4*(omegaVar + 9)*(10*m_xx_I - m_zz_I))/(99*(3*m_yy_I - 3*omegaVar*m_yy_I + 4));
    m_xy_t90 = (2*m_xy_I*(omegaVar + 9))/(3*(3*m_yy_I - 3*omegaVar*m_yy_I + 4));
    m_xz_t90 = (m_xz_I*(omegaVar + 9))/(3*(3*m_yy_I - 3*omegaVar*m_yy_I + 4));
    m_yy_t45 = (15*m_yy_I + 2)/(3*(3*m_yy_I - 3*omegaVar*m_yy_I + 4));
    m_yz_t90 = (2*m_yz_I*(omegaVar + 9))/(3*(3*m_yy_I - 3*omegaVar*m_yy_I + 4));
    m_zz_t45 = -(4*(m_xx_I - 10*m_zz_I)*(omegaVar + 9))/(99*(3*m_yy_I - 3*omegaVar*m_yy_I + 4));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = (m_xy_t90*rho*(omegaVar - 2))/2;
    L_BC_Fy = ((omegaVar*m_yy_t45)/2 - m_yy_t45 - 1.0/3.0)*rho + RHO_0/3;
    L_BC_Fz = (m_yz_t90*rho*(omegaVar - 2))/2;
    #endif //_BC_FORCES 

    break;
case 0b0000010100110100:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);

    rho = (12*(15*rho_I - m_xx_I*rho_I + 2*m_yy_I*rho_I - 6*m_yz_I*rho_I + 2*m_zz_I*rho_I + omegaVar*m_xx_I*rho_I - 2*omegaVar*m_yy_I*rho_I + 6*omegaVar*m_yz_I*rho_I - 2*omegaVar*m_zz_I*rho_I))/(3*omegaVar + 172);

    m_xx_t45 = (792*m_xx_I - 36*m_yy_I + 108*m_yz_I - 36*m_zz_I + 13*omegaVar*m_xx_I + omegaVar*m_yy_I - 3*omegaVar*m_yz_I + omegaVar*m_zz_I - 12)/(54*(2*m_yy_I - m_xx_I - 6*m_yz_I + 2*m_zz_I + omegaVar*m_xx_I - 2*omegaVar*m_yy_I + 6*omegaVar*m_yz_I - 2*omegaVar*m_zz_I + 15));
    m_xy_t90 = (m_xy_I*(3*omegaVar + 172))/(12*(2*m_yy_I - m_xx_I - 6*m_yz_I + 2*m_zz_I + omegaVar*m_xx_I - 2*omegaVar*m_yy_I + 6*omegaVar*m_yz_I - 2*omegaVar*m_zz_I + 15));
    m_xz_t90 = (m_xz_I*(3*omegaVar + 172))/(12*(2*m_yy_I - m_xx_I - 6*m_yz_I + 2*m_zz_I + omegaVar*m_xx_I - 2*omegaVar*m_yy_I + 6*omegaVar*m_yz_I - 2*omegaVar*m_zz_I + 15));
    m_yy_t45 = (1692*m_yy_I - 72*m_xx_I - 432*m_yz_I + 144*m_zz_I + 2*omegaVar*m_xx_I + 23*omegaVar*m_yy_I + 12*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 48)/(108*(2*m_yy_I - m_xx_I - 6*m_yz_I + 2*m_zz_I + omegaVar*m_xx_I - 2*omegaVar*m_yy_I + 6*omegaVar*m_yz_I - 2*omegaVar*m_zz_I + 15));
    m_yz_t90 = (36*m_xx_I - 72*m_yy_I + 732*m_yz_I - 72*m_zz_I - omegaVar*m_xx_I + 2*omegaVar*m_yy_I + 3*omegaVar*m_yz_I + 2*omegaVar*m_zz_I - 24)/(36*(2*m_yy_I - m_xx_I - 6*m_yz_I + 2*m_zz_I + omegaVar*m_xx_I - 2*omegaVar*m_yy_I + 6*omegaVar*m_yz_I - 2*omegaVar*m_zz_I + 15));
    m_zz_t45 = (144*m_yy_I - 72*m_xx_I - 432*m_yz_I + 1692*m_zz_I + 2*omegaVar*m_xx_I - 4*omegaVar*m_yy_I + 12*omegaVar*m_yz_I + 23*omegaVar*m_zz_I + 48)/(108*(2*m_yy_I - m_xx_I - 6*m_yz_I + 2*m_zz_I + omegaVar*m_xx_I - 2*omegaVar*m_yy_I + 6*omegaVar*m_yz_I - 2*omegaVar*m_zz_I + 15));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = 0;
    L_BC_Fy = (m_xx_t45/12 - m_yy_t45/6 + m_yz_t90/2 - m_zz_t45/6 - (omegaVar*m_xx_t45)/24 + (omegaVar*m_yy_t45)/12 - (omegaVar*m_yz_t90)/4 + (omegaVar*m_zz_t45)/12 - 1.0/18.0)*rho + RHO_0/18;
    L_BC_Fz = (m_yy_t45/6 - m_xx_t45/12 - m_yz_t90/2 + m_zz_t45/6 + (omegaVar*m_xx_t45)/24 - (omegaVar*m_yy_t45)/12 + (omegaVar*m_yz_t90)/4 - (omegaVar*m_zz_t45)/12 + 1.0/18.0)*rho - RHO_0/18;
    #endif //_BC_FORCES 

    break;
case 0b0000010100110101:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[8] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);

    rho = (12*(5*rho_I + m_xx_I*rho_I + 2*m_xy_I*rho_I + 2*m_xz_I*rho_I + m_yy_I*rho_I - 2*m_yz_I*rho_I + m_zz_I*rho_I - omegaVar*m_xx_I*rho_I - 2*omegaVar*m_xy_I*rho_I - 2*omegaVar*m_xz_I*rho_I - omegaVar*m_yy_I*rho_I + 2*omegaVar*m_yz_I*rho_I - omegaVar*m_zz_I*rho_I))/(3*omegaVar + 52);

    m_xx_t45 = (472*m_xx_I + 112*m_xy_I + 112*m_xz_I + 4*m_yy_I + 44*m_yz_I + 4*m_zz_I + 23*omegaVar*m_xx_I - 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - 4*omegaVar*m_yy_I + 11*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 20)/(90*(m_xx_I + 2*m_xy_I + 2*m_xz_I + m_yy_I - 2*m_yz_I + m_zz_I - omegaVar*m_xx_I - 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - omegaVar*m_yy_I + 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_xy_t90 = (112*m_xx_I + 1108*m_xy_I + 16*m_xz_I + 112*m_yy_I - 16*m_yz_I - 44*m_zz_I - 2*omegaVar*m_xx_I + 47*omegaVar*m_xy_I - 16*omegaVar*m_xz_I - 2*omegaVar*m_yy_I + 16*omegaVar*m_yz_I - 11*omegaVar*m_zz_I + 40)/(180*(m_xx_I + 2*m_xy_I + 2*m_xz_I + m_yy_I - 2*m_yz_I + m_zz_I - omegaVar*m_xx_I - 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - omegaVar*m_yy_I + 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_xz_t90 = (112*m_xx_I + 16*m_xy_I + 1108*m_xz_I - 44*m_yy_I - 16*m_yz_I + 112*m_zz_I - 2*omegaVar*m_xx_I - 16*omegaVar*m_xy_I + 47*omegaVar*m_xz_I - 11*omegaVar*m_yy_I + 16*omegaVar*m_yz_I - 2*omegaVar*m_zz_I + 40)/(180*(m_xx_I + 2*m_xy_I + 2*m_xz_I + m_yy_I - 2*m_yz_I + m_zz_I - omegaVar*m_xx_I - 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - omegaVar*m_yy_I + 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_yy_t45 = (4*m_xx_I + 112*m_xy_I - 44*m_xz_I + 472*m_yy_I - 112*m_yz_I + 4*m_zz_I - 4*omegaVar*m_xx_I - 2*omegaVar*m_xy_I - 11*omegaVar*m_xz_I + 23*omegaVar*m_yy_I + 2*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 20)/(90*(m_xx_I + 2*m_xy_I + 2*m_xz_I + m_yy_I - 2*m_yz_I + m_zz_I - omegaVar*m_xx_I - 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - omegaVar*m_yy_I + 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_yz_t90 = (44*m_xx_I - 16*m_xy_I - 16*m_xz_I - 112*m_yy_I + 1108*m_yz_I - 112*m_zz_I + 11*omegaVar*m_xx_I + 16*omegaVar*m_xy_I + 16*omegaVar*m_xz_I + 2*omegaVar*m_yy_I + 47*omegaVar*m_yz_I + 2*omegaVar*m_zz_I - 40)/(180*(m_xx_I + 2*m_xy_I + 2*m_xz_I + m_yy_I - 2*m_yz_I + m_zz_I - omegaVar*m_xx_I - 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - omegaVar*m_yy_I + 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_zz_t45 = (4*m_xx_I - 44*m_xy_I + 112*m_xz_I + 4*m_yy_I - 112*m_yz_I + 472*m_zz_I - 4*omegaVar*m_xx_I - 11*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - 4*omegaVar*m_yy_I + 2*omegaVar*m_yz_I + 23*omegaVar*m_zz_I + 20)/(90*(m_xx_I + 2*m_xy_I + 2*m_xz_I + m_yy_I - 2*m_yz_I + m_zz_I - omegaVar*m_xx_I - 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - omegaVar*m_yy_I + 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = ((omegaVar*m_xx_t45)/6 - m_xy_t90/2 - m_xz_t90/2 - m_yy_t45/12 - m_zz_t45/12 - m_xx_t45/3 + (omegaVar*m_xy_t90)/4 + (omegaVar*m_xz_t90)/4 + (omegaVar*m_yy_t45)/24 + (omegaVar*m_zz_t45)/24 - 1.0/9.0)*rho + RHO_0/9;
    L_BC_Fy = (m_yz_t90/2 - m_xy_t90/2 - m_yy_t45/3 - m_xx_t45/12 - m_zz_t45/12 + (omegaVar*m_xx_t45)/24 + (omegaVar*m_xy_t90)/4 + (omegaVar*m_yy_t45)/6 - (omegaVar*m_yz_t90)/4 + (omegaVar*m_zz_t45)/24 - 1.0/9.0)*rho + RHO_0/9;
    L_BC_Fz = (rho*(omegaVar - 2)*(m_xx_t45 + 2*m_xz_t90 - m_yy_t45 + 2*m_yz_t90))/8;
    #endif //_BC_FORCES 

    break;
case 0b0000010100110110:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[15] + pop[16] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);

    rho = (6*(15*rho_I + m_xx_I*rho_I - 6*m_xy_I*rho_I + 4*m_yy_I*rho_I - 6*m_yz_I*rho_I + m_zz_I*rho_I - omegaVar*m_xx_I*rho_I + 6*omegaVar*m_xy_I*rho_I - 4*omegaVar*m_yy_I*rho_I + 6*omegaVar*m_yz_I*rho_I - omegaVar*m_zz_I*rho_I))/(3*omegaVar + 82);

    m_xx_t45 = (2052*m_xx_I - 504*m_xy_I + 90*m_yy_I + 234*m_yz_I - 162*m_zz_I + 73*omegaVar*m_xx_I - 6*omegaVar*m_xy_I - 5*omegaVar*m_yy_I + 21*omegaVar*m_yz_I - 8*omegaVar*m_zz_I + 30)/(135*(m_xx_I - 6*m_xy_I + 4*m_yy_I - 6*m_yz_I + m_zz_I - omegaVar*m_xx_I + 6*omegaVar*m_xy_I - 4*omegaVar*m_yy_I + 6*omegaVar*m_yz_I - omegaVar*m_zz_I + 15));
    m_xy_t90 = (1746*m_xy_I - 168*m_xx_I - 180*m_yy_I + 24*m_yz_I + 78*m_zz_I - 2*omegaVar*m_xx_I + 39*omegaVar*m_xy_I + 10*omegaVar*m_yy_I - 24*omegaVar*m_yz_I + 7*omegaVar*m_zz_I - 60)/(90*(m_xx_I - 6*m_xy_I + 4*m_yy_I - 6*m_yz_I + m_zz_I - omegaVar*m_xx_I + 6*omegaVar*m_xy_I - 4*omegaVar*m_yy_I + 6*omegaVar*m_yz_I - omegaVar*m_zz_I + 15));
    m_xz_t90 = (m_xz_I*(3*omegaVar + 82))/(6*(m_xx_I - 6*m_xy_I + 4*m_yy_I - 6*m_yz_I + m_zz_I - omegaVar*m_xx_I + 6*omegaVar*m_xy_I - 4*omegaVar*m_yy_I + 6*omegaVar*m_yz_I - omegaVar*m_zz_I + 15));
    m_yy_t45 = (36*m_xx_I - 216*m_xy_I + 882*m_yy_I - 216*m_yz_I + 36*m_zz_I - 2*omegaVar*m_xx_I + 12*omegaVar*m_xy_I + 19*omegaVar*m_yy_I + 12*omegaVar*m_yz_I - 2*omegaVar*m_zz_I + 48)/(54*(m_xx_I - 6*m_xy_I + 4*m_yy_I - 6*m_yz_I + m_zz_I - omegaVar*m_xx_I + 6*omegaVar*m_xy_I - 4*omegaVar*m_yy_I + 6*omegaVar*m_yz_I - omegaVar*m_zz_I + 15));
    m_yz_t90 = (78*m_xx_I + 24*m_xy_I - 180*m_yy_I + 1746*m_yz_I - 168*m_zz_I + 7*omegaVar*m_xx_I - 24*omegaVar*m_xy_I + 10*omegaVar*m_yy_I + 39*omegaVar*m_yz_I - 2*omegaVar*m_zz_I - 60)/(90*(m_xx_I - 6*m_xy_I + 4*m_yy_I - 6*m_yz_I + m_zz_I - omegaVar*m_xx_I + 6*omegaVar*m_xy_I - 4*omegaVar*m_yy_I + 6*omegaVar*m_yz_I - omegaVar*m_zz_I + 15));
    m_zz_t45 = (234*m_xy_I - 162*m_xx_I + 90*m_yy_I - 504*m_yz_I + 2052*m_zz_I - 8*omegaVar*m_xx_I + 21*omegaVar*m_xy_I - 5*omegaVar*m_yy_I - 6*omegaVar*m_yz_I + 73*omegaVar*m_zz_I + 30)/(135*(m_xx_I - 6*m_xy_I + 4*m_yy_I - 6*m_yz_I + m_zz_I - omegaVar*m_xx_I + 6*omegaVar*m_xy_I - 4*omegaVar*m_yy_I + 6*omegaVar*m_yz_I - omegaVar*m_zz_I + 15));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = (m_xx_t45/6 - m_xy_t90/2 + m_yy_t45/6 - m_zz_t45/12 - (omegaVar*m_xx_t45)/12 + (omegaVar*m_xy_t90)/4 - (omegaVar*m_yy_t45)/12 + (omegaVar*m_zz_t45)/24 + 1.0/18.0)*rho - RHO_0/18;
    L_BC_Fy = (m_xy_t90/2 - m_xx_t45/12 - m_yy_t45/3 + m_yz_t90/2 - m_zz_t45/12 + (omegaVar*m_xx_t45)/24 - (omegaVar*m_xy_t90)/4 + (omegaVar*m_yy_t45)/6 - (omegaVar*m_yz_t90)/4 + (omegaVar*m_zz_t45)/24 - 1.0/9.0)*rho + RHO_0/9;
    L_BC_Fz = (m_yy_t45/6 - m_xx_t45/12 - m_yz_t90/2 + m_zz_t45/6 + (omegaVar*m_xx_t45)/24 - (omegaVar*m_yy_t45)/12 + (omegaVar*m_yz_t90)/4 - (omegaVar*m_zz_t45)/12 + 1.0/18.0)*rho - RHO_0/18;
    #endif //_BC_FORCES 

    break;
case 0b0000010100110111:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[4] + pop[5] + pop[6] + pop[8] + pop[10] + pop[12] + pop[13] + pop[15] + pop[16] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[8] - pop[13]);
    m_xz_I = inv_rho_I * (pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[12] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);

    rho = (36*(107*rho_I + 12*m_xx_I*rho_I + 33*m_xz_I*rho_I + 72*m_yy_I*rho_I + 12*m_zz_I*rho_I - 12*omegaVar*m_xx_I*rho_I - 33*omegaVar*m_xz_I*rho_I - 72*omegaVar*m_yy_I*rho_I - 12*omegaVar*m_zz_I*rho_I))/(313*omegaVar + 2790);

    m_xx_t45 = (2772*m_xx_I + 648*m_xz_I - 108*m_yy_I - 18*m_zz_I + 302*omegaVar*m_xx_I + 48*omegaVar*m_xz_I - 66*omegaVar*m_yy_I - 11*omegaVar*m_zz_I + 72)/(27*(12*m_xx_I + 33*m_xz_I + 72*m_yy_I + 12*m_zz_I - 12*omegaVar*m_xx_I - 33*omegaVar*m_xz_I - 72*omegaVar*m_yy_I - 12*omegaVar*m_zz_I + 107));
    m_xy_t90 = (m_xy_I*(313*omegaVar + 2790))/(18*(12*m_xx_I + 33*m_xz_I + 72*m_yy_I + 12*m_zz_I - 12*omegaVar*m_xx_I - 33*omegaVar*m_xz_I - 72*omegaVar*m_yy_I - 12*omegaVar*m_zz_I + 107));
    m_xz_t90 = (432*m_xx_I + 3978*m_xz_I - 198*m_yy_I + 432*m_zz_I + 32*omegaVar*m_xx_I + 401*omegaVar*m_xz_I - 121*omegaVar*m_yy_I + 32*omegaVar*m_zz_I + 132)/(36*(12*m_xx_I + 33*m_xz_I + 72*m_yy_I + 12*m_zz_I - 12*omegaVar*m_xx_I - 33*omegaVar*m_xz_I - 72*omegaVar*m_yy_I - 12*omegaVar*m_zz_I + 107));
    m_yy_t45 = -(72*m_xx_I + 198*m_xz_I - 2358*m_yy_I + 72*m_zz_I + 44*omegaVar*m_xx_I + 121*omegaVar*m_xz_I - 49*omegaVar*m_yy_I + 44*omegaVar*m_zz_I - 288)/(18*(12*m_xx_I + 33*m_xz_I + 72*m_yy_I + 12*m_zz_I - 12*omegaVar*m_xx_I - 33*omegaVar*m_xz_I - 72*omegaVar*m_yy_I - 12*omegaVar*m_zz_I + 107));
    m_yz_t90 = (m_yz_I*(313*omegaVar + 2790))/(18*(12*m_xx_I + 33*m_xz_I + 72*m_yy_I + 12*m_zz_I - 12*omegaVar*m_xx_I - 33*omegaVar*m_xz_I - 72*omegaVar*m_yy_I - 12*omegaVar*m_zz_I + 107));
    m_zz_t45 = (648*m_xz_I - 18*m_xx_I - 108*m_yy_I + 2772*m_zz_I - 11*omegaVar*m_xx_I + 48*omegaVar*m_xz_I - 66*omegaVar*m_yy_I + 302*omegaVar*m_zz_I + 72)/(27*(12*m_xx_I + 33*m_xz_I + 72*m_yy_I + 12*m_zz_I - 12*omegaVar*m_xx_I - 33*omegaVar*m_xz_I - 72*omegaVar*m_yy_I - 12*omegaVar*m_zz_I + 107));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = (m_yy_t45/12 - m_xy_t90 - m_xz_t90/2 - m_xx_t45/6 - m_zz_t45/6 + (omegaVar*m_xx_t45)/12 + (omegaVar*m_xy_t90)/2 + (omegaVar*m_xz_t90)/4 - (omegaVar*m_yy_t45)/24 + (omegaVar*m_zz_t45)/12 - 1.0/18.0)*rho + RHO_0/18;
    L_BC_Fy = ((omegaVar*m_yy_t45)/2 - m_yy_t45 - 1.0/3.0)*rho + RHO_0/3;
    L_BC_Fz = (m_yy_t45/12 - m_xz_t90/2 - m_xx_t45/6 - m_yz_t90 - m_zz_t45/6 + (omegaVar*m_xx_t45)/12 + (omegaVar*m_xz_t90)/4 - (omegaVar*m_yy_t45)/24 + (omegaVar*m_yz_t90)/2 + (omegaVar*m_zz_t45)/12 - 1.0/18.0)*rho + RHO_0/18;
    #endif //_BC_FORCES 

    break;
case 0b0000010100111000:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);

    rho = (12*(15*rho_I - m_xx_I*rho_I + 2*m_yy_I*rho_I - 6*m_yz_I*rho_I + 2*m_zz_I*rho_I + omegaVar*m_xx_I*rho_I - 2*omegaVar*m_yy_I*rho_I + 6*omegaVar*m_yz_I*rho_I - 2*omegaVar*m_zz_I*rho_I))/(3*omegaVar + 172);

    m_xx_t45 = (792*m_xx_I - 36*m_yy_I + 108*m_yz_I - 36*m_zz_I + 13*omegaVar*m_xx_I + omegaVar*m_yy_I - 3*omegaVar*m_yz_I + omegaVar*m_zz_I - 12)/(54*(2*m_yy_I - m_xx_I - 6*m_yz_I + 2*m_zz_I + omegaVar*m_xx_I - 2*omegaVar*m_yy_I + 6*omegaVar*m_yz_I - 2*omegaVar*m_zz_I + 15));
    m_xy_t90 = (m_xy_I*(3*omegaVar + 172))/(12*(2*m_yy_I - m_xx_I - 6*m_yz_I + 2*m_zz_I + omegaVar*m_xx_I - 2*omegaVar*m_yy_I + 6*omegaVar*m_yz_I - 2*omegaVar*m_zz_I + 15));
    m_xz_t90 = (m_xz_I*(3*omegaVar + 172))/(12*(2*m_yy_I - m_xx_I - 6*m_yz_I + 2*m_zz_I + omegaVar*m_xx_I - 2*omegaVar*m_yy_I + 6*omegaVar*m_yz_I - 2*omegaVar*m_zz_I + 15));
    m_yy_t45 = (1692*m_yy_I - 72*m_xx_I - 432*m_yz_I + 144*m_zz_I + 2*omegaVar*m_xx_I + 23*omegaVar*m_yy_I + 12*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 48)/(108*(2*m_yy_I - m_xx_I - 6*m_yz_I + 2*m_zz_I + omegaVar*m_xx_I - 2*omegaVar*m_yy_I + 6*omegaVar*m_yz_I - 2*omegaVar*m_zz_I + 15));
    m_yz_t90 = (36*m_xx_I - 72*m_yy_I + 732*m_yz_I - 72*m_zz_I - omegaVar*m_xx_I + 2*omegaVar*m_yy_I + 3*omegaVar*m_yz_I + 2*omegaVar*m_zz_I - 24)/(36*(2*m_yy_I - m_xx_I - 6*m_yz_I + 2*m_zz_I + omegaVar*m_xx_I - 2*omegaVar*m_yy_I + 6*omegaVar*m_yz_I - 2*omegaVar*m_zz_I + 15));
    m_zz_t45 = (144*m_yy_I - 72*m_xx_I - 432*m_yz_I + 1692*m_zz_I + 2*omegaVar*m_xx_I - 4*omegaVar*m_yy_I + 12*omegaVar*m_yz_I + 23*omegaVar*m_zz_I + 48)/(108*(2*m_yy_I - m_xx_I - 6*m_yz_I + 2*m_zz_I + omegaVar*m_xx_I - 2*omegaVar*m_yy_I + 6*omegaVar*m_yz_I - 2*omegaVar*m_zz_I + 15));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = 0;
    L_BC_Fy = (m_xx_t45/12 - m_yy_t45/6 + m_yz_t90/2 - m_zz_t45/6 - (omegaVar*m_xx_t45)/24 + (omegaVar*m_yy_t45)/12 - (omegaVar*m_yz_t90)/4 + (omegaVar*m_zz_t45)/12 - 1.0/18.0)*rho + RHO_0/18;
    L_BC_Fz = (m_yy_t45/6 - m_xx_t45/12 - m_yz_t90/2 + m_zz_t45/6 + (omegaVar*m_xx_t45)/24 - (omegaVar*m_yy_t45)/12 + (omegaVar*m_yz_t90)/4 - (omegaVar*m_zz_t45)/12 + 1.0/18.0)*rho - RHO_0/18;
    #endif //_BC_FORCES 

    break;
case 0b0000010100111001:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);

    rho = (6*(15*rho_I + m_xx_I*rho_I + 6*m_xy_I*rho_I + 4*m_yy_I*rho_I - 6*m_yz_I*rho_I + m_zz_I*rho_I - omegaVar*m_xx_I*rho_I - 6*omegaVar*m_xy_I*rho_I - 4*omegaVar*m_yy_I*rho_I + 6*omegaVar*m_yz_I*rho_I - omegaVar*m_zz_I*rho_I))/(3*omegaVar + 82);

    m_xx_t45 = (2052*m_xx_I + 504*m_xy_I + 90*m_yy_I + 234*m_yz_I - 162*m_zz_I + 73*omegaVar*m_xx_I + 6*omegaVar*m_xy_I - 5*omegaVar*m_yy_I + 21*omegaVar*m_yz_I - 8*omegaVar*m_zz_I + 30)/(135*(m_xx_I + 6*m_xy_I + 4*m_yy_I - 6*m_yz_I + m_zz_I - omegaVar*m_xx_I - 6*omegaVar*m_xy_I - 4*omegaVar*m_yy_I + 6*omegaVar*m_yz_I - omegaVar*m_zz_I + 15));
    m_xy_t90 = (168*m_xx_I + 1746*m_xy_I + 180*m_yy_I - 24*m_yz_I - 78*m_zz_I + 2*omegaVar*m_xx_I + 39*omegaVar*m_xy_I - 10*omegaVar*m_yy_I + 24*omegaVar*m_yz_I - 7*omegaVar*m_zz_I + 60)/(90*(m_xx_I + 6*m_xy_I + 4*m_yy_I - 6*m_yz_I + m_zz_I - omegaVar*m_xx_I - 6*omegaVar*m_xy_I - 4*omegaVar*m_yy_I + 6*omegaVar*m_yz_I - omegaVar*m_zz_I + 15));
    m_xz_t90 = (m_xz_I*(3*omegaVar + 82))/(6*(m_xx_I + 6*m_xy_I + 4*m_yy_I - 6*m_yz_I + m_zz_I - omegaVar*m_xx_I - 6*omegaVar*m_xy_I - 4*omegaVar*m_yy_I + 6*omegaVar*m_yz_I - omegaVar*m_zz_I + 15));
    m_yy_t45 = (36*m_xx_I + 216*m_xy_I + 882*m_yy_I - 216*m_yz_I + 36*m_zz_I - 2*omegaVar*m_xx_I - 12*omegaVar*m_xy_I + 19*omegaVar*m_yy_I + 12*omegaVar*m_yz_I - 2*omegaVar*m_zz_I + 48)/(54*(m_xx_I + 6*m_xy_I + 4*m_yy_I - 6*m_yz_I + m_zz_I - omegaVar*m_xx_I - 6*omegaVar*m_xy_I - 4*omegaVar*m_yy_I + 6*omegaVar*m_yz_I - omegaVar*m_zz_I + 15));
    m_yz_t90 = (78*m_xx_I - 24*m_xy_I - 180*m_yy_I + 1746*m_yz_I - 168*m_zz_I + 7*omegaVar*m_xx_I + 24*omegaVar*m_xy_I + 10*omegaVar*m_yy_I + 39*omegaVar*m_yz_I - 2*omegaVar*m_zz_I - 60)/(90*(m_xx_I + 6*m_xy_I + 4*m_yy_I - 6*m_yz_I + m_zz_I - omegaVar*m_xx_I - 6*omegaVar*m_xy_I - 4*omegaVar*m_yy_I + 6*omegaVar*m_yz_I - omegaVar*m_zz_I + 15));
    m_zz_t45 = -(162*m_xx_I + 234*m_xy_I - 90*m_yy_I + 504*m_yz_I - 2052*m_zz_I + 8*omegaVar*m_xx_I + 21*omegaVar*m_xy_I + 5*omegaVar*m_yy_I + 6*omegaVar*m_yz_I - 73*omegaVar*m_zz_I - 30)/(135*(m_xx_I + 6*m_xy_I + 4*m_yy_I - 6*m_yz_I + m_zz_I - omegaVar*m_xx_I - 6*omegaVar*m_xy_I - 4*omegaVar*m_yy_I + 6*omegaVar*m_yz_I - omegaVar*m_zz_I + 15));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = (m_zz_t45/12 - m_xy_t90/2 - m_yy_t45/6 - m_xx_t45/6 + (omegaVar*m_xx_t45)/12 + (omegaVar*m_xy_t90)/4 + (omegaVar*m_yy_t45)/12 - (omegaVar*m_zz_t45)/24 - 1.0/18.0)*rho + RHO_0/18;
    L_BC_Fy = (m_yz_t90/2 - m_xy_t90/2 - m_yy_t45/3 - m_xx_t45/12 - m_zz_t45/12 + (omegaVar*m_xx_t45)/24 + (omegaVar*m_xy_t90)/4 + (omegaVar*m_yy_t45)/6 - (omegaVar*m_yz_t90)/4 + (omegaVar*m_zz_t45)/24 - 1.0/9.0)*rho + RHO_0/9;
    L_BC_Fz = (m_yy_t45/6 - m_xx_t45/12 - m_yz_t90/2 + m_zz_t45/6 + (omegaVar*m_xx_t45)/24 - (omegaVar*m_yy_t45)/12 + (omegaVar*m_yz_t90)/4 - (omegaVar*m_zz_t45)/12 + 1.0/18.0)*rho - RHO_0/18;
    #endif //_BC_FORCES 

    break;
case 0b0000010100111010:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[15] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[15] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] - (1.0/3.0)*pop[15] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[18]);

    rho = (12*(5*rho_I + m_xx_I*rho_I - 2*m_xy_I*rho_I - 2*m_xz_I*rho_I + m_yy_I*rho_I - 2*m_yz_I*rho_I + m_zz_I*rho_I - omegaVar*m_xx_I*rho_I + 2*omegaVar*m_xy_I*rho_I + 2*omegaVar*m_xz_I*rho_I - omegaVar*m_yy_I*rho_I + 2*omegaVar*m_yz_I*rho_I - omegaVar*m_zz_I*rho_I))/(3*omegaVar + 52);

    m_xx_t45 = (472*m_xx_I - 112*m_xy_I - 112*m_xz_I + 4*m_yy_I + 44*m_yz_I + 4*m_zz_I + 23*omegaVar*m_xx_I + 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - 4*omegaVar*m_yy_I + 11*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 20)/(90*(m_xx_I - 2*m_xy_I - 2*m_xz_I + m_yy_I - 2*m_yz_I + m_zz_I - omegaVar*m_xx_I + 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - omegaVar*m_yy_I + 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_xy_t90 = (1108*m_xy_I - 112*m_xx_I + 16*m_xz_I - 112*m_yy_I + 16*m_yz_I + 44*m_zz_I + 2*omegaVar*m_xx_I + 47*omegaVar*m_xy_I - 16*omegaVar*m_xz_I + 2*omegaVar*m_yy_I - 16*omegaVar*m_yz_I + 11*omegaVar*m_zz_I - 40)/(180*(m_xx_I - 2*m_xy_I - 2*m_xz_I + m_yy_I - 2*m_yz_I + m_zz_I - omegaVar*m_xx_I + 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - omegaVar*m_yy_I + 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_xz_t90 = (16*m_xy_I - 112*m_xx_I + 1108*m_xz_I + 44*m_yy_I + 16*m_yz_I - 112*m_zz_I + 2*omegaVar*m_xx_I - 16*omegaVar*m_xy_I + 47*omegaVar*m_xz_I + 11*omegaVar*m_yy_I - 16*omegaVar*m_yz_I + 2*omegaVar*m_zz_I - 40)/(180*(m_xx_I - 2*m_xy_I - 2*m_xz_I + m_yy_I - 2*m_yz_I + m_zz_I - omegaVar*m_xx_I + 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - omegaVar*m_yy_I + 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_yy_t45 = (4*m_xx_I - 112*m_xy_I + 44*m_xz_I + 472*m_yy_I - 112*m_yz_I + 4*m_zz_I - 4*omegaVar*m_xx_I + 2*omegaVar*m_xy_I + 11*omegaVar*m_xz_I + 23*omegaVar*m_yy_I + 2*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 20)/(90*(m_xx_I - 2*m_xy_I - 2*m_xz_I + m_yy_I - 2*m_yz_I + m_zz_I - omegaVar*m_xx_I + 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - omegaVar*m_yy_I + 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_yz_t90 = (44*m_xx_I + 16*m_xy_I + 16*m_xz_I - 112*m_yy_I + 1108*m_yz_I - 112*m_zz_I + 11*omegaVar*m_xx_I - 16*omegaVar*m_xy_I - 16*omegaVar*m_xz_I + 2*omegaVar*m_yy_I + 47*omegaVar*m_yz_I + 2*omegaVar*m_zz_I - 40)/(180*(m_xx_I - 2*m_xy_I - 2*m_xz_I + m_yy_I - 2*m_yz_I + m_zz_I - omegaVar*m_xx_I + 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - omegaVar*m_yy_I + 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_zz_t45 = (4*m_xx_I + 44*m_xy_I - 112*m_xz_I + 4*m_yy_I - 112*m_yz_I + 472*m_zz_I - 4*omegaVar*m_xx_I + 11*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - 4*omegaVar*m_yy_I + 2*omegaVar*m_yz_I + 23*omegaVar*m_zz_I + 20)/(90*(m_xx_I - 2*m_xy_I - 2*m_xz_I + m_yy_I - 2*m_yz_I + m_zz_I - omegaVar*m_xx_I + 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - omegaVar*m_yy_I + 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = (m_xx_t45/3 - m_xy_t90/2 - m_xz_t90/2 + m_yy_t45/12 + m_zz_t45/12 - (omegaVar*m_xx_t45)/6 + (omegaVar*m_xy_t90)/4 + (omegaVar*m_xz_t90)/4 - (omegaVar*m_yy_t45)/24 - (omegaVar*m_zz_t45)/24 + 1.0/9.0)*rho - RHO_0/9;
    L_BC_Fy = (m_xy_t90/2 - m_xx_t45/12 - m_yy_t45/3 + m_yz_t90/2 - m_zz_t45/12 + (omegaVar*m_xx_t45)/24 - (omegaVar*m_xy_t90)/4 + (omegaVar*m_yy_t45)/6 - (omegaVar*m_yz_t90)/4 + (omegaVar*m_zz_t45)/24 - 1.0/9.0)*rho + RHO_0/9;
    L_BC_Fz = (rho*(omegaVar - 2)*(m_xx_t45 - 2*m_xz_t90 - m_yy_t45 + 2*m_yz_t90))/8;
    #endif //_BC_FORCES 

    break;
case 0b0000010100111011:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[4] + pop[5] + pop[6] + pop[8] + pop[9] + pop[10] + pop[12] + pop[13] + pop[15] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[15] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[8] - pop[13]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] - (1.0/3.0)*pop[15] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[12] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[18]);

    rho = (36*(107*rho_I + 12*m_xx_I*rho_I - 33*m_xz_I*rho_I + 72*m_yy_I*rho_I + 12*m_zz_I*rho_I - 12*omegaVar*m_xx_I*rho_I + 33*omegaVar*m_xz_I*rho_I - 72*omegaVar*m_yy_I*rho_I - 12*omegaVar*m_zz_I*rho_I))/(313*omegaVar + 2790);

    m_xx_t45 = -(648*m_xz_I - 2772*m_xx_I + 108*m_yy_I + 18*m_zz_I - 302*omegaVar*m_xx_I + 48*omegaVar*m_xz_I + 66*omegaVar*m_yy_I + 11*omegaVar*m_zz_I - 72)/(27*(12*m_xx_I - 33*m_xz_I + 72*m_yy_I + 12*m_zz_I - 12*omegaVar*m_xx_I + 33*omegaVar*m_xz_I - 72*omegaVar*m_yy_I - 12*omegaVar*m_zz_I + 107));
    m_xy_t90 = (m_xy_I*(313*omegaVar + 2790))/(18*(12*m_xx_I - 33*m_xz_I + 72*m_yy_I + 12*m_zz_I - 12*omegaVar*m_xx_I + 33*omegaVar*m_xz_I - 72*omegaVar*m_yy_I - 12*omegaVar*m_zz_I + 107));
    m_xz_t90 = -(432*m_xx_I - 3978*m_xz_I - 198*m_yy_I + 432*m_zz_I + 32*omegaVar*m_xx_I - 401*omegaVar*m_xz_I - 121*omegaVar*m_yy_I + 32*omegaVar*m_zz_I + 132)/(36*(12*m_xx_I - 33*m_xz_I + 72*m_yy_I + 12*m_zz_I - 12*omegaVar*m_xx_I + 33*omegaVar*m_xz_I - 72*omegaVar*m_yy_I - 12*omegaVar*m_zz_I + 107));
    m_yy_t45 = (198*m_xz_I - 72*m_xx_I + 2358*m_yy_I - 72*m_zz_I - 44*omegaVar*m_xx_I + 121*omegaVar*m_xz_I + 49*omegaVar*m_yy_I - 44*omegaVar*m_zz_I + 288)/(18*(12*m_xx_I - 33*m_xz_I + 72*m_yy_I + 12*m_zz_I - 12*omegaVar*m_xx_I + 33*omegaVar*m_xz_I - 72*omegaVar*m_yy_I - 12*omegaVar*m_zz_I + 107));
    m_yz_t90 = (m_yz_I*(313*omegaVar + 2790))/(18*(12*m_xx_I - 33*m_xz_I + 72*m_yy_I + 12*m_zz_I - 12*omegaVar*m_xx_I + 33*omegaVar*m_xz_I - 72*omegaVar*m_yy_I - 12*omegaVar*m_zz_I + 107));
    m_zz_t45 = -(18*m_xx_I + 648*m_xz_I + 108*m_yy_I - 2772*m_zz_I + 11*omegaVar*m_xx_I + 48*omegaVar*m_xz_I + 66*omegaVar*m_yy_I - 302*omegaVar*m_zz_I - 72)/(27*(12*m_xx_I - 33*m_xz_I + 72*m_yy_I + 12*m_zz_I - 12*omegaVar*m_xx_I + 33*omegaVar*m_xz_I - 72*omegaVar*m_yy_I - 12*omegaVar*m_zz_I + 107));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = (m_xx_t45/6 - m_xy_t90 - m_xz_t90/2 - m_yy_t45/12 + m_zz_t45/6 - (omegaVar*m_xx_t45)/12 + (omegaVar*m_xy_t90)/2 + (omegaVar*m_xz_t90)/4 + (omegaVar*m_yy_t45)/24 - (omegaVar*m_zz_t45)/12 + 1.0/18.0)*rho - RHO_0/18;
    L_BC_Fy = ((omegaVar*m_yy_t45)/2 - m_yy_t45 - 1.0/3.0)*rho + RHO_0/3;
    L_BC_Fz = (m_xz_t90/2 - m_xx_t45/6 + m_yy_t45/12 - m_yz_t90 - m_zz_t45/6 + (omegaVar*m_xx_t45)/12 - (omegaVar*m_xz_t90)/4 - (omegaVar*m_yy_t45)/24 + (omegaVar*m_yz_t90)/2 + (omegaVar*m_zz_t45)/12 - 1.0/18.0)*rho + RHO_0/18;
    #endif //_BC_FORCES 

    break;
case 0b0000010100111100:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16]);

    rho = (6*(3*rho_I - m_xx_I*rho_I + 2*m_yy_I*rho_I - 6*m_yz_I*rho_I + 2*m_zz_I*rho_I + omegaVar*m_xx_I*rho_I - 2*omegaVar*m_yy_I*rho_I + 6*omegaVar*m_yz_I*rho_I - 2*omegaVar*m_zz_I*rho_I))/(3*omegaVar + 14);

    m_xx_t45 = (72*m_xx_I - 18*m_yy_I + 54*m_yz_I - 18*m_zz_I + 13*omegaVar*m_xx_I + omegaVar*m_yy_I - 3*omegaVar*m_yz_I + omegaVar*m_zz_I - 6)/(27*(2*m_yy_I - m_xx_I - 6*m_yz_I + 2*m_zz_I + omegaVar*m_xx_I - 2*omegaVar*m_yy_I + 6*omegaVar*m_yz_I - 2*omegaVar*m_zz_I + 3));
    m_xy_t90 = (m_xy_I*(3*omegaVar + 14))/(6*(2*m_yy_I - m_xx_I - 6*m_yz_I + 2*m_zz_I + omegaVar*m_xx_I - 2*omegaVar*m_yy_I + 6*omegaVar*m_yz_I - 2*omegaVar*m_zz_I + 3));
    m_xz_t90 = (m_xz_I*(3*omegaVar + 14))/(6*(2*m_yy_I - m_xx_I - 6*m_yz_I + 2*m_zz_I + omegaVar*m_xx_I - 2*omegaVar*m_yy_I + 6*omegaVar*m_yz_I - 2*omegaVar*m_zz_I + 3));
    m_yy_t45 = (198*m_yy_I - 36*m_xx_I - 216*m_yz_I + 72*m_zz_I + 2*omegaVar*m_xx_I + 23*omegaVar*m_yy_I + 12*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 24)/(54*(2*m_yy_I - m_xx_I - 6*m_yz_I + 2*m_zz_I + omegaVar*m_xx_I - 2*omegaVar*m_yy_I + 6*omegaVar*m_yz_I - 2*omegaVar*m_zz_I + 3));
    m_yz_t90 = (18*m_xx_I - 36*m_yy_I + 150*m_yz_I - 36*m_zz_I - omegaVar*m_xx_I + 2*omegaVar*m_yy_I + 3*omegaVar*m_yz_I + 2*omegaVar*m_zz_I - 12)/(18*(2*m_yy_I - m_xx_I - 6*m_yz_I + 2*m_zz_I + omegaVar*m_xx_I - 2*omegaVar*m_yy_I + 6*omegaVar*m_yz_I - 2*omegaVar*m_zz_I + 3));
    m_zz_t45 = (72*m_yy_I - 36*m_xx_I - 216*m_yz_I + 198*m_zz_I + 2*omegaVar*m_xx_I - 4*omegaVar*m_yy_I + 12*omegaVar*m_yz_I + 23*omegaVar*m_zz_I + 24)/(54*(2*m_yy_I - m_xx_I - 6*m_yz_I + 2*m_zz_I + omegaVar*m_xx_I - 2*omegaVar*m_yy_I + 6*omegaVar*m_yz_I - 2*omegaVar*m_zz_I + 3));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = 0;
    L_BC_Fy = 0;
    L_BC_Fz = 0;
    #endif //_BC_FORCES 

    break;
case 0b0000010100111101:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[8] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16]);
    m_xy_I = inv_rho_I * (pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16]);

    rho = (3*(15*rho_I - m_xx_I*rho_I + 6*m_xy_I*rho_I + 6*m_xz_I*rho_I + 11*m_yy_I*rho_I - 30*m_yz_I*rho_I + 11*m_zz_I*rho_I + omegaVar*m_xx_I*rho_I - 6*omegaVar*m_xy_I*rho_I - 6*omegaVar*m_xz_I*rho_I - 11*omegaVar*m_yy_I*rho_I + 30*omegaVar*m_yz_I*rho_I - 11*omegaVar*m_zz_I*rho_I))/(9*omegaVar + 31);

    m_xx_t45 = (2*(180*m_xx_I + 36*m_xy_I + 36*m_xz_I - 27*m_yy_I + 99*m_yz_I - 27*m_zz_I + 52*omegaVar*m_xx_I + 12*omegaVar*m_xy_I + 12*omegaVar*m_xz_I - 5*omegaVar*m_yy_I + 21*omegaVar*m_yz_I - 5*omegaVar*m_zz_I - 3))/(27*(6*m_xy_I - m_xx_I + 6*m_xz_I + 11*m_yy_I - 30*m_yz_I + 11*m_zz_I + omegaVar*m_xx_I - 6*omegaVar*m_xy_I - 6*omegaVar*m_xz_I - 11*omegaVar*m_yy_I + 30*omegaVar*m_yz_I - 11*omegaVar*m_zz_I + 15));
    m_xy_t90 = (60*m_xx_I + 663*m_xy_I + 12*m_xz_I + 84*m_yy_I - 60*m_yz_I - 9*m_zz_I + 20*omegaVar*m_xx_I + 177*omegaVar*m_xy_I - 12*omegaVar*m_xz_I - 4*omegaVar*m_yy_I + 60*omegaVar*m_yz_I - 31*omegaVar*m_zz_I + 30)/(45*(6*m_xy_I - m_xx_I + 6*m_xz_I + 11*m_yy_I - 30*m_yz_I + 11*m_zz_I + omegaVar*m_xx_I - 6*omegaVar*m_xy_I - 6*omegaVar*m_xz_I - 11*omegaVar*m_yy_I + 30*omegaVar*m_yz_I - 11*omegaVar*m_zz_I + 15));
    m_xz_t90 = (60*m_xx_I + 12*m_xy_I + 663*m_xz_I - 9*m_yy_I - 60*m_yz_I + 84*m_zz_I + 20*omegaVar*m_xx_I - 12*omegaVar*m_xy_I + 177*omegaVar*m_xz_I - 31*omegaVar*m_yy_I + 60*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 30)/(45*(6*m_xy_I - m_xx_I + 6*m_xz_I + 11*m_yy_I - 30*m_yz_I + 11*m_zz_I + omegaVar*m_xx_I - 6*omegaVar*m_xy_I - 6*omegaVar*m_xz_I - 11*omegaVar*m_yy_I + 30*omegaVar*m_yz_I - 11*omegaVar*m_zz_I + 15));
    m_yy_t45 = -(2*(135*m_xx_I - 252*m_xy_I + 27*m_xz_I - 1206*m_yy_I + 1260*m_yz_I - 369*m_zz_I + 25*omegaVar*m_xx_I + 12*omegaVar*m_xy_I + 93*omegaVar*m_xz_I - 194*omegaVar*m_yy_I - 60*omegaVar*m_yz_I + 49*omegaVar*m_zz_I - 165))/(135*(6*m_xy_I - m_xx_I + 6*m_xz_I + 11*m_yy_I - 30*m_yz_I + 11*m_zz_I + omegaVar*m_xx_I - 6*omegaVar*m_xy_I - 6*omegaVar*m_xz_I - 11*omegaVar*m_yy_I + 30*omegaVar*m_yz_I - 11*omegaVar*m_zz_I + 15));
    m_yz_t90 = (33*m_xx_I - 12*m_xy_I - 12*m_xz_I - 84*m_yy_I + 339*m_yz_I - 84*m_zz_I + 7*omegaVar*m_xx_I + 12*omegaVar*m_xy_I + 12*omegaVar*m_xz_I + 4*omegaVar*m_yy_I + 21*omegaVar*m_yz_I + 4*omegaVar*m_zz_I - 30)/(9*(6*m_xy_I - m_xx_I + 6*m_xz_I + 11*m_yy_I - 30*m_yz_I + 11*m_zz_I + omegaVar*m_xx_I - 6*omegaVar*m_xy_I - 6*omegaVar*m_xz_I - 11*omegaVar*m_yy_I + 30*omegaVar*m_yz_I - 11*omegaVar*m_zz_I + 15));
    m_zz_t45 = -(2*(135*m_xx_I + 27*m_xy_I - 252*m_xz_I - 369*m_yy_I + 1260*m_yz_I - 1206*m_zz_I + 25*omegaVar*m_xx_I + 93*omegaVar*m_xy_I + 12*omegaVar*m_xz_I + 49*omegaVar*m_yy_I - 60*omegaVar*m_yz_I - 194*omegaVar*m_zz_I - 165))/(135*(6*m_xy_I - m_xx_I + 6*m_xz_I + 11*m_yy_I - 30*m_yz_I + 11*m_zz_I + omegaVar*m_xx_I - 6*omegaVar*m_xy_I - 6*omegaVar*m_xz_I - 11*omegaVar*m_yy_I + 30*omegaVar*m_yz_I - 11*omegaVar*m_zz_I + 15));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = ((omegaVar*m_xx_t45)/6 - m_xy_t90/2 - m_xz_t90/2 - m_yy_t45/12 - m_zz_t45/12 - m_xx_t45/3 + (omegaVar*m_xy_t90)/4 + (omegaVar*m_xz_t90)/4 + (omegaVar*m_yy_t45)/24 + (omegaVar*m_zz_t45)/24 - 1.0/9.0)*rho + RHO_0/9;
    L_BC_Fy = (m_zz_t45/12 - m_xy_t90/2 - m_yy_t45/6 - m_xx_t45/6 + (omegaVar*m_xx_t45)/12 + (omegaVar*m_xy_t90)/4 + (omegaVar*m_yy_t45)/12 - (omegaVar*m_zz_t45)/24 - 1.0/18.0)*rho + RHO_0/18;
    L_BC_Fz = (m_yy_t45/12 - m_xz_t90/2 - m_xx_t45/6 - m_zz_t45/6 + (omegaVar*m_xx_t45)/12 + (omegaVar*m_xz_t90)/4 - (omegaVar*m_yy_t45)/24 + (omegaVar*m_zz_t45)/12 - 1.0/18.0)*rho + RHO_0/18;
    #endif //_BC_FORCES 

    break;
case 0b0000010100111110:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[15];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[15]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] - (1.0/3.0)*pop[15]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] + (2.0/3.0)*pop[15]);

    rho = (3*(15*rho_I - m_xx_I*rho_I - 6*m_xy_I*rho_I - 6*m_xz_I*rho_I + 11*m_yy_I*rho_I - 30*m_yz_I*rho_I + 11*m_zz_I*rho_I + omegaVar*m_xx_I*rho_I + 6*omegaVar*m_xy_I*rho_I + 6*omegaVar*m_xz_I*rho_I - 11*omegaVar*m_yy_I*rho_I + 30*omegaVar*m_yz_I*rho_I - 11*omegaVar*m_zz_I*rho_I))/(9*omegaVar + 31);

    m_xx_t45 = -(2*(36*m_xy_I - 180*m_xx_I + 36*m_xz_I + 27*m_yy_I - 99*m_yz_I + 27*m_zz_I - 52*omegaVar*m_xx_I + 12*omegaVar*m_xy_I + 12*omegaVar*m_xz_I + 5*omegaVar*m_yy_I - 21*omegaVar*m_yz_I + 5*omegaVar*m_zz_I + 3))/(27*(11*m_yy_I - 6*m_xy_I - 6*m_xz_I - m_xx_I - 30*m_yz_I + 11*m_zz_I + omegaVar*m_xx_I + 6*omegaVar*m_xy_I + 6*omegaVar*m_xz_I - 11*omegaVar*m_yy_I + 30*omegaVar*m_yz_I - 11*omegaVar*m_zz_I + 15));
    m_xy_t90 = (663*m_xy_I - 60*m_xx_I + 12*m_xz_I - 84*m_yy_I + 60*m_yz_I + 9*m_zz_I - 20*omegaVar*m_xx_I + 177*omegaVar*m_xy_I - 12*omegaVar*m_xz_I + 4*omegaVar*m_yy_I - 60*omegaVar*m_yz_I + 31*omegaVar*m_zz_I - 30)/(45*(11*m_yy_I - 6*m_xy_I - 6*m_xz_I - m_xx_I - 30*m_yz_I + 11*m_zz_I + omegaVar*m_xx_I + 6*omegaVar*m_xy_I + 6*omegaVar*m_xz_I - 11*omegaVar*m_yy_I + 30*omegaVar*m_yz_I - 11*omegaVar*m_zz_I + 15));
    m_xz_t90 = (12*m_xy_I - 60*m_xx_I + 663*m_xz_I + 9*m_yy_I + 60*m_yz_I - 84*m_zz_I - 20*omegaVar*m_xx_I - 12*omegaVar*m_xy_I + 177*omegaVar*m_xz_I + 31*omegaVar*m_yy_I - 60*omegaVar*m_yz_I + 4*omegaVar*m_zz_I - 30)/(45*(11*m_yy_I - 6*m_xy_I - 6*m_xz_I - m_xx_I - 30*m_yz_I + 11*m_zz_I + omegaVar*m_xx_I + 6*omegaVar*m_xy_I + 6*omegaVar*m_xz_I - 11*omegaVar*m_yy_I + 30*omegaVar*m_yz_I - 11*omegaVar*m_zz_I + 15));
    m_yy_t45 = (2*(27*m_xz_I - 252*m_xy_I - 135*m_xx_I + 1206*m_yy_I - 1260*m_yz_I + 369*m_zz_I - 25*omegaVar*m_xx_I + 12*omegaVar*m_xy_I + 93*omegaVar*m_xz_I + 194*omegaVar*m_yy_I + 60*omegaVar*m_yz_I - 49*omegaVar*m_zz_I + 165))/(135*(11*m_yy_I - 6*m_xy_I - 6*m_xz_I - m_xx_I - 30*m_yz_I + 11*m_zz_I + omegaVar*m_xx_I + 6*omegaVar*m_xy_I + 6*omegaVar*m_xz_I - 11*omegaVar*m_yy_I + 30*omegaVar*m_yz_I - 11*omegaVar*m_zz_I + 15));
    m_yz_t90 = (33*m_xx_I + 12*m_xy_I + 12*m_xz_I - 84*m_yy_I + 339*m_yz_I - 84*m_zz_I + 7*omegaVar*m_xx_I - 12*omegaVar*m_xy_I - 12*omegaVar*m_xz_I + 4*omegaVar*m_yy_I + 21*omegaVar*m_yz_I + 4*omegaVar*m_zz_I - 30)/(9*(11*m_yy_I - 6*m_xy_I - 6*m_xz_I - m_xx_I - 30*m_yz_I + 11*m_zz_I + omegaVar*m_xx_I + 6*omegaVar*m_xy_I + 6*omegaVar*m_xz_I - 11*omegaVar*m_yy_I + 30*omegaVar*m_yz_I - 11*omegaVar*m_zz_I + 15));
    m_zz_t45 = (2*(27*m_xy_I - 135*m_xx_I - 252*m_xz_I + 369*m_yy_I - 1260*m_yz_I + 1206*m_zz_I - 25*omegaVar*m_xx_I + 93*omegaVar*m_xy_I + 12*omegaVar*m_xz_I - 49*omegaVar*m_yy_I + 60*omegaVar*m_yz_I + 194*omegaVar*m_zz_I + 165))/(135*(11*m_yy_I - 6*m_xy_I - 6*m_xz_I - m_xx_I - 30*m_yz_I + 11*m_zz_I + omegaVar*m_xx_I + 6*omegaVar*m_xy_I + 6*omegaVar*m_xz_I - 11*omegaVar*m_yy_I + 30*omegaVar*m_yz_I - 11*omegaVar*m_zz_I + 15));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = (m_xx_t45/3 - m_xy_t90/2 - m_xz_t90/2 + m_yy_t45/12 + m_zz_t45/12 - (omegaVar*m_xx_t45)/6 + (omegaVar*m_xy_t90)/4 + (omegaVar*m_xz_t90)/4 - (omegaVar*m_yy_t45)/24 - (omegaVar*m_zz_t45)/24 + 1.0/9.0)*rho - RHO_0/9;
    L_BC_Fy = (m_xy_t90/2 - m_xx_t45/6 - m_yy_t45/6 + m_zz_t45/12 + (omegaVar*m_xx_t45)/12 - (omegaVar*m_xy_t90)/4 + (omegaVar*m_yy_t45)/12 - (omegaVar*m_zz_t45)/24 - 1.0/18.0)*rho + RHO_0/18;
    L_BC_Fz = (m_xz_t90/2 - m_xx_t45/6 + m_yy_t45/12 - m_zz_t45/6 + (omegaVar*m_xx_t45)/12 - (omegaVar*m_xz_t90)/4 - (omegaVar*m_yy_t45)/24 + (omegaVar*m_zz_t45)/12 - 1.0/18.0)*rho + RHO_0/18;
    #endif //_BC_FORCES 

    break;
case 0b0000010100111111:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[4] + pop[6] + pop[8] + pop[10] + pop[12] + pop[13] + pop[15];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[15]);
    m_xy_I = inv_rho_I * (pop[8] - pop[13]);
    m_xz_I = inv_rho_I * (pop[10] - pop[15]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] - (1.0/3.0)*pop[15]);
    m_yz_I = inv_rho_I * (pop[12]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] + (2.0/3.0)*pop[15]);

    rho = (36*(23*rho_I - 6*m_xx_I*rho_I + 24*m_yy_I*rho_I - 57*m_yz_I*rho_I + 24*m_zz_I*rho_I + 6*omegaVar*m_xx_I*rho_I - 24*omegaVar*m_yy_I*rho_I + 57*omegaVar*m_yz_I*rho_I - 24*omegaVar*m_zz_I*rho_I))/(5*(43*omegaVar + 72));

    m_xx_t45 = -(72*m_yy_I - 288*m_xx_I - 216*m_yz_I + 72*m_zz_I - 162*omegaVar*m_xx_I + 3*omegaVar*m_yy_I - 34*omegaVar*m_yz_I + 3*omegaVar*m_zz_I + 24)/(18*(24*m_yy_I - 6*m_xx_I - 57*m_yz_I + 24*m_zz_I + 6*omegaVar*m_xx_I - 24*omegaVar*m_yy_I + 57*omegaVar*m_yz_I - 24*omegaVar*m_zz_I + 23));
    m_xy_t90 = (5*m_xy_I*(43*omegaVar + 72))/(18*(24*m_yy_I - 6*m_xx_I - 57*m_yz_I + 24*m_zz_I + 6*omegaVar*m_xx_I - 24*omegaVar*m_yy_I + 57*omegaVar*m_yz_I - 24*omegaVar*m_zz_I + 23));
    m_xz_t90 = (5*m_xz_I*(43*omegaVar + 72))/(18*(24*m_yy_I - 6*m_xx_I - 57*m_yz_I + 24*m_zz_I + 6*omegaVar*m_xx_I - 24*omegaVar*m_yy_I + 57*omegaVar*m_yz_I - 24*omegaVar*m_zz_I + 23));
    m_yy_t45 = (936*m_yy_I - 144*m_xx_I - 1008*m_yz_I + 216*m_zz_I - 6*omegaVar*m_xx_I + 239*omegaVar*m_yy_I + 158*omegaVar*m_yz_I - 191*omegaVar*m_zz_I + 192)/(36*(24*m_yy_I - 6*m_xx_I - 57*m_yz_I + 24*m_zz_I + 6*omegaVar*m_xx_I - 24*omegaVar*m_yy_I + 57*omegaVar*m_yz_I - 24*omegaVar*m_zz_I + 23));
    m_yz_t90 = (216*m_xx_I - 504*m_yy_I + 2412*m_yz_I - 504*m_zz_I + 34*omegaVar*m_xx_I + 79*omegaVar*m_yy_I + 538*omegaVar*m_yz_I + 79*omegaVar*m_zz_I - 228)/(36*(24*m_yy_I - 6*m_xx_I - 57*m_yz_I + 24*m_zz_I + 6*omegaVar*m_xx_I - 24*omegaVar*m_yy_I + 57*omegaVar*m_yz_I - 24*omegaVar*m_zz_I + 23));
    m_zz_t45 = (216*m_yy_I - 144*m_xx_I - 1008*m_yz_I + 936*m_zz_I - 6*omegaVar*m_xx_I - 191*omegaVar*m_yy_I + 158*omegaVar*m_yz_I + 239*omegaVar*m_zz_I + 192)/(36*(24*m_yy_I - 6*m_xx_I - 57*m_yz_I + 24*m_zz_I + 6*omegaVar*m_xx_I - 24*omegaVar*m_yy_I + 57*omegaVar*m_yz_I - 24*omegaVar*m_zz_I + 23));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = (rho*(m_xy_t90 + m_xz_t90)*(omegaVar - 2))/2;
    L_BC_Fy = (m_zz_t45/6 - (5*m_yy_t45)/6 - m_yz_t90/2 - m_xx_t45/12 + (omegaVar*m_xx_t45)/24 + (5*omegaVar*m_yy_t45)/12 + (omegaVar*m_yz_t90)/4 - (omegaVar*m_zz_t45)/12 - 5/18)*rho + (5*RHO_0)/18;
    L_BC_Fz = (m_yy_t45/6 - m_xx_t45/12 - m_yz_t90/2 - (5*m_zz_t45)/6 + (omegaVar*m_xx_t45)/24 - (omegaVar*m_yy_t45)/12 + (omegaVar*m_yz_t90)/4 + (5*omegaVar*m_zz_t45)/12 - 5/18)*rho + (5*RHO_0)/18;
    #endif //_BC_FORCES 

    break;
case 0b0000010101000000:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = rho_I;

    m_xx_t45 = m_xx_I;
    m_xy_t90 = m_xy_I;
    m_xz_t90 = m_xz_I;
    m_yy_t45 = m_yy_I;
    m_yz_t90 = m_yz_I;
    m_zz_t45 = m_zz_I;

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = 0;
    L_BC_Fy = 0;
    L_BC_Fz = 0;
    #endif //_BC_FORCES 

    break;
case 0b0000010101000001:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = rho_I;

    m_xx_t45 = m_xx_I;
    m_xy_t90 = m_xy_I;
    m_xz_t90 = m_xz_I;
    m_yy_t45 = m_yy_I;
    m_yz_t90 = m_yz_I;
    m_zz_t45 = m_zz_I;

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = 0;
    L_BC_Fy = 0;
    L_BC_Fz = 0;
    #endif //_BC_FORCES 

    break;
case 0b0000010101000010:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = rho_I;

    m_xx_t45 = m_xx_I;
    m_xy_t90 = m_xy_I;
    m_xz_t90 = m_xz_I;
    m_yy_t45 = m_yy_I;
    m_yz_t90 = m_yz_I;
    m_zz_t45 = m_zz_I;

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = 0;
    L_BC_Fy = 0;
    L_BC_Fz = 0;
    #endif //_BC_FORCES 

    break;
case 0b0000010101000011:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (12*(15*rho_I - m_xx_I*rho_I + 2*m_yy_I*rho_I + 6*m_yz_I*rho_I + 2*m_zz_I*rho_I + omegaVar*m_xx_I*rho_I - 2*omegaVar*m_yy_I*rho_I - 6*omegaVar*m_yz_I*rho_I - 2*omegaVar*m_zz_I*rho_I))/(3*omegaVar + 172);

    m_xx_t45 = (792*m_xx_I - 36*m_yy_I - 108*m_yz_I - 36*m_zz_I + 13*omegaVar*m_xx_I + omegaVar*m_yy_I + 3*omegaVar*m_yz_I + omegaVar*m_zz_I - 12)/(54*(2*m_yy_I - m_xx_I + 6*m_yz_I + 2*m_zz_I + omegaVar*m_xx_I - 2*omegaVar*m_yy_I - 6*omegaVar*m_yz_I - 2*omegaVar*m_zz_I + 15));
    m_xy_t90 = (m_xy_I*(3*omegaVar + 172))/(12*(2*m_yy_I - m_xx_I + 6*m_yz_I + 2*m_zz_I + omegaVar*m_xx_I - 2*omegaVar*m_yy_I - 6*omegaVar*m_yz_I - 2*omegaVar*m_zz_I + 15));
    m_xz_t90 = (m_xz_I*(3*omegaVar + 172))/(12*(2*m_yy_I - m_xx_I + 6*m_yz_I + 2*m_zz_I + omegaVar*m_xx_I - 2*omegaVar*m_yy_I - 6*omegaVar*m_yz_I - 2*omegaVar*m_zz_I + 15));
    m_yy_t45 = (1692*m_yy_I - 72*m_xx_I + 432*m_yz_I + 144*m_zz_I + 2*omegaVar*m_xx_I + 23*omegaVar*m_yy_I - 12*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 48)/(108*(2*m_yy_I - m_xx_I + 6*m_yz_I + 2*m_zz_I + omegaVar*m_xx_I - 2*omegaVar*m_yy_I - 6*omegaVar*m_yz_I - 2*omegaVar*m_zz_I + 15));
    m_yz_t90 = (72*m_yy_I - 36*m_xx_I + 732*m_yz_I + 72*m_zz_I + omegaVar*m_xx_I - 2*omegaVar*m_yy_I + 3*omegaVar*m_yz_I - 2*omegaVar*m_zz_I + 24)/(36*(2*m_yy_I - m_xx_I + 6*m_yz_I + 2*m_zz_I + omegaVar*m_xx_I - 2*omegaVar*m_yy_I - 6*omegaVar*m_yz_I - 2*omegaVar*m_zz_I + 15));
    m_zz_t45 = (144*m_yy_I - 72*m_xx_I + 432*m_yz_I + 1692*m_zz_I + 2*omegaVar*m_xx_I - 4*omegaVar*m_yy_I - 12*omegaVar*m_yz_I + 23*omegaVar*m_zz_I + 48)/(108*(2*m_yy_I - m_xx_I + 6*m_yz_I + 2*m_zz_I + omegaVar*m_xx_I - 2*omegaVar*m_yy_I - 6*omegaVar*m_yz_I - 2*omegaVar*m_zz_I + 15));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = 0;
    L_BC_Fy = (m_xx_t45/12 - m_yy_t45/6 - m_yz_t90/2 - m_zz_t45/6 - (omegaVar*m_xx_t45)/24 + (omegaVar*m_yy_t45)/12 + (omegaVar*m_yz_t90)/4 + (omegaVar*m_zz_t45)/12 - 1.0/18.0)*rho + RHO_0/18;
    L_BC_Fz = (m_xx_t45/12 - m_yy_t45/6 - m_yz_t90/2 - m_zz_t45/6 - (omegaVar*m_xx_t45)/24 + (omegaVar*m_yy_t45)/12 + (omegaVar*m_yz_t90)/4 + (omegaVar*m_zz_t45)/12 - 1.0/18.0)*rho + RHO_0/18;
    #endif //_BC_FORCES 

    break;
case 0b0000010101000100:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (12*(15*rho_I + 2*m_xx_I*rho_I - 6*m_xy_I*rho_I + 2*m_yy_I*rho_I - m_zz_I*rho_I - 2*omegaVar*m_xx_I*rho_I + 6*omegaVar*m_xy_I*rho_I - 2*omegaVar*m_yy_I*rho_I + omegaVar*m_zz_I*rho_I))/(3*omegaVar + 172);

    m_xx_t45 = (1692*m_xx_I - 432*m_xy_I + 144*m_yy_I - 72*m_zz_I + 23*omegaVar*m_xx_I + 12*omegaVar*m_xy_I - 4*omegaVar*m_yy_I + 2*omegaVar*m_zz_I + 48)/(108*(2*m_xx_I - 6*m_xy_I + 2*m_yy_I - m_zz_I - 2*omegaVar*m_xx_I + 6*omegaVar*m_xy_I - 2*omegaVar*m_yy_I + omegaVar*m_zz_I + 15));
    m_xy_t90 = (732*m_xy_I - 72*m_xx_I - 72*m_yy_I + 36*m_zz_I + 2*omegaVar*m_xx_I + 3*omegaVar*m_xy_I + 2*omegaVar*m_yy_I - omegaVar*m_zz_I - 24)/(36*(2*m_xx_I - 6*m_xy_I + 2*m_yy_I - m_zz_I - 2*omegaVar*m_xx_I + 6*omegaVar*m_xy_I - 2*omegaVar*m_yy_I + omegaVar*m_zz_I + 15));
    m_xz_t90 = (m_xz_I*(3*omegaVar + 172))/(12*(2*m_xx_I - 6*m_xy_I + 2*m_yy_I - m_zz_I - 2*omegaVar*m_xx_I + 6*omegaVar*m_xy_I - 2*omegaVar*m_yy_I + omegaVar*m_zz_I + 15));
    m_yy_t45 = (144*m_xx_I - 432*m_xy_I + 1692*m_yy_I - 72*m_zz_I - 4*omegaVar*m_xx_I + 12*omegaVar*m_xy_I + 23*omegaVar*m_yy_I + 2*omegaVar*m_zz_I + 48)/(108*(2*m_xx_I - 6*m_xy_I + 2*m_yy_I - m_zz_I - 2*omegaVar*m_xx_I + 6*omegaVar*m_xy_I - 2*omegaVar*m_yy_I + omegaVar*m_zz_I + 15));
    m_yz_t90 = (m_yz_I*(3*omegaVar + 172))/(12*(2*m_xx_I - 6*m_xy_I + 2*m_yy_I - m_zz_I - 2*omegaVar*m_xx_I + 6*omegaVar*m_xy_I - 2*omegaVar*m_yy_I + omegaVar*m_zz_I + 15));
    m_zz_t45 = (108*m_xy_I - 36*m_xx_I - 36*m_yy_I + 792*m_zz_I + omegaVar*m_xx_I - 3*omegaVar*m_xy_I + omegaVar*m_yy_I + 13*omegaVar*m_zz_I - 12)/(54*(2*m_xx_I - 6*m_xy_I + 2*m_yy_I - m_zz_I - 2*omegaVar*m_xx_I + 6*omegaVar*m_xy_I - 2*omegaVar*m_yy_I + omegaVar*m_zz_I + 15));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = (m_xy_t90/2 - m_xx_t45/6 - m_yy_t45/6 + m_zz_t45/12 + (omegaVar*m_xx_t45)/12 - (omegaVar*m_xy_t90)/4 + (omegaVar*m_yy_t45)/12 - (omegaVar*m_zz_t45)/24 - 1.0/18.0)*rho + RHO_0/18;
    L_BC_Fy = (m_xx_t45/6 - m_xy_t90/2 + m_yy_t45/6 - m_zz_t45/12 - (omegaVar*m_xx_t45)/12 + (omegaVar*m_xy_t90)/4 - (omegaVar*m_yy_t45)/12 + (omegaVar*m_zz_t45)/24 + 1.0/18.0)*rho - RHO_0/18;
    L_BC_Fz = 0;
    #endif //_BC_FORCES 

    break;
case 0b0000010101000101:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[10] + pop[11] + pop[12] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[14]);
    m_xz_I = inv_rho_I * (pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (6*(15*rho_I + 4*m_xx_I*rho_I - 6*m_xy_I*rho_I + 6*m_xz_I*rho_I + m_yy_I*rho_I + m_zz_I*rho_I - 4*omegaVar*m_xx_I*rho_I + 6*omegaVar*m_xy_I*rho_I - 6*omegaVar*m_xz_I*rho_I - omegaVar*m_yy_I*rho_I - omegaVar*m_zz_I*rho_I))/(3*omegaVar + 82);

    m_xx_t45 = (882*m_xx_I - 216*m_xy_I + 216*m_xz_I + 36*m_yy_I + 36*m_zz_I + 19*omegaVar*m_xx_I + 12*omegaVar*m_xy_I - 12*omegaVar*m_xz_I - 2*omegaVar*m_yy_I - 2*omegaVar*m_zz_I + 48)/(54*(4*m_xx_I - 6*m_xy_I + 6*m_xz_I + m_yy_I + m_zz_I - 4*omegaVar*m_xx_I + 6*omegaVar*m_xy_I - 6*omegaVar*m_xz_I - omegaVar*m_yy_I - omegaVar*m_zz_I + 15));
    m_xy_t90 = (1746*m_xy_I - 180*m_xx_I - 24*m_xz_I - 168*m_yy_I + 78*m_zz_I + 10*omegaVar*m_xx_I + 39*omegaVar*m_xy_I + 24*omegaVar*m_xz_I - 2*omegaVar*m_yy_I + 7*omegaVar*m_zz_I - 60)/(90*(4*m_xx_I - 6*m_xy_I + 6*m_xz_I + m_yy_I + m_zz_I - 4*omegaVar*m_xx_I + 6*omegaVar*m_xy_I - 6*omegaVar*m_xz_I - omegaVar*m_yy_I - omegaVar*m_zz_I + 15));
    m_xz_t90 = (180*m_xx_I - 24*m_xy_I + 1746*m_xz_I - 78*m_yy_I + 168*m_zz_I - 10*omegaVar*m_xx_I + 24*omegaVar*m_xy_I + 39*omegaVar*m_xz_I - 7*omegaVar*m_yy_I + 2*omegaVar*m_zz_I + 60)/(90*(4*m_xx_I - 6*m_xy_I + 6*m_xz_I + m_yy_I + m_zz_I - 4*omegaVar*m_xx_I + 6*omegaVar*m_xy_I - 6*omegaVar*m_xz_I - omegaVar*m_yy_I - omegaVar*m_zz_I + 15));
    m_yy_t45 = -(504*m_xy_I - 90*m_xx_I + 234*m_xz_I - 2052*m_yy_I + 162*m_zz_I + 5*omegaVar*m_xx_I + 6*omegaVar*m_xy_I + 21*omegaVar*m_xz_I - 73*omegaVar*m_yy_I + 8*omegaVar*m_zz_I - 30)/(135*(4*m_xx_I - 6*m_xy_I + 6*m_xz_I + m_yy_I + m_zz_I - 4*omegaVar*m_xx_I + 6*omegaVar*m_xy_I - 6*omegaVar*m_xz_I - omegaVar*m_yy_I - omegaVar*m_zz_I + 15));
    m_yz_t90 = (m_yz_I*(3*omegaVar + 82))/(6*(4*m_xx_I - 6*m_xy_I + 6*m_xz_I + m_yy_I + m_zz_I - 4*omegaVar*m_xx_I + 6*omegaVar*m_xy_I - 6*omegaVar*m_xz_I - omegaVar*m_yy_I - omegaVar*m_zz_I + 15));
    m_zz_t45 = (90*m_xx_I + 234*m_xy_I + 504*m_xz_I - 162*m_yy_I + 2052*m_zz_I - 5*omegaVar*m_xx_I + 21*omegaVar*m_xy_I + 6*omegaVar*m_xz_I - 8*omegaVar*m_yy_I + 73*omegaVar*m_zz_I + 30)/(135*(4*m_xx_I - 6*m_xy_I + 6*m_xz_I + m_yy_I + m_zz_I - 4*omegaVar*m_xx_I + 6*omegaVar*m_xy_I - 6*omegaVar*m_xz_I - omegaVar*m_yy_I - omegaVar*m_zz_I + 15));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = (m_xy_t90/2 - m_xx_t45/3 - m_xz_t90/2 - m_yy_t45/12 - m_zz_t45/12 + (omegaVar*m_xx_t45)/6 - (omegaVar*m_xy_t90)/4 + (omegaVar*m_xz_t90)/4 + (omegaVar*m_yy_t45)/24 + (omegaVar*m_zz_t45)/24 - 1.0/9.0)*rho + RHO_0/9;
    L_BC_Fy = (m_xx_t45/6 - m_xy_t90/2 + m_yy_t45/6 - m_zz_t45/12 - (omegaVar*m_xx_t45)/12 + (omegaVar*m_xy_t90)/4 - (omegaVar*m_yy_t45)/12 + (omegaVar*m_zz_t45)/24 + 1.0/18.0)*rho - RHO_0/18;
    L_BC_Fz = (m_yy_t45/12 - m_xz_t90/2 - m_xx_t45/6 - m_zz_t45/6 + (omegaVar*m_xx_t45)/12 + (omegaVar*m_xz_t90)/4 - (omegaVar*m_yy_t45)/24 + (omegaVar*m_zz_t45)/12 - 1.0/18.0)*rho + RHO_0/18;
    #endif //_BC_FORCES 

    break;
case 0b0000010101000110:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (12*(15*rho_I + 2*m_xx_I*rho_I - 6*m_xy_I*rho_I + 2*m_yy_I*rho_I - m_zz_I*rho_I - 2*omegaVar*m_xx_I*rho_I + 6*omegaVar*m_xy_I*rho_I - 2*omegaVar*m_yy_I*rho_I + omegaVar*m_zz_I*rho_I))/(3*omegaVar + 172);

    m_xx_t45 = (1692*m_xx_I - 432*m_xy_I + 144*m_yy_I - 72*m_zz_I + 23*omegaVar*m_xx_I + 12*omegaVar*m_xy_I - 4*omegaVar*m_yy_I + 2*omegaVar*m_zz_I + 48)/(108*(2*m_xx_I - 6*m_xy_I + 2*m_yy_I - m_zz_I - 2*omegaVar*m_xx_I + 6*omegaVar*m_xy_I - 2*omegaVar*m_yy_I + omegaVar*m_zz_I + 15));
    m_xy_t90 = (732*m_xy_I - 72*m_xx_I - 72*m_yy_I + 36*m_zz_I + 2*omegaVar*m_xx_I + 3*omegaVar*m_xy_I + 2*omegaVar*m_yy_I - omegaVar*m_zz_I - 24)/(36*(2*m_xx_I - 6*m_xy_I + 2*m_yy_I - m_zz_I - 2*omegaVar*m_xx_I + 6*omegaVar*m_xy_I - 2*omegaVar*m_yy_I + omegaVar*m_zz_I + 15));
    m_xz_t90 = (m_xz_I*(3*omegaVar + 172))/(12*(2*m_xx_I - 6*m_xy_I + 2*m_yy_I - m_zz_I - 2*omegaVar*m_xx_I + 6*omegaVar*m_xy_I - 2*omegaVar*m_yy_I + omegaVar*m_zz_I + 15));
    m_yy_t45 = (144*m_xx_I - 432*m_xy_I + 1692*m_yy_I - 72*m_zz_I - 4*omegaVar*m_xx_I + 12*omegaVar*m_xy_I + 23*omegaVar*m_yy_I + 2*omegaVar*m_zz_I + 48)/(108*(2*m_xx_I - 6*m_xy_I + 2*m_yy_I - m_zz_I - 2*omegaVar*m_xx_I + 6*omegaVar*m_xy_I - 2*omegaVar*m_yy_I + omegaVar*m_zz_I + 15));
    m_yz_t90 = (m_yz_I*(3*omegaVar + 172))/(12*(2*m_xx_I - 6*m_xy_I + 2*m_yy_I - m_zz_I - 2*omegaVar*m_xx_I + 6*omegaVar*m_xy_I - 2*omegaVar*m_yy_I + omegaVar*m_zz_I + 15));
    m_zz_t45 = (108*m_xy_I - 36*m_xx_I - 36*m_yy_I + 792*m_zz_I + omegaVar*m_xx_I - 3*omegaVar*m_xy_I + omegaVar*m_yy_I + 13*omegaVar*m_zz_I - 12)/(54*(2*m_xx_I - 6*m_xy_I + 2*m_yy_I - m_zz_I - 2*omegaVar*m_xx_I + 6*omegaVar*m_xy_I - 2*omegaVar*m_yy_I + omegaVar*m_zz_I + 15));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = (m_xy_t90/2 - m_xx_t45/6 - m_yy_t45/6 + m_zz_t45/12 + (omegaVar*m_xx_t45)/12 - (omegaVar*m_xy_t90)/4 + (omegaVar*m_yy_t45)/12 - (omegaVar*m_zz_t45)/24 - 1.0/18.0)*rho + RHO_0/18;
    L_BC_Fy = (m_xx_t45/6 - m_xy_t90/2 + m_yy_t45/6 - m_zz_t45/12 - (omegaVar*m_xx_t45)/12 + (omegaVar*m_xy_t90)/4 - (omegaVar*m_yy_t45)/12 + (omegaVar*m_zz_t45)/24 + 1.0/18.0)*rho - RHO_0/18;
    L_BC_Fz = 0;
    #endif //_BC_FORCES 

    break;
case 0b0000010101000111:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[10] + pop[12] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[14]);
    m_xz_I = inv_rho_I * (pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (12*(5*rho_I + m_xx_I*rho_I - 2*m_xy_I*rho_I + 2*m_xz_I*rho_I + m_yy_I*rho_I + 2*m_yz_I*rho_I + m_zz_I*rho_I - omegaVar*m_xx_I*rho_I + 2*omegaVar*m_xy_I*rho_I - 2*omegaVar*m_xz_I*rho_I - omegaVar*m_yy_I*rho_I - 2*omegaVar*m_yz_I*rho_I - omegaVar*m_zz_I*rho_I))/(3*omegaVar + 52);

    m_xx_t45 = (472*m_xx_I - 112*m_xy_I + 112*m_xz_I + 4*m_yy_I - 44*m_yz_I + 4*m_zz_I + 23*omegaVar*m_xx_I + 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - 4*omegaVar*m_yy_I - 11*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 20)/(90*(m_xx_I - 2*m_xy_I + 2*m_xz_I + m_yy_I + 2*m_yz_I + m_zz_I - omegaVar*m_xx_I + 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - omegaVar*m_yy_I - 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_xy_t90 = (1108*m_xy_I - 112*m_xx_I - 16*m_xz_I - 112*m_yy_I - 16*m_yz_I + 44*m_zz_I + 2*omegaVar*m_xx_I + 47*omegaVar*m_xy_I + 16*omegaVar*m_xz_I + 2*omegaVar*m_yy_I + 16*omegaVar*m_yz_I + 11*omegaVar*m_zz_I - 40)/(180*(m_xx_I - 2*m_xy_I + 2*m_xz_I + m_yy_I + 2*m_yz_I + m_zz_I - omegaVar*m_xx_I + 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - omegaVar*m_yy_I - 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_xz_t90 = (112*m_xx_I - 16*m_xy_I + 1108*m_xz_I - 44*m_yy_I + 16*m_yz_I + 112*m_zz_I - 2*omegaVar*m_xx_I + 16*omegaVar*m_xy_I + 47*omegaVar*m_xz_I - 11*omegaVar*m_yy_I - 16*omegaVar*m_yz_I - 2*omegaVar*m_zz_I + 40)/(180*(m_xx_I - 2*m_xy_I + 2*m_xz_I + m_yy_I + 2*m_yz_I + m_zz_I - omegaVar*m_xx_I + 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - omegaVar*m_yy_I - 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_yy_t45 = (4*m_xx_I - 112*m_xy_I - 44*m_xz_I + 472*m_yy_I + 112*m_yz_I + 4*m_zz_I - 4*omegaVar*m_xx_I + 2*omegaVar*m_xy_I - 11*omegaVar*m_xz_I + 23*omegaVar*m_yy_I - 2*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 20)/(90*(m_xx_I - 2*m_xy_I + 2*m_xz_I + m_yy_I + 2*m_yz_I + m_zz_I - omegaVar*m_xx_I + 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - omegaVar*m_yy_I - 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_yz_t90 = (16*m_xz_I - 16*m_xy_I - 44*m_xx_I + 112*m_yy_I + 1108*m_yz_I + 112*m_zz_I - 11*omegaVar*m_xx_I + 16*omegaVar*m_xy_I - 16*omegaVar*m_xz_I - 2*omegaVar*m_yy_I + 47*omegaVar*m_yz_I - 2*omegaVar*m_zz_I + 40)/(180*(m_xx_I - 2*m_xy_I + 2*m_xz_I + m_yy_I + 2*m_yz_I + m_zz_I - omegaVar*m_xx_I + 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - omegaVar*m_yy_I - 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_zz_t45 = (4*m_xx_I + 44*m_xy_I + 112*m_xz_I + 4*m_yy_I + 112*m_yz_I + 472*m_zz_I - 4*omegaVar*m_xx_I + 11*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - 4*omegaVar*m_yy_I - 2*omegaVar*m_yz_I + 23*omegaVar*m_zz_I + 20)/(90*(m_xx_I - 2*m_xy_I + 2*m_xz_I + m_yy_I + 2*m_yz_I + m_zz_I - omegaVar*m_xx_I + 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - omegaVar*m_yy_I - 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = (m_xy_t90/2 - m_xx_t45/3 - m_xz_t90/2 - m_yy_t45/12 - m_zz_t45/12 + (omegaVar*m_xx_t45)/6 - (omegaVar*m_xy_t90)/4 + (omegaVar*m_xz_t90)/4 + (omegaVar*m_yy_t45)/24 + (omegaVar*m_zz_t45)/24 - 1.0/9.0)*rho + RHO_0/9;
    L_BC_Fy = (rho*(omegaVar - 2)*(2*m_xy_t90 - m_xx_t45 + 2*m_yz_t90 + m_zz_t45))/8;
    L_BC_Fz = ((omegaVar*m_xx_t45)/24 - m_xz_t90/2 - m_yy_t45/12 - m_yz_t90/2 - m_zz_t45/3 - m_xx_t45/12 + (omegaVar*m_xz_t90)/4 + (omegaVar*m_yy_t45)/24 + (omegaVar*m_yz_t90)/4 + (omegaVar*m_zz_t45)/6 - 1.0/9.0)*rho + RHO_0/9;
    #endif //_BC_FORCES 

    break;
case 0b0000010101001000:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = rho_I;

    m_xx_t45 = m_xx_I;
    m_xy_t90 = m_xy_I;
    m_xz_t90 = m_xz_I;
    m_yy_t45 = m_yy_I;
    m_yz_t90 = m_yz_I;
    m_zz_t45 = m_zz_I;

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = 0;
    L_BC_Fy = 0;
    L_BC_Fz = 0;
    #endif //_BC_FORCES 

    break;
case 0b0000010101001001:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = rho_I;

    m_xx_t45 = m_xx_I;
    m_xy_t90 = m_xy_I;
    m_xz_t90 = m_xz_I;
    m_yy_t45 = m_yy_I;
    m_yz_t90 = m_yz_I;
    m_zz_t45 = m_zz_I;

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = 0;
    L_BC_Fy = 0;
    L_BC_Fz = 0;
    #endif //_BC_FORCES 

    break;
case 0b0000010101001010:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (12*(15*rho_I + 2*m_xx_I*rho_I - 6*m_xz_I*rho_I - m_yy_I*rho_I + 2*m_zz_I*rho_I - 2*omegaVar*m_xx_I*rho_I + 6*omegaVar*m_xz_I*rho_I + omegaVar*m_yy_I*rho_I - 2*omegaVar*m_zz_I*rho_I))/(3*omegaVar + 172);

    m_xx_t45 = (1692*m_xx_I - 432*m_xz_I - 72*m_yy_I + 144*m_zz_I + 23*omegaVar*m_xx_I + 12*omegaVar*m_xz_I + 2*omegaVar*m_yy_I - 4*omegaVar*m_zz_I + 48)/(108*(2*m_xx_I - 6*m_xz_I - m_yy_I + 2*m_zz_I - 2*omegaVar*m_xx_I + 6*omegaVar*m_xz_I + omegaVar*m_yy_I - 2*omegaVar*m_zz_I + 15));
    m_xy_t90 = (m_xy_I*(3*omegaVar + 172))/(12*(2*m_xx_I - 6*m_xz_I - m_yy_I + 2*m_zz_I - 2*omegaVar*m_xx_I + 6*omegaVar*m_xz_I + omegaVar*m_yy_I - 2*omegaVar*m_zz_I + 15));
    m_xz_t90 = (732*m_xz_I - 72*m_xx_I + 36*m_yy_I - 72*m_zz_I + 2*omegaVar*m_xx_I + 3*omegaVar*m_xz_I - omegaVar*m_yy_I + 2*omegaVar*m_zz_I - 24)/(36*(2*m_xx_I - 6*m_xz_I - m_yy_I + 2*m_zz_I - 2*omegaVar*m_xx_I + 6*omegaVar*m_xz_I + omegaVar*m_yy_I - 2*omegaVar*m_zz_I + 15));
    m_yy_t45 = (108*m_xz_I - 36*m_xx_I + 792*m_yy_I - 36*m_zz_I + omegaVar*m_xx_I - 3*omegaVar*m_xz_I + 13*omegaVar*m_yy_I + omegaVar*m_zz_I - 12)/(54*(2*m_xx_I - 6*m_xz_I - m_yy_I + 2*m_zz_I - 2*omegaVar*m_xx_I + 6*omegaVar*m_xz_I + omegaVar*m_yy_I - 2*omegaVar*m_zz_I + 15));
    m_yz_t90 = (m_yz_I*(3*omegaVar + 172))/(12*(2*m_xx_I - 6*m_xz_I - m_yy_I + 2*m_zz_I - 2*omegaVar*m_xx_I + 6*omegaVar*m_xz_I + omegaVar*m_yy_I - 2*omegaVar*m_zz_I + 15));
    m_zz_t45 = (144*m_xx_I - 432*m_xz_I - 72*m_yy_I + 1692*m_zz_I - 4*omegaVar*m_xx_I + 12*omegaVar*m_xz_I + 2*omegaVar*m_yy_I + 23*omegaVar*m_zz_I + 48)/(108*(2*m_xx_I - 6*m_xz_I - m_yy_I + 2*m_zz_I - 2*omegaVar*m_xx_I + 6*omegaVar*m_xz_I + omegaVar*m_yy_I - 2*omegaVar*m_zz_I + 15));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = (m_xx_t45/6 - m_xz_t90/2 - m_yy_t45/12 + m_zz_t45/6 - (omegaVar*m_xx_t45)/12 + (omegaVar*m_xz_t90)/4 + (omegaVar*m_yy_t45)/24 - (omegaVar*m_zz_t45)/12 + 1.0/18.0)*rho - RHO_0/18;
    L_BC_Fy = 0;
    L_BC_Fz = (m_xz_t90/2 - m_xx_t45/6 + m_yy_t45/12 - m_zz_t45/6 + (omegaVar*m_xx_t45)/12 - (omegaVar*m_xz_t90)/4 - (omegaVar*m_yy_t45)/24 + (omegaVar*m_zz_t45)/12 - 1.0/18.0)*rho + RHO_0/18;
    #endif //_BC_FORCES 

    break;
case 0b0000010101001011:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[12] + pop[13] + pop[14] + pop[15] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (6*(15*rho_I + m_xx_I*rho_I - 6*m_xz_I*rho_I + m_yy_I*rho_I + 6*m_yz_I*rho_I + 4*m_zz_I*rho_I - omegaVar*m_xx_I*rho_I + 6*omegaVar*m_xz_I*rho_I - omegaVar*m_yy_I*rho_I - 6*omegaVar*m_yz_I*rho_I - 4*omegaVar*m_zz_I*rho_I))/(3*omegaVar + 82);

    m_xx_t45 = -(504*m_xz_I - 2052*m_xx_I + 162*m_yy_I + 234*m_yz_I - 90*m_zz_I - 73*omegaVar*m_xx_I + 6*omegaVar*m_xz_I + 8*omegaVar*m_yy_I + 21*omegaVar*m_yz_I + 5*omegaVar*m_zz_I - 30)/(135*(m_xx_I - 6*m_xz_I + m_yy_I + 6*m_yz_I + 4*m_zz_I - omegaVar*m_xx_I + 6*omegaVar*m_xz_I - omegaVar*m_yy_I - 6*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 15));
    m_xy_t90 = (m_xy_I*(3*omegaVar + 82))/(6*(m_xx_I - 6*m_xz_I + m_yy_I + 6*m_yz_I + 4*m_zz_I - omegaVar*m_xx_I + 6*omegaVar*m_xz_I - omegaVar*m_yy_I - 6*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 15));
    m_xz_t90 = (1746*m_xz_I - 168*m_xx_I + 78*m_yy_I - 24*m_yz_I - 180*m_zz_I - 2*omegaVar*m_xx_I + 39*omegaVar*m_xz_I + 7*omegaVar*m_yy_I + 24*omegaVar*m_yz_I + 10*omegaVar*m_zz_I - 60)/(90*(m_xx_I - 6*m_xz_I + m_yy_I + 6*m_yz_I + 4*m_zz_I - omegaVar*m_xx_I + 6*omegaVar*m_xz_I - omegaVar*m_yy_I - 6*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 15));
    m_yy_t45 = (234*m_xz_I - 162*m_xx_I + 2052*m_yy_I + 504*m_yz_I + 90*m_zz_I - 8*omegaVar*m_xx_I + 21*omegaVar*m_xz_I + 73*omegaVar*m_yy_I + 6*omegaVar*m_yz_I - 5*omegaVar*m_zz_I + 30)/(135*(m_xx_I - 6*m_xz_I + m_yy_I + 6*m_yz_I + 4*m_zz_I - omegaVar*m_xx_I + 6*omegaVar*m_xz_I - omegaVar*m_yy_I - 6*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 15));
    m_yz_t90 = (168*m_yy_I - 24*m_xz_I - 78*m_xx_I + 1746*m_yz_I + 180*m_zz_I - 7*omegaVar*m_xx_I + 24*omegaVar*m_xz_I + 2*omegaVar*m_yy_I + 39*omegaVar*m_yz_I - 10*omegaVar*m_zz_I + 60)/(90*(m_xx_I - 6*m_xz_I + m_yy_I + 6*m_yz_I + 4*m_zz_I - omegaVar*m_xx_I + 6*omegaVar*m_xz_I - omegaVar*m_yy_I - 6*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 15));
    m_zz_t45 = (36*m_xx_I - 216*m_xz_I + 36*m_yy_I + 216*m_yz_I + 882*m_zz_I - 2*omegaVar*m_xx_I + 12*omegaVar*m_xz_I - 2*omegaVar*m_yy_I - 12*omegaVar*m_yz_I + 19*omegaVar*m_zz_I + 48)/(54*(m_xx_I - 6*m_xz_I + m_yy_I + 6*m_yz_I + 4*m_zz_I - omegaVar*m_xx_I + 6*omegaVar*m_xz_I - omegaVar*m_yy_I - 6*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 15));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = (m_xx_t45/6 - m_xz_t90/2 - m_yy_t45/12 + m_zz_t45/6 - (omegaVar*m_xx_t45)/12 + (omegaVar*m_xz_t90)/4 + (omegaVar*m_yy_t45)/24 - (omegaVar*m_zz_t45)/12 + 1.0/18.0)*rho - RHO_0/18;
    L_BC_Fy = (m_xx_t45/12 - m_yy_t45/6 - m_yz_t90/2 - m_zz_t45/6 - (omegaVar*m_xx_t45)/24 + (omegaVar*m_yy_t45)/12 + (omegaVar*m_yz_t90)/4 + (omegaVar*m_zz_t45)/12 - 1.0/18.0)*rho + RHO_0/18;
    L_BC_Fz = (m_xz_t90/2 - m_xx_t45/12 - m_yy_t45/12 - m_yz_t90/2 - m_zz_t45/3 + (omegaVar*m_xx_t45)/24 - (omegaVar*m_xz_t90)/4 + (omegaVar*m_yy_t45)/24 + (omegaVar*m_yz_t90)/4 + (omegaVar*m_zz_t45)/6 - 1.0/9.0)*rho + RHO_0/9;
    #endif //_BC_FORCES 

    break;
case 0b0000010101001100:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[14] + pop[15] + pop[16] + pop[17];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17]);

    rho = (6*(15*rho_I + m_xx_I*rho_I - 6*m_xy_I*rho_I + 4*m_yy_I*rho_I - 6*m_yz_I*rho_I + m_zz_I*rho_I - omegaVar*m_xx_I*rho_I + 6*omegaVar*m_xy_I*rho_I - 4*omegaVar*m_yy_I*rho_I + 6*omegaVar*m_yz_I*rho_I - omegaVar*m_zz_I*rho_I))/(3*omegaVar + 82);

    m_xx_t45 = (2052*m_xx_I - 504*m_xy_I + 90*m_yy_I + 234*m_yz_I - 162*m_zz_I + 73*omegaVar*m_xx_I - 6*omegaVar*m_xy_I - 5*omegaVar*m_yy_I + 21*omegaVar*m_yz_I - 8*omegaVar*m_zz_I + 30)/(135*(m_xx_I - 6*m_xy_I + 4*m_yy_I - 6*m_yz_I + m_zz_I - omegaVar*m_xx_I + 6*omegaVar*m_xy_I - 4*omegaVar*m_yy_I + 6*omegaVar*m_yz_I - omegaVar*m_zz_I + 15));
    m_xy_t90 = (1746*m_xy_I - 168*m_xx_I - 180*m_yy_I + 24*m_yz_I + 78*m_zz_I - 2*omegaVar*m_xx_I + 39*omegaVar*m_xy_I + 10*omegaVar*m_yy_I - 24*omegaVar*m_yz_I + 7*omegaVar*m_zz_I - 60)/(90*(m_xx_I - 6*m_xy_I + 4*m_yy_I - 6*m_yz_I + m_zz_I - omegaVar*m_xx_I + 6*omegaVar*m_xy_I - 4*omegaVar*m_yy_I + 6*omegaVar*m_yz_I - omegaVar*m_zz_I + 15));
    m_xz_t90 = (m_xz_I*(3*omegaVar + 82))/(6*(m_xx_I - 6*m_xy_I + 4*m_yy_I - 6*m_yz_I + m_zz_I - omegaVar*m_xx_I + 6*omegaVar*m_xy_I - 4*omegaVar*m_yy_I + 6*omegaVar*m_yz_I - omegaVar*m_zz_I + 15));
    m_yy_t45 = (36*m_xx_I - 216*m_xy_I + 882*m_yy_I - 216*m_yz_I + 36*m_zz_I - 2*omegaVar*m_xx_I + 12*omegaVar*m_xy_I + 19*omegaVar*m_yy_I + 12*omegaVar*m_yz_I - 2*omegaVar*m_zz_I + 48)/(54*(m_xx_I - 6*m_xy_I + 4*m_yy_I - 6*m_yz_I + m_zz_I - omegaVar*m_xx_I + 6*omegaVar*m_xy_I - 4*omegaVar*m_yy_I + 6*omegaVar*m_yz_I - omegaVar*m_zz_I + 15));
    m_yz_t90 = (78*m_xx_I + 24*m_xy_I - 180*m_yy_I + 1746*m_yz_I - 168*m_zz_I + 7*omegaVar*m_xx_I - 24*omegaVar*m_xy_I + 10*omegaVar*m_yy_I + 39*omegaVar*m_yz_I - 2*omegaVar*m_zz_I - 60)/(90*(m_xx_I - 6*m_xy_I + 4*m_yy_I - 6*m_yz_I + m_zz_I - omegaVar*m_xx_I + 6*omegaVar*m_xy_I - 4*omegaVar*m_yy_I + 6*omegaVar*m_yz_I - omegaVar*m_zz_I + 15));
    m_zz_t45 = (234*m_xy_I - 162*m_xx_I + 90*m_yy_I - 504*m_yz_I + 2052*m_zz_I - 8*omegaVar*m_xx_I + 21*omegaVar*m_xy_I - 5*omegaVar*m_yy_I - 6*omegaVar*m_yz_I + 73*omegaVar*m_zz_I + 30)/(135*(m_xx_I - 6*m_xy_I + 4*m_yy_I - 6*m_yz_I + m_zz_I - omegaVar*m_xx_I + 6*omegaVar*m_xy_I - 4*omegaVar*m_yy_I + 6*omegaVar*m_yz_I - omegaVar*m_zz_I + 15));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = (m_xy_t90/2 - m_xx_t45/6 - m_yy_t45/6 + m_zz_t45/12 + (omegaVar*m_xx_t45)/12 - (omegaVar*m_xy_t90)/4 + (omegaVar*m_yy_t45)/12 - (omegaVar*m_zz_t45)/24 - 1.0/18.0)*rho + RHO_0/18;
    L_BC_Fy = (m_xx_t45/12 - m_xy_t90/2 + m_yy_t45/3 - m_yz_t90/2 + m_zz_t45/12 - (omegaVar*m_xx_t45)/24 + (omegaVar*m_xy_t90)/4 - (omegaVar*m_yy_t45)/6 + (omegaVar*m_yz_t90)/4 - (omegaVar*m_zz_t45)/24 + 1.0/9.0)*rho - RHO_0/9;
    L_BC_Fz = (m_xx_t45/12 - m_yy_t45/6 + m_yz_t90/2 - m_zz_t45/6 - (omegaVar*m_xx_t45)/24 + (omegaVar*m_yy_t45)/12 - (omegaVar*m_yz_t90)/4 + (omegaVar*m_zz_t45)/12 - 1.0/18.0)*rho + RHO_0/18;
    #endif //_BC_FORCES 

    break;
case 0b0000010101001101:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[10] + pop[11] + pop[12] + pop[14] + pop[15] + pop[16] + pop[17];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[14]);
    m_xz_I = inv_rho_I * (pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17]);

    rho = (12*(5*rho_I + m_xx_I*rho_I - 2*m_xy_I*rho_I + 2*m_xz_I*rho_I + m_yy_I*rho_I - 2*m_yz_I*rho_I + m_zz_I*rho_I - omegaVar*m_xx_I*rho_I + 2*omegaVar*m_xy_I*rho_I - 2*omegaVar*m_xz_I*rho_I - omegaVar*m_yy_I*rho_I + 2*omegaVar*m_yz_I*rho_I - omegaVar*m_zz_I*rho_I))/(3*omegaVar + 52);

    m_xx_t45 = (472*m_xx_I - 112*m_xy_I + 112*m_xz_I + 4*m_yy_I + 44*m_yz_I + 4*m_zz_I + 23*omegaVar*m_xx_I + 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - 4*omegaVar*m_yy_I + 11*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 20)/(90*(m_xx_I - 2*m_xy_I + 2*m_xz_I + m_yy_I - 2*m_yz_I + m_zz_I - omegaVar*m_xx_I + 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - omegaVar*m_yy_I + 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_xy_t90 = (1108*m_xy_I - 112*m_xx_I - 16*m_xz_I - 112*m_yy_I + 16*m_yz_I + 44*m_zz_I + 2*omegaVar*m_xx_I + 47*omegaVar*m_xy_I + 16*omegaVar*m_xz_I + 2*omegaVar*m_yy_I - 16*omegaVar*m_yz_I + 11*omegaVar*m_zz_I - 40)/(180*(m_xx_I - 2*m_xy_I + 2*m_xz_I + m_yy_I - 2*m_yz_I + m_zz_I - omegaVar*m_xx_I + 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - omegaVar*m_yy_I + 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_xz_t90 = (112*m_xx_I - 16*m_xy_I + 1108*m_xz_I - 44*m_yy_I - 16*m_yz_I + 112*m_zz_I - 2*omegaVar*m_xx_I + 16*omegaVar*m_xy_I + 47*omegaVar*m_xz_I - 11*omegaVar*m_yy_I + 16*omegaVar*m_yz_I - 2*omegaVar*m_zz_I + 40)/(180*(m_xx_I - 2*m_xy_I + 2*m_xz_I + m_yy_I - 2*m_yz_I + m_zz_I - omegaVar*m_xx_I + 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - omegaVar*m_yy_I + 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_yy_t45 = (4*m_xx_I - 112*m_xy_I - 44*m_xz_I + 472*m_yy_I - 112*m_yz_I + 4*m_zz_I - 4*omegaVar*m_xx_I + 2*omegaVar*m_xy_I - 11*omegaVar*m_xz_I + 23*omegaVar*m_yy_I + 2*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 20)/(90*(m_xx_I - 2*m_xy_I + 2*m_xz_I + m_yy_I - 2*m_yz_I + m_zz_I - omegaVar*m_xx_I + 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - omegaVar*m_yy_I + 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_yz_t90 = (44*m_xx_I + 16*m_xy_I - 16*m_xz_I - 112*m_yy_I + 1108*m_yz_I - 112*m_zz_I + 11*omegaVar*m_xx_I - 16*omegaVar*m_xy_I + 16*omegaVar*m_xz_I + 2*omegaVar*m_yy_I + 47*omegaVar*m_yz_I + 2*omegaVar*m_zz_I - 40)/(180*(m_xx_I - 2*m_xy_I + 2*m_xz_I + m_yy_I - 2*m_yz_I + m_zz_I - omegaVar*m_xx_I + 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - omegaVar*m_yy_I + 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_zz_t45 = (4*m_xx_I + 44*m_xy_I + 112*m_xz_I + 4*m_yy_I - 112*m_yz_I + 472*m_zz_I - 4*omegaVar*m_xx_I + 11*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - 4*omegaVar*m_yy_I + 2*omegaVar*m_yz_I + 23*omegaVar*m_zz_I + 20)/(90*(m_xx_I - 2*m_xy_I + 2*m_xz_I + m_yy_I - 2*m_yz_I + m_zz_I - omegaVar*m_xx_I + 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - omegaVar*m_yy_I + 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = (m_xy_t90/2 - m_xx_t45/3 - m_xz_t90/2 - m_yy_t45/12 - m_zz_t45/12 + (omegaVar*m_xx_t45)/6 - (omegaVar*m_xy_t90)/4 + (omegaVar*m_xz_t90)/4 + (omegaVar*m_yy_t45)/24 + (omegaVar*m_zz_t45)/24 - 1.0/9.0)*rho + RHO_0/9;
    L_BC_Fy = (m_xx_t45/12 - m_xy_t90/2 + m_yy_t45/3 - m_yz_t90/2 + m_zz_t45/12 - (omegaVar*m_xx_t45)/24 + (omegaVar*m_xy_t90)/4 - (omegaVar*m_yy_t45)/6 + (omegaVar*m_yz_t90)/4 - (omegaVar*m_zz_t45)/24 + 1.0/9.0)*rho - RHO_0/9;
    L_BC_Fz = (m_yz_t90/2 - m_xz_t90/2 - m_yy_t45/12 - m_xx_t45/12 - m_zz_t45/3 + (omegaVar*m_xx_t45)/24 + (omegaVar*m_xz_t90)/4 + (omegaVar*m_yy_t45)/24 - (omegaVar*m_yz_t90)/4 + (omegaVar*m_zz_t45)/6 - 1.0/9.0)*rho + RHO_0/9;
    #endif //_BC_FORCES 

    break;
case 0b0000010101001110:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[14] + pop[15] + pop[17];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] - (1.0/3.0)*pop[17]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] + (2.0/3.0)*pop[17]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[17]);

    rho = (12*(5*rho_I + m_xx_I*rho_I - 2*m_xy_I*rho_I - 2*m_xz_I*rho_I + m_yy_I*rho_I - 2*m_yz_I*rho_I + m_zz_I*rho_I - omegaVar*m_xx_I*rho_I + 2*omegaVar*m_xy_I*rho_I + 2*omegaVar*m_xz_I*rho_I - omegaVar*m_yy_I*rho_I + 2*omegaVar*m_yz_I*rho_I - omegaVar*m_zz_I*rho_I))/(3*omegaVar + 52);

    m_xx_t45 = (472*m_xx_I - 112*m_xy_I - 112*m_xz_I + 4*m_yy_I + 44*m_yz_I + 4*m_zz_I + 23*omegaVar*m_xx_I + 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - 4*omegaVar*m_yy_I + 11*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 20)/(90*(m_xx_I - 2*m_xy_I - 2*m_xz_I + m_yy_I - 2*m_yz_I + m_zz_I - omegaVar*m_xx_I + 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - omegaVar*m_yy_I + 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_xy_t90 = (1108*m_xy_I - 112*m_xx_I + 16*m_xz_I - 112*m_yy_I + 16*m_yz_I + 44*m_zz_I + 2*omegaVar*m_xx_I + 47*omegaVar*m_xy_I - 16*omegaVar*m_xz_I + 2*omegaVar*m_yy_I - 16*omegaVar*m_yz_I + 11*omegaVar*m_zz_I - 40)/(180*(m_xx_I - 2*m_xy_I - 2*m_xz_I + m_yy_I - 2*m_yz_I + m_zz_I - omegaVar*m_xx_I + 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - omegaVar*m_yy_I + 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_xz_t90 = (16*m_xy_I - 112*m_xx_I + 1108*m_xz_I + 44*m_yy_I + 16*m_yz_I - 112*m_zz_I + 2*omegaVar*m_xx_I - 16*omegaVar*m_xy_I + 47*omegaVar*m_xz_I + 11*omegaVar*m_yy_I - 16*omegaVar*m_yz_I + 2*omegaVar*m_zz_I - 40)/(180*(m_xx_I - 2*m_xy_I - 2*m_xz_I + m_yy_I - 2*m_yz_I + m_zz_I - omegaVar*m_xx_I + 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - omegaVar*m_yy_I + 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_yy_t45 = (4*m_xx_I - 112*m_xy_I + 44*m_xz_I + 472*m_yy_I - 112*m_yz_I + 4*m_zz_I - 4*omegaVar*m_xx_I + 2*omegaVar*m_xy_I + 11*omegaVar*m_xz_I + 23*omegaVar*m_yy_I + 2*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 20)/(90*(m_xx_I - 2*m_xy_I - 2*m_xz_I + m_yy_I - 2*m_yz_I + m_zz_I - omegaVar*m_xx_I + 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - omegaVar*m_yy_I + 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_yz_t90 = (44*m_xx_I + 16*m_xy_I + 16*m_xz_I - 112*m_yy_I + 1108*m_yz_I - 112*m_zz_I + 11*omegaVar*m_xx_I - 16*omegaVar*m_xy_I - 16*omegaVar*m_xz_I + 2*omegaVar*m_yy_I + 47*omegaVar*m_yz_I + 2*omegaVar*m_zz_I - 40)/(180*(m_xx_I - 2*m_xy_I - 2*m_xz_I + m_yy_I - 2*m_yz_I + m_zz_I - omegaVar*m_xx_I + 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - omegaVar*m_yy_I + 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_zz_t45 = (4*m_xx_I + 44*m_xy_I - 112*m_xz_I + 4*m_yy_I - 112*m_yz_I + 472*m_zz_I - 4*omegaVar*m_xx_I + 11*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - 4*omegaVar*m_yy_I + 2*omegaVar*m_yz_I + 23*omegaVar*m_zz_I + 20)/(90*(m_xx_I - 2*m_xy_I - 2*m_xz_I + m_yy_I - 2*m_yz_I + m_zz_I - omegaVar*m_xx_I + 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - omegaVar*m_yy_I + 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = -(rho*(omegaVar - 2)*(2*m_xy_t90 - 2*m_xz_t90 - m_yy_t45 + m_zz_t45))/8;
    L_BC_Fy = (m_xx_t45/12 - m_xy_t90/2 + m_yy_t45/3 - m_yz_t90/2 + m_zz_t45/12 - (omegaVar*m_xx_t45)/24 + (omegaVar*m_xy_t90)/4 - (omegaVar*m_yy_t45)/6 + (omegaVar*m_yz_t90)/4 - (omegaVar*m_zz_t45)/24 + 1.0/9.0)*rho - RHO_0/9;
    L_BC_Fz = (m_xz_t90/2 - m_xx_t45/12 - m_yy_t45/12 + m_yz_t90/2 - m_zz_t45/3 + (omegaVar*m_xx_t45)/24 - (omegaVar*m_xz_t90)/4 + (omegaVar*m_yy_t45)/24 - (omegaVar*m_yz_t90)/4 + (omegaVar*m_zz_t45)/6 - 1.0/9.0)*rho + RHO_0/9;
    #endif //_BC_FORCES 

    break;
case 0b0000010101001111:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[6] + pop[7] + pop[8] + pop[10] + pop[12] + pop[14] + pop[15] + pop[17];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] - (1.0/3.0)*pop[17]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[14]);
    m_xz_I = inv_rho_I * (pop[10] - pop[15]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] + (2.0/3.0)*pop[17]);
    m_yz_I = inv_rho_I * (pop[12] - pop[17]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[17]);

    rho = (36*(107*rho_I + 12*m_xx_I*rho_I - 33*m_xy_I*rho_I + 12*m_yy_I*rho_I + 72*m_zz_I*rho_I - 12*omegaVar*m_xx_I*rho_I + 33*omegaVar*m_xy_I*rho_I - 12*omegaVar*m_yy_I*rho_I - 72*omegaVar*m_zz_I*rho_I))/(313*omegaVar + 2790);

    m_xx_t45 = -(648*m_xy_I - 2772*m_xx_I + 18*m_yy_I + 108*m_zz_I - 302*omegaVar*m_xx_I + 48*omegaVar*m_xy_I + 11*omegaVar*m_yy_I + 66*omegaVar*m_zz_I - 72)/(27*(12*m_xx_I - 33*m_xy_I + 12*m_yy_I + 72*m_zz_I - 12*omegaVar*m_xx_I + 33*omegaVar*m_xy_I - 12*omegaVar*m_yy_I - 72*omegaVar*m_zz_I + 107));
    m_xy_t90 = -(432*m_xx_I - 3978*m_xy_I + 432*m_yy_I - 198*m_zz_I + 32*omegaVar*m_xx_I - 401*omegaVar*m_xy_I + 32*omegaVar*m_yy_I - 121*omegaVar*m_zz_I + 132)/(36*(12*m_xx_I - 33*m_xy_I + 12*m_yy_I + 72*m_zz_I - 12*omegaVar*m_xx_I + 33*omegaVar*m_xy_I - 12*omegaVar*m_yy_I - 72*omegaVar*m_zz_I + 107));
    m_xz_t90 = (m_xz_I*(313*omegaVar + 2790))/(18*(12*m_xx_I - 33*m_xy_I + 12*m_yy_I + 72*m_zz_I - 12*omegaVar*m_xx_I + 33*omegaVar*m_xy_I - 12*omegaVar*m_yy_I - 72*omegaVar*m_zz_I + 107));
    m_yy_t45 = -(18*m_xx_I + 648*m_xy_I - 2772*m_yy_I + 108*m_zz_I + 11*omegaVar*m_xx_I + 48*omegaVar*m_xy_I - 302*omegaVar*m_yy_I + 66*omegaVar*m_zz_I - 72)/(27*(12*m_xx_I - 33*m_xy_I + 12*m_yy_I + 72*m_zz_I - 12*omegaVar*m_xx_I + 33*omegaVar*m_xy_I - 12*omegaVar*m_yy_I - 72*omegaVar*m_zz_I + 107));
    m_yz_t90 = (m_yz_I*(313*omegaVar + 2790))/(18*(12*m_xx_I - 33*m_xy_I + 12*m_yy_I + 72*m_zz_I - 12*omegaVar*m_xx_I + 33*omegaVar*m_xy_I - 12*omegaVar*m_yy_I - 72*omegaVar*m_zz_I + 107));
    m_zz_t45 = (198*m_xy_I - 72*m_xx_I - 72*m_yy_I + 2358*m_zz_I - 44*omegaVar*m_xx_I + 121*omegaVar*m_xy_I - 44*omegaVar*m_yy_I + 49*omegaVar*m_zz_I + 288)/(18*(12*m_xx_I - 33*m_xy_I + 12*m_yy_I + 72*m_zz_I - 12*omegaVar*m_xx_I + 33*omegaVar*m_xy_I - 12*omegaVar*m_yy_I - 72*omegaVar*m_zz_I + 107));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = (m_xy_t90/2 - m_xx_t45/6 - m_xz_t90 - m_yy_t45/6 + m_zz_t45/12 + (omegaVar*m_xx_t45)/12 - (omegaVar*m_xy_t90)/4 + (omegaVar*m_xz_t90)/2 + (omegaVar*m_yy_t45)/12 - (omegaVar*m_zz_t45)/24 - 1.0/18.0)*rho + RHO_0/18;
    L_BC_Fy = (m_xx_t45/6 - m_xy_t90/2 + m_yy_t45/6 - m_yz_t90 - m_zz_t45/12 - (omegaVar*m_xx_t45)/12 + (omegaVar*m_xy_t90)/4 - (omegaVar*m_yy_t45)/12 + (omegaVar*m_yz_t90)/2 + (omegaVar*m_zz_t45)/24 + 1.0/18.0)*rho - RHO_0/18;
    L_BC_Fz = ((omegaVar*m_zz_t45)/2 - m_zz_t45 - 1.0/3.0)*rho + RHO_0/3;
    #endif //_BC_FORCES 

    break;
case 0b0000010101010000:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (12*(15*rho_I + 2*m_xx_I*rho_I - 6*m_xz_I*rho_I - m_yy_I*rho_I + 2*m_zz_I*rho_I - 2*omegaVar*m_xx_I*rho_I + 6*omegaVar*m_xz_I*rho_I + omegaVar*m_yy_I*rho_I - 2*omegaVar*m_zz_I*rho_I))/(3*omegaVar + 172);

    m_xx_t45 = (1692*m_xx_I - 432*m_xz_I - 72*m_yy_I + 144*m_zz_I + 23*omegaVar*m_xx_I + 12*omegaVar*m_xz_I + 2*omegaVar*m_yy_I - 4*omegaVar*m_zz_I + 48)/(108*(2*m_xx_I - 6*m_xz_I - m_yy_I + 2*m_zz_I - 2*omegaVar*m_xx_I + 6*omegaVar*m_xz_I + omegaVar*m_yy_I - 2*omegaVar*m_zz_I + 15));
    m_xy_t90 = (m_xy_I*(3*omegaVar + 172))/(12*(2*m_xx_I - 6*m_xz_I - m_yy_I + 2*m_zz_I - 2*omegaVar*m_xx_I + 6*omegaVar*m_xz_I + omegaVar*m_yy_I - 2*omegaVar*m_zz_I + 15));
    m_xz_t90 = (732*m_xz_I - 72*m_xx_I + 36*m_yy_I - 72*m_zz_I + 2*omegaVar*m_xx_I + 3*omegaVar*m_xz_I - omegaVar*m_yy_I + 2*omegaVar*m_zz_I - 24)/(36*(2*m_xx_I - 6*m_xz_I - m_yy_I + 2*m_zz_I - 2*omegaVar*m_xx_I + 6*omegaVar*m_xz_I + omegaVar*m_yy_I - 2*omegaVar*m_zz_I + 15));
    m_yy_t45 = (108*m_xz_I - 36*m_xx_I + 792*m_yy_I - 36*m_zz_I + omegaVar*m_xx_I - 3*omegaVar*m_xz_I + 13*omegaVar*m_yy_I + omegaVar*m_zz_I - 12)/(54*(2*m_xx_I - 6*m_xz_I - m_yy_I + 2*m_zz_I - 2*omegaVar*m_xx_I + 6*omegaVar*m_xz_I + omegaVar*m_yy_I - 2*omegaVar*m_zz_I + 15));
    m_yz_t90 = (m_yz_I*(3*omegaVar + 172))/(12*(2*m_xx_I - 6*m_xz_I - m_yy_I + 2*m_zz_I - 2*omegaVar*m_xx_I + 6*omegaVar*m_xz_I + omegaVar*m_yy_I - 2*omegaVar*m_zz_I + 15));
    m_zz_t45 = (144*m_xx_I - 432*m_xz_I - 72*m_yy_I + 1692*m_zz_I - 4*omegaVar*m_xx_I + 12*omegaVar*m_xz_I + 2*omegaVar*m_yy_I + 23*omegaVar*m_zz_I + 48)/(108*(2*m_xx_I - 6*m_xz_I - m_yy_I + 2*m_zz_I - 2*omegaVar*m_xx_I + 6*omegaVar*m_xz_I + omegaVar*m_yy_I - 2*omegaVar*m_zz_I + 15));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = (m_xz_t90/2 - m_xx_t45/6 + m_yy_t45/12 - m_zz_t45/6 + (omegaVar*m_xx_t45)/12 - (omegaVar*m_xz_t90)/4 - (omegaVar*m_yy_t45)/24 + (omegaVar*m_zz_t45)/12 - 1.0/18.0)*rho + RHO_0/18;
    L_BC_Fy = 0;
    L_BC_Fz = (m_xx_t45/6 - m_xz_t90/2 - m_yy_t45/12 + m_zz_t45/6 - (omegaVar*m_xx_t45)/12 + (omegaVar*m_xz_t90)/4 + (omegaVar*m_yy_t45)/24 - (omegaVar*m_zz_t45)/12 + 1.0/18.0)*rho - RHO_0/18;
    #endif //_BC_FORCES 

    break;
case 0b0000010101010001:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (6*(15*rho_I + 4*m_xx_I*rho_I + 6*m_xy_I*rho_I - 6*m_xz_I*rho_I + m_yy_I*rho_I + m_zz_I*rho_I - 4*omegaVar*m_xx_I*rho_I - 6*omegaVar*m_xy_I*rho_I + 6*omegaVar*m_xz_I*rho_I - omegaVar*m_yy_I*rho_I - omegaVar*m_zz_I*rho_I))/(3*omegaVar + 82);

    m_xx_t45 = (882*m_xx_I + 216*m_xy_I - 216*m_xz_I + 36*m_yy_I + 36*m_zz_I + 19*omegaVar*m_xx_I - 12*omegaVar*m_xy_I + 12*omegaVar*m_xz_I - 2*omegaVar*m_yy_I - 2*omegaVar*m_zz_I + 48)/(54*(4*m_xx_I + 6*m_xy_I - 6*m_xz_I + m_yy_I + m_zz_I - 4*omegaVar*m_xx_I - 6*omegaVar*m_xy_I + 6*omegaVar*m_xz_I - omegaVar*m_yy_I - omegaVar*m_zz_I + 15));
    m_xy_t90 = (180*m_xx_I + 1746*m_xy_I - 24*m_xz_I + 168*m_yy_I - 78*m_zz_I - 10*omegaVar*m_xx_I + 39*omegaVar*m_xy_I + 24*omegaVar*m_xz_I + 2*omegaVar*m_yy_I - 7*omegaVar*m_zz_I + 60)/(90*(4*m_xx_I + 6*m_xy_I - 6*m_xz_I + m_yy_I + m_zz_I - 4*omegaVar*m_xx_I - 6*omegaVar*m_xy_I + 6*omegaVar*m_xz_I - omegaVar*m_yy_I - omegaVar*m_zz_I + 15));
    m_xz_t90 = (1746*m_xz_I - 24*m_xy_I - 180*m_xx_I + 78*m_yy_I - 168*m_zz_I + 10*omegaVar*m_xx_I + 24*omegaVar*m_xy_I + 39*omegaVar*m_xz_I + 7*omegaVar*m_yy_I - 2*omegaVar*m_zz_I - 60)/(90*(4*m_xx_I + 6*m_xy_I - 6*m_xz_I + m_yy_I + m_zz_I - 4*omegaVar*m_xx_I - 6*omegaVar*m_xy_I + 6*omegaVar*m_xz_I - omegaVar*m_yy_I - omegaVar*m_zz_I + 15));
    m_yy_t45 = (90*m_xx_I + 504*m_xy_I + 234*m_xz_I + 2052*m_yy_I - 162*m_zz_I - 5*omegaVar*m_xx_I + 6*omegaVar*m_xy_I + 21*omegaVar*m_xz_I + 73*omegaVar*m_yy_I - 8*omegaVar*m_zz_I + 30)/(135*(4*m_xx_I + 6*m_xy_I - 6*m_xz_I + m_yy_I + m_zz_I - 4*omegaVar*m_xx_I - 6*omegaVar*m_xy_I + 6*omegaVar*m_xz_I - omegaVar*m_yy_I - omegaVar*m_zz_I + 15));
    m_yz_t90 = (m_yz_I*(3*omegaVar + 82))/(6*(4*m_xx_I + 6*m_xy_I - 6*m_xz_I + m_yy_I + m_zz_I - 4*omegaVar*m_xx_I - 6*omegaVar*m_xy_I + 6*omegaVar*m_xz_I - omegaVar*m_yy_I - omegaVar*m_zz_I + 15));
    m_zz_t45 = -(234*m_xy_I - 90*m_xx_I + 504*m_xz_I + 162*m_yy_I - 2052*m_zz_I + 5*omegaVar*m_xx_I + 21*omegaVar*m_xy_I + 6*omegaVar*m_xz_I + 8*omegaVar*m_yy_I - 73*omegaVar*m_zz_I - 30)/(135*(4*m_xx_I + 6*m_xy_I - 6*m_xz_I + m_yy_I + m_zz_I - 4*omegaVar*m_xx_I - 6*omegaVar*m_xy_I + 6*omegaVar*m_xz_I - omegaVar*m_yy_I - omegaVar*m_zz_I + 15));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = (m_xz_t90/2 - m_xy_t90/2 - m_xx_t45/3 - m_yy_t45/12 - m_zz_t45/12 + (omegaVar*m_xx_t45)/6 + (omegaVar*m_xy_t90)/4 - (omegaVar*m_xz_t90)/4 + (omegaVar*m_yy_t45)/24 + (omegaVar*m_zz_t45)/24 - 1.0/9.0)*rho + RHO_0/9;
    L_BC_Fy = (m_zz_t45/12 - m_xy_t90/2 - m_yy_t45/6 - m_xx_t45/6 + (omegaVar*m_xx_t45)/12 + (omegaVar*m_xy_t90)/4 + (omegaVar*m_yy_t45)/12 - (omegaVar*m_zz_t45)/24 - 1.0/18.0)*rho + RHO_0/18;
    L_BC_Fz = (m_xx_t45/6 - m_xz_t90/2 - m_yy_t45/12 + m_zz_t45/6 - (omegaVar*m_xx_t45)/12 + (omegaVar*m_xz_t90)/4 + (omegaVar*m_yy_t45)/24 - (omegaVar*m_zz_t45)/12 + 1.0/18.0)*rho - RHO_0/18;
    #endif //_BC_FORCES 

    break;
case 0b0000010101010010:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (12*(15*rho_I + 2*m_xx_I*rho_I - 6*m_xz_I*rho_I - m_yy_I*rho_I + 2*m_zz_I*rho_I - 2*omegaVar*m_xx_I*rho_I + 6*omegaVar*m_xz_I*rho_I + omegaVar*m_yy_I*rho_I - 2*omegaVar*m_zz_I*rho_I))/(3*omegaVar + 172);

    m_xx_t45 = (1692*m_xx_I - 432*m_xz_I - 72*m_yy_I + 144*m_zz_I + 23*omegaVar*m_xx_I + 12*omegaVar*m_xz_I + 2*omegaVar*m_yy_I - 4*omegaVar*m_zz_I + 48)/(108*(2*m_xx_I - 6*m_xz_I - m_yy_I + 2*m_zz_I - 2*omegaVar*m_xx_I + 6*omegaVar*m_xz_I + omegaVar*m_yy_I - 2*omegaVar*m_zz_I + 15));
    m_xy_t90 = (m_xy_I*(3*omegaVar + 172))/(12*(2*m_xx_I - 6*m_xz_I - m_yy_I + 2*m_zz_I - 2*omegaVar*m_xx_I + 6*omegaVar*m_xz_I + omegaVar*m_yy_I - 2*omegaVar*m_zz_I + 15));
    m_xz_t90 = (732*m_xz_I - 72*m_xx_I + 36*m_yy_I - 72*m_zz_I + 2*omegaVar*m_xx_I + 3*omegaVar*m_xz_I - omegaVar*m_yy_I + 2*omegaVar*m_zz_I - 24)/(36*(2*m_xx_I - 6*m_xz_I - m_yy_I + 2*m_zz_I - 2*omegaVar*m_xx_I + 6*omegaVar*m_xz_I + omegaVar*m_yy_I - 2*omegaVar*m_zz_I + 15));
    m_yy_t45 = (108*m_xz_I - 36*m_xx_I + 792*m_yy_I - 36*m_zz_I + omegaVar*m_xx_I - 3*omegaVar*m_xz_I + 13*omegaVar*m_yy_I + omegaVar*m_zz_I - 12)/(54*(2*m_xx_I - 6*m_xz_I - m_yy_I + 2*m_zz_I - 2*omegaVar*m_xx_I + 6*omegaVar*m_xz_I + omegaVar*m_yy_I - 2*omegaVar*m_zz_I + 15));
    m_yz_t90 = (m_yz_I*(3*omegaVar + 172))/(12*(2*m_xx_I - 6*m_xz_I - m_yy_I + 2*m_zz_I - 2*omegaVar*m_xx_I + 6*omegaVar*m_xz_I + omegaVar*m_yy_I - 2*omegaVar*m_zz_I + 15));
    m_zz_t45 = (144*m_xx_I - 432*m_xz_I - 72*m_yy_I + 1692*m_zz_I - 4*omegaVar*m_xx_I + 12*omegaVar*m_xz_I + 2*omegaVar*m_yy_I + 23*omegaVar*m_zz_I + 48)/(108*(2*m_xx_I - 6*m_xz_I - m_yy_I + 2*m_zz_I - 2*omegaVar*m_xx_I + 6*omegaVar*m_xz_I + omegaVar*m_yy_I - 2*omegaVar*m_zz_I + 15));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = (m_xz_t90/2 - m_xx_t45/6 + m_yy_t45/12 - m_zz_t45/6 + (omegaVar*m_xx_t45)/12 - (omegaVar*m_xz_t90)/4 - (omegaVar*m_yy_t45)/24 + (omegaVar*m_zz_t45)/12 - 1.0/18.0)*rho + RHO_0/18;
    L_BC_Fy = 0;
    L_BC_Fz = (m_xx_t45/6 - m_xz_t90/2 - m_yy_t45/12 + m_zz_t45/6 - (omegaVar*m_xx_t45)/12 + (omegaVar*m_xz_t90)/4 + (omegaVar*m_yy_t45)/24 - (omegaVar*m_zz_t45)/12 + 1.0/18.0)*rho - RHO_0/18;
    #endif //_BC_FORCES 

    break;
case 0b0000010101010011:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[8] + pop[9] + pop[10] + pop[12] + pop[13] + pop[14] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (12*(5*rho_I + m_xx_I*rho_I + 2*m_xy_I*rho_I - 2*m_xz_I*rho_I + m_yy_I*rho_I + 2*m_yz_I*rho_I + m_zz_I*rho_I - omegaVar*m_xx_I*rho_I - 2*omegaVar*m_xy_I*rho_I + 2*omegaVar*m_xz_I*rho_I - omegaVar*m_yy_I*rho_I - 2*omegaVar*m_yz_I*rho_I - omegaVar*m_zz_I*rho_I))/(3*omegaVar + 52);

    m_xx_t45 = (472*m_xx_I + 112*m_xy_I - 112*m_xz_I + 4*m_yy_I - 44*m_yz_I + 4*m_zz_I + 23*omegaVar*m_xx_I - 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - 4*omegaVar*m_yy_I - 11*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 20)/(90*(m_xx_I + 2*m_xy_I - 2*m_xz_I + m_yy_I + 2*m_yz_I + m_zz_I - omegaVar*m_xx_I - 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - omegaVar*m_yy_I - 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_xy_t90 = (112*m_xx_I + 1108*m_xy_I - 16*m_xz_I + 112*m_yy_I + 16*m_yz_I - 44*m_zz_I - 2*omegaVar*m_xx_I + 47*omegaVar*m_xy_I + 16*omegaVar*m_xz_I - 2*omegaVar*m_yy_I - 16*omegaVar*m_yz_I - 11*omegaVar*m_zz_I + 40)/(180*(m_xx_I + 2*m_xy_I - 2*m_xz_I + m_yy_I + 2*m_yz_I + m_zz_I - omegaVar*m_xx_I - 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - omegaVar*m_yy_I - 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_xz_t90 = (1108*m_xz_I - 16*m_xy_I - 112*m_xx_I + 44*m_yy_I - 16*m_yz_I - 112*m_zz_I + 2*omegaVar*m_xx_I + 16*omegaVar*m_xy_I + 47*omegaVar*m_xz_I + 11*omegaVar*m_yy_I + 16*omegaVar*m_yz_I + 2*omegaVar*m_zz_I - 40)/(180*(m_xx_I + 2*m_xy_I - 2*m_xz_I + m_yy_I + 2*m_yz_I + m_zz_I - omegaVar*m_xx_I - 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - omegaVar*m_yy_I - 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_yy_t45 = (4*m_xx_I + 112*m_xy_I + 44*m_xz_I + 472*m_yy_I + 112*m_yz_I + 4*m_zz_I - 4*omegaVar*m_xx_I - 2*omegaVar*m_xy_I + 11*omegaVar*m_xz_I + 23*omegaVar*m_yy_I - 2*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 20)/(90*(m_xx_I + 2*m_xy_I - 2*m_xz_I + m_yy_I + 2*m_yz_I + m_zz_I - omegaVar*m_xx_I - 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - omegaVar*m_yy_I - 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_yz_t90 = (16*m_xy_I - 44*m_xx_I - 16*m_xz_I + 112*m_yy_I + 1108*m_yz_I + 112*m_zz_I - 11*omegaVar*m_xx_I - 16*omegaVar*m_xy_I + 16*omegaVar*m_xz_I - 2*omegaVar*m_yy_I + 47*omegaVar*m_yz_I - 2*omegaVar*m_zz_I + 40)/(180*(m_xx_I + 2*m_xy_I - 2*m_xz_I + m_yy_I + 2*m_yz_I + m_zz_I - omegaVar*m_xx_I - 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - omegaVar*m_yy_I - 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_zz_t45 = (4*m_xx_I - 44*m_xy_I - 112*m_xz_I + 4*m_yy_I + 112*m_yz_I + 472*m_zz_I - 4*omegaVar*m_xx_I - 11*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - 4*omegaVar*m_yy_I - 2*omegaVar*m_yz_I + 23*omegaVar*m_zz_I + 20)/(90*(m_xx_I + 2*m_xy_I - 2*m_xz_I + m_yy_I + 2*m_yz_I + m_zz_I - omegaVar*m_xx_I - 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - omegaVar*m_yy_I - 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = (m_xz_t90/2 - m_xy_t90/2 - m_xx_t45/3 - m_yy_t45/12 - m_zz_t45/12 + (omegaVar*m_xx_t45)/6 + (omegaVar*m_xy_t90)/4 - (omegaVar*m_xz_t90)/4 + (omegaVar*m_yy_t45)/24 + (omegaVar*m_zz_t45)/24 - 1.0/9.0)*rho + RHO_0/9;
    L_BC_Fy = ((omegaVar*m_xx_t45)/24 - m_xy_t90/2 - m_yy_t45/3 - m_yz_t90/2 - m_zz_t45/12 - m_xx_t45/12 + (omegaVar*m_xy_t90)/4 + (omegaVar*m_yy_t45)/6 + (omegaVar*m_yz_t90)/4 + (omegaVar*m_zz_t45)/24 - 1.0/9.0)*rho + RHO_0/9;
    L_BC_Fz = (rho*(omegaVar - 2)*(2*m_xz_t90 - m_xx_t45 + m_yy_t45 + 2*m_yz_t90))/8;
    #endif //_BC_FORCES 

    break;
case 0b0000010101010100:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[14] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (6*(15*rho_I + 4*m_xx_I*rho_I - 6*m_xy_I*rho_I - 6*m_xz_I*rho_I + m_yy_I*rho_I + m_zz_I*rho_I - 4*omegaVar*m_xx_I*rho_I + 6*omegaVar*m_xy_I*rho_I + 6*omegaVar*m_xz_I*rho_I - omegaVar*m_yy_I*rho_I - omegaVar*m_zz_I*rho_I))/(3*omegaVar + 82);

    m_xx_t45 = (882*m_xx_I - 216*m_xy_I - 216*m_xz_I + 36*m_yy_I + 36*m_zz_I + 19*omegaVar*m_xx_I + 12*omegaVar*m_xy_I + 12*omegaVar*m_xz_I - 2*omegaVar*m_yy_I - 2*omegaVar*m_zz_I + 48)/(54*(4*m_xx_I - 6*m_xy_I - 6*m_xz_I + m_yy_I + m_zz_I - 4*omegaVar*m_xx_I + 6*omegaVar*m_xy_I + 6*omegaVar*m_xz_I - omegaVar*m_yy_I - omegaVar*m_zz_I + 15));
    m_xy_t90 = (1746*m_xy_I - 180*m_xx_I + 24*m_xz_I - 168*m_yy_I + 78*m_zz_I + 10*omegaVar*m_xx_I + 39*omegaVar*m_xy_I - 24*omegaVar*m_xz_I - 2*omegaVar*m_yy_I + 7*omegaVar*m_zz_I - 60)/(90*(4*m_xx_I - 6*m_xy_I - 6*m_xz_I + m_yy_I + m_zz_I - 4*omegaVar*m_xx_I + 6*omegaVar*m_xy_I + 6*omegaVar*m_xz_I - omegaVar*m_yy_I - omegaVar*m_zz_I + 15));
    m_xz_t90 = (24*m_xy_I - 180*m_xx_I + 1746*m_xz_I + 78*m_yy_I - 168*m_zz_I + 10*omegaVar*m_xx_I - 24*omegaVar*m_xy_I + 39*omegaVar*m_xz_I + 7*omegaVar*m_yy_I - 2*omegaVar*m_zz_I - 60)/(90*(4*m_xx_I - 6*m_xy_I - 6*m_xz_I + m_yy_I + m_zz_I - 4*omegaVar*m_xx_I + 6*omegaVar*m_xy_I + 6*omegaVar*m_xz_I - omegaVar*m_yy_I - omegaVar*m_zz_I + 15));
    m_yy_t45 = (90*m_xx_I - 504*m_xy_I + 234*m_xz_I + 2052*m_yy_I - 162*m_zz_I - 5*omegaVar*m_xx_I - 6*omegaVar*m_xy_I + 21*omegaVar*m_xz_I + 73*omegaVar*m_yy_I - 8*omegaVar*m_zz_I + 30)/(135*(4*m_xx_I - 6*m_xy_I - 6*m_xz_I + m_yy_I + m_zz_I - 4*omegaVar*m_xx_I + 6*omegaVar*m_xy_I + 6*omegaVar*m_xz_I - omegaVar*m_yy_I - omegaVar*m_zz_I + 15));
    m_yz_t90 = (m_yz_I*(3*omegaVar + 82))/(6*(4*m_xx_I - 6*m_xy_I - 6*m_xz_I + m_yy_I + m_zz_I - 4*omegaVar*m_xx_I + 6*omegaVar*m_xy_I + 6*omegaVar*m_xz_I - omegaVar*m_yy_I - omegaVar*m_zz_I + 15));
    m_zz_t45 = (90*m_xx_I + 234*m_xy_I - 504*m_xz_I - 162*m_yy_I + 2052*m_zz_I - 5*omegaVar*m_xx_I + 21*omegaVar*m_xy_I - 6*omegaVar*m_xz_I - 8*omegaVar*m_yy_I + 73*omegaVar*m_zz_I + 30)/(135*(4*m_xx_I - 6*m_xy_I - 6*m_xz_I + m_yy_I + m_zz_I - 4*omegaVar*m_xx_I + 6*omegaVar*m_xy_I + 6*omegaVar*m_xz_I - omegaVar*m_yy_I - omegaVar*m_zz_I + 15));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = (m_xy_t90/2 - m_xx_t45/3 + m_xz_t90/2 - m_yy_t45/12 - m_zz_t45/12 + (omegaVar*m_xx_t45)/6 - (omegaVar*m_xy_t90)/4 - (omegaVar*m_xz_t90)/4 + (omegaVar*m_yy_t45)/24 + (omegaVar*m_zz_t45)/24 - 1.0/9.0)*rho + RHO_0/9;
    L_BC_Fy = (m_xx_t45/6 - m_xy_t90/2 + m_yy_t45/6 - m_zz_t45/12 - (omegaVar*m_xx_t45)/12 + (omegaVar*m_xy_t90)/4 - (omegaVar*m_yy_t45)/12 + (omegaVar*m_zz_t45)/24 + 1.0/18.0)*rho - RHO_0/18;
    L_BC_Fz = (m_xx_t45/6 - m_xz_t90/2 - m_yy_t45/12 + m_zz_t45/6 - (omegaVar*m_xx_t45)/12 + (omegaVar*m_xz_t90)/4 + (omegaVar*m_yy_t45)/24 - (omegaVar*m_zz_t45)/12 + 1.0/18.0)*rho - RHO_0/18;
    #endif //_BC_FORCES 

    break;
case 0b0000010101010101:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[8] + pop[10] + pop[11] + pop[12] + pop[14] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[8] - pop[14]);
    m_xz_I = inv_rho_I * (pop[10] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (3*rho_I*(3*m_xx_I - 3*omegaVar*m_xx_I + 4))/(omegaVar + 9);

    m_xx_t45 = (15*m_xx_I + 2)/(3*(3*m_xx_I - 3*omegaVar*m_xx_I + 4));
    m_xy_t90 = (2*m_xy_I*(omegaVar + 9))/(3*(3*m_xx_I - 3*omegaVar*m_xx_I + 4));
    m_xz_t90 = (2*m_xz_I*(omegaVar + 9))/(3*(3*m_xx_I - 3*omegaVar*m_xx_I + 4));
    m_yy_t45 = (4*(omegaVar + 9)*(10*m_yy_I - m_zz_I))/(99*(3*m_xx_I - 3*omegaVar*m_xx_I + 4));
    m_yz_t90 = (m_yz_I*(omegaVar + 9))/(3*(3*m_xx_I - 3*omegaVar*m_xx_I + 4));
    m_zz_t45 = -(4*(m_yy_I - 10*m_zz_I)*(omegaVar + 9))/(99*(3*m_xx_I - 3*omegaVar*m_xx_I + 4));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = ((omegaVar*m_xx_t45)/2 - m_xx_t45 - 1.0/3.0)*rho + RHO_0/3;
    L_BC_Fy = (m_xy_t90*rho*(omegaVar - 2))/2;
    L_BC_Fz = (m_xz_t90*rho*(omegaVar - 2))/2;
    #endif //_BC_FORCES 

    break;
case 0b0000010101010110:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[14] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (6*(15*rho_I + 4*m_xx_I*rho_I - 6*m_xy_I*rho_I - 6*m_xz_I*rho_I + m_yy_I*rho_I + m_zz_I*rho_I - 4*omegaVar*m_xx_I*rho_I + 6*omegaVar*m_xy_I*rho_I + 6*omegaVar*m_xz_I*rho_I - omegaVar*m_yy_I*rho_I - omegaVar*m_zz_I*rho_I))/(3*omegaVar + 82);

    m_xx_t45 = (882*m_xx_I - 216*m_xy_I - 216*m_xz_I + 36*m_yy_I + 36*m_zz_I + 19*omegaVar*m_xx_I + 12*omegaVar*m_xy_I + 12*omegaVar*m_xz_I - 2*omegaVar*m_yy_I - 2*omegaVar*m_zz_I + 48)/(54*(4*m_xx_I - 6*m_xy_I - 6*m_xz_I + m_yy_I + m_zz_I - 4*omegaVar*m_xx_I + 6*omegaVar*m_xy_I + 6*omegaVar*m_xz_I - omegaVar*m_yy_I - omegaVar*m_zz_I + 15));
    m_xy_t90 = (1746*m_xy_I - 180*m_xx_I + 24*m_xz_I - 168*m_yy_I + 78*m_zz_I + 10*omegaVar*m_xx_I + 39*omegaVar*m_xy_I - 24*omegaVar*m_xz_I - 2*omegaVar*m_yy_I + 7*omegaVar*m_zz_I - 60)/(90*(4*m_xx_I - 6*m_xy_I - 6*m_xz_I + m_yy_I + m_zz_I - 4*omegaVar*m_xx_I + 6*omegaVar*m_xy_I + 6*omegaVar*m_xz_I - omegaVar*m_yy_I - omegaVar*m_zz_I + 15));
    m_xz_t90 = (24*m_xy_I - 180*m_xx_I + 1746*m_xz_I + 78*m_yy_I - 168*m_zz_I + 10*omegaVar*m_xx_I - 24*omegaVar*m_xy_I + 39*omegaVar*m_xz_I + 7*omegaVar*m_yy_I - 2*omegaVar*m_zz_I - 60)/(90*(4*m_xx_I - 6*m_xy_I - 6*m_xz_I + m_yy_I + m_zz_I - 4*omegaVar*m_xx_I + 6*omegaVar*m_xy_I + 6*omegaVar*m_xz_I - omegaVar*m_yy_I - omegaVar*m_zz_I + 15));
    m_yy_t45 = (90*m_xx_I - 504*m_xy_I + 234*m_xz_I + 2052*m_yy_I - 162*m_zz_I - 5*omegaVar*m_xx_I - 6*omegaVar*m_xy_I + 21*omegaVar*m_xz_I + 73*omegaVar*m_yy_I - 8*omegaVar*m_zz_I + 30)/(135*(4*m_xx_I - 6*m_xy_I - 6*m_xz_I + m_yy_I + m_zz_I - 4*omegaVar*m_xx_I + 6*omegaVar*m_xy_I + 6*omegaVar*m_xz_I - omegaVar*m_yy_I - omegaVar*m_zz_I + 15));
    m_yz_t90 = (m_yz_I*(3*omegaVar + 82))/(6*(4*m_xx_I - 6*m_xy_I - 6*m_xz_I + m_yy_I + m_zz_I - 4*omegaVar*m_xx_I + 6*omegaVar*m_xy_I + 6*omegaVar*m_xz_I - omegaVar*m_yy_I - omegaVar*m_zz_I + 15));
    m_zz_t45 = (90*m_xx_I + 234*m_xy_I - 504*m_xz_I - 162*m_yy_I + 2052*m_zz_I - 5*omegaVar*m_xx_I + 21*omegaVar*m_xy_I - 6*omegaVar*m_xz_I - 8*omegaVar*m_yy_I + 73*omegaVar*m_zz_I + 30)/(135*(4*m_xx_I - 6*m_xy_I - 6*m_xz_I + m_yy_I + m_zz_I - 4*omegaVar*m_xx_I + 6*omegaVar*m_xy_I + 6*omegaVar*m_xz_I - omegaVar*m_yy_I - omegaVar*m_zz_I + 15));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = (m_xy_t90/2 - m_xx_t45/3 + m_xz_t90/2 - m_yy_t45/12 - m_zz_t45/12 + (omegaVar*m_xx_t45)/6 - (omegaVar*m_xy_t90)/4 - (omegaVar*m_xz_t90)/4 + (omegaVar*m_yy_t45)/24 + (omegaVar*m_zz_t45)/24 - 1.0/9.0)*rho + RHO_0/9;
    L_BC_Fy = (m_xx_t45/6 - m_xy_t90/2 + m_yy_t45/6 - m_zz_t45/12 - (omegaVar*m_xx_t45)/12 + (omegaVar*m_xy_t90)/4 - (omegaVar*m_yy_t45)/12 + (omegaVar*m_zz_t45)/24 + 1.0/18.0)*rho - RHO_0/18;
    L_BC_Fz = (m_xx_t45/6 - m_xz_t90/2 - m_yy_t45/12 + m_zz_t45/6 - (omegaVar*m_xx_t45)/12 + (omegaVar*m_xz_t90)/4 + (omegaVar*m_yy_t45)/24 - (omegaVar*m_zz_t45)/12 + 1.0/18.0)*rho - RHO_0/18;
    #endif //_BC_FORCES 

    break;
case 0b0000010101010111:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[8] + pop[10] + pop[12] + pop[14] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[8] - pop[14]);
    m_xz_I = inv_rho_I * (pop[10] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (36*(107*rho_I + 72*m_xx_I*rho_I + 12*m_yy_I*rho_I + 33*m_yz_I*rho_I + 12*m_zz_I*rho_I - 72*omegaVar*m_xx_I*rho_I - 12*omegaVar*m_yy_I*rho_I - 33*omegaVar*m_yz_I*rho_I - 12*omegaVar*m_zz_I*rho_I))/(313*omegaVar + 2790);

    m_xx_t45 = -(72*m_yy_I - 2358*m_xx_I + 198*m_yz_I + 72*m_zz_I - 49*omegaVar*m_xx_I + 44*omegaVar*m_yy_I + 121*omegaVar*m_yz_I + 44*omegaVar*m_zz_I - 288)/(18*(72*m_xx_I + 12*m_yy_I + 33*m_yz_I + 12*m_zz_I - 72*omegaVar*m_xx_I - 12*omegaVar*m_yy_I - 33*omegaVar*m_yz_I - 12*omegaVar*m_zz_I + 107));
    m_xy_t90 = (m_xy_I*(313*omegaVar + 2790))/(18*(72*m_xx_I + 12*m_yy_I + 33*m_yz_I + 12*m_zz_I - 72*omegaVar*m_xx_I - 12*omegaVar*m_yy_I - 33*omegaVar*m_yz_I - 12*omegaVar*m_zz_I + 107));
    m_xz_t90 = (m_xz_I*(313*omegaVar + 2790))/(18*(72*m_xx_I + 12*m_yy_I + 33*m_yz_I + 12*m_zz_I - 72*omegaVar*m_xx_I - 12*omegaVar*m_yy_I - 33*omegaVar*m_yz_I - 12*omegaVar*m_zz_I + 107));
    m_yy_t45 = (2772*m_yy_I - 108*m_xx_I + 648*m_yz_I - 18*m_zz_I - 66*omegaVar*m_xx_I + 302*omegaVar*m_yy_I + 48*omegaVar*m_yz_I - 11*omegaVar*m_zz_I + 72)/(27*(72*m_xx_I + 12*m_yy_I + 33*m_yz_I + 12*m_zz_I - 72*omegaVar*m_xx_I - 12*omegaVar*m_yy_I - 33*omegaVar*m_yz_I - 12*omegaVar*m_zz_I + 107));
    m_yz_t90 = (432*m_yy_I - 198*m_xx_I + 3978*m_yz_I + 432*m_zz_I - 121*omegaVar*m_xx_I + 32*omegaVar*m_yy_I + 401*omegaVar*m_yz_I + 32*omegaVar*m_zz_I + 132)/(36*(72*m_xx_I + 12*m_yy_I + 33*m_yz_I + 12*m_zz_I - 72*omegaVar*m_xx_I - 12*omegaVar*m_yy_I - 33*omegaVar*m_yz_I - 12*omegaVar*m_zz_I + 107));
    m_zz_t45 = (648*m_yz_I - 18*m_yy_I - 108*m_xx_I + 2772*m_zz_I - 66*omegaVar*m_xx_I - 11*omegaVar*m_yy_I + 48*omegaVar*m_yz_I + 302*omegaVar*m_zz_I + 72)/(27*(72*m_xx_I + 12*m_yy_I + 33*m_yz_I + 12*m_zz_I - 72*omegaVar*m_xx_I - 12*omegaVar*m_yy_I - 33*omegaVar*m_yz_I - 12*omegaVar*m_zz_I + 107));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = ((omegaVar*m_xx_t45)/2 - m_xx_t45 - 1.0/3.0)*rho + RHO_0/3;
    L_BC_Fy = (m_xx_t45/12 - m_xy_t90 - m_yy_t45/6 - m_yz_t90/2 - m_zz_t45/6 - (omegaVar*m_xx_t45)/24 + (omegaVar*m_xy_t90)/2 + (omegaVar*m_yy_t45)/12 + (omegaVar*m_yz_t90)/4 + (omegaVar*m_zz_t45)/12 - 1.0/18.0)*rho + RHO_0/18;
    L_BC_Fz = (m_xx_t45/12 - m_xz_t90 - m_yy_t45/6 - m_yz_t90/2 - m_zz_t45/6 - (omegaVar*m_xx_t45)/24 + (omegaVar*m_xz_t90)/2 + (omegaVar*m_yy_t45)/12 + (omegaVar*m_yz_t90)/4 + (omegaVar*m_zz_t45)/12 - 1.0/18.0)*rho + RHO_0/18;
    #endif //_BC_FORCES 

    break;
case 0b0000010101011000:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (12*(15*rho_I + 2*m_xx_I*rho_I - 6*m_xz_I*rho_I - m_yy_I*rho_I + 2*m_zz_I*rho_I - 2*omegaVar*m_xx_I*rho_I + 6*omegaVar*m_xz_I*rho_I + omegaVar*m_yy_I*rho_I - 2*omegaVar*m_zz_I*rho_I))/(3*omegaVar + 172);

    m_xx_t45 = (1692*m_xx_I - 432*m_xz_I - 72*m_yy_I + 144*m_zz_I + 23*omegaVar*m_xx_I + 12*omegaVar*m_xz_I + 2*omegaVar*m_yy_I - 4*omegaVar*m_zz_I + 48)/(108*(2*m_xx_I - 6*m_xz_I - m_yy_I + 2*m_zz_I - 2*omegaVar*m_xx_I + 6*omegaVar*m_xz_I + omegaVar*m_yy_I - 2*omegaVar*m_zz_I + 15));
    m_xy_t90 = (m_xy_I*(3*omegaVar + 172))/(12*(2*m_xx_I - 6*m_xz_I - m_yy_I + 2*m_zz_I - 2*omegaVar*m_xx_I + 6*omegaVar*m_xz_I + omegaVar*m_yy_I - 2*omegaVar*m_zz_I + 15));
    m_xz_t90 = (732*m_xz_I - 72*m_xx_I + 36*m_yy_I - 72*m_zz_I + 2*omegaVar*m_xx_I + 3*omegaVar*m_xz_I - omegaVar*m_yy_I + 2*omegaVar*m_zz_I - 24)/(36*(2*m_xx_I - 6*m_xz_I - m_yy_I + 2*m_zz_I - 2*omegaVar*m_xx_I + 6*omegaVar*m_xz_I + omegaVar*m_yy_I - 2*omegaVar*m_zz_I + 15));
    m_yy_t45 = (108*m_xz_I - 36*m_xx_I + 792*m_yy_I - 36*m_zz_I + omegaVar*m_xx_I - 3*omegaVar*m_xz_I + 13*omegaVar*m_yy_I + omegaVar*m_zz_I - 12)/(54*(2*m_xx_I - 6*m_xz_I - m_yy_I + 2*m_zz_I - 2*omegaVar*m_xx_I + 6*omegaVar*m_xz_I + omegaVar*m_yy_I - 2*omegaVar*m_zz_I + 15));
    m_yz_t90 = (m_yz_I*(3*omegaVar + 172))/(12*(2*m_xx_I - 6*m_xz_I - m_yy_I + 2*m_zz_I - 2*omegaVar*m_xx_I + 6*omegaVar*m_xz_I + omegaVar*m_yy_I - 2*omegaVar*m_zz_I + 15));
    m_zz_t45 = (144*m_xx_I - 432*m_xz_I - 72*m_yy_I + 1692*m_zz_I - 4*omegaVar*m_xx_I + 12*omegaVar*m_xz_I + 2*omegaVar*m_yy_I + 23*omegaVar*m_zz_I + 48)/(108*(2*m_xx_I - 6*m_xz_I - m_yy_I + 2*m_zz_I - 2*omegaVar*m_xx_I + 6*omegaVar*m_xz_I + omegaVar*m_yy_I - 2*omegaVar*m_zz_I + 15));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = (m_xz_t90/2 - m_xx_t45/6 + m_yy_t45/12 - m_zz_t45/6 + (omegaVar*m_xx_t45)/12 - (omegaVar*m_xz_t90)/4 - (omegaVar*m_yy_t45)/24 + (omegaVar*m_zz_t45)/12 - 1.0/18.0)*rho + RHO_0/18;
    L_BC_Fy = 0;
    L_BC_Fz = (m_xx_t45/6 - m_xz_t90/2 - m_yy_t45/12 + m_zz_t45/6 - (omegaVar*m_xx_t45)/12 + (omegaVar*m_xz_t90)/4 + (omegaVar*m_yy_t45)/24 - (omegaVar*m_zz_t45)/12 + 1.0/18.0)*rho - RHO_0/18;
    #endif //_BC_FORCES 

    break;
case 0b0000010101011001:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (6*(15*rho_I + 4*m_xx_I*rho_I + 6*m_xy_I*rho_I - 6*m_xz_I*rho_I + m_yy_I*rho_I + m_zz_I*rho_I - 4*omegaVar*m_xx_I*rho_I - 6*omegaVar*m_xy_I*rho_I + 6*omegaVar*m_xz_I*rho_I - omegaVar*m_yy_I*rho_I - omegaVar*m_zz_I*rho_I))/(3*omegaVar + 82);

    m_xx_t45 = (882*m_xx_I + 216*m_xy_I - 216*m_xz_I + 36*m_yy_I + 36*m_zz_I + 19*omegaVar*m_xx_I - 12*omegaVar*m_xy_I + 12*omegaVar*m_xz_I - 2*omegaVar*m_yy_I - 2*omegaVar*m_zz_I + 48)/(54*(4*m_xx_I + 6*m_xy_I - 6*m_xz_I + m_yy_I + m_zz_I - 4*omegaVar*m_xx_I - 6*omegaVar*m_xy_I + 6*omegaVar*m_xz_I - omegaVar*m_yy_I - omegaVar*m_zz_I + 15));
    m_xy_t90 = (180*m_xx_I + 1746*m_xy_I - 24*m_xz_I + 168*m_yy_I - 78*m_zz_I - 10*omegaVar*m_xx_I + 39*omegaVar*m_xy_I + 24*omegaVar*m_xz_I + 2*omegaVar*m_yy_I - 7*omegaVar*m_zz_I + 60)/(90*(4*m_xx_I + 6*m_xy_I - 6*m_xz_I + m_yy_I + m_zz_I - 4*omegaVar*m_xx_I - 6*omegaVar*m_xy_I + 6*omegaVar*m_xz_I - omegaVar*m_yy_I - omegaVar*m_zz_I + 15));
    m_xz_t90 = (1746*m_xz_I - 24*m_xy_I - 180*m_xx_I + 78*m_yy_I - 168*m_zz_I + 10*omegaVar*m_xx_I + 24*omegaVar*m_xy_I + 39*omegaVar*m_xz_I + 7*omegaVar*m_yy_I - 2*omegaVar*m_zz_I - 60)/(90*(4*m_xx_I + 6*m_xy_I - 6*m_xz_I + m_yy_I + m_zz_I - 4*omegaVar*m_xx_I - 6*omegaVar*m_xy_I + 6*omegaVar*m_xz_I - omegaVar*m_yy_I - omegaVar*m_zz_I + 15));
    m_yy_t45 = (90*m_xx_I + 504*m_xy_I + 234*m_xz_I + 2052*m_yy_I - 162*m_zz_I - 5*omegaVar*m_xx_I + 6*omegaVar*m_xy_I + 21*omegaVar*m_xz_I + 73*omegaVar*m_yy_I - 8*omegaVar*m_zz_I + 30)/(135*(4*m_xx_I + 6*m_xy_I - 6*m_xz_I + m_yy_I + m_zz_I - 4*omegaVar*m_xx_I - 6*omegaVar*m_xy_I + 6*omegaVar*m_xz_I - omegaVar*m_yy_I - omegaVar*m_zz_I + 15));
    m_yz_t90 = (m_yz_I*(3*omegaVar + 82))/(6*(4*m_xx_I + 6*m_xy_I - 6*m_xz_I + m_yy_I + m_zz_I - 4*omegaVar*m_xx_I - 6*omegaVar*m_xy_I + 6*omegaVar*m_xz_I - omegaVar*m_yy_I - omegaVar*m_zz_I + 15));
    m_zz_t45 = -(234*m_xy_I - 90*m_xx_I + 504*m_xz_I + 162*m_yy_I - 2052*m_zz_I + 5*omegaVar*m_xx_I + 21*omegaVar*m_xy_I + 6*omegaVar*m_xz_I + 8*omegaVar*m_yy_I - 73*omegaVar*m_zz_I - 30)/(135*(4*m_xx_I + 6*m_xy_I - 6*m_xz_I + m_yy_I + m_zz_I - 4*omegaVar*m_xx_I - 6*omegaVar*m_xy_I + 6*omegaVar*m_xz_I - omegaVar*m_yy_I - omegaVar*m_zz_I + 15));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = (m_xz_t90/2 - m_xy_t90/2 - m_xx_t45/3 - m_yy_t45/12 - m_zz_t45/12 + (omegaVar*m_xx_t45)/6 + (omegaVar*m_xy_t90)/4 - (omegaVar*m_xz_t90)/4 + (omegaVar*m_yy_t45)/24 + (omegaVar*m_zz_t45)/24 - 1.0/9.0)*rho + RHO_0/9;
    L_BC_Fy = (m_zz_t45/12 - m_xy_t90/2 - m_yy_t45/6 - m_xx_t45/6 + (omegaVar*m_xx_t45)/12 + (omegaVar*m_xy_t90)/4 + (omegaVar*m_yy_t45)/12 - (omegaVar*m_zz_t45)/24 - 1.0/18.0)*rho + RHO_0/18;
    L_BC_Fz = (m_xx_t45/6 - m_xz_t90/2 - m_yy_t45/12 + m_zz_t45/6 - (omegaVar*m_xx_t45)/12 + (omegaVar*m_xz_t90)/4 + (omegaVar*m_yy_t45)/24 - (omegaVar*m_zz_t45)/12 + 1.0/18.0)*rho - RHO_0/18;
    #endif //_BC_FORCES 

    break;
case 0b0000010101011010:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (6*(3*rho_I + 2*m_xx_I*rho_I - 6*m_xz_I*rho_I - m_yy_I*rho_I + 2*m_zz_I*rho_I - 2*omegaVar*m_xx_I*rho_I + 6*omegaVar*m_xz_I*rho_I + omegaVar*m_yy_I*rho_I - 2*omegaVar*m_zz_I*rho_I))/(3*omegaVar + 14);

    m_xx_t45 = (198*m_xx_I - 216*m_xz_I - 36*m_yy_I + 72*m_zz_I + 23*omegaVar*m_xx_I + 12*omegaVar*m_xz_I + 2*omegaVar*m_yy_I - 4*omegaVar*m_zz_I + 24)/(54*(2*m_xx_I - 6*m_xz_I - m_yy_I + 2*m_zz_I - 2*omegaVar*m_xx_I + 6*omegaVar*m_xz_I + omegaVar*m_yy_I - 2*omegaVar*m_zz_I + 3));
    m_xy_t90 = (m_xy_I*(3*omegaVar + 14))/(6*(2*m_xx_I - 6*m_xz_I - m_yy_I + 2*m_zz_I - 2*omegaVar*m_xx_I + 6*omegaVar*m_xz_I + omegaVar*m_yy_I - 2*omegaVar*m_zz_I + 3));
    m_xz_t90 = (150*m_xz_I - 36*m_xx_I + 18*m_yy_I - 36*m_zz_I + 2*omegaVar*m_xx_I + 3*omegaVar*m_xz_I - omegaVar*m_yy_I + 2*omegaVar*m_zz_I - 12)/(18*(2*m_xx_I - 6*m_xz_I - m_yy_I + 2*m_zz_I - 2*omegaVar*m_xx_I + 6*omegaVar*m_xz_I + omegaVar*m_yy_I - 2*omegaVar*m_zz_I + 3));
    m_yy_t45 = (54*m_xz_I - 18*m_xx_I + 72*m_yy_I - 18*m_zz_I + omegaVar*m_xx_I - 3*omegaVar*m_xz_I + 13*omegaVar*m_yy_I + omegaVar*m_zz_I - 6)/(27*(2*m_xx_I - 6*m_xz_I - m_yy_I + 2*m_zz_I - 2*omegaVar*m_xx_I + 6*omegaVar*m_xz_I + omegaVar*m_yy_I - 2*omegaVar*m_zz_I + 3));
    m_yz_t90 = (m_yz_I*(3*omegaVar + 14))/(6*(2*m_xx_I - 6*m_xz_I - m_yy_I + 2*m_zz_I - 2*omegaVar*m_xx_I + 6*omegaVar*m_xz_I + omegaVar*m_yy_I - 2*omegaVar*m_zz_I + 3));
    m_zz_t45 = (72*m_xx_I - 216*m_xz_I - 36*m_yy_I + 198*m_zz_I - 4*omegaVar*m_xx_I + 12*omegaVar*m_xz_I + 2*omegaVar*m_yy_I + 23*omegaVar*m_zz_I + 24)/(54*(2*m_xx_I - 6*m_xz_I - m_yy_I + 2*m_zz_I - 2*omegaVar*m_xx_I + 6*omegaVar*m_xz_I + omegaVar*m_yy_I - 2*omegaVar*m_zz_I + 3));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = 0;
    L_BC_Fy = 0;
    L_BC_Fz = 0;
    #endif //_BC_FORCES 

    break;
case 0b0000010101011011:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[8] + pop[9] + pop[10] + pop[12] + pop[13] + pop[14] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (3*(15*rho_I + 11*m_xx_I*rho_I + 6*m_xy_I*rho_I - 30*m_xz_I*rho_I - m_yy_I*rho_I + 6*m_yz_I*rho_I + 11*m_zz_I*rho_I - 11*omegaVar*m_xx_I*rho_I - 6*omegaVar*m_xy_I*rho_I + 30*omegaVar*m_xz_I*rho_I + omegaVar*m_yy_I*rho_I - 6*omegaVar*m_yz_I*rho_I - 11*omegaVar*m_zz_I*rho_I))/(9*omegaVar + 31);

    m_xx_t45 = -(2*(1260*m_xz_I - 252*m_xy_I - 1206*m_xx_I + 135*m_yy_I + 27*m_yz_I - 369*m_zz_I - 194*omegaVar*m_xx_I + 12*omegaVar*m_xy_I - 60*omegaVar*m_xz_I + 25*omegaVar*m_yy_I + 93*omegaVar*m_yz_I + 49*omegaVar*m_zz_I - 165))/(135*(11*m_xx_I + 6*m_xy_I - 30*m_xz_I - m_yy_I + 6*m_yz_I + 11*m_zz_I - 11*omegaVar*m_xx_I - 6*omegaVar*m_xy_I + 30*omegaVar*m_xz_I + omegaVar*m_yy_I - 6*omegaVar*m_yz_I - 11*omegaVar*m_zz_I + 15));
    m_xy_t90 = (84*m_xx_I + 663*m_xy_I - 60*m_xz_I + 60*m_yy_I + 12*m_yz_I - 9*m_zz_I - 4*omegaVar*m_xx_I + 177*omegaVar*m_xy_I + 60*omegaVar*m_xz_I + 20*omegaVar*m_yy_I - 12*omegaVar*m_yz_I - 31*omegaVar*m_zz_I + 30)/(45*(11*m_xx_I + 6*m_xy_I - 30*m_xz_I - m_yy_I + 6*m_yz_I + 11*m_zz_I - 11*omegaVar*m_xx_I - 6*omegaVar*m_xy_I + 30*omegaVar*m_xz_I + omegaVar*m_yy_I - 6*omegaVar*m_yz_I - 11*omegaVar*m_zz_I + 15));
    m_xz_t90 = (339*m_xz_I - 12*m_xy_I - 84*m_xx_I + 33*m_yy_I - 12*m_yz_I - 84*m_zz_I + 4*omegaVar*m_xx_I + 12*omegaVar*m_xy_I + 21*omegaVar*m_xz_I + 7*omegaVar*m_yy_I + 12*omegaVar*m_yz_I + 4*omegaVar*m_zz_I - 30)/(9*(11*m_xx_I + 6*m_xy_I - 30*m_xz_I - m_yy_I + 6*m_yz_I + 11*m_zz_I - 11*omegaVar*m_xx_I - 6*omegaVar*m_xy_I + 30*omegaVar*m_xz_I + omegaVar*m_yy_I - 6*omegaVar*m_yz_I - 11*omegaVar*m_zz_I + 15));
    m_yy_t45 = (2*(36*m_xy_I - 27*m_xx_I + 99*m_xz_I + 180*m_yy_I + 36*m_yz_I - 27*m_zz_I - 5*omegaVar*m_xx_I + 12*omegaVar*m_xy_I + 21*omegaVar*m_xz_I + 52*omegaVar*m_yy_I + 12*omegaVar*m_yz_I - 5*omegaVar*m_zz_I - 3))/(27*(11*m_xx_I + 6*m_xy_I - 30*m_xz_I - m_yy_I + 6*m_yz_I + 11*m_zz_I - 11*omegaVar*m_xx_I - 6*omegaVar*m_xy_I + 30*omegaVar*m_xz_I + omegaVar*m_yy_I - 6*omegaVar*m_yz_I - 11*omegaVar*m_zz_I + 15));
    m_yz_t90 = (12*m_xy_I - 9*m_xx_I - 60*m_xz_I + 60*m_yy_I + 663*m_yz_I + 84*m_zz_I - 31*omegaVar*m_xx_I - 12*omegaVar*m_xy_I + 60*omegaVar*m_xz_I + 20*omegaVar*m_yy_I + 177*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 30)/(45*(11*m_xx_I + 6*m_xy_I - 30*m_xz_I - m_yy_I + 6*m_yz_I + 11*m_zz_I - 11*omegaVar*m_xx_I - 6*omegaVar*m_xy_I + 30*omegaVar*m_xz_I + omegaVar*m_yy_I - 6*omegaVar*m_yz_I - 11*omegaVar*m_zz_I + 15));
    m_zz_t45 = -(2*(27*m_xy_I - 369*m_xx_I + 1260*m_xz_I + 135*m_yy_I - 252*m_yz_I - 1206*m_zz_I + 49*omegaVar*m_xx_I + 93*omegaVar*m_xy_I - 60*omegaVar*m_xz_I + 25*omegaVar*m_yy_I + 12*omegaVar*m_yz_I - 194*omegaVar*m_zz_I - 165))/(135*(11*m_xx_I + 6*m_xy_I - 30*m_xz_I - m_yy_I + 6*m_yz_I + 11*m_zz_I - 11*omegaVar*m_xx_I - 6*omegaVar*m_xy_I + 30*omegaVar*m_xz_I + omegaVar*m_yy_I - 6*omegaVar*m_yz_I - 11*omegaVar*m_zz_I + 15));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = (m_zz_t45/12 - m_xy_t90/2 - m_yy_t45/6 - m_xx_t45/6 + (omegaVar*m_xx_t45)/12 + (omegaVar*m_xy_t90)/4 + (omegaVar*m_yy_t45)/12 - (omegaVar*m_zz_t45)/24 - 1.0/18.0)*rho + RHO_0/18;
    L_BC_Fy = ((omegaVar*m_xx_t45)/24 - m_xy_t90/2 - m_yy_t45/3 - m_yz_t90/2 - m_zz_t45/12 - m_xx_t45/12 + (omegaVar*m_xy_t90)/4 + (omegaVar*m_yy_t45)/6 + (omegaVar*m_yz_t90)/4 + (omegaVar*m_zz_t45)/24 - 1.0/9.0)*rho + RHO_0/9;
    L_BC_Fz = (m_xx_t45/12 - m_yy_t45/6 - m_yz_t90/2 - m_zz_t45/6 - (omegaVar*m_xx_t45)/24 + (omegaVar*m_yy_t45)/12 + (omegaVar*m_yz_t90)/4 + (omegaVar*m_zz_t45)/12 - 1.0/18.0)*rho + RHO_0/18;
    #endif //_BC_FORCES 

    break;
case 0b0000010101011100:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[14] + pop[16] + pop[17];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17]);

    rho = (12*(5*rho_I + m_xx_I*rho_I - 2*m_xy_I*rho_I - 2*m_xz_I*rho_I + m_yy_I*rho_I - 2*m_yz_I*rho_I + m_zz_I*rho_I - omegaVar*m_xx_I*rho_I + 2*omegaVar*m_xy_I*rho_I + 2*omegaVar*m_xz_I*rho_I - omegaVar*m_yy_I*rho_I + 2*omegaVar*m_yz_I*rho_I - omegaVar*m_zz_I*rho_I))/(3*omegaVar + 52);

    m_xx_t45 = (472*m_xx_I - 112*m_xy_I - 112*m_xz_I + 4*m_yy_I + 44*m_yz_I + 4*m_zz_I + 23*omegaVar*m_xx_I + 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - 4*omegaVar*m_yy_I + 11*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 20)/(90*(m_xx_I - 2*m_xy_I - 2*m_xz_I + m_yy_I - 2*m_yz_I + m_zz_I - omegaVar*m_xx_I + 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - omegaVar*m_yy_I + 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_xy_t90 = (1108*m_xy_I - 112*m_xx_I + 16*m_xz_I - 112*m_yy_I + 16*m_yz_I + 44*m_zz_I + 2*omegaVar*m_xx_I + 47*omegaVar*m_xy_I - 16*omegaVar*m_xz_I + 2*omegaVar*m_yy_I - 16*omegaVar*m_yz_I + 11*omegaVar*m_zz_I - 40)/(180*(m_xx_I - 2*m_xy_I - 2*m_xz_I + m_yy_I - 2*m_yz_I + m_zz_I - omegaVar*m_xx_I + 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - omegaVar*m_yy_I + 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_xz_t90 = (16*m_xy_I - 112*m_xx_I + 1108*m_xz_I + 44*m_yy_I + 16*m_yz_I - 112*m_zz_I + 2*omegaVar*m_xx_I - 16*omegaVar*m_xy_I + 47*omegaVar*m_xz_I + 11*omegaVar*m_yy_I - 16*omegaVar*m_yz_I + 2*omegaVar*m_zz_I - 40)/(180*(m_xx_I - 2*m_xy_I - 2*m_xz_I + m_yy_I - 2*m_yz_I + m_zz_I - omegaVar*m_xx_I + 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - omegaVar*m_yy_I + 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_yy_t45 = (4*m_xx_I - 112*m_xy_I + 44*m_xz_I + 472*m_yy_I - 112*m_yz_I + 4*m_zz_I - 4*omegaVar*m_xx_I + 2*omegaVar*m_xy_I + 11*omegaVar*m_xz_I + 23*omegaVar*m_yy_I + 2*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 20)/(90*(m_xx_I - 2*m_xy_I - 2*m_xz_I + m_yy_I - 2*m_yz_I + m_zz_I - omegaVar*m_xx_I + 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - omegaVar*m_yy_I + 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_yz_t90 = (44*m_xx_I + 16*m_xy_I + 16*m_xz_I - 112*m_yy_I + 1108*m_yz_I - 112*m_zz_I + 11*omegaVar*m_xx_I - 16*omegaVar*m_xy_I - 16*omegaVar*m_xz_I + 2*omegaVar*m_yy_I + 47*omegaVar*m_yz_I + 2*omegaVar*m_zz_I - 40)/(180*(m_xx_I - 2*m_xy_I - 2*m_xz_I + m_yy_I - 2*m_yz_I + m_zz_I - omegaVar*m_xx_I + 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - omegaVar*m_yy_I + 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_zz_t45 = (4*m_xx_I + 44*m_xy_I - 112*m_xz_I + 4*m_yy_I - 112*m_yz_I + 472*m_zz_I - 4*omegaVar*m_xx_I + 11*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - 4*omegaVar*m_yy_I + 2*omegaVar*m_yz_I + 23*omegaVar*m_zz_I + 20)/(90*(m_xx_I - 2*m_xy_I - 2*m_xz_I + m_yy_I - 2*m_yz_I + m_zz_I - omegaVar*m_xx_I + 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - omegaVar*m_yy_I + 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = (m_xy_t90/2 - m_xx_t45/3 + m_xz_t90/2 - m_yy_t45/12 - m_zz_t45/12 + (omegaVar*m_xx_t45)/6 - (omegaVar*m_xy_t90)/4 - (omegaVar*m_xz_t90)/4 + (omegaVar*m_yy_t45)/24 + (omegaVar*m_zz_t45)/24 - 1.0/9.0)*rho + RHO_0/9;
    L_BC_Fy = (m_xx_t45/12 - m_xy_t90/2 + m_yy_t45/3 - m_yz_t90/2 + m_zz_t45/12 - (omegaVar*m_xx_t45)/24 + (omegaVar*m_xy_t90)/4 - (omegaVar*m_yy_t45)/6 + (omegaVar*m_yz_t90)/4 - (omegaVar*m_zz_t45)/24 + 1.0/9.0)*rho - RHO_0/9;
    L_BC_Fz = -(rho*(omegaVar - 2)*(m_xx_t45 - 2*m_xz_t90 - m_yy_t45 + 2*m_yz_t90))/8;
    #endif //_BC_FORCES 

    break;
case 0b0000010101011101:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[8] + pop[10] + pop[11] + pop[12] + pop[14] + pop[16] + pop[17];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17]);
    m_xy_I = inv_rho_I * (pop[8] - pop[14]);
    m_xz_I = inv_rho_I * (pop[10] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17]);

    rho = (36*(107*rho_I + 72*m_xx_I*rho_I + 12*m_yy_I*rho_I - 33*m_yz_I*rho_I + 12*m_zz_I*rho_I - 72*omegaVar*m_xx_I*rho_I - 12*omegaVar*m_yy_I*rho_I + 33*omegaVar*m_yz_I*rho_I - 12*omegaVar*m_zz_I*rho_I))/(313*omegaVar + 2790);

    m_xx_t45 = (2358*m_xx_I - 72*m_yy_I + 198*m_yz_I - 72*m_zz_I + 49*omegaVar*m_xx_I - 44*omegaVar*m_yy_I + 121*omegaVar*m_yz_I - 44*omegaVar*m_zz_I + 288)/(18*(72*m_xx_I + 12*m_yy_I - 33*m_yz_I + 12*m_zz_I - 72*omegaVar*m_xx_I - 12*omegaVar*m_yy_I + 33*omegaVar*m_yz_I - 12*omegaVar*m_zz_I + 107));
    m_xy_t90 = (m_xy_I*(313*omegaVar + 2790))/(18*(72*m_xx_I + 12*m_yy_I - 33*m_yz_I + 12*m_zz_I - 72*omegaVar*m_xx_I - 12*omegaVar*m_yy_I + 33*omegaVar*m_yz_I - 12*omegaVar*m_zz_I + 107));
    m_xz_t90 = (m_xz_I*(313*omegaVar + 2790))/(18*(72*m_xx_I + 12*m_yy_I - 33*m_yz_I + 12*m_zz_I - 72*omegaVar*m_xx_I - 12*omegaVar*m_yy_I + 33*omegaVar*m_yz_I - 12*omegaVar*m_zz_I + 107));
    m_yy_t45 = -(108*m_xx_I - 2772*m_yy_I + 648*m_yz_I + 18*m_zz_I + 66*omegaVar*m_xx_I - 302*omegaVar*m_yy_I + 48*omegaVar*m_yz_I + 11*omegaVar*m_zz_I - 72)/(27*(72*m_xx_I + 12*m_yy_I - 33*m_yz_I + 12*m_zz_I - 72*omegaVar*m_xx_I - 12*omegaVar*m_yy_I + 33*omegaVar*m_yz_I - 12*omegaVar*m_zz_I + 107));
    m_yz_t90 = -(432*m_yy_I - 198*m_xx_I - 3978*m_yz_I + 432*m_zz_I - 121*omegaVar*m_xx_I + 32*omegaVar*m_yy_I - 401*omegaVar*m_yz_I + 32*omegaVar*m_zz_I + 132)/(36*(72*m_xx_I + 12*m_yy_I - 33*m_yz_I + 12*m_zz_I - 72*omegaVar*m_xx_I - 12*omegaVar*m_yy_I + 33*omegaVar*m_yz_I - 12*omegaVar*m_zz_I + 107));
    m_zz_t45 = -(108*m_xx_I + 18*m_yy_I + 648*m_yz_I - 2772*m_zz_I + 66*omegaVar*m_xx_I + 11*omegaVar*m_yy_I + 48*omegaVar*m_yz_I - 302*omegaVar*m_zz_I - 72)/(27*(72*m_xx_I + 12*m_yy_I - 33*m_yz_I + 12*m_zz_I - 72*omegaVar*m_xx_I - 12*omegaVar*m_yy_I + 33*omegaVar*m_yz_I - 12*omegaVar*m_zz_I + 107));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = ((omegaVar*m_xx_t45)/2 - m_xx_t45 - 1.0/3.0)*rho + RHO_0/3;
    L_BC_Fy = (m_yy_t45/6 - m_xy_t90 - m_xx_t45/12 - m_yz_t90/2 + m_zz_t45/6 + (omegaVar*m_xx_t45)/24 + (omegaVar*m_xy_t90)/2 - (omegaVar*m_yy_t45)/12 + (omegaVar*m_yz_t90)/4 - (omegaVar*m_zz_t45)/12 + 1.0/18.0)*rho - RHO_0/18;
    L_BC_Fz = (m_xx_t45/12 - m_xz_t90 - m_yy_t45/6 + m_yz_t90/2 - m_zz_t45/6 - (omegaVar*m_xx_t45)/24 + (omegaVar*m_xz_t90)/2 + (omegaVar*m_yy_t45)/12 - (omegaVar*m_yz_t90)/4 + (omegaVar*m_zz_t45)/12 - 1.0/18.0)*rho + RHO_0/18;
    #endif //_BC_FORCES 

    break;
case 0b0000010101011110:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[14] + pop[17];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[17]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[17]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[17]);

    rho = (3*(15*rho_I + 11*m_xx_I*rho_I - 6*m_xy_I*rho_I - 30*m_xz_I*rho_I - m_yy_I*rho_I - 6*m_yz_I*rho_I + 11*m_zz_I*rho_I - 11*omegaVar*m_xx_I*rho_I + 6*omegaVar*m_xy_I*rho_I + 30*omegaVar*m_xz_I*rho_I + omegaVar*m_yy_I*rho_I + 6*omegaVar*m_yz_I*rho_I - 11*omegaVar*m_zz_I*rho_I))/(9*omegaVar + 31);

    m_xx_t45 = (2*(1206*m_xx_I - 252*m_xy_I - 1260*m_xz_I - 135*m_yy_I + 27*m_yz_I + 369*m_zz_I + 194*omegaVar*m_xx_I + 12*omegaVar*m_xy_I + 60*omegaVar*m_xz_I - 25*omegaVar*m_yy_I + 93*omegaVar*m_yz_I - 49*omegaVar*m_zz_I + 165))/(135*(11*m_xx_I - 6*m_xy_I - 30*m_xz_I - m_yy_I - 6*m_yz_I + 11*m_zz_I - 11*omegaVar*m_xx_I + 6*omegaVar*m_xy_I + 30*omegaVar*m_xz_I + omegaVar*m_yy_I + 6*omegaVar*m_yz_I - 11*omegaVar*m_zz_I + 15));
    m_xy_t90 = (663*m_xy_I - 84*m_xx_I + 60*m_xz_I - 60*m_yy_I + 12*m_yz_I + 9*m_zz_I + 4*omegaVar*m_xx_I + 177*omegaVar*m_xy_I - 60*omegaVar*m_xz_I - 20*omegaVar*m_yy_I - 12*omegaVar*m_yz_I + 31*omegaVar*m_zz_I - 30)/(45*(11*m_xx_I - 6*m_xy_I - 30*m_xz_I - m_yy_I - 6*m_yz_I + 11*m_zz_I - 11*omegaVar*m_xx_I + 6*omegaVar*m_xy_I + 30*omegaVar*m_xz_I + omegaVar*m_yy_I + 6*omegaVar*m_yz_I - 11*omegaVar*m_zz_I + 15));
    m_xz_t90 = (12*m_xy_I - 84*m_xx_I + 339*m_xz_I + 33*m_yy_I + 12*m_yz_I - 84*m_zz_I + 4*omegaVar*m_xx_I - 12*omegaVar*m_xy_I + 21*omegaVar*m_xz_I + 7*omegaVar*m_yy_I - 12*omegaVar*m_yz_I + 4*omegaVar*m_zz_I - 30)/(9*(11*m_xx_I - 6*m_xy_I - 30*m_xz_I - m_yy_I - 6*m_yz_I + 11*m_zz_I - 11*omegaVar*m_xx_I + 6*omegaVar*m_xy_I + 30*omegaVar*m_xz_I + omegaVar*m_yy_I + 6*omegaVar*m_yz_I - 11*omegaVar*m_zz_I + 15));
    m_yy_t45 = -(2*(27*m_xx_I + 36*m_xy_I - 99*m_xz_I - 180*m_yy_I + 36*m_yz_I + 27*m_zz_I + 5*omegaVar*m_xx_I + 12*omegaVar*m_xy_I - 21*omegaVar*m_xz_I - 52*omegaVar*m_yy_I + 12*omegaVar*m_yz_I + 5*omegaVar*m_zz_I + 3))/(27*(11*m_xx_I - 6*m_xy_I - 30*m_xz_I - m_yy_I - 6*m_yz_I + 11*m_zz_I - 11*omegaVar*m_xx_I + 6*omegaVar*m_xy_I + 30*omegaVar*m_xz_I + omegaVar*m_yy_I + 6*omegaVar*m_yz_I - 11*omegaVar*m_zz_I + 15));
    m_yz_t90 = (9*m_xx_I + 12*m_xy_I + 60*m_xz_I - 60*m_yy_I + 663*m_yz_I - 84*m_zz_I + 31*omegaVar*m_xx_I - 12*omegaVar*m_xy_I - 60*omegaVar*m_xz_I - 20*omegaVar*m_yy_I + 177*omegaVar*m_yz_I + 4*omegaVar*m_zz_I - 30)/(45*(11*m_xx_I - 6*m_xy_I - 30*m_xz_I - m_yy_I - 6*m_yz_I + 11*m_zz_I - 11*omegaVar*m_xx_I + 6*omegaVar*m_xy_I + 30*omegaVar*m_xz_I + omegaVar*m_yy_I + 6*omegaVar*m_yz_I - 11*omegaVar*m_zz_I + 15));
    m_zz_t45 = (2*(369*m_xx_I + 27*m_xy_I - 1260*m_xz_I - 135*m_yy_I - 252*m_yz_I + 1206*m_zz_I - 49*omegaVar*m_xx_I + 93*omegaVar*m_xy_I + 60*omegaVar*m_xz_I - 25*omegaVar*m_yy_I + 12*omegaVar*m_yz_I + 194*omegaVar*m_zz_I + 165))/(135*(11*m_xx_I - 6*m_xy_I - 30*m_xz_I - m_yy_I - 6*m_yz_I + 11*m_zz_I - 11*omegaVar*m_xx_I + 6*omegaVar*m_xy_I + 30*omegaVar*m_xz_I + omegaVar*m_yy_I + 6*omegaVar*m_yz_I - 11*omegaVar*m_zz_I + 15));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = (m_xy_t90/2 - m_xx_t45/6 - m_yy_t45/6 + m_zz_t45/12 + (omegaVar*m_xx_t45)/12 - (omegaVar*m_xy_t90)/4 + (omegaVar*m_yy_t45)/12 - (omegaVar*m_zz_t45)/24 - 1.0/18.0)*rho + RHO_0/18;
    L_BC_Fy = (m_xx_t45/12 - m_xy_t90/2 + m_yy_t45/3 - m_yz_t90/2 + m_zz_t45/12 - (omegaVar*m_xx_t45)/24 + (omegaVar*m_xy_t90)/4 - (omegaVar*m_yy_t45)/6 + (omegaVar*m_yz_t90)/4 - (omegaVar*m_zz_t45)/24 + 1.0/9.0)*rho - RHO_0/9;
    L_BC_Fz = (m_xx_t45/12 - m_yy_t45/6 + m_yz_t90/2 - m_zz_t45/6 - (omegaVar*m_xx_t45)/24 + (omegaVar*m_yy_t45)/12 - (omegaVar*m_yz_t90)/4 + (omegaVar*m_zz_t45)/12 - 1.0/18.0)*rho + RHO_0/18;
    #endif //_BC_FORCES 

    break;
case 0b0000010101011111:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[2] + pop[3] + pop[4] + pop[6] + pop[8] + pop[10] + pop[12] + pop[14] + pop[17];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[17]);
    m_xy_I = inv_rho_I * (pop[8] - pop[14]);
    m_xz_I = inv_rho_I * (pop[10]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[17]);
    m_yz_I = inv_rho_I * (pop[12] - pop[17]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[17]);

    rho = (36*(23*rho_I + 24*m_xx_I*rho_I - 57*m_xz_I*rho_I - 6*m_yy_I*rho_I + 24*m_zz_I*rho_I - 24*omegaVar*m_xx_I*rho_I + 57*omegaVar*m_xz_I*rho_I + 6*omegaVar*m_yy_I*rho_I - 24*omegaVar*m_zz_I*rho_I))/(5*(43*omegaVar + 72));

    m_xx_t45 = (936*m_xx_I - 1008*m_xz_I - 144*m_yy_I + 216*m_zz_I + 239*omegaVar*m_xx_I + 158*omegaVar*m_xz_I - 6*omegaVar*m_yy_I - 191*omegaVar*m_zz_I + 192)/(36*(24*m_xx_I - 57*m_xz_I - 6*m_yy_I + 24*m_zz_I - 24*omegaVar*m_xx_I + 57*omegaVar*m_xz_I + 6*omegaVar*m_yy_I - 24*omegaVar*m_zz_I + 23));
    m_xy_t90 = (5*m_xy_I*(43*omegaVar + 72))/(18*(24*m_xx_I - 57*m_xz_I - 6*m_yy_I + 24*m_zz_I - 24*omegaVar*m_xx_I + 57*omegaVar*m_xz_I + 6*omegaVar*m_yy_I - 24*omegaVar*m_zz_I + 23));
    m_xz_t90 = (2412*m_xz_I - 504*m_xx_I + 216*m_yy_I - 504*m_zz_I + 79*omegaVar*m_xx_I + 538*omegaVar*m_xz_I + 34*omegaVar*m_yy_I + 79*omegaVar*m_zz_I - 228)/(36*(24*m_xx_I - 57*m_xz_I - 6*m_yy_I + 24*m_zz_I - 24*omegaVar*m_xx_I + 57*omegaVar*m_xz_I + 6*omegaVar*m_yy_I - 24*omegaVar*m_zz_I + 23));
    m_yy_t45 = -(72*m_xx_I - 216*m_xz_I - 288*m_yy_I + 72*m_zz_I + 3*omegaVar*m_xx_I - 34*omegaVar*m_xz_I - 162*omegaVar*m_yy_I + 3*omegaVar*m_zz_I + 24)/(18*(24*m_xx_I - 57*m_xz_I - 6*m_yy_I + 24*m_zz_I - 24*omegaVar*m_xx_I + 57*omegaVar*m_xz_I + 6*omegaVar*m_yy_I - 24*omegaVar*m_zz_I + 23));
    m_yz_t90 = (5*m_yz_I*(43*omegaVar + 72))/(18*(24*m_xx_I - 57*m_xz_I - 6*m_yy_I + 24*m_zz_I - 24*omegaVar*m_xx_I + 57*omegaVar*m_xz_I + 6*omegaVar*m_yy_I - 24*omegaVar*m_zz_I + 23));
    m_zz_t45 = (216*m_xx_I - 1008*m_xz_I - 144*m_yy_I + 936*m_zz_I - 191*omegaVar*m_xx_I + 158*omegaVar*m_xz_I - 6*omegaVar*m_yy_I + 239*omegaVar*m_zz_I + 192)/(36*(24*m_xx_I - 57*m_xz_I - 6*m_yy_I + 24*m_zz_I - 24*omegaVar*m_xx_I + 57*omegaVar*m_xz_I + 6*omegaVar*m_yy_I - 24*omegaVar*m_zz_I + 23));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = (m_zz_t45/6 - m_xz_t90/2 - m_yy_t45/12 - (5*m_xx_t45)/6 + (5*omegaVar*m_xx_t45)/12 + (omegaVar*m_xz_t90)/4 + (omegaVar*m_yy_t45)/24 - (omegaVar*m_zz_t45)/12 - 5/18)*rho + (5*RHO_0)/18;
    L_BC_Fy = (rho*(m_xy_t90 + m_yz_t90)*(omegaVar - 2))/2;
    L_BC_Fz = (m_xx_t45/6 - m_xz_t90/2 - m_yy_t45/12 - (5*m_zz_t45)/6 - (omegaVar*m_xx_t45)/12 + (omegaVar*m_xz_t90)/4 + (omegaVar*m_yy_t45)/24 + (5*omegaVar*m_zz_t45)/12 - 5/18)*rho + (5*RHO_0)/18;
    #endif //_BC_FORCES 

    break;
case 0b0000010101100000:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = rho_I;

    m_xx_t45 = m_xx_I;
    m_xy_t90 = m_xy_I;
    m_xz_t90 = m_xz_I;
    m_yy_t45 = m_yy_I;
    m_yz_t90 = m_yz_I;
    m_zz_t45 = m_zz_I;

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = 0;
    L_BC_Fy = 0;
    L_BC_Fz = 0;
    #endif //_BC_FORCES 

    break;
case 0b0000010101100001:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = rho_I;

    m_xx_t45 = m_xx_I;
    m_xy_t90 = m_xy_I;
    m_xz_t90 = m_xz_I;
    m_yy_t45 = m_yy_I;
    m_yz_t90 = m_yz_I;
    m_zz_t45 = m_zz_I;

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = 0;
    L_BC_Fy = 0;
    L_BC_Fz = 0;
    #endif //_BC_FORCES 

    break;
case 0b0000010101100010:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (12*(15*rho_I + 2*m_xx_I*rho_I - 6*m_xy_I*rho_I + 2*m_yy_I*rho_I - m_zz_I*rho_I - 2*omegaVar*m_xx_I*rho_I + 6*omegaVar*m_xy_I*rho_I - 2*omegaVar*m_yy_I*rho_I + omegaVar*m_zz_I*rho_I))/(3*omegaVar + 172);

    m_xx_t45 = (1692*m_xx_I - 432*m_xy_I + 144*m_yy_I - 72*m_zz_I + 23*omegaVar*m_xx_I + 12*omegaVar*m_xy_I - 4*omegaVar*m_yy_I + 2*omegaVar*m_zz_I + 48)/(108*(2*m_xx_I - 6*m_xy_I + 2*m_yy_I - m_zz_I - 2*omegaVar*m_xx_I + 6*omegaVar*m_xy_I - 2*omegaVar*m_yy_I + omegaVar*m_zz_I + 15));
    m_xy_t90 = (732*m_xy_I - 72*m_xx_I - 72*m_yy_I + 36*m_zz_I + 2*omegaVar*m_xx_I + 3*omegaVar*m_xy_I + 2*omegaVar*m_yy_I - omegaVar*m_zz_I - 24)/(36*(2*m_xx_I - 6*m_xy_I + 2*m_yy_I - m_zz_I - 2*omegaVar*m_xx_I + 6*omegaVar*m_xy_I - 2*omegaVar*m_yy_I + omegaVar*m_zz_I + 15));
    m_xz_t90 = (m_xz_I*(3*omegaVar + 172))/(12*(2*m_xx_I - 6*m_xy_I + 2*m_yy_I - m_zz_I - 2*omegaVar*m_xx_I + 6*omegaVar*m_xy_I - 2*omegaVar*m_yy_I + omegaVar*m_zz_I + 15));
    m_yy_t45 = (144*m_xx_I - 432*m_xy_I + 1692*m_yy_I - 72*m_zz_I - 4*omegaVar*m_xx_I + 12*omegaVar*m_xy_I + 23*omegaVar*m_yy_I + 2*omegaVar*m_zz_I + 48)/(108*(2*m_xx_I - 6*m_xy_I + 2*m_yy_I - m_zz_I - 2*omegaVar*m_xx_I + 6*omegaVar*m_xy_I - 2*omegaVar*m_yy_I + omegaVar*m_zz_I + 15));
    m_yz_t90 = (m_yz_I*(3*omegaVar + 172))/(12*(2*m_xx_I - 6*m_xy_I + 2*m_yy_I - m_zz_I - 2*omegaVar*m_xx_I + 6*omegaVar*m_xy_I - 2*omegaVar*m_yy_I + omegaVar*m_zz_I + 15));
    m_zz_t45 = (108*m_xy_I - 36*m_xx_I - 36*m_yy_I + 792*m_zz_I + omegaVar*m_xx_I - 3*omegaVar*m_xy_I + omegaVar*m_yy_I + 13*omegaVar*m_zz_I - 12)/(54*(2*m_xx_I - 6*m_xy_I + 2*m_yy_I - m_zz_I - 2*omegaVar*m_xx_I + 6*omegaVar*m_xy_I - 2*omegaVar*m_yy_I + omegaVar*m_zz_I + 15));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = (m_xx_t45/6 - m_xy_t90/2 + m_yy_t45/6 - m_zz_t45/12 - (omegaVar*m_xx_t45)/12 + (omegaVar*m_xy_t90)/4 - (omegaVar*m_yy_t45)/12 + (omegaVar*m_zz_t45)/24 + 1.0/18.0)*rho - RHO_0/18;
    L_BC_Fy = (m_xy_t90/2 - m_xx_t45/6 - m_yy_t45/6 + m_zz_t45/12 + (omegaVar*m_xx_t45)/12 - (omegaVar*m_xy_t90)/4 + (omegaVar*m_yy_t45)/12 - (omegaVar*m_zz_t45)/24 - 1.0/18.0)*rho + RHO_0/18;
    L_BC_Fz = 0;
    #endif //_BC_FORCES 

    break;
case 0b0000010101100011:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[12] + pop[13] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (6*(15*rho_I + m_xx_I*rho_I - 6*m_xy_I*rho_I + 4*m_yy_I*rho_I + 6*m_yz_I*rho_I + m_zz_I*rho_I - omegaVar*m_xx_I*rho_I + 6*omegaVar*m_xy_I*rho_I - 4*omegaVar*m_yy_I*rho_I - 6*omegaVar*m_yz_I*rho_I - omegaVar*m_zz_I*rho_I))/(3*omegaVar + 82);

    m_xx_t45 = -(504*m_xy_I - 2052*m_xx_I - 90*m_yy_I + 234*m_yz_I + 162*m_zz_I - 73*omegaVar*m_xx_I + 6*omegaVar*m_xy_I + 5*omegaVar*m_yy_I + 21*omegaVar*m_yz_I + 8*omegaVar*m_zz_I - 30)/(135*(m_xx_I - 6*m_xy_I + 4*m_yy_I + 6*m_yz_I + m_zz_I - omegaVar*m_xx_I + 6*omegaVar*m_xy_I - 4*omegaVar*m_yy_I - 6*omegaVar*m_yz_I - omegaVar*m_zz_I + 15));
    m_xy_t90 = (1746*m_xy_I - 168*m_xx_I - 180*m_yy_I - 24*m_yz_I + 78*m_zz_I - 2*omegaVar*m_xx_I + 39*omegaVar*m_xy_I + 10*omegaVar*m_yy_I + 24*omegaVar*m_yz_I + 7*omegaVar*m_zz_I - 60)/(90*(m_xx_I - 6*m_xy_I + 4*m_yy_I + 6*m_yz_I + m_zz_I - omegaVar*m_xx_I + 6*omegaVar*m_xy_I - 4*omegaVar*m_yy_I - 6*omegaVar*m_yz_I - omegaVar*m_zz_I + 15));
    m_xz_t90 = (m_xz_I*(3*omegaVar + 82))/(6*(m_xx_I - 6*m_xy_I + 4*m_yy_I + 6*m_yz_I + m_zz_I - omegaVar*m_xx_I + 6*omegaVar*m_xy_I - 4*omegaVar*m_yy_I - 6*omegaVar*m_yz_I - omegaVar*m_zz_I + 15));
    m_yy_t45 = (36*m_xx_I - 216*m_xy_I + 882*m_yy_I + 216*m_yz_I + 36*m_zz_I - 2*omegaVar*m_xx_I + 12*omegaVar*m_xy_I + 19*omegaVar*m_yy_I - 12*omegaVar*m_yz_I - 2*omegaVar*m_zz_I + 48)/(54*(m_xx_I - 6*m_xy_I + 4*m_yy_I + 6*m_yz_I + m_zz_I - omegaVar*m_xx_I + 6*omegaVar*m_xy_I - 4*omegaVar*m_yy_I - 6*omegaVar*m_yz_I - omegaVar*m_zz_I + 15));
    m_yz_t90 = (180*m_yy_I - 24*m_xy_I - 78*m_xx_I + 1746*m_yz_I + 168*m_zz_I - 7*omegaVar*m_xx_I + 24*omegaVar*m_xy_I - 10*omegaVar*m_yy_I + 39*omegaVar*m_yz_I + 2*omegaVar*m_zz_I + 60)/(90*(m_xx_I - 6*m_xy_I + 4*m_yy_I + 6*m_yz_I + m_zz_I - omegaVar*m_xx_I + 6*omegaVar*m_xy_I - 4*omegaVar*m_yy_I - 6*omegaVar*m_yz_I - omegaVar*m_zz_I + 15));
    m_zz_t45 = (234*m_xy_I - 162*m_xx_I + 90*m_yy_I + 504*m_yz_I + 2052*m_zz_I - 8*omegaVar*m_xx_I + 21*omegaVar*m_xy_I - 5*omegaVar*m_yy_I + 6*omegaVar*m_yz_I + 73*omegaVar*m_zz_I + 30)/(135*(m_xx_I - 6*m_xy_I + 4*m_yy_I + 6*m_yz_I + m_zz_I - omegaVar*m_xx_I + 6*omegaVar*m_xy_I - 4*omegaVar*m_yy_I - 6*omegaVar*m_yz_I - omegaVar*m_zz_I + 15));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = (m_xx_t45/6 - m_xy_t90/2 + m_yy_t45/6 - m_zz_t45/12 - (omegaVar*m_xx_t45)/12 + (omegaVar*m_xy_t90)/4 - (omegaVar*m_yy_t45)/12 + (omegaVar*m_zz_t45)/24 + 1.0/18.0)*rho - RHO_0/18;
    L_BC_Fy = (m_xy_t90/2 - m_xx_t45/12 - m_yy_t45/3 - m_yz_t90/2 - m_zz_t45/12 + (omegaVar*m_xx_t45)/24 - (omegaVar*m_xy_t90)/4 + (omegaVar*m_yy_t45)/6 + (omegaVar*m_yz_t90)/4 + (omegaVar*m_zz_t45)/24 - 1.0/9.0)*rho + RHO_0/9;
    L_BC_Fz = (m_xx_t45/12 - m_yy_t45/6 - m_yz_t90/2 - m_zz_t45/6 - (omegaVar*m_xx_t45)/24 + (omegaVar*m_yy_t45)/12 + (omegaVar*m_yz_t90)/4 + (omegaVar*m_zz_t45)/12 - 1.0/18.0)*rho + RHO_0/18;
    #endif //_BC_FORCES 

    break;
case 0b0000010101100100:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (12*(15*rho_I + 2*m_xx_I*rho_I - 6*m_xy_I*rho_I + 2*m_yy_I*rho_I - m_zz_I*rho_I - 2*omegaVar*m_xx_I*rho_I + 6*omegaVar*m_xy_I*rho_I - 2*omegaVar*m_yy_I*rho_I + omegaVar*m_zz_I*rho_I))/(3*omegaVar + 172);

    m_xx_t45 = (1692*m_xx_I - 432*m_xy_I + 144*m_yy_I - 72*m_zz_I + 23*omegaVar*m_xx_I + 12*omegaVar*m_xy_I - 4*omegaVar*m_yy_I + 2*omegaVar*m_zz_I + 48)/(108*(2*m_xx_I - 6*m_xy_I + 2*m_yy_I - m_zz_I - 2*omegaVar*m_xx_I + 6*omegaVar*m_xy_I - 2*omegaVar*m_yy_I + omegaVar*m_zz_I + 15));
    m_xy_t90 = (732*m_xy_I - 72*m_xx_I - 72*m_yy_I + 36*m_zz_I + 2*omegaVar*m_xx_I + 3*omegaVar*m_xy_I + 2*omegaVar*m_yy_I - omegaVar*m_zz_I - 24)/(36*(2*m_xx_I - 6*m_xy_I + 2*m_yy_I - m_zz_I - 2*omegaVar*m_xx_I + 6*omegaVar*m_xy_I - 2*omegaVar*m_yy_I + omegaVar*m_zz_I + 15));
    m_xz_t90 = (m_xz_I*(3*omegaVar + 172))/(12*(2*m_xx_I - 6*m_xy_I + 2*m_yy_I - m_zz_I - 2*omegaVar*m_xx_I + 6*omegaVar*m_xy_I - 2*omegaVar*m_yy_I + omegaVar*m_zz_I + 15));
    m_yy_t45 = (144*m_xx_I - 432*m_xy_I + 1692*m_yy_I - 72*m_zz_I - 4*omegaVar*m_xx_I + 12*omegaVar*m_xy_I + 23*omegaVar*m_yy_I + 2*omegaVar*m_zz_I + 48)/(108*(2*m_xx_I - 6*m_xy_I + 2*m_yy_I - m_zz_I - 2*omegaVar*m_xx_I + 6*omegaVar*m_xy_I - 2*omegaVar*m_yy_I + omegaVar*m_zz_I + 15));
    m_yz_t90 = (m_yz_I*(3*omegaVar + 172))/(12*(2*m_xx_I - 6*m_xy_I + 2*m_yy_I - m_zz_I - 2*omegaVar*m_xx_I + 6*omegaVar*m_xy_I - 2*omegaVar*m_yy_I + omegaVar*m_zz_I + 15));
    m_zz_t45 = (108*m_xy_I - 36*m_xx_I - 36*m_yy_I + 792*m_zz_I + omegaVar*m_xx_I - 3*omegaVar*m_xy_I + omegaVar*m_yy_I + 13*omegaVar*m_zz_I - 12)/(54*(2*m_xx_I - 6*m_xy_I + 2*m_yy_I - m_zz_I - 2*omegaVar*m_xx_I + 6*omegaVar*m_xy_I - 2*omegaVar*m_yy_I + omegaVar*m_zz_I + 15));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = (m_xy_t90/2 - m_xx_t45/6 - m_yy_t45/6 + m_zz_t45/12 + (omegaVar*m_xx_t45)/12 - (omegaVar*m_xy_t90)/4 + (omegaVar*m_yy_t45)/12 - (omegaVar*m_zz_t45)/24 - 1.0/18.0)*rho + RHO_0/18;
    L_BC_Fy = (m_xx_t45/6 - m_xy_t90/2 + m_yy_t45/6 - m_zz_t45/12 - (omegaVar*m_xx_t45)/12 + (omegaVar*m_xy_t90)/4 - (omegaVar*m_yy_t45)/12 + (omegaVar*m_zz_t45)/24 + 1.0/18.0)*rho - RHO_0/18;
    L_BC_Fz = 0;
    #endif //_BC_FORCES 

    break;
case 0b0000010101100101:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[10] + pop[11] + pop[12] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[14]);
    m_xz_I = inv_rho_I * (pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (6*(15*rho_I + 4*m_xx_I*rho_I - 6*m_xy_I*rho_I + 6*m_xz_I*rho_I + m_yy_I*rho_I + m_zz_I*rho_I - 4*omegaVar*m_xx_I*rho_I + 6*omegaVar*m_xy_I*rho_I - 6*omegaVar*m_xz_I*rho_I - omegaVar*m_yy_I*rho_I - omegaVar*m_zz_I*rho_I))/(3*omegaVar + 82);

    m_xx_t45 = (882*m_xx_I - 216*m_xy_I + 216*m_xz_I + 36*m_yy_I + 36*m_zz_I + 19*omegaVar*m_xx_I + 12*omegaVar*m_xy_I - 12*omegaVar*m_xz_I - 2*omegaVar*m_yy_I - 2*omegaVar*m_zz_I + 48)/(54*(4*m_xx_I - 6*m_xy_I + 6*m_xz_I + m_yy_I + m_zz_I - 4*omegaVar*m_xx_I + 6*omegaVar*m_xy_I - 6*omegaVar*m_xz_I - omegaVar*m_yy_I - omegaVar*m_zz_I + 15));
    m_xy_t90 = (1746*m_xy_I - 180*m_xx_I - 24*m_xz_I - 168*m_yy_I + 78*m_zz_I + 10*omegaVar*m_xx_I + 39*omegaVar*m_xy_I + 24*omegaVar*m_xz_I - 2*omegaVar*m_yy_I + 7*omegaVar*m_zz_I - 60)/(90*(4*m_xx_I - 6*m_xy_I + 6*m_xz_I + m_yy_I + m_zz_I - 4*omegaVar*m_xx_I + 6*omegaVar*m_xy_I - 6*omegaVar*m_xz_I - omegaVar*m_yy_I - omegaVar*m_zz_I + 15));
    m_xz_t90 = (180*m_xx_I - 24*m_xy_I + 1746*m_xz_I - 78*m_yy_I + 168*m_zz_I - 10*omegaVar*m_xx_I + 24*omegaVar*m_xy_I + 39*omegaVar*m_xz_I - 7*omegaVar*m_yy_I + 2*omegaVar*m_zz_I + 60)/(90*(4*m_xx_I - 6*m_xy_I + 6*m_xz_I + m_yy_I + m_zz_I - 4*omegaVar*m_xx_I + 6*omegaVar*m_xy_I - 6*omegaVar*m_xz_I - omegaVar*m_yy_I - omegaVar*m_zz_I + 15));
    m_yy_t45 = -(504*m_xy_I - 90*m_xx_I + 234*m_xz_I - 2052*m_yy_I + 162*m_zz_I + 5*omegaVar*m_xx_I + 6*omegaVar*m_xy_I + 21*omegaVar*m_xz_I - 73*omegaVar*m_yy_I + 8*omegaVar*m_zz_I - 30)/(135*(4*m_xx_I - 6*m_xy_I + 6*m_xz_I + m_yy_I + m_zz_I - 4*omegaVar*m_xx_I + 6*omegaVar*m_xy_I - 6*omegaVar*m_xz_I - omegaVar*m_yy_I - omegaVar*m_zz_I + 15));
    m_yz_t90 = (m_yz_I*(3*omegaVar + 82))/(6*(4*m_xx_I - 6*m_xy_I + 6*m_xz_I + m_yy_I + m_zz_I - 4*omegaVar*m_xx_I + 6*omegaVar*m_xy_I - 6*omegaVar*m_xz_I - omegaVar*m_yy_I - omegaVar*m_zz_I + 15));
    m_zz_t45 = (90*m_xx_I + 234*m_xy_I + 504*m_xz_I - 162*m_yy_I + 2052*m_zz_I - 5*omegaVar*m_xx_I + 21*omegaVar*m_xy_I + 6*omegaVar*m_xz_I - 8*omegaVar*m_yy_I + 73*omegaVar*m_zz_I + 30)/(135*(4*m_xx_I - 6*m_xy_I + 6*m_xz_I + m_yy_I + m_zz_I - 4*omegaVar*m_xx_I + 6*omegaVar*m_xy_I - 6*omegaVar*m_xz_I - omegaVar*m_yy_I - omegaVar*m_zz_I + 15));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = (m_xy_t90/2 - m_xx_t45/3 - m_xz_t90/2 - m_yy_t45/12 - m_zz_t45/12 + (omegaVar*m_xx_t45)/6 - (omegaVar*m_xy_t90)/4 + (omegaVar*m_xz_t90)/4 + (omegaVar*m_yy_t45)/24 + (omegaVar*m_zz_t45)/24 - 1.0/9.0)*rho + RHO_0/9;
    L_BC_Fy = (m_xx_t45/6 - m_xy_t90/2 + m_yy_t45/6 - m_zz_t45/12 - (omegaVar*m_xx_t45)/12 + (omegaVar*m_xy_t90)/4 - (omegaVar*m_yy_t45)/12 + (omegaVar*m_zz_t45)/24 + 1.0/18.0)*rho - RHO_0/18;
    L_BC_Fz = (m_yy_t45/12 - m_xz_t90/2 - m_xx_t45/6 - m_zz_t45/6 + (omegaVar*m_xx_t45)/12 + (omegaVar*m_xz_t90)/4 - (omegaVar*m_yy_t45)/24 + (omegaVar*m_zz_t45)/12 - 1.0/18.0)*rho + RHO_0/18;
    #endif //_BC_FORCES 

    break;
case 0b0000010101100110:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (6*(3*rho_I + 2*m_xx_I*rho_I - 6*m_xy_I*rho_I + 2*m_yy_I*rho_I - m_zz_I*rho_I - 2*omegaVar*m_xx_I*rho_I + 6*omegaVar*m_xy_I*rho_I - 2*omegaVar*m_yy_I*rho_I + omegaVar*m_zz_I*rho_I))/(3*omegaVar + 14);

    m_xx_t45 = (198*m_xx_I - 216*m_xy_I + 72*m_yy_I - 36*m_zz_I + 23*omegaVar*m_xx_I + 12*omegaVar*m_xy_I - 4*omegaVar*m_yy_I + 2*omegaVar*m_zz_I + 24)/(54*(2*m_xx_I - 6*m_xy_I + 2*m_yy_I - m_zz_I - 2*omegaVar*m_xx_I + 6*omegaVar*m_xy_I - 2*omegaVar*m_yy_I + omegaVar*m_zz_I + 3));
    m_xy_t90 = (150*m_xy_I - 36*m_xx_I - 36*m_yy_I + 18*m_zz_I + 2*omegaVar*m_xx_I + 3*omegaVar*m_xy_I + 2*omegaVar*m_yy_I - omegaVar*m_zz_I - 12)/(18*(2*m_xx_I - 6*m_xy_I + 2*m_yy_I - m_zz_I - 2*omegaVar*m_xx_I + 6*omegaVar*m_xy_I - 2*omegaVar*m_yy_I + omegaVar*m_zz_I + 3));
    m_xz_t90 = (m_xz_I*(3*omegaVar + 14))/(6*(2*m_xx_I - 6*m_xy_I + 2*m_yy_I - m_zz_I - 2*omegaVar*m_xx_I + 6*omegaVar*m_xy_I - 2*omegaVar*m_yy_I + omegaVar*m_zz_I + 3));
    m_yy_t45 = (72*m_xx_I - 216*m_xy_I + 198*m_yy_I - 36*m_zz_I - 4*omegaVar*m_xx_I + 12*omegaVar*m_xy_I + 23*omegaVar*m_yy_I + 2*omegaVar*m_zz_I + 24)/(54*(2*m_xx_I - 6*m_xy_I + 2*m_yy_I - m_zz_I - 2*omegaVar*m_xx_I + 6*omegaVar*m_xy_I - 2*omegaVar*m_yy_I + omegaVar*m_zz_I + 3));
    m_yz_t90 = (m_yz_I*(3*omegaVar + 14))/(6*(2*m_xx_I - 6*m_xy_I + 2*m_yy_I - m_zz_I - 2*omegaVar*m_xx_I + 6*omegaVar*m_xy_I - 2*omegaVar*m_yy_I + omegaVar*m_zz_I + 3));
    m_zz_t45 = (54*m_xy_I - 18*m_xx_I - 18*m_yy_I + 72*m_zz_I + omegaVar*m_xx_I - 3*omegaVar*m_xy_I + omegaVar*m_yy_I + 13*omegaVar*m_zz_I - 6)/(27*(2*m_xx_I - 6*m_xy_I + 2*m_yy_I - m_zz_I - 2*omegaVar*m_xx_I + 6*omegaVar*m_xy_I - 2*omegaVar*m_yy_I + omegaVar*m_zz_I + 3));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = 0;
    L_BC_Fy = 0;
    L_BC_Fz = 0;
    #endif //_BC_FORCES 

    break;
case 0b0000010101100111:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[10] + pop[12] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8]);
    m_xz_I = inv_rho_I * (pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (3*(15*rho_I + 11*m_xx_I*rho_I - 30*m_xy_I*rho_I + 6*m_xz_I*rho_I + 11*m_yy_I*rho_I + 6*m_yz_I*rho_I - m_zz_I*rho_I - 11*omegaVar*m_xx_I*rho_I + 30*omegaVar*m_xy_I*rho_I - 6*omegaVar*m_xz_I*rho_I - 11*omegaVar*m_yy_I*rho_I - 6*omegaVar*m_yz_I*rho_I + omegaVar*m_zz_I*rho_I))/(9*omegaVar + 31);

    m_xx_t45 = -(2*(1260*m_xy_I - 1206*m_xx_I - 252*m_xz_I - 369*m_yy_I + 27*m_yz_I + 135*m_zz_I - 194*omegaVar*m_xx_I - 60*omegaVar*m_xy_I + 12*omegaVar*m_xz_I + 49*omegaVar*m_yy_I + 93*omegaVar*m_yz_I + 25*omegaVar*m_zz_I - 165))/(135*(11*m_xx_I - 30*m_xy_I + 6*m_xz_I + 11*m_yy_I + 6*m_yz_I - m_zz_I - 11*omegaVar*m_xx_I + 30*omegaVar*m_xy_I - 6*omegaVar*m_xz_I - 11*omegaVar*m_yy_I - 6*omegaVar*m_yz_I + omegaVar*m_zz_I + 15));
    m_xy_t90 = (339*m_xy_I - 84*m_xx_I - 12*m_xz_I - 84*m_yy_I - 12*m_yz_I + 33*m_zz_I + 4*omegaVar*m_xx_I + 21*omegaVar*m_xy_I + 12*omegaVar*m_xz_I + 4*omegaVar*m_yy_I + 12*omegaVar*m_yz_I + 7*omegaVar*m_zz_I - 30)/(9*(11*m_xx_I - 30*m_xy_I + 6*m_xz_I + 11*m_yy_I + 6*m_yz_I - m_zz_I - 11*omegaVar*m_xx_I + 30*omegaVar*m_xy_I - 6*omegaVar*m_xz_I - 11*omegaVar*m_yy_I - 6*omegaVar*m_yz_I + omegaVar*m_zz_I + 15));
    m_xz_t90 = (84*m_xx_I - 60*m_xy_I + 663*m_xz_I - 9*m_yy_I + 12*m_yz_I + 60*m_zz_I - 4*omegaVar*m_xx_I + 60*omegaVar*m_xy_I + 177*omegaVar*m_xz_I - 31*omegaVar*m_yy_I - 12*omegaVar*m_yz_I + 20*omegaVar*m_zz_I + 30)/(45*(11*m_xx_I - 30*m_xy_I + 6*m_xz_I + 11*m_yy_I + 6*m_yz_I - m_zz_I - 11*omegaVar*m_xx_I + 30*omegaVar*m_xy_I - 6*omegaVar*m_xz_I - 11*omegaVar*m_yy_I - 6*omegaVar*m_yz_I + omegaVar*m_zz_I + 15));
    m_yy_t45 = -(2*(1260*m_xy_I - 369*m_xx_I + 27*m_xz_I - 1206*m_yy_I - 252*m_yz_I + 135*m_zz_I + 49*omegaVar*m_xx_I - 60*omegaVar*m_xy_I + 93*omegaVar*m_xz_I - 194*omegaVar*m_yy_I + 12*omegaVar*m_yz_I + 25*omegaVar*m_zz_I - 165))/(135*(11*m_xx_I - 30*m_xy_I + 6*m_xz_I + 11*m_yy_I + 6*m_yz_I - m_zz_I - 11*omegaVar*m_xx_I + 30*omegaVar*m_xy_I - 6*omegaVar*m_xz_I - 11*omegaVar*m_yy_I - 6*omegaVar*m_yz_I + omegaVar*m_zz_I + 15));
    m_yz_t90 = (12*m_xz_I - 60*m_xy_I - 9*m_xx_I + 84*m_yy_I + 663*m_yz_I + 60*m_zz_I - 31*omegaVar*m_xx_I + 60*omegaVar*m_xy_I - 12*omegaVar*m_xz_I - 4*omegaVar*m_yy_I + 177*omegaVar*m_yz_I + 20*omegaVar*m_zz_I + 30)/(45*(11*m_xx_I - 30*m_xy_I + 6*m_xz_I + 11*m_yy_I + 6*m_yz_I - m_zz_I - 11*omegaVar*m_xx_I + 30*omegaVar*m_xy_I - 6*omegaVar*m_xz_I - 11*omegaVar*m_yy_I - 6*omegaVar*m_yz_I + omegaVar*m_zz_I + 15));
    m_zz_t45 = (2*(99*m_xy_I - 27*m_xx_I + 36*m_xz_I - 27*m_yy_I + 36*m_yz_I + 180*m_zz_I - 5*omegaVar*m_xx_I + 21*omegaVar*m_xy_I + 12*omegaVar*m_xz_I - 5*omegaVar*m_yy_I + 12*omegaVar*m_yz_I + 52*omegaVar*m_zz_I - 3))/(27*(11*m_xx_I - 30*m_xy_I + 6*m_xz_I + 11*m_yy_I + 6*m_yz_I - m_zz_I - 11*omegaVar*m_xx_I + 30*omegaVar*m_xy_I - 6*omegaVar*m_xz_I - 11*omegaVar*m_yy_I - 6*omegaVar*m_yz_I + omegaVar*m_zz_I + 15));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = (m_yy_t45/12 - m_xz_t90/2 - m_xx_t45/6 - m_zz_t45/6 + (omegaVar*m_xx_t45)/12 + (omegaVar*m_xz_t90)/4 - (omegaVar*m_yy_t45)/24 + (omegaVar*m_zz_t45)/12 - 1.0/18.0)*rho + RHO_0/18;
    L_BC_Fy = (m_xx_t45/12 - m_yy_t45/6 - m_yz_t90/2 - m_zz_t45/6 - (omegaVar*m_xx_t45)/24 + (omegaVar*m_yy_t45)/12 + (omegaVar*m_yz_t90)/4 + (omegaVar*m_zz_t45)/12 - 1.0/18.0)*rho + RHO_0/18;
    L_BC_Fz = ((omegaVar*m_xx_t45)/24 - m_xz_t90/2 - m_yy_t45/12 - m_yz_t90/2 - m_zz_t45/3 - m_xx_t45/12 + (omegaVar*m_xz_t90)/4 + (omegaVar*m_yy_t45)/24 + (omegaVar*m_yz_t90)/4 + (omegaVar*m_zz_t45)/6 - 1.0/9.0)*rho + RHO_0/9;
    #endif //_BC_FORCES 

    break;
case 0b0000010101101000:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = rho_I;

    m_xx_t45 = m_xx_I;
    m_xy_t90 = m_xy_I;
    m_xz_t90 = m_xz_I;
    m_yy_t45 = m_yy_I;
    m_yz_t90 = m_yz_I;
    m_zz_t45 = m_zz_I;

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = 0;
    L_BC_Fy = 0;
    L_BC_Fz = 0;
    #endif //_BC_FORCES 

    break;
case 0b0000010101101001:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = rho_I;

    m_xx_t45 = m_xx_I;
    m_xy_t90 = m_xy_I;
    m_xz_t90 = m_xz_I;
    m_yy_t45 = m_yy_I;
    m_yz_t90 = m_yz_I;
    m_zz_t45 = m_zz_I;

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = 0;
    L_BC_Fy = 0;
    L_BC_Fz = 0;
    #endif //_BC_FORCES 

    break;
case 0b0000010101101010:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[15] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[15] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] - (1.0/3.0)*pop[15] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (6*(15*rho_I + 4*m_xx_I*rho_I - 6*m_xy_I*rho_I - 6*m_xz_I*rho_I + m_yy_I*rho_I + m_zz_I*rho_I - 4*omegaVar*m_xx_I*rho_I + 6*omegaVar*m_xy_I*rho_I + 6*omegaVar*m_xz_I*rho_I - omegaVar*m_yy_I*rho_I - omegaVar*m_zz_I*rho_I))/(3*omegaVar + 82);

    m_xx_t45 = (882*m_xx_I - 216*m_xy_I - 216*m_xz_I + 36*m_yy_I + 36*m_zz_I + 19*omegaVar*m_xx_I + 12*omegaVar*m_xy_I + 12*omegaVar*m_xz_I - 2*omegaVar*m_yy_I - 2*omegaVar*m_zz_I + 48)/(54*(4*m_xx_I - 6*m_xy_I - 6*m_xz_I + m_yy_I + m_zz_I - 4*omegaVar*m_xx_I + 6*omegaVar*m_xy_I + 6*omegaVar*m_xz_I - omegaVar*m_yy_I - omegaVar*m_zz_I + 15));
    m_xy_t90 = (1746*m_xy_I - 180*m_xx_I + 24*m_xz_I - 168*m_yy_I + 78*m_zz_I + 10*omegaVar*m_xx_I + 39*omegaVar*m_xy_I - 24*omegaVar*m_xz_I - 2*omegaVar*m_yy_I + 7*omegaVar*m_zz_I - 60)/(90*(4*m_xx_I - 6*m_xy_I - 6*m_xz_I + m_yy_I + m_zz_I - 4*omegaVar*m_xx_I + 6*omegaVar*m_xy_I + 6*omegaVar*m_xz_I - omegaVar*m_yy_I - omegaVar*m_zz_I + 15));
    m_xz_t90 = (24*m_xy_I - 180*m_xx_I + 1746*m_xz_I + 78*m_yy_I - 168*m_zz_I + 10*omegaVar*m_xx_I - 24*omegaVar*m_xy_I + 39*omegaVar*m_xz_I + 7*omegaVar*m_yy_I - 2*omegaVar*m_zz_I - 60)/(90*(4*m_xx_I - 6*m_xy_I - 6*m_xz_I + m_yy_I + m_zz_I - 4*omegaVar*m_xx_I + 6*omegaVar*m_xy_I + 6*omegaVar*m_xz_I - omegaVar*m_yy_I - omegaVar*m_zz_I + 15));
    m_yy_t45 = (90*m_xx_I - 504*m_xy_I + 234*m_xz_I + 2052*m_yy_I - 162*m_zz_I - 5*omegaVar*m_xx_I - 6*omegaVar*m_xy_I + 21*omegaVar*m_xz_I + 73*omegaVar*m_yy_I - 8*omegaVar*m_zz_I + 30)/(135*(4*m_xx_I - 6*m_xy_I - 6*m_xz_I + m_yy_I + m_zz_I - 4*omegaVar*m_xx_I + 6*omegaVar*m_xy_I + 6*omegaVar*m_xz_I - omegaVar*m_yy_I - omegaVar*m_zz_I + 15));
    m_yz_t90 = (m_yz_I*(3*omegaVar + 82))/(6*(4*m_xx_I - 6*m_xy_I - 6*m_xz_I + m_yy_I + m_zz_I - 4*omegaVar*m_xx_I + 6*omegaVar*m_xy_I + 6*omegaVar*m_xz_I - omegaVar*m_yy_I - omegaVar*m_zz_I + 15));
    m_zz_t45 = (90*m_xx_I + 234*m_xy_I - 504*m_xz_I - 162*m_yy_I + 2052*m_zz_I - 5*omegaVar*m_xx_I + 21*omegaVar*m_xy_I - 6*omegaVar*m_xz_I - 8*omegaVar*m_yy_I + 73*omegaVar*m_zz_I + 30)/(135*(4*m_xx_I - 6*m_xy_I - 6*m_xz_I + m_yy_I + m_zz_I - 4*omegaVar*m_xx_I + 6*omegaVar*m_xy_I + 6*omegaVar*m_xz_I - omegaVar*m_yy_I - omegaVar*m_zz_I + 15));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = (m_xx_t45/3 - m_xy_t90/2 - m_xz_t90/2 + m_yy_t45/12 + m_zz_t45/12 - (omegaVar*m_xx_t45)/6 + (omegaVar*m_xy_t90)/4 + (omegaVar*m_xz_t90)/4 - (omegaVar*m_yy_t45)/24 - (omegaVar*m_zz_t45)/24 + 1.0/9.0)*rho - RHO_0/9;
    L_BC_Fy = (m_xy_t90/2 - m_xx_t45/6 - m_yy_t45/6 + m_zz_t45/12 + (omegaVar*m_xx_t45)/12 - (omegaVar*m_xy_t90)/4 + (omegaVar*m_yy_t45)/12 - (omegaVar*m_zz_t45)/24 - 1.0/18.0)*rho + RHO_0/18;
    L_BC_Fz = (m_xz_t90/2 - m_xx_t45/6 + m_yy_t45/12 - m_zz_t45/6 + (omegaVar*m_xx_t45)/12 - (omegaVar*m_xz_t90)/4 - (omegaVar*m_yy_t45)/24 + (omegaVar*m_zz_t45)/12 - 1.0/18.0)*rho + RHO_0/18;
    #endif //_BC_FORCES 

    break;
case 0b0000010101101011:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[12] + pop[13] + pop[15] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[15] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] - (1.0/3.0)*pop[15] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (12*(5*rho_I + m_xx_I*rho_I - 2*m_xy_I*rho_I - 2*m_xz_I*rho_I + m_yy_I*rho_I + 2*m_yz_I*rho_I + m_zz_I*rho_I - omegaVar*m_xx_I*rho_I + 2*omegaVar*m_xy_I*rho_I + 2*omegaVar*m_xz_I*rho_I - omegaVar*m_yy_I*rho_I - 2*omegaVar*m_yz_I*rho_I - omegaVar*m_zz_I*rho_I))/(3*omegaVar + 52);

    m_xx_t45 = (472*m_xx_I - 112*m_xy_I - 112*m_xz_I + 4*m_yy_I - 44*m_yz_I + 4*m_zz_I + 23*omegaVar*m_xx_I + 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - 4*omegaVar*m_yy_I - 11*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 20)/(90*(m_xx_I - 2*m_xy_I - 2*m_xz_I + m_yy_I + 2*m_yz_I + m_zz_I - omegaVar*m_xx_I + 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - omegaVar*m_yy_I - 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_xy_t90 = (1108*m_xy_I - 112*m_xx_I + 16*m_xz_I - 112*m_yy_I - 16*m_yz_I + 44*m_zz_I + 2*omegaVar*m_xx_I + 47*omegaVar*m_xy_I - 16*omegaVar*m_xz_I + 2*omegaVar*m_yy_I + 16*omegaVar*m_yz_I + 11*omegaVar*m_zz_I - 40)/(180*(m_xx_I - 2*m_xy_I - 2*m_xz_I + m_yy_I + 2*m_yz_I + m_zz_I - omegaVar*m_xx_I + 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - omegaVar*m_yy_I - 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_xz_t90 = (16*m_xy_I - 112*m_xx_I + 1108*m_xz_I + 44*m_yy_I - 16*m_yz_I - 112*m_zz_I + 2*omegaVar*m_xx_I - 16*omegaVar*m_xy_I + 47*omegaVar*m_xz_I + 11*omegaVar*m_yy_I + 16*omegaVar*m_yz_I + 2*omegaVar*m_zz_I - 40)/(180*(m_xx_I - 2*m_xy_I - 2*m_xz_I + m_yy_I + 2*m_yz_I + m_zz_I - omegaVar*m_xx_I + 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - omegaVar*m_yy_I - 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_yy_t45 = (4*m_xx_I - 112*m_xy_I + 44*m_xz_I + 472*m_yy_I + 112*m_yz_I + 4*m_zz_I - 4*omegaVar*m_xx_I + 2*omegaVar*m_xy_I + 11*omegaVar*m_xz_I + 23*omegaVar*m_yy_I - 2*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 20)/(90*(m_xx_I - 2*m_xy_I - 2*m_xz_I + m_yy_I + 2*m_yz_I + m_zz_I - omegaVar*m_xx_I + 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - omegaVar*m_yy_I - 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_yz_t90 = (112*m_yy_I - 16*m_xy_I - 16*m_xz_I - 44*m_xx_I + 1108*m_yz_I + 112*m_zz_I - 11*omegaVar*m_xx_I + 16*omegaVar*m_xy_I + 16*omegaVar*m_xz_I - 2*omegaVar*m_yy_I + 47*omegaVar*m_yz_I - 2*omegaVar*m_zz_I + 40)/(180*(m_xx_I - 2*m_xy_I - 2*m_xz_I + m_yy_I + 2*m_yz_I + m_zz_I - omegaVar*m_xx_I + 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - omegaVar*m_yy_I - 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_zz_t45 = (4*m_xx_I + 44*m_xy_I - 112*m_xz_I + 4*m_yy_I + 112*m_yz_I + 472*m_zz_I - 4*omegaVar*m_xx_I + 11*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - 4*omegaVar*m_yy_I - 2*omegaVar*m_yz_I + 23*omegaVar*m_zz_I + 20)/(90*(m_xx_I - 2*m_xy_I - 2*m_xz_I + m_yy_I + 2*m_yz_I + m_zz_I - omegaVar*m_xx_I + 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - omegaVar*m_yy_I - 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = (m_xx_t45/3 - m_xy_t90/2 - m_xz_t90/2 + m_yy_t45/12 + m_zz_t45/12 - (omegaVar*m_xx_t45)/6 + (omegaVar*m_xy_t90)/4 + (omegaVar*m_xz_t90)/4 - (omegaVar*m_yy_t45)/24 - (omegaVar*m_zz_t45)/24 + 1.0/9.0)*rho - RHO_0/9;
    L_BC_Fy = (m_xy_t90/2 - m_xx_t45/12 - m_yy_t45/3 - m_yz_t90/2 - m_zz_t45/12 + (omegaVar*m_xx_t45)/24 - (omegaVar*m_xy_t90)/4 + (omegaVar*m_yy_t45)/6 + (omegaVar*m_yz_t90)/4 + (omegaVar*m_zz_t45)/24 - 1.0/9.0)*rho + RHO_0/9;
    L_BC_Fz = (m_xz_t90/2 - m_xx_t45/12 - m_yy_t45/12 - m_yz_t90/2 - m_zz_t45/3 + (omegaVar*m_xx_t45)/24 - (omegaVar*m_xz_t90)/4 + (omegaVar*m_yy_t45)/24 + (omegaVar*m_yz_t90)/4 + (omegaVar*m_zz_t45)/6 - 1.0/9.0)*rho + RHO_0/9;
    #endif //_BC_FORCES 

    break;
case 0b0000010101101100:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[14] + pop[15] + pop[16] + pop[17];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17]);

    rho = (6*(15*rho_I + m_xx_I*rho_I - 6*m_xy_I*rho_I + 4*m_yy_I*rho_I - 6*m_yz_I*rho_I + m_zz_I*rho_I - omegaVar*m_xx_I*rho_I + 6*omegaVar*m_xy_I*rho_I - 4*omegaVar*m_yy_I*rho_I + 6*omegaVar*m_yz_I*rho_I - omegaVar*m_zz_I*rho_I))/(3*omegaVar + 82);

    m_xx_t45 = (2052*m_xx_I - 504*m_xy_I + 90*m_yy_I + 234*m_yz_I - 162*m_zz_I + 73*omegaVar*m_xx_I - 6*omegaVar*m_xy_I - 5*omegaVar*m_yy_I + 21*omegaVar*m_yz_I - 8*omegaVar*m_zz_I + 30)/(135*(m_xx_I - 6*m_xy_I + 4*m_yy_I - 6*m_yz_I + m_zz_I - omegaVar*m_xx_I + 6*omegaVar*m_xy_I - 4*omegaVar*m_yy_I + 6*omegaVar*m_yz_I - omegaVar*m_zz_I + 15));
    m_xy_t90 = (1746*m_xy_I - 168*m_xx_I - 180*m_yy_I + 24*m_yz_I + 78*m_zz_I - 2*omegaVar*m_xx_I + 39*omegaVar*m_xy_I + 10*omegaVar*m_yy_I - 24*omegaVar*m_yz_I + 7*omegaVar*m_zz_I - 60)/(90*(m_xx_I - 6*m_xy_I + 4*m_yy_I - 6*m_yz_I + m_zz_I - omegaVar*m_xx_I + 6*omegaVar*m_xy_I - 4*omegaVar*m_yy_I + 6*omegaVar*m_yz_I - omegaVar*m_zz_I + 15));
    m_xz_t90 = (m_xz_I*(3*omegaVar + 82))/(6*(m_xx_I - 6*m_xy_I + 4*m_yy_I - 6*m_yz_I + m_zz_I - omegaVar*m_xx_I + 6*omegaVar*m_xy_I - 4*omegaVar*m_yy_I + 6*omegaVar*m_yz_I - omegaVar*m_zz_I + 15));
    m_yy_t45 = (36*m_xx_I - 216*m_xy_I + 882*m_yy_I - 216*m_yz_I + 36*m_zz_I - 2*omegaVar*m_xx_I + 12*omegaVar*m_xy_I + 19*omegaVar*m_yy_I + 12*omegaVar*m_yz_I - 2*omegaVar*m_zz_I + 48)/(54*(m_xx_I - 6*m_xy_I + 4*m_yy_I - 6*m_yz_I + m_zz_I - omegaVar*m_xx_I + 6*omegaVar*m_xy_I - 4*omegaVar*m_yy_I + 6*omegaVar*m_yz_I - omegaVar*m_zz_I + 15));
    m_yz_t90 = (78*m_xx_I + 24*m_xy_I - 180*m_yy_I + 1746*m_yz_I - 168*m_zz_I + 7*omegaVar*m_xx_I - 24*omegaVar*m_xy_I + 10*omegaVar*m_yy_I + 39*omegaVar*m_yz_I - 2*omegaVar*m_zz_I - 60)/(90*(m_xx_I - 6*m_xy_I + 4*m_yy_I - 6*m_yz_I + m_zz_I - omegaVar*m_xx_I + 6*omegaVar*m_xy_I - 4*omegaVar*m_yy_I + 6*omegaVar*m_yz_I - omegaVar*m_zz_I + 15));
    m_zz_t45 = (234*m_xy_I - 162*m_xx_I + 90*m_yy_I - 504*m_yz_I + 2052*m_zz_I - 8*omegaVar*m_xx_I + 21*omegaVar*m_xy_I - 5*omegaVar*m_yy_I - 6*omegaVar*m_yz_I + 73*omegaVar*m_zz_I + 30)/(135*(m_xx_I - 6*m_xy_I + 4*m_yy_I - 6*m_yz_I + m_zz_I - omegaVar*m_xx_I + 6*omegaVar*m_xy_I - 4*omegaVar*m_yy_I + 6*omegaVar*m_yz_I - omegaVar*m_zz_I + 15));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = (m_xy_t90/2 - m_xx_t45/6 - m_yy_t45/6 + m_zz_t45/12 + (omegaVar*m_xx_t45)/12 - (omegaVar*m_xy_t90)/4 + (omegaVar*m_yy_t45)/12 - (omegaVar*m_zz_t45)/24 - 1.0/18.0)*rho + RHO_0/18;
    L_BC_Fy = (m_xx_t45/12 - m_xy_t90/2 + m_yy_t45/3 - m_yz_t90/2 + m_zz_t45/12 - (omegaVar*m_xx_t45)/24 + (omegaVar*m_xy_t90)/4 - (omegaVar*m_yy_t45)/6 + (omegaVar*m_yz_t90)/4 - (omegaVar*m_zz_t45)/24 + 1.0/9.0)*rho - RHO_0/9;
    L_BC_Fz = (m_xx_t45/12 - m_yy_t45/6 + m_yz_t90/2 - m_zz_t45/6 - (omegaVar*m_xx_t45)/24 + (omegaVar*m_yy_t45)/12 - (omegaVar*m_yz_t90)/4 + (omegaVar*m_zz_t45)/12 - 1.0/18.0)*rho + RHO_0/18;
    #endif //_BC_FORCES 

    break;
case 0b0000010101101101:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[10] + pop[11] + pop[12] + pop[14] + pop[15] + pop[16] + pop[17];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[14]);
    m_xz_I = inv_rho_I * (pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17]);

    rho = (12*(5*rho_I + m_xx_I*rho_I - 2*m_xy_I*rho_I + 2*m_xz_I*rho_I + m_yy_I*rho_I - 2*m_yz_I*rho_I + m_zz_I*rho_I - omegaVar*m_xx_I*rho_I + 2*omegaVar*m_xy_I*rho_I - 2*omegaVar*m_xz_I*rho_I - omegaVar*m_yy_I*rho_I + 2*omegaVar*m_yz_I*rho_I - omegaVar*m_zz_I*rho_I))/(3*omegaVar + 52);

    m_xx_t45 = (472*m_xx_I - 112*m_xy_I + 112*m_xz_I + 4*m_yy_I + 44*m_yz_I + 4*m_zz_I + 23*omegaVar*m_xx_I + 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - 4*omegaVar*m_yy_I + 11*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 20)/(90*(m_xx_I - 2*m_xy_I + 2*m_xz_I + m_yy_I - 2*m_yz_I + m_zz_I - omegaVar*m_xx_I + 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - omegaVar*m_yy_I + 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_xy_t90 = (1108*m_xy_I - 112*m_xx_I - 16*m_xz_I - 112*m_yy_I + 16*m_yz_I + 44*m_zz_I + 2*omegaVar*m_xx_I + 47*omegaVar*m_xy_I + 16*omegaVar*m_xz_I + 2*omegaVar*m_yy_I - 16*omegaVar*m_yz_I + 11*omegaVar*m_zz_I - 40)/(180*(m_xx_I - 2*m_xy_I + 2*m_xz_I + m_yy_I - 2*m_yz_I + m_zz_I - omegaVar*m_xx_I + 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - omegaVar*m_yy_I + 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_xz_t90 = (112*m_xx_I - 16*m_xy_I + 1108*m_xz_I - 44*m_yy_I - 16*m_yz_I + 112*m_zz_I - 2*omegaVar*m_xx_I + 16*omegaVar*m_xy_I + 47*omegaVar*m_xz_I - 11*omegaVar*m_yy_I + 16*omegaVar*m_yz_I - 2*omegaVar*m_zz_I + 40)/(180*(m_xx_I - 2*m_xy_I + 2*m_xz_I + m_yy_I - 2*m_yz_I + m_zz_I - omegaVar*m_xx_I + 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - omegaVar*m_yy_I + 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_yy_t45 = (4*m_xx_I - 112*m_xy_I - 44*m_xz_I + 472*m_yy_I - 112*m_yz_I + 4*m_zz_I - 4*omegaVar*m_xx_I + 2*omegaVar*m_xy_I - 11*omegaVar*m_xz_I + 23*omegaVar*m_yy_I + 2*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 20)/(90*(m_xx_I - 2*m_xy_I + 2*m_xz_I + m_yy_I - 2*m_yz_I + m_zz_I - omegaVar*m_xx_I + 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - omegaVar*m_yy_I + 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_yz_t90 = (44*m_xx_I + 16*m_xy_I - 16*m_xz_I - 112*m_yy_I + 1108*m_yz_I - 112*m_zz_I + 11*omegaVar*m_xx_I - 16*omegaVar*m_xy_I + 16*omegaVar*m_xz_I + 2*omegaVar*m_yy_I + 47*omegaVar*m_yz_I + 2*omegaVar*m_zz_I - 40)/(180*(m_xx_I - 2*m_xy_I + 2*m_xz_I + m_yy_I - 2*m_yz_I + m_zz_I - omegaVar*m_xx_I + 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - omegaVar*m_yy_I + 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_zz_t45 = (4*m_xx_I + 44*m_xy_I + 112*m_xz_I + 4*m_yy_I - 112*m_yz_I + 472*m_zz_I - 4*omegaVar*m_xx_I + 11*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - 4*omegaVar*m_yy_I + 2*omegaVar*m_yz_I + 23*omegaVar*m_zz_I + 20)/(90*(m_xx_I - 2*m_xy_I + 2*m_xz_I + m_yy_I - 2*m_yz_I + m_zz_I - omegaVar*m_xx_I + 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - omegaVar*m_yy_I + 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = (m_xy_t90/2 - m_xx_t45/3 - m_xz_t90/2 - m_yy_t45/12 - m_zz_t45/12 + (omegaVar*m_xx_t45)/6 - (omegaVar*m_xy_t90)/4 + (omegaVar*m_xz_t90)/4 + (omegaVar*m_yy_t45)/24 + (omegaVar*m_zz_t45)/24 - 1.0/9.0)*rho + RHO_0/9;
    L_BC_Fy = (m_xx_t45/12 - m_xy_t90/2 + m_yy_t45/3 - m_yz_t90/2 + m_zz_t45/12 - (omegaVar*m_xx_t45)/24 + (omegaVar*m_xy_t90)/4 - (omegaVar*m_yy_t45)/6 + (omegaVar*m_yz_t90)/4 - (omegaVar*m_zz_t45)/24 + 1.0/9.0)*rho - RHO_0/9;
    L_BC_Fz = (m_yz_t90/2 - m_xz_t90/2 - m_yy_t45/12 - m_xx_t45/12 - m_zz_t45/3 + (omegaVar*m_xx_t45)/24 + (omegaVar*m_xz_t90)/4 + (omegaVar*m_yy_t45)/24 - (omegaVar*m_yz_t90)/4 + (omegaVar*m_zz_t45)/6 - 1.0/9.0)*rho + RHO_0/9;
    #endif //_BC_FORCES 

    break;
case 0b0000010101101110:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[15] + pop[17];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[15] - (1.0/3.0)*pop[17]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[15] + (2.0/3.0)*pop[17]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[17]);

    rho = (3*(15*rho_I + 11*m_xx_I*rho_I - 30*m_xy_I*rho_I - 6*m_xz_I*rho_I + 11*m_yy_I*rho_I - 6*m_yz_I*rho_I - m_zz_I*rho_I - 11*omegaVar*m_xx_I*rho_I + 30*omegaVar*m_xy_I*rho_I + 6*omegaVar*m_xz_I*rho_I - 11*omegaVar*m_yy_I*rho_I + 6*omegaVar*m_yz_I*rho_I + omegaVar*m_zz_I*rho_I))/(9*omegaVar + 31);

    m_xx_t45 = (2*(1206*m_xx_I - 1260*m_xy_I - 252*m_xz_I + 369*m_yy_I + 27*m_yz_I - 135*m_zz_I + 194*omegaVar*m_xx_I + 60*omegaVar*m_xy_I + 12*omegaVar*m_xz_I - 49*omegaVar*m_yy_I + 93*omegaVar*m_yz_I - 25*omegaVar*m_zz_I + 165))/(135*(11*m_xx_I - 30*m_xy_I - 6*m_xz_I + 11*m_yy_I - 6*m_yz_I - m_zz_I - 11*omegaVar*m_xx_I + 30*omegaVar*m_xy_I + 6*omegaVar*m_xz_I - 11*omegaVar*m_yy_I + 6*omegaVar*m_yz_I + omegaVar*m_zz_I + 15));
    m_xy_t90 = (339*m_xy_I - 84*m_xx_I + 12*m_xz_I - 84*m_yy_I + 12*m_yz_I + 33*m_zz_I + 4*omegaVar*m_xx_I + 21*omegaVar*m_xy_I - 12*omegaVar*m_xz_I + 4*omegaVar*m_yy_I - 12*omegaVar*m_yz_I + 7*omegaVar*m_zz_I - 30)/(9*(11*m_xx_I - 30*m_xy_I - 6*m_xz_I + 11*m_yy_I - 6*m_yz_I - m_zz_I - 11*omegaVar*m_xx_I + 30*omegaVar*m_xy_I + 6*omegaVar*m_xz_I - 11*omegaVar*m_yy_I + 6*omegaVar*m_yz_I + omegaVar*m_zz_I + 15));
    m_xz_t90 = (60*m_xy_I - 84*m_xx_I + 663*m_xz_I + 9*m_yy_I + 12*m_yz_I - 60*m_zz_I + 4*omegaVar*m_xx_I - 60*omegaVar*m_xy_I + 177*omegaVar*m_xz_I + 31*omegaVar*m_yy_I - 12*omegaVar*m_yz_I - 20*omegaVar*m_zz_I - 30)/(45*(11*m_xx_I - 30*m_xy_I - 6*m_xz_I + 11*m_yy_I - 6*m_yz_I - m_zz_I - 11*omegaVar*m_xx_I + 30*omegaVar*m_xy_I + 6*omegaVar*m_xz_I - 11*omegaVar*m_yy_I + 6*omegaVar*m_yz_I + omegaVar*m_zz_I + 15));
    m_yy_t45 = (2*(369*m_xx_I - 1260*m_xy_I + 27*m_xz_I + 1206*m_yy_I - 252*m_yz_I - 135*m_zz_I - 49*omegaVar*m_xx_I + 60*omegaVar*m_xy_I + 93*omegaVar*m_xz_I + 194*omegaVar*m_yy_I + 12*omegaVar*m_yz_I - 25*omegaVar*m_zz_I + 165))/(135*(11*m_xx_I - 30*m_xy_I - 6*m_xz_I + 11*m_yy_I - 6*m_yz_I - m_zz_I - 11*omegaVar*m_xx_I + 30*omegaVar*m_xy_I + 6*omegaVar*m_xz_I - 11*omegaVar*m_yy_I + 6*omegaVar*m_yz_I + omegaVar*m_zz_I + 15));
    m_yz_t90 = (9*m_xx_I + 60*m_xy_I + 12*m_xz_I - 84*m_yy_I + 663*m_yz_I - 60*m_zz_I + 31*omegaVar*m_xx_I - 60*omegaVar*m_xy_I - 12*omegaVar*m_xz_I + 4*omegaVar*m_yy_I + 177*omegaVar*m_yz_I - 20*omegaVar*m_zz_I - 30)/(45*(11*m_xx_I - 30*m_xy_I - 6*m_xz_I + 11*m_yy_I - 6*m_yz_I - m_zz_I - 11*omegaVar*m_xx_I + 30*omegaVar*m_xy_I + 6*omegaVar*m_xz_I - 11*omegaVar*m_yy_I + 6*omegaVar*m_yz_I + omegaVar*m_zz_I + 15));
    m_zz_t45 = -(2*(27*m_xx_I - 99*m_xy_I + 36*m_xz_I + 27*m_yy_I + 36*m_yz_I - 180*m_zz_I + 5*omegaVar*m_xx_I - 21*omegaVar*m_xy_I + 12*omegaVar*m_xz_I + 5*omegaVar*m_yy_I + 12*omegaVar*m_yz_I - 52*omegaVar*m_zz_I + 3))/(27*(11*m_xx_I - 30*m_xy_I - 6*m_xz_I + 11*m_yy_I - 6*m_yz_I - m_zz_I - 11*omegaVar*m_xx_I + 30*omegaVar*m_xy_I + 6*omegaVar*m_xz_I - 11*omegaVar*m_yy_I + 6*omegaVar*m_yz_I + omegaVar*m_zz_I + 15));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = (m_xx_t45/6 - m_xz_t90/2 - m_yy_t45/12 + m_zz_t45/6 - (omegaVar*m_xx_t45)/12 + (omegaVar*m_xz_t90)/4 + (omegaVar*m_yy_t45)/24 - (omegaVar*m_zz_t45)/12 + 1.0/18.0)*rho - RHO_0/18;
    L_BC_Fy = (m_yy_t45/6 - m_xx_t45/12 - m_yz_t90/2 + m_zz_t45/6 + (omegaVar*m_xx_t45)/24 - (omegaVar*m_yy_t45)/12 + (omegaVar*m_yz_t90)/4 - (omegaVar*m_zz_t45)/12 + 1.0/18.0)*rho - RHO_0/18;
    L_BC_Fz = (m_xz_t90/2 - m_xx_t45/12 - m_yy_t45/12 + m_yz_t90/2 - m_zz_t45/3 + (omegaVar*m_xx_t45)/24 - (omegaVar*m_xz_t90)/4 + (omegaVar*m_yy_t45)/24 - (omegaVar*m_yz_t90)/4 + (omegaVar*m_zz_t45)/6 - 1.0/9.0)*rho + RHO_0/9;
    #endif //_BC_FORCES 

    break;
case 0b0000010101101111:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[6] + pop[7] + pop[8] + pop[10] + pop[12] + pop[15] + pop[17];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[15] - (1.0/3.0)*pop[17]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8]);
    m_xz_I = inv_rho_I * (pop[10] - pop[15]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[15] + (2.0/3.0)*pop[17]);
    m_yz_I = inv_rho_I * (pop[12] - pop[17]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[17]);

    rho = (18*(19*rho_I + 12*m_xx_I*rho_I - 33*m_xy_I*rho_I + 12*m_yy_I*rho_I + 6*m_zz_I*rho_I - 12*omegaVar*m_xx_I*rho_I + 33*omegaVar*m_xy_I*rho_I - 12*omegaVar*m_yy_I*rho_I - 6*omegaVar*m_zz_I*rho_I))/(59*omegaVar + 207);

    m_xx_t45 = -(2*(324*m_xy_I - 306*m_xx_I - 99*m_yy_I + 54*m_zz_I - 58*omegaVar*m_xx_I + 12*omegaVar*m_xy_I + omegaVar*m_yy_I + 30*omegaVar*m_zz_I - 36))/(27*(12*m_xx_I - 33*m_xy_I + 12*m_yy_I + 6*m_zz_I - 12*omegaVar*m_xx_I + 33*omegaVar*m_xy_I - 12*omegaVar*m_yy_I - 6*omegaVar*m_zz_I + 19));
    m_xy_t90 = -(216*m_xx_I - 801*m_xy_I + 216*m_yy_I - 99*m_zz_I + 8*omegaVar*m_xx_I - 81*omegaVar*m_xy_I + 8*omegaVar*m_yy_I - 55*omegaVar*m_zz_I + 66)/(18*(12*m_xx_I - 33*m_xy_I + 12*m_yy_I + 6*m_zz_I - 12*omegaVar*m_xx_I + 33*omegaVar*m_xy_I - 12*omegaVar*m_yy_I - 6*omegaVar*m_zz_I + 19));
    m_xz_t90 = (m_xz_I*(59*omegaVar + 207))/(9*(12*m_xx_I - 33*m_xy_I + 12*m_yy_I + 6*m_zz_I - 12*omegaVar*m_xx_I + 33*omegaVar*m_xy_I - 12*omegaVar*m_yy_I - 6*omegaVar*m_zz_I + 19));
    m_yy_t45 = -(2*(324*m_xy_I - 99*m_xx_I - 306*m_yy_I + 54*m_zz_I + omegaVar*m_xx_I + 12*omegaVar*m_xy_I - 58*omegaVar*m_yy_I + 30*omegaVar*m_zz_I - 36))/(27*(12*m_xx_I - 33*m_xy_I + 12*m_yy_I + 6*m_zz_I - 12*omegaVar*m_xx_I + 33*omegaVar*m_xy_I - 12*omegaVar*m_yy_I - 6*omegaVar*m_zz_I + 19));
    m_yz_t90 = (m_yz_I*(59*omegaVar + 207))/(9*(12*m_xx_I - 33*m_xy_I + 12*m_yy_I + 6*m_zz_I - 12*omegaVar*m_xx_I + 33*omegaVar*m_xy_I - 12*omegaVar*m_yy_I - 6*omegaVar*m_zz_I + 19));
    m_zz_t45 = (99*m_xy_I - 36*m_xx_I - 36*m_yy_I + 189*m_zz_I - 20*omegaVar*m_xx_I + 55*omegaVar*m_xy_I - 20*omegaVar*m_yy_I + 49*omegaVar*m_zz_I + 12)/(9*(12*m_xx_I - 33*m_xy_I + 12*m_yy_I + 6*m_zz_I - 12*omegaVar*m_xx_I + 33*omegaVar*m_xy_I - 12*omegaVar*m_yy_I - 6*omegaVar*m_zz_I + 19));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = (m_xz_t90*rho*(omegaVar - 2))/2;
    L_BC_Fy = (m_yz_t90*rho*(omegaVar - 2))/2;
    L_BC_Fz = ((omegaVar*m_zz_t45)/2 - m_zz_t45 - 1.0/3.0)*rho + RHO_0/3;
    #endif //_BC_FORCES 

    break;
case 0b0000010101110000:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[16] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);

    rho = (6*(15*rho_I + m_xx_I*rho_I - 6*m_xz_I*rho_I + m_yy_I*rho_I - 6*m_yz_I*rho_I + 4*m_zz_I*rho_I - omegaVar*m_xx_I*rho_I + 6*omegaVar*m_xz_I*rho_I - omegaVar*m_yy_I*rho_I + 6*omegaVar*m_yz_I*rho_I - 4*omegaVar*m_zz_I*rho_I))/(3*omegaVar + 82);

    m_xx_t45 = (2052*m_xx_I - 504*m_xz_I - 162*m_yy_I + 234*m_yz_I + 90*m_zz_I + 73*omegaVar*m_xx_I - 6*omegaVar*m_xz_I - 8*omegaVar*m_yy_I + 21*omegaVar*m_yz_I - 5*omegaVar*m_zz_I + 30)/(135*(m_xx_I - 6*m_xz_I + m_yy_I - 6*m_yz_I + 4*m_zz_I - omegaVar*m_xx_I + 6*omegaVar*m_xz_I - omegaVar*m_yy_I + 6*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 15));
    m_xy_t90 = (m_xy_I*(3*omegaVar + 82))/(6*(m_xx_I - 6*m_xz_I + m_yy_I - 6*m_yz_I + 4*m_zz_I - omegaVar*m_xx_I + 6*omegaVar*m_xz_I - omegaVar*m_yy_I + 6*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 15));
    m_xz_t90 = (1746*m_xz_I - 168*m_xx_I + 78*m_yy_I + 24*m_yz_I - 180*m_zz_I - 2*omegaVar*m_xx_I + 39*omegaVar*m_xz_I + 7*omegaVar*m_yy_I - 24*omegaVar*m_yz_I + 10*omegaVar*m_zz_I - 60)/(90*(m_xx_I - 6*m_xz_I + m_yy_I - 6*m_yz_I + 4*m_zz_I - omegaVar*m_xx_I + 6*omegaVar*m_xz_I - omegaVar*m_yy_I + 6*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 15));
    m_yy_t45 = (234*m_xz_I - 162*m_xx_I + 2052*m_yy_I - 504*m_yz_I + 90*m_zz_I - 8*omegaVar*m_xx_I + 21*omegaVar*m_xz_I + 73*omegaVar*m_yy_I - 6*omegaVar*m_yz_I - 5*omegaVar*m_zz_I + 30)/(135*(m_xx_I - 6*m_xz_I + m_yy_I - 6*m_yz_I + 4*m_zz_I - omegaVar*m_xx_I + 6*omegaVar*m_xz_I - omegaVar*m_yy_I + 6*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 15));
    m_yz_t90 = (78*m_xx_I + 24*m_xz_I - 168*m_yy_I + 1746*m_yz_I - 180*m_zz_I + 7*omegaVar*m_xx_I - 24*omegaVar*m_xz_I - 2*omegaVar*m_yy_I + 39*omegaVar*m_yz_I + 10*omegaVar*m_zz_I - 60)/(90*(m_xx_I - 6*m_xz_I + m_yy_I - 6*m_yz_I + 4*m_zz_I - omegaVar*m_xx_I + 6*omegaVar*m_xz_I - omegaVar*m_yy_I + 6*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 15));
    m_zz_t45 = (36*m_xx_I - 216*m_xz_I + 36*m_yy_I - 216*m_yz_I + 882*m_zz_I - 2*omegaVar*m_xx_I + 12*omegaVar*m_xz_I - 2*omegaVar*m_yy_I + 12*omegaVar*m_yz_I + 19*omegaVar*m_zz_I + 48)/(54*(m_xx_I - 6*m_xz_I + m_yy_I - 6*m_yz_I + 4*m_zz_I - omegaVar*m_xx_I + 6*omegaVar*m_xz_I - omegaVar*m_yy_I + 6*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 15));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = (m_xz_t90/2 - m_xx_t45/6 + m_yy_t45/12 - m_zz_t45/6 + (omegaVar*m_xx_t45)/12 - (omegaVar*m_xz_t90)/4 - (omegaVar*m_yy_t45)/24 + (omegaVar*m_zz_t45)/12 - 1.0/18.0)*rho + RHO_0/18;
    L_BC_Fy = (m_xx_t45/12 - m_yy_t45/6 + m_yz_t90/2 - m_zz_t45/6 - (omegaVar*m_xx_t45)/24 + (omegaVar*m_yy_t45)/12 - (omegaVar*m_yz_t90)/4 + (omegaVar*m_zz_t45)/12 - 1.0/18.0)*rho + RHO_0/18;
    L_BC_Fz = (m_xx_t45/12 - m_xz_t90/2 + m_yy_t45/12 - m_yz_t90/2 + m_zz_t45/3 - (omegaVar*m_xx_t45)/24 + (omegaVar*m_xz_t90)/4 - (omegaVar*m_yy_t45)/24 + (omegaVar*m_yz_t90)/4 - (omegaVar*m_zz_t45)/6 + 1.0/9.0)*rho - RHO_0/9;
    #endif //_BC_FORCES 

    break;
case 0b0000010101110001:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[16] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);

    rho = (12*(5*rho_I + m_xx_I*rho_I + 2*m_xy_I*rho_I - 2*m_xz_I*rho_I + m_yy_I*rho_I - 2*m_yz_I*rho_I + m_zz_I*rho_I - omegaVar*m_xx_I*rho_I - 2*omegaVar*m_xy_I*rho_I + 2*omegaVar*m_xz_I*rho_I - omegaVar*m_yy_I*rho_I + 2*omegaVar*m_yz_I*rho_I - omegaVar*m_zz_I*rho_I))/(3*omegaVar + 52);

    m_xx_t45 = (472*m_xx_I + 112*m_xy_I - 112*m_xz_I + 4*m_yy_I + 44*m_yz_I + 4*m_zz_I + 23*omegaVar*m_xx_I - 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - 4*omegaVar*m_yy_I + 11*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 20)/(90*(m_xx_I + 2*m_xy_I - 2*m_xz_I + m_yy_I - 2*m_yz_I + m_zz_I - omegaVar*m_xx_I - 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - omegaVar*m_yy_I + 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_xy_t90 = (112*m_xx_I + 1108*m_xy_I - 16*m_xz_I + 112*m_yy_I - 16*m_yz_I - 44*m_zz_I - 2*omegaVar*m_xx_I + 47*omegaVar*m_xy_I + 16*omegaVar*m_xz_I - 2*omegaVar*m_yy_I + 16*omegaVar*m_yz_I - 11*omegaVar*m_zz_I + 40)/(180*(m_xx_I + 2*m_xy_I - 2*m_xz_I + m_yy_I - 2*m_yz_I + m_zz_I - omegaVar*m_xx_I - 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - omegaVar*m_yy_I + 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_xz_t90 = (1108*m_xz_I - 16*m_xy_I - 112*m_xx_I + 44*m_yy_I + 16*m_yz_I - 112*m_zz_I + 2*omegaVar*m_xx_I + 16*omegaVar*m_xy_I + 47*omegaVar*m_xz_I + 11*omegaVar*m_yy_I - 16*omegaVar*m_yz_I + 2*omegaVar*m_zz_I - 40)/(180*(m_xx_I + 2*m_xy_I - 2*m_xz_I + m_yy_I - 2*m_yz_I + m_zz_I - omegaVar*m_xx_I - 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - omegaVar*m_yy_I + 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_yy_t45 = (4*m_xx_I + 112*m_xy_I + 44*m_xz_I + 472*m_yy_I - 112*m_yz_I + 4*m_zz_I - 4*omegaVar*m_xx_I - 2*omegaVar*m_xy_I + 11*omegaVar*m_xz_I + 23*omegaVar*m_yy_I + 2*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 20)/(90*(m_xx_I + 2*m_xy_I - 2*m_xz_I + m_yy_I - 2*m_yz_I + m_zz_I - omegaVar*m_xx_I - 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - omegaVar*m_yy_I + 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_yz_t90 = (44*m_xx_I - 16*m_xy_I + 16*m_xz_I - 112*m_yy_I + 1108*m_yz_I - 112*m_zz_I + 11*omegaVar*m_xx_I + 16*omegaVar*m_xy_I - 16*omegaVar*m_xz_I + 2*omegaVar*m_yy_I + 47*omegaVar*m_yz_I + 2*omegaVar*m_zz_I - 40)/(180*(m_xx_I + 2*m_xy_I - 2*m_xz_I + m_yy_I - 2*m_yz_I + m_zz_I - omegaVar*m_xx_I - 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - omegaVar*m_yy_I + 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_zz_t45 = (4*m_xx_I - 44*m_xy_I - 112*m_xz_I + 4*m_yy_I - 112*m_yz_I + 472*m_zz_I - 4*omegaVar*m_xx_I - 11*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - 4*omegaVar*m_yy_I + 2*omegaVar*m_yz_I + 23*omegaVar*m_zz_I + 20)/(90*(m_xx_I + 2*m_xy_I - 2*m_xz_I + m_yy_I - 2*m_yz_I + m_zz_I - omegaVar*m_xx_I - 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - omegaVar*m_yy_I + 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = (m_xz_t90/2 - m_xy_t90/2 - m_xx_t45/3 - m_yy_t45/12 - m_zz_t45/12 + (omegaVar*m_xx_t45)/6 + (omegaVar*m_xy_t90)/4 - (omegaVar*m_xz_t90)/4 + (omegaVar*m_yy_t45)/24 + (omegaVar*m_zz_t45)/24 - 1.0/9.0)*rho + RHO_0/9;
    L_BC_Fy = (m_yz_t90/2 - m_xy_t90/2 - m_yy_t45/3 - m_xx_t45/12 - m_zz_t45/12 + (omegaVar*m_xx_t45)/24 + (omegaVar*m_xy_t90)/4 + (omegaVar*m_yy_t45)/6 - (omegaVar*m_yz_t90)/4 + (omegaVar*m_zz_t45)/24 - 1.0/9.0)*rho + RHO_0/9;
    L_BC_Fz = (m_xx_t45/12 - m_xz_t90/2 + m_yy_t45/12 - m_yz_t90/2 + m_zz_t45/3 - (omegaVar*m_xx_t45)/24 + (omegaVar*m_xz_t90)/4 - (omegaVar*m_yy_t45)/24 + (omegaVar*m_yz_t90)/4 - (omegaVar*m_zz_t45)/6 + 1.0/9.0)*rho - RHO_0/9;
    #endif //_BC_FORCES 

    break;
case 0b0000010101110010:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[16] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);

    rho = (12*(5*rho_I + m_xx_I*rho_I - 2*m_xy_I*rho_I - 2*m_xz_I*rho_I + m_yy_I*rho_I - 2*m_yz_I*rho_I + m_zz_I*rho_I - omegaVar*m_xx_I*rho_I + 2*omegaVar*m_xy_I*rho_I + 2*omegaVar*m_xz_I*rho_I - omegaVar*m_yy_I*rho_I + 2*omegaVar*m_yz_I*rho_I - omegaVar*m_zz_I*rho_I))/(3*omegaVar + 52);

    m_xx_t45 = (472*m_xx_I - 112*m_xy_I - 112*m_xz_I + 4*m_yy_I + 44*m_yz_I + 4*m_zz_I + 23*omegaVar*m_xx_I + 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - 4*omegaVar*m_yy_I + 11*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 20)/(90*(m_xx_I - 2*m_xy_I - 2*m_xz_I + m_yy_I - 2*m_yz_I + m_zz_I - omegaVar*m_xx_I + 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - omegaVar*m_yy_I + 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_xy_t90 = (1108*m_xy_I - 112*m_xx_I + 16*m_xz_I - 112*m_yy_I + 16*m_yz_I + 44*m_zz_I + 2*omegaVar*m_xx_I + 47*omegaVar*m_xy_I - 16*omegaVar*m_xz_I + 2*omegaVar*m_yy_I - 16*omegaVar*m_yz_I + 11*omegaVar*m_zz_I - 40)/(180*(m_xx_I - 2*m_xy_I - 2*m_xz_I + m_yy_I - 2*m_yz_I + m_zz_I - omegaVar*m_xx_I + 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - omegaVar*m_yy_I + 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_xz_t90 = (16*m_xy_I - 112*m_xx_I + 1108*m_xz_I + 44*m_yy_I + 16*m_yz_I - 112*m_zz_I + 2*omegaVar*m_xx_I - 16*omegaVar*m_xy_I + 47*omegaVar*m_xz_I + 11*omegaVar*m_yy_I - 16*omegaVar*m_yz_I + 2*omegaVar*m_zz_I - 40)/(180*(m_xx_I - 2*m_xy_I - 2*m_xz_I + m_yy_I - 2*m_yz_I + m_zz_I - omegaVar*m_xx_I + 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - omegaVar*m_yy_I + 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_yy_t45 = (4*m_xx_I - 112*m_xy_I + 44*m_xz_I + 472*m_yy_I - 112*m_yz_I + 4*m_zz_I - 4*omegaVar*m_xx_I + 2*omegaVar*m_xy_I + 11*omegaVar*m_xz_I + 23*omegaVar*m_yy_I + 2*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 20)/(90*(m_xx_I - 2*m_xy_I - 2*m_xz_I + m_yy_I - 2*m_yz_I + m_zz_I - omegaVar*m_xx_I + 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - omegaVar*m_yy_I + 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_yz_t90 = (44*m_xx_I + 16*m_xy_I + 16*m_xz_I - 112*m_yy_I + 1108*m_yz_I - 112*m_zz_I + 11*omegaVar*m_xx_I - 16*omegaVar*m_xy_I - 16*omegaVar*m_xz_I + 2*omegaVar*m_yy_I + 47*omegaVar*m_yz_I + 2*omegaVar*m_zz_I - 40)/(180*(m_xx_I - 2*m_xy_I - 2*m_xz_I + m_yy_I - 2*m_yz_I + m_zz_I - omegaVar*m_xx_I + 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - omegaVar*m_yy_I + 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_zz_t45 = (4*m_xx_I + 44*m_xy_I - 112*m_xz_I + 4*m_yy_I - 112*m_yz_I + 472*m_zz_I - 4*omegaVar*m_xx_I + 11*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - 4*omegaVar*m_yy_I + 2*omegaVar*m_yz_I + 23*omegaVar*m_zz_I + 20)/(90*(m_xx_I - 2*m_xy_I - 2*m_xz_I + m_yy_I - 2*m_yz_I + m_zz_I - omegaVar*m_xx_I + 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - omegaVar*m_yy_I + 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = (rho*(omegaVar - 2)*(2*m_xy_t90 - 2*m_xz_t90 - m_yy_t45 + m_zz_t45))/8;
    L_BC_Fy = (m_xy_t90/2 - m_xx_t45/12 - m_yy_t45/3 + m_yz_t90/2 - m_zz_t45/12 + (omegaVar*m_xx_t45)/24 - (omegaVar*m_xy_t90)/4 + (omegaVar*m_yy_t45)/6 - (omegaVar*m_yz_t90)/4 + (omegaVar*m_zz_t45)/24 - 1.0/9.0)*rho + RHO_0/9;
    L_BC_Fz = (m_xx_t45/12 - m_xz_t90/2 + m_yy_t45/12 - m_yz_t90/2 + m_zz_t45/3 - (omegaVar*m_xx_t45)/24 + (omegaVar*m_xz_t90)/4 - (omegaVar*m_yy_t45)/24 + (omegaVar*m_yz_t90)/4 - (omegaVar*m_zz_t45)/6 + 1.0/9.0)*rho - RHO_0/9;
    #endif //_BC_FORCES 

    break;
case 0b0000010101110011:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[4] + pop[5] + pop[6] + pop[8] + pop[9] + pop[10] + pop[12] + pop[13] + pop[16] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[8] - pop[13]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[12] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);

    rho = (36*(107*rho_I + 12*m_xx_I*rho_I - 33*m_xz_I*rho_I + 72*m_yy_I*rho_I + 12*m_zz_I*rho_I - 12*omegaVar*m_xx_I*rho_I + 33*omegaVar*m_xz_I*rho_I - 72*omegaVar*m_yy_I*rho_I - 12*omegaVar*m_zz_I*rho_I))/(313*omegaVar + 2790);

    m_xx_t45 = -(648*m_xz_I - 2772*m_xx_I + 108*m_yy_I + 18*m_zz_I - 302*omegaVar*m_xx_I + 48*omegaVar*m_xz_I + 66*omegaVar*m_yy_I + 11*omegaVar*m_zz_I - 72)/(27*(12*m_xx_I - 33*m_xz_I + 72*m_yy_I + 12*m_zz_I - 12*omegaVar*m_xx_I + 33*omegaVar*m_xz_I - 72*omegaVar*m_yy_I - 12*omegaVar*m_zz_I + 107));
    m_xy_t90 = (m_xy_I*(313*omegaVar + 2790))/(18*(12*m_xx_I - 33*m_xz_I + 72*m_yy_I + 12*m_zz_I - 12*omegaVar*m_xx_I + 33*omegaVar*m_xz_I - 72*omegaVar*m_yy_I - 12*omegaVar*m_zz_I + 107));
    m_xz_t90 = -(432*m_xx_I - 3978*m_xz_I - 198*m_yy_I + 432*m_zz_I + 32*omegaVar*m_xx_I - 401*omegaVar*m_xz_I - 121*omegaVar*m_yy_I + 32*omegaVar*m_zz_I + 132)/(36*(12*m_xx_I - 33*m_xz_I + 72*m_yy_I + 12*m_zz_I - 12*omegaVar*m_xx_I + 33*omegaVar*m_xz_I - 72*omegaVar*m_yy_I - 12*omegaVar*m_zz_I + 107));
    m_yy_t45 = (198*m_xz_I - 72*m_xx_I + 2358*m_yy_I - 72*m_zz_I - 44*omegaVar*m_xx_I + 121*omegaVar*m_xz_I + 49*omegaVar*m_yy_I - 44*omegaVar*m_zz_I + 288)/(18*(12*m_xx_I - 33*m_xz_I + 72*m_yy_I + 12*m_zz_I - 12*omegaVar*m_xx_I + 33*omegaVar*m_xz_I - 72*omegaVar*m_yy_I - 12*omegaVar*m_zz_I + 107));
    m_yz_t90 = (m_yz_I*(313*omegaVar + 2790))/(18*(12*m_xx_I - 33*m_xz_I + 72*m_yy_I + 12*m_zz_I - 12*omegaVar*m_xx_I + 33*omegaVar*m_xz_I - 72*omegaVar*m_yy_I - 12*omegaVar*m_zz_I + 107));
    m_zz_t45 = -(18*m_xx_I + 648*m_xz_I + 108*m_yy_I - 2772*m_zz_I + 11*omegaVar*m_xx_I + 48*omegaVar*m_xz_I + 66*omegaVar*m_yy_I - 302*omegaVar*m_zz_I - 72)/(27*(12*m_xx_I - 33*m_xz_I + 72*m_yy_I + 12*m_zz_I - 12*omegaVar*m_xx_I + 33*omegaVar*m_xz_I - 72*omegaVar*m_yy_I - 12*omegaVar*m_zz_I + 107));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = (m_xz_t90/2 - m_xy_t90 - m_xx_t45/6 + m_yy_t45/12 - m_zz_t45/6 + (omegaVar*m_xx_t45)/12 + (omegaVar*m_xy_t90)/2 - (omegaVar*m_xz_t90)/4 - (omegaVar*m_yy_t45)/24 + (omegaVar*m_zz_t45)/12 - 1.0/18.0)*rho + RHO_0/18;
    L_BC_Fy = ((omegaVar*m_yy_t45)/2 - m_yy_t45 - 1.0/3.0)*rho + RHO_0/3;
    L_BC_Fz = (m_xx_t45/6 - m_xz_t90/2 - m_yy_t45/12 - m_yz_t90 + m_zz_t45/6 - (omegaVar*m_xx_t45)/12 + (omegaVar*m_xz_t90)/4 + (omegaVar*m_yy_t45)/24 + (omegaVar*m_yz_t90)/2 - (omegaVar*m_zz_t45)/12 + 1.0/18.0)*rho - RHO_0/18;
    #endif //_BC_FORCES 

    break;
case 0b0000010101110100:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[14] + pop[16] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);

    rho = (12*(5*rho_I + m_xx_I*rho_I - 2*m_xy_I*rho_I - 2*m_xz_I*rho_I + m_yy_I*rho_I - 2*m_yz_I*rho_I + m_zz_I*rho_I - omegaVar*m_xx_I*rho_I + 2*omegaVar*m_xy_I*rho_I + 2*omegaVar*m_xz_I*rho_I - omegaVar*m_yy_I*rho_I + 2*omegaVar*m_yz_I*rho_I - omegaVar*m_zz_I*rho_I))/(3*omegaVar + 52);

    m_xx_t45 = (472*m_xx_I - 112*m_xy_I - 112*m_xz_I + 4*m_yy_I + 44*m_yz_I + 4*m_zz_I + 23*omegaVar*m_xx_I + 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - 4*omegaVar*m_yy_I + 11*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 20)/(90*(m_xx_I - 2*m_xy_I - 2*m_xz_I + m_yy_I - 2*m_yz_I + m_zz_I - omegaVar*m_xx_I + 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - omegaVar*m_yy_I + 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_xy_t90 = (1108*m_xy_I - 112*m_xx_I + 16*m_xz_I - 112*m_yy_I + 16*m_yz_I + 44*m_zz_I + 2*omegaVar*m_xx_I + 47*omegaVar*m_xy_I - 16*omegaVar*m_xz_I + 2*omegaVar*m_yy_I - 16*omegaVar*m_yz_I + 11*omegaVar*m_zz_I - 40)/(180*(m_xx_I - 2*m_xy_I - 2*m_xz_I + m_yy_I - 2*m_yz_I + m_zz_I - omegaVar*m_xx_I + 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - omegaVar*m_yy_I + 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_xz_t90 = (16*m_xy_I - 112*m_xx_I + 1108*m_xz_I + 44*m_yy_I + 16*m_yz_I - 112*m_zz_I + 2*omegaVar*m_xx_I - 16*omegaVar*m_xy_I + 47*omegaVar*m_xz_I + 11*omegaVar*m_yy_I - 16*omegaVar*m_yz_I + 2*omegaVar*m_zz_I - 40)/(180*(m_xx_I - 2*m_xy_I - 2*m_xz_I + m_yy_I - 2*m_yz_I + m_zz_I - omegaVar*m_xx_I + 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - omegaVar*m_yy_I + 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_yy_t45 = (4*m_xx_I - 112*m_xy_I + 44*m_xz_I + 472*m_yy_I - 112*m_yz_I + 4*m_zz_I - 4*omegaVar*m_xx_I + 2*omegaVar*m_xy_I + 11*omegaVar*m_xz_I + 23*omegaVar*m_yy_I + 2*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 20)/(90*(m_xx_I - 2*m_xy_I - 2*m_xz_I + m_yy_I - 2*m_yz_I + m_zz_I - omegaVar*m_xx_I + 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - omegaVar*m_yy_I + 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_yz_t90 = (44*m_xx_I + 16*m_xy_I + 16*m_xz_I - 112*m_yy_I + 1108*m_yz_I - 112*m_zz_I + 11*omegaVar*m_xx_I - 16*omegaVar*m_xy_I - 16*omegaVar*m_xz_I + 2*omegaVar*m_yy_I + 47*omegaVar*m_yz_I + 2*omegaVar*m_zz_I - 40)/(180*(m_xx_I - 2*m_xy_I - 2*m_xz_I + m_yy_I - 2*m_yz_I + m_zz_I - omegaVar*m_xx_I + 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - omegaVar*m_yy_I + 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_zz_t45 = (4*m_xx_I + 44*m_xy_I - 112*m_xz_I + 4*m_yy_I - 112*m_yz_I + 472*m_zz_I - 4*omegaVar*m_xx_I + 11*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - 4*omegaVar*m_yy_I + 2*omegaVar*m_yz_I + 23*omegaVar*m_zz_I + 20)/(90*(m_xx_I - 2*m_xy_I - 2*m_xz_I + m_yy_I - 2*m_yz_I + m_zz_I - omegaVar*m_xx_I + 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - omegaVar*m_yy_I + 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = (m_xy_t90/2 - m_xx_t45/3 + m_xz_t90/2 - m_yy_t45/12 - m_zz_t45/12 + (omegaVar*m_xx_t45)/6 - (omegaVar*m_xy_t90)/4 - (omegaVar*m_xz_t90)/4 + (omegaVar*m_yy_t45)/24 + (omegaVar*m_zz_t45)/24 - 1.0/9.0)*rho + RHO_0/9;
    L_BC_Fy = -(rho*(omegaVar - 2)*(m_xx_t45 - 2*m_xy_t90 + 2*m_yz_t90 - m_zz_t45))/8;
    L_BC_Fz = (m_xx_t45/12 - m_xz_t90/2 + m_yy_t45/12 - m_yz_t90/2 + m_zz_t45/3 - (omegaVar*m_xx_t45)/24 + (omegaVar*m_xz_t90)/4 - (omegaVar*m_yy_t45)/24 + (omegaVar*m_yz_t90)/4 - (omegaVar*m_zz_t45)/6 + 1.0/9.0)*rho - RHO_0/9;
    #endif //_BC_FORCES 

    break;
case 0b0000010101110101:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[8] + pop[10] + pop[11] + pop[12] + pop[14] + pop[16] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[8] - pop[14]);
    m_xz_I = inv_rho_I * (pop[10] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);

    rho = (36*(107*rho_I + 72*m_xx_I*rho_I + 12*m_yy_I*rho_I - 33*m_yz_I*rho_I + 12*m_zz_I*rho_I - 72*omegaVar*m_xx_I*rho_I - 12*omegaVar*m_yy_I*rho_I + 33*omegaVar*m_yz_I*rho_I - 12*omegaVar*m_zz_I*rho_I))/(313*omegaVar + 2790);

    m_xx_t45 = (2358*m_xx_I - 72*m_yy_I + 198*m_yz_I - 72*m_zz_I + 49*omegaVar*m_xx_I - 44*omegaVar*m_yy_I + 121*omegaVar*m_yz_I - 44*omegaVar*m_zz_I + 288)/(18*(72*m_xx_I + 12*m_yy_I - 33*m_yz_I + 12*m_zz_I - 72*omegaVar*m_xx_I - 12*omegaVar*m_yy_I + 33*omegaVar*m_yz_I - 12*omegaVar*m_zz_I + 107));
    m_xy_t90 = (m_xy_I*(313*omegaVar + 2790))/(18*(72*m_xx_I + 12*m_yy_I - 33*m_yz_I + 12*m_zz_I - 72*omegaVar*m_xx_I - 12*omegaVar*m_yy_I + 33*omegaVar*m_yz_I - 12*omegaVar*m_zz_I + 107));
    m_xz_t90 = (m_xz_I*(313*omegaVar + 2790))/(18*(72*m_xx_I + 12*m_yy_I - 33*m_yz_I + 12*m_zz_I - 72*omegaVar*m_xx_I - 12*omegaVar*m_yy_I + 33*omegaVar*m_yz_I - 12*omegaVar*m_zz_I + 107));
    m_yy_t45 = -(108*m_xx_I - 2772*m_yy_I + 648*m_yz_I + 18*m_zz_I + 66*omegaVar*m_xx_I - 302*omegaVar*m_yy_I + 48*omegaVar*m_yz_I + 11*omegaVar*m_zz_I - 72)/(27*(72*m_xx_I + 12*m_yy_I - 33*m_yz_I + 12*m_zz_I - 72*omegaVar*m_xx_I - 12*omegaVar*m_yy_I + 33*omegaVar*m_yz_I - 12*omegaVar*m_zz_I + 107));
    m_yz_t90 = -(432*m_yy_I - 198*m_xx_I - 3978*m_yz_I + 432*m_zz_I - 121*omegaVar*m_xx_I + 32*omegaVar*m_yy_I - 401*omegaVar*m_yz_I + 32*omegaVar*m_zz_I + 132)/(36*(72*m_xx_I + 12*m_yy_I - 33*m_yz_I + 12*m_zz_I - 72*omegaVar*m_xx_I - 12*omegaVar*m_yy_I + 33*omegaVar*m_yz_I - 12*omegaVar*m_zz_I + 107));
    m_zz_t45 = -(108*m_xx_I + 18*m_yy_I + 648*m_yz_I - 2772*m_zz_I + 66*omegaVar*m_xx_I + 11*omegaVar*m_yy_I + 48*omegaVar*m_yz_I - 302*omegaVar*m_zz_I - 72)/(27*(72*m_xx_I + 12*m_yy_I - 33*m_yz_I + 12*m_zz_I - 72*omegaVar*m_xx_I - 12*omegaVar*m_yy_I + 33*omegaVar*m_yz_I - 12*omegaVar*m_zz_I + 107));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = ((omegaVar*m_xx_t45)/2 - m_xx_t45 - 1.0/3.0)*rho + RHO_0/3;
    L_BC_Fy = (m_xx_t45/12 - m_xy_t90 - m_yy_t45/6 + m_yz_t90/2 - m_zz_t45/6 - (omegaVar*m_xx_t45)/24 + (omegaVar*m_xy_t90)/2 + (omegaVar*m_yy_t45)/12 - (omegaVar*m_yz_t90)/4 + (omegaVar*m_zz_t45)/12 - 1.0/18.0)*rho + RHO_0/18;
    L_BC_Fz = (m_yy_t45/6 - m_xz_t90 - m_xx_t45/12 - m_yz_t90/2 + m_zz_t45/6 + (omegaVar*m_xx_t45)/24 + (omegaVar*m_xz_t90)/2 - (omegaVar*m_yy_t45)/12 + (omegaVar*m_yz_t90)/4 - (omegaVar*m_zz_t45)/12 + 1.0/18.0)*rho - RHO_0/18;
    #endif //_BC_FORCES 

    break;
case 0b0000010101110110:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[16] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);

    rho = (3*(15*rho_I + 11*m_xx_I*rho_I - 30*m_xy_I*rho_I - 6*m_xz_I*rho_I + 11*m_yy_I*rho_I - 6*m_yz_I*rho_I - m_zz_I*rho_I - 11*omegaVar*m_xx_I*rho_I + 30*omegaVar*m_xy_I*rho_I + 6*omegaVar*m_xz_I*rho_I - 11*omegaVar*m_yy_I*rho_I + 6*omegaVar*m_yz_I*rho_I + omegaVar*m_zz_I*rho_I))/(9*omegaVar + 31);

    m_xx_t45 = (2*(1206*m_xx_I - 1260*m_xy_I - 252*m_xz_I + 369*m_yy_I + 27*m_yz_I - 135*m_zz_I + 194*omegaVar*m_xx_I + 60*omegaVar*m_xy_I + 12*omegaVar*m_xz_I - 49*omegaVar*m_yy_I + 93*omegaVar*m_yz_I - 25*omegaVar*m_zz_I + 165))/(135*(11*m_xx_I - 30*m_xy_I - 6*m_xz_I + 11*m_yy_I - 6*m_yz_I - m_zz_I - 11*omegaVar*m_xx_I + 30*omegaVar*m_xy_I + 6*omegaVar*m_xz_I - 11*omegaVar*m_yy_I + 6*omegaVar*m_yz_I + omegaVar*m_zz_I + 15));
    m_xy_t90 = (339*m_xy_I - 84*m_xx_I + 12*m_xz_I - 84*m_yy_I + 12*m_yz_I + 33*m_zz_I + 4*omegaVar*m_xx_I + 21*omegaVar*m_xy_I - 12*omegaVar*m_xz_I + 4*omegaVar*m_yy_I - 12*omegaVar*m_yz_I + 7*omegaVar*m_zz_I - 30)/(9*(11*m_xx_I - 30*m_xy_I - 6*m_xz_I + 11*m_yy_I - 6*m_yz_I - m_zz_I - 11*omegaVar*m_xx_I + 30*omegaVar*m_xy_I + 6*omegaVar*m_xz_I - 11*omegaVar*m_yy_I + 6*omegaVar*m_yz_I + omegaVar*m_zz_I + 15));
    m_xz_t90 = (60*m_xy_I - 84*m_xx_I + 663*m_xz_I + 9*m_yy_I + 12*m_yz_I - 60*m_zz_I + 4*omegaVar*m_xx_I - 60*omegaVar*m_xy_I + 177*omegaVar*m_xz_I + 31*omegaVar*m_yy_I - 12*omegaVar*m_yz_I - 20*omegaVar*m_zz_I - 30)/(45*(11*m_xx_I - 30*m_xy_I - 6*m_xz_I + 11*m_yy_I - 6*m_yz_I - m_zz_I - 11*omegaVar*m_xx_I + 30*omegaVar*m_xy_I + 6*omegaVar*m_xz_I - 11*omegaVar*m_yy_I + 6*omegaVar*m_yz_I + omegaVar*m_zz_I + 15));
    m_yy_t45 = (2*(369*m_xx_I - 1260*m_xy_I + 27*m_xz_I + 1206*m_yy_I - 252*m_yz_I - 135*m_zz_I - 49*omegaVar*m_xx_I + 60*omegaVar*m_xy_I + 93*omegaVar*m_xz_I + 194*omegaVar*m_yy_I + 12*omegaVar*m_yz_I - 25*omegaVar*m_zz_I + 165))/(135*(11*m_xx_I - 30*m_xy_I - 6*m_xz_I + 11*m_yy_I - 6*m_yz_I - m_zz_I - 11*omegaVar*m_xx_I + 30*omegaVar*m_xy_I + 6*omegaVar*m_xz_I - 11*omegaVar*m_yy_I + 6*omegaVar*m_yz_I + omegaVar*m_zz_I + 15));
    m_yz_t90 = (9*m_xx_I + 60*m_xy_I + 12*m_xz_I - 84*m_yy_I + 663*m_yz_I - 60*m_zz_I + 31*omegaVar*m_xx_I - 60*omegaVar*m_xy_I - 12*omegaVar*m_xz_I + 4*omegaVar*m_yy_I + 177*omegaVar*m_yz_I - 20*omegaVar*m_zz_I - 30)/(45*(11*m_xx_I - 30*m_xy_I - 6*m_xz_I + 11*m_yy_I - 6*m_yz_I - m_zz_I - 11*omegaVar*m_xx_I + 30*omegaVar*m_xy_I + 6*omegaVar*m_xz_I - 11*omegaVar*m_yy_I + 6*omegaVar*m_yz_I + omegaVar*m_zz_I + 15));
    m_zz_t45 = -(2*(27*m_xx_I - 99*m_xy_I + 36*m_xz_I + 27*m_yy_I + 36*m_yz_I - 180*m_zz_I + 5*omegaVar*m_xx_I - 21*omegaVar*m_xy_I + 12*omegaVar*m_xz_I + 5*omegaVar*m_yy_I + 12*omegaVar*m_yz_I - 52*omegaVar*m_zz_I + 3))/(27*(11*m_xx_I - 30*m_xy_I - 6*m_xz_I + 11*m_yy_I - 6*m_yz_I - m_zz_I - 11*omegaVar*m_xx_I + 30*omegaVar*m_xy_I + 6*omegaVar*m_xz_I - 11*omegaVar*m_yy_I + 6*omegaVar*m_yz_I + omegaVar*m_zz_I + 15));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = (m_xz_t90/2 - m_xx_t45/6 + m_yy_t45/12 - m_zz_t45/6 + (omegaVar*m_xx_t45)/12 - (omegaVar*m_xz_t90)/4 - (omegaVar*m_yy_t45)/24 + (omegaVar*m_zz_t45)/12 - 1.0/18.0)*rho + RHO_0/18;
    L_BC_Fy = (m_xx_t45/12 - m_yy_t45/6 + m_yz_t90/2 - m_zz_t45/6 - (omegaVar*m_xx_t45)/24 + (omegaVar*m_yy_t45)/12 - (omegaVar*m_yz_t90)/4 + (omegaVar*m_zz_t45)/12 - 1.0/18.0)*rho + RHO_0/18;
    L_BC_Fz = (m_xx_t45/12 - m_xz_t90/2 + m_yy_t45/12 - m_yz_t90/2 + m_zz_t45/3 - (omegaVar*m_xx_t45)/24 + (omegaVar*m_xz_t90)/4 - (omegaVar*m_yy_t45)/24 + (omegaVar*m_yz_t90)/4 - (omegaVar*m_zz_t45)/6 + 1.0/9.0)*rho - RHO_0/9;
    #endif //_BC_FORCES 

    break;
case 0b0000010101110111:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[2] + pop[4] + pop[5] + pop[6] + pop[8] + pop[10] + pop[12] + pop[16] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[8]);
    m_xz_I = inv_rho_I * (pop[10] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[12] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);

    rho = (36*(23*rho_I + 24*m_xx_I*rho_I - 57*m_xy_I*rho_I + 24*m_yy_I*rho_I - 6*m_zz_I*rho_I - 24*omegaVar*m_xx_I*rho_I + 57*omegaVar*m_xy_I*rho_I - 24*omegaVar*m_yy_I*rho_I + 6*omegaVar*m_zz_I*rho_I))/(5*(43*omegaVar + 72));

    m_xx_t45 = (936*m_xx_I - 1008*m_xy_I + 216*m_yy_I - 144*m_zz_I + 239*omegaVar*m_xx_I + 158*omegaVar*m_xy_I - 191*omegaVar*m_yy_I - 6*omegaVar*m_zz_I + 192)/(36*(24*m_xx_I - 57*m_xy_I + 24*m_yy_I - 6*m_zz_I - 24*omegaVar*m_xx_I + 57*omegaVar*m_xy_I - 24*omegaVar*m_yy_I + 6*omegaVar*m_zz_I + 23));
    m_xy_t90 = (2412*m_xy_I - 504*m_xx_I - 504*m_yy_I + 216*m_zz_I + 79*omegaVar*m_xx_I + 538*omegaVar*m_xy_I + 79*omegaVar*m_yy_I + 34*omegaVar*m_zz_I - 228)/(36*(24*m_xx_I - 57*m_xy_I + 24*m_yy_I - 6*m_zz_I - 24*omegaVar*m_xx_I + 57*omegaVar*m_xy_I - 24*omegaVar*m_yy_I + 6*omegaVar*m_zz_I + 23));
    m_xz_t90 = (5*m_xz_I*(43*omegaVar + 72))/(18*(24*m_xx_I - 57*m_xy_I + 24*m_yy_I - 6*m_zz_I - 24*omegaVar*m_xx_I + 57*omegaVar*m_xy_I - 24*omegaVar*m_yy_I + 6*omegaVar*m_zz_I + 23));
    m_yy_t45 = (216*m_xx_I - 1008*m_xy_I + 936*m_yy_I - 144*m_zz_I - 191*omegaVar*m_xx_I + 158*omegaVar*m_xy_I + 239*omegaVar*m_yy_I - 6*omegaVar*m_zz_I + 192)/(36*(24*m_xx_I - 57*m_xy_I + 24*m_yy_I - 6*m_zz_I - 24*omegaVar*m_xx_I + 57*omegaVar*m_xy_I - 24*omegaVar*m_yy_I + 6*omegaVar*m_zz_I + 23));
    m_yz_t90 = (5*m_yz_I*(43*omegaVar + 72))/(18*(24*m_xx_I - 57*m_xy_I + 24*m_yy_I - 6*m_zz_I - 24*omegaVar*m_xx_I + 57*omegaVar*m_xy_I - 24*omegaVar*m_yy_I + 6*omegaVar*m_zz_I + 23));
    m_zz_t45 = -(72*m_xx_I - 216*m_xy_I + 72*m_yy_I - 288*m_zz_I + 3*omegaVar*m_xx_I - 34*omegaVar*m_xy_I + 3*omegaVar*m_yy_I - 162*omegaVar*m_zz_I + 24)/(18*(24*m_xx_I - 57*m_xy_I + 24*m_yy_I - 6*m_zz_I - 24*omegaVar*m_xx_I + 57*omegaVar*m_xy_I - 24*omegaVar*m_yy_I + 6*omegaVar*m_zz_I + 23));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = (m_yy_t45/6 - m_xy_t90/2 - (5*m_xx_t45)/6 - m_zz_t45/12 + (5*omegaVar*m_xx_t45)/12 + (omegaVar*m_xy_t90)/4 - (omegaVar*m_yy_t45)/12 + (omegaVar*m_zz_t45)/24 - 5/18)*rho + (5*RHO_0)/18;
    L_BC_Fy = (m_xx_t45/6 - m_xy_t90/2 - (5*m_yy_t45)/6 - m_zz_t45/12 - (omegaVar*m_xx_t45)/12 + (omegaVar*m_xy_t90)/4 + (5*omegaVar*m_yy_t45)/12 + (omegaVar*m_zz_t45)/24 - 5/18)*rho + (5*RHO_0)/18;
    L_BC_Fz = (rho*(m_xz_t90 + m_yz_t90)*(omegaVar - 2))/2;
    #endif //_BC_FORCES 

    break;
case 0b0000010101111000:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[16] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);

    rho = (6*(15*rho_I + m_xx_I*rho_I - 6*m_xz_I*rho_I + m_yy_I*rho_I - 6*m_yz_I*rho_I + 4*m_zz_I*rho_I - omegaVar*m_xx_I*rho_I + 6*omegaVar*m_xz_I*rho_I - omegaVar*m_yy_I*rho_I + 6*omegaVar*m_yz_I*rho_I - 4*omegaVar*m_zz_I*rho_I))/(3*omegaVar + 82);

    m_xx_t45 = (2052*m_xx_I - 504*m_xz_I - 162*m_yy_I + 234*m_yz_I + 90*m_zz_I + 73*omegaVar*m_xx_I - 6*omegaVar*m_xz_I - 8*omegaVar*m_yy_I + 21*omegaVar*m_yz_I - 5*omegaVar*m_zz_I + 30)/(135*(m_xx_I - 6*m_xz_I + m_yy_I - 6*m_yz_I + 4*m_zz_I - omegaVar*m_xx_I + 6*omegaVar*m_xz_I - omegaVar*m_yy_I + 6*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 15));
    m_xy_t90 = (m_xy_I*(3*omegaVar + 82))/(6*(m_xx_I - 6*m_xz_I + m_yy_I - 6*m_yz_I + 4*m_zz_I - omegaVar*m_xx_I + 6*omegaVar*m_xz_I - omegaVar*m_yy_I + 6*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 15));
    m_xz_t90 = (1746*m_xz_I - 168*m_xx_I + 78*m_yy_I + 24*m_yz_I - 180*m_zz_I - 2*omegaVar*m_xx_I + 39*omegaVar*m_xz_I + 7*omegaVar*m_yy_I - 24*omegaVar*m_yz_I + 10*omegaVar*m_zz_I - 60)/(90*(m_xx_I - 6*m_xz_I + m_yy_I - 6*m_yz_I + 4*m_zz_I - omegaVar*m_xx_I + 6*omegaVar*m_xz_I - omegaVar*m_yy_I + 6*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 15));
    m_yy_t45 = (234*m_xz_I - 162*m_xx_I + 2052*m_yy_I - 504*m_yz_I + 90*m_zz_I - 8*omegaVar*m_xx_I + 21*omegaVar*m_xz_I + 73*omegaVar*m_yy_I - 6*omegaVar*m_yz_I - 5*omegaVar*m_zz_I + 30)/(135*(m_xx_I - 6*m_xz_I + m_yy_I - 6*m_yz_I + 4*m_zz_I - omegaVar*m_xx_I + 6*omegaVar*m_xz_I - omegaVar*m_yy_I + 6*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 15));
    m_yz_t90 = (78*m_xx_I + 24*m_xz_I - 168*m_yy_I + 1746*m_yz_I - 180*m_zz_I + 7*omegaVar*m_xx_I - 24*omegaVar*m_xz_I - 2*omegaVar*m_yy_I + 39*omegaVar*m_yz_I + 10*omegaVar*m_zz_I - 60)/(90*(m_xx_I - 6*m_xz_I + m_yy_I - 6*m_yz_I + 4*m_zz_I - omegaVar*m_xx_I + 6*omegaVar*m_xz_I - omegaVar*m_yy_I + 6*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 15));
    m_zz_t45 = (36*m_xx_I - 216*m_xz_I + 36*m_yy_I - 216*m_yz_I + 882*m_zz_I - 2*omegaVar*m_xx_I + 12*omegaVar*m_xz_I - 2*omegaVar*m_yy_I + 12*omegaVar*m_yz_I + 19*omegaVar*m_zz_I + 48)/(54*(m_xx_I - 6*m_xz_I + m_yy_I - 6*m_yz_I + 4*m_zz_I - omegaVar*m_xx_I + 6*omegaVar*m_xz_I - omegaVar*m_yy_I + 6*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 15));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = (m_xz_t90/2 - m_xx_t45/6 + m_yy_t45/12 - m_zz_t45/6 + (omegaVar*m_xx_t45)/12 - (omegaVar*m_xz_t90)/4 - (omegaVar*m_yy_t45)/24 + (omegaVar*m_zz_t45)/12 - 1.0/18.0)*rho + RHO_0/18;
    L_BC_Fy = (m_xx_t45/12 - m_yy_t45/6 + m_yz_t90/2 - m_zz_t45/6 - (omegaVar*m_xx_t45)/24 + (omegaVar*m_yy_t45)/12 - (omegaVar*m_yz_t90)/4 + (omegaVar*m_zz_t45)/12 - 1.0/18.0)*rho + RHO_0/18;
    L_BC_Fz = (m_xx_t45/12 - m_xz_t90/2 + m_yy_t45/12 - m_yz_t90/2 + m_zz_t45/3 - (omegaVar*m_xx_t45)/24 + (omegaVar*m_xz_t90)/4 - (omegaVar*m_yy_t45)/24 + (omegaVar*m_yz_t90)/4 - (omegaVar*m_zz_t45)/6 + 1.0/9.0)*rho - RHO_0/9;
    #endif //_BC_FORCES 

    break;
case 0b0000010101111001:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[16] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);

    rho = (12*(5*rho_I + m_xx_I*rho_I + 2*m_xy_I*rho_I - 2*m_xz_I*rho_I + m_yy_I*rho_I - 2*m_yz_I*rho_I + m_zz_I*rho_I - omegaVar*m_xx_I*rho_I - 2*omegaVar*m_xy_I*rho_I + 2*omegaVar*m_xz_I*rho_I - omegaVar*m_yy_I*rho_I + 2*omegaVar*m_yz_I*rho_I - omegaVar*m_zz_I*rho_I))/(3*omegaVar + 52);

    m_xx_t45 = (472*m_xx_I + 112*m_xy_I - 112*m_xz_I + 4*m_yy_I + 44*m_yz_I + 4*m_zz_I + 23*omegaVar*m_xx_I - 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - 4*omegaVar*m_yy_I + 11*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 20)/(90*(m_xx_I + 2*m_xy_I - 2*m_xz_I + m_yy_I - 2*m_yz_I + m_zz_I - omegaVar*m_xx_I - 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - omegaVar*m_yy_I + 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_xy_t90 = (112*m_xx_I + 1108*m_xy_I - 16*m_xz_I + 112*m_yy_I - 16*m_yz_I - 44*m_zz_I - 2*omegaVar*m_xx_I + 47*omegaVar*m_xy_I + 16*omegaVar*m_xz_I - 2*omegaVar*m_yy_I + 16*omegaVar*m_yz_I - 11*omegaVar*m_zz_I + 40)/(180*(m_xx_I + 2*m_xy_I - 2*m_xz_I + m_yy_I - 2*m_yz_I + m_zz_I - omegaVar*m_xx_I - 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - omegaVar*m_yy_I + 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_xz_t90 = (1108*m_xz_I - 16*m_xy_I - 112*m_xx_I + 44*m_yy_I + 16*m_yz_I - 112*m_zz_I + 2*omegaVar*m_xx_I + 16*omegaVar*m_xy_I + 47*omegaVar*m_xz_I + 11*omegaVar*m_yy_I - 16*omegaVar*m_yz_I + 2*omegaVar*m_zz_I - 40)/(180*(m_xx_I + 2*m_xy_I - 2*m_xz_I + m_yy_I - 2*m_yz_I + m_zz_I - omegaVar*m_xx_I - 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - omegaVar*m_yy_I + 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_yy_t45 = (4*m_xx_I + 112*m_xy_I + 44*m_xz_I + 472*m_yy_I - 112*m_yz_I + 4*m_zz_I - 4*omegaVar*m_xx_I - 2*omegaVar*m_xy_I + 11*omegaVar*m_xz_I + 23*omegaVar*m_yy_I + 2*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 20)/(90*(m_xx_I + 2*m_xy_I - 2*m_xz_I + m_yy_I - 2*m_yz_I + m_zz_I - omegaVar*m_xx_I - 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - omegaVar*m_yy_I + 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_yz_t90 = (44*m_xx_I - 16*m_xy_I + 16*m_xz_I - 112*m_yy_I + 1108*m_yz_I - 112*m_zz_I + 11*omegaVar*m_xx_I + 16*omegaVar*m_xy_I - 16*omegaVar*m_xz_I + 2*omegaVar*m_yy_I + 47*omegaVar*m_yz_I + 2*omegaVar*m_zz_I - 40)/(180*(m_xx_I + 2*m_xy_I - 2*m_xz_I + m_yy_I - 2*m_yz_I + m_zz_I - omegaVar*m_xx_I - 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - omegaVar*m_yy_I + 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_zz_t45 = (4*m_xx_I - 44*m_xy_I - 112*m_xz_I + 4*m_yy_I - 112*m_yz_I + 472*m_zz_I - 4*omegaVar*m_xx_I - 11*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - 4*omegaVar*m_yy_I + 2*omegaVar*m_yz_I + 23*omegaVar*m_zz_I + 20)/(90*(m_xx_I + 2*m_xy_I - 2*m_xz_I + m_yy_I - 2*m_yz_I + m_zz_I - omegaVar*m_xx_I - 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - omegaVar*m_yy_I + 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = (m_xz_t90/2 - m_xy_t90/2 - m_xx_t45/3 - m_yy_t45/12 - m_zz_t45/12 + (omegaVar*m_xx_t45)/6 + (omegaVar*m_xy_t90)/4 - (omegaVar*m_xz_t90)/4 + (omegaVar*m_yy_t45)/24 + (omegaVar*m_zz_t45)/24 - 1.0/9.0)*rho + RHO_0/9;
    L_BC_Fy = (m_yz_t90/2 - m_xy_t90/2 - m_yy_t45/3 - m_xx_t45/12 - m_zz_t45/12 + (omegaVar*m_xx_t45)/24 + (omegaVar*m_xy_t90)/4 + (omegaVar*m_yy_t45)/6 - (omegaVar*m_yz_t90)/4 + (omegaVar*m_zz_t45)/24 - 1.0/9.0)*rho + RHO_0/9;
    L_BC_Fz = (m_xx_t45/12 - m_xz_t90/2 + m_yy_t45/12 - m_yz_t90/2 + m_zz_t45/3 - (omegaVar*m_xx_t45)/24 + (omegaVar*m_xz_t90)/4 - (omegaVar*m_yy_t45)/24 + (omegaVar*m_yz_t90)/4 - (omegaVar*m_zz_t45)/6 + 1.0/9.0)*rho - RHO_0/9;
    #endif //_BC_FORCES 

    break;
case 0b0000010101111010:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] + (2.0/3.0)*pop[18]);

    rho = (3*(15*rho_I + 11*m_xx_I*rho_I - 6*m_xy_I*rho_I - 30*m_xz_I*rho_I - m_yy_I*rho_I - 6*m_yz_I*rho_I + 11*m_zz_I*rho_I - 11*omegaVar*m_xx_I*rho_I + 6*omegaVar*m_xy_I*rho_I + 30*omegaVar*m_xz_I*rho_I + omegaVar*m_yy_I*rho_I + 6*omegaVar*m_yz_I*rho_I - 11*omegaVar*m_zz_I*rho_I))/(9*omegaVar + 31);

    m_xx_t45 = (2*(1206*m_xx_I - 252*m_xy_I - 1260*m_xz_I - 135*m_yy_I + 27*m_yz_I + 369*m_zz_I + 194*omegaVar*m_xx_I + 12*omegaVar*m_xy_I + 60*omegaVar*m_xz_I - 25*omegaVar*m_yy_I + 93*omegaVar*m_yz_I - 49*omegaVar*m_zz_I + 165))/(135*(11*m_xx_I - 6*m_xy_I - 30*m_xz_I - m_yy_I - 6*m_yz_I + 11*m_zz_I - 11*omegaVar*m_xx_I + 6*omegaVar*m_xy_I + 30*omegaVar*m_xz_I + omegaVar*m_yy_I + 6*omegaVar*m_yz_I - 11*omegaVar*m_zz_I + 15));
    m_xy_t90 = (663*m_xy_I - 84*m_xx_I + 60*m_xz_I - 60*m_yy_I + 12*m_yz_I + 9*m_zz_I + 4*omegaVar*m_xx_I + 177*omegaVar*m_xy_I - 60*omegaVar*m_xz_I - 20*omegaVar*m_yy_I - 12*omegaVar*m_yz_I + 31*omegaVar*m_zz_I - 30)/(45*(11*m_xx_I - 6*m_xy_I - 30*m_xz_I - m_yy_I - 6*m_yz_I + 11*m_zz_I - 11*omegaVar*m_xx_I + 6*omegaVar*m_xy_I + 30*omegaVar*m_xz_I + omegaVar*m_yy_I + 6*omegaVar*m_yz_I - 11*omegaVar*m_zz_I + 15));
    m_xz_t90 = (12*m_xy_I - 84*m_xx_I + 339*m_xz_I + 33*m_yy_I + 12*m_yz_I - 84*m_zz_I + 4*omegaVar*m_xx_I - 12*omegaVar*m_xy_I + 21*omegaVar*m_xz_I + 7*omegaVar*m_yy_I - 12*omegaVar*m_yz_I + 4*omegaVar*m_zz_I - 30)/(9*(11*m_xx_I - 6*m_xy_I - 30*m_xz_I - m_yy_I - 6*m_yz_I + 11*m_zz_I - 11*omegaVar*m_xx_I + 6*omegaVar*m_xy_I + 30*omegaVar*m_xz_I + omegaVar*m_yy_I + 6*omegaVar*m_yz_I - 11*omegaVar*m_zz_I + 15));
    m_yy_t45 = -(2*(27*m_xx_I + 36*m_xy_I - 99*m_xz_I - 180*m_yy_I + 36*m_yz_I + 27*m_zz_I + 5*omegaVar*m_xx_I + 12*omegaVar*m_xy_I - 21*omegaVar*m_xz_I - 52*omegaVar*m_yy_I + 12*omegaVar*m_yz_I + 5*omegaVar*m_zz_I + 3))/(27*(11*m_xx_I - 6*m_xy_I - 30*m_xz_I - m_yy_I - 6*m_yz_I + 11*m_zz_I - 11*omegaVar*m_xx_I + 6*omegaVar*m_xy_I + 30*omegaVar*m_xz_I + omegaVar*m_yy_I + 6*omegaVar*m_yz_I - 11*omegaVar*m_zz_I + 15));
    m_yz_t90 = (9*m_xx_I + 12*m_xy_I + 60*m_xz_I - 60*m_yy_I + 663*m_yz_I - 84*m_zz_I + 31*omegaVar*m_xx_I - 12*omegaVar*m_xy_I - 60*omegaVar*m_xz_I - 20*omegaVar*m_yy_I + 177*omegaVar*m_yz_I + 4*omegaVar*m_zz_I - 30)/(45*(11*m_xx_I - 6*m_xy_I - 30*m_xz_I - m_yy_I - 6*m_yz_I + 11*m_zz_I - 11*omegaVar*m_xx_I + 6*omegaVar*m_xy_I + 30*omegaVar*m_xz_I + omegaVar*m_yy_I + 6*omegaVar*m_yz_I - 11*omegaVar*m_zz_I + 15));
    m_zz_t45 = (2*(369*m_xx_I + 27*m_xy_I - 1260*m_xz_I - 135*m_yy_I - 252*m_yz_I + 1206*m_zz_I - 49*omegaVar*m_xx_I + 93*omegaVar*m_xy_I + 60*omegaVar*m_xz_I - 25*omegaVar*m_yy_I + 12*omegaVar*m_yz_I + 194*omegaVar*m_zz_I + 165))/(135*(11*m_xx_I - 6*m_xy_I - 30*m_xz_I - m_yy_I - 6*m_yz_I + 11*m_zz_I - 11*omegaVar*m_xx_I + 6*omegaVar*m_xy_I + 30*omegaVar*m_xz_I + omegaVar*m_yy_I + 6*omegaVar*m_yz_I - 11*omegaVar*m_zz_I + 15));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = (m_xx_t45/6 - m_xy_t90/2 + m_yy_t45/6 - m_zz_t45/12 - (omegaVar*m_xx_t45)/12 + (omegaVar*m_xy_t90)/4 - (omegaVar*m_yy_t45)/12 + (omegaVar*m_zz_t45)/24 + 1.0/18.0)*rho - RHO_0/18;
    L_BC_Fy = (m_xy_t90/2 - m_xx_t45/12 - m_yy_t45/3 + m_yz_t90/2 - m_zz_t45/12 + (omegaVar*m_xx_t45)/24 - (omegaVar*m_xy_t90)/4 + (omegaVar*m_yy_t45)/6 - (omegaVar*m_yz_t90)/4 + (omegaVar*m_zz_t45)/24 - 1.0/9.0)*rho + RHO_0/9;
    L_BC_Fz = (m_yy_t45/6 - m_xx_t45/12 - m_yz_t90/2 + m_zz_t45/6 + (omegaVar*m_xx_t45)/24 - (omegaVar*m_yy_t45)/12 + (omegaVar*m_yz_t90)/4 - (omegaVar*m_zz_t45)/12 + 1.0/18.0)*rho - RHO_0/18;
    #endif //_BC_FORCES 

    break;
case 0b0000010101111011:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[4] + pop[5] + pop[6] + pop[8] + pop[9] + pop[10] + pop[12] + pop[13] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[8] - pop[13]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[12] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] + (2.0/3.0)*pop[18]);

    rho = (18*(19*rho_I + 12*m_xx_I*rho_I - 33*m_xz_I*rho_I + 6*m_yy_I*rho_I + 12*m_zz_I*rho_I - 12*omegaVar*m_xx_I*rho_I + 33*omegaVar*m_xz_I*rho_I - 6*omegaVar*m_yy_I*rho_I - 12*omegaVar*m_zz_I*rho_I))/(59*omegaVar + 207);

    m_xx_t45 = -(2*(324*m_xz_I - 306*m_xx_I + 54*m_yy_I - 99*m_zz_I - 58*omegaVar*m_xx_I + 12*omegaVar*m_xz_I + 30*omegaVar*m_yy_I + omegaVar*m_zz_I - 36))/(27*(12*m_xx_I - 33*m_xz_I + 6*m_yy_I + 12*m_zz_I - 12*omegaVar*m_xx_I + 33*omegaVar*m_xz_I - 6*omegaVar*m_yy_I - 12*omegaVar*m_zz_I + 19));
    m_xy_t90 = (m_xy_I*(59*omegaVar + 207))/(9*(12*m_xx_I - 33*m_xz_I + 6*m_yy_I + 12*m_zz_I - 12*omegaVar*m_xx_I + 33*omegaVar*m_xz_I - 6*omegaVar*m_yy_I - 12*omegaVar*m_zz_I + 19));
    m_xz_t90 = -(216*m_xx_I - 801*m_xz_I - 99*m_yy_I + 216*m_zz_I + 8*omegaVar*m_xx_I - 81*omegaVar*m_xz_I - 55*omegaVar*m_yy_I + 8*omegaVar*m_zz_I + 66)/(18*(12*m_xx_I - 33*m_xz_I + 6*m_yy_I + 12*m_zz_I - 12*omegaVar*m_xx_I + 33*omegaVar*m_xz_I - 6*omegaVar*m_yy_I - 12*omegaVar*m_zz_I + 19));
    m_yy_t45 = (99*m_xz_I - 36*m_xx_I + 189*m_yy_I - 36*m_zz_I - 20*omegaVar*m_xx_I + 55*omegaVar*m_xz_I + 49*omegaVar*m_yy_I - 20*omegaVar*m_zz_I + 12)/(9*(12*m_xx_I - 33*m_xz_I + 6*m_yy_I + 12*m_zz_I - 12*omegaVar*m_xx_I + 33*omegaVar*m_xz_I - 6*omegaVar*m_yy_I - 12*omegaVar*m_zz_I + 19));
    m_yz_t90 = (m_yz_I*(59*omegaVar + 207))/(9*(12*m_xx_I - 33*m_xz_I + 6*m_yy_I + 12*m_zz_I - 12*omegaVar*m_xx_I + 33*omegaVar*m_xz_I - 6*omegaVar*m_yy_I - 12*omegaVar*m_zz_I + 19));
    m_zz_t45 = -(2*(324*m_xz_I - 99*m_xx_I + 54*m_yy_I - 306*m_zz_I + omegaVar*m_xx_I + 12*omegaVar*m_xz_I + 30*omegaVar*m_yy_I - 58*omegaVar*m_zz_I - 36))/(27*(12*m_xx_I - 33*m_xz_I + 6*m_yy_I + 12*m_zz_I - 12*omegaVar*m_xx_I + 33*omegaVar*m_xz_I - 6*omegaVar*m_yy_I - 12*omegaVar*m_zz_I + 19));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = (m_xy_t90*rho*(omegaVar - 2))/2;
    L_BC_Fy = ((omegaVar*m_yy_t45)/2 - m_yy_t45 - 1.0/3.0)*rho + RHO_0/3;
    L_BC_Fz = (m_yz_t90*rho*(omegaVar - 2))/2;
    #endif //_BC_FORCES 

    break;
case 0b0000010101111100:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[14] + pop[16];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[16]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[16]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[16]);

    rho = (3*(15*rho_I - m_xx_I*rho_I - 6*m_xy_I*rho_I - 6*m_xz_I*rho_I + 11*m_yy_I*rho_I - 30*m_yz_I*rho_I + 11*m_zz_I*rho_I + omegaVar*m_xx_I*rho_I + 6*omegaVar*m_xy_I*rho_I + 6*omegaVar*m_xz_I*rho_I - 11*omegaVar*m_yy_I*rho_I + 30*omegaVar*m_yz_I*rho_I - 11*omegaVar*m_zz_I*rho_I))/(9*omegaVar + 31);

    m_xx_t45 = -(2*(36*m_xy_I - 180*m_xx_I + 36*m_xz_I + 27*m_yy_I - 99*m_yz_I + 27*m_zz_I - 52*omegaVar*m_xx_I + 12*omegaVar*m_xy_I + 12*omegaVar*m_xz_I + 5*omegaVar*m_yy_I - 21*omegaVar*m_yz_I + 5*omegaVar*m_zz_I + 3))/(27*(11*m_yy_I - 6*m_xy_I - 6*m_xz_I - m_xx_I - 30*m_yz_I + 11*m_zz_I + omegaVar*m_xx_I + 6*omegaVar*m_xy_I + 6*omegaVar*m_xz_I - 11*omegaVar*m_yy_I + 30*omegaVar*m_yz_I - 11*omegaVar*m_zz_I + 15));
    m_xy_t90 = (663*m_xy_I - 60*m_xx_I + 12*m_xz_I - 84*m_yy_I + 60*m_yz_I + 9*m_zz_I - 20*omegaVar*m_xx_I + 177*omegaVar*m_xy_I - 12*omegaVar*m_xz_I + 4*omegaVar*m_yy_I - 60*omegaVar*m_yz_I + 31*omegaVar*m_zz_I - 30)/(45*(11*m_yy_I - 6*m_xy_I - 6*m_xz_I - m_xx_I - 30*m_yz_I + 11*m_zz_I + omegaVar*m_xx_I + 6*omegaVar*m_xy_I + 6*omegaVar*m_xz_I - 11*omegaVar*m_yy_I + 30*omegaVar*m_yz_I - 11*omegaVar*m_zz_I + 15));
    m_xz_t90 = (12*m_xy_I - 60*m_xx_I + 663*m_xz_I + 9*m_yy_I + 60*m_yz_I - 84*m_zz_I - 20*omegaVar*m_xx_I - 12*omegaVar*m_xy_I + 177*omegaVar*m_xz_I + 31*omegaVar*m_yy_I - 60*omegaVar*m_yz_I + 4*omegaVar*m_zz_I - 30)/(45*(11*m_yy_I - 6*m_xy_I - 6*m_xz_I - m_xx_I - 30*m_yz_I + 11*m_zz_I + omegaVar*m_xx_I + 6*omegaVar*m_xy_I + 6*omegaVar*m_xz_I - 11*omegaVar*m_yy_I + 30*omegaVar*m_yz_I - 11*omegaVar*m_zz_I + 15));
    m_yy_t45 = (2*(27*m_xz_I - 252*m_xy_I - 135*m_xx_I + 1206*m_yy_I - 1260*m_yz_I + 369*m_zz_I - 25*omegaVar*m_xx_I + 12*omegaVar*m_xy_I + 93*omegaVar*m_xz_I + 194*omegaVar*m_yy_I + 60*omegaVar*m_yz_I - 49*omegaVar*m_zz_I + 165))/(135*(11*m_yy_I - 6*m_xy_I - 6*m_xz_I - m_xx_I - 30*m_yz_I + 11*m_zz_I + omegaVar*m_xx_I + 6*omegaVar*m_xy_I + 6*omegaVar*m_xz_I - 11*omegaVar*m_yy_I + 30*omegaVar*m_yz_I - 11*omegaVar*m_zz_I + 15));
    m_yz_t90 = (33*m_xx_I + 12*m_xy_I + 12*m_xz_I - 84*m_yy_I + 339*m_yz_I - 84*m_zz_I + 7*omegaVar*m_xx_I - 12*omegaVar*m_xy_I - 12*omegaVar*m_xz_I + 4*omegaVar*m_yy_I + 21*omegaVar*m_yz_I + 4*omegaVar*m_zz_I - 30)/(9*(11*m_yy_I - 6*m_xy_I - 6*m_xz_I - m_xx_I - 30*m_yz_I + 11*m_zz_I + omegaVar*m_xx_I + 6*omegaVar*m_xy_I + 6*omegaVar*m_xz_I - 11*omegaVar*m_yy_I + 30*omegaVar*m_yz_I - 11*omegaVar*m_zz_I + 15));
    m_zz_t45 = (2*(27*m_xy_I - 135*m_xx_I - 252*m_xz_I + 369*m_yy_I - 1260*m_yz_I + 1206*m_zz_I - 25*omegaVar*m_xx_I + 93*omegaVar*m_xy_I + 12*omegaVar*m_xz_I - 49*omegaVar*m_yy_I + 60*omegaVar*m_yz_I + 194*omegaVar*m_zz_I + 165))/(135*(11*m_yy_I - 6*m_xy_I - 6*m_xz_I - m_xx_I - 30*m_yz_I + 11*m_zz_I + omegaVar*m_xx_I + 6*omegaVar*m_xy_I + 6*omegaVar*m_xz_I - 11*omegaVar*m_yy_I + 30*omegaVar*m_yz_I - 11*omegaVar*m_zz_I + 15));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = (m_xy_t90/2 - m_xx_t45/3 + m_xz_t90/2 - m_yy_t45/12 - m_zz_t45/12 + (omegaVar*m_xx_t45)/6 - (omegaVar*m_xy_t90)/4 - (omegaVar*m_xz_t90)/4 + (omegaVar*m_yy_t45)/24 + (omegaVar*m_zz_t45)/24 - 1.0/9.0)*rho + RHO_0/9;
    L_BC_Fy = (m_xx_t45/6 - m_xy_t90/2 + m_yy_t45/6 - m_zz_t45/12 - (omegaVar*m_xx_t45)/12 + (omegaVar*m_xy_t90)/4 - (omegaVar*m_yy_t45)/12 + (omegaVar*m_zz_t45)/24 + 1.0/18.0)*rho - RHO_0/18;
    L_BC_Fz = (m_xx_t45/6 - m_xz_t90/2 - m_yy_t45/12 + m_zz_t45/6 - (omegaVar*m_xx_t45)/12 + (omegaVar*m_xz_t90)/4 + (omegaVar*m_yy_t45)/24 - (omegaVar*m_zz_t45)/12 + 1.0/18.0)*rho - RHO_0/18;
    #endif //_BC_FORCES 

    break;
case 0b0000010101111101:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[8] + pop[10] + pop[11] + pop[12] + pop[14] + pop[16];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[16]);
    m_xy_I = inv_rho_I * (pop[8] - pop[14]);
    m_xz_I = inv_rho_I * (pop[10] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[16]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[16]);

    rho = (18*(19*rho_I + 6*m_xx_I*rho_I + 12*m_yy_I*rho_I - 33*m_yz_I*rho_I + 12*m_zz_I*rho_I - 6*omegaVar*m_xx_I*rho_I - 12*omegaVar*m_yy_I*rho_I + 33*omegaVar*m_yz_I*rho_I - 12*omegaVar*m_zz_I*rho_I))/(59*omegaVar + 207);

    m_xx_t45 = (189*m_xx_I - 36*m_yy_I + 99*m_yz_I - 36*m_zz_I + 49*omegaVar*m_xx_I - 20*omegaVar*m_yy_I + 55*omegaVar*m_yz_I - 20*omegaVar*m_zz_I + 12)/(9*(6*m_xx_I + 12*m_yy_I - 33*m_yz_I + 12*m_zz_I - 6*omegaVar*m_xx_I - 12*omegaVar*m_yy_I + 33*omegaVar*m_yz_I - 12*omegaVar*m_zz_I + 19));
    m_xy_t90 = (m_xy_I*(59*omegaVar + 207))/(9*(6*m_xx_I + 12*m_yy_I - 33*m_yz_I + 12*m_zz_I - 6*omegaVar*m_xx_I - 12*omegaVar*m_yy_I + 33*omegaVar*m_yz_I - 12*omegaVar*m_zz_I + 19));
    m_xz_t90 = (m_xz_I*(59*omegaVar + 207))/(9*(6*m_xx_I + 12*m_yy_I - 33*m_yz_I + 12*m_zz_I - 6*omegaVar*m_xx_I - 12*omegaVar*m_yy_I + 33*omegaVar*m_yz_I - 12*omegaVar*m_zz_I + 19));
    m_yy_t45 = -(2*(54*m_xx_I - 306*m_yy_I + 324*m_yz_I - 99*m_zz_I + 30*omegaVar*m_xx_I - 58*omegaVar*m_yy_I + 12*omegaVar*m_yz_I + omegaVar*m_zz_I - 36))/(27*(6*m_xx_I + 12*m_yy_I - 33*m_yz_I + 12*m_zz_I - 6*omegaVar*m_xx_I - 12*omegaVar*m_yy_I + 33*omegaVar*m_yz_I - 12*omegaVar*m_zz_I + 19));
    m_yz_t90 = -(216*m_yy_I - 99*m_xx_I - 801*m_yz_I + 216*m_zz_I - 55*omegaVar*m_xx_I + 8*omegaVar*m_yy_I - 81*omegaVar*m_yz_I + 8*omegaVar*m_zz_I + 66)/(18*(6*m_xx_I + 12*m_yy_I - 33*m_yz_I + 12*m_zz_I - 6*omegaVar*m_xx_I - 12*omegaVar*m_yy_I + 33*omegaVar*m_yz_I - 12*omegaVar*m_zz_I + 19));
    m_zz_t45 = -(2*(54*m_xx_I - 99*m_yy_I + 324*m_yz_I - 306*m_zz_I + 30*omegaVar*m_xx_I + omegaVar*m_yy_I + 12*omegaVar*m_yz_I - 58*omegaVar*m_zz_I - 36))/(27*(6*m_xx_I + 12*m_yy_I - 33*m_yz_I + 12*m_zz_I - 6*omegaVar*m_xx_I - 12*omegaVar*m_yy_I + 33*omegaVar*m_yz_I - 12*omegaVar*m_zz_I + 19));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = ((omegaVar*m_xx_t45)/2 - m_xx_t45 - 1.0/3.0)*rho + RHO_0/3;
    L_BC_Fy = (m_xy_t90*rho*(omegaVar - 2))/2;
    L_BC_Fz = (m_xz_t90*rho*(omegaVar - 2))/2;
    #endif //_BC_FORCES 

    break;
case 0b0000010101111110:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12]);

    rho = (6*(rho_I + m_xx_I*rho_I - 2*m_xy_I*rho_I - 2*m_xz_I*rho_I + m_yy_I*rho_I - 2*m_yz_I*rho_I + m_zz_I*rho_I - omegaVar*m_xx_I*rho_I + 2*omegaVar*m_xy_I*rho_I + 2*omegaVar*m_xz_I*rho_I - omegaVar*m_yy_I*rho_I + 2*omegaVar*m_yz_I*rho_I - omegaVar*m_zz_I*rho_I))/(3*omegaVar + 2);

    m_xx_t45 = -(8*m_xy_I - 8*m_xx_I + 8*m_xz_I - 2*m_yy_I + 2*m_yz_I - 2*m_zz_I - 7*omegaVar*m_xx_I + 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I + 2*omegaVar*m_yy_I - 7*omegaVar*m_yz_I + 2*omegaVar*m_zz_I - 2)/(9*(m_xx_I - 2*m_xy_I - 2*m_xz_I + m_yy_I - 2*m_yz_I + m_zz_I - omegaVar*m_xx_I + 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - omegaVar*m_yy_I + 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 1));
    m_xy_t90 = -(8*m_xx_I - 26*m_xy_I - 8*m_xz_I + 8*m_yy_I - 8*m_yz_I + 2*m_zz_I + 2*omegaVar*m_xx_I - 19*omegaVar*m_xy_I + 8*omegaVar*m_xz_I + 2*omegaVar*m_yy_I + 8*omegaVar*m_yz_I - 7*omegaVar*m_zz_I + 4)/(18*(m_xx_I - 2*m_xy_I - 2*m_xz_I + m_yy_I - 2*m_yz_I + m_zz_I - omegaVar*m_xx_I + 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - omegaVar*m_yy_I + 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 1));
    m_xz_t90 = -(8*m_xx_I - 8*m_xy_I - 26*m_xz_I + 2*m_yy_I - 8*m_yz_I + 8*m_zz_I + 2*omegaVar*m_xx_I + 8*omegaVar*m_xy_I - 19*omegaVar*m_xz_I - 7*omegaVar*m_yy_I + 8*omegaVar*m_yz_I + 2*omegaVar*m_zz_I + 4)/(18*(m_xx_I - 2*m_xy_I - 2*m_xz_I + m_yy_I - 2*m_yz_I + m_zz_I - omegaVar*m_xx_I + 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - omegaVar*m_yy_I + 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 1));
    m_yy_t45 = -(8*m_xy_I - 2*m_xx_I + 2*m_xz_I - 8*m_yy_I + 8*m_yz_I - 2*m_zz_I + 2*omegaVar*m_xx_I + 2*omegaVar*m_xy_I - 7*omegaVar*m_xz_I - 7*omegaVar*m_yy_I + 2*omegaVar*m_yz_I + 2*omegaVar*m_zz_I - 2)/(9*(m_xx_I - 2*m_xy_I - 2*m_xz_I + m_yy_I - 2*m_yz_I + m_zz_I - omegaVar*m_xx_I + 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - omegaVar*m_yy_I + 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 1));
    m_yz_t90 = -(2*m_xx_I - 8*m_xy_I - 8*m_xz_I + 8*m_yy_I - 26*m_yz_I + 8*m_zz_I - 7*omegaVar*m_xx_I + 8*omegaVar*m_xy_I + 8*omegaVar*m_xz_I + 2*omegaVar*m_yy_I - 19*omegaVar*m_yz_I + 2*omegaVar*m_zz_I + 4)/(18*(m_xx_I - 2*m_xy_I - 2*m_xz_I + m_yy_I - 2*m_yz_I + m_zz_I - omegaVar*m_xx_I + 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - omegaVar*m_yy_I + 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 1));
    m_zz_t45 = -(2*m_xy_I - 2*m_xx_I + 8*m_xz_I - 2*m_yy_I + 8*m_yz_I - 8*m_zz_I + 2*omegaVar*m_xx_I - 7*omegaVar*m_xy_I + 2*omegaVar*m_xz_I + 2*omegaVar*m_yy_I + 2*omegaVar*m_yz_I - 7*omegaVar*m_zz_I - 2)/(9*(m_xx_I - 2*m_xy_I - 2*m_xz_I + m_yy_I - 2*m_yz_I + m_zz_I - omegaVar*m_xx_I + 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - omegaVar*m_yy_I + 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 1));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = 0;
    L_BC_Fy = 0;
    L_BC_Fz = 0;
    #endif //_BC_FORCES 

    break;
case 0b0000010101111111:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[2] + pop[4] + pop[6] + pop[8] + pop[10] + pop[12];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[12]);
    m_xy_I = inv_rho_I * (pop[8]);
    m_xz_I = inv_rho_I * (pop[10]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[12]);
    m_yz_I = inv_rho_I * (pop[12]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[12]);

    rho = (12*(rho_I + m_xx_I*rho_I - 2*m_xy_I*rho_I - 2*m_xz_I*rho_I + m_yy_I*rho_I - 2*m_yz_I*rho_I + m_zz_I*rho_I - omegaVar*m_xx_I*rho_I + 2*omegaVar*m_xy_I*rho_I + 2*omegaVar*m_xz_I*rho_I - omegaVar*m_yy_I*rho_I + 2*omegaVar*m_yz_I*rho_I - omegaVar*m_zz_I*rho_I))/(5*omegaVar + 2);

    m_xx_t45 = -(14*m_xy_I - 14*m_xx_I + 14*m_xz_I - 2*m_yy_I + 2*m_yz_I - 2*m_zz_I - 21*omegaVar*m_xx_I + 7*omegaVar*m_xy_I + 7*omegaVar*m_xz_I + 9*omegaVar*m_yy_I - 23*omegaVar*m_yz_I + 9*omegaVar*m_zz_I - 4)/(18*(m_xx_I - 2*m_xy_I - 2*m_xz_I + m_yy_I - 2*m_yz_I + m_zz_I - omegaVar*m_xx_I + 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - omegaVar*m_yy_I + 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 1));
    m_xy_t90 = -(14*m_xx_I - 50*m_xy_I - 14*m_xz_I + 14*m_yy_I - 14*m_yz_I + 2*m_zz_I + 7*omegaVar*m_xx_I - 69*omegaVar*m_xy_I + 21*omegaVar*m_xz_I + 7*omegaVar*m_yy_I + 21*omegaVar*m_yz_I - 23*omegaVar*m_zz_I + 8)/(36*(m_xx_I - 2*m_xy_I - 2*m_xz_I + m_yy_I - 2*m_yz_I + m_zz_I - omegaVar*m_xx_I + 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - omegaVar*m_yy_I + 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 1));
    m_xz_t90 = -(14*m_xx_I - 14*m_xy_I - 50*m_xz_I + 2*m_yy_I - 14*m_yz_I + 14*m_zz_I + 7*omegaVar*m_xx_I + 21*omegaVar*m_xy_I - 69*omegaVar*m_xz_I - 23*omegaVar*m_yy_I + 21*omegaVar*m_yz_I + 7*omegaVar*m_zz_I + 8)/(36*(m_xx_I - 2*m_xy_I - 2*m_xz_I + m_yy_I - 2*m_yz_I + m_zz_I - omegaVar*m_xx_I + 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - omegaVar*m_yy_I + 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 1));
    m_yy_t45 = -(14*m_xy_I - 2*m_xx_I + 2*m_xz_I - 14*m_yy_I + 14*m_yz_I - 2*m_zz_I + 9*omegaVar*m_xx_I + 7*omegaVar*m_xy_I - 23*omegaVar*m_xz_I - 21*omegaVar*m_yy_I + 7*omegaVar*m_yz_I + 9*omegaVar*m_zz_I - 4)/(18*(m_xx_I - 2*m_xy_I - 2*m_xz_I + m_yy_I - 2*m_yz_I + m_zz_I - omegaVar*m_xx_I + 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - omegaVar*m_yy_I + 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 1));
    m_yz_t90 = -(2*m_xx_I - 14*m_xy_I - 14*m_xz_I + 14*m_yy_I - 50*m_yz_I + 14*m_zz_I - 23*omegaVar*m_xx_I + 21*omegaVar*m_xy_I + 21*omegaVar*m_xz_I + 7*omegaVar*m_yy_I - 69*omegaVar*m_yz_I + 7*omegaVar*m_zz_I + 8)/(36*(m_xx_I - 2*m_xy_I - 2*m_xz_I + m_yy_I - 2*m_yz_I + m_zz_I - omegaVar*m_xx_I + 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - omegaVar*m_yy_I + 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 1));
    m_zz_t45 = -(2*m_xy_I - 2*m_xx_I + 14*m_xz_I - 2*m_yy_I + 14*m_yz_I - 14*m_zz_I + 9*omegaVar*m_xx_I - 23*omegaVar*m_xy_I + 7*omegaVar*m_xz_I + 9*omegaVar*m_yy_I + 7*omegaVar*m_yz_I - 21*omegaVar*m_zz_I - 4)/(18*(m_xx_I - 2*m_xy_I - 2*m_xz_I + m_yy_I - 2*m_yz_I + m_zz_I - omegaVar*m_xx_I + 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - omegaVar*m_yy_I + 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 1));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = (m_yy_t45/12 - m_xy_t90/2 - m_xz_t90/2 - (2*m_xx_t45)/3 + m_zz_t45/12 + (omegaVar*m_xx_t45)/3 + (omegaVar*m_xy_t90)/4 + (omegaVar*m_xz_t90)/4 - (omegaVar*m_yy_t45)/24 - (omegaVar*m_zz_t45)/24 - 2.0/9.0)*rho + (2*RHO_0)/9;
    L_BC_Fy = (m_xx_t45/12 - m_xy_t90/2 - (2*m_yy_t45)/3 - m_yz_t90/2 + m_zz_t45/12 - (omegaVar*m_xx_t45)/24 + (omegaVar*m_xy_t90)/4 + (omegaVar*m_yy_t45)/3 + (omegaVar*m_yz_t90)/4 - (omegaVar*m_zz_t45)/24 - 2.0/9.0)*rho + (2*RHO_0)/9;
    L_BC_Fz = (m_xx_t45/12 - m_xz_t90/2 + m_yy_t45/12 - m_yz_t90/2 - (2*m_zz_t45)/3 - (omegaVar*m_xx_t45)/24 + (omegaVar*m_xz_t90)/4 - (omegaVar*m_yy_t45)/24 + (omegaVar*m_yz_t90)/4 + (omegaVar*m_zz_t45)/3 - 2.0/9.0)*rho + (2*RHO_0)/9;
    #endif //_BC_FORCES 

    break;
case 0b0000010110000000:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = rho_I;

    m_xx_t45 = m_xx_I;
    m_xy_t90 = m_xy_I;
    m_xz_t90 = m_xz_I;
    m_yy_t45 = m_yy_I;
    m_yz_t90 = m_yz_I;
    m_zz_t45 = m_zz_I;

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = 0;
    L_BC_Fy = 0;
    L_BC_Fz = 0;
    #endif //_BC_FORCES 

    break;
case 0b0000010110000001:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = rho_I;

    m_xx_t45 = m_xx_I;
    m_xy_t90 = m_xy_I;
    m_xz_t90 = m_xz_I;
    m_yy_t45 = m_yy_I;
    m_yz_t90 = m_yz_I;
    m_zz_t45 = m_zz_I;

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = 0;
    L_BC_Fy = 0;
    L_BC_Fz = 0;
    #endif //_BC_FORCES 

    break;
case 0b0000010110000010:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = rho_I;

    m_xx_t45 = m_xx_I;
    m_xy_t90 = m_xy_I;
    m_xz_t90 = m_xz_I;
    m_yy_t45 = m_yy_I;
    m_yz_t90 = m_yz_I;
    m_zz_t45 = m_zz_I;

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = 0;
    L_BC_Fy = 0;
    L_BC_Fz = 0;
    #endif //_BC_FORCES 

    break;
case 0b0000010110000011:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (12*(15*rho_I - m_xx_I*rho_I + 2*m_yy_I*rho_I + 6*m_yz_I*rho_I + 2*m_zz_I*rho_I + omegaVar*m_xx_I*rho_I - 2*omegaVar*m_yy_I*rho_I - 6*omegaVar*m_yz_I*rho_I - 2*omegaVar*m_zz_I*rho_I))/(3*omegaVar + 172);

    m_xx_t45 = (792*m_xx_I - 36*m_yy_I - 108*m_yz_I - 36*m_zz_I + 13*omegaVar*m_xx_I + omegaVar*m_yy_I + 3*omegaVar*m_yz_I + omegaVar*m_zz_I - 12)/(54*(2*m_yy_I - m_xx_I + 6*m_yz_I + 2*m_zz_I + omegaVar*m_xx_I - 2*omegaVar*m_yy_I - 6*omegaVar*m_yz_I - 2*omegaVar*m_zz_I + 15));
    m_xy_t90 = (m_xy_I*(3*omegaVar + 172))/(12*(2*m_yy_I - m_xx_I + 6*m_yz_I + 2*m_zz_I + omegaVar*m_xx_I - 2*omegaVar*m_yy_I - 6*omegaVar*m_yz_I - 2*omegaVar*m_zz_I + 15));
    m_xz_t90 = (m_xz_I*(3*omegaVar + 172))/(12*(2*m_yy_I - m_xx_I + 6*m_yz_I + 2*m_zz_I + omegaVar*m_xx_I - 2*omegaVar*m_yy_I - 6*omegaVar*m_yz_I - 2*omegaVar*m_zz_I + 15));
    m_yy_t45 = (1692*m_yy_I - 72*m_xx_I + 432*m_yz_I + 144*m_zz_I + 2*omegaVar*m_xx_I + 23*omegaVar*m_yy_I - 12*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 48)/(108*(2*m_yy_I - m_xx_I + 6*m_yz_I + 2*m_zz_I + omegaVar*m_xx_I - 2*omegaVar*m_yy_I - 6*omegaVar*m_yz_I - 2*omegaVar*m_zz_I + 15));
    m_yz_t90 = (72*m_yy_I - 36*m_xx_I + 732*m_yz_I + 72*m_zz_I + omegaVar*m_xx_I - 2*omegaVar*m_yy_I + 3*omegaVar*m_yz_I - 2*omegaVar*m_zz_I + 24)/(36*(2*m_yy_I - m_xx_I + 6*m_yz_I + 2*m_zz_I + omegaVar*m_xx_I - 2*omegaVar*m_yy_I - 6*omegaVar*m_yz_I - 2*omegaVar*m_zz_I + 15));
    m_zz_t45 = (144*m_yy_I - 72*m_xx_I + 432*m_yz_I + 1692*m_zz_I + 2*omegaVar*m_xx_I - 4*omegaVar*m_yy_I - 12*omegaVar*m_yz_I + 23*omegaVar*m_zz_I + 48)/(108*(2*m_yy_I - m_xx_I + 6*m_yz_I + 2*m_zz_I + omegaVar*m_xx_I - 2*omegaVar*m_yy_I - 6*omegaVar*m_yz_I - 2*omegaVar*m_zz_I + 15));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = 0;
    L_BC_Fy = (m_xx_t45/12 - m_yy_t45/6 - m_yz_t90/2 - m_zz_t45/6 - (omegaVar*m_xx_t45)/24 + (omegaVar*m_yy_t45)/12 + (omegaVar*m_yz_t90)/4 + (omegaVar*m_zz_t45)/12 - 1.0/18.0)*rho + RHO_0/18;
    L_BC_Fz = (m_xx_t45/12 - m_yy_t45/6 - m_yz_t90/2 - m_zz_t45/6 - (omegaVar*m_xx_t45)/24 + (omegaVar*m_yy_t45)/12 + (omegaVar*m_yz_t90)/4 + (omegaVar*m_zz_t45)/12 - 1.0/18.0)*rho + RHO_0/18;
    #endif //_BC_FORCES 

    break;
case 0b0000010110000100:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = rho_I;

    m_xx_t45 = m_xx_I;
    m_xy_t90 = m_xy_I;
    m_xz_t90 = m_xz_I;
    m_yy_t45 = m_yy_I;
    m_yz_t90 = m_yz_I;
    m_zz_t45 = m_zz_I;

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = 0;
    L_BC_Fy = 0;
    L_BC_Fz = 0;
    #endif //_BC_FORCES 

    break;
case 0b0000010110000101:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (12*(15*rho_I + 2*m_xx_I*rho_I + 6*m_xz_I*rho_I - m_yy_I*rho_I + 2*m_zz_I*rho_I - 2*omegaVar*m_xx_I*rho_I - 6*omegaVar*m_xz_I*rho_I + omegaVar*m_yy_I*rho_I - 2*omegaVar*m_zz_I*rho_I))/(3*omegaVar + 172);

    m_xx_t45 = (1692*m_xx_I + 432*m_xz_I - 72*m_yy_I + 144*m_zz_I + 23*omegaVar*m_xx_I - 12*omegaVar*m_xz_I + 2*omegaVar*m_yy_I - 4*omegaVar*m_zz_I + 48)/(108*(2*m_xx_I + 6*m_xz_I - m_yy_I + 2*m_zz_I - 2*omegaVar*m_xx_I - 6*omegaVar*m_xz_I + omegaVar*m_yy_I - 2*omegaVar*m_zz_I + 15));
    m_xy_t90 = (m_xy_I*(3*omegaVar + 172))/(12*(2*m_xx_I + 6*m_xz_I - m_yy_I + 2*m_zz_I - 2*omegaVar*m_xx_I - 6*omegaVar*m_xz_I + omegaVar*m_yy_I - 2*omegaVar*m_zz_I + 15));
    m_xz_t90 = (72*m_xx_I + 732*m_xz_I - 36*m_yy_I + 72*m_zz_I - 2*omegaVar*m_xx_I + 3*omegaVar*m_xz_I + omegaVar*m_yy_I - 2*omegaVar*m_zz_I + 24)/(36*(2*m_xx_I + 6*m_xz_I - m_yy_I + 2*m_zz_I - 2*omegaVar*m_xx_I - 6*omegaVar*m_xz_I + omegaVar*m_yy_I - 2*omegaVar*m_zz_I + 15));
    m_yy_t45 = (792*m_yy_I - 108*m_xz_I - 36*m_xx_I - 36*m_zz_I + omegaVar*m_xx_I + 3*omegaVar*m_xz_I + 13*omegaVar*m_yy_I + omegaVar*m_zz_I - 12)/(54*(2*m_xx_I + 6*m_xz_I - m_yy_I + 2*m_zz_I - 2*omegaVar*m_xx_I - 6*omegaVar*m_xz_I + omegaVar*m_yy_I - 2*omegaVar*m_zz_I + 15));
    m_yz_t90 = (m_yz_I*(3*omegaVar + 172))/(12*(2*m_xx_I + 6*m_xz_I - m_yy_I + 2*m_zz_I - 2*omegaVar*m_xx_I - 6*omegaVar*m_xz_I + omegaVar*m_yy_I - 2*omegaVar*m_zz_I + 15));
    m_zz_t45 = (144*m_xx_I + 432*m_xz_I - 72*m_yy_I + 1692*m_zz_I - 4*omegaVar*m_xx_I - 12*omegaVar*m_xz_I + 2*omegaVar*m_yy_I + 23*omegaVar*m_zz_I + 48)/(108*(2*m_xx_I + 6*m_xz_I - m_yy_I + 2*m_zz_I - 2*omegaVar*m_xx_I - 6*omegaVar*m_xz_I + omegaVar*m_yy_I - 2*omegaVar*m_zz_I + 15));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = (m_yy_t45/12 - m_xz_t90/2 - m_xx_t45/6 - m_zz_t45/6 + (omegaVar*m_xx_t45)/12 + (omegaVar*m_xz_t90)/4 - (omegaVar*m_yy_t45)/24 + (omegaVar*m_zz_t45)/12 - 1.0/18.0)*rho + RHO_0/18;
    L_BC_Fy = 0;
    L_BC_Fz = (m_yy_t45/12 - m_xz_t90/2 - m_xx_t45/6 - m_zz_t45/6 + (omegaVar*m_xx_t45)/12 + (omegaVar*m_xz_t90)/4 - (omegaVar*m_yy_t45)/24 + (omegaVar*m_zz_t45)/12 - 1.0/18.0)*rho + RHO_0/18;
    #endif //_BC_FORCES 

    break;
case 0b0000010110000110:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = rho_I;

    m_xx_t45 = m_xx_I;
    m_xy_t90 = m_xy_I;
    m_xz_t90 = m_xz_I;
    m_yy_t45 = m_yy_I;
    m_yz_t90 = m_yz_I;
    m_zz_t45 = m_zz_I;

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = 0;
    L_BC_Fy = 0;
    L_BC_Fz = 0;
    #endif //_BC_FORCES 

    break;
case 0b0000010110000111:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[10] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (6*(15*rho_I + m_xx_I*rho_I + 6*m_xz_I*rho_I + m_yy_I*rho_I + 6*m_yz_I*rho_I + 4*m_zz_I*rho_I - omegaVar*m_xx_I*rho_I - 6*omegaVar*m_xz_I*rho_I - omegaVar*m_yy_I*rho_I - 6*omegaVar*m_yz_I*rho_I - 4*omegaVar*m_zz_I*rho_I))/(3*omegaVar + 82);

    m_xx_t45 = (2052*m_xx_I + 504*m_xz_I - 162*m_yy_I - 234*m_yz_I + 90*m_zz_I + 73*omegaVar*m_xx_I + 6*omegaVar*m_xz_I - 8*omegaVar*m_yy_I - 21*omegaVar*m_yz_I - 5*omegaVar*m_zz_I + 30)/(135*(m_xx_I + 6*m_xz_I + m_yy_I + 6*m_yz_I + 4*m_zz_I - omegaVar*m_xx_I - 6*omegaVar*m_xz_I - omegaVar*m_yy_I - 6*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 15));
    m_xy_t90 = (m_xy_I*(3*omegaVar + 82))/(6*(m_xx_I + 6*m_xz_I + m_yy_I + 6*m_yz_I + 4*m_zz_I - omegaVar*m_xx_I - 6*omegaVar*m_xz_I - omegaVar*m_yy_I - 6*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 15));
    m_xz_t90 = (168*m_xx_I + 1746*m_xz_I - 78*m_yy_I + 24*m_yz_I + 180*m_zz_I + 2*omegaVar*m_xx_I + 39*omegaVar*m_xz_I - 7*omegaVar*m_yy_I - 24*omegaVar*m_yz_I - 10*omegaVar*m_zz_I + 60)/(90*(m_xx_I + 6*m_xz_I + m_yy_I + 6*m_yz_I + 4*m_zz_I - omegaVar*m_xx_I - 6*omegaVar*m_xz_I - omegaVar*m_yy_I - 6*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 15));
    m_yy_t45 = (2052*m_yy_I - 234*m_xz_I - 162*m_xx_I + 504*m_yz_I + 90*m_zz_I - 8*omegaVar*m_xx_I - 21*omegaVar*m_xz_I + 73*omegaVar*m_yy_I + 6*omegaVar*m_yz_I - 5*omegaVar*m_zz_I + 30)/(135*(m_xx_I + 6*m_xz_I + m_yy_I + 6*m_yz_I + 4*m_zz_I - omegaVar*m_xx_I - 6*omegaVar*m_xz_I - omegaVar*m_yy_I - 6*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 15));
    m_yz_t90 = (24*m_xz_I - 78*m_xx_I + 168*m_yy_I + 1746*m_yz_I + 180*m_zz_I - 7*omegaVar*m_xx_I - 24*omegaVar*m_xz_I + 2*omegaVar*m_yy_I + 39*omegaVar*m_yz_I - 10*omegaVar*m_zz_I + 60)/(90*(m_xx_I + 6*m_xz_I + m_yy_I + 6*m_yz_I + 4*m_zz_I - omegaVar*m_xx_I - 6*omegaVar*m_xz_I - omegaVar*m_yy_I - 6*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 15));
    m_zz_t45 = (36*m_xx_I + 216*m_xz_I + 36*m_yy_I + 216*m_yz_I + 882*m_zz_I - 2*omegaVar*m_xx_I - 12*omegaVar*m_xz_I - 2*omegaVar*m_yy_I - 12*omegaVar*m_yz_I + 19*omegaVar*m_zz_I + 48)/(54*(m_xx_I + 6*m_xz_I + m_yy_I + 6*m_yz_I + 4*m_zz_I - omegaVar*m_xx_I - 6*omegaVar*m_xz_I - omegaVar*m_yy_I - 6*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 15));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = (m_yy_t45/12 - m_xz_t90/2 - m_xx_t45/6 - m_zz_t45/6 + (omegaVar*m_xx_t45)/12 + (omegaVar*m_xz_t90)/4 - (omegaVar*m_yy_t45)/24 + (omegaVar*m_zz_t45)/12 - 1.0/18.0)*rho + RHO_0/18;
    L_BC_Fy = (m_xx_t45/12 - m_yy_t45/6 - m_yz_t90/2 - m_zz_t45/6 - (omegaVar*m_xx_t45)/24 + (omegaVar*m_yy_t45)/12 + (omegaVar*m_yz_t90)/4 + (omegaVar*m_zz_t45)/12 - 1.0/18.0)*rho + RHO_0/18;
    L_BC_Fz = ((omegaVar*m_xx_t45)/24 - m_xz_t90/2 - m_yy_t45/12 - m_yz_t90/2 - m_zz_t45/3 - m_xx_t45/12 + (omegaVar*m_xz_t90)/4 + (omegaVar*m_yy_t45)/24 + (omegaVar*m_yz_t90)/4 + (omegaVar*m_zz_t45)/6 - 1.0/9.0)*rho + RHO_0/9;
    #endif //_BC_FORCES 

    break;
case 0b0000010110001000:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (12*(15*rho_I + 2*m_xx_I*rho_I + 6*m_xy_I*rho_I + 2*m_yy_I*rho_I - m_zz_I*rho_I - 2*omegaVar*m_xx_I*rho_I - 6*omegaVar*m_xy_I*rho_I - 2*omegaVar*m_yy_I*rho_I + omegaVar*m_zz_I*rho_I))/(3*omegaVar + 172);

    m_xx_t45 = (1692*m_xx_I + 432*m_xy_I + 144*m_yy_I - 72*m_zz_I + 23*omegaVar*m_xx_I - 12*omegaVar*m_xy_I - 4*omegaVar*m_yy_I + 2*omegaVar*m_zz_I + 48)/(108*(2*m_xx_I + 6*m_xy_I + 2*m_yy_I - m_zz_I - 2*omegaVar*m_xx_I - 6*omegaVar*m_xy_I - 2*omegaVar*m_yy_I + omegaVar*m_zz_I + 15));
    m_xy_t90 = (72*m_xx_I + 732*m_xy_I + 72*m_yy_I - 36*m_zz_I - 2*omegaVar*m_xx_I + 3*omegaVar*m_xy_I - 2*omegaVar*m_yy_I + omegaVar*m_zz_I + 24)/(36*(2*m_xx_I + 6*m_xy_I + 2*m_yy_I - m_zz_I - 2*omegaVar*m_xx_I - 6*omegaVar*m_xy_I - 2*omegaVar*m_yy_I + omegaVar*m_zz_I + 15));
    m_xz_t90 = (m_xz_I*(3*omegaVar + 172))/(12*(2*m_xx_I + 6*m_xy_I + 2*m_yy_I - m_zz_I - 2*omegaVar*m_xx_I - 6*omegaVar*m_xy_I - 2*omegaVar*m_yy_I + omegaVar*m_zz_I + 15));
    m_yy_t45 = (144*m_xx_I + 432*m_xy_I + 1692*m_yy_I - 72*m_zz_I - 4*omegaVar*m_xx_I - 12*omegaVar*m_xy_I + 23*omegaVar*m_yy_I + 2*omegaVar*m_zz_I + 48)/(108*(2*m_xx_I + 6*m_xy_I + 2*m_yy_I - m_zz_I - 2*omegaVar*m_xx_I - 6*omegaVar*m_xy_I - 2*omegaVar*m_yy_I + omegaVar*m_zz_I + 15));
    m_yz_t90 = (m_yz_I*(3*omegaVar + 172))/(12*(2*m_xx_I + 6*m_xy_I + 2*m_yy_I - m_zz_I - 2*omegaVar*m_xx_I - 6*omegaVar*m_xy_I - 2*omegaVar*m_yy_I + omegaVar*m_zz_I + 15));
    m_zz_t45 = (792*m_zz_I - 108*m_xy_I - 36*m_yy_I - 36*m_xx_I + omegaVar*m_xx_I + 3*omegaVar*m_xy_I + omegaVar*m_yy_I + 13*omegaVar*m_zz_I - 12)/(54*(2*m_xx_I + 6*m_xy_I + 2*m_yy_I - m_zz_I - 2*omegaVar*m_xx_I - 6*omegaVar*m_xy_I - 2*omegaVar*m_yy_I + omegaVar*m_zz_I + 15));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = (m_xx_t45/6 + m_xy_t90/2 + m_yy_t45/6 - m_zz_t45/12 - (omegaVar*m_xx_t45)/12 - (omegaVar*m_xy_t90)/4 - (omegaVar*m_yy_t45)/12 + (omegaVar*m_zz_t45)/24 + 1.0/18.0)*rho - RHO_0/18;
    L_BC_Fy = (m_xx_t45/6 + m_xy_t90/2 + m_yy_t45/6 - m_zz_t45/12 - (omegaVar*m_xx_t45)/12 - (omegaVar*m_xy_t90)/4 - (omegaVar*m_yy_t45)/12 + (omegaVar*m_zz_t45)/24 + 1.0/18.0)*rho - RHO_0/18;
    L_BC_Fz = 0;
    #endif //_BC_FORCES 

    break;
case 0b0000010110001001:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (12*(15*rho_I + 2*m_xx_I*rho_I + 6*m_xy_I*rho_I + 2*m_yy_I*rho_I - m_zz_I*rho_I - 2*omegaVar*m_xx_I*rho_I - 6*omegaVar*m_xy_I*rho_I - 2*omegaVar*m_yy_I*rho_I + omegaVar*m_zz_I*rho_I))/(3*omegaVar + 172);

    m_xx_t45 = (1692*m_xx_I + 432*m_xy_I + 144*m_yy_I - 72*m_zz_I + 23*omegaVar*m_xx_I - 12*omegaVar*m_xy_I - 4*omegaVar*m_yy_I + 2*omegaVar*m_zz_I + 48)/(108*(2*m_xx_I + 6*m_xy_I + 2*m_yy_I - m_zz_I - 2*omegaVar*m_xx_I - 6*omegaVar*m_xy_I - 2*omegaVar*m_yy_I + omegaVar*m_zz_I + 15));
    m_xy_t90 = (72*m_xx_I + 732*m_xy_I + 72*m_yy_I - 36*m_zz_I - 2*omegaVar*m_xx_I + 3*omegaVar*m_xy_I - 2*omegaVar*m_yy_I + omegaVar*m_zz_I + 24)/(36*(2*m_xx_I + 6*m_xy_I + 2*m_yy_I - m_zz_I - 2*omegaVar*m_xx_I - 6*omegaVar*m_xy_I - 2*omegaVar*m_yy_I + omegaVar*m_zz_I + 15));
    m_xz_t90 = (m_xz_I*(3*omegaVar + 172))/(12*(2*m_xx_I + 6*m_xy_I + 2*m_yy_I - m_zz_I - 2*omegaVar*m_xx_I - 6*omegaVar*m_xy_I - 2*omegaVar*m_yy_I + omegaVar*m_zz_I + 15));
    m_yy_t45 = (144*m_xx_I + 432*m_xy_I + 1692*m_yy_I - 72*m_zz_I - 4*omegaVar*m_xx_I - 12*omegaVar*m_xy_I + 23*omegaVar*m_yy_I + 2*omegaVar*m_zz_I + 48)/(108*(2*m_xx_I + 6*m_xy_I + 2*m_yy_I - m_zz_I - 2*omegaVar*m_xx_I - 6*omegaVar*m_xy_I - 2*omegaVar*m_yy_I + omegaVar*m_zz_I + 15));
    m_yz_t90 = (m_yz_I*(3*omegaVar + 172))/(12*(2*m_xx_I + 6*m_xy_I + 2*m_yy_I - m_zz_I - 2*omegaVar*m_xx_I - 6*omegaVar*m_xy_I - 2*omegaVar*m_yy_I + omegaVar*m_zz_I + 15));
    m_zz_t45 = (792*m_zz_I - 108*m_xy_I - 36*m_yy_I - 36*m_xx_I + omegaVar*m_xx_I + 3*omegaVar*m_xy_I + omegaVar*m_yy_I + 13*omegaVar*m_zz_I - 12)/(54*(2*m_xx_I + 6*m_xy_I + 2*m_yy_I - m_zz_I - 2*omegaVar*m_xx_I - 6*omegaVar*m_xy_I - 2*omegaVar*m_yy_I + omegaVar*m_zz_I + 15));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = (m_xx_t45/6 + m_xy_t90/2 + m_yy_t45/6 - m_zz_t45/12 - (omegaVar*m_xx_t45)/12 - (omegaVar*m_xy_t90)/4 - (omegaVar*m_yy_t45)/12 + (omegaVar*m_zz_t45)/24 + 1.0/18.0)*rho - RHO_0/18;
    L_BC_Fy = (m_xx_t45/6 + m_xy_t90/2 + m_yy_t45/6 - m_zz_t45/12 - (omegaVar*m_xx_t45)/12 - (omegaVar*m_xy_t90)/4 - (omegaVar*m_yy_t45)/12 + (omegaVar*m_zz_t45)/24 + 1.0/18.0)*rho - RHO_0/18;
    L_BC_Fz = 0;
    #endif //_BC_FORCES 

    break;
case 0b0000010110001010:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (6*(15*rho_I + 4*m_xx_I*rho_I + 6*m_xy_I*rho_I - 6*m_xz_I*rho_I + m_yy_I*rho_I + m_zz_I*rho_I - 4*omegaVar*m_xx_I*rho_I - 6*omegaVar*m_xy_I*rho_I + 6*omegaVar*m_xz_I*rho_I - omegaVar*m_yy_I*rho_I - omegaVar*m_zz_I*rho_I))/(3*omegaVar + 82);

    m_xx_t45 = (882*m_xx_I + 216*m_xy_I - 216*m_xz_I + 36*m_yy_I + 36*m_zz_I + 19*omegaVar*m_xx_I - 12*omegaVar*m_xy_I + 12*omegaVar*m_xz_I - 2*omegaVar*m_yy_I - 2*omegaVar*m_zz_I + 48)/(54*(4*m_xx_I + 6*m_xy_I - 6*m_xz_I + m_yy_I + m_zz_I - 4*omegaVar*m_xx_I - 6*omegaVar*m_xy_I + 6*omegaVar*m_xz_I - omegaVar*m_yy_I - omegaVar*m_zz_I + 15));
    m_xy_t90 = (180*m_xx_I + 1746*m_xy_I - 24*m_xz_I + 168*m_yy_I - 78*m_zz_I - 10*omegaVar*m_xx_I + 39*omegaVar*m_xy_I + 24*omegaVar*m_xz_I + 2*omegaVar*m_yy_I - 7*omegaVar*m_zz_I + 60)/(90*(4*m_xx_I + 6*m_xy_I - 6*m_xz_I + m_yy_I + m_zz_I - 4*omegaVar*m_xx_I - 6*omegaVar*m_xy_I + 6*omegaVar*m_xz_I - omegaVar*m_yy_I - omegaVar*m_zz_I + 15));
    m_xz_t90 = (1746*m_xz_I - 24*m_xy_I - 180*m_xx_I + 78*m_yy_I - 168*m_zz_I + 10*omegaVar*m_xx_I + 24*omegaVar*m_xy_I + 39*omegaVar*m_xz_I + 7*omegaVar*m_yy_I - 2*omegaVar*m_zz_I - 60)/(90*(4*m_xx_I + 6*m_xy_I - 6*m_xz_I + m_yy_I + m_zz_I - 4*omegaVar*m_xx_I - 6*omegaVar*m_xy_I + 6*omegaVar*m_xz_I - omegaVar*m_yy_I - omegaVar*m_zz_I + 15));
    m_yy_t45 = (90*m_xx_I + 504*m_xy_I + 234*m_xz_I + 2052*m_yy_I - 162*m_zz_I - 5*omegaVar*m_xx_I + 6*omegaVar*m_xy_I + 21*omegaVar*m_xz_I + 73*omegaVar*m_yy_I - 8*omegaVar*m_zz_I + 30)/(135*(4*m_xx_I + 6*m_xy_I - 6*m_xz_I + m_yy_I + m_zz_I - 4*omegaVar*m_xx_I - 6*omegaVar*m_xy_I + 6*omegaVar*m_xz_I - omegaVar*m_yy_I - omegaVar*m_zz_I + 15));
    m_yz_t90 = (m_yz_I*(3*omegaVar + 82))/(6*(4*m_xx_I + 6*m_xy_I - 6*m_xz_I + m_yy_I + m_zz_I - 4*omegaVar*m_xx_I - 6*omegaVar*m_xy_I + 6*omegaVar*m_xz_I - omegaVar*m_yy_I - omegaVar*m_zz_I + 15));
    m_zz_t45 = -(234*m_xy_I - 90*m_xx_I + 504*m_xz_I + 162*m_yy_I - 2052*m_zz_I + 5*omegaVar*m_xx_I + 21*omegaVar*m_xy_I + 6*omegaVar*m_xz_I + 8*omegaVar*m_yy_I - 73*omegaVar*m_zz_I - 30)/(135*(4*m_xx_I + 6*m_xy_I - 6*m_xz_I + m_yy_I + m_zz_I - 4*omegaVar*m_xx_I - 6*omegaVar*m_xy_I + 6*omegaVar*m_xz_I - omegaVar*m_yy_I - omegaVar*m_zz_I + 15));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = (m_xx_t45/3 + m_xy_t90/2 - m_xz_t90/2 + m_yy_t45/12 + m_zz_t45/12 - (omegaVar*m_xx_t45)/6 - (omegaVar*m_xy_t90)/4 + (omegaVar*m_xz_t90)/4 - (omegaVar*m_yy_t45)/24 - (omegaVar*m_zz_t45)/24 + 1.0/9.0)*rho - RHO_0/9;
    L_BC_Fy = (m_xx_t45/6 + m_xy_t90/2 + m_yy_t45/6 - m_zz_t45/12 - (omegaVar*m_xx_t45)/12 - (omegaVar*m_xy_t90)/4 - (omegaVar*m_yy_t45)/12 + (omegaVar*m_zz_t45)/24 + 1.0/18.0)*rho - RHO_0/18;
    L_BC_Fz = (m_xz_t90/2 - m_xx_t45/6 + m_yy_t45/12 - m_zz_t45/6 + (omegaVar*m_xx_t45)/12 - (omegaVar*m_xz_t90)/4 - (omegaVar*m_yy_t45)/24 + (omegaVar*m_zz_t45)/12 - 1.0/18.0)*rho + RHO_0/18;
    #endif //_BC_FORCES 

    break;
case 0b0000010110001011:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[9] + pop[10] + pop[12] + pop[13] + pop[14] + pop[15] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (12*(5*rho_I + m_xx_I*rho_I + 2*m_xy_I*rho_I - 2*m_xz_I*rho_I + m_yy_I*rho_I + 2*m_yz_I*rho_I + m_zz_I*rho_I - omegaVar*m_xx_I*rho_I - 2*omegaVar*m_xy_I*rho_I + 2*omegaVar*m_xz_I*rho_I - omegaVar*m_yy_I*rho_I - 2*omegaVar*m_yz_I*rho_I - omegaVar*m_zz_I*rho_I))/(3*omegaVar + 52);

    m_xx_t45 = (472*m_xx_I + 112*m_xy_I - 112*m_xz_I + 4*m_yy_I - 44*m_yz_I + 4*m_zz_I + 23*omegaVar*m_xx_I - 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - 4*omegaVar*m_yy_I - 11*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 20)/(90*(m_xx_I + 2*m_xy_I - 2*m_xz_I + m_yy_I + 2*m_yz_I + m_zz_I - omegaVar*m_xx_I - 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - omegaVar*m_yy_I - 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_xy_t90 = (112*m_xx_I + 1108*m_xy_I - 16*m_xz_I + 112*m_yy_I + 16*m_yz_I - 44*m_zz_I - 2*omegaVar*m_xx_I + 47*omegaVar*m_xy_I + 16*omegaVar*m_xz_I - 2*omegaVar*m_yy_I - 16*omegaVar*m_yz_I - 11*omegaVar*m_zz_I + 40)/(180*(m_xx_I + 2*m_xy_I - 2*m_xz_I + m_yy_I + 2*m_yz_I + m_zz_I - omegaVar*m_xx_I - 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - omegaVar*m_yy_I - 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_xz_t90 = (1108*m_xz_I - 16*m_xy_I - 112*m_xx_I + 44*m_yy_I - 16*m_yz_I - 112*m_zz_I + 2*omegaVar*m_xx_I + 16*omegaVar*m_xy_I + 47*omegaVar*m_xz_I + 11*omegaVar*m_yy_I + 16*omegaVar*m_yz_I + 2*omegaVar*m_zz_I - 40)/(180*(m_xx_I + 2*m_xy_I - 2*m_xz_I + m_yy_I + 2*m_yz_I + m_zz_I - omegaVar*m_xx_I - 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - omegaVar*m_yy_I - 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_yy_t45 = (4*m_xx_I + 112*m_xy_I + 44*m_xz_I + 472*m_yy_I + 112*m_yz_I + 4*m_zz_I - 4*omegaVar*m_xx_I - 2*omegaVar*m_xy_I + 11*omegaVar*m_xz_I + 23*omegaVar*m_yy_I - 2*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 20)/(90*(m_xx_I + 2*m_xy_I - 2*m_xz_I + m_yy_I + 2*m_yz_I + m_zz_I - omegaVar*m_xx_I - 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - omegaVar*m_yy_I - 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_yz_t90 = (16*m_xy_I - 44*m_xx_I - 16*m_xz_I + 112*m_yy_I + 1108*m_yz_I + 112*m_zz_I - 11*omegaVar*m_xx_I - 16*omegaVar*m_xy_I + 16*omegaVar*m_xz_I - 2*omegaVar*m_yy_I + 47*omegaVar*m_yz_I - 2*omegaVar*m_zz_I + 40)/(180*(m_xx_I + 2*m_xy_I - 2*m_xz_I + m_yy_I + 2*m_yz_I + m_zz_I - omegaVar*m_xx_I - 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - omegaVar*m_yy_I - 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_zz_t45 = (4*m_xx_I - 44*m_xy_I - 112*m_xz_I + 4*m_yy_I + 112*m_yz_I + 472*m_zz_I - 4*omegaVar*m_xx_I - 11*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - 4*omegaVar*m_yy_I - 2*omegaVar*m_yz_I + 23*omegaVar*m_zz_I + 20)/(90*(m_xx_I + 2*m_xy_I - 2*m_xz_I + m_yy_I + 2*m_yz_I + m_zz_I - omegaVar*m_xx_I - 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - omegaVar*m_yy_I - 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = (m_xx_t45/3 + m_xy_t90/2 - m_xz_t90/2 + m_yy_t45/12 + m_zz_t45/12 - (omegaVar*m_xx_t45)/6 - (omegaVar*m_xy_t90)/4 + (omegaVar*m_xz_t90)/4 - (omegaVar*m_yy_t45)/24 - (omegaVar*m_zz_t45)/24 + 1.0/9.0)*rho - RHO_0/9;
    L_BC_Fy = -(rho*(omegaVar - 2)*(m_xx_t45 + 2*m_xy_t90 - 2*m_yz_t90 - m_zz_t45))/8;
    L_BC_Fz = (m_xz_t90/2 - m_xx_t45/12 - m_yy_t45/12 - m_yz_t90/2 - m_zz_t45/3 + (omegaVar*m_xx_t45)/24 - (omegaVar*m_xz_t90)/4 + (omegaVar*m_yy_t45)/24 + (omegaVar*m_yz_t90)/4 + (omegaVar*m_zz_t45)/6 - 1.0/9.0)*rho + RHO_0/9;
    #endif //_BC_FORCES 

    break;
case 0b0000010110001100:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17]);
    m_xy_I = inv_rho_I * (pop[7] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17]);

    rho = (6*(15*rho_I + m_xx_I*rho_I + 6*m_xy_I*rho_I + 4*m_yy_I*rho_I - 6*m_yz_I*rho_I + m_zz_I*rho_I - omegaVar*m_xx_I*rho_I - 6*omegaVar*m_xy_I*rho_I - 4*omegaVar*m_yy_I*rho_I + 6*omegaVar*m_yz_I*rho_I - omegaVar*m_zz_I*rho_I))/(3*omegaVar + 82);

    m_xx_t45 = (2052*m_xx_I + 504*m_xy_I + 90*m_yy_I + 234*m_yz_I - 162*m_zz_I + 73*omegaVar*m_xx_I + 6*omegaVar*m_xy_I - 5*omegaVar*m_yy_I + 21*omegaVar*m_yz_I - 8*omegaVar*m_zz_I + 30)/(135*(m_xx_I + 6*m_xy_I + 4*m_yy_I - 6*m_yz_I + m_zz_I - omegaVar*m_xx_I - 6*omegaVar*m_xy_I - 4*omegaVar*m_yy_I + 6*omegaVar*m_yz_I - omegaVar*m_zz_I + 15));
    m_xy_t90 = (168*m_xx_I + 1746*m_xy_I + 180*m_yy_I - 24*m_yz_I - 78*m_zz_I + 2*omegaVar*m_xx_I + 39*omegaVar*m_xy_I - 10*omegaVar*m_yy_I + 24*omegaVar*m_yz_I - 7*omegaVar*m_zz_I + 60)/(90*(m_xx_I + 6*m_xy_I + 4*m_yy_I - 6*m_yz_I + m_zz_I - omegaVar*m_xx_I - 6*omegaVar*m_xy_I - 4*omegaVar*m_yy_I + 6*omegaVar*m_yz_I - omegaVar*m_zz_I + 15));
    m_xz_t90 = (m_xz_I*(3*omegaVar + 82))/(6*(m_xx_I + 6*m_xy_I + 4*m_yy_I - 6*m_yz_I + m_zz_I - omegaVar*m_xx_I - 6*omegaVar*m_xy_I - 4*omegaVar*m_yy_I + 6*omegaVar*m_yz_I - omegaVar*m_zz_I + 15));
    m_yy_t45 = (36*m_xx_I + 216*m_xy_I + 882*m_yy_I - 216*m_yz_I + 36*m_zz_I - 2*omegaVar*m_xx_I - 12*omegaVar*m_xy_I + 19*omegaVar*m_yy_I + 12*omegaVar*m_yz_I - 2*omegaVar*m_zz_I + 48)/(54*(m_xx_I + 6*m_xy_I + 4*m_yy_I - 6*m_yz_I + m_zz_I - omegaVar*m_xx_I - 6*omegaVar*m_xy_I - 4*omegaVar*m_yy_I + 6*omegaVar*m_yz_I - omegaVar*m_zz_I + 15));
    m_yz_t90 = (78*m_xx_I - 24*m_xy_I - 180*m_yy_I + 1746*m_yz_I - 168*m_zz_I + 7*omegaVar*m_xx_I + 24*omegaVar*m_xy_I + 10*omegaVar*m_yy_I + 39*omegaVar*m_yz_I - 2*omegaVar*m_zz_I - 60)/(90*(m_xx_I + 6*m_xy_I + 4*m_yy_I - 6*m_yz_I + m_zz_I - omegaVar*m_xx_I - 6*omegaVar*m_xy_I - 4*omegaVar*m_yy_I + 6*omegaVar*m_yz_I - omegaVar*m_zz_I + 15));
    m_zz_t45 = -(162*m_xx_I + 234*m_xy_I - 90*m_yy_I + 504*m_yz_I - 2052*m_zz_I + 8*omegaVar*m_xx_I + 21*omegaVar*m_xy_I + 5*omegaVar*m_yy_I + 6*omegaVar*m_yz_I - 73*omegaVar*m_zz_I - 30)/(135*(m_xx_I + 6*m_xy_I + 4*m_yy_I - 6*m_yz_I + m_zz_I - omegaVar*m_xx_I - 6*omegaVar*m_xy_I - 4*omegaVar*m_yy_I + 6*omegaVar*m_yz_I - omegaVar*m_zz_I + 15));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = (m_xx_t45/6 + m_xy_t90/2 + m_yy_t45/6 - m_zz_t45/12 - (omegaVar*m_xx_t45)/12 - (omegaVar*m_xy_t90)/4 - (omegaVar*m_yy_t45)/12 + (omegaVar*m_zz_t45)/24 + 1.0/18.0)*rho - RHO_0/18;
    L_BC_Fy = (m_xx_t45/12 + m_xy_t90/2 + m_yy_t45/3 - m_yz_t90/2 + m_zz_t45/12 - (omegaVar*m_xx_t45)/24 - (omegaVar*m_xy_t90)/4 - (omegaVar*m_yy_t45)/6 + (omegaVar*m_yz_t90)/4 - (omegaVar*m_zz_t45)/24 + 1.0/9.0)*rho - RHO_0/9;
    L_BC_Fz = (m_xx_t45/12 - m_yy_t45/6 + m_yz_t90/2 - m_zz_t45/6 - (omegaVar*m_xx_t45)/24 + (omegaVar*m_yy_t45)/12 - (omegaVar*m_yz_t90)/4 + (omegaVar*m_zz_t45)/12 - 1.0/18.0)*rho + RHO_0/18;
    #endif //_BC_FORCES 

    break;
case 0b0000010110001101:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17]);
    m_xy_I = inv_rho_I * (pop[7] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17]);

    rho = (12*(5*rho_I + m_xx_I*rho_I + 2*m_xy_I*rho_I + 2*m_xz_I*rho_I + m_yy_I*rho_I - 2*m_yz_I*rho_I + m_zz_I*rho_I - omegaVar*m_xx_I*rho_I - 2*omegaVar*m_xy_I*rho_I - 2*omegaVar*m_xz_I*rho_I - omegaVar*m_yy_I*rho_I + 2*omegaVar*m_yz_I*rho_I - omegaVar*m_zz_I*rho_I))/(3*omegaVar + 52);

    m_xx_t45 = (472*m_xx_I + 112*m_xy_I + 112*m_xz_I + 4*m_yy_I + 44*m_yz_I + 4*m_zz_I + 23*omegaVar*m_xx_I - 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - 4*omegaVar*m_yy_I + 11*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 20)/(90*(m_xx_I + 2*m_xy_I + 2*m_xz_I + m_yy_I - 2*m_yz_I + m_zz_I - omegaVar*m_xx_I - 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - omegaVar*m_yy_I + 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_xy_t90 = (112*m_xx_I + 1108*m_xy_I + 16*m_xz_I + 112*m_yy_I - 16*m_yz_I - 44*m_zz_I - 2*omegaVar*m_xx_I + 47*omegaVar*m_xy_I - 16*omegaVar*m_xz_I - 2*omegaVar*m_yy_I + 16*omegaVar*m_yz_I - 11*omegaVar*m_zz_I + 40)/(180*(m_xx_I + 2*m_xy_I + 2*m_xz_I + m_yy_I - 2*m_yz_I + m_zz_I - omegaVar*m_xx_I - 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - omegaVar*m_yy_I + 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_xz_t90 = (112*m_xx_I + 16*m_xy_I + 1108*m_xz_I - 44*m_yy_I - 16*m_yz_I + 112*m_zz_I - 2*omegaVar*m_xx_I - 16*omegaVar*m_xy_I + 47*omegaVar*m_xz_I - 11*omegaVar*m_yy_I + 16*omegaVar*m_yz_I - 2*omegaVar*m_zz_I + 40)/(180*(m_xx_I + 2*m_xy_I + 2*m_xz_I + m_yy_I - 2*m_yz_I + m_zz_I - omegaVar*m_xx_I - 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - omegaVar*m_yy_I + 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_yy_t45 = (4*m_xx_I + 112*m_xy_I - 44*m_xz_I + 472*m_yy_I - 112*m_yz_I + 4*m_zz_I - 4*omegaVar*m_xx_I - 2*omegaVar*m_xy_I - 11*omegaVar*m_xz_I + 23*omegaVar*m_yy_I + 2*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 20)/(90*(m_xx_I + 2*m_xy_I + 2*m_xz_I + m_yy_I - 2*m_yz_I + m_zz_I - omegaVar*m_xx_I - 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - omegaVar*m_yy_I + 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_yz_t90 = (44*m_xx_I - 16*m_xy_I - 16*m_xz_I - 112*m_yy_I + 1108*m_yz_I - 112*m_zz_I + 11*omegaVar*m_xx_I + 16*omegaVar*m_xy_I + 16*omegaVar*m_xz_I + 2*omegaVar*m_yy_I + 47*omegaVar*m_yz_I + 2*omegaVar*m_zz_I - 40)/(180*(m_xx_I + 2*m_xy_I + 2*m_xz_I + m_yy_I - 2*m_yz_I + m_zz_I - omegaVar*m_xx_I - 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - omegaVar*m_yy_I + 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_zz_t45 = (4*m_xx_I - 44*m_xy_I + 112*m_xz_I + 4*m_yy_I - 112*m_yz_I + 472*m_zz_I - 4*omegaVar*m_xx_I - 11*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - 4*omegaVar*m_yy_I + 2*omegaVar*m_yz_I + 23*omegaVar*m_zz_I + 20)/(90*(m_xx_I + 2*m_xy_I + 2*m_xz_I + m_yy_I - 2*m_yz_I + m_zz_I - omegaVar*m_xx_I - 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - omegaVar*m_yy_I + 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = -(rho*(omegaVar - 2)*(2*m_xy_t90 - 2*m_xz_t90 + m_yy_t45 - m_zz_t45))/8;
    L_BC_Fy = (m_xx_t45/12 + m_xy_t90/2 + m_yy_t45/3 - m_yz_t90/2 + m_zz_t45/12 - (omegaVar*m_xx_t45)/24 - (omegaVar*m_xy_t90)/4 - (omegaVar*m_yy_t45)/6 + (omegaVar*m_yz_t90)/4 - (omegaVar*m_zz_t45)/24 + 1.0/9.0)*rho - RHO_0/9;
    L_BC_Fz = (m_yz_t90/2 - m_xz_t90/2 - m_yy_t45/12 - m_xx_t45/12 - m_zz_t45/3 + (omegaVar*m_xx_t45)/24 + (omegaVar*m_xz_t90)/4 + (omegaVar*m_yy_t45)/24 - (omegaVar*m_yz_t90)/4 + (omegaVar*m_zz_t45)/6 - 1.0/9.0)*rho + RHO_0/9;
    #endif //_BC_FORCES 

    break;
case 0b0000010110001110:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[17];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] - (1.0/3.0)*pop[17]);
    m_xy_I = inv_rho_I * (pop[7] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] + (2.0/3.0)*pop[17]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[17]);

    rho = (12*(5*rho_I + m_xx_I*rho_I + 2*m_xy_I*rho_I - 2*m_xz_I*rho_I + m_yy_I*rho_I - 2*m_yz_I*rho_I + m_zz_I*rho_I - omegaVar*m_xx_I*rho_I - 2*omegaVar*m_xy_I*rho_I + 2*omegaVar*m_xz_I*rho_I - omegaVar*m_yy_I*rho_I + 2*omegaVar*m_yz_I*rho_I - omegaVar*m_zz_I*rho_I))/(3*omegaVar + 52);

    m_xx_t45 = (472*m_xx_I + 112*m_xy_I - 112*m_xz_I + 4*m_yy_I + 44*m_yz_I + 4*m_zz_I + 23*omegaVar*m_xx_I - 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - 4*omegaVar*m_yy_I + 11*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 20)/(90*(m_xx_I + 2*m_xy_I - 2*m_xz_I + m_yy_I - 2*m_yz_I + m_zz_I - omegaVar*m_xx_I - 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - omegaVar*m_yy_I + 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_xy_t90 = (112*m_xx_I + 1108*m_xy_I - 16*m_xz_I + 112*m_yy_I - 16*m_yz_I - 44*m_zz_I - 2*omegaVar*m_xx_I + 47*omegaVar*m_xy_I + 16*omegaVar*m_xz_I - 2*omegaVar*m_yy_I + 16*omegaVar*m_yz_I - 11*omegaVar*m_zz_I + 40)/(180*(m_xx_I + 2*m_xy_I - 2*m_xz_I + m_yy_I - 2*m_yz_I + m_zz_I - omegaVar*m_xx_I - 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - omegaVar*m_yy_I + 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_xz_t90 = (1108*m_xz_I - 16*m_xy_I - 112*m_xx_I + 44*m_yy_I + 16*m_yz_I - 112*m_zz_I + 2*omegaVar*m_xx_I + 16*omegaVar*m_xy_I + 47*omegaVar*m_xz_I + 11*omegaVar*m_yy_I - 16*omegaVar*m_yz_I + 2*omegaVar*m_zz_I - 40)/(180*(m_xx_I + 2*m_xy_I - 2*m_xz_I + m_yy_I - 2*m_yz_I + m_zz_I - omegaVar*m_xx_I - 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - omegaVar*m_yy_I + 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_yy_t45 = (4*m_xx_I + 112*m_xy_I + 44*m_xz_I + 472*m_yy_I - 112*m_yz_I + 4*m_zz_I - 4*omegaVar*m_xx_I - 2*omegaVar*m_xy_I + 11*omegaVar*m_xz_I + 23*omegaVar*m_yy_I + 2*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 20)/(90*(m_xx_I + 2*m_xy_I - 2*m_xz_I + m_yy_I - 2*m_yz_I + m_zz_I - omegaVar*m_xx_I - 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - omegaVar*m_yy_I + 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_yz_t90 = (44*m_xx_I - 16*m_xy_I + 16*m_xz_I - 112*m_yy_I + 1108*m_yz_I - 112*m_zz_I + 11*omegaVar*m_xx_I + 16*omegaVar*m_xy_I - 16*omegaVar*m_xz_I + 2*omegaVar*m_yy_I + 47*omegaVar*m_yz_I + 2*omegaVar*m_zz_I - 40)/(180*(m_xx_I + 2*m_xy_I - 2*m_xz_I + m_yy_I - 2*m_yz_I + m_zz_I - omegaVar*m_xx_I - 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - omegaVar*m_yy_I + 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_zz_t45 = (4*m_xx_I - 44*m_xy_I - 112*m_xz_I + 4*m_yy_I - 112*m_yz_I + 472*m_zz_I - 4*omegaVar*m_xx_I - 11*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - 4*omegaVar*m_yy_I + 2*omegaVar*m_yz_I + 23*omegaVar*m_zz_I + 20)/(90*(m_xx_I + 2*m_xy_I - 2*m_xz_I + m_yy_I - 2*m_yz_I + m_zz_I - omegaVar*m_xx_I - 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - omegaVar*m_yy_I + 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = (m_xx_t45/3 + m_xy_t90/2 - m_xz_t90/2 + m_yy_t45/12 + m_zz_t45/12 - (omegaVar*m_xx_t45)/6 - (omegaVar*m_xy_t90)/4 + (omegaVar*m_xz_t90)/4 - (omegaVar*m_yy_t45)/24 - (omegaVar*m_zz_t45)/24 + 1.0/9.0)*rho - RHO_0/9;
    L_BC_Fy = (m_xx_t45/12 + m_xy_t90/2 + m_yy_t45/3 - m_yz_t90/2 + m_zz_t45/12 - (omegaVar*m_xx_t45)/24 - (omegaVar*m_xy_t90)/4 - (omegaVar*m_yy_t45)/6 + (omegaVar*m_yz_t90)/4 - (omegaVar*m_zz_t45)/24 + 1.0/9.0)*rho - RHO_0/9;
    L_BC_Fz = (m_xz_t90/2 - m_xx_t45/12 - m_yy_t45/12 + m_yz_t90/2 - m_zz_t45/3 + (omegaVar*m_xx_t45)/24 - (omegaVar*m_xz_t90)/4 + (omegaVar*m_yy_t45)/24 - (omegaVar*m_yz_t90)/4 + (omegaVar*m_zz_t45)/6 - 1.0/9.0)*rho + RHO_0/9;
    #endif //_BC_FORCES 

    break;
case 0b0000010110001111:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[6] + pop[7] + pop[10] + pop[12] + pop[13] + pop[14] + pop[15] + pop[17];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] - (1.0/3.0)*pop[17]);
    m_xy_I = inv_rho_I * (pop[7] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[10] - pop[15]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] + (2.0/3.0)*pop[17]);
    m_yz_I = inv_rho_I * (pop[12] - pop[17]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[17]);

    rho = (36*(107*rho_I + 12*m_xx_I*rho_I + 33*m_xy_I*rho_I + 12*m_yy_I*rho_I + 72*m_zz_I*rho_I - 12*omegaVar*m_xx_I*rho_I - 33*omegaVar*m_xy_I*rho_I - 12*omegaVar*m_yy_I*rho_I - 72*omegaVar*m_zz_I*rho_I))/(313*omegaVar + 2790);

    m_xx_t45 = (2772*m_xx_I + 648*m_xy_I - 18*m_yy_I - 108*m_zz_I + 302*omegaVar*m_xx_I + 48*omegaVar*m_xy_I - 11*omegaVar*m_yy_I - 66*omegaVar*m_zz_I + 72)/(27*(12*m_xx_I + 33*m_xy_I + 12*m_yy_I + 72*m_zz_I - 12*omegaVar*m_xx_I - 33*omegaVar*m_xy_I - 12*omegaVar*m_yy_I - 72*omegaVar*m_zz_I + 107));
    m_xy_t90 = (432*m_xx_I + 3978*m_xy_I + 432*m_yy_I - 198*m_zz_I + 32*omegaVar*m_xx_I + 401*omegaVar*m_xy_I + 32*omegaVar*m_yy_I - 121*omegaVar*m_zz_I + 132)/(36*(12*m_xx_I + 33*m_xy_I + 12*m_yy_I + 72*m_zz_I - 12*omegaVar*m_xx_I - 33*omegaVar*m_xy_I - 12*omegaVar*m_yy_I - 72*omegaVar*m_zz_I + 107));
    m_xz_t90 = (m_xz_I*(313*omegaVar + 2790))/(18*(12*m_xx_I + 33*m_xy_I + 12*m_yy_I + 72*m_zz_I - 12*omegaVar*m_xx_I - 33*omegaVar*m_xy_I - 12*omegaVar*m_yy_I - 72*omegaVar*m_zz_I + 107));
    m_yy_t45 = (648*m_xy_I - 18*m_xx_I + 2772*m_yy_I - 108*m_zz_I - 11*omegaVar*m_xx_I + 48*omegaVar*m_xy_I + 302*omegaVar*m_yy_I - 66*omegaVar*m_zz_I + 72)/(27*(12*m_xx_I + 33*m_xy_I + 12*m_yy_I + 72*m_zz_I - 12*omegaVar*m_xx_I - 33*omegaVar*m_xy_I - 12*omegaVar*m_yy_I - 72*omegaVar*m_zz_I + 107));
    m_yz_t90 = (m_yz_I*(313*omegaVar + 2790))/(18*(12*m_xx_I + 33*m_xy_I + 12*m_yy_I + 72*m_zz_I - 12*omegaVar*m_xx_I - 33*omegaVar*m_xy_I - 12*omegaVar*m_yy_I - 72*omegaVar*m_zz_I + 107));
    m_zz_t45 = -(72*m_xx_I + 198*m_xy_I + 72*m_yy_I - 2358*m_zz_I + 44*omegaVar*m_xx_I + 121*omegaVar*m_xy_I + 44*omegaVar*m_yy_I - 49*omegaVar*m_zz_I - 288)/(18*(12*m_xx_I + 33*m_xy_I + 12*m_yy_I + 72*m_zz_I - 12*omegaVar*m_xx_I - 33*omegaVar*m_xy_I - 12*omegaVar*m_yy_I - 72*omegaVar*m_zz_I + 107));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = (m_xx_t45/6 + m_xy_t90/2 - m_xz_t90 + m_yy_t45/6 - m_zz_t45/12 - (omegaVar*m_xx_t45)/12 - (omegaVar*m_xy_t90)/4 + (omegaVar*m_xz_t90)/2 - (omegaVar*m_yy_t45)/12 + (omegaVar*m_zz_t45)/24 + 1.0/18.0)*rho - RHO_0/18;
    L_BC_Fy = (m_xx_t45/6 + m_xy_t90/2 + m_yy_t45/6 - m_yz_t90 - m_zz_t45/12 - (omegaVar*m_xx_t45)/12 - (omegaVar*m_xy_t90)/4 - (omegaVar*m_yy_t45)/12 + (omegaVar*m_yz_t90)/2 + (omegaVar*m_zz_t45)/24 + 1.0/18.0)*rho - RHO_0/18;
    L_BC_Fz = ((omegaVar*m_zz_t45)/2 - m_zz_t45 - 1.0/3.0)*rho + RHO_0/3;
    #endif //_BC_FORCES 

    break;
case 0b0000010110010000:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = rho_I;

    m_xx_t45 = m_xx_I;
    m_xy_t90 = m_xy_I;
    m_xz_t90 = m_xz_I;
    m_yy_t45 = m_yy_I;
    m_yz_t90 = m_yz_I;
    m_zz_t45 = m_zz_I;

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = 0;
    L_BC_Fy = 0;
    L_BC_Fz = 0;
    #endif //_BC_FORCES 

    break;
case 0b0000010110010001:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (12*(15*rho_I + 2*m_xx_I*rho_I + 6*m_xy_I*rho_I + 2*m_yy_I*rho_I - m_zz_I*rho_I - 2*omegaVar*m_xx_I*rho_I - 6*omegaVar*m_xy_I*rho_I - 2*omegaVar*m_yy_I*rho_I + omegaVar*m_zz_I*rho_I))/(3*omegaVar + 172);

    m_xx_t45 = (1692*m_xx_I + 432*m_xy_I + 144*m_yy_I - 72*m_zz_I + 23*omegaVar*m_xx_I - 12*omegaVar*m_xy_I - 4*omegaVar*m_yy_I + 2*omegaVar*m_zz_I + 48)/(108*(2*m_xx_I + 6*m_xy_I + 2*m_yy_I - m_zz_I - 2*omegaVar*m_xx_I - 6*omegaVar*m_xy_I - 2*omegaVar*m_yy_I + omegaVar*m_zz_I + 15));
    m_xy_t90 = (72*m_xx_I + 732*m_xy_I + 72*m_yy_I - 36*m_zz_I - 2*omegaVar*m_xx_I + 3*omegaVar*m_xy_I - 2*omegaVar*m_yy_I + omegaVar*m_zz_I + 24)/(36*(2*m_xx_I + 6*m_xy_I + 2*m_yy_I - m_zz_I - 2*omegaVar*m_xx_I - 6*omegaVar*m_xy_I - 2*omegaVar*m_yy_I + omegaVar*m_zz_I + 15));
    m_xz_t90 = (m_xz_I*(3*omegaVar + 172))/(12*(2*m_xx_I + 6*m_xy_I + 2*m_yy_I - m_zz_I - 2*omegaVar*m_xx_I - 6*omegaVar*m_xy_I - 2*omegaVar*m_yy_I + omegaVar*m_zz_I + 15));
    m_yy_t45 = (144*m_xx_I + 432*m_xy_I + 1692*m_yy_I - 72*m_zz_I - 4*omegaVar*m_xx_I - 12*omegaVar*m_xy_I + 23*omegaVar*m_yy_I + 2*omegaVar*m_zz_I + 48)/(108*(2*m_xx_I + 6*m_xy_I + 2*m_yy_I - m_zz_I - 2*omegaVar*m_xx_I - 6*omegaVar*m_xy_I - 2*omegaVar*m_yy_I + omegaVar*m_zz_I + 15));
    m_yz_t90 = (m_yz_I*(3*omegaVar + 172))/(12*(2*m_xx_I + 6*m_xy_I + 2*m_yy_I - m_zz_I - 2*omegaVar*m_xx_I - 6*omegaVar*m_xy_I - 2*omegaVar*m_yy_I + omegaVar*m_zz_I + 15));
    m_zz_t45 = (792*m_zz_I - 108*m_xy_I - 36*m_yy_I - 36*m_xx_I + omegaVar*m_xx_I + 3*omegaVar*m_xy_I + omegaVar*m_yy_I + 13*omegaVar*m_zz_I - 12)/(54*(2*m_xx_I + 6*m_xy_I + 2*m_yy_I - m_zz_I - 2*omegaVar*m_xx_I - 6*omegaVar*m_xy_I - 2*omegaVar*m_yy_I + omegaVar*m_zz_I + 15));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = (m_zz_t45/12 - m_xy_t90/2 - m_yy_t45/6 - m_xx_t45/6 + (omegaVar*m_xx_t45)/12 + (omegaVar*m_xy_t90)/4 + (omegaVar*m_yy_t45)/12 - (omegaVar*m_zz_t45)/24 - 1.0/18.0)*rho + RHO_0/18;
    L_BC_Fy = (m_zz_t45/12 - m_xy_t90/2 - m_yy_t45/6 - m_xx_t45/6 + (omegaVar*m_xx_t45)/12 + (omegaVar*m_xy_t90)/4 + (omegaVar*m_yy_t45)/12 - (omegaVar*m_zz_t45)/24 - 1.0/18.0)*rho + RHO_0/18;
    L_BC_Fz = 0;
    #endif //_BC_FORCES 

    break;
case 0b0000010110010010:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = rho_I;

    m_xx_t45 = m_xx_I;
    m_xy_t90 = m_xy_I;
    m_xz_t90 = m_xz_I;
    m_yy_t45 = m_yy_I;
    m_yz_t90 = m_yz_I;
    m_zz_t45 = m_zz_I;

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = 0;
    L_BC_Fy = 0;
    L_BC_Fz = 0;
    #endif //_BC_FORCES 

    break;
case 0b0000010110010011:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[8] + pop[9] + pop[10] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (6*(15*rho_I + m_xx_I*rho_I + 6*m_xy_I*rho_I + 4*m_yy_I*rho_I + 6*m_yz_I*rho_I + m_zz_I*rho_I - omegaVar*m_xx_I*rho_I - 6*omegaVar*m_xy_I*rho_I - 4*omegaVar*m_yy_I*rho_I - 6*omegaVar*m_yz_I*rho_I - omegaVar*m_zz_I*rho_I))/(3*omegaVar + 82);

    m_xx_t45 = (2052*m_xx_I + 504*m_xy_I + 90*m_yy_I - 234*m_yz_I - 162*m_zz_I + 73*omegaVar*m_xx_I + 6*omegaVar*m_xy_I - 5*omegaVar*m_yy_I - 21*omegaVar*m_yz_I - 8*omegaVar*m_zz_I + 30)/(135*(m_xx_I + 6*m_xy_I + 4*m_yy_I + 6*m_yz_I + m_zz_I - omegaVar*m_xx_I - 6*omegaVar*m_xy_I - 4*omegaVar*m_yy_I - 6*omegaVar*m_yz_I - omegaVar*m_zz_I + 15));
    m_xy_t90 = (168*m_xx_I + 1746*m_xy_I + 180*m_yy_I + 24*m_yz_I - 78*m_zz_I + 2*omegaVar*m_xx_I + 39*omegaVar*m_xy_I - 10*omegaVar*m_yy_I - 24*omegaVar*m_yz_I - 7*omegaVar*m_zz_I + 60)/(90*(m_xx_I + 6*m_xy_I + 4*m_yy_I + 6*m_yz_I + m_zz_I - omegaVar*m_xx_I - 6*omegaVar*m_xy_I - 4*omegaVar*m_yy_I - 6*omegaVar*m_yz_I - omegaVar*m_zz_I + 15));
    m_xz_t90 = (m_xz_I*(3*omegaVar + 82))/(6*(m_xx_I + 6*m_xy_I + 4*m_yy_I + 6*m_yz_I + m_zz_I - omegaVar*m_xx_I - 6*omegaVar*m_xy_I - 4*omegaVar*m_yy_I - 6*omegaVar*m_yz_I - omegaVar*m_zz_I + 15));
    m_yy_t45 = (36*m_xx_I + 216*m_xy_I + 882*m_yy_I + 216*m_yz_I + 36*m_zz_I - 2*omegaVar*m_xx_I - 12*omegaVar*m_xy_I + 19*omegaVar*m_yy_I - 12*omegaVar*m_yz_I - 2*omegaVar*m_zz_I + 48)/(54*(m_xx_I + 6*m_xy_I + 4*m_yy_I + 6*m_yz_I + m_zz_I - omegaVar*m_xx_I - 6*omegaVar*m_xy_I - 4*omegaVar*m_yy_I - 6*omegaVar*m_yz_I - omegaVar*m_zz_I + 15));
    m_yz_t90 = (24*m_xy_I - 78*m_xx_I + 180*m_yy_I + 1746*m_yz_I + 168*m_zz_I - 7*omegaVar*m_xx_I - 24*omegaVar*m_xy_I - 10*omegaVar*m_yy_I + 39*omegaVar*m_yz_I + 2*omegaVar*m_zz_I + 60)/(90*(m_xx_I + 6*m_xy_I + 4*m_yy_I + 6*m_yz_I + m_zz_I - omegaVar*m_xx_I - 6*omegaVar*m_xy_I - 4*omegaVar*m_yy_I - 6*omegaVar*m_yz_I - omegaVar*m_zz_I + 15));
    m_zz_t45 = (90*m_yy_I - 234*m_xy_I - 162*m_xx_I + 504*m_yz_I + 2052*m_zz_I - 8*omegaVar*m_xx_I - 21*omegaVar*m_xy_I - 5*omegaVar*m_yy_I + 6*omegaVar*m_yz_I + 73*omegaVar*m_zz_I + 30)/(135*(m_xx_I + 6*m_xy_I + 4*m_yy_I + 6*m_yz_I + m_zz_I - omegaVar*m_xx_I - 6*omegaVar*m_xy_I - 4*omegaVar*m_yy_I - 6*omegaVar*m_yz_I - omegaVar*m_zz_I + 15));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = (m_zz_t45/12 - m_xy_t90/2 - m_yy_t45/6 - m_xx_t45/6 + (omegaVar*m_xx_t45)/12 + (omegaVar*m_xy_t90)/4 + (omegaVar*m_yy_t45)/12 - (omegaVar*m_zz_t45)/24 - 1.0/18.0)*rho + RHO_0/18;
    L_BC_Fy = ((omegaVar*m_xx_t45)/24 - m_xy_t90/2 - m_yy_t45/3 - m_yz_t90/2 - m_zz_t45/12 - m_xx_t45/12 + (omegaVar*m_xy_t90)/4 + (omegaVar*m_yy_t45)/6 + (omegaVar*m_yz_t90)/4 + (omegaVar*m_zz_t45)/24 - 1.0/9.0)*rho + RHO_0/9;
    L_BC_Fz = (m_xx_t45/12 - m_yy_t45/6 - m_yz_t90/2 - m_zz_t45/6 - (omegaVar*m_xx_t45)/24 + (omegaVar*m_yy_t45)/12 + (omegaVar*m_yz_t90)/4 + (omegaVar*m_zz_t45)/12 - 1.0/18.0)*rho + RHO_0/18;
    #endif //_BC_FORCES 

    break;
case 0b0000010110010100:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = rho_I;

    m_xx_t45 = m_xx_I;
    m_xy_t90 = m_xy_I;
    m_xz_t90 = m_xz_I;
    m_yy_t45 = m_yy_I;
    m_yz_t90 = m_yz_I;
    m_zz_t45 = m_zz_I;

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = 0;
    L_BC_Fy = 0;
    L_BC_Fz = 0;
    #endif //_BC_FORCES 

    break;
case 0b0000010110010101:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[8] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (6*(15*rho_I + 4*m_xx_I*rho_I + 6*m_xy_I*rho_I + 6*m_xz_I*rho_I + m_yy_I*rho_I + m_zz_I*rho_I - 4*omegaVar*m_xx_I*rho_I - 6*omegaVar*m_xy_I*rho_I - 6*omegaVar*m_xz_I*rho_I - omegaVar*m_yy_I*rho_I - omegaVar*m_zz_I*rho_I))/(3*omegaVar + 82);

    m_xx_t45 = (882*m_xx_I + 216*m_xy_I + 216*m_xz_I + 36*m_yy_I + 36*m_zz_I + 19*omegaVar*m_xx_I - 12*omegaVar*m_xy_I - 12*omegaVar*m_xz_I - 2*omegaVar*m_yy_I - 2*omegaVar*m_zz_I + 48)/(54*(4*m_xx_I + 6*m_xy_I + 6*m_xz_I + m_yy_I + m_zz_I - 4*omegaVar*m_xx_I - 6*omegaVar*m_xy_I - 6*omegaVar*m_xz_I - omegaVar*m_yy_I - omegaVar*m_zz_I + 15));
    m_xy_t90 = (180*m_xx_I + 1746*m_xy_I + 24*m_xz_I + 168*m_yy_I - 78*m_zz_I - 10*omegaVar*m_xx_I + 39*omegaVar*m_xy_I - 24*omegaVar*m_xz_I + 2*omegaVar*m_yy_I - 7*omegaVar*m_zz_I + 60)/(90*(4*m_xx_I + 6*m_xy_I + 6*m_xz_I + m_yy_I + m_zz_I - 4*omegaVar*m_xx_I - 6*omegaVar*m_xy_I - 6*omegaVar*m_xz_I - omegaVar*m_yy_I - omegaVar*m_zz_I + 15));
    m_xz_t90 = (180*m_xx_I + 24*m_xy_I + 1746*m_xz_I - 78*m_yy_I + 168*m_zz_I - 10*omegaVar*m_xx_I - 24*omegaVar*m_xy_I + 39*omegaVar*m_xz_I - 7*omegaVar*m_yy_I + 2*omegaVar*m_zz_I + 60)/(90*(4*m_xx_I + 6*m_xy_I + 6*m_xz_I + m_yy_I + m_zz_I - 4*omegaVar*m_xx_I - 6*omegaVar*m_xy_I - 6*omegaVar*m_xz_I - omegaVar*m_yy_I - omegaVar*m_zz_I + 15));
    m_yy_t45 = (90*m_xx_I + 504*m_xy_I - 234*m_xz_I + 2052*m_yy_I - 162*m_zz_I - 5*omegaVar*m_xx_I + 6*omegaVar*m_xy_I - 21*omegaVar*m_xz_I + 73*omegaVar*m_yy_I - 8*omegaVar*m_zz_I + 30)/(135*(4*m_xx_I + 6*m_xy_I + 6*m_xz_I + m_yy_I + m_zz_I - 4*omegaVar*m_xx_I - 6*omegaVar*m_xy_I - 6*omegaVar*m_xz_I - omegaVar*m_yy_I - omegaVar*m_zz_I + 15));
    m_yz_t90 = (m_yz_I*(3*omegaVar + 82))/(6*(4*m_xx_I + 6*m_xy_I + 6*m_xz_I + m_yy_I + m_zz_I - 4*omegaVar*m_xx_I - 6*omegaVar*m_xy_I - 6*omegaVar*m_xz_I - omegaVar*m_yy_I - omegaVar*m_zz_I + 15));
    m_zz_t45 = (90*m_xx_I - 234*m_xy_I + 504*m_xz_I - 162*m_yy_I + 2052*m_zz_I - 5*omegaVar*m_xx_I - 21*omegaVar*m_xy_I + 6*omegaVar*m_xz_I - 8*omegaVar*m_yy_I + 73*omegaVar*m_zz_I + 30)/(135*(4*m_xx_I + 6*m_xy_I + 6*m_xz_I + m_yy_I + m_zz_I - 4*omegaVar*m_xx_I - 6*omegaVar*m_xy_I - 6*omegaVar*m_xz_I - omegaVar*m_yy_I - omegaVar*m_zz_I + 15));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = ((omegaVar*m_xx_t45)/6 - m_xy_t90/2 - m_xz_t90/2 - m_yy_t45/12 - m_zz_t45/12 - m_xx_t45/3 + (omegaVar*m_xy_t90)/4 + (omegaVar*m_xz_t90)/4 + (omegaVar*m_yy_t45)/24 + (omegaVar*m_zz_t45)/24 - 1.0/9.0)*rho + RHO_0/9;
    L_BC_Fy = (m_zz_t45/12 - m_xy_t90/2 - m_yy_t45/6 - m_xx_t45/6 + (omegaVar*m_xx_t45)/12 + (omegaVar*m_xy_t90)/4 + (omegaVar*m_yy_t45)/12 - (omegaVar*m_zz_t45)/24 - 1.0/18.0)*rho + RHO_0/18;
    L_BC_Fz = (m_yy_t45/12 - m_xz_t90/2 - m_xx_t45/6 - m_zz_t45/6 + (omegaVar*m_xx_t45)/12 + (omegaVar*m_xz_t90)/4 - (omegaVar*m_yy_t45)/24 + (omegaVar*m_zz_t45)/12 - 1.0/18.0)*rho + RHO_0/18;
    #endif //_BC_FORCES 

    break;
case 0b0000010110010110:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = rho_I;

    m_xx_t45 = m_xx_I;
    m_xy_t90 = m_xy_I;
    m_xz_t90 = m_xz_I;
    m_yy_t45 = m_yy_I;
    m_yz_t90 = m_yz_I;
    m_zz_t45 = m_zz_I;

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = 0;
    L_BC_Fy = 0;
    L_BC_Fz = 0;
    #endif //_BC_FORCES 

    break;
case 0b0000010110010111:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[8] + pop[10] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (12*(5*rho_I + m_xx_I*rho_I + 2*m_xy_I*rho_I + 2*m_xz_I*rho_I + m_yy_I*rho_I + 2*m_yz_I*rho_I + m_zz_I*rho_I - omegaVar*m_xx_I*rho_I - 2*omegaVar*m_xy_I*rho_I - 2*omegaVar*m_xz_I*rho_I - omegaVar*m_yy_I*rho_I - 2*omegaVar*m_yz_I*rho_I - omegaVar*m_zz_I*rho_I))/(3*omegaVar + 52);

    m_xx_t45 = (472*m_xx_I + 112*m_xy_I + 112*m_xz_I + 4*m_yy_I - 44*m_yz_I + 4*m_zz_I + 23*omegaVar*m_xx_I - 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - 4*omegaVar*m_yy_I - 11*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 20)/(90*(m_xx_I + 2*m_xy_I + 2*m_xz_I + m_yy_I + 2*m_yz_I + m_zz_I - omegaVar*m_xx_I - 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - omegaVar*m_yy_I - 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_xy_t90 = (112*m_xx_I + 1108*m_xy_I + 16*m_xz_I + 112*m_yy_I + 16*m_yz_I - 44*m_zz_I - 2*omegaVar*m_xx_I + 47*omegaVar*m_xy_I - 16*omegaVar*m_xz_I - 2*omegaVar*m_yy_I - 16*omegaVar*m_yz_I - 11*omegaVar*m_zz_I + 40)/(180*(m_xx_I + 2*m_xy_I + 2*m_xz_I + m_yy_I + 2*m_yz_I + m_zz_I - omegaVar*m_xx_I - 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - omegaVar*m_yy_I - 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_xz_t90 = (112*m_xx_I + 16*m_xy_I + 1108*m_xz_I - 44*m_yy_I + 16*m_yz_I + 112*m_zz_I - 2*omegaVar*m_xx_I - 16*omegaVar*m_xy_I + 47*omegaVar*m_xz_I - 11*omegaVar*m_yy_I - 16*omegaVar*m_yz_I - 2*omegaVar*m_zz_I + 40)/(180*(m_xx_I + 2*m_xy_I + 2*m_xz_I + m_yy_I + 2*m_yz_I + m_zz_I - omegaVar*m_xx_I - 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - omegaVar*m_yy_I - 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_yy_t45 = (4*m_xx_I + 112*m_xy_I - 44*m_xz_I + 472*m_yy_I + 112*m_yz_I + 4*m_zz_I - 4*omegaVar*m_xx_I - 2*omegaVar*m_xy_I - 11*omegaVar*m_xz_I + 23*omegaVar*m_yy_I - 2*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 20)/(90*(m_xx_I + 2*m_xy_I + 2*m_xz_I + m_yy_I + 2*m_yz_I + m_zz_I - omegaVar*m_xx_I - 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - omegaVar*m_yy_I - 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_yz_t90 = (16*m_xy_I - 44*m_xx_I + 16*m_xz_I + 112*m_yy_I + 1108*m_yz_I + 112*m_zz_I - 11*omegaVar*m_xx_I - 16*omegaVar*m_xy_I - 16*omegaVar*m_xz_I - 2*omegaVar*m_yy_I + 47*omegaVar*m_yz_I - 2*omegaVar*m_zz_I + 40)/(180*(m_xx_I + 2*m_xy_I + 2*m_xz_I + m_yy_I + 2*m_yz_I + m_zz_I - omegaVar*m_xx_I - 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - omegaVar*m_yy_I - 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_zz_t45 = (4*m_xx_I - 44*m_xy_I + 112*m_xz_I + 4*m_yy_I + 112*m_yz_I + 472*m_zz_I - 4*omegaVar*m_xx_I - 11*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - 4*omegaVar*m_yy_I - 2*omegaVar*m_yz_I + 23*omegaVar*m_zz_I + 20)/(90*(m_xx_I + 2*m_xy_I + 2*m_xz_I + m_yy_I + 2*m_yz_I + m_zz_I - omegaVar*m_xx_I - 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - omegaVar*m_yy_I - 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = ((omegaVar*m_xx_t45)/6 - m_xy_t90/2 - m_xz_t90/2 - m_yy_t45/12 - m_zz_t45/12 - m_xx_t45/3 + (omegaVar*m_xy_t90)/4 + (omegaVar*m_xz_t90)/4 + (omegaVar*m_yy_t45)/24 + (omegaVar*m_zz_t45)/24 - 1.0/9.0)*rho + RHO_0/9;
    L_BC_Fy = ((omegaVar*m_xx_t45)/24 - m_xy_t90/2 - m_yy_t45/3 - m_yz_t90/2 - m_zz_t45/12 - m_xx_t45/12 + (omegaVar*m_xy_t90)/4 + (omegaVar*m_yy_t45)/6 + (omegaVar*m_yz_t90)/4 + (omegaVar*m_zz_t45)/24 - 1.0/9.0)*rho + RHO_0/9;
    L_BC_Fz = ((omegaVar*m_xx_t45)/24 - m_xz_t90/2 - m_yy_t45/12 - m_yz_t90/2 - m_zz_t45/3 - m_xx_t45/12 + (omegaVar*m_xz_t90)/4 + (omegaVar*m_yy_t45)/24 + (omegaVar*m_yz_t90)/4 + (omegaVar*m_zz_t45)/6 - 1.0/9.0)*rho + RHO_0/9;
    #endif //_BC_FORCES 

    break;
case 0b0000010110011000:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (12*(15*rho_I + 2*m_xx_I*rho_I + 6*m_xy_I*rho_I + 2*m_yy_I*rho_I - m_zz_I*rho_I - 2*omegaVar*m_xx_I*rho_I - 6*omegaVar*m_xy_I*rho_I - 2*omegaVar*m_yy_I*rho_I + omegaVar*m_zz_I*rho_I))/(3*omegaVar + 172);

    m_xx_t45 = (1692*m_xx_I + 432*m_xy_I + 144*m_yy_I - 72*m_zz_I + 23*omegaVar*m_xx_I - 12*omegaVar*m_xy_I - 4*omegaVar*m_yy_I + 2*omegaVar*m_zz_I + 48)/(108*(2*m_xx_I + 6*m_xy_I + 2*m_yy_I - m_zz_I - 2*omegaVar*m_xx_I - 6*omegaVar*m_xy_I - 2*omegaVar*m_yy_I + omegaVar*m_zz_I + 15));
    m_xy_t90 = (72*m_xx_I + 732*m_xy_I + 72*m_yy_I - 36*m_zz_I - 2*omegaVar*m_xx_I + 3*omegaVar*m_xy_I - 2*omegaVar*m_yy_I + omegaVar*m_zz_I + 24)/(36*(2*m_xx_I + 6*m_xy_I + 2*m_yy_I - m_zz_I - 2*omegaVar*m_xx_I - 6*omegaVar*m_xy_I - 2*omegaVar*m_yy_I + omegaVar*m_zz_I + 15));
    m_xz_t90 = (m_xz_I*(3*omegaVar + 172))/(12*(2*m_xx_I + 6*m_xy_I + 2*m_yy_I - m_zz_I - 2*omegaVar*m_xx_I - 6*omegaVar*m_xy_I - 2*omegaVar*m_yy_I + omegaVar*m_zz_I + 15));
    m_yy_t45 = (144*m_xx_I + 432*m_xy_I + 1692*m_yy_I - 72*m_zz_I - 4*omegaVar*m_xx_I - 12*omegaVar*m_xy_I + 23*omegaVar*m_yy_I + 2*omegaVar*m_zz_I + 48)/(108*(2*m_xx_I + 6*m_xy_I + 2*m_yy_I - m_zz_I - 2*omegaVar*m_xx_I - 6*omegaVar*m_xy_I - 2*omegaVar*m_yy_I + omegaVar*m_zz_I + 15));
    m_yz_t90 = (m_yz_I*(3*omegaVar + 172))/(12*(2*m_xx_I + 6*m_xy_I + 2*m_yy_I - m_zz_I - 2*omegaVar*m_xx_I - 6*omegaVar*m_xy_I - 2*omegaVar*m_yy_I + omegaVar*m_zz_I + 15));
    m_zz_t45 = (792*m_zz_I - 108*m_xy_I - 36*m_yy_I - 36*m_xx_I + omegaVar*m_xx_I + 3*omegaVar*m_xy_I + omegaVar*m_yy_I + 13*omegaVar*m_zz_I - 12)/(54*(2*m_xx_I + 6*m_xy_I + 2*m_yy_I - m_zz_I - 2*omegaVar*m_xx_I - 6*omegaVar*m_xy_I - 2*omegaVar*m_yy_I + omegaVar*m_zz_I + 15));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = (m_xx_t45/6 + m_xy_t90/2 + m_yy_t45/6 - m_zz_t45/12 - (omegaVar*m_xx_t45)/12 - (omegaVar*m_xy_t90)/4 - (omegaVar*m_yy_t45)/12 + (omegaVar*m_zz_t45)/24 + 1.0/18.0)*rho - RHO_0/18;
    L_BC_Fy = (m_xx_t45/6 + m_xy_t90/2 + m_yy_t45/6 - m_zz_t45/12 - (omegaVar*m_xx_t45)/12 - (omegaVar*m_xy_t90)/4 - (omegaVar*m_yy_t45)/12 + (omegaVar*m_zz_t45)/24 + 1.0/18.0)*rho - RHO_0/18;
    L_BC_Fz = 0;
    #endif //_BC_FORCES 

    break;
case 0b0000010110011001:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (-pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (6*(3*rho_I + 2*m_xx_I*rho_I + 6*m_xy_I*rho_I + 2*m_yy_I*rho_I - m_zz_I*rho_I - 2*omegaVar*m_xx_I*rho_I - 6*omegaVar*m_xy_I*rho_I - 2*omegaVar*m_yy_I*rho_I + omegaVar*m_zz_I*rho_I))/(3*omegaVar + 14);

    m_xx_t45 = (198*m_xx_I + 216*m_xy_I + 72*m_yy_I - 36*m_zz_I + 23*omegaVar*m_xx_I - 12*omegaVar*m_xy_I - 4*omegaVar*m_yy_I + 2*omegaVar*m_zz_I + 24)/(54*(2*m_xx_I + 6*m_xy_I + 2*m_yy_I - m_zz_I - 2*omegaVar*m_xx_I - 6*omegaVar*m_xy_I - 2*omegaVar*m_yy_I + omegaVar*m_zz_I + 3));
    m_xy_t90 = (36*m_xx_I + 150*m_xy_I + 36*m_yy_I - 18*m_zz_I - 2*omegaVar*m_xx_I + 3*omegaVar*m_xy_I - 2*omegaVar*m_yy_I + omegaVar*m_zz_I + 12)/(18*(2*m_xx_I + 6*m_xy_I + 2*m_yy_I - m_zz_I - 2*omegaVar*m_xx_I - 6*omegaVar*m_xy_I - 2*omegaVar*m_yy_I + omegaVar*m_zz_I + 3));
    m_xz_t90 = (m_xz_I*(3*omegaVar + 14))/(6*(2*m_xx_I + 6*m_xy_I + 2*m_yy_I - m_zz_I - 2*omegaVar*m_xx_I - 6*omegaVar*m_xy_I - 2*omegaVar*m_yy_I + omegaVar*m_zz_I + 3));
    m_yy_t45 = (72*m_xx_I + 216*m_xy_I + 198*m_yy_I - 36*m_zz_I - 4*omegaVar*m_xx_I - 12*omegaVar*m_xy_I + 23*omegaVar*m_yy_I + 2*omegaVar*m_zz_I + 24)/(54*(2*m_xx_I + 6*m_xy_I + 2*m_yy_I - m_zz_I - 2*omegaVar*m_xx_I - 6*omegaVar*m_xy_I - 2*omegaVar*m_yy_I + omegaVar*m_zz_I + 3));
    m_yz_t90 = (m_yz_I*(3*omegaVar + 14))/(6*(2*m_xx_I + 6*m_xy_I + 2*m_yy_I - m_zz_I - 2*omegaVar*m_xx_I - 6*omegaVar*m_xy_I - 2*omegaVar*m_yy_I + omegaVar*m_zz_I + 3));
    m_zz_t45 = (72*m_zz_I - 54*m_xy_I - 18*m_yy_I - 18*m_xx_I + omegaVar*m_xx_I + 3*omegaVar*m_xy_I + omegaVar*m_yy_I + 13*omegaVar*m_zz_I - 6)/(27*(2*m_xx_I + 6*m_xy_I + 2*m_yy_I - m_zz_I - 2*omegaVar*m_xx_I - 6*omegaVar*m_xy_I - 2*omegaVar*m_yy_I + omegaVar*m_zz_I + 3));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = 0;
    L_BC_Fy = 0;
    L_BC_Fz = 0;
    #endif //_BC_FORCES 

    break;
case 0b0000010110011010:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (6*(15*rho_I + 4*m_xx_I*rho_I + 6*m_xy_I*rho_I - 6*m_xz_I*rho_I + m_yy_I*rho_I + m_zz_I*rho_I - 4*omegaVar*m_xx_I*rho_I - 6*omegaVar*m_xy_I*rho_I + 6*omegaVar*m_xz_I*rho_I - omegaVar*m_yy_I*rho_I - omegaVar*m_zz_I*rho_I))/(3*omegaVar + 82);

    m_xx_t45 = (882*m_xx_I + 216*m_xy_I - 216*m_xz_I + 36*m_yy_I + 36*m_zz_I + 19*omegaVar*m_xx_I - 12*omegaVar*m_xy_I + 12*omegaVar*m_xz_I - 2*omegaVar*m_yy_I - 2*omegaVar*m_zz_I + 48)/(54*(4*m_xx_I + 6*m_xy_I - 6*m_xz_I + m_yy_I + m_zz_I - 4*omegaVar*m_xx_I - 6*omegaVar*m_xy_I + 6*omegaVar*m_xz_I - omegaVar*m_yy_I - omegaVar*m_zz_I + 15));
    m_xy_t90 = (180*m_xx_I + 1746*m_xy_I - 24*m_xz_I + 168*m_yy_I - 78*m_zz_I - 10*omegaVar*m_xx_I + 39*omegaVar*m_xy_I + 24*omegaVar*m_xz_I + 2*omegaVar*m_yy_I - 7*omegaVar*m_zz_I + 60)/(90*(4*m_xx_I + 6*m_xy_I - 6*m_xz_I + m_yy_I + m_zz_I - 4*omegaVar*m_xx_I - 6*omegaVar*m_xy_I + 6*omegaVar*m_xz_I - omegaVar*m_yy_I - omegaVar*m_zz_I + 15));
    m_xz_t90 = (1746*m_xz_I - 24*m_xy_I - 180*m_xx_I + 78*m_yy_I - 168*m_zz_I + 10*omegaVar*m_xx_I + 24*omegaVar*m_xy_I + 39*omegaVar*m_xz_I + 7*omegaVar*m_yy_I - 2*omegaVar*m_zz_I - 60)/(90*(4*m_xx_I + 6*m_xy_I - 6*m_xz_I + m_yy_I + m_zz_I - 4*omegaVar*m_xx_I - 6*omegaVar*m_xy_I + 6*omegaVar*m_xz_I - omegaVar*m_yy_I - omegaVar*m_zz_I + 15));
    m_yy_t45 = (90*m_xx_I + 504*m_xy_I + 234*m_xz_I + 2052*m_yy_I - 162*m_zz_I - 5*omegaVar*m_xx_I + 6*omegaVar*m_xy_I + 21*omegaVar*m_xz_I + 73*omegaVar*m_yy_I - 8*omegaVar*m_zz_I + 30)/(135*(4*m_xx_I + 6*m_xy_I - 6*m_xz_I + m_yy_I + m_zz_I - 4*omegaVar*m_xx_I - 6*omegaVar*m_xy_I + 6*omegaVar*m_xz_I - omegaVar*m_yy_I - omegaVar*m_zz_I + 15));
    m_yz_t90 = (m_yz_I*(3*omegaVar + 82))/(6*(4*m_xx_I + 6*m_xy_I - 6*m_xz_I + m_yy_I + m_zz_I - 4*omegaVar*m_xx_I - 6*omegaVar*m_xy_I + 6*omegaVar*m_xz_I - omegaVar*m_yy_I - omegaVar*m_zz_I + 15));
    m_zz_t45 = -(234*m_xy_I - 90*m_xx_I + 504*m_xz_I + 162*m_yy_I - 2052*m_zz_I + 5*omegaVar*m_xx_I + 21*omegaVar*m_xy_I + 6*omegaVar*m_xz_I + 8*omegaVar*m_yy_I - 73*omegaVar*m_zz_I - 30)/(135*(4*m_xx_I + 6*m_xy_I - 6*m_xz_I + m_yy_I + m_zz_I - 4*omegaVar*m_xx_I - 6*omegaVar*m_xy_I + 6*omegaVar*m_xz_I - omegaVar*m_yy_I - omegaVar*m_zz_I + 15));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = (m_xx_t45/3 + m_xy_t90/2 - m_xz_t90/2 + m_yy_t45/12 + m_zz_t45/12 - (omegaVar*m_xx_t45)/6 - (omegaVar*m_xy_t90)/4 + (omegaVar*m_xz_t90)/4 - (omegaVar*m_yy_t45)/24 - (omegaVar*m_zz_t45)/24 + 1.0/9.0)*rho - RHO_0/9;
    L_BC_Fy = (m_xx_t45/6 + m_xy_t90/2 + m_yy_t45/6 - m_zz_t45/12 - (omegaVar*m_xx_t45)/12 - (omegaVar*m_xy_t90)/4 - (omegaVar*m_yy_t45)/12 + (omegaVar*m_zz_t45)/24 + 1.0/18.0)*rho - RHO_0/18;
    L_BC_Fz = (m_xz_t90/2 - m_xx_t45/6 + m_yy_t45/12 - m_zz_t45/6 + (omegaVar*m_xx_t45)/12 - (omegaVar*m_xz_t90)/4 - (omegaVar*m_yy_t45)/24 + (omegaVar*m_zz_t45)/12 - 1.0/18.0)*rho + RHO_0/18;
    #endif //_BC_FORCES 

    break;
case 0b0000010110011011:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[9] + pop[10] + pop[12] + pop[13] + pop[14] + pop[15] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (-pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (3*(15*rho_I + 11*m_xx_I*rho_I + 30*m_xy_I*rho_I - 6*m_xz_I*rho_I + 11*m_yy_I*rho_I + 6*m_yz_I*rho_I - m_zz_I*rho_I - 11*omegaVar*m_xx_I*rho_I - 30*omegaVar*m_xy_I*rho_I + 6*omegaVar*m_xz_I*rho_I - 11*omegaVar*m_yy_I*rho_I - 6*omegaVar*m_yz_I*rho_I + omegaVar*m_zz_I*rho_I))/(9*omegaVar + 31);

    m_xx_t45 = -(2*(252*m_xz_I - 1260*m_xy_I - 1206*m_xx_I - 369*m_yy_I + 27*m_yz_I + 135*m_zz_I - 194*omegaVar*m_xx_I + 60*omegaVar*m_xy_I - 12*omegaVar*m_xz_I + 49*omegaVar*m_yy_I + 93*omegaVar*m_yz_I + 25*omegaVar*m_zz_I - 165))/(135*(11*m_xx_I + 30*m_xy_I - 6*m_xz_I + 11*m_yy_I + 6*m_yz_I - m_zz_I - 11*omegaVar*m_xx_I - 30*omegaVar*m_xy_I + 6*omegaVar*m_xz_I - 11*omegaVar*m_yy_I - 6*omegaVar*m_yz_I + omegaVar*m_zz_I + 15));
    m_xy_t90 = (84*m_xx_I + 339*m_xy_I - 12*m_xz_I + 84*m_yy_I + 12*m_yz_I - 33*m_zz_I - 4*omegaVar*m_xx_I + 21*omegaVar*m_xy_I + 12*omegaVar*m_xz_I - 4*omegaVar*m_yy_I - 12*omegaVar*m_yz_I - 7*omegaVar*m_zz_I + 30)/(9*(11*m_xx_I + 30*m_xy_I - 6*m_xz_I + 11*m_yy_I + 6*m_yz_I - m_zz_I - 11*omegaVar*m_xx_I - 30*omegaVar*m_xy_I + 6*omegaVar*m_xz_I - 11*omegaVar*m_yy_I - 6*omegaVar*m_yz_I + omegaVar*m_zz_I + 15));
    m_xz_t90 = (663*m_xz_I - 60*m_xy_I - 84*m_xx_I + 9*m_yy_I - 12*m_yz_I - 60*m_zz_I + 4*omegaVar*m_xx_I + 60*omegaVar*m_xy_I + 177*omegaVar*m_xz_I + 31*omegaVar*m_yy_I + 12*omegaVar*m_yz_I - 20*omegaVar*m_zz_I - 30)/(45*(11*m_xx_I + 30*m_xy_I - 6*m_xz_I + 11*m_yy_I + 6*m_yz_I - m_zz_I - 11*omegaVar*m_xx_I - 30*omegaVar*m_xy_I + 6*omegaVar*m_xz_I - 11*omegaVar*m_yy_I - 6*omegaVar*m_yz_I + omegaVar*m_zz_I + 15));
    m_yy_t45 = (2*(369*m_xx_I + 1260*m_xy_I + 27*m_xz_I + 1206*m_yy_I + 252*m_yz_I - 135*m_zz_I - 49*omegaVar*m_xx_I - 60*omegaVar*m_xy_I + 93*omegaVar*m_xz_I + 194*omegaVar*m_yy_I - 12*omegaVar*m_yz_I - 25*omegaVar*m_zz_I + 165))/(135*(11*m_xx_I + 30*m_xy_I - 6*m_xz_I + 11*m_yy_I + 6*m_yz_I - m_zz_I - 11*omegaVar*m_xx_I - 30*omegaVar*m_xy_I + 6*omegaVar*m_xz_I - 11*omegaVar*m_yy_I - 6*omegaVar*m_yz_I + omegaVar*m_zz_I + 15));
    m_yz_t90 = (60*m_xy_I - 9*m_xx_I - 12*m_xz_I + 84*m_yy_I + 663*m_yz_I + 60*m_zz_I - 31*omegaVar*m_xx_I - 60*omegaVar*m_xy_I + 12*omegaVar*m_xz_I - 4*omegaVar*m_yy_I + 177*omegaVar*m_yz_I + 20*omegaVar*m_zz_I + 30)/(45*(11*m_xx_I + 30*m_xy_I - 6*m_xz_I + 11*m_yy_I + 6*m_yz_I - m_zz_I - 11*omegaVar*m_xx_I - 30*omegaVar*m_xy_I + 6*omegaVar*m_xz_I - 11*omegaVar*m_yy_I - 6*omegaVar*m_yz_I + omegaVar*m_zz_I + 15));
    m_zz_t45 = -(2*(27*m_xx_I + 99*m_xy_I + 36*m_xz_I + 27*m_yy_I - 36*m_yz_I - 180*m_zz_I + 5*omegaVar*m_xx_I + 21*omegaVar*m_xy_I + 12*omegaVar*m_xz_I + 5*omegaVar*m_yy_I - 12*omegaVar*m_yz_I - 52*omegaVar*m_zz_I + 3))/(27*(11*m_xx_I + 30*m_xy_I - 6*m_xz_I + 11*m_yy_I + 6*m_yz_I - m_zz_I - 11*omegaVar*m_xx_I - 30*omegaVar*m_xy_I + 6*omegaVar*m_xz_I - 11*omegaVar*m_yy_I - 6*omegaVar*m_yz_I + omegaVar*m_zz_I + 15));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = (m_xx_t45/6 - m_xz_t90/2 - m_yy_t45/12 + m_zz_t45/6 - (omegaVar*m_xx_t45)/12 + (omegaVar*m_xz_t90)/4 + (omegaVar*m_yy_t45)/24 - (omegaVar*m_zz_t45)/12 + 1.0/18.0)*rho - RHO_0/18;
    L_BC_Fy = (m_xx_t45/12 - m_yy_t45/6 - m_yz_t90/2 - m_zz_t45/6 - (omegaVar*m_xx_t45)/24 + (omegaVar*m_yy_t45)/12 + (omegaVar*m_yz_t90)/4 + (omegaVar*m_zz_t45)/12 - 1.0/18.0)*rho + RHO_0/18;
    L_BC_Fz = (m_xz_t90/2 - m_xx_t45/12 - m_yy_t45/12 - m_yz_t90/2 - m_zz_t45/3 + (omegaVar*m_xx_t45)/24 - (omegaVar*m_xz_t90)/4 + (omegaVar*m_yy_t45)/24 + (omegaVar*m_yz_t90)/4 + (omegaVar*m_zz_t45)/6 - 1.0/9.0)*rho + RHO_0/9;
    #endif //_BC_FORCES 

    break;
case 0b0000010110011100:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17]);
    m_xy_I = inv_rho_I * (pop[7] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17]);

    rho = (6*(15*rho_I + m_xx_I*rho_I + 6*m_xy_I*rho_I + 4*m_yy_I*rho_I - 6*m_yz_I*rho_I + m_zz_I*rho_I - omegaVar*m_xx_I*rho_I - 6*omegaVar*m_xy_I*rho_I - 4*omegaVar*m_yy_I*rho_I + 6*omegaVar*m_yz_I*rho_I - omegaVar*m_zz_I*rho_I))/(3*omegaVar + 82);

    m_xx_t45 = (2052*m_xx_I + 504*m_xy_I + 90*m_yy_I + 234*m_yz_I - 162*m_zz_I + 73*omegaVar*m_xx_I + 6*omegaVar*m_xy_I - 5*omegaVar*m_yy_I + 21*omegaVar*m_yz_I - 8*omegaVar*m_zz_I + 30)/(135*(m_xx_I + 6*m_xy_I + 4*m_yy_I - 6*m_yz_I + m_zz_I - omegaVar*m_xx_I - 6*omegaVar*m_xy_I - 4*omegaVar*m_yy_I + 6*omegaVar*m_yz_I - omegaVar*m_zz_I + 15));
    m_xy_t90 = (168*m_xx_I + 1746*m_xy_I + 180*m_yy_I - 24*m_yz_I - 78*m_zz_I + 2*omegaVar*m_xx_I + 39*omegaVar*m_xy_I - 10*omegaVar*m_yy_I + 24*omegaVar*m_yz_I - 7*omegaVar*m_zz_I + 60)/(90*(m_xx_I + 6*m_xy_I + 4*m_yy_I - 6*m_yz_I + m_zz_I - omegaVar*m_xx_I - 6*omegaVar*m_xy_I - 4*omegaVar*m_yy_I + 6*omegaVar*m_yz_I - omegaVar*m_zz_I + 15));
    m_xz_t90 = (m_xz_I*(3*omegaVar + 82))/(6*(m_xx_I + 6*m_xy_I + 4*m_yy_I - 6*m_yz_I + m_zz_I - omegaVar*m_xx_I - 6*omegaVar*m_xy_I - 4*omegaVar*m_yy_I + 6*omegaVar*m_yz_I - omegaVar*m_zz_I + 15));
    m_yy_t45 = (36*m_xx_I + 216*m_xy_I + 882*m_yy_I - 216*m_yz_I + 36*m_zz_I - 2*omegaVar*m_xx_I - 12*omegaVar*m_xy_I + 19*omegaVar*m_yy_I + 12*omegaVar*m_yz_I - 2*omegaVar*m_zz_I + 48)/(54*(m_xx_I + 6*m_xy_I + 4*m_yy_I - 6*m_yz_I + m_zz_I - omegaVar*m_xx_I - 6*omegaVar*m_xy_I - 4*omegaVar*m_yy_I + 6*omegaVar*m_yz_I - omegaVar*m_zz_I + 15));
    m_yz_t90 = (78*m_xx_I - 24*m_xy_I - 180*m_yy_I + 1746*m_yz_I - 168*m_zz_I + 7*omegaVar*m_xx_I + 24*omegaVar*m_xy_I + 10*omegaVar*m_yy_I + 39*omegaVar*m_yz_I - 2*omegaVar*m_zz_I - 60)/(90*(m_xx_I + 6*m_xy_I + 4*m_yy_I - 6*m_yz_I + m_zz_I - omegaVar*m_xx_I - 6*omegaVar*m_xy_I - 4*omegaVar*m_yy_I + 6*omegaVar*m_yz_I - omegaVar*m_zz_I + 15));
    m_zz_t45 = -(162*m_xx_I + 234*m_xy_I - 90*m_yy_I + 504*m_yz_I - 2052*m_zz_I + 8*omegaVar*m_xx_I + 21*omegaVar*m_xy_I + 5*omegaVar*m_yy_I + 6*omegaVar*m_yz_I - 73*omegaVar*m_zz_I - 30)/(135*(m_xx_I + 6*m_xy_I + 4*m_yy_I - 6*m_yz_I + m_zz_I - omegaVar*m_xx_I - 6*omegaVar*m_xy_I - 4*omegaVar*m_yy_I + 6*omegaVar*m_yz_I - omegaVar*m_zz_I + 15));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = (m_xx_t45/6 + m_xy_t90/2 + m_yy_t45/6 - m_zz_t45/12 - (omegaVar*m_xx_t45)/12 - (omegaVar*m_xy_t90)/4 - (omegaVar*m_yy_t45)/12 + (omegaVar*m_zz_t45)/24 + 1.0/18.0)*rho - RHO_0/18;
    L_BC_Fy = (m_xx_t45/12 + m_xy_t90/2 + m_yy_t45/3 - m_yz_t90/2 + m_zz_t45/12 - (omegaVar*m_xx_t45)/24 - (omegaVar*m_xy_t90)/4 - (omegaVar*m_yy_t45)/6 + (omegaVar*m_yz_t90)/4 - (omegaVar*m_zz_t45)/24 + 1.0/9.0)*rho - RHO_0/9;
    L_BC_Fz = (m_xx_t45/12 - m_yy_t45/6 + m_yz_t90/2 - m_zz_t45/6 - (omegaVar*m_xx_t45)/24 + (omegaVar*m_yy_t45)/12 - (omegaVar*m_yz_t90)/4 + (omegaVar*m_zz_t45)/12 - 1.0/18.0)*rho + RHO_0/18;
    #endif //_BC_FORCES 

    break;
case 0b0000010110011101:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17]);
    m_xy_I = inv_rho_I * (-pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17]);

    rho = (3*(15*rho_I + 11*m_xx_I*rho_I + 30*m_xy_I*rho_I + 6*m_xz_I*rho_I + 11*m_yy_I*rho_I - 6*m_yz_I*rho_I - m_zz_I*rho_I - 11*omegaVar*m_xx_I*rho_I - 30*omegaVar*m_xy_I*rho_I - 6*omegaVar*m_xz_I*rho_I - 11*omegaVar*m_yy_I*rho_I + 6*omegaVar*m_yz_I*rho_I + omegaVar*m_zz_I*rho_I))/(9*omegaVar + 31);

    m_xx_t45 = (2*(1206*m_xx_I + 1260*m_xy_I + 252*m_xz_I + 369*m_yy_I + 27*m_yz_I - 135*m_zz_I + 194*omegaVar*m_xx_I - 60*omegaVar*m_xy_I - 12*omegaVar*m_xz_I - 49*omegaVar*m_yy_I + 93*omegaVar*m_yz_I - 25*omegaVar*m_zz_I + 165))/(135*(11*m_xx_I + 30*m_xy_I + 6*m_xz_I + 11*m_yy_I - 6*m_yz_I - m_zz_I - 11*omegaVar*m_xx_I - 30*omegaVar*m_xy_I - 6*omegaVar*m_xz_I - 11*omegaVar*m_yy_I + 6*omegaVar*m_yz_I + omegaVar*m_zz_I + 15));
    m_xy_t90 = (84*m_xx_I + 339*m_xy_I + 12*m_xz_I + 84*m_yy_I - 12*m_yz_I - 33*m_zz_I - 4*omegaVar*m_xx_I + 21*omegaVar*m_xy_I - 12*omegaVar*m_xz_I - 4*omegaVar*m_yy_I + 12*omegaVar*m_yz_I - 7*omegaVar*m_zz_I + 30)/(9*(11*m_xx_I + 30*m_xy_I + 6*m_xz_I + 11*m_yy_I - 6*m_yz_I - m_zz_I - 11*omegaVar*m_xx_I - 30*omegaVar*m_xy_I - 6*omegaVar*m_xz_I - 11*omegaVar*m_yy_I + 6*omegaVar*m_yz_I + omegaVar*m_zz_I + 15));
    m_xz_t90 = (84*m_xx_I + 60*m_xy_I + 663*m_xz_I - 9*m_yy_I - 12*m_yz_I + 60*m_zz_I - 4*omegaVar*m_xx_I - 60*omegaVar*m_xy_I + 177*omegaVar*m_xz_I - 31*omegaVar*m_yy_I + 12*omegaVar*m_yz_I + 20*omegaVar*m_zz_I + 30)/(45*(11*m_xx_I + 30*m_xy_I + 6*m_xz_I + 11*m_yy_I - 6*m_yz_I - m_zz_I - 11*omegaVar*m_xx_I - 30*omegaVar*m_xy_I - 6*omegaVar*m_xz_I - 11*omegaVar*m_yy_I + 6*omegaVar*m_yz_I + omegaVar*m_zz_I + 15));
    m_yy_t45 = -(2*(27*m_xz_I - 1260*m_xy_I - 369*m_xx_I - 1206*m_yy_I + 252*m_yz_I + 135*m_zz_I + 49*omegaVar*m_xx_I + 60*omegaVar*m_xy_I + 93*omegaVar*m_xz_I - 194*omegaVar*m_yy_I - 12*omegaVar*m_yz_I + 25*omegaVar*m_zz_I - 165))/(135*(11*m_xx_I + 30*m_xy_I + 6*m_xz_I + 11*m_yy_I - 6*m_yz_I - m_zz_I - 11*omegaVar*m_xx_I - 30*omegaVar*m_xy_I - 6*omegaVar*m_xz_I - 11*omegaVar*m_yy_I + 6*omegaVar*m_yz_I + omegaVar*m_zz_I + 15));
    m_yz_t90 = (9*m_xx_I - 60*m_xy_I - 12*m_xz_I - 84*m_yy_I + 663*m_yz_I - 60*m_zz_I + 31*omegaVar*m_xx_I + 60*omegaVar*m_xy_I + 12*omegaVar*m_xz_I + 4*omegaVar*m_yy_I + 177*omegaVar*m_yz_I - 20*omegaVar*m_zz_I - 30)/(45*(11*m_xx_I + 30*m_xy_I + 6*m_xz_I + 11*m_yy_I - 6*m_yz_I - m_zz_I - 11*omegaVar*m_xx_I - 30*omegaVar*m_xy_I - 6*omegaVar*m_xz_I - 11*omegaVar*m_yy_I + 6*omegaVar*m_yz_I + omegaVar*m_zz_I + 15));
    m_zz_t45 = -(2*(27*m_xx_I + 99*m_xy_I - 36*m_xz_I + 27*m_yy_I + 36*m_yz_I - 180*m_zz_I + 5*omegaVar*m_xx_I + 21*omegaVar*m_xy_I - 12*omegaVar*m_xz_I + 5*omegaVar*m_yy_I + 12*omegaVar*m_yz_I - 52*omegaVar*m_zz_I + 3))/(27*(11*m_xx_I + 30*m_xy_I + 6*m_xz_I + 11*m_yy_I - 6*m_yz_I - m_zz_I - 11*omegaVar*m_xx_I - 30*omegaVar*m_xy_I - 6*omegaVar*m_xz_I - 11*omegaVar*m_yy_I + 6*omegaVar*m_yz_I + omegaVar*m_zz_I + 15));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = (m_yy_t45/12 - m_xz_t90/2 - m_xx_t45/6 - m_zz_t45/6 + (omegaVar*m_xx_t45)/12 + (omegaVar*m_xz_t90)/4 - (omegaVar*m_yy_t45)/24 + (omegaVar*m_zz_t45)/12 - 1.0/18.0)*rho + RHO_0/18;
    L_BC_Fy = (m_yy_t45/6 - m_xx_t45/12 - m_yz_t90/2 + m_zz_t45/6 + (omegaVar*m_xx_t45)/24 - (omegaVar*m_yy_t45)/12 + (omegaVar*m_yz_t90)/4 - (omegaVar*m_zz_t45)/12 + 1.0/18.0)*rho - RHO_0/18;
    L_BC_Fz = (m_yz_t90/2 - m_xz_t90/2 - m_yy_t45/12 - m_xx_t45/12 - m_zz_t45/3 + (omegaVar*m_xx_t45)/24 + (omegaVar*m_xz_t90)/4 + (omegaVar*m_yy_t45)/24 - (omegaVar*m_yz_t90)/4 + (omegaVar*m_zz_t45)/6 - 1.0/9.0)*rho + RHO_0/9;
    #endif //_BC_FORCES 

    break;
case 0b0000010110011110:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[17];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] - (1.0/3.0)*pop[17]);
    m_xy_I = inv_rho_I * (pop[7] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] + (2.0/3.0)*pop[17]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[17]);

    rho = (12*(5*rho_I + m_xx_I*rho_I + 2*m_xy_I*rho_I - 2*m_xz_I*rho_I + m_yy_I*rho_I - 2*m_yz_I*rho_I + m_zz_I*rho_I - omegaVar*m_xx_I*rho_I - 2*omegaVar*m_xy_I*rho_I + 2*omegaVar*m_xz_I*rho_I - omegaVar*m_yy_I*rho_I + 2*omegaVar*m_yz_I*rho_I - omegaVar*m_zz_I*rho_I))/(3*omegaVar + 52);

    m_xx_t45 = (472*m_xx_I + 112*m_xy_I - 112*m_xz_I + 4*m_yy_I + 44*m_yz_I + 4*m_zz_I + 23*omegaVar*m_xx_I - 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - 4*omegaVar*m_yy_I + 11*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 20)/(90*(m_xx_I + 2*m_xy_I - 2*m_xz_I + m_yy_I - 2*m_yz_I + m_zz_I - omegaVar*m_xx_I - 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - omegaVar*m_yy_I + 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_xy_t90 = (112*m_xx_I + 1108*m_xy_I - 16*m_xz_I + 112*m_yy_I - 16*m_yz_I - 44*m_zz_I - 2*omegaVar*m_xx_I + 47*omegaVar*m_xy_I + 16*omegaVar*m_xz_I - 2*omegaVar*m_yy_I + 16*omegaVar*m_yz_I - 11*omegaVar*m_zz_I + 40)/(180*(m_xx_I + 2*m_xy_I - 2*m_xz_I + m_yy_I - 2*m_yz_I + m_zz_I - omegaVar*m_xx_I - 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - omegaVar*m_yy_I + 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_xz_t90 = (1108*m_xz_I - 16*m_xy_I - 112*m_xx_I + 44*m_yy_I + 16*m_yz_I - 112*m_zz_I + 2*omegaVar*m_xx_I + 16*omegaVar*m_xy_I + 47*omegaVar*m_xz_I + 11*omegaVar*m_yy_I - 16*omegaVar*m_yz_I + 2*omegaVar*m_zz_I - 40)/(180*(m_xx_I + 2*m_xy_I - 2*m_xz_I + m_yy_I - 2*m_yz_I + m_zz_I - omegaVar*m_xx_I - 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - omegaVar*m_yy_I + 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_yy_t45 = (4*m_xx_I + 112*m_xy_I + 44*m_xz_I + 472*m_yy_I - 112*m_yz_I + 4*m_zz_I - 4*omegaVar*m_xx_I - 2*omegaVar*m_xy_I + 11*omegaVar*m_xz_I + 23*omegaVar*m_yy_I + 2*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 20)/(90*(m_xx_I + 2*m_xy_I - 2*m_xz_I + m_yy_I - 2*m_yz_I + m_zz_I - omegaVar*m_xx_I - 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - omegaVar*m_yy_I + 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_yz_t90 = (44*m_xx_I - 16*m_xy_I + 16*m_xz_I - 112*m_yy_I + 1108*m_yz_I - 112*m_zz_I + 11*omegaVar*m_xx_I + 16*omegaVar*m_xy_I - 16*omegaVar*m_xz_I + 2*omegaVar*m_yy_I + 47*omegaVar*m_yz_I + 2*omegaVar*m_zz_I - 40)/(180*(m_xx_I + 2*m_xy_I - 2*m_xz_I + m_yy_I - 2*m_yz_I + m_zz_I - omegaVar*m_xx_I - 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - omegaVar*m_yy_I + 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_zz_t45 = (4*m_xx_I - 44*m_xy_I - 112*m_xz_I + 4*m_yy_I - 112*m_yz_I + 472*m_zz_I - 4*omegaVar*m_xx_I - 11*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - 4*omegaVar*m_yy_I + 2*omegaVar*m_yz_I + 23*omegaVar*m_zz_I + 20)/(90*(m_xx_I + 2*m_xy_I - 2*m_xz_I + m_yy_I - 2*m_yz_I + m_zz_I - omegaVar*m_xx_I - 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - omegaVar*m_yy_I + 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = (m_xx_t45/3 + m_xy_t90/2 - m_xz_t90/2 + m_yy_t45/12 + m_zz_t45/12 - (omegaVar*m_xx_t45)/6 - (omegaVar*m_xy_t90)/4 + (omegaVar*m_xz_t90)/4 - (omegaVar*m_yy_t45)/24 - (omegaVar*m_zz_t45)/24 + 1.0/9.0)*rho - RHO_0/9;
    L_BC_Fy = (m_xx_t45/12 + m_xy_t90/2 + m_yy_t45/3 - m_yz_t90/2 + m_zz_t45/12 - (omegaVar*m_xx_t45)/24 - (omegaVar*m_xy_t90)/4 - (omegaVar*m_yy_t45)/6 + (omegaVar*m_yz_t90)/4 - (omegaVar*m_zz_t45)/24 + 1.0/9.0)*rho - RHO_0/9;
    L_BC_Fz = (m_xz_t90/2 - m_xx_t45/12 - m_yy_t45/12 + m_yz_t90/2 - m_zz_t45/3 + (omegaVar*m_xx_t45)/24 - (omegaVar*m_xz_t90)/4 + (omegaVar*m_yy_t45)/24 - (omegaVar*m_yz_t90)/4 + (omegaVar*m_zz_t45)/6 - 1.0/9.0)*rho + RHO_0/9;
    #endif //_BC_FORCES 

    break;
case 0b0000010110011111:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[6] + pop[10] + pop[12] + pop[13] + pop[14] + pop[15] + pop[17];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] - (1.0/3.0)*pop[17]);
    m_xy_I = inv_rho_I * (-pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[10] - pop[15]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[6] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] + (2.0/3.0)*pop[17]);
    m_yz_I = inv_rho_I * (pop[12] - pop[17]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[6] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[17]);

    rho = (18*(19*rho_I + 12*m_xx_I*rho_I + 33*m_xy_I*rho_I + 12*m_yy_I*rho_I + 6*m_zz_I*rho_I - 12*omegaVar*m_xx_I*rho_I - 33*omegaVar*m_xy_I*rho_I - 12*omegaVar*m_yy_I*rho_I - 6*omegaVar*m_zz_I*rho_I))/(59*omegaVar + 207);

    m_xx_t45 = (2*(306*m_xx_I + 324*m_xy_I + 99*m_yy_I - 54*m_zz_I + 58*omegaVar*m_xx_I + 12*omegaVar*m_xy_I - omegaVar*m_yy_I - 30*omegaVar*m_zz_I + 36))/(27*(12*m_xx_I + 33*m_xy_I + 12*m_yy_I + 6*m_zz_I - 12*omegaVar*m_xx_I - 33*omegaVar*m_xy_I - 12*omegaVar*m_yy_I - 6*omegaVar*m_zz_I + 19));
    m_xy_t90 = (216*m_xx_I + 801*m_xy_I + 216*m_yy_I - 99*m_zz_I + 8*omegaVar*m_xx_I + 81*omegaVar*m_xy_I + 8*omegaVar*m_yy_I - 55*omegaVar*m_zz_I + 66)/(18*(12*m_xx_I + 33*m_xy_I + 12*m_yy_I + 6*m_zz_I - 12*omegaVar*m_xx_I - 33*omegaVar*m_xy_I - 12*omegaVar*m_yy_I - 6*omegaVar*m_zz_I + 19));
    m_xz_t90 = (m_xz_I*(59*omegaVar + 207))/(9*(12*m_xx_I + 33*m_xy_I + 12*m_yy_I + 6*m_zz_I - 12*omegaVar*m_xx_I - 33*omegaVar*m_xy_I - 12*omegaVar*m_yy_I - 6*omegaVar*m_zz_I + 19));
    m_yy_t45 = (2*(99*m_xx_I + 324*m_xy_I + 306*m_yy_I - 54*m_zz_I - omegaVar*m_xx_I + 12*omegaVar*m_xy_I + 58*omegaVar*m_yy_I - 30*omegaVar*m_zz_I + 36))/(27*(12*m_xx_I + 33*m_xy_I + 12*m_yy_I + 6*m_zz_I - 12*omegaVar*m_xx_I - 33*omegaVar*m_xy_I - 12*omegaVar*m_yy_I - 6*omegaVar*m_zz_I + 19));
    m_yz_t90 = (m_yz_I*(59*omegaVar + 207))/(9*(12*m_xx_I + 33*m_xy_I + 12*m_yy_I + 6*m_zz_I - 12*omegaVar*m_xx_I - 33*omegaVar*m_xy_I - 12*omegaVar*m_yy_I - 6*omegaVar*m_zz_I + 19));
    m_zz_t45 = -(36*m_xx_I + 99*m_xy_I + 36*m_yy_I - 189*m_zz_I + 20*omegaVar*m_xx_I + 55*omegaVar*m_xy_I + 20*omegaVar*m_yy_I - 49*omegaVar*m_zz_I - 12)/(9*(12*m_xx_I + 33*m_xy_I + 12*m_yy_I + 6*m_zz_I - 12*omegaVar*m_xx_I - 33*omegaVar*m_xy_I - 12*omegaVar*m_yy_I - 6*omegaVar*m_zz_I + 19));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = (m_xz_t90*rho*(omegaVar - 2))/2;
    L_BC_Fy = (m_yz_t90*rho*(omegaVar - 2))/2;
    L_BC_Fz = ((omegaVar*m_zz_t45)/2 - m_zz_t45 - 1.0/3.0)*rho + RHO_0/3;
    #endif //_BC_FORCES 

    break;
case 0b0000010110100000:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (12*(15*rho_I + 2*m_xx_I*rho_I + 6*m_xz_I*rho_I - m_yy_I*rho_I + 2*m_zz_I*rho_I - 2*omegaVar*m_xx_I*rho_I - 6*omegaVar*m_xz_I*rho_I + omegaVar*m_yy_I*rho_I - 2*omegaVar*m_zz_I*rho_I))/(3*omegaVar + 172);

    m_xx_t45 = (1692*m_xx_I + 432*m_xz_I - 72*m_yy_I + 144*m_zz_I + 23*omegaVar*m_xx_I - 12*omegaVar*m_xz_I + 2*omegaVar*m_yy_I - 4*omegaVar*m_zz_I + 48)/(108*(2*m_xx_I + 6*m_xz_I - m_yy_I + 2*m_zz_I - 2*omegaVar*m_xx_I - 6*omegaVar*m_xz_I + omegaVar*m_yy_I - 2*omegaVar*m_zz_I + 15));
    m_xy_t90 = (m_xy_I*(3*omegaVar + 172))/(12*(2*m_xx_I + 6*m_xz_I - m_yy_I + 2*m_zz_I - 2*omegaVar*m_xx_I - 6*omegaVar*m_xz_I + omegaVar*m_yy_I - 2*omegaVar*m_zz_I + 15));
    m_xz_t90 = (72*m_xx_I + 732*m_xz_I - 36*m_yy_I + 72*m_zz_I - 2*omegaVar*m_xx_I + 3*omegaVar*m_xz_I + omegaVar*m_yy_I - 2*omegaVar*m_zz_I + 24)/(36*(2*m_xx_I + 6*m_xz_I - m_yy_I + 2*m_zz_I - 2*omegaVar*m_xx_I - 6*omegaVar*m_xz_I + omegaVar*m_yy_I - 2*omegaVar*m_zz_I + 15));
    m_yy_t45 = (792*m_yy_I - 108*m_xz_I - 36*m_xx_I - 36*m_zz_I + omegaVar*m_xx_I + 3*omegaVar*m_xz_I + 13*omegaVar*m_yy_I + omegaVar*m_zz_I - 12)/(54*(2*m_xx_I + 6*m_xz_I - m_yy_I + 2*m_zz_I - 2*omegaVar*m_xx_I - 6*omegaVar*m_xz_I + omegaVar*m_yy_I - 2*omegaVar*m_zz_I + 15));
    m_yz_t90 = (m_yz_I*(3*omegaVar + 172))/(12*(2*m_xx_I + 6*m_xz_I - m_yy_I + 2*m_zz_I - 2*omegaVar*m_xx_I - 6*omegaVar*m_xz_I + omegaVar*m_yy_I - 2*omegaVar*m_zz_I + 15));
    m_zz_t45 = (144*m_xx_I + 432*m_xz_I - 72*m_yy_I + 1692*m_zz_I - 4*omegaVar*m_xx_I - 12*omegaVar*m_xz_I + 2*omegaVar*m_yy_I + 23*omegaVar*m_zz_I + 48)/(108*(2*m_xx_I + 6*m_xz_I - m_yy_I + 2*m_zz_I - 2*omegaVar*m_xx_I - 6*omegaVar*m_xz_I + omegaVar*m_yy_I - 2*omegaVar*m_zz_I + 15));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = (m_xx_t45/6 + m_xz_t90/2 - m_yy_t45/12 + m_zz_t45/6 - (omegaVar*m_xx_t45)/12 - (omegaVar*m_xz_t90)/4 + (omegaVar*m_yy_t45)/24 - (omegaVar*m_zz_t45)/12 + 1.0/18.0)*rho - RHO_0/18;
    L_BC_Fy = 0;
    L_BC_Fz = (m_xx_t45/6 + m_xz_t90/2 - m_yy_t45/12 + m_zz_t45/6 - (omegaVar*m_xx_t45)/12 - (omegaVar*m_xz_t90)/4 + (omegaVar*m_yy_t45)/24 - (omegaVar*m_zz_t45)/12 + 1.0/18.0)*rho - RHO_0/18;
    #endif //_BC_FORCES 

    break;
case 0b0000010110100001:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (12*(15*rho_I + 2*m_xx_I*rho_I + 6*m_xz_I*rho_I - m_yy_I*rho_I + 2*m_zz_I*rho_I - 2*omegaVar*m_xx_I*rho_I - 6*omegaVar*m_xz_I*rho_I + omegaVar*m_yy_I*rho_I - 2*omegaVar*m_zz_I*rho_I))/(3*omegaVar + 172);

    m_xx_t45 = (1692*m_xx_I + 432*m_xz_I - 72*m_yy_I + 144*m_zz_I + 23*omegaVar*m_xx_I - 12*omegaVar*m_xz_I + 2*omegaVar*m_yy_I - 4*omegaVar*m_zz_I + 48)/(108*(2*m_xx_I + 6*m_xz_I - m_yy_I + 2*m_zz_I - 2*omegaVar*m_xx_I - 6*omegaVar*m_xz_I + omegaVar*m_yy_I - 2*omegaVar*m_zz_I + 15));
    m_xy_t90 = (m_xy_I*(3*omegaVar + 172))/(12*(2*m_xx_I + 6*m_xz_I - m_yy_I + 2*m_zz_I - 2*omegaVar*m_xx_I - 6*omegaVar*m_xz_I + omegaVar*m_yy_I - 2*omegaVar*m_zz_I + 15));
    m_xz_t90 = (72*m_xx_I + 732*m_xz_I - 36*m_yy_I + 72*m_zz_I - 2*omegaVar*m_xx_I + 3*omegaVar*m_xz_I + omegaVar*m_yy_I - 2*omegaVar*m_zz_I + 24)/(36*(2*m_xx_I + 6*m_xz_I - m_yy_I + 2*m_zz_I - 2*omegaVar*m_xx_I - 6*omegaVar*m_xz_I + omegaVar*m_yy_I - 2*omegaVar*m_zz_I + 15));
    m_yy_t45 = (792*m_yy_I - 108*m_xz_I - 36*m_xx_I - 36*m_zz_I + omegaVar*m_xx_I + 3*omegaVar*m_xz_I + 13*omegaVar*m_yy_I + omegaVar*m_zz_I - 12)/(54*(2*m_xx_I + 6*m_xz_I - m_yy_I + 2*m_zz_I - 2*omegaVar*m_xx_I - 6*omegaVar*m_xz_I + omegaVar*m_yy_I - 2*omegaVar*m_zz_I + 15));
    m_yz_t90 = (m_yz_I*(3*omegaVar + 172))/(12*(2*m_xx_I + 6*m_xz_I - m_yy_I + 2*m_zz_I - 2*omegaVar*m_xx_I - 6*omegaVar*m_xz_I + omegaVar*m_yy_I - 2*omegaVar*m_zz_I + 15));
    m_zz_t45 = (144*m_xx_I + 432*m_xz_I - 72*m_yy_I + 1692*m_zz_I - 4*omegaVar*m_xx_I - 12*omegaVar*m_xz_I + 2*omegaVar*m_yy_I + 23*omegaVar*m_zz_I + 48)/(108*(2*m_xx_I + 6*m_xz_I - m_yy_I + 2*m_zz_I - 2*omegaVar*m_xx_I - 6*omegaVar*m_xz_I + omegaVar*m_yy_I - 2*omegaVar*m_zz_I + 15));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = (m_xx_t45/6 + m_xz_t90/2 - m_yy_t45/12 + m_zz_t45/6 - (omegaVar*m_xx_t45)/12 - (omegaVar*m_xz_t90)/4 + (omegaVar*m_yy_t45)/24 - (omegaVar*m_zz_t45)/12 + 1.0/18.0)*rho - RHO_0/18;
    L_BC_Fy = 0;
    L_BC_Fz = (m_xx_t45/6 + m_xz_t90/2 - m_yy_t45/12 + m_zz_t45/6 - (omegaVar*m_xx_t45)/12 - (omegaVar*m_xz_t90)/4 + (omegaVar*m_yy_t45)/24 - (omegaVar*m_zz_t45)/12 + 1.0/18.0)*rho - RHO_0/18;
    #endif //_BC_FORCES 

    break;
case 0b0000010110100010:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[11] + pop[12] + pop[13] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13]);
    m_xz_I = inv_rho_I * (pop[9] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (6*(15*rho_I + 4*m_xx_I*rho_I - 6*m_xy_I*rho_I + 6*m_xz_I*rho_I + m_yy_I*rho_I + m_zz_I*rho_I - 4*omegaVar*m_xx_I*rho_I + 6*omegaVar*m_xy_I*rho_I - 6*omegaVar*m_xz_I*rho_I - omegaVar*m_yy_I*rho_I - omegaVar*m_zz_I*rho_I))/(3*omegaVar + 82);

    m_xx_t45 = (882*m_xx_I - 216*m_xy_I + 216*m_xz_I + 36*m_yy_I + 36*m_zz_I + 19*omegaVar*m_xx_I + 12*omegaVar*m_xy_I - 12*omegaVar*m_xz_I - 2*omegaVar*m_yy_I - 2*omegaVar*m_zz_I + 48)/(54*(4*m_xx_I - 6*m_xy_I + 6*m_xz_I + m_yy_I + m_zz_I - 4*omegaVar*m_xx_I + 6*omegaVar*m_xy_I - 6*omegaVar*m_xz_I - omegaVar*m_yy_I - omegaVar*m_zz_I + 15));
    m_xy_t90 = (1746*m_xy_I - 180*m_xx_I - 24*m_xz_I - 168*m_yy_I + 78*m_zz_I + 10*omegaVar*m_xx_I + 39*omegaVar*m_xy_I + 24*omegaVar*m_xz_I - 2*omegaVar*m_yy_I + 7*omegaVar*m_zz_I - 60)/(90*(4*m_xx_I - 6*m_xy_I + 6*m_xz_I + m_yy_I + m_zz_I - 4*omegaVar*m_xx_I + 6*omegaVar*m_xy_I - 6*omegaVar*m_xz_I - omegaVar*m_yy_I - omegaVar*m_zz_I + 15));
    m_xz_t90 = (180*m_xx_I - 24*m_xy_I + 1746*m_xz_I - 78*m_yy_I + 168*m_zz_I - 10*omegaVar*m_xx_I + 24*omegaVar*m_xy_I + 39*omegaVar*m_xz_I - 7*omegaVar*m_yy_I + 2*omegaVar*m_zz_I + 60)/(90*(4*m_xx_I - 6*m_xy_I + 6*m_xz_I + m_yy_I + m_zz_I - 4*omegaVar*m_xx_I + 6*omegaVar*m_xy_I - 6*omegaVar*m_xz_I - omegaVar*m_yy_I - omegaVar*m_zz_I + 15));
    m_yy_t45 = -(504*m_xy_I - 90*m_xx_I + 234*m_xz_I - 2052*m_yy_I + 162*m_zz_I + 5*omegaVar*m_xx_I + 6*omegaVar*m_xy_I + 21*omegaVar*m_xz_I - 73*omegaVar*m_yy_I + 8*omegaVar*m_zz_I - 30)/(135*(4*m_xx_I - 6*m_xy_I + 6*m_xz_I + m_yy_I + m_zz_I - 4*omegaVar*m_xx_I + 6*omegaVar*m_xy_I - 6*omegaVar*m_xz_I - omegaVar*m_yy_I - omegaVar*m_zz_I + 15));
    m_yz_t90 = (m_yz_I*(3*omegaVar + 82))/(6*(4*m_xx_I - 6*m_xy_I + 6*m_xz_I + m_yy_I + m_zz_I - 4*omegaVar*m_xx_I + 6*omegaVar*m_xy_I - 6*omegaVar*m_xz_I - omegaVar*m_yy_I - omegaVar*m_zz_I + 15));
    m_zz_t45 = (90*m_xx_I + 234*m_xy_I + 504*m_xz_I - 162*m_yy_I + 2052*m_zz_I - 5*omegaVar*m_xx_I + 21*omegaVar*m_xy_I + 6*omegaVar*m_xz_I - 8*omegaVar*m_yy_I + 73*omegaVar*m_zz_I + 30)/(135*(4*m_xx_I - 6*m_xy_I + 6*m_xz_I + m_yy_I + m_zz_I - 4*omegaVar*m_xx_I + 6*omegaVar*m_xy_I - 6*omegaVar*m_xz_I - omegaVar*m_yy_I - omegaVar*m_zz_I + 15));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = (m_xx_t45/3 - m_xy_t90/2 + m_xz_t90/2 + m_yy_t45/12 + m_zz_t45/12 - (omegaVar*m_xx_t45)/6 + (omegaVar*m_xy_t90)/4 - (omegaVar*m_xz_t90)/4 - (omegaVar*m_yy_t45)/24 - (omegaVar*m_zz_t45)/24 + 1.0/9.0)*rho - RHO_0/9;
    L_BC_Fy = (m_xy_t90/2 - m_xx_t45/6 - m_yy_t45/6 + m_zz_t45/12 + (omegaVar*m_xx_t45)/12 - (omegaVar*m_xy_t90)/4 + (omegaVar*m_yy_t45)/12 - (omegaVar*m_zz_t45)/24 - 1.0/18.0)*rho + RHO_0/18;
    L_BC_Fz = (m_xx_t45/6 + m_xz_t90/2 - m_yy_t45/12 + m_zz_t45/6 - (omegaVar*m_xx_t45)/12 - (omegaVar*m_xz_t90)/4 + (omegaVar*m_yy_t45)/24 - (omegaVar*m_zz_t45)/12 + 1.0/18.0)*rho - RHO_0/18;
    #endif //_BC_FORCES 

    break;
case 0b0000010110100011:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[12] + pop[13] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13]);
    m_xz_I = inv_rho_I * (pop[9] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (12*(5*rho_I + m_xx_I*rho_I - 2*m_xy_I*rho_I + 2*m_xz_I*rho_I + m_yy_I*rho_I + 2*m_yz_I*rho_I + m_zz_I*rho_I - omegaVar*m_xx_I*rho_I + 2*omegaVar*m_xy_I*rho_I - 2*omegaVar*m_xz_I*rho_I - omegaVar*m_yy_I*rho_I - 2*omegaVar*m_yz_I*rho_I - omegaVar*m_zz_I*rho_I))/(3*omegaVar + 52);

    m_xx_t45 = (472*m_xx_I - 112*m_xy_I + 112*m_xz_I + 4*m_yy_I - 44*m_yz_I + 4*m_zz_I + 23*omegaVar*m_xx_I + 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - 4*omegaVar*m_yy_I - 11*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 20)/(90*(m_xx_I - 2*m_xy_I + 2*m_xz_I + m_yy_I + 2*m_yz_I + m_zz_I - omegaVar*m_xx_I + 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - omegaVar*m_yy_I - 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_xy_t90 = (1108*m_xy_I - 112*m_xx_I - 16*m_xz_I - 112*m_yy_I - 16*m_yz_I + 44*m_zz_I + 2*omegaVar*m_xx_I + 47*omegaVar*m_xy_I + 16*omegaVar*m_xz_I + 2*omegaVar*m_yy_I + 16*omegaVar*m_yz_I + 11*omegaVar*m_zz_I - 40)/(180*(m_xx_I - 2*m_xy_I + 2*m_xz_I + m_yy_I + 2*m_yz_I + m_zz_I - omegaVar*m_xx_I + 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - omegaVar*m_yy_I - 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_xz_t90 = (112*m_xx_I - 16*m_xy_I + 1108*m_xz_I - 44*m_yy_I + 16*m_yz_I + 112*m_zz_I - 2*omegaVar*m_xx_I + 16*omegaVar*m_xy_I + 47*omegaVar*m_xz_I - 11*omegaVar*m_yy_I - 16*omegaVar*m_yz_I - 2*omegaVar*m_zz_I + 40)/(180*(m_xx_I - 2*m_xy_I + 2*m_xz_I + m_yy_I + 2*m_yz_I + m_zz_I - omegaVar*m_xx_I + 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - omegaVar*m_yy_I - 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_yy_t45 = (4*m_xx_I - 112*m_xy_I - 44*m_xz_I + 472*m_yy_I + 112*m_yz_I + 4*m_zz_I - 4*omegaVar*m_xx_I + 2*omegaVar*m_xy_I - 11*omegaVar*m_xz_I + 23*omegaVar*m_yy_I - 2*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 20)/(90*(m_xx_I - 2*m_xy_I + 2*m_xz_I + m_yy_I + 2*m_yz_I + m_zz_I - omegaVar*m_xx_I + 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - omegaVar*m_yy_I - 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_yz_t90 = (16*m_xz_I - 16*m_xy_I - 44*m_xx_I + 112*m_yy_I + 1108*m_yz_I + 112*m_zz_I - 11*omegaVar*m_xx_I + 16*omegaVar*m_xy_I - 16*omegaVar*m_xz_I - 2*omegaVar*m_yy_I + 47*omegaVar*m_yz_I - 2*omegaVar*m_zz_I + 40)/(180*(m_xx_I - 2*m_xy_I + 2*m_xz_I + m_yy_I + 2*m_yz_I + m_zz_I - omegaVar*m_xx_I + 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - omegaVar*m_yy_I - 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_zz_t45 = (4*m_xx_I + 44*m_xy_I + 112*m_xz_I + 4*m_yy_I + 112*m_yz_I + 472*m_zz_I - 4*omegaVar*m_xx_I + 11*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - 4*omegaVar*m_yy_I - 2*omegaVar*m_yz_I + 23*omegaVar*m_zz_I + 20)/(90*(m_xx_I - 2*m_xy_I + 2*m_xz_I + m_yy_I + 2*m_yz_I + m_zz_I - omegaVar*m_xx_I + 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - omegaVar*m_yy_I - 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = (m_xx_t45/3 - m_xy_t90/2 + m_xz_t90/2 + m_yy_t45/12 + m_zz_t45/12 - (omegaVar*m_xx_t45)/6 + (omegaVar*m_xy_t90)/4 - (omegaVar*m_xz_t90)/4 - (omegaVar*m_yy_t45)/24 - (omegaVar*m_zz_t45)/24 + 1.0/9.0)*rho - RHO_0/9;
    L_BC_Fy = (m_xy_t90/2 - m_xx_t45/12 - m_yy_t45/3 - m_yz_t90/2 - m_zz_t45/12 + (omegaVar*m_xx_t45)/24 - (omegaVar*m_xy_t90)/4 + (omegaVar*m_yy_t45)/6 + (omegaVar*m_yz_t90)/4 + (omegaVar*m_zz_t45)/24 - 1.0/9.0)*rho + RHO_0/9;
    L_BC_Fz = -(rho*(omegaVar - 2)*(m_xx_t45 + 2*m_xz_t90 - m_yy_t45 - 2*m_yz_t90))/8;
    #endif //_BC_FORCES 

    break;
case 0b0000010110100100:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (12*(15*rho_I + 2*m_xx_I*rho_I + 6*m_xz_I*rho_I - m_yy_I*rho_I + 2*m_zz_I*rho_I - 2*omegaVar*m_xx_I*rho_I - 6*omegaVar*m_xz_I*rho_I + omegaVar*m_yy_I*rho_I - 2*omegaVar*m_zz_I*rho_I))/(3*omegaVar + 172);

    m_xx_t45 = (1692*m_xx_I + 432*m_xz_I - 72*m_yy_I + 144*m_zz_I + 23*omegaVar*m_xx_I - 12*omegaVar*m_xz_I + 2*omegaVar*m_yy_I - 4*omegaVar*m_zz_I + 48)/(108*(2*m_xx_I + 6*m_xz_I - m_yy_I + 2*m_zz_I - 2*omegaVar*m_xx_I - 6*omegaVar*m_xz_I + omegaVar*m_yy_I - 2*omegaVar*m_zz_I + 15));
    m_xy_t90 = (m_xy_I*(3*omegaVar + 172))/(12*(2*m_xx_I + 6*m_xz_I - m_yy_I + 2*m_zz_I - 2*omegaVar*m_xx_I - 6*omegaVar*m_xz_I + omegaVar*m_yy_I - 2*omegaVar*m_zz_I + 15));
    m_xz_t90 = (72*m_xx_I + 732*m_xz_I - 36*m_yy_I + 72*m_zz_I - 2*omegaVar*m_xx_I + 3*omegaVar*m_xz_I + omegaVar*m_yy_I - 2*omegaVar*m_zz_I + 24)/(36*(2*m_xx_I + 6*m_xz_I - m_yy_I + 2*m_zz_I - 2*omegaVar*m_xx_I - 6*omegaVar*m_xz_I + omegaVar*m_yy_I - 2*omegaVar*m_zz_I + 15));
    m_yy_t45 = (792*m_yy_I - 108*m_xz_I - 36*m_xx_I - 36*m_zz_I + omegaVar*m_xx_I + 3*omegaVar*m_xz_I + 13*omegaVar*m_yy_I + omegaVar*m_zz_I - 12)/(54*(2*m_xx_I + 6*m_xz_I - m_yy_I + 2*m_zz_I - 2*omegaVar*m_xx_I - 6*omegaVar*m_xz_I + omegaVar*m_yy_I - 2*omegaVar*m_zz_I + 15));
    m_yz_t90 = (m_yz_I*(3*omegaVar + 172))/(12*(2*m_xx_I + 6*m_xz_I - m_yy_I + 2*m_zz_I - 2*omegaVar*m_xx_I - 6*omegaVar*m_xz_I + omegaVar*m_yy_I - 2*omegaVar*m_zz_I + 15));
    m_zz_t45 = (144*m_xx_I + 432*m_xz_I - 72*m_yy_I + 1692*m_zz_I - 4*omegaVar*m_xx_I - 12*omegaVar*m_xz_I + 2*omegaVar*m_yy_I + 23*omegaVar*m_zz_I + 48)/(108*(2*m_xx_I + 6*m_xz_I - m_yy_I + 2*m_zz_I - 2*omegaVar*m_xx_I - 6*omegaVar*m_xz_I + omegaVar*m_yy_I - 2*omegaVar*m_zz_I + 15));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = (m_xx_t45/6 + m_xz_t90/2 - m_yy_t45/12 + m_zz_t45/6 - (omegaVar*m_xx_t45)/12 - (omegaVar*m_xz_t90)/4 + (omegaVar*m_yy_t45)/24 - (omegaVar*m_zz_t45)/12 + 1.0/18.0)*rho - RHO_0/18;
    L_BC_Fy = 0;
    L_BC_Fz = (m_xx_t45/6 + m_xz_t90/2 - m_yy_t45/12 + m_zz_t45/6 - (omegaVar*m_xx_t45)/12 - (omegaVar*m_xz_t90)/4 + (omegaVar*m_yy_t45)/24 - (omegaVar*m_zz_t45)/12 + 1.0/18.0)*rho - RHO_0/18;
    #endif //_BC_FORCES 

    break;
case 0b0000010110100101:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (-pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (6*(3*rho_I + 2*m_xx_I*rho_I + 6*m_xz_I*rho_I - m_yy_I*rho_I + 2*m_zz_I*rho_I - 2*omegaVar*m_xx_I*rho_I - 6*omegaVar*m_xz_I*rho_I + omegaVar*m_yy_I*rho_I - 2*omegaVar*m_zz_I*rho_I))/(3*omegaVar + 14);

    m_xx_t45 = (198*m_xx_I + 216*m_xz_I - 36*m_yy_I + 72*m_zz_I + 23*omegaVar*m_xx_I - 12*omegaVar*m_xz_I + 2*omegaVar*m_yy_I - 4*omegaVar*m_zz_I + 24)/(54*(2*m_xx_I + 6*m_xz_I - m_yy_I + 2*m_zz_I - 2*omegaVar*m_xx_I - 6*omegaVar*m_xz_I + omegaVar*m_yy_I - 2*omegaVar*m_zz_I + 3));
    m_xy_t90 = (m_xy_I*(3*omegaVar + 14))/(6*(2*m_xx_I + 6*m_xz_I - m_yy_I + 2*m_zz_I - 2*omegaVar*m_xx_I - 6*omegaVar*m_xz_I + omegaVar*m_yy_I - 2*omegaVar*m_zz_I + 3));
    m_xz_t90 = (36*m_xx_I + 150*m_xz_I - 18*m_yy_I + 36*m_zz_I - 2*omegaVar*m_xx_I + 3*omegaVar*m_xz_I + omegaVar*m_yy_I - 2*omegaVar*m_zz_I + 12)/(18*(2*m_xx_I + 6*m_xz_I - m_yy_I + 2*m_zz_I - 2*omegaVar*m_xx_I - 6*omegaVar*m_xz_I + omegaVar*m_yy_I - 2*omegaVar*m_zz_I + 3));
    m_yy_t45 = (72*m_yy_I - 54*m_xz_I - 18*m_xx_I - 18*m_zz_I + omegaVar*m_xx_I + 3*omegaVar*m_xz_I + 13*omegaVar*m_yy_I + omegaVar*m_zz_I - 6)/(27*(2*m_xx_I + 6*m_xz_I - m_yy_I + 2*m_zz_I - 2*omegaVar*m_xx_I - 6*omegaVar*m_xz_I + omegaVar*m_yy_I - 2*omegaVar*m_zz_I + 3));
    m_yz_t90 = (m_yz_I*(3*omegaVar + 14))/(6*(2*m_xx_I + 6*m_xz_I - m_yy_I + 2*m_zz_I - 2*omegaVar*m_xx_I - 6*omegaVar*m_xz_I + omegaVar*m_yy_I - 2*omegaVar*m_zz_I + 3));
    m_zz_t45 = (72*m_xx_I + 216*m_xz_I - 36*m_yy_I + 198*m_zz_I - 4*omegaVar*m_xx_I - 12*omegaVar*m_xz_I + 2*omegaVar*m_yy_I + 23*omegaVar*m_zz_I + 24)/(54*(2*m_xx_I + 6*m_xz_I - m_yy_I + 2*m_zz_I - 2*omegaVar*m_xx_I - 6*omegaVar*m_xz_I + omegaVar*m_yy_I - 2*omegaVar*m_zz_I + 3));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = 0;
    L_BC_Fy = 0;
    L_BC_Fz = 0;
    #endif //_BC_FORCES 

    break;
case 0b0000010110100110:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[11] + pop[12] + pop[13] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13]);
    m_xz_I = inv_rho_I * (pop[9] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (6*(15*rho_I + 4*m_xx_I*rho_I - 6*m_xy_I*rho_I + 6*m_xz_I*rho_I + m_yy_I*rho_I + m_zz_I*rho_I - 4*omegaVar*m_xx_I*rho_I + 6*omegaVar*m_xy_I*rho_I - 6*omegaVar*m_xz_I*rho_I - omegaVar*m_yy_I*rho_I - omegaVar*m_zz_I*rho_I))/(3*omegaVar + 82);

    m_xx_t45 = (882*m_xx_I - 216*m_xy_I + 216*m_xz_I + 36*m_yy_I + 36*m_zz_I + 19*omegaVar*m_xx_I + 12*omegaVar*m_xy_I - 12*omegaVar*m_xz_I - 2*omegaVar*m_yy_I - 2*omegaVar*m_zz_I + 48)/(54*(4*m_xx_I - 6*m_xy_I + 6*m_xz_I + m_yy_I + m_zz_I - 4*omegaVar*m_xx_I + 6*omegaVar*m_xy_I - 6*omegaVar*m_xz_I - omegaVar*m_yy_I - omegaVar*m_zz_I + 15));
    m_xy_t90 = (1746*m_xy_I - 180*m_xx_I - 24*m_xz_I - 168*m_yy_I + 78*m_zz_I + 10*omegaVar*m_xx_I + 39*omegaVar*m_xy_I + 24*omegaVar*m_xz_I - 2*omegaVar*m_yy_I + 7*omegaVar*m_zz_I - 60)/(90*(4*m_xx_I - 6*m_xy_I + 6*m_xz_I + m_yy_I + m_zz_I - 4*omegaVar*m_xx_I + 6*omegaVar*m_xy_I - 6*omegaVar*m_xz_I - omegaVar*m_yy_I - omegaVar*m_zz_I + 15));
    m_xz_t90 = (180*m_xx_I - 24*m_xy_I + 1746*m_xz_I - 78*m_yy_I + 168*m_zz_I - 10*omegaVar*m_xx_I + 24*omegaVar*m_xy_I + 39*omegaVar*m_xz_I - 7*omegaVar*m_yy_I + 2*omegaVar*m_zz_I + 60)/(90*(4*m_xx_I - 6*m_xy_I + 6*m_xz_I + m_yy_I + m_zz_I - 4*omegaVar*m_xx_I + 6*omegaVar*m_xy_I - 6*omegaVar*m_xz_I - omegaVar*m_yy_I - omegaVar*m_zz_I + 15));
    m_yy_t45 = -(504*m_xy_I - 90*m_xx_I + 234*m_xz_I - 2052*m_yy_I + 162*m_zz_I + 5*omegaVar*m_xx_I + 6*omegaVar*m_xy_I + 21*omegaVar*m_xz_I - 73*omegaVar*m_yy_I + 8*omegaVar*m_zz_I - 30)/(135*(4*m_xx_I - 6*m_xy_I + 6*m_xz_I + m_yy_I + m_zz_I - 4*omegaVar*m_xx_I + 6*omegaVar*m_xy_I - 6*omegaVar*m_xz_I - omegaVar*m_yy_I - omegaVar*m_zz_I + 15));
    m_yz_t90 = (m_yz_I*(3*omegaVar + 82))/(6*(4*m_xx_I - 6*m_xy_I + 6*m_xz_I + m_yy_I + m_zz_I - 4*omegaVar*m_xx_I + 6*omegaVar*m_xy_I - 6*omegaVar*m_xz_I - omegaVar*m_yy_I - omegaVar*m_zz_I + 15));
    m_zz_t45 = (90*m_xx_I + 234*m_xy_I + 504*m_xz_I - 162*m_yy_I + 2052*m_zz_I - 5*omegaVar*m_xx_I + 21*omegaVar*m_xy_I + 6*omegaVar*m_xz_I - 8*omegaVar*m_yy_I + 73*omegaVar*m_zz_I + 30)/(135*(4*m_xx_I - 6*m_xy_I + 6*m_xz_I + m_yy_I + m_zz_I - 4*omegaVar*m_xx_I + 6*omegaVar*m_xy_I - 6*omegaVar*m_xz_I - omegaVar*m_yy_I - omegaVar*m_zz_I + 15));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = (m_xx_t45/3 - m_xy_t90/2 + m_xz_t90/2 + m_yy_t45/12 + m_zz_t45/12 - (omegaVar*m_xx_t45)/6 + (omegaVar*m_xy_t90)/4 - (omegaVar*m_xz_t90)/4 - (omegaVar*m_yy_t45)/24 - (omegaVar*m_zz_t45)/24 + 1.0/9.0)*rho - RHO_0/9;
    L_BC_Fy = (m_xy_t90/2 - m_xx_t45/6 - m_yy_t45/6 + m_zz_t45/12 + (omegaVar*m_xx_t45)/12 - (omegaVar*m_xy_t90)/4 + (omegaVar*m_yy_t45)/12 - (omegaVar*m_zz_t45)/24 - 1.0/18.0)*rho + RHO_0/18;
    L_BC_Fz = (m_xx_t45/6 + m_xz_t90/2 - m_yy_t45/12 + m_zz_t45/6 - (omegaVar*m_xx_t45)/12 - (omegaVar*m_xz_t90)/4 + (omegaVar*m_yy_t45)/24 - (omegaVar*m_zz_t45)/12 + 1.0/18.0)*rho - RHO_0/18;
    #endif //_BC_FORCES 

    break;
case 0b0000010110100111:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[12] + pop[13] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13]);
    m_xz_I = inv_rho_I * (-pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (3*(15*rho_I + 11*m_xx_I*rho_I - 6*m_xy_I*rho_I + 30*m_xz_I*rho_I - m_yy_I*rho_I + 6*m_yz_I*rho_I + 11*m_zz_I*rho_I - 11*omegaVar*m_xx_I*rho_I + 6*omegaVar*m_xy_I*rho_I - 30*omegaVar*m_xz_I*rho_I + omegaVar*m_yy_I*rho_I - 6*omegaVar*m_yz_I*rho_I - 11*omegaVar*m_zz_I*rho_I))/(9*omegaVar + 31);

    m_xx_t45 = -(2*(252*m_xy_I - 1206*m_xx_I - 1260*m_xz_I + 135*m_yy_I + 27*m_yz_I - 369*m_zz_I - 194*omegaVar*m_xx_I - 12*omegaVar*m_xy_I + 60*omegaVar*m_xz_I + 25*omegaVar*m_yy_I + 93*omegaVar*m_yz_I + 49*omegaVar*m_zz_I - 165))/(135*(11*m_xx_I - 6*m_xy_I + 30*m_xz_I - m_yy_I + 6*m_yz_I + 11*m_zz_I - 11*omegaVar*m_xx_I + 6*omegaVar*m_xy_I - 30*omegaVar*m_xz_I + omegaVar*m_yy_I - 6*omegaVar*m_yz_I - 11*omegaVar*m_zz_I + 15));
    m_xy_t90 = (663*m_xy_I - 84*m_xx_I - 60*m_xz_I - 60*m_yy_I - 12*m_yz_I + 9*m_zz_I + 4*omegaVar*m_xx_I + 177*omegaVar*m_xy_I + 60*omegaVar*m_xz_I - 20*omegaVar*m_yy_I + 12*omegaVar*m_yz_I + 31*omegaVar*m_zz_I - 30)/(45*(11*m_xx_I - 6*m_xy_I + 30*m_xz_I - m_yy_I + 6*m_yz_I + 11*m_zz_I - 11*omegaVar*m_xx_I + 6*omegaVar*m_xy_I - 30*omegaVar*m_xz_I + omegaVar*m_yy_I - 6*omegaVar*m_yz_I - 11*omegaVar*m_zz_I + 15));
    m_xz_t90 = (84*m_xx_I - 12*m_xy_I + 339*m_xz_I - 33*m_yy_I + 12*m_yz_I + 84*m_zz_I - 4*omegaVar*m_xx_I + 12*omegaVar*m_xy_I + 21*omegaVar*m_xz_I - 7*omegaVar*m_yy_I - 12*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 30)/(9*(11*m_xx_I - 6*m_xy_I + 30*m_xz_I - m_yy_I + 6*m_yz_I + 11*m_zz_I - 11*omegaVar*m_xx_I + 6*omegaVar*m_xy_I - 30*omegaVar*m_xz_I + omegaVar*m_yy_I - 6*omegaVar*m_yz_I - 11*omegaVar*m_zz_I + 15));
    m_yy_t45 = -(2*(27*m_xx_I + 36*m_xy_I + 99*m_xz_I - 180*m_yy_I - 36*m_yz_I + 27*m_zz_I + 5*omegaVar*m_xx_I + 12*omegaVar*m_xy_I + 21*omegaVar*m_xz_I - 52*omegaVar*m_yy_I - 12*omegaVar*m_yz_I + 5*omegaVar*m_zz_I + 3))/(27*(11*m_xx_I - 6*m_xy_I + 30*m_xz_I - m_yy_I + 6*m_yz_I + 11*m_zz_I - 11*omegaVar*m_xx_I + 6*omegaVar*m_xy_I - 30*omegaVar*m_xz_I + omegaVar*m_yy_I - 6*omegaVar*m_yz_I - 11*omegaVar*m_zz_I + 15));
    m_yz_t90 = (60*m_xz_I - 12*m_xy_I - 9*m_xx_I + 60*m_yy_I + 663*m_yz_I + 84*m_zz_I - 31*omegaVar*m_xx_I + 12*omegaVar*m_xy_I - 60*omegaVar*m_xz_I + 20*omegaVar*m_yy_I + 177*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 30)/(45*(11*m_xx_I - 6*m_xy_I + 30*m_xz_I - m_yy_I + 6*m_yz_I + 11*m_zz_I - 11*omegaVar*m_xx_I + 6*omegaVar*m_xy_I - 30*omegaVar*m_xz_I + omegaVar*m_yy_I - 6*omegaVar*m_yz_I - 11*omegaVar*m_zz_I + 15));
    m_zz_t45 = (2*(369*m_xx_I + 27*m_xy_I + 1260*m_xz_I - 135*m_yy_I + 252*m_yz_I + 1206*m_zz_I - 49*omegaVar*m_xx_I + 93*omegaVar*m_xy_I - 60*omegaVar*m_xz_I - 25*omegaVar*m_yy_I - 12*omegaVar*m_yz_I + 194*omegaVar*m_zz_I + 165))/(135*(11*m_xx_I - 6*m_xy_I + 30*m_xz_I - m_yy_I + 6*m_yz_I + 11*m_zz_I - 11*omegaVar*m_xx_I + 6*omegaVar*m_xy_I - 30*omegaVar*m_xz_I + omegaVar*m_yy_I - 6*omegaVar*m_yz_I - 11*omegaVar*m_zz_I + 15));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = (m_xx_t45/6 - m_xy_t90/2 + m_yy_t45/6 - m_zz_t45/12 - (omegaVar*m_xx_t45)/12 + (omegaVar*m_xy_t90)/4 - (omegaVar*m_yy_t45)/12 + (omegaVar*m_zz_t45)/24 + 1.0/18.0)*rho - RHO_0/18;
    L_BC_Fy = (m_xy_t90/2 - m_xx_t45/12 - m_yy_t45/3 - m_yz_t90/2 - m_zz_t45/12 + (omegaVar*m_xx_t45)/24 - (omegaVar*m_xy_t90)/4 + (omegaVar*m_yy_t45)/6 + (omegaVar*m_yz_t90)/4 + (omegaVar*m_zz_t45)/24 - 1.0/9.0)*rho + RHO_0/9;
    L_BC_Fz = (m_xx_t45/12 - m_yy_t45/6 - m_yz_t90/2 - m_zz_t45/6 - (omegaVar*m_xx_t45)/24 + (omegaVar*m_yy_t45)/12 + (omegaVar*m_yz_t90)/4 + (omegaVar*m_zz_t45)/12 - 1.0/18.0)*rho + RHO_0/18;
    #endif //_BC_FORCES 

    break;
case 0b0000010110101000:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[9] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[9] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] - (1.0/3.0)*pop[9] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (6*(15*rho_I + 4*m_xx_I*rho_I + 6*m_xy_I*rho_I + 6*m_xz_I*rho_I + m_yy_I*rho_I + m_zz_I*rho_I - 4*omegaVar*m_xx_I*rho_I - 6*omegaVar*m_xy_I*rho_I - 6*omegaVar*m_xz_I*rho_I - omegaVar*m_yy_I*rho_I - omegaVar*m_zz_I*rho_I))/(3*omegaVar + 82);

    m_xx_t45 = (882*m_xx_I + 216*m_xy_I + 216*m_xz_I + 36*m_yy_I + 36*m_zz_I + 19*omegaVar*m_xx_I - 12*omegaVar*m_xy_I - 12*omegaVar*m_xz_I - 2*omegaVar*m_yy_I - 2*omegaVar*m_zz_I + 48)/(54*(4*m_xx_I + 6*m_xy_I + 6*m_xz_I + m_yy_I + m_zz_I - 4*omegaVar*m_xx_I - 6*omegaVar*m_xy_I - 6*omegaVar*m_xz_I - omegaVar*m_yy_I - omegaVar*m_zz_I + 15));
    m_xy_t90 = (180*m_xx_I + 1746*m_xy_I + 24*m_xz_I + 168*m_yy_I - 78*m_zz_I - 10*omegaVar*m_xx_I + 39*omegaVar*m_xy_I - 24*omegaVar*m_xz_I + 2*omegaVar*m_yy_I - 7*omegaVar*m_zz_I + 60)/(90*(4*m_xx_I + 6*m_xy_I + 6*m_xz_I + m_yy_I + m_zz_I - 4*omegaVar*m_xx_I - 6*omegaVar*m_xy_I - 6*omegaVar*m_xz_I - omegaVar*m_yy_I - omegaVar*m_zz_I + 15));
    m_xz_t90 = (180*m_xx_I + 24*m_xy_I + 1746*m_xz_I - 78*m_yy_I + 168*m_zz_I - 10*omegaVar*m_xx_I - 24*omegaVar*m_xy_I + 39*omegaVar*m_xz_I - 7*omegaVar*m_yy_I + 2*omegaVar*m_zz_I + 60)/(90*(4*m_xx_I + 6*m_xy_I + 6*m_xz_I + m_yy_I + m_zz_I - 4*omegaVar*m_xx_I - 6*omegaVar*m_xy_I - 6*omegaVar*m_xz_I - omegaVar*m_yy_I - omegaVar*m_zz_I + 15));
    m_yy_t45 = (90*m_xx_I + 504*m_xy_I - 234*m_xz_I + 2052*m_yy_I - 162*m_zz_I - 5*omegaVar*m_xx_I + 6*omegaVar*m_xy_I - 21*omegaVar*m_xz_I + 73*omegaVar*m_yy_I - 8*omegaVar*m_zz_I + 30)/(135*(4*m_xx_I + 6*m_xy_I + 6*m_xz_I + m_yy_I + m_zz_I - 4*omegaVar*m_xx_I - 6*omegaVar*m_xy_I - 6*omegaVar*m_xz_I - omegaVar*m_yy_I - omegaVar*m_zz_I + 15));
    m_yz_t90 = (m_yz_I*(3*omegaVar + 82))/(6*(4*m_xx_I + 6*m_xy_I + 6*m_xz_I + m_yy_I + m_zz_I - 4*omegaVar*m_xx_I - 6*omegaVar*m_xy_I - 6*omegaVar*m_xz_I - omegaVar*m_yy_I - omegaVar*m_zz_I + 15));
    m_zz_t45 = (90*m_xx_I - 234*m_xy_I + 504*m_xz_I - 162*m_yy_I + 2052*m_zz_I - 5*omegaVar*m_xx_I - 21*omegaVar*m_xy_I + 6*omegaVar*m_xz_I - 8*omegaVar*m_yy_I + 73*omegaVar*m_zz_I + 30)/(135*(4*m_xx_I + 6*m_xy_I + 6*m_xz_I + m_yy_I + m_zz_I - 4*omegaVar*m_xx_I - 6*omegaVar*m_xy_I - 6*omegaVar*m_xz_I - omegaVar*m_yy_I - omegaVar*m_zz_I + 15));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = (m_xx_t45/3 + m_xy_t90/2 + m_xz_t90/2 + m_yy_t45/12 + m_zz_t45/12 - (omegaVar*m_xx_t45)/6 - (omegaVar*m_xy_t90)/4 - (omegaVar*m_xz_t90)/4 - (omegaVar*m_yy_t45)/24 - (omegaVar*m_zz_t45)/24 + 1.0/9.0)*rho - RHO_0/9;
    L_BC_Fy = (m_xx_t45/6 + m_xy_t90/2 + m_yy_t45/6 - m_zz_t45/12 - (omegaVar*m_xx_t45)/12 - (omegaVar*m_xy_t90)/4 - (omegaVar*m_yy_t45)/12 + (omegaVar*m_zz_t45)/24 + 1.0/18.0)*rho - RHO_0/18;
    L_BC_Fz = (m_xx_t45/6 + m_xz_t90/2 - m_yy_t45/12 + m_zz_t45/6 - (omegaVar*m_xx_t45)/12 - (omegaVar*m_xz_t90)/4 + (omegaVar*m_yy_t45)/24 - (omegaVar*m_zz_t45)/12 + 1.0/18.0)*rho - RHO_0/18;
    #endif //_BC_FORCES 

    break;
case 0b0000010110101001:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[9] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[9] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] - (1.0/3.0)*pop[9] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (6*(15*rho_I + 4*m_xx_I*rho_I + 6*m_xy_I*rho_I + 6*m_xz_I*rho_I + m_yy_I*rho_I + m_zz_I*rho_I - 4*omegaVar*m_xx_I*rho_I - 6*omegaVar*m_xy_I*rho_I - 6*omegaVar*m_xz_I*rho_I - omegaVar*m_yy_I*rho_I - omegaVar*m_zz_I*rho_I))/(3*omegaVar + 82);

    m_xx_t45 = (882*m_xx_I + 216*m_xy_I + 216*m_xz_I + 36*m_yy_I + 36*m_zz_I + 19*omegaVar*m_xx_I - 12*omegaVar*m_xy_I - 12*omegaVar*m_xz_I - 2*omegaVar*m_yy_I - 2*omegaVar*m_zz_I + 48)/(54*(4*m_xx_I + 6*m_xy_I + 6*m_xz_I + m_yy_I + m_zz_I - 4*omegaVar*m_xx_I - 6*omegaVar*m_xy_I - 6*omegaVar*m_xz_I - omegaVar*m_yy_I - omegaVar*m_zz_I + 15));
    m_xy_t90 = (180*m_xx_I + 1746*m_xy_I + 24*m_xz_I + 168*m_yy_I - 78*m_zz_I - 10*omegaVar*m_xx_I + 39*omegaVar*m_xy_I - 24*omegaVar*m_xz_I + 2*omegaVar*m_yy_I - 7*omegaVar*m_zz_I + 60)/(90*(4*m_xx_I + 6*m_xy_I + 6*m_xz_I + m_yy_I + m_zz_I - 4*omegaVar*m_xx_I - 6*omegaVar*m_xy_I - 6*omegaVar*m_xz_I - omegaVar*m_yy_I - omegaVar*m_zz_I + 15));
    m_xz_t90 = (180*m_xx_I + 24*m_xy_I + 1746*m_xz_I - 78*m_yy_I + 168*m_zz_I - 10*omegaVar*m_xx_I - 24*omegaVar*m_xy_I + 39*omegaVar*m_xz_I - 7*omegaVar*m_yy_I + 2*omegaVar*m_zz_I + 60)/(90*(4*m_xx_I + 6*m_xy_I + 6*m_xz_I + m_yy_I + m_zz_I - 4*omegaVar*m_xx_I - 6*omegaVar*m_xy_I - 6*omegaVar*m_xz_I - omegaVar*m_yy_I - omegaVar*m_zz_I + 15));
    m_yy_t45 = (90*m_xx_I + 504*m_xy_I - 234*m_xz_I + 2052*m_yy_I - 162*m_zz_I - 5*omegaVar*m_xx_I + 6*omegaVar*m_xy_I - 21*omegaVar*m_xz_I + 73*omegaVar*m_yy_I - 8*omegaVar*m_zz_I + 30)/(135*(4*m_xx_I + 6*m_xy_I + 6*m_xz_I + m_yy_I + m_zz_I - 4*omegaVar*m_xx_I - 6*omegaVar*m_xy_I - 6*omegaVar*m_xz_I - omegaVar*m_yy_I - omegaVar*m_zz_I + 15));
    m_yz_t90 = (m_yz_I*(3*omegaVar + 82))/(6*(4*m_xx_I + 6*m_xy_I + 6*m_xz_I + m_yy_I + m_zz_I - 4*omegaVar*m_xx_I - 6*omegaVar*m_xy_I - 6*omegaVar*m_xz_I - omegaVar*m_yy_I - omegaVar*m_zz_I + 15));
    m_zz_t45 = (90*m_xx_I - 234*m_xy_I + 504*m_xz_I - 162*m_yy_I + 2052*m_zz_I - 5*omegaVar*m_xx_I - 21*omegaVar*m_xy_I + 6*omegaVar*m_xz_I - 8*omegaVar*m_yy_I + 73*omegaVar*m_zz_I + 30)/(135*(4*m_xx_I + 6*m_xy_I + 6*m_xz_I + m_yy_I + m_zz_I - 4*omegaVar*m_xx_I - 6*omegaVar*m_xy_I - 6*omegaVar*m_xz_I - omegaVar*m_yy_I - omegaVar*m_zz_I + 15));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = (m_xx_t45/3 + m_xy_t90/2 + m_xz_t90/2 + m_yy_t45/12 + m_zz_t45/12 - (omegaVar*m_xx_t45)/6 - (omegaVar*m_xy_t90)/4 - (omegaVar*m_xz_t90)/4 - (omegaVar*m_yy_t45)/24 - (omegaVar*m_zz_t45)/24 + 1.0/9.0)*rho - RHO_0/9;
    L_BC_Fy = (m_xx_t45/6 + m_xy_t90/2 + m_yy_t45/6 - m_zz_t45/12 - (omegaVar*m_xx_t45)/12 - (omegaVar*m_xy_t90)/4 - (omegaVar*m_yy_t45)/12 + (omegaVar*m_zz_t45)/24 + 1.0/18.0)*rho - RHO_0/18;
    L_BC_Fz = (m_xx_t45/6 + m_xz_t90/2 - m_yy_t45/12 + m_zz_t45/6 - (omegaVar*m_xx_t45)/12 - (omegaVar*m_xz_t90)/4 + (omegaVar*m_yy_t45)/24 - (omegaVar*m_zz_t45)/12 + 1.0/18.0)*rho - RHO_0/18;
    #endif //_BC_FORCES 

    break;
case 0b0000010110101010:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[9] + pop[11] + pop[12] + pop[13] + pop[15] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[9] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[15] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] - pop[13]);
    m_xz_I = inv_rho_I * (pop[9] - pop[15]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] - (1.0/3.0)*pop[9] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] - (1.0/3.0)*pop[15] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (3*rho_I*(3*m_xx_I - 3*omegaVar*m_xx_I + 4))/(omegaVar + 9);

    m_xx_t45 = (15*m_xx_I + 2)/(3*(3*m_xx_I - 3*omegaVar*m_xx_I + 4));
    m_xy_t90 = (2*m_xy_I*(omegaVar + 9))/(3*(3*m_xx_I - 3*omegaVar*m_xx_I + 4));
    m_xz_t90 = (2*m_xz_I*(omegaVar + 9))/(3*(3*m_xx_I - 3*omegaVar*m_xx_I + 4));
    m_yy_t45 = (4*(omegaVar + 9)*(10*m_yy_I - m_zz_I))/(99*(3*m_xx_I - 3*omegaVar*m_xx_I + 4));
    m_yz_t90 = (m_yz_I*(omegaVar + 9))/(3*(3*m_xx_I - 3*omegaVar*m_xx_I + 4));
    m_zz_t45 = -(4*(m_yy_I - 10*m_zz_I)*(omegaVar + 9))/(99*(3*m_xx_I - 3*omegaVar*m_xx_I + 4));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = (m_xx_t45 - (omegaVar*m_xx_t45)/2 + 1.0/3.0)*rho - RHO_0/3;
    L_BC_Fy = -(m_xy_t90*rho*(omegaVar - 2))/2;
    L_BC_Fz = -(m_xz_t90*rho*(omegaVar - 2))/2;
    #endif //_BC_FORCES 

    break;
case 0b0000010110101011:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[9] + pop[12] + pop[13] + pop[15] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[9] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[15] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] - pop[13]);
    m_xz_I = inv_rho_I * (pop[9] - pop[15]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] - (1.0/3.0)*pop[9] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] - (1.0/3.0)*pop[15] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (36*(107*rho_I + 72*m_xx_I*rho_I + 12*m_yy_I*rho_I + 33*m_yz_I*rho_I + 12*m_zz_I*rho_I - 72*omegaVar*m_xx_I*rho_I - 12*omegaVar*m_yy_I*rho_I - 33*omegaVar*m_yz_I*rho_I - 12*omegaVar*m_zz_I*rho_I))/(313*omegaVar + 2790);

    m_xx_t45 = -(72*m_yy_I - 2358*m_xx_I + 198*m_yz_I + 72*m_zz_I - 49*omegaVar*m_xx_I + 44*omegaVar*m_yy_I + 121*omegaVar*m_yz_I + 44*omegaVar*m_zz_I - 288)/(18*(72*m_xx_I + 12*m_yy_I + 33*m_yz_I + 12*m_zz_I - 72*omegaVar*m_xx_I - 12*omegaVar*m_yy_I - 33*omegaVar*m_yz_I - 12*omegaVar*m_zz_I + 107));
    m_xy_t90 = (m_xy_I*(313*omegaVar + 2790))/(18*(72*m_xx_I + 12*m_yy_I + 33*m_yz_I + 12*m_zz_I - 72*omegaVar*m_xx_I - 12*omegaVar*m_yy_I - 33*omegaVar*m_yz_I - 12*omegaVar*m_zz_I + 107));
    m_xz_t90 = (m_xz_I*(313*omegaVar + 2790))/(18*(72*m_xx_I + 12*m_yy_I + 33*m_yz_I + 12*m_zz_I - 72*omegaVar*m_xx_I - 12*omegaVar*m_yy_I - 33*omegaVar*m_yz_I - 12*omegaVar*m_zz_I + 107));
    m_yy_t45 = (2772*m_yy_I - 108*m_xx_I + 648*m_yz_I - 18*m_zz_I - 66*omegaVar*m_xx_I + 302*omegaVar*m_yy_I + 48*omegaVar*m_yz_I - 11*omegaVar*m_zz_I + 72)/(27*(72*m_xx_I + 12*m_yy_I + 33*m_yz_I + 12*m_zz_I - 72*omegaVar*m_xx_I - 12*omegaVar*m_yy_I - 33*omegaVar*m_yz_I - 12*omegaVar*m_zz_I + 107));
    m_yz_t90 = (432*m_yy_I - 198*m_xx_I + 3978*m_yz_I + 432*m_zz_I - 121*omegaVar*m_xx_I + 32*omegaVar*m_yy_I + 401*omegaVar*m_yz_I + 32*omegaVar*m_zz_I + 132)/(36*(72*m_xx_I + 12*m_yy_I + 33*m_yz_I + 12*m_zz_I - 72*omegaVar*m_xx_I - 12*omegaVar*m_yy_I - 33*omegaVar*m_yz_I - 12*omegaVar*m_zz_I + 107));
    m_zz_t45 = (648*m_yz_I - 18*m_yy_I - 108*m_xx_I + 2772*m_zz_I - 66*omegaVar*m_xx_I - 11*omegaVar*m_yy_I + 48*omegaVar*m_yz_I + 302*omegaVar*m_zz_I + 72)/(27*(72*m_xx_I + 12*m_yy_I + 33*m_yz_I + 12*m_zz_I - 72*omegaVar*m_xx_I - 12*omegaVar*m_yy_I - 33*omegaVar*m_yz_I - 12*omegaVar*m_zz_I + 107));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = (m_xx_t45 - (omegaVar*m_xx_t45)/2 + 1.0/3.0)*rho - RHO_0/3;
    L_BC_Fy = (m_xx_t45/12 + m_xy_t90 - m_yy_t45/6 - m_yz_t90/2 - m_zz_t45/6 - (omegaVar*m_xx_t45)/24 - (omegaVar*m_xy_t90)/2 + (omegaVar*m_yy_t45)/12 + (omegaVar*m_yz_t90)/4 + (omegaVar*m_zz_t45)/12 - 1.0/18.0)*rho + RHO_0/18;
    L_BC_Fz = (m_xx_t45/12 + m_xz_t90 - m_yy_t45/6 - m_yz_t90/2 - m_zz_t45/6 - (omegaVar*m_xx_t45)/24 - (omegaVar*m_xz_t90)/2 + (omegaVar*m_yy_t45)/12 + (omegaVar*m_yz_t90)/4 + (omegaVar*m_zz_t45)/12 - 1.0/18.0)*rho + RHO_0/18;
    #endif //_BC_FORCES 

    break;
case 0b0000010110101100:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[9] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[9] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17]);
    m_xy_I = inv_rho_I * (pop[7] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] - (1.0/3.0)*pop[9] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17]);

    rho = (12*(5*rho_I + m_xx_I*rho_I + 2*m_xy_I*rho_I + 2*m_xz_I*rho_I + m_yy_I*rho_I - 2*m_yz_I*rho_I + m_zz_I*rho_I - omegaVar*m_xx_I*rho_I - 2*omegaVar*m_xy_I*rho_I - 2*omegaVar*m_xz_I*rho_I - omegaVar*m_yy_I*rho_I + 2*omegaVar*m_yz_I*rho_I - omegaVar*m_zz_I*rho_I))/(3*omegaVar + 52);

    m_xx_t45 = (472*m_xx_I + 112*m_xy_I + 112*m_xz_I + 4*m_yy_I + 44*m_yz_I + 4*m_zz_I + 23*omegaVar*m_xx_I - 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - 4*omegaVar*m_yy_I + 11*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 20)/(90*(m_xx_I + 2*m_xy_I + 2*m_xz_I + m_yy_I - 2*m_yz_I + m_zz_I - omegaVar*m_xx_I - 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - omegaVar*m_yy_I + 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_xy_t90 = (112*m_xx_I + 1108*m_xy_I + 16*m_xz_I + 112*m_yy_I - 16*m_yz_I - 44*m_zz_I - 2*omegaVar*m_xx_I + 47*omegaVar*m_xy_I - 16*omegaVar*m_xz_I - 2*omegaVar*m_yy_I + 16*omegaVar*m_yz_I - 11*omegaVar*m_zz_I + 40)/(180*(m_xx_I + 2*m_xy_I + 2*m_xz_I + m_yy_I - 2*m_yz_I + m_zz_I - omegaVar*m_xx_I - 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - omegaVar*m_yy_I + 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_xz_t90 = (112*m_xx_I + 16*m_xy_I + 1108*m_xz_I - 44*m_yy_I - 16*m_yz_I + 112*m_zz_I - 2*omegaVar*m_xx_I - 16*omegaVar*m_xy_I + 47*omegaVar*m_xz_I - 11*omegaVar*m_yy_I + 16*omegaVar*m_yz_I - 2*omegaVar*m_zz_I + 40)/(180*(m_xx_I + 2*m_xy_I + 2*m_xz_I + m_yy_I - 2*m_yz_I + m_zz_I - omegaVar*m_xx_I - 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - omegaVar*m_yy_I + 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_yy_t45 = (4*m_xx_I + 112*m_xy_I - 44*m_xz_I + 472*m_yy_I - 112*m_yz_I + 4*m_zz_I - 4*omegaVar*m_xx_I - 2*omegaVar*m_xy_I - 11*omegaVar*m_xz_I + 23*omegaVar*m_yy_I + 2*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 20)/(90*(m_xx_I + 2*m_xy_I + 2*m_xz_I + m_yy_I - 2*m_yz_I + m_zz_I - omegaVar*m_xx_I - 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - omegaVar*m_yy_I + 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_yz_t90 = (44*m_xx_I - 16*m_xy_I - 16*m_xz_I - 112*m_yy_I + 1108*m_yz_I - 112*m_zz_I + 11*omegaVar*m_xx_I + 16*omegaVar*m_xy_I + 16*omegaVar*m_xz_I + 2*omegaVar*m_yy_I + 47*omegaVar*m_yz_I + 2*omegaVar*m_zz_I - 40)/(180*(m_xx_I + 2*m_xy_I + 2*m_xz_I + m_yy_I - 2*m_yz_I + m_zz_I - omegaVar*m_xx_I - 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - omegaVar*m_yy_I + 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_zz_t45 = (4*m_xx_I - 44*m_xy_I + 112*m_xz_I + 4*m_yy_I - 112*m_yz_I + 472*m_zz_I - 4*omegaVar*m_xx_I - 11*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - 4*omegaVar*m_yy_I + 2*omegaVar*m_yz_I + 23*omegaVar*m_zz_I + 20)/(90*(m_xx_I + 2*m_xy_I + 2*m_xz_I + m_yy_I - 2*m_yz_I + m_zz_I - omegaVar*m_xx_I - 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - omegaVar*m_yy_I + 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = (m_xx_t45/3 + m_xy_t90/2 + m_xz_t90/2 + m_yy_t45/12 + m_zz_t45/12 - (omegaVar*m_xx_t45)/6 - (omegaVar*m_xy_t90)/4 - (omegaVar*m_xz_t90)/4 - (omegaVar*m_yy_t45)/24 - (omegaVar*m_zz_t45)/24 + 1.0/9.0)*rho - RHO_0/9;
    L_BC_Fy = (m_xx_t45/12 + m_xy_t90/2 + m_yy_t45/3 - m_yz_t90/2 + m_zz_t45/12 - (omegaVar*m_xx_t45)/24 - (omegaVar*m_xy_t90)/4 - (omegaVar*m_yy_t45)/6 + (omegaVar*m_yz_t90)/4 - (omegaVar*m_zz_t45)/24 + 1.0/9.0)*rho - RHO_0/9;
    L_BC_Fz = -(rho*(omegaVar - 2)*(m_xx_t45 + 2*m_xz_t90 - m_yy_t45 + 2*m_yz_t90))/8;
    #endif //_BC_FORCES 

    break;
case 0b0000010110101101:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17]);
    m_xy_I = inv_rho_I * (pop[7] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (-pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17]);

    rho = (3*(15*rho_I + 11*m_xx_I*rho_I + 6*m_xy_I*rho_I + 30*m_xz_I*rho_I - m_yy_I*rho_I - 6*m_yz_I*rho_I + 11*m_zz_I*rho_I - 11*omegaVar*m_xx_I*rho_I - 6*omegaVar*m_xy_I*rho_I - 30*omegaVar*m_xz_I*rho_I + omegaVar*m_yy_I*rho_I + 6*omegaVar*m_yz_I*rho_I - 11*omegaVar*m_zz_I*rho_I))/(9*omegaVar + 31);

    m_xx_t45 = (2*(1206*m_xx_I + 252*m_xy_I + 1260*m_xz_I - 135*m_yy_I + 27*m_yz_I + 369*m_zz_I + 194*omegaVar*m_xx_I - 12*omegaVar*m_xy_I - 60*omegaVar*m_xz_I - 25*omegaVar*m_yy_I + 93*omegaVar*m_yz_I - 49*omegaVar*m_zz_I + 165))/(135*(11*m_xx_I + 6*m_xy_I + 30*m_xz_I - m_yy_I - 6*m_yz_I + 11*m_zz_I - 11*omegaVar*m_xx_I - 6*omegaVar*m_xy_I - 30*omegaVar*m_xz_I + omegaVar*m_yy_I + 6*omegaVar*m_yz_I - 11*omegaVar*m_zz_I + 15));
    m_xy_t90 = (84*m_xx_I + 663*m_xy_I + 60*m_xz_I + 60*m_yy_I - 12*m_yz_I - 9*m_zz_I - 4*omegaVar*m_xx_I + 177*omegaVar*m_xy_I - 60*omegaVar*m_xz_I + 20*omegaVar*m_yy_I + 12*omegaVar*m_yz_I - 31*omegaVar*m_zz_I + 30)/(45*(11*m_xx_I + 6*m_xy_I + 30*m_xz_I - m_yy_I - 6*m_yz_I + 11*m_zz_I - 11*omegaVar*m_xx_I - 6*omegaVar*m_xy_I - 30*omegaVar*m_xz_I + omegaVar*m_yy_I + 6*omegaVar*m_yz_I - 11*omegaVar*m_zz_I + 15));
    m_xz_t90 = (84*m_xx_I + 12*m_xy_I + 339*m_xz_I - 33*m_yy_I - 12*m_yz_I + 84*m_zz_I - 4*omegaVar*m_xx_I - 12*omegaVar*m_xy_I + 21*omegaVar*m_xz_I - 7*omegaVar*m_yy_I + 12*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 30)/(9*(11*m_xx_I + 6*m_xy_I + 30*m_xz_I - m_yy_I - 6*m_yz_I + 11*m_zz_I - 11*omegaVar*m_xx_I - 6*omegaVar*m_xy_I - 30*omegaVar*m_xz_I + omegaVar*m_yy_I + 6*omegaVar*m_yz_I - 11*omegaVar*m_zz_I + 15));
    m_yy_t45 = -(2*(27*m_xx_I - 36*m_xy_I + 99*m_xz_I - 180*m_yy_I + 36*m_yz_I + 27*m_zz_I + 5*omegaVar*m_xx_I - 12*omegaVar*m_xy_I + 21*omegaVar*m_xz_I - 52*omegaVar*m_yy_I + 12*omegaVar*m_yz_I + 5*omegaVar*m_zz_I + 3))/(27*(11*m_xx_I + 6*m_xy_I + 30*m_xz_I - m_yy_I - 6*m_yz_I + 11*m_zz_I - 11*omegaVar*m_xx_I - 6*omegaVar*m_xy_I - 30*omegaVar*m_xz_I + omegaVar*m_yy_I + 6*omegaVar*m_yz_I - 11*omegaVar*m_zz_I + 15));
    m_yz_t90 = (9*m_xx_I - 12*m_xy_I - 60*m_xz_I - 60*m_yy_I + 663*m_yz_I - 84*m_zz_I + 31*omegaVar*m_xx_I + 12*omegaVar*m_xy_I + 60*omegaVar*m_xz_I - 20*omegaVar*m_yy_I + 177*omegaVar*m_yz_I + 4*omegaVar*m_zz_I - 30)/(45*(11*m_xx_I + 6*m_xy_I + 30*m_xz_I - m_yy_I - 6*m_yz_I + 11*m_zz_I - 11*omegaVar*m_xx_I - 6*omegaVar*m_xy_I - 30*omegaVar*m_xz_I + omegaVar*m_yy_I + 6*omegaVar*m_yz_I - 11*omegaVar*m_zz_I + 15));
    m_zz_t45 = -(2*(27*m_xy_I - 369*m_xx_I - 1260*m_xz_I + 135*m_yy_I + 252*m_yz_I - 1206*m_zz_I + 49*omegaVar*m_xx_I + 93*omegaVar*m_xy_I + 60*omegaVar*m_xz_I + 25*omegaVar*m_yy_I - 12*omegaVar*m_yz_I - 194*omegaVar*m_zz_I - 165))/(135*(11*m_xx_I + 6*m_xy_I + 30*m_xz_I - m_yy_I - 6*m_yz_I + 11*m_zz_I - 11*omegaVar*m_xx_I - 6*omegaVar*m_xy_I - 30*omegaVar*m_xz_I + omegaVar*m_yy_I + 6*omegaVar*m_yz_I - 11*omegaVar*m_zz_I + 15));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = (m_xx_t45/6 + m_xy_t90/2 + m_yy_t45/6 - m_zz_t45/12 - (omegaVar*m_xx_t45)/12 - (omegaVar*m_xy_t90)/4 - (omegaVar*m_yy_t45)/12 + (omegaVar*m_zz_t45)/24 + 1.0/18.0)*rho - RHO_0/18;
    L_BC_Fy = (m_xx_t45/12 + m_xy_t90/2 + m_yy_t45/3 - m_yz_t90/2 + m_zz_t45/12 - (omegaVar*m_xx_t45)/24 - (omegaVar*m_xy_t90)/4 - (omegaVar*m_yy_t45)/6 + (omegaVar*m_yz_t90)/4 - (omegaVar*m_zz_t45)/24 + 1.0/9.0)*rho - RHO_0/9;
    L_BC_Fz = (m_xx_t45/12 - m_yy_t45/6 + m_yz_t90/2 - m_zz_t45/6 - (omegaVar*m_xx_t45)/24 + (omegaVar*m_yy_t45)/12 - (omegaVar*m_yz_t90)/4 + (omegaVar*m_zz_t45)/12 - 1.0/18.0)*rho + RHO_0/18;
    #endif //_BC_FORCES 

    break;
case 0b0000010110101110:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[9] + pop[11] + pop[12] + pop[13] + pop[15] + pop[17];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[9] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[15] - (1.0/3.0)*pop[17]);
    m_xy_I = inv_rho_I * (pop[7] - pop[13]);
    m_xz_I = inv_rho_I * (pop[9] - pop[15]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] - (1.0/3.0)*pop[9] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] - (1.0/3.0)*pop[15] + (2.0/3.0)*pop[17]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[17]);

    rho = (36*(107*rho_I + 72*m_xx_I*rho_I + 12*m_yy_I*rho_I - 33*m_yz_I*rho_I + 12*m_zz_I*rho_I - 72*omegaVar*m_xx_I*rho_I - 12*omegaVar*m_yy_I*rho_I + 33*omegaVar*m_yz_I*rho_I - 12*omegaVar*m_zz_I*rho_I))/(313*omegaVar + 2790);

    m_xx_t45 = (2358*m_xx_I - 72*m_yy_I + 198*m_yz_I - 72*m_zz_I + 49*omegaVar*m_xx_I - 44*omegaVar*m_yy_I + 121*omegaVar*m_yz_I - 44*omegaVar*m_zz_I + 288)/(18*(72*m_xx_I + 12*m_yy_I - 33*m_yz_I + 12*m_zz_I - 72*omegaVar*m_xx_I - 12*omegaVar*m_yy_I + 33*omegaVar*m_yz_I - 12*omegaVar*m_zz_I + 107));
    m_xy_t90 = (m_xy_I*(313*omegaVar + 2790))/(18*(72*m_xx_I + 12*m_yy_I - 33*m_yz_I + 12*m_zz_I - 72*omegaVar*m_xx_I - 12*omegaVar*m_yy_I + 33*omegaVar*m_yz_I - 12*omegaVar*m_zz_I + 107));
    m_xz_t90 = (m_xz_I*(313*omegaVar + 2790))/(18*(72*m_xx_I + 12*m_yy_I - 33*m_yz_I + 12*m_zz_I - 72*omegaVar*m_xx_I - 12*omegaVar*m_yy_I + 33*omegaVar*m_yz_I - 12*omegaVar*m_zz_I + 107));
    m_yy_t45 = -(108*m_xx_I - 2772*m_yy_I + 648*m_yz_I + 18*m_zz_I + 66*omegaVar*m_xx_I - 302*omegaVar*m_yy_I + 48*omegaVar*m_yz_I + 11*omegaVar*m_zz_I - 72)/(27*(72*m_xx_I + 12*m_yy_I - 33*m_yz_I + 12*m_zz_I - 72*omegaVar*m_xx_I - 12*omegaVar*m_yy_I + 33*omegaVar*m_yz_I - 12*omegaVar*m_zz_I + 107));
    m_yz_t90 = -(432*m_yy_I - 198*m_xx_I - 3978*m_yz_I + 432*m_zz_I - 121*omegaVar*m_xx_I + 32*omegaVar*m_yy_I - 401*omegaVar*m_yz_I + 32*omegaVar*m_zz_I + 132)/(36*(72*m_xx_I + 12*m_yy_I - 33*m_yz_I + 12*m_zz_I - 72*omegaVar*m_xx_I - 12*omegaVar*m_yy_I + 33*omegaVar*m_yz_I - 12*omegaVar*m_zz_I + 107));
    m_zz_t45 = -(108*m_xx_I + 18*m_yy_I + 648*m_yz_I - 2772*m_zz_I + 66*omegaVar*m_xx_I + 11*omegaVar*m_yy_I + 48*omegaVar*m_yz_I - 302*omegaVar*m_zz_I - 72)/(27*(72*m_xx_I + 12*m_yy_I - 33*m_yz_I + 12*m_zz_I - 72*omegaVar*m_xx_I - 12*omegaVar*m_yy_I + 33*omegaVar*m_yz_I - 12*omegaVar*m_zz_I + 107));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = (m_xx_t45 - (omegaVar*m_xx_t45)/2 + 1.0/3.0)*rho - RHO_0/3;
    L_BC_Fy = (m_xy_t90 - m_xx_t45/12 + m_yy_t45/6 - m_yz_t90/2 + m_zz_t45/6 + (omegaVar*m_xx_t45)/24 - (omegaVar*m_xy_t90)/2 - (omegaVar*m_yy_t45)/12 + (omegaVar*m_yz_t90)/4 - (omegaVar*m_zz_t45)/12 + 1.0/18.0)*rho - RHO_0/18;
    L_BC_Fz = (m_xx_t45/12 + m_xz_t90 - m_yy_t45/6 + m_yz_t90/2 - m_zz_t45/6 - (omegaVar*m_xx_t45)/24 - (omegaVar*m_xz_t90)/2 + (omegaVar*m_yy_t45)/12 - (omegaVar*m_yz_t90)/4 + (omegaVar*m_zz_t45)/12 - 1.0/18.0)*rho + RHO_0/18;
    #endif //_BC_FORCES 

    break;
case 0b0000010110101111:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[3] + pop[4] + pop[6] + pop[7] + pop[12] + pop[13] + pop[15] + pop[17];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[15] - (1.0/3.0)*pop[17]);
    m_xy_I = inv_rho_I * (pop[7] - pop[13]);
    m_xz_I = inv_rho_I * (-pop[15]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] - (1.0/3.0)*pop[15] + (2.0/3.0)*pop[17]);
    m_yz_I = inv_rho_I * (pop[12] - pop[17]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[17]);

    rho = (36*(23*rho_I + 24*m_xx_I*rho_I + 57*m_xz_I*rho_I - 6*m_yy_I*rho_I + 24*m_zz_I*rho_I - 24*omegaVar*m_xx_I*rho_I - 57*omegaVar*m_xz_I*rho_I + 6*omegaVar*m_yy_I*rho_I - 24*omegaVar*m_zz_I*rho_I))/(5*(43*omegaVar + 72));

    m_xx_t45 = (936*m_xx_I + 1008*m_xz_I - 144*m_yy_I + 216*m_zz_I + 239*omegaVar*m_xx_I - 158*omegaVar*m_xz_I - 6*omegaVar*m_yy_I - 191*omegaVar*m_zz_I + 192)/(36*(24*m_xx_I + 57*m_xz_I - 6*m_yy_I + 24*m_zz_I - 24*omegaVar*m_xx_I - 57*omegaVar*m_xz_I + 6*omegaVar*m_yy_I - 24*omegaVar*m_zz_I + 23));
    m_xy_t90 = (5*m_xy_I*(43*omegaVar + 72))/(18*(24*m_xx_I + 57*m_xz_I - 6*m_yy_I + 24*m_zz_I - 24*omegaVar*m_xx_I - 57*omegaVar*m_xz_I + 6*omegaVar*m_yy_I - 24*omegaVar*m_zz_I + 23));
    m_xz_t90 = (504*m_xx_I + 2412*m_xz_I - 216*m_yy_I + 504*m_zz_I - 79*omegaVar*m_xx_I + 538*omegaVar*m_xz_I - 34*omegaVar*m_yy_I - 79*omegaVar*m_zz_I + 228)/(36*(24*m_xx_I + 57*m_xz_I - 6*m_yy_I + 24*m_zz_I - 24*omegaVar*m_xx_I - 57*omegaVar*m_xz_I + 6*omegaVar*m_yy_I - 24*omegaVar*m_zz_I + 23));
    m_yy_t45 = -(72*m_xx_I + 216*m_xz_I - 288*m_yy_I + 72*m_zz_I + 3*omegaVar*m_xx_I + 34*omegaVar*m_xz_I - 162*omegaVar*m_yy_I + 3*omegaVar*m_zz_I + 24)/(18*(24*m_xx_I + 57*m_xz_I - 6*m_yy_I + 24*m_zz_I - 24*omegaVar*m_xx_I - 57*omegaVar*m_xz_I + 6*omegaVar*m_yy_I - 24*omegaVar*m_zz_I + 23));
    m_yz_t90 = (5*m_yz_I*(43*omegaVar + 72))/(18*(24*m_xx_I + 57*m_xz_I - 6*m_yy_I + 24*m_zz_I - 24*omegaVar*m_xx_I - 57*omegaVar*m_xz_I + 6*omegaVar*m_yy_I - 24*omegaVar*m_zz_I + 23));
    m_zz_t45 = (216*m_xx_I + 1008*m_xz_I - 144*m_yy_I + 936*m_zz_I - 191*omegaVar*m_xx_I - 158*omegaVar*m_xz_I - 6*omegaVar*m_yy_I + 239*omegaVar*m_zz_I + 192)/(36*(24*m_xx_I + 57*m_xz_I - 6*m_yy_I + 24*m_zz_I - 24*omegaVar*m_xx_I - 57*omegaVar*m_xz_I + 6*omegaVar*m_yy_I - 24*omegaVar*m_zz_I + 23));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = ((5*m_xx_t45)/6 - m_xz_t90/2 + m_yy_t45/12 - m_zz_t45/6 - (5*omegaVar*m_xx_t45)/12 + (omegaVar*m_xz_t90)/4 - (omegaVar*m_yy_t45)/24 + (omegaVar*m_zz_t45)/12 + 5/18)*rho - (5*RHO_0)/18;
    L_BC_Fy = -(rho*(m_xy_t90 - m_yz_t90)*(omegaVar - 2))/2;
    L_BC_Fz = (m_xx_t45/6 + m_xz_t90/2 - m_yy_t45/12 - (5*m_zz_t45)/6 - (omegaVar*m_xx_t45)/12 - (omegaVar*m_xz_t90)/4 + (omegaVar*m_yy_t45)/24 + (5*omegaVar*m_zz_t45)/12 - 5/18)*rho + (5*RHO_0)/18;
    #endif //_BC_FORCES 

    break;
case 0b0000010110110000:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);

    rho = (6*(15*rho_I + m_xx_I*rho_I + 6*m_xz_I*rho_I + m_yy_I*rho_I - 6*m_yz_I*rho_I + 4*m_zz_I*rho_I - omegaVar*m_xx_I*rho_I - 6*omegaVar*m_xz_I*rho_I - omegaVar*m_yy_I*rho_I + 6*omegaVar*m_yz_I*rho_I - 4*omegaVar*m_zz_I*rho_I))/(3*omegaVar + 82);

    m_xx_t45 = (2052*m_xx_I + 504*m_xz_I - 162*m_yy_I + 234*m_yz_I + 90*m_zz_I + 73*omegaVar*m_xx_I + 6*omegaVar*m_xz_I - 8*omegaVar*m_yy_I + 21*omegaVar*m_yz_I - 5*omegaVar*m_zz_I + 30)/(135*(m_xx_I + 6*m_xz_I + m_yy_I - 6*m_yz_I + 4*m_zz_I - omegaVar*m_xx_I - 6*omegaVar*m_xz_I - omegaVar*m_yy_I + 6*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 15));
    m_xy_t90 = (m_xy_I*(3*omegaVar + 82))/(6*(m_xx_I + 6*m_xz_I + m_yy_I - 6*m_yz_I + 4*m_zz_I - omegaVar*m_xx_I - 6*omegaVar*m_xz_I - omegaVar*m_yy_I + 6*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 15));
    m_xz_t90 = (168*m_xx_I + 1746*m_xz_I - 78*m_yy_I - 24*m_yz_I + 180*m_zz_I + 2*omegaVar*m_xx_I + 39*omegaVar*m_xz_I - 7*omegaVar*m_yy_I + 24*omegaVar*m_yz_I - 10*omegaVar*m_zz_I + 60)/(90*(m_xx_I + 6*m_xz_I + m_yy_I - 6*m_yz_I + 4*m_zz_I - omegaVar*m_xx_I - 6*omegaVar*m_xz_I - omegaVar*m_yy_I + 6*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 15));
    m_yy_t45 = -(162*m_xx_I + 234*m_xz_I - 2052*m_yy_I + 504*m_yz_I - 90*m_zz_I + 8*omegaVar*m_xx_I + 21*omegaVar*m_xz_I - 73*omegaVar*m_yy_I + 6*omegaVar*m_yz_I + 5*omegaVar*m_zz_I - 30)/(135*(m_xx_I + 6*m_xz_I + m_yy_I - 6*m_yz_I + 4*m_zz_I - omegaVar*m_xx_I - 6*omegaVar*m_xz_I - omegaVar*m_yy_I + 6*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 15));
    m_yz_t90 = (78*m_xx_I - 24*m_xz_I - 168*m_yy_I + 1746*m_yz_I - 180*m_zz_I + 7*omegaVar*m_xx_I + 24*omegaVar*m_xz_I - 2*omegaVar*m_yy_I + 39*omegaVar*m_yz_I + 10*omegaVar*m_zz_I - 60)/(90*(m_xx_I + 6*m_xz_I + m_yy_I - 6*m_yz_I + 4*m_zz_I - omegaVar*m_xx_I - 6*omegaVar*m_xz_I - omegaVar*m_yy_I + 6*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 15));
    m_zz_t45 = (36*m_xx_I + 216*m_xz_I + 36*m_yy_I - 216*m_yz_I + 882*m_zz_I - 2*omegaVar*m_xx_I - 12*omegaVar*m_xz_I - 2*omegaVar*m_yy_I + 12*omegaVar*m_yz_I + 19*omegaVar*m_zz_I + 48)/(54*(m_xx_I + 6*m_xz_I + m_yy_I - 6*m_yz_I + 4*m_zz_I - omegaVar*m_xx_I - 6*omegaVar*m_xz_I - omegaVar*m_yy_I + 6*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 15));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = (m_xx_t45/6 + m_xz_t90/2 - m_yy_t45/12 + m_zz_t45/6 - (omegaVar*m_xx_t45)/12 - (omegaVar*m_xz_t90)/4 + (omegaVar*m_yy_t45)/24 - (omegaVar*m_zz_t45)/12 + 1.0/18.0)*rho - RHO_0/18;
    L_BC_Fy = (m_xx_t45/12 - m_yy_t45/6 + m_yz_t90/2 - m_zz_t45/6 - (omegaVar*m_xx_t45)/24 + (omegaVar*m_yy_t45)/12 - (omegaVar*m_yz_t90)/4 + (omegaVar*m_zz_t45)/12 - 1.0/18.0)*rho + RHO_0/18;
    L_BC_Fz = (m_xx_t45/12 + m_xz_t90/2 + m_yy_t45/12 - m_yz_t90/2 + m_zz_t45/3 - (omegaVar*m_xx_t45)/24 - (omegaVar*m_xz_t90)/4 - (omegaVar*m_yy_t45)/24 + (omegaVar*m_yz_t90)/4 - (omegaVar*m_zz_t45)/6 + 1.0/9.0)*rho - RHO_0/9;
    #endif //_BC_FORCES 

    break;
case 0b0000010110110001:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[8] + pop[9] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);

    rho = (12*(5*rho_I + m_xx_I*rho_I + 2*m_xy_I*rho_I + 2*m_xz_I*rho_I + m_yy_I*rho_I - 2*m_yz_I*rho_I + m_zz_I*rho_I - omegaVar*m_xx_I*rho_I - 2*omegaVar*m_xy_I*rho_I - 2*omegaVar*m_xz_I*rho_I - omegaVar*m_yy_I*rho_I + 2*omegaVar*m_yz_I*rho_I - omegaVar*m_zz_I*rho_I))/(3*omegaVar + 52);

    m_xx_t45 = (472*m_xx_I + 112*m_xy_I + 112*m_xz_I + 4*m_yy_I + 44*m_yz_I + 4*m_zz_I + 23*omegaVar*m_xx_I - 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - 4*omegaVar*m_yy_I + 11*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 20)/(90*(m_xx_I + 2*m_xy_I + 2*m_xz_I + m_yy_I - 2*m_yz_I + m_zz_I - omegaVar*m_xx_I - 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - omegaVar*m_yy_I + 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_xy_t90 = (112*m_xx_I + 1108*m_xy_I + 16*m_xz_I + 112*m_yy_I - 16*m_yz_I - 44*m_zz_I - 2*omegaVar*m_xx_I + 47*omegaVar*m_xy_I - 16*omegaVar*m_xz_I - 2*omegaVar*m_yy_I + 16*omegaVar*m_yz_I - 11*omegaVar*m_zz_I + 40)/(180*(m_xx_I + 2*m_xy_I + 2*m_xz_I + m_yy_I - 2*m_yz_I + m_zz_I - omegaVar*m_xx_I - 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - omegaVar*m_yy_I + 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_xz_t90 = (112*m_xx_I + 16*m_xy_I + 1108*m_xz_I - 44*m_yy_I - 16*m_yz_I + 112*m_zz_I - 2*omegaVar*m_xx_I - 16*omegaVar*m_xy_I + 47*omegaVar*m_xz_I - 11*omegaVar*m_yy_I + 16*omegaVar*m_yz_I - 2*omegaVar*m_zz_I + 40)/(180*(m_xx_I + 2*m_xy_I + 2*m_xz_I + m_yy_I - 2*m_yz_I + m_zz_I - omegaVar*m_xx_I - 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - omegaVar*m_yy_I + 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_yy_t45 = (4*m_xx_I + 112*m_xy_I - 44*m_xz_I + 472*m_yy_I - 112*m_yz_I + 4*m_zz_I - 4*omegaVar*m_xx_I - 2*omegaVar*m_xy_I - 11*omegaVar*m_xz_I + 23*omegaVar*m_yy_I + 2*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 20)/(90*(m_xx_I + 2*m_xy_I + 2*m_xz_I + m_yy_I - 2*m_yz_I + m_zz_I - omegaVar*m_xx_I - 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - omegaVar*m_yy_I + 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_yz_t90 = (44*m_xx_I - 16*m_xy_I - 16*m_xz_I - 112*m_yy_I + 1108*m_yz_I - 112*m_zz_I + 11*omegaVar*m_xx_I + 16*omegaVar*m_xy_I + 16*omegaVar*m_xz_I + 2*omegaVar*m_yy_I + 47*omegaVar*m_yz_I + 2*omegaVar*m_zz_I - 40)/(180*(m_xx_I + 2*m_xy_I + 2*m_xz_I + m_yy_I - 2*m_yz_I + m_zz_I - omegaVar*m_xx_I - 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - omegaVar*m_yy_I + 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_zz_t45 = (4*m_xx_I - 44*m_xy_I + 112*m_xz_I + 4*m_yy_I - 112*m_yz_I + 472*m_zz_I - 4*omegaVar*m_xx_I - 11*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - 4*omegaVar*m_yy_I + 2*omegaVar*m_yz_I + 23*omegaVar*m_zz_I + 20)/(90*(m_xx_I + 2*m_xy_I + 2*m_xz_I + m_yy_I - 2*m_yz_I + m_zz_I - omegaVar*m_xx_I - 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - omegaVar*m_yy_I + 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = (rho*(omegaVar - 2)*(2*m_xy_t90 - 2*m_xz_t90 + m_yy_t45 - m_zz_t45))/8;
    L_BC_Fy = (m_yz_t90/2 - m_xy_t90/2 - m_yy_t45/3 - m_xx_t45/12 - m_zz_t45/12 + (omegaVar*m_xx_t45)/24 + (omegaVar*m_xy_t90)/4 + (omegaVar*m_yy_t45)/6 - (omegaVar*m_yz_t90)/4 + (omegaVar*m_zz_t45)/24 - 1.0/9.0)*rho + RHO_0/9;
    L_BC_Fz = (m_xx_t45/12 + m_xz_t90/2 + m_yy_t45/12 - m_yz_t90/2 + m_zz_t45/3 - (omegaVar*m_xx_t45)/24 - (omegaVar*m_xz_t90)/4 - (omegaVar*m_yy_t45)/24 + (omegaVar*m_yz_t90)/4 - (omegaVar*m_zz_t45)/6 + 1.0/9.0)*rho - RHO_0/9;
    #endif //_BC_FORCES 

    break;
case 0b0000010110110010:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[11] + pop[12] + pop[13] + pop[15] + pop[16] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13]);
    m_xz_I = inv_rho_I * (pop[9] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);

    rho = (12*(5*rho_I + m_xx_I*rho_I - 2*m_xy_I*rho_I + 2*m_xz_I*rho_I + m_yy_I*rho_I - 2*m_yz_I*rho_I + m_zz_I*rho_I - omegaVar*m_xx_I*rho_I + 2*omegaVar*m_xy_I*rho_I - 2*omegaVar*m_xz_I*rho_I - omegaVar*m_yy_I*rho_I + 2*omegaVar*m_yz_I*rho_I - omegaVar*m_zz_I*rho_I))/(3*omegaVar + 52);

    m_xx_t45 = (472*m_xx_I - 112*m_xy_I + 112*m_xz_I + 4*m_yy_I + 44*m_yz_I + 4*m_zz_I + 23*omegaVar*m_xx_I + 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - 4*omegaVar*m_yy_I + 11*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 20)/(90*(m_xx_I - 2*m_xy_I + 2*m_xz_I + m_yy_I - 2*m_yz_I + m_zz_I - omegaVar*m_xx_I + 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - omegaVar*m_yy_I + 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_xy_t90 = (1108*m_xy_I - 112*m_xx_I - 16*m_xz_I - 112*m_yy_I + 16*m_yz_I + 44*m_zz_I + 2*omegaVar*m_xx_I + 47*omegaVar*m_xy_I + 16*omegaVar*m_xz_I + 2*omegaVar*m_yy_I - 16*omegaVar*m_yz_I + 11*omegaVar*m_zz_I - 40)/(180*(m_xx_I - 2*m_xy_I + 2*m_xz_I + m_yy_I - 2*m_yz_I + m_zz_I - omegaVar*m_xx_I + 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - omegaVar*m_yy_I + 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_xz_t90 = (112*m_xx_I - 16*m_xy_I + 1108*m_xz_I - 44*m_yy_I - 16*m_yz_I + 112*m_zz_I - 2*omegaVar*m_xx_I + 16*omegaVar*m_xy_I + 47*omegaVar*m_xz_I - 11*omegaVar*m_yy_I + 16*omegaVar*m_yz_I - 2*omegaVar*m_zz_I + 40)/(180*(m_xx_I - 2*m_xy_I + 2*m_xz_I + m_yy_I - 2*m_yz_I + m_zz_I - omegaVar*m_xx_I + 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - omegaVar*m_yy_I + 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_yy_t45 = (4*m_xx_I - 112*m_xy_I - 44*m_xz_I + 472*m_yy_I - 112*m_yz_I + 4*m_zz_I - 4*omegaVar*m_xx_I + 2*omegaVar*m_xy_I - 11*omegaVar*m_xz_I + 23*omegaVar*m_yy_I + 2*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 20)/(90*(m_xx_I - 2*m_xy_I + 2*m_xz_I + m_yy_I - 2*m_yz_I + m_zz_I - omegaVar*m_xx_I + 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - omegaVar*m_yy_I + 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_yz_t90 = (44*m_xx_I + 16*m_xy_I - 16*m_xz_I - 112*m_yy_I + 1108*m_yz_I - 112*m_zz_I + 11*omegaVar*m_xx_I - 16*omegaVar*m_xy_I + 16*omegaVar*m_xz_I + 2*omegaVar*m_yy_I + 47*omegaVar*m_yz_I + 2*omegaVar*m_zz_I - 40)/(180*(m_xx_I - 2*m_xy_I + 2*m_xz_I + m_yy_I - 2*m_yz_I + m_zz_I - omegaVar*m_xx_I + 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - omegaVar*m_yy_I + 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_zz_t45 = (4*m_xx_I + 44*m_xy_I + 112*m_xz_I + 4*m_yy_I - 112*m_yz_I + 472*m_zz_I - 4*omegaVar*m_xx_I + 11*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - 4*omegaVar*m_yy_I + 2*omegaVar*m_yz_I + 23*omegaVar*m_zz_I + 20)/(90*(m_xx_I - 2*m_xy_I + 2*m_xz_I + m_yy_I - 2*m_yz_I + m_zz_I - omegaVar*m_xx_I + 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - omegaVar*m_yy_I + 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = (m_xx_t45/3 - m_xy_t90/2 + m_xz_t90/2 + m_yy_t45/12 + m_zz_t45/12 - (omegaVar*m_xx_t45)/6 + (omegaVar*m_xy_t90)/4 - (omegaVar*m_xz_t90)/4 - (omegaVar*m_yy_t45)/24 - (omegaVar*m_zz_t45)/24 + 1.0/9.0)*rho - RHO_0/9;
    L_BC_Fy = (m_xy_t90/2 - m_xx_t45/12 - m_yy_t45/3 + m_yz_t90/2 - m_zz_t45/12 + (omegaVar*m_xx_t45)/24 - (omegaVar*m_xy_t90)/4 + (omegaVar*m_yy_t45)/6 - (omegaVar*m_yz_t90)/4 + (omegaVar*m_zz_t45)/24 - 1.0/9.0)*rho + RHO_0/9;
    L_BC_Fz = (m_xx_t45/12 + m_xz_t90/2 + m_yy_t45/12 - m_yz_t90/2 + m_zz_t45/3 - (omegaVar*m_xx_t45)/24 - (omegaVar*m_xz_t90)/4 - (omegaVar*m_yy_t45)/24 + (omegaVar*m_yz_t90)/4 - (omegaVar*m_zz_t45)/6 + 1.0/9.0)*rho - RHO_0/9;
    #endif //_BC_FORCES 

    break;
case 0b0000010110110011:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[4] + pop[5] + pop[6] + pop[8] + pop[9] + pop[12] + pop[13] + pop[15] + pop[16] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[8] - pop[13]);
    m_xz_I = inv_rho_I * (pop[9] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[12] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);

    rho = (36*(107*rho_I + 12*m_xx_I*rho_I + 33*m_xz_I*rho_I + 72*m_yy_I*rho_I + 12*m_zz_I*rho_I - 12*omegaVar*m_xx_I*rho_I - 33*omegaVar*m_xz_I*rho_I - 72*omegaVar*m_yy_I*rho_I - 12*omegaVar*m_zz_I*rho_I))/(313*omegaVar + 2790);

    m_xx_t45 = (2772*m_xx_I + 648*m_xz_I - 108*m_yy_I - 18*m_zz_I + 302*omegaVar*m_xx_I + 48*omegaVar*m_xz_I - 66*omegaVar*m_yy_I - 11*omegaVar*m_zz_I + 72)/(27*(12*m_xx_I + 33*m_xz_I + 72*m_yy_I + 12*m_zz_I - 12*omegaVar*m_xx_I - 33*omegaVar*m_xz_I - 72*omegaVar*m_yy_I - 12*omegaVar*m_zz_I + 107));
    m_xy_t90 = (m_xy_I*(313*omegaVar + 2790))/(18*(12*m_xx_I + 33*m_xz_I + 72*m_yy_I + 12*m_zz_I - 12*omegaVar*m_xx_I - 33*omegaVar*m_xz_I - 72*omegaVar*m_yy_I - 12*omegaVar*m_zz_I + 107));
    m_xz_t90 = (432*m_xx_I + 3978*m_xz_I - 198*m_yy_I + 432*m_zz_I + 32*omegaVar*m_xx_I + 401*omegaVar*m_xz_I - 121*omegaVar*m_yy_I + 32*omegaVar*m_zz_I + 132)/(36*(12*m_xx_I + 33*m_xz_I + 72*m_yy_I + 12*m_zz_I - 12*omegaVar*m_xx_I - 33*omegaVar*m_xz_I - 72*omegaVar*m_yy_I - 12*omegaVar*m_zz_I + 107));
    m_yy_t45 = -(72*m_xx_I + 198*m_xz_I - 2358*m_yy_I + 72*m_zz_I + 44*omegaVar*m_xx_I + 121*omegaVar*m_xz_I - 49*omegaVar*m_yy_I + 44*omegaVar*m_zz_I - 288)/(18*(12*m_xx_I + 33*m_xz_I + 72*m_yy_I + 12*m_zz_I - 12*omegaVar*m_xx_I - 33*omegaVar*m_xz_I - 72*omegaVar*m_yy_I - 12*omegaVar*m_zz_I + 107));
    m_yz_t90 = (m_yz_I*(313*omegaVar + 2790))/(18*(12*m_xx_I + 33*m_xz_I + 72*m_yy_I + 12*m_zz_I - 12*omegaVar*m_xx_I - 33*omegaVar*m_xz_I - 72*omegaVar*m_yy_I - 12*omegaVar*m_zz_I + 107));
    m_zz_t45 = (648*m_xz_I - 18*m_xx_I - 108*m_yy_I + 2772*m_zz_I - 11*omegaVar*m_xx_I + 48*omegaVar*m_xz_I - 66*omegaVar*m_yy_I + 302*omegaVar*m_zz_I + 72)/(27*(12*m_xx_I + 33*m_xz_I + 72*m_yy_I + 12*m_zz_I - 12*omegaVar*m_xx_I - 33*omegaVar*m_xz_I - 72*omegaVar*m_yy_I - 12*omegaVar*m_zz_I + 107));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = (m_xx_t45/6 - m_xy_t90 + m_xz_t90/2 - m_yy_t45/12 + m_zz_t45/6 - (omegaVar*m_xx_t45)/12 + (omegaVar*m_xy_t90)/2 - (omegaVar*m_xz_t90)/4 + (omegaVar*m_yy_t45)/24 - (omegaVar*m_zz_t45)/12 + 1.0/18.0)*rho - RHO_0/18;
    L_BC_Fy = ((omegaVar*m_yy_t45)/2 - m_yy_t45 - 1.0/3.0)*rho + RHO_0/3;
    L_BC_Fz = (m_xx_t45/6 + m_xz_t90/2 - m_yy_t45/12 - m_yz_t90 + m_zz_t45/6 - (omegaVar*m_xx_t45)/12 - (omegaVar*m_xz_t90)/4 + (omegaVar*m_yy_t45)/24 + (omegaVar*m_yz_t90)/2 - (omegaVar*m_zz_t45)/12 + 1.0/18.0)*rho - RHO_0/18;
    #endif //_BC_FORCES 

    break;
case 0b0000010110110100:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);

    rho = (6*(15*rho_I + m_xx_I*rho_I + 6*m_xz_I*rho_I + m_yy_I*rho_I - 6*m_yz_I*rho_I + 4*m_zz_I*rho_I - omegaVar*m_xx_I*rho_I - 6*omegaVar*m_xz_I*rho_I - omegaVar*m_yy_I*rho_I + 6*omegaVar*m_yz_I*rho_I - 4*omegaVar*m_zz_I*rho_I))/(3*omegaVar + 82);

    m_xx_t45 = (2052*m_xx_I + 504*m_xz_I - 162*m_yy_I + 234*m_yz_I + 90*m_zz_I + 73*omegaVar*m_xx_I + 6*omegaVar*m_xz_I - 8*omegaVar*m_yy_I + 21*omegaVar*m_yz_I - 5*omegaVar*m_zz_I + 30)/(135*(m_xx_I + 6*m_xz_I + m_yy_I - 6*m_yz_I + 4*m_zz_I - omegaVar*m_xx_I - 6*omegaVar*m_xz_I - omegaVar*m_yy_I + 6*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 15));
    m_xy_t90 = (m_xy_I*(3*omegaVar + 82))/(6*(m_xx_I + 6*m_xz_I + m_yy_I - 6*m_yz_I + 4*m_zz_I - omegaVar*m_xx_I - 6*omegaVar*m_xz_I - omegaVar*m_yy_I + 6*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 15));
    m_xz_t90 = (168*m_xx_I + 1746*m_xz_I - 78*m_yy_I - 24*m_yz_I + 180*m_zz_I + 2*omegaVar*m_xx_I + 39*omegaVar*m_xz_I - 7*omegaVar*m_yy_I + 24*omegaVar*m_yz_I - 10*omegaVar*m_zz_I + 60)/(90*(m_xx_I + 6*m_xz_I + m_yy_I - 6*m_yz_I + 4*m_zz_I - omegaVar*m_xx_I - 6*omegaVar*m_xz_I - omegaVar*m_yy_I + 6*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 15));
    m_yy_t45 = -(162*m_xx_I + 234*m_xz_I - 2052*m_yy_I + 504*m_yz_I - 90*m_zz_I + 8*omegaVar*m_xx_I + 21*omegaVar*m_xz_I - 73*omegaVar*m_yy_I + 6*omegaVar*m_yz_I + 5*omegaVar*m_zz_I - 30)/(135*(m_xx_I + 6*m_xz_I + m_yy_I - 6*m_yz_I + 4*m_zz_I - omegaVar*m_xx_I - 6*omegaVar*m_xz_I - omegaVar*m_yy_I + 6*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 15));
    m_yz_t90 = (78*m_xx_I - 24*m_xz_I - 168*m_yy_I + 1746*m_yz_I - 180*m_zz_I + 7*omegaVar*m_xx_I + 24*omegaVar*m_xz_I - 2*omegaVar*m_yy_I + 39*omegaVar*m_yz_I + 10*omegaVar*m_zz_I - 60)/(90*(m_xx_I + 6*m_xz_I + m_yy_I - 6*m_yz_I + 4*m_zz_I - omegaVar*m_xx_I - 6*omegaVar*m_xz_I - omegaVar*m_yy_I + 6*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 15));
    m_zz_t45 = (36*m_xx_I + 216*m_xz_I + 36*m_yy_I - 216*m_yz_I + 882*m_zz_I - 2*omegaVar*m_xx_I - 12*omegaVar*m_xz_I - 2*omegaVar*m_yy_I + 12*omegaVar*m_yz_I + 19*omegaVar*m_zz_I + 48)/(54*(m_xx_I + 6*m_xz_I + m_yy_I - 6*m_yz_I + 4*m_zz_I - omegaVar*m_xx_I - 6*omegaVar*m_xz_I - omegaVar*m_yy_I + 6*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 15));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = (m_xx_t45/6 + m_xz_t90/2 - m_yy_t45/12 + m_zz_t45/6 - (omegaVar*m_xx_t45)/12 - (omegaVar*m_xz_t90)/4 + (omegaVar*m_yy_t45)/24 - (omegaVar*m_zz_t45)/12 + 1.0/18.0)*rho - RHO_0/18;
    L_BC_Fy = (m_xx_t45/12 - m_yy_t45/6 + m_yz_t90/2 - m_zz_t45/6 - (omegaVar*m_xx_t45)/24 + (omegaVar*m_yy_t45)/12 - (omegaVar*m_yz_t90)/4 + (omegaVar*m_zz_t45)/12 - 1.0/18.0)*rho + RHO_0/18;
    L_BC_Fz = (m_xx_t45/12 + m_xz_t90/2 + m_yy_t45/12 - m_yz_t90/2 + m_zz_t45/3 - (omegaVar*m_xx_t45)/24 - (omegaVar*m_xz_t90)/4 - (omegaVar*m_yy_t45)/24 + (omegaVar*m_yz_t90)/4 - (omegaVar*m_zz_t45)/6 + 1.0/9.0)*rho - RHO_0/9;
    #endif //_BC_FORCES 

    break;
case 0b0000010110110101:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[8] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (-pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);

    rho = (3*(15*rho_I + 11*m_xx_I*rho_I + 6*m_xy_I*rho_I + 30*m_xz_I*rho_I - m_yy_I*rho_I - 6*m_yz_I*rho_I + 11*m_zz_I*rho_I - 11*omegaVar*m_xx_I*rho_I - 6*omegaVar*m_xy_I*rho_I - 30*omegaVar*m_xz_I*rho_I + omegaVar*m_yy_I*rho_I + 6*omegaVar*m_yz_I*rho_I - 11*omegaVar*m_zz_I*rho_I))/(9*omegaVar + 31);

    m_xx_t45 = (2*(1206*m_xx_I + 252*m_xy_I + 1260*m_xz_I - 135*m_yy_I + 27*m_yz_I + 369*m_zz_I + 194*omegaVar*m_xx_I - 12*omegaVar*m_xy_I - 60*omegaVar*m_xz_I - 25*omegaVar*m_yy_I + 93*omegaVar*m_yz_I - 49*omegaVar*m_zz_I + 165))/(135*(11*m_xx_I + 6*m_xy_I + 30*m_xz_I - m_yy_I - 6*m_yz_I + 11*m_zz_I - 11*omegaVar*m_xx_I - 6*omegaVar*m_xy_I - 30*omegaVar*m_xz_I + omegaVar*m_yy_I + 6*omegaVar*m_yz_I - 11*omegaVar*m_zz_I + 15));
    m_xy_t90 = (84*m_xx_I + 663*m_xy_I + 60*m_xz_I + 60*m_yy_I - 12*m_yz_I - 9*m_zz_I - 4*omegaVar*m_xx_I + 177*omegaVar*m_xy_I - 60*omegaVar*m_xz_I + 20*omegaVar*m_yy_I + 12*omegaVar*m_yz_I - 31*omegaVar*m_zz_I + 30)/(45*(11*m_xx_I + 6*m_xy_I + 30*m_xz_I - m_yy_I - 6*m_yz_I + 11*m_zz_I - 11*omegaVar*m_xx_I - 6*omegaVar*m_xy_I - 30*omegaVar*m_xz_I + omegaVar*m_yy_I + 6*omegaVar*m_yz_I - 11*omegaVar*m_zz_I + 15));
    m_xz_t90 = (84*m_xx_I + 12*m_xy_I + 339*m_xz_I - 33*m_yy_I - 12*m_yz_I + 84*m_zz_I - 4*omegaVar*m_xx_I - 12*omegaVar*m_xy_I + 21*omegaVar*m_xz_I - 7*omegaVar*m_yy_I + 12*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 30)/(9*(11*m_xx_I + 6*m_xy_I + 30*m_xz_I - m_yy_I - 6*m_yz_I + 11*m_zz_I - 11*omegaVar*m_xx_I - 6*omegaVar*m_xy_I - 30*omegaVar*m_xz_I + omegaVar*m_yy_I + 6*omegaVar*m_yz_I - 11*omegaVar*m_zz_I + 15));
    m_yy_t45 = -(2*(27*m_xx_I - 36*m_xy_I + 99*m_xz_I - 180*m_yy_I + 36*m_yz_I + 27*m_zz_I + 5*omegaVar*m_xx_I - 12*omegaVar*m_xy_I + 21*omegaVar*m_xz_I - 52*omegaVar*m_yy_I + 12*omegaVar*m_yz_I + 5*omegaVar*m_zz_I + 3))/(27*(11*m_xx_I + 6*m_xy_I + 30*m_xz_I - m_yy_I - 6*m_yz_I + 11*m_zz_I - 11*omegaVar*m_xx_I - 6*omegaVar*m_xy_I - 30*omegaVar*m_xz_I + omegaVar*m_yy_I + 6*omegaVar*m_yz_I - 11*omegaVar*m_zz_I + 15));
    m_yz_t90 = (9*m_xx_I - 12*m_xy_I - 60*m_xz_I - 60*m_yy_I + 663*m_yz_I - 84*m_zz_I + 31*omegaVar*m_xx_I + 12*omegaVar*m_xy_I + 60*omegaVar*m_xz_I - 20*omegaVar*m_yy_I + 177*omegaVar*m_yz_I + 4*omegaVar*m_zz_I - 30)/(45*(11*m_xx_I + 6*m_xy_I + 30*m_xz_I - m_yy_I - 6*m_yz_I + 11*m_zz_I - 11*omegaVar*m_xx_I - 6*omegaVar*m_xy_I - 30*omegaVar*m_xz_I + omegaVar*m_yy_I + 6*omegaVar*m_yz_I - 11*omegaVar*m_zz_I + 15));
    m_zz_t45 = -(2*(27*m_xy_I - 369*m_xx_I - 1260*m_xz_I + 135*m_yy_I + 252*m_yz_I - 1206*m_zz_I + 49*omegaVar*m_xx_I + 93*omegaVar*m_xy_I + 60*omegaVar*m_xz_I + 25*omegaVar*m_yy_I - 12*omegaVar*m_yz_I - 194*omegaVar*m_zz_I - 165))/(135*(11*m_xx_I + 6*m_xy_I + 30*m_xz_I - m_yy_I - 6*m_yz_I + 11*m_zz_I - 11*omegaVar*m_xx_I - 6*omegaVar*m_xy_I - 30*omegaVar*m_xz_I + omegaVar*m_yy_I + 6*omegaVar*m_yz_I - 11*omegaVar*m_zz_I + 15));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = (m_zz_t45/12 - m_xy_t90/2 - m_yy_t45/6 - m_xx_t45/6 + (omegaVar*m_xx_t45)/12 + (omegaVar*m_xy_t90)/4 + (omegaVar*m_yy_t45)/12 - (omegaVar*m_zz_t45)/24 - 1.0/18.0)*rho + RHO_0/18;
    L_BC_Fy = (m_yz_t90/2 - m_xy_t90/2 - m_yy_t45/3 - m_xx_t45/12 - m_zz_t45/12 + (omegaVar*m_xx_t45)/24 + (omegaVar*m_xy_t90)/4 + (omegaVar*m_yy_t45)/6 - (omegaVar*m_yz_t90)/4 + (omegaVar*m_zz_t45)/24 - 1.0/9.0)*rho + RHO_0/9;
    L_BC_Fz = (m_yy_t45/6 - m_xx_t45/12 - m_yz_t90/2 + m_zz_t45/6 + (omegaVar*m_xx_t45)/24 - (omegaVar*m_yy_t45)/12 + (omegaVar*m_yz_t90)/4 - (omegaVar*m_zz_t45)/12 + 1.0/18.0)*rho - RHO_0/18;
    #endif //_BC_FORCES 

    break;
case 0b0000010110110110:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[11] + pop[12] + pop[13] + pop[15] + pop[16] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13]);
    m_xz_I = inv_rho_I * (pop[9] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);

    rho = (12*(5*rho_I + m_xx_I*rho_I - 2*m_xy_I*rho_I + 2*m_xz_I*rho_I + m_yy_I*rho_I - 2*m_yz_I*rho_I + m_zz_I*rho_I - omegaVar*m_xx_I*rho_I + 2*omegaVar*m_xy_I*rho_I - 2*omegaVar*m_xz_I*rho_I - omegaVar*m_yy_I*rho_I + 2*omegaVar*m_yz_I*rho_I - omegaVar*m_zz_I*rho_I))/(3*omegaVar + 52);

    m_xx_t45 = (472*m_xx_I - 112*m_xy_I + 112*m_xz_I + 4*m_yy_I + 44*m_yz_I + 4*m_zz_I + 23*omegaVar*m_xx_I + 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - 4*omegaVar*m_yy_I + 11*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 20)/(90*(m_xx_I - 2*m_xy_I + 2*m_xz_I + m_yy_I - 2*m_yz_I + m_zz_I - omegaVar*m_xx_I + 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - omegaVar*m_yy_I + 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_xy_t90 = (1108*m_xy_I - 112*m_xx_I - 16*m_xz_I - 112*m_yy_I + 16*m_yz_I + 44*m_zz_I + 2*omegaVar*m_xx_I + 47*omegaVar*m_xy_I + 16*omegaVar*m_xz_I + 2*omegaVar*m_yy_I - 16*omegaVar*m_yz_I + 11*omegaVar*m_zz_I - 40)/(180*(m_xx_I - 2*m_xy_I + 2*m_xz_I + m_yy_I - 2*m_yz_I + m_zz_I - omegaVar*m_xx_I + 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - omegaVar*m_yy_I + 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_xz_t90 = (112*m_xx_I - 16*m_xy_I + 1108*m_xz_I - 44*m_yy_I - 16*m_yz_I + 112*m_zz_I - 2*omegaVar*m_xx_I + 16*omegaVar*m_xy_I + 47*omegaVar*m_xz_I - 11*omegaVar*m_yy_I + 16*omegaVar*m_yz_I - 2*omegaVar*m_zz_I + 40)/(180*(m_xx_I - 2*m_xy_I + 2*m_xz_I + m_yy_I - 2*m_yz_I + m_zz_I - omegaVar*m_xx_I + 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - omegaVar*m_yy_I + 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_yy_t45 = (4*m_xx_I - 112*m_xy_I - 44*m_xz_I + 472*m_yy_I - 112*m_yz_I + 4*m_zz_I - 4*omegaVar*m_xx_I + 2*omegaVar*m_xy_I - 11*omegaVar*m_xz_I + 23*omegaVar*m_yy_I + 2*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 20)/(90*(m_xx_I - 2*m_xy_I + 2*m_xz_I + m_yy_I - 2*m_yz_I + m_zz_I - omegaVar*m_xx_I + 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - omegaVar*m_yy_I + 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_yz_t90 = (44*m_xx_I + 16*m_xy_I - 16*m_xz_I - 112*m_yy_I + 1108*m_yz_I - 112*m_zz_I + 11*omegaVar*m_xx_I - 16*omegaVar*m_xy_I + 16*omegaVar*m_xz_I + 2*omegaVar*m_yy_I + 47*omegaVar*m_yz_I + 2*omegaVar*m_zz_I - 40)/(180*(m_xx_I - 2*m_xy_I + 2*m_xz_I + m_yy_I - 2*m_yz_I + m_zz_I - omegaVar*m_xx_I + 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - omegaVar*m_yy_I + 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_zz_t45 = (4*m_xx_I + 44*m_xy_I + 112*m_xz_I + 4*m_yy_I - 112*m_yz_I + 472*m_zz_I - 4*omegaVar*m_xx_I + 11*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - 4*omegaVar*m_yy_I + 2*omegaVar*m_yz_I + 23*omegaVar*m_zz_I + 20)/(90*(m_xx_I - 2*m_xy_I + 2*m_xz_I + m_yy_I - 2*m_yz_I + m_zz_I - omegaVar*m_xx_I + 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - omegaVar*m_yy_I + 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = (m_xx_t45/3 - m_xy_t90/2 + m_xz_t90/2 + m_yy_t45/12 + m_zz_t45/12 - (omegaVar*m_xx_t45)/6 + (omegaVar*m_xy_t90)/4 - (omegaVar*m_xz_t90)/4 - (omegaVar*m_yy_t45)/24 - (omegaVar*m_zz_t45)/24 + 1.0/9.0)*rho - RHO_0/9;
    L_BC_Fy = (m_xy_t90/2 - m_xx_t45/12 - m_yy_t45/3 + m_yz_t90/2 - m_zz_t45/12 + (omegaVar*m_xx_t45)/24 - (omegaVar*m_xy_t90)/4 + (omegaVar*m_yy_t45)/6 - (omegaVar*m_yz_t90)/4 + (omegaVar*m_zz_t45)/24 - 1.0/9.0)*rho + RHO_0/9;
    L_BC_Fz = (m_xx_t45/12 + m_xz_t90/2 + m_yy_t45/12 - m_yz_t90/2 + m_zz_t45/3 - (omegaVar*m_xx_t45)/24 - (omegaVar*m_xz_t90)/4 - (omegaVar*m_yy_t45)/24 + (omegaVar*m_yz_t90)/4 - (omegaVar*m_zz_t45)/6 + 1.0/9.0)*rho - RHO_0/9;
    #endif //_BC_FORCES 

    break;
case 0b0000010110110111:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[4] + pop[5] + pop[6] + pop[8] + pop[12] + pop[13] + pop[15] + pop[16] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[8] - pop[13]);
    m_xz_I = inv_rho_I * (-pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[12] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);

    rho = (18*(19*rho_I + 12*m_xx_I*rho_I + 33*m_xz_I*rho_I + 6*m_yy_I*rho_I + 12*m_zz_I*rho_I - 12*omegaVar*m_xx_I*rho_I - 33*omegaVar*m_xz_I*rho_I - 6*omegaVar*m_yy_I*rho_I - 12*omegaVar*m_zz_I*rho_I))/(59*omegaVar + 207);

    m_xx_t45 = (2*(306*m_xx_I + 324*m_xz_I - 54*m_yy_I + 99*m_zz_I + 58*omegaVar*m_xx_I + 12*omegaVar*m_xz_I - 30*omegaVar*m_yy_I - omegaVar*m_zz_I + 36))/(27*(12*m_xx_I + 33*m_xz_I + 6*m_yy_I + 12*m_zz_I - 12*omegaVar*m_xx_I - 33*omegaVar*m_xz_I - 6*omegaVar*m_yy_I - 12*omegaVar*m_zz_I + 19));
    m_xy_t90 = (m_xy_I*(59*omegaVar + 207))/(9*(12*m_xx_I + 33*m_xz_I + 6*m_yy_I + 12*m_zz_I - 12*omegaVar*m_xx_I - 33*omegaVar*m_xz_I - 6*omegaVar*m_yy_I - 12*omegaVar*m_zz_I + 19));
    m_xz_t90 = (216*m_xx_I + 801*m_xz_I - 99*m_yy_I + 216*m_zz_I + 8*omegaVar*m_xx_I + 81*omegaVar*m_xz_I - 55*omegaVar*m_yy_I + 8*omegaVar*m_zz_I + 66)/(18*(12*m_xx_I + 33*m_xz_I + 6*m_yy_I + 12*m_zz_I - 12*omegaVar*m_xx_I - 33*omegaVar*m_xz_I - 6*omegaVar*m_yy_I - 12*omegaVar*m_zz_I + 19));
    m_yy_t45 = -(36*m_xx_I + 99*m_xz_I - 189*m_yy_I + 36*m_zz_I + 20*omegaVar*m_xx_I + 55*omegaVar*m_xz_I - 49*omegaVar*m_yy_I + 20*omegaVar*m_zz_I - 12)/(9*(12*m_xx_I + 33*m_xz_I + 6*m_yy_I + 12*m_zz_I - 12*omegaVar*m_xx_I - 33*omegaVar*m_xz_I - 6*omegaVar*m_yy_I - 12*omegaVar*m_zz_I + 19));
    m_yz_t90 = (m_yz_I*(59*omegaVar + 207))/(9*(12*m_xx_I + 33*m_xz_I + 6*m_yy_I + 12*m_zz_I - 12*omegaVar*m_xx_I - 33*omegaVar*m_xz_I - 6*omegaVar*m_yy_I - 12*omegaVar*m_zz_I + 19));
    m_zz_t45 = (2*(99*m_xx_I + 324*m_xz_I - 54*m_yy_I + 306*m_zz_I - omegaVar*m_xx_I + 12*omegaVar*m_xz_I - 30*omegaVar*m_yy_I + 58*omegaVar*m_zz_I + 36))/(27*(12*m_xx_I + 33*m_xz_I + 6*m_yy_I + 12*m_zz_I - 12*omegaVar*m_xx_I - 33*omegaVar*m_xz_I - 6*omegaVar*m_yy_I - 12*omegaVar*m_zz_I + 19));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = (m_xy_t90*rho*(omegaVar - 2))/2;
    L_BC_Fy = ((omegaVar*m_yy_t45)/2 - m_yy_t45 - 1.0/3.0)*rho + RHO_0/3;
    L_BC_Fz = (m_yz_t90*rho*(omegaVar - 2))/2;
    #endif //_BC_FORCES 

    break;
case 0b0000010110111000:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[9] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[9] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] - (1.0/3.0)*pop[9] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);

    rho = (12*(5*rho_I + m_xx_I*rho_I + 2*m_xy_I*rho_I + 2*m_xz_I*rho_I + m_yy_I*rho_I - 2*m_yz_I*rho_I + m_zz_I*rho_I - omegaVar*m_xx_I*rho_I - 2*omegaVar*m_xy_I*rho_I - 2*omegaVar*m_xz_I*rho_I - omegaVar*m_yy_I*rho_I + 2*omegaVar*m_yz_I*rho_I - omegaVar*m_zz_I*rho_I))/(3*omegaVar + 52);

    m_xx_t45 = (472*m_xx_I + 112*m_xy_I + 112*m_xz_I + 4*m_yy_I + 44*m_yz_I + 4*m_zz_I + 23*omegaVar*m_xx_I - 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - 4*omegaVar*m_yy_I + 11*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 20)/(90*(m_xx_I + 2*m_xy_I + 2*m_xz_I + m_yy_I - 2*m_yz_I + m_zz_I - omegaVar*m_xx_I - 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - omegaVar*m_yy_I + 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_xy_t90 = (112*m_xx_I + 1108*m_xy_I + 16*m_xz_I + 112*m_yy_I - 16*m_yz_I - 44*m_zz_I - 2*omegaVar*m_xx_I + 47*omegaVar*m_xy_I - 16*omegaVar*m_xz_I - 2*omegaVar*m_yy_I + 16*omegaVar*m_yz_I - 11*omegaVar*m_zz_I + 40)/(180*(m_xx_I + 2*m_xy_I + 2*m_xz_I + m_yy_I - 2*m_yz_I + m_zz_I - omegaVar*m_xx_I - 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - omegaVar*m_yy_I + 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_xz_t90 = (112*m_xx_I + 16*m_xy_I + 1108*m_xz_I - 44*m_yy_I - 16*m_yz_I + 112*m_zz_I - 2*omegaVar*m_xx_I - 16*omegaVar*m_xy_I + 47*omegaVar*m_xz_I - 11*omegaVar*m_yy_I + 16*omegaVar*m_yz_I - 2*omegaVar*m_zz_I + 40)/(180*(m_xx_I + 2*m_xy_I + 2*m_xz_I + m_yy_I - 2*m_yz_I + m_zz_I - omegaVar*m_xx_I - 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - omegaVar*m_yy_I + 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_yy_t45 = (4*m_xx_I + 112*m_xy_I - 44*m_xz_I + 472*m_yy_I - 112*m_yz_I + 4*m_zz_I - 4*omegaVar*m_xx_I - 2*omegaVar*m_xy_I - 11*omegaVar*m_xz_I + 23*omegaVar*m_yy_I + 2*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 20)/(90*(m_xx_I + 2*m_xy_I + 2*m_xz_I + m_yy_I - 2*m_yz_I + m_zz_I - omegaVar*m_xx_I - 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - omegaVar*m_yy_I + 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_yz_t90 = (44*m_xx_I - 16*m_xy_I - 16*m_xz_I - 112*m_yy_I + 1108*m_yz_I - 112*m_zz_I + 11*omegaVar*m_xx_I + 16*omegaVar*m_xy_I + 16*omegaVar*m_xz_I + 2*omegaVar*m_yy_I + 47*omegaVar*m_yz_I + 2*omegaVar*m_zz_I - 40)/(180*(m_xx_I + 2*m_xy_I + 2*m_xz_I + m_yy_I - 2*m_yz_I + m_zz_I - omegaVar*m_xx_I - 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - omegaVar*m_yy_I + 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_zz_t45 = (4*m_xx_I - 44*m_xy_I + 112*m_xz_I + 4*m_yy_I - 112*m_yz_I + 472*m_zz_I - 4*omegaVar*m_xx_I - 11*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - 4*omegaVar*m_yy_I + 2*omegaVar*m_yz_I + 23*omegaVar*m_zz_I + 20)/(90*(m_xx_I + 2*m_xy_I + 2*m_xz_I + m_yy_I - 2*m_yz_I + m_zz_I - omegaVar*m_xx_I - 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - omegaVar*m_yy_I + 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = (m_xx_t45/3 + m_xy_t90/2 + m_xz_t90/2 + m_yy_t45/12 + m_zz_t45/12 - (omegaVar*m_xx_t45)/6 - (omegaVar*m_xy_t90)/4 - (omegaVar*m_xz_t90)/4 - (omegaVar*m_yy_t45)/24 - (omegaVar*m_zz_t45)/24 + 1.0/9.0)*rho - RHO_0/9;
    L_BC_Fy = -(rho*(omegaVar - 2)*(m_xx_t45 + 2*m_xy_t90 + 2*m_yz_t90 - m_zz_t45))/8;
    L_BC_Fz = (m_xx_t45/12 + m_xz_t90/2 + m_yy_t45/12 - m_yz_t90/2 + m_zz_t45/3 - (omegaVar*m_xx_t45)/24 - (omegaVar*m_xz_t90)/4 - (omegaVar*m_yy_t45)/24 + (omegaVar*m_yz_t90)/4 - (omegaVar*m_zz_t45)/6 + 1.0/9.0)*rho - RHO_0/9;
    #endif //_BC_FORCES 

    break;
case 0b0000010110111001:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[9] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[9] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (-pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] - (1.0/3.0)*pop[9] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);

    rho = (3*(15*rho_I + 11*m_xx_I*rho_I + 30*m_xy_I*rho_I + 6*m_xz_I*rho_I + 11*m_yy_I*rho_I - 6*m_yz_I*rho_I - m_zz_I*rho_I - 11*omegaVar*m_xx_I*rho_I - 30*omegaVar*m_xy_I*rho_I - 6*omegaVar*m_xz_I*rho_I - 11*omegaVar*m_yy_I*rho_I + 6*omegaVar*m_yz_I*rho_I + omegaVar*m_zz_I*rho_I))/(9*omegaVar + 31);

    m_xx_t45 = (2*(1206*m_xx_I + 1260*m_xy_I + 252*m_xz_I + 369*m_yy_I + 27*m_yz_I - 135*m_zz_I + 194*omegaVar*m_xx_I - 60*omegaVar*m_xy_I - 12*omegaVar*m_xz_I - 49*omegaVar*m_yy_I + 93*omegaVar*m_yz_I - 25*omegaVar*m_zz_I + 165))/(135*(11*m_xx_I + 30*m_xy_I + 6*m_xz_I + 11*m_yy_I - 6*m_yz_I - m_zz_I - 11*omegaVar*m_xx_I - 30*omegaVar*m_xy_I - 6*omegaVar*m_xz_I - 11*omegaVar*m_yy_I + 6*omegaVar*m_yz_I + omegaVar*m_zz_I + 15));
    m_xy_t90 = (84*m_xx_I + 339*m_xy_I + 12*m_xz_I + 84*m_yy_I - 12*m_yz_I - 33*m_zz_I - 4*omegaVar*m_xx_I + 21*omegaVar*m_xy_I - 12*omegaVar*m_xz_I - 4*omegaVar*m_yy_I + 12*omegaVar*m_yz_I - 7*omegaVar*m_zz_I + 30)/(9*(11*m_xx_I + 30*m_xy_I + 6*m_xz_I + 11*m_yy_I - 6*m_yz_I - m_zz_I - 11*omegaVar*m_xx_I - 30*omegaVar*m_xy_I - 6*omegaVar*m_xz_I - 11*omegaVar*m_yy_I + 6*omegaVar*m_yz_I + omegaVar*m_zz_I + 15));
    m_xz_t90 = (84*m_xx_I + 60*m_xy_I + 663*m_xz_I - 9*m_yy_I - 12*m_yz_I + 60*m_zz_I - 4*omegaVar*m_xx_I - 60*omegaVar*m_xy_I + 177*omegaVar*m_xz_I - 31*omegaVar*m_yy_I + 12*omegaVar*m_yz_I + 20*omegaVar*m_zz_I + 30)/(45*(11*m_xx_I + 30*m_xy_I + 6*m_xz_I + 11*m_yy_I - 6*m_yz_I - m_zz_I - 11*omegaVar*m_xx_I - 30*omegaVar*m_xy_I - 6*omegaVar*m_xz_I - 11*omegaVar*m_yy_I + 6*omegaVar*m_yz_I + omegaVar*m_zz_I + 15));
    m_yy_t45 = -(2*(27*m_xz_I - 1260*m_xy_I - 369*m_xx_I - 1206*m_yy_I + 252*m_yz_I + 135*m_zz_I + 49*omegaVar*m_xx_I + 60*omegaVar*m_xy_I + 93*omegaVar*m_xz_I - 194*omegaVar*m_yy_I - 12*omegaVar*m_yz_I + 25*omegaVar*m_zz_I - 165))/(135*(11*m_xx_I + 30*m_xy_I + 6*m_xz_I + 11*m_yy_I - 6*m_yz_I - m_zz_I - 11*omegaVar*m_xx_I - 30*omegaVar*m_xy_I - 6*omegaVar*m_xz_I - 11*omegaVar*m_yy_I + 6*omegaVar*m_yz_I + omegaVar*m_zz_I + 15));
    m_yz_t90 = (9*m_xx_I - 60*m_xy_I - 12*m_xz_I - 84*m_yy_I + 663*m_yz_I - 60*m_zz_I + 31*omegaVar*m_xx_I + 60*omegaVar*m_xy_I + 12*omegaVar*m_xz_I + 4*omegaVar*m_yy_I + 177*omegaVar*m_yz_I - 20*omegaVar*m_zz_I - 30)/(45*(11*m_xx_I + 30*m_xy_I + 6*m_xz_I + 11*m_yy_I - 6*m_yz_I - m_zz_I - 11*omegaVar*m_xx_I - 30*omegaVar*m_xy_I - 6*omegaVar*m_xz_I - 11*omegaVar*m_yy_I + 6*omegaVar*m_yz_I + omegaVar*m_zz_I + 15));
    m_zz_t45 = -(2*(27*m_xx_I + 99*m_xy_I - 36*m_xz_I + 27*m_yy_I + 36*m_yz_I - 180*m_zz_I + 5*omegaVar*m_xx_I + 21*omegaVar*m_xy_I - 12*omegaVar*m_xz_I + 5*omegaVar*m_yy_I + 12*omegaVar*m_yz_I - 52*omegaVar*m_zz_I + 3))/(27*(11*m_xx_I + 30*m_xy_I + 6*m_xz_I + 11*m_yy_I - 6*m_yz_I - m_zz_I - 11*omegaVar*m_xx_I - 30*omegaVar*m_xy_I - 6*omegaVar*m_xz_I - 11*omegaVar*m_yy_I + 6*omegaVar*m_yz_I + omegaVar*m_zz_I + 15));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = (m_xx_t45/6 + m_xz_t90/2 - m_yy_t45/12 + m_zz_t45/6 - (omegaVar*m_xx_t45)/12 - (omegaVar*m_xz_t90)/4 + (omegaVar*m_yy_t45)/24 - (omegaVar*m_zz_t45)/12 + 1.0/18.0)*rho - RHO_0/18;
    L_BC_Fy = (m_xx_t45/12 - m_yy_t45/6 + m_yz_t90/2 - m_zz_t45/6 - (omegaVar*m_xx_t45)/24 + (omegaVar*m_yy_t45)/12 - (omegaVar*m_yz_t90)/4 + (omegaVar*m_zz_t45)/12 - 1.0/18.0)*rho + RHO_0/18;
    L_BC_Fz = (m_xx_t45/12 + m_xz_t90/2 + m_yy_t45/12 - m_yz_t90/2 + m_zz_t45/3 - (omegaVar*m_xx_t45)/24 - (omegaVar*m_xz_t90)/4 - (omegaVar*m_yy_t45)/24 + (omegaVar*m_yz_t90)/4 - (omegaVar*m_zz_t45)/6 + 1.0/9.0)*rho - RHO_0/9;
    #endif //_BC_FORCES 

    break;
case 0b0000010110111010:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[9] + pop[11] + pop[12] + pop[13] + pop[15] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[9] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[15] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] - pop[13]);
    m_xz_I = inv_rho_I * (pop[9] - pop[15]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] - (1.0/3.0)*pop[9] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] - (1.0/3.0)*pop[15] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[18]);

    rho = (36*(107*rho_I + 72*m_xx_I*rho_I + 12*m_yy_I*rho_I - 33*m_yz_I*rho_I + 12*m_zz_I*rho_I - 72*omegaVar*m_xx_I*rho_I - 12*omegaVar*m_yy_I*rho_I + 33*omegaVar*m_yz_I*rho_I - 12*omegaVar*m_zz_I*rho_I))/(313*omegaVar + 2790);

    m_xx_t45 = (2358*m_xx_I - 72*m_yy_I + 198*m_yz_I - 72*m_zz_I + 49*omegaVar*m_xx_I - 44*omegaVar*m_yy_I + 121*omegaVar*m_yz_I - 44*omegaVar*m_zz_I + 288)/(18*(72*m_xx_I + 12*m_yy_I - 33*m_yz_I + 12*m_zz_I - 72*omegaVar*m_xx_I - 12*omegaVar*m_yy_I + 33*omegaVar*m_yz_I - 12*omegaVar*m_zz_I + 107));
    m_xy_t90 = (m_xy_I*(313*omegaVar + 2790))/(18*(72*m_xx_I + 12*m_yy_I - 33*m_yz_I + 12*m_zz_I - 72*omegaVar*m_xx_I - 12*omegaVar*m_yy_I + 33*omegaVar*m_yz_I - 12*omegaVar*m_zz_I + 107));
    m_xz_t90 = (m_xz_I*(313*omegaVar + 2790))/(18*(72*m_xx_I + 12*m_yy_I - 33*m_yz_I + 12*m_zz_I - 72*omegaVar*m_xx_I - 12*omegaVar*m_yy_I + 33*omegaVar*m_yz_I - 12*omegaVar*m_zz_I + 107));
    m_yy_t45 = -(108*m_xx_I - 2772*m_yy_I + 648*m_yz_I + 18*m_zz_I + 66*omegaVar*m_xx_I - 302*omegaVar*m_yy_I + 48*omegaVar*m_yz_I + 11*omegaVar*m_zz_I - 72)/(27*(72*m_xx_I + 12*m_yy_I - 33*m_yz_I + 12*m_zz_I - 72*omegaVar*m_xx_I - 12*omegaVar*m_yy_I + 33*omegaVar*m_yz_I - 12*omegaVar*m_zz_I + 107));
    m_yz_t90 = -(432*m_yy_I - 198*m_xx_I - 3978*m_yz_I + 432*m_zz_I - 121*omegaVar*m_xx_I + 32*omegaVar*m_yy_I - 401*omegaVar*m_yz_I + 32*omegaVar*m_zz_I + 132)/(36*(72*m_xx_I + 12*m_yy_I - 33*m_yz_I + 12*m_zz_I - 72*omegaVar*m_xx_I - 12*omegaVar*m_yy_I + 33*omegaVar*m_yz_I - 12*omegaVar*m_zz_I + 107));
    m_zz_t45 = -(108*m_xx_I + 18*m_yy_I + 648*m_yz_I - 2772*m_zz_I + 66*omegaVar*m_xx_I + 11*omegaVar*m_yy_I + 48*omegaVar*m_yz_I - 302*omegaVar*m_zz_I - 72)/(27*(72*m_xx_I + 12*m_yy_I - 33*m_yz_I + 12*m_zz_I - 72*omegaVar*m_xx_I - 12*omegaVar*m_yy_I + 33*omegaVar*m_yz_I - 12*omegaVar*m_zz_I + 107));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = (m_xx_t45 - (omegaVar*m_xx_t45)/2 + 1.0/3.0)*rho - RHO_0/3;
    L_BC_Fy = (m_xx_t45/12 + m_xy_t90 - m_yy_t45/6 + m_yz_t90/2 - m_zz_t45/6 - (omegaVar*m_xx_t45)/24 - (omegaVar*m_xy_t90)/2 + (omegaVar*m_yy_t45)/12 - (omegaVar*m_yz_t90)/4 + (omegaVar*m_zz_t45)/12 - 1.0/18.0)*rho + RHO_0/18;
    L_BC_Fz = (m_xz_t90 - m_xx_t45/12 + m_yy_t45/6 - m_yz_t90/2 + m_zz_t45/6 + (omegaVar*m_xx_t45)/24 - (omegaVar*m_xz_t90)/2 - (omegaVar*m_yy_t45)/12 + (omegaVar*m_yz_t90)/4 - (omegaVar*m_zz_t45)/12 + 1.0/18.0)*rho - RHO_0/18;
    #endif //_BC_FORCES 

    break;
case 0b0000010110111011:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[4] + pop[5] + pop[6] + pop[9] + pop[12] + pop[13] + pop[15] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[9] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[15] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (-pop[13]);
    m_xz_I = inv_rho_I * (pop[9] - pop[15]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] - (1.0/3.0)*pop[9] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] - (1.0/3.0)*pop[15] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[12] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[18]);

    rho = (36*(23*rho_I + 24*m_xx_I*rho_I + 57*m_xy_I*rho_I + 24*m_yy_I*rho_I - 6*m_zz_I*rho_I - 24*omegaVar*m_xx_I*rho_I - 57*omegaVar*m_xy_I*rho_I - 24*omegaVar*m_yy_I*rho_I + 6*omegaVar*m_zz_I*rho_I))/(5*(43*omegaVar + 72));

    m_xx_t45 = (936*m_xx_I + 1008*m_xy_I + 216*m_yy_I - 144*m_zz_I + 239*omegaVar*m_xx_I - 158*omegaVar*m_xy_I - 191*omegaVar*m_yy_I - 6*omegaVar*m_zz_I + 192)/(36*(24*m_xx_I + 57*m_xy_I + 24*m_yy_I - 6*m_zz_I - 24*omegaVar*m_xx_I - 57*omegaVar*m_xy_I - 24*omegaVar*m_yy_I + 6*omegaVar*m_zz_I + 23));
    m_xy_t90 = (504*m_xx_I + 2412*m_xy_I + 504*m_yy_I - 216*m_zz_I - 79*omegaVar*m_xx_I + 538*omegaVar*m_xy_I - 79*omegaVar*m_yy_I - 34*omegaVar*m_zz_I + 228)/(36*(24*m_xx_I + 57*m_xy_I + 24*m_yy_I - 6*m_zz_I - 24*omegaVar*m_xx_I - 57*omegaVar*m_xy_I - 24*omegaVar*m_yy_I + 6*omegaVar*m_zz_I + 23));
    m_xz_t90 = (5*m_xz_I*(43*omegaVar + 72))/(18*(24*m_xx_I + 57*m_xy_I + 24*m_yy_I - 6*m_zz_I - 24*omegaVar*m_xx_I - 57*omegaVar*m_xy_I - 24*omegaVar*m_yy_I + 6*omegaVar*m_zz_I + 23));
    m_yy_t45 = (216*m_xx_I + 1008*m_xy_I + 936*m_yy_I - 144*m_zz_I - 191*omegaVar*m_xx_I - 158*omegaVar*m_xy_I + 239*omegaVar*m_yy_I - 6*omegaVar*m_zz_I + 192)/(36*(24*m_xx_I + 57*m_xy_I + 24*m_yy_I - 6*m_zz_I - 24*omegaVar*m_xx_I - 57*omegaVar*m_xy_I - 24*omegaVar*m_yy_I + 6*omegaVar*m_zz_I + 23));
    m_yz_t90 = (5*m_yz_I*(43*omegaVar + 72))/(18*(24*m_xx_I + 57*m_xy_I + 24*m_yy_I - 6*m_zz_I - 24*omegaVar*m_xx_I - 57*omegaVar*m_xy_I - 24*omegaVar*m_yy_I + 6*omegaVar*m_zz_I + 23));
    m_zz_t45 = -(72*m_xx_I + 216*m_xy_I + 72*m_yy_I - 288*m_zz_I + 3*omegaVar*m_xx_I + 34*omegaVar*m_xy_I + 3*omegaVar*m_yy_I - 162*omegaVar*m_zz_I + 24)/(18*(24*m_xx_I + 57*m_xy_I + 24*m_yy_I - 6*m_zz_I - 24*omegaVar*m_xx_I - 57*omegaVar*m_xy_I - 24*omegaVar*m_yy_I + 6*omegaVar*m_zz_I + 23));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = ((5*m_xx_t45)/6 - m_xy_t90/2 - m_yy_t45/6 + m_zz_t45/12 - (5*omegaVar*m_xx_t45)/12 + (omegaVar*m_xy_t90)/4 + (omegaVar*m_yy_t45)/12 - (omegaVar*m_zz_t45)/24 + 5/18)*rho - (5*RHO_0)/18;
    L_BC_Fy = (m_xx_t45/6 + m_xy_t90/2 - (5*m_yy_t45)/6 - m_zz_t45/12 - (omegaVar*m_xx_t45)/12 - (omegaVar*m_xy_t90)/4 + (5*omegaVar*m_yy_t45)/12 + (omegaVar*m_zz_t45)/24 - 5/18)*rho + (5*RHO_0)/18;
    L_BC_Fz = -(rho*(m_xz_t90 - m_yz_t90)*(omegaVar - 2))/2;
    #endif //_BC_FORCES 

    break;
case 0b0000010110111100:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[9] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[9] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16]);
    m_xy_I = inv_rho_I * (pop[7] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] - (1.0/3.0)*pop[9] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16]);

    rho = (3*(15*rho_I - m_xx_I*rho_I + 6*m_xy_I*rho_I + 6*m_xz_I*rho_I + 11*m_yy_I*rho_I - 30*m_yz_I*rho_I + 11*m_zz_I*rho_I + omegaVar*m_xx_I*rho_I - 6*omegaVar*m_xy_I*rho_I - 6*omegaVar*m_xz_I*rho_I - 11*omegaVar*m_yy_I*rho_I + 30*omegaVar*m_yz_I*rho_I - 11*omegaVar*m_zz_I*rho_I))/(9*omegaVar + 31);

    m_xx_t45 = (2*(180*m_xx_I + 36*m_xy_I + 36*m_xz_I - 27*m_yy_I + 99*m_yz_I - 27*m_zz_I + 52*omegaVar*m_xx_I + 12*omegaVar*m_xy_I + 12*omegaVar*m_xz_I - 5*omegaVar*m_yy_I + 21*omegaVar*m_yz_I - 5*omegaVar*m_zz_I - 3))/(27*(6*m_xy_I - m_xx_I + 6*m_xz_I + 11*m_yy_I - 30*m_yz_I + 11*m_zz_I + omegaVar*m_xx_I - 6*omegaVar*m_xy_I - 6*omegaVar*m_xz_I - 11*omegaVar*m_yy_I + 30*omegaVar*m_yz_I - 11*omegaVar*m_zz_I + 15));
    m_xy_t90 = (60*m_xx_I + 663*m_xy_I + 12*m_xz_I + 84*m_yy_I - 60*m_yz_I - 9*m_zz_I + 20*omegaVar*m_xx_I + 177*omegaVar*m_xy_I - 12*omegaVar*m_xz_I - 4*omegaVar*m_yy_I + 60*omegaVar*m_yz_I - 31*omegaVar*m_zz_I + 30)/(45*(6*m_xy_I - m_xx_I + 6*m_xz_I + 11*m_yy_I - 30*m_yz_I + 11*m_zz_I + omegaVar*m_xx_I - 6*omegaVar*m_xy_I - 6*omegaVar*m_xz_I - 11*omegaVar*m_yy_I + 30*omegaVar*m_yz_I - 11*omegaVar*m_zz_I + 15));
    m_xz_t90 = (60*m_xx_I + 12*m_xy_I + 663*m_xz_I - 9*m_yy_I - 60*m_yz_I + 84*m_zz_I + 20*omegaVar*m_xx_I - 12*omegaVar*m_xy_I + 177*omegaVar*m_xz_I - 31*omegaVar*m_yy_I + 60*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 30)/(45*(6*m_xy_I - m_xx_I + 6*m_xz_I + 11*m_yy_I - 30*m_yz_I + 11*m_zz_I + omegaVar*m_xx_I - 6*omegaVar*m_xy_I - 6*omegaVar*m_xz_I - 11*omegaVar*m_yy_I + 30*omegaVar*m_yz_I - 11*omegaVar*m_zz_I + 15));
    m_yy_t45 = -(2*(135*m_xx_I - 252*m_xy_I + 27*m_xz_I - 1206*m_yy_I + 1260*m_yz_I - 369*m_zz_I + 25*omegaVar*m_xx_I + 12*omegaVar*m_xy_I + 93*omegaVar*m_xz_I - 194*omegaVar*m_yy_I - 60*omegaVar*m_yz_I + 49*omegaVar*m_zz_I - 165))/(135*(6*m_xy_I - m_xx_I + 6*m_xz_I + 11*m_yy_I - 30*m_yz_I + 11*m_zz_I + omegaVar*m_xx_I - 6*omegaVar*m_xy_I - 6*omegaVar*m_xz_I - 11*omegaVar*m_yy_I + 30*omegaVar*m_yz_I - 11*omegaVar*m_zz_I + 15));
    m_yz_t90 = (33*m_xx_I - 12*m_xy_I - 12*m_xz_I - 84*m_yy_I + 339*m_yz_I - 84*m_zz_I + 7*omegaVar*m_xx_I + 12*omegaVar*m_xy_I + 12*omegaVar*m_xz_I + 4*omegaVar*m_yy_I + 21*omegaVar*m_yz_I + 4*omegaVar*m_zz_I - 30)/(9*(6*m_xy_I - m_xx_I + 6*m_xz_I + 11*m_yy_I - 30*m_yz_I + 11*m_zz_I + omegaVar*m_xx_I - 6*omegaVar*m_xy_I - 6*omegaVar*m_xz_I - 11*omegaVar*m_yy_I + 30*omegaVar*m_yz_I - 11*omegaVar*m_zz_I + 15));
    m_zz_t45 = -(2*(135*m_xx_I + 27*m_xy_I - 252*m_xz_I - 369*m_yy_I + 1260*m_yz_I - 1206*m_zz_I + 25*omegaVar*m_xx_I + 93*omegaVar*m_xy_I + 12*omegaVar*m_xz_I + 49*omegaVar*m_yy_I - 60*omegaVar*m_yz_I - 194*omegaVar*m_zz_I - 165))/(135*(6*m_xy_I - m_xx_I + 6*m_xz_I + 11*m_yy_I - 30*m_yz_I + 11*m_zz_I + omegaVar*m_xx_I - 6*omegaVar*m_xy_I - 6*omegaVar*m_xz_I - 11*omegaVar*m_yy_I + 30*omegaVar*m_yz_I - 11*omegaVar*m_zz_I + 15));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = (m_xx_t45/3 + m_xy_t90/2 + m_xz_t90/2 + m_yy_t45/12 + m_zz_t45/12 - (omegaVar*m_xx_t45)/6 - (omegaVar*m_xy_t90)/4 - (omegaVar*m_xz_t90)/4 - (omegaVar*m_yy_t45)/24 - (omegaVar*m_zz_t45)/24 + 1.0/9.0)*rho - RHO_0/9;
    L_BC_Fy = (m_xx_t45/6 + m_xy_t90/2 + m_yy_t45/6 - m_zz_t45/12 - (omegaVar*m_xx_t45)/12 - (omegaVar*m_xy_t90)/4 - (omegaVar*m_yy_t45)/12 + (omegaVar*m_zz_t45)/24 + 1.0/18.0)*rho - RHO_0/18;
    L_BC_Fz = (m_xx_t45/6 + m_xz_t90/2 - m_yy_t45/12 + m_zz_t45/6 - (omegaVar*m_xx_t45)/12 - (omegaVar*m_xz_t90)/4 + (omegaVar*m_yy_t45)/24 - (omegaVar*m_zz_t45)/12 + 1.0/18.0)*rho - RHO_0/18;
    #endif //_BC_FORCES 

    break;
case 0b0000010110111101:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16]);
    m_xy_I = inv_rho_I * (-pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (-pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16]);

    rho = (6*(rho_I + m_xx_I*rho_I + 2*m_xy_I*rho_I + 2*m_xz_I*rho_I + m_yy_I*rho_I - 2*m_yz_I*rho_I + m_zz_I*rho_I - omegaVar*m_xx_I*rho_I - 2*omegaVar*m_xy_I*rho_I - 2*omegaVar*m_xz_I*rho_I - omegaVar*m_yy_I*rho_I + 2*omegaVar*m_yz_I*rho_I - omegaVar*m_zz_I*rho_I))/(3*omegaVar + 2);

    m_xx_t45 = (8*m_xx_I + 8*m_xy_I + 8*m_xz_I + 2*m_yy_I - 2*m_yz_I + 2*m_zz_I + 7*omegaVar*m_xx_I + 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - 2*omegaVar*m_yy_I + 7*omegaVar*m_yz_I - 2*omegaVar*m_zz_I + 2)/(9*(m_xx_I + 2*m_xy_I + 2*m_xz_I + m_yy_I - 2*m_yz_I + m_zz_I - omegaVar*m_xx_I - 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - omegaVar*m_yy_I + 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 1));
    m_xy_t90 = (8*m_xx_I + 26*m_xy_I + 8*m_xz_I + 8*m_yy_I - 8*m_yz_I + 2*m_zz_I + 2*omegaVar*m_xx_I + 19*omegaVar*m_xy_I - 8*omegaVar*m_xz_I + 2*omegaVar*m_yy_I + 8*omegaVar*m_yz_I - 7*omegaVar*m_zz_I + 4)/(18*(m_xx_I + 2*m_xy_I + 2*m_xz_I + m_yy_I - 2*m_yz_I + m_zz_I - omegaVar*m_xx_I - 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - omegaVar*m_yy_I + 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 1));
    m_xz_t90 = (8*m_xx_I + 8*m_xy_I + 26*m_xz_I + 2*m_yy_I - 8*m_yz_I + 8*m_zz_I + 2*omegaVar*m_xx_I - 8*omegaVar*m_xy_I + 19*omegaVar*m_xz_I - 7*omegaVar*m_yy_I + 8*omegaVar*m_yz_I + 2*omegaVar*m_zz_I + 4)/(18*(m_xx_I + 2*m_xy_I + 2*m_xz_I + m_yy_I - 2*m_yz_I + m_zz_I - omegaVar*m_xx_I - 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - omegaVar*m_yy_I + 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 1));
    m_yy_t45 = (2*m_xx_I + 8*m_xy_I + 2*m_xz_I + 8*m_yy_I - 8*m_yz_I + 2*m_zz_I - 2*omegaVar*m_xx_I + 2*omegaVar*m_xy_I - 7*omegaVar*m_xz_I + 7*omegaVar*m_yy_I - 2*omegaVar*m_yz_I - 2*omegaVar*m_zz_I + 2)/(9*(m_xx_I + 2*m_xy_I + 2*m_xz_I + m_yy_I - 2*m_yz_I + m_zz_I - omegaVar*m_xx_I - 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - omegaVar*m_yy_I + 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 1));
    m_yz_t90 = -(2*m_xx_I + 8*m_xy_I + 8*m_xz_I + 8*m_yy_I - 26*m_yz_I + 8*m_zz_I - 7*omegaVar*m_xx_I - 8*omegaVar*m_xy_I - 8*omegaVar*m_xz_I + 2*omegaVar*m_yy_I - 19*omegaVar*m_yz_I + 2*omegaVar*m_zz_I + 4)/(18*(m_xx_I + 2*m_xy_I + 2*m_xz_I + m_yy_I - 2*m_yz_I + m_zz_I - omegaVar*m_xx_I - 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - omegaVar*m_yy_I + 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 1));
    m_zz_t45 = (2*m_xx_I + 2*m_xy_I + 8*m_xz_I + 2*m_yy_I - 8*m_yz_I + 8*m_zz_I - 2*omegaVar*m_xx_I - 7*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - 2*omegaVar*m_yy_I - 2*omegaVar*m_yz_I + 7*omegaVar*m_zz_I + 2)/(9*(m_xx_I + 2*m_xy_I + 2*m_xz_I + m_yy_I - 2*m_yz_I + m_zz_I - omegaVar*m_xx_I - 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - omegaVar*m_yy_I + 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 1));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = 0;
    L_BC_Fy = 0;
    L_BC_Fz = 0;
    #endif //_BC_FORCES 

    break;
case 0b0000010110111110:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[9] + pop[11] + pop[12] + pop[13] + pop[15];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[9] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[15]);
    m_xy_I = inv_rho_I * (pop[7] - pop[13]);
    m_xz_I = inv_rho_I * (pop[9] - pop[15]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] - (1.0/3.0)*pop[9] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] - (1.0/3.0)*pop[15]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] + (2.0/3.0)*pop[15]);

    rho = (18*(19*rho_I + 6*m_xx_I*rho_I + 12*m_yy_I*rho_I - 33*m_yz_I*rho_I + 12*m_zz_I*rho_I - 6*omegaVar*m_xx_I*rho_I - 12*omegaVar*m_yy_I*rho_I + 33*omegaVar*m_yz_I*rho_I - 12*omegaVar*m_zz_I*rho_I))/(59*omegaVar + 207);

    m_xx_t45 = (189*m_xx_I - 36*m_yy_I + 99*m_yz_I - 36*m_zz_I + 49*omegaVar*m_xx_I - 20*omegaVar*m_yy_I + 55*omegaVar*m_yz_I - 20*omegaVar*m_zz_I + 12)/(9*(6*m_xx_I + 12*m_yy_I - 33*m_yz_I + 12*m_zz_I - 6*omegaVar*m_xx_I - 12*omegaVar*m_yy_I + 33*omegaVar*m_yz_I - 12*omegaVar*m_zz_I + 19));
    m_xy_t90 = (m_xy_I*(59*omegaVar + 207))/(9*(6*m_xx_I + 12*m_yy_I - 33*m_yz_I + 12*m_zz_I - 6*omegaVar*m_xx_I - 12*omegaVar*m_yy_I + 33*omegaVar*m_yz_I - 12*omegaVar*m_zz_I + 19));
    m_xz_t90 = (m_xz_I*(59*omegaVar + 207))/(9*(6*m_xx_I + 12*m_yy_I - 33*m_yz_I + 12*m_zz_I - 6*omegaVar*m_xx_I - 12*omegaVar*m_yy_I + 33*omegaVar*m_yz_I - 12*omegaVar*m_zz_I + 19));
    m_yy_t45 = -(2*(54*m_xx_I - 306*m_yy_I + 324*m_yz_I - 99*m_zz_I + 30*omegaVar*m_xx_I - 58*omegaVar*m_yy_I + 12*omegaVar*m_yz_I + omegaVar*m_zz_I - 36))/(27*(6*m_xx_I + 12*m_yy_I - 33*m_yz_I + 12*m_zz_I - 6*omegaVar*m_xx_I - 12*omegaVar*m_yy_I + 33*omegaVar*m_yz_I - 12*omegaVar*m_zz_I + 19));
    m_yz_t90 = -(216*m_yy_I - 99*m_xx_I - 801*m_yz_I + 216*m_zz_I - 55*omegaVar*m_xx_I + 8*omegaVar*m_yy_I - 81*omegaVar*m_yz_I + 8*omegaVar*m_zz_I + 66)/(18*(6*m_xx_I + 12*m_yy_I - 33*m_yz_I + 12*m_zz_I - 6*omegaVar*m_xx_I - 12*omegaVar*m_yy_I + 33*omegaVar*m_yz_I - 12*omegaVar*m_zz_I + 19));
    m_zz_t45 = -(2*(54*m_xx_I - 99*m_yy_I + 324*m_yz_I - 306*m_zz_I + 30*omegaVar*m_xx_I + omegaVar*m_yy_I + 12*omegaVar*m_yz_I - 58*omegaVar*m_zz_I - 36))/(27*(6*m_xx_I + 12*m_yy_I - 33*m_yz_I + 12*m_zz_I - 6*omegaVar*m_xx_I - 12*omegaVar*m_yy_I + 33*omegaVar*m_yz_I - 12*omegaVar*m_zz_I + 19));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = (m_xx_t45 - (omegaVar*m_xx_t45)/2 + 1.0/3.0)*rho - RHO_0/3;
    L_BC_Fy = -(m_xy_t90*rho*(omegaVar - 2))/2;
    L_BC_Fz = -(m_xz_t90*rho*(omegaVar - 2))/2;
    #endif //_BC_FORCES 

    break;
case 0b0000010110111111:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[4] + pop[6] + pop[12] + pop[13] + pop[15];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[6] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[15]);
    m_xy_I = inv_rho_I * (-pop[13]);
    m_xz_I = inv_rho_I * (-pop[15]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] - (1.0/3.0)*pop[15]);
    m_yz_I = inv_rho_I * (pop[12]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[6] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] + (2.0/3.0)*pop[15]);

    rho = (12*(rho_I + m_xx_I*rho_I + 2*m_xy_I*rho_I + 2*m_xz_I*rho_I + m_yy_I*rho_I - 2*m_yz_I*rho_I + m_zz_I*rho_I - omegaVar*m_xx_I*rho_I - 2*omegaVar*m_xy_I*rho_I - 2*omegaVar*m_xz_I*rho_I - omegaVar*m_yy_I*rho_I + 2*omegaVar*m_yz_I*rho_I - omegaVar*m_zz_I*rho_I))/(5*omegaVar + 2);

    m_xx_t45 = (14*m_xx_I + 14*m_xy_I + 14*m_xz_I + 2*m_yy_I - 2*m_yz_I + 2*m_zz_I + 21*omegaVar*m_xx_I + 7*omegaVar*m_xy_I + 7*omegaVar*m_xz_I - 9*omegaVar*m_yy_I + 23*omegaVar*m_yz_I - 9*omegaVar*m_zz_I + 4)/(18*(m_xx_I + 2*m_xy_I + 2*m_xz_I + m_yy_I - 2*m_yz_I + m_zz_I - omegaVar*m_xx_I - 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - omegaVar*m_yy_I + 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 1));
    m_xy_t90 = (14*m_xx_I + 50*m_xy_I + 14*m_xz_I + 14*m_yy_I - 14*m_yz_I + 2*m_zz_I + 7*omegaVar*m_xx_I + 69*omegaVar*m_xy_I - 21*omegaVar*m_xz_I + 7*omegaVar*m_yy_I + 21*omegaVar*m_yz_I - 23*omegaVar*m_zz_I + 8)/(36*(m_xx_I + 2*m_xy_I + 2*m_xz_I + m_yy_I - 2*m_yz_I + m_zz_I - omegaVar*m_xx_I - 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - omegaVar*m_yy_I + 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 1));
    m_xz_t90 = (14*m_xx_I + 14*m_xy_I + 50*m_xz_I + 2*m_yy_I - 14*m_yz_I + 14*m_zz_I + 7*omegaVar*m_xx_I - 21*omegaVar*m_xy_I + 69*omegaVar*m_xz_I - 23*omegaVar*m_yy_I + 21*omegaVar*m_yz_I + 7*omegaVar*m_zz_I + 8)/(36*(m_xx_I + 2*m_xy_I + 2*m_xz_I + m_yy_I - 2*m_yz_I + m_zz_I - omegaVar*m_xx_I - 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - omegaVar*m_yy_I + 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 1));
    m_yy_t45 = (2*m_xx_I + 14*m_xy_I + 2*m_xz_I + 14*m_yy_I - 14*m_yz_I + 2*m_zz_I - 9*omegaVar*m_xx_I + 7*omegaVar*m_xy_I - 23*omegaVar*m_xz_I + 21*omegaVar*m_yy_I - 7*omegaVar*m_yz_I - 9*omegaVar*m_zz_I + 4)/(18*(m_xx_I + 2*m_xy_I + 2*m_xz_I + m_yy_I - 2*m_yz_I + m_zz_I - omegaVar*m_xx_I - 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - omegaVar*m_yy_I + 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 1));
    m_yz_t90 = -(2*m_xx_I + 14*m_xy_I + 14*m_xz_I + 14*m_yy_I - 50*m_yz_I + 14*m_zz_I - 23*omegaVar*m_xx_I - 21*omegaVar*m_xy_I - 21*omegaVar*m_xz_I + 7*omegaVar*m_yy_I - 69*omegaVar*m_yz_I + 7*omegaVar*m_zz_I + 8)/(36*(m_xx_I + 2*m_xy_I + 2*m_xz_I + m_yy_I - 2*m_yz_I + m_zz_I - omegaVar*m_xx_I - 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - omegaVar*m_yy_I + 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 1));
    m_zz_t45 = (2*m_xx_I + 2*m_xy_I + 14*m_xz_I + 2*m_yy_I - 14*m_yz_I + 14*m_zz_I - 9*omegaVar*m_xx_I - 23*omegaVar*m_xy_I + 7*omegaVar*m_xz_I - 9*omegaVar*m_yy_I - 7*omegaVar*m_yz_I + 21*omegaVar*m_zz_I + 4)/(18*(m_xx_I + 2*m_xy_I + 2*m_xz_I + m_yy_I - 2*m_yz_I + m_zz_I - omegaVar*m_xx_I - 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - omegaVar*m_yy_I + 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 1));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = ((2*m_xx_t45)/3 - m_xy_t90/2 - m_xz_t90/2 - m_yy_t45/12 - m_zz_t45/12 - (omegaVar*m_xx_t45)/3 + (omegaVar*m_xy_t90)/4 + (omegaVar*m_xz_t90)/4 + (omegaVar*m_yy_t45)/24 + (omegaVar*m_zz_t45)/24 + 2.0/9.0)*rho - (2*RHO_0)/9;
    L_BC_Fy = (m_xx_t45/12 + m_xy_t90/2 - (2*m_yy_t45)/3 - m_yz_t90/2 + m_zz_t45/12 - (omegaVar*m_xx_t45)/24 - (omegaVar*m_xy_t90)/4 + (omegaVar*m_yy_t45)/3 + (omegaVar*m_yz_t90)/4 - (omegaVar*m_zz_t45)/24 - 2.0/9.0)*rho + (2*RHO_0)/9;
    L_BC_Fz = (m_xx_t45/12 + m_xz_t90/2 + m_yy_t45/12 - m_yz_t90/2 - (2*m_zz_t45)/3 - (omegaVar*m_xx_t45)/24 - (omegaVar*m_xz_t90)/4 - (omegaVar*m_yy_t45)/24 + (omegaVar*m_yz_t90)/4 + (omegaVar*m_zz_t45)/3 - 2.0/9.0)*rho + (2*RHO_0)/9;
    #endif //_BC_FORCES 

    break;
case 0b0000010111000000:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (12*(15*rho_I - m_xx_I*rho_I + 2*m_yy_I*rho_I + 6*m_yz_I*rho_I + 2*m_zz_I*rho_I + omegaVar*m_xx_I*rho_I - 2*omegaVar*m_yy_I*rho_I - 6*omegaVar*m_yz_I*rho_I - 2*omegaVar*m_zz_I*rho_I))/(3*omegaVar + 172);

    m_xx_t45 = (792*m_xx_I - 36*m_yy_I - 108*m_yz_I - 36*m_zz_I + 13*omegaVar*m_xx_I + omegaVar*m_yy_I + 3*omegaVar*m_yz_I + omegaVar*m_zz_I - 12)/(54*(2*m_yy_I - m_xx_I + 6*m_yz_I + 2*m_zz_I + omegaVar*m_xx_I - 2*omegaVar*m_yy_I - 6*omegaVar*m_yz_I - 2*omegaVar*m_zz_I + 15));
    m_xy_t90 = (m_xy_I*(3*omegaVar + 172))/(12*(2*m_yy_I - m_xx_I + 6*m_yz_I + 2*m_zz_I + omegaVar*m_xx_I - 2*omegaVar*m_yy_I - 6*omegaVar*m_yz_I - 2*omegaVar*m_zz_I + 15));
    m_xz_t90 = (m_xz_I*(3*omegaVar + 172))/(12*(2*m_yy_I - m_xx_I + 6*m_yz_I + 2*m_zz_I + omegaVar*m_xx_I - 2*omegaVar*m_yy_I - 6*omegaVar*m_yz_I - 2*omegaVar*m_zz_I + 15));
    m_yy_t45 = (1692*m_yy_I - 72*m_xx_I + 432*m_yz_I + 144*m_zz_I + 2*omegaVar*m_xx_I + 23*omegaVar*m_yy_I - 12*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 48)/(108*(2*m_yy_I - m_xx_I + 6*m_yz_I + 2*m_zz_I + omegaVar*m_xx_I - 2*omegaVar*m_yy_I - 6*omegaVar*m_yz_I - 2*omegaVar*m_zz_I + 15));
    m_yz_t90 = (72*m_yy_I - 36*m_xx_I + 732*m_yz_I + 72*m_zz_I + omegaVar*m_xx_I - 2*omegaVar*m_yy_I + 3*omegaVar*m_yz_I - 2*omegaVar*m_zz_I + 24)/(36*(2*m_yy_I - m_xx_I + 6*m_yz_I + 2*m_zz_I + omegaVar*m_xx_I - 2*omegaVar*m_yy_I - 6*omegaVar*m_yz_I - 2*omegaVar*m_zz_I + 15));
    m_zz_t45 = (144*m_yy_I - 72*m_xx_I + 432*m_yz_I + 1692*m_zz_I + 2*omegaVar*m_xx_I - 4*omegaVar*m_yy_I - 12*omegaVar*m_yz_I + 23*omegaVar*m_zz_I + 48)/(108*(2*m_yy_I - m_xx_I + 6*m_yz_I + 2*m_zz_I + omegaVar*m_xx_I - 2*omegaVar*m_yy_I - 6*omegaVar*m_yz_I - 2*omegaVar*m_zz_I + 15));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = 0;
    L_BC_Fy = (m_yy_t45/6 - m_xx_t45/12 + m_yz_t90/2 + m_zz_t45/6 + (omegaVar*m_xx_t45)/24 - (omegaVar*m_yy_t45)/12 - (omegaVar*m_yz_t90)/4 - (omegaVar*m_zz_t45)/12 + 1.0/18.0)*rho - RHO_0/18;
    L_BC_Fz = (m_yy_t45/6 - m_xx_t45/12 + m_yz_t90/2 + m_zz_t45/6 + (omegaVar*m_xx_t45)/24 - (omegaVar*m_yy_t45)/12 - (omegaVar*m_yz_t90)/4 - (omegaVar*m_zz_t45)/12 + 1.0/18.0)*rho - RHO_0/18;
    #endif //_BC_FORCES 

    break;
case 0b0000010111000001:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (12*(15*rho_I - m_xx_I*rho_I + 2*m_yy_I*rho_I + 6*m_yz_I*rho_I + 2*m_zz_I*rho_I + omegaVar*m_xx_I*rho_I - 2*omegaVar*m_yy_I*rho_I - 6*omegaVar*m_yz_I*rho_I - 2*omegaVar*m_zz_I*rho_I))/(3*omegaVar + 172);

    m_xx_t45 = (792*m_xx_I - 36*m_yy_I - 108*m_yz_I - 36*m_zz_I + 13*omegaVar*m_xx_I + omegaVar*m_yy_I + 3*omegaVar*m_yz_I + omegaVar*m_zz_I - 12)/(54*(2*m_yy_I - m_xx_I + 6*m_yz_I + 2*m_zz_I + omegaVar*m_xx_I - 2*omegaVar*m_yy_I - 6*omegaVar*m_yz_I - 2*omegaVar*m_zz_I + 15));
    m_xy_t90 = (m_xy_I*(3*omegaVar + 172))/(12*(2*m_yy_I - m_xx_I + 6*m_yz_I + 2*m_zz_I + omegaVar*m_xx_I - 2*omegaVar*m_yy_I - 6*omegaVar*m_yz_I - 2*omegaVar*m_zz_I + 15));
    m_xz_t90 = (m_xz_I*(3*omegaVar + 172))/(12*(2*m_yy_I - m_xx_I + 6*m_yz_I + 2*m_zz_I + omegaVar*m_xx_I - 2*omegaVar*m_yy_I - 6*omegaVar*m_yz_I - 2*omegaVar*m_zz_I + 15));
    m_yy_t45 = (1692*m_yy_I - 72*m_xx_I + 432*m_yz_I + 144*m_zz_I + 2*omegaVar*m_xx_I + 23*omegaVar*m_yy_I - 12*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 48)/(108*(2*m_yy_I - m_xx_I + 6*m_yz_I + 2*m_zz_I + omegaVar*m_xx_I - 2*omegaVar*m_yy_I - 6*omegaVar*m_yz_I - 2*omegaVar*m_zz_I + 15));
    m_yz_t90 = (72*m_yy_I - 36*m_xx_I + 732*m_yz_I + 72*m_zz_I + omegaVar*m_xx_I - 2*omegaVar*m_yy_I + 3*omegaVar*m_yz_I - 2*omegaVar*m_zz_I + 24)/(36*(2*m_yy_I - m_xx_I + 6*m_yz_I + 2*m_zz_I + omegaVar*m_xx_I - 2*omegaVar*m_yy_I - 6*omegaVar*m_yz_I - 2*omegaVar*m_zz_I + 15));
    m_zz_t45 = (144*m_yy_I - 72*m_xx_I + 432*m_yz_I + 1692*m_zz_I + 2*omegaVar*m_xx_I - 4*omegaVar*m_yy_I - 12*omegaVar*m_yz_I + 23*omegaVar*m_zz_I + 48)/(108*(2*m_yy_I - m_xx_I + 6*m_yz_I + 2*m_zz_I + omegaVar*m_xx_I - 2*omegaVar*m_yy_I - 6*omegaVar*m_yz_I - 2*omegaVar*m_zz_I + 15));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = 0;
    L_BC_Fy = (m_yy_t45/6 - m_xx_t45/12 + m_yz_t90/2 + m_zz_t45/6 + (omegaVar*m_xx_t45)/24 - (omegaVar*m_yy_t45)/12 - (omegaVar*m_yz_t90)/4 - (omegaVar*m_zz_t45)/12 + 1.0/18.0)*rho - RHO_0/18;
    L_BC_Fz = (m_yy_t45/6 - m_xx_t45/12 + m_yz_t90/2 + m_zz_t45/6 + (omegaVar*m_xx_t45)/24 - (omegaVar*m_yy_t45)/12 - (omegaVar*m_yz_t90)/4 - (omegaVar*m_zz_t45)/12 + 1.0/18.0)*rho - RHO_0/18;
    #endif //_BC_FORCES 

    break;
case 0b0000010111000010:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (12*(15*rho_I - m_xx_I*rho_I + 2*m_yy_I*rho_I + 6*m_yz_I*rho_I + 2*m_zz_I*rho_I + omegaVar*m_xx_I*rho_I - 2*omegaVar*m_yy_I*rho_I - 6*omegaVar*m_yz_I*rho_I - 2*omegaVar*m_zz_I*rho_I))/(3*omegaVar + 172);

    m_xx_t45 = (792*m_xx_I - 36*m_yy_I - 108*m_yz_I - 36*m_zz_I + 13*omegaVar*m_xx_I + omegaVar*m_yy_I + 3*omegaVar*m_yz_I + omegaVar*m_zz_I - 12)/(54*(2*m_yy_I - m_xx_I + 6*m_yz_I + 2*m_zz_I + omegaVar*m_xx_I - 2*omegaVar*m_yy_I - 6*omegaVar*m_yz_I - 2*omegaVar*m_zz_I + 15));
    m_xy_t90 = (m_xy_I*(3*omegaVar + 172))/(12*(2*m_yy_I - m_xx_I + 6*m_yz_I + 2*m_zz_I + omegaVar*m_xx_I - 2*omegaVar*m_yy_I - 6*omegaVar*m_yz_I - 2*omegaVar*m_zz_I + 15));
    m_xz_t90 = (m_xz_I*(3*omegaVar + 172))/(12*(2*m_yy_I - m_xx_I + 6*m_yz_I + 2*m_zz_I + omegaVar*m_xx_I - 2*omegaVar*m_yy_I - 6*omegaVar*m_yz_I - 2*omegaVar*m_zz_I + 15));
    m_yy_t45 = (1692*m_yy_I - 72*m_xx_I + 432*m_yz_I + 144*m_zz_I + 2*omegaVar*m_xx_I + 23*omegaVar*m_yy_I - 12*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 48)/(108*(2*m_yy_I - m_xx_I + 6*m_yz_I + 2*m_zz_I + omegaVar*m_xx_I - 2*omegaVar*m_yy_I - 6*omegaVar*m_yz_I - 2*omegaVar*m_zz_I + 15));
    m_yz_t90 = (72*m_yy_I - 36*m_xx_I + 732*m_yz_I + 72*m_zz_I + omegaVar*m_xx_I - 2*omegaVar*m_yy_I + 3*omegaVar*m_yz_I - 2*omegaVar*m_zz_I + 24)/(36*(2*m_yy_I - m_xx_I + 6*m_yz_I + 2*m_zz_I + omegaVar*m_xx_I - 2*omegaVar*m_yy_I - 6*omegaVar*m_yz_I - 2*omegaVar*m_zz_I + 15));
    m_zz_t45 = (144*m_yy_I - 72*m_xx_I + 432*m_yz_I + 1692*m_zz_I + 2*omegaVar*m_xx_I - 4*omegaVar*m_yy_I - 12*omegaVar*m_yz_I + 23*omegaVar*m_zz_I + 48)/(108*(2*m_yy_I - m_xx_I + 6*m_yz_I + 2*m_zz_I + omegaVar*m_xx_I - 2*omegaVar*m_yy_I - 6*omegaVar*m_yz_I - 2*omegaVar*m_zz_I + 15));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = 0;
    L_BC_Fy = (m_yy_t45/6 - m_xx_t45/12 + m_yz_t90/2 + m_zz_t45/6 + (omegaVar*m_xx_t45)/24 - (omegaVar*m_yy_t45)/12 - (omegaVar*m_yz_t90)/4 - (omegaVar*m_zz_t45)/12 + 1.0/18.0)*rho - RHO_0/18;
    L_BC_Fz = (m_yy_t45/6 - m_xx_t45/12 + m_yz_t90/2 + m_zz_t45/6 + (omegaVar*m_xx_t45)/24 - (omegaVar*m_yy_t45)/12 - (omegaVar*m_yz_t90)/4 - (omegaVar*m_zz_t45)/12 + 1.0/18.0)*rho - RHO_0/18;
    #endif //_BC_FORCES 

    break;
case 0b0000010111000011:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (-pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (6*(3*rho_I - m_xx_I*rho_I + 2*m_yy_I*rho_I + 6*m_yz_I*rho_I + 2*m_zz_I*rho_I + omegaVar*m_xx_I*rho_I - 2*omegaVar*m_yy_I*rho_I - 6*omegaVar*m_yz_I*rho_I - 2*omegaVar*m_zz_I*rho_I))/(3*omegaVar + 14);

    m_xx_t45 = (72*m_xx_I - 18*m_yy_I - 54*m_yz_I - 18*m_zz_I + 13*omegaVar*m_xx_I + omegaVar*m_yy_I + 3*omegaVar*m_yz_I + omegaVar*m_zz_I - 6)/(27*(2*m_yy_I - m_xx_I + 6*m_yz_I + 2*m_zz_I + omegaVar*m_xx_I - 2*omegaVar*m_yy_I - 6*omegaVar*m_yz_I - 2*omegaVar*m_zz_I + 3));
    m_xy_t90 = (m_xy_I*(3*omegaVar + 14))/(6*(2*m_yy_I - m_xx_I + 6*m_yz_I + 2*m_zz_I + omegaVar*m_xx_I - 2*omegaVar*m_yy_I - 6*omegaVar*m_yz_I - 2*omegaVar*m_zz_I + 3));
    m_xz_t90 = (m_xz_I*(3*omegaVar + 14))/(6*(2*m_yy_I - m_xx_I + 6*m_yz_I + 2*m_zz_I + omegaVar*m_xx_I - 2*omegaVar*m_yy_I - 6*omegaVar*m_yz_I - 2*omegaVar*m_zz_I + 3));
    m_yy_t45 = (198*m_yy_I - 36*m_xx_I + 216*m_yz_I + 72*m_zz_I + 2*omegaVar*m_xx_I + 23*omegaVar*m_yy_I - 12*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 24)/(54*(2*m_yy_I - m_xx_I + 6*m_yz_I + 2*m_zz_I + omegaVar*m_xx_I - 2*omegaVar*m_yy_I - 6*omegaVar*m_yz_I - 2*omegaVar*m_zz_I + 3));
    m_yz_t90 = (36*m_yy_I - 18*m_xx_I + 150*m_yz_I + 36*m_zz_I + omegaVar*m_xx_I - 2*omegaVar*m_yy_I + 3*omegaVar*m_yz_I - 2*omegaVar*m_zz_I + 12)/(18*(2*m_yy_I - m_xx_I + 6*m_yz_I + 2*m_zz_I + omegaVar*m_xx_I - 2*omegaVar*m_yy_I - 6*omegaVar*m_yz_I - 2*omegaVar*m_zz_I + 3));
    m_zz_t45 = (72*m_yy_I - 36*m_xx_I + 216*m_yz_I + 198*m_zz_I + 2*omegaVar*m_xx_I - 4*omegaVar*m_yy_I - 12*omegaVar*m_yz_I + 23*omegaVar*m_zz_I + 24)/(54*(2*m_yy_I - m_xx_I + 6*m_yz_I + 2*m_zz_I + omegaVar*m_xx_I - 2*omegaVar*m_yy_I - 6*omegaVar*m_yz_I - 2*omegaVar*m_zz_I + 3));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = 0;
    L_BC_Fy = 0;
    L_BC_Fz = 0;
    #endif //_BC_FORCES 

    break;
case 0b0000010111000100:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (6*(15*rho_I + m_xx_I*rho_I - 6*m_xy_I*rho_I + 4*m_yy_I*rho_I + 6*m_yz_I*rho_I + m_zz_I*rho_I - omegaVar*m_xx_I*rho_I + 6*omegaVar*m_xy_I*rho_I - 4*omegaVar*m_yy_I*rho_I - 6*omegaVar*m_yz_I*rho_I - omegaVar*m_zz_I*rho_I))/(3*omegaVar + 82);

    m_xx_t45 = -(504*m_xy_I - 2052*m_xx_I - 90*m_yy_I + 234*m_yz_I + 162*m_zz_I - 73*omegaVar*m_xx_I + 6*omegaVar*m_xy_I + 5*omegaVar*m_yy_I + 21*omegaVar*m_yz_I + 8*omegaVar*m_zz_I - 30)/(135*(m_xx_I - 6*m_xy_I + 4*m_yy_I + 6*m_yz_I + m_zz_I - omegaVar*m_xx_I + 6*omegaVar*m_xy_I - 4*omegaVar*m_yy_I - 6*omegaVar*m_yz_I - omegaVar*m_zz_I + 15));
    m_xy_t90 = (1746*m_xy_I - 168*m_xx_I - 180*m_yy_I - 24*m_yz_I + 78*m_zz_I - 2*omegaVar*m_xx_I + 39*omegaVar*m_xy_I + 10*omegaVar*m_yy_I + 24*omegaVar*m_yz_I + 7*omegaVar*m_zz_I - 60)/(90*(m_xx_I - 6*m_xy_I + 4*m_yy_I + 6*m_yz_I + m_zz_I - omegaVar*m_xx_I + 6*omegaVar*m_xy_I - 4*omegaVar*m_yy_I - 6*omegaVar*m_yz_I - omegaVar*m_zz_I + 15));
    m_xz_t90 = (m_xz_I*(3*omegaVar + 82))/(6*(m_xx_I - 6*m_xy_I + 4*m_yy_I + 6*m_yz_I + m_zz_I - omegaVar*m_xx_I + 6*omegaVar*m_xy_I - 4*omegaVar*m_yy_I - 6*omegaVar*m_yz_I - omegaVar*m_zz_I + 15));
    m_yy_t45 = (36*m_xx_I - 216*m_xy_I + 882*m_yy_I + 216*m_yz_I + 36*m_zz_I - 2*omegaVar*m_xx_I + 12*omegaVar*m_xy_I + 19*omegaVar*m_yy_I - 12*omegaVar*m_yz_I - 2*omegaVar*m_zz_I + 48)/(54*(m_xx_I - 6*m_xy_I + 4*m_yy_I + 6*m_yz_I + m_zz_I - omegaVar*m_xx_I + 6*omegaVar*m_xy_I - 4*omegaVar*m_yy_I - 6*omegaVar*m_yz_I - omegaVar*m_zz_I + 15));
    m_yz_t90 = (180*m_yy_I - 24*m_xy_I - 78*m_xx_I + 1746*m_yz_I + 168*m_zz_I - 7*omegaVar*m_xx_I + 24*omegaVar*m_xy_I - 10*omegaVar*m_yy_I + 39*omegaVar*m_yz_I + 2*omegaVar*m_zz_I + 60)/(90*(m_xx_I - 6*m_xy_I + 4*m_yy_I + 6*m_yz_I + m_zz_I - omegaVar*m_xx_I + 6*omegaVar*m_xy_I - 4*omegaVar*m_yy_I - 6*omegaVar*m_yz_I - omegaVar*m_zz_I + 15));
    m_zz_t45 = (234*m_xy_I - 162*m_xx_I + 90*m_yy_I + 504*m_yz_I + 2052*m_zz_I - 8*omegaVar*m_xx_I + 21*omegaVar*m_xy_I - 5*omegaVar*m_yy_I + 6*omegaVar*m_yz_I + 73*omegaVar*m_zz_I + 30)/(135*(m_xx_I - 6*m_xy_I + 4*m_yy_I + 6*m_yz_I + m_zz_I - omegaVar*m_xx_I + 6*omegaVar*m_xy_I - 4*omegaVar*m_yy_I - 6*omegaVar*m_yz_I - omegaVar*m_zz_I + 15));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = (m_xy_t90/2 - m_xx_t45/6 - m_yy_t45/6 + m_zz_t45/12 + (omegaVar*m_xx_t45)/12 - (omegaVar*m_xy_t90)/4 + (omegaVar*m_yy_t45)/12 - (omegaVar*m_zz_t45)/24 - 1.0/18.0)*rho + RHO_0/18;
    L_BC_Fy = (m_xx_t45/12 - m_xy_t90/2 + m_yy_t45/3 + m_yz_t90/2 + m_zz_t45/12 - (omegaVar*m_xx_t45)/24 + (omegaVar*m_xy_t90)/4 - (omegaVar*m_yy_t45)/6 - (omegaVar*m_yz_t90)/4 - (omegaVar*m_zz_t45)/24 + 1.0/9.0)*rho - RHO_0/9;
    L_BC_Fz = (m_yy_t45/6 - m_xx_t45/12 + m_yz_t90/2 + m_zz_t45/6 + (omegaVar*m_xx_t45)/24 - (omegaVar*m_yy_t45)/12 - (omegaVar*m_yz_t90)/4 - (omegaVar*m_zz_t45)/12 + 1.0/18.0)*rho - RHO_0/18;
    #endif //_BC_FORCES 

    break;
case 0b0000010111000101:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[10] + pop[11] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[14]);
    m_xz_I = inv_rho_I * (pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (12*(5*rho_I + m_xx_I*rho_I - 2*m_xy_I*rho_I + 2*m_xz_I*rho_I + m_yy_I*rho_I + 2*m_yz_I*rho_I + m_zz_I*rho_I - omegaVar*m_xx_I*rho_I + 2*omegaVar*m_xy_I*rho_I - 2*omegaVar*m_xz_I*rho_I - omegaVar*m_yy_I*rho_I - 2*omegaVar*m_yz_I*rho_I - omegaVar*m_zz_I*rho_I))/(3*omegaVar + 52);

    m_xx_t45 = (472*m_xx_I - 112*m_xy_I + 112*m_xz_I + 4*m_yy_I - 44*m_yz_I + 4*m_zz_I + 23*omegaVar*m_xx_I + 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - 4*omegaVar*m_yy_I - 11*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 20)/(90*(m_xx_I - 2*m_xy_I + 2*m_xz_I + m_yy_I + 2*m_yz_I + m_zz_I - omegaVar*m_xx_I + 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - omegaVar*m_yy_I - 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_xy_t90 = (1108*m_xy_I - 112*m_xx_I - 16*m_xz_I - 112*m_yy_I - 16*m_yz_I + 44*m_zz_I + 2*omegaVar*m_xx_I + 47*omegaVar*m_xy_I + 16*omegaVar*m_xz_I + 2*omegaVar*m_yy_I + 16*omegaVar*m_yz_I + 11*omegaVar*m_zz_I - 40)/(180*(m_xx_I - 2*m_xy_I + 2*m_xz_I + m_yy_I + 2*m_yz_I + m_zz_I - omegaVar*m_xx_I + 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - omegaVar*m_yy_I - 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_xz_t90 = (112*m_xx_I - 16*m_xy_I + 1108*m_xz_I - 44*m_yy_I + 16*m_yz_I + 112*m_zz_I - 2*omegaVar*m_xx_I + 16*omegaVar*m_xy_I + 47*omegaVar*m_xz_I - 11*omegaVar*m_yy_I - 16*omegaVar*m_yz_I - 2*omegaVar*m_zz_I + 40)/(180*(m_xx_I - 2*m_xy_I + 2*m_xz_I + m_yy_I + 2*m_yz_I + m_zz_I - omegaVar*m_xx_I + 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - omegaVar*m_yy_I - 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_yy_t45 = (4*m_xx_I - 112*m_xy_I - 44*m_xz_I + 472*m_yy_I + 112*m_yz_I + 4*m_zz_I - 4*omegaVar*m_xx_I + 2*omegaVar*m_xy_I - 11*omegaVar*m_xz_I + 23*omegaVar*m_yy_I - 2*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 20)/(90*(m_xx_I - 2*m_xy_I + 2*m_xz_I + m_yy_I + 2*m_yz_I + m_zz_I - omegaVar*m_xx_I + 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - omegaVar*m_yy_I - 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_yz_t90 = (16*m_xz_I - 16*m_xy_I - 44*m_xx_I + 112*m_yy_I + 1108*m_yz_I + 112*m_zz_I - 11*omegaVar*m_xx_I + 16*omegaVar*m_xy_I - 16*omegaVar*m_xz_I - 2*omegaVar*m_yy_I + 47*omegaVar*m_yz_I - 2*omegaVar*m_zz_I + 40)/(180*(m_xx_I - 2*m_xy_I + 2*m_xz_I + m_yy_I + 2*m_yz_I + m_zz_I - omegaVar*m_xx_I + 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - omegaVar*m_yy_I - 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_zz_t45 = (4*m_xx_I + 44*m_xy_I + 112*m_xz_I + 4*m_yy_I + 112*m_yz_I + 472*m_zz_I - 4*omegaVar*m_xx_I + 11*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - 4*omegaVar*m_yy_I - 2*omegaVar*m_yz_I + 23*omegaVar*m_zz_I + 20)/(90*(m_xx_I - 2*m_xy_I + 2*m_xz_I + m_yy_I + 2*m_yz_I + m_zz_I - omegaVar*m_xx_I + 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - omegaVar*m_yy_I - 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = (m_xy_t90/2 - m_xx_t45/3 - m_xz_t90/2 - m_yy_t45/12 - m_zz_t45/12 + (omegaVar*m_xx_t45)/6 - (omegaVar*m_xy_t90)/4 + (omegaVar*m_xz_t90)/4 + (omegaVar*m_yy_t45)/24 + (omegaVar*m_zz_t45)/24 - 1.0/9.0)*rho + RHO_0/9;
    L_BC_Fy = (m_xx_t45/12 - m_xy_t90/2 + m_yy_t45/3 + m_yz_t90/2 + m_zz_t45/12 - (omegaVar*m_xx_t45)/24 + (omegaVar*m_xy_t90)/4 - (omegaVar*m_yy_t45)/6 - (omegaVar*m_yz_t90)/4 - (omegaVar*m_zz_t45)/24 + 1.0/9.0)*rho - RHO_0/9;
    L_BC_Fz = (rho*(omegaVar - 2)*(m_xx_t45 + 2*m_xz_t90 - m_yy_t45 - 2*m_yz_t90))/8;
    #endif //_BC_FORCES 

    break;
case 0b0000010111000110:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (6*(15*rho_I + m_xx_I*rho_I - 6*m_xy_I*rho_I + 4*m_yy_I*rho_I + 6*m_yz_I*rho_I + m_zz_I*rho_I - omegaVar*m_xx_I*rho_I + 6*omegaVar*m_xy_I*rho_I - 4*omegaVar*m_yy_I*rho_I - 6*omegaVar*m_yz_I*rho_I - omegaVar*m_zz_I*rho_I))/(3*omegaVar + 82);

    m_xx_t45 = -(504*m_xy_I - 2052*m_xx_I - 90*m_yy_I + 234*m_yz_I + 162*m_zz_I - 73*omegaVar*m_xx_I + 6*omegaVar*m_xy_I + 5*omegaVar*m_yy_I + 21*omegaVar*m_yz_I + 8*omegaVar*m_zz_I - 30)/(135*(m_xx_I - 6*m_xy_I + 4*m_yy_I + 6*m_yz_I + m_zz_I - omegaVar*m_xx_I + 6*omegaVar*m_xy_I - 4*omegaVar*m_yy_I - 6*omegaVar*m_yz_I - omegaVar*m_zz_I + 15));
    m_xy_t90 = (1746*m_xy_I - 168*m_xx_I - 180*m_yy_I - 24*m_yz_I + 78*m_zz_I - 2*omegaVar*m_xx_I + 39*omegaVar*m_xy_I + 10*omegaVar*m_yy_I + 24*omegaVar*m_yz_I + 7*omegaVar*m_zz_I - 60)/(90*(m_xx_I - 6*m_xy_I + 4*m_yy_I + 6*m_yz_I + m_zz_I - omegaVar*m_xx_I + 6*omegaVar*m_xy_I - 4*omegaVar*m_yy_I - 6*omegaVar*m_yz_I - omegaVar*m_zz_I + 15));
    m_xz_t90 = (m_xz_I*(3*omegaVar + 82))/(6*(m_xx_I - 6*m_xy_I + 4*m_yy_I + 6*m_yz_I + m_zz_I - omegaVar*m_xx_I + 6*omegaVar*m_xy_I - 4*omegaVar*m_yy_I - 6*omegaVar*m_yz_I - omegaVar*m_zz_I + 15));
    m_yy_t45 = (36*m_xx_I - 216*m_xy_I + 882*m_yy_I + 216*m_yz_I + 36*m_zz_I - 2*omegaVar*m_xx_I + 12*omegaVar*m_xy_I + 19*omegaVar*m_yy_I - 12*omegaVar*m_yz_I - 2*omegaVar*m_zz_I + 48)/(54*(m_xx_I - 6*m_xy_I + 4*m_yy_I + 6*m_yz_I + m_zz_I - omegaVar*m_xx_I + 6*omegaVar*m_xy_I - 4*omegaVar*m_yy_I - 6*omegaVar*m_yz_I - omegaVar*m_zz_I + 15));
    m_yz_t90 = (180*m_yy_I - 24*m_xy_I - 78*m_xx_I + 1746*m_yz_I + 168*m_zz_I - 7*omegaVar*m_xx_I + 24*omegaVar*m_xy_I - 10*omegaVar*m_yy_I + 39*omegaVar*m_yz_I + 2*omegaVar*m_zz_I + 60)/(90*(m_xx_I - 6*m_xy_I + 4*m_yy_I + 6*m_yz_I + m_zz_I - omegaVar*m_xx_I + 6*omegaVar*m_xy_I - 4*omegaVar*m_yy_I - 6*omegaVar*m_yz_I - omegaVar*m_zz_I + 15));
    m_zz_t45 = (234*m_xy_I - 162*m_xx_I + 90*m_yy_I + 504*m_yz_I + 2052*m_zz_I - 8*omegaVar*m_xx_I + 21*omegaVar*m_xy_I - 5*omegaVar*m_yy_I + 6*omegaVar*m_yz_I + 73*omegaVar*m_zz_I + 30)/(135*(m_xx_I - 6*m_xy_I + 4*m_yy_I + 6*m_yz_I + m_zz_I - omegaVar*m_xx_I + 6*omegaVar*m_xy_I - 4*omegaVar*m_yy_I - 6*omegaVar*m_yz_I - omegaVar*m_zz_I + 15));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = (m_xy_t90/2 - m_xx_t45/6 - m_yy_t45/6 + m_zz_t45/12 + (omegaVar*m_xx_t45)/12 - (omegaVar*m_xy_t90)/4 + (omegaVar*m_yy_t45)/12 - (omegaVar*m_zz_t45)/24 - 1.0/18.0)*rho + RHO_0/18;
    L_BC_Fy = (m_xx_t45/12 - m_xy_t90/2 + m_yy_t45/3 + m_yz_t90/2 + m_zz_t45/12 - (omegaVar*m_xx_t45)/24 + (omegaVar*m_xy_t90)/4 - (omegaVar*m_yy_t45)/6 - (omegaVar*m_yz_t90)/4 - (omegaVar*m_zz_t45)/24 + 1.0/9.0)*rho - RHO_0/9;
    L_BC_Fz = (m_yy_t45/6 - m_xx_t45/12 + m_yz_t90/2 + m_zz_t45/6 + (omegaVar*m_xx_t45)/24 - (omegaVar*m_yy_t45)/12 - (omegaVar*m_yz_t90)/4 - (omegaVar*m_zz_t45)/12 + 1.0/18.0)*rho - RHO_0/18;
    #endif //_BC_FORCES 

    break;
case 0b0000010111000111:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[10] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[14]);
    m_xz_I = inv_rho_I * (pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (-pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (3*(15*rho_I - m_xx_I*rho_I - 6*m_xy_I*rho_I + 6*m_xz_I*rho_I + 11*m_yy_I*rho_I + 30*m_yz_I*rho_I + 11*m_zz_I*rho_I + omegaVar*m_xx_I*rho_I + 6*omegaVar*m_xy_I*rho_I - 6*omegaVar*m_xz_I*rho_I - 11*omegaVar*m_yy_I*rho_I - 30*omegaVar*m_yz_I*rho_I - 11*omegaVar*m_zz_I*rho_I))/(9*omegaVar + 31);

    m_xx_t45 = -(2*(36*m_xy_I - 180*m_xx_I - 36*m_xz_I + 27*m_yy_I + 99*m_yz_I + 27*m_zz_I - 52*omegaVar*m_xx_I + 12*omegaVar*m_xy_I - 12*omegaVar*m_xz_I + 5*omegaVar*m_yy_I + 21*omegaVar*m_yz_I + 5*omegaVar*m_zz_I + 3))/(27*(6*m_xz_I - 6*m_xy_I - m_xx_I + 11*m_yy_I + 30*m_yz_I + 11*m_zz_I + omegaVar*m_xx_I + 6*omegaVar*m_xy_I - 6*omegaVar*m_xz_I - 11*omegaVar*m_yy_I - 30*omegaVar*m_yz_I - 11*omegaVar*m_zz_I + 15));
    m_xy_t90 = (663*m_xy_I - 60*m_xx_I - 12*m_xz_I - 84*m_yy_I - 60*m_yz_I + 9*m_zz_I - 20*omegaVar*m_xx_I + 177*omegaVar*m_xy_I + 12*omegaVar*m_xz_I + 4*omegaVar*m_yy_I + 60*omegaVar*m_yz_I + 31*omegaVar*m_zz_I - 30)/(45*(6*m_xz_I - 6*m_xy_I - m_xx_I + 11*m_yy_I + 30*m_yz_I + 11*m_zz_I + omegaVar*m_xx_I + 6*omegaVar*m_xy_I - 6*omegaVar*m_xz_I - 11*omegaVar*m_yy_I - 30*omegaVar*m_yz_I - 11*omegaVar*m_zz_I + 15));
    m_xz_t90 = (60*m_xx_I - 12*m_xy_I + 663*m_xz_I - 9*m_yy_I + 60*m_yz_I + 84*m_zz_I + 20*omegaVar*m_xx_I + 12*omegaVar*m_xy_I + 177*omegaVar*m_xz_I - 31*omegaVar*m_yy_I - 60*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 30)/(45*(6*m_xz_I - 6*m_xy_I - m_xx_I + 11*m_yy_I + 30*m_yz_I + 11*m_zz_I + omegaVar*m_xx_I + 6*omegaVar*m_xy_I - 6*omegaVar*m_xz_I - 11*omegaVar*m_yy_I - 30*omegaVar*m_yz_I - 11*omegaVar*m_zz_I + 15));
    m_yy_t45 = -(2*(135*m_xx_I + 252*m_xy_I + 27*m_xz_I - 1206*m_yy_I - 1260*m_yz_I - 369*m_zz_I + 25*omegaVar*m_xx_I - 12*omegaVar*m_xy_I + 93*omegaVar*m_xz_I - 194*omegaVar*m_yy_I + 60*omegaVar*m_yz_I + 49*omegaVar*m_zz_I - 165))/(135*(6*m_xz_I - 6*m_xy_I - m_xx_I + 11*m_yy_I + 30*m_yz_I + 11*m_zz_I + omegaVar*m_xx_I + 6*omegaVar*m_xy_I - 6*omegaVar*m_xz_I - 11*omegaVar*m_yy_I - 30*omegaVar*m_yz_I - 11*omegaVar*m_zz_I + 15));
    m_yz_t90 = (12*m_xz_I - 12*m_xy_I - 33*m_xx_I + 84*m_yy_I + 339*m_yz_I + 84*m_zz_I - 7*omegaVar*m_xx_I + 12*omegaVar*m_xy_I - 12*omegaVar*m_xz_I - 4*omegaVar*m_yy_I + 21*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 30)/(9*(6*m_xz_I - 6*m_xy_I - m_xx_I + 11*m_yy_I + 30*m_yz_I + 11*m_zz_I + omegaVar*m_xx_I + 6*omegaVar*m_xy_I - 6*omegaVar*m_xz_I - 11*omegaVar*m_yy_I - 30*omegaVar*m_yz_I - 11*omegaVar*m_zz_I + 15));
    m_zz_t45 = (2*(27*m_xy_I - 135*m_xx_I + 252*m_xz_I + 369*m_yy_I + 1260*m_yz_I + 1206*m_zz_I - 25*omegaVar*m_xx_I + 93*omegaVar*m_xy_I - 12*omegaVar*m_xz_I - 49*omegaVar*m_yy_I - 60*omegaVar*m_yz_I + 194*omegaVar*m_zz_I + 165))/(135*(6*m_xz_I - 6*m_xy_I - m_xx_I + 11*m_yy_I + 30*m_yz_I + 11*m_zz_I + omegaVar*m_xx_I + 6*omegaVar*m_xy_I - 6*omegaVar*m_xz_I - 11*omegaVar*m_yy_I - 30*omegaVar*m_yz_I - 11*omegaVar*m_zz_I + 15));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = (m_xy_t90/2 - m_xx_t45/3 - m_xz_t90/2 - m_yy_t45/12 - m_zz_t45/12 + (omegaVar*m_xx_t45)/6 - (omegaVar*m_xy_t90)/4 + (omegaVar*m_xz_t90)/4 + (omegaVar*m_yy_t45)/24 + (omegaVar*m_zz_t45)/24 - 1.0/9.0)*rho + RHO_0/9;
    L_BC_Fy = (m_xx_t45/6 - m_xy_t90/2 + m_yy_t45/6 - m_zz_t45/12 - (omegaVar*m_xx_t45)/12 + (omegaVar*m_xy_t90)/4 - (omegaVar*m_yy_t45)/12 + (omegaVar*m_zz_t45)/24 + 1.0/18.0)*rho - RHO_0/18;
    L_BC_Fz = (m_yy_t45/12 - m_xz_t90/2 - m_xx_t45/6 - m_zz_t45/6 + (omegaVar*m_xx_t45)/12 + (omegaVar*m_xz_t90)/4 - (omegaVar*m_yy_t45)/24 + (omegaVar*m_zz_t45)/12 - 1.0/18.0)*rho + RHO_0/18;
    #endif //_BC_FORCES 

    break;
case 0b0000010111001000:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[9] + pop[10] + pop[11] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (6*(15*rho_I + m_xx_I*rho_I + 6*m_xy_I*rho_I + 4*m_yy_I*rho_I + 6*m_yz_I*rho_I + m_zz_I*rho_I - omegaVar*m_xx_I*rho_I - 6*omegaVar*m_xy_I*rho_I - 4*omegaVar*m_yy_I*rho_I - 6*omegaVar*m_yz_I*rho_I - omegaVar*m_zz_I*rho_I))/(3*omegaVar + 82);

    m_xx_t45 = (2052*m_xx_I + 504*m_xy_I + 90*m_yy_I - 234*m_yz_I - 162*m_zz_I + 73*omegaVar*m_xx_I + 6*omegaVar*m_xy_I - 5*omegaVar*m_yy_I - 21*omegaVar*m_yz_I - 8*omegaVar*m_zz_I + 30)/(135*(m_xx_I + 6*m_xy_I + 4*m_yy_I + 6*m_yz_I + m_zz_I - omegaVar*m_xx_I - 6*omegaVar*m_xy_I - 4*omegaVar*m_yy_I - 6*omegaVar*m_yz_I - omegaVar*m_zz_I + 15));
    m_xy_t90 = (168*m_xx_I + 1746*m_xy_I + 180*m_yy_I + 24*m_yz_I - 78*m_zz_I + 2*omegaVar*m_xx_I + 39*omegaVar*m_xy_I - 10*omegaVar*m_yy_I - 24*omegaVar*m_yz_I - 7*omegaVar*m_zz_I + 60)/(90*(m_xx_I + 6*m_xy_I + 4*m_yy_I + 6*m_yz_I + m_zz_I - omegaVar*m_xx_I - 6*omegaVar*m_xy_I - 4*omegaVar*m_yy_I - 6*omegaVar*m_yz_I - omegaVar*m_zz_I + 15));
    m_xz_t90 = (m_xz_I*(3*omegaVar + 82))/(6*(m_xx_I + 6*m_xy_I + 4*m_yy_I + 6*m_yz_I + m_zz_I - omegaVar*m_xx_I - 6*omegaVar*m_xy_I - 4*omegaVar*m_yy_I - 6*omegaVar*m_yz_I - omegaVar*m_zz_I + 15));
    m_yy_t45 = (36*m_xx_I + 216*m_xy_I + 882*m_yy_I + 216*m_yz_I + 36*m_zz_I - 2*omegaVar*m_xx_I - 12*omegaVar*m_xy_I + 19*omegaVar*m_yy_I - 12*omegaVar*m_yz_I - 2*omegaVar*m_zz_I + 48)/(54*(m_xx_I + 6*m_xy_I + 4*m_yy_I + 6*m_yz_I + m_zz_I - omegaVar*m_xx_I - 6*omegaVar*m_xy_I - 4*omegaVar*m_yy_I - 6*omegaVar*m_yz_I - omegaVar*m_zz_I + 15));
    m_yz_t90 = (24*m_xy_I - 78*m_xx_I + 180*m_yy_I + 1746*m_yz_I + 168*m_zz_I - 7*omegaVar*m_xx_I - 24*omegaVar*m_xy_I - 10*omegaVar*m_yy_I + 39*omegaVar*m_yz_I + 2*omegaVar*m_zz_I + 60)/(90*(m_xx_I + 6*m_xy_I + 4*m_yy_I + 6*m_yz_I + m_zz_I - omegaVar*m_xx_I - 6*omegaVar*m_xy_I - 4*omegaVar*m_yy_I - 6*omegaVar*m_yz_I - omegaVar*m_zz_I + 15));
    m_zz_t45 = (90*m_yy_I - 234*m_xy_I - 162*m_xx_I + 504*m_yz_I + 2052*m_zz_I - 8*omegaVar*m_xx_I - 21*omegaVar*m_xy_I - 5*omegaVar*m_yy_I + 6*omegaVar*m_yz_I + 73*omegaVar*m_zz_I + 30)/(135*(m_xx_I + 6*m_xy_I + 4*m_yy_I + 6*m_yz_I + m_zz_I - omegaVar*m_xx_I - 6*omegaVar*m_xy_I - 4*omegaVar*m_yy_I - 6*omegaVar*m_yz_I - omegaVar*m_zz_I + 15));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = (m_xx_t45/6 + m_xy_t90/2 + m_yy_t45/6 - m_zz_t45/12 - (omegaVar*m_xx_t45)/12 - (omegaVar*m_xy_t90)/4 - (omegaVar*m_yy_t45)/12 + (omegaVar*m_zz_t45)/24 + 1.0/18.0)*rho - RHO_0/18;
    L_BC_Fy = (m_xx_t45/12 + m_xy_t90/2 + m_yy_t45/3 + m_yz_t90/2 + m_zz_t45/12 - (omegaVar*m_xx_t45)/24 - (omegaVar*m_xy_t90)/4 - (omegaVar*m_yy_t45)/6 - (omegaVar*m_yz_t90)/4 - (omegaVar*m_zz_t45)/24 + 1.0/9.0)*rho - RHO_0/9;
    L_BC_Fz = (m_yy_t45/6 - m_xx_t45/12 + m_yz_t90/2 + m_zz_t45/6 + (omegaVar*m_xx_t45)/24 - (omegaVar*m_yy_t45)/12 - (omegaVar*m_yz_t90)/4 - (omegaVar*m_zz_t45)/12 + 1.0/18.0)*rho - RHO_0/18;
    #endif //_BC_FORCES 

    break;
case 0b0000010111001001:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[9] + pop[10] + pop[11] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (6*(15*rho_I + m_xx_I*rho_I + 6*m_xy_I*rho_I + 4*m_yy_I*rho_I + 6*m_yz_I*rho_I + m_zz_I*rho_I - omegaVar*m_xx_I*rho_I - 6*omegaVar*m_xy_I*rho_I - 4*omegaVar*m_yy_I*rho_I - 6*omegaVar*m_yz_I*rho_I - omegaVar*m_zz_I*rho_I))/(3*omegaVar + 82);

    m_xx_t45 = (2052*m_xx_I + 504*m_xy_I + 90*m_yy_I - 234*m_yz_I - 162*m_zz_I + 73*omegaVar*m_xx_I + 6*omegaVar*m_xy_I - 5*omegaVar*m_yy_I - 21*omegaVar*m_yz_I - 8*omegaVar*m_zz_I + 30)/(135*(m_xx_I + 6*m_xy_I + 4*m_yy_I + 6*m_yz_I + m_zz_I - omegaVar*m_xx_I - 6*omegaVar*m_xy_I - 4*omegaVar*m_yy_I - 6*omegaVar*m_yz_I - omegaVar*m_zz_I + 15));
    m_xy_t90 = (168*m_xx_I + 1746*m_xy_I + 180*m_yy_I + 24*m_yz_I - 78*m_zz_I + 2*omegaVar*m_xx_I + 39*omegaVar*m_xy_I - 10*omegaVar*m_yy_I - 24*omegaVar*m_yz_I - 7*omegaVar*m_zz_I + 60)/(90*(m_xx_I + 6*m_xy_I + 4*m_yy_I + 6*m_yz_I + m_zz_I - omegaVar*m_xx_I - 6*omegaVar*m_xy_I - 4*omegaVar*m_yy_I - 6*omegaVar*m_yz_I - omegaVar*m_zz_I + 15));
    m_xz_t90 = (m_xz_I*(3*omegaVar + 82))/(6*(m_xx_I + 6*m_xy_I + 4*m_yy_I + 6*m_yz_I + m_zz_I - omegaVar*m_xx_I - 6*omegaVar*m_xy_I - 4*omegaVar*m_yy_I - 6*omegaVar*m_yz_I - omegaVar*m_zz_I + 15));
    m_yy_t45 = (36*m_xx_I + 216*m_xy_I + 882*m_yy_I + 216*m_yz_I + 36*m_zz_I - 2*omegaVar*m_xx_I - 12*omegaVar*m_xy_I + 19*omegaVar*m_yy_I - 12*omegaVar*m_yz_I - 2*omegaVar*m_zz_I + 48)/(54*(m_xx_I + 6*m_xy_I + 4*m_yy_I + 6*m_yz_I + m_zz_I - omegaVar*m_xx_I - 6*omegaVar*m_xy_I - 4*omegaVar*m_yy_I - 6*omegaVar*m_yz_I - omegaVar*m_zz_I + 15));
    m_yz_t90 = (24*m_xy_I - 78*m_xx_I + 180*m_yy_I + 1746*m_yz_I + 168*m_zz_I - 7*omegaVar*m_xx_I - 24*omegaVar*m_xy_I - 10*omegaVar*m_yy_I + 39*omegaVar*m_yz_I + 2*omegaVar*m_zz_I + 60)/(90*(m_xx_I + 6*m_xy_I + 4*m_yy_I + 6*m_yz_I + m_zz_I - omegaVar*m_xx_I - 6*omegaVar*m_xy_I - 4*omegaVar*m_yy_I - 6*omegaVar*m_yz_I - omegaVar*m_zz_I + 15));
    m_zz_t45 = (90*m_yy_I - 234*m_xy_I - 162*m_xx_I + 504*m_yz_I + 2052*m_zz_I - 8*omegaVar*m_xx_I - 21*omegaVar*m_xy_I - 5*omegaVar*m_yy_I + 6*omegaVar*m_yz_I + 73*omegaVar*m_zz_I + 30)/(135*(m_xx_I + 6*m_xy_I + 4*m_yy_I + 6*m_yz_I + m_zz_I - omegaVar*m_xx_I - 6*omegaVar*m_xy_I - 4*omegaVar*m_yy_I - 6*omegaVar*m_yz_I - omegaVar*m_zz_I + 15));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = (m_xx_t45/6 + m_xy_t90/2 + m_yy_t45/6 - m_zz_t45/12 - (omegaVar*m_xx_t45)/12 - (omegaVar*m_xy_t90)/4 - (omegaVar*m_yy_t45)/12 + (omegaVar*m_zz_t45)/24 + 1.0/18.0)*rho - RHO_0/18;
    L_BC_Fy = (m_xx_t45/12 + m_xy_t90/2 + m_yy_t45/3 + m_yz_t90/2 + m_zz_t45/12 - (omegaVar*m_xx_t45)/24 - (omegaVar*m_xy_t90)/4 - (omegaVar*m_yy_t45)/6 - (omegaVar*m_yz_t90)/4 - (omegaVar*m_zz_t45)/24 + 1.0/9.0)*rho - RHO_0/9;
    L_BC_Fz = (m_yy_t45/6 - m_xx_t45/12 + m_yz_t90/2 + m_zz_t45/6 + (omegaVar*m_xx_t45)/24 - (omegaVar*m_yy_t45)/12 - (omegaVar*m_yz_t90)/4 - (omegaVar*m_zz_t45)/12 + 1.0/18.0)*rho - RHO_0/18;
    #endif //_BC_FORCES 

    break;
case 0b0000010111001010:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[9] + pop[10] + pop[11] + pop[13] + pop[14] + pop[15] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (12*(5*rho_I + m_xx_I*rho_I + 2*m_xy_I*rho_I - 2*m_xz_I*rho_I + m_yy_I*rho_I + 2*m_yz_I*rho_I + m_zz_I*rho_I - omegaVar*m_xx_I*rho_I - 2*omegaVar*m_xy_I*rho_I + 2*omegaVar*m_xz_I*rho_I - omegaVar*m_yy_I*rho_I - 2*omegaVar*m_yz_I*rho_I - omegaVar*m_zz_I*rho_I))/(3*omegaVar + 52);

    m_xx_t45 = (472*m_xx_I + 112*m_xy_I - 112*m_xz_I + 4*m_yy_I - 44*m_yz_I + 4*m_zz_I + 23*omegaVar*m_xx_I - 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - 4*omegaVar*m_yy_I - 11*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 20)/(90*(m_xx_I + 2*m_xy_I - 2*m_xz_I + m_yy_I + 2*m_yz_I + m_zz_I - omegaVar*m_xx_I - 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - omegaVar*m_yy_I - 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_xy_t90 = (112*m_xx_I + 1108*m_xy_I - 16*m_xz_I + 112*m_yy_I + 16*m_yz_I - 44*m_zz_I - 2*omegaVar*m_xx_I + 47*omegaVar*m_xy_I + 16*omegaVar*m_xz_I - 2*omegaVar*m_yy_I - 16*omegaVar*m_yz_I - 11*omegaVar*m_zz_I + 40)/(180*(m_xx_I + 2*m_xy_I - 2*m_xz_I + m_yy_I + 2*m_yz_I + m_zz_I - omegaVar*m_xx_I - 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - omegaVar*m_yy_I - 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_xz_t90 = (1108*m_xz_I - 16*m_xy_I - 112*m_xx_I + 44*m_yy_I - 16*m_yz_I - 112*m_zz_I + 2*omegaVar*m_xx_I + 16*omegaVar*m_xy_I + 47*omegaVar*m_xz_I + 11*omegaVar*m_yy_I + 16*omegaVar*m_yz_I + 2*omegaVar*m_zz_I - 40)/(180*(m_xx_I + 2*m_xy_I - 2*m_xz_I + m_yy_I + 2*m_yz_I + m_zz_I - omegaVar*m_xx_I - 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - omegaVar*m_yy_I - 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_yy_t45 = (4*m_xx_I + 112*m_xy_I + 44*m_xz_I + 472*m_yy_I + 112*m_yz_I + 4*m_zz_I - 4*omegaVar*m_xx_I - 2*omegaVar*m_xy_I + 11*omegaVar*m_xz_I + 23*omegaVar*m_yy_I - 2*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 20)/(90*(m_xx_I + 2*m_xy_I - 2*m_xz_I + m_yy_I + 2*m_yz_I + m_zz_I - omegaVar*m_xx_I - 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - omegaVar*m_yy_I - 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_yz_t90 = (16*m_xy_I - 44*m_xx_I - 16*m_xz_I + 112*m_yy_I + 1108*m_yz_I + 112*m_zz_I - 11*omegaVar*m_xx_I - 16*omegaVar*m_xy_I + 16*omegaVar*m_xz_I - 2*omegaVar*m_yy_I + 47*omegaVar*m_yz_I - 2*omegaVar*m_zz_I + 40)/(180*(m_xx_I + 2*m_xy_I - 2*m_xz_I + m_yy_I + 2*m_yz_I + m_zz_I - omegaVar*m_xx_I - 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - omegaVar*m_yy_I - 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_zz_t45 = (4*m_xx_I - 44*m_xy_I - 112*m_xz_I + 4*m_yy_I + 112*m_yz_I + 472*m_zz_I - 4*omegaVar*m_xx_I - 11*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - 4*omegaVar*m_yy_I - 2*omegaVar*m_yz_I + 23*omegaVar*m_zz_I + 20)/(90*(m_xx_I + 2*m_xy_I - 2*m_xz_I + m_yy_I + 2*m_yz_I + m_zz_I - omegaVar*m_xx_I - 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - omegaVar*m_yy_I - 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = (m_xx_t45/3 + m_xy_t90/2 - m_xz_t90/2 + m_yy_t45/12 + m_zz_t45/12 - (omegaVar*m_xx_t45)/6 - (omegaVar*m_xy_t90)/4 + (omegaVar*m_xz_t90)/4 - (omegaVar*m_yy_t45)/24 - (omegaVar*m_zz_t45)/24 + 1.0/9.0)*rho - RHO_0/9;
    L_BC_Fy = (m_xx_t45/12 + m_xy_t90/2 + m_yy_t45/3 + m_yz_t90/2 + m_zz_t45/12 - (omegaVar*m_xx_t45)/24 - (omegaVar*m_xy_t90)/4 - (omegaVar*m_yy_t45)/6 - (omegaVar*m_yz_t90)/4 - (omegaVar*m_zz_t45)/24 + 1.0/9.0)*rho - RHO_0/9;
    L_BC_Fz = -(rho*(omegaVar - 2)*(2*m_xz_t90 - m_xx_t45 + m_yy_t45 + 2*m_yz_t90))/8;
    #endif //_BC_FORCES 

    break;
case 0b0000010111001011:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[9] + pop[10] + pop[13] + pop[14] + pop[15] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (-pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (3*(15*rho_I - m_xx_I*rho_I + 6*m_xy_I*rho_I - 6*m_xz_I*rho_I + 11*m_yy_I*rho_I + 30*m_yz_I*rho_I + 11*m_zz_I*rho_I + omegaVar*m_xx_I*rho_I - 6*omegaVar*m_xy_I*rho_I + 6*omegaVar*m_xz_I*rho_I - 11*omegaVar*m_yy_I*rho_I - 30*omegaVar*m_yz_I*rho_I - 11*omegaVar*m_zz_I*rho_I))/(9*omegaVar + 31);

    m_xx_t45 = -(2*(36*m_xz_I - 36*m_xy_I - 180*m_xx_I + 27*m_yy_I + 99*m_yz_I + 27*m_zz_I - 52*omegaVar*m_xx_I - 12*omegaVar*m_xy_I + 12*omegaVar*m_xz_I + 5*omegaVar*m_yy_I + 21*omegaVar*m_yz_I + 5*omegaVar*m_zz_I + 3))/(27*(6*m_xy_I - m_xx_I - 6*m_xz_I + 11*m_yy_I + 30*m_yz_I + 11*m_zz_I + omegaVar*m_xx_I - 6*omegaVar*m_xy_I + 6*omegaVar*m_xz_I - 11*omegaVar*m_yy_I - 30*omegaVar*m_yz_I - 11*omegaVar*m_zz_I + 15));
    m_xy_t90 = (60*m_xx_I + 663*m_xy_I - 12*m_xz_I + 84*m_yy_I + 60*m_yz_I - 9*m_zz_I + 20*omegaVar*m_xx_I + 177*omegaVar*m_xy_I + 12*omegaVar*m_xz_I - 4*omegaVar*m_yy_I - 60*omegaVar*m_yz_I - 31*omegaVar*m_zz_I + 30)/(45*(6*m_xy_I - m_xx_I - 6*m_xz_I + 11*m_yy_I + 30*m_yz_I + 11*m_zz_I + omegaVar*m_xx_I - 6*omegaVar*m_xy_I + 6*omegaVar*m_xz_I - 11*omegaVar*m_yy_I - 30*omegaVar*m_yz_I - 11*omegaVar*m_zz_I + 15));
    m_xz_t90 = (663*m_xz_I - 12*m_xy_I - 60*m_xx_I + 9*m_yy_I - 60*m_yz_I - 84*m_zz_I - 20*omegaVar*m_xx_I + 12*omegaVar*m_xy_I + 177*omegaVar*m_xz_I + 31*omegaVar*m_yy_I + 60*omegaVar*m_yz_I + 4*omegaVar*m_zz_I - 30)/(45*(6*m_xy_I - m_xx_I - 6*m_xz_I + 11*m_yy_I + 30*m_yz_I + 11*m_zz_I + omegaVar*m_xx_I - 6*omegaVar*m_xy_I + 6*omegaVar*m_xz_I - 11*omegaVar*m_yy_I - 30*omegaVar*m_yz_I - 11*omegaVar*m_zz_I + 15));
    m_yy_t45 = (2*(252*m_xy_I - 135*m_xx_I + 27*m_xz_I + 1206*m_yy_I + 1260*m_yz_I + 369*m_zz_I - 25*omegaVar*m_xx_I - 12*omegaVar*m_xy_I + 93*omegaVar*m_xz_I + 194*omegaVar*m_yy_I - 60*omegaVar*m_yz_I - 49*omegaVar*m_zz_I + 165))/(135*(6*m_xy_I - m_xx_I - 6*m_xz_I + 11*m_yy_I + 30*m_yz_I + 11*m_zz_I + omegaVar*m_xx_I - 6*omegaVar*m_xy_I + 6*omegaVar*m_xz_I - 11*omegaVar*m_yy_I - 30*omegaVar*m_yz_I - 11*omegaVar*m_zz_I + 15));
    m_yz_t90 = (12*m_xy_I - 33*m_xx_I - 12*m_xz_I + 84*m_yy_I + 339*m_yz_I + 84*m_zz_I - 7*omegaVar*m_xx_I - 12*omegaVar*m_xy_I + 12*omegaVar*m_xz_I - 4*omegaVar*m_yy_I + 21*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 30)/(9*(6*m_xy_I - m_xx_I - 6*m_xz_I + 11*m_yy_I + 30*m_yz_I + 11*m_zz_I + omegaVar*m_xx_I - 6*omegaVar*m_xy_I + 6*omegaVar*m_xz_I - 11*omegaVar*m_yy_I - 30*omegaVar*m_yz_I - 11*omegaVar*m_zz_I + 15));
    m_zz_t45 = -(2*(135*m_xx_I + 27*m_xy_I + 252*m_xz_I - 369*m_yy_I - 1260*m_yz_I - 1206*m_zz_I + 25*omegaVar*m_xx_I + 93*omegaVar*m_xy_I - 12*omegaVar*m_xz_I + 49*omegaVar*m_yy_I + 60*omegaVar*m_yz_I - 194*omegaVar*m_zz_I - 165))/(135*(6*m_xy_I - m_xx_I - 6*m_xz_I + 11*m_yy_I + 30*m_yz_I + 11*m_zz_I + omegaVar*m_xx_I - 6*omegaVar*m_xy_I + 6*omegaVar*m_xz_I - 11*omegaVar*m_yy_I - 30*omegaVar*m_yz_I - 11*omegaVar*m_zz_I + 15));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = (m_xx_t45/3 + m_xy_t90/2 - m_xz_t90/2 + m_yy_t45/12 + m_zz_t45/12 - (omegaVar*m_xx_t45)/6 - (omegaVar*m_xy_t90)/4 + (omegaVar*m_xz_t90)/4 - (omegaVar*m_yy_t45)/24 - (omegaVar*m_zz_t45)/24 + 1.0/9.0)*rho - RHO_0/9;
    L_BC_Fy = (m_xx_t45/6 + m_xy_t90/2 + m_yy_t45/6 - m_zz_t45/12 - (omegaVar*m_xx_t45)/12 - (omegaVar*m_xy_t90)/4 - (omegaVar*m_yy_t45)/12 + (omegaVar*m_zz_t45)/24 + 1.0/18.0)*rho - RHO_0/18;
    L_BC_Fz = (m_xz_t90/2 - m_xx_t45/6 + m_yy_t45/12 - m_zz_t45/6 + (omegaVar*m_xx_t45)/12 - (omegaVar*m_xz_t90)/4 - (omegaVar*m_yy_t45)/24 + (omegaVar*m_zz_t45)/12 - 1.0/18.0)*rho + RHO_0/18;
    #endif //_BC_FORCES 

    break;
case 0b0000010111001100:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[5] + pop[6] + pop[7] + pop[9] + pop[10] + pop[11] + pop[14] + pop[15] + pop[16] + pop[17];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17]);
    m_xy_I = inv_rho_I * (pop[7] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17]);
    m_yz_I = inv_rho_I * (pop[11] - pop[17]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17]);

    rho = (3*rho_I*(3*m_yy_I - 3*omegaVar*m_yy_I + 4))/(omegaVar + 9);

    m_xx_t45 = (4*(omegaVar + 9)*(10*m_xx_I - m_zz_I))/(99*(3*m_yy_I - 3*omegaVar*m_yy_I + 4));
    m_xy_t90 = (2*m_xy_I*(omegaVar + 9))/(3*(3*m_yy_I - 3*omegaVar*m_yy_I + 4));
    m_xz_t90 = (m_xz_I*(omegaVar + 9))/(3*(3*m_yy_I - 3*omegaVar*m_yy_I + 4));
    m_yy_t45 = (15*m_yy_I + 2)/(3*(3*m_yy_I - 3*omegaVar*m_yy_I + 4));
    m_yz_t90 = (2*m_yz_I*(omegaVar + 9))/(3*(3*m_yy_I - 3*omegaVar*m_yy_I + 4));
    m_zz_t45 = -(4*(m_xx_I - 10*m_zz_I)*(omegaVar + 9))/(99*(3*m_yy_I - 3*omegaVar*m_yy_I + 4));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = -(m_xy_t90*rho*(omegaVar - 2))/2;
    L_BC_Fy = (m_yy_t45 - (omegaVar*m_yy_t45)/2 + 1.0/3.0)*rho - RHO_0/3;
    L_BC_Fz = -(m_yz_t90*rho*(omegaVar - 2))/2;
    #endif //_BC_FORCES 

    break;
case 0b0000010111001101:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[5] + pop[6] + pop[7] + pop[10] + pop[11] + pop[14] + pop[15] + pop[16] + pop[17];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17]);
    m_xy_I = inv_rho_I * (pop[7] - pop[14]);
    m_xz_I = inv_rho_I * (pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17]);
    m_yz_I = inv_rho_I * (pop[11] - pop[17]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17]);

    rho = (36*(107*rho_I + 12*m_xx_I*rho_I + 33*m_xz_I*rho_I + 72*m_yy_I*rho_I + 12*m_zz_I*rho_I - 12*omegaVar*m_xx_I*rho_I - 33*omegaVar*m_xz_I*rho_I - 72*omegaVar*m_yy_I*rho_I - 12*omegaVar*m_zz_I*rho_I))/(313*omegaVar + 2790);

    m_xx_t45 = (2772*m_xx_I + 648*m_xz_I - 108*m_yy_I - 18*m_zz_I + 302*omegaVar*m_xx_I + 48*omegaVar*m_xz_I - 66*omegaVar*m_yy_I - 11*omegaVar*m_zz_I + 72)/(27*(12*m_xx_I + 33*m_xz_I + 72*m_yy_I + 12*m_zz_I - 12*omegaVar*m_xx_I - 33*omegaVar*m_xz_I - 72*omegaVar*m_yy_I - 12*omegaVar*m_zz_I + 107));
    m_xy_t90 = (m_xy_I*(313*omegaVar + 2790))/(18*(12*m_xx_I + 33*m_xz_I + 72*m_yy_I + 12*m_zz_I - 12*omegaVar*m_xx_I - 33*omegaVar*m_xz_I - 72*omegaVar*m_yy_I - 12*omegaVar*m_zz_I + 107));
    m_xz_t90 = (432*m_xx_I + 3978*m_xz_I - 198*m_yy_I + 432*m_zz_I + 32*omegaVar*m_xx_I + 401*omegaVar*m_xz_I - 121*omegaVar*m_yy_I + 32*omegaVar*m_zz_I + 132)/(36*(12*m_xx_I + 33*m_xz_I + 72*m_yy_I + 12*m_zz_I - 12*omegaVar*m_xx_I - 33*omegaVar*m_xz_I - 72*omegaVar*m_yy_I - 12*omegaVar*m_zz_I + 107));
    m_yy_t45 = -(72*m_xx_I + 198*m_xz_I - 2358*m_yy_I + 72*m_zz_I + 44*omegaVar*m_xx_I + 121*omegaVar*m_xz_I - 49*omegaVar*m_yy_I + 44*omegaVar*m_zz_I - 288)/(18*(12*m_xx_I + 33*m_xz_I + 72*m_yy_I + 12*m_zz_I - 12*omegaVar*m_xx_I - 33*omegaVar*m_xz_I - 72*omegaVar*m_yy_I - 12*omegaVar*m_zz_I + 107));
    m_yz_t90 = (m_yz_I*(313*omegaVar + 2790))/(18*(12*m_xx_I + 33*m_xz_I + 72*m_yy_I + 12*m_zz_I - 12*omegaVar*m_xx_I - 33*omegaVar*m_xz_I - 72*omegaVar*m_yy_I - 12*omegaVar*m_zz_I + 107));
    m_zz_t45 = (648*m_xz_I - 18*m_xx_I - 108*m_yy_I + 2772*m_zz_I - 11*omegaVar*m_xx_I + 48*omegaVar*m_xz_I - 66*omegaVar*m_yy_I + 302*omegaVar*m_zz_I + 72)/(27*(12*m_xx_I + 33*m_xz_I + 72*m_yy_I + 12*m_zz_I - 12*omegaVar*m_xx_I - 33*omegaVar*m_xz_I - 72*omegaVar*m_yy_I - 12*omegaVar*m_zz_I + 107));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = (m_xy_t90 - m_xx_t45/6 - m_xz_t90/2 + m_yy_t45/12 - m_zz_t45/6 + (omegaVar*m_xx_t45)/12 - (omegaVar*m_xy_t90)/2 + (omegaVar*m_xz_t90)/4 - (omegaVar*m_yy_t45)/24 + (omegaVar*m_zz_t45)/12 - 1.0/18.0)*rho + RHO_0/18;
    L_BC_Fy = (m_yy_t45 - (omegaVar*m_yy_t45)/2 + 1.0/3.0)*rho - RHO_0/3;
    L_BC_Fz = (m_yy_t45/12 - m_xz_t90/2 - m_xx_t45/6 + m_yz_t90 - m_zz_t45/6 + (omegaVar*m_xx_t45)/12 + (omegaVar*m_xz_t90)/4 - (omegaVar*m_yy_t45)/24 - (omegaVar*m_yz_t90)/2 + (omegaVar*m_zz_t45)/12 - 1.0/18.0)*rho + RHO_0/18;
    #endif //_BC_FORCES 

    break;
case 0b0000010111001110:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[5] + pop[6] + pop[7] + pop[9] + pop[10] + pop[11] + pop[14] + pop[15] + pop[17];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] - (1.0/3.0)*pop[17]);
    m_xy_I = inv_rho_I * (pop[7] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] + (2.0/3.0)*pop[17]);
    m_yz_I = inv_rho_I * (pop[11] - pop[17]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[17]);

    rho = (36*(107*rho_I + 12*m_xx_I*rho_I - 33*m_xz_I*rho_I + 72*m_yy_I*rho_I + 12*m_zz_I*rho_I - 12*omegaVar*m_xx_I*rho_I + 33*omegaVar*m_xz_I*rho_I - 72*omegaVar*m_yy_I*rho_I - 12*omegaVar*m_zz_I*rho_I))/(313*omegaVar + 2790);

    m_xx_t45 = -(648*m_xz_I - 2772*m_xx_I + 108*m_yy_I + 18*m_zz_I - 302*omegaVar*m_xx_I + 48*omegaVar*m_xz_I + 66*omegaVar*m_yy_I + 11*omegaVar*m_zz_I - 72)/(27*(12*m_xx_I - 33*m_xz_I + 72*m_yy_I + 12*m_zz_I - 12*omegaVar*m_xx_I + 33*omegaVar*m_xz_I - 72*omegaVar*m_yy_I - 12*omegaVar*m_zz_I + 107));
    m_xy_t90 = (m_xy_I*(313*omegaVar + 2790))/(18*(12*m_xx_I - 33*m_xz_I + 72*m_yy_I + 12*m_zz_I - 12*omegaVar*m_xx_I + 33*omegaVar*m_xz_I - 72*omegaVar*m_yy_I - 12*omegaVar*m_zz_I + 107));
    m_xz_t90 = -(432*m_xx_I - 3978*m_xz_I - 198*m_yy_I + 432*m_zz_I + 32*omegaVar*m_xx_I - 401*omegaVar*m_xz_I - 121*omegaVar*m_yy_I + 32*omegaVar*m_zz_I + 132)/(36*(12*m_xx_I - 33*m_xz_I + 72*m_yy_I + 12*m_zz_I - 12*omegaVar*m_xx_I + 33*omegaVar*m_xz_I - 72*omegaVar*m_yy_I - 12*omegaVar*m_zz_I + 107));
    m_yy_t45 = (198*m_xz_I - 72*m_xx_I + 2358*m_yy_I - 72*m_zz_I - 44*omegaVar*m_xx_I + 121*omegaVar*m_xz_I + 49*omegaVar*m_yy_I - 44*omegaVar*m_zz_I + 288)/(18*(12*m_xx_I - 33*m_xz_I + 72*m_yy_I + 12*m_zz_I - 12*omegaVar*m_xx_I + 33*omegaVar*m_xz_I - 72*omegaVar*m_yy_I - 12*omegaVar*m_zz_I + 107));
    m_yz_t90 = (m_yz_I*(313*omegaVar + 2790))/(18*(12*m_xx_I - 33*m_xz_I + 72*m_yy_I + 12*m_zz_I - 12*omegaVar*m_xx_I + 33*omegaVar*m_xz_I - 72*omegaVar*m_yy_I - 12*omegaVar*m_zz_I + 107));
    m_zz_t45 = -(18*m_xx_I + 648*m_xz_I + 108*m_yy_I - 2772*m_zz_I + 11*omegaVar*m_xx_I + 48*omegaVar*m_xz_I + 66*omegaVar*m_yy_I - 302*omegaVar*m_zz_I - 72)/(27*(12*m_xx_I - 33*m_xz_I + 72*m_yy_I + 12*m_zz_I - 12*omegaVar*m_xx_I + 33*omegaVar*m_xz_I - 72*omegaVar*m_yy_I - 12*omegaVar*m_zz_I + 107));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = (m_xx_t45/6 + m_xy_t90 - m_xz_t90/2 - m_yy_t45/12 + m_zz_t45/6 - (omegaVar*m_xx_t45)/12 - (omegaVar*m_xy_t90)/2 + (omegaVar*m_xz_t90)/4 + (omegaVar*m_yy_t45)/24 - (omegaVar*m_zz_t45)/12 + 1.0/18.0)*rho - RHO_0/18;
    L_BC_Fy = (m_yy_t45 - (omegaVar*m_yy_t45)/2 + 1.0/3.0)*rho - RHO_0/3;
    L_BC_Fz = (m_xz_t90/2 - m_xx_t45/6 + m_yy_t45/12 + m_yz_t90 - m_zz_t45/6 + (omegaVar*m_xx_t45)/12 - (omegaVar*m_xz_t90)/4 - (omegaVar*m_yy_t45)/24 - (omegaVar*m_yz_t90)/2 + (omegaVar*m_zz_t45)/12 - 1.0/18.0)*rho + RHO_0/18;
    #endif //_BC_FORCES 

    break;
case 0b0000010111001111:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[6] + pop[7] + pop[10] + pop[14] + pop[15] + pop[17];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] - (1.0/3.0)*pop[17]);
    m_xy_I = inv_rho_I * (pop[7] - pop[14]);
    m_xz_I = inv_rho_I * (pop[10] - pop[15]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] + (2.0/3.0)*pop[17]);
    m_yz_I = inv_rho_I * (-pop[17]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[17]);

    rho = (36*(23*rho_I - 6*m_xx_I*rho_I + 24*m_yy_I*rho_I + 57*m_yz_I*rho_I + 24*m_zz_I*rho_I + 6*omegaVar*m_xx_I*rho_I - 24*omegaVar*m_yy_I*rho_I - 57*omegaVar*m_yz_I*rho_I - 24*omegaVar*m_zz_I*rho_I))/(5*(43*omegaVar + 72));

    m_xx_t45 = -(72*m_yy_I - 288*m_xx_I + 216*m_yz_I + 72*m_zz_I - 162*omegaVar*m_xx_I + 3*omegaVar*m_yy_I + 34*omegaVar*m_yz_I + 3*omegaVar*m_zz_I + 24)/(18*(24*m_yy_I - 6*m_xx_I + 57*m_yz_I + 24*m_zz_I + 6*omegaVar*m_xx_I - 24*omegaVar*m_yy_I - 57*omegaVar*m_yz_I - 24*omegaVar*m_zz_I + 23));
    m_xy_t90 = (5*m_xy_I*(43*omegaVar + 72))/(18*(24*m_yy_I - 6*m_xx_I + 57*m_yz_I + 24*m_zz_I + 6*omegaVar*m_xx_I - 24*omegaVar*m_yy_I - 57*omegaVar*m_yz_I - 24*omegaVar*m_zz_I + 23));
    m_xz_t90 = (5*m_xz_I*(43*omegaVar + 72))/(18*(24*m_yy_I - 6*m_xx_I + 57*m_yz_I + 24*m_zz_I + 6*omegaVar*m_xx_I - 24*omegaVar*m_yy_I - 57*omegaVar*m_yz_I - 24*omegaVar*m_zz_I + 23));
    m_yy_t45 = (936*m_yy_I - 144*m_xx_I + 1008*m_yz_I + 216*m_zz_I - 6*omegaVar*m_xx_I + 239*omegaVar*m_yy_I - 158*omegaVar*m_yz_I - 191*omegaVar*m_zz_I + 192)/(36*(24*m_yy_I - 6*m_xx_I + 57*m_yz_I + 24*m_zz_I + 6*omegaVar*m_xx_I - 24*omegaVar*m_yy_I - 57*omegaVar*m_yz_I - 24*omegaVar*m_zz_I + 23));
    m_yz_t90 = (504*m_yy_I - 216*m_xx_I + 2412*m_yz_I + 504*m_zz_I - 34*omegaVar*m_xx_I - 79*omegaVar*m_yy_I + 538*omegaVar*m_yz_I - 79*omegaVar*m_zz_I + 228)/(36*(24*m_yy_I - 6*m_xx_I + 57*m_yz_I + 24*m_zz_I + 6*omegaVar*m_xx_I - 24*omegaVar*m_yy_I - 57*omegaVar*m_yz_I - 24*omegaVar*m_zz_I + 23));
    m_zz_t45 = (216*m_yy_I - 144*m_xx_I + 1008*m_yz_I + 936*m_zz_I - 6*omegaVar*m_xx_I - 191*omegaVar*m_yy_I - 158*omegaVar*m_yz_I + 239*omegaVar*m_zz_I + 192)/(36*(24*m_yy_I - 6*m_xx_I + 57*m_yz_I + 24*m_zz_I + 6*omegaVar*m_xx_I - 24*omegaVar*m_yy_I - 57*omegaVar*m_yz_I - 24*omegaVar*m_zz_I + 23));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = -(rho*(m_xy_t90 - m_xz_t90)*(omegaVar - 2))/2;
    L_BC_Fy = (m_xx_t45/12 + (5*m_yy_t45)/6 - m_yz_t90/2 - m_zz_t45/6 - (omegaVar*m_xx_t45)/24 - (5*omegaVar*m_yy_t45)/12 + (omegaVar*m_yz_t90)/4 + (omegaVar*m_zz_t45)/12 + 5/18)*rho - (5*RHO_0)/18;
    L_BC_Fz = (m_yy_t45/6 - m_xx_t45/12 + m_yz_t90/2 - (5*m_zz_t45)/6 + (omegaVar*m_xx_t45)/24 - (omegaVar*m_yy_t45)/12 - (omegaVar*m_yz_t90)/4 + (5*omegaVar*m_zz_t45)/12 - 5/18)*rho + (5*RHO_0)/18;
    #endif //_BC_FORCES 

    break;
case 0b0000010111010000:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[13] + pop[14] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (6*(15*rho_I + m_xx_I*rho_I - 6*m_xz_I*rho_I + m_yy_I*rho_I + 6*m_yz_I*rho_I + 4*m_zz_I*rho_I - omegaVar*m_xx_I*rho_I + 6*omegaVar*m_xz_I*rho_I - omegaVar*m_yy_I*rho_I - 6*omegaVar*m_yz_I*rho_I - 4*omegaVar*m_zz_I*rho_I))/(3*omegaVar + 82);

    m_xx_t45 = -(504*m_xz_I - 2052*m_xx_I + 162*m_yy_I + 234*m_yz_I - 90*m_zz_I - 73*omegaVar*m_xx_I + 6*omegaVar*m_xz_I + 8*omegaVar*m_yy_I + 21*omegaVar*m_yz_I + 5*omegaVar*m_zz_I - 30)/(135*(m_xx_I - 6*m_xz_I + m_yy_I + 6*m_yz_I + 4*m_zz_I - omegaVar*m_xx_I + 6*omegaVar*m_xz_I - omegaVar*m_yy_I - 6*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 15));
    m_xy_t90 = (m_xy_I*(3*omegaVar + 82))/(6*(m_xx_I - 6*m_xz_I + m_yy_I + 6*m_yz_I + 4*m_zz_I - omegaVar*m_xx_I + 6*omegaVar*m_xz_I - omegaVar*m_yy_I - 6*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 15));
    m_xz_t90 = (1746*m_xz_I - 168*m_xx_I + 78*m_yy_I - 24*m_yz_I - 180*m_zz_I - 2*omegaVar*m_xx_I + 39*omegaVar*m_xz_I + 7*omegaVar*m_yy_I + 24*omegaVar*m_yz_I + 10*omegaVar*m_zz_I - 60)/(90*(m_xx_I - 6*m_xz_I + m_yy_I + 6*m_yz_I + 4*m_zz_I - omegaVar*m_xx_I + 6*omegaVar*m_xz_I - omegaVar*m_yy_I - 6*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 15));
    m_yy_t45 = (234*m_xz_I - 162*m_xx_I + 2052*m_yy_I + 504*m_yz_I + 90*m_zz_I - 8*omegaVar*m_xx_I + 21*omegaVar*m_xz_I + 73*omegaVar*m_yy_I + 6*omegaVar*m_yz_I - 5*omegaVar*m_zz_I + 30)/(135*(m_xx_I - 6*m_xz_I + m_yy_I + 6*m_yz_I + 4*m_zz_I - omegaVar*m_xx_I + 6*omegaVar*m_xz_I - omegaVar*m_yy_I - 6*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 15));
    m_yz_t90 = (168*m_yy_I - 24*m_xz_I - 78*m_xx_I + 1746*m_yz_I + 180*m_zz_I - 7*omegaVar*m_xx_I + 24*omegaVar*m_xz_I + 2*omegaVar*m_yy_I + 39*omegaVar*m_yz_I - 10*omegaVar*m_zz_I + 60)/(90*(m_xx_I - 6*m_xz_I + m_yy_I + 6*m_yz_I + 4*m_zz_I - omegaVar*m_xx_I + 6*omegaVar*m_xz_I - omegaVar*m_yy_I - 6*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 15));
    m_zz_t45 = (36*m_xx_I - 216*m_xz_I + 36*m_yy_I + 216*m_yz_I + 882*m_zz_I - 2*omegaVar*m_xx_I + 12*omegaVar*m_xz_I - 2*omegaVar*m_yy_I - 12*omegaVar*m_yz_I + 19*omegaVar*m_zz_I + 48)/(54*(m_xx_I - 6*m_xz_I + m_yy_I + 6*m_yz_I + 4*m_zz_I - omegaVar*m_xx_I + 6*omegaVar*m_xz_I - omegaVar*m_yy_I - 6*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 15));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = (m_xz_t90/2 - m_xx_t45/6 + m_yy_t45/12 - m_zz_t45/6 + (omegaVar*m_xx_t45)/12 - (omegaVar*m_xz_t90)/4 - (omegaVar*m_yy_t45)/24 + (omegaVar*m_zz_t45)/12 - 1.0/18.0)*rho + RHO_0/18;
    L_BC_Fy = (m_yy_t45/6 - m_xx_t45/12 + m_yz_t90/2 + m_zz_t45/6 + (omegaVar*m_xx_t45)/24 - (omegaVar*m_yy_t45)/12 - (omegaVar*m_yz_t90)/4 - (omegaVar*m_zz_t45)/12 + 1.0/18.0)*rho - RHO_0/18;
    L_BC_Fz = (m_xx_t45/12 - m_xz_t90/2 + m_yy_t45/12 + m_yz_t90/2 + m_zz_t45/3 - (omegaVar*m_xx_t45)/24 + (omegaVar*m_xz_t90)/4 - (omegaVar*m_yy_t45)/24 - (omegaVar*m_yz_t90)/4 - (omegaVar*m_zz_t45)/6 + 1.0/9.0)*rho - RHO_0/9;
    #endif //_BC_FORCES 

    break;
case 0b0000010111010001:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[8] + pop[9] + pop[10] + pop[11] + pop[13] + pop[14] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (12*(5*rho_I + m_xx_I*rho_I + 2*m_xy_I*rho_I - 2*m_xz_I*rho_I + m_yy_I*rho_I + 2*m_yz_I*rho_I + m_zz_I*rho_I - omegaVar*m_xx_I*rho_I - 2*omegaVar*m_xy_I*rho_I + 2*omegaVar*m_xz_I*rho_I - omegaVar*m_yy_I*rho_I - 2*omegaVar*m_yz_I*rho_I - omegaVar*m_zz_I*rho_I))/(3*omegaVar + 52);

    m_xx_t45 = (472*m_xx_I + 112*m_xy_I - 112*m_xz_I + 4*m_yy_I - 44*m_yz_I + 4*m_zz_I + 23*omegaVar*m_xx_I - 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - 4*omegaVar*m_yy_I - 11*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 20)/(90*(m_xx_I + 2*m_xy_I - 2*m_xz_I + m_yy_I + 2*m_yz_I + m_zz_I - omegaVar*m_xx_I - 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - omegaVar*m_yy_I - 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_xy_t90 = (112*m_xx_I + 1108*m_xy_I - 16*m_xz_I + 112*m_yy_I + 16*m_yz_I - 44*m_zz_I - 2*omegaVar*m_xx_I + 47*omegaVar*m_xy_I + 16*omegaVar*m_xz_I - 2*omegaVar*m_yy_I - 16*omegaVar*m_yz_I - 11*omegaVar*m_zz_I + 40)/(180*(m_xx_I + 2*m_xy_I - 2*m_xz_I + m_yy_I + 2*m_yz_I + m_zz_I - omegaVar*m_xx_I - 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - omegaVar*m_yy_I - 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_xz_t90 = (1108*m_xz_I - 16*m_xy_I - 112*m_xx_I + 44*m_yy_I - 16*m_yz_I - 112*m_zz_I + 2*omegaVar*m_xx_I + 16*omegaVar*m_xy_I + 47*omegaVar*m_xz_I + 11*omegaVar*m_yy_I + 16*omegaVar*m_yz_I + 2*omegaVar*m_zz_I - 40)/(180*(m_xx_I + 2*m_xy_I - 2*m_xz_I + m_yy_I + 2*m_yz_I + m_zz_I - omegaVar*m_xx_I - 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - omegaVar*m_yy_I - 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_yy_t45 = (4*m_xx_I + 112*m_xy_I + 44*m_xz_I + 472*m_yy_I + 112*m_yz_I + 4*m_zz_I - 4*omegaVar*m_xx_I - 2*omegaVar*m_xy_I + 11*omegaVar*m_xz_I + 23*omegaVar*m_yy_I - 2*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 20)/(90*(m_xx_I + 2*m_xy_I - 2*m_xz_I + m_yy_I + 2*m_yz_I + m_zz_I - omegaVar*m_xx_I - 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - omegaVar*m_yy_I - 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_yz_t90 = (16*m_xy_I - 44*m_xx_I - 16*m_xz_I + 112*m_yy_I + 1108*m_yz_I + 112*m_zz_I - 11*omegaVar*m_xx_I - 16*omegaVar*m_xy_I + 16*omegaVar*m_xz_I - 2*omegaVar*m_yy_I + 47*omegaVar*m_yz_I - 2*omegaVar*m_zz_I + 40)/(180*(m_xx_I + 2*m_xy_I - 2*m_xz_I + m_yy_I + 2*m_yz_I + m_zz_I - omegaVar*m_xx_I - 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - omegaVar*m_yy_I - 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_zz_t45 = (4*m_xx_I - 44*m_xy_I - 112*m_xz_I + 4*m_yy_I + 112*m_yz_I + 472*m_zz_I - 4*omegaVar*m_xx_I - 11*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - 4*omegaVar*m_yy_I - 2*omegaVar*m_yz_I + 23*omegaVar*m_zz_I + 20)/(90*(m_xx_I + 2*m_xy_I - 2*m_xz_I + m_yy_I + 2*m_yz_I + m_zz_I - omegaVar*m_xx_I - 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - omegaVar*m_yy_I - 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = (m_xz_t90/2 - m_xy_t90/2 - m_xx_t45/3 - m_yy_t45/12 - m_zz_t45/12 + (omegaVar*m_xx_t45)/6 + (omegaVar*m_xy_t90)/4 - (omegaVar*m_xz_t90)/4 + (omegaVar*m_yy_t45)/24 + (omegaVar*m_zz_t45)/24 - 1.0/9.0)*rho + RHO_0/9;
    L_BC_Fy = (rho*(omegaVar - 2)*(m_xx_t45 + 2*m_xy_t90 - 2*m_yz_t90 - m_zz_t45))/8;
    L_BC_Fz = (m_xx_t45/12 - m_xz_t90/2 + m_yy_t45/12 + m_yz_t90/2 + m_zz_t45/3 - (omegaVar*m_xx_t45)/24 + (omegaVar*m_xz_t90)/4 - (omegaVar*m_yy_t45)/24 - (omegaVar*m_yz_t90)/4 - (omegaVar*m_zz_t45)/6 + 1.0/9.0)*rho - RHO_0/9;
    #endif //_BC_FORCES 

    break;
case 0b0000010111010010:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[13] + pop[14] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (6*(15*rho_I + m_xx_I*rho_I - 6*m_xz_I*rho_I + m_yy_I*rho_I + 6*m_yz_I*rho_I + 4*m_zz_I*rho_I - omegaVar*m_xx_I*rho_I + 6*omegaVar*m_xz_I*rho_I - omegaVar*m_yy_I*rho_I - 6*omegaVar*m_yz_I*rho_I - 4*omegaVar*m_zz_I*rho_I))/(3*omegaVar + 82);

    m_xx_t45 = -(504*m_xz_I - 2052*m_xx_I + 162*m_yy_I + 234*m_yz_I - 90*m_zz_I - 73*omegaVar*m_xx_I + 6*omegaVar*m_xz_I + 8*omegaVar*m_yy_I + 21*omegaVar*m_yz_I + 5*omegaVar*m_zz_I - 30)/(135*(m_xx_I - 6*m_xz_I + m_yy_I + 6*m_yz_I + 4*m_zz_I - omegaVar*m_xx_I + 6*omegaVar*m_xz_I - omegaVar*m_yy_I - 6*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 15));
    m_xy_t90 = (m_xy_I*(3*omegaVar + 82))/(6*(m_xx_I - 6*m_xz_I + m_yy_I + 6*m_yz_I + 4*m_zz_I - omegaVar*m_xx_I + 6*omegaVar*m_xz_I - omegaVar*m_yy_I - 6*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 15));
    m_xz_t90 = (1746*m_xz_I - 168*m_xx_I + 78*m_yy_I - 24*m_yz_I - 180*m_zz_I - 2*omegaVar*m_xx_I + 39*omegaVar*m_xz_I + 7*omegaVar*m_yy_I + 24*omegaVar*m_yz_I + 10*omegaVar*m_zz_I - 60)/(90*(m_xx_I - 6*m_xz_I + m_yy_I + 6*m_yz_I + 4*m_zz_I - omegaVar*m_xx_I + 6*omegaVar*m_xz_I - omegaVar*m_yy_I - 6*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 15));
    m_yy_t45 = (234*m_xz_I - 162*m_xx_I + 2052*m_yy_I + 504*m_yz_I + 90*m_zz_I - 8*omegaVar*m_xx_I + 21*omegaVar*m_xz_I + 73*omegaVar*m_yy_I + 6*omegaVar*m_yz_I - 5*omegaVar*m_zz_I + 30)/(135*(m_xx_I - 6*m_xz_I + m_yy_I + 6*m_yz_I + 4*m_zz_I - omegaVar*m_xx_I + 6*omegaVar*m_xz_I - omegaVar*m_yy_I - 6*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 15));
    m_yz_t90 = (168*m_yy_I - 24*m_xz_I - 78*m_xx_I + 1746*m_yz_I + 180*m_zz_I - 7*omegaVar*m_xx_I + 24*omegaVar*m_xz_I + 2*omegaVar*m_yy_I + 39*omegaVar*m_yz_I - 10*omegaVar*m_zz_I + 60)/(90*(m_xx_I - 6*m_xz_I + m_yy_I + 6*m_yz_I + 4*m_zz_I - omegaVar*m_xx_I + 6*omegaVar*m_xz_I - omegaVar*m_yy_I - 6*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 15));
    m_zz_t45 = (36*m_xx_I - 216*m_xz_I + 36*m_yy_I + 216*m_yz_I + 882*m_zz_I - 2*omegaVar*m_xx_I + 12*omegaVar*m_xz_I - 2*omegaVar*m_yy_I - 12*omegaVar*m_yz_I + 19*omegaVar*m_zz_I + 48)/(54*(m_xx_I - 6*m_xz_I + m_yy_I + 6*m_yz_I + 4*m_zz_I - omegaVar*m_xx_I + 6*omegaVar*m_xz_I - omegaVar*m_yy_I - 6*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 15));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = (m_xz_t90/2 - m_xx_t45/6 + m_yy_t45/12 - m_zz_t45/6 + (omegaVar*m_xx_t45)/12 - (omegaVar*m_xz_t90)/4 - (omegaVar*m_yy_t45)/24 + (omegaVar*m_zz_t45)/12 - 1.0/18.0)*rho + RHO_0/18;
    L_BC_Fy = (m_yy_t45/6 - m_xx_t45/12 + m_yz_t90/2 + m_zz_t45/6 + (omegaVar*m_xx_t45)/24 - (omegaVar*m_yy_t45)/12 - (omegaVar*m_yz_t90)/4 - (omegaVar*m_zz_t45)/12 + 1.0/18.0)*rho - RHO_0/18;
    L_BC_Fz = (m_xx_t45/12 - m_xz_t90/2 + m_yy_t45/12 + m_yz_t90/2 + m_zz_t45/3 - (omegaVar*m_xx_t45)/24 + (omegaVar*m_xz_t90)/4 - (omegaVar*m_yy_t45)/24 - (omegaVar*m_yz_t90)/4 - (omegaVar*m_zz_t45)/6 + 1.0/9.0)*rho - RHO_0/9;
    #endif //_BC_FORCES 

    break;
case 0b0000010111010011:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[8] + pop[9] + pop[10] + pop[13] + pop[14] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (-pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (3*(15*rho_I - m_xx_I*rho_I + 6*m_xy_I*rho_I - 6*m_xz_I*rho_I + 11*m_yy_I*rho_I + 30*m_yz_I*rho_I + 11*m_zz_I*rho_I + omegaVar*m_xx_I*rho_I - 6*omegaVar*m_xy_I*rho_I + 6*omegaVar*m_xz_I*rho_I - 11*omegaVar*m_yy_I*rho_I - 30*omegaVar*m_yz_I*rho_I - 11*omegaVar*m_zz_I*rho_I))/(9*omegaVar + 31);

    m_xx_t45 = -(2*(36*m_xz_I - 36*m_xy_I - 180*m_xx_I + 27*m_yy_I + 99*m_yz_I + 27*m_zz_I - 52*omegaVar*m_xx_I - 12*omegaVar*m_xy_I + 12*omegaVar*m_xz_I + 5*omegaVar*m_yy_I + 21*omegaVar*m_yz_I + 5*omegaVar*m_zz_I + 3))/(27*(6*m_xy_I - m_xx_I - 6*m_xz_I + 11*m_yy_I + 30*m_yz_I + 11*m_zz_I + omegaVar*m_xx_I - 6*omegaVar*m_xy_I + 6*omegaVar*m_xz_I - 11*omegaVar*m_yy_I - 30*omegaVar*m_yz_I - 11*omegaVar*m_zz_I + 15));
    m_xy_t90 = (60*m_xx_I + 663*m_xy_I - 12*m_xz_I + 84*m_yy_I + 60*m_yz_I - 9*m_zz_I + 20*omegaVar*m_xx_I + 177*omegaVar*m_xy_I + 12*omegaVar*m_xz_I - 4*omegaVar*m_yy_I - 60*omegaVar*m_yz_I - 31*omegaVar*m_zz_I + 30)/(45*(6*m_xy_I - m_xx_I - 6*m_xz_I + 11*m_yy_I + 30*m_yz_I + 11*m_zz_I + omegaVar*m_xx_I - 6*omegaVar*m_xy_I + 6*omegaVar*m_xz_I - 11*omegaVar*m_yy_I - 30*omegaVar*m_yz_I - 11*omegaVar*m_zz_I + 15));
    m_xz_t90 = (663*m_xz_I - 12*m_xy_I - 60*m_xx_I + 9*m_yy_I - 60*m_yz_I - 84*m_zz_I - 20*omegaVar*m_xx_I + 12*omegaVar*m_xy_I + 177*omegaVar*m_xz_I + 31*omegaVar*m_yy_I + 60*omegaVar*m_yz_I + 4*omegaVar*m_zz_I - 30)/(45*(6*m_xy_I - m_xx_I - 6*m_xz_I + 11*m_yy_I + 30*m_yz_I + 11*m_zz_I + omegaVar*m_xx_I - 6*omegaVar*m_xy_I + 6*omegaVar*m_xz_I - 11*omegaVar*m_yy_I - 30*omegaVar*m_yz_I - 11*omegaVar*m_zz_I + 15));
    m_yy_t45 = (2*(252*m_xy_I - 135*m_xx_I + 27*m_xz_I + 1206*m_yy_I + 1260*m_yz_I + 369*m_zz_I - 25*omegaVar*m_xx_I - 12*omegaVar*m_xy_I + 93*omegaVar*m_xz_I + 194*omegaVar*m_yy_I - 60*omegaVar*m_yz_I - 49*omegaVar*m_zz_I + 165))/(135*(6*m_xy_I - m_xx_I - 6*m_xz_I + 11*m_yy_I + 30*m_yz_I + 11*m_zz_I + omegaVar*m_xx_I - 6*omegaVar*m_xy_I + 6*omegaVar*m_xz_I - 11*omegaVar*m_yy_I - 30*omegaVar*m_yz_I - 11*omegaVar*m_zz_I + 15));
    m_yz_t90 = (12*m_xy_I - 33*m_xx_I - 12*m_xz_I + 84*m_yy_I + 339*m_yz_I + 84*m_zz_I - 7*omegaVar*m_xx_I - 12*omegaVar*m_xy_I + 12*omegaVar*m_xz_I - 4*omegaVar*m_yy_I + 21*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 30)/(9*(6*m_xy_I - m_xx_I - 6*m_xz_I + 11*m_yy_I + 30*m_yz_I + 11*m_zz_I + omegaVar*m_xx_I - 6*omegaVar*m_xy_I + 6*omegaVar*m_xz_I - 11*omegaVar*m_yy_I - 30*omegaVar*m_yz_I - 11*omegaVar*m_zz_I + 15));
    m_zz_t45 = -(2*(135*m_xx_I + 27*m_xy_I + 252*m_xz_I - 369*m_yy_I - 1260*m_yz_I - 1206*m_zz_I + 25*omegaVar*m_xx_I + 93*omegaVar*m_xy_I - 12*omegaVar*m_xz_I + 49*omegaVar*m_yy_I + 60*omegaVar*m_yz_I - 194*omegaVar*m_zz_I - 165))/(135*(6*m_xy_I - m_xx_I - 6*m_xz_I + 11*m_yy_I + 30*m_yz_I + 11*m_zz_I + omegaVar*m_xx_I - 6*omegaVar*m_xy_I + 6*omegaVar*m_xz_I - 11*omegaVar*m_yy_I - 30*omegaVar*m_yz_I - 11*omegaVar*m_zz_I + 15));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = (m_xz_t90/2 - m_xy_t90/2 - m_xx_t45/3 - m_yy_t45/12 - m_zz_t45/12 + (omegaVar*m_xx_t45)/6 + (omegaVar*m_xy_t90)/4 - (omegaVar*m_xz_t90)/4 + (omegaVar*m_yy_t45)/24 + (omegaVar*m_zz_t45)/24 - 1.0/9.0)*rho + RHO_0/9;
    L_BC_Fy = (m_zz_t45/12 - m_xy_t90/2 - m_yy_t45/6 - m_xx_t45/6 + (omegaVar*m_xx_t45)/12 + (omegaVar*m_xy_t90)/4 + (omegaVar*m_yy_t45)/12 - (omegaVar*m_zz_t45)/24 - 1.0/18.0)*rho + RHO_0/18;
    L_BC_Fz = (m_xx_t45/6 - m_xz_t90/2 - m_yy_t45/12 + m_zz_t45/6 - (omegaVar*m_xx_t45)/12 + (omegaVar*m_xz_t90)/4 + (omegaVar*m_yy_t45)/24 - (omegaVar*m_zz_t45)/12 + 1.0/18.0)*rho - RHO_0/18;
    #endif //_BC_FORCES 

    break;
case 0b0000010111010100:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[14] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (12*(5*rho_I + m_xx_I*rho_I - 2*m_xy_I*rho_I - 2*m_xz_I*rho_I + m_yy_I*rho_I + 2*m_yz_I*rho_I + m_zz_I*rho_I - omegaVar*m_xx_I*rho_I + 2*omegaVar*m_xy_I*rho_I + 2*omegaVar*m_xz_I*rho_I - omegaVar*m_yy_I*rho_I - 2*omegaVar*m_yz_I*rho_I - omegaVar*m_zz_I*rho_I))/(3*omegaVar + 52);

    m_xx_t45 = (472*m_xx_I - 112*m_xy_I - 112*m_xz_I + 4*m_yy_I - 44*m_yz_I + 4*m_zz_I + 23*omegaVar*m_xx_I + 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - 4*omegaVar*m_yy_I - 11*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 20)/(90*(m_xx_I - 2*m_xy_I - 2*m_xz_I + m_yy_I + 2*m_yz_I + m_zz_I - omegaVar*m_xx_I + 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - omegaVar*m_yy_I - 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_xy_t90 = (1108*m_xy_I - 112*m_xx_I + 16*m_xz_I - 112*m_yy_I - 16*m_yz_I + 44*m_zz_I + 2*omegaVar*m_xx_I + 47*omegaVar*m_xy_I - 16*omegaVar*m_xz_I + 2*omegaVar*m_yy_I + 16*omegaVar*m_yz_I + 11*omegaVar*m_zz_I - 40)/(180*(m_xx_I - 2*m_xy_I - 2*m_xz_I + m_yy_I + 2*m_yz_I + m_zz_I - omegaVar*m_xx_I + 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - omegaVar*m_yy_I - 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_xz_t90 = (16*m_xy_I - 112*m_xx_I + 1108*m_xz_I + 44*m_yy_I - 16*m_yz_I - 112*m_zz_I + 2*omegaVar*m_xx_I - 16*omegaVar*m_xy_I + 47*omegaVar*m_xz_I + 11*omegaVar*m_yy_I + 16*omegaVar*m_yz_I + 2*omegaVar*m_zz_I - 40)/(180*(m_xx_I - 2*m_xy_I - 2*m_xz_I + m_yy_I + 2*m_yz_I + m_zz_I - omegaVar*m_xx_I + 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - omegaVar*m_yy_I - 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_yy_t45 = (4*m_xx_I - 112*m_xy_I + 44*m_xz_I + 472*m_yy_I + 112*m_yz_I + 4*m_zz_I - 4*omegaVar*m_xx_I + 2*omegaVar*m_xy_I + 11*omegaVar*m_xz_I + 23*omegaVar*m_yy_I - 2*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 20)/(90*(m_xx_I - 2*m_xy_I - 2*m_xz_I + m_yy_I + 2*m_yz_I + m_zz_I - omegaVar*m_xx_I + 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - omegaVar*m_yy_I - 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_yz_t90 = (112*m_yy_I - 16*m_xy_I - 16*m_xz_I - 44*m_xx_I + 1108*m_yz_I + 112*m_zz_I - 11*omegaVar*m_xx_I + 16*omegaVar*m_xy_I + 16*omegaVar*m_xz_I - 2*omegaVar*m_yy_I + 47*omegaVar*m_yz_I - 2*omegaVar*m_zz_I + 40)/(180*(m_xx_I - 2*m_xy_I - 2*m_xz_I + m_yy_I + 2*m_yz_I + m_zz_I - omegaVar*m_xx_I + 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - omegaVar*m_yy_I - 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_zz_t45 = (4*m_xx_I + 44*m_xy_I - 112*m_xz_I + 4*m_yy_I + 112*m_yz_I + 472*m_zz_I - 4*omegaVar*m_xx_I + 11*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - 4*omegaVar*m_yy_I - 2*omegaVar*m_yz_I + 23*omegaVar*m_zz_I + 20)/(90*(m_xx_I - 2*m_xy_I - 2*m_xz_I + m_yy_I + 2*m_yz_I + m_zz_I - omegaVar*m_xx_I + 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - omegaVar*m_yy_I - 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = (m_xy_t90/2 - m_xx_t45/3 + m_xz_t90/2 - m_yy_t45/12 - m_zz_t45/12 + (omegaVar*m_xx_t45)/6 - (omegaVar*m_xy_t90)/4 - (omegaVar*m_xz_t90)/4 + (omegaVar*m_yy_t45)/24 + (omegaVar*m_zz_t45)/24 - 1.0/9.0)*rho + RHO_0/9;
    L_BC_Fy = (m_xx_t45/12 - m_xy_t90/2 + m_yy_t45/3 + m_yz_t90/2 + m_zz_t45/12 - (omegaVar*m_xx_t45)/24 + (omegaVar*m_xy_t90)/4 - (omegaVar*m_yy_t45)/6 - (omegaVar*m_yz_t90)/4 - (omegaVar*m_zz_t45)/24 + 1.0/9.0)*rho - RHO_0/9;
    L_BC_Fz = (m_xx_t45/12 - m_xz_t90/2 + m_yy_t45/12 + m_yz_t90/2 + m_zz_t45/3 - (omegaVar*m_xx_t45)/24 + (omegaVar*m_xz_t90)/4 - (omegaVar*m_yy_t45)/24 - (omegaVar*m_yz_t90)/4 - (omegaVar*m_zz_t45)/6 + 1.0/9.0)*rho - RHO_0/9;
    #endif //_BC_FORCES 

    break;
case 0b0000010111010101:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[8] + pop[10] + pop[11] + pop[14] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[8] - pop[14]);
    m_xz_I = inv_rho_I * (pop[10] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (36*(107*rho_I + 72*m_xx_I*rho_I + 12*m_yy_I*rho_I + 33*m_yz_I*rho_I + 12*m_zz_I*rho_I - 72*omegaVar*m_xx_I*rho_I - 12*omegaVar*m_yy_I*rho_I - 33*omegaVar*m_yz_I*rho_I - 12*omegaVar*m_zz_I*rho_I))/(313*omegaVar + 2790);

    m_xx_t45 = -(72*m_yy_I - 2358*m_xx_I + 198*m_yz_I + 72*m_zz_I - 49*omegaVar*m_xx_I + 44*omegaVar*m_yy_I + 121*omegaVar*m_yz_I + 44*omegaVar*m_zz_I - 288)/(18*(72*m_xx_I + 12*m_yy_I + 33*m_yz_I + 12*m_zz_I - 72*omegaVar*m_xx_I - 12*omegaVar*m_yy_I - 33*omegaVar*m_yz_I - 12*omegaVar*m_zz_I + 107));
    m_xy_t90 = (m_xy_I*(313*omegaVar + 2790))/(18*(72*m_xx_I + 12*m_yy_I + 33*m_yz_I + 12*m_zz_I - 72*omegaVar*m_xx_I - 12*omegaVar*m_yy_I - 33*omegaVar*m_yz_I - 12*omegaVar*m_zz_I + 107));
    m_xz_t90 = (m_xz_I*(313*omegaVar + 2790))/(18*(72*m_xx_I + 12*m_yy_I + 33*m_yz_I + 12*m_zz_I - 72*omegaVar*m_xx_I - 12*omegaVar*m_yy_I - 33*omegaVar*m_yz_I - 12*omegaVar*m_zz_I + 107));
    m_yy_t45 = (2772*m_yy_I - 108*m_xx_I + 648*m_yz_I - 18*m_zz_I - 66*omegaVar*m_xx_I + 302*omegaVar*m_yy_I + 48*omegaVar*m_yz_I - 11*omegaVar*m_zz_I + 72)/(27*(72*m_xx_I + 12*m_yy_I + 33*m_yz_I + 12*m_zz_I - 72*omegaVar*m_xx_I - 12*omegaVar*m_yy_I - 33*omegaVar*m_yz_I - 12*omegaVar*m_zz_I + 107));
    m_yz_t90 = (432*m_yy_I - 198*m_xx_I + 3978*m_yz_I + 432*m_zz_I - 121*omegaVar*m_xx_I + 32*omegaVar*m_yy_I + 401*omegaVar*m_yz_I + 32*omegaVar*m_zz_I + 132)/(36*(72*m_xx_I + 12*m_yy_I + 33*m_yz_I + 12*m_zz_I - 72*omegaVar*m_xx_I - 12*omegaVar*m_yy_I - 33*omegaVar*m_yz_I - 12*omegaVar*m_zz_I + 107));
    m_zz_t45 = (648*m_yz_I - 18*m_yy_I - 108*m_xx_I + 2772*m_zz_I - 66*omegaVar*m_xx_I - 11*omegaVar*m_yy_I + 48*omegaVar*m_yz_I + 302*omegaVar*m_zz_I + 72)/(27*(72*m_xx_I + 12*m_yy_I + 33*m_yz_I + 12*m_zz_I - 72*omegaVar*m_xx_I - 12*omegaVar*m_yy_I - 33*omegaVar*m_yz_I - 12*omegaVar*m_zz_I + 107));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = ((omegaVar*m_xx_t45)/2 - m_xx_t45 - 1.0/3.0)*rho + RHO_0/3;
    L_BC_Fy = (m_yy_t45/6 - m_xy_t90 - m_xx_t45/12 + m_yz_t90/2 + m_zz_t45/6 + (omegaVar*m_xx_t45)/24 + (omegaVar*m_xy_t90)/2 - (omegaVar*m_yy_t45)/12 - (omegaVar*m_yz_t90)/4 - (omegaVar*m_zz_t45)/12 + 1.0/18.0)*rho - RHO_0/18;
    L_BC_Fz = (m_yy_t45/6 - m_xz_t90 - m_xx_t45/12 + m_yz_t90/2 + m_zz_t45/6 + (omegaVar*m_xx_t45)/24 + (omegaVar*m_xz_t90)/2 - (omegaVar*m_yy_t45)/12 - (omegaVar*m_yz_t90)/4 - (omegaVar*m_zz_t45)/12 + 1.0/18.0)*rho - RHO_0/18;
    #endif //_BC_FORCES 

    break;
case 0b0000010111010110:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[14] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (12*(5*rho_I + m_xx_I*rho_I - 2*m_xy_I*rho_I - 2*m_xz_I*rho_I + m_yy_I*rho_I + 2*m_yz_I*rho_I + m_zz_I*rho_I - omegaVar*m_xx_I*rho_I + 2*omegaVar*m_xy_I*rho_I + 2*omegaVar*m_xz_I*rho_I - omegaVar*m_yy_I*rho_I - 2*omegaVar*m_yz_I*rho_I - omegaVar*m_zz_I*rho_I))/(3*omegaVar + 52);

    m_xx_t45 = (472*m_xx_I - 112*m_xy_I - 112*m_xz_I + 4*m_yy_I - 44*m_yz_I + 4*m_zz_I + 23*omegaVar*m_xx_I + 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - 4*omegaVar*m_yy_I - 11*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 20)/(90*(m_xx_I - 2*m_xy_I - 2*m_xz_I + m_yy_I + 2*m_yz_I + m_zz_I - omegaVar*m_xx_I + 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - omegaVar*m_yy_I - 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_xy_t90 = (1108*m_xy_I - 112*m_xx_I + 16*m_xz_I - 112*m_yy_I - 16*m_yz_I + 44*m_zz_I + 2*omegaVar*m_xx_I + 47*omegaVar*m_xy_I - 16*omegaVar*m_xz_I + 2*omegaVar*m_yy_I + 16*omegaVar*m_yz_I + 11*omegaVar*m_zz_I - 40)/(180*(m_xx_I - 2*m_xy_I - 2*m_xz_I + m_yy_I + 2*m_yz_I + m_zz_I - omegaVar*m_xx_I + 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - omegaVar*m_yy_I - 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_xz_t90 = (16*m_xy_I - 112*m_xx_I + 1108*m_xz_I + 44*m_yy_I - 16*m_yz_I - 112*m_zz_I + 2*omegaVar*m_xx_I - 16*omegaVar*m_xy_I + 47*omegaVar*m_xz_I + 11*omegaVar*m_yy_I + 16*omegaVar*m_yz_I + 2*omegaVar*m_zz_I - 40)/(180*(m_xx_I - 2*m_xy_I - 2*m_xz_I + m_yy_I + 2*m_yz_I + m_zz_I - omegaVar*m_xx_I + 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - omegaVar*m_yy_I - 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_yy_t45 = (4*m_xx_I - 112*m_xy_I + 44*m_xz_I + 472*m_yy_I + 112*m_yz_I + 4*m_zz_I - 4*omegaVar*m_xx_I + 2*omegaVar*m_xy_I + 11*omegaVar*m_xz_I + 23*omegaVar*m_yy_I - 2*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 20)/(90*(m_xx_I - 2*m_xy_I - 2*m_xz_I + m_yy_I + 2*m_yz_I + m_zz_I - omegaVar*m_xx_I + 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - omegaVar*m_yy_I - 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_yz_t90 = (112*m_yy_I - 16*m_xy_I - 16*m_xz_I - 44*m_xx_I + 1108*m_yz_I + 112*m_zz_I - 11*omegaVar*m_xx_I + 16*omegaVar*m_xy_I + 16*omegaVar*m_xz_I - 2*omegaVar*m_yy_I + 47*omegaVar*m_yz_I - 2*omegaVar*m_zz_I + 40)/(180*(m_xx_I - 2*m_xy_I - 2*m_xz_I + m_yy_I + 2*m_yz_I + m_zz_I - omegaVar*m_xx_I + 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - omegaVar*m_yy_I - 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_zz_t45 = (4*m_xx_I + 44*m_xy_I - 112*m_xz_I + 4*m_yy_I + 112*m_yz_I + 472*m_zz_I - 4*omegaVar*m_xx_I + 11*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - 4*omegaVar*m_yy_I - 2*omegaVar*m_yz_I + 23*omegaVar*m_zz_I + 20)/(90*(m_xx_I - 2*m_xy_I - 2*m_xz_I + m_yy_I + 2*m_yz_I + m_zz_I - omegaVar*m_xx_I + 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - omegaVar*m_yy_I - 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = (m_xy_t90/2 - m_xx_t45/3 + m_xz_t90/2 - m_yy_t45/12 - m_zz_t45/12 + (omegaVar*m_xx_t45)/6 - (omegaVar*m_xy_t90)/4 - (omegaVar*m_xz_t90)/4 + (omegaVar*m_yy_t45)/24 + (omegaVar*m_zz_t45)/24 - 1.0/9.0)*rho + RHO_0/9;
    L_BC_Fy = (m_xx_t45/12 - m_xy_t90/2 + m_yy_t45/3 + m_yz_t90/2 + m_zz_t45/12 - (omegaVar*m_xx_t45)/24 + (omegaVar*m_xy_t90)/4 - (omegaVar*m_yy_t45)/6 - (omegaVar*m_yz_t90)/4 - (omegaVar*m_zz_t45)/24 + 1.0/9.0)*rho - RHO_0/9;
    L_BC_Fz = (m_xx_t45/12 - m_xz_t90/2 + m_yy_t45/12 + m_yz_t90/2 + m_zz_t45/3 - (omegaVar*m_xx_t45)/24 + (omegaVar*m_xz_t90)/4 - (omegaVar*m_yy_t45)/24 - (omegaVar*m_yz_t90)/4 - (omegaVar*m_zz_t45)/6 + 1.0/9.0)*rho - RHO_0/9;
    #endif //_BC_FORCES 

    break;
case 0b0000010111010111:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[8] + pop[10] + pop[14] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[8] - pop[14]);
    m_xz_I = inv_rho_I * (pop[10] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (-pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (18*(19*rho_I + 6*m_xx_I*rho_I + 12*m_yy_I*rho_I + 33*m_yz_I*rho_I + 12*m_zz_I*rho_I - 6*omegaVar*m_xx_I*rho_I - 12*omegaVar*m_yy_I*rho_I - 33*omegaVar*m_yz_I*rho_I - 12*omegaVar*m_zz_I*rho_I))/(59*omegaVar + 207);

    m_xx_t45 = -(36*m_yy_I - 189*m_xx_I + 99*m_yz_I + 36*m_zz_I - 49*omegaVar*m_xx_I + 20*omegaVar*m_yy_I + 55*omegaVar*m_yz_I + 20*omegaVar*m_zz_I - 12)/(9*(6*m_xx_I + 12*m_yy_I + 33*m_yz_I + 12*m_zz_I - 6*omegaVar*m_xx_I - 12*omegaVar*m_yy_I - 33*omegaVar*m_yz_I - 12*omegaVar*m_zz_I + 19));
    m_xy_t90 = (m_xy_I*(59*omegaVar + 207))/(9*(6*m_xx_I + 12*m_yy_I + 33*m_yz_I + 12*m_zz_I - 6*omegaVar*m_xx_I - 12*omegaVar*m_yy_I - 33*omegaVar*m_yz_I - 12*omegaVar*m_zz_I + 19));
    m_xz_t90 = (m_xz_I*(59*omegaVar + 207))/(9*(6*m_xx_I + 12*m_yy_I + 33*m_yz_I + 12*m_zz_I - 6*omegaVar*m_xx_I - 12*omegaVar*m_yy_I - 33*omegaVar*m_yz_I - 12*omegaVar*m_zz_I + 19));
    m_yy_t45 = (2*(306*m_yy_I - 54*m_xx_I + 324*m_yz_I + 99*m_zz_I - 30*omegaVar*m_xx_I + 58*omegaVar*m_yy_I + 12*omegaVar*m_yz_I - omegaVar*m_zz_I + 36))/(27*(6*m_xx_I + 12*m_yy_I + 33*m_yz_I + 12*m_zz_I - 6*omegaVar*m_xx_I - 12*omegaVar*m_yy_I - 33*omegaVar*m_yz_I - 12*omegaVar*m_zz_I + 19));
    m_yz_t90 = (216*m_yy_I - 99*m_xx_I + 801*m_yz_I + 216*m_zz_I - 55*omegaVar*m_xx_I + 8*omegaVar*m_yy_I + 81*omegaVar*m_yz_I + 8*omegaVar*m_zz_I + 66)/(18*(6*m_xx_I + 12*m_yy_I + 33*m_yz_I + 12*m_zz_I - 6*omegaVar*m_xx_I - 12*omegaVar*m_yy_I - 33*omegaVar*m_yz_I - 12*omegaVar*m_zz_I + 19));
    m_zz_t45 = (2*(99*m_yy_I - 54*m_xx_I + 324*m_yz_I + 306*m_zz_I - 30*omegaVar*m_xx_I - omegaVar*m_yy_I + 12*omegaVar*m_yz_I + 58*omegaVar*m_zz_I + 36))/(27*(6*m_xx_I + 12*m_yy_I + 33*m_yz_I + 12*m_zz_I - 6*omegaVar*m_xx_I - 12*omegaVar*m_yy_I - 33*omegaVar*m_yz_I - 12*omegaVar*m_zz_I + 19));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = ((omegaVar*m_xx_t45)/2 - m_xx_t45 - 1.0/3.0)*rho + RHO_0/3;
    L_BC_Fy = (m_xy_t90*rho*(omegaVar - 2))/2;
    L_BC_Fz = (m_xz_t90*rho*(omegaVar - 2))/2;
    #endif //_BC_FORCES 

    break;
case 0b0000010111011000:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[9] + pop[10] + pop[11] + pop[13] + pop[14] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (12*(5*rho_I + m_xx_I*rho_I + 2*m_xy_I*rho_I - 2*m_xz_I*rho_I + m_yy_I*rho_I + 2*m_yz_I*rho_I + m_zz_I*rho_I - omegaVar*m_xx_I*rho_I - 2*omegaVar*m_xy_I*rho_I + 2*omegaVar*m_xz_I*rho_I - omegaVar*m_yy_I*rho_I - 2*omegaVar*m_yz_I*rho_I - omegaVar*m_zz_I*rho_I))/(3*omegaVar + 52);

    m_xx_t45 = (472*m_xx_I + 112*m_xy_I - 112*m_xz_I + 4*m_yy_I - 44*m_yz_I + 4*m_zz_I + 23*omegaVar*m_xx_I - 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - 4*omegaVar*m_yy_I - 11*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 20)/(90*(m_xx_I + 2*m_xy_I - 2*m_xz_I + m_yy_I + 2*m_yz_I + m_zz_I - omegaVar*m_xx_I - 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - omegaVar*m_yy_I - 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_xy_t90 = (112*m_xx_I + 1108*m_xy_I - 16*m_xz_I + 112*m_yy_I + 16*m_yz_I - 44*m_zz_I - 2*omegaVar*m_xx_I + 47*omegaVar*m_xy_I + 16*omegaVar*m_xz_I - 2*omegaVar*m_yy_I - 16*omegaVar*m_yz_I - 11*omegaVar*m_zz_I + 40)/(180*(m_xx_I + 2*m_xy_I - 2*m_xz_I + m_yy_I + 2*m_yz_I + m_zz_I - omegaVar*m_xx_I - 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - omegaVar*m_yy_I - 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_xz_t90 = (1108*m_xz_I - 16*m_xy_I - 112*m_xx_I + 44*m_yy_I - 16*m_yz_I - 112*m_zz_I + 2*omegaVar*m_xx_I + 16*omegaVar*m_xy_I + 47*omegaVar*m_xz_I + 11*omegaVar*m_yy_I + 16*omegaVar*m_yz_I + 2*omegaVar*m_zz_I - 40)/(180*(m_xx_I + 2*m_xy_I - 2*m_xz_I + m_yy_I + 2*m_yz_I + m_zz_I - omegaVar*m_xx_I - 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - omegaVar*m_yy_I - 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_yy_t45 = (4*m_xx_I + 112*m_xy_I + 44*m_xz_I + 472*m_yy_I + 112*m_yz_I + 4*m_zz_I - 4*omegaVar*m_xx_I - 2*omegaVar*m_xy_I + 11*omegaVar*m_xz_I + 23*omegaVar*m_yy_I - 2*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 20)/(90*(m_xx_I + 2*m_xy_I - 2*m_xz_I + m_yy_I + 2*m_yz_I + m_zz_I - omegaVar*m_xx_I - 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - omegaVar*m_yy_I - 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_yz_t90 = (16*m_xy_I - 44*m_xx_I - 16*m_xz_I + 112*m_yy_I + 1108*m_yz_I + 112*m_zz_I - 11*omegaVar*m_xx_I - 16*omegaVar*m_xy_I + 16*omegaVar*m_xz_I - 2*omegaVar*m_yy_I + 47*omegaVar*m_yz_I - 2*omegaVar*m_zz_I + 40)/(180*(m_xx_I + 2*m_xy_I - 2*m_xz_I + m_yy_I + 2*m_yz_I + m_zz_I - omegaVar*m_xx_I - 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - omegaVar*m_yy_I - 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_zz_t45 = (4*m_xx_I - 44*m_xy_I - 112*m_xz_I + 4*m_yy_I + 112*m_yz_I + 472*m_zz_I - 4*omegaVar*m_xx_I - 11*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - 4*omegaVar*m_yy_I - 2*omegaVar*m_yz_I + 23*omegaVar*m_zz_I + 20)/(90*(m_xx_I + 2*m_xy_I - 2*m_xz_I + m_yy_I + 2*m_yz_I + m_zz_I - omegaVar*m_xx_I - 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - omegaVar*m_yy_I - 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = -(rho*(omegaVar - 2)*(2*m_xy_t90 + 2*m_xz_t90 + m_yy_t45 - m_zz_t45))/8;
    L_BC_Fy = (m_xx_t45/12 + m_xy_t90/2 + m_yy_t45/3 + m_yz_t90/2 + m_zz_t45/12 - (omegaVar*m_xx_t45)/24 - (omegaVar*m_xy_t90)/4 - (omegaVar*m_yy_t45)/6 - (omegaVar*m_yz_t90)/4 - (omegaVar*m_zz_t45)/24 + 1.0/9.0)*rho - RHO_0/9;
    L_BC_Fz = (m_xx_t45/12 - m_xz_t90/2 + m_yy_t45/12 + m_yz_t90/2 + m_zz_t45/3 - (omegaVar*m_xx_t45)/24 + (omegaVar*m_xz_t90)/4 - (omegaVar*m_yy_t45)/24 - (omegaVar*m_yz_t90)/4 - (omegaVar*m_zz_t45)/6 + 1.0/9.0)*rho - RHO_0/9;
    #endif //_BC_FORCES 

    break;
case 0b0000010111011001:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[9] + pop[10] + pop[11] + pop[13] + pop[14] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (-pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (3*(15*rho_I + 11*m_xx_I*rho_I + 30*m_xy_I*rho_I - 6*m_xz_I*rho_I + 11*m_yy_I*rho_I + 6*m_yz_I*rho_I - m_zz_I*rho_I - 11*omegaVar*m_xx_I*rho_I - 30*omegaVar*m_xy_I*rho_I + 6*omegaVar*m_xz_I*rho_I - 11*omegaVar*m_yy_I*rho_I - 6*omegaVar*m_yz_I*rho_I + omegaVar*m_zz_I*rho_I))/(9*omegaVar + 31);

    m_xx_t45 = -(2*(252*m_xz_I - 1260*m_xy_I - 1206*m_xx_I - 369*m_yy_I + 27*m_yz_I + 135*m_zz_I - 194*omegaVar*m_xx_I + 60*omegaVar*m_xy_I - 12*omegaVar*m_xz_I + 49*omegaVar*m_yy_I + 93*omegaVar*m_yz_I + 25*omegaVar*m_zz_I - 165))/(135*(11*m_xx_I + 30*m_xy_I - 6*m_xz_I + 11*m_yy_I + 6*m_yz_I - m_zz_I - 11*omegaVar*m_xx_I - 30*omegaVar*m_xy_I + 6*omegaVar*m_xz_I - 11*omegaVar*m_yy_I - 6*omegaVar*m_yz_I + omegaVar*m_zz_I + 15));
    m_xy_t90 = (84*m_xx_I + 339*m_xy_I - 12*m_xz_I + 84*m_yy_I + 12*m_yz_I - 33*m_zz_I - 4*omegaVar*m_xx_I + 21*omegaVar*m_xy_I + 12*omegaVar*m_xz_I - 4*omegaVar*m_yy_I - 12*omegaVar*m_yz_I - 7*omegaVar*m_zz_I + 30)/(9*(11*m_xx_I + 30*m_xy_I - 6*m_xz_I + 11*m_yy_I + 6*m_yz_I - m_zz_I - 11*omegaVar*m_xx_I - 30*omegaVar*m_xy_I + 6*omegaVar*m_xz_I - 11*omegaVar*m_yy_I - 6*omegaVar*m_yz_I + omegaVar*m_zz_I + 15));
    m_xz_t90 = (663*m_xz_I - 60*m_xy_I - 84*m_xx_I + 9*m_yy_I - 12*m_yz_I - 60*m_zz_I + 4*omegaVar*m_xx_I + 60*omegaVar*m_xy_I + 177*omegaVar*m_xz_I + 31*omegaVar*m_yy_I + 12*omegaVar*m_yz_I - 20*omegaVar*m_zz_I - 30)/(45*(11*m_xx_I + 30*m_xy_I - 6*m_xz_I + 11*m_yy_I + 6*m_yz_I - m_zz_I - 11*omegaVar*m_xx_I - 30*omegaVar*m_xy_I + 6*omegaVar*m_xz_I - 11*omegaVar*m_yy_I - 6*omegaVar*m_yz_I + omegaVar*m_zz_I + 15));
    m_yy_t45 = (2*(369*m_xx_I + 1260*m_xy_I + 27*m_xz_I + 1206*m_yy_I + 252*m_yz_I - 135*m_zz_I - 49*omegaVar*m_xx_I - 60*omegaVar*m_xy_I + 93*omegaVar*m_xz_I + 194*omegaVar*m_yy_I - 12*omegaVar*m_yz_I - 25*omegaVar*m_zz_I + 165))/(135*(11*m_xx_I + 30*m_xy_I - 6*m_xz_I + 11*m_yy_I + 6*m_yz_I - m_zz_I - 11*omegaVar*m_xx_I - 30*omegaVar*m_xy_I + 6*omegaVar*m_xz_I - 11*omegaVar*m_yy_I - 6*omegaVar*m_yz_I + omegaVar*m_zz_I + 15));
    m_yz_t90 = (60*m_xy_I - 9*m_xx_I - 12*m_xz_I + 84*m_yy_I + 663*m_yz_I + 60*m_zz_I - 31*omegaVar*m_xx_I - 60*omegaVar*m_xy_I + 12*omegaVar*m_xz_I - 4*omegaVar*m_yy_I + 177*omegaVar*m_yz_I + 20*omegaVar*m_zz_I + 30)/(45*(11*m_xx_I + 30*m_xy_I - 6*m_xz_I + 11*m_yy_I + 6*m_yz_I - m_zz_I - 11*omegaVar*m_xx_I - 30*omegaVar*m_xy_I + 6*omegaVar*m_xz_I - 11*omegaVar*m_yy_I - 6*omegaVar*m_yz_I + omegaVar*m_zz_I + 15));
    m_zz_t45 = -(2*(27*m_xx_I + 99*m_xy_I + 36*m_xz_I + 27*m_yy_I - 36*m_yz_I - 180*m_zz_I + 5*omegaVar*m_xx_I + 21*omegaVar*m_xy_I + 12*omegaVar*m_xz_I + 5*omegaVar*m_yy_I - 12*omegaVar*m_yz_I - 52*omegaVar*m_zz_I + 3))/(27*(11*m_xx_I + 30*m_xy_I - 6*m_xz_I + 11*m_yy_I + 6*m_yz_I - m_zz_I - 11*omegaVar*m_xx_I - 30*omegaVar*m_xy_I + 6*omegaVar*m_xz_I - 11*omegaVar*m_yy_I - 6*omegaVar*m_yz_I + omegaVar*m_zz_I + 15));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = (m_xz_t90/2 - m_xx_t45/6 + m_yy_t45/12 - m_zz_t45/6 + (omegaVar*m_xx_t45)/12 - (omegaVar*m_xz_t90)/4 - (omegaVar*m_yy_t45)/24 + (omegaVar*m_zz_t45)/12 - 1.0/18.0)*rho + RHO_0/18;
    L_BC_Fy = (m_yy_t45/6 - m_xx_t45/12 + m_yz_t90/2 + m_zz_t45/6 + (omegaVar*m_xx_t45)/24 - (omegaVar*m_yy_t45)/12 - (omegaVar*m_yz_t90)/4 - (omegaVar*m_zz_t45)/12 + 1.0/18.0)*rho - RHO_0/18;
    L_BC_Fz = (m_xx_t45/12 - m_xz_t90/2 + m_yy_t45/12 + m_yz_t90/2 + m_zz_t45/3 - (omegaVar*m_xx_t45)/24 + (omegaVar*m_xz_t90)/4 - (omegaVar*m_yy_t45)/24 - (omegaVar*m_yz_t90)/4 - (omegaVar*m_zz_t45)/6 + 1.0/9.0)*rho - RHO_0/9;
    #endif //_BC_FORCES 

    break;
case 0b0000010111011010:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[9] + pop[10] + pop[11] + pop[13] + pop[14] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (3*(15*rho_I + 11*m_xx_I*rho_I + 6*m_xy_I*rho_I - 30*m_xz_I*rho_I - m_yy_I*rho_I + 6*m_yz_I*rho_I + 11*m_zz_I*rho_I - 11*omegaVar*m_xx_I*rho_I - 6*omegaVar*m_xy_I*rho_I + 30*omegaVar*m_xz_I*rho_I + omegaVar*m_yy_I*rho_I - 6*omegaVar*m_yz_I*rho_I - 11*omegaVar*m_zz_I*rho_I))/(9*omegaVar + 31);

    m_xx_t45 = -(2*(1260*m_xz_I - 252*m_xy_I - 1206*m_xx_I + 135*m_yy_I + 27*m_yz_I - 369*m_zz_I - 194*omegaVar*m_xx_I + 12*omegaVar*m_xy_I - 60*omegaVar*m_xz_I + 25*omegaVar*m_yy_I + 93*omegaVar*m_yz_I + 49*omegaVar*m_zz_I - 165))/(135*(11*m_xx_I + 6*m_xy_I - 30*m_xz_I - m_yy_I + 6*m_yz_I + 11*m_zz_I - 11*omegaVar*m_xx_I - 6*omegaVar*m_xy_I + 30*omegaVar*m_xz_I + omegaVar*m_yy_I - 6*omegaVar*m_yz_I - 11*omegaVar*m_zz_I + 15));
    m_xy_t90 = (84*m_xx_I + 663*m_xy_I - 60*m_xz_I + 60*m_yy_I + 12*m_yz_I - 9*m_zz_I - 4*omegaVar*m_xx_I + 177*omegaVar*m_xy_I + 60*omegaVar*m_xz_I + 20*omegaVar*m_yy_I - 12*omegaVar*m_yz_I - 31*omegaVar*m_zz_I + 30)/(45*(11*m_xx_I + 6*m_xy_I - 30*m_xz_I - m_yy_I + 6*m_yz_I + 11*m_zz_I - 11*omegaVar*m_xx_I - 6*omegaVar*m_xy_I + 30*omegaVar*m_xz_I + omegaVar*m_yy_I - 6*omegaVar*m_yz_I - 11*omegaVar*m_zz_I + 15));
    m_xz_t90 = (339*m_xz_I - 12*m_xy_I - 84*m_xx_I + 33*m_yy_I - 12*m_yz_I - 84*m_zz_I + 4*omegaVar*m_xx_I + 12*omegaVar*m_xy_I + 21*omegaVar*m_xz_I + 7*omegaVar*m_yy_I + 12*omegaVar*m_yz_I + 4*omegaVar*m_zz_I - 30)/(9*(11*m_xx_I + 6*m_xy_I - 30*m_xz_I - m_yy_I + 6*m_yz_I + 11*m_zz_I - 11*omegaVar*m_xx_I - 6*omegaVar*m_xy_I + 30*omegaVar*m_xz_I + omegaVar*m_yy_I - 6*omegaVar*m_yz_I - 11*omegaVar*m_zz_I + 15));
    m_yy_t45 = (2*(36*m_xy_I - 27*m_xx_I + 99*m_xz_I + 180*m_yy_I + 36*m_yz_I - 27*m_zz_I - 5*omegaVar*m_xx_I + 12*omegaVar*m_xy_I + 21*omegaVar*m_xz_I + 52*omegaVar*m_yy_I + 12*omegaVar*m_yz_I - 5*omegaVar*m_zz_I - 3))/(27*(11*m_xx_I + 6*m_xy_I - 30*m_xz_I - m_yy_I + 6*m_yz_I + 11*m_zz_I - 11*omegaVar*m_xx_I - 6*omegaVar*m_xy_I + 30*omegaVar*m_xz_I + omegaVar*m_yy_I - 6*omegaVar*m_yz_I - 11*omegaVar*m_zz_I + 15));
    m_yz_t90 = (12*m_xy_I - 9*m_xx_I - 60*m_xz_I + 60*m_yy_I + 663*m_yz_I + 84*m_zz_I - 31*omegaVar*m_xx_I - 12*omegaVar*m_xy_I + 60*omegaVar*m_xz_I + 20*omegaVar*m_yy_I + 177*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 30)/(45*(11*m_xx_I + 6*m_xy_I - 30*m_xz_I - m_yy_I + 6*m_yz_I + 11*m_zz_I - 11*omegaVar*m_xx_I - 6*omegaVar*m_xy_I + 30*omegaVar*m_xz_I + omegaVar*m_yy_I - 6*omegaVar*m_yz_I - 11*omegaVar*m_zz_I + 15));
    m_zz_t45 = -(2*(27*m_xy_I - 369*m_xx_I + 1260*m_xz_I + 135*m_yy_I - 252*m_yz_I - 1206*m_zz_I + 49*omegaVar*m_xx_I + 93*omegaVar*m_xy_I - 60*omegaVar*m_xz_I + 25*omegaVar*m_yy_I + 12*omegaVar*m_yz_I - 194*omegaVar*m_zz_I - 165))/(135*(11*m_xx_I + 6*m_xy_I - 30*m_xz_I - m_yy_I + 6*m_yz_I + 11*m_zz_I - 11*omegaVar*m_xx_I - 6*omegaVar*m_xy_I + 30*omegaVar*m_xz_I + omegaVar*m_yy_I - 6*omegaVar*m_yz_I - 11*omegaVar*m_zz_I + 15));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = (m_xx_t45/6 + m_xy_t90/2 + m_yy_t45/6 - m_zz_t45/12 - (omegaVar*m_xx_t45)/12 - (omegaVar*m_xy_t90)/4 - (omegaVar*m_yy_t45)/12 + (omegaVar*m_zz_t45)/24 + 1.0/18.0)*rho - RHO_0/18;
    L_BC_Fy = (m_xx_t45/12 + m_xy_t90/2 + m_yy_t45/3 + m_yz_t90/2 + m_zz_t45/12 - (omegaVar*m_xx_t45)/24 - (omegaVar*m_xy_t90)/4 - (omegaVar*m_yy_t45)/6 - (omegaVar*m_yz_t90)/4 - (omegaVar*m_zz_t45)/24 + 1.0/9.0)*rho - RHO_0/9;
    L_BC_Fz = (m_yy_t45/6 - m_xx_t45/12 + m_yz_t90/2 + m_zz_t45/6 + (omegaVar*m_xx_t45)/24 - (omegaVar*m_yy_t45)/12 - (omegaVar*m_yz_t90)/4 - (omegaVar*m_zz_t45)/12 + 1.0/18.0)*rho - RHO_0/18;
    #endif //_BC_FORCES 

    break;
case 0b0000010111011011:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[9] + pop[10] + pop[13] + pop[14] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (-pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (-pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (6*(rho_I + m_xx_I*rho_I + 2*m_xy_I*rho_I - 2*m_xz_I*rho_I + m_yy_I*rho_I + 2*m_yz_I*rho_I + m_zz_I*rho_I - omegaVar*m_xx_I*rho_I - 2*omegaVar*m_xy_I*rho_I + 2*omegaVar*m_xz_I*rho_I - omegaVar*m_yy_I*rho_I - 2*omegaVar*m_yz_I*rho_I - omegaVar*m_zz_I*rho_I))/(3*omegaVar + 2);

    m_xx_t45 = (8*m_xx_I + 8*m_xy_I - 8*m_xz_I + 2*m_yy_I + 2*m_yz_I + 2*m_zz_I + 7*omegaVar*m_xx_I + 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - 2*omegaVar*m_yy_I - 7*omegaVar*m_yz_I - 2*omegaVar*m_zz_I + 2)/(9*(m_xx_I + 2*m_xy_I - 2*m_xz_I + m_yy_I + 2*m_yz_I + m_zz_I - omegaVar*m_xx_I - 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - omegaVar*m_yy_I - 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 1));
    m_xy_t90 = (8*m_xx_I + 26*m_xy_I - 8*m_xz_I + 8*m_yy_I + 8*m_yz_I + 2*m_zz_I + 2*omegaVar*m_xx_I + 19*omegaVar*m_xy_I + 8*omegaVar*m_xz_I + 2*omegaVar*m_yy_I - 8*omegaVar*m_yz_I - 7*omegaVar*m_zz_I + 4)/(18*(m_xx_I + 2*m_xy_I - 2*m_xz_I + m_yy_I + 2*m_yz_I + m_zz_I - omegaVar*m_xx_I - 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - omegaVar*m_yy_I - 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 1));
    m_xz_t90 = -(8*m_xx_I + 8*m_xy_I - 26*m_xz_I + 2*m_yy_I + 8*m_yz_I + 8*m_zz_I + 2*omegaVar*m_xx_I - 8*omegaVar*m_xy_I - 19*omegaVar*m_xz_I - 7*omegaVar*m_yy_I - 8*omegaVar*m_yz_I + 2*omegaVar*m_zz_I + 4)/(18*(m_xx_I + 2*m_xy_I - 2*m_xz_I + m_yy_I + 2*m_yz_I + m_zz_I - omegaVar*m_xx_I - 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - omegaVar*m_yy_I - 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 1));
    m_yy_t45 = (2*m_xx_I + 8*m_xy_I - 2*m_xz_I + 8*m_yy_I + 8*m_yz_I + 2*m_zz_I - 2*omegaVar*m_xx_I + 2*omegaVar*m_xy_I + 7*omegaVar*m_xz_I + 7*omegaVar*m_yy_I + 2*omegaVar*m_yz_I - 2*omegaVar*m_zz_I + 2)/(9*(m_xx_I + 2*m_xy_I - 2*m_xz_I + m_yy_I + 2*m_yz_I + m_zz_I - omegaVar*m_xx_I - 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - omegaVar*m_yy_I - 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 1));
    m_yz_t90 = (2*m_xx_I + 8*m_xy_I - 8*m_xz_I + 8*m_yy_I + 26*m_yz_I + 8*m_zz_I - 7*omegaVar*m_xx_I - 8*omegaVar*m_xy_I + 8*omegaVar*m_xz_I + 2*omegaVar*m_yy_I + 19*omegaVar*m_yz_I + 2*omegaVar*m_zz_I + 4)/(18*(m_xx_I + 2*m_xy_I - 2*m_xz_I + m_yy_I + 2*m_yz_I + m_zz_I - omegaVar*m_xx_I - 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - omegaVar*m_yy_I - 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 1));
    m_zz_t45 = (2*m_xx_I + 2*m_xy_I - 8*m_xz_I + 2*m_yy_I + 8*m_yz_I + 8*m_zz_I - 2*omegaVar*m_xx_I - 7*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - 2*omegaVar*m_yy_I + 2*omegaVar*m_yz_I + 7*omegaVar*m_zz_I + 2)/(9*(m_xx_I + 2*m_xy_I - 2*m_xz_I + m_yy_I + 2*m_yz_I + m_zz_I - omegaVar*m_xx_I - 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - omegaVar*m_yy_I - 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 1));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = 0;
    L_BC_Fy = 0;
    L_BC_Fz = 0;
    #endif //_BC_FORCES 

    break;
case 0b0000010111011100:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[5] + pop[6] + pop[7] + pop[9] + pop[10] + pop[11] + pop[14] + pop[16] + pop[17];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17]);
    m_xy_I = inv_rho_I * (pop[7] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17]);
    m_yz_I = inv_rho_I * (pop[11] - pop[17]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17]);

    rho = (36*(107*rho_I + 12*m_xx_I*rho_I - 33*m_xz_I*rho_I + 72*m_yy_I*rho_I + 12*m_zz_I*rho_I - 12*omegaVar*m_xx_I*rho_I + 33*omegaVar*m_xz_I*rho_I - 72*omegaVar*m_yy_I*rho_I - 12*omegaVar*m_zz_I*rho_I))/(313*omegaVar + 2790);

    m_xx_t45 = -(648*m_xz_I - 2772*m_xx_I + 108*m_yy_I + 18*m_zz_I - 302*omegaVar*m_xx_I + 48*omegaVar*m_xz_I + 66*omegaVar*m_yy_I + 11*omegaVar*m_zz_I - 72)/(27*(12*m_xx_I - 33*m_xz_I + 72*m_yy_I + 12*m_zz_I - 12*omegaVar*m_xx_I + 33*omegaVar*m_xz_I - 72*omegaVar*m_yy_I - 12*omegaVar*m_zz_I + 107));
    m_xy_t90 = (m_xy_I*(313*omegaVar + 2790))/(18*(12*m_xx_I - 33*m_xz_I + 72*m_yy_I + 12*m_zz_I - 12*omegaVar*m_xx_I + 33*omegaVar*m_xz_I - 72*omegaVar*m_yy_I - 12*omegaVar*m_zz_I + 107));
    m_xz_t90 = -(432*m_xx_I - 3978*m_xz_I - 198*m_yy_I + 432*m_zz_I + 32*omegaVar*m_xx_I - 401*omegaVar*m_xz_I - 121*omegaVar*m_yy_I + 32*omegaVar*m_zz_I + 132)/(36*(12*m_xx_I - 33*m_xz_I + 72*m_yy_I + 12*m_zz_I - 12*omegaVar*m_xx_I + 33*omegaVar*m_xz_I - 72*omegaVar*m_yy_I - 12*omegaVar*m_zz_I + 107));
    m_yy_t45 = (198*m_xz_I - 72*m_xx_I + 2358*m_yy_I - 72*m_zz_I - 44*omegaVar*m_xx_I + 121*omegaVar*m_xz_I + 49*omegaVar*m_yy_I - 44*omegaVar*m_zz_I + 288)/(18*(12*m_xx_I - 33*m_xz_I + 72*m_yy_I + 12*m_zz_I - 12*omegaVar*m_xx_I + 33*omegaVar*m_xz_I - 72*omegaVar*m_yy_I - 12*omegaVar*m_zz_I + 107));
    m_yz_t90 = (m_yz_I*(313*omegaVar + 2790))/(18*(12*m_xx_I - 33*m_xz_I + 72*m_yy_I + 12*m_zz_I - 12*omegaVar*m_xx_I + 33*omegaVar*m_xz_I - 72*omegaVar*m_yy_I - 12*omegaVar*m_zz_I + 107));
    m_zz_t45 = -(18*m_xx_I + 648*m_xz_I + 108*m_yy_I - 2772*m_zz_I + 11*omegaVar*m_xx_I + 48*omegaVar*m_xz_I + 66*omegaVar*m_yy_I - 302*omegaVar*m_zz_I - 72)/(27*(12*m_xx_I - 33*m_xz_I + 72*m_yy_I + 12*m_zz_I - 12*omegaVar*m_xx_I + 33*omegaVar*m_xz_I - 72*omegaVar*m_yy_I - 12*omegaVar*m_zz_I + 107));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = (m_xy_t90 - m_xx_t45/6 + m_xz_t90/2 + m_yy_t45/12 - m_zz_t45/6 + (omegaVar*m_xx_t45)/12 - (omegaVar*m_xy_t90)/2 - (omegaVar*m_xz_t90)/4 - (omegaVar*m_yy_t45)/24 + (omegaVar*m_zz_t45)/12 - 1.0/18.0)*rho + RHO_0/18;
    L_BC_Fy = (m_yy_t45 - (omegaVar*m_yy_t45)/2 + 1.0/3.0)*rho - RHO_0/3;
    L_BC_Fz = (m_xx_t45/6 - m_xz_t90/2 - m_yy_t45/12 + m_yz_t90 + m_zz_t45/6 - (omegaVar*m_xx_t45)/12 + (omegaVar*m_xz_t90)/4 + (omegaVar*m_yy_t45)/24 - (omegaVar*m_yz_t90)/2 - (omegaVar*m_zz_t45)/12 + 1.0/18.0)*rho - RHO_0/18;
    #endif //_BC_FORCES 

    break;
case 0b0000010111011101:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[2] + pop[3] + pop[5] + pop[6] + pop[10] + pop[11] + pop[14] + pop[16] + pop[17];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17]);
    m_xy_I = inv_rho_I * (-pop[14]);
    m_xz_I = inv_rho_I * (pop[10] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17]);
    m_yz_I = inv_rho_I * (pop[11] - pop[17]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17]);

    rho = (36*(23*rho_I + 24*m_xx_I*rho_I + 57*m_xy_I*rho_I + 24*m_yy_I*rho_I - 6*m_zz_I*rho_I - 24*omegaVar*m_xx_I*rho_I - 57*omegaVar*m_xy_I*rho_I - 24*omegaVar*m_yy_I*rho_I + 6*omegaVar*m_zz_I*rho_I))/(5*(43*omegaVar + 72));

    m_xx_t45 = (936*m_xx_I + 1008*m_xy_I + 216*m_yy_I - 144*m_zz_I + 239*omegaVar*m_xx_I - 158*omegaVar*m_xy_I - 191*omegaVar*m_yy_I - 6*omegaVar*m_zz_I + 192)/(36*(24*m_xx_I + 57*m_xy_I + 24*m_yy_I - 6*m_zz_I - 24*omegaVar*m_xx_I - 57*omegaVar*m_xy_I - 24*omegaVar*m_yy_I + 6*omegaVar*m_zz_I + 23));
    m_xy_t90 = (504*m_xx_I + 2412*m_xy_I + 504*m_yy_I - 216*m_zz_I - 79*omegaVar*m_xx_I + 538*omegaVar*m_xy_I - 79*omegaVar*m_yy_I - 34*omegaVar*m_zz_I + 228)/(36*(24*m_xx_I + 57*m_xy_I + 24*m_yy_I - 6*m_zz_I - 24*omegaVar*m_xx_I - 57*omegaVar*m_xy_I - 24*omegaVar*m_yy_I + 6*omegaVar*m_zz_I + 23));
    m_xz_t90 = (5*m_xz_I*(43*omegaVar + 72))/(18*(24*m_xx_I + 57*m_xy_I + 24*m_yy_I - 6*m_zz_I - 24*omegaVar*m_xx_I - 57*omegaVar*m_xy_I - 24*omegaVar*m_yy_I + 6*omegaVar*m_zz_I + 23));
    m_yy_t45 = (216*m_xx_I + 1008*m_xy_I + 936*m_yy_I - 144*m_zz_I - 191*omegaVar*m_xx_I - 158*omegaVar*m_xy_I + 239*omegaVar*m_yy_I - 6*omegaVar*m_zz_I + 192)/(36*(24*m_xx_I + 57*m_xy_I + 24*m_yy_I - 6*m_zz_I - 24*omegaVar*m_xx_I - 57*omegaVar*m_xy_I - 24*omegaVar*m_yy_I + 6*omegaVar*m_zz_I + 23));
    m_yz_t90 = (5*m_yz_I*(43*omegaVar + 72))/(18*(24*m_xx_I + 57*m_xy_I + 24*m_yy_I - 6*m_zz_I - 24*omegaVar*m_xx_I - 57*omegaVar*m_xy_I - 24*omegaVar*m_yy_I + 6*omegaVar*m_zz_I + 23));
    m_zz_t45 = -(72*m_xx_I + 216*m_xy_I + 72*m_yy_I - 288*m_zz_I + 3*omegaVar*m_xx_I + 34*omegaVar*m_xy_I + 3*omegaVar*m_yy_I - 162*omegaVar*m_zz_I + 24)/(18*(24*m_xx_I + 57*m_xy_I + 24*m_yy_I - 6*m_zz_I - 24*omegaVar*m_xx_I - 57*omegaVar*m_xy_I - 24*omegaVar*m_yy_I + 6*omegaVar*m_zz_I + 23));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = (m_xy_t90/2 - (5*m_xx_t45)/6 + m_yy_t45/6 - m_zz_t45/12 + (5*omegaVar*m_xx_t45)/12 - (omegaVar*m_xy_t90)/4 - (omegaVar*m_yy_t45)/12 + (omegaVar*m_zz_t45)/24 - 5/18)*rho + (5*RHO_0)/18;
    L_BC_Fy = ((5*m_yy_t45)/6 - m_xy_t90/2 - m_xx_t45/6 + m_zz_t45/12 + (omegaVar*m_xx_t45)/12 + (omegaVar*m_xy_t90)/4 - (5*omegaVar*m_yy_t45)/12 - (omegaVar*m_zz_t45)/24 + 5/18)*rho - (5*RHO_0)/18;
    L_BC_Fz = (rho*(m_xz_t90 - m_yz_t90)*(omegaVar - 2))/2;
    #endif //_BC_FORCES 

    break;
case 0b0000010111011110:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[5] + pop[6] + pop[7] + pop[9] + pop[10] + pop[11] + pop[14] + pop[17];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[17]);
    m_xy_I = inv_rho_I * (pop[7] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[17]);
    m_yz_I = inv_rho_I * (pop[11] - pop[17]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[17]);

    rho = (18*(19*rho_I + 12*m_xx_I*rho_I - 33*m_xz_I*rho_I + 6*m_yy_I*rho_I + 12*m_zz_I*rho_I - 12*omegaVar*m_xx_I*rho_I + 33*omegaVar*m_xz_I*rho_I - 6*omegaVar*m_yy_I*rho_I - 12*omegaVar*m_zz_I*rho_I))/(59*omegaVar + 207);

    m_xx_t45 = -(2*(324*m_xz_I - 306*m_xx_I + 54*m_yy_I - 99*m_zz_I - 58*omegaVar*m_xx_I + 12*omegaVar*m_xz_I + 30*omegaVar*m_yy_I + omegaVar*m_zz_I - 36))/(27*(12*m_xx_I - 33*m_xz_I + 6*m_yy_I + 12*m_zz_I - 12*omegaVar*m_xx_I + 33*omegaVar*m_xz_I - 6*omegaVar*m_yy_I - 12*omegaVar*m_zz_I + 19));
    m_xy_t90 = (m_xy_I*(59*omegaVar + 207))/(9*(12*m_xx_I - 33*m_xz_I + 6*m_yy_I + 12*m_zz_I - 12*omegaVar*m_xx_I + 33*omegaVar*m_xz_I - 6*omegaVar*m_yy_I - 12*omegaVar*m_zz_I + 19));
    m_xz_t90 = -(216*m_xx_I - 801*m_xz_I - 99*m_yy_I + 216*m_zz_I + 8*omegaVar*m_xx_I - 81*omegaVar*m_xz_I - 55*omegaVar*m_yy_I + 8*omegaVar*m_zz_I + 66)/(18*(12*m_xx_I - 33*m_xz_I + 6*m_yy_I + 12*m_zz_I - 12*omegaVar*m_xx_I + 33*omegaVar*m_xz_I - 6*omegaVar*m_yy_I - 12*omegaVar*m_zz_I + 19));
    m_yy_t45 = (99*m_xz_I - 36*m_xx_I + 189*m_yy_I - 36*m_zz_I - 20*omegaVar*m_xx_I + 55*omegaVar*m_xz_I + 49*omegaVar*m_yy_I - 20*omegaVar*m_zz_I + 12)/(9*(12*m_xx_I - 33*m_xz_I + 6*m_yy_I + 12*m_zz_I - 12*omegaVar*m_xx_I + 33*omegaVar*m_xz_I - 6*omegaVar*m_yy_I - 12*omegaVar*m_zz_I + 19));
    m_yz_t90 = (m_yz_I*(59*omegaVar + 207))/(9*(12*m_xx_I - 33*m_xz_I + 6*m_yy_I + 12*m_zz_I - 12*omegaVar*m_xx_I + 33*omegaVar*m_xz_I - 6*omegaVar*m_yy_I - 12*omegaVar*m_zz_I + 19));
    m_zz_t45 = -(2*(324*m_xz_I - 99*m_xx_I + 54*m_yy_I - 306*m_zz_I + omegaVar*m_xx_I + 12*omegaVar*m_xz_I + 30*omegaVar*m_yy_I - 58*omegaVar*m_zz_I - 36))/(27*(12*m_xx_I - 33*m_xz_I + 6*m_yy_I + 12*m_zz_I - 12*omegaVar*m_xx_I + 33*omegaVar*m_xz_I - 6*omegaVar*m_yy_I - 12*omegaVar*m_zz_I + 19));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = -(m_xy_t90*rho*(omegaVar - 2))/2;
    L_BC_Fy = (m_yy_t45 - (omegaVar*m_yy_t45)/2 + 1.0/3.0)*rho - RHO_0/3;
    L_BC_Fz = -(m_yz_t90*rho*(omegaVar - 2))/2;
    #endif //_BC_FORCES 

    break;
case 0b0000010111011111:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[2] + pop[3] + pop[6] + pop[10] + pop[14] + pop[17];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[17]);
    m_xy_I = inv_rho_I * (-pop[14]);
    m_xz_I = inv_rho_I * (pop[10]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] - (1.0/3.0)*pop[6] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[17]);
    m_yz_I = inv_rho_I * (-pop[17]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] + (2.0/3.0)*pop[6] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[17]);

    rho = (12*(rho_I + m_xx_I*rho_I + 2*m_xy_I*rho_I - 2*m_xz_I*rho_I + m_yy_I*rho_I + 2*m_yz_I*rho_I + m_zz_I*rho_I - omegaVar*m_xx_I*rho_I - 2*omegaVar*m_xy_I*rho_I + 2*omegaVar*m_xz_I*rho_I - omegaVar*m_yy_I*rho_I - 2*omegaVar*m_yz_I*rho_I - omegaVar*m_zz_I*rho_I))/(5*omegaVar + 2);

    m_xx_t45 = (14*m_xx_I + 14*m_xy_I - 14*m_xz_I + 2*m_yy_I + 2*m_yz_I + 2*m_zz_I + 21*omegaVar*m_xx_I + 7*omegaVar*m_xy_I - 7*omegaVar*m_xz_I - 9*omegaVar*m_yy_I - 23*omegaVar*m_yz_I - 9*omegaVar*m_zz_I + 4)/(18*(m_xx_I + 2*m_xy_I - 2*m_xz_I + m_yy_I + 2*m_yz_I + m_zz_I - omegaVar*m_xx_I - 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - omegaVar*m_yy_I - 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 1));
    m_xy_t90 = (14*m_xx_I + 50*m_xy_I - 14*m_xz_I + 14*m_yy_I + 14*m_yz_I + 2*m_zz_I + 7*omegaVar*m_xx_I + 69*omegaVar*m_xy_I + 21*omegaVar*m_xz_I + 7*omegaVar*m_yy_I - 21*omegaVar*m_yz_I - 23*omegaVar*m_zz_I + 8)/(36*(m_xx_I + 2*m_xy_I - 2*m_xz_I + m_yy_I + 2*m_yz_I + m_zz_I - omegaVar*m_xx_I - 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - omegaVar*m_yy_I - 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 1));
    m_xz_t90 = -(14*m_xx_I + 14*m_xy_I - 50*m_xz_I + 2*m_yy_I + 14*m_yz_I + 14*m_zz_I + 7*omegaVar*m_xx_I - 21*omegaVar*m_xy_I - 69*omegaVar*m_xz_I - 23*omegaVar*m_yy_I - 21*omegaVar*m_yz_I + 7*omegaVar*m_zz_I + 8)/(36*(m_xx_I + 2*m_xy_I - 2*m_xz_I + m_yy_I + 2*m_yz_I + m_zz_I - omegaVar*m_xx_I - 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - omegaVar*m_yy_I - 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 1));
    m_yy_t45 = (2*m_xx_I + 14*m_xy_I - 2*m_xz_I + 14*m_yy_I + 14*m_yz_I + 2*m_zz_I - 9*omegaVar*m_xx_I + 7*omegaVar*m_xy_I + 23*omegaVar*m_xz_I + 21*omegaVar*m_yy_I + 7*omegaVar*m_yz_I - 9*omegaVar*m_zz_I + 4)/(18*(m_xx_I + 2*m_xy_I - 2*m_xz_I + m_yy_I + 2*m_yz_I + m_zz_I - omegaVar*m_xx_I - 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - omegaVar*m_yy_I - 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 1));
    m_yz_t90 = (2*m_xx_I + 14*m_xy_I - 14*m_xz_I + 14*m_yy_I + 50*m_yz_I + 14*m_zz_I - 23*omegaVar*m_xx_I - 21*omegaVar*m_xy_I + 21*omegaVar*m_xz_I + 7*omegaVar*m_yy_I + 69*omegaVar*m_yz_I + 7*omegaVar*m_zz_I + 8)/(36*(m_xx_I + 2*m_xy_I - 2*m_xz_I + m_yy_I + 2*m_yz_I + m_zz_I - omegaVar*m_xx_I - 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - omegaVar*m_yy_I - 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 1));
    m_zz_t45 = (2*m_xx_I + 2*m_xy_I - 14*m_xz_I + 2*m_yy_I + 14*m_yz_I + 14*m_zz_I - 9*omegaVar*m_xx_I - 23*omegaVar*m_xy_I - 7*omegaVar*m_xz_I - 9*omegaVar*m_yy_I + 7*omegaVar*m_yz_I + 21*omegaVar*m_zz_I + 4)/(18*(m_xx_I + 2*m_xy_I - 2*m_xz_I + m_yy_I + 2*m_yz_I + m_zz_I - omegaVar*m_xx_I - 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - omegaVar*m_yy_I - 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 1));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = (m_xy_t90/2 - (2*m_xx_t45)/3 - m_xz_t90/2 + m_yy_t45/12 + m_zz_t45/12 + (omegaVar*m_xx_t45)/3 - (omegaVar*m_xy_t90)/4 + (omegaVar*m_xz_t90)/4 - (omegaVar*m_yy_t45)/24 - (omegaVar*m_zz_t45)/24 - 2.0/9.0)*rho + (2*RHO_0)/9;
    L_BC_Fy = ((2*m_yy_t45)/3 - m_xy_t90/2 - m_xx_t45/12 - m_yz_t90/2 - m_zz_t45/12 + (omegaVar*m_xx_t45)/24 + (omegaVar*m_xy_t90)/4 - (omegaVar*m_yy_t45)/3 + (omegaVar*m_yz_t90)/4 + (omegaVar*m_zz_t45)/24 + 2.0/9.0)*rho - (2*RHO_0)/9;
    L_BC_Fz = (m_xx_t45/12 - m_xz_t90/2 + m_yy_t45/12 + m_yz_t90/2 - (2*m_zz_t45)/3 - (omegaVar*m_xx_t45)/24 + (omegaVar*m_xz_t90)/4 - (omegaVar*m_yy_t45)/24 - (omegaVar*m_yz_t90)/4 + (omegaVar*m_zz_t45)/3 - 2.0/9.0)*rho + (2*RHO_0)/9;
    #endif //_BC_FORCES 

    break;
case 0b0000010111100000:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[11] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] - (1.0/3.0)*pop[11] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[11] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (6*(15*rho_I + m_xx_I*rho_I + 6*m_xz_I*rho_I + m_yy_I*rho_I + 6*m_yz_I*rho_I + 4*m_zz_I*rho_I - omegaVar*m_xx_I*rho_I - 6*omegaVar*m_xz_I*rho_I - omegaVar*m_yy_I*rho_I - 6*omegaVar*m_yz_I*rho_I - 4*omegaVar*m_zz_I*rho_I))/(3*omegaVar + 82);

    m_xx_t45 = (2052*m_xx_I + 504*m_xz_I - 162*m_yy_I - 234*m_yz_I + 90*m_zz_I + 73*omegaVar*m_xx_I + 6*omegaVar*m_xz_I - 8*omegaVar*m_yy_I - 21*omegaVar*m_yz_I - 5*omegaVar*m_zz_I + 30)/(135*(m_xx_I + 6*m_xz_I + m_yy_I + 6*m_yz_I + 4*m_zz_I - omegaVar*m_xx_I - 6*omegaVar*m_xz_I - omegaVar*m_yy_I - 6*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 15));
    m_xy_t90 = (m_xy_I*(3*omegaVar + 82))/(6*(m_xx_I + 6*m_xz_I + m_yy_I + 6*m_yz_I + 4*m_zz_I - omegaVar*m_xx_I - 6*omegaVar*m_xz_I - omegaVar*m_yy_I - 6*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 15));
    m_xz_t90 = (168*m_xx_I + 1746*m_xz_I - 78*m_yy_I + 24*m_yz_I + 180*m_zz_I + 2*omegaVar*m_xx_I + 39*omegaVar*m_xz_I - 7*omegaVar*m_yy_I - 24*omegaVar*m_yz_I - 10*omegaVar*m_zz_I + 60)/(90*(m_xx_I + 6*m_xz_I + m_yy_I + 6*m_yz_I + 4*m_zz_I - omegaVar*m_xx_I - 6*omegaVar*m_xz_I - omegaVar*m_yy_I - 6*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 15));
    m_yy_t45 = (2052*m_yy_I - 234*m_xz_I - 162*m_xx_I + 504*m_yz_I + 90*m_zz_I - 8*omegaVar*m_xx_I - 21*omegaVar*m_xz_I + 73*omegaVar*m_yy_I + 6*omegaVar*m_yz_I - 5*omegaVar*m_zz_I + 30)/(135*(m_xx_I + 6*m_xz_I + m_yy_I + 6*m_yz_I + 4*m_zz_I - omegaVar*m_xx_I - 6*omegaVar*m_xz_I - omegaVar*m_yy_I - 6*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 15));
    m_yz_t90 = (24*m_xz_I - 78*m_xx_I + 168*m_yy_I + 1746*m_yz_I + 180*m_zz_I - 7*omegaVar*m_xx_I - 24*omegaVar*m_xz_I + 2*omegaVar*m_yy_I + 39*omegaVar*m_yz_I - 10*omegaVar*m_zz_I + 60)/(90*(m_xx_I + 6*m_xz_I + m_yy_I + 6*m_yz_I + 4*m_zz_I - omegaVar*m_xx_I - 6*omegaVar*m_xz_I - omegaVar*m_yy_I - 6*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 15));
    m_zz_t45 = (36*m_xx_I + 216*m_xz_I + 36*m_yy_I + 216*m_yz_I + 882*m_zz_I - 2*omegaVar*m_xx_I - 12*omegaVar*m_xz_I - 2*omegaVar*m_yy_I - 12*omegaVar*m_yz_I + 19*omegaVar*m_zz_I + 48)/(54*(m_xx_I + 6*m_xz_I + m_yy_I + 6*m_yz_I + 4*m_zz_I - omegaVar*m_xx_I - 6*omegaVar*m_xz_I - omegaVar*m_yy_I - 6*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 15));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = (m_xx_t45/6 + m_xz_t90/2 - m_yy_t45/12 + m_zz_t45/6 - (omegaVar*m_xx_t45)/12 - (omegaVar*m_xz_t90)/4 + (omegaVar*m_yy_t45)/24 - (omegaVar*m_zz_t45)/12 + 1.0/18.0)*rho - RHO_0/18;
    L_BC_Fy = (m_yy_t45/6 - m_xx_t45/12 + m_yz_t90/2 + m_zz_t45/6 + (omegaVar*m_xx_t45)/24 - (omegaVar*m_yy_t45)/12 - (omegaVar*m_yz_t90)/4 - (omegaVar*m_zz_t45)/12 + 1.0/18.0)*rho - RHO_0/18;
    L_BC_Fz = (m_xx_t45/12 + m_xz_t90/2 + m_yy_t45/12 + m_yz_t90/2 + m_zz_t45/3 - (omegaVar*m_xx_t45)/24 - (omegaVar*m_xz_t90)/4 - (omegaVar*m_yy_t45)/24 - (omegaVar*m_yz_t90)/4 - (omegaVar*m_zz_t45)/6 + 1.0/9.0)*rho - RHO_0/9;
    #endif //_BC_FORCES 

    break;
case 0b0000010111100001:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[11] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] - (1.0/3.0)*pop[11] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[11] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (6*(15*rho_I + m_xx_I*rho_I + 6*m_xz_I*rho_I + m_yy_I*rho_I + 6*m_yz_I*rho_I + 4*m_zz_I*rho_I - omegaVar*m_xx_I*rho_I - 6*omegaVar*m_xz_I*rho_I - omegaVar*m_yy_I*rho_I - 6*omegaVar*m_yz_I*rho_I - 4*omegaVar*m_zz_I*rho_I))/(3*omegaVar + 82);

    m_xx_t45 = (2052*m_xx_I + 504*m_xz_I - 162*m_yy_I - 234*m_yz_I + 90*m_zz_I + 73*omegaVar*m_xx_I + 6*omegaVar*m_xz_I - 8*omegaVar*m_yy_I - 21*omegaVar*m_yz_I - 5*omegaVar*m_zz_I + 30)/(135*(m_xx_I + 6*m_xz_I + m_yy_I + 6*m_yz_I + 4*m_zz_I - omegaVar*m_xx_I - 6*omegaVar*m_xz_I - omegaVar*m_yy_I - 6*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 15));
    m_xy_t90 = (m_xy_I*(3*omegaVar + 82))/(6*(m_xx_I + 6*m_xz_I + m_yy_I + 6*m_yz_I + 4*m_zz_I - omegaVar*m_xx_I - 6*omegaVar*m_xz_I - omegaVar*m_yy_I - 6*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 15));
    m_xz_t90 = (168*m_xx_I + 1746*m_xz_I - 78*m_yy_I + 24*m_yz_I + 180*m_zz_I + 2*omegaVar*m_xx_I + 39*omegaVar*m_xz_I - 7*omegaVar*m_yy_I - 24*omegaVar*m_yz_I - 10*omegaVar*m_zz_I + 60)/(90*(m_xx_I + 6*m_xz_I + m_yy_I + 6*m_yz_I + 4*m_zz_I - omegaVar*m_xx_I - 6*omegaVar*m_xz_I - omegaVar*m_yy_I - 6*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 15));
    m_yy_t45 = (2052*m_yy_I - 234*m_xz_I - 162*m_xx_I + 504*m_yz_I + 90*m_zz_I - 8*omegaVar*m_xx_I - 21*omegaVar*m_xz_I + 73*omegaVar*m_yy_I + 6*omegaVar*m_yz_I - 5*omegaVar*m_zz_I + 30)/(135*(m_xx_I + 6*m_xz_I + m_yy_I + 6*m_yz_I + 4*m_zz_I - omegaVar*m_xx_I - 6*omegaVar*m_xz_I - omegaVar*m_yy_I - 6*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 15));
    m_yz_t90 = (24*m_xz_I - 78*m_xx_I + 168*m_yy_I + 1746*m_yz_I + 180*m_zz_I - 7*omegaVar*m_xx_I - 24*omegaVar*m_xz_I + 2*omegaVar*m_yy_I + 39*omegaVar*m_yz_I - 10*omegaVar*m_zz_I + 60)/(90*(m_xx_I + 6*m_xz_I + m_yy_I + 6*m_yz_I + 4*m_zz_I - omegaVar*m_xx_I - 6*omegaVar*m_xz_I - omegaVar*m_yy_I - 6*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 15));
    m_zz_t45 = (36*m_xx_I + 216*m_xz_I + 36*m_yy_I + 216*m_yz_I + 882*m_zz_I - 2*omegaVar*m_xx_I - 12*omegaVar*m_xz_I - 2*omegaVar*m_yy_I - 12*omegaVar*m_yz_I + 19*omegaVar*m_zz_I + 48)/(54*(m_xx_I + 6*m_xz_I + m_yy_I + 6*m_yz_I + 4*m_zz_I - omegaVar*m_xx_I - 6*omegaVar*m_xz_I - omegaVar*m_yy_I - 6*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 15));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = (m_xx_t45/6 + m_xz_t90/2 - m_yy_t45/12 + m_zz_t45/6 - (omegaVar*m_xx_t45)/12 - (omegaVar*m_xz_t90)/4 + (omegaVar*m_yy_t45)/24 - (omegaVar*m_zz_t45)/12 + 1.0/18.0)*rho - RHO_0/18;
    L_BC_Fy = (m_yy_t45/6 - m_xx_t45/12 + m_yz_t90/2 + m_zz_t45/6 + (omegaVar*m_xx_t45)/24 - (omegaVar*m_yy_t45)/12 - (omegaVar*m_yz_t90)/4 - (omegaVar*m_zz_t45)/12 + 1.0/18.0)*rho - RHO_0/18;
    L_BC_Fz = (m_xx_t45/12 + m_xz_t90/2 + m_yy_t45/12 + m_yz_t90/2 + m_zz_t45/3 - (omegaVar*m_xx_t45)/24 - (omegaVar*m_xz_t90)/4 - (omegaVar*m_yy_t45)/24 - (omegaVar*m_yz_t90)/4 - (omegaVar*m_zz_t45)/6 + 1.0/9.0)*rho - RHO_0/9;
    #endif //_BC_FORCES 

    break;
case 0b0000010111100010:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[11] + pop[13] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] - (1.0/3.0)*pop[11] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13]);
    m_xz_I = inv_rho_I * (pop[9] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[13] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[11] - (1.0/3.0)*pop[13] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (12*(5*rho_I + m_xx_I*rho_I - 2*m_xy_I*rho_I + 2*m_xz_I*rho_I + m_yy_I*rho_I + 2*m_yz_I*rho_I + m_zz_I*rho_I - omegaVar*m_xx_I*rho_I + 2*omegaVar*m_xy_I*rho_I - 2*omegaVar*m_xz_I*rho_I - omegaVar*m_yy_I*rho_I - 2*omegaVar*m_yz_I*rho_I - omegaVar*m_zz_I*rho_I))/(3*omegaVar + 52);

    m_xx_t45 = (472*m_xx_I - 112*m_xy_I + 112*m_xz_I + 4*m_yy_I - 44*m_yz_I + 4*m_zz_I + 23*omegaVar*m_xx_I + 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - 4*omegaVar*m_yy_I - 11*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 20)/(90*(m_xx_I - 2*m_xy_I + 2*m_xz_I + m_yy_I + 2*m_yz_I + m_zz_I - omegaVar*m_xx_I + 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - omegaVar*m_yy_I - 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_xy_t90 = (1108*m_xy_I - 112*m_xx_I - 16*m_xz_I - 112*m_yy_I - 16*m_yz_I + 44*m_zz_I + 2*omegaVar*m_xx_I + 47*omegaVar*m_xy_I + 16*omegaVar*m_xz_I + 2*omegaVar*m_yy_I + 16*omegaVar*m_yz_I + 11*omegaVar*m_zz_I - 40)/(180*(m_xx_I - 2*m_xy_I + 2*m_xz_I + m_yy_I + 2*m_yz_I + m_zz_I - omegaVar*m_xx_I + 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - omegaVar*m_yy_I - 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_xz_t90 = (112*m_xx_I - 16*m_xy_I + 1108*m_xz_I - 44*m_yy_I + 16*m_yz_I + 112*m_zz_I - 2*omegaVar*m_xx_I + 16*omegaVar*m_xy_I + 47*omegaVar*m_xz_I - 11*omegaVar*m_yy_I - 16*omegaVar*m_yz_I - 2*omegaVar*m_zz_I + 40)/(180*(m_xx_I - 2*m_xy_I + 2*m_xz_I + m_yy_I + 2*m_yz_I + m_zz_I - omegaVar*m_xx_I + 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - omegaVar*m_yy_I - 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_yy_t45 = (4*m_xx_I - 112*m_xy_I - 44*m_xz_I + 472*m_yy_I + 112*m_yz_I + 4*m_zz_I - 4*omegaVar*m_xx_I + 2*omegaVar*m_xy_I - 11*omegaVar*m_xz_I + 23*omegaVar*m_yy_I - 2*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 20)/(90*(m_xx_I - 2*m_xy_I + 2*m_xz_I + m_yy_I + 2*m_yz_I + m_zz_I - omegaVar*m_xx_I + 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - omegaVar*m_yy_I - 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_yz_t90 = (16*m_xz_I - 16*m_xy_I - 44*m_xx_I + 112*m_yy_I + 1108*m_yz_I + 112*m_zz_I - 11*omegaVar*m_xx_I + 16*omegaVar*m_xy_I - 16*omegaVar*m_xz_I - 2*omegaVar*m_yy_I + 47*omegaVar*m_yz_I - 2*omegaVar*m_zz_I + 40)/(180*(m_xx_I - 2*m_xy_I + 2*m_xz_I + m_yy_I + 2*m_yz_I + m_zz_I - omegaVar*m_xx_I + 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - omegaVar*m_yy_I - 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_zz_t45 = (4*m_xx_I + 44*m_xy_I + 112*m_xz_I + 4*m_yy_I + 112*m_yz_I + 472*m_zz_I - 4*omegaVar*m_xx_I + 11*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - 4*omegaVar*m_yy_I - 2*omegaVar*m_yz_I + 23*omegaVar*m_zz_I + 20)/(90*(m_xx_I - 2*m_xy_I + 2*m_xz_I + m_yy_I + 2*m_yz_I + m_zz_I - omegaVar*m_xx_I + 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - omegaVar*m_yy_I - 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = (m_xx_t45/3 - m_xy_t90/2 + m_xz_t90/2 + m_yy_t45/12 + m_zz_t45/12 - (omegaVar*m_xx_t45)/6 + (omegaVar*m_xy_t90)/4 - (omegaVar*m_xz_t90)/4 - (omegaVar*m_yy_t45)/24 - (omegaVar*m_zz_t45)/24 + 1.0/9.0)*rho - RHO_0/9;
    L_BC_Fy = -(rho*(omegaVar - 2)*(2*m_xy_t90 - m_xx_t45 + 2*m_yz_t90 + m_zz_t45))/8;
    L_BC_Fz = (m_xx_t45/12 + m_xz_t90/2 + m_yy_t45/12 + m_yz_t90/2 + m_zz_t45/3 - (omegaVar*m_xx_t45)/24 - (omegaVar*m_xz_t90)/4 - (omegaVar*m_yy_t45)/24 - (omegaVar*m_yz_t90)/4 - (omegaVar*m_zz_t45)/6 + 1.0/9.0)*rho - RHO_0/9;
    #endif //_BC_FORCES 

    break;
case 0b0000010111100011:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[13] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13]);
    m_xz_I = inv_rho_I * (pop[9] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] + (2.0/3.0)*pop[13] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (-pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] - (1.0/3.0)*pop[13] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (3*(15*rho_I - m_xx_I*rho_I - 6*m_xy_I*rho_I + 6*m_xz_I*rho_I + 11*m_yy_I*rho_I + 30*m_yz_I*rho_I + 11*m_zz_I*rho_I + omegaVar*m_xx_I*rho_I + 6*omegaVar*m_xy_I*rho_I - 6*omegaVar*m_xz_I*rho_I - 11*omegaVar*m_yy_I*rho_I - 30*omegaVar*m_yz_I*rho_I - 11*omegaVar*m_zz_I*rho_I))/(9*omegaVar + 31);

    m_xx_t45 = -(2*(36*m_xy_I - 180*m_xx_I - 36*m_xz_I + 27*m_yy_I + 99*m_yz_I + 27*m_zz_I - 52*omegaVar*m_xx_I + 12*omegaVar*m_xy_I - 12*omegaVar*m_xz_I + 5*omegaVar*m_yy_I + 21*omegaVar*m_yz_I + 5*omegaVar*m_zz_I + 3))/(27*(6*m_xz_I - 6*m_xy_I - m_xx_I + 11*m_yy_I + 30*m_yz_I + 11*m_zz_I + omegaVar*m_xx_I + 6*omegaVar*m_xy_I - 6*omegaVar*m_xz_I - 11*omegaVar*m_yy_I - 30*omegaVar*m_yz_I - 11*omegaVar*m_zz_I + 15));
    m_xy_t90 = (663*m_xy_I - 60*m_xx_I - 12*m_xz_I - 84*m_yy_I - 60*m_yz_I + 9*m_zz_I - 20*omegaVar*m_xx_I + 177*omegaVar*m_xy_I + 12*omegaVar*m_xz_I + 4*omegaVar*m_yy_I + 60*omegaVar*m_yz_I + 31*omegaVar*m_zz_I - 30)/(45*(6*m_xz_I - 6*m_xy_I - m_xx_I + 11*m_yy_I + 30*m_yz_I + 11*m_zz_I + omegaVar*m_xx_I + 6*omegaVar*m_xy_I - 6*omegaVar*m_xz_I - 11*omegaVar*m_yy_I - 30*omegaVar*m_yz_I - 11*omegaVar*m_zz_I + 15));
    m_xz_t90 = (60*m_xx_I - 12*m_xy_I + 663*m_xz_I - 9*m_yy_I + 60*m_yz_I + 84*m_zz_I + 20*omegaVar*m_xx_I + 12*omegaVar*m_xy_I + 177*omegaVar*m_xz_I - 31*omegaVar*m_yy_I - 60*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 30)/(45*(6*m_xz_I - 6*m_xy_I - m_xx_I + 11*m_yy_I + 30*m_yz_I + 11*m_zz_I + omegaVar*m_xx_I + 6*omegaVar*m_xy_I - 6*omegaVar*m_xz_I - 11*omegaVar*m_yy_I - 30*omegaVar*m_yz_I - 11*omegaVar*m_zz_I + 15));
    m_yy_t45 = -(2*(135*m_xx_I + 252*m_xy_I + 27*m_xz_I - 1206*m_yy_I - 1260*m_yz_I - 369*m_zz_I + 25*omegaVar*m_xx_I - 12*omegaVar*m_xy_I + 93*omegaVar*m_xz_I - 194*omegaVar*m_yy_I + 60*omegaVar*m_yz_I + 49*omegaVar*m_zz_I - 165))/(135*(6*m_xz_I - 6*m_xy_I - m_xx_I + 11*m_yy_I + 30*m_yz_I + 11*m_zz_I + omegaVar*m_xx_I + 6*omegaVar*m_xy_I - 6*omegaVar*m_xz_I - 11*omegaVar*m_yy_I - 30*omegaVar*m_yz_I - 11*omegaVar*m_zz_I + 15));
    m_yz_t90 = (12*m_xz_I - 12*m_xy_I - 33*m_xx_I + 84*m_yy_I + 339*m_yz_I + 84*m_zz_I - 7*omegaVar*m_xx_I + 12*omegaVar*m_xy_I - 12*omegaVar*m_xz_I - 4*omegaVar*m_yy_I + 21*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 30)/(9*(6*m_xz_I - 6*m_xy_I - m_xx_I + 11*m_yy_I + 30*m_yz_I + 11*m_zz_I + omegaVar*m_xx_I + 6*omegaVar*m_xy_I - 6*omegaVar*m_xz_I - 11*omegaVar*m_yy_I - 30*omegaVar*m_yz_I - 11*omegaVar*m_zz_I + 15));
    m_zz_t45 = (2*(27*m_xy_I - 135*m_xx_I + 252*m_xz_I + 369*m_yy_I + 1260*m_yz_I + 1206*m_zz_I - 25*omegaVar*m_xx_I + 93*omegaVar*m_xy_I - 12*omegaVar*m_xz_I - 49*omegaVar*m_yy_I - 60*omegaVar*m_yz_I + 194*omegaVar*m_zz_I + 165))/(135*(6*m_xz_I - 6*m_xy_I - m_xx_I + 11*m_yy_I + 30*m_yz_I + 11*m_zz_I + omegaVar*m_xx_I + 6*omegaVar*m_xy_I - 6*omegaVar*m_xz_I - 11*omegaVar*m_yy_I - 30*omegaVar*m_yz_I - 11*omegaVar*m_zz_I + 15));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = (m_xx_t45/3 - m_xy_t90/2 + m_xz_t90/2 + m_yy_t45/12 + m_zz_t45/12 - (omegaVar*m_xx_t45)/6 + (omegaVar*m_xy_t90)/4 - (omegaVar*m_xz_t90)/4 - (omegaVar*m_yy_t45)/24 - (omegaVar*m_zz_t45)/24 + 1.0/9.0)*rho - RHO_0/9;
    L_BC_Fy = (m_xy_t90/2 - m_xx_t45/6 - m_yy_t45/6 + m_zz_t45/12 + (omegaVar*m_xx_t45)/12 - (omegaVar*m_xy_t90)/4 + (omegaVar*m_yy_t45)/12 - (omegaVar*m_zz_t45)/24 - 1.0/18.0)*rho + RHO_0/18;
    L_BC_Fz = (m_xx_t45/6 + m_xz_t90/2 - m_yy_t45/12 + m_zz_t45/6 - (omegaVar*m_xx_t45)/12 - (omegaVar*m_xz_t90)/4 + (omegaVar*m_yy_t45)/24 - (omegaVar*m_zz_t45)/12 + 1.0/18.0)*rho - RHO_0/18;
    #endif //_BC_FORCES 

    break;
case 0b0000010111100100:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[11] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] - (1.0/3.0)*pop[11] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[11] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (12*(5*rho_I + m_xx_I*rho_I - 2*m_xy_I*rho_I + 2*m_xz_I*rho_I + m_yy_I*rho_I + 2*m_yz_I*rho_I + m_zz_I*rho_I - omegaVar*m_xx_I*rho_I + 2*omegaVar*m_xy_I*rho_I - 2*omegaVar*m_xz_I*rho_I - omegaVar*m_yy_I*rho_I - 2*omegaVar*m_yz_I*rho_I - omegaVar*m_zz_I*rho_I))/(3*omegaVar + 52);

    m_xx_t45 = (472*m_xx_I - 112*m_xy_I + 112*m_xz_I + 4*m_yy_I - 44*m_yz_I + 4*m_zz_I + 23*omegaVar*m_xx_I + 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - 4*omegaVar*m_yy_I - 11*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 20)/(90*(m_xx_I - 2*m_xy_I + 2*m_xz_I + m_yy_I + 2*m_yz_I + m_zz_I - omegaVar*m_xx_I + 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - omegaVar*m_yy_I - 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_xy_t90 = (1108*m_xy_I - 112*m_xx_I - 16*m_xz_I - 112*m_yy_I - 16*m_yz_I + 44*m_zz_I + 2*omegaVar*m_xx_I + 47*omegaVar*m_xy_I + 16*omegaVar*m_xz_I + 2*omegaVar*m_yy_I + 16*omegaVar*m_yz_I + 11*omegaVar*m_zz_I - 40)/(180*(m_xx_I - 2*m_xy_I + 2*m_xz_I + m_yy_I + 2*m_yz_I + m_zz_I - omegaVar*m_xx_I + 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - omegaVar*m_yy_I - 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_xz_t90 = (112*m_xx_I - 16*m_xy_I + 1108*m_xz_I - 44*m_yy_I + 16*m_yz_I + 112*m_zz_I - 2*omegaVar*m_xx_I + 16*omegaVar*m_xy_I + 47*omegaVar*m_xz_I - 11*omegaVar*m_yy_I - 16*omegaVar*m_yz_I - 2*omegaVar*m_zz_I + 40)/(180*(m_xx_I - 2*m_xy_I + 2*m_xz_I + m_yy_I + 2*m_yz_I + m_zz_I - omegaVar*m_xx_I + 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - omegaVar*m_yy_I - 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_yy_t45 = (4*m_xx_I - 112*m_xy_I - 44*m_xz_I + 472*m_yy_I + 112*m_yz_I + 4*m_zz_I - 4*omegaVar*m_xx_I + 2*omegaVar*m_xy_I - 11*omegaVar*m_xz_I + 23*omegaVar*m_yy_I - 2*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 20)/(90*(m_xx_I - 2*m_xy_I + 2*m_xz_I + m_yy_I + 2*m_yz_I + m_zz_I - omegaVar*m_xx_I + 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - omegaVar*m_yy_I - 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_yz_t90 = (16*m_xz_I - 16*m_xy_I - 44*m_xx_I + 112*m_yy_I + 1108*m_yz_I + 112*m_zz_I - 11*omegaVar*m_xx_I + 16*omegaVar*m_xy_I - 16*omegaVar*m_xz_I - 2*omegaVar*m_yy_I + 47*omegaVar*m_yz_I - 2*omegaVar*m_zz_I + 40)/(180*(m_xx_I - 2*m_xy_I + 2*m_xz_I + m_yy_I + 2*m_yz_I + m_zz_I - omegaVar*m_xx_I + 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - omegaVar*m_yy_I - 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_zz_t45 = (4*m_xx_I + 44*m_xy_I + 112*m_xz_I + 4*m_yy_I + 112*m_yz_I + 472*m_zz_I - 4*omegaVar*m_xx_I + 11*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - 4*omegaVar*m_yy_I - 2*omegaVar*m_yz_I + 23*omegaVar*m_zz_I + 20)/(90*(m_xx_I - 2*m_xy_I + 2*m_xz_I + m_yy_I + 2*m_yz_I + m_zz_I - omegaVar*m_xx_I + 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - omegaVar*m_yy_I - 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = -(rho*(omegaVar - 2)*(2*m_xy_t90 + 2*m_xz_t90 - m_yy_t45 + m_zz_t45))/8;
    L_BC_Fy = (m_xx_t45/12 - m_xy_t90/2 + m_yy_t45/3 + m_yz_t90/2 + m_zz_t45/12 - (omegaVar*m_xx_t45)/24 + (omegaVar*m_xy_t90)/4 - (omegaVar*m_yy_t45)/6 - (omegaVar*m_yz_t90)/4 - (omegaVar*m_zz_t45)/24 + 1.0/9.0)*rho - RHO_0/9;
    L_BC_Fz = (m_xx_t45/12 + m_xz_t90/2 + m_yy_t45/12 + m_yz_t90/2 + m_zz_t45/3 - (omegaVar*m_xx_t45)/24 - (omegaVar*m_xz_t90)/4 - (omegaVar*m_yy_t45)/24 - (omegaVar*m_yz_t90)/4 - (omegaVar*m_zz_t45)/6 + 1.0/9.0)*rho - RHO_0/9;
    #endif //_BC_FORCES 

    break;
case 0b0000010111100101:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[11] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[11] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[14]);
    m_xz_I = inv_rho_I * (-pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[11] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (3*(15*rho_I + 11*m_xx_I*rho_I - 6*m_xy_I*rho_I + 30*m_xz_I*rho_I - m_yy_I*rho_I + 6*m_yz_I*rho_I + 11*m_zz_I*rho_I - 11*omegaVar*m_xx_I*rho_I + 6*omegaVar*m_xy_I*rho_I - 30*omegaVar*m_xz_I*rho_I + omegaVar*m_yy_I*rho_I - 6*omegaVar*m_yz_I*rho_I - 11*omegaVar*m_zz_I*rho_I))/(9*omegaVar + 31);

    m_xx_t45 = -(2*(252*m_xy_I - 1206*m_xx_I - 1260*m_xz_I + 135*m_yy_I + 27*m_yz_I - 369*m_zz_I - 194*omegaVar*m_xx_I - 12*omegaVar*m_xy_I + 60*omegaVar*m_xz_I + 25*omegaVar*m_yy_I + 93*omegaVar*m_yz_I + 49*omegaVar*m_zz_I - 165))/(135*(11*m_xx_I - 6*m_xy_I + 30*m_xz_I - m_yy_I + 6*m_yz_I + 11*m_zz_I - 11*omegaVar*m_xx_I + 6*omegaVar*m_xy_I - 30*omegaVar*m_xz_I + omegaVar*m_yy_I - 6*omegaVar*m_yz_I - 11*omegaVar*m_zz_I + 15));
    m_xy_t90 = (663*m_xy_I - 84*m_xx_I - 60*m_xz_I - 60*m_yy_I - 12*m_yz_I + 9*m_zz_I + 4*omegaVar*m_xx_I + 177*omegaVar*m_xy_I + 60*omegaVar*m_xz_I - 20*omegaVar*m_yy_I + 12*omegaVar*m_yz_I + 31*omegaVar*m_zz_I - 30)/(45*(11*m_xx_I - 6*m_xy_I + 30*m_xz_I - m_yy_I + 6*m_yz_I + 11*m_zz_I - 11*omegaVar*m_xx_I + 6*omegaVar*m_xy_I - 30*omegaVar*m_xz_I + omegaVar*m_yy_I - 6*omegaVar*m_yz_I - 11*omegaVar*m_zz_I + 15));
    m_xz_t90 = (84*m_xx_I - 12*m_xy_I + 339*m_xz_I - 33*m_yy_I + 12*m_yz_I + 84*m_zz_I - 4*omegaVar*m_xx_I + 12*omegaVar*m_xy_I + 21*omegaVar*m_xz_I - 7*omegaVar*m_yy_I - 12*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 30)/(9*(11*m_xx_I - 6*m_xy_I + 30*m_xz_I - m_yy_I + 6*m_yz_I + 11*m_zz_I - 11*omegaVar*m_xx_I + 6*omegaVar*m_xy_I - 30*omegaVar*m_xz_I + omegaVar*m_yy_I - 6*omegaVar*m_yz_I - 11*omegaVar*m_zz_I + 15));
    m_yy_t45 = -(2*(27*m_xx_I + 36*m_xy_I + 99*m_xz_I - 180*m_yy_I - 36*m_yz_I + 27*m_zz_I + 5*omegaVar*m_xx_I + 12*omegaVar*m_xy_I + 21*omegaVar*m_xz_I - 52*omegaVar*m_yy_I - 12*omegaVar*m_yz_I + 5*omegaVar*m_zz_I + 3))/(27*(11*m_xx_I - 6*m_xy_I + 30*m_xz_I - m_yy_I + 6*m_yz_I + 11*m_zz_I - 11*omegaVar*m_xx_I + 6*omegaVar*m_xy_I - 30*omegaVar*m_xz_I + omegaVar*m_yy_I - 6*omegaVar*m_yz_I - 11*omegaVar*m_zz_I + 15));
    m_yz_t90 = (60*m_xz_I - 12*m_xy_I - 9*m_xx_I + 60*m_yy_I + 663*m_yz_I + 84*m_zz_I - 31*omegaVar*m_xx_I + 12*omegaVar*m_xy_I - 60*omegaVar*m_xz_I + 20*omegaVar*m_yy_I + 177*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 30)/(45*(11*m_xx_I - 6*m_xy_I + 30*m_xz_I - m_yy_I + 6*m_yz_I + 11*m_zz_I - 11*omegaVar*m_xx_I + 6*omegaVar*m_xy_I - 30*omegaVar*m_xz_I + omegaVar*m_yy_I - 6*omegaVar*m_yz_I - 11*omegaVar*m_zz_I + 15));
    m_zz_t45 = (2*(369*m_xx_I + 27*m_xy_I + 1260*m_xz_I - 135*m_yy_I + 252*m_yz_I + 1206*m_zz_I - 49*omegaVar*m_xx_I + 93*omegaVar*m_xy_I - 60*omegaVar*m_xz_I - 25*omegaVar*m_yy_I - 12*omegaVar*m_yz_I + 194*omegaVar*m_zz_I + 165))/(135*(11*m_xx_I - 6*m_xy_I + 30*m_xz_I - m_yy_I + 6*m_yz_I + 11*m_zz_I - 11*omegaVar*m_xx_I + 6*omegaVar*m_xy_I - 30*omegaVar*m_xz_I + omegaVar*m_yy_I - 6*omegaVar*m_yz_I - 11*omegaVar*m_zz_I + 15));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = (m_xy_t90/2 - m_xx_t45/6 - m_yy_t45/6 + m_zz_t45/12 + (omegaVar*m_xx_t45)/12 - (omegaVar*m_xy_t90)/4 + (omegaVar*m_yy_t45)/12 - (omegaVar*m_zz_t45)/24 - 1.0/18.0)*rho + RHO_0/18;
    L_BC_Fy = (m_xx_t45/12 - m_xy_t90/2 + m_yy_t45/3 + m_yz_t90/2 + m_zz_t45/12 - (omegaVar*m_xx_t45)/24 + (omegaVar*m_xy_t90)/4 - (omegaVar*m_yy_t45)/6 - (omegaVar*m_yz_t90)/4 - (omegaVar*m_zz_t45)/24 + 1.0/9.0)*rho - RHO_0/9;
    L_BC_Fz = (m_yy_t45/6 - m_xx_t45/12 + m_yz_t90/2 + m_zz_t45/6 + (omegaVar*m_xx_t45)/24 - (omegaVar*m_yy_t45)/12 - (omegaVar*m_yz_t90)/4 - (omegaVar*m_zz_t45)/12 + 1.0/18.0)*rho - RHO_0/18;
    #endif //_BC_FORCES 

    break;
case 0b0000010111100110:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[11] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] - (1.0/3.0)*pop[11] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8]);
    m_xz_I = inv_rho_I * (pop[9] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] + (2.0/3.0)*pop[11] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (3*(15*rho_I + 11*m_xx_I*rho_I - 30*m_xy_I*rho_I + 6*m_xz_I*rho_I + 11*m_yy_I*rho_I + 6*m_yz_I*rho_I - m_zz_I*rho_I - 11*omegaVar*m_xx_I*rho_I + 30*omegaVar*m_xy_I*rho_I - 6*omegaVar*m_xz_I*rho_I - 11*omegaVar*m_yy_I*rho_I - 6*omegaVar*m_yz_I*rho_I + omegaVar*m_zz_I*rho_I))/(9*omegaVar + 31);

    m_xx_t45 = -(2*(1260*m_xy_I - 1206*m_xx_I - 252*m_xz_I - 369*m_yy_I + 27*m_yz_I + 135*m_zz_I - 194*omegaVar*m_xx_I - 60*omegaVar*m_xy_I + 12*omegaVar*m_xz_I + 49*omegaVar*m_yy_I + 93*omegaVar*m_yz_I + 25*omegaVar*m_zz_I - 165))/(135*(11*m_xx_I - 30*m_xy_I + 6*m_xz_I + 11*m_yy_I + 6*m_yz_I - m_zz_I - 11*omegaVar*m_xx_I + 30*omegaVar*m_xy_I - 6*omegaVar*m_xz_I - 11*omegaVar*m_yy_I - 6*omegaVar*m_yz_I + omegaVar*m_zz_I + 15));
    m_xy_t90 = (339*m_xy_I - 84*m_xx_I - 12*m_xz_I - 84*m_yy_I - 12*m_yz_I + 33*m_zz_I + 4*omegaVar*m_xx_I + 21*omegaVar*m_xy_I + 12*omegaVar*m_xz_I + 4*omegaVar*m_yy_I + 12*omegaVar*m_yz_I + 7*omegaVar*m_zz_I - 30)/(9*(11*m_xx_I - 30*m_xy_I + 6*m_xz_I + 11*m_yy_I + 6*m_yz_I - m_zz_I - 11*omegaVar*m_xx_I + 30*omegaVar*m_xy_I - 6*omegaVar*m_xz_I - 11*omegaVar*m_yy_I - 6*omegaVar*m_yz_I + omegaVar*m_zz_I + 15));
    m_xz_t90 = (84*m_xx_I - 60*m_xy_I + 663*m_xz_I - 9*m_yy_I + 12*m_yz_I + 60*m_zz_I - 4*omegaVar*m_xx_I + 60*omegaVar*m_xy_I + 177*omegaVar*m_xz_I - 31*omegaVar*m_yy_I - 12*omegaVar*m_yz_I + 20*omegaVar*m_zz_I + 30)/(45*(11*m_xx_I - 30*m_xy_I + 6*m_xz_I + 11*m_yy_I + 6*m_yz_I - m_zz_I - 11*omegaVar*m_xx_I + 30*omegaVar*m_xy_I - 6*omegaVar*m_xz_I - 11*omegaVar*m_yy_I - 6*omegaVar*m_yz_I + omegaVar*m_zz_I + 15));
    m_yy_t45 = -(2*(1260*m_xy_I - 369*m_xx_I + 27*m_xz_I - 1206*m_yy_I - 252*m_yz_I + 135*m_zz_I + 49*omegaVar*m_xx_I - 60*omegaVar*m_xy_I + 93*omegaVar*m_xz_I - 194*omegaVar*m_yy_I + 12*omegaVar*m_yz_I + 25*omegaVar*m_zz_I - 165))/(135*(11*m_xx_I - 30*m_xy_I + 6*m_xz_I + 11*m_yy_I + 6*m_yz_I - m_zz_I - 11*omegaVar*m_xx_I + 30*omegaVar*m_xy_I - 6*omegaVar*m_xz_I - 11*omegaVar*m_yy_I - 6*omegaVar*m_yz_I + omegaVar*m_zz_I + 15));
    m_yz_t90 = (12*m_xz_I - 60*m_xy_I - 9*m_xx_I + 84*m_yy_I + 663*m_yz_I + 60*m_zz_I - 31*omegaVar*m_xx_I + 60*omegaVar*m_xy_I - 12*omegaVar*m_xz_I - 4*omegaVar*m_yy_I + 177*omegaVar*m_yz_I + 20*omegaVar*m_zz_I + 30)/(45*(11*m_xx_I - 30*m_xy_I + 6*m_xz_I + 11*m_yy_I + 6*m_yz_I - m_zz_I - 11*omegaVar*m_xx_I + 30*omegaVar*m_xy_I - 6*omegaVar*m_xz_I - 11*omegaVar*m_yy_I - 6*omegaVar*m_yz_I + omegaVar*m_zz_I + 15));
    m_zz_t45 = (2*(99*m_xy_I - 27*m_xx_I + 36*m_xz_I - 27*m_yy_I + 36*m_yz_I + 180*m_zz_I - 5*omegaVar*m_xx_I + 21*omegaVar*m_xy_I + 12*omegaVar*m_xz_I - 5*omegaVar*m_yy_I + 12*omegaVar*m_yz_I + 52*omegaVar*m_zz_I - 3))/(27*(11*m_xx_I - 30*m_xy_I + 6*m_xz_I + 11*m_yy_I + 6*m_yz_I - m_zz_I - 11*omegaVar*m_xx_I + 30*omegaVar*m_xy_I - 6*omegaVar*m_xz_I - 11*omegaVar*m_yy_I - 6*omegaVar*m_yz_I + omegaVar*m_zz_I + 15));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = (m_xx_t45/6 + m_xz_t90/2 - m_yy_t45/12 + m_zz_t45/6 - (omegaVar*m_xx_t45)/12 - (omegaVar*m_xz_t90)/4 + (omegaVar*m_yy_t45)/24 - (omegaVar*m_zz_t45)/12 + 1.0/18.0)*rho - RHO_0/18;
    L_BC_Fy = (m_yy_t45/6 - m_xx_t45/12 + m_yz_t90/2 + m_zz_t45/6 + (omegaVar*m_xx_t45)/24 - (omegaVar*m_yy_t45)/12 - (omegaVar*m_yz_t90)/4 - (omegaVar*m_zz_t45)/12 + 1.0/18.0)*rho - RHO_0/18;
    L_BC_Fz = (m_xx_t45/12 + m_xz_t90/2 + m_yy_t45/12 + m_yz_t90/2 + m_zz_t45/3 - (omegaVar*m_xx_t45)/24 - (omegaVar*m_xz_t90)/4 - (omegaVar*m_yy_t45)/24 - (omegaVar*m_yz_t90)/4 - (omegaVar*m_zz_t45)/6 + 1.0/9.0)*rho - RHO_0/9;
    #endif //_BC_FORCES 

    break;
case 0b0000010111100111:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8]);
    m_xz_I = inv_rho_I * (-pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (-pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (6*(rho_I + m_xx_I*rho_I - 2*m_xy_I*rho_I + 2*m_xz_I*rho_I + m_yy_I*rho_I + 2*m_yz_I*rho_I + m_zz_I*rho_I - omegaVar*m_xx_I*rho_I + 2*omegaVar*m_xy_I*rho_I - 2*omegaVar*m_xz_I*rho_I - omegaVar*m_yy_I*rho_I - 2*omegaVar*m_yz_I*rho_I - omegaVar*m_zz_I*rho_I))/(3*omegaVar + 2);

    m_xx_t45 = (8*m_xx_I - 8*m_xy_I + 8*m_xz_I + 2*m_yy_I + 2*m_yz_I + 2*m_zz_I + 7*omegaVar*m_xx_I - 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - 2*omegaVar*m_yy_I - 7*omegaVar*m_yz_I - 2*omegaVar*m_zz_I + 2)/(9*(m_xx_I - 2*m_xy_I + 2*m_xz_I + m_yy_I + 2*m_yz_I + m_zz_I - omegaVar*m_xx_I + 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - omegaVar*m_yy_I - 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 1));
    m_xy_t90 = -(8*m_xx_I - 26*m_xy_I + 8*m_xz_I + 8*m_yy_I + 8*m_yz_I + 2*m_zz_I + 2*omegaVar*m_xx_I - 19*omegaVar*m_xy_I - 8*omegaVar*m_xz_I + 2*omegaVar*m_yy_I - 8*omegaVar*m_yz_I - 7*omegaVar*m_zz_I + 4)/(18*(m_xx_I - 2*m_xy_I + 2*m_xz_I + m_yy_I + 2*m_yz_I + m_zz_I - omegaVar*m_xx_I + 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - omegaVar*m_yy_I - 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 1));
    m_xz_t90 = (8*m_xx_I - 8*m_xy_I + 26*m_xz_I + 2*m_yy_I + 8*m_yz_I + 8*m_zz_I + 2*omegaVar*m_xx_I + 8*omegaVar*m_xy_I + 19*omegaVar*m_xz_I - 7*omegaVar*m_yy_I - 8*omegaVar*m_yz_I + 2*omegaVar*m_zz_I + 4)/(18*(m_xx_I - 2*m_xy_I + 2*m_xz_I + m_yy_I + 2*m_yz_I + m_zz_I - omegaVar*m_xx_I + 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - omegaVar*m_yy_I - 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 1));
    m_yy_t45 = (2*m_xx_I - 8*m_xy_I + 2*m_xz_I + 8*m_yy_I + 8*m_yz_I + 2*m_zz_I - 2*omegaVar*m_xx_I - 2*omegaVar*m_xy_I - 7*omegaVar*m_xz_I + 7*omegaVar*m_yy_I + 2*omegaVar*m_yz_I - 2*omegaVar*m_zz_I + 2)/(9*(m_xx_I - 2*m_xy_I + 2*m_xz_I + m_yy_I + 2*m_yz_I + m_zz_I - omegaVar*m_xx_I + 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - omegaVar*m_yy_I - 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 1));
    m_yz_t90 = (2*m_xx_I - 8*m_xy_I + 8*m_xz_I + 8*m_yy_I + 26*m_yz_I + 8*m_zz_I - 7*omegaVar*m_xx_I + 8*omegaVar*m_xy_I - 8*omegaVar*m_xz_I + 2*omegaVar*m_yy_I + 19*omegaVar*m_yz_I + 2*omegaVar*m_zz_I + 4)/(18*(m_xx_I - 2*m_xy_I + 2*m_xz_I + m_yy_I + 2*m_yz_I + m_zz_I - omegaVar*m_xx_I + 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - omegaVar*m_yy_I - 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 1));
    m_zz_t45 = (2*m_xx_I - 2*m_xy_I + 8*m_xz_I + 2*m_yy_I + 8*m_yz_I + 8*m_zz_I - 2*omegaVar*m_xx_I + 7*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - 2*omegaVar*m_yy_I + 2*omegaVar*m_yz_I + 7*omegaVar*m_zz_I + 2)/(9*(m_xx_I - 2*m_xy_I + 2*m_xz_I + m_yy_I + 2*m_yz_I + m_zz_I - omegaVar*m_xx_I + 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - omegaVar*m_yy_I - 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 1));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = 0;
    L_BC_Fy = 0;
    L_BC_Fz = 0;
    #endif //_BC_FORCES 

    break;
case 0b0000010111101000:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[9] + pop[11] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[9] - (1.0/3.0)*pop[11] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] - (1.0/3.0)*pop[9] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[11] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (12*(5*rho_I + m_xx_I*rho_I + 2*m_xy_I*rho_I + 2*m_xz_I*rho_I + m_yy_I*rho_I + 2*m_yz_I*rho_I + m_zz_I*rho_I - omegaVar*m_xx_I*rho_I - 2*omegaVar*m_xy_I*rho_I - 2*omegaVar*m_xz_I*rho_I - omegaVar*m_yy_I*rho_I - 2*omegaVar*m_yz_I*rho_I - omegaVar*m_zz_I*rho_I))/(3*omegaVar + 52);

    m_xx_t45 = (472*m_xx_I + 112*m_xy_I + 112*m_xz_I + 4*m_yy_I - 44*m_yz_I + 4*m_zz_I + 23*omegaVar*m_xx_I - 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - 4*omegaVar*m_yy_I - 11*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 20)/(90*(m_xx_I + 2*m_xy_I + 2*m_xz_I + m_yy_I + 2*m_yz_I + m_zz_I - omegaVar*m_xx_I - 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - omegaVar*m_yy_I - 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_xy_t90 = (112*m_xx_I + 1108*m_xy_I + 16*m_xz_I + 112*m_yy_I + 16*m_yz_I - 44*m_zz_I - 2*omegaVar*m_xx_I + 47*omegaVar*m_xy_I - 16*omegaVar*m_xz_I - 2*omegaVar*m_yy_I - 16*omegaVar*m_yz_I - 11*omegaVar*m_zz_I + 40)/(180*(m_xx_I + 2*m_xy_I + 2*m_xz_I + m_yy_I + 2*m_yz_I + m_zz_I - omegaVar*m_xx_I - 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - omegaVar*m_yy_I - 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_xz_t90 = (112*m_xx_I + 16*m_xy_I + 1108*m_xz_I - 44*m_yy_I + 16*m_yz_I + 112*m_zz_I - 2*omegaVar*m_xx_I - 16*omegaVar*m_xy_I + 47*omegaVar*m_xz_I - 11*omegaVar*m_yy_I - 16*omegaVar*m_yz_I - 2*omegaVar*m_zz_I + 40)/(180*(m_xx_I + 2*m_xy_I + 2*m_xz_I + m_yy_I + 2*m_yz_I + m_zz_I - omegaVar*m_xx_I - 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - omegaVar*m_yy_I - 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_yy_t45 = (4*m_xx_I + 112*m_xy_I - 44*m_xz_I + 472*m_yy_I + 112*m_yz_I + 4*m_zz_I - 4*omegaVar*m_xx_I - 2*omegaVar*m_xy_I - 11*omegaVar*m_xz_I + 23*omegaVar*m_yy_I - 2*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 20)/(90*(m_xx_I + 2*m_xy_I + 2*m_xz_I + m_yy_I + 2*m_yz_I + m_zz_I - omegaVar*m_xx_I - 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - omegaVar*m_yy_I - 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_yz_t90 = (16*m_xy_I - 44*m_xx_I + 16*m_xz_I + 112*m_yy_I + 1108*m_yz_I + 112*m_zz_I - 11*omegaVar*m_xx_I - 16*omegaVar*m_xy_I - 16*omegaVar*m_xz_I - 2*omegaVar*m_yy_I + 47*omegaVar*m_yz_I - 2*omegaVar*m_zz_I + 40)/(180*(m_xx_I + 2*m_xy_I + 2*m_xz_I + m_yy_I + 2*m_yz_I + m_zz_I - omegaVar*m_xx_I - 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - omegaVar*m_yy_I - 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_zz_t45 = (4*m_xx_I - 44*m_xy_I + 112*m_xz_I + 4*m_yy_I + 112*m_yz_I + 472*m_zz_I - 4*omegaVar*m_xx_I - 11*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - 4*omegaVar*m_yy_I - 2*omegaVar*m_yz_I + 23*omegaVar*m_zz_I + 20)/(90*(m_xx_I + 2*m_xy_I + 2*m_xz_I + m_yy_I + 2*m_yz_I + m_zz_I - omegaVar*m_xx_I - 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - omegaVar*m_yy_I - 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = (m_xx_t45/3 + m_xy_t90/2 + m_xz_t90/2 + m_yy_t45/12 + m_zz_t45/12 - (omegaVar*m_xx_t45)/6 - (omegaVar*m_xy_t90)/4 - (omegaVar*m_xz_t90)/4 - (omegaVar*m_yy_t45)/24 - (omegaVar*m_zz_t45)/24 + 1.0/9.0)*rho - RHO_0/9;
    L_BC_Fy = (m_xx_t45/12 + m_xy_t90/2 + m_yy_t45/3 + m_yz_t90/2 + m_zz_t45/12 - (omegaVar*m_xx_t45)/24 - (omegaVar*m_xy_t90)/4 - (omegaVar*m_yy_t45)/6 - (omegaVar*m_yz_t90)/4 - (omegaVar*m_zz_t45)/24 + 1.0/9.0)*rho - RHO_0/9;
    L_BC_Fz = (m_xx_t45/12 + m_xz_t90/2 + m_yy_t45/12 + m_yz_t90/2 + m_zz_t45/3 - (omegaVar*m_xx_t45)/24 - (omegaVar*m_xz_t90)/4 - (omegaVar*m_yy_t45)/24 - (omegaVar*m_yz_t90)/4 - (omegaVar*m_zz_t45)/6 + 1.0/9.0)*rho - RHO_0/9;
    #endif //_BC_FORCES 

    break;
case 0b0000010111101001:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[9] + pop[11] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[9] - (1.0/3.0)*pop[11] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] - (1.0/3.0)*pop[9] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[11] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (12*(5*rho_I + m_xx_I*rho_I + 2*m_xy_I*rho_I + 2*m_xz_I*rho_I + m_yy_I*rho_I + 2*m_yz_I*rho_I + m_zz_I*rho_I - omegaVar*m_xx_I*rho_I - 2*omegaVar*m_xy_I*rho_I - 2*omegaVar*m_xz_I*rho_I - omegaVar*m_yy_I*rho_I - 2*omegaVar*m_yz_I*rho_I - omegaVar*m_zz_I*rho_I))/(3*omegaVar + 52);

    m_xx_t45 = (472*m_xx_I + 112*m_xy_I + 112*m_xz_I + 4*m_yy_I - 44*m_yz_I + 4*m_zz_I + 23*omegaVar*m_xx_I - 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - 4*omegaVar*m_yy_I - 11*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 20)/(90*(m_xx_I + 2*m_xy_I + 2*m_xz_I + m_yy_I + 2*m_yz_I + m_zz_I - omegaVar*m_xx_I - 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - omegaVar*m_yy_I - 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_xy_t90 = (112*m_xx_I + 1108*m_xy_I + 16*m_xz_I + 112*m_yy_I + 16*m_yz_I - 44*m_zz_I - 2*omegaVar*m_xx_I + 47*omegaVar*m_xy_I - 16*omegaVar*m_xz_I - 2*omegaVar*m_yy_I - 16*omegaVar*m_yz_I - 11*omegaVar*m_zz_I + 40)/(180*(m_xx_I + 2*m_xy_I + 2*m_xz_I + m_yy_I + 2*m_yz_I + m_zz_I - omegaVar*m_xx_I - 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - omegaVar*m_yy_I - 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_xz_t90 = (112*m_xx_I + 16*m_xy_I + 1108*m_xz_I - 44*m_yy_I + 16*m_yz_I + 112*m_zz_I - 2*omegaVar*m_xx_I - 16*omegaVar*m_xy_I + 47*omegaVar*m_xz_I - 11*omegaVar*m_yy_I - 16*omegaVar*m_yz_I - 2*omegaVar*m_zz_I + 40)/(180*(m_xx_I + 2*m_xy_I + 2*m_xz_I + m_yy_I + 2*m_yz_I + m_zz_I - omegaVar*m_xx_I - 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - omegaVar*m_yy_I - 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_yy_t45 = (4*m_xx_I + 112*m_xy_I - 44*m_xz_I + 472*m_yy_I + 112*m_yz_I + 4*m_zz_I - 4*omegaVar*m_xx_I - 2*omegaVar*m_xy_I - 11*omegaVar*m_xz_I + 23*omegaVar*m_yy_I - 2*omegaVar*m_yz_I - 4*omegaVar*m_zz_I + 20)/(90*(m_xx_I + 2*m_xy_I + 2*m_xz_I + m_yy_I + 2*m_yz_I + m_zz_I - omegaVar*m_xx_I - 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - omegaVar*m_yy_I - 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_yz_t90 = (16*m_xy_I - 44*m_xx_I + 16*m_xz_I + 112*m_yy_I + 1108*m_yz_I + 112*m_zz_I - 11*omegaVar*m_xx_I - 16*omegaVar*m_xy_I - 16*omegaVar*m_xz_I - 2*omegaVar*m_yy_I + 47*omegaVar*m_yz_I - 2*omegaVar*m_zz_I + 40)/(180*(m_xx_I + 2*m_xy_I + 2*m_xz_I + m_yy_I + 2*m_yz_I + m_zz_I - omegaVar*m_xx_I - 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - omegaVar*m_yy_I - 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));
    m_zz_t45 = (4*m_xx_I - 44*m_xy_I + 112*m_xz_I + 4*m_yy_I + 112*m_yz_I + 472*m_zz_I - 4*omegaVar*m_xx_I - 11*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - 4*omegaVar*m_yy_I - 2*omegaVar*m_yz_I + 23*omegaVar*m_zz_I + 20)/(90*(m_xx_I + 2*m_xy_I + 2*m_xz_I + m_yy_I + 2*m_yz_I + m_zz_I - omegaVar*m_xx_I - 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - omegaVar*m_yy_I - 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 5));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = (m_xx_t45/3 + m_xy_t90/2 + m_xz_t90/2 + m_yy_t45/12 + m_zz_t45/12 - (omegaVar*m_xx_t45)/6 - (omegaVar*m_xy_t90)/4 - (omegaVar*m_xz_t90)/4 - (omegaVar*m_yy_t45)/24 - (omegaVar*m_zz_t45)/24 + 1.0/9.0)*rho - RHO_0/9;
    L_BC_Fy = (m_xx_t45/12 + m_xy_t90/2 + m_yy_t45/3 + m_yz_t90/2 + m_zz_t45/12 - (omegaVar*m_xx_t45)/24 - (omegaVar*m_xy_t90)/4 - (omegaVar*m_yy_t45)/6 - (omegaVar*m_yz_t90)/4 - (omegaVar*m_zz_t45)/24 + 1.0/9.0)*rho - RHO_0/9;
    L_BC_Fz = (m_xx_t45/12 + m_xz_t90/2 + m_yy_t45/12 + m_yz_t90/2 + m_zz_t45/3 - (omegaVar*m_xx_t45)/24 - (omegaVar*m_xz_t90)/4 - (omegaVar*m_yy_t45)/24 - (omegaVar*m_yz_t90)/4 - (omegaVar*m_zz_t45)/6 + 1.0/9.0)*rho - RHO_0/9;
    #endif //_BC_FORCES 

    break;
case 0b0000010111101010:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[9] + pop[11] + pop[13] + pop[15] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[9] - (1.0/3.0)*pop[11] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[15] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] - pop[13]);
    m_xz_I = inv_rho_I * (pop[9] - pop[15]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] - (1.0/3.0)*pop[9] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[13] - (1.0/3.0)*pop[15] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[11] - (1.0/3.0)*pop[13] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (36*(107*rho_I + 72*m_xx_I*rho_I + 12*m_yy_I*rho_I + 33*m_yz_I*rho_I + 12*m_zz_I*rho_I - 72*omegaVar*m_xx_I*rho_I - 12*omegaVar*m_yy_I*rho_I - 33*omegaVar*m_yz_I*rho_I - 12*omegaVar*m_zz_I*rho_I))/(313*omegaVar + 2790);

    m_xx_t45 = -(72*m_yy_I - 2358*m_xx_I + 198*m_yz_I + 72*m_zz_I - 49*omegaVar*m_xx_I + 44*omegaVar*m_yy_I + 121*omegaVar*m_yz_I + 44*omegaVar*m_zz_I - 288)/(18*(72*m_xx_I + 12*m_yy_I + 33*m_yz_I + 12*m_zz_I - 72*omegaVar*m_xx_I - 12*omegaVar*m_yy_I - 33*omegaVar*m_yz_I - 12*omegaVar*m_zz_I + 107));
    m_xy_t90 = (m_xy_I*(313*omegaVar + 2790))/(18*(72*m_xx_I + 12*m_yy_I + 33*m_yz_I + 12*m_zz_I - 72*omegaVar*m_xx_I - 12*omegaVar*m_yy_I - 33*omegaVar*m_yz_I - 12*omegaVar*m_zz_I + 107));
    m_xz_t90 = (m_xz_I*(313*omegaVar + 2790))/(18*(72*m_xx_I + 12*m_yy_I + 33*m_yz_I + 12*m_zz_I - 72*omegaVar*m_xx_I - 12*omegaVar*m_yy_I - 33*omegaVar*m_yz_I - 12*omegaVar*m_zz_I + 107));
    m_yy_t45 = (2772*m_yy_I - 108*m_xx_I + 648*m_yz_I - 18*m_zz_I - 66*omegaVar*m_xx_I + 302*omegaVar*m_yy_I + 48*omegaVar*m_yz_I - 11*omegaVar*m_zz_I + 72)/(27*(72*m_xx_I + 12*m_yy_I + 33*m_yz_I + 12*m_zz_I - 72*omegaVar*m_xx_I - 12*omegaVar*m_yy_I - 33*omegaVar*m_yz_I - 12*omegaVar*m_zz_I + 107));
    m_yz_t90 = (432*m_yy_I - 198*m_xx_I + 3978*m_yz_I + 432*m_zz_I - 121*omegaVar*m_xx_I + 32*omegaVar*m_yy_I + 401*omegaVar*m_yz_I + 32*omegaVar*m_zz_I + 132)/(36*(72*m_xx_I + 12*m_yy_I + 33*m_yz_I + 12*m_zz_I - 72*omegaVar*m_xx_I - 12*omegaVar*m_yy_I - 33*omegaVar*m_yz_I - 12*omegaVar*m_zz_I + 107));
    m_zz_t45 = (648*m_yz_I - 18*m_yy_I - 108*m_xx_I + 2772*m_zz_I - 66*omegaVar*m_xx_I - 11*omegaVar*m_yy_I + 48*omegaVar*m_yz_I + 302*omegaVar*m_zz_I + 72)/(27*(72*m_xx_I + 12*m_yy_I + 33*m_yz_I + 12*m_zz_I - 72*omegaVar*m_xx_I - 12*omegaVar*m_yy_I - 33*omegaVar*m_yz_I - 12*omegaVar*m_zz_I + 107));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = (m_xx_t45 - (omegaVar*m_xx_t45)/2 + 1.0/3.0)*rho - RHO_0/3;
    L_BC_Fy = (m_xy_t90 - m_xx_t45/12 + m_yy_t45/6 + m_yz_t90/2 + m_zz_t45/6 + (omegaVar*m_xx_t45)/24 - (omegaVar*m_xy_t90)/2 - (omegaVar*m_yy_t45)/12 - (omegaVar*m_yz_t90)/4 - (omegaVar*m_zz_t45)/12 + 1.0/18.0)*rho - RHO_0/18;
    L_BC_Fz = (m_xz_t90 - m_xx_t45/12 + m_yy_t45/6 + m_yz_t90/2 + m_zz_t45/6 + (omegaVar*m_xx_t45)/24 - (omegaVar*m_xz_t90)/2 - (omegaVar*m_yy_t45)/12 - (omegaVar*m_yz_t90)/4 - (omegaVar*m_zz_t45)/12 + 1.0/18.0)*rho - RHO_0/18;
    #endif //_BC_FORCES 

    break;
case 0b0000010111101011:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[9] + pop[13] + pop[15] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[15] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] - pop[13]);
    m_xz_I = inv_rho_I * (pop[9] - pop[15]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] - (1.0/3.0)*pop[9] + (2.0/3.0)*pop[13] - (1.0/3.0)*pop[15] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (-pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] + (2.0/3.0)*pop[9] - (1.0/3.0)*pop[13] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (18*(19*rho_I + 6*m_xx_I*rho_I + 12*m_yy_I*rho_I + 33*m_yz_I*rho_I + 12*m_zz_I*rho_I - 6*omegaVar*m_xx_I*rho_I - 12*omegaVar*m_yy_I*rho_I - 33*omegaVar*m_yz_I*rho_I - 12*omegaVar*m_zz_I*rho_I))/(59*omegaVar + 207);

    m_xx_t45 = -(36*m_yy_I - 189*m_xx_I + 99*m_yz_I + 36*m_zz_I - 49*omegaVar*m_xx_I + 20*omegaVar*m_yy_I + 55*omegaVar*m_yz_I + 20*omegaVar*m_zz_I - 12)/(9*(6*m_xx_I + 12*m_yy_I + 33*m_yz_I + 12*m_zz_I - 6*omegaVar*m_xx_I - 12*omegaVar*m_yy_I - 33*omegaVar*m_yz_I - 12*omegaVar*m_zz_I + 19));
    m_xy_t90 = (m_xy_I*(59*omegaVar + 207))/(9*(6*m_xx_I + 12*m_yy_I + 33*m_yz_I + 12*m_zz_I - 6*omegaVar*m_xx_I - 12*omegaVar*m_yy_I - 33*omegaVar*m_yz_I - 12*omegaVar*m_zz_I + 19));
    m_xz_t90 = (m_xz_I*(59*omegaVar + 207))/(9*(6*m_xx_I + 12*m_yy_I + 33*m_yz_I + 12*m_zz_I - 6*omegaVar*m_xx_I - 12*omegaVar*m_yy_I - 33*omegaVar*m_yz_I - 12*omegaVar*m_zz_I + 19));
    m_yy_t45 = (2*(306*m_yy_I - 54*m_xx_I + 324*m_yz_I + 99*m_zz_I - 30*omegaVar*m_xx_I + 58*omegaVar*m_yy_I + 12*omegaVar*m_yz_I - omegaVar*m_zz_I + 36))/(27*(6*m_xx_I + 12*m_yy_I + 33*m_yz_I + 12*m_zz_I - 6*omegaVar*m_xx_I - 12*omegaVar*m_yy_I - 33*omegaVar*m_yz_I - 12*omegaVar*m_zz_I + 19));
    m_yz_t90 = (216*m_yy_I - 99*m_xx_I + 801*m_yz_I + 216*m_zz_I - 55*omegaVar*m_xx_I + 8*omegaVar*m_yy_I + 81*omegaVar*m_yz_I + 8*omegaVar*m_zz_I + 66)/(18*(6*m_xx_I + 12*m_yy_I + 33*m_yz_I + 12*m_zz_I - 6*omegaVar*m_xx_I - 12*omegaVar*m_yy_I - 33*omegaVar*m_yz_I - 12*omegaVar*m_zz_I + 19));
    m_zz_t45 = (2*(99*m_yy_I - 54*m_xx_I + 324*m_yz_I + 306*m_zz_I - 30*omegaVar*m_xx_I - omegaVar*m_yy_I + 12*omegaVar*m_yz_I + 58*omegaVar*m_zz_I + 36))/(27*(6*m_xx_I + 12*m_yy_I + 33*m_yz_I + 12*m_zz_I - 6*omegaVar*m_xx_I - 12*omegaVar*m_yy_I - 33*omegaVar*m_yz_I - 12*omegaVar*m_zz_I + 19));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = (m_xx_t45 - (omegaVar*m_xx_t45)/2 + 1.0/3.0)*rho - RHO_0/3;
    L_BC_Fy = -(m_xy_t90*rho*(omegaVar - 2))/2;
    L_BC_Fz = -(m_xz_t90*rho*(omegaVar - 2))/2;
    #endif //_BC_FORCES 

    break;
case 0b0000010111101100:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[5] + pop[6] + pop[7] + pop[9] + pop[11] + pop[14] + pop[15] + pop[16] + pop[17];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[9] - (1.0/3.0)*pop[11] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17]);
    m_xy_I = inv_rho_I * (pop[7] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] - (1.0/3.0)*pop[9] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17]);
    m_yz_I = inv_rho_I * (pop[11] - pop[17]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[11] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17]);

    rho = (36*(107*rho_I + 12*m_xx_I*rho_I + 33*m_xz_I*rho_I + 72*m_yy_I*rho_I + 12*m_zz_I*rho_I - 12*omegaVar*m_xx_I*rho_I - 33*omegaVar*m_xz_I*rho_I - 72*omegaVar*m_yy_I*rho_I - 12*omegaVar*m_zz_I*rho_I))/(313*omegaVar + 2790);

    m_xx_t45 = (2772*m_xx_I + 648*m_xz_I - 108*m_yy_I - 18*m_zz_I + 302*omegaVar*m_xx_I + 48*omegaVar*m_xz_I - 66*omegaVar*m_yy_I - 11*omegaVar*m_zz_I + 72)/(27*(12*m_xx_I + 33*m_xz_I + 72*m_yy_I + 12*m_zz_I - 12*omegaVar*m_xx_I - 33*omegaVar*m_xz_I - 72*omegaVar*m_yy_I - 12*omegaVar*m_zz_I + 107));
    m_xy_t90 = (m_xy_I*(313*omegaVar + 2790))/(18*(12*m_xx_I + 33*m_xz_I + 72*m_yy_I + 12*m_zz_I - 12*omegaVar*m_xx_I - 33*omegaVar*m_xz_I - 72*omegaVar*m_yy_I - 12*omegaVar*m_zz_I + 107));
    m_xz_t90 = (432*m_xx_I + 3978*m_xz_I - 198*m_yy_I + 432*m_zz_I + 32*omegaVar*m_xx_I + 401*omegaVar*m_xz_I - 121*omegaVar*m_yy_I + 32*omegaVar*m_zz_I + 132)/(36*(12*m_xx_I + 33*m_xz_I + 72*m_yy_I + 12*m_zz_I - 12*omegaVar*m_xx_I - 33*omegaVar*m_xz_I - 72*omegaVar*m_yy_I - 12*omegaVar*m_zz_I + 107));
    m_yy_t45 = -(72*m_xx_I + 198*m_xz_I - 2358*m_yy_I + 72*m_zz_I + 44*omegaVar*m_xx_I + 121*omegaVar*m_xz_I - 49*omegaVar*m_yy_I + 44*omegaVar*m_zz_I - 288)/(18*(12*m_xx_I + 33*m_xz_I + 72*m_yy_I + 12*m_zz_I - 12*omegaVar*m_xx_I - 33*omegaVar*m_xz_I - 72*omegaVar*m_yy_I - 12*omegaVar*m_zz_I + 107));
    m_yz_t90 = (m_yz_I*(313*omegaVar + 2790))/(18*(12*m_xx_I + 33*m_xz_I + 72*m_yy_I + 12*m_zz_I - 12*omegaVar*m_xx_I - 33*omegaVar*m_xz_I - 72*omegaVar*m_yy_I - 12*omegaVar*m_zz_I + 107));
    m_zz_t45 = (648*m_xz_I - 18*m_xx_I - 108*m_yy_I + 2772*m_zz_I - 11*omegaVar*m_xx_I + 48*omegaVar*m_xz_I - 66*omegaVar*m_yy_I + 302*omegaVar*m_zz_I + 72)/(27*(12*m_xx_I + 33*m_xz_I + 72*m_yy_I + 12*m_zz_I - 12*omegaVar*m_xx_I - 33*omegaVar*m_xz_I - 72*omegaVar*m_yy_I - 12*omegaVar*m_zz_I + 107));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = (m_xx_t45/6 + m_xy_t90 + m_xz_t90/2 - m_yy_t45/12 + m_zz_t45/6 - (omegaVar*m_xx_t45)/12 - (omegaVar*m_xy_t90)/2 - (omegaVar*m_xz_t90)/4 + (omegaVar*m_yy_t45)/24 - (omegaVar*m_zz_t45)/12 + 1.0/18.0)*rho - RHO_0/18;
    L_BC_Fy = (m_yy_t45 - (omegaVar*m_yy_t45)/2 + 1.0/3.0)*rho - RHO_0/3;
    L_BC_Fz = (m_xx_t45/6 + m_xz_t90/2 - m_yy_t45/12 + m_yz_t90 + m_zz_t45/6 - (omegaVar*m_xx_t45)/12 - (omegaVar*m_xz_t90)/4 + (omegaVar*m_yy_t45)/24 - (omegaVar*m_yz_t90)/2 - (omegaVar*m_zz_t45)/12 + 1.0/18.0)*rho - RHO_0/18;
    #endif //_BC_FORCES 

    break;
case 0b0000010111101101:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[5] + pop[6] + pop[7] + pop[11] + pop[14] + pop[15] + pop[16] + pop[17];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] - (1.0/3.0)*pop[11] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17]);
    m_xy_I = inv_rho_I * (pop[7] - pop[14]);
    m_xz_I = inv_rho_I * (-pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17]);
    m_yz_I = inv_rho_I * (pop[11] - pop[17]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] + (2.0/3.0)*pop[11] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17]);

    rho = (18*(19*rho_I + 12*m_xx_I*rho_I + 33*m_xz_I*rho_I + 6*m_yy_I*rho_I + 12*m_zz_I*rho_I - 12*omegaVar*m_xx_I*rho_I - 33*omegaVar*m_xz_I*rho_I - 6*omegaVar*m_yy_I*rho_I - 12*omegaVar*m_zz_I*rho_I))/(59*omegaVar + 207);

    m_xx_t45 = (2*(306*m_xx_I + 324*m_xz_I - 54*m_yy_I + 99*m_zz_I + 58*omegaVar*m_xx_I + 12*omegaVar*m_xz_I - 30*omegaVar*m_yy_I - omegaVar*m_zz_I + 36))/(27*(12*m_xx_I + 33*m_xz_I + 6*m_yy_I + 12*m_zz_I - 12*omegaVar*m_xx_I - 33*omegaVar*m_xz_I - 6*omegaVar*m_yy_I - 12*omegaVar*m_zz_I + 19));
    m_xy_t90 = (m_xy_I*(59*omegaVar + 207))/(9*(12*m_xx_I + 33*m_xz_I + 6*m_yy_I + 12*m_zz_I - 12*omegaVar*m_xx_I - 33*omegaVar*m_xz_I - 6*omegaVar*m_yy_I - 12*omegaVar*m_zz_I + 19));
    m_xz_t90 = (216*m_xx_I + 801*m_xz_I - 99*m_yy_I + 216*m_zz_I + 8*omegaVar*m_xx_I + 81*omegaVar*m_xz_I - 55*omegaVar*m_yy_I + 8*omegaVar*m_zz_I + 66)/(18*(12*m_xx_I + 33*m_xz_I + 6*m_yy_I + 12*m_zz_I - 12*omegaVar*m_xx_I - 33*omegaVar*m_xz_I - 6*omegaVar*m_yy_I - 12*omegaVar*m_zz_I + 19));
    m_yy_t45 = -(36*m_xx_I + 99*m_xz_I - 189*m_yy_I + 36*m_zz_I + 20*omegaVar*m_xx_I + 55*omegaVar*m_xz_I - 49*omegaVar*m_yy_I + 20*omegaVar*m_zz_I - 12)/(9*(12*m_xx_I + 33*m_xz_I + 6*m_yy_I + 12*m_zz_I - 12*omegaVar*m_xx_I - 33*omegaVar*m_xz_I - 6*omegaVar*m_yy_I - 12*omegaVar*m_zz_I + 19));
    m_yz_t90 = (m_yz_I*(59*omegaVar + 207))/(9*(12*m_xx_I + 33*m_xz_I + 6*m_yy_I + 12*m_zz_I - 12*omegaVar*m_xx_I - 33*omegaVar*m_xz_I - 6*omegaVar*m_yy_I - 12*omegaVar*m_zz_I + 19));
    m_zz_t45 = (2*(99*m_xx_I + 324*m_xz_I - 54*m_yy_I + 306*m_zz_I - omegaVar*m_xx_I + 12*omegaVar*m_xz_I - 30*omegaVar*m_yy_I + 58*omegaVar*m_zz_I + 36))/(27*(12*m_xx_I + 33*m_xz_I + 6*m_yy_I + 12*m_zz_I - 12*omegaVar*m_xx_I - 33*omegaVar*m_xz_I - 6*omegaVar*m_yy_I - 12*omegaVar*m_zz_I + 19));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = -(m_xy_t90*rho*(omegaVar - 2))/2;
    L_BC_Fy = (m_yy_t45 - (omegaVar*m_yy_t45)/2 + 1.0/3.0)*rho - RHO_0/3;
    L_BC_Fz = -(m_yz_t90*rho*(omegaVar - 2))/2;
    #endif //_BC_FORCES 

    break;
case 0b0000010111101110:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[3] + pop[5] + pop[6] + pop[7] + pop[9] + pop[11] + pop[15] + pop[17];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[9] - (1.0/3.0)*pop[11] + (2.0/3.0)*pop[15] - (1.0/3.0)*pop[17]);
    m_xy_I = inv_rho_I * (pop[7]);
    m_xz_I = inv_rho_I * (pop[9] - pop[15]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] + (2.0/3.0)*pop[3] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] - (1.0/3.0)*pop[9] + (2.0/3.0)*pop[11] - (1.0/3.0)*pop[15] + (2.0/3.0)*pop[17]);
    m_yz_I = inv_rho_I * (pop[11] - pop[17]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[3] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[17]);

    rho = (36*(23*rho_I + 24*m_xx_I*rho_I - 57*m_xy_I*rho_I + 24*m_yy_I*rho_I - 6*m_zz_I*rho_I - 24*omegaVar*m_xx_I*rho_I + 57*omegaVar*m_xy_I*rho_I - 24*omegaVar*m_yy_I*rho_I + 6*omegaVar*m_zz_I*rho_I))/(5*(43*omegaVar + 72));

    m_xx_t45 = (936*m_xx_I - 1008*m_xy_I + 216*m_yy_I - 144*m_zz_I + 239*omegaVar*m_xx_I + 158*omegaVar*m_xy_I - 191*omegaVar*m_yy_I - 6*omegaVar*m_zz_I + 192)/(36*(24*m_xx_I - 57*m_xy_I + 24*m_yy_I - 6*m_zz_I - 24*omegaVar*m_xx_I + 57*omegaVar*m_xy_I - 24*omegaVar*m_yy_I + 6*omegaVar*m_zz_I + 23));
    m_xy_t90 = (2412*m_xy_I - 504*m_xx_I - 504*m_yy_I + 216*m_zz_I + 79*omegaVar*m_xx_I + 538*omegaVar*m_xy_I + 79*omegaVar*m_yy_I + 34*omegaVar*m_zz_I - 228)/(36*(24*m_xx_I - 57*m_xy_I + 24*m_yy_I - 6*m_zz_I - 24*omegaVar*m_xx_I + 57*omegaVar*m_xy_I - 24*omegaVar*m_yy_I + 6*omegaVar*m_zz_I + 23));
    m_xz_t90 = (5*m_xz_I*(43*omegaVar + 72))/(18*(24*m_xx_I - 57*m_xy_I + 24*m_yy_I - 6*m_zz_I - 24*omegaVar*m_xx_I + 57*omegaVar*m_xy_I - 24*omegaVar*m_yy_I + 6*omegaVar*m_zz_I + 23));
    m_yy_t45 = (216*m_xx_I - 1008*m_xy_I + 936*m_yy_I - 144*m_zz_I - 191*omegaVar*m_xx_I + 158*omegaVar*m_xy_I + 239*omegaVar*m_yy_I - 6*omegaVar*m_zz_I + 192)/(36*(24*m_xx_I - 57*m_xy_I + 24*m_yy_I - 6*m_zz_I - 24*omegaVar*m_xx_I + 57*omegaVar*m_xy_I - 24*omegaVar*m_yy_I + 6*omegaVar*m_zz_I + 23));
    m_yz_t90 = (5*m_yz_I*(43*omegaVar + 72))/(18*(24*m_xx_I - 57*m_xy_I + 24*m_yy_I - 6*m_zz_I - 24*omegaVar*m_xx_I + 57*omegaVar*m_xy_I - 24*omegaVar*m_yy_I + 6*omegaVar*m_zz_I + 23));
    m_zz_t45 = -(72*m_xx_I - 216*m_xy_I + 72*m_yy_I - 288*m_zz_I + 3*omegaVar*m_xx_I - 34*omegaVar*m_xy_I + 3*omegaVar*m_yy_I - 162*omegaVar*m_zz_I + 24)/(18*(24*m_xx_I - 57*m_xy_I + 24*m_yy_I - 6*m_zz_I - 24*omegaVar*m_xx_I + 57*omegaVar*m_xy_I - 24*omegaVar*m_yy_I + 6*omegaVar*m_zz_I + 23));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = ((5*m_xx_t45)/6 + m_xy_t90/2 - m_yy_t45/6 + m_zz_t45/12 - (5*omegaVar*m_xx_t45)/12 - (omegaVar*m_xy_t90)/4 + (omegaVar*m_yy_t45)/12 - (omegaVar*m_zz_t45)/24 + 5/18)*rho - (5*RHO_0)/18;
    L_BC_Fy = (m_xy_t90/2 - m_xx_t45/6 + (5*m_yy_t45)/6 + m_zz_t45/12 + (omegaVar*m_xx_t45)/12 - (omegaVar*m_xy_t90)/4 - (5*omegaVar*m_yy_t45)/12 - (omegaVar*m_zz_t45)/24 + 5/18)*rho - (5*RHO_0)/18;
    L_BC_Fz = -(rho*(m_xz_t90 + m_yz_t90)*(omegaVar - 2))/2;
    #endif //_BC_FORCES 

    break;
case 0b0000010111101111:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[3] + pop[6] + pop[7] + pop[15] + pop[17];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[15] - (1.0/3.0)*pop[17]);
    m_xy_I = inv_rho_I * (pop[7]);
    m_xz_I = inv_rho_I * (-pop[15]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] + (2.0/3.0)*pop[3] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] - (1.0/3.0)*pop[15] + (2.0/3.0)*pop[17]);
    m_yz_I = inv_rho_I * (-pop[17]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[3] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[17]);

    rho = (12*(rho_I + m_xx_I*rho_I - 2*m_xy_I*rho_I + 2*m_xz_I*rho_I + m_yy_I*rho_I + 2*m_yz_I*rho_I + m_zz_I*rho_I - omegaVar*m_xx_I*rho_I + 2*omegaVar*m_xy_I*rho_I - 2*omegaVar*m_xz_I*rho_I - omegaVar*m_yy_I*rho_I - 2*omegaVar*m_yz_I*rho_I - omegaVar*m_zz_I*rho_I))/(5*omegaVar + 2);

    m_xx_t45 = (14*m_xx_I - 14*m_xy_I + 14*m_xz_I + 2*m_yy_I + 2*m_yz_I + 2*m_zz_I + 21*omegaVar*m_xx_I - 7*omegaVar*m_xy_I + 7*omegaVar*m_xz_I - 9*omegaVar*m_yy_I - 23*omegaVar*m_yz_I - 9*omegaVar*m_zz_I + 4)/(18*(m_xx_I - 2*m_xy_I + 2*m_xz_I + m_yy_I + 2*m_yz_I + m_zz_I - omegaVar*m_xx_I + 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - omegaVar*m_yy_I - 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 1));
    m_xy_t90 = -(14*m_xx_I - 50*m_xy_I + 14*m_xz_I + 14*m_yy_I + 14*m_yz_I + 2*m_zz_I + 7*omegaVar*m_xx_I - 69*omegaVar*m_xy_I - 21*omegaVar*m_xz_I + 7*omegaVar*m_yy_I - 21*omegaVar*m_yz_I - 23*omegaVar*m_zz_I + 8)/(36*(m_xx_I - 2*m_xy_I + 2*m_xz_I + m_yy_I + 2*m_yz_I + m_zz_I - omegaVar*m_xx_I + 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - omegaVar*m_yy_I - 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 1));
    m_xz_t90 = (14*m_xx_I - 14*m_xy_I + 50*m_xz_I + 2*m_yy_I + 14*m_yz_I + 14*m_zz_I + 7*omegaVar*m_xx_I + 21*omegaVar*m_xy_I + 69*omegaVar*m_xz_I - 23*omegaVar*m_yy_I - 21*omegaVar*m_yz_I + 7*omegaVar*m_zz_I + 8)/(36*(m_xx_I - 2*m_xy_I + 2*m_xz_I + m_yy_I + 2*m_yz_I + m_zz_I - omegaVar*m_xx_I + 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - omegaVar*m_yy_I - 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 1));
    m_yy_t45 = (2*m_xx_I - 14*m_xy_I + 2*m_xz_I + 14*m_yy_I + 14*m_yz_I + 2*m_zz_I - 9*omegaVar*m_xx_I - 7*omegaVar*m_xy_I - 23*omegaVar*m_xz_I + 21*omegaVar*m_yy_I + 7*omegaVar*m_yz_I - 9*omegaVar*m_zz_I + 4)/(18*(m_xx_I - 2*m_xy_I + 2*m_xz_I + m_yy_I + 2*m_yz_I + m_zz_I - omegaVar*m_xx_I + 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - omegaVar*m_yy_I - 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 1));
    m_yz_t90 = (2*m_xx_I - 14*m_xy_I + 14*m_xz_I + 14*m_yy_I + 50*m_yz_I + 14*m_zz_I - 23*omegaVar*m_xx_I + 21*omegaVar*m_xy_I - 21*omegaVar*m_xz_I + 7*omegaVar*m_yy_I + 69*omegaVar*m_yz_I + 7*omegaVar*m_zz_I + 8)/(36*(m_xx_I - 2*m_xy_I + 2*m_xz_I + m_yy_I + 2*m_yz_I + m_zz_I - omegaVar*m_xx_I + 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - omegaVar*m_yy_I - 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 1));
    m_zz_t45 = (2*m_xx_I - 2*m_xy_I + 14*m_xz_I + 2*m_yy_I + 14*m_yz_I + 14*m_zz_I - 9*omegaVar*m_xx_I + 23*omegaVar*m_xy_I + 7*omegaVar*m_xz_I - 9*omegaVar*m_yy_I + 7*omegaVar*m_yz_I + 21*omegaVar*m_zz_I + 4)/(18*(m_xx_I - 2*m_xy_I + 2*m_xz_I + m_yy_I + 2*m_yz_I + m_zz_I - omegaVar*m_xx_I + 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - omegaVar*m_yy_I - 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 1));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = ((2*m_xx_t45)/3 + m_xy_t90/2 - m_xz_t90/2 - m_yy_t45/12 - m_zz_t45/12 - (omegaVar*m_xx_t45)/3 - (omegaVar*m_xy_t90)/4 + (omegaVar*m_xz_t90)/4 + (omegaVar*m_yy_t45)/24 + (omegaVar*m_zz_t45)/24 + 2.0/9.0)*rho - (2*RHO_0)/9;
    L_BC_Fy = (m_xy_t90/2 - m_xx_t45/12 + (2*m_yy_t45)/3 - m_yz_t90/2 - m_zz_t45/12 + (omegaVar*m_xx_t45)/24 - (omegaVar*m_xy_t90)/4 - (omegaVar*m_yy_t45)/3 + (omegaVar*m_yz_t90)/4 + (omegaVar*m_zz_t45)/24 + 2.0/9.0)*rho - (2*RHO_0)/9;
    L_BC_Fz = (m_xx_t45/12 + m_xz_t90/2 + m_yy_t45/12 + m_yz_t90/2 - (2*m_zz_t45)/3 - (omegaVar*m_xx_t45)/24 - (omegaVar*m_xz_t90)/4 - (omegaVar*m_yy_t45)/24 - (omegaVar*m_yz_t90)/4 + (omegaVar*m_zz_t45)/3 - 2.0/9.0)*rho + (2*RHO_0)/9;
    #endif //_BC_FORCES 

    break;
case 0b0000010111110000:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[7] + pop[8] + pop[9] + pop[11] + pop[13] + pop[14] + pop[16] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] - (1.0/3.0)*pop[11] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[11] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);

    rho = (3*rho_I*(3*m_zz_I - 3*omegaVar*m_zz_I + 4))/(omegaVar + 9);

    m_xx_t45 = (4*(omegaVar + 9)*(10*m_xx_I - m_yy_I))/(99*(3*m_zz_I - 3*omegaVar*m_zz_I + 4));
    m_xy_t90 = (m_xy_I*(omegaVar + 9))/(3*(3*m_zz_I - 3*omegaVar*m_zz_I + 4));
    m_xz_t90 = (2*m_xz_I*(omegaVar + 9))/(3*(3*m_zz_I - 3*omegaVar*m_zz_I + 4));
    m_yy_t45 = -(4*(m_xx_I - 10*m_yy_I)*(omegaVar + 9))/(99*(3*m_zz_I - 3*omegaVar*m_zz_I + 4));
    m_yz_t90 = (2*m_yz_I*(omegaVar + 9))/(3*(3*m_zz_I - 3*omegaVar*m_zz_I + 4));
    m_zz_t45 = (15*m_zz_I + 2)/(3*(3*m_zz_I - 3*omegaVar*m_zz_I + 4));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = -(m_xz_t90*rho*(omegaVar - 2))/2;
    L_BC_Fy = -(m_yz_t90*rho*(omegaVar - 2))/2;
    L_BC_Fz = (m_zz_t45 - (omegaVar*m_zz_t45)/2 + 1.0/3.0)*rho - RHO_0/3;
    #endif //_BC_FORCES 

    break;
case 0b0000010111110001:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[8] + pop[9] + pop[11] + pop[13] + pop[14] + pop[16] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] - (1.0/3.0)*pop[11] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[11] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);

    rho = (36*(107*rho_I + 12*m_xx_I*rho_I + 33*m_xy_I*rho_I + 12*m_yy_I*rho_I + 72*m_zz_I*rho_I - 12*omegaVar*m_xx_I*rho_I - 33*omegaVar*m_xy_I*rho_I - 12*omegaVar*m_yy_I*rho_I - 72*omegaVar*m_zz_I*rho_I))/(313*omegaVar + 2790);

    m_xx_t45 = (2772*m_xx_I + 648*m_xy_I - 18*m_yy_I - 108*m_zz_I + 302*omegaVar*m_xx_I + 48*omegaVar*m_xy_I - 11*omegaVar*m_yy_I - 66*omegaVar*m_zz_I + 72)/(27*(12*m_xx_I + 33*m_xy_I + 12*m_yy_I + 72*m_zz_I - 12*omegaVar*m_xx_I - 33*omegaVar*m_xy_I - 12*omegaVar*m_yy_I - 72*omegaVar*m_zz_I + 107));
    m_xy_t90 = (432*m_xx_I + 3978*m_xy_I + 432*m_yy_I - 198*m_zz_I + 32*omegaVar*m_xx_I + 401*omegaVar*m_xy_I + 32*omegaVar*m_yy_I - 121*omegaVar*m_zz_I + 132)/(36*(12*m_xx_I + 33*m_xy_I + 12*m_yy_I + 72*m_zz_I - 12*omegaVar*m_xx_I - 33*omegaVar*m_xy_I - 12*omegaVar*m_yy_I - 72*omegaVar*m_zz_I + 107));
    m_xz_t90 = (m_xz_I*(313*omegaVar + 2790))/(18*(12*m_xx_I + 33*m_xy_I + 12*m_yy_I + 72*m_zz_I - 12*omegaVar*m_xx_I - 33*omegaVar*m_xy_I - 12*omegaVar*m_yy_I - 72*omegaVar*m_zz_I + 107));
    m_yy_t45 = (648*m_xy_I - 18*m_xx_I + 2772*m_yy_I - 108*m_zz_I - 11*omegaVar*m_xx_I + 48*omegaVar*m_xy_I + 302*omegaVar*m_yy_I - 66*omegaVar*m_zz_I + 72)/(27*(12*m_xx_I + 33*m_xy_I + 12*m_yy_I + 72*m_zz_I - 12*omegaVar*m_xx_I - 33*omegaVar*m_xy_I - 12*omegaVar*m_yy_I - 72*omegaVar*m_zz_I + 107));
    m_yz_t90 = (m_yz_I*(313*omegaVar + 2790))/(18*(12*m_xx_I + 33*m_xy_I + 12*m_yy_I + 72*m_zz_I - 12*omegaVar*m_xx_I - 33*omegaVar*m_xy_I - 12*omegaVar*m_yy_I - 72*omegaVar*m_zz_I + 107));
    m_zz_t45 = -(72*m_xx_I + 198*m_xy_I + 72*m_yy_I - 2358*m_zz_I + 44*omegaVar*m_xx_I + 121*omegaVar*m_xy_I + 44*omegaVar*m_yy_I - 49*omegaVar*m_zz_I - 288)/(18*(12*m_xx_I + 33*m_xy_I + 12*m_yy_I + 72*m_zz_I - 12*omegaVar*m_xx_I - 33*omegaVar*m_xy_I - 12*omegaVar*m_yy_I - 72*omegaVar*m_zz_I + 107));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = (m_xz_t90 - m_xy_t90/2 - m_xx_t45/6 - m_yy_t45/6 + m_zz_t45/12 + (omegaVar*m_xx_t45)/12 + (omegaVar*m_xy_t90)/4 - (omegaVar*m_xz_t90)/2 + (omegaVar*m_yy_t45)/12 - (omegaVar*m_zz_t45)/24 - 1.0/18.0)*rho + RHO_0/18;
    L_BC_Fy = (m_yz_t90 - m_xy_t90/2 - m_yy_t45/6 - m_xx_t45/6 + m_zz_t45/12 + (omegaVar*m_xx_t45)/12 + (omegaVar*m_xy_t90)/4 + (omegaVar*m_yy_t45)/12 - (omegaVar*m_yz_t90)/2 - (omegaVar*m_zz_t45)/24 - 1.0/18.0)*rho + RHO_0/18;
    L_BC_Fz = (m_zz_t45 - (omegaVar*m_zz_t45)/2 + 1.0/3.0)*rho - RHO_0/3;
    #endif //_BC_FORCES 

    break;
case 0b0000010111110010:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[7] + pop[8] + pop[9] + pop[11] + pop[13] + pop[16] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] - (1.0/3.0)*pop[11] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13]);
    m_xz_I = inv_rho_I * (pop[9] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[13] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[11] - (1.0/3.0)*pop[13] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);

    rho = (36*(107*rho_I + 12*m_xx_I*rho_I - 33*m_xy_I*rho_I + 12*m_yy_I*rho_I + 72*m_zz_I*rho_I - 12*omegaVar*m_xx_I*rho_I + 33*omegaVar*m_xy_I*rho_I - 12*omegaVar*m_yy_I*rho_I - 72*omegaVar*m_zz_I*rho_I))/(313*omegaVar + 2790);

    m_xx_t45 = -(648*m_xy_I - 2772*m_xx_I + 18*m_yy_I + 108*m_zz_I - 302*omegaVar*m_xx_I + 48*omegaVar*m_xy_I + 11*omegaVar*m_yy_I + 66*omegaVar*m_zz_I - 72)/(27*(12*m_xx_I - 33*m_xy_I + 12*m_yy_I + 72*m_zz_I - 12*omegaVar*m_xx_I + 33*omegaVar*m_xy_I - 12*omegaVar*m_yy_I - 72*omegaVar*m_zz_I + 107));
    m_xy_t90 = -(432*m_xx_I - 3978*m_xy_I + 432*m_yy_I - 198*m_zz_I + 32*omegaVar*m_xx_I - 401*omegaVar*m_xy_I + 32*omegaVar*m_yy_I - 121*omegaVar*m_zz_I + 132)/(36*(12*m_xx_I - 33*m_xy_I + 12*m_yy_I + 72*m_zz_I - 12*omegaVar*m_xx_I + 33*omegaVar*m_xy_I - 12*omegaVar*m_yy_I - 72*omegaVar*m_zz_I + 107));
    m_xz_t90 = (m_xz_I*(313*omegaVar + 2790))/(18*(12*m_xx_I - 33*m_xy_I + 12*m_yy_I + 72*m_zz_I - 12*omegaVar*m_xx_I + 33*omegaVar*m_xy_I - 12*omegaVar*m_yy_I - 72*omegaVar*m_zz_I + 107));
    m_yy_t45 = -(18*m_xx_I + 648*m_xy_I - 2772*m_yy_I + 108*m_zz_I + 11*omegaVar*m_xx_I + 48*omegaVar*m_xy_I - 302*omegaVar*m_yy_I + 66*omegaVar*m_zz_I - 72)/(27*(12*m_xx_I - 33*m_xy_I + 12*m_yy_I + 72*m_zz_I - 12*omegaVar*m_xx_I + 33*omegaVar*m_xy_I - 12*omegaVar*m_yy_I - 72*omegaVar*m_zz_I + 107));
    m_yz_t90 = (m_yz_I*(313*omegaVar + 2790))/(18*(12*m_xx_I - 33*m_xy_I + 12*m_yy_I + 72*m_zz_I - 12*omegaVar*m_xx_I + 33*omegaVar*m_xy_I - 12*omegaVar*m_yy_I - 72*omegaVar*m_zz_I + 107));
    m_zz_t45 = (198*m_xy_I - 72*m_xx_I - 72*m_yy_I + 2358*m_zz_I - 44*omegaVar*m_xx_I + 121*omegaVar*m_xy_I - 44*omegaVar*m_yy_I + 49*omegaVar*m_zz_I + 288)/(18*(12*m_xx_I - 33*m_xy_I + 12*m_yy_I + 72*m_zz_I - 12*omegaVar*m_xx_I + 33*omegaVar*m_xy_I - 12*omegaVar*m_yy_I - 72*omegaVar*m_zz_I + 107));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = (m_xx_t45/6 - m_xy_t90/2 + m_xz_t90 + m_yy_t45/6 - m_zz_t45/12 - (omegaVar*m_xx_t45)/12 + (omegaVar*m_xy_t90)/4 - (omegaVar*m_xz_t90)/2 - (omegaVar*m_yy_t45)/12 + (omegaVar*m_zz_t45)/24 + 1.0/18.0)*rho - RHO_0/18;
    L_BC_Fy = (m_xy_t90/2 - m_xx_t45/6 - m_yy_t45/6 + m_yz_t90 + m_zz_t45/12 + (omegaVar*m_xx_t45)/12 - (omegaVar*m_xy_t90)/4 + (omegaVar*m_yy_t45)/12 - (omegaVar*m_yz_t90)/2 - (omegaVar*m_zz_t45)/24 - 1.0/18.0)*rho + RHO_0/18;
    L_BC_Fz = (m_zz_t45 - (omegaVar*m_zz_t45)/2 + 1.0/3.0)*rho - RHO_0/3;
    #endif //_BC_FORCES 

    break;
case 0b0000010111110011:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[4] + pop[5] + pop[8] + pop[9] + pop[13] + pop[16] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[8] - pop[13]);
    m_xz_I = inv_rho_I * (pop[9] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] + (2.0/3.0)*pop[13] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (-pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] - (1.0/3.0)*pop[13] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);

    rho = (36*(23*rho_I - 6*m_xx_I*rho_I + 24*m_yy_I*rho_I + 57*m_yz_I*rho_I + 24*m_zz_I*rho_I + 6*omegaVar*m_xx_I*rho_I - 24*omegaVar*m_yy_I*rho_I - 57*omegaVar*m_yz_I*rho_I - 24*omegaVar*m_zz_I*rho_I))/(5*(43*omegaVar + 72));

    m_xx_t45 = -(72*m_yy_I - 288*m_xx_I + 216*m_yz_I + 72*m_zz_I - 162*omegaVar*m_xx_I + 3*omegaVar*m_yy_I + 34*omegaVar*m_yz_I + 3*omegaVar*m_zz_I + 24)/(18*(24*m_yy_I - 6*m_xx_I + 57*m_yz_I + 24*m_zz_I + 6*omegaVar*m_xx_I - 24*omegaVar*m_yy_I - 57*omegaVar*m_yz_I - 24*omegaVar*m_zz_I + 23));
    m_xy_t90 = (5*m_xy_I*(43*omegaVar + 72))/(18*(24*m_yy_I - 6*m_xx_I + 57*m_yz_I + 24*m_zz_I + 6*omegaVar*m_xx_I - 24*omegaVar*m_yy_I - 57*omegaVar*m_yz_I - 24*omegaVar*m_zz_I + 23));
    m_xz_t90 = (5*m_xz_I*(43*omegaVar + 72))/(18*(24*m_yy_I - 6*m_xx_I + 57*m_yz_I + 24*m_zz_I + 6*omegaVar*m_xx_I - 24*omegaVar*m_yy_I - 57*omegaVar*m_yz_I - 24*omegaVar*m_zz_I + 23));
    m_yy_t45 = (936*m_yy_I - 144*m_xx_I + 1008*m_yz_I + 216*m_zz_I - 6*omegaVar*m_xx_I + 239*omegaVar*m_yy_I - 158*omegaVar*m_yz_I - 191*omegaVar*m_zz_I + 192)/(36*(24*m_yy_I - 6*m_xx_I + 57*m_yz_I + 24*m_zz_I + 6*omegaVar*m_xx_I - 24*omegaVar*m_yy_I - 57*omegaVar*m_yz_I - 24*omegaVar*m_zz_I + 23));
    m_yz_t90 = (504*m_yy_I - 216*m_xx_I + 2412*m_yz_I + 504*m_zz_I - 34*omegaVar*m_xx_I - 79*omegaVar*m_yy_I + 538*omegaVar*m_yz_I - 79*omegaVar*m_zz_I + 228)/(36*(24*m_yy_I - 6*m_xx_I + 57*m_yz_I + 24*m_zz_I + 6*omegaVar*m_xx_I - 24*omegaVar*m_yy_I - 57*omegaVar*m_yz_I - 24*omegaVar*m_zz_I + 23));
    m_zz_t45 = (216*m_yy_I - 144*m_xx_I + 1008*m_yz_I + 936*m_zz_I - 6*omegaVar*m_xx_I - 191*omegaVar*m_yy_I - 158*omegaVar*m_yz_I + 239*omegaVar*m_zz_I + 192)/(36*(24*m_yy_I - 6*m_xx_I + 57*m_yz_I + 24*m_zz_I + 6*omegaVar*m_xx_I - 24*omegaVar*m_yy_I - 57*omegaVar*m_yz_I - 24*omegaVar*m_zz_I + 23));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = (rho*(m_xy_t90 - m_xz_t90)*(omegaVar - 2))/2;
    L_BC_Fy = (m_yz_t90/2 - (5*m_yy_t45)/6 - m_xx_t45/12 + m_zz_t45/6 + (omegaVar*m_xx_t45)/24 + (5*omegaVar*m_yy_t45)/12 - (omegaVar*m_yz_t90)/4 - (omegaVar*m_zz_t45)/12 - 5/18)*rho + (5*RHO_0)/18;
    L_BC_Fz = (m_xx_t45/12 - m_yy_t45/6 - m_yz_t90/2 + (5*m_zz_t45)/6 - (omegaVar*m_xx_t45)/24 + (omegaVar*m_yy_t45)/12 + (omegaVar*m_yz_t90)/4 - (5*omegaVar*m_zz_t45)/12 + 5/18)*rho - (5*RHO_0)/18;
    #endif //_BC_FORCES 

    break;
case 0b0000010111110100:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[7] + pop[8] + pop[9] + pop[11] + pop[14] + pop[16] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] - (1.0/3.0)*pop[11] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[11] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);

    rho = (36*(107*rho_I + 12*m_xx_I*rho_I - 33*m_xy_I*rho_I + 12*m_yy_I*rho_I + 72*m_zz_I*rho_I - 12*omegaVar*m_xx_I*rho_I + 33*omegaVar*m_xy_I*rho_I - 12*omegaVar*m_yy_I*rho_I - 72*omegaVar*m_zz_I*rho_I))/(313*omegaVar + 2790);

    m_xx_t45 = -(648*m_xy_I - 2772*m_xx_I + 18*m_yy_I + 108*m_zz_I - 302*omegaVar*m_xx_I + 48*omegaVar*m_xy_I + 11*omegaVar*m_yy_I + 66*omegaVar*m_zz_I - 72)/(27*(12*m_xx_I - 33*m_xy_I + 12*m_yy_I + 72*m_zz_I - 12*omegaVar*m_xx_I + 33*omegaVar*m_xy_I - 12*omegaVar*m_yy_I - 72*omegaVar*m_zz_I + 107));
    m_xy_t90 = -(432*m_xx_I - 3978*m_xy_I + 432*m_yy_I - 198*m_zz_I + 32*omegaVar*m_xx_I - 401*omegaVar*m_xy_I + 32*omegaVar*m_yy_I - 121*omegaVar*m_zz_I + 132)/(36*(12*m_xx_I - 33*m_xy_I + 12*m_yy_I + 72*m_zz_I - 12*omegaVar*m_xx_I + 33*omegaVar*m_xy_I - 12*omegaVar*m_yy_I - 72*omegaVar*m_zz_I + 107));
    m_xz_t90 = (m_xz_I*(313*omegaVar + 2790))/(18*(12*m_xx_I - 33*m_xy_I + 12*m_yy_I + 72*m_zz_I - 12*omegaVar*m_xx_I + 33*omegaVar*m_xy_I - 12*omegaVar*m_yy_I - 72*omegaVar*m_zz_I + 107));
    m_yy_t45 = -(18*m_xx_I + 648*m_xy_I - 2772*m_yy_I + 108*m_zz_I + 11*omegaVar*m_xx_I + 48*omegaVar*m_xy_I - 302*omegaVar*m_yy_I + 66*omegaVar*m_zz_I - 72)/(27*(12*m_xx_I - 33*m_xy_I + 12*m_yy_I + 72*m_zz_I - 12*omegaVar*m_xx_I + 33*omegaVar*m_xy_I - 12*omegaVar*m_yy_I - 72*omegaVar*m_zz_I + 107));
    m_yz_t90 = (m_yz_I*(313*omegaVar + 2790))/(18*(12*m_xx_I - 33*m_xy_I + 12*m_yy_I + 72*m_zz_I - 12*omegaVar*m_xx_I + 33*omegaVar*m_xy_I - 12*omegaVar*m_yy_I - 72*omegaVar*m_zz_I + 107));
    m_zz_t45 = (198*m_xy_I - 72*m_xx_I - 72*m_yy_I + 2358*m_zz_I - 44*omegaVar*m_xx_I + 121*omegaVar*m_xy_I - 44*omegaVar*m_yy_I + 49*omegaVar*m_zz_I + 288)/(18*(12*m_xx_I - 33*m_xy_I + 12*m_yy_I + 72*m_zz_I - 12*omegaVar*m_xx_I + 33*omegaVar*m_xy_I - 12*omegaVar*m_yy_I - 72*omegaVar*m_zz_I + 107));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = (m_xy_t90/2 - m_xx_t45/6 + m_xz_t90 - m_yy_t45/6 + m_zz_t45/12 + (omegaVar*m_xx_t45)/12 - (omegaVar*m_xy_t90)/4 - (omegaVar*m_xz_t90)/2 + (omegaVar*m_yy_t45)/12 - (omegaVar*m_zz_t45)/24 - 1.0/18.0)*rho + RHO_0/18;
    L_BC_Fy = (m_xx_t45/6 - m_xy_t90/2 + m_yy_t45/6 + m_yz_t90 - m_zz_t45/12 - (omegaVar*m_xx_t45)/12 + (omegaVar*m_xy_t90)/4 - (omegaVar*m_yy_t45)/12 - (omegaVar*m_yz_t90)/2 + (omegaVar*m_zz_t45)/24 + 1.0/18.0)*rho - RHO_0/18;
    L_BC_Fz = (m_zz_t45 - (omegaVar*m_zz_t45)/2 + 1.0/3.0)*rho - RHO_0/3;
    #endif //_BC_FORCES 

    break;
case 0b0000010111110101:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[2] + pop[3] + pop[4] + pop[5] + pop[8] + pop[11] + pop[14] + pop[16] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[11] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[8] - pop[14]);
    m_xz_I = inv_rho_I * (-pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[11] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);

    rho = (36*(23*rho_I + 24*m_xx_I*rho_I + 57*m_xz_I*rho_I - 6*m_yy_I*rho_I + 24*m_zz_I*rho_I - 24*omegaVar*m_xx_I*rho_I - 57*omegaVar*m_xz_I*rho_I + 6*omegaVar*m_yy_I*rho_I - 24*omegaVar*m_zz_I*rho_I))/(5*(43*omegaVar + 72));

    m_xx_t45 = (936*m_xx_I + 1008*m_xz_I - 144*m_yy_I + 216*m_zz_I + 239*omegaVar*m_xx_I - 158*omegaVar*m_xz_I - 6*omegaVar*m_yy_I - 191*omegaVar*m_zz_I + 192)/(36*(24*m_xx_I + 57*m_xz_I - 6*m_yy_I + 24*m_zz_I - 24*omegaVar*m_xx_I - 57*omegaVar*m_xz_I + 6*omegaVar*m_yy_I - 24*omegaVar*m_zz_I + 23));
    m_xy_t90 = (5*m_xy_I*(43*omegaVar + 72))/(18*(24*m_xx_I + 57*m_xz_I - 6*m_yy_I + 24*m_zz_I - 24*omegaVar*m_xx_I - 57*omegaVar*m_xz_I + 6*omegaVar*m_yy_I - 24*omegaVar*m_zz_I + 23));
    m_xz_t90 = (504*m_xx_I + 2412*m_xz_I - 216*m_yy_I + 504*m_zz_I - 79*omegaVar*m_xx_I + 538*omegaVar*m_xz_I - 34*omegaVar*m_yy_I - 79*omegaVar*m_zz_I + 228)/(36*(24*m_xx_I + 57*m_xz_I - 6*m_yy_I + 24*m_zz_I - 24*omegaVar*m_xx_I - 57*omegaVar*m_xz_I + 6*omegaVar*m_yy_I - 24*omegaVar*m_zz_I + 23));
    m_yy_t45 = -(72*m_xx_I + 216*m_xz_I - 288*m_yy_I + 72*m_zz_I + 3*omegaVar*m_xx_I + 34*omegaVar*m_xz_I - 162*omegaVar*m_yy_I + 3*omegaVar*m_zz_I + 24)/(18*(24*m_xx_I + 57*m_xz_I - 6*m_yy_I + 24*m_zz_I - 24*omegaVar*m_xx_I - 57*omegaVar*m_xz_I + 6*omegaVar*m_yy_I - 24*omegaVar*m_zz_I + 23));
    m_yz_t90 = (5*m_yz_I*(43*omegaVar + 72))/(18*(24*m_xx_I + 57*m_xz_I - 6*m_yy_I + 24*m_zz_I - 24*omegaVar*m_xx_I - 57*omegaVar*m_xz_I + 6*omegaVar*m_yy_I - 24*omegaVar*m_zz_I + 23));
    m_zz_t45 = (216*m_xx_I + 1008*m_xz_I - 144*m_yy_I + 936*m_zz_I - 191*omegaVar*m_xx_I - 158*omegaVar*m_xz_I - 6*omegaVar*m_yy_I + 239*omegaVar*m_zz_I + 192)/(36*(24*m_xx_I + 57*m_xz_I - 6*m_yy_I + 24*m_zz_I - 24*omegaVar*m_xx_I - 57*omegaVar*m_xz_I + 6*omegaVar*m_yy_I - 24*omegaVar*m_zz_I + 23));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = (m_xz_t90/2 - (5*m_xx_t45)/6 - m_yy_t45/12 + m_zz_t45/6 + (5*omegaVar*m_xx_t45)/12 - (omegaVar*m_xz_t90)/4 + (omegaVar*m_yy_t45)/24 - (omegaVar*m_zz_t45)/12 - 5/18)*rho + (5*RHO_0)/18;
    L_BC_Fy = (rho*(m_xy_t90 - m_yz_t90)*(omegaVar - 2))/2;
    L_BC_Fz = (m_yy_t45/12 - m_xz_t90/2 - m_xx_t45/6 + (5*m_zz_t45)/6 + (omegaVar*m_xx_t45)/12 + (omegaVar*m_xz_t90)/4 - (omegaVar*m_yy_t45)/24 - (5*omegaVar*m_zz_t45)/12 + 5/18)*rho - (5*RHO_0)/18;
    #endif //_BC_FORCES 

    break;
case 0b0000010111110110:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[7] + pop[8] + pop[9] + pop[11] + pop[16] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] - (1.0/3.0)*pop[11] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8]);
    m_xz_I = inv_rho_I * (pop[9] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] + (2.0/3.0)*pop[11] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);

    rho = (18*(19*rho_I + 12*m_xx_I*rho_I - 33*m_xy_I*rho_I + 12*m_yy_I*rho_I + 6*m_zz_I*rho_I - 12*omegaVar*m_xx_I*rho_I + 33*omegaVar*m_xy_I*rho_I - 12*omegaVar*m_yy_I*rho_I - 6*omegaVar*m_zz_I*rho_I))/(59*omegaVar + 207);

    m_xx_t45 = -(2*(324*m_xy_I - 306*m_xx_I - 99*m_yy_I + 54*m_zz_I - 58*omegaVar*m_xx_I + 12*omegaVar*m_xy_I + omegaVar*m_yy_I + 30*omegaVar*m_zz_I - 36))/(27*(12*m_xx_I - 33*m_xy_I + 12*m_yy_I + 6*m_zz_I - 12*omegaVar*m_xx_I + 33*omegaVar*m_xy_I - 12*omegaVar*m_yy_I - 6*omegaVar*m_zz_I + 19));
    m_xy_t90 = -(216*m_xx_I - 801*m_xy_I + 216*m_yy_I - 99*m_zz_I + 8*omegaVar*m_xx_I - 81*omegaVar*m_xy_I + 8*omegaVar*m_yy_I - 55*omegaVar*m_zz_I + 66)/(18*(12*m_xx_I - 33*m_xy_I + 12*m_yy_I + 6*m_zz_I - 12*omegaVar*m_xx_I + 33*omegaVar*m_xy_I - 12*omegaVar*m_yy_I - 6*omegaVar*m_zz_I + 19));
    m_xz_t90 = (m_xz_I*(59*omegaVar + 207))/(9*(12*m_xx_I - 33*m_xy_I + 12*m_yy_I + 6*m_zz_I - 12*omegaVar*m_xx_I + 33*omegaVar*m_xy_I - 12*omegaVar*m_yy_I - 6*omegaVar*m_zz_I + 19));
    m_yy_t45 = -(2*(324*m_xy_I - 99*m_xx_I - 306*m_yy_I + 54*m_zz_I + omegaVar*m_xx_I + 12*omegaVar*m_xy_I - 58*omegaVar*m_yy_I + 30*omegaVar*m_zz_I - 36))/(27*(12*m_xx_I - 33*m_xy_I + 12*m_yy_I + 6*m_zz_I - 12*omegaVar*m_xx_I + 33*omegaVar*m_xy_I - 12*omegaVar*m_yy_I - 6*omegaVar*m_zz_I + 19));
    m_yz_t90 = (m_yz_I*(59*omegaVar + 207))/(9*(12*m_xx_I - 33*m_xy_I + 12*m_yy_I + 6*m_zz_I - 12*omegaVar*m_xx_I + 33*omegaVar*m_xy_I - 12*omegaVar*m_yy_I - 6*omegaVar*m_zz_I + 19));
    m_zz_t45 = (99*m_xy_I - 36*m_xx_I - 36*m_yy_I + 189*m_zz_I - 20*omegaVar*m_xx_I + 55*omegaVar*m_xy_I - 20*omegaVar*m_yy_I + 49*omegaVar*m_zz_I + 12)/(9*(12*m_xx_I - 33*m_xy_I + 12*m_yy_I + 6*m_zz_I - 12*omegaVar*m_xx_I + 33*omegaVar*m_xy_I - 12*omegaVar*m_yy_I - 6*omegaVar*m_zz_I + 19));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = -(m_xz_t90*rho*(omegaVar - 2))/2;
    L_BC_Fy = -(m_yz_t90*rho*(omegaVar - 2))/2;
    L_BC_Fz = (m_zz_t45 - (omegaVar*m_zz_t45)/2 + 1.0/3.0)*rho - RHO_0/3;
    #endif //_BC_FORCES 

    break;
case 0b0000010111110111:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[2] + pop[4] + pop[5] + pop[8] + pop[16] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[8]);
    m_xz_I = inv_rho_I * (-pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (-pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);

    rho = (12*(rho_I + m_xx_I*rho_I - 2*m_xy_I*rho_I + 2*m_xz_I*rho_I + m_yy_I*rho_I + 2*m_yz_I*rho_I + m_zz_I*rho_I - omegaVar*m_xx_I*rho_I + 2*omegaVar*m_xy_I*rho_I - 2*omegaVar*m_xz_I*rho_I - omegaVar*m_yy_I*rho_I - 2*omegaVar*m_yz_I*rho_I - omegaVar*m_zz_I*rho_I))/(5*omegaVar + 2);

    m_xx_t45 = (14*m_xx_I - 14*m_xy_I + 14*m_xz_I + 2*m_yy_I + 2*m_yz_I + 2*m_zz_I + 21*omegaVar*m_xx_I - 7*omegaVar*m_xy_I + 7*omegaVar*m_xz_I - 9*omegaVar*m_yy_I - 23*omegaVar*m_yz_I - 9*omegaVar*m_zz_I + 4)/(18*(m_xx_I - 2*m_xy_I + 2*m_xz_I + m_yy_I + 2*m_yz_I + m_zz_I - omegaVar*m_xx_I + 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - omegaVar*m_yy_I - 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 1));
    m_xy_t90 = -(14*m_xx_I - 50*m_xy_I + 14*m_xz_I + 14*m_yy_I + 14*m_yz_I + 2*m_zz_I + 7*omegaVar*m_xx_I - 69*omegaVar*m_xy_I - 21*omegaVar*m_xz_I + 7*omegaVar*m_yy_I - 21*omegaVar*m_yz_I - 23*omegaVar*m_zz_I + 8)/(36*(m_xx_I - 2*m_xy_I + 2*m_xz_I + m_yy_I + 2*m_yz_I + m_zz_I - omegaVar*m_xx_I + 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - omegaVar*m_yy_I - 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 1));
    m_xz_t90 = (14*m_xx_I - 14*m_xy_I + 50*m_xz_I + 2*m_yy_I + 14*m_yz_I + 14*m_zz_I + 7*omegaVar*m_xx_I + 21*omegaVar*m_xy_I + 69*omegaVar*m_xz_I - 23*omegaVar*m_yy_I - 21*omegaVar*m_yz_I + 7*omegaVar*m_zz_I + 8)/(36*(m_xx_I - 2*m_xy_I + 2*m_xz_I + m_yy_I + 2*m_yz_I + m_zz_I - omegaVar*m_xx_I + 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - omegaVar*m_yy_I - 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 1));
    m_yy_t45 = (2*m_xx_I - 14*m_xy_I + 2*m_xz_I + 14*m_yy_I + 14*m_yz_I + 2*m_zz_I - 9*omegaVar*m_xx_I - 7*omegaVar*m_xy_I - 23*omegaVar*m_xz_I + 21*omegaVar*m_yy_I + 7*omegaVar*m_yz_I - 9*omegaVar*m_zz_I + 4)/(18*(m_xx_I - 2*m_xy_I + 2*m_xz_I + m_yy_I + 2*m_yz_I + m_zz_I - omegaVar*m_xx_I + 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - omegaVar*m_yy_I - 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 1));
    m_yz_t90 = (2*m_xx_I - 14*m_xy_I + 14*m_xz_I + 14*m_yy_I + 50*m_yz_I + 14*m_zz_I - 23*omegaVar*m_xx_I + 21*omegaVar*m_xy_I - 21*omegaVar*m_xz_I + 7*omegaVar*m_yy_I + 69*omegaVar*m_yz_I + 7*omegaVar*m_zz_I + 8)/(36*(m_xx_I - 2*m_xy_I + 2*m_xz_I + m_yy_I + 2*m_yz_I + m_zz_I - omegaVar*m_xx_I + 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - omegaVar*m_yy_I - 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 1));
    m_zz_t45 = (2*m_xx_I - 2*m_xy_I + 14*m_xz_I + 2*m_yy_I + 14*m_yz_I + 14*m_zz_I - 9*omegaVar*m_xx_I + 23*omegaVar*m_xy_I + 7*omegaVar*m_xz_I - 9*omegaVar*m_yy_I + 7*omegaVar*m_yz_I + 21*omegaVar*m_zz_I + 4)/(18*(m_xx_I - 2*m_xy_I + 2*m_xz_I + m_yy_I + 2*m_yz_I + m_zz_I - omegaVar*m_xx_I + 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - omegaVar*m_yy_I - 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 1));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = (m_xz_t90/2 - m_xy_t90/2 - (2*m_xx_t45)/3 + m_yy_t45/12 + m_zz_t45/12 + (omegaVar*m_xx_t45)/3 + (omegaVar*m_xy_t90)/4 - (omegaVar*m_xz_t90)/4 - (omegaVar*m_yy_t45)/24 - (omegaVar*m_zz_t45)/24 - 2.0/9.0)*rho + (2*RHO_0)/9;
    L_BC_Fy = (m_xx_t45/12 - m_xy_t90/2 - (2*m_yy_t45)/3 + m_yz_t90/2 + m_zz_t45/12 - (omegaVar*m_xx_t45)/24 + (omegaVar*m_xy_t90)/4 + (omegaVar*m_yy_t45)/3 - (omegaVar*m_yz_t90)/4 - (omegaVar*m_zz_t45)/24 - 2.0/9.0)*rho + (2*RHO_0)/9;
    L_BC_Fz = ((2*m_zz_t45)/3 - m_xz_t90/2 - m_yy_t45/12 - m_yz_t90/2 - m_xx_t45/12 + (omegaVar*m_xx_t45)/24 + (omegaVar*m_xz_t90)/4 + (omegaVar*m_yy_t45)/24 + (omegaVar*m_yz_t90)/4 - (omegaVar*m_zz_t45)/3 + 2.0/9.0)*rho - (2*RHO_0)/9;
    #endif //_BC_FORCES 

    break;
case 0b0000010111111000:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[7] + pop[9] + pop[11] + pop[13] + pop[14] + pop[16] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[9] - (1.0/3.0)*pop[11] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] + (2.0/3.0)*pop[7] - (1.0/3.0)*pop[9] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] - (1.0/3.0)*pop[7] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[11] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);

    rho = (36*(107*rho_I + 12*m_xx_I*rho_I + 33*m_xy_I*rho_I + 12*m_yy_I*rho_I + 72*m_zz_I*rho_I - 12*omegaVar*m_xx_I*rho_I - 33*omegaVar*m_xy_I*rho_I - 12*omegaVar*m_yy_I*rho_I - 72*omegaVar*m_zz_I*rho_I))/(313*omegaVar + 2790);

    m_xx_t45 = (2772*m_xx_I + 648*m_xy_I - 18*m_yy_I - 108*m_zz_I + 302*omegaVar*m_xx_I + 48*omegaVar*m_xy_I - 11*omegaVar*m_yy_I - 66*omegaVar*m_zz_I + 72)/(27*(12*m_xx_I + 33*m_xy_I + 12*m_yy_I + 72*m_zz_I - 12*omegaVar*m_xx_I - 33*omegaVar*m_xy_I - 12*omegaVar*m_yy_I - 72*omegaVar*m_zz_I + 107));
    m_xy_t90 = (432*m_xx_I + 3978*m_xy_I + 432*m_yy_I - 198*m_zz_I + 32*omegaVar*m_xx_I + 401*omegaVar*m_xy_I + 32*omegaVar*m_yy_I - 121*omegaVar*m_zz_I + 132)/(36*(12*m_xx_I + 33*m_xy_I + 12*m_yy_I + 72*m_zz_I - 12*omegaVar*m_xx_I - 33*omegaVar*m_xy_I - 12*omegaVar*m_yy_I - 72*omegaVar*m_zz_I + 107));
    m_xz_t90 = (m_xz_I*(313*omegaVar + 2790))/(18*(12*m_xx_I + 33*m_xy_I + 12*m_yy_I + 72*m_zz_I - 12*omegaVar*m_xx_I - 33*omegaVar*m_xy_I - 12*omegaVar*m_yy_I - 72*omegaVar*m_zz_I + 107));
    m_yy_t45 = (648*m_xy_I - 18*m_xx_I + 2772*m_yy_I - 108*m_zz_I - 11*omegaVar*m_xx_I + 48*omegaVar*m_xy_I + 302*omegaVar*m_yy_I - 66*omegaVar*m_zz_I + 72)/(27*(12*m_xx_I + 33*m_xy_I + 12*m_yy_I + 72*m_zz_I - 12*omegaVar*m_xx_I - 33*omegaVar*m_xy_I - 12*omegaVar*m_yy_I - 72*omegaVar*m_zz_I + 107));
    m_yz_t90 = (m_yz_I*(313*omegaVar + 2790))/(18*(12*m_xx_I + 33*m_xy_I + 12*m_yy_I + 72*m_zz_I - 12*omegaVar*m_xx_I - 33*omegaVar*m_xy_I - 12*omegaVar*m_yy_I - 72*omegaVar*m_zz_I + 107));
    m_zz_t45 = -(72*m_xx_I + 198*m_xy_I + 72*m_yy_I - 2358*m_zz_I + 44*omegaVar*m_xx_I + 121*omegaVar*m_xy_I + 44*omegaVar*m_yy_I - 49*omegaVar*m_zz_I - 288)/(18*(12*m_xx_I + 33*m_xy_I + 12*m_yy_I + 72*m_zz_I - 12*omegaVar*m_xx_I - 33*omegaVar*m_xy_I - 12*omegaVar*m_yy_I - 72*omegaVar*m_zz_I + 107));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = (m_xx_t45/6 + m_xy_t90/2 + m_xz_t90 + m_yy_t45/6 - m_zz_t45/12 - (omegaVar*m_xx_t45)/12 - (omegaVar*m_xy_t90)/4 - (omegaVar*m_xz_t90)/2 - (omegaVar*m_yy_t45)/12 + (omegaVar*m_zz_t45)/24 + 1.0/18.0)*rho - RHO_0/18;
    L_BC_Fy = (m_xx_t45/6 + m_xy_t90/2 + m_yy_t45/6 + m_yz_t90 - m_zz_t45/12 - (omegaVar*m_xx_t45)/12 - (omegaVar*m_xy_t90)/4 - (omegaVar*m_yy_t45)/12 - (omegaVar*m_yz_t90)/2 + (omegaVar*m_zz_t45)/24 + 1.0/18.0)*rho - RHO_0/18;
    L_BC_Fz = (m_zz_t45 - (omegaVar*m_zz_t45)/2 + 1.0/3.0)*rho - RHO_0/3;
    #endif //_BC_FORCES 

    break;
case 0b0000010111111001:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[9] + pop[11] + pop[13] + pop[14] + pop[16] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] + (2.0/3.0)*pop[9] - (1.0/3.0)*pop[11] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (-pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[9] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[11] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);

    rho = (18*(19*rho_I + 12*m_xx_I*rho_I + 33*m_xy_I*rho_I + 12*m_yy_I*rho_I + 6*m_zz_I*rho_I - 12*omegaVar*m_xx_I*rho_I - 33*omegaVar*m_xy_I*rho_I - 12*omegaVar*m_yy_I*rho_I - 6*omegaVar*m_zz_I*rho_I))/(59*omegaVar + 207);

    m_xx_t45 = (2*(306*m_xx_I + 324*m_xy_I + 99*m_yy_I - 54*m_zz_I + 58*omegaVar*m_xx_I + 12*omegaVar*m_xy_I - omegaVar*m_yy_I - 30*omegaVar*m_zz_I + 36))/(27*(12*m_xx_I + 33*m_xy_I + 12*m_yy_I + 6*m_zz_I - 12*omegaVar*m_xx_I - 33*omegaVar*m_xy_I - 12*omegaVar*m_yy_I - 6*omegaVar*m_zz_I + 19));
    m_xy_t90 = (216*m_xx_I + 801*m_xy_I + 216*m_yy_I - 99*m_zz_I + 8*omegaVar*m_xx_I + 81*omegaVar*m_xy_I + 8*omegaVar*m_yy_I - 55*omegaVar*m_zz_I + 66)/(18*(12*m_xx_I + 33*m_xy_I + 12*m_yy_I + 6*m_zz_I - 12*omegaVar*m_xx_I - 33*omegaVar*m_xy_I - 12*omegaVar*m_yy_I - 6*omegaVar*m_zz_I + 19));
    m_xz_t90 = (m_xz_I*(59*omegaVar + 207))/(9*(12*m_xx_I + 33*m_xy_I + 12*m_yy_I + 6*m_zz_I - 12*omegaVar*m_xx_I - 33*omegaVar*m_xy_I - 12*omegaVar*m_yy_I - 6*omegaVar*m_zz_I + 19));
    m_yy_t45 = (2*(99*m_xx_I + 324*m_xy_I + 306*m_yy_I - 54*m_zz_I - omegaVar*m_xx_I + 12*omegaVar*m_xy_I + 58*omegaVar*m_yy_I - 30*omegaVar*m_zz_I + 36))/(27*(12*m_xx_I + 33*m_xy_I + 12*m_yy_I + 6*m_zz_I - 12*omegaVar*m_xx_I - 33*omegaVar*m_xy_I - 12*omegaVar*m_yy_I - 6*omegaVar*m_zz_I + 19));
    m_yz_t90 = (m_yz_I*(59*omegaVar + 207))/(9*(12*m_xx_I + 33*m_xy_I + 12*m_yy_I + 6*m_zz_I - 12*omegaVar*m_xx_I - 33*omegaVar*m_xy_I - 12*omegaVar*m_yy_I - 6*omegaVar*m_zz_I + 19));
    m_zz_t45 = -(36*m_xx_I + 99*m_xy_I + 36*m_yy_I - 189*m_zz_I + 20*omegaVar*m_xx_I + 55*omegaVar*m_xy_I + 20*omegaVar*m_yy_I - 49*omegaVar*m_zz_I - 12)/(9*(12*m_xx_I + 33*m_xy_I + 12*m_yy_I + 6*m_zz_I - 12*omegaVar*m_xx_I - 33*omegaVar*m_xy_I - 12*omegaVar*m_yy_I - 6*omegaVar*m_zz_I + 19));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = -(m_xz_t90*rho*(omegaVar - 2))/2;
    L_BC_Fy = -(m_yz_t90*rho*(omegaVar - 2))/2;
    L_BC_Fz = (m_zz_t45 - (omegaVar*m_zz_t45)/2 + 1.0/3.0)*rho - RHO_0/3;
    #endif //_BC_FORCES 

    break;
case 0b0000010111111010:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[3] + pop[4] + pop[5] + pop[7] + pop[9] + pop[11] + pop[13] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[9] - (1.0/3.0)*pop[11] + (2.0/3.0)*pop[13] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] - pop[13]);
    m_xz_I = inv_rho_I * (pop[9]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] + (2.0/3.0)*pop[7] - (1.0/3.0)*pop[9] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] - (1.0/3.0)*pop[7] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[11] - (1.0/3.0)*pop[13] + (2.0/3.0)*pop[18]);

    rho = (36*(23*rho_I + 24*m_xx_I*rho_I - 57*m_xz_I*rho_I - 6*m_yy_I*rho_I + 24*m_zz_I*rho_I - 24*omegaVar*m_xx_I*rho_I + 57*omegaVar*m_xz_I*rho_I + 6*omegaVar*m_yy_I*rho_I - 24*omegaVar*m_zz_I*rho_I))/(5*(43*omegaVar + 72));

    m_xx_t45 = (936*m_xx_I - 1008*m_xz_I - 144*m_yy_I + 216*m_zz_I + 239*omegaVar*m_xx_I + 158*omegaVar*m_xz_I - 6*omegaVar*m_yy_I - 191*omegaVar*m_zz_I + 192)/(36*(24*m_xx_I - 57*m_xz_I - 6*m_yy_I + 24*m_zz_I - 24*omegaVar*m_xx_I + 57*omegaVar*m_xz_I + 6*omegaVar*m_yy_I - 24*omegaVar*m_zz_I + 23));
    m_xy_t90 = (5*m_xy_I*(43*omegaVar + 72))/(18*(24*m_xx_I - 57*m_xz_I - 6*m_yy_I + 24*m_zz_I - 24*omegaVar*m_xx_I + 57*omegaVar*m_xz_I + 6*omegaVar*m_yy_I - 24*omegaVar*m_zz_I + 23));
    m_xz_t90 = (2412*m_xz_I - 504*m_xx_I + 216*m_yy_I - 504*m_zz_I + 79*omegaVar*m_xx_I + 538*omegaVar*m_xz_I + 34*omegaVar*m_yy_I + 79*omegaVar*m_zz_I - 228)/(36*(24*m_xx_I - 57*m_xz_I - 6*m_yy_I + 24*m_zz_I - 24*omegaVar*m_xx_I + 57*omegaVar*m_xz_I + 6*omegaVar*m_yy_I - 24*omegaVar*m_zz_I + 23));
    m_yy_t45 = -(72*m_xx_I - 216*m_xz_I - 288*m_yy_I + 72*m_zz_I + 3*omegaVar*m_xx_I - 34*omegaVar*m_xz_I - 162*omegaVar*m_yy_I + 3*omegaVar*m_zz_I + 24)/(18*(24*m_xx_I - 57*m_xz_I - 6*m_yy_I + 24*m_zz_I - 24*omegaVar*m_xx_I + 57*omegaVar*m_xz_I + 6*omegaVar*m_yy_I - 24*omegaVar*m_zz_I + 23));
    m_yz_t90 = (5*m_yz_I*(43*omegaVar + 72))/(18*(24*m_xx_I - 57*m_xz_I - 6*m_yy_I + 24*m_zz_I - 24*omegaVar*m_xx_I + 57*omegaVar*m_xz_I + 6*omegaVar*m_yy_I - 24*omegaVar*m_zz_I + 23));
    m_zz_t45 = (216*m_xx_I - 1008*m_xz_I - 144*m_yy_I + 936*m_zz_I - 191*omegaVar*m_xx_I + 158*omegaVar*m_xz_I - 6*omegaVar*m_yy_I + 239*omegaVar*m_zz_I + 192)/(36*(24*m_xx_I - 57*m_xz_I - 6*m_yy_I + 24*m_zz_I - 24*omegaVar*m_xx_I + 57*omegaVar*m_xz_I + 6*omegaVar*m_yy_I - 24*omegaVar*m_zz_I + 23));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = ((5*m_xx_t45)/6 + m_xz_t90/2 + m_yy_t45/12 - m_zz_t45/6 - (5*omegaVar*m_xx_t45)/12 - (omegaVar*m_xz_t90)/4 - (omegaVar*m_yy_t45)/24 + (omegaVar*m_zz_t45)/12 + 5/18)*rho - (5*RHO_0)/18;
    L_BC_Fy = -(rho*(m_xy_t90 + m_yz_t90)*(omegaVar - 2))/2;
    L_BC_Fz = (m_xz_t90/2 - m_xx_t45/6 + m_yy_t45/12 + (5*m_zz_t45)/6 + (omegaVar*m_xx_t45)/12 - (omegaVar*m_xz_t90)/4 - (omegaVar*m_yy_t45)/24 - (5*omegaVar*m_zz_t45)/12 + 5/18)*rho - (5*RHO_0)/18;
    #endif //_BC_FORCES 

    break;
case 0b0000010111111011:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[4] + pop[5] + pop[9] + pop[13] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[13] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (-pop[13]);
    m_xz_I = inv_rho_I * (pop[9]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[9] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (-pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[9] - (1.0/3.0)*pop[13] + (2.0/3.0)*pop[18]);

    rho = (12*(rho_I + m_xx_I*rho_I + 2*m_xy_I*rho_I - 2*m_xz_I*rho_I + m_yy_I*rho_I + 2*m_yz_I*rho_I + m_zz_I*rho_I - omegaVar*m_xx_I*rho_I - 2*omegaVar*m_xy_I*rho_I + 2*omegaVar*m_xz_I*rho_I - omegaVar*m_yy_I*rho_I - 2*omegaVar*m_yz_I*rho_I - omegaVar*m_zz_I*rho_I))/(5*omegaVar + 2);

    m_xx_t45 = (14*m_xx_I + 14*m_xy_I - 14*m_xz_I + 2*m_yy_I + 2*m_yz_I + 2*m_zz_I + 21*omegaVar*m_xx_I + 7*omegaVar*m_xy_I - 7*omegaVar*m_xz_I - 9*omegaVar*m_yy_I - 23*omegaVar*m_yz_I - 9*omegaVar*m_zz_I + 4)/(18*(m_xx_I + 2*m_xy_I - 2*m_xz_I + m_yy_I + 2*m_yz_I + m_zz_I - omegaVar*m_xx_I - 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - omegaVar*m_yy_I - 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 1));
    m_xy_t90 = (14*m_xx_I + 50*m_xy_I - 14*m_xz_I + 14*m_yy_I + 14*m_yz_I + 2*m_zz_I + 7*omegaVar*m_xx_I + 69*omegaVar*m_xy_I + 21*omegaVar*m_xz_I + 7*omegaVar*m_yy_I - 21*omegaVar*m_yz_I - 23*omegaVar*m_zz_I + 8)/(36*(m_xx_I + 2*m_xy_I - 2*m_xz_I + m_yy_I + 2*m_yz_I + m_zz_I - omegaVar*m_xx_I - 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - omegaVar*m_yy_I - 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 1));
    m_xz_t90 = -(14*m_xx_I + 14*m_xy_I - 50*m_xz_I + 2*m_yy_I + 14*m_yz_I + 14*m_zz_I + 7*omegaVar*m_xx_I - 21*omegaVar*m_xy_I - 69*omegaVar*m_xz_I - 23*omegaVar*m_yy_I - 21*omegaVar*m_yz_I + 7*omegaVar*m_zz_I + 8)/(36*(m_xx_I + 2*m_xy_I - 2*m_xz_I + m_yy_I + 2*m_yz_I + m_zz_I - omegaVar*m_xx_I - 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - omegaVar*m_yy_I - 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 1));
    m_yy_t45 = (2*m_xx_I + 14*m_xy_I - 2*m_xz_I + 14*m_yy_I + 14*m_yz_I + 2*m_zz_I - 9*omegaVar*m_xx_I + 7*omegaVar*m_xy_I + 23*omegaVar*m_xz_I + 21*omegaVar*m_yy_I + 7*omegaVar*m_yz_I - 9*omegaVar*m_zz_I + 4)/(18*(m_xx_I + 2*m_xy_I - 2*m_xz_I + m_yy_I + 2*m_yz_I + m_zz_I - omegaVar*m_xx_I - 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - omegaVar*m_yy_I - 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 1));
    m_yz_t90 = (2*m_xx_I + 14*m_xy_I - 14*m_xz_I + 14*m_yy_I + 50*m_yz_I + 14*m_zz_I - 23*omegaVar*m_xx_I - 21*omegaVar*m_xy_I + 21*omegaVar*m_xz_I + 7*omegaVar*m_yy_I + 69*omegaVar*m_yz_I + 7*omegaVar*m_zz_I + 8)/(36*(m_xx_I + 2*m_xy_I - 2*m_xz_I + m_yy_I + 2*m_yz_I + m_zz_I - omegaVar*m_xx_I - 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - omegaVar*m_yy_I - 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 1));
    m_zz_t45 = (2*m_xx_I + 2*m_xy_I - 14*m_xz_I + 2*m_yy_I + 14*m_yz_I + 14*m_zz_I - 9*omegaVar*m_xx_I - 23*omegaVar*m_xy_I - 7*omegaVar*m_xz_I - 9*omegaVar*m_yy_I + 7*omegaVar*m_yz_I + 21*omegaVar*m_zz_I + 4)/(18*(m_xx_I + 2*m_xy_I - 2*m_xz_I + m_yy_I + 2*m_yz_I + m_zz_I - omegaVar*m_xx_I - 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - omegaVar*m_yy_I - 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 1));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = ((2*m_xx_t45)/3 - m_xy_t90/2 + m_xz_t90/2 - m_yy_t45/12 - m_zz_t45/12 - (omegaVar*m_xx_t45)/3 + (omegaVar*m_xy_t90)/4 - (omegaVar*m_xz_t90)/4 + (omegaVar*m_yy_t45)/24 + (omegaVar*m_zz_t45)/24 + 2.0/9.0)*rho - (2*RHO_0)/9;
    L_BC_Fy = (m_xx_t45/12 + m_xy_t90/2 - (2*m_yy_t45)/3 + m_yz_t90/2 + m_zz_t45/12 - (omegaVar*m_xx_t45)/24 - (omegaVar*m_xy_t90)/4 + (omegaVar*m_yy_t45)/3 - (omegaVar*m_yz_t90)/4 - (omegaVar*m_zz_t45)/24 - 2.0/9.0)*rho + (2*RHO_0)/9;
    L_BC_Fz = (m_xz_t90/2 - m_xx_t45/12 - m_yy_t45/12 - m_yz_t90/2 + (2*m_zz_t45)/3 + (omegaVar*m_xx_t45)/24 - (omegaVar*m_xz_t90)/4 + (omegaVar*m_yy_t45)/24 + (omegaVar*m_yz_t90)/4 - (omegaVar*m_zz_t45)/3 + 2.0/9.0)*rho - (2*RHO_0)/9;
    #endif //_BC_FORCES 

    break;
case 0b0000010111111100:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[5] + pop[7] + pop[9] + pop[11] + pop[14] + pop[16];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[5] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[9] - (1.0/3.0)*pop[11] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[16]);
    m_xy_I = inv_rho_I * (pop[7] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] - (1.0/3.0)*pop[5] + (2.0/3.0)*pop[7] - (1.0/3.0)*pop[9] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[16]);
    m_yz_I = inv_rho_I * (pop[11]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] + (2.0/3.0)*pop[5] - (1.0/3.0)*pop[7] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[11] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[16]);

    rho = (36*(23*rho_I - 6*m_xx_I*rho_I + 24*m_yy_I*rho_I - 57*m_yz_I*rho_I + 24*m_zz_I*rho_I + 6*omegaVar*m_xx_I*rho_I - 24*omegaVar*m_yy_I*rho_I + 57*omegaVar*m_yz_I*rho_I - 24*omegaVar*m_zz_I*rho_I))/(5*(43*omegaVar + 72));

    m_xx_t45 = -(72*m_yy_I - 288*m_xx_I - 216*m_yz_I + 72*m_zz_I - 162*omegaVar*m_xx_I + 3*omegaVar*m_yy_I - 34*omegaVar*m_yz_I + 3*omegaVar*m_zz_I + 24)/(18*(24*m_yy_I - 6*m_xx_I - 57*m_yz_I + 24*m_zz_I + 6*omegaVar*m_xx_I - 24*omegaVar*m_yy_I + 57*omegaVar*m_yz_I - 24*omegaVar*m_zz_I + 23));
    m_xy_t90 = (5*m_xy_I*(43*omegaVar + 72))/(18*(24*m_yy_I - 6*m_xx_I - 57*m_yz_I + 24*m_zz_I + 6*omegaVar*m_xx_I - 24*omegaVar*m_yy_I + 57*omegaVar*m_yz_I - 24*omegaVar*m_zz_I + 23));
    m_xz_t90 = (5*m_xz_I*(43*omegaVar + 72))/(18*(24*m_yy_I - 6*m_xx_I - 57*m_yz_I + 24*m_zz_I + 6*omegaVar*m_xx_I - 24*omegaVar*m_yy_I + 57*omegaVar*m_yz_I - 24*omegaVar*m_zz_I + 23));
    m_yy_t45 = (936*m_yy_I - 144*m_xx_I - 1008*m_yz_I + 216*m_zz_I - 6*omegaVar*m_xx_I + 239*omegaVar*m_yy_I + 158*omegaVar*m_yz_I - 191*omegaVar*m_zz_I + 192)/(36*(24*m_yy_I - 6*m_xx_I - 57*m_yz_I + 24*m_zz_I + 6*omegaVar*m_xx_I - 24*omegaVar*m_yy_I + 57*omegaVar*m_yz_I - 24*omegaVar*m_zz_I + 23));
    m_yz_t90 = (216*m_xx_I - 504*m_yy_I + 2412*m_yz_I - 504*m_zz_I + 34*omegaVar*m_xx_I + 79*omegaVar*m_yy_I + 538*omegaVar*m_yz_I + 79*omegaVar*m_zz_I - 228)/(36*(24*m_yy_I - 6*m_xx_I - 57*m_yz_I + 24*m_zz_I + 6*omegaVar*m_xx_I - 24*omegaVar*m_yy_I + 57*omegaVar*m_yz_I - 24*omegaVar*m_zz_I + 23));
    m_zz_t45 = (216*m_yy_I - 144*m_xx_I - 1008*m_yz_I + 936*m_zz_I - 6*omegaVar*m_xx_I - 191*omegaVar*m_yy_I + 158*omegaVar*m_yz_I + 239*omegaVar*m_zz_I + 192)/(36*(24*m_yy_I - 6*m_xx_I - 57*m_yz_I + 24*m_zz_I + 6*omegaVar*m_xx_I - 24*omegaVar*m_yy_I + 57*omegaVar*m_yz_I - 24*omegaVar*m_zz_I + 23));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = -(rho*(m_xy_t90 + m_xz_t90)*(omegaVar - 2))/2;
    L_BC_Fy = (m_xx_t45/12 + (5*m_yy_t45)/6 + m_yz_t90/2 - m_zz_t45/6 - (omegaVar*m_xx_t45)/24 - (5*omegaVar*m_yy_t45)/12 - (omegaVar*m_yz_t90)/4 + (omegaVar*m_zz_t45)/12 + 5/18)*rho - (5*RHO_0)/18;
    L_BC_Fz = (m_xx_t45/12 - m_yy_t45/6 + m_yz_t90/2 + (5*m_zz_t45)/6 - (omegaVar*m_xx_t45)/24 + (omegaVar*m_yy_t45)/12 - (omegaVar*m_yz_t90)/4 - (5*omegaVar*m_zz_t45)/12 + 5/18)*rho - (5*RHO_0)/18;
    #endif //_BC_FORCES 

    break;
case 0b0000010111111101:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[2] + pop[3] + pop[5] + pop[11] + pop[14] + pop[16];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[11] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[16]);
    m_xy_I = inv_rho_I * (-pop[14]);
    m_xz_I = inv_rho_I * (-pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] - (1.0/3.0)*pop[5] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[16]);
    m_yz_I = inv_rho_I * (pop[11]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[11] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[16]);

    rho = (12*(rho_I + m_xx_I*rho_I + 2*m_xy_I*rho_I + 2*m_xz_I*rho_I + m_yy_I*rho_I - 2*m_yz_I*rho_I + m_zz_I*rho_I - omegaVar*m_xx_I*rho_I - 2*omegaVar*m_xy_I*rho_I - 2*omegaVar*m_xz_I*rho_I - omegaVar*m_yy_I*rho_I + 2*omegaVar*m_yz_I*rho_I - omegaVar*m_zz_I*rho_I))/(5*omegaVar + 2);

    m_xx_t45 = (14*m_xx_I + 14*m_xy_I + 14*m_xz_I + 2*m_yy_I - 2*m_yz_I + 2*m_zz_I + 21*omegaVar*m_xx_I + 7*omegaVar*m_xy_I + 7*omegaVar*m_xz_I - 9*omegaVar*m_yy_I + 23*omegaVar*m_yz_I - 9*omegaVar*m_zz_I + 4)/(18*(m_xx_I + 2*m_xy_I + 2*m_xz_I + m_yy_I - 2*m_yz_I + m_zz_I - omegaVar*m_xx_I - 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - omegaVar*m_yy_I + 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 1));
    m_xy_t90 = (14*m_xx_I + 50*m_xy_I + 14*m_xz_I + 14*m_yy_I - 14*m_yz_I + 2*m_zz_I + 7*omegaVar*m_xx_I + 69*omegaVar*m_xy_I - 21*omegaVar*m_xz_I + 7*omegaVar*m_yy_I + 21*omegaVar*m_yz_I - 23*omegaVar*m_zz_I + 8)/(36*(m_xx_I + 2*m_xy_I + 2*m_xz_I + m_yy_I - 2*m_yz_I + m_zz_I - omegaVar*m_xx_I - 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - omegaVar*m_yy_I + 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 1));
    m_xz_t90 = (14*m_xx_I + 14*m_xy_I + 50*m_xz_I + 2*m_yy_I - 14*m_yz_I + 14*m_zz_I + 7*omegaVar*m_xx_I - 21*omegaVar*m_xy_I + 69*omegaVar*m_xz_I - 23*omegaVar*m_yy_I + 21*omegaVar*m_yz_I + 7*omegaVar*m_zz_I + 8)/(36*(m_xx_I + 2*m_xy_I + 2*m_xz_I + m_yy_I - 2*m_yz_I + m_zz_I - omegaVar*m_xx_I - 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - omegaVar*m_yy_I + 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 1));
    m_yy_t45 = (2*m_xx_I + 14*m_xy_I + 2*m_xz_I + 14*m_yy_I - 14*m_yz_I + 2*m_zz_I - 9*omegaVar*m_xx_I + 7*omegaVar*m_xy_I - 23*omegaVar*m_xz_I + 21*omegaVar*m_yy_I - 7*omegaVar*m_yz_I - 9*omegaVar*m_zz_I + 4)/(18*(m_xx_I + 2*m_xy_I + 2*m_xz_I + m_yy_I - 2*m_yz_I + m_zz_I - omegaVar*m_xx_I - 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - omegaVar*m_yy_I + 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 1));
    m_yz_t90 = -(2*m_xx_I + 14*m_xy_I + 14*m_xz_I + 14*m_yy_I - 50*m_yz_I + 14*m_zz_I - 23*omegaVar*m_xx_I - 21*omegaVar*m_xy_I - 21*omegaVar*m_xz_I + 7*omegaVar*m_yy_I - 69*omegaVar*m_yz_I + 7*omegaVar*m_zz_I + 8)/(36*(m_xx_I + 2*m_xy_I + 2*m_xz_I + m_yy_I - 2*m_yz_I + m_zz_I - omegaVar*m_xx_I - 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - omegaVar*m_yy_I + 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 1));
    m_zz_t45 = (2*m_xx_I + 2*m_xy_I + 14*m_xz_I + 2*m_yy_I - 14*m_yz_I + 14*m_zz_I - 9*omegaVar*m_xx_I - 23*omegaVar*m_xy_I + 7*omegaVar*m_xz_I - 9*omegaVar*m_yy_I - 7*omegaVar*m_yz_I + 21*omegaVar*m_zz_I + 4)/(18*(m_xx_I + 2*m_xy_I + 2*m_xz_I + m_yy_I - 2*m_yz_I + m_zz_I - omegaVar*m_xx_I - 2*omegaVar*m_xy_I - 2*omegaVar*m_xz_I - omegaVar*m_yy_I + 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 1));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = (m_xy_t90/2 - (2*m_xx_t45)/3 + m_xz_t90/2 + m_yy_t45/12 + m_zz_t45/12 + (omegaVar*m_xx_t45)/3 - (omegaVar*m_xy_t90)/4 - (omegaVar*m_xz_t90)/4 - (omegaVar*m_yy_t45)/24 - (omegaVar*m_zz_t45)/24 - 2.0/9.0)*rho + (2*RHO_0)/9;
    L_BC_Fy = ((2*m_yy_t45)/3 - m_xy_t90/2 - m_xx_t45/12 + m_yz_t90/2 - m_zz_t45/12 + (omegaVar*m_xx_t45)/24 + (omegaVar*m_xy_t90)/4 - (omegaVar*m_yy_t45)/3 - (omegaVar*m_yz_t90)/4 + (omegaVar*m_zz_t45)/24 + 2.0/9.0)*rho - (2*RHO_0)/9;
    L_BC_Fz = (m_yz_t90/2 - m_xz_t90/2 - m_yy_t45/12 - m_xx_t45/12 + (2*m_zz_t45)/3 + (omegaVar*m_xx_t45)/24 + (omegaVar*m_xz_t90)/4 + (omegaVar*m_yy_t45)/24 - (omegaVar*m_yz_t90)/4 - (omegaVar*m_zz_t45)/3 + 2.0/9.0)*rho - (2*RHO_0)/9;
    #endif //_BC_FORCES 

    break;
case 0b0000010111111110:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[3] + pop[5] + pop[7] + pop[9] + pop[11];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[5] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[9] - (1.0/3.0)*pop[11]);
    m_xy_I = inv_rho_I * (pop[7]);
    m_xz_I = inv_rho_I * (pop[9]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] + (2.0/3.0)*pop[3] - (1.0/3.0)*pop[5] + (2.0/3.0)*pop[7] - (1.0/3.0)*pop[9] + (2.0/3.0)*pop[11]);
    m_yz_I = inv_rho_I * (pop[11]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[3] + (2.0/3.0)*pop[5] - (1.0/3.0)*pop[7] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[11]);

    rho = (12*(rho_I + m_xx_I*rho_I - 2*m_xy_I*rho_I - 2*m_xz_I*rho_I + m_yy_I*rho_I - 2*m_yz_I*rho_I + m_zz_I*rho_I - omegaVar*m_xx_I*rho_I + 2*omegaVar*m_xy_I*rho_I + 2*omegaVar*m_xz_I*rho_I - omegaVar*m_yy_I*rho_I + 2*omegaVar*m_yz_I*rho_I - omegaVar*m_zz_I*rho_I))/(5*omegaVar + 2);

    m_xx_t45 = -(14*m_xy_I - 14*m_xx_I + 14*m_xz_I - 2*m_yy_I + 2*m_yz_I - 2*m_zz_I - 21*omegaVar*m_xx_I + 7*omegaVar*m_xy_I + 7*omegaVar*m_xz_I + 9*omegaVar*m_yy_I - 23*omegaVar*m_yz_I + 9*omegaVar*m_zz_I - 4)/(18*(m_xx_I - 2*m_xy_I - 2*m_xz_I + m_yy_I - 2*m_yz_I + m_zz_I - omegaVar*m_xx_I + 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - omegaVar*m_yy_I + 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 1));
    m_xy_t90 = -(14*m_xx_I - 50*m_xy_I - 14*m_xz_I + 14*m_yy_I - 14*m_yz_I + 2*m_zz_I + 7*omegaVar*m_xx_I - 69*omegaVar*m_xy_I + 21*omegaVar*m_xz_I + 7*omegaVar*m_yy_I + 21*omegaVar*m_yz_I - 23*omegaVar*m_zz_I + 8)/(36*(m_xx_I - 2*m_xy_I - 2*m_xz_I + m_yy_I - 2*m_yz_I + m_zz_I - omegaVar*m_xx_I + 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - omegaVar*m_yy_I + 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 1));
    m_xz_t90 = -(14*m_xx_I - 14*m_xy_I - 50*m_xz_I + 2*m_yy_I - 14*m_yz_I + 14*m_zz_I + 7*omegaVar*m_xx_I + 21*omegaVar*m_xy_I - 69*omegaVar*m_xz_I - 23*omegaVar*m_yy_I + 21*omegaVar*m_yz_I + 7*omegaVar*m_zz_I + 8)/(36*(m_xx_I - 2*m_xy_I - 2*m_xz_I + m_yy_I - 2*m_yz_I + m_zz_I - omegaVar*m_xx_I + 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - omegaVar*m_yy_I + 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 1));
    m_yy_t45 = -(14*m_xy_I - 2*m_xx_I + 2*m_xz_I - 14*m_yy_I + 14*m_yz_I - 2*m_zz_I + 9*omegaVar*m_xx_I + 7*omegaVar*m_xy_I - 23*omegaVar*m_xz_I - 21*omegaVar*m_yy_I + 7*omegaVar*m_yz_I + 9*omegaVar*m_zz_I - 4)/(18*(m_xx_I - 2*m_xy_I - 2*m_xz_I + m_yy_I - 2*m_yz_I + m_zz_I - omegaVar*m_xx_I + 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - omegaVar*m_yy_I + 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 1));
    m_yz_t90 = -(2*m_xx_I - 14*m_xy_I - 14*m_xz_I + 14*m_yy_I - 50*m_yz_I + 14*m_zz_I - 23*omegaVar*m_xx_I + 21*omegaVar*m_xy_I + 21*omegaVar*m_xz_I + 7*omegaVar*m_yy_I - 69*omegaVar*m_yz_I + 7*omegaVar*m_zz_I + 8)/(36*(m_xx_I - 2*m_xy_I - 2*m_xz_I + m_yy_I - 2*m_yz_I + m_zz_I - omegaVar*m_xx_I + 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - omegaVar*m_yy_I + 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 1));
    m_zz_t45 = -(2*m_xy_I - 2*m_xx_I + 14*m_xz_I - 2*m_yy_I + 14*m_yz_I - 14*m_zz_I + 9*omegaVar*m_xx_I - 23*omegaVar*m_xy_I + 7*omegaVar*m_xz_I + 9*omegaVar*m_yy_I + 7*omegaVar*m_yz_I - 21*omegaVar*m_zz_I - 4)/(18*(m_xx_I - 2*m_xy_I - 2*m_xz_I + m_yy_I - 2*m_yz_I + m_zz_I - omegaVar*m_xx_I + 2*omegaVar*m_xy_I + 2*omegaVar*m_xz_I - omegaVar*m_yy_I + 2*omegaVar*m_yz_I - omegaVar*m_zz_I + 1));

    rhoVar = rho;
    #ifdef BC_FORCES
    L_BC_Fx = ((2*m_xx_t45)/3 + m_xy_t90/2 + m_xz_t90/2 - m_yy_t45/12 - m_zz_t45/12 - (omegaVar*m_xx_t45)/3 - (omegaVar*m_xy_t90)/4 - (omegaVar*m_xz_t90)/4 + (omegaVar*m_yy_t45)/24 + (omegaVar*m_zz_t45)/24 + 2.0/9.0)*rho - (2*RHO_0)/9;
    L_BC_Fy = (m_xy_t90/2 - m_xx_t45/12 + (2*m_yy_t45)/3 + m_yz_t90/2 - m_zz_t45/12 + (omegaVar*m_xx_t45)/24 - (omegaVar*m_xy_t90)/4 - (omegaVar*m_yy_t45)/3 - (omegaVar*m_yz_t90)/4 + (omegaVar*m_zz_t45)/24 + 2.0/9.0)*rho - (2*RHO_0)/9;
    L_BC_Fz = (m_xz_t90/2 - m_xx_t45/12 - m_yy_t45/12 + m_yz_t90/2 + (2*m_zz_t45)/3 + (omegaVar*m_xx_t45)/24 - (omegaVar*m_xz_t90)/4 + (omegaVar*m_yy_t45)/24 - (omegaVar*m_yz_t90)/4 - (omegaVar*m_zz_t45)/3 + 2.0/9.0)*rho - (2*RHO_0)/9;
    #endif //_BC_FORCES 

    break;
}