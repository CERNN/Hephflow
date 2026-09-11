
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
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = rho_I;

    m_xx_t45 = (m_xx_I*rho_I)/rho;
    m_xy_t90 = (m_xy_I*rho_I)/rho;
    m_xz_t90 = (m_xz_I*rho_I)/rho;
    m_yy_t45 = (m_yy_I*rho_I)/rho;
    m_yz_t90 = (m_yz_I*rho_I)/rho;
    m_zz_t45 = (m_zz_I*rho_I)/rho;

    rhoVar = rho;

    break;
case 0b0000010100000010:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = rho_I;

    m_xx_t45 = (m_xx_I*rho_I)/rho;
    m_xy_t90 = (m_xy_I*rho_I)/rho;
    m_xz_t90 = (m_xz_I*rho_I)/rho;
    m_yy_t45 = (m_yy_I*rho_I)/rho;
    m_yz_t90 = (m_yz_I*rho_I)/rho;
    m_zz_t45 = (m_zz_I*rho_I)/rho;

    rhoVar = rho;

    break;
case 0b0000010100000011:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (24*(15*rho_I - m_xx_I*rho_I + 2*m_yy_I*rho_I + 6*m_yz_I*rho_I + 2*m_zz_I*rho_I + omegaVar*m_xx_I*rho_I - 2*omegaVar*m_yy_I*rho_I - 6*omegaVar*m_yz_I*rho_I - 2*omegaVar*m_zz_I*rho_I))/(6*omegaVar + 12*uy0 + 12*uz0 + 18*omegaVar*uy0 + 18*omegaVar*uz0 + 15*omegaVar*ux0*ux0 - 30*omegaVar*uy0*uy0 - 30*omegaVar*uz0*uz0 - 90*omegaVar*uy0*uz0 + 344);

    m_xx_t45 = -(2*(rho - 69*m_xx_I*rho_I + 3*m_yy_I*rho_I + 9*m_yz_I*rho_I + 3*m_zz_I*rho_I + 3*rho*uy0 + 3*rho*uz0))/(135*rho);
    m_xy_t90 = (m_xy_I*rho_I)/rho;
    m_xz_t90 = (m_xz_I*rho_I)/rho;
    m_yy_t45 = (4*rho - 6*m_xx_I*rho_I + 147*m_yy_I*rho_I + 36*m_yz_I*rho_I + 12*m_zz_I*rho_I + 12*rho*uy0 + 12*rho*uz0)/(135*rho);
    m_yz_t90 = (2*rho - 3*m_xx_I*rho_I + 6*m_yy_I*rho_I + 63*m_yz_I*rho_I + 6*m_zz_I*rho_I + 6*rho*uy0 + 6*rho*uz0)/(45*rho);
    m_zz_t45 = (4*rho - 6*m_xx_I*rho_I + 12*m_yy_I*rho_I + 36*m_yz_I*rho_I + 147*m_zz_I*rho_I + 12*rho*uy0 + 12*rho*uz0)/(135*rho);

    rhoVar = rho;

    break;
case 0b0000010100000100:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = rho_I;

    m_xx_t45 = (m_xx_I*rho_I)/rho;
    m_xy_t90 = (m_xy_I*rho_I)/rho;
    m_xz_t90 = (m_xz_I*rho_I)/rho;
    m_yy_t45 = (m_yy_I*rho_I)/rho;
    m_yz_t90 = (m_yz_I*rho_I)/rho;
    m_zz_t45 = (m_zz_I*rho_I)/rho;

    rhoVar = rho;

    break;
case 0b0000010100000101:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (24*(15*rho_I + 2*m_xx_I*rho_I + 6*m_xz_I*rho_I - m_yy_I*rho_I + 2*m_zz_I*rho_I - 2*omegaVar*m_xx_I*rho_I - 6*omegaVar*m_xz_I*rho_I + omegaVar*m_yy_I*rho_I - 2*omegaVar*m_zz_I*rho_I))/(6*omegaVar + 12*ux0 + 12*uz0 + 18*omegaVar*ux0 + 18*omegaVar*uz0 - 30*omegaVar*ux0*ux0 + 15*omegaVar*uy0*uy0 - 30*omegaVar*uz0*uz0 - 90*omegaVar*ux0*uz0 + 344);

    m_xx_t45 = (4*rho + 147*m_xx_I*rho_I + 36*m_xz_I*rho_I - 6*m_yy_I*rho_I + 12*m_zz_I*rho_I + 12*rho*ux0 + 12*rho*uz0)/(135*rho);
    m_xy_t90 = (m_xy_I*rho_I)/rho;
    m_xz_t90 = (2*rho + 6*m_xx_I*rho_I + 63*m_xz_I*rho_I - 3*m_yy_I*rho_I + 6*m_zz_I*rho_I + 6*rho*ux0 + 6*rho*uz0)/(45*rho);
    m_yy_t45 = -(2*(rho + 3*m_xx_I*rho_I + 9*m_xz_I*rho_I - 69*m_yy_I*rho_I + 3*m_zz_I*rho_I + 3*rho*ux0 + 3*rho*uz0))/(135*rho);
    m_yz_t90 = (m_yz_I*rho_I)/rho;
    m_zz_t45 = (4*rho + 12*m_xx_I*rho_I + 36*m_xz_I*rho_I - 6*m_yy_I*rho_I + 147*m_zz_I*rho_I + 12*rho*ux0 + 12*rho*uz0)/(135*rho);

    rhoVar = rho;

    break;
case 0b0000010100000110:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = rho_I;

    m_xx_t45 = (m_xx_I*rho_I)/rho;
    m_xy_t90 = (m_xy_I*rho_I)/rho;
    m_xz_t90 = (m_xz_I*rho_I)/rho;
    m_yy_t45 = (m_yy_I*rho_I)/rho;
    m_yz_t90 = (m_yz_I*rho_I)/rho;
    m_zz_t45 = (m_zz_I*rho_I)/rho;

    rhoVar = rho;

    break;
case 0b0000010100000111:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[10] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (24*(15*rho_I + m_xx_I*rho_I + 6*m_xz_I*rho_I + m_yy_I*rho_I + 6*m_yz_I*rho_I + 4*m_zz_I*rho_I - omegaVar*m_xx_I*rho_I - 6*omegaVar*m_xz_I*rho_I - omegaVar*m_yy_I*rho_I - 6*omegaVar*m_yz_I*rho_I - 4*omegaVar*m_zz_I*rho_I))/(12*omegaVar + 12*ux0 + 12*uy0 + 24*uz0 + 18*omegaVar*ux0 + 18*omegaVar*uy0 + 36*omegaVar*uz0 - 15*omegaVar*ux0*ux0 - 15*omegaVar*uy0*uy0 - 60*omegaVar*uz0*uz0 - 90*omegaVar*ux0*uz0 - 90*omegaVar*uy0*uz0 + 328);

    m_xx_t45 = (2*(rho + 75*m_xx_I*rho_I + 18*m_xz_I*rho_I - 6*m_yy_I*rho_I - 9*m_yz_I*rho_I + 3*m_zz_I*rho_I + 6*rho*ux0 - 3*rho*uy0 + 3*rho*uz0))/(135*rho);
    m_xy_t90 = (m_xy_I*rho_I)/rho;
    m_xz_t90 = (2*rho + 6*m_xx_I*rho_I + 63*m_xz_I*rho_I - 3*m_yy_I*rho_I + 6*m_zz_I*rho_I + 6*rho*ux0 + 6*rho*uz0)/(45*rho);
    m_yy_t45 = (2*(rho - 6*m_xx_I*rho_I - 9*m_xz_I*rho_I + 75*m_yy_I*rho_I + 18*m_yz_I*rho_I + 3*m_zz_I*rho_I - 3*rho*ux0 + 6*rho*uy0 + 3*rho*uz0))/(135*rho);
    m_yz_t90 = (2*rho - 3*m_xx_I*rho_I + 6*m_yy_I*rho_I + 63*m_yz_I*rho_I + 6*m_zz_I*rho_I + 6*rho*uy0 + 6*rho*uz0)/(45*rho);
    m_zz_t45 = (8*rho + 6*m_xx_I*rho_I + 36*m_xz_I*rho_I + 6*m_yy_I*rho_I + 36*m_yz_I*rho_I + 159*m_zz_I*rho_I + 12*rho*ux0 + 12*rho*uy0 + 24*rho*uz0)/(135*rho);

    rhoVar = rho;

    break;
case 0b0000010100001000:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = rho_I;

    m_xx_t45 = (m_xx_I*rho_I)/rho;
    m_xy_t90 = (m_xy_I*rho_I)/rho;
    m_xz_t90 = (m_xz_I*rho_I)/rho;
    m_yy_t45 = (m_yy_I*rho_I)/rho;
    m_yz_t90 = (m_yz_I*rho_I)/rho;
    m_zz_t45 = (m_zz_I*rho_I)/rho;

    rhoVar = rho;

    break;
case 0b0000010100001001:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = rho_I;

    m_xx_t45 = (m_xx_I*rho_I)/rho;
    m_xy_t90 = (m_xy_I*rho_I)/rho;
    m_xz_t90 = (m_xz_I*rho_I)/rho;
    m_yy_t45 = (m_yy_I*rho_I)/rho;
    m_yz_t90 = (m_yz_I*rho_I)/rho;
    m_zz_t45 = (m_zz_I*rho_I)/rho;

    rhoVar = rho;

    break;
case 0b0000010100001010:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (24*(15*rho_I + 2*m_xx_I*rho_I - 6*m_xz_I*rho_I - m_yy_I*rho_I + 2*m_zz_I*rho_I - 2*omegaVar*m_xx_I*rho_I + 6*omegaVar*m_xz_I*rho_I + omegaVar*m_yy_I*rho_I - 2*omegaVar*m_zz_I*rho_I))/(6*omegaVar - 12*ux0 + 12*uz0 - 18*omegaVar*ux0 + 18*omegaVar*uz0 - 30*omegaVar*ux0*ux0 + 15*omegaVar*uy0*uy0 - 30*omegaVar*uz0*uz0 + 90*omegaVar*ux0*uz0 + 344);

    m_xx_t45 = (4*rho + 147*m_xx_I*rho_I - 36*m_xz_I*rho_I - 6*m_yy_I*rho_I + 12*m_zz_I*rho_I - 12*rho*ux0 + 12*rho*uz0)/(135*rho);
    m_xy_t90 = (m_xy_I*rho_I)/rho;
    m_xz_t90 = -(2*rho + 6*m_xx_I*rho_I - 63*m_xz_I*rho_I - 3*m_yy_I*rho_I + 6*m_zz_I*rho_I - 6*rho*ux0 + 6*rho*uz0)/(45*rho);
    m_yy_t45 = -(2*(rho + 3*m_xx_I*rho_I - 9*m_xz_I*rho_I - 69*m_yy_I*rho_I + 3*m_zz_I*rho_I - 3*rho*ux0 + 3*rho*uz0))/(135*rho);
    m_yz_t90 = (m_yz_I*rho_I)/rho;
    m_zz_t45 = (4*rho + 12*m_xx_I*rho_I - 36*m_xz_I*rho_I - 6*m_yy_I*rho_I + 147*m_zz_I*rho_I - 12*rho*ux0 + 12*rho*uz0)/(135*rho);

    rhoVar = rho;

    break;
case 0b0000010100001011:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[12] + pop[13] + pop[14] + pop[15] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (24*(15*rho_I + m_xx_I*rho_I - 6*m_xz_I*rho_I + m_yy_I*rho_I + 6*m_yz_I*rho_I + 4*m_zz_I*rho_I - omegaVar*m_xx_I*rho_I + 6*omegaVar*m_xz_I*rho_I - omegaVar*m_yy_I*rho_I - 6*omegaVar*m_yz_I*rho_I - 4*omegaVar*m_zz_I*rho_I))/(12*omegaVar - 12*ux0 + 12*uy0 + 24*uz0 - 18*omegaVar*ux0 + 18*omegaVar*uy0 + 36*omegaVar*uz0 - 15*omegaVar*ux0*ux0 - 15*omegaVar*uy0*uy0 - 60*omegaVar*uz0*uz0 + 90*omegaVar*ux0*uz0 - 90*omegaVar*uy0*uz0 + 328);

    m_xx_t45 = -(2*(18*m_xz_I*rho_I - 75*m_xx_I*rho_I - rho + 6*m_yy_I*rho_I + 9*m_yz_I*rho_I - 3*m_zz_I*rho_I + 6*rho*ux0 + 3*rho*uy0 - 3*rho*uz0))/(135*rho);
    m_xy_t90 = (m_xy_I*rho_I)/rho;
    m_xz_t90 = -(2*rho + 6*m_xx_I*rho_I - 63*m_xz_I*rho_I - 3*m_yy_I*rho_I + 6*m_zz_I*rho_I - 6*rho*ux0 + 6*rho*uz0)/(45*rho);
    m_yy_t45 = (2*(rho - 6*m_xx_I*rho_I + 9*m_xz_I*rho_I + 75*m_yy_I*rho_I + 18*m_yz_I*rho_I + 3*m_zz_I*rho_I + 3*rho*ux0 + 6*rho*uy0 + 3*rho*uz0))/(135*rho);
    m_yz_t90 = (2*rho - 3*m_xx_I*rho_I + 6*m_yy_I*rho_I + 63*m_yz_I*rho_I + 6*m_zz_I*rho_I + 6*rho*uy0 + 6*rho*uz0)/(45*rho);
    m_zz_t45 = (8*rho + 6*m_xx_I*rho_I - 36*m_xz_I*rho_I + 6*m_yy_I*rho_I + 36*m_yz_I*rho_I + 159*m_zz_I*rho_I - 12*rho*ux0 + 12*rho*uy0 + 24*rho*uz0)/(135*rho);

    rhoVar = rho;

    break;
case 0b0000010100001100:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17]);

    rho = (24*(15*rho_I - m_xx_I*rho_I + 2*m_yy_I*rho_I - 6*m_yz_I*rho_I + 2*m_zz_I*rho_I + omegaVar*m_xx_I*rho_I - 2*omegaVar*m_yy_I*rho_I + 6*omegaVar*m_yz_I*rho_I - 2*omegaVar*m_zz_I*rho_I))/(6*omegaVar - 12*uy0 + 12*uz0 - 18*omegaVar*uy0 + 18*omegaVar*uz0 + 15*omegaVar*ux0*ux0 - 30*omegaVar*uy0*uy0 - 30*omegaVar*uz0*uz0 + 90*omegaVar*uy0*uz0 + 344);

    m_xx_t45 = -(2*(rho - 69*m_xx_I*rho_I + 3*m_yy_I*rho_I - 9*m_yz_I*rho_I + 3*m_zz_I*rho_I - 3*rho*uy0 + 3*rho*uz0))/(135*rho);
    m_xy_t90 = (m_xy_I*rho_I)/rho;
    m_xz_t90 = (m_xz_I*rho_I)/rho;
    m_yy_t45 = (4*rho - 6*m_xx_I*rho_I + 147*m_yy_I*rho_I - 36*m_yz_I*rho_I + 12*m_zz_I*rho_I - 12*rho*uy0 + 12*rho*uz0)/(135*rho);
    m_yz_t90 = -(2*rho - 3*m_xx_I*rho_I + 6*m_yy_I*rho_I - 63*m_yz_I*rho_I + 6*m_zz_I*rho_I - 6*rho*uy0 + 6*rho*uz0)/(45*rho);
    m_zz_t45 = (4*rho - 6*m_xx_I*rho_I + 12*m_yy_I*rho_I - 36*m_yz_I*rho_I + 147*m_zz_I*rho_I - 12*rho*uy0 + 12*rho*uz0)/(135*rho);

    rhoVar = rho;

    break;
case 0b0000010100001101:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17]);

    rho = (24*(15*rho_I + m_xx_I*rho_I + 6*m_xz_I*rho_I + m_yy_I*rho_I - 6*m_yz_I*rho_I + 4*m_zz_I*rho_I - omegaVar*m_xx_I*rho_I - 6*omegaVar*m_xz_I*rho_I - omegaVar*m_yy_I*rho_I + 6*omegaVar*m_yz_I*rho_I - 4*omegaVar*m_zz_I*rho_I))/(12*omegaVar + 12*ux0 - 12*uy0 + 24*uz0 + 18*omegaVar*ux0 - 18*omegaVar*uy0 + 36*omegaVar*uz0 - 15*omegaVar*ux0*ux0 - 15*omegaVar*uy0*uy0 - 60*omegaVar*uz0*uz0 - 90*omegaVar*ux0*uz0 + 90*omegaVar*uy0*uz0 + 328);

    m_xx_t45 = (2*(rho + 75*m_xx_I*rho_I + 18*m_xz_I*rho_I - 6*m_yy_I*rho_I + 9*m_yz_I*rho_I + 3*m_zz_I*rho_I + 6*rho*ux0 + 3*rho*uy0 + 3*rho*uz0))/(135*rho);
    m_xy_t90 = (m_xy_I*rho_I)/rho;
    m_xz_t90 = (2*rho + 6*m_xx_I*rho_I + 63*m_xz_I*rho_I - 3*m_yy_I*rho_I + 6*m_zz_I*rho_I + 6*rho*ux0 + 6*rho*uz0)/(45*rho);
    m_yy_t45 = -(2*(6*m_xx_I*rho_I - rho + 9*m_xz_I*rho_I - 75*m_yy_I*rho_I + 18*m_yz_I*rho_I - 3*m_zz_I*rho_I + 3*rho*ux0 + 6*rho*uy0 - 3*rho*uz0))/(135*rho);
    m_yz_t90 = -(2*rho - 3*m_xx_I*rho_I + 6*m_yy_I*rho_I - 63*m_yz_I*rho_I + 6*m_zz_I*rho_I - 6*rho*uy0 + 6*rho*uz0)/(45*rho);
    m_zz_t45 = (8*rho + 6*m_xx_I*rho_I + 36*m_xz_I*rho_I + 6*m_yy_I*rho_I - 36*m_yz_I*rho_I + 159*m_zz_I*rho_I + 12*rho*ux0 - 12*rho*uy0 + 24*rho*uz0)/(135*rho);

    rhoVar = rho;

    break;
case 0b0000010100001110:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[17];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] - (1.0/3.0)*pop[17]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] + (2.0/3.0)*pop[17]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[17]);

    rho = -(24*(15*rho_I + m_xx_I*rho_I - 6*m_xz_I*rho_I + m_yy_I*rho_I - 6*m_yz_I*rho_I + 4*m_zz_I*rho_I - omegaVar*m_xx_I*rho_I + 6*omegaVar*m_xz_I*rho_I - omegaVar*m_yy_I*rho_I + 6*omegaVar*m_yz_I*rho_I - 4*omegaVar*m_zz_I*rho_I))/(12*ux0 - 12*omegaVar + 12*uy0 - 24*uz0 + 18*omegaVar*ux0 + 18*omegaVar*uy0 - 36*omegaVar*uz0 + 15*omegaVar*ux0*ux0 + 15*omegaVar*uy0*uy0 + 60*omegaVar*uz0*uz0 - 90*omegaVar*ux0*uz0 - 90*omegaVar*uy0*uz0 - 328);

    m_xx_t45 = (2*(rho + 75*m_xx_I*rho_I - 18*m_xz_I*rho_I - 6*m_yy_I*rho_I + 9*m_yz_I*rho_I + 3*m_zz_I*rho_I - 6*rho*ux0 + 3*rho*uy0 + 3*rho*uz0))/(135*rho);
    m_xy_t90 = (m_xy_I*rho_I)/rho;
    m_xz_t90 = -(2*rho + 6*m_xx_I*rho_I - 63*m_xz_I*rho_I - 3*m_yy_I*rho_I + 6*m_zz_I*rho_I - 6*rho*ux0 + 6*rho*uz0)/(45*rho);
    m_yy_t45 = (2*(rho - 6*m_xx_I*rho_I + 9*m_xz_I*rho_I + 75*m_yy_I*rho_I - 18*m_yz_I*rho_I + 3*m_zz_I*rho_I + 3*rho*ux0 - 6*rho*uy0 + 3*rho*uz0))/(135*rho);
    m_yz_t90 = -(2*rho - 3*m_xx_I*rho_I + 6*m_yy_I*rho_I - 63*m_yz_I*rho_I + 6*m_zz_I*rho_I - 6*rho*uy0 + 6*rho*uz0)/(45*rho);
    m_zz_t45 = (8*rho + 6*m_xx_I*rho_I - 36*m_xz_I*rho_I + 6*m_yy_I*rho_I - 36*m_yz_I*rho_I + 159*m_zz_I*rho_I - 12*rho*ux0 - 12*rho*uy0 + 24*rho*uz0)/(135*rho);

    rhoVar = rho;

    break;
case 0b0000010100001111:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[6] + pop[7] + pop[8] + pop[10] + pop[12] + pop[13] + pop[14] + pop[15] + pop[17];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] - (1.0/3.0)*pop[17]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[10] - pop[15]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] + (2.0/3.0)*pop[17]);
    m_yz_I = inv_rho_I * (pop[12] - pop[17]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[17]);

    rho = (3*rho_I*(3*m_zz_I - 3*omegaVar*m_zz_I + 4))/(omegaVar + 3*uz0 + 3*omegaVar*uz0 - 6*omegaVar*uz0*uz0 + 9);

    m_xx_t45 = (4*(10*m_xx_I*rho_I - m_yy_I*rho_I))/(33*rho);
    m_xy_t90 = (m_xy_I*rho_I)/rho;
    m_xz_t90 = (6*m_xz_I*rho_I + rho*ux0)/(3*rho);
    m_yy_t45 = -(4*(m_xx_I*rho_I - 10*m_yy_I*rho_I))/(33*rho);
    m_yz_t90 = (6*m_yz_I*rho_I + rho*uy0)/(3*rho);
    m_zz_t45 = (rho + 9*m_zz_I*rho_I + 3*rho*uz0)/(6*rho);

    rhoVar = rho;

    break;
case 0b0000010100010000:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = rho_I;

    m_xx_t45 = (m_xx_I*rho_I)/rho;
    m_xy_t90 = (m_xy_I*rho_I)/rho;
    m_xz_t90 = (m_xz_I*rho_I)/rho;
    m_yy_t45 = (m_yy_I*rho_I)/rho;
    m_yz_t90 = (m_yz_I*rho_I)/rho;
    m_zz_t45 = (m_zz_I*rho_I)/rho;

    rhoVar = rho;

    break;
case 0b0000010100010001:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (24*(15*rho_I + 2*m_xx_I*rho_I + 6*m_xy_I*rho_I + 2*m_yy_I*rho_I - m_zz_I*rho_I - 2*omegaVar*m_xx_I*rho_I - 6*omegaVar*m_xy_I*rho_I - 2*omegaVar*m_yy_I*rho_I + omegaVar*m_zz_I*rho_I))/(6*omegaVar + 12*ux0 + 12*uy0 + 18*omegaVar*ux0 + 18*omegaVar*uy0 - 30*omegaVar*ux0*ux0 - 30*omegaVar*uy0*uy0 + 15*omegaVar*uz0*uz0 - 90*omegaVar*ux0*uy0 + 344);

    m_xx_t45 = (4*rho + 147*m_xx_I*rho_I + 36*m_xy_I*rho_I + 12*m_yy_I*rho_I - 6*m_zz_I*rho_I + 12*rho*ux0 + 12*rho*uy0)/(135*rho);
    m_xy_t90 = (2*rho + 6*m_xx_I*rho_I + 63*m_xy_I*rho_I + 6*m_yy_I*rho_I - 3*m_zz_I*rho_I + 6*rho*ux0 + 6*rho*uy0)/(45*rho);
    m_xz_t90 = (m_xz_I*rho_I)/rho;
    m_yy_t45 = (4*rho + 12*m_xx_I*rho_I + 36*m_xy_I*rho_I + 147*m_yy_I*rho_I - 6*m_zz_I*rho_I + 12*rho*ux0 + 12*rho*uy0)/(135*rho);
    m_yz_t90 = (m_yz_I*rho_I)/rho;
    m_zz_t45 = -(2*(rho + 3*m_xx_I*rho_I + 9*m_xy_I*rho_I + 3*m_yy_I*rho_I - 69*m_zz_I*rho_I + 3*rho*ux0 + 3*rho*uy0))/(135*rho);

    rhoVar = rho;

    break;
case 0b0000010100010010:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = rho_I;

    m_xx_t45 = (m_xx_I*rho_I)/rho;
    m_xy_t90 = (m_xy_I*rho_I)/rho;
    m_xz_t90 = (m_xz_I*rho_I)/rho;
    m_yy_t45 = (m_yy_I*rho_I)/rho;
    m_yz_t90 = (m_yz_I*rho_I)/rho;
    m_zz_t45 = (m_zz_I*rho_I)/rho;

    rhoVar = rho;

    break;
case 0b0000010100010011:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[8] + pop[9] + pop[10] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (24*(15*rho_I + m_xx_I*rho_I + 6*m_xy_I*rho_I + 4*m_yy_I*rho_I + 6*m_yz_I*rho_I + m_zz_I*rho_I - omegaVar*m_xx_I*rho_I - 6*omegaVar*m_xy_I*rho_I - 4*omegaVar*m_yy_I*rho_I - 6*omegaVar*m_yz_I*rho_I - omegaVar*m_zz_I*rho_I))/(12*omegaVar + 12*ux0 + 24*uy0 + 12*uz0 + 18*omegaVar*ux0 + 36*omegaVar*uy0 + 18*omegaVar*uz0 - 15*omegaVar*ux0*ux0 - 60*omegaVar*uy0*uy0 - 15*omegaVar*uz0*uz0 - 90*omegaVar*ux0*uy0 - 90*omegaVar*uy0*uz0 + 328);

    m_xx_t45 = (2*(rho + 75*m_xx_I*rho_I + 18*m_xy_I*rho_I + 3*m_yy_I*rho_I - 9*m_yz_I*rho_I - 6*m_zz_I*rho_I + 6*rho*ux0 + 3*rho*uy0 - 3*rho*uz0))/(135*rho);
    m_xy_t90 = (2*rho + 6*m_xx_I*rho_I + 63*m_xy_I*rho_I + 6*m_yy_I*rho_I - 3*m_zz_I*rho_I + 6*rho*ux0 + 6*rho*uy0)/(45*rho);
    m_xz_t90 = (m_xz_I*rho_I)/rho;
    m_yy_t45 = (8*rho + 6*m_xx_I*rho_I + 36*m_xy_I*rho_I + 159*m_yy_I*rho_I + 36*m_yz_I*rho_I + 6*m_zz_I*rho_I + 12*rho*ux0 + 24*rho*uy0 + 12*rho*uz0)/(135*rho);
    m_yz_t90 = (2*rho - 3*m_xx_I*rho_I + 6*m_yy_I*rho_I + 63*m_yz_I*rho_I + 6*m_zz_I*rho_I + 6*rho*uy0 + 6*rho*uz0)/(45*rho);
    m_zz_t45 = (2*(rho - 6*m_xx_I*rho_I - 9*m_xy_I*rho_I + 3*m_yy_I*rho_I + 18*m_yz_I*rho_I + 75*m_zz_I*rho_I - 3*rho*ux0 + 3*rho*uy0 + 6*rho*uz0))/(135*rho);

    rhoVar = rho;

    break;
case 0b0000010100010100:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = rho_I;

    m_xx_t45 = (m_xx_I*rho_I)/rho;
    m_xy_t90 = (m_xy_I*rho_I)/rho;
    m_xz_t90 = (m_xz_I*rho_I)/rho;
    m_yy_t45 = (m_yy_I*rho_I)/rho;
    m_yz_t90 = (m_yz_I*rho_I)/rho;
    m_zz_t45 = (m_zz_I*rho_I)/rho;

    rhoVar = rho;

    break;
case 0b0000010100010101:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[8] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (24*(15*rho_I + 4*m_xx_I*rho_I + 6*m_xy_I*rho_I + 6*m_xz_I*rho_I + m_yy_I*rho_I + m_zz_I*rho_I - 4*omegaVar*m_xx_I*rho_I - 6*omegaVar*m_xy_I*rho_I - 6*omegaVar*m_xz_I*rho_I - omegaVar*m_yy_I*rho_I - omegaVar*m_zz_I*rho_I))/(12*omegaVar + 24*ux0 + 12*uy0 + 12*uz0 + 36*omegaVar*ux0 + 18*omegaVar*uy0 + 18*omegaVar*uz0 - 60*omegaVar*ux0*ux0 - 15*omegaVar*uy0*uy0 - 15*omegaVar*uz0*uz0 - 90*omegaVar*ux0*uy0 - 90*omegaVar*ux0*uz0 + 328);

    m_xx_t45 = (8*rho + 159*m_xx_I*rho_I + 36*m_xy_I*rho_I + 36*m_xz_I*rho_I + 6*m_yy_I*rho_I + 6*m_zz_I*rho_I + 24*rho*ux0 + 12*rho*uy0 + 12*rho*uz0)/(135*rho);
    m_xy_t90 = (2*rho + 6*m_xx_I*rho_I + 63*m_xy_I*rho_I + 6*m_yy_I*rho_I - 3*m_zz_I*rho_I + 6*rho*ux0 + 6*rho*uy0)/(45*rho);
    m_xz_t90 = (2*rho + 6*m_xx_I*rho_I + 63*m_xz_I*rho_I - 3*m_yy_I*rho_I + 6*m_zz_I*rho_I + 6*rho*ux0 + 6*rho*uz0)/(45*rho);
    m_yy_t45 = (2*(rho + 3*m_xx_I*rho_I + 18*m_xy_I*rho_I - 9*m_xz_I*rho_I + 75*m_yy_I*rho_I - 6*m_zz_I*rho_I + 3*rho*ux0 + 6*rho*uy0 - 3*rho*uz0))/(135*rho);
    m_yz_t90 = (m_yz_I*rho_I)/rho;
    m_zz_t45 = (2*(rho + 3*m_xx_I*rho_I - 9*m_xy_I*rho_I + 18*m_xz_I*rho_I - 6*m_yy_I*rho_I + 75*m_zz_I*rho_I + 3*rho*ux0 - 3*rho*uy0 + 6*rho*uz0))/(135*rho);

    rhoVar = rho;

    break;
case 0b0000010100010110:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = rho_I;

    m_xx_t45 = (m_xx_I*rho_I)/rho;
    m_xy_t90 = (m_xy_I*rho_I)/rho;
    m_xz_t90 = (m_xz_I*rho_I)/rho;
    m_yy_t45 = (m_yy_I*rho_I)/rho;
    m_yz_t90 = (m_yz_I*rho_I)/rho;
    m_zz_t45 = (m_zz_I*rho_I)/rho;

    rhoVar = rho;

    break;
case 0b0000010100010111:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[8] + pop[10] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (24*(5*rho_I + m_xx_I*rho_I + 2*m_xy_I*rho_I + 2*m_xz_I*rho_I + m_yy_I*rho_I + 2*m_yz_I*rho_I + m_zz_I*rho_I - omegaVar*m_xx_I*rho_I - 2*omegaVar*m_xy_I*rho_I - 2*omegaVar*m_xz_I*rho_I - omegaVar*m_yy_I*rho_I - 2*omegaVar*m_yz_I*rho_I - omegaVar*m_zz_I*rho_I))/(6*omegaVar + 8*ux0 + 8*uy0 + 8*uz0 + 12*omegaVar*ux0 + 12*omegaVar*uy0 + 12*omegaVar*uz0 - 15*omegaVar*ux0*ux0 - 15*omegaVar*uy0*uy0 - 15*omegaVar*uz0*uz0 - 30*omegaVar*ux0*uy0 - 30*omegaVar*ux0*uz0 - 30*omegaVar*uy0*uz0 + 104);

    m_xx_t45 = (2*(rho + 27*m_xx_I*rho_I + 6*m_xy_I*rho_I + 6*m_xz_I*rho_I - 3*m_yz_I*rho_I + 4*rho*ux0 + rho*uy0 + rho*uz0))/(45*rho);
    m_xy_t90 = (2*rho + 6*m_xx_I*rho_I + 63*m_xy_I*rho_I + 6*m_yy_I*rho_I - 3*m_zz_I*rho_I + 6*rho*ux0 + 6*rho*uy0)/(45*rho);
    m_xz_t90 = (2*rho + 6*m_xx_I*rho_I + 63*m_xz_I*rho_I - 3*m_yy_I*rho_I + 6*m_zz_I*rho_I + 6*rho*ux0 + 6*rho*uz0)/(45*rho);
    m_yy_t45 = (2*(rho + 6*m_xy_I*rho_I - 3*m_xz_I*rho_I + 27*m_yy_I*rho_I + 6*m_yz_I*rho_I + rho*ux0 + 4*rho*uy0 + rho*uz0))/(45*rho);
    m_yz_t90 = (2*rho - 3*m_xx_I*rho_I + 6*m_yy_I*rho_I + 63*m_yz_I*rho_I + 6*m_zz_I*rho_I + 6*rho*uy0 + 6*rho*uz0)/(45*rho);
    m_zz_t45 = (2*(rho - 3*m_xy_I*rho_I + 6*m_xz_I*rho_I + 6*m_yz_I*rho_I + 27*m_zz_I*rho_I + rho*ux0 + rho*uy0 + 4*rho*uz0))/(45*rho);

    rhoVar = rho;

    break;
case 0b0000010100011000:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = rho_I;

    m_xx_t45 = (m_xx_I*rho_I)/rho;
    m_xy_t90 = (m_xy_I*rho_I)/rho;
    m_xz_t90 = (m_xz_I*rho_I)/rho;
    m_yy_t45 = (m_yy_I*rho_I)/rho;
    m_yz_t90 = (m_yz_I*rho_I)/rho;
    m_zz_t45 = (m_zz_I*rho_I)/rho;

    rhoVar = rho;

    break;
case 0b0000010100011001:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (24*(15*rho_I + 2*m_xx_I*rho_I + 6*m_xy_I*rho_I + 2*m_yy_I*rho_I - m_zz_I*rho_I - 2*omegaVar*m_xx_I*rho_I - 6*omegaVar*m_xy_I*rho_I - 2*omegaVar*m_yy_I*rho_I + omegaVar*m_zz_I*rho_I))/(6*omegaVar + 12*ux0 + 12*uy0 + 18*omegaVar*ux0 + 18*omegaVar*uy0 - 30*omegaVar*ux0*ux0 - 30*omegaVar*uy0*uy0 + 15*omegaVar*uz0*uz0 - 90*omegaVar*ux0*uy0 + 344);

    m_xx_t45 = (4*rho + 147*m_xx_I*rho_I + 36*m_xy_I*rho_I + 12*m_yy_I*rho_I - 6*m_zz_I*rho_I + 12*rho*ux0 + 12*rho*uy0)/(135*rho);
    m_xy_t90 = (2*rho + 6*m_xx_I*rho_I + 63*m_xy_I*rho_I + 6*m_yy_I*rho_I - 3*m_zz_I*rho_I + 6*rho*ux0 + 6*rho*uy0)/(45*rho);
    m_xz_t90 = (m_xz_I*rho_I)/rho;
    m_yy_t45 = (4*rho + 12*m_xx_I*rho_I + 36*m_xy_I*rho_I + 147*m_yy_I*rho_I - 6*m_zz_I*rho_I + 12*rho*ux0 + 12*rho*uy0)/(135*rho);
    m_yz_t90 = (m_yz_I*rho_I)/rho;
    m_zz_t45 = -(2*(rho + 3*m_xx_I*rho_I + 9*m_xy_I*rho_I + 3*m_yy_I*rho_I - 69*m_zz_I*rho_I + 3*rho*ux0 + 3*rho*uy0))/(135*rho);

    rhoVar = rho;

    break;
case 0b0000010100011010:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (24*(15*rho_I + 2*m_xx_I*rho_I - 6*m_xz_I*rho_I - m_yy_I*rho_I + 2*m_zz_I*rho_I - 2*omegaVar*m_xx_I*rho_I + 6*omegaVar*m_xz_I*rho_I + omegaVar*m_yy_I*rho_I - 2*omegaVar*m_zz_I*rho_I))/(6*omegaVar - 12*ux0 + 12*uz0 - 18*omegaVar*ux0 + 18*omegaVar*uz0 - 30*omegaVar*ux0*ux0 + 15*omegaVar*uy0*uy0 - 30*omegaVar*uz0*uz0 + 90*omegaVar*ux0*uz0 + 344);

    m_xx_t45 = (4*rho + 147*m_xx_I*rho_I - 36*m_xz_I*rho_I - 6*m_yy_I*rho_I + 12*m_zz_I*rho_I - 12*rho*ux0 + 12*rho*uz0)/(135*rho);
    m_xy_t90 = (m_xy_I*rho_I)/rho;
    m_xz_t90 = -(2*rho + 6*m_xx_I*rho_I - 63*m_xz_I*rho_I - 3*m_yy_I*rho_I + 6*m_zz_I*rho_I - 6*rho*ux0 + 6*rho*uz0)/(45*rho);
    m_yy_t45 = -(2*(rho + 3*m_xx_I*rho_I - 9*m_xz_I*rho_I - 69*m_yy_I*rho_I + 3*m_zz_I*rho_I - 3*rho*ux0 + 3*rho*uz0))/(135*rho);
    m_yz_t90 = (m_yz_I*rho_I)/rho;
    m_zz_t45 = (4*rho + 12*m_xx_I*rho_I - 36*m_xz_I*rho_I - 6*m_yy_I*rho_I + 147*m_zz_I*rho_I - 12*rho*ux0 + 12*rho*uz0)/(135*rho);

    rhoVar = rho;

    break;
case 0b0000010100011011:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[8] + pop[9] + pop[10] + pop[12] + pop[13] + pop[14] + pop[15] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (24*(5*rho_I + m_xx_I*rho_I + 2*m_xy_I*rho_I - 2*m_xz_I*rho_I + m_yy_I*rho_I + 2*m_yz_I*rho_I + m_zz_I*rho_I - omegaVar*m_xx_I*rho_I - 2*omegaVar*m_xy_I*rho_I + 2*omegaVar*m_xz_I*rho_I - omegaVar*m_yy_I*rho_I - 2*omegaVar*m_yz_I*rho_I - omegaVar*m_zz_I*rho_I))/(6*omegaVar + 8*uy0 + 8*uz0 + 12*omegaVar*uy0 + 12*omegaVar*uz0 - 15*omegaVar*ux0*ux0 - 15*omegaVar*uy0*uy0 - 15*omegaVar*uz0*uz0 - 30*omegaVar*ux0*uy0 + 30*omegaVar*ux0*uz0 - 30*omegaVar*uy0*uz0 + 104);

    m_xx_t45 = (2*(rho + 27*m_xx_I*rho_I + 6*m_xy_I*rho_I - 6*m_xz_I*rho_I - 3*m_yz_I*rho_I + rho*uy0 + rho*uz0))/(45*rho);
    m_xy_t90 = (2*rho + 6*m_xx_I*rho_I + 63*m_xy_I*rho_I + 6*m_yy_I*rho_I - 3*m_zz_I*rho_I + 6*rho*ux0 + 6*rho*uy0)/(45*rho);
    m_xz_t90 = -(2*rho + 6*m_xx_I*rho_I - 63*m_xz_I*rho_I - 3*m_yy_I*rho_I + 6*m_zz_I*rho_I - 6*rho*ux0 + 6*rho*uz0)/(45*rho);
    m_yy_t45 = (2*(rho + 6*m_xy_I*rho_I + 3*m_xz_I*rho_I + 27*m_yy_I*rho_I + 6*m_yz_I*rho_I + 3*rho*ux0 + 4*rho*uy0 + rho*uz0))/(45*rho);
    m_yz_t90 = (2*rho - 3*m_xx_I*rho_I + 6*m_yy_I*rho_I + 63*m_yz_I*rho_I + 6*m_zz_I*rho_I + 6*rho*uy0 + 6*rho*uz0)/(45*rho);
    m_zz_t45 = (2*(rho - 3*m_xy_I*rho_I - 6*m_xz_I*rho_I + 6*m_yz_I*rho_I + 27*m_zz_I*rho_I - 3*rho*ux0 + rho*uy0 + 4*rho*uz0))/(45*rho);

    rhoVar = rho;

    break;
case 0b0000010100011100:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17]);

    rho = (24*(15*rho_I - m_xx_I*rho_I + 2*m_yy_I*rho_I - 6*m_yz_I*rho_I + 2*m_zz_I*rho_I + omegaVar*m_xx_I*rho_I - 2*omegaVar*m_yy_I*rho_I + 6*omegaVar*m_yz_I*rho_I - 2*omegaVar*m_zz_I*rho_I))/(6*omegaVar - 12*uy0 + 12*uz0 - 18*omegaVar*uy0 + 18*omegaVar*uz0 + 15*omegaVar*ux0*ux0 - 30*omegaVar*uy0*uy0 - 30*omegaVar*uz0*uz0 + 90*omegaVar*uy0*uz0 + 344);

    m_xx_t45 = -(2*(rho - 69*m_xx_I*rho_I + 3*m_yy_I*rho_I - 9*m_yz_I*rho_I + 3*m_zz_I*rho_I - 3*rho*uy0 + 3*rho*uz0))/(135*rho);
    m_xy_t90 = (m_xy_I*rho_I)/rho;
    m_xz_t90 = (m_xz_I*rho_I)/rho;
    m_yy_t45 = (4*rho - 6*m_xx_I*rho_I + 147*m_yy_I*rho_I - 36*m_yz_I*rho_I + 12*m_zz_I*rho_I - 12*rho*uy0 + 12*rho*uz0)/(135*rho);
    m_yz_t90 = -(2*rho - 3*m_xx_I*rho_I + 6*m_yy_I*rho_I - 63*m_yz_I*rho_I + 6*m_zz_I*rho_I - 6*rho*uy0 + 6*rho*uz0)/(45*rho);
    m_zz_t45 = (4*rho - 6*m_xx_I*rho_I + 12*m_yy_I*rho_I - 36*m_yz_I*rho_I + 147*m_zz_I*rho_I - 12*rho*uy0 + 12*rho*uz0)/(135*rho);

    rhoVar = rho;

    break;
case 0b0000010100011101:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[8] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17]);
    m_xy_I = inv_rho_I * (pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17]);

    rho = (24*(5*rho_I + m_xx_I*rho_I + 2*m_xy_I*rho_I + 2*m_xz_I*rho_I + m_yy_I*rho_I - 2*m_yz_I*rho_I + m_zz_I*rho_I - omegaVar*m_xx_I*rho_I - 2*omegaVar*m_xy_I*rho_I - 2*omegaVar*m_xz_I*rho_I - omegaVar*m_yy_I*rho_I + 2*omegaVar*m_yz_I*rho_I - omegaVar*m_zz_I*rho_I))/(6*omegaVar + 8*ux0 + 8*uz0 + 12*omegaVar*ux0 + 12*omegaVar*uz0 - 15*omegaVar*ux0*ux0 - 15*omegaVar*uy0*uy0 - 15*omegaVar*uz0*uz0 - 30*omegaVar*ux0*uy0 - 30*omegaVar*ux0*uz0 + 30*omegaVar*uy0*uz0 + 104);

    m_xx_t45 = (2*(rho + 27*m_xx_I*rho_I + 6*m_xy_I*rho_I + 6*m_xz_I*rho_I + 3*m_yz_I*rho_I + 4*rho*ux0 + 3*rho*uy0 + rho*uz0))/(45*rho);
    m_xy_t90 = (2*rho + 6*m_xx_I*rho_I + 63*m_xy_I*rho_I + 6*m_yy_I*rho_I - 3*m_zz_I*rho_I + 6*rho*ux0 + 6*rho*uy0)/(45*rho);
    m_xz_t90 = (2*rho + 6*m_xx_I*rho_I + 63*m_xz_I*rho_I - 3*m_yy_I*rho_I + 6*m_zz_I*rho_I + 6*rho*ux0 + 6*rho*uz0)/(45*rho);
    m_yy_t45 = (2*(rho + 6*m_xy_I*rho_I - 3*m_xz_I*rho_I + 27*m_yy_I*rho_I - 6*m_yz_I*rho_I + rho*ux0 + rho*uz0))/(45*rho);
    m_yz_t90 = -(2*rho - 3*m_xx_I*rho_I + 6*m_yy_I*rho_I - 63*m_yz_I*rho_I + 6*m_zz_I*rho_I - 6*rho*uy0 + 6*rho*uz0)/(45*rho);
    m_zz_t45 = (2*(rho - 3*m_xy_I*rho_I + 6*m_xz_I*rho_I - 6*m_yz_I*rho_I + 27*m_zz_I*rho_I + rho*ux0 - 3*rho*uy0 + 4*rho*uz0))/(45*rho);

    rhoVar = rho;

    break;
case 0b0000010100011110:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[17];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] - (1.0/3.0)*pop[17]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] + (2.0/3.0)*pop[17]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[17]);

    rho = -(24*(15*rho_I + m_xx_I*rho_I - 6*m_xz_I*rho_I + m_yy_I*rho_I - 6*m_yz_I*rho_I + 4*m_zz_I*rho_I - omegaVar*m_xx_I*rho_I + 6*omegaVar*m_xz_I*rho_I - omegaVar*m_yy_I*rho_I + 6*omegaVar*m_yz_I*rho_I - 4*omegaVar*m_zz_I*rho_I))/(12*ux0 - 12*omegaVar + 12*uy0 - 24*uz0 + 18*omegaVar*ux0 + 18*omegaVar*uy0 - 36*omegaVar*uz0 + 15*omegaVar*ux0*ux0 + 15*omegaVar*uy0*uy0 + 60*omegaVar*uz0*uz0 - 90*omegaVar*ux0*uz0 - 90*omegaVar*uy0*uz0 - 328);

    m_xx_t45 = (2*(rho + 75*m_xx_I*rho_I - 18*m_xz_I*rho_I - 6*m_yy_I*rho_I + 9*m_yz_I*rho_I + 3*m_zz_I*rho_I - 6*rho*ux0 + 3*rho*uy0 + 3*rho*uz0))/(135*rho);
    m_xy_t90 = (m_xy_I*rho_I)/rho;
    m_xz_t90 = -(2*rho + 6*m_xx_I*rho_I - 63*m_xz_I*rho_I - 3*m_yy_I*rho_I + 6*m_zz_I*rho_I - 6*rho*ux0 + 6*rho*uz0)/(45*rho);
    m_yy_t45 = (2*(rho - 6*m_xx_I*rho_I + 9*m_xz_I*rho_I + 75*m_yy_I*rho_I - 18*m_yz_I*rho_I + 3*m_zz_I*rho_I + 3*rho*ux0 - 6*rho*uy0 + 3*rho*uz0))/(135*rho);
    m_yz_t90 = -(2*rho - 3*m_xx_I*rho_I + 6*m_yy_I*rho_I - 63*m_yz_I*rho_I + 6*m_zz_I*rho_I - 6*rho*uy0 + 6*rho*uz0)/(45*rho);
    m_zz_t45 = (8*rho + 6*m_xx_I*rho_I - 36*m_xz_I*rho_I + 6*m_yy_I*rho_I - 36*m_yz_I*rho_I + 159*m_zz_I*rho_I - 12*rho*ux0 - 12*rho*uy0 + 24*rho*uz0)/(135*rho);

    rhoVar = rho;

    break;
case 0b0000010100011111:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[6] + pop[8] + pop[10] + pop[12] + pop[13] + pop[14] + pop[15] + pop[17];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] - (1.0/3.0)*pop[17]);
    m_xy_I = inv_rho_I * (pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[10] - pop[15]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] + (2.0/3.0)*pop[17]);
    m_yz_I = inv_rho_I * (pop[12] - pop[17]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[17]);

    rho = (72*(107*rho_I + 12*m_xx_I*rho_I + 33*m_xy_I*rho_I + 12*m_yy_I*rho_I + 72*m_zz_I*rho_I - 12*omegaVar*m_xx_I*rho_I - 33*omegaVar*m_xy_I*rho_I - 12*omegaVar*m_yy_I*rho_I - 72*omegaVar*m_zz_I*rho_I))/(626*omegaVar + 492*ux0 + 492*uy0 + 2124*uz0 + 150*omegaVar*ux0 + 150*omegaVar*uy0 + 1728*omegaVar*uz0 - 642*omegaVar*ux0*ux0 - 642*omegaVar*uy0*uy0 - 3531*omegaVar*uz0*uz0 - 1926*omegaVar*ux0*uy0 + 5580);

    m_xx_t45 = (4*(2*rho + 106*m_xx_I*rho_I + 24*m_xy_I*rho_I - m_yy_I*rho_I - 6*m_zz_I*rho_I + 8*rho*ux0 + 8*rho*uy0 - 2*rho*uz0))/(321*rho);
    m_xy_t90 = (11*rho + 48*m_xx_I*rho_I + 453*m_xy_I*rho_I + 48*m_yy_I*rho_I - 33*m_zz_I*rho_I + 44*rho*ux0 + 44*rho*uy0 - 11*rho*uz0)/(321*rho);
    m_xz_t90 = (6*m_xz_I*rho_I + rho*ux0)/(3*rho);
    m_yy_t45 = (4*(2*rho - m_xx_I*rho_I + 24*m_xy_I*rho_I + 106*m_yy_I*rho_I - 6*m_zz_I*rho_I + 8*rho*ux0 + 8*rho*uy0 - 2*rho*uz0))/(321*rho);
    m_yz_t90 = (6*m_yz_I*rho_I + rho*uy0)/(3*rho);
    m_zz_t45 = -(2*(12*m_xx_I*rho_I - 24*rho + 33*m_xy_I*rho_I + 12*m_yy_I*rho_I - 249*m_zz_I*rho_I + 11*rho*ux0 + 11*rho*uy0 - 83*rho*uz0))/(321*rho);

    rhoVar = rho;

    break;
case 0b0000010100100000:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = rho_I;

    m_xx_t45 = (m_xx_I*rho_I)/rho;
    m_xy_t90 = (m_xy_I*rho_I)/rho;
    m_xz_t90 = (m_xz_I*rho_I)/rho;
    m_yy_t45 = (m_yy_I*rho_I)/rho;
    m_yz_t90 = (m_yz_I*rho_I)/rho;
    m_zz_t45 = (m_zz_I*rho_I)/rho;

    rhoVar = rho;

    break;
case 0b0000010100100001:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = rho_I;

    m_xx_t45 = (m_xx_I*rho_I)/rho;
    m_xy_t90 = (m_xy_I*rho_I)/rho;
    m_xz_t90 = (m_xz_I*rho_I)/rho;
    m_yy_t45 = (m_yy_I*rho_I)/rho;
    m_yz_t90 = (m_yz_I*rho_I)/rho;
    m_zz_t45 = (m_zz_I*rho_I)/rho;

    rhoVar = rho;

    break;
case 0b0000010100100010:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (24*(15*rho_I + 2*m_xx_I*rho_I - 6*m_xy_I*rho_I + 2*m_yy_I*rho_I - m_zz_I*rho_I - 2*omegaVar*m_xx_I*rho_I + 6*omegaVar*m_xy_I*rho_I - 2*omegaVar*m_yy_I*rho_I + omegaVar*m_zz_I*rho_I))/(6*omegaVar - 12*ux0 + 12*uy0 - 18*omegaVar*ux0 + 18*omegaVar*uy0 - 30*omegaVar*ux0*ux0 - 30*omegaVar*uy0*uy0 + 15*omegaVar*uz0*uz0 + 90*omegaVar*ux0*uy0 + 344);

    m_xx_t45 = (4*rho + 147*m_xx_I*rho_I - 36*m_xy_I*rho_I + 12*m_yy_I*rho_I - 6*m_zz_I*rho_I - 12*rho*ux0 + 12*rho*uy0)/(135*rho);
    m_xy_t90 = -(2*rho + 6*m_xx_I*rho_I - 63*m_xy_I*rho_I + 6*m_yy_I*rho_I - 3*m_zz_I*rho_I - 6*rho*ux0 + 6*rho*uy0)/(45*rho);
    m_xz_t90 = (m_xz_I*rho_I)/rho;
    m_yy_t45 = (4*rho + 12*m_xx_I*rho_I - 36*m_xy_I*rho_I + 147*m_yy_I*rho_I - 6*m_zz_I*rho_I - 12*rho*ux0 + 12*rho*uy0)/(135*rho);
    m_yz_t90 = (m_yz_I*rho_I)/rho;
    m_zz_t45 = -(2*(rho + 3*m_xx_I*rho_I - 9*m_xy_I*rho_I + 3*m_yy_I*rho_I - 69*m_zz_I*rho_I - 3*rho*ux0 + 3*rho*uy0))/(135*rho);

    rhoVar = rho;

    break;
case 0b0000010100100011:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[12] + pop[13] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (24*(15*rho_I + m_xx_I*rho_I - 6*m_xy_I*rho_I + 4*m_yy_I*rho_I + 6*m_yz_I*rho_I + m_zz_I*rho_I - omegaVar*m_xx_I*rho_I + 6*omegaVar*m_xy_I*rho_I - 4*omegaVar*m_yy_I*rho_I - 6*omegaVar*m_yz_I*rho_I - omegaVar*m_zz_I*rho_I))/(12*omegaVar - 12*ux0 + 24*uy0 + 12*uz0 - 18*omegaVar*ux0 + 36*omegaVar*uy0 + 18*omegaVar*uz0 - 15*omegaVar*ux0*ux0 - 60*omegaVar*uy0*uy0 - 15*omegaVar*uz0*uz0 + 90*omegaVar*ux0*uy0 - 90*omegaVar*uy0*uz0 + 328);

    m_xx_t45 = -(2*(18*m_xy_I*rho_I - 75*m_xx_I*rho_I - rho - 3*m_yy_I*rho_I + 9*m_yz_I*rho_I + 6*m_zz_I*rho_I + 6*rho*ux0 - 3*rho*uy0 + 3*rho*uz0))/(135*rho);
    m_xy_t90 = -(2*rho + 6*m_xx_I*rho_I - 63*m_xy_I*rho_I + 6*m_yy_I*rho_I - 3*m_zz_I*rho_I - 6*rho*ux0 + 6*rho*uy0)/(45*rho);
    m_xz_t90 = (m_xz_I*rho_I)/rho;
    m_yy_t45 = (8*rho + 6*m_xx_I*rho_I - 36*m_xy_I*rho_I + 159*m_yy_I*rho_I + 36*m_yz_I*rho_I + 6*m_zz_I*rho_I - 12*rho*ux0 + 24*rho*uy0 + 12*rho*uz0)/(135*rho);
    m_yz_t90 = (2*rho - 3*m_xx_I*rho_I + 6*m_yy_I*rho_I + 63*m_yz_I*rho_I + 6*m_zz_I*rho_I + 6*rho*uy0 + 6*rho*uz0)/(45*rho);
    m_zz_t45 = (2*(rho - 6*m_xx_I*rho_I + 9*m_xy_I*rho_I + 3*m_yy_I*rho_I + 18*m_yz_I*rho_I + 75*m_zz_I*rho_I + 3*rho*ux0 + 3*rho*uy0 + 6*rho*uz0))/(135*rho);

    rhoVar = rho;

    break;
case 0b0000010100100100:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = rho_I;

    m_xx_t45 = (m_xx_I*rho_I)/rho;
    m_xy_t90 = (m_xy_I*rho_I)/rho;
    m_xz_t90 = (m_xz_I*rho_I)/rho;
    m_yy_t45 = (m_yy_I*rho_I)/rho;
    m_yz_t90 = (m_yz_I*rho_I)/rho;
    m_zz_t45 = (m_zz_I*rho_I)/rho;

    rhoVar = rho;

    break;
case 0b0000010100100101:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (24*(15*rho_I + 2*m_xx_I*rho_I + 6*m_xz_I*rho_I - m_yy_I*rho_I + 2*m_zz_I*rho_I - 2*omegaVar*m_xx_I*rho_I - 6*omegaVar*m_xz_I*rho_I + omegaVar*m_yy_I*rho_I - 2*omegaVar*m_zz_I*rho_I))/(6*omegaVar + 12*ux0 + 12*uz0 + 18*omegaVar*ux0 + 18*omegaVar*uz0 - 30*omegaVar*ux0*ux0 + 15*omegaVar*uy0*uy0 - 30*omegaVar*uz0*uz0 - 90*omegaVar*ux0*uz0 + 344);

    m_xx_t45 = (4*rho + 147*m_xx_I*rho_I + 36*m_xz_I*rho_I - 6*m_yy_I*rho_I + 12*m_zz_I*rho_I + 12*rho*ux0 + 12*rho*uz0)/(135*rho);
    m_xy_t90 = (m_xy_I*rho_I)/rho;
    m_xz_t90 = (2*rho + 6*m_xx_I*rho_I + 63*m_xz_I*rho_I - 3*m_yy_I*rho_I + 6*m_zz_I*rho_I + 6*rho*ux0 + 6*rho*uz0)/(45*rho);
    m_yy_t45 = -(2*(rho + 3*m_xx_I*rho_I + 9*m_xz_I*rho_I - 69*m_yy_I*rho_I + 3*m_zz_I*rho_I + 3*rho*ux0 + 3*rho*uz0))/(135*rho);
    m_yz_t90 = (m_yz_I*rho_I)/rho;
    m_zz_t45 = (4*rho + 12*m_xx_I*rho_I + 36*m_xz_I*rho_I - 6*m_yy_I*rho_I + 147*m_zz_I*rho_I + 12*rho*ux0 + 12*rho*uz0)/(135*rho);

    rhoVar = rho;

    break;
case 0b0000010100100110:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (24*(15*rho_I + 2*m_xx_I*rho_I - 6*m_xy_I*rho_I + 2*m_yy_I*rho_I - m_zz_I*rho_I - 2*omegaVar*m_xx_I*rho_I + 6*omegaVar*m_xy_I*rho_I - 2*omegaVar*m_yy_I*rho_I + omegaVar*m_zz_I*rho_I))/(6*omegaVar - 12*ux0 + 12*uy0 - 18*omegaVar*ux0 + 18*omegaVar*uy0 - 30*omegaVar*ux0*ux0 - 30*omegaVar*uy0*uy0 + 15*omegaVar*uz0*uz0 + 90*omegaVar*ux0*uy0 + 344);

    m_xx_t45 = (4*rho + 147*m_xx_I*rho_I - 36*m_xy_I*rho_I + 12*m_yy_I*rho_I - 6*m_zz_I*rho_I - 12*rho*ux0 + 12*rho*uy0)/(135*rho);
    m_xy_t90 = -(2*rho + 6*m_xx_I*rho_I - 63*m_xy_I*rho_I + 6*m_yy_I*rho_I - 3*m_zz_I*rho_I - 6*rho*ux0 + 6*rho*uy0)/(45*rho);
    m_xz_t90 = (m_xz_I*rho_I)/rho;
    m_yy_t45 = (4*rho + 12*m_xx_I*rho_I - 36*m_xy_I*rho_I + 147*m_yy_I*rho_I - 6*m_zz_I*rho_I - 12*rho*ux0 + 12*rho*uy0)/(135*rho);
    m_yz_t90 = (m_yz_I*rho_I)/rho;
    m_zz_t45 = -(2*(rho + 3*m_xx_I*rho_I - 9*m_xy_I*rho_I + 3*m_yy_I*rho_I - 69*m_zz_I*rho_I - 3*rho*ux0 + 3*rho*uy0))/(135*rho);

    rhoVar = rho;

    break;
case 0b0000010100100111:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[10] + pop[12] + pop[13] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13]);
    m_xz_I = inv_rho_I * (pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (24*(5*rho_I + m_xx_I*rho_I - 2*m_xy_I*rho_I + 2*m_xz_I*rho_I + m_yy_I*rho_I + 2*m_yz_I*rho_I + m_zz_I*rho_I - omegaVar*m_xx_I*rho_I + 2*omegaVar*m_xy_I*rho_I - 2*omegaVar*m_xz_I*rho_I - omegaVar*m_yy_I*rho_I - 2*omegaVar*m_yz_I*rho_I - omegaVar*m_zz_I*rho_I))/(6*omegaVar + 8*uy0 + 8*uz0 + 12*omegaVar*uy0 + 12*omegaVar*uz0 - 15*omegaVar*ux0*ux0 - 15*omegaVar*uy0*uy0 - 15*omegaVar*uz0*uz0 + 30*omegaVar*ux0*uy0 - 30*omegaVar*ux0*uz0 - 30*omegaVar*uy0*uz0 + 104);

    m_xx_t45 = (2*(rho + 27*m_xx_I*rho_I - 6*m_xy_I*rho_I + 6*m_xz_I*rho_I - 3*m_yz_I*rho_I + rho*uy0 + rho*uz0))/(45*rho);
    m_xy_t90 = -(2*rho + 6*m_xx_I*rho_I - 63*m_xy_I*rho_I + 6*m_yy_I*rho_I - 3*m_zz_I*rho_I - 6*rho*ux0 + 6*rho*uy0)/(45*rho);
    m_xz_t90 = (2*rho + 6*m_xx_I*rho_I + 63*m_xz_I*rho_I - 3*m_yy_I*rho_I + 6*m_zz_I*rho_I + 6*rho*ux0 + 6*rho*uz0)/(45*rho);
    m_yy_t45 = (2*(rho - 6*m_xy_I*rho_I - 3*m_xz_I*rho_I + 27*m_yy_I*rho_I + 6*m_yz_I*rho_I - 3*rho*ux0 + 4*rho*uy0 + rho*uz0))/(45*rho);
    m_yz_t90 = (2*rho - 3*m_xx_I*rho_I + 6*m_yy_I*rho_I + 63*m_yz_I*rho_I + 6*m_zz_I*rho_I + 6*rho*uy0 + 6*rho*uz0)/(45*rho);
    m_zz_t45 = (2*(rho + 3*m_xy_I*rho_I + 6*m_xz_I*rho_I + 6*m_yz_I*rho_I + 27*m_zz_I*rho_I + 3*rho*ux0 + rho*uy0 + 4*rho*uz0))/(45*rho);

    rhoVar = rho;

    break;
case 0b0000010100101000:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = rho_I;

    m_xx_t45 = (m_xx_I*rho_I)/rho;
    m_xy_t90 = (m_xy_I*rho_I)/rho;
    m_xz_t90 = (m_xz_I*rho_I)/rho;
    m_yy_t45 = (m_yy_I*rho_I)/rho;
    m_yz_t90 = (m_yz_I*rho_I)/rho;
    m_zz_t45 = (m_zz_I*rho_I)/rho;

    rhoVar = rho;

    break;
case 0b0000010100101001:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = rho_I;

    m_xx_t45 = (m_xx_I*rho_I)/rho;
    m_xy_t90 = (m_xy_I*rho_I)/rho;
    m_xz_t90 = (m_xz_I*rho_I)/rho;
    m_yy_t45 = (m_yy_I*rho_I)/rho;
    m_yz_t90 = (m_yz_I*rho_I)/rho;
    m_zz_t45 = (m_zz_I*rho_I)/rho;

    rhoVar = rho;

    break;
case 0b0000010100101010:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[15] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[15] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] - (1.0/3.0)*pop[15] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (24*(15*rho_I + 4*m_xx_I*rho_I - 6*m_xy_I*rho_I - 6*m_xz_I*rho_I + m_yy_I*rho_I + m_zz_I*rho_I - 4*omegaVar*m_xx_I*rho_I + 6*omegaVar*m_xy_I*rho_I + 6*omegaVar*m_xz_I*rho_I - omegaVar*m_yy_I*rho_I - omegaVar*m_zz_I*rho_I))/(12*omegaVar - 24*ux0 + 12*uy0 + 12*uz0 - 36*omegaVar*ux0 + 18*omegaVar*uy0 + 18*omegaVar*uz0 - 60*omegaVar*ux0*ux0 - 15*omegaVar*uy0*uy0 - 15*omegaVar*uz0*uz0 + 90*omegaVar*ux0*uy0 + 90*omegaVar*ux0*uz0 + 328);

    m_xx_t45 = (8*rho + 159*m_xx_I*rho_I - 36*m_xy_I*rho_I - 36*m_xz_I*rho_I + 6*m_yy_I*rho_I + 6*m_zz_I*rho_I - 24*rho*ux0 + 12*rho*uy0 + 12*rho*uz0)/(135*rho);
    m_xy_t90 = -(2*rho + 6*m_xx_I*rho_I - 63*m_xy_I*rho_I + 6*m_yy_I*rho_I - 3*m_zz_I*rho_I - 6*rho*ux0 + 6*rho*uy0)/(45*rho);
    m_xz_t90 = -(2*rho + 6*m_xx_I*rho_I - 63*m_xz_I*rho_I - 3*m_yy_I*rho_I + 6*m_zz_I*rho_I - 6*rho*ux0 + 6*rho*uz0)/(45*rho);
    m_yy_t45 = (2*(rho + 3*m_xx_I*rho_I - 18*m_xy_I*rho_I + 9*m_xz_I*rho_I + 75*m_yy_I*rho_I - 6*m_zz_I*rho_I - 3*rho*ux0 + 6*rho*uy0 - 3*rho*uz0))/(135*rho);
    m_yz_t90 = (m_yz_I*rho_I)/rho;
    m_zz_t45 = (2*(rho + 3*m_xx_I*rho_I + 9*m_xy_I*rho_I - 18*m_xz_I*rho_I - 6*m_yy_I*rho_I + 75*m_zz_I*rho_I - 3*rho*ux0 - 3*rho*uy0 + 6*rho*uz0))/(135*rho);

    rhoVar = rho;

    break;
case 0b0000010100101011:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[12] + pop[13] + pop[15] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[15] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] - (1.0/3.0)*pop[15] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (24*(5*rho_I + m_xx_I*rho_I - 2*m_xy_I*rho_I - 2*m_xz_I*rho_I + m_yy_I*rho_I + 2*m_yz_I*rho_I + m_zz_I*rho_I - omegaVar*m_xx_I*rho_I + 2*omegaVar*m_xy_I*rho_I + 2*omegaVar*m_xz_I*rho_I - omegaVar*m_yy_I*rho_I - 2*omegaVar*m_yz_I*rho_I - omegaVar*m_zz_I*rho_I))/(6*omegaVar - 8*ux0 + 8*uy0 + 8*uz0 - 12*omegaVar*ux0 + 12*omegaVar*uy0 + 12*omegaVar*uz0 - 15*omegaVar*ux0*ux0 - 15*omegaVar*uy0*uy0 - 15*omegaVar*uz0*uz0 + 30*omegaVar*ux0*uy0 + 30*omegaVar*ux0*uz0 - 30*omegaVar*uy0*uz0 + 104);

    m_xx_t45 = (2*(rho + 27*m_xx_I*rho_I - 6*m_xy_I*rho_I - 6*m_xz_I*rho_I - 3*m_yz_I*rho_I - 4*rho*ux0 + rho*uy0 + rho*uz0))/(45*rho);
    m_xy_t90 = -(2*rho + 6*m_xx_I*rho_I - 63*m_xy_I*rho_I + 6*m_yy_I*rho_I - 3*m_zz_I*rho_I - 6*rho*ux0 + 6*rho*uy0)/(45*rho);
    m_xz_t90 = -(2*rho + 6*m_xx_I*rho_I - 63*m_xz_I*rho_I - 3*m_yy_I*rho_I + 6*m_zz_I*rho_I - 6*rho*ux0 + 6*rho*uz0)/(45*rho);
    m_yy_t45 = (2*(rho - 6*m_xy_I*rho_I + 3*m_xz_I*rho_I + 27*m_yy_I*rho_I + 6*m_yz_I*rho_I - rho*ux0 + 4*rho*uy0 + rho*uz0))/(45*rho);
    m_yz_t90 = (2*rho - 3*m_xx_I*rho_I + 6*m_yy_I*rho_I + 63*m_yz_I*rho_I + 6*m_zz_I*rho_I + 6*rho*uy0 + 6*rho*uz0)/(45*rho);
    m_zz_t45 = (2*(rho + 3*m_xy_I*rho_I - 6*m_xz_I*rho_I + 6*m_yz_I*rho_I + 27*m_zz_I*rho_I - rho*ux0 + rho*uy0 + 4*rho*uz0))/(45*rho);

    rhoVar = rho;

    break;
case 0b0000010100101100:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17]);

    rho = (24*(15*rho_I - m_xx_I*rho_I + 2*m_yy_I*rho_I - 6*m_yz_I*rho_I + 2*m_zz_I*rho_I + omegaVar*m_xx_I*rho_I - 2*omegaVar*m_yy_I*rho_I + 6*omegaVar*m_yz_I*rho_I - 2*omegaVar*m_zz_I*rho_I))/(6*omegaVar - 12*uy0 + 12*uz0 - 18*omegaVar*uy0 + 18*omegaVar*uz0 + 15*omegaVar*ux0*ux0 - 30*omegaVar*uy0*uy0 - 30*omegaVar*uz0*uz0 + 90*omegaVar*uy0*uz0 + 344);

    m_xx_t45 = -(2*(rho - 69*m_xx_I*rho_I + 3*m_yy_I*rho_I - 9*m_yz_I*rho_I + 3*m_zz_I*rho_I - 3*rho*uy0 + 3*rho*uz0))/(135*rho);
    m_xy_t90 = (m_xy_I*rho_I)/rho;
    m_xz_t90 = (m_xz_I*rho_I)/rho;
    m_yy_t45 = (4*rho - 6*m_xx_I*rho_I + 147*m_yy_I*rho_I - 36*m_yz_I*rho_I + 12*m_zz_I*rho_I - 12*rho*uy0 + 12*rho*uz0)/(135*rho);
    m_yz_t90 = -(2*rho - 3*m_xx_I*rho_I + 6*m_yy_I*rho_I - 63*m_yz_I*rho_I + 6*m_zz_I*rho_I - 6*rho*uy0 + 6*rho*uz0)/(45*rho);
    m_zz_t45 = (4*rho - 6*m_xx_I*rho_I + 12*m_yy_I*rho_I - 36*m_yz_I*rho_I + 147*m_zz_I*rho_I - 12*rho*uy0 + 12*rho*uz0)/(135*rho);

    rhoVar = rho;

    break;
case 0b0000010100101101:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17]);

    rho = (24*(15*rho_I + m_xx_I*rho_I + 6*m_xz_I*rho_I + m_yy_I*rho_I - 6*m_yz_I*rho_I + 4*m_zz_I*rho_I - omegaVar*m_xx_I*rho_I - 6*omegaVar*m_xz_I*rho_I - omegaVar*m_yy_I*rho_I + 6*omegaVar*m_yz_I*rho_I - 4*omegaVar*m_zz_I*rho_I))/(12*omegaVar + 12*ux0 - 12*uy0 + 24*uz0 + 18*omegaVar*ux0 - 18*omegaVar*uy0 + 36*omegaVar*uz0 - 15*omegaVar*ux0*ux0 - 15*omegaVar*uy0*uy0 - 60*omegaVar*uz0*uz0 - 90*omegaVar*ux0*uz0 + 90*omegaVar*uy0*uz0 + 328);

    m_xx_t45 = (2*(rho + 75*m_xx_I*rho_I + 18*m_xz_I*rho_I - 6*m_yy_I*rho_I + 9*m_yz_I*rho_I + 3*m_zz_I*rho_I + 6*rho*ux0 + 3*rho*uy0 + 3*rho*uz0))/(135*rho);
    m_xy_t90 = (m_xy_I*rho_I)/rho;
    m_xz_t90 = (2*rho + 6*m_xx_I*rho_I + 63*m_xz_I*rho_I - 3*m_yy_I*rho_I + 6*m_zz_I*rho_I + 6*rho*ux0 + 6*rho*uz0)/(45*rho);
    m_yy_t45 = -(2*(6*m_xx_I*rho_I - rho + 9*m_xz_I*rho_I - 75*m_yy_I*rho_I + 18*m_yz_I*rho_I - 3*m_zz_I*rho_I + 3*rho*ux0 + 6*rho*uy0 - 3*rho*uz0))/(135*rho);
    m_yz_t90 = -(2*rho - 3*m_xx_I*rho_I + 6*m_yy_I*rho_I - 63*m_yz_I*rho_I + 6*m_zz_I*rho_I - 6*rho*uy0 + 6*rho*uz0)/(45*rho);
    m_zz_t45 = (8*rho + 6*m_xx_I*rho_I + 36*m_xz_I*rho_I + 6*m_yy_I*rho_I - 36*m_yz_I*rho_I + 159*m_zz_I*rho_I + 12*rho*ux0 - 12*rho*uy0 + 24*rho*uz0)/(135*rho);

    rhoVar = rho;

    break;
case 0b0000010100101110:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[15] + pop[17];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[15] - (1.0/3.0)*pop[17]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] - (1.0/3.0)*pop[15] + (2.0/3.0)*pop[17]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[17]);

    rho = (24*(5*rho_I + m_xx_I*rho_I - 2*m_xy_I*rho_I - 2*m_xz_I*rho_I + m_yy_I*rho_I - 2*m_yz_I*rho_I + m_zz_I*rho_I - omegaVar*m_xx_I*rho_I + 2*omegaVar*m_xy_I*rho_I + 2*omegaVar*m_xz_I*rho_I - omegaVar*m_yy_I*rho_I + 2*omegaVar*m_yz_I*rho_I - omegaVar*m_zz_I*rho_I))/(6*omegaVar - 8*ux0 + 8*uz0 - 12*omegaVar*ux0 + 12*omegaVar*uz0 - 15*omegaVar*ux0*ux0 - 15*omegaVar*uy0*uy0 - 15*omegaVar*uz0*uz0 + 30*omegaVar*ux0*uy0 + 30*omegaVar*ux0*uz0 + 30*omegaVar*uy0*uz0 + 104);

    m_xx_t45 = (2*(rho + 27*m_xx_I*rho_I - 6*m_xy_I*rho_I - 6*m_xz_I*rho_I + 3*m_yz_I*rho_I - 4*rho*ux0 + 3*rho*uy0 + rho*uz0))/(45*rho);
    m_xy_t90 = -(2*rho + 6*m_xx_I*rho_I - 63*m_xy_I*rho_I + 6*m_yy_I*rho_I - 3*m_zz_I*rho_I - 6*rho*ux0 + 6*rho*uy0)/(45*rho);
    m_xz_t90 = -(2*rho + 6*m_xx_I*rho_I - 63*m_xz_I*rho_I - 3*m_yy_I*rho_I + 6*m_zz_I*rho_I - 6*rho*ux0 + 6*rho*uz0)/(45*rho);
    m_yy_t45 = (2*(rho - 6*m_xy_I*rho_I + 3*m_xz_I*rho_I + 27*m_yy_I*rho_I - 6*m_yz_I*rho_I - rho*ux0 + rho*uz0))/(45*rho);
    m_yz_t90 = -(2*rho - 3*m_xx_I*rho_I + 6*m_yy_I*rho_I - 63*m_yz_I*rho_I + 6*m_zz_I*rho_I - 6*rho*uy0 + 6*rho*uz0)/(45*rho);
    m_zz_t45 = (2*(rho + 3*m_xy_I*rho_I - 6*m_xz_I*rho_I - 6*m_yz_I*rho_I + 27*m_zz_I*rho_I - rho*ux0 - 3*rho*uy0 + 4*rho*uz0))/(45*rho);

    rhoVar = rho;

    break;
case 0b0000010100101111:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[6] + pop[7] + pop[8] + pop[10] + pop[12] + pop[13] + pop[15] + pop[17];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[15] - (1.0/3.0)*pop[17]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13]);
    m_xz_I = inv_rho_I * (pop[10] - pop[15]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] - (1.0/3.0)*pop[15] + (2.0/3.0)*pop[17]);
    m_yz_I = inv_rho_I * (pop[12] - pop[17]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[17]);

    rho = (72*(107*rho_I + 12*m_xx_I*rho_I - 33*m_xy_I*rho_I + 12*m_yy_I*rho_I + 72*m_zz_I*rho_I - 12*omegaVar*m_xx_I*rho_I + 33*omegaVar*m_xy_I*rho_I - 12*omegaVar*m_yy_I*rho_I - 72*omegaVar*m_zz_I*rho_I))/(626*omegaVar - 492*ux0 + 492*uy0 + 2124*uz0 - 150*omegaVar*ux0 + 150*omegaVar*uy0 + 1728*omegaVar*uz0 - 642*omegaVar*ux0*ux0 - 642*omegaVar*uy0*uy0 - 3531*omegaVar*uz0*uz0 + 1926*omegaVar*ux0*uy0 + 5580);

    m_xx_t45 = -(4*(24*m_xy_I*rho_I - 106*m_xx_I*rho_I - 2*rho + m_yy_I*rho_I + 6*m_zz_I*rho_I + 8*rho*ux0 - 8*rho*uy0 + 2*rho*uz0))/(321*rho);
    m_xy_t90 = -(11*rho + 48*m_xx_I*rho_I - 453*m_xy_I*rho_I + 48*m_yy_I*rho_I - 33*m_zz_I*rho_I - 44*rho*ux0 + 44*rho*uy0 - 11*rho*uz0)/(321*rho);
    m_xz_t90 = (6*m_xz_I*rho_I + rho*ux0)/(3*rho);
    m_yy_t45 = -(4*(m_xx_I*rho_I - 2*rho + 24*m_xy_I*rho_I - 106*m_yy_I*rho_I + 6*m_zz_I*rho_I + 8*rho*ux0 - 8*rho*uy0 + 2*rho*uz0))/(321*rho);
    m_yz_t90 = (6*m_yz_I*rho_I + rho*uy0)/(3*rho);
    m_zz_t45 = (2*(24*rho - 12*m_xx_I*rho_I + 33*m_xy_I*rho_I - 12*m_yy_I*rho_I + 249*m_zz_I*rho_I + 11*rho*ux0 - 11*rho*uy0 + 83*rho*uz0))/(321*rho);

    rhoVar = rho;

    break;
case 0b0000010100110000:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);

    rho = (24*(15*rho_I - m_xx_I*rho_I + 2*m_yy_I*rho_I - 6*m_yz_I*rho_I + 2*m_zz_I*rho_I + omegaVar*m_xx_I*rho_I - 2*omegaVar*m_yy_I*rho_I + 6*omegaVar*m_yz_I*rho_I - 2*omegaVar*m_zz_I*rho_I))/(6*omegaVar + 12*uy0 - 12*uz0 + 18*omegaVar*uy0 - 18*omegaVar*uz0 + 15*omegaVar*ux0*ux0 - 30*omegaVar*uy0*uy0 - 30*omegaVar*uz0*uz0 + 90*omegaVar*uy0*uz0 + 344);

    m_xx_t45 = -(2*(rho - 69*m_xx_I*rho_I + 3*m_yy_I*rho_I - 9*m_yz_I*rho_I + 3*m_zz_I*rho_I + 3*rho*uy0 - 3*rho*uz0))/(135*rho);
    m_xy_t90 = (m_xy_I*rho_I)/rho;
    m_xz_t90 = (m_xz_I*rho_I)/rho;
    m_yy_t45 = (4*rho - 6*m_xx_I*rho_I + 147*m_yy_I*rho_I - 36*m_yz_I*rho_I + 12*m_zz_I*rho_I + 12*rho*uy0 - 12*rho*uz0)/(135*rho);
    m_yz_t90 = -(2*rho - 3*m_xx_I*rho_I + 6*m_yy_I*rho_I - 63*m_yz_I*rho_I + 6*m_zz_I*rho_I + 6*rho*uy0 - 6*rho*uz0)/(45*rho);
    m_zz_t45 = (4*rho - 6*m_xx_I*rho_I + 12*m_yy_I*rho_I - 36*m_yz_I*rho_I + 147*m_zz_I*rho_I + 12*rho*uy0 - 12*rho*uz0)/(135*rho);

    rhoVar = rho;

    break;
case 0b0000010100110001:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);

    rho = (24*(15*rho_I + m_xx_I*rho_I + 6*m_xy_I*rho_I + 4*m_yy_I*rho_I - 6*m_yz_I*rho_I + m_zz_I*rho_I - omegaVar*m_xx_I*rho_I - 6*omegaVar*m_xy_I*rho_I - 4*omegaVar*m_yy_I*rho_I + 6*omegaVar*m_yz_I*rho_I - omegaVar*m_zz_I*rho_I))/(12*omegaVar + 12*ux0 + 24*uy0 - 12*uz0 + 18*omegaVar*ux0 + 36*omegaVar*uy0 - 18*omegaVar*uz0 - 15*omegaVar*ux0*ux0 - 60*omegaVar*uy0*uy0 - 15*omegaVar*uz0*uz0 - 90*omegaVar*ux0*uy0 + 90*omegaVar*uy0*uz0 + 328);

    m_xx_t45 = (2*(rho + 75*m_xx_I*rho_I + 18*m_xy_I*rho_I + 3*m_yy_I*rho_I + 9*m_yz_I*rho_I - 6*m_zz_I*rho_I + 6*rho*ux0 + 3*rho*uy0 + 3*rho*uz0))/(135*rho);
    m_xy_t90 = (2*rho + 6*m_xx_I*rho_I + 63*m_xy_I*rho_I + 6*m_yy_I*rho_I - 3*m_zz_I*rho_I + 6*rho*ux0 + 6*rho*uy0)/(45*rho);
    m_xz_t90 = (m_xz_I*rho_I)/rho;
    m_yy_t45 = (8*rho + 6*m_xx_I*rho_I + 36*m_xy_I*rho_I + 159*m_yy_I*rho_I - 36*m_yz_I*rho_I + 6*m_zz_I*rho_I + 12*rho*ux0 + 24*rho*uy0 - 12*rho*uz0)/(135*rho);
    m_yz_t90 = -(2*rho - 3*m_xx_I*rho_I + 6*m_yy_I*rho_I - 63*m_yz_I*rho_I + 6*m_zz_I*rho_I + 6*rho*uy0 - 6*rho*uz0)/(45*rho);
    m_zz_t45 = -(2*(6*m_xx_I*rho_I - rho + 9*m_xy_I*rho_I - 3*m_yy_I*rho_I + 18*m_yz_I*rho_I - 75*m_zz_I*rho_I + 3*rho*ux0 - 3*rho*uy0 + 6*rho*uz0))/(135*rho);

    rhoVar = rho;

    break;
case 0b0000010100110010:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[15] + pop[16] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);

    rho = -(24*(15*rho_I + m_xx_I*rho_I - 6*m_xy_I*rho_I + 4*m_yy_I*rho_I - 6*m_yz_I*rho_I + m_zz_I*rho_I - omegaVar*m_xx_I*rho_I + 6*omegaVar*m_xy_I*rho_I - 4*omegaVar*m_yy_I*rho_I + 6*omegaVar*m_yz_I*rho_I - omegaVar*m_zz_I*rho_I))/(12*ux0 - 12*omegaVar - 24*uy0 + 12*uz0 + 18*omegaVar*ux0 - 36*omegaVar*uy0 + 18*omegaVar*uz0 + 15*omegaVar*ux0*ux0 + 60*omegaVar*uy0*uy0 + 15*omegaVar*uz0*uz0 - 90*omegaVar*ux0*uy0 - 90*omegaVar*uy0*uz0 - 328);

    m_xx_t45 = (2*(rho + 75*m_xx_I*rho_I - 18*m_xy_I*rho_I + 3*m_yy_I*rho_I + 9*m_yz_I*rho_I - 6*m_zz_I*rho_I - 6*rho*ux0 + 3*rho*uy0 + 3*rho*uz0))/(135*rho);
    m_xy_t90 = -(2*rho + 6*m_xx_I*rho_I - 63*m_xy_I*rho_I + 6*m_yy_I*rho_I - 3*m_zz_I*rho_I - 6*rho*ux0 + 6*rho*uy0)/(45*rho);
    m_xz_t90 = (m_xz_I*rho_I)/rho;
    m_yy_t45 = (8*rho + 6*m_xx_I*rho_I - 36*m_xy_I*rho_I + 159*m_yy_I*rho_I - 36*m_yz_I*rho_I + 6*m_zz_I*rho_I - 12*rho*ux0 + 24*rho*uy0 - 12*rho*uz0)/(135*rho);
    m_yz_t90 = -(2*rho - 3*m_xx_I*rho_I + 6*m_yy_I*rho_I - 63*m_yz_I*rho_I + 6*m_zz_I*rho_I + 6*rho*uy0 - 6*rho*uz0)/(45*rho);
    m_zz_t45 = (2*(rho - 6*m_xx_I*rho_I + 9*m_xy_I*rho_I + 3*m_yy_I*rho_I - 18*m_yz_I*rho_I + 75*m_zz_I*rho_I + 3*rho*ux0 + 3*rho*uy0 - 6*rho*uz0))/(135*rho);

    rhoVar = rho;

    break;
case 0b0000010100110011:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[4] + pop[5] + pop[6] + pop[8] + pop[9] + pop[10] + pop[12] + pop[13] + pop[15] + pop[16] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[8] - pop[13]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[12] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);

    rho = (3*rho_I*(3*m_yy_I - 3*omegaVar*m_yy_I + 4))/(omegaVar + 3*uy0 + 3*omegaVar*uy0 - 6*omegaVar*uy0*uy0 + 9);

    m_xx_t45 = (4*(10*m_xx_I*rho_I - m_zz_I*rho_I))/(33*rho);
    m_xy_t90 = (6*m_xy_I*rho_I + rho*ux0)/(3*rho);
    m_xz_t90 = (m_xz_I*rho_I)/rho;
    m_yy_t45 = (rho + 9*m_yy_I*rho_I + 3*rho*uy0)/(6*rho);
    m_yz_t90 = (6*m_yz_I*rho_I + rho*uz0)/(3*rho);
    m_zz_t45 = -(4*(m_xx_I*rho_I - 10*m_zz_I*rho_I))/(33*rho);

    rhoVar = rho;

    break;
case 0b0000010100110100:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);

    rho = (24*(15*rho_I - m_xx_I*rho_I + 2*m_yy_I*rho_I - 6*m_yz_I*rho_I + 2*m_zz_I*rho_I + omegaVar*m_xx_I*rho_I - 2*omegaVar*m_yy_I*rho_I + 6*omegaVar*m_yz_I*rho_I - 2*omegaVar*m_zz_I*rho_I))/(6*omegaVar + 12*uy0 - 12*uz0 + 18*omegaVar*uy0 - 18*omegaVar*uz0 + 15*omegaVar*ux0*ux0 - 30*omegaVar*uy0*uy0 - 30*omegaVar*uz0*uz0 + 90*omegaVar*uy0*uz0 + 344);

    m_xx_t45 = -(2*(rho - 69*m_xx_I*rho_I + 3*m_yy_I*rho_I - 9*m_yz_I*rho_I + 3*m_zz_I*rho_I + 3*rho*uy0 - 3*rho*uz0))/(135*rho);
    m_xy_t90 = (m_xy_I*rho_I)/rho;
    m_xz_t90 = (m_xz_I*rho_I)/rho;
    m_yy_t45 = (4*rho - 6*m_xx_I*rho_I + 147*m_yy_I*rho_I - 36*m_yz_I*rho_I + 12*m_zz_I*rho_I + 12*rho*uy0 - 12*rho*uz0)/(135*rho);
    m_yz_t90 = -(2*rho - 3*m_xx_I*rho_I + 6*m_yy_I*rho_I - 63*m_yz_I*rho_I + 6*m_zz_I*rho_I + 6*rho*uy0 - 6*rho*uz0)/(45*rho);
    m_zz_t45 = (4*rho - 6*m_xx_I*rho_I + 12*m_yy_I*rho_I - 36*m_yz_I*rho_I + 147*m_zz_I*rho_I + 12*rho*uy0 - 12*rho*uz0)/(135*rho);

    rhoVar = rho;

    break;
case 0b0000010100110101:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[8] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);

    rho = (24*(5*rho_I + m_xx_I*rho_I + 2*m_xy_I*rho_I + 2*m_xz_I*rho_I + m_yy_I*rho_I - 2*m_yz_I*rho_I + m_zz_I*rho_I - omegaVar*m_xx_I*rho_I - 2*omegaVar*m_xy_I*rho_I - 2*omegaVar*m_xz_I*rho_I - omegaVar*m_yy_I*rho_I + 2*omegaVar*m_yz_I*rho_I - omegaVar*m_zz_I*rho_I))/(6*omegaVar + 8*ux0 + 8*uy0 + 12*omegaVar*ux0 + 12*omegaVar*uy0 - 15*omegaVar*ux0*ux0 - 15*omegaVar*uy0*uy0 - 15*omegaVar*uz0*uz0 - 30*omegaVar*ux0*uy0 - 30*omegaVar*ux0*uz0 + 30*omegaVar*uy0*uz0 + 104);

    m_xx_t45 = (2*(rho + 27*m_xx_I*rho_I + 6*m_xy_I*rho_I + 6*m_xz_I*rho_I + 3*m_yz_I*rho_I + 4*rho*ux0 + rho*uy0 + 3*rho*uz0))/(45*rho);
    m_xy_t90 = (2*rho + 6*m_xx_I*rho_I + 63*m_xy_I*rho_I + 6*m_yy_I*rho_I - 3*m_zz_I*rho_I + 6*rho*ux0 + 6*rho*uy0)/(45*rho);
    m_xz_t90 = (2*rho + 6*m_xx_I*rho_I + 63*m_xz_I*rho_I - 3*m_yy_I*rho_I + 6*m_zz_I*rho_I + 6*rho*ux0 + 6*rho*uz0)/(45*rho);
    m_yy_t45 = (2*(rho + 6*m_xy_I*rho_I - 3*m_xz_I*rho_I + 27*m_yy_I*rho_I - 6*m_yz_I*rho_I + rho*ux0 + 4*rho*uy0 - 3*rho*uz0))/(45*rho);
    m_yz_t90 = -(2*rho - 3*m_xx_I*rho_I + 6*m_yy_I*rho_I - 63*m_yz_I*rho_I + 6*m_zz_I*rho_I + 6*rho*uy0 - 6*rho*uz0)/(45*rho);
    m_zz_t45 = (2*(rho - 3*m_xy_I*rho_I + 6*m_xz_I*rho_I - 6*m_yz_I*rho_I + 27*m_zz_I*rho_I + rho*ux0 + rho*uy0))/(45*rho);

    rhoVar = rho;

    break;
case 0b0000010100110110:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[15] + pop[16] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);

    rho = -(24*(15*rho_I + m_xx_I*rho_I - 6*m_xy_I*rho_I + 4*m_yy_I*rho_I - 6*m_yz_I*rho_I + m_zz_I*rho_I - omegaVar*m_xx_I*rho_I + 6*omegaVar*m_xy_I*rho_I - 4*omegaVar*m_yy_I*rho_I + 6*omegaVar*m_yz_I*rho_I - omegaVar*m_zz_I*rho_I))/(12*ux0 - 12*omegaVar - 24*uy0 + 12*uz0 + 18*omegaVar*ux0 - 36*omegaVar*uy0 + 18*omegaVar*uz0 + 15*omegaVar*ux0*ux0 + 60*omegaVar*uy0*uy0 + 15*omegaVar*uz0*uz0 - 90*omegaVar*ux0*uy0 - 90*omegaVar*uy0*uz0 - 328);

    m_xx_t45 = (2*(rho + 75*m_xx_I*rho_I - 18*m_xy_I*rho_I + 3*m_yy_I*rho_I + 9*m_yz_I*rho_I - 6*m_zz_I*rho_I - 6*rho*ux0 + 3*rho*uy0 + 3*rho*uz0))/(135*rho);
    m_xy_t90 = -(2*rho + 6*m_xx_I*rho_I - 63*m_xy_I*rho_I + 6*m_yy_I*rho_I - 3*m_zz_I*rho_I - 6*rho*ux0 + 6*rho*uy0)/(45*rho);
    m_xz_t90 = (m_xz_I*rho_I)/rho;
    m_yy_t45 = (8*rho + 6*m_xx_I*rho_I - 36*m_xy_I*rho_I + 159*m_yy_I*rho_I - 36*m_yz_I*rho_I + 6*m_zz_I*rho_I - 12*rho*ux0 + 24*rho*uy0 - 12*rho*uz0)/(135*rho);
    m_yz_t90 = -(2*rho - 3*m_xx_I*rho_I + 6*m_yy_I*rho_I - 63*m_yz_I*rho_I + 6*m_zz_I*rho_I + 6*rho*uy0 - 6*rho*uz0)/(45*rho);
    m_zz_t45 = (2*(rho - 6*m_xx_I*rho_I + 9*m_xy_I*rho_I + 3*m_yy_I*rho_I - 18*m_yz_I*rho_I + 75*m_zz_I*rho_I + 3*rho*ux0 + 3*rho*uy0 - 6*rho*uz0))/(135*rho);

    rhoVar = rho;

    break;
case 0b0000010100110111:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[4] + pop[5] + pop[6] + pop[8] + pop[10] + pop[12] + pop[13] + pop[15] + pop[16] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[8] - pop[13]);
    m_xz_I = inv_rho_I * (pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[12] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);

    rho = (72*(107*rho_I + 12*m_xx_I*rho_I + 33*m_xz_I*rho_I + 72*m_yy_I*rho_I + 12*m_zz_I*rho_I - 12*omegaVar*m_xx_I*rho_I - 33*omegaVar*m_xz_I*rho_I - 72*omegaVar*m_yy_I*rho_I - 12*omegaVar*m_zz_I*rho_I))/(626*omegaVar + 492*ux0 + 2124*uy0 + 492*uz0 + 150*omegaVar*ux0 + 1728*omegaVar*uy0 + 150*omegaVar*uz0 - 642*omegaVar*ux0*ux0 - 3531*omegaVar*uy0*uy0 - 642*omegaVar*uz0*uz0 - 1926*omegaVar*ux0*uz0 + 5580);

    m_xx_t45 = (4*(2*rho + 106*m_xx_I*rho_I + 24*m_xz_I*rho_I - 6*m_yy_I*rho_I - m_zz_I*rho_I + 8*rho*ux0 - 2*rho*uy0 + 8*rho*uz0))/(321*rho);
    m_xy_t90 = (6*m_xy_I*rho_I + rho*ux0)/(3*rho);
    m_xz_t90 = (11*rho + 48*m_xx_I*rho_I + 453*m_xz_I*rho_I - 33*m_yy_I*rho_I + 48*m_zz_I*rho_I + 44*rho*ux0 - 11*rho*uy0 + 44*rho*uz0)/(321*rho);
    m_yy_t45 = -(2*(12*m_xx_I*rho_I - 24*rho + 33*m_xz_I*rho_I - 249*m_yy_I*rho_I + 12*m_zz_I*rho_I + 11*rho*ux0 - 83*rho*uy0 + 11*rho*uz0))/(321*rho);
    m_yz_t90 = (6*m_yz_I*rho_I + rho*uz0)/(3*rho);
    m_zz_t45 = (4*(2*rho - m_xx_I*rho_I + 24*m_xz_I*rho_I - 6*m_yy_I*rho_I + 106*m_zz_I*rho_I + 8*rho*ux0 - 2*rho*uy0 + 8*rho*uz0))/(321*rho);

    rhoVar = rho;

    break;
case 0b0000010100111000:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);

    rho = (24*(15*rho_I - m_xx_I*rho_I + 2*m_yy_I*rho_I - 6*m_yz_I*rho_I + 2*m_zz_I*rho_I + omegaVar*m_xx_I*rho_I - 2*omegaVar*m_yy_I*rho_I + 6*omegaVar*m_yz_I*rho_I - 2*omegaVar*m_zz_I*rho_I))/(6*omegaVar + 12*uy0 - 12*uz0 + 18*omegaVar*uy0 - 18*omegaVar*uz0 + 15*omegaVar*ux0*ux0 - 30*omegaVar*uy0*uy0 - 30*omegaVar*uz0*uz0 + 90*omegaVar*uy0*uz0 + 344);

    m_xx_t45 = -(2*(rho - 69*m_xx_I*rho_I + 3*m_yy_I*rho_I - 9*m_yz_I*rho_I + 3*m_zz_I*rho_I + 3*rho*uy0 - 3*rho*uz0))/(135*rho);
    m_xy_t90 = (m_xy_I*rho_I)/rho;
    m_xz_t90 = (m_xz_I*rho_I)/rho;
    m_yy_t45 = (4*rho - 6*m_xx_I*rho_I + 147*m_yy_I*rho_I - 36*m_yz_I*rho_I + 12*m_zz_I*rho_I + 12*rho*uy0 - 12*rho*uz0)/(135*rho);
    m_yz_t90 = -(2*rho - 3*m_xx_I*rho_I + 6*m_yy_I*rho_I - 63*m_yz_I*rho_I + 6*m_zz_I*rho_I + 6*rho*uy0 - 6*rho*uz0)/(45*rho);
    m_zz_t45 = (4*rho - 6*m_xx_I*rho_I + 12*m_yy_I*rho_I - 36*m_yz_I*rho_I + 147*m_zz_I*rho_I + 12*rho*uy0 - 12*rho*uz0)/(135*rho);

    rhoVar = rho;

    break;
case 0b0000010100111001:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);

    rho = (24*(15*rho_I + m_xx_I*rho_I + 6*m_xy_I*rho_I + 4*m_yy_I*rho_I - 6*m_yz_I*rho_I + m_zz_I*rho_I - omegaVar*m_xx_I*rho_I - 6*omegaVar*m_xy_I*rho_I - 4*omegaVar*m_yy_I*rho_I + 6*omegaVar*m_yz_I*rho_I - omegaVar*m_zz_I*rho_I))/(12*omegaVar + 12*ux0 + 24*uy0 - 12*uz0 + 18*omegaVar*ux0 + 36*omegaVar*uy0 - 18*omegaVar*uz0 - 15*omegaVar*ux0*ux0 - 60*omegaVar*uy0*uy0 - 15*omegaVar*uz0*uz0 - 90*omegaVar*ux0*uy0 + 90*omegaVar*uy0*uz0 + 328);

    m_xx_t45 = (2*(rho + 75*m_xx_I*rho_I + 18*m_xy_I*rho_I + 3*m_yy_I*rho_I + 9*m_yz_I*rho_I - 6*m_zz_I*rho_I + 6*rho*ux0 + 3*rho*uy0 + 3*rho*uz0))/(135*rho);
    m_xy_t90 = (2*rho + 6*m_xx_I*rho_I + 63*m_xy_I*rho_I + 6*m_yy_I*rho_I - 3*m_zz_I*rho_I + 6*rho*ux0 + 6*rho*uy0)/(45*rho);
    m_xz_t90 = (m_xz_I*rho_I)/rho;
    m_yy_t45 = (8*rho + 6*m_xx_I*rho_I + 36*m_xy_I*rho_I + 159*m_yy_I*rho_I - 36*m_yz_I*rho_I + 6*m_zz_I*rho_I + 12*rho*ux0 + 24*rho*uy0 - 12*rho*uz0)/(135*rho);
    m_yz_t90 = -(2*rho - 3*m_xx_I*rho_I + 6*m_yy_I*rho_I - 63*m_yz_I*rho_I + 6*m_zz_I*rho_I + 6*rho*uy0 - 6*rho*uz0)/(45*rho);
    m_zz_t45 = -(2*(6*m_xx_I*rho_I - rho + 9*m_xy_I*rho_I - 3*m_yy_I*rho_I + 18*m_yz_I*rho_I - 75*m_zz_I*rho_I + 3*rho*ux0 - 3*rho*uy0 + 6*rho*uz0))/(135*rho);

    rhoVar = rho;

    break;
case 0b0000010100111010:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[15] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[15] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] - (1.0/3.0)*pop[15] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[18]);

    rho = (24*(5*rho_I + m_xx_I*rho_I - 2*m_xy_I*rho_I - 2*m_xz_I*rho_I + m_yy_I*rho_I - 2*m_yz_I*rho_I + m_zz_I*rho_I - omegaVar*m_xx_I*rho_I + 2*omegaVar*m_xy_I*rho_I + 2*omegaVar*m_xz_I*rho_I - omegaVar*m_yy_I*rho_I + 2*omegaVar*m_yz_I*rho_I - omegaVar*m_zz_I*rho_I))/(6*omegaVar - 8*ux0 + 8*uy0 - 12*omegaVar*ux0 + 12*omegaVar*uy0 - 15*omegaVar*ux0*ux0 - 15*omegaVar*uy0*uy0 - 15*omegaVar*uz0*uz0 + 30*omegaVar*ux0*uy0 + 30*omegaVar*ux0*uz0 + 30*omegaVar*uy0*uz0 + 104);

    m_xx_t45 = (2*(rho + 27*m_xx_I*rho_I - 6*m_xy_I*rho_I - 6*m_xz_I*rho_I + 3*m_yz_I*rho_I - 4*rho*ux0 + rho*uy0 + 3*rho*uz0))/(45*rho);
    m_xy_t90 = -(2*rho + 6*m_xx_I*rho_I - 63*m_xy_I*rho_I + 6*m_yy_I*rho_I - 3*m_zz_I*rho_I - 6*rho*ux0 + 6*rho*uy0)/(45*rho);
    m_xz_t90 = -(2*rho + 6*m_xx_I*rho_I - 63*m_xz_I*rho_I - 3*m_yy_I*rho_I + 6*m_zz_I*rho_I - 6*rho*ux0 + 6*rho*uz0)/(45*rho);
    m_yy_t45 = (2*(rho - 6*m_xy_I*rho_I + 3*m_xz_I*rho_I + 27*m_yy_I*rho_I - 6*m_yz_I*rho_I - rho*ux0 + 4*rho*uy0 - 3*rho*uz0))/(45*rho);
    m_yz_t90 = -(2*rho - 3*m_xx_I*rho_I + 6*m_yy_I*rho_I - 63*m_yz_I*rho_I + 6*m_zz_I*rho_I + 6*rho*uy0 - 6*rho*uz0)/(45*rho);
    m_zz_t45 = (2*(rho + 3*m_xy_I*rho_I - 6*m_xz_I*rho_I - 6*m_yz_I*rho_I + 27*m_zz_I*rho_I - rho*ux0 + rho*uy0))/(45*rho);

    rhoVar = rho;

    break;
case 0b0000010100111011:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[4] + pop[5] + pop[6] + pop[8] + pop[9] + pop[10] + pop[12] + pop[13] + pop[15] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[15] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[8] - pop[13]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] - (1.0/3.0)*pop[15] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[12] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[18]);

    rho = (72*(107*rho_I + 12*m_xx_I*rho_I - 33*m_xz_I*rho_I + 72*m_yy_I*rho_I + 12*m_zz_I*rho_I - 12*omegaVar*m_xx_I*rho_I + 33*omegaVar*m_xz_I*rho_I - 72*omegaVar*m_yy_I*rho_I - 12*omegaVar*m_zz_I*rho_I))/(626*omegaVar - 492*ux0 + 2124*uy0 + 492*uz0 - 150*omegaVar*ux0 + 1728*omegaVar*uy0 + 150*omegaVar*uz0 - 642*omegaVar*ux0*ux0 - 3531*omegaVar*uy0*uy0 - 642*omegaVar*uz0*uz0 + 1926*omegaVar*ux0*uz0 + 5580);

    m_xx_t45 = -(4*(24*m_xz_I*rho_I - 106*m_xx_I*rho_I - 2*rho + 6*m_yy_I*rho_I + m_zz_I*rho_I + 8*rho*ux0 + 2*rho*uy0 - 8*rho*uz0))/(321*rho);
    m_xy_t90 = (6*m_xy_I*rho_I + rho*ux0)/(3*rho);
    m_xz_t90 = -(11*rho + 48*m_xx_I*rho_I - 453*m_xz_I*rho_I - 33*m_yy_I*rho_I + 48*m_zz_I*rho_I - 44*rho*ux0 - 11*rho*uy0 + 44*rho*uz0)/(321*rho);
    m_yy_t45 = (2*(24*rho - 12*m_xx_I*rho_I + 33*m_xz_I*rho_I + 249*m_yy_I*rho_I - 12*m_zz_I*rho_I + 11*rho*ux0 + 83*rho*uy0 - 11*rho*uz0))/(321*rho);
    m_yz_t90 = (6*m_yz_I*rho_I + rho*uz0)/(3*rho);
    m_zz_t45 = -(4*(m_xx_I*rho_I - 2*rho + 24*m_xz_I*rho_I + 6*m_yy_I*rho_I - 106*m_zz_I*rho_I + 8*rho*ux0 + 2*rho*uy0 - 8*rho*uz0))/(321*rho);

    rhoVar = rho;

    break;
case 0b0000010100111100:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16]);

    rho = (12*(3*rho_I - m_xx_I*rho_I + 2*m_yy_I*rho_I - 6*m_yz_I*rho_I + 2*m_zz_I*rho_I + omegaVar*m_xx_I*rho_I - 2*omegaVar*m_yy_I*rho_I + 6*omegaVar*m_yz_I*rho_I - 2*omegaVar*m_zz_I*rho_I))/(6*omegaVar + 3*omegaVar*ux0*ux0 - 6*omegaVar*uy0*uy0 - 6*omegaVar*uz0*uz0 + 18*omegaVar*uy0*uz0 + 28);

    m_xx_t45 = -(2*(rho - 15*m_xx_I*rho_I + 3*m_yy_I*rho_I - 9*m_yz_I*rho_I + 3*m_zz_I*rho_I))/(27*rho);
    m_xy_t90 = (m_xy_I*rho_I)/rho;
    m_xz_t90 = (m_xz_I*rho_I)/rho;
    m_yy_t45 = (4*rho - 6*m_xx_I*rho_I + 39*m_yy_I*rho_I - 36*m_yz_I*rho_I + 12*m_zz_I*rho_I)/(27*rho);
    m_yz_t90 = -(2*rho - 3*m_xx_I*rho_I + 6*m_yy_I*rho_I - 27*m_yz_I*rho_I + 6*m_zz_I*rho_I)/(9*rho);
    m_zz_t45 = (4*rho - 6*m_xx_I*rho_I + 12*m_yy_I*rho_I - 36*m_yz_I*rho_I + 39*m_zz_I*rho_I)/(27*rho);

    rhoVar = rho;

    break;
case 0b0000010100111101:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[8] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16]);
    m_xy_I = inv_rho_I * (pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16]);

    rho = (24*(15*rho_I - m_xx_I*rho_I + 6*m_xy_I*rho_I + 6*m_xz_I*rho_I + 11*m_yy_I*rho_I - 30*m_yz_I*rho_I + 11*m_zz_I*rho_I + omegaVar*m_xx_I*rho_I - 6*omegaVar*m_xy_I*rho_I - 6*omegaVar*m_xz_I*rho_I - 11*omegaVar*m_yy_I*rho_I + 30*omegaVar*m_yz_I*rho_I - 11*omegaVar*m_zz_I*rho_I))/(72*omegaVar + 24*ux0 + 12*uy0 + 12*uz0 + 36*omegaVar*ux0 + 18*omegaVar*uy0 + 18*omegaVar*uz0 - 30*omegaVar*ux0*ux0 - 75*omegaVar*uy0*uy0 - 75*omegaVar*uz0*uz0 - 90*omegaVar*ux0*uy0 - 90*omegaVar*ux0*uz0 + 180*omegaVar*uy0*uz0 + 248);

    m_xx_t45 = (2*(87*m_xx_I*rho_I - rho + 18*m_xy_I*rho_I + 18*m_xz_I*rho_I - 12*m_yy_I*rho_I + 45*m_yz_I*rho_I - 12*m_zz_I*rho_I + 12*rho*ux0 + 6*rho*uy0 + 6*rho*uz0))/(135*rho);
    m_xy_t90 = (2*rho + 6*m_xx_I*rho_I + 63*m_xy_I*rho_I + 6*m_yy_I*rho_I - 3*m_zz_I*rho_I + 6*rho*ux0 + 6*rho*uy0)/(45*rho);
    m_xz_t90 = (2*rho + 6*m_xx_I*rho_I + 63*m_xz_I*rho_I - 3*m_yy_I*rho_I + 6*m_zz_I*rho_I + 6*rho*ux0 + 6*rho*uz0)/(45*rho);
    m_yy_t45 = (2*(11*rho - 12*m_xx_I*rho_I + 18*m_xy_I*rho_I - 9*m_xz_I*rho_I + 105*m_yy_I*rho_I - 90*m_yz_I*rho_I + 24*m_zz_I*rho_I + 3*rho*ux0 + 6*rho*uy0 - 3*rho*uz0))/(135*rho);
    m_yz_t90 = -(2*rho - 3*m_xx_I*rho_I + 6*m_yy_I*rho_I - 27*m_yz_I*rho_I + 6*m_zz_I*rho_I)/(9*rho);
    m_zz_t45 = (2*(11*rho - 12*m_xx_I*rho_I - 9*m_xy_I*rho_I + 18*m_xz_I*rho_I + 24*m_yy_I*rho_I - 90*m_yz_I*rho_I + 105*m_zz_I*rho_I + 3*rho*ux0 - 3*rho*uy0 + 6*rho*uz0))/(135*rho);

    rhoVar = rho;

    break;
case 0b0000010100111110:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[15];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[15]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] - (1.0/3.0)*pop[15]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] + (2.0/3.0)*pop[15]);

    rho = (24*(15*rho_I - m_xx_I*rho_I - 6*m_xy_I*rho_I - 6*m_xz_I*rho_I + 11*m_yy_I*rho_I - 30*m_yz_I*rho_I + 11*m_zz_I*rho_I + omegaVar*m_xx_I*rho_I + 6*omegaVar*m_xy_I*rho_I + 6*omegaVar*m_xz_I*rho_I - 11*omegaVar*m_yy_I*rho_I + 30*omegaVar*m_yz_I*rho_I - 11*omegaVar*m_zz_I*rho_I))/(72*omegaVar - 24*ux0 + 12*uy0 + 12*uz0 - 36*omegaVar*ux0 + 18*omegaVar*uy0 + 18*omegaVar*uz0 - 30*omegaVar*ux0*ux0 - 75*omegaVar*uy0*uy0 - 75*omegaVar*uz0*uz0 + 90*omegaVar*ux0*uy0 + 90*omegaVar*ux0*uz0 + 180*omegaVar*uy0*uz0 + 248);

    m_xx_t45 = -(2*(rho - 87*m_xx_I*rho_I + 18*m_xy_I*rho_I + 18*m_xz_I*rho_I + 12*m_yy_I*rho_I - 45*m_yz_I*rho_I + 12*m_zz_I*rho_I + 12*rho*ux0 - 6*rho*uy0 - 6*rho*uz0))/(135*rho);
    m_xy_t90 = -(2*rho + 6*m_xx_I*rho_I - 63*m_xy_I*rho_I + 6*m_yy_I*rho_I - 3*m_zz_I*rho_I - 6*rho*ux0 + 6*rho*uy0)/(45*rho);
    m_xz_t90 = -(2*rho + 6*m_xx_I*rho_I - 63*m_xz_I*rho_I - 3*m_yy_I*rho_I + 6*m_zz_I*rho_I - 6*rho*ux0 + 6*rho*uz0)/(45*rho);
    m_yy_t45 = (2*(11*rho - 12*m_xx_I*rho_I - 18*m_xy_I*rho_I + 9*m_xz_I*rho_I + 105*m_yy_I*rho_I - 90*m_yz_I*rho_I + 24*m_zz_I*rho_I - 3*rho*ux0 + 6*rho*uy0 - 3*rho*uz0))/(135*rho);
    m_yz_t90 = -(2*rho - 3*m_xx_I*rho_I + 6*m_yy_I*rho_I - 27*m_yz_I*rho_I + 6*m_zz_I*rho_I)/(9*rho);
    m_zz_t45 = (2*(11*rho - 12*m_xx_I*rho_I + 9*m_xy_I*rho_I - 18*m_xz_I*rho_I + 24*m_yy_I*rho_I - 90*m_yz_I*rho_I + 105*m_zz_I*rho_I - 3*rho*ux0 - 3*rho*uy0 + 6*rho*uz0))/(135*rho);

    rhoVar = rho;

    break;
case 0b0000010100111111:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[4] + pop[6] + pop[8] + pop[10] + pop[12] + pop[13] + pop[15];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[15]);
    m_xy_I = inv_rho_I * (pop[8] - pop[13]);
    m_xz_I = inv_rho_I * (pop[10] - pop[15]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] - (1.0/3.0)*pop[15]);
    m_yz_I = inv_rho_I * (pop[12]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] + (2.0/3.0)*pop[15]);

    rho = (72*(23*rho_I - 6*m_xx_I*rho_I + 24*m_yy_I*rho_I - 57*m_yz_I*rho_I + 24*m_zz_I*rho_I + 6*omegaVar*m_xx_I*rho_I - 24*omegaVar*m_yy_I*rho_I + 57*omegaVar*m_yz_I*rho_I - 24*omegaVar*m_zz_I*rho_I))/(430*omegaVar + 660*uy0 + 660*uz0 + 30*omegaVar*uy0 + 30*omegaVar*uz0 - 69*omegaVar*ux0*ux0 - 690*omegaVar*uy0*uy0 - 690*omegaVar*uz0*uz0 + 414*omegaVar*uy0*uz0 + 720);

    m_xx_t45 = (2*(54*m_xx_I*rho_I - 2*rho - 9*m_yy_I*rho_I + 30*m_yz_I*rho_I - 9*m_zz_I*rho_I + 2*rho*uy0 + 2*rho*uz0))/(69*rho);
    m_xy_t90 = (6*m_xy_I*rho_I + rho*ux0)/(3*rho);
    m_xz_t90 = (6*m_xz_I*rho_I + rho*ux0)/(3*rho);
    m_yy_t45 = (16*rho - 18*m_xx_I*rho_I + 141*m_yy_I*rho_I - 102*m_yz_I*rho_I + 3*m_zz_I*rho_I + 30*rho*uy0 - 16*rho*uz0)/(69*rho);
    m_yz_t90 = (30*m_xx_I*rho_I - 19*rho - 51*m_yy_I*rho_I + 354*m_yz_I*rho_I - 51*m_zz_I*rho_I + 19*rho*uy0 + 19*rho*uz0)/(69*rho);
    m_zz_t45 = (16*rho - 18*m_xx_I*rho_I + 3*m_yy_I*rho_I - 102*m_yz_I*rho_I + 141*m_zz_I*rho_I - 16*rho*uy0 + 30*rho*uz0)/(69*rho);

    rhoVar = rho;

    break;
case 0b0000010101000000:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = rho_I;

    m_xx_t45 = (m_xx_I*rho_I)/rho;
    m_xy_t90 = (m_xy_I*rho_I)/rho;
    m_xz_t90 = (m_xz_I*rho_I)/rho;
    m_yy_t45 = (m_yy_I*rho_I)/rho;
    m_yz_t90 = (m_yz_I*rho_I)/rho;
    m_zz_t45 = (m_zz_I*rho_I)/rho;

    rhoVar = rho;

    break;
case 0b0000010101000001:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = rho_I;

    m_xx_t45 = (m_xx_I*rho_I)/rho;
    m_xy_t90 = (m_xy_I*rho_I)/rho;
    m_xz_t90 = (m_xz_I*rho_I)/rho;
    m_yy_t45 = (m_yy_I*rho_I)/rho;
    m_yz_t90 = (m_yz_I*rho_I)/rho;
    m_zz_t45 = (m_zz_I*rho_I)/rho;

    rhoVar = rho;

    break;
case 0b0000010101000010:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = rho_I;

    m_xx_t45 = (m_xx_I*rho_I)/rho;
    m_xy_t90 = (m_xy_I*rho_I)/rho;
    m_xz_t90 = (m_xz_I*rho_I)/rho;
    m_yy_t45 = (m_yy_I*rho_I)/rho;
    m_yz_t90 = (m_yz_I*rho_I)/rho;
    m_zz_t45 = (m_zz_I*rho_I)/rho;

    rhoVar = rho;

    break;
case 0b0000010101000011:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (24*(15*rho_I - m_xx_I*rho_I + 2*m_yy_I*rho_I + 6*m_yz_I*rho_I + 2*m_zz_I*rho_I + omegaVar*m_xx_I*rho_I - 2*omegaVar*m_yy_I*rho_I - 6*omegaVar*m_yz_I*rho_I - 2*omegaVar*m_zz_I*rho_I))/(6*omegaVar + 12*uy0 + 12*uz0 + 18*omegaVar*uy0 + 18*omegaVar*uz0 + 15*omegaVar*ux0*ux0 - 30*omegaVar*uy0*uy0 - 30*omegaVar*uz0*uz0 - 90*omegaVar*uy0*uz0 + 344);

    m_xx_t45 = -(2*(rho - 69*m_xx_I*rho_I + 3*m_yy_I*rho_I + 9*m_yz_I*rho_I + 3*m_zz_I*rho_I + 3*rho*uy0 + 3*rho*uz0))/(135*rho);
    m_xy_t90 = (m_xy_I*rho_I)/rho;
    m_xz_t90 = (m_xz_I*rho_I)/rho;
    m_yy_t45 = (4*rho - 6*m_xx_I*rho_I + 147*m_yy_I*rho_I + 36*m_yz_I*rho_I + 12*m_zz_I*rho_I + 12*rho*uy0 + 12*rho*uz0)/(135*rho);
    m_yz_t90 = (2*rho - 3*m_xx_I*rho_I + 6*m_yy_I*rho_I + 63*m_yz_I*rho_I + 6*m_zz_I*rho_I + 6*rho*uy0 + 6*rho*uz0)/(45*rho);
    m_zz_t45 = (4*rho - 6*m_xx_I*rho_I + 12*m_yy_I*rho_I + 36*m_yz_I*rho_I + 147*m_zz_I*rho_I + 12*rho*uy0 + 12*rho*uz0)/(135*rho);

    rhoVar = rho;

    break;
case 0b0000010101000100:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (24*(15*rho_I + 2*m_xx_I*rho_I - 6*m_xy_I*rho_I + 2*m_yy_I*rho_I - m_zz_I*rho_I - 2*omegaVar*m_xx_I*rho_I + 6*omegaVar*m_xy_I*rho_I - 2*omegaVar*m_yy_I*rho_I + omegaVar*m_zz_I*rho_I))/(6*omegaVar + 12*ux0 - 12*uy0 + 18*omegaVar*ux0 - 18*omegaVar*uy0 - 30*omegaVar*ux0*ux0 - 30*omegaVar*uy0*uy0 + 15*omegaVar*uz0*uz0 + 90*omegaVar*ux0*uy0 + 344);

    m_xx_t45 = (4*rho + 147*m_xx_I*rho_I - 36*m_xy_I*rho_I + 12*m_yy_I*rho_I - 6*m_zz_I*rho_I + 12*rho*ux0 - 12*rho*uy0)/(135*rho);
    m_xy_t90 = -(2*rho + 6*m_xx_I*rho_I - 63*m_xy_I*rho_I + 6*m_yy_I*rho_I - 3*m_zz_I*rho_I + 6*rho*ux0 - 6*rho*uy0)/(45*rho);
    m_xz_t90 = (m_xz_I*rho_I)/rho;
    m_yy_t45 = (4*rho + 12*m_xx_I*rho_I - 36*m_xy_I*rho_I + 147*m_yy_I*rho_I - 6*m_zz_I*rho_I + 12*rho*ux0 - 12*rho*uy0)/(135*rho);
    m_yz_t90 = (m_yz_I*rho_I)/rho;
    m_zz_t45 = -(2*(rho + 3*m_xx_I*rho_I - 9*m_xy_I*rho_I + 3*m_yy_I*rho_I - 69*m_zz_I*rho_I + 3*rho*ux0 - 3*rho*uy0))/(135*rho);

    rhoVar = rho;

    break;
case 0b0000010101000101:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[10] + pop[11] + pop[12] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[14]);
    m_xz_I = inv_rho_I * (pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (24*(15*rho_I + 4*m_xx_I*rho_I - 6*m_xy_I*rho_I + 6*m_xz_I*rho_I + m_yy_I*rho_I + m_zz_I*rho_I - 4*omegaVar*m_xx_I*rho_I + 6*omegaVar*m_xy_I*rho_I - 6*omegaVar*m_xz_I*rho_I - omegaVar*m_yy_I*rho_I - omegaVar*m_zz_I*rho_I))/(12*omegaVar + 24*ux0 - 12*uy0 + 12*uz0 + 36*omegaVar*ux0 - 18*omegaVar*uy0 + 18*omegaVar*uz0 - 60*omegaVar*ux0*ux0 - 15*omegaVar*uy0*uy0 - 15*omegaVar*uz0*uz0 + 90*omegaVar*ux0*uy0 - 90*omegaVar*ux0*uz0 + 328);

    m_xx_t45 = (8*rho + 159*m_xx_I*rho_I - 36*m_xy_I*rho_I + 36*m_xz_I*rho_I + 6*m_yy_I*rho_I + 6*m_zz_I*rho_I + 24*rho*ux0 - 12*rho*uy0 + 12*rho*uz0)/(135*rho);
    m_xy_t90 = -(2*rho + 6*m_xx_I*rho_I - 63*m_xy_I*rho_I + 6*m_yy_I*rho_I - 3*m_zz_I*rho_I + 6*rho*ux0 - 6*rho*uy0)/(45*rho);
    m_xz_t90 = (2*rho + 6*m_xx_I*rho_I + 63*m_xz_I*rho_I - 3*m_yy_I*rho_I + 6*m_zz_I*rho_I + 6*rho*ux0 + 6*rho*uz0)/(45*rho);
    m_yy_t45 = -(2*(18*m_xy_I*rho_I - 3*m_xx_I*rho_I - rho + 9*m_xz_I*rho_I - 75*m_yy_I*rho_I + 6*m_zz_I*rho_I - 3*rho*ux0 + 6*rho*uy0 + 3*rho*uz0))/(135*rho);
    m_yz_t90 = (m_yz_I*rho_I)/rho;
    m_zz_t45 = (2*(rho + 3*m_xx_I*rho_I + 9*m_xy_I*rho_I + 18*m_xz_I*rho_I - 6*m_yy_I*rho_I + 75*m_zz_I*rho_I + 3*rho*ux0 + 3*rho*uy0 + 6*rho*uz0))/(135*rho);

    rhoVar = rho;

    break;
case 0b0000010101000110:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (24*(15*rho_I + 2*m_xx_I*rho_I - 6*m_xy_I*rho_I + 2*m_yy_I*rho_I - m_zz_I*rho_I - 2*omegaVar*m_xx_I*rho_I + 6*omegaVar*m_xy_I*rho_I - 2*omegaVar*m_yy_I*rho_I + omegaVar*m_zz_I*rho_I))/(6*omegaVar + 12*ux0 - 12*uy0 + 18*omegaVar*ux0 - 18*omegaVar*uy0 - 30*omegaVar*ux0*ux0 - 30*omegaVar*uy0*uy0 + 15*omegaVar*uz0*uz0 + 90*omegaVar*ux0*uy0 + 344);

    m_xx_t45 = (4*rho + 147*m_xx_I*rho_I - 36*m_xy_I*rho_I + 12*m_yy_I*rho_I - 6*m_zz_I*rho_I + 12*rho*ux0 - 12*rho*uy0)/(135*rho);
    m_xy_t90 = -(2*rho + 6*m_xx_I*rho_I - 63*m_xy_I*rho_I + 6*m_yy_I*rho_I - 3*m_zz_I*rho_I + 6*rho*ux0 - 6*rho*uy0)/(45*rho);
    m_xz_t90 = (m_xz_I*rho_I)/rho;
    m_yy_t45 = (4*rho + 12*m_xx_I*rho_I - 36*m_xy_I*rho_I + 147*m_yy_I*rho_I - 6*m_zz_I*rho_I + 12*rho*ux0 - 12*rho*uy0)/(135*rho);
    m_yz_t90 = (m_yz_I*rho_I)/rho;
    m_zz_t45 = -(2*(rho + 3*m_xx_I*rho_I - 9*m_xy_I*rho_I + 3*m_yy_I*rho_I - 69*m_zz_I*rho_I + 3*rho*ux0 - 3*rho*uy0))/(135*rho);

    rhoVar = rho;

    break;
case 0b0000010101000111:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[10] + pop[12] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[14]);
    m_xz_I = inv_rho_I * (pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (24*(5*rho_I + m_xx_I*rho_I - 2*m_xy_I*rho_I + 2*m_xz_I*rho_I + m_yy_I*rho_I + 2*m_yz_I*rho_I + m_zz_I*rho_I - omegaVar*m_xx_I*rho_I + 2*omegaVar*m_xy_I*rho_I - 2*omegaVar*m_xz_I*rho_I - omegaVar*m_yy_I*rho_I - 2*omegaVar*m_yz_I*rho_I - omegaVar*m_zz_I*rho_I))/(6*omegaVar + 8*ux0 + 8*uz0 + 12*omegaVar*ux0 + 12*omegaVar*uz0 - 15*omegaVar*ux0*ux0 - 15*omegaVar*uy0*uy0 - 15*omegaVar*uz0*uz0 + 30*omegaVar*ux0*uy0 - 30*omegaVar*ux0*uz0 - 30*omegaVar*uy0*uz0 + 104);

    m_xx_t45 = (2*(rho + 27*m_xx_I*rho_I - 6*m_xy_I*rho_I + 6*m_xz_I*rho_I - 3*m_yz_I*rho_I + 4*rho*ux0 - 3*rho*uy0 + rho*uz0))/(45*rho);
    m_xy_t90 = -(2*rho + 6*m_xx_I*rho_I - 63*m_xy_I*rho_I + 6*m_yy_I*rho_I - 3*m_zz_I*rho_I + 6*rho*ux0 - 6*rho*uy0)/(45*rho);
    m_xz_t90 = (2*rho + 6*m_xx_I*rho_I + 63*m_xz_I*rho_I - 3*m_yy_I*rho_I + 6*m_zz_I*rho_I + 6*rho*ux0 + 6*rho*uz0)/(45*rho);
    m_yy_t45 = (2*(rho - 6*m_xy_I*rho_I - 3*m_xz_I*rho_I + 27*m_yy_I*rho_I + 6*m_yz_I*rho_I + rho*ux0 + rho*uz0))/(45*rho);
    m_yz_t90 = (2*rho - 3*m_xx_I*rho_I + 6*m_yy_I*rho_I + 63*m_yz_I*rho_I + 6*m_zz_I*rho_I + 6*rho*uy0 + 6*rho*uz0)/(45*rho);
    m_zz_t45 = (2*(rho + 3*m_xy_I*rho_I + 6*m_xz_I*rho_I + 6*m_yz_I*rho_I + 27*m_zz_I*rho_I + rho*ux0 + 3*rho*uy0 + 4*rho*uz0))/(45*rho);

    rhoVar = rho;

    break;
case 0b0000010101001000:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = rho_I;

    m_xx_t45 = (m_xx_I*rho_I)/rho;
    m_xy_t90 = (m_xy_I*rho_I)/rho;
    m_xz_t90 = (m_xz_I*rho_I)/rho;
    m_yy_t45 = (m_yy_I*rho_I)/rho;
    m_yz_t90 = (m_yz_I*rho_I)/rho;
    m_zz_t45 = (m_zz_I*rho_I)/rho;

    rhoVar = rho;

    break;
case 0b0000010101001001:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = rho_I;

    m_xx_t45 = (m_xx_I*rho_I)/rho;
    m_xy_t90 = (m_xy_I*rho_I)/rho;
    m_xz_t90 = (m_xz_I*rho_I)/rho;
    m_yy_t45 = (m_yy_I*rho_I)/rho;
    m_yz_t90 = (m_yz_I*rho_I)/rho;
    m_zz_t45 = (m_zz_I*rho_I)/rho;

    rhoVar = rho;

    break;
case 0b0000010101001010:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (24*(15*rho_I + 2*m_xx_I*rho_I - 6*m_xz_I*rho_I - m_yy_I*rho_I + 2*m_zz_I*rho_I - 2*omegaVar*m_xx_I*rho_I + 6*omegaVar*m_xz_I*rho_I + omegaVar*m_yy_I*rho_I - 2*omegaVar*m_zz_I*rho_I))/(6*omegaVar - 12*ux0 + 12*uz0 - 18*omegaVar*ux0 + 18*omegaVar*uz0 - 30*omegaVar*ux0*ux0 + 15*omegaVar*uy0*uy0 - 30*omegaVar*uz0*uz0 + 90*omegaVar*ux0*uz0 + 344);

    m_xx_t45 = (4*rho + 147*m_xx_I*rho_I - 36*m_xz_I*rho_I - 6*m_yy_I*rho_I + 12*m_zz_I*rho_I - 12*rho*ux0 + 12*rho*uz0)/(135*rho);
    m_xy_t90 = (m_xy_I*rho_I)/rho;
    m_xz_t90 = -(2*rho + 6*m_xx_I*rho_I - 63*m_xz_I*rho_I - 3*m_yy_I*rho_I + 6*m_zz_I*rho_I - 6*rho*ux0 + 6*rho*uz0)/(45*rho);
    m_yy_t45 = -(2*(rho + 3*m_xx_I*rho_I - 9*m_xz_I*rho_I - 69*m_yy_I*rho_I + 3*m_zz_I*rho_I - 3*rho*ux0 + 3*rho*uz0))/(135*rho);
    m_yz_t90 = (m_yz_I*rho_I)/rho;
    m_zz_t45 = (4*rho + 12*m_xx_I*rho_I - 36*m_xz_I*rho_I - 6*m_yy_I*rho_I + 147*m_zz_I*rho_I - 12*rho*ux0 + 12*rho*uz0)/(135*rho);

    rhoVar = rho;

    break;
case 0b0000010101001011:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[12] + pop[13] + pop[14] + pop[15] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (24*(15*rho_I + m_xx_I*rho_I - 6*m_xz_I*rho_I + m_yy_I*rho_I + 6*m_yz_I*rho_I + 4*m_zz_I*rho_I - omegaVar*m_xx_I*rho_I + 6*omegaVar*m_xz_I*rho_I - omegaVar*m_yy_I*rho_I - 6*omegaVar*m_yz_I*rho_I - 4*omegaVar*m_zz_I*rho_I))/(12*omegaVar - 12*ux0 + 12*uy0 + 24*uz0 - 18*omegaVar*ux0 + 18*omegaVar*uy0 + 36*omegaVar*uz0 - 15*omegaVar*ux0*ux0 - 15*omegaVar*uy0*uy0 - 60*omegaVar*uz0*uz0 + 90*omegaVar*ux0*uz0 - 90*omegaVar*uy0*uz0 + 328);

    m_xx_t45 = -(2*(18*m_xz_I*rho_I - 75*m_xx_I*rho_I - rho + 6*m_yy_I*rho_I + 9*m_yz_I*rho_I - 3*m_zz_I*rho_I + 6*rho*ux0 + 3*rho*uy0 - 3*rho*uz0))/(135*rho);
    m_xy_t90 = (m_xy_I*rho_I)/rho;
    m_xz_t90 = -(2*rho + 6*m_xx_I*rho_I - 63*m_xz_I*rho_I - 3*m_yy_I*rho_I + 6*m_zz_I*rho_I - 6*rho*ux0 + 6*rho*uz0)/(45*rho);
    m_yy_t45 = (2*(rho - 6*m_xx_I*rho_I + 9*m_xz_I*rho_I + 75*m_yy_I*rho_I + 18*m_yz_I*rho_I + 3*m_zz_I*rho_I + 3*rho*ux0 + 6*rho*uy0 + 3*rho*uz0))/(135*rho);
    m_yz_t90 = (2*rho - 3*m_xx_I*rho_I + 6*m_yy_I*rho_I + 63*m_yz_I*rho_I + 6*m_zz_I*rho_I + 6*rho*uy0 + 6*rho*uz0)/(45*rho);
    m_zz_t45 = (8*rho + 6*m_xx_I*rho_I - 36*m_xz_I*rho_I + 6*m_yy_I*rho_I + 36*m_yz_I*rho_I + 159*m_zz_I*rho_I - 12*rho*ux0 + 12*rho*uy0 + 24*rho*uz0)/(135*rho);

    rhoVar = rho;

    break;
case 0b0000010101001100:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[14] + pop[15] + pop[16] + pop[17];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17]);

    rho = (24*(15*rho_I + m_xx_I*rho_I - 6*m_xy_I*rho_I + 4*m_yy_I*rho_I - 6*m_yz_I*rho_I + m_zz_I*rho_I - omegaVar*m_xx_I*rho_I + 6*omegaVar*m_xy_I*rho_I - 4*omegaVar*m_yy_I*rho_I + 6*omegaVar*m_yz_I*rho_I - omegaVar*m_zz_I*rho_I))/(12*omegaVar + 12*ux0 - 24*uy0 + 12*uz0 + 18*omegaVar*ux0 - 36*omegaVar*uy0 + 18*omegaVar*uz0 - 15*omegaVar*ux0*ux0 - 60*omegaVar*uy0*uy0 - 15*omegaVar*uz0*uz0 + 90*omegaVar*ux0*uy0 + 90*omegaVar*uy0*uz0 + 328);

    m_xx_t45 = (2*(rho + 75*m_xx_I*rho_I - 18*m_xy_I*rho_I + 3*m_yy_I*rho_I + 9*m_yz_I*rho_I - 6*m_zz_I*rho_I + 6*rho*ux0 - 3*rho*uy0 - 3*rho*uz0))/(135*rho);
    m_xy_t90 = -(2*rho + 6*m_xx_I*rho_I - 63*m_xy_I*rho_I + 6*m_yy_I*rho_I - 3*m_zz_I*rho_I + 6*rho*ux0 - 6*rho*uy0)/(45*rho);
    m_xz_t90 = (m_xz_I*rho_I)/rho;
    m_yy_t45 = (8*rho + 6*m_xx_I*rho_I - 36*m_xy_I*rho_I + 159*m_yy_I*rho_I - 36*m_yz_I*rho_I + 6*m_zz_I*rho_I + 12*rho*ux0 - 24*rho*uy0 + 12*rho*uz0)/(135*rho);
    m_yz_t90 = -(2*rho - 3*m_xx_I*rho_I + 6*m_yy_I*rho_I - 63*m_yz_I*rho_I + 6*m_zz_I*rho_I - 6*rho*uy0 + 6*rho*uz0)/(45*rho);
    m_zz_t45 = (2*(rho - 6*m_xx_I*rho_I + 9*m_xy_I*rho_I + 3*m_yy_I*rho_I - 18*m_yz_I*rho_I + 75*m_zz_I*rho_I - 3*rho*ux0 - 3*rho*uy0 + 6*rho*uz0))/(135*rho);

    rhoVar = rho;

    break;
case 0b0000010101001101:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[10] + pop[11] + pop[12] + pop[14] + pop[15] + pop[16] + pop[17];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[14]);
    m_xz_I = inv_rho_I * (pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17]);

    rho = (24*(5*rho_I + m_xx_I*rho_I - 2*m_xy_I*rho_I + 2*m_xz_I*rho_I + m_yy_I*rho_I - 2*m_yz_I*rho_I + m_zz_I*rho_I - omegaVar*m_xx_I*rho_I + 2*omegaVar*m_xy_I*rho_I - 2*omegaVar*m_xz_I*rho_I - omegaVar*m_yy_I*rho_I + 2*omegaVar*m_yz_I*rho_I - omegaVar*m_zz_I*rho_I))/(6*omegaVar + 8*ux0 - 8*uy0 + 8*uz0 + 12*omegaVar*ux0 - 12*omegaVar*uy0 + 12*omegaVar*uz0 - 15*omegaVar*ux0*ux0 - 15*omegaVar*uy0*uy0 - 15*omegaVar*uz0*uz0 + 30*omegaVar*ux0*uy0 - 30*omegaVar*ux0*uz0 + 30*omegaVar*uy0*uz0 + 104);

    m_xx_t45 = (2*(rho + 27*m_xx_I*rho_I - 6*m_xy_I*rho_I + 6*m_xz_I*rho_I + 3*m_yz_I*rho_I + 4*rho*ux0 - rho*uy0 + rho*uz0))/(45*rho);
    m_xy_t90 = -(2*rho + 6*m_xx_I*rho_I - 63*m_xy_I*rho_I + 6*m_yy_I*rho_I - 3*m_zz_I*rho_I + 6*rho*ux0 - 6*rho*uy0)/(45*rho);
    m_xz_t90 = (2*rho + 6*m_xx_I*rho_I + 63*m_xz_I*rho_I - 3*m_yy_I*rho_I + 6*m_zz_I*rho_I + 6*rho*ux0 + 6*rho*uz0)/(45*rho);
    m_yy_t45 = (2*(rho - 6*m_xy_I*rho_I - 3*m_xz_I*rho_I + 27*m_yy_I*rho_I - 6*m_yz_I*rho_I + rho*ux0 - 4*rho*uy0 + rho*uz0))/(45*rho);
    m_yz_t90 = -(2*rho - 3*m_xx_I*rho_I + 6*m_yy_I*rho_I - 63*m_yz_I*rho_I + 6*m_zz_I*rho_I - 6*rho*uy0 + 6*rho*uz0)/(45*rho);
    m_zz_t45 = (2*(rho + 3*m_xy_I*rho_I + 6*m_xz_I*rho_I - 6*m_yz_I*rho_I + 27*m_zz_I*rho_I + rho*ux0 - rho*uy0 + 4*rho*uz0))/(45*rho);

    rhoVar = rho;

    break;
case 0b0000010101001110:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[14] + pop[15] + pop[17];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] - (1.0/3.0)*pop[17]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] + (2.0/3.0)*pop[17]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[17]);

    rho = (24*(5*rho_I + m_xx_I*rho_I - 2*m_xy_I*rho_I - 2*m_xz_I*rho_I + m_yy_I*rho_I - 2*m_yz_I*rho_I + m_zz_I*rho_I - omegaVar*m_xx_I*rho_I + 2*omegaVar*m_xy_I*rho_I + 2*omegaVar*m_xz_I*rho_I - omegaVar*m_yy_I*rho_I + 2*omegaVar*m_yz_I*rho_I - omegaVar*m_zz_I*rho_I))/(6*omegaVar - 8*uy0 + 8*uz0 - 12*omegaVar*uy0 + 12*omegaVar*uz0 - 15*omegaVar*ux0*ux0 - 15*omegaVar*uy0*uy0 - 15*omegaVar*uz0*uz0 + 30*omegaVar*ux0*uy0 + 30*omegaVar*ux0*uz0 + 30*omegaVar*uy0*uz0 + 104);

    m_xx_t45 = (2*(rho + 27*m_xx_I*rho_I - 6*m_xy_I*rho_I - 6*m_xz_I*rho_I + 3*m_yz_I*rho_I - rho*uy0 + rho*uz0))/(45*rho);
    m_xy_t90 = -(2*rho + 6*m_xx_I*rho_I - 63*m_xy_I*rho_I + 6*m_yy_I*rho_I - 3*m_zz_I*rho_I + 6*rho*ux0 - 6*rho*uy0)/(45*rho);
    m_xz_t90 = -(2*rho + 6*m_xx_I*rho_I - 63*m_xz_I*rho_I - 3*m_yy_I*rho_I + 6*m_zz_I*rho_I - 6*rho*ux0 + 6*rho*uz0)/(45*rho);
    m_yy_t45 = (2*(rho - 6*m_xy_I*rho_I + 3*m_xz_I*rho_I + 27*m_yy_I*rho_I - 6*m_yz_I*rho_I + 3*rho*ux0 - 4*rho*uy0 + rho*uz0))/(45*rho);
    m_yz_t90 = -(2*rho - 3*m_xx_I*rho_I + 6*m_yy_I*rho_I - 63*m_yz_I*rho_I + 6*m_zz_I*rho_I - 6*rho*uy0 + 6*rho*uz0)/(45*rho);
    m_zz_t45 = (2*(rho + 3*m_xy_I*rho_I - 6*m_xz_I*rho_I - 6*m_yz_I*rho_I + 27*m_zz_I*rho_I - 3*rho*ux0 - rho*uy0 + 4*rho*uz0))/(45*rho);

    rhoVar = rho;

    break;
case 0b0000010101001111:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[6] + pop[7] + pop[8] + pop[10] + pop[12] + pop[14] + pop[15] + pop[17];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] - (1.0/3.0)*pop[17]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[14]);
    m_xz_I = inv_rho_I * (pop[10] - pop[15]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] + (2.0/3.0)*pop[17]);
    m_yz_I = inv_rho_I * (pop[12] - pop[17]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[17]);

    rho = (72*(107*rho_I + 12*m_xx_I*rho_I - 33*m_xy_I*rho_I + 12*m_yy_I*rho_I + 72*m_zz_I*rho_I - 12*omegaVar*m_xx_I*rho_I + 33*omegaVar*m_xy_I*rho_I - 12*omegaVar*m_yy_I*rho_I - 72*omegaVar*m_zz_I*rho_I))/(626*omegaVar + 492*ux0 - 492*uy0 + 2124*uz0 + 150*omegaVar*ux0 - 150*omegaVar*uy0 + 1728*omegaVar*uz0 - 642*omegaVar*ux0*ux0 - 642*omegaVar*uy0*uy0 - 3531*omegaVar*uz0*uz0 + 1926*omegaVar*ux0*uy0 + 5580);

    m_xx_t45 = -(4*(24*m_xy_I*rho_I - 106*m_xx_I*rho_I - 2*rho + m_yy_I*rho_I + 6*m_zz_I*rho_I - 8*rho*ux0 + 8*rho*uy0 + 2*rho*uz0))/(321*rho);
    m_xy_t90 = -(11*rho + 48*m_xx_I*rho_I - 453*m_xy_I*rho_I + 48*m_yy_I*rho_I - 33*m_zz_I*rho_I + 44*rho*ux0 - 44*rho*uy0 - 11*rho*uz0)/(321*rho);
    m_xz_t90 = (6*m_xz_I*rho_I + rho*ux0)/(3*rho);
    m_yy_t45 = -(4*(m_xx_I*rho_I - 2*rho + 24*m_xy_I*rho_I - 106*m_yy_I*rho_I + 6*m_zz_I*rho_I - 8*rho*ux0 + 8*rho*uy0 + 2*rho*uz0))/(321*rho);
    m_yz_t90 = (6*m_yz_I*rho_I + rho*uy0)/(3*rho);
    m_zz_t45 = (2*(24*rho - 12*m_xx_I*rho_I + 33*m_xy_I*rho_I - 12*m_yy_I*rho_I + 249*m_zz_I*rho_I - 11*rho*ux0 + 11*rho*uy0 + 83*rho*uz0))/(321*rho);

    rhoVar = rho;

    break;
case 0b0000010101010000:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (24*(15*rho_I + 2*m_xx_I*rho_I - 6*m_xz_I*rho_I - m_yy_I*rho_I + 2*m_zz_I*rho_I - 2*omegaVar*m_xx_I*rho_I + 6*omegaVar*m_xz_I*rho_I + omegaVar*m_yy_I*rho_I - 2*omegaVar*m_zz_I*rho_I))/(6*omegaVar + 12*ux0 - 12*uz0 + 18*omegaVar*ux0 - 18*omegaVar*uz0 - 30*omegaVar*ux0*ux0 + 15*omegaVar*uy0*uy0 - 30*omegaVar*uz0*uz0 + 90*omegaVar*ux0*uz0 + 344);

    m_xx_t45 = (4*rho + 147*m_xx_I*rho_I - 36*m_xz_I*rho_I - 6*m_yy_I*rho_I + 12*m_zz_I*rho_I + 12*rho*ux0 - 12*rho*uz0)/(135*rho);
    m_xy_t90 = (m_xy_I*rho_I)/rho;
    m_xz_t90 = -(2*rho + 6*m_xx_I*rho_I - 63*m_xz_I*rho_I - 3*m_yy_I*rho_I + 6*m_zz_I*rho_I + 6*rho*ux0 - 6*rho*uz0)/(45*rho);
    m_yy_t45 = -(2*(rho + 3*m_xx_I*rho_I - 9*m_xz_I*rho_I - 69*m_yy_I*rho_I + 3*m_zz_I*rho_I + 3*rho*ux0 - 3*rho*uz0))/(135*rho);
    m_yz_t90 = (m_yz_I*rho_I)/rho;
    m_zz_t45 = (4*rho + 12*m_xx_I*rho_I - 36*m_xz_I*rho_I - 6*m_yy_I*rho_I + 147*m_zz_I*rho_I + 12*rho*ux0 - 12*rho*uz0)/(135*rho);

    rhoVar = rho;

    break;
case 0b0000010101010001:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (24*(15*rho_I + 4*m_xx_I*rho_I + 6*m_xy_I*rho_I - 6*m_xz_I*rho_I + m_yy_I*rho_I + m_zz_I*rho_I - 4*omegaVar*m_xx_I*rho_I - 6*omegaVar*m_xy_I*rho_I + 6*omegaVar*m_xz_I*rho_I - omegaVar*m_yy_I*rho_I - omegaVar*m_zz_I*rho_I))/(12*omegaVar + 24*ux0 + 12*uy0 - 12*uz0 + 36*omegaVar*ux0 + 18*omegaVar*uy0 - 18*omegaVar*uz0 - 60*omegaVar*ux0*ux0 - 15*omegaVar*uy0*uy0 - 15*omegaVar*uz0*uz0 - 90*omegaVar*ux0*uy0 + 90*omegaVar*ux0*uz0 + 328);

    m_xx_t45 = (8*rho + 159*m_xx_I*rho_I + 36*m_xy_I*rho_I - 36*m_xz_I*rho_I + 6*m_yy_I*rho_I + 6*m_zz_I*rho_I + 24*rho*ux0 + 12*rho*uy0 - 12*rho*uz0)/(135*rho);
    m_xy_t90 = (2*rho + 6*m_xx_I*rho_I + 63*m_xy_I*rho_I + 6*m_yy_I*rho_I - 3*m_zz_I*rho_I + 6*rho*ux0 + 6*rho*uy0)/(45*rho);
    m_xz_t90 = -(2*rho + 6*m_xx_I*rho_I - 63*m_xz_I*rho_I - 3*m_yy_I*rho_I + 6*m_zz_I*rho_I + 6*rho*ux0 - 6*rho*uz0)/(45*rho);
    m_yy_t45 = (2*(rho + 3*m_xx_I*rho_I + 18*m_xy_I*rho_I + 9*m_xz_I*rho_I + 75*m_yy_I*rho_I - 6*m_zz_I*rho_I + 3*rho*ux0 + 6*rho*uy0 + 3*rho*uz0))/(135*rho);
    m_yz_t90 = (m_yz_I*rho_I)/rho;
    m_zz_t45 = -(2*(9*m_xy_I*rho_I - 3*m_xx_I*rho_I - rho + 18*m_xz_I*rho_I + 6*m_yy_I*rho_I - 75*m_zz_I*rho_I - 3*rho*ux0 + 3*rho*uy0 + 6*rho*uz0))/(135*rho);

    rhoVar = rho;

    break;
case 0b0000010101010010:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (24*(15*rho_I + 2*m_xx_I*rho_I - 6*m_xz_I*rho_I - m_yy_I*rho_I + 2*m_zz_I*rho_I - 2*omegaVar*m_xx_I*rho_I + 6*omegaVar*m_xz_I*rho_I + omegaVar*m_yy_I*rho_I - 2*omegaVar*m_zz_I*rho_I))/(6*omegaVar + 12*ux0 - 12*uz0 + 18*omegaVar*ux0 - 18*omegaVar*uz0 - 30*omegaVar*ux0*ux0 + 15*omegaVar*uy0*uy0 - 30*omegaVar*uz0*uz0 + 90*omegaVar*ux0*uz0 + 344);

    m_xx_t45 = (4*rho + 147*m_xx_I*rho_I - 36*m_xz_I*rho_I - 6*m_yy_I*rho_I + 12*m_zz_I*rho_I + 12*rho*ux0 - 12*rho*uz0)/(135*rho);
    m_xy_t90 = (m_xy_I*rho_I)/rho;
    m_xz_t90 = -(2*rho + 6*m_xx_I*rho_I - 63*m_xz_I*rho_I - 3*m_yy_I*rho_I + 6*m_zz_I*rho_I + 6*rho*ux0 - 6*rho*uz0)/(45*rho);
    m_yy_t45 = -(2*(rho + 3*m_xx_I*rho_I - 9*m_xz_I*rho_I - 69*m_yy_I*rho_I + 3*m_zz_I*rho_I + 3*rho*ux0 - 3*rho*uz0))/(135*rho);
    m_yz_t90 = (m_yz_I*rho_I)/rho;
    m_zz_t45 = (4*rho + 12*m_xx_I*rho_I - 36*m_xz_I*rho_I - 6*m_yy_I*rho_I + 147*m_zz_I*rho_I + 12*rho*ux0 - 12*rho*uz0)/(135*rho);

    rhoVar = rho;

    break;
case 0b0000010101010011:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[8] + pop[9] + pop[10] + pop[12] + pop[13] + pop[14] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (24*(5*rho_I + m_xx_I*rho_I + 2*m_xy_I*rho_I - 2*m_xz_I*rho_I + m_yy_I*rho_I + 2*m_yz_I*rho_I + m_zz_I*rho_I - omegaVar*m_xx_I*rho_I - 2*omegaVar*m_xy_I*rho_I + 2*omegaVar*m_xz_I*rho_I - omegaVar*m_yy_I*rho_I - 2*omegaVar*m_yz_I*rho_I - omegaVar*m_zz_I*rho_I))/(6*omegaVar + 8*ux0 + 8*uy0 + 12*omegaVar*ux0 + 12*omegaVar*uy0 - 15*omegaVar*ux0*ux0 - 15*omegaVar*uy0*uy0 - 15*omegaVar*uz0*uz0 - 30*omegaVar*ux0*uy0 + 30*omegaVar*ux0*uz0 - 30*omegaVar*uy0*uz0 + 104);

    m_xx_t45 = (2*(rho + 27*m_xx_I*rho_I + 6*m_xy_I*rho_I - 6*m_xz_I*rho_I - 3*m_yz_I*rho_I + 4*rho*ux0 + rho*uy0 - 3*rho*uz0))/(45*rho);
    m_xy_t90 = (2*rho + 6*m_xx_I*rho_I + 63*m_xy_I*rho_I + 6*m_yy_I*rho_I - 3*m_zz_I*rho_I + 6*rho*ux0 + 6*rho*uy0)/(45*rho);
    m_xz_t90 = -(2*rho + 6*m_xx_I*rho_I - 63*m_xz_I*rho_I - 3*m_yy_I*rho_I + 6*m_zz_I*rho_I + 6*rho*ux0 - 6*rho*uz0)/(45*rho);
    m_yy_t45 = (2*(rho + 6*m_xy_I*rho_I + 3*m_xz_I*rho_I + 27*m_yy_I*rho_I + 6*m_yz_I*rho_I + rho*ux0 + 4*rho*uy0 + 3*rho*uz0))/(45*rho);
    m_yz_t90 = (2*rho - 3*m_xx_I*rho_I + 6*m_yy_I*rho_I + 63*m_yz_I*rho_I + 6*m_zz_I*rho_I + 6*rho*uy0 + 6*rho*uz0)/(45*rho);
    m_zz_t45 = (2*(rho - 3*m_xy_I*rho_I - 6*m_xz_I*rho_I + 6*m_yz_I*rho_I + 27*m_zz_I*rho_I + rho*ux0 + rho*uy0))/(45*rho);

    rhoVar = rho;

    break;
case 0b0000010101010100:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[14] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = -(24*(15*rho_I + 4*m_xx_I*rho_I - 6*m_xy_I*rho_I - 6*m_xz_I*rho_I + m_yy_I*rho_I + m_zz_I*rho_I - 4*omegaVar*m_xx_I*rho_I + 6*omegaVar*m_xy_I*rho_I + 6*omegaVar*m_xz_I*rho_I - omegaVar*m_yy_I*rho_I - omegaVar*m_zz_I*rho_I))/(12*uy0 - 24*ux0 - 12*omegaVar + 12*uz0 - 36*omegaVar*ux0 + 18*omegaVar*uy0 + 18*omegaVar*uz0 + 60*omegaVar*ux0*ux0 + 15*omegaVar*uy0*uy0 + 15*omegaVar*uz0*uz0 - 90*omegaVar*ux0*uy0 - 90*omegaVar*ux0*uz0 - 328);

    m_xx_t45 = (8*rho + 159*m_xx_I*rho_I - 36*m_xy_I*rho_I - 36*m_xz_I*rho_I + 6*m_yy_I*rho_I + 6*m_zz_I*rho_I + 24*rho*ux0 - 12*rho*uy0 - 12*rho*uz0)/(135*rho);
    m_xy_t90 = -(2*rho + 6*m_xx_I*rho_I - 63*m_xy_I*rho_I + 6*m_yy_I*rho_I - 3*m_zz_I*rho_I + 6*rho*ux0 - 6*rho*uy0)/(45*rho);
    m_xz_t90 = -(2*rho + 6*m_xx_I*rho_I - 63*m_xz_I*rho_I - 3*m_yy_I*rho_I + 6*m_zz_I*rho_I + 6*rho*ux0 - 6*rho*uz0)/(45*rho);
    m_yy_t45 = (2*(rho + 3*m_xx_I*rho_I - 18*m_xy_I*rho_I + 9*m_xz_I*rho_I + 75*m_yy_I*rho_I - 6*m_zz_I*rho_I + 3*rho*ux0 - 6*rho*uy0 + 3*rho*uz0))/(135*rho);
    m_yz_t90 = (m_yz_I*rho_I)/rho;
    m_zz_t45 = (2*(rho + 3*m_xx_I*rho_I + 9*m_xy_I*rho_I - 18*m_xz_I*rho_I - 6*m_yy_I*rho_I + 75*m_zz_I*rho_I + 3*rho*ux0 + 3*rho*uy0 - 6*rho*uz0))/(135*rho);

    rhoVar = rho;

    break;
case 0b0000010101010101:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[8] + pop[10] + pop[11] + pop[12] + pop[14] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[8] - pop[14]);
    m_xz_I = inv_rho_I * (pop[10] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (3*rho_I*(3*m_xx_I - 3*omegaVar*m_xx_I + 4))/(omegaVar + 3*ux0 + 3*omegaVar*ux0 - 6*omegaVar*ux0*ux0 + 9);

    m_xx_t45 = (rho + 9*m_xx_I*rho_I + 3*rho*ux0)/(6*rho);
    m_xy_t90 = (6*m_xy_I*rho_I + rho*uy0)/(3*rho);
    m_xz_t90 = (6*m_xz_I*rho_I + rho*uz0)/(3*rho);
    m_yy_t45 = (4*(10*m_yy_I*rho_I - m_zz_I*rho_I))/(33*rho);
    m_yz_t90 = (m_yz_I*rho_I)/rho;
    m_zz_t45 = -(4*(m_yy_I*rho_I - 10*m_zz_I*rho_I))/(33*rho);

    rhoVar = rho;

    break;
case 0b0000010101010110:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[14] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = -(24*(15*rho_I + 4*m_xx_I*rho_I - 6*m_xy_I*rho_I - 6*m_xz_I*rho_I + m_yy_I*rho_I + m_zz_I*rho_I - 4*omegaVar*m_xx_I*rho_I + 6*omegaVar*m_xy_I*rho_I + 6*omegaVar*m_xz_I*rho_I - omegaVar*m_yy_I*rho_I - omegaVar*m_zz_I*rho_I))/(12*uy0 - 24*ux0 - 12*omegaVar + 12*uz0 - 36*omegaVar*ux0 + 18*omegaVar*uy0 + 18*omegaVar*uz0 + 60*omegaVar*ux0*ux0 + 15*omegaVar*uy0*uy0 + 15*omegaVar*uz0*uz0 - 90*omegaVar*ux0*uy0 - 90*omegaVar*ux0*uz0 - 328);

    m_xx_t45 = (8*rho + 159*m_xx_I*rho_I - 36*m_xy_I*rho_I - 36*m_xz_I*rho_I + 6*m_yy_I*rho_I + 6*m_zz_I*rho_I + 24*rho*ux0 - 12*rho*uy0 - 12*rho*uz0)/(135*rho);
    m_xy_t90 = -(2*rho + 6*m_xx_I*rho_I - 63*m_xy_I*rho_I + 6*m_yy_I*rho_I - 3*m_zz_I*rho_I + 6*rho*ux0 - 6*rho*uy0)/(45*rho);
    m_xz_t90 = -(2*rho + 6*m_xx_I*rho_I - 63*m_xz_I*rho_I - 3*m_yy_I*rho_I + 6*m_zz_I*rho_I + 6*rho*ux0 - 6*rho*uz0)/(45*rho);
    m_yy_t45 = (2*(rho + 3*m_xx_I*rho_I - 18*m_xy_I*rho_I + 9*m_xz_I*rho_I + 75*m_yy_I*rho_I - 6*m_zz_I*rho_I + 3*rho*ux0 - 6*rho*uy0 + 3*rho*uz0))/(135*rho);
    m_yz_t90 = (m_yz_I*rho_I)/rho;
    m_zz_t45 = (2*(rho + 3*m_xx_I*rho_I + 9*m_xy_I*rho_I - 18*m_xz_I*rho_I - 6*m_yy_I*rho_I + 75*m_zz_I*rho_I + 3*rho*ux0 + 3*rho*uy0 - 6*rho*uz0))/(135*rho);

    rhoVar = rho;

    break;
case 0b0000010101010111:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[8] + pop[10] + pop[12] + pop[14] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[8] - pop[14]);
    m_xz_I = inv_rho_I * (pop[10] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (72*(107*rho_I + 72*m_xx_I*rho_I + 12*m_yy_I*rho_I + 33*m_yz_I*rho_I + 12*m_zz_I*rho_I - 72*omegaVar*m_xx_I*rho_I - 12*omegaVar*m_yy_I*rho_I - 33*omegaVar*m_yz_I*rho_I - 12*omegaVar*m_zz_I*rho_I))/(626*omegaVar + 2124*ux0 + 492*uy0 + 492*uz0 + 1728*omegaVar*ux0 + 150*omegaVar*uy0 + 150*omegaVar*uz0 - 3531*omegaVar*ux0*ux0 - 642*omegaVar*uy0*uy0 - 642*omegaVar*uz0*uz0 - 1926*omegaVar*uy0*uz0 + 5580);

    m_xx_t45 = -(2*(12*m_yy_I*rho_I - 249*m_xx_I*rho_I - 24*rho + 33*m_yz_I*rho_I + 12*m_zz_I*rho_I - 83*rho*ux0 + 11*rho*uy0 + 11*rho*uz0))/(321*rho);
    m_xy_t90 = (6*m_xy_I*rho_I + rho*uy0)/(3*rho);
    m_xz_t90 = (6*m_xz_I*rho_I + rho*uz0)/(3*rho);
    m_yy_t45 = (4*(2*rho - 6*m_xx_I*rho_I + 106*m_yy_I*rho_I + 24*m_yz_I*rho_I - m_zz_I*rho_I - 2*rho*ux0 + 8*rho*uy0 + 8*rho*uz0))/(321*rho);
    m_yz_t90 = (11*rho - 33*m_xx_I*rho_I + 48*m_yy_I*rho_I + 453*m_yz_I*rho_I + 48*m_zz_I*rho_I - 11*rho*ux0 + 44*rho*uy0 + 44*rho*uz0)/(321*rho);
    m_zz_t45 = (4*(2*rho - 6*m_xx_I*rho_I - m_yy_I*rho_I + 24*m_yz_I*rho_I + 106*m_zz_I*rho_I - 2*rho*ux0 + 8*rho*uy0 + 8*rho*uz0))/(321*rho);

    rhoVar = rho;

    break;
case 0b0000010101011000:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (24*(15*rho_I + 2*m_xx_I*rho_I - 6*m_xz_I*rho_I - m_yy_I*rho_I + 2*m_zz_I*rho_I - 2*omegaVar*m_xx_I*rho_I + 6*omegaVar*m_xz_I*rho_I + omegaVar*m_yy_I*rho_I - 2*omegaVar*m_zz_I*rho_I))/(6*omegaVar + 12*ux0 - 12*uz0 + 18*omegaVar*ux0 - 18*omegaVar*uz0 - 30*omegaVar*ux0*ux0 + 15*omegaVar*uy0*uy0 - 30*omegaVar*uz0*uz0 + 90*omegaVar*ux0*uz0 + 344);

    m_xx_t45 = (4*rho + 147*m_xx_I*rho_I - 36*m_xz_I*rho_I - 6*m_yy_I*rho_I + 12*m_zz_I*rho_I + 12*rho*ux0 - 12*rho*uz0)/(135*rho);
    m_xy_t90 = (m_xy_I*rho_I)/rho;
    m_xz_t90 = -(2*rho + 6*m_xx_I*rho_I - 63*m_xz_I*rho_I - 3*m_yy_I*rho_I + 6*m_zz_I*rho_I + 6*rho*ux0 - 6*rho*uz0)/(45*rho);
    m_yy_t45 = -(2*(rho + 3*m_xx_I*rho_I - 9*m_xz_I*rho_I - 69*m_yy_I*rho_I + 3*m_zz_I*rho_I + 3*rho*ux0 - 3*rho*uz0))/(135*rho);
    m_yz_t90 = (m_yz_I*rho_I)/rho;
    m_zz_t45 = (4*rho + 12*m_xx_I*rho_I - 36*m_xz_I*rho_I - 6*m_yy_I*rho_I + 147*m_zz_I*rho_I + 12*rho*ux0 - 12*rho*uz0)/(135*rho);

    rhoVar = rho;

    break;
case 0b0000010101011001:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (24*(15*rho_I + 4*m_xx_I*rho_I + 6*m_xy_I*rho_I - 6*m_xz_I*rho_I + m_yy_I*rho_I + m_zz_I*rho_I - 4*omegaVar*m_xx_I*rho_I - 6*omegaVar*m_xy_I*rho_I + 6*omegaVar*m_xz_I*rho_I - omegaVar*m_yy_I*rho_I - omegaVar*m_zz_I*rho_I))/(12*omegaVar + 24*ux0 + 12*uy0 - 12*uz0 + 36*omegaVar*ux0 + 18*omegaVar*uy0 - 18*omegaVar*uz0 - 60*omegaVar*ux0*ux0 - 15*omegaVar*uy0*uy0 - 15*omegaVar*uz0*uz0 - 90*omegaVar*ux0*uy0 + 90*omegaVar*ux0*uz0 + 328);

    m_xx_t45 = (8*rho + 159*m_xx_I*rho_I + 36*m_xy_I*rho_I - 36*m_xz_I*rho_I + 6*m_yy_I*rho_I + 6*m_zz_I*rho_I + 24*rho*ux0 + 12*rho*uy0 - 12*rho*uz0)/(135*rho);
    m_xy_t90 = (2*rho + 6*m_xx_I*rho_I + 63*m_xy_I*rho_I + 6*m_yy_I*rho_I - 3*m_zz_I*rho_I + 6*rho*ux0 + 6*rho*uy0)/(45*rho);
    m_xz_t90 = -(2*rho + 6*m_xx_I*rho_I - 63*m_xz_I*rho_I - 3*m_yy_I*rho_I + 6*m_zz_I*rho_I + 6*rho*ux0 - 6*rho*uz0)/(45*rho);
    m_yy_t45 = (2*(rho + 3*m_xx_I*rho_I + 18*m_xy_I*rho_I + 9*m_xz_I*rho_I + 75*m_yy_I*rho_I - 6*m_zz_I*rho_I + 3*rho*ux0 + 6*rho*uy0 + 3*rho*uz0))/(135*rho);
    m_yz_t90 = (m_yz_I*rho_I)/rho;
    m_zz_t45 = -(2*(9*m_xy_I*rho_I - 3*m_xx_I*rho_I - rho + 18*m_xz_I*rho_I + 6*m_yy_I*rho_I - 75*m_zz_I*rho_I - 3*rho*ux0 + 3*rho*uy0 + 6*rho*uz0))/(135*rho);

    rhoVar = rho;

    break;
case 0b0000010101011010:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (12*(3*rho_I + 2*m_xx_I*rho_I - 6*m_xz_I*rho_I - m_yy_I*rho_I + 2*m_zz_I*rho_I - 2*omegaVar*m_xx_I*rho_I + 6*omegaVar*m_xz_I*rho_I + omegaVar*m_yy_I*rho_I - 2*omegaVar*m_zz_I*rho_I))/(6*omegaVar - 6*omegaVar*ux0*ux0 + 3*omegaVar*uy0*uy0 - 6*omegaVar*uz0*uz0 + 18*omegaVar*ux0*uz0 + 28);

    m_xx_t45 = (4*rho + 39*m_xx_I*rho_I - 36*m_xz_I*rho_I - 6*m_yy_I*rho_I + 12*m_zz_I*rho_I)/(27*rho);
    m_xy_t90 = (m_xy_I*rho_I)/rho;
    m_xz_t90 = -(2*rho + 6*m_xx_I*rho_I - 27*m_xz_I*rho_I - 3*m_yy_I*rho_I + 6*m_zz_I*rho_I)/(9*rho);
    m_yy_t45 = -(2*(rho + 3*m_xx_I*rho_I - 9*m_xz_I*rho_I - 15*m_yy_I*rho_I + 3*m_zz_I*rho_I))/(27*rho);
    m_yz_t90 = (m_yz_I*rho_I)/rho;
    m_zz_t45 = (4*rho + 12*m_xx_I*rho_I - 36*m_xz_I*rho_I - 6*m_yy_I*rho_I + 39*m_zz_I*rho_I)/(27*rho);

    rhoVar = rho;

    break;
case 0b0000010101011011:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[8] + pop[9] + pop[10] + pop[12] + pop[13] + pop[14] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (24*(15*rho_I + 11*m_xx_I*rho_I + 6*m_xy_I*rho_I - 30*m_xz_I*rho_I - m_yy_I*rho_I + 6*m_yz_I*rho_I + 11*m_zz_I*rho_I - 11*omegaVar*m_xx_I*rho_I - 6*omegaVar*m_xy_I*rho_I + 30*omegaVar*m_xz_I*rho_I + omegaVar*m_yy_I*rho_I - 6*omegaVar*m_yz_I*rho_I - 11*omegaVar*m_zz_I*rho_I))/(72*omegaVar + 12*ux0 + 24*uy0 + 12*uz0 + 18*omegaVar*ux0 + 36*omegaVar*uy0 + 18*omegaVar*uz0 - 75*omegaVar*ux0*ux0 - 30*omegaVar*uy0*uy0 - 75*omegaVar*uz0*uz0 - 90*omegaVar*ux0*uy0 + 180*omegaVar*ux0*uz0 - 90*omegaVar*uy0*uz0 + 248);

    m_xx_t45 = (2*(11*rho + 105*m_xx_I*rho_I + 18*m_xy_I*rho_I - 90*m_xz_I*rho_I - 12*m_yy_I*rho_I - 9*m_yz_I*rho_I + 24*m_zz_I*rho_I + 6*rho*ux0 + 3*rho*uy0 - 3*rho*uz0))/(135*rho);
    m_xy_t90 = (2*rho + 6*m_xx_I*rho_I + 63*m_xy_I*rho_I + 6*m_yy_I*rho_I - 3*m_zz_I*rho_I + 6*rho*ux0 + 6*rho*uy0)/(45*rho);
    m_xz_t90 = -(2*rho + 6*m_xx_I*rho_I - 27*m_xz_I*rho_I - 3*m_yy_I*rho_I + 6*m_zz_I*rho_I)/(9*rho);
    m_yy_t45 = (2*(18*m_xy_I*rho_I - 12*m_xx_I*rho_I - rho + 45*m_xz_I*rho_I + 87*m_yy_I*rho_I + 18*m_yz_I*rho_I - 12*m_zz_I*rho_I + 6*rho*ux0 + 12*rho*uy0 + 6*rho*uz0))/(135*rho);
    m_yz_t90 = (2*rho - 3*m_xx_I*rho_I + 6*m_yy_I*rho_I + 63*m_yz_I*rho_I + 6*m_zz_I*rho_I + 6*rho*uy0 + 6*rho*uz0)/(45*rho);
    m_zz_t45 = (2*(11*rho + 24*m_xx_I*rho_I - 9*m_xy_I*rho_I - 90*m_xz_I*rho_I - 12*m_yy_I*rho_I + 18*m_yz_I*rho_I + 105*m_zz_I*rho_I - 3*rho*ux0 + 3*rho*uy0 + 6*rho*uz0))/(135*rho);

    rhoVar = rho;

    break;
case 0b0000010101011100:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[14] + pop[16] + pop[17];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17]);

    rho = (24*(5*rho_I + m_xx_I*rho_I - 2*m_xy_I*rho_I - 2*m_xz_I*rho_I + m_yy_I*rho_I - 2*m_yz_I*rho_I + m_zz_I*rho_I - omegaVar*m_xx_I*rho_I + 2*omegaVar*m_xy_I*rho_I + 2*omegaVar*m_xz_I*rho_I - omegaVar*m_yy_I*rho_I + 2*omegaVar*m_yz_I*rho_I - omegaVar*m_zz_I*rho_I))/(6*omegaVar + 8*ux0 - 8*uy0 + 12*omegaVar*ux0 - 12*omegaVar*uy0 - 15*omegaVar*ux0*ux0 - 15*omegaVar*uy0*uy0 - 15*omegaVar*uz0*uz0 + 30*omegaVar*ux0*uy0 + 30*omegaVar*ux0*uz0 + 30*omegaVar*uy0*uz0 + 104);

    m_xx_t45 = (2*(rho + 27*m_xx_I*rho_I - 6*m_xy_I*rho_I - 6*m_xz_I*rho_I + 3*m_yz_I*rho_I + 4*rho*ux0 - rho*uy0 - 3*rho*uz0))/(45*rho);
    m_xy_t90 = -(2*rho + 6*m_xx_I*rho_I - 63*m_xy_I*rho_I + 6*m_yy_I*rho_I - 3*m_zz_I*rho_I + 6*rho*ux0 - 6*rho*uy0)/(45*rho);
    m_xz_t90 = -(2*rho + 6*m_xx_I*rho_I - 63*m_xz_I*rho_I - 3*m_yy_I*rho_I + 6*m_zz_I*rho_I + 6*rho*ux0 - 6*rho*uz0)/(45*rho);
    m_yy_t45 = (2*(rho - 6*m_xy_I*rho_I + 3*m_xz_I*rho_I + 27*m_yy_I*rho_I - 6*m_yz_I*rho_I + rho*ux0 - 4*rho*uy0 + 3*rho*uz0))/(45*rho);
    m_yz_t90 = -(2*rho - 3*m_xx_I*rho_I + 6*m_yy_I*rho_I - 63*m_yz_I*rho_I + 6*m_zz_I*rho_I - 6*rho*uy0 + 6*rho*uz0)/(45*rho);
    m_zz_t45 = (2*(rho + 3*m_xy_I*rho_I - 6*m_xz_I*rho_I - 6*m_yz_I*rho_I + 27*m_zz_I*rho_I + rho*ux0 - rho*uy0))/(45*rho);

    rhoVar = rho;

    break;
case 0b0000010101011101:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[8] + pop[10] + pop[11] + pop[12] + pop[14] + pop[16] + pop[17];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17]);
    m_xy_I = inv_rho_I * (pop[8] - pop[14]);
    m_xz_I = inv_rho_I * (pop[10] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17]);

    rho = (72*(107*rho_I + 72*m_xx_I*rho_I + 12*m_yy_I*rho_I - 33*m_yz_I*rho_I + 12*m_zz_I*rho_I - 72*omegaVar*m_xx_I*rho_I - 12*omegaVar*m_yy_I*rho_I + 33*omegaVar*m_yz_I*rho_I - 12*omegaVar*m_zz_I*rho_I))/(626*omegaVar + 2124*ux0 - 492*uy0 + 492*uz0 + 1728*omegaVar*ux0 - 150*omegaVar*uy0 + 150*omegaVar*uz0 - 3531*omegaVar*ux0*ux0 - 642*omegaVar*uy0*uy0 - 642*omegaVar*uz0*uz0 + 1926*omegaVar*uy0*uz0 + 5580);

    m_xx_t45 = (2*(24*rho + 249*m_xx_I*rho_I - 12*m_yy_I*rho_I + 33*m_yz_I*rho_I - 12*m_zz_I*rho_I + 83*rho*ux0 + 11*rho*uy0 - 11*rho*uz0))/(321*rho);
    m_xy_t90 = (6*m_xy_I*rho_I + rho*uy0)/(3*rho);
    m_xz_t90 = (6*m_xz_I*rho_I + rho*uz0)/(3*rho);
    m_yy_t45 = -(4*(6*m_xx_I*rho_I - 2*rho - 106*m_yy_I*rho_I + 24*m_yz_I*rho_I + m_zz_I*rho_I + 2*rho*ux0 + 8*rho*uy0 - 8*rho*uz0))/(321*rho);
    m_yz_t90 = -(11*rho - 33*m_xx_I*rho_I + 48*m_yy_I*rho_I - 453*m_yz_I*rho_I + 48*m_zz_I*rho_I - 11*rho*ux0 - 44*rho*uy0 + 44*rho*uz0)/(321*rho);
    m_zz_t45 = -(4*(6*m_xx_I*rho_I - 2*rho + m_yy_I*rho_I + 24*m_yz_I*rho_I - 106*m_zz_I*rho_I + 2*rho*ux0 + 8*rho*uy0 - 8*rho*uz0))/(321*rho);

    rhoVar = rho;

    break;
case 0b0000010101011110:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[14] + pop[17];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[17]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[17]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[17]);

    rho = (24*(15*rho_I + 11*m_xx_I*rho_I - 6*m_xy_I*rho_I - 30*m_xz_I*rho_I - m_yy_I*rho_I - 6*m_yz_I*rho_I + 11*m_zz_I*rho_I - 11*omegaVar*m_xx_I*rho_I + 6*omegaVar*m_xy_I*rho_I + 30*omegaVar*m_xz_I*rho_I + omegaVar*m_yy_I*rho_I + 6*omegaVar*m_yz_I*rho_I - 11*omegaVar*m_zz_I*rho_I))/(72*omegaVar + 12*ux0 - 24*uy0 + 12*uz0 + 18*omegaVar*ux0 - 36*omegaVar*uy0 + 18*omegaVar*uz0 - 75*omegaVar*ux0*ux0 - 30*omegaVar*uy0*uy0 - 75*omegaVar*uz0*uz0 + 90*omegaVar*ux0*uy0 + 180*omegaVar*ux0*uz0 + 90*omegaVar*uy0*uz0 + 248);

    m_xx_t45 = (2*(11*rho + 105*m_xx_I*rho_I - 18*m_xy_I*rho_I - 90*m_xz_I*rho_I - 12*m_yy_I*rho_I + 9*m_yz_I*rho_I + 24*m_zz_I*rho_I + 6*rho*ux0 - 3*rho*uy0 - 3*rho*uz0))/(135*rho);
    m_xy_t90 = -(2*rho + 6*m_xx_I*rho_I - 63*m_xy_I*rho_I + 6*m_yy_I*rho_I - 3*m_zz_I*rho_I + 6*rho*ux0 - 6*rho*uy0)/(45*rho);
    m_xz_t90 = -(2*rho + 6*m_xx_I*rho_I - 27*m_xz_I*rho_I - 3*m_yy_I*rho_I + 6*m_zz_I*rho_I)/(9*rho);
    m_yy_t45 = -(2*(rho + 12*m_xx_I*rho_I + 18*m_xy_I*rho_I - 45*m_xz_I*rho_I - 87*m_yy_I*rho_I + 18*m_yz_I*rho_I + 12*m_zz_I*rho_I - 6*rho*ux0 + 12*rho*uy0 - 6*rho*uz0))/(135*rho);
    m_yz_t90 = -(2*rho - 3*m_xx_I*rho_I + 6*m_yy_I*rho_I - 63*m_yz_I*rho_I + 6*m_zz_I*rho_I - 6*rho*uy0 + 6*rho*uz0)/(45*rho);
    m_zz_t45 = (2*(11*rho + 24*m_xx_I*rho_I + 9*m_xy_I*rho_I - 90*m_xz_I*rho_I - 12*m_yy_I*rho_I - 18*m_yz_I*rho_I + 105*m_zz_I*rho_I - 3*rho*ux0 - 3*rho*uy0 + 6*rho*uz0))/(135*rho);

    rhoVar = rho;

    break;
case 0b0000010101011111:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[2] + pop[3] + pop[4] + pop[6] + pop[8] + pop[10] + pop[12] + pop[14] + pop[17];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[17]);
    m_xy_I = inv_rho_I * (pop[8] - pop[14]);
    m_xz_I = inv_rho_I * (pop[10]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[17]);
    m_yz_I = inv_rho_I * (pop[12] - pop[17]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[17]);

    rho = (72*(23*rho_I + 24*m_xx_I*rho_I - 57*m_xz_I*rho_I - 6*m_yy_I*rho_I + 24*m_zz_I*rho_I - 24*omegaVar*m_xx_I*rho_I + 57*omegaVar*m_xz_I*rho_I + 6*omegaVar*m_yy_I*rho_I - 24*omegaVar*m_zz_I*rho_I))/(430*omegaVar + 660*ux0 + 660*uz0 + 30*omegaVar*ux0 + 30*omegaVar*uz0 - 690*omegaVar*ux0*ux0 - 69*omegaVar*uy0*uy0 - 690*omegaVar*uz0*uz0 + 414*omegaVar*ux0*uz0 + 720);

    m_xx_t45 = (16*rho + 141*m_xx_I*rho_I - 102*m_xz_I*rho_I - 18*m_yy_I*rho_I + 3*m_zz_I*rho_I + 30*rho*ux0 - 16*rho*uz0)/(69*rho);
    m_xy_t90 = (6*m_xy_I*rho_I + rho*uy0)/(3*rho);
    m_xz_t90 = (354*m_xz_I*rho_I - 51*m_xx_I*rho_I - 19*rho + 30*m_yy_I*rho_I - 51*m_zz_I*rho_I + 19*rho*ux0 + 19*rho*uz0)/(69*rho);
    m_yy_t45 = (2*(30*m_xz_I*rho_I - 9*m_xx_I*rho_I - 2*rho + 54*m_yy_I*rho_I - 9*m_zz_I*rho_I + 2*rho*ux0 + 2*rho*uz0))/(69*rho);
    m_yz_t90 = (6*m_yz_I*rho_I + rho*uy0)/(3*rho);
    m_zz_t45 = (16*rho + 3*m_xx_I*rho_I - 102*m_xz_I*rho_I - 18*m_yy_I*rho_I + 141*m_zz_I*rho_I - 16*rho*ux0 + 30*rho*uz0)/(69*rho);

    rhoVar = rho;

    break;
case 0b0000010101100000:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = rho_I;

    m_xx_t45 = (m_xx_I*rho_I)/rho;
    m_xy_t90 = (m_xy_I*rho_I)/rho;
    m_xz_t90 = (m_xz_I*rho_I)/rho;
    m_yy_t45 = (m_yy_I*rho_I)/rho;
    m_yz_t90 = (m_yz_I*rho_I)/rho;
    m_zz_t45 = (m_zz_I*rho_I)/rho;

    rhoVar = rho;

    break;
case 0b0000010101100001:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = rho_I;

    m_xx_t45 = (m_xx_I*rho_I)/rho;
    m_xy_t90 = (m_xy_I*rho_I)/rho;
    m_xz_t90 = (m_xz_I*rho_I)/rho;
    m_yy_t45 = (m_yy_I*rho_I)/rho;
    m_yz_t90 = (m_yz_I*rho_I)/rho;
    m_zz_t45 = (m_zz_I*rho_I)/rho;

    rhoVar = rho;

    break;
case 0b0000010101100010:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (24*(15*rho_I + 2*m_xx_I*rho_I - 6*m_xy_I*rho_I + 2*m_yy_I*rho_I - m_zz_I*rho_I - 2*omegaVar*m_xx_I*rho_I + 6*omegaVar*m_xy_I*rho_I - 2*omegaVar*m_yy_I*rho_I + omegaVar*m_zz_I*rho_I))/(6*omegaVar - 12*ux0 + 12*uy0 - 18*omegaVar*ux0 + 18*omegaVar*uy0 - 30*omegaVar*ux0*ux0 - 30*omegaVar*uy0*uy0 + 15*omegaVar*uz0*uz0 + 90*omegaVar*ux0*uy0 + 344);

    m_xx_t45 = (4*rho + 147*m_xx_I*rho_I - 36*m_xy_I*rho_I + 12*m_yy_I*rho_I - 6*m_zz_I*rho_I - 12*rho*ux0 + 12*rho*uy0)/(135*rho);
    m_xy_t90 = -(2*rho + 6*m_xx_I*rho_I - 63*m_xy_I*rho_I + 6*m_yy_I*rho_I - 3*m_zz_I*rho_I - 6*rho*ux0 + 6*rho*uy0)/(45*rho);
    m_xz_t90 = (m_xz_I*rho_I)/rho;
    m_yy_t45 = (4*rho + 12*m_xx_I*rho_I - 36*m_xy_I*rho_I + 147*m_yy_I*rho_I - 6*m_zz_I*rho_I - 12*rho*ux0 + 12*rho*uy0)/(135*rho);
    m_yz_t90 = (m_yz_I*rho_I)/rho;
    m_zz_t45 = -(2*(rho + 3*m_xx_I*rho_I - 9*m_xy_I*rho_I + 3*m_yy_I*rho_I - 69*m_zz_I*rho_I - 3*rho*ux0 + 3*rho*uy0))/(135*rho);

    rhoVar = rho;

    break;
case 0b0000010101100011:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[12] + pop[13] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (24*(15*rho_I + m_xx_I*rho_I - 6*m_xy_I*rho_I + 4*m_yy_I*rho_I + 6*m_yz_I*rho_I + m_zz_I*rho_I - omegaVar*m_xx_I*rho_I + 6*omegaVar*m_xy_I*rho_I - 4*omegaVar*m_yy_I*rho_I - 6*omegaVar*m_yz_I*rho_I - omegaVar*m_zz_I*rho_I))/(12*omegaVar - 12*ux0 + 24*uy0 + 12*uz0 - 18*omegaVar*ux0 + 36*omegaVar*uy0 + 18*omegaVar*uz0 - 15*omegaVar*ux0*ux0 - 60*omegaVar*uy0*uy0 - 15*omegaVar*uz0*uz0 + 90*omegaVar*ux0*uy0 - 90*omegaVar*uy0*uz0 + 328);

    m_xx_t45 = -(2*(18*m_xy_I*rho_I - 75*m_xx_I*rho_I - rho - 3*m_yy_I*rho_I + 9*m_yz_I*rho_I + 6*m_zz_I*rho_I + 6*rho*ux0 - 3*rho*uy0 + 3*rho*uz0))/(135*rho);
    m_xy_t90 = -(2*rho + 6*m_xx_I*rho_I - 63*m_xy_I*rho_I + 6*m_yy_I*rho_I - 3*m_zz_I*rho_I - 6*rho*ux0 + 6*rho*uy0)/(45*rho);
    m_xz_t90 = (m_xz_I*rho_I)/rho;
    m_yy_t45 = (8*rho + 6*m_xx_I*rho_I - 36*m_xy_I*rho_I + 159*m_yy_I*rho_I + 36*m_yz_I*rho_I + 6*m_zz_I*rho_I - 12*rho*ux0 + 24*rho*uy0 + 12*rho*uz0)/(135*rho);
    m_yz_t90 = (2*rho - 3*m_xx_I*rho_I + 6*m_yy_I*rho_I + 63*m_yz_I*rho_I + 6*m_zz_I*rho_I + 6*rho*uy0 + 6*rho*uz0)/(45*rho);
    m_zz_t45 = (2*(rho - 6*m_xx_I*rho_I + 9*m_xy_I*rho_I + 3*m_yy_I*rho_I + 18*m_yz_I*rho_I + 75*m_zz_I*rho_I + 3*rho*ux0 + 3*rho*uy0 + 6*rho*uz0))/(135*rho);

    rhoVar = rho;

    break;
case 0b0000010101100100:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (24*(15*rho_I + 2*m_xx_I*rho_I - 6*m_xy_I*rho_I + 2*m_yy_I*rho_I - m_zz_I*rho_I - 2*omegaVar*m_xx_I*rho_I + 6*omegaVar*m_xy_I*rho_I - 2*omegaVar*m_yy_I*rho_I + omegaVar*m_zz_I*rho_I))/(6*omegaVar + 12*ux0 - 12*uy0 + 18*omegaVar*ux0 - 18*omegaVar*uy0 - 30*omegaVar*ux0*ux0 - 30*omegaVar*uy0*uy0 + 15*omegaVar*uz0*uz0 + 90*omegaVar*ux0*uy0 + 344);

    m_xx_t45 = (4*rho + 147*m_xx_I*rho_I - 36*m_xy_I*rho_I + 12*m_yy_I*rho_I - 6*m_zz_I*rho_I + 12*rho*ux0 - 12*rho*uy0)/(135*rho);
    m_xy_t90 = -(2*rho + 6*m_xx_I*rho_I - 63*m_xy_I*rho_I + 6*m_yy_I*rho_I - 3*m_zz_I*rho_I + 6*rho*ux0 - 6*rho*uy0)/(45*rho);
    m_xz_t90 = (m_xz_I*rho_I)/rho;
    m_yy_t45 = (4*rho + 12*m_xx_I*rho_I - 36*m_xy_I*rho_I + 147*m_yy_I*rho_I - 6*m_zz_I*rho_I + 12*rho*ux0 - 12*rho*uy0)/(135*rho);
    m_yz_t90 = (m_yz_I*rho_I)/rho;
    m_zz_t45 = -(2*(rho + 3*m_xx_I*rho_I - 9*m_xy_I*rho_I + 3*m_yy_I*rho_I - 69*m_zz_I*rho_I + 3*rho*ux0 - 3*rho*uy0))/(135*rho);

    rhoVar = rho;

    break;
case 0b0000010101100101:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[10] + pop[11] + pop[12] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[14]);
    m_xz_I = inv_rho_I * (pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (24*(15*rho_I + 4*m_xx_I*rho_I - 6*m_xy_I*rho_I + 6*m_xz_I*rho_I + m_yy_I*rho_I + m_zz_I*rho_I - 4*omegaVar*m_xx_I*rho_I + 6*omegaVar*m_xy_I*rho_I - 6*omegaVar*m_xz_I*rho_I - omegaVar*m_yy_I*rho_I - omegaVar*m_zz_I*rho_I))/(12*omegaVar + 24*ux0 - 12*uy0 + 12*uz0 + 36*omegaVar*ux0 - 18*omegaVar*uy0 + 18*omegaVar*uz0 - 60*omegaVar*ux0*ux0 - 15*omegaVar*uy0*uy0 - 15*omegaVar*uz0*uz0 + 90*omegaVar*ux0*uy0 - 90*omegaVar*ux0*uz0 + 328);

    m_xx_t45 = (8*rho + 159*m_xx_I*rho_I - 36*m_xy_I*rho_I + 36*m_xz_I*rho_I + 6*m_yy_I*rho_I + 6*m_zz_I*rho_I + 24*rho*ux0 - 12*rho*uy0 + 12*rho*uz0)/(135*rho);
    m_xy_t90 = -(2*rho + 6*m_xx_I*rho_I - 63*m_xy_I*rho_I + 6*m_yy_I*rho_I - 3*m_zz_I*rho_I + 6*rho*ux0 - 6*rho*uy0)/(45*rho);
    m_xz_t90 = (2*rho + 6*m_xx_I*rho_I + 63*m_xz_I*rho_I - 3*m_yy_I*rho_I + 6*m_zz_I*rho_I + 6*rho*ux0 + 6*rho*uz0)/(45*rho);
    m_yy_t45 = -(2*(18*m_xy_I*rho_I - 3*m_xx_I*rho_I - rho + 9*m_xz_I*rho_I - 75*m_yy_I*rho_I + 6*m_zz_I*rho_I - 3*rho*ux0 + 6*rho*uy0 + 3*rho*uz0))/(135*rho);
    m_yz_t90 = (m_yz_I*rho_I)/rho;
    m_zz_t45 = (2*(rho + 3*m_xx_I*rho_I + 9*m_xy_I*rho_I + 18*m_xz_I*rho_I - 6*m_yy_I*rho_I + 75*m_zz_I*rho_I + 3*rho*ux0 + 3*rho*uy0 + 6*rho*uz0))/(135*rho);

    rhoVar = rho;

    break;
case 0b0000010101100110:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (12*(3*rho_I + 2*m_xx_I*rho_I - 6*m_xy_I*rho_I + 2*m_yy_I*rho_I - m_zz_I*rho_I - 2*omegaVar*m_xx_I*rho_I + 6*omegaVar*m_xy_I*rho_I - 2*omegaVar*m_yy_I*rho_I + omegaVar*m_zz_I*rho_I))/(6*omegaVar - 6*omegaVar*ux0*ux0 - 6*omegaVar*uy0*uy0 + 3*omegaVar*uz0*uz0 + 18*omegaVar*ux0*uy0 + 28);

    m_xx_t45 = (4*rho + 39*m_xx_I*rho_I - 36*m_xy_I*rho_I + 12*m_yy_I*rho_I - 6*m_zz_I*rho_I)/(27*rho);
    m_xy_t90 = -(2*rho + 6*m_xx_I*rho_I - 27*m_xy_I*rho_I + 6*m_yy_I*rho_I - 3*m_zz_I*rho_I)/(9*rho);
    m_xz_t90 = (m_xz_I*rho_I)/rho;
    m_yy_t45 = (4*rho + 12*m_xx_I*rho_I - 36*m_xy_I*rho_I + 39*m_yy_I*rho_I - 6*m_zz_I*rho_I)/(27*rho);
    m_yz_t90 = (m_yz_I*rho_I)/rho;
    m_zz_t45 = -(2*(rho + 3*m_xx_I*rho_I - 9*m_xy_I*rho_I + 3*m_yy_I*rho_I - 15*m_zz_I*rho_I))/(27*rho);

    rhoVar = rho;

    break;
case 0b0000010101100111:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[10] + pop[12] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8]);
    m_xz_I = inv_rho_I * (pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (24*(15*rho_I + 11*m_xx_I*rho_I - 30*m_xy_I*rho_I + 6*m_xz_I*rho_I + 11*m_yy_I*rho_I + 6*m_yz_I*rho_I - m_zz_I*rho_I - 11*omegaVar*m_xx_I*rho_I + 30*omegaVar*m_xy_I*rho_I - 6*omegaVar*m_xz_I*rho_I - 11*omegaVar*m_yy_I*rho_I - 6*omegaVar*m_yz_I*rho_I + omegaVar*m_zz_I*rho_I))/(72*omegaVar + 12*ux0 + 12*uy0 + 24*uz0 + 18*omegaVar*ux0 + 18*omegaVar*uy0 + 36*omegaVar*uz0 - 75*omegaVar*ux0*ux0 - 75*omegaVar*uy0*uy0 - 30*omegaVar*uz0*uz0 + 180*omegaVar*ux0*uy0 - 90*omegaVar*ux0*uz0 - 90*omegaVar*uy0*uz0 + 248);

    m_xx_t45 = (2*(11*rho + 105*m_xx_I*rho_I - 90*m_xy_I*rho_I + 18*m_xz_I*rho_I + 24*m_yy_I*rho_I - 9*m_yz_I*rho_I - 12*m_zz_I*rho_I + 6*rho*ux0 - 3*rho*uy0 + 3*rho*uz0))/(135*rho);
    m_xy_t90 = -(2*rho + 6*m_xx_I*rho_I - 27*m_xy_I*rho_I + 6*m_yy_I*rho_I - 3*m_zz_I*rho_I)/(9*rho);
    m_xz_t90 = (2*rho + 6*m_xx_I*rho_I + 63*m_xz_I*rho_I - 3*m_yy_I*rho_I + 6*m_zz_I*rho_I + 6*rho*ux0 + 6*rho*uz0)/(45*rho);
    m_yy_t45 = (2*(11*rho + 24*m_xx_I*rho_I - 90*m_xy_I*rho_I - 9*m_xz_I*rho_I + 105*m_yy_I*rho_I + 18*m_yz_I*rho_I - 12*m_zz_I*rho_I - 3*rho*ux0 + 6*rho*uy0 + 3*rho*uz0))/(135*rho);
    m_yz_t90 = (2*rho - 3*m_xx_I*rho_I + 6*m_yy_I*rho_I + 63*m_yz_I*rho_I + 6*m_zz_I*rho_I + 6*rho*uy0 + 6*rho*uz0)/(45*rho);
    m_zz_t45 = (2*(45*m_xy_I*rho_I - 12*m_xx_I*rho_I - rho + 18*m_xz_I*rho_I - 12*m_yy_I*rho_I + 18*m_yz_I*rho_I + 87*m_zz_I*rho_I + 6*rho*ux0 + 6*rho*uy0 + 12*rho*uz0))/(135*rho);

    rhoVar = rho;

    break;
case 0b0000010101101000:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = rho_I;

    m_xx_t45 = (m_xx_I*rho_I)/rho;
    m_xy_t90 = (m_xy_I*rho_I)/rho;
    m_xz_t90 = (m_xz_I*rho_I)/rho;
    m_yy_t45 = (m_yy_I*rho_I)/rho;
    m_yz_t90 = (m_yz_I*rho_I)/rho;
    m_zz_t45 = (m_zz_I*rho_I)/rho;

    rhoVar = rho;

    break;
case 0b0000010101101001:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = rho_I;

    m_xx_t45 = (m_xx_I*rho_I)/rho;
    m_xy_t90 = (m_xy_I*rho_I)/rho;
    m_xz_t90 = (m_xz_I*rho_I)/rho;
    m_yy_t45 = (m_yy_I*rho_I)/rho;
    m_yz_t90 = (m_yz_I*rho_I)/rho;
    m_zz_t45 = (m_zz_I*rho_I)/rho;

    rhoVar = rho;

    break;
case 0b0000010101101010:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[15] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[15] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] - (1.0/3.0)*pop[15] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (24*(15*rho_I + 4*m_xx_I*rho_I - 6*m_xy_I*rho_I - 6*m_xz_I*rho_I + m_yy_I*rho_I + m_zz_I*rho_I - 4*omegaVar*m_xx_I*rho_I + 6*omegaVar*m_xy_I*rho_I + 6*omegaVar*m_xz_I*rho_I - omegaVar*m_yy_I*rho_I - omegaVar*m_zz_I*rho_I))/(12*omegaVar - 24*ux0 + 12*uy0 + 12*uz0 - 36*omegaVar*ux0 + 18*omegaVar*uy0 + 18*omegaVar*uz0 - 60*omegaVar*ux0*ux0 - 15*omegaVar*uy0*uy0 - 15*omegaVar*uz0*uz0 + 90*omegaVar*ux0*uy0 + 90*omegaVar*ux0*uz0 + 328);

    m_xx_t45 = (8*rho + 159*m_xx_I*rho_I - 36*m_xy_I*rho_I - 36*m_xz_I*rho_I + 6*m_yy_I*rho_I + 6*m_zz_I*rho_I - 24*rho*ux0 + 12*rho*uy0 + 12*rho*uz0)/(135*rho);
    m_xy_t90 = -(2*rho + 6*m_xx_I*rho_I - 63*m_xy_I*rho_I + 6*m_yy_I*rho_I - 3*m_zz_I*rho_I - 6*rho*ux0 + 6*rho*uy0)/(45*rho);
    m_xz_t90 = -(2*rho + 6*m_xx_I*rho_I - 63*m_xz_I*rho_I - 3*m_yy_I*rho_I + 6*m_zz_I*rho_I - 6*rho*ux0 + 6*rho*uz0)/(45*rho);
    m_yy_t45 = (2*(rho + 3*m_xx_I*rho_I - 18*m_xy_I*rho_I + 9*m_xz_I*rho_I + 75*m_yy_I*rho_I - 6*m_zz_I*rho_I - 3*rho*ux0 + 6*rho*uy0 - 3*rho*uz0))/(135*rho);
    m_yz_t90 = (m_yz_I*rho_I)/rho;
    m_zz_t45 = (2*(rho + 3*m_xx_I*rho_I + 9*m_xy_I*rho_I - 18*m_xz_I*rho_I - 6*m_yy_I*rho_I + 75*m_zz_I*rho_I - 3*rho*ux0 - 3*rho*uy0 + 6*rho*uz0))/(135*rho);

    rhoVar = rho;

    break;
case 0b0000010101101011:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[12] + pop[13] + pop[15] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[15] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] - (1.0/3.0)*pop[15] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (24*(5*rho_I + m_xx_I*rho_I - 2*m_xy_I*rho_I - 2*m_xz_I*rho_I + m_yy_I*rho_I + 2*m_yz_I*rho_I + m_zz_I*rho_I - omegaVar*m_xx_I*rho_I + 2*omegaVar*m_xy_I*rho_I + 2*omegaVar*m_xz_I*rho_I - omegaVar*m_yy_I*rho_I - 2*omegaVar*m_yz_I*rho_I - omegaVar*m_zz_I*rho_I))/(6*omegaVar - 8*ux0 + 8*uy0 + 8*uz0 - 12*omegaVar*ux0 + 12*omegaVar*uy0 + 12*omegaVar*uz0 - 15*omegaVar*ux0*ux0 - 15*omegaVar*uy0*uy0 - 15*omegaVar*uz0*uz0 + 30*omegaVar*ux0*uy0 + 30*omegaVar*ux0*uz0 - 30*omegaVar*uy0*uz0 + 104);

    m_xx_t45 = (2*(rho + 27*m_xx_I*rho_I - 6*m_xy_I*rho_I - 6*m_xz_I*rho_I - 3*m_yz_I*rho_I - 4*rho*ux0 + rho*uy0 + rho*uz0))/(45*rho);
    m_xy_t90 = -(2*rho + 6*m_xx_I*rho_I - 63*m_xy_I*rho_I + 6*m_yy_I*rho_I - 3*m_zz_I*rho_I - 6*rho*ux0 + 6*rho*uy0)/(45*rho);
    m_xz_t90 = -(2*rho + 6*m_xx_I*rho_I - 63*m_xz_I*rho_I - 3*m_yy_I*rho_I + 6*m_zz_I*rho_I - 6*rho*ux0 + 6*rho*uz0)/(45*rho);
    m_yy_t45 = (2*(rho - 6*m_xy_I*rho_I + 3*m_xz_I*rho_I + 27*m_yy_I*rho_I + 6*m_yz_I*rho_I - rho*ux0 + 4*rho*uy0 + rho*uz0))/(45*rho);
    m_yz_t90 = (2*rho - 3*m_xx_I*rho_I + 6*m_yy_I*rho_I + 63*m_yz_I*rho_I + 6*m_zz_I*rho_I + 6*rho*uy0 + 6*rho*uz0)/(45*rho);
    m_zz_t45 = (2*(rho + 3*m_xy_I*rho_I - 6*m_xz_I*rho_I + 6*m_yz_I*rho_I + 27*m_zz_I*rho_I - rho*ux0 + rho*uy0 + 4*rho*uz0))/(45*rho);

    rhoVar = rho;

    break;
case 0b0000010101101100:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[14] + pop[15] + pop[16] + pop[17];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17]);

    rho = (24*(15*rho_I + m_xx_I*rho_I - 6*m_xy_I*rho_I + 4*m_yy_I*rho_I - 6*m_yz_I*rho_I + m_zz_I*rho_I - omegaVar*m_xx_I*rho_I + 6*omegaVar*m_xy_I*rho_I - 4*omegaVar*m_yy_I*rho_I + 6*omegaVar*m_yz_I*rho_I - omegaVar*m_zz_I*rho_I))/(12*omegaVar + 12*ux0 - 24*uy0 + 12*uz0 + 18*omegaVar*ux0 - 36*omegaVar*uy0 + 18*omegaVar*uz0 - 15*omegaVar*ux0*ux0 - 60*omegaVar*uy0*uy0 - 15*omegaVar*uz0*uz0 + 90*omegaVar*ux0*uy0 + 90*omegaVar*uy0*uz0 + 328);

    m_xx_t45 = (2*(rho + 75*m_xx_I*rho_I - 18*m_xy_I*rho_I + 3*m_yy_I*rho_I + 9*m_yz_I*rho_I - 6*m_zz_I*rho_I + 6*rho*ux0 - 3*rho*uy0 - 3*rho*uz0))/(135*rho);
    m_xy_t90 = -(2*rho + 6*m_xx_I*rho_I - 63*m_xy_I*rho_I + 6*m_yy_I*rho_I - 3*m_zz_I*rho_I + 6*rho*ux0 - 6*rho*uy0)/(45*rho);
    m_xz_t90 = (m_xz_I*rho_I)/rho;
    m_yy_t45 = (8*rho + 6*m_xx_I*rho_I - 36*m_xy_I*rho_I + 159*m_yy_I*rho_I - 36*m_yz_I*rho_I + 6*m_zz_I*rho_I + 12*rho*ux0 - 24*rho*uy0 + 12*rho*uz0)/(135*rho);
    m_yz_t90 = -(2*rho - 3*m_xx_I*rho_I + 6*m_yy_I*rho_I - 63*m_yz_I*rho_I + 6*m_zz_I*rho_I - 6*rho*uy0 + 6*rho*uz0)/(45*rho);
    m_zz_t45 = (2*(rho - 6*m_xx_I*rho_I + 9*m_xy_I*rho_I + 3*m_yy_I*rho_I - 18*m_yz_I*rho_I + 75*m_zz_I*rho_I - 3*rho*ux0 - 3*rho*uy0 + 6*rho*uz0))/(135*rho);

    rhoVar = rho;

    break;
case 0b0000010101101101:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[10] + pop[11] + pop[12] + pop[14] + pop[15] + pop[16] + pop[17];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[14]);
    m_xz_I = inv_rho_I * (pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17]);

    rho = (24*(5*rho_I + m_xx_I*rho_I - 2*m_xy_I*rho_I + 2*m_xz_I*rho_I + m_yy_I*rho_I - 2*m_yz_I*rho_I + m_zz_I*rho_I - omegaVar*m_xx_I*rho_I + 2*omegaVar*m_xy_I*rho_I - 2*omegaVar*m_xz_I*rho_I - omegaVar*m_yy_I*rho_I + 2*omegaVar*m_yz_I*rho_I - omegaVar*m_zz_I*rho_I))/(6*omegaVar + 8*ux0 - 8*uy0 + 8*uz0 + 12*omegaVar*ux0 - 12*omegaVar*uy0 + 12*omegaVar*uz0 - 15*omegaVar*ux0*ux0 - 15*omegaVar*uy0*uy0 - 15*omegaVar*uz0*uz0 + 30*omegaVar*ux0*uy0 - 30*omegaVar*ux0*uz0 + 30*omegaVar*uy0*uz0 + 104);

    m_xx_t45 = (2*(rho + 27*m_xx_I*rho_I - 6*m_xy_I*rho_I + 6*m_xz_I*rho_I + 3*m_yz_I*rho_I + 4*rho*ux0 - rho*uy0 + rho*uz0))/(45*rho);
    m_xy_t90 = -(2*rho + 6*m_xx_I*rho_I - 63*m_xy_I*rho_I + 6*m_yy_I*rho_I - 3*m_zz_I*rho_I + 6*rho*ux0 - 6*rho*uy0)/(45*rho);
    m_xz_t90 = (2*rho + 6*m_xx_I*rho_I + 63*m_xz_I*rho_I - 3*m_yy_I*rho_I + 6*m_zz_I*rho_I + 6*rho*ux0 + 6*rho*uz0)/(45*rho);
    m_yy_t45 = (2*(rho - 6*m_xy_I*rho_I - 3*m_xz_I*rho_I + 27*m_yy_I*rho_I - 6*m_yz_I*rho_I + rho*ux0 - 4*rho*uy0 + rho*uz0))/(45*rho);
    m_yz_t90 = -(2*rho - 3*m_xx_I*rho_I + 6*m_yy_I*rho_I - 63*m_yz_I*rho_I + 6*m_zz_I*rho_I - 6*rho*uy0 + 6*rho*uz0)/(45*rho);
    m_zz_t45 = (2*(rho + 3*m_xy_I*rho_I + 6*m_xz_I*rho_I - 6*m_yz_I*rho_I + 27*m_zz_I*rho_I + rho*ux0 - rho*uy0 + 4*rho*uz0))/(45*rho);

    rhoVar = rho;

    break;
case 0b0000010101101110:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[15] + pop[17];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[15] - (1.0/3.0)*pop[17]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[15] + (2.0/3.0)*pop[17]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[17]);

    rho = (24*(15*rho_I + 11*m_xx_I*rho_I - 30*m_xy_I*rho_I - 6*m_xz_I*rho_I + 11*m_yy_I*rho_I - 6*m_yz_I*rho_I - m_zz_I*rho_I - 11*omegaVar*m_xx_I*rho_I + 30*omegaVar*m_xy_I*rho_I + 6*omegaVar*m_xz_I*rho_I - 11*omegaVar*m_yy_I*rho_I + 6*omegaVar*m_yz_I*rho_I + omegaVar*m_zz_I*rho_I))/(72*omegaVar - 12*ux0 - 12*uy0 + 24*uz0 - 18*omegaVar*ux0 - 18*omegaVar*uy0 + 36*omegaVar*uz0 - 75*omegaVar*ux0*ux0 - 75*omegaVar*uy0*uy0 - 30*omegaVar*uz0*uz0 + 180*omegaVar*ux0*uy0 + 90*omegaVar*ux0*uz0 + 90*omegaVar*uy0*uz0 + 248);

    m_xx_t45 = (2*(11*rho + 105*m_xx_I*rho_I - 90*m_xy_I*rho_I - 18*m_xz_I*rho_I + 24*m_yy_I*rho_I + 9*m_yz_I*rho_I - 12*m_zz_I*rho_I - 6*rho*ux0 + 3*rho*uy0 + 3*rho*uz0))/(135*rho);
    m_xy_t90 = -(2*rho + 6*m_xx_I*rho_I - 27*m_xy_I*rho_I + 6*m_yy_I*rho_I - 3*m_zz_I*rho_I)/(9*rho);
    m_xz_t90 = -(2*rho + 6*m_xx_I*rho_I - 63*m_xz_I*rho_I - 3*m_yy_I*rho_I + 6*m_zz_I*rho_I - 6*rho*ux0 + 6*rho*uz0)/(45*rho);
    m_yy_t45 = (2*(11*rho + 24*m_xx_I*rho_I - 90*m_xy_I*rho_I + 9*m_xz_I*rho_I + 105*m_yy_I*rho_I - 18*m_yz_I*rho_I - 12*m_zz_I*rho_I + 3*rho*ux0 - 6*rho*uy0 + 3*rho*uz0))/(135*rho);
    m_yz_t90 = -(2*rho - 3*m_xx_I*rho_I + 6*m_yy_I*rho_I - 63*m_yz_I*rho_I + 6*m_zz_I*rho_I - 6*rho*uy0 + 6*rho*uz0)/(45*rho);
    m_zz_t45 = -(2*(rho + 12*m_xx_I*rho_I - 45*m_xy_I*rho_I + 18*m_xz_I*rho_I + 12*m_yy_I*rho_I + 18*m_yz_I*rho_I - 87*m_zz_I*rho_I + 6*rho*ux0 + 6*rho*uy0 - 12*rho*uz0))/(135*rho);

    rhoVar = rho;

    break;
case 0b0000010101101111:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[6] + pop[7] + pop[8] + pop[10] + pop[12] + pop[15] + pop[17];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[15] - (1.0/3.0)*pop[17]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8]);
    m_xz_I = inv_rho_I * (pop[10] - pop[15]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[15] + (2.0/3.0)*pop[17]);
    m_yz_I = inv_rho_I * (pop[12] - pop[17]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[17]);

    rho = (36*(19*rho_I + 12*m_xx_I*rho_I - 33*m_xy_I*rho_I + 12*m_yy_I*rho_I + 6*m_zz_I*rho_I - 12*omegaVar*m_xx_I*rho_I + 33*omegaVar*m_xy_I*rho_I - 12*omegaVar*m_yy_I*rho_I - 6*omegaVar*m_zz_I*rho_I))/(118*omegaVar + 270*uz0 + 72*omegaVar*uz0 - 114*omegaVar*ux0*ux0 - 114*omegaVar*uy0*uy0 - 285*omegaVar*uz0*uz0 + 342*omegaVar*ux0*uy0 + 414);

    m_xx_t45 = (4*(2*rho + 26*m_xx_I*rho_I - 24*m_xy_I*rho_I + 7*m_yy_I*rho_I - 6*m_zz_I*rho_I - 2*rho*uz0))/(57*rho);
    m_xy_t90 = -(11*rho + 48*m_xx_I*rho_I - 189*m_xy_I*rho_I + 48*m_yy_I*rho_I - 33*m_zz_I*rho_I - 11*rho*uz0)/(57*rho);
    m_xz_t90 = (6*m_xz_I*rho_I + rho*ux0)/(3*rho);
    m_yy_t45 = (4*(2*rho + 7*m_xx_I*rho_I - 24*m_xy_I*rho_I + 26*m_yy_I*rho_I - 6*m_zz_I*rho_I - 2*rho*uz0))/(57*rho);
    m_yz_t90 = (6*m_yz_I*rho_I + rho*uy0)/(3*rho);
    m_zz_t45 = (2*(2*rho - 12*m_xx_I*rho_I + 33*m_xy_I*rho_I - 12*m_yy_I*rho_I + 51*m_zz_I*rho_I + 17*rho*uz0))/(57*rho);

    rhoVar = rho;

    break;
case 0b0000010101110000:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[16] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);

    rho = (24*(15*rho_I + m_xx_I*rho_I - 6*m_xz_I*rho_I + m_yy_I*rho_I - 6*m_yz_I*rho_I + 4*m_zz_I*rho_I - omegaVar*m_xx_I*rho_I + 6*omegaVar*m_xz_I*rho_I - omegaVar*m_yy_I*rho_I + 6*omegaVar*m_yz_I*rho_I - 4*omegaVar*m_zz_I*rho_I))/(12*omegaVar + 12*ux0 + 12*uy0 - 24*uz0 + 18*omegaVar*ux0 + 18*omegaVar*uy0 - 36*omegaVar*uz0 - 15*omegaVar*ux0*ux0 - 15*omegaVar*uy0*uy0 - 60*omegaVar*uz0*uz0 + 90*omegaVar*ux0*uz0 + 90*omegaVar*uy0*uz0 + 328);

    m_xx_t45 = (2*(rho + 75*m_xx_I*rho_I - 18*m_xz_I*rho_I - 6*m_yy_I*rho_I + 9*m_yz_I*rho_I + 3*m_zz_I*rho_I + 6*rho*ux0 - 3*rho*uy0 - 3*rho*uz0))/(135*rho);
    m_xy_t90 = (m_xy_I*rho_I)/rho;
    m_xz_t90 = -(2*rho + 6*m_xx_I*rho_I - 63*m_xz_I*rho_I - 3*m_yy_I*rho_I + 6*m_zz_I*rho_I + 6*rho*ux0 - 6*rho*uz0)/(45*rho);
    m_yy_t45 = (2*(rho - 6*m_xx_I*rho_I + 9*m_xz_I*rho_I + 75*m_yy_I*rho_I - 18*m_yz_I*rho_I + 3*m_zz_I*rho_I - 3*rho*ux0 + 6*rho*uy0 - 3*rho*uz0))/(135*rho);
    m_yz_t90 = -(2*rho - 3*m_xx_I*rho_I + 6*m_yy_I*rho_I - 63*m_yz_I*rho_I + 6*m_zz_I*rho_I + 6*rho*uy0 - 6*rho*uz0)/(45*rho);
    m_zz_t45 = (8*rho + 6*m_xx_I*rho_I - 36*m_xz_I*rho_I + 6*m_yy_I*rho_I - 36*m_yz_I*rho_I + 159*m_zz_I*rho_I + 12*rho*ux0 + 12*rho*uy0 - 24*rho*uz0)/(135*rho);

    rhoVar = rho;

    break;
case 0b0000010101110001:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[16] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);

    rho = (24*(5*rho_I + m_xx_I*rho_I + 2*m_xy_I*rho_I - 2*m_xz_I*rho_I + m_yy_I*rho_I - 2*m_yz_I*rho_I + m_zz_I*rho_I - omegaVar*m_xx_I*rho_I - 2*omegaVar*m_xy_I*rho_I + 2*omegaVar*m_xz_I*rho_I - omegaVar*m_yy_I*rho_I + 2*omegaVar*m_yz_I*rho_I - omegaVar*m_zz_I*rho_I))/(6*omegaVar + 8*ux0 + 8*uy0 - 8*uz0 + 12*omegaVar*ux0 + 12*omegaVar*uy0 - 12*omegaVar*uz0 - 15*omegaVar*ux0*ux0 - 15*omegaVar*uy0*uy0 - 15*omegaVar*uz0*uz0 - 30*omegaVar*ux0*uy0 + 30*omegaVar*ux0*uz0 + 30*omegaVar*uy0*uz0 + 104);

    m_xx_t45 = (2*(rho + 27*m_xx_I*rho_I + 6*m_xy_I*rho_I - 6*m_xz_I*rho_I + 3*m_yz_I*rho_I + 4*rho*ux0 + rho*uy0 - rho*uz0))/(45*rho);
    m_xy_t90 = (2*rho + 6*m_xx_I*rho_I + 63*m_xy_I*rho_I + 6*m_yy_I*rho_I - 3*m_zz_I*rho_I + 6*rho*ux0 + 6*rho*uy0)/(45*rho);
    m_xz_t90 = -(2*rho + 6*m_xx_I*rho_I - 63*m_xz_I*rho_I - 3*m_yy_I*rho_I + 6*m_zz_I*rho_I + 6*rho*ux0 - 6*rho*uz0)/(45*rho);
    m_yy_t45 = (2*(rho + 6*m_xy_I*rho_I + 3*m_xz_I*rho_I + 27*m_yy_I*rho_I - 6*m_yz_I*rho_I + rho*ux0 + 4*rho*uy0 - rho*uz0))/(45*rho);
    m_yz_t90 = -(2*rho - 3*m_xx_I*rho_I + 6*m_yy_I*rho_I - 63*m_yz_I*rho_I + 6*m_zz_I*rho_I + 6*rho*uy0 - 6*rho*uz0)/(45*rho);
    m_zz_t45 = (2*(rho - 3*m_xy_I*rho_I - 6*m_xz_I*rho_I - 6*m_yz_I*rho_I + 27*m_zz_I*rho_I + rho*ux0 + rho*uy0 - 4*rho*uz0))/(45*rho);

    rhoVar = rho;

    break;
case 0b0000010101110010:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[16] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);

    rho = (24*(5*rho_I + m_xx_I*rho_I - 2*m_xy_I*rho_I - 2*m_xz_I*rho_I + m_yy_I*rho_I - 2*m_yz_I*rho_I + m_zz_I*rho_I - omegaVar*m_xx_I*rho_I + 2*omegaVar*m_xy_I*rho_I + 2*omegaVar*m_xz_I*rho_I - omegaVar*m_yy_I*rho_I + 2*omegaVar*m_yz_I*rho_I - omegaVar*m_zz_I*rho_I))/(6*omegaVar + 8*uy0 - 8*uz0 + 12*omegaVar*uy0 - 12*omegaVar*uz0 - 15*omegaVar*ux0*ux0 - 15*omegaVar*uy0*uy0 - 15*omegaVar*uz0*uz0 + 30*omegaVar*ux0*uy0 + 30*omegaVar*ux0*uz0 + 30*omegaVar*uy0*uz0 + 104);

    m_xx_t45 = (2*(rho + 27*m_xx_I*rho_I - 6*m_xy_I*rho_I - 6*m_xz_I*rho_I + 3*m_yz_I*rho_I + rho*uy0 - rho*uz0))/(45*rho);
    m_xy_t90 = -(2*rho + 6*m_xx_I*rho_I - 63*m_xy_I*rho_I + 6*m_yy_I*rho_I - 3*m_zz_I*rho_I - 6*rho*ux0 + 6*rho*uy0)/(45*rho);
    m_xz_t90 = -(2*rho + 6*m_xx_I*rho_I - 63*m_xz_I*rho_I - 3*m_yy_I*rho_I + 6*m_zz_I*rho_I + 6*rho*ux0 - 6*rho*uz0)/(45*rho);
    m_yy_t45 = (2*(rho - 6*m_xy_I*rho_I + 3*m_xz_I*rho_I + 27*m_yy_I*rho_I - 6*m_yz_I*rho_I - 3*rho*ux0 + 4*rho*uy0 - rho*uz0))/(45*rho);
    m_yz_t90 = -(2*rho - 3*m_xx_I*rho_I + 6*m_yy_I*rho_I - 63*m_yz_I*rho_I + 6*m_zz_I*rho_I + 6*rho*uy0 - 6*rho*uz0)/(45*rho);
    m_zz_t45 = (2*(rho + 3*m_xy_I*rho_I - 6*m_xz_I*rho_I - 6*m_yz_I*rho_I + 27*m_zz_I*rho_I + 3*rho*ux0 + rho*uy0 - 4*rho*uz0))/(45*rho);

    rhoVar = rho;

    break;
case 0b0000010101110011:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[4] + pop[5] + pop[6] + pop[8] + pop[9] + pop[10] + pop[12] + pop[13] + pop[16] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[8] - pop[13]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[12] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);

    rho = (72*(107*rho_I + 12*m_xx_I*rho_I - 33*m_xz_I*rho_I + 72*m_yy_I*rho_I + 12*m_zz_I*rho_I - 12*omegaVar*m_xx_I*rho_I + 33*omegaVar*m_xz_I*rho_I - 72*omegaVar*m_yy_I*rho_I - 12*omegaVar*m_zz_I*rho_I))/(626*omegaVar + 492*ux0 + 2124*uy0 - 492*uz0 + 150*omegaVar*ux0 + 1728*omegaVar*uy0 - 150*omegaVar*uz0 - 642*omegaVar*ux0*ux0 - 3531*omegaVar*uy0*uy0 - 642*omegaVar*uz0*uz0 + 1926*omegaVar*ux0*uz0 + 5580);

    m_xx_t45 = -(4*(24*m_xz_I*rho_I - 106*m_xx_I*rho_I - 2*rho + 6*m_yy_I*rho_I + m_zz_I*rho_I - 8*rho*ux0 + 2*rho*uy0 + 8*rho*uz0))/(321*rho);
    m_xy_t90 = (6*m_xy_I*rho_I + rho*ux0)/(3*rho);
    m_xz_t90 = -(11*rho + 48*m_xx_I*rho_I - 453*m_xz_I*rho_I - 33*m_yy_I*rho_I + 48*m_zz_I*rho_I + 44*rho*ux0 - 11*rho*uy0 - 44*rho*uz0)/(321*rho);
    m_yy_t45 = (2*(24*rho - 12*m_xx_I*rho_I + 33*m_xz_I*rho_I + 249*m_yy_I*rho_I - 12*m_zz_I*rho_I - 11*rho*ux0 + 83*rho*uy0 + 11*rho*uz0))/(321*rho);
    m_yz_t90 = (6*m_yz_I*rho_I + rho*uz0)/(3*rho);
    m_zz_t45 = -(4*(m_xx_I*rho_I - 2*rho + 24*m_xz_I*rho_I + 6*m_yy_I*rho_I - 106*m_zz_I*rho_I - 8*rho*ux0 + 2*rho*uy0 + 8*rho*uz0))/(321*rho);

    rhoVar = rho;

    break;
case 0b0000010101110100:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[14] + pop[16] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);

    rho = (24*(5*rho_I + m_xx_I*rho_I - 2*m_xy_I*rho_I - 2*m_xz_I*rho_I + m_yy_I*rho_I - 2*m_yz_I*rho_I + m_zz_I*rho_I - omegaVar*m_xx_I*rho_I + 2*omegaVar*m_xy_I*rho_I + 2*omegaVar*m_xz_I*rho_I - omegaVar*m_yy_I*rho_I + 2*omegaVar*m_yz_I*rho_I - omegaVar*m_zz_I*rho_I))/(6*omegaVar + 8*ux0 - 8*uz0 + 12*omegaVar*ux0 - 12*omegaVar*uz0 - 15*omegaVar*ux0*ux0 - 15*omegaVar*uy0*uy0 - 15*omegaVar*uz0*uz0 + 30*omegaVar*ux0*uy0 + 30*omegaVar*ux0*uz0 + 30*omegaVar*uy0*uz0 + 104);

    m_xx_t45 = (2*(rho + 27*m_xx_I*rho_I - 6*m_xy_I*rho_I - 6*m_xz_I*rho_I + 3*m_yz_I*rho_I + 4*rho*ux0 - 3*rho*uy0 - rho*uz0))/(45*rho);
    m_xy_t90 = -(2*rho + 6*m_xx_I*rho_I - 63*m_xy_I*rho_I + 6*m_yy_I*rho_I - 3*m_zz_I*rho_I + 6*rho*ux0 - 6*rho*uy0)/(45*rho);
    m_xz_t90 = -(2*rho + 6*m_xx_I*rho_I - 63*m_xz_I*rho_I - 3*m_yy_I*rho_I + 6*m_zz_I*rho_I + 6*rho*ux0 - 6*rho*uz0)/(45*rho);
    m_yy_t45 = (2*(rho - 6*m_xy_I*rho_I + 3*m_xz_I*rho_I + 27*m_yy_I*rho_I - 6*m_yz_I*rho_I + rho*ux0 - rho*uz0))/(45*rho);
    m_yz_t90 = -(2*rho - 3*m_xx_I*rho_I + 6*m_yy_I*rho_I - 63*m_yz_I*rho_I + 6*m_zz_I*rho_I + 6*rho*uy0 - 6*rho*uz0)/(45*rho);
    m_zz_t45 = (2*(rho + 3*m_xy_I*rho_I - 6*m_xz_I*rho_I - 6*m_yz_I*rho_I + 27*m_zz_I*rho_I + rho*ux0 + 3*rho*uy0 - 4*rho*uz0))/(45*rho);

    rhoVar = rho;

    break;
case 0b0000010101110101:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[8] + pop[10] + pop[11] + pop[12] + pop[14] + pop[16] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[8] - pop[14]);
    m_xz_I = inv_rho_I * (pop[10] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);

    rho = (72*(107*rho_I + 72*m_xx_I*rho_I + 12*m_yy_I*rho_I - 33*m_yz_I*rho_I + 12*m_zz_I*rho_I - 72*omegaVar*m_xx_I*rho_I - 12*omegaVar*m_yy_I*rho_I + 33*omegaVar*m_yz_I*rho_I - 12*omegaVar*m_zz_I*rho_I))/(626*omegaVar + 2124*ux0 + 492*uy0 - 492*uz0 + 1728*omegaVar*ux0 + 150*omegaVar*uy0 - 150*omegaVar*uz0 - 3531*omegaVar*ux0*ux0 - 642*omegaVar*uy0*uy0 - 642*omegaVar*uz0*uz0 + 1926*omegaVar*uy0*uz0 + 5580);

    m_xx_t45 = (2*(24*rho + 249*m_xx_I*rho_I - 12*m_yy_I*rho_I + 33*m_yz_I*rho_I - 12*m_zz_I*rho_I + 83*rho*ux0 - 11*rho*uy0 + 11*rho*uz0))/(321*rho);
    m_xy_t90 = (6*m_xy_I*rho_I + rho*uy0)/(3*rho);
    m_xz_t90 = (6*m_xz_I*rho_I + rho*uz0)/(3*rho);
    m_yy_t45 = -(4*(6*m_xx_I*rho_I - 2*rho - 106*m_yy_I*rho_I + 24*m_yz_I*rho_I + m_zz_I*rho_I + 2*rho*ux0 - 8*rho*uy0 + 8*rho*uz0))/(321*rho);
    m_yz_t90 = -(11*rho - 33*m_xx_I*rho_I + 48*m_yy_I*rho_I - 453*m_yz_I*rho_I + 48*m_zz_I*rho_I - 11*rho*ux0 + 44*rho*uy0 - 44*rho*uz0)/(321*rho);
    m_zz_t45 = -(4*(6*m_xx_I*rho_I - 2*rho + m_yy_I*rho_I + 24*m_yz_I*rho_I - 106*m_zz_I*rho_I + 2*rho*ux0 - 8*rho*uy0 + 8*rho*uz0))/(321*rho);

    rhoVar = rho;

    break;
case 0b0000010101110110:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[16] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);

    rho = (24*(15*rho_I + 11*m_xx_I*rho_I - 30*m_xy_I*rho_I - 6*m_xz_I*rho_I + 11*m_yy_I*rho_I - 6*m_yz_I*rho_I - m_zz_I*rho_I - 11*omegaVar*m_xx_I*rho_I + 30*omegaVar*m_xy_I*rho_I + 6*omegaVar*m_xz_I*rho_I - 11*omegaVar*m_yy_I*rho_I + 6*omegaVar*m_yz_I*rho_I + omegaVar*m_zz_I*rho_I))/(72*omegaVar + 12*ux0 + 12*uy0 - 24*uz0 + 18*omegaVar*ux0 + 18*omegaVar*uy0 - 36*omegaVar*uz0 - 75*omegaVar*ux0*ux0 - 75*omegaVar*uy0*uy0 - 30*omegaVar*uz0*uz0 + 180*omegaVar*ux0*uy0 + 90*omegaVar*ux0*uz0 + 90*omegaVar*uy0*uz0 + 248);

    m_xx_t45 = (2*(11*rho + 105*m_xx_I*rho_I - 90*m_xy_I*rho_I - 18*m_xz_I*rho_I + 24*m_yy_I*rho_I + 9*m_yz_I*rho_I - 12*m_zz_I*rho_I + 6*rho*ux0 - 3*rho*uy0 - 3*rho*uz0))/(135*rho);
    m_xy_t90 = -(2*rho + 6*m_xx_I*rho_I - 27*m_xy_I*rho_I + 6*m_yy_I*rho_I - 3*m_zz_I*rho_I)/(9*rho);
    m_xz_t90 = -(2*rho + 6*m_xx_I*rho_I - 63*m_xz_I*rho_I - 3*m_yy_I*rho_I + 6*m_zz_I*rho_I + 6*rho*ux0 - 6*rho*uz0)/(45*rho);
    m_yy_t45 = (2*(11*rho + 24*m_xx_I*rho_I - 90*m_xy_I*rho_I + 9*m_xz_I*rho_I + 105*m_yy_I*rho_I - 18*m_yz_I*rho_I - 12*m_zz_I*rho_I - 3*rho*ux0 + 6*rho*uy0 - 3*rho*uz0))/(135*rho);
    m_yz_t90 = -(2*rho - 3*m_xx_I*rho_I + 6*m_yy_I*rho_I - 63*m_yz_I*rho_I + 6*m_zz_I*rho_I + 6*rho*uy0 - 6*rho*uz0)/(45*rho);
    m_zz_t45 = -(2*(rho + 12*m_xx_I*rho_I - 45*m_xy_I*rho_I + 18*m_xz_I*rho_I + 12*m_yy_I*rho_I + 18*m_yz_I*rho_I - 87*m_zz_I*rho_I - 6*rho*ux0 - 6*rho*uy0 + 12*rho*uz0))/(135*rho);

    rhoVar = rho;

    break;
case 0b0000010101110111:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[2] + pop[4] + pop[5] + pop[6] + pop[8] + pop[10] + pop[12] + pop[16] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[8]);
    m_xz_I = inv_rho_I * (pop[10] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[12] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);

    rho = (72*(23*rho_I + 24*m_xx_I*rho_I - 57*m_xy_I*rho_I + 24*m_yy_I*rho_I - 6*m_zz_I*rho_I - 24*omegaVar*m_xx_I*rho_I + 57*omegaVar*m_xy_I*rho_I - 24*omegaVar*m_yy_I*rho_I + 6*omegaVar*m_zz_I*rho_I))/(430*omegaVar + 660*ux0 + 660*uy0 + 30*omegaVar*ux0 + 30*omegaVar*uy0 - 690*omegaVar*ux0*ux0 - 690*omegaVar*uy0*uy0 - 69*omegaVar*uz0*uz0 + 414*omegaVar*ux0*uy0 + 720);

    m_xx_t45 = (16*rho + 141*m_xx_I*rho_I - 102*m_xy_I*rho_I + 3*m_yy_I*rho_I - 18*m_zz_I*rho_I + 30*rho*ux0 - 16*rho*uy0)/(69*rho);
    m_xy_t90 = (354*m_xy_I*rho_I - 51*m_xx_I*rho_I - 19*rho - 51*m_yy_I*rho_I + 30*m_zz_I*rho_I + 19*rho*ux0 + 19*rho*uy0)/(69*rho);
    m_xz_t90 = (6*m_xz_I*rho_I + rho*uz0)/(3*rho);
    m_yy_t45 = (16*rho + 3*m_xx_I*rho_I - 102*m_xy_I*rho_I + 141*m_yy_I*rho_I - 18*m_zz_I*rho_I - 16*rho*ux0 + 30*rho*uy0)/(69*rho);
    m_yz_t90 = (6*m_yz_I*rho_I + rho*uz0)/(3*rho);
    m_zz_t45 = (2*(30*m_xy_I*rho_I - 9*m_xx_I*rho_I - 2*rho - 9*m_yy_I*rho_I + 54*m_zz_I*rho_I + 2*rho*ux0 + 2*rho*uy0))/(69*rho);

    rhoVar = rho;

    break;
case 0b0000010101111000:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[16] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);

    rho = (24*(15*rho_I + m_xx_I*rho_I - 6*m_xz_I*rho_I + m_yy_I*rho_I - 6*m_yz_I*rho_I + 4*m_zz_I*rho_I - omegaVar*m_xx_I*rho_I + 6*omegaVar*m_xz_I*rho_I - omegaVar*m_yy_I*rho_I + 6*omegaVar*m_yz_I*rho_I - 4*omegaVar*m_zz_I*rho_I))/(12*omegaVar + 12*ux0 + 12*uy0 - 24*uz0 + 18*omegaVar*ux0 + 18*omegaVar*uy0 - 36*omegaVar*uz0 - 15*omegaVar*ux0*ux0 - 15*omegaVar*uy0*uy0 - 60*omegaVar*uz0*uz0 + 90*omegaVar*ux0*uz0 + 90*omegaVar*uy0*uz0 + 328);

    m_xx_t45 = (2*(rho + 75*m_xx_I*rho_I - 18*m_xz_I*rho_I - 6*m_yy_I*rho_I + 9*m_yz_I*rho_I + 3*m_zz_I*rho_I + 6*rho*ux0 - 3*rho*uy0 - 3*rho*uz0))/(135*rho);
    m_xy_t90 = (m_xy_I*rho_I)/rho;
    m_xz_t90 = -(2*rho + 6*m_xx_I*rho_I - 63*m_xz_I*rho_I - 3*m_yy_I*rho_I + 6*m_zz_I*rho_I + 6*rho*ux0 - 6*rho*uz0)/(45*rho);
    m_yy_t45 = (2*(rho - 6*m_xx_I*rho_I + 9*m_xz_I*rho_I + 75*m_yy_I*rho_I - 18*m_yz_I*rho_I + 3*m_zz_I*rho_I - 3*rho*ux0 + 6*rho*uy0 - 3*rho*uz0))/(135*rho);
    m_yz_t90 = -(2*rho - 3*m_xx_I*rho_I + 6*m_yy_I*rho_I - 63*m_yz_I*rho_I + 6*m_zz_I*rho_I + 6*rho*uy0 - 6*rho*uz0)/(45*rho);
    m_zz_t45 = (8*rho + 6*m_xx_I*rho_I - 36*m_xz_I*rho_I + 6*m_yy_I*rho_I - 36*m_yz_I*rho_I + 159*m_zz_I*rho_I + 12*rho*ux0 + 12*rho*uy0 - 24*rho*uz0)/(135*rho);

    rhoVar = rho;

    break;
case 0b0000010101111001:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[16] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);

    rho = (24*(5*rho_I + m_xx_I*rho_I + 2*m_xy_I*rho_I - 2*m_xz_I*rho_I + m_yy_I*rho_I - 2*m_yz_I*rho_I + m_zz_I*rho_I - omegaVar*m_xx_I*rho_I - 2*omegaVar*m_xy_I*rho_I + 2*omegaVar*m_xz_I*rho_I - omegaVar*m_yy_I*rho_I + 2*omegaVar*m_yz_I*rho_I - omegaVar*m_zz_I*rho_I))/(6*omegaVar + 8*ux0 + 8*uy0 - 8*uz0 + 12*omegaVar*ux0 + 12*omegaVar*uy0 - 12*omegaVar*uz0 - 15*omegaVar*ux0*ux0 - 15*omegaVar*uy0*uy0 - 15*omegaVar*uz0*uz0 - 30*omegaVar*ux0*uy0 + 30*omegaVar*ux0*uz0 + 30*omegaVar*uy0*uz0 + 104);

    m_xx_t45 = (2*(rho + 27*m_xx_I*rho_I + 6*m_xy_I*rho_I - 6*m_xz_I*rho_I + 3*m_yz_I*rho_I + 4*rho*ux0 + rho*uy0 - rho*uz0))/(45*rho);
    m_xy_t90 = (2*rho + 6*m_xx_I*rho_I + 63*m_xy_I*rho_I + 6*m_yy_I*rho_I - 3*m_zz_I*rho_I + 6*rho*ux0 + 6*rho*uy0)/(45*rho);
    m_xz_t90 = -(2*rho + 6*m_xx_I*rho_I - 63*m_xz_I*rho_I - 3*m_yy_I*rho_I + 6*m_zz_I*rho_I + 6*rho*ux0 - 6*rho*uz0)/(45*rho);
    m_yy_t45 = (2*(rho + 6*m_xy_I*rho_I + 3*m_xz_I*rho_I + 27*m_yy_I*rho_I - 6*m_yz_I*rho_I + rho*ux0 + 4*rho*uy0 - rho*uz0))/(45*rho);
    m_yz_t90 = -(2*rho - 3*m_xx_I*rho_I + 6*m_yy_I*rho_I - 63*m_yz_I*rho_I + 6*m_zz_I*rho_I + 6*rho*uy0 - 6*rho*uz0)/(45*rho);
    m_zz_t45 = (2*(rho - 3*m_xy_I*rho_I - 6*m_xz_I*rho_I - 6*m_yz_I*rho_I + 27*m_zz_I*rho_I + rho*ux0 + rho*uy0 - 4*rho*uz0))/(45*rho);

    rhoVar = rho;

    break;
case 0b0000010101111010:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] + (2.0/3.0)*pop[18]);

    rho = (24*(15*rho_I + 11*m_xx_I*rho_I - 6*m_xy_I*rho_I - 30*m_xz_I*rho_I - m_yy_I*rho_I - 6*m_yz_I*rho_I + 11*m_zz_I*rho_I - 11*omegaVar*m_xx_I*rho_I + 6*omegaVar*m_xy_I*rho_I + 30*omegaVar*m_xz_I*rho_I + omegaVar*m_yy_I*rho_I + 6*omegaVar*m_yz_I*rho_I - 11*omegaVar*m_zz_I*rho_I))/(72*omegaVar - 12*ux0 + 24*uy0 - 12*uz0 - 18*omegaVar*ux0 + 36*omegaVar*uy0 - 18*omegaVar*uz0 - 75*omegaVar*ux0*ux0 - 30*omegaVar*uy0*uy0 - 75*omegaVar*uz0*uz0 + 90*omegaVar*ux0*uy0 + 180*omegaVar*ux0*uz0 + 90*omegaVar*uy0*uz0 + 248);

    m_xx_t45 = (2*(11*rho + 105*m_xx_I*rho_I - 18*m_xy_I*rho_I - 90*m_xz_I*rho_I - 12*m_yy_I*rho_I + 9*m_yz_I*rho_I + 24*m_zz_I*rho_I - 6*rho*ux0 + 3*rho*uy0 + 3*rho*uz0))/(135*rho);
    m_xy_t90 = -(2*rho + 6*m_xx_I*rho_I - 63*m_xy_I*rho_I + 6*m_yy_I*rho_I - 3*m_zz_I*rho_I - 6*rho*ux0 + 6*rho*uy0)/(45*rho);
    m_xz_t90 = -(2*rho + 6*m_xx_I*rho_I - 27*m_xz_I*rho_I - 3*m_yy_I*rho_I + 6*m_zz_I*rho_I)/(9*rho);
    m_yy_t45 = -(2*(rho + 12*m_xx_I*rho_I + 18*m_xy_I*rho_I - 45*m_xz_I*rho_I - 87*m_yy_I*rho_I + 18*m_yz_I*rho_I + 12*m_zz_I*rho_I + 6*rho*ux0 - 12*rho*uy0 + 6*rho*uz0))/(135*rho);
    m_yz_t90 = -(2*rho - 3*m_xx_I*rho_I + 6*m_yy_I*rho_I - 63*m_yz_I*rho_I + 6*m_zz_I*rho_I + 6*rho*uy0 - 6*rho*uz0)/(45*rho);
    m_zz_t45 = (2*(11*rho + 24*m_xx_I*rho_I + 9*m_xy_I*rho_I - 90*m_xz_I*rho_I - 12*m_yy_I*rho_I - 18*m_yz_I*rho_I + 105*m_zz_I*rho_I + 3*rho*ux0 + 3*rho*uy0 - 6*rho*uz0))/(135*rho);

    rhoVar = rho;

    break;
case 0b0000010101111011:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[4] + pop[5] + pop[6] + pop[8] + pop[9] + pop[10] + pop[12] + pop[13] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[8] - pop[13]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[12] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] + (2.0/3.0)*pop[18]);

    rho = (36*(19*rho_I + 12*m_xx_I*rho_I - 33*m_xz_I*rho_I + 6*m_yy_I*rho_I + 12*m_zz_I*rho_I - 12*omegaVar*m_xx_I*rho_I + 33*omegaVar*m_xz_I*rho_I - 6*omegaVar*m_yy_I*rho_I - 12*omegaVar*m_zz_I*rho_I))/(118*omegaVar + 270*uy0 + 72*omegaVar*uy0 - 114*omegaVar*ux0*ux0 - 285*omegaVar*uy0*uy0 - 114*omegaVar*uz0*uz0 + 342*omegaVar*ux0*uz0 + 414);

    m_xx_t45 = (4*(2*rho + 26*m_xx_I*rho_I - 24*m_xz_I*rho_I - 6*m_yy_I*rho_I + 7*m_zz_I*rho_I - 2*rho*uy0))/(57*rho);
    m_xy_t90 = (6*m_xy_I*rho_I + rho*ux0)/(3*rho);
    m_xz_t90 = -(11*rho + 48*m_xx_I*rho_I - 189*m_xz_I*rho_I - 33*m_yy_I*rho_I + 48*m_zz_I*rho_I - 11*rho*uy0)/(57*rho);
    m_yy_t45 = (2*(2*rho - 12*m_xx_I*rho_I + 33*m_xz_I*rho_I + 51*m_yy_I*rho_I - 12*m_zz_I*rho_I + 17*rho*uy0))/(57*rho);
    m_yz_t90 = (6*m_yz_I*rho_I + rho*uz0)/(3*rho);
    m_zz_t45 = (4*(2*rho + 7*m_xx_I*rho_I - 24*m_xz_I*rho_I - 6*m_yy_I*rho_I + 26*m_zz_I*rho_I - 2*rho*uy0))/(57*rho);

    rhoVar = rho;

    break;
case 0b0000010101111100:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[14] + pop[16];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[16]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[16]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[16]);

    rho = (24*(15*rho_I - m_xx_I*rho_I - 6*m_xy_I*rho_I - 6*m_xz_I*rho_I + 11*m_yy_I*rho_I - 30*m_yz_I*rho_I + 11*m_zz_I*rho_I + omegaVar*m_xx_I*rho_I + 6*omegaVar*m_xy_I*rho_I + 6*omegaVar*m_xz_I*rho_I - 11*omegaVar*m_yy_I*rho_I + 30*omegaVar*m_yz_I*rho_I - 11*omegaVar*m_zz_I*rho_I))/(72*omegaVar + 24*ux0 - 12*uy0 - 12*uz0 + 36*omegaVar*ux0 - 18*omegaVar*uy0 - 18*omegaVar*uz0 - 30*omegaVar*ux0*ux0 - 75*omegaVar*uy0*uy0 - 75*omegaVar*uz0*uz0 + 90*omegaVar*ux0*uy0 + 90*omegaVar*ux0*uz0 + 180*omegaVar*uy0*uz0 + 248);

    m_xx_t45 = -(2*(rho - 87*m_xx_I*rho_I + 18*m_xy_I*rho_I + 18*m_xz_I*rho_I + 12*m_yy_I*rho_I - 45*m_yz_I*rho_I + 12*m_zz_I*rho_I - 12*rho*ux0 + 6*rho*uy0 + 6*rho*uz0))/(135*rho);
    m_xy_t90 = -(2*rho + 6*m_xx_I*rho_I - 63*m_xy_I*rho_I + 6*m_yy_I*rho_I - 3*m_zz_I*rho_I + 6*rho*ux0 - 6*rho*uy0)/(45*rho);
    m_xz_t90 = -(2*rho + 6*m_xx_I*rho_I - 63*m_xz_I*rho_I - 3*m_yy_I*rho_I + 6*m_zz_I*rho_I + 6*rho*ux0 - 6*rho*uz0)/(45*rho);
    m_yy_t45 = (2*(11*rho - 12*m_xx_I*rho_I - 18*m_xy_I*rho_I + 9*m_xz_I*rho_I + 105*m_yy_I*rho_I - 90*m_yz_I*rho_I + 24*m_zz_I*rho_I + 3*rho*ux0 - 6*rho*uy0 + 3*rho*uz0))/(135*rho);
    m_yz_t90 = -(2*rho - 3*m_xx_I*rho_I + 6*m_yy_I*rho_I - 27*m_yz_I*rho_I + 6*m_zz_I*rho_I)/(9*rho);
    m_zz_t45 = (2*(11*rho - 12*m_xx_I*rho_I + 9*m_xy_I*rho_I - 18*m_xz_I*rho_I + 24*m_yy_I*rho_I - 90*m_yz_I*rho_I + 105*m_zz_I*rho_I + 3*rho*ux0 + 3*rho*uy0 - 6*rho*uz0))/(135*rho);

    rhoVar = rho;

    break;
case 0b0000010101111101:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[8] + pop[10] + pop[11] + pop[12] + pop[14] + pop[16];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[16]);
    m_xy_I = inv_rho_I * (pop[8] - pop[14]);
    m_xz_I = inv_rho_I * (pop[10] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[16]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[16]);

    rho = (36*(19*rho_I + 6*m_xx_I*rho_I + 12*m_yy_I*rho_I - 33*m_yz_I*rho_I + 12*m_zz_I*rho_I - 6*omegaVar*m_xx_I*rho_I - 12*omegaVar*m_yy_I*rho_I + 33*omegaVar*m_yz_I*rho_I - 12*omegaVar*m_zz_I*rho_I))/(118*omegaVar + 270*ux0 + 72*omegaVar*ux0 - 285*omegaVar*ux0*ux0 - 114*omegaVar*uy0*uy0 - 114*omegaVar*uz0*uz0 + 342*omegaVar*uy0*uz0 + 414);

    m_xx_t45 = (2*(2*rho + 51*m_xx_I*rho_I - 12*m_yy_I*rho_I + 33*m_yz_I*rho_I - 12*m_zz_I*rho_I + 17*rho*ux0))/(57*rho);
    m_xy_t90 = (6*m_xy_I*rho_I + rho*uy0)/(3*rho);
    m_xz_t90 = (6*m_xz_I*rho_I + rho*uz0)/(3*rho);
    m_yy_t45 = (4*(2*rho - 6*m_xx_I*rho_I + 26*m_yy_I*rho_I - 24*m_yz_I*rho_I + 7*m_zz_I*rho_I - 2*rho*ux0))/(57*rho);
    m_yz_t90 = -(11*rho - 33*m_xx_I*rho_I + 48*m_yy_I*rho_I - 189*m_yz_I*rho_I + 48*m_zz_I*rho_I - 11*rho*ux0)/(57*rho);
    m_zz_t45 = (4*(2*rho - 6*m_xx_I*rho_I + 7*m_yy_I*rho_I - 24*m_yz_I*rho_I + 26*m_zz_I*rho_I - 2*rho*ux0))/(57*rho);

    rhoVar = rho;

    break;
case 0b0000010101111110:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12]);

    rho = (12*(rho_I + m_xx_I*rho_I - 2*m_xy_I*rho_I - 2*m_xz_I*rho_I + m_yy_I*rho_I - 2*m_yz_I*rho_I + m_zz_I*rho_I - omegaVar*m_xx_I*rho_I + 2*omegaVar*m_xy_I*rho_I + 2*omegaVar*m_xz_I*rho_I - omegaVar*m_yy_I*rho_I + 2*omegaVar*m_yz_I*rho_I - omegaVar*m_zz_I*rho_I))/(6*omegaVar - 3*omegaVar*ux0*ux0 - 3*omegaVar*uy0*uy0 - 3*omegaVar*uz0*uz0 + 6*omegaVar*ux0*uy0 + 6*omegaVar*ux0*uz0 + 6*omegaVar*uy0*uz0 + 4);

    m_xx_t45 = (2*(rho + 9*m_xx_I*rho_I - 6*m_xy_I*rho_I - 6*m_xz_I*rho_I + 3*m_yz_I*rho_I))/(9*rho);
    m_xy_t90 = -(2*rho + 6*m_xx_I*rho_I - 27*m_xy_I*rho_I + 6*m_yy_I*rho_I - 3*m_zz_I*rho_I)/(9*rho);
    m_xz_t90 = -(2*rho + 6*m_xx_I*rho_I - 27*m_xz_I*rho_I - 3*m_yy_I*rho_I + 6*m_zz_I*rho_I)/(9*rho);
    m_yy_t45 = (2*(rho - 6*m_xy_I*rho_I + 3*m_xz_I*rho_I + 9*m_yy_I*rho_I - 6*m_yz_I*rho_I))/(9*rho);
    m_yz_t90 = -(2*rho - 3*m_xx_I*rho_I + 6*m_yy_I*rho_I - 27*m_yz_I*rho_I + 6*m_zz_I*rho_I)/(9*rho);
    m_zz_t45 = (2*(rho + 3*m_xy_I*rho_I - 6*m_xz_I*rho_I - 6*m_yz_I*rho_I + 9*m_zz_I*rho_I))/(9*rho);

    rhoVar = rho;

    break;
case 0b0000010101111111:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[2] + pop[4] + pop[6] + pop[8] + pop[10] + pop[12];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[12]);
    m_xy_I = inv_rho_I * (pop[8]);
    m_xz_I = inv_rho_I * (pop[10]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[12]);
    m_yz_I = inv_rho_I * (pop[12]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[12]);

    rho = (24*(rho_I + m_xx_I*rho_I - 2*m_xy_I*rho_I - 2*m_xz_I*rho_I + m_yy_I*rho_I - 2*m_yz_I*rho_I + m_zz_I*rho_I - omegaVar*m_xx_I*rho_I + 2*omegaVar*m_xy_I*rho_I + 2*omegaVar*m_xz_I*rho_I - omegaVar*m_yy_I*rho_I + 2*omegaVar*m_yz_I*rho_I - omegaVar*m_zz_I*rho_I))/(10*omegaVar + 12*ux0 + 12*uy0 + 12*uz0 - 4*omegaVar*ux0 - 4*omegaVar*uy0 - 4*omegaVar*uz0 - 9*omegaVar*ux0*ux0 - 9*omegaVar*uy0*uy0 - 9*omegaVar*uz0*uz0 + 6*omegaVar*ux0*uy0 + 6*omegaVar*ux0*uz0 + 6*omegaVar*uy0*uz0 + 4);

    m_xx_t45 = -(2*(9*m_xy_I*rho_I - 15*m_xx_I*rho_I - rho + 9*m_xz_I*rho_I + 3*m_yy_I*rho_I - 9*m_yz_I*rho_I + 3*m_zz_I*rho_I - 2*rho*ux0 + rho*uy0 + rho*uz0))/(9*rho);
    m_xy_t90 = -(2*rho + 9*m_xx_I*rho_I - 51*m_xy_I*rho_I + 3*m_xz_I*rho_I + 9*m_yy_I*rho_I + 3*m_yz_I*rho_I - 9*m_zz_I*rho_I - 2*rho*ux0 - 2*rho*uy0 - 2*rho*uz0)/(9*rho);
    m_xz_t90 = -(2*rho + 9*m_xx_I*rho_I + 3*m_xy_I*rho_I - 51*m_xz_I*rho_I - 9*m_yy_I*rho_I + 3*m_yz_I*rho_I + 9*m_zz_I*rho_I - 2*rho*ux0 - 2*rho*uy0 - 2*rho*uz0)/(9*rho);
    m_yy_t45 = -(2*(3*m_xx_I*rho_I - rho + 9*m_xy_I*rho_I - 9*m_xz_I*rho_I - 15*m_yy_I*rho_I + 9*m_yz_I*rho_I + 3*m_zz_I*rho_I + rho*ux0 - 2*rho*uy0 + rho*uz0))/(9*rho);
    m_yz_t90 = -(2*rho - 9*m_xx_I*rho_I + 3*m_xy_I*rho_I + 3*m_xz_I*rho_I + 9*m_yy_I*rho_I - 51*m_yz_I*rho_I + 9*m_zz_I*rho_I - 2*rho*ux0 - 2*rho*uy0 - 2*rho*uz0)/(9*rho);
    m_zz_t45 = -(2*(3*m_xx_I*rho_I - rho - 9*m_xy_I*rho_I + 9*m_xz_I*rho_I + 3*m_yy_I*rho_I + 9*m_yz_I*rho_I - 15*m_zz_I*rho_I + rho*ux0 + rho*uy0 - 2*rho*uz0))/(9*rho);

    rhoVar = rho;

    break;
case 0b0000010110000000:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = rho_I;

    m_xx_t45 = (m_xx_I*rho_I)/rho;
    m_xy_t90 = (m_xy_I*rho_I)/rho;
    m_xz_t90 = (m_xz_I*rho_I)/rho;
    m_yy_t45 = (m_yy_I*rho_I)/rho;
    m_yz_t90 = (m_yz_I*rho_I)/rho;
    m_zz_t45 = (m_zz_I*rho_I)/rho;

    rhoVar = rho;

    break;
case 0b0000010110000001:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = rho_I;

    m_xx_t45 = (m_xx_I*rho_I)/rho;
    m_xy_t90 = (m_xy_I*rho_I)/rho;
    m_xz_t90 = (m_xz_I*rho_I)/rho;
    m_yy_t45 = (m_yy_I*rho_I)/rho;
    m_yz_t90 = (m_yz_I*rho_I)/rho;
    m_zz_t45 = (m_zz_I*rho_I)/rho;

    rhoVar = rho;

    break;
case 0b0000010110000010:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = rho_I;

    m_xx_t45 = (m_xx_I*rho_I)/rho;
    m_xy_t90 = (m_xy_I*rho_I)/rho;
    m_xz_t90 = (m_xz_I*rho_I)/rho;
    m_yy_t45 = (m_yy_I*rho_I)/rho;
    m_yz_t90 = (m_yz_I*rho_I)/rho;
    m_zz_t45 = (m_zz_I*rho_I)/rho;

    rhoVar = rho;

    break;
case 0b0000010110000011:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (24*(15*rho_I - m_xx_I*rho_I + 2*m_yy_I*rho_I + 6*m_yz_I*rho_I + 2*m_zz_I*rho_I + omegaVar*m_xx_I*rho_I - 2*omegaVar*m_yy_I*rho_I - 6*omegaVar*m_yz_I*rho_I - 2*omegaVar*m_zz_I*rho_I))/(6*omegaVar + 12*uy0 + 12*uz0 + 18*omegaVar*uy0 + 18*omegaVar*uz0 + 15*omegaVar*ux0*ux0 - 30*omegaVar*uy0*uy0 - 30*omegaVar*uz0*uz0 - 90*omegaVar*uy0*uz0 + 344);

    m_xx_t45 = -(2*(rho - 69*m_xx_I*rho_I + 3*m_yy_I*rho_I + 9*m_yz_I*rho_I + 3*m_zz_I*rho_I + 3*rho*uy0 + 3*rho*uz0))/(135*rho);
    m_xy_t90 = (m_xy_I*rho_I)/rho;
    m_xz_t90 = (m_xz_I*rho_I)/rho;
    m_yy_t45 = (4*rho - 6*m_xx_I*rho_I + 147*m_yy_I*rho_I + 36*m_yz_I*rho_I + 12*m_zz_I*rho_I + 12*rho*uy0 + 12*rho*uz0)/(135*rho);
    m_yz_t90 = (2*rho - 3*m_xx_I*rho_I + 6*m_yy_I*rho_I + 63*m_yz_I*rho_I + 6*m_zz_I*rho_I + 6*rho*uy0 + 6*rho*uz0)/(45*rho);
    m_zz_t45 = (4*rho - 6*m_xx_I*rho_I + 12*m_yy_I*rho_I + 36*m_yz_I*rho_I + 147*m_zz_I*rho_I + 12*rho*uy0 + 12*rho*uz0)/(135*rho);

    rhoVar = rho;

    break;
case 0b0000010110000100:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = rho_I;

    m_xx_t45 = (m_xx_I*rho_I)/rho;
    m_xy_t90 = (m_xy_I*rho_I)/rho;
    m_xz_t90 = (m_xz_I*rho_I)/rho;
    m_yy_t45 = (m_yy_I*rho_I)/rho;
    m_yz_t90 = (m_yz_I*rho_I)/rho;
    m_zz_t45 = (m_zz_I*rho_I)/rho;

    rhoVar = rho;

    break;
case 0b0000010110000101:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (24*(15*rho_I + 2*m_xx_I*rho_I + 6*m_xz_I*rho_I - m_yy_I*rho_I + 2*m_zz_I*rho_I - 2*omegaVar*m_xx_I*rho_I - 6*omegaVar*m_xz_I*rho_I + omegaVar*m_yy_I*rho_I - 2*omegaVar*m_zz_I*rho_I))/(6*omegaVar + 12*ux0 + 12*uz0 + 18*omegaVar*ux0 + 18*omegaVar*uz0 - 30*omegaVar*ux0*ux0 + 15*omegaVar*uy0*uy0 - 30*omegaVar*uz0*uz0 - 90*omegaVar*ux0*uz0 + 344);

    m_xx_t45 = (4*rho + 147*m_xx_I*rho_I + 36*m_xz_I*rho_I - 6*m_yy_I*rho_I + 12*m_zz_I*rho_I + 12*rho*ux0 + 12*rho*uz0)/(135*rho);
    m_xy_t90 = (m_xy_I*rho_I)/rho;
    m_xz_t90 = (2*rho + 6*m_xx_I*rho_I + 63*m_xz_I*rho_I - 3*m_yy_I*rho_I + 6*m_zz_I*rho_I + 6*rho*ux0 + 6*rho*uz0)/(45*rho);
    m_yy_t45 = -(2*(rho + 3*m_xx_I*rho_I + 9*m_xz_I*rho_I - 69*m_yy_I*rho_I + 3*m_zz_I*rho_I + 3*rho*ux0 + 3*rho*uz0))/(135*rho);
    m_yz_t90 = (m_yz_I*rho_I)/rho;
    m_zz_t45 = (4*rho + 12*m_xx_I*rho_I + 36*m_xz_I*rho_I - 6*m_yy_I*rho_I + 147*m_zz_I*rho_I + 12*rho*ux0 + 12*rho*uz0)/(135*rho);

    rhoVar = rho;

    break;
case 0b0000010110000110:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = rho_I;

    m_xx_t45 = (m_xx_I*rho_I)/rho;
    m_xy_t90 = (m_xy_I*rho_I)/rho;
    m_xz_t90 = (m_xz_I*rho_I)/rho;
    m_yy_t45 = (m_yy_I*rho_I)/rho;
    m_yz_t90 = (m_yz_I*rho_I)/rho;
    m_zz_t45 = (m_zz_I*rho_I)/rho;

    rhoVar = rho;

    break;
case 0b0000010110000111:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[10] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (24*(15*rho_I + m_xx_I*rho_I + 6*m_xz_I*rho_I + m_yy_I*rho_I + 6*m_yz_I*rho_I + 4*m_zz_I*rho_I - omegaVar*m_xx_I*rho_I - 6*omegaVar*m_xz_I*rho_I - omegaVar*m_yy_I*rho_I - 6*omegaVar*m_yz_I*rho_I - 4*omegaVar*m_zz_I*rho_I))/(12*omegaVar + 12*ux0 + 12*uy0 + 24*uz0 + 18*omegaVar*ux0 + 18*omegaVar*uy0 + 36*omegaVar*uz0 - 15*omegaVar*ux0*ux0 - 15*omegaVar*uy0*uy0 - 60*omegaVar*uz0*uz0 - 90*omegaVar*ux0*uz0 - 90*omegaVar*uy0*uz0 + 328);

    m_xx_t45 = (2*(rho + 75*m_xx_I*rho_I + 18*m_xz_I*rho_I - 6*m_yy_I*rho_I - 9*m_yz_I*rho_I + 3*m_zz_I*rho_I + 6*rho*ux0 - 3*rho*uy0 + 3*rho*uz0))/(135*rho);
    m_xy_t90 = (m_xy_I*rho_I)/rho;
    m_xz_t90 = (2*rho + 6*m_xx_I*rho_I + 63*m_xz_I*rho_I - 3*m_yy_I*rho_I + 6*m_zz_I*rho_I + 6*rho*ux0 + 6*rho*uz0)/(45*rho);
    m_yy_t45 = (2*(rho - 6*m_xx_I*rho_I - 9*m_xz_I*rho_I + 75*m_yy_I*rho_I + 18*m_yz_I*rho_I + 3*m_zz_I*rho_I - 3*rho*ux0 + 6*rho*uy0 + 3*rho*uz0))/(135*rho);
    m_yz_t90 = (2*rho - 3*m_xx_I*rho_I + 6*m_yy_I*rho_I + 63*m_yz_I*rho_I + 6*m_zz_I*rho_I + 6*rho*uy0 + 6*rho*uz0)/(45*rho);
    m_zz_t45 = (8*rho + 6*m_xx_I*rho_I + 36*m_xz_I*rho_I + 6*m_yy_I*rho_I + 36*m_yz_I*rho_I + 159*m_zz_I*rho_I + 12*rho*ux0 + 12*rho*uy0 + 24*rho*uz0)/(135*rho);

    rhoVar = rho;

    break;
case 0b0000010110001000:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = -(24*(15*rho_I + 2*m_xx_I*rho_I + 6*m_xy_I*rho_I + 2*m_yy_I*rho_I - m_zz_I*rho_I - 2*omegaVar*m_xx_I*rho_I - 6*omegaVar*m_xy_I*rho_I - 2*omegaVar*m_yy_I*rho_I + omegaVar*m_zz_I*rho_I))/(12*ux0 - 6*omegaVar + 12*uy0 + 18*omegaVar*ux0 + 18*omegaVar*uy0 + 30*omegaVar*ux0*ux0 + 30*omegaVar*uy0*uy0 - 15*omegaVar*uz0*uz0 + 90*omegaVar*ux0*uy0 - 344);

    m_xx_t45 = (4*rho + 147*m_xx_I*rho_I + 36*m_xy_I*rho_I + 12*m_yy_I*rho_I - 6*m_zz_I*rho_I - 12*rho*ux0 - 12*rho*uy0)/(135*rho);
    m_xy_t90 = (2*rho + 6*m_xx_I*rho_I + 63*m_xy_I*rho_I + 6*m_yy_I*rho_I - 3*m_zz_I*rho_I - 6*rho*ux0 - 6*rho*uy0)/(45*rho);
    m_xz_t90 = (m_xz_I*rho_I)/rho;
    m_yy_t45 = (4*rho + 12*m_xx_I*rho_I + 36*m_xy_I*rho_I + 147*m_yy_I*rho_I - 6*m_zz_I*rho_I - 12*rho*ux0 - 12*rho*uy0)/(135*rho);
    m_yz_t90 = (m_yz_I*rho_I)/rho;
    m_zz_t45 = -(2*(rho + 3*m_xx_I*rho_I + 9*m_xy_I*rho_I + 3*m_yy_I*rho_I - 69*m_zz_I*rho_I - 3*rho*ux0 - 3*rho*uy0))/(135*rho);

    rhoVar = rho;

    break;
case 0b0000010110001001:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = -(24*(15*rho_I + 2*m_xx_I*rho_I + 6*m_xy_I*rho_I + 2*m_yy_I*rho_I - m_zz_I*rho_I - 2*omegaVar*m_xx_I*rho_I - 6*omegaVar*m_xy_I*rho_I - 2*omegaVar*m_yy_I*rho_I + omegaVar*m_zz_I*rho_I))/(12*ux0 - 6*omegaVar + 12*uy0 + 18*omegaVar*ux0 + 18*omegaVar*uy0 + 30*omegaVar*ux0*ux0 + 30*omegaVar*uy0*uy0 - 15*omegaVar*uz0*uz0 + 90*omegaVar*ux0*uy0 - 344);

    m_xx_t45 = (4*rho + 147*m_xx_I*rho_I + 36*m_xy_I*rho_I + 12*m_yy_I*rho_I - 6*m_zz_I*rho_I - 12*rho*ux0 - 12*rho*uy0)/(135*rho);
    m_xy_t90 = (2*rho + 6*m_xx_I*rho_I + 63*m_xy_I*rho_I + 6*m_yy_I*rho_I - 3*m_zz_I*rho_I - 6*rho*ux0 - 6*rho*uy0)/(45*rho);
    m_xz_t90 = (m_xz_I*rho_I)/rho;
    m_yy_t45 = (4*rho + 12*m_xx_I*rho_I + 36*m_xy_I*rho_I + 147*m_yy_I*rho_I - 6*m_zz_I*rho_I - 12*rho*ux0 - 12*rho*uy0)/(135*rho);
    m_yz_t90 = (m_yz_I*rho_I)/rho;
    m_zz_t45 = -(2*(rho + 3*m_xx_I*rho_I + 9*m_xy_I*rho_I + 3*m_yy_I*rho_I - 69*m_zz_I*rho_I - 3*rho*ux0 - 3*rho*uy0))/(135*rho);

    rhoVar = rho;

    break;
case 0b0000010110001010:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = -(24*(15*rho_I + 4*m_xx_I*rho_I + 6*m_xy_I*rho_I - 6*m_xz_I*rho_I + m_yy_I*rho_I + m_zz_I*rho_I - 4*omegaVar*m_xx_I*rho_I - 6*omegaVar*m_xy_I*rho_I + 6*omegaVar*m_xz_I*rho_I - omegaVar*m_yy_I*rho_I - omegaVar*m_zz_I*rho_I))/(24*ux0 - 12*omegaVar + 12*uy0 - 12*uz0 + 36*omegaVar*ux0 + 18*omegaVar*uy0 - 18*omegaVar*uz0 + 60*omegaVar*ux0*ux0 + 15*omegaVar*uy0*uy0 + 15*omegaVar*uz0*uz0 + 90*omegaVar*ux0*uy0 - 90*omegaVar*ux0*uz0 - 328);

    m_xx_t45 = (8*rho + 159*m_xx_I*rho_I + 36*m_xy_I*rho_I - 36*m_xz_I*rho_I + 6*m_yy_I*rho_I + 6*m_zz_I*rho_I - 24*rho*ux0 - 12*rho*uy0 + 12*rho*uz0)/(135*rho);
    m_xy_t90 = (2*rho + 6*m_xx_I*rho_I + 63*m_xy_I*rho_I + 6*m_yy_I*rho_I - 3*m_zz_I*rho_I - 6*rho*ux0 - 6*rho*uy0)/(45*rho);
    m_xz_t90 = -(2*rho + 6*m_xx_I*rho_I - 63*m_xz_I*rho_I - 3*m_yy_I*rho_I + 6*m_zz_I*rho_I - 6*rho*ux0 + 6*rho*uz0)/(45*rho);
    m_yy_t45 = (2*(rho + 3*m_xx_I*rho_I + 18*m_xy_I*rho_I + 9*m_xz_I*rho_I + 75*m_yy_I*rho_I - 6*m_zz_I*rho_I - 3*rho*ux0 - 6*rho*uy0 - 3*rho*uz0))/(135*rho);
    m_yz_t90 = (m_yz_I*rho_I)/rho;
    m_zz_t45 = (2*(rho + 3*m_xx_I*rho_I - 9*m_xy_I*rho_I - 18*m_xz_I*rho_I - 6*m_yy_I*rho_I + 75*m_zz_I*rho_I - 3*rho*ux0 + 3*rho*uy0 + 6*rho*uz0))/(135*rho);

    rhoVar = rho;

    break;
case 0b0000010110001011:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[9] + pop[10] + pop[12] + pop[13] + pop[14] + pop[15] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = -(24*(5*rho_I + m_xx_I*rho_I + 2*m_xy_I*rho_I - 2*m_xz_I*rho_I + m_yy_I*rho_I + 2*m_yz_I*rho_I + m_zz_I*rho_I - omegaVar*m_xx_I*rho_I - 2*omegaVar*m_xy_I*rho_I + 2*omegaVar*m_xz_I*rho_I - omegaVar*m_yy_I*rho_I - 2*omegaVar*m_yz_I*rho_I - omegaVar*m_zz_I*rho_I))/(8*ux0 - 6*omegaVar - 8*uz0 + 12*omegaVar*ux0 - 12*omegaVar*uz0 + 15*omegaVar*ux0*ux0 + 15*omegaVar*uy0*uy0 + 15*omegaVar*uz0*uz0 + 30*omegaVar*ux0*uy0 - 30*omegaVar*ux0*uz0 + 30*omegaVar*uy0*uz0 - 104);

    m_xx_t45 = (2*(rho + 27*m_xx_I*rho_I + 6*m_xy_I*rho_I - 6*m_xz_I*rho_I - 3*m_yz_I*rho_I - 4*rho*ux0 - 3*rho*uy0 + rho*uz0))/(45*rho);
    m_xy_t90 = (2*rho + 6*m_xx_I*rho_I + 63*m_xy_I*rho_I + 6*m_yy_I*rho_I - 3*m_zz_I*rho_I - 6*rho*ux0 - 6*rho*uy0)/(45*rho);
    m_xz_t90 = -(2*rho + 6*m_xx_I*rho_I - 63*m_xz_I*rho_I - 3*m_yy_I*rho_I + 6*m_zz_I*rho_I - 6*rho*ux0 + 6*rho*uz0)/(45*rho);
    m_yy_t45 = (2*(rho + 6*m_xy_I*rho_I + 3*m_xz_I*rho_I + 27*m_yy_I*rho_I + 6*m_yz_I*rho_I - rho*ux0 + rho*uz0))/(45*rho);
    m_yz_t90 = (2*rho - 3*m_xx_I*rho_I + 6*m_yy_I*rho_I + 63*m_yz_I*rho_I + 6*m_zz_I*rho_I + 6*rho*uy0 + 6*rho*uz0)/(45*rho);
    m_zz_t45 = (2*(rho - 3*m_xy_I*rho_I - 6*m_xz_I*rho_I + 6*m_yz_I*rho_I + 27*m_zz_I*rho_I - rho*ux0 + 3*rho*uy0 + 4*rho*uz0))/(45*rho);

    rhoVar = rho;

    break;
case 0b0000010110001100:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17]);
    m_xy_I = inv_rho_I * (pop[7] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17]);

    rho = -(24*(15*rho_I + m_xx_I*rho_I + 6*m_xy_I*rho_I + 4*m_yy_I*rho_I - 6*m_yz_I*rho_I + m_zz_I*rho_I - omegaVar*m_xx_I*rho_I - 6*omegaVar*m_xy_I*rho_I - 4*omegaVar*m_yy_I*rho_I + 6*omegaVar*m_yz_I*rho_I - omegaVar*m_zz_I*rho_I))/(12*ux0 - 12*omegaVar + 24*uy0 - 12*uz0 + 18*omegaVar*ux0 + 36*omegaVar*uy0 - 18*omegaVar*uz0 + 15*omegaVar*ux0*ux0 + 60*omegaVar*uy0*uy0 + 15*omegaVar*uz0*uz0 + 90*omegaVar*ux0*uy0 - 90*omegaVar*uy0*uz0 - 328);

    m_xx_t45 = (2*(rho + 75*m_xx_I*rho_I + 18*m_xy_I*rho_I + 3*m_yy_I*rho_I + 9*m_yz_I*rho_I - 6*m_zz_I*rho_I - 6*rho*ux0 - 3*rho*uy0 - 3*rho*uz0))/(135*rho);
    m_xy_t90 = (2*rho + 6*m_xx_I*rho_I + 63*m_xy_I*rho_I + 6*m_yy_I*rho_I - 3*m_zz_I*rho_I - 6*rho*ux0 - 6*rho*uy0)/(45*rho);
    m_xz_t90 = (m_xz_I*rho_I)/rho;
    m_yy_t45 = (8*rho + 6*m_xx_I*rho_I + 36*m_xy_I*rho_I + 159*m_yy_I*rho_I - 36*m_yz_I*rho_I + 6*m_zz_I*rho_I - 12*rho*ux0 - 24*rho*uy0 + 12*rho*uz0)/(135*rho);
    m_yz_t90 = -(2*rho - 3*m_xx_I*rho_I + 6*m_yy_I*rho_I - 63*m_yz_I*rho_I + 6*m_zz_I*rho_I - 6*rho*uy0 + 6*rho*uz0)/(45*rho);
    m_zz_t45 = (2*(rho - 6*m_xx_I*rho_I - 9*m_xy_I*rho_I + 3*m_yy_I*rho_I - 18*m_yz_I*rho_I + 75*m_zz_I*rho_I + 3*rho*ux0 - 3*rho*uy0 + 6*rho*uz0))/(135*rho);

    rhoVar = rho;

    break;
case 0b0000010110001101:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17]);
    m_xy_I = inv_rho_I * (pop[7] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17]);

    rho = -(24*(5*rho_I + m_xx_I*rho_I + 2*m_xy_I*rho_I + 2*m_xz_I*rho_I + m_yy_I*rho_I - 2*m_yz_I*rho_I + m_zz_I*rho_I - omegaVar*m_xx_I*rho_I - 2*omegaVar*m_xy_I*rho_I - 2*omegaVar*m_xz_I*rho_I - omegaVar*m_yy_I*rho_I + 2*omegaVar*m_yz_I*rho_I - omegaVar*m_zz_I*rho_I))/(8*uy0 - 6*omegaVar - 8*uz0 + 12*omegaVar*uy0 - 12*omegaVar*uz0 + 15*omegaVar*ux0*ux0 + 15*omegaVar*uy0*uy0 + 15*omegaVar*uz0*uz0 + 30*omegaVar*ux0*uy0 + 30*omegaVar*ux0*uz0 - 30*omegaVar*uy0*uz0 - 104);

    m_xx_t45 = (2*(rho + 27*m_xx_I*rho_I + 6*m_xy_I*rho_I + 6*m_xz_I*rho_I + 3*m_yz_I*rho_I - rho*uy0 + rho*uz0))/(45*rho);
    m_xy_t90 = (2*rho + 6*m_xx_I*rho_I + 63*m_xy_I*rho_I + 6*m_yy_I*rho_I - 3*m_zz_I*rho_I - 6*rho*ux0 - 6*rho*uy0)/(45*rho);
    m_xz_t90 = (2*rho + 6*m_xx_I*rho_I + 63*m_xz_I*rho_I - 3*m_yy_I*rho_I + 6*m_zz_I*rho_I + 6*rho*ux0 + 6*rho*uz0)/(45*rho);
    m_yy_t45 = (2*(rho + 6*m_xy_I*rho_I - 3*m_xz_I*rho_I + 27*m_yy_I*rho_I - 6*m_yz_I*rho_I - 3*rho*ux0 - 4*rho*uy0 + rho*uz0))/(45*rho);
    m_yz_t90 = -(2*rho - 3*m_xx_I*rho_I + 6*m_yy_I*rho_I - 63*m_yz_I*rho_I + 6*m_zz_I*rho_I - 6*rho*uy0 + 6*rho*uz0)/(45*rho);
    m_zz_t45 = (2*(rho - 3*m_xy_I*rho_I + 6*m_xz_I*rho_I - 6*m_yz_I*rho_I + 27*m_zz_I*rho_I + 3*rho*ux0 - rho*uy0 + 4*rho*uz0))/(45*rho);

    rhoVar = rho;

    break;
case 0b0000010110001110:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[17];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] - (1.0/3.0)*pop[17]);
    m_xy_I = inv_rho_I * (pop[7] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] + (2.0/3.0)*pop[17]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[17]);

    rho = -(24*(5*rho_I + m_xx_I*rho_I + 2*m_xy_I*rho_I - 2*m_xz_I*rho_I + m_yy_I*rho_I - 2*m_yz_I*rho_I + m_zz_I*rho_I - omegaVar*m_xx_I*rho_I - 2*omegaVar*m_xy_I*rho_I + 2*omegaVar*m_xz_I*rho_I - omegaVar*m_yy_I*rho_I + 2*omegaVar*m_yz_I*rho_I - omegaVar*m_zz_I*rho_I))/(8*ux0 - 6*omegaVar + 8*uy0 - 8*uz0 + 12*omegaVar*ux0 + 12*omegaVar*uy0 - 12*omegaVar*uz0 + 15*omegaVar*ux0*ux0 + 15*omegaVar*uy0*uy0 + 15*omegaVar*uz0*uz0 + 30*omegaVar*ux0*uy0 - 30*omegaVar*ux0*uz0 - 30*omegaVar*uy0*uz0 - 104);

    m_xx_t45 = (2*(rho + 27*m_xx_I*rho_I + 6*m_xy_I*rho_I - 6*m_xz_I*rho_I + 3*m_yz_I*rho_I - 4*rho*ux0 - rho*uy0 + rho*uz0))/(45*rho);
    m_xy_t90 = (2*rho + 6*m_xx_I*rho_I + 63*m_xy_I*rho_I + 6*m_yy_I*rho_I - 3*m_zz_I*rho_I - 6*rho*ux0 - 6*rho*uy0)/(45*rho);
    m_xz_t90 = -(2*rho + 6*m_xx_I*rho_I - 63*m_xz_I*rho_I - 3*m_yy_I*rho_I + 6*m_zz_I*rho_I - 6*rho*ux0 + 6*rho*uz0)/(45*rho);
    m_yy_t45 = (2*(rho + 6*m_xy_I*rho_I + 3*m_xz_I*rho_I + 27*m_yy_I*rho_I - 6*m_yz_I*rho_I - rho*ux0 - 4*rho*uy0 + rho*uz0))/(45*rho);
    m_yz_t90 = -(2*rho - 3*m_xx_I*rho_I + 6*m_yy_I*rho_I - 63*m_yz_I*rho_I + 6*m_zz_I*rho_I - 6*rho*uy0 + 6*rho*uz0)/(45*rho);
    m_zz_t45 = -(2*(3*m_xy_I*rho_I - rho + 6*m_xz_I*rho_I + 6*m_yz_I*rho_I - 27*m_zz_I*rho_I + rho*ux0 + rho*uy0 - 4*rho*uz0))/(45*rho);

    rhoVar = rho;

    break;
case 0b0000010110001111:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[6] + pop[7] + pop[10] + pop[12] + pop[13] + pop[14] + pop[15] + pop[17];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] - (1.0/3.0)*pop[17]);
    m_xy_I = inv_rho_I * (pop[7] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[10] - pop[15]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] + (2.0/3.0)*pop[17]);
    m_yz_I = inv_rho_I * (pop[12] - pop[17]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[17]);

    rho = -(72*(107*rho_I + 12*m_xx_I*rho_I + 33*m_xy_I*rho_I + 12*m_yy_I*rho_I + 72*m_zz_I*rho_I - 12*omegaVar*m_xx_I*rho_I - 33*omegaVar*m_xy_I*rho_I - 12*omegaVar*m_yy_I*rho_I - 72*omegaVar*m_zz_I*rho_I))/(492*ux0 - 626*omegaVar + 492*uy0 - 2124*uz0 + 150*omegaVar*ux0 + 150*omegaVar*uy0 - 1728*omegaVar*uz0 + 642*omegaVar*ux0*ux0 + 642*omegaVar*uy0*uy0 + 3531*omegaVar*uz0*uz0 + 1926*omegaVar*ux0*uy0 - 5580);

    m_xx_t45 = -(4*(m_yy_I*rho_I - 106*m_xx_I*rho_I - 24*m_xy_I*rho_I - 2*rho + 6*m_zz_I*rho_I + 8*rho*ux0 + 8*rho*uy0 + 2*rho*uz0))/(321*rho);
    m_xy_t90 = (11*rho + 48*m_xx_I*rho_I + 453*m_xy_I*rho_I + 48*m_yy_I*rho_I - 33*m_zz_I*rho_I - 44*rho*ux0 - 44*rho*uy0 - 11*rho*uz0)/(321*rho);
    m_xz_t90 = (6*m_xz_I*rho_I + rho*ux0)/(3*rho);
    m_yy_t45 = -(4*(m_xx_I*rho_I - 2*rho - 24*m_xy_I*rho_I - 106*m_yy_I*rho_I + 6*m_zz_I*rho_I + 8*rho*ux0 + 8*rho*uy0 + 2*rho*uz0))/(321*rho);
    m_yz_t90 = (6*m_yz_I*rho_I + rho*uy0)/(3*rho);
    m_zz_t45 = (2*(24*rho - 12*m_xx_I*rho_I - 33*m_xy_I*rho_I - 12*m_yy_I*rho_I + 249*m_zz_I*rho_I + 11*rho*ux0 + 11*rho*uy0 + 83*rho*uz0))/(321*rho);

    rhoVar = rho;

    break;
case 0b0000010110010000:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = rho_I;

    m_xx_t45 = (m_xx_I*rho_I)/rho;
    m_xy_t90 = (m_xy_I*rho_I)/rho;
    m_xz_t90 = (m_xz_I*rho_I)/rho;
    m_yy_t45 = (m_yy_I*rho_I)/rho;
    m_yz_t90 = (m_yz_I*rho_I)/rho;
    m_zz_t45 = (m_zz_I*rho_I)/rho;

    rhoVar = rho;

    break;
case 0b0000010110010001:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (24*(15*rho_I + 2*m_xx_I*rho_I + 6*m_xy_I*rho_I + 2*m_yy_I*rho_I - m_zz_I*rho_I - 2*omegaVar*m_xx_I*rho_I - 6*omegaVar*m_xy_I*rho_I - 2*omegaVar*m_yy_I*rho_I + omegaVar*m_zz_I*rho_I))/(6*omegaVar + 12*ux0 + 12*uy0 + 18*omegaVar*ux0 + 18*omegaVar*uy0 - 30*omegaVar*ux0*ux0 - 30*omegaVar*uy0*uy0 + 15*omegaVar*uz0*uz0 - 90*omegaVar*ux0*uy0 + 344);

    m_xx_t45 = (4*rho + 147*m_xx_I*rho_I + 36*m_xy_I*rho_I + 12*m_yy_I*rho_I - 6*m_zz_I*rho_I + 12*rho*ux0 + 12*rho*uy0)/(135*rho);
    m_xy_t90 = (2*rho + 6*m_xx_I*rho_I + 63*m_xy_I*rho_I + 6*m_yy_I*rho_I - 3*m_zz_I*rho_I + 6*rho*ux0 + 6*rho*uy0)/(45*rho);
    m_xz_t90 = (m_xz_I*rho_I)/rho;
    m_yy_t45 = (4*rho + 12*m_xx_I*rho_I + 36*m_xy_I*rho_I + 147*m_yy_I*rho_I - 6*m_zz_I*rho_I + 12*rho*ux0 + 12*rho*uy0)/(135*rho);
    m_yz_t90 = (m_yz_I*rho_I)/rho;
    m_zz_t45 = -(2*(rho + 3*m_xx_I*rho_I + 9*m_xy_I*rho_I + 3*m_yy_I*rho_I - 69*m_zz_I*rho_I + 3*rho*ux0 + 3*rho*uy0))/(135*rho);

    rhoVar = rho;

    break;
case 0b0000010110010010:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = rho_I;

    m_xx_t45 = (m_xx_I*rho_I)/rho;
    m_xy_t90 = (m_xy_I*rho_I)/rho;
    m_xz_t90 = (m_xz_I*rho_I)/rho;
    m_yy_t45 = (m_yy_I*rho_I)/rho;
    m_yz_t90 = (m_yz_I*rho_I)/rho;
    m_zz_t45 = (m_zz_I*rho_I)/rho;

    rhoVar = rho;

    break;
case 0b0000010110010011:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[8] + pop[9] + pop[10] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (24*(15*rho_I + m_xx_I*rho_I + 6*m_xy_I*rho_I + 4*m_yy_I*rho_I + 6*m_yz_I*rho_I + m_zz_I*rho_I - omegaVar*m_xx_I*rho_I - 6*omegaVar*m_xy_I*rho_I - 4*omegaVar*m_yy_I*rho_I - 6*omegaVar*m_yz_I*rho_I - omegaVar*m_zz_I*rho_I))/(12*omegaVar + 12*ux0 + 24*uy0 + 12*uz0 + 18*omegaVar*ux0 + 36*omegaVar*uy0 + 18*omegaVar*uz0 - 15*omegaVar*ux0*ux0 - 60*omegaVar*uy0*uy0 - 15*omegaVar*uz0*uz0 - 90*omegaVar*ux0*uy0 - 90*omegaVar*uy0*uz0 + 328);

    m_xx_t45 = (2*(rho + 75*m_xx_I*rho_I + 18*m_xy_I*rho_I + 3*m_yy_I*rho_I - 9*m_yz_I*rho_I - 6*m_zz_I*rho_I + 6*rho*ux0 + 3*rho*uy0 - 3*rho*uz0))/(135*rho);
    m_xy_t90 = (2*rho + 6*m_xx_I*rho_I + 63*m_xy_I*rho_I + 6*m_yy_I*rho_I - 3*m_zz_I*rho_I + 6*rho*ux0 + 6*rho*uy0)/(45*rho);
    m_xz_t90 = (m_xz_I*rho_I)/rho;
    m_yy_t45 = (8*rho + 6*m_xx_I*rho_I + 36*m_xy_I*rho_I + 159*m_yy_I*rho_I + 36*m_yz_I*rho_I + 6*m_zz_I*rho_I + 12*rho*ux0 + 24*rho*uy0 + 12*rho*uz0)/(135*rho);
    m_yz_t90 = (2*rho - 3*m_xx_I*rho_I + 6*m_yy_I*rho_I + 63*m_yz_I*rho_I + 6*m_zz_I*rho_I + 6*rho*uy0 + 6*rho*uz0)/(45*rho);
    m_zz_t45 = (2*(rho - 6*m_xx_I*rho_I - 9*m_xy_I*rho_I + 3*m_yy_I*rho_I + 18*m_yz_I*rho_I + 75*m_zz_I*rho_I - 3*rho*ux0 + 3*rho*uy0 + 6*rho*uz0))/(135*rho);

    rhoVar = rho;

    break;
case 0b0000010110010100:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = rho_I;

    m_xx_t45 = (m_xx_I*rho_I)/rho;
    m_xy_t90 = (m_xy_I*rho_I)/rho;
    m_xz_t90 = (m_xz_I*rho_I)/rho;
    m_yy_t45 = (m_yy_I*rho_I)/rho;
    m_yz_t90 = (m_yz_I*rho_I)/rho;
    m_zz_t45 = (m_zz_I*rho_I)/rho;

    rhoVar = rho;

    break;
case 0b0000010110010101:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[8] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (24*(15*rho_I + 4*m_xx_I*rho_I + 6*m_xy_I*rho_I + 6*m_xz_I*rho_I + m_yy_I*rho_I + m_zz_I*rho_I - 4*omegaVar*m_xx_I*rho_I - 6*omegaVar*m_xy_I*rho_I - 6*omegaVar*m_xz_I*rho_I - omegaVar*m_yy_I*rho_I - omegaVar*m_zz_I*rho_I))/(12*omegaVar + 24*ux0 + 12*uy0 + 12*uz0 + 36*omegaVar*ux0 + 18*omegaVar*uy0 + 18*omegaVar*uz0 - 60*omegaVar*ux0*ux0 - 15*omegaVar*uy0*uy0 - 15*omegaVar*uz0*uz0 - 90*omegaVar*ux0*uy0 - 90*omegaVar*ux0*uz0 + 328);

    m_xx_t45 = (8*rho + 159*m_xx_I*rho_I + 36*m_xy_I*rho_I + 36*m_xz_I*rho_I + 6*m_yy_I*rho_I + 6*m_zz_I*rho_I + 24*rho*ux0 + 12*rho*uy0 + 12*rho*uz0)/(135*rho);
    m_xy_t90 = (2*rho + 6*m_xx_I*rho_I + 63*m_xy_I*rho_I + 6*m_yy_I*rho_I - 3*m_zz_I*rho_I + 6*rho*ux0 + 6*rho*uy0)/(45*rho);
    m_xz_t90 = (2*rho + 6*m_xx_I*rho_I + 63*m_xz_I*rho_I - 3*m_yy_I*rho_I + 6*m_zz_I*rho_I + 6*rho*ux0 + 6*rho*uz0)/(45*rho);
    m_yy_t45 = (2*(rho + 3*m_xx_I*rho_I + 18*m_xy_I*rho_I - 9*m_xz_I*rho_I + 75*m_yy_I*rho_I - 6*m_zz_I*rho_I + 3*rho*ux0 + 6*rho*uy0 - 3*rho*uz0))/(135*rho);
    m_yz_t90 = (m_yz_I*rho_I)/rho;
    m_zz_t45 = (2*(rho + 3*m_xx_I*rho_I - 9*m_xy_I*rho_I + 18*m_xz_I*rho_I - 6*m_yy_I*rho_I + 75*m_zz_I*rho_I + 3*rho*ux0 - 3*rho*uy0 + 6*rho*uz0))/(135*rho);

    rhoVar = rho;

    break;
case 0b0000010110010110:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = rho_I;

    m_xx_t45 = (m_xx_I*rho_I)/rho;
    m_xy_t90 = (m_xy_I*rho_I)/rho;
    m_xz_t90 = (m_xz_I*rho_I)/rho;
    m_yy_t45 = (m_yy_I*rho_I)/rho;
    m_yz_t90 = (m_yz_I*rho_I)/rho;
    m_zz_t45 = (m_zz_I*rho_I)/rho;

    rhoVar = rho;

    break;
case 0b0000010110010111:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[8] + pop[10] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (24*(5*rho_I + m_xx_I*rho_I + 2*m_xy_I*rho_I + 2*m_xz_I*rho_I + m_yy_I*rho_I + 2*m_yz_I*rho_I + m_zz_I*rho_I - omegaVar*m_xx_I*rho_I - 2*omegaVar*m_xy_I*rho_I - 2*omegaVar*m_xz_I*rho_I - omegaVar*m_yy_I*rho_I - 2*omegaVar*m_yz_I*rho_I - omegaVar*m_zz_I*rho_I))/(6*omegaVar + 8*ux0 + 8*uy0 + 8*uz0 + 12*omegaVar*ux0 + 12*omegaVar*uy0 + 12*omegaVar*uz0 - 15*omegaVar*ux0*ux0 - 15*omegaVar*uy0*uy0 - 15*omegaVar*uz0*uz0 - 30*omegaVar*ux0*uy0 - 30*omegaVar*ux0*uz0 - 30*omegaVar*uy0*uz0 + 104);

    m_xx_t45 = (2*(rho + 27*m_xx_I*rho_I + 6*m_xy_I*rho_I + 6*m_xz_I*rho_I - 3*m_yz_I*rho_I + 4*rho*ux0 + rho*uy0 + rho*uz0))/(45*rho);
    m_xy_t90 = (2*rho + 6*m_xx_I*rho_I + 63*m_xy_I*rho_I + 6*m_yy_I*rho_I - 3*m_zz_I*rho_I + 6*rho*ux0 + 6*rho*uy0)/(45*rho);
    m_xz_t90 = (2*rho + 6*m_xx_I*rho_I + 63*m_xz_I*rho_I - 3*m_yy_I*rho_I + 6*m_zz_I*rho_I + 6*rho*ux0 + 6*rho*uz0)/(45*rho);
    m_yy_t45 = (2*(rho + 6*m_xy_I*rho_I - 3*m_xz_I*rho_I + 27*m_yy_I*rho_I + 6*m_yz_I*rho_I + rho*ux0 + 4*rho*uy0 + rho*uz0))/(45*rho);
    m_yz_t90 = (2*rho - 3*m_xx_I*rho_I + 6*m_yy_I*rho_I + 63*m_yz_I*rho_I + 6*m_zz_I*rho_I + 6*rho*uy0 + 6*rho*uz0)/(45*rho);
    m_zz_t45 = (2*(rho - 3*m_xy_I*rho_I + 6*m_xz_I*rho_I + 6*m_yz_I*rho_I + 27*m_zz_I*rho_I + rho*ux0 + rho*uy0 + 4*rho*uz0))/(45*rho);

    rhoVar = rho;

    break;
case 0b0000010110011000:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = -(24*(15*rho_I + 2*m_xx_I*rho_I + 6*m_xy_I*rho_I + 2*m_yy_I*rho_I - m_zz_I*rho_I - 2*omegaVar*m_xx_I*rho_I - 6*omegaVar*m_xy_I*rho_I - 2*omegaVar*m_yy_I*rho_I + omegaVar*m_zz_I*rho_I))/(12*ux0 - 6*omegaVar + 12*uy0 + 18*omegaVar*ux0 + 18*omegaVar*uy0 + 30*omegaVar*ux0*ux0 + 30*omegaVar*uy0*uy0 - 15*omegaVar*uz0*uz0 + 90*omegaVar*ux0*uy0 - 344);

    m_xx_t45 = (4*rho + 147*m_xx_I*rho_I + 36*m_xy_I*rho_I + 12*m_yy_I*rho_I - 6*m_zz_I*rho_I - 12*rho*ux0 - 12*rho*uy0)/(135*rho);
    m_xy_t90 = (2*rho + 6*m_xx_I*rho_I + 63*m_xy_I*rho_I + 6*m_yy_I*rho_I - 3*m_zz_I*rho_I - 6*rho*ux0 - 6*rho*uy0)/(45*rho);
    m_xz_t90 = (m_xz_I*rho_I)/rho;
    m_yy_t45 = (4*rho + 12*m_xx_I*rho_I + 36*m_xy_I*rho_I + 147*m_yy_I*rho_I - 6*m_zz_I*rho_I - 12*rho*ux0 - 12*rho*uy0)/(135*rho);
    m_yz_t90 = (m_yz_I*rho_I)/rho;
    m_zz_t45 = -(2*(rho + 3*m_xx_I*rho_I + 9*m_xy_I*rho_I + 3*m_yy_I*rho_I - 69*m_zz_I*rho_I - 3*rho*ux0 - 3*rho*uy0))/(135*rho);

    rhoVar = rho;

    break;
case 0b0000010110011001:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (-pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (12*(3*rho_I + 2*m_xx_I*rho_I + 6*m_xy_I*rho_I + 2*m_yy_I*rho_I - m_zz_I*rho_I - 2*omegaVar*m_xx_I*rho_I - 6*omegaVar*m_xy_I*rho_I - 2*omegaVar*m_yy_I*rho_I + omegaVar*m_zz_I*rho_I))/(6*omegaVar - 6*omegaVar*ux0*ux0 - 6*omegaVar*uy0*uy0 + 3*omegaVar*uz0*uz0 - 18*omegaVar*ux0*uy0 + 28);

    m_xx_t45 = (4*rho + 39*m_xx_I*rho_I + 36*m_xy_I*rho_I + 12*m_yy_I*rho_I - 6*m_zz_I*rho_I)/(27*rho);
    m_xy_t90 = (2*rho + 6*m_xx_I*rho_I + 27*m_xy_I*rho_I + 6*m_yy_I*rho_I - 3*m_zz_I*rho_I)/(9*rho);
    m_xz_t90 = (m_xz_I*rho_I)/rho;
    m_yy_t45 = (4*rho + 12*m_xx_I*rho_I + 36*m_xy_I*rho_I + 39*m_yy_I*rho_I - 6*m_zz_I*rho_I)/(27*rho);
    m_yz_t90 = (m_yz_I*rho_I)/rho;
    m_zz_t45 = -(2*(rho + 3*m_xx_I*rho_I + 9*m_xy_I*rho_I + 3*m_yy_I*rho_I - 15*m_zz_I*rho_I))/(27*rho);

    rhoVar = rho;

    break;
case 0b0000010110011010:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = -(24*(15*rho_I + 4*m_xx_I*rho_I + 6*m_xy_I*rho_I - 6*m_xz_I*rho_I + m_yy_I*rho_I + m_zz_I*rho_I - 4*omegaVar*m_xx_I*rho_I - 6*omegaVar*m_xy_I*rho_I + 6*omegaVar*m_xz_I*rho_I - omegaVar*m_yy_I*rho_I - omegaVar*m_zz_I*rho_I))/(24*ux0 - 12*omegaVar + 12*uy0 - 12*uz0 + 36*omegaVar*ux0 + 18*omegaVar*uy0 - 18*omegaVar*uz0 + 60*omegaVar*ux0*ux0 + 15*omegaVar*uy0*uy0 + 15*omegaVar*uz0*uz0 + 90*omegaVar*ux0*uy0 - 90*omegaVar*ux0*uz0 - 328);

    m_xx_t45 = (8*rho + 159*m_xx_I*rho_I + 36*m_xy_I*rho_I - 36*m_xz_I*rho_I + 6*m_yy_I*rho_I + 6*m_zz_I*rho_I - 24*rho*ux0 - 12*rho*uy0 + 12*rho*uz0)/(135*rho);
    m_xy_t90 = (2*rho + 6*m_xx_I*rho_I + 63*m_xy_I*rho_I + 6*m_yy_I*rho_I - 3*m_zz_I*rho_I - 6*rho*ux0 - 6*rho*uy0)/(45*rho);
    m_xz_t90 = -(2*rho + 6*m_xx_I*rho_I - 63*m_xz_I*rho_I - 3*m_yy_I*rho_I + 6*m_zz_I*rho_I - 6*rho*ux0 + 6*rho*uz0)/(45*rho);
    m_yy_t45 = (2*(rho + 3*m_xx_I*rho_I + 18*m_xy_I*rho_I + 9*m_xz_I*rho_I + 75*m_yy_I*rho_I - 6*m_zz_I*rho_I - 3*rho*ux0 - 6*rho*uy0 - 3*rho*uz0))/(135*rho);
    m_yz_t90 = (m_yz_I*rho_I)/rho;
    m_zz_t45 = (2*(rho + 3*m_xx_I*rho_I - 9*m_xy_I*rho_I - 18*m_xz_I*rho_I - 6*m_yy_I*rho_I + 75*m_zz_I*rho_I - 3*rho*ux0 + 3*rho*uy0 + 6*rho*uz0))/(135*rho);

    rhoVar = rho;

    break;
case 0b0000010110011011:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[9] + pop[10] + pop[12] + pop[13] + pop[14] + pop[15] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (-pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (24*(15*rho_I + 11*m_xx_I*rho_I + 30*m_xy_I*rho_I - 6*m_xz_I*rho_I + 11*m_yy_I*rho_I + 6*m_yz_I*rho_I - m_zz_I*rho_I - 11*omegaVar*m_xx_I*rho_I - 30*omegaVar*m_xy_I*rho_I + 6*omegaVar*m_xz_I*rho_I - 11*omegaVar*m_yy_I*rho_I - 6*omegaVar*m_yz_I*rho_I + omegaVar*m_zz_I*rho_I))/(72*omegaVar - 12*ux0 + 12*uy0 + 24*uz0 - 18*omegaVar*ux0 + 18*omegaVar*uy0 + 36*omegaVar*uz0 - 75*omegaVar*ux0*ux0 - 75*omegaVar*uy0*uy0 - 30*omegaVar*uz0*uz0 - 180*omegaVar*ux0*uy0 + 90*omegaVar*ux0*uz0 - 90*omegaVar*uy0*uz0 + 248);

    m_xx_t45 = (2*(11*rho + 105*m_xx_I*rho_I + 90*m_xy_I*rho_I - 18*m_xz_I*rho_I + 24*m_yy_I*rho_I - 9*m_yz_I*rho_I - 12*m_zz_I*rho_I - 6*rho*ux0 - 3*rho*uy0 + 3*rho*uz0))/(135*rho);
    m_xy_t90 = (2*rho + 6*m_xx_I*rho_I + 27*m_xy_I*rho_I + 6*m_yy_I*rho_I - 3*m_zz_I*rho_I)/(9*rho);
    m_xz_t90 = -(2*rho + 6*m_xx_I*rho_I - 63*m_xz_I*rho_I - 3*m_yy_I*rho_I + 6*m_zz_I*rho_I - 6*rho*ux0 + 6*rho*uz0)/(45*rho);
    m_yy_t45 = (2*(11*rho + 24*m_xx_I*rho_I + 90*m_xy_I*rho_I + 9*m_xz_I*rho_I + 105*m_yy_I*rho_I + 18*m_yz_I*rho_I - 12*m_zz_I*rho_I + 3*rho*ux0 + 6*rho*uy0 + 3*rho*uz0))/(135*rho);
    m_yz_t90 = (2*rho - 3*m_xx_I*rho_I + 6*m_yy_I*rho_I + 63*m_yz_I*rho_I + 6*m_zz_I*rho_I + 6*rho*uy0 + 6*rho*uz0)/(45*rho);
    m_zz_t45 = -(2*(rho + 12*m_xx_I*rho_I + 45*m_xy_I*rho_I + 18*m_xz_I*rho_I + 12*m_yy_I*rho_I - 18*m_yz_I*rho_I - 87*m_zz_I*rho_I + 6*rho*ux0 - 6*rho*uy0 - 12*rho*uz0))/(135*rho);

    rhoVar = rho;

    break;
case 0b0000010110011100:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17]);
    m_xy_I = inv_rho_I * (pop[7] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17]);

    rho = -(24*(15*rho_I + m_xx_I*rho_I + 6*m_xy_I*rho_I + 4*m_yy_I*rho_I - 6*m_yz_I*rho_I + m_zz_I*rho_I - omegaVar*m_xx_I*rho_I - 6*omegaVar*m_xy_I*rho_I - 4*omegaVar*m_yy_I*rho_I + 6*omegaVar*m_yz_I*rho_I - omegaVar*m_zz_I*rho_I))/(12*ux0 - 12*omegaVar + 24*uy0 - 12*uz0 + 18*omegaVar*ux0 + 36*omegaVar*uy0 - 18*omegaVar*uz0 + 15*omegaVar*ux0*ux0 + 60*omegaVar*uy0*uy0 + 15*omegaVar*uz0*uz0 + 90*omegaVar*ux0*uy0 - 90*omegaVar*uy0*uz0 - 328);

    m_xx_t45 = (2*(rho + 75*m_xx_I*rho_I + 18*m_xy_I*rho_I + 3*m_yy_I*rho_I + 9*m_yz_I*rho_I - 6*m_zz_I*rho_I - 6*rho*ux0 - 3*rho*uy0 - 3*rho*uz0))/(135*rho);
    m_xy_t90 = (2*rho + 6*m_xx_I*rho_I + 63*m_xy_I*rho_I + 6*m_yy_I*rho_I - 3*m_zz_I*rho_I - 6*rho*ux0 - 6*rho*uy0)/(45*rho);
    m_xz_t90 = (m_xz_I*rho_I)/rho;
    m_yy_t45 = (8*rho + 6*m_xx_I*rho_I + 36*m_xy_I*rho_I + 159*m_yy_I*rho_I - 36*m_yz_I*rho_I + 6*m_zz_I*rho_I - 12*rho*ux0 - 24*rho*uy0 + 12*rho*uz0)/(135*rho);
    m_yz_t90 = -(2*rho - 3*m_xx_I*rho_I + 6*m_yy_I*rho_I - 63*m_yz_I*rho_I + 6*m_zz_I*rho_I - 6*rho*uy0 + 6*rho*uz0)/(45*rho);
    m_zz_t45 = (2*(rho - 6*m_xx_I*rho_I - 9*m_xy_I*rho_I + 3*m_yy_I*rho_I - 18*m_yz_I*rho_I + 75*m_zz_I*rho_I + 3*rho*ux0 - 3*rho*uy0 + 6*rho*uz0))/(135*rho);

    rhoVar = rho;

    break;
case 0b0000010110011101:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17]);
    m_xy_I = inv_rho_I * (-pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17]);

    rho = (24*(15*rho_I + 11*m_xx_I*rho_I + 30*m_xy_I*rho_I + 6*m_xz_I*rho_I + 11*m_yy_I*rho_I - 6*m_yz_I*rho_I - m_zz_I*rho_I - 11*omegaVar*m_xx_I*rho_I - 30*omegaVar*m_xy_I*rho_I - 6*omegaVar*m_xz_I*rho_I - 11*omegaVar*m_yy_I*rho_I + 6*omegaVar*m_yz_I*rho_I + omegaVar*m_zz_I*rho_I))/(72*omegaVar + 12*ux0 - 12*uy0 + 24*uz0 + 18*omegaVar*ux0 - 18*omegaVar*uy0 + 36*omegaVar*uz0 - 75*omegaVar*ux0*ux0 - 75*omegaVar*uy0*uy0 - 30*omegaVar*uz0*uz0 - 180*omegaVar*ux0*uy0 - 90*omegaVar*ux0*uz0 + 90*omegaVar*uy0*uz0 + 248);

    m_xx_t45 = (2*(11*rho + 105*m_xx_I*rho_I + 90*m_xy_I*rho_I + 18*m_xz_I*rho_I + 24*m_yy_I*rho_I + 9*m_yz_I*rho_I - 12*m_zz_I*rho_I + 6*rho*ux0 + 3*rho*uy0 + 3*rho*uz0))/(135*rho);
    m_xy_t90 = (2*rho + 6*m_xx_I*rho_I + 27*m_xy_I*rho_I + 6*m_yy_I*rho_I - 3*m_zz_I*rho_I)/(9*rho);
    m_xz_t90 = (2*rho + 6*m_xx_I*rho_I + 63*m_xz_I*rho_I - 3*m_yy_I*rho_I + 6*m_zz_I*rho_I + 6*rho*ux0 + 6*rho*uz0)/(45*rho);
    m_yy_t45 = (2*(11*rho + 24*m_xx_I*rho_I + 90*m_xy_I*rho_I - 9*m_xz_I*rho_I + 105*m_yy_I*rho_I - 18*m_yz_I*rho_I - 12*m_zz_I*rho_I - 3*rho*ux0 - 6*rho*uy0 + 3*rho*uz0))/(135*rho);
    m_yz_t90 = -(2*rho - 3*m_xx_I*rho_I + 6*m_yy_I*rho_I - 63*m_yz_I*rho_I + 6*m_zz_I*rho_I - 6*rho*uy0 + 6*rho*uz0)/(45*rho);
    m_zz_t45 = -(2*(rho + 12*m_xx_I*rho_I + 45*m_xy_I*rho_I - 18*m_xz_I*rho_I + 12*m_yy_I*rho_I + 18*m_yz_I*rho_I - 87*m_zz_I*rho_I - 6*rho*ux0 + 6*rho*uy0 - 12*rho*uz0))/(135*rho);

    rhoVar = rho;

    break;
case 0b0000010110011110:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[17];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] - (1.0/3.0)*pop[17]);
    m_xy_I = inv_rho_I * (pop[7] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] + (2.0/3.0)*pop[17]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[17]);

    rho = -(24*(5*rho_I + m_xx_I*rho_I + 2*m_xy_I*rho_I - 2*m_xz_I*rho_I + m_yy_I*rho_I - 2*m_yz_I*rho_I + m_zz_I*rho_I - omegaVar*m_xx_I*rho_I - 2*omegaVar*m_xy_I*rho_I + 2*omegaVar*m_xz_I*rho_I - omegaVar*m_yy_I*rho_I + 2*omegaVar*m_yz_I*rho_I - omegaVar*m_zz_I*rho_I))/(8*ux0 - 6*omegaVar + 8*uy0 - 8*uz0 + 12*omegaVar*ux0 + 12*omegaVar*uy0 - 12*omegaVar*uz0 + 15*omegaVar*ux0*ux0 + 15*omegaVar*uy0*uy0 + 15*omegaVar*uz0*uz0 + 30*omegaVar*ux0*uy0 - 30*omegaVar*ux0*uz0 - 30*omegaVar*uy0*uz0 - 104);

    m_xx_t45 = (2*(rho + 27*m_xx_I*rho_I + 6*m_xy_I*rho_I - 6*m_xz_I*rho_I + 3*m_yz_I*rho_I - 4*rho*ux0 - rho*uy0 + rho*uz0))/(45*rho);
    m_xy_t90 = (2*rho + 6*m_xx_I*rho_I + 63*m_xy_I*rho_I + 6*m_yy_I*rho_I - 3*m_zz_I*rho_I - 6*rho*ux0 - 6*rho*uy0)/(45*rho);
    m_xz_t90 = -(2*rho + 6*m_xx_I*rho_I - 63*m_xz_I*rho_I - 3*m_yy_I*rho_I + 6*m_zz_I*rho_I - 6*rho*ux0 + 6*rho*uz0)/(45*rho);
    m_yy_t45 = (2*(rho + 6*m_xy_I*rho_I + 3*m_xz_I*rho_I + 27*m_yy_I*rho_I - 6*m_yz_I*rho_I - rho*ux0 - 4*rho*uy0 + rho*uz0))/(45*rho);
    m_yz_t90 = -(2*rho - 3*m_xx_I*rho_I + 6*m_yy_I*rho_I - 63*m_yz_I*rho_I + 6*m_zz_I*rho_I - 6*rho*uy0 + 6*rho*uz0)/(45*rho);
    m_zz_t45 = -(2*(3*m_xy_I*rho_I - rho + 6*m_xz_I*rho_I + 6*m_yz_I*rho_I - 27*m_zz_I*rho_I + rho*ux0 + rho*uy0 - 4*rho*uz0))/(45*rho);

    rhoVar = rho;

    break;
case 0b0000010110011111:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[6] + pop[10] + pop[12] + pop[13] + pop[14] + pop[15] + pop[17];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] - (1.0/3.0)*pop[17]);
    m_xy_I = inv_rho_I * (-pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[10] - pop[15]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[6] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] + (2.0/3.0)*pop[17]);
    m_yz_I = inv_rho_I * (pop[12] - pop[17]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[6] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[17]);

    rho = (36*(19*rho_I + 12*m_xx_I*rho_I + 33*m_xy_I*rho_I + 12*m_yy_I*rho_I + 6*m_zz_I*rho_I - 12*omegaVar*m_xx_I*rho_I - 33*omegaVar*m_xy_I*rho_I - 12*omegaVar*m_yy_I*rho_I - 6*omegaVar*m_zz_I*rho_I))/(118*omegaVar + 270*uz0 + 72*omegaVar*uz0 - 114*omegaVar*ux0*ux0 - 114*omegaVar*uy0*uy0 - 285*omegaVar*uz0*uz0 - 342*omegaVar*ux0*uy0 + 414);

    m_xx_t45 = (4*(2*rho + 26*m_xx_I*rho_I + 24*m_xy_I*rho_I + 7*m_yy_I*rho_I - 6*m_zz_I*rho_I - 2*rho*uz0))/(57*rho);
    m_xy_t90 = (11*rho + 48*m_xx_I*rho_I + 189*m_xy_I*rho_I + 48*m_yy_I*rho_I - 33*m_zz_I*rho_I - 11*rho*uz0)/(57*rho);
    m_xz_t90 = (6*m_xz_I*rho_I + rho*ux0)/(3*rho);
    m_yy_t45 = (4*(2*rho + 7*m_xx_I*rho_I + 24*m_xy_I*rho_I + 26*m_yy_I*rho_I - 6*m_zz_I*rho_I - 2*rho*uz0))/(57*rho);
    m_yz_t90 = (6*m_yz_I*rho_I + rho*uy0)/(3*rho);
    m_zz_t45 = (2*(2*rho - 12*m_xx_I*rho_I - 33*m_xy_I*rho_I - 12*m_yy_I*rho_I + 51*m_zz_I*rho_I + 17*rho*uz0))/(57*rho);

    rhoVar = rho;

    break;
case 0b0000010110100000:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = -(24*(15*rho_I + 2*m_xx_I*rho_I + 6*m_xz_I*rho_I - m_yy_I*rho_I + 2*m_zz_I*rho_I - 2*omegaVar*m_xx_I*rho_I - 6*omegaVar*m_xz_I*rho_I + omegaVar*m_yy_I*rho_I - 2*omegaVar*m_zz_I*rho_I))/(12*ux0 - 6*omegaVar + 12*uz0 + 18*omegaVar*ux0 + 18*omegaVar*uz0 + 30*omegaVar*ux0*ux0 - 15*omegaVar*uy0*uy0 + 30*omegaVar*uz0*uz0 + 90*omegaVar*ux0*uz0 - 344);

    m_xx_t45 = (4*rho + 147*m_xx_I*rho_I + 36*m_xz_I*rho_I - 6*m_yy_I*rho_I + 12*m_zz_I*rho_I - 12*rho*ux0 - 12*rho*uz0)/(135*rho);
    m_xy_t90 = (m_xy_I*rho_I)/rho;
    m_xz_t90 = (2*rho + 6*m_xx_I*rho_I + 63*m_xz_I*rho_I - 3*m_yy_I*rho_I + 6*m_zz_I*rho_I - 6*rho*ux0 - 6*rho*uz0)/(45*rho);
    m_yy_t45 = -(2*(rho + 3*m_xx_I*rho_I + 9*m_xz_I*rho_I - 69*m_yy_I*rho_I + 3*m_zz_I*rho_I - 3*rho*ux0 - 3*rho*uz0))/(135*rho);
    m_yz_t90 = (m_yz_I*rho_I)/rho;
    m_zz_t45 = (4*rho + 12*m_xx_I*rho_I + 36*m_xz_I*rho_I - 6*m_yy_I*rho_I + 147*m_zz_I*rho_I - 12*rho*ux0 - 12*rho*uz0)/(135*rho);

    rhoVar = rho;

    break;
case 0b0000010110100001:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = -(24*(15*rho_I + 2*m_xx_I*rho_I + 6*m_xz_I*rho_I - m_yy_I*rho_I + 2*m_zz_I*rho_I - 2*omegaVar*m_xx_I*rho_I - 6*omegaVar*m_xz_I*rho_I + omegaVar*m_yy_I*rho_I - 2*omegaVar*m_zz_I*rho_I))/(12*ux0 - 6*omegaVar + 12*uz0 + 18*omegaVar*ux0 + 18*omegaVar*uz0 + 30*omegaVar*ux0*ux0 - 15*omegaVar*uy0*uy0 + 30*omegaVar*uz0*uz0 + 90*omegaVar*ux0*uz0 - 344);

    m_xx_t45 = (4*rho + 147*m_xx_I*rho_I + 36*m_xz_I*rho_I - 6*m_yy_I*rho_I + 12*m_zz_I*rho_I - 12*rho*ux0 - 12*rho*uz0)/(135*rho);
    m_xy_t90 = (m_xy_I*rho_I)/rho;
    m_xz_t90 = (2*rho + 6*m_xx_I*rho_I + 63*m_xz_I*rho_I - 3*m_yy_I*rho_I + 6*m_zz_I*rho_I - 6*rho*ux0 - 6*rho*uz0)/(45*rho);
    m_yy_t45 = -(2*(rho + 3*m_xx_I*rho_I + 9*m_xz_I*rho_I - 69*m_yy_I*rho_I + 3*m_zz_I*rho_I - 3*rho*ux0 - 3*rho*uz0))/(135*rho);
    m_yz_t90 = (m_yz_I*rho_I)/rho;
    m_zz_t45 = (4*rho + 12*m_xx_I*rho_I + 36*m_xz_I*rho_I - 6*m_yy_I*rho_I + 147*m_zz_I*rho_I - 12*rho*ux0 - 12*rho*uz0)/(135*rho);

    rhoVar = rho;

    break;
case 0b0000010110100010:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[11] + pop[12] + pop[13] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13]);
    m_xz_I = inv_rho_I * (pop[9] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = -(24*(15*rho_I + 4*m_xx_I*rho_I - 6*m_xy_I*rho_I + 6*m_xz_I*rho_I + m_yy_I*rho_I + m_zz_I*rho_I - 4*omegaVar*m_xx_I*rho_I + 6*omegaVar*m_xy_I*rho_I - 6*omegaVar*m_xz_I*rho_I - omegaVar*m_yy_I*rho_I - omegaVar*m_zz_I*rho_I))/(24*ux0 - 12*omegaVar - 12*uy0 + 12*uz0 + 36*omegaVar*ux0 - 18*omegaVar*uy0 + 18*omegaVar*uz0 + 60*omegaVar*ux0*ux0 + 15*omegaVar*uy0*uy0 + 15*omegaVar*uz0*uz0 - 90*omegaVar*ux0*uy0 + 90*omegaVar*ux0*uz0 - 328);

    m_xx_t45 = (8*rho + 159*m_xx_I*rho_I - 36*m_xy_I*rho_I + 36*m_xz_I*rho_I + 6*m_yy_I*rho_I + 6*m_zz_I*rho_I - 24*rho*ux0 + 12*rho*uy0 - 12*rho*uz0)/(135*rho);
    m_xy_t90 = -(2*rho + 6*m_xx_I*rho_I - 63*m_xy_I*rho_I + 6*m_yy_I*rho_I - 3*m_zz_I*rho_I - 6*rho*ux0 + 6*rho*uy0)/(45*rho);
    m_xz_t90 = (2*rho + 6*m_xx_I*rho_I + 63*m_xz_I*rho_I - 3*m_yy_I*rho_I + 6*m_zz_I*rho_I - 6*rho*ux0 - 6*rho*uz0)/(45*rho);
    m_yy_t45 = (2*(rho + 3*m_xx_I*rho_I - 18*m_xy_I*rho_I - 9*m_xz_I*rho_I + 75*m_yy_I*rho_I - 6*m_zz_I*rho_I - 3*rho*ux0 + 6*rho*uy0 + 3*rho*uz0))/(135*rho);
    m_yz_t90 = (m_yz_I*rho_I)/rho;
    m_zz_t45 = (2*(rho + 3*m_xx_I*rho_I + 9*m_xy_I*rho_I + 18*m_xz_I*rho_I - 6*m_yy_I*rho_I + 75*m_zz_I*rho_I - 3*rho*ux0 - 3*rho*uy0 - 6*rho*uz0))/(135*rho);

    rhoVar = rho;

    break;
case 0b0000010110100011:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[12] + pop[13] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13]);
    m_xz_I = inv_rho_I * (pop[9] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = -(24*(5*rho_I + m_xx_I*rho_I - 2*m_xy_I*rho_I + 2*m_xz_I*rho_I + m_yy_I*rho_I + 2*m_yz_I*rho_I + m_zz_I*rho_I - omegaVar*m_xx_I*rho_I + 2*omegaVar*m_xy_I*rho_I - 2*omegaVar*m_xz_I*rho_I - omegaVar*m_yy_I*rho_I - 2*omegaVar*m_yz_I*rho_I - omegaVar*m_zz_I*rho_I))/(8*ux0 - 6*omegaVar - 8*uy0 + 12*omegaVar*ux0 - 12*omegaVar*uy0 + 15*omegaVar*ux0*ux0 + 15*omegaVar*uy0*uy0 + 15*omegaVar*uz0*uz0 - 30*omegaVar*ux0*uy0 + 30*omegaVar*ux0*uz0 + 30*omegaVar*uy0*uz0 - 104);

    m_xx_t45 = (2*(rho + 27*m_xx_I*rho_I - 6*m_xy_I*rho_I + 6*m_xz_I*rho_I - 3*m_yz_I*rho_I - 4*rho*ux0 + rho*uy0 - 3*rho*uz0))/(45*rho);
    m_xy_t90 = -(2*rho + 6*m_xx_I*rho_I - 63*m_xy_I*rho_I + 6*m_yy_I*rho_I - 3*m_zz_I*rho_I - 6*rho*ux0 + 6*rho*uy0)/(45*rho);
    m_xz_t90 = (2*rho + 6*m_xx_I*rho_I + 63*m_xz_I*rho_I - 3*m_yy_I*rho_I + 6*m_zz_I*rho_I - 6*rho*ux0 - 6*rho*uz0)/(45*rho);
    m_yy_t45 = (2*(rho - 6*m_xy_I*rho_I - 3*m_xz_I*rho_I + 27*m_yy_I*rho_I + 6*m_yz_I*rho_I - rho*ux0 + 4*rho*uy0 + 3*rho*uz0))/(45*rho);
    m_yz_t90 = (2*rho - 3*m_xx_I*rho_I + 6*m_yy_I*rho_I + 63*m_yz_I*rho_I + 6*m_zz_I*rho_I + 6*rho*uy0 + 6*rho*uz0)/(45*rho);
    m_zz_t45 = (2*(rho + 3*m_xy_I*rho_I + 6*m_xz_I*rho_I + 6*m_yz_I*rho_I + 27*m_zz_I*rho_I - rho*ux0 + rho*uy0))/(45*rho);

    rhoVar = rho;

    break;
case 0b0000010110100100:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = -(24*(15*rho_I + 2*m_xx_I*rho_I + 6*m_xz_I*rho_I - m_yy_I*rho_I + 2*m_zz_I*rho_I - 2*omegaVar*m_xx_I*rho_I - 6*omegaVar*m_xz_I*rho_I + omegaVar*m_yy_I*rho_I - 2*omegaVar*m_zz_I*rho_I))/(12*ux0 - 6*omegaVar + 12*uz0 + 18*omegaVar*ux0 + 18*omegaVar*uz0 + 30*omegaVar*ux0*ux0 - 15*omegaVar*uy0*uy0 + 30*omegaVar*uz0*uz0 + 90*omegaVar*ux0*uz0 - 344);

    m_xx_t45 = (4*rho + 147*m_xx_I*rho_I + 36*m_xz_I*rho_I - 6*m_yy_I*rho_I + 12*m_zz_I*rho_I - 12*rho*ux0 - 12*rho*uz0)/(135*rho);
    m_xy_t90 = (m_xy_I*rho_I)/rho;
    m_xz_t90 = (2*rho + 6*m_xx_I*rho_I + 63*m_xz_I*rho_I - 3*m_yy_I*rho_I + 6*m_zz_I*rho_I - 6*rho*ux0 - 6*rho*uz0)/(45*rho);
    m_yy_t45 = -(2*(rho + 3*m_xx_I*rho_I + 9*m_xz_I*rho_I - 69*m_yy_I*rho_I + 3*m_zz_I*rho_I - 3*rho*ux0 - 3*rho*uz0))/(135*rho);
    m_yz_t90 = (m_yz_I*rho_I)/rho;
    m_zz_t45 = (4*rho + 12*m_xx_I*rho_I + 36*m_xz_I*rho_I - 6*m_yy_I*rho_I + 147*m_zz_I*rho_I - 12*rho*ux0 - 12*rho*uz0)/(135*rho);

    rhoVar = rho;

    break;
case 0b0000010110100101:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (-pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (12*(3*rho_I + 2*m_xx_I*rho_I + 6*m_xz_I*rho_I - m_yy_I*rho_I + 2*m_zz_I*rho_I - 2*omegaVar*m_xx_I*rho_I - 6*omegaVar*m_xz_I*rho_I + omegaVar*m_yy_I*rho_I - 2*omegaVar*m_zz_I*rho_I))/(6*omegaVar - 6*omegaVar*ux0*ux0 + 3*omegaVar*uy0*uy0 - 6*omegaVar*uz0*uz0 - 18*omegaVar*ux0*uz0 + 28);

    m_xx_t45 = (4*rho + 39*m_xx_I*rho_I + 36*m_xz_I*rho_I - 6*m_yy_I*rho_I + 12*m_zz_I*rho_I)/(27*rho);
    m_xy_t90 = (m_xy_I*rho_I)/rho;
    m_xz_t90 = (2*rho + 6*m_xx_I*rho_I + 27*m_xz_I*rho_I - 3*m_yy_I*rho_I + 6*m_zz_I*rho_I)/(9*rho);
    m_yy_t45 = -(2*(rho + 3*m_xx_I*rho_I + 9*m_xz_I*rho_I - 15*m_yy_I*rho_I + 3*m_zz_I*rho_I))/(27*rho);
    m_yz_t90 = (m_yz_I*rho_I)/rho;
    m_zz_t45 = (4*rho + 12*m_xx_I*rho_I + 36*m_xz_I*rho_I - 6*m_yy_I*rho_I + 39*m_zz_I*rho_I)/(27*rho);

    rhoVar = rho;

    break;
case 0b0000010110100110:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[11] + pop[12] + pop[13] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13]);
    m_xz_I = inv_rho_I * (pop[9] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = -(24*(15*rho_I + 4*m_xx_I*rho_I - 6*m_xy_I*rho_I + 6*m_xz_I*rho_I + m_yy_I*rho_I + m_zz_I*rho_I - 4*omegaVar*m_xx_I*rho_I + 6*omegaVar*m_xy_I*rho_I - 6*omegaVar*m_xz_I*rho_I - omegaVar*m_yy_I*rho_I - omegaVar*m_zz_I*rho_I))/(24*ux0 - 12*omegaVar - 12*uy0 + 12*uz0 + 36*omegaVar*ux0 - 18*omegaVar*uy0 + 18*omegaVar*uz0 + 60*omegaVar*ux0*ux0 + 15*omegaVar*uy0*uy0 + 15*omegaVar*uz0*uz0 - 90*omegaVar*ux0*uy0 + 90*omegaVar*ux0*uz0 - 328);

    m_xx_t45 = (8*rho + 159*m_xx_I*rho_I - 36*m_xy_I*rho_I + 36*m_xz_I*rho_I + 6*m_yy_I*rho_I + 6*m_zz_I*rho_I - 24*rho*ux0 + 12*rho*uy0 - 12*rho*uz0)/(135*rho);
    m_xy_t90 = -(2*rho + 6*m_xx_I*rho_I - 63*m_xy_I*rho_I + 6*m_yy_I*rho_I - 3*m_zz_I*rho_I - 6*rho*ux0 + 6*rho*uy0)/(45*rho);
    m_xz_t90 = (2*rho + 6*m_xx_I*rho_I + 63*m_xz_I*rho_I - 3*m_yy_I*rho_I + 6*m_zz_I*rho_I - 6*rho*ux0 - 6*rho*uz0)/(45*rho);
    m_yy_t45 = (2*(rho + 3*m_xx_I*rho_I - 18*m_xy_I*rho_I - 9*m_xz_I*rho_I + 75*m_yy_I*rho_I - 6*m_zz_I*rho_I - 3*rho*ux0 + 6*rho*uy0 + 3*rho*uz0))/(135*rho);
    m_yz_t90 = (m_yz_I*rho_I)/rho;
    m_zz_t45 = (2*(rho + 3*m_xx_I*rho_I + 9*m_xy_I*rho_I + 18*m_xz_I*rho_I - 6*m_yy_I*rho_I + 75*m_zz_I*rho_I - 3*rho*ux0 - 3*rho*uy0 - 6*rho*uz0))/(135*rho);

    rhoVar = rho;

    break;
case 0b0000010110100111:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[12] + pop[13] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13]);
    m_xz_I = inv_rho_I * (-pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (24*(15*rho_I + 11*m_xx_I*rho_I - 6*m_xy_I*rho_I + 30*m_xz_I*rho_I - m_yy_I*rho_I + 6*m_yz_I*rho_I + 11*m_zz_I*rho_I - 11*omegaVar*m_xx_I*rho_I + 6*omegaVar*m_xy_I*rho_I - 30*omegaVar*m_xz_I*rho_I + omegaVar*m_yy_I*rho_I - 6*omegaVar*m_yz_I*rho_I - 11*omegaVar*m_zz_I*rho_I))/(72*omegaVar - 12*ux0 + 24*uy0 + 12*uz0 - 18*omegaVar*ux0 + 36*omegaVar*uy0 + 18*omegaVar*uz0 - 75*omegaVar*ux0*ux0 - 30*omegaVar*uy0*uy0 - 75*omegaVar*uz0*uz0 + 90*omegaVar*ux0*uy0 - 180*omegaVar*ux0*uz0 - 90*omegaVar*uy0*uz0 + 248);

    m_xx_t45 = (2*(11*rho + 105*m_xx_I*rho_I - 18*m_xy_I*rho_I + 90*m_xz_I*rho_I - 12*m_yy_I*rho_I - 9*m_yz_I*rho_I + 24*m_zz_I*rho_I - 6*rho*ux0 + 3*rho*uy0 - 3*rho*uz0))/(135*rho);
    m_xy_t90 = -(2*rho + 6*m_xx_I*rho_I - 63*m_xy_I*rho_I + 6*m_yy_I*rho_I - 3*m_zz_I*rho_I - 6*rho*ux0 + 6*rho*uy0)/(45*rho);
    m_xz_t90 = (2*rho + 6*m_xx_I*rho_I + 27*m_xz_I*rho_I - 3*m_yy_I*rho_I + 6*m_zz_I*rho_I)/(9*rho);
    m_yy_t45 = -(2*(rho + 12*m_xx_I*rho_I + 18*m_xy_I*rho_I + 45*m_xz_I*rho_I - 87*m_yy_I*rho_I - 18*m_yz_I*rho_I + 12*m_zz_I*rho_I + 6*rho*ux0 - 12*rho*uy0 - 6*rho*uz0))/(135*rho);
    m_yz_t90 = (2*rho - 3*m_xx_I*rho_I + 6*m_yy_I*rho_I + 63*m_yz_I*rho_I + 6*m_zz_I*rho_I + 6*rho*uy0 + 6*rho*uz0)/(45*rho);
    m_zz_t45 = (2*(11*rho + 24*m_xx_I*rho_I + 9*m_xy_I*rho_I + 90*m_xz_I*rho_I - 12*m_yy_I*rho_I + 18*m_yz_I*rho_I + 105*m_zz_I*rho_I + 3*rho*ux0 + 3*rho*uy0 + 6*rho*uz0))/(135*rho);

    rhoVar = rho;

    break;
case 0b0000010110101000:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[9] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[9] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] - (1.0/3.0)*pop[9] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = -(24*(15*rho_I + 4*m_xx_I*rho_I + 6*m_xy_I*rho_I + 6*m_xz_I*rho_I + m_yy_I*rho_I + m_zz_I*rho_I - 4*omegaVar*m_xx_I*rho_I - 6*omegaVar*m_xy_I*rho_I - 6*omegaVar*m_xz_I*rho_I - omegaVar*m_yy_I*rho_I - omegaVar*m_zz_I*rho_I))/(24*ux0 - 12*omegaVar + 12*uy0 + 12*uz0 + 36*omegaVar*ux0 + 18*omegaVar*uy0 + 18*omegaVar*uz0 + 60*omegaVar*ux0*ux0 + 15*omegaVar*uy0*uy0 + 15*omegaVar*uz0*uz0 + 90*omegaVar*ux0*uy0 + 90*omegaVar*ux0*uz0 - 328);

    m_xx_t45 = (8*rho + 159*m_xx_I*rho_I + 36*m_xy_I*rho_I + 36*m_xz_I*rho_I + 6*m_yy_I*rho_I + 6*m_zz_I*rho_I - 24*rho*ux0 - 12*rho*uy0 - 12*rho*uz0)/(135*rho);
    m_xy_t90 = (2*rho + 6*m_xx_I*rho_I + 63*m_xy_I*rho_I + 6*m_yy_I*rho_I - 3*m_zz_I*rho_I - 6*rho*ux0 - 6*rho*uy0)/(45*rho);
    m_xz_t90 = (2*rho + 6*m_xx_I*rho_I + 63*m_xz_I*rho_I - 3*m_yy_I*rho_I + 6*m_zz_I*rho_I - 6*rho*ux0 - 6*rho*uz0)/(45*rho);
    m_yy_t45 = (2*(rho + 3*m_xx_I*rho_I + 18*m_xy_I*rho_I - 9*m_xz_I*rho_I + 75*m_yy_I*rho_I - 6*m_zz_I*rho_I - 3*rho*ux0 - 6*rho*uy0 + 3*rho*uz0))/(135*rho);
    m_yz_t90 = (m_yz_I*rho_I)/rho;
    m_zz_t45 = (2*(rho + 3*m_xx_I*rho_I - 9*m_xy_I*rho_I + 18*m_xz_I*rho_I - 6*m_yy_I*rho_I + 75*m_zz_I*rho_I - 3*rho*ux0 + 3*rho*uy0 - 6*rho*uz0))/(135*rho);

    rhoVar = rho;

    break;
case 0b0000010110101001:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[9] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[9] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] - (1.0/3.0)*pop[9] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = -(24*(15*rho_I + 4*m_xx_I*rho_I + 6*m_xy_I*rho_I + 6*m_xz_I*rho_I + m_yy_I*rho_I + m_zz_I*rho_I - 4*omegaVar*m_xx_I*rho_I - 6*omegaVar*m_xy_I*rho_I - 6*omegaVar*m_xz_I*rho_I - omegaVar*m_yy_I*rho_I - omegaVar*m_zz_I*rho_I))/(24*ux0 - 12*omegaVar + 12*uy0 + 12*uz0 + 36*omegaVar*ux0 + 18*omegaVar*uy0 + 18*omegaVar*uz0 + 60*omegaVar*ux0*ux0 + 15*omegaVar*uy0*uy0 + 15*omegaVar*uz0*uz0 + 90*omegaVar*ux0*uy0 + 90*omegaVar*ux0*uz0 - 328);

    m_xx_t45 = (8*rho + 159*m_xx_I*rho_I + 36*m_xy_I*rho_I + 36*m_xz_I*rho_I + 6*m_yy_I*rho_I + 6*m_zz_I*rho_I - 24*rho*ux0 - 12*rho*uy0 - 12*rho*uz0)/(135*rho);
    m_xy_t90 = (2*rho + 6*m_xx_I*rho_I + 63*m_xy_I*rho_I + 6*m_yy_I*rho_I - 3*m_zz_I*rho_I - 6*rho*ux0 - 6*rho*uy0)/(45*rho);
    m_xz_t90 = (2*rho + 6*m_xx_I*rho_I + 63*m_xz_I*rho_I - 3*m_yy_I*rho_I + 6*m_zz_I*rho_I - 6*rho*ux0 - 6*rho*uz0)/(45*rho);
    m_yy_t45 = (2*(rho + 3*m_xx_I*rho_I + 18*m_xy_I*rho_I - 9*m_xz_I*rho_I + 75*m_yy_I*rho_I - 6*m_zz_I*rho_I - 3*rho*ux0 - 6*rho*uy0 + 3*rho*uz0))/(135*rho);
    m_yz_t90 = (m_yz_I*rho_I)/rho;
    m_zz_t45 = (2*(rho + 3*m_xx_I*rho_I - 9*m_xy_I*rho_I + 18*m_xz_I*rho_I - 6*m_yy_I*rho_I + 75*m_zz_I*rho_I - 3*rho*ux0 + 3*rho*uy0 - 6*rho*uz0))/(135*rho);

    rhoVar = rho;

    break;
case 0b0000010110101010:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[9] + pop[11] + pop[12] + pop[13] + pop[15] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[9] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[15] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] - pop[13]);
    m_xz_I = inv_rho_I * (pop[9] - pop[15]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] - (1.0/3.0)*pop[9] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] - (1.0/3.0)*pop[15] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = -(3*rho_I*(3*m_xx_I - 3*omegaVar*m_xx_I + 4))/(3*ux0 - omegaVar + 3*omegaVar*ux0 + 6*omegaVar*ux0*ux0 - 9);

    m_xx_t45 = (rho + 9*m_xx_I*rho_I - 3*rho*ux0)/(6*rho);
    m_xy_t90 = (6*m_xy_I*rho_I - rho*uy0)/(3*rho);
    m_xz_t90 = (6*m_xz_I*rho_I - rho*uz0)/(3*rho);
    m_yy_t45 = (4*(10*m_yy_I*rho_I - m_zz_I*rho_I))/(33*rho);
    m_yz_t90 = (m_yz_I*rho_I)/rho;
    m_zz_t45 = -(4*(m_yy_I*rho_I - 10*m_zz_I*rho_I))/(33*rho);

    rhoVar = rho;

    break;
case 0b0000010110101011:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[9] + pop[12] + pop[13] + pop[15] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[9] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[15] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] - pop[13]);
    m_xz_I = inv_rho_I * (pop[9] - pop[15]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] - (1.0/3.0)*pop[9] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] - (1.0/3.0)*pop[15] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (72*(107*rho_I + 72*m_xx_I*rho_I + 12*m_yy_I*rho_I + 33*m_yz_I*rho_I + 12*m_zz_I*rho_I - 72*omegaVar*m_xx_I*rho_I - 12*omegaVar*m_yy_I*rho_I - 33*omegaVar*m_yz_I*rho_I - 12*omegaVar*m_zz_I*rho_I))/(626*omegaVar - 2124*ux0 + 492*uy0 + 492*uz0 - 1728*omegaVar*ux0 + 150*omegaVar*uy0 + 150*omegaVar*uz0 - 3531*omegaVar*ux0*ux0 - 642*omegaVar*uy0*uy0 - 642*omegaVar*uz0*uz0 - 1926*omegaVar*uy0*uz0 + 5580);

    m_xx_t45 = -(2*(12*m_yy_I*rho_I - 249*m_xx_I*rho_I - 24*rho + 33*m_yz_I*rho_I + 12*m_zz_I*rho_I + 83*rho*ux0 + 11*rho*uy0 + 11*rho*uz0))/(321*rho);
    m_xy_t90 = (6*m_xy_I*rho_I - rho*uy0)/(3*rho);
    m_xz_t90 = (6*m_xz_I*rho_I - rho*uz0)/(3*rho);
    m_yy_t45 = (4*(2*rho - 6*m_xx_I*rho_I + 106*m_yy_I*rho_I + 24*m_yz_I*rho_I - m_zz_I*rho_I + 2*rho*ux0 + 8*rho*uy0 + 8*rho*uz0))/(321*rho);
    m_yz_t90 = (11*rho - 33*m_xx_I*rho_I + 48*m_yy_I*rho_I + 453*m_yz_I*rho_I + 48*m_zz_I*rho_I + 11*rho*ux0 + 44*rho*uy0 + 44*rho*uz0)/(321*rho);
    m_zz_t45 = (4*(2*rho - 6*m_xx_I*rho_I - m_yy_I*rho_I + 24*m_yz_I*rho_I + 106*m_zz_I*rho_I + 2*rho*ux0 + 8*rho*uy0 + 8*rho*uz0))/(321*rho);

    rhoVar = rho;

    break;
case 0b0000010110101100:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[9] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[9] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17]);
    m_xy_I = inv_rho_I * (pop[7] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] - (1.0/3.0)*pop[9] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17]);

    rho = -(24*(5*rho_I + m_xx_I*rho_I + 2*m_xy_I*rho_I + 2*m_xz_I*rho_I + m_yy_I*rho_I - 2*m_yz_I*rho_I + m_zz_I*rho_I - omegaVar*m_xx_I*rho_I - 2*omegaVar*m_xy_I*rho_I - 2*omegaVar*m_xz_I*rho_I - omegaVar*m_yy_I*rho_I + 2*omegaVar*m_yz_I*rho_I - omegaVar*m_zz_I*rho_I))/(8*ux0 - 6*omegaVar + 8*uy0 + 12*omegaVar*ux0 + 12*omegaVar*uy0 + 15*omegaVar*ux0*ux0 + 15*omegaVar*uy0*uy0 + 15*omegaVar*uz0*uz0 + 30*omegaVar*ux0*uy0 + 30*omegaVar*ux0*uz0 - 30*omegaVar*uy0*uz0 - 104);

    m_xx_t45 = (2*(rho + 27*m_xx_I*rho_I + 6*m_xy_I*rho_I + 6*m_xz_I*rho_I + 3*m_yz_I*rho_I - 4*rho*ux0 - rho*uy0 - 3*rho*uz0))/(45*rho);
    m_xy_t90 = (2*rho + 6*m_xx_I*rho_I + 63*m_xy_I*rho_I + 6*m_yy_I*rho_I - 3*m_zz_I*rho_I - 6*rho*ux0 - 6*rho*uy0)/(45*rho);
    m_xz_t90 = (2*rho + 6*m_xx_I*rho_I + 63*m_xz_I*rho_I - 3*m_yy_I*rho_I + 6*m_zz_I*rho_I - 6*rho*ux0 - 6*rho*uz0)/(45*rho);
    m_yy_t45 = (2*(rho + 6*m_xy_I*rho_I - 3*m_xz_I*rho_I + 27*m_yy_I*rho_I - 6*m_yz_I*rho_I - rho*ux0 - 4*rho*uy0 + 3*rho*uz0))/(45*rho);
    m_yz_t90 = -(2*rho - 3*m_xx_I*rho_I + 6*m_yy_I*rho_I - 63*m_yz_I*rho_I + 6*m_zz_I*rho_I - 6*rho*uy0 + 6*rho*uz0)/(45*rho);
    m_zz_t45 = -(2*(3*m_xy_I*rho_I - rho - 6*m_xz_I*rho_I + 6*m_yz_I*rho_I - 27*m_zz_I*rho_I + rho*ux0 + rho*uy0))/(45*rho);

    rhoVar = rho;

    break;
case 0b0000010110101101:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17]);
    m_xy_I = inv_rho_I * (pop[7] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (-pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17]);

    rho = -(24*(15*rho_I + 11*m_xx_I*rho_I + 6*m_xy_I*rho_I + 30*m_xz_I*rho_I - m_yy_I*rho_I - 6*m_yz_I*rho_I + 11*m_zz_I*rho_I - 11*omegaVar*m_xx_I*rho_I - 6*omegaVar*m_xy_I*rho_I - 30*omegaVar*m_xz_I*rho_I + omegaVar*m_yy_I*rho_I + 6*omegaVar*m_yz_I*rho_I - 11*omegaVar*m_zz_I*rho_I))/(12*ux0 - 72*omegaVar + 24*uy0 - 12*uz0 + 18*omegaVar*ux0 + 36*omegaVar*uy0 - 18*omegaVar*uz0 + 75*omegaVar*ux0*ux0 + 30*omegaVar*uy0*uy0 + 75*omegaVar*uz0*uz0 + 90*omegaVar*ux0*uy0 + 180*omegaVar*ux0*uz0 - 90*omegaVar*uy0*uz0 - 248);

    m_xx_t45 = (2*(11*rho + 105*m_xx_I*rho_I + 18*m_xy_I*rho_I + 90*m_xz_I*rho_I - 12*m_yy_I*rho_I + 9*m_yz_I*rho_I + 24*m_zz_I*rho_I - 6*rho*ux0 - 3*rho*uy0 - 3*rho*uz0))/(135*rho);
    m_xy_t90 = (2*rho + 6*m_xx_I*rho_I + 63*m_xy_I*rho_I + 6*m_yy_I*rho_I - 3*m_zz_I*rho_I - 6*rho*ux0 - 6*rho*uy0)/(45*rho);
    m_xz_t90 = (2*rho + 6*m_xx_I*rho_I + 27*m_xz_I*rho_I - 3*m_yy_I*rho_I + 6*m_zz_I*rho_I)/(9*rho);
    m_yy_t45 = -(2*(rho + 12*m_xx_I*rho_I - 18*m_xy_I*rho_I + 45*m_xz_I*rho_I - 87*m_yy_I*rho_I + 18*m_yz_I*rho_I + 12*m_zz_I*rho_I + 6*rho*ux0 + 12*rho*uy0 - 6*rho*uz0))/(135*rho);
    m_yz_t90 = -(2*rho - 3*m_xx_I*rho_I + 6*m_yy_I*rho_I - 63*m_yz_I*rho_I + 6*m_zz_I*rho_I - 6*rho*uy0 + 6*rho*uz0)/(45*rho);
    m_zz_t45 = (2*(11*rho + 24*m_xx_I*rho_I - 9*m_xy_I*rho_I + 90*m_xz_I*rho_I - 12*m_yy_I*rho_I - 18*m_yz_I*rho_I + 105*m_zz_I*rho_I + 3*rho*ux0 - 3*rho*uy0 + 6*rho*uz0))/(135*rho);

    rhoVar = rho;

    break;
case 0b0000010110101110:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[9] + pop[11] + pop[12] + pop[13] + pop[15] + pop[17];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[9] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[15] - (1.0/3.0)*pop[17]);
    m_xy_I = inv_rho_I * (pop[7] - pop[13]);
    m_xz_I = inv_rho_I * (pop[9] - pop[15]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] - (1.0/3.0)*pop[9] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] - (1.0/3.0)*pop[15] + (2.0/3.0)*pop[17]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[17]);

    rho = -(72*(107*rho_I + 72*m_xx_I*rho_I + 12*m_yy_I*rho_I - 33*m_yz_I*rho_I + 12*m_zz_I*rho_I - 72*omegaVar*m_xx_I*rho_I - 12*omegaVar*m_yy_I*rho_I + 33*omegaVar*m_yz_I*rho_I - 12*omegaVar*m_zz_I*rho_I))/(2124*ux0 - 626*omegaVar + 492*uy0 - 492*uz0 + 1728*omegaVar*ux0 + 150*omegaVar*uy0 - 150*omegaVar*uz0 + 3531*omegaVar*ux0*ux0 + 642*omegaVar*uy0*uy0 + 642*omegaVar*uz0*uz0 - 1926*omegaVar*uy0*uz0 - 5580);

    m_xx_t45 = (2*(24*rho + 249*m_xx_I*rho_I - 12*m_yy_I*rho_I + 33*m_yz_I*rho_I - 12*m_zz_I*rho_I - 83*rho*ux0 + 11*rho*uy0 - 11*rho*uz0))/(321*rho);
    m_xy_t90 = (6*m_xy_I*rho_I - rho*uy0)/(3*rho);
    m_xz_t90 = (6*m_xz_I*rho_I - rho*uz0)/(3*rho);
    m_yy_t45 = (4*(2*rho - 6*m_xx_I*rho_I + 106*m_yy_I*rho_I - 24*m_yz_I*rho_I - m_zz_I*rho_I + 2*rho*ux0 - 8*rho*uy0 + 8*rho*uz0))/(321*rho);
    m_yz_t90 = -(11*rho - 33*m_xx_I*rho_I + 48*m_yy_I*rho_I - 453*m_yz_I*rho_I + 48*m_zz_I*rho_I + 11*rho*ux0 - 44*rho*uy0 + 44*rho*uz0)/(321*rho);
    m_zz_t45 = (4*(2*rho - 6*m_xx_I*rho_I - m_yy_I*rho_I - 24*m_yz_I*rho_I + 106*m_zz_I*rho_I + 2*rho*ux0 - 8*rho*uy0 + 8*rho*uz0))/(321*rho);

    rhoVar = rho;

    break;
case 0b0000010110101111:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[3] + pop[4] + pop[6] + pop[7] + pop[12] + pop[13] + pop[15] + pop[17];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[15] - (1.0/3.0)*pop[17]);
    m_xy_I = inv_rho_I * (pop[7] - pop[13]);
    m_xz_I = inv_rho_I * (-pop[15]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] - (1.0/3.0)*pop[15] + (2.0/3.0)*pop[17]);
    m_yz_I = inv_rho_I * (pop[12] - pop[17]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[17]);

    rho = -(72*(23*rho_I + 24*m_xx_I*rho_I + 57*m_xz_I*rho_I - 6*m_yy_I*rho_I + 24*m_zz_I*rho_I - 24*omegaVar*m_xx_I*rho_I - 57*omegaVar*m_xz_I*rho_I + 6*omegaVar*m_yy_I*rho_I - 24*omegaVar*m_zz_I*rho_I))/(660*ux0 - 430*omegaVar - 660*uz0 + 30*omegaVar*ux0 - 30*omegaVar*uz0 + 690*omegaVar*ux0*ux0 + 69*omegaVar*uy0*uy0 + 690*omegaVar*uz0*uz0 + 414*omegaVar*ux0*uz0 - 720);

    m_xx_t45 = (16*rho + 141*m_xx_I*rho_I + 102*m_xz_I*rho_I - 18*m_yy_I*rho_I + 3*m_zz_I*rho_I - 30*rho*ux0 - 16*rho*uz0)/(69*rho);
    m_xy_t90 = (6*m_xy_I*rho_I - rho*uy0)/(3*rho);
    m_xz_t90 = (19*rho + 51*m_xx_I*rho_I + 354*m_xz_I*rho_I - 30*m_yy_I*rho_I + 51*m_zz_I*rho_I + 19*rho*ux0 - 19*rho*uz0)/(69*rho);
    m_yy_t45 = -(2*(2*rho + 9*m_xx_I*rho_I + 30*m_xz_I*rho_I - 54*m_yy_I*rho_I + 9*m_zz_I*rho_I + 2*rho*ux0 - 2*rho*uz0))/(69*rho);
    m_yz_t90 = (6*m_yz_I*rho_I + rho*uy0)/(3*rho);
    m_zz_t45 = (16*rho + 3*m_xx_I*rho_I + 102*m_xz_I*rho_I - 18*m_yy_I*rho_I + 141*m_zz_I*rho_I + 16*rho*ux0 + 30*rho*uz0)/(69*rho);

    rhoVar = rho;

    break;
case 0b0000010110110000:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);

    rho = -(24*(15*rho_I + m_xx_I*rho_I + 6*m_xz_I*rho_I + m_yy_I*rho_I - 6*m_yz_I*rho_I + 4*m_zz_I*rho_I - omegaVar*m_xx_I*rho_I - 6*omegaVar*m_xz_I*rho_I - omegaVar*m_yy_I*rho_I + 6*omegaVar*m_yz_I*rho_I - 4*omegaVar*m_zz_I*rho_I))/(12*ux0 - 12*omegaVar - 12*uy0 + 24*uz0 + 18*omegaVar*ux0 - 18*omegaVar*uy0 + 36*omegaVar*uz0 + 15*omegaVar*ux0*ux0 + 15*omegaVar*uy0*uy0 + 60*omegaVar*uz0*uz0 + 90*omegaVar*ux0*uz0 - 90*omegaVar*uy0*uz0 - 328);

    m_xx_t45 = (2*(rho + 75*m_xx_I*rho_I + 18*m_xz_I*rho_I - 6*m_yy_I*rho_I + 9*m_yz_I*rho_I + 3*m_zz_I*rho_I - 6*rho*ux0 - 3*rho*uy0 - 3*rho*uz0))/(135*rho);
    m_xy_t90 = (m_xy_I*rho_I)/rho;
    m_xz_t90 = (2*rho + 6*m_xx_I*rho_I + 63*m_xz_I*rho_I - 3*m_yy_I*rho_I + 6*m_zz_I*rho_I - 6*rho*ux0 - 6*rho*uz0)/(45*rho);
    m_yy_t45 = (2*(rho - 6*m_xx_I*rho_I - 9*m_xz_I*rho_I + 75*m_yy_I*rho_I - 18*m_yz_I*rho_I + 3*m_zz_I*rho_I + 3*rho*ux0 + 6*rho*uy0 - 3*rho*uz0))/(135*rho);
    m_yz_t90 = -(2*rho - 3*m_xx_I*rho_I + 6*m_yy_I*rho_I - 63*m_yz_I*rho_I + 6*m_zz_I*rho_I + 6*rho*uy0 - 6*rho*uz0)/(45*rho);
    m_zz_t45 = (8*rho + 6*m_xx_I*rho_I + 36*m_xz_I*rho_I + 6*m_yy_I*rho_I - 36*m_yz_I*rho_I + 159*m_zz_I*rho_I - 12*rho*ux0 + 12*rho*uy0 - 24*rho*uz0)/(135*rho);

    rhoVar = rho;

    break;
case 0b0000010110110001:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[8] + pop[9] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);

    rho = -(24*(5*rho_I + m_xx_I*rho_I + 2*m_xy_I*rho_I + 2*m_xz_I*rho_I + m_yy_I*rho_I - 2*m_yz_I*rho_I + m_zz_I*rho_I - omegaVar*m_xx_I*rho_I - 2*omegaVar*m_xy_I*rho_I - 2*omegaVar*m_xz_I*rho_I - omegaVar*m_yy_I*rho_I + 2*omegaVar*m_yz_I*rho_I - omegaVar*m_zz_I*rho_I))/(8*uz0 - 8*uy0 - 6*omegaVar - 12*omegaVar*uy0 + 12*omegaVar*uz0 + 15*omegaVar*ux0*ux0 + 15*omegaVar*uy0*uy0 + 15*omegaVar*uz0*uz0 + 30*omegaVar*ux0*uy0 + 30*omegaVar*ux0*uz0 - 30*omegaVar*uy0*uz0 - 104);

    m_xx_t45 = (2*(rho + 27*m_xx_I*rho_I + 6*m_xy_I*rho_I + 6*m_xz_I*rho_I + 3*m_yz_I*rho_I + rho*uy0 - rho*uz0))/(45*rho);
    m_xy_t90 = (2*rho + 6*m_xx_I*rho_I + 63*m_xy_I*rho_I + 6*m_yy_I*rho_I - 3*m_zz_I*rho_I + 6*rho*ux0 + 6*rho*uy0)/(45*rho);
    m_xz_t90 = (2*rho + 6*m_xx_I*rho_I + 63*m_xz_I*rho_I - 3*m_yy_I*rho_I + 6*m_zz_I*rho_I - 6*rho*ux0 - 6*rho*uz0)/(45*rho);
    m_yy_t45 = (2*(rho + 6*m_xy_I*rho_I - 3*m_xz_I*rho_I + 27*m_yy_I*rho_I - 6*m_yz_I*rho_I + 3*rho*ux0 + 4*rho*uy0 - rho*uz0))/(45*rho);
    m_yz_t90 = -(2*rho - 3*m_xx_I*rho_I + 6*m_yy_I*rho_I - 63*m_yz_I*rho_I + 6*m_zz_I*rho_I + 6*rho*uy0 - 6*rho*uz0)/(45*rho);
    m_zz_t45 = (2*(rho - 3*m_xy_I*rho_I + 6*m_xz_I*rho_I - 6*m_yz_I*rho_I + 27*m_zz_I*rho_I - 3*rho*ux0 + rho*uy0 - 4*rho*uz0))/(45*rho);

    rhoVar = rho;

    break;
case 0b0000010110110010:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[11] + pop[12] + pop[13] + pop[15] + pop[16] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13]);
    m_xz_I = inv_rho_I * (pop[9] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);

    rho = -(24*(5*rho_I + m_xx_I*rho_I - 2*m_xy_I*rho_I + 2*m_xz_I*rho_I + m_yy_I*rho_I - 2*m_yz_I*rho_I + m_zz_I*rho_I - omegaVar*m_xx_I*rho_I + 2*omegaVar*m_xy_I*rho_I - 2*omegaVar*m_xz_I*rho_I - omegaVar*m_yy_I*rho_I + 2*omegaVar*m_yz_I*rho_I - omegaVar*m_zz_I*rho_I))/(8*ux0 - 6*omegaVar - 8*uy0 + 8*uz0 + 12*omegaVar*ux0 - 12*omegaVar*uy0 + 12*omegaVar*uz0 + 15*omegaVar*ux0*ux0 + 15*omegaVar*uy0*uy0 + 15*omegaVar*uz0*uz0 - 30*omegaVar*ux0*uy0 + 30*omegaVar*ux0*uz0 - 30*omegaVar*uy0*uz0 - 104);

    m_xx_t45 = (2*(rho + 27*m_xx_I*rho_I - 6*m_xy_I*rho_I + 6*m_xz_I*rho_I + 3*m_yz_I*rho_I - 4*rho*ux0 + rho*uy0 - rho*uz0))/(45*rho);
    m_xy_t90 = -(2*rho + 6*m_xx_I*rho_I - 63*m_xy_I*rho_I + 6*m_yy_I*rho_I - 3*m_zz_I*rho_I - 6*rho*ux0 + 6*rho*uy0)/(45*rho);
    m_xz_t90 = (2*rho + 6*m_xx_I*rho_I + 63*m_xz_I*rho_I - 3*m_yy_I*rho_I + 6*m_zz_I*rho_I - 6*rho*ux0 - 6*rho*uz0)/(45*rho);
    m_yy_t45 = -(2*(6*m_xy_I*rho_I - rho + 3*m_xz_I*rho_I - 27*m_yy_I*rho_I + 6*m_yz_I*rho_I + rho*ux0 - 4*rho*uy0 + rho*uz0))/(45*rho);
    m_yz_t90 = -(2*rho - 3*m_xx_I*rho_I + 6*m_yy_I*rho_I - 63*m_yz_I*rho_I + 6*m_zz_I*rho_I + 6*rho*uy0 - 6*rho*uz0)/(45*rho);
    m_zz_t45 = (2*(rho + 3*m_xy_I*rho_I + 6*m_xz_I*rho_I - 6*m_yz_I*rho_I + 27*m_zz_I*rho_I - rho*ux0 + rho*uy0 - 4*rho*uz0))/(45*rho);

    rhoVar = rho;

    break;
case 0b0000010110110011:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[4] + pop[5] + pop[6] + pop[8] + pop[9] + pop[12] + pop[13] + pop[15] + pop[16] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[8] - pop[13]);
    m_xz_I = inv_rho_I * (pop[9] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[12] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);

    rho = -(72*(107*rho_I + 12*m_xx_I*rho_I + 33*m_xz_I*rho_I + 72*m_yy_I*rho_I + 12*m_zz_I*rho_I - 12*omegaVar*m_xx_I*rho_I - 33*omegaVar*m_xz_I*rho_I - 72*omegaVar*m_yy_I*rho_I - 12*omegaVar*m_zz_I*rho_I))/(492*ux0 - 626*omegaVar - 2124*uy0 + 492*uz0 + 150*omegaVar*ux0 - 1728*omegaVar*uy0 + 150*omegaVar*uz0 + 642*omegaVar*ux0*ux0 + 3531*omegaVar*uy0*uy0 + 642*omegaVar*uz0*uz0 + 1926*omegaVar*ux0*uz0 - 5580);

    m_xx_t45 = -(4*(6*m_yy_I*rho_I - 106*m_xx_I*rho_I - 24*m_xz_I*rho_I - 2*rho + m_zz_I*rho_I + 8*rho*ux0 + 2*rho*uy0 + 8*rho*uz0))/(321*rho);
    m_xy_t90 = (6*m_xy_I*rho_I + rho*ux0)/(3*rho);
    m_xz_t90 = (11*rho + 48*m_xx_I*rho_I + 453*m_xz_I*rho_I - 33*m_yy_I*rho_I + 48*m_zz_I*rho_I - 44*rho*ux0 - 11*rho*uy0 - 44*rho*uz0)/(321*rho);
    m_yy_t45 = (2*(24*rho - 12*m_xx_I*rho_I - 33*m_xz_I*rho_I + 249*m_yy_I*rho_I - 12*m_zz_I*rho_I + 11*rho*ux0 + 83*rho*uy0 + 11*rho*uz0))/(321*rho);
    m_yz_t90 = (6*m_yz_I*rho_I + rho*uz0)/(3*rho);
    m_zz_t45 = -(4*(m_xx_I*rho_I - 2*rho - 24*m_xz_I*rho_I + 6*m_yy_I*rho_I - 106*m_zz_I*rho_I + 8*rho*ux0 + 2*rho*uy0 + 8*rho*uz0))/(321*rho);

    rhoVar = rho;

    break;
case 0b0000010110110100:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);

    rho = -(24*(15*rho_I + m_xx_I*rho_I + 6*m_xz_I*rho_I + m_yy_I*rho_I - 6*m_yz_I*rho_I + 4*m_zz_I*rho_I - omegaVar*m_xx_I*rho_I - 6*omegaVar*m_xz_I*rho_I - omegaVar*m_yy_I*rho_I + 6*omegaVar*m_yz_I*rho_I - 4*omegaVar*m_zz_I*rho_I))/(12*ux0 - 12*omegaVar - 12*uy0 + 24*uz0 + 18*omegaVar*ux0 - 18*omegaVar*uy0 + 36*omegaVar*uz0 + 15*omegaVar*ux0*ux0 + 15*omegaVar*uy0*uy0 + 60*omegaVar*uz0*uz0 + 90*omegaVar*ux0*uz0 - 90*omegaVar*uy0*uz0 - 328);

    m_xx_t45 = (2*(rho + 75*m_xx_I*rho_I + 18*m_xz_I*rho_I - 6*m_yy_I*rho_I + 9*m_yz_I*rho_I + 3*m_zz_I*rho_I - 6*rho*ux0 - 3*rho*uy0 - 3*rho*uz0))/(135*rho);
    m_xy_t90 = (m_xy_I*rho_I)/rho;
    m_xz_t90 = (2*rho + 6*m_xx_I*rho_I + 63*m_xz_I*rho_I - 3*m_yy_I*rho_I + 6*m_zz_I*rho_I - 6*rho*ux0 - 6*rho*uz0)/(45*rho);
    m_yy_t45 = (2*(rho - 6*m_xx_I*rho_I - 9*m_xz_I*rho_I + 75*m_yy_I*rho_I - 18*m_yz_I*rho_I + 3*m_zz_I*rho_I + 3*rho*ux0 + 6*rho*uy0 - 3*rho*uz0))/(135*rho);
    m_yz_t90 = -(2*rho - 3*m_xx_I*rho_I + 6*m_yy_I*rho_I - 63*m_yz_I*rho_I + 6*m_zz_I*rho_I + 6*rho*uy0 - 6*rho*uz0)/(45*rho);
    m_zz_t45 = (8*rho + 6*m_xx_I*rho_I + 36*m_xz_I*rho_I + 6*m_yy_I*rho_I - 36*m_yz_I*rho_I + 159*m_zz_I*rho_I - 12*rho*ux0 + 12*rho*uy0 - 24*rho*uz0)/(135*rho);

    rhoVar = rho;

    break;
case 0b0000010110110101:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[8] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (-pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);

    rho = (24*(15*rho_I + 11*m_xx_I*rho_I + 6*m_xy_I*rho_I + 30*m_xz_I*rho_I - m_yy_I*rho_I - 6*m_yz_I*rho_I + 11*m_zz_I*rho_I - 11*omegaVar*m_xx_I*rho_I - 6*omegaVar*m_xy_I*rho_I - 30*omegaVar*m_xz_I*rho_I + omegaVar*m_yy_I*rho_I + 6*omegaVar*m_yz_I*rho_I - 11*omegaVar*m_zz_I*rho_I))/(72*omegaVar + 12*ux0 + 24*uy0 - 12*uz0 + 18*omegaVar*ux0 + 36*omegaVar*uy0 - 18*omegaVar*uz0 - 75*omegaVar*ux0*ux0 - 30*omegaVar*uy0*uy0 - 75*omegaVar*uz0*uz0 - 90*omegaVar*ux0*uy0 - 180*omegaVar*ux0*uz0 + 90*omegaVar*uy0*uz0 + 248);

    m_xx_t45 = (2*(11*rho + 105*m_xx_I*rho_I + 18*m_xy_I*rho_I + 90*m_xz_I*rho_I - 12*m_yy_I*rho_I + 9*m_yz_I*rho_I + 24*m_zz_I*rho_I + 6*rho*ux0 + 3*rho*uy0 + 3*rho*uz0))/(135*rho);
    m_xy_t90 = (2*rho + 6*m_xx_I*rho_I + 63*m_xy_I*rho_I + 6*m_yy_I*rho_I - 3*m_zz_I*rho_I + 6*rho*ux0 + 6*rho*uy0)/(45*rho);
    m_xz_t90 = (2*rho + 6*m_xx_I*rho_I + 27*m_xz_I*rho_I - 3*m_yy_I*rho_I + 6*m_zz_I*rho_I)/(9*rho);
    m_yy_t45 = -(2*(rho + 12*m_xx_I*rho_I - 18*m_xy_I*rho_I + 45*m_xz_I*rho_I - 87*m_yy_I*rho_I + 18*m_yz_I*rho_I + 12*m_zz_I*rho_I - 6*rho*ux0 - 12*rho*uy0 + 6*rho*uz0))/(135*rho);
    m_yz_t90 = -(2*rho - 3*m_xx_I*rho_I + 6*m_yy_I*rho_I - 63*m_yz_I*rho_I + 6*m_zz_I*rho_I + 6*rho*uy0 - 6*rho*uz0)/(45*rho);
    m_zz_t45 = (2*(11*rho + 24*m_xx_I*rho_I - 9*m_xy_I*rho_I + 90*m_xz_I*rho_I - 12*m_yy_I*rho_I - 18*m_yz_I*rho_I + 105*m_zz_I*rho_I - 3*rho*ux0 + 3*rho*uy0 - 6*rho*uz0))/(135*rho);

    rhoVar = rho;

    break;
case 0b0000010110110110:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[11] + pop[12] + pop[13] + pop[15] + pop[16] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13]);
    m_xz_I = inv_rho_I * (pop[9] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);

    rho = -(24*(5*rho_I + m_xx_I*rho_I - 2*m_xy_I*rho_I + 2*m_xz_I*rho_I + m_yy_I*rho_I - 2*m_yz_I*rho_I + m_zz_I*rho_I - omegaVar*m_xx_I*rho_I + 2*omegaVar*m_xy_I*rho_I - 2*omegaVar*m_xz_I*rho_I - omegaVar*m_yy_I*rho_I + 2*omegaVar*m_yz_I*rho_I - omegaVar*m_zz_I*rho_I))/(8*ux0 - 6*omegaVar - 8*uy0 + 8*uz0 + 12*omegaVar*ux0 - 12*omegaVar*uy0 + 12*omegaVar*uz0 + 15*omegaVar*ux0*ux0 + 15*omegaVar*uy0*uy0 + 15*omegaVar*uz0*uz0 - 30*omegaVar*ux0*uy0 + 30*omegaVar*ux0*uz0 - 30*omegaVar*uy0*uz0 - 104);

    m_xx_t45 = (2*(rho + 27*m_xx_I*rho_I - 6*m_xy_I*rho_I + 6*m_xz_I*rho_I + 3*m_yz_I*rho_I - 4*rho*ux0 + rho*uy0 - rho*uz0))/(45*rho);
    m_xy_t90 = -(2*rho + 6*m_xx_I*rho_I - 63*m_xy_I*rho_I + 6*m_yy_I*rho_I - 3*m_zz_I*rho_I - 6*rho*ux0 + 6*rho*uy0)/(45*rho);
    m_xz_t90 = (2*rho + 6*m_xx_I*rho_I + 63*m_xz_I*rho_I - 3*m_yy_I*rho_I + 6*m_zz_I*rho_I - 6*rho*ux0 - 6*rho*uz0)/(45*rho);
    m_yy_t45 = -(2*(6*m_xy_I*rho_I - rho + 3*m_xz_I*rho_I - 27*m_yy_I*rho_I + 6*m_yz_I*rho_I + rho*ux0 - 4*rho*uy0 + rho*uz0))/(45*rho);
    m_yz_t90 = -(2*rho - 3*m_xx_I*rho_I + 6*m_yy_I*rho_I - 63*m_yz_I*rho_I + 6*m_zz_I*rho_I + 6*rho*uy0 - 6*rho*uz0)/(45*rho);
    m_zz_t45 = (2*(rho + 3*m_xy_I*rho_I + 6*m_xz_I*rho_I - 6*m_yz_I*rho_I + 27*m_zz_I*rho_I - rho*ux0 + rho*uy0 - 4*rho*uz0))/(45*rho);

    rhoVar = rho;

    break;
case 0b0000010110110111:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[4] + pop[5] + pop[6] + pop[8] + pop[12] + pop[13] + pop[15] + pop[16] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[8] - pop[13]);
    m_xz_I = inv_rho_I * (-pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[12] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);

    rho = (36*(19*rho_I + 12*m_xx_I*rho_I + 33*m_xz_I*rho_I + 6*m_yy_I*rho_I + 12*m_zz_I*rho_I - 12*omegaVar*m_xx_I*rho_I - 33*omegaVar*m_xz_I*rho_I - 6*omegaVar*m_yy_I*rho_I - 12*omegaVar*m_zz_I*rho_I))/(118*omegaVar + 270*uy0 + 72*omegaVar*uy0 - 114*omegaVar*ux0*ux0 - 285*omegaVar*uy0*uy0 - 114*omegaVar*uz0*uz0 - 342*omegaVar*ux0*uz0 + 414);

    m_xx_t45 = (4*(2*rho + 26*m_xx_I*rho_I + 24*m_xz_I*rho_I - 6*m_yy_I*rho_I + 7*m_zz_I*rho_I - 2*rho*uy0))/(57*rho);
    m_xy_t90 = (6*m_xy_I*rho_I + rho*ux0)/(3*rho);
    m_xz_t90 = (11*rho + 48*m_xx_I*rho_I + 189*m_xz_I*rho_I - 33*m_yy_I*rho_I + 48*m_zz_I*rho_I - 11*rho*uy0)/(57*rho);
    m_yy_t45 = (2*(2*rho - 12*m_xx_I*rho_I - 33*m_xz_I*rho_I + 51*m_yy_I*rho_I - 12*m_zz_I*rho_I + 17*rho*uy0))/(57*rho);
    m_yz_t90 = (6*m_yz_I*rho_I + rho*uz0)/(3*rho);
    m_zz_t45 = (4*(2*rho + 7*m_xx_I*rho_I + 24*m_xz_I*rho_I - 6*m_yy_I*rho_I + 26*m_zz_I*rho_I - 2*rho*uy0))/(57*rho);

    rhoVar = rho;

    break;
case 0b0000010110111000:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[9] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[9] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] - (1.0/3.0)*pop[9] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);

    rho = -(24*(5*rho_I + m_xx_I*rho_I + 2*m_xy_I*rho_I + 2*m_xz_I*rho_I + m_yy_I*rho_I - 2*m_yz_I*rho_I + m_zz_I*rho_I - omegaVar*m_xx_I*rho_I - 2*omegaVar*m_xy_I*rho_I - 2*omegaVar*m_xz_I*rho_I - omegaVar*m_yy_I*rho_I + 2*omegaVar*m_yz_I*rho_I - omegaVar*m_zz_I*rho_I))/(8*ux0 - 6*omegaVar + 8*uz0 + 12*omegaVar*ux0 + 12*omegaVar*uz0 + 15*omegaVar*ux0*ux0 + 15*omegaVar*uy0*uy0 + 15*omegaVar*uz0*uz0 + 30*omegaVar*ux0*uy0 + 30*omegaVar*ux0*uz0 - 30*omegaVar*uy0*uz0 - 104);

    m_xx_t45 = (2*(rho + 27*m_xx_I*rho_I + 6*m_xy_I*rho_I + 6*m_xz_I*rho_I + 3*m_yz_I*rho_I - 4*rho*ux0 - 3*rho*uy0 - rho*uz0))/(45*rho);
    m_xy_t90 = (2*rho + 6*m_xx_I*rho_I + 63*m_xy_I*rho_I + 6*m_yy_I*rho_I - 3*m_zz_I*rho_I - 6*rho*ux0 - 6*rho*uy0)/(45*rho);
    m_xz_t90 = (2*rho + 6*m_xx_I*rho_I + 63*m_xz_I*rho_I - 3*m_yy_I*rho_I + 6*m_zz_I*rho_I - 6*rho*ux0 - 6*rho*uz0)/(45*rho);
    m_yy_t45 = -(2*(3*m_xz_I*rho_I - 6*m_xy_I*rho_I - rho - 27*m_yy_I*rho_I + 6*m_yz_I*rho_I + rho*ux0 + rho*uz0))/(45*rho);
    m_yz_t90 = -(2*rho - 3*m_xx_I*rho_I + 6*m_yy_I*rho_I - 63*m_yz_I*rho_I + 6*m_zz_I*rho_I + 6*rho*uy0 - 6*rho*uz0)/(45*rho);
    m_zz_t45 = (2*(rho - 3*m_xy_I*rho_I + 6*m_xz_I*rho_I - 6*m_yz_I*rho_I + 27*m_zz_I*rho_I - rho*ux0 + 3*rho*uy0 - 4*rho*uz0))/(45*rho);

    rhoVar = rho;

    break;
case 0b0000010110111001:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[9] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[9] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (-pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] - (1.0/3.0)*pop[9] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);

    rho = -(24*(15*rho_I + 11*m_xx_I*rho_I + 30*m_xy_I*rho_I + 6*m_xz_I*rho_I + 11*m_yy_I*rho_I - 6*m_yz_I*rho_I - m_zz_I*rho_I - 11*omegaVar*m_xx_I*rho_I - 30*omegaVar*m_xy_I*rho_I - 6*omegaVar*m_xz_I*rho_I - 11*omegaVar*m_yy_I*rho_I + 6*omegaVar*m_yz_I*rho_I + omegaVar*m_zz_I*rho_I))/(12*ux0 - 72*omegaVar - 12*uy0 + 24*uz0 + 18*omegaVar*ux0 - 18*omegaVar*uy0 + 36*omegaVar*uz0 + 75*omegaVar*ux0*ux0 + 75*omegaVar*uy0*uy0 + 30*omegaVar*uz0*uz0 + 180*omegaVar*ux0*uy0 + 90*omegaVar*ux0*uz0 - 90*omegaVar*uy0*uz0 - 248);

    m_xx_t45 = (2*(11*rho + 105*m_xx_I*rho_I + 90*m_xy_I*rho_I + 18*m_xz_I*rho_I + 24*m_yy_I*rho_I + 9*m_yz_I*rho_I - 12*m_zz_I*rho_I - 6*rho*ux0 - 3*rho*uy0 - 3*rho*uz0))/(135*rho);
    m_xy_t90 = (2*rho + 6*m_xx_I*rho_I + 27*m_xy_I*rho_I + 6*m_yy_I*rho_I - 3*m_zz_I*rho_I)/(9*rho);
    m_xz_t90 = (2*rho + 6*m_xx_I*rho_I + 63*m_xz_I*rho_I - 3*m_yy_I*rho_I + 6*m_zz_I*rho_I - 6*rho*ux0 - 6*rho*uz0)/(45*rho);
    m_yy_t45 = (2*(11*rho + 24*m_xx_I*rho_I + 90*m_xy_I*rho_I - 9*m_xz_I*rho_I + 105*m_yy_I*rho_I - 18*m_yz_I*rho_I - 12*m_zz_I*rho_I + 3*rho*ux0 + 6*rho*uy0 - 3*rho*uz0))/(135*rho);
    m_yz_t90 = -(2*rho - 3*m_xx_I*rho_I + 6*m_yy_I*rho_I - 63*m_yz_I*rho_I + 6*m_zz_I*rho_I + 6*rho*uy0 - 6*rho*uz0)/(45*rho);
    m_zz_t45 = -(2*(rho + 12*m_xx_I*rho_I + 45*m_xy_I*rho_I - 18*m_xz_I*rho_I + 12*m_yy_I*rho_I + 18*m_yz_I*rho_I - 87*m_zz_I*rho_I + 6*rho*ux0 - 6*rho*uy0 + 12*rho*uz0))/(135*rho);

    rhoVar = rho;

    break;
case 0b0000010110111010:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[9] + pop[11] + pop[12] + pop[13] + pop[15] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[9] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[15] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] - pop[13]);
    m_xz_I = inv_rho_I * (pop[9] - pop[15]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] - (1.0/3.0)*pop[9] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] - (1.0/3.0)*pop[15] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[18]);

    rho = -(72*(107*rho_I + 72*m_xx_I*rho_I + 12*m_yy_I*rho_I - 33*m_yz_I*rho_I + 12*m_zz_I*rho_I - 72*omegaVar*m_xx_I*rho_I - 12*omegaVar*m_yy_I*rho_I + 33*omegaVar*m_yz_I*rho_I - 12*omegaVar*m_zz_I*rho_I))/(2124*ux0 - 626*omegaVar - 492*uy0 + 492*uz0 + 1728*omegaVar*ux0 - 150*omegaVar*uy0 + 150*omegaVar*uz0 + 3531*omegaVar*ux0*ux0 + 642*omegaVar*uy0*uy0 + 642*omegaVar*uz0*uz0 - 1926*omegaVar*uy0*uz0 - 5580);

    m_xx_t45 = (2*(24*rho + 249*m_xx_I*rho_I - 12*m_yy_I*rho_I + 33*m_yz_I*rho_I - 12*m_zz_I*rho_I - 83*rho*ux0 - 11*rho*uy0 + 11*rho*uz0))/(321*rho);
    m_xy_t90 = (6*m_xy_I*rho_I - rho*uy0)/(3*rho);
    m_xz_t90 = (6*m_xz_I*rho_I - rho*uz0)/(3*rho);
    m_yy_t45 = (4*(2*rho - 6*m_xx_I*rho_I + 106*m_yy_I*rho_I - 24*m_yz_I*rho_I - m_zz_I*rho_I + 2*rho*ux0 + 8*rho*uy0 - 8*rho*uz0))/(321*rho);
    m_yz_t90 = -(11*rho - 33*m_xx_I*rho_I + 48*m_yy_I*rho_I - 453*m_yz_I*rho_I + 48*m_zz_I*rho_I + 11*rho*ux0 + 44*rho*uy0 - 44*rho*uz0)/(321*rho);
    m_zz_t45 = (4*(2*rho - 6*m_xx_I*rho_I - m_yy_I*rho_I - 24*m_yz_I*rho_I + 106*m_zz_I*rho_I + 2*rho*ux0 + 8*rho*uy0 - 8*rho*uz0))/(321*rho);

    rhoVar = rho;

    break;
case 0b0000010110111011:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[4] + pop[5] + pop[6] + pop[9] + pop[12] + pop[13] + pop[15] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[9] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[15] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (-pop[13]);
    m_xz_I = inv_rho_I * (pop[9] - pop[15]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] - (1.0/3.0)*pop[9] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] - (1.0/3.0)*pop[15] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[12] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[18]);

    rho = -(72*(23*rho_I + 24*m_xx_I*rho_I + 57*m_xy_I*rho_I + 24*m_yy_I*rho_I - 6*m_zz_I*rho_I - 24*omegaVar*m_xx_I*rho_I - 57*omegaVar*m_xy_I*rho_I - 24*omegaVar*m_yy_I*rho_I + 6*omegaVar*m_zz_I*rho_I))/(660*ux0 - 430*omegaVar - 660*uy0 + 30*omegaVar*ux0 - 30*omegaVar*uy0 + 690*omegaVar*ux0*ux0 + 690*omegaVar*uy0*uy0 + 69*omegaVar*uz0*uz0 + 414*omegaVar*ux0*uy0 - 720);

    m_xx_t45 = (16*rho + 141*m_xx_I*rho_I + 102*m_xy_I*rho_I + 3*m_yy_I*rho_I - 18*m_zz_I*rho_I - 30*rho*ux0 - 16*rho*uy0)/(69*rho);
    m_xy_t90 = (19*rho + 51*m_xx_I*rho_I + 354*m_xy_I*rho_I + 51*m_yy_I*rho_I - 30*m_zz_I*rho_I + 19*rho*ux0 - 19*rho*uy0)/(69*rho);
    m_xz_t90 = (6*m_xz_I*rho_I - rho*uz0)/(3*rho);
    m_yy_t45 = (16*rho + 3*m_xx_I*rho_I + 102*m_xy_I*rho_I + 141*m_yy_I*rho_I - 18*m_zz_I*rho_I + 16*rho*ux0 + 30*rho*uy0)/(69*rho);
    m_yz_t90 = (6*m_yz_I*rho_I + rho*uz0)/(3*rho);
    m_zz_t45 = -(2*(2*rho + 9*m_xx_I*rho_I + 30*m_xy_I*rho_I + 9*m_yy_I*rho_I - 54*m_zz_I*rho_I + 2*rho*ux0 - 2*rho*uy0))/(69*rho);

    rhoVar = rho;

    break;
case 0b0000010110111100:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[9] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[9] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16]);
    m_xy_I = inv_rho_I * (pop[7] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] - (1.0/3.0)*pop[9] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16]);

    rho = -(24*(15*rho_I - m_xx_I*rho_I + 6*m_xy_I*rho_I + 6*m_xz_I*rho_I + 11*m_yy_I*rho_I - 30*m_yz_I*rho_I + 11*m_zz_I*rho_I + omegaVar*m_xx_I*rho_I - 6*omegaVar*m_xy_I*rho_I - 6*omegaVar*m_xz_I*rho_I - 11*omegaVar*m_yy_I*rho_I + 30*omegaVar*m_yz_I*rho_I - 11*omegaVar*m_zz_I*rho_I))/(24*ux0 - 72*omegaVar + 12*uy0 + 12*uz0 + 36*omegaVar*ux0 + 18*omegaVar*uy0 + 18*omegaVar*uz0 + 30*omegaVar*ux0*ux0 + 75*omegaVar*uy0*uy0 + 75*omegaVar*uz0*uz0 + 90*omegaVar*ux0*uy0 + 90*omegaVar*ux0*uz0 - 180*omegaVar*uy0*uz0 - 248);

    m_xx_t45 = -(2*(rho - 87*m_xx_I*rho_I - 18*m_xy_I*rho_I - 18*m_xz_I*rho_I + 12*m_yy_I*rho_I - 45*m_yz_I*rho_I + 12*m_zz_I*rho_I + 12*rho*ux0 + 6*rho*uy0 + 6*rho*uz0))/(135*rho);
    m_xy_t90 = (2*rho + 6*m_xx_I*rho_I + 63*m_xy_I*rho_I + 6*m_yy_I*rho_I - 3*m_zz_I*rho_I - 6*rho*ux0 - 6*rho*uy0)/(45*rho);
    m_xz_t90 = (2*rho + 6*m_xx_I*rho_I + 63*m_xz_I*rho_I - 3*m_yy_I*rho_I + 6*m_zz_I*rho_I - 6*rho*ux0 - 6*rho*uz0)/(45*rho);
    m_yy_t45 = (2*(11*rho - 12*m_xx_I*rho_I + 18*m_xy_I*rho_I - 9*m_xz_I*rho_I + 105*m_yy_I*rho_I - 90*m_yz_I*rho_I + 24*m_zz_I*rho_I - 3*rho*ux0 - 6*rho*uy0 + 3*rho*uz0))/(135*rho);
    m_yz_t90 = -(2*rho - 3*m_xx_I*rho_I + 6*m_yy_I*rho_I - 27*m_yz_I*rho_I + 6*m_zz_I*rho_I)/(9*rho);
    m_zz_t45 = (2*(11*rho - 12*m_xx_I*rho_I - 9*m_xy_I*rho_I + 18*m_xz_I*rho_I + 24*m_yy_I*rho_I - 90*m_yz_I*rho_I + 105*m_zz_I*rho_I - 3*rho*ux0 + 3*rho*uy0 - 6*rho*uz0))/(135*rho);

    rhoVar = rho;

    break;
case 0b0000010110111101:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16]);
    m_xy_I = inv_rho_I * (-pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (-pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16]);

    rho = -(12*(rho_I + m_xx_I*rho_I + 2*m_xy_I*rho_I + 2*m_xz_I*rho_I + m_yy_I*rho_I - 2*m_yz_I*rho_I + m_zz_I*rho_I - omegaVar*m_xx_I*rho_I - 2*omegaVar*m_xy_I*rho_I - 2*omegaVar*m_xz_I*rho_I - omegaVar*m_yy_I*rho_I + 2*omegaVar*m_yz_I*rho_I - omegaVar*m_zz_I*rho_I))/(3*omegaVar*ux0*ux0 - 6*omegaVar + 3*omegaVar*uy0*uy0 + 3*omegaVar*uz0*uz0 + 6*omegaVar*ux0*uy0 + 6*omegaVar*ux0*uz0 - 6*omegaVar*uy0*uz0 - 4);

    m_xx_t45 = (2*(rho + 9*m_xx_I*rho_I + 6*m_xy_I*rho_I + 6*m_xz_I*rho_I + 3*m_yz_I*rho_I))/(9*rho);
    m_xy_t90 = (2*rho + 6*m_xx_I*rho_I + 27*m_xy_I*rho_I + 6*m_yy_I*rho_I - 3*m_zz_I*rho_I)/(9*rho);
    m_xz_t90 = (2*rho + 6*m_xx_I*rho_I + 27*m_xz_I*rho_I - 3*m_yy_I*rho_I + 6*m_zz_I*rho_I)/(9*rho);
    m_yy_t45 = (2*(rho + 6*m_xy_I*rho_I - 3*m_xz_I*rho_I + 9*m_yy_I*rho_I - 6*m_yz_I*rho_I))/(9*rho);
    m_yz_t90 = -(2*rho - 3*m_xx_I*rho_I + 6*m_yy_I*rho_I - 27*m_yz_I*rho_I + 6*m_zz_I*rho_I)/(9*rho);
    m_zz_t45 = (2*(rho - 3*m_xy_I*rho_I + 6*m_xz_I*rho_I - 6*m_yz_I*rho_I + 9*m_zz_I*rho_I))/(9*rho);

    rhoVar = rho;

    break;
case 0b0000010110111110:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[9] + pop[11] + pop[12] + pop[13] + pop[15];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[9] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[15]);
    m_xy_I = inv_rho_I * (pop[7] - pop[13]);
    m_xz_I = inv_rho_I * (pop[9] - pop[15]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] - (1.0/3.0)*pop[9] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] - (1.0/3.0)*pop[15]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] + (2.0/3.0)*pop[15]);

    rho = -(36*(19*rho_I + 6*m_xx_I*rho_I + 12*m_yy_I*rho_I - 33*m_yz_I*rho_I + 12*m_zz_I*rho_I - 6*omegaVar*m_xx_I*rho_I - 12*omegaVar*m_yy_I*rho_I + 33*omegaVar*m_yz_I*rho_I - 12*omegaVar*m_zz_I*rho_I))/(270*ux0 - 118*omegaVar + 72*omegaVar*ux0 + 285*omegaVar*ux0*ux0 + 114*omegaVar*uy0*uy0 + 114*omegaVar*uz0*uz0 - 342*omegaVar*uy0*uz0 - 414);

    m_xx_t45 = (2*(2*rho + 51*m_xx_I*rho_I - 12*m_yy_I*rho_I + 33*m_yz_I*rho_I - 12*m_zz_I*rho_I - 17*rho*ux0))/(57*rho);
    m_xy_t90 = (6*m_xy_I*rho_I - rho*uy0)/(3*rho);
    m_xz_t90 = (6*m_xz_I*rho_I - rho*uz0)/(3*rho);
    m_yy_t45 = (4*(2*rho - 6*m_xx_I*rho_I + 26*m_yy_I*rho_I - 24*m_yz_I*rho_I + 7*m_zz_I*rho_I + 2*rho*ux0))/(57*rho);
    m_yz_t90 = -(11*rho - 33*m_xx_I*rho_I + 48*m_yy_I*rho_I - 189*m_yz_I*rho_I + 48*m_zz_I*rho_I + 11*rho*ux0)/(57*rho);
    m_zz_t45 = (4*(2*rho - 6*m_xx_I*rho_I + 7*m_yy_I*rho_I - 24*m_yz_I*rho_I + 26*m_zz_I*rho_I + 2*rho*ux0))/(57*rho);

    rhoVar = rho;

    break;
case 0b0000010110111111:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[4] + pop[6] + pop[12] + pop[13] + pop[15];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[6] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[15]);
    m_xy_I = inv_rho_I * (-pop[13]);
    m_xz_I = inv_rho_I * (-pop[15]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] - (1.0/3.0)*pop[15]);
    m_yz_I = inv_rho_I * (pop[12]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[6] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] + (2.0/3.0)*pop[15]);

    rho = -(24*(rho_I + m_xx_I*rho_I + 2*m_xy_I*rho_I + 2*m_xz_I*rho_I + m_yy_I*rho_I - 2*m_yz_I*rho_I + m_zz_I*rho_I - omegaVar*m_xx_I*rho_I - 2*omegaVar*m_xy_I*rho_I - 2*omegaVar*m_xz_I*rho_I - omegaVar*m_yy_I*rho_I + 2*omegaVar*m_yz_I*rho_I - omegaVar*m_zz_I*rho_I))/(12*ux0 - 10*omegaVar - 12*uy0 - 12*uz0 - 4*omegaVar*ux0 + 4*omegaVar*uy0 + 4*omegaVar*uz0 + 9*omegaVar*ux0*ux0 + 9*omegaVar*uy0*uy0 + 9*omegaVar*uz0*uz0 + 6*omegaVar*ux0*uy0 + 6*omegaVar*ux0*uz0 - 6*omegaVar*uy0*uz0 - 4);

    m_xx_t45 = (2*(rho + 15*m_xx_I*rho_I + 9*m_xy_I*rho_I + 9*m_xz_I*rho_I - 3*m_yy_I*rho_I + 9*m_yz_I*rho_I - 3*m_zz_I*rho_I - 2*rho*ux0 - rho*uy0 - rho*uz0))/(9*rho);
    m_xy_t90 = (2*rho + 9*m_xx_I*rho_I + 51*m_xy_I*rho_I - 3*m_xz_I*rho_I + 9*m_yy_I*rho_I + 3*m_yz_I*rho_I - 9*m_zz_I*rho_I + 2*rho*ux0 - 2*rho*uy0 - 2*rho*uz0)/(9*rho);
    m_xz_t90 = (2*rho + 9*m_xx_I*rho_I - 3*m_xy_I*rho_I + 51*m_xz_I*rho_I - 9*m_yy_I*rho_I + 3*m_yz_I*rho_I + 9*m_zz_I*rho_I + 2*rho*ux0 - 2*rho*uy0 - 2*rho*uz0)/(9*rho);
    m_yy_t45 = (2*(rho - 3*m_xx_I*rho_I + 9*m_xy_I*rho_I - 9*m_xz_I*rho_I + 15*m_yy_I*rho_I - 9*m_yz_I*rho_I - 3*m_zz_I*rho_I + rho*ux0 + 2*rho*uy0 - rho*uz0))/(9*rho);
    m_yz_t90 = (9*m_xx_I*rho_I - 2*rho + 3*m_xy_I*rho_I + 3*m_xz_I*rho_I - 9*m_yy_I*rho_I + 51*m_yz_I*rho_I - 9*m_zz_I*rho_I - 2*rho*ux0 + 2*rho*uy0 + 2*rho*uz0)/(9*rho);
    m_zz_t45 = (2*(rho - 3*m_xx_I*rho_I - 9*m_xy_I*rho_I + 9*m_xz_I*rho_I - 3*m_yy_I*rho_I - 9*m_yz_I*rho_I + 15*m_zz_I*rho_I + rho*ux0 - rho*uy0 + 2*rho*uz0))/(9*rho);

    rhoVar = rho;

    break;
case 0b0000010111000000:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = -(24*(15*rho_I - m_xx_I*rho_I + 2*m_yy_I*rho_I + 6*m_yz_I*rho_I + 2*m_zz_I*rho_I + omegaVar*m_xx_I*rho_I - 2*omegaVar*m_yy_I*rho_I - 6*omegaVar*m_yz_I*rho_I - 2*omegaVar*m_zz_I*rho_I))/(12*uy0 - 6*omegaVar + 12*uz0 + 18*omegaVar*uy0 + 18*omegaVar*uz0 - 15*omegaVar*ux0*ux0 + 30*omegaVar*uy0*uy0 + 30*omegaVar*uz0*uz0 + 90*omegaVar*uy0*uz0 - 344);

    m_xx_t45 = -(2*(rho - 69*m_xx_I*rho_I + 3*m_yy_I*rho_I + 9*m_yz_I*rho_I + 3*m_zz_I*rho_I - 3*rho*uy0 - 3*rho*uz0))/(135*rho);
    m_xy_t90 = (m_xy_I*rho_I)/rho;
    m_xz_t90 = (m_xz_I*rho_I)/rho;
    m_yy_t45 = (4*rho - 6*m_xx_I*rho_I + 147*m_yy_I*rho_I + 36*m_yz_I*rho_I + 12*m_zz_I*rho_I - 12*rho*uy0 - 12*rho*uz0)/(135*rho);
    m_yz_t90 = (2*rho - 3*m_xx_I*rho_I + 6*m_yy_I*rho_I + 63*m_yz_I*rho_I + 6*m_zz_I*rho_I - 6*rho*uy0 - 6*rho*uz0)/(45*rho);
    m_zz_t45 = (4*rho - 6*m_xx_I*rho_I + 12*m_yy_I*rho_I + 36*m_yz_I*rho_I + 147*m_zz_I*rho_I - 12*rho*uy0 - 12*rho*uz0)/(135*rho);

    rhoVar = rho;

    break;
case 0b0000010111000001:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = -(24*(15*rho_I - m_xx_I*rho_I + 2*m_yy_I*rho_I + 6*m_yz_I*rho_I + 2*m_zz_I*rho_I + omegaVar*m_xx_I*rho_I - 2*omegaVar*m_yy_I*rho_I - 6*omegaVar*m_yz_I*rho_I - 2*omegaVar*m_zz_I*rho_I))/(12*uy0 - 6*omegaVar + 12*uz0 + 18*omegaVar*uy0 + 18*omegaVar*uz0 - 15*omegaVar*ux0*ux0 + 30*omegaVar*uy0*uy0 + 30*omegaVar*uz0*uz0 + 90*omegaVar*uy0*uz0 - 344);

    m_xx_t45 = -(2*(rho - 69*m_xx_I*rho_I + 3*m_yy_I*rho_I + 9*m_yz_I*rho_I + 3*m_zz_I*rho_I - 3*rho*uy0 - 3*rho*uz0))/(135*rho);
    m_xy_t90 = (m_xy_I*rho_I)/rho;
    m_xz_t90 = (m_xz_I*rho_I)/rho;
    m_yy_t45 = (4*rho - 6*m_xx_I*rho_I + 147*m_yy_I*rho_I + 36*m_yz_I*rho_I + 12*m_zz_I*rho_I - 12*rho*uy0 - 12*rho*uz0)/(135*rho);
    m_yz_t90 = (2*rho - 3*m_xx_I*rho_I + 6*m_yy_I*rho_I + 63*m_yz_I*rho_I + 6*m_zz_I*rho_I - 6*rho*uy0 - 6*rho*uz0)/(45*rho);
    m_zz_t45 = (4*rho - 6*m_xx_I*rho_I + 12*m_yy_I*rho_I + 36*m_yz_I*rho_I + 147*m_zz_I*rho_I - 12*rho*uy0 - 12*rho*uz0)/(135*rho);

    rhoVar = rho;

    break;
case 0b0000010111000010:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = -(24*(15*rho_I - m_xx_I*rho_I + 2*m_yy_I*rho_I + 6*m_yz_I*rho_I + 2*m_zz_I*rho_I + omegaVar*m_xx_I*rho_I - 2*omegaVar*m_yy_I*rho_I - 6*omegaVar*m_yz_I*rho_I - 2*omegaVar*m_zz_I*rho_I))/(12*uy0 - 6*omegaVar + 12*uz0 + 18*omegaVar*uy0 + 18*omegaVar*uz0 - 15*omegaVar*ux0*ux0 + 30*omegaVar*uy0*uy0 + 30*omegaVar*uz0*uz0 + 90*omegaVar*uy0*uz0 - 344);

    m_xx_t45 = -(2*(rho - 69*m_xx_I*rho_I + 3*m_yy_I*rho_I + 9*m_yz_I*rho_I + 3*m_zz_I*rho_I - 3*rho*uy0 - 3*rho*uz0))/(135*rho);
    m_xy_t90 = (m_xy_I*rho_I)/rho;
    m_xz_t90 = (m_xz_I*rho_I)/rho;
    m_yy_t45 = (4*rho - 6*m_xx_I*rho_I + 147*m_yy_I*rho_I + 36*m_yz_I*rho_I + 12*m_zz_I*rho_I - 12*rho*uy0 - 12*rho*uz0)/(135*rho);
    m_yz_t90 = (2*rho - 3*m_xx_I*rho_I + 6*m_yy_I*rho_I + 63*m_yz_I*rho_I + 6*m_zz_I*rho_I - 6*rho*uy0 - 6*rho*uz0)/(45*rho);
    m_zz_t45 = (4*rho - 6*m_xx_I*rho_I + 12*m_yy_I*rho_I + 36*m_yz_I*rho_I + 147*m_zz_I*rho_I - 12*rho*uy0 - 12*rho*uz0)/(135*rho);

    rhoVar = rho;

    break;
case 0b0000010111000011:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (-pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (12*(3*rho_I - m_xx_I*rho_I + 2*m_yy_I*rho_I + 6*m_yz_I*rho_I + 2*m_zz_I*rho_I + omegaVar*m_xx_I*rho_I - 2*omegaVar*m_yy_I*rho_I - 6*omegaVar*m_yz_I*rho_I - 2*omegaVar*m_zz_I*rho_I))/(6*omegaVar + 3*omegaVar*ux0*ux0 - 6*omegaVar*uy0*uy0 - 6*omegaVar*uz0*uz0 - 18*omegaVar*uy0*uz0 + 28);

    m_xx_t45 = -(2*(rho - 15*m_xx_I*rho_I + 3*m_yy_I*rho_I + 9*m_yz_I*rho_I + 3*m_zz_I*rho_I))/(27*rho);
    m_xy_t90 = (m_xy_I*rho_I)/rho;
    m_xz_t90 = (m_xz_I*rho_I)/rho;
    m_yy_t45 = (4*rho - 6*m_xx_I*rho_I + 39*m_yy_I*rho_I + 36*m_yz_I*rho_I + 12*m_zz_I*rho_I)/(27*rho);
    m_yz_t90 = (2*rho - 3*m_xx_I*rho_I + 6*m_yy_I*rho_I + 27*m_yz_I*rho_I + 6*m_zz_I*rho_I)/(9*rho);
    m_zz_t45 = (4*rho - 6*m_xx_I*rho_I + 12*m_yy_I*rho_I + 36*m_yz_I*rho_I + 39*m_zz_I*rho_I)/(27*rho);

    rhoVar = rho;

    break;
case 0b0000010111000100:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = -(24*(15*rho_I + m_xx_I*rho_I - 6*m_xy_I*rho_I + 4*m_yy_I*rho_I + 6*m_yz_I*rho_I + m_zz_I*rho_I - omegaVar*m_xx_I*rho_I + 6*omegaVar*m_xy_I*rho_I - 4*omegaVar*m_yy_I*rho_I - 6*omegaVar*m_yz_I*rho_I - omegaVar*m_zz_I*rho_I))/(24*uy0 - 12*ux0 - 12*omegaVar + 12*uz0 - 18*omegaVar*ux0 + 36*omegaVar*uy0 + 18*omegaVar*uz0 + 15*omegaVar*ux0*ux0 + 60*omegaVar*uy0*uy0 + 15*omegaVar*uz0*uz0 - 90*omegaVar*ux0*uy0 + 90*omegaVar*uy0*uz0 - 328);

    m_xx_t45 = (2*(rho + 75*m_xx_I*rho_I - 18*m_xy_I*rho_I + 3*m_yy_I*rho_I - 9*m_yz_I*rho_I - 6*m_zz_I*rho_I + 6*rho*ux0 - 3*rho*uy0 + 3*rho*uz0))/(135*rho);
    m_xy_t90 = -(2*rho + 6*m_xx_I*rho_I - 63*m_xy_I*rho_I + 6*m_yy_I*rho_I - 3*m_zz_I*rho_I + 6*rho*ux0 - 6*rho*uy0)/(45*rho);
    m_xz_t90 = (m_xz_I*rho_I)/rho;
    m_yy_t45 = (8*rho + 6*m_xx_I*rho_I - 36*m_xy_I*rho_I + 159*m_yy_I*rho_I + 36*m_yz_I*rho_I + 6*m_zz_I*rho_I + 12*rho*ux0 - 24*rho*uy0 - 12*rho*uz0)/(135*rho);
    m_yz_t90 = (2*rho - 3*m_xx_I*rho_I + 6*m_yy_I*rho_I + 63*m_yz_I*rho_I + 6*m_zz_I*rho_I - 6*rho*uy0 - 6*rho*uz0)/(45*rho);
    m_zz_t45 = (2*(rho - 6*m_xx_I*rho_I + 9*m_xy_I*rho_I + 3*m_yy_I*rho_I + 18*m_yz_I*rho_I + 75*m_zz_I*rho_I - 3*rho*ux0 - 3*rho*uy0 - 6*rho*uz0))/(135*rho);

    rhoVar = rho;

    break;
case 0b0000010111000101:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[10] + pop[11] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[14]);
    m_xz_I = inv_rho_I * (pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = -(24*(5*rho_I + m_xx_I*rho_I - 2*m_xy_I*rho_I + 2*m_xz_I*rho_I + m_yy_I*rho_I + 2*m_yz_I*rho_I + m_zz_I*rho_I - omegaVar*m_xx_I*rho_I + 2*omegaVar*m_xy_I*rho_I - 2*omegaVar*m_xz_I*rho_I - omegaVar*m_yy_I*rho_I - 2*omegaVar*m_yz_I*rho_I - omegaVar*m_zz_I*rho_I))/(8*uy0 - 8*ux0 - 6*omegaVar - 12*omegaVar*ux0 + 12*omegaVar*uy0 + 15*omegaVar*ux0*ux0 + 15*omegaVar*uy0*uy0 + 15*omegaVar*uz0*uz0 - 30*omegaVar*ux0*uy0 + 30*omegaVar*ux0*uz0 + 30*omegaVar*uy0*uz0 - 104);

    m_xx_t45 = (2*(rho + 27*m_xx_I*rho_I - 6*m_xy_I*rho_I + 6*m_xz_I*rho_I - 3*m_yz_I*rho_I + 4*rho*ux0 - rho*uy0 + 3*rho*uz0))/(45*rho);
    m_xy_t90 = -(2*rho + 6*m_xx_I*rho_I - 63*m_xy_I*rho_I + 6*m_yy_I*rho_I - 3*m_zz_I*rho_I + 6*rho*ux0 - 6*rho*uy0)/(45*rho);
    m_xz_t90 = (2*rho + 6*m_xx_I*rho_I + 63*m_xz_I*rho_I - 3*m_yy_I*rho_I + 6*m_zz_I*rho_I + 6*rho*ux0 + 6*rho*uz0)/(45*rho);
    m_yy_t45 = (2*(rho - 6*m_xy_I*rho_I - 3*m_xz_I*rho_I + 27*m_yy_I*rho_I + 6*m_yz_I*rho_I + rho*ux0 - 4*rho*uy0 - 3*rho*uz0))/(45*rho);
    m_yz_t90 = (2*rho - 3*m_xx_I*rho_I + 6*m_yy_I*rho_I + 63*m_yz_I*rho_I + 6*m_zz_I*rho_I - 6*rho*uy0 - 6*rho*uz0)/(45*rho);
    m_zz_t45 = (2*(rho + 3*m_xy_I*rho_I + 6*m_xz_I*rho_I + 6*m_yz_I*rho_I + 27*m_zz_I*rho_I + rho*ux0 - rho*uy0))/(45*rho);

    rhoVar = rho;

    break;
case 0b0000010111000110:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = -(24*(15*rho_I + m_xx_I*rho_I - 6*m_xy_I*rho_I + 4*m_yy_I*rho_I + 6*m_yz_I*rho_I + m_zz_I*rho_I - omegaVar*m_xx_I*rho_I + 6*omegaVar*m_xy_I*rho_I - 4*omegaVar*m_yy_I*rho_I - 6*omegaVar*m_yz_I*rho_I - omegaVar*m_zz_I*rho_I))/(24*uy0 - 12*ux0 - 12*omegaVar + 12*uz0 - 18*omegaVar*ux0 + 36*omegaVar*uy0 + 18*omegaVar*uz0 + 15*omegaVar*ux0*ux0 + 60*omegaVar*uy0*uy0 + 15*omegaVar*uz0*uz0 - 90*omegaVar*ux0*uy0 + 90*omegaVar*uy0*uz0 - 328);

    m_xx_t45 = (2*(rho + 75*m_xx_I*rho_I - 18*m_xy_I*rho_I + 3*m_yy_I*rho_I - 9*m_yz_I*rho_I - 6*m_zz_I*rho_I + 6*rho*ux0 - 3*rho*uy0 + 3*rho*uz0))/(135*rho);
    m_xy_t90 = -(2*rho + 6*m_xx_I*rho_I - 63*m_xy_I*rho_I + 6*m_yy_I*rho_I - 3*m_zz_I*rho_I + 6*rho*ux0 - 6*rho*uy0)/(45*rho);
    m_xz_t90 = (m_xz_I*rho_I)/rho;
    m_yy_t45 = (8*rho + 6*m_xx_I*rho_I - 36*m_xy_I*rho_I + 159*m_yy_I*rho_I + 36*m_yz_I*rho_I + 6*m_zz_I*rho_I + 12*rho*ux0 - 24*rho*uy0 - 12*rho*uz0)/(135*rho);
    m_yz_t90 = (2*rho - 3*m_xx_I*rho_I + 6*m_yy_I*rho_I + 63*m_yz_I*rho_I + 6*m_zz_I*rho_I - 6*rho*uy0 - 6*rho*uz0)/(45*rho);
    m_zz_t45 = (2*(rho - 6*m_xx_I*rho_I + 9*m_xy_I*rho_I + 3*m_yy_I*rho_I + 18*m_yz_I*rho_I + 75*m_zz_I*rho_I - 3*rho*ux0 - 3*rho*uy0 - 6*rho*uz0))/(135*rho);

    rhoVar = rho;

    break;
case 0b0000010111000111:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[10] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[14]);
    m_xz_I = inv_rho_I * (pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (-pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (24*(15*rho_I - m_xx_I*rho_I - 6*m_xy_I*rho_I + 6*m_xz_I*rho_I + 11*m_yy_I*rho_I + 30*m_yz_I*rho_I + 11*m_zz_I*rho_I + omegaVar*m_xx_I*rho_I + 6*omegaVar*m_xy_I*rho_I - 6*omegaVar*m_xz_I*rho_I - 11*omegaVar*m_yy_I*rho_I - 30*omegaVar*m_yz_I*rho_I - 11*omegaVar*m_zz_I*rho_I))/(72*omegaVar + 24*ux0 - 12*uy0 + 12*uz0 + 36*omegaVar*ux0 - 18*omegaVar*uy0 + 18*omegaVar*uz0 - 30*omegaVar*ux0*ux0 - 75*omegaVar*uy0*uy0 - 75*omegaVar*uz0*uz0 + 90*omegaVar*ux0*uy0 - 90*omegaVar*ux0*uz0 - 180*omegaVar*uy0*uz0 + 248);

    m_xx_t45 = -(2*(rho - 87*m_xx_I*rho_I + 18*m_xy_I*rho_I - 18*m_xz_I*rho_I + 12*m_yy_I*rho_I + 45*m_yz_I*rho_I + 12*m_zz_I*rho_I - 12*rho*ux0 + 6*rho*uy0 - 6*rho*uz0))/(135*rho);
    m_xy_t90 = -(2*rho + 6*m_xx_I*rho_I - 63*m_xy_I*rho_I + 6*m_yy_I*rho_I - 3*m_zz_I*rho_I + 6*rho*ux0 - 6*rho*uy0)/(45*rho);
    m_xz_t90 = (2*rho + 6*m_xx_I*rho_I + 63*m_xz_I*rho_I - 3*m_yy_I*rho_I + 6*m_zz_I*rho_I + 6*rho*ux0 + 6*rho*uz0)/(45*rho);
    m_yy_t45 = (2*(11*rho - 12*m_xx_I*rho_I - 18*m_xy_I*rho_I - 9*m_xz_I*rho_I + 105*m_yy_I*rho_I + 90*m_yz_I*rho_I + 24*m_zz_I*rho_I + 3*rho*ux0 - 6*rho*uy0 - 3*rho*uz0))/(135*rho);
    m_yz_t90 = (2*rho - 3*m_xx_I*rho_I + 6*m_yy_I*rho_I + 27*m_yz_I*rho_I + 6*m_zz_I*rho_I)/(9*rho);
    m_zz_t45 = (2*(11*rho - 12*m_xx_I*rho_I + 9*m_xy_I*rho_I + 18*m_xz_I*rho_I + 24*m_yy_I*rho_I + 90*m_yz_I*rho_I + 105*m_zz_I*rho_I + 3*rho*ux0 + 3*rho*uy0 + 6*rho*uz0))/(135*rho);

    rhoVar = rho;

    break;
case 0b0000010111001000:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[9] + pop[10] + pop[11] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = -(24*(15*rho_I + m_xx_I*rho_I + 6*m_xy_I*rho_I + 4*m_yy_I*rho_I + 6*m_yz_I*rho_I + m_zz_I*rho_I - omegaVar*m_xx_I*rho_I - 6*omegaVar*m_xy_I*rho_I - 4*omegaVar*m_yy_I*rho_I - 6*omegaVar*m_yz_I*rho_I - omegaVar*m_zz_I*rho_I))/(12*ux0 - 12*omegaVar + 24*uy0 + 12*uz0 + 18*omegaVar*ux0 + 36*omegaVar*uy0 + 18*omegaVar*uz0 + 15*omegaVar*ux0*ux0 + 60*omegaVar*uy0*uy0 + 15*omegaVar*uz0*uz0 + 90*omegaVar*ux0*uy0 + 90*omegaVar*uy0*uz0 - 328);

    m_xx_t45 = (2*(rho + 75*m_xx_I*rho_I + 18*m_xy_I*rho_I + 3*m_yy_I*rho_I - 9*m_yz_I*rho_I - 6*m_zz_I*rho_I - 6*rho*ux0 - 3*rho*uy0 + 3*rho*uz0))/(135*rho);
    m_xy_t90 = (2*rho + 6*m_xx_I*rho_I + 63*m_xy_I*rho_I + 6*m_yy_I*rho_I - 3*m_zz_I*rho_I - 6*rho*ux0 - 6*rho*uy0)/(45*rho);
    m_xz_t90 = (m_xz_I*rho_I)/rho;
    m_yy_t45 = (8*rho + 6*m_xx_I*rho_I + 36*m_xy_I*rho_I + 159*m_yy_I*rho_I + 36*m_yz_I*rho_I + 6*m_zz_I*rho_I - 12*rho*ux0 - 24*rho*uy0 - 12*rho*uz0)/(135*rho);
    m_yz_t90 = (2*rho - 3*m_xx_I*rho_I + 6*m_yy_I*rho_I + 63*m_yz_I*rho_I + 6*m_zz_I*rho_I - 6*rho*uy0 - 6*rho*uz0)/(45*rho);
    m_zz_t45 = (2*(rho - 6*m_xx_I*rho_I - 9*m_xy_I*rho_I + 3*m_yy_I*rho_I + 18*m_yz_I*rho_I + 75*m_zz_I*rho_I + 3*rho*ux0 - 3*rho*uy0 - 6*rho*uz0))/(135*rho);

    rhoVar = rho;

    break;
case 0b0000010111001001:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[9] + pop[10] + pop[11] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = -(24*(15*rho_I + m_xx_I*rho_I + 6*m_xy_I*rho_I + 4*m_yy_I*rho_I + 6*m_yz_I*rho_I + m_zz_I*rho_I - omegaVar*m_xx_I*rho_I - 6*omegaVar*m_xy_I*rho_I - 4*omegaVar*m_yy_I*rho_I - 6*omegaVar*m_yz_I*rho_I - omegaVar*m_zz_I*rho_I))/(12*ux0 - 12*omegaVar + 24*uy0 + 12*uz0 + 18*omegaVar*ux0 + 36*omegaVar*uy0 + 18*omegaVar*uz0 + 15*omegaVar*ux0*ux0 + 60*omegaVar*uy0*uy0 + 15*omegaVar*uz0*uz0 + 90*omegaVar*ux0*uy0 + 90*omegaVar*uy0*uz0 - 328);

    m_xx_t45 = (2*(rho + 75*m_xx_I*rho_I + 18*m_xy_I*rho_I + 3*m_yy_I*rho_I - 9*m_yz_I*rho_I - 6*m_zz_I*rho_I - 6*rho*ux0 - 3*rho*uy0 + 3*rho*uz0))/(135*rho);
    m_xy_t90 = (2*rho + 6*m_xx_I*rho_I + 63*m_xy_I*rho_I + 6*m_yy_I*rho_I - 3*m_zz_I*rho_I - 6*rho*ux0 - 6*rho*uy0)/(45*rho);
    m_xz_t90 = (m_xz_I*rho_I)/rho;
    m_yy_t45 = (8*rho + 6*m_xx_I*rho_I + 36*m_xy_I*rho_I + 159*m_yy_I*rho_I + 36*m_yz_I*rho_I + 6*m_zz_I*rho_I - 12*rho*ux0 - 24*rho*uy0 - 12*rho*uz0)/(135*rho);
    m_yz_t90 = (2*rho - 3*m_xx_I*rho_I + 6*m_yy_I*rho_I + 63*m_yz_I*rho_I + 6*m_zz_I*rho_I - 6*rho*uy0 - 6*rho*uz0)/(45*rho);
    m_zz_t45 = (2*(rho - 6*m_xx_I*rho_I - 9*m_xy_I*rho_I + 3*m_yy_I*rho_I + 18*m_yz_I*rho_I + 75*m_zz_I*rho_I + 3*rho*ux0 - 3*rho*uy0 - 6*rho*uz0))/(135*rho);

    rhoVar = rho;

    break;
case 0b0000010111001010:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[9] + pop[10] + pop[11] + pop[13] + pop[14] + pop[15] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = -(24*(5*rho_I + m_xx_I*rho_I + 2*m_xy_I*rho_I - 2*m_xz_I*rho_I + m_yy_I*rho_I + 2*m_yz_I*rho_I + m_zz_I*rho_I - omegaVar*m_xx_I*rho_I - 2*omegaVar*m_xy_I*rho_I + 2*omegaVar*m_xz_I*rho_I - omegaVar*m_yy_I*rho_I - 2*omegaVar*m_yz_I*rho_I - omegaVar*m_zz_I*rho_I))/(8*ux0 - 6*omegaVar + 8*uy0 + 12*omegaVar*ux0 + 12*omegaVar*uy0 + 15*omegaVar*ux0*ux0 + 15*omegaVar*uy0*uy0 + 15*omegaVar*uz0*uz0 + 30*omegaVar*ux0*uy0 - 30*omegaVar*ux0*uz0 + 30*omegaVar*uy0*uz0 - 104);

    m_xx_t45 = (2*(rho + 27*m_xx_I*rho_I + 6*m_xy_I*rho_I - 6*m_xz_I*rho_I - 3*m_yz_I*rho_I - 4*rho*ux0 - rho*uy0 + 3*rho*uz0))/(45*rho);
    m_xy_t90 = (2*rho + 6*m_xx_I*rho_I + 63*m_xy_I*rho_I + 6*m_yy_I*rho_I - 3*m_zz_I*rho_I - 6*rho*ux0 - 6*rho*uy0)/(45*rho);
    m_xz_t90 = -(2*rho + 6*m_xx_I*rho_I - 63*m_xz_I*rho_I - 3*m_yy_I*rho_I + 6*m_zz_I*rho_I - 6*rho*ux0 + 6*rho*uz0)/(45*rho);
    m_yy_t45 = (2*(rho + 6*m_xy_I*rho_I + 3*m_xz_I*rho_I + 27*m_yy_I*rho_I + 6*m_yz_I*rho_I - rho*ux0 - 4*rho*uy0 - 3*rho*uz0))/(45*rho);
    m_yz_t90 = (2*rho - 3*m_xx_I*rho_I + 6*m_yy_I*rho_I + 63*m_yz_I*rho_I + 6*m_zz_I*rho_I - 6*rho*uy0 - 6*rho*uz0)/(45*rho);
    m_zz_t45 = -(2*(3*m_xy_I*rho_I - rho + 6*m_xz_I*rho_I - 6*m_yz_I*rho_I - 27*m_zz_I*rho_I + rho*ux0 + rho*uy0))/(45*rho);

    rhoVar = rho;

    break;
case 0b0000010111001011:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[9] + pop[10] + pop[13] + pop[14] + pop[15] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (-pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = -(24*(15*rho_I - m_xx_I*rho_I + 6*m_xy_I*rho_I - 6*m_xz_I*rho_I + 11*m_yy_I*rho_I + 30*m_yz_I*rho_I + 11*m_zz_I*rho_I + omegaVar*m_xx_I*rho_I - 6*omegaVar*m_xy_I*rho_I + 6*omegaVar*m_xz_I*rho_I - 11*omegaVar*m_yy_I*rho_I - 30*omegaVar*m_yz_I*rho_I - 11*omegaVar*m_zz_I*rho_I))/(24*ux0 - 72*omegaVar + 12*uy0 - 12*uz0 + 36*omegaVar*ux0 + 18*omegaVar*uy0 - 18*omegaVar*uz0 + 30*omegaVar*ux0*ux0 + 75*omegaVar*uy0*uy0 + 75*omegaVar*uz0*uz0 + 90*omegaVar*ux0*uy0 - 90*omegaVar*ux0*uz0 + 180*omegaVar*uy0*uz0 - 248);

    m_xx_t45 = -(2*(rho - 87*m_xx_I*rho_I - 18*m_xy_I*rho_I + 18*m_xz_I*rho_I + 12*m_yy_I*rho_I + 45*m_yz_I*rho_I + 12*m_zz_I*rho_I + 12*rho*ux0 + 6*rho*uy0 - 6*rho*uz0))/(135*rho);
    m_xy_t90 = (2*rho + 6*m_xx_I*rho_I + 63*m_xy_I*rho_I + 6*m_yy_I*rho_I - 3*m_zz_I*rho_I - 6*rho*ux0 - 6*rho*uy0)/(45*rho);
    m_xz_t90 = -(2*rho + 6*m_xx_I*rho_I - 63*m_xz_I*rho_I - 3*m_yy_I*rho_I + 6*m_zz_I*rho_I - 6*rho*ux0 + 6*rho*uz0)/(45*rho);
    m_yy_t45 = (2*(11*rho - 12*m_xx_I*rho_I + 18*m_xy_I*rho_I + 9*m_xz_I*rho_I + 105*m_yy_I*rho_I + 90*m_yz_I*rho_I + 24*m_zz_I*rho_I - 3*rho*ux0 - 6*rho*uy0 - 3*rho*uz0))/(135*rho);
    m_yz_t90 = (2*rho - 3*m_xx_I*rho_I + 6*m_yy_I*rho_I + 27*m_yz_I*rho_I + 6*m_zz_I*rho_I)/(9*rho);
    m_zz_t45 = (2*(11*rho - 12*m_xx_I*rho_I - 9*m_xy_I*rho_I - 18*m_xz_I*rho_I + 24*m_yy_I*rho_I + 90*m_yz_I*rho_I + 105*m_zz_I*rho_I - 3*rho*ux0 + 3*rho*uy0 + 6*rho*uz0))/(135*rho);

    rhoVar = rho;

    break;
case 0b0000010111001100:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[5] + pop[6] + pop[7] + pop[9] + pop[10] + pop[11] + pop[14] + pop[15] + pop[16] + pop[17];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17]);
    m_xy_I = inv_rho_I * (pop[7] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17]);
    m_yz_I = inv_rho_I * (pop[11] - pop[17]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17]);

    rho = -(3*rho_I*(3*m_yy_I - 3*omegaVar*m_yy_I + 4))/(3*uy0 - omegaVar + 3*omegaVar*uy0 + 6*omegaVar*uy0*uy0 - 9);

    m_xx_t45 = (4*(10*m_xx_I*rho_I - m_zz_I*rho_I))/(33*rho);
    m_xy_t90 = (6*m_xy_I*rho_I - rho*ux0)/(3*rho);
    m_xz_t90 = (m_xz_I*rho_I)/rho;
    m_yy_t45 = (rho + 9*m_yy_I*rho_I - 3*rho*uy0)/(6*rho);
    m_yz_t90 = (6*m_yz_I*rho_I - rho*uz0)/(3*rho);
    m_zz_t45 = -(4*(m_xx_I*rho_I - 10*m_zz_I*rho_I))/(33*rho);

    rhoVar = rho;

    break;
case 0b0000010111001101:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[5] + pop[6] + pop[7] + pop[10] + pop[11] + pop[14] + pop[15] + pop[16] + pop[17];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17]);
    m_xy_I = inv_rho_I * (pop[7] - pop[14]);
    m_xz_I = inv_rho_I * (pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17]);
    m_yz_I = inv_rho_I * (pop[11] - pop[17]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17]);

    rho = (72*(107*rho_I + 12*m_xx_I*rho_I + 33*m_xz_I*rho_I + 72*m_yy_I*rho_I + 12*m_zz_I*rho_I - 12*omegaVar*m_xx_I*rho_I - 33*omegaVar*m_xz_I*rho_I - 72*omegaVar*m_yy_I*rho_I - 12*omegaVar*m_zz_I*rho_I))/(626*omegaVar + 492*ux0 - 2124*uy0 + 492*uz0 + 150*omegaVar*ux0 - 1728*omegaVar*uy0 + 150*omegaVar*uz0 - 642*omegaVar*ux0*ux0 - 3531*omegaVar*uy0*uy0 - 642*omegaVar*uz0*uz0 - 1926*omegaVar*ux0*uz0 + 5580);

    m_xx_t45 = (4*(2*rho + 106*m_xx_I*rho_I + 24*m_xz_I*rho_I - 6*m_yy_I*rho_I - m_zz_I*rho_I + 8*rho*ux0 + 2*rho*uy0 + 8*rho*uz0))/(321*rho);
    m_xy_t90 = (6*m_xy_I*rho_I - rho*ux0)/(3*rho);
    m_xz_t90 = (11*rho + 48*m_xx_I*rho_I + 453*m_xz_I*rho_I - 33*m_yy_I*rho_I + 48*m_zz_I*rho_I + 44*rho*ux0 + 11*rho*uy0 + 44*rho*uz0)/(321*rho);
    m_yy_t45 = -(2*(12*m_xx_I*rho_I - 24*rho + 33*m_xz_I*rho_I - 249*m_yy_I*rho_I + 12*m_zz_I*rho_I + 11*rho*ux0 + 83*rho*uy0 + 11*rho*uz0))/(321*rho);
    m_yz_t90 = (6*m_yz_I*rho_I - rho*uz0)/(3*rho);
    m_zz_t45 = (4*(2*rho - m_xx_I*rho_I + 24*m_xz_I*rho_I - 6*m_yy_I*rho_I + 106*m_zz_I*rho_I + 8*rho*ux0 + 2*rho*uy0 + 8*rho*uz0))/(321*rho);

    rhoVar = rho;

    break;
case 0b0000010111001110:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[5] + pop[6] + pop[7] + pop[9] + pop[10] + pop[11] + pop[14] + pop[15] + pop[17];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] - (1.0/3.0)*pop[17]);
    m_xy_I = inv_rho_I * (pop[7] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] + (2.0/3.0)*pop[17]);
    m_yz_I = inv_rho_I * (pop[11] - pop[17]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[17]);

    rho = -(72*(107*rho_I + 12*m_xx_I*rho_I - 33*m_xz_I*rho_I + 72*m_yy_I*rho_I + 12*m_zz_I*rho_I - 12*omegaVar*m_xx_I*rho_I + 33*omegaVar*m_xz_I*rho_I - 72*omegaVar*m_yy_I*rho_I - 12*omegaVar*m_zz_I*rho_I))/(492*ux0 - 626*omegaVar + 2124*uy0 - 492*uz0 + 150*omegaVar*ux0 + 1728*omegaVar*uy0 - 150*omegaVar*uz0 + 642*omegaVar*ux0*ux0 + 3531*omegaVar*uy0*uy0 + 642*omegaVar*uz0*uz0 - 1926*omegaVar*ux0*uz0 - 5580);

    m_xx_t45 = (4*(2*rho + 106*m_xx_I*rho_I - 24*m_xz_I*rho_I - 6*m_yy_I*rho_I - m_zz_I*rho_I - 8*rho*ux0 + 2*rho*uy0 + 8*rho*uz0))/(321*rho);
    m_xy_t90 = (6*m_xy_I*rho_I - rho*ux0)/(3*rho);
    m_xz_t90 = -(11*rho + 48*m_xx_I*rho_I - 453*m_xz_I*rho_I - 33*m_yy_I*rho_I + 48*m_zz_I*rho_I - 44*rho*ux0 + 11*rho*uy0 + 44*rho*uz0)/(321*rho);
    m_yy_t45 = (2*(24*rho - 12*m_xx_I*rho_I + 33*m_xz_I*rho_I + 249*m_yy_I*rho_I - 12*m_zz_I*rho_I + 11*rho*ux0 - 83*rho*uy0 - 11*rho*uz0))/(321*rho);
    m_yz_t90 = (6*m_yz_I*rho_I - rho*uz0)/(3*rho);
    m_zz_t45 = (4*(2*rho - m_xx_I*rho_I - 24*m_xz_I*rho_I - 6*m_yy_I*rho_I + 106*m_zz_I*rho_I - 8*rho*ux0 + 2*rho*uy0 + 8*rho*uz0))/(321*rho);

    rhoVar = rho;

    break;
case 0b0000010111001111:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[6] + pop[7] + pop[10] + pop[14] + pop[15] + pop[17];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] - (1.0/3.0)*pop[17]);
    m_xy_I = inv_rho_I * (pop[7] - pop[14]);
    m_xz_I = inv_rho_I * (pop[10] - pop[15]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] + (2.0/3.0)*pop[17]);
    m_yz_I = inv_rho_I * (-pop[17]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[17]);

    rho = -(72*(23*rho_I - 6*m_xx_I*rho_I + 24*m_yy_I*rho_I + 57*m_yz_I*rho_I + 24*m_zz_I*rho_I + 6*omegaVar*m_xx_I*rho_I - 24*omegaVar*m_yy_I*rho_I - 57*omegaVar*m_yz_I*rho_I - 24*omegaVar*m_zz_I*rho_I))/(660*uy0 - 430*omegaVar - 660*uz0 + 30*omegaVar*uy0 - 30*omegaVar*uz0 + 69*omegaVar*ux0*ux0 + 690*omegaVar*uy0*uy0 + 690*omegaVar*uz0*uz0 + 414*omegaVar*uy0*uz0 - 720);

    m_xx_t45 = -(2*(2*rho - 54*m_xx_I*rho_I + 9*m_yy_I*rho_I + 30*m_yz_I*rho_I + 9*m_zz_I*rho_I + 2*rho*uy0 - 2*rho*uz0))/(69*rho);
    m_xy_t90 = (6*m_xy_I*rho_I - rho*ux0)/(3*rho);
    m_xz_t90 = (6*m_xz_I*rho_I + rho*ux0)/(3*rho);
    m_yy_t45 = (16*rho - 18*m_xx_I*rho_I + 141*m_yy_I*rho_I + 102*m_yz_I*rho_I + 3*m_zz_I*rho_I - 30*rho*uy0 - 16*rho*uz0)/(69*rho);
    m_yz_t90 = (19*rho - 30*m_xx_I*rho_I + 51*m_yy_I*rho_I + 354*m_yz_I*rho_I + 51*m_zz_I*rho_I + 19*rho*uy0 - 19*rho*uz0)/(69*rho);
    m_zz_t45 = (16*rho - 18*m_xx_I*rho_I + 3*m_yy_I*rho_I + 102*m_yz_I*rho_I + 141*m_zz_I*rho_I + 16*rho*uy0 + 30*rho*uz0)/(69*rho);

    rhoVar = rho;

    break;
case 0b0000010111010000:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[13] + pop[14] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = -(24*(15*rho_I + m_xx_I*rho_I - 6*m_xz_I*rho_I + m_yy_I*rho_I + 6*m_yz_I*rho_I + 4*m_zz_I*rho_I - omegaVar*m_xx_I*rho_I + 6*omegaVar*m_xz_I*rho_I - omegaVar*m_yy_I*rho_I - 6*omegaVar*m_yz_I*rho_I - 4*omegaVar*m_zz_I*rho_I))/(12*uy0 - 12*ux0 - 12*omegaVar + 24*uz0 - 18*omegaVar*ux0 + 18*omegaVar*uy0 + 36*omegaVar*uz0 + 15*omegaVar*ux0*ux0 + 15*omegaVar*uy0*uy0 + 60*omegaVar*uz0*uz0 - 90*omegaVar*ux0*uz0 + 90*omegaVar*uy0*uz0 - 328);

    m_xx_t45 = (2*(rho + 75*m_xx_I*rho_I - 18*m_xz_I*rho_I - 6*m_yy_I*rho_I - 9*m_yz_I*rho_I + 3*m_zz_I*rho_I + 6*rho*ux0 + 3*rho*uy0 - 3*rho*uz0))/(135*rho);
    m_xy_t90 = (m_xy_I*rho_I)/rho;
    m_xz_t90 = -(2*rho + 6*m_xx_I*rho_I - 63*m_xz_I*rho_I - 3*m_yy_I*rho_I + 6*m_zz_I*rho_I + 6*rho*ux0 - 6*rho*uz0)/(45*rho);
    m_yy_t45 = (2*(rho - 6*m_xx_I*rho_I + 9*m_xz_I*rho_I + 75*m_yy_I*rho_I + 18*m_yz_I*rho_I + 3*m_zz_I*rho_I - 3*rho*ux0 - 6*rho*uy0 - 3*rho*uz0))/(135*rho);
    m_yz_t90 = (2*rho - 3*m_xx_I*rho_I + 6*m_yy_I*rho_I + 63*m_yz_I*rho_I + 6*m_zz_I*rho_I - 6*rho*uy0 - 6*rho*uz0)/(45*rho);
    m_zz_t45 = (8*rho + 6*m_xx_I*rho_I - 36*m_xz_I*rho_I + 6*m_yy_I*rho_I + 36*m_yz_I*rho_I + 159*m_zz_I*rho_I + 12*rho*ux0 - 12*rho*uy0 - 24*rho*uz0)/(135*rho);

    rhoVar = rho;

    break;
case 0b0000010111010001:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[8] + pop[9] + pop[10] + pop[11] + pop[13] + pop[14] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = -(24*(5*rho_I + m_xx_I*rho_I + 2*m_xy_I*rho_I - 2*m_xz_I*rho_I + m_yy_I*rho_I + 2*m_yz_I*rho_I + m_zz_I*rho_I - omegaVar*m_xx_I*rho_I - 2*omegaVar*m_xy_I*rho_I + 2*omegaVar*m_xz_I*rho_I - omegaVar*m_yy_I*rho_I - 2*omegaVar*m_yz_I*rho_I - omegaVar*m_zz_I*rho_I))/(8*uz0 - 8*ux0 - 6*omegaVar - 12*omegaVar*ux0 + 12*omegaVar*uz0 + 15*omegaVar*ux0*ux0 + 15*omegaVar*uy0*uy0 + 15*omegaVar*uz0*uz0 + 30*omegaVar*ux0*uy0 - 30*omegaVar*ux0*uz0 + 30*omegaVar*uy0*uz0 - 104);

    m_xx_t45 = (2*(rho + 27*m_xx_I*rho_I + 6*m_xy_I*rho_I - 6*m_xz_I*rho_I - 3*m_yz_I*rho_I + 4*rho*ux0 + 3*rho*uy0 - rho*uz0))/(45*rho);
    m_xy_t90 = (2*rho + 6*m_xx_I*rho_I + 63*m_xy_I*rho_I + 6*m_yy_I*rho_I - 3*m_zz_I*rho_I + 6*rho*ux0 + 6*rho*uy0)/(45*rho);
    m_xz_t90 = -(2*rho + 6*m_xx_I*rho_I - 63*m_xz_I*rho_I - 3*m_yy_I*rho_I + 6*m_zz_I*rho_I + 6*rho*ux0 - 6*rho*uz0)/(45*rho);
    m_yy_t45 = (2*(rho + 6*m_xy_I*rho_I + 3*m_xz_I*rho_I + 27*m_yy_I*rho_I + 6*m_yz_I*rho_I + rho*ux0 - rho*uz0))/(45*rho);
    m_yz_t90 = (2*rho - 3*m_xx_I*rho_I + 6*m_yy_I*rho_I + 63*m_yz_I*rho_I + 6*m_zz_I*rho_I - 6*rho*uy0 - 6*rho*uz0)/(45*rho);
    m_zz_t45 = (2*(rho - 3*m_xy_I*rho_I - 6*m_xz_I*rho_I + 6*m_yz_I*rho_I + 27*m_zz_I*rho_I + rho*ux0 - 3*rho*uy0 - 4*rho*uz0))/(45*rho);

    rhoVar = rho;

    break;
case 0b0000010111010010:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[13] + pop[14] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = -(24*(15*rho_I + m_xx_I*rho_I - 6*m_xz_I*rho_I + m_yy_I*rho_I + 6*m_yz_I*rho_I + 4*m_zz_I*rho_I - omegaVar*m_xx_I*rho_I + 6*omegaVar*m_xz_I*rho_I - omegaVar*m_yy_I*rho_I - 6*omegaVar*m_yz_I*rho_I - 4*omegaVar*m_zz_I*rho_I))/(12*uy0 - 12*ux0 - 12*omegaVar + 24*uz0 - 18*omegaVar*ux0 + 18*omegaVar*uy0 + 36*omegaVar*uz0 + 15*omegaVar*ux0*ux0 + 15*omegaVar*uy0*uy0 + 60*omegaVar*uz0*uz0 - 90*omegaVar*ux0*uz0 + 90*omegaVar*uy0*uz0 - 328);

    m_xx_t45 = (2*(rho + 75*m_xx_I*rho_I - 18*m_xz_I*rho_I - 6*m_yy_I*rho_I - 9*m_yz_I*rho_I + 3*m_zz_I*rho_I + 6*rho*ux0 + 3*rho*uy0 - 3*rho*uz0))/(135*rho);
    m_xy_t90 = (m_xy_I*rho_I)/rho;
    m_xz_t90 = -(2*rho + 6*m_xx_I*rho_I - 63*m_xz_I*rho_I - 3*m_yy_I*rho_I + 6*m_zz_I*rho_I + 6*rho*ux0 - 6*rho*uz0)/(45*rho);
    m_yy_t45 = (2*(rho - 6*m_xx_I*rho_I + 9*m_xz_I*rho_I + 75*m_yy_I*rho_I + 18*m_yz_I*rho_I + 3*m_zz_I*rho_I - 3*rho*ux0 - 6*rho*uy0 - 3*rho*uz0))/(135*rho);
    m_yz_t90 = (2*rho - 3*m_xx_I*rho_I + 6*m_yy_I*rho_I + 63*m_yz_I*rho_I + 6*m_zz_I*rho_I - 6*rho*uy0 - 6*rho*uz0)/(45*rho);
    m_zz_t45 = (8*rho + 6*m_xx_I*rho_I - 36*m_xz_I*rho_I + 6*m_yy_I*rho_I + 36*m_yz_I*rho_I + 159*m_zz_I*rho_I + 12*rho*ux0 - 12*rho*uy0 - 24*rho*uz0)/(135*rho);

    rhoVar = rho;

    break;
case 0b0000010111010011:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[8] + pop[9] + pop[10] + pop[13] + pop[14] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (-pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (24*(15*rho_I - m_xx_I*rho_I + 6*m_xy_I*rho_I - 6*m_xz_I*rho_I + 11*m_yy_I*rho_I + 30*m_yz_I*rho_I + 11*m_zz_I*rho_I + omegaVar*m_xx_I*rho_I - 6*omegaVar*m_xy_I*rho_I + 6*omegaVar*m_xz_I*rho_I - 11*omegaVar*m_yy_I*rho_I - 30*omegaVar*m_yz_I*rho_I - 11*omegaVar*m_zz_I*rho_I))/(72*omegaVar + 24*ux0 + 12*uy0 - 12*uz0 + 36*omegaVar*ux0 + 18*omegaVar*uy0 - 18*omegaVar*uz0 - 30*omegaVar*ux0*ux0 - 75*omegaVar*uy0*uy0 - 75*omegaVar*uz0*uz0 - 90*omegaVar*ux0*uy0 + 90*omegaVar*ux0*uz0 - 180*omegaVar*uy0*uz0 + 248);

    m_xx_t45 = -(2*(rho - 87*m_xx_I*rho_I - 18*m_xy_I*rho_I + 18*m_xz_I*rho_I + 12*m_yy_I*rho_I + 45*m_yz_I*rho_I + 12*m_zz_I*rho_I - 12*rho*ux0 - 6*rho*uy0 + 6*rho*uz0))/(135*rho);
    m_xy_t90 = (2*rho + 6*m_xx_I*rho_I + 63*m_xy_I*rho_I + 6*m_yy_I*rho_I - 3*m_zz_I*rho_I + 6*rho*ux0 + 6*rho*uy0)/(45*rho);
    m_xz_t90 = -(2*rho + 6*m_xx_I*rho_I - 63*m_xz_I*rho_I - 3*m_yy_I*rho_I + 6*m_zz_I*rho_I + 6*rho*ux0 - 6*rho*uz0)/(45*rho);
    m_yy_t45 = (2*(11*rho - 12*m_xx_I*rho_I + 18*m_xy_I*rho_I + 9*m_xz_I*rho_I + 105*m_yy_I*rho_I + 90*m_yz_I*rho_I + 24*m_zz_I*rho_I + 3*rho*ux0 + 6*rho*uy0 + 3*rho*uz0))/(135*rho);
    m_yz_t90 = (2*rho - 3*m_xx_I*rho_I + 6*m_yy_I*rho_I + 27*m_yz_I*rho_I + 6*m_zz_I*rho_I)/(9*rho);
    m_zz_t45 = (2*(11*rho - 12*m_xx_I*rho_I - 9*m_xy_I*rho_I - 18*m_xz_I*rho_I + 24*m_yy_I*rho_I + 90*m_yz_I*rho_I + 105*m_zz_I*rho_I + 3*rho*ux0 - 3*rho*uy0 - 6*rho*uz0))/(135*rho);

    rhoVar = rho;

    break;
case 0b0000010111010100:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[14] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = -(24*(5*rho_I + m_xx_I*rho_I - 2*m_xy_I*rho_I - 2*m_xz_I*rho_I + m_yy_I*rho_I + 2*m_yz_I*rho_I + m_zz_I*rho_I - omegaVar*m_xx_I*rho_I + 2*omegaVar*m_xy_I*rho_I + 2*omegaVar*m_xz_I*rho_I - omegaVar*m_yy_I*rho_I - 2*omegaVar*m_yz_I*rho_I - omegaVar*m_zz_I*rho_I))/(8*uy0 - 8*ux0 - 6*omegaVar + 8*uz0 - 12*omegaVar*ux0 + 12*omegaVar*uy0 + 12*omegaVar*uz0 + 15*omegaVar*ux0*ux0 + 15*omegaVar*uy0*uy0 + 15*omegaVar*uz0*uz0 - 30*omegaVar*ux0*uy0 - 30*omegaVar*ux0*uz0 + 30*omegaVar*uy0*uz0 - 104);

    m_xx_t45 = -(2*(6*m_xy_I*rho_I - 27*m_xx_I*rho_I - rho + 6*m_xz_I*rho_I + 3*m_yz_I*rho_I - 4*rho*ux0 + rho*uy0 + rho*uz0))/(45*rho);
    m_xy_t90 = -(2*rho + 6*m_xx_I*rho_I - 63*m_xy_I*rho_I + 6*m_yy_I*rho_I - 3*m_zz_I*rho_I + 6*rho*ux0 - 6*rho*uy0)/(45*rho);
    m_xz_t90 = -(2*rho + 6*m_xx_I*rho_I - 63*m_xz_I*rho_I - 3*m_yy_I*rho_I + 6*m_zz_I*rho_I + 6*rho*ux0 - 6*rho*uz0)/(45*rho);
    m_yy_t45 = (2*(rho - 6*m_xy_I*rho_I + 3*m_xz_I*rho_I + 27*m_yy_I*rho_I + 6*m_yz_I*rho_I + rho*ux0 - 4*rho*uy0 - rho*uz0))/(45*rho);
    m_yz_t90 = (2*rho - 3*m_xx_I*rho_I + 6*m_yy_I*rho_I + 63*m_yz_I*rho_I + 6*m_zz_I*rho_I - 6*rho*uy0 - 6*rho*uz0)/(45*rho);
    m_zz_t45 = (2*(rho + 3*m_xy_I*rho_I - 6*m_xz_I*rho_I + 6*m_yz_I*rho_I + 27*m_zz_I*rho_I + rho*ux0 - rho*uy0 - 4*rho*uz0))/(45*rho);

    rhoVar = rho;

    break;
case 0b0000010111010101:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[8] + pop[10] + pop[11] + pop[14] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[8] - pop[14]);
    m_xz_I = inv_rho_I * (pop[10] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = -(72*(107*rho_I + 72*m_xx_I*rho_I + 12*m_yy_I*rho_I + 33*m_yz_I*rho_I + 12*m_zz_I*rho_I - 72*omegaVar*m_xx_I*rho_I - 12*omegaVar*m_yy_I*rho_I - 33*omegaVar*m_yz_I*rho_I - 12*omegaVar*m_zz_I*rho_I))/(492*uy0 - 2124*ux0 - 626*omegaVar + 492*uz0 - 1728*omegaVar*ux0 + 150*omegaVar*uy0 + 150*omegaVar*uz0 + 3531*omegaVar*ux0*ux0 + 642*omegaVar*uy0*uy0 + 642*omegaVar*uz0*uz0 + 1926*omegaVar*uy0*uz0 - 5580);

    m_xx_t45 = (2*(24*rho + 249*m_xx_I*rho_I - 12*m_yy_I*rho_I - 33*m_yz_I*rho_I - 12*m_zz_I*rho_I + 83*rho*ux0 + 11*rho*uy0 + 11*rho*uz0))/(321*rho);
    m_xy_t90 = (6*m_xy_I*rho_I + rho*uy0)/(3*rho);
    m_xz_t90 = (6*m_xz_I*rho_I + rho*uz0)/(3*rho);
    m_yy_t45 = -(4*(6*m_xx_I*rho_I - 2*rho - 106*m_yy_I*rho_I - 24*m_yz_I*rho_I + m_zz_I*rho_I + 2*rho*ux0 + 8*rho*uy0 + 8*rho*uz0))/(321*rho);
    m_yz_t90 = (11*rho - 33*m_xx_I*rho_I + 48*m_yy_I*rho_I + 453*m_yz_I*rho_I + 48*m_zz_I*rho_I - 11*rho*ux0 - 44*rho*uy0 - 44*rho*uz0)/(321*rho);
    m_zz_t45 = -(4*(6*m_xx_I*rho_I - 2*rho + m_yy_I*rho_I - 24*m_yz_I*rho_I - 106*m_zz_I*rho_I + 2*rho*ux0 + 8*rho*uy0 + 8*rho*uz0))/(321*rho);

    rhoVar = rho;

    break;
case 0b0000010111010110:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[14] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = -(24*(5*rho_I + m_xx_I*rho_I - 2*m_xy_I*rho_I - 2*m_xz_I*rho_I + m_yy_I*rho_I + 2*m_yz_I*rho_I + m_zz_I*rho_I - omegaVar*m_xx_I*rho_I + 2*omegaVar*m_xy_I*rho_I + 2*omegaVar*m_xz_I*rho_I - omegaVar*m_yy_I*rho_I - 2*omegaVar*m_yz_I*rho_I - omegaVar*m_zz_I*rho_I))/(8*uy0 - 8*ux0 - 6*omegaVar + 8*uz0 - 12*omegaVar*ux0 + 12*omegaVar*uy0 + 12*omegaVar*uz0 + 15*omegaVar*ux0*ux0 + 15*omegaVar*uy0*uy0 + 15*omegaVar*uz0*uz0 - 30*omegaVar*ux0*uy0 - 30*omegaVar*ux0*uz0 + 30*omegaVar*uy0*uz0 - 104);

    m_xx_t45 = -(2*(6*m_xy_I*rho_I - 27*m_xx_I*rho_I - rho + 6*m_xz_I*rho_I + 3*m_yz_I*rho_I - 4*rho*ux0 + rho*uy0 + rho*uz0))/(45*rho);
    m_xy_t90 = -(2*rho + 6*m_xx_I*rho_I - 63*m_xy_I*rho_I + 6*m_yy_I*rho_I - 3*m_zz_I*rho_I + 6*rho*ux0 - 6*rho*uy0)/(45*rho);
    m_xz_t90 = -(2*rho + 6*m_xx_I*rho_I - 63*m_xz_I*rho_I - 3*m_yy_I*rho_I + 6*m_zz_I*rho_I + 6*rho*ux0 - 6*rho*uz0)/(45*rho);
    m_yy_t45 = (2*(rho - 6*m_xy_I*rho_I + 3*m_xz_I*rho_I + 27*m_yy_I*rho_I + 6*m_yz_I*rho_I + rho*ux0 - 4*rho*uy0 - rho*uz0))/(45*rho);
    m_yz_t90 = (2*rho - 3*m_xx_I*rho_I + 6*m_yy_I*rho_I + 63*m_yz_I*rho_I + 6*m_zz_I*rho_I - 6*rho*uy0 - 6*rho*uz0)/(45*rho);
    m_zz_t45 = (2*(rho + 3*m_xy_I*rho_I - 6*m_xz_I*rho_I + 6*m_yz_I*rho_I + 27*m_zz_I*rho_I + rho*ux0 - rho*uy0 - 4*rho*uz0))/(45*rho);

    rhoVar = rho;

    break;
case 0b0000010111010111:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[8] + pop[10] + pop[14] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[8] - pop[14]);
    m_xz_I = inv_rho_I * (pop[10] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (-pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = (36*(19*rho_I + 6*m_xx_I*rho_I + 12*m_yy_I*rho_I + 33*m_yz_I*rho_I + 12*m_zz_I*rho_I - 6*omegaVar*m_xx_I*rho_I - 12*omegaVar*m_yy_I*rho_I - 33*omegaVar*m_yz_I*rho_I - 12*omegaVar*m_zz_I*rho_I))/(118*omegaVar + 270*ux0 + 72*omegaVar*ux0 - 285*omegaVar*ux0*ux0 - 114*omegaVar*uy0*uy0 - 114*omegaVar*uz0*uz0 - 342*omegaVar*uy0*uz0 + 414);

    m_xx_t45 = (2*(2*rho + 51*m_xx_I*rho_I - 12*m_yy_I*rho_I - 33*m_yz_I*rho_I - 12*m_zz_I*rho_I + 17*rho*ux0))/(57*rho);
    m_xy_t90 = (6*m_xy_I*rho_I + rho*uy0)/(3*rho);
    m_xz_t90 = (6*m_xz_I*rho_I + rho*uz0)/(3*rho);
    m_yy_t45 = (4*(2*rho - 6*m_xx_I*rho_I + 26*m_yy_I*rho_I + 24*m_yz_I*rho_I + 7*m_zz_I*rho_I - 2*rho*ux0))/(57*rho);
    m_yz_t90 = (11*rho - 33*m_xx_I*rho_I + 48*m_yy_I*rho_I + 189*m_yz_I*rho_I + 48*m_zz_I*rho_I - 11*rho*ux0)/(57*rho);
    m_zz_t45 = (4*(2*rho - 6*m_xx_I*rho_I + 7*m_yy_I*rho_I + 24*m_yz_I*rho_I + 26*m_zz_I*rho_I - 2*rho*ux0))/(57*rho);

    rhoVar = rho;

    break;
case 0b0000010111011000:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[9] + pop[10] + pop[11] + pop[13] + pop[14] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = -(24*(5*rho_I + m_xx_I*rho_I + 2*m_xy_I*rho_I - 2*m_xz_I*rho_I + m_yy_I*rho_I + 2*m_yz_I*rho_I + m_zz_I*rho_I - omegaVar*m_xx_I*rho_I - 2*omegaVar*m_xy_I*rho_I + 2*omegaVar*m_xz_I*rho_I - omegaVar*m_yy_I*rho_I - 2*omegaVar*m_yz_I*rho_I - omegaVar*m_zz_I*rho_I))/(8*uy0 - 6*omegaVar + 8*uz0 + 12*omegaVar*uy0 + 12*omegaVar*uz0 + 15*omegaVar*ux0*ux0 + 15*omegaVar*uy0*uy0 + 15*omegaVar*uz0*uz0 + 30*omegaVar*ux0*uy0 - 30*omegaVar*ux0*uz0 + 30*omegaVar*uy0*uz0 - 104);

    m_xx_t45 = -(2*(6*m_xz_I*rho_I - 27*m_xx_I*rho_I - 6*m_xy_I*rho_I - rho + 3*m_yz_I*rho_I + rho*uy0 + rho*uz0))/(45*rho);
    m_xy_t90 = (2*rho + 6*m_xx_I*rho_I + 63*m_xy_I*rho_I + 6*m_yy_I*rho_I - 3*m_zz_I*rho_I - 6*rho*ux0 - 6*rho*uy0)/(45*rho);
    m_xz_t90 = -(2*rho + 6*m_xx_I*rho_I - 63*m_xz_I*rho_I - 3*m_yy_I*rho_I + 6*m_zz_I*rho_I + 6*rho*ux0 - 6*rho*uz0)/(45*rho);
    m_yy_t45 = (2*(rho + 6*m_xy_I*rho_I + 3*m_xz_I*rho_I + 27*m_yy_I*rho_I + 6*m_yz_I*rho_I - 3*rho*ux0 - 4*rho*uy0 - rho*uz0))/(45*rho);
    m_yz_t90 = (2*rho - 3*m_xx_I*rho_I + 6*m_yy_I*rho_I + 63*m_yz_I*rho_I + 6*m_zz_I*rho_I - 6*rho*uy0 - 6*rho*uz0)/(45*rho);
    m_zz_t45 = (2*(rho - 3*m_xy_I*rho_I - 6*m_xz_I*rho_I + 6*m_yz_I*rho_I + 27*m_zz_I*rho_I + 3*rho*ux0 - rho*uy0 - 4*rho*uz0))/(45*rho);

    rhoVar = rho;

    break;
case 0b0000010111011001:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[9] + pop[10] + pop[11] + pop[13] + pop[14] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (-pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = -(24*(15*rho_I + 11*m_xx_I*rho_I + 30*m_xy_I*rho_I - 6*m_xz_I*rho_I + 11*m_yy_I*rho_I + 6*m_yz_I*rho_I - m_zz_I*rho_I - 11*omegaVar*m_xx_I*rho_I - 30*omegaVar*m_xy_I*rho_I + 6*omegaVar*m_xz_I*rho_I - 11*omegaVar*m_yy_I*rho_I - 6*omegaVar*m_yz_I*rho_I + omegaVar*m_zz_I*rho_I))/(12*uy0 - 12*ux0 - 72*omegaVar + 24*uz0 - 18*omegaVar*ux0 + 18*omegaVar*uy0 + 36*omegaVar*uz0 + 75*omegaVar*ux0*ux0 + 75*omegaVar*uy0*uy0 + 30*omegaVar*uz0*uz0 + 180*omegaVar*ux0*uy0 - 90*omegaVar*ux0*uz0 + 90*omegaVar*uy0*uz0 - 248);

    m_xx_t45 = (2*(11*rho + 105*m_xx_I*rho_I + 90*m_xy_I*rho_I - 18*m_xz_I*rho_I + 24*m_yy_I*rho_I - 9*m_yz_I*rho_I - 12*m_zz_I*rho_I + 6*rho*ux0 + 3*rho*uy0 - 3*rho*uz0))/(135*rho);
    m_xy_t90 = (2*rho + 6*m_xx_I*rho_I + 27*m_xy_I*rho_I + 6*m_yy_I*rho_I - 3*m_zz_I*rho_I)/(9*rho);
    m_xz_t90 = -(2*rho + 6*m_xx_I*rho_I - 63*m_xz_I*rho_I - 3*m_yy_I*rho_I + 6*m_zz_I*rho_I + 6*rho*ux0 - 6*rho*uz0)/(45*rho);
    m_yy_t45 = (2*(11*rho + 24*m_xx_I*rho_I + 90*m_xy_I*rho_I + 9*m_xz_I*rho_I + 105*m_yy_I*rho_I + 18*m_yz_I*rho_I - 12*m_zz_I*rho_I - 3*rho*ux0 - 6*rho*uy0 - 3*rho*uz0))/(135*rho);
    m_yz_t90 = (2*rho - 3*m_xx_I*rho_I + 6*m_yy_I*rho_I + 63*m_yz_I*rho_I + 6*m_zz_I*rho_I - 6*rho*uy0 - 6*rho*uz0)/(45*rho);
    m_zz_t45 = -(2*(rho + 12*m_xx_I*rho_I + 45*m_xy_I*rho_I + 18*m_xz_I*rho_I + 12*m_yy_I*rho_I - 18*m_yz_I*rho_I - 87*m_zz_I*rho_I - 6*rho*ux0 + 6*rho*uy0 + 12*rho*uz0))/(135*rho);

    rhoVar = rho;

    break;
case 0b0000010111011010:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[9] + pop[10] + pop[11] + pop[13] + pop[14] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = -(24*(15*rho_I + 11*m_xx_I*rho_I + 6*m_xy_I*rho_I - 30*m_xz_I*rho_I - m_yy_I*rho_I + 6*m_yz_I*rho_I + 11*m_zz_I*rho_I - 11*omegaVar*m_xx_I*rho_I - 6*omegaVar*m_xy_I*rho_I + 30*omegaVar*m_xz_I*rho_I + omegaVar*m_yy_I*rho_I - 6*omegaVar*m_yz_I*rho_I - 11*omegaVar*m_zz_I*rho_I))/(12*ux0 - 72*omegaVar + 24*uy0 + 12*uz0 + 18*omegaVar*ux0 + 36*omegaVar*uy0 + 18*omegaVar*uz0 + 75*omegaVar*ux0*ux0 + 30*omegaVar*uy0*uy0 + 75*omegaVar*uz0*uz0 + 90*omegaVar*ux0*uy0 - 180*omegaVar*ux0*uz0 + 90*omegaVar*uy0*uz0 - 248);

    m_xx_t45 = (2*(11*rho + 105*m_xx_I*rho_I + 18*m_xy_I*rho_I - 90*m_xz_I*rho_I - 12*m_yy_I*rho_I - 9*m_yz_I*rho_I + 24*m_zz_I*rho_I - 6*rho*ux0 - 3*rho*uy0 + 3*rho*uz0))/(135*rho);
    m_xy_t90 = (2*rho + 6*m_xx_I*rho_I + 63*m_xy_I*rho_I + 6*m_yy_I*rho_I - 3*m_zz_I*rho_I - 6*rho*ux0 - 6*rho*uy0)/(45*rho);
    m_xz_t90 = -(2*rho + 6*m_xx_I*rho_I - 27*m_xz_I*rho_I - 3*m_yy_I*rho_I + 6*m_zz_I*rho_I)/(9*rho);
    m_yy_t45 = -(2*(rho + 12*m_xx_I*rho_I - 18*m_xy_I*rho_I - 45*m_xz_I*rho_I - 87*m_yy_I*rho_I - 18*m_yz_I*rho_I + 12*m_zz_I*rho_I + 6*rho*ux0 + 12*rho*uy0 + 6*rho*uz0))/(135*rho);
    m_yz_t90 = (2*rho - 3*m_xx_I*rho_I + 6*m_yy_I*rho_I + 63*m_yz_I*rho_I + 6*m_zz_I*rho_I - 6*rho*uy0 - 6*rho*uz0)/(45*rho);
    m_zz_t45 = (2*(11*rho + 24*m_xx_I*rho_I - 9*m_xy_I*rho_I - 90*m_xz_I*rho_I - 12*m_yy_I*rho_I + 18*m_yz_I*rho_I + 105*m_zz_I*rho_I + 3*rho*ux0 - 3*rho*uy0 - 6*rho*uz0))/(135*rho);

    rhoVar = rho;

    break;
case 0b0000010111011011:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[9] + pop[10] + pop[13] + pop[14] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (-pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (-pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = -(12*(rho_I + m_xx_I*rho_I + 2*m_xy_I*rho_I - 2*m_xz_I*rho_I + m_yy_I*rho_I + 2*m_yz_I*rho_I + m_zz_I*rho_I - omegaVar*m_xx_I*rho_I - 2*omegaVar*m_xy_I*rho_I + 2*omegaVar*m_xz_I*rho_I - omegaVar*m_yy_I*rho_I - 2*omegaVar*m_yz_I*rho_I - omegaVar*m_zz_I*rho_I))/(3*omegaVar*ux0*ux0 - 6*omegaVar + 3*omegaVar*uy0*uy0 + 3*omegaVar*uz0*uz0 + 6*omegaVar*ux0*uy0 - 6*omegaVar*ux0*uz0 + 6*omegaVar*uy0*uz0 - 4);

    m_xx_t45 = (2*(rho + 9*m_xx_I*rho_I + 6*m_xy_I*rho_I - 6*m_xz_I*rho_I - 3*m_yz_I*rho_I))/(9*rho);
    m_xy_t90 = (2*rho + 6*m_xx_I*rho_I + 27*m_xy_I*rho_I + 6*m_yy_I*rho_I - 3*m_zz_I*rho_I)/(9*rho);
    m_xz_t90 = -(2*rho + 6*m_xx_I*rho_I - 27*m_xz_I*rho_I - 3*m_yy_I*rho_I + 6*m_zz_I*rho_I)/(9*rho);
    m_yy_t45 = (2*(rho + 6*m_xy_I*rho_I + 3*m_xz_I*rho_I + 9*m_yy_I*rho_I + 6*m_yz_I*rho_I))/(9*rho);
    m_yz_t90 = (2*rho - 3*m_xx_I*rho_I + 6*m_yy_I*rho_I + 27*m_yz_I*rho_I + 6*m_zz_I*rho_I)/(9*rho);
    m_zz_t45 = (2*(rho - 3*m_xy_I*rho_I - 6*m_xz_I*rho_I + 6*m_yz_I*rho_I + 9*m_zz_I*rho_I))/(9*rho);

    rhoVar = rho;

    break;
case 0b0000010111011100:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[5] + pop[6] + pop[7] + pop[9] + pop[10] + pop[11] + pop[14] + pop[16] + pop[17];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17]);
    m_xy_I = inv_rho_I * (pop[7] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17]);
    m_yz_I = inv_rho_I * (pop[11] - pop[17]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17]);

    rho = -(72*(107*rho_I + 12*m_xx_I*rho_I - 33*m_xz_I*rho_I + 72*m_yy_I*rho_I + 12*m_zz_I*rho_I - 12*omegaVar*m_xx_I*rho_I + 33*omegaVar*m_xz_I*rho_I - 72*omegaVar*m_yy_I*rho_I - 12*omegaVar*m_zz_I*rho_I))/(2124*uy0 - 492*ux0 - 626*omegaVar + 492*uz0 - 150*omegaVar*ux0 + 1728*omegaVar*uy0 + 150*omegaVar*uz0 + 642*omegaVar*ux0*ux0 + 3531*omegaVar*uy0*uy0 + 642*omegaVar*uz0*uz0 - 1926*omegaVar*ux0*uz0 - 5580);

    m_xx_t45 = (4*(2*rho + 106*m_xx_I*rho_I - 24*m_xz_I*rho_I - 6*m_yy_I*rho_I - m_zz_I*rho_I + 8*rho*ux0 + 2*rho*uy0 - 8*rho*uz0))/(321*rho);
    m_xy_t90 = (6*m_xy_I*rho_I - rho*ux0)/(3*rho);
    m_xz_t90 = -(11*rho + 48*m_xx_I*rho_I - 453*m_xz_I*rho_I - 33*m_yy_I*rho_I + 48*m_zz_I*rho_I + 44*rho*ux0 + 11*rho*uy0 - 44*rho*uz0)/(321*rho);
    m_yy_t45 = (2*(24*rho - 12*m_xx_I*rho_I + 33*m_xz_I*rho_I + 249*m_yy_I*rho_I - 12*m_zz_I*rho_I - 11*rho*ux0 - 83*rho*uy0 + 11*rho*uz0))/(321*rho);
    m_yz_t90 = (6*m_yz_I*rho_I - rho*uz0)/(3*rho);
    m_zz_t45 = (4*(2*rho - m_xx_I*rho_I - 24*m_xz_I*rho_I - 6*m_yy_I*rho_I + 106*m_zz_I*rho_I + 8*rho*ux0 + 2*rho*uy0 - 8*rho*uz0))/(321*rho);

    rhoVar = rho;

    break;
case 0b0000010111011101:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[2] + pop[3] + pop[5] + pop[6] + pop[10] + pop[11] + pop[14] + pop[16] + pop[17];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17]);
    m_xy_I = inv_rho_I * (-pop[14]);
    m_xz_I = inv_rho_I * (pop[10] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17]);
    m_yz_I = inv_rho_I * (pop[11] - pop[17]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17]);

    rho = -(72*(23*rho_I + 24*m_xx_I*rho_I + 57*m_xy_I*rho_I + 24*m_yy_I*rho_I - 6*m_zz_I*rho_I - 24*omegaVar*m_xx_I*rho_I - 57*omegaVar*m_xy_I*rho_I - 24*omegaVar*m_yy_I*rho_I + 6*omegaVar*m_zz_I*rho_I))/(660*uy0 - 660*ux0 - 430*omegaVar - 30*omegaVar*ux0 + 30*omegaVar*uy0 + 690*omegaVar*ux0*ux0 + 690*omegaVar*uy0*uy0 + 69*omegaVar*uz0*uz0 + 414*omegaVar*ux0*uy0 - 720);

    m_xx_t45 = (16*rho + 141*m_xx_I*rho_I + 102*m_xy_I*rho_I + 3*m_yy_I*rho_I - 18*m_zz_I*rho_I + 30*rho*ux0 + 16*rho*uy0)/(69*rho);
    m_xy_t90 = (19*rho + 51*m_xx_I*rho_I + 354*m_xy_I*rho_I + 51*m_yy_I*rho_I - 30*m_zz_I*rho_I - 19*rho*ux0 + 19*rho*uy0)/(69*rho);
    m_xz_t90 = (6*m_xz_I*rho_I + rho*uz0)/(3*rho);
    m_yy_t45 = (16*rho + 3*m_xx_I*rho_I + 102*m_xy_I*rho_I + 141*m_yy_I*rho_I - 18*m_zz_I*rho_I - 16*rho*ux0 - 30*rho*uy0)/(69*rho);
    m_yz_t90 = (6*m_yz_I*rho_I - rho*uz0)/(3*rho);
    m_zz_t45 = -(2*(2*rho + 9*m_xx_I*rho_I + 30*m_xy_I*rho_I + 9*m_yy_I*rho_I - 54*m_zz_I*rho_I - 2*rho*ux0 + 2*rho*uy0))/(69*rho);

    rhoVar = rho;

    break;
case 0b0000010111011110:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[5] + pop[6] + pop[7] + pop[9] + pop[10] + pop[11] + pop[14] + pop[17];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[17]);
    m_xy_I = inv_rho_I * (pop[7] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[17]);
    m_yz_I = inv_rho_I * (pop[11] - pop[17]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[17]);

    rho = -(36*(19*rho_I + 12*m_xx_I*rho_I - 33*m_xz_I*rho_I + 6*m_yy_I*rho_I + 12*m_zz_I*rho_I - 12*omegaVar*m_xx_I*rho_I + 33*omegaVar*m_xz_I*rho_I - 6*omegaVar*m_yy_I*rho_I - 12*omegaVar*m_zz_I*rho_I))/(270*uy0 - 118*omegaVar + 72*omegaVar*uy0 + 114*omegaVar*ux0*ux0 + 285*omegaVar*uy0*uy0 + 114*omegaVar*uz0*uz0 - 342*omegaVar*ux0*uz0 - 414);

    m_xx_t45 = (4*(2*rho + 26*m_xx_I*rho_I - 24*m_xz_I*rho_I - 6*m_yy_I*rho_I + 7*m_zz_I*rho_I + 2*rho*uy0))/(57*rho);
    m_xy_t90 = (6*m_xy_I*rho_I - rho*ux0)/(3*rho);
    m_xz_t90 = -(11*rho + 48*m_xx_I*rho_I - 189*m_xz_I*rho_I - 33*m_yy_I*rho_I + 48*m_zz_I*rho_I + 11*rho*uy0)/(57*rho);
    m_yy_t45 = (2*(2*rho - 12*m_xx_I*rho_I + 33*m_xz_I*rho_I + 51*m_yy_I*rho_I - 12*m_zz_I*rho_I - 17*rho*uy0))/(57*rho);
    m_yz_t90 = (6*m_yz_I*rho_I - rho*uz0)/(3*rho);
    m_zz_t45 = (4*(2*rho + 7*m_xx_I*rho_I - 24*m_xz_I*rho_I - 6*m_yy_I*rho_I + 26*m_zz_I*rho_I + 2*rho*uy0))/(57*rho);

    rhoVar = rho;

    break;
case 0b0000010111011111:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[2] + pop[3] + pop[6] + pop[10] + pop[14] + pop[17];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[17]);
    m_xy_I = inv_rho_I * (-pop[14]);
    m_xz_I = inv_rho_I * (pop[10]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] - (1.0/3.0)*pop[6] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[17]);
    m_yz_I = inv_rho_I * (-pop[17]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] + (2.0/3.0)*pop[6] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[17]);

    rho = -(24*(rho_I + m_xx_I*rho_I + 2*m_xy_I*rho_I - 2*m_xz_I*rho_I + m_yy_I*rho_I + 2*m_yz_I*rho_I + m_zz_I*rho_I - omegaVar*m_xx_I*rho_I - 2*omegaVar*m_xy_I*rho_I + 2*omegaVar*m_xz_I*rho_I - omegaVar*m_yy_I*rho_I - 2*omegaVar*m_yz_I*rho_I - omegaVar*m_zz_I*rho_I))/(12*uy0 - 12*ux0 - 10*omegaVar - 12*uz0 + 4*omegaVar*ux0 - 4*omegaVar*uy0 + 4*omegaVar*uz0 + 9*omegaVar*ux0*ux0 + 9*omegaVar*uy0*uy0 + 9*omegaVar*uz0*uz0 + 6*omegaVar*ux0*uy0 - 6*omegaVar*ux0*uz0 + 6*omegaVar*uy0*uz0 - 4);

    m_xx_t45 = (2*(rho + 15*m_xx_I*rho_I + 9*m_xy_I*rho_I - 9*m_xz_I*rho_I - 3*m_yy_I*rho_I - 9*m_yz_I*rho_I - 3*m_zz_I*rho_I + 2*rho*ux0 + rho*uy0 - rho*uz0))/(9*rho);
    m_xy_t90 = (2*rho + 9*m_xx_I*rho_I + 51*m_xy_I*rho_I + 3*m_xz_I*rho_I + 9*m_yy_I*rho_I - 3*m_yz_I*rho_I - 9*m_zz_I*rho_I - 2*rho*ux0 + 2*rho*uy0 - 2*rho*uz0)/(9*rho);
    m_xz_t90 = (3*m_xy_I*rho_I - 9*m_xx_I*rho_I - 2*rho + 51*m_xz_I*rho_I + 9*m_yy_I*rho_I + 3*m_yz_I*rho_I - 9*m_zz_I*rho_I + 2*rho*ux0 - 2*rho*uy0 + 2*rho*uz0)/(9*rho);
    m_yy_t45 = (2*(rho - 3*m_xx_I*rho_I + 9*m_xy_I*rho_I + 9*m_xz_I*rho_I + 15*m_yy_I*rho_I + 9*m_yz_I*rho_I - 3*m_zz_I*rho_I - rho*ux0 - 2*rho*uy0 - rho*uz0))/(9*rho);
    m_yz_t90 = (2*rho - 9*m_xx_I*rho_I - 3*m_xy_I*rho_I + 3*m_xz_I*rho_I + 9*m_yy_I*rho_I + 51*m_yz_I*rho_I + 9*m_zz_I*rho_I - 2*rho*ux0 + 2*rho*uy0 - 2*rho*uz0)/(9*rho);
    m_zz_t45 = (2*(rho - 3*m_xx_I*rho_I - 9*m_xy_I*rho_I - 9*m_xz_I*rho_I - 3*m_yy_I*rho_I + 9*m_yz_I*rho_I + 15*m_zz_I*rho_I - rho*ux0 + rho*uy0 + 2*rho*uz0))/(9*rho);

    rhoVar = rho;

    break;
case 0b0000010111100000:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[11] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] - (1.0/3.0)*pop[11] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[11] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = -(24*(15*rho_I + m_xx_I*rho_I + 6*m_xz_I*rho_I + m_yy_I*rho_I + 6*m_yz_I*rho_I + 4*m_zz_I*rho_I - omegaVar*m_xx_I*rho_I - 6*omegaVar*m_xz_I*rho_I - omegaVar*m_yy_I*rho_I - 6*omegaVar*m_yz_I*rho_I - 4*omegaVar*m_zz_I*rho_I))/(12*ux0 - 12*omegaVar + 12*uy0 + 24*uz0 + 18*omegaVar*ux0 + 18*omegaVar*uy0 + 36*omegaVar*uz0 + 15*omegaVar*ux0*ux0 + 15*omegaVar*uy0*uy0 + 60*omegaVar*uz0*uz0 + 90*omegaVar*ux0*uz0 + 90*omegaVar*uy0*uz0 - 328);

    m_xx_t45 = (2*(rho + 75*m_xx_I*rho_I + 18*m_xz_I*rho_I - 6*m_yy_I*rho_I - 9*m_yz_I*rho_I + 3*m_zz_I*rho_I - 6*rho*ux0 + 3*rho*uy0 - 3*rho*uz0))/(135*rho);
    m_xy_t90 = (m_xy_I*rho_I)/rho;
    m_xz_t90 = (2*rho + 6*m_xx_I*rho_I + 63*m_xz_I*rho_I - 3*m_yy_I*rho_I + 6*m_zz_I*rho_I - 6*rho*ux0 - 6*rho*uz0)/(45*rho);
    m_yy_t45 = (2*(rho - 6*m_xx_I*rho_I - 9*m_xz_I*rho_I + 75*m_yy_I*rho_I + 18*m_yz_I*rho_I + 3*m_zz_I*rho_I + 3*rho*ux0 - 6*rho*uy0 - 3*rho*uz0))/(135*rho);
    m_yz_t90 = (2*rho - 3*m_xx_I*rho_I + 6*m_yy_I*rho_I + 63*m_yz_I*rho_I + 6*m_zz_I*rho_I - 6*rho*uy0 - 6*rho*uz0)/(45*rho);
    m_zz_t45 = (8*rho + 6*m_xx_I*rho_I + 36*m_xz_I*rho_I + 6*m_yy_I*rho_I + 36*m_yz_I*rho_I + 159*m_zz_I*rho_I - 12*rho*ux0 - 12*rho*uy0 - 24*rho*uz0)/(135*rho);

    rhoVar = rho;

    break;
case 0b0000010111100001:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[11] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] - (1.0/3.0)*pop[11] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[11] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = -(24*(15*rho_I + m_xx_I*rho_I + 6*m_xz_I*rho_I + m_yy_I*rho_I + 6*m_yz_I*rho_I + 4*m_zz_I*rho_I - omegaVar*m_xx_I*rho_I - 6*omegaVar*m_xz_I*rho_I - omegaVar*m_yy_I*rho_I - 6*omegaVar*m_yz_I*rho_I - 4*omegaVar*m_zz_I*rho_I))/(12*ux0 - 12*omegaVar + 12*uy0 + 24*uz0 + 18*omegaVar*ux0 + 18*omegaVar*uy0 + 36*omegaVar*uz0 + 15*omegaVar*ux0*ux0 + 15*omegaVar*uy0*uy0 + 60*omegaVar*uz0*uz0 + 90*omegaVar*ux0*uz0 + 90*omegaVar*uy0*uz0 - 328);

    m_xx_t45 = (2*(rho + 75*m_xx_I*rho_I + 18*m_xz_I*rho_I - 6*m_yy_I*rho_I - 9*m_yz_I*rho_I + 3*m_zz_I*rho_I - 6*rho*ux0 + 3*rho*uy0 - 3*rho*uz0))/(135*rho);
    m_xy_t90 = (m_xy_I*rho_I)/rho;
    m_xz_t90 = (2*rho + 6*m_xx_I*rho_I + 63*m_xz_I*rho_I - 3*m_yy_I*rho_I + 6*m_zz_I*rho_I - 6*rho*ux0 - 6*rho*uz0)/(45*rho);
    m_yy_t45 = (2*(rho - 6*m_xx_I*rho_I - 9*m_xz_I*rho_I + 75*m_yy_I*rho_I + 18*m_yz_I*rho_I + 3*m_zz_I*rho_I + 3*rho*ux0 - 6*rho*uy0 - 3*rho*uz0))/(135*rho);
    m_yz_t90 = (2*rho - 3*m_xx_I*rho_I + 6*m_yy_I*rho_I + 63*m_yz_I*rho_I + 6*m_zz_I*rho_I - 6*rho*uy0 - 6*rho*uz0)/(45*rho);
    m_zz_t45 = (8*rho + 6*m_xx_I*rho_I + 36*m_xz_I*rho_I + 6*m_yy_I*rho_I + 36*m_yz_I*rho_I + 159*m_zz_I*rho_I - 12*rho*ux0 - 12*rho*uy0 - 24*rho*uz0)/(135*rho);

    rhoVar = rho;

    break;
case 0b0000010111100010:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[11] + pop[13] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] - (1.0/3.0)*pop[11] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13]);
    m_xz_I = inv_rho_I * (pop[9] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[13] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[11] - (1.0/3.0)*pop[13] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = -(24*(5*rho_I + m_xx_I*rho_I - 2*m_xy_I*rho_I + 2*m_xz_I*rho_I + m_yy_I*rho_I + 2*m_yz_I*rho_I + m_zz_I*rho_I - omegaVar*m_xx_I*rho_I + 2*omegaVar*m_xy_I*rho_I - 2*omegaVar*m_xz_I*rho_I - omegaVar*m_yy_I*rho_I - 2*omegaVar*m_yz_I*rho_I - omegaVar*m_zz_I*rho_I))/(8*ux0 - 6*omegaVar + 8*uz0 + 12*omegaVar*ux0 + 12*omegaVar*uz0 + 15*omegaVar*ux0*ux0 + 15*omegaVar*uy0*uy0 + 15*omegaVar*uz0*uz0 - 30*omegaVar*ux0*uy0 + 30*omegaVar*ux0*uz0 + 30*omegaVar*uy0*uz0 - 104);

    m_xx_t45 = (2*(rho + 27*m_xx_I*rho_I - 6*m_xy_I*rho_I + 6*m_xz_I*rho_I - 3*m_yz_I*rho_I - 4*rho*ux0 + 3*rho*uy0 - rho*uz0))/(45*rho);
    m_xy_t90 = -(2*rho + 6*m_xx_I*rho_I - 63*m_xy_I*rho_I + 6*m_yy_I*rho_I - 3*m_zz_I*rho_I - 6*rho*ux0 + 6*rho*uy0)/(45*rho);
    m_xz_t90 = (2*rho + 6*m_xx_I*rho_I + 63*m_xz_I*rho_I - 3*m_yy_I*rho_I + 6*m_zz_I*rho_I - 6*rho*ux0 - 6*rho*uz0)/(45*rho);
    m_yy_t45 = -(2*(6*m_xy_I*rho_I - rho + 3*m_xz_I*rho_I - 27*m_yy_I*rho_I - 6*m_yz_I*rho_I + rho*ux0 + rho*uz0))/(45*rho);
    m_yz_t90 = (2*rho - 3*m_xx_I*rho_I + 6*m_yy_I*rho_I + 63*m_yz_I*rho_I + 6*m_zz_I*rho_I - 6*rho*uy0 - 6*rho*uz0)/(45*rho);
    m_zz_t45 = (2*(rho + 3*m_xy_I*rho_I + 6*m_xz_I*rho_I + 6*m_yz_I*rho_I + 27*m_zz_I*rho_I - rho*ux0 - 3*rho*uy0 - 4*rho*uz0))/(45*rho);

    rhoVar = rho;

    break;
case 0b0000010111100011:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[13] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13]);
    m_xz_I = inv_rho_I * (pop[9] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] + (2.0/3.0)*pop[13] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (-pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] - (1.0/3.0)*pop[13] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = -(24*(15*rho_I - m_xx_I*rho_I - 6*m_xy_I*rho_I + 6*m_xz_I*rho_I + 11*m_yy_I*rho_I + 30*m_yz_I*rho_I + 11*m_zz_I*rho_I + omegaVar*m_xx_I*rho_I + 6*omegaVar*m_xy_I*rho_I - 6*omegaVar*m_xz_I*rho_I - 11*omegaVar*m_yy_I*rho_I - 30*omegaVar*m_yz_I*rho_I - 11*omegaVar*m_zz_I*rho_I))/(24*ux0 - 72*omegaVar - 12*uy0 + 12*uz0 + 36*omegaVar*ux0 - 18*omegaVar*uy0 + 18*omegaVar*uz0 + 30*omegaVar*ux0*ux0 + 75*omegaVar*uy0*uy0 + 75*omegaVar*uz0*uz0 - 90*omegaVar*ux0*uy0 + 90*omegaVar*ux0*uz0 + 180*omegaVar*uy0*uz0 - 248);

    m_xx_t45 = -(2*(rho - 87*m_xx_I*rho_I + 18*m_xy_I*rho_I - 18*m_xz_I*rho_I + 12*m_yy_I*rho_I + 45*m_yz_I*rho_I + 12*m_zz_I*rho_I + 12*rho*ux0 - 6*rho*uy0 + 6*rho*uz0))/(135*rho);
    m_xy_t90 = -(2*rho + 6*m_xx_I*rho_I - 63*m_xy_I*rho_I + 6*m_yy_I*rho_I - 3*m_zz_I*rho_I - 6*rho*ux0 + 6*rho*uy0)/(45*rho);
    m_xz_t90 = (2*rho + 6*m_xx_I*rho_I + 63*m_xz_I*rho_I - 3*m_yy_I*rho_I + 6*m_zz_I*rho_I - 6*rho*ux0 - 6*rho*uz0)/(45*rho);
    m_yy_t45 = (2*(11*rho - 12*m_xx_I*rho_I - 18*m_xy_I*rho_I - 9*m_xz_I*rho_I + 105*m_yy_I*rho_I + 90*m_yz_I*rho_I + 24*m_zz_I*rho_I - 3*rho*ux0 + 6*rho*uy0 + 3*rho*uz0))/(135*rho);
    m_yz_t90 = (2*rho - 3*m_xx_I*rho_I + 6*m_yy_I*rho_I + 27*m_yz_I*rho_I + 6*m_zz_I*rho_I)/(9*rho);
    m_zz_t45 = (2*(11*rho - 12*m_xx_I*rho_I + 9*m_xy_I*rho_I + 18*m_xz_I*rho_I + 24*m_yy_I*rho_I + 90*m_yz_I*rho_I + 105*m_zz_I*rho_I - 3*rho*ux0 - 3*rho*uy0 - 6*rho*uz0))/(135*rho);

    rhoVar = rho;

    break;
case 0b0000010111100100:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[11] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] - (1.0/3.0)*pop[11] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[11] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = -(24*(5*rho_I + m_xx_I*rho_I - 2*m_xy_I*rho_I + 2*m_xz_I*rho_I + m_yy_I*rho_I + 2*m_yz_I*rho_I + m_zz_I*rho_I - omegaVar*m_xx_I*rho_I + 2*omegaVar*m_xy_I*rho_I - 2*omegaVar*m_xz_I*rho_I - omegaVar*m_yy_I*rho_I - 2*omegaVar*m_yz_I*rho_I - omegaVar*m_zz_I*rho_I))/(8*uy0 - 6*omegaVar + 8*uz0 + 12*omegaVar*uy0 + 12*omegaVar*uz0 + 15*omegaVar*ux0*ux0 + 15*omegaVar*uy0*uy0 + 15*omegaVar*uz0*uz0 - 30*omegaVar*ux0*uy0 + 30*omegaVar*ux0*uz0 + 30*omegaVar*uy0*uz0 - 104);

    m_xx_t45 = -(2*(6*m_xy_I*rho_I - 27*m_xx_I*rho_I - rho - 6*m_xz_I*rho_I + 3*m_yz_I*rho_I + rho*uy0 + rho*uz0))/(45*rho);
    m_xy_t90 = -(2*rho + 6*m_xx_I*rho_I - 63*m_xy_I*rho_I + 6*m_yy_I*rho_I - 3*m_zz_I*rho_I + 6*rho*ux0 - 6*rho*uy0)/(45*rho);
    m_xz_t90 = (2*rho + 6*m_xx_I*rho_I + 63*m_xz_I*rho_I - 3*m_yy_I*rho_I + 6*m_zz_I*rho_I - 6*rho*ux0 - 6*rho*uz0)/(45*rho);
    m_yy_t45 = (2*(rho - 6*m_xy_I*rho_I - 3*m_xz_I*rho_I + 27*m_yy_I*rho_I + 6*m_yz_I*rho_I + 3*rho*ux0 - 4*rho*uy0 - rho*uz0))/(45*rho);
    m_yz_t90 = (2*rho - 3*m_xx_I*rho_I + 6*m_yy_I*rho_I + 63*m_yz_I*rho_I + 6*m_zz_I*rho_I - 6*rho*uy0 - 6*rho*uz0)/(45*rho);
    m_zz_t45 = (2*(rho + 3*m_xy_I*rho_I + 6*m_xz_I*rho_I + 6*m_yz_I*rho_I + 27*m_zz_I*rho_I - 3*rho*ux0 - rho*uy0 - 4*rho*uz0))/(45*rho);

    rhoVar = rho;

    break;
case 0b0000010111100101:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[11] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[11] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[14]);
    m_xz_I = inv_rho_I * (-pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[11] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = -(24*(15*rho_I + 11*m_xx_I*rho_I - 6*m_xy_I*rho_I + 30*m_xz_I*rho_I - m_yy_I*rho_I + 6*m_yz_I*rho_I + 11*m_zz_I*rho_I - 11*omegaVar*m_xx_I*rho_I + 6*omegaVar*m_xy_I*rho_I - 30*omegaVar*m_xz_I*rho_I + omegaVar*m_yy_I*rho_I - 6*omegaVar*m_yz_I*rho_I - 11*omegaVar*m_zz_I*rho_I))/(24*uy0 - 12*ux0 - 72*omegaVar + 12*uz0 - 18*omegaVar*ux0 + 36*omegaVar*uy0 + 18*omegaVar*uz0 + 75*omegaVar*ux0*ux0 + 30*omegaVar*uy0*uy0 + 75*omegaVar*uz0*uz0 - 90*omegaVar*ux0*uy0 + 180*omegaVar*ux0*uz0 + 90*omegaVar*uy0*uz0 - 248);

    m_xx_t45 = (2*(11*rho + 105*m_xx_I*rho_I - 18*m_xy_I*rho_I + 90*m_xz_I*rho_I - 12*m_yy_I*rho_I - 9*m_yz_I*rho_I + 24*m_zz_I*rho_I + 6*rho*ux0 - 3*rho*uy0 + 3*rho*uz0))/(135*rho);
    m_xy_t90 = -(2*rho + 6*m_xx_I*rho_I - 63*m_xy_I*rho_I + 6*m_yy_I*rho_I - 3*m_zz_I*rho_I + 6*rho*ux0 - 6*rho*uy0)/(45*rho);
    m_xz_t90 = (2*rho + 6*m_xx_I*rho_I + 27*m_xz_I*rho_I - 3*m_yy_I*rho_I + 6*m_zz_I*rho_I)/(9*rho);
    m_yy_t45 = -(2*(rho + 12*m_xx_I*rho_I + 18*m_xy_I*rho_I + 45*m_xz_I*rho_I - 87*m_yy_I*rho_I - 18*m_yz_I*rho_I + 12*m_zz_I*rho_I - 6*rho*ux0 + 12*rho*uy0 + 6*rho*uz0))/(135*rho);
    m_yz_t90 = (2*rho - 3*m_xx_I*rho_I + 6*m_yy_I*rho_I + 63*m_yz_I*rho_I + 6*m_zz_I*rho_I - 6*rho*uy0 - 6*rho*uz0)/(45*rho);
    m_zz_t45 = (2*(11*rho + 24*m_xx_I*rho_I + 9*m_xy_I*rho_I + 90*m_xz_I*rho_I - 12*m_yy_I*rho_I + 18*m_yz_I*rho_I + 105*m_zz_I*rho_I - 3*rho*ux0 - 3*rho*uy0 - 6*rho*uz0))/(135*rho);

    rhoVar = rho;

    break;
case 0b0000010111100110:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[11] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] - (1.0/3.0)*pop[11] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8]);
    m_xz_I = inv_rho_I * (pop[9] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] + (2.0/3.0)*pop[11] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = -(24*(15*rho_I + 11*m_xx_I*rho_I - 30*m_xy_I*rho_I + 6*m_xz_I*rho_I + 11*m_yy_I*rho_I + 6*m_yz_I*rho_I - m_zz_I*rho_I - 11*omegaVar*m_xx_I*rho_I + 30*omegaVar*m_xy_I*rho_I - 6*omegaVar*m_xz_I*rho_I - 11*omegaVar*m_yy_I*rho_I - 6*omegaVar*m_yz_I*rho_I + omegaVar*m_zz_I*rho_I))/(12*ux0 - 72*omegaVar + 12*uy0 + 24*uz0 + 18*omegaVar*ux0 + 18*omegaVar*uy0 + 36*omegaVar*uz0 + 75*omegaVar*ux0*ux0 + 75*omegaVar*uy0*uy0 + 30*omegaVar*uz0*uz0 - 180*omegaVar*ux0*uy0 + 90*omegaVar*ux0*uz0 + 90*omegaVar*uy0*uz0 - 248);

    m_xx_t45 = (2*(11*rho + 105*m_xx_I*rho_I - 90*m_xy_I*rho_I + 18*m_xz_I*rho_I + 24*m_yy_I*rho_I - 9*m_yz_I*rho_I - 12*m_zz_I*rho_I - 6*rho*ux0 + 3*rho*uy0 - 3*rho*uz0))/(135*rho);
    m_xy_t90 = -(2*rho + 6*m_xx_I*rho_I - 27*m_xy_I*rho_I + 6*m_yy_I*rho_I - 3*m_zz_I*rho_I)/(9*rho);
    m_xz_t90 = (2*rho + 6*m_xx_I*rho_I + 63*m_xz_I*rho_I - 3*m_yy_I*rho_I + 6*m_zz_I*rho_I - 6*rho*ux0 - 6*rho*uz0)/(45*rho);
    m_yy_t45 = (2*(11*rho + 24*m_xx_I*rho_I - 90*m_xy_I*rho_I - 9*m_xz_I*rho_I + 105*m_yy_I*rho_I + 18*m_yz_I*rho_I - 12*m_zz_I*rho_I + 3*rho*ux0 - 6*rho*uy0 - 3*rho*uz0))/(135*rho);
    m_yz_t90 = (2*rho - 3*m_xx_I*rho_I + 6*m_yy_I*rho_I + 63*m_yz_I*rho_I + 6*m_zz_I*rho_I - 6*rho*uy0 - 6*rho*uz0)/(45*rho);
    m_zz_t45 = -(2*(rho + 12*m_xx_I*rho_I - 45*m_xy_I*rho_I - 18*m_xz_I*rho_I + 12*m_yy_I*rho_I - 18*m_yz_I*rho_I - 87*m_zz_I*rho_I + 6*rho*ux0 + 6*rho*uy0 + 12*rho*uz0))/(135*rho);

    rhoVar = rho;

    break;
case 0b0000010111100111:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8]);
    m_xz_I = inv_rho_I * (-pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (-pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = -(12*(rho_I + m_xx_I*rho_I - 2*m_xy_I*rho_I + 2*m_xz_I*rho_I + m_yy_I*rho_I + 2*m_yz_I*rho_I + m_zz_I*rho_I - omegaVar*m_xx_I*rho_I + 2*omegaVar*m_xy_I*rho_I - 2*omegaVar*m_xz_I*rho_I - omegaVar*m_yy_I*rho_I - 2*omegaVar*m_yz_I*rho_I - omegaVar*m_zz_I*rho_I))/(3*omegaVar*ux0*ux0 - 6*omegaVar + 3*omegaVar*uy0*uy0 + 3*omegaVar*uz0*uz0 - 6*omegaVar*ux0*uy0 + 6*omegaVar*ux0*uz0 + 6*omegaVar*uy0*uz0 - 4);

    m_xx_t45 = (2*(rho + 9*m_xx_I*rho_I - 6*m_xy_I*rho_I + 6*m_xz_I*rho_I - 3*m_yz_I*rho_I))/(9*rho);
    m_xy_t90 = -(2*rho + 6*m_xx_I*rho_I - 27*m_xy_I*rho_I + 6*m_yy_I*rho_I - 3*m_zz_I*rho_I)/(9*rho);
    m_xz_t90 = (2*rho + 6*m_xx_I*rho_I + 27*m_xz_I*rho_I - 3*m_yy_I*rho_I + 6*m_zz_I*rho_I)/(9*rho);
    m_yy_t45 = (2*(rho - 6*m_xy_I*rho_I - 3*m_xz_I*rho_I + 9*m_yy_I*rho_I + 6*m_yz_I*rho_I))/(9*rho);
    m_yz_t90 = (2*rho - 3*m_xx_I*rho_I + 6*m_yy_I*rho_I + 27*m_yz_I*rho_I + 6*m_zz_I*rho_I)/(9*rho);
    m_zz_t45 = (2*(rho + 3*m_xy_I*rho_I + 6*m_xz_I*rho_I + 6*m_yz_I*rho_I + 9*m_zz_I*rho_I))/(9*rho);

    rhoVar = rho;

    break;
case 0b0000010111101000:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[9] + pop[11] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[9] - (1.0/3.0)*pop[11] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] - (1.0/3.0)*pop[9] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[11] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = -(24*(5*rho_I + m_xx_I*rho_I + 2*m_xy_I*rho_I + 2*m_xz_I*rho_I + m_yy_I*rho_I + 2*m_yz_I*rho_I + m_zz_I*rho_I - omegaVar*m_xx_I*rho_I - 2*omegaVar*m_xy_I*rho_I - 2*omegaVar*m_xz_I*rho_I - omegaVar*m_yy_I*rho_I - 2*omegaVar*m_yz_I*rho_I - omegaVar*m_zz_I*rho_I))/(8*ux0 - 6*omegaVar + 8*uy0 + 8*uz0 + 12*omegaVar*ux0 + 12*omegaVar*uy0 + 12*omegaVar*uz0 + 15*omegaVar*ux0*ux0 + 15*omegaVar*uy0*uy0 + 15*omegaVar*uz0*uz0 + 30*omegaVar*ux0*uy0 + 30*omegaVar*ux0*uz0 + 30*omegaVar*uy0*uz0 - 104);

    m_xx_t45 = (2*(rho + 27*m_xx_I*rho_I + 6*m_xy_I*rho_I + 6*m_xz_I*rho_I - 3*m_yz_I*rho_I - 4*rho*ux0 - rho*uy0 - rho*uz0))/(45*rho);
    m_xy_t90 = (2*rho + 6*m_xx_I*rho_I + 63*m_xy_I*rho_I + 6*m_yy_I*rho_I - 3*m_zz_I*rho_I - 6*rho*ux0 - 6*rho*uy0)/(45*rho);
    m_xz_t90 = (2*rho + 6*m_xx_I*rho_I + 63*m_xz_I*rho_I - 3*m_yy_I*rho_I + 6*m_zz_I*rho_I - 6*rho*ux0 - 6*rho*uz0)/(45*rho);
    m_yy_t45 = (2*(rho + 6*m_xy_I*rho_I - 3*m_xz_I*rho_I + 27*m_yy_I*rho_I + 6*m_yz_I*rho_I - rho*ux0 - 4*rho*uy0 - rho*uz0))/(45*rho);
    m_yz_t90 = (2*rho - 3*m_xx_I*rho_I + 6*m_yy_I*rho_I + 63*m_yz_I*rho_I + 6*m_zz_I*rho_I - 6*rho*uy0 - 6*rho*uz0)/(45*rho);
    m_zz_t45 = (2*(rho - 3*m_xy_I*rho_I + 6*m_xz_I*rho_I + 6*m_yz_I*rho_I + 27*m_zz_I*rho_I - rho*ux0 - rho*uy0 - 4*rho*uz0))/(45*rho);

    rhoVar = rho;

    break;
case 0b0000010111101001:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[9] + pop[11] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[9] - (1.0/3.0)*pop[11] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] - (1.0/3.0)*pop[9] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[11] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = -(24*(5*rho_I + m_xx_I*rho_I + 2*m_xy_I*rho_I + 2*m_xz_I*rho_I + m_yy_I*rho_I + 2*m_yz_I*rho_I + m_zz_I*rho_I - omegaVar*m_xx_I*rho_I - 2*omegaVar*m_xy_I*rho_I - 2*omegaVar*m_xz_I*rho_I - omegaVar*m_yy_I*rho_I - 2*omegaVar*m_yz_I*rho_I - omegaVar*m_zz_I*rho_I))/(8*ux0 - 6*omegaVar + 8*uy0 + 8*uz0 + 12*omegaVar*ux0 + 12*omegaVar*uy0 + 12*omegaVar*uz0 + 15*omegaVar*ux0*ux0 + 15*omegaVar*uy0*uy0 + 15*omegaVar*uz0*uz0 + 30*omegaVar*ux0*uy0 + 30*omegaVar*ux0*uz0 + 30*omegaVar*uy0*uz0 - 104);

    m_xx_t45 = (2*(rho + 27*m_xx_I*rho_I + 6*m_xy_I*rho_I + 6*m_xz_I*rho_I - 3*m_yz_I*rho_I - 4*rho*ux0 - rho*uy0 - rho*uz0))/(45*rho);
    m_xy_t90 = (2*rho + 6*m_xx_I*rho_I + 63*m_xy_I*rho_I + 6*m_yy_I*rho_I - 3*m_zz_I*rho_I - 6*rho*ux0 - 6*rho*uy0)/(45*rho);
    m_xz_t90 = (2*rho + 6*m_xx_I*rho_I + 63*m_xz_I*rho_I - 3*m_yy_I*rho_I + 6*m_zz_I*rho_I - 6*rho*ux0 - 6*rho*uz0)/(45*rho);
    m_yy_t45 = (2*(rho + 6*m_xy_I*rho_I - 3*m_xz_I*rho_I + 27*m_yy_I*rho_I + 6*m_yz_I*rho_I - rho*ux0 - 4*rho*uy0 - rho*uz0))/(45*rho);
    m_yz_t90 = (2*rho - 3*m_xx_I*rho_I + 6*m_yy_I*rho_I + 63*m_yz_I*rho_I + 6*m_zz_I*rho_I - 6*rho*uy0 - 6*rho*uz0)/(45*rho);
    m_zz_t45 = (2*(rho - 3*m_xy_I*rho_I + 6*m_xz_I*rho_I + 6*m_yz_I*rho_I + 27*m_zz_I*rho_I - rho*ux0 - rho*uy0 - 4*rho*uz0))/(45*rho);

    rhoVar = rho;

    break;
case 0b0000010111101010:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[9] + pop[11] + pop[13] + pop[15] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[9] - (1.0/3.0)*pop[11] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[15] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] - pop[13]);
    m_xz_I = inv_rho_I * (pop[9] - pop[15]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] - (1.0/3.0)*pop[9] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[13] - (1.0/3.0)*pop[15] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[11] - (1.0/3.0)*pop[13] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = -(72*(107*rho_I + 72*m_xx_I*rho_I + 12*m_yy_I*rho_I + 33*m_yz_I*rho_I + 12*m_zz_I*rho_I - 72*omegaVar*m_xx_I*rho_I - 12*omegaVar*m_yy_I*rho_I - 33*omegaVar*m_yz_I*rho_I - 12*omegaVar*m_zz_I*rho_I))/(2124*ux0 - 626*omegaVar + 492*uy0 + 492*uz0 + 1728*omegaVar*ux0 + 150*omegaVar*uy0 + 150*omegaVar*uz0 + 3531*omegaVar*ux0*ux0 + 642*omegaVar*uy0*uy0 + 642*omegaVar*uz0*uz0 + 1926*omegaVar*uy0*uz0 - 5580);

    m_xx_t45 = (2*(24*rho + 249*m_xx_I*rho_I - 12*m_yy_I*rho_I - 33*m_yz_I*rho_I - 12*m_zz_I*rho_I - 83*rho*ux0 + 11*rho*uy0 + 11*rho*uz0))/(321*rho);
    m_xy_t90 = (6*m_xy_I*rho_I - rho*uy0)/(3*rho);
    m_xz_t90 = (6*m_xz_I*rho_I - rho*uz0)/(3*rho);
    m_yy_t45 = (4*(2*rho - 6*m_xx_I*rho_I + 106*m_yy_I*rho_I + 24*m_yz_I*rho_I - m_zz_I*rho_I + 2*rho*ux0 - 8*rho*uy0 - 8*rho*uz0))/(321*rho);
    m_yz_t90 = (11*rho - 33*m_xx_I*rho_I + 48*m_yy_I*rho_I + 453*m_yz_I*rho_I + 48*m_zz_I*rho_I + 11*rho*ux0 - 44*rho*uy0 - 44*rho*uz0)/(321*rho);
    m_zz_t45 = (4*(2*rho - 6*m_xx_I*rho_I - m_yy_I*rho_I + 24*m_yz_I*rho_I + 106*m_zz_I*rho_I + 2*rho*ux0 - 8*rho*uy0 - 8*rho*uz0))/(321*rho);

    rhoVar = rho;

    break;
case 0b0000010111101011:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[9] + pop[13] + pop[15] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[15] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] - pop[13]);
    m_xz_I = inv_rho_I * (pop[9] - pop[15]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] - (1.0/3.0)*pop[9] + (2.0/3.0)*pop[13] - (1.0/3.0)*pop[15] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (-pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] + (2.0/3.0)*pop[9] - (1.0/3.0)*pop[13] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    rho = -(36*(19*rho_I + 6*m_xx_I*rho_I + 12*m_yy_I*rho_I + 33*m_yz_I*rho_I + 12*m_zz_I*rho_I - 6*omegaVar*m_xx_I*rho_I - 12*omegaVar*m_yy_I*rho_I - 33*omegaVar*m_yz_I*rho_I - 12*omegaVar*m_zz_I*rho_I))/(270*ux0 - 118*omegaVar + 72*omegaVar*ux0 + 285*omegaVar*ux0*ux0 + 114*omegaVar*uy0*uy0 + 114*omegaVar*uz0*uz0 + 342*omegaVar*uy0*uz0 - 414);

    m_xx_t45 = -(2*(12*m_yy_I*rho_I - 51*m_xx_I*rho_I - 2*rho + 33*m_yz_I*rho_I + 12*m_zz_I*rho_I + 17*rho*ux0))/(57*rho);
    m_xy_t90 = (6*m_xy_I*rho_I - rho*uy0)/(3*rho);
    m_xz_t90 = (6*m_xz_I*rho_I - rho*uz0)/(3*rho);
    m_yy_t45 = (4*(2*rho - 6*m_xx_I*rho_I + 26*m_yy_I*rho_I + 24*m_yz_I*rho_I + 7*m_zz_I*rho_I + 2*rho*ux0))/(57*rho);
    m_yz_t90 = (11*rho - 33*m_xx_I*rho_I + 48*m_yy_I*rho_I + 189*m_yz_I*rho_I + 48*m_zz_I*rho_I + 11*rho*ux0)/(57*rho);
    m_zz_t45 = (4*(2*rho - 6*m_xx_I*rho_I + 7*m_yy_I*rho_I + 24*m_yz_I*rho_I + 26*m_zz_I*rho_I + 2*rho*ux0))/(57*rho);

    rhoVar = rho;

    break;
case 0b0000010111101100:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[5] + pop[6] + pop[7] + pop[9] + pop[11] + pop[14] + pop[15] + pop[16] + pop[17];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[9] - (1.0/3.0)*pop[11] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17]);
    m_xy_I = inv_rho_I * (pop[7] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] - (1.0/3.0)*pop[9] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17]);
    m_yz_I = inv_rho_I * (pop[11] - pop[17]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[11] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17]);

    rho = -(72*(107*rho_I + 12*m_xx_I*rho_I + 33*m_xz_I*rho_I + 72*m_yy_I*rho_I + 12*m_zz_I*rho_I - 12*omegaVar*m_xx_I*rho_I - 33*omegaVar*m_xz_I*rho_I - 72*omegaVar*m_yy_I*rho_I - 12*omegaVar*m_zz_I*rho_I))/(492*ux0 - 626*omegaVar + 2124*uy0 + 492*uz0 + 150*omegaVar*ux0 + 1728*omegaVar*uy0 + 150*omegaVar*uz0 + 642*omegaVar*ux0*ux0 + 3531*omegaVar*uy0*uy0 + 642*omegaVar*uz0*uz0 + 1926*omegaVar*ux0*uz0 - 5580);

    m_xx_t45 = (4*(2*rho + 106*m_xx_I*rho_I + 24*m_xz_I*rho_I - 6*m_yy_I*rho_I - m_zz_I*rho_I - 8*rho*ux0 + 2*rho*uy0 - 8*rho*uz0))/(321*rho);
    m_xy_t90 = (6*m_xy_I*rho_I - rho*ux0)/(3*rho);
    m_xz_t90 = (11*rho + 48*m_xx_I*rho_I + 453*m_xz_I*rho_I - 33*m_yy_I*rho_I + 48*m_zz_I*rho_I - 44*rho*ux0 + 11*rho*uy0 - 44*rho*uz0)/(321*rho);
    m_yy_t45 = (2*(24*rho - 12*m_xx_I*rho_I - 33*m_xz_I*rho_I + 249*m_yy_I*rho_I - 12*m_zz_I*rho_I + 11*rho*ux0 - 83*rho*uy0 + 11*rho*uz0))/(321*rho);
    m_yz_t90 = (6*m_yz_I*rho_I - rho*uz0)/(3*rho);
    m_zz_t45 = (4*(2*rho - m_xx_I*rho_I + 24*m_xz_I*rho_I - 6*m_yy_I*rho_I + 106*m_zz_I*rho_I - 8*rho*ux0 + 2*rho*uy0 - 8*rho*uz0))/(321*rho);

    rhoVar = rho;

    break;
case 0b0000010111101101:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[5] + pop[6] + pop[7] + pop[11] + pop[14] + pop[15] + pop[16] + pop[17];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] - (1.0/3.0)*pop[11] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17]);
    m_xy_I = inv_rho_I * (pop[7] - pop[14]);
    m_xz_I = inv_rho_I * (-pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17]);
    m_yz_I = inv_rho_I * (pop[11] - pop[17]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] + (2.0/3.0)*pop[11] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17]);

    rho = -(36*(19*rho_I + 12*m_xx_I*rho_I + 33*m_xz_I*rho_I + 6*m_yy_I*rho_I + 12*m_zz_I*rho_I - 12*omegaVar*m_xx_I*rho_I - 33*omegaVar*m_xz_I*rho_I - 6*omegaVar*m_yy_I*rho_I - 12*omegaVar*m_zz_I*rho_I))/(270*uy0 - 118*omegaVar + 72*omegaVar*uy0 + 114*omegaVar*ux0*ux0 + 285*omegaVar*uy0*uy0 + 114*omegaVar*uz0*uz0 + 342*omegaVar*ux0*uz0 - 414);

    m_xx_t45 = (4*(2*rho + 26*m_xx_I*rho_I + 24*m_xz_I*rho_I - 6*m_yy_I*rho_I + 7*m_zz_I*rho_I + 2*rho*uy0))/(57*rho);
    m_xy_t90 = (6*m_xy_I*rho_I - rho*ux0)/(3*rho);
    m_xz_t90 = (11*rho + 48*m_xx_I*rho_I + 189*m_xz_I*rho_I - 33*m_yy_I*rho_I + 48*m_zz_I*rho_I + 11*rho*uy0)/(57*rho);
    m_yy_t45 = -(2*(12*m_xx_I*rho_I - 2*rho + 33*m_xz_I*rho_I - 51*m_yy_I*rho_I + 12*m_zz_I*rho_I + 17*rho*uy0))/(57*rho);
    m_yz_t90 = (6*m_yz_I*rho_I - rho*uz0)/(3*rho);
    m_zz_t45 = (4*(2*rho + 7*m_xx_I*rho_I + 24*m_xz_I*rho_I - 6*m_yy_I*rho_I + 26*m_zz_I*rho_I + 2*rho*uy0))/(57*rho);

    rhoVar = rho;

    break;
case 0b0000010111101110:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[3] + pop[5] + pop[6] + pop[7] + pop[9] + pop[11] + pop[15] + pop[17];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[9] - (1.0/3.0)*pop[11] + (2.0/3.0)*pop[15] - (1.0/3.0)*pop[17]);
    m_xy_I = inv_rho_I * (pop[7]);
    m_xz_I = inv_rho_I * (pop[9] - pop[15]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] + (2.0/3.0)*pop[3] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] - (1.0/3.0)*pop[9] + (2.0/3.0)*pop[11] - (1.0/3.0)*pop[15] + (2.0/3.0)*pop[17]);
    m_yz_I = inv_rho_I * (pop[11] - pop[17]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[3] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[17]);

    rho = -(72*(23*rho_I + 24*m_xx_I*rho_I - 57*m_xy_I*rho_I + 24*m_yy_I*rho_I - 6*m_zz_I*rho_I - 24*omegaVar*m_xx_I*rho_I + 57*omegaVar*m_xy_I*rho_I - 24*omegaVar*m_yy_I*rho_I + 6*omegaVar*m_zz_I*rho_I))/(660*ux0 - 430*omegaVar + 660*uy0 + 30*omegaVar*ux0 + 30*omegaVar*uy0 + 690*omegaVar*ux0*ux0 + 690*omegaVar*uy0*uy0 + 69*omegaVar*uz0*uz0 - 414*omegaVar*ux0*uy0 - 720);

    m_xx_t45 = (16*rho + 141*m_xx_I*rho_I - 102*m_xy_I*rho_I + 3*m_yy_I*rho_I - 18*m_zz_I*rho_I - 30*rho*ux0 + 16*rho*uy0)/(69*rho);
    m_xy_t90 = -(19*rho + 51*m_xx_I*rho_I - 354*m_xy_I*rho_I + 51*m_yy_I*rho_I - 30*m_zz_I*rho_I + 19*rho*ux0 + 19*rho*uy0)/(69*rho);
    m_xz_t90 = (6*m_xz_I*rho_I - rho*uz0)/(3*rho);
    m_yy_t45 = (16*rho + 3*m_xx_I*rho_I - 102*m_xy_I*rho_I + 141*m_yy_I*rho_I - 18*m_zz_I*rho_I + 16*rho*ux0 - 30*rho*uy0)/(69*rho);
    m_yz_t90 = (6*m_yz_I*rho_I - rho*uz0)/(3*rho);
    m_zz_t45 = -(2*(2*rho + 9*m_xx_I*rho_I - 30*m_xy_I*rho_I + 9*m_yy_I*rho_I - 54*m_zz_I*rho_I + 2*rho*ux0 + 2*rho*uy0))/(69*rho);

    rhoVar = rho;

    break;
case 0b0000010111101111:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[3] + pop[6] + pop[7] + pop[15] + pop[17];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[15] - (1.0/3.0)*pop[17]);
    m_xy_I = inv_rho_I * (pop[7]);
    m_xz_I = inv_rho_I * (-pop[15]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] + (2.0/3.0)*pop[3] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] - (1.0/3.0)*pop[15] + (2.0/3.0)*pop[17]);
    m_yz_I = inv_rho_I * (-pop[17]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[3] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[17]);

    rho = -(24*(rho_I + m_xx_I*rho_I - 2*m_xy_I*rho_I + 2*m_xz_I*rho_I + m_yy_I*rho_I + 2*m_yz_I*rho_I + m_zz_I*rho_I - omegaVar*m_xx_I*rho_I + 2*omegaVar*m_xy_I*rho_I - 2*omegaVar*m_xz_I*rho_I - omegaVar*m_yy_I*rho_I - 2*omegaVar*m_yz_I*rho_I - omegaVar*m_zz_I*rho_I))/(12*ux0 - 10*omegaVar + 12*uy0 - 12*uz0 - 4*omegaVar*ux0 - 4*omegaVar*uy0 + 4*omegaVar*uz0 + 9*omegaVar*ux0*ux0 + 9*omegaVar*uy0*uy0 + 9*omegaVar*uz0*uz0 - 6*omegaVar*ux0*uy0 + 6*omegaVar*ux0*uz0 + 6*omegaVar*uy0*uz0 - 4);

    m_xx_t45 = -(2*(9*m_xy_I*rho_I - 15*m_xx_I*rho_I - rho - 9*m_xz_I*rho_I + 3*m_yy_I*rho_I + 9*m_yz_I*rho_I + 3*m_zz_I*rho_I + 2*rho*ux0 - rho*uy0 + rho*uz0))/(9*rho);
    m_xy_t90 = -(2*rho + 9*m_xx_I*rho_I - 51*m_xy_I*rho_I - 3*m_xz_I*rho_I + 9*m_yy_I*rho_I - 3*m_yz_I*rho_I - 9*m_zz_I*rho_I + 2*rho*ux0 + 2*rho*uy0 - 2*rho*uz0)/(9*rho);
    m_xz_t90 = (2*rho + 9*m_xx_I*rho_I + 3*m_xy_I*rho_I + 51*m_xz_I*rho_I - 9*m_yy_I*rho_I - 3*m_yz_I*rho_I + 9*m_zz_I*rho_I + 2*rho*ux0 + 2*rho*uy0 - 2*rho*uz0)/(9*rho);
    m_yy_t45 = -(2*(3*m_xx_I*rho_I - rho + 9*m_xy_I*rho_I + 9*m_xz_I*rho_I - 15*m_yy_I*rho_I - 9*m_yz_I*rho_I + 3*m_zz_I*rho_I - rho*ux0 + 2*rho*uy0 + rho*uz0))/(9*rho);
    m_yz_t90 = (2*rho - 9*m_xx_I*rho_I + 3*m_xy_I*rho_I - 3*m_xz_I*rho_I + 9*m_yy_I*rho_I + 51*m_yz_I*rho_I + 9*m_zz_I*rho_I + 2*rho*ux0 + 2*rho*uy0 - 2*rho*uz0)/(9*rho);
    m_zz_t45 = (2*(rho - 3*m_xx_I*rho_I + 9*m_xy_I*rho_I + 9*m_xz_I*rho_I - 3*m_yy_I*rho_I + 9*m_yz_I*rho_I + 15*m_zz_I*rho_I + rho*ux0 + rho*uy0 + 2*rho*uz0))/(9*rho);

    rhoVar = rho;

    break;
case 0b0000010111110000:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[7] + pop[8] + pop[9] + pop[11] + pop[13] + pop[14] + pop[16] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] - (1.0/3.0)*pop[11] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[11] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);

    rho = -(3*rho_I*(3*m_zz_I - 3*omegaVar*m_zz_I + 4))/(3*uz0 - omegaVar + 3*omegaVar*uz0 + 6*omegaVar*uz0*uz0 - 9);

    m_xx_t45 = (4*(10*m_xx_I*rho_I - m_yy_I*rho_I))/(33*rho);
    m_xy_t90 = (m_xy_I*rho_I)/rho;
    m_xz_t90 = (6*m_xz_I*rho_I - rho*ux0)/(3*rho);
    m_yy_t45 = -(4*(m_xx_I*rho_I - 10*m_yy_I*rho_I))/(33*rho);
    m_yz_t90 = (6*m_yz_I*rho_I - rho*uy0)/(3*rho);
    m_zz_t45 = (rho + 9*m_zz_I*rho_I - 3*rho*uz0)/(6*rho);

    rhoVar = rho;

    break;
case 0b0000010111110001:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[8] + pop[9] + pop[11] + pop[13] + pop[14] + pop[16] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] - (1.0/3.0)*pop[11] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[11] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);

    rho = (72*(107*rho_I + 12*m_xx_I*rho_I + 33*m_xy_I*rho_I + 12*m_yy_I*rho_I + 72*m_zz_I*rho_I - 12*omegaVar*m_xx_I*rho_I - 33*omegaVar*m_xy_I*rho_I - 12*omegaVar*m_yy_I*rho_I - 72*omegaVar*m_zz_I*rho_I))/(626*omegaVar + 492*ux0 + 492*uy0 - 2124*uz0 + 150*omegaVar*ux0 + 150*omegaVar*uy0 - 1728*omegaVar*uz0 - 642*omegaVar*ux0*ux0 - 642*omegaVar*uy0*uy0 - 3531*omegaVar*uz0*uz0 - 1926*omegaVar*ux0*uy0 + 5580);

    m_xx_t45 = (4*(2*rho + 106*m_xx_I*rho_I + 24*m_xy_I*rho_I - m_yy_I*rho_I - 6*m_zz_I*rho_I + 8*rho*ux0 + 8*rho*uy0 + 2*rho*uz0))/(321*rho);
    m_xy_t90 = (11*rho + 48*m_xx_I*rho_I + 453*m_xy_I*rho_I + 48*m_yy_I*rho_I - 33*m_zz_I*rho_I + 44*rho*ux0 + 44*rho*uy0 + 11*rho*uz0)/(321*rho);
    m_xz_t90 = (6*m_xz_I*rho_I - rho*ux0)/(3*rho);
    m_yy_t45 = (4*(2*rho - m_xx_I*rho_I + 24*m_xy_I*rho_I + 106*m_yy_I*rho_I - 6*m_zz_I*rho_I + 8*rho*ux0 + 8*rho*uy0 + 2*rho*uz0))/(321*rho);
    m_yz_t90 = (6*m_yz_I*rho_I - rho*uy0)/(3*rho);
    m_zz_t45 = -(2*(12*m_xx_I*rho_I - 24*rho + 33*m_xy_I*rho_I + 12*m_yy_I*rho_I - 249*m_zz_I*rho_I + 11*rho*ux0 + 11*rho*uy0 + 83*rho*uz0))/(321*rho);

    rhoVar = rho;

    break;
case 0b0000010111110010:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[7] + pop[8] + pop[9] + pop[11] + pop[13] + pop[16] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] - (1.0/3.0)*pop[11] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13]);
    m_xz_I = inv_rho_I * (pop[9] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[13] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[11] - (1.0/3.0)*pop[13] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);

    rho = -(72*(107*rho_I + 12*m_xx_I*rho_I - 33*m_xy_I*rho_I + 12*m_yy_I*rho_I + 72*m_zz_I*rho_I - 12*omegaVar*m_xx_I*rho_I + 33*omegaVar*m_xy_I*rho_I - 12*omegaVar*m_yy_I*rho_I - 72*omegaVar*m_zz_I*rho_I))/(492*ux0 - 626*omegaVar - 492*uy0 + 2124*uz0 + 150*omegaVar*ux0 - 150*omegaVar*uy0 + 1728*omegaVar*uz0 + 642*omegaVar*ux0*ux0 + 642*omegaVar*uy0*uy0 + 3531*omegaVar*uz0*uz0 - 1926*omegaVar*ux0*uy0 - 5580);

    m_xx_t45 = (4*(2*rho + 106*m_xx_I*rho_I - 24*m_xy_I*rho_I - m_yy_I*rho_I - 6*m_zz_I*rho_I - 8*rho*ux0 + 8*rho*uy0 + 2*rho*uz0))/(321*rho);
    m_xy_t90 = -(11*rho + 48*m_xx_I*rho_I - 453*m_xy_I*rho_I + 48*m_yy_I*rho_I - 33*m_zz_I*rho_I - 44*rho*ux0 + 44*rho*uy0 + 11*rho*uz0)/(321*rho);
    m_xz_t90 = (6*m_xz_I*rho_I - rho*ux0)/(3*rho);
    m_yy_t45 = (4*(2*rho - m_xx_I*rho_I - 24*m_xy_I*rho_I + 106*m_yy_I*rho_I - 6*m_zz_I*rho_I - 8*rho*ux0 + 8*rho*uy0 + 2*rho*uz0))/(321*rho);
    m_yz_t90 = (6*m_yz_I*rho_I - rho*uy0)/(3*rho);
    m_zz_t45 = (2*(24*rho - 12*m_xx_I*rho_I + 33*m_xy_I*rho_I - 12*m_yy_I*rho_I + 249*m_zz_I*rho_I + 11*rho*ux0 - 11*rho*uy0 - 83*rho*uz0))/(321*rho);

    rhoVar = rho;

    break;
case 0b0000010111110011:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[4] + pop[5] + pop[8] + pop[9] + pop[13] + pop[16] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[8] - pop[13]);
    m_xz_I = inv_rho_I * (pop[9] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] + (2.0/3.0)*pop[13] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (-pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] - (1.0/3.0)*pop[13] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);

    rho = -(72*(23*rho_I - 6*m_xx_I*rho_I + 24*m_yy_I*rho_I + 57*m_yz_I*rho_I + 24*m_zz_I*rho_I + 6*omegaVar*m_xx_I*rho_I - 24*omegaVar*m_yy_I*rho_I - 57*omegaVar*m_yz_I*rho_I - 24*omegaVar*m_zz_I*rho_I))/(660*uz0 - 660*uy0 - 430*omegaVar - 30*omegaVar*uy0 + 30*omegaVar*uz0 + 69*omegaVar*ux0*ux0 + 690*omegaVar*uy0*uy0 + 690*omegaVar*uz0*uz0 + 414*omegaVar*uy0*uz0 - 720);

    m_xx_t45 = -(2*(2*rho - 54*m_xx_I*rho_I + 9*m_yy_I*rho_I + 30*m_yz_I*rho_I + 9*m_zz_I*rho_I - 2*rho*uy0 + 2*rho*uz0))/(69*rho);
    m_xy_t90 = (6*m_xy_I*rho_I + rho*ux0)/(3*rho);
    m_xz_t90 = (6*m_xz_I*rho_I - rho*ux0)/(3*rho);
    m_yy_t45 = (16*rho - 18*m_xx_I*rho_I + 141*m_yy_I*rho_I + 102*m_yz_I*rho_I + 3*m_zz_I*rho_I + 30*rho*uy0 + 16*rho*uz0)/(69*rho);
    m_yz_t90 = (19*rho - 30*m_xx_I*rho_I + 51*m_yy_I*rho_I + 354*m_yz_I*rho_I + 51*m_zz_I*rho_I - 19*rho*uy0 + 19*rho*uz0)/(69*rho);
    m_zz_t45 = (16*rho - 18*m_xx_I*rho_I + 3*m_yy_I*rho_I + 102*m_yz_I*rho_I + 141*m_zz_I*rho_I - 16*rho*uy0 - 30*rho*uz0)/(69*rho);

    rhoVar = rho;

    break;
case 0b0000010111110100:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[7] + pop[8] + pop[9] + pop[11] + pop[14] + pop[16] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] - (1.0/3.0)*pop[11] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[11] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);

    rho = -(72*(107*rho_I + 12*m_xx_I*rho_I - 33*m_xy_I*rho_I + 12*m_yy_I*rho_I + 72*m_zz_I*rho_I - 12*omegaVar*m_xx_I*rho_I + 33*omegaVar*m_xy_I*rho_I - 12*omegaVar*m_yy_I*rho_I - 72*omegaVar*m_zz_I*rho_I))/(492*uy0 - 492*ux0 - 626*omegaVar + 2124*uz0 - 150*omegaVar*ux0 + 150*omegaVar*uy0 + 1728*omegaVar*uz0 + 642*omegaVar*ux0*ux0 + 642*omegaVar*uy0*uy0 + 3531*omegaVar*uz0*uz0 - 1926*omegaVar*ux0*uy0 - 5580);

    m_xx_t45 = (4*(2*rho + 106*m_xx_I*rho_I - 24*m_xy_I*rho_I - m_yy_I*rho_I - 6*m_zz_I*rho_I + 8*rho*ux0 - 8*rho*uy0 + 2*rho*uz0))/(321*rho);
    m_xy_t90 = -(11*rho + 48*m_xx_I*rho_I - 453*m_xy_I*rho_I + 48*m_yy_I*rho_I - 33*m_zz_I*rho_I + 44*rho*ux0 - 44*rho*uy0 + 11*rho*uz0)/(321*rho);
    m_xz_t90 = (6*m_xz_I*rho_I - rho*ux0)/(3*rho);
    m_yy_t45 = (4*(2*rho - m_xx_I*rho_I - 24*m_xy_I*rho_I + 106*m_yy_I*rho_I - 6*m_zz_I*rho_I + 8*rho*ux0 - 8*rho*uy0 + 2*rho*uz0))/(321*rho);
    m_yz_t90 = (6*m_yz_I*rho_I - rho*uy0)/(3*rho);
    m_zz_t45 = (2*(24*rho - 12*m_xx_I*rho_I + 33*m_xy_I*rho_I - 12*m_yy_I*rho_I + 249*m_zz_I*rho_I - 11*rho*ux0 + 11*rho*uy0 - 83*rho*uz0))/(321*rho);

    rhoVar = rho;

    break;
case 0b0000010111110101:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[2] + pop[3] + pop[4] + pop[5] + pop[8] + pop[11] + pop[14] + pop[16] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[11] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[8] - pop[14]);
    m_xz_I = inv_rho_I * (-pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[11] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);

    rho = -(72*(23*rho_I + 24*m_xx_I*rho_I + 57*m_xz_I*rho_I - 6*m_yy_I*rho_I + 24*m_zz_I*rho_I - 24*omegaVar*m_xx_I*rho_I - 57*omegaVar*m_xz_I*rho_I + 6*omegaVar*m_yy_I*rho_I - 24*omegaVar*m_zz_I*rho_I))/(660*uz0 - 660*ux0 - 430*omegaVar - 30*omegaVar*ux0 + 30*omegaVar*uz0 + 690*omegaVar*ux0*ux0 + 69*omegaVar*uy0*uy0 + 690*omegaVar*uz0*uz0 + 414*omegaVar*ux0*uz0 - 720);

    m_xx_t45 = (16*rho + 141*m_xx_I*rho_I + 102*m_xz_I*rho_I - 18*m_yy_I*rho_I + 3*m_zz_I*rho_I + 30*rho*ux0 + 16*rho*uz0)/(69*rho);
    m_xy_t90 = (6*m_xy_I*rho_I + rho*uy0)/(3*rho);
    m_xz_t90 = (19*rho + 51*m_xx_I*rho_I + 354*m_xz_I*rho_I - 30*m_yy_I*rho_I + 51*m_zz_I*rho_I - 19*rho*ux0 + 19*rho*uz0)/(69*rho);
    m_yy_t45 = -(2*(2*rho + 9*m_xx_I*rho_I + 30*m_xz_I*rho_I - 54*m_yy_I*rho_I + 9*m_zz_I*rho_I - 2*rho*ux0 + 2*rho*uz0))/(69*rho);
    m_yz_t90 = (6*m_yz_I*rho_I - rho*uy0)/(3*rho);
    m_zz_t45 = (16*rho + 3*m_xx_I*rho_I + 102*m_xz_I*rho_I - 18*m_yy_I*rho_I + 141*m_zz_I*rho_I - 16*rho*ux0 - 30*rho*uz0)/(69*rho);

    rhoVar = rho;

    break;
case 0b0000010111110110:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[7] + pop[8] + pop[9] + pop[11] + pop[16] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] - (1.0/3.0)*pop[11] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8]);
    m_xz_I = inv_rho_I * (pop[9] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] + (2.0/3.0)*pop[11] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);

    rho = -(36*(19*rho_I + 12*m_xx_I*rho_I - 33*m_xy_I*rho_I + 12*m_yy_I*rho_I + 6*m_zz_I*rho_I - 12*omegaVar*m_xx_I*rho_I + 33*omegaVar*m_xy_I*rho_I - 12*omegaVar*m_yy_I*rho_I - 6*omegaVar*m_zz_I*rho_I))/(270*uz0 - 118*omegaVar + 72*omegaVar*uz0 + 114*omegaVar*ux0*ux0 + 114*omegaVar*uy0*uy0 + 285*omegaVar*uz0*uz0 - 342*omegaVar*ux0*uy0 - 414);

    m_xx_t45 = (4*(2*rho + 26*m_xx_I*rho_I - 24*m_xy_I*rho_I + 7*m_yy_I*rho_I - 6*m_zz_I*rho_I + 2*rho*uz0))/(57*rho);
    m_xy_t90 = -(11*rho + 48*m_xx_I*rho_I - 189*m_xy_I*rho_I + 48*m_yy_I*rho_I - 33*m_zz_I*rho_I + 11*rho*uz0)/(57*rho);
    m_xz_t90 = (6*m_xz_I*rho_I - rho*ux0)/(3*rho);
    m_yy_t45 = (4*(2*rho + 7*m_xx_I*rho_I - 24*m_xy_I*rho_I + 26*m_yy_I*rho_I - 6*m_zz_I*rho_I + 2*rho*uz0))/(57*rho);
    m_yz_t90 = (6*m_yz_I*rho_I - rho*uy0)/(3*rho);
    m_zz_t45 = (2*(2*rho - 12*m_xx_I*rho_I + 33*m_xy_I*rho_I - 12*m_yy_I*rho_I + 51*m_zz_I*rho_I - 17*rho*uz0))/(57*rho);

    rhoVar = rho;

    break;
case 0b0000010111110111:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[2] + pop[4] + pop[5] + pop[8] + pop[16] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[8]);
    m_xz_I = inv_rho_I * (-pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (-pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);

    rho = -(24*(rho_I + m_xx_I*rho_I - 2*m_xy_I*rho_I + 2*m_xz_I*rho_I + m_yy_I*rho_I + 2*m_yz_I*rho_I + m_zz_I*rho_I - omegaVar*m_xx_I*rho_I + 2*omegaVar*m_xy_I*rho_I - 2*omegaVar*m_xz_I*rho_I - omegaVar*m_yy_I*rho_I - 2*omegaVar*m_yz_I*rho_I - omegaVar*m_zz_I*rho_I))/(12*uz0 - 12*ux0 - 12*uy0 - 10*omegaVar + 4*omegaVar*ux0 + 4*omegaVar*uy0 - 4*omegaVar*uz0 + 9*omegaVar*ux0*ux0 + 9*omegaVar*uy0*uy0 + 9*omegaVar*uz0*uz0 - 6*omegaVar*ux0*uy0 + 6*omegaVar*ux0*uz0 + 6*omegaVar*uy0*uz0 - 4);

    m_xx_t45 = (2*(rho + 15*m_xx_I*rho_I - 9*m_xy_I*rho_I + 9*m_xz_I*rho_I - 3*m_yy_I*rho_I - 9*m_yz_I*rho_I - 3*m_zz_I*rho_I + 2*rho*ux0 - rho*uy0 + rho*uz0))/(9*rho);
    m_xy_t90 = (51*m_xy_I*rho_I - 9*m_xx_I*rho_I - 2*rho + 3*m_xz_I*rho_I - 9*m_yy_I*rho_I + 3*m_yz_I*rho_I + 9*m_zz_I*rho_I + 2*rho*ux0 + 2*rho*uy0 - 2*rho*uz0)/(9*rho);
    m_xz_t90 = (2*rho + 9*m_xx_I*rho_I + 3*m_xy_I*rho_I + 51*m_xz_I*rho_I - 9*m_yy_I*rho_I - 3*m_yz_I*rho_I + 9*m_zz_I*rho_I - 2*rho*ux0 - 2*rho*uy0 + 2*rho*uz0)/(9*rho);
    m_yy_t45 = (2*(rho - 3*m_xx_I*rho_I - 9*m_xy_I*rho_I - 9*m_xz_I*rho_I + 15*m_yy_I*rho_I + 9*m_yz_I*rho_I - 3*m_zz_I*rho_I - rho*ux0 + 2*rho*uy0 + rho*uz0))/(9*rho);
    m_yz_t90 = (2*rho - 9*m_xx_I*rho_I + 3*m_xy_I*rho_I - 3*m_xz_I*rho_I + 9*m_yy_I*rho_I + 51*m_yz_I*rho_I + 9*m_zz_I*rho_I - 2*rho*ux0 - 2*rho*uy0 + 2*rho*uz0)/(9*rho);
    m_zz_t45 = (2*(rho - 3*m_xx_I*rho_I + 9*m_xy_I*rho_I + 9*m_xz_I*rho_I - 3*m_yy_I*rho_I + 9*m_yz_I*rho_I + 15*m_zz_I*rho_I - rho*ux0 - rho*uy0 - 2*rho*uz0))/(9*rho);

    rhoVar = rho;

    break;
case 0b0000010111111000:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[7] + pop[9] + pop[11] + pop[13] + pop[14] + pop[16] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[9] - (1.0/3.0)*pop[11] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] + (2.0/3.0)*pop[7] - (1.0/3.0)*pop[9] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] - (1.0/3.0)*pop[7] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[11] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);

    rho = -(72*(107*rho_I + 12*m_xx_I*rho_I + 33*m_xy_I*rho_I + 12*m_yy_I*rho_I + 72*m_zz_I*rho_I - 12*omegaVar*m_xx_I*rho_I - 33*omegaVar*m_xy_I*rho_I - 12*omegaVar*m_yy_I*rho_I - 72*omegaVar*m_zz_I*rho_I))/(492*ux0 - 626*omegaVar + 492*uy0 + 2124*uz0 + 150*omegaVar*ux0 + 150*omegaVar*uy0 + 1728*omegaVar*uz0 + 642*omegaVar*ux0*ux0 + 642*omegaVar*uy0*uy0 + 3531*omegaVar*uz0*uz0 + 1926*omegaVar*ux0*uy0 - 5580);

    m_xx_t45 = (4*(2*rho + 106*m_xx_I*rho_I + 24*m_xy_I*rho_I - m_yy_I*rho_I - 6*m_zz_I*rho_I - 8*rho*ux0 - 8*rho*uy0 + 2*rho*uz0))/(321*rho);
    m_xy_t90 = (11*rho + 48*m_xx_I*rho_I + 453*m_xy_I*rho_I + 48*m_yy_I*rho_I - 33*m_zz_I*rho_I - 44*rho*ux0 - 44*rho*uy0 + 11*rho*uz0)/(321*rho);
    m_xz_t90 = (6*m_xz_I*rho_I - rho*ux0)/(3*rho);
    m_yy_t45 = (4*(2*rho - m_xx_I*rho_I + 24*m_xy_I*rho_I + 106*m_yy_I*rho_I - 6*m_zz_I*rho_I - 8*rho*ux0 - 8*rho*uy0 + 2*rho*uz0))/(321*rho);
    m_yz_t90 = (6*m_yz_I*rho_I - rho*uy0)/(3*rho);
    m_zz_t45 = (2*(24*rho - 12*m_xx_I*rho_I - 33*m_xy_I*rho_I - 12*m_yy_I*rho_I + 249*m_zz_I*rho_I + 11*rho*ux0 + 11*rho*uy0 - 83*rho*uz0))/(321*rho);

    rhoVar = rho;

    break;
case 0b0000010111111001:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[9] + pop[11] + pop[13] + pop[14] + pop[16] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] + (2.0/3.0)*pop[9] - (1.0/3.0)*pop[11] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (-pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[9] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[11] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);

    rho = -(36*(19*rho_I + 12*m_xx_I*rho_I + 33*m_xy_I*rho_I + 12*m_yy_I*rho_I + 6*m_zz_I*rho_I - 12*omegaVar*m_xx_I*rho_I - 33*omegaVar*m_xy_I*rho_I - 12*omegaVar*m_yy_I*rho_I - 6*omegaVar*m_zz_I*rho_I))/(270*uz0 - 118*omegaVar + 72*omegaVar*uz0 + 114*omegaVar*ux0*ux0 + 114*omegaVar*uy0*uy0 + 285*omegaVar*uz0*uz0 + 342*omegaVar*ux0*uy0 - 414);

    m_xx_t45 = (4*(2*rho + 26*m_xx_I*rho_I + 24*m_xy_I*rho_I + 7*m_yy_I*rho_I - 6*m_zz_I*rho_I + 2*rho*uz0))/(57*rho);
    m_xy_t90 = (11*rho + 48*m_xx_I*rho_I + 189*m_xy_I*rho_I + 48*m_yy_I*rho_I - 33*m_zz_I*rho_I + 11*rho*uz0)/(57*rho);
    m_xz_t90 = (6*m_xz_I*rho_I - rho*ux0)/(3*rho);
    m_yy_t45 = (4*(2*rho + 7*m_xx_I*rho_I + 24*m_xy_I*rho_I + 26*m_yy_I*rho_I - 6*m_zz_I*rho_I + 2*rho*uz0))/(57*rho);
    m_yz_t90 = (6*m_yz_I*rho_I - rho*uy0)/(3*rho);
    m_zz_t45 = -(2*(12*m_xx_I*rho_I - 2*rho + 33*m_xy_I*rho_I + 12*m_yy_I*rho_I - 51*m_zz_I*rho_I + 17*rho*uz0))/(57*rho);

    rhoVar = rho;

    break;
case 0b0000010111111010:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[3] + pop[4] + pop[5] + pop[7] + pop[9] + pop[11] + pop[13] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[9] - (1.0/3.0)*pop[11] + (2.0/3.0)*pop[13] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] - pop[13]);
    m_xz_I = inv_rho_I * (pop[9]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] + (2.0/3.0)*pop[7] - (1.0/3.0)*pop[9] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] - (1.0/3.0)*pop[7] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[11] - (1.0/3.0)*pop[13] + (2.0/3.0)*pop[18]);

    rho = -(72*(23*rho_I + 24*m_xx_I*rho_I - 57*m_xz_I*rho_I - 6*m_yy_I*rho_I + 24*m_zz_I*rho_I - 24*omegaVar*m_xx_I*rho_I + 57*omegaVar*m_xz_I*rho_I + 6*omegaVar*m_yy_I*rho_I - 24*omegaVar*m_zz_I*rho_I))/(660*ux0 - 430*omegaVar + 660*uz0 + 30*omegaVar*ux0 + 30*omegaVar*uz0 + 690*omegaVar*ux0*ux0 + 69*omegaVar*uy0*uy0 + 690*omegaVar*uz0*uz0 - 414*omegaVar*ux0*uz0 - 720);

    m_xx_t45 = (16*rho + 141*m_xx_I*rho_I - 102*m_xz_I*rho_I - 18*m_yy_I*rho_I + 3*m_zz_I*rho_I - 30*rho*ux0 + 16*rho*uz0)/(69*rho);
    m_xy_t90 = (6*m_xy_I*rho_I - rho*uy0)/(3*rho);
    m_xz_t90 = -(19*rho + 51*m_xx_I*rho_I - 354*m_xz_I*rho_I - 30*m_yy_I*rho_I + 51*m_zz_I*rho_I + 19*rho*ux0 + 19*rho*uz0)/(69*rho);
    m_yy_t45 = -(2*(2*rho + 9*m_xx_I*rho_I - 30*m_xz_I*rho_I - 54*m_yy_I*rho_I + 9*m_zz_I*rho_I + 2*rho*ux0 + 2*rho*uz0))/(69*rho);
    m_yz_t90 = (6*m_yz_I*rho_I - rho*uy0)/(3*rho);
    m_zz_t45 = (16*rho + 3*m_xx_I*rho_I - 102*m_xz_I*rho_I - 18*m_yy_I*rho_I + 141*m_zz_I*rho_I + 16*rho*ux0 - 30*rho*uz0)/(69*rho);

    rhoVar = rho;

    break;
case 0b0000010111111011:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[4] + pop[5] + pop[9] + pop[13] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[13] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (-pop[13]);
    m_xz_I = inv_rho_I * (pop[9]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[9] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (-pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[9] - (1.0/3.0)*pop[13] + (2.0/3.0)*pop[18]);

    rho = -(24*(rho_I + m_xx_I*rho_I + 2*m_xy_I*rho_I - 2*m_xz_I*rho_I + m_yy_I*rho_I + 2*m_yz_I*rho_I + m_zz_I*rho_I - omegaVar*m_xx_I*rho_I - 2*omegaVar*m_xy_I*rho_I + 2*omegaVar*m_xz_I*rho_I - omegaVar*m_yy_I*rho_I - 2*omegaVar*m_yz_I*rho_I - omegaVar*m_zz_I*rho_I))/(12*ux0 - 10*omegaVar - 12*uy0 + 12*uz0 - 4*omegaVar*ux0 + 4*omegaVar*uy0 - 4*omegaVar*uz0 + 9*omegaVar*ux0*ux0 + 9*omegaVar*uy0*uy0 + 9*omegaVar*uz0*uz0 + 6*omegaVar*ux0*uy0 - 6*omegaVar*ux0*uz0 + 6*omegaVar*uy0*uz0 - 4);

    m_xx_t45 = -(2*(9*m_xz_I*rho_I - 15*m_xx_I*rho_I - 9*m_xy_I*rho_I - rho + 3*m_yy_I*rho_I + 9*m_yz_I*rho_I + 3*m_zz_I*rho_I + 2*rho*ux0 + rho*uy0 - rho*uz0))/(9*rho);
    m_xy_t90 = (2*rho + 9*m_xx_I*rho_I + 51*m_xy_I*rho_I + 3*m_xz_I*rho_I + 9*m_yy_I*rho_I - 3*m_yz_I*rho_I - 9*m_zz_I*rho_I + 2*rho*ux0 - 2*rho*uy0 + 2*rho*uz0)/(9*rho);
    m_xz_t90 = -(2*rho + 9*m_xx_I*rho_I - 3*m_xy_I*rho_I - 51*m_xz_I*rho_I - 9*m_yy_I*rho_I - 3*m_yz_I*rho_I + 9*m_zz_I*rho_I + 2*rho*ux0 - 2*rho*uy0 + 2*rho*uz0)/(9*rho);
    m_yy_t45 = (2*(rho - 3*m_xx_I*rho_I + 9*m_xy_I*rho_I + 9*m_xz_I*rho_I + 15*m_yy_I*rho_I + 9*m_yz_I*rho_I - 3*m_zz_I*rho_I + rho*ux0 + 2*rho*uy0 + rho*uz0))/(9*rho);
    m_yz_t90 = (2*rho - 9*m_xx_I*rho_I - 3*m_xy_I*rho_I + 3*m_xz_I*rho_I + 9*m_yy_I*rho_I + 51*m_yz_I*rho_I + 9*m_zz_I*rho_I + 2*rho*ux0 - 2*rho*uy0 + 2*rho*uz0)/(9*rho);
    m_zz_t45 = -(2*(3*m_xx_I*rho_I - rho + 9*m_xy_I*rho_I + 9*m_xz_I*rho_I + 3*m_yy_I*rho_I - 9*m_yz_I*rho_I - 15*m_zz_I*rho_I - rho*ux0 + rho*uy0 + 2*rho*uz0))/(9*rho);

    rhoVar = rho;

    break;
case 0b0000010111111100:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[5] + pop[7] + pop[9] + pop[11] + pop[14] + pop[16];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[5] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[9] - (1.0/3.0)*pop[11] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[16]);
    m_xy_I = inv_rho_I * (pop[7] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] - (1.0/3.0)*pop[5] + (2.0/3.0)*pop[7] - (1.0/3.0)*pop[9] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[16]);
    m_yz_I = inv_rho_I * (pop[11]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] + (2.0/3.0)*pop[5] - (1.0/3.0)*pop[7] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[11] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[16]);

    rho = -(72*(23*rho_I - 6*m_xx_I*rho_I + 24*m_yy_I*rho_I - 57*m_yz_I*rho_I + 24*m_zz_I*rho_I + 6*omegaVar*m_xx_I*rho_I - 24*omegaVar*m_yy_I*rho_I + 57*omegaVar*m_yz_I*rho_I - 24*omegaVar*m_zz_I*rho_I))/(660*uy0 - 430*omegaVar + 660*uz0 + 30*omegaVar*uy0 + 30*omegaVar*uz0 + 69*omegaVar*ux0*ux0 + 690*omegaVar*uy0*uy0 + 690*omegaVar*uz0*uz0 - 414*omegaVar*uy0*uz0 - 720);

    m_xx_t45 = -(2*(2*rho - 54*m_xx_I*rho_I + 9*m_yy_I*rho_I - 30*m_yz_I*rho_I + 9*m_zz_I*rho_I + 2*rho*uy0 + 2*rho*uz0))/(69*rho);
    m_xy_t90 = (6*m_xy_I*rho_I - rho*ux0)/(3*rho);
    m_xz_t90 = (6*m_xz_I*rho_I - rho*ux0)/(3*rho);
    m_yy_t45 = (16*rho - 18*m_xx_I*rho_I + 141*m_yy_I*rho_I - 102*m_yz_I*rho_I + 3*m_zz_I*rho_I - 30*rho*uy0 + 16*rho*uz0)/(69*rho);
    m_yz_t90 = -(19*rho - 30*m_xx_I*rho_I + 51*m_yy_I*rho_I - 354*m_yz_I*rho_I + 51*m_zz_I*rho_I + 19*rho*uy0 + 19*rho*uz0)/(69*rho);
    m_zz_t45 = (16*rho - 18*m_xx_I*rho_I + 3*m_yy_I*rho_I - 102*m_yz_I*rho_I + 141*m_zz_I*rho_I + 16*rho*uy0 - 30*rho*uz0)/(69*rho);

    rhoVar = rho;

    break;
case 0b0000010111111101:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[2] + pop[3] + pop[5] + pop[11] + pop[14] + pop[16];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[11] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[16]);
    m_xy_I = inv_rho_I * (-pop[14]);
    m_xz_I = inv_rho_I * (-pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] - (1.0/3.0)*pop[5] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[16]);
    m_yz_I = inv_rho_I * (pop[11]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[11] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[16]);

    rho = -(24*(rho_I + m_xx_I*rho_I + 2*m_xy_I*rho_I + 2*m_xz_I*rho_I + m_yy_I*rho_I - 2*m_yz_I*rho_I + m_zz_I*rho_I - omegaVar*m_xx_I*rho_I - 2*omegaVar*m_xy_I*rho_I - 2*omegaVar*m_xz_I*rho_I - omegaVar*m_yy_I*rho_I + 2*omegaVar*m_yz_I*rho_I - omegaVar*m_zz_I*rho_I))/(12*uy0 - 12*ux0 - 10*omegaVar + 12*uz0 + 4*omegaVar*ux0 - 4*omegaVar*uy0 - 4*omegaVar*uz0 + 9*omegaVar*ux0*ux0 + 9*omegaVar*uy0*uy0 + 9*omegaVar*uz0*uz0 + 6*omegaVar*ux0*uy0 + 6*omegaVar*ux0*uz0 - 6*omegaVar*uy0*uz0 - 4);

    m_xx_t45 = (2*(rho + 15*m_xx_I*rho_I + 9*m_xy_I*rho_I + 9*m_xz_I*rho_I - 3*m_yy_I*rho_I + 9*m_yz_I*rho_I - 3*m_zz_I*rho_I + 2*rho*ux0 + rho*uy0 + rho*uz0))/(9*rho);
    m_xy_t90 = (2*rho + 9*m_xx_I*rho_I + 51*m_xy_I*rho_I - 3*m_xz_I*rho_I + 9*m_yy_I*rho_I + 3*m_yz_I*rho_I - 9*m_zz_I*rho_I - 2*rho*ux0 + 2*rho*uy0 + 2*rho*uz0)/(9*rho);
    m_xz_t90 = (2*rho + 9*m_xx_I*rho_I - 3*m_xy_I*rho_I + 51*m_xz_I*rho_I - 9*m_yy_I*rho_I + 3*m_yz_I*rho_I + 9*m_zz_I*rho_I - 2*rho*ux0 + 2*rho*uy0 + 2*rho*uz0)/(9*rho);
    m_yy_t45 = -(2*(3*m_xx_I*rho_I - rho - 9*m_xy_I*rho_I + 9*m_xz_I*rho_I - 15*m_yy_I*rho_I + 9*m_yz_I*rho_I + 3*m_zz_I*rho_I + rho*ux0 + 2*rho*uy0 - rho*uz0))/(9*rho);
    m_yz_t90 = -(2*rho - 9*m_xx_I*rho_I - 3*m_xy_I*rho_I - 3*m_xz_I*rho_I + 9*m_yy_I*rho_I - 51*m_yz_I*rho_I + 9*m_zz_I*rho_I - 2*rho*ux0 + 2*rho*uy0 + 2*rho*uz0)/(9*rho);
    m_zz_t45 = -(2*(3*m_xx_I*rho_I - rho + 9*m_xy_I*rho_I - 9*m_xz_I*rho_I + 3*m_yy_I*rho_I + 9*m_yz_I*rho_I - 15*m_zz_I*rho_I + rho*ux0 - rho*uy0 + 2*rho*uz0))/(9*rho);

    rhoVar = rho;

    break;
case 0b0000010111111110:
    ux_t30 = ux0;
    uy_t30 = uy0;
    uz_t30 = uz0;

    rho_I = pop[0] + pop[1] + pop[3] + pop[5] + pop[7] + pop[9] + pop[11];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[5] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[9] - (1.0/3.0)*pop[11]);
    m_xy_I = inv_rho_I * (pop[7]);
    m_xz_I = inv_rho_I * (pop[9]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] + (2.0/3.0)*pop[3] - (1.0/3.0)*pop[5] + (2.0/3.0)*pop[7] - (1.0/3.0)*pop[9] + (2.0/3.0)*pop[11]);
    m_yz_I = inv_rho_I * (pop[11]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[3] + (2.0/3.0)*pop[5] - (1.0/3.0)*pop[7] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[11]);

    rho = (24*(rho_I + m_xx_I*rho_I - 2*m_xy_I*rho_I - 2*m_xz_I*rho_I + m_yy_I*rho_I - 2*m_yz_I*rho_I + m_zz_I*rho_I - omegaVar*m_xx_I*rho_I + 2*omegaVar*m_xy_I*rho_I + 2*omegaVar*m_xz_I*rho_I - omegaVar*m_yy_I*rho_I + 2*omegaVar*m_yz_I*rho_I - omegaVar*m_zz_I*rho_I))/(10*omegaVar - 12*ux0 - 12*uy0 - 12*uz0 + 4*omegaVar*ux0 + 4*omegaVar*uy0 + 4*omegaVar*uz0 - 9*omegaVar*ux0*ux0 - 9*omegaVar*uy0*uy0 - 9*omegaVar*uz0*uz0 + 6*omegaVar*ux0*uy0 + 6*omegaVar*ux0*uz0 + 6*omegaVar*uy0*uz0 + 4);

    m_xx_t45 = (2*(rho + 15*m_xx_I*rho_I - 9*m_xy_I*rho_I - 9*m_xz_I*rho_I - 3*m_yy_I*rho_I + 9*m_yz_I*rho_I - 3*m_zz_I*rho_I - 2*rho*ux0 + rho*uy0 + rho*uz0))/(9*rho);
    m_xy_t90 = -(2*rho + 9*m_xx_I*rho_I - 51*m_xy_I*rho_I + 3*m_xz_I*rho_I + 9*m_yy_I*rho_I + 3*m_yz_I*rho_I - 9*m_zz_I*rho_I + 2*rho*ux0 + 2*rho*uy0 + 2*rho*uz0)/(9*rho);
    m_xz_t90 = -(2*rho + 9*m_xx_I*rho_I + 3*m_xy_I*rho_I - 51*m_xz_I*rho_I - 9*m_yy_I*rho_I + 3*m_yz_I*rho_I + 9*m_zz_I*rho_I + 2*rho*ux0 + 2*rho*uy0 + 2*rho*uz0)/(9*rho);
    m_yy_t45 = (2*(rho - 3*m_xx_I*rho_I - 9*m_xy_I*rho_I + 9*m_xz_I*rho_I + 15*m_yy_I*rho_I - 9*m_yz_I*rho_I - 3*m_zz_I*rho_I + rho*ux0 - 2*rho*uy0 + rho*uz0))/(9*rho);
    m_yz_t90 = -(2*rho - 9*m_xx_I*rho_I + 3*m_xy_I*rho_I + 3*m_xz_I*rho_I + 9*m_yy_I*rho_I - 51*m_yz_I*rho_I + 9*m_zz_I*rho_I + 2*rho*ux0 + 2*rho*uy0 + 2*rho*uz0)/(9*rho);
    m_zz_t45 = (2*(rho - 3*m_xx_I*rho_I + 9*m_xy_I*rho_I - 9*m_xz_I*rho_I - 3*m_yy_I*rho_I - 9*m_yz_I*rho_I + 15*m_zz_I*rho_I + rho*ux0 + rho*uy0 - 2*rho*uz0))/(9*rho);

    rhoVar = rho;

    break;
}