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
case BULK:
    break;
case SOUTH:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[4] + pop[5] + pop[6] + pop[8] + pop[9] + pop[10] + pop[12] + pop[13] + pop[15] + pop[16] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xy_I = inv_rho_I * (pop[8] - pop[13]);
    m_yz_I = inv_rho_I * (pop[12] - pop[18]);

    rho = (6*rho_I)/5;
    

    m_xx_t45 = 0;
    m_xy_t90 = (5*m_xy_I)/3;
    m_xz_t90 = 0;
    m_yy_t45 = 0;
    m_yz_t90 = (5*m_yz_I)/3;
    m_zz_t45 = 0;

    rhoVar = rho;

    break;
case NORTH:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[5] + pop[6] + pop[7] + pop[9] + pop[10] + pop[11] + pop[14] + pop[15] + pop[16] + pop[17];
    inv_rho_I = 1.0 / rho_I;
    m_xy_I = inv_rho_I * (pop[7] - pop[14]);
    m_yz_I = inv_rho_I * (pop[11] - pop[17]);

    rho = (6*rho_I)/5;
    

    m_xx_t45 = 0;
    m_xy_t90 = (5*m_xy_I)/3;
    m_xz_t90 = 0;
    m_yy_t45 = 0;
    m_yz_t90 = (5*m_yz_I)/3;
    m_zz_t45 = 0;

    rhoVar = rho;
    case WEST + BC_VELOCITY_WALL:
        ux_t30 = U_MAX;
        uy_t30 = 0;
        uz_t30 = 0;

        rho_I = pop[0] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[8] + pop[10] + pop[11] + pop[12] + pop[14] + pop[16] + pop[17] + pop[18];
        inv_rho_I = 1.0 / rho_I;
        m_xy_I = inv_rho_I * (pop[8] - pop[14]);
        m_xz_I = inv_rho_I * (pop[10] - pop[16]);

        rho = -6.0_df * rho_I / (-5.0_df + 3.0_df * U_MAX + 3.0_df * U_MAX * U_MAX);


        m_xx_t45 = ux_t30 * ux_t30;
        m_xy_t90 = (2*m_xy_I*rho_I / rho);
        m_xz_t90 = (2*m_xz_I*rho_I / rho);
        m_yy_t45 = 0;
        m_yz_t90 = 0;
        m_zz_t45 = 0;

        rhoVar = rho;
    case EAST + BC_OUTFLOW: //copy values from last row - 1
        rho_I = pop[0] + pop[1] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[9] + pop[11] + pop[12] + pop[13] + pop[15] + pop[17] + pop[18];
        inv_rho_I = 1.0 / rho_I;
        m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[9] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[15] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
        m_xy_I = inv_rho_I * (pop[7] - pop[13]);
        m_xz_I = inv_rho_I * (pop[9] - pop[15]);
        m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] - (1.0/3.0)*pop[9] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] - (1.0/3.0)*pop[15] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
        m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
        m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);


        rhoVar = RHO_0 + fMom[idxMom(threadIdx.x-1, threadIdx.y, threadIdx.z, M_RHO_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
        ux_t30     =     fMom[idxMom(threadIdx.x-1, threadIdx.y, threadIdx.z, M_UX_INDEX , blockIdx.x, blockIdx.y, blockIdx.z)]/F_M_I_SCALE;
        uy_t30     =     fMom[idxMom(threadIdx.x-1, threadIdx.y, threadIdx.z, M_UY_INDEX , blockIdx.x, blockIdx.y, blockIdx.z)]/F_M_I_SCALE;
        uz_t30     =     fMom[idxMom(threadIdx.x-1, threadIdx.y, threadIdx.z, M_UZ_INDEX , blockIdx.x, blockIdx.y, blockIdx.z)]/F_M_I_SCALE;
        m_xx_t45   = ux_t30 * ux_t30; 
        m_xy_t90   = (-2.0_df * m_xy_I * rho_I) / rhoVar + uy_t30 * ONE_THIRD;    
        m_xz_t90   = (-2.0_df * m_xz_I * rho_I) / rhoVar + uz_t30 * ONE_THIRD;
        m_yy_t45   = -(4.0_df * rho_I * (-m_yy_I + m_zz_I) - 3.0_df * rhoVar * (-uy_t30*uy_t30 - uz_t30*uz_t30)) / (6.0_df * rhoVar);
        m_yz_t90   = rho_I * m_yz_I / rhoVar;    
        m_zz_t45   = -(4.0_df * rho_I * (-m_zz_I + m_yy_I) - 3.0_df * rhoVar * (-uy_t30*uy_t30 - uz_t30*uz_t30)) / (6.0_df * rhoVar);
        break;
    break;
}