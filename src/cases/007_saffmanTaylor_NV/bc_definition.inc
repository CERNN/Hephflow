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
case SOUTH_BACK:
    ux_t30 = 0.0_df;
    uy_t30 = 0.0_df;
    uz_t30 = 0.0_df;

    rho_I = pop[0] + pop[1] + pop[2] + pop[4] + pop[6] + pop[8] + pop[10] + pop[12] + pop[13] + pop[15];
    inv_rho_I = 1.0_df / rho_I;
    m_yz_I = inv_rho_I * pop[12];
    rho = (36.0_df*rho_I*(omegaVar*m_yz_I - m_yz_I + 1.0_df))/(omegaVar + 24.0_df);

    m_xx_t45 = 0.0_df;
    m_xy_t90 = 0.0_df;
    m_xz_t90 = 0.0_df;
    m_yy_t45 = 0.0_df;
    m_yz_t90 = (25.0_df*m_yz_I - 1.0_df)/(9.0_df*(omegaVar*m_yz_I - m_yz_I + 1.0_df));
    m_zz_t45 = 0.0_df;
    rhoVar = rho;
    break;
case NORTH_BACK:
    ux_t30 = 0.0_df;
    uy_t30 = 0.0_df;
    uz_t30 = 0.0_df;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[6] + pop[7] + pop[10] + pop[14] + pop[15] + pop[17];
    inv_rho_I = 1.0_df / rho_I;
    m_yz_I = -inv_rho_I * pop[17];
    rho = (36.0_df*rho_I*(m_yz_I - omegaVar*m_yz_I + 1.0_df))/(omegaVar + 24.0_df);

    m_xx_t45 = 0.0_df;
    m_xy_t90 = 0.0_df;
    m_xz_t90 = 0.0_df;
    m_yy_t45 = 0.0_df;
    m_yz_t90 = (25.0_df*m_yz_I + 1.0_df)/(9.0_df*(m_yz_I - omegaVar*m_yz_I + 1.0_df));
    m_zz_t45 = 0.0_df;
    rhoVar = rho;
    break;
case SOUTH_FRONT:
    ux_t30 = 0.0_df;
    uy_t30 = 0.0_df;
    uz_t30 = 0.0_df;

    rho_I = pop[0] + pop[1] + pop[2] + pop[4] + pop[5] + pop[8] + pop[9] + pop[13] + pop[16] + pop[18];
    inv_rho_I = 1.0_df / rho_I;
    m_yz_I = -inv_rho_I * pop[18];
    rho = (36.0_df*rho_I*(m_yz_I - omegaVar*m_yz_I + 1.0_df))/(omegaVar + 24.0_df);

    m_xx_t45 = 0.0_df;
    m_xy_t90 = 0.0_df;
    m_xz_t90 = 0.0_df;
    m_yy_t45 = 0.0_df;
    m_yz_t90 = (25.0_df*m_yz_I + 1.0_df)/(9.0_df*(m_yz_I - omegaVar*m_yz_I + 1.0_df));
    m_zz_t45 = 0.0_df;
    rhoVar = rho;
    break;
case NORTH_FRONT:
    ux_t30 = 0.0_df;
    uy_t30 = 0.0_df;
    uz_t30 = 0.0_df;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[5] + pop[7] + pop[9] + pop[11] + pop[14] + pop[16];
    inv_rho_I = 1.0_df / rho_I;
    m_yz_I = inv_rho_I * pop[11];
    rho = (36.0_df*rho_I*(omegaVar*m_yz_I - m_yz_I + 1.0_df))/(omegaVar + 24.0_df);

    m_xx_t45 = 0.0_df;
    m_xy_t90 = 0.0_df;
    m_xz_t90 = 0.0_df;
    m_yy_t45 = 0.0_df;
    m_yz_t90 = (25.0_df*m_yz_I - 1.0_df)/(9.0_df*(omegaVar*m_yz_I - m_yz_I + 1.0_df));
    m_zz_t45 = 0.0_df;
    rhoVar = rho;
    break;
case BACK: {

    // Velocity inlet at z = 0 (BACK face).
    // Planar Poiseuille profile between the plates, with U_IN the bulk/mean velocity:
    //     u_z(y) = 6 U_IN (y/H) [1 - y/H],   H = NY - 1.
    // The profile is zero at y = 0 and y = NY-1, matching the no-slip walls.
    ux_t30 = 0.0_df;
    uy_t30 = 0.0_df;
    uz_t30 = 6.0_df * U_IN * (((dfloat)y) / H) * (1.0_df - ((dfloat)y) / H);

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[6] + pop[7] + pop[8] + pop[10] + pop[12] + pop[13] + pop[14] + pop[15] + pop[17];
    inv_rho_I = 1.0_df / rho_I;
    m_xz_I = inv_rho_I * (pop[10] - pop[15]);
    m_yz_I = inv_rho_I * (pop[12] - pop[17]);

    // Exact mass-momentum closure for a prescribed BACK velocity:
    // rho (1-u_z) = sum(c_z=0) f_i + 2 sum(c_z<0) f_i.
    const dfloat f_zm = pop[6] + pop[10] + pop[12] + pop[15] + pop[17];
    rho = (rho_I + f_zm) / (1.0_df - uz_t30);

    m_xx_t45 = 0.0_df;
    m_xy_t90 = 0.0_df;
    m_xz_t90 = -m_xz_I*uz_t30*uz_t30 + m_xz_I*uz_t30 + (5.0_df/3.0_df)*m_xz_I;
    m_yy_t45 = 0.0_df;
    m_yz_t90 = -m_yz_I*uz_t30*uz_t30 + m_yz_I*uz_t30 + (5.0_df/3.0_df)*m_yz_I;
    m_zz_t45 = uz_t30*uz_t30;

    rhoVar = rho;

    break;
}
case FRONT: {

    // Outlet at z = NZ_TOTAL-1 (FRONT face): pressure condition.
    // Fully-developed channel outflow has no transverse mean velocity. Copying
    // these components allowed outlet-local transverse oscillations to grow.
    ux_t30 = 0.0_df;
    uy_t30 = 0.0_df;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[7] + pop[8] + pop[9] + pop[11] + pop[13] + pop[14] + pop[16] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-1.0/3.0*pop[0] + 2.0/3.0*pop[1] + 2.0/3.0*pop[2] - 1.0/3.0*pop[3] - 1.0/3.0*pop[4] - 1.0/3.0*pop[5] + 2.0/3.0*pop[7] + 2.0/3.0*pop[8] + 2.0/3.0*pop[9] - 1.0/3.0*pop[11] + 2.0/3.0*pop[13] + 2.0/3.0*pop[14] + 2.0/3.0*pop[16] - 1.0/3.0*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] - pop[16]);
    m_yz_I = inv_rho_I * (pop[11] - pop[18]);

    // Pressure outlet: anchor the reference density, then obtain u_z from the
    // exact mass-momentum identity. This avoids prescribing both rho and u_z
    // and prevents secular mean-density drift in the open domain.
    rho = RHO_0;
    const dfloat f_zp = pop[5] + pop[9] + pop[11] + pop[16] + pop[18];
    const dfloat uz_out = -1.0_df + (rho_I + f_zp) / rho;
    // Reversed normal flow would turn phi/A_ij into uncontrolled inflow.
    uz_t30 = myMax(uz_out, 0.0_df);

    m_xx_t45 = -2.0/3.0*m_xx_I*uz_t30*uz_t30 - 2.0/3.0*m_xx_I*uz_t30 + (10.0/9.0)*m_xx_I - 1.0/9.0*uy_t30*uy_t30 - 1.0/9.0*ux_t30*ux_t30;
    m_xy_t90 = -1.0/2.0*m_xy_I*uz_t30*uz_t30 - 1.0/2.0*m_xy_I*uz_t30 + (5.0/6.0)*m_xy_I;
    m_xz_t90 = -m_xz_I*uz_t30*uz_t30 - m_xz_I*uz_t30 + (5.0/3.0)*m_xz_I - 1.0/3.0*ux_t30;
    m_yy_t45 = (2.0/3.0)*m_xx_I*uz_t30*uz_t30 + (2.0/3.0)*m_xx_I*uz_t30 - 10.0/9.0*m_xx_I + (10.0/9.0)*uy_t30*uy_t30 + (10.0/9.0)*ux_t30*ux_t30;
    m_yz_t90 = -m_yz_I*uz_t30*uz_t30 - m_yz_I*uz_t30 + (5.0/3.0)*m_yz_I - 1.0/3.0*uy_t30;
    m_zz_t45 = uz_t30*uz_t30;

    rhoVar = rho;

    break;
}
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

    break;
}
