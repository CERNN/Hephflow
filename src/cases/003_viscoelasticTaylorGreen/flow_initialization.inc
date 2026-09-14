rho = RHO_0;
ux = U_0_X;
uy = U_0_Y;
uz = U_0_Z;


//on malaspinas paper the theta is already accounted for.
//dfloat thetaa = M_PI/2;

// ux =  U_MAX*(2.0/sqrtf(3.0))*sin(thetaa + 2.0*M_PI/3.0)*sin(xx)*cos(yy)*cos(zz);
// uy =  U_MAX*(2.0/sqrtf(3.0))*sin(thetaa - 2.0*M_PI/3.0)*cos(xx)*sin(yy)*cos(zz);
// uz =  U_MAX*(2.0/sqrtf(3.0))*sin(thetaa)*cos(xx)*cos(yy)*sin(zz);

//TODO however when deriving the poisson equation it does not include the theta term
// the RHO_0 + (U_MAX*U_MAX/cs2) comes from converting pressure to densisty. U_max comes from non-dimensional units
//rho = RHO_0 + (U_MAX*U_MAX/cs2)*(1.0/6.0)*(0.25*cos(2*yy) + 0.25*cos(-2*yy+2*xx) + 0.25*cos(2*xx+2*yy)
//                                         + cos(2*zz) + (1.0/24)*cos(-2*zz+2*yy) + (1.0/24)*cos( 2*zz+2*yy)
//                                                     + (1.0/24)*cos( 2*zz+2*xx) + (1.0/24)*cos(-2*zz+2*xx) + 0.25*cos(2*xx) - 1);


rho = RHO_0 + (1.0/(16.0*cs2))*RHO_0*U_MAX*U_MAX*(cos(2*(x) / L) + cos(2*(y) / L))*(cos(2*(z) / L) + 2.0);
ux =   U_MAX * sin((x) / L) * cos((y) / L) * cos((z) / L);
uy = - U_MAX * cos((x) / L) * sin((y) / L) * cos((z) / L);
uz = 0.0;

#ifdef A_XX_DIST
AxxVar = CONF_ZERO + 1.0;
Axx_qx_t30 = 3.0*(ux + 0.0);
Axx_qy_t30 = 3.0*(uy + 0.0);
Axx_qz_t30 = 3.0*(uz + 0.0);
#endif
#ifdef A_XY_DIST
AxyVar = CONF_ZERO;
Axy_qx_t30 = 3.0*(ux + 0.0);
Axy_qy_t30 = 3.0*(uy + 0.0);
Axy_qz_t30 = 3.0*(uz + 0.0);
#endif
#ifdef A_XZ_DIST
AxzVar = CONF_ZERO;
Axz_qx_t30 = 3.0*(ux + 0.0);
Axz_qy_t30 = 3.0*(uy + 0.0);
Axz_qz_t30 = 3.0*(uz + 0.0);
#endif
#ifdef A_YY_DIST
AyyVar = CONF_ZERO + 1.0;
Ayy_qx_t30 = 3.0*(ux + 0.0);
Ayy_qy_t30 = 3.0*(uy + 0.0);
Ayy_qz_t30 = 3.0*(uz + 0.0);
#endif
#ifdef A_YZ_DIST
AyzVar = CONF_ZERO;
Ayz_qx_t30 = 3.0*(ux + 0.0);
Ayz_qy_t30 = 3.0*(uy + 0.0);
Ayz_qz_t30 = 3.0*(uz + 0.0);
#endif
#ifdef A_ZZ_DIST
AzzVar = CONF_ZERO + 1.0;
Azz_qx_t30 = 3.0*(ux + 0.0);
Azz_qy_t30 = 3.0*(uy + 0.0);
Azz_qz_t30 = 3.0*(uz + 0.0);
#endif


#ifdef NON_NEWTONIAN_FLUID
omega = OMEGA;
#endif
