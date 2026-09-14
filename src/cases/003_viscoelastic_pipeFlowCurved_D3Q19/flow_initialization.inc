
/*
// Example of usage of random numbers for turbulence in parallel plates flow in z  
    dfloat y_visc = 6.59, ub_f = 15.6, uc_f = 18.2;
    // logaritimic velocity profile
    dfloat uz_log; 
    dfloat pos = (y < NY/2 ? y + 0.5 : NY - (y + 0.5));
    uz_log = -(uc_f*U_TAU)*(((pos-NY/2)/del)*((pos-NY/2)/del)) + (uc_f*U_TAU);
    
    uz = uz_log;
    ux = 0.0;
    uy = 0.0;
    rho = RHO_0;


    // perturbation
    dfloat pert = 0.1;
    int l = idxScalarGlobal(x, y, z);
    int Nt = NUMBER_LBM_NODES;
    uz += (ub_f*U_TAU)*pert*randomNumbers[l + z - Nt*((l + z) / Nt)];
    ux += (ub_f*U_TAU)*pert*randomNumbers[l + x - Nt*((l + x) / Nt)];
    uy += (ub_f*U_TAU)*pert*randomNumbers[l + y - Nt*((l + y) / Nt)];
*/ 


/*dfloat pert = 0.05;
int l = idxScalarGlobal(x, y, z);
int Nt = NUMBER_LBM_NODES;

ux += (U_MAX)*pert*randomNumbers[l + x - Nt*((l + x) / Nt)];
uy += (U_MAX)*pert*randomNumbers[l + y - Nt*((l + y) / Nt)];
uz += (U_MAX)*pert*randomNumbers[l + z - Nt*((l + z) / Nt)];*/

rho = RHO_0;
ux = U_0_X;
uy = U_0_Y;
uz = U_0_Z;

#ifdef ANALYTICAL_STEADY_INITIALIZATION
const dfloat initialDx = (dfloat)x - DUCT_CENTER_X;
const dfloat initialDy = (dfloat)y - DUCT_CENTER_Y;
const dfloat initialRadiusSq = initialDx * initialDx + initialDy * initialDy;
const bool initialFluidNode = initialRadiusSq < OUTER_RADIUS * OUTER_RADIUS;
if (initialFluidNode) {
    uz = U_C * (1.0_df - initialRadiusSq / (OUTER_RADIUS * OUTER_RADIUS));
}
const dfloat initialDuzDx = initialFluidNode
    ? -2.0_df * U_C * initialDx / (OUTER_RADIUS * OUTER_RADIUS) : 0.0_df;
const dfloat initialDuzDy = initialFluidNode
    ? -2.0_df * U_C * initialDy / (OUTER_RADIUS * OUTER_RADIUS) : 0.0_df;
#endif

#ifdef OMEGA_FIELD
omega = OMEGA;
#endif

#ifdef A_XX_DIST
AxxVar = CONF_ZERO + 1.0_df;
Axx_qx_t30 = F_M_I_SCALE * AxxVar * ux;
Axx_qy_t30 = F_M_I_SCALE * AxxVar * uy;
Axx_qz_t30 = F_M_I_SCALE * AxxVar * uz;
#endif
#ifdef A_XY_DIST
AxyVar = CONF_ZERO;
Axy_qx_t30 = F_M_I_SCALE * AxyVar * ux;
Axy_qy_t30 = F_M_I_SCALE * AxyVar * uy;
Axy_qz_t30 = F_M_I_SCALE * AxyVar * uz;
#endif
#ifdef A_XZ_DIST
#ifdef ANALYTICAL_STEADY_INITIALIZATION
AxzVar = CONF_ZERO + LAMBDA * initialDuzDx;
#else
AxzVar = CONF_ZERO;
#endif
Axz_qx_t30 = F_M_I_SCALE * AxzVar * ux;
Axz_qy_t30 = F_M_I_SCALE * AxzVar * uy;
Axz_qz_t30 = F_M_I_SCALE * AxzVar * uz;
#endif
#ifdef A_YY_DIST
AyyVar = CONF_ZERO + 1.0_df;
Ayy_qx_t30 = F_M_I_SCALE * AyyVar * ux;
Ayy_qy_t30 = F_M_I_SCALE * AyyVar * uy;
Ayy_qz_t30 = F_M_I_SCALE * AyyVar * uz;
#endif
#ifdef A_YZ_DIST
#ifdef ANALYTICAL_STEADY_INITIALIZATION
AyzVar = CONF_ZERO + LAMBDA * initialDuzDy;
#else
AyzVar = CONF_ZERO;
#endif
Ayz_qx_t30 = F_M_I_SCALE * AyzVar * ux;
Ayz_qy_t30 = F_M_I_SCALE * AyzVar * uy;
Ayz_qz_t30 = F_M_I_SCALE * AyzVar * uz;
#endif
#ifdef A_ZZ_DIST
#ifdef ANALYTICAL_STEADY_INITIALIZATION
AzzVar = CONF_ZERO + 1.0_df + 2.0_df * LAMBDA * LAMBDA
       * (initialDuzDx * initialDuzDx + initialDuzDy * initialDuzDy);
#else
AzzVar = CONF_ZERO + 1.0_df;
#endif
Azz_qx_t30 = F_M_I_SCALE * AzzVar * ux;
Azz_qy_t30 = F_M_I_SCALE * AzzVar * uy;
Azz_qz_t30 = F_M_I_SCALE * AzzVar * uz;
#endif
