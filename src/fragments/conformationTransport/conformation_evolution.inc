
//remove offset
AxxVar -= CONF_ZERO;
AxyVar -= CONF_ZERO;
AxzVar -= CONF_ZERO;
AyyVar -= CONF_ZERO;
AyzVar -= CONF_ZERO;
AzzVar -= CONF_ZERO;

dfloat switch_conf = 0.0_df;
if (step>CONFORMATION_SWITCH){
    switch_conf = 1.0_df;     
}

dfloat h_phi = 1.0_df;
#ifdef PHI_DIST
    // h(phi) = 0.5*(1+phi), strictly bounded to [0,1] to avoid overshoots.
    dfloat phi_local = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M3_PHI_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
    h_phi = fmax(0.0_df, fmin(1.0_df, 0.5_df * (1.0_df + phi_local)));
    const veFluidProps veLocal = blendVeProps(phasePropsA.ve, phasePropsB.ve, h_phi);
#else
    const veFluidProps& veLocal = phasePropsA.ve;
#endif

dfloat eta_p_local      = veLocal.eta_p;
dfloat inv_lambda_local = 1.0_df / veLocal.lambda;

// (A . div U + div U^T . A) * 3.0_df
/*
dfloat ADU_DUTA_xx_t30 = 2*AxxVar*duxdx_t30 + 2*AxyVar*duxdy_t30 + 2*0*0;
dfloat ADU_DUTA_yy_t30 = 2*AxyVar*duydx_t30 + 2*AyyVar*duydy_t30 + 2*0*0;
dfloat ADU_DUTA_zz_t30 = 2*0*0 + 2*0*0 + 2*0*0;

dfloat ADU_DUTA_xy_t30 = AxxVar*duydx_t30 + AxyVar*duxdx_t30 + AxyVar*duydy_t30 + 0*0 + AyyVar*duxdy_t30 + 0*0;
dfloat ADU_DUTA_xz_t30 = AxxVar*0 + 0*duxdx_t30 + AxyVar*0 + 0*0 + 0*duxdy_t30 + 0*0;
dfloat ADU_DUTA_yz_t30 = AxyVar*0 + 0*duydx_t30 + AyyVar*0 + 0*duydy_t30 + 0*0 + 0*0;
*/

dfloat ADU_DUTA_xx_t30 = 2*AxxVar*duxdx_t30 + 2*AxyVar*duxdy_t30 + 2*AxzVar*duxdz_t30;
dfloat ADU_DUTA_yy_t30 = 2*AxyVar*duydx_t30 + 2*AyyVar*duydy_t30 + 2*AyzVar*duydz_t30;
dfloat ADU_DUTA_zz_t30 = 2*AxzVar*duzdx_t30 + 2*AyzVar*duzdy_t30 + 2*AzzVar*duzdz_t30;

dfloat ADU_DUTA_xy_t30 = AxxVar*duydx_t30 + AxyVar*duxdx_t30 + AxyVar*duydy_t30 + AxzVar*duydz_t30 + AyyVar*duxdy_t30 + AyzVar*duxdz_t30;
dfloat ADU_DUTA_xz_t30 = AxxVar*duzdx_t30 + AxzVar*duxdx_t30 + AxyVar*duzdy_t30 + AxzVar*duzdz_t30 + AyzVar*duxdy_t30 + AzzVar*duxdz_t30;
dfloat ADU_DUTA_yz_t30 = AxyVar*duzdx_t30 + AxzVar*duydx_t30 + AyyVar*duzdy_t30 + AyzVar*duydy_t30 + AyzVar*duzdz_t30 + AzzVar*duydz_t30;

// Compute model-specific relaxation term via struct-based dispatch (nnf.h)
const veRelaxTerm veR = calcVeRelaxationTerm(
    veLocal, inv_lambda_local,
    AxxVar, AxyVar, AxzVar, AyyVar, AyzVar, AzzVar);

//compute source term for the conformation transport
// G_ij = R_ij(model) + (A.du + du^T.A)_ij / F_M_I_SCALE
dfloat Gxx = veR.xx + ADU_DUTA_xx_t30/F_M_I_SCALE;
dfloat Gyy = veR.yy + ADU_DUTA_yy_t30/F_M_I_SCALE;
dfloat Gzz = veR.zz + ADU_DUTA_zz_t30/F_M_I_SCALE;

dfloat Gxy = veR.xy + ADU_DUTA_xy_t30/F_M_I_SCALE;
dfloat Gxz = veR.xz + ADU_DUTA_xz_t30/F_M_I_SCALE;
dfloat Gyz = veR.yz + ADU_DUTA_yz_t30/F_M_I_SCALE;

//compute the soource term for mass-momentum transport, eq36 malaspinas
// rho.g = div * PI =  (mu/lambda)*(a*A - b*I)
//F = (mu/lambda) * div(A) - div(I); but div(I) = 0
//F = (mu/lambda)*div(A)

// Polymeric coupling is explicitly multiplied by eta_p_local. In the Newtonian phase
// h(phi)=0 => eta_p_local=0, so the regularized inv_lambda_local introduces no stress.
L_Fx += (dAxxdx + dAxydy + dAxzdz)*(eta_p_local*inv_lambda_local)*switch_conf;
L_Fy += (dAxydx + dAyydy + dAyzdz)*(eta_p_local*inv_lambda_local)*switch_conf;
L_Fz += (dAxzdx + dAyzdy + dAzzdz)*(eta_p_local*inv_lambda_local)*switch_conf;


AxxVar += CONF_ZERO;
AxyVar += CONF_ZERO;
AxzVar += CONF_ZERO;
AyyVar += CONF_ZERO;
AyzVar += CONF_ZERO;
AzzVar += CONF_ZERO;



