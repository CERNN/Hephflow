
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

    dfloat phi_local = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M3_PHI_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
    h_phi = (phi_local - PHI_ONE) / (PHI_TWO - PHI_ONE);    h_phi = fmax(0.0_df, fmin(1.0_df, h_phi)); // normalize from [-1,1] -> [0,1]
    const veFluidProps veLocal = blendVeProps(phasePropsA.ve, phasePropsB.ve, h_phi);
#else
    const veFluidProps& veLocal = phasePropsA.ve;
#endif

dfloat vePhaseWeight = 1.0_df;
#ifdef PHI_DIST
    // In NV mixtures, attenuate conformation source terms in interface/mixed cells
    // according to local viscoelastic phase fraction to reduce amplitude bias.
    if (phasePropsA.ve.type == VE_NEWTONIAN && phasePropsB.ve.type != VE_NEWTONIAN) {
        vePhaseWeight = h_phi;
    } else if (phasePropsB.ve.type == VE_NEWTONIAN && phasePropsA.ve.type != VE_NEWTONIAN) {
        vePhaseWeight = 1.0_df - h_phi;
    }
    vePhaseWeight = fmax(0.0_df, fmin(1.0_df, vePhaseWeight));
#endif

dfloat eta_p_local      = veLocal.eta_p;
const bool conformationActive = switch_conf > 0.0_df && veLocal.type != VE_NEWTONIAN && eta_p_local > 1.0e-14_df && veLocal.lambda > 0.0_df;
const dfloat confMask = conformationActive ? 1.0_df : 0.0_df;
const dfloat confSourceWeight = confMask * vePhaseWeight;
dfloat inv_lambda_local = conformationActive ? (1.0_df / veLocal.lambda) : 0.0_df;

dfloat duxdx_eff = duxdx_t30;
dfloat duxdy_eff = duxdy_t30;
dfloat duxdz_eff = duxdz_t30;
dfloat duydx_eff = duydx_t30;
dfloat duydy_eff = duydy_t30;
dfloat duydz_eff = duydz_t30;
dfloat duzdx_eff = duzdx_t30;
dfloat duzdy_eff = duzdy_t30;
dfloat duzdz_eff = duzdz_t30;

#ifdef PHI_DIST
    duxdx_eff -= phase_du_xx; duxdy_eff -= phase_du_xy; duxdz_eff -= phase_du_xz;
    duydx_eff -= phase_du_yx; duydy_eff -= phase_du_yy; duydz_eff -= phase_du_yz;
    duzdx_eff -= phase_du_zx; duzdy_eff -= phase_du_zy; duzdz_eff -= phase_du_zz;
#endif

// (A . div U + div U^T . A) * 3.0_df
/*
dfloat ADU_DUTA_xx_t30 = 2*AxxVar*duxdx_t30 + 2*AxyVar*duxdy_t30 + 2*0*0;
dfloat ADU_DUTA_yy_t30 = 2*AxyVar*duydx_t30 + 2*AyyVar*duydy_t30 + 2*0*0;
dfloat ADU_DUTA_zz_t30 = 2*0*0 + 2*0*0 + 2*0*0;

dfloat ADU_DUTA_xy_t30 = AxxVar*duydx_t30 + AxyVar*duxdx_t30 + AxyVar*duydy_t30 + 0*0 + AyyVar*duxdy_t30 + 0*0;
dfloat ADU_DUTA_xz_t30 = AxxVar*0 + 0*duxdx_t30 + AxyVar*0 + 0*0 + 0*duxdy_t30 + 0*0;
dfloat ADU_DUTA_yz_t30 = AxyVar*0 + 0*duydx_t30 + AyyVar*0 + 0*duydy_t30 + 0*0 + 0*0;
*/

dfloat ADU_DUTA_xx_t30 = 2*AxxVar*duxdx_eff + 2*AxyVar*duxdy_eff + 2*AxzVar*duxdz_eff;
dfloat ADU_DUTA_yy_t30 = 2*AxyVar*duydx_eff + 2*AyyVar*duydy_eff + 2*AyzVar*duydz_eff;
dfloat ADU_DUTA_zz_t30 = 2*AxzVar*duzdx_eff + 2*AyzVar*duzdy_eff + 2*AzzVar*duzdz_eff;

dfloat ADU_DUTA_xy_t30 = AxxVar*duydx_eff + AxyVar*duxdx_eff + AxyVar*duydy_eff + AxzVar*duydz_eff + AyyVar*duxdy_eff + AyzVar*duxdz_eff;
dfloat ADU_DUTA_xz_t30 = AxxVar*duzdx_eff + AxzVar*duxdx_eff + AxyVar*duzdy_eff + AxzVar*duzdz_eff + AyzVar*duxdy_eff + AzzVar*duxdz_eff;
dfloat ADU_DUTA_yz_t30 = AxyVar*duzdx_eff + AxzVar*duydx_eff + AyyVar*duzdy_eff + AyzVar*duydy_eff + AyzVar*duzdz_eff + AzzVar*duydz_eff;

// Compute model-specific relaxation term via struct-based dispatch (nnf.h)
const veRelaxTerm veR = calcVeRelaxationTerm(
    veLocal, inv_lambda_local,
    AxxVar, AxyVar, AxzVar, AyyVar, AyzVar, AzzVar);

//compute source term for the conformation transport
// G_ij = R_ij(model) + (A.du + du^T.A)_ij / F_M_I_SCALE
dfloat Gxx = confSourceWeight * (veR.xx + ADU_DUTA_xx_t30/F_M_I_SCALE);
dfloat Gyy = confSourceWeight * (veR.yy + ADU_DUTA_yy_t30/F_M_I_SCALE);
dfloat Gzz = confSourceWeight * (veR.zz + ADU_DUTA_zz_t30/F_M_I_SCALE);

dfloat Gxy = confSourceWeight * (veR.xy + ADU_DUTA_xy_t30/F_M_I_SCALE);
dfloat Gxz = confSourceWeight * (veR.xz + ADU_DUTA_xz_t30/F_M_I_SCALE);
dfloat Gyz = confSourceWeight * (veR.yz + ADU_DUTA_yz_t30/F_M_I_SCALE);

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



