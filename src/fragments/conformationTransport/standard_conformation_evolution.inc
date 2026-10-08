
// Standard conformation representation.
//
// The transported tensor is A and its upper-convected evolution is
//
//   DA/Dt = L A + A L^T + R(A).
//
// Momentum coupling uses Fp = (eta_p/lambda) div(S(A)). The stored fields
// include the legacy CONF_ZERO offset, which is removed before evaluating the
// physical equation and restored before collision.
// Remove the storage offset before evaluating the physical equation.
AxxVar -= CONF_ZERO;
AxyVar -= CONF_ZERO;
AxzVar -= CONF_ZERO;
AyyVar -= CONF_ZERO;
AyzVar -= CONF_ZERO;
AzzVar -= CONF_ZERO;

// Local properties and activation. Multiphase cases blend constitutive properties and attenuate the source in mixed
// cells so a Newtonian phase contributes neither relaxation nor polymer stress.
dfloat switch_conf = 0.0_df;
if (step > CONFORMATION_SWITCH) {
    switch_conf = 1.0_df;
}

#ifndef PHI_DIST

dfloat h_phi = 1.0_df;
#endif // PHI_DIST

#ifdef PHI_DIST
    const veFluidProps veLocal = blendVeProps(phasePropsA.ve, phasePropsB.ve, h_phi);
#else
    const veFluidProps& veLocal = phasePropsA.ve;
#endif // PHI_DIST

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
#endif // PHI_DIST

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

// Remove the phase-field contribution before forming L A + A L^T.
#ifdef PHI_DIST
    duxdx_eff -= phase_du_xx;
    duxdy_eff -= phase_du_xy;
    duxdz_eff -= phase_du_xz;
    duydx_eff -= phase_du_yx;
    duydy_eff -= phase_du_yy;
    duydz_eff -= phase_du_yz;
    duzdx_eff -= phase_du_zx;
    duzdy_eff -= phase_du_zy;
    duzdz_eff -= phase_du_zz;
#endif // PHI_DIST

// Upper-convected stretching, L A + A L^T. These six expressions follow Sym3 component order and retain their
// established floating-point evaluation order.
dfloat ADU_DUTA_xx_t30 = 2*AxxVar*duxdx_eff + 2*AxyVar*duxdy_eff + 2*AxzVar*duxdz_eff;
dfloat ADU_DUTA_yy_t30 = 2*AxyVar*duydx_eff + 2*AyyVar*duydy_eff + 2*AyzVar*duydz_eff;
dfloat ADU_DUTA_zz_t30 = 2*AxzVar*duzdx_eff + 2*AyzVar*duzdy_eff + 2*AzzVar*duzdz_eff;

dfloat ADU_DUTA_xy_t30 = AxxVar*duydx_eff + AxyVar*duxdx_eff + AxyVar*duydy_eff + AxzVar*duydz_eff + AyyVar*duxdy_eff + AyzVar*duxdz_eff;
dfloat ADU_DUTA_xz_t30 = AxxVar*duzdx_eff + AxzVar*duxdx_eff + AxyVar*duzdy_eff + AxzVar*duzdz_eff + AyzVar*duxdy_eff + AzzVar*duxdz_eff;
dfloat ADU_DUTA_yz_t30 = AxyVar*duzdx_eff + AxzVar*duydx_eff + AyyVar*duzdy_eff + AyzVar*duydy_eff + AyzVar*duzdz_eff + AzzVar*duydz_eff;

const Sym3 veA(AxxVar, AxyVar, AxzVar, AyyVar, AyzVar, AzzVar);
const Sym3 veR = calcVeRelaxation(veLocal, veA);

// Add the model-specific relaxation R(A), scale the lattice velocity gradient, and apply local activation.
// G_ij = R_ij(model) + (A.du + du^T.A)_ij / F_M_I_SCALE
dfloat Gxx = confSourceWeight * (veR.xx + ADU_DUTA_xx_t30/F_M_I_SCALE);
dfloat Gyy = confSourceWeight * (veR.yy + ADU_DUTA_yy_t30/F_M_I_SCALE);
dfloat Gzz = confSourceWeight * (veR.zz + ADU_DUTA_zz_t30/F_M_I_SCALE);

dfloat Gxy = confSourceWeight * (veR.xy + ADU_DUTA_xy_t30/F_M_I_SCALE);
dfloat Gxz = confSourceWeight * (veR.xz + ADU_DUTA_xz_t30/F_M_I_SCALE);
dfloat Gyz = confSourceWeight * (veR.yz + ADU_DUTA_yz_t30/F_M_I_SCALE);

// Force coupling: F_p = (eta_p/lambda) div(S(A)), corresponding to the polymer contribution in the momentum source.
// For Oldroyd-B, S=A-I and div(I)=0, so differentiating A is exactly equivalent.

// Polymeric coupling is explicitly multiplied by eta_p_local. In the Newtonian phase
// h(phi)=0 => eta_p_local=0, so the regularized inv_lambda_local introduces no stress.
#if !defined(DISABLE_POLYMER_MOMENTUM_FEEDBACK) && \
    !defined(CONFORMATION_DIRECT_STRESS_HALO)
    dfloat polymerFeedbackRamp = 1.0_df;
    #ifdef VISCOELASTIC_STARTUP_RAMP
    {
        const dfloat rampCoordinate = fmin(
            1.0_df, (dfloat)step / (dfloat)STARTUP_RAMP_STEPS);
        polymerFeedbackRamp = rampCoordinate * rampCoordinate
                            * (3.0_df - 2.0_df * rampCoordinate);
    }
    #endif // VISCOELASTIC_STARTUP_RAMP
    const Sym3 dSdx = calcVeStressReducedDifferential(
        veLocal, veA, Sym3(dAxxdx, dAxydx, dAxzdx,
                          dAyydx, dAyzdx, dAzzdx));
    const Sym3 dSdy = calcVeStressReducedDifferential(
        veLocal, veA, Sym3(dAxxdy, dAxydy, dAxzdy,
                          dAyydy, dAyzdy, dAzzdy));
    const Sym3 dSdz = calcVeStressReducedDifferential(
        veLocal, veA, Sym3(dAxxdz, dAxydz, dAxzdz,
                          dAyydz, dAyzdz, dAzzdz));

    dfloat polymerForceX = calcVePolymerModulus(veLocal) * (dSdx.xx + dSdy.xy + dSdz.xz);
    dfloat polymerForceY = calcVePolymerModulus(veLocal) * (dSdx.xy + dSdy.yy + dSdz.yz);
    dfloat polymerForceZ = calcVePolymerModulus(veLocal) * (dSdx.xz + dSdy.yz + dSdz.zz);

    #if defined(VE_FORCE_FULL_DIVERGENCE) && defined(PHI_DIST)
    {
        const dfloat phaseRange = PHI_TWO - PHI_ONE;
        const dfloat etaSlope = phasePropsB.ve.eta_p - phasePropsA.ve.eta_p;
        const dfloat lambdaSlope = phasePropsB.ve.lambda - phasePropsA.ve.lambda;
        const dfloat lambdaLocal = veLocal.lambda;

        dfloat dGpDh = 0.0_df;

        if (h_phi > 1.0e-4_df && h_phi < 1.0_df-1.0e-4_df &&
            veLocal.type != VE_NEWTONIAN && lambdaLocal > 0.0_df) {
            dGpDh = (etaSlope*lambdaLocal - eta_p_local*lambdaSlope) / (lambdaLocal*lambdaLocal);
        }

        const dfloat dGpDx = dGpDh*phase_dphidx/phaseRange;
        const dfloat dGpDy = dGpDh*phase_dphidy/phaseRange;
        const dfloat dGpDz = dGpDh*phase_dphidz/phaseRange;

        const Sym3 stressReduced = calcVeStressReduced(veLocal, veA);

        polymerForceX += dGpDx*stressReduced.xx + dGpDy*stressReduced.xy + dGpDz*stressReduced.xz;
        polymerForceY += dGpDx*stressReduced.xy + dGpDy*stressReduced.yy + dGpDz*stressReduced.yz;
        polymerForceZ += dGpDx*stressReduced.xz + dGpDy*stressReduced.yz + dGpDz*stressReduced.zz;
    }
    #endif // VE_FORCE_FULL_DIVERGENCE && PHI_DIST

    const dfloat feedbackWeight = switch_conf*polymerFeedbackRamp;

    L_Fx += polymerForceX*feedbackWeight;
    L_Fy += polymerForceY*feedbackWeight;
    L_Fz += polymerForceZ*feedbackWeight;

#endif // !DISABLE_POLYMER_MOMENTUM_FEEDBACK && !CONFORMATION_DIRECT_STRESS_HALO

// Restore the storage offset expected by collision/transport storage.
AxxVar += CONF_ZERO;
AxyVar += CONF_ZERO;
AxzVar += CONF_ZERO;
AyyVar += CONF_ZERO;
AyzVar += CONF_ZERO;
AzzVar += CONF_ZERO;
