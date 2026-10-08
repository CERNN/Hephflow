// Log-conformation representation. The transported field is Psi = log(A), governed by
//
//   D Psi/Dt = D log_A[L A + A L^T + R(A)],
//
// where D log_A is the Fréchet derivative evaluated in A's eigensystem. Its divided difference tends to 1/lambda
// for repeated eigenvalues, avoiding a singular quotient when two principal stretches coincide.

// Local properties
// The source is disabled during startup and in locally Newtonian cells. Multiphase cases blend properties and source
// amplitude with the same phase indicator used by the other conformation representations.
dfloat switch_conf = step > CONFORMATION_SWITCH ? 1.0_df : 0.0_df;
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
    if (phasePropsA.ve.type == VE_NEWTONIAN && phasePropsB.ve.type != VE_NEWTONIAN) {
        vePhaseWeight = h_phi;
    } else if (phasePropsB.ve.type == VE_NEWTONIAN && phasePropsA.ve.type != VE_NEWTONIAN) {
        vePhaseWeight = 1.0_df - h_phi;
    }
    vePhaseWeight = fmax(0.0_df, fmin(1.0_df, vePhaseWeight));
#endif // PHI_DIST

const bool conformationActive = switch_conf > 0.0_df && veLocal.type != VE_NEWTONIAN && veLocal.eta_p > 1.0e-14_df && veLocal.lambda > 0.0_df;
const dfloat confSourceWeight = conformationActive ? vePhaseWeight : 0.0_df;

dfloat duxdx_eff = duxdx_t30, duxdy_eff = duxdy_t30, duxdz_eff = duxdz_t30;
dfloat duydx_eff = duydx_t30, duydy_eff = duydy_t30, duydz_eff = duydz_t30;
dfloat duzdx_eff = duzdx_t30, duzdy_eff = duzdy_t30, duzdz_eff = duzdz_t30;

// Remove the phase-field contribution before evaluating the constitutive stretching term.
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

// Reconstruct physical A = exp(Psi). The direct-stress path already computed this eigensystem before the barrier.
const Sym3 vePsi(AxxVar, AxyVar, AxzVar, AyyVar, AyzVar, AzzVar);

#ifdef CONFORMATION_PHYSICAL_A_HALO
    const VeEigenSym3& vePsiEigen = veDirectPsiEigen;
    const Sym3& veA = veDirectPhysicalA;
#else
    VeEigenSym3 vePsiEigen;
    const Sym3 veA = veExpSymmetric(vePsi, &vePsiEigen);
#endif // CONFORMATION_PHYSICAL_A_HALO

// Physical-A source: materialA = L A + A L^T + R(A). The velocity scale converts the stored lattice gradient to the
// conformation transport scale. Constructor arguments follow Sym3 component order: xx, xy, xz, yy, yz, zz.
const Sym3 veR = calcVeRelaxation(veLocal, veA);
const dfloat vs = 1.0_df / F_M_I_SCALE;

const Sym3 materialA(
    vs*2.0_df*(duxdx_eff*veA.xx + duxdy_eff*veA.xy + duxdz_eff*veA.xz) + veR.xx,
    vs*(duxdx_eff*veA.xy + duxdy_eff*veA.yy + duxdz_eff*veA.yz
      + duydx_eff*veA.xx + duydy_eff*veA.xy + duydz_eff*veA.xz) + veR.xy,
    vs*(duxdx_eff*veA.xz + duxdy_eff*veA.yz + duxdz_eff*veA.zz
      + duzdx_eff*veA.xx + duzdy_eff*veA.xy + duzdz_eff*veA.xz) + veR.xz,
    vs*2.0_df*(duydx_eff*veA.xy + duydy_eff*veA.yy + duydz_eff*veA.yz) + veR.yy,
    vs*(duydx_eff*veA.xz + duydy_eff*veA.yz + duydz_eff*veA.zz
      + duzdx_eff*veA.xy + duzdy_eff*veA.yy + duzdz_eff*veA.yz) + veR.yz,
    vs*2.0_df*(duzdx_eff*veA.xz + duzdy_eff*veA.yz + duzdz_eff*veA.zz) + veR.zz);


// Apply D log_A to the physical-A material derivative. The scalar auxiliary
// LBM already contributes D*laplacian(Psi); replace it by
// D log_A[D*laplacian(A)] so the regularized physical equation matches the
// standard representation rather than diffusing a different stored variable.
const Sym3 veLogDiffusionCorrection =
    veFrechetLogFromPsi(vePsiEigen,veLaplacianPhysicalA)*CONF_DIFFUSIVITY
    -veLaplacianStored*CONF_DIFFUSIVITY;
const Sym3 sourcePsi = veFrechetLogFromPsi(vePsiEigen,materialA)*confSourceWeight
    + veLogDiffusionCorrection;

dfloat Gxx = sourcePsi.xx, Gxy = sourcePsi.xy, Gxz = sourcePsi.xz;
dfloat Gyy = sourcePsi.yy, Gyz = sourcePsi.yz, Gzz = sourcePsi.zz;

// Force coupling: F_p = (eta_p/lambda) div(S(A)).
#if !defined(DISABLE_POLYMER_MOMENTUM_FEEDBACK) && !defined(CONFORMATION_DIRECT_STRESS_HALO)
dfloat polymerFeedbackRamp = 1.0_df;

#ifdef VISCOELASTIC_STARTUP_RAMP
{
    const dfloat r = fmin(1.0_df, (dfloat)step / (dfloat)STARTUP_RAMP_STEPS);
    polymerFeedbackRamp = r*r*(3.0_df - 2.0_df*r);
}
#endif // VISCOELASTIC_STARTUP_RAMP

// Stored gradients are gradients of Psi. D exp_Psi maps each directional derivative to dA, then the constitutive
// differential maps dA to dS. Contracting the directional dS tensors produces div(S).
const Sym3 dAdx = veFrechetExp(vePsiEigen, Sym3(dAxxdx, dAxydx, dAxzdx, dAyydx, dAyzdx, dAzzdx));
const Sym3 dAdy = veFrechetExp(vePsiEigen, Sym3(dAxxdy, dAxydy, dAxzdy, dAyydy, dAyzdy, dAzzdy));
const Sym3 dAdz = veFrechetExp(vePsiEigen, Sym3(dAxxdz, dAxydz, dAxzdz, dAyydz, dAyzdz, dAzzdz));
const Sym3 dSdx = calcVeStressReducedDifferential(veLocal, veA, dAdx);
const Sym3 dSdy = calcVeStressReducedDifferential(veLocal, veA, dAdy);
const Sym3 dSdz = calcVeStressReducedDifferential(veLocal, veA, dAdz);

const dfloat modulus = calcVePolymerModulus(veLocal);

dfloat polymerForceX = modulus*(dSdx.xx + dSdy.xy + dSdz.xz);
dfloat polymerForceY = modulus*(dSdx.xy + dSdy.yy + dSdz.yz);
dfloat polymerForceZ = modulus*(dSdx.xz + dSdy.yz + dSdz.zz);

#if defined(VE_FORCE_FULL_DIVERGENCE) && defined(PHI_DIST)
{
    const dfloat phaseRange = PHI_TWO - PHI_ONE;
    const dfloat etaSlope = phasePropsB.ve.eta_p - phasePropsA.ve.eta_p;
    const dfloat lambdaSlope = phasePropsB.ve.lambda - phasePropsA.ve.lambda;
    dfloat dGpDh = 0.0_df;
    if (h_phi > 1.0e-4_df && h_phi < 1.0_df - 1.0e-4_df &&
        veLocal.type != VE_NEWTONIAN && veLocal.lambda > 0.0_df) {
        dGpDh = (etaSlope*veLocal.lambda - veLocal.eta_p*lambdaSlope) / (veLocal.lambda*veLocal.lambda);
    }
    const Sym3 S = calcVeStressReduced(veLocal, veA);
    const dfloat dGx = dGpDh*phase_dphidx/phaseRange;
    const dfloat dGy = dGpDh*phase_dphidy/phaseRange;
    const dfloat dGz = dGpDh*phase_dphidz/phaseRange;
    polymerForceX += dGx*S.xx + dGy*S.xy + dGz*S.xz;
    polymerForceY += dGx*S.xy + dGy*S.yy + dGz*S.yz;
    polymerForceZ += dGx*S.xz + dGy*S.yz + dGz*S.zz;
}
#endif // VE_FORCE_FULL_DIVERGENCE && PHI_DIST

const dfloat feedbackWeight = switch_conf*polymerFeedbackRamp;
L_Fx += polymerForceX*feedbackWeight;
L_Fy += polymerForceY*feedbackWeight;
L_Fz += polymerForceZ*feedbackWeight;
#endif // !DISABLE_POLYMER_MOMENTUM_FEEDBACK && !CONFORMATION_DIRECT_STRESS_HALO
