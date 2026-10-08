// Symmetric square-root representation. The transported field is Q = sqrt(A), governed by
//
//   DQ/Dt = G Q + Q L^T + 1/2 Q^-1 R(A).
//
// The Balci construction chooses the antisymmetric part of G from a three-equation linear system so that Q remains
// symmetric. The SPD solve and conditioning checks report diagnostic flags instead of silently accepting invalid Q.

// Local properties
dfloat switch_conf = step > CONFORMATION_SWITCH ? 1.0_df : 0.0_df;

#ifndef PHI_DIST
    dfloat h_phi = 1.0_df;
#endif // PHI_DIST
#ifdef PHI_DIST
    const veFluidProps veLocal = blendVeProps(phasePropsA.ve, phasePropsB.ve, h_phi);
#else
    const veFluidProps& veLocal=phasePropsA.ve;
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

// Remove the phase-field contribution before building the velocity-gradient matrix L.
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

// Reconstruct A=Q^2. The direct-stress fragment performs this once before the shared-memory barrier and exposes the
// result here; square-root builds require that path, so no alternate reconstruction branch is needed.
const Sym3 veQ(AxxVar, AxyVar, AxzVar, AyyVar, AyzVar, AzzVar);
const Sym3& veA = veDirectPhysicalA;
const Sym3 veR = calcVeRelaxation(veLocal, veA);
const dfloat velocityScale = 1.0_df/F_M_I_SCALE;

// L uses the row/column convention expected by veSquareRootSource. That helper solves the Balci antisymmetric system
// and evaluates GQ + QL^T + 1/2 Q^-1 R(A) while checking that Q remains sufficiently well-conditioned and SPD.
const VeMat3 veL{{
    {velocityScale*duxdx_eff,velocityScale*duxdy_eff,velocityScale*duxdz_eff},
    {velocityScale*duydx_eff,velocityScale*duydy_eff,velocityScale*duydz_eff},
    {velocityScale*duzdx_eff,velocityScale*duzdy_eff,velocityScale*duzdz_eff}}};

bool veSqrtSourceOk = false;

unsigned int veSqrtDiagnostics = VE_SQRT_DIAG_NONE;

// The scalar auxiliary LBM already contributes D*laplacian(Q). Map physical
// D*laplacian(A) through the square-root Sylvester operator and subtract the
// stored-variable diffusion so Q and A solve the same regularized equation.
const Sym3 veSqrtDiffusionCorrection =
    veSquareRootTangent(veQ,veLaplacianPhysicalA)*CONF_DIFFUSIVITY
    -veLaplacianStored*CONF_DIFFUSIVITY;
const Sym3 veQSource = veSquareRootSource(
    veQ,veL,veR,&veSqrtSourceOk,&veSqrtDiagnostics)*confSourceWeight
    +veSqrtDiffusionCorrection;

dfloat Gxx = veQSource.xx, Gxy = veQSource.xy, Gxz = veQSource.xz;
dfloat Gyy = veQSource.yy, Gyz = veQSource.yz, Gzz = veQSource.zz;

// Diagnostics are compiled out of production builds and do not alter the source.
#ifdef DEBUG_CONFORMATION_SQRT
    if (!veSqrtSourceOk || !isfinite(Gxx) || !isfinite(Gxy) || !isfinite(Gxz) || !isfinite(Gyy) || !isfinite(Gyz) || !isfinite(Gzz)) {
        printf("Invalid square-root source at (%d,%d,%d), step=%u, diagnostics=0x%x\n", x,y,z,step,veSqrtDiagnostics);
    }
#endif // DEBUG_CONFORMATION_SQRT
