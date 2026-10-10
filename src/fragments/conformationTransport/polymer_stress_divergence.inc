// Race-free centered divergence for polymer momentum coupling:
//
//   Fp = (eta_p/lambda) div(S(A)).
//
// For Oldroyd-B the staged tensor is A rather than A-I; the identity has zero
// divergence, so both forms produce the same force. Nonlinear models stage the
// reduced stress S(A) directly. Core values use shared memory, while neighbors
// across CUDA-block faces use the immutable halo generation for this timestep.

// Reconstruct the physical conformation tensor. Log builds retain the eigensystem because their evolution source uses
// it later in the same collision kernel; standard and square-root builds use the representation dispatcher directly.
const Sym3 veDirectStored(
    fMom[idxMom(tx, ty, tz, A_XX_C_INDEX, bx, by, bz)],
    fMom[idxMom(tx, ty, tz, A_XY_C_INDEX, bx, by, bz)],
    fMom[idxMom(tx, ty, tz, A_XZ_C_INDEX, bx, by, bz)],
    fMom[idxMom(tx, ty, tz, A_YY_C_INDEX, bx, by, bz)],
    fMom[idxMom(tx, ty, tz, A_YZ_C_INDEX, bx, by, bz)],
    fMom[idxMom(tx, ty, tz, A_ZZ_C_INDEX, bx, by, bz)]);

#if CONF_REPRESENTATION == CONF_REP_LOG
    VeEigenSym3 veDirectPsiEigen;
    const Sym3 veDirectPhysicalA = veExpSymmetric(veDirectStored,&veDirectPsiEigen);
#else
    const Sym3 veDirectPhysicalA =vePhysicalAFromStored(veDirectStored);
#endif // CONF_REPRESENTATION == CONF_REP_LOG

// Select the tensor whose divergence enters momentum. Oldroyd-B uses A because subtracting I cannot change its
// divergence; every nonlinear closure must use its model-specific reduced stress S(A).
const Sym3 veDirectForceTensor = phasePropsA.ve.type == VE_OLDROYD_B ? veDirectPhysicalA : calcVeStressReduced(phasePropsA.ve,veDirectPhysicalA);

// Stage all six symmetric components for neighbors inside this CUDA block. The following barrier is block-local;
// values across block faces therefore come from the immutable, prebuilt halo generation rather than another block.
const int veDirectLocalIndex = tx + BLOCK_NX*(ty + BLOCK_NY*tz);

s_pop[veDirectLocalIndex + 0*BLOCK_LBM_SIZE] = veDirectForceTensor.xx;
s_pop[veDirectLocalIndex + 1*BLOCK_LBM_SIZE] = veDirectForceTensor.xy;
s_pop[veDirectLocalIndex + 2*BLOCK_LBM_SIZE] = veDirectForceTensor.xz;
s_pop[veDirectLocalIndex + 3*BLOCK_LBM_SIZE] = veDirectForceTensor.yy;
s_pop[veDirectLocalIndex + 4*BLOCK_LBM_SIZE] = veDirectForceTensor.yz;
s_pop[veDirectLocalIndex + 5*BLOCK_LBM_SIZE] = veDirectForceTensor.zz;

activeBarrier.arrive_and_wait();

const int veXm=veDirectLocalIndex -1;
const int veXp=veDirectLocalIndex +1;
const int veYm=veDirectLocalIndex - BLOCK_NX;
const int veYp=veDirectLocalIndex + BLOCK_NX;
const int veZm=veDirectLocalIndex - BLOCK_NX*BLOCK_NY;
const int veZp=veDirectLocalIndex + BLOCK_NX*BLOCK_NY;

// Gather the minus/plus neighbors needed by each row of div(S). X faces provide (Sxx,Sxy,Sxz), Y faces provide
// (Sxy,Syy,Syz), and Z faces provide (Sxz,Syz,Szz); no unused tensor components are allocated in the face buffers.
// X-face neighbors provide (Sxx, Sxy, Sxz).
dfloat Sxx_xm1, Sxy_xm1, Sxz_xm1, Sxx_xp1, Sxy_xp1, Sxz_xp1;
if(x>0 && tx>0) {
    Sxx_xm1 = s_pop[veXm + 0*BLOCK_LBM_SIZE];
    Sxy_xm1 = s_pop[veXm + 1*BLOCK_LBM_SIZE];
    Sxz_xm1 = s_pop[veXm + 2*BLOCK_LBM_SIZE];
} else {
    const size_t h = g_idxPolymerStressHaloX(ty,tz,0,bxm1,by,bz);
    Sxx_xm1 = polymerStressHalo.X_1[h];
    Sxy_xm1 = polymerStressHalo.X_1[h+1];
    Sxz_xm1 = polymerStressHalo.X_1[h+2];
}
if(x+1<NX && tx<BLOCK_NX-1) {
    Sxx_xp1 = s_pop[veXp + 0*BLOCK_LBM_SIZE];
    Sxy_xp1 = s_pop[veXp + 1*BLOCK_LBM_SIZE];
    Sxz_xp1 = s_pop[veXp + 2*BLOCK_LBM_SIZE];
} else {
    const size_t h = g_idxPolymerStressHaloX(ty,tz,0,bxp1,by,bz);
    Sxx_xp1 = polymerStressHalo.X_0[h];
    Sxy_xp1 = polymerStressHalo.X_0[h+1];
    Sxz_xp1 = polymerStressHalo.X_0[h+2];
}

// Y-face neighbors provide (Sxy, Syy, Syz).
dfloat Sxy_ym1, Syy_ym1, Syz_ym1, Sxy_yp1, Syy_yp1, Syz_yp1;
if(y>0 && ty>0) {
    Sxy_ym1 = s_pop[veYm + 1*BLOCK_LBM_SIZE];
    Syy_ym1 = s_pop[veYm + 3*BLOCK_LBM_SIZE];
    Syz_ym1 = s_pop[veYm + 4*BLOCK_LBM_SIZE];
} else {
    const size_t h = g_idxPolymerStressHaloY(tx,tz,0,bx,bym1,bz);
    Sxy_ym1 = polymerStressHalo.Y_1[h];
    Syy_ym1 = polymerStressHalo.Y_1[h+1];
    Syz_ym1 = polymerStressHalo.Y_1[h+2];
}
if(y+1<NY && ty<BLOCK_NY-1) {
    Sxy_yp1 = s_pop[veYp + 1*BLOCK_LBM_SIZE];
    Syy_yp1 = s_pop[veYp + 3*BLOCK_LBM_SIZE];
    Syz_yp1 = s_pop[veYp + 4*BLOCK_LBM_SIZE];
} else {
    const size_t h = g_idxPolymerStressHaloY(tx,tz,0,bx,byp1,bz);
    Sxy_yp1 = polymerStressHalo.Y_0[h];
    Syy_yp1 = polymerStressHalo.Y_0[h+1];
    Syz_yp1 = polymerStressHalo.Y_0[h+2];
}

// Z-face neighbors provide (Sxz, Syz, Szz).
dfloat Sxz_zm1, Syz_zm1, Szz_zm1, Sxz_zp1, Syz_zp1, Szz_zp1;
if(z_local>0 && tz>0) {
    Sxz_zm1 = s_pop[veZm + 2*BLOCK_LBM_SIZE];
    Syz_zm1 = s_pop[veZm + 4*BLOCK_LBM_SIZE];
    Szz_zm1 = s_pop[veZm + 5*BLOCK_LBM_SIZE];
} else {
    const size_t h = g_idxPolymerStressHaloZ(tx,ty,0,bx,by,bzm1);
    Sxz_zm1 = polymerStressHalo.Z_1[h];
    Syz_zm1 = polymerStressHalo.Z_1[h+1];
    Szz_zm1 = polymerStressHalo.Z_1[h+2];
}
if(z_local+1<localNZ && tz<BLOCK_NZ-1) {
    Sxz_zp1 = s_pop[veZp + 2*BLOCK_LBM_SIZE];
    Syz_zp1 = s_pop[veZp + 4*BLOCK_LBM_SIZE];
    Szz_zp1 = s_pop[veZp + 5*BLOCK_LBM_SIZE];
} else {
    const size_t h = g_idxPolymerStressHaloZ(tx,ty,0,bx,by,bzp1);
    Sxz_zp1 = polymerStressHalo.Z_0[h];
    Syz_zp1 = polymerStressHalo.Z_0[h+1];
    Szz_zp1 = polymerStressHalo.Z_0[h+2];
}

// Centered derivatives are used in the interior. At a physical wall, the node-type masks below replace the centered
// value with the established one-sided derivative using the local stress and the available fluid-side neighbor.
dfloat direct_dSxxdx = 0.5_df * (Sxx_xp1 - Sxx_xm1);
dfloat direct_dSxydx = 0.5_df * (Sxy_xp1 - Sxy_xm1);
dfloat direct_dSxzdx = 0.5_df * (Sxz_xp1 - Sxz_xm1);
dfloat direct_dSxydy = 0.5_df * (Sxy_yp1 - Sxy_ym1);
dfloat direct_dSyydy = 0.5_df * (Syy_yp1 - Syy_ym1);
dfloat direct_dSyzdy = 0.5_df * (Syz_yp1 - Syz_ym1);
dfloat direct_dSxzdz = 0.5_df * (Sxz_zp1 - Sxz_zm1);
dfloat direct_dSyzdz = 0.5_df * (Syz_zp1 - Syz_zm1);
dfloat direct_dSzzdz = 0.5_df * (Szz_zp1 - Szz_zm1);

if ((nodeType & 0b01010101)==0b01010101) {
    direct_dSxxdx = Sxx_xp1 - veDirectForceTensor.xx;
    direct_dSxydx = Sxy_xp1 - veDirectForceTensor.xy;
    direct_dSxzdx = Sxz_xp1 - veDirectForceTensor.xz;
}
if ((nodeType & 0b10101010)==0b10101010) {
    direct_dSxxdx = veDirectForceTensor.xx - Sxx_xm1;
    direct_dSxydx = veDirectForceTensor.xy - Sxy_xm1;
    direct_dSxzdx = veDirectForceTensor.xz - Sxz_xm1;
}
if ((nodeType & 0b00110011)==0b00110011) {
    direct_dSxydy = Sxy_yp1 - veDirectForceTensor.xy;
    direct_dSyydy = Syy_yp1 - veDirectForceTensor.yy;
    direct_dSyzdy = Syz_yp1 - veDirectForceTensor.yz;
}
if ((nodeType & 0b11001100)==0b11001100) {
    direct_dSxydy = veDirectForceTensor.xy - Sxy_ym1;
    direct_dSyydy = veDirectForceTensor.yy - Syy_ym1;
    direct_dSyzdy = veDirectForceTensor.yz - Syz_ym1;
}
if ((nodeType & 0b00001111)==0b00001111) {
    direct_dSxzdz = Sxz_zp1 - veDirectForceTensor.xz;
    direct_dSyzdz = Syz_zp1 - veDirectForceTensor.yz;
    direct_dSzzdz = Szz_zp1 - veDirectForceTensor.zz;
}
if ((nodeType & 0b11110000)==0b11110000) {
    direct_dSxzdz = veDirectForceTensor.xz - Sxz_zm1;
    direct_dSyzdz = veDirectForceTensor.yz - Syz_zm1;
    direct_dSzzdz = veDirectForceTensor.zz - Szz_zm1;
}

// Contract the directional derivatives: div(S)_i = partial_j S_ij. The caller multiplies these reduced-stress
// divergences by eta_p/lambda and by the startup/activation weight.
const dfloat directDivStressX = direct_dSxxdx + direct_dSxydy + direct_dSxzdz;
const dfloat directDivStressY = direct_dSxydx + direct_dSyydy + direct_dSyzdz;
const dfloat directDivStressZ = direct_dSxzdx + direct_dSyzdz + direct_dSzzdz;
activeBarrier.arrive_and_wait();
