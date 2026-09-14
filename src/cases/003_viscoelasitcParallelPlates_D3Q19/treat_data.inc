// ---------------------------------------------------------------------------
//  treat_data.inc  —  viscoelastic Poiseuille (D3Q19)
//  Exports:
//    _l2_error   :  step, t/t_c, L2(ux), L2(Axx), L2(Axy)
// ---------------------------------------------------------------------------

// ---- helpers ---------------------------------------------------------------
auto getMom = [&](int x, int y, int z, int momIndex) {
    return h_fMom[idxMom(x % BLOCK_NX, y % BLOCK_NY, z % BLOCK_NZ, momIndex,
                         x / BLOCK_NX, y / BLOCK_NY, z / BLOCK_NZ)];
};
auto getConf = [&](int x, int y, int z, int momIndex) {
    return getMom(x, y, z, momIndex) - CONF_ZERO;
};

// ---- fixed x,z for the centreline profile ----------------------------------
int x_loc = NX / 2;
int z_loc = NZ / 2;

// ---- analytical constants (Poiseuille between y = 0 and y = L) -------------
//  lambda = Wi * t_c = Wi * (L / U_C)  — convective relaxation time
const dfloat invL    = 1.0 / L;
const dfloat invL2   = invL  * invL;
const dfloat invL4   = invL2 * invL2;
const dfloat lambda  = Weissenberg_number * t_c;

// ---- loop over ALL y-nodes: accumulate relative L2 for ux, Axx, Axy -------
//  Matches MATLAB computeL2Error: all nodes included, NaN-filtered only.
//  Wall nodes (yy=0, yy=NY-1) have ux_ana=0 → contribute 0 to sumDen_ux
//  and any boundary residual to sumNum_ux, consistent with the MATLAB norm.
dfloat sumNum_ux  = 0.0,  sumDen_ux  = 0.0;
dfloat sumNum_Axx = 0.0,  sumDen_Axx = 0.0;
dfloat sumNum_Axy = 0.0,  sumDen_Axy = 0.0;

for (int yy = 0; yy < NY; ++yy)
{
    dfloat y_phys  = (dfloat)yy / (dfloat)(NY - 1) * L;
    dfloat L_2y    = L - 2.0 * y_phys;          // (L - 2y), reused below

    // --- numerical values ---
    dfloat ux  = getMom(x_loc, yy, z_loc, M_UX_INDEX) / F_M_I_SCALE;
    dfloat Axx = getConf(x_loc, yy, z_loc, A_XX_C_INDEX);
    dfloat Axy = getConf(x_loc, yy, z_loc, A_XY_C_INDEX);

    // --- analytical solutions (steady-state Oldroyd-B Poiseuille) ---
    dfloat ux_ana  = 4.0 * U_C  * y_phys * (L - y_phys) * invL2;
    dfloat Axx_ana = 1.0 + 32.0 * lambda * lambda * U_C * U_C * L_2y * L_2y * invL4;
    dfloat Axy_ana = -4.0 * lambda * U_C * (2.0 * y_phys - L) * invL2;

    // --- accumulate relative L2 sums ---
    dfloat dux  = ux  - ux_ana;
    dfloat dAxx = Axx - Axx_ana;
    dfloat dAxy = Axy - Axy_ana;

    sumNum_ux  += dux  * dux;    sumDen_ux  += ux_ana  * ux_ana;
    sumNum_Axx += dAxx * dAxx;   sumDen_Axx += Axx_ana * Axx_ana;
    sumNum_Axy += dAxy * dAxy;   sumDen_Axy += Axy_ana * Axy_ana;
}

dfloat l2_ux  = sqrt(sumNum_ux  / sumDen_ux);
dfloat l2_Axx = sqrt(sumNum_Axx / sumDen_Axx);
dfloat l2_Axy = sqrt(sumNum_Axy / sumDen_Axy);

// ---- export L2 errors ------------------------------------------------------
std::ostringstream strL2;
strL2 << std::scientific << std::setprecision(8);
strL2 << step << "," << (dfloat)step / t_c << ","
      << l2_ux << "," << l2_Axx << "," << l2_Axy;
saveTreatData("_l2_error", strL2.str(), step);

velocityProfile(h_fMom, 1, x_loc, 0, z_loc, step);
conformationProfile(h_fMom, 1, x_loc, 0, z_loc, step);

#ifdef SAVE_LOCAL_FORCES
forceProfile(d_Local_Fx, d_Local_Fy, d_Local_Fz, 1, x_loc, 0, z_loc, step);
#ifdef A_XX_DIST
sourceProfile(d_Source_Gxx, "source_Gxx", 1, x_loc, 0, z_loc, step);
#endif
#ifdef A_XY_DIST
sourceProfile(d_Source_Gxy, "source_Gxy", 1, x_loc, 0, z_loc, step);
#endif
#ifdef A_XZ_DIST
sourceProfile(d_Source_Gxz, "source_Gxz", 1, x_loc, 0, z_loc, step);
#endif
#ifdef A_YY_DIST
sourceProfile(d_Source_Gyy, "source_Gyy", 1, x_loc, 0, z_loc, step);
#endif
#ifdef A_YZ_DIST
sourceProfile(d_Source_Gyz, "source_Gyz", 1, x_loc, 0, z_loc, step);
#endif
#ifdef A_ZZ_DIST
sourceProfile(d_Source_Gzz, "source_Gzz", 1, x_loc, 0, z_loc, step);
#endif
#endif

