// Centered finite differences of the transported symmetric tensor. Depending
// on CONF_REPRESENTATION, these components represent physical A or Psi=log(A).
// The full 3x3 derivative is required by nonlinear stress differentials and by
// the log-conformation Frechet force path.

// Fetch neighboring values for all six stored components.
const int topConf = params.numBlocksZ - 1;

const int planeIdxConf = x + y * NX;
const int bxEastConf = (x == NX - 1 || tx == BLOCK_NX - 1) ? bxp1 : bx;
const int byNorthConf = (y == NY - 1 || ty == BLOCK_NY - 1) ? byp1 : by;
const int bzFrontConf = (z_local + 1 >= localNZ || tz == BLOCK_NZ - 1) ? bzp1 : bz;
const bool atFrontConf = z_local + 1 >= localNZ;

dfloat Axx_xm1 = fMom[idxMom(txm1, ty, tz, A_XX_C_INDEX, ((tx == 0) ? bxm1 : bx) , by, bz)];
dfloat Axx_xp1 = fMom[idxMom(txp1, ty, tz, A_XX_C_INDEX, bxEastConf, by, bz)];

dfloat Ayy_xm1 = fMom[idxMom(txm1, ty, tz, A_YY_C_INDEX, ((tx == 0) ? bxm1 : bx), by, bz)];
dfloat Ayy_xp1 = fMom[idxMom(txp1, ty, tz, A_YY_C_INDEX, bxEastConf, by, bz)];
dfloat Azz_xm1 = fMom[idxMom(txm1, ty, tz, A_ZZ_C_INDEX, ((tx == 0) ? bxm1 : bx), by, bz)];
dfloat Azz_xp1 = fMom[idxMom(txp1, ty, tz, A_ZZ_C_INDEX, bxEastConf, by, bz)];
dfloat Ayz_xm1 = fMom[idxMom(txm1, ty, tz, A_YZ_C_INDEX, ((tx == 0) ? bxm1 : bx), by, bz)];
dfloat Ayz_xp1 = fMom[idxMom(txp1, ty, tz, A_YZ_C_INDEX, bxEastConf, by, bz)];

dfloat Axy_ym1 = fMom[idxMom(tx, tym1, tz, A_XY_C_INDEX, bx, ((ty == 0) ? bym1 : by), bz)];
dfloat Axy_yp1 = fMom[idxMom(tx, typ1, tz, A_XY_C_INDEX, bx, byNorthConf, bz)];

dfloat Axx_ym1 = fMom[idxMom(tx, tym1, tz, A_XX_C_INDEX, bx, ((ty == 0) ? bym1 : by), bz)];
dfloat Axx_yp1 = fMom[idxMom(tx, typ1, tz, A_XX_C_INDEX, bx, byNorthConf, bz)];
dfloat Axz_ym1 = fMom[idxMom(tx, tym1, tz, A_XZ_C_INDEX, bx, ((ty == 0) ? bym1 : by), bz)];
dfloat Axz_yp1 = fMom[idxMom(tx, typ1, tz, A_XZ_C_INDEX, bx, byNorthConf, bz)];
dfloat Azz_ym1 = fMom[idxMom(tx, tym1, tz, A_ZZ_C_INDEX, bx, ((ty == 0) ? bym1 : by), bz)];
dfloat Azz_yp1 = fMom[idxMom(tx, typ1, tz, A_ZZ_C_INDEX, bx, byNorthConf, bz)];

dfloat Axz_zm1 = ((tz == 0) && (bz == 0)) ? macroInterfaceGPUAxz.auxZ_1[planeIdxConf] : fMom[idxMom(tx, ty, tzm1, A_XZ_C_INDEX, bx, by, ((tz == 0) ? bzm1 : bz))];
dfloat Axz_zp1 = (atFrontConf && bz == topConf) ? macroInterfaceGPUAxz.auxZ_0[planeIdxConf] : fMom[idxMom(tx, ty, tzp1, A_XZ_C_INDEX, bx, by, bzFrontConf)];

dfloat Axx_zm1 = ((tz == 0) && (bz == 0)) ? macroInterfaceGPUAxx.auxZ_1[planeIdxConf] : fMom[idxMom(tx, ty, tzm1, A_XX_C_INDEX, bx, by, ((tz == 0) ? bzm1 : bz))];
dfloat Axx_zp1 = (atFrontConf && bz == topConf) ? macroInterfaceGPUAxx.auxZ_0[planeIdxConf] : fMom[idxMom(tx, ty, tzp1, A_XX_C_INDEX, bx, by, bzFrontConf)];
dfloat Axy_zm1 = ((tz == 0) && (bz == 0)) ? macroInterfaceGPUAxy.auxZ_1[planeIdxConf] : fMom[idxMom(tx, ty, tzm1, A_XY_C_INDEX, bx, by, ((tz == 0) ? bzm1 : bz))];
dfloat Axy_zp1 = (atFrontConf && bz == topConf) ? macroInterfaceGPUAxy.auxZ_0[planeIdxConf] : fMom[idxMom(tx, ty, tzp1, A_XY_C_INDEX, bx, by, bzFrontConf)];
dfloat Ayy_zm1 = ((tz == 0) && (bz == 0)) ? macroInterfaceGPUAyy.auxZ_1[planeIdxConf] : fMom[idxMom(tx, ty, tzm1, A_YY_C_INDEX, bx, by, ((tz == 0) ? bzm1 : bz))];
dfloat Ayy_zp1 = (atFrontConf && bz == topConf) ? macroInterfaceGPUAyy.auxZ_0[planeIdxConf] : fMom[idxMom(tx, ty, tzp1, A_YY_C_INDEX, bx, by, bzFrontConf)];

dfloat Axy_xm1 = fMom[idxMom(txm1, ty, tz, A_XY_C_INDEX, ((tx == 0) ? bxm1 : bx) , by, bz)];
dfloat Axy_xp1 = fMom[idxMom(txp1, ty, tz, A_XY_C_INDEX, bxEastConf, by, bz)];

dfloat Ayy_ym1 = fMom[idxMom(tx, tym1, tz, A_YY_C_INDEX, bx, ((ty == 0) ? bym1 : by), bz)];
dfloat Ayy_yp1 = fMom[idxMom(tx, typ1, tz, A_YY_C_INDEX, bx, byNorthConf, bz)];

dfloat Ayz_zm1 = ((tz == 0) && (bz == 0)) ? macroInterfaceGPUAyz.auxZ_1[planeIdxConf] : fMom[idxMom(tx, ty, tzm1, A_YZ_C_INDEX, bx, by, ((tz == 0) ? bzm1 : bz))];
dfloat Ayz_zp1 = (atFrontConf && bz == topConf) ? macroInterfaceGPUAyz.auxZ_0[planeIdxConf] : fMom[idxMom(tx, ty, tzp1, A_YZ_C_INDEX, bx, by, bzFrontConf)];

dfloat Axz_xm1 = fMom[idxMom(txm1, ty, tz, A_XZ_C_INDEX, ((tx == 0) ? bxm1 : bx) , by, bz)];
dfloat Axz_xp1 = fMom[idxMom(txp1, ty, tz, A_XZ_C_INDEX, bxEastConf, by, bz)];

dfloat Ayz_ym1 = fMom[idxMom(tx, tym1, tz, A_YZ_C_INDEX, bx, ((ty == 0) ? bym1 : by), bz)];
dfloat Ayz_yp1 = fMom[idxMom(tx, typ1, tz, A_YZ_C_INDEX, bx, byNorthConf, bz)];

dfloat Azz_zm1 = ((tz == 0) && (bz == 0)) ? macroInterfaceGPUAzz.auxZ_1[planeIdxConf] : fMom[idxMom(tx, ty, tzm1, A_ZZ_C_INDEX, bx, by, ((tz == 0) ? bzm1 : bz))];
dfloat Azz_zp1 = (atFrontConf && bz == topConf) ? macroInterfaceGPUAzz.auxZ_0[planeIdxConf] : fMom[idxMom(tx, ty, tzp1, A_ZZ_C_INDEX, bx, by, bzFrontConf)];

// Compute derivatives
dfloat dAxxdx = (Axx_xp1 - Axx_xm1) *0.5_df;
dfloat dAxydy = (Axy_yp1 - Axy_ym1) *0.5_df;
dfloat dAxzdz = (Axz_zp1 - Axz_zm1) *0.5_df;

dfloat dAxydx = (Axy_xp1 - Axy_xm1) *0.5_df;
dfloat dAyydy = (Ayy_yp1 - Ayy_ym1) *0.5_df;
dfloat dAyzdz = (Ayz_zp1 - Ayz_zm1) *0.5_df;

dfloat dAxzdx = (Axz_xp1 - Axz_xm1) *0.5_df;
dfloat dAyzdy = (Ayz_yp1 - Ayz_ym1) *0.5_df;
dfloat dAzzdz = (Azz_zp1 - Azz_zm1) *0.5_df;

// Remaining components are required by nonlinear stress differentials.
dfloat dAyydx = (Ayy_xp1 - Ayy_xm1) * 0.5_df;
dfloat dAzzdx = (Azz_xp1 - Azz_xm1) * 0.5_df;
dfloat dAyzdx = (Ayz_xp1 - Ayz_xm1) * 0.5_df;
dfloat dAxxdy = (Axx_yp1 - Axx_ym1) * 0.5_df;
dfloat dAxzdy = (Axz_yp1 - Axz_ym1) * 0.5_df;
dfloat dAzzdy = (Azz_yp1 - Azz_ym1) * 0.5_df;
dfloat dAxxdz = (Axx_zp1 - Axx_zm1) * 0.5_df;
dfloat dAxydz = (Axy_zp1 - Axy_zm1) * 0.5_df;
dfloat dAyydz = (Ayy_zp1 - Ayy_zm1) * 0.5_df;

if ((nodeType & 0b01010101) == 0b01010101) { // wall west
    dAxxdx = (Axx_xp1 - AxxVar);
    dAxydx = (Axy_xp1 - AxyVar);
    dAxzdx = (Axz_xp1 - AxzVar);
    dAyydx = (Ayy_xp1 - AyyVar);
    dAyzdx = (Ayz_xp1 - AyzVar);
    dAzzdx = (Azz_xp1 - AzzVar);
}
if ((nodeType & 0b10101010) == 0b10101010) { // wall east
    dAxxdx = (AxxVar - Axx_xm1);
    dAxydx = (AxyVar - Axy_xm1);
    dAxzdx = (AxzVar - Axz_xm1);
    dAyydx = (AyyVar - Ayy_xm1);
    dAyzdx = (AyzVar - Ayz_xm1);
    dAzzdx = (AzzVar - Azz_xm1);
}
if ((nodeType & 0b00110011) == 0b00110011) { // wall south
    dAxxdy = (Axx_yp1 - AxxVar);
    dAxydy = (Axy_yp1 - AxyVar);
    dAyydy = (Ayy_yp1 - AyyVar);
    dAyzdy = (Ayz_yp1 - AyzVar);
    dAxzdy = (Axz_yp1 - AxzVar);
    dAzzdy = (Azz_yp1 - AzzVar);
}
if ((nodeType & 0b11001100) == 0b11001100) { // wall north
    dAxxdy = (AxxVar - Axx_ym1);
    dAxydy = (AxyVar - Axy_ym1);
    dAyydy = (AyyVar - Ayy_ym1);
    dAyzdy = (AyzVar - Ayz_ym1);
    dAxzdy = (AxzVar - Axz_ym1);
    dAzzdy = (AzzVar - Azz_ym1);
}
if ((nodeType & 0b00001111) == 0b00001111) { // wall back
    dAxxdz = (Axx_zp1 - AxxVar);
    dAxydz = (Axy_zp1 - AxyVar);
    dAxzdz = (Axz_zp1 - AxzVar);
    dAyzdz = (Ayz_zp1 - AyzVar);
    dAzzdz = (Azz_zp1 - AzzVar);
    dAyydz = (Ayy_zp1 - AyyVar);
}
if ((nodeType & 0b11110000) == 0b11110000) { // wall front
    dAxxdz = (AxxVar - Axx_zm1);
    dAxydz = (AxyVar - Axy_zm1);
    dAxzdz = (AxzVar - Axz_zm1);
    dAyzdz = (AyzVar - Ayz_zm1);
    dAzzdz = (AzzVar - Azz_zm1);
    dAyydz = (AyyVar - Ayy_zm1);
}
