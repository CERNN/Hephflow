dfloat nx_xm1 = fMom[idxMom(txm1, ty, tz, M3_NX_INDEX, ((x - 1 + NX) % NX) / BLOCK_NX, by, bz)];
//dfloat ny_xm1 = fMom[idxMom(txm1, ty, tz, M3_NY_INDEX, ((x - 1 + NX) % NX) / BLOCK_NX, by, bz)];
//dfloat nz_xm1 = fMom[idxMom(txm1, ty, tz, M3_NZ_INDEX, ((x - 1 + NX) % NX) / BLOCK_NX, by, bz)];

dfloat nx_xp1 = fMom[idxMom(txp1, ty, tz, M3_NX_INDEX, ((x + 1) % NX) / BLOCK_NX, by, bz)];
//dfloat ny_xp1 = fMom[idxMom(txp1, ty, tz, M3_NY_INDEX, ((x + 1) % NX) / BLOCK_NX, by, bz)];
//dfloat nz_xp1 = fMom[idxMom(txp1, ty, tz, M3_NZ_INDEX, ((x + 1) % NX) / BLOCK_NX, by, bz)];

//dfloat nx_ym1 = fMom[idxMom(tx, tym1, tz, M3_NX_INDEX, bx, ((y - 1 + NY) % NY) / BLOCK_NY, bz)];
dfloat ny_ym1 = fMom[idxMom(tx, tym1, tz, M3_NY_INDEX, bx, ((y - 1 + NY) % NY) / BLOCK_NY, bz)];
//dfloat nz_ym1 = fMom[idxMom(tx, tym1, tz, M3_NZ_INDEX, bx, ((y - 1 + NY) % NY) / BLOCK_NY, bz)];

//dfloat nx_yp1 = fMom[idxMom(tx, typ1, tz, M3_NX_INDEX, bx, ((y + 1) % NY) / BLOCK_NY, bz)];
dfloat ny_yp1 = fMom[idxMom(tx, typ1, tz, M3_NY_INDEX, bx, ((y + 1) % NY) / BLOCK_NY, bz)];
//dfloat nz_yp1 = fMom[idxMom(tx, typ1, tz, M3_NZ_INDEX, bx, ((y + 1) % NY) / BLOCK_NY, bz)];

//dfloat nx_zm1 = fMom[idxMom(tx, ty, tzm1, M3_NX_INDEX, bx, by, ((z - 1 + NZ) % NZ) / BLOCK_NZ)];
//dfloat ny_zm1 = fMom[idxMom(tx, ty, tzm1, M3_NY_INDEX, bx, by, ((z - 1 + NZ) % NZ) / BLOCK_NZ)];
dfloat nz_zm1 = fMom[idxMom(tx, ty, tzm1, M3_NZ_INDEX, bx, by, ((z - 1 + NZ) % NZ) / BLOCK_NZ)];

//dfloat nx_zp1 = fMom[idxMom(tx, ty, tzp1, M3_NX_INDEX, bx, by, ((z + 1) % NZ) / BLOCK_NZ)];
//dfloat ny_zp1 = fMom[idxMom(tx, ty, tzp1, M3_NY_INDEX, bx, by, ((z + 1) % NZ) / BLOCK_NZ)];
dfloat nz_zp1 = fMom[idxMom(tx, ty, tzp1, M3_NZ_INDEX, bx, by, ((z + 1) % NZ) / BLOCK_NZ)];


dfloat dnxdx = (nx_xp1 - nx_xm1) *0.5;
//dfloat dnydx = (ny_xp1 - ny_xm1) *0.5;
//dfloat dnzdx = (nz_xp1 - nz_xm1) *0.5;

//dfloat dnxdy = (nx_yp1 - nx_ym1) *0.5;
dfloat dnydy = (ny_yp1 - ny_ym1) *0.5;
//dfloat dnzdy = (nz_yp1 - nz_ym1) *0.5;

//dfloat dnxdz = (nx_zp1 - nx_zm1) *0.5;
//dfloat dnydz = (ny_zp1 - ny_zm1) *0.5;
dfloat dnzdz = (nz_zp1 - nz_zm1) *0.5;

if ((nodeType & 0b01010101) == 0b01010101) { // wall west
    dnxdx = (nx_xp1 - pnx);
    //dnydx = (ny_xp1 - pny);
    //dnzdx = (nz_xp1 - pnz);
}
if ((nodeType & 0b10101010) == 0b10101010) { // wall east
    dnxdx = (pnx - nx_xm1);
    //dnydx = (pny - ny_xm1);
    //dnzdx = (pnz - nz_xm1);
}
if ((nodeType & 0b00110011) == 0b00110011) { // wall south
    //dnxdy = (nx_yp1 - pnx);
    dnydy = (ny_yp1 - pny);
    //dnzdy = (nz_yp1 - pnz);
}
if ((nodeType & 0b11001100) == 0b11001100) { // wall north
    //dnxdy = (pnx - nx_ym1);
    dnydy = (pny - ny_ym1);
    //dnzdy = (pnz - nz_ym1);
}
if ((nodeType & 0b00001111) == 0b00001111) { // wall back
    //dnxdz = (nx_zp1 - pnx);
    //dnydz = (ny_zp1 - pny);
    dnzdz = (nz_zp1 - pnz);
}
if ((nodeType & 0b11110000) == 0b11110000) { // wall front
    //dnxdz = (nx_zm1 - pnx);
    //dnydz = (ny_zm1 - pny);
    dnzdz = (nz_zm1 - pnz);
}


dfloat curvature = dnxdx + dnydy + dnzdz;