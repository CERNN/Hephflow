// gradient
dfloat phi_xm1 = fMom[idxMom(txm1, ty, tz, M3_PHI_INDEX, ((x - 1 + NX) % NX) / BLOCK_NX, by, bz)];
dfloat phi_xp1 = fMom[idxMom(txp1, ty, tz, M3_PHI_INDEX, ((x + 1) % NX) / BLOCK_NX, by, bz)];
dfloat phi_ym1 = fMom[idxMom(tx, tym1, tz, M3_PHI_INDEX, bx, ((y - 1 + NY) % NY) / BLOCK_NY, bz)];
dfloat phi_yp1 = fMom[idxMom(tx, typ1, tz, M3_PHI_INDEX, bx, ((y + 1) % NY) / BLOCK_NY, bz)];
dfloat phi_zm1 = fMom[idxMom(tx, ty, tzm1, M3_PHI_INDEX, bx, by, ((z - 1 + NZ) % NZ) / BLOCK_NZ)];
dfloat phi_zp1 = fMom[idxMom(tx, ty, tzp1, M3_PHI_INDEX, bx, by, ((z + 1) % NZ) / BLOCK_NZ)];


dfloat dphidx = (phi_xp1 - phi_xm1)/2;
dfloat dphidy = (phi_yp1 - phi_ym1)/2;
dfloat dphidz = (phi_zp1 - phi_zm1)/2;


if ((nodeType & 0b01010101) == 0b01010101) { // wall west
    dphidx = (phi_xp1 - phiVar);
}
if ((nodeType & 0b10101010) == 0b10101010) { // wall east
    dphidx = (phiVar - phi_xm1);
}
if ((nodeType & 0b00110011) == 0b00110011) { // wall south
    dphidy = (phi_yp1 - phiVar);
}
if ((nodeType & 0b11001100) == 0b11001100) { // wall north
    dphidy = (phiVar - phi_ym1);
}
if ((nodeType & 0b00001111) == 0b00001111) { // wall back
    dphidz = (phi_yp1 - phiVar);
}
if ((nodeType & 0b11110000) == 0b11110000) { // wall front
    dphidz = (phi_ym1 - phiVar);
}

dfloat mod_grad = sqrtf(dphidx*dphidx + dphidy*dphidy + dphidz*dphidz);
dfloat pnx = dphidx / (mod_grad + 1e-9);
dfloat pny = dphidy / (mod_grad + 1e-9);
dfloat pnz = dphidz / (mod_grad + 1e-9);

fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M3_NX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = pnx;
fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M3_NY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = pny;
fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M3_NZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = pnz;



