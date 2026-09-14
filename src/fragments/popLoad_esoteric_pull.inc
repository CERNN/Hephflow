// Esoteric Pull on the coarse block lattice.  Odd D3Q19 directions form the
// zero-offset half-neighborhood; each even direction is stored at c_opp.
// Swapping opposite population pointers is represented by these uniform,
// parity-selected logical views (no host-side pointer mutation required).
const dfloat* etXPos = stepParity ? ghostInterface.X_0 : ghostInterface.X_1;
const dfloat* etXNeg = stepParity ? ghostInterface.X_1 : ghostInterface.X_0;
const dfloat* etYPos = stepParity ? ghostInterface.Y_0 : ghostInterface.Y_1;
const dfloat* etYNeg = stepParity ? ghostInterface.Y_1 : ghostInterface.Y_0;
const dfloat* etZPos = stepParity ? ghostInterface.Z_0 : ghostInterface.Z_1;
const dfloat* etZNeg = stepParity ? ghostInterface.Z_1 : ghostInterface.Z_0;

const int bxAtEast  = (tx == (BLOCK_NX - 1)) ? bxp1 : bx;
const int byAtSouth = (ty == 0)              ? bym1 : by;
const int byAtNorth = (ty == (BLOCK_NY - 1)) ? byp1 : by;
const int bzAtBack  = (tz == 0)              ? bzm1 : bz;
const int bzAtFront = (tz == (BLOCK_NZ - 1)) ? bzp1 : bz;

// Incoming +x members have zero offset.  Incoming -x members use +c_opp.
if (tx == 0) {
    pop[ 1] = etXPos[idxPopX(ty, tz, 0, bx, by, bz)];
    pop[ 7] = etXPos[idxPopX(ty, tz, 1, bx, by, bz)];
    pop[ 9] = etXPos[idxPopX(ty, tz, 2, bx, by, bz)];
    pop[13] = etXPos[idxPopX(ty, tz, 3, bx, by, bz)];
    pop[15] = etXPos[idxPopX(ty, tz, 4, bx, by, bz)];
} else if (tx == (BLOCK_NX - 1)) {
    pop[ 2] = etXNeg[idxPopX(ty,   tz,   0, bxp1, by,        bz)];
    pop[ 8] = etXNeg[idxPopX(typ1, tz,   1, bxp1, byAtNorth, bz)];
    pop[10] = etXNeg[idxPopX(ty,   tzp1, 2, bxp1, by,        bzAtFront)];
    pop[14] = etXNeg[idxPopX(tym1, tz,   3, bxp1, byAtSouth, bz)];
    pop[16] = etXNeg[idxPopX(ty,   tzm1, 4, bxp1, by,        bzAtBack)];
}

if (ty == 0) {
    pop[ 3] = etYPos[idxPopY(tx,   tz, 0, bx,       by,   bz)];
    pop[ 7] = etYPos[idxPopY(tx,   tz, 1, bx,       by,   bz)];
    pop[11] = etYPos[idxPopY(tx,   tz, 2, bx,       by,   bz)];
    pop[14] = etYPos[idxPopY(txp1, tz, 3, bxAtEast, bym1, bz)];
    pop[17] = etYPos[idxPopY(tx,   tz, 4, bx,       by,   bz)];
} else if (ty == (BLOCK_NY - 1)) {
    pop[ 4] = etYNeg[idxPopY(tx,   tz,   0, bx,       byp1, bz)];
    pop[ 8] = etYNeg[idxPopY(txp1, tz,   1, bxAtEast, byp1, bz)];
    pop[12] = etYNeg[idxPopY(tx,   tzp1, 2, bx,       byp1, bzAtFront)];
    pop[13] = etYNeg[idxPopY(tx,   tz,   3, bx,       by,   bz)];
    pop[18] = etYNeg[idxPopY(tx,   tzm1, 4, bx,       byp1, bzAtBack)];
}

// The single-GPU experiment addresses periodic Z blocks directly.  A future
// multi-GPU version must exchange the parity-selected ET planes.
if (tz == 0) {
    pop[ 5] = etZPos[idxPopZ(tx,   ty,   0, bx,       by,        bz)];
    pop[ 9] = etZPos[idxPopZ(tx,   ty,   1, bx,       by,        bz)];
    pop[11] = etZPos[idxPopZ(tx,   ty,   2, bx,       by,        bz)];
    pop[16] = etZPos[idxPopZ(txp1, ty,   3, bxAtEast, by,        bzm1)];
    pop[18] = etZPos[idxPopZ(tx,   typ1, 4, bx,       byAtNorth, bzm1)];
} else if (tz == (BLOCK_NZ - 1)) {
    pop[ 6] = etZNeg[idxPopZ(tx,   ty,   0, bx,       by,        bzp1)];
    pop[10] = etZNeg[idxPopZ(txp1, ty,   1, bxAtEast, by,        bzp1)];
    pop[12] = etZNeg[idxPopZ(tx,   typ1, 2, bx,       byAtNorth, bzp1)];
    pop[15] = etZNeg[idxPopZ(tx,   ty,   3, bx,       by,        bz)];
    pop[17] = etZNeg[idxPopZ(tx,   ty,   4, bx,       by,        bz)];
}
