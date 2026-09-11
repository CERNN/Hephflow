// Listing A4 Esoteric Twist applied to block-face populations.  The final
// three diagonal pairs use ET's criss-cross axial offsets.
const dfloat* twistXPos = stepParity ? ghostInterface.X_0 : ghostInterface.X_1;
const dfloat* twistXNeg = stepParity ? ghostInterface.X_1 : ghostInterface.X_0;
const dfloat* twistYPos = stepParity ? ghostInterface.Y_0 : ghostInterface.Y_1;
const dfloat* twistYNeg = stepParity ? ghostInterface.Y_1 : ghostInterface.Y_0;
const dfloat* twistZPos = stepParity ? ghostInterface.Z_0 : ghostInterface.Z_1;
const dfloat* twistZNeg = stepParity ? ghostInterface.Z_1 : ghostInterface.Z_0;

const int twistBxAtEast  = (tx == (BLOCK_NX - 1)) ? bxp1 : bx;
const int twistByAtNorth = (ty == (BLOCK_NY - 1)) ? byp1 : by;
const int twistBzAtFront = (tz == (BLOCK_NZ - 1)) ? bzp1 : bz;

if (tx == 0) {
    pop[ 1] = twistXPos[idxPopX(ty,   tz,   0, bx, by,             bz)];
    pop[ 7] = twistXPos[idxPopX(ty,   tz,   1, bx, by,             bz)];
    pop[ 9] = twistXPos[idxPopX(ty,   tz,   2, bx, by,             bz)];
    pop[13] = twistXPos[idxPopX(typ1, tz,   3, bx, twistByAtNorth, bz)];
    pop[15] = twistXPos[idxPopX(ty,   tzp1, 4, bx, by,             twistBzAtFront)];
} else if (tx == (BLOCK_NX - 1)) {
    pop[ 2] = twistXNeg[idxPopX(ty,   tz,   0, bxp1, by, bz)];
    pop[ 8] = twistXNeg[idxPopX(typ1, tz,   1, bxp1, twistByAtNorth, bz)];
    pop[10] = twistXNeg[idxPopX(ty,   tzp1, 2, bxp1, by, twistBzAtFront)];
    pop[14] = twistXNeg[idxPopX(ty,   tz,   3, bxp1, by, bz)];
    pop[16] = twistXNeg[idxPopX(ty,   tz,   4, bxp1, by, bz)];
}

if (ty == 0) {
    pop[ 3] = twistYPos[idxPopY(tx,   tz,   0, bx,             by, bz)];
    pop[ 7] = twistYPos[idxPopY(tx,   tz,   1, bx,             by, bz)];
    pop[11] = twistYPos[idxPopY(tx,   tz,   2, bx,             by, bz)];
    pop[14] = twistYPos[idxPopY(txp1, tz,   3, twistBxAtEast,  by, bz)];
    pop[17] = twistYPos[idxPopY(tx,   tzp1, 4, bx,             by, twistBzAtFront)];
} else if (ty == (BLOCK_NY - 1)) {
    pop[ 4] = twistYNeg[idxPopY(tx,   tz,   0, bx,            byp1, bz)];
    pop[ 8] = twistYNeg[idxPopY(txp1, tz,   1, twistBxAtEast, byp1, bz)];
    pop[12] = twistYNeg[idxPopY(tx,   tzp1, 2, bx,            byp1, twistBzAtFront)];
    pop[13] = twistYNeg[idxPopY(tx,   tz,   3, bx,            byp1, bz)];
    pop[18] = twistYNeg[idxPopY(tx,   tz,   4, bx,            byp1, bz)];
}

// Direct periodic Z addressing intentionally limits this experiment to one GPU.
if (tz == 0) {
    pop[ 5] = twistZPos[idxPopZ(tx,   ty,   0, bx,            by,             bz)];
    pop[ 9] = twistZPos[idxPopZ(tx,   ty,   1, bx,            by,             bz)];
    pop[11] = twistZPos[idxPopZ(tx,   ty,   2, bx,            by,             bz)];
    pop[16] = twistZPos[idxPopZ(txp1, ty,   3, twistBxAtEast, by,             bz)];
    pop[18] = twistZPos[idxPopZ(tx,   typ1, 4, bx,            twistByAtNorth, bz)];
} else if (tz == (BLOCK_NZ - 1)) {
    pop[ 6] = twistZNeg[idxPopZ(tx,   ty,   0, bx,            by,             bzp1)];
    pop[10] = twistZNeg[idxPopZ(txp1, ty,   1, twistBxAtEast, by,             bzp1)];
    pop[12] = twistZNeg[idxPopZ(tx,   typ1, 2, bx,            twistByAtNorth, bzp1)];
    pop[15] = twistZNeg[idxPopZ(tx,   ty,   3, bx,            by,             bzp1)];
    pop[17] = twistZNeg[idxPopZ(tx,   ty,   4, bx,            by,             bzp1)];
}
