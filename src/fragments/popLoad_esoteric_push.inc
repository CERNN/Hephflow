// Listing A6 Esoteric Push.  Even directions are local; each odd direction
// is displaced by the complete velocity of its even opposite.
const dfloat* pushXPos = stepParity ? ghostInterface.X_0 : ghostInterface.X_1;
const dfloat* pushXNeg = stepParity ? ghostInterface.X_1 : ghostInterface.X_0;
const dfloat* pushYPos = stepParity ? ghostInterface.Y_0 : ghostInterface.Y_1;
const dfloat* pushYNeg = stepParity ? ghostInterface.Y_1 : ghostInterface.Y_0;
const dfloat* pushZPos = stepParity ? ghostInterface.Z_0 : ghostInterface.Z_1;
const dfloat* pushZNeg = stepParity ? ghostInterface.Z_1 : ghostInterface.Z_0;

const int pushBxAtWest  = (tx == 0)              ? bxm1 : bx;
const int pushByAtSouth = (ty == 0)              ? bym1 : by;
const int pushByAtNorth = (ty == (BLOCK_NY - 1) || y == NY - 1) ? byp1 : by;
const int pushBzAtBack  = (tz == 0)              ? bzm1 : bz;
const int pushBzAtFront = (tz == (BLOCK_NZ - 1) || z_local + 1 >= localNZ) ? bzp1 : bz;

if (tx == 0) {
    pop[ 1] = pushXPos[idxPopX(ty,   tz,   0, bxm1, by,             bz)];
    pop[ 7] = pushXPos[idxPopX(tym1, tz,   1, bxm1, pushByAtSouth, bz)];
    pop[ 9] = pushXPos[idxPopX(ty,   tzm1, 2, bxm1, by,             pushBzAtBack)];
    pop[13] = pushXPos[idxPopX(typ1, tz,   3, bxm1, pushByAtNorth, bz)];
    pop[15] = pushXPos[idxPopX(ty,   tzp1, 4, bxm1, by,             pushBzAtFront)];
}
if (tx == (BLOCK_NX - 1) || x == NX - 1) {
    pop[ 2] = pushXNeg[idxPopX(ty, tz, 0, bx, by, bz)];
    pop[ 8] = pushXNeg[idxPopX(ty, tz, 1, bx, by, bz)];
    pop[10] = pushXNeg[idxPopX(ty, tz, 2, bx, by, bz)];
    pop[14] = pushXNeg[idxPopX(ty, tz, 3, bx, by, bz)];
    pop[16] = pushXNeg[idxPopX(ty, tz, 4, bx, by, bz)];
}

if (ty == 0) {
    pop[ 3] = pushYPos[idxPopY(tx,   tz,   0, bx,           bym1, bz)];
    pop[ 7] = pushYPos[idxPopY(txm1, tz,   1, pushBxAtWest, bym1, bz)];
    pop[11] = pushYPos[idxPopY(tx,   tzm1, 2, bx,           bym1, pushBzAtBack)];
    pop[14] = pushYPos[idxPopY(tx,   tz,   3, bx,           by,   bz)];
    pop[17] = pushYPos[idxPopY(tx,   tzp1, 4, bx,           bym1, pushBzAtFront)];
}
if (ty == (BLOCK_NY - 1) || y == NY - 1) {
    pop[ 4] = pushYNeg[idxPopY(tx, tz, 0, bx, by, bz)];
    pop[ 8] = pushYNeg[idxPopY(tx, tz, 1, bx, by, bz)];
    pop[12] = pushYNeg[idxPopY(tx, tz, 2, bx, by, bz)];
    pop[13] = pushYNeg[idxPopY(tx, tz, 3, bx, by, bz)];
    pop[18] = pushYNeg[idxPopY(tx, tz, 4, bx, by, bz)];
}

if (tz == 0) {
    const dfloat* zSource = (bz == 0)
        ? (stepParity ? ghostInterface.auxZ_1 : ghostInterfaceCopy.auxZ_1)
        : pushZPos;
    const int sourceBz = (bz == 0) ? 0 : bzm1;
    pop[ 5] = zSource[idxPopZ(tx,   ty,   0, bx,           by,            sourceBz)];
    pop[ 9] = zSource[idxPopZ(txm1, ty,   1, pushBxAtWest, by,            sourceBz)];
    pop[11] = zSource[idxPopZ(tx,   tym1, 2, bx,           pushByAtSouth, sourceBz)];
    pop[16] = zSource[idxPopZ(tx,   ty,   3, bx,           by,            sourceBz)];
    pop[18] = zSource[idxPopZ(tx,   ty,   4, bx,           by,            sourceBz)];
}
if (tz == (BLOCK_NZ - 1) || z_local + 1 >= localNZ) {
    const bool slabTop = (bz + 1 == params.numBlocksZ);
    const dfloat* zSource = slabTop
        ? (stepParity ? ghostInterface.auxZ_0 : ghostInterfaceCopy.auxZ_0)
        : pushZNeg;
    const int sourceBz = slabTop ? 0 : bz;
    pop[ 6] = zSource[idxPopZ(tx, ty, 0, bx, by, sourceBz)];
    pop[10] = zSource[idxPopZ(tx, ty, 1, bx, by, sourceBz)];
    pop[12] = zSource[idxPopZ(tx, ty, 2, bx, by, sourceBz)];
    pop[15] = zSource[idxPopZ(tx, ty, 3, bx, by, sourceBz)];
    pop[17] = zSource[idxPopZ(tx, ty, 4, bx, by, sourceBz)];
}
