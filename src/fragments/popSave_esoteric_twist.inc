dfloat* twistXPosStore = stepParity ? ghostInterface.X_0 : ghostInterface.X_1;
dfloat* twistXNegStore = stepParity ? ghostInterface.X_1 : ghostInterface.X_0;
dfloat* twistYPosStore = stepParity ? ghostInterface.Y_0 : ghostInterface.Y_1;
dfloat* twistYNegStore = stepParity ? ghostInterface.Y_1 : ghostInterface.Y_0;
dfloat* twistZPosStore = stepParity ? ghostInterface.Z_0 : ghostInterface.Z_1;
dfloat* twistZNegStore = stepParity ? ghostInterface.Z_1 : ghostInterface.Z_0;

if (INTERFACE_BC_WEST) {
    twistXPosStore[idxPopX(ty,   tz,   0, bx, by,             bz)] = pop[ 2];
    twistXPosStore[idxPopX(ty,   tz,   1, bx, by,             bz)] = pop[ 8];
    twistXPosStore[idxPopX(ty,   tz,   2, bx, by,             bz)] = pop[10];
    twistXPosStore[idxPopX(typ1, tz,   3, bx, twistByAtNorth, bz)] = pop[14];
    twistXPosStore[idxPopX(ty,   tzp1, 4, bx, by,             twistBzAtFront)] = pop[16];
}
if (INTERFACE_BC_EAST) {
    twistXNegStore[idxPopX(ty,   tz,   0, bxp1, by, bz)] = pop[ 1];
    twistXNegStore[idxPopX(typ1, tz,   1, bxp1, twistByAtNorth, bz)] = pop[ 7];
    twistXNegStore[idxPopX(ty,   tzp1, 2, bxp1, by, twistBzAtFront)] = pop[ 9];
    twistXNegStore[idxPopX(ty,   tz,   3, bxp1, by, bz)] = pop[13];
    twistXNegStore[idxPopX(ty,   tz,   4, bxp1, by, bz)] = pop[15];
}

if (INTERFACE_BC_SOUTH) {
    twistYPosStore[idxPopY(tx,   tz,   0, bx,            by, bz)] = pop[ 4];
    twistYPosStore[idxPopY(tx,   tz,   1, bx,            by, bz)] = pop[ 8];
    twistYPosStore[idxPopY(tx,   tz,   2, bx,            by, bz)] = pop[12];
    twistYPosStore[idxPopY(txp1, tz,   3, twistBxAtEast, by, bz)] = pop[13];
    twistYPosStore[idxPopY(tx,   tzp1, 4, bx,            by, twistBzAtFront)] = pop[18];
}
if (INTERFACE_BC_NORTH) {
    twistYNegStore[idxPopY(tx,   tz,   0, bx,            byp1, bz)] = pop[ 3];
    twistYNegStore[idxPopY(txp1, tz,   1, twistBxAtEast, byp1, bz)] = pop[ 7];
    twistYNegStore[idxPopY(tx,   tzp1, 2, bx,            byp1, twistBzAtFront)] = pop[11];
    twistYNegStore[idxPopY(tx,   tz,   3, bx,            byp1, bz)] = pop[14];
    twistYNegStore[idxPopY(tx,   tz,   4, bx,            byp1, bz)] = pop[17];
}

if (INTERFACE_BC_BACK) {
    twistZPosStore[idxPopZ(tx,   ty,   0, bx,            by,             bz)] = pop[ 6];
    twistZPosStore[idxPopZ(tx,   ty,   1, bx,            by,             bz)] = pop[10];
    twistZPosStore[idxPopZ(tx,   ty,   2, bx,            by,             bz)] = pop[12];
    twistZPosStore[idxPopZ(txp1, ty,   3, twistBxAtEast, by,             bz)] = pop[15];
    twistZPosStore[idxPopZ(tx,   typ1, 4, bx,            twistByAtNorth, bz)] = pop[17];
}
if (INTERFACE_BC_FRONT) {
    twistZNegStore[idxPopZ(tx,   ty,   0, bx,            by,             bzp1)] = pop[ 5];
    twistZNegStore[idxPopZ(txp1, ty,   1, twistBxAtEast, by,             bzp1)] = pop[ 9];
    twistZNegStore[idxPopZ(tx,   typ1, 2, bx,            twistByAtNorth, bzp1)] = pop[11];
    twistZNegStore[idxPopZ(tx,   ty,   3, bx,            by,             bzp1)] = pop[16];
    twistZNegStore[idxPopZ(tx,   ty,   4, bx,            by,             bzp1)] = pop[18];
}
