dfloat* etXPosStore = stepParity ? ghostInterface.X_0 : ghostInterface.X_1;
dfloat* etXNegStore = stepParity ? ghostInterface.X_1 : ghostInterface.X_0;
dfloat* etYPosStore = stepParity ? ghostInterface.Y_0 : ghostInterface.Y_1;
dfloat* etYNegStore = stepParity ? ghostInterface.Y_1 : ghostInterface.Y_0;
dfloat* etZPosStore = stepParity ? ghostInterface.Z_0 : ghostInterface.Z_1;
dfloat* etZNegStore = stepParity ? ghostInterface.Z_1 : ghostInterface.Z_0;

// Esoteric Pull writes g_i to P_opp(i)[x + o_opp(i)].  Each write below targets exactly
// the physical face address read by the same boundary thread in popLoad.
if (INTERFACE_BC_WEST) {
    etXPosStore[idxPopX(ty, tz, 0, bx, by, bz)] = pop[ 2];
    etXPosStore[idxPopX(ty, tz, 1, bx, by, bz)] = pop[ 8];
    etXPosStore[idxPopX(ty, tz, 2, bx, by, bz)] = pop[10];
    etXPosStore[idxPopX(ty, tz, 3, bx, by, bz)] = pop[14];
    etXPosStore[idxPopX(ty, tz, 4, bx, by, bz)] = pop[16];
}
if (INTERFACE_BC_EAST) {
    etXNegStore[idxPopX(ty,   tz,   0, bxp1, by,        bz)] = pop[ 1];
    etXNegStore[idxPopX(typ1, tz,   1, bxp1, byAtNorth, bz)] = pop[ 7];
    etXNegStore[idxPopX(ty,   tzp1, 2, bxp1, by,        bzAtFront)] = pop[ 9];
    etXNegStore[idxPopX(tym1, tz,   3, bxp1, byAtSouth, bz)] = pop[13];
    etXNegStore[idxPopX(ty,   tzm1, 4, bxp1, by,        bzAtBack)] = pop[15];
}

if (INTERFACE_BC_SOUTH) {
    etYPosStore[idxPopY(tx,   tz, 0, bx,       by,   bz)] = pop[ 4];
    etYPosStore[idxPopY(tx,   tz, 1, bx,       by,   bz)] = pop[ 8];
    etYPosStore[idxPopY(tx,   tz, 2, bx,       by,   bz)] = pop[12];
    etYPosStore[idxPopY(txp1, tz, 3, bxAtEast, bym1, bz)] = pop[13];
    etYPosStore[idxPopY(tx,   tz, 4, bx,       by,   bz)] = pop[18];
}
if (INTERFACE_BC_NORTH) {
    etYNegStore[idxPopY(tx,   tz,   0, bx,       byp1, bz)] = pop[ 3];
    etYNegStore[idxPopY(txp1, tz,   1, bxAtEast, byp1, bz)] = pop[ 7];
    etYNegStore[idxPopY(tx,   tzp1, 2, bx,       byp1, bzAtFront)] = pop[11];
    etYNegStore[idxPopY(tx,   tz,   3, bx,       by,   bz)] = pop[14];
    etYNegStore[idxPopY(tx,   tzm1, 4, bx,       byp1, bzAtBack)] = pop[17];
}

if (INTERFACE_BC_BACK) {
    etZPosStore[idxPopZ(tx,   ty,   0, bx,       by,        bz)] = pop[ 6];
    etZPosStore[idxPopZ(tx,   ty,   1, bx,       by,        bz)] = pop[10];
    etZPosStore[idxPopZ(tx,   ty,   2, bx,       by,        bz)] = pop[12];
    etZPosStore[idxPopZ(txp1, ty,   3, bxAtEast, by,        bzm1)] = pop[16];
    etZPosStore[idxPopZ(tx,   typ1, 4, bx,       byAtNorth, bzm1)] = pop[18];
}
if (INTERFACE_BC_FRONT) {
    etZNegStore[idxPopZ(tx,   ty,   0, bx,       by,        bzp1)] = pop[ 5];
    etZNegStore[idxPopZ(txp1, ty,   1, bxAtEast, by,        bzp1)] = pop[ 9];
    etZNegStore[idxPopZ(tx,   typ1, 2, bx,       byAtNorth, bzp1)] = pop[11];
    etZNegStore[idxPopZ(tx,   ty,   3, bx,       by,        bz)] = pop[15];
    etZNegStore[idxPopZ(tx,   ty,   4, bx,       by,        bz)] = pop[17];
}
