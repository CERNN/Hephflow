dfloat* pushXPosStore = stepParity ? ghostInterface.X_0 : ghostInterface.X_1;
dfloat* pushXNegStore = stepParity ? ghostInterface.X_1 : ghostInterface.X_0;
dfloat* pushYPosStore = stepParity ? ghostInterface.Y_0 : ghostInterface.Y_1;
dfloat* pushYNegStore = stepParity ? ghostInterface.Y_1 : ghostInterface.Y_0;
dfloat* pushZPosStore = stepParity ? ghostInterface.Z_0 : ghostInterface.Z_1;
dfloat* pushZNegStore = stepParity ? ghostInterface.Z_1 : ghostInterface.Z_0;

if (INTERFACE_BC_WEST) {
    pushXPosStore[idxPopX(ty,   tz,   0, bxm1, by,             bz)] = pop[ 2];
    pushXPosStore[idxPopX(tym1, tz,   1, bxm1, pushByAtSouth, bz)] = pop[ 8];
    pushXPosStore[idxPopX(ty,   tzm1, 2, bxm1, by,             pushBzAtBack)] = pop[10];
    pushXPosStore[idxPopX(typ1, tz,   3, bxm1, pushByAtNorth, bz)] = pop[14];
    pushXPosStore[idxPopX(ty,   tzp1, 4, bxm1, by,             pushBzAtFront)] = pop[16];
}
if (INTERFACE_BC_EAST) {
    pushXNegStore[idxPopX(ty, tz, 0, bx, by, bz)] = pop[ 1];
    pushXNegStore[idxPopX(ty, tz, 1, bx, by, bz)] = pop[ 7];
    pushXNegStore[idxPopX(ty, tz, 2, bx, by, bz)] = pop[ 9];
    pushXNegStore[idxPopX(ty, tz, 3, bx, by, bz)] = pop[13];
    pushXNegStore[idxPopX(ty, tz, 4, bx, by, bz)] = pop[15];
}

if (INTERFACE_BC_SOUTH) {
    pushYPosStore[idxPopY(tx,   tz,   0, bx,           bym1, bz)] = pop[ 4];
    pushYPosStore[idxPopY(txm1, tz,   1, pushBxAtWest, bym1, bz)] = pop[ 8];
    pushYPosStore[idxPopY(tx,   tzm1, 2, bx,           bym1, pushBzAtBack)] = pop[12];
    pushYPosStore[idxPopY(tx,   tz,   3, bx,           by,   bz)] = pop[13];
    pushYPosStore[idxPopY(tx,   tzp1, 4, bx,           bym1, pushBzAtFront)] = pop[18];
}
if (INTERFACE_BC_NORTH) {
    pushYNegStore[idxPopY(tx,   tz, 0, bx,           by, bz)] = pop[ 3];
    pushYNegStore[idxPopY(tx,   tz, 1, bx,           by, bz)] = pop[ 7];
    pushYNegStore[idxPopY(tx,   tz, 2, bx,           by, bz)] = pop[11];
    pushYNegStore[idxPopY(txm1, tz, 3, pushBxAtWest, byp1, bz)] = pop[14];
    pushYNegStore[idxPopY(tx,   tz, 4, bx,           by, bz)] = pop[17];
}

if (INTERFACE_BC_BACK) {
    pushZPosStore[idxPopZ(tx,   ty,   0, bx,           by,            bzm1)] = pop[ 6];
    pushZPosStore[idxPopZ(txm1, ty,   1, pushBxAtWest, by,            bzm1)] = pop[10];
    pushZPosStore[idxPopZ(tx,   tym1, 2, bx,           pushByAtSouth, bzm1)] = pop[12];
    pushZPosStore[idxPopZ(tx,   ty,   3, bx,           by,            bz)] = pop[15];
    pushZPosStore[idxPopZ(tx,   ty,   4, bx,           by,            bz)] = pop[17];
}
if (INTERFACE_BC_FRONT) {
    pushZNegStore[idxPopZ(tx,   ty, 0, bx,           by,            bz)] = pop[ 5];
    pushZNegStore[idxPopZ(tx,   ty, 1, bx,           by,            bz)] = pop[ 9];
    pushZNegStore[idxPopZ(tx,   ty, 2, bx,           by,            bz)] = pop[11];
    pushZNegStore[idxPopZ(txm1, ty, 3, pushBxAtWest, by,            bzp1)] = pop[16];
    pushZNegStore[idxPopZ(tx,   tym1, 4, bx,           pushByAtSouth, bzp1)] = pop[18];
}
