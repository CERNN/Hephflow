// Encode arbitrary populations in Listing A6's parity-0 Esoteric Push layout.
const int initPuTxM1 = (tx - 1 + BLOCK_NX) % BLOCK_NX;
const int initPuTyM1 = (ty - 1 + BLOCK_NY) % BLOCK_NY;
const int initPuTyP1 = (ty + 1) % BLOCK_NY;
const int initPuTzM1 = (tz - 1 + BLOCK_NZ) % BLOCK_NZ;
const int initPuTzP1 = (tz + 1) % BLOCK_NZ;
const int initPuBxM1 = (bx - 1 + NUM_BLOCK_X) % NUM_BLOCK_X;
const int initPuByM1 = (by - 1 + NUM_BLOCK_Y) % NUM_BLOCK_Y;
const int initPuByP1 = (by + 1) % NUM_BLOCK_Y;
const int initPuBzM1 = (bz - 1 + NUM_BLOCK_Z_LOCAL) % NUM_BLOCK_Z_LOCAL;
const int initPuBzP1 = (bz + 1) % NUM_BLOCK_Z_LOCAL;

const int initPuBxAtWest  = (tx == 0)              ? initPuBxM1 : bx;
const int initPuByAtSouth = (ty == 0)              ? initPuByM1 : by;
const int initPuByAtNorth = (ty == (BLOCK_NY - 1)) ? initPuByP1 : by;
const int initPuBzAtBack  = (tz == 0)              ? initPuBzM1 : bz;
const int initPuBzAtFront = (tz == (BLOCK_NZ - 1)) ? initPuBzP1 : bz;

if (tx == 0) {
    ghostInterface.pop.X_0[idxPopX(ty,         tz,         0, initPuBxM1, by,                bz)] = pop[ 2];
    ghostInterface.pop.X_0[idxPopX(initPuTyM1, tz,         1, initPuBxM1, initPuByAtSouth,   bz)] = pop[ 8];
    ghostInterface.pop.X_0[idxPopX(ty,         initPuTzM1, 2, initPuBxM1, by,                initPuBzAtBack)] = pop[10];
    ghostInterface.pop.X_0[idxPopX(initPuTyP1, tz,         3, initPuBxM1, initPuByAtNorth,   bz)] = pop[14];
    ghostInterface.pop.X_0[idxPopX(ty,         initPuTzP1, 4, initPuBxM1, by,                initPuBzAtFront)] = pop[16];
} else if (tx == (BLOCK_NX - 1)) {
    ghostInterface.pop.X_1[idxPopX(ty, tz, 0, bx, by, bz)] = pop[ 1];
    ghostInterface.pop.X_1[idxPopX(ty, tz, 1, bx, by, bz)] = pop[ 7];
    ghostInterface.pop.X_1[idxPopX(ty, tz, 2, bx, by, bz)] = pop[ 9];
    ghostInterface.pop.X_1[idxPopX(ty, tz, 3, bx, by, bz)] = pop[13];
    ghostInterface.pop.X_1[idxPopX(ty, tz, 4, bx, by, bz)] = pop[15];
}

if (ty == 0) {
    ghostInterface.pop.Y_0[idxPopY(tx,         tz,         0, bx,             initPuByM1, bz)] = pop[ 4];
    ghostInterface.pop.Y_0[idxPopY(initPuTxM1, tz,         1, initPuBxAtWest, initPuByM1, bz)] = pop[ 8];
    ghostInterface.pop.Y_0[idxPopY(tx,         initPuTzM1, 2, bx,             initPuByM1, initPuBzAtBack)] = pop[12];
    ghostInterface.pop.Y_0[idxPopY(tx,         tz,         3, bx,             by,         bz)] = pop[13];
    ghostInterface.pop.Y_0[idxPopY(tx,         initPuTzP1, 4, bx,             initPuByM1, initPuBzAtFront)] = pop[18];
} else if (ty == (BLOCK_NY - 1)) {
    ghostInterface.pop.Y_1[idxPopY(tx,         tz, 0, bx,             by,         bz)] = pop[ 3];
    ghostInterface.pop.Y_1[idxPopY(tx,         tz, 1, bx,             by,         bz)] = pop[ 7];
    ghostInterface.pop.Y_1[idxPopY(tx,         tz, 2, bx,             by,         bz)] = pop[11];
    ghostInterface.pop.Y_1[idxPopY(initPuTxM1, tz, 3, initPuBxAtWest, initPuByP1, bz)] = pop[14];
    ghostInterface.pop.Y_1[idxPopY(tx,         tz, 4, bx,             by,         bz)] = pop[17];
}

if (tz == 0) {
    ghostInterface.pop.Z_0[idxPopZ(tx,         ty,         0, bx,             by,              initPuBzM1)] = pop[ 6];
    ghostInterface.pop.Z_0[idxPopZ(initPuTxM1, ty,         1, initPuBxAtWest, by,              initPuBzM1)] = pop[10];
    ghostInterface.pop.Z_0[idxPopZ(tx,         initPuTyM1, 2, bx,             initPuByAtSouth, initPuBzM1)] = pop[12];
    ghostInterface.pop.Z_0[idxPopZ(tx,         ty,         3, bx,             by,              bz)] = pop[15];
    ghostInterface.pop.Z_0[idxPopZ(tx,         ty,         4, bx,             by,              bz)] = pop[17];
} else if (tz == (BLOCK_NZ - 1)) {
    ghostInterface.pop.Z_1[idxPopZ(tx,         ty,         0, bx,             by,              bz)] = pop[ 5];
    ghostInterface.pop.Z_1[idxPopZ(tx,         ty,         1, bx,             by,              bz)] = pop[ 9];
    ghostInterface.pop.Z_1[idxPopZ(tx,         ty,         2, bx,             by,              bz)] = pop[11];
    ghostInterface.pop.Z_1[idxPopZ(initPuTxM1, ty,         3, initPuBxAtWest, by,              initPuBzP1)] = pop[16];
    ghostInterface.pop.Z_1[idxPopZ(tx,         initPuTyM1, 4, bx,             initPuByAtSouth, initPuBzP1)] = pop[18];
}
