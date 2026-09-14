// Encode arbitrary populations in Listing A4's parity-0 Esoteric Twist layout.
const int initTwTxP1 = (tx + 1) % BLOCK_NX;
const int initTwTyP1 = (ty + 1) % BLOCK_NY;
const int initTwTzP1 = (tz + 1) % BLOCK_NZ;
const int initTwBxP1 = (bx + 1) % NUM_BLOCK_X;
const int initTwByP1 = (by + 1) % NUM_BLOCK_Y;
const int initTwBzP1 = (bz + 1) % NUM_BLOCK_Z_LOCAL;

const int initTwBxAtEast  = (tx == (BLOCK_NX - 1)) ? initTwBxP1 : bx;
const int initTwByAtNorth = (ty == (BLOCK_NY - 1)) ? initTwByP1 : by;
const int initTwBzAtFront = (tz == (BLOCK_NZ - 1)) ? initTwBzP1 : bz;

if (tx == 0) {
    ghostInterface.pop.X_0[idxPopX(ty,         tz,         0, bx, by,                bz)] = pop[ 2];
    ghostInterface.pop.X_0[idxPopX(ty,         tz,         1, bx, by,                bz)] = pop[ 8];
    ghostInterface.pop.X_0[idxPopX(ty,         tz,         2, bx, by,                bz)] = pop[10];
    ghostInterface.pop.X_0[idxPopX(initTwTyP1, tz,         3, bx, initTwByAtNorth,   bz)] = pop[14];
    ghostInterface.pop.X_0[idxPopX(ty,         initTwTzP1, 4, bx, by,                initTwBzAtFront)] = pop[16];
} else if (tx == (BLOCK_NX - 1)) {
    ghostInterface.pop.X_1[idxPopX(ty,         tz,         0, initTwBxP1, by,              bz)] = pop[ 1];
    ghostInterface.pop.X_1[idxPopX(initTwTyP1, tz,         1, initTwBxP1, initTwByAtNorth, bz)] = pop[ 7];
    ghostInterface.pop.X_1[idxPopX(ty,         initTwTzP1, 2, initTwBxP1, by,              initTwBzAtFront)] = pop[ 9];
    ghostInterface.pop.X_1[idxPopX(ty,         tz,         3, initTwBxP1, by,              bz)] = pop[13];
    ghostInterface.pop.X_1[idxPopX(ty,         tz,         4, initTwBxP1, by,              bz)] = pop[15];
}

if (ty == 0) {
    ghostInterface.pop.Y_0[idxPopY(tx,         tz,         0, bx,             by, bz)] = pop[ 4];
    ghostInterface.pop.Y_0[idxPopY(tx,         tz,         1, bx,             by, bz)] = pop[ 8];
    ghostInterface.pop.Y_0[idxPopY(tx,         tz,         2, bx,             by, bz)] = pop[12];
    ghostInterface.pop.Y_0[idxPopY(initTwTxP1, tz,         3, initTwBxAtEast, by, bz)] = pop[13];
    ghostInterface.pop.Y_0[idxPopY(tx,         initTwTzP1, 4, bx,             by, initTwBzAtFront)] = pop[18];
} else if (ty == (BLOCK_NY - 1)) {
    ghostInterface.pop.Y_1[idxPopY(tx,         tz,         0, bx,             initTwByP1, bz)] = pop[ 3];
    ghostInterface.pop.Y_1[idxPopY(initTwTxP1, tz,         1, initTwBxAtEast, initTwByP1, bz)] = pop[ 7];
    ghostInterface.pop.Y_1[idxPopY(tx,         initTwTzP1, 2, bx,             initTwByP1, initTwBzAtFront)] = pop[11];
    ghostInterface.pop.Y_1[idxPopY(tx,         tz,         3, bx,             initTwByP1, bz)] = pop[14];
    ghostInterface.pop.Y_1[idxPopY(tx,         tz,         4, bx,             initTwByP1, bz)] = pop[17];
}

if (tz == 0) {
    ghostInterface.pop.Z_0[idxPopZ(tx,         ty,         0, bx,             by,              bz)] = pop[ 6];
    ghostInterface.pop.Z_0[idxPopZ(tx,         ty,         1, bx,             by,              bz)] = pop[10];
    ghostInterface.pop.Z_0[idxPopZ(tx,         ty,         2, bx,             by,              bz)] = pop[12];
    ghostInterface.pop.Z_0[idxPopZ(initTwTxP1, ty,         3, initTwBxAtEast, by,              bz)] = pop[15];
    ghostInterface.pop.Z_0[idxPopZ(tx,         initTwTyP1, 4, bx,             initTwByAtNorth, bz)] = pop[17];
} else if (tz == (BLOCK_NZ - 1)) {
    ghostInterface.pop.Z_1[idxPopZ(tx,         ty,         0, bx,             by,              initTwBzP1)] = pop[ 5];
    ghostInterface.pop.Z_1[idxPopZ(initTwTxP1, ty,         1, initTwBxAtEast, by,              initTwBzP1)] = pop[ 9];
    ghostInterface.pop.Z_1[idxPopZ(tx,         initTwTyP1, 2, bx,             initTwByAtNorth, initTwBzP1)] = pop[11];
    ghostInterface.pop.Z_1[idxPopZ(tx,         ty,         3, bx,             by,              initTwBzP1)] = pop[16];
    ghostInterface.pop.Z_1[idxPopZ(tx,         ty,         4, bx,             by,              initTwBzP1)] = pop[18];
}
