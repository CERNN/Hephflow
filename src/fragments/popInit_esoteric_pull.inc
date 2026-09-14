// Encode arbitrary reconstructed populations in the parity-0 Esoteric Pull
// face representation.  For an outgoing f_i at source x, initialize logical
// P_i at destination + o_i = x + o_opp(i), exactly where the first pull reads.
const int initTxP1 = (tx + 1) % BLOCK_NX;
const int initTyP1 = (ty + 1) % BLOCK_NY;
const int initTyM1 = (ty - 1 + BLOCK_NY) % BLOCK_NY;
const int initTzP1 = (tz + 1) % BLOCK_NZ;
const int initTzM1 = (tz - 1 + BLOCK_NZ) % BLOCK_NZ;

const int initBxP1 = (bx + 1) % NUM_BLOCK_X;
const int initByM1 = (by - 1 + NUM_BLOCK_Y) % NUM_BLOCK_Y;
const int initByP1 = (by + 1) % NUM_BLOCK_Y;
const int initBzM1 = (bz - 1 + NUM_BLOCK_Z_LOCAL) % NUM_BLOCK_Z_LOCAL;
const int initBzP1 = (bz + 1) % NUM_BLOCK_Z_LOCAL;

const int initBxAtEast  = (tx == (BLOCK_NX - 1)) ? initBxP1 : bx;
const int initByAtSouth = (ty == 0)              ? initByM1 : by;
const int initByAtNorth = (ty == (BLOCK_NY - 1)) ? initByP1 : by;
const int initBzAtBack  = (tz == 0)              ? initBzM1 : bz;
const int initBzAtFront = (tz == (BLOCK_NZ - 1)) ? initBzP1 : bz;

if (tx == 0) {
    ghostInterface.pop.X_0[idxPopX(ty, tz, 0, bx, by, bz)] = pop[ 2];
    ghostInterface.pop.X_0[idxPopX(ty, tz, 1, bx, by, bz)] = pop[ 8];
    ghostInterface.pop.X_0[idxPopX(ty, tz, 2, bx, by, bz)] = pop[10];
    ghostInterface.pop.X_0[idxPopX(ty, tz, 3, bx, by, bz)] = pop[14];
    ghostInterface.pop.X_0[idxPopX(ty, tz, 4, bx, by, bz)] = pop[16];
} else if (tx == (BLOCK_NX - 1)) {
    ghostInterface.pop.X_1[idxPopX(ty,       tz,       0, initBxP1, by,            bz)] = pop[ 1];
    ghostInterface.pop.X_1[idxPopX(initTyP1, tz,       1, initBxP1, initByAtNorth, bz)] = pop[ 7];
    ghostInterface.pop.X_1[idxPopX(ty,       initTzP1, 2, initBxP1, by,            initBzAtFront)] = pop[ 9];
    ghostInterface.pop.X_1[idxPopX(initTyM1, tz,       3, initBxP1, initByAtSouth, bz)] = pop[13];
    ghostInterface.pop.X_1[idxPopX(ty,       initTzM1, 4, initBxP1, by,            initBzAtBack)] = pop[15];
}

if (ty == 0) {
    ghostInterface.pop.Y_0[idxPopY(tx,       tz, 0, bx,           by,       bz)] = pop[ 4];
    ghostInterface.pop.Y_0[idxPopY(tx,       tz, 1, bx,           by,       bz)] = pop[ 8];
    ghostInterface.pop.Y_0[idxPopY(tx,       tz, 2, bx,           by,       bz)] = pop[12];
    ghostInterface.pop.Y_0[idxPopY(initTxP1, tz, 3, initBxAtEast, initByM1, bz)] = pop[13];
    ghostInterface.pop.Y_0[idxPopY(tx,       tz, 4, bx,           by,       bz)] = pop[18];
} else if (ty == (BLOCK_NY - 1)) {
    ghostInterface.pop.Y_1[idxPopY(tx,       tz,       0, bx,           initByP1, bz)] = pop[ 3];
    ghostInterface.pop.Y_1[idxPopY(initTxP1, tz,       1, initBxAtEast, initByP1, bz)] = pop[ 7];
    ghostInterface.pop.Y_1[idxPopY(tx,       initTzP1, 2, bx,           initByP1, initBzAtFront)] = pop[11];
    ghostInterface.pop.Y_1[idxPopY(tx,       tz,       3, bx,           by,       bz)] = pop[14];
    ghostInterface.pop.Y_1[idxPopY(tx,       initTzM1, 4, bx,           initByP1, initBzAtBack)] = pop[17];
}

if (tz == 0) {
    ghostInterface.pop.Z_0[idxPopZ(tx,       ty,       0, bx,           by,            bz)] = pop[ 6];
    ghostInterface.pop.Z_0[idxPopZ(tx,       ty,       1, bx,           by,            bz)] = pop[10];
    ghostInterface.pop.Z_0[idxPopZ(tx,       ty,       2, bx,           by,            bz)] = pop[12];
    ghostInterface.pop.Z_0[idxPopZ(initTxP1, ty,       3, initBxAtEast, by,            initBzM1)] = pop[15];
    ghostInterface.pop.Z_0[idxPopZ(tx,       initTyP1, 4, bx,           initByAtNorth, initBzM1)] = pop[17];
} else if (tz == (BLOCK_NZ - 1)) {
    ghostInterface.pop.Z_1[idxPopZ(tx,       ty,       0, bx,           by,            initBzP1)] = pop[ 5];
    ghostInterface.pop.Z_1[idxPopZ(initTxP1, ty,       1, initBxAtEast, by,            initBzP1)] = pop[ 9];
    ghostInterface.pop.Z_1[idxPopZ(tx,       initTyP1, 2, bx,           initByAtNorth, initBzP1)] = pop[11];
    ghostInterface.pop.Z_1[idxPopZ(tx,       ty,       3, bx,           by,            bz)] = pop[16];
    ghostInterface.pop.Z_1[idxPopZ(tx,       ty,       4, bx,           by,            bz)] = pop[18];
}
