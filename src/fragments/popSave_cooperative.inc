// D3Q19 cooperative save for the directions in COOPERATIVE_GHOST_SAVE_MASK.
// Selected faces share one packed transfer and one barrier pair.
activeBarrier.arrive_and_wait();

constexpr int coopSaveFaceNodes = BLOCK_NY * BLOCK_NZ;
static_assert(BLOCK_NX * BLOCK_NZ == coopSaveFaceNodes &&
              BLOCK_NX * BLOCK_NY == coopSaveFaceNodes,
              "Cooperative multi-direction save requires equal face areas");
constexpr int coopSavePopCount = 5;
constexpr int coopSaveFaceValues = coopSaveFaceNodes * coopSavePopCount;
constexpr int coopSaveDirectionCount =
    ((COOPERATIVE_GHOST_SAVE_MASK & 1) ? 1 : 0) +
    ((COOPERATIVE_GHOST_SAVE_MASK & 2) ? 1 : 0) +
    ((COOPERATIVE_GHOST_SAVE_MASK & 4) ? 1 : 0);
constexpr int coopSaveValues = 2 * coopSaveDirectionCount * coopSaveFaceValues;
constexpr int coopSaveThreads = BLOCK_NX * BLOCK_NY * BLOCK_NZ;
constexpr int coopSaveXOffset = 0;
constexpr int coopSaveYOffset = ((COOPERATIVE_GHOST_SAVE_MASK & 1) ? 2 : 0) * coopSaveFaceValues;
constexpr int coopSaveZOffset = (((COOPERATIVE_GHOST_SAVE_MASK & 1) ? 2 : 0) +
                                 ((COOPERATIVE_GHOST_SAVE_MASK & 2) ? 2 : 0)) * coopSaveFaceValues;
static_assert(coopSaveValues <= BLOCK_LBM_SIZE_POP,
    "The shared population tile is too small for cooperative ghost save");

#if COOPERATIVE_GHOST_SAVE_MASK & 1
const int coopSaveXNode = ty + BLOCK_NY * tz;
if (INTERFACE_BC_WEST) {
    s_pop[coopSaveXOffset + 0 * coopSaveFaceValues + 0 * coopSaveFaceNodes + coopSaveXNode] = pop[2];
    s_pop[coopSaveXOffset + 0 * coopSaveFaceValues + 1 * coopSaveFaceNodes + coopSaveXNode] = pop[8];
    s_pop[coopSaveXOffset + 0 * coopSaveFaceValues + 2 * coopSaveFaceNodes + coopSaveXNode] = pop[10];
    s_pop[coopSaveXOffset + 0 * coopSaveFaceValues + 3 * coopSaveFaceNodes + coopSaveXNode] = pop[14];
    s_pop[coopSaveXOffset + 0 * coopSaveFaceValues + 4 * coopSaveFaceNodes + coopSaveXNode] = pop[16];
}
if (INTERFACE_BC_EAST) {
    s_pop[coopSaveXOffset + 1 * coopSaveFaceValues + 0 * coopSaveFaceNodes + coopSaveXNode] = pop[1];
    s_pop[coopSaveXOffset + 1 * coopSaveFaceValues + 1 * coopSaveFaceNodes + coopSaveXNode] = pop[7];
    s_pop[coopSaveXOffset + 1 * coopSaveFaceValues + 2 * coopSaveFaceNodes + coopSaveXNode] = pop[9];
    s_pop[coopSaveXOffset + 1 * coopSaveFaceValues + 3 * coopSaveFaceNodes + coopSaveXNode] = pop[13];
    s_pop[coopSaveXOffset + 1 * coopSaveFaceValues + 4 * coopSaveFaceNodes + coopSaveXNode] = pop[15];
}

#endif // COOPERATIVE_GHOST_SAVE_MASK & 1

#if COOPERATIVE_GHOST_SAVE_MASK & 2
const int coopSaveYNode = tx + BLOCK_NX * tz;
if (INTERFACE_BC_SOUTH) {
    s_pop[coopSaveYOffset + 0 * coopSaveFaceValues + 0 * coopSaveFaceNodes + coopSaveYNode] = pop[4];
    s_pop[coopSaveYOffset + 0 * coopSaveFaceValues + 1 * coopSaveFaceNodes + coopSaveYNode] = pop[8];
    s_pop[coopSaveYOffset + 0 * coopSaveFaceValues + 2 * coopSaveFaceNodes + coopSaveYNode] = pop[12];
    s_pop[coopSaveYOffset + 0 * coopSaveFaceValues + 3 * coopSaveFaceNodes + coopSaveYNode] = pop[13];
    s_pop[coopSaveYOffset + 0 * coopSaveFaceValues + 4 * coopSaveFaceNodes + coopSaveYNode] = pop[18];
}
if (INTERFACE_BC_NORTH) {
    s_pop[coopSaveYOffset + 1 * coopSaveFaceValues + 0 * coopSaveFaceNodes + coopSaveYNode] = pop[3];
    s_pop[coopSaveYOffset + 1 * coopSaveFaceValues + 1 * coopSaveFaceNodes + coopSaveYNode] = pop[7];
    s_pop[coopSaveYOffset + 1 * coopSaveFaceValues + 2 * coopSaveFaceNodes + coopSaveYNode] = pop[11];
    s_pop[coopSaveYOffset + 1 * coopSaveFaceValues + 3 * coopSaveFaceNodes + coopSaveYNode] = pop[14];
    s_pop[coopSaveYOffset + 1 * coopSaveFaceValues + 4 * coopSaveFaceNodes + coopSaveYNode] = pop[17];
}

#endif // COOPERATIVE_GHOST_SAVE_MASK & 2

#if COOPERATIVE_GHOST_SAVE_MASK & 4
const int coopSaveZNode = tx + BLOCK_NX * ty;
if (INTERFACE_BC_BACK) {
    s_pop[coopSaveZOffset + 0 * coopSaveFaceValues + 0 * coopSaveFaceNodes + coopSaveZNode] = pop[6];
    s_pop[coopSaveZOffset + 0 * coopSaveFaceValues + 1 * coopSaveFaceNodes + coopSaveZNode] = pop[10];
    s_pop[coopSaveZOffset + 0 * coopSaveFaceValues + 2 * coopSaveFaceNodes + coopSaveZNode] = pop[12];
    s_pop[coopSaveZOffset + 0 * coopSaveFaceValues + 3 * coopSaveFaceNodes + coopSaveZNode] = pop[15];
    s_pop[coopSaveZOffset + 0 * coopSaveFaceValues + 4 * coopSaveFaceNodes + coopSaveZNode] = pop[17];
}
if (INTERFACE_BC_FRONT) {
    s_pop[coopSaveZOffset + 1 * coopSaveFaceValues + 0 * coopSaveFaceNodes + coopSaveZNode] = pop[5];
    s_pop[coopSaveZOffset + 1 * coopSaveFaceValues + 1 * coopSaveFaceNodes + coopSaveZNode] = pop[9];
    s_pop[coopSaveZOffset + 1 * coopSaveFaceValues + 2 * coopSaveFaceNodes + coopSaveZNode] = pop[11];
    s_pop[coopSaveZOffset + 1 * coopSaveFaceValues + 3 * coopSaveFaceNodes + coopSaveZNode] = pop[16];
    s_pop[coopSaveZOffset + 1 * coopSaveFaceValues + 4 * coopSaveFaceNodes + coopSaveZNode] = pop[18];
}

#endif // COOPERATIVE_GHOST_SAVE_MASK & 4

activeBarrier.arrive_and_wait();

const int coopSaveTid = tx + BLOCK_NX * (ty + BLOCK_NY * tz);
for (int coopLinear = coopSaveTid;
     coopLinear < coopSaveValues;
     coopLinear += coopSaveThreads) {
    const int coopCompactFace = coopLinear / coopSaveFaceValues;
    int coopFace = coopCompactFace;
    if (!(COOPERATIVE_GHOST_SAVE_MASK & 1)) coopFace += 2;
    if (coopFace >= 2 && !(COOPERATIVE_GHOST_SAVE_MASK & 2)) coopFace += 2;
    const int coopInFace = coopLinear - coopCompactFace * coopSaveFaceValues;
    const int coopPop = coopInFace / coopSaveFaceNodes;
    const int coopNode = coopInFace - coopPop * coopSaveFaceNodes;

    dfloat* coopDestination = nullptr;
    int coopDestinationIndex = 0;
    bool coopActive = true;

    if (coopFace <= 1) {
        const bool west = coopFace == 0;
        const int faceTy = coopNode % BLOCK_NY;
        const int faceTz = coopNode / BLOCK_NY;
        int destTy = faceTy, destTz = faceTz, destBy = by, destBz = bz;
        #ifndef BC_X_PERIODIC
        coopActive = west ? (bx != 0) : (bx != NUM_BLOCK_X - 1);
#endif // BC_X_PERIODIC
        if (popSaveParity) {
            coopDestination = west ? storeGhostXWest : storeGhostXEast;
        } else {
            coopDestination = west ? storeGhostXEast : storeGhostXWest;
            if (west) {
                if (coopPop == 1) { destTy = (faceTy - 1 + BLOCK_NY) % BLOCK_NY; destBy = (faceTy == 0) ? bym1 : by; }
                else if (coopPop == 2) { destTz = (faceTz - 1 + BLOCK_NZ) % BLOCK_NZ; destBz = (faceTz == 0) ? bzm1 : bz; }
                else if (coopPop == 3) { destTy = (faceTy + 1) % BLOCK_NY; destBy = (faceTy == BLOCK_NY - 1) ? byp1 : by; }
                else if (coopPop == 4) { destTz = (faceTz + 1) % BLOCK_NZ; destBz = (faceTz == BLOCK_NZ - 1) ? bzp1 : bz; }
            } else {
                if (coopPop == 1) { destTy = (faceTy + 1) % BLOCK_NY; destBy = (faceTy == BLOCK_NY - 1) ? byp1 : by; }
                else if (coopPop == 2) { destTz = (faceTz + 1) % BLOCK_NZ; destBz = (faceTz == BLOCK_NZ - 1) ? bzp1 : bz; }
                else if (coopPop == 3) { destTy = (faceTy - 1 + BLOCK_NY) % BLOCK_NY; destBy = (faceTy == 0) ? bym1 : by; }
                else if (coopPop == 4) { destTz = (faceTz - 1 + BLOCK_NZ) % BLOCK_NZ; destBz = (faceTz == 0) ? bzm1 : bz; }
            }
        }
        const int destBx = west ? storeBxWest : storeBxEast;
        coopDestinationIndex = idxPopX(destTy, destTz, coopPop, destBx, destBy, destBz);
    } else if (coopFace <= 3) {
        const bool south = coopFace == 2;
        const int faceTx = coopNode % BLOCK_NX;
        const int faceTz = coopNode / BLOCK_NX;
        int destTx = faceTx, destTz = faceTz, destBx = bx, destBz = bz;
        #ifndef BC_Y_PERIODIC
        coopActive = south ? (by != 0) : (by != NUM_BLOCK_Y - 1);
#endif // BC_Y_PERIODIC
        if (popSaveParity) {
            coopDestination = south ? storeGhostYSouth : storeGhostYNorth;
        } else {
            coopDestination = south ? storeGhostYNorth : storeGhostYSouth;
            if (south) {
                if (coopPop == 1) { destTx = (faceTx - 1 + BLOCK_NX) % BLOCK_NX; destBx = (faceTx == 0) ? bxm1 : bx; }
                else if (coopPop == 2) { destTz = (faceTz - 1 + BLOCK_NZ) % BLOCK_NZ; destBz = (faceTz == 0) ? bzm1 : bz; }
                else if (coopPop == 3) { destTx = (faceTx + 1) % BLOCK_NX; destBx = (faceTx == BLOCK_NX - 1) ? bxp1 : bx; }
                else if (coopPop == 4) { destTz = (faceTz + 1) % BLOCK_NZ; destBz = (faceTz == BLOCK_NZ - 1) ? bzp1 : bz; }
            } else {
                if (coopPop == 1) { destTx = (faceTx + 1) % BLOCK_NX; destBx = (faceTx == BLOCK_NX - 1) ? bxp1 : bx; }
                else if (coopPop == 2) { destTz = (faceTz + 1) % BLOCK_NZ; destBz = (faceTz == BLOCK_NZ - 1) ? bzp1 : bz; }
                else if (coopPop == 3) { destTx = (faceTx - 1 + BLOCK_NX) % BLOCK_NX; destBx = (faceTx == 0) ? bxm1 : bx; }
                else if (coopPop == 4) { destTz = (faceTz - 1 + BLOCK_NZ) % BLOCK_NZ; destBz = (faceTz == 0) ? bzm1 : bz; }
            }
        }
        const int destBy = south ? storeBySouth : storeByNorth;
        coopDestinationIndex = idxPopY(destTx, destTz, coopPop, destBx, destBy, destBz);
    } else {
        const bool back = coopFace == 4;
        const int faceTx = coopNode % BLOCK_NX;
        const int faceTy = coopNode / BLOCK_NX;
        int destTx = faceTx, destTy = faceTy, destBx = bx, destBy = by;
        int destBz = back ? storeBzBack : storeBzFront;
        #ifndef BC_Z_PERIODIC
        coopActive = back ? (bz != 0) : (bz != top);
#endif // BC_Z_PERIODIC
        if (popSaveParity) {
            coopDestination = back ? storeGhostZBack : storeGhostZFront;
        } else {
            if (back && bz == 0) {
                coopDestination = ghostInterface.Z_0;
                destBz = 0;
            } else if (!back && bz == top) {
                coopDestination = ghostInterface.Z_1;
                destBz = top;
            } else {
                coopDestination = back ? storeGhostZFront : storeGhostZBack;
            }

            if (back) {
                if (coopPop == 1) { destTx = (faceTx - 1 + BLOCK_NX) % BLOCK_NX; destBx = (faceTx == 0) ? bxm1 : bx; }
                else if (coopPop == 2) { destTy = (faceTy - 1 + BLOCK_NY) % BLOCK_NY; destBy = (faceTy == 0) ? bym1 : by; }
                else if (coopPop == 3) { destTx = (faceTx + 1) % BLOCK_NX; destBx = (faceTx == BLOCK_NX - 1) ? bxp1 : bx; }
                else if (coopPop == 4) { destTy = (faceTy + 1) % BLOCK_NY; destBy = (faceTy == BLOCK_NY - 1) ? byp1 : by; }
            } else {
                if (coopPop == 1) { destTx = (faceTx + 1) % BLOCK_NX; destBx = (faceTx == BLOCK_NX - 1) ? bxp1 : bx; }
                else if (coopPop == 2) { destTy = (faceTy + 1) % BLOCK_NY; destBy = (faceTy == BLOCK_NY - 1) ? byp1 : by; }
                else if (coopPop == 3) { destTx = (faceTx - 1 + BLOCK_NX) % BLOCK_NX; destBx = (faceTx == 0) ? bxm1 : bx; }
                else if (coopPop == 4) { destTy = (faceTy - 1 + BLOCK_NY) % BLOCK_NY; destBy = (faceTy == 0) ? bym1 : by; }
            }
        }
        coopDestinationIndex = idxPopZ(destTx, destTy, coopPop, destBx, destBy, destBz);
    }

    if (coopActive) {
        coopDestination[coopDestinationIndex] = s_pop[coopLinear];
    }
}
