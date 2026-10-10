// D3Q19 cooperative load for the direction set selected by
// COOPERATIVE_GHOST_LOAD_MASK. Selected faces share one barrier pair.
activeBarrier.arrive_and_wait();

constexpr int coopAllFaceNodes = BLOCK_NY * BLOCK_NZ;
static_assert(BLOCK_NX * BLOCK_NZ == coopAllFaceNodes &&
              BLOCK_NX * BLOCK_NY == coopAllFaceNodes,
              "Cooperative multi-direction load requires equal face areas");
constexpr int coopAllPopCount = 5;
constexpr int coopAllFaceValues = coopAllFaceNodes * coopAllPopCount;
constexpr int coopAllDirectionCount =
    ((COOPERATIVE_GHOST_LOAD_MASK & 1) ? 1 : 0) +
    ((COOPERATIVE_GHOST_LOAD_MASK & 2) ? 1 : 0) +
    ((COOPERATIVE_GHOST_LOAD_MASK & 4) ? 1 : 0);
constexpr int coopAllValues = 2 * coopAllDirectionCount * coopAllFaceValues;
constexpr int coopAllThreads = BLOCK_NX * BLOCK_NY * BLOCK_NZ;
constexpr int coopXOffset = 0;
constexpr int coopYOffset = ((COOPERATIVE_GHOST_LOAD_MASK & 1) ? 2 : 0) * coopAllFaceValues;
constexpr int coopZOffset = (((COOPERATIVE_GHOST_LOAD_MASK & 1) ? 2 : 0) +
                             ((COOPERATIVE_GHOST_LOAD_MASK & 2) ? 2 : 0)) * coopAllFaceValues;
static_assert(coopAllValues <= BLOCK_LBM_SIZE_POP,
    "The shared population tile is too small for cooperative ghost load");

const int coopAllTid = tx + BLOCK_NX * (ty + BLOCK_NY * tz);
for (int coopLinear = coopAllTid;
     coopLinear < coopAllValues;
     coopLinear += coopAllThreads) {
    const int coopCompactFace = coopLinear / coopAllFaceValues;
    int coopFace = coopCompactFace;
    if (!(COOPERATIVE_GHOST_LOAD_MASK & 1)) coopFace += 2;
    if (coopFace >= 2 && !(COOPERATIVE_GHOST_LOAD_MASK & 2)) coopFace += 2;
    const int coopInFace = coopLinear - coopCompactFace * coopAllFaceValues;
    const int coopPop = coopInFace / coopAllFaceNodes;
    const int coopNode = coopInFace - coopPop * coopAllFaceNodes;

    const dfloat* coopSource = nullptr;
    int coopSourceIndex = 0;

    if (coopFace <= 1) {
        const bool west = coopFace == 0;
        const int faceTy = coopNode % BLOCK_NY;
        const int faceTz = coopNode / BLOCK_NY;
        int sourceTy = faceTy;
        int sourceTz = faceTz;
        int sourceBy = by;
        int sourceBz = bz;

        if (popLoadParity) {
            coopSource = west ? loadGhostXEast : loadGhostXWest;
        } else {
            coopSource = west ? loadGhostXWest : loadGhostXEast;
            if (west) {
                if (coopPop == 1) { sourceTy = (faceTy - 1 + BLOCK_NY) % BLOCK_NY; sourceBy = (faceTy == 0) ? bym1 : by; }
                else if (coopPop == 2) { sourceTz = (faceTz - 1 + BLOCK_NZ) % BLOCK_NZ; sourceBz = (faceTz == 0) ? bzm1 : bz; }
                else if (coopPop == 3) { sourceTy = (faceTy + 1) % BLOCK_NY; sourceBy = (faceTy == BLOCK_NY - 1) ? byp1 : by; }
                else if (coopPop == 4) { sourceTz = (faceTz + 1) % BLOCK_NZ; sourceBz = (faceTz == BLOCK_NZ - 1) ? bzp1 : bz; }
            } else {
                if (coopPop == 1) { sourceTy = (faceTy + 1) % BLOCK_NY; sourceBy = (faceTy == BLOCK_NY - 1) ? byp1 : by; }
                else if (coopPop == 2) { sourceTz = (faceTz + 1) % BLOCK_NZ; sourceBz = (faceTz == BLOCK_NZ - 1) ? bzp1 : bz; }
                else if (coopPop == 3) { sourceTy = (faceTy - 1 + BLOCK_NY) % BLOCK_NY; sourceBy = (faceTy == 0) ? bym1 : by; }
                else if (coopPop == 4) { sourceTz = (faceTz - 1 + BLOCK_NZ) % BLOCK_NZ; sourceBz = (faceTz == 0) ? bzm1 : bz; }
            }
        }
        const int sourceBx = west ? loadBxWest : loadBxEast;
        coopSourceIndex = idxPopX(sourceTy, sourceTz, coopPop,
            sourceBx, sourceBy, sourceBz);
    } else if (coopFace <= 3) {
        const bool south = coopFace == 2;
        const int faceTx = coopNode % BLOCK_NX;
        const int faceTz = coopNode / BLOCK_NX;
        int sourceTx = faceTx;
        int sourceTz = faceTz;
        int sourceBx = bx;
        int sourceBz = bz;

        if (popLoadParity) {
            coopSource = south ? loadGhostYNorth : loadGhostYSouth;
        } else {
            coopSource = south ? loadGhostYSouth : loadGhostYNorth;
            if (south) {
                if (coopPop == 1) { sourceTx = (faceTx - 1 + BLOCK_NX) % BLOCK_NX; sourceBx = (faceTx == 0) ? bxm1 : bx; }
                else if (coopPop == 2) { sourceTz = (faceTz - 1 + BLOCK_NZ) % BLOCK_NZ; sourceBz = (faceTz == 0) ? bzm1 : bz; }
                else if (coopPop == 3) { sourceTx = (faceTx + 1) % BLOCK_NX; sourceBx = (faceTx == BLOCK_NX - 1) ? bxp1 : bx; }
                else if (coopPop == 4) { sourceTz = (faceTz + 1) % BLOCK_NZ; sourceBz = (faceTz == BLOCK_NZ - 1) ? bzp1 : bz; }
            } else {
                if (coopPop == 1) { sourceTx = (faceTx + 1) % BLOCK_NX; sourceBx = (faceTx == BLOCK_NX - 1) ? bxp1 : bx; }
                else if (coopPop == 2) { sourceTz = (faceTz + 1) % BLOCK_NZ; sourceBz = (faceTz == BLOCK_NZ - 1) ? bzp1 : bz; }
                else if (coopPop == 3) { sourceTx = (faceTx - 1 + BLOCK_NX) % BLOCK_NX; sourceBx = (faceTx == 0) ? bxm1 : bx; }
                else if (coopPop == 4) { sourceTz = (faceTz - 1 + BLOCK_NZ) % BLOCK_NZ; sourceBz = (faceTz == 0) ? bzm1 : bz; }
            }
        }
        const int sourceBy = south ? loadBySouth : loadByNorth;
        coopSourceIndex = idxPopY(sourceTx, sourceTz, coopPop,
            sourceBx, sourceBy, sourceBz);
    } else {
        const bool back = coopFace == 4;
        const int faceTx = coopNode % BLOCK_NX;
        const int faceTy = coopNode / BLOCK_NX;
        int sourceTx = faceTx;
        int sourceTy = faceTy;
        int sourceBx = bx;
        int sourceBy = by;
        int sourceBz = back ? loadBzBack : loadBzFront;

        if (popLoadParity) {
            if (back && bz == 0) {
                coopSource = ghostInterface.auxZ_1;
                sourceBz = 0;
            } else if (!back && bz == top) {
                coopSource = ghostInterface.auxZ_0;
                sourceBz = 0;
            } else {
                coopSource = back ? loadGhostZFront : loadGhostZBack;
            }
        } else {
            if (back && bz == 0) {
                coopSource = ghostInterfaceCopy.auxZ_1;
                sourceBz = 0;
            } else if (!back && bz == top) {
                coopSource = ghostInterfaceCopy.auxZ_0;
                sourceBz = 0;
            } else {
                coopSource = back ? loadGhostZBack : loadGhostZFront;
            }

            if (back) {
                if (coopPop == 1) { sourceTx = (faceTx - 1 + BLOCK_NX) % BLOCK_NX; sourceBx = (faceTx == 0) ? bxm1 : bx; }
                else if (coopPop == 2) { sourceTy = (faceTy - 1 + BLOCK_NY) % BLOCK_NY; sourceBy = (faceTy == 0) ? bym1 : by; }
                else if (coopPop == 3) { sourceTx = (faceTx + 1) % BLOCK_NX; sourceBx = (faceTx == BLOCK_NX - 1) ? bxp1 : bx; }
                else if (coopPop == 4) { sourceTy = (faceTy + 1) % BLOCK_NY; sourceBy = (faceTy == BLOCK_NY - 1) ? byp1 : by; }
            } else {
                if (coopPop == 1) { sourceTx = (faceTx + 1) % BLOCK_NX; sourceBx = (faceTx == BLOCK_NX - 1) ? bxp1 : bx; }
                else if (coopPop == 2) { sourceTy = (faceTy + 1) % BLOCK_NY; sourceBy = (faceTy == BLOCK_NY - 1) ? byp1 : by; }
                else if (coopPop == 3) { sourceTx = (faceTx - 1 + BLOCK_NX) % BLOCK_NX; sourceBx = (faceTx == 0) ? bxm1 : bx; }
                else if (coopPop == 4) { sourceTy = (faceTy - 1 + BLOCK_NY) % BLOCK_NY; sourceBy = (faceTy == 0) ? bym1 : by; }
            }
        }
        coopSourceIndex = idxPopZ(sourceTx, sourceTy, coopPop,
            sourceBx, sourceBy, sourceBz);
    }

    s_pop[coopLinear] = coopSource[coopSourceIndex];
}

activeBarrier.arrive_and_wait();

#if COOPERATIVE_GHOST_LOAD_MASK & 1
const int coopXNode = ty + BLOCK_NY * tz;
if (tx == 0) {
    pop[1] = s_pop[coopXOffset + 0 * coopAllFaceValues + 0 * coopAllFaceNodes + coopXNode];
    pop[7] = s_pop[coopXOffset + 0 * coopAllFaceValues + 1 * coopAllFaceNodes + coopXNode];
    pop[9] = s_pop[coopXOffset + 0 * coopAllFaceValues + 2 * coopAllFaceNodes + coopXNode];
    pop[13] = s_pop[coopXOffset + 0 * coopAllFaceValues + 3 * coopAllFaceNodes + coopXNode];
    pop[15] = s_pop[coopXOffset + 0 * coopAllFaceValues + 4 * coopAllFaceNodes + coopXNode];
} else if (tx == BLOCK_NX - 1) {
    pop[2] = s_pop[coopXOffset + 1 * coopAllFaceValues + 0 * coopAllFaceNodes + coopXNode];
    pop[8] = s_pop[coopXOffset + 1 * coopAllFaceValues + 1 * coopAllFaceNodes + coopXNode];
    pop[10] = s_pop[coopXOffset + 1 * coopAllFaceValues + 2 * coopAllFaceNodes + coopXNode];
    pop[14] = s_pop[coopXOffset + 1 * coopAllFaceValues + 3 * coopAllFaceNodes + coopXNode];
    pop[16] = s_pop[coopXOffset + 1 * coopAllFaceValues + 4 * coopAllFaceNodes + coopXNode];
}
#endif // COOPERATIVE_GHOST_LOAD_MASK & 1

#if COOPERATIVE_GHOST_LOAD_MASK & 2
const int coopYNode = tx + BLOCK_NX * tz;
if (ty == 0) {
    pop[3] = s_pop[coopYOffset + 0 * coopAllFaceValues + 0 * coopAllFaceNodes + coopYNode];
    pop[7] = s_pop[coopYOffset + 0 * coopAllFaceValues + 1 * coopAllFaceNodes + coopYNode];
    pop[11] = s_pop[coopYOffset + 0 * coopAllFaceValues + 2 * coopAllFaceNodes + coopYNode];
    pop[14] = s_pop[coopYOffset + 0 * coopAllFaceValues + 3 * coopAllFaceNodes + coopYNode];
    pop[17] = s_pop[coopYOffset + 0 * coopAllFaceValues + 4 * coopAllFaceNodes + coopYNode];
} else if (ty == BLOCK_NY - 1) {
    pop[4] = s_pop[coopYOffset + 1 * coopAllFaceValues + 0 * coopAllFaceNodes + coopYNode];
    pop[8] = s_pop[coopYOffset + 1 * coopAllFaceValues + 1 * coopAllFaceNodes + coopYNode];
    pop[12] = s_pop[coopYOffset + 1 * coopAllFaceValues + 2 * coopAllFaceNodes + coopYNode];
    pop[13] = s_pop[coopYOffset + 1 * coopAllFaceValues + 3 * coopAllFaceNodes + coopYNode];
    pop[18] = s_pop[coopYOffset + 1 * coopAllFaceValues + 4 * coopAllFaceNodes + coopYNode];
}
#endif // COOPERATIVE_GHOST_LOAD_MASK & 2

#if COOPERATIVE_GHOST_LOAD_MASK & 4
const int coopZNode = tx + BLOCK_NX * ty;
if (tz == 0) {
    pop[5] = s_pop[coopZOffset + 0 * coopAllFaceValues + 0 * coopAllFaceNodes + coopZNode];
    pop[9] = s_pop[coopZOffset + 0 * coopAllFaceValues + 1 * coopAllFaceNodes + coopZNode];
    pop[11] = s_pop[coopZOffset + 0 * coopAllFaceValues + 2 * coopAllFaceNodes + coopZNode];
    pop[16] = s_pop[coopZOffset + 0 * coopAllFaceValues + 3 * coopAllFaceNodes + coopZNode];
    pop[18] = s_pop[coopZOffset + 0 * coopAllFaceValues + 4 * coopAllFaceNodes + coopZNode];
} else if (tz == BLOCK_NZ - 1) {
    pop[6] = s_pop[coopZOffset + 1 * coopAllFaceValues + 0 * coopAllFaceNodes + coopZNode];
    pop[10] = s_pop[coopZOffset + 1 * coopAllFaceValues + 1 * coopAllFaceNodes + coopZNode];
    pop[12] = s_pop[coopZOffset + 1 * coopAllFaceValues + 2 * coopAllFaceNodes + coopZNode];
    pop[15] = s_pop[coopZOffset + 1 * coopAllFaceValues + 3 * coopAllFaceNodes + coopZNode];
    pop[17] = s_pop[coopZOffset + 1 * coopAllFaceValues + 4 * coopAllFaceNodes + coopZNode];
}
#endif // COOPERATIVE_GHOST_LOAD_MASK & 4
