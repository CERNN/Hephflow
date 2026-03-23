{
    // AA scheme: parity selects both face pointer and block index
    // Even step: standard face mapping (West<>X_1, East<>X_0, etc.) with neighbor block index
    // Odd step:  swapped face mapping (West<>X_0, East<>X_1, etc.) with own block index
    const dfloat* loadGhostXWest  = stepParity ? ghostInterface.pop.X_0 : ghostInterface.pop.X_1;
    const dfloat* loadGhostXEast  = stepParity ? ghostInterface.pop.X_1 : ghostInterface.pop.X_0;
    const dfloat* loadGhostYSouth = stepParity ? ghostInterface.pop.Y_0 : ghostInterface.pop.Y_1;
    const dfloat* loadGhostYNorth = stepParity ? ghostInterface.pop.Y_1 : ghostInterface.pop.Y_0;
    const dfloat* loadGhostZBack  = stepParity ? ghostInterface.pop.Z_0 : ghostInterface.pop.Z_1;
    const dfloat* loadGhostZFront = stepParity ? ghostInterface.pop.Z_1 : ghostInterface.pop.Z_0;

    // Parity-aware block indices (even: neighbor, odd: own)
    const int loadBxWest  = (stepParity ? bx : bxm1);
    const int loadBxEast  = (stepParity ? bx : bxp1);
    const int loadBySouth = (stepParity ? by : bym1);
    const int loadByNorth = (stepParity ? by : byp1);
    const int loadBzBack  = (stepParity ? bz : bzm1);
    const int loadBzFront = (stepParity ? bz : bzp1);

    // Linearized thread ID within the block
    const int ghostTid = threadIdx.x + threadIdx.y * BLOCK_NX + threadIdx.z * BLOCK_NX * BLOCK_NY;
    const int ghostBlockSize = BLOCK_NX * BLOCK_NY * BLOCK_NZ;

    // --- Load West face ---
    for (int i = ghostTid; i < GHOST_FACE_SIZE_X; i += ghostBlockSize) {
        const int pop_i = i / (BLOCK_NY * BLOCK_NZ);
        const int rem   = i % (BLOCK_NY * BLOCK_NZ);
        const int tz_i  = rem / BLOCK_NY;
        const int ty_i  = rem % BLOCK_NY;
        s_pop[GHOST_OFF_W + i] = loadGhostXWest[idxPopX(ty_i, tz_i, pop_i, loadBxWest, by, bz)];
    }

    // --- Load East face ---
    for (int i = ghostTid; i < GHOST_FACE_SIZE_X; i += ghostBlockSize) {
        const int pop_i = i / (BLOCK_NY * BLOCK_NZ);
        const int rem   = i % (BLOCK_NY * BLOCK_NZ);
        const int tz_i  = rem / BLOCK_NY;
        const int ty_i  = rem % BLOCK_NY;
        s_pop[GHOST_OFF_E + i] = loadGhostXEast[idxPopX(ty_i, tz_i, pop_i, loadBxEast, by, bz)];
    }

    // --- Load South face ---
    for (int i = ghostTid; i < GHOST_FACE_SIZE_Y; i += ghostBlockSize) {
        const int pop_i = i / (BLOCK_NX * BLOCK_NZ);
        const int rem   = i % (BLOCK_NX * BLOCK_NZ);
        const int tz_i  = rem / BLOCK_NX;
        const int tx_i  = rem % BLOCK_NX;
        s_pop[GHOST_OFF_S + i] = loadGhostYSouth[idxPopY(tx_i, tz_i, pop_i, bx, loadBySouth, bz)];
    }

    // --- Load North face ---
    for (int i = ghostTid; i < GHOST_FACE_SIZE_Y; i += ghostBlockSize) {
        const int pop_i = i / (BLOCK_NX * BLOCK_NZ);
        const int rem   = i % (BLOCK_NX * BLOCK_NZ);
        const int tz_i  = rem / BLOCK_NX;
        const int tx_i  = rem % BLOCK_NX;
        s_pop[GHOST_OFF_N + i] = loadGhostYNorth[idxPopY(tx_i, tz_i, pop_i, bx, loadByNorth, bz)];
    }

    // --- Load Back face ---
    for (int i = ghostTid; i < GHOST_FACE_SIZE_Z; i += ghostBlockSize) {
        const int pop_i = i / (BLOCK_NX * BLOCK_NY);
        const int rem   = i % (BLOCK_NX * BLOCK_NY);
        const int ty_i  = rem / BLOCK_NX;
        const int tx_i  = rem % BLOCK_NX;
        s_pop[GHOST_OFF_B + i] = loadGhostZBack[idxPopZ(tx_i, ty_i, pop_i, bx, by, loadBzBack)];
    }

    // --- Load Front face ---
    for (int i = ghostTid; i < GHOST_FACE_SIZE_Z; i += ghostBlockSize) {
        const int pop_i = i / (BLOCK_NX * BLOCK_NY);
        const int rem   = i % (BLOCK_NX * BLOCK_NY);
        const int ty_i  = rem / BLOCK_NX;
        const int tx_i  = rem % BLOCK_NX;
        s_pop[GHOST_OFF_F + i] = loadGhostZFront[idxPopZ(tx_i, ty_i, pop_i, bx, by, loadBzFront)];
    }
}
