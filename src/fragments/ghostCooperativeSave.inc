{
    // AA scheme: cooperative save only runs on odd steps.
    // On even steps, popSave writes directly to global with shifted indices.
    if (stepParity) {

    // Odd step: write to own face buffers at own block index (no shift)
    dfloat* storeGhostXWest  = ghostInterface.pop.X_0;
    dfloat* storeGhostXEast  = ghostInterface.pop.X_1;
    dfloat* storeGhostYSouth = ghostInterface.pop.Y_0;
    dfloat* storeGhostYNorth = ghostInterface.pop.Y_1;
    dfloat* storeGhostZBack  = ghostInterface.pop.Z_0;
    dfloat* storeGhostZFront = ghostInterface.pop.Z_1;

    // Odd step: own block indices (no neighbor writes)
    const int storeBxWest  = bx;
    const int storeBxEast  = bx;
    const int storeBySouth = by;
    const int storeByNorth = by;
    const int storeBzBack  = bz;
    const int storeBzFront = bz;

    // Linearized thread ID within the block
    const int saveTid = threadIdx.x + threadIdx.y * BLOCK_NX + threadIdx.z * BLOCK_NX * BLOCK_NY;
    const int saveBlockSize = BLOCK_NX * BLOCK_NY * BLOCK_NZ;

    // --- Write West face ---
    if (GHOST_WRITE_WEST) {
        for (int i = saveTid; i < GHOST_FACE_SIZE_X; i += saveBlockSize) {
            const int pop_i = i / (BLOCK_NY * BLOCK_NZ);
            const int rem   = i % (BLOCK_NY * BLOCK_NZ);
            const int tz_i  = rem / BLOCK_NY;
            const int ty_i  = rem % BLOCK_NY;
            storeGhostXWest[idxPopX(ty_i, tz_i, pop_i, storeBxWest, by, bz)] = s_pop[GHOST_OFF_W + i];
        }
    }

    // --- Write East face ---
    if (GHOST_WRITE_EAST) {
        for (int i = saveTid; i < GHOST_FACE_SIZE_X; i += saveBlockSize) {
            const int pop_i = i / (BLOCK_NY * BLOCK_NZ);
            const int rem   = i % (BLOCK_NY * BLOCK_NZ);
            const int tz_i  = rem / BLOCK_NY;
            const int ty_i  = rem % BLOCK_NY;
            storeGhostXEast[idxPopX(ty_i, tz_i, pop_i, storeBxEast, by, bz)] = s_pop[GHOST_OFF_E + i];
        }
    }

    // --- Write South face ---
    if (GHOST_WRITE_SOUTH) {
        for (int i = saveTid; i < GHOST_FACE_SIZE_Y; i += saveBlockSize) {
            const int pop_i = i / (BLOCK_NX * BLOCK_NZ);
            const int rem   = i % (BLOCK_NX * BLOCK_NZ);
            const int tz_i  = rem / BLOCK_NX;
            const int tx_i  = rem % BLOCK_NX;
            storeGhostYSouth[idxPopY(tx_i, tz_i, pop_i, bx, storeBySouth, bz)] = s_pop[GHOST_OFF_S + i];
        }
    }

    // --- Write North face ---
    if (GHOST_WRITE_NORTH) {
        for (int i = saveTid; i < GHOST_FACE_SIZE_Y; i += saveBlockSize) {
            const int pop_i = i / (BLOCK_NX * BLOCK_NZ);
            const int rem   = i % (BLOCK_NX * BLOCK_NZ);
            const int tz_i  = rem / BLOCK_NX;
            const int tx_i  = rem % BLOCK_NX;
            storeGhostYNorth[idxPopY(tx_i, tz_i, pop_i, bx, storeByNorth, bz)] = s_pop[GHOST_OFF_N + i];
        }
    }

    // --- Write Back face ---
    if (GHOST_WRITE_BACK) {
        for (int i = saveTid; i < GHOST_FACE_SIZE_Z; i += saveBlockSize) {
            const int pop_i = i / (BLOCK_NX * BLOCK_NY);
            const int rem   = i % (BLOCK_NX * BLOCK_NY);
            const int ty_i  = rem / BLOCK_NX;
            const int tx_i  = rem % BLOCK_NX;
            storeGhostZBack[idxPopZ(tx_i, ty_i, pop_i, bx, by, storeBzBack)] = s_pop[GHOST_OFF_B + i];
        }
    }

    // --- Write Front face ---
    if (GHOST_WRITE_FRONT) {
        for (int i = saveTid; i < GHOST_FACE_SIZE_Z; i += saveBlockSize) {
            const int pop_i = i / (BLOCK_NX * BLOCK_NY);
            const int rem   = i % (BLOCK_NX * BLOCK_NY);
            const int ty_i  = rem / BLOCK_NX;
            const int tx_i  = rem % BLOCK_NX;
            storeGhostZFront[idxPopZ(tx_i, ty_i, pop_i, bx, by, storeBzFront)] = s_pop[GHOST_OFF_F + i];
        }
    }

    } // end if (stepParity)
}
