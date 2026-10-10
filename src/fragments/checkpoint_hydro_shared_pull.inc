// Reproduce the collision kernel's in-block pull before face populations are
// overridden from global storage.  Canonical checkpoints must contain the
// logical incoming population at every physical node, including nodes that
// are interiors in the source decomposition but faces after repartitioning.
s_pop[idxPopBlock(tx, ty, tz,  0)] = pop[ 1];
s_pop[idxPopBlock(tx, ty, tz,  1)] = pop[ 2];
s_pop[idxPopBlock(tx, ty, tz,  2)] = pop[ 3];
s_pop[idxPopBlock(tx, ty, tz,  3)] = pop[ 4];
s_pop[idxPopBlock(tx, ty, tz,  4)] = pop[ 5];
s_pop[idxPopBlock(tx, ty, tz,  5)] = pop[ 6];
s_pop[idxPopBlock(tx, ty, tz,  6)] = pop[ 7];
s_pop[idxPopBlock(tx, ty, tz,  7)] = pop[ 8];
s_pop[idxPopBlock(tx, ty, tz,  8)] = pop[ 9];
s_pop[idxPopBlock(tx, ty, tz,  9)] = pop[10];
s_pop[idxPopBlock(tx, ty, tz, 10)] = pop[11];
s_pop[idxPopBlock(tx, ty, tz, 11)] = pop[12];
s_pop[idxPopBlock(tx, ty, tz, 12)] = pop[13];
s_pop[idxPopBlock(tx, ty, tz, 13)] = pop[14];
s_pop[idxPopBlock(tx, ty, tz, 14)] = pop[15];
s_pop[idxPopBlock(tx, ty, tz, 15)] = pop[16];
s_pop[idxPopBlock(tx, ty, tz, 16)] = pop[17];
s_pop[idxPopBlock(tx, ty, tz, 17)] = pop[18];
#ifdef D3Q27
s_pop[idxPopBlock(tx, ty, tz, 18)] = pop[19];
s_pop[idxPopBlock(tx, ty, tz, 19)] = pop[20];
s_pop[idxPopBlock(tx, ty, tz, 20)] = pop[21];
s_pop[idxPopBlock(tx, ty, tz, 21)] = pop[22];
s_pop[idxPopBlock(tx, ty, tz, 22)] = pop[23];
s_pop[idxPopBlock(tx, ty, tz, 23)] = pop[24];
s_pop[idxPopBlock(tx, ty, tz, 24)] = pop[25];
s_pop[idxPopBlock(tx, ty, tz, 25)] = pop[26];
#endif // D3Q27

activeBarrier.arrive_and_wait();

pop[ 1] = s_pop[idxPopBlock((tx - 1 + BLOCK_NX) % BLOCK_NX, ty, tz, 0)];
pop[ 2] = s_pop[idxPopBlock((tx + 1) % BLOCK_NX, ty, tz, 1)];
pop[ 3] = s_pop[idxPopBlock(tx, (ty - 1 + BLOCK_NY) % BLOCK_NY, tz, 2)];
pop[ 4] = s_pop[idxPopBlock(tx, (ty + 1) % BLOCK_NY, tz, 3)];
pop[ 5] = s_pop[idxPopBlock(tx, ty, (tz - 1 + BLOCK_NZ) % BLOCK_NZ, 4)];
pop[ 6] = s_pop[idxPopBlock(tx, ty, (tz + 1) % BLOCK_NZ, 5)];
pop[ 7] = s_pop[idxPopBlock((tx - 1 + BLOCK_NX) % BLOCK_NX, (ty - 1 + BLOCK_NY) % BLOCK_NY, tz, 6)];
pop[ 8] = s_pop[idxPopBlock((tx + 1) % BLOCK_NX, (ty + 1) % BLOCK_NY, tz, 7)];
pop[ 9] = s_pop[idxPopBlock((tx - 1 + BLOCK_NX) % BLOCK_NX, ty, (tz - 1 + BLOCK_NZ) % BLOCK_NZ, 8)];
pop[10] = s_pop[idxPopBlock((tx + 1) % BLOCK_NX, ty, (tz + 1) % BLOCK_NZ, 9)];
pop[11] = s_pop[idxPopBlock(tx, (ty - 1 + BLOCK_NY) % BLOCK_NY, (tz - 1 + BLOCK_NZ) % BLOCK_NZ, 10)];
pop[12] = s_pop[idxPopBlock(tx, (ty + 1) % BLOCK_NY, (tz + 1) % BLOCK_NZ, 11)];
pop[13] = s_pop[idxPopBlock((tx - 1 + BLOCK_NX) % BLOCK_NX, (ty + 1) % BLOCK_NY, tz, 12)];
pop[14] = s_pop[idxPopBlock((tx + 1) % BLOCK_NX, (ty - 1 + BLOCK_NY) % BLOCK_NY, tz, 13)];
pop[15] = s_pop[idxPopBlock((tx - 1 + BLOCK_NX) % BLOCK_NX, ty, (tz + 1) % BLOCK_NZ, 14)];
pop[16] = s_pop[idxPopBlock((tx + 1) % BLOCK_NX, ty, (tz - 1 + BLOCK_NZ) % BLOCK_NZ, 15)];
pop[17] = s_pop[idxPopBlock(tx, (ty - 1 + BLOCK_NY) % BLOCK_NY, (tz + 1) % BLOCK_NZ, 16)];
pop[18] = s_pop[idxPopBlock(tx, (ty + 1) % BLOCK_NY, (tz - 1 + BLOCK_NZ) % BLOCK_NZ, 17)];
#ifdef D3Q27
pop[19] = s_pop[idxPopBlock((tx - 1 + BLOCK_NX) % BLOCK_NX, (ty - 1 + BLOCK_NY) % BLOCK_NY, (tz - 1 + BLOCK_NZ) % BLOCK_NZ, 18)];
pop[20] = s_pop[idxPopBlock((tx + 1) % BLOCK_NX, (ty + 1) % BLOCK_NY, (tz + 1) % BLOCK_NZ, 19)];
pop[21] = s_pop[idxPopBlock((tx - 1 + BLOCK_NX) % BLOCK_NX, (ty - 1 + BLOCK_NY) % BLOCK_NY, (tz + 1) % BLOCK_NZ, 20)];
pop[22] = s_pop[idxPopBlock((tx + 1) % BLOCK_NX, (ty + 1) % BLOCK_NY, (tz - 1 + BLOCK_NZ) % BLOCK_NZ, 21)];
pop[23] = s_pop[idxPopBlock((tx - 1 + BLOCK_NX) % BLOCK_NX, (ty + 1) % BLOCK_NY, (tz - 1 + BLOCK_NZ) % BLOCK_NZ, 22)];
pop[24] = s_pop[idxPopBlock((tx + 1) % BLOCK_NX, (ty - 1 + BLOCK_NY) % BLOCK_NY, (tz + 1) % BLOCK_NZ, 23)];
pop[25] = s_pop[idxPopBlock((tx + 1) % BLOCK_NX, (ty - 1 + BLOCK_NY) % BLOCK_NY, (tz - 1 + BLOCK_NZ) % BLOCK_NZ, 24)];
pop[26] = s_pop[idxPopBlock((tx - 1 + BLOCK_NX) % BLOCK_NX, (ty + 1) % BLOCK_NY, (tz + 1) % BLOCK_NZ, 25)];
#endif // D3Q27
