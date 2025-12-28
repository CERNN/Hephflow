#if (BLOCK_NX == 2 && BLOCK_NY == 2 && BLOCK_NZ == 2)
    __shared__ dfloat sdata[8];
#elif (BLOCK_NX == 2 && BLOCK_NY == 2 && BLOCK_NZ == 4)
    __shared__ dfloat sdata[16];
#elif (BLOCK_NX == 2 && BLOCK_NY == 2 && BLOCK_NZ == 8)
    __shared__ dfloat sdata[32];
#elif (BLOCK_NX == 2 && BLOCK_NY == 4 && BLOCK_NZ == 2)
    __shared__ dfloat sdata[16];
#elif (BLOCK_NX == 2 && BLOCK_NY == 4 && BLOCK_NZ == 4)
    __shared__ dfloat sdata[32];
#elif (BLOCK_NX == 2 && BLOCK_NY == 4 && BLOCK_NZ == 8)
    __shared__ dfloat sdata[64];
#elif (BLOCK_NX == 2 && BLOCK_NY == 8 && BLOCK_NZ == 2)
    __shared__ dfloat sdata[32];
#elif (BLOCK_NX == 2 && BLOCK_NY == 8 && BLOCK_NZ == 4)
    __shared__ dfloat sdata[64];
#elif (BLOCK_NX == 2 && BLOCK_NY == 8 && BLOCK_NZ == 8)
    __shared__ dfloat sdata[128];
#elif (BLOCK_NX == 4 && BLOCK_NY == 2 && BLOCK_NZ == 2)
    __shared__ dfloat sdata[16];
#elif (BLOCK_NX == 4 && BLOCK_NY == 2 && BLOCK_NZ == 4)
    __shared__ dfloat sdata[32];
#elif (BLOCK_NX == 4 && BLOCK_NY == 2 && BLOCK_NZ == 8)
    __shared__ dfloat sdata[64];
#elif (BLOCK_NX == 4 && BLOCK_NY == 4 && BLOCK_NZ == 2)
    __shared__ dfloat sdata[32];
#elif (BLOCK_NX == 4 && BLOCK_NY == 4 && BLOCK_NZ == 4)
    __shared__ dfloat sdata[64];
#elif (BLOCK_NX == 4 && BLOCK_NY == 4 && BLOCK_NZ == 8)
    __shared__ dfloat sdata[128];
#elif (BLOCK_NX == 4 && BLOCK_NY == 8 && BLOCK_NZ == 2)
    __shared__ dfloat sdata[64];
#elif (BLOCK_NX == 4 && BLOCK_NY == 8 && BLOCK_NZ == 4)
    __shared__ dfloat sdata[128];
#elif (BLOCK_NX == 4 && BLOCK_NY == 8 && BLOCK_NZ == 8)
    __shared__ dfloat sdata[256];
#elif (BLOCK_NX == 8 && BLOCK_NY == 2 && BLOCK_NZ == 2)
    __shared__ dfloat sdata[32];
#elif (BLOCK_NX == 8 && BLOCK_NY == 2 && BLOCK_NZ == 4)
    __shared__ dfloat sdata[64];
#elif (BLOCK_NX == 8 && BLOCK_NY == 2 && BLOCK_NZ == 8)
    __shared__ dfloat sdata[128];
#elif (BLOCK_NX == 8 && BLOCK_NY == 4 && BLOCK_NZ == 2)
    __shared__ dfloat sdata[64];
#elif (BLOCK_NX == 8 && BLOCK_NY == 4 && BLOCK_NZ == 4)
    __shared__ dfloat sdata[128];
#elif (BLOCK_NX == 8 && BLOCK_NY == 4 && BLOCK_NZ == 8)
    __shared__ dfloat sdata[256];
#elif (BLOCK_NX == 8 && BLOCK_NY == 8 && BLOCK_NZ == 2)
    __shared__ dfloat sdata[128];
#elif (BLOCK_NX == 8 && BLOCK_NY == 8 && BLOCK_NZ == 4)
    __shared__ dfloat sdata[256];
#elif (BLOCK_NX == 8 && BLOCK_NY == 8 && BLOCK_NZ == 8)
    __shared__ dfloat sdata[512];
#elif (BLOCK_NX == 16 && BLOCK_NY == 8 && BLOCK_NZ == 8)
    __shared__ dfloat sdata[1024];
#elif (BLOCK_NX == 8 && BLOCK_NY == 16 && BLOCK_NZ == 8)
    __shared__ dfloat sdata[1024];
#elif (BLOCK_NX == 8 && BLOCK_NY == 8 && BLOCK_NZ == 16)
    __shared__ dfloat sdata[1024];
#elif (BLOCK_NX == 16 && BLOCK_NY == 16 && BLOCK_NZ == 4)
    __shared__ dfloat sdata[1024];
#elif (BLOCK_NX == 16 && BLOCK_NY == 4 && BLOCK_NZ == 16)
    __shared__ dfloat sdata[1024];
#elif (BLOCK_NX == 4 && BLOCK_NY == 16 && BLOCK_NZ == 16)
    __shared__ dfloat sdata[1024];
#elif (BLOCK_NX == 32 && BLOCK_NY == 4 && BLOCK_NZ == 4)
    __shared__ dfloat sdata[512];
#elif (BLOCK_NX == 4 && BLOCK_NY == 32 && BLOCK_NZ == 4)
    __shared__ dfloat sdata[512];
#elif (BLOCK_NX == 4 && BLOCK_NY == 4 && BLOCK_NZ == 32)
    __shared__ dfloat sdata[512];
#else
    // Fallback: compute block size dynamically (requires dynamic shared memory launch parameter)
    __shared__ dfloat sdata[BLOCK_LBM_SIZE];
#endif