// 1. Retrieve precomputed gradients (Isotropic)
dfloat dphidx = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M3_NX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
dfloat dphidy = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M3_NY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
dfloat dphidz = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M3_NZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
dfloat laplacian_phi = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M3_LP_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];

#ifdef PHASE_FORCE_CSF
dfloat div_normal = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z,
    M3_DIV_NORMAL_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
#endif
