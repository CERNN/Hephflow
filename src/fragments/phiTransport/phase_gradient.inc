// Retrieve precomputed gradients from gpuComputePhaseNormals
dfloat dphidx = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M3_NX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
dfloat dphidy = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M3_NY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
dfloat dphidz = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M3_NZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];

dfloat mod_grad = sqrtf(dphidx*dphidx + dphidy*dphidy + dphidz*dphidz);

constexpr dfloat GRAD_NORMAL_THRESHOLD = 1e-4_df;
dfloat pnx = 0.0_df;
dfloat pny = 0.0_df;
dfloat pnz = 0.0_df;
if (mod_grad > GRAD_NORMAL_THRESHOLD) {
    dfloat inv_mod_grad = 1.0_df / mod_grad;
    //normal
    pnx = dphidx * inv_mod_grad;
    pny = dphidy * inv_mod_grad;
    pnz = dphidz * inv_mod_grad;
}



