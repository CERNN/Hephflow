// 1. Retrieve precomputed gradients (Isotropic)
dfloat dphidx = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M3_NX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
dfloat dphidy = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M3_NY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
dfloat dphidz = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M3_NZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
dfloat laplacian_phi = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M3_LP_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];

dfloat grad2 = dphidx*dphidx + dphidy*dphidy + dphidz*dphidz;
dfloat mod_grad = sqrtf(grad2 + 1e-12_df); 

constexpr dfloat GRAD_NORMAL_THRESHOLD = 1e-8_df; 
dfloat pnx = 0.0_df;
dfloat pny = 0.0_df;
dfloat pnz = 0.0_df;

if (mod_grad > GRAD_NORMAL_THRESHOLD) {
    dfloat inv_mod_grad = 1.0_df / mod_grad;
    pnx = dphidx * inv_mod_grad;
    pny = dphidy * inv_mod_grad;
    pnz = dphidz * inv_mod_grad;
}


dfloat curvature = 0.0_df;
// Only compute curvature near the interface
if (mod_grad > 1e-6_df) {
    dfloat inv_mod_grad = 1.0_df / mod_grad;
    curvature = laplacian_phi * inv_mod_grad; 
}