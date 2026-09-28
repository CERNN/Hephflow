// ---------------------------------------------------------
// FILE: phi_sharpening.inc
// ---------------------------------------------------------

dfloat phiSharpSource = 0.0_df;

#if defined(PHASE_TRANSPORT_CAHN_HILLIARD)
    const dfloat laplacian_mu = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M3_LAPLACIAN_MU_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
    phiSharpSource = M_CH * laplacian_mu - PHI_LBM_DIFF * laplacian_phi;
#elif defined(PHASE_TRANSPORT_CONSERVATIVE_ALLEN_CAHN)
    const dfloat div_lambda_normal = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z,M3_DIV_LAMBDA_NORMAL_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
    phiSharpSource = (M_AC - PHI_LBM_DIFF) * laplacian_phi - M_AC * div_lambda_normal;
#endif

