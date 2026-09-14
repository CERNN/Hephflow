// ---------------------------------------------------------
// FILE: phi_sharpening.inc
// ---------------------------------------------------------

dfloat phiSharpSource = 0.0_df;


    // -----------------------------------------------------
    // CASE 1: Cahn-Hilliard (Mass Conserving)
    // -----------------------------------------------------
    // Logic: S = Mobility * Laplacian(Chemical Potential)
    // Requires: M_CH (Mobility constant)
    // Requires: M3_LM_INDEX (Precomputed in gpuComputeLaplacianMu)
    
    dfloat laplacian_mu = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M3_LM_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
    // laplacian_phi already declared in phase_gradient.inc
    
   /* CH diffusion */
    dfloat S_CH = M_CH * laplacian_mu;

    /* Counteract bare LBM diffusion */
    dfloat S_counter = -PHI_LBM_DIFF * laplacian_phi;

    /* Total source */
    phiSharpSource = S_CH + S_counter;

