// ---------------------------------------------------------
// FILE: phi_sharpening.inc
// ---------------------------------------------------------

dfloat phiSharpSource = 0.0_df;

#ifdef METHOD_CAHN_HILLIARD
    // -----------------------------------------------------
    // CASE 1: Cahn-Hilliard (Mass Conserving)
    // -----------------------------------------------------
    // Logic: S = Mobility * Laplacian(Chemical Potential)
    // Requires: M_CH (Mobility constant)
    // Requires: M3_LM_INDEX (Precomputed in gpuComputeLaplacianMu)
    
    dfloat laplacian_mu = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M3_LM_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
    dfloat laplacian_phi  = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M3_LP_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
    
   /* CH diffusion */
    dfloat S_CH = M_CH * laplacian_mu;

    /* Counteract bare LBM diffusion */
    dfloat S_counter = -PHI_DIFF_REF * laplacian_phi;

    /* Total source */
    phiSharpSource = S_CH + S_counter;

#else
    // -----------------------------------------------------
    // CASE 2: Allen-Cahn / Interface Sharpening
    // -----------------------------------------------------
    // Logic: S = -beta * [(1-2phi)|grad phi| - phi(1-phi)kappa]
    // Requires: phi_sharp_c, mod_grad, curvature, phiVar

    // Only apply in the interface band to prevent bulk noise
    if(phiVar > 0.001_df && phiVar < 0.999_df) {
        
        dfloat phi_factor = phiVar * (1.0_df - phiVar);  // phi(1-phi)
        dfloat sign_term  = 1.0_df - 2.0_df * phiVar;    // (1-2phi)

        // Anti-diffusion: keeps the interface from spreading
        dfloat sharp_antidiff = sign_term * mod_grad;

        // Curvature term: regularizes the sharpening to keep it smooth
        dfloat sharp_curv = phi_factor * curvature;

        // Total Source
        phiSharpSource = - phi_sharp_c * (sharp_antidiff - sharp_curv);
    }
    // Else stays 0.0_df


    if (phiVar < 0.0045_df) {
        // Gently pull very small noise toward zero
        phiSharpSource -= 0.0005_df * phiVar; 
    } 
    else if (phiVar > 0.99_df) {
        // Gently push near-1 values toward one
        phiSharpSource += 0.0006_df * (1.0_df - phiVar);
    }
#endif