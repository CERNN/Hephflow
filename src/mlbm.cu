#include "mlbm.cuh"

__global__ void gpuMomCollisionStream(DeviceKernelParams params)
{
    // Split launches use a smaller grid in Z. Shadow CUDA's launch block index
    // with the corresponding block index in the complete local slab so the
    // existing indexing and AA face-neighbor logic remain unchanged.
    const uint3 launchBlockIdx = blockIdx;
    const unsigned int logicalBlockZ = (params.mapBoundaryBlocks && launchBlockIdx.z == 1) ? params.numBlocksZ - 1 : launchBlockIdx.z + params.zBlockOffset;
    const dim3 blockIdx(launchBlockIdx.x, launchBlockIdx.y, logicalBlockZ);

    // Unpack parameters from struct (passed by value - CUDA optimized!)
    dfloat *fMom = params.fMom;
    unsigned int *dNodeType = params.dNodeType;
    const ghostFacePtrs ghostInterface = params.pop;  // current interface generation
    #ifdef STREAMING_AB
    // AB reads the current generation and writes the alternate generation.
    // Its halo pointers belong to the same current generation.
        const ghostFacePtrs ghostInterfaceCopy = params.pop;
        const ghostFacePtrs storeGhostInterface = params.popCopy;
    #else
        const ghostFacePtrs ghostInterfaceCopy = params.popCopy;
        const ghostFacePtrs storeGhostInterface = params.pop;
    #endif // STREAMING_AB
    const gpuDirection macroInterfaceGPUrho = params.rho_macro;
    const gpuDirection macroInterfaceGPUux = params.ux_macro;
    const gpuDirection macroInterfaceGPUuy = params.uy_macro;
    const gpuDirection macroInterfaceGPUuz = params.uz_macro;

    const gpuDirection macroInterfaceGPUrhoCopy = params.rho_macroCopy;
    const gpuDirection macroInterfaceGPUuxCopy = params.ux_macroCopy;
    const gpuDirection macroInterfaceGPUuyCopy = params.uy_macroCopy;
    const gpuDirection macroInterfaceGPUuzCopy = params.uz_macroCopy;

    unsigned int step = params.step;
    bool save = params.save;
    size_t localNZ = params.localNZ;
    int zStart = params.zStart;

    #ifdef SECOND_DIST
        const ghostFacePtrs ghostInterfaceG = params.g;
        const ghostFacePtrs ghostInterfaceGCopy = params.gCopy;
        const gpuDirection macroInterfaceGPUg = params.g_macro;
        const gpuDirection macroInterfaceGPUgCopy = params.g_macroCopy;
    #endif //SECOND_DIST

    #ifdef A_XX_DIST
        const ghostFacePtrs ghostInterfaceAxx = params.Axx;
        const ghostFacePtrs ghostInterfaceAxxCopy = params.AxxCopy;
        const gpuDirection macroInterfaceGPUAxx = params.Axx_macro;
        const gpuDirection macroInterfaceGPUAxxCopy = params.Axx_macroCopy;
    #endif //A_XX_DIST
    #ifdef A_XY_DIST
        const ghostFacePtrs ghostInterfaceAxy = params.Axy;
        const ghostFacePtrs ghostInterfaceAxyCopy = params.AxyCopy;
        const gpuDirection macroInterfaceGPUAxy = params.Axy_macro;
        const gpuDirection macroInterfaceGPUAxyCopy = params.Axy_macroCopy;
    #endif //A_XY_DIST
    #ifdef A_XZ_DIST
        const ghostFacePtrs ghostInterfaceAxz = params.Axz;
        const ghostFacePtrs ghostInterfaceAxzCopy = params.AxzCopy;
        const gpuDirection macroInterfaceGPUAxz = params.Axz_macro;
        const gpuDirection macroInterfaceGPUAxzCopy = params.Axz_macroCopy;
    #endif //A_XZ_DIST
    #ifdef A_YY_DIST
        const ghostFacePtrs ghostInterfaceAyy = params.Ayy;
        const ghostFacePtrs ghostInterfaceAyyCopy = params.AyyCopy;
        const gpuDirection macroInterfaceGPUAyy = params.Ayy_macro;
        const gpuDirection macroInterfaceGPUAyyCopy = params.Ayy_macroCopy;
    #endif //A_YY_DIST
    #ifdef A_YZ_DIST
        const ghostFacePtrs ghostInterfaceAyz = params.Ayz;
        const ghostFacePtrs ghostInterfaceAyzCopy = params.AyzCopy;
        const gpuDirection macroInterfaceGPUAyz = params.Ayz_macro;
        const gpuDirection macroInterfaceGPUAyzCopy = params.Ayz_macroCopy;
    #endif //A_YZ_DIST
    #ifdef A_ZZ_DIST
        const ghostFacePtrs ghostInterfaceAzz = params.Azz;
        const ghostFacePtrs ghostInterfaceAzzCopy = params.AzzCopy;
        const gpuDirection macroInterfaceGPUAzz = params.Azz_macro;
        const gpuDirection macroInterfaceGPUAzzCopy = params.Azz_macroCopy;
    #endif //A_ZZ_DIST

    #ifdef CONFORMATION_PHYSICAL_A_HALO
        const PolymerStressHaloPtrs polymerStressHalo = params.polymerStressHalo;
    #endif // CONFORMATION_PHYSICAL_A_HALO

    #ifdef LAMBDA_DIST
        const ghostFacePtrs ghostInterfaceLambda = params.lambda;
        const ghostFacePtrs ghostInterfaceLambdaCopy = params.lambdaCopy;
        const gpuDirection macroInterfaceGPUlambda = params.lambda_macro;
        const gpuDirection macroInterfaceGPUlambdaCopy = params.lambda_macroCopy;
    #endif //LAMBDA_DIST

    #ifdef DENSITY_CORRECTION
        dfloat* d_mean_rho = params.d_mean_rho;
    #endif //DENSITY_CORRECTION

    #ifdef BC_FORCES
        dfloat* d_BC_Fx = params.d_BC_Fx;
        dfloat* d_BC_Fy = params.d_BC_Fy;
        dfloat* d_BC_Fz = params.d_BC_Fz;
    #endif //BC_FORCES

    #ifdef SAVE_LOCAL_FORCES
        dfloat* d_Local_Fx = params.d_Local_Fx;
        dfloat* d_Local_Fy = params.d_Local_Fy;
        dfloat* d_Local_Fz = params.d_Local_Fz;
    #ifdef SECOND_DIST
            dfloat* d_Source_C = params.d_Source_C;
    #endif // SECOND_DIST
    #ifdef PHI_DIST
            dfloat* d_Source_Phi = params.d_Source_Phi;
    #endif // PHI_DIST
    #ifdef LAMBDA_DIST
            dfloat* d_Source_Lambda = params.d_Source_Lambda;
    #endif // LAMBDA_DIST
    #ifdef CONFORMATION_TENSOR
    #ifdef A_XX_DIST
                dfloat* d_Source_Gxx = params.d_Source_Gxx;
    #endif // A_XX_DIST
    #ifdef A_XY_DIST
                dfloat* d_Source_Gxy = params.d_Source_Gxy;
    #endif // A_XY_DIST
    #ifdef A_XZ_DIST
                dfloat* d_Source_Gxz = params.d_Source_Gxz;
    #endif // A_XZ_DIST
    #ifdef A_YY_DIST
                dfloat* d_Source_Gyy = params.d_Source_Gyy;
    #endif // A_YY_DIST
    #ifdef A_YZ_DIST
                dfloat* d_Source_Gyz = params.d_Source_Gyz;
    #endif // A_YZ_DIST
    #ifdef A_ZZ_DIST
                dfloat* d_Source_Gzz = params.d_Source_Gzz;
    #endif // A_ZZ_DIST
    #endif //CONFORMATION_TENSOR
    #endif //SAVE_LOCAL_FORCES

    #ifdef CURVED_BOUNDARY_CONDITION
        CurvedBoundary** d_curvedBC = params.d_curvedBC;
        CurvedBoundary* d_curvedBC_array = params.d_curvedBC_array;
    #endif //CURVED_BOUNDARY_CONDITION

    #if defined(NON_NEWTONIAN_FLUID) || defined(CONFORMATION_TENSOR)
        const fluidPhaseProps phasePropsA = params.phasePropsA;
    #ifdef PHI_DIST
            const fluidPhaseProps phasePropsB = params.phasePropsB;
            const ghostFacePtrs ghostInterfacePhi = params.phi;
            const ghostFacePtrs ghostInterfacePhiCopy = params.phiCopy;
            const gpuDirection macroInterfaceGPUphi = params.phi_macro;
            const gpuDirection macroInterfaceGPUnx = params.nx_macro;
            const gpuDirection macroInterfaceGPUny = params.ny_macro;
            const gpuDirection macroInterfaceGPUnz = params.nz_macro;
    #ifdef PHASE_NEEDS_CHEMICAL_POTENTIAL
                const gpuDirection macroInterfaceGPUmu = params.mu_macro;
    #endif // PHASE_NEEDS_CHEMICAL_POTENTIAL

            const gpuDirection macroInterfaceGPUphiCopy = params.phi_macroCopy;
            const gpuDirection macroInterfaceGPUnxCopy = params.nx_macroCopy;
            const gpuDirection macroInterfaceGPUnyCopy = params.ny_macroCopy;
            const gpuDirection macroInterfaceGPUnzCopy = params.nz_macroCopy;
    #ifdef PHASE_NEEDS_CHEMICAL_POTENTIAL
                const gpuDirection macroInterfaceGPUmuCopy = params.mu_macroCopy;
    #endif // PHASE_NEEDS_CHEMICAL_POTENTIAL
    #endif // PHI_DIST
    #endif //NON_NEWTONIAN_FLUID || CONFORMATION_TENSOR

    const int x = threadIdx.x + blockDim.x * blockIdx.x;
    const int y = threadIdx.y + blockDim.y * blockIdx.y;
    const int z_local = threadIdx.z + blockDim.z * blockIdx.z;
    const bool active = x < NX && y < NY && z_local < localNZ;

    // Count physical lanes while the complete CUDA block is still present.
    // Padded lanes retire only after the active-lane barrier is initialized.
    __shared__ cuda::barrier<cuda::thread_scope_block> activeBarrier;
    const unsigned int activeCount = __syncthreads_count(active);
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
        init(&activeBarrier, activeCount);
    }
    __syncthreads();
    if (!active) return;

    const int z = zStart + z_local;

    dfloat pop[Q];
    #ifdef CONVECTION_DIFFUSION_TRANSPORT
        dfloat gNode[GQ];
    #endif //CONVECTION_DIFFUSION_TRANSPORT
    dfloat pics2;
    dfloat multiplyTerm;

    #ifdef DYNAMIC_SHARED_MEMORY
        extern __shared__ dfloat s_pop[];
    #else
        __shared__ dfloat s_pop[MAX_SHARED_MEMORY_SIZE/sizeof(dfloat)];
    #endif //DYNAMIC_SHARED_MEMORY

    const int tx=threadIdx.x, ty=threadIdx.y, tz=threadIdx.z;
    const int bx=blockIdx.x, by=blockIdx.y, bz=blockIdx.z;
    const int txm1 = (x == 0) ? (NX - 1) % BLOCK_NX : (tx - 1 + BLOCK_NX) % BLOCK_NX;
    const int txp1 = (x + 1 >= NX) ? 0 : (tx + 1) % BLOCK_NX;
    const int tym1 = (y == 0) ? (NY - 1) % BLOCK_NY : (ty - 1 + BLOCK_NY) % BLOCK_NY;
    const int typ1 = (y + 1 >= NY) ? 0 : (ty + 1) % BLOCK_NY;
    const int tzm1 = (z_local == 0) ? (localNZ - 1) % BLOCK_NZ : (tz - 1 + BLOCK_NZ) % BLOCK_NZ;
    const int tzp1 = (z_local + 1 >= localNZ) ? 0 : (tz + 1) % BLOCK_NZ;
    const int bxm1=(bx-1+NUM_BLOCK_X)%NUM_BLOCK_X, bxp1=(bx+1)%NUM_BLOCK_X;
    const int bym1=(by-1+NUM_BLOCK_Y)%NUM_BLOCK_Y, byp1=(by+1)%NUM_BLOCK_Y;
    const int bzm1=(bz-1+params.numBlocksZ)%params.numBlocksZ;
    const int bzp1=(bz+1)%params.numBlocksZ;

    const int baseIdx = idxMom(threadIdx.x, threadIdx.y, threadIdx.z, 0, blockIdx.x, blockIdx.y, blockIdx.z);
    const int baseIdxPop = idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  0);

    // Load moments from global memory

    //rho'
    unsigned int nodeType = dNodeType[idxScalarBlock(threadIdx.x, threadIdx.y, threadIdx.z,blockIdx.x, blockIdx.y, blockIdx.z)];
    #ifdef CONFORMATION_PHYSICAL_A_HALO
        #include "fragments/conformationTransport/physical_conformation_halo.inc"
    #endif // CONFORMATION_PHYSICAL_A_HALO
    // Physical node types are initialized before collision. No active lane may
    // retire after activeBarrier has been initialized.

    #define LOAD_HYDRO_MOMENT(moment) loadMoment(fMom, idxMom(threadIdx.x, threadIdx.y, threadIdx.z, moment, blockIdx.x, blockIdx.y, blockIdx.z), moment)

    dfloat rhoVar = RHO_0 + LOAD_HYDRO_MOMENT(M_RHO_INDEX);
    dfloat ux_t30     = LOAD_HYDRO_MOMENT(M_UX_INDEX);
    dfloat uy_t30     = LOAD_HYDRO_MOMENT(M_UY_INDEX);
    dfloat uz_t30     = LOAD_HYDRO_MOMENT(M_UZ_INDEX);
    dfloat m_xx_t45   = LOAD_HYDRO_MOMENT(M_MXX_INDEX);
    dfloat m_xy_t90   = LOAD_HYDRO_MOMENT(M_MXY_INDEX);
    dfloat m_xz_t90   = LOAD_HYDRO_MOMENT(M_MXZ_INDEX);
    dfloat m_yy_t45   = LOAD_HYDRO_MOMENT(M_MYY_INDEX);
    dfloat m_yz_t90   = LOAD_HYDRO_MOMENT(M_MYZ_INDEX);
    dfloat m_zz_t45   = LOAD_HYDRO_MOMENT(M_MZZ_INDEX);
    #undef LOAD_HYDRO_MOMENT



    #ifdef OMEGA_FIELD
        //dfloat omegaVar = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_OMEGA_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];

        dfloat omegaVar = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_OMEGA_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
        dfloat t_omegaVar = 1.0_df - omegaVar;
        dfloat tt_omegaVar = 1.0_df - omegaVar*0.5_df;
        dfloat omegaVar_d2 = omegaVar*0.5_df;
        dfloat tt_omega_t3 = tt_omegaVar * 3.0_df;
    #else
        const dfloat omegaVar = OMEGA;
        const dfloat t_omegaVar = 1.0_df - omegaVar;
        const dfloat tt_omegaVar = 1.0_df - omegaVar*0.5_df;
        const dfloat omegaVar_d2 = omegaVar*0.5_df;
        const dfloat tt_omega_t3 = tt_omegaVar * 3.0_df;
    #endif //OMEGA_FIELD

    /*
    if(z > (NZ_TOTAL-50)){
        dfloat dist = (z - (NZ_TOTAL-50))/((NZ_TOTAL)- (NZ_TOTAL-50));
        dfloat ttau = 0.5_df+ 3*VISC*(1000.0_df*dist*dist*dist+1.0_df);
        omegaVar = 1/ttau;
    }*/

    //Local forces
    //dfloat K_const = 2.0_df*M_PI/(dfloat)N;
   // dfloat xx = 2.0_df * M_PI * x / L;
   // dfloat yy = 2.0_df * M_PI * y / L;
   // dfloat zz = 2.0_df * M_PI * z / L;


    #ifdef LOCAL_FORCES
        dfloat L_Fx = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_FX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
        dfloat L_Fy = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_FY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
        dfloat L_Fz = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_FZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
    #else
        dfloat L_Fx = FX;
        dfloat L_Fy = FY;
        dfloat L_Fz = FZ;
    #endif //LOCAL_FORCES

    #ifdef FORCE_FIELD_INCLUDE
        #include CASE_FORCE_FIELD
    #endif //FORCE_FIELD_INCLUDE


    #ifdef BC_FORCES
        dfloat L_BC_Fx = 0.0_df;
        dfloat L_BC_Fy = 0.0_df;
        dfloat L_BC_Fz = 0.0_df;
    #endif //BC_FORCES

    // Load phi once here and compute all phase-property scalars used throughout the kernel:
    // h_phi e [0,1], rho_pf (physical mass density), invRhoPF.
    // Avoids redundant global memory reads in PHI_DIST blocks, OMEGA_FIELD, and conformation_evolution.
    #ifdef PHI_DIST
        const dfloat phiVar_phi = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M3_PHI_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
        dfloat h_phi = (phiVar_phi - PHI_ONE) / (PHI_TWO - PHI_ONE);
        h_phi = fmaxf(0.0_df, fminf(1.0_df, h_phi));
        const dfloat rho_pf   = PHI_RHO_PHASE1 + PHI_DRHO_PHASE12 * h_phi;
        const dfloat invRhoPF = 1.0_df / rho_pf;
    #endif //PHI_DIST


    #include COLREC_RECONSTRUCTION

    const unsigned short int xp1 = (threadIdx.x + 1 + BLOCK_NX) % BLOCK_NX;
    const unsigned short int xm1 = (threadIdx.x - 1 + BLOCK_NX) % BLOCK_NX;

    const unsigned short int yp1 = (threadIdx.y + 1 + BLOCK_NY) % BLOCK_NY;
    const unsigned short int ym1 = (threadIdx.y - 1 + BLOCK_NY) % BLOCK_NY;

    const unsigned short int zp1 = (threadIdx.z + 1 + BLOCK_NZ) % BLOCK_NZ;
    const unsigned short int zm1 = (threadIdx.z - 1 + BLOCK_NZ) % BLOCK_NZ;

    const bool stepParity = params.streamingParity;

    //need to compute the gradient before the moments are recalculated
    #ifdef COMPUTE_VEL_GRADIENT_FINITE_DIFFERENCE
        #include "fragments/velocity_gradient.inc"
    #endif //COMPUTE_VEL_GRADIENT_FINITE_DIFFERENCE

    #ifdef PHI_DIST
        // Precompute phase-force contribution before conformation evolution so we can
        // remove its induced part from velocity gradients used by conformation transport.
        dfloat F_phase_x = 0.0_df;
        dfloat F_phase_y = 0.0_df;
        dfloat F_phase_z = 0.0_df;

        dfloat phase_dphidx = 0.0_df;
        dfloat phase_dphidy = 0.0_df;
        dfloat phase_dphidz = 0.0_df;
        dfloat phase_laplacian_phi = 0.0_df;

        dfloat phase_du_xx = 0.0_df, phase_du_xy = 0.0_df, phase_du_xz = 0.0_df;
        dfloat phase_du_yx = 0.0_df, phase_du_yy = 0.0_df, phase_du_yz = 0.0_df;
        dfloat phase_du_zx = 0.0_df, phase_du_zy = 0.0_df, phase_du_zz = 0.0_df;

        {
            const dfloat phiVar = phiVar_phi;  // reuse pre-loaded phi; avoids a second global memory read

            #include "fragments/phiTransport/phase_gradient.inc"
            #include "fragments/phiTransport/phase_coupling_forces.inc"

            phase_dphidx = dphidx;
            phase_dphidy = dphidy;
            phase_dphidz = dphidz;
            phase_laplacian_phi = laplacian_phi;

            // First-order estimate: u_phase ~ F_phase/(2*rho). Approximate grad(u_phase)
            // by density-gradient coupling to remove phase-force-induced strain from VE evolution.
            const dfloat invRhoLocal = 1.0_df / fmax(rhoVar, 1.0e-12_df);
            const dfloat phaseVelCoef = 0.5_df * invRhoLocal * invRhoLocal;

            phase_du_xx = -phaseVelCoef * F_phase_x * drhox;
            phase_du_xy = -phaseVelCoef * F_phase_x * drhoy;
            phase_du_xz = -phaseVelCoef * F_phase_x * drhoz;

            phase_du_yx = -phaseVelCoef * F_phase_y * drhox;
            phase_du_yy = -phaseVelCoef * F_phase_y * drhoy;
            phase_du_yz = -phaseVelCoef * F_phase_y * drhoz;

            phase_du_zx = -phaseVelCoef * F_phase_z * drhox;
            phase_du_zy = -phaseVelCoef * F_phase_z * drhoy;
            phase_du_zz = -phaseVelCoef * F_phase_z * drhoz;
            }
    #endif //PHI_DIST


    #ifdef CONFORMATION_TENSOR
    #ifdef A_XX_DIST
            dfloat AxxVar = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_XX_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
    #endif //A_XX_DIST
    #ifdef A_XY_DIST
            dfloat AxyVar = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_XY_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
    #endif //A_XY_DIST
    #ifdef A_XZ_DIST
            dfloat AxzVar = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_XZ_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
    #endif //A_XZ_DIST
    #ifdef A_YY_DIST
            dfloat AyyVar = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_YY_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
    #endif //A_YY_DIST
    #ifdef A_YZ_DIST
            dfloat AyzVar = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_YZ_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
    #endif //A_YZ_DIST
    #ifdef A_ZZ_DIST
            dfloat AzzVar = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_ZZ_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
    #endif //A_ZZ_DIST

    #if !defined(CONFORMATION_DIRECT_STRESS_HALO) && defined(COMPUTE_CONF_GRADIENT_FINITE_DIFFERENCE)
            #include "fragments/conformationTransport/stored_tensor_gradient.inc"
    #endif //COMPUTE_CONF_GRADIENT_FINITE_DIFFERENCE

    #if CONF_REPRESENTATION == CONF_REP_LOG
            #include "fragments/conformationTransport/log_conformation_evolution.inc"
    #elif CONF_REPRESENTATION == CONF_REP_SQRT
            #include "fragments/conformationTransport/sqrt_conformation_evolution.inc"
    #else
            #include "fragments/conformationTransport/standard_conformation_evolution.inc"
    #endif // CONF_REPRESENTATION == CONF_REP_LOG
    #ifdef CONFORMATION_DIRECT_STRESS_HALO
    #ifndef DISABLE_POLYMER_MOMENTUM_FEEDBACK
                {
                    dfloat polymerFeedbackRamp=1.0_df;
        #ifdef VISCOELASTIC_STARTUP_RAMP
                        const dfloat r=fmin(1.0_df,(dfloat)step/(dfloat)STARTUP_RAMP_STEPS);
                        polymerFeedbackRamp=r*r*(3.0_df-2.0_df*r);
        #endif // VISCOELASTIC_STARTUP_RAMP

                    const dfloat feedbackWeight=switch_conf*polymerFeedbackRamp;
                    const dfloat modulus=calcVePolymerModulus(veLocal);

                    L_Fx += modulus * directDivStressX * feedbackWeight;
                    L_Fy += modulus * directDivStressY * feedbackWeight;
                    L_Fz += modulus * directDivStressZ * feedbackWeight;
                    }
    #endif // DISABLE_POLYMER_MOMENTUM_FEEDBACK
    #endif // CONFORMATION_DIRECT_STRESS_HALO
    #ifdef SAVE_LOCAL_FORCES
            if(save){
        #ifdef A_XX_DIST
                    d_Source_Gxx[idxScalarBlock(threadIdx.x, threadIdx.y, threadIdx.z,blockIdx.x, blockIdx.y, blockIdx.z)] = Gxx;
        #endif // A_XX_DIST
        #ifdef A_XY_DIST
                    d_Source_Gxy[idxScalarBlock(threadIdx.x, threadIdx.y, threadIdx.z,blockIdx.x, blockIdx.y, blockIdx.z)] = Gxy;
        #endif // A_XY_DIST
        #ifdef A_XZ_DIST
                    d_Source_Gxz[idxScalarBlock(threadIdx.x, threadIdx.y, threadIdx.z,blockIdx.x, blockIdx.y, blockIdx.z)] = Gxz;
        #endif // A_XZ_DIST
        #ifdef A_YY_DIST
                    d_Source_Gyy[idxScalarBlock(threadIdx.x, threadIdx.y, threadIdx.z,blockIdx.x, blockIdx.y, blockIdx.z)] = Gyy;
        #endif // A_YY_DIST
        #ifdef A_YZ_DIST
                    d_Source_Gyz[idxScalarBlock(threadIdx.x, threadIdx.y, threadIdx.z,blockIdx.x, blockIdx.y, blockIdx.z)] = Gyz;
        #endif // A_YZ_DIST
        #ifdef A_ZZ_DIST
                    d_Source_Gzz[idxScalarBlock(threadIdx.x, threadIdx.y, threadIdx.z,blockIdx.x, blockIdx.y, blockIdx.z)] = Gzz;
        #endif // A_ZZ_DIST
                }
    #endif //SAVE_LOCAL_FORCES
    #endif //CONFORMATION_TENSOR

    #ifdef CONVECTION_DIFFUSION_TRANSPORT
    #ifdef SECOND_DIST

            dfloat cVar = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M2_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];

            dfloat invC = 1/cVar;
            dfloat qx_t30   = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M2_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
            dfloat qy_t30   = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M2_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
            dfloat qz_t30   = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M2_CZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];

            #include  COLREC_G_RECONSTRUCTION

            activeBarrier.arrive_and_wait();

            #include "fragments/convection_diffusion_streaming.inc"
            /* load pop from global in cover nodes */
            {
                #include "fragments/gTransport/g_popLoad.inc"
                }


            if(nodeType != BULK){
                #include CASE_G_BC_DEF
                }else{
                cVar = gNode[0] + gNode[1] + gNode[2] + gNode[3] + gNode[4] + gNode[5] + gNode[6] + gNode[7] + gNode[8] + gNode[9] + gNode[10] + gNode[11] + gNode[12] + gNode[13] + gNode[14] + gNode[15] + gNode[16] + gNode[17] + gNode[18];
                cVar = cVar + T_Q_INTERNAL_D_Cp;
                invC= 1.0_df/cVar;

                qx_t30 = ((gNode[1] - gNode[2] + gNode[7] - gNode[ 8] + gNode[ 9] - gNode[10] + gNode[13] - gNode[14] + gNode[15] - gNode[16]))*invC;
                qy_t30 = ((gNode[3] - gNode[4] + gNode[7] - gNode[ 8] + gNode[11] - gNode[12] + gNode[14] - gNode[13] + gNode[17] - gNode[18]))*invC;
                qz_t30 = ((gNode[5] - gNode[6] + gNode[9] - gNode[10] + gNode[11] - gNode[12] + gNode[16] - gNode[15] + gNode[18] - gNode[17]))*invC;
                }

            qx_t30 = F_M_I_SCALE * qx_t30;
            qy_t30 = F_M_I_SCALE * qy_t30;
            qz_t30 = F_M_I_SCALE * qz_t30;

            #include COLREC_G_COLLISION

    #endif //SECOND_DIST
    #ifdef PHI_DIST

            dfloat phiVar = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M3_PHI_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
            dfloat dphidx = phase_dphidx;
            dfloat dphidy = phase_dphidy;
            dfloat dphidz = phase_dphidz;
            dfloat laplacian_phi = phase_laplacian_phi;

            L_Fx += F_phase_x;
            L_Fy += F_phase_y;
            L_Fz += F_phase_z;

            dfloat phiSource = 0.0_df;
    #ifdef INTERFACE_SHARPENING
                #include "fragments/phiTransport/phi_sharpening.inc"
                phiSource += phiSharpSource;
    #else
                dfloat phiSharpSource = 0.0_df;
    #endif //INTERFACE_SHARPENING
    #ifdef SAVE_LOCAL_FORCES
                if(save){ d_Source_Phi[idxScalarBlock(threadIdx.x, threadIdx.y, threadIdx.z,blockIdx.x, blockIdx.y, blockIdx.z)] = phiSource; }
    #endif // SAVE_LOCAL_FORCES
            dfloat phi_qx_t30   = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M3_PX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
            dfloat phi_qy_t30   = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M3_PY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
            dfloat phi_qz_t30   = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M3_PZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];

            #include  COLREC_PHI_RECONSTRUCTION

            activeBarrier.arrive_and_wait();

            #include "fragments/convection_diffusion_streaming.inc"
            /* load pop from global in cover nodes */
            {
                #include "fragments/phiTransport/phi_popLoad.inc"
                }

            if(nodeType != BULK){
                #include CASE_PHI_BC_DEF
                }else{
                phiVar = gNode[0] + gNode[1] + gNode[2] + gNode[3] + gNode[4] + gNode[5] + gNode[6]
        #ifdef D3G19
                    + gNode[7] + gNode[8] + gNode[9] + gNode[10] + gNode[11] + gNode[12] + gNode[13] + gNode[14] + gNode[15] + gNode[16] + gNode[17] + gNode[18]
        #endif // D3G19
        #ifdef D3G27
                    + gNode[7] + gNode[8] + gNode[9] + gNode[10] + gNode[11] + gNode[12] + gNode[13] + gNode[14] + gNode[15] + gNode[16] + gNode[17] + gNode[18]
                    + gNode[19] + gNode[20] + gNode[21] + gNode[22] + gNode[23] + gNode[24] + gNode[25] + gNode[26]
        #endif // D3G27
                ;
                phiVar = phiVar + phiSource;
        #if PHI_CLAMP_ENABLED
                    const dfloat unclippedPhi = phiVar;
                    phiVar = myMin(myMax(phiVar, PHI_ONE), PHI_TWO);
        #ifdef PHASE_DIAGNOSTICS
                        const dfloat clipCorrection = phiVar - unclippedPhi;
                        const size_t clipSignedIndex = idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M3_CLIP_SIGNED_INDEX, blockIdx.x, blockIdx.y, blockIdx.z);
                        const size_t clipAbsoluteIndex = idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M3_CLIP_ABSOLUTE_INDEX, blockIdx.x, blockIdx.y, blockIdx.z);
                        storeMoment(fMom, clipSignedIndex, M3_CLIP_SIGNED_INDEX, loadMoment(fMom, clipSignedIndex, M3_CLIP_SIGNED_INDEX) + clipCorrection);
                        storeMoment(fMom, clipAbsoluteIndex, M3_CLIP_ABSOLUTE_INDEX, loadMoment(fMom, clipAbsoluteIndex, M3_CLIP_ABSOLUTE_INDEX) + fabs(clipCorrection));
        #endif // PHASE_DIAGNOSTICS
        #endif // PHI_CLAMP_ENABLED

                phi_qx_t30 = (gNode[1] - gNode[2]
        #ifdef D3G19
                    + gNode[7] - gNode[ 8] + gNode[ 9] - gNode[10] + gNode[13] - gNode[14] + gNode[15] - gNode[16]
        #endif // D3G19
        #ifdef D3G27
                    + gNode[7] - gNode[ 8] + gNode[ 9] - gNode[10] + gNode[13] - gNode[14] + gNode[15] - gNode[16]
                    + gNode[19] - gNode[20] + gNode[21] - gNode[22] + gNode[23] - gNode[24] - gNode[25] + gNode[26]
        #endif // D3G27
                );
                phi_qy_t30 = (gNode[3] - gNode[4]
        #ifdef D3G19
                    + gNode[7] - gNode[ 8] + gNode[11] - gNode[12] + gNode[14] - gNode[13] + gNode[17] - gNode[18]
        #endif // D3G19
        #ifdef D3G27
                    + gNode[7] - gNode[ 8] + gNode[11] - gNode[12] + gNode[14] - gNode[13] + gNode[17] - gNode[18]
                    + gNode[19] - gNode[20] + gNode[21] - gNode[22] - gNode[23] + gNode[24] + gNode[25] - gNode[26]
        #endif // D3G27
                );
                phi_qz_t30 = (gNode[5] - gNode[6]
        #ifdef D3G19
                    + gNode[9] - gNode[10] + gNode[11] - gNode[12] + gNode[16] - gNode[15] + gNode[18] - gNode[17]
        #endif // D3G19
        #ifdef D3G27
                    + gNode[9] - gNode[10] + gNode[11] - gNode[12] + gNode[16] - gNode[15] + gNode[18] - gNode[17]
                    + gNode[19] - gNode[20] - gNode[21] + gNode[22] + gNode[23] - gNode[24] + gNode[25] - gNode[26]
        #endif // D3G27
                );
                }

    #ifdef PHI_INLET_RESERVOIR_Z_NODES
                if (z < PHI_INLET_RESERVOIR_Z_NODES) {
                    phiVar = PHI_TWO;
                // ux_t30 etc. are still stored in scaled form at this point;
                // the phase moments below are converted back to stored form.
                    phi_qx_t30 = PHI_TWO * ux_t30 / F_M_I_SCALE;
                    phi_qy_t30 = PHI_TWO * uy_t30 / F_M_I_SCALE;
                    phi_qz_t30 = PHI_TWO * uz_t30 / F_M_I_SCALE;
                    }
    #endif // PHI_INLET_RESERVOIR_Z_NODES


            phi_qx_t30 = F_M_I_SCALE * phi_qx_t30;
            phi_qy_t30 = F_M_I_SCALE * phi_qy_t30;
            phi_qz_t30 = F_M_I_SCALE * phi_qz_t30;

            #include COLREC_PHI_COLLISION
    #endif //PHI_DIST
    #ifdef LAMBDA_DIST

            dfloat lambdaVar = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M4_LAMBDA_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];

            // Compute source term using function-based dispatch
            dfloat lambdaSource = computeLambdaSourceFromStress( phasePropsA.nnf, rhoVar, ux_t30, uy_t30, uz_t30, m_xx_t45, m_yy_t45, m_zz_t45, m_xy_t90, m_xz_t90, m_yz_t90, omegaVar, lambdaVar );
    #ifdef SAVE_LOCAL_FORCES
                if(save){ d_Source_Lambda[idxScalarBlock(threadIdx.x, threadIdx.y, threadIdx.z,blockIdx.x, blockIdx.y, blockIdx.z)] = lambdaSource; }
    #endif // SAVE_LOCAL_FORCES

            dfloat invLambda = 1.0_df/(lambdaVar);
            dfloat lambda_qx_t30   = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M4_LX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
            dfloat lambda_qy_t30   = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M4_LY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
            dfloat lambda_qz_t30   = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M4_LZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];

            dfloat lambda_udx_t30 = LAMBDA_DIFF_FLUC_COEF * (lambda_qx_t30*invLambda - ux_t30);
            dfloat lambda_udy_t30 = LAMBDA_DIFF_FLUC_COEF * (lambda_qy_t30*invLambda - uy_t30);
            dfloat lambda_udz_t30 = LAMBDA_DIFF_FLUC_COEF * (lambda_qz_t30*invLambda - uz_t30);

            #include  COLREC_LAMBDA_RECONSTRUCTION

            activeBarrier.arrive_and_wait();

            #include "fragments/convection_diffusion_streaming.inc"
            /* load pop from global in cover nodes */
            {
                #include "fragments/lambdaTransport/lambda_popLoad.inc"
                }

            if(nodeType != BULK){
                #include CASE_LAMBDA_BC_DEF
                }else{
                dfloat lambdaFromPop = gNode[0] + gNode[1] + gNode[2] + gNode[3] + gNode[4] + gNode[5] + gNode[6] + gNode[7] + gNode[8] + gNode[9] + gNode[10] + gNode[11] + gNode[12] + gNode[13] + gNode[14] + gNode[15] + gNode[16] + gNode[17] + gNode[18];

                // Apply source term (build/break kinetics)
                lambdaVar = lambdaFromPop + lambdaSource;

                // Clamp to [LAMBDA_ZERO, LAMBDA_ZERO + 1]
                lambdaVar = fmaxf(LAMBDA_ZERO, fminf(LAMBDA_ZERO + 1.0_df, lambdaVar));
                invLambda = 1.0_df/lambdaVar;

                lambda_qx_t30 = F_M_I_SCALE*((gNode[1] - gNode[2] + gNode[7] - gNode[ 8] + gNode[ 9] - gNode[10] + gNode[13] - gNode[14] + gNode[15] - gNode[16]));
                lambda_qy_t30 = F_M_I_SCALE*((gNode[3] - gNode[4] + gNode[7] - gNode[ 8] + gNode[11] - gNode[12] + gNode[14] - gNode[13] + gNode[17] - gNode[18]));
                lambda_qz_t30 = F_M_I_SCALE*((gNode[5] - gNode[6] + gNode[9] - gNode[10] + gNode[11] - gNode[12] + gNode[16] - gNode[15] + gNode[18] - gNode[17]));
                }
    #endif //LAMBDA_DIST
    #ifdef A_XX_DIST
            dfloat Axx_qx_t30 = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_XX_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
            dfloat Axx_qy_t30 = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_XX_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
            dfloat Axx_qz_t30 = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_XX_CZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];

            #include COLREC_AXX_RECONSTRUCTION

            activeBarrier.arrive_and_wait();

            #include "fragments/convection_diffusion_streaming.inc"
            /* load pop from global in cover nodes */
            {
                #include "fragments/conformationTransport/popLoad_Axx.inc"
                }

            if(nodeType != BULK){
                 #include CASE_AXX_BC_DEF
                }else{
                AxxVar = gNode[0] + gNode[1] + gNode[2] + gNode[3] + gNode[4] + gNode[5] + gNode[6]
        #ifdef D3G19
                    + gNode[7] + gNode[8] + gNode[9] + gNode[10] + gNode[11] + gNode[12] + gNode[13] + gNode[14] + gNode[15] + gNode[16] + gNode[17] + gNode[18]
        #endif // D3G19
        #ifdef D3G27
                    + gNode[7] + gNode[8] + gNode[9] + gNode[10] + gNode[11] + gNode[12] + gNode[13] + gNode[14] + gNode[15] + gNode[16] + gNode[17] + gNode[18]
                    + gNode[19] + gNode[20] + gNode[21] + gNode[22] + gNode[23] + gNode[24] + gNode[25] + gNode[26]
        #endif // D3G27
                ;
                AxxVar = AxxVar + Gxx;

                Axx_qx_t30 = (gNode[1] - gNode[2]
        #ifdef D3G19
                    + gNode[7] - gNode[ 8] + gNode[ 9] - gNode[10] + gNode[13] - gNode[14] + gNode[15] - gNode[16]
        #endif // D3G19
        #ifdef D3G27
                    + gNode[7] - gNode[ 8] + gNode[ 9] - gNode[10] + gNode[13] - gNode[14] + gNode[15] - gNode[16]
                    + gNode[19] - gNode[20] + gNode[21] - gNode[22] + gNode[23] - gNode[24] - gNode[25] + gNode[26]
        #endif // D3G27
                );
                Axx_qy_t30 = (gNode[3] - gNode[4]
        #ifdef D3G19
                    + gNode[7] - gNode[ 8] + gNode[11] - gNode[12] + gNode[14] - gNode[13] + gNode[17] - gNode[18]
        #endif // D3G19
        #ifdef D3G27
                    + gNode[7] - gNode[ 8] + gNode[11] - gNode[12] + gNode[14] - gNode[13] + gNode[17] - gNode[18]
                    + gNode[19] - gNode[20] + gNode[21] - gNode[22] - gNode[23] + gNode[24] + gNode[25] - gNode[26]
        #endif // D3G27
                );
                Axx_qz_t30 = (gNode[5] - gNode[6]
        #ifdef D3G19
                    + gNode[9] - gNode[10] + gNode[11] - gNode[12] + gNode[16] - gNode[15] + gNode[18] - gNode[17]
        #endif // D3G19
        #ifdef D3G27
                    + gNode[9] - gNode[10] + gNode[11] - gNode[12] + gNode[16] - gNode[15] + gNode[18] - gNode[17]
                    + gNode[19] - gNode[20] - gNode[21] + gNode[22] + gNode[23] - gNode[24] + gNode[25] - gNode[26]
        #endif // D3G27
                );
                }

            Axx_qx_t30 = F_M_I_SCALE * Axx_qx_t30;
            Axx_qy_t30 = F_M_I_SCALE * Axx_qy_t30;
            Axx_qz_t30 = F_M_I_SCALE * Axx_qz_t30;

            #include COLREC_AXX_COLLISION

    #endif //A_XX_DIST
    #ifdef A_XY_DIST
                        dfloat Axy_qx_t30 = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_XY_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
            dfloat Axy_qy_t30 = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_XY_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
            dfloat Axy_qz_t30 = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_XY_CZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];

            #include COLREC_AXY_RECONSTRUCTION

            activeBarrier.arrive_and_wait();

            #include "fragments/convection_diffusion_streaming.inc"
            /* load pop from global in cover nodes */
            {
                #include "fragments/conformationTransport/popLoad_Axy.inc"
                }

            if(nodeType != BULK){
                 #include CASE_AXY_BC_DEF
                }else{
                AxyVar = gNode[0] + gNode[1] + gNode[2] + gNode[3] + gNode[4] + gNode[5] + gNode[6]
        #ifdef D3G19
                    + gNode[7] + gNode[8] + gNode[9] + gNode[10] + gNode[11] + gNode[12] + gNode[13] + gNode[14] + gNode[15] + gNode[16] + gNode[17] + gNode[18]
        #endif // D3G19
        #ifdef D3G27
                    + gNode[7] + gNode[8] + gNode[9] + gNode[10] + gNode[11] + gNode[12] + gNode[13] + gNode[14] + gNode[15] + gNode[16] + gNode[17] + gNode[18]
                    + gNode[19] + gNode[20] + gNode[21] + gNode[22] + gNode[23] + gNode[24] + gNode[25] + gNode[26]
        #endif // D3G27
                ;
                AxyVar = AxyVar + Gxy;

                Axy_qx_t30 = (gNode[1] - gNode[2]
        #ifdef D3G19
                    + gNode[7] - gNode[ 8] + gNode[ 9] - gNode[10] + gNode[13] - gNode[14] + gNode[15] - gNode[16]
        #endif // D3G19
        #ifdef D3G27
                    + gNode[7] - gNode[ 8] + gNode[ 9] - gNode[10] + gNode[13] - gNode[14] + gNode[15] - gNode[16]
                    + gNode[19] - gNode[20] + gNode[21] - gNode[22] + gNode[23] - gNode[24] - gNode[25] + gNode[26]
        #endif // D3G27
                );
                Axy_qy_t30 = (gNode[3] - gNode[4]
        #ifdef D3G19
                    + gNode[7] - gNode[ 8] + gNode[11] - gNode[12] + gNode[14] - gNode[13] + gNode[17] - gNode[18]
        #endif // D3G19
        #ifdef D3G27
                    + gNode[7] - gNode[ 8] + gNode[11] - gNode[12] + gNode[14] - gNode[13] + gNode[17] - gNode[18]
                    + gNode[19] - gNode[20] + gNode[21] - gNode[22] - gNode[23] + gNode[24] + gNode[25] - gNode[26]
        #endif // D3G27
                );
                Axy_qz_t30 = (gNode[5] - gNode[6]
        #ifdef D3G19
                    + gNode[9] - gNode[10] + gNode[11] - gNode[12] + gNode[16] - gNode[15] + gNode[18] - gNode[17]
        #endif // D3G19
        #ifdef D3G27
                    + gNode[9] - gNode[10] + gNode[11] - gNode[12] + gNode[16] - gNode[15] + gNode[18] - gNode[17]
                    + gNode[19] - gNode[20] - gNode[21] + gNode[22] + gNode[23] - gNode[24] + gNode[25] - gNode[26]
        #endif // D3G27
                );
                }

            Axy_qx_t30 = F_M_I_SCALE * Axy_qx_t30;
            Axy_qy_t30 = F_M_I_SCALE * Axy_qy_t30;
            Axy_qz_t30 = F_M_I_SCALE * Axy_qz_t30;

            #include COLREC_AXY_COLLISION

    #endif //A_XY_DIST
    #ifdef A_XZ_DIST
            dfloat Axz_qx_t30 = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_XZ_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
            dfloat Axz_qy_t30 = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_XZ_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
            dfloat Axz_qz_t30 = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_XZ_CZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];

            #include COLREC_AXZ_RECONSTRUCTION

            activeBarrier.arrive_and_wait();

            #include "fragments/convection_diffusion_streaming.inc"
            /* load pop from global in cover nodes */
            {
                #include "fragments/conformationTransport/popLoad_Axz.inc"
                }

            if(nodeType != BULK){
                 #include CASE_AXZ_BC_DEF
                }else{
                AxzVar = gNode[0] + gNode[1] + gNode[2] + gNode[3] + gNode[4] + gNode[5] + gNode[6]
        #ifdef D3G19
                    + gNode[7] + gNode[8] + gNode[9] + gNode[10] + gNode[11] + gNode[12] + gNode[13] + gNode[14] + gNode[15] + gNode[16] + gNode[17] + gNode[18]
        #endif // D3G19
        #ifdef D3G27
                    + gNode[7] + gNode[8] + gNode[9] + gNode[10] + gNode[11] + gNode[12] + gNode[13] + gNode[14] + gNode[15] + gNode[16] + gNode[17] + gNode[18]
                    + gNode[19] + gNode[20] + gNode[21] + gNode[22] + gNode[23] + gNode[24] + gNode[25] + gNode[26]
        #endif // D3G27
                ;
                AxzVar = AxzVar + Gxz;

                Axz_qx_t30 = (gNode[1] - gNode[2]
        #ifdef D3G19
                    + gNode[7] - gNode[ 8] + gNode[ 9] - gNode[10] + gNode[13] - gNode[14] + gNode[15] - gNode[16]
        #endif // D3G19
        #ifdef D3G27
                    + gNode[7] - gNode[ 8] + gNode[ 9] - gNode[10] + gNode[13] - gNode[14] + gNode[15] - gNode[16]
                    + gNode[19] - gNode[20] + gNode[21] - gNode[22] + gNode[23] - gNode[24] - gNode[25] + gNode[26]
        #endif // D3G27
                );
                Axz_qy_t30 = (gNode[3] - gNode[4]
        #ifdef D3G19
                    + gNode[7] - gNode[ 8] + gNode[11] - gNode[12] + gNode[14] - gNode[13] + gNode[17] - gNode[18]
        #endif // D3G19
        #ifdef D3G27
                    + gNode[7] - gNode[ 8] + gNode[11] - gNode[12] + gNode[14] - gNode[13] + gNode[17] - gNode[18]
                    + gNode[19] - gNode[20] + gNode[21] - gNode[22] - gNode[23] + gNode[24] + gNode[25] - gNode[26]
        #endif // D3G27
                );
                Axz_qz_t30 = (gNode[5] - gNode[6]
        #ifdef D3G19
                    + gNode[9] - gNode[10] + gNode[11] - gNode[12] + gNode[16] - gNode[15] + gNode[18] - gNode[17]
        #endif // D3G19
        #ifdef D3G27
                    + gNode[9] - gNode[10] + gNode[11] - gNode[12] + gNode[16] - gNode[15] + gNode[18] - gNode[17]
                    + gNode[19] - gNode[20] - gNode[21] + gNode[22] + gNode[23] - gNode[24] + gNode[25] - gNode[26]
        #endif // D3G27
                );
                }

            Axz_qx_t30 = F_M_I_SCALE * Axz_qx_t30;
            Axz_qy_t30 = F_M_I_SCALE * Axz_qy_t30;
            Axz_qz_t30 = F_M_I_SCALE * Axz_qz_t30;

            #include COLREC_AXZ_COLLISION

    #endif //A_XZ_DIST
    #ifdef A_YY_DIST
                      dfloat Ayy_qx_t30 = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_YY_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
            dfloat Ayy_qy_t30 = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_YY_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
            dfloat Ayy_qz_t30 = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_YY_CZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];

            #include COLREC_AYY_RECONSTRUCTION

            activeBarrier.arrive_and_wait();

            #include "fragments/convection_diffusion_streaming.inc"
            /* load pop from global in cover nodes */
            {
                #include "fragments/conformationTransport/popLoad_Ayy.inc"
                }

            if(nodeType != BULK){
                 #include CASE_AYY_BC_DEF
                }else{
                AyyVar = gNode[0] + gNode[1] + gNode[2] + gNode[3] + gNode[4] + gNode[5] + gNode[6]
        #ifdef D3G19
                    + gNode[7] + gNode[8] + gNode[9] + gNode[10] + gNode[11] + gNode[12] + gNode[13] + gNode[14] + gNode[15] + gNode[16] + gNode[17] + gNode[18]
        #endif // D3G19
        #ifdef D3G27
                    + gNode[7] + gNode[8] + gNode[9] + gNode[10] + gNode[11] + gNode[12] + gNode[13] + gNode[14] + gNode[15] + gNode[16] + gNode[17] + gNode[18]
                    + gNode[19] + gNode[20] + gNode[21] + gNode[22] + gNode[23] + gNode[24] + gNode[25] + gNode[26]
        #endif // D3G27
                ;
                AyyVar = AyyVar + Gyy;

                Ayy_qx_t30 = (gNode[1] - gNode[2]
        #ifdef D3G19
                    + gNode[7] - gNode[ 8] + gNode[ 9] - gNode[10] + gNode[13] - gNode[14] + gNode[15] - gNode[16]
        #endif // D3G19
        #ifdef D3G27
                    + gNode[7] - gNode[ 8] + gNode[ 9] - gNode[10] + gNode[13] - gNode[14] + gNode[15] - gNode[16]
                    + gNode[19] - gNode[20] + gNode[21] - gNode[22] + gNode[23] - gNode[24] - gNode[25] + gNode[26]
        #endif // D3G27
                );
                Ayy_qy_t30 = (gNode[3] - gNode[4]
        #ifdef D3G19
                    + gNode[7] - gNode[ 8] + gNode[11] - gNode[12] + gNode[14] - gNode[13] + gNode[17] - gNode[18]
        #endif // D3G19
        #ifdef D3G27
                    + gNode[7] - gNode[ 8] + gNode[11] - gNode[12] + gNode[14] - gNode[13] + gNode[17] - gNode[18]
                    + gNode[19] - gNode[20] + gNode[21] - gNode[22] - gNode[23] + gNode[24] + gNode[25] - gNode[26]
        #endif // D3G27
                );
                Ayy_qz_t30 = (gNode[5] - gNode[6]
        #ifdef D3G19
                    + gNode[9] - gNode[10] + gNode[11] - gNode[12] + gNode[16] - gNode[15] + gNode[18] - gNode[17]
        #endif // D3G19
        #ifdef D3G27
                    + gNode[9] - gNode[10] + gNode[11] - gNode[12] + gNode[16] - gNode[15] + gNode[18] - gNode[17]
                    + gNode[19] - gNode[20] - gNode[21] + gNode[22] + gNode[23] - gNode[24] + gNode[25] - gNode[26]
        #endif // D3G27
                );
                }

            Ayy_qx_t30 = F_M_I_SCALE * Ayy_qx_t30;
            Ayy_qy_t30 = F_M_I_SCALE * Ayy_qy_t30;
            Ayy_qz_t30 = F_M_I_SCALE * Ayy_qz_t30;

            #include COLREC_AYY_COLLISION

    #endif //A_YY_DIST
    #ifdef A_YZ_DIST
            dfloat Ayz_qx_t30 = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_YZ_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
            dfloat Ayz_qy_t30 = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_YZ_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
            dfloat Ayz_qz_t30 = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_YZ_CZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];

            #include COLREC_AYZ_RECONSTRUCTION

            activeBarrier.arrive_and_wait();

            #include "fragments/convection_diffusion_streaming.inc"
            /* load pop from global in cover nodes */
            {
                #include "fragments/conformationTransport/popLoad_Ayz.inc"
                }

            if(nodeType != BULK){
                 #include CASE_AYZ_BC_DEF
                }else{
                AyzVar = gNode[0] + gNode[1] + gNode[2] + gNode[3] + gNode[4] + gNode[5] + gNode[6]
        #ifdef D3G19
                    + gNode[7] + gNode[8] + gNode[9] + gNode[10] + gNode[11] + gNode[12] + gNode[13] + gNode[14] + gNode[15] + gNode[16] + gNode[17] + gNode[18]
        #endif // D3G19
        #ifdef D3G27
                    + gNode[7] + gNode[8] + gNode[9] + gNode[10] + gNode[11] + gNode[12] + gNode[13] + gNode[14] + gNode[15] + gNode[16] + gNode[17] + gNode[18]
                    + gNode[19] + gNode[20] + gNode[21] + gNode[22] + gNode[23] + gNode[24] + gNode[25] + gNode[26]
        #endif // D3G27
                ;
                AyzVar = AyzVar + Gyz;

                Ayz_qx_t30 = (gNode[1] - gNode[2]
        #ifdef D3G19
                    + gNode[7] - gNode[ 8] + gNode[ 9] - gNode[10] + gNode[13] - gNode[14] + gNode[15] - gNode[16]
        #endif // D3G19
        #ifdef D3G27
                    + gNode[7] - gNode[ 8] + gNode[ 9] - gNode[10] + gNode[13] - gNode[14] + gNode[15] - gNode[16]
                    + gNode[19] - gNode[20] + gNode[21] - gNode[22] + gNode[23] - gNode[24] - gNode[25] + gNode[26]
        #endif // D3G27
                );
                Ayz_qy_t30 = (gNode[3] - gNode[4]
        #ifdef D3G19
                    + gNode[7] - gNode[ 8] + gNode[11] - gNode[12] + gNode[14] - gNode[13] + gNode[17] - gNode[18]
        #endif // D3G19
        #ifdef D3G27
                    + gNode[7] - gNode[ 8] + gNode[11] - gNode[12] + gNode[14] - gNode[13] + gNode[17] - gNode[18]
                    + gNode[19] - gNode[20] + gNode[21] - gNode[22] - gNode[23] + gNode[24] + gNode[25] - gNode[26]
        #endif // D3G27
                );
                Ayz_qz_t30 = (gNode[5] - gNode[6]
        #ifdef D3G19
                    + gNode[9] - gNode[10] + gNode[11] - gNode[12] + gNode[16] - gNode[15] + gNode[18] - gNode[17]
        #endif // D3G19
        #ifdef D3G27
                    + gNode[9] - gNode[10] + gNode[11] - gNode[12] + gNode[16] - gNode[15] + gNode[18] - gNode[17]
                    + gNode[19] - gNode[20] - gNode[21] + gNode[22] + gNode[23] - gNode[24] + gNode[25] - gNode[26]
        #endif // D3G27
                );
                }

            Ayz_qx_t30 = F_M_I_SCALE * Ayz_qx_t30;
            Ayz_qy_t30 = F_M_I_SCALE * Ayz_qy_t30;
            Ayz_qz_t30 = F_M_I_SCALE * Ayz_qz_t30;

            #include COLREC_AYZ_COLLISION

    #endif //A_YZ_DIST
    #ifdef A_ZZ_DIST
            dfloat Azz_qx_t30 = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_ZZ_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
            dfloat Azz_qy_t30 = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_ZZ_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
            dfloat Azz_qz_t30 = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_ZZ_CZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];

            #include COLREC_AZZ_RECONSTRUCTION

            activeBarrier.arrive_and_wait();

            #include "fragments/convection_diffusion_streaming.inc"
            /* load pop from global in cover nodes */
            {
                #include "fragments/conformationTransport/popLoad_Azz.inc"
                }

            if(nodeType != BULK){
                 #include CASE_AZZ_BC_DEF
                }else{
                AzzVar = gNode[0] + gNode[1] + gNode[2] + gNode[3] + gNode[4] + gNode[5] + gNode[6]
        #ifdef D3G19
                    + gNode[7] + gNode[8] + gNode[9] + gNode[10] + gNode[11] + gNode[12] + gNode[13] + gNode[14] + gNode[15] + gNode[16] + gNode[17] + gNode[18]
        #endif // D3G19
        #ifdef D3G27
                    + gNode[7] + gNode[8] + gNode[9] + gNode[10] + gNode[11] + gNode[12] + gNode[13] + gNode[14] + gNode[15] + gNode[16] + gNode[17] + gNode[18]
                    + gNode[19] + gNode[20] + gNode[21] + gNode[22] + gNode[23] + gNode[24] + gNode[25] + gNode[26]
        #endif // D3G27
                ;
                AzzVar = AzzVar + Gzz;

                Azz_qx_t30 = (gNode[1] - gNode[2]
        #ifdef D3G19
                    + gNode[7] - gNode[ 8] + gNode[ 9] - gNode[10] + gNode[13] - gNode[14] + gNode[15] - gNode[16]
        #endif // D3G19
        #ifdef D3G27
                    + gNode[7] - gNode[ 8] + gNode[ 9] - gNode[10] + gNode[13] - gNode[14] + gNode[15] - gNode[16]
                    + gNode[19] - gNode[20] + gNode[21] - gNode[22] + gNode[23] - gNode[24] - gNode[25] + gNode[26]
        #endif // D3G27
                );
                Azz_qy_t30 = (gNode[3] - gNode[4]
        #ifdef D3G19
                    + gNode[7] - gNode[ 8] + gNode[11] - gNode[12] + gNode[14] - gNode[13] + gNode[17] - gNode[18]
        #endif // D3G19
        #ifdef D3G27
                    + gNode[7] - gNode[ 8] + gNode[11] - gNode[12] + gNode[14] - gNode[13] + gNode[17] - gNode[18]
                    + gNode[19] - gNode[20] + gNode[21] - gNode[22] - gNode[23] + gNode[24] + gNode[25] - gNode[26]
        #endif // D3G27
                );
                Azz_qz_t30 = (gNode[5] - gNode[6]
        #ifdef D3G19
                    + gNode[9] - gNode[10] + gNode[11] - gNode[12] + gNode[16] - gNode[15] + gNode[18] - gNode[17]
        #endif // D3G19
        #ifdef D3G27
                    + gNode[9] - gNode[10] + gNode[11] - gNode[12] + gNode[16] - gNode[15] + gNode[18] - gNode[17]
                    + gNode[19] - gNode[20] - gNode[21] + gNode[22] + gNode[23] - gNode[24] + gNode[25] - gNode[26]
        #endif // D3G27
                );
                }

            Azz_qx_t30 = F_M_I_SCALE * Azz_qx_t30;
            Azz_qy_t30 = F_M_I_SCALE * Azz_qy_t30;
            Azz_qz_t30 = F_M_I_SCALE * Azz_qz_t30;

            #include COLREC_AZZ_COLLISION

    #endif //A_ZZ_DIST


    #endif //CONVECTION_DIFFUSION_TRANSPORT

    //save populations in shared memory
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  0)] = pop[ 1];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  1)] = pop[ 2];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  2)] = pop[ 3];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  3)] = pop[ 4];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  4)] = pop[ 5];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  5)] = pop[ 6];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  6)] = pop[ 7];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  7)] = pop[ 8];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  8)] = pop[ 9];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  9)] = pop[10];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 10)] = pop[11];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 11)] = pop[12];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 12)] = pop[13];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 13)] = pop[14];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 14)] = pop[15];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 15)] = pop[16];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 16)] = pop[17];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 17)] = pop[18];
    #ifdef D3Q27
        s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 18)] = pop[19];
        s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 19)] = pop[20];
        s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 20)] = pop[21];
        s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 21)] = pop[22];
        s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 22)] = pop[23];
        s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 23)] = pop[24];
        s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 24)] = pop[25];
        s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 25)] = pop[26];
    #endif //D3Q27


    //sync threads of the block so all populations are saved
    activeBarrier.arrive_and_wait();

    /* pull */

    pop[ 1] = s_pop[idxPopBlock(xm1, threadIdx.y, threadIdx.z, 0)];
    pop[ 2] = s_pop[idxPopBlock(xp1, threadIdx.y, threadIdx.z, 1)];
    pop[ 3] = s_pop[idxPopBlock(threadIdx.x, ym1, threadIdx.z, 2)];
    pop[ 4] = s_pop[idxPopBlock(threadIdx.x, yp1, threadIdx.z, 3)];
    pop[ 5] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, zm1, 4)];
    pop[ 6] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, zp1, 5)];
    pop[ 7] = s_pop[idxPopBlock(xm1, ym1, threadIdx.z, 6)];
    pop[ 8] = s_pop[idxPopBlock(xp1, yp1, threadIdx.z, 7)];
    pop[ 9] = s_pop[idxPopBlock(xm1, threadIdx.y, zm1, 8)];
    pop[10] = s_pop[idxPopBlock(xp1, threadIdx.y, zp1, 9)];
    pop[11] = s_pop[idxPopBlock(threadIdx.x, ym1, zm1, 10)];
    pop[12] = s_pop[idxPopBlock(threadIdx.x, yp1, zp1, 11)];
    pop[13] = s_pop[idxPopBlock(xm1, yp1, threadIdx.z, 12)];
    pop[14] = s_pop[idxPopBlock(xp1, ym1, threadIdx.z, 13)];
    pop[15] = s_pop[idxPopBlock(xm1, threadIdx.y, zp1, 14)];
    pop[16] = s_pop[idxPopBlock(xp1, threadIdx.y, zm1, 15)];
    pop[17] = s_pop[idxPopBlock(threadIdx.x, ym1, zp1, 16)];
    pop[18] = s_pop[idxPopBlock(threadIdx.x, yp1, zm1, 17)];
    #ifdef D3Q27
        pop[19] = s_pop[idxPopBlock(xm1, ym1, zm1, 18)];
        pop[20] = s_pop[idxPopBlock(xp1, yp1, zp1, 19)];
        pop[21] = s_pop[idxPopBlock(xm1, ym1, zp1, 20)];
        pop[22] = s_pop[idxPopBlock(xp1, yp1, zm1, 21)];
        pop[23] = s_pop[idxPopBlock(xm1, yp1, zm1, 22)];
        pop[24] = s_pop[idxPopBlock(xp1, ym1, zp1, 23)];
        pop[25] = s_pop[idxPopBlock(xp1, ym1, zm1, 24)];
        pop[26] = s_pop[idxPopBlock(xm1, yp1, zp1, 25)];
    #endif //D3Q27

    /* load pop from global in cover nodes */
    #include "fragments/popLoad.inc"

    dfloat invRho;

    if(nodeType != BULK){
        #ifdef CURVED_BOUNDARY_CONDITION
            dfloat ux0 = 0.0_df;
            dfloat uy0 = 0.0_df;
            dfloat uz0 = 0.0_df;

            // Retrieve wall velocity from curved boundary condition data
            CurvedBoundary* curvedBC = d_curvedBC[idxScalarBlock(threadIdx.x, threadIdx.y, threadIdx.z,blockIdx.x, blockIdx.y, blockIdx.z)];
            if(curvedBC != nullptr){
                ux0 = curvedBC->vel.x;
                uy0 = curvedBC->vel.y;
                uz0 = curvedBC->vel.z;

                // Concave and convex walls use the same reconstruction
                // equations. The original tag remains in dNodeType for
                // geometry setup; only this local switch key is normalized.
                nodeType = curvedBoundaryReconstructionNodeType(nodeType);
                }
        #endif //CURVED_BOUNDARY_CONDITION

        #include CASE_BC_DEF

        invRho = 1.0_df / rhoVar;
    }else{

        //calculate streaming moments
        #ifdef D3Q19
            //equation3
            rhoVar = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
            invRho = 1 / rhoVar;
            //equation4 + force correction
            ux_t30 = ((pop[1] - pop[2] + pop[7] - pop[ 8] + pop[ 9] - pop[10] + pop[13] - pop[14] + pop[15] - pop[16])) * invRho;
            uy_t30 = ((pop[3] - pop[4] + pop[7] - pop[ 8] + pop[11] - pop[12] + pop[14] - pop[13] + pop[17] - pop[18])) * invRho;
            uz_t30 = ((pop[5] - pop[6] + pop[9] - pop[10] + pop[11] - pop[12] + pop[16] - pop[15] + pop[18] - pop[17])) * invRho;

            //equation5
            m_xx_t45 = (pop[1] + pop[2] + pop[7] + pop[8] + pop[9] + pop[10] + pop[13] + pop[14] + pop[15] + pop[16]) * invRho - cs2;
            m_xy_t90 = (pop[7] - pop[13] + pop[8] - pop[14])* invRho;
            m_xz_t90 = (pop[9] - pop[15] + pop[10] - pop[16])* invRho;
            m_yy_t45 = (pop[3] + pop[4] + pop[7] + pop[8] + pop[11] + pop[12] + pop[13] + pop[14] + pop[17] + pop[18]) * invRho - cs2;
            m_yz_t90 = (pop[11] - pop[17] + pop[12] - pop[18])* invRho;
            m_zz_t45 = (pop[5] + pop[6] + pop[9] + pop[10] + pop[11] + pop[12] + pop[15] + pop[16] + pop[17] + pop[18]) * invRho - cs2;


        #endif //D3Q19
        #ifdef D3Q27
            rhoVar = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18] + pop[19] + pop[20] + pop[21] + pop[22] + pop[23] + pop[24] + pop[25] + pop[26];
            invRho = 1 / rhoVar;
            ux_t30 = ((pop[1] + pop[7] + pop[9] + pop[13] + pop[15] + pop[19] + pop[21] + pop[23] + pop[26])  - (pop[ 2] + pop[ 8] + pop[10] + pop[14] + pop[16] + pop[20] + pop[22] + pop[24] + pop[25])) * invRho;
            uy_t30 = ((pop[3] + pop[7] + pop[11] + pop[14] + pop[17] + pop[19] + pop[21] + pop[24] + pop[25]) - (pop[ 4] + pop[ 8] + pop[12] + pop[13] + pop[18] + pop[20] + pop[22] + pop[23] + pop[26])) * invRho;
            uz_t30 = ((pop[5] + pop[9] + pop[11] + pop[16] + pop[18] + pop[19] + pop[22] + pop[23] + pop[25]) - (pop[ 6] + pop[10] + pop[12] + pop[15] + pop[17] + pop[20] + pop[21] + pop[24] + pop[26])) * invRho;

            m_xx_t45 = (pop[ 1] + pop[ 2] + pop[ 7] + pop[ 8] + pop[ 9] + pop[10] + pop[13] + pop[14] + pop[15] + pop[16] + pop[19] + pop[20] + pop[21] + pop[22] + pop[23] + pop[24] + pop[25] + pop[26]) * invRho - cs2;
            m_xy_t90 = (((pop[ 7] + pop[ 8] + pop[19] + pop[20] + pop[21] + pop[22]) - (pop[13] + pop[14] + pop[23] + pop[24] + pop[25] + pop[26])) )* invRho;
            m_xz_t90 = (((pop[ 9] + pop[10] + pop[19] + pop[20] + pop[23] + pop[24]) - (pop[15] + pop[16] + pop[21] + pop[22] + pop[25] + pop[26])) )* invRho;
            m_yy_t45 = (pop[ 3] + pop[ 4] + pop[ 7] + pop[ 8] + pop[11] + pop[12] + pop[13] + pop[14] + pop[17] + pop[18] + pop[19] + pop[20] + pop[21] + pop[22] + pop[23] + pop[24] + pop[25] + pop[26]) * invRho - cs2;
            m_yz_t90 = (((pop[11] + pop[12] + pop[19] + pop[20] + pop[25] + pop[26]) - (pop[17] + pop[18] + pop[21] + pop[22] + pop[23] + pop[24])))* invRho;
            m_zz_t45 = (pop[ 5] + pop[ 6] + pop[ 9] + pop[10] + pop[11] + pop[12] + pop[15] + pop[16] + pop[17] + pop[18] + pop[19] + pop[20] + pop[21] + pop[22] + pop[23] + pop[24] + pop[25] + pop[26]) * invRho - cs2;
        #endif //D3Q27
    }

    // multiply moments by as2 -- as4*0.5_df -- as4 - add correction to m_alpha_beta
    ux_t30 = F_M_I_SCALE * ux_t30;
    uy_t30 = F_M_I_SCALE * uy_t30;
    uz_t30 = F_M_I_SCALE * uz_t30;

    m_xx_t45 = F_M_II_SCALE * (m_xx_t45);
    m_xy_t90 = F_M_IJ_SCALE * (m_xy_t90);
    m_xz_t90 = F_M_IJ_SCALE * (m_xz_t90);
    m_yy_t45 = F_M_II_SCALE * (m_yy_t45);
    m_yz_t90 = F_M_IJ_SCALE * (m_yz_t90);
    m_zz_t45 = F_M_II_SCALE * (m_zz_t45);


    #ifdef DENSITY_CORRECTION
        rhoVar -= (d_mean_rho[0]) ;
        invRho = 1/rhoVar;
    #endif // DENSITY_CORRECTION
    #ifdef THERMAL_MODEL //Boussinesq Approximation
        if(nodeType == BULK && T_BOUYANCY){
                L_Fx += gravity_vector[0] * T_gravity_t_beta * RHO_0*((cVar-T_REFERENCE));
                L_Fy += gravity_vector[1] * T_gravity_t_beta * RHO_0*((cVar-T_REFERENCE));
                L_Fz += gravity_vector[2] * T_gravity_t_beta * RHO_0*((cVar-T_REFERENCE));
            }

    #endif //THERMAL_MODEL

    #ifdef COMPUTE_SHEAR
        const dfloat S_XX = rhoVar * (m_xx_t45/F_M_II_SCALE - ux_t30*ux_t30/(F_M_I_SCALE*F_M_I_SCALE));
        const dfloat S_YY = rhoVar * (m_yy_t45/F_M_II_SCALE - uy_t30*uy_t30/(F_M_I_SCALE*F_M_I_SCALE));
        const dfloat S_ZZ = rhoVar * (m_zz_t45/F_M_II_SCALE - uz_t30*uz_t30/(F_M_I_SCALE*F_M_I_SCALE));
        const dfloat S_XY = rhoVar * (m_xy_t90/F_M_IJ_SCALE - ux_t30*uy_t30/(F_M_I_SCALE*F_M_I_SCALE));
        const dfloat S_XZ = rhoVar * (m_xz_t90/F_M_IJ_SCALE - ux_t30*uz_t30/(F_M_I_SCALE*F_M_I_SCALE));
        const dfloat S_YZ = rhoVar * (m_yz_t90/F_M_IJ_SCALE - uy_t30*uz_t30/(F_M_I_SCALE*F_M_I_SCALE));

        const dfloat uFxxd2 = ux_t30*L_Fx/F_M_I_SCALE; // d2 = uFxx Divided by two
        const dfloat uFyyd2 = uy_t30*L_Fy/F_M_I_SCALE;
        const dfloat uFzzd2 = uz_t30*L_Fz/F_M_I_SCALE;
        const dfloat uFxyd2 = (ux_t30*L_Fy + uy_t30*L_Fx) / (2.0_df*F_M_I_SCALE);
        const dfloat uFxzd2 = (ux_t30*L_Fz + uz_t30*L_Fx) / (2.0_df*F_M_I_SCALE);
        const dfloat uFyzd2 = (uy_t30*L_Fz + uz_t30*L_Fy) / (2.0_df*F_M_I_SCALE);

        const dfloat auxStressMag = sqrt(0.5_df * ( (S_XX + uFxxd2) * (S_XX + uFxxd2) +(S_YY + uFyyd2) * (S_YY + uFyyd2) + (S_ZZ + uFzzd2) * (S_ZZ + uFzzd2) + 2 * ((S_XY + uFxyd2) * (S_XY + uFxyd2) + (S_XZ + uFxzd2) * (S_XZ + uFxzd2) + (S_YZ + uFyzd2) * (S_YZ + uFyzd2))));

    #endif //COMPUTE_SHEAR
    // MOMENTS DETERMINED, COMPUTE OMEGA IF NON-NEWTONIAN FLUID
    #if defined(OMEGA_FIELD)
    #ifndef LAMBDA_DIST
              dfloat lambdaVar = 1.0_df;
    #endif // LAMBDA_DIST
    #ifdef NON_NEWTONIAN_FLUID
                dfloat gammaDot = omegaVar * auxStressMag * as2;
    #if defined(PHI_DIST)

                    // h_phi is pre-computed from the early phi load - avoids redundant global memory read.
                    // Compute phase-specific omegas, convert to apparent viscosities, then blend back.
                    const dfloat omega_eps = 1.0e-12_df;

                    dfloat omegaA = calcOmega(phasePropsA.nnf, omegaVar, auxStressMag, lambdaVar, gammaDot, rhoVar, step);
                    dfloat omegaB = calcOmega(phasePropsB.nnf, omegaVar, auxStressMag, lambdaVar, gammaDot, rhoVar, step);

                    dfloat tauA = (omegaA > omega_eps) ? (1.0_df / omegaA) : (1.0_df / omega_eps);
                    dfloat tauB = (omegaB > omega_eps) ? (1.0_df / omegaB) : (1.0_df / omega_eps);

                    dfloat muA = (tauA - 0.5_df) * PHI_RHO_PHASE1 * cs2;
                    dfloat muB = (tauB - 0.5_df) * PHI_RHO_PHASE2 * cs2;

                    dfloat muMix = interpolateProperty(muA, muB, h_phi);
                    dfloat tauMix = muMix / (rho_pf * cs2) + 0.5_df;
                    omegaVar = 1.0_df / fmax(tauMix, omega_eps);
    #else
                    omegaVar = calcOmega(phasePropsA.nnf, omegaVar, auxStressMag, lambdaVar, gammaDot, rhoVar, step);
    #endif // defined(PHI_DIST)
    #endif //NON_NEWTONIAN_FLUID

    #ifdef LES_MODEL
                dfloat tau_t = calcTau_les(omegaVar, auxStressMag,step);
                omegaVar = 1.0_df/(TAU + tau_t);
    #endif //LES_MODEL

            //Compute new auxiliary variables
            t_omegaVar = 1.0_df - omegaVar;
            tt_omegaVar = 1.0_df - omegaVar*0.5_df;
            omegaVar_d2 = omegaVar*0.5_df;
            tt_omega_t3 = tt_omegaVar * 3.0_df;
    #endif //OMEGA_FIELD

    // zero forces in directions that are not fluid
    if (((nodeType & EAST)  == EAST)  || ((nodeType & WEST)  == WEST)) {
        L_Fx = 0;
    }

    if (((nodeType & NORTH) == NORTH) || ((nodeType & SOUTH) == SOUTH)) {
        L_Fy = 0;
    }

    if (((nodeType & FRONT) == FRONT) || ((nodeType & BACK)  == BACK)) {
        L_Fz = 0;
    }

    // COLLIDE
    #include COLREC_COLLISION


    //calculate post collision populations
    #include COLREC_RECONSTRUCTION


    /* write to global mom */

    #define STORE_HYDRO_MOMENT(moment, value) \
        storeMoment(fMom, idxMom(threadIdx.x, threadIdx.y, threadIdx.z, moment, \
            blockIdx.x, blockIdx.y, blockIdx.z), moment, value)
    STORE_HYDRO_MOMENT(M_RHO_INDEX, rhoVar - RHO_0);
    STORE_HYDRO_MOMENT(M_UX_INDEX, ux_t30);
    STORE_HYDRO_MOMENT(M_UY_INDEX, uy_t30);
    STORE_HYDRO_MOMENT(M_UZ_INDEX, uz_t30);
    STORE_HYDRO_MOMENT(M_MXX_INDEX, m_xx_t45);
    STORE_HYDRO_MOMENT(M_MXY_INDEX, m_xy_t90);
    STORE_HYDRO_MOMENT(M_MXZ_INDEX, m_xz_t90);
    STORE_HYDRO_MOMENT(M_MYY_INDEX, m_yy_t45);
    STORE_HYDRO_MOMENT(M_MYZ_INDEX, m_yz_t90);
    STORE_HYDRO_MOMENT(M_MZZ_INDEX, m_zz_t45);
    #undef STORE_HYDRO_MOMENT

    #ifdef OMEGA_FIELD
        fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_OMEGA_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = omegaVar;
    #endif //OMEGA_FIELD


    if(save){
        #ifdef BC_FORCES
        //update boundary forces
            d_BC_Fx[idxScalarBlock(threadIdx.x, threadIdx.y, threadIdx.z,blockIdx.x, blockIdx.y, blockIdx.z)] = (L_BC_Fx);
            d_BC_Fy[idxScalarBlock(threadIdx.x, threadIdx.y, threadIdx.z,blockIdx.x, blockIdx.y, blockIdx.z)] = (L_BC_Fy);
            d_BC_Fz[idxScalarBlock(threadIdx.x, threadIdx.y, threadIdx.z,blockIdx.x, blockIdx.y, blockIdx.z)] = (L_BC_Fz);
        #endif //BC_FORCES
        #ifdef SAVE_LOCAL_FORCES
        //save total local body force (includes external + phase + thermal contributions)
            d_Local_Fx[idxScalarBlock(threadIdx.x, threadIdx.y, threadIdx.z,blockIdx.x, blockIdx.y, blockIdx.z)] = (L_Fx);
            d_Local_Fy[idxScalarBlock(threadIdx.x, threadIdx.y, threadIdx.z,blockIdx.x, blockIdx.y, blockIdx.z)] = (L_Fy);
            d_Local_Fz[idxScalarBlock(threadIdx.x, threadIdx.y, threadIdx.z,blockIdx.x, blockIdx.y, blockIdx.z)] = (L_Fz);
        #ifdef SECOND_DIST
                d_Source_C[idxScalarBlock(threadIdx.x, threadIdx.y, threadIdx.z,blockIdx.x, blockIdx.y, blockIdx.z)] = T_Q_INTERNAL_D_Cp;
        #endif // SECOND_DIST
        #endif //SAVE_LOCAL_FORCES
    }
    #ifdef CONVECTION_DIFFUSION_TRANSPORT
    #ifdef SECOND_DIST

            #include COLREC_G_RECONSTRUCTION

            {
                #include "fragments/gTransport/g_popSave.inc"
                }

            fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M2_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = cVar;
            fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M2_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = qx_t30;
            fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M2_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = qy_t30;
            fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M2_CZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = qz_t30;

    #endif //SECOND_DIST
    #ifdef PHI_DIST

            #include COLREC_PHI_RECONSTRUCTION

            {
                #include "fragments/phiTransport/phi_popSave.inc"
                }

            fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M3_PHI_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = phiVar;
            fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M3_PX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = phi_qx_t30;
            fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M3_PY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = phi_qy_t30;
            fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M3_PZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = phi_qz_t30;

    #endif //PHI_DIST
    #ifdef LAMBDA_DIST
            lambda_udx_t30 = LAMBDA_DIFF_FLUC_COEF * (lambda_qx_t30*invLambda - ux_t30);
            lambda_udy_t30 = LAMBDA_DIFF_FLUC_COEF * (lambda_qy_t30*invLambda - uy_t30);
            lambda_udz_t30 = LAMBDA_DIFF_FLUC_COEF * (lambda_qz_t30*invLambda - uz_t30);

            #include COLREC_LAMBDA_RECONSTRUCTION

            {
                #include "fragments/lambdaTransport/lambda_popSave.inc"
                }

            fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M4_LAMBDA_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = lambdaVar;
            fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M4_LX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = lambda_qx_t30;
            fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M4_LY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = lambda_qy_t30;
            fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M4_LZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = lambda_qz_t30;

    #endif //LAMBDA_DIST
    #ifdef A_XX_DIST

            #include COLREC_AXX_RECONSTRUCTION

            {
                #include "fragments/conformationTransport/popSave_Axx.inc"
                }

            fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_XX_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = AxxVar;
            fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_XX_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = Axx_qx_t30;
            fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_XX_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = Axx_qy_t30;
            fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_XX_CZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = Axx_qz_t30;
    #endif //A_XX_DIST
    #ifdef A_XY_DIST

            #include COLREC_AXY_RECONSTRUCTION

            {
                #include "fragments/conformationTransport/popSave_Axy.inc"
                }

            fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_XY_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = AxyVar;
            fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_XY_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = Axy_qx_t30;
            fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_XY_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = Axy_qy_t30;
            fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_XY_CZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = Axy_qz_t30;
    #endif //A_XY_DIST
    #ifdef A_XZ_DIST

            #include COLREC_AXZ_RECONSTRUCTION

            {
                #include "fragments/conformationTransport/popSave_Axz.inc"
                }

            fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_XZ_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = AxzVar;
            fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_XZ_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = Axz_qx_t30;
            fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_XZ_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = Axz_qy_t30;
            fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_XZ_CZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = Axz_qz_t30;
    #endif //A_XZ_DIST
    #ifdef A_YY_DIST

            #include COLREC_AYY_RECONSTRUCTION

            {
                #include "fragments/conformationTransport/popSave_Ayy.inc"
                }

            fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_YY_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = AyyVar;
            fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_YY_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = Ayy_qx_t30;
            fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_YY_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = Ayy_qy_t30;
            fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_YY_CZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = Ayy_qz_t30;
    #endif //A_YY_DIST
    #ifdef A_YZ_DIST

            #include COLREC_AYZ_RECONSTRUCTION

            {
                #include "fragments/conformationTransport/popSave_Ayz.inc"
                }

            fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_YZ_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = AyzVar;
            fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_YZ_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = Ayz_qx_t30;
            fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_YZ_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = Ayz_qy_t30;
            fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_YZ_CZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = Ayz_qz_t30;
    #endif //A_YZ_DIST
    #ifdef A_ZZ_DIST

            #include COLREC_AZZ_RECONSTRUCTION

            {
                #include "fragments/conformationTransport/popSave_Azz.inc"
                }

            fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_ZZ_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = AzzVar;
            fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_ZZ_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = Azz_qx_t30;
            fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_ZZ_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = Azz_qy_t30;
            fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_ZZ_CZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = Azz_qz_t30;
    #endif //A_ZZ_DIST
    #endif //CONVECTION_DIFFUSION_TRANSPORT

    #include "fragments/popSave.inc"
    #include "fragments/macroSave.inc"

    //save velocities in the end in order to load next step to compute the gradient
    #ifdef COMPUTE_VEL_GRADIENT_FINITE_DIFFERENCE
    //#include "fragments/velSave.inc"
    //save conformation tensor components in the halo
    //#include "fragments/conformationTransport/confSave.inc"
    #endif //COMPUTE_VEL_GRADIENT_FINITE_DIFFERENCE

}

#ifdef LOCAL_FORCES
__global__
void gpuResetMacroForces(dfloat *fMom){
    const int x = threadIdx.x + blockDim.x * blockIdx.x;
    const int y = threadIdx.y + blockDim.y * blockIdx.y;
    const int z = threadIdx.z + blockDim.z * blockIdx.z;
    if (x >= NX || y >= NY || z >= NZ) return;

    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_FX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = FX;
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_FY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = FY;
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_FZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = FZ;
}
#endif //LOCAL_FORCES


/**/

__global__ void gpuPackMacroHalos(const dfloat *fMom,
    macroInterfaceGPUData halos, size_t localNZ)
{
    const int planeIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (planeIdx >= NX * NY) return;
    const int x = planeIdx % NX, y = planeIdx / NX;
    const int z = blockIdx.y == 0 ? 0 : (int)localNZ - 1;
    auto pack = [&](gpuDirection field, int moment) {
        dfloat *face = blockIdx.y == 0 ? field.Z_0 : field.Z_1;
        const size_t index = idxMom(x % BLOCK_NX, y % BLOCK_NY, z % BLOCK_NZ, moment, x / BLOCK_NX, y / BLOCK_NY, z / BLOCK_NZ);
        face[planeIdx] = loadMoment(fMom, index, moment);
    };
    pack(halos.rho, M_RHO_INDEX);
    pack(halos.ux, M_UX_INDEX);
    pack(halos.uy, M_UY_INDEX);
    pack(halos.uz, M_UZ_INDEX);
    #ifdef SECOND_DIST
        pack(halos.g, M2_C_INDEX);
    #endif // SECOND_DIST
    #ifdef LAMBDA_DIST
        pack(halos.lambda, M4_LAMBDA_INDEX);
    #endif // LAMBDA_DIST
    #ifdef A_XX_DIST
        pack(halos.Axx, A_XX_C_INDEX);
    #endif // A_XX_DIST
    #ifdef A_XY_DIST
        pack(halos.Axy, A_XY_C_INDEX);
    #endif // A_XY_DIST
    #ifdef A_XZ_DIST
        pack(halos.Axz, A_XZ_C_INDEX);
    #endif // A_XZ_DIST
    #ifdef A_YY_DIST
        pack(halos.Ayy, A_YY_C_INDEX);
    #endif // A_YY_DIST
    #ifdef A_YZ_DIST
        pack(halos.Ayz, A_YZ_C_INDEX);
    #endif // A_YZ_DIST
    #ifdef A_ZZ_DIST
        pack(halos.Azz, A_ZZ_C_INDEX);
    #endif // A_ZZ_DIST
    #ifdef PHI_DIST
        pack(halos.phi, M3_PHI_INDEX);
        pack(halos.nx, M3_NX_INDEX);
        pack(halos.ny, M3_NY_INDEX);
        pack(halos.nz, M3_NZ_INDEX);
    #ifdef PHASE_NEEDS_CHEMICAL_POTENTIAL
            pack(halos.mu, M3_MU_INDEX);
    #endif // PHASE_NEEDS_CHEMICAL_POTENTIAL
    #endif // PHI_DIST
}

// ============================================================================
// PHYSICAL-CONFORMATION HALO
// ============================================================================

#ifdef CONFORMATION_PHYSICAL_A_HALO
// This must remain a post-transport kernel. gpuMomCollisionStream reads halo generation n while producing
// conformation generation n+1. Publishing n+1 inside that kernel would require a grid-wide barrier before any block
// could safely consume neighboring faces on the next launch, which ordinary CUDA kernels do not provide.
__global__ void gpuFillPolymerStressHalo(
    const dfloat* fMom, PolymerStressHaloPtrs halo, size_t localNZ)
{
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tz = threadIdx.z;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int bz = blockIdx.z;
    const int x = tx + BLOCK_NX * bx;
    const int y = ty + BLOCK_NY * by;
    const int z = tz + BLOCK_NZ * bz;
    if (x >= NX || y >= NY || z >= localNZ) return;

    const Sym3 stored(
        fMom[idxMom(tx,ty,tz,A_XX_C_INDEX,bx,by,bz)],
        fMom[idxMom(tx,ty,tz,A_XY_C_INDEX,bx,by,bz)],
        fMom[idxMom(tx,ty,tz,A_XZ_C_INDEX,bx,by,bz)],
        fMom[idxMom(tx,ty,tz,A_YY_C_INDEX,bx,by,bz)],
        fMom[idxMom(tx,ty,tz,A_YZ_C_INDEX,bx,by,bz)],
        fMom[idxMom(tx,ty,tz,A_ZZ_C_INDEX,bx,by,bz)]);
    const Sym3 physicalA = vePhysicalAFromStored(stored);

    #define STORE_PHYSICAL_A_FACE(face, base) \
        do { \
            (face)[(base)] = physicalA.xx; \
            (face)[(base) + 1] = physicalA.xy; \
            (face)[(base) + 2] = physicalA.xz; \
            (face)[(base) + 3] = physicalA.yy; \
            (face)[(base) + 4] = physicalA.yz; \
            (face)[(base) + 5] = physicalA.zz; \
            (face)[(base) + 6] = stored.xx; \
            (face)[(base) + 7] = stored.xy; \
            (face)[(base) + 8] = stored.xz; \
            (face)[(base) + 9] = stored.yy; \
            (face)[(base) + 10] = stored.yz; \
            (face)[(base) + 11] = stored.zz; \
        } while (false)

    if (tx == 0) {
        const size_t base = g_idxPolymerStressHaloX(ty, tz, 0, bx, by, bz);
        STORE_PHYSICAL_A_FACE(halo.X_0, base);
    }
    if (tx == BLOCK_NX - 1 || x == NX - 1) {
        const size_t base = g_idxPolymerStressHaloX(ty, tz, 0, bx, by, bz);
        STORE_PHYSICAL_A_FACE(halo.X_1, base);
    }
    if (ty == 0) {
        const size_t base = g_idxPolymerStressHaloY(tx, tz, 0, bx, by, bz);
        STORE_PHYSICAL_A_FACE(halo.Y_0, base);
    }
    if (ty == BLOCK_NY - 1 || y == NY - 1) {
        const size_t base = g_idxPolymerStressHaloY(tx, tz, 0, bx, by, bz);
        STORE_PHYSICAL_A_FACE(halo.Y_1, base);
    }
    if (tz == 0) {
        const size_t base = g_idxPolymerStressHaloZ(tx, ty, 0, bx, by, bz);
        STORE_PHYSICAL_A_FACE(halo.Z_0, base);
    }
    if (tz == BLOCK_NZ - 1 || z + 1 == localNZ) {
        const size_t base = g_idxPolymerStressHaloZ(tx, ty, 0, bx, by, bz);
        STORE_PHYSICAL_A_FACE(halo.Z_1, base);
    }

    #undef STORE_PHYSICAL_A_FACE
}
#endif // CONFORMATION_PHYSICAL_A_HALO
