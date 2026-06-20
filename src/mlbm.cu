#include "mlbm.cuh"

__global__ void gpuMomCollisionStream(DeviceKernelParams params)
{
    // Unpack parameters from struct (passed by value - CUDA optimized!)
    dfloat *fMom = params.fMom;
    unsigned int *dNodeType = params.dNodeType;
    ghostInterfaceData ghostInterface = params.ghostInterface;
    unsigned int step = params.step;
    bool save = params.save;
    
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
        #endif
        #ifdef PHI_DIST
    dfloat* d_Source_Phi = params.d_Source_Phi;
        #endif
        #ifdef LAMBDA_DIST
    dfloat* d_Source_Lambda = params.d_Source_Lambda;
        #endif
        #ifdef CONFORMATION_TENSOR
            #ifdef A_XX_DIST
    dfloat* d_Source_Gxx = params.d_Source_Gxx;
            #endif
            #ifdef A_XY_DIST
    dfloat* d_Source_Gxy = params.d_Source_Gxy;
            #endif
            #ifdef A_XZ_DIST
    dfloat* d_Source_Gxz = params.d_Source_Gxz;
            #endif
            #ifdef A_YY_DIST
    dfloat* d_Source_Gyy = params.d_Source_Gyy;
            #endif
            #ifdef A_YZ_DIST
    dfloat* d_Source_Gyz = params.d_Source_Gyz;
            #endif
            #ifdef A_ZZ_DIST
    dfloat* d_Source_Gzz = params.d_Source_Gzz;
            #endif
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
    #endif
    #endif //NON_NEWTONIAN_FLUID || CONFORMATION_TENSOR

    const int x = threadIdx.x + blockDim.x * blockIdx.x;
    const int y = threadIdx.y + blockDim.y * blockIdx.y;
    const int z = threadIdx.z + blockDim.z * blockIdx.z;
    if (x >= NX || y >= NY || z >= NZ)
        return;
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
    
    const int baseIdx = idxMom(threadIdx.x, threadIdx.y, threadIdx.z, 0, blockIdx.x, blockIdx.y, blockIdx.z);
    const int baseIdxPop = idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  0);

    // Load moments from global memory

    //rho'
    unsigned int nodeType = dNodeType[idxScalarBlock(threadIdx.x, threadIdx.y, threadIdx.z,blockIdx.x, blockIdx.y, blockIdx.z)];
    if (nodeType == 0b11111111)  return;

    dfloat rhoVar = RHO_0 + fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_RHO_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
    dfloat ux_t30     = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_UX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
    dfloat uy_t30     = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_UY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
    dfloat uz_t30     = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_UZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
    dfloat m_xx_t45   = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_MXX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
    dfloat m_xy_t90   = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_MXY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
    dfloat m_xz_t90   = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_MXZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
    dfloat m_yy_t45   = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_MYY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
    dfloat m_yz_t90   = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_MYZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
    dfloat m_zz_t45   = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_MZZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];

  

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
    
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tz = threadIdx.z;
    
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int bz = blockIdx.z;

    const int txm1 = (tx-1+BLOCK_NX)%BLOCK_NX;
    const int txp1 = (tx+1+BLOCK_NX)%BLOCK_NX;

    const int tym1 = (ty-1+BLOCK_NY)%BLOCK_NY;
    const int typ1 = (ty+1+BLOCK_NY)%BLOCK_NY;

    const int tzm1 = (tz-1+BLOCK_NZ)%BLOCK_NZ;
    const int tzp1 = (tz+1+BLOCK_NZ)%BLOCK_NZ;

    const int bxm1 = (bx-1+NUM_BLOCK_X)%NUM_BLOCK_X;
    const int bxp1 = (bx+1+NUM_BLOCK_X)%NUM_BLOCK_X;

    const int bym1 = (by-1+NUM_BLOCK_Y)%NUM_BLOCK_Y;
    const int byp1 = (by+1+NUM_BLOCK_Y)%NUM_BLOCK_Y;

    const int bzm1 = (bz-1+NUM_BLOCK_Z)%NUM_BLOCK_Z;
    const int bzp1 = (bz+1+NUM_BLOCK_Z)%NUM_BLOCK_Z;

    const bool stepParity = (step & 1u);

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

        #ifdef COMPUTE_CONF_GRADIENT_FINITE_DIFFERENCE
            #include "fragments/conformationTransport/conformation_gradient.inc"   
        #endif //COMPUTE_CONF_GRADIENT_FINITE_DIFFERENCE

        #include "fragments/conformationTransport/conformation_evolution.inc"
        #ifdef SAVE_LOCAL_FORCES
        if(save){
            #ifdef A_XX_DIST
            d_Source_Gxx[idxScalarBlock(threadIdx.x, threadIdx.y, threadIdx.z,blockIdx.x, blockIdx.y, blockIdx.z)] = Gxx;
            #endif
            #ifdef A_XY_DIST
            d_Source_Gxy[idxScalarBlock(threadIdx.x, threadIdx.y, threadIdx.z,blockIdx.x, blockIdx.y, blockIdx.z)] = Gxy;
            #endif
            #ifdef A_XZ_DIST
            d_Source_Gxz[idxScalarBlock(threadIdx.x, threadIdx.y, threadIdx.z,blockIdx.x, blockIdx.y, blockIdx.z)] = Gxz;
            #endif
            #ifdef A_YY_DIST
            d_Source_Gyy[idxScalarBlock(threadIdx.x, threadIdx.y, threadIdx.z,blockIdx.x, blockIdx.y, blockIdx.z)] = Gyy;
            #endif
            #ifdef A_YZ_DIST
            d_Source_Gyz[idxScalarBlock(threadIdx.x, threadIdx.y, threadIdx.z,blockIdx.x, blockIdx.y, blockIdx.z)] = Gyz;
            #endif
            #ifdef A_ZZ_DIST
            d_Source_Gzz[idxScalarBlock(threadIdx.x, threadIdx.y, threadIdx.z,blockIdx.x, blockIdx.y, blockIdx.z)] = Gzz;
            #endif
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

            __syncthreads();

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
            #endif
            dfloat phi_qx_t30   = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M3_PX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
            dfloat phi_qy_t30   = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M3_PY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
            dfloat phi_qz_t30   = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M3_PZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];

            #include  COLREC_PHI_RECONSTRUCTION

            __syncthreads();

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
                #endif 
                #ifdef D3G27
                + gNode[7] + gNode[8] + gNode[9] + gNode[10] + gNode[11] + gNode[12] + gNode[13] + gNode[14] + gNode[15] + gNode[16] + gNode[17] + gNode[18]
                + gNode[19] + gNode[20] + gNode[21] + gNode[22] + gNode[23] + gNode[24] + gNode[25] + gNode[26]
                #endif
                ;
                phiVar = phiVar + phiSource; 
                //clamp 
                if(phiVar > PHI_TWO)
                    phiVar = PHI_TWO;
                if(phiVar < PHI_ONE)
                    phiVar = PHI_ONE;

                phi_qx_t30 = (gNode[1] - gNode[2] 
                    #ifdef D3G19
                    + gNode[7] - gNode[ 8] + gNode[ 9] - gNode[10] + gNode[13] - gNode[14] + gNode[15] - gNode[16]
                    #endif
                    #ifdef D3G27
                    + gNode[7] - gNode[ 8] + gNode[ 9] - gNode[10] + gNode[13] - gNode[14] + gNode[15] - gNode[16]
                    + gNode[19] - gNode[20] + gNode[21] - gNode[22] + gNode[23] - gNode[24] - gNode[25] + gNode[26]
                    #endif
                );
                phi_qy_t30 = (gNode[3] - gNode[4]
                    #ifdef D3G19 
                    + gNode[7] - gNode[ 8] + gNode[11] - gNode[12] + gNode[14] - gNode[13] + gNode[17] - gNode[18]
                    #endif
                    #ifdef D3G27 
                    + gNode[7] - gNode[ 8] + gNode[11] - gNode[12] + gNode[14] - gNode[13] + gNode[17] - gNode[18]
                    + gNode[19] - gNode[20] + gNode[21] - gNode[22] - gNode[23] + gNode[24] + gNode[25] - gNode[26]
                    #endif
                );
                phi_qz_t30 = (gNode[5] - gNode[6]
                    #ifdef D3G19 
                    + gNode[9] - gNode[10] + gNode[11] - gNode[12] + gNode[16] - gNode[15] + gNode[18] - gNode[17]
                    #endif
                    #ifdef D3G27 
                    + gNode[9] - gNode[10] + gNode[11] - gNode[12] + gNode[16] - gNode[15] + gNode[18] - gNode[17]
                    + gNode[19] - gNode[20] - gNode[21] + gNode[22] + gNode[23] - gNode[24] + gNode[25] - gNode[26]
                    #endif
                );
            }

            
            phi_qx_t30 = F_M_I_SCALE * phi_qx_t30;
            phi_qy_t30 = F_M_I_SCALE * phi_qy_t30;
            phi_qz_t30 = F_M_I_SCALE * phi_qz_t30;

            #include COLREC_PHI_COLLISION
        #endif //PHI_DIST
        #ifdef LAMBDA_DIST 

            dfloat lambdaVar = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M4_LAMBDA_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];

            // Compute source term using function-based dispatch
            dfloat lambdaSource = computeLambdaSourceFromStress(
                phasePropsA.nnf,
                rhoVar, ux_t30, uy_t30, uz_t30,
                m_xx_t45, m_yy_t45, m_zz_t45,
                m_xy_t90, m_xz_t90, m_yz_t90,
                omegaVar, lambdaVar
            );
            #ifdef SAVE_LOCAL_FORCES
            if(save){ d_Source_Lambda[idxScalarBlock(threadIdx.x, threadIdx.y, threadIdx.z,blockIdx.x, blockIdx.y, blockIdx.z)] = lambdaSource; }
            #endif

            dfloat invLambda = 1.0_df/(lambdaVar);
            dfloat lambda_qx_t30   = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M4_LX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
            dfloat lambda_qy_t30   = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M4_LY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
            dfloat lambda_qz_t30   = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M4_LZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];

            dfloat lambda_udx_t30 = LAMBDA_DIFF_FLUC_COEF * (lambda_qx_t30*invLambda - ux_t30);
            dfloat lambda_udy_t30 = LAMBDA_DIFF_FLUC_COEF * (lambda_qy_t30*invLambda - uy_t30);
            dfloat lambda_udz_t30 = LAMBDA_DIFF_FLUC_COEF * (lambda_qz_t30*invLambda - uz_t30);

            #include  COLREC_LAMBDA_RECONSTRUCTION

            __syncthreads();

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
            dfloat invAxx = 1/AxxVar;
            dfloat Axx_qx_t30 = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_XX_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
            dfloat Axx_qy_t30 = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_XX_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
            dfloat Axx_qz_t30 = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_XX_CZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];

                        dfloat Axx_udx_t30 = CONF_DIFF_FLUC_COEF * (Axx_qx_t30*invAxx - ux_t30);
                        dfloat Axx_udy_t30 = CONF_DIFF_FLUC_COEF * (Axx_qy_t30*invAxx - uy_t30);
                        dfloat Axx_udz_t30 = CONF_DIFF_FLUC_COEF * (Axx_qz_t30*invAxx - uz_t30);

            #include COLREC_AXX_RECONSTRUCTION

            __syncthreads();

            #include "fragments/convection_diffusion_streaming.inc"
            /* load pop from global in cover nodes */
            {
                #include "fragments/conformationTransport/popLoad_Axx.inc"
            }

            if(nodeType != BULK){
                 #include CASE_AXX_BC_DEF
            }else{
                AxxVar = gNode[0] + gNode[1] + gNode[2] + gNode[3] + gNode[4] + gNode[5] + gNode[6] + gNode[7] + gNode[8] + gNode[9] + gNode[10] + gNode[11] + gNode[12] + gNode[13] + gNode[14] + gNode[15] + gNode[16] + gNode[17] + gNode[18];
                AxxVar = AxxVar + Gxx;
                invAxx= 1.0/AxxVar;

                Axx_qx_t30 = F_M_I_SCALE*((gNode[1] - gNode[2] + gNode[7] - gNode[ 8] + gNode[ 9] - gNode[10] + gNode[13] - gNode[14] + gNode[15] - gNode[16]));
                Axx_qy_t30 = F_M_I_SCALE*((gNode[3] - gNode[4] + gNode[7] - gNode[ 8] + gNode[11] - gNode[12] + gNode[14] - gNode[13] + gNode[17] - gNode[18]));
                Axx_qz_t30 = F_M_I_SCALE*((gNode[5] - gNode[6] + gNode[9] - gNode[10] + gNode[11] - gNode[12] + gNode[16] - gNode[15] + gNode[18] - gNode[17]));
            }
        #endif //A_XX_DIST
        #ifdef A_XY_DIST
            dfloat invAxy = 1/AxyVar;
            dfloat Axy_qx_t30 = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_XY_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
            dfloat Axy_qy_t30 = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_XY_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
            dfloat Axy_qz_t30 = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_XY_CZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];

            dfloat Axy_udx_t30 = CONF_DIFF_FLUC_COEF * (Axy_qx_t30*invAxy - ux_t30);
            dfloat Axy_udy_t30 = CONF_DIFF_FLUC_COEF * (Axy_qy_t30*invAxy - uy_t30);
            dfloat Axy_udz_t30 = CONF_DIFF_FLUC_COEF * (Axy_qz_t30*invAxy - uz_t30);

            #include COLREC_AXY_RECONSTRUCTION

            __syncthreads();

            #include "fragments/convection_diffusion_streaming.inc"
            /* load pop from global in cover nodes */
            {
                #include "fragments/conformationTransport/popLoad_Axy.inc"
            }

            if(nodeType != BULK){
                    #include CASE_AXY_BC_DEF
            }else{
                AxyVar = gNode[0] + gNode[1] + gNode[2] + gNode[3] + gNode[4] + gNode[5] + gNode[6] + gNode[7] + gNode[8] + gNode[9] + gNode[10] + gNode[11] + gNode[12] + gNode[13] + gNode[14] + gNode[15] + gNode[16] + gNode[17] + gNode[18];
                AxyVar = AxyVar + Gxy;
                invAxy= 1.0/AxyVar;

                Axy_qx_t30 = F_M_I_SCALE*((gNode[1] - gNode[2] + gNode[7] - gNode[ 8] + gNode[ 9] - gNode[10] + gNode[13] - gNode[14] + gNode[15] - gNode[16]));
                Axy_qy_t30 = F_M_I_SCALE*((gNode[3] - gNode[4] + gNode[7] - gNode[ 8] + gNode[11] - gNode[12] + gNode[14] - gNode[13] + gNode[17] - gNode[18]));
                Axy_qz_t30 = F_M_I_SCALE*((gNode[5] - gNode[6] + gNode[9] - gNode[10] + gNode[11] - gNode[12] + gNode[16] - gNode[15] + gNode[18] - gNode[17]));
            }
        #endif //A_XY_DIST
        #ifdef A_XZ_DIST
            dfloat invAxz = 1/AxzVar;
            dfloat Axz_qx_t30 = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_XZ_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
            dfloat Axz_qy_t30 = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_XZ_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
            dfloat Axz_qz_t30 = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_XZ_CZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];

            dfloat Axz_udx_t30 = CONF_DIFF_FLUC_COEF * (Axz_qx_t30*invAxz - ux_t30);
            dfloat Axz_udy_t30 = CONF_DIFF_FLUC_COEF * (Axz_qy_t30*invAxz - uy_t30);
            dfloat Axz_udz_t30 = CONF_DIFF_FLUC_COEF * (Axz_qz_t30*invAxz - uz_t30);

            #include COLREC_AXZ_RECONSTRUCTION

            __syncthreads();

            #include "fragments/convection_diffusion_streaming.inc"
            /* load pop from global in cover nodes */
            {
                #include "fragments/conformationTransport/popLoad_Axz.inc"
            }

            if(nodeType != BULK){
                    #include CASE_AXZ_BC_DEF
            }else{
                AxzVar = gNode[0] + gNode[1] + gNode[2] + gNode[3] + gNode[4] + gNode[5] + gNode[6] + gNode[7] + gNode[8] + gNode[9] + gNode[10] + gNode[11] + gNode[12] + gNode[13] + gNode[14] + gNode[15] + gNode[16] + gNode[17] + gNode[18];
                AxzVar = AxzVar + Gxz;
                invAxz= 1.0/AxzVar;

                Axz_qx_t30 = F_M_I_SCALE*((gNode[1] - gNode[2] + gNode[7] - gNode[ 8] + gNode[ 9] - gNode[10] + gNode[13] - gNode[14] + gNode[15] - gNode[16]));
                Axz_qy_t30 = F_M_I_SCALE*((gNode[3] - gNode[4] + gNode[7] - gNode[ 8] + gNode[11] - gNode[12] + gNode[14] - gNode[13] + gNode[17] - gNode[18]));
                Axz_qz_t30 = F_M_I_SCALE*((gNode[5] - gNode[6] + gNode[9] - gNode[10] + gNode[11] - gNode[12] + gNode[16] - gNode[15] + gNode[18] - gNode[17]));
            }
        #endif //A_XZ_DIST
        #ifdef A_YY_DIST
            dfloat invAyy = 1/AyyVar;
            dfloat Ayy_qx_t30 = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_YY_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
            dfloat Ayy_qy_t30 = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_YY_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
            dfloat Ayy_qz_t30 = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_YY_CZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];

            dfloat Ayy_udx_t30 = CONF_DIFF_FLUC_COEF * (Ayy_qx_t30*invAyy - ux_t30);
            dfloat Ayy_udy_t30 = CONF_DIFF_FLUC_COEF * (Ayy_qy_t30*invAyy - uy_t30);
            dfloat Ayy_udz_t30 = CONF_DIFF_FLUC_COEF * (Ayy_qz_t30*invAyy - uz_t30);

            #include COLREC_AYY_RECONSTRUCTION

            __syncthreads();

            #include "fragments/convection_diffusion_streaming.inc"
            /* load pop from global in cover nodes */
            {
                #include "fragments/conformationTransport/popLoad_Ayy.inc"
            }

            if(nodeType != BULK){
                    #include CASE_AYY_BC_DEF
            }else{
                AyyVar = gNode[0] + gNode[1] + gNode[2] + gNode[3] + gNode[4] + gNode[5] + gNode[6] + gNode[7] + gNode[8] + gNode[9] + gNode[10] + gNode[11] + gNode[12] + gNode[13] + gNode[14] + gNode[15] + gNode[16] + gNode[17] + gNode[18];
                AyyVar = AyyVar + Gyy;
                invAyy= 1.0/AyyVar;

                Ayy_qx_t30 = F_M_I_SCALE*((gNode[1] - gNode[2] + gNode[7] - gNode[ 8] + gNode[ 9] - gNode[10] + gNode[13] - gNode[14] + gNode[15] - gNode[16]));
                Ayy_qy_t30 = F_M_I_SCALE*((gNode[3] - gNode[4] + gNode[7] - gNode[ 8] + gNode[11] - gNode[12] + gNode[14] - gNode[13] + gNode[17] - gNode[18]));
                Ayy_qz_t30 = F_M_I_SCALE*((gNode[5] - gNode[6] + gNode[9] - gNode[10] + gNode[11] - gNode[12] + gNode[16] - gNode[15] + gNode[18] - gNode[17]));
            }
        #endif //A_YY_DIST
        #ifdef A_YZ_DIST
            dfloat invAyz = 1/AyzVar;
            dfloat Ayz_qx_t30 = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_YZ_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
            dfloat Ayz_qy_t30 = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_YZ_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
            dfloat Ayz_qz_t30 = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_YZ_CZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];

            dfloat Ayz_udx_t30 = CONF_DIFF_FLUC_COEF * (Ayz_qx_t30*invAyz - ux_t30);
            dfloat Ayz_udy_t30 = CONF_DIFF_FLUC_COEF * (Ayz_qy_t30*invAyz - uy_t30);
            dfloat Ayz_udz_t30 = CONF_DIFF_FLUC_COEF * (Ayz_qz_t30*invAyz - uz_t30);

            #include COLREC_AYZ_RECONSTRUCTION

            __syncthreads();

            #include "fragments/convection_diffusion_streaming.inc"
            /* load pop from global in cover nodes */
            {
                #include "fragments/conformationTransport/popLoad_Ayz.inc"
            }

            if(nodeType != BULK){
                    #include CASE_AYZ_BC_DEF
            }else{
                AyzVar = gNode[0] + gNode[1] + gNode[2] + gNode[3] + gNode[4] + gNode[5] + gNode[6] + gNode[7] + gNode[8] + gNode[9] + gNode[10] + gNode[11] + gNode[12] + gNode[13] + gNode[14] + gNode[15] + gNode[16] + gNode[17] + gNode[18];
                AyzVar = AyzVar + Gyz;
                invAyz= 1.0/AyzVar;

                Ayz_qx_t30 = F_M_I_SCALE*((gNode[1] - gNode[2] + gNode[7] - gNode[ 8] + gNode[ 9] - gNode[10] + gNode[13] - gNode[14] + gNode[15] - gNode[16]));
                Ayz_qy_t30 = F_M_I_SCALE*((gNode[3] - gNode[4] + gNode[7] - gNode[ 8] + gNode[11] - gNode[12] + gNode[14] - gNode[13] + gNode[17] - gNode[18]));
                Ayz_qz_t30 = F_M_I_SCALE*((gNode[5] - gNode[6] + gNode[9] - gNode[10] + gNode[11] - gNode[12] + gNode[16] - gNode[15] + gNode[18] - gNode[17]));
            }
        #endif //A_YZ_DIST
        #ifdef A_ZZ_DIST
            dfloat invAzz = 1/AzzVar;
            dfloat Azz_qx_t30 = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_ZZ_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
            dfloat Azz_qy_t30 = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_ZZ_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
            dfloat Azz_qz_t30 = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_ZZ_CZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];

            dfloat Azz_udx_t30 = CONF_DIFF_FLUC_COEF * (Azz_qx_t30*invAzz - ux_t30);
            dfloat Azz_udy_t30 = CONF_DIFF_FLUC_COEF * (Azz_qy_t30*invAzz - uy_t30);
            dfloat Azz_udz_t30 = CONF_DIFF_FLUC_COEF * (Azz_qz_t30*invAzz - uz_t30);

            #include COLREC_AZZ_RECONSTRUCTION

            __syncthreads();

            #include "fragments/convection_diffusion_streaming.inc"
            /* load pop from global in cover nodes */
            {
                #include "fragments/conformationTransport/popLoad_Azz.inc"
            }

            if(nodeType != BULK){
                    #include CASE_AZZ_BC_DEF
            }else{
                AzzVar = gNode[0] + gNode[1] + gNode[2] + gNode[3] + gNode[4] + gNode[5] + gNode[6] + gNode[7] + gNode[8] + gNode[9] + gNode[10] + gNode[11] + gNode[12] + gNode[13] + gNode[14] + gNode[15] + gNode[16] + gNode[17] + gNode[18];
                AzzVar = AzzVar + Gzz;
                invAzz= 1.0/AzzVar;

                Azz_qx_t30 = F_M_I_SCALE*((gNode[1] - gNode[2] + gNode[7] - gNode[ 8] + gNode[ 9] - gNode[10] + gNode[13] - gNode[14] + gNode[15] - gNode[16]));
                Azz_qy_t30 = F_M_I_SCALE*((gNode[3] - gNode[4] + gNode[7] - gNode[ 8] + gNode[11] - gNode[12] + gNode[14] - gNode[13] + gNode[17] - gNode[18]));
                Azz_qz_t30 = F_M_I_SCALE*((gNode[5] - gNode[6] + gNode[9] - gNode[10] + gNode[11] - gNode[12] + gNode[16] - gNode[15] + gNode[18] - gNode[17]));
            }
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
    __syncthreads();

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
            m_xx_t45 = (pop[1] + pop[2] + pop[7] + pop[8] + pop[9] + pop[10] + pop[13] + pop[14] + pop[15] + pop[16])* invRho - cs2;
            m_xy_t90 = (pop[7] - pop[13] + pop[8] - pop[14])* invRho;
            m_xz_t90 = (pop[9] - pop[15] + pop[10] - pop[16])* invRho;
            m_yy_t45 = (pop[3] + pop[4] + pop[7] + pop[8] + pop[11] + pop[12] + pop[13] + pop[14] + pop[17] + pop[18])* invRho - cs2;
            m_yz_t90 = (pop[11] - pop[17] + pop[12] - pop[18])* invRho;
            m_zz_t45 = (pop[5] + pop[6] + pop[9] + pop[10] + pop[11] + pop[12] + pop[15] + pop[16] + pop[17] + pop[18])* invRho - cs2;


        #endif //D3Q19
        #ifdef D3Q27
            rhoVar = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18] + pop[19] + pop[20] + pop[21] + pop[22] + pop[23] + pop[24] + pop[25] + pop[26];
            invRho = 1 / rhoVar;
            ux_t30 = ((pop[1] + pop[7] + pop[9] + pop[13] + pop[15] + pop[19] + pop[21] + pop[23] + pop[26])  - (pop[ 2] + pop[ 8] + pop[10] + pop[14] + pop[16] + pop[20] + pop[22] + pop[24] + pop[25])) * invRho;
            uy_t30 = ((pop[3] + pop[7] + pop[11] + pop[14] + pop[17] + pop[19] + pop[21] + pop[24] + pop[25]) - (pop[ 4] + pop[ 8] + pop[12] + pop[13] + pop[18] + pop[20] + pop[22] + pop[23] + pop[26])) * invRho;
            uz_t30 = ((pop[5] + pop[9] + pop[11] + pop[16] + pop[18] + pop[19] + pop[22] + pop[23] + pop[25]) - (pop[ 6] + pop[10] + pop[12] + pop[15] + pop[17] + pop[20] + pop[21] + pop[24] + pop[26])) * invRho;

            m_xx_t45 = ( (pop[ 1] + pop[ 2] + pop[ 7] + pop[ 8] + pop[ 9] + pop[10]  +  pop[13] + pop[14] + pop[15] + pop[16] + pop[19] + pop[20] + pop[21] + pop[22] + pop[23] + pop[24] + pop[25] + pop[26]))* invRho - cs2;
            m_xy_t90 = (((pop[ 7] + pop[ 8] + pop[19] + pop[20] + pop[21] + pop[22]) - (pop[13] + pop[14] + pop[23] + pop[24] + pop[25] + pop[26])) )* invRho;
            m_xz_t90 = (((pop[ 9] + pop[10] + pop[19] + pop[20] + pop[23] + pop[24]) - (pop[15] + pop[16] + pop[21] + pop[22] + pop[25] + pop[26])) )* invRho;
            m_yy_t45 = ( (pop[ 3] + pop[ 4] + pop[ 7] + pop[ 8] + pop[11] + pop[12]  +  pop[13] + pop[14] + pop[17] + pop[18] + pop[19] + pop[20] + pop[21] + pop[22] + pop[23] + pop[24] + pop[25] + pop[26]))* invRho - cs2;
            m_yz_t90 = (((pop[11] + pop[12] + pop[19] + pop[20] + pop[25] + pop[26]) - (pop[17] + pop[18] + pop[21] + pop[22] + pop[23] + pop[24])))* invRho;
            m_zz_t45 = ( (pop[ 5] + pop[ 6] + pop[ 9] + pop[10] + pop[11] + pop[12]  +  pop[15] + pop[16] + pop[17] + pop[18] + pop[19] + pop[20] + pop[21] + pop[22] + pop[23] + pop[24] + pop[25] + pop[26]))* invRho - cs2;
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

        const dfloat auxStressMag = sqrt(0.5_df * (
            (S_XX + uFxxd2) * (S_XX + uFxxd2) +(S_YY + uFyyd2) * (S_YY + uFyyd2) + (S_ZZ + uFzzd2) * (S_ZZ + uFzzd2) +
            2 * ((S_XY + uFxyd2) * (S_XY + uFxyd2) + (S_XZ + uFxzd2) * (S_XZ + uFxzd2) + (S_YZ + uFyzd2) * (S_YZ + uFyzd2))));

    #endif //COMPUTE_SHEAR
    // MOMENTS DETERMINED, COMPUTE OMEGA IF NON-NEWTONIAN FLUID
    #if defined(OMEGA_FIELD)
            #ifndef LAMBDA_DIST
              dfloat lambdaVar = 1.0_df;
            #endif
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
                #endif
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

    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_RHO_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = rhoVar - RHO_0;

    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_UX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = ux_t30;
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_UY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = uy_t30;
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_UZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = uz_t30;

    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_MXX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = m_xx_t45;
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_MXY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = m_xy_t90;
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_MXZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = m_xz_t90;
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_MYY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = m_yy_t45;
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_MYZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = m_yz_t90;
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_MZZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = m_zz_t45;
    
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
            #endif
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

            Axx_udx_t30 = CONF_DIFF_FLUC_COEF * (Axx_qx_t30*invAxx - ux_t30);
            Axx_udy_t30 = CONF_DIFF_FLUC_COEF * (Axx_qy_t30*invAxx - uy_t30);
            Axx_udz_t30 = CONF_DIFF_FLUC_COEF * (Axx_qz_t30*invAxx - uz_t30);

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

            Axy_udx_t30 = CONF_DIFF_FLUC_COEF * (Axy_qx_t30*invAxy - ux_t30);
            Axy_udy_t30 = CONF_DIFF_FLUC_COEF * (Axy_qy_t30*invAxy - uy_t30);
            Axy_udz_t30 = CONF_DIFF_FLUC_COEF * (Axy_qz_t30*invAxy - uz_t30);

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

            Axz_udx_t30 = CONF_DIFF_FLUC_COEF * (Axz_qx_t30*invAxz - ux_t30);
            Axz_udy_t30 = CONF_DIFF_FLUC_COEF * (Axz_qy_t30*invAxz - uy_t30);
            Axz_udz_t30 = CONF_DIFF_FLUC_COEF * (Axz_qz_t30*invAxz - uz_t30);

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

            Ayy_udx_t30 = CONF_DIFF_FLUC_COEF * (Ayy_qx_t30*invAyy - ux_t30);
            Ayy_udy_t30 = CONF_DIFF_FLUC_COEF * (Ayy_qy_t30*invAyy - uy_t30);
            Ayy_udz_t30 = CONF_DIFF_FLUC_COEF * (Ayy_qz_t30*invAyy - uz_t30);

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

            Ayz_udx_t30 = CONF_DIFF_FLUC_COEF * (Ayz_qx_t30*invAyz - ux_t30);
            Ayz_udy_t30 = CONF_DIFF_FLUC_COEF * (Ayz_qy_t30*invAyz - uy_t30);
            Ayz_udz_t30 = CONF_DIFF_FLUC_COEF * (Ayz_qz_t30*invAyz - uz_t30);

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

            Azz_udx_t30 = CONF_DIFF_FLUC_COEF * (Azz_qx_t30*invAzz - ux_t30);
            Azz_udy_t30 = CONF_DIFF_FLUC_COEF * (Azz_qy_t30*invAzz - uy_t30);
            Azz_udz_t30 = CONF_DIFF_FLUC_COEF * (Azz_qz_t30*invAzz - uz_t30);

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
    if (x >= NX || y >= NY || z >= NZ)
        return;

    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_FX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = FX;
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_FY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = FY;
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_FZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = FZ;
}
#endif //LOCAL_FORCES


#ifdef PHI_DIST
/*
__global__ void gpuComputePhaseNormals(
    dfloat *fMom,
    unsigned int *dNodeType
)
{
    const int x = threadIdx.x + blockDim.x * blockIdx.x;
    const int y = threadIdx.y + blockDim.y * blockIdx.y;
    const int z = threadIdx.z + blockDim.z * blockIdx.z;
    if (x >= NX || y >= NY || z >= NZ) return;

    const int tx = threadIdx.x, ty = threadIdx.y, tz = threadIdx.z;
    const int bx = blockIdx.x,  by = blockIdx.y,  bz = blockIdx.z;

    unsigned int nodeType = dNodeType[idxScalarBlock(tx, ty, tz, bx, by, bz)];
    if (nodeType == 0b11111111) return;
    if (nodeType != BULK) return;

    auto getPhi = [&](int dx, int dy, int dz) -> dfloat {
        #ifdef BC_X_WALL
        int nx = min(NX - 1, max(0, x + dx));
        #else
        int nx = (x + dx + NX) % NX;
        #endif
        #ifdef BC_Y_WALL
        int ny = min(NY - 1, max(0, y + dy));
        #else
        int ny = (y + dy + NY) % NY;
        #endif
        #ifdef BC_Z_WALL
        int nz = min(NZ - 1, max(0, z + dz));
        #else
        int nz = (z + dz + NZ) % NZ;
        #endif
        int ntx = nx % BLOCK_NX, nty = ny % BLOCK_NY, ntz = nz % BLOCK_NZ;
        int nbx = nx / BLOCK_NX, nby = ny / BLOCK_NY, nbz = nz / BLOCK_NZ;
        return fMom[idxMom(ntx, nty, ntz, M3_PHI_INDEX, nbx, nby, nbz)];
    };

    dfloat phi_c = fMom[idxMom(tx, ty, tz, M3_PHI_INDEX, bx, by, bz)];

    // ---- load all neighbors with standard clamping/periodic ----
    dfloat phi_xm1 = getPhi(-1, 0, 0);  dfloat phi_xp1 = getPhi(+1, 0, 0);
    dfloat phi_ym1 = getPhi( 0,-1, 0);  dfloat phi_yp1 = getPhi( 0,+1, 0);
    dfloat phi_zm1 = getPhi( 0, 0,-1);  dfloat phi_zp1 = getPhi( 0, 0,+1);

    dfloat phi_xm1_ym1 = getPhi(-1,-1, 0);  dfloat phi_xp1_ym1 = getPhi(+1,-1, 0);
    dfloat phi_xm1_yp1 = getPhi(-1,+1, 0);  dfloat phi_xp1_yp1 = getPhi(+1,+1, 0);
    dfloat phi_xm1_zm1 = getPhi(-1, 0,-1);  dfloat phi_xp1_zm1 = getPhi(+1, 0,-1);
    dfloat phi_xm1_zp1 = getPhi(-1, 0,+1);  dfloat phi_xp1_zp1 = getPhi(+1, 0,+1);
    dfloat phi_ym1_zm1 = getPhi( 0,-1,-1);  dfloat phi_yp1_zm1 = getPhi( 0,+1,-1);
    dfloat phi_ym1_zp1 = getPhi( 0,-1,+1);  dfloat phi_yp1_zp1 = getPhi( 0,+1,+1);

    #ifdef BC_Y_WALL
    if (y == 1) {           // adjacent to SOUTH wall (y=0)
        phi_ym1     = phi_c;    // axis
        phi_xp1_ym1 = phi_xp1; // edges: project y-1 out, keep lateral offset
        phi_xm1_ym1 = phi_xm1;
        phi_ym1_zp1 = phi_zp1;
        phi_ym1_zm1 = phi_zm1;
    }
    if (y == NY-2) {        // adjacent to NORTH wall (y=NY-1)
        phi_yp1     = phi_c;
        phi_xp1_yp1 = phi_xp1;
        phi_xm1_yp1 = phi_xm1;
        phi_yp1_zp1 = phi_zp1;
        phi_yp1_zm1 = phi_zm1;
    }
    #endif

    #ifdef BC_X_WALL
    if (x == 1) {           // adjacent to WEST wall (x=0)
        phi_xm1     = phi_c;
        phi_xm1_yp1 = phi_yp1;
        phi_xm1_ym1 = phi_ym1;
        phi_xm1_zp1 = phi_zp1;
        phi_xm1_zm1 = phi_zm1;
    }
    if (x == NX-2) {        // adjacent to EAST wall (x=NX-1)
        phi_xp1     = phi_c;
        phi_xp1_yp1 = phi_yp1;
        phi_xp1_ym1 = phi_ym1;
        phi_xp1_zp1 = phi_zp1;
        phi_xp1_zm1 = phi_zm1;
    }
    #endif

    #ifdef BC_Z_WALL
    if (z == 1) {           // adjacent to BACK wall (z=0)
        phi_zm1     = phi_c;
        phi_xp1_zm1 = phi_xp1;
        phi_xm1_zm1 = phi_xm1;
        phi_yp1_zm1 = phi_yp1;
        phi_ym1_zm1 = phi_ym1;
    }
    if (z == NZ-2) {        // adjacent to FRONT wall (z=NZ-1)
        phi_zp1     = phi_c;
        phi_xp1_zp1 = phi_xp1;
        phi_xm1_zp1 = phi_xm1;
        phi_yp1_zp1 = phi_yp1;
        phi_ym1_zp1 = phi_ym1;
    }
    #endif

    // ---- isotropic gradient  ----
    constexpr dfloat w_axis = 1.0_df / 6.0_df;
    constexpr dfloat w_edge = 1.0_df / 12.0_df;

    dfloat dphidx = w_axis * (phi_xp1 - phi_xm1)
                  + w_edge * (phi_xp1_ym1 - phi_xm1_ym1 + phi_xp1_yp1 - phi_xm1_yp1
                            + phi_xp1_zm1 - phi_xm1_zm1 + phi_xp1_zp1 - phi_xm1_zp1);

    dfloat dphidy = w_axis * (phi_yp1 - phi_ym1)
                  + w_edge * (phi_xm1_yp1 - phi_xm1_ym1 + phi_xp1_yp1 - phi_xp1_ym1
                            + phi_yp1_zm1 - phi_ym1_zm1 + phi_yp1_zp1 - phi_ym1_zp1);

    dfloat dphidz = w_axis * (phi_zp1 - phi_zm1)
                  + w_edge * (phi_xm1_zp1 - phi_xm1_zm1 + phi_xp1_zp1 - phi_xp1_zm1
                            + phi_ym1_zp1 - phi_ym1_zm1 + phi_yp1_zp1 - phi_yp1_zm1);

    // ---- isotropic Laplacian ----
    constexpr dfloat w_lap_axis = 1.0_df / 3.0_df;
    constexpr dfloat w_lap_edge = 1.0_df / 6.0_df;
    constexpr dfloat w_lap_zero = -4.0_df;

    dfloat laplacian_phi =
        w_lap_zero * phi_c
        + w_lap_axis * (phi_xp1 + phi_xm1 + phi_yp1 + phi_ym1 + phi_zp1 + phi_zm1)
        + w_lap_edge * (phi_xp1_yp1 + phi_xp1_ym1 + phi_xm1_yp1 + phi_xm1_ym1
                      + phi_xp1_zp1 + phi_xp1_zm1 + phi_xm1_zp1 + phi_xm1_zm1
                      + phi_yp1_zp1 + phi_yp1_zm1 + phi_ym1_zp1 + phi_ym1_zm1);

    fMom[idxMom(tx, ty, tz, M3_NX_INDEX, bx, by, bz)] = dphidx;
    fMom[idxMom(tx, ty, tz, M3_NY_INDEX, bx, by, bz)] = dphidy;
    fMom[idxMom(tx, ty, tz, M3_NZ_INDEX, bx, by, bz)] = dphidz;
    fMom[idxMom(tx, ty, tz, M3_LP_INDEX, bx, by, bz)] = laplacian_phi;
}


__global__ void gpuComputeChemicalPotential(
    dfloat *fMom,
    unsigned int *dNodeType
)
{
    const int x = threadIdx.x + blockDim.x * blockIdx.x;
    const int y = threadIdx.y + blockDim.y * blockIdx.y;
    const int z = threadIdx.z + blockDim.z * blockIdx.z;
    if (x >= NX || y >= NY || z >= NZ) return;

    unsigned int nodeType =
        dNodeType[idxScalarBlock(threadIdx.x, threadIdx.y, threadIdx.z,
                                 blockIdx.x, blockIdx.y, blockIdx.z)];
    if (nodeType == 0b11111111) return;
    if (nodeType != BULK) return;

    const int tx = threadIdx.x, ty = threadIdx.y, tz = threadIdx.z;
    const int bx = blockIdx.x,  by = blockIdx.y,  bz = blockIdx.z;

    dfloat phi     = fMom[idxMom(tx, ty, tz, M3_PHI_INDEX, bx, by, bz)];
    dfloat lap_phi = fMom[idxMom(tx, ty, tz, M3_LP_INDEX,  bx, by, bz)]; // Neumann from above

    dfloat dfdphi = A_CH * (phi*phi*phi - phi);
    dfloat mu     = dfdphi - kappa_CH * lap_phi;
/*
    // ---- wetting surface energy correction ----
    auto solve_wetting = [](dfloat phi_p, dfloat q) -> dfloat {
        if (fabs(q) < 1e-6_df) return phi_p;
        phi_p = fmax(PHI_ONE, fmin(PHI_TWO, phi_p));
        const dfloat indicator = 1.0_df - phi_p * phi_p;
        if (indicator < 1e-4_df) return phi_p;  // bulk: no correction

        const dfloat a = q, b = -1.0_df, c = phi_p - q;
        const dfloat disc = b*b - 4.0_df*a*c;
        if (disc < 0.0_df) return phi_p;

        const dfloat sq       = sqrt(disc);
        const dfloat phi_plus  = (-b + sq) / (2.0_df*a);
        const dfloat phi_minus = (-b - sq) / (2.0_df*a);

        const dfloat lo = PHI_ONE - 1e-6_df, hi = PHI_TWO + 1e-6_df;
        const bool p_ok = (phi_plus  >= lo && phi_plus  <= hi);
        const bool m_ok = (phi_minus >= lo && phi_minus <= hi);

        dfloat raw;
        if      (p_ok && m_ok) raw = (fabs(phi_plus-phi_p) <= fabs(phi_minus-phi_p))
                                      ? phi_plus : phi_minus;
        else if (p_ok)         raw = phi_plus;
        else if (m_ok)         raw = phi_minus;
        else                   raw = phi_p;

        return fmax(PHI_ONE, fmin(PHI_TWO, phi_p + indicator*(raw - phi_p)));
    };

    #ifdef BC_Y_WALL
    {
        const dfloat q_wet = -sqrt(A_CH / (2.0_df * kappa_CH)) * cos(M_PI/2.0); //FIX FOR CONTACT ANGLE
        if (y == 1 || y == NY-2) {
            const dfloat phi_wall = solve_wetting(phi, q_wet);
            mu += -kappa_CH * (phi_wall - phi);
        }
    }
    #endif

    #ifdef BC_X_WALL
    {
        const dfloat q_wet = -sqrt(A_CH / (2.0_df * kappa_CH)) * cos(contact_angle);
        if (x == 1 || x == NX-2) {
            const dfloat phi_wall = solve_wetting(phi, q_wet);
            mu += -kappa_CH * (phi_wall - phi);
        }
    }
    #endif

    #ifdef BC_Z_WALL
    {
        const dfloat q_wet = -sqrt(A_CH / (2.0_df * kappa_CH)) * cos(contact_angle);
        if (z == 1 || z == NZ-2) {
            const dfloat phi_wall = solve_wetting(phi, q_wet);
            mu += -kappa_CH * (phi_wall - phi);
        }
    }
    #endif
    *//*

    fMom[idxMom(tx, ty, tz, M3_MU_INDEX, bx, by, bz)] = mu;
}

__global__ void gpuComputeLaplacianMu(
    dfloat *fMom,
    unsigned int *dNodeType
)
{
    const int x = threadIdx.x + blockDim.x * blockIdx.x;
    const int y = threadIdx.y + blockDim.y * blockIdx.y;
    const int z = threadIdx.z + blockDim.z * blockIdx.z;
    if (x >= NX || y >= NY || z >= NZ) return;

    unsigned int nodeType =
        dNodeType[idxScalarBlock(threadIdx.x, threadIdx.y, threadIdx.z,
                                 blockIdx.x, blockIdx.y, blockIdx.z)];
    if (nodeType == 0b11111111) return;
    if (nodeType != BULK) return;

    const int tx = threadIdx.x, ty = threadIdx.y, tz = threadIdx.z;
    const int bx = blockIdx.x,  by = blockIdx.y,  bz = blockIdx.z;

    dfloat mu0 = fMom[idxMom(tx, ty, tz, M3_MU_INDEX, bx, by, bz)];

    auto getMu = [&](int dx, int dy, int dz) -> dfloat {
        #ifdef BC_X_WALL
        int nx = min(NX - 1, max(0, x + dx));
        #else
        int nx = (x + dx + NX) % NX;
        #endif
        #ifdef BC_Y_WALL
        int ny = min(NY - 1, max(0, y + dy));
        #else
        int ny = (y + dy + NY) % NY;
        #endif
        #ifdef BC_Z_WALL
        int nz = min(NZ - 1, max(0, z + dz));
        #else
        int nz = (z + dz + NZ) % NZ;
        #endif
        int ntx = nx % BLOCK_NX, nty = ny % BLOCK_NY, ntz = nz % BLOCK_NZ;
        int nbx = nx / BLOCK_NX, nby = ny / BLOCK_NY, nbz = nz / BLOCK_NZ;
        return fMom[idxMom(ntx, nty, ntz, M3_MU_INDEX, nbx, nby, nbz)];
    };

    // ---- load all mu neighbors ----
    dfloat mu_xp1 = getMu(+1, 0, 0);  dfloat mu_xm1 = getMu(-1, 0, 0);
    dfloat mu_yp1 = getMu( 0,+1, 0);  dfloat mu_ym1 = getMu( 0,-1, 0);
    dfloat mu_zp1 = getMu( 0, 0,+1);  dfloat mu_zm1 = getMu( 0, 0,-1);

    dfloat mu_xp1_yp1 = getMu(+1,+1, 0);  dfloat mu_xp1_ym1 = getMu(+1,-1, 0);
    dfloat mu_xm1_yp1 = getMu(-1,+1, 0);  dfloat mu_xm1_ym1 = getMu(-1,-1, 0);
    dfloat mu_xp1_zp1 = getMu(+1, 0,+1);  dfloat mu_xp1_zm1 = getMu(+1, 0,-1);
    dfloat mu_xm1_zp1 = getMu(-1, 0,+1);  dfloat mu_xm1_zm1 = getMu(-1, 0,-1);
    dfloat mu_yp1_zp1 = getMu( 0,+1,+1);  dfloat mu_yp1_zm1 = getMu( 0,+1,-1);
    dfloat mu_ym1_zp1 = getMu( 0,-1,+1);  dfloat mu_ym1_zm1 = getMu( 0,-1,-1);

    #ifdef BC_Y_WALL
    if (y == 1) {           // SOUTH wall adjacent
        mu_ym1     = mu0;
        mu_xp1_ym1 = mu_xp1;
        mu_xm1_ym1 = mu_xm1;
        mu_ym1_zp1 = mu_zp1;
        mu_ym1_zm1 = mu_zm1;
    }
    if (y == NY-2) {        // NORTH wall adjacent
        mu_yp1     = mu0;
        mu_xp1_yp1 = mu_xp1;
        mu_xm1_yp1 = mu_xm1;
        mu_yp1_zp1 = mu_zp1;
        mu_yp1_zm1 = mu_zm1;
    }
    #endif

    #ifdef BC_X_WALL
    if (x == 1) {
        mu_xm1     = mu0;
        mu_xm1_yp1 = mu_yp1;
        mu_xm1_ym1 = mu_ym1;
        mu_xm1_zp1 = mu_zp1;
        mu_xm1_zm1 = mu_zm1;
    }
    if (x == NX-2) {
        mu_xp1     = mu0;
        mu_xp1_yp1 = mu_yp1;
        mu_xp1_ym1 = mu_ym1;
        mu_xp1_zp1 = mu_zp1;
        mu_xp1_zm1 = mu_zm1;
    }
    #endif

    #ifdef BC_Z_WALL
    if (z == 1) {
        mu_zm1     = mu0;
        mu_xp1_zm1 = mu_xp1;
        mu_xm1_zm1 = mu_xm1;
        mu_yp1_zm1 = mu_yp1;
        mu_ym1_zm1 = mu_ym1;
    }
    if (z == NZ-2) {
        mu_zp1     = mu0;
        mu_xp1_zp1 = mu_xp1;
        mu_xm1_zp1 = mu_xm1;
        mu_yp1_zp1 = mu_yp1;
        mu_ym1_zp1 = mu_ym1;
    }
    #endif

    // ---- isotropic Laplacian ----
    constexpr dfloat w0 = -4.0_df;
    constexpr dfloat w1 =  1.0_df / 3.0_df;
    constexpr dfloat w2 =  1.0_df / 6.0_df;

    dfloat laplacian_mu =
        w0 * mu0
      + w1 * (mu_xp1 + mu_xm1 + mu_yp1 + mu_ym1 + mu_zp1 + mu_zm1)
      + w2 * (mu_xp1_yp1 + mu_xp1_ym1 + mu_xm1_yp1 + mu_xm1_ym1
            + mu_xp1_zp1 + mu_xp1_zm1 + mu_xm1_zp1 + mu_xm1_zm1
            + mu_yp1_zp1 + mu_yp1_zm1 + mu_ym1_zp1 + mu_ym1_zm1);

    fMom[idxMom(tx, ty, tz, M3_LM_INDEX, bx, by, bz)] = laplacian_mu;
}


*/
__global__ void gpuComputePhaseNormals(
    dfloat *fMom,
    unsigned int *dNodeType
)
{
    const int x = threadIdx.x + blockDim.x * blockIdx.x;
    const int y = threadIdx.y + blockDim.y * blockIdx.y;
    const int z = threadIdx.z + blockDim.z * blockIdx.z;
    if (x >= NX || y >= NY || z >= NZ) return;

    const int tx = threadIdx.x, ty = threadIdx.y, tz = threadIdx.z;
    const int bx = blockIdx.x,  by = blockIdx.y,  bz = blockIdx.z;

    unsigned int nodeType = dNodeType[idxScalarBlock(tx, ty, tz, bx, by, bz)];
    if (nodeType == 0b11111111) return;  // only skip fully solid

    auto getPhi = [&](int dx, int dy, int dz) -> dfloat {
        #ifdef BC_X_WALL
        int nx = min(NX - 1, max(0, x + dx));
        #else
        int nx = (x + dx + NX) % NX;
        #endif
        #ifdef BC_Y_WALL
        int ny = min(NY - 1, max(0, y + dy));
        #else
        int ny = (y + dy + NY) % NY;
        #endif
        #ifdef BC_Z_WALL
        int nz = min(NZ - 1, max(0, z + dz));
        #else
        int nz = (z + dz + NZ) % NZ;
        #endif
        int ntx = nx % BLOCK_NX, nty = ny % BLOCK_NY, ntz = nz % BLOCK_NZ;
        int nbx = nx / BLOCK_NX, nby = ny / BLOCK_NY, nbz = nz / BLOCK_NZ;
        return fMom[idxMom(ntx, nty, ntz, M3_PHI_INDEX, nbx, nby, nbz)];
    };

    dfloat phi_c = fMom[idxMom(tx, ty, tz, M3_PHI_INDEX, bx, by, bz)];

    // load all neighbors with standard clamping
    dfloat phi_xm1 = getPhi(-1, 0, 0);  dfloat phi_xp1 = getPhi(+1, 0, 0);
    dfloat phi_ym1 = getPhi( 0,-1, 0);  dfloat phi_yp1 = getPhi( 0,+1, 0);
    dfloat phi_zm1 = getPhi( 0, 0,-1);  dfloat phi_zp1 = getPhi( 0, 0,+1);

    dfloat phi_xm1_ym1 = getPhi(-1,-1, 0);  dfloat phi_xp1_ym1 = getPhi(+1,-1, 0);
    dfloat phi_xm1_yp1 = getPhi(-1,+1, 0);  dfloat phi_xp1_yp1 = getPhi(+1,+1, 0);
    dfloat phi_xm1_zm1 = getPhi(-1, 0,-1);  dfloat phi_xp1_zm1 = getPhi(+1, 0,-1);
    dfloat phi_xm1_zp1 = getPhi(-1, 0,+1);  dfloat phi_xp1_zp1 = getPhi(+1, 0,+1);
    dfloat phi_ym1_zm1 = getPhi( 0,-1,-1);  dfloat phi_yp1_zm1 = getPhi( 0,+1,-1);
    dfloat phi_ym1_zp1 = getPhi( 0,-1,+1);  dfloat phi_yp1_zp1 = getPhi( 0,+1,+1);

    // ---- bitmask wall detection (fires at the wall node itself) ----
    const bool wall_xm = (nodeType & 0b01010101) == 0b01010101;  // WEST  wall in -x
    const bool wall_xp = (nodeType & 0b10101010) == 0b10101010;  // EAST  wall in +x
    const bool wall_ym = (nodeType & 0b00110011) == 0b00110011;  // SOUTH wall in -y
    const bool wall_yp = (nodeType & 0b11001100) == 0b11001100;  // NORTH wall in +y
    const bool wall_zm = (nodeType & 0b00001111) == 0b00001111;  // BACK  wall in -z
    const bool wall_zp = (nodeType & 0b11110000) == 0b11110000;  // FRONT wall in +z

    // Neumann BC: ghost = phi_c for each wall direction.
    // Edges at corners use phi_c if BOTH directions are walls, else the
    // single-direction Neumann value.
    if (wall_ym) {
        phi_ym1     = phi_c;
        phi_xp1_ym1 = wall_xp ? phi_c : phi_xp1;
        phi_xm1_ym1 = wall_xm ? phi_c : phi_xm1;
        phi_ym1_zp1 = wall_zp ? phi_c : phi_zp1;
        phi_ym1_zm1 = wall_zm ? phi_c : phi_zm1;
    }
    if (wall_yp) {
        phi_yp1     = phi_c;
        phi_xp1_yp1 = wall_xp ? phi_c : phi_xp1;
        phi_xm1_yp1 = wall_xm ? phi_c : phi_xm1;
        phi_yp1_zp1 = wall_zp ? phi_c : phi_zp1;
        phi_yp1_zm1 = wall_zm ? phi_c : phi_zm1;
    }
    if (wall_xm) {
        phi_xm1     = phi_c;
        phi_xm1_yp1 = wall_yp ? phi_c : phi_yp1;
        phi_xm1_ym1 = wall_ym ? phi_c : phi_ym1;
        phi_xm1_zp1 = wall_zp ? phi_c : phi_zp1;
        phi_xm1_zm1 = wall_zm ? phi_c : phi_zm1;
    }
    if (wall_xp) {
        phi_xp1     = phi_c;
        phi_xp1_yp1 = wall_yp ? phi_c : phi_yp1;
        phi_xp1_ym1 = wall_ym ? phi_c : phi_ym1;
        phi_xp1_zp1 = wall_zp ? phi_c : phi_zp1;
        phi_xp1_zm1 = wall_zm ? phi_c : phi_zm1;
    }
    if (wall_zm) {
        phi_zm1     = phi_c;
        phi_xp1_zm1 = wall_xp ? phi_c : phi_xp1;
        phi_xm1_zm1 = wall_xm ? phi_c : phi_xm1;
        phi_yp1_zm1 = wall_yp ? phi_c : phi_yp1;
        phi_ym1_zm1 = wall_ym ? phi_c : phi_ym1;
    }
    if (wall_zp) {
        phi_zp1     = phi_c;
        phi_xp1_zp1 = wall_xp ? phi_c : phi_xp1;
        phi_xm1_zp1 = wall_xm ? phi_c : phi_xm1;
        phi_yp1_zp1 = wall_yp ? phi_c : phi_yp1;
        phi_ym1_zp1 = wall_ym ? phi_c : phi_ym1;
    }

    // ---- isotropic gradient ----
    constexpr dfloat w_axis = 1.0_df / 6.0_df;
    constexpr dfloat w_edge = 1.0_df / 12.0_df;

    dfloat dphidx = w_axis * (phi_xp1 - phi_xm1)
                  + w_edge * (phi_xp1_ym1 - phi_xm1_ym1 + phi_xp1_yp1 - phi_xm1_yp1
                            + phi_xp1_zm1 - phi_xm1_zm1 + phi_xp1_zp1 - phi_xm1_zp1);

    dfloat dphidy = w_axis * (phi_yp1 - phi_ym1)
                  + w_edge * (phi_xm1_yp1 - phi_xm1_ym1 + phi_xp1_yp1 - phi_xp1_ym1
                            + phi_yp1_zm1 - phi_ym1_zm1 + phi_yp1_zp1 - phi_ym1_zp1);

    dfloat dphidz = w_axis * (phi_zp1 - phi_zm1)
                  + w_edge * (phi_xm1_zp1 - phi_xm1_zm1 + phi_xp1_zp1 - phi_xp1_zm1
                            + phi_ym1_zp1 - phi_ym1_zm1 + phi_yp1_zp1 - phi_yp1_zm1);

    // ---- isotropic Laplacian ----
    constexpr dfloat w_lap_zero = -4.0_df;
    constexpr dfloat w_lap_axis =  1.0_df / 3.0_df;
    constexpr dfloat w_lap_edge =  1.0_df / 6.0_df;

    dfloat laplacian_phi =
        w_lap_zero * phi_c
        + w_lap_axis * (phi_xp1 + phi_xm1 + phi_yp1 + phi_ym1 + phi_zp1 + phi_zm1)
        + w_lap_edge * (phi_xp1_yp1 + phi_xp1_ym1 + phi_xm1_yp1 + phi_xm1_ym1
                      + phi_xp1_zp1 + phi_xp1_zm1 + phi_xm1_zp1 + phi_xm1_zm1
                      + phi_yp1_zp1 + phi_yp1_zm1 + phi_ym1_zp1 + phi_ym1_zm1);

    fMom[idxMom(tx, ty, tz, M3_NX_INDEX, bx, by, bz)] = dphidx;
    fMom[idxMom(tx, ty, tz, M3_NY_INDEX, bx, by, bz)] = dphidy;
    fMom[idxMom(tx, ty, tz, M3_NZ_INDEX, bx, by, bz)] = dphidz;
    fMom[idxMom(tx, ty, tz, M3_LP_INDEX, bx, by, bz)] = laplacian_phi;
}



__global__ void gpuComputeChemicalPotential(
    dfloat *fMom,
    unsigned int *dNodeType
)
{
    const int x = threadIdx.x + blockDim.x * blockIdx.x;
    const int y = threadIdx.y + blockDim.y * blockIdx.y;
    const int z = threadIdx.z + blockDim.z * blockIdx.z;
    if (x >= NX || y >= NY || z >= NZ) return;

    const int tx = threadIdx.x, ty = threadIdx.y, tz = threadIdx.z;
    const int bx = blockIdx.x,  by = blockIdx.y,  bz = blockIdx.z;

    unsigned int nodeType =
        dNodeType[idxScalarBlock(tx, ty, tz, bx, by, bz)];
    if (nodeType == 0b11111111) return;  // only skip fully solid

    dfloat phi     = fMom[idxMom(tx, ty, tz, M3_PHI_INDEX, bx, by, bz)];
    dfloat lap_phi = fMom[idxMom(tx, ty, tz, M3_LP_INDEX,  bx, by, bz)];

    dfloat dfdphi = A_CH * (phi*phi*phi - phi);
    dfloat mu     = dfdphi - kappa_CH * lap_phi;

    // ---- bitmask wall detection ----
    const bool wall_xm = (nodeType & 0b01010101) == 0b01010101;  // WEST
    const bool wall_xp = (nodeType & 0b10101010) == 0b10101010;  // EAST
    const bool wall_ym = (nodeType & 0b00110011) == 0b00110011;  // SOUTH
    const bool wall_yp = (nodeType & 0b11001100) == 0b11001100;  // NORTH
    const bool wall_zm = (nodeType & 0b00001111) == 0b00001111;  // BACK
    const bool wall_zp = (nodeType & 0b11110000) == 0b11110000;  // FRONT

    const bool is_wall = wall_xm || wall_xp || wall_ym || wall_yp
                      || wall_zm || wall_zp;
                      
    /*
    if (is_wall) {
        // Load fluid-side phi neighbors (needed for solve_wetting)
        auto getPhi = [&](int dx, int dy, int dz) -> dfloat {
            #ifdef BC_X_WALL
            int nx = min(NX - 1, max(0, x + dx));
            #else
            int nx = (x + dx + NX) % NX;
            #endif
            #ifdef BC_Y_WALL
            int ny = min(NY - 1, max(0, y + dy));
            #else
            int ny = (y + dy + NY) % NY;
            #endif
            #ifdef BC_Z_WALL
            int nz = min(NZ - 1, max(0, z + dz));
            #else
            int nz = (z + dz + NZ) % NZ;
            #endif
            int ntx = nx % BLOCK_NX, nty = ny % BLOCK_NY, ntz = nz % BLOCK_NZ;
            int nbx = nx / BLOCK_NX, nby = ny / BLOCK_NY, nbz = nz / BLOCK_NZ;
            return fMom[idxMom(ntx, nty, ntz, M3_PHI_INDEX, nbx, nby, nbz)];
        };
        
        auto solve_wetting = [](dfloat phi_p, dfloat q) -> dfloat {
            if (fabs(q) < 1e-6_df) return phi_p;
            phi_p = fmax(PHI_ONE, fmin(PHI_TWO, phi_p));
            const dfloat indicator = 1.0_df - phi_p * phi_p;
            if (indicator < 1e-4_df) return phi_p;  // bulk: no correction

            const dfloat a = q, b = -1.0_df, c = phi_p - q;
            const dfloat disc = b*b - 4.0_df*a*c;
            if (disc < 0.0_df) return phi_p;

            const dfloat sq        = sqrt(disc);
            const dfloat phi_plus  = (-b + sq) / (2.0_df*a);
            const dfloat phi_minus = (-b - sq) / (2.0_df*a);

            const dfloat lo = PHI_ONE - 1e-6_df, hi = PHI_TWO + 1e-6_df;
            const bool p_ok = (phi_plus  >= lo && phi_plus  <= hi);
            const bool m_ok = (phi_minus >= lo && phi_minus <= hi);

            dfloat raw;
            if      (p_ok && m_ok) raw = (fabs(phi_plus-phi_p) <= fabs(phi_minus-phi_p))
                                          ? phi_plus : phi_minus;
            else if (p_ok)         raw = phi_plus;
            else if (m_ok)         raw = phi_minus;
            else                   raw = phi_p;

            return fmax(PHI_ONE, fmin(PHI_TWO, phi_p + indicator*(raw - phi_p)));
        };

        const dfloat q_wet = -sqrt(A_CH / (2.0_df * kappa_CH)) * cos(contact_angle);

        if (wall_ym) {
            const dfloat phi_wall = solve_wetting(getPhi(0,+1,0), q_wet);
            mu += -kappa_CH * (phi_wall - phi);
        }
        if (wall_yp) {
            const dfloat phi_wall = solve_wetting(getPhi(0,-1,0), q_wet);
            mu += -kappa_CH * (phi_wall - phi);
        }
        if (wall_xm) {
            const dfloat phi_wall = solve_wetting(getPhi(+1,0,0), q_wet);
            mu += -kappa_CH * (phi_wall - phi);
        }
        if (wall_xp) {
            const dfloat phi_wall = solve_wetting(getPhi(-1,0,0), q_wet);
            mu += -kappa_CH * (phi_wall - phi);
        }
        if (wall_zm) {
            const dfloat phi_wall = solve_wetting(getPhi(0,0,+1), q_wet);
            mu += -kappa_CH * (phi_wall - phi);
        }
        if (wall_zp) {
            const dfloat phi_wall = solve_wetting(getPhi(0,0,-1), q_wet);
            mu += -kappa_CH * (phi_wall - phi);
        }
        
    }
    */
    fMom[idxMom(tx, ty, tz, M3_MU_INDEX, bx, by, bz)] = mu;
}



__global__ void gpuComputeLaplacianMu(
    dfloat *fMom,
    unsigned int *dNodeType
)
{
    const int x = threadIdx.x + blockDim.x * blockIdx.x;
    const int y = threadIdx.y + blockDim.y * blockIdx.y;
    const int z = threadIdx.z + blockDim.z * blockIdx.z;
    if (x >= NX || y >= NY || z >= NZ) return;

    const int tx = threadIdx.x, ty = threadIdx.y, tz = threadIdx.z;
    const int bx = blockIdx.x,  by = blockIdx.y,  bz = blockIdx.z;

    unsigned int nodeType =
        dNodeType[idxScalarBlock(tx, ty, tz, bx, by, bz)];
    if (nodeType == 0b11111111) return;  // only skip fully solid

    dfloat mu0 = fMom[idxMom(tx, ty, tz, M3_MU_INDEX, bx, by, bz)];


    auto getMu = [&](int dx, int dy, int dz) -> dfloat {
        #ifdef BC_X_WALL
        int nx = min(NX - 1, max(0, x + dx));
        #else
        int nx = (x + dx + NX) % NX;
        #endif
        #ifdef BC_Y_WALL
        int ny = min(NY - 1, max(0, y + dy));
        #else
        int ny = (y + dy + NY) % NY;
        #endif
        #ifdef BC_Z_WALL
        int nz = min(NZ - 1, max(0, z + dz));
        #else
        int nz = (z + dz + NZ) % NZ;
        #endif
        int ntx = nx % BLOCK_NX, nty = ny % BLOCK_NY, ntz = nz % BLOCK_NZ;
        int nbx = nx / BLOCK_NX, nby = ny / BLOCK_NY, nbz = nz / BLOCK_NZ;
        return fMom[idxMom(ntx, nty, ntz, M3_MU_INDEX, nbx, nby, nbz)];
    };

    dfloat mu_xp1 = getMu(+1, 0, 0);  dfloat mu_xm1 = getMu(-1, 0, 0);
    dfloat mu_yp1 = getMu( 0,+1, 0);  dfloat mu_ym1 = getMu( 0,-1, 0);
    dfloat mu_zp1 = getMu( 0, 0,+1);  dfloat mu_zm1 = getMu( 0, 0,-1);

    dfloat mu_xp1_yp1 = getMu(+1,+1, 0);  dfloat mu_xp1_ym1 = getMu(+1,-1, 0);
    dfloat mu_xm1_yp1 = getMu(-1,+1, 0);  dfloat mu_xm1_ym1 = getMu(-1,-1, 0);
    dfloat mu_xp1_zp1 = getMu(+1, 0,+1);  dfloat mu_xp1_zm1 = getMu(+1, 0,-1);
    dfloat mu_xm1_zp1 = getMu(-1, 0,+1);  dfloat mu_xm1_zm1 = getMu(-1, 0,-1);
    dfloat mu_yp1_zp1 = getMu( 0,+1,+1);  dfloat mu_yp1_zm1 = getMu( 0,+1,-1);
    dfloat mu_ym1_zp1 = getMu( 0,-1,+1);  dfloat mu_ym1_zm1 = getMu( 0,-1,-1);

    constexpr dfloat w0 = -4.0_df;
    constexpr dfloat w1 =  1.0_df / 3.0_df;
    constexpr dfloat w2 =  1.0_df / 6.0_df;

    dfloat laplacian_mu =
        w0 * mu0
      + w1 * (mu_xp1 + mu_xm1 + mu_yp1 + mu_ym1 + mu_zp1 + mu_zm1)
      + w2 * (mu_xp1_yp1 + mu_xp1_ym1 + mu_xm1_yp1 + mu_xm1_ym1
            + mu_xp1_zp1 + mu_xp1_zm1 + mu_xm1_zp1 + mu_xm1_zm1
            + mu_yp1_zp1 + mu_yp1_zm1 + mu_ym1_zp1 + mu_ym1_zm1);

    fMom[idxMom(tx, ty, tz, M3_LM_INDEX, bx, by, bz)] = laplacian_mu;
}

/**/

#endif // PHI_DIST
