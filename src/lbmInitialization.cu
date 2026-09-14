#include "lbmInitialization.cuh"



__host__
void initializationRandomNumbers(
    dfloat* randomNumbers, int seed)
{
    curandGenerator_t gen;

    // Create pseudo-random number generator
    checkCurandStatus(curandCreateGenerator(&gen,
        CURAND_RNG_PSEUDO_DEFAULT));
    
    // Set generator seed
    checkCurandStatus(curandSetPseudoRandomGeneratorSeed(gen,
        CURAND_SEED));
    
    // Generate NX*NY*NZ floats on device, using normal distribution
    // with mean=0 and std_dev=NORMAL_STD_DEV
    #ifdef SINGLE_PRECISION 
    checkCurandStatus(curandGenerateNormal(gen, randomNumbers, NUMBER_LBM_NODES_LOCAL,
        0, CURAND_STD_DEV));
    #endif //SINGLE_PRECISION
    #ifdef DOUBLE_PRECISION
    checkCurandStatus(curandGenerateNormalDouble (gen, randomNumbers, NUMBER_LBM_NODES_LOCAL,
        0, CURAND_STD_DEV));
    #endif

    checkCurandStatus(curandDestroyGenerator(gen));
}


__global__ void gpuInitialization_mom(
    dfloat *fMom, dfloat* randomNumbers, size_t localNZ, int zStart)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int z_local = threadIdx.z + blockDim.z * blockIdx.z;
    
    if (x >= NX || y >= NY || z_local >= localNZ)
        return;
    
    int z = zStart + z_local;

    size_t index = idxScalarGlobal(x, y, z_local);

    //first moments
    dfloat rho = RHO_0, ux = U_0_X, uy = U_0_Y, uz = U_0_Z;
    #ifdef OMEGA_FIELD
    dfloat omega;
    #endif //OMEGA_FIELD
    #ifdef SECOND_DIST 
    dfloat cVar = 1.0_df;
    dfloat qx_t30 = 3.0_df*(ux - 0.0_df);
    dfloat qy_t30 = 3.0_df*(uy - 0.0_df);
    dfloat qz_t30 = 3.0_df*(uz - 0.0_df);
    #endif //SECOND_DIST
    #ifdef PHI_DIST 
    dfloat phiVar = PHI_TWO;  // default: all phase 2; overridden by CASE_FIELD_INIT
    dfloat phi_qx_t30 = 3.0_df*(ux - 0.0_df);
    dfloat phi_qy_t30 = 3.0_df*(uy - 0.0_df);
    dfloat phi_qz_t30 = 3.0_df*(uz - 0.0_df);
    #endif //PHI_DIST
    #ifdef LAMBDA_DIST 
    dfloat lambdaVar = 0.0_df + LAMBDA_ZERO;
    dfloat lambda_qx_t30 = 3.0_df*lambdaVar*(ux - 0.0_df);
    dfloat lambda_qy_t30 = 3.0_df*lambdaVar*(uy - 0.0_df);
    dfloat lambda_qz_t30 = 3.0_df*lambdaVar*(uz - 0.0_df);
    #endif //LAMBDA_DIST
    #ifdef CONFORMATION_TENSOR
        //assuming that velocity has grad = 0 
        #ifdef A_XX_DIST 
        dfloat AxxVar = 1.0_df + CONF_ZERO; 
        dfloat Axx_qx_t30 = 3.0_df*(ux + 0.0_df);
        dfloat Axx_qy_t30 = 3.0_df*(uy + 0.0_df);
        dfloat Axx_qz_t30 = 3.0_df*(uz + 0.0_df);
        #endif
        #ifdef A_XY_DIST 
        dfloat AxyVar = 0.0_df + CONF_ZERO;
        dfloat Axy_qx_t30 = 3.0_df*(ux + 0.0_df);
        dfloat Axy_qy_t30 = 3.0_df*(uy + 0.0_df);
        dfloat Axy_qz_t30 = 3.0_df*(uz + 0.0_df);
        #endif
        #ifdef A_XZ_DIST 
        dfloat AxzVar = 0.0_df + CONF_ZERO;
        dfloat Axz_qx_t30 = 3.0_df*(ux + 0.0_df);
        dfloat Axz_qy_t30 = 3.0_df*(uy + 0.0_df);
        dfloat Axz_qz_t30 = 3.0_df*(uz + 0.0_df);
        #endif
        #ifdef A_YY_DIST 
        dfloat AyyVar = 1.0_df + CONF_ZERO;
        dfloat Ayy_qx_t30 = 3.0_df*(ux + 0.0_df);
        dfloat Ayy_qy_t30 = 3.0_df*(uy + 0.0_df);
        dfloat Ayy_qz_t30 = 3.0_df*(uz + 0.0_df);
        #endif
        #ifdef A_YZ_DIST 
        dfloat AyzVar = 0.0_df + CONF_ZERO;
        dfloat Ayz_qx_t30 = 3.0_df*(ux + 0.0_df);
        dfloat Ayz_qy_t30 = 3.0_df*(uy + 0.0_df);
        dfloat Ayz_qz_t30 = 3.0_df*(uz + 0.0_df);
        #endif
        #ifdef A_ZZ_DIST 
        dfloat AzzVar = 1.0_df + CONF_ZERO;
        dfloat Azz_qx_t30 = 3.0_df*(ux + 0.0_df);
        dfloat Azz_qy_t30 = 3.0_df*(uy + 0.0_df);
        dfloat Azz_qz_t30 = 3.0_df*(uz + 0.0_df);
        #endif
    #endif


    #include CASE_FLOW_INITIALIZATION

    #ifdef OMEGA_FIELD
    omega = OMEGA;
    #endif

   
    // zeroth moment
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_RHO_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = rho-RHO_0;
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_UX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = F_M_I_SCALE*ux;
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_UY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = F_M_I_SCALE*uy;
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_UZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = F_M_I_SCALE*uz;

    //second moments
    //define equilibrium populations
    dfloat pop[Q];
    for (int i = 0; i < Q; i++)
    {
        pop[i] = gpu_f_eq(w[i] * RHO_0,
                          3.0_df * (ux * cx[i] + uy * cy[i] + uz * cz[i]),
                          1.0_df - 1.5_df * (ux * ux + uy * uy + uz * uz));
    }
    
    dfloat invRho = 1.0_df/rho;
    dfloat pixx =  (pop[1] + pop[2] + pop[7] + pop[8] + pop[9] + pop[10] + pop[13] + pop[14] + pop[15] + pop[16]) * invRho - cs2;
    dfloat pixy = ((pop[7] + pop[ 8]) - (pop[13] + pop[14])) * invRho;
    dfloat pixz = ((pop[9] + pop[10]) - (pop[15] + pop[16])) * invRho;
    dfloat piyy =  (pop[3] + pop[4] + pop[7] + pop[8] + pop[11] + pop[12] + pop[13] + pop[14] + pop[17] + pop[18]) * invRho - cs2;
    dfloat piyz = ((pop[11]+pop[12])-(pop[17]+pop[18])) * invRho;
    dfloat pizz =  (pop[5] + pop[6] + pop[9] + pop[10] + pop[11] + pop[12] + pop[15] + pop[16] + pop[17] + pop[18]) * invRho - cs2;

    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_MXX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = F_M_II_SCALE*pixx;
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_MXY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = F_M_IJ_SCALE*pixy;
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_MXZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = F_M_IJ_SCALE*pixz;
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_MYY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = F_M_II_SCALE*piyy;
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_MYZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = F_M_IJ_SCALE*piyz;
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_MZZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = F_M_II_SCALE*pizz;

    #ifdef OMEGA_FIELD
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_OMEGA_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = omega;
    #endif 
    
    
    #ifdef SECOND_DIST 
    dfloat invC= 1.0_df/cVar;

    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M2_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = cVar;
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M2_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = qx_t30;
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M2_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = qy_t30;
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M2_CZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = qz_t30;
    #endif //SECOND_DIST

        
    #ifdef PHI_DIST 
    dfloat invC= 1.0_df/phiVar;

    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M3_PHI_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = phiVar;
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M3_PX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = phi_qx_t30;
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M3_PY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = phi_qy_t30;
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M3_PZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = phi_qz_t30;
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M3_NX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = 0;
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M3_NY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = 0;
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M3_NZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = 0;
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M3_LP_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = 0;
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M3_MU_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = 0;
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M3_LM_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = 0;
    #endif //PHI_DIST

    #ifdef LAMBDA_DIST 
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M4_LAMBDA_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = lambdaVar;
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M4_LX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = lambda_qx_t30;
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M4_LY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = lambda_qy_t30;
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M4_LZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = lambda_qz_t30;
    #endif //LAMBDA_DIST

    #ifdef A_XX_DIST 
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_XX_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] =  AxxVar;
    //fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, G_XX_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] =  0.0_df;
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_XX_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = Axx_qx_t30;
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_XX_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = Axx_qy_t30;
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_XX_CZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = Axx_qz_t30;
    #endif 
    #ifdef A_XY_DIST 
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_XY_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] =  AxyVar;
    //fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, G_XY_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] =  0.0_df;
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_XY_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = Axy_qx_t30;
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_XY_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = Axy_qy_t30;
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_XY_CZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = Axy_qz_t30;
    #endif 
    #ifdef A_XZ_DIST 
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_XZ_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] =  AxzVar;
    //fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, G_XZ_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] =  0.0_df;
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_XZ_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = Axz_qx_t30;
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_XZ_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = Axz_qy_t30;
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_XZ_CZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = Axz_qz_t30;
    #endif
    #ifdef A_YY_DIST 
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_YY_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] =  AyyVar;
    //fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, G_YY_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] =  0.0_df;
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_YY_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = Ayy_qx_t30;
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_YY_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = Ayy_qy_t30;
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_YY_CZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = Ayy_qz_t30;
    #endif
    #ifdef A_YZ_DIST 
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_YZ_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] =  AyzVar;
    //fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, G_YZ_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] =  0.0_df;
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_YZ_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = Ayz_qx_t30;
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_YZ_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = Ayz_qy_t30;
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_YZ_CZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = Ayz_qz_t30;
    #endif
    #ifdef A_ZZ_DIST 
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_ZZ_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] =  AzzVar;
    //fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, G_ZZ_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] =  0.0_df;
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_ZZ_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = Azz_qx_t30;
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_ZZ_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = Azz_qy_t30;
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_ZZ_CZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = Azz_qz_t30;
    #endif

    #ifdef LOCAL_FORCES
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_FX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = FX;
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_FY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = FY;
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_FZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = FZ;
    #endif 


}

__global__ void gpuInitialization_pop(
    dfloat *fMom, ghostInterfaceData ghostInterface, size_t localNZ, int zStart)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int z_local = threadIdx.z + blockDim.z * blockIdx.z;

    if (x >= NX || y >= NY || z_local >= localNZ)
        return;

    int z = zStart + z_local;

    size_t index = idxScalarGlobal(x, y, z_local);
    // zeroth moment

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

    dfloat pop[Q];
    dfloat multiplyTerm;
    dfloat pics2;

    #ifdef PHI_DIST
        const dfloat phiVar_phi_init = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M3_PHI_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
        dfloat h_phi = (phiVar_phi_init - PHI_ONE) / (PHI_TWO - PHI_ONE);
        h_phi = fmaxf(0.0_df, fminf(1.0_df, h_phi));
        const dfloat rho_pf   = PHI_RHO_PHASE1 + PHI_DRHO_PHASE12 * h_phi;
        const dfloat invRhoPF = 1.0_df / rho_pf;
    #endif //PHI_DIST

    #include COLREC_RECONSTRUCTION
    
    //thread xyz
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;

    
    //block xyz
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int bz = blockIdx.z;

    #if defined(STEP7_ESOTERIC_TWIST) && defined(D3Q19)
    #include "fragments/popInit_esoteric_twist.inc"
    #elif defined(STEP7_ESOTERIC_PUSH) && defined(D3Q19)
    #include "fragments/popInit_esoteric_push.inc"
    #elif defined(STEP7_ESOTERIC_PULL) && defined(D3Q19)
    #include "fragments/popInit_esoteric_pull.inc"
    #else
    if (threadIdx.x == 0) { //w
        ghostInterface.pop.X_0[idxPopX(ty, tz, 0, bx, by, bz)] = pop[ 2]; 
        ghostInterface.pop.X_0[idxPopX(ty, tz, 1, bx, by, bz)] = pop[ 8];
        ghostInterface.pop.X_0[idxPopX(ty, tz, 2, bx, by, bz)] = pop[10];
        ghostInterface.pop.X_0[idxPopX(ty, tz, 3, bx, by, bz)] = pop[14];
        ghostInterface.pop.X_0[idxPopX(ty, tz, 4, bx, by, bz)] = pop[16];
        #ifdef D3Q27                                                                                                           
        ghostInterface.pop.X_0[idxPopX(ty, tz, 5, bx, by, bz)] = pop[20];
        ghostInterface.pop.X_0[idxPopX(ty, tz, 6, bx, by, bz)] = pop[22];
        ghostInterface.pop.X_0[idxPopX(ty, tz, 7, bx, by, bz)] = pop[24];
        ghostInterface.pop.X_0[idxPopX(ty, tz, 8, bx, by, bz)] = pop[25];
        #endif //D3Q27                                                                                                           
    }else if (threadIdx.x == (BLOCK_NX - 1)){                                                                                                                                                                               
        ghostInterface.pop.X_1[idxPopX(ty, tz, 0, bx, by, bz)] = pop[ 1];
        ghostInterface.pop.X_1[idxPopX(ty, tz, 1, bx, by, bz)] = pop[ 7];
        ghostInterface.pop.X_1[idxPopX(ty, tz, 2, bx, by, bz)] = pop[ 9];
        ghostInterface.pop.X_1[idxPopX(ty, tz, 3, bx, by, bz)] = pop[13];
        ghostInterface.pop.X_1[idxPopX(ty, tz, 4, bx, by, bz)] = pop[15];
        #ifdef D3Q27                                                                                                           
        ghostInterface.pop.X_1[idxPopX(ty, tz, 5, bx, by, bz)] = pop[19];
        ghostInterface.pop.X_1[idxPopX(ty, tz, 6, bx, by, bz)] = pop[21];
        ghostInterface.pop.X_1[idxPopX(ty, tz, 7, bx, by, bz)] = pop[23];
        ghostInterface.pop.X_1[idxPopX(ty, tz, 8, bx, by, bz)] = pop[26];
        #endif //D3Q27       
    }

    if (threadIdx.y == 0)  { //s                                                                                                                                                                                        
        ghostInterface.pop.Y_0[idxPopY(tx, tz, 0, bx, by, bz)] = pop[ 4];
        ghostInterface.pop.Y_0[idxPopY(tx, tz, 1, bx, by, bz)] = pop[ 8];
        ghostInterface.pop.Y_0[idxPopY(tx, tz, 2, bx, by, bz)] = pop[12];
        ghostInterface.pop.Y_0[idxPopY(tx, tz, 3, bx, by, bz)] = pop[13];
        ghostInterface.pop.Y_0[idxPopY(tx, tz, 4, bx, by, bz)] = pop[18];
        #ifdef D3Q27                                                                                                           
        ghostInterface.pop.Y_0[idxPopY(tx, tz, 5, bx, by, bz)] = pop[20];
        ghostInterface.pop.Y_0[idxPopY(tx, tz, 6, bx, by, bz)] = pop[22];
        ghostInterface.pop.Y_0[idxPopY(tx, tz, 7, bx, by, bz)] = pop[23];
        ghostInterface.pop.Y_0[idxPopY(tx, tz, 8, bx, by, bz)] = pop[26];
        #endif //D3Q27                                                                                                           
    }else if (threadIdx.y == (BLOCK_NY - 1)){                                                                                                                                                                        
        ghostInterface.pop.Y_1[idxPopY(tx, tz, 0, bx, by, bz)] = pop[ 3];
        ghostInterface.pop.Y_1[idxPopY(tx, tz, 1, bx, by, bz)] = pop[ 7];
        ghostInterface.pop.Y_1[idxPopY(tx, tz, 2, bx, by, bz)] = pop[11];
        ghostInterface.pop.Y_1[idxPopY(tx, tz, 3, bx, by, bz)] = pop[14];
        ghostInterface.pop.Y_1[idxPopY(tx, tz, 4, bx, by, bz)] = pop[17];
        #ifdef D3Q27                                                                                                           
        ghostInterface.pop.Y_1[idxPopY(tx, tz, 5, bx, by, bz)] = pop[19];
        ghostInterface.pop.Y_1[idxPopY(tx, tz, 6, bx, by, bz)] = pop[21];
        ghostInterface.pop.Y_1[idxPopY(tx, tz, 7, bx, by, bz)] = pop[24];
        ghostInterface.pop.Y_1[idxPopY(tx, tz, 8, bx, by, bz)] = pop[25];
        #endif //D3Q27                                                                                                           
    }
    
    if (threadIdx.z == 0){ //b                                                                                                                                                                                     
        ghostInterface.pop.Z_0[idxPopZ(tx, ty, 0, bx, by, bz)] = pop[ 6];
        ghostInterface.pop.Z_0[idxPopZ(tx, ty, 1, bx, by, bz)] = pop[10];
        ghostInterface.pop.Z_0[idxPopZ(tx, ty, 2, bx, by, bz)] = pop[12];
        ghostInterface.pop.Z_0[idxPopZ(tx, ty, 3, bx, by, bz)] = pop[15];
        ghostInterface.pop.Z_0[idxPopZ(tx, ty, 4, bx, by, bz)] = pop[17];
        #ifdef D3Q27                                                                                                           
        ghostInterface.pop.Z_0[idxPopZ(tx, ty, 5, bx, by, bz)] = pop[20];
        ghostInterface.pop.Z_0[idxPopZ(tx, ty, 6, bx, by, bz)] = pop[21];
        ghostInterface.pop.Z_0[idxPopZ(tx, ty, 7, bx, by, bz)] = pop[24];
        ghostInterface.pop.Z_0[idxPopZ(tx, ty, 8, bx, by, bz)] = pop[26];
        #endif //D3Q27                                                                                                           
    }else if (threadIdx.z == (BLOCK_NZ - 1)){                                                                                                               
        ghostInterface.pop.Z_1[idxPopZ(tx, ty, 0, bx, by, bz)] = pop[ 5];
        ghostInterface.pop.Z_1[idxPopZ(tx, ty, 1, bx, by, bz)] = pop[ 9];
        ghostInterface.pop.Z_1[idxPopZ(tx, ty, 2, bx, by, bz)] = pop[11];
        ghostInterface.pop.Z_1[idxPopZ(tx, ty, 3, bx, by, bz)] = pop[16];
        ghostInterface.pop.Z_1[idxPopZ(tx, ty, 4, bx, by, bz)] = pop[18];
        #ifdef D3Q27                                                                                                           
        ghostInterface.pop.Z_1[idxPopZ(tx, ty, 5, bx, by, bz)] = pop[19];
        ghostInterface.pop.Z_1[idxPopZ(tx, ty, 6, bx, by, bz)] = pop[22];
        ghostInterface.pop.Z_1[idxPopZ(tx, ty, 7, bx, by, bz)] = pop[23];
        ghostInterface.pop.Z_1[idxPopZ(tx, ty, 8, bx, by, bz)] = pop[25];
        #endif //D3Q27                                                                                                                                                                                                                    
    }
    #endif

    #ifdef CONVECTION_DIFFUSION_TRANSPORT
        dfloat gNode[GQ];


    #ifdef SECOND_DIST 
        
        dfloat cVar = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M2_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
        dfloat invC = 1/cVar;
        dfloat qx_t30 = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M2_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
        dfloat qy_t30 = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M2_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
        dfloat qz_t30 = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M2_CZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];

        #include COLREC_G_RECONSTRUCTION

        if (threadIdx.x == 0) { //w
            ghostInterface.g.X_0[g_idxPopX(ty, tz, 0, bx, by, bz)] = gNode[ 2]; 
            #ifdef D3G19
            ghostInterface.g.X_0[g_idxPopX(ty, tz, 1, bx, by, bz)] = gNode[ 8];
            ghostInterface.g.X_0[g_idxPopX(ty, tz, 2, bx, by, bz)] = gNode[10];
            ghostInterface.g.X_0[g_idxPopX(ty, tz, 3, bx, by, bz)] = gNode[14];
            ghostInterface.g.X_0[g_idxPopX(ty, tz, 4, bx, by, bz)] = gNode[16];
            #endif            
            #ifdef D3Q27
            ghostInterface.g.X_0[g_idxPopX(ty, tz, 1, bx, by, bz)] = gNode[ 8];
            ghostInterface.g.X_0[g_idxPopX(ty, tz, 2, bx, by, bz)] = gNode[10];
            ghostInterface.g.X_0[g_idxPopX(ty, tz, 3, bx, by, bz)] = gNode[14];
            ghostInterface.g.X_0[g_idxPopX(ty, tz, 4, bx, by, bz)] = gNode[16];
            ghostInterface.g.X_0[g_idxPopX(ty, tz, 5, bx, by, bz)] = gNode[20];
            ghostInterface.g.X_0[g_idxPopX(ty, tz, 6, bx, by, bz)] = gNode[22];
            ghostInterface.g.X_0[g_idxPopX(ty, tz, 7, bx, by, bz)] = gNode[24];
            ghostInterface.g.X_0[g_idxPopX(ty, tz, 8, bx, by, bz)] = gNode[25];
            #endif
        }else if (threadIdx.x == (BLOCK_NX - 1)){                    
            ghostInterface.g.X_1[g_idxPopX(ty, tz, 0, bx, by, bz)] = gNode[ 1];
            #ifdef D3G19
            ghostInterface.g.X_1[g_idxPopX(ty, tz, 1, bx, by, bz)] = gNode[ 7];
            ghostInterface.g.X_1[g_idxPopX(ty, tz, 2, bx, by, bz)] = gNode[ 9];
            ghostInterface.g.X_1[g_idxPopX(ty, tz, 3, bx, by, bz)] = gNode[13];
            ghostInterface.g.X_1[g_idxPopX(ty, tz, 4, bx, by, bz)] = gNode[15];     
            #endif    
            #ifdef D3Q27
            ghostInterface.g.X_1[g_idxPopX(ty, tz, 1, bx, by, bz)] = gNode[ 7];
            ghostInterface.g.X_1[g_idxPopX(ty, tz, 2, bx, by, bz)] = gNode[ 9];
            ghostInterface.g.X_1[g_idxPopX(ty, tz, 3, bx, by, bz)] = gNode[13];
            ghostInterface.g.X_1[g_idxPopX(ty, tz, 4, bx, by, bz)] = gNode[15];
            ghostInterface.g.X_1[g_idxPopX(ty, tz, 5, bx, by, bz)] = gNode[19];
            ghostInterface.g.X_1[g_idxPopX(ty, tz, 6, bx, by, bz)] = gNode[21];
            ghostInterface.g.X_1[g_idxPopX(ty, tz, 7, bx, by, bz)] = gNode[23];
            ghostInterface.g.X_1[g_idxPopX(ty, tz, 8, bx, by, bz)] = gNode[26];
            #endif
        }
        if (threadIdx.y == 0)  { //s                             
            ghostInterface.g.Y_0[g_idxPopY(tx, tz, 0, bx, by, bz)] = gNode[ 4];
            #ifdef D3G19
            ghostInterface.g.Y_0[g_idxPopY(tx, tz, 1, bx, by, bz)] = gNode[ 8];
            ghostInterface.g.Y_0[g_idxPopY(tx, tz, 2, bx, by, bz)] = gNode[12];
            ghostInterface.g.Y_0[g_idxPopY(tx, tz, 3, bx, by, bz)] = gNode[13];
            ghostInterface.g.Y_0[g_idxPopY(tx, tz, 4, bx, by, bz)] = gNode[18];           
            #endif           
            #ifdef D3Q27
            ghostInterface.g.Y_0[g_idxPopY(tx, tz, 1, bx, by, bz)] = gNode[ 8];
            ghostInterface.g.Y_0[g_idxPopY(tx, tz, 2, bx, by, bz)] = gNode[12];
            ghostInterface.g.Y_0[g_idxPopY(tx, tz, 3, bx, by, bz)] = gNode[13];
            ghostInterface.g.Y_0[g_idxPopY(tx, tz, 4, bx, by, bz)] = gNode[18];
            ghostInterface.g.Y_0[g_idxPopY(tx, tz, 5, bx, by, bz)] = gNode[20];
            ghostInterface.g.Y_0[g_idxPopY(tx, tz, 6, bx, by, bz)] = gNode[22];
            ghostInterface.g.Y_0[g_idxPopY(tx, tz, 7, bx, by, bz)] = gNode[23];
            ghostInterface.g.Y_0[g_idxPopY(tx, tz, 8, bx, by, bz)] = gNode[26];
            #endif
        }else if (threadIdx.y == (BLOCK_NY - 1)){             
            ghostInterface.g.Y_1[g_idxPopY(tx, tz, 0, bx, by, bz)] = gNode[ 3];
            #ifdef D3G19
            ghostInterface.g.Y_1[g_idxPopY(tx, tz, 1, bx, by, bz)] = gNode[ 7];
            ghostInterface.g.Y_1[g_idxPopY(tx, tz, 2, bx, by, bz)] = gNode[11];
            ghostInterface.g.Y_1[g_idxPopY(tx, tz, 3, bx, by, bz)] = gNode[14];
            ghostInterface.g.Y_1[g_idxPopY(tx, tz, 4, bx, by, bz)] = gNode[17];         
            #endif        
            #ifdef D3Q27
            ghostInterface.g.Y_1[g_idxPopY(tx, tz, 1, bx, by, bz)] = gNode[ 7];
            ghostInterface.g.Y_1[g_idxPopY(tx, tz, 2, bx, by, bz)] = gNode[11];
            ghostInterface.g.Y_1[g_idxPopY(tx, tz, 3, bx, by, bz)] = gNode[14];
            ghostInterface.g.Y_1[g_idxPopY(tx, tz, 4, bx, by, bz)] = gNode[17];
            ghostInterface.g.Y_1[g_idxPopY(tx, tz, 5, bx, by, bz)] = gNode[19];
            ghostInterface.g.Y_1[g_idxPopY(tx, tz, 6, bx, by, bz)] = gNode[21];
            ghostInterface.g.Y_1[g_idxPopY(tx, tz, 7, bx, by, bz)] = gNode[24];
            ghostInterface.g.Y_1[g_idxPopY(tx, tz, 8, bx, by, bz)] = gNode[25];
            #endif
        }
        
        if (threadIdx.z == 0){ //b                          
            ghostInterface.g.Z_0[g_idxPopZ(tx, ty, 0, bx, by, bz)] = gNode[ 6];
            #ifdef D3G19
            ghostInterface.g.Z_0[g_idxPopZ(tx, ty, 1, bx, by, bz)] = gNode[10];
            ghostInterface.g.Z_0[g_idxPopZ(tx, ty, 2, bx, by, bz)] = gNode[12];
            ghostInterface.g.Z_0[g_idxPopZ(tx, ty, 3, bx, by, bz)] = gNode[15];
            ghostInterface.g.Z_0[g_idxPopZ(tx, ty, 4, bx, by, bz)] = gNode[17]; 
            #endif    
            #ifdef D3Q27
            ghostInterface.g.Z_0[g_idxPopZ(tx, ty, 1, bx, by, bz)] = gNode[10];
            ghostInterface.g.Z_0[g_idxPopZ(tx, ty, 2, bx, by, bz)] = gNode[12];
            ghostInterface.g.Z_0[g_idxPopZ(tx, ty, 3, bx, by, bz)] = gNode[15];
            ghostInterface.g.Z_0[g_idxPopZ(tx, ty, 4, bx, by, bz)] = gNode[17];
            ghostInterface.g.Z_0[g_idxPopZ(tx, ty, 5, bx, by, bz)] = gNode[20];
            ghostInterface.g.Z_0[g_idxPopZ(tx, ty, 6, bx, by, bz)] = gNode[21];
            ghostInterface.g.Z_0[g_idxPopZ(tx, ty, 7, bx, by, bz)] = gNode[24];
            ghostInterface.g.Z_0[g_idxPopZ(tx, ty, 8, bx, by, bz)] = gNode[26];
            #endif
        }else if (threadIdx.z == (BLOCK_NZ - 1)){                  
            ghostInterface.g.Z_1[g_idxPopZ(tx, ty, 0, bx, by, bz)] = gNode[ 5];
            #ifdef D3G19
            ghostInterface.g.Z_1[g_idxPopZ(tx, ty, 1, bx, by, bz)] = gNode[ 9];
            ghostInterface.g.Z_1[g_idxPopZ(tx, ty, 2, bx, by, bz)] = gNode[11];
            ghostInterface.g.Z_1[g_idxPopZ(tx, ty, 3, bx, by, bz)] = gNode[16];
            ghostInterface.g.Z_1[g_idxPopZ(tx, ty, 4, bx, by, bz)] = gNode[18];    
            #endif                    
            #ifdef D3Q27
            ghostInterface.g.Z_1[g_idxPopZ(tx, ty, 1, bx, by, bz)] = gNode[ 9];
            ghostInterface.g.Z_1[g_idxPopZ(tx, ty, 2, bx, by, bz)] = gNode[11];
            ghostInterface.g.Z_1[g_idxPopZ(tx, ty, 3, bx, by, bz)] = gNode[16];
            ghostInterface.g.Z_1[g_idxPopZ(tx, ty, 4, bx, by, bz)] = gNode[18];
            ghostInterface.g.Z_1[g_idxPopZ(tx, ty, 5, bx, by, bz)] = gNode[19];
            ghostInterface.g.Z_1[g_idxPopZ(tx, ty, 6, bx, by, bz)] = gNode[22];
            ghostInterface.g.Z_1[g_idxPopZ(tx, ty, 7, bx, by, bz)] = gNode[23];
            ghostInterface.g.Z_1[g_idxPopZ(tx, ty, 8, bx, by, bz)] = gNode[25];
            #endif
        }
    #endif //SECOND_DIST
    #ifdef PHI_DIST 
        
        dfloat phiVar = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M3_PHI_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
        dfloat invPhi = 1/phiVar;
        dfloat phi_qx_t30 = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M3_PX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
        dfloat phi_qy_t30 = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M3_PY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
        dfloat phi_qz_t30 = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M3_PZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];

        #include COLREC_PHI_RECONSTRUCTION

        if (threadIdx.x == 0) { //w
            ghostInterface.phi.X_0[g_idxPopX(ty, tz, 0, bx, by, bz)] = gNode[ 2]; 
            #ifdef D3G19
            ghostInterface.phi.X_0[g_idxPopX(ty, tz, 1, bx, by, bz)] = gNode[ 8];
            ghostInterface.phi.X_0[g_idxPopX(ty, tz, 2, bx, by, bz)] = gNode[10];
            ghostInterface.phi.X_0[g_idxPopX(ty, tz, 3, bx, by, bz)] = gNode[14];
            ghostInterface.phi.X_0[g_idxPopX(ty, tz, 4, bx, by, bz)] = gNode[16];
            #endif            
            #ifdef D3Q27
            ghostInterface.phi.X_0[g_idxPopX(ty, tz, 1, bx, by, bz)] = gNode[ 8];
            ghostInterface.phi.X_0[g_idxPopX(ty, tz, 2, bx, by, bz)] = gNode[10];
            ghostInterface.phi.X_0[g_idxPopX(ty, tz, 3, bx, by, bz)] = gNode[14];
            ghostInterface.phi.X_0[g_idxPopX(ty, tz, 4, bx, by, bz)] = gNode[16];
            ghostInterface.phi.X_0[g_idxPopX(ty, tz, 5, bx, by, bz)] = gNode[20];
            ghostInterface.phi.X_0[g_idxPopX(ty, tz, 6, bx, by, bz)] = gNode[22];
            ghostInterface.phi.X_0[g_idxPopX(ty, tz, 7, bx, by, bz)] = gNode[24];
            ghostInterface.phi.X_0[g_idxPopX(ty, tz, 8, bx, by, bz)] = gNode[25];
            #endif
        }else if (threadIdx.x == (BLOCK_NX - 1)){                    
            ghostInterface.phi.X_1[g_idxPopX(ty, tz, 0, bx, by, bz)] = gNode[ 1];
            #ifdef D3G19
            ghostInterface.phi.X_1[g_idxPopX(ty, tz, 1, bx, by, bz)] = gNode[ 7];
            ghostInterface.phi.X_1[g_idxPopX(ty, tz, 2, bx, by, bz)] = gNode[ 9];
            ghostInterface.phi.X_1[g_idxPopX(ty, tz, 3, bx, by, bz)] = gNode[13];
            ghostInterface.phi.X_1[g_idxPopX(ty, tz, 4, bx, by, bz)] = gNode[15];     
            #endif    
            #ifdef D3Q27
            ghostInterface.phi.X_1[g_idxPopX(ty, tz, 1, bx, by, bz)] = gNode[ 7];
            ghostInterface.phi.X_1[g_idxPopX(ty, tz, 2, bx, by, bz)] = gNode[ 9];
            ghostInterface.phi.X_1[g_idxPopX(ty, tz, 3, bx, by, bz)] = gNode[13];
            ghostInterface.phi.X_1[g_idxPopX(ty, tz, 4, bx, by, bz)] = gNode[15];
            ghostInterface.phi.X_1[g_idxPopX(ty, tz, 5, bx, by, bz)] = gNode[19];
            ghostInterface.phi.X_1[g_idxPopX(ty, tz, 6, bx, by, bz)] = gNode[21];
            ghostInterface.phi.X_1[g_idxPopX(ty, tz, 7, bx, by, bz)] = gNode[23];
            ghostInterface.phi.X_1[g_idxPopX(ty, tz, 8, bx, by, bz)] = gNode[26];
            #endif
        }

        if (threadIdx.y == 0)  { //s                             
            ghostInterface.phi.Y_0[g_idxPopY(tx, tz, 0, bx, by, bz)] = gNode[ 4];
            #ifdef D3G19
            ghostInterface.phi.Y_0[g_idxPopY(tx, tz, 1, bx, by, bz)] = gNode[ 8];
            ghostInterface.phi.Y_0[g_idxPopY(tx, tz, 2, bx, by, bz)] = gNode[12];
            ghostInterface.phi.Y_0[g_idxPopY(tx, tz, 3, bx, by, bz)] = gNode[13];
            ghostInterface.phi.Y_0[g_idxPopY(tx, tz, 4, bx, by, bz)] = gNode[18];           
            #endif           
            #ifdef D3Q27
            ghostInterface.phi.Y_0[g_idxPopY(tx, tz, 1, bx, by, bz)] = gNode[ 8];
            ghostInterface.phi.Y_0[g_idxPopY(tx, tz, 2, bx, by, bz)] = gNode[12];
            ghostInterface.phi.Y_0[g_idxPopY(tx, tz, 3, bx, by, bz)] = gNode[13];
            ghostInterface.phi.Y_0[g_idxPopY(tx, tz, 4, bx, by, bz)] = gNode[18];
            ghostInterface.phi.Y_0[g_idxPopY(tx, tz, 5, bx, by, bz)] = gNode[20];
            ghostInterface.phi.Y_0[g_idxPopY(tx, tz, 6, bx, by, bz)] = gNode[22];
            ghostInterface.phi.Y_0[g_idxPopY(tx, tz, 7, bx, by, bz)] = gNode[23];
            ghostInterface.phi.Y_0[g_idxPopY(tx, tz, 8, bx, by, bz)] = gNode[26];
            #endif
        }else if (threadIdx.y == (BLOCK_NY - 1)){             
            ghostInterface.phi.Y_1[g_idxPopY(tx, tz, 0, bx, by, bz)] = gNode[ 3];
            #ifdef D3G19
            ghostInterface.phi.Y_1[g_idxPopY(tx, tz, 1, bx, by, bz)] = gNode[ 7];
            ghostInterface.phi.Y_1[g_idxPopY(tx, tz, 2, bx, by, bz)] = gNode[11];
            ghostInterface.phi.Y_1[g_idxPopY(tx, tz, 3, bx, by, bz)] = gNode[14];
            ghostInterface.phi.Y_1[g_idxPopY(tx, tz, 4, bx, by, bz)] = gNode[17];         
            #endif        
            #ifdef D3Q27
            ghostInterface.phi.Y_1[g_idxPopY(tx, tz, 1, bx, by, bz)] = gNode[ 7];
            ghostInterface.phi.Y_1[g_idxPopY(tx, tz, 2, bx, by, bz)] = gNode[11];
            ghostInterface.phi.Y_1[g_idxPopY(tx, tz, 3, bx, by, bz)] = gNode[14];
            ghostInterface.phi.Y_1[g_idxPopY(tx, tz, 4, bx, by, bz)] = gNode[17];
            ghostInterface.phi.Y_1[g_idxPopY(tx, tz, 5, bx, by, bz)] = gNode[19];
            ghostInterface.phi.Y_1[g_idxPopY(tx, tz, 6, bx, by, bz)] = gNode[21];
            ghostInterface.phi.Y_1[g_idxPopY(tx, tz, 7, bx, by, bz)] = gNode[24];
            ghostInterface.phi.Y_1[g_idxPopY(tx, tz, 8, bx, by, bz)] = gNode[25];
            #endif
        }
        
        if (threadIdx.z == 0){ //b                          
            ghostInterface.phi.Z_0[g_idxPopZ(tx, ty, 0, bx, by, bz)] = gNode[ 6];
            #ifdef D3G19
            ghostInterface.phi.Z_0[g_idxPopZ(tx, ty, 1, bx, by, bz)] = gNode[10];
            ghostInterface.phi.Z_0[g_idxPopZ(tx, ty, 2, bx, by, bz)] = gNode[12];
            ghostInterface.phi.Z_0[g_idxPopZ(tx, ty, 3, bx, by, bz)] = gNode[15];
            ghostInterface.phi.Z_0[g_idxPopZ(tx, ty, 4, bx, by, bz)] = gNode[17]; 
            #endif    
            #ifdef D3Q27
            ghostInterface.phi.Z_0[g_idxPopZ(tx, ty, 1, bx, by, bz)] = gNode[10];
            ghostInterface.phi.Z_0[g_idxPopZ(tx, ty, 2, bx, by, bz)] = gNode[12];
            ghostInterface.phi.Z_0[g_idxPopZ(tx, ty, 3, bx, by, bz)] = gNode[15];
            ghostInterface.phi.Z_0[g_idxPopZ(tx, ty, 4, bx, by, bz)] = gNode[17];
            ghostInterface.phi.Z_0[g_idxPopZ(tx, ty, 5, bx, by, bz)] = gNode[20];
            ghostInterface.phi.Z_0[g_idxPopZ(tx, ty, 6, bx, by, bz)] = gNode[21];
            ghostInterface.phi.Z_0[g_idxPopZ(tx, ty, 7, bx, by, bz)] = gNode[24];
            ghostInterface.phi.Z_0[g_idxPopZ(tx, ty, 8, bx, by, bz)] = gNode[26];
            #endif
        }else if (threadIdx.z == (BLOCK_NZ - 1)){                  
            ghostInterface.phi.Z_1[g_idxPopZ(tx, ty, 0, bx, by, bz)] = gNode[ 5];
            #ifdef D3G19
            ghostInterface.phi.Z_1[g_idxPopZ(tx, ty, 1, bx, by, bz)] = gNode[ 9];
            ghostInterface.phi.Z_1[g_idxPopZ(tx, ty, 2, bx, by, bz)] = gNode[11];
            ghostInterface.phi.Z_1[g_idxPopZ(tx, ty, 3, bx, by, bz)] = gNode[16];
            ghostInterface.phi.Z_1[g_idxPopZ(tx, ty, 4, bx, by, bz)] = gNode[18];    
            #endif                    
            #ifdef D3Q27
            ghostInterface.phi.Z_1[g_idxPopZ(tx, ty, 1, bx, by, bz)] = gNode[ 9];
            ghostInterface.phi.Z_1[g_idxPopZ(tx, ty, 2, bx, by, bz)] = gNode[11];
            ghostInterface.phi.Z_1[g_idxPopZ(tx, ty, 3, bx, by, bz)] = gNode[16];
            ghostInterface.phi.Z_1[g_idxPopZ(tx, ty, 4, bx, by, bz)] = gNode[18];
            ghostInterface.phi.Z_1[g_idxPopZ(tx, ty, 5, bx, by, bz)] = gNode[19];
            ghostInterface.phi.Z_1[g_idxPopZ(tx, ty, 6, bx, by, bz)] = gNode[22];
            ghostInterface.phi.Z_1[g_idxPopZ(tx, ty, 7, bx, by, bz)] = gNode[23];
            ghostInterface.phi.Z_1[g_idxPopZ(tx, ty, 8, bx, by, bz)] = gNode[25];
            #endif
        }
    #endif //PHI_DIST
    #ifdef A_XX_DIST 
        
        dfloat AxxVar = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_XX_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
        dfloat invAxx = 1/AxxVar;
        dfloat Axx_qx_t30   = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_XX_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
        dfloat Axx_qy_t30   = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_XX_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
        dfloat Axx_qz_t30   = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_XX_CZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];


        #include COLREC_AXX_RECONSTRUCTION

        if (threadIdx.x == 0) { //w
            ghostInterface.Axx.X_0[g_idxPopX(ty, tz, 0, bx, by, bz)] = gNode[ 2]; 
            #ifdef D3G19
            ghostInterface.Axx.X_0[g_idxPopX(ty, tz, 1, bx, by, bz)] = gNode[ 8];
            ghostInterface.Axx.X_0[g_idxPopX(ty, tz, 2, bx, by, bz)] = gNode[10];
            ghostInterface.Axx.X_0[g_idxPopX(ty, tz, 3, bx, by, bz)] = gNode[14];
            ghostInterface.Axx.X_0[g_idxPopX(ty, tz, 4, bx, by, bz)] = gNode[16];
            #endif            
            #ifdef D3Q27
            ghostInterface.Axx.X_0[g_idxPopX(ty, tz, 1, bx, by, bz)] = gNode[ 8];
            ghostInterface.Axx.X_0[g_idxPopX(ty, tz, 2, bx, by, bz)] = gNode[10];
            ghostInterface.Axx.X_0[g_idxPopX(ty, tz, 3, bx, by, bz)] = gNode[14];
            ghostInterface.Axx.X_0[g_idxPopX(ty, tz, 4, bx, by, bz)] = gNode[16];
            ghostInterface.Axx.X_0[g_idxPopX(ty, tz, 5, bx, by, bz)] = gNode[20];
            ghostInterface.Axx.X_0[g_idxPopX(ty, tz, 6, bx, by, bz)] = gNode[22];
            ghostInterface.Axx.X_0[g_idxPopX(ty, tz, 7, bx, by, bz)] = gNode[24];
            ghostInterface.Axx.X_0[g_idxPopX(ty, tz, 8, bx, by, bz)] = gNode[25];
            #endif
        }else if (threadIdx.x == (BLOCK_NX - 1)){                    
            ghostInterface.Axx.X_1[g_idxPopX(ty, tz, 0, bx, by, bz)] = gNode[ 1];
            #ifdef D3G19
            ghostInterface.Axx.X_1[g_idxPopX(ty, tz, 1, bx, by, bz)] = gNode[ 7];
            ghostInterface.Axx.X_1[g_idxPopX(ty, tz, 2, bx, by, bz)] = gNode[ 9];
            ghostInterface.Axx.X_1[g_idxPopX(ty, tz, 3, bx, by, bz)] = gNode[13];
            ghostInterface.Axx.X_1[g_idxPopX(ty, tz, 4, bx, by, bz)] = gNode[15];     
            #endif    
            #ifdef D3Q27
            ghostInterface.Axx.X_1[g_idxPopX(ty, tz, 1, bx, by, bz)] = gNode[ 7];
            ghostInterface.Axx.X_1[g_idxPopX(ty, tz, 2, bx, by, bz)] = gNode[ 9];
            ghostInterface.Axx.X_1[g_idxPopX(ty, tz, 3, bx, by, bz)] = gNode[13];
            ghostInterface.Axx.X_1[g_idxPopX(ty, tz, 4, bx, by, bz)] = gNode[15];
            ghostInterface.Axx.X_1[g_idxPopX(ty, tz, 5, bx, by, bz)] = gNode[19];
            ghostInterface.Axx.X_1[g_idxPopX(ty, tz, 6, bx, by, bz)] = gNode[21];
            ghostInterface.Axx.X_1[g_idxPopX(ty, tz, 7, bx, by, bz)] = gNode[23];
            ghostInterface.Axx.X_1[g_idxPopX(ty, tz, 8, bx, by, bz)] = gNode[26];
            #endif
        }

        if (threadIdx.y == 0)  { //s                             
            ghostInterface.Axx.Y_0[g_idxPopY(tx, tz, 0, bx, by, bz)] = gNode[ 4];
            #ifdef D3G19
            ghostInterface.Axx.Y_0[g_idxPopY(tx, tz, 1, bx, by, bz)] = gNode[ 8];
            ghostInterface.Axx.Y_0[g_idxPopY(tx, tz, 2, bx, by, bz)] = gNode[12];
            ghostInterface.Axx.Y_0[g_idxPopY(tx, tz, 3, bx, by, bz)] = gNode[13];
            ghostInterface.Axx.Y_0[g_idxPopY(tx, tz, 4, bx, by, bz)] = gNode[18];           
            #endif           
            #ifdef D3Q27
            ghostInterface.Axx.Y_0[g_idxPopY(tx, tz, 1, bx, by, bz)] = gNode[ 8];
            ghostInterface.Axx.Y_0[g_idxPopY(tx, tz, 2, bx, by, bz)] = gNode[12];
            ghostInterface.Axx.Y_0[g_idxPopY(tx, tz, 3, bx, by, bz)] = gNode[13];
            ghostInterface.Axx.Y_0[g_idxPopY(tx, tz, 4, bx, by, bz)] = gNode[18];
            ghostInterface.Axx.Y_0[g_idxPopY(tx, tz, 5, bx, by, bz)] = gNode[20];
            ghostInterface.Axx.Y_0[g_idxPopY(tx, tz, 6, bx, by, bz)] = gNode[22];
            ghostInterface.Axx.Y_0[g_idxPopY(tx, tz, 7, bx, by, bz)] = gNode[23];
            ghostInterface.Axx.Y_0[g_idxPopY(tx, tz, 8, bx, by, bz)] = gNode[26];
            #endif
        }else if (threadIdx.y == (BLOCK_NY - 1)){             
            ghostInterface.Axx.Y_1[g_idxPopY(tx, tz, 0, bx, by, bz)] = gNode[ 3];
            #ifdef D3G19
            ghostInterface.Axx.Y_1[g_idxPopY(tx, tz, 1, bx, by, bz)] = gNode[ 7];
            ghostInterface.Axx.Y_1[g_idxPopY(tx, tz, 2, bx, by, bz)] = gNode[11];
            ghostInterface.Axx.Y_1[g_idxPopY(tx, tz, 3, bx, by, bz)] = gNode[14];
            ghostInterface.Axx.Y_1[g_idxPopY(tx, tz, 4, bx, by, bz)] = gNode[17];         
            #endif        
            #ifdef D3Q27
            ghostInterface.Axx.Y_1[g_idxPopY(tx, tz, 1, bx, by, bz)] = gNode[ 7];
            ghostInterface.Axx.Y_1[g_idxPopY(tx, tz, 2, bx, by, bz)] = gNode[11];
            ghostInterface.Axx.Y_1[g_idxPopY(tx, tz, 3, bx, by, bz)] = gNode[14];
            ghostInterface.Axx.Y_1[g_idxPopY(tx, tz, 4, bx, by, bz)] = gNode[17];
            ghostInterface.Axx.Y_1[g_idxPopY(tx, tz, 5, bx, by, bz)] = gNode[19];
            ghostInterface.Axx.Y_1[g_idxPopY(tx, tz, 6, bx, by, bz)] = gNode[21];
            ghostInterface.Axx.Y_1[g_idxPopY(tx, tz, 7, bx, by, bz)] = gNode[24];
            ghostInterface.Axx.Y_1[g_idxPopY(tx, tz, 8, bx, by, bz)] = gNode[25];
            #endif
        }
        
        if (threadIdx.z == 0){ //b                          
            ghostInterface.Axx.Z_0[g_idxPopZ(tx, ty, 0, bx, by, bz)] = gNode[ 6];
            #ifdef D3G19
            ghostInterface.Axx.Z_0[g_idxPopZ(tx, ty, 1, bx, by, bz)] = gNode[10];
            ghostInterface.Axx.Z_0[g_idxPopZ(tx, ty, 2, bx, by, bz)] = gNode[12];
            ghostInterface.Axx.Z_0[g_idxPopZ(tx, ty, 3, bx, by, bz)] = gNode[15];
            ghostInterface.Axx.Z_0[g_idxPopZ(tx, ty, 4, bx, by, bz)] = gNode[17]; 
            #endif    
            #ifdef D3Q27
            ghostInterface.Axx.Z_0[g_idxPopZ(tx, ty, 1, bx, by, bz)] = gNode[10];
            ghostInterface.Axx.Z_0[g_idxPopZ(tx, ty, 2, bx, by, bz)] = gNode[12];
            ghostInterface.Axx.Z_0[g_idxPopZ(tx, ty, 3, bx, by, bz)] = gNode[15];
            ghostInterface.Axx.Z_0[g_idxPopZ(tx, ty, 4, bx, by, bz)] = gNode[17];
            ghostInterface.Axx.Z_0[g_idxPopZ(tx, ty, 5, bx, by, bz)] = gNode[20];
            ghostInterface.Axx.Z_0[g_idxPopZ(tx, ty, 6, bx, by, bz)] = gNode[21];
            ghostInterface.Axx.Z_0[g_idxPopZ(tx, ty, 7, bx, by, bz)] = gNode[24];
            ghostInterface.Axx.Z_0[g_idxPopZ(tx, ty, 8, bx, by, bz)] = gNode[26];
            #endif
        }else if (threadIdx.z == (BLOCK_NZ - 1)){                  
            ghostInterface.Axx.Z_1[g_idxPopZ(tx, ty, 0, bx, by, bz)] = gNode[ 5];
            #ifdef D3G19
            ghostInterface.Axx.Z_1[g_idxPopZ(tx, ty, 1, bx, by, bz)] = gNode[ 9];
            ghostInterface.Axx.Z_1[g_idxPopZ(tx, ty, 2, bx, by, bz)] = gNode[11];
            ghostInterface.Axx.Z_1[g_idxPopZ(tx, ty, 3, bx, by, bz)] = gNode[16];
            ghostInterface.Axx.Z_1[g_idxPopZ(tx, ty, 4, bx, by, bz)] = gNode[18];    
            #endif                    
            #ifdef D3Q27
            ghostInterface.Axx.Z_1[g_idxPopZ(tx, ty, 1, bx, by, bz)] = gNode[ 9];
            ghostInterface.Axx.Z_1[g_idxPopZ(tx, ty, 2, bx, by, bz)] = gNode[11];
            ghostInterface.Axx.Z_1[g_idxPopZ(tx, ty, 3, bx, by, bz)] = gNode[16];
            ghostInterface.Axx.Z_1[g_idxPopZ(tx, ty, 4, bx, by, bz)] = gNode[18];
            ghostInterface.Axx.Z_1[g_idxPopZ(tx, ty, 5, bx, by, bz)] = gNode[19];
            ghostInterface.Axx.Z_1[g_idxPopZ(tx, ty, 6, bx, by, bz)] = gNode[22];
            ghostInterface.Axx.Z_1[g_idxPopZ(tx, ty, 7, bx, by, bz)] = gNode[23];
            ghostInterface.Axx.Z_1[g_idxPopZ(tx, ty, 8, bx, by, bz)] = gNode[25];
            #endif
        }
    #endif //A_XX_DIST
        #ifdef A_XY_DIST 
        
        dfloat AxyVar = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_XY_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
        dfloat invAxy = 1/AxyVar;
        dfloat Axy_qx_t30   = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_XY_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
        dfloat Axy_qy_t30   = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_XY_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
        dfloat Axy_qz_t30   = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_XY_CZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];

        #include COLREC_AXY_RECONSTRUCTION

        if (threadIdx.x == 0) { //w
            ghostInterface.Axy.X_0[g_idxPopX(ty, tz, 0, bx, by, bz)] = gNode[ 2]; 
            #ifdef D3G19
            ghostInterface.Axy.X_0[g_idxPopX(ty, tz, 1, bx, by, bz)] = gNode[ 8];
            ghostInterface.Axy.X_0[g_idxPopX(ty, tz, 2, bx, by, bz)] = gNode[10];
            ghostInterface.Axy.X_0[g_idxPopX(ty, tz, 3, bx, by, bz)] = gNode[14];
            ghostInterface.Axy.X_0[g_idxPopX(ty, tz, 4, bx, by, bz)] = gNode[16];
            #endif            
            #ifdef D3Q27
            ghostInterface.Axy.X_0[g_idxPopX(ty, tz, 1, bx, by, bz)] = gNode[ 8];
            ghostInterface.Axy.X_0[g_idxPopX(ty, tz, 2, bx, by, bz)] = gNode[10];
            ghostInterface.Axy.X_0[g_idxPopX(ty, tz, 3, bx, by, bz)] = gNode[14];
            ghostInterface.Axy.X_0[g_idxPopX(ty, tz, 4, bx, by, bz)] = gNode[16];
            ghostInterface.Axy.X_0[g_idxPopX(ty, tz, 5, bx, by, bz)] = gNode[20];
            ghostInterface.Axy.X_0[g_idxPopX(ty, tz, 6, bx, by, bz)] = gNode[22];
            ghostInterface.Axy.X_0[g_idxPopX(ty, tz, 7, bx, by, bz)] = gNode[24];
            ghostInterface.Axy.X_0[g_idxPopX(ty, tz, 8, bx, by, bz)] = gNode[25];
            #endif
        }else if (threadIdx.x == (BLOCK_NX - 1)){                    
            ghostInterface.Axy.X_1[g_idxPopX(ty, tz, 0, bx, by, bz)] = gNode[ 1];
            #ifdef D3G19
            ghostInterface.Axy.X_1[g_idxPopX(ty, tz, 1, bx, by, bz)] = gNode[ 7];
            ghostInterface.Axy.X_1[g_idxPopX(ty, tz, 2, bx, by, bz)] = gNode[ 9];
            ghostInterface.Axy.X_1[g_idxPopX(ty, tz, 3, bx, by, bz)] = gNode[13];
            ghostInterface.Axy.X_1[g_idxPopX(ty, tz, 4, bx, by, bz)] = gNode[15];     
            #endif    
            #ifdef D3Q27
            ghostInterface.Axy.X_1[g_idxPopX(ty, tz, 1, bx, by, bz)] = gNode[ 7];
            ghostInterface.Axy.X_1[g_idxPopX(ty, tz, 2, bx, by, bz)] = gNode[ 9];
            ghostInterface.Axy.X_1[g_idxPopX(ty, tz, 3, bx, by, bz)] = gNode[13];
            ghostInterface.Axy.X_1[g_idxPopX(ty, tz, 4, bx, by, bz)] = gNode[15];
            ghostInterface.Axy.X_1[g_idxPopX(ty, tz, 5, bx, by, bz)] = gNode[19];
            ghostInterface.Axy.X_1[g_idxPopX(ty, tz, 6, bx, by, bz)] = gNode[21];
            ghostInterface.Axy.X_1[g_idxPopX(ty, tz, 7, bx, by, bz)] = gNode[23];
            ghostInterface.Axy.X_1[g_idxPopX(ty, tz, 8, bx, by, bz)] = gNode[26];
            #endif
        }

        if (threadIdx.y == 0)  { //s                             
            ghostInterface.Axy.Y_0[g_idxPopY(tx, tz, 0, bx, by, bz)] = gNode[ 4];
            #ifdef D3G19
            ghostInterface.Axy.Y_0[g_idxPopY(tx, tz, 1, bx, by, bz)] = gNode[ 8];
            ghostInterface.Axy.Y_0[g_idxPopY(tx, tz, 2, bx, by, bz)] = gNode[12];
            ghostInterface.Axy.Y_0[g_idxPopY(tx, tz, 3, bx, by, bz)] = gNode[13];
            ghostInterface.Axy.Y_0[g_idxPopY(tx, tz, 4, bx, by, bz)] = gNode[18];           
            #endif           
            #ifdef D3Q27
            ghostInterface.Axy.Y_0[g_idxPopY(tx, tz, 1, bx, by, bz)] = gNode[ 8];
            ghostInterface.Axy.Y_0[g_idxPopY(tx, tz, 2, bx, by, bz)] = gNode[12];
            ghostInterface.Axy.Y_0[g_idxPopY(tx, tz, 3, bx, by, bz)] = gNode[13];
            ghostInterface.Axy.Y_0[g_idxPopY(tx, tz, 4, bx, by, bz)] = gNode[18];
            ghostInterface.Axy.Y_0[g_idxPopY(tx, tz, 5, bx, by, bz)] = gNode[20];
            ghostInterface.Axy.Y_0[g_idxPopY(tx, tz, 6, bx, by, bz)] = gNode[22];
            ghostInterface.Axy.Y_0[g_idxPopY(tx, tz, 7, bx, by, bz)] = gNode[23];
            ghostInterface.Axy.Y_0[g_idxPopY(tx, tz, 8, bx, by, bz)] = gNode[26];
            #endif
        }else if (threadIdx.y == (BLOCK_NY - 1)){             
            ghostInterface.Axy.Y_1[g_idxPopY(tx, tz, 0, bx, by, bz)] = gNode[ 3];
            #ifdef D3G19
            ghostInterface.Axy.Y_1[g_idxPopY(tx, tz, 1, bx, by, bz)] = gNode[ 7];
            ghostInterface.Axy.Y_1[g_idxPopY(tx, tz, 2, bx, by, bz)] = gNode[11];
            ghostInterface.Axy.Y_1[g_idxPopY(tx, tz, 3, bx, by, bz)] = gNode[14];
            ghostInterface.Axy.Y_1[g_idxPopY(tx, tz, 4, bx, by, bz)] = gNode[17];         
            #endif        
            #ifdef D3Q27
            ghostInterface.Axy.Y_1[g_idxPopY(tx, tz, 1, bx, by, bz)] = gNode[ 7];
            ghostInterface.Axy.Y_1[g_idxPopY(tx, tz, 2, bx, by, bz)] = gNode[11];
            ghostInterface.Axy.Y_1[g_idxPopY(tx, tz, 3, bx, by, bz)] = gNode[14];
            ghostInterface.Axy.Y_1[g_idxPopY(tx, tz, 4, bx, by, bz)] = gNode[17];
            ghostInterface.Axy.Y_1[g_idxPopY(tx, tz, 5, bx, by, bz)] = gNode[19];
            ghostInterface.Axy.Y_1[g_idxPopY(tx, tz, 6, bx, by, bz)] = gNode[21];
            ghostInterface.Axy.Y_1[g_idxPopY(tx, tz, 7, bx, by, bz)] = gNode[24];
            ghostInterface.Axy.Y_1[g_idxPopY(tx, tz, 8, bx, by, bz)] = gNode[25];
            #endif
        }
        
        if (threadIdx.z == 0){ //b                          
            ghostInterface.Axy.Z_0[g_idxPopZ(tx, ty, 0, bx, by, bz)] = gNode[ 6];
            #ifdef D3G19
            ghostInterface.Axy.Z_0[g_idxPopZ(tx, ty, 1, bx, by, bz)] = gNode[10];
            ghostInterface.Axy.Z_0[g_idxPopZ(tx, ty, 2, bx, by, bz)] = gNode[12];
            ghostInterface.Axy.Z_0[g_idxPopZ(tx, ty, 3, bx, by, bz)] = gNode[15];
            ghostInterface.Axy.Z_0[g_idxPopZ(tx, ty, 4, bx, by, bz)] = gNode[17]; 
            #endif    
            #ifdef D3Q27
            ghostInterface.Axy.Z_0[g_idxPopZ(tx, ty, 1, bx, by, bz)] = gNode[10];
            ghostInterface.Axy.Z_0[g_idxPopZ(tx, ty, 2, bx, by, bz)] = gNode[12];
            ghostInterface.Axy.Z_0[g_idxPopZ(tx, ty, 3, bx, by, bz)] = gNode[15];
            ghostInterface.Axy.Z_0[g_idxPopZ(tx, ty, 4, bx, by, bz)] = gNode[17];
            ghostInterface.Axy.Z_0[g_idxPopZ(tx, ty, 5, bx, by, bz)] = gNode[20];
            ghostInterface.Axy.Z_0[g_idxPopZ(tx, ty, 6, bx, by, bz)] = gNode[21];
            ghostInterface.Axy.Z_0[g_idxPopZ(tx, ty, 7, bx, by, bz)] = gNode[24];
            ghostInterface.Axy.Z_0[g_idxPopZ(tx, ty, 8, bx, by, bz)] = gNode[26];
            #endif
        }else if (threadIdx.z == (BLOCK_NZ - 1)){                  
            ghostInterface.Axy.Z_1[g_idxPopZ(tx, ty, 0, bx, by, bz)] = gNode[ 5];
            #ifdef D3G19
            ghostInterface.Axy.Z_1[g_idxPopZ(tx, ty, 1, bx, by, bz)] = gNode[ 9];
            ghostInterface.Axy.Z_1[g_idxPopZ(tx, ty, 2, bx, by, bz)] = gNode[11];
            ghostInterface.Axy.Z_1[g_idxPopZ(tx, ty, 3, bx, by, bz)] = gNode[16];
            ghostInterface.Axy.Z_1[g_idxPopZ(tx, ty, 4, bx, by, bz)] = gNode[18];    
            #endif                    
            #ifdef D3Q27
            ghostInterface.Axy.Z_1[g_idxPopZ(tx, ty, 1, bx, by, bz)] = gNode[ 9];
            ghostInterface.Axy.Z_1[g_idxPopZ(tx, ty, 2, bx, by, bz)] = gNode[11];
            ghostInterface.Axy.Z_1[g_idxPopZ(tx, ty, 3, bx, by, bz)] = gNode[16];
            ghostInterface.Axy.Z_1[g_idxPopZ(tx, ty, 4, bx, by, bz)] = gNode[18];
            ghostInterface.Axy.Z_1[g_idxPopZ(tx, ty, 5, bx, by, bz)] = gNode[19];
            ghostInterface.Axy.Z_1[g_idxPopZ(tx, ty, 6, bx, by, bz)] = gNode[22];
            ghostInterface.Axy.Z_1[g_idxPopZ(tx, ty, 7, bx, by, bz)] = gNode[23];
            ghostInterface.Axy.Z_1[g_idxPopZ(tx, ty, 8, bx, by, bz)] = gNode[25];
            #endif
        }
    #endif //A_XY_DIST
    #ifdef A_XZ_DIST 
        
        dfloat AxzVar = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_XZ_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
        dfloat invAxz = 1/AxzVar;
        dfloat Axz_qx_t30   = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_XZ_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
        dfloat Axz_qy_t30   = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_XZ_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
        dfloat Axz_qz_t30   = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_XZ_CZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];

        #include COLREC_AXZ_RECONSTRUCTION

        if (threadIdx.x == 0) { //w
            ghostInterface.Axz.X_0[g_idxPopX(ty, tz, 0, bx, by, bz)] = gNode[ 2]; 
            #ifdef D3G19
            ghostInterface.Axz.X_0[g_idxPopX(ty, tz, 1, bx, by, bz)] = gNode[ 8];
            ghostInterface.Axz.X_0[g_idxPopX(ty, tz, 2, bx, by, bz)] = gNode[10];
            ghostInterface.Axz.X_0[g_idxPopX(ty, tz, 3, bx, by, bz)] = gNode[14];
            ghostInterface.Axz.X_0[g_idxPopX(ty, tz, 4, bx, by, bz)] = gNode[16];
            #endif            
            #ifdef D3Q27
            ghostInterface.Axz.X_0[g_idxPopX(ty, tz, 1, bx, by, bz)] = gNode[ 8];
            ghostInterface.Axz.X_0[g_idxPopX(ty, tz, 2, bx, by, bz)] = gNode[10];
            ghostInterface.Axz.X_0[g_idxPopX(ty, tz, 3, bx, by, bz)] = gNode[14];
            ghostInterface.Axz.X_0[g_idxPopX(ty, tz, 4, bx, by, bz)] = gNode[16];
            ghostInterface.Axz.X_0[g_idxPopX(ty, tz, 5, bx, by, bz)] = gNode[20];
            ghostInterface.Axz.X_0[g_idxPopX(ty, tz, 6, bx, by, bz)] = gNode[22];
            ghostInterface.Axz.X_0[g_idxPopX(ty, tz, 7, bx, by, bz)] = gNode[24];
            ghostInterface.Axz.X_0[g_idxPopX(ty, tz, 8, bx, by, bz)] = gNode[25];
            #endif
        }else if (threadIdx.x == (BLOCK_NX - 1)){                    
            ghostInterface.Axz.X_1[g_idxPopX(ty, tz, 0, bx, by, bz)] = gNode[ 1];
            #ifdef D3G19
            ghostInterface.Axz.X_1[g_idxPopX(ty, tz, 1, bx, by, bz)] = gNode[ 7];
            ghostInterface.Axz.X_1[g_idxPopX(ty, tz, 2, bx, by, bz)] = gNode[ 9];
            ghostInterface.Axz.X_1[g_idxPopX(ty, tz, 3, bx, by, bz)] = gNode[13];
            ghostInterface.Axz.X_1[g_idxPopX(ty, tz, 4, bx, by, bz)] = gNode[15];     
            #endif    
            #ifdef D3Q27
            ghostInterface.Axz.X_1[g_idxPopX(ty, tz, 1, bx, by, bz)] = gNode[ 7];
            ghostInterface.Axz.X_1[g_idxPopX(ty, tz, 2, bx, by, bz)] = gNode[ 9];
            ghostInterface.Axz.X_1[g_idxPopX(ty, tz, 3, bx, by, bz)] = gNode[13];
            ghostInterface.Axz.X_1[g_idxPopX(ty, tz, 4, bx, by, bz)] = gNode[15];
            ghostInterface.Axz.X_1[g_idxPopX(ty, tz, 5, bx, by, bz)] = gNode[19];
            ghostInterface.Axz.X_1[g_idxPopX(ty, tz, 6, bx, by, bz)] = gNode[21];
            ghostInterface.Axz.X_1[g_idxPopX(ty, tz, 7, bx, by, bz)] = gNode[23];
            ghostInterface.Axz.X_1[g_idxPopX(ty, tz, 8, bx, by, bz)] = gNode[26];
            #endif
        }

        if (threadIdx.y == 0)  { //s                             
            ghostInterface.Axz.Y_0[g_idxPopY(tx, tz, 0, bx, by, bz)] = gNode[ 4];
            #ifdef D3G19
            ghostInterface.Axz.Y_0[g_idxPopY(tx, tz, 1, bx, by, bz)] = gNode[ 8];
            ghostInterface.Axz.Y_0[g_idxPopY(tx, tz, 2, bx, by, bz)] = gNode[12];
            ghostInterface.Axz.Y_0[g_idxPopY(tx, tz, 3, bx, by, bz)] = gNode[13];
            ghostInterface.Axz.Y_0[g_idxPopY(tx, tz, 4, bx, by, bz)] = gNode[18];           
            #endif           
            #ifdef D3Q27
            ghostInterface.Axz.Y_0[g_idxPopY(tx, tz, 1, bx, by, bz)] = gNode[ 8];
            ghostInterface.Axz.Y_0[g_idxPopY(tx, tz, 2, bx, by, bz)] = gNode[12];
            ghostInterface.Axz.Y_0[g_idxPopY(tx, tz, 3, bx, by, bz)] = gNode[13];
            ghostInterface.Axz.Y_0[g_idxPopY(tx, tz, 4, bx, by, bz)] = gNode[18];
            ghostInterface.Axz.Y_0[g_idxPopY(tx, tz, 5, bx, by, bz)] = gNode[20];
            ghostInterface.Axz.Y_0[g_idxPopY(tx, tz, 6, bx, by, bz)] = gNode[22];
            ghostInterface.Axz.Y_0[g_idxPopY(tx, tz, 7, bx, by, bz)] = gNode[23];
            ghostInterface.Axz.Y_0[g_idxPopY(tx, tz, 8, bx, by, bz)] = gNode[26];
            #endif
        }else if (threadIdx.y == (BLOCK_NY - 1)){             
            ghostInterface.Axz.Y_1[g_idxPopY(tx, tz, 0, bx, by, bz)] = gNode[ 3];
            #ifdef D3G19
            ghostInterface.Axz.Y_1[g_idxPopY(tx, tz, 1, bx, by, bz)] = gNode[ 7];
            ghostInterface.Axz.Y_1[g_idxPopY(tx, tz, 2, bx, by, bz)] = gNode[11];
            ghostInterface.Axz.Y_1[g_idxPopY(tx, tz, 3, bx, by, bz)] = gNode[14];
            ghostInterface.Axz.Y_1[g_idxPopY(tx, tz, 4, bx, by, bz)] = gNode[17];         
            #endif        
            #ifdef D3Q27
            ghostInterface.Axz.Y_1[g_idxPopY(tx, tz, 1, bx, by, bz)] = gNode[ 7];
            ghostInterface.Axz.Y_1[g_idxPopY(tx, tz, 2, bx, by, bz)] = gNode[11];
            ghostInterface.Axz.Y_1[g_idxPopY(tx, tz, 3, bx, by, bz)] = gNode[14];
            ghostInterface.Axz.Y_1[g_idxPopY(tx, tz, 4, bx, by, bz)] = gNode[17];
            ghostInterface.Axz.Y_1[g_idxPopY(tx, tz, 5, bx, by, bz)] = gNode[19];
            ghostInterface.Axz.Y_1[g_idxPopY(tx, tz, 6, bx, by, bz)] = gNode[21];
            ghostInterface.Axz.Y_1[g_idxPopY(tx, tz, 7, bx, by, bz)] = gNode[24];
            ghostInterface.Axz.Y_1[g_idxPopY(tx, tz, 8, bx, by, bz)] = gNode[25];
            #endif
        }
        
        if (threadIdx.z == 0){ //b                          
            ghostInterface.Axz.Z_0[g_idxPopZ(tx, ty, 0, bx, by, bz)] = gNode[ 6];
            #ifdef D3G19
            ghostInterface.Axz.Z_0[g_idxPopZ(tx, ty, 1, bx, by, bz)] = gNode[10];
            ghostInterface.Axz.Z_0[g_idxPopZ(tx, ty, 2, bx, by, bz)] = gNode[12];
            ghostInterface.Axz.Z_0[g_idxPopZ(tx, ty, 3, bx, by, bz)] = gNode[15];
            ghostInterface.Axz.Z_0[g_idxPopZ(tx, ty, 4, bx, by, bz)] = gNode[17]; 
            #endif    
            #ifdef D3Q27
            ghostInterface.Axz.Z_0[g_idxPopZ(tx, ty, 1, bx, by, bz)] = gNode[10];
            ghostInterface.Axz.Z_0[g_idxPopZ(tx, ty, 2, bx, by, bz)] = gNode[12];
            ghostInterface.Axz.Z_0[g_idxPopZ(tx, ty, 3, bx, by, bz)] = gNode[15];
            ghostInterface.Axz.Z_0[g_idxPopZ(tx, ty, 4, bx, by, bz)] = gNode[17];
            ghostInterface.Axz.Z_0[g_idxPopZ(tx, ty, 5, bx, by, bz)] = gNode[20];
            ghostInterface.Axz.Z_0[g_idxPopZ(tx, ty, 6, bx, by, bz)] = gNode[21];
            ghostInterface.Axz.Z_0[g_idxPopZ(tx, ty, 7, bx, by, bz)] = gNode[24];
            ghostInterface.Axz.Z_0[g_idxPopZ(tx, ty, 8, bx, by, bz)] = gNode[26];
            #endif
        }else if (threadIdx.z == (BLOCK_NZ - 1)){                  
            ghostInterface.Axz.Z_1[g_idxPopZ(tx, ty, 0, bx, by, bz)] = gNode[ 5];
            #ifdef D3G19
            ghostInterface.Axz.Z_1[g_idxPopZ(tx, ty, 1, bx, by, bz)] = gNode[ 9];
            ghostInterface.Axz.Z_1[g_idxPopZ(tx, ty, 2, bx, by, bz)] = gNode[11];
            ghostInterface.Axz.Z_1[g_idxPopZ(tx, ty, 3, bx, by, bz)] = gNode[16];
            ghostInterface.Axz.Z_1[g_idxPopZ(tx, ty, 4, bx, by, bz)] = gNode[18];    
            #endif                    
            #ifdef D3Q27
            ghostInterface.Axz.Z_1[g_idxPopZ(tx, ty, 1, bx, by, bz)] = gNode[ 9];
            ghostInterface.Axz.Z_1[g_idxPopZ(tx, ty, 2, bx, by, bz)] = gNode[11];
            ghostInterface.Axz.Z_1[g_idxPopZ(tx, ty, 3, bx, by, bz)] = gNode[16];
            ghostInterface.Axz.Z_1[g_idxPopZ(tx, ty, 4, bx, by, bz)] = gNode[18];
            ghostInterface.Axz.Z_1[g_idxPopZ(tx, ty, 5, bx, by, bz)] = gNode[19];
            ghostInterface.Axz.Z_1[g_idxPopZ(tx, ty, 6, bx, by, bz)] = gNode[22];
            ghostInterface.Axz.Z_1[g_idxPopZ(tx, ty, 7, bx, by, bz)] = gNode[23];
            ghostInterface.Axz.Z_1[g_idxPopZ(tx, ty, 8, bx, by, bz)] = gNode[25];
            #endif
        }
    #endif //A_XZ_DIST
    #ifdef A_YY_DIST 
        
        dfloat AyyVar = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_YY_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
        dfloat invAyy = 1/AyyVar;
        dfloat Ayy_qx_t30   = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_YY_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
        dfloat Ayy_qy_t30   = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_YY_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
        dfloat Ayy_qz_t30   = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_YY_CZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];

        #include COLREC_AYY_RECONSTRUCTION

        if (threadIdx.x == 0) { //w
            ghostInterface.Ayy.X_0[g_idxPopX(ty, tz, 0, bx, by, bz)] = gNode[ 2]; 
            #ifdef D3G19
            ghostInterface.Ayy.X_0[g_idxPopX(ty, tz, 1, bx, by, bz)] = gNode[ 8];
            ghostInterface.Ayy.X_0[g_idxPopX(ty, tz, 2, bx, by, bz)] = gNode[10];
            ghostInterface.Ayy.X_0[g_idxPopX(ty, tz, 3, bx, by, bz)] = gNode[14];
            ghostInterface.Ayy.X_0[g_idxPopX(ty, tz, 4, bx, by, bz)] = gNode[16];
            #endif            
            #ifdef D3Q27
            ghostInterface.Ayy.X_0[g_idxPopX(ty, tz, 1, bx, by, bz)] = gNode[ 8];
            ghostInterface.Ayy.X_0[g_idxPopX(ty, tz, 2, bx, by, bz)] = gNode[10];
            ghostInterface.Ayy.X_0[g_idxPopX(ty, tz, 3, bx, by, bz)] = gNode[14];
            ghostInterface.Ayy.X_0[g_idxPopX(ty, tz, 4, bx, by, bz)] = gNode[16];
            ghostInterface.Ayy.X_0[g_idxPopX(ty, tz, 5, bx, by, bz)] = gNode[20];
            ghostInterface.Ayy.X_0[g_idxPopX(ty, tz, 6, bx, by, bz)] = gNode[22];
            ghostInterface.Ayy.X_0[g_idxPopX(ty, tz, 7, bx, by, bz)] = gNode[24];
            ghostInterface.Ayy.X_0[g_idxPopX(ty, tz, 8, bx, by, bz)] = gNode[25];
            #endif
        }else if (threadIdx.x == (BLOCK_NX - 1)){                    
            ghostInterface.Ayy.X_1[g_idxPopX(ty, tz, 0, bx, by, bz)] = gNode[ 1];
            #ifdef D3G19
            ghostInterface.Ayy.X_1[g_idxPopX(ty, tz, 1, bx, by, bz)] = gNode[ 7];
            ghostInterface.Ayy.X_1[g_idxPopX(ty, tz, 2, bx, by, bz)] = gNode[ 9];
            ghostInterface.Ayy.X_1[g_idxPopX(ty, tz, 3, bx, by, bz)] = gNode[13];
            ghostInterface.Ayy.X_1[g_idxPopX(ty, tz, 4, bx, by, bz)] = gNode[15];     
            #endif    
            #ifdef D3Q27
            ghostInterface.Ayy.X_1[g_idxPopX(ty, tz, 1, bx, by, bz)] = gNode[ 7];
            ghostInterface.Ayy.X_1[g_idxPopX(ty, tz, 2, bx, by, bz)] = gNode[ 9];
            ghostInterface.Ayy.X_1[g_idxPopX(ty, tz, 3, bx, by, bz)] = gNode[13];
            ghostInterface.Ayy.X_1[g_idxPopX(ty, tz, 4, bx, by, bz)] = gNode[15];
            ghostInterface.Ayy.X_1[g_idxPopX(ty, tz, 5, bx, by, bz)] = gNode[19];
            ghostInterface.Ayy.X_1[g_idxPopX(ty, tz, 6, bx, by, bz)] = gNode[21];
            ghostInterface.Ayy.X_1[g_idxPopX(ty, tz, 7, bx, by, bz)] = gNode[23];
            ghostInterface.Ayy.X_1[g_idxPopX(ty, tz, 8, bx, by, bz)] = gNode[26];
            #endif
        }

        if (threadIdx.y == 0)  { //s                             
            ghostInterface.Ayy.Y_0[g_idxPopY(tx, tz, 0, bx, by, bz)] = gNode[ 4];
            #ifdef D3G19
            ghostInterface.Ayy.Y_0[g_idxPopY(tx, tz, 1, bx, by, bz)] = gNode[ 8];
            ghostInterface.Ayy.Y_0[g_idxPopY(tx, tz, 2, bx, by, bz)] = gNode[12];
            ghostInterface.Ayy.Y_0[g_idxPopY(tx, tz, 3, bx, by, bz)] = gNode[13];
            ghostInterface.Ayy.Y_0[g_idxPopY(tx, tz, 4, bx, by, bz)] = gNode[18];           
            #endif           
            #ifdef D3Q27
            ghostInterface.Ayy.Y_0[g_idxPopY(tx, tz, 1, bx, by, bz)] = gNode[ 8];
            ghostInterface.Ayy.Y_0[g_idxPopY(tx, tz, 2, bx, by, bz)] = gNode[12];
            ghostInterface.Ayy.Y_0[g_idxPopY(tx, tz, 3, bx, by, bz)] = gNode[13];
            ghostInterface.Ayy.Y_0[g_idxPopY(tx, tz, 4, bx, by, bz)] = gNode[18];
            ghostInterface.Ayy.Y_0[g_idxPopY(tx, tz, 5, bx, by, bz)] = gNode[20];
            ghostInterface.Ayy.Y_0[g_idxPopY(tx, tz, 6, bx, by, bz)] = gNode[22];
            ghostInterface.Ayy.Y_0[g_idxPopY(tx, tz, 7, bx, by, bz)] = gNode[23];
            ghostInterface.Ayy.Y_0[g_idxPopY(tx, tz, 8, bx, by, bz)] = gNode[26];
            #endif
        }else if (threadIdx.y == (BLOCK_NY - 1)){             
            ghostInterface.Ayy.Y_1[g_idxPopY(tx, tz, 0, bx, by, bz)] = gNode[ 3];
            #ifdef D3G19
            ghostInterface.Ayy.Y_1[g_idxPopY(tx, tz, 1, bx, by, bz)] = gNode[ 7];
            ghostInterface.Ayy.Y_1[g_idxPopY(tx, tz, 2, bx, by, bz)] = gNode[11];
            ghostInterface.Ayy.Y_1[g_idxPopY(tx, tz, 3, bx, by, bz)] = gNode[14];
            ghostInterface.Ayy.Y_1[g_idxPopY(tx, tz, 4, bx, by, bz)] = gNode[17];         
            #endif        
            #ifdef D3Q27
            ghostInterface.Ayy.Y_1[g_idxPopY(tx, tz, 1, bx, by, bz)] = gNode[ 7];
            ghostInterface.Ayy.Y_1[g_idxPopY(tx, tz, 2, bx, by, bz)] = gNode[11];
            ghostInterface.Ayy.Y_1[g_idxPopY(tx, tz, 3, bx, by, bz)] = gNode[14];
            ghostInterface.Ayy.Y_1[g_idxPopY(tx, tz, 4, bx, by, bz)] = gNode[17];
            ghostInterface.Ayy.Y_1[g_idxPopY(tx, tz, 5, bx, by, bz)] = gNode[19];
            ghostInterface.Ayy.Y_1[g_idxPopY(tx, tz, 6, bx, by, bz)] = gNode[21];
            ghostInterface.Ayy.Y_1[g_idxPopY(tx, tz, 7, bx, by, bz)] = gNode[24];
            ghostInterface.Ayy.Y_1[g_idxPopY(tx, tz, 8, bx, by, bz)] = gNode[25];
            #endif
        }
        
        if (threadIdx.z == 0){ //b                          
            ghostInterface.Ayy.Z_0[g_idxPopZ(tx, ty, 0, bx, by, bz)] = gNode[ 6];
            #ifdef D3G19
            ghostInterface.Ayy.Z_0[g_idxPopZ(tx, ty, 1, bx, by, bz)] = gNode[10];
            ghostInterface.Ayy.Z_0[g_idxPopZ(tx, ty, 2, bx, by, bz)] = gNode[12];
            ghostInterface.Ayy.Z_0[g_idxPopZ(tx, ty, 3, bx, by, bz)] = gNode[15];
            ghostInterface.Ayy.Z_0[g_idxPopZ(tx, ty, 4, bx, by, bz)] = gNode[17]; 
            #endif    
            #ifdef D3Q27
            ghostInterface.Ayy.Z_0[g_idxPopZ(tx, ty, 1, bx, by, bz)] = gNode[10];
            ghostInterface.Ayy.Z_0[g_idxPopZ(tx, ty, 2, bx, by, bz)] = gNode[12];
            ghostInterface.Ayy.Z_0[g_idxPopZ(tx, ty, 3, bx, by, bz)] = gNode[15];
            ghostInterface.Ayy.Z_0[g_idxPopZ(tx, ty, 4, bx, by, bz)] = gNode[17];
            ghostInterface.Ayy.Z_0[g_idxPopZ(tx, ty, 5, bx, by, bz)] = gNode[20];
            ghostInterface.Ayy.Z_0[g_idxPopZ(tx, ty, 6, bx, by, bz)] = gNode[21];
            ghostInterface.Ayy.Z_0[g_idxPopZ(tx, ty, 7, bx, by, bz)] = gNode[24];
            ghostInterface.Ayy.Z_0[g_idxPopZ(tx, ty, 8, bx, by, bz)] = gNode[26];
            #endif
        }else if (threadIdx.z == (BLOCK_NZ - 1)){                  
            ghostInterface.Ayy.Z_1[g_idxPopZ(tx, ty, 0, bx, by, bz)] = gNode[ 5];
            #ifdef D3G19
            ghostInterface.Ayy.Z_1[g_idxPopZ(tx, ty, 1, bx, by, bz)] = gNode[ 9];
            ghostInterface.Ayy.Z_1[g_idxPopZ(tx, ty, 2, bx, by, bz)] = gNode[11];
            ghostInterface.Ayy.Z_1[g_idxPopZ(tx, ty, 3, bx, by, bz)] = gNode[16];
            ghostInterface.Ayy.Z_1[g_idxPopZ(tx, ty, 4, bx, by, bz)] = gNode[18];    
            #endif                    
            #ifdef D3Q27
            ghostInterface.Ayy.Z_1[g_idxPopZ(tx, ty, 1, bx, by, bz)] = gNode[ 9];
            ghostInterface.Ayy.Z_1[g_idxPopZ(tx, ty, 2, bx, by, bz)] = gNode[11];
            ghostInterface.Ayy.Z_1[g_idxPopZ(tx, ty, 3, bx, by, bz)] = gNode[16];
            ghostInterface.Ayy.Z_1[g_idxPopZ(tx, ty, 4, bx, by, bz)] = gNode[18];
            ghostInterface.Ayy.Z_1[g_idxPopZ(tx, ty, 5, bx, by, bz)] = gNode[19];
            ghostInterface.Ayy.Z_1[g_idxPopZ(tx, ty, 6, bx, by, bz)] = gNode[22];
            ghostInterface.Ayy.Z_1[g_idxPopZ(tx, ty, 7, bx, by, bz)] = gNode[23];
            ghostInterface.Ayy.Z_1[g_idxPopZ(tx, ty, 8, bx, by, bz)] = gNode[25];
            #endif
        }
    #endif //A_YY_DIST
        #ifdef A_YZ_DIST 
        
        dfloat AyzVar = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_YZ_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
        dfloat invAyz = 1/AyzVar;
        dfloat Ayz_qx_t30   = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_YZ_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
        dfloat Ayz_qy_t30   = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_YZ_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
        dfloat Ayz_qz_t30   = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_YZ_CZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];

        #include COLREC_AYZ_RECONSTRUCTION

        if (threadIdx.x == 0) { //w
            ghostInterface.Ayz.X_0[g_idxPopX(ty, tz, 0, bx, by, bz)] = gNode[ 2]; 
            #ifdef D3G19
            ghostInterface.Ayz.X_0[g_idxPopX(ty, tz, 1, bx, by, bz)] = gNode[ 8];
            ghostInterface.Ayz.X_0[g_idxPopX(ty, tz, 2, bx, by, bz)] = gNode[10];
            ghostInterface.Ayz.X_0[g_idxPopX(ty, tz, 3, bx, by, bz)] = gNode[14];
            ghostInterface.Ayz.X_0[g_idxPopX(ty, tz, 4, bx, by, bz)] = gNode[16];
            #endif            
            #ifdef D3Q27
            ghostInterface.Ayz.X_0[g_idxPopX(ty, tz, 1, bx, by, bz)] = gNode[ 8];
            ghostInterface.Ayz.X_0[g_idxPopX(ty, tz, 2, bx, by, bz)] = gNode[10];
            ghostInterface.Ayz.X_0[g_idxPopX(ty, tz, 3, bx, by, bz)] = gNode[14];
            ghostInterface.Ayz.X_0[g_idxPopX(ty, tz, 4, bx, by, bz)] = gNode[16];
            ghostInterface.Ayz.X_0[g_idxPopX(ty, tz, 5, bx, by, bz)] = gNode[20];
            ghostInterface.Ayz.X_0[g_idxPopX(ty, tz, 6, bx, by, bz)] = gNode[22];
            ghostInterface.Ayz.X_0[g_idxPopX(ty, tz, 7, bx, by, bz)] = gNode[24];
            ghostInterface.Ayz.X_0[g_idxPopX(ty, tz, 8, bx, by, bz)] = gNode[25];
            #endif
        }else if (threadIdx.x == (BLOCK_NX - 1)){                    
            ghostInterface.Ayz.X_1[g_idxPopX(ty, tz, 0, bx, by, bz)] = gNode[ 1];
            #ifdef D3G19
            ghostInterface.Ayz.X_1[g_idxPopX(ty, tz, 1, bx, by, bz)] = gNode[ 7];
            ghostInterface.Ayz.X_1[g_idxPopX(ty, tz, 2, bx, by, bz)] = gNode[ 9];
            ghostInterface.Ayz.X_1[g_idxPopX(ty, tz, 3, bx, by, bz)] = gNode[13];
            ghostInterface.Ayz.X_1[g_idxPopX(ty, tz, 4, bx, by, bz)] = gNode[15];     
            #endif    
            #ifdef D3Q27
            ghostInterface.Ayz.X_1[g_idxPopX(ty, tz, 1, bx, by, bz)] = gNode[ 7];
            ghostInterface.Ayz.X_1[g_idxPopX(ty, tz, 2, bx, by, bz)] = gNode[ 9];
            ghostInterface.Ayz.X_1[g_idxPopX(ty, tz, 3, bx, by, bz)] = gNode[13];
            ghostInterface.Ayz.X_1[g_idxPopX(ty, tz, 4, bx, by, bz)] = gNode[15];
            ghostInterface.Ayz.X_1[g_idxPopX(ty, tz, 5, bx, by, bz)] = gNode[19];
            ghostInterface.Ayz.X_1[g_idxPopX(ty, tz, 6, bx, by, bz)] = gNode[21];
            ghostInterface.Ayz.X_1[g_idxPopX(ty, tz, 7, bx, by, bz)] = gNode[23];
            ghostInterface.Ayz.X_1[g_idxPopX(ty, tz, 8, bx, by, bz)] = gNode[26];
            #endif
        }

        if (threadIdx.y == 0)  { //s                             
            ghostInterface.Ayz.Y_0[g_idxPopY(tx, tz, 0, bx, by, bz)] = gNode[ 4];
            #ifdef D3G19
            ghostInterface.Ayz.Y_0[g_idxPopY(tx, tz, 1, bx, by, bz)] = gNode[ 8];
            ghostInterface.Ayz.Y_0[g_idxPopY(tx, tz, 2, bx, by, bz)] = gNode[12];
            ghostInterface.Ayz.Y_0[g_idxPopY(tx, tz, 3, bx, by, bz)] = gNode[13];
            ghostInterface.Ayz.Y_0[g_idxPopY(tx, tz, 4, bx, by, bz)] = gNode[18];           
            #endif           
            #ifdef D3Q27
            ghostInterface.Ayz.Y_0[g_idxPopY(tx, tz, 1, bx, by, bz)] = gNode[ 8];
            ghostInterface.Ayz.Y_0[g_idxPopY(tx, tz, 2, bx, by, bz)] = gNode[12];
            ghostInterface.Ayz.Y_0[g_idxPopY(tx, tz, 3, bx, by, bz)] = gNode[13];
            ghostInterface.Ayz.Y_0[g_idxPopY(tx, tz, 4, bx, by, bz)] = gNode[18];
            ghostInterface.Ayz.Y_0[g_idxPopY(tx, tz, 5, bx, by, bz)] = gNode[20];
            ghostInterface.Ayz.Y_0[g_idxPopY(tx, tz, 6, bx, by, bz)] = gNode[22];
            ghostInterface.Ayz.Y_0[g_idxPopY(tx, tz, 7, bx, by, bz)] = gNode[23];
            ghostInterface.Ayz.Y_0[g_idxPopY(tx, tz, 8, bx, by, bz)] = gNode[26];
            #endif
        }else if (threadIdx.y == (BLOCK_NY - 1)){             
            ghostInterface.Ayz.Y_1[g_idxPopY(tx, tz, 0, bx, by, bz)] = gNode[ 3];
            #ifdef D3G19
            ghostInterface.Ayz.Y_1[g_idxPopY(tx, tz, 1, bx, by, bz)] = gNode[ 7];
            ghostInterface.Ayz.Y_1[g_idxPopY(tx, tz, 2, bx, by, bz)] = gNode[11];
            ghostInterface.Ayz.Y_1[g_idxPopY(tx, tz, 3, bx, by, bz)] = gNode[14];
            ghostInterface.Ayz.Y_1[g_idxPopY(tx, tz, 4, bx, by, bz)] = gNode[17];         
            #endif        
            #ifdef D3Q27
            ghostInterface.Ayz.Y_1[g_idxPopY(tx, tz, 1, bx, by, bz)] = gNode[ 7];
            ghostInterface.Ayz.Y_1[g_idxPopY(tx, tz, 2, bx, by, bz)] = gNode[11];
            ghostInterface.Ayz.Y_1[g_idxPopY(tx, tz, 3, bx, by, bz)] = gNode[14];
            ghostInterface.Ayz.Y_1[g_idxPopY(tx, tz, 4, bx, by, bz)] = gNode[17];
            ghostInterface.Ayz.Y_1[g_idxPopY(tx, tz, 5, bx, by, bz)] = gNode[19];
            ghostInterface.Ayz.Y_1[g_idxPopY(tx, tz, 6, bx, by, bz)] = gNode[21];
            ghostInterface.Ayz.Y_1[g_idxPopY(tx, tz, 7, bx, by, bz)] = gNode[24];
            ghostInterface.Ayz.Y_1[g_idxPopY(tx, tz, 8, bx, by, bz)] = gNode[25];
            #endif
        }
        
        if (threadIdx.z == 0){ //b                          
            ghostInterface.Ayz.Z_0[g_idxPopZ(tx, ty, 0, bx, by, bz)] = gNode[ 6];
            #ifdef D3G19
            ghostInterface.Ayz.Z_0[g_idxPopZ(tx, ty, 1, bx, by, bz)] = gNode[10];
            ghostInterface.Ayz.Z_0[g_idxPopZ(tx, ty, 2, bx, by, bz)] = gNode[12];
            ghostInterface.Ayz.Z_0[g_idxPopZ(tx, ty, 3, bx, by, bz)] = gNode[15];
            ghostInterface.Ayz.Z_0[g_idxPopZ(tx, ty, 4, bx, by, bz)] = gNode[17]; 
            #endif    
            #ifdef D3Q27
            ghostInterface.Ayz.Z_0[g_idxPopZ(tx, ty, 1, bx, by, bz)] = gNode[10];
            ghostInterface.Ayz.Z_0[g_idxPopZ(tx, ty, 2, bx, by, bz)] = gNode[12];
            ghostInterface.Ayz.Z_0[g_idxPopZ(tx, ty, 3, bx, by, bz)] = gNode[15];
            ghostInterface.Ayz.Z_0[g_idxPopZ(tx, ty, 4, bx, by, bz)] = gNode[17];
            ghostInterface.Ayz.Z_0[g_idxPopZ(tx, ty, 5, bx, by, bz)] = gNode[20];
            ghostInterface.Ayz.Z_0[g_idxPopZ(tx, ty, 6, bx, by, bz)] = gNode[21];
            ghostInterface.Ayz.Z_0[g_idxPopZ(tx, ty, 7, bx, by, bz)] = gNode[24];
            ghostInterface.Ayz.Z_0[g_idxPopZ(tx, ty, 8, bx, by, bz)] = gNode[26];
            #endif
        }else if (threadIdx.z == (BLOCK_NZ - 1)){                  
            ghostInterface.Ayz.Z_1[g_idxPopZ(tx, ty, 0, bx, by, bz)] = gNode[ 5];
            #ifdef D3G19
            ghostInterface.Ayz.Z_1[g_idxPopZ(tx, ty, 1, bx, by, bz)] = gNode[ 9];
            ghostInterface.Ayz.Z_1[g_idxPopZ(tx, ty, 2, bx, by, bz)] = gNode[11];
            ghostInterface.Ayz.Z_1[g_idxPopZ(tx, ty, 3, bx, by, bz)] = gNode[16];
            ghostInterface.Ayz.Z_1[g_idxPopZ(tx, ty, 4, bx, by, bz)] = gNode[18];    
            #endif                    
            #ifdef D3Q27
            ghostInterface.Ayz.Z_1[g_idxPopZ(tx, ty, 1, bx, by, bz)] = gNode[ 9];
            ghostInterface.Ayz.Z_1[g_idxPopZ(tx, ty, 2, bx, by, bz)] = gNode[11];
            ghostInterface.Ayz.Z_1[g_idxPopZ(tx, ty, 3, bx, by, bz)] = gNode[16];
            ghostInterface.Ayz.Z_1[g_idxPopZ(tx, ty, 4, bx, by, bz)] = gNode[18];
            ghostInterface.Ayz.Z_1[g_idxPopZ(tx, ty, 5, bx, by, bz)] = gNode[19];
            ghostInterface.Ayz.Z_1[g_idxPopZ(tx, ty, 6, bx, by, bz)] = gNode[22];
            ghostInterface.Ayz.Z_1[g_idxPopZ(tx, ty, 7, bx, by, bz)] = gNode[23];
            ghostInterface.Ayz.Z_1[g_idxPopZ(tx, ty, 8, bx, by, bz)] = gNode[25];
            #endif
        }
    #endif //A_YZ_DIST
        #ifdef A_ZZ_DIST 
        
        dfloat AzzVar = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_ZZ_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
        dfloat invAzz = 1/AzzVar;
        dfloat Azz_qx_t30   = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_ZZ_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
        dfloat Azz_qy_t30   = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_ZZ_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
        dfloat Azz_qz_t30   = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_ZZ_CZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];


        #include COLREC_AZZ_RECONSTRUCTION

        if (threadIdx.x == 0) { //w
            ghostInterface.Azz.X_0[g_idxPopX(ty, tz, 0, bx, by, bz)] = gNode[ 2]; 
            #ifdef D3G19
            ghostInterface.Azz.X_0[g_idxPopX(ty, tz, 1, bx, by, bz)] = gNode[ 8];
            ghostInterface.Azz.X_0[g_idxPopX(ty, tz, 2, bx, by, bz)] = gNode[10];
            ghostInterface.Azz.X_0[g_idxPopX(ty, tz, 3, bx, by, bz)] = gNode[14];
            ghostInterface.Azz.X_0[g_idxPopX(ty, tz, 4, bx, by, bz)] = gNode[16];
            #endif            
            #ifdef D3Q27
            ghostInterface.Azz.X_0[g_idxPopX(ty, tz, 1, bx, by, bz)] = gNode[ 8];
            ghostInterface.Azz.X_0[g_idxPopX(ty, tz, 2, bx, by, bz)] = gNode[10];
            ghostInterface.Azz.X_0[g_idxPopX(ty, tz, 3, bx, by, bz)] = gNode[14];
            ghostInterface.Azz.X_0[g_idxPopX(ty, tz, 4, bx, by, bz)] = gNode[16];
            ghostInterface.Azz.X_0[g_idxPopX(ty, tz, 5, bx, by, bz)] = gNode[20];
            ghostInterface.Azz.X_0[g_idxPopX(ty, tz, 6, bx, by, bz)] = gNode[22];
            ghostInterface.Azz.X_0[g_idxPopX(ty, tz, 7, bx, by, bz)] = gNode[24];
            ghostInterface.Azz.X_0[g_idxPopX(ty, tz, 8, bx, by, bz)] = gNode[25];
            #endif
        }else if (threadIdx.x == (BLOCK_NX - 1)){                    
            ghostInterface.Azz.X_1[g_idxPopX(ty, tz, 0, bx, by, bz)] = gNode[ 1];
            #ifdef D3G19
            ghostInterface.Azz.X_1[g_idxPopX(ty, tz, 1, bx, by, bz)] = gNode[ 7];
            ghostInterface.Azz.X_1[g_idxPopX(ty, tz, 2, bx, by, bz)] = gNode[ 9];
            ghostInterface.Azz.X_1[g_idxPopX(ty, tz, 3, bx, by, bz)] = gNode[13];
            ghostInterface.Azz.X_1[g_idxPopX(ty, tz, 4, bx, by, bz)] = gNode[15];     
            #endif    
            #ifdef D3Q27
            ghostInterface.Azz.X_1[g_idxPopX(ty, tz, 1, bx, by, bz)] = gNode[ 7];
            ghostInterface.Azz.X_1[g_idxPopX(ty, tz, 2, bx, by, bz)] = gNode[ 9];
            ghostInterface.Azz.X_1[g_idxPopX(ty, tz, 3, bx, by, bz)] = gNode[13];
            ghostInterface.Azz.X_1[g_idxPopX(ty, tz, 4, bx, by, bz)] = gNode[15];
            ghostInterface.Azz.X_1[g_idxPopX(ty, tz, 5, bx, by, bz)] = gNode[19];
            ghostInterface.Azz.X_1[g_idxPopX(ty, tz, 6, bx, by, bz)] = gNode[21];
            ghostInterface.Azz.X_1[g_idxPopX(ty, tz, 7, bx, by, bz)] = gNode[23];
            ghostInterface.Azz.X_1[g_idxPopX(ty, tz, 8, bx, by, bz)] = gNode[26];
            #endif
        }

        if (threadIdx.y == 0)  { //s                             
            ghostInterface.Azz.Y_0[g_idxPopY(tx, tz, 0, bx, by, bz)] = gNode[ 4];
            #ifdef D3G19
            ghostInterface.Azz.Y_0[g_idxPopY(tx, tz, 1, bx, by, bz)] = gNode[ 8];
            ghostInterface.Azz.Y_0[g_idxPopY(tx, tz, 2, bx, by, bz)] = gNode[12];
            ghostInterface.Azz.Y_0[g_idxPopY(tx, tz, 3, bx, by, bz)] = gNode[13];
            ghostInterface.Azz.Y_0[g_idxPopY(tx, tz, 4, bx, by, bz)] = gNode[18];           
            #endif           
            #ifdef D3Q27
            ghostInterface.Azz.Y_0[g_idxPopY(tx, tz, 1, bx, by, bz)] = gNode[ 8];
            ghostInterface.Azz.Y_0[g_idxPopY(tx, tz, 2, bx, by, bz)] = gNode[12];
            ghostInterface.Azz.Y_0[g_idxPopY(tx, tz, 3, bx, by, bz)] = gNode[13];
            ghostInterface.Azz.Y_0[g_idxPopY(tx, tz, 4, bx, by, bz)] = gNode[18];
            ghostInterface.Azz.Y_0[g_idxPopY(tx, tz, 5, bx, by, bz)] = gNode[20];
            ghostInterface.Azz.Y_0[g_idxPopY(tx, tz, 6, bx, by, bz)] = gNode[22];
            ghostInterface.Azz.Y_0[g_idxPopY(tx, tz, 7, bx, by, bz)] = gNode[23];
            ghostInterface.Azz.Y_0[g_idxPopY(tx, tz, 8, bx, by, bz)] = gNode[26];
            #endif
        }else if (threadIdx.y == (BLOCK_NY - 1)){             
            ghostInterface.Azz.Y_1[g_idxPopY(tx, tz, 0, bx, by, bz)] = gNode[ 3];
            #ifdef D3G19
            ghostInterface.Azz.Y_1[g_idxPopY(tx, tz, 1, bx, by, bz)] = gNode[ 7];
            ghostInterface.Azz.Y_1[g_idxPopY(tx, tz, 2, bx, by, bz)] = gNode[11];
            ghostInterface.Azz.Y_1[g_idxPopY(tx, tz, 3, bx, by, bz)] = gNode[14];
            ghostInterface.Azz.Y_1[g_idxPopY(tx, tz, 4, bx, by, bz)] = gNode[17];         
            #endif        
            #ifdef D3Q27
            ghostInterface.Azz.Y_1[g_idxPopY(tx, tz, 1, bx, by, bz)] = gNode[ 7];
            ghostInterface.Azz.Y_1[g_idxPopY(tx, tz, 2, bx, by, bz)] = gNode[11];
            ghostInterface.Azz.Y_1[g_idxPopY(tx, tz, 3, bx, by, bz)] = gNode[14];
            ghostInterface.Azz.Y_1[g_idxPopY(tx, tz, 4, bx, by, bz)] = gNode[17];
            ghostInterface.Azz.Y_1[g_idxPopY(tx, tz, 5, bx, by, bz)] = gNode[19];
            ghostInterface.Azz.Y_1[g_idxPopY(tx, tz, 6, bx, by, bz)] = gNode[21];
            ghostInterface.Azz.Y_1[g_idxPopY(tx, tz, 7, bx, by, bz)] = gNode[24];
            ghostInterface.Azz.Y_1[g_idxPopY(tx, tz, 8, bx, by, bz)] = gNode[25];
            #endif
        }
        
        if (threadIdx.z == 0){ //b                          
            ghostInterface.Azz.Z_0[g_idxPopZ(tx, ty, 0, bx, by, bz)] = gNode[ 6];
            #ifdef D3G19
            ghostInterface.Azz.Z_0[g_idxPopZ(tx, ty, 1, bx, by, bz)] = gNode[10];
            ghostInterface.Azz.Z_0[g_idxPopZ(tx, ty, 2, bx, by, bz)] = gNode[12];
            ghostInterface.Azz.Z_0[g_idxPopZ(tx, ty, 3, bx, by, bz)] = gNode[15];
            ghostInterface.Azz.Z_0[g_idxPopZ(tx, ty, 4, bx, by, bz)] = gNode[17]; 
            #endif    
            #ifdef D3Q27
            ghostInterface.Azz.Z_0[g_idxPopZ(tx, ty, 1, bx, by, bz)] = gNode[10];
            ghostInterface.Azz.Z_0[g_idxPopZ(tx, ty, 2, bx, by, bz)] = gNode[12];
            ghostInterface.Azz.Z_0[g_idxPopZ(tx, ty, 3, bx, by, bz)] = gNode[15];
            ghostInterface.Azz.Z_0[g_idxPopZ(tx, ty, 4, bx, by, bz)] = gNode[17];
            ghostInterface.Azz.Z_0[g_idxPopZ(tx, ty, 5, bx, by, bz)] = gNode[20];
            ghostInterface.Azz.Z_0[g_idxPopZ(tx, ty, 6, bx, by, bz)] = gNode[21];
            ghostInterface.Azz.Z_0[g_idxPopZ(tx, ty, 7, bx, by, bz)] = gNode[24];
            ghostInterface.Azz.Z_0[g_idxPopZ(tx, ty, 8, bx, by, bz)] = gNode[26];
            #endif
        }else if (threadIdx.z == (BLOCK_NZ - 1)){                  
            ghostInterface.Azz.Z_1[g_idxPopZ(tx, ty, 0, bx, by, bz)] = gNode[ 5];
            #ifdef D3G19
            ghostInterface.Azz.Z_1[g_idxPopZ(tx, ty, 1, bx, by, bz)] = gNode[ 9];
            ghostInterface.Azz.Z_1[g_idxPopZ(tx, ty, 2, bx, by, bz)] = gNode[11];
            ghostInterface.Azz.Z_1[g_idxPopZ(tx, ty, 3, bx, by, bz)] = gNode[16];
            ghostInterface.Azz.Z_1[g_idxPopZ(tx, ty, 4, bx, by, bz)] = gNode[18];    
            #endif                    
            #ifdef D3Q27
            ghostInterface.Azz.Z_1[g_idxPopZ(tx, ty, 1, bx, by, bz)] = gNode[ 9];
            ghostInterface.Azz.Z_1[g_idxPopZ(tx, ty, 2, bx, by, bz)] = gNode[11];
            ghostInterface.Azz.Z_1[g_idxPopZ(tx, ty, 3, bx, by, bz)] = gNode[16];
            ghostInterface.Azz.Z_1[g_idxPopZ(tx, ty, 4, bx, by, bz)] = gNode[18];
            ghostInterface.Azz.Z_1[g_idxPopZ(tx, ty, 5, bx, by, bz)] = gNode[19];
            ghostInterface.Azz.Z_1[g_idxPopZ(tx, ty, 6, bx, by, bz)] = gNode[22];
            ghostInterface.Azz.Z_1[g_idxPopZ(tx, ty, 7, bx, by, bz)] = gNode[23];
            ghostInterface.Azz.Z_1[g_idxPopZ(tx, ty, 8, bx, by, bz)] = gNode[25];
            #endif
        }
    #endif //A_ZZ_DIST
    #endif //CONVECTION_DIFFUSION_TRANSPORT   
}


__global__ void gpuInitialization_nodeType(
    unsigned int *dNodeType)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int z = threadIdx.z + blockDim.z * blockIdx.z;
    if (x >= NX || y >= NY || z >= NZ)
        return;
    
    unsigned int nodeType;

    #include CASE_BC_INIT

    dNodeType[idxScalarBlock(threadIdx.x, threadIdx.y, threadIdx.z,blockIdx.x, blockIdx.y, blockIdx.z)] = nodeType;
}

__host__ void hostInitialization_nodeType_bulk(
    unsigned int *hNodeType)
{
    int x,y,z;
    //unsigned int nodeType;

    for (x = 0; x<NX;x++){
        for (y = 0; y<NY;y++){
            for (z = 0; z<NZ_TOTAL;z++){
                hNodeType[idxScalarBlock(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)] = BULK;
            }
        }
    }
    printf("bulk done\n");
}

__host__ void hostInitialization_nodeType(
    unsigned int *hNodeType,
    int zStart,
    int localNZ
    #ifdef CURVED_BOUNDARY_CONDITION
    , unsigned int* numberCurvedBoundaryNodes
    #endif
){
    int x,y,z;
    unsigned int nodeType;

    for (x = 0; x<NX;x++){
        for (y = 0; y<NY;y++){
            for (z = zStart; z < zStart + localNZ; z++){


                int zLocal = z - zStart;
                
                
                #include CASE_BC_INIT
                
                
                if (nodeType != BULK){
                size_t idx =
                idxScalarBlock(
                x % BLOCK_NX,
                y % BLOCK_NY,
                zLocal % BLOCK_NZ,
                x / BLOCK_NX,
                y / BLOCK_NY,
                zLocal / BLOCK_NZ
                );
                
                
                hNodeType[idx] = (unsigned int)nodeType;
            // for (z = 0; z<NZ_TOTAL;z++){
                
            //     #include CASE_BC_INIT

            //     if (nodeType != BULK){
            //         hNodeType[idxScalarBlock(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)] = (unsigned int)nodeType;

                    #ifdef CURVED_BOUNDARY_CONDITION
                    if ( (nodeType & (0b111 << 8)) == (0b101 << 8) ){ //mask bits 8,9,10 then compare with BC_CURVED_BC
                        numberCurvedBoundaryNodes[0]++;
                    }
                    #endif
                }

            }
        }
    }

    printf("Setting boundary condition completed\n");
}

__global__ void gpuInitialization_force(
    dfloat *d_BC_Fx, dfloat* d_BC_Fy, dfloat* d_BC_Fz)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int z = threadIdx.z + blockDim.z * blockIdx.z;
    if (x >= NX || y >= NY || z >= NZ)
        return;

    size_t index = idxScalarGlobal(x, y, z);

    d_BC_Fx[idxScalarBlock(threadIdx.x, threadIdx.y, threadIdx.z,blockIdx.x, blockIdx.y, blockIdx.z)] = 0.0_df;
    d_BC_Fy[idxScalarBlock(threadIdx.x, threadIdx.y, threadIdx.z,blockIdx.x, blockIdx.y, blockIdx.z)] = 0.0_df;
    d_BC_Fz[idxScalarBlock(threadIdx.x, threadIdx.y, threadIdx.z,blockIdx.x, blockIdx.y, blockIdx.z)] = 0.0_df; 
}


#ifdef CURVED_BOUNDARY_CONDITION
__global__ void deviceInitializeCurvedBC(
    unsigned int *dNodeType, 
    CurvedBoundary** d_curvedBC
){
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int z = threadIdx.z + blockDim.z * blockIdx.z;
    if (x >= NX || y >= NY || z >= NZ)
        return;
    
     d_curvedBC[idxScalarBlock(threadIdx.x, threadIdx.y, threadIdx.z,blockIdx.x, blockIdx.y, blockIdx.z)] = nullptr;
}
#endif




void read_xyz_file(
    const std::string& filename,
    unsigned int* dNodeType
) {
    std::ifstream csv_file(filename);
    if (!csv_file)
    {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }

    int x, y, z;
    size_t index, index_n;

    int xi, yi, zi;

    std::string line;
    while (std::getline(csv_file, line)) {
        std::stringstream ss(line);
        std::string field;

        std::getline(ss, field, ',');
        x = std::stoi(field);

        std::getline(ss, field, ',');
        y = std::stoi(field);

        std::getline(ss, field, ',');
        z = std::stoi(field);

        if((x>=NX)||(y>=NY)||(z>=NZ_TOTAL))
            continue;


        index = idxScalarBlock(x % BLOCK_NX, y % BLOCK_NY, z % BLOCK_NZ, x / BLOCK_NX, y / BLOCK_NY, z / BLOCK_NZ);
        dNodeType[idxScalarBlock(x % BLOCK_NX, y % BLOCK_NY, z % BLOCK_NZ, x / BLOCK_NX, y / BLOCK_NY, z / BLOCK_NZ)] = SOLID_NODE;


        //set neighborings to be BC
        for (int xn = -1; xn < 2; xn++) {
            for (int yn = -1; yn < 2; yn++) {
                for (int zn = -1; zn < 2; zn++) {

                    xi = (x + xn + NX) % NX;
                    yi = (y + yn + NY) % NY;
                    zi = (z + zn + NZ) % NZ;


                    index_n = idxScalarBlock(xi% BLOCK_NX, yi % BLOCK_NY, zi % BLOCK_NZ, xi / BLOCK_NX, yi / BLOCK_NY, zi / BLOCK_NZ);

                    if ((index_n == index) || dNodeType[index_n] == 255) // check if is the center of the cuboid or if is already a solid node
                        continue;
                    else //set flag to max int 
                        dNodeType[index_n] = MISSING_DEFINITION;
                }
            }
        }
    }
    csv_file.close();
    printf("voxels imported \n");
}


__global__ 
void define_voxel_bc(
    unsigned int *dNodeType
){
    const int x = threadIdx.x + blockDim.x * blockIdx.x;
    const int y = threadIdx.y + blockDim.y * blockIdx.y;
    const int z = threadIdx.z + blockDim.z * blockIdx.z;
    if (x >= NX || y >= NY || z >= NZ)
        return;

    unsigned int index = idxScalarBlock(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ);
    if(dNodeType[index] == MISSING_DEFINITION){
        dNodeType[index] = bc_id(dNodeType,x,y,z);
        #ifdef CURVED_BOUNDARY_CONDITION //TODO: this only works if is the only boundary conditiond being made, if there is an obstacle will stop working
            if((dNodeType[index] != BULK )&& (dNodeType[index] != SOLID_NODE)){
                dNodeType[index] += BC_CURVED_BC_CONCAV;
            }
        #endif
    }
}



/*
Note: Due to the way the BC are set up, it possible when setting a solid node to also set the bit flags of neighboring nodes
However if attempt to perform in device, need to pay attention of two solid nodes setting the same flag at same time 
*/
__host__ __device__
unsigned int bc_id(unsigned int *dNodeType, int x, int y, int z){

    unsigned int bc_d = BULK;

    int xp1 = (x+1+NX)%NX;
    int xm1 = (x-1+NX)%NX;
    int yp1 = (y+1+NY)%NY;
    int ym1 = (y-1+NY)%NY;
    int zp1 = (z+1+NZ)%NZ;
    int zm1 = (z-1+NZ)%NZ;

    // 1
    if(dNodeType[idxScalarBlock(xp1%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, xp1/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)] == 255){
        bc_d |= (1 << 1);
        bc_d |= (1 << 3);
        bc_d |= (1 << 5);
        bc_d |= (1 << 7);
    }
     // 2
    if(dNodeType[idxScalarBlock(xm1%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, xm1/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)] == 255){
        bc_d |= (1 << 0);
        bc_d |= (1 << 2);
        bc_d |= (1 << 4);
        bc_d |= (1 << 6);
    }
    // 3
    if(dNodeType[idxScalarBlock(x%BLOCK_NX, yp1%BLOCK_NY, z%BLOCK_NZ, x/BLOCK_NX, yp1/BLOCK_NY, z/BLOCK_NZ)] == 255){
        bc_d |= (1 << 2);
        bc_d |= (1 << 3);
        bc_d |= (1 << 6);
        bc_d |= (1 << 7);
    }
    // 4
    if(dNodeType[idxScalarBlock(x%BLOCK_NX, ym1%BLOCK_NY, z%BLOCK_NZ, x/BLOCK_NX, ym1/BLOCK_NY, z/BLOCK_NZ)] == 255){
        bc_d |= (1 << 0);
        bc_d |= (1 << 1);
        bc_d |= (1 << 4);
        bc_d |= (1 << 5);
    }
    // 5
    if(dNodeType[idxScalarBlock(x%BLOCK_NX, y%BLOCK_NY, zp1%BLOCK_NZ, x/BLOCK_NX, y/BLOCK_NY, zp1/BLOCK_NZ)] == 255){
        bc_d |= (1 << 4);
        bc_d |= (1 << 5);
        bc_d |= (1 << 6);
        bc_d |= (1 << 7);
    }
    // 6
    if(dNodeType[idxScalarBlock(x%BLOCK_NX, y%BLOCK_NY, zm1%BLOCK_NZ, x/BLOCK_NX, y/BLOCK_NY, zm1/BLOCK_NZ)] == 255){
        bc_d |= (1 << 0);
        bc_d |= (1 << 1);
        bc_d |= (1 << 2);
        bc_d |= (1 << 3);
    }
    // 7
    if(dNodeType[idxScalarBlock(xp1%BLOCK_NX, yp1%BLOCK_NY, z%BLOCK_NZ, xp1/BLOCK_NX, yp1/BLOCK_NY, z/BLOCK_NZ)] == 255){
        bc_d |= (1 << 3);
        bc_d |= (1 << 7);
    }
    // 8
    if(dNodeType[idxScalarBlock(xm1%BLOCK_NX, ym1%BLOCK_NY, z%BLOCK_NZ, xm1/BLOCK_NX, ym1/BLOCK_NY, z/BLOCK_NZ)] == 255){
        bc_d |= (1 << 0);
        bc_d |= (1 << 4);
    }
    // 9
    if(dNodeType[idxScalarBlock(xp1%BLOCK_NX, y%BLOCK_NY, zp1%BLOCK_NZ, xp1/BLOCK_NX, y/BLOCK_NY, zp1/BLOCK_NZ)] == 255){
        bc_d |= (1 << 5);
        bc_d |= (1 << 7);
    }
    // 10
    if(dNodeType[idxScalarBlock(xm1%BLOCK_NX, y%BLOCK_NY, zm1%BLOCK_NZ, xm1/BLOCK_NX, y/BLOCK_NY, zm1/BLOCK_NZ)] == 255){
        bc_d |= (1 << 0);
        bc_d |= (1 << 2);
    }
    // 11
    if(dNodeType[idxScalarBlock(x%BLOCK_NX, yp1%BLOCK_NY, zp1%BLOCK_NZ, x/BLOCK_NX, yp1/BLOCK_NY, zp1/BLOCK_NZ)] == 255){
        bc_d |= (1 << 6);
        bc_d |= (1 << 7);
    }
    // 12
    if(dNodeType[idxScalarBlock(x%BLOCK_NX, ym1%BLOCK_NY, zm1%BLOCK_NZ, x/BLOCK_NX, ym1/BLOCK_NY, zm1/BLOCK_NZ)] == 255){
        bc_d |= (1 << 0);
        bc_d |= (1 << 1);
    }
    // 13
    if(dNodeType[idxScalarBlock(xp1%BLOCK_NX, ym1%BLOCK_NY, z%BLOCK_NZ, xp1/BLOCK_NX, ym1/BLOCK_NY, z/BLOCK_NZ)] == 255){
        bc_d |= (1 << 1);
        bc_d |= (1 << 5);
    }
    // 14
    if(dNodeType[idxScalarBlock(xm1%BLOCK_NX, yp1%BLOCK_NY, z%BLOCK_NZ, xm1/BLOCK_NX, yp1/BLOCK_NY, z/BLOCK_NZ)] == 255){
        bc_d |= (1 << 2);
        bc_d |= (1 << 6);
    }
    // 15
    if(dNodeType[idxScalarBlock(xp1%BLOCK_NX, y%BLOCK_NY, zm1%BLOCK_NZ, xp1/BLOCK_NX, y/BLOCK_NY, zm1/BLOCK_NZ)] == 255){
        bc_d |= (1 << 1);
        bc_d |= (1 << 3);
    }
    // 16
    if(dNodeType[idxScalarBlock(xm1%BLOCK_NX, y%BLOCK_NY, zp1%BLOCK_NZ, xm1/BLOCK_NX, y/BLOCK_NY, zp1/BLOCK_NZ)] == 255){
        bc_d |= (1 << 4);
        bc_d |= (1 << 6);
    }
    // 17
    if(dNodeType[idxScalarBlock(x%BLOCK_NX, yp1%BLOCK_NY, zm1%BLOCK_NZ, x/BLOCK_NX, yp1/BLOCK_NY, zm1/BLOCK_NZ)] == 255){
        bc_d |= (1 << 2);
        bc_d |= (1 << 3);
    }
    // 18
    if(dNodeType[idxScalarBlock(x%BLOCK_NX, ym1%BLOCK_NY, zp1%BLOCK_NZ, x/BLOCK_NX, ym1/BLOCK_NY, zp1/BLOCK_NZ)] == 255){
        bc_d |= (1 << 4);
        bc_d |= (1 << 5);
    }
    // 19
    if(dNodeType[idxScalarBlock(xp1%BLOCK_NX, yp1%BLOCK_NY, zp1%BLOCK_NZ, xp1/BLOCK_NX, yp1/BLOCK_NY, zp1/BLOCK_NZ)] == 255){
        bc_d |= (1 << 7);
    }
    // 20
    if(dNodeType[idxScalarBlock(xm1%BLOCK_NX, ym1%BLOCK_NY, zm1%BLOCK_NZ, xm1/BLOCK_NX, ym1/BLOCK_NY, zm1/BLOCK_NZ)] == 255){
        bc_d |= (1 << 0);
    }
    // 21
    if(dNodeType[idxScalarBlock(xp1%BLOCK_NX, yp1%BLOCK_NY, zm1%BLOCK_NZ, xp1/BLOCK_NX, yp1/BLOCK_NY, zm1/BLOCK_NZ)] == 255){
        bc_d |= (1 << 3);
    }
    // 22
    if(dNodeType[idxScalarBlock(xm1%BLOCK_NX, ym1%BLOCK_NY, zp1%BLOCK_NZ, xm1/BLOCK_NX, ym1/BLOCK_NY, zp1/BLOCK_NZ)] == 255){
        bc_d |= (1 << 4);
    }
    // 23
    if(dNodeType[idxScalarBlock(xp1%BLOCK_NX, ym1%BLOCK_NY, zp1%BLOCK_NZ, xp1/BLOCK_NX, ym1/BLOCK_NY, zp1/BLOCK_NZ)] == 255){
        bc_d |= (1 << 5);
    }
    // 24
    if(dNodeType[idxScalarBlock(xm1%BLOCK_NX, yp1%BLOCK_NY, zm1%BLOCK_NZ, xm1/BLOCK_NX, yp1/BLOCK_NY, zm1/BLOCK_NZ)] == 255){
        bc_d |= (1 << 2);
    }
    // 25
    if(dNodeType[idxScalarBlock(xm1%BLOCK_NX, yp1%BLOCK_NY, zp1%BLOCK_NZ, xm1/BLOCK_NX, yp1/BLOCK_NY, zp1/BLOCK_NZ)] == 255){
        bc_d |= (1 << 6);
    }
    // 26
    if(dNodeType[idxScalarBlock(xp1%BLOCK_NX, ym1%BLOCK_NY, zm1%BLOCK_NZ, xp1/BLOCK_NX, ym1/BLOCK_NY, zm1/BLOCK_NZ)] == 255){
        bc_d |= (1 << 1);   
    }

    return bc_d;
}


#ifdef CURVED_BOUNDARY_CONDITION
    unsigned int getNumberCurvedBoundaryNodes(const unsigned int *hNodeType, int localNZ){
        unsigned int numberCurvedBoundaryNodes = 0;
        for (int x = 0; x < NX; x++) {
            for (int y = 0; y < NY; y++) {
                for (int zLocal = 0; zLocal < localNZ; zLocal++) {
                    const size_t idx = idxScalarBlock(
                        x % BLOCK_NX,
                        y % BLOCK_NY,
                        zLocal % BLOCK_NZ,
                        x / BLOCK_NX,
                        y / BLOCK_NY,
                        zLocal / BLOCK_NZ);
                    const unsigned int nodeType = hNodeType[idx];
                    if ((nodeType & (0b111 << 8)) == (0b101 << 8)) {
                        numberCurvedBoundaryNodes++;
                    }
                }
            }
        }
        printf("Found %u curved boundary nodes\n", numberCurvedBoundaryNodes);
        return numberCurvedBoundaryNodes;
    }


    void allocateDeviceMemoryCurvedBoundary(CurvedBoundary** &d_curvedBC, CurvedBoundary* &d_curvedBC_array, unsigned int numberCurvedBoundaryNodes){
        unsigned int memAllocated = 0;

        checkCudaErrors(cudaMalloc((void**)&d_curvedBC, sizeof(CurvedBoundary*) * NUMBER_LBM_NODES_LOCAL));
        if (numberCurvedBoundaryNodes > 0) {
            checkCudaErrors(cudaMalloc((void**)&d_curvedBC_array, sizeof(CurvedBoundary) * numberCurvedBoundaryNodes));
        } else {
            d_curvedBC_array = nullptr;
        }

        memAllocated += sizeof(CurvedBoundary*) * NUMBER_LBM_NODES_LOCAL
                      + sizeof(CurvedBoundary) * numberCurvedBoundaryNodes;

        printf("Device Memory Allocated for Curved Boundary: %.2f MB \n", (float)memAllocated /(1024.0_df * 1024.0_df));
    }


    void initializeCurvedBoundaryArray(
        const unsigned int *hNodeType,
        unsigned int *dNodeType,
        CurvedBoundary** &d_curvedBC, 
        CurvedBoundary* &d_curvedBC_array, 
        unsigned int numberCurvedBoundaryNodes,
        int zStart,
        int localNZ
    ){
        CurvedBoundary** h_curvedBC_ptrs = (CurvedBoundary**)malloc(sizeof(CurvedBoundary*) * NUMBER_LBM_NODES_LOCAL);
        CurvedBoundary* h_curvedBC_array = numberCurvedBoundaryNodes > 0
            ? (CurvedBoundary*)malloc(sizeof(CurvedBoundary) * numberCurvedBoundaryNodes)
            : nullptr;

        for (size_t i = 0; i < NUMBER_LBM_NODES_LOCAL; i++){
            h_curvedBC_ptrs[i] = nullptr;
        }

        unsigned int curvedBCCount = 0;
        unsigned int idx;
        unsigned int nodeType;
        for(int x = 0; x < NX; x++){
            for(int y = 0; y < NY; y++){
                for(int zLocal = 0; zLocal < localNZ; zLocal++){
                    const int z = zStart + zLocal;
                    idx = idxScalarBlock(x%BLOCK_NX, y%BLOCK_NY, zLocal%BLOCK_NZ, x/BLOCK_NX, y/BLOCK_NY, zLocal/BLOCK_NZ);
                    nodeType = hNodeType[idx];
                    if((nodeType & (0b111 << 8)) == (0b101 << 8) ){ //mask bits 8,9,10 then compare with 
                        h_curvedBC_ptrs[idx] = d_curvedBC_array + curvedBCCount;
                        #include CASE_CURVED_BC_DEF

                        // Geometry is evaluated with global z coordinates, but interpolation
                        // indexes the local per-GPU moment array.
                        h_curvedBC_array[curvedBCCount].b.z -= zStart;
                        h_curvedBC_array[curvedBCCount].w.z -= zStart;
                        h_curvedBC_array[curvedBCCount].pf1.z -= zStart;
                        h_curvedBC_array[curvedBCCount].pf2.z -= zStart;
                        h_curvedBC_array[curvedBCCount].pf3.z -= zStart;
                        curvedBCCount++;
                    }
                }
            }
        }

        // Copy the indices to device
        checkCudaErrors(cudaMemcpy(d_curvedBC, h_curvedBC_ptrs, sizeof(CurvedBoundary*) * NUMBER_LBM_NODES_LOCAL, cudaMemcpyHostToDevice));
        // Copy the CurvedBoundary array to device
        if (numberCurvedBoundaryNodes > 0) {
            checkCudaErrors(cudaMemcpy(d_curvedBC_array, h_curvedBC_array, sizeof(CurvedBoundary) * numberCurvedBoundaryNodes, cudaMemcpyHostToDevice));
        }

        free(h_curvedBC_ptrs);
        free(h_curvedBC_array);
    }

    unsigned int initializeCurvedBoundaryDeviceField(const unsigned int *hNodeType, unsigned int *dNodeType, CurvedBoundary** &d_curvedBC, CurvedBoundary* &d_curvedBC_array, int zStart, int localNZ){
        unsigned int numberCurvedBoundaryNodes = getNumberCurvedBoundaryNodes(hNodeType, localNZ);
        allocateDeviceMemoryCurvedBoundary(d_curvedBC, d_curvedBC_array, numberCurvedBoundaryNodes);
        initializeCurvedBoundaryArray(hNodeType, dNodeType, d_curvedBC, d_curvedBC_array, numberCurvedBoundaryNodes, zStart, localNZ);
        return numberCurvedBoundaryNodes;
    }
#endif //CURVED_BOUNDARY_CONDITION
