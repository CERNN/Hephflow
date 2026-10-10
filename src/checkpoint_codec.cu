#include "checkpoint_codec.cuh"

#define CCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING



namespace {

__device__ __forceinline__ size_t canonicalNodeIndex(int x, int y, int zLocal)
{
    return static_cast<size_t>(x) + static_cast<size_t>(NX) * (static_cast<size_t>(y) + static_cast<size_t>(NY) * zLocal);
}

__device__ __forceinline__ ghostFacePtrs checkpointFacePtrs(const ghostData& faces, const ghostData& auxiliary)
{
    return {faces.X_0, faces.X_1, faces.Y_0, faces.Y_1, faces.Z_0, faces.Z_1, auxiliary.Z_0, auxiliary.Z_1};
}

    #ifdef CONVECTION_DIFFUSION_TRANSPORT
        __device__ __forceinline__ void loadShiftedCanonicalScalar(const dfloat* input, dfloat* values, int x, int y, int zLocal)
        {
            const size_t planeNodes = static_cast<size_t>(NX) * NY;
            for (int q = 0; q < GQ; ++q) {
                const int destinationX = (x + gcx[q] + NX) % NX;
                const int destinationY = (y + gcy[q] + NY) % NY;
                const size_t destinationZ = static_cast<size_t>(zLocal + 1 + gcz[q]);
                const size_t destinationNode = static_cast<size_t>(destinationX) + static_cast<size_t>(NX) * destinationY + planeNodes * destinationZ;
                values[q] = input[destinationNode * GQ + q];
                }
            }

    #endif // CONVECTION_DIFFUSION_TRANSPORT

} // namespace

__global__ void gpuPackCanonicalCheckpointPopulations(
    const dfloat* fMom,
    ghostInterfaceData allInterfaces,
    ghostInterfaceData allInterfacesCopy,
    CanonicalPopulationDeviceBuffers output,
    size_t localNZ,
    int zStart,
    size_t numBlocksZ,
    bool stepParity)
{
    // Each thread translates one physical node from the active streaming
    // representation into the decomposition-independent checkpoint layout.
    const int x = threadIdx.x + blockDim.x * blockIdx.x;
    const int y = threadIdx.y + blockDim.y * blockIdx.y;
    const int zLocal = threadIdx.z + blockDim.z * blockIdx.z;
    const int z_local = zLocal;
    const bool active = x < NX && y < NY && static_cast<size_t>(zLocal) < localNZ;

    __shared__ cuda::barrier<cuda::thread_scope_block> activeBarrier;

    const unsigned int activeCount = __syncthreads_count(active);

    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) init(&activeBarrier, activeCount);

    __syncthreads();

    if (!active) return;

    #ifdef DYNAMIC_SHARED_MEMORY
        extern __shared__ dfloat s_pop[];
    #else
        __shared__ dfloat s_pop[MAX_SHARED_MEMORY_SIZE / sizeof(dfloat)];
    #endif // DYNAMIC_SHARED_MEMORY

    const int z = zStart + zLocal;
    const size_t node = canonicalNodeIndex(x, y, zLocal);
    const int tx = threadIdx.x, ty = threadIdx.y, tz = threadIdx.z;
    const int bx = blockIdx.x, by = blockIdx.y, bz = blockIdx.z;

    const unsigned short xp1 = (threadIdx.x + 1 + BLOCK_NX) % BLOCK_NX;
    const unsigned short xm1 = (threadIdx.x - 1 + BLOCK_NX) % BLOCK_NX;
    const unsigned short yp1 = (threadIdx.y + 1 + BLOCK_NY) % BLOCK_NY;
    const unsigned short ym1 = (threadIdx.y - 1 + BLOCK_NY) % BLOCK_NY;
    const unsigned short zp1 = (threadIdx.z + 1 + BLOCK_NZ) % BLOCK_NZ;
    const unsigned short zm1 = (threadIdx.z - 1 + BLOCK_NZ) % BLOCK_NZ;

    const int txm1 = (x == 0) ? (NX - 1) % BLOCK_NX : (tx - 1 + BLOCK_NX) % BLOCK_NX;
    const int txp1 = (x + 1 >= NX) ? 0 : (tx + 1) % BLOCK_NX;
    const int tym1 = (y == 0) ? (NY - 1) % BLOCK_NY : (ty - 1 + BLOCK_NY) % BLOCK_NY;
    const int typ1 = (y + 1 >= NY) ? 0 : (ty + 1) % BLOCK_NY;
    const int tzm1 = (zLocal == 0) ? (localNZ - 1) % BLOCK_NZ : (tz - 1 + BLOCK_NZ) % BLOCK_NZ;
    const int tzp1 = (static_cast<size_t>(zLocal + 1) >= localNZ) ? 0 : (tz + 1) % BLOCK_NZ;
    const int bxm1 = (bx - 1 + NUM_BLOCK_X) % NUM_BLOCK_X;
    const int bxp1 = (bx + 1) % NUM_BLOCK_X;
    const int bym1 = (by - 1 + NUM_BLOCK_Y) % NUM_BLOCK_Y;
    const int byp1 = (by + 1) % NUM_BLOCK_Y;
    const int bzm1 = (bz - 1 + static_cast<int>(numBlocksZ)) % static_cast<int>(numBlocksZ);
    const int bzp1 = (bz + 1) % static_cast<int>(numBlocksZ);

    DeviceKernelParams params{};
    
    params.numBlocksZ = static_cast<unsigned int>(numBlocksZ);
    (void)z;

    #define LOAD_CHECKPOINT_MOMENT(moment) loadMoment(fMom, idxMom( \
        threadIdx.x, threadIdx.y, threadIdx.z, moment, blockIdx.x, \
        blockIdx.y, blockIdx.z), moment)
    dfloat rhoVar = RHO_0 + LOAD_CHECKPOINT_MOMENT(M_RHO_INDEX);
    dfloat ux_t30 = LOAD_CHECKPOINT_MOMENT(M_UX_INDEX);
    dfloat uy_t30 = LOAD_CHECKPOINT_MOMENT(M_UY_INDEX);
    dfloat uz_t30 = LOAD_CHECKPOINT_MOMENT(M_UZ_INDEX);
    dfloat m_xx_t45 = LOAD_CHECKPOINT_MOMENT(M_MXX_INDEX);
    dfloat m_xy_t90 = LOAD_CHECKPOINT_MOMENT(M_MXY_INDEX);
    dfloat m_xz_t90 = LOAD_CHECKPOINT_MOMENT(M_MXZ_INDEX);
    dfloat m_yy_t45 = LOAD_CHECKPOINT_MOMENT(M_MYY_INDEX);
    dfloat m_yz_t90 = LOAD_CHECKPOINT_MOMENT(M_MYZ_INDEX);
    dfloat m_zz_t45 = LOAD_CHECKPOINT_MOMENT(M_MZZ_INDEX);
    #undef LOAD_CHECKPOINT_MOMENT

    dfloat pop[Q];
    dfloat multiplyTerm;
    dfloat pics2;

    #ifdef PHI_DIST
        const dfloat phiVar_phi_init = fMom[idxMom(threadIdx.x, threadIdx.y,
        threadIdx.z, M3_PHI_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
        dfloat h_phi = (phiVar_phi_init - PHI_ONE) / (PHI_TWO - PHI_ONE);
        h_phi = fmaxf(0.0_df, fminf(1.0_df, h_phi));
        const dfloat rho_pf = PHI_RHO_PHASE1 + PHI_DRHO_PHASE12 * h_phi;
        const dfloat invRhoPF = 1.0_df / rho_pf;
    #endif // PHI_DIST

    #include COLREC_RECONSTRUCTION
    #include "fragments/checkpoint_hydro_shared_pull.inc"

    #if defined(STREAMING_AA_STD) || defined(STREAMING_AB)
        {
            const ghostFacePtrs ghostInterface{
                allInterfaces.pop.X_0, allInterfaces.pop.X_1,
                allInterfaces.pop.Y_0, allInterfaces.pop.Y_1,
                allInterfaces.pop.Z_0, allInterfaces.pop.Z_1,
                allInterfaces.popAux.Z_0, allInterfaces.popAux.Z_1};
            const ghostFacePtrs ghostInterfaceCopy{
                allInterfacesCopy.pop.X_0, allInterfacesCopy.pop.X_1,
                allInterfacesCopy.pop.Y_0, allInterfacesCopy.pop.Y_1,
                allInterfacesCopy.pop.Z_0, allInterfacesCopy.pop.Z_1,
                allInterfacesCopy.popAux.Z_0, allInterfacesCopy.popAux.Z_1};


        #ifdef COOPERATIVE_GHOST_LOAD_MASK
                #undef COOPERATIVE_GHOST_LOAD_MASK
        #endif // COOPERATIVE_GHOST_LOAD_MASK

            #include "fragments/popLoad_standard.inc"
        }
    #elif defined(STREAMING_AA_ESOTERIC_TWIST)
        {
            const ghostFacePtrs ghostInterface = checkpointFacePtrs(allInterfaces.pop, allInterfaces.popAux);
            const ghostFacePtrs ghostInterfaceCopy = checkpointFacePtrs(allInterfacesCopy.pop, allInterfacesCopy.popAux);
            #include "fragments/popLoad_esoteric_twist.inc"
        }
    #elif defined(STREAMING_AA_ESOTERIC_PUSH)
        {
            const ghostFacePtrs ghostInterface = checkpointFacePtrs(allInterfaces.pop, allInterfaces.popAux);
            const ghostFacePtrs ghostInterfaceCopy = checkpointFacePtrs(allInterfacesCopy.pop, allInterfacesCopy.popAux);
            #include "fragments/popLoad_esoteric_push.inc"
        }
    #elif defined(STREAMING_AA_ESOTERIC_PULL)
        {
            const ghostFacePtrs ghostInterface = checkpointFacePtrs(allInterfaces.pop, allInterfaces.popAux);
            const ghostFacePtrs ghostInterfaceCopy = checkpointFacePtrs(allInterfacesCopy.pop, allInterfacesCopy.popAux);
            #include "fragments/popLoad_esoteric_pull.inc"
        }
    #endif // defined(STREAMING_AA_STD) || defined(STREAMING_AB)

    for (int q = 0; q < Q; ++q) output.hydro[node * Q + q] = pop[q];

    #ifdef CONVECTION_DIFFUSION_TRANSPORT
        dfloat gNode[GQ];
    #ifdef SECOND_DIST
            dfloat cVar = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M2_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
            dfloat invC = 1 / cVar;
            dfloat qx_t30 = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M2_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
            dfloat qy_t30 = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M2_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
            dfloat qz_t30 = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M2_CZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
            #include COLREC_G_RECONSTRUCTION
            #include "fragments/checkpoint_scalar_shared_pull.inc"
            {
                const ghostFacePtrs ghostInterfaceG = checkpointFacePtrs(allInterfaces.g, allInterfaces.gAux);
                const ghostFacePtrs ghostInterfaceGCopy = checkpointFacePtrs(allInterfacesCopy.g, allInterfacesCopy.gAux);
                #include "fragments/gTransport/g_popLoad.inc"
            }

            for (int q = 0; q < GQ; ++q) output.g[node * GQ + q] = gNode[q];

    #endif // SECOND_DIST
    #ifdef PHI_DIST
            dfloat phiVar = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M3_PHI_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
            dfloat invPhi = 1 / phiVar;
            dfloat phi_qx_t30 = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M3_PX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
            dfloat phi_qy_t30 = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M3_PY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
            dfloat phi_qz_t30 = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M3_PZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];

            #include COLREC_PHI_RECONSTRUCTION
            #include "fragments/checkpoint_scalar_shared_pull.inc"
            {
                const ghostFacePtrs ghostInterfacePhi = checkpointFacePtrs(allInterfaces.phi, allInterfaces.phiAux);
                const ghostFacePtrs ghostInterfacePhiCopy = checkpointFacePtrs(allInterfacesCopy.phi, allInterfacesCopy.phiAux);
                #include "fragments/phiTransport/phi_popLoad.inc"
            }

            for (int q = 0; q < GQ; ++q) output.phi[node * GQ + q] = gNode[q];

    #endif // PHI_DIST
    #ifdef LAMBDA_DIST
            dfloat lambdaVar = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z,M4_LAMBDA_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
            dfloat invLambda = 1 / lambdaVar;
            dfloat lambda_qx_t30 = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M4_LX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
            dfloat lambda_qy_t30 = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M4_LY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
            dfloat lambda_qz_t30 = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M4_LZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
            dfloat lambda_udx_t30 = LAMBDA_DIFF_FLUC_COEF * (lambda_qx_t30 * invLambda - ux_t30);
            dfloat lambda_udy_t30 = LAMBDA_DIFF_FLUC_COEF * (lambda_qy_t30 * invLambda - uy_t30);
            dfloat lambda_udz_t30 = LAMBDA_DIFF_FLUC_COEF * (lambda_qz_t30 * invLambda - uz_t30);
            #include COLREC_LAMBDA_RECONSTRUCTION
            #include "fragments/checkpoint_scalar_shared_pull.inc"
            {
                const ghostFacePtrs ghostInterfaceLambda = checkpointFacePtrs(allInterfaces.lambda, allInterfaces.lambdaAux);
                const ghostFacePtrs ghostInterfaceLambdaCopy = checkpointFacePtrs(allInterfacesCopy.lambda, allInterfacesCopy.lambdaAux);
                #include "fragments/lambdaTransport/lambda_popLoad.inc"
            }

            for (int q = 0; q < GQ; ++q) output.lambda[node * GQ + q] = gNode[q];

    #endif // LAMBDA_DIST

    #ifdef A_XX_DIST
            dfloat AxxVar = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_XX_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
            dfloat invAxx = 1 / AxxVar;
            dfloat Axx_qx_t30 = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_XX_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
            dfloat Axx_qy_t30 = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_XX_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
            dfloat Axx_qz_t30 = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_XX_CZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
            #include COLREC_AXX_RECONSTRUCTION
            #include "fragments/checkpoint_scalar_shared_pull.inc"
            {
                const ghostFacePtrs ghostInterfaceAxx = checkpointFacePtrs(allInterfaces.Axx, allInterfaces.AxxAux);
                const ghostFacePtrs ghostInterfaceAxxCopy = checkpointFacePtrs(allInterfacesCopy.Axx, allInterfacesCopy.AxxAux);
                #include "fragments/conformationTransport/popLoad_Axx.inc"
            }

            for (int q = 0; q < GQ; ++q) output.Axx[node * GQ + q] = gNode[q];

    #endif // A_XX_DIST
    #ifdef A_XY_DIST
            dfloat AxyVar = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_XY_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
            dfloat invAxy = 1 / AxyVar;
            dfloat Axy_qx_t30 = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_XY_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
            dfloat Axy_qy_t30 = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_XY_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
            dfloat Axy_qz_t30 = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_XY_CZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
            #include COLREC_AXY_RECONSTRUCTION
            #include "fragments/checkpoint_scalar_shared_pull.inc"
            {
                const ghostFacePtrs ghostInterfaceAxy = checkpointFacePtrs(allInterfaces.Axy, allInterfaces.AxyAux);
                const ghostFacePtrs ghostInterfaceAxyCopy = checkpointFacePtrs(allInterfacesCopy.Axy, allInterfacesCopy.AxyAux);
                #include "fragments/conformationTransport/popLoad_Axy.inc"
            }

            for (int q = 0; q < GQ; ++q) output.Axy[node * GQ + q] = gNode[q];

    #endif // A_XY_DIST
    #ifdef A_XZ_DIST
            dfloat AxzVar = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_XZ_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
            dfloat invAxz = 1 / AxzVar;
            dfloat Axz_qx_t30 = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_XZ_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
            dfloat Axz_qy_t30 = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_XZ_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
            dfloat Axz_qz_t30 = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_XZ_CZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
            #include COLREC_AXZ_RECONSTRUCTION
            #include "fragments/checkpoint_scalar_shared_pull.inc"
            {
                const ghostFacePtrs ghostInterfaceAxz = checkpointFacePtrs(allInterfaces.Axz, allInterfaces.AxzAux);
                const ghostFacePtrs ghostInterfaceAxzCopy = checkpointFacePtrs(allInterfacesCopy.Axz, allInterfacesCopy.AxzAux);
                #include "fragments/conformationTransport/popLoad_Axz.inc"
            }

            for (int q = 0; q < GQ; ++q) output.Axz[node * GQ + q] = gNode[q];

    #endif // A_XZ_DIST
    #ifdef A_YY_DIST
            dfloat AyyVar = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_YY_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
            dfloat invAyy = 1 / AyyVar;
            dfloat Ayy_qx_t30 = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_YY_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
            dfloat Ayy_qy_t30 = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_YY_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
            dfloat Ayy_qz_t30 = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_YY_CZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
            #include COLREC_AYY_RECONSTRUCTION
            #include "fragments/checkpoint_scalar_shared_pull.inc"
            {
                const ghostFacePtrs ghostInterfaceAyy = checkpointFacePtrs(allInterfaces.Ayy, allInterfaces.AyyAux);
                const ghostFacePtrs ghostInterfaceAyyCopy = checkpointFacePtrs(allInterfacesCopy.Ayy, allInterfacesCopy.AyyAux);
                #include "fragments/conformationTransport/popLoad_Ayy.inc"
            }

            for (int q = 0; q < GQ; ++q) output.Ayy[node * GQ + q] = gNode[q];

    #endif // A_YY_DIST
    #ifdef A_YZ_DIST
            dfloat AyzVar = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_YZ_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
            dfloat invAyz = 1 / AyzVar;
            dfloat Ayz_qx_t30 = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_YZ_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
            dfloat Ayz_qy_t30 = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_YZ_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
            dfloat Ayz_qz_t30 = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_YZ_CZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
            #include COLREC_AYZ_RECONSTRUCTION
            #include "fragments/checkpoint_scalar_shared_pull.inc"
            {
                const ghostFacePtrs ghostInterfaceAyz = checkpointFacePtrs(allInterfaces.Ayz, allInterfaces.AyzAux);
                const ghostFacePtrs ghostInterfaceAyzCopy = checkpointFacePtrs(allInterfacesCopy.Ayz, allInterfacesCopy.AyzAux);
                #include "fragments/conformationTransport/popLoad_Ayz.inc"
            }

            for (int q = 0; q < GQ; ++q) output.Ayz[node * GQ + q] = gNode[q];

    #endif // A_YZ_DIST
    #ifdef A_ZZ_DIST
            dfloat AzzVar = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_ZZ_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
            dfloat invAzz = 1 / AzzVar;
            dfloat Azz_qx_t30 = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_ZZ_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
            dfloat Azz_qy_t30 = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_ZZ_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
            dfloat Azz_qz_t30 = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_ZZ_CZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
            #include COLREC_AZZ_RECONSTRUCTION
            #include "fragments/checkpoint_scalar_shared_pull.inc"
            {
                const ghostFacePtrs ghostInterfaceAzz = checkpointFacePtrs(allInterfaces.Azz, allInterfaces.AzzAux);
                const ghostFacePtrs ghostInterfaceAzzCopy = checkpointFacePtrs(allInterfacesCopy.Azz, allInterfacesCopy.AzzAux);
                #include "fragments/conformationTransport/popLoad_Azz.inc"
            }

            for (int q = 0; q < GQ; ++q) output.Azz[node * GQ + q] = gNode[q];

    #endif // A_ZZ_DIST
    #endif // CONVECTION_DIFFUSION_TRANSPORT
}

__global__ void gpuEncodeCanonicalCheckpointPopulations(
    CanonicalPopulationDeviceBuffers input,
    ghostInterfaceData allInterfaces,
    size_t localNZ,
    size_t numBlocksZ,
    int zStart,
    bool completedStepParity)
{
    // Each thread converts one checkpoint node back into the active streaming
    // representation and repopulates the runtime face interfaces.
    const int x = threadIdx.x + blockDim.x * blockIdx.x;
    const int y = threadIdx.y + blockDim.y * blockIdx.y;
    const int zLocal = threadIdx.z + blockDim.z * blockIdx.z;

    if (x >= NX || y >= NY || static_cast<size_t>(zLocal) >= localNZ) return;

    const int z = zStart + zLocal;
    const int z_local = zLocal;
    const size_t node = canonicalNodeIndex(x, y, zLocal);
    const size_t extendedPlaneNodes = static_cast<size_t>(NX) * NY;

    const int tx = threadIdx.x, ty = threadIdx.y, tz = threadIdx.z;
    const int bx = blockIdx.x, by = blockIdx.y, bz = blockIdx.z;

    const int txm1 = (x == 0) ? (NX - 1) % BLOCK_NX : (tx - 1 + BLOCK_NX) % BLOCK_NX;
    const int txp1 = (x + 1 >= NX) ? 0 : (tx + 1) % BLOCK_NX;
    const int tym1 = (y == 0) ? (NY - 1) % BLOCK_NY : (ty - 1 + BLOCK_NY) % BLOCK_NY;
    const int typ1 = (y + 1 >= NY) ? 0 : (ty + 1) % BLOCK_NY;
    const int tzm1 = (zLocal == 0) ? (localNZ - 1) % BLOCK_NZ : (tz - 1 + BLOCK_NZ) % BLOCK_NZ;
    const int tzp1 = (static_cast<size_t>(zLocal + 1) >= localNZ) ? 0 : (tz + 1) % BLOCK_NZ;
    const int bxm1 = (bx - 1 + NUM_BLOCK_X) % NUM_BLOCK_X;
    const int bxp1 = (bx + 1) % NUM_BLOCK_X;
    const int bym1 = (by - 1 + NUM_BLOCK_Y) % NUM_BLOCK_Y;
    const int byp1 = (by + 1) % NUM_BLOCK_Y;
    const int bzm1 = (bz - 1 + static_cast<int>(numBlocksZ)) % static_cast<int>(numBlocksZ);
    const int bzp1 = (bz + 1) % static_cast<int>(numBlocksZ);

    const int bxAtWest = tx == 0 ? bxm1 : bx;
    const int bxAtEast = (tx == BLOCK_NX - 1 || x == NX - 1) ? bxp1 : bx;
    const int byAtSouth = ty == 0 ? bym1 : by;
    const int byAtNorth = (ty == BLOCK_NY - 1 || y == NY - 1) ? byp1 : by;
    const int bzAtBack = tz == 0 ? bzm1 : bz;
    const int bzAtFront = (tz == BLOCK_NZ - 1 || static_cast<size_t>(zLocal + 1) >= localNZ) ? bzp1 : bz;
    const int top = static_cast<int>(numBlocksZ) - 1;

    DeviceKernelParams params{};
    params.numBlocksZ = static_cast<unsigned int>(numBlocksZ);
    const bool cooperativePhysicalBlock = false;
    const bool stepParity = completedStepParity;
    dfloat pop[Q];

    for (int q = 0; q < Q; ++q) {
        const int destinationX = (x + cx[q] + NX) % NX;
        const int destinationY = (y + cy[q] + NY) % NY;
        const size_t destinationZ = static_cast<size_t>(zLocal + 1 + cz[q]);
        const size_t destinationNode = static_cast<size_t>(destinationX) + static_cast<size_t>(NX) * destinationY + extendedPlaneNodes * destinationZ;
        pop[q] = input.hydro[destinationNode * Q + q];
    }

    #if defined(STREAMING_AA_ESOTERIC_TWIST)
        {
            const ghostFacePtrs ghostInterface = checkpointFacePtrs(allInterfaces.pop, allInterfaces.popAux);
            const int twistBxAtEast = (tx == BLOCK_NX - 1 || x == NX - 1) ? bxp1 : bx;
            const int twistByAtNorth = (ty == BLOCK_NY - 1 || y == NY - 1) ? byp1 : by;
            const int twistBzAtFront = (tz == BLOCK_NZ - 1 || static_cast<size_t>(zLocal + 1) >= localNZ) ? bzp1 : bz;
            #include "fragments/popSave_esoteric_twist.inc"
        }
    #elif defined(STREAMING_AA_ESOTERIC_PUSH)
        {
            const ghostFacePtrs ghostInterface = checkpointFacePtrs(allInterfaces.pop, allInterfaces.popAux);
            const int pushBxAtWest = tx == 0 ? bxm1 : bx;
            const int pushByAtSouth = ty == 0 ? bym1 : by;
            const int pushByAtNorth = (ty == BLOCK_NY - 1 || y == NY - 1) ? byp1 : by;
            const int pushBzAtBack = tz == 0 ? bzm1 : bz;
            const int pushBzAtFront = (tz == BLOCK_NZ - 1 || static_cast<size_t>(zLocal + 1) >= localNZ) ? bzp1 : bz;
            #include "fragments/popSave_esoteric_push.inc"
        }
    #elif defined(STREAMING_AA_ESOTERIC_PULL)
        {
            const ghostFacePtrs ghostInterface = checkpointFacePtrs(allInterfaces.pop, allInterfaces.popAux);
            #include "fragments/popSave_esoteric_pull.inc"
        }
    #else
        const ghostFacePtrs ghostInterface{
            allInterfaces.pop.X_0, allInterfaces.pop.X_1,
            allInterfaces.pop.Y_0, allInterfaces.pop.Y_1,
            allInterfaces.pop.Z_0, allInterfaces.pop.Z_1, nullptr, nullptr};
        const ghostFacePtrs storeGhostInterface = ghostInterface;

    // Cooperative population saves require shared-memory staging.  The
    // checkpoint encoder already owns canonical per-node values, so use the
    // production non-cooperative face mapping even in cooperative builds.
    #ifdef COOPERATIVE_GHOST_SAVE_MASK
            #undef COOPERATIVE_GHOST_SAVE_MASK
    #endif // COOPERATIVE_GHOST_SAVE_MASK
        #include "fragments/popSave_standard.inc"
    #endif // defined(STREAMING_AA_ESOTERIC_TWIST)

    #ifdef CONVECTION_DIFFUSION_TRANSPORT
        dfloat gNode[GQ];
    #ifdef SECOND_DIST
            loadShiftedCanonicalScalar(input.g, gNode, x, y, zLocal);
            {
                const ghostFacePtrs ghostInterfaceG = checkpointFacePtrs(allInterfaces.g, allInterfaces.gAux);
                #include "fragments/gTransport/g_popSave.inc"
            }
    #endif // SECOND_DIST
    #ifdef PHI_DIST
            loadShiftedCanonicalScalar(input.phi, gNode, x, y, zLocal);
            {
                const ghostFacePtrs ghostInterfacePhi = checkpointFacePtrs(allInterfaces.phi, allInterfaces.phiAux);
                #include "fragments/phiTransport/phi_popSave.inc"
            }
    #endif // PHI_DIST
    #ifdef LAMBDA_DIST
            loadShiftedCanonicalScalar(input.lambda, gNode, x, y, zLocal);
            {
                const ghostFacePtrs ghostInterfaceLambda = checkpointFacePtrs(allInterfaces.lambda, allInterfaces.lambdaAux);
                #include "fragments/lambdaTransport/lambda_popSave.inc"
            }
    #endif // LAMBDA_DIST
    #ifdef A_XX_DIST
            loadShiftedCanonicalScalar(input.Axx, gNode, x, y, zLocal);
            {
                const ghostFacePtrs ghostInterfaceAxx = checkpointFacePtrs(allInterfaces.Axx, allInterfaces.AxxAux);
                #include "fragments/conformationTransport/popSave_Axx.inc"
            }
    #endif // A_XX_DIST
    #ifdef A_XY_DIST
            loadShiftedCanonicalScalar(input.Axy, gNode, x, y, zLocal);
            {
                const ghostFacePtrs ghostInterfaceAxy = checkpointFacePtrs(allInterfaces.Axy, allInterfaces.AxyAux);
                #include "fragments/conformationTransport/popSave_Axy.inc"
            }
    #endif // A_XY_DIST
    #ifdef A_XZ_DIST
            loadShiftedCanonicalScalar(input.Axz, gNode, x, y, zLocal);
            {
                const ghostFacePtrs ghostInterfaceAxz = checkpointFacePtrs(allInterfaces.Axz, allInterfaces.AxzAux);
                #include "fragments/conformationTransport/popSave_Axz.inc"
            }
    #endif // A_XZ_DIST
    #ifdef A_YY_DIST
            loadShiftedCanonicalScalar(input.Ayy, gNode, x, y, zLocal);
            {
                const ghostFacePtrs ghostInterfaceAyy = checkpointFacePtrs(allInterfaces.Ayy, allInterfaces.AyyAux);
                #include "fragments/conformationTransport/popSave_Ayy.inc"
            }
    #endif // A_YY_DIST
    #ifdef A_YZ_DIST
            loadShiftedCanonicalScalar(input.Ayz, gNode, x, y, zLocal);
            {
                const ghostFacePtrs ghostInterfaceAyz = checkpointFacePtrs(allInterfaces.Ayz, allInterfaces.AyzAux);
                #include "fragments/conformationTransport/popSave_Ayz.inc"
            }
    #endif // A_YZ_DIST
    #ifdef A_ZZ_DIST
            loadShiftedCanonicalScalar(input.Azz, gNode, x, y, zLocal);
            {
                const ghostFacePtrs ghostInterfaceAzz = checkpointFacePtrs(allInterfaces.Azz, allInterfaces.AzzAux);
                #include "fragments/conformationTransport/popSave_Azz.inc"
            }
    #endif // A_ZZ_DIST
    #endif // CONVECTION_DIFFUSION_TRANSPORT
}

namespace {

void allocateCanonicalDeviceBuffers(
    CanonicalPopulationDeviceBuffers& buffers, size_t nodes)
{
    checkCudaErrors(cudaMalloc(&buffers.hydro, nodes * Q * sizeof(dfloat)));
        #ifdef SECOND_DIST
            checkCudaErrors(cudaMalloc(&buffers.g, nodes * GQ * sizeof(dfloat)));
        #endif // SECOND_DIST
        #ifdef PHI_DIST
            checkCudaErrors(cudaMalloc(&buffers.phi, nodes * GQ * sizeof(dfloat)));
        #endif // PHI_DIST
        #ifdef LAMBDA_DIST
            checkCudaErrors(cudaMalloc(&buffers.lambda, nodes * GQ * sizeof(dfloat)));
        #endif // LAMBDA_DIST
        #ifdef A_XX_DIST
            checkCudaErrors(cudaMalloc(&buffers.Axx, nodes * GQ * sizeof(dfloat)));
        #endif // A_XX_DIST
        #ifdef A_XY_DIST
            checkCudaErrors(cudaMalloc(&buffers.Axy, nodes * GQ * sizeof(dfloat)));
        #endif // A_XY_DIST
        #ifdef A_XZ_DIST
            checkCudaErrors(cudaMalloc(&buffers.Axz, nodes * GQ * sizeof(dfloat)));
        #endif // A_XZ_DIST
        #ifdef A_YY_DIST
            checkCudaErrors(cudaMalloc(&buffers.Ayy, nodes * GQ * sizeof(dfloat)));
        #endif // A_YY_DIST
        #ifdef A_YZ_DIST
            checkCudaErrors(cudaMalloc(&buffers.Ayz, nodes * GQ * sizeof(dfloat)));
        #endif // A_YZ_DIST
        #ifdef A_ZZ_DIST
            checkCudaErrors(cudaMalloc(&buffers.Azz, nodes * GQ * sizeof(dfloat)));
        #endif // A_ZZ_DIST
}

void freeCanonicalDeviceBuffers(CanonicalPopulationDeviceBuffers& buffers)
{
    checkCudaErrors(cudaFree(buffers.hydro));
        #ifdef SECOND_DIST
            checkCudaErrors(cudaFree(buffers.g));
        #endif // SECOND_DIST
        #ifdef PHI_DIST
            checkCudaErrors(cudaFree(buffers.phi));
        #endif // PHI_DIST
        #ifdef LAMBDA_DIST
            checkCudaErrors(cudaFree(buffers.lambda));
        #endif // LAMBDA_DIST
        #ifdef A_XX_DIST
            checkCudaErrors(cudaFree(buffers.Axx));
        #endif // A_XX_DIST
        #ifdef A_XY_DIST
            checkCudaErrors(cudaFree(buffers.Axy));
        #endif // A_XY_DIST
        #ifdef A_XZ_DIST
            checkCudaErrors(cudaFree(buffers.Axz));
        #endif // A_XZ_DIST
        #ifdef A_YY_DIST
            checkCudaErrors(cudaFree(buffers.Ayy));
        #endif // A_YY_DIST
        #ifdef A_YZ_DIST
            checkCudaErrors(cudaFree(buffers.Ayz));
        #endif // A_YZ_DIST
        #ifdef A_ZZ_DIST
            checkCudaErrors(cudaFree(buffers.Azz));
        #endif // A_ZZ_DIST
}

template<class Vector>
void copyDeviceSlabToGlobal(Vector& destination, const dfloat* source, size_t nodeOffset, size_t localNodes, size_t populations)
{
    checkCudaErrors(cudaMemcpy(destination.data() + nodeOffset * populations, source, localNodes * populations * sizeof(dfloat), cudaMemcpyDeviceToHost));
}

template<class Vector>
void copyGlobalSlabWithPeriodicZHaloToDevice(
    dfloat* destination, const Vector& source, const GpuDomainLayout& layout,
    size_t populations)
{
    const size_t planeNodes = static_cast<size_t>(NX) * NY;
    const size_t localNodes = planeNodes * layout.localNZ;
    const size_t localOffset = planeNodes * layout.zStart;
    const size_t bottomZ = (layout.zStart + NZ - 1u) % NZ;
    const size_t topZ = (layout.zStart + layout.localNZ) % NZ;
    checkCudaErrors(cudaMemcpy(destination, source.data() + bottomZ * planeNodes * populations, planeNodes * populations * sizeof(dfloat), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(destination + planeNodes * populations, source.data() + localOffset * populations, localNodes * populations * sizeof(dfloat), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy( destination + (layout.localNZ + 1u) * planeNodes * populations, source.data() + topZ * planeNodes * populations, planeNodes * populations * sizeof(dfloat), cudaMemcpyHostToDevice));
}

} // namespace

void CanonicalCheckpointHostState::resizeForGlobalDomain()
{
    // Allocate only the populations enabled by this build for the full domain.
    const size_t nodes = static_cast<size_t>(NX) * NY * NZ;
    hydro.resize(nodes * Q);
    #ifdef SECOND_DIST
        g.resize(nodes * GQ);
    #endif // SECOND_DIST
    #ifdef PHI_DIST
        phi.resize(nodes * GQ);
    #endif // PHI_DIST
    #ifdef LAMBDA_DIST
        lambda.resize(nodes * GQ);
    #endif // LAMBDA_DIST
    #ifdef A_XX_DIST
        Axx.resize(nodes * GQ);
    #endif // A_XX_DIST
    #ifdef A_XY_DIST
        Axy.resize(nodes * GQ);
    #endif // A_XY_DIST
    #ifdef A_XZ_DIST
        Axz.resize(nodes * GQ);
    #endif // A_XZ_DIST
    #ifdef A_YY_DIST
        Ayy.resize(nodes * GQ);
    #endif // A_YY_DIST
    #ifdef A_YZ_DIST
        Ayz.resize(nodes * GQ);
    #endif // A_YZ_DIST
    #ifdef A_ZZ_DIST
        Azz.resize(nodes * GQ);
    #endif // A_ZZ_DIST
}

__host__ void captureCanonicalCheckpointPopulations(
    const dfloat* fMom, ghostInterfaceData ghostInterface,
    ghostInterfaceData ghostInterfaceCopy, const GpuDomainLayout& layout,
    dim3 threadBlock, bool stepParity, CanonicalCheckpointHostState& output)
{
    // Decode this slab on the GPU, then place it at its global Z offset.
    const size_t localNodes = static_cast<size_t>(NX) * NY * layout.localNZ;
    const size_t nodeOffset = static_cast<size_t>(NX) * NY * layout.zStart;
    CanonicalPopulationDeviceBuffers device{};
    allocateCanonicalDeviceBuffers(device, localNodes);
    #ifdef DYNAMIC_SHARED_MEMORY
        checkCudaErrors(cudaFuncSetAttribute(gpuPackCanonicalCheckpointPopulations, cudaFuncAttributeMaxDynamicSharedMemorySize, MAX_SHARED_MEMORY_SIZE));
        gpuPackCanonicalCheckpointPopulations<<<layout.fullGrid, threadBlock,
        MAX_SHARED_MEMORY_SIZE>>>(
    #else
        gpuPackCanonicalCheckpointPopulations<<<layout.fullGrid, threadBlock>>>(
    #endif // DYNAMIC_SHARED_MEMORY
        fMom, ghostInterface, ghostInterfaceCopy, device, layout.localNZ,
        static_cast<int>(layout.zStart), layout.numBlocksZ, stepParity);


    CHECK_KERNEL_ERR("canonical checkpoint population pack");

    copyDeviceSlabToGlobal(output.hydro, device.hydro, nodeOffset, localNodes, Q);

    #ifdef SECOND_DIST
        copyDeviceSlabToGlobal(output.g, device.g, nodeOffset, localNodes, GQ);
    #endif // SECOND_DIST
    #ifdef PHI_DIST
        copyDeviceSlabToGlobal(output.phi, device.phi, nodeOffset, localNodes, GQ);
    #endif // PHI_DIST
    #ifdef LAMBDA_DIST
        copyDeviceSlabToGlobal(output.lambda, device.lambda, nodeOffset, localNodes, GQ);
    #endif // LAMBDA_DIST
    #ifdef A_XX_DIST
        copyDeviceSlabToGlobal(output.Axx, device.Axx, nodeOffset, localNodes, GQ);
    #endif // A_XX_DIST
    #ifdef A_XY_DIST
        copyDeviceSlabToGlobal(output.Axy, device.Axy, nodeOffset, localNodes, GQ);
    #endif // A_XY_DIST
    #ifdef A_XZ_DIST
        copyDeviceSlabToGlobal(output.Axz, device.Axz, nodeOffset, localNodes, GQ);
    #endif // A_XZ_DIST
    #ifdef A_YY_DIST
        copyDeviceSlabToGlobal(output.Ayy, device.Ayy, nodeOffset, localNodes, GQ);
    #endif // A_YY_DIST
    #ifdef A_YZ_DIST
        copyDeviceSlabToGlobal(output.Ayz, device.Ayz, nodeOffset, localNodes, GQ);
    #endif // A_YZ_DIST
    #ifdef A_ZZ_DIST
        copyDeviceSlabToGlobal(output.Azz, device.Azz, nodeOffset, localNodes, GQ);
    #endif // A_ZZ_DIST
    freeCanonicalDeviceBuffers(device);
}

__host__ void restoreCanonicalCheckpointPopulations(
    const CanonicalCheckpointHostState& input, ghostInterfaceData ghostInterface,
    const GpuDomainLayout& layout, dim3 threadBlock, bool completedStepParity)
{
    // Upload this slab plus periodic Z halos before encoding the runtime form.
    const size_t extendedNodes = static_cast<size_t>(NX) * NY * (layout.localNZ + 2u);
    CanonicalPopulationDeviceBuffers device{};
    allocateCanonicalDeviceBuffers(device, extendedNodes);
    copyGlobalSlabWithPeriodicZHaloToDevice( device.hydro, input.hydro, layout, Q);

    #ifdef SECOND_DIST
        copyGlobalSlabWithPeriodicZHaloToDevice(device.g, input.g, layout, GQ);
    #endif // SECOND_DIST
    #ifdef PHI_DIST
        copyGlobalSlabWithPeriodicZHaloToDevice(device.phi, input.phi, layout, GQ);
    #endif // PHI_DIST
    #ifdef LAMBDA_DIST
        copyGlobalSlabWithPeriodicZHaloToDevice(device.lambda, input.lambda, layout, GQ);
    #endif // LAMBDA_DIST
    #ifdef A_XX_DIST
        copyGlobalSlabWithPeriodicZHaloToDevice(device.Axx, input.Axx, layout, GQ);
    #endif // A_XX_DIST
    #ifdef A_XY_DIST
        copyGlobalSlabWithPeriodicZHaloToDevice(device.Axy, input.Axy, layout, GQ);
    #endif // A_XY_DIST
    #ifdef A_XZ_DIST
        copyGlobalSlabWithPeriodicZHaloToDevice(device.Axz, input.Axz, layout, GQ);
    #endif // A_XZ_DIST
    #ifdef A_YY_DIST
        copyGlobalSlabWithPeriodicZHaloToDevice(device.Ayy, input.Ayy, layout, GQ);
    #endif // A_YY_DIST
    #ifdef A_YZ_DIST
        copyGlobalSlabWithPeriodicZHaloToDevice(device.Ayz, input.Ayz, layout, GQ);
    #endif // A_YZ_DIST
    #ifdef A_ZZ_DIST
        copyGlobalSlabWithPeriodicZHaloToDevice(device.Azz, input.Azz, layout, GQ);
    #endif // A_ZZ_DIST
    gpuEncodeCanonicalCheckpointPopulations<<<layout.fullGrid, threadBlock>>>( device, ghostInterface, layout.localNZ, layout.numBlocksZ, static_cast<int>(layout.zStart), completedStepParity);
    
    CHECK_KERNEL_ERR("canonical checkpoint population encode");
    checkCudaErrors(cudaDeviceSynchronize());
    freeCanonicalDeviceBuffers(device);
}
