#ifndef __DEVICEFIELD_STRUCTS_H
#define __DEVICEFIELD_STRUCTS_H

#include "var.h"
#include "hostField.cuh"
#include "ghostInterface.cuh"
#include "fragments/phiTransport/phase_kernels.cuh"
#include <algorithm>
#include <thread>
#include <vector>

/** @brief Owns per-GPU memory, halo interfaces, and device-side operations. */
typedef struct deviceField{
    GpuDomainLayout layout;
    unsigned int streamingParityOffset = 0;
    ghostInterfaceData ghostInterface[N_GPUS];
    ghostInterfaceData ghostInterfaceCopy[N_GPUS];
    macroInterfaceGPUData macroInterfaceGPU[N_GPUS];
    macroInterfaceGPUData macroInterfaceGPUCopy[N_GPUS];

    dfloat* d_fMom[N_GPUS];
    unsigned int* dNodeType[N_GPUS];

    #ifdef CURVED_BOUNDARY_CONDITION
        CurvedBoundary** d_curvedBC[N_GPUS];
        CurvedBoundary* d_curvedBC_array[N_GPUS];
        unsigned int numberCurvedBoundaryNodes;  // Number of curved boundary nodes
    #endif // CURVED_BOUNDARY_CONDITION

    #ifdef DENSITY_CORRECTION
        dfloat* d_mean_rho[N_GPUS];
        dfloat* d_density_sum[N_GPUS][2];
        unsigned long long* d_density_count[N_GPUS][2];
    #endif //DENSITY_CORRECTION

    #ifdef BC_FORCES
        dfloat* d_BC_Fx[N_GPUS];
        dfloat* d_BC_Fy[N_GPUS];
        dfloat* d_BC_Fz[N_GPUS];
    #endif //_BC_FORCES

    #ifdef CONFORMATION_PHYSICAL_A_HALO
        PolymerStressHaloPtrs polymerStressHalo[N_GPUS][2]{};
    #endif // CONFORMATION_PHYSICAL_A_HALO

    #ifdef SAVE_LOCAL_FORCES
        dfloat* d_Local_Fx;
        dfloat* d_Local_Fy;
        dfloat* d_Local_Fz;
    #ifdef SECOND_DIST
            dfloat* d_Source_C;
    #endif // SECOND_DIST
    #ifdef PHI_DIST
            dfloat* d_Source_Phi;
    #endif // PHI_DIST
    #ifdef LAMBDA_DIST
            dfloat* d_Source_Lambda;
    #endif // LAMBDA_DIST
    #ifdef CONFORMATION_TENSOR
    #ifdef A_XX_DIST
                dfloat* d_Source_Gxx;
    #endif // A_XX_DIST
    #ifdef A_XY_DIST
                dfloat* d_Source_Gxy;
    #endif // A_XY_DIST
    #ifdef A_XZ_DIST
                dfloat* d_Source_Gxz;
    #endif // A_XZ_DIST
    #ifdef A_YY_DIST
                dfloat* d_Source_Gyy;
    #endif // A_YY_DIST
    #ifdef A_YZ_DIST
                dfloat* d_Source_Gyz;
    #endif // A_YZ_DIST
    #ifdef A_ZZ_DIST
                dfloat* d_Source_Gzz;
    #endif // A_ZZ_DIST
    #endif //CONFORMATION_TENSOR
    #endif //SAVE_LOCAL_FORCES

    #if defined(NON_NEWTONIAN_FLUID) || defined(CONFORMATION_TENSOR)
        fluidPhaseProps phasePropsA;             ///< Phase-1 fluid properties (viscous + viscoelastic)
    #endif // defined(NON_NEWTONIAN_FLUID) || defined(CONFORMATION_TENSOR)

    #ifdef PHI_DIST
        fluidPhaseProps phasePropsB;             ///< Phase-2 fluid properties (viscous + viscoelastic)
    #endif // PHI_DIST

    /** @brief Enables peer access between every configured CUDA device. */
    void enablePeerAccessDeviceField(){
        // Check if all GPUs are the same physical device (virtual multi-GPU emulation)
        bool allSameGpu = true;
        int firstGpu = GPUS_TO_USE[0];
        for (int i = 1; i < N_GPUS; i++) {
            if (GPUS_TO_USE[i] != firstGpu) { allSameGpu = false; break; }
        }

        if (allSameGpu) {
            // Single physical GPU: no P2P needed. cudaMemcpyPeerAsync works as regular memcpy on same device.
            printf("Virtual multi-GPU mode: all %d partitions on GPU %d (no P2P required)\n", N_GPUS, firstGpu);
            return;
        }

        for (int i = 0; i < N_GPUS; i++) {
            cudaSetDevice(GPUS_TO_USE[i]);
            for (int j = 0; j < N_GPUS; j++) {
                if (i == j) continue;
                int canAccessPeer = 0;
                checkCudaErrors(cudaDeviceCanAccessPeer(&canAccessPeer, GPUS_TO_USE[i], GPUS_TO_USE[j]));
                if (canAccessPeer) {
                    checkCudaErrors(cudaDeviceEnablePeerAccess(GPUS_TO_USE[j], 0));
                    printf("P2P access enabled: GPU %d -> GPU %d\n", GPUS_TO_USE[i], GPUS_TO_USE[j]);
                } else {
                    printf("ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â  GPU %d cannot access GPU %d via P2P\n", GPUS_TO_USE[i], GPUS_TO_USE[j]);
                }
            }
        }
    }

    /** @brief Allocates all lattice, halo, and auxiliary storage for GPU @p g. */
    void allocateDeviceMemoryDeviceField(int g, const GpuDomainLayout& gpuLayout) {
        layout = gpuLayout;
        cudaSetDevice(GPUS_TO_USE[g]);
        unsigned int memAllocated = 0;

        const size_t NUMBER_LBM_NODES_LOCAL = layout.paddedNodeCount;
        const size_t MEM_SIZE_MOM_STORAGE_LOCAL = layout.momentStorageBytes;
        const size_t MEM_SIZE_SCALAR = layout.scalarBytes;
        #ifdef CONFORMATION_PHYSICAL_A_HALO
            const size_t MEM_SIZE_POLYMER_STRESS_HALO_X = sizeof(dfloat) * 12 * layout.ghostFaceYZCount;
            const size_t MEM_SIZE_POLYMER_STRESS_HALO_Y = sizeof(dfloat) * 12 * layout.ghostFaceXZCount;
            const size_t MEM_SIZE_POLYMER_STRESS_HALO_Z = sizeof(dfloat) * 12 * layout.ghostFaceXYCount;
            const size_t MEM_SIZE_POLYMER_STRESS_HALO_AUX = sizeof(dfloat) * 12 * layout.exchangePlaneCount;
            const size_t MEM_SIZE_POLYMER_STRESS_HALO_TOTAL = 4 * (MEM_SIZE_POLYMER_STRESS_HALO_X + MEM_SIZE_POLYMER_STRESS_HALO_Y + MEM_SIZE_POLYMER_STRESS_HALO_Z) + 4 * MEM_SIZE_POLYMER_STRESS_HALO_AUX;
        #endif // CONFORMATION_PHYSICAL_A_HALO

        cudaMalloc((void**)&d_fMom[g], MEM_SIZE_MOM_STORAGE_LOCAL);
        cudaMalloc((void**)&dNodeType[g], sizeof(int) * NUMBER_LBM_NODES_LOCAL);
        checkCudaErrors(cudaMemset(d_fMom[g], 0, MEM_SIZE_MOM_STORAGE_LOCAL));
        checkCudaErrors(cudaMemset(dNodeType[g], 0xff,
            sizeof(unsigned int) * NUMBER_LBM_NODES_LOCAL));
        interfaceMalloc(ghostInterface[g], ghostInterfaceCopy[g], macroInterfaceGPU[g], macroInterfaceGPUCopy[g], layout);

        #ifdef CONFORMATION_PHYSICAL_A_HALO
        // These buffers are derived block-face data, not population or inter-GPU interfaces. They have two AB
        // generations and store physical A for direct stress divergence and transformed-representation diffusion.
        // Keeping them separate avoids changing ghostInterfaceData's population layout and checkpoint roles.
            for (int generation=0; generation<2; ++generation) {
                auto& halo=polymerStressHalo[g][generation];
                checkCudaErrors(cudaMalloc((void**)&halo.X_0, MEM_SIZE_POLYMER_STRESS_HALO_X));
                checkCudaErrors(cudaMalloc((void**)&halo.X_1, MEM_SIZE_POLYMER_STRESS_HALO_X));
                checkCudaErrors(cudaMalloc((void**)&halo.Y_0, MEM_SIZE_POLYMER_STRESS_HALO_Y));
                checkCudaErrors(cudaMalloc((void**)&halo.Y_1, MEM_SIZE_POLYMER_STRESS_HALO_Y));
                checkCudaErrors(cudaMalloc((void**)&halo.Z_0, MEM_SIZE_POLYMER_STRESS_HALO_Z));
                checkCudaErrors(cudaMalloc((void**)&halo.Z_1, MEM_SIZE_POLYMER_STRESS_HALO_Z));
                checkCudaErrors(cudaMalloc((void**)&halo.auxZ_0, MEM_SIZE_POLYMER_STRESS_HALO_AUX));
                checkCudaErrors(cudaMalloc((void**)&halo.auxZ_1, MEM_SIZE_POLYMER_STRESS_HALO_AUX));
                }
            memAllocated += MEM_SIZE_POLYMER_STRESS_HALO_TOTAL;
        #endif // CONFORMATION_PHYSICAL_A_HALO

        memAllocated += MEM_SIZE_MOM_STORAGE_LOCAL + sizeof(int) * NUMBER_LBM_NODES_LOCAL;

        #ifdef BC_FORCES
            cudaMalloc((void**)&d_BC_Fx[g], MEM_SIZE_SCALAR);
            cudaMalloc((void**)&d_BC_Fy[g], MEM_SIZE_SCALAR);
            cudaMalloc((void**)&d_BC_Fz[g], MEM_SIZE_SCALAR);
            memAllocated += 3 * MEM_SIZE_SCALAR;
        #endif //BC_FORCES

        #ifdef SAVE_LOCAL_FORCES
            cudaMalloc((void**)&d_Local_Fx, MEM_SIZE_SCALAR);
            cudaMalloc((void**)&d_Local_Fy, MEM_SIZE_SCALAR);
            cudaMalloc((void**)&d_Local_Fz, MEM_SIZE_SCALAR);
            memAllocated += 3 * MEM_SIZE_SCALAR;
        #ifdef SECOND_DIST
                cudaMalloc((void**)&d_Source_C, MEM_SIZE_SCALAR);
                memAllocated += MEM_SIZE_SCALAR;
        #endif // SECOND_DIST
        #ifdef PHI_DIST
                cudaMalloc((void**)&d_Source_Phi, MEM_SIZE_SCALAR);
                memAllocated += MEM_SIZE_SCALAR;
        #endif // PHI_DIST
        #ifdef LAMBDA_DIST
                cudaMalloc((void**)&d_Source_Lambda, MEM_SIZE_SCALAR);
                memAllocated += MEM_SIZE_SCALAR;
        #endif // LAMBDA_DIST
        #ifdef CONFORMATION_TENSOR
        #ifdef A_XX_DIST
                    cudaMalloc((void**)&d_Source_Gxx, MEM_SIZE_SCALAR);
                    memAllocated += MEM_SIZE_SCALAR;
        #endif // A_XX_DIST
        #ifdef A_XY_DIST
                    cudaMalloc((void**)&d_Source_Gxy, MEM_SIZE_SCALAR);
                    memAllocated += MEM_SIZE_SCALAR;
        #endif // A_XY_DIST
        #ifdef A_XZ_DIST
                    cudaMalloc((void**)&d_Source_Gxz, MEM_SIZE_SCALAR);
                    memAllocated += MEM_SIZE_SCALAR;
        #endif // A_XZ_DIST
        #ifdef A_YY_DIST
                    cudaMalloc((void**)&d_Source_Gyy, MEM_SIZE_SCALAR);
                    memAllocated += MEM_SIZE_SCALAR;
        #endif // A_YY_DIST
        #ifdef A_YZ_DIST
                    cudaMalloc((void**)&d_Source_Gyz, MEM_SIZE_SCALAR);
                    memAllocated += MEM_SIZE_SCALAR;
        #endif // A_YZ_DIST
        #ifdef A_ZZ_DIST
                    cudaMalloc((void**)&d_Source_Gzz, MEM_SIZE_SCALAR);
                    memAllocated += MEM_SIZE_SCALAR;
        #endif // A_ZZ_DIST
        #endif //CONFORMATION_TENSOR
        #endif //SAVE_LOCAL_FORCES

        printf("Device Memory Allocated for Bulk flow: %.2f MB \n", (float)memAllocated /(1024.0 * 1024.0));
    }

    #ifdef DENSITY_CORRECTION
    /** @brief Allocates density-reduction buffers for GPU @p g when enabled. */
        void allocateDensityCorrectionMemory(int g){
            const size_t numBlocksLocal = layout.numBlocksX * layout.numBlocksY * layout.numBlocksZ;
            checkCudaErrors(cudaMalloc((void**)&d_mean_rho[g], sizeof(dfloat)));
            checkCudaErrors(cudaMemset(d_mean_rho[g], 0, sizeof(dfloat)));
            for (int buffer = 0; buffer < 2; ++buffer) {
                checkCudaErrors(cudaMalloc((void**)&d_density_sum[g][buffer],numBlocksLocal * sizeof(dfloat)));
                checkCudaErrors(cudaMalloc((void**)&d_density_count[g][buffer],numBlocksLocal * sizeof(unsigned long long)));
            }
        }
    #endif //DENSITY_CORRECTION

    /** @brief Initializes lattice moments, node types, interfaces, and optional restart state. */
    void initializeDomainDeviceField(hostField &hostField, dfloat **&randomNumbers, int &step,
                                     dim3 threadBlock, int g){
        // ========== INITIALIZATION DOMAIN INLINED ==========
        checkCudaErrors(cudaSetDevice(GPUS_TO_USE[g]));

        const int zStart = static_cast<int>(layout.zStart);
        const int zEnd = static_cast<int>(layout.zStart + layout.localNZ);
        const size_t localNZ = layout.localNZ;
        const dim3 gridBlock = layout.fullGrid;
        const size_t NUMBER_LBM_NODES_LOCAL = layout.paddedNodeCount;
        const size_t MEM_SIZE_MOM_STORAGE_LOCAL = layout.momentStorageBytes;

        // Offset within a global vector
        size_t zOffset = zStart * NX * NY * NUMBER_MOMENTS;

        printf("GPU: %d\n", (int)GPUS_TO_USE[g]);
        printf("Start initialize: %d\n", (int)zStart);
        printf("End initialize: %d\n", (int)zEnd);
        printf("localNZ initialize: %d\n", (int)localNZ);

        // Random numbers initialization
        #ifdef RANDOM_NUMBERS
            if(console_flush) fflush(stdout);
            // Replicate the same global physical sequence on every device.
            // This makes case initialization independent of slab ownership.
            const size_t randomPhysicalCount = static_cast<size_t>(NX) * NY * NZ;
            const size_t randomGeneratedCount = randomPhysicalCount + (randomPhysicalCount & 1u);
            checkCudaErrors(cudaMalloc((void**)&randomNumbers[g], sizeof(dfloat) * randomGeneratedCount));
            initializationRandomNumbers(randomNumbers[g], CURAND_SEED, randomGeneratedCount);
            checkCudaErrors(cudaDeviceSynchronize());

            getLastCudaError("random numbers transfer error");
            printf("Random numbers initialized - Seed used: %u\n", CURAND_SEED);
            printf("Device memory allocated for random numbers: %.2f MB\n", (float)(sizeof(dfloat) * randomGeneratedCount) / (1024.0 * 1024.0));
            if(console_flush) fflush(stdout);
        #endif //RANDOM_NUMBERS

        int checkpoint_state = 0;
        // LBM Initialization
        if (LOAD_CHECKPOINT) {

            printf("Loading checkpoint\n");
            checkpoint_state = loadGlobalSimCheckpoint(d_fMom[g], ghostInterface[g], &step, layout, threadBlock);
            if (checkpoint_state != 0) streamingParityOffset = 0;

            // The global loader restores device faces directly from canonical
            // logical populations. Legacy host face buffers are not involved.
            if (false && checkpoint_state != 0){
                interfaceCudaMemcpy(ghostInterface[g], ghostInterface[g].pop, ghostInterface[g].h_pop, cudaMemcpyHostToDevice, QF, g);

                #ifdef SECOND_DIST
                    interfaceCudaMemcpy(ghostInterface[g], ghostInterface[g].g, ghostInterface[g].h_g, cudaMemcpyHostToDevice, GF, g);
                #endif //SECOND_DIST

                #ifdef PHI_DIST
                    interfaceCudaMemcpy(ghostInterface[g], ghostInterface[g].phi, ghostInterface[g].h_phi, cudaMemcpyHostToDevice, GF, g);
                #endif //PHI_DIST

                #ifdef LAMBDA_DIST
                    interfaceCudaMemcpy(ghostInterface[g], ghostInterface[g].lambda, ghostInterface[g].h_lambda, cudaMemcpyHostToDevice, GF, g);
                #endif //LAMBDA_DIST

                #ifdef A_XX_DIST
                    interfaceCudaMemcpy(ghostInterface[g], ghostInterface[g].Axx, ghostInterface[g].h_Axx, cudaMemcpyHostToDevice, GF, g);
                #endif //A_XX_DIST
                #ifdef A_XY_DIST
                    interfaceCudaMemcpy(ghostInterface[g], ghostInterface[g].Axy, ghostInterface[g].h_Axy, cudaMemcpyHostToDevice, GF, g);
                #endif //A_XY_DIST
                #ifdef A_XZ_DIST
                    interfaceCudaMemcpy(ghostInterface[g], ghostInterface[g].Axz, ghostInterface[g].h_Axz, cudaMemcpyHostToDevice, GF, g);
                #endif //A_XZ_DIST
                #ifdef A_YY_DIST
                    interfaceCudaMemcpy(ghostInterface[g], ghostInterface[g].Ayy, ghostInterface[g].h_Ayy, cudaMemcpyHostToDevice, GF, g);
                #endif //A_YY_DIST
                #ifdef A_YZ_DIST
                    interfaceCudaMemcpy(ghostInterface[g], ghostInterface[g].Ayz, ghostInterface[g].h_Ayz, cudaMemcpyHostToDevice, GF, g);
                #endif //A_YZ_DIST
                #ifdef A_ZZ_DIST
                    interfaceCudaMemcpy(ghostInterface[g], ghostInterface[g].Azz, ghostInterface[g].h_Azz, cudaMemcpyHostToDevice, GF, g);
                #endif //A_ZZ_DIST
            }
        }
        if (!checkpoint_state) {
            if (LOAD_FIELD) {
                // Implement LOAD_FIELD logic if needed
            } else {
                gpuInitialization_mom<<<gridBlock, threadBlock>>>(d_fMom[g], randomNumbers[g], localNZ, zStart);
            }
            gpuInitialization_pop<<<gridBlock, threadBlock>>>(d_fMom[g], ghostInterface[g], localNZ, layout.numBlocksZ, zStart);
        }

        #ifdef STREAMING_AB
        // Both AB generations start from the same reconstructed interface.
            interfaceCudaMemcpy(ghostInterfaceCopy[g], ghostInterfaceCopy[g].pop,
            ghostInterface[g].pop, cudaMemcpyDeviceToDevice, QF, g);
        #endif // STREAMING_AB

        // Mean flow initialization
        #if MEAN_FLOW
            // The decomposition-independent mean baseline is populated by the
            // physical-cell scatter performed after initialization below.
        #endif //MEAN_FLOW

        // Node type initialization - hNodeType is allocated once globally in allocateHostMemoryHostField().
        // Each GPU operates on its own slice: offset = zStart * NX * NY into the global array.
        std::vector<unsigned int> localNodeType(NUMBER_LBM_NODES_LOCAL, SOLID_NODE);
        unsigned int* const hNodeType_slice = localNodeType.data();

        unsigned int numberCurvedBoundaryNodes_local = 0;

        #ifndef VOXEL_FILENAME
            // hostInitialization_nodeType only overwrites boundary nodes.
            // Initialize physical bulk nodes and padded storage deterministically.
            for (size_t zLocal = 0; zLocal < localNZ; ++zLocal)
                for (size_t y = 0; y < static_cast<size_t>(NY); ++y)
                    for (size_t x = 0; x < static_cast<size_t>(NX); ++x)
                        hNodeType_slice[idxScalarBlock(
                            x % BLOCK_NX, y % BLOCK_NY, zLocal % BLOCK_NZ,
                            x / BLOCK_NX, y / BLOCK_NY, zLocal / BLOCK_NZ)] = BULK;
            hostInitialization_nodeType(hNodeType_slice, zStart, localNZ
        #ifdef CURVED_BOUNDARY_CONDITION
                ,&numberCurvedBoundaryNodes_local
        #endif // CURVED_BOUNDARY_CONDITION
            );

            checkCudaErrors(cudaMemcpy(dNodeType[g], hNodeType_slice, sizeof(unsigned int) * NUMBER_LBM_NODES_LOCAL, cudaMemcpyHostToDevice));
            checkCudaErrors(cudaDeviceSynchronize());

        #ifdef FORCE_VOXEL_BC_BUILDING
                define_voxel_bc<<<gridBlock, threadBlock>>>(dNodeType[g]);
                checkCudaErrors(cudaMemcpy(hNodeType_slice, dNodeType[g], sizeof(unsigned int) * NUMBER_LBM_NODES_LOCAL, cudaMemcpyDeviceToHost));
        #endif // FORCE_VOXEL_BC_BUILDING

        #else
            hostInitialization_nodeType_bulk(hostField.hNodeType);
            read_xyz_file(VOXEL_FILENAME, hostField.hNodeType);
            for (size_t zLocal = 0; zLocal < localNZ; ++zLocal) {
                const size_t zGlobal = layout.zStart + zLocal;
                for (size_t y = 0; y < static_cast<size_t>(NY); ++y)
                    for (size_t x = 0; x < static_cast<size_t>(NX); ++x) {
                        const size_t src = idxScalarBlock( x % BLOCK_NX, y % BLOCK_NY, zGlobal % BLOCK_NZ, x / BLOCK_NX, y / BLOCK_NY, zGlobal / BLOCK_NZ);
                        const size_t dst = idxScalarBlock( x % BLOCK_NX, y % BLOCK_NY, zLocal % BLOCK_NZ, x / BLOCK_NX, y / BLOCK_NY, zLocal / BLOCK_NZ);
                        hNodeType_slice[dst] = hostField.hNodeType[src];
                    }
                }
            hostInitialization_nodeType(hNodeType_slice, zStart, localNZ
        #ifdef CURVED_BOUNDARY_CONDITION
                ,&numberCurvedBoundaryNodes_local
        #endif // CURVED_BOUNDARY_CONDITION
            );
            checkCudaErrors(cudaMemcpy(dNodeType[g], hNodeType_slice, sizeof(unsigned int) * NUMBER_LBM_NODES_LOCAL, cudaMemcpyHostToDevice));
            checkCudaErrors(cudaDeviceSynchronize());
            define_voxel_bc<<<gridBlock, threadBlock>>>(dNodeType[g]);
            checkCudaErrors(cudaMemcpy(hNodeType_slice, dNodeType[g], sizeof(unsigned int) * NUMBER_LBM_NODES_LOCAL, cudaMemcpyDeviceToHost));
        #endif //!VOXEL_FILENAME

        for (size_t zLocal = 0; zLocal < localNZ; ++zLocal) {
            const size_t zGlobal = layout.zStart + zLocal;
            for (size_t y = 0; y < static_cast<size_t>(NY); ++y)
                for (size_t x = 0; x < static_cast<size_t>(NX); ++x) {
                    const size_t src = idxScalarBlock(x % BLOCK_NX, y % BLOCK_NY, zLocal % BLOCK_NZ, x / BLOCK_NX, y / BLOCK_NY, zLocal / BLOCK_NZ);
                    const size_t dst = idxScalarBlock(x % BLOCK_NX, y % BLOCK_NY, zGlobal % BLOCK_NZ, x / BLOCK_NX, y / BLOCK_NY, zGlobal / BLOCK_NZ);
                    hostField.hNodeType[dst] = hNodeType_slice[src];
                }
        }

        // Boundary condition forces initialization
        #ifdef BC_FORCES
            gpuInitialization_force<<<gridBlock, threadBlock>>>(d_BC_Fx[g], d_BC_Fy[g], d_BC_Fz[g]);
        #endif //BC_FORCES

        #ifdef SAVE_LOCAL_FORCES
            gpuInitialization_force<<<gridBlock, threadBlock>>>(d_Local_Fx, d_Local_Fy, d_Local_Fz);
        #ifdef SECOND_DIST
                gpuInitialization_force<<<gridBlock, threadBlock>>>(d_Source_C, d_Source_C, d_Source_C);
        #endif // SECOND_DIST
        #ifdef PHI_DIST
                gpuInitialization_force<<<gridBlock, threadBlock>>>(d_Source_Phi, d_Source_Phi, d_Source_Phi);
        #endif // PHI_DIST
        #ifdef LAMBDA_DIST
                gpuInitialization_force<<<gridBlock, threadBlock>>>(d_Source_Lambda, d_Source_Lambda, d_Source_Lambda);
        #endif // LAMBDA_DIST
        #ifdef CONFORMATION_TENSOR
        #ifdef A_XX_DIST
                    gpuInitialization_force<<<gridBlock, threadBlock>>>(d_Source_Gxx, d_Source_Gxx, d_Source_Gxx);
        #endif // A_XX_DIST
        #ifdef A_XY_DIST
                    gpuInitialization_force<<<gridBlock, threadBlock>>>(d_Source_Gxy, d_Source_Gxy, d_Source_Gxy);
        #endif // A_XY_DIST
        #ifdef A_XZ_DIST
                    gpuInitialization_force<<<gridBlock, threadBlock>>>(d_Source_Gxz, d_Source_Gxz, d_Source_Gxz);
        #endif // A_XZ_DIST
        #ifdef A_YY_DIST
                    gpuInitialization_force<<<gridBlock, threadBlock>>>(d_Source_Gyy, d_Source_Gyy, d_Source_Gyy);
        #endif // A_YY_DIST
        #ifdef A_YZ_DIST
                    gpuInitialization_force<<<gridBlock, threadBlock>>>(d_Source_Gyz, d_Source_Gyz, d_Source_Gyz);
        #endif // A_YZ_DIST
        #ifdef A_ZZ_DIST
                    gpuInitialization_force<<<gridBlock, threadBlock>>>(d_Source_Gzz, d_Source_Gzz, d_Source_Gzz);
        #endif // A_ZZ_DIST
        #endif //CONFORMATION_TENSOR
        #endif //SAVE_LOCAL_FORCES

        #ifdef CURVED_BOUNDARY_CONDITION
            numberCurvedBoundaryNodes = initializeCurvedBoundaryDeviceField(
                hNodeType_slice,
                dNodeType[g],
                d_curvedBC[g],
                d_curvedBC_array[g],
                zStart,
                localNZ,
                layout.paddedNodeCount
            );
            // The main collision kernel consumes curvedBC->vel. Populate it from
            // the initialized moment field before the first simulation step.
            updateCurvedBoundaryVelocitiesDeviceField(g, 0);
            CHECK_KERNEL_ERR("Initial curved BC kernel");
        #if defined(CONFORMATION_TENSOR) && defined(D3G19)
                updateCurvedBoundaryConformationDeviceField(g, 0);
                CHECK_KERNEL_ERR("Initial curved conformation BC kernel");
        #endif // defined(CONFORMATION_TENSOR) && defined(D3G19)
            checkCudaErrors(cudaDeviceSynchronize());
        #endif // CURVED_BOUNDARY_CONDITION

        // Interface population initialization
        #ifdef SECOND_DIST
        #endif //SECOND_DIST
        #ifdef PHI_DIST
        #endif //PHI_DIST
        #ifdef LAMBDA_DIST
        #endif //LAMBDA_DIST
        #ifdef A_XX_DIST
        #endif //A_XX_DIST
        #ifdef A_XY_DIST
        #endif //A_XY_DIST
        #ifdef A_XZ_DIST
        #endif //A_XZ_DIST
        #ifdef A_YY_DIST
        #endif //A_YY_DIST
        #ifdef A_YZ_DIST
        #endif //A_YZ_DIST
        #ifdef A_ZZ_DIST
        #endif //A_ZZ_DIST

        // Synchronize after all initializations
        checkCudaErrors(cudaDeviceSynchronize());

        // Synchronize and transfer data back to host if needed
        checkCudaErrors(cudaDeviceSynchronize());
        std::vector<dfloat> initializedMom(NUMBER_LBM_NODES_LOCAL * NUMBER_MOMENTS);
        checkCudaErrors(cudaMemcpy(initializedMom.data(), d_fMom[g],
            MEM_SIZE_MOM_STORAGE_LOCAL, cudaMemcpyDeviceToHost));
        for (size_t zLocal = 0; zLocal < localNZ; ++zLocal) {
            const size_t zGlobal = layout.zStart + zLocal;
            for (size_t y = 0; y < static_cast<size_t>(NY); ++y)
                for (size_t x = 0; x < static_cast<size_t>(NX); ++x)
                    for (int moment = 0; moment < NUMBER_MOMENTS; ++moment) {
                        const size_t src = idxMom(x % BLOCK_NX, y % BLOCK_NY, zLocal % BLOCK_NZ, moment, x / BLOCK_NX, y / BLOCK_NY, zLocal / BLOCK_NZ);
                        const size_t dst = idxMom(x % BLOCK_NX, y % BLOCK_NY, zGlobal % BLOCK_NZ, moment, x / BLOCK_NX, y / BLOCK_NY, zGlobal / BLOCK_NZ);
                        hostField.h_fMom[dst] = initializedMom[src];
                #if MEAN_FLOW
                        hostField.m_fMom[dst] = initializedMom[src];
                #endif // MEAN_FLOW
                    }
        }
        checkCudaErrors(cudaDeviceSynchronize());

        printf("Syncing data back to host (g=%d) \n", g); if(console_flush) fflush(stdout);


    }

    #ifdef CURVED_BOUNDARY_CONDITION
    /** @brief Updates moving curved-boundary velocities on GPU @p g. */
        void updateCurvedBoundaryVelocitiesDeviceField(int g, cudaStream_t stream){
        // Skip launch when no curved-boundary nodes were found; zero grid size is invalid
            if (numberCurvedBoundaryNodes == 0) {
                return;
                }

            const int curvedBCBlockSize = 256;
            const int curvedBCGridSize = (numberCurvedBoundaryNodes + curvedBCBlockSize - 1) / curvedBCBlockSize;
            updateCurvedBoundaryVelocities<<<curvedBCGridSize, curvedBCBlockSize, 0, stream>>>(d_curvedBC_array[g], d_fMom[g], numberCurvedBoundaryNodes);
            }

    #if defined(CONFORMATION_TENSOR) && defined(D3G19)
    /** @brief Updates curved-boundary conformation data on GPU @p g. */
            void updateCurvedBoundaryConformationDeviceField(int g, cudaStream_t stream){
                if (numberCurvedBoundaryNodes == 0) return;
                const int curvedBCBlockSize = 256;
                const int curvedBCGridSize = (numberCurvedBoundaryNodes + curvedBCBlockSize - 1) / curvedBCBlockSize;
                updateCurvedBoundaryConformation<<<curvedBCGridSize, curvedBCBlockSize, 0, stream>>>(
                d_curvedBC_array[g], d_fMom[g], numberCurvedBoundaryNodes);
                }
    #endif // defined(CONFORMATION_TENSOR) && defined(D3G19)
    #endif //CURVED_BOUNDARY_CONDITION

    #ifdef DENSITY_CORRECTION
    /** @brief Launches and collects the local density reduction for one GPU. */
        void reduceDensityPartialDeviceField(
        int g, size_t localNZ, dfloat* hostSum,
        unsigned long long* hostCount, cudaStream_t stream){
            reduceDensityPartial(
            d_fMom[g], dNodeType[g],
            d_density_sum[g][0], d_density_sum[g][1],
            d_density_count[g][0], d_density_count[g][1],
            localNZ, hostSum, hostCount, stream);
            }
    #endif //DENSITY_CORRECTION

    /** @brief Launches the collision-and-streaming kernel for a local slab. */
    void gpuMomCollisionStreamDeviceField(dim3 threadBlock, unsigned int step, bool save,
        int g, cudaStream_t stream,
        ZKernelLaunchRegion launchRegion = ZKernelLaunchRegion::All){
        // Create parameter struct and pass by value (most efficient!)
        cudaSetDevice(GPUS_TO_USE[g]);
        DeviceKernelParams params;
        const dim3 gridBlock = layout.fullGrid;
        const int zStart = static_cast<int>(layout.zStart);
        const size_t localNZ = layout.localNZ;

        params.fMom = d_fMom[g];
        params.dNodeType = dNodeType[g];
        params.pop.X_0    = ghostInterface[g].pop.X_0;
        params.pop.X_1    = ghostInterface[g].pop.X_1;
        params.pop.Y_0    = ghostInterface[g].pop.Y_0;
        params.pop.Y_1    = ghostInterface[g].pop.Y_1;
        params.pop.Z_0    = ghostInterface[g].pop.Z_0;
        params.pop.Z_1    = ghostInterface[g].pop.Z_1;
        params.pop.auxZ_0 = ghostInterface[g].popAux.Z_0;
        params.pop.auxZ_1 = ghostInterface[g].popAux.Z_1;

        params.popCopy.auxZ_0 = ghostInterfaceCopy[g].popAux.Z_0;
        params.popCopy.auxZ_1 = ghostInterfaceCopy[g].popAux.Z_1;

        #ifdef STREAMING_AB
            params.popCopy.X_0 = ghostInterfaceCopy[g].pop.X_0;
            params.popCopy.X_1 = ghostInterfaceCopy[g].pop.X_1;
            params.popCopy.Y_0 = ghostInterfaceCopy[g].pop.Y_0;
            params.popCopy.Y_1 = ghostInterfaceCopy[g].pop.Y_1;
            params.popCopy.Z_0 = ghostInterfaceCopy[g].pop.Z_0;
            params.popCopy.Z_1 = ghostInterfaceCopy[g].pop.Z_1;
        #endif // STREAMING_AB

        params.step = step;
        params.streamingParity = effectiveStreamingParity(step);
        params.save = save;
        params.zStart = zStart;
        params.localNZ = localNZ;
        params.numBlocksZ = static_cast<unsigned int>(layout.numBlocksZ);
        params.zBlockOffset = 0;
        params.mapBoundaryBlocks = false;

        #ifdef CONFORMATION_PHYSICAL_A_HALO
            params.polymerStressHalo = polymerStressHalo[g][params.streamingParity];
        #endif // CONFORMATION_PHYSICAL_A_HALO

        params.rho_macro.Z_0 = macroInterfaceGPU[g].rho.Z_0;
        params.rho_macro.Z_1 = macroInterfaceGPU[g].rho.Z_1;
        params.rho_macro.auxZ_0 = macroInterfaceGPU[g].rho.auxZ_0;
        params.rho_macro.auxZ_1 = macroInterfaceGPU[g].rho.auxZ_1;

        params.rho_macroCopy.auxZ_0 = macroInterfaceGPUCopy[g].rho.auxZ_0;
        params.rho_macroCopy.auxZ_1 = macroInterfaceGPUCopy[g].rho.auxZ_1;

        params.ux_macro.Z_0 = macroInterfaceGPU[g].ux.Z_0;
        params.ux_macro.Z_1 = macroInterfaceGPU[g].ux.Z_1;
        params.ux_macro.auxZ_0 = macroInterfaceGPU[g].ux.auxZ_0;
        params.ux_macro.auxZ_1 = macroInterfaceGPU[g].ux.auxZ_1;

        params.ux_macroCopy.auxZ_0 = macroInterfaceGPUCopy[g].ux.auxZ_0;
        params.ux_macroCopy.auxZ_1 = macroInterfaceGPUCopy[g].ux.auxZ_1;

        params.uy_macro.Z_0 = macroInterfaceGPU[g].uy.Z_0;
        params.uy_macro.Z_1 = macroInterfaceGPU[g].uy.Z_1;
        params.uy_macro.auxZ_0 = macroInterfaceGPU[g].uy.auxZ_0;
        params.uy_macro.auxZ_1 = macroInterfaceGPU[g].uy.auxZ_1;

        params.uy_macroCopy.auxZ_0 = macroInterfaceGPUCopy[g].uy.auxZ_0;
        params.uy_macroCopy.auxZ_1 = macroInterfaceGPUCopy[g].uy.auxZ_1;

        params.uz_macro.Z_0 = macroInterfaceGPU[g].uz.Z_0;
        params.uz_macro.Z_1 = macroInterfaceGPU[g].uz.Z_1;
        params.uz_macro.auxZ_0 = macroInterfaceGPU[g].uz.auxZ_0;
        params.uz_macro.auxZ_1 = macroInterfaceGPU[g].uz.auxZ_1;

        params.uz_macroCopy.auxZ_0 = macroInterfaceGPUCopy[g].uz.auxZ_0;
        params.uz_macroCopy.auxZ_1 = macroInterfaceGPUCopy[g].uz.auxZ_1;

        #ifdef SECOND_DIST
            params.g.X_0    = ghostInterface[g].g.X_0;
            params.g.X_1    = ghostInterface[g].g.X_1;
            params.g.Y_0    = ghostInterface[g].g.Y_0;
            params.g.Y_1    = ghostInterface[g].g.Y_1;
            params.g.Z_0    = ghostInterface[g].g.Z_0;
            params.g.Z_1    = ghostInterface[g].g.Z_1;
            params.g.auxZ_0 = ghostInterface[g].gAux.Z_0;
            params.g.auxZ_1 = ghostInterface[g].gAux.Z_1;

            params.gCopy.auxZ_0 = ghostInterfaceCopy[g].gAux.Z_0;
            params.gCopy.auxZ_1 = ghostInterfaceCopy[g].gAux.Z_1;

            params.g_macro.Z_0 = macroInterfaceGPU[g].g.Z_0;
            params.g_macro.Z_1 = macroInterfaceGPU[g].g.Z_1;
            params.g_macro.auxZ_0 = macroInterfaceGPU[g].g.auxZ_0;
            params.g_macro.auxZ_1 = macroInterfaceGPU[g].g.auxZ_1;

            params.g_macroCopy.auxZ_0 = macroInterfaceGPUCopy[g].g.auxZ_0;
            params.g_macroCopy.auxZ_1 = macroInterfaceGPUCopy[g].g.auxZ_1;
        #endif //SECOND_DIST

        #ifdef LAMBDA_DIST
            params.lambda.X_0    = ghostInterface[g].lambda.X_0;
            params.lambda.X_1    = ghostInterface[g].lambda.X_1;
            params.lambda.Y_0    = ghostInterface[g].lambda.Y_0;
            params.lambda.Y_1    = ghostInterface[g].lambda.Y_1;
            params.lambda.Z_0    = ghostInterface[g].lambda.Z_0;
            params.lambda.Z_1    = ghostInterface[g].lambda.Z_1;
            params.lambda.auxZ_0 = ghostInterface[g].lambdaAux.Z_0;
            params.lambda.auxZ_1 = ghostInterface[g].lambdaAux.Z_1;

            params.lambdaCopy.auxZ_0 = ghostInterfaceCopy[g].lambdaAux.Z_0;
            params.lambdaCopy.auxZ_1 = ghostInterfaceCopy[g].lambdaAux.Z_1;

            params.lambda_macro.Z_0 = macroInterfaceGPU[g].lambda.Z_0;
            params.lambda_macro.Z_1 = macroInterfaceGPU[g].lambda.Z_1;
            params.lambda_macro.auxZ_0 = macroInterfaceGPU[g].lambda.auxZ_0;
            params.lambda_macro.auxZ_1 = macroInterfaceGPU[g].lambda.auxZ_1;

            params.lambda_macroCopy.auxZ_0 = macroInterfaceGPUCopy[g].lambda.auxZ_0;
            params.lambda_macroCopy.auxZ_1 = macroInterfaceGPUCopy[g].lambda.auxZ_1;
        #endif //LAMBDA_DIST

        #ifdef A_XX_DIST
            params.Axx.X_0    = ghostInterface[g].Axx.X_0;
            params.Axx.X_1    = ghostInterface[g].Axx.X_1;
            params.Axx.Y_0    = ghostInterface[g].Axx.Y_0;
            params.Axx.Y_1    = ghostInterface[g].Axx.Y_1;
            params.Axx.Z_0    = ghostInterface[g].Axx.Z_0;
            params.Axx.Z_1    = ghostInterface[g].Axx.Z_1;
            params.Axx.auxZ_0 = ghostInterface[g].AxxAux.Z_0;
            params.Axx.auxZ_1 = ghostInterface[g].AxxAux.Z_1;

            params.AxxCopy.auxZ_0 = ghostInterfaceCopy[g].AxxAux.Z_0;
            params.AxxCopy.auxZ_1 = ghostInterfaceCopy[g].AxxAux.Z_1;

            params.Axx_macro.Z_0 = macroInterfaceGPU[g].Axx.Z_0;
            params.Axx_macro.Z_1 = macroInterfaceGPU[g].Axx.Z_1;
            params.Axx_macro.auxZ_0 = macroInterfaceGPU[g].Axx.auxZ_0;
            params.Axx_macro.auxZ_1 = macroInterfaceGPU[g].Axx.auxZ_1;

            params.Axx_macroCopy.auxZ_0 = macroInterfaceGPUCopy[g].Axx.auxZ_0;
            params.Axx_macroCopy.auxZ_1 = macroInterfaceGPUCopy[g].Axx.auxZ_1;
        #endif //A_XX_DIST
        #ifdef A_XY_DIST
            params.Axy.X_0    = ghostInterface[g].Axy.X_0;
            params.Axy.X_1    = ghostInterface[g].Axy.X_1;
            params.Axy.Y_0    = ghostInterface[g].Axy.Y_0;
            params.Axy.Y_1    = ghostInterface[g].Axy.Y_1;
            params.Axy.Z_0    = ghostInterface[g].Axy.Z_0;
            params.Axy.Z_1    = ghostInterface[g].Axy.Z_1;
            params.Axy.auxZ_0 = ghostInterface[g].AxyAux.Z_0;
            params.Axy.auxZ_1 = ghostInterface[g].AxyAux.Z_1;

            params.AxyCopy.auxZ_0 = ghostInterfaceCopy[g].AxyAux.Z_0;
            params.AxyCopy.auxZ_1 = ghostInterfaceCopy[g].AxyAux.Z_1;

            params.Axy_macro.Z_0 = macroInterfaceGPU[g].Axy.Z_0;
            params.Axy_macro.Z_1 = macroInterfaceGPU[g].Axy.Z_1;
            params.Axy_macro.auxZ_0 = macroInterfaceGPU[g].Axy.auxZ_0;
            params.Axy_macro.auxZ_1 = macroInterfaceGPU[g].Axy.auxZ_1;

            params.Axy_macroCopy.auxZ_0 = macroInterfaceGPUCopy[g].Axy.auxZ_0;
            params.Axy_macroCopy.auxZ_1 = macroInterfaceGPUCopy[g].Axy.auxZ_1;

        #endif //A_XY_DIST
        #ifdef A_XZ_DIST
            params.Axz.X_0    = ghostInterface[g].Axz.X_0;
            params.Axz.X_1    = ghostInterface[g].Axz.X_1;
            params.Axz.Y_0    = ghostInterface[g].Axz.Y_0;
            params.Axz.Y_1    = ghostInterface[g].Axz.Y_1;
            params.Axz.Z_0    = ghostInterface[g].Axz.Z_0;
            params.Axz.Z_1    = ghostInterface[g].Axz.Z_1;
            params.Axz.auxZ_0 = ghostInterface[g].AxzAux.Z_0;
            params.Axz.auxZ_1 = ghostInterface[g].AxzAux.Z_1;

            params.AxzCopy.auxZ_0 = ghostInterfaceCopy[g].AxzAux.Z_0;
            params.AxzCopy.auxZ_1 = ghostInterfaceCopy[g].AxzAux.Z_1;

            params.Axz_macro.Z_0 = macroInterfaceGPU[g].Axz.Z_0;
            params.Axz_macro.Z_1 = macroInterfaceGPU[g].Axz.Z_1;
            params.Axz_macro.auxZ_0 = macroInterfaceGPU[g].Axz.auxZ_0;
            params.Axz_macro.auxZ_1 = macroInterfaceGPU[g].Axz.auxZ_1;

            params.Axz_macroCopy.auxZ_0 = macroInterfaceGPUCopy[g].Axz.auxZ_0;
            params.Axz_macroCopy.auxZ_1 = macroInterfaceGPUCopy[g].Axz.auxZ_1;
        #endif //A_XZ_DIST
        #ifdef A_YY_DIST
            params.Ayy.X_0    = ghostInterface[g].Ayy.X_0;
            params.Ayy.X_1    = ghostInterface[g].Ayy.X_1;
            params.Ayy.Y_0    = ghostInterface[g].Ayy.Y_0;
            params.Ayy.Y_1    = ghostInterface[g].Ayy.Y_1;
            params.Ayy.Z_0    = ghostInterface[g].Ayy.Z_0;
            params.Ayy.Z_1    = ghostInterface[g].Ayy.Z_1;
            params.Ayy.auxZ_0 = ghostInterface[g].AyyAux.Z_0;
            params.Ayy.auxZ_1 = ghostInterface[g].AyyAux.Z_1;

            params.AyyCopy.auxZ_0 = ghostInterfaceCopy[g].AyyAux.Z_0;
            params.AyyCopy.auxZ_1 = ghostInterfaceCopy[g].AyyAux.Z_1;

            params.Ayy_macro.Z_0 = macroInterfaceGPU[g].Ayy.Z_0;
            params.Ayy_macro.Z_1 = macroInterfaceGPU[g].Ayy.Z_1;
            params.Ayy_macro.auxZ_0 = macroInterfaceGPU[g].Ayy.auxZ_0;
            params.Ayy_macro.auxZ_1 = macroInterfaceGPU[g].Ayy.auxZ_1;

            params.Ayy_macroCopy.auxZ_0 = macroInterfaceGPUCopy[g].Ayy.auxZ_0;
            params.Ayy_macroCopy.auxZ_1 = macroInterfaceGPUCopy[g].Ayy.auxZ_1;
        #endif //A_YY_DIST
        #ifdef A_YZ_DIST
            params.Ayz.X_0    = ghostInterface[g].Ayz.X_0;
            params.Ayz.X_1    = ghostInterface[g].Ayz.X_1;
            params.Ayz.Y_0    = ghostInterface[g].Ayz.Y_0;
            params.Ayz.Y_1    = ghostInterface[g].Ayz.Y_1;
            params.Ayz.Z_0    = ghostInterface[g].Ayz.Z_0;
            params.Ayz.Z_1    = ghostInterface[g].Ayz.Z_1;
            params.Ayz.auxZ_0 = ghostInterface[g].AyzAux.Z_0;
            params.Ayz.auxZ_1 = ghostInterface[g].AyzAux.Z_1;

            params.AyzCopy.auxZ_0 = ghostInterfaceCopy[g].AyzAux.Z_0;
            params.AyzCopy.auxZ_1 = ghostInterfaceCopy[g].AyzAux.Z_1;

            params.Ayz_macro.Z_0 = macroInterfaceGPU[g].Ayz.Z_0;
            params.Ayz_macro.Z_1 = macroInterfaceGPU[g].Ayz.Z_1;
            params.Ayz_macro.auxZ_0 = macroInterfaceGPU[g].Ayz.auxZ_0;
            params.Ayz_macro.auxZ_1 = macroInterfaceGPU[g].Ayz.auxZ_1;

            params.Ayz_macroCopy.auxZ_0 = macroInterfaceGPUCopy[g].Ayz.auxZ_0;
            params.Ayz_macroCopy.auxZ_1 = macroInterfaceGPUCopy[g].Ayz.auxZ_1;
        #endif //A_YZ_DIST
        #ifdef A_ZZ_DIST
            params.Azz.X_0    = ghostInterface[g].Azz.X_0;
            params.Azz.X_1    = ghostInterface[g].Azz.X_1;
            params.Azz.Y_0    = ghostInterface[g].Azz.Y_0;
            params.Azz.Y_1    = ghostInterface[g].Azz.Y_1;
            params.Azz.Z_0    = ghostInterface[g].Azz.Z_0;
            params.Azz.Z_1    = ghostInterface[g].Azz.Z_1;
            params.Azz.auxZ_0 = ghostInterface[g].AzzAux.Z_0;
            params.Azz.auxZ_1 = ghostInterface[g].AzzAux.Z_1;

            params.AzzCopy.auxZ_0 = ghostInterfaceCopy[g].AzzAux.Z_0;
            params.AzzCopy.auxZ_1 = ghostInterfaceCopy[g].AzzAux.Z_1;

            params.Azz_macro.Z_0 = macroInterfaceGPU[g].Azz.Z_0;
            params.Azz_macro.Z_1 = macroInterfaceGPU[g].Azz.Z_1;
            params.Azz_macro.auxZ_0 = macroInterfaceGPU[g].Azz.auxZ_0;
            params.Azz_macro.auxZ_1 = macroInterfaceGPU[g].Azz.auxZ_1;

            params.Azz_macroCopy.auxZ_0 = macroInterfaceGPUCopy[g].Azz.auxZ_0;
            params.Azz_macroCopy.auxZ_1 = macroInterfaceGPUCopy[g].Azz.auxZ_1;
        #endif //A_ZZ_DIST

        #ifdef DENSITY_CORRECTION
            params.d_mean_rho = d_mean_rho[g];
        #endif //DENSITY_CORRECTION

        #ifdef BC_FORCES
            params.d_BC_Fx = d_BC_Fx[g];
            params.d_BC_Fy = d_BC_Fy[g];
            params.d_BC_Fz = d_BC_Fz[g];
        #endif //BC_FORCES

        #ifdef SAVE_LOCAL_FORCES
            params.d_Local_Fx = d_Local_Fx;
            params.d_Local_Fy = d_Local_Fy;
            params.d_Local_Fz = d_Local_Fz;
        #ifdef SECOND_DIST
                params.d_Source_C = d_Source_C;
        #endif // SECOND_DIST
        #ifdef PHI_DIST
                params.d_Source_Phi = d_Source_Phi;
        #endif // PHI_DIST
        #ifdef LAMBDA_DIST
                params.d_Source_Lambda = d_Source_Lambda;
        #endif // LAMBDA_DIST
        #ifdef CONFORMATION_TENSOR
        #ifdef A_XX_DIST
                    params.d_Source_Gxx = d_Source_Gxx;
        #endif // A_XX_DIST
        #ifdef A_XY_DIST
                    params.d_Source_Gxy = d_Source_Gxy;
        #endif // A_XY_DIST
        #ifdef A_XZ_DIST
                    params.d_Source_Gxz = d_Source_Gxz;
        #endif // A_XZ_DIST
        #ifdef A_YY_DIST
                    params.d_Source_Gyy = d_Source_Gyy;
        #endif // A_YY_DIST
        #ifdef A_YZ_DIST
                    params.d_Source_Gyz = d_Source_Gyz;
        #endif // A_YZ_DIST
        #ifdef A_ZZ_DIST
                    params.d_Source_Gzz = d_Source_Gzz;
        #endif // A_ZZ_DIST
        #endif //CONFORMATION_TENSOR
        #endif //SAVE_LOCAL_FORCES

        #ifdef CURVED_BOUNDARY_CONDITION
            params.d_curvedBC = d_curvedBC[g];
            params.d_curvedBC_array = d_curvedBC_array[g];
        #endif //CURVED_BOUNDARY_CONDITION

        #if defined(NON_NEWTONIAN_FLUID) || defined(CONFORMATION_TENSOR)
            params.phasePropsA = phasePropsA;
        #ifdef PHI_DIST
                params.phasePropsB = phasePropsB;
                params.phi.X_0    = ghostInterface[g].phi.X_0;
                params.phi.X_1    = ghostInterface[g].phi.X_1;
                params.phi.Y_0    = ghostInterface[g].phi.Y_0;
                params.phi.Y_1    = ghostInterface[g].phi.Y_1;
                params.phi.Z_0    = ghostInterface[g].phi.Z_0;
                params.phi.Z_1    = ghostInterface[g].phi.Z_1;
                params.phi.auxZ_0 = ghostInterface[g].phiAux.Z_0;
                params.phi.auxZ_1 = ghostInterface[g].phiAux.Z_1;

                params.phiCopy.auxZ_0 = ghostInterfaceCopy[g].phiAux.Z_0;
                params.phiCopy.auxZ_1 = ghostInterfaceCopy[g].phiAux.Z_1;

                params.phi_macro.Z_0 = macroInterfaceGPU[g].phi.Z_0;
                params.phi_macro.Z_1 = macroInterfaceGPU[g].phi.Z_1;
                params.phi_macro.auxZ_0 = macroInterfaceGPU[g].phi.auxZ_0;
                params.phi_macro.auxZ_1 = macroInterfaceGPU[g].phi.auxZ_1;

                params.phi_macroCopy.auxZ_0 = macroInterfaceGPUCopy[g].phi.auxZ_0;
                params.phi_macroCopy.auxZ_1 = macroInterfaceGPUCopy[g].phi.auxZ_1;

                params.nx_macro.Z_0 = macroInterfaceGPU[g].nx.Z_0;
                params.nx_macro.Z_1 = macroInterfaceGPU[g].nx.Z_1;
                params.nx_macro.auxZ_0 = macroInterfaceGPU[g].nx.auxZ_0;
                params.nx_macro.auxZ_1 = macroInterfaceGPU[g].nx.auxZ_1;

                params.nx_macroCopy.auxZ_0 = macroInterfaceGPUCopy[g].nx.auxZ_0;
                params.nx_macroCopy.auxZ_1 = macroInterfaceGPUCopy[g].nx.auxZ_1;

                params.ny_macro.Z_0 = macroInterfaceGPU[g].ny.Z_0;
                params.ny_macro.Z_1 = macroInterfaceGPU[g].ny.Z_1;
                params.ny_macro.auxZ_0 = macroInterfaceGPU[g].ny.auxZ_0;
                params.ny_macro.auxZ_1 = macroInterfaceGPU[g].ny.auxZ_1;

                params.ny_macroCopy.auxZ_0 = macroInterfaceGPUCopy[g].ny.auxZ_0;
                params.ny_macroCopy.auxZ_1 = macroInterfaceGPUCopy[g].ny.auxZ_1;

                params.nz_macro.Z_0 = macroInterfaceGPU[g].nz.Z_0;
                params.nz_macro.Z_1 = macroInterfaceGPU[g].nz.Z_1;
                params.nz_macro.auxZ_0 = macroInterfaceGPU[g].nz.auxZ_0;
                params.nz_macro.auxZ_1 = macroInterfaceGPU[g].nz.auxZ_1;

                params.nz_macroCopy.auxZ_0 = macroInterfaceGPUCopy[g].nz.auxZ_0;
                params.nz_macroCopy.auxZ_1 = macroInterfaceGPUCopy[g].nz.auxZ_1;

        #ifdef PHASE_NEEDS_CHEMICAL_POTENTIAL
                    params.mu_macro.Z_0 = macroInterfaceGPU[g].mu.Z_0;
                    params.mu_macro.Z_1 = macroInterfaceGPU[g].mu.Z_1;
                    params.mu_macro.auxZ_0 = macroInterfaceGPU[g].mu.auxZ_0;
                    params.mu_macro.auxZ_1 = macroInterfaceGPU[g].mu.auxZ_1;

                    params.mu_macroCopy.auxZ_0 = macroInterfaceGPUCopy[g].mu.auxZ_0;
                    params.mu_macroCopy.auxZ_1 = macroInterfaceGPUCopy[g].mu.auxZ_1;
        #endif // PHASE_NEEDS_CHEMICAL_POTENTIAL

        #endif //PHI_DIST
        #endif //NON_NEWTONIAN_FLUID || CONFORMATION_TENSOR

        // Pass struct by value - CUDA handles this efficiently.
        #ifdef SPLIT_Z_COLLISION_KERNEL
            auto launchZRange = [&](unsigned int firstBlock, unsigned int blockCount) {
                if (blockCount == 0) return;
                params.zBlockOffset = firstBlock;
                const dim3 splitGrid(gridBlock.x, gridBlock.y, blockCount);
            #ifdef DYNAMIC_SHARED_MEMORY
                    gpuMomCollisionStream<<<splitGrid, threadBlock, MAX_SHARED_MEMORY_SIZE, stream>>>(params);
            #else
                    gpuMomCollisionStream<<<splitGrid, threadBlock, 0, stream>>>(params);
            #endif // DYNAMIC_SHARED_MEMORY
                };

            const unsigned int localBlockCount = gridBlock.z;
            if (launchRegion != ZKernelLaunchRegion::Boundaries && layout.interiorGrid.z != 0) {
                launchZRange(1, layout.interiorGrid.z); // Interior: no incoming Z halo dependency.
                }
            if (launchRegion == ZKernelLaunchRegion::Boundaries) {
                params.mapBoundaryBlocks = true;
                launchZRange(0, layout.boundaryGrid.z);
                } else if (launchRegion == ZKernelLaunchRegion::All) {
                launchZRange(0, 1);                  // Back boundary.
                if (localBlockCount > 1) {
                    launchZRange(localBlockCount - 1, 1); // Front boundary; avoid a duplicate one-block slab.
                    }
                }
        #else
        #ifdef DYNAMIC_SHARED_MEMORY
                gpuMomCollisionStream<<<gridBlock, threadBlock, MAX_SHARED_MEMORY_SIZE, stream>>>(params);
        #else
                gpuMomCollisionStream<<<gridBlock, threadBlock, 0, stream>>>(params);
        #endif // DYNAMIC_SHARED_MEMORY
        #endif // SPLIT_Z_COLLISION_KERNEL
    }

    #ifdef CONFORMATION_PHYSICAL_A_HALO
    /** @brief Reconstructs physical polymer-stress halos after transport. */
        void fillPolymerStressHaloDeviceField(
        dim3 gridBlock, dim3 threadBlock, unsigned int generation,
        int g, cudaStream_t stream)
        {
            checkCudaErrors(cudaSetDevice(GPUS_TO_USE[g]));
            gpuFillPolymerStressHalo<<<gridBlock,threadBlock,0,stream>>>(
            d_fMom[g], polymerStressHalo[g][generation & 1u],
            layout.localNZ);
            CHECK_KERNEL_ERR("Physical-conformation halo fill kernel");
        }

    /** @brief Exchanges physical polymer-stress Z halos between GPU slabs. */
        void exchangePolymerStressHalosDeviceField(
        int g, deviceField* allDevices, unsigned int generation,
        cudaStream_t stream)
        {
            checkCudaErrors(cudaSetDevice(GPUS_TO_USE[g]));
            const int next = (g + 1) % N_GPUS;
            auto& own = polymerStressHalo[g][generation & 1u];
            auto& nextHalo = allDevices[next].polymerStressHalo[next][generation & 1u];
            const size_t planeElements = 12 * layout.exchangePlaneCount;
            const size_t bytes = planeElements * sizeof(dfloat);
            const size_t topOffset = (layout.numBlocksZ - 1) * planeElements;
            checkCudaErrors(cudaMemcpyPeerAsync(
            nextHalo.auxZ_1, GPUS_TO_USE[next], own.Z_1 + topOffset,
            GPUS_TO_USE[g], bytes, stream));
            checkCudaErrors(cudaMemcpyPeerAsync(
            own.auxZ_0, GPUS_TO_USE[g], nextHalo.Z_0,
            GPUS_TO_USE[next], bytes, stream));
        }
    #endif // CONFORMATION_PHYSICAL_A_HALO

    /** @brief Packs macroscopic Z faces for GPU @p g. */
    void packMacroHalosDeviceField(int g, cudaStream_t stream) {
        checkCudaErrors(cudaSetDevice(GPUS_TO_USE[g]));
        const dim3 grid((NX * NY + 255) / 256, 2);
        gpuPackMacroHalos<<<grid, 256, 0, stream>>>(d_fMom[g], macroInterfaceGPU[g], layout.localNZ);
        CHECK_KERNEL_ERR("Pack macro halos kernel");
    }

    // Each sender pushes its top face and pulls the next slab's bottom face.
    /** @brief Exchanges one macroscopic field across neighboring GPU slabs. */
    void exchangeMacroField(int g, deviceField* allDevices,
        gpuDirection macroInterfaceGPUData::*field, cudaStream_t stream, bool stepParity) {
        checkCudaErrors(cudaSetDevice(GPUS_TO_USE[g]));
        #ifdef BC_Z_WALL
            if (g == N_GPUS - 1) return;
        #endif // BC_Z_WALL
        const int next = (g + 1) % N_GPUS;

        const gpuDirection ownSrc      = macroInterfaceGPU[g].*field;
        const gpuDirection neighborSrc = allDevices[next].macroInterfaceGPU[next].*field;

        macroInterfaceGPUData& ownDstIface = stepParity
            ? macroInterfaceGPU[g] : macroInterfaceGPUCopy[g];
        macroInterfaceGPUData& neighborDstIface = stepParity
            ? allDevices[next].macroInterfaceGPU[next] : allDevices[next].macroInterfaceGPUCopy[next];

        const gpuDirection ownDst      = ownDstIface.*field;
        const gpuDirection neighborDst = neighborDstIface.*field;

        const size_t bytes = (size_t)NX * NY * sizeof(dfloat);
        checkCudaErrors(cudaMemcpyPeerAsync(neighborDst.auxZ_1, GPUS_TO_USE[next],ownSrc.Z_1, GPUS_TO_USE[g], bytes, stream));
        checkCudaErrors(cudaMemcpyPeerAsync(ownDst.auxZ_0, GPUS_TO_USE[g],neighborSrc.Z_0, GPUS_TO_USE[next], bytes, stream));
    }

    /** @brief Sends the top macroscopic halo from GPU @p g to its successor. */
    void sendMacroTopToNext(int g, deviceField* allDevices, cudaStream_t streamLBM, bool stepParity)
    {
        checkCudaErrors(cudaSetDevice(GPUS_TO_USE[g]));
        #ifdef BC_Z_WALL
            if (g == N_GPUS - 1) return;
        #endif // BC_Z_WALL

        const int gNext = (g + 1) % N_GPUS;
        const size_t MacroHaloSize = NX * NY * sizeof(dfloat);

        auto& dstIface = stepParity ? allDevices[gNext].macroInterfaceGPU[gNext] : allDevices[gNext].macroInterfaceGPUCopy[gNext];

        checkCudaErrors(cudaMemcpyPeerAsync(
            dstIface.rho.auxZ_1, GPUS_TO_USE[gNext],
            allDevices[g].macroInterfaceGPU[g].rho.Z_1, GPUS_TO_USE[g],
            MacroHaloSize, streamLBM
        ));

        checkCudaErrors(cudaMemcpyPeerAsync(
            dstIface.ux.auxZ_1, GPUS_TO_USE[gNext],
            allDevices[g].macroInterfaceGPU[g].ux.Z_1, GPUS_TO_USE[g],
            MacroHaloSize, streamLBM
        ));

        checkCudaErrors(cudaMemcpyPeerAsync(
            dstIface.uy.auxZ_1, GPUS_TO_USE[gNext],
            allDevices[g].macroInterfaceGPU[g].uy.Z_1, GPUS_TO_USE[g],
            MacroHaloSize, streamLBM
        ));

        checkCudaErrors(cudaMemcpyPeerAsync(
            dstIface.uz.auxZ_1, GPUS_TO_USE[gNext],
            allDevices[g].macroInterfaceGPU[g].uz.Z_1, GPUS_TO_USE[g],
            MacroHaloSize, streamLBM
        ));

        #ifdef SECOND_DIST
            checkCudaErrors(cudaMemcpyPeerAsync(
            dstIface.g.auxZ_1, GPUS_TO_USE[gNext],
            allDevices[g].macroInterfaceGPU[g].g.Z_1, GPUS_TO_USE[g],
            MacroHaloSize, streamLBM
            ));
        #endif //SECOND_DIST
        #ifdef LAMBDA_DIST
            checkCudaErrors(cudaMemcpyPeerAsync(
            dstIface.lambda.auxZ_1, GPUS_TO_USE[gNext],
            allDevices[g].macroInterfaceGPU[g].lambda.Z_1, GPUS_TO_USE[g],
            MacroHaloSize, streamLBM
            ));
        #endif //LAMBDA_DIST
        #ifdef A_XX_DIST
            checkCudaErrors(cudaMemcpyPeerAsync(
            dstIface.Axx.auxZ_1, GPUS_TO_USE[gNext],
            allDevices[g].macroInterfaceGPU[g].Axx.Z_1, GPUS_TO_USE[g],
            MacroHaloSize, streamLBM
            ));
        #endif //A_XX_DIST
        #ifdef A_XY_DIST
            checkCudaErrors(cudaMemcpyPeerAsync(
            dstIface.Axy.auxZ_1, GPUS_TO_USE[gNext],
            allDevices[g].macroInterfaceGPU[g].Axy.Z_1, GPUS_TO_USE[g],
            MacroHaloSize, streamLBM
            ));
        #endif //A_XY_DIST
        #ifdef A_XZ_DIST
            checkCudaErrors(cudaMemcpyPeerAsync(
            dstIface.Axz.auxZ_1, GPUS_TO_USE[gNext],
            allDevices[g].macroInterfaceGPU[g].Axz.Z_1, GPUS_TO_USE[g],
            MacroHaloSize, streamLBM
            ));
        #endif //A_XZ_DIST
        #ifdef A_YY_DIST
            checkCudaErrors(cudaMemcpyPeerAsync(
            dstIface.Ayy.auxZ_1, GPUS_TO_USE[gNext],
            allDevices[g].macroInterfaceGPU[g].Ayy.Z_1, GPUS_TO_USE[g],
            MacroHaloSize, streamLBM
            ));
        #endif //A_YY_DIST
        #ifdef A_YZ_DIST
            checkCudaErrors(cudaMemcpyPeerAsync(
            dstIface.Ayz.auxZ_1, GPUS_TO_USE[gNext],
            allDevices[g].macroInterfaceGPU[g].Ayz.Z_1, GPUS_TO_USE[g],
            MacroHaloSize, streamLBM
            ));
        #endif //A_YZ_DIST
        #ifdef A_ZZ_DIST
            checkCudaErrors(cudaMemcpyPeerAsync(
            dstIface.Azz.auxZ_1, GPUS_TO_USE[gNext],
            allDevices[g].macroInterfaceGPU[g].Azz.Z_1, GPUS_TO_USE[g],
            MacroHaloSize, streamLBM
            ));
        #endif //A_ZZ_DIST

        #ifdef PHI_DIST
            checkCudaErrors(cudaMemcpyPeerAsync(
            dstIface.phi.auxZ_1, GPUS_TO_USE[gNext],
            allDevices[g].macroInterfaceGPU[g].phi.Z_1, GPUS_TO_USE[g],
            MacroHaloSize, streamLBM
            ));

            checkCudaErrors(cudaMemcpyPeerAsync(
            dstIface.nx.auxZ_1, GPUS_TO_USE[gNext],
            allDevices[g].macroInterfaceGPU[g].nx.Z_1, GPUS_TO_USE[g],
            MacroHaloSize, streamLBM
            ));

            checkCudaErrors(cudaMemcpyPeerAsync(
            dstIface.ny.auxZ_1, GPUS_TO_USE[gNext],
            allDevices[g].macroInterfaceGPU[g].ny.Z_1, GPUS_TO_USE[g],
            MacroHaloSize, streamLBM
            ));

            checkCudaErrors(cudaMemcpyPeerAsync(
            dstIface.nz.auxZ_1, GPUS_TO_USE[gNext],
            allDevices[g].macroInterfaceGPU[g].nz.Z_1, GPUS_TO_USE[g],
            MacroHaloSize, streamLBM
            ));

        #ifdef PHASE_NEEDS_CHEMICAL_POTENTIAL
                checkCudaErrors(cudaMemcpyPeerAsync(
                dstIface.mu.auxZ_1, GPUS_TO_USE[gNext],
                allDevices[g].macroInterfaceGPU[g].mu.Z_1, GPUS_TO_USE[g],
                MacroHaloSize, streamLBM
                ));
        #endif // PHASE_NEEDS_CHEMICAL_POTENTIAL
        #endif //PHI_DIST
    }

    /** @brief Sends the bottom macroscopic halo from GPU @p g to its predecessor. */
    void sendMacroBottomToPrev(int g, deviceField* allDevices, cudaStream_t streamLBM, bool stepParity)
    {
        checkCudaErrors(cudaSetDevice(GPUS_TO_USE[g]));
        const int gNext = (g + 1) % N_GPUS;
        const size_t MacroHaloSize = NX * NY * sizeof(dfloat);

        #ifdef BC_Z_WALL
            if (g == N_GPUS - 1) return;
        #endif // BC_Z_WALL

        auto& dstIface = stepParity ? allDevices[g].macroInterfaceGPU[g]
                                    : allDevices[g].macroInterfaceGPUCopy[g];

        checkCudaErrors(cudaMemcpyPeerAsync(
            dstIface.rho.auxZ_0, GPUS_TO_USE[g],
            allDevices[gNext].macroInterfaceGPU[gNext].rho.Z_0, GPUS_TO_USE[gNext],
            MacroHaloSize, streamLBM
        ));

        checkCudaErrors(cudaMemcpyPeerAsync(
            dstIface.ux.auxZ_0, GPUS_TO_USE[g],
            allDevices[gNext].macroInterfaceGPU[gNext].ux.Z_0, GPUS_TO_USE[gNext],
            MacroHaloSize, streamLBM
        ));

        checkCudaErrors(cudaMemcpyPeerAsync(
            dstIface.uy.auxZ_0, GPUS_TO_USE[g],
            allDevices[gNext].macroInterfaceGPU[gNext].uy.Z_0, GPUS_TO_USE[gNext],
            MacroHaloSize, streamLBM
        ));

        checkCudaErrors(cudaMemcpyPeerAsync(
            dstIface.uz.auxZ_0, GPUS_TO_USE[g],
            allDevices[gNext].macroInterfaceGPU[gNext].uz.Z_0, GPUS_TO_USE[gNext],
            MacroHaloSize, streamLBM
        ));

        #ifdef SECOND_DIST
            checkCudaErrors(cudaMemcpyPeerAsync(
            dstIface.g.auxZ_0, GPUS_TO_USE[g],
            allDevices[gNext].macroInterfaceGPU[gNext].g.Z_0, GPUS_TO_USE[gNext],
            MacroHaloSize, streamLBM
            ));
        #endif //SECOND_DIST
        #ifdef LAMBDA_DIST
            checkCudaErrors(cudaMemcpyPeerAsync(
            dstIface.lambda.auxZ_0, GPUS_TO_USE[g],
            allDevices[gNext].macroInterfaceGPU[gNext].lambda.Z_0, GPUS_TO_USE[gNext],
            MacroHaloSize, streamLBM
            ));
        #endif //LAMBDA_DIST
        #ifdef A_XX_DIST
            checkCudaErrors(cudaMemcpyPeerAsync(
            dstIface.Axx.auxZ_0, GPUS_TO_USE[g],
            allDevices[gNext].macroInterfaceGPU[gNext].Axx.Z_0, GPUS_TO_USE[gNext],
            MacroHaloSize, streamLBM
            ));
        #endif //A_XX_DIST
        #ifdef A_XY_DIST
            checkCudaErrors(cudaMemcpyPeerAsync(
            dstIface.Axy.auxZ_0, GPUS_TO_USE[g],
            allDevices[gNext].macroInterfaceGPU[gNext].Axy.Z_0, GPUS_TO_USE[gNext],
            MacroHaloSize, streamLBM
            ));
        #endif //A_XY_DIST
        #ifdef A_XZ_DIST
            checkCudaErrors(cudaMemcpyPeerAsync(
            dstIface.Axz.auxZ_0, GPUS_TO_USE[g],
            allDevices[gNext].macroInterfaceGPU[gNext].Axz.Z_0, GPUS_TO_USE[gNext],
            MacroHaloSize, streamLBM
            ));
        #endif //A_XZ_DIST
        #ifdef A_YY_DIST
            checkCudaErrors(cudaMemcpyPeerAsync(
            dstIface.Ayy.auxZ_0, GPUS_TO_USE[g],
            allDevices[gNext].macroInterfaceGPU[gNext].Ayy.Z_0, GPUS_TO_USE[gNext],
            MacroHaloSize, streamLBM
            ));
        #endif //A_YY_DIST
        #ifdef A_YZ_DIST
            checkCudaErrors(cudaMemcpyPeerAsync(
            dstIface.Ayz.auxZ_0, GPUS_TO_USE[g],
            allDevices[gNext].macroInterfaceGPU[gNext].Ayz.Z_0, GPUS_TO_USE[gNext],
            MacroHaloSize, streamLBM
            ));
        #endif //A_YZ_DIST
        #ifdef A_ZZ_DIST
            checkCudaErrors(cudaMemcpyPeerAsync(
            dstIface.Azz.auxZ_0, GPUS_TO_USE[g],
            allDevices[gNext].macroInterfaceGPU[gNext].Azz.Z_0, GPUS_TO_USE[gNext],
            MacroHaloSize, streamLBM
            ));
        #endif //A_ZZ_DIST

        #ifdef PHI_DIST
            checkCudaErrors(cudaMemcpyPeerAsync(
            dstIface.phi.auxZ_0, GPUS_TO_USE[g],
            allDevices[gNext].macroInterfaceGPU[gNext].phi.Z_0, GPUS_TO_USE[gNext],
            MacroHaloSize, streamLBM
            ));
            checkCudaErrors(cudaMemcpyPeerAsync(
            dstIface.nx.auxZ_0, GPUS_TO_USE[g],
            allDevices[gNext].macroInterfaceGPU[gNext].nx.Z_0, GPUS_TO_USE[gNext],
            MacroHaloSize, streamLBM
            ));
            checkCudaErrors(cudaMemcpyPeerAsync(
            dstIface.ny.auxZ_0, GPUS_TO_USE[g],
            allDevices[gNext].macroInterfaceGPU[gNext].ny.Z_0, GPUS_TO_USE[gNext],
            MacroHaloSize, streamLBM
            ));
            checkCudaErrors(cudaMemcpyPeerAsync(
            dstIface.nz.auxZ_0, GPUS_TO_USE[g],
            allDevices[gNext].macroInterfaceGPU[gNext].nz.Z_0, GPUS_TO_USE[gNext],
            MacroHaloSize, streamLBM
            ));
        #ifdef PHASE_NEEDS_CHEMICAL_POTENTIAL
                checkCudaErrors(cudaMemcpyPeerAsync(
                dstIface.mu.auxZ_0, GPUS_TO_USE[g],
                allDevices[gNext].macroInterfaceGPU[gNext].mu.Z_0, GPUS_TO_USE[gNext],
                MacroHaloSize, streamLBM
                ));
        #endif // PHASE_NEEDS_CHEMICAL_POTENTIAL
        #endif //PHI_DIST
    }

    /* -------------- Send the current GPU's Z_1 top to the nearest GPU's Z_1 base   ------------- */
    /** @brief Sends population interfaces from GPU @p g to its successor. */
    void sendTopToNext(int g, deviceField* allDevices, cudaStream_t streamLBM, bool stepParity)
    {
        checkCudaErrors(cudaSetDevice(GPUS_TO_USE[g]));
        const int gNext = (g + 1) % N_GPUS;
        const size_t planeSize = layout.exchangePlaneCount * QF;
        const size_t haloSize  = planeSize * sizeof(dfloat);
        const size_t topOffset = (allDevices[g].layout.numBlocksZ - 1) * planeSize;

        #ifdef STREAMING_AB
            auto& dstIface = allDevices[gNext].ghostInterface[gNext];
        #else
            auto& dstIface = stepParity ? allDevices[gNext].ghostInterface[gNext]
                             : allDevices[gNext].ghostInterfaceCopy[gNext];
        #endif // STREAMING_AB

        checkCudaErrors(cudaMemcpyPeerAsync(
            dstIface.popAux.Z_1, GPUS_TO_USE[gNext],
            allDevices[g].ghostInterface[g].pop.Z_1 + topOffset, GPUS_TO_USE[g],
            haloSize, streamLBM
        ));

        #if defined(SECOND_DIST) || defined(PHI_DIST) || defined(LAMBDA_DIST) || \
            defined(A_XX_DIST) || defined(A_XY_DIST) || defined(A_XZ_DIST) || \
            defined(A_YY_DIST) || defined(A_YZ_DIST) || defined(A_ZZ_DIST)
            const size_t scalarPlaneSize = layout.exchangePlaneCount * GF;
            const size_t scalarHaloSize = scalarPlaneSize * sizeof(dfloat);
            const size_t scalarTopOffset = (allDevices[g].layout.numBlocksZ - 1) * scalarPlaneSize;
        #endif // defined(SECOND_DIST) || defined(PHI_DIST) || defined(LAMBDA_DIST) ||

        #ifdef SECOND_DIST
            checkCudaErrors(cudaMemcpyPeerAsync(
            dstIface.gAux.Z_1, GPUS_TO_USE[gNext],
            allDevices[g].ghostInterface[g].g.Z_1 + scalarTopOffset, GPUS_TO_USE[g],
            scalarHaloSize, streamLBM
            ));
        #endif //SECOND_DIST

        #ifdef PHI_DIST
            checkCudaErrors(cudaMemcpyPeerAsync(
            dstIface.phiAux.Z_1, GPUS_TO_USE[gNext],
            allDevices[g].ghostInterface[g].phi.Z_1 + scalarTopOffset, GPUS_TO_USE[g],
            scalarHaloSize, streamLBM
            ));
        #endif //PHI_DIST

        #ifdef LAMBDA_DIST
            checkCudaErrors(cudaMemcpyPeerAsync(
            dstIface.lambdaAux.Z_1, GPUS_TO_USE[gNext],
            allDevices[g].ghostInterface[g].lambda.Z_1 + scalarTopOffset, GPUS_TO_USE[g],
            scalarHaloSize, streamLBM
            ));
        #endif //LAMBDA_DIST

        #ifdef A_XX_DIST
            checkCudaErrors(cudaMemcpyPeerAsync(
            dstIface.AxxAux.Z_1, GPUS_TO_USE[gNext],
            allDevices[g].ghostInterface[g].Axx.Z_1 + scalarTopOffset, GPUS_TO_USE[g],
            scalarHaloSize, streamLBM
            ));
        #endif //A_XX_DIST

        #ifdef A_XY_DIST
            checkCudaErrors(cudaMemcpyPeerAsync(
            dstIface.AxyAux.Z_1, GPUS_TO_USE[gNext],
            allDevices[g].ghostInterface[g].Axy.Z_1 + scalarTopOffset, GPUS_TO_USE[g],
            scalarHaloSize, streamLBM
            ));
        #endif //A_XY_DIST

        #ifdef A_XZ_DIST
            checkCudaErrors(cudaMemcpyPeerAsync(
            dstIface.AxzAux.Z_1, GPUS_TO_USE[gNext],
            allDevices[g].ghostInterface[g].Axz.Z_1 + scalarTopOffset, GPUS_TO_USE[g],
            scalarHaloSize, streamLBM
            ));
        #endif //A_XZ_DIST

        #ifdef A_YY_DIST
            checkCudaErrors(cudaMemcpyPeerAsync(
            dstIface.AyyAux.Z_1, GPUS_TO_USE[gNext],
            allDevices[g].ghostInterface[g].Ayy.Z_1 + scalarTopOffset, GPUS_TO_USE[g],
            scalarHaloSize, streamLBM
            ));
        #endif //A_YY_DIST

        #ifdef A_YZ_DIST
            checkCudaErrors(cudaMemcpyPeerAsync(
            dstIface.AyzAux.Z_1, GPUS_TO_USE[gNext],
            allDevices[g].ghostInterface[g].Ayz.Z_1 + scalarTopOffset, GPUS_TO_USE[g],
            scalarHaloSize, streamLBM
            ));
        #endif //A_YZ_DIST

        #ifdef A_ZZ_DIST
            checkCudaErrors(cudaMemcpyPeerAsync(
            dstIface.AzzAux.Z_1, GPUS_TO_USE[gNext],
            allDevices[g].ghostInterface[g].Azz.Z_1 + scalarTopOffset, GPUS_TO_USE[g],
            scalarHaloSize, streamLBM
            ));
        #endif //A_ZZ_DIST
    }

    /* -------------- Receives the Z_0 base of the next GPU and adds it to the Z_0 top of the current GPU  ------------- */
    /** @brief Sends population interfaces from GPU @p g to its predecessor. */
    void sendBottomToPrev(int g, deviceField* allDevices, cudaStream_t streamLBM,  bool stepParity)
    {
        checkCudaErrors(cudaSetDevice(GPUS_TO_USE[g]));
        const int gNext = (g + 1) % N_GPUS;
        const size_t planeSize = layout.exchangePlaneCount * QF;
        const size_t haloSize  = planeSize * sizeof(dfloat);

        #ifdef STREAMING_AB
            auto& dstIface = allDevices[g].ghostInterface[g];
        #else
            auto& dstIface = stepParity ? allDevices[g].ghostInterface[g]
                             : allDevices[g].ghostInterfaceCopy[g];
        #endif // STREAMING_AB

        checkCudaErrors(cudaMemcpyPeerAsync(
            dstIface.popAux.Z_0, GPUS_TO_USE[g],
            allDevices[gNext].ghostInterface[gNext].pop.Z_0, GPUS_TO_USE[gNext],
            haloSize, streamLBM
        ));

        #if defined(SECOND_DIST) || defined(PHI_DIST) || defined(LAMBDA_DIST) || \
            defined(A_XX_DIST) || defined(A_XY_DIST) || defined(A_XZ_DIST) || \
            defined(A_YY_DIST) || defined(A_YZ_DIST) || defined(A_ZZ_DIST)
            const size_t scalarPlaneSize = layout.exchangePlaneCount * GF;
            const size_t scalarHaloSize = scalarPlaneSize * sizeof(dfloat);
        #endif // defined(SECOND_DIST) || defined(PHI_DIST) || defined(LAMBDA_DIST) ||

        #ifdef SECOND_DIST
            checkCudaErrors(cudaMemcpyPeerAsync(
            dstIface.gAux.Z_0, GPUS_TO_USE[g],
            allDevices[gNext].ghostInterface[gNext].g.Z_0, GPUS_TO_USE[gNext],
            scalarHaloSize, streamLBM
            ));
        #endif //SECOND_DIST

        #ifdef PHI_DIST
            checkCudaErrors(cudaMemcpyPeerAsync(
            dstIface.phiAux.Z_0, GPUS_TO_USE[g],
            allDevices[gNext].ghostInterface[gNext].phi.Z_0, GPUS_TO_USE[gNext],
            scalarHaloSize, streamLBM
            ));
        #endif //PHI_DIST

        #ifdef LAMBDA_DIST
            checkCudaErrors(cudaMemcpyPeerAsync(
            dstIface.lambdaAux.Z_0, GPUS_TO_USE[g],
            allDevices[gNext].ghostInterface[gNext].lambda.Z_0, GPUS_TO_USE[gNext],
            scalarHaloSize, streamLBM
            ));
        #endif //LAMBDA_DIST

        #ifdef A_XX_DIST
            checkCudaErrors(cudaMemcpyPeerAsync(
            dstIface.AxxAux.Z_0, GPUS_TO_USE[g],
            allDevices[gNext].ghostInterface[gNext].Axx.Z_0, GPUS_TO_USE[gNext],
            scalarHaloSize, streamLBM
            ));
        #endif //A_XX_DIST

        #ifdef A_XY_DIST
            checkCudaErrors(cudaMemcpyPeerAsync(
            dstIface.AxyAux.Z_0, GPUS_TO_USE[g],
            allDevices[gNext].ghostInterface[gNext].Axy.Z_0, GPUS_TO_USE[gNext],
            scalarHaloSize, streamLBM
            ));
        #endif //A_XY_DIST

        #ifdef A_XZ_DIST
            checkCudaErrors(cudaMemcpyPeerAsync(
            dstIface.AxzAux.Z_0, GPUS_TO_USE[g],
            allDevices[gNext].ghostInterface[gNext].Axz.Z_0, GPUS_TO_USE[gNext],
            scalarHaloSize, streamLBM
            ));
        #endif //A_XZ_DIST

        #ifdef A_YY_DIST
            checkCudaErrors(cudaMemcpyPeerAsync(
            dstIface.AyyAux.Z_0, GPUS_TO_USE[g],
            allDevices[gNext].ghostInterface[gNext].Ayy.Z_0, GPUS_TO_USE[gNext],
            scalarHaloSize, streamLBM
            ));
        #endif //A_YY_DIST

        #ifdef A_YZ_DIST
            checkCudaErrors(cudaMemcpyPeerAsync(
            dstIface.AyzAux.Z_0, GPUS_TO_USE[g],
            allDevices[gNext].ghostInterface[gNext].Ayz.Z_0, GPUS_TO_USE[gNext],
            scalarHaloSize, streamLBM
            ));
        #endif //A_YZ_DIST

        #ifdef A_ZZ_DIST
            checkCudaErrors(cudaMemcpyPeerAsync(
            dstIface.AzzAux.Z_0, GPUS_TO_USE[g],
            allDevices[gNext].ghostInterface[gNext].Azz.Z_0, GPUS_TO_USE[gNext],
            scalarHaloSize, streamLBM
            ));
        #endif //A_ZZ_DIST
    }

    #ifdef PHI_DIST
    // Caller must exchange fresh PHI before this stage and publish the
    // resulting gradients/chemical potential before auxiliary derivatives.
    /** @brief Computes phase normals for GPU @p g. */
        void computePhaseNormalsDeviceField(dim3 threadBlock, int g, cudaStream_t stream, bool stepParity){
            checkCudaErrors(cudaSetDevice(GPUS_TO_USE[g]));
            const dim3 gridBlock = layout.fullGrid;
            const int zStart = static_cast<int>(layout.zStart);
            const size_t localNZ = layout.localNZ;
            macroInterfaceGPUData& macroIF = stepParity ? macroInterfaceGPU[g] : macroInterfaceGPUCopy[g];

            gpuComputePhaseNormals<<<gridBlock, threadBlock, 0, stream>>>(d_fMom[g], dNodeType[g], macroIF, localNZ, zStart);
        #ifdef PHASE_NEEDS_CHEMICAL_POTENTIAL
                gpuComputeChemicalPotential<<<gridBlock, threadBlock, 0, stream>>>(d_fMom[g], dNodeType[g], localNZ, zStart);
        #endif // PHASE_NEEDS_CHEMICAL_POTENTIAL
            }

    /** @brief Computes phase-transport auxiliary fields for GPU @p g. */
        void computePhaseAuxiliariesDeviceField(dim3 threadBlock, int g, cudaStream_t stream, bool stepParity){
            checkCudaErrors(cudaSetDevice(GPUS_TO_USE[g]));
            const dim3 gridBlock = layout.fullGrid;
            const int zStart = static_cast<int>(layout.zStart);
            const size_t localNZ = layout.localNZ;
            macroInterfaceGPUData& macroIF = stepParity ? macroInterfaceGPU[g] : macroInterfaceGPUCopy[g];

        #ifdef PHASE_TRANSPORT_CAHN_HILLIARD
                gpuComputeLaplacianMu<<<gridBlock, threadBlock, 0, stream>>>(d_fMom[g], dNodeType[g], macroIF, localNZ, zStart);
        #endif // PHASE_TRANSPORT_CAHN_HILLIARD
        #ifdef PHASE_NEEDS_NORMAL_HALOS
                gpuComputePhaseFluxDivergences<<<gridBlock, threadBlock, 0, stream>>>(d_fMom[g], dNodeType[g], macroIF, localNZ, zStart);
        #endif // PHASE_NEEDS_NORMAL_HALOS
            }

    #endif //PHI_DIST

    /** @brief Swaps the active and alternate ghost-interface generations. */
    void swapGhostInterfacesDeviceField(int g){
        swapGhostInterfaces(ghostInterface[g], ghostInterfaceCopy[g]);
    }

    /** @brief Runs half-step kernels required by the selected physics model. */
    void halfStepKernels(dim3 threadBlock, size_t step, int g, cudaStream_t stream){
        checkCudaErrors(cudaSetDevice(GPUS_TO_USE[g]));
        const dim3 gridBlock = layout.fullGrid;
        #ifdef LOCAL_FORCES
            gpuResetMacroForcesDeviceField(gridBlock, threadBlock, g, stream);
            CHECK_KERNEL_ERR("Force Reset kernel");
        #endif //LOCAL_FORCES
        #ifdef CURVED_BOUNDARY_CONDITION
            updateCurvedBoundaryVelocitiesDeviceField(g, stream);
            CHECK_KERNEL_ERR("Curved BC kernel");
        #if defined(CONFORMATION_TENSOR) && defined(D3G19)
                updateCurvedBoundaryConformationDeviceField(g, stream);
                CHECK_KERNEL_ERR("Curved conformation BC kernel");
        #endif // defined(CONFORMATION_TENSOR) && defined(D3G19)
        #endif //CURVED_BOUNDARY_CONDITION
    }

    #ifdef LOCAL_FORCES
    /** @brief Resets macroscopic forces on GPU @p g. */
        void gpuResetMacroForcesDeviceField(dim3 gridBlock, dim3 threadBlock, int g, cudaStream_t stream){
            gpuResetMacroForces<<<gridBlock, threadBlock, 0, stream>>>(d_fMom[g]);
            }
    #endif //LOCAL_FORCES

    /** @brief Copies ghost-interface data between host and GPU @p g. */
    void interfaceCudaMemcpyDeviceField(bool ghost, int g){
        // AA layout: always copy from ghost (single buffer)
        interfaceCudaMemcpy(ghostInterface[g], ghostInterface[g].h_pop, ghostInterface[g].pop, cudaMemcpyDeviceToHost, QF, g);
        #ifdef SECOND_DIST
            interfaceCudaMemcpy(ghostInterface[g],ghostInterface[g].h_g,ghostInterface[g].g,cudaMemcpyDeviceToHost,GF, g);
        #endif //SECOND_DIST
        #ifdef PHI_DIST
            interfaceCudaMemcpy(ghostInterface[g],ghostInterface[g].h_phi,ghostInterface[g].phi,cudaMemcpyDeviceToHost,GF, g);
        #endif //PHI_DIST
        #ifdef LAMBDA_DIST
            interfaceCudaMemcpy(ghostInterface[g],ghostInterface[g].h_lambda,ghostInterface[g].lambda,cudaMemcpyDeviceToHost,GF, g);
        #endif //LAMBDA_DIST
        #ifdef A_XX_DIST
            interfaceCudaMemcpy(ghostInterface[g],ghostInterface[g].h_Axx,ghostInterface[g].Axx,cudaMemcpyDeviceToHost,GF, g);
        #endif //A_XX_DIST
        #ifdef A_XY_DIST
            interfaceCudaMemcpy(ghostInterface[g],ghostInterface[g].h_Axy,ghostInterface[g].Axy,cudaMemcpyDeviceToHost,GF, g);
        #endif //A_XX_DIST
        #ifdef A_XZ_DIST
            interfaceCudaMemcpy(ghostInterface[g],ghostInterface[g].h_Axz,ghostInterface[g].Axz,cudaMemcpyDeviceToHost,GF, g);
        #endif //A_XZ_DIST
        #ifdef A_YY_DIST
            interfaceCudaMemcpy(ghostInterface[g],ghostInterface[g].h_Ayy,ghostInterface[g].Ayy,cudaMemcpyDeviceToHost,GF, g);
        #endif //A_YY_DIST
        #ifdef A_YZ_DIST
            interfaceCudaMemcpy(ghostInterface[g],ghostInterface[g].h_Ayz,ghostInterface[g].Ayz,cudaMemcpyDeviceToHost,GF, g);
        #endif //A_YZ_DIST
        #ifdef A_ZZ_DIST
            interfaceCudaMemcpy(ghostInterface[g],ghostInterface[g].h_Azz,ghostInterface[g].Azz,cudaMemcpyDeviceToHost,GF, g);
        #endif //A_ZZ_DIST
    }

    /** @brief Captures GPU @p g populations into canonical checkpoint order. */
    void captureCanonicalCheckpointDeviceField(
        CanonicalCheckpointHostState& state, dim3 threadBlock,
        unsigned int checkpointStep, int g)
    {
        checkCudaErrors(cudaSetDevice(GPUS_TO_USE[g]));
        captureCanonicalCheckpointPopulations(
            d_fMom[g], ghostInterface[g], ghostInterfaceCopy[g], layout,
            threadBlock, effectiveStreamingParity(checkpointStep), state);
    }

    /** @brief Returns the population parity after the configured streaming offset. */
    bool effectiveStreamingParity(unsigned int step) const
    {
        return ((step & 1u) ^ streamingParityOffset) != 0;
    }

    /** @brief Collects all GPU macroscopic slabs into the global host snapshot. */
    void copyMacroscopicDeviceField(hostField &hostField, deviceField* allDevices){
        std::vector<std::thread> copyThreads;
        copyThreads.reserve(N_GPUS);

        for(int g = 0; g < N_GPUS; g++){
            copyThreads.emplace_back([&, g]() {
                checkCudaErrors(cudaSetDevice(GPUS_TO_USE[g]));
                checkCudaErrors(cudaDeviceSynchronize());
                const GpuDomainLayout& local = allDevices[g].layout;
                std::vector<dfloat> localMom(local.paddedNodeCount * NUMBER_MOMENTS);
                checkCudaErrors(cudaMemcpy(localMom.data(), allDevices[g].d_fMom[g],
                    local.momentStorageBytes, cudaMemcpyDeviceToHost));

                for (size_t zLocal = 0; zLocal < local.localNZ; ++zLocal) {
                    const size_t zGlobal = local.zStart + zLocal;
                    for (size_t y = 0; y < static_cast<size_t>(NY); ++y) {
                        for (size_t x = 0; x < static_cast<size_t>(NX); ++x) {
                            for (int moment = 0; moment < NUMBER_MOMENTS; ++moment) {
                                const size_t src = idxMom(x % BLOCK_NX, y % BLOCK_NY, zLocal % BLOCK_NZ, moment, x / BLOCK_NX, y / BLOCK_NY, zLocal / BLOCK_NZ);
                                const size_t dst = idxMom(x % BLOCK_NX, y % BLOCK_NY, zGlobal % BLOCK_NZ, moment, x / BLOCK_NX, y / BLOCK_NY, zGlobal / BLOCK_NZ);
                                hostField.h_fMom[dst] = localMom[src];
                            }
                        }
                    }
                }

                // Each worker writes only its own Z slab in the pinned global
                // snapshot, so transfers from different devices can overlap.
                auto gatherScalar = [&](dfloat* host, const dfloat* device) {
                    std::vector<dfloat> values(local.paddedNodeCount);
                    checkCudaErrors(cudaMemcpy(values.data(), device, local.scalarBytes, cudaMemcpyDeviceToHost));
                    for (size_t zLocal = 0; zLocal < local.localNZ; ++zLocal)
                        for (size_t y = 0; y < static_cast<size_t>(NY); ++y)
                            for (size_t x = 0; x < static_cast<size_t>(NX); ++x) {
                                const size_t src = idxScalarBlock(x % BLOCK_NX, y % BLOCK_NY, zLocal % BLOCK_NZ, x / BLOCK_NX, y / BLOCK_NY, zLocal / BLOCK_NZ);
                                const size_t zGlobal = local.zStart + zLocal;
                                const size_t dst = idxScalarBlock(x % BLOCK_NX, y % BLOCK_NY, zGlobal % BLOCK_NZ, x / BLOCK_NX, y / BLOCK_NY, zGlobal / BLOCK_NZ);
                                host[dst] = values[src];
                            }
                };
                #if defined(BC_FORCES) && defined(SAVE_BC_FORCES)
                    gatherScalar(hostField.h_BC_Fx, allDevices[g].d_BC_Fx[g]);
                    gatherScalar(hostField.h_BC_Fy, allDevices[g].d_BC_Fy[g]);
                    gatherScalar(hostField.h_BC_Fz, allDevices[g].d_BC_Fz[g]);
                #endif // defined(BC_FORCES) && defined(SAVE_BC_FORCES)
                #ifdef SAVE_LOCAL_FORCES
                    gatherScalar(hostField.h_Local_Fx, allDevices[g].d_Local_Fx);
                    gatherScalar(hostField.h_Local_Fy, allDevices[g].d_Local_Fy);
                    gatherScalar(hostField.h_Local_Fz, allDevices[g].d_Local_Fz);
                #ifdef SECOND_DIST
                        gatherScalar(hostField.h_Source_C, allDevices[g].d_Source_C);
                #endif // SECOND_DIST
                #ifdef PHI_DIST
                        gatherScalar(hostField.h_Source_Phi, allDevices[g].d_Source_Phi);
                #endif // PHI_DIST
                #ifdef LAMBDA_DIST
                        gatherScalar(hostField.h_Source_Lambda, allDevices[g].d_Source_Lambda);
                #endif // LAMBDA_DIST
                #ifdef CONFORMATION_TENSOR
                #ifdef A_XX_DIST
                            gatherScalar(hostField.h_Source_Gxx, allDevices[g].d_Source_Gxx);
                #endif // A_XX_DIST
                #ifdef A_XY_DIST
                            gatherScalar(hostField.h_Source_Gxy, allDevices[g].d_Source_Gxy);
                #endif // A_XY_DIST
                #ifdef A_XZ_DIST
                            gatherScalar(hostField.h_Source_Gxz, allDevices[g].d_Source_Gxz);
                #endif // A_XZ_DIST
                #ifdef A_YY_DIST
                            gatherScalar(hostField.h_Source_Gyy, allDevices[g].d_Source_Gyy);
                #endif // A_YY_DIST
                #ifdef A_YZ_DIST
                            gatherScalar(hostField.h_Source_Gyz, allDevices[g].d_Source_Gyz);
                #endif // A_YZ_DIST
                #ifdef A_ZZ_DIST
                            gatherScalar(hostField.h_Source_Gzz, allDevices[g].d_Source_Gzz);
                #endif // A_ZZ_DIST
                #endif //CONFORMATION_TENSOR
                #endif //SAVE_LOCAL_FORCES

                checkCudaErrors(cudaDeviceSynchronize());
            });
        }

        for (auto &copyThread : copyThreads) {
            copyThread.join();
        }
    }

    /** @brief Executes post-processing using the global host snapshot. */
    void treatDataDeviceField(hostField &hostField, int step){
        TreatDataParams treatDataParams{};
        treatDataParams.h_fMom = hostField.h_fMom;
        treatDataParams.h_nodeType = hostField.hNodeType;
        #if MEAN_FLOW
            treatDataParams.d_fMom_mean = hostField.m_fMom;
        #endif // MEAN_FLOW
        #ifdef BC_FORCES
            treatDataParams.d_BC_Fx = nullptr;
            treatDataParams.d_BC_Fy = nullptr;
            treatDataParams.d_BC_Fz = nullptr;
        #endif // BC_FORCES
        #ifdef SAVE_LOCAL_FORCES
            treatDataParams.d_Local_Fx = hostField.h_Local_Fx;
            treatDataParams.d_Local_Fy = hostField.h_Local_Fy;
            treatDataParams.d_Local_Fz = hostField.h_Local_Fz;
        #ifdef SECOND_DIST
                treatDataParams.d_Source_C = hostField.h_Source_C;
        #endif // SECOND_DIST
        #ifdef PHI_DIST
                treatDataParams.d_Source_Phi = hostField.h_Source_Phi;
        #endif // PHI_DIST
        #ifdef LAMBDA_DIST
                treatDataParams.d_Source_Lambda = hostField.h_Source_Lambda;
        #endif // LAMBDA_DIST
        #ifdef CONFORMATION_TENSOR
        #ifdef A_XX_DIST
                    treatDataParams.d_Source_Gxx = hostField.h_Source_Gxx;
        #endif // A_XX_DIST
        #ifdef A_XY_DIST
                    treatDataParams.d_Source_Gxy = hostField.h_Source_Gxy;
        #endif // A_XY_DIST
        #ifdef A_XZ_DIST
                    treatDataParams.d_Source_Gxz = hostField.h_Source_Gxz;
        #endif // A_XZ_DIST
        #ifdef A_YY_DIST
                    treatDataParams.d_Source_Gyy = hostField.h_Source_Gyy;
        #endif // A_YY_DIST
        #ifdef A_YZ_DIST
                    treatDataParams.d_Source_Gyz = hostField.h_Source_Gyz;
        #endif // A_YZ_DIST
        #ifdef A_ZZ_DIST
                    treatDataParams.d_Source_Gzz = hostField.h_Source_Gzz;
        #endif // A_ZZ_DIST
        #endif //CONFORMATION_TENSOR
        #endif //SAVE_LOCAL_FORCES
        treatDataParams.step = step;
        treatData(&treatDataParams);
    }

    /** @brief Releases all allocations owned by GPU @p g. */
    void freeDeviceField(int g) {
        interfaceFree(ghostInterface[g], ghostInterfaceCopy[g], macroInterfaceGPU[g], macroInterfaceGPUCopy[g]);

        #ifdef CONFORMATION_PHYSICAL_A_HALO
            for (int generation=0; generation<2; ++generation) {
                auto& halo=polymerStressHalo[g][generation];
                cudaFree(halo.X_0); cudaFree(halo.X_1);
                cudaFree(halo.Y_0); cudaFree(halo.Y_1);
                cudaFree(halo.Z_0); cudaFree(halo.Z_1);
                cudaFree(halo.auxZ_0); cudaFree(halo.auxZ_1);
                }
        #endif // CONFORMATION_PHYSICAL_A_HALO

        #ifdef CURVED_BOUNDARY_CONDITION
            cudaFree(d_curvedBC[g]);
            cudaFree(d_curvedBC_array[g]);
            d_curvedBC[g] = nullptr;
            d_curvedBC_array[g] = nullptr;
        #endif // CURVED_BOUNDARY_CONDITION

        cudaFree(d_fMom[g]);
        cudaFree(dNodeType[g]);

        #ifdef DENSITY_CORRECTION
            cudaFree(d_mean_rho[g]);
            for (int buffer = 0; buffer < 2; ++buffer) {
                cudaFree(d_density_sum[g][buffer]);
                cudaFree(d_density_count[g][buffer]);
                }
        #endif //DENSITY_CORRECTION

        #ifdef BC_FORCES
            cudaFree(d_BC_Fx[g]);
            cudaFree(d_BC_Fy[g]);
            cudaFree(d_BC_Fz[g]);
        #endif //_BC_FORCES

        #ifdef SAVE_LOCAL_FORCES
            cudaFree(d_Local_Fx);
            cudaFree(d_Local_Fy);
            cudaFree(d_Local_Fz);
        #ifdef SECOND_DIST
                cudaFree(d_Source_C);
        #endif // SECOND_DIST
        #ifdef PHI_DIST
                cudaFree(d_Source_Phi);
        #endif // PHI_DIST
        #ifdef LAMBDA_DIST
                cudaFree(d_Source_Lambda);
        #endif // LAMBDA_DIST
        #ifdef CONFORMATION_TENSOR
        #ifdef A_XX_DIST
                    cudaFree(d_Source_Gxx);
        #endif // A_XX_DIST
        #ifdef A_XY_DIST
                    cudaFree(d_Source_Gxy);
        #endif // A_XY_DIST
        #ifdef A_XZ_DIST
                    cudaFree(d_Source_Gxz);
        #endif // A_XZ_DIST
        #ifdef A_YY_DIST
                    cudaFree(d_Source_Gyy);
        #endif // A_YY_DIST
        #ifdef A_YZ_DIST
                    cudaFree(d_Source_Gyz);
        #endif // A_YZ_DIST
        #ifdef A_ZZ_DIST
                    cudaFree(d_Source_Gzz);
        #endif // A_ZZ_DIST
        #endif //CONFORMATION_TENSOR
        #endif //SAVE_LOCAL_FORCES
    }
} DeviceField;

#endif //__DEVICEFIELD_STRUCTS_H
