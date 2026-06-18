#ifndef __DEVICEFIELD_STRUCTS_H
#define __DEVICEFIELD_STRUCTS_H

#include "var.h"
#include "main.cuh"
#include "hostField.cuh"

typedef struct deviceField{
    ghostInterfaceData ghostInterface[N_GPUS];

    dfloat* d_fMom[N_GPUS];
    unsigned int* dNodeType[N_GPUS];

    #ifdef CURVED_BOUNDARY_CONDITION
    CurvedBoundary** d_curvedBC[N_GPUS];
    CurvedBoundary* d_curvedBC_array[N_GPUS];
    unsigned int numberCurvedBoundaryNodes;  // Number of curved boundary nodes
    #endif

    #ifdef DENSITY_CORRECTION
    dfloat* d_mean_rho[N_GPUS];
    #endif //DENSITY_CORRECTION

    #ifdef BC_FORCES
        dfloat* d_BC_Fx[N_GPUS];
        dfloat* d_BC_Fy[N_GPUS];
        dfloat* d_BC_Fz[N_GPUS];
    #endif //_BC_FORCES

    fluidPhaseProps phasePropsA;             ///< Phase-1 fluid properties (viscous + viscoelastic), always present
    #ifdef PHI_DIST
    fluidPhaseProps phasePropsB;             ///< Phase-2 fluid properties (viscous + viscoelastic)
    #endif

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
                    printf("⚠ GPU %d cannot access GPU %d via P2P\n", GPUS_TO_USE[i], GPUS_TO_USE[j]);
                }
            }
        }
    }

    void allocateDeviceMemoryDeviceField(int g) {
        cudaSetDevice(GPUS_TO_USE[g]);
        unsigned int memAllocated = 0;

        cudaMalloc((void**)&d_fMom[g], MEM_SIZE_MOM_LOCAL);
        cudaMalloc((void**)&dNodeType[g], sizeof(int) * NUMBER_LBM_NODES_LOCAL);
        interfaceMalloc(ghostInterface[g]);

        memAllocated += MEM_SIZE_MOM_LOCAL + sizeof(int) * NUMBER_LBM_NODES_LOCAL;

        #ifdef BC_FORCES
        cudaMalloc((void**)&d_BC_Fx[g], MEM_SIZE_SCALAR);
        cudaMalloc((void**)&d_BC_Fy[g], MEM_SIZE_SCALAR);
        cudaMalloc((void**)&d_BC_Fz[g], MEM_SIZE_SCALAR);
        memAllocated += 3 * MEM_SIZE_SCALAR;
        #endif //BC_FORCES

        printf("Device Memory Allocated for Bulk flow: %.2f MB \n", (float)memAllocated /(1024.0 * 1024.0));
    }

    #ifdef DENSITY_CORRECTION
    void allocateDensityCorrectionMemory(int g){
        cudaMalloc((void**)&d_mean_rho[g], sizeof(dfloat));
    }
    #endif //DENSITY_CORRECTION

    void initializeDomainDeviceField(hostField &hostField, dfloat **&randomNumbers, int &step, dim3 gridBlock, dim3 threadBlock, int g, int slice){
        // ========== INITIALIZATION DOMAIN INLINED ==========
        cudaSetDevice(GPUS_TO_USE[g]);

        int zStart = g * slice;
        int zEnd   = (g == N_GPUS - 1) ? NZ : zStart + slice;

        size_t localNZ_physical = zEnd - zStart;
        size_t localNZ = localNZ_physical;

        // Offset within a global vector
        size_t zOffset = zStart * NX * NY * NUMBER_MOMENTS;

        printf("GPU: %d\n", (int)GPUS_TO_USE[g]);
        printf("Start initialize: %d\n", (int)zStart);
        printf("End initialize: %d\n", (int)zEnd);
        printf("localNZ initialize: %d\n", (int)localNZ);

        // Random numbers initialization
        #ifdef RANDOM_NUMBERS 
            if(console_flush) fflush(stdout);
            checkCudaErrors(cudaMallocManaged((void**)&randomNumbers[g], sizeof(dfloat) * NUMBER_LBM_NODES_LOCAL));
            initializationRandomNumbers(randomNumbers[g], CURAND_SEED);
            checkCudaErrors(cudaDeviceSynchronize());
            getLastCudaError("random numbers transfer error");
            printf("Random numbers initialized - Seed used: %u\n", CURAND_SEED); 
            printf("Device memory allocated for random numbers: %.2f MB\n", (float)(sizeof(dfloat) * NUMBER_LBM_NODES_LOCAL) / (1024.0 * 1024.0));
            if(console_flush) fflush(stdout);
        #endif //RANDOM_NUMBERS

        int checkpoint_state = 0;
        // LBM Initialization
        if (LOAD_CHECKPOINT) {

            printf("Loading checkpoint\n");
            checkpoint_state = loadSimCheckpoint(hostField.h_fMom, ghostInterface[g], &step, g);

            if (checkpoint_state != 0){
                checkCudaErrors(cudaMemcpy(d_fMom[g], hostField.h_fMom + zOffset, sizeof(dfloat) * NUMBER_LBM_NODES_LOCAL * NUMBER_MOMENTS, cudaMemcpyHostToDevice));
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
            gpuInitialization_pop<<<gridBlock, threadBlock>>>(d_fMom[g], ghostInterface[g], localNZ, zStart);
        }

        // Mean flow initialization
        #if MEAN_FLOW
            // Copy mean baseline from device to host so mean flow accumulation starts from the initial state
            checkCudaErrors(cudaMemcpy(hostField.m_fMom, d_fMom, sizeof(dfloat) * NUMBER_LBM_NODES * NUMBER_MOMENTS, cudaMemcpyDeviceToHost));
        #endif //MEAN_FLOW

        // Node type initialization - hNodeType is allocated once globally in allocateHostMemoryHostField().
        // Each GPU operates on its own slice: offset = zStart * NX * NY into the global array.
        unsigned int* const hNodeType_slice = hostField.hNodeType + zStart * NX * NY;

        unsigned int numberCurvedBoundaryNodes_local = 0;

        #ifndef VOXEL_FILENAME
            hostInitialization_nodeType(hNodeType_slice, zStart, localNZ
            #ifdef CURVED_BOUNDARY_CONDITION
            ,&numberCurvedBoundaryNodes_local
            #endif
            );
            checkCudaErrors(cudaMemcpy(dNodeType[g], hNodeType_slice, sizeof(unsigned int) * NUMBER_LBM_NODES_LOCAL, cudaMemcpyHostToDevice));  
            checkCudaErrors(cudaDeviceSynchronize());
            #ifdef FORCE_VOXEL_BC_BUILDING
                define_voxel_bc<<<gridBlock, threadBlock>>>(dNodeType); 
                checkCudaErrors(cudaMemcpy(hNodeType_slice, dNodeType[g], sizeof(unsigned int) * NUMBER_LBM_NODES_LOCAL, cudaMemcpyDeviceToHost)); 
            #endif
        #else
            hostInitialization_nodeType_bulk(hostField.hNodeType); 
            read_xyz_file(VOXEL_FILENAME, hostField.hNodeType);
            hostInitialization_nodeType(hNodeType_slice
            #ifdef CURVED_BOUNDARY_CONDITION
            ,&numberCurvedBoundaryNodes_local
            #endif
            );
            checkCudaErrors(cudaMemcpy(dNodeType[g], hNodeType_slice, sizeof(unsigned int) * NUMBER_LBM_NODES_LOCAL, cudaMemcpyHostToDevice));  
            checkCudaErrors(cudaDeviceSynchronize());
            define_voxel_bc<<<gridBlock, threadBlock>>>(dNodeType[g]); 
            checkCudaErrors(cudaMemcpy(hNodeType_slice, dNodeType[g], sizeof(unsigned int) * NUMBER_LBM_NODES_LOCAL, cudaMemcpyDeviceToHost)); 
        #endif //!VOXEL_FILENAME

        // Boundary condition forces initialization
        #ifdef BC_FORCES
            gpuInitialization_force<<<gridBlock, threadBlock>>>(d_BC_Fx[g], d_BC_Fy[g], d_BC_Fz[g]);
        #endif //BC_FORCES

        #ifdef CURVED_BOUNDARY_CONDITION
            numberCurvedBoundaryNodes = initializeCurvedBoundaryDeviceField(
                hostField.hNodeType,
                dNodeType[g],
                d_curvedBC[g],
                d_curvedBC_array[g]
            );
        #endif

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
        checkCudaErrors(cudaMemcpy(hostField.h_fMom + zOffset, d_fMom[g], sizeof(dfloat) * NUMBER_LBM_NODES_LOCAL * NUMBER_MOMENTS, cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaDeviceSynchronize());

        printf("Syncing data back to host (g=%d) \n", g); if(console_flush) fflush(stdout);


        // Free random numbers if initialized
        #ifdef RANDOM_NUMBERS
            checkCudaErrors(cudaSetDevice(GPUS_TO_USE[g]));
            cudaFree(randomNumbers[g]);
            free(randomNumbers);
            printf("Random numbers free \n"); if(console_flush) fflush(stdout);
        #endif //RANDOM_NUMBERS
        
    }

    #ifdef CURVED_BOUNDARY_CONDITION
    void updateCurvedBoundaryVelocitiesDeviceField(int g, cudaStream_t stream){
        // Skip launch when no curved-boundary nodes were found; zero grid size is invalid
        if (numberCurvedBoundaryNodes == 0) {
            return;
        }

        const int curvedBCBlockSize = 256;
        const int curvedBCGridSize = (numberCurvedBoundaryNodes + curvedBCBlockSize - 1) / curvedBCBlockSize;
        updateCurvedBoundaryVelocities<<<curvedBCGridSize, curvedBCBlockSize, 0, stream>>>(d_curvedBC_array[g], d_fMom[g], numberCurvedBoundaryNodes);
    }
    #endif //CURVED_BOUNDARY_CONDITION

    #ifdef DENSITY_CORRECTION
    void mean_rhoDeviceField(size_t step, int g, cudaStream_t stream){
        mean_rho(d_fMom[g], step, d_mean_rho[g], stream);
    }
    #endif //DENSITY_CORRECTION

    void gpuMomCollisionStreamDeviceField(dim3 gridBlock, dim3 threadBlock, unsigned int step, bool save, int g, int slice, cudaStream_t stream){
        // Create parameter struct and pass by value (most efficient!)
        cudaSetDevice(GPUS_TO_USE[g]);
        DeviceKernelParams params;
        int zStart = g * slice;
        int zEnd   = (g == N_GPUS - 1) ? NZ : zStart + slice;
        size_t localNZ = zEnd - zStart;

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
        params.step = step;
        params.save = save;
        params.zStart = zStart;
        params.localNZ = localNZ;
        
        #ifdef DENSITY_CORRECTION
        params.d_mean_rho = d_mean_rho[g];
        #endif //DENSITY_CORRECTION
        
        #ifdef BC_FORCES
        params.d_BC_Fx = d_BC_Fx[g];
        params.d_BC_Fy = d_BC_Fy[g];
        params.d_BC_Fz = d_BC_Fz[g];
        #endif //BC_FORCES
        
        #ifdef CURVED_BOUNDARY_CONDITION
        params.d_curvedBC = d_curvedBC[g];
        params.d_curvedBC_array = d_curvedBC_array[g];
        #endif //CURVED_BOUNDARY_CONDITION
        
        #if defined(NON_NEWTONIAN_FLUID) || defined(CONFORMATION_TENSOR)
        params.phasePropsA = phasePropsA;
        #ifdef PHI_DIST
        params.phasePropsB = phasePropsB;
        #endif
        #endif //NON_NEWTONIAN_FLUID || CONFORMATION_TENSOR
        
        // Pass struct by value - CUDA handles this efficiently
        #ifdef DYNAMIC_SHARED_MEMORY
        gpuMomCollisionStream<<<gridBlock, threadBlock, MAX_SHARED_MEMORY_SIZE, stream>>>(params);
        #else
        gpuMomCollisionStream<<<gridBlock, threadBlock, 0, stream>>>(params);
        #endif
    }

    /* -------------- Send the current GPU's Z_1 top to the nearest GPU's Z_1 base   ------------- */
    void sendTopToNext(int g, deviceField* allDevices, cudaStream_t streamLBM)
    {
        checkCudaErrors(cudaSetDevice(GPUS_TO_USE[g]));
        const int gNext = (g + 1) % N_GPUS;
        const size_t planeSize = (size_t)BLOCK_NX * BLOCK_NY * NUM_BLOCK_X * NUM_BLOCK_Y * QF;
        const size_t haloSize  = planeSize * sizeof(dfloat);
        const size_t topOffset = (size_t)(NUM_BLOCK_Z_LOCAL - 1) * planeSize;

        checkCudaErrors(cudaMemcpyPeerAsync(
            allDevices[gNext].ghostInterface[gNext].popAux.Z_1,
            GPUS_TO_USE[gNext],
            allDevices[g].ghostInterface[g].pop.Z_1 + topOffset,
            GPUS_TO_USE[g],
            haloSize, streamLBM
        ));
    }

    /* -------------- Receives the Z_0 base of the next GPU and adds it to the Z_0 top of the current GPU  ------------- */
    void sendBottomToPrev(int g, deviceField* allDevices, cudaStream_t streamLBM)
    {
        checkCudaErrors(cudaSetDevice(GPUS_TO_USE[g]));
        const int gNext = (g + 1) % N_GPUS;
        const size_t planeSize = (size_t)BLOCK_NX * BLOCK_NY * NUM_BLOCK_X * NUM_BLOCK_Y * QF;
        const size_t haloSize  = planeSize * sizeof(dfloat);
        const size_t topOffset = (size_t)(NUM_BLOCK_Z_LOCAL - 1) * planeSize;

        checkCudaErrors(cudaMemcpyPeerAsync(
            allDevices[g].ghostInterface[g].popAux.Z_0,
            GPUS_TO_USE[g],
            allDevices[gNext].ghostInterface[gNext].pop.Z_0,
            GPUS_TO_USE[gNext],
            haloSize, streamLBM
        ));
    }

    #ifdef PHI_DIST
    void computePhaseNormalsDeviceField(dim3 gridBlock, dim3 threadBlock, int g, cudaStream_t stream){
        gpuComputePhaseNormals<<<gridBlock, threadBlock, 0, stream>>>(d_fMom[g], dNodeType[g]);
        gpuComputeChemicalPotential<<<gridBlock, threadBlock, 0, stream>>>(d_fMom[g], dNodeType[g]);
        gpuComputeLaplacianMu<<<gridBlock, threadBlock, 0, stream>>>(d_fMom[g], dNodeType[g]);
    }
    #endif //PHI_DIST

    void swapGhostInterfacesDeviceField(int g){
        swapGhostInterfaces(ghostInterface[g]);
    }

    void halfStepKernels(dim3 gridBlock, dim3 threadBlock, size_t step, int g, cudaStream_t stream){
        #ifdef LOCAL_FORCES
            gpuResetMacroForcesDeviceField(gridBlock, threadBlock, g, stream);
            CHECK_KERNEL_ERR("Force Reset kernel");
        #endif //LOCAL_FORCES
        #ifdef CURVED_BOUNDARY_CONDITION
            updateCurvedBoundaryVelocitiesDeviceField(g, stream);
            CHECK_KERNEL_ERR("Curved BC kernel");
        #endif //CURVED_BOUNDARY_CONDITION
        #ifdef PHI_DIST
            computePhaseNormalsDeviceField(gridBlock, threadBlock, g, stream);
            CHECK_KERNEL_ERR("Phi gradients kernel");
        #endif //PHI_DIST
        #ifdef DENSITY_CORRECTION
            mean_rhoDeviceField(step, g, stream);
            CHECK_KERNEL_ERR("Density correction kernel");
        #endif //DENSITY_CORRECTION
    }
    
    #ifdef LOCAL_FORCES
    void gpuResetMacroForcesDeviceField(dim3 gridBlock, dim3 threadBlock, int g, cudaStream_t stream){
        gpuResetMacroForces<<<gridBlock, threadBlock, 0, stream>>>(d_fMom[g]);
    }
    #endif //LOCAL_FORCES

    #ifdef PARTICLE_MODEL
    void particleSimulationDeviceField(ParticlesSoA &particlesSoA, cudaStream_t *streamsPart, ParticleWallForces *d_pwForces,unsigned int step){
        particleSimulation(&particlesSoA,d_fMom,streamsPart,d_pwForces,step);
    }
    #endif //PARTICLE_MODEL

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

    void cudaMemcpyDeviceField(hostField &hostField, int g, int slice){
        checkCudaErrors(cudaSetDevice(GPUS_TO_USE[g]));
        int zStart = g * slice;
        size_t zOffset = zStart * NX * NY * NUMBER_MOMENTS;
        checkCudaErrors(cudaMemcpy(hostField.h_fMom + zOffset, d_fMom[g], sizeof(dfloat) * NUMBER_LBM_NODES_LOCAL*NUMBER_MOMENTS, cudaMemcpyDeviceToHost));
        
        // Copy BC forces arrays if enabled
        #if defined BC_FORCES && defined SAVE_BC_FORCES
        checkCudaErrors(cudaMemcpy(hostField.h_BC_Fx, d_BC_Fx[g], MEM_SIZE_SCALAR_LOCAL, cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(hostField.h_BC_Fy, d_BC_Fy[g], MEM_SIZE_SCALAR_LOCAL, cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(hostField.h_BC_Fz, d_BC_Fz[g], MEM_SIZE_SCALAR_LOCAL, cudaMemcpyDeviceToHost));
        #endif //BC_FORCES && SAVE_BC_FORCES
    }

    void saveSimCheckpointHostDeviceField(hostField &hostField, int &step, int g){
        saveSimCheckpoint(hostField.h_fMom, ghostInterface[g], &step, g);
    }

    void saveSimCheckpointDeviceField( int &step, int g){
        saveSimCheckpoint(d_fMom[g],ghostInterface[g],&step, g);
    }

    void treatDataDeviceField(hostField &hostField, 
        int step, int g){
        checkCudaErrors(cudaSetDevice(GPUS_TO_USE[g]));
        TreatDataParams treatDataParams;
        treatDataParams.h_fMom = hostField.h_fMom;
        treatDataParams.d_fMom = d_fMom[g];
        #if MEAN_FLOW
        treatDataParams.d_fMom_mean = hostField.m_fMom;
        #endif
        #ifdef BC_FORCES
        treatDataParams.d_BC_Fx = d_BC_Fx[g];
        treatDataParams.d_BC_Fy = d_BC_Fy[g];
        treatDataParams.d_BC_Fz = d_BC_Fz[g];
        #endif
        treatDataParams.step = step;
        treatData(&treatDataParams);
    }

    #if defined BC_FORCES && defined SAVE_BC_FORCES
    void saveBcForces(hostField &hostField){
        checkCudaErrors(cudaDeviceSynchronize()); 
        checkCudaErrors(cudaMemcpy(hostField.h_BC_Fx, d_BC_Fx[g], MEM_SIZE_SCALAR_LOCAL, cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(hostField.h_BC_Fy, d_BC_Fy[g], MEM_SIZE_SCALAR_LOCAL, cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(hostField.h_BC_Fz, d_BC_Fz[g], MEM_SIZE_SCALAR_LOCAL, cudaMemcpyDeviceToHost));
    }
    #endif //BC_FORCES && SAVE_BC_FORCES

    void freeDeviceField(int g) {
        interfaceFree(ghostInterface[g]);

        cudaFree(d_fMom[g]);
        cudaFree(dNodeType[g]);

        #ifdef DENSITY_CORRECTION
        cudaFree(d_mean_rho[g]);
        #endif //DENSITY_CORRECTION

        #ifdef BC_FORCES
        cudaFree(d_BC_Fx[g]);
        cudaFree(d_BC_Fy[g]);
        cudaFree(d_BC_Fz[g]);
        #endif //_BC_FORCES
    }
} DeviceField;

#endif //__DEVICEFIELD_STRUCTS_H