#ifndef __DEVICEFIELD_STRUCTS_H
#define __DEVICEFIELD_STRUCTS_H

#include "var.h"
#include "main.cuh"
#include "hostField.cuh"

typedef struct deviceField{
    ghostInterfaceData ghostInterface[N_GPUS];
    macroInterfaceGPUData macroInterfaceGPU[N_GPUS];

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

    #ifdef SAVE_LOCAL_FORCES
        dfloat* d_Local_Fx;
        dfloat* d_Local_Fy;
        dfloat* d_Local_Fz;
        #ifdef SECOND_DIST
        dfloat* d_Source_C;
        #endif
        #ifdef PHI_DIST
        dfloat* d_Source_Phi;
        #endif
        #ifdef LAMBDA_DIST
        dfloat* d_Source_Lambda;
        #endif
        #ifdef CONFORMATION_TENSOR
            #ifdef A_XX_DIST
        dfloat* d_Source_Gxx;
            #endif
            #ifdef A_XY_DIST
        dfloat* d_Source_Gxy;
            #endif
            #ifdef A_XZ_DIST
        dfloat* d_Source_Gxz;
            #endif
            #ifdef A_YY_DIST
        dfloat* d_Source_Gyy;
            #endif
            #ifdef A_YZ_DIST
        dfloat* d_Source_Gyz;
            #endif
            #ifdef A_ZZ_DIST
        dfloat* d_Source_Gzz;
            #endif
        #endif //CONFORMATION_TENSOR
    #endif //SAVE_LOCAL_FORCES

    #if defined(NON_NEWTONIAN_FLUID) || defined(CONFORMATION_TENSOR)
    fluidPhaseProps phasePropsA;             ///< Phase-1 fluid properties (viscous + viscoelastic)
    #endif 


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
        interfaceMalloc(ghostInterface[g], macroInterfaceGPU[g]);

        memAllocated += MEM_SIZE_MOM_LOCAL + sizeof(int) * NUMBER_LBM_NODES_LOCAL;

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
            #endif
            #ifdef PHI_DIST
        cudaMalloc((void**)&d_Source_Phi, MEM_SIZE_SCALAR);
        memAllocated += MEM_SIZE_SCALAR;
            #endif
            #ifdef LAMBDA_DIST
        cudaMalloc((void**)&d_Source_Lambda, MEM_SIZE_SCALAR);
        memAllocated += MEM_SIZE_SCALAR;
            #endif
            #ifdef CONFORMATION_TENSOR
                #ifdef A_XX_DIST
        cudaMalloc((void**)&d_Source_Gxx, MEM_SIZE_SCALAR);
        memAllocated += MEM_SIZE_SCALAR;
                #endif
                #ifdef A_XY_DIST
        cudaMalloc((void**)&d_Source_Gxy, MEM_SIZE_SCALAR);
        memAllocated += MEM_SIZE_SCALAR;
                #endif
                #ifdef A_XZ_DIST
        cudaMalloc((void**)&d_Source_Gxz, MEM_SIZE_SCALAR);
        memAllocated += MEM_SIZE_SCALAR;
                #endif
                #ifdef A_YY_DIST
        cudaMalloc((void**)&d_Source_Gyy, MEM_SIZE_SCALAR);
        memAllocated += MEM_SIZE_SCALAR;
                #endif
                #ifdef A_YZ_DIST
        cudaMalloc((void**)&d_Source_Gyz, MEM_SIZE_SCALAR);
        memAllocated += MEM_SIZE_SCALAR;
                #endif
                #ifdef A_ZZ_DIST
        cudaMalloc((void**)&d_Source_Gzz, MEM_SIZE_SCALAR);
        memAllocated += MEM_SIZE_SCALAR;
                #endif
            #endif //CONFORMATION_TENSOR
        #endif //SAVE_LOCAL_FORCES

        printf("Device Memory Allocated for Bulk flow: %.2f MB \n", (float)memAllocated /(1024.0 * 1024.0));
    }

    #ifdef DENSITY_CORRECTION
    void allocateDensityCorrectionMemory(int g){
        cudaMalloc((void**)&d_mean_rho[g], sizeof(dfloat));
    }
    #endif //DENSITY_CORRECTION

    void initializeDomainDeviceField(hostField &hostField, dfloat **&randomNumbers, int &step, dim3 gridBlock, dim3 threadBlock, int g, int slice){
        // ========== INITIALIZATION DOMAIN INLINED ==========
        checkCudaErrors(cudaSetDevice(GPUS_TO_USE[g]));

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
            checkCudaErrors(cudaMalloc((void**)&randomNumbers[g], sizeof(dfloat) * NUMBER_LBM_NODES_LOCAL));
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
            checkpoint_state = loadSimCheckpoint(hostField.h_fMom + zOffset, ghostInterface[g], &step, g);

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
            checkCudaErrors(cudaMemcpy(hostField.m_fMom + zOffset, d_fMom[g], sizeof(dfloat) * NUMBER_LBM_NODES_LOCAL * NUMBER_MOMENTS, cudaMemcpyDeviceToHost));
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
                define_voxel_bc<<<gridBlock, threadBlock>>>(dNodeType[g]); 
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

        #ifdef SAVE_LOCAL_FORCES
            gpuInitialization_force<<<gridBlock, threadBlock>>>(d_Local_Fx, d_Local_Fy, d_Local_Fz);
            #ifdef SECOND_DIST
            gpuInitialization_force<<<gridBlock, threadBlock>>>(d_Source_C, d_Source_C, d_Source_C);
            #endif
            #ifdef PHI_DIST
            gpuInitialization_force<<<gridBlock, threadBlock>>>(d_Source_Phi, d_Source_Phi, d_Source_Phi);
            #endif
            #ifdef LAMBDA_DIST
            gpuInitialization_force<<<gridBlock, threadBlock>>>(d_Source_Lambda, d_Source_Lambda, d_Source_Lambda);
            #endif
            #ifdef CONFORMATION_TENSOR
                #ifdef A_XX_DIST
            gpuInitialization_force<<<gridBlock, threadBlock>>>(d_Source_Gxx, d_Source_Gxx, d_Source_Gxx);
                #endif
                #ifdef A_XY_DIST
            gpuInitialization_force<<<gridBlock, threadBlock>>>(d_Source_Gxy, d_Source_Gxy, d_Source_Gxy);
                #endif
                #ifdef A_XZ_DIST
            gpuInitialization_force<<<gridBlock, threadBlock>>>(d_Source_Gxz, d_Source_Gxz, d_Source_Gxz);
                #endif
                #ifdef A_YY_DIST
            gpuInitialization_force<<<gridBlock, threadBlock>>>(d_Source_Gyy, d_Source_Gyy, d_Source_Gyy);
                #endif
                #ifdef A_YZ_DIST
            gpuInitialization_force<<<gridBlock, threadBlock>>>(d_Source_Gyz, d_Source_Gyz, d_Source_Gyz);
                #endif
                #ifdef A_ZZ_DIST
            gpuInitialization_force<<<gridBlock, threadBlock>>>(d_Source_Gzz, d_Source_Gzz, d_Source_Gzz);
                #endif
            #endif //CONFORMATION_TENSOR
        #endif //SAVE_LOCAL_FORCES

        #ifdef CURVED_BOUNDARY_CONDITION
            numberCurvedBoundaryNodes = initializeCurvedBoundaryDeviceField(
                hNodeType_slice,
                dNodeType[g],
                d_curvedBC[g],
                d_curvedBC_array[g],
                zStart,
                localNZ
            );
            // The main collision kernel consumes curvedBC->vel. Populate it from
            // the initialized moment field before the first simulation step.
            updateCurvedBoundaryVelocitiesDeviceField(g, 0);
            CHECK_KERNEL_ERR("Initial curved BC kernel");
            checkCudaErrors(cudaDeviceSynchronize());
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

        params.rho_macro.Z_0 = macroInterfaceGPU[g].rho.Z_0;
        params.rho_macro.Z_1 = macroInterfaceGPU[g].rho.Z_1;
        params.rho_macro.auxZ_0 = macroInterfaceGPU[g].rho.auxZ_0;
        params.rho_macro.auxZ_1 = macroInterfaceGPU[g].rho.auxZ_1;

        params.ux_macro.Z_0 = macroInterfaceGPU[g].ux.Z_0;
        params.ux_macro.Z_1 = macroInterfaceGPU[g].ux.Z_1;
        params.ux_macro.auxZ_0 = macroInterfaceGPU[g].ux.auxZ_0;
        params.ux_macro.auxZ_1 = macroInterfaceGPU[g].ux.auxZ_1;

        params.uy_macro.Z_0 = macroInterfaceGPU[g].uy.Z_0;
        params.uy_macro.Z_1 = macroInterfaceGPU[g].uy.Z_1;
        params.uy_macro.auxZ_0 = macroInterfaceGPU[g].uy.auxZ_0;
        params.uy_macro.auxZ_1 = macroInterfaceGPU[g].uy.auxZ_1;

        params.uz_macro.Z_0 = macroInterfaceGPU[g].uz.Z_0;
        params.uz_macro.Z_1 = macroInterfaceGPU[g].uz.Z_1;
        params.uz_macro.auxZ_0 = macroInterfaceGPU[g].uz.auxZ_0;
        params.uz_macro.auxZ_1 = macroInterfaceGPU[g].uz.auxZ_1;

        #ifdef SECOND_DIST
        params.g.X_0    = ghostInterface[g].g.X_0;
        params.g.X_1    = ghostInterface[g].g.X_1;
        params.g.Y_0    = ghostInterface[g].g.Y_0;
        params.g.Y_1    = ghostInterface[g].g.Y_1;
        params.g.Z_0    = ghostInterface[g].g.Z_0;
        params.g.Z_1    = ghostInterface[g].g.Z_1;
        params.g.auxZ_0 = ghostInterface[g].gAux.Z_0;
        params.g.auxZ_1 = ghostInterface[g].gAux.Z_1;

        params.g_macro.Z_0 = macroInterfaceGPU[g].g.Z_0;
        params.g_macro.Z_1 = macroInterfaceGPU[g].g.Z_1;
        params.g_macro.auxZ_0 = macroInterfaceGPU[g].g.auxZ_0;
        params.g_macro.auxZ_1 = macroInterfaceGPU[g].g.auxZ_1;
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

        params.lambda_macro.Z_0 = macroInterfaceGPU[g].lambda.Z_0;
        params.lambda_macro.Z_1 = macroInterfaceGPU[g].lambda.Z_1;
        params.lambda_macro.auxZ_0 = macroInterfaceGPU[g].lambda.auxZ_0;
        params.lambda_macro.auxZ_1 = macroInterfaceGPU[g].lambda.auxZ_1;
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

        params.Axx_macro.Z_0 = macroInterfaceGPU[g].Axx.Z_0;
        params.Axx_macro.Z_1 = macroInterfaceGPU[g].Axx.Z_1;
        params.Axx_macro.auxZ_0 = macroInterfaceGPU[g].Axx.auxZ_0;
        params.Axx_macro.auxZ_1 = macroInterfaceGPU[g].Axx.auxZ_1;
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

        params.Axy_macro.Z_0 = macroInterfaceGPU[g].Axy.Z_0;
        params.Axy_macro.Z_1 = macroInterfaceGPU[g].Axy.Z_1;
        params.Axy_macro.auxZ_0 = macroInterfaceGPU[g].Axy.auxZ_0;
        params.Axy_macro.auxZ_1 = macroInterfaceGPU[g].Axy.auxZ_1;
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

        params.Axz_macro.Z_0 = macroInterfaceGPU[g].Axz.Z_0;
        params.Axz_macro.Z_1 = macroInterfaceGPU[g].Axz.Z_1;
        params.Axz_macro.auxZ_0 = macroInterfaceGPU[g].Axz.auxZ_0;
        params.Axz_macro.auxZ_1 = macroInterfaceGPU[g].Axz.auxZ_1;
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

        params.Ayy_macro.Z_0 = macroInterfaceGPU[g].Ayy.Z_0;
        params.Ayy_macro.Z_1 = macroInterfaceGPU[g].Ayy.Z_1;
        params.Ayy_macro.auxZ_0 = macroInterfaceGPU[g].Ayy.auxZ_0;
        params.Ayy_macro.auxZ_1 = macroInterfaceGPU[g].Ayy.auxZ_1;
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

        params.Ayz_macro.Z_0 = macroInterfaceGPU[g].Ayz.Z_0;
        params.Ayz_macro.Z_1 = macroInterfaceGPU[g].Ayz.Z_1;
        params.Ayz_macro.auxZ_0 = macroInterfaceGPU[g].Ayz.auxZ_0;
        params.Ayz_macro.auxZ_1 = macroInterfaceGPU[g].Ayz.auxZ_1;
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

        params.Azz_macro.Z_0 = macroInterfaceGPU[g].Azz.Z_0;
        params.Azz_macro.Z_1 = macroInterfaceGPU[g].Azz.Z_1;
        params.Azz_macro.auxZ_0 = macroInterfaceGPU[g].Azz.auxZ_0;
        params.Azz_macro.auxZ_1 = macroInterfaceGPU[g].Azz.auxZ_1;
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
            #endif
            #ifdef PHI_DIST
        params.d_Source_Phi = d_Source_Phi;
            #endif
            #ifdef LAMBDA_DIST
        params.d_Source_Lambda = d_Source_Lambda;
            #endif
            #ifdef CONFORMATION_TENSOR
                #ifdef A_XX_DIST
        params.d_Source_Gxx = d_Source_Gxx;
                #endif
                #ifdef A_XY_DIST
        params.d_Source_Gxy = d_Source_Gxy;
                #endif
                #ifdef A_XZ_DIST
        params.d_Source_Gxz = d_Source_Gxz;
                #endif
                #ifdef A_YY_DIST
        params.d_Source_Gyy = d_Source_Gyy;
                #endif
                #ifdef A_YZ_DIST
        params.d_Source_Gyz = d_Source_Gyz;
                #endif
                #ifdef A_ZZ_DIST
        params.d_Source_Gzz = d_Source_Gzz;
                #endif
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

        params.phi_macro.Z_0 = macroInterfaceGPU[g].phi.Z_0;
        params.phi_macro.Z_1 = macroInterfaceGPU[g].phi.Z_1;
        params.phi_macro.auxZ_0 = macroInterfaceGPU[g].phi.auxZ_0;
        params.phi_macro.auxZ_1 = macroInterfaceGPU[g].phi.auxZ_1;
        params.nx_macro.Z_0 = macroInterfaceGPU[g].nx.Z_0;
        params.nx_macro.Z_1 = macroInterfaceGPU[g].nx.Z_1;
        params.nx_macro.auxZ_0 = macroInterfaceGPU[g].nx.auxZ_0;
        params.nx_macro.auxZ_1 = macroInterfaceGPU[g].nx.auxZ_1;
        params.ny_macro.Z_0 = macroInterfaceGPU[g].ny.Z_0;
        params.ny_macro.Z_1 = macroInterfaceGPU[g].ny.Z_1;
        params.ny_macro.auxZ_0 = macroInterfaceGPU[g].ny.auxZ_0;
        params.ny_macro.auxZ_1 = macroInterfaceGPU[g].ny.auxZ_1;
        params.nz_macro.Z_0 = macroInterfaceGPU[g].nz.Z_0;
        params.nz_macro.Z_1 = macroInterfaceGPU[g].nz.Z_1;
        params.nz_macro.auxZ_0 = macroInterfaceGPU[g].nz.auxZ_0;
        params.nz_macro.auxZ_1 = macroInterfaceGPU[g].nz.auxZ_1;

        params.mu_macro.Z_0 = macroInterfaceGPU[g].mu.Z_0;
        params.mu_macro.Z_1 = macroInterfaceGPU[g].mu.Z_1;
        params.mu_macro.auxZ_0 = macroInterfaceGPU[g].mu.auxZ_0;
        params.mu_macro.auxZ_1 = macroInterfaceGPU[g].mu.auxZ_1;

        #endif //PHI_DIST
        #endif //NON_NEWTONIAN_FLUID || CONFORMATION_TENSOR
        
        // Pass struct by value - CUDA handles this efficiently
        #ifdef DYNAMIC_SHARED_MEMORY
        gpuMomCollisionStream<<<gridBlock, threadBlock, MAX_SHARED_MEMORY_SIZE, stream>>>(params);
        #else

        gpuMomCollisionStream<<<gridBlock, threadBlock, 0, stream>>>(params);

        #endif
    }

    void packMacroHalosDeviceField(int g, int slice, cudaStream_t stream) {
        checkCudaErrors(cudaSetDevice(GPUS_TO_USE[g]));
        const dim3 grid((NX * NY + 255) / 256, 2);
        gpuPackMacroHalos<<<grid, 256, 0, stream>>>(d_fMom[g], macroInterfaceGPU[g], slice);
        CHECK_KERNEL_ERR("Pack macro halos kernel");
    }

    // Each sender pushes its top face and pulls the next slab's bottom face.
    void exchangeMacroField(int g, deviceField* allDevices,
        gpuDirection macroInterfaceGPUData::*field, cudaStream_t stream) {
        checkCudaErrors(cudaSetDevice(GPUS_TO_USE[g]));
        #ifdef BC_Z_WALL
        if (g == N_GPUS - 1) return;
        #endif
        const int next = (g + 1) % N_GPUS;
        const gpuDirection own = macroInterfaceGPU[g].*field;
        const gpuDirection neighbor = allDevices[next].macroInterfaceGPU[next].*field;
        const size_t bytes = (size_t)NX * NY * sizeof(dfloat);
        checkCudaErrors(cudaMemcpyPeerAsync(neighbor.auxZ_1, GPUS_TO_USE[next],
            own.Z_1, GPUS_TO_USE[g], bytes, stream));
        checkCudaErrors(cudaMemcpyPeerAsync(own.auxZ_0, GPUS_TO_USE[g],
            neighbor.Z_0, GPUS_TO_USE[next], bytes, stream));
    }

    void sendMacroTopToNext(int g, deviceField* allDevices, cudaStream_t streamLBM)
    {
        checkCudaErrors(cudaSetDevice(GPUS_TO_USE[g]));
        #ifdef BC_Z_WALL
        if (g == N_GPUS - 1) return;
        #endif

        const int gNext = (g + 1) % N_GPUS;
        const size_t MacroHaloSize = NX * NY * sizeof(dfloat);

        checkCudaErrors(cudaMemcpyPeerAsync(
            allDevices[gNext].macroInterfaceGPU[gNext].rho.auxZ_1,
            GPUS_TO_USE[gNext],
            allDevices[g].macroInterfaceGPU[g].rho.Z_1,
            GPUS_TO_USE[g],
            MacroHaloSize, streamLBM
        ));

        checkCudaErrors(cudaMemcpyPeerAsync(
            allDevices[gNext].macroInterfaceGPU[gNext].ux.auxZ_1,
            GPUS_TO_USE[gNext],
            allDevices[g].macroInterfaceGPU[g].ux.Z_1,
            GPUS_TO_USE[g],
            MacroHaloSize, streamLBM
        ));

        checkCudaErrors(cudaMemcpyPeerAsync(
            allDevices[gNext].macroInterfaceGPU[gNext].uy.auxZ_1,
            GPUS_TO_USE[gNext],
            allDevices[g].macroInterfaceGPU[g].uy.Z_1,
            GPUS_TO_USE[g],
            MacroHaloSize, streamLBM
        ));

        checkCudaErrors(cudaMemcpyPeerAsync(
            allDevices[gNext].macroInterfaceGPU[gNext].uz.auxZ_1,
            GPUS_TO_USE[gNext],
            allDevices[g].macroInterfaceGPU[g].uz.Z_1,
            GPUS_TO_USE[g],
            MacroHaloSize, streamLBM
        ));

        #ifdef SECOND_DIST
        checkCudaErrors(cudaMemcpyPeerAsync(
            allDevices[gNext].macroInterfaceGPU[gNext].g.auxZ_1,
            GPUS_TO_USE[gNext],
            allDevices[g].macroInterfaceGPU[g].g.Z_1,
            GPUS_TO_USE[g],
            MacroHaloSize, streamLBM
        ));
        #endif //SECOND_DIST
        #ifdef LAMBDA_DIST
        checkCudaErrors(cudaMemcpyPeerAsync(
            allDevices[gNext].macroInterfaceGPU[gNext].lambda.auxZ_1,
            GPUS_TO_USE[gNext],
            allDevices[g].macroInterfaceGPU[g].lambda.Z_1,
            GPUS_TO_USE[g],
            MacroHaloSize, streamLBM
        ));
        #endif //LAMBDA_DIST
        #ifdef A_XX_DIST
        checkCudaErrors(cudaMemcpyPeerAsync(
            allDevices[gNext].macroInterfaceGPU[gNext].Axx.auxZ_1,
            GPUS_TO_USE[gNext],
            allDevices[g].macroInterfaceGPU[g].Axx.Z_1,
            GPUS_TO_USE[g],
            MacroHaloSize, streamLBM
        ));
        #endif //A_XX_DIST
        #ifdef A_XY_DIST
        checkCudaErrors(cudaMemcpyPeerAsync(
            allDevices[gNext].macroInterfaceGPU[gNext].Axy.auxZ_1,
            GPUS_TO_USE[gNext],
            allDevices[g].macroInterfaceGPU[g].Axy.Z_1,
            GPUS_TO_USE[g],
            MacroHaloSize, streamLBM
        ));
        #endif //A_XY_DIST
        #ifdef A_XZ_DIST
        checkCudaErrors(cudaMemcpyPeerAsync(
            allDevices[gNext].macroInterfaceGPU[gNext].Axz.auxZ_1,
            GPUS_TO_USE[gNext],
            allDevices[g].macroInterfaceGPU[g].Axz.Z_1,
            GPUS_TO_USE[g],
            MacroHaloSize, streamLBM
        ));
        #endif //A_XZ_DIST
        #ifdef A_YY_DIST
        checkCudaErrors(cudaMemcpyPeerAsync(
            allDevices[gNext].macroInterfaceGPU[gNext].Ayy.auxZ_1,
            GPUS_TO_USE[gNext],
            allDevices[g].macroInterfaceGPU[g].Ayy.Z_1,
            GPUS_TO_USE[g],
            MacroHaloSize, streamLBM
        ));
        #endif //A_YY_DIST
        #ifdef A_YZ_DIST
        checkCudaErrors(cudaMemcpyPeerAsync(
            allDevices[gNext].macroInterfaceGPU[gNext].Ayz.auxZ_1,
            GPUS_TO_USE[gNext],
            allDevices[g].macroInterfaceGPU[g].Ayz.Z_1,
            GPUS_TO_USE[g],
            MacroHaloSize, streamLBM
        ));
        #endif //A_YZ_DIST
        #ifdef A_ZZ_DIST
        checkCudaErrors(cudaMemcpyPeerAsync(
            allDevices[gNext].macroInterfaceGPU[gNext].Azz.auxZ_1,
            GPUS_TO_USE[gNext],
            allDevices[g].macroInterfaceGPU[g].Azz.Z_1,
            GPUS_TO_USE[g],
            MacroHaloSize, streamLBM
        ));
        #endif //A_ZZ_DIST

        #ifdef PHI_DIST
        checkCudaErrors(cudaMemcpyPeerAsync(
            allDevices[gNext].macroInterfaceGPU[gNext].phi.auxZ_1,
            GPUS_TO_USE[gNext],
            allDevices[g].macroInterfaceGPU[g].phi.Z_1,
            GPUS_TO_USE[g],
            MacroHaloSize, streamLBM
        ));

        checkCudaErrors(cudaMemcpyPeerAsync(
            allDevices[gNext].macroInterfaceGPU[gNext].nx.auxZ_1,
            GPUS_TO_USE[gNext],
            allDevices[g].macroInterfaceGPU[g].nx.Z_1,
            GPUS_TO_USE[g],
            MacroHaloSize, streamLBM
        ));

        checkCudaErrors(cudaMemcpyPeerAsync(
            allDevices[gNext].macroInterfaceGPU[gNext].ny.auxZ_1,
            GPUS_TO_USE[gNext],
            allDevices[g].macroInterfaceGPU[g].ny.Z_1,
            GPUS_TO_USE[g],
            MacroHaloSize, streamLBM
        ));

        checkCudaErrors(cudaMemcpyPeerAsync(
            allDevices[gNext].macroInterfaceGPU[gNext].nz.auxZ_1,
            GPUS_TO_USE[gNext],
            allDevices[g].macroInterfaceGPU[g].nz.Z_1,
            GPUS_TO_USE[g],
            MacroHaloSize, streamLBM
        ));

        checkCudaErrors(cudaMemcpyPeerAsync(
            allDevices[gNext].macroInterfaceGPU[gNext].mu.auxZ_1,
            GPUS_TO_USE[gNext],
            allDevices[g].macroInterfaceGPU[g].mu.Z_1,
            GPUS_TO_USE[g],
            MacroHaloSize, streamLBM
        ));

        #endif //PHI_DIST
    }

    void sendMacroBottomToPrev(int g, deviceField* allDevices, cudaStream_t streamLBM)
    {
        checkCudaErrors(cudaSetDevice(GPUS_TO_USE[g]));
        const int gNext = (g + 1) % N_GPUS;
        const size_t MacroHaloSize = NX * NY * sizeof(dfloat);

        #ifdef BC_Z_WALL
        if (g == N_GPUS - 1) return;
        #endif

        checkCudaErrors(cudaMemcpyPeerAsync(
            allDevices[g].macroInterfaceGPU[g].rho.auxZ_0,
            GPUS_TO_USE[g],
            allDevices[gNext].macroInterfaceGPU[gNext].rho.Z_0,
            GPUS_TO_USE[gNext],
            MacroHaloSize, streamLBM
        ));

        checkCudaErrors(cudaMemcpyPeerAsync(
            allDevices[g].macroInterfaceGPU[g].ux.auxZ_0,
            GPUS_TO_USE[g],
            allDevices[gNext].macroInterfaceGPU[gNext].ux.Z_0,
            GPUS_TO_USE[gNext],
            MacroHaloSize, streamLBM
        ));

        checkCudaErrors(cudaMemcpyPeerAsync(
            allDevices[g].macroInterfaceGPU[g].uy.auxZ_0,
            GPUS_TO_USE[g],
            allDevices[gNext].macroInterfaceGPU[gNext].uy.Z_0,
            GPUS_TO_USE[gNext],
            MacroHaloSize, streamLBM
        ));

        checkCudaErrors(cudaMemcpyPeerAsync(
            allDevices[g].macroInterfaceGPU[g].uz.auxZ_0,
            GPUS_TO_USE[g],
            allDevices[gNext].macroInterfaceGPU[gNext].uz.Z_0,
            GPUS_TO_USE[gNext],
            MacroHaloSize, streamLBM
        ));

        #ifdef SECOND_DIST
        checkCudaErrors(cudaMemcpyPeerAsync(
            allDevices[g].macroInterfaceGPU[g].g.auxZ_0,
            GPUS_TO_USE[g],
            allDevices[gNext].macroInterfaceGPU[gNext].g.Z_0,
            GPUS_TO_USE[gNext],
            MacroHaloSize, streamLBM
        ));
        #endif //SECOND_DIST
        #ifdef LAMBDA_DIST
        checkCudaErrors(cudaMemcpyPeerAsync(
            allDevices[g].macroInterfaceGPU[g].lambda.auxZ_0,
            GPUS_TO_USE[g],
            allDevices[gNext].macroInterfaceGPU[gNext].lambda.Z_0,
            GPUS_TO_USE[gNext],
            MacroHaloSize, streamLBM
        ));
        #endif //LAMBDA_DIST
        #ifdef A_XX_DIST
        checkCudaErrors(cudaMemcpyPeerAsync(
            allDevices[g].macroInterfaceGPU[g].Axx.auxZ_0,
            GPUS_TO_USE[g],
            allDevices[gNext].macroInterfaceGPU[gNext].Axx.Z_0,
            GPUS_TO_USE[gNext],
            MacroHaloSize, streamLBM
        ));
        #endif //A_XX_DIST
        #ifdef A_XY_DIST
        checkCudaErrors(cudaMemcpyPeerAsync(
            allDevices[g].macroInterfaceGPU[g].Axy.auxZ_0,
            GPUS_TO_USE[g],
            allDevices[gNext].macroInterfaceGPU[gNext].Axy.Z_0,
            GPUS_TO_USE[gNext],
            MacroHaloSize, streamLBM
        ));
        #endif //A_XY_DIST
        #ifdef A_XZ_DIST
        checkCudaErrors(cudaMemcpyPeerAsync(
            allDevices[g].macroInterfaceGPU[g].Axz.auxZ_0,
            GPUS_TO_USE[g],
            allDevices[gNext].macroInterfaceGPU[gNext].Axz.Z_0,
            GPUS_TO_USE[gNext],
            MacroHaloSize, streamLBM
        ));
        #endif //A_XZ_DIST
        #ifdef A_YY_DIST
        checkCudaErrors(cudaMemcpyPeerAsync(
            allDevices[g].macroInterfaceGPU[g].Ayy.auxZ_0,
            GPUS_TO_USE[g],
            allDevices[gNext].macroInterfaceGPU[gNext].Ayy.Z_0,
            GPUS_TO_USE[gNext],
            MacroHaloSize, streamLBM
        ));
        #endif //A_YY_DIST
        #ifdef A_YZ_DIST
        checkCudaErrors(cudaMemcpyPeerAsync(
            allDevices[g].macroInterfaceGPU[g].Ayz.auxZ_0,
            GPUS_TO_USE[g],
            allDevices[gNext].macroInterfaceGPU[gNext].Ayz.Z_0,
            GPUS_TO_USE[gNext],
            MacroHaloSize, streamLBM
        ));
        #endif //A_YZ_DIST
        #ifdef A_ZZ_DIST
        checkCudaErrors(cudaMemcpyPeerAsync(
            allDevices[g].macroInterfaceGPU[g].Azz.auxZ_0,
            GPUS_TO_USE[g],
            allDevices[gNext].macroInterfaceGPU[gNext].Azz.Z_0,
            GPUS_TO_USE[gNext],
            MacroHaloSize, streamLBM
        ));
        #endif //A_ZZ_DIST

        #ifdef PHI_DIST
        checkCudaErrors(cudaMemcpyPeerAsync(
            allDevices[g].macroInterfaceGPU[g].phi.auxZ_0,
            GPUS_TO_USE[g],
            allDevices[gNext].macroInterfaceGPU[gNext].phi.Z_0,
            GPUS_TO_USE[gNext],
            MacroHaloSize, streamLBM
        ));
        checkCudaErrors(cudaMemcpyPeerAsync(
            allDevices[g].macroInterfaceGPU[g].nx.auxZ_0,
            GPUS_TO_USE[g],
            allDevices[gNext].macroInterfaceGPU[gNext].nx.Z_0,
            GPUS_TO_USE[gNext],
            MacroHaloSize, streamLBM
        ));
        checkCudaErrors(cudaMemcpyPeerAsync(
            allDevices[g].macroInterfaceGPU[g].ny.auxZ_0,
            GPUS_TO_USE[g],
            allDevices[gNext].macroInterfaceGPU[gNext].ny.Z_0,
            GPUS_TO_USE[gNext],
            MacroHaloSize, streamLBM
        ));
        checkCudaErrors(cudaMemcpyPeerAsync(
            allDevices[g].macroInterfaceGPU[g].nz.auxZ_0,
            GPUS_TO_USE[g],
            allDevices[gNext].macroInterfaceGPU[gNext].nz.Z_0,
            GPUS_TO_USE[gNext],
            MacroHaloSize, streamLBM
        ));
        checkCudaErrors(cudaMemcpyPeerAsync(
            allDevices[g].macroInterfaceGPU[g].mu.auxZ_0,
            GPUS_TO_USE[g],
            allDevices[gNext].macroInterfaceGPU[gNext].mu.Z_0,
            GPUS_TO_USE[gNext],
            MacroHaloSize, streamLBM
        ));
        #endif //PHI_DIST

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

        #if defined(SECOND_DIST) || defined(PHI_DIST) || defined(LAMBDA_DIST) || \
            defined(A_XX_DIST) || defined(A_XY_DIST) || defined(A_XZ_DIST) || \
            defined(A_YY_DIST) || defined(A_YZ_DIST) || defined(A_ZZ_DIST)
        const size_t scalarPlaneSize = (size_t)BLOCK_NX * BLOCK_NY * NUM_BLOCK_X * NUM_BLOCK_Y * GF;
        const size_t scalarHaloSize = scalarPlaneSize * sizeof(dfloat);
        const size_t scalarTopOffset = (size_t)(NUM_BLOCK_Z_LOCAL - 1) * scalarPlaneSize;
        #endif

        #ifdef SECOND_DIST
        checkCudaErrors(cudaMemcpyPeerAsync(
            allDevices[gNext].ghostInterface[gNext].gAux.Z_1,
            GPUS_TO_USE[gNext],
            allDevices[g].ghostInterface[g].g.Z_1 + scalarTopOffset,
            GPUS_TO_USE[g],
            scalarHaloSize, streamLBM
        ));
        #endif //SECOND_DIST

        #ifdef PHI_DIST
        checkCudaErrors(cudaMemcpyPeerAsync(
            allDevices[gNext].ghostInterface[gNext].phiAux.Z_1,
            GPUS_TO_USE[gNext],
            allDevices[g].ghostInterface[g].phi.Z_1 + scalarTopOffset,
            GPUS_TO_USE[g],
            scalarHaloSize, streamLBM
        ));
        #endif //PHI_DIST

        #ifdef LAMBDA_DIST
        checkCudaErrors(cudaMemcpyPeerAsync(
            allDevices[gNext].ghostInterface[gNext].lambdaAux.Z_1,
            GPUS_TO_USE[gNext],
            allDevices[g].ghostInterface[g].lambda.Z_1 + scalarTopOffset,
            GPUS_TO_USE[g],
            scalarHaloSize, streamLBM
        ));
        #endif //LAMBDA_DIST

        #ifdef A_XX_DIST
        checkCudaErrors(cudaMemcpyPeerAsync(
            allDevices[gNext].ghostInterface[gNext].AxxAux.Z_1,
            GPUS_TO_USE[gNext],
            allDevices[g].ghostInterface[g].Axx.Z_1 + scalarTopOffset,
            GPUS_TO_USE[g],
            scalarHaloSize, streamLBM
        ));
        #endif //A_XX_DIST

        #ifdef A_XY_DIST
        checkCudaErrors(cudaMemcpyPeerAsync(
            allDevices[gNext].ghostInterface[gNext].AxyAux.Z_1,
            GPUS_TO_USE[gNext],
            allDevices[g].ghostInterface[g].Axy.Z_1 + scalarTopOffset,
            GPUS_TO_USE[g],
            scalarHaloSize, streamLBM
        ));
        #endif //A_XY_DIST

        #ifdef A_XZ_DIST
        checkCudaErrors(cudaMemcpyPeerAsync(
            allDevices[gNext].ghostInterface[gNext].AxzAux.Z_1,
            GPUS_TO_USE[gNext],
            allDevices[g].ghostInterface[g].Axz.Z_1 + scalarTopOffset,
            GPUS_TO_USE[g],
            scalarHaloSize, streamLBM
        ));
        #endif //A_XZ_DIST

        #ifdef A_YY_DIST
        checkCudaErrors(cudaMemcpyPeerAsync(
            allDevices[gNext].ghostInterface[gNext].AyyAux.Z_1,
            GPUS_TO_USE[gNext],
            allDevices[g].ghostInterface[g].Ayy.Z_1 + scalarTopOffset,
            GPUS_TO_USE[g],
            scalarHaloSize, streamLBM
        ));
        #endif //A_YY_DIST

        #ifdef A_YZ_DIST
        checkCudaErrors(cudaMemcpyPeerAsync(
            allDevices[gNext].ghostInterface[gNext].AyzAux.Z_1,
            GPUS_TO_USE[gNext],
            allDevices[g].ghostInterface[g].Ayz.Z_1 + scalarTopOffset,
            GPUS_TO_USE[g],
            scalarHaloSize, streamLBM
        ));
        #endif //A_YZ_DIST

        #ifdef A_ZZ_DIST
        checkCudaErrors(cudaMemcpyPeerAsync(
            allDevices[gNext].ghostInterface[gNext].AzzAux.Z_1,
            GPUS_TO_USE[gNext],
            allDevices[g].ghostInterface[g].Azz.Z_1 + scalarTopOffset,
            GPUS_TO_USE[g],
            scalarHaloSize, streamLBM
        ));
        #endif //A_ZZ_DIST
    }

    /* -------------- Receives the Z_0 base of the next GPU and adds it to the Z_0 top of the current GPU  ------------- */
    void sendBottomToPrev(int g, deviceField* allDevices, cudaStream_t streamLBM)
    {
        checkCudaErrors(cudaSetDevice(GPUS_TO_USE[g]));
        const int gNext = (g + 1) % N_GPUS;
        const size_t planeSize = (size_t)BLOCK_NX * BLOCK_NY * NUM_BLOCK_X * NUM_BLOCK_Y * QF;
        const size_t haloSize  = planeSize * sizeof(dfloat);


        checkCudaErrors(cudaMemcpyPeerAsync(
            allDevices[g].ghostInterface[g].popAux.Z_0,
            GPUS_TO_USE[g],
            allDevices[gNext].ghostInterface[gNext].pop.Z_0,
            GPUS_TO_USE[gNext],
            haloSize, streamLBM
        ));

        #if defined(SECOND_DIST) || defined(PHI_DIST) || defined(LAMBDA_DIST) || \
            defined(A_XX_DIST) || defined(A_XY_DIST) || defined(A_XZ_DIST) || \
            defined(A_YY_DIST) || defined(A_YZ_DIST) || defined(A_ZZ_DIST)
        const size_t scalarPlaneSize = (size_t)BLOCK_NX * BLOCK_NY * NUM_BLOCK_X * NUM_BLOCK_Y * GF;
        const size_t scalarHaloSize = scalarPlaneSize * sizeof(dfloat);
        #endif

        #ifdef SECOND_DIST
        checkCudaErrors(cudaMemcpyPeerAsync(
            allDevices[g].ghostInterface[g].gAux.Z_0,
            GPUS_TO_USE[g],
            allDevices[gNext].ghostInterface[gNext].g.Z_0,
            GPUS_TO_USE[gNext],
            scalarHaloSize, streamLBM
        ));
        #endif //SECOND_DIST

        #ifdef PHI_DIST
        checkCudaErrors(cudaMemcpyPeerAsync(
            allDevices[g].ghostInterface[g].phiAux.Z_0,
            GPUS_TO_USE[g],
            allDevices[gNext].ghostInterface[gNext].phi.Z_0,
            GPUS_TO_USE[gNext],
            scalarHaloSize, streamLBM
        ));
        #endif //PHI_DIST

        #ifdef LAMBDA_DIST
        checkCudaErrors(cudaMemcpyPeerAsync(
            allDevices[g].ghostInterface[g].lambdaAux.Z_0,
            GPUS_TO_USE[g],
            allDevices[gNext].ghostInterface[gNext].lambda.Z_0,
            GPUS_TO_USE[gNext],
            scalarHaloSize, streamLBM
        ));
        #endif //LAMBDA_DIST

        #ifdef A_XX_DIST
        checkCudaErrors(cudaMemcpyPeerAsync(
            allDevices[g].ghostInterface[g].AxxAux.Z_0,
            GPUS_TO_USE[g],
            allDevices[gNext].ghostInterface[gNext].Axx.Z_0,
            GPUS_TO_USE[gNext],
            scalarHaloSize, streamLBM
        ));
        #endif //A_XX_DIST
        #ifdef A_XY_DIST
        checkCudaErrors(cudaMemcpyPeerAsync(
            allDevices[g].ghostInterface[g].AxyAux.Z_0,
            GPUS_TO_USE[g],
            allDevices[gNext].ghostInterface[gNext].Axy.Z_0,
            GPUS_TO_USE[gNext],
            scalarHaloSize, streamLBM
        ));
        #endif //A_XY_DIST
        #ifdef A_XZ_DIST
        checkCudaErrors(cudaMemcpyPeerAsync(
            allDevices[g].ghostInterface[g].AxzAux.Z_0,
            GPUS_TO_USE[g],
            allDevices[gNext].ghostInterface[gNext].Axz.Z_0,
            GPUS_TO_USE[gNext],
            scalarHaloSize, streamLBM
        ));
        #endif //A_XZ_DIST
        #ifdef A_YY_DIST
        checkCudaErrors(cudaMemcpyPeerAsync(
            allDevices[g].ghostInterface[g].AyyAux.Z_0,
            GPUS_TO_USE[g],
            allDevices[gNext].ghostInterface[gNext].Ayy.Z_0,
            GPUS_TO_USE[gNext],
            scalarHaloSize, streamLBM
        ));
        #endif //A_YY_DIST
        #ifdef A_YZ_DIST
        checkCudaErrors(cudaMemcpyPeerAsync(
            allDevices[g].ghostInterface[g].AyzAux.Z_0,
            GPUS_TO_USE[g],
            allDevices[gNext].ghostInterface[gNext].Ayz.Z_0,
            GPUS_TO_USE[gNext],
            scalarHaloSize, streamLBM
        ));
        #endif //A_YZ_DIST
        #ifdef A_ZZ_DIST
        checkCudaErrors(cudaMemcpyPeerAsync(
            allDevices[g].ghostInterface[g].AzzAux.Z_0,
            GPUS_TO_USE[g],
            allDevices[gNext].ghostInterface[gNext].Azz.Z_0,
            GPUS_TO_USE[gNext],
            scalarHaloSize, streamLBM
        ));
        #endif //A_ZZ_DIST
    }

    #ifdef PHI_DIST
    // Caller must exchange fresh PHI before this stage and fresh mu afterward.
    void computePhaseNormalsDeviceField(dim3 gridBlock, dim3 threadBlock, int g, int slice, cudaStream_t stream){
        checkCudaErrors(cudaSetDevice(GPUS_TO_USE[g]));
        int zStart = g * slice;
        int zEnd   = (g == N_GPUS - 1) ? NZ : zStart + slice;
        size_t localNZ = zEnd - zStart;
        gpuComputePhaseNormals<<<gridBlock, threadBlock, 0, stream>>>(d_fMom[g], dNodeType[g], macroInterfaceGPU[g], localNZ, zStart);
        gpuComputeChemicalPotential<<<gridBlock, threadBlock, 0, stream>>>(d_fMom[g], dNodeType[g], localNZ, zStart);
    }

    void gpuComputeLaplacianMuDeviceField(dim3 gridBlock, dim3 threadBlock, int g, int slice, cudaStream_t stream){
        checkCudaErrors(cudaSetDevice(GPUS_TO_USE[g]));
        int zStart = g * slice;
        int zEnd   = (g == N_GPUS - 1) ? NZ : zStart + slice;
        size_t localNZ = zEnd - zStart;
        gpuComputeLaplacianMu<<<gridBlock, threadBlock, 0, stream>>>(d_fMom[g], dNodeType[g], macroInterfaceGPU[g], localNZ, zStart);
    }
    
    #endif //PHI_DIST

    void swapGhostInterfacesDeviceField(int g){
        swapGhostInterfaces(ghostInterface[g]);
    }

    void halfStepKernels(dim3 gridBlock, dim3 threadBlock, size_t step, int g, int slice, cudaStream_t stream){
        checkCudaErrors(cudaSetDevice(GPUS_TO_USE[g]));
        #ifdef LOCAL_FORCES
            gpuResetMacroForcesDeviceField(gridBlock, threadBlock, g, stream);
            CHECK_KERNEL_ERR("Force Reset kernel");
        #endif //LOCAL_FORCES
        #ifdef CURVED_BOUNDARY_CONDITION
            updateCurvedBoundaryVelocitiesDeviceField(g, stream);
            CHECK_KERNEL_ERR("Curved BC kernel");
        #endif //CURVED_BOUNDARY_CONDITION
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

        // Copy local forces arrays if enabled
        #ifdef SAVE_LOCAL_FORCES
        checkCudaErrors(cudaMemcpy(hostField.h_Local_Fx, d_Local_Fx, MEM_SIZE_SCALAR, cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(hostField.h_Local_Fy, d_Local_Fy, MEM_SIZE_SCALAR, cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(hostField.h_Local_Fz, d_Local_Fz, MEM_SIZE_SCALAR, cudaMemcpyDeviceToHost));
            #ifdef SECOND_DIST
        checkCudaErrors(cudaMemcpy(hostField.h_Source_C, d_Source_C, MEM_SIZE_SCALAR, cudaMemcpyDeviceToHost));
            #endif
            #ifdef PHI_DIST
        checkCudaErrors(cudaMemcpy(hostField.h_Source_Phi, d_Source_Phi, MEM_SIZE_SCALAR, cudaMemcpyDeviceToHost));
            #endif
            #ifdef LAMBDA_DIST
        checkCudaErrors(cudaMemcpy(hostField.h_Source_Lambda, d_Source_Lambda, MEM_SIZE_SCALAR, cudaMemcpyDeviceToHost));
            #endif
            #ifdef CONFORMATION_TENSOR
                #ifdef A_XX_DIST
        checkCudaErrors(cudaMemcpy(hostField.h_Source_Gxx, d_Source_Gxx, MEM_SIZE_SCALAR, cudaMemcpyDeviceToHost));
                #endif
                #ifdef A_XY_DIST
        checkCudaErrors(cudaMemcpy(hostField.h_Source_Gxy, d_Source_Gxy, MEM_SIZE_SCALAR, cudaMemcpyDeviceToHost));
                #endif
                #ifdef A_XZ_DIST
        checkCudaErrors(cudaMemcpy(hostField.h_Source_Gxz, d_Source_Gxz, MEM_SIZE_SCALAR, cudaMemcpyDeviceToHost));
                #endif
                #ifdef A_YY_DIST
        checkCudaErrors(cudaMemcpy(hostField.h_Source_Gyy, d_Source_Gyy, MEM_SIZE_SCALAR, cudaMemcpyDeviceToHost));
                #endif
                #ifdef A_YZ_DIST
        checkCudaErrors(cudaMemcpy(hostField.h_Source_Gyz, d_Source_Gyz, MEM_SIZE_SCALAR, cudaMemcpyDeviceToHost));
                #endif
                #ifdef A_ZZ_DIST
        checkCudaErrors(cudaMemcpy(hostField.h_Source_Gzz, d_Source_Gzz, MEM_SIZE_SCALAR, cudaMemcpyDeviceToHost));
                #endif
            #endif //CONFORMATION_TENSOR
        #endif //SAVE_LOCAL_FORCES
    }

    void saveSimCheckpointHostDeviceField(hostField &hostField, int &step, int g, int slice){
        checkCudaErrors(cudaSetDevice(GPUS_TO_USE[g]));
        int zStart = g * slice;
        size_t zOffset = zStart * NX * NY * NUMBER_MOMENTS;
        saveSimCheckpoint(hostField.h_fMom+zOffset, ghostInterface[g], &step, g);
    }

    void saveSimCheckpointDeviceField( int &step, int g, int slice){
        checkCudaErrors(cudaSetDevice(GPUS_TO_USE[g]));
        saveSimCheckpoint(d_fMom[g],ghostInterface[g],&step, g);
    }

    #ifdef TREAT_DATA_INCLUDE
    void copyMacroscopicDeviceField(hostField &hostField, deviceField* allDevices, int slice){
        for(int g = 0; g < N_GPUS; g++){
            checkCudaErrors(cudaSetDevice(GPUS_TO_USE[g]));
            checkCudaErrors(cudaDeviceSynchronize());
            int zStart = g * slice;
            size_t zOffset = zStart * NX * NY * NUMBER_MOMENTS;
            size_t zOffsetScalar = zStart * NX * NY;
            checkCudaErrors(cudaMemcpy(hostField.h_fMom+zOffset, allDevices[g].d_fMom[g], sizeof(dfloat) * NUMBER_LBM_NODES_LOCAL*NUMBER_MOMENTS, cudaMemcpyDeviceToHost));

            // Local forces / source terms (scalar fields, assembled on host)
            #ifdef SAVE_LOCAL_FORCES
            checkCudaErrors(cudaMemcpy(hostField.h_Local_Fx+zOffsetScalar, allDevices[g].d_Local_Fx, sizeof(dfloat) * NUMBER_LBM_NODES_LOCAL, cudaMemcpyDeviceToHost));
            checkCudaErrors(cudaMemcpy(hostField.h_Local_Fy+zOffsetScalar, allDevices[g].d_Local_Fy, sizeof(dfloat) * NUMBER_LBM_NODES_LOCAL, cudaMemcpyDeviceToHost));
            checkCudaErrors(cudaMemcpy(hostField.h_Local_Fz+zOffsetScalar, allDevices[g].d_Local_Fz, sizeof(dfloat) * NUMBER_LBM_NODES_LOCAL, cudaMemcpyDeviceToHost));
                #ifdef SECOND_DIST
            checkCudaErrors(cudaMemcpy(hostField.h_Source_C+zOffsetScalar, allDevices[g].d_Source_C, sizeof(dfloat) * NUMBER_LBM_NODES_LOCAL, cudaMemcpyDeviceToHost));
                #endif
                #ifdef PHI_DIST
            checkCudaErrors(cudaMemcpy(hostField.h_Source_Phi+zOffsetScalar, allDevices[g].d_Source_Phi, sizeof(dfloat) * NUMBER_LBM_NODES_LOCAL, cudaMemcpyDeviceToHost));
                #endif
                #ifdef LAMBDA_DIST
            checkCudaErrors(cudaMemcpy(hostField.h_Source_Lambda+zOffsetScalar, allDevices[g].d_Source_Lambda, sizeof(dfloat) * NUMBER_LBM_NODES_LOCAL, cudaMemcpyDeviceToHost));
                #endif
                #ifdef CONFORMATION_TENSOR
                    #ifdef A_XX_DIST
            checkCudaErrors(cudaMemcpy(hostField.h_Source_Gxx+zOffsetScalar, allDevices[g].d_Source_Gxx, sizeof(dfloat) * NUMBER_LBM_NODES_LOCAL, cudaMemcpyDeviceToHost));
                    #endif
                    #ifdef A_XY_DIST
            checkCudaErrors(cudaMemcpy(hostField.h_Source_Gxy+zOffsetScalar, allDevices[g].d_Source_Gxy, sizeof(dfloat) * NUMBER_LBM_NODES_LOCAL, cudaMemcpyDeviceToHost));
                    #endif
                    #ifdef A_XZ_DIST
            checkCudaErrors(cudaMemcpy(hostField.h_Source_Gxz+zOffsetScalar, allDevices[g].d_Source_Gxz, sizeof(dfloat) * NUMBER_LBM_NODES_LOCAL, cudaMemcpyDeviceToHost));
                    #endif
                    #ifdef A_YY_DIST
            checkCudaErrors(cudaMemcpy(hostField.h_Source_Gyy+zOffsetScalar, allDevices[g].d_Source_Gyy, sizeof(dfloat) * NUMBER_LBM_NODES_LOCAL, cudaMemcpyDeviceToHost));
                    #endif
                    #ifdef A_YZ_DIST
            checkCudaErrors(cudaMemcpy(hostField.h_Source_Gyz+zOffsetScalar, allDevices[g].d_Source_Gyz, sizeof(dfloat) * NUMBER_LBM_NODES_LOCAL, cudaMemcpyDeviceToHost));
                    #endif
                    #ifdef A_ZZ_DIST
            checkCudaErrors(cudaMemcpy(hostField.h_Source_Gzz+zOffsetScalar, allDevices[g].d_Source_Gzz, sizeof(dfloat) * NUMBER_LBM_NODES_LOCAL, cudaMemcpyDeviceToHost));
                    #endif
                #endif //CONFORMATION_TENSOR
            #endif //SAVE_LOCAL_FORCES

            checkCudaErrors(cudaDeviceSynchronize());
        }
    }
    #endif //TREAT_DATA_INCLUDE

    void treatDataDeviceField(hostField &hostField, 
        int step, int slice){
        TreatDataParams treatDataParams;
        treatDataParams.h_fMom = hostField.h_fMom;
        #if MEAN_FLOW
        treatDataParams.d_fMom_mean = hostField.m_fMom;
        #endif
        #ifdef BC_FORCES
        treatDataParams.d_BC_Fx = nullptr;
        treatDataParams.d_BC_Fy = nullptr;
        treatDataParams.d_BC_Fz = nullptr;
        #endif
        #ifdef SAVE_LOCAL_FORCES
        treatDataParams.d_Local_Fx = hostField.h_Local_Fx;
        treatDataParams.d_Local_Fy = hostField.h_Local_Fy;
        treatDataParams.d_Local_Fz = hostField.h_Local_Fz;
            #ifdef SECOND_DIST
        treatDataParams.d_Source_C = hostField.h_Source_C;
            #endif
            #ifdef PHI_DIST
        treatDataParams.d_Source_Phi = hostField.h_Source_Phi;
            #endif
            #ifdef LAMBDA_DIST
        treatDataParams.d_Source_Lambda = hostField.h_Source_Lambda;
            #endif
            #ifdef CONFORMATION_TENSOR
                #ifdef A_XX_DIST
        treatDataParams.d_Source_Gxx = hostField.h_Source_Gxx;
                #endif
                #ifdef A_XY_DIST
        treatDataParams.d_Source_Gxy = hostField.h_Source_Gxy;
                #endif
                #ifdef A_XZ_DIST
        treatDataParams.d_Source_Gxz = hostField.h_Source_Gxz;
                #endif
                #ifdef A_YY_DIST
        treatDataParams.d_Source_Gyy = hostField.h_Source_Gyy;
                #endif
                #ifdef A_YZ_DIST
        treatDataParams.d_Source_Gyz = hostField.h_Source_Gyz;
                #endif
                #ifdef A_ZZ_DIST
        treatDataParams.d_Source_Gzz = hostField.h_Source_Gzz;
                #endif
            #endif //CONFORMATION_TENSOR
        #endif //SAVE_LOCAL_FORCES
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
        interfaceFree(ghostInterface[g], macroInterfaceGPU[g]);

        #ifdef CURVED_BOUNDARY_CONDITION
        cudaFree(d_curvedBC[g]);
        cudaFree(d_curvedBC_array[g]);
        d_curvedBC[g] = nullptr;
        d_curvedBC_array[g] = nullptr;
        #endif

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

        #ifdef SAVE_LOCAL_FORCES
        cudaFree(d_Local_Fx);
        cudaFree(d_Local_Fy);
        cudaFree(d_Local_Fz);
            #ifdef SECOND_DIST
        cudaFree(d_Source_C);
            #endif
            #ifdef PHI_DIST
        cudaFree(d_Source_Phi);
            #endif
            #ifdef LAMBDA_DIST
        cudaFree(d_Source_Lambda);
            #endif
            #ifdef CONFORMATION_TENSOR
                #ifdef A_XX_DIST
        cudaFree(d_Source_Gxx);
                #endif
                #ifdef A_XY_DIST
        cudaFree(d_Source_Gxy);
                #endif
                #ifdef A_XZ_DIST
        cudaFree(d_Source_Gxz);
                #endif
                #ifdef A_YY_DIST
        cudaFree(d_Source_Gyy);
                #endif
                #ifdef A_YZ_DIST
        cudaFree(d_Source_Gyz);
                #endif
                #ifdef A_ZZ_DIST
        cudaFree(d_Source_Gzz);
                #endif
            #endif //CONFORMATION_TENSOR
        #endif //SAVE_LOCAL_FORCES
    }
} DeviceField;

#endif //__DEVICEFIELD_STRUCTS_H
