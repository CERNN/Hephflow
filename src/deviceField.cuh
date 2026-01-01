#ifndef __DEVICEFIELD_STRUCTS_H
#define __DEVICEFIELD_STRUCTS_H

#include "var.h"
#include "main.cuh"
#include "hostField.cuh"

typedef struct deviceField{
    ghostInterfaceData ghostInterface;

    dfloat* d_fMom;
    unsigned int* dNodeType;

    #ifdef CURVED_BOUNDARY_CONDITION
    CurvedBoundary** d_curvedBC;
    CurvedBoundary* d_curvedBC_array;
    unsigned int numberCurvedBoundaryNodes;  // Number of curved boundary nodes
    #endif

    #ifdef DENSITY_CORRECTION
    dfloat* d_mean_rho;
    #endif //DENSITY_CORRECTION

    #ifdef BC_FORCES
        dfloat* d_BC_Fx;
        dfloat* d_BC_Fy;
        dfloat* d_BC_Fz;
    #endif //_BC_FORCES

    void allocateDeviceMemoryDeviceField() {
        unsigned int memAllocated = 0;

        cudaMalloc((void**)&d_fMom, MEM_SIZE_MOM);
        cudaMalloc((void**)&dNodeType, sizeof(int) * NUMBER_LBM_NODES);
        interfaceMalloc(ghostInterface);

        memAllocated += MEM_SIZE_MOM + sizeof(int) * NUMBER_LBM_NODES;

        #ifdef BC_FORCES
        cudaMalloc((void**)&d_BC_Fx, MEM_SIZE_SCALAR);
        cudaMalloc((void**)&d_BC_Fy, MEM_SIZE_SCALAR);
        cudaMalloc((void**)&d_BC_Fz, MEM_SIZE_SCALAR);
        memAllocated += 3 * MEM_SIZE_SCALAR;
        #endif //BC_FORCES

        printf("Device Memory Allocated for Bulk flow: %.2f MB \n", (float)memAllocated /(1024.0 * 1024.0));
    }

    #ifdef DENSITY_CORRECTION
    void allocateDensityCorrectionMemory(){
        cudaMalloc((void**)&d_mean_rho, sizeof(dfloat));
    }
    #endif //DENSITY_CORRECTION

    void initializeDomainDeviceField(hostField &hostField, dfloat **&randomNumbers, int &step, dim3 gridBlock, dim3 threadBlock){
        // ========== INITIALIZATION DOMAIN INLINED ==========
        
        // Random numbers initialization
        #ifdef RANDOM_NUMBERS 
            if(console_flush) fflush(stdout);
            checkCudaErrors(cudaMallocManaged((void**)&randomNumbers[0], sizeof(dfloat) * NUMBER_LBM_NODES));
            initializationRandomNumbers(randomNumbers[0], CURAND_SEED);
            checkCudaErrors(cudaDeviceSynchronize());
            getLastCudaError("random numbers transfer error");
            printf("Random numbers initialized - Seed used: %u\n", CURAND_SEED); 
            printf("Device memory allocated for random numbers: %.2f MB\n", (float)(sizeof(dfloat) * NUMBER_LBM_NODES) / (1024.0 * 1024.0));
            if(console_flush) fflush(stdout);
        #endif //RANDOM_NUMBERS

        int checkpoint_state = 0;
        // LBM Initialization
        if (LOAD_CHECKPOINT) {

            printf("Loading checkpoint\n");
            checkpoint_state = loadSimCheckpoint(hostField.h_fMom, ghostInterface, &step);

            if (checkpoint_state != 0){
                checkCudaErrors(cudaMemcpy(d_fMom, hostField.h_fMom, sizeof(dfloat) * NUMBER_LBM_NODES * NUMBER_MOMENTS, cudaMemcpyHostToDevice));
                interfaceCudaMemcpy(ghostInterface, ghostInterface.fGhost, ghostInterface.h_fGhost, cudaMemcpyHostToDevice, QF);

                #ifdef SECOND_DIST
                    interfaceCudaMemcpy(ghostInterface, ghostInterface.g_fGhost, ghostInterface.g_h_fGhost, cudaMemcpyHostToDevice, GF);
                #endif //SECOND_DIST

                #ifdef PHI_DIST
                    interfaceCudaMemcpy(ghostInterface, ghostInterface.phi_fGhost, ghostInterface.phi_h_fGhost, cudaMemcpyHostToDevice, GF);
                #endif //PHI_DIST

                #ifdef A_XX_DIST
                    interfaceCudaMemcpy(ghostInterface, ghostInterface.Axx_fGhost, ghostInterface.Axx_h_fGhost, cudaMemcpyHostToDevice, GF);
                #endif //A_XX_DIST
                #ifdef A_XY_DIST
                    interfaceCudaMemcpy(ghostInterface, ghostInterface.Axy_fGhost, ghostInterface.Axy_h_fGhost, cudaMemcpyHostToDevice, GF);
                #endif //A_XY_DIST
                #ifdef A_XZ_DIST
                    interfaceCudaMemcpy(ghostInterface, ghostInterface.Axz_fGhost, ghostInterface.Axz_h_fGhost, cudaMemcpyHostToDevice, GF);
                #endif //A_XZ_DIST
                #ifdef A_YY_DIST
                    interfaceCudaMemcpy(ghostInterface, ghostInterface.Ayy_fGhost, ghostInterface.Ayy_h_fGhost, cudaMemcpyHostToDevice, GF);
                #endif //A_YY_DIST
                #ifdef A_YZ_DIST
                    interfaceCudaMemcpy(ghostInterface, ghostInterface.Ayz_fGhost, ghostInterface.Ayz_h_fGhost, cudaMemcpyHostToDevice, GF);
                #endif //A_YZ_DIST
                #ifdef A_ZZ_DIST
                    interfaceCudaMemcpy(ghostInterface, ghostInterface.Azz_fGhost, ghostInterface.Azz_h_fGhost, cudaMemcpyHostToDevice, GF);
                #endif //A_ZZ_DIST
            }
        } 
        if (!checkpoint_state) {
            if (LOAD_FIELD) {
                // Implement LOAD_FIELD logic if needed
            } else {
                gpuInitialization_mom<<<gridBlock, threadBlock>>>(d_fMom, randomNumbers[0]);
            }
            gpuInitialization_pop<<<gridBlock, threadBlock>>>(d_fMom, ghostInterface);
        }

        // Mean flow initialization
        #if MEAN_FLOW
            checkCudaErrors(cudaMemcpy(hostField.m_fMom, d_fMom, sizeof(dfloat) * NUMBER_LBM_NODES * NUMBER_MOMENTS, cudaMemcpyDeviceToDevice));
        #endif //MEAN_FLOW

        // Node type initialization
        checkCudaErrors(cudaMallocHost((void**)&hostField.hNodeType, sizeof(unsigned int) * NUMBER_LBM_NODES));
        #if NODE_TYPE_SAVE
            checkCudaErrors(cudaMallocHost((void**)&dNodeType, sizeof(unsigned int) * NUMBER_LBM_NODES));
        #endif //NODE_TYPE_SAVE

        unsigned int numberCurvedBoundaryNodes_local = 0;

        #ifndef VOXEL_FILENAME
            hostInitialization_nodeType(hostField.hNodeType
            #ifdef CURVED_BOUNDARY_CONDITION
            ,&numberCurvedBoundaryNodes_local
            #endif
            );
            checkCudaErrors(cudaMemcpy(dNodeType, hostField.hNodeType, sizeof(unsigned int) * NUMBER_LBM_NODES, cudaMemcpyHostToDevice));  
            checkCudaErrors(cudaDeviceSynchronize());
            #ifdef FORCE_VOXEL_BC_BUILDING
                define_voxel_bc<<<gridBlock, threadBlock>>>(dNodeType); 
                checkCudaErrors(cudaMemcpy(hostField.hNodeType, dNodeType, sizeof(unsigned int) * NUMBER_LBM_NODES, cudaMemcpyDeviceToHost)); 
            #endif
        #else
            hostInitialization_nodeType_bulk(hostField.hNodeType); 
            read_xyz_file(VOXEL_FILENAME, hostField.hNodeType);
            hostInitialization_nodeType(hostField.hNodeType
            #ifdef CURVED_BOUNDARY_CONDITION
            ,&numberCurvedBoundaryNodes_local
            #endif
            );
            checkCudaErrors(cudaMemcpy(dNodeType, hostField.hNodeType, sizeof(unsigned int) * NUMBER_LBM_NODES, cudaMemcpyHostToDevice));  
            checkCudaErrors(cudaDeviceSynchronize());
            define_voxel_bc<<<gridBlock, threadBlock>>>(dNodeType); 
            checkCudaErrors(cudaMemcpy(hostField.hNodeType, dNodeType, sizeof(unsigned int) * NUMBER_LBM_NODES, cudaMemcpyDeviceToHost)); 
        #endif //!VOXEL_FILENAME

        // Boundary condition forces initialization
        #ifdef BC_FORCES
            gpuInitialization_force<<<gridBlock, threadBlock>>>(d_BC_Fx, d_BC_Fy, d_BC_Fz);
        #endif //BC_FORCES

        #ifdef CURVED_BOUNDARY_CONDITION
            initializeCurvedBoundaryDeviceField(
                hostField.hNodeType,
                dNodeType,
                d_curvedBC,
                d_curvedBC_array
            );
        #endif

        // Interface population initialization
        interfaceCudaMemcpy(ghostInterface, ghostInterface.gGhost, ghostInterface.fGhost, cudaMemcpyDeviceToDevice, QF);
        #ifdef SECOND_DIST
            interfaceCudaMemcpy(ghostInterface, ghostInterface.g_gGhost, ghostInterface.g_fGhost, cudaMemcpyDeviceToDevice, GF);
        #endif //SECOND_DIST
        #ifdef PHI_DIST
            interfaceCudaMemcpy(ghostInterface, ghostInterface.phi_gGhost, ghostInterface.phi_fGhost, cudaMemcpyDeviceToDevice, GF);
        #endif //PHI_DIST
        #ifdef A_XX_DIST
            interfaceCudaMemcpy(ghostInterface, ghostInterface.Axx_gGhost, ghostInterface.Axx_fGhost, cudaMemcpyDeviceToDevice, GF);
        #endif //A_XX_DIST
        #ifdef A_XY_DIST
            interfaceCudaMemcpy(ghostInterface, ghostInterface.Axy_gGhost, ghostInterface.Axy_fGhost, cudaMemcpyDeviceToDevice, GF);
        #endif //A_XY_DIST
        #ifdef A_XZ_DIST
            interfaceCudaMemcpy(ghostInterface, ghostInterface.Axz_gGhost, ghostInterface.Axz_fGhost, cudaMemcpyDeviceToDevice, GF);
        #endif //A_XZ_DIST
        #ifdef A_YY_DIST
            interfaceCudaMemcpy(ghostInterface, ghostInterface.Ayy_gGhost, ghostInterface.Ayy_fGhost, cudaMemcpyDeviceToDevice, GF);
        #endif //A_YY_DIST
        #ifdef A_YZ_DIST
            interfaceCudaMemcpy(ghostInterface, ghostInterface.Ayz_gGhost, ghostInterface.Ayz_fGhost, cudaMemcpyDeviceToDevice, GF);
        #endif //A_YZ_DIST
        #ifdef A_ZZ_DIST
            interfaceCudaMemcpy(ghostInterface, ghostInterface.Azz_gGhost, ghostInterface.Azz_fGhost, cudaMemcpyDeviceToDevice, GF);
        #endif //A_ZZ_DIST
        
        // Synchronize after all initializations
        checkCudaErrors(cudaDeviceSynchronize());

        // Synchronize and transfer data back to host if needed
        checkCudaErrors(cudaDeviceSynchronize());
        checkCudaErrors(cudaMemcpy(hostField.h_fMom, d_fMom, sizeof(dfloat) * NUMBER_LBM_NODES * NUMBER_MOMENTS, cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaDeviceSynchronize());
        printf("Synchorizing data back to host \n"); if(console_flush) fflush(stdout);

        // Free random numbers if initialized
        #ifdef RANDOM_NUMBERS
            checkCudaErrors(cudaSetDevice(GPU_INDEX));
            cudaFree(randomNumbers[0]);
            free(randomNumbers);
            printf("Random numbers free \n"); if(console_flush) fflush(stdout);
        #endif //RANDOM_NUMBERS
        
        #ifdef CURVED_BOUNDARY_CONDITION
            // Initialize curved boundary node count
            numberCurvedBoundaryNodes = numberCurvedBoundaryNodes_local;
        #endif //CURVED_BOUNDARY_CONDITION
    }

    #ifdef CURVED_BOUNDARY_CONDITION
    void updateCurvedBoundaryVelocitiesDeviceField(){
        const int curvedBCBlockSize = 256;
        const int curvedBCGridSize = (numberCurvedBoundaryNodes + curvedBCBlockSize - 1) / curvedBCBlockSize;
        updateCurvedBoundaryVelocities<<<curvedBCGridSize, curvedBCBlockSize>>>(d_curvedBC_array, d_fMom, numberCurvedBoundaryNodes);
    }
    #endif //CURVED_BOUNDARY_CONDITION

    #ifdef DENSITY_CORRECTION
    void mean_rhoDeviceField(size_t step){
        mean_rho(d_fMom,step,d_mean_rho);
    }
    #endif //DENSITY_CORRECTION

    void gpuMomCollisionStreamDeviceField(dim3 gridBlock, dim3 threadBlock, unsigned int step, bool save){
        // Create parameter struct and pass by value (most efficient!)
        DeviceKernelParams params;
        params.fMom = d_fMom;
        params.dNodeType = dNodeType;
        params.ghostInterface = ghostInterface;
        params.step = step;
        params.save = save;
        
        #ifdef DENSITY_CORRECTION
        params.d_mean_rho = d_mean_rho;
        #endif //DENSITY_CORRECTION
        
        #ifdef BC_FORCES
        params.d_BC_Fx = d_BC_Fx;
        params.d_BC_Fy = d_BC_Fy;
        params.d_BC_Fz = d_BC_Fz;
        #endif //BC_FORCES
        
        #ifdef CURVED_BOUNDARY_CONDITION
        params.d_curvedBC = d_curvedBC;
        params.d_curvedBC_array = d_curvedBC_array;
        #endif //CURVED_BOUNDARY_CONDITION
        
        // Pass struct by value - CUDA handles this efficiently
        gpuMomCollisionStream<<<gridBlock, threadBlock DYNAMIC_SHARED_MEMORY_PARAMS>>>(params);
    }

    #ifdef PHI_DIST
    void computePhaseNormalsDeviceField(dim3 gridBlock, dim3 threadBlock){
        gpuComputePhaseNormals<<<gridBlock, threadBlock>>>(d_fMom, dNodeType);
        cudaDeviceSynchronize();
    }
    #endif //PHI_DIST

    void swapGhostInterfacesDeviceField(){
        swapGhostInterfaces(ghostInterface);
    }

    void halfStepKernels(dim3 gridBlock, dim3 threadBlock, size_t step){
        #ifdef LOCAL_FORCES
            gpuResetMacroForcesDeviceField(gridBlock, threadBlock);
            CHECK_KERNEL_ERR("Force Reset kernel");
        #endif //LOCAL_FORCES
        #ifdef CURVED_BOUNDARY_CONDITION
            updateCurvedBoundaryVelocitiesDeviceField();
            CHECK_KERNEL_ERR("Curved BC kernel");
        #endif //CURVED_BOUNDARY_CONDITION
        #ifdef PHI_DIST
            computePhaseNormalsDeviceField(gridBlock, threadBlock);
            CHECK_KERNEL_ERR("Phi gradients kernel");
        #endif //PHI_DIST
        #ifdef DENSITY_CORRECTION
            mean_rhoDeviceField(step);
            CHECK_KERNEL_ERR("Density correction kernel");
        #endif //DENSITY_CORRECTION
    }
    
    #ifdef LOCAL_FORCES
    void gpuResetMacroForcesDeviceField(dim3 gridBlock, dim3 threadBlock){
        gpuResetMacroForces<<<gridBlock, threadBlock>>>(d_fMom);
    }
    #endif //LOCAL_FORCES

    #ifdef PARTICLE_MODEL
    void particleSimulationDeviceField(ParticlesSoA &particlesSoA, cudaStream_t *streamsPart, ParticleWallForces *d_pwForces,unsigned int step){
        particleSimulation(&particlesSoA,d_fMom,streamsPart,d_pwForces,step);
    }
    #endif //PARTICLE_MODEL

    void interfaceCudaMemcpyDeviceField(bool fGhost){
        if (fGhost) {
            interfaceCudaMemcpy(ghostInterface, ghostInterface.h_fGhost, ghostInterface.fGhost, cudaMemcpyDeviceToHost, QF);
        } else {
            interfaceCudaMemcpy(ghostInterface, ghostInterface.h_fGhost, ghostInterface.gGhost, cudaMemcpyDeviceToHost, QF);
            
        }
        #ifdef SECOND_DIST 
        interfaceCudaMemcpy(ghostInterface,ghostInterface.g_h_fGhost,ghostInterface.g_fGhost,cudaMemcpyDeviceToHost,GF);
        #endif //SECOND_DIST
        #ifdef PHI_DIST 
        interfaceCudaMemcpy(ghostInterface,ghostInterface.phi_h_fGhost,ghostInterface.phi_fGhost,cudaMemcpyDeviceToHost,GF);
        #endif //PHI_DIST
        #ifdef A_XX_DIST 
        interfaceCudaMemcpy(ghostInterface,ghostInterface.Axx_h_fGhost,ghostInterface.Axx_fGhost,cudaMemcpyDeviceToHost,GF);
        #endif //A_XX_DIST     
        #ifdef A_XY_DIST 
        interfaceCudaMemcpy(ghostInterface,ghostInterface.Axy_h_fGhost,ghostInterface.Axy_fGhost,cudaMemcpyDeviceToHost,GF);
        #endif //A_XX_DIST        
        #ifdef A_XZ_DIST 
        interfaceCudaMemcpy(ghostInterface,ghostInterface.Axz_h_fGhost,ghostInterface.Axz_fGhost,cudaMemcpyDeviceToHost,GF);
        #endif //A_XZ_DIST
        #ifdef A_YY_DIST 
        interfaceCudaMemcpy(ghostInterface,ghostInterface.Ayy_h_fGhost,ghostInterface.Ayy_fGhost,cudaMemcpyDeviceToHost,GF);
        #endif //A_YY_DIST        
        #ifdef A_YZ_DIST 
        interfaceCudaMemcpy(ghostInterface,ghostInterface.Ayz_h_fGhost,ghostInterface.Ayz_fGhost,cudaMemcpyDeviceToHost,GF);
        #endif //A_YZ_DIST      
        #ifdef A_ZZ_DIST 
        interfaceCudaMemcpy(ghostInterface,ghostInterface.Azz_h_fGhost,ghostInterface.Azz_fGhost,cudaMemcpyDeviceToHost,GF);
        #endif //A_ZZ_DIST
    }

    void cudaMemcpyDeviceField(hostField &hostField){
        checkCudaErrors(cudaMemcpy(hostField.h_fMom, d_fMom, sizeof(dfloat) * NUMBER_LBM_NODES*NUMBER_MOMENTS, cudaMemcpyDeviceToHost));
        
        // Copy BC forces arrays if enabled
        #if defined BC_FORCES && defined SAVE_BC_FORCES
        checkCudaErrors(cudaMemcpy(hostField.h_BC_Fx, d_BC_Fx, MEM_SIZE_SCALAR, cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(hostField.h_BC_Fy, d_BC_Fy, MEM_SIZE_SCALAR, cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(hostField.h_BC_Fz, d_BC_Fz, MEM_SIZE_SCALAR, cudaMemcpyDeviceToHost));
        #endif //BC_FORCES && SAVE_BC_FORCES
    }

    void saveSimCheckpointHostDeviceField(hostField &hostField, int &step){
        saveSimCheckpoint(hostField.h_fMom, ghostInterface, &step);
    }

    void saveSimCheckpointDeviceField( int &step){
        saveSimCheckpoint(d_fMom,ghostInterface,&step);
    }

    void treatDataDeviceField(hostField &hostField, 
        int step){
        TreatDataParams treatDataParams;
        treatDataParams.h_fMom = hostField.h_fMom;
        treatDataParams.d_fMom = d_fMom;
        #if MEAN_FLOW
        treatDataParams.d_fMom_mean = hostField.m_fMom;
        #endif
        #ifdef BC_FORCES
        treatDataParams.d_BC_Fx = d_BC_Fx;
        treatDataParams.d_BC_Fy = d_BC_Fy;
        treatDataParams.d_BC_Fz = d_BC_Fz;
        #endif
        treatDataParams.step = step;
        treatData(&treatDataParams);
    }

    #if defined BC_FORCES && defined SAVE_BC_FORCES
    void saveBcForces(hostField &hostField){
        checkCudaErrors(cudaDeviceSynchronize()); 
        checkCudaErrors(cudaMemcpy(hostField.h_BC_Fx, d_BC_Fx, MEM_SIZE_SCALAR, cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(hostField.h_BC_Fy, d_BC_Fy, MEM_SIZE_SCALAR, cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(hostField.h_BC_Fz, d_BC_Fz, MEM_SIZE_SCALAR, cudaMemcpyDeviceToHost));
    }
    #endif //BC_FORCES && SAVE_BC_FORCES

    void freeDeviceField() {
        interfaceFree(ghostInterface);

        cudaFree(d_fMom);
        cudaFree(dNodeType);

        #ifdef DENSITY_CORRECTION
        cudaFree(d_mean_rho);
        #endif //DENSITY_CORRECTION

        #ifdef BC_FORCES
        cudaFree(d_BC_Fx);
        cudaFree(d_BC_Fy);
        cudaFree(d_BC_Fz);
        #endif //_BC_FORCES
    }
} DeviceField;

#endif //__DEVICEFIELD_STRUCTS_H