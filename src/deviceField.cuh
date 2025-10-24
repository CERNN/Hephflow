#ifndef __DEVICEFIELD_STRUCTS_H
#define __DEVICEFIELD_STRUCTS_H

#include "var.h"
#include "main.cuh"
#include "hostField.cuh"

typedef struct deviceField{
    dfloat* d_fMom;
    unsigned int* dNodeType;

    #ifdef DENSITY_CORRECTION
    dfloat* d_mean_rho;
    #endif //DENSITY_CORRECTION

    #ifdef BC_FORCES
        dfloat* d_BC_Fx;
        dfloat* d_BC_Fy;
        dfloat* d_BC_Fz;
    #endif //_BC_FORCES

    void allocateDeviceMemoryDeviceField(ghostInterfaceData& ghostInterface) {
        allocateDeviceMemory(
            &d_fMom, &dNodeType, &ghostInterface
            BC_FORCES_PARAMS_PTR(d_)
        );
    }

    void initializeDomain(
        GhostInterfaceData &ghostInterface, 
        hostField &hostField, 
        dfloat **&randomNumbers,
        DENSITY_CORRECTION_PARAMS_DECLARATION(&h_)
        DENSITY_CORRECTION_PARAMS_DECLARATION(&d_)
        int *step, dim3 gridBlock, dim3 threadBlock
        ){
        
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
            checkpoint_state = loadSimCheckpoint(hostField.h_fMom, ghostInterface, step);

            if (checkpoint_state != 0){
                checkCudaErrors(cudaMemcpy(d_fMom, hostField.h_fMom, sizeof(dfloat) * NUMBER_LBM_NODES * NUMBER_MOMENTS, cudaMemcpyHostToDevice));
                interfaceCudaMemcpy(ghostInterface, ghostInterface.fGhost, ghostInterface.h_fGhost, cudaMemcpyHostToDevice, QF);

                #ifdef SECOND_DIST
                    interfaceCudaMemcpy(ghostInterface, ghostInterface.g_fGhost, ghostInterface.g_h_fGhost, cudaMemcpyHostToDevice, GF);
                #endif //SECOND_DIST

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

        #ifndef VOXEL_FILENAME
            hostInitialization_nodeType(hostField.hNodeType);
            checkCudaErrors(cudaMemcpy(dNodeType, hostField.hNodeType, sizeof(unsigned int) * NUMBER_LBM_NODES, cudaMemcpyHostToDevice));  
            checkCudaErrors(cudaDeviceSynchronize());
        #else
            hostInitialization_nodeType_bulk(hostField.hNodeType); 
            read_xyz_file(VOXEL_FILENAME, hostField.hNodeType);
            hostInitialization_nodeType(hostField.hNodeType);
            checkCudaErrors(cudaMemcpy(dNodeType, hostField.hNodeType, sizeof(unsigned int) * NUMBER_LBM_NODES, cudaMemcpyHostToDevice));  
            checkCudaErrors(cudaDeviceSynchronize());
            define_voxel_bc<<<gridBlock, threadBlock>>>(dNodeType); 
            checkCudaErrors(cudaMemcpy(hostField.hNodeType, dNodeType, sizeof(unsigned int) * NUMBER_LBM_NODES, cudaMemcpyDeviceToHost)); 
        #endif //!VOXEL_FILENAME

        // Boundary condition forces initialization
        #ifdef BC_FORCES
            gpuInitialization_force<<<gridBlock, threadBlock>>>(d_BC_Fx, d_BC_Fy, d_BC_Fz);
        #endif //BC_FORCES

        // Interface population initialization
        interfaceCudaMemcpy(ghostInterface, ghostInterface.gGhost, ghostInterface.fGhost, cudaMemcpyDeviceToDevice, QF);
        #ifdef SECOND_DIST
            interfaceCudaMemcpy(ghostInterface, ghostInterface.g_gGhost, ghostInterface.g_fGhost, cudaMemcpyDeviceToDevice, GF);
            printf("Interface pop copied \n"); if(console_flush) fflush(stdout);
        #endif //SECOND_DIST
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
    }

    void saveBcForces(hostField &hostField){
        #if defined BC_FORCES && defined SAVE_BC_FORCES
        checkCudaErrors(cudaDeviceSynchronize()); 
        checkCudaErrors(cudaMemcpy(hostField.h_BC_Fx, d_BC_Fx, MEM_SIZE_SCALAR, cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(hostField.h_BC_Fy, d_BC_Fy, MEM_SIZE_SCALAR, cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(hostField.h_BC_Fz, d_BC_Fz, MEM_SIZE_SCALAR, cudaMemcpyDeviceToHost));
        #endif //BC_FORCES && SAVE_BC_FORCES
    }

    void freeDeviceField() {
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