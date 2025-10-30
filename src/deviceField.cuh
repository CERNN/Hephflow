#ifndef __DEVICEFIELD_STRUCTS_H
#define __DEVICEFIELD_STRUCTS_H

#include "var.h"
#include "main.cuh"
#include "hostField.cuh"

typedef struct deviceField{
    ghostInterfaceData ghostInterface;

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

    

    void allocateDeviceMemoryDeviceField() {
        allocateDeviceMemory(
            &d_fMom, &dNodeType, &ghostInterface
            BC_FORCES_PARAMS_PTR(d_)
        );
    }

    void initializeDomainDeviceField(hostField &hostField, dfloat **&randomNumbers, int &step, dim3 gridBlock, dim3 threadBlock){
        initializeDomain(ghostInterface,     
            d_fMom, hostField.h_fMom, 
            #if MEAN_FLOW
            hostField.m_fMom,
            #endif //MEAN_FLOW
            hostField.hNodeType, dNodeType, randomNumbers, 
            BC_FORCES_PARAMS(d_)
            DENSITY_CORRECTION_PARAMS(h_)
            DENSITY_CORRECTION_PARAMS(d_)
            &step, gridBlock, threadBlock);
    }

    void mean_rhoDeviceField(size_t step){
        #ifdef DENSITY_CORRECTION
        mean_rho(d_fMom,step,d_mean_rho);
        #endif //DENSITY_CORRECTION

    }

    void gpuMomCollisionStreamDeviceField(dim3 gridBlock, dim3 threadBlock, unsigned int step, bool save){
            gpuMomCollisionStream << <gridBlock, threadBlock DYNAMIC_SHARED_MEMORY_PARAMS>> >(d_fMom, dNodeType,ghostInterface, DENSITY_CORRECTION_PARAMS(d_) BC_FORCES_PARAMS(d_) step, save);
    }

    void swapGhostInterfacesDeviceField(){
        swapGhostInterfaces(ghostInterface);
    }

    void gpuResetMacroForcesDeviceField(dim3 gridBlock, dim3 threadBlock){
        #ifdef LOCAL_FORCES
        gpuResetMacroForces<<<gridBlock, threadBlock>>>(d_fMom);
        #endif //LOCAL_FORCES

    }

    void particleSimulationDeviceField(ParticlesSoA &particlesSoA, cudaStream_t *streamsPart, unsigned int step){
         #ifdef PARTICLE_MODEL
            particleSimulation(&particlesSoA,d_fMom,streamsPart,step);
        #endif //PARTICLE_MODEL
    }

    void interfaceCudaMemcpyDeviceField(bool fGhost){
        if (fGhost) {
            interfaceCudaMemcpy(ghostInterface, ghostInterface.h_fGhost, ghostInterface.fGhost, cudaMemcpyDeviceToHost, QF);
        } else {
            interfaceCudaMemcpy(ghostInterface, ghostInterface.h_fGhost, ghostInterface.gGhost, cudaMemcpyDeviceToHost, QF);
            
        }
        #ifdef SECOND_DIST 
        interfaceCudaMemcpy(ghostInterface,ghostInterface.g_h_fGhost,ghostInterface.g_fGhost,cudaMemcpyDeviceToHost,GF);
        #endif //SECOND_DIST
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
    }

    void saveSimCheckpointHostDeviceField(hostField &hostField, int &step){
        saveSimCheckpoint(hostField.h_fMom, ghostInterface, &step);
    }

    void saveSimCheckpointDeviceField( int &step){
        saveSimCheckpoint(d_fMom,ghostInterface,&step);
    }

    void treatDataDeviceField(hostField &hostField, int step){
        treatData(hostField.h_fMom, d_fMom,
        #if MEAN_FLOW
        hostField.m_fMom,
        #endif //MEAN_FLOW
        step);
    }

    void totalBcDragDeviceField(size_t step){
        #ifdef BC_FORCES
        totalBcDrag(d_BC_Fx, d_BC_Fy, d_BC_Fz, step);
        #endif //BC_FORCES
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