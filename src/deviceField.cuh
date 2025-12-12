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
    #endif

    #ifdef DENSITY_CORRECTION
    dfloat* d_mean_rho[N_GPUS];
    #endif //DENSITY_CORRECTION

    #ifdef BC_FORCES
        dfloat* d_BC_Fx[N_GPUS];
        dfloat* d_BC_Fy[N_GPUS];
        dfloat* d_BC_Fz[N_GPUS];
    #endif //_BC_FORCES

    

    void allocateDeviceMemoryDeviceField(int g) {
        allocateDeviceMemory(
            &d_fMom[g], &dNodeType[g], 
            BC_FORCES_PARAMS_PTR(d_)
            CURVED_BC_PARAMS_PTR(d_)
            &ghostInterface[g]
        );
    }

    void initializeDomainDeviceField(hostField &hostField, dfloat **&randomNumbers, int &step, dim3 gridBlock, dim3 threadBlock, int g){
        initializeDomain(ghostInterface[g],     
            d_fMom[g], hostField.h_fMom, 
            #if MEAN_FLOW
            hostField.m_fMom,
            #endif //MEAN_FLOW
            hostField.hNodeType, dNodeType[g], randomNumbers, 
            BC_FORCES_PARAMS(d_)
            DENSITY_CORRECTION_PARAMS(h_)
            DENSITY_CORRECTION_PARAMS(d_)
            CURVED_BC_PTRS(d_)
            CURVED_BC_ARRAY(d_)
            &step, gridBlock, threadBlock, g);
    }

    #ifdef DENSITY_CORRECTION
    void mean_rhoDeviceField(size_t step, int g){
        mean_rho(d_fMom[g],step,d_mean_rho[g]);
    }
    #endif //DENSITY_CORRECTION

    void gpuMomCollisionStreamDeviceField(dim3 gridBlock, dim3 threadBlock, unsigned int step, bool save, int g){
            gpuMomCollisionStream << <gridBlock, threadBlock DYNAMIC_SHARED_MEMORY_PARAMS>> >(d_fMom[g], dNodeType[g],ghostInterface[g], DENSITY_CORRECTION_PARAMS(d_) BC_FORCES_PARAMS(d_) step, save
        #ifdef CURVED_BOUNDARY_CONDITION
        , d_curvedBC[g], d_curvedBC_array[g]
        #endif //CURVED_BOUNDARY_CONDITION
        );
        
        
    }

    #ifdef CURVED_BOUNDARY_CONDITION
    void updateCurvedBoundaryVelocitiesDeviceField(unsigned int numberCurvedBoundaryNodes, int g){
        updateCurvedBoundaryVelocities << <1,numberCurvedBoundaryNodes>> >(d_curvedBC_array[g],d_fMom[g],numberCurvedBoundaryNodes);
        cudaDeviceSynchronize();
    }
    #endif

    void swapGhostInterfacesDeviceField(int g){
        swapGhostInterfaces(ghostInterface[g]);
    }
    
    #ifdef LOCAL_FORCES
    void gpuResetMacroForcesDeviceField(dim3 gridBlock, dim3 threadBlock, int g){
        gpuResetMacroForces<<<gridBlock, threadBlock>>>(d_fMom[g]);
    }
    #endif //LOCAL_FORCES

    #ifdef PARTICLE_MODEL
    void particleSimulationDeviceField(ParticlesSoA &particlesSoA, cudaStream_t *streamsPart, unsigned int step){
        particleSimulation(&particlesSoA,d_fMom,streamsPart,step);
    }
    #endif //PARTICLE_MODEL

    void interfaceCudaMemcpyDeviceField(bool fGhost, int g){
            if (fGhost) {
                interfaceCudaMemcpy(ghostInterface[g], ghostInterface[g].h_fGhost, ghostInterface[g].fGhost, cudaMemcpyDeviceToHost, QF);
            } else {
                interfaceCudaMemcpy(ghostInterface[g], ghostInterface[g].h_fGhost, ghostInterface[g].gGhost, cudaMemcpyDeviceToHost, QF);
                
            }
            #ifdef SECOND_DIST 
            interfaceCudaMemcpy(ghostInterface[g],ghostInterface[g].g_h_fGhost,ghostInterface[g].g_fGhost,cudaMemcpyDeviceToHost,GF);
            #endif //SECOND_DIST
            #ifdef A_XX_DIST 
            interfaceCudaMemcpy(ghostInterface[g],ghostInterface[g].Axx_h_fGhost,ghostInterface[g].Axx_fGhost,cudaMemcpyDeviceToHost,GF);
            #endif //A_XX_DIST     
            #ifdef A_XY_DIST 
            interfaceCudaMemcpy(ghostInterface[g],ghostInterface[g].Axy_h_fGhost,ghostInterface[g].Axy_fGhost,cudaMemcpyDeviceToHost,GF);
            #endif //A_XX_DIST        
            #ifdef A_XZ_DIST 
            interfaceCudaMemcpy(ghostInterface[g],ghostInterface[g].Axz_h_fGhost,ghostInterface[g].Axz_fGhost,cudaMemcpyDeviceToHost,GF);
            #endif //A_XZ_DIST
            #ifdef A_YY_DIST 
            interfaceCudaMemcpy(ghostInterface[g],ghostInterface[g].Ayy_h_fGhost,ghostInterface[g].Ayy_fGhost,cudaMemcpyDeviceToHost,GF);
            #endif //A_YY_DIST        
            #ifdef A_YZ_DIST 
            interfaceCudaMemcpy(ghostInterface[g],ghostInterface[g].Ayz_h_fGhost,ghostInterface[g].Ayz_fGhost,cudaMemcpyDeviceToHost,GF);
            #endif //A_YZ_DIST      
            #ifdef A_ZZ_DIST 
            interfaceCudaMemcpy(ghostInterface[g],ghostInterface[g].Azz_h_fGhost,ghostInterface[g].Azz_fGhost,cudaMemcpyDeviceToHost,GF);
            #endif //A_ZZ_DIST
    }

    void cudaMemcpyDeviceField(hostField &hostField, int g){
        checkCudaErrors(cudaMemcpy(hostField.h_fMom, d_fMom[g], sizeof(dfloat) * NUMBER_LBM_NODES*NUMBER_MOMENTS, cudaMemcpyDeviceToHost));
    }

    void saveSimCheckpointHostDeviceField(hostField &hostField, int &step, int g){
        saveSimCheckpoint(hostField.h_fMom, ghostInterface[g], &step);
    }

    void saveSimCheckpointDeviceField( int &step, int g){
        saveSimCheckpoint(d_fMom[g],ghostInterface[g],&step);
    }

    void treatDataDeviceField(hostField &hostField, int step, int g){
        treatData(hostField.h_fMom, d_fMom[g],
        #if MEAN_FLOW
        hostField.m_fMom,
        #endif //MEAN_FLOW
        step);
    }

    #ifdef BC_FORCES
    void totalBcDragDeviceField(size_t step, int g){
        totalBcDrag(d_BC_Fx[g], d_BC_Fy[g], d_BC_Fz[g], step);
    }
    #endif //BC_FORCES

    #if defined BC_FORCES && defined SAVE_BC_FORCES
    void saveBcForces(hostField &hostField, int g){
        checkCudaErrors(cudaDeviceSynchronize()); 
        checkCudaErrors(cudaMemcpy(hostField.h_BC_Fx, d_BC_Fx[g], MEM_SIZE_SCALAR, cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(hostField.h_BC_Fy, d_BC_Fy[g], MEM_SIZE_SCALAR, cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(hostField.h_BC_Fz, d_BC_Fz[g], MEM_SIZE_SCALAR, cudaMemcpyDeviceToHost));
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