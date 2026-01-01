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
        allocateDeviceMemory(
            &d_fMom, &dNodeType, 
            BC_FORCES_PARAMS_PTR(d_)
            CURVED_BC_PARAMS_PTR(d_)
            &ghostInterface
        );
    }

    #ifdef DENSITY_CORRECTION
    void allocateDensityCorrectionMemory(){
        cudaMalloc((void**)&d_mean_rho, sizeof(dfloat));
    }
    #endif //DENSITY_CORRECTION

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
            CURVED_BC_PTRS(d_)
            CURVED_BC_ARRAY(d_)
            &step, gridBlock, threadBlock);
        
        #ifdef CURVED_BOUNDARY_CONDITION
            // Initialize curved boundary node count
            numberCurvedBoundaryNodes = getNumberCurvedBoundaryNodes(hostField.hNodeType);
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
            gpuMomCollisionStream << <gridBlock, threadBlock DYNAMIC_SHARED_MEMORY_PARAMS>> >(d_fMom, dNodeType,ghostInterface, DENSITY_CORRECTION_PARAMS(d_) BC_FORCES_PARAMS(d_) step, save
        #ifdef CURVED_BOUNDARY_CONDITION
        , d_curvedBC, d_curvedBC_array
        #endif //CURVED_BOUNDARY_CONDITION
        );
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
        treatData(hostField.h_fMom, d_fMom,
        #if MEAN_FLOW
        hostField.m_fMom,
        #endif //MEAN_FLOW
        #ifdef BC_FORCES
        d_BC_Fx, d_BC_Fy, d_BC_Fz,
        #endif //BC_FORCES
        step);
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