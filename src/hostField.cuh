
#ifndef __HOSTFIELD_STRUCTS_H
#define __HOSTFIELD_STRUCTS_H

#include "var.h"
#include "errorDef.h"
#include "main.cuh"

typedef struct hostField{
    dfloat* h_fMom;
    dfloat* rho;
    dfloat* ux;
    dfloat* uy;
    dfloat* uz;
    
    unsigned int* hNodeType;

    int NThread = 4;

    #if NODE_TYPE_SAVE
    NThread++;
    unsigned int* nodeTypeSave; 
    #endif //NODE_TYPE_SAVE

    #ifdef OMEGA_FIELD
    NThread++;
    dfloat* omega;
    #endif //OMEGA_FIELD

    #ifdef SECOND_DIST
    NThread++;
    dfloat* C;
    #endif //SECOND_DIST

    #ifdef A_XX_DIST
    NThread++;
    dfloat* Axx;
    #endif //A_XX_DIST
    #ifdef A_XY_DIST
    NThread++;
    dfloat* Axy;
    #endif //A_XY_DIST
    #ifdef A_XZ_DIST
    NThread++;
    dfloat* Axz;
    #endif //A_XZ_DIST
    #ifdef A_YY_DIST
    NThread++;
    dfloat* Ayy;
    #endif //A_YY_DIST
    #ifdef A_YZ_DIST
    NThread++;
    dfloat* Ayz;
    #endif //A_YZ_DIST
    #ifdef A_ZZ_DIST
    NThread++;
    dfloat* Azz;
    #endif //A_ZZ_DIST

    
    #ifdef DENSITY_CORRECTION
    dfloat* h_mean_rho;
    #endif //DENSITY_CORRECTION

    #if MEAN_FLOW
        dfloat* m_fMom;
        dfloat* m_rho;
        dfloat* m_ux;
        dfloat* m_uy;
        dfloat* m_uz;
        #ifdef SECOND_DIST
        dfloat* m_c;
        #endif //SECOND_DIST
    #endif //MEAN_FLOW

    #ifdef BC_FORCES
        #ifdef SAVE_BC_FORCES
        NThread += 3;
        dfloat* h_BC_Fx;
        dfloat* h_BC_Fy;
        dfloat* h_BC_Fz;
        #endif //SAVE_BC_FORCES
    #endif //_BC_FORCES

    void allocateHostMemoryHostField(){
        allocateHostMemory(
            &h_fMom, &rho, &ux, &uy, &uz
            OMEGA_FIELD_PARAMS_PTR
            SECOND_DIST_PARAMS_PTR
            A_XX_DIST_PARAMS_PTR
            A_XY_DIST_PARAMS_PTR
            A_XZ_DIST_PARAMS_PTR
            A_YY_DIST_PARAMS_PTR
            A_YZ_DIST_PARAMS_PTR
            A_ZZ_DIST_PARAMS_PTR
            MEAN_FLOW_PARAMS_PTR
            MEAN_FLOW_SECOND_DIST_PARAMS_PTR
            #if NODE_TYPE_SAVE
            , &nodeTypeSave
            #endif //NODE_TYPE_SAVE
            BC_FORCES_PARAMS_PTR(h_)
        );
    }

    void saveMacrHostField(unsigned int nSteps, std::atomic<bool>& savingMacrVtk, std::vector<std::atomic<bool>>& savingMacrBin, bool meanFlow){
        if(meanFlow){
            #if MEAN_FLOW
                saveMacr(m_fMom,m_rho,m_ux,m_uy,m_uz, NodeType, OMEGA_FIELD_PARAMS
                    #ifdef SECOND_DIST 
                    hostField.m_c,
                    #endif  //SECOND_DIST
                    NODE_TYPE_SAVE_PARAMS BC_FORCES_PARAMS(h_) nSteps, savingMacrVtk, savingMacrBin);
            #endif //MEAN_FLOW
        } else {
            saveMacr(h_fMom,rho,ux,uy,uz, hNodeType, OMEGA_FIELD_PARAMS
                #ifdef SECOND_DIST 
                C,
                #endif //SECOND_DIST
                #ifdef A_XX_DIST 
                Axx,
                #endif //A_XX_DIST
                #ifdef A_XY_DIST 
                Axy,
                #endif //A_XY_DIST
                #ifdef A_XY_DIST 
                Axz,
                #endif //A_XY_DIST
                #ifdef A_YY_DIST 
                Ayy,
                #endif //A_YY_DIST
                #ifdef A_YZ_DIST 
                Ayz,
                #endif //A_YY_DIST
                #ifdef A_ZZ_DIST 
                Azz,
                #endif //A_ZZ_DIST
                NODE_TYPE_SAVE_PARAMS BC_FORCES_PARAMS(h_) nSteps, savingMacrVtk, savingMacrBin);
        }
    }

    void interfaceCudaMemcpyLoop(ghostInterfaceData ghostInterface) {
        interfaceCudaMemcpy(ghostInterface,ghostInterface.h_fGhost,ghostInterface.fGhost,cudaMemcpyDeviceToHost,QF); 
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

    void interfaceCudaMemcpyEnd(ghostInterfaceData ghostInterface) {
        interfaceCudaMemcpy(ghostInterface,ghostInterface.h_fGhost,ghostInterface.gGhost,cudaMemcpyDeviceToHost,QF);    
        #ifdef SECOND_DIST 
        interfaceCudaMemcpy(ghostInterface,ghostInterface.g_h_fGhost,ghostInterface.g_fGhost,cudaMemcpyDeviceToHost,GF);
        #endif //SECOND_DIST 
        #ifdef A_XX_DIST 
        interfaceCudaMemcpy(ghostInterface,ghostInterface.Axx_h_fGhost,ghostInterface.Axx_fGhost,cudaMemcpyDeviceToHost,GF);
        #endif //A_XX_DIST
        #ifdef A_XY_DIST 
        interfaceCudaMemcpy(ghostInterface,ghostInterface.Axy_h_fGhost,ghostInterface.Axy_fGhost,cudaMemcpyDeviceToHost,GF);
        #endif //A_XY_DIST
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

    void freeHostField() {
        cudaFree(h_fMom);
        cudaFree(rho);
        cudaFree(ux);
        cudaFree(uy);
        cudaFree(uz);
        
        cudaFree(hNodeType);

        #if NODE_TYPE_SAVE
        cudaFree(nodeTypeSave);
        #endif //NODE_TYPE_SAVE
        
        #ifdef SECOND_DIST 
        cudaFree(C);
        #endif //SECOND_DIST
        #ifdef A_XX_DIST 
        cudaFree(Axx);
        #endif //A_XX_DIST
        #ifdef A_XY_DIST 
        cudaFree(Axy);
        #endif //A_XY_DIST
        #ifdef A_XZ_DIST 
        cudaFree(Axz);
        #endif //A_XZ_DIST
        #ifdef A_YY_DIST 
        cudaFree(Ayy);
        #endif //A_YY_DIST
        #ifdef A_YZ_DIST 
        cudaFree(Ayz);
        #endif //A_YZ_DIST
        #ifdef A_ZZ_DIST 
        cudaFree(Azz);
        #endif //A_ZZ_DIST
    
        #if MEAN_FLOW
            cudaFree(m_fMom);
            cudaFree(m_rho);
            cudaFree(m_ux);
            cudaFree(m_uy);
            cudaFree(m_uz);
            #ifdef SECOND_DIST
            cudaFree(m_c);
            #endif //MEAN_FLOW
        #endif //MEAN_FLOW
    
        #ifdef BC_FORCES
            #ifdef SAVE_BC_FORCES
            cudaFree(h_BC_Fx);
            cudaFree(h_BC_Fy);
            cudaFree(h_BC_Fz);
            #endif //SAVE_BC_FORCES
        #endif //_BC_FORCES
    
        #ifdef DENSITY_CORRECTION
            free(h_mean_rho);
        #endif //DENSITY_CORRECTION
    }

} HostField;

#endif //__HOSTFIELD_STRUCTS_H