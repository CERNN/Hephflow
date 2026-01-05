
#ifndef __HOSTFIELD_STRUCTS_H
#define __HOSTFIELD_STRUCTS_H

#include "var.h"
#include "include/errorDef.h"
#include "globalStructs.h"

// ============================================================================
// FORWARD DECLARATIONS OF FUNCTIONS
// ============================================================================

// These functions are implemented in saveData.cu and treatData.cu
// Forward declarations here to avoid circular includes
__host__ void saveMacr(const SaveDataParams* params);
__host__ void saveVarVTK(const SaveDataParams* params);
__host__ void treatData(const TreatDataParams* params);

// ============================================================================
// HOST FIELD STRUCTURE
// ============================================================================
// NOTE: This struct requires main.cuh to be included first for allocateHostMemory()
//       and other function declarations. main.cuh must be included before hostField.cuh

typedef struct hostField{
    dfloat* h_fMom;
    dfloat* rho;
    dfloat* ux;
    dfloat* uy;
    dfloat* uz;
    
    unsigned int* hNodeType;

    int NThread;

    #if NODE_TYPE_SAVE
    unsigned int* nodeTypeSave; 
    #endif //NODE_TYPE_SAVE

    #ifdef OMEGA_FIELD
    dfloat* omega;
    #endif //OMEGA_FIELD

    #ifdef SECOND_DIST
    dfloat* C;
    #endif //SECOND_DIST

    #ifdef PHI_DIST
    dfloat* phi;
    #endif //PHI_DIST

    #ifdef LAMBDA_DIST
    dfloat* lambda;
    #endif //LAMBDA_DIST

    #ifdef A_XX_DIST
    dfloat* Axx;
    #endif //A_XX_DIST
    #ifdef A_XY_DIST
    dfloat* Axy;
    #endif //A_XY_DIST
    #ifdef A_XZ_DIST
    dfloat* Axz;
    #endif //A_XZ_DIST
    #ifdef A_YY_DIST
    dfloat* Ayy;
    #endif //A_YY_DIST
    #ifdef A_YZ_DIST
    dfloat* Ayz;
    #endif //A_YZ_DIST
    #ifdef A_ZZ_DIST
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
        #ifdef PHI_DIST
        dfloat* m_phi;
        #endif //PHI_DIST
        #ifdef LAMBDA_DIST
        dfloat* m_lambda;
        #endif //LAMBDA_DIST
    #endif //MEAN_FLOW

    #ifdef BC_FORCES
        #ifdef SAVE_BC_FORCES
        dfloat* h_BC_Fx;
        dfloat* h_BC_Fy;
        dfloat* h_BC_Fz;
        #endif //SAVE_BC_FORCES
    #endif //_BC_FORCES

    // Constructor: initialize pointers and compute NThread based on compile-time flags
    hostField()
        : h_fMom(nullptr), rho(nullptr), ux(nullptr), uy(nullptr), uz(nullptr),
          hNodeType(nullptr)
        #if NODE_TYPE_SAVE
        , nodeTypeSave(nullptr)
        #endif
        #ifdef OMEGA_FIELD
        , omega(nullptr)
        #endif
        #ifdef SECOND_DIST
        , C(nullptr)
        #endif //SECOND_DIST
        #ifdef PHI_DIST
        , phi(nullptr)
        #endif //PHI_DIST
        #ifdef LAMBDA_DIST
        , lambda(nullptr)
        #endif //LAMBDA_DIST
        #ifdef A_XX_DIST
        , Axx(nullptr)
        #endif
        #ifdef A_XY_DIST
        , Axy(nullptr)
        #endif
        #ifdef A_XZ_DIST
        , Axz(nullptr)
        #endif
        #ifdef A_YY_DIST
        , Ayy(nullptr)
        #endif
        #ifdef A_YZ_DIST
        , Ayz(nullptr)
        #endif
        #ifdef A_ZZ_DIST
        , Azz(nullptr)
        #endif
        #ifdef DENSITY_CORRECTION
        , h_mean_rho(nullptr)
        #endif
        #if MEAN_FLOW
        , m_fMom(nullptr), m_rho(nullptr), m_ux(nullptr), m_uy(nullptr), m_uz(nullptr)
            #ifdef SECOND_DIST
            , m_c(nullptr)
            #endif
            #ifdef PHI_DIST
            , m_phi(nullptr)
            #endif
        #endif
        #ifdef BC_FORCES
            #ifdef SAVE_BC_FORCES
            , h_BC_Fx(nullptr), h_BC_Fy(nullptr), h_BC_Fz(nullptr)
            #endif
        #endif
    {
        NThread = 4;
        #if NODE_TYPE_SAVE
        NThread++;
        #endif
        #ifdef OMEGA_FIELD
        NThread++;
        #endif
        #ifdef SECOND_DIST
        NThread++;
        #endif
        #ifdef PHI_DIST
        NThread++;
        #endif
        #ifdef LAMBDA_DIST
        NThread++;
        #endif
        #ifdef A_XX_DIST
        NThread++;
        #endif
        #ifdef A_XY_DIST
        NThread++;
        #endif
        #ifdef A_XZ_DIST
        NThread++;
        #endif
        #ifdef A_YY_DIST
        NThread++;
        #endif
        #ifdef A_YZ_DIST
        NThread++;
        #endif
        #ifdef A_ZZ_DIST
        NThread++;
        #endif
        #ifdef BC_FORCES
            #ifdef SAVE_BC_FORCES
            NThread += 3;
            #endif
        #endif
    }

    void allocateHostMemoryHostField(){
        unsigned int memAllocated = 0;

        checkCudaErrors(cudaMallocHost((void**)&h_fMom, MEM_SIZE_MOM));
        checkCudaErrors(cudaMallocHost((void**)&rho, MEM_SIZE_SCALAR));
        checkCudaErrors(cudaMallocHost((void**)&ux, MEM_SIZE_SCALAR));
        checkCudaErrors(cudaMallocHost((void**)&uy, MEM_SIZE_SCALAR));
        checkCudaErrors(cudaMallocHost((void**)&uz, MEM_SIZE_SCALAR));

        memAllocated += MEM_SIZE_MOM + 4 * MEM_SIZE_SCALAR;

        #ifdef OMEGA_FIELD
        checkCudaErrors(cudaMallocHost((void**)&omega, MEM_SIZE_SCALAR));
        memAllocated += MEM_SIZE_SCALAR;
        #endif //OMEGA_FIELD

        #ifdef SECOND_DIST
        checkCudaErrors(cudaMallocHost((void**)&C, MEM_SIZE_SCALAR));
        memAllocated += MEM_SIZE_SCALAR;
        #endif //SECOND_DIST

        #ifdef PHI_DIST
        checkCudaErrors(cudaMallocHost((void**)&phi, MEM_SIZE_SCALAR));
        memAllocated += MEM_SIZE_SCALAR;
        #endif //PHI_DIST

        #ifdef LAMBDA_DIST
        checkCudaErrors(cudaMallocHost((void**)&lambda, MEM_SIZE_SCALAR));
        memAllocated += MEM_SIZE_SCALAR;
        #endif //LAMBDA_DIST

        #ifdef A_XX_DIST
        checkCudaErrors(cudaMallocHost((void**)&Axx, MEM_SIZE_SCALAR));
        memAllocated += MEM_SIZE_SCALAR;
        #endif //A_XX_DIST
        #ifdef A_XY_DIST
        checkCudaErrors(cudaMallocHost((void**)&Axy, MEM_SIZE_SCALAR));
        memAllocated += MEM_SIZE_SCALAR;
        #endif //A_XY_DIST
        #ifdef A_XZ_DIST
        checkCudaErrors(cudaMallocHost((void**)&Axz, MEM_SIZE_SCALAR));
        memAllocated += MEM_SIZE_SCALAR;
        #endif //A_XZ_DIST
        #ifdef A_YY_DIST
        checkCudaErrors(cudaMallocHost((void**)&Ayy, MEM_SIZE_SCALAR));
        memAllocated += MEM_SIZE_SCALAR;
        #endif //A_YY_DIST
        #ifdef A_YZ_DIST
        checkCudaErrors(cudaMallocHost((void**)&Ayz, MEM_SIZE_SCALAR));
        memAllocated += MEM_SIZE_SCALAR;
        #endif //A_YZ_DIST
        #ifdef A_ZZ_DIST
        checkCudaErrors(cudaMallocHost((void**)&Azz, MEM_SIZE_SCALAR));
        memAllocated += MEM_SIZE_SCALAR;
        #endif //A_ZZ_DIST

        #if MEAN_FLOW
        checkCudaErrors(cudaMallocHost((void**)&m_fMom, MEM_SIZE_MOM));
        checkCudaErrors(cudaMallocHost((void**)&m_rho, MEM_SIZE_SCALAR));
        checkCudaErrors(cudaMallocHost((void**)&m_ux, MEM_SIZE_SCALAR));
        checkCudaErrors(cudaMallocHost((void**)&m_uy, MEM_SIZE_SCALAR));
        checkCudaErrors(cudaMallocHost((void**)&m_uz, MEM_SIZE_SCALAR));
        memAllocated += MEM_SIZE_MOM + 4 * MEM_SIZE_SCALAR;
        #ifdef SECOND_DIST
        checkCudaErrors(cudaMallocHost((void**)&m_c, MEM_SIZE_SCALAR));
        memAllocated += MEM_SIZE_SCALAR;
        #endif //SECOND_DIST
        #ifdef PHI_DIST
        checkCudaErrors(cudaMallocHost((void**)&m_phi, MEM_SIZE_SCALAR));
        memAllocated += MEM_SIZE_SCALAR;
        #endif //PHI_DIST
        #ifdef LAMBDA_DIST
        checkCudaErrors(cudaMallocHost((void**)&m_lambda, MEM_SIZE_SCALAR));
        memAllocated += MEM_SIZE_SCALAR;
        #endif //LAMBDA_DIST
        #endif // MEAN_FLOW

        #ifdef BC_FORCES
        #ifdef SAVE_BC_FORCES
        checkCudaErrors(cudaMallocHost((void**)&h_BC_Fx, MEM_SIZE_SCALAR));
        checkCudaErrors(cudaMallocHost((void**)&h_BC_Fy, MEM_SIZE_SCALAR));
        checkCudaErrors(cudaMallocHost((void**)&h_BC_Fz, MEM_SIZE_SCALAR));
        memAllocated += 3 * MEM_SIZE_SCALAR;
        #endif //SAVE_BC_FORCES
        #endif //BC_FORCES

        
        #if NODE_TYPE_SAVE
        checkCudaErrors(cudaMallocHost((void**)&nodeTypeSave, sizeof(unsigned int) * NUMBER_LBM_NODES));
        memAllocated += sizeof(unsigned int) * NUMBER_LBM_NODES;
        #endif //NODE_TYPE_SAVE

        printf("Host Memory Allocated: %0.2f MB\n", (float)memAllocated / (1024.0 * 1024.0)); if(console_flush) fflush(stdout);
    }

    #ifdef DENSITY_CORRECTION
    void allocateDensityCorrectionMemory(){
        checkCudaErrors(cudaMallocHost((void**)&(h_mean_rho), sizeof(dfloat)));
    }
    #endif //DENSITY_CORRECTION

    void saveMacrHostField(unsigned int nSteps, std::atomic<bool>& savingMacrVtk, std::vector<std::atomic<bool>>& savingMacrBin, bool meanFlow){
        SaveDataParams saveMacrParams;
        saveMacrParams.nSteps = nSteps;
        saveMacrParams.savingMacrVtk = &savingMacrVtk;
        saveMacrParams.savingMacrBin = &savingMacrBin;
        saveMacrParams.vtkFilename = nullptr;  // Binary save, not VTK
        
        if(meanFlow){
            #if MEAN_FLOW
                saveMacrParams.h_fMom = m_fMom;
                saveMacrParams.h_rho = m_rho;
                saveMacrParams.h_ux = m_ux;
                saveMacrParams.h_uy = m_uy;
                saveMacrParams.h_uz = m_uz;
                saveMacrParams.h_nodeType = hNodeType;
                #ifdef OMEGA_FIELD
                saveMacrParams.h_omega = omega;
                #endif
                #ifdef SECOND_DIST
                saveMacrParams.h_C = m_c;
                #endif
                #ifdef PHI_DIST
                saveMacrParams.h_phi = m_phi;
                #endif
                #ifdef LAMBDA_DIST
                saveMacrParams.h_lambda = m_lambda;
                #endif
                #if NODE_TYPE_SAVE
                saveMacrParams.h_nodeTypeSave = nodeTypeSave;
                #endif
                #ifdef BC_FORCES
                saveMacrParams.h_BC_Fx = h_BC_Fx;
                saveMacrParams.h_BC_Fy = h_BC_Fy;
                saveMacrParams.h_BC_Fz = h_BC_Fz;
                #endif
                saveMacr(&saveMacrParams);
            #endif //MEAN_FLOW
        } else {
            saveMacrParams.h_fMom = h_fMom;
            saveMacrParams.h_rho = rho;
            saveMacrParams.h_ux = ux;
            saveMacrParams.h_uy = uy;
            saveMacrParams.h_uz = uz;
            saveMacrParams.h_nodeType = hNodeType;
            #ifdef OMEGA_FIELD
            saveMacrParams.h_omega = omega;
            #endif
            #ifdef SECOND_DIST
            saveMacrParams.h_C = C;
            #endif
            #ifdef PHI_DIST
            saveMacrParams.h_phi = phi;
            #endif
            #ifdef LAMBDA_DIST
            saveMacrParams.h_lambda = lambda;
            #endif
            #ifdef A_XX_DIST
            saveMacrParams.h_Axx = Axx;
            #endif
            #ifdef A_XY_DIST
            saveMacrParams.h_Axy = Axy;
            #endif
            #ifdef A_XZ_DIST
            saveMacrParams.h_Axz = Axz;
            #endif
            #ifdef A_YY_DIST
            saveMacrParams.h_Ayy = Ayy;
            #endif
            #ifdef A_YZ_DIST
            saveMacrParams.h_Ayz = Ayz;
            #endif
            #ifdef A_ZZ_DIST
            saveMacrParams.h_Azz = Azz;
            #endif
            #if NODE_TYPE_SAVE
            saveMacrParams.h_nodeTypeSave = nodeTypeSave;
            #endif
            #ifdef BC_FORCES
            saveMacrParams.h_BC_Fx = h_BC_Fx;
            saveMacrParams.h_BC_Fy = h_BC_Fy;
            saveMacrParams.h_BC_Fz = h_BC_Fz;
            #endif
            saveMacr(&saveMacrParams);
        }
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
        #ifdef PHI_DIST 
        cudaFree(phi);
        #endif //PHI_DIST
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
            #endif //SECOND_DIST
            #ifdef PHI_DIST
            cudaFree(m_phi);
            #endif //PHI_DIST
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