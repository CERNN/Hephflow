/**
 *  @file globalStructs.h
 *  Contributors history:
 *  @author Waine Jr. (waine@alunos.utfpr.edu.br)
 *  @author Marco Aurelio Ferrari (marcoferrari@alunos.utfpr.edu.br)
 *  @brief Global general structs
 *  @version 0.4.0
 *  @date 01/09/2025
 */

#ifndef __GLOBAL_STRUCTS_H
#define __GLOBAL_STRUCTS_H

#include "var.h"
#include "include/errorDef.h"
#include "include/interface.h"
#include <atomic>
#include <vector>
#include <string>

/*
*   Struct for dfloat in x, y, z
*/
typedef struct dfloat3 {
    dfloat x;
    dfloat y;
    dfloat z;

    __host__ __device__
    dfloat3(dfloat x = 0, dfloat y = 0, dfloat z = 0)
    {
        this->x = x;
        this->y = y;
        this->z = z;
    }

    // Overload the unary - operator
    __host__ __device__
    dfloat3 operator-() const {
        return dfloat3(-x, -y, -z);
    }

    // Element-wise addition
    __host__ __device__
    friend dfloat3 operator+(const dfloat3& a, const dfloat3& b) {
        return dfloat3(a.x + b.x, a.y + b.y, a.z + b.z);
    }

    // Element-wise subtraction
    __host__ __device__
    friend dfloat3 operator-(const dfloat3& a, const dfloat3& b) {
        return dfloat3(a.x - b.x, a.y - b.y, a.z - b.z);
    }

    // Element-wise multiplication
    __host__ __device__
    friend dfloat3 operator*(const dfloat3& a, const dfloat3& b) {
        return dfloat3(a.x * b.x, a.y * b.y, a.z * b.z);
    }

    // Element-wise division
    __host__ __device__
    friend dfloat3 operator/(const dfloat3& a, const dfloat3& b) {
        return dfloat3(a.x / b.x, a.y / b.y, a.z / b.z);
    }
    
    //between 1 dfloat and dfloat3
    // Element-wise addition with scalar
    __host__ __device__
    friend dfloat3 operator+(const dfloat3& vec, const dfloat scalar) {
        return dfloat3(vec.x + scalar, vec.y + scalar, vec.z + scalar);
    }
    // Element-wise addition with scalar
    __host__ __device__
    friend dfloat3 operator+(const dfloat scalar, const dfloat3& vec) {
        return dfloat3(scalar + vec.x, scalar + vec.y, scalar + vec.z);
    }

    // Element-wise subtraction with scalar
    __host__ __device__
    friend dfloat3 operator-(const dfloat3& vec, const dfloat scalar) {
        return dfloat3(vec.x - scalar, vec.y - scalar, vec.z - scalar);
    }
    // Element-wise subtraction with scalar
    __host__ __device__
    friend dfloat3 operator-(const dfloat scalar, const dfloat3& vec) {
        return dfloat3(scalar - vec.x, scalar - vec.y, scalar - vec.z);
    }

    // Element-wise multiplication with scalar
    __host__ __device__
    friend dfloat3 operator*(const dfloat3& vec, const dfloat scalar) {
        return dfloat3(vec.x * scalar, vec.y * scalar, vec.z * scalar);
    }
    // Element-wise multiplication with scalar
    __host__ __device__
    friend dfloat3 operator*(const dfloat scalar, const dfloat3& vec) {
        return dfloat3(scalar * vec.x, scalar * vec.y, scalar * vec.z);
    }

    // Element-wise division with scalar
    __host__ __device__
    friend dfloat3 operator/(const dfloat3& vec, const dfloat scalar) {
        return dfloat3(vec.x / scalar, vec.y / scalar, vec.z / scalar);
    }
    // Element-wise division with scalar
    __host__ __device__
    friend dfloat3 operator/(const dfloat scalar, const dfloat3& vec) {
        return dfloat3(scalar / vec.x, scalar / vec.y, scalar / vec.z);
    }

} dfloat3;

/*
*   Struct for dfloat in x, y, z, w (quartenion)
*/
typedef struct dfloat4{
    dfloat x;
    dfloat y;
    dfloat z;
    dfloat w;

    __host__ __device__
    dfloat4(dfloat x = 0, dfloat y = 0, dfloat z = 0, dfloat w = 0)
    {
        this->x = x;
        this->y = y;
        this->z = z;
        this->w = w;
    }
} dfloat4;

typedef struct dfloat6{
    dfloat xx;
    dfloat yy;
    dfloat zz;
    dfloat xy;
    dfloat xz;
    dfloat yz;

    __host__ __device__
    dfloat6(dfloat xx = 0, dfloat yy = 0, dfloat zz = 0, dfloat xy = 0, dfloat xz = 0, dfloat yz = 0)
    {
        this->xx = xx;
        this->yy = yy;
        this->zz = zz;
        this->xy = xy;
        this->xz = xz;
        this->yz = yz;
    }
} dfloat6;

typedef struct dfloat3SoA {
    int varLocation; // IN_VIRTUAL or IN_HOST
    dfloat* x; // x array
    dfloat* y; // y array
    dfloat* z; // z array

    __host__ __device__
    dfloat3SoA()
    {
        varLocation = 0;
        x = nullptr;
        y = nullptr;
        z = nullptr;
    }

    __host__ __device__
    ~dfloat3SoA()
    {
        varLocation = 0;
        x = nullptr;
        y = nullptr;
        z = nullptr;
    }

    /**
     *  @brief Allocate memory for SoA
    *   
     *  @param arraySize: array size, in number of elements
     *  @param location: array location, IN_VIRTUAL or IN_HOST
    */
    __host__
    void allocateMemory(size_t arraySize, int location = IN_VIRTUAL){
        size_t memSize = sizeof(dfloat) * arraySize;

        this->varLocation = location;
        switch(location){
        case IN_VIRTUAL:
            checkCudaErrors(cudaMallocManaged((void**)&(this->x), memSize));
            checkCudaErrors(cudaMallocManaged((void**)&(this->y), memSize));
            checkCudaErrors(cudaMallocManaged((void**)&(this->z), memSize));
            break;
        case IN_HOST:
            checkCudaErrors(cudaMallocHost((void**)&(this->x), memSize));
            checkCudaErrors(cudaMallocHost((void**)&(this->y), memSize));
            checkCudaErrors(cudaMallocHost((void**)&(this->z), memSize));
            break;
        default:
            break;
        }
    }

    /**
     *  @brief Free memory of SoA
    */
    __host__
    void freeMemory(){
        switch (this->varLocation)
        {
        case IN_VIRTUAL:
            checkCudaErrors(cudaFree(this->x));
            checkCudaErrors(cudaFree(this->y));
            checkCudaErrors(cudaFree(this->z));
            break;

        case IN_HOST:
            checkCudaErrors(cudaFreeHost(this->x));
            checkCudaErrors(cudaFreeHost(this->y));
            checkCudaErrors(cudaFreeHost(this->z));
            break;
        default:
            break;
        }
    }

    /**
     *  @brief Copy values from another dfloat3SoA array  
     *  @param arrayRef: arrays to copy values
     *  @param memSize: size of memory to copy, in bytes
     *  @param baseIdx: base index for this
     *  @param baseIdxRef: base index for arrayRef
    */
   __host__
    void copyFromDfloat3SoA(dfloat3SoA arrayRef, size_t memSize, size_t baseIdx=0, size_t baseIdxRef=0){

        cudaStream_t streamX, streamY, streamZ;
        checkCudaErrors(cudaStreamCreate(&(streamX)));
        checkCudaErrors(cudaStreamCreate(&(streamY)));
        checkCudaErrors(cudaStreamCreate(&(streamZ)));

        checkCudaErrors(cudaMemcpyAsync(this->x+baseIdx, arrayRef.x+baseIdxRef, 
            memSize, cudaMemcpyDefault, streamX));
        checkCudaErrors(cudaMemcpyAsync(this->y+baseIdx, arrayRef.y+baseIdxRef, 
            memSize, cudaMemcpyDefault, streamY));
        checkCudaErrors(cudaMemcpyAsync(this->z+baseIdx, arrayRef.z+baseIdxRef, 
            memSize, cudaMemcpyDefault, streamZ));

        checkCudaErrors(cudaStreamSynchronize(streamX));
        checkCudaErrors(cudaStreamSynchronize(streamY));
        checkCudaErrors(cudaStreamSynchronize(streamZ));

        checkCudaErrors(cudaStreamDestroy(streamX));
        checkCudaErrors(cudaStreamDestroy(streamY));
        checkCudaErrors(cudaStreamDestroy(streamZ));
    }

    /**
     *  @brief Copy value from dfloat3 
     *  @param val: dfloat3 to copy values
     *  @param idx: index to write values to
     */
    __host__ __device__
    void copyValuesFromFloat3(dfloat3 val, size_t idx){
        this->x[idx] = val.x;
        this->y[idx] = val.y;
        this->z[idx] = val.z;
    }

    /**
     *  @brief Get the falues from given index
     *  @param idx: index to copy from
     *  @return dfloat3: dfloat3 with values
     */
    __host__ __device__
    dfloat3 getValuesFromIdx(size_t idx){
        return dfloat3(this->x[idx], this->y[idx], this->z[idx]);
    }

    __host__ __device__
    void leftShift(size_t idx, size_t left_shift){
        this->x[idx-left_shift] = this->x[idx];
        this->y[idx-left_shift] = this->y[idx];
        this->z[idx-left_shift] = this->z[idx];
    }

} dfloat3SoA;

/**
 *  @brief Structure of Arrays for dfloat4 (quaternions: w, x, y, z)
 *  Stores 4-component quaternions in separated arrays for CUDA memory coalescing
 */
typedef struct dfloat4SoA {
    int varLocation; // IN_VIRTUAL or IN_HOST
    dfloat* w; // w component array
    dfloat* x; // x component array
    dfloat* y; // y component array
    dfloat* z; // z component array

    __host__ __device__
    dfloat4SoA()
    {
        varLocation = 0;
        w = nullptr;
        x = nullptr;
        y = nullptr;
        z = nullptr;
    }

    __host__ __device__
    ~dfloat4SoA()
    {
        varLocation = 0;
        w = nullptr;
        x = nullptr;
        y = nullptr;
        z = nullptr;
    }

    /**
     *  @brief Allocate memory for quaternion SoA
     *  @param arraySize: array size, in number of elements
     *  @param location: array location, IN_VIRTUAL or IN_HOST
     */
    __host__
    void allocateMemory(size_t arraySize, int location = IN_VIRTUAL){
        size_t memSize = sizeof(dfloat) * arraySize;

        this->varLocation = location;
        switch(location){
        case IN_VIRTUAL:
            checkCudaErrors(cudaMallocManaged((void**)&(this->w), memSize));
            checkCudaErrors(cudaMallocManaged((void**)&(this->x), memSize));
            checkCudaErrors(cudaMallocManaged((void**)&(this->y), memSize));
            checkCudaErrors(cudaMallocManaged((void**)&(this->z), memSize));
            break;
        case IN_HOST:
            checkCudaErrors(cudaMallocHost((void**)&(this->w), memSize));
            checkCudaErrors(cudaMallocHost((void**)&(this->x), memSize));
            checkCudaErrors(cudaMallocHost((void**)&(this->y), memSize));
            checkCudaErrors(cudaMallocHost((void**)&(this->z), memSize));
            break;
        default:
            break;
        }
    }

    /**
     *  @brief Free memory of quaternion SoA
     */
    __host__
    void freeMemory(){
        switch (this->varLocation)
        {
        case IN_VIRTUAL:
            checkCudaErrors(cudaFree(this->w));
            checkCudaErrors(cudaFree(this->x));
            checkCudaErrors(cudaFree(this->y));
            checkCudaErrors(cudaFree(this->z));
            break;

        case IN_HOST:
            checkCudaErrors(cudaFreeHost(this->w));
            checkCudaErrors(cudaFreeHost(this->x));
            checkCudaErrors(cudaFreeHost(this->y));
            checkCudaErrors(cudaFreeHost(this->z));
            break;
        default:
            break;
        }
    }

    /**
     *  @brief Copy value from dfloat4 (quaternion)
     *  @param val: dfloat4 to copy values
     *  @param idx: index to write values to
     */
    __host__ __device__
    void copyValuesFromFloat4(dfloat4 val, size_t idx){
        this->w[idx] = val.w;
        this->x[idx] = val.x;
        this->y[idx] = val.y;
        this->z[idx] = val.z;
    }

    /**
     *  @brief Get the values from given index
     *  @param idx: index to copy from
     *  @return dfloat4: dfloat4 with quaternion values
     */
    __host__ __device__
    dfloat4 getValuesFromIdx(size_t idx){
        return dfloat4{this->w[idx], this->x[idx], this->y[idx], this->z[idx]};
    }

    /**
     *  @brief Left shift values (used for array compaction after deletions)
     *  @param idx: source index
     *  @param left_shift: number of positions to shift left
     */
    __host__ __device__
    void leftShift(size_t idx, size_t left_shift){
        this->w[idx-left_shift] = this->w[idx];
        this->x[idx-left_shift] = this->x[idx];
        this->y[idx-left_shift] = this->y[idx];
        this->z[idx-left_shift] = this->z[idx];
    }

} dfloat4SoA;

typedef struct gpuDirection{
    dfloat* Z_0;
    dfloat* Z_1;
    dfloat* auxZ_0;
    dfloat* auxZ_1;
} GpuDirection;

typedef struct ghostData {
    dfloat* X_0;
    dfloat* X_1;
    dfloat* Y_0;
    dfloat* Y_1;
    dfloat* Z_0;
    dfloat* Z_1;
} GhostData;

typedef struct macroInterfaceGPUData {
    gpuDirection rho;
    gpuDirection ux;
    gpuDirection uy;
    gpuDirection uz;

    #ifdef SECOND_DIST
        gpuDirection g;
    #endif //SECOND_DIST

    #ifdef PHI_DIST
        gpuDirection phi;
        gpuDirection nx;
        gpuDirection ny;
        gpuDirection nz;
        gpuDirection mu;
    #endif //PHI_DIST
    #ifdef A_XX_DIST
        gpuDirection Axx;
    #endif //A_XX_DIST
    #ifdef A_XY_DIST
        gpuDirection Axy;
    #endif //A_XY_DIST
    #ifdef A_XZ_DIST
        gpuDirection Axz;
    #endif //A_XZ_DIST
    #ifdef A_YY_DIST
        gpuDirection Ayy;
    #endif //A_YY_DIST
    #ifdef A_YZ_DIST
        gpuDirection Ayz;
    #endif //A_YZ_DIST
    #ifdef A_ZZ_DIST
        gpuDirection Azz;
    #endif //A_ZZ_DIST
    #ifdef LAMBDA_DIST
        gpuDirection lambda;
    #endif //LAMBDA_DIST

} macroInterfaceGPUData;

typedef struct ghostInterfaceData  {
    ghostData pop;
    ghostData popAux;
    ghostData h_pop;

    #ifdef SECOND_DIST
        ghostData g;
        ghostData gAux;
        ghostData h_g;
    #endif //SECOND_DIST
    #ifdef PHI_DIST
        ghostData phi;
        ghostData phiAux;
        ghostData h_phi;
    #endif //PHI_DIST
    #ifdef A_XX_DIST
        ghostData Axx;
        ghostData AxxAux;
        ghostData h_Axx;
    #endif //A_XX_DIST
    #ifdef A_XY_DIST
        ghostData Axy;
        ghostData AxyAux;
        ghostData h_Axy;
    #endif //A_XY_DIST
    #ifdef A_XZ_DIST
        ghostData Axz;
        ghostData AxzAux;
        ghostData h_Axz;
    #endif //A_XZ_DIST
    #ifdef A_YY_DIST
        ghostData Ayy;
        ghostData AyyAux;
        ghostData h_Ayy;
    #endif //A_YY_DIST
    #ifdef A_YZ_DIST
        ghostData Ayz;
        ghostData AyzAux;
        ghostData h_Ayz;
    #endif //A_YZ_DIST
    #ifdef A_ZZ_DIST
        ghostData Azz;
        ghostData AzzAux;
        ghostData h_Azz;
    #endif //A_ZZ_DIST
    #ifdef LAMBDA_DIST
        ghostData lambda;
        ghostData lambdaAux;
        ghostData h_lambda;
    #endif //LAMBDA_DIST

} GhostInterfaceData;

typedef struct wall{
    dfloat3 normal;
    dfloat distance;
    dfloat3 velocity;

    __host__ __device__
    wall(dfloat3 n = dfloat3(0,0,0),
         dfloat d = 0,
         dfloat3 v = dfloat3(0,0,0))
        : normal(n), distance(d), velocity(v) {}
} Wall;

typedef struct curvedBoundary{
    dfloat3 b;
    dfloat3 w;
    dfloat3 pf1;
    dfloat3 pf2;
    dfloat3 pf3;

    dfloat delta;
    dfloat delta_r;  // Spacing between fluid points (lattice spacing in normal direction)
    dfloat theta;

    dfloat3 wallVel; // Prescribed velocity at the physical wall point
    dfloat3 vel; //extrapolated velocity, which will be used on the boundary condition

    #ifdef CONFORMATION_TENSOR
    dfloat conformation[6]; // precomputed Axx, Axy, Axz, Ayy, Ayz, Azz
    // Stored t30 moments at the nearest interior sampling point. Component
    // order matches conformation[]; direction order is x, y, z.
    dfloat conformationFluxT30[18];
    dfloat conformationInteriorVelocityT30[3];
    #endif

}CurvedBoundary;

struct ParticleWallForce {
    dfloat Fx;   // sum of particle forces on wall (x)
    dfloat Fy;   // sum of particle forces on wall (y)
    dfloat Fz;   // sum of particle forces on wall (z)

    dfloat Fn;   // sum of normal force magnitudes
    dfloat Ft;   // sum of tangential force magnitudes

    int    nContacts; // number of particle-wall contacts
};

struct ParticleWallForces {
    ParticleWallForce bottom;
    ParticleWallForce top;
};

// ============================================================================
// DEVICE KERNEL PARAMETER STRUCTURES
// ============================================================================

/**
 * @struct ghostFacePtrs
 * @brief Minimal device-side ghost face pointers
 * @details Extracted from ghostInterfaceData to avoid local memory spills.
 *          The full ghostInterfaceData is never needed on device;
 *          only these 8 face pointers are read/written by the kernel.
 */
struct ghostFacePtrs {
    dfloat* X_0;    ///< Own west face (pop store / neighbor east face load)
    dfloat* X_1;    ///< Own east face
    dfloat* Y_0;    ///< Own south face
    dfloat* Y_1;    ///< Own north face
    dfloat* Z_0;    ///< Own back face (Z-direction, intra-GPU)
    dfloat* Z_1;    ///< Own front face (Z-direction, intra-GPU)
    dfloat* auxZ_0; ///< Received halo from next GPU (top)
    dfloat* auxZ_1; ///< Received halo from prev GPU (bottom)
};

/**
 * @struct DeviceKernelParams
 * @brief Parameters for gpuMomCollisionStream kernel
 * @details Consolidates all parameters previously passed via scattered macros
 *          Passed by value for optimal CUDA performance
 */
struct DeviceKernelParams {
    // Core parameters
    dfloat *fMom;                           ///< Device array of macroscopic moments
    unsigned int *dNodeType;                ///< Device array of node type information
    ghostFacePtrs pop;                      ///< Ghost face pointers (8 ptrs, no host data)
    ghostFacePtrs popCopy;
    unsigned int step;                      ///< Current time step
    bool save;                              ///< Whether to save data
    size_t localNZ; 
    int zStart;
    unsigned int zBlockOffset;              ///< Maps split-kernel launch Z to the local slab block index.
    bool mapBoundaryBlocks;                 ///< Maps a two-plane launch to local Z blocks 0 and N-1.

    gpuDirection rho_macro;
    gpuDirection ux_macro;
    gpuDirection uy_macro;
    gpuDirection uz_macro;

    gpuDirection rho_macroCopy;
    gpuDirection ux_macroCopy;
    gpuDirection uy_macroCopy;
    gpuDirection uz_macroCopy;

    #ifdef SECOND_DIST
    ghostFacePtrs g;
    ghostFacePtrs gCopy;
    gpuDirection g_macro;
    gpuDirection g_macroCopy;
    #endif //SECOND_DIST

    #ifdef A_XX_DIST
        ghostFacePtrs Axx;
        ghostFacePtrs AxxCopy;
        gpuDirection Axx_macro;
        gpuDirection Axx_macroCopy;
    #endif //A_XX_DIST
    #ifdef A_XY_DIST
        ghostFacePtrs Axy;
        ghostFacePtrs AxyCopy;
        gpuDirection Axy_macro;
        gpuDirection Axy_macroCopy;
    #endif //A_XY_DIST
    #ifdef A_XZ_DIST
        ghostFacePtrs Axz;
        ghostFacePtrs AxzCopy;
        gpuDirection Axz_macro;
        gpuDirection Axz_macroCopy;
    #endif //A_XZ_DIST
    #ifdef A_YY_DIST
        ghostFacePtrs Ayy;
        ghostFacePtrs AyyCopy;
        gpuDirection Ayy_macro;
        gpuDirection Ayy_macroCopy;
    #endif //A_YY_DIST
    #ifdef A_YZ_DIST
        ghostFacePtrs Ayz;
        ghostFacePtrs AyzCopy;
        gpuDirection Ayz_macro;
        gpuDirection Ayz_macroCopy;
    #endif //A_YZ_DIST
    #ifdef A_ZZ_DIST
        ghostFacePtrs Azz;
        ghostFacePtrs AzzCopy;
        gpuDirection Azz_macro;
        gpuDirection Azz_macroCopy;
    #endif //A_ZZ_DIST

    #ifdef LAMBDA_DIST
    ghostFacePtrs lambda;
    ghostFacePtrs lambdaCopy;
    gpuDirection lambda_macro;
    gpuDirection lambda_macroCopy;
    #endif //LAMBDA_DIST

    #if defined(NON_NEWTONIAN_FLUID) || defined(CONFORMATION_TENSOR)
    fluidPhaseProps phasePropsA;            ///< Phase-1 fluid properties (viscous + viscoelastic)
    #ifdef PHI_DIST
    fluidPhaseProps phasePropsB;            ///< Phase-2 fluid properties (viscous + viscoelastic)
    ghostFacePtrs phi;
    ghostFacePtrs phiCopy;
    gpuDirection phi_macro;
    gpuDirection nx_macro;
    gpuDirection ny_macro;
    gpuDirection nz_macro;
    gpuDirection mu_macro;

    gpuDirection phi_macroCopy;
    gpuDirection nx_macroCopy;
    gpuDirection ny_macroCopy;
    gpuDirection nz_macroCopy;
    gpuDirection mu_macroCopy;
    #endif //PHI_DIST
    #endif //NON_NEWTONIAN_FLUID || CONFORMATION_TENSOR

    // Conditional parameters with #ifdef guards
    #ifdef DENSITY_CORRECTION
    dfloat* d_mean_rho;                     ///< Mean density for density correction
    #endif //DENSITY_CORRECTION
    
    #ifdef BC_FORCES
    dfloat* d_BC_Fx;                        ///< Boundary condition force X component
    dfloat* d_BC_Fy;                        ///< Boundary condition force Y component
    dfloat* d_BC_Fz;                        ///< Boundary condition force Z component
    #endif //BC_FORCES
    
    #ifdef SAVE_LOCAL_FORCES
    dfloat* d_Local_Fx;                     ///< Local body force X component (for export)
    dfloat* d_Local_Fy;                     ///< Local body force Y component (for export)
    dfloat* d_Local_Fz;                     ///< Local body force Z component (for export)
        #ifdef SECOND_DIST
    dfloat* d_Source_C;                     ///< Temperature source term T_Q_INTERNAL_D_Cp
        #endif
        #ifdef PHI_DIST
    dfloat* d_Source_Phi;                   ///< Phase-field source term
        #endif
        #ifdef LAMBDA_DIST
    dfloat* d_Source_Lambda;                ///< Structure-parameter source term
        #endif
        #ifdef CONFORMATION_TENSOR
            #ifdef A_XX_DIST
    dfloat* d_Source_Gxx;                   ///< Conformation source Gxx
            #endif
            #ifdef A_XY_DIST
    dfloat* d_Source_Gxy;                   ///< Conformation source Gxy
            #endif
            #ifdef A_XZ_DIST
    dfloat* d_Source_Gxz;                   ///< Conformation source Gxz
            #endif
            #ifdef A_YY_DIST
    dfloat* d_Source_Gyy;                   ///< Conformation source Gyy
            #endif
            #ifdef A_YZ_DIST
    dfloat* d_Source_Gyz;                   ///< Conformation source Gyz
            #endif
            #ifdef A_ZZ_DIST
    dfloat* d_Source_Gzz;                   ///< Conformation source Gzz
            #endif
        #endif //CONFORMATION_TENSOR
    #endif //SAVE_LOCAL_FORCES
    
    #ifdef CURVED_BOUNDARY_CONDITION
    CurvedBoundary** d_curvedBC;            ///< Curved boundary condition data
    CurvedBoundary* d_curvedBC_array;       ///< Curved boundary condition array
    #endif //CURVED_BOUNDARY_CONDITION
};

enum class ZKernelLaunchRegion {
    All,
    Interior,
    Boundaries
};

// ============================================================================
// HOST FUNCTION PARAMETER STRUCTURES
// ============================================================================

/**
 * @struct SaveDataParams
 * @brief Unified parameters for saveMacr() and saveVarVTK() host functions
 * @details Consolidates all save operation parameters (macroscopic and VTK)
 *          Optional filename for VTK output; omit for binary macroscopic saves
 */
struct SaveDataParams {
    // Core fields
    dfloat* h_fMom;
    dfloat* h_rho;
    dfloat* h_ux;
    dfloat* h_uy;
    dfloat* h_uz;
    unsigned int* h_nodeType;
    
    // Optional: filename for VTK saves (empty/nullptr for binary saves)
    const char* vtkFilename;
    
    // Optional fields (compiled conditionally)
    #ifdef OMEGA_FIELD
    dfloat* h_omega;
    #endif
    
    #ifdef SECOND_DIST
    dfloat* h_C;
    #endif
    
    #ifdef PHI_DIST
    dfloat* h_phi;
    #endif
    
    #ifdef LAMBDA_DIST
    dfloat* h_lambda;
    #endif
    
    #ifdef A_XX_DIST
    dfloat* h_Axx;
    #endif
    #ifdef A_XY_DIST
    dfloat* h_Axy;
    #endif
    #ifdef A_XZ_DIST
    dfloat* h_Axz;
    #endif
    #ifdef A_YY_DIST
    dfloat* h_Ayy;
    #endif
    #ifdef A_YZ_DIST
    dfloat* h_Ayz;
    #endif
    #ifdef A_ZZ_DIST
    dfloat* h_Azz;
    #endif
    
    #if NODE_TYPE_SAVE
    unsigned int* h_nodeTypeSave;
    #endif
    
    #ifdef BC_FORCES
    dfloat* h_BC_Fx;
    dfloat* h_BC_Fy;
    dfloat* h_BC_Fz;
    #endif
    
    #ifdef SAVE_LOCAL_FORCES
    dfloat* h_Local_Fx;
    dfloat* h_Local_Fy;
    dfloat* h_Local_Fz;
        #ifdef SECOND_DIST
    dfloat* h_Source_C;
        #endif
        #ifdef PHI_DIST
    dfloat* h_Source_Phi;
        #endif
        #ifdef LAMBDA_DIST
    dfloat* h_Source_Lambda;
        #endif
        #ifdef CONFORMATION_TENSOR
            #ifdef A_XX_DIST
    dfloat* h_Source_Gxx;
            #endif
            #ifdef A_XY_DIST
    dfloat* h_Source_Gxy;
            #endif
            #ifdef A_XZ_DIST
    dfloat* h_Source_Gxz;
            #endif
            #ifdef A_YY_DIST
    dfloat* h_Source_Gyy;
            #endif
            #ifdef A_YZ_DIST
    dfloat* h_Source_Gyz;
            #endif
            #ifdef A_ZZ_DIST
    dfloat* h_Source_Gzz;
            #endif
        #endif //CONFORMATION_TENSOR
    #endif //SAVE_LOCAL_FORCES

    // Metadata
    unsigned int nSteps;
    std::atomic<bool>* savingMacrVtk;
    std::vector<std::atomic<bool>>* savingMacrBin;
};

/**
 * @struct TreatDataParams
 * @brief Parameters for treatData() host function
 * @details Consolidates all post-processing treatment parameters
 *          Passed by pointer for cleaner function signatures
 */
struct TreatDataParams {
    // Core fields
    dfloat* h_fMom;
    dfloat* d_fMom;
    
    // Optional fields (compiled conditionally)
    #if MEAN_FLOW
    dfloat* d_fMom_mean;
    #endif
    
    #ifdef BC_FORCES
    dfloat* d_BC_Fx;
    dfloat* d_BC_Fy;
    dfloat* d_BC_Fz;
    #endif
    
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
    #endif
    
    // Metadata
    unsigned int step;
    size_t zOffset;
};

#endif //__GLOBAL_STRUCTS_H
