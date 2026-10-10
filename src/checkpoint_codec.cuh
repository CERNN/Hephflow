#ifndef HEPHFLOW_CHECKPOINT_CODEC_CUH
#define HEPHFLOW_CHECKPOINT_CODEC_CUH

#include "globalStructs.h"
#include "var.h"
#include "globalFunctions.h"
#include "non_newtonian/conformation_representation.cuh"
#define CCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING
#include <cuda/barrier>
#include <vector>

struct CanonicalCheckpointHostState {
    std::vector<dfloat> hydro;
    #ifdef SECOND_DIST
    std::vector<dfloat> g;
    #endif // SECOND_DIST
    #ifdef PHI_DIST
    std::vector<dfloat> phi;
    #endif // PHI_DIST
    #ifdef LAMBDA_DIST
    std::vector<dfloat> lambda;
    #endif // LAMBDA_DIST
    #ifdef A_XX_DIST
    std::vector<dfloat> Axx;
    #endif // A_XX_DIST
    #ifdef A_XY_DIST
    std::vector<dfloat> Axy;
    #endif // A_XY_DIST
    #ifdef A_XZ_DIST
    std::vector<dfloat> Axz;
    #endif // A_XZ_DIST
    #ifdef A_YY_DIST
    std::vector<dfloat> Ayy;
    #endif // A_YY_DIST
    #ifdef A_YZ_DIST
    std::vector<dfloat> Ayz;
    #endif // A_YZ_DIST
    #ifdef A_ZZ_DIST
    std::vector<dfloat> Azz;
    #endif // A_ZZ_DIST

    /** @brief Resizes every enabled population vector for the global domain. */
    void resizeForGlobalDomain();
};

struct CanonicalPopulationDeviceBuffers {
    dfloat* hydro = nullptr;
    #ifdef SECOND_DIST
    dfloat* g = nullptr;
    #endif // SECOND_DIST
    #ifdef PHI_DIST
    dfloat* phi = nullptr;
    #endif // PHI_DIST
    #ifdef LAMBDA_DIST
    dfloat* lambda = nullptr;
    #endif // LAMBDA_DIST
    #ifdef A_XX_DIST
    dfloat* Axx = nullptr;
    #endif // A_XX_DIST
    #ifdef A_XY_DIST
    dfloat* Axy = nullptr;
    #endif // A_XY_DIST
    #ifdef A_XZ_DIST
    dfloat* Axz = nullptr;
    #endif // A_XZ_DIST
    #ifdef A_YY_DIST
    dfloat* Ayy = nullptr;
    #endif // A_YY_DIST
    #ifdef A_YZ_DIST
    dfloat* Ayz = nullptr;
    #endif // A_YZ_DIST
    #ifdef A_ZZ_DIST
    dfloat* Azz = nullptr;
    #endif // A_ZZ_DIST
};

/**
 * @brief Decodes runtime populations into canonical physical-node order.
 * @param fMom Runtime moment storage.
 * @param ghostInterface Current population interfaces.
 * @param ghostInterfaceCopy Alternate population interfaces.
 * @param output Canonical population buffers written by the kernel.
 * @param localNZ Number of physical planes in this GPU slab.
 * @param zStart Global Z coordinate of the slab.
 * @param numBlocksZ Number of CUDA blocks in the slab's Z direction.
 * @param stepParity Runtime population parity to decode.
 */
__global__ void gpuPackCanonicalCheckpointPopulations(
    const dfloat* fMom,
    ghostInterfaceData ghostInterface,
    ghostInterfaceData ghostInterfaceCopy,
    CanonicalPopulationDeviceBuffers output,
    size_t localNZ,
    int zStart,
    size_t numBlocksZ,
    bool stepParity);

/**
 * @brief Encodes canonical populations into the active runtime layout.
 * @param input Canonical population buffers, including Z halos.
 * @param ghostInterface Runtime interfaces to populate.
 * @param localNZ Number of physical planes in this GPU slab.
 * @param numBlocksZ Number of CUDA blocks in the slab's Z direction.
 * @param zStart Global Z coordinate of the slab.
 * @param completedStepParity Population parity represented by the checkpoint.
 */
__global__ void gpuEncodeCanonicalCheckpointPopulations( CanonicalPopulationDeviceBuffers input, ghostInterfaceData ghostInterface, size_t localNZ, size_t numBlocksZ, int zStart, bool completedStepParity);

/**
 * @brief Captures a GPU slab into the global canonical population snapshot.
 * @param fMom Runtime moment storage.
 * @param ghostInterface Current population interfaces.
 * @param ghostInterfaceCopy Alternate population interfaces.
 * @param layout This GPU's physical slab layout.
 * @param threadBlock CUDA block dimensions for codec kernels.
 * @param stepParity Runtime population parity to decode.
 * @param output Destination global canonical snapshot.
 */
__host__ void captureCanonicalCheckpointPopulations( const dfloat* fMom, ghostInterfaceData ghostInterface, ghostInterfaceData ghostInterfaceCopy, const GpuDomainLayout& layout, dim3 threadBlock, bool stepParity, CanonicalCheckpointHostState& output);

/**
 * @brief Restores one GPU slab from canonical checkpoint populations.
 * @param input Global canonical checkpoint snapshot.
 * @param ghostInterface Runtime interfaces to populate.
 * @param layout This GPU's physical slab layout.
 * @param threadBlock CUDA block dimensions for codec kernels.
 * @param completedStepParity Population parity represented by the checkpoint.
 */
__host__ void restoreCanonicalCheckpointPopulations( const CanonicalCheckpointHostState& input, ghostInterfaceData ghostInterface, const GpuDomainLayout& layout, dim3 threadBlock, bool completedStepParity);

#endif // HEPHFLOW_CHECKPOINT_CODEC_CUH
