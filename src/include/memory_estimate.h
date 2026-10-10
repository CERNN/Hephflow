#ifndef __MEMORY_ESTIMATE_H
#define __MEMORY_ESTIMATE_H

#include <cstddef>
#include <cstdio>

#include "../var.h"

struct CompileTimeMemoryEstimate {
    size_t hostPrimary = 0;
    size_t hostCheckpointInterfaces = 0;

    constexpr size_t hostTotal() const {
        return hostPrimary + hostCheckpointInterfaces;
    }

};

constexpr size_t enabledScalarDistributionCount() {
    size_t count = 0;
    #ifdef SECOND_DIST
    ++count;
#endif // SECOND_DIST
    #ifdef PHI_DIST
    ++count;
#endif // PHI_DIST
    #ifdef LAMBDA_DIST
    ++count;
#endif // LAMBDA_DIST
    #ifdef A_XX_DIST
    ++count;
#endif // A_XX_DIST
    #ifdef A_XY_DIST
    ++count;
#endif // A_XY_DIST
    #ifdef A_XZ_DIST
    ++count;
#endif // A_XZ_DIST
    #ifdef A_YY_DIST
    ++count;
#endif // A_YY_DIST
    #ifdef A_YZ_DIST
    ++count;
#endif // A_YZ_DIST
    #ifdef A_ZZ_DIST
    ++count;
#endif // A_ZZ_DIST
    return count;
}

constexpr size_t enabledConformationDistributionCount() {
    size_t count = 0;
    #ifdef A_XX_DIST
    ++count;
#endif // A_XX_DIST
    #ifdef A_XY_DIST
    ++count;
#endif // A_XY_DIST
    #ifdef A_XZ_DIST
    ++count;
#endif // A_XZ_DIST
    #ifdef A_YY_DIST
    ++count;
#endif // A_YY_DIST
    #ifdef A_YZ_DIST
    ++count;
#endif // A_YZ_DIST
    #ifdef A_ZZ_DIST
    ++count;
#endif // A_ZZ_DIST
    return count;
}

constexpr size_t enabledMacroInterfaceFieldCount() {
    // rho, ux, uy and uz are always allocated by interfaceMalloc().
    size_t count = 4;
    #ifdef SECOND_DIST
    ++count; // concentration
#endif // SECOND_DIST
    #ifdef PHI_DIST
    count += 5; // phi, nx, ny, nz and mu
#endif // PHI_DIST
    #ifdef LAMBDA_DIST
    ++count;
#endif // LAMBDA_DIST
    #ifdef A_XX_DIST
    ++count;
#endif // A_XX_DIST
    #ifdef A_XY_DIST
    ++count;
#endif // A_XY_DIST
    #ifdef A_XZ_DIST
    ++count;
#endif // A_XZ_DIST
    #ifdef A_YY_DIST
    ++count;
#endif // A_YY_DIST
    #ifdef A_YZ_DIST
    ++count;
#endif // A_YZ_DIST
    #ifdef A_ZZ_DIST
    ++count;
#endif // A_ZZ_DIST
    return count;
}

constexpr CompileTimeMemoryEstimate compileTimeMemoryEstimate() {
    CompileTimeMemoryEstimate estimate{};
    const size_t scalarBytes = MEM_SIZE_SCALAR;
    const size_t nodeTypeBytes = sizeof(unsigned int) * NUMBER_LBM_NODES;

    // Primary host field: full decoded moments, output macros and node types.
    estimate.hostPrimary = MEM_SIZE_MOM + 4 * scalarBytes + nodeTypeBytes;
    #ifdef OMEGA_FIELD
    estimate.hostPrimary += scalarBytes;
#endif // OMEGA_FIELD
    #ifdef SECOND_DIST
    estimate.hostPrimary += scalarBytes;
#endif // SECOND_DIST
    #ifdef PHI_DIST
    estimate.hostPrimary += scalarBytes;
#endif // PHI_DIST
    #ifdef LAMBDA_DIST
    estimate.hostPrimary += scalarBytes;
#endif // LAMBDA_DIST
    #ifdef A_XX_DIST
    estimate.hostPrimary += scalarBytes;
#endif // A_XX_DIST
    #ifdef A_XY_DIST
    estimate.hostPrimary += scalarBytes;
#endif // A_XY_DIST
    #ifdef A_XZ_DIST
    estimate.hostPrimary += scalarBytes;
#endif // A_XZ_DIST
    #ifdef A_YY_DIST
    estimate.hostPrimary += scalarBytes;
#endif // A_YY_DIST
    #ifdef A_YZ_DIST
    estimate.hostPrimary += scalarBytes;
#endif // A_YZ_DIST
    #ifdef A_ZZ_DIST
    estimate.hostPrimary += scalarBytes;
#endif // A_ZZ_DIST
    #if MEAN_FLOW
    estimate.hostPrimary += MEM_SIZE_MOM + 4 * scalarBytes;
        #ifdef SECOND_DIST
    estimate.hostPrimary += scalarBytes;
#endif // SECOND_DIST
        #ifdef PHI_DIST
    estimate.hostPrimary += scalarBytes;
#endif // PHI_DIST
        #ifdef LAMBDA_DIST
    estimate.hostPrimary += scalarBytes;
#endif // LAMBDA_DIST
#endif // MEAN_FLOW
    #if defined(BC_FORCES) && defined(SAVE_BC_FORCES)
    estimate.hostPrimary += 3 * scalarBytes;
#endif // defined(BC_FORCES) && defined(SAVE_BC_FORCES)
    #ifdef SAVE_LOCAL_FORCES
    estimate.hostPrimary += 3 * scalarBytes;
        #ifdef SECOND_DIST
    estimate.hostPrimary += scalarBytes;
#endif // SECOND_DIST
        #ifdef PHI_DIST
    estimate.hostPrimary += scalarBytes;
#endif // PHI_DIST
        #ifdef LAMBDA_DIST
    estimate.hostPrimary += scalarBytes;
#endif // LAMBDA_DIST
        #ifdef CONFORMATION_TENSOR
    estimate.hostPrimary += enabledConformationDistributionCount() * scalarBytes;
#endif // CONFORMATION_TENSOR
#endif // SAVE_LOCAL_FORCES
    #if NODE_TYPE_SAVE
    estimate.hostPrimary += nodeTypeBytes;
#endif // NODE_TYPE_SAVE
    #ifdef DENSITY_CORRECTION
    estimate.hostPrimary += sizeof(dfloat) + N_GPUS * (sizeof(dfloat) + sizeof(unsigned long long));
#endif // DENSITY_CORRECTION

    return estimate;
}

inline size_t runtimeHostCheckpointInterfaceBytes(
    const DomainDecomposition& decomposition) {
    (void)decomposition;
    return 0;
}

inline double bytesToMiB(size_t bytes) {
    return static_cast<double>(bytes) / (1024.0 * 1024.0);
}

inline double bytesToGiB(size_t bytes) {
    return static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0);
}

inline size_t runtimeDeviceBytes(const GpuDomainLayout& layout) {
    size_t bytes = layout.momentStorageBytes + sizeof(unsigned int) * layout.paddedNodeCount;
    size_t hydroInterfaces = sizeof(dfloat) * static_cast<size_t>(QF) * (2 * layout.ghostFaceYZCount + 2 * layout.ghostFaceXZCount + 2 * layout.ghostFaceXYCount + 4 * layout.exchangePlaneCount);
    #ifdef STREAMING_AB
    hydroInterfaces += 2 * sizeof(dfloat) * static_cast<size_t>(QF) * (layout.ghostFaceYZCount + layout.ghostFaceXZCount + layout.ghostFaceXYCount);
#endif // STREAMING_AB
    bytes += hydroInterfaces;

    #ifdef CONVECTION_DIFFUSION_TRANSPORT
    bytes += sizeof(dfloat) * static_cast<size_t>(GF) * enabledScalarDistributionCount() * (2 * layout.ghostFaceYZCount + 2 * layout.ghostFaceXZCount + 2 * layout.ghostFaceXYCount + 4 * layout.exchangePlaneCount);
#endif // CONVECTION_DIFFUSION_TRANSPORT
    bytes += enabledMacroInterfaceFieldCount() * 6 * static_cast<size_t>(NX) * NY * sizeof(dfloat);

    size_t scalarFieldCount = 0;
    #ifdef BC_FORCES
    scalarFieldCount += 3;
#endif // BC_FORCES
    #ifdef SAVE_LOCAL_FORCES
    scalarFieldCount += 3;
        #ifdef SECOND_DIST
    ++scalarFieldCount;
#endif // SECOND_DIST
        #ifdef PHI_DIST
    ++scalarFieldCount;
#endif // PHI_DIST
        #ifdef LAMBDA_DIST
    ++scalarFieldCount;
#endif // LAMBDA_DIST
        #ifdef CONFORMATION_TENSOR
    scalarFieldCount += enabledConformationDistributionCount();
#endif // CONFORMATION_TENSOR
#endif // SAVE_LOCAL_FORCES
    bytes += scalarFieldCount * layout.scalarBytes;
    #ifdef DENSITY_CORRECTION
    bytes += sizeof(dfloat) + 2 * layout.numBlocksX * layout.numBlocksY * layout.numBlocksZ * (sizeof(dfloat) + sizeof(unsigned long long));
#endif // DENSITY_CORRECTION
    #ifdef CONFORMATION_PHYSICAL_A_HALO
    bytes += 4 * sizeof(dfloat) * 12 * (layout.ghostFaceYZCount + layout.ghostFaceXZCount + layout.ghostFaceXYCount + layout.exchangePlaneCount);
#endif // CONFORMATION_PHYSICAL_A_HALO
    return bytes;
}

inline void printCompileTimeMemoryEstimate() {
    const DomainDecomposition decomposition(N_GPUS);
    CompileTimeMemoryEstimate estimate = compileTimeMemoryEstimate();
    estimate.hostCheckpointInterfaces = runtimeHostCheckpointInterfaceBytes(decomposition);
    size_t allDevicePartitions = 0;

    std::printf("\nCompile-time stationary memory estimate\n");
    std::printf("  Domain: %d x %d x %d, GPUs/partitions: %u\n", NX, NY, NZ, N_GPUS);
    std::printf("  Host primary/save fields:       %10.2f MiB\n", bytesToMiB(estimate.hostPrimary));
    std::printf("    shared h_fMom (included):     %10.2f MiB\n", bytesToMiB(MEM_SIZE_MOM));
    std::printf("  Host checkpoint interfaces:     %10.2f MiB\n", bytesToMiB(estimate.hostCheckpointInterfaces));
    std::printf("  Host stationary total:          %10.2f MiB  (%0.3f GiB)\n",
        bytesToMiB(estimate.hostTotal()), bytesToGiB(estimate.hostTotal()));
    for (unsigned int g = 0; g < N_GPUS; ++g) {
        const size_t partitionBytes = runtimeDeviceBytes(decomposition[g]);
        allDevicePartitions += partitionBytes;
        std::printf("  Partition %u: z=[%zu,%zu), blocksZ=%zu: %10.2f MiB  (%0.3f GiB)\n",
            g, decomposition[g].zStart,
            decomposition[g].zStart + decomposition[g].localNZ,
            decomposition[g].numBlocksZ, bytesToMiB(partitionBytes),
            bytesToGiB(partitionBytes));
    }

    for (unsigned int g = 0; g < N_GPUS; ++g) {
        bool firstPartitionForDevice = true;
        size_t physicalDeviceBytes = 0;
        unsigned int partitionCount = 0;
        for (unsigned int other = 0; other < N_GPUS; ++other) {
            if (GPUS_TO_USE[other] == GPUS_TO_USE[g]) {
                ++partitionCount;
                physicalDeviceBytes += runtimeDeviceBytes(decomposition[other]);
                if (other < g) firstPartitionForDevice = false;
            }
        }
        if (firstPartitionForDevice) {
            std::printf("  CUDA device %u (%u partition%s): %10.2f MiB  (%0.3f GiB)\n",
                GPUS_TO_USE[g], partitionCount, partitionCount == 1 ? "" : "s",
                bytesToMiB(physicalDeviceBytes), bytesToGiB(physicalDeviceBytes));
        }
    }
    std::printf("  GPU aggregate allocations:      %10.2f MiB  (%0.3f GiB)\n",
        bytesToMiB(allDevicePartitions), bytesToGiB(allDevicePartitions));
    std::printf("  Excludes CUDA context/allocator overhead, initialization scratch,\n");
    std::printf("  runtime-sized curved-boundary data, particles, and file-I/O temporaries.\n\n");
}

#endif //__MEMORY_ESTIMATE_H
