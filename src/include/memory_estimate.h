#ifndef __MEMORY_ESTIMATE_H
#define __MEMORY_ESTIMATE_H

#include <cstddef>
#include <cstdio>

#include "../var.h"

struct CompileTimeMemoryEstimate {
    size_t hostPrimary = 0;
    size_t hostCheckpointInterfaces = 0;
    size_t deviceMoments = 0;
    size_t deviceNodeTypes = 0;
    size_t devicePopulationInterfaces = 0;
    size_t deviceMacroInterfaces = 0;
    size_t deviceOptionalFields = 0;

    constexpr size_t hostTotal() const {
        return hostPrimary + hostCheckpointInterfaces;
    }

    constexpr size_t devicePerPartition() const {
        return deviceMoments + deviceNodeTypes + devicePopulationInterfaces
            + deviceMacroInterfaces + deviceOptionalFields;
    }
};

constexpr size_t enabledScalarDistributionCount() {
    size_t count = 0;
    #ifdef SECOND_DIST
    ++count;
    #endif
    #ifdef PHI_DIST
    ++count;
    #endif
    #ifdef LAMBDA_DIST
    ++count;
    #endif
    #ifdef A_XX_DIST
    ++count;
    #endif
    #ifdef A_XY_DIST
    ++count;
    #endif
    #ifdef A_XZ_DIST
    ++count;
    #endif
    #ifdef A_YY_DIST
    ++count;
    #endif
    #ifdef A_YZ_DIST
    ++count;
    #endif
    #ifdef A_ZZ_DIST
    ++count;
    #endif
    return count;
}

constexpr size_t enabledConformationDistributionCount() {
    size_t count = 0;
    #ifdef A_XX_DIST
    ++count;
    #endif
    #ifdef A_XY_DIST
    ++count;
    #endif
    #ifdef A_XZ_DIST
    ++count;
    #endif
    #ifdef A_YY_DIST
    ++count;
    #endif
    #ifdef A_YZ_DIST
    ++count;
    #endif
    #ifdef A_ZZ_DIST
    ++count;
    #endif
    return count;
}

constexpr size_t enabledMacroInterfaceFieldCount() {
    // rho, ux, uy and uz are always allocated by interfaceMalloc().
    size_t count = 4;
    #ifdef SECOND_DIST
    ++count; // concentration
    #endif
    #ifdef PHI_DIST
    count += 5; // phi, nx, ny, nz and mu
    #endif
    #ifdef LAMBDA_DIST
    ++count;
    #endif
    #ifdef A_XX_DIST
    ++count;
    #endif
    #ifdef A_XY_DIST
    ++count;
    #endif
    #ifdef A_XZ_DIST
    ++count;
    #endif
    #ifdef A_YY_DIST
    ++count;
    #endif
    #ifdef A_YZ_DIST
    ++count;
    #endif
    #ifdef A_ZZ_DIST
    ++count;
    #endif
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
    #endif
    #ifdef SECOND_DIST
    estimate.hostPrimary += scalarBytes;
    #endif
    #ifdef PHI_DIST
    estimate.hostPrimary += scalarBytes;
    #endif
    #ifdef LAMBDA_DIST
    estimate.hostPrimary += scalarBytes;
    #endif
    #ifdef A_XX_DIST
    estimate.hostPrimary += scalarBytes;
    #endif
    #ifdef A_XY_DIST
    estimate.hostPrimary += scalarBytes;
    #endif
    #ifdef A_XZ_DIST
    estimate.hostPrimary += scalarBytes;
    #endif
    #ifdef A_YY_DIST
    estimate.hostPrimary += scalarBytes;
    #endif
    #ifdef A_YZ_DIST
    estimate.hostPrimary += scalarBytes;
    #endif
    #ifdef A_ZZ_DIST
    estimate.hostPrimary += scalarBytes;
    #endif
    #if MEAN_FLOW
    estimate.hostPrimary += MEM_SIZE_MOM + 4 * scalarBytes;
        #ifdef SECOND_DIST
    estimate.hostPrimary += scalarBytes;
        #endif
        #ifdef PHI_DIST
    estimate.hostPrimary += scalarBytes;
        #endif
        #ifdef LAMBDA_DIST
    estimate.hostPrimary += scalarBytes;
        #endif
    #endif
    #if defined(BC_FORCES) && defined(SAVE_BC_FORCES)
    estimate.hostPrimary += 3 * scalarBytes;
    #endif
    #ifdef SAVE_LOCAL_FORCES
    estimate.hostPrimary += 3 * scalarBytes;
        #ifdef SECOND_DIST
    estimate.hostPrimary += scalarBytes;
        #endif
        #ifdef PHI_DIST
    estimate.hostPrimary += scalarBytes;
        #endif
        #ifdef LAMBDA_DIST
    estimate.hostPrimary += scalarBytes;
        #endif
        #ifdef CONFORMATION_TENSOR
    estimate.hostPrimary += enabledConformationDistributionCount() * scalarBytes;
        #endif
    #endif
    #if NODE_TYPE_SAVE
    estimate.hostPrimary += nodeTypeBytes;
    #endif
    #ifdef DENSITY_CORRECTION
    estimate.hostPrimary += sizeof(dfloat);
    #endif

    const size_t xyzInterfaceFaces = NUMBER_GHOST_FACE_YZ
        + NUMBER_GHOST_FACE_XZ + NUMBER_GHOST_FACE_XY_LOCAL;
    if (LOAD_CHECKPOINT || CHECKPOINT_SAVE) {
        size_t checkpointPerPartition = 2 * sizeof(dfloat)
            * static_cast<size_t>(QF) * xyzInterfaceFaces;
        #if defined(SECOND_DIST) || defined(PHI_DIST) || defined(LAMBDA_DIST) || \
            defined(A_XX_DIST) || defined(A_XY_DIST) || defined(A_XZ_DIST) || \
            defined(A_YY_DIST) || defined(A_YZ_DIST) || defined(A_ZZ_DIST)
        checkpointPerPartition += 2 * sizeof(dfloat) * static_cast<size_t>(GF)
            * enabledScalarDistributionCount() * xyzInterfaceFaces;
        #endif
        estimate.hostCheckpointInterfaces = N_GPUS * checkpointPerPartition;
    }

    // Fixed device allocations made once for every domain partition.
    estimate.deviceMoments = MEM_SIZE_MOM_STORAGE_LOCAL;
    estimate.deviceNodeTypes = sizeof(unsigned int) * NUMBER_LBM_NODES_LOCAL;

    size_t hydroPopulationInterfaces = sizeof(dfloat) * static_cast<size_t>(QF)
        * (2 * NUMBER_GHOST_FACE_YZ + 2 * NUMBER_GHOST_FACE_XZ
           + 2 * NUMBER_GHOST_FACE_XY_LOCAL + 4 * NUMBER_GHOST_FACE_XY_AUX);
    #ifdef STREAMING_AB
    hydroPopulationInterfaces += 2 * sizeof(dfloat) * static_cast<size_t>(QF)
        * xyzInterfaceFaces;
    #endif
    size_t scalarPopulationInterfaces = 0;
    #if defined(SECOND_DIST) || defined(PHI_DIST) || defined(LAMBDA_DIST) || \
        defined(A_XX_DIST) || defined(A_XY_DIST) || defined(A_XZ_DIST) || \
        defined(A_YY_DIST) || defined(A_YZ_DIST) || defined(A_ZZ_DIST)
    scalarPopulationInterfaces = sizeof(dfloat) * static_cast<size_t>(GF)
        * enabledScalarDistributionCount()
        * (2 * NUMBER_GHOST_FACE_YZ + 2 * NUMBER_GHOST_FACE_XZ
           + 6 * NUMBER_GHOST_FACE_XY_LOCAL);
    #endif
    estimate.devicePopulationInterfaces = hydroPopulationInterfaces
        + scalarPopulationInterfaces;

    const size_t macroPlaneBytes = static_cast<size_t>(NX) * NY * sizeof(dfloat);
    estimate.deviceMacroInterfaces = enabledMacroInterfaceFieldCount()
        * 6 * macroPlaneBytes;

    #ifdef BC_FORCES
    estimate.deviceOptionalFields += 3 * scalarBytes;
    #endif
    #ifdef SAVE_LOCAL_FORCES
    estimate.deviceOptionalFields += 3 * scalarBytes;
        #ifdef SECOND_DIST
    estimate.deviceOptionalFields += scalarBytes;
        #endif
        #ifdef PHI_DIST
    estimate.deviceOptionalFields += scalarBytes;
        #endif
        #ifdef LAMBDA_DIST
    estimate.deviceOptionalFields += scalarBytes;
        #endif
        #ifdef CONFORMATION_TENSOR
    estimate.deviceOptionalFields += enabledConformationDistributionCount() * scalarBytes;
        #endif
    #endif
    #ifdef DENSITY_CORRECTION
    estimate.deviceOptionalFields += sizeof(dfloat);
    #endif

    return estimate;
}

inline double bytesToMiB(size_t bytes) {
    return static_cast<double>(bytes) / (1024.0 * 1024.0);
}

inline double bytesToGiB(size_t bytes) {
    return static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0);
}

inline void printCompileTimeMemoryEstimate() {
    constexpr CompileTimeMemoryEstimate estimate = compileTimeMemoryEstimate();
    const size_t devicePerPartition = estimate.devicePerPartition();
    const size_t allDevicePartitions = devicePerPartition * N_GPUS;

    std::printf("\nCompile-time stationary memory estimate\n");
    std::printf("  Domain: %d x %d x %d, GPUs/partitions: %u\n", NX, NY, NZ, N_GPUS);
    std::printf("  Host primary/save fields:       %10.2f MiB\n", bytesToMiB(estimate.hostPrimary));
    std::printf("    shared h_fMom (included):     %10.2f MiB\n", bytesToMiB(MEM_SIZE_MOM));
    std::printf("  Host checkpoint interfaces:     %10.2f MiB\n", bytesToMiB(estimate.hostCheckpointInterfaces));
    std::printf("  Host stationary total:          %10.2f MiB  (%0.3f GiB)\n",
        bytesToMiB(estimate.hostTotal()), bytesToGiB(estimate.hostTotal()));
    std::printf("  GPU moments per partition:      %10.2f MiB\n", bytesToMiB(estimate.deviceMoments));
    std::printf("  GPU node types per partition:   %10.2f MiB\n", bytesToMiB(estimate.deviceNodeTypes));
    std::printf("  GPU population interfaces:      %10.2f MiB\n", bytesToMiB(estimate.devicePopulationInterfaces));
    std::printf("  GPU macro interfaces:           %10.2f MiB\n", bytesToMiB(estimate.deviceMacroInterfaces));
    std::printf("  GPU optional stationary fields: %10.2f MiB\n", bytesToMiB(estimate.deviceOptionalFields));
    std::printf("  GPU per partition:              %10.2f MiB  (%0.3f GiB)\n",
        bytesToMiB(devicePerPartition), bytesToGiB(devicePerPartition));

    for (unsigned int g = 0; g < N_GPUS; ++g) {
        bool firstPartitionForDevice = true;
        unsigned int partitionCount = 0;
        for (unsigned int other = 0; other < N_GPUS; ++other) {
            if (GPUS_TO_USE[other] == GPUS_TO_USE[g]) {
                ++partitionCount;
                if (other < g) firstPartitionForDevice = false;
            }
        }
        if (firstPartitionForDevice) {
            const size_t physicalDeviceBytes = devicePerPartition * partitionCount;
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
