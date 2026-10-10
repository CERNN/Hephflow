/**
 * @file ghostInterface.cuh
 * @brief Host-side allocation and transfer helpers for GPU ghost interfaces.
 */
#ifndef GHOST_INTERFACE_CUH
#define GHOST_INTERFACE_CUH

#include <cuda_runtime.h>

#include "globalStructs.h"
#include "include/domain_decomposition.h"

/** @brief Frees device and pinned-host storage owned by a ghost interface. */
void interfaceFree(ghostInterfaceData& ghostInterface, ghostInterfaceData& ghostInterfaceCopy,
    macroInterfaceGPUData& macroInterfaceGPU, macroInterfaceGPUData& macroInterfaceGPUCopy);

/** @brief Copies all populated faces between two ghost-data collections. */
void interfaceCudaMemcpy(GhostInterfaceData& ghostInterface, ghostData& dst,
    const ghostData& src, cudaMemcpyKind kind, int quantityCount, int gpuIndex);

/** @brief Swaps the active and alternate population interfaces for AB streaming. */
void swapGhostInterfaces(GhostInterfaceData& ghostInterface,
    GhostInterfaceData& ghostInterfaceCopy);

/** @brief Allocates and zero-initializes all interfaces required by one partition. */
void interfaceMalloc(ghostInterfaceData& ghostInterface,
    ghostInterfaceData& ghostInterfaceCopy,
    macroInterfaceGPUData& macroInterfaceGPU,
    macroInterfaceGPUData& macroInterfaceGPUCopy,
    const GpuDomainLayout& layout);

#endif // GHOST_INTERFACE_CUH
