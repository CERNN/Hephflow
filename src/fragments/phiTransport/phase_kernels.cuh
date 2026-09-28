/**
 * @file phase_kernels.cuh
 * @brief CUDA kernels that compute phase-field auxiliary quantities.
 */

#ifndef __PHASE_KERNELS_CUH
#define __PHASE_KERNELS_CUH

#include "../../globalFunctions.h"
#include "../../var.h"
#include "../../include/interface.h"

#ifdef PHI_DIST

/** Compute grad(phi) and laplacian(phi) with the D3Q19 isotropic stencil. */
__global__ void gpuComputePhaseNormals(
    dfloat *fMom,
    unsigned int *dNodeType,
    macroInterfaceGPUData macroInterfaceGPU,
    size_t localNZ,
    int zStart);

#ifdef PHASE_NEEDS_CHEMICAL_POTENTIAL
/** Compute the chemical potential used by CH transport or mu-grad(phi) force. */
__global__ void gpuComputeChemicalPotential(
    dfloat *fMom,
    unsigned int *dNodeType,
    size_t localNZ,
    int zStart);
#endif

#ifdef PHASE_TRANSPORT_CAHN_HILLIARD
/** Compute laplacian(mu) for Cahn-Hilliard transport. */
__global__ void gpuComputeLaplacianMu(
    dfloat *fMom,
    unsigned int *dNodeType,
    macroInterfaceGPUData macroInterfaceGPU,
    size_t localNZ,
    int zStart);
#endif

#ifdef PHASE_NEEDS_NORMAL_HALOS
/** Compute div(lambda(phi) n) for CAC and div(n) for CSF. */
__global__ void gpuComputePhaseFluxDivergences(
    dfloat *fMom,
    unsigned int *dNodeType,
    macroInterfaceGPUData macroInterfaceGPU,
    size_t localNZ,
    int zStart);
#endif

#endif // PHI_DIST

#endif // __PHASE_KERNELS_CUH
