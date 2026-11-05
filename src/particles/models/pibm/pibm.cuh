#ifndef __PARTICLE_MODEL_PIBM_CUH
#define __PARTICLE_MODEL_PIBM_CUH


#include "../../../globalStructs.h"
#include "../../../globalFunctions.h"
#include "../particleSharedFunctions.cuh"
#include "../../../include/interface.h"
#include "../../../include/errorDef.h"
#include "../../../saveData.cuh"
#include "./../../class/Particle.cuh"
#include "pibmVar.h"

#include "../dem/particleMovement.cuh"
#include "../dem/collision/collisionDetection.cuh"

#ifdef PARTICLE_MODEL

namespace pibm_detail {

#if PIBM_COUPLING_MODE == PIBM_COUPLING_TRILINEAR
constexpr int PIBM_STENCIL_WIDTH = 2;
#else
constexpr int PIBM_STENCIL_WIDTH = 4;
#endif

struct PibmStencil {
    int base[3];
    dfloat weight[3][4];
};

/**
 * @brief Validate that a particle center can be converted to a lattice stencil.
 * @param x Particle center x-coordinate.
 * @param y Particle center y-coordinate.
 * @param z Particle center z-coordinate.
 * @return True when all coordinates are finite and inside the local domain.
 */
__device__ __forceinline__ bool validPibmParticlePosition(float x, dfloat y, dfloat z){
    return isfinite(x) && isfinite(y) && isfinite(z) &&
           x >= 0.0_df && x <= static_cast<dfloat>(NX - 1) &&
           y >= 0.0_df && y <= static_cast<dfloat>(NY - 1) &&
           z >= 0.0_df && z <= static_cast<dfloat>(NZ - 1);
}

/**
 * @brief Construct the configured interpolation/spreading stencil for a particle.
 * @param x Particle center x-coordinate.
 * @param y Particle center y-coordinate.
 * @param z Particle center z-coordinate.
 * @return The base coordinates and separable weights for the selected stencil.
 */
__device__ __forceinline__ PibmStencil makePibmStencil(dfloat x, dfloat y, dfloat z){
    PibmStencil stencilData = {};
    const dfloat position[3] = {x, y, z};

    #if PIBM_COUPLING_MODE == PIBM_COUPLING_TRILINEAR
    for (int axis = 0; axis < 3; ++axis) {
        stencilData.base[axis] = static_cast<int>(floor(position[axis]));
        const dfloat fraction = position[axis] - stencilData.base[axis];
        stencilData.weight[axis][0] = 1.0_df - fraction;
        stencilData.weight[axis][1] = fraction;
    }
    #else
    for (int axis = 0; axis < 3; ++axis) {
        stencilData.base[axis] = static_cast<int>(ceil(position[axis])) - 2;
        for (int offset = 0; offset < PIBM_STENCIL_WIDTH; ++offset) {
            const dfloat distance = static_cast<dfloat>(stencilData.base[axis] + offset) - position[axis];
            stencilData.weight[axis][offset] = (1.0_df + cos(M_PI * distance / 2.0_df)) / 4.0_df;
        }
    }
    #endif

    return stencilData;
}

/**
 * @brief Map one x-axis stencil coordinate according to the boundary condition.
 * @param coordinate Unmapped lattice coordinate.
 * @param mapped Output coordinate after clamping or periodic wrapping.
 * @return False when the coordinate is outside a truncated wall stencil.
 */
__device__ __forceinline__ bool mapPibmX(int coordinate, int* mapped){
    #ifdef BC_X_WALL
        #if PIBM_COUPLING_MODE == PIBM_COUPLING_TRILINEAR
        *mapped = coordinate < 0 ? 0 : (coordinate >= NX ? NX - 1 : coordinate);
        return true;
        #else
        if (coordinate < 0 || coordinate >= NX) return false;
        *mapped = coordinate;
        return true;
        #endif
    #else
    *mapped = ((coordinate % NX) + NX) % NX;
    return true;
    #endif
}

/**
 * @brief Map one y-axis stencil coordinate according to the boundary condition.
 * @param coordinate Unmapped lattice coordinate.
 * @param mapped Output coordinate after clamping or periodic wrapping.
 * @return False when the coordinate is outside a truncated wall stencil.
 */
__device__ __forceinline__ bool mapPibmY(int coordinate, int* mapped){
    #ifdef BC_Y_WALL
        #if PIBM_COUPLING_MODE == PIBM_COUPLING_TRILINEAR
        *mapped = coordinate < 0 ? 0 : (coordinate >= NY ? NY - 1 : coordinate);
        return true;
        #else
        if (coordinate < 0 || coordinate >= NY) return false;
        *mapped = coordinate;
        return true;
        #endif
    #else
    *mapped = ((coordinate % NY) + NY) % NY;
    return true;
    #endif
}

/**
 * @brief Map one z-axis stencil coordinate according to the boundary condition.
 * @param coordinate Unmapped lattice coordinate.
 * @param mapped Output coordinate after clamping or periodic wrapping.
 * @return False when the coordinate is outside a truncated wall stencil.
 */
__device__ __forceinline__ bool mapPibmZ(int coordinate, int* mapped){
    #ifdef BC_Z_WALL
        #if PIBM_COUPLING_MODE == PIBM_COUPLING_TRILINEAR
        *mapped = coordinate < 0 ? 0 : (coordinate >= NZ ? NZ - 1 : coordinate);
        return true;
        #else
        if (coordinate < 0 || coordinate >= NZ) return false;
        *mapped = coordinate;
        return true;
        #endif
    #else
    *mapped = ((coordinate % NZ) + NZ) % NZ;
    return true;
    #endif
}

/**
 * @brief Check optional external-duct admissibility for an Eulerian node.
 * @param x Mapped Eulerian x-coordinate.
 * @param y Mapped Eulerian y-coordinate.
 * @return True when the node belongs to the admissible coupling region.
 */
__device__ __forceinline__ bool insidePibmDuct(int x, int y){
    #ifdef EXTERNAL_DUCT_BC
    const dfloat dx = static_cast<dfloat>(x) - DUCT_CENTER_X;
    const dfloat dy = static_cast<dfloat>(y) - DUCT_CENTER_Y;
    return dx * dx + dy * dy < OUTER_RADIUS * OUTER_RADIUS;
    #else
    return true;
    #endif
}

/**
 * @brief Interpolate density and force-corrected physical velocity at a particle.
 * @param stencil Stencil coordinates and separable weights.
 * @param fMom Device macroscopic moment array.
 * @param rhoInterpolated Output interpolated density.
 * @param velocityInterpolated Output interpolated physical velocity.
 * @return False when a sampled Eulerian state is invalid.
 */
__device__ bool interpolatePibmField(
    const PibmStencil& stencil,
    const dfloat* fMom,
    dfloat* rhoInterpolated,
    dfloat3* velocityInterpolated);

/**
 * @brief Return the common PIBM thread-block size.
 * @return Number of threads used by each PIBM block.
 */
constexpr unsigned int pibmThreads(){
    return NUM_PARTICLES > 64 ? 64 : NUM_PARTICLES;
}

/**
 * @brief Return the common PIBM grid size.
 * @return Number of blocks required for all particle centers.
 */
constexpr unsigned int pibmBlocks(){
    return (NUM_PARTICLES + pibmThreads() - 1) / pibmThreads();
};

} // namespace pibm_detail

/**
 * @brief Compute one relaxed PIBM force correction per particle.
 *
 * The kernel interpolates the force-corrected Eulerian state, evaluates the
 * target point-particle drag, and updates the accumulated force. Only the
 * incremental correction is added to the particle; the matching opposite
 * correction is spread by spreadPibmParticleForce.
 * @param pArray Device particle-center array.
 * @param methods Device coupling-method array.
 * @param fMom Device macroscopic moment array.
 * @param accumulatedForces Per-particle force accumulated in this timestep.
 * @param forceCorrections Per-particle correction for the current iteration.
 * @param nParticles Number of particle centers in the arrays.
 * @return No value; writes the accumulated force and current correction arrays.
 */
__global__
void computePibmParticleForce(
    ParticleCenter *pArray,
    const ParticleMethod *methods,
    const dfloat *fMom,
    dfloat3 *accumulatedForces,
    dfloat3 *forceCorrections,
    unsigned int nParticles
);

/**
 * @brief Spread the current PIBM force corrections to Eulerian force moments.
 *
 * The correction is applied with the same interpolation/spreading operator
 * selected by PIBM_COUPLING_MODE and with the opposite sign to the particle
 * force, preserving discrete action--reaction.
 * @param pArray Device particle-center array.
 * @param methods Device coupling-method array.
 * @param forceCorrections Per-particle correction for the current iteration.
 * @param fMom Device macroscopic moment array updated with atomic additions.
 * @param nParticles Number of particle centers in the arrays.
 * @return No value; updates Eulerian force moments by atomic addition.
 */
__global__
void spreadPibmParticleForce(
    const ParticleCenter *pArray,
    const ParticleMethod *methods,
    const dfloat3 *forceCorrections,
    dfloat *fMom,
    unsigned int nParticles
);

/**
 * @brief Launch the PIBM force-computation kernel for one iteration.
 * @param particles Particle structure containing device arrays and scratch force storage.
 * @param fMom Device macroscopic moment array.
 * @param streamParticles CUDA stream that orders PIBM work with particle dynamics.
 * @return No value; queues one PIBM force-computation kernel.
 */
__host__ void pibmComputeParticleForces(
    ParticlesSoA* particles, const dfloat* fMom, cudaStream_t streamParticles);

/**
 * @brief Launch the PIBM force-spreading kernel for one iteration.
 * @param particles Particle structure containing device arrays and scratch force storage.
 * @param fMom Device macroscopic moment array receiving the reaction force.
 * @param streamParticles CUDA stream used for the launch.
 * @return No value; queues one PIBM force-spreading kernel.
 */
__host__ void pibmSpreadParticleForces(
    ParticlesSoA* particles, dfloat* fMom, cudaStream_t streamParticles);

/**
 * @brief Perform all configured PIBM force-coupling iterations for one timestep.
 *
 * The accumulated force is reset once, then each iteration computes and spreads
 * one relaxed correction on the supplied stream. The stream's final ordering
 * is preserved for the subsequent particle-kinematics update.
 * @param particles Particle structure containing device arrays and scratch force storage.
 * @param fMom Device macroscopic moment array.
 * @param streamParticles CUDA stream used for all PIBM launches.
 * @param step Current simulation timestep (reserved for timestep diagnostics).
 * @return No value; queues all configured PIBM iterations on the supplied stream.
 */
void pibmSimulation(
    ParticlesSoA* particles,
    dfloat *fMom,
    cudaStream_t streamParticles,
    unsigned int step
);

#endif //PARTICLE_MODEL
#endif //__PARTICLE_MODEL_PIBM_CUH
