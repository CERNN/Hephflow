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

enum class KernelType {
    TRILINEAR,
    STENCIL
};

constexpr KernelType PIBM_INTERPOLATION_KERNEL_TYPE =
    (PIBM_INTERPOLATION_KERNEL <= PIBM_TRILINEAR_4) ? KernelType::TRILINEAR : KernelType::STENCIL;
constexpr KernelType PIBM_SPREAD_KERNEL_TYPE =
    (PIBM_SPREAD_KERNEL <= PIBM_TRILINEAR_4) ? KernelType::TRILINEAR : KernelType::STENCIL;

#ifdef BC_X_WALL
constexpr bool PIBM_WALL_X = true;
#else
constexpr bool PIBM_WALL_X = false;
#endif

#ifdef BC_Y_WALL
constexpr bool PIBM_WALL_Y = true;
#else
constexpr bool PIBM_WALL_Y = false;
#endif

#ifdef BC_Z_WALL
constexpr bool PIBM_WALL_Z = true;
#else
constexpr bool PIBM_WALL_Z = false;
#endif

/**
 * @brief Separable tensor-product stencil: lower lattice corner and 1D weights per axis.
 * @tparam Width Number of lattice nodes per axis.
 */
template <int Width>
struct PibmStencil {
    int base[3];
    dfloat weight[3][Width];
};

using PibmInterpolationStencil = PibmStencil<PIBM_INTERPOLATION_KERNEL_WIDTH>;
using PibmSpreadStencil        = PibmStencil<PIBM_SPREAD_KERNEL_WIDTH>;

/**
 * @brief Validate that a particle center can be converted to a lattice stencil.
 * @param x Particle center x-coordinate.
 * @param y Particle center y-coordinate.
 * @param z Particle center z-coordinate.
 * @return True when all coordinates are finite and inside the local domain.
 */
__device__ __forceinline__ bool validPibmParticlePosition(dfloat x, dfloat y, dfloat z){
    return isfinite(x) && isfinite(y) && isfinite(z) &&
           x >= 0.0_df && x <= static_cast<dfloat>(NX - 1) &&
           y >= 0.0_df && y <= static_cast<dfloat>(NY - 1) &&
           z >= 0.0_df && z <= static_cast<dfloat>(NZ - 1);
}

/**
 * @brief Lowest lattice index covered by a 1D stencil of the given width.
 * @tparam Width Number of lattice nodes per axis.
 * @param coord Particle coordinate along the axis.
 * @return Lower-bound lattice index of the 1D stencil.
 */
template <int Width>
__device__ __forceinline__ int pibmStencilBase(dfloat coord){
    if constexpr (Width % 2 == 0) {
        return static_cast<int>(ceil(coord)) - Width / 2;
    } else {
        return static_cast<int>(round(coord)) - Width / 2;
    }
}

/**
 * @brief 1D kernel weight at a signed lattice distance from the particle.
 * @tparam Width Number of lattice nodes per axis.
 * @tparam KType Kernel shape (linear hat or raised cosine).
 * @param distance Signed distance between lattice node and particle.
 * @return Kernel weight for that node.
 */
template <int Width, KernelType KType>
__device__ __forceinline__ dfloat pibmKernelWeight(dfloat distance){
    const dfloat absR = fabs(distance);
    if constexpr (KType == KernelType::TRILINEAR) {
        constexpr dfloat halfWidth = 0.5_df * static_cast<dfloat>(Width);
        const int target_index = halfWidth + distance;

        double weight = 1.0;
        for (int sweep_index = 0; sweep_index < Width; ++sweep_index) {
            if (sweep_index == target_index) {
                continue;
            }
            weight *= (1 + distance / (target_index - sweep_index));
        }
        return weight;
    } else {
        if constexpr (Width == 2) {
            // Peskin 2-point stencil (only for PIBM_STENCIL_2)
            return absR <= 1.0_df ? (1.0_df - absR) : 0.0_df;
        } else if constexpr (Width == 3) {
            // Peskin 3-point stencil (only for PIBM_STENCIL_3)
            if (absR <= 0.5_df) {
                return (1.0_df + sqrt(1.0_df - 3.0_df * absR * absR)) / 3.0_df;
            } else if (absR <= 1.5_df) {
                return (5.0_df - 3.0_df * absR - sqrt(-2.0_df + 6.0_df * absR - 3.0_df * absR * absR)) / 6.0_df;
            } else {
                return 0.0_df;
            }
        } else if constexpr (Width == 4) {
            // Cosine stencil (only for PIBM_STENCIL_4)
            if (absR <= 2.0_df) {
                return (1.0_df + cos(M_PI * distance * 0.5_df)) * 0.25_df;
            } else {
                return 0.0_df;
            }
        }
    }
}

/**
 * @brief Construct a separable interpolation/spreading stencil for a particle.
 * @tparam Width Number of lattice nodes per axis.
 * @tparam KType Kernel shape.
 * @param position Particle center.
 * @return Lower lattice corner and per-axis weights.
 */
template <int Width, KernelType KType>
__device__ __forceinline__ PibmStencil<Width> makePibmStencil(const dfloat3& position){
    PibmStencil<Width> stencil;
    const dfloat coord[3] = {position.x, position.y, position.z};

    #pragma unroll
    for (int axis = 0; axis < 3; ++axis) {
        stencil.base[axis] = pibmStencilBase<Width>(coord[axis]);
        #pragma unroll
        for (int offset = 0; offset < Width; ++offset) {
            const dfloat distance = static_cast<dfloat>(stencil.base[axis] + offset) - coord[axis];
            stencil.weight[axis][offset] = pibmKernelWeight<Width, KType>(distance);
        }
    }
    return stencil;
}

/**
 * @brief Map one stencil coordinate onto the lattice along an axis.
 *
 * Periodic axes wrap; wall axes clamp for the trilinear kernel and truncate
 * (reject the node) for the cosine stencil.
 * @tparam KType Kernel shape.
 * @tparam IsWall True when the axis has wall boundaries.
 * @tparam N Number of lattice nodes along the axis.
 * @param coordinate Unmapped lattice coordinate.
 * @param mapped Output mapped coordinate.
 * @return False when the node falls outside a truncated wall stencil.
 */
template <KernelType KType, bool IsWall, int N>
__device__ __forceinline__ bool mapPibmAxis(int coordinate, int* mapped){
    if constexpr (!IsWall) {
        *mapped = ((coordinate % N) + N) % N;
        return true;
    } else if constexpr (KType == KernelType::TRILINEAR) {
        *mapped = coordinate < 0 ? 0 : (coordinate >= N ? N - 1 : coordinate);
        return true;
    } else {
        if (coordinate < 0 || coordinate >= N) return false;
        *mapped = coordinate;
        return true;
    }
}

template <KernelType KType>
__device__ __forceinline__ bool mapPibmX(int coordinate, int* mapped){
    return mapPibmAxis<KType, PIBM_WALL_X, NX>(coordinate, mapped);
}

template <KernelType KType>
__device__ __forceinline__ bool mapPibmY(int coordinate, int* mapped){
    return mapPibmAxis<KType, PIBM_WALL_Y, NY>(coordinate, mapped);
}

template <KernelType KType>
__device__ __forceinline__ bool mapPibmZ(int coordinate, int* mapped){
    return mapPibmAxis<KType, PIBM_WALL_Z, NZ>(coordinate, mapped);
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
 * @param stencil Interpolation stencil coordinates and separable weights.
 * @param fMom Device macroscopic moment array.
 * @param rhoInterpolated Output interpolated density.
 * @param velocityInterpolated Output interpolated physical velocity.
 * @return False when a sampled Eulerian state is invalid.
 */
__device__ __forceinline__ bool interpolatePibmField(
    const PibmInterpolationStencil& stencil,
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
 * The correction is applied with the stencil selected by PIBM_SPREAD_KERNEL and
 * with the opposite sign to the particle force, preserving discrete
 * action--reaction.
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
