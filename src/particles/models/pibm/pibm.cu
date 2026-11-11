#include "pibm.cuh"

#ifdef PARTICLE_MODEL

namespace pibm_detail {

// Interpolate density and force-corrected physical velocity at a particle.
__device__ __forceinline__ bool interpolatePibmField(
    const PibmInterpolationStencil& stencilData,
    const dfloat *fMom,
    dfloat *rhoInterpolated,
    dfloat3 *velocityInterpolated){
    // Accumulate stencil-weighted fluid properties at the particle center.
    dfloat rho = 0.0_df;
    dfloat3 velocity(0.0_df, 0.0_df, 0.0_df);
    dfloat weightSum = 0.0_df;

    constexpr int W = PIBM_INTERPOLATION_KERNEL_WIDTH;
    constexpr KernelType K = PIBM_INTERPOLATION_KERNEL_TYPE;

    // Traverse the configured tensor-product stencil and map valid lattice nodes.
    #pragma unroll
    for (int zOffset = 0; zOffset < W; ++zOffset) {
        int z;
        if (!mapPibmZ<K>(stencilData.base[2] + zOffset, &z))
            continue;
        #pragma unroll
        for (int yOffset = 0; yOffset < W; ++yOffset) {
            int y;
            if (!mapPibmY<K>(stencilData.base[1] + yOffset, &y))
                continue;
            const dfloat yzWeight = stencilData.weight[2][zOffset] * stencilData.weight[1][yOffset];

            #pragma unroll
            for (int xOffset = 0; xOffset < W; ++xOffset) {
                int x;
                if (!mapPibmX<K>(stencilData.base[0] + xOffset, &x))
                    continue;
                if (!insidePibmDuct(x, y))
                    continue;

                const dfloat weight = yzWeight * stencilData.weight[0][xOffset];
                const size_t momentBase = idxMom(x % BLOCK_NX, y % BLOCK_NY, z % BLOCK_NZ, M_RHO_INDEX, x / BLOCK_NX, y / BLOCK_NY, z / BLOCK_NZ);

                // Recover density and physical velocity, including the current force correction.
                const dfloat rhoNode = RHO_0 + fMom[momentBase];
                if (!isfinite(rhoNode) || rhoNode <= 1.0e-6_df)
                    return false;

                const dfloat ux = fMom[momentBase + M_UX_INDEX * BLOCK_LBM_SIZE] / F_M_I_SCALE + (fMom[momentBase + M_FX_INDEX * BLOCK_LBM_SIZE] - FX) / rhoNode;
                const dfloat uy = fMom[momentBase + M_UY_INDEX * BLOCK_LBM_SIZE] / F_M_I_SCALE + (fMom[momentBase + M_FY_INDEX * BLOCK_LBM_SIZE] - FY) / rhoNode;
                const dfloat uz = fMom[momentBase + M_UZ_INDEX * BLOCK_LBM_SIZE] / F_M_I_SCALE + (fMom[momentBase + M_FZ_INDEX * BLOCK_LBM_SIZE] - FZ) / rhoNode;
                if (!isfinite(ux) || !isfinite(uy) || !isfinite(uz))
                    return false;

                rho += weight * rhoNode;
                velocity.x += weight * ux;
                velocity.y += weight * uy;
                velocity.z += weight * uz;
                weightSum += weight;
            }
        }
    }

    // Reject an invalid interpolation before exposing its result to the force kernel.
    if (!isfinite(rho) || rho <= 1.0e-6_df || !isfinite(weightSum) || weightSum <= 0.0_df)
        return false;

    // Publish the interpolated field only after every contributing node is valid.
    *rhoInterpolated = rho;
    *velocityInterpolated = velocity;
    return true;
}


} // namespace pibm_detail

__global__
void computePibmParticleForce(
    ParticleCenter *pArray,
    const ParticleMethod *methods,
    const dfloat *fMom,
    dfloat3 *accumulatedForces,
    dfloat3 *forceCorrections,
    unsigned int nParticles)
{
    // Assign one thread to each particle and ignore particles handled by other methods.
    const unsigned int particle = threadIdx.x + blockDim.x * blockIdx.x;
    if (particle >= nParticles || methods[particle] != PIBM) return;

    // Validate the center before constructing integer stencil coordinates.
    ParticleCenter *center = &pArray[particle];
    const dfloat3 position = center->getPos();
    if (!pibm_detail::validPibmParticlePosition(position.x, position.y, position.z)) {
        forceCorrections[particle] = dfloat3(0.0_df, 0.0_df, 0.0_df);
        return;
    }

    // Interpolate the force-corrected fluid state at the particle center using the interpolation stencil.
    const pibm_detail::PibmInterpolationStencil stencilData =
        pibm_detail::makePibmStencil<PIBM_INTERPOLATION_KERNEL_WIDTH, pibm_detail::PIBM_INTERPOLATION_KERNEL_TYPE>(position);
    dfloat rho;
    dfloat3 fluidVelocity;
    const dfloat3 particleVelocity = center->getVel();
    if (!pibm_detail::interpolatePibmField(stencilData, fMom, &rho, &fluidVelocity) ||
        !isfinite(particleVelocity.x) || !isfinite(particleVelocity.y) || !isfinite(particleVelocity.z))
    {
        forceCorrections[particle] = dfloat3(0.0_df, 0.0_df, 0.0_df);
        return;
    }

    // Evaluate the target drag and relax only the remaining force residual.
    const dfloat diameter = center->getDiameter();
    const dfloat area = M_PI * diameter * diameter / 4.0_df;
    const dfloat3 targetForce = 2.0_df * area * rho * (fluidVelocity - particleVelocity);
    const dfloat3 previousForce = accumulatedForces[particle];
    const dfloat3 correction = static_cast<dfloat>(PIBM_FORCE_RELAXATION) * (targetForce - previousForce);
    const dfloat3 updatedForce = previousForce + correction;

    // Suppress invalid corrections before they reach either particle or fluid state.
    if (!isfinite(targetForce.x) || !isfinite(targetForce.y) ||
        !isfinite(targetForce.z) || !isfinite(correction.x) ||
        !isfinite(correction.y) || !isfinite(correction.z) ||
        !isfinite(updatedForce.x) || !isfinite(updatedForce.y) ||
        !isfinite(updatedForce.z)) {
        forceCorrections[particle] = dfloat3(0.0_df, 0.0_df, 0.0_df);
        return;
    }

    // Retain the iteration state and apply this correction to the particle.
    accumulatedForces[particle] = updatedForce;
    forceCorrections[particle] = correction;
    accumulateForceAndTorque(center, correction, dfloat3(0.0_df, 0.0_df, 0.0_df));
}

__global__
void spreadPibmParticleForce(
    const ParticleCenter *pArray,
    const ParticleMethod *methods,
    const dfloat3 *forceCorrections,
    dfloat *fMom,
    unsigned int nParticles)
{
    // Assign one thread to each PIBM particle.
    const unsigned int particle = threadIdx.x + blockDim.x * blockIdx.x;
    if (particle >= nParticles || methods[particle] != PIBM) return;

    // Skip particles whose position or current iteration correction is unusable.
    const dfloat3 position = pArray[particle].getPos();
    const dfloat3 correction = forceCorrections[particle];
    if (!pibm_detail::validPibmParticlePosition(position.x, position.y, position.z) ||
        !isfinite(correction.x) || !isfinite(correction.y) ||
        !isfinite(correction.z) ||
        (correction.x == 0.0_df && correction.y == 0.0_df && correction.z == 0.0_df))
        return;

    constexpr int W = PIBM_SPREAD_KERNEL_WIDTH;
    constexpr pibm_detail::KernelType K = pibm_detail::PIBM_SPREAD_KERNEL_TYPE;

    // Build the spreading stencil configured for the particle.
    const pibm_detail::PibmSpreadStencil stencilData = pibm_detail::makePibmStencil<W, K>(position);

    // Distribute the opposite correction over every admissible Eulerian node.
    #pragma unroll
    for (int zOffset = 0; zOffset < W; ++zOffset) {
        int z;
        if (!pibm_detail::mapPibmZ<K>(stencilData.base[2] + zOffset, &z))
            continue;
        #pragma unroll
        for (int yOffset = 0; yOffset < W; ++yOffset) {
            int y;
            if (!pibm_detail::mapPibmY<K>(stencilData.base[1] + yOffset, &y))
                continue;
            const dfloat yzWeight = stencilData.weight[2][zOffset] * stencilData.weight[1][yOffset];

            #pragma unroll
            for (int xOffset = 0; xOffset < W; ++xOffset) {
                int x;
                if (!pibm_detail::mapPibmX<K>(stencilData.base[0] + xOffset, &x))
                    continue;
                if (!pibm_detail::insidePibmDuct(x, y))
                    continue;

                const dfloat weight = yzWeight * stencilData.weight[0][xOffset];
                const size_t momentBase = idxMom(x % BLOCK_NX, y % BLOCK_NY, z % BLOCK_NZ, M_RHO_INDEX, x / BLOCK_NX, y / BLOCK_NY, z / BLOCK_NZ);

                // Atomics preserve contributions from particles with overlapping stencils.
                atomicAdd(&fMom[momentBase + M_FX_INDEX * BLOCK_LBM_SIZE], -correction.x * weight);
                atomicAdd(&fMom[momentBase + M_FY_INDEX * BLOCK_LBM_SIZE], -correction.y * weight);
                atomicAdd(&fMom[momentBase + M_FZ_INDEX * BLOCK_LBM_SIZE], -correction.z * weight);
            }
        }
    }
}

__host__ void pibmComputeParticleForces(
    ParticlesSoA *particles,
    const dfloat *fMom,
    cudaStream_t streamParticles)
{
    computePibmParticleForce<<<pibm_detail::pibmBlocks(), pibm_detail::pibmThreads(), 0, streamParticles>>>(
        particles->getPCenterArray(), particles->getPMethod(), fMom,
        particles->getPibmForceArray(), particles->getPibmForceCorrectionArray(),
        NUM_PARTICLES);
}

__host__ void pibmSpreadParticleForces(
    ParticlesSoA *particles,
    dfloat *fMom,
    cudaStream_t streamParticles)
{
    spreadPibmParticleForce<<<pibm_detail::pibmBlocks(), pibm_detail::pibmThreads(), 0, streamParticles>>>(
        particles->getPCenterArray(), particles->getPMethod(),
        particles->getPibmForceCorrectionArray(), fMom, NUM_PARTICLES);
}

__host__ void pibmSimulation(
    ParticlesSoA *particles,
    dfloat *fMom,
    cudaStream_t streamParticles,
    unsigned int step)
{
    (void)step;
    checkCudaErrors(cudaMemsetAsync(
        particles->getPibmForceArray(), 0,
        sizeof(dfloat3) * NUM_PARTICLES, streamParticles));

    for (int iteration = 0; iteration < PIBM_MAX_ITERATION; ++iteration) {
        pibmComputeParticleForces(particles, fMom, streamParticles);
        pibmSpreadParticleForces(particles, fMom, streamParticles);
    }
}

#endif // PARTICLE_MODEL
