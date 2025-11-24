

#include "pibm.cuh"

#ifdef PARTICLE_MODEL

void pibmSimulation(
    ParticlesSoA *particles,
    dfloat *fMom,
    cudaStream_t streamParticles,
    unsigned int step)
{
    IbmNodesSoA h_nodes = *(particles->getNodesSoA());
    MethodRange range = particles->getMethodRange(PIBM);
    // const unsigned int n_particles = range.last - range.first + 1;

    for (unsigned int p_idx = range.first; p_idx <= range.last; p_idx++)
    {
        // Drag force following stokes law
        // Check if M_PI does not cause mixed-precision (maybe use a constexpr for this)
        dfloat drag_force = 3 * M_PI * viscosity * &particles[p_idx].pCenter.getDiameter();
    }

    // const unsigned int THREADS_PARTICLES_PIBM = n_particles > 64 ? 64 : n_particles;
    // const unsigned int GRID_PARTICLES_PIBM = (n_particles % THREADS_PARTICLES_PIBM ? (n_particles / THREADS_PARTICLES_PIBM + 1) : (n_particles / THREADS_PARTICLES_PIBM));

    // atomicAdd(&(fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_FX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]), Fx);
}

#endif // PARTICLE_MODEL