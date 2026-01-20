

#include "pibm.cuh"

#ifdef PARTICLE_MODEL

__global__ void spreadParticleForce(ParticleCenter *pArray, dfloat *fMom, unsigned int nParticles)
{
    int p_idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (p_idx >= nParticles)
        return;

    ParticleCenter *pc_i = &pArray[p_idx];

    dfloat px = pc_i->getPosX();
    dfloat py = pc_i->getPosY();
    dfloat pz = pc_i->getPosZ();

    // Use a Lagrangian interpolator
    dfloat ux_interpolated = mom_trilinear_interp(px, py, pz, M_UX_INDEX, fMom);
    dfloat uy_interpolated = mom_trilinear_interp(px, py, pz, M_UY_INDEX, fMom);
    dfloat uz_interpolated = mom_trilinear_interp(px, py, pz, M_UZ_INDEX, fMom);
    dfloat3 fluid_velocity = {ux_interpolated, uy_interpolated, uz_interpolated};

    dfloat particle_area = M_PI * pc_i->getDiameter() * pc_i->getDiameter() / 4;
    dfloat3 drag_force = particle_area * mom_trilinear_interp(px, py, pz, M_RHO_INDEX, fMom) * (fluid_velocity - pc_i->getVel());

    pc_i->setF(pc_i->getF() + drag_force);

    dim3 stencil_bound_start, stencil_bound_end;

#ifdef defined(FORCE_SPREAD_X_NODES) && defined(FORCE_SPREAD_Y_NODES) && defined(FORCE_SPREAD_Z_NODES)
    stencil_bound_start.x = (int)ceil(px) - FORCE_SPREAD_X_NODES;
    stencil_bound_start.y = (int)ceil(py) - FORCE_SPREAD_Y_NODES;
    stencil_bound_start.z = (int)ceil(pz) - FORCE_SPREAD_Z_NODES;

    stencil_bound_end.x = (int)floor(px) + FORCE_SPREAD_X_NODES;
    stencil_bound_end.y = (int)floor(py) + FORCE_SPREAD_Y_NODES;
    stencil_bound_end.z = (int)floor(pz) + FORCE_SPREAD_Z_NODES;
#endif

    // For periodic boundary
    // TODO: Update this to react to BC definitions
    if (stencil_bound_start.x < 0)
        stencil_bound_start.x += NX;
    if (stencil_bound_start.y < 0)
        stencil_bound_start.y += NY;
    if (stencil_bound_start.z < 0)
        stencil_bound_start.z += NZ;

    if (stencil_bound_start.x >= NX)
        stencil_bound_start.x = stencil_bound_start.x % NX;
    if (stencil_bound_start.y >= NY)
        stencil_bound_start.y = stencil_bound_start.y % NY;
    if (stencil_bound_start.z >= NZ)
        stencil_bound_start.z = stencil_bound_start.z % NZ;

    // Use correct stencil
    for (int zk = stencil_bound_start.z; zk <= stencil_bound_end.z; zk++) // z
    {
        for (int yj = stencil_bound_start.y; yj <= stencil_bound_end.y; yj++) // y
        {
            for (int xi = stencil_bound_start.x; xi <= stencil_bound_end.x; xi++) // x
            {
                dfloat spread_filter = (1 + cos(M_PI * (xi - px) / 2)) * (1 + cos(M_PI * (yj - py) / 2)) * (1 + cos(M_PI * (zk - pz) / 2)) / 64;

                unsigned int xx = (stencil_bound_start.x + xi + NX) % (NX);
                unsigned int yy = (stencil_bound_start.y + yj + NY) % (NY);
                unsigned int zz = (stencil_bound_start.z + zk + NZ) % (NZ);

                atomicAdd(&(fMom[idxMom(xx % BLOCK_NX, yy % BLOCK_NY, zz % BLOCK_NZ, M_FX_INDEX, xx / BLOCK_NX, yy / BLOCK_NY, zz / BLOCK_NZ)]), -drag_force.x * spread_filter);
                atomicAdd(&(fMom[idxMom(xx % BLOCK_NX, yy % BLOCK_NY, zz % BLOCK_NZ, M_FY_INDEX, xx / BLOCK_NX, yy / BLOCK_NY, zz / BLOCK_NZ)]), -drag_force.y * spread_filter);
                atomicAdd(&(fMom[idxMom(xx % BLOCK_NX, yy % BLOCK_NY, zz % BLOCK_NZ, M_FZ_INDEX, xx / BLOCK_NX, yy / BLOCK_NY, zz / BLOCK_NZ)]), -drag_force.z * spread_filter);
            }
        }
    }

    // printf("Drag Force         x: %e y: %e z: %e\n", drag_force.x, drag_force.y, drag_force.z);
    // printf("Particle Position  x: %e y: %e z: %e\n", pc_i->getPosX(), pc_i->getPosY(), pc_i->getPosZ());
    // printf("Particle Velocity  x: %e y: %e z: %e\n", pc_i->getVelX(), pc_i->getVelY(), pc_i->getVelZ());
    // printf("Particle Ang Vel   x: %e y: %e z: %e\n", pc_i->getWX(), pc_i->getWY(), pc_i->getWZ());
    // printf("Particle Force     x: %e y: %e z: %e\n", pc_i->getFX(), pc_i->getFY(), pc_i->getFZ());
    // printf("Particle Moment    x: %e y: %e z: %e\n", pc_i->getMX(), pc_i->getMY(), pc_i->getMZ());
}

__global__ void resetParticleForce(ParticleCenter *pArray, unsigned int nParticles)
{
    int p_idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (p_idx >= nParticles)
        return;

    ParticleCenter *pc_i = &pArray[p_idx];

    pc_i->setF(dfloat3(0.0, 0.0, 0.0));
}

__host__ void pibmSimulation(
    ParticlesSoA *particles,
    dfloat *fMom,
    cudaStream_t streamParticles,
    unsigned int step)
{
    // IbmNodesSoA h_nodes = *(particles->getNodesSoA());
    // IbmNodesSoA *d_nodes = &h_nodes;
    // cudaMalloc(&d_nodes, sizeof(IbmNodesSoA));
    // cudaMemcpy(d_nodes, &h_nodes, sizeof(IbmNodesSoA), cudaMemcpyHostToDevice);

    // checkCudaErrors(cudaSetDevice(GPU_INDEX));

    MethodRange range = particles->getMethodRange(PIBM);
    const unsigned int N_PARTICLES = range.last - range.first + 1;

    const unsigned int THREADS_PARTICLES_PIBM = N_PARTICLES > 64 ? 64 : N_PARTICLES;
    const unsigned int GRID_PARTICLES_PIBM = (N_PARTICLES % THREADS_PARTICLES_PIBM ? (N_PARTICLES / THREADS_PARTICLES_PIBM + 1) : (N_PARTICLES / THREADS_PARTICLES_PIBM));

    const unsigned int TOTAL_PCOLLISION_PIBM_THREADS = (N_PARTICLES * (N_PARTICLES + 1)) / 2;
    const unsigned int THREADS_PCOLLISION_PIBM = (TOTAL_PCOLLISION_PIBM_THREADS > 64) ? 64 : TOTAL_PCOLLISION_PIBM_THREADS;
    const unsigned int GRID_PCOLLISION_PIBM =
        (TOTAL_PCOLLISION_PIBM_THREADS % THREADS_PCOLLISION_PIBM ? (TOTAL_PCOLLISION_PIBM_THREADS / THREADS_PCOLLISION_PIBM + 1)
                                                                 : (TOTAL_PCOLLISION_PIBM_THREADS / THREADS_PCOLLISION_PIBM));

    ParticleCenter *pArray = particles->getPCenterArray();
    ParticleShape *shape = particles->getPShape();

    resetParticleForce<<<GRID_PARTICLES_PIBM, THREADS_PARTICLES_PIBM, 0, streamParticles>>>(pArray, N_PARTICLES);
    spreadParticleForce<<<GRID_PARTICLES_PIBM, THREADS_PARTICLES_PIBM, 0, streamParticles>>>(pArray, fMom, N_PARTICLES);

    checkCudaErrors(cudaStreamSynchronize(streamParticles));
}

#endif // PARTICLE_MODEL