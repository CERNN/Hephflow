
#include "pibm.cuh"

#ifdef PARTICLE_MODEL

__global__ 
void spreadParticleForce(
    ParticleCenter *pArray,
    const ParticleMethod *methods,
    dfloat *fMom,
    unsigned int nParticles)
{
    int p_idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (p_idx >= nParticles || methods[p_idx] != PIBM)
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
    dfloat3 drag_force = 2 * particle_area * (RHO_0 + mom_trilinear_interp(px, py, pz, M_RHO_INDEX, fMom)) * (fluid_velocity - pc_i->getVel());

    accumulateForceAndTorque(pc_i, drag_force, {0, 0, 0});

    // Stencil bounds (integer lattice range around particle position)
    int stencil_start_x = (int)ceil(px) - FORCE_SPREAD_X_NODES;
    int stencil_start_y = (int)ceil(py) - FORCE_SPREAD_Y_NODES;
    int stencil_start_z = (int)ceil(pz) - FORCE_SPREAD_Z_NODES;

    int stencil_end_x = stencil_start_x + 2 * FORCE_SPREAD_X_NODES - 1;
    int stencil_end_y = stencil_start_y + 2 * FORCE_SPREAD_Y_NODES - 1;
    int stencil_end_z = stencil_start_z + 2 * FORCE_SPREAD_Z_NODES - 1;

    // Use correct stencil with boundary handling
    for (int zk = stencil_start_z; zk <= stencil_end_z; zk++) // z
    {
        int zz;
        #ifdef BC_Z_WALL
            if (zk < 0 || zk >= NZ) continue;
            zz = zk;
        #endif
        #ifdef BC_Z_PERIODIC
            zz = ((zk % NZ) + NZ) % NZ;
        #endif

        for (int yj = stencil_start_y; yj <= stencil_end_y; yj++) // y
        {
            int yy;
            #ifdef BC_Y_WALL
                if (yj < 0 || yj >= NY) continue;
                yy = yj;
            #endif
            #ifdef BC_Y_PERIODIC
                yy = ((yj % NY) + NY) % NY;
            #endif

            for (int xi = stencil_start_x; xi <= stencil_end_x; xi++) // x
            {
                int xx;
                #ifdef BC_X_WALL
                    if (xi < 0 || xi >= NX) continue;
                    xx = xi;
                #endif
                #ifdef BC_X_PERIODIC
                    xx = ((xi % NX) + NX) % NX;
                #endif

                dfloat spread_filter_x = (1.0_df + cos(M_PI * (dfloat(xi) - px) / 2.0_df)) / 4.0_df;
                dfloat spread_filter_y = (1.0_df + cos(M_PI * (dfloat(yj) - py) / 2.0_df)) / 4.0_df;
                dfloat spread_filter_z = (1.0_df + cos(M_PI * (dfloat(zk) - pz) / 2.0_df)) / 4.0_df;
                dfloat spread_filter = spread_filter_x * spread_filter_y * spread_filter_z;

                atomicAdd(&(fMom[idxMom(xx % BLOCK_NX, yy % BLOCK_NY, zz % BLOCK_NZ, M_FX_INDEX, xx / BLOCK_NX, yy / BLOCK_NY, zz / BLOCK_NZ)]), -drag_force.x * spread_filter);
                atomicAdd(&(fMom[idxMom(xx % BLOCK_NX, yy % BLOCK_NY, zz % BLOCK_NZ, M_FY_INDEX, xx / BLOCK_NX, yy / BLOCK_NY, zz / BLOCK_NZ)]), -drag_force.y * spread_filter);
                atomicAdd(&(fMom[idxMom(xx % BLOCK_NX, yy % BLOCK_NY, zz % BLOCK_NZ, M_FZ_INDEX, xx / BLOCK_NX, yy / BLOCK_NY, zz / BLOCK_NZ)]), -drag_force.z * spread_filter);
            }
        }
    }
}

__host__ void pibmSimulation(
    ParticlesSoA *particles,
    dfloat *fMom,
    cudaStream_t streamParticles,
    unsigned int step)
{
    constexpr unsigned int N_PARTICLES = NUM_PARTICLES;

    const unsigned int THREADS_PARTICLES_PIBM = N_PARTICLES > 64 ? 64 : N_PARTICLES;
    const unsigned int GRID_PARTICLES_PIBM = (N_PARTICLES % THREADS_PARTICLES_PIBM ? (N_PARTICLES / THREADS_PARTICLES_PIBM + 1) : (N_PARTICLES / THREADS_PARTICLES_PIBM));

    ParticleCenter *pArray = particles->getPCenterArray();
    const ParticleMethod *methods = particles->getPMethod();

    spreadParticleForce<<<GRID_PARTICLES_PIBM, THREADS_PARTICLES_PIBM, 0, streamParticles>>>(
        pArray, methods, fMom, N_PARTICLES);
}

#endif // PARTICLE_MODEL
