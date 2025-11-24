

#include "pibm.cuh"

#ifdef PARTICLE_MODEL


__host__
void pibmSimulation(
    ParticlesSoA *particles,
    dfloat *fMom,
    cudaStream_t streamParticles,
    unsigned int step)
{
    IbmNodesSoA h_nodes = *(particles->getNodesSoA());
    MethodRange range = particles->getMethodRange(PIBM);
    // const unsigned int n_particles = range.last - range.first + 1;

    ParticleCenter* pArray = particles->getPCenterArray();

    updateParticleOldValues<<<GRID_PARTICLES_IBM, THREADS_PARTICLES_IBM, 0, streamParticles>>>(pArray,range.first,range.last,step);

    ParticleShape* shape = particles->getPShape();
    particlesCollisionHandler<<<GRID_PCOLLISION_IBM, THREADS_PCOLLISION_IBM, 0, streamParticles>>>(shape,pArray,step);

    for (unsigned int p_idx = range.first; p_idx <= range.last; p_idx++)
    {
        ParticleCenter* pc_i = &pArray[p_idx];
        // Drag force following stokes law
        // Check if M_PI does not cause mixed-precision (maybe use a constexpr for this)

        dfloat px = pc_i->getPosX();
        dfloat py = pc_i->getPosY();
        dfloat pz = pc_i->getPosZ();

        dfloat ux_interpolated = mom_trilinear_interp(px,py,pz,M_UX_INDEX,fMom);
        dfloat uy_interpolated = mom_trilinear_interp(px,py,pz,M_UY_INDEX,fMom);
        dfloat uz_interpolated = mom_trilinear_interp(px,py,pz,M_UZ_INDEX,fMom);
        dfloat3 fluid_velocity = {ux_interpolated,uy_interpolated,uz_interpolated};


        dfloat3 drag_force = (3.0 * M_PI * VISC * pc_i->getDiameter()) * (fluid_velocity -  pc_i->getVel());

        // Check if particle collision happened before. If so, add to the current particle force
        // If not, then zero the force before summing drag force
        pc_i->setF(pc_i->getF() + drag_force);


            for (int zk = minIdx[2]; zk <= maxIdx[2]; zk++) // z
    {
        for (int yj = minIdx[1]; yj <= maxIdx[1]; yj++) // y
        {
            aux1 = stencilVal[2][zk]*stencilVal[1][yj];
            for (int xi = minIdx[0]; xi <= maxIdx[0]; xi++) // x
            {
                // Dirac delta (kernel)
                aux = aux1 * stencilVal[0][xi];
                // same as aux = stencil(x - xIBM) * stencil(y - yIBM) * stencil(z - zIBM);
 
                xx = (posBase[0] + xi + NX)%(NX);
                yy = (posBase[1] + yj + NY)%(NY);
                zz = (posBase[2] + zk + NZ)%(NZ);
                
                atomicAdd(&(fMom[idxMom(xx%BLOCK_NX, yy%BLOCK_NY, zz%BLOCK_NZ, M_FX_INDEX, xx/BLOCK_NX, yy/BLOCK_NY, zz/BLOCK_NZ)]), -drag_foce.x * aux);
                atomicAdd(&(fMom[idxMom(xx%BLOCK_NX, yy%BLOCK_NY, zz%BLOCK_NZ, M_FY_INDEX, xx/BLOCK_NX, yy/BLOCK_NY, zz/BLOCK_NZ)]), -drag_foce.y * aux);
                atomicAdd(&(fMom[idxMom(xx%BLOCK_NX, yy%BLOCK_NY, zz%BLOCK_NZ, M_FZ_INDEX, xx/BLOCK_NX, yy/BLOCK_NY, zz/BLOCK_NZ)]), -drag_foce.z * aux);

            }
        }
    }
    }

    // const unsigned int THREADS_PARTICLES_PIBM = n_particles > 64 ? 64 : n_particles;
    // const unsigned int GRID_PARTICLES_PIBM = (n_particles % THREADS_PARTICLES_PIBM ? (n_particles / THREADS_PARTICLES_PIBM + 1) : (n_particles / THREADS_PARTICLES_PIBM));

    // atomicAdd(&(fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_FX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]), Fx);

    updateParticleCenterVelocityAndRotation<<<GRID_PARTICLES_IBM, THREADS_PARTICLES_IBM, 0, streamParticles>>>(pArray,range.first,range.last,step);
    updateParticlePosition<<<GRID_PARTICLES_IBM, THREADS_PARTICLES_IBM, 0, streamParticles>>>(pArray,range.first,range.last,step);

    checkCudaErrors(cudaStreamSynchronize(streamParticles));

}

#endif // PARTICLE_MODEL