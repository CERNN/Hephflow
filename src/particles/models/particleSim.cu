

#include "particleSim.cuh"

#ifdef PARTICLE_MODEL

void particleSimulation(
    ParticlesSoA *particles,
    dfloat *fMom,
    cudaStream_t *streamParticles,
    ParticleWallForces *d_pwForces,
    unsigned int step
){
    // reset force wall    
    cudaMemset(d_pwForces, 0, sizeof(ParticleWallForces));
    // Calculate collision force between particles
    ParticleCenter* pArray = particles->getPCenterArray();
    ParticleShape* shape = particles->getPShape();
    updateParticleOldValues<<<GRID_PARTICLES, THREADS_PARTICLES, 0, streamParticles[0]>>>(pArray,step);
    particlesCollisionHandler<<<GRID_PCOLLISION, TOTAL_PCOLLISION, 0, streamParticles[0]>>>(shape,pArray,d_pwForces,step);
    checkCudaErrors(cudaStreamSynchronize(streamParticles[0]));

    int numIBM    = particles->getMethodCount(IBM);
    int numPIBM   = particles->getMethodCount(PIBM);
    int numTRACER = particles->getMethodCount(TRACER);

    if(numIBM>0){
       ibmSimulation(particles,fMom,streamParticles[0],step);
    }
    if(numPIBM>0){
        pibmSimulation(particles,fMom,streamParticles[0],step);
    }
    if(numTRACER>0){
        tracerSimulation(particles,fMom,streamParticles[0],step);
    }

    updateParticleCenterVelocityAndRotation<<<GRID_PARTICLES, THREADS_PARTICLES, 0, streamParticles[0]>>>(pArray,step);
    updateParticlePosition<<<GRID_PARTICLES, THREADS_PARTICLES, 0, streamParticles[0]>>>(pArray,step);
    checkCudaErrors(cudaStreamSynchronize(streamParticles[0]));
}

#endif //PARTICLE_MODEL