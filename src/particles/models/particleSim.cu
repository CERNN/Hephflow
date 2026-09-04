

#include "particleSim.cuh"

#ifdef PARTICLE_FORCE_DEBUG
#include <fstream>
#include <iomanip>
#include <vector>
#endif

#ifdef PARTICLE_MODEL

#ifdef PARTICLE_FORCE_DEBUG
namespace {
std::vector<ParticleCenter> collisionSnapshot;

bool forceDebugStep(unsigned int step) {
    return step - PARTICLE_FORCE_DEBUG_START_STEP <=
           PARTICLE_FORCE_DEBUG_END_STEP - PARTICLE_FORCE_DEBUG_START_STEP;
}

void captureCollisionSnapshot(ParticleCenter* deviceParticles, unsigned int step) {
    if (!forceDebugStep(step)) return;
    collisionSnapshot.resize(NUM_PARTICLES);
    checkCudaErrors(cudaMemcpy(
        collisionSnapshot.data(), deviceParticles,
        NUM_PARTICLES * sizeof(ParticleCenter), cudaMemcpyDeviceToHost));
}

void writeVec3(std::ofstream& out, const dfloat3& value) {
    out << ',' << value.x << ',' << value.y << ',' << value.z;
}

void exportParticleForceDebug(ParticleCenter* deviceParticles, unsigned int step) {
    if (!forceDebugStep(step) || collisionSnapshot.size() != NUM_PARTICLES) return;

    std::vector<ParticleCenter> finalSnapshot(NUM_PARTICLES);
    checkCudaErrors(cudaMemcpy(
        finalSnapshot.data(), deviceParticles,
        NUM_PARTICLES * sizeof(ParticleCenter), cudaMemcpyDeviceToHost));

    static bool initialized = false;
    const std::ios::openmode mode = initialized ? std::ios::app : std::ios::trunc;
    std::ofstream forces("particle_force_debug.csv", mode);
    std::ofstream histories("particle_contact_history_debug.csv", mode);
    if (!initialized) {
        forces << "step,particle"
               << ",pos_x,pos_y,pos_z,vel_x,vel_y,vel_z,omega_x,omega_y,omega_z"
               << ",force_old_x,force_old_y,force_old_z,torque_old_x,torque_old_y,torque_old_z"
               << ",collision_force_x,collision_force_y,collision_force_z"
               << ",collision_torque_x,collision_torque_y,collision_torque_z"
               << ",pp_force_x,pp_force_y,pp_force_z,pp_torque_x,pp_torque_y,pp_torque_z"
               << ",wall_force_x,wall_force_y,wall_force_z,wall_torque_x,wall_torque_y,wall_torque_z"
               << ",fluid_force_x,fluid_force_y,fluid_force_z,fluid_torque_x,fluid_torque_y,fluid_torque_z"
               << ",total_force_x,total_force_y,total_force_z,total_torque_x,total_torque_y,total_torque_z"
               << ",rotation_rhs_x,rotation_rhs_y,rotation_rhs_z"
               << ",body_force_x,body_force_y,body_force_z,q_x,q_y,q_z,q_w\n";
        histories << "step,particle,slot,kind,partner,last_step,overlap"
                  << ",xi_x,xi_y,xi_z,fn_x,fn_y,fn_z,fn_mag"
                  << ",ft_x,ft_y,ft_z,ft_mag,coulomb_limit\n";
        initialized = true;
    }
    forces << std::scientific << std::setprecision(9);
    histories << std::scientific << std::setprecision(9);

    const dfloat3 gravity(GX, GY, GZ);
    for (int i = 0; i < NUM_PARTICLES; ++i) {
        ParticleCenter& collision = collisionSnapshot[i];
        ParticleCenter& final = finalSnapshot[i];
        const dfloat3 collisionForce = collision.getF();
        const dfloat3 collisionTorque = collision.getM();
        const dfloat3 fluidForce = final.getF() - collisionForce;
        const dfloat3 fluidTorque = final.getM() - collisionTorque;
        const dfloat3 rotationRhs = final.getDL_internal() +
            0.5_df * (final.getM_old() + final.getM());
        const dfloat3 bodyForce =
            final.getVolume() * (final.getDensity() - FLUID_DENSITY) * gravity;
        const dfloat4 q = final.getQ_cumulative_rot();

        forces << step << ',' << i;
        writeVec3(forces, final.getPos());
        writeVec3(forces, final.getVel());
        writeVec3(forces, final.getW());
        writeVec3(forces, final.getF_old());
        writeVec3(forces, final.getM_old());
        writeVec3(forces, collisionForce);
        writeVec3(forces, collisionTorque);
        writeVec3(forces, collision.getDebugPPForce());
        writeVec3(forces, collision.getDebugPPTorque());
        writeVec3(forces, collision.getDebugWallForce());
        writeVec3(forces, collision.getDebugWallTorque());
        writeVec3(forces, fluidForce);
        writeVec3(forces, fluidTorque);
        writeVec3(forces, final.getF());
        writeVec3(forces, final.getM());
        writeVec3(forces, rotationRhs);
        writeVec3(forces, bodyForce);
        forces << ',' << q.x << ',' << q.y << ',' << q.z << ',' << q.w << '\n';

        const CollisionData& contacts = collision.getCollision();
        for (int slot = 0; slot < MAX_ACTIVE_COLLISIONS; ++slot) {
            if (contacts.getLastCollisionStep(slot) != static_cast<int>(step)) continue;
            const bool wall = slot < FIRST_PARTICLE_COLLISION_SLOT;
            const dfloat3 xi = contacts.getTangentialDisplacement(slot);
            const dfloat3 fn = contacts.getDebugNormalForce(slot);
            const dfloat3 ft = contacts.getDebugTangentialForce(slot);
            const dfloat friction = wall ? PW_FRICTION_COEF : PP_FRICTION_COEF;
            histories << step << ',' << i << ',' << slot << ','
                      << (wall ? "wall" : "particle") << ','
                      << (wall ? slot : contacts.getCollisionPartnerID(slot)) << ','
                      << contacts.getLastCollisionStep(slot) << ','
                      << contacts.getDebugOverlap(slot) << ','
                      << xi.x << ',' << xi.y << ',' << xi.z << ','
                      << fn.x << ',' << fn.y << ',' << fn.z << ',' << vector_length(fn) << ','
                      << ft.x << ',' << ft.y << ',' << ft.z << ',' << vector_length(ft) << ','
                      << friction * vector_length(fn) << '\n';
        }
    }
}
} // namespace
#endif

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
    checkCudaErrors(cudaStreamSynchronize(streamParticles[0]));
    particlesCollisionHandler<<<GRID_PCOLLISION, TOTAL_PCOLLISION, 0, streamParticles[0]>>>(shape,pArray,d_pwForces,step);
    checkCudaErrors(cudaStreamSynchronize(streamParticles[0]));
    #ifdef PARTICLE_FORCE_DEBUG
    captureCollisionSnapshot(pArray, step);
    #endif

    int numIBM    = particles->getMethodCount(IBM);
    int numPIBM   = particles->getMethodCount(PIBM);
    int numTRACER = particles->getMethodCount(TRACER);

    if(numIBM>0){
       ibmSimulation(particles,fMom,streamParticles[0],step);
       // Synchronize after IBM to catch any errors early
       checkCudaErrors(cudaStreamSynchronize(streamParticles[0]));
    }
    if(numPIBM>0){
        pibmSimulation(particles,fMom,streamParticles[0],step);
        // Synchronize after PIBM to catch any errors early
        checkCudaErrors(cudaStreamSynchronize(streamParticles[0]));
    }
    if(numTRACER>0){
        tracerSimulation(particles,fMom,streamParticles[0],step);
        // Synchronize after TRACER to catch any errors early
        checkCudaErrors(cudaStreamSynchronize(streamParticles[0]));
    }

    #ifdef PARTICLE_FORCE_DEBUG
    exportParticleForceDebug(pArray, step);
    #endif

    updateParticleCenterVelocityAndRotation<<<GRID_PARTICLES, THREADS_PARTICLES, 0, streamParticles[0]>>>(pArray,step);
    checkCudaErrors(cudaStreamSynchronize(streamParticles[0]));
    updateParticlePosition<<<GRID_PARTICLES, THREADS_PARTICLES, 0, streamParticles[0]>>>(pArray,step);
    checkCudaErrors(cudaStreamSynchronize(streamParticles[0]));
}

#endif //PARTICLE_MODEL
