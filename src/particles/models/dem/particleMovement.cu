

//functions related to the rigid body body of the particle and discretization

#include "particleMovement.cuh"

#ifdef PARTICLE_MODEL
__global__
void updateParticleOldValues(
    ParticleCenter *pArray,
    unsigned int step)
{
    unsigned int localIdx = threadIdx.x + blockDim.x * blockIdx.x;
    int globalIdx = localIdx;

    if (globalIdx >= NUM_PARTICLES) {
        return;
    }

    if (pArray == nullptr) {
        printf("ERROR: particles is nullptr\n");
        return;
    }


    ParticleCenter* pc_i = &pArray[globalIdx];

    // Internal linear momentum delta = rho*volume*delta(v)/delta(t)
    // https://doi.org/10.1016/j.compfluid.2011.05.011
    //pc_i->setDPInternalX(RHO_0 * pc_i->getVolume() * (pc_i->getVelX() - pc_i->getVelOldX())); //;
    //pc_i->setDPInternalY(RHO_0 * pc_i->getVolume() * (pc_i->getVelY() - pc_i->getVelOldY())); //;
    //pc_i->setDPInternalZ(RHO_0 * pc_i->getVolume() * (pc_i->getVelZ() - pc_i->getVelOldZ())); //;
    pc_i->setDPInternalX(0.0);
    pc_i->setDPInternalY(0.0);
    pc_i->setDPInternalZ(0.0);

    // Internal angular momentum delta = (rho_f/rho_p)*I*delta(omega)/delta(t)
    // https://doi.org/10.1016/j.compfluid.2011.05.011
    
    //pc_i->setDLInternalX((RHO_0 / pc_i->getDensity()) * pc_i->getIXX() * (pc_i->getWX() - pc_i->getWOldX())); 
    //pc_i->setDLInternalY((RHO_0 / pc_i->getDensity()) * pc_i->getIYY() * (pc_i->getWY() - pc_i->getWOldY())); 
    //pc_i->setDLInternalZ((RHO_0 / pc_i->getDensity()) * pc_i->getIZZ() * (pc_i->getWZ() - pc_i->getWOldZ())); 
    pc_i->setDLInternalX(0.0);
    pc_i->setDLInternalY(0.0);
    pc_i->setDLInternalZ(0.0);

    #ifdef PARTICLE_DEBUG
    printf("updateParticleOldValues 2 pos  x: %e y: %e z: %e\n",pc_i->getPosOldX(),pc_i->getPosOldY(),pc_i->getPosOldZ());
    printf("updateParticleOldValues 3 pos  x: %e y: %e z: %e\n",pc_i->getVelOldX(),pc_i->getVelOldY(),pc_i->getVelOldZ());
    printf("updateParticleOldValues 4 pos  x: %e y: %e z: %e\n",pc_i->getWOldX(),pc_i->getWOldY(),pc_i->getWOldZ());
    printf("updateParticleOldValues 5 pos  x: %e y: %e z: %e\n",pc_i->getFOldX(),pc_i->getFOldY(),pc_i->getFOldZ());
    printf("updateParticleOldValues 6 pos  x: %e y: %e z: %e\n",pc_i->getFX(),pc_i->getFY(),pc_i->getFZ());
    printf("updateParticleOldValues 7 pos  x: %e y: %e z: %e\n",pc_i->getMX(),pc_i->getMY(),pc_i->getMZ());
    #endif //PARTICLE_DEBUG

    pc_i->setPos_old(pc_i->getPos());
    pc_i->setVel_old(pc_i->getVel());
    pc_i->setW_old(pc_i->getW());
    pc_i->setF_old(pc_i->getF());
    pc_i->setM_old(pc_i->getM());
    pc_i->setF(dfloat3(0,0,0));
    pc_i->setM(dfloat3(0,0,0));
    #ifdef PARTICLE_FORCE_DEBUG
    pc_i->resetDebugCollisionLoads();
    #endif

}

namespace {

enum RotationFailureStage {
    ROTATION_OK = 0,
    ROTATION_INVALID_INERTIA = 1,
    ROTATION_INVALID_INPUT = 2,
    ROTATION_INVALID_QUATERNION = 3,
    ROTATION_INVALID_MOMENTUM = 4,
    ROTATION_INVALID_OMEGA = 5,
    ROTATION_NO_CONVERGENCE = 6
};

__device__ __forceinline__
const char* rotationFailureStageName(RotationFailureStage stage) {
    switch (stage) {
        case ROTATION_INVALID_INERTIA: return "invalid-inertia";
        case ROTATION_INVALID_INPUT: return "invalid-input";
        case ROTATION_INVALID_QUATERNION: return "invalid-quaternion";
        case ROTATION_INVALID_MOMENTUM: return "invalid-momentum";
        case ROTATION_INVALID_OMEGA: return "invalid-omega";
        case ROTATION_NO_CONVERGENCE: return "no-convergence";
        default: return "unknown";
    }
}

struct RotationAdvanceResult {
    bool ok;
    dfloat3 omega;
    dfloat4 q_relative;
    dfloat6 inertia;
    int substep;
    int iteration;
    dfloat convergence_metric;
    RotationFailureStage failure_stage;
};

__host__ __device__ __forceinline__
bool finite3(const dfloat3& v) {
    return isfinite(v.x) && isfinite(v.y) && isfinite(v.z);
}

__host__ __device__ __forceinline__
dfloat inertiaRelativeChange(const dfloat6& current, const dfloat6& previous, const dfloat3& principal) {
    const dfloat scale = fmaxf(principal.x, fmaxf(principal.y, principal.z));
    const dfloat inv_scale = 1.0_df / scale;
    const dfloat dxx = (current.xx - previous.xx) * inv_scale;
    const dfloat dyy = (current.yy - previous.yy) * inv_scale;
    const dfloat dzz = (current.zz - previous.zz) * inv_scale;
    const dfloat dxy = (current.xy - previous.xy) * inv_scale;
    const dfloat dxz = (current.xz - previous.xz) * inv_scale;
    const dfloat dyz = (current.yz - previous.yz) * inv_scale;
    return sqrtf(dxx*dxx + dyy*dyy + dzz*dzz +
                 2.0_df*(dxy*dxy + dxz*dxz + dyz*dyz));
}

__host__ __device__ __forceinline__
bool incrementalRotation(const dfloat3& omega, dfloat stage_dt, dfloat4* rotation) {
    if (rotation == nullptr || !finite3(omega) || !isfinite(stage_dt) || stage_dt <= 0.0_df) {
        return false;
    }

    const dfloat norm_sq = dot_product(omega, omega);
    if (!isfinite(norm_sq) || norm_sq < 0.0_df) {
        return false;
    }

    const dfloat norm = sqrtf(norm_sq);
    const dfloat half_angle = 0.5_df * stage_dt * norm;
    if (!isfinite(half_angle)) {
        return false;
    }

    // sin(x)/|omega| tends to stage_dt/2 as |omega| tends to zero.
    const dfloat vector_scale = norm > 1.0e-7_df ? sinf(half_angle) / norm : 0.5_df * stage_dt;
    const dfloat4 raw(
        omega.x * vector_scale,
        omega.y * vector_scale,
        omega.z * vector_scale,
        cosf(half_angle));
    return quart_normalize_safe(raw, rotation);
}

__host__ __device__
RotationAdvanceResult advanceRotationArdekani(
    dfloat3 omega_initial,
    dfloat4 q_relative_initial,
    dfloat4 q_reference,
    dfloat3 principal_inertia,
    dfloat3 torque_rate)
{
    RotationAdvanceResult result = {};
    result.ok = false;
    result.omega = omega_initial;
    result.q_relative = q_relative_initial;
    result.failure_stage = ROTATION_INVALID_INPUT;

    if (!finite3(principal_inertia) || principal_inertia.x <= 0.0_df ||
        principal_inertia.y <= 0.0_df || principal_inertia.z <= 0.0_df) {
        result.failure_stage = ROTATION_INVALID_INERTIA;
        return result;
    }
    if (!finite3(omega_initial) || !finite3(torque_rate)) {
        return result;
    }
    if (!quart_normalize_safe(q_relative_initial, &result.q_relative) ||
        !quart_normalize_safe(q_reference, &q_reference)) {
        result.failure_stage = ROTATION_INVALID_QUATERNION;
        return result;
    }

    const dfloat stage_dt = 1.0_df / (dfloat)PARTICLE_ROTATION_SUBSTEPS;
    dfloat3 omega_start = omega_initial;
    dfloat4 q_relative_start = result.q_relative;
    dfloat6 inertia_final = {};

    for (int substep = 0; substep < PARTICLE_ROTATION_SUBSTEPS; ++substep) {
        result.substep = substep;
        dfloat4 q_absolute_start;
        if (!quart_normalize_safe(quart_multiplication(q_relative_start, q_reference), &q_absolute_start)) {
            result.failure_stage = ROTATION_INVALID_QUATERNION;
            return result;
        }

        const dfloat3 angular_momentum_start =
            apply_world_inertia(omega_start, q_absolute_start, principal_inertia);
        const dfloat3 target_momentum = angular_momentum_start + stage_dt * torque_rate;
        if (!finite3(target_momentum)) {
            result.failure_stage = ROTATION_INVALID_MOMENTUM;
            return result;
        }

        dfloat6 inertia_guess = world_inertia_from_principal(q_absolute_start, principal_inertia);
        dfloat3 omega_guess = apply_world_inverse_inertia(target_momentum, q_absolute_start, principal_inertia);
        if (!finite3(omega_guess)) {
            result.failure_stage = ROTATION_INVALID_OMEGA;
            return result;
        }

        bool converged = false;
        dfloat4 q_relative_candidate = q_relative_start;
        dfloat3 omega_candidate = omega_guess;
        for (int iteration = 0; iteration < PARTICLE_ROTATION_MAX_ITERS; ++iteration) {
            result.iteration = iteration;
            const dfloat3 omega_average = 0.5_df * (omega_start + omega_guess);
            dfloat4 q_increment;
            if (!incrementalRotation(omega_average, stage_dt, &q_increment) ||
                !quart_normalize_safe(
                    quart_multiplication(q_increment, q_relative_start),
                    &q_relative_candidate)) {
                result.failure_stage = ROTATION_INVALID_QUATERNION;
                return result;
            }

            dfloat4 q_absolute_candidate;
            if (!quart_normalize_safe(
                    quart_multiplication(q_relative_candidate, q_reference),
                    &q_absolute_candidate)) {
                result.failure_stage = ROTATION_INVALID_QUATERNION;
                return result;
            }

            const dfloat6 inertia_candidate =
                world_inertia_from_principal(q_absolute_candidate, principal_inertia);
            const dfloat relative_change =
                inertiaRelativeChange(inertia_candidate, inertia_guess, principal_inertia);
            result.convergence_metric = relative_change;
            omega_candidate =
                apply_world_inverse_inertia(target_momentum, q_absolute_candidate, principal_inertia);
            if (!isfinite(relative_change) || !finite3(omega_candidate)) {
                result.failure_stage = ROTATION_INVALID_OMEGA;
                return result;
            }

            inertia_guess = inertia_candidate;
            omega_guess = omega_candidate;
            inertia_final = inertia_candidate;
            if (relative_change <= (dfloat)PARTICLE_ROTATION_REL_TOL) {
                converged = true;
                break;
            }
        }

        if (!converged) {
            result.failure_stage = ROTATION_NO_CONVERGENCE;
            return result;
        }

        omega_start = omega_candidate;
        q_relative_start = q_relative_candidate;
    }

    result.ok = true;
    result.failure_stage = ROTATION_OK;
    result.omega = omega_start;
    result.q_relative = q_relative_start;
    result.inertia = inertia_final;
    return result;
}

__device__
void reportRotationFailure(
    ParticleCenter* particle,
    int particle_index,
    unsigned int step,
    const RotationAdvanceResult& result,
    const dfloat3& principal,
    const dfloat3& torque)
{
    if (!particle->getRotationFaulted()) {
        const dfloat4 q = particle->getQ_cumulative_rot();
        const dfloat3 w = particle->getW_old();
        printf("ERROR: Particle rotation fault particle=%d step=%u substep=%d iteration=%d stage=%s "
               "rel_change=%e tol=%e I=(%e,%e,%e) torque=(%e,%e,%e) "
               "omega=(%e,%e,%e) q=(%e,%e,%e,%e)\n",
               particle_index, step, result.substep, result.iteration,
               rotationFailureStageName(result.failure_stage),
               result.convergence_metric, (dfloat)PARTICLE_ROTATION_REL_TOL,
               principal.x, principal.y, principal.z, torque.x, torque.y, torque.z,
               w.x, w.y, w.z, q.x, q.y, q.z, q.w);
    }
    particle->setRotationFaulted(true);
    particle->setQ_cumulative_rot(particle->getQ_cumulative_rot_last_valid());
    particle->setW(dfloat3(0.0_df, 0.0_df, 0.0_df));
    particle->setW_old(dfloat3(0.0_df, 0.0_df, 0.0_df));
    particle->setW_avg(dfloat3(0.0_df, 0.0_df, 0.0_df));
}

} // namespace

__global__ 
void updateParticleCenterVelocityAndRotation(
    ParticleCenter *pArray,
    unsigned int step)
{
    unsigned int localIdx = threadIdx.x + blockDim.x * blockIdx.x;
    int globalIdx = localIdx;

    if (globalIdx >= NUM_PARTICLES) {
        return;
    }

    if (pArray == nullptr) {
        printf("ERROR: particles is nullptr\n");
        return;
    }

    ParticleCenter* pc_i = &pArray[globalIdx];

    if(!pc_i->getMovable())
        return;

    #ifdef PARTICLE_DEBUG
    printf("updateParticleCenterVelocityAndRotation 1 pos  x: %e y: %e z: %e\n",pc_i->getPosX(),pc_i->getPosY(),pc_i->getPosZ());
    printf("updateParticleCenterVelocityAndRotation 1 vel  x: %e y: %e z: %e\n",pc_i->getVel().x,pc_i->getVel().y,pc_i->getVel().z);
    printf("updateParticleCenterVelocityAndRotation 1 w  x: %e y: %e z: %e\n",pc_i->getWX(),pc_i->getWY(),pc_i->getWZ());
    printf("updateParticleCenterVelocityAndRotation 1 f  x: %e y: %e z: %e\n",pc_i->getF().x,pc_i->getF().y,pc_i->getF().z);
    printf("updateParticleCenterVelocityAndRotation 1 m  x: %e y: %e z: %e\n",pc_i->getMX(),pc_i->getMY(),pc_i->getMZ());
    printf("updateParticleCenterVelocityAndRotation 1 DP  x: %e y: %e z: %e\n",pc_i->getDP_internal().x,pc_i->getDP_internal().y,pc_i->getDP_internal().z);
    printf("updateParticleCenterVelocityAndRotation 1 pos_old  x: %e y: %e z: %e\n",pc_i->getPosOldX(),pc_i->getPosOldY(),pc_i->getPosOldZ());
    printf("updateParticleCenterVelocityAndRotation 1 vel_old  x: %e y: %e z: %e\n",pc_i->getVelOldX(),pc_i->getVelOldY(),pc_i->getVelOldZ());
    printf("updateParticleCenterVelocityAndRotation 1 w_old  x: %e y: %e z: %e\n",pc_i->getWOldX(),pc_i->getWOldY(),pc_i->getWOldZ());
    printf("updateParticleCenterVelocityAndRotation 1 f_old  x: %e y: %e z: %e\n",pc_i->getFOldX(),pc_i->getFOldY(),pc_i->getFOldZ());
    printf("updateParticleCenterVelocityAndRotation 1 m_old  x: %e y: %e z: %e\n",pc_i->getMOldX(),pc_i->getMOldY(),pc_i->getMOldZ());
    printf("updateParticleCenterVelocityAndRotation 1 volume %e\n",pc_i->getVolume());
    printf("updateParticleCenterVelocityAndRotation 1 density %e\n",pc_i->getDensity());
    #endif //PARTICLE_DEBUG

    // Update particle center velocity using its surface forces and the body forces
    dfloat3 g = {GX,GY,GZ};
    dfloat volume = pc_i->getVolume();
    const dfloat inv_volume = 1 / volume;
    pc_i->setVel(pc_i->getVel_old() + (((pc_i->getF_old() + pc_i->getF())/2 + pc_i->getDP_internal())*inv_volume
                + (pc_i->getDensity() - FLUID_DENSITY)*g) / (pc_i->getDensity()));
    //pc_i->setVel(pc_i->getVel_old() + (((pc_i->getF_old() + pc_i->getF())/2 + pc_i->getDP_internal())) / (pc_i->getVolume()) 
    //            + (1.0 - FLUID_DENSITY/pc_i->getDensity()) * g);


    // Ardekani et al. (2016), Eqs. (7)-(8): advance angular momentum and
    // iteratively couple angular velocity to the end-of-stage orientation.
    if (pc_i->getRotationFaulted()) {
        pc_i->setW(dfloat3(0.0_df, 0.0_df, 0.0_df));
        pc_i->setW_old(dfloat3(0.0_df, 0.0_df, 0.0_df));
        pc_i->setW_avg(dfloat3(0.0_df, 0.0_df, 0.0_df));
        return;
    }

    const dfloat3 principal_inertia = pc_i->getPrincipalInertia();
    const dfloat3 torque_rate =
        pc_i->getDL_internal() + 0.5_df * (pc_i->getM_old() + pc_i->getM());
    const RotationAdvanceResult rotation = advanceRotationArdekani(
        pc_i->getW_old(),
        pc_i->getQ_cumulative_rot(),
        pc_i->getQ_inertia_reference(),
        principal_inertia,
        torque_rate);

    if (!rotation.ok) {
        reportRotationFailure(pc_i, globalIdx, step, rotation, principal_inertia, torque_rate);
        return;
    }

    pc_i->setW(rotation.omega);
    pc_i->setW_avg(0.5_df * (pc_i->getW_old() + rotation.omega));
    pc_i->setQ_cumulative_rot(rotation.q_relative);
    pc_i->setI(rotation.inertia);

    #ifdef PARTICLE_DEBUG
    printf("updateParticleCenterVelocityAndRotation 2 pos  x: %e y: %e z: %e\n",pc_i->getPosX(),pc_i->getPosY(),pc_i->getPosZ());
    printf("updateParticleCenterVelocityAndRotation 2 vel  x: %e y: %e z: %e\n",pc_i->getVel().x,pc_i->getVel().y,pc_i->getVel().z);
    printf("updateParticleCenterVelocityAndRotation 2 w  x: %e y: %e z: %e\n",pc_i->getWX(),pc_i->getWY(),pc_i->getWZ());
    #endif //PARTICLE_DEBUG
}

__global__
void updateParticlePosition(
    ParticleCenter *pArray,
    unsigned int step)
{
    unsigned int localIdx = threadIdx.x + blockDim.x * blockIdx.x;
    int globalIdx = localIdx;

    if (globalIdx >= NUM_PARTICLES) {
        return;
    }

    if (pArray == nullptr) {
        printf("ERROR: particles is nullptr\n");
        return;
    }

    ParticleCenter* pc_i = &pArray[globalIdx];

    if(!pc_i->getMovable())
        return;

    #ifdef PARTICLE_DEBUG
    printf("updateParticlePosition 1 pos  x: %e y: %e z: %e\n",pc_i->getPosX(),pc_i->getPosY(),pc_i->getPosZ());
    printf("updateParticlePosition 1 vel  x: %e y: %e z: %e\n",pc_i->getVel().x,pc_i->getVel().y,pc_i->getVel().z);
    printf("updateParticlePosition 1 w  x: %e y: %e z: %e\n",pc_i->getWX(),pc_i->getWY(),pc_i->getWZ());
    #endif //PARTICLE_DEBUG

    #ifdef BC_X_WALL
        pc_i->setPosX(pc_i->getPosX() + (pc_i->getVelX() + pc_i->getVelOldX())/2);
    #endif //BC_X_WALL
    #ifdef BC_X_PERIODIC
        dfloat dx  = (pc_i->getVelX() + pc_i->getVelOldX())/2;
        dfloat new_x = pc_i->getPosX() + dx;
        dfloat mod_x = std::fmod(new_x, (dfloat)NX);
        pc_i->setPosX((mod_x < 0) ? mod_x + (dfloat)NX : mod_x);
    #endif //BC_X_PERIODIC

    #ifdef BC_Y_WALL
        pc_i->setPosY(pc_i->getPosY() + (pc_i->getVelY() + pc_i->getVelOldY())/2);
    #endif //BC_Y_WALL
    #ifdef BC_Y_PERIODIC
        dfloat dy  = (pc_i->getVelY() + pc_i->getVelOldY())/2;
        dfloat new_y = pc_i->getPosY() + dy;
        dfloat mod_y = std::fmod(new_y, (dfloat)NY);
        pc_i->setPosY((mod_y < 0) ? mod_y + (dfloat)NY : mod_y);
    #endif //BC_Y_PERIODIC

    #ifdef BC_Z_WALL
        pc_i->setPosZ(pc_i->getPosZ() + (pc_i->getVelZ() + pc_i->getVelOldZ())/2);
    #endif //BC_Z_WALL
    #ifdef BC_Z_PERIODIC
        dfloat dz  = (pc_i->getVelZ() + pc_i->getVelOldZ())/2;
        dfloat new_z = pc_i->getPosZ() + dz;
        dfloat mod_z = std::fmod(new_z, (dfloat)NZ_TOTAL);
        pc_i->setPosZ((mod_z < 0) ? mod_z + (dfloat)NZ_TOTAL : mod_z);
    #endif //BC_Z_PERIODIC

    pc_i->setW_pos(pc_i->getW_pos() + pc_i->getW_avg());

    #ifdef PARTICLE_DEBUG
    printf("updateParticlePosition 2 pos  x: %e y: %e z: %e\n",pc_i->getPosX(),pc_i->getPosY(),pc_i->getPosZ());
    printf("updateParticlePosition 2 vel  x: %e y: %e z: %e\n",pc_i->getVel().x,pc_i->getVel().y,pc_i->getVel().z);
    printf("updateParticlePosition 2 w  x: %e y: %e z: %e\n",pc_i->getWX(),pc_i->getWY(),pc_i->getWZ());
    #endif //PARTICLE_DEBUG


    const dfloat4 q_cumulative = pc_i->getQ_cumulative_rot();
    
    dfloat3 pos_old = pc_i->getPos_old();

    dfloat3 pos_new = pc_i->getPos();

    pc_i->setDx(pos_new - pos_old);

    // Update semi-axes using cumulative rotation and original offsets
    // This avoids error accumulation from incremental updates
    pc_i->setSemiAxis1(updateSemiAxis(pc_i->getSemiAxis1Original(), pos_new, q_cumulative));
    pc_i->setSemiAxis2(updateSemiAxis(pc_i->getSemiAxis2Original(), pos_new, q_cumulative));
    pc_i->setSemiAxis3(updateSemiAxis(pc_i->getSemiAxis3Original(), pos_new, q_cumulative));
}


__host__ __device__
dfloat3 updateSemiAxis(
    const dfloat3 semi_offset_original,
    const dfloat3 particle_center,
    const dfloat4 q_cumulative
){
    const dfloat3 rotated_offset = rotate_vector_by_quart_R(semi_offset_original, q_cumulative);
    dfloat3 newSemi = {
        particle_center.x + rotated_offset.x,
        particle_center.y + rotated_offset.y,
        particle_center.z + rotated_offset.z
    };

    return newSemi;
}

#endif //PARTICLE_MODEL
