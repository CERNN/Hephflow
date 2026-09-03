/**
 *  @file ibm.cuh
 *  Contributors history:
 *  @author Waine Jr. (waine@alunos.utfpr.edu.br)
 *  @author Marco Aurelio Ferrari (e.marcoferrari@utfpr.edu.br)
 *  @author Ricardo de Souza
 *  @brief Perform the particle dynamics
 *  @version 0.4.0
 *  @date 01/01/2025
 */



#ifndef __PARTICLE_MOVEMENT_H
#define __PARTICLE_MOVEMENT_H

#include "../../../globalStructs.h"
#include "../../../globalFunctions.h"
#include "../particleSharedFunctions.cuh"
#include "../../../include/interface.h"
#include "../../../include/errorDef.h"
#include "../../../saveData.cuh"
#include "../../class/Particle.cuh"

#ifndef PARTICLE_ROTATION_SUBSTEPS
#define PARTICLE_ROTATION_SUBSTEPS 1
#endif

#ifndef PARTICLE_ROTATION_MAX_ITERS
#define PARTICLE_ROTATION_MAX_ITERS 5
#endif

#ifndef PARTICLE_ROTATION_REL_TOL
#define PARTICLE_ROTATION_REL_TOL 1.0e-5f
#endif

static_assert(PARTICLE_ROTATION_SUBSTEPS > 0, "PARTICLE_ROTATION_SUBSTEPS must be positive");
static_assert(PARTICLE_ROTATION_MAX_ITERS > 0, "PARTICLE_ROTATION_MAX_ITERS must be positive");
static_assert(PARTICLE_ROTATION_REL_TOL > 0.0f, "PARTICLE_ROTATION_REL_TOL must be positive");

#ifdef PARTICLE_MODEL

/**
 *  @brief Update the old values of particle properties (position, velocity, angular velocity, force and torque).
 *  @param pArray: Pointer to the array of ParticleCenter objects.
 *  @param step: The current simulation time step for collision checking.
 */
__global__
void updateParticleOldValues(
    ParticleCenter *pArray,
    unsigned int step
);

/**
 *  @brief Compute the new particle properties (velocity, angular velocity, force and torque).
 *  @param pArray: Pointer to the array of ParticleCenter objects.
 *  @param step: The current simulation time step for collision checking.
 */
__global__ 
void updateParticleCenterVelocityAndRotation(
    ParticleCenter *pArray,
    unsigned int step
);

/**
 *  @brief Compute the new particle position.
 *  @param pArray: Pointer to the array of ParticleCenter objects.
 *  @param step: The current simulation time step for collision checking.
 */
__global__
void updateParticlePosition(
    ParticleCenter *pArray,
    unsigned int step
);

/**
 * @brief Update the semi-axis positions using cumulative rotation and original offsets.
 * Reconstructs semi-axis position from first principles each frame to avoid error accumulation.
 * Periodic wrapping is implicit through particle_center position (which is already wrapped).
 * @param semi_offset_original: The immutable original offset (semi_axis - center) at initialization.
 * @param particle_center: The current particle center position (already wrapped if periodic BC).
 * @param q_cumulative: The cumulative rotation quaternion (complete rotation history).
 * @return The updated semi-axis position.
 */
__host__ __device__
dfloat3 updateSemiAxis(
    const dfloat3 semi_offset_original,
    const dfloat3 particle_center,
    const dfloat4 q_cumulative
);

#endif //PARTICLE_MODEL
#endif // !__PARTICLE_MOVEMENT_H
