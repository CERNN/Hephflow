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
#include "../../../includeFiles/interface.h"
#include "../../../errorDef.h"
#include "../../../saveData.cuh"
#include "../../class/Particle.cuh"

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
 * @brief Update the semi-axis positions considering periodic boundaries.
 * @param semi: The semi-axis positions to be updated.
 * @param pos_old: The old position of the particle.
 * @param pos_new: The current position of the particle.
 * @param q: The quaternion representing the particle's orientation.
 * @return The updated semi-axis positions.
 */
__host__ __device__
dfloat3 updateSemiAxis(
    dfloat3 semi,
    const dfloat3 pos_old,
    const dfloat3 pos_new,
    const dfloat4 q
);

#endif //PARTICLE_MODEL
#endif // !__PARTICLE_MOVEMENT_H


