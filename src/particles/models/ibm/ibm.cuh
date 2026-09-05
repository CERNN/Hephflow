/**
 *  @file ibm.cuh
 *  @author Waine Jr. (waine@alunos.utfpr.edu.br)
 *  @author Marco Aurelio Ferrari (e.marcoferrari@utfpr.edu.br)
 *  @author Ricardo de Souza
 *  @brief IBM steps: perform interpolation and spread force
 *  @version 0.4.0
 *  @date 01/09/2025
 */


#ifndef __PARTICLE_MODEL_IBM_CUH
#define __PARTICLE_MODEL_IBM_CUH

#include "../../../globalStructs.h"
#include "../../../globalFunctions.h"
#include "../particleSharedFunctions.cuh"
#include "../../../include/interface.h"
#include "../../../include/errorDef.h"
#include "../../../saveData.cuh"
#include "../../class/Particle.cuh"

#include "../dem/particleMovement.cuh"
#include "../dem/collision/collisionDetection.cuh"

#ifdef PARTICLE_MODEL

enum IbmNodeDebugStatus {
    IBM_DEBUG_VALID = 0,
    IBM_DEBUG_NONFINITE_POSITION = 1,
    IBM_DEBUG_OUTSIDE_DOMAIN = 2,
    IBM_DEBUG_INVALID_STENCIL = 3,
    IBM_DEBUG_INVALID_EULERIAN_STATE = 4
};

struct IbmNodeDebugRecord {
    int nodeIndex;
    int particleIndex;
    int stencilStatus;
    int clippedPoints;
    int minIdx[3];
    int maxIdx[3];
    dfloat3 position;
    dfloat3 fluidVelocity;
    dfloat3 rigidVelocity;
    dfloat3 deltaForce;
    dfloat rho;
    dfloat forceScale;
    dfloat stencilWeightSum;
};


/**
 *  @brief Perform IBM simulation steps including force interpolation and spreading.
 *  @param particles: Pointer to the ParticlesSoA structure containing particle data.
 *  @param fMom: Pointer to the device array containing the current macroscopic moments.
 *  @param streamParticles: cuda stream for particles
 *  @param step: The current simulation time step for collision checking.
 */
void ibmSimulation(
    ParticlesSoA* particles,
    dfloat *fMom,
    cudaStream_t streamParticles,
    unsigned int step
);

/**
 *  @brief Reset the forces on IBM nodes to zero.
 *  @param particlesNodes: Pointer to the IbmNodesSoA structure containing IBM node data.
 *  @param step: The current simulation time step for collision checking.
 */
__global__ 
void ibmResetNodesForces(
    IbmNodesSoA* particlesNodes,
    unsigned int step
);


/**
 *  @brief Interpolate the predicted fluid velocity and compute one relaxed
 *         Lagrangian force correction.
 *  @param particlesNodes: Pointer to the IbmNodesSoA structure containing IBM node data.
 *  @param pArray: Pointer to the array of ParticleCenter objects.
 *  @param fMom: Pointer to the device array containing the current macroscopic moments.
 *  @param step: The current simulation time step for collision checking.
 */
__global__
void ibmForceInterpolationSpread(
    IbmNodesSoA* particlesNodes,
    ParticleCenter *pArray,
    dfloat *fMom,
    unsigned int step,
    IbmNodeDebugRecord* debugRecords
);

/**
 *  @brief Spread the current Lagrangian force correction to the Eulerian
 *         force moments. This is deliberately separate from interpolation so
 *         every IBM node in an iteration observes the same force field.
 */
__global__
void ibmSpreadForceCorrection(
    IbmNodesSoA* particlesNodes,
    dfloat *fMom
);

/**
 *  @brief 
 *  @param particlesNodes: Pointer to the IbmNodesSoA structure containing IBM node data.
 *  @param pArray: Pointer to the array of ParticleCenter objects.
 *  @param firstIndex: The first index of the particle array to be processed.
 *  @param lastIndex: The last index of the particle array to be processed.
 *  @param step: The current simulation time step for collision checking.
 */
__global__
void ibmParticleNodeMovement(
    IbmNodesSoA* particlesNodes,
    ParticleCenter *pArray,
    int firstIndex,
    int lastIndex,
    unsigned int step
);

#endif //PARTICLE_MODEL
#endif
