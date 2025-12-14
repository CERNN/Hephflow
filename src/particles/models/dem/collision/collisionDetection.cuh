/**
 *  @file collisionDetection.cuh
 *  Contributors history:
 *  @author Waine Jr. (waine@alunos.utfpr.edu.br)
 *  @author Marco Aurelio Ferrari (e.marcoferrari@utfpr.edu.br)
 *  @author Ricardo de Souza
 *  @brief Perform the collision detection between particles
 *  @version 0.4.0
 *  @date 01/01/2025
 */

#ifndef ___COLLISION_DETECTION_H
#define __COLLISION_DETECTION_H

#include "../../ibm/ibmVar.h"
#include "../../../../globalStructs.h"
#include "../../../../globalFunctions.h"
#include "../../../class/Particle.cuh"
#include "collision.cuh"

#ifdef PARTICLE_MODEL

#ifndef WALL_VEL_UX
#define WALL_VEL_UX 0
#endif

#ifndef WALL_VEL_UY
#define WALL_VEL_UY 0
#endif

#ifndef WALL_VEL_UZ
#define WALL_VEL_UZ 0
#endif


//collission 

/**
 *  @brief Handles collisions between particles and walls or between pairs of particles on the GPU.
 *  @param particle: Array of `ParticleCenter` structures representing all particles.
 *  @param step: The current time step for collision processing.
 */
__global__
void particlesCollisionHandler(ParticleShape *shape, ParticleCenter *pArray, ParticleWallForces *d_pwForces, unsigned int step);

// collision between particles themselves
/**
 *  @brief Check for collisions between two particles by comparing the pair types and calling the proper function.
 *  @param column: The column index in a grid or matrix representing the particle's position.
 *  @param row: The row index in a grid or matrix representing the particle's position.
 *  @param pc_i: Pointer to the `ParticleCenter` structure containing information about the first particle.
 *  @param pc_j: Pointer to the `ParticleCenter` structure containing information about the second particle.
 *  @param step: The current time step for collision checking.
 */
__device__
void checkCollisionBetweenParticles(unsigned int column, unsigned int row, ParticleShape *shape_i,ParticleShape *shape_j, ParticleCenter* pc_i, ParticleCenter* pc_j, int step);

/**
 *  @brief Check for collisions between a particle and walls based on the particle's shape.
 *  @param pc_i: Pointer to the `ParticleCenter` structure containing particle information.
 *  @param step: The current time step for collision checking.
 */
__device__
void checkCollisionWalls(ParticleShape *shape, ParticleCenter* pc_i, ParticleWallForces *d_pwForces, unsigned int step);
/**
 *  @brief Check for collisions between a sphere and walls.
 *  @param pc_i: Pointer to the `ParticleCenter` structure containing sphere information.
 *  @param step: The current time step for collision checking.
 */
__device__
void checkCollisionWallsSphere(ParticleCenter* pc_i, ParticleWallForces *d_pwForces, unsigned int step);
/**
 *  @brief Check for collisions between a capsule and walls.
 *  @param pc_i: Pointer to the `ParticleCenter` structure containing capsule information.
 *  @param step: The current time step for collision checking.
 */
__device__
void checkCollisionWallsCapsule(ParticleCenter* pc_i, unsigned int step);
/**
 *  @brief Check for collisions between an ellipsoid particle and walls.
 *  @param pc_i: Pointer to the ParticleCenter structure representing the ellipsoid particle.
 *  @param step: The current simulation step or time index.
 */
__device__
void checkCollisionWallsElipsoid(ParticleCenter* pc_i, unsigned int step);

/**
 * @brief Handle collision type between two spheres.
 * @param column: The column index in a grid or matrix representing the particles' positions.
 * @param row: The row index in a grid or matrix representing the particles' positions.
 * @param pc_i: Pointer to the `ParticleCenter` structure containing information about the first sphere.
 * @param pc_j: Pointer to the `ParticleCenter` structure containing information about the second sphere.
 * @param step: The current time step for collision processing.
 */
__device__
void sphereSphereCollisionCheck(unsigned int column,unsigned int row,ParticleCenter* pc_i, ParticleCenter* pc_j, int step);

/**
 *  @brief Handle collision type between two capsules.
 *  @param pc_i: Pointer to the `ParticleCenter` structure containing information about the first capsule.
 *  @param pc_j: Pointer to the `ParticleCenter` structure containing information about the second capsule.
 *  @param step: The current time step for collision processing.
 *  @param capA1: Center of the cap1 of particle i
 *  @param capA2: Center of the cap2 of particle i
 *  @param radiusA: Radius of particle i
 *  @param capB1: Center of the cap1 of particle j
 *  @param capB2: Center of the cap2 of particle j
 *  @param radiusB: Radius of particle j
 */
__device__
void capsuleCapsuleCollisionCheck(unsigned int column,    unsigned int row, ParticleCenter* pc_i, ParticleCenter* pc_j, int step, dfloat3 capA1, dfloat3 capA2,dfloat radiusA, dfloat3 capB1, dfloat3 capB2,dfloat radiusB);

/**
 *  @brief Handle collision mechanics between a capsule and an ellipsoid.
 *  @param column: The column index in a grid or matrix representing the particles' positions.
 *  @param row: The row index in a grid or matrix representing the particles' positions.
 *  @param pc_i: Pointer to the `ParticleCenter` structure containing information about the capsule.
 *  @param pc_j: Pointer to the `ParticleCenter` structure containing information about the ellipsoid.
 *  @param step: The current time step for collision processing.
 */
__device__
void capsuleSphereCollisionCheck( unsigned int column, unsigned int row, ParticleShape *shape, ParticleCenter* pc_i,  ParticleCenter* pc_j, int step);

/**
 *  @brief Check for a potential collision between two ellipsoids and trigger collision response if necessary.
 *  @param column: The column index representing the position of the first ellipsoid in the grid or matrix.
 *  @param row: The row index representing the position of the second ellipsoid in the grid or matrix.
 *  @param pc_i: Pointer to the `ParticleCenter` structure containing information about the first ellipsoid.
 *  @param pc_j: Pointer to the `ParticleCenter` structure containing information about the second ellipsoid.
 *  @param step: The current simulation time step for collision checking.
 */
__device__
void ellipsoidEllipsoidCollisionCheck(unsigned int column, unsigned int row, ParticleCenter* pc_i,ParticleCenter* pc_j, int step);

/**
 *  @brief Compute the gap between two spheres.
 *  @param pc_i: Pointer to the `ParticleCenter` structure containing information about the first sphere.
 *  @param pc_j: Pointer to the `ParticleCenter` structure containing information about the second sphere.
 *  @return The distance between the surfaces of the two spheres.
 */
__device__
dfloat sphereSphereGap(ParticleCenter*  pc_i, ParticleCenter*  pc_j);

#ifdef CURVED_BOUNDARY_CONDITION
/**
*   @brief determine wall properties for duct, based on contact position and radius
*   @param pos_i: coordinates of the collision point in the body center .
*   @param R: Radius of the duct
*   @param dir: direction of the duct, -1 for internal collision and +1 for external collision
*/
__device__
Wall determineCircularWall(dfloat3 pos_i, dfloat R, dfloat dir);
#endif 

/**
 * @brief Calculate the collision distance between an ellipsoid and a cylinder segment.
 * @param pc_i: Pointer to the `ParticleCenter` structure representing the ellipsoid.
 * @param P1: The starting point of the cylinder segment.
 * @param P2: The ending point of the cylinder segment.
 * @param cRadius: The radius of the cylinder surrounding the segment.
 * @param contactPoint1: Output parameter to store the contact point on the ellipsoid.
 * @param contactPoint2: Output parameter to store the contact point on the cylinder segment.
 * @param cr: Output parameter to store the contact radius.
 * @param cyDir: Direction indicator for the cylinder (-1 for externa, 1 for internal
 * @param step: The current simulation time step for collision checking.
 * @return The distance between the surfaces of the cylinder and ellipsoid
 */
__device__
dfloat ellipsoidSegmentCollisionDistance( ParticleCenter* pc_i, dfloat3 P1, dfloat3 P2, dfloat cRadius ,dfloat3 contactPoint1[1], dfloat3 contactPoint2[1], dfloat cr[1], int cyDir, unsigned int step);

#endif //PARTICLE_MODEL
#endif // !__COLLISION_H
