/**
 *  @file particleNearFieldForces.cuh
 *  Contributors history:
 *  @author Marco Aurelio Ferrari (e.marcoferrari@utfpr.edu.br)
 *  @brief Perform the collision detection between particles
 *  @version 0.1.0
 *  @date 14/12/2025
 */

#ifndef __NEAR_FIELD_FORCES_H
#define __NEAR_FIELD_FORCES_H

#include "../../../../globalStructs.h"
#include "../../../../globalFunctions.h"
#include "../../../class/Particle.cuh"
#include "collision.cuh"


#ifdef PARTICLE_MODEL
#ifdef ENABLE_LUBRICATION
/**
 * @brief Compute near-field hydrodynamic lubrication forces between a sphere and a wall.
 * @param ctx  Collision context containing particle and wall data.
 * @param gap  Normal gap distance between the particle surface and the wall.
 */
__device__ 
void sphereWallLubrication(const CollisionContext& ctx, dfloat gap);
#endif
#ifdef ENABLE_REPULSIVE_FORCE
/**
 * @brief Compute near-field repulsive forces between a sphere and a wall.
 * @param ctx  Collision context containing particle and wall data.
 * @param gap  Normal gap distance between the particle surface and the wall.
 */
__device__ 
void sphereWallRepulsion (const CollisionContext& ctx, dfloat gap);
#endif
#ifdef ENABLE_ATTRACTIVE_FORCE
/**
 * @brief Compute near-field attractive forces between a sphere and a wall.
 * @param ctx  Collision context containing particle and wall data.
 * @param gap  Normal gap distance between the particle surface and the wall.
 */
__device__ 
void sphereWallAttraction(const CollisionContext& ctx, dfloat gap);
#endif
#ifdef ENABLE_LUBRICATION
/**
 * @brief Compute near-field hydrodynamic lubrication forces between two spheres.
 * @param ctx  Collision context containing particle-particle interaction data.
 * @param gap  Normal gap distance between the surfaces of the two particles.
 */
__device__
void sphereSphereLubrication(const CollisionContext& ctx, dfloat gap);
#endif
#ifdef ENABLE_REPULSIVE_FORCE
/**
 * @brief Compute near-field repulsive forces between two spheres.
 * @param ctx  Collision context containing particle-particle interaction data.
 * @param gap  Normal gap distance between the surfaces of the two particles.
 */
__device__
void sphereSphereRepulsion(const CollisionContext& ctx, dfloat gap);
#endif
#ifdef ENABLE_ATTRACTIVE_FORCE
/**
 * @brief Compute near-field attractive forces between two spheres.
 * @param ctx  Collision context containing particle-particle interaction data.
 * @param gap  Normal gap distance between the surfaces of the two particles.
 */
__device__
void sphereSphereAttraction(const CollisionContext& ctx, dfloat gap);
#endif
#endif //PARTICLE_MODEL
#endif // !__NEAR_FIELD_FORCES_H
