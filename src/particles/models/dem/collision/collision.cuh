/**
 *  @file collision.cuh
 *  Contributors history:
 *  @author Waine Jr. (waine@alunos.utfpr.edu.br)
 *  @author Marco Aurelio Ferrari (e.marcoferrari@utfpr.edu.br)
 *  @author Ricardo de Souza
 *  @brief Handle the collision dynamics between particles
 *  @version 0.4.0
 *  @date 01/01/2025
 */

#ifndef __IBM_COLLISION_H
#define __IBM_COLLISION_H

#include "../../ibm/ibmVar.h"
#include "../../../../globalStructs.h"
#include "../../../../globalFunctions.h"
#include "../../../class/Particle.cuh"

#ifdef PARTICLE_MODEL

#ifndef ELLIPSOID_PENETRATION_SPHERE_RADIUS
#define ELLIPSOID_PENETRATION_SPHERE_RADIUS 3.0f
#endif

enum EllipsoidContactStatus {
    ELLIPSOID_SEPARATED = 0,
    ELLIPSOID_INTERSECTING = 1,
    ELLIPSOID_PROXY_INVALID = 2,
    ELLIPSOID_MAX_ITERATIONS = 3,
    ELLIPSOID_INVALID_GEOMETRY = 4,
    ELLIPSOID_NUMERICAL_FAILURE = 5
};

struct EllipsoidContactResult {
    EllipsoidContactStatus status;
    dfloat signedDisplacement;
    dfloat3 pointA;
    // Point on the selected periodic image of particle B.
    dfloat3 pointBImage;
    // Contact normal from B's selected image toward A.
    dfloat3 normalBToA;
    dfloat curvatureRadiusA;
    dfloat curvatureRadiusB;
    int iterations;
};

struct EllipsoidWallContactResult {
    EllipsoidContactStatus status;
    // Positive when separated, zero at tangency, negative when penetrating.
    dfloat signedDistance;
    dfloat3 pointEllipsoid;
    dfloat3 pointWall;
    dfloat3 wallNormal;
    dfloat curvatureRadius;
};

// ****************************************************************************
// ************************   FORCE COMPUTATION   *****************************
// ****************************************************************************

/**
 * @brief Compute the normal force during a collision.
 * @param n: The normal vector at the point of contact.
 * @param G: The relative velocity vector at the contact point.
 * @param displacement: The displacement value representing the overlap or penetration depth.
 * @param stiffness: The stiffness coefficient for the normal force calculation.
 * @param damping: The damping coefficient for the normal force calculation.
 * @return The computed normal force vector.
 */
__device__ 
dfloat3 computeNormalForce(const dfloat3& n, const dfloat3& G, dfloat displacement, dfloat stiffness, dfloat damping); 

/**
 * @brief Compute the tangential force during a collision, updating tangential displacement if slip occurs.
 * @param tang_disp: The current tangential displacement vector, which will be updated if slip occurs.
 * @param G_ct: The relative tangential velocity vector at the contact point.
 * @param stiffness: The stiffness coefficient for the tangential force calculation.
 * @param damping: The damping coefficient for the tangential force calculation.
 * @param friction_coef: The coefficient of friction between the colliding bodies.
 * @param f_n: The normal force magnitude.
 * @param t: The tangential direction vector at the contact point.
 * @param pc_i: Pointer to the ParticleCenter structure representing the particle.
 * @param tang_index: The index of the tangential displacement record for the collision.
 * @param step: The current simulation step or time index.
 * @return The computed tangential force vector.
 */
__device__ 
dfloat3 computeTangentialForce(
    dfloat3& tang_disp, // will be updated if slip occurs
    const dfloat3& G_ct,
    dfloat stiffness,
    dfloat damping,
    dfloat friction_coef,
    dfloat f_n,
    const dfloat3& t,
    ParticleCenter* pc_i,
    int tang_index,
    int step
); 

/**
 * @brief Accumulate forces and torques on a particle atomically.
 * @param pc_i: Pointer to the ParticleCenter structure representing the particle.
 * @param f_dirs: The force vector to be accumulated.
 * @param m_dirs: The torque vector to be accumulated.
 */
__device__ 
void accumulateForceAndTorque(ParticleCenter* pc_i, const dfloat3& f_dirs, const dfloat3& m_dirs);

// ****************************************************************************
// ************************   COLLISION TRACKING   ****************************
// ****************************************************************************

/**
 *  @brief Calculate the index for wall collisions based on the normal vector.
 *  @param n: The normal vector of the wall.
 *  @return The calculated index used to identify wall collisions.
 */
__device__ 
int calculateWallIndex(const dfloat3 &n);
/**
 *  @brief Find the index of the collision record for a given partnerID.
 *  @param collisionData: The data structure containing collision information.
 *  @param partnerID: The ID of the collision partner to search for.
 *  @param currentTimeStep: The current time step to check collision validity.
 *  @return The index of the collision record if found, otherwise -1.
 */
__device__ 
int getCollisionIndexByPartnerID(const CollisionData &collisionData, int partnerID, int currentTimeStep);
/**
 *  @brief Start a new collision record for a given partnerID.
 *  @param collisionData: The data structure to store collision information.
 *  @param partnerID: The ID of the collision partner.
 *  @param isWall: Boolean indicating whether the collision involves a wall.
 *  @param wallNormal: The normal vector of the wall (if isWall is true).
 *  @param currentTimeStep: The current time step to record the collision.
 *  @return The index of the newly created collision record or -1 if none available.
 */
__device__ 
int startCollision(CollisionData &collisionData, int partnerID, bool isWall, const dfloat3 &wallNormal, int currentTimeStep);
/**
 *  @brief Update the tangential displacement and collision step time for a collision record.
 *  @param collisionData: The data structure containing collision information.
 *  @param index: The index of the collision record to update.
 *  @param displacement: The displacement to add to the tangential displacement.
 *  @param n: The current contact normal, used to re-project the accumulated displacement
 *            back onto the tangent plane (the normal migrates step to step, so a pure
 *            running sum otherwise slowly acquires a spurious normal-direction component).
 *  @param currentTimeStep: The current time step to update the last collision step.
 *  @return The updated tangential displacement for the collision record.
 */
__device__ 
dfloat3 updateTangentialDisplacement(CollisionData &collisionData, int index, const dfloat3 &displacement, const dfloat3 &n, int currentTimeStep);
/**
 *  @brief Zero out the stored tangential (elastic) displacement for a collision record.
 *         Used when slip occurs, so the spring actually releases instead of retaining
 *         its previous accumulated value.
 *  @param collisionData: The data structure containing collision information.
 *  @param index: The index of the collision record to reset.
 *  @param currentTimeStep: The current time step to update the last collision step.
 *  @return The reset (zero) tangential displacement.
 */
__device__
dfloat3 resetTangentialDisplacement(CollisionData &collisionData, int index, int currentTimeStep);
/**
 *  @brief End a collision record and reset its data if necessary.
 *  @param collisionData: The data structure containing collision information.
 *  @param index: The index of the collision record to end.
 *  @param currentTimeStep: The current time step to check the validity of ending the collision.
 */
__device__ 
void endCollision(CollisionData &collisionData, int index, int currentTimeStep);


/**
 * @brief Retrieve or update the tangential displacement for a collision, starting a new collision if necessary.
 * @param pc_i: Pointer to the ParticleCenter structure representing the particle.
 * @param identifier: The ID of the collision partner (another particle or wall).
 * @param isWall: Boolean indicating whether the collision involves a wall.
 * @param step: The current simulation step or time index.
 * @param G_ct: The relative tangential velocity vector at the contact point.
 * @param G: The relative velocity vector at the contact point.
 * @param tang_index_out: Reference to an integer to store the index of the tangential displacement record.
 * @param wallNormal: The normal vector of the wall (only used if isWall is true), or a hack-encoded
 *                    value used by ellipsoidCylinderCollision to force a specific tang_index.
 * @param n: The actual current contact normal, used to re-project the accumulated tangential
 *           displacement onto the current tangent plane before it is used/stored.
 * @return The tangential displacement vector associated with the collision.
 */
__device__ 
dfloat3 getOrUpdateTangentialDisplacement(
    ParticleCenter* pc_i,
    int identifier, // wall index or partner ID
    bool isWall, //true if wall, false if particle-particle collision
    int step,
    const dfloat3& G_ct,
    const dfloat3& G,
    int& tang_index_out,
    const dfloat3& wallNormal = dfloat3{0,0,0}, // only used for wall
    const dfloat3& n = dfloat3{0,0,0}           // contact normal, for tangent-plane projection
);

// ****************************************************************************
// ****************************   WALL COLLISION   ****************************
// ****************************************************************************



/**
 *  @brief Handle collision mechanics between a sphere and a wall.
 *  @param pc_i: Pointer to the `ParticleCenter` structure containing sphere information.
 *  @param wallData: The data structure representing the wall.
 *  @param displacement: The displacement value representing how far the sphere has moved.
 *  @param step: The current time step for collision processing.
 */
__device__
void sphereWallCollision(const CollisionContext& ctx, ParticleWallForces *d_pwForces);

/**
 *  @brief Handle collision mechanics between a capsule's end cap and a wall.
 *  @param pc_i: Pointer to the `ParticleCenter` structure containing capsule information.
 *  @param wallData: The data structure representing the wall.
 *  @param displacement: The displacement value for the capsule's end cap.
 *  @param step: The current time step for collision processing.
 *  @param endpoint: The endpoint of the capsule's end cap.
 */
__device__
void capsuleWallCollisionCap(const CollisionContext& ctx);

/**
 *  @brief Handle collision mechanics between an ellipsoid particle and a wall.
 *  @param pc_i: Pointer to the ParticleCenter structure representing the ellipsoid particle.
 *  @param wallData: Structure containing information about the wall, such as normal and distance.
 *  @param displacement: The displacement value 
 *  @param endpoint: The point on the ellipsoid's surface where the collision occurs.
 *  @param cr: gaussian radius on the contact point
 *  @param step: The current simulation step or time index.
 */
__device__
void ellipsoidWallCollision(const CollisionContext& ctx, dfloat cr[1]);

/**
 * @brief Compute the exact normal gap between an ellipsoid and an axis-aligned
 * wall using the ellipsoid support point.
 * @return Signed distance, ellipsoid surface point, its projection on the wall,
 * inward wall normal, curvature, and an explicit status.
 */
__device__
EllipsoidWallContactResult ellipsoidWallCollisionDistance(
    ParticleCenter* pc_i,
    Wall wallData,
    unsigned int step
);

// ****************************************************************************
// ************************   PARTICLE COLLISION   ****************************
// ****************************************************************************

/**
 *  @brief Handle collision mechanics between two spheres.
 *  @param column: The column index in a grid or matrix representing the particles' positions.
 *  @param row: The row index in a grid or matrix representing the particles' positions.
 *  @param pc_i: Pointer to the `ParticleCenter` structure containing information about the first sphere.
 *  @param pc_j: Pointer to the `ParticleCenter` structure containing information about the second sphere.
 *  @param step: The current time step for collision processing.
 */
__device__
void sphereSphereCollision(const CollisionContext& ctx);

/**
 *  @brief Handle collision mechanics between two capsules.
 *  @param column: The column index in a grid or matrix representing the particles' positions.
 *  @param row: The row index in a grid or matrix representing the particles' positions.
 *  @param pc_i: Pointer to the `ParticleCenter` structure containing information about the first capsule.
 *  @param pc_j: Pointer to the `ParticleCenter` structure containing information about the second capsule.
 *  @param closestOnA: Closest point in the axis of particle i.
 *  @param closestOnB: Closest point in the axis of particle j.
 *  @param step: The current time step for collision processing.
 */
__device__
void capsuleCapsuleCollision(const CollisionContext& ctx, dfloat3 closestOnA[1], dfloat3 closestOnB[1]) ;

/**
 *  @brief Process the collision between two ellipsoids by determining their closest points and applying collision response.
 *  @param pc_i: Pointer to the `ParticleCenter` structure containing information about the first ellipsoid.
 *  @param pc_j: Pointer to the `ParticleCenter` structure containing information about the second ellipsoid.
 *  @param closestOnA: Array to store the closest point on the surface of the first ellipsoid (A).
 *  @param closestOnB: Array to store the closest point on the surface of the second ellipsoid (B).
 *  @param dist: The calculated distance between the two ellipsoids at their closest points.
 *  @param cr1: gaussian radius on the contact point for ellipsoid 1
 *  @param cr2: gaussian radius on the contact point for ellipsoid 2
 *  @param step: The current simulation time step for collision processing.
 */
__device__
void ellipsoidEllipsoidCollision(const CollisionContext& ctx,dfloat3 closestOnA[1], dfloat3 closestOnB[1],dfloat cr1[1], dfloat cr2[1], dfloat3 translation);

/**
 * @brief Handle collision mechanics between an ellipsoid and a cylinder.
 * @param pc_i: Pointer to the ParticleCenter structure representing the ellipsoid particle.
 * @param closestOnB: Closest point on the cylinder surface.
 * @param cr1: gaussian radius on the contact point for ellipsoid 1
 * @param P1: One endpoint of the cylinder axis.
 * @param P2: The other endpoint of the cylinder axis.
 * @param cRadius: The radius of the cylinder.
 * @param cyDir: The direction of the cylinder surface normal
 */
__device__
void ellipsoidCylinderCollision(const CollisionContext& ctx, dfloat3 closestOnB[1],dfloat cr1[1], dfloat3 P1, dfloat3 P2, dfloat cRadius, int cyDir); 

// ****************************************************************************
// ******************   AUXILIARY COLLISION FUNCTIONS  ************************
// ****************************************************************************

/**
 *  @brief Compute the intersection point between a line and the ellipsoid.
 *  @param pc_i: Pointer to the ParticleCenter structure representing the ellipsoid particle.
 *  @param R: 3x3 rotation matrix used to transform the ellipsoid's orientation.
 *  @param line_origin: The origin of the line used for intersection calculation.
 *  @param line_dir: The direction of the line used for intersection calculation.
 *  @return The intersection parameter between the line and the ellipsoid on .x and .y; ;.z is trash
 */
__device__
dfloat3 ellipsoid_intersection(ParticleCenter* pc_i, dfloat R[3][3],dfloat3 line_origin, dfloat3 line_dir,dfloat3 translation);

/**
 *  @brief Compute the normal vector at a given point on the ellipsoid's surface.
 *  @param pc_i: Pointer to the ParticleCenter structure representing the ellipsoid particle.
 *  @param R: 3x3 rotation matrix used to transform the ellipsoid's orientation.
 *  @param point: The point on the ellipsoid's surface where the normal vector is computed.
 *  @param radius: gaussian radius on the point
 *  @return The normal vector at the specified point on the ellipsoid's surface.
 */
__device__
dfloat3 ellipsoid_normal(ParticleCenter* pc_i, dfloat R[3][3],dfloat3 point, dfloat radius[1],dfloat3 translation);

/**
 * @brief Solve one periodic image of an ellipsoid pair.
 * @param pc_i: Pointer to the ParticleCenter structure representing the first ellipsoid particle.
 * @param pc_j: Pointer to the ParticleCenter structure representing the second ellipsoid particle.
 * @param translation: The translation vector to apply to the second ellipsoid for periodic boundary conditions.
 * @param step: The current simulation time step for collision processing.
 * @return Explicit status, image-space contact points, B-to-A normal, and a
 * signed displacement. Separation is positive; a valid fixed-sphere
 * penetration proxy is negative.
 */
__device__
EllipsoidContactResult ellipsoidEllipsoidCollisionDistance(
    ParticleCenter* pc_i,
    ParticleCenter* pc_j,
    dfloat3 translation,
    unsigned int step
);

#endif //PARTICLE_MODEL
#endif // !__IBM_COLLISION_H

