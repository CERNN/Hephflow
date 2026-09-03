//functions that determine HOW colide
#include "collision.cuh"

#ifdef PARTICLE_MODEL

// ****************************************************************************
// ************************   FORCE COMPUTATION   *****************************
// ****************************************************************************

__device__ 
dfloat3 computeNormalForce(const dfloat3& n, const dfloat3& G, dfloat displacement, dfloat stiffness, dfloat damping) {
    dfloat f_kn = -stiffness * sqrt(abs(displacement*displacement*displacement));
    return f_kn * n - damping * dot_product(G, n) * n * POW_FUNCTION(abs(displacement), 0.25);
}

__device__ dfloat3 computeTangentialForce(
    dfloat3& tang_disp, // will be updated if slip occurs
    const dfloat3& G_ct,
    dfloat stiffness,
    dfloat damping,
    dfloat friction_coef,
    dfloat f_n,
    const dfloat3& n,
    ParticleCenter* pc_i,
    int tang_index,
    int step
) {
    // DAMPING_TANGENTIAL is formed from sqrt(m_eff*k_t), and k_t already
    // contains sqrt(normal overlap). It therefore already has the required
    // Hertz overlap dependence. Component-wise powers of tangential history
    // made the old force anisotropic and impossible to back-project exactly.
    const dfloat3 damping_force = -damping * G_ct;
    dfloat3 f_tang = -stiffness * tang_disp + damping_force;
    const dfloat mag = vector_length(f_tang);
    const dfloat coulomb_limit = friction_coef * fabsf(f_n);
    if (mag > coulomb_limit && mag > 0.0f) {
        // Scale the complete trial force. This remains well-defined if the
        // instantaneous tangential velocity is zero but the spring is loaded.
        f_tang = (coulomb_limit / mag) * f_tang;

        // Store the spring state that reproduces the capped total force:
        // F_cap = -k_t*xi_corrected + F_damping.
        if (stiffness > 0.0f) {
            tang_disp = (damping_force - f_tang) / stiffness;
            tang_disp = tang_disp - dot_product(tang_disp, n) * n;
        } else {
            tang_disp = dfloat3(0.0f, 0.0f, 0.0f);
        }
        if (tang_index >= 0) {
            pc_i->getCollision().setTangentialDisplacement(tang_index, tang_disp);
            pc_i->getCollision().setLastCollisionStep(tang_index, step);
        }
    }
    return f_tang;
}

__device__ void accumulateForceAndTorque(
    ParticleCenter* pc_i,
    const dfloat3& f_dirs,
    const dfloat3& m_dirs) {
    atomicAdd(&(pc_i->getFXatomic()), f_dirs.x);
    atomicAdd(&(pc_i->getFYatomic()), f_dirs.y);
    atomicAdd(&(pc_i->getFZatomic()), f_dirs.z);
    atomicAdd(&(pc_i->getMXatomic()), m_dirs.x);
    atomicAdd(&(pc_i->getMYatomic()), m_dirs.y);
    atomicAdd(&(pc_i->getMZatomic()), m_dirs.z);
}



// ****************************************************************************
// ************************   COLLISION TRACKING   ****************************
// ****************************************************************************

__device__ 
int calculateWallIndex(const dfloat3 &n) {
    // Calculate the index based on the normal vector

    /*
    n.x n.y n.z index
    1   0   0   2
    -1  0   0   4
    0   1   0   5
    0   -1  0   1
    0   0   1   6
    0   0   -1  0
    0   0   0   3 -> external duct, since normal will be reduced to 0,0,0 when convert to integer
    */

    return 7 + (1 - (int)n.x) - 2 * (1 - (int)n.y) - 3 * (1 - (int)n.z);
}

__device__ 
int getCollisionIndexByPartnerID(const CollisionData &collisionData, int partnerID, int currentTimeStep) {
    for (int i = FIRST_PARTICLE_COLLISION_SLOT; i < MAX_ACTIVE_COLLISIONS ; i++) {
        if (collisionData.getCollisionPartnerID(i) == partnerID &&
            currentTimeStep - collisionData.getLastCollisionStep(i) <= 1) {
            return i; // Found the collision index for a particle
        }
    }
    // If no match is found, return -1
    return -1;
}

__device__ 
int startCollision(CollisionData &collisionData, int partnerID, bool isWall, const dfloat3 &wallNormal, int currentTimeStep) {
    int index = -1;
    if (isWall) {
        index = calculateWallIndex(wallNormal);
        // Initialize wall collision data
        collisionData.setTangentialDisplacement(index, {0.0, 0.0, 0.0});
        collisionData.setLastCollisionStep(index, currentTimeStep);
    } else {
        index = collisionData.claimParticleCollisionSlot(partnerID, currentTimeStep);
    }

    return index;
}

__device__ 
dfloat3 updateTangentialDisplacement(CollisionData &collisionData, int index, const dfloat3 &displacement, const dfloat3 &n, int currentTimeStep) {
    dfloat3 new_disp = collisionData.getTangentialDisplacement(index) + displacement;
    // Re-project onto the current tangent plane. The contact normal migrates
    // step to step (especially for ellipsoids), so a pure running sum slowly
    // acquires a spurious normal-direction component that leaks energy into
    // the normal force. Strip it back out every update.
    new_disp = new_disp - dot_product(new_disp, n) * n;
    collisionData.setTangentialDisplacement(index, new_disp);
    collisionData.setLastCollisionStep(index, currentTimeStep);
    return new_disp;
}

__device__
dfloat3 resetTangentialDisplacement(CollisionData &collisionData, int index, int currentTimeStep) {
    // Actually zero the stored spring, unlike calling updateTangentialDisplacement
    // with a zero delta (which just adds 0 to whatever was already stored).
    collisionData.setTangentialDisplacement(index, dfloat3{0.0, 0.0, 0.0});
    collisionData.setLastCollisionStep(index, currentTimeStep);
    return dfloat3{0.0, 0.0, 0.0};
}

__device__ 
void endCollision(CollisionData &collisionData, int index, int currentTimeStep) {
    // Check if index is valid for wall collisions (0 to 6)
    if (index >= 0 && index <= 6) {
        collisionData.setLastCollisionStep(index, -1);
    }
    // Check if index is valid for particle collisions 
    else if (index >= FIRST_PARTICLE_COLLISION_SLOT && index < MAX_ACTIVE_COLLISIONS ) {
        if (currentTimeStep - collisionData.getLastCollisionStep(index) > 1) {
            collisionData.setTangentialDisplacement(index, {0.0, 0.0, 0.0});
            collisionData.setLastCollisionStep(index, -1); // Indicate the slot is available '
            collisionData.setCollisionPartnerID(index, -1); // Optionally reset
        }
    }
}


__device__ 
dfloat3 getOrUpdateTangentialDisplacement(
    ParticleCenter* pc_i,
    int identifier,          // wall index or partner ID
    bool isWall,             // true if wall, false if particle-particle collision
    int step,
    const dfloat3& G_ct,
    const dfloat3& G_cn,
    int& tang_index_out,
    const dfloat3& wallNormal, // only used for wall index calc / hack encoding
    const dfloat3& n           // actual contact normal, used for tangent-plane projection
) {
    dfloat3 tang_disp;
    int tang_index = -1;
    
    //get the collision index 
    if (isWall) {
        tang_index = calculateWallIndex(wallNormal);
    } else {
        tang_index = getCollisionIndexByPartnerID(pc_i->getCollision(), identifier, step);
    }

    //check if there is a current collision
    if (tang_index == -1) {
        // NEW COLLISION: Initialize with zero displacement
        tang_index = startCollision(pc_i->getCollision(), identifier, isWall, wallNormal, step);
        tang_disp = dfloat3{0.0, 0.0, 0.0};
    } else {
        //check if the collision already exited in the past
        if (step - pc_i->getCollision().getLastCollisionStep(tang_index) > 1) {
            // COLLISION ENDED: Clean up old data
            endCollision(pc_i->getCollision(), tang_index, step);
            // START NEW COLLISION with fresh tracking
            tang_index = startCollision(pc_i->getCollision(), identifier, isWall, wallNormal, step);
            tang_disp = dfloat3{0.0, 0.0, 0.0};
        } else {
            // ONGOING COLLISION: Accumulate tangential displacement (with projection)
            tang_disp = updateTangentialDisplacement(pc_i->getCollision(), tang_index, G_ct, n, step);
        }
    }
    //return the index by address
    tang_index_out = tang_index;
    //return the tangential displacement
    return tang_disp;
}


// ****************************************************************************
// **************************   WALL COLLISION   ******************************
// ****************************************************************************

// -------------------------- SPHERE COLLISION --------------------------------
__device__
void sphereWallCollision(const CollisionContext& ctx, ParticleWallForces *d_pwForces){

    ParticleCenter* pc_i = ctx.pc_i;
    Wall wallData = ctx.wall;
    dfloat displacement = ctx.displacement;
    int step = ctx.step;

    // Particle info
    const dfloat m_i = pc_i->getVolume() * pc_i->getDensity();
    const dfloat r_i = pc_i->getRadius();
    const dfloat3 v_i = pc_i->getVel();
    const dfloat3 w_i = pc_i->getW();

    // Wall info
    dfloat3 wall_speed = wallData.velocity;
    dfloat3 n = wallData.normal;
    n = n * -1.0f; //invert collision direction since is from sphere to wall

    // Relative velocity
    dfloat3 G = v_i - wall_speed;

    //Collision constants
    const dfloat STIFFNESS_NORMAL = SPHERE_WALL_STIFFNESS_NORMAL_CONST * sqrt(abs(r_i));
    const dfloat STIFFNESS_TANGENTIAL = SPHERE_WALL_STIFFNESS_TANGENTIAL_CONST * sqrt(r_i) * sqrt(abs(displacement));
    const dfloat DAMPING_NORMAL = SPHERE_WALL_DAMPING_CONST * sqrt(m_i * STIFFNESS_NORMAL);
    const dfloat DAMPING_TANGENTIAL = SPHERE_WALL_DAMPING_CONST * sqrt(m_i * STIFFNESS_TANGENTIAL);

    // Normal force
    dfloat3 f_normal = computeNormalForce(n, G, displacement, STIFFNESS_NORMAL, DAMPING_NORMAL);
    dfloat f_n = vector_length(f_normal);

    // Relative tangential velocity
    dfloat3 G_ct = G + r_i * cross_product(w_i, n) - dot_product(G, n) * n;


    //retrive and update tangential displacement
    int tang_index = calculateWallIndex(n); //wall can be directly computed
    dfloat3 tang_disp = getOrUpdateTangentialDisplacement(pc_i, 0, true, step, G_ct, dot_product(G,n), tang_index, n, n);

    // Compute tangential force
    dfloat3 f_tang = computeTangentialForce(
        tang_disp, G_ct, STIFFNESS_TANGENTIAL, DAMPING_TANGENTIAL,
        PW_FRICTION_COEF, f_n, n, pc_i, tang_index, step
    );

    //Total forces and torque in the particle
    dfloat3 f_dirs = f_normal + f_tang;
    dfloat3 m_dirs = r_i * cross_product(n, f_tang);

    //Save data in the particle information
    accumulateForceAndTorque(pc_i, f_dirs, m_dirs);

    // Force exerted by particle on wall
    dfloat3 f_on_wall = -f_dirs;

    // Magnitudes
    dfloat Fn = vector_length(f_normal);
    dfloat Ft = vector_length(f_tang);

    // Select wall accumulator
    ParticleWallForce* pwf = nullptr;

    if (wallData.normal.y > 0) {
        pwf = &d_pwForces->bottom;
    } else if (wallData.normal.y < 0) {
        pwf = &d_pwForces->top;
    }

    if (pwf) {
        atomicAdd(&pwf->Fx, f_on_wall.x);
        atomicAdd(&pwf->Fy, f_on_wall.y);
        atomicAdd(&pwf->Fz, f_on_wall.z);

        atomicAdd(&pwf->Fn, Fn);
        atomicAdd(&pwf->Ft, Ft);

        atomicAdd(&pwf->nContacts, 1);
    }
}

// ------------------------- CAPSULE COLLISIONS -------------------------------
__device__
void capsuleWallCollisionCap(const CollisionContext& ctx) {
    ParticleCenter* pc_i = ctx.pc_i;
    Wall wallData = ctx.wall;
    dfloat displacement = ctx.displacement;
    int step = ctx.step;
    dfloat3 endpoint = ctx.contactPoint;
    // Particle info
    const dfloat3 center_pos = pc_i->getPos();      // Particle center position
    const dfloat3 cap_pos = endpoint;               // Capsule cap position
    const dfloat m_i = pc_i->getVolume() * pc_i->getDensity();
    const dfloat r_i = pc_i->getRadius();
    const dfloat3 v_i = pc_i->getVel();             // Center of mass velocity
    const dfloat3 w_i = pc_i->getW();
    // Wall info
    dfloat3 wall_speed = dfloat3(0, 0, 0);          // Wall velocity (assumed zero)
    dfloat3 n = wallData.normal * -1.0f;            // Invert normal: collision from sphere to wall

    // Relative velocity
    dfloat3 rri = cap_pos - center_pos; 
    dfloat3 G = (v_i + cross_product(w_i,r_i*n + rri)) - (wall_speed);
    dfloat3 G_cn = dot_product(G,n) * n;
    dfloat3 G_ct = G - G_cn;

    //Collision constants
    const dfloat STIFFNESS_NORMAL = SPHERE_WALL_STIFFNESS_NORMAL_CONST * sqrt(abs(r_i));
    const dfloat STIFFNESS_TANGENTIAL = SPHERE_WALL_STIFFNESS_TANGENTIAL_CONST * sqrt(r_i) * sqrt(abs(displacement));
    const dfloat DAMPING_NORMAL = SPHERE_WALL_DAMPING_CONST * sqrt(m_i * STIFFNESS_NORMAL);
    const dfloat DAMPING_TANGENTIAL = SPHERE_WALL_DAMPING_CONST * sqrt(m_i * STIFFNESS_TANGENTIAL);


    // Normal force
    dfloat3 f_normal = computeNormalForce(n, G, displacement, STIFFNESS_NORMAL, DAMPING_NORMAL);
    dfloat f_n = vector_length(f_normal);

    // Relative tangential velocity

    // Tangential displacement tracking
    //retrive and update tangential displacement
    int tang_index = calculateWallIndex(n); //wall can be directly computed
    dfloat3 tang_disp = getOrUpdateTangentialDisplacement(pc_i, 0, true, step, G_ct, G_cn, tang_index, n, n);


    // Compute tangential force
    dfloat3 f_tang = computeTangentialForce(
        tang_disp, G_ct, STIFFNESS_TANGENTIAL, DAMPING_TANGENTIAL,
        PW_FRICTION_COEF, f_n, n, pc_i, tang_index, step
    );

    //Total forces and torque in the particle
    dfloat3 f_dirs = f_normal + f_tang;
    dfloat3 m_dirs = cross_product((n * r_i) + rri, f_dirs);

    //Save data in the particle information
    accumulateForceAndTorque(pc_i, f_dirs, m_dirs);
}

// ------------------------ ELLIPSOID COLLISIONS ------------------------------
__device__
void ellipsoidWallCollision(const CollisionContext& ctx, dfloat cr[1]) {
    ParticleCenter* pc_i = ctx.pc_i;
    Wall wallData = ctx.wall;
    dfloat displacement = ctx.displacement;
    int step = ctx.step;
    dfloat3 endpoint = ctx.contactPoint;
    // Particle info
    const dfloat3 pos_i = pc_i->getPos(); //center position
    const dfloat3 pos_c_i = endpoint; //contact point + n * radius of contact
    const dfloat m_i = pc_i->getVolume() * pc_i->getDensity();
    const dfloat r_i = pc_i->getRadius(); //TODO: find a way to calculate the correct radius of contact
    const dfloat3 v_i = pc_i->getVel(); //VELOCITY OF THE CENTER OF MASS
    const dfloat3 w_i = pc_i->getW();
    // Wall info
    dfloat3 wall_speed = wallData.velocity;
    dfloat3 n = wallData.normal * -1.0f; //invert collision direction since is from sphere to wall

    //vector center-> contact 
    dfloat3 rri = pos_c_i - pos_i;
    dfloat3 G = (v_i + cross_product(w_i,rri)) - (wall_speed);
    dfloat3 G_cn = dot_product(G,n) * n;
    dfloat3 G_ct = G - G_cn;

    //collision constants
    const dfloat STIFFNESS_NORMAL = SPHERE_WALL_STIFFNESS_NORMAL_CONST * sqrt(abs(cr[0]));
    const dfloat STIFFNESS_TANGENTIAL = SPHERE_WALL_STIFFNESS_TANGENTIAL_CONST * sqrt(cr[0]) * sqrt (abs(displacement));
    const dfloat DAMPING_NORMAL = SPHERE_WALL_DAMPING_CONST * sqrt(m_i * STIFFNESS_NORMAL);
    const dfloat DAMPING_TANGENTIAL = SPHERE_WALL_DAMPING_CONST * sqrt(m_i * STIFFNESS_TANGENTIAL);

    // Normal force
    dfloat3 f_normal = computeNormalForce(n, G, displacement, STIFFNESS_NORMAL, DAMPING_NORMAL);
    dfloat f_n = vector_length(f_normal);

    // Relative tangential velocity


    //retrive and update tangential displacement
    int tang_index = calculateWallIndex(n); //wall can be directly computed
    dfloat3 tang_disp = getOrUpdateTangentialDisplacement(pc_i, 0, true, step, G_ct, G_cn, tang_index, n, n);

    // Compute tangential force
    dfloat3 f_tang = computeTangentialForce(
        tang_disp, G_ct, STIFFNESS_TANGENTIAL, DAMPING_TANGENTIAL,
        PW_FRICTION_COEF, f_n, n, pc_i, tang_index, step
    );

    //sum the forces
    dfloat3 f_dirs = f_normal + f_tang;

    //calculate moments
    dfloat3 m_dirs = cross_product(rri,f_dirs);

    //save date in the particle information
    accumulateForceAndTorque(pc_i, f_dirs, m_dirs);
}

__device__
EllipsoidWallContactResult ellipsoidWallCollisionDistance(
    ParticleCenter* pc_i,
    Wall wallData,
    unsigned int step
) {
    (void)step;
    EllipsoidWallContactResult result = {};
    result.status = ELLIPSOID_INVALID_GEOMETRY;
    result.signedDistance = 1.0e37f;

    const dfloat3 center = pc_i->getPos();
    const dfloat3 axisVector1 = pc_i->getSemiAxis1()-center;
    const dfloat3 axisVector2 = pc_i->getSemiAxis2()-center;
    const dfloat3 axisVector3 = pc_i->getSemiAxis3()-center;
    const dfloat a = vector_length(axisVector1);
    const dfloat b = vector_length(axisVector2);
    const dfloat c = vector_length(axisVector3);
    const dfloat normalLength = vector_length(wallData.normal);
    if (!(a>1.0e-8f && b>1.0e-8f && c>1.0e-8f && normalLength>1.0e-8f)) return result;

    const dfloat3 u1=axisVector1/a;
    const dfloat3 u2=axisVector2/b;
    const dfloat3 u3=axisVector3/c;
    if (fabsf(dot_product(u1,u2))>1.0e-3f ||
        fabsf(dot_product(u1,u3))>1.0e-3f ||
        fabsf(dot_product(u2,u3))>1.0e-3f) return result;

    const dfloat3 n=wallData.normal/normalLength; // inward normal
    const dfloat n1=dot_product(n,u1);
    const dfloat n2=dot_product(n,u2);
    const dfloat n3=dot_product(n,u3);
    const dfloat supportDenom=sqrtf(a*a*n1*n1+b*b*n2*n2+c*c*n3*n3);
    if (!(supportDenom>1.0e-8f) || !isfinite(supportDenom)) {
        result.status=ELLIPSOID_NUMERICAL_FAILURE;
        return result;
    }

    // Exact ellipsoid support point in the direction opposite the inward wall normal.
    const dfloat3 supportNumerator=(a*a*n1)*u1+(b*b*n2)*u2+(c*c*n3)*u3;
    const dfloat3 pointEllipsoid=center-supportNumerator/supportDenom;

    // Existing Wall stores the positive coordinate of an axis-aligned plane.
    // Construct a point on that plane, then use the normal form consistently.
    const dfloat3 pointOnPlane=wallData.distance*dfloat3(fabsf(n.x),fabsf(n.y),fabsf(n.z));
    const dfloat planeConstant=dot_product(n,pointOnPlane);
    const dfloat signedDistance=dot_product(n,pointEllipsoid)-planeConstant;
    const dfloat3 pointWall=pointEllipsoid-signedDistance*n;

    dfloat R[3][3];
    rotationMatrixFromVectors(u1,u2,u3,R);
    dfloat curvature[1];
    ellipsoid_normal(pc_i,R,pointEllipsoid,curvature,dfloat3(0,0,0));
    if (!isfinite(signedDistance) || !isfinite(curvature[0]) || !(curvature[0]>0.0f)) {
        result.status=ELLIPSOID_NUMERICAL_FAILURE;
        return result;
    }

    result.status=signedDistance<0.0f ? ELLIPSOID_INTERSECTING : ELLIPSOID_SEPARATED;
    result.signedDistance=signedDistance;
    result.pointEllipsoid=pointEllipsoid;
    result.pointWall=pointWall;
    result.wallNormal=n;
    result.curvatureRadius=curvature[0];
    return result;
}

// ****************************************************************************
// ************************   PARTICLE COLLISION   ****************************
// ****************************************************************************

// -------------------------- SPHERE COLLISION --------------------------------
__device__
void sphereSphereCollision(const CollisionContext& ctx){
    ParticleCenter* pc_i = ctx.pc_i;
    ParticleCenter* pc_j = ctx.pc_j;
    int step = ctx.step;
    dfloat displacement = ctx.displacement;
    int partnerID = ctx.partnerID;
    // Particle i info
    const dfloat r_i = pc_i->getRadius();
    const dfloat m_i = pc_i->getVolume() * pc_i->getDensity();
    const dfloat3 v_i = pc_i->getVel();
    const dfloat3 w_i = pc_i->getW();
    // Particle j info
    const dfloat r_j = pc_j->getRadius();
    const dfloat m_j = pc_j->getVolume() * pc_j->getDensity();
    const dfloat3 v_j = pc_j->getVel();
    const dfloat3 w_j = pc_j->getW();

    // Use normal direction from context (set in wall.normal)
    const dfloat3 n = ctx.wall.normal;
    // Relative velocity
    dfloat3 G = v_i - v_j;

    // Hertz contact theory
    dfloat effective_radius = 1.0 / ((r_i + r_j) / (r_i * r_j));
    dfloat effective_mass = 1.0 / ((m_i + m_j) / (m_i * m_j));
    const dfloat STIFFNESS_NORMAL = SPHERE_SPHERE_STIFFNESS_NORMAL_CONST * sqrt(effective_radius);
    const dfloat STIFFNESS_TANGENTIAL = SPHERE_SPHERE_STIFFNESS_TANGENTIAL_CONST * sqrt(effective_radius) * sqrt(abs(displacement));
    const dfloat DAMPING_NORMAL = SPHERE_SPHERE_DAMPING_CONST * sqrt(effective_mass * STIFFNESS_NORMAL);
    const dfloat DAMPING_TANGENTIAL = SPHERE_SPHERE_DAMPING_CONST * sqrt(effective_mass * STIFFNESS_TANGENTIAL);

    // Normal force
    dfloat3 f_normal = computeNormalForce(n, G, displacement, STIFFNESS_NORMAL, DAMPING_NORMAL);
    dfloat f_n = vector_length(f_normal);

    // Relative tangential velocity
    dfloat3 G_ct = G + r_i * cross_product(w_i, n) + r_j * cross_product(w_j, n) - dot_product(G, n) * n;

    int tang_index = -1;
    dfloat3 tang_disp = getOrUpdateTangentialDisplacement(pc_i, partnerID, false, step, G_ct, G, tang_index, dfloat3(0, 0, 0), n);

    // Compute tangential force
    dfloat3 f_tang = computeTangentialForce(
        tang_disp, G_ct, STIFFNESS_TANGENTIAL, DAMPING_TANGENTIAL,
        PP_FRICTION_COEF, f_n, n, pc_i, tang_index, step
    );

    // Final force results
    dfloat3 f_dirs = f_normal + f_tang;
    dfloat3 m_dirs_i = r_i * cross_product(n, f_tang);
    dfloat3 m_dirs_j = r_j * cross_product(n, f_tang);

    //Save data in the particle information
    accumulateForceAndTorque(pc_i, -f_dirs, m_dirs_i);
    accumulateForceAndTorque(pc_j,  f_dirs, m_dirs_j);
}

// ------------------------- CAPSULE COLLISIONS -------------------------------
__device__
void capsuleCapsuleCollision(const CollisionContext& ctx, dfloat3 closestOnA[1], dfloat3 closestOnB[1]) {
    ParticleCenter* pc_i = ctx.pc_i;
    ParticleCenter* pc_j = ctx.pc_j;
    int step = ctx.step;
    int partnerID = ctx.partnerID;
    dfloat displacement = ctx.displacement;
    // Contact points (capsule ends)
    dfloat3 pos_i = closestOnA[0]; // closest point on capsule i
    dfloat3 pos_c_i = pc_i->getPos();
    dfloat3 pos_j = closestOnB[0]; // closest point on capsule j
    dfloat3 pos_c_j = pc_j->getPos();
    
    // Particle info
    const dfloat r_i = pc_i->getRadius();
    const dfloat m_i = pc_i->getVolume() * pc_i->getDensity();
    const dfloat3 v_i = pc_i->getVel();
    const dfloat3 w_i = pc_i->getW();
    const dfloat r_j = pc_j->getRadius();
    const dfloat m_j = pc_j->getVolume() * pc_j->getDensity();
    const dfloat3 v_j = pc_j->getVel();
    const dfloat3 w_j = pc_j->getW();

    // Use normal direction from context (set in wall.normal)
    const dfloat3 n = ctx.wall.normal;
    // Relative velocity
    dfloat3 G = v_i - v_j;   

    // Hertz contact theory
    dfloat effective_radius = 1.0 / ((r_i + r_j) / (r_i * r_j));
    dfloat effective_mass = 1.0 / ((m_i + m_j) / (m_i * m_j));
    const dfloat STIFFNESS_NORMAL = SPHERE_SPHERE_STIFFNESS_NORMAL_CONST * sqrt(effective_radius);
    const dfloat STIFFNESS_TANGENTIAL = SPHERE_SPHERE_STIFFNESS_TANGENTIAL_CONST * sqrt(effective_radius) * sqrt(abs(displacement));
    const dfloat DAMPING_NORMAL = SPHERE_SPHERE_DAMPING_CONST * sqrt(effective_mass * STIFFNESS_NORMAL);
    const dfloat DAMPING_TANGENTIAL = SPHERE_SPHERE_DAMPING_CONST * sqrt(effective_mass * STIFFNESS_TANGENTIAL);

    // Normal force
    dfloat3 f_normal = computeNormalForce(n, G, displacement, STIFFNESS_NORMAL, DAMPING_NORMAL);
    dfloat f_n = vector_length(f_normal);

    // Relative tangential velocity
    dfloat3 G_ct = G + r_i * cross_product(w_i, n) + r_j * cross_product(w_j, n) - dot_product(G, n) * n;

    int tang_index = -1;
    dfloat3 tang_disp = getOrUpdateTangentialDisplacement(pc_i, partnerID, false, step, G_ct, dot_product(G,n), tang_index, dfloat3(0, 0, 0), n);

    // Compute tangential force
    dfloat3 f_tang = computeTangentialForce(
        tang_disp, G_ct, STIFFNESS_TANGENTIAL, DAMPING_TANGENTIAL,
        PP_FRICTION_COEF, f_n, n, pc_i, tang_index, step
    );

    // Final force results
    dfloat3 f_dirs = f_normal + f_tang;
    dfloat3 m_dirs_i = cross_product((pos_i - pos_c_i) + (-n * r_i), -f_dirs);
    dfloat3 m_dirs_j = cross_product((pos_j - pos_c_j) + (n * r_j), f_dirs);

    //Save data in the particle information
    accumulateForceAndTorque(pc_i, -f_dirs, m_dirs_i);
    accumulateForceAndTorque(pc_j,  f_dirs, m_dirs_j);
}

// ------------------------ ELLIPSOID COLLISIONS ------------------------------
__device__
void ellipsoidEllipsoidCollision(const CollisionContext& ctx,dfloat3 closestOnA[1], dfloat3 closestOnB[1],dfloat cr1[1], dfloat cr2[1], dfloat3 translation) {
    ParticleCenter* pc_i = ctx.pc_i;
    ParticleCenter* pc_j = ctx.pc_j;
    int step = ctx.step;
    int partnerID = ctx.partnerID;
    dfloat displacement = ctx.displacement;
    // Contact points
    dfloat3 pos_i = closestOnA[0]; // closest point on ellipsoid i
    dfloat3 pos_c_i = pc_i->getPos();
    dfloat3 pos_j = closestOnB[0] + translation; // closest point on ellipsoid j
    dfloat3 pos_c_j = pc_j->getPos() + translation;

    // Particle info
    const dfloat r_i = pc_i->getRadius();
    const dfloat m_i = pc_i->getVolume() * pc_i->getDensity();
    const dfloat3 v_i = pc_i->getVel();
    const dfloat3 w_i = pc_i->getW();
    const dfloat r_j = pc_j->getRadius();
    const dfloat m_j = pc_j->getVolume() * pc_j->getDensity();
    const dfloat3 v_j = pc_j->getVel();
    const dfloat3 w_j = pc_j->getW();

    // Use normal direction from context (set in wall.normal)
    const dfloat3 n = ctx.wall.normal;
    // Relative velocity
    dfloat3 rri = pos_i - pos_c_i;
    dfloat3 rrj = pos_j - pos_c_j;
    dfloat3 G = (v_i + cross_product(w_i,rri)) - (v_j + cross_product(w_j,rrj));
    dfloat3 G_cn = dot_product(G,n) * n;
    dfloat3 G_ct = G - G_cn;

    // Hertz contact theory
    dfloat effective_radius = 1.0 / ((cr1[0] + cr2[0]) / (cr1[0] * cr2[0]));
    dfloat effective_mass = 1.0 / ((m_i + m_j) / (m_i * m_j));
    const dfloat STIFFNESS_NORMAL = SPHERE_SPHERE_STIFFNESS_NORMAL_CONST * sqrt(effective_radius);
    const dfloat STIFFNESS_TANGENTIAL = SPHERE_SPHERE_STIFFNESS_TANGENTIAL_CONST * sqrt(effective_radius) * sqrt(abs(displacement));

    const dfloat DAMPING_NORMAL = SPHERE_SPHERE_DAMPING_CONST * sqrt(effective_mass * STIFFNESS_NORMAL);
    const dfloat DAMPING_TANGENTIAL = SPHERE_SPHERE_DAMPING_CONST * sqrt(effective_mass * STIFFNESS_TANGENTIAL);

    // Normal force
    dfloat3 f_normal = computeNormalForce(n, G, displacement, STIFFNESS_NORMAL, DAMPING_NORMAL);
    dfloat f_n = vector_length(f_normal);

    // Relative tangential velocity

    int tang_index = -1;
    dfloat3 tang_disp = getOrUpdateTangentialDisplacement(pc_i, partnerID, false, step, G_ct, G_cn, tang_index, dfloat3(0, 0, 0), n);

    dfloat3 f_tang = computeTangentialForce(
        tang_disp, G_ct, STIFFNESS_TANGENTIAL, DAMPING_TANGENTIAL,
        PP_FRICTION_COEF, f_n, n, pc_i, tang_index, step
    );

    // Final force results
    dfloat3 f_dirs = f_normal + f_tang;
    dfloat3 m_dirs_i = cross_product(rri, -f_dirs);
    dfloat3 m_dirs_j = cross_product(rrj,  f_dirs);

    //Save data in the particle information
    accumulateForceAndTorque(pc_i, -f_dirs, m_dirs_i);
    accumulateForceAndTorque(pc_j,  f_dirs, m_dirs_j);
}


__device__
void ellipsoidCylinderCollision(const CollisionContext& ctx, dfloat3 closestOnB[1],dfloat cr1[1], dfloat3 P1, dfloat3 P2, dfloat cRadius, int cyDir) {
    ParticleCenter* pc_i = ctx.pc_i;
    int step = ctx.step;
    dfloat displacement = ctx.displacement;
    dfloat3 endpoint = closestOnB[0];

    dfloat3 pos_i = pc_i->getPos(); 
    dfloat3 pos_c_i = endpoint;
    const dfloat3 v_i = pc_i->getVel();
    const dfloat3 w_i = pc_i->getW();
    const dfloat r_i = pc_i->getRadius();
    const dfloat m_i = pc_i->getVolume() * pc_i->getDensity();


    //wall information        
    dfloat3 wall_speed = dfloat3(0,0,0); 
    dfloat3 wall_rotation = dfloat3(0,0,0);

    dfloat3 proj = segmentProjection(endpoint,P1,P2,cRadius,cyDir);
    dfloat3 dir = pc_i->getPos() - proj;
    dfloat3 n = vector_normalize(proj - endpoint);

    //invert collision direction since is from sphere to wall
    n = -n;

    //vector center-> contact 
    dfloat3 rri = pos_c_i - pos_i;

    dfloat3 G = (v_i + cross_product(w_i,rri)) - (wall_speed + cross_product(wall_rotation,cRadius));
    dfloat3 G_cn = dot_product(G,n) * n;
    dfloat3 G_ct = G - G_cn;

    dfloat effective_radius = 1.0 / ((cr1[0] + cRadius) / (cr1[0] * cRadius));
    dfloat effective_mass = 1.0 / (m_i); //wall has infinite mass

    const dfloat STIFFNESS_NORMAL = SPHERE_WALL_STIFFNESS_NORMAL_CONST * sqrt(effective_radius);
    const dfloat STIFFNESS_TANGENTIAL = SPHERE_WALL_STIFFNESS_TANGENTIAL_CONST * sqrt(effective_radius) * sqrt(abs(displacement));

    const dfloat DAMPING_NORMAL = SPHERE_WALL_DAMPING_CONST * sqrt(effective_mass * STIFFNESS_NORMAL);
    const dfloat DAMPING_TANGENTIAL = SPHERE_WALL_DAMPING_CONST * sqrt(effective_mass * STIFFNESS_TANGENTIAL);

    // Normal force
    dfloat3 f_normal = computeNormalForce(n, G, displacement, STIFFNESS_NORMAL, DAMPING_NORMAL);
    dfloat f_n = vector_length(f_normal);

    // Relative tangential velocity

    // Slot 3 is reserved for the cylindrical/duct wall. The former encoded
    // NUM_PARTICLES+10 index could overlap particle-contact slots or exceed
    // MAX_ACTIVE_COLLISIONS.
    int tang_index = -1;
    dfloat3 tang_disp = getOrUpdateTangentialDisplacement(
        pc_i, 0, true, step, G_ct, G_cn, tang_index, dfloat3(0, 0, 0), n);

    dfloat3 f_tang = computeTangentialForce(
        tang_disp, G_ct, STIFFNESS_TANGENTIAL, DAMPING_TANGENTIAL,
        PW_FRICTION_COEF, f_n, n, pc_i, tang_index, step
    );

    // Final force results
    dfloat3 f_dirs = f_normal + f_tang;
    dfloat3 m_dirs_i = cross_product(rri, f_dirs);

    //Save data in the particle information
    accumulateForceAndTorque(pc_i, f_dirs, m_dirs_i);
}


// ****************************************************************************
// ******************   AUXILIARY COLLISION FUNCTIONS  ************************
// ****************************************************************************
__device__
dfloat3 ellipsoid_intersection(ParticleCenter* pc_i, dfloat R[3][3], dfloat3 line_origin, dfloat3 line_dir, dfloat3 translation) {
    dfloat3 p0, p0_rotated, d_rotated;
    dfloat A, B, C;
    dfloat DELTA;
    dfloat3 t;

    dfloat3 pos = pc_i->getPos();
    dfloat3 center = pos + translation;

    dfloat a_axis = vector_length(pc_i->getSemiAxis1() - pos);
    dfloat b_axis = vector_length(pc_i->getSemiAxis2() - pos);
    dfloat c_axis = vector_length(pc_i->getSemiAxis3() - pos);

    dfloat inva2 = 1.0f / (a_axis * a_axis);
    dfloat invb2 = 1.0f / (b_axis * b_axis);
    dfloat invc2 = 1.0f / (c_axis * c_axis);

    p0 = line_origin - center;

    // Apply rotation to the line origin
    p0_rotated.x = R[0][0] * p0.x + R[0][1] * p0.y + R[0][2] * p0.z;
    p0_rotated.y = R[1][0] * p0.x + R[1][1] * p0.y + R[1][2] * p0.z;
    p0_rotated.z = R[2][0] * p0.x + R[2][1] * p0.y + R[2][2] * p0.z;

    // Transform the line direction into the ellipsoid's coordinate system
    d_rotated.x = R[0][0] * line_dir.x + R[0][1] * line_dir.y + R[0][2] * line_dir.z;
    d_rotated.y = R[1][0] * line_dir.x + R[1][1] * line_dir.y + R[1][2] * line_dir.z;
    d_rotated.z = R[2][0] * line_dir.x + R[2][1] * line_dir.y + R[2][2] * line_dir.z;

    // Precompute scaled components
    dfloat px_inva2 = p0_rotated.x * inva2;
    dfloat py_invb2 = p0_rotated.y * invb2;
    dfloat pz_invc2 = p0_rotated.z * invc2;

    dfloat dx_inva2 = d_rotated.x * inva2;
    dfloat dy_invb2 = d_rotated.y * invb2;
    dfloat dz_invc2 = d_rotated.z * invc2;

    // Ellipsoid equation coefficients (in rotated space)
    A = d_rotated.x * dx_inva2 + d_rotated.y * dy_invb2 + d_rotated.z * dz_invc2;
    B = 2.0f * (p0_rotated.x * dx_inva2 + p0_rotated.y * dy_invb2 + p0_rotated.z * dz_invc2);
    C = p0_rotated.x * px_inva2 + p0_rotated.y * py_invb2 + p0_rotated.z * pz_invc2 - 1.0f;

    DELTA = B * B - 4.0f * A * C;

    dfloat sqrt_delta = sqrtf(DELTA);
    dfloat denom = 2.0f * A;
    t = dfloat3((-B + sqrt_delta) / denom, (-B - sqrt_delta) / denom, 0.0f);

    return t;
}

__device__
dfloat3 ellipsoid_normal(ParticleCenter* pc_i, dfloat R[3][3], dfloat3 point, dfloat radius[1], dfloat3 translation) {
    dfloat3 local_point;
    dfloat3 grad_local;
    dfloat3 normal;
    dfloat norm;

    dfloat3 pos = pc_i->getPos();
    dfloat3 center = pos + translation;

    dfloat a_axis = vector_length(pc_i->getSemiAxis1() - pos);
    dfloat b_axis = vector_length(pc_i->getSemiAxis2() - pos);
    dfloat c_axis = vector_length(pc_i->getSemiAxis3() - pos);

    dfloat a2 = a_axis * a_axis;
    dfloat b2 = b_axis * b_axis;
    dfloat c2 = c_axis * c_axis;

    dfloat inva2 = 1.0f / a2;
    dfloat invb2 = 1.0f / b2;
    dfloat invc2 = 1.0f / c2;

    dfloat a4 = a2 * a2;
    dfloat b4 = b2 * b2;
    dfloat c4 = c2 * c2;

    dfloat3 diff = point - center;

    // Transform the point into the ellipsoid's local coordinates
    local_point.x = R[0][0] * diff.x + R[0][1] * diff.y + R[0][2] * diff.z;
    local_point.y = R[1][0] * diff.x + R[1][1] * diff.y + R[1][2] * diff.z;
    local_point.z = R[2][0] * diff.x + R[2][1] * diff.y + R[2][2] * diff.z;

    dfloat x2 = local_point.x * local_point.x;
    dfloat y2 = local_point.y * local_point.y;
    dfloat z2 = local_point.z * local_point.z;

    dfloat a4b4 = a4 * b4;
    dfloat a4c4 = a4 * c4;
    dfloat b4c4 = b4 * c4;

    dfloat inner = a4b4 * z2 + a4c4 * y2 + b4c4 * x2;
    dfloat abc = a_axis * b_axis * c_axis;
    dfloat denom = a4 * b4 * c4;
    radius[0] = abc * inner / denom;

    // Compute the gradient in local coordinates
    grad_local.x = 2.0f * local_point.x * inva2;
    grad_local.y = 2.0f * local_point.y * invb2;
    grad_local.z = 2.0f * local_point.z * invc2;

    // Transform the gradient back to global coordinates
    normal.x = R[0][0] * grad_local.x + R[1][0] * grad_local.y + R[2][0] * grad_local.z;
    normal.y = R[0][1] * grad_local.x + R[1][1] * grad_local.y + R[2][1] * grad_local.z;
    normal.z = R[0][2] * grad_local.x + R[1][2] * grad_local.y + R[2][2] * grad_local.z;

    // Normalize the normal vector
    dfloat norm2 = normal.x * normal.x + normal.y * normal.y + normal.z * normal.z;
    if (norm2 > 0.0f) { // Avoid division by zero
        norm = sqrtf(norm2);
        dfloat inv_norm = 1.0f / norm;
        normal.x *= inv_norm;
        normal.y *= inv_norm;
        normal.z *= inv_norm;
    }

    return normal;
}

namespace {

struct EllipsoidLineRoots {
    dfloat enter;
    dfloat exit;
    bool valid;
};

__device__ EllipsoidLineRoots orderedEllipsoidRoots(
    ParticleCenter* particle,
    dfloat R[3][3],
    const dfloat3& origin,
    const dfloat3& direction,
    const dfloat3& translation
) {
    EllipsoidLineRoots result = {0.0f, 0.0f, false};
    const dfloat3 pos = particle->getPos();
    const dfloat3 center = pos + translation;
    const dfloat a = vector_length(particle->getSemiAxis1() - pos);
    const dfloat b = vector_length(particle->getSemiAxis2() - pos);
    const dfloat c = vector_length(particle->getSemiAxis3() - pos);
    if (!(a > 1.0e-8f && b > 1.0e-8f && c > 1.0e-8f)) return result;

    const dfloat3 p = origin - center;
    dfloat3 pl, dl;
    pl.x = R[0][0]*p.x + R[0][1]*p.y + R[0][2]*p.z;
    pl.y = R[1][0]*p.x + R[1][1]*p.y + R[1][2]*p.z;
    pl.z = R[2][0]*p.x + R[2][1]*p.y + R[2][2]*p.z;
    dl.x = R[0][0]*direction.x + R[0][1]*direction.y + R[0][2]*direction.z;
    dl.y = R[1][0]*direction.x + R[1][1]*direction.y + R[1][2]*direction.z;
    dl.z = R[2][0]*direction.x + R[2][1]*direction.y + R[2][2]*direction.z;

    const dfloat ia2 = 1.0f/(a*a), ib2 = 1.0f/(b*b), ic2 = 1.0f/(c*c);
    const dfloat qa = dl.x*dl.x*ia2 + dl.y*dl.y*ib2 + dl.z*dl.z*ic2;
    const dfloat qb = 2.0f*(pl.x*dl.x*ia2 + pl.y*dl.y*ib2 + pl.z*dl.z*ic2);
    const dfloat qc = pl.x*pl.x*ia2 + pl.y*pl.y*ib2 + pl.z*pl.z*ic2 - 1.0f;
    if (!isfinite(qa) || !isfinite(qb) || !isfinite(qc) || qa <= 1.0e-20f) return result;

    dfloat disc = qb*qb - 4.0f*qa*qc;
    const dfloat discTolerance = 1.0e-6f*(qb*qb + fabsf(4.0f*qa*qc) + 1.0f);
    if (!isfinite(disc) || disc < -discTolerance) return result;
    if (disc < 0.0f) disc = 0.0f;
    const dfloat root = sqrtf(disc);
    const dfloat invDenom = 0.5f/qa;
    const dfloat r0 = (-qb-root)*invDenom;
    const dfloat r1 = (-qb+root)*invDenom;
    result.enter = r0 < r1 ? r0 : r1;
    result.exit = r0 < r1 ? r1 : r0;
    result.valid = isfinite(result.enter) && isfinite(result.exit);
    return result;
}

__device__ dfloat3 ellipsoidGradient(
    ParticleCenter* particle,
    dfloat R[3][3],
    const dfloat3& point,
    const dfloat3& translation
) {
    const dfloat3 pos = particle->getPos();
    const dfloat a = vector_length(particle->getSemiAxis1() - pos);
    const dfloat b = vector_length(particle->getSemiAxis2() - pos);
    const dfloat c = vector_length(particle->getSemiAxis3() - pos);
    const dfloat3 d = point - (pos + translation);
    dfloat3 local;
    local.x = R[0][0]*d.x + R[0][1]*d.y + R[0][2]*d.z;
    local.y = R[1][0]*d.x + R[1][1]*d.y + R[1][2]*d.z;
    local.z = R[2][0]*d.x + R[2][1]*d.y + R[2][2]*d.z;
    const dfloat3 gl = dfloat3(2.0f*local.x/(a*a), 2.0f*local.y/(b*b), 2.0f*local.z/(c*c));
    return dfloat3(
        R[0][0]*gl.x + R[1][0]*gl.y + R[2][0]*gl.z,
        R[0][1]*gl.x + R[1][1]*gl.y + R[2][1]*gl.z,
        R[0][2]*gl.x + R[1][2]*gl.y + R[2][2]*gl.z
    );
}

__device__ dfloat ellipsoidLevel(
    ParticleCenter* particle,
    dfloat R[3][3],
    const dfloat3& point,
    const dfloat3& translation
) {
    const dfloat3 pos=particle->getPos();
    const dfloat a=vector_length(particle->getSemiAxis1()-pos);
    const dfloat b=vector_length(particle->getSemiAxis2()-pos);
    const dfloat c=vector_length(particle->getSemiAxis3()-pos);
    const dfloat3 d=point-(pos+translation);
    const dfloat x=R[0][0]*d.x+R[0][1]*d.y+R[0][2]*d.z;
    const dfloat y=R[1][0]*d.x+R[1][1]*d.y+R[1][2]*d.z;
    const dfloat z=R[2][0]*d.x+R[2][1]*d.y+R[2][2]*d.z;
    return x*x/(a*a)+y*y/(b*b)+z*z/(c*c);
}

__device__ bool aligned(const dfloat3& a, const dfloat3& b) {
    const dfloat aa = dot_product(a,a);
    const dfloat bb = dot_product(b,b);
    if (aa <= 1.0e-20f || bb <= 1.0e-20f) return false;
    return dot_product(a,b)/sqrtf(aa*bb) >= 0.9999995f;
}

__device__ bool makeEllipsoidFrame(
    ParticleCenter* particle,
    dfloat R[3][3],
    dfloat& a,
    dfloat& b,
    dfloat& c
) {
    const dfloat3 pos = particle->getPos();
    const dfloat3 va = particle->getSemiAxis1()-pos;
    const dfloat3 vb = particle->getSemiAxis2()-pos;
    const dfloat3 vc = particle->getSemiAxis3()-pos;
    a = vector_length(va); b = vector_length(vb); c = vector_length(vc);
    if (!(a > 1.0e-8f && b > 1.0e-8f && c > 1.0e-8f)) return false;
    const dfloat3 ua=va/a, ub=vb/b, uc=vc/c;
    if (fabsf(dot_product(ua,ub)) > 1.0e-3f ||
        fabsf(dot_product(ua,uc)) > 1.0e-3f ||
        fabsf(dot_product(ub,uc)) > 1.0e-3f) return false;
    rotationMatrixFromVectors(ua,ub,uc,R);
    return true;
}

} // namespace

__device__
EllipsoidContactResult ellipsoidEllipsoidCollisionDistance(
    ParticleCenter* pc_i,
    ParticleCenter* pc_j,
    dfloat3 translation,
    unsigned int step
) {
    (void)step;
    EllipsoidContactResult out = {};
    out.status = ELLIPSOID_INVALID_GEOMETRY;
    out.signedDisplacement = 1.0e37f;

    dfloat R1[3][3], R2[3][3];
    dfloat a1,b1,c1,a2,b2,c2;
    if (!makeEllipsoidFrame(pc_i,R1,a1,b1,c1) || !makeEllipsoidFrame(pc_j,R2,a2,b2,c2)) return out;

    dfloat3 center1 = pc_i->getPos();
    dfloat3 center2 = pc_j->getPos()+translation;
    dfloat3 interior1=center1, interior2=center2;
    dfloat3 point1=center1, point2=center2;
    const int maxIterations=40;
    const dfloat rootTolerance=1.0e-6f;

    for (int iteration=0; iteration<maxIterations; ++iteration) {
        const dfloat3 direction=interior2-interior1;
        if (dot_product(direction,direction) <= 1.0e-20f) {
            out.status=ELLIPSOID_PROXY_INVALID;
            out.iterations=iteration;
            return out;
        }
        const EllipsoidLineRoots roots1=orderedEllipsoidRoots(pc_i,R1,interior1,direction,dfloat3(0,0,0));
        const EllipsoidLineRoots roots2=orderedEllipsoidRoots(pc_j,R2,interior1,direction,translation);
        if (!roots1.valid || !roots2.valid) {
            out.status=ELLIPSOID_NUMERICAL_FAILURE;
            out.iterations=iteration;
            return out;
        }
        point1=interior1+roots1.exit*direction;
        point2=interior1+roots2.enter*direction;

        if (roots2.enter <= roots1.exit+rootTolerance) {
            // Intersection is established by interval overlap. The following loop only
            // computes the legacy, length-valued fixed-sphere penetration proxy.
            const dfloat radius=(dfloat)ELLIPSOID_PENETRATION_SPHERE_RADIUS;
            if (!(radius > 0.0f) || !isfinite(radius)) {
                out.status=ELLIPSOID_PROXY_INVALID;
                return out;
            }
            for (int overlapIteration=0; overlapIteration<maxIterations; ++overlapIteration) {
                dfloat dummy1[1],dummy2[1];
                const dfloat3 normal1=ellipsoid_normal(pc_i,R1,point1,dummy1,dfloat3(0,0,0));
                const dfloat3 normal2=ellipsoid_normal(pc_j,R2,point2,dummy2,translation);
                const dfloat3 overlap12=point1-point2;
                const dfloat3 overlap21=point2-point1;
                const dfloat3 sphere1=point1-radius*normal1;
                const dfloat3 sphere2=point2-radius*normal2;

                if (aligned(overlap12,normal1) && aligned(overlap21,normal2)) {
                    const dfloat proxy=vector_length(sphere2-sphere1)-2.0f*radius;
                    out.pointA=point1;
                    out.pointBImage=point2;
                    out.normalBToA=-normal1;
                    out.curvatureRadiusA=dummy1[0];
                    out.curvatureRadiusB=dummy2[0];
                    out.signedDisplacement=proxy;
                    out.iterations=iteration+overlapIteration+1;
                    const bool centersInside=ellipsoidLevel(pc_i,R1,sphere1,dfloat3(0,0,0)) < 1.0f &&
                                             ellipsoidLevel(pc_j,R2,sphere2,translation) < 1.0f;
                    // Once penetration exceeds 2r the absolute center distance folds
                    // over and no longer represents the intended signed overlap.
                    const bool shallowBranch=dot_product(sphere2-sphere1,normal1) > 0.0f;
                    out.status=(isfinite(proxy) && proxy < 0.0f && centersInside && shallowBranch)
                        ? ELLIPSOID_INTERSECTING : ELLIPSOID_PROXY_INVALID;
                    return out;
                }

                const dfloat3 proxyDirection=sphere2-sphere1;
                if (dot_product(proxyDirection,proxyDirection) <= 1.0e-20f) {
                    out.status=ELLIPSOID_PROXY_INVALID;
                    return out;
                }
                const EllipsoidLineRoots overlapRoots1=orderedEllipsoidRoots(pc_i,R1,sphere1,proxyDirection,dfloat3(0,0,0));
                const EllipsoidLineRoots overlapRoots2=orderedEllipsoidRoots(pc_j,R2,sphere1,proxyDirection,translation);
                if (!overlapRoots1.valid || !overlapRoots2.valid) {
                    out.status=ELLIPSOID_NUMERICAL_FAILURE;
                    return out;
                }
                point1=sphere1+overlapRoots1.exit*proxyDirection;
                point2=sphere1+overlapRoots2.enter*proxyDirection;
            }
            out.status=ELLIPSOID_MAX_ITERATIONS;
            return out;
        }

        const dfloat3 gradient1=ellipsoidGradient(pc_i,R1,point1,dfloat3(0,0,0));
        const dfloat3 gradient2=ellipsoidGradient(pc_j,R2,point2,translation);
        if (aligned(point2-point1,gradient1) && aligned(point1-point2,gradient2)) {
            dfloat curvature1[1],curvature2[1];
            const dfloat3 normal1=ellipsoid_normal(pc_i,R1,point1,curvature1,dfloat3(0,0,0));
            ellipsoid_normal(pc_j,R2,point2,curvature2,translation);
            out.status=ELLIPSOID_SEPARATED;
            out.signedDisplacement=vector_length(point2-point1);
            out.pointA=point1;
            out.pointBImage=point2;
            out.normalBToA=-normal1;
            out.curvatureRadiusA=curvature1[0];
            out.curvatureRadiusB=curvature2[0];
            out.iterations=iteration+1;
            return out;
        }

        dfloat minAxis1=a1; if (b1<minAxis1) minAxis1=b1; if (c1<minAxis1) minAxis1=c1;
        dfloat minAxis2=a2; if (b2<minAxis2) minAxis2=b2; if (c2<minAxis2) minAxis2=c2;
        const dfloat gamma1=0.5f*minAxis1*minAxis1;
        const dfloat gamma2=0.5f*minAxis2*minAxis2;
        interior1=point1-gamma1*gradient1;
        interior2=point2-gamma2*gradient2;
    }
    out.status=ELLIPSOID_MAX_ITERATIONS;
    out.iterations=maxIterations;
    return out;
}
#endif //PARTICLE_MODEL