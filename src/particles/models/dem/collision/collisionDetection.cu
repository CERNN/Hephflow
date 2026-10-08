//functions to determine if the particle will collide
#include "collisionDetection.cuh"

#ifdef PARTICLE_MODEL

// ------------------------------------------------------------------------ 
// -------------------- COLLISION HANDLER --------- -----------------------
// ------------------------------------------------------------------------ 

//collision
__global__
void particlesCollisionHandler(
    ParticleShape *shape,
    ParticleCenter *pArray,
    const unsigned char *collisionFlags,
    unsigned int step){
    /* Maps a 1D array to the strict lower triangle, with one thread for each
    unique particle pair. Example for 7 particles:

    FLOYD TRIANGLE
        c0  c1  c2  c3  c4  c5  c6
    r0  0
    r1  1   2
    r2  3   4   5
    r3  6   7   8   9
    r4  10  11  12  13  14
    r5  15  16  17  18  19  20
    r6  21  22  23  24  25  26  27

    Index 7 is in r3, c1. It will compare p[1] (particle in index 1), from column,
    with p[4], from row (this is because for all rows one is added to its index)

    Index 0 will compare p[0] (column) and p[1] (row)
    Index 13 will compare p[3] (column) and p[5] (row)
    Index 20 will compare p[5] (column) and p[6] (row)

    The inverse triangular estimate is evaluated in double precision and then
    corrected using exact 64-bit triangular numbers. The correction is needed
    because applying ceil() directly to a single-precision square root creates
    self-pairs and skipped/duplicated pairs for sufficiently large arrays.
    */
    const unsigned long long idx =
        static_cast<unsigned long long>(threadIdx.x) +
        static_cast<unsigned long long>(blockDim.x) * blockIdx.x;

    if(idx >= TOTAL_PCOLLISION_THREADS)
        return;

    const CollisionPairIndex pair = collisionPairFromLinearIndex(idx);
    const unsigned int row = pair.row;
    const unsigned int column = pair.column;

    if (row >= NUM_PARTICLES || column >= row)
        return;

    if (!particlePairCollisionEnabled(collisionFlags[column], collisionFlags[row]))
        return;

    ParticleCenter* pc_i = &pArray[column];
    ParticleShape* shape_i = &shape[column];
    ParticleCenter* pc_j = &pArray[row];
    ParticleShape* shape_j = &shape[row];
    if(!pc_i->getMovable() && !pc_j->getMovable())
        return;
    checkCollisionBetweenParticles(column,row,shape_i,shape_j,pc_i,pc_j,step);
}

__global__
void particleWallCollisionHandler(
    ParticleShape *shape,
    ParticleCenter *pArray,
    const unsigned char *collisionFlags,
    ParticleWallForces *d_pwForces,
    unsigned int step){
    const unsigned int particle = threadIdx.x + blockDim.x * blockIdx.x;
    if (particle >= NUM_PARTICLES ||
        !particleWallCollisionEnabled(collisionFlags[particle]))
        return;

    ParticleCenter* pc = &pArray[particle];
    if (!pc->getMovable())
        return;
    checkCollisionWalls(&shape[particle], pc, d_pwForces, step);
}

#if defined(EXTERNAL_DUCT_BC)
__device__
void checkCollisionExternalDuctSphere(
    ParticleCenter* pc_i,
    ParticleWallForces* d_pwForces,
    unsigned int step
) {
    const dfloat3 ductCenter=dfloat3(
        DUCT_CENTER_X,DUCT_CENTER_Y,pc_i->getPosZ());
    const dfloat radialDistance=vector_length(pc_i->getPos()-ductCenter);
    const dfloat gap=cylinderContactGap(
        radialDistance,pc_i->getRadius(),EXTERNAL_DUCT_BC_RADIUS,-1);
    if (gap<0.0_df) {
        const Wall wallData=determineCircularWall(
            pc_i->getPos(),EXTERNAL_DUCT_BC_RADIUS,-1);
        sphereWallCollision({pc_i,wallData,-gap,step},d_pwForces);
    }
}

__device__
void checkCollisionExternalDuctCapsule(ParticleCenter* pc_i,unsigned int step) {
    const dfloat3 ductCenter=dfloat3(
        DUCT_CENTER_X,DUCT_CENTER_Y,pc_i->getPosZ());
    const dfloat3 endpoints[2]={pc_i->getSemiAxis1(),pc_i->getSemiAxis2()};
    for (int i=0; i<2; ++i) {
        const dfloat gap=cylinderContactGap(
            vector_length(endpoints[i]-ductCenter),pc_i->getRadius(),
            EXTERNAL_DUCT_BC_RADIUS,-1);
        if (gap<0.0_df) {
            const Wall wallData=determineCircularWall(
                endpoints[i],EXTERNAL_DUCT_BC_RADIUS,-1);
            capsuleWallCollisionCap({pc_i,wallData,-gap,step,endpoints[i]});
        }
    }
}

__device__
void checkCollisionExternalDuctEllipsoid(ParticleCenter* pc_i,unsigned int step) {
    const dfloat3 axisStart=dfloat3(
        DUCT_CENTER_X,DUCT_CENTER_Y,-EXTERNAL_DUCT_BC_RADIUS);
    const dfloat3 axisEnd=dfloat3(
        DUCT_CENTER_X,DUCT_CENTER_Y,NZ_TOTAL+EXTERNAL_DUCT_BC_RADIUS);
    dfloat3 pointCylinder[1];
    dfloat3 pointEllipsoid[1];
    dfloat curvatureRadius[1];
    const dfloat gap=ellipsoidSegmentCollisionDistance(
        pc_i,axisStart,axisEnd,EXTERNAL_DUCT_BC_RADIUS,
        pointCylinder,pointEllipsoid,curvatureRadius,-1,step);
    if (gap<0.0_df) {
        const Wall wallData=determineCircularWall(
            pointCylinder[0],EXTERNAL_DUCT_BC_RADIUS,-1);
        ellipsoidCylinderCollision(
            {pc_i,wallData,-gap,step},pointEllipsoid,curvatureRadius,
            axisStart,axisEnd,EXTERNAL_DUCT_BC_RADIUS,-1);
    }
}
#endif

__device__
void checkCollisionWalls(ParticleShape *shape, ParticleCenter* pc_i, ParticleWallForces *d_pwForces, unsigned int step){
    switch (*shape) {
        case SPHERE:
            checkCollisionWallsSphere(pc_i,d_pwForces,step);
            #if defined(EXTERNAL_DUCT_BC)
            checkCollisionExternalDuctSphere(pc_i,d_pwForces,step);
            #endif
            break;
        case CAPSULE:
            checkCollisionWallsCapsule(pc_i,step);
            #if defined(EXTERNAL_DUCT_BC)
            checkCollisionExternalDuctCapsule(pc_i,step);
            #endif
            break;
        case ELLIPSOID:
            checkCollisionWallsElipsoid(pc_i,step);
            #if defined(EXTERNAL_DUCT_BC)
            checkCollisionExternalDuctEllipsoid(pc_i,step);
            #endif
            break;
        default:
            // Handle unknown particle types
            break;
    }
}

__device__
void checkCollisionWallsSphere(ParticleCenter* pc_i, ParticleWallForces *d_pwForces, unsigned int step) {
    const dfloat3 pos_i = pc_i->getPos();
    const dfloat radius = pc_i->getRadius();

    const int maxWalls = 6;
    Wall walls[maxWalls];
    int wallCount = 0;

    #ifdef BC_X_WALL
    walls[wallCount++] = wall(dfloat3(1, 0, 0), 0,dfloat3(0,-WALL_VEL_UY,-WALL_VEL_UZ));
    walls[wallCount++] = wall(dfloat3(-1, 0, 0), (NX - 1),dfloat3(0,WALL_VEL_UY,WALL_VEL_UZ));
    #endif

    #ifdef BC_Y_WALL
    walls[wallCount++] = wall(dfloat3(0, 1, 0), 0,dfloat3(-WALL_VEL_UX,0,-WALL_VEL_UZ));
    walls[wallCount++] = wall(dfloat3(0, -1, 0), (NY - 1),dfloat3(WALL_VEL_UX,0,WALL_VEL_UZ));
    #endif

    #ifdef BC_Z_WALL
    walls[wallCount++] = wall(dfloat3(0, 0, 1), 0,dfloat3(-WALL_VEL_UX,-WALL_VEL_UY,0));
    walls[wallCount++] = wall(dfloat3(0, 0, -1), (NZ_TOTAL - 1),dfloat3(WALL_VEL_UX,WALL_VEL_UY,0));
    #endif

    for (int i = 0; i < wallCount; ++i) {
        dfloat distanceWall;
        
        // Check the direction of the wall normal to use the correct distance calculation
        // The positive normal vectors point inward from a boundary at 0, while
        // the negative normal vectors point inward from a boundary at N-1.
        if (walls[i].normal.x + walls[i].normal.y + walls[i].normal.z > 0) {
            distanceWall = dot_product(pos_i, walls[i].normal) - walls[i].distance;
        } else {
            distanceWall = walls[i].distance + dot_product(pos_i, walls[i].normal);
        }

        if (distanceWall < radius) {
            sphereWallCollision({pc_i, walls[i], radius - distanceWall, step}, d_pwForces);
        }
    }
}

__device__
void checkCollisionWallsCapsule(ParticleCenter* pc_i, unsigned int step) {
    const dfloat halfLength = vector_length(pc_i->getSemiAxis1());
    const dfloat radius = pc_i->getRadius();
    const dfloat3 endpoint1 = pc_i->getSemiAxis1();
    const dfloat3 endpoint2 = pc_i->getSemiAxis2();

    const int maxWalls = 6;
    Wall walls[maxWalls];
    int wallCount = 0;

    #ifdef BC_X_WALL
    walls[wallCount++] = wall(dfloat3(1,0,0),0,dfloat3(0,-WALL_VEL_UY,-WALL_VEL_UZ));
    walls[wallCount++] = wall(dfloat3(-1,0,0),NX-1,dfloat3(0,WALL_VEL_UY,WALL_VEL_UZ));
    #endif

    #ifdef BC_Y_WALL
    walls[wallCount++] = wall(dfloat3(0,1,0),0,dfloat3(-WALL_VEL_UX,0,-WALL_VEL_UZ));
    walls[wallCount++] = wall(dfloat3(0,-1,0),NY-1,dfloat3(WALL_VEL_UX,0,WALL_VEL_UZ));
    #endif
    
    #ifdef BC_Z_WALL
    walls[wallCount++] = wall(dfloat3(0,0,1),0,dfloat3(-WALL_VEL_UX,-WALL_VEL_UY,0));
    walls[wallCount++] = wall(dfloat3(0,0,-1),NZ_TOTAL-1,dfloat3(WALL_VEL_UX,WALL_VEL_UY,0));
    #endif

    for (int i = 0; i < wallCount; ++i) {
        //Need to handle the two different distance calculations based on the wall
        dfloat distanceWall1;
        dfloat distanceWall2;
        
        // We need to determine if it's an inward- or outward-facing wall.
        // A simple way is to check the normal vector.
        if (walls[i].normal.x + walls[i].normal.y + walls[i].normal.z > 0) { // Outward facing
            distanceWall1 = dot_product(endpoint1, walls[i].normal) - walls[i].distance;
            distanceWall2 = dot_product(endpoint2, walls[i].normal) - walls[i].distance;
        } else { // Inward facing
            distanceWall1 = walls[i].distance + dot_product(endpoint1, walls[i].normal);
            distanceWall2 = walls[i].distance + dot_product(endpoint2, walls[i].normal);
        }

        if (distanceWall1 < radius) {
            capsuleWallCollisionCap({pc_i, walls[i], radius - distanceWall1, step, endpoint1});
        }
        if (distanceWall2 < radius) {
            capsuleWallCollisionCap({pc_i, walls[i], radius - distanceWall2, step, endpoint2});
        }
    }
}

__device__
void checkCollisionWallsElipsoid(ParticleCenter* pc_i, unsigned int step) {
    const int maxWalls = 6;
    Wall walls[maxWalls];
    int wallCount = 0;

    #ifdef BC_X_WALL
    walls[wallCount++] = wall(dfloat3(1,0,0),0,dfloat3(0,-WALL_VEL_UY,-WALL_VEL_UZ));
    walls[wallCount++] = wall(dfloat3(-1,0,0),NX-1,dfloat3(0,WALL_VEL_UY,WALL_VEL_UZ));
    #endif

    #ifdef BC_Y_WALL
    walls[wallCount++] = wall(dfloat3(0,1,0),0,dfloat3(-WALL_VEL_UX,0,-WALL_VEL_UZ));
    walls[wallCount++] = wall(dfloat3(0,-1,0),NY-1,dfloat3(WALL_VEL_UX,0,WALL_VEL_UZ));
    #endif
    
    #ifdef BC_Z_WALL
    walls[wallCount++] = wall(dfloat3(0,0,1),0,dfloat3(-WALL_VEL_UX,-WALL_VEL_UY,0));
    walls[wallCount++] = wall(dfloat3(0,0,-1),NZ_TOTAL-1,dfloat3(WALL_VEL_UX,WALL_VEL_UY,0));
    #endif

    // Loop through all defined walls and check for collision
    for (int i = 0; i < wallCount; ++i) {
        const EllipsoidWallContactResult contact=ellipsoidWallCollisionDistance(pc_i,walls[i],step);
        if (contact.status==ELLIPSOID_INTERSECTING && contact.signedDistance<0.0f) {
            dfloat cr[1]={contact.curvatureRadius};
            ellipsoidWallCollision(
                {pc_i,walls[i],-contact.signedDistance,step,contact.pointEllipsoid},
                cr
            );
        }
    }
}

// ------------------------------------------------------------------------
// -------------------- COLLISION BETWEEN PARTICLES -----------------------
// ------------------------------------------------------------------------ 

__device__
void checkCollisionBetweenParticles( unsigned int column,unsigned int row,ParticleShape *shape_i,ParticleShape *shape_j,ParticleCenter* pc_i,ParticleCenter* pc_j,int step){

    switch (*shape_i) {
        case SPHERE:
            switch (*shape_j) {
            case SPHERE:
                sphereSphereCollisionCheck(column,row,pc_i,pc_j,step);
                break;
            case CAPSULE:
                capsuleSphereCollisionCheck(column,row,shape_i,pc_i,pc_j,step);
                break;
            case ELLIPSOID:
                break;
            default:
                // Handle unknown particle types
                break;
            }
            break;
        case CAPSULE:
            switch (*shape_j) {
            case SPHERE:
                capsuleSphereCollisionCheck(column,row,shape_i,pc_i,pc_j,step);
                break;
            case CAPSULE:
                capsuleCapsuleCollisionCheck(column,row,pc_i,pc_j, step, pc_i->getSemiAxis1(),pc_i->getSemiAxis2(), pc_i->getRadius(),pc_j->getSemiAxis1(), pc_j->getSemiAxis2(), pc_j->getRadius());
            case ELLIPSOID:
                //collision capsule-ellipsoid
                break;
            default:
                break;
            }
            break;
        case ELLIPSOID:
            switch (*shape_j) {
            case SPHERE:
                //collision ellipsoid-sphere
                break;
            case CAPSULE:
                //collision ellipsoid-capsule
                break;
            case ELLIPSOID:
                ellipsoidEllipsoidCollisionCheck(column,row,pc_i,pc_j, step);
                break;
            default:
                // Handle unknown particle types
                break;
            }
            break;
        default:
            // Handle unknown particle types
            break;
    }
}

// ------------------------------------------------------------------------ 
// -------------------- INTER PARTICLE COLLISION CHECK---------------------
// ------------------------------------------------------------------------ 


void sphereSphereCollisionCheck(unsigned int column,unsigned int row,ParticleCenter* pc_i, ParticleCenter* pc_j, int step){
    dfloat gap = sphereSphereGap(pc_i, pc_j);
    dfloat3 diff_pos = getDiffPeriodic(pc_i->getPos(), pc_j->getPos());
    dfloat mag_dist = vector_length(diff_pos);
    dfloat3 normal = (mag_dist != 0) ? diff_pos / mag_dist : dfloat3{0,0,0};

    CollisionContext ctx = {};
    ctx.pc_i = pc_i;
    ctx.pc_j = pc_j;
    ctx.step = step;
    ctx.partnerID = row;
    ctx.wall.normal = normal;

    if (gap < 0) {
        ctx.displacement = -gap;
        sphereSphereCollision(ctx); // Hertz
    }
    else {
        //sphereSphereLubrication(ctx, gap);
        //sphereSphereRepulsion(ctx, gap);
        //sphereSphereAttraction(ctx, gap);
    }
}

__device__
void capsuleCapsuleCollisionCheck(unsigned int column, unsigned int row, ParticleCenter* pc_i, ParticleCenter* pc_j, int step, dfloat3 cylA1, dfloat3 cylA2, dfloat radiusA, dfloat3 cylB1, dfloat3 cylB2, dfloat radiusB) {
    dfloat3 closestOnA[1];
    dfloat3 closestOnB[1];

    dfloat dist = segment_segment_closest_points_periodic(cylA1, cylA2, cylB1, cylB2, closestOnA, closestOnB);
    if (dist < radiusA + radiusB) {
        // Compute normal direction
        dfloat3 diff_pos = getDiffPeriodic(closestOnA[0], closestOnB[0]);
        dfloat mag_dist = vector_length(diff_pos);
        dfloat3 normal = (mag_dist != 0) ? (diff_pos / mag_dist) : dfloat3{0.0, 0.0, 0.0};

        CollisionContext ctx = {};
        ctx.pc_i = pc_i;
        ctx.pc_j = pc_j;
        ctx.displacement = radiusA + radiusB - mag_dist; // overlap
        ctx.step = step;
        ctx.partnerID = row;
        ctx.wall.normal = normal;
        capsuleCapsuleCollision(ctx,closestOnA,closestOnB);
    }
    return;
}

__device__
void capsuleSphereCollisionCheck(unsigned int column, unsigned int row, ParticleShape *shape, ParticleCenter* pc_i, ParticleCenter* pc_j, int step) {
    dfloat3 closestOnAB[1];

    if (*shape == SPHERE) {
        dfloat3 pos_i = pc_i->getPos();
        dfloat dist = point_to_segment_distance_periodic(pc_i->getPos(), pc_j->getSemiAxis1(), pc_j->getSemiAxis2(), closestOnAB);
        if (dist < pc_i->getRadius() + pc_j->getRadius()) {
            dfloat3 diff_pos = getDiffPeriodic(pos_i, closestOnAB[0]);
            dfloat mag_dist = vector_length(diff_pos);
            dfloat3 normal = (mag_dist != 0) ? (diff_pos / mag_dist) : dfloat3{0.0, 0.0, 0.0};

            CollisionContext ctx = {};
            ctx.pc_i = pc_i;
            ctx.pc_j = pc_j;
            ctx.displacement = pc_i->getRadius() + pc_j->getRadius() - mag_dist;
            ctx.step = step;
            ctx.partnerID = row;
            ctx.wall.normal = normal;
            capsuleCapsuleCollision(ctx,&pos_i,closestOnAB);
        }
    } else {
        dfloat3 pos_j = pc_j->getPos();
        dfloat dist = point_to_segment_distance_periodic(pc_j->getPos(), pc_i->getSemiAxis1(), pc_i->getSemiAxis2(), closestOnAB);
        if (dist < pc_i->getRadius() + pc_j->getRadius()) {
            dfloat3 diff_pos = getDiffPeriodic(pos_j, closestOnAB[0]);
            dfloat mag_dist = vector_length(diff_pos);
            dfloat3 normal = (mag_dist != 0) ? (diff_pos / mag_dist) : dfloat3{0.0, 0.0, 0.0};

            CollisionContext ctx = {};
            ctx.pc_i = pc_i;
            ctx.pc_j = pc_j;
            ctx.displacement = pc_i->getRadius() + pc_j->getRadius() - mag_dist;
            ctx.step = step;
            ctx.partnerID = row;
            ctx.wall.normal = normal;
            capsuleCapsuleCollision(ctx,&pos_j,closestOnAB);
        }
    }
    return;
}

__device__
void ellipsoidEllipsoidCollisionCheck(unsigned int column, unsigned int row, ParticleCenter* pc_i, ParticleCenter* pc_j, int step) {
    EllipsoidContactResult bestResult = {};
    bestResult.status = ELLIPSOID_INVALID_GEOMETRY;
    bestResult.signedDisplacement = 1E+37f;
    dfloat3 bestTranslation;
    bool haveResult = false;

    for (int i = 0; i < NUM_PERIODIC_DOMAIN_OFFSET; ++i) {
        int dx = PERIODIC_DOMAIN_OFFSET[i][0];
        int dy = PERIODIC_DOMAIN_OFFSET[i][1];
        int dz = PERIODIC_DOMAIN_OFFSET[i][2];

        dfloat3 translation = dfloat3(dx * NX, dy * NY, dz * NZ_TOTAL);
        EllipsoidContactResult candidate = ellipsoidEllipsoidCollisionDistance(pc_i, pc_j, translation, step);
        const bool candidateUsable = candidate.status == ELLIPSOID_SEPARATED ||
                                     candidate.status == ELLIPSOID_INTERSECTING;
        if (candidateUsable && (!haveResult || candidate.signedDisplacement < bestResult.signedDisplacement)) {
            bestResult = candidate;
            bestTranslation = translation;
            haveResult = true;
        }
    }

    #ifdef PARTICLE_DEBUG
    if (!haveResult) {
        printf("ELLIPSOID CONTACT FAILURE i=%u j=%u status=%d step=%d\n",
               column,row,(int)bestResult.status,step);
    }
    #endif

    if (haveResult && bestResult.status == ELLIPSOID_INTERSECTING && bestResult.signedDisplacement < 0.0f) {
        dfloat3 closestOnA[1] = {bestResult.pointA};
        // ellipsoidEllipsoidCollision currently accepts B in the base frame and
        // adds the selected image translation internally.
        dfloat3 closestOnB[1] = {bestResult.pointBImage-bestTranslation};
        dfloat cr1[1] = {bestResult.curvatureRadiusA};
        dfloat cr2[1] = {bestResult.curvatureRadiusB};

        CollisionContext ctx = {};
        ctx.pc_i = pc_i;
        ctx.pc_j = pc_j;
        ctx.displacement = bestResult.signedDisplacement;
        ctx.step = step;
        ctx.partnerID = row;
        ctx.wall.normal = bestResult.normalBToA;


        ellipsoidEllipsoidCollision(ctx,closestOnA, closestOnB, cr1, cr2, bestTranslation);
    }
}

__device__
dfloat sphereSphereGap(ParticleCenter* pc_i, ParticleCenter* pc_j) {
    dfloat3 p1 = pc_i->getPos();
    dfloat3 p2 = pc_j->getPos();

    dfloat r1 = pc_i->getRadius();
    dfloat r2 = pc_j->getRadius();

    // Calculate the distance between the particle centers, accounting for periodic boundaries.
    dfloat dist = point_to_point_distance_periodic(p1, p2);
    
    // Return the gap between the spheres.
    return dist - (r1 + r2);
}

#if defined(CURVED_BOUNDARY_CONDITION) || defined(EXTERNAL_DUCT_BC)
__device__
Wall determineCircularWall(dfloat3 pos_i, dfloat R, dfloat dir){
    Wall tempWall;

    dfloat3 center = dfloat3(DUCT_CENTER_X,DUCT_CENTER_Y,0.0);

    tempWall.normal = dfloat3( dir * (pos_i.x-DUCT_CENTER_X), dir * (pos_i.y-DUCT_CENTER_Y),0.0);
    tempWall.normal = vector_normalize(tempWall.normal);

    dfloat3 contactPoint = dfloat3(center - R * tempWall.normal);

    tempWall.distance = vector_length(contactPoint - pos_i);

    return tempWall;
}
#endif

__device__
dfloat ellipsoidSegmentCollisionDistance( ParticleCenter* pc_i, dfloat3 P1, dfloat3 P2, dfloat cRadius ,dfloat3 contactPoint1[1], dfloat3 contactPoint2[1], dfloat cr[1], int cyDir, unsigned int step){
    (void)step;
    dfloat RE[3][3];
    const dfloat3 center=pc_i->getPos();
    const dfloat3 axisVector1=pc_i->getSemiAxis1()-center;
    const dfloat3 axisVector2=pc_i->getSemiAxis2()-center;
    const dfloat3 axisVector3=pc_i->getSemiAxis3()-center;
    const dfloat a=vector_length(axisVector1);
    const dfloat b=vector_length(axisVector2);
    const dfloat c=vector_length(axisVector3);
    const dfloat3 cylinderAxisVector=P2-P1;
    const dfloat cylinderAxisLength=vector_length(cylinderAxisVector);
    if (!(a>1.0e-8f && b>1.0e-8f && c>1.0e-8f &&
          cRadius>0.0f && cylinderAxisLength>1.0e-8f)) return 1.0e37f;

    const dfloat3 u1=axisVector1/a;
    const dfloat3 u2=axisVector2/b;
    const dfloat3 u3=axisVector3/c;
    rotationMatrixFromVectors(u1,u2,u3,RE);

    const dfloat3 cylinderAxis=cylinderAxisVector/cylinderAxisLength;
    const dfloat3 reference=fabsf(cylinderAxis.x)<0.8f
        ? dfloat3(1.0f,0.0f,0.0f) : dfloat3(0.0f,1.0f,0.0f);
    const dfloat3 radialU=vector_normalize(cross_product(cylinderAxis,reference));
    const dfloat3 radialV=vector_normalize(cross_product(cylinderAxis,radialU));
    const dfloat axialCenter=dot_product(center-P1,cylinderAxis);
    const dfloat3 radialCenter=center-(P1+axialCenter*cylinderAxis);
    const bool containing=cyDir<0;

    const int angularSamples=32;
    const dfloat twoPi=6.2831853071795864769f;
    const dfloat angularStep=twoPi/(dfloat)angularSamples;
    dfloat bestTheta=0.0f;
    dfloat bestValue=-1.0e37f;

    // Maximize the radial support function in the plane normal to the cylinder
    // axis.  This gives the exact side-wall extremum without an auxiliary
    // sphere.  The angular search is dimensionless, so uniform geometry scaling
    // scales the returned signed gap by the same factor.
    for (int sample=0; sample<angularSamples; ++sample) {
        const dfloat theta=angularStep*(dfloat)sample;
        const dfloat3 direction=cosf(theta)*radialU+sinf(theta)*radialV;
        const dfloat x=dot_product(direction,u1);
        const dfloat y=dot_product(direction,u2);
        const dfloat z=dot_product(direction,u3);
        const dfloat support=sqrtf(a*a*x*x+b*b*y*y+c*c*z*z);
        const dfloat value=dot_product(radialCenter,direction)+(containing ? support : -support);
        if (value>bestValue) {
            bestValue=value;
            bestTheta=theta;
        }
    }

    dfloat left=bestTheta-angularStep;
    dfloat right=bestTheta+angularStep;
    for (int iteration=0; iteration<24; ++iteration) {
        const dfloat theta1=(2.0f*left+right)/3.0f;
        const dfloat theta2=(left+2.0f*right)/3.0f;
        const dfloat3 direction1=cosf(theta1)*radialU+sinf(theta1)*radialV;
        const dfloat3 direction2=cosf(theta2)*radialU+sinf(theta2)*radialV;
        const dfloat x1=dot_product(direction1,u1), y1=dot_product(direction1,u2), z1=dot_product(direction1,u3);
        const dfloat x2=dot_product(direction2,u1), y2=dot_product(direction2,u2), z2=dot_product(direction2,u3);
        const dfloat support1=sqrtf(a*a*x1*x1+b*b*y1*y1+c*c*z1*z1);
        const dfloat support2=sqrtf(a*a*x2*x2+b*b*y2*y2+c*c*z2*z2);
        const dfloat value1=dot_product(radialCenter,direction1)+(containing ? support1 : -support1);
        const dfloat value2=dot_product(radialCenter,direction2)+(containing ? support2 : -support2);
        if (value1<value2) left=theta1; else right=theta2;
    }

    bestTheta=0.5f*(left+right);
    const dfloat3 radialDirection=cosf(bestTheta)*radialU+sinf(bestTheta)*radialV;
    const dfloat x=dot_product(radialDirection,u1);
    const dfloat y=dot_product(radialDirection,u2);
    const dfloat z=dot_product(radialDirection,u3);
    const dfloat support=sqrtf(a*a*x*x+b*b*y*y+c*c*z*z);
    if (!(support>1.0e-20f) || !isfinite(support)) return 1.0e37f;
    const dfloat3 supportVector=(
        (a*a*x)*u1+(b*b*y)*u2+(c*c*z)*u3)/support;
    const dfloat3 ellipsoidPoint=center+(containing ? supportVector : -supportVector);
    const dfloat axialPoint=dot_product(ellipsoidPoint-P1,cylinderAxis);
    const dfloat3 cylinderPoint=P1+axialPoint*cylinderAxis+cRadius*radialDirection;
    const dfloat radialExtent=dot_product(radialCenter,radialDirection)+
        (containing ? support : -support);

    contactPoint1[0]=cylinderPoint;
    contactPoint2[0]=ellipsoidPoint;
    ellipsoid_normal(pc_i,RE,ellipsoidPoint,cr,dfloat3(0,0,0));
    return containing ? cRadius-radialExtent : radialExtent-cRadius;
}

#endif //PARTICLE_MODEL
