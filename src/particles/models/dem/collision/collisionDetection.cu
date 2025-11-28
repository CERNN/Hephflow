//functions to determine if the particle will collide
#include "collisionDetection.cuh"

#ifdef PARTICLE_MODEL

// ------------------------------------------------------------------------ 
// -------------------- COLLISION HANDLER --------- -----------------------
// ------------------------------------------------------------------------ 

//collision
__global__
void particlesCollisionHandler(ParticleShape *shape, ParticleCenter *pArray, unsigned int step){
    /* Maps a 1D array to a Floyd triangle, where the last row is for checking
    collision against the wall and the other ones to check collision between 
    particles, with index given by row/column. Example for 7 particles:

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

    For the last column, the particles check collision against the wall.
    Index 21 will check p[0] (column) collision against the wall
    Index 27 will check p[6] (column) collision against the wall
    Index 24 will check p[3] (column) collision against the wall

    FROM INDEX TO ROW/COLUMN
    Starting column/row from 1, the n'th row always ends (n)*(n+1)/2+1. So:

    k = (n)*(n+1)/2+1
    n^2 + n - (2k+1) = 0

    (with k=particle index)
    n_row = ceil((-1 + Sqrt(1 + 8(k+1))) / 2)
    n_column = k - n_row * (n_row - 1) / 2
    */
    const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if(idx > TOTAL_PCOLLISION_IBM_THREADS)
        return;
    
    const unsigned int row = ceil((-1.0+sqrt((dfloat)1+8*(idx+1)))/2);
    const unsigned int column = idx - ((row-1)*row)/2;

    ParticleCenter* pc_i = &pArray[column];
    ParticleShape* shape_i = &shape[column];
   
    //collision against walls
    if(row == NUM_PARTICLES){
        if(!pc_i->getMovable())
            return;
        checkCollisionWalls(shape_i,pc_i,step);
    }else{    //Collision between particles
        ParticleCenter* pc_j = &pArray[row]; 
        ParticleShape* shape_j = &shape[row];
        if(!pc_i->getMovable() && !pc_j->getMovable())
            return;
        checkCollisionBetweenParticles(column,row,shape_i,shape_j,pc_i,pc_j,step);
    }
}

__device__
void checkCollisionWalls(ParticleShape *shape, ParticleCenter* pc_i, unsigned int step){
    switch (*shape) {
        case SPHERE:
            checkCollisionWallsSphere(pc_i,step);
            break;
        case CAPSULE:
            checkCollisionWallsCapsule(pc_i,step);
            break;
        case ELLIPSOID:
            checkCollisionWallsElipsoid(pc_i,step);
            break;
        default:
            // Handle unknown particle types
            break;
    }  

    //TODO: find a way to remove the double check
    #if defined (EXTERNAL_DUCT_BC)
        dfloat dist;
        Wall wallData;
        dfloat3 ductCenter = dfloat3(DUCT_CENTER_X, DUCT_CENTER_Y, pc_i->getPosZ());
        dfloat3 endpoint;
        dfloat3 closestOnA[1];
        dfloat3 closestOnB[1];
        dfloat cr[1];

        switch (*shape) {
            case SPHERE:
                dist = vector_length(pc_i->getPos() - ductCenter);
                if(EXTERNAL_DUCT_BC_RADIUS - dist < pc_i->getRadius()){
                    wallData = determineCircularWall(pc_i->getPos(),EXTERNAL_DUCT_BC_RADIUS,-1);
                    sphereWallCollision({pc_i, wallData, (dfloat)EXTERNAL_DUCT_BC_RADIUS - (pc_i->getRadius() + dist), step});
                }
                break;
            case CAPSULE:
                //two cases (it always collide on the hemispheres)
                endpoint = pc_i->getSemiAxis1();
                dist = vector_length(endpoint - ductCenter);
                if (EXTERNAL_DUCT_BC_RADIUS - dist < pc_i->getRadius()){
                    wallData = determineCircularWall(endpoint,EXTERNAL_DUCT_BC_RADIUS,-1);
                    capsuleWallCollisionCap({pc_i, wallData, (dfloat)EXTERNAL_DUCT_BC_RADIUS - (pc_i->getRadius() + dist), step, endpoint});
                }
                
                endpoint = pc_i->getSemiAxis2();
                dist = vector_length(endpoint - ductCenter);
                if( EXTERNAL_DUCT_BC_RADIUS - dist < pc_i->getRadius()){
                    wallData = determineCircularWall(endpoint,EXTERNAL_DUCT_BC_RADIUS,-1);
                    capsuleWallCollisionCap({pc_i, wallData, (dfloat)EXTERNAL_DUCT_BC_RADIUS - (pc_i->getRadius() + dist), step, endpoint});
                }
                break;
            case ELLIPSOID:
                dist = ellipsoidSegmentCollisionDistance(pc_i, dfloat3(DUCT_CENTER_X,DUCT_CENTER_Y,-EXTERNAL_DUCT_BC_RADIUS),dfloat3(DUCT_CENTER_X,DUCT_CENTER_Y,NZ_TOTAL+EXTERNAL_DUCT_BC_RADIUS), EXTERNAL_DUCT_BC_RADIUS,closestOnA,closestOnB,cr,-1,step);
                if(dist < 0){
                    wallData = determineCircularWall(closestOnA[0],EXTERNAL_DUCT_BC_RADIUS,-1);
                    ellipsoidCylinderCollision({pc_i, wallData,-dist,step},closestOnB,cr,dfloat3(DUCT_CENTER_X,DUCT_CENTER_Y,-EXTERNAL_DUCT_BC_RADIUS),dfloat3(DUCT_CENTER_X,DUCT_CENTER_Y,NZ_TOTAL+EXTERNAL_DUCT_BC_RADIUS),EXTERNAL_DUCT_BC_RADIUS,-1);
                }
                break;
            default:
                break;
        }  


    #endif //(EXTERNAL_DUCT_BC)
}

__device__
void checkCollisionWallsSphere(ParticleCenter* pc_i, unsigned int step) {
    const dfloat3 pos_i = pc_i->getPos();
    const dfloat radius = pc_i->getRadius();

    const int maxWalls = 6;
    Wall walls[maxWalls];
    int wallCount = 0;

    #ifdef BC_X_WALL
    walls[wallCount++] = wall(dfloat3(1, 0, 0), 0);
    walls[wallCount++] = wall(dfloat3(-1, 0, 0), (NX - 1));
    #endif

    #ifdef BC_Y_WALL
    walls[wallCount++] = wall(dfloat3(0, 1, 0), 0);
    walls[wallCount++] = wall(dfloat3(0, -1, 0), (NY - 1));
    #endif

    #ifdef BC_Z_WALL
    walls[wallCount++] = wall(dfloat3(0, 0, 1), 0);
    walls[wallCount++] = wall(dfloat3(0, 0, -1), (NZ_TOTAL - 1));
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
            sphereWallCollision({pc_i, walls[i], radius - distanceWall, step});
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
    walls[wallCount++] = wall(dfloat3(1, 0, 0), 0);
    walls[wallCount++] = wall(dfloat3(-1, 0, 0), (NX - 1));
    #endif

    #ifdef BC_Y_WALL
    walls[wallCount++] = wall(dfloat3(0, 1, 0), 0);
    walls[wallCount++] = wall(dfloat3(0, -1, 0), (NY - 1));
    #endif
    
    #ifdef BC_Z_WALL
    walls[wallCount++] = wall(dfloat3(0, 0, 1), 0);
    walls[wallCount++] = wall(dfloat3(0, 0, -1), (NZ - 1));
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
    walls[wallCount++] = wall(dfloat3(1, 0, 0), 0);
    walls[wallCount++] = wall(dfloat3(-1, 0, 0), NX - 1);
    #endif

    #ifdef BC_Y_WALL
    walls[wallCount++] = wall(dfloat3(0, 1, 0), 0);
    walls[wallCount++] = wall(dfloat3(0, -1, 0), NY - 1);
    #endif
    
    #ifdef BC_Z_WALL
    walls[wallCount++] = wall(dfloat3(0, 0, 1), 0);
    walls[wallCount++] = wall(dfloat3(0, 0, -1), NZ - 1);
    #endif

    // Loop through all defined walls and check for collision
    for (int i = 0; i < wallCount; ++i) {
        dfloat distanceWall = 0;
        dfloat3 contactPoint2[1];
        dfloat cr[1];

        distanceWall = ellipsoidWallCollisionDistance(pc_i, walls[i], contactPoint2, cr, step);
        if (distanceWall < 0) {
            ellipsoidWallCollision({pc_i, walls[i], -distanceWall, step, contactPoint2[0]}, cr);
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


__device__
void sphereSphereCollisionCheck(unsigned int column,unsigned int row,ParticleCenter* pc_i, ParticleCenter* pc_j, int step){
    dfloat gap = sphereSphereGap(pc_i, pc_j);
    if (gap < 0) {
        dfloat3 diff_pos = getDiffPeriodic(pc_i->getPos(), pc_j->getPos());
        dfloat mag_dist = vector_length(diff_pos);
        dfloat3 normal = (mag_dist != 0) ? (diff_pos / mag_dist) : dfloat3{0.0, 0.0, 0.0};

        CollisionContext ctx = {};
        ctx.pc_i = pc_i;
        ctx.pc_j = pc_j;
        ctx.displacement = -gap; // overlap
        ctx.step = step;
        ctx.partnerID = row;
        ctx.wall.normal = normal;
        sphereSphereCollision(ctx);
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
    dfloat minDist = 1E+37f;
    dfloat3 bestClosestOnA, bestClosestOnB;
    dfloat bestcr1, bestcr2;
    dfloat3 bestTranslation;
    dfloat dist;

    // The temporary arrays must be declared to pass to the called function
    dfloat3 closestOnA[1], closestOnB[1];
    dfloat cr1[1], cr2[1];

    for (int i = 0; i < NUM_PERIODIC_DOMAIN_OFFSET; ++i) {
        int dx = PERIODIC_DOMAIN_OFFSET[i][0];
        int dy = PERIODIC_DOMAIN_OFFSET[i][1];
        int dz = PERIODIC_DOMAIN_OFFSET[i][2];

        dfloat3 translation = dfloat3(dx * NX, dy * NY, dz * NZ);
        dist = ellipsoidEllipsoidCollisionDistance(pc_i, pc_j, closestOnA, closestOnB, cr1, cr2, translation, step);

        if (dist < minDist) {
            minDist = dist;
            bestClosestOnA = closestOnA[0];
            bestClosestOnB = closestOnB[0];
            bestcr1 = cr1[0];
            bestcr2 = cr2[0];
            bestTranslation = translation;
        }
    }

    closestOnA[0] = bestClosestOnA;
    closestOnB[0] = bestClosestOnB - bestTranslation;
    cr1[0] = bestcr1;
    cr2[0] = bestcr2;
    dist = minDist;

    if (dist < 0) {
        dfloat3 diff_pos = getDiffPeriodic(closestOnA[0], closestOnB[0]);
        dfloat mag_dist = vector_length(diff_pos);
        dfloat3 normal = (mag_dist != 0) ? (diff_pos / mag_dist) : dfloat3{0.0, 0.0, 0.0};

        CollisionContext ctx = {};
        ctx.pc_i = pc_i;
        ctx.pc_j = pc_j;
        ctx.displacement = dist;
        ctx.step = step;
        ctx.partnerID = row;
        ctx.wall.normal = normal;


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

__device__
dfloat ellipsoidSegmentCollisionDistance( ParticleCenter* pc_i, dfloat3 P1, dfloat3 P2, dfloat cRadius ,dfloat3 contactPoint1[1], dfloat3 contactPoint2[1], dfloat cr[1], int cyDir, unsigned int step){
    dfloat RE[3][3];
    dfloat R2[3][3];
    dfloat dist, error;
    dfloat3 new_sphere_center1, new_sphere_center2;
    dfloat3 closest_point1, closest_point2;


    //obtain semi-axis values
    dfloat a = vector_length(pc_i->getSemiAxis1() - pc_i->getPos());
    dfloat b = vector_length(pc_i->getSemiAxis2() - pc_i->getPos());
    dfloat c = vector_length(pc_i->getSemiAxis3() - pc_i->getPos());

    rotationMatrixFromVectors((pc_i->getSemiAxis1() - pc_i->getPos())/a,(pc_i->getSemiAxis2() - pc_i->getPos())/b,(pc_i->getSemiAxis3() - pc_i->getPos())/c,RE);


    //projection of center into segment
    dfloat3 proj = segmentProjection(pc_i->getPos(),P1,P2,cRadius,cyDir);
    dfloat3 dir = pc_i->getPos() - proj;
    dfloat3 t = ellipsoid_intersection(pc_i,RE,proj,dir,dfloat3(0,0,0));
    dfloat3 inter1 = proj + t.x*dir;
    dfloat3 inter2 = proj + t.y*dir;
    

    if (vector_length(inter1 - proj) < vector_length(inter2 - proj)){
        closest_point2 = inter1;
    }else{
        closest_point2 = inter2;
    }    

    dfloat r = 3; //TODO: FIND A BETTER WAY TI DETERMINE IT

    //compute normal vector at intersection
    dfloat3 normal2 = ellipsoid_normal(pc_i,RE,closest_point2,cr,dfloat3(0,0,0));


    //Compute the centers of the spheres in the opposite direction of the normals
    dfloat3 sphere_center2 = closest_point2 - r * normal2;

    //Iteration loop
    dfloat max_iters = 20;
    dfloat tolerance = 1e-3;
    for(int i = 0; i< max_iters;i++){
        proj = segmentProjection(sphere_center2,P1,P2,cRadius, cyDir);
        dir = sphere_center2 - proj;
        t = ellipsoid_intersection(pc_i,RE,proj,dir,dfloat3(0,0,0));

        inter1 = proj + t.x*dir;
        inter2 = proj + t.y*dir;

        if (vector_length(inter1 - proj) < vector_length(inter2 - proj)){
            closest_point2 = inter1;
        }else{
            closest_point2 = inter2;
        }        

        normal2 = ellipsoid_normal(pc_i,RE,closest_point2,cr,dfloat3(0,0,0));
        new_sphere_center2 = closest_point2 - r * normal2;

        error = vector_length(new_sphere_center2 - sphere_center2);
        if (error < tolerance ){
            break;      
        }else{
            //update values
            sphere_center2 = new_sphere_center2;
        }
    }


    contactPoint1[0] = proj;
    contactPoint2[0] = closest_point2;
    dist = vector_length(sphere_center2 - proj) - r;
    return dist;
}

#endif //PARTICLE_MODEL