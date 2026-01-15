srand(12345);  // fixed seed for reproducibility


dfloat diameter = 20.0;
dfloat radius   = diameter / 2.0;

for (int i = 0; i < NUM_PARTICLES; i++) {
    ParticleCenter& center = centerStorage[i];
    dfloat3 pos;
    bool valid = false;

    while (!valid) {
        pos.x = (dfloat)((dfloat)rand() / RAND_MAX) * (NX - 1 - 2*radius) + radius;
        pos.y = (dfloat)((dfloat)rand() / RAND_MAX) * (NY - 1 - 2*radius) + radius;
        pos.z = (dfloat)((dfloat)rand() / RAND_MAX) * (NZ - 1 - 2*radius) + radius;
        
        valid = true;

        for (int j = 0; j < i; j++) {
            dfloat3 otherPos = centerStorage[j].getPos();
            dfloat dx = pos.x - otherPos.x;
            dfloat dy = pos.y - otherPos.y;
            dfloat dz = pos.z - otherPos.z;
            dfloat dist = sqrt(dx*dx + dy*dy + dz*dz);

            if (dist < diameter) {
                valid = false;
                break;
            }
        }
    }
        
    printf("Particle index %d Pos %f %f %f \n",i,pos.x,pos.y,pos.z);



    dfloat3 vel = {0.0, 0.0, 0.0};
    dfloat3 sa1 = {10, 0, 0};
    dfloat3 sa2 = {0, 10, 0};
    dfloat3 sa3 = {0, 0, 10};

    this->pShape[i] = SPHERE;
    center.setPos(pos);
    center.setPos_old(pos);
    center.setVel(vel);
    center.setVel_old(vel);
    center.setDensity(1.166666);
    center.setDiameter(diameter);
    center.setSemiAxis1(sa1);
    center.setSemiAxis2(sa2);
    center.setSemiAxis3(sa3);
    center.initializeSemiAxesFromCurrent(); // Precision fix: store original semi-axis positions
    center.setQ_cumulative_rot(dfloat4(0, 0, 0, 1)); // Initialize cumulative rotation to identity
    center.setMovable(true);
    center.setQ_pos(dfloat4(1,0,0,0));
        
    particles[i].setMethod(IBM);
    particles[i].setShape(&pShape[i]);
    particles[i].setPCenter(&center);
    particles[i].setCollideParticle(false);
    particles[i].setCollideWall(true);
}