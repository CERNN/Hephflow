srand(12345);  // fixed seed for reproducibility


dfloat diameter = 20.0;
dfloat radius   = diameter / 2.0;

for (int i = 0; i < NUM_PARTICLES; i++) {
    ParticleCenter& center = centerStorage[i];
    dfloat3 pos;

    pos.x =(NX-1)/2.0;
    pos.y =(NY-1)/2.0;
    pos.z =(NZ-1)/2.0;

    dfloat3 vel = {0.0, 0.0, 0.0};
    dfloat3 sa  = {10, 20, 30};

    this->pShape[i] = SPHERE;
    center.setPos(pos);
    center.setPos_old(pos);
    center.setVel(vel);
    center.setVel_old(vel);
    center.setDensity(1.166666);
    center.setDiameter(diameter);
    center.setSemiAxis1(sa);
    center.setMovable(true);
    center.setQ_pos(dfloat4(1,0,0,0));
        
    particles[i].setMethod(IBM);
    particles[i].setShape(&pShape[i]);
    particles[i].setPCenter(&center);
    particles[i].setCollideParticle(false);
    particles[i].setCollideWall(true);
}