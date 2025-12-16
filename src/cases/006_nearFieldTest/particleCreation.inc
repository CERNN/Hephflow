dfloat3 pos[NUM_PARTICLES];
pos[0] = {(NX-1)/2.0,(NY-1)/2.0+10.1,(NZ-1)/2.0};
pos[1] = {(NX-1)/2.0,(NY-1)/2.0-10.1,(NZ-1)/2.0};
dfloat3 vel[NUM_PARTICLES];
vel[0] = {0.0, 0.0, 0.0};
vel[1] = {0.0, 0.0, 0.0};
dfloat3 w = {0.0, 0.0, 0.0};


for (int i = 0; i<NUM_PARTICLES;i++){
    ParticleCenter& center = centerStorage[i];

    dfloat3 sa = {10, 20, 30};

    this->pShape[i] = SPHERE;
    center.setPos(pos[i]);
    center.setPos_old(pos[i]);
    center.setVel(vel[i]);
    center.setVel_old(vel[i]);
    center.setDensity(1.166666);
    center.setDiameter(20);
    center.setSemiAxis1(sa);
    center.setMovable(true);
    center.setQ_pos(dfloat4(1,0,0,0));
        
    particles[i].setMethod(IBM);
    particles[i].setShape(&pShape[i]);
    particles[i].setPCenter(&center);
    particles[i].setCollideParticle(false);
    particles[i].setCollideWall(true);

}
