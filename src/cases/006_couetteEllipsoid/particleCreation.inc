dfloat3 pos = {(NX-1)/2.0,(NY-1)/2.0,(NZ-1)/2.0};
dfloat3 vel = {0.0, 0.0, 0.0};
dfloat3 w = {0.0, 0.0, 0.0};

dfloat3 sa = {a_semi/2, b_semi/2, c_semi/2};




for (int i = 0; i<NUM_PARTICLES;i++){
    ParticleCenter& center = centerStorage[i];


    this->pShape[i] = ELLIPSOID;
    center.setPos(pos);
    center.setPos_old(pos);
    center.setVel(vel);
    center.setVel_old(vel);
    center.setDensity(1.0);
    dfloat3 sa1 = {10, 0, 0};
    dfloat3 sa2 = {0, 20, 0};
    dfloat3 sa3 = {0, 0, 30};
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
