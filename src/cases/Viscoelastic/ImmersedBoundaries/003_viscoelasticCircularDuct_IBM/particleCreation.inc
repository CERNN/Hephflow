ParticleCenter& duct = centerStorage[0];
this->pShape[0] = PERIODIC_DUCT;
particles[0].setDuctAxisEndpoints(
    dfloat3(DUCT_CENTER_X, DUCT_CENTER_Y, 0.0_df),
    dfloat3(DUCT_CENTER_X, DUCT_CENTER_Y, (dfloat)NZ_TOTAL));
particles[0].setDuctRadii(DUCT_RADIUS_A, DUCT_RADIUS_B);
particles[0].setDuctStaggeredRings(DUCT_USE_STAGGERED_RINGS);
duct.setDensity(FLUID_DENSITY);
duct.setVel(dfloat3(0.0_df, 0.0_df, 0.0_df));
duct.setVel_old(dfloat3(0.0_df, 0.0_df, 0.0_df));
duct.setW(dfloat3(0.0_df, 0.0_df, 0.0_df));
duct.setW_old(dfloat3(0.0_df, 0.0_df, 0.0_df));
duct.setMovable(false);

particles[0].setMethod(IBM);
particles[0].setShape(&this->pShape[0]);
particles[0].setPCenter(&duct);
particles[0].setCollideParticle(false);
particles[0].setCollideWall(false);
