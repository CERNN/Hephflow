    multiplyTerm = rho * W0;
    pics2 = static_cast<dfloat>(1.0) - cs2 * (mxx + myy + mzz);

    const dfloat mxxy = ux * mxy + uy * mxx - ux * ux * uy;
    const dfloat mxxz = ux * mxz + uz * mxx - ux * ux * uz;
    const dfloat mxyy = ux * myy + uy * mxy - ux * uy * uy;
    const dfloat mxzz = ux * mzz + uz * mxz - ux * uz * uz;
    const dfloat myzz = uy * mzz + uz * myz - uy * uz * uz;
    const dfloat myyz = uy * myz + uz * myy - uy * uy * uz;
    const dfloat mxyz = ux * myz + uy * mxz + uz * mxy - static_cast<dfloat>(2) * ux * uy * uz;

    const dfloat one_minus_cs2 = static_cast<dfloat>(1) - cs2;

    // cix = 0; ciy = 0; ciz = 0;
    pop[0] = multiplyTerm * (pics2);

    multiplyTerm = rho * W1;
    // cix = 1; ciy = 0; ciz = 0;
    pop[1] = multiplyTerm * (pics2 + ux + mxx -
                             // third-order
                             cs2 * mxyy - cs2 * mxzz);

    // cix = -1; ciy = 0; ciz = 0;
    pop[2] = multiplyTerm * (pics2 - ux + mxx +
                             // third-order
                             cs2 * mxyy + cs2 * mxzz);

    // cix = 0; ciy = 1; ciz = 0;
    pop[3] = multiplyTerm * (pics2 + uy + myy -
                             // third-order
                             cs2 * mxxy - cs2 * myzz);

    // cix = 0; ciy = -1; ciz = 0;
    pop[4] = multiplyTerm * (pics2 - uy + myy +
                             // third-order
                             cs2 * mxxy + cs2 * myzz);

    // cix = 0; ciy = 0; ciz = 1;
    pop[5] = multiplyTerm * (pics2 + uz + mzz -
                             // third-order
                             cs2 * mxxz - cs2 * myyz);

    // cix = 0; ciy = 0; ciz = -1;
    pop[6] = multiplyTerm * (pics2 - uz + mzz +
                             // third-order
                             cs2 * mxxz + cs2 * myyz);

    multiplyTerm = rho * W2;

    // cix = 1; ciy = 1; ciz = 0;
    pop[7] = multiplyTerm * (pics2 + ux + uy + mxx + myy + mxy +
                             // third-order
                             one_minus_cs2 * mxxy + one_minus_cs2 * mxyy - cs2 * mxzz - cs2 * myzz);

    // cix = -1; ciy = -1; ciz = 0;
    pop[8] = multiplyTerm * (pics2 - ux - uy + mxx + myy + mxy -
                             // third-order
                             one_minus_cs2 * mxxy - one_minus_cs2 * mxyy + cs2 * mxzz + cs2 * myzz);

    // cix = 1; ciy = 0; ciz = 1;
    pop[9] = multiplyTerm * (pics2 + ux + uz + mxx + mzz + mxz +
                             // third-order
                             one_minus_cs2 * mxxz - cs2 * mxyy + one_minus_cs2 * mxzz - cs2 * myyz);

    // cix = -1; ciy = 0; ciz = -1;
    pop[10] = multiplyTerm * (pics2 - ux - uz + mxx + mzz + mxz -
                              // third-order
                              one_minus_cs2 * mxxz + cs2 * mxyy - one_minus_cs2 * mxzz + cs2 * myyz);

    // cix = 0; ciy = 1; ciz = 1;
    pop[11] = multiplyTerm * (pics2 + uy + uz + myy + mzz + myz -
                              // third-order
                              cs2 * mxxy - cs2 * mxxz + one_minus_cs2 * myzz + one_minus_cs2 * myyz);

    // cix = 0; ciy = -1; ciz = -1;
    pop[12] = multiplyTerm * (pics2 - uy - uz + myy + mzz + myz +
                              // third-order
                              cs2 * mxxy + cs2 * mxxz - one_minus_cs2 * myzz - one_minus_cs2 * myyz);

    // cix = 1; ciy = -1; ciz = 0;
    pop[13] = multiplyTerm * (pics2 + ux - uy + mxx + myy - mxy -
                              // third-order
                              one_minus_cs2 * mxxy + one_minus_cs2 * mxyy - cs2 * mxzz + cs2 * myzz);

    // cix = -1; ciy = 1; ciz = 0;
    pop[14] = multiplyTerm * (pics2 - ux + uy + mxx + myy - mxy +
                              // third-order
                              one_minus_cs2 * mxxy - one_minus_cs2 * mxyy + cs2 * mxzz - cs2 * myzz);

    // cix = 1; ciy = 0; ciz = -1;
    pop[15] = multiplyTerm * (pics2 + ux - uz + mxx + mzz - mxz -
                              // third-order
                              one_minus_cs2 * mxxz - cs2 * mxyy + one_minus_cs2 * mxzz + cs2 * myyz);

    // cix = -1; ciy = 0; ciz = 1;
    pop[16] = multiplyTerm * (pics2 - ux + uz + mxx + mzz - mxz +
                              // third-order
                              one_minus_cs2 * mxxz + cs2 * mxyy - one_minus_cs2 * mxzz - cs2 * myyz);

    // cix = 0; ciy = 1; ciz = -1;
    pop[17] = multiplyTerm * (pics2 + uy - uz + myy + mzz - myz -
                              // third-order
                              cs2 * mxxy + cs2 * mxxz + one_minus_cs2 * myzz - one_minus_cs2 * myyz);

    // cix = 0; ciy = -1; ciz = 1;
    pop[18] = multiplyTerm * (pics2 - uy + uz + myy + mzz - myz +
                              // third-order
                              cs2 * mxxy - cs2 * mxxz - one_minus_cs2 * myzz + one_minus_cs2 * myyz);
    #ifdef D3Q27
    multiplyTerm = rho * W3;
    // cix = 1; ciy = 1; ciz = 1;
    pop[19] = multiplyTerm * (pics2 + ux + uy + uz + mxx + myy + mzz + (mxy + mxz + myz) +
                              // third-order
                              one_minus_cs2 * mxxy + one_minus_cs2 * mxxz + one_minus_cs2 * mxyy + one_minus_cs2 * mxzz +
                              one_minus_cs2 * myzz + one_minus_cs2 * myyz + one_minus_cs2 * mxyz);

    // cix -1; ciy = -1; ciz = -1;
    pop[20] = multiplyTerm * (pics2 - ux - uy - uz + mxx + myy + mzz + (mxy + mxz + myz) -
                              // third-order
                              one_minus_cs2 * mxxy - one_minus_cs2 * mxxz - one_minus_cs2 * mxyy - one_minus_cs2 * mxzz -
                              one_minus_cs2 * myzz - one_minus_cs2 * myyz - one_minus_cs2 * mxyz);

    // cix = 1; ciy = 1; ciz = -1;
    pop[21] = multiplyTerm * (pics2 + ux + uy - uz + mxx + myy + mzz + (mxy - mxz - myz) +
                              // third-order
                              one_minus_cs2 * mxxy - one_minus_cs2 * mxxz + one_minus_cs2 * mxyy + one_minus_cs2 * mxzz +
                              one_minus_cs2 * myzz - one_minus_cs2 * myyz - one_minus_cs2 * mxyz);

    // cix = -1; ciy = -1; ciz = 1;
    pop[22] = multiplyTerm * (pics2 - ux - uy + uz + mxx + myy + mzz + (mxy - mxz - myz) -
                              // third-order
                              one_minus_cs2 * mxxy + one_minus_cs2 * mxxz - one_minus_cs2 * mxyy - one_minus_cs2 * mxzz -
                              one_minus_cs2 * myzz + one_minus_cs2 * myyz + one_minus_cs2 * mxyz);

    // cix = 1; ciy = -1; ciz = 1;
    pop[23] = multiplyTerm * (pics2 + ux - uy + uz + mxx + myy + mzz - (mxy - mxz + myz) -
                              // third-order
                              one_minus_cs2 * mxxy + one_minus_cs2 * mxxz + one_minus_cs2 * mxyy + one_minus_cs2 * mxzz -
                              one_minus_cs2 * myzz + one_minus_cs2 * myyz - one_minus_cs2 * mxyz);

    // cix = -1; ciy = 1; ciz = -1;
    pop[24] = multiplyTerm * (pics2 - ux + uy - uz + mxx + myy + mzz - (mxy - mxz + myz) +
                              // third-order
                              one_minus_cs2 * mxxy - one_minus_cs2 * mxxz - one_minus_cs2 * mxyy - one_minus_cs2 * mxzz +
                              one_minus_cs2 * myzz - one_minus_cs2 * myyz + one_minus_cs2 * mxyz);

    // cix = -1; ciy = 1; ciz = 1;
    pop[25] = multiplyTerm * (pics2 - ux + uy + uz + mxx + myy + mzz - (mxy + mxz - myz) +
                              // third-order
                              one_minus_cs2 * mxxy + one_minus_cs2 * mxxz - one_minus_cs2 * mxyy - one_minus_cs2 * mxzz +
                              one_minus_cs2 * myzz + one_minus_cs2 * myyz - one_minus_cs2 * mxyz);

    // cix = 1; ciy = -1; ciz = -1;
    pop[26] = multiplyTerm * (pics2 + ux - uy - uz + mxx + myy + mzz - (mxy + mxz - myz) -
                              // third-order
                              one_minus_cs2 * mxxy - one_minus_cs2 * mxxz + one_minus_cs2 * mxyy + one_minus_cs2 * mxzz -
                              one_minus_cs2 * myzz - one_minus_cs2 * myyz + one_minus_cs2 * mxyz);
    #endif //D3Q27