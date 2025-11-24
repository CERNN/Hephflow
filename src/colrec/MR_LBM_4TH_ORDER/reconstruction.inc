    multiplyTerm = rho * W0;
    pics2 = static_cast<dfloat>(1.0) - cs2 * (mxx + myy + mzz);

    const dfloat mxxy = ux * mxy + uy * mxx - ux * ux * uy;
    const dfloat mxxz = ux * mxz + uz * mxx - ux * ux * uz;
    const dfloat mxyy = ux * myy + uy * mxy - ux * uy * uy;
    const dfloat mxzz = ux * mzz + uz * mxz - ux * uz * uz;
    const dfloat myzz = uy * mzz + uz * myz - uy * uz * uz;
    const dfloat myyz = uy * myz + uz * myy - uy * uy * uz;
    const dfloat mxyz = ux * myz + uy * mxz + uz * mxy - static_cast<dfloat>(2) * ux * uy * uz;

    const dfloat mxxyy = static_cast<dfloat>(0.5) * (ux * ux * myy + ux * uy * mxy + uy * mxxy - static_cast<dfloat>(1.5) * ux * ux * uy * uy);
    const dfloat mxxzz = static_cast<dfloat>(0.5) * (ux * ux * mzz + ux * uz * mxz + uz * mxxz - static_cast<dfloat>(1.5) * ux * ux * uz * uz);
    const dfloat myyzz = static_cast<dfloat>(0.5) * (uy * uy * mzz + uy * uz * myz + uz * myyz - static_cast<dfloat>(1.5) * uy * uy * uz * uz);
    const dfloat mxyzz = static_cast<dfloat>(0.5) * (static_cast<dfloat>(2) * ux * uy * mzz + ux * uz * myz + uy * uz * mxz + uz * mxyz - static_cast<dfloat>(3) * ux * uy * uz * uz);
    const dfloat mxyyz = ux * uy * myz + static_cast<dfloat>(0.5) * uy * uy * mxz + uz * mxyy - static_cast<dfloat>(1.5) * ux * uy * uy * uz;
    const dfloat mxxyz = static_cast<dfloat>(0.5) * ux * ux * myz + ux * uy * mxz + uz * mxxy - static_cast<dfloat>(1.5) * ux * ux * uy * uz;

    const dfloat cs4 = cs2 * cs2;

    const dfloat one_minus_cs2 = static_cast<dfloat>(1) - cs2;
    const dfloat one_minus_cs2_square = one_minus_cs2 * one_minus_cs2;

    // cix = 0; ciy = 0; ciz = 0;
    pop[0] = multiplyTerm * (pics2 +
                             // fourth-order
                             cs4 * mxxyy + cs4 * mxxzz + cs4 * myyzz);

    multiplyTerm = rho * W1;
    // cix = 1; ciy = 0; ciz = 0;
    pop[1] = multiplyTerm * (pics2 + ux + mxx -
                             // third-order
                             cs2 * mxyy - cs2 * mxzz -
                             // fourth-order
                             one_minus_cs2 * cs2 * mxxyy - one_minus_cs2 * cs2 * mxxzz + cs4 * myyzz);

    // cix = -1; ciy = 0; ciz = 0;
    pop[2] = multiplyTerm * (pics2 - ux + mxx +
                             // third-order
                             cs2 * mxyy + cs2 * mxzz -
                             // fourth-order
                             one_minus_cs2 * cs2 * mxxyy - one_minus_cs2 * cs2 * mxxzz + cs4 * myyzz);

    // cix = 0; ciy = 1; ciz = 0;
    pop[3] = multiplyTerm * (pics2 + uy + myy -
                             // third-order
                             cs2 * mxxy - cs2 * myzz -
                             // fourth-order
                             one_minus_cs2 * cs2 * mxxyy + cs4 * mxxzz - one_minus_cs2 * cs2 * myyzz);

    // cix = 0; ciy = -1; ciz = 0;
    pop[4] = multiplyTerm * (pics2 - uy + myy +
                             // third-order
                             cs2 * mxxy + cs2 * myzz -
                             // fourth-order
                             one_minus_cs2 * cs2 * mxxyy + cs4 * mxxzz - one_minus_cs2 * cs2 * myyzz);

    // cix = 0; ciy = 0; ciz = 1;
    pop[5] = multiplyTerm * (pics2 + uz + mzz -
                             // third-order
                             cs2 * mxxz - cs2 * myyz +
                             // fourth-order
                             cs4 * mxxyy - one_minus_cs2 * cs2 * mxxzz - one_minus_cs2 * cs2 * myyzz);

    // cix = 0; ciy = 0; ciz = -1;
    pop[6] = multiplyTerm * (pics2 - uz + mzz +
                             // third-order
                             cs2 * mxxz + cs2 * myyz +
                             // fourth-order
                             cs4 * mxxyy - one_minus_cs2 * cs2 * mxxzz - one_minus_cs2 * cs2 * myyzz);

    multiplyTerm = rho * W2;

    // cix = 1; ciy = 1; ciz = 0;
    pop[7] = multiplyTerm * (pics2 + ux + uy + mxx + myy + mxy +
                             // third-order
                             one_minus_cs2 * mxxy + one_minus_cs2 * mxyy - cs2 * mxzz - cs2 * myzz +
                             // fourth-order
                             one_minus_cs2_square * mxxyy - one_minus_cs2 * cs2 * mxxzz - one_minus_cs2 * cs2 * myyzz -
                             cs2 * mxyzz);

    // cix = -1; ciy = -1; ciz = 0;
    pop[8] = multiplyTerm * (pics2 - ux - uy + mxx + myy + mxy -
                             // third-order
                             one_minus_cs2 * mxxy - one_minus_cs2 * mxyy + cs2 * mxzz + cs2 * myzz +
                             // fourth-order
                             one_minus_cs2_square * mxxyy - one_minus_cs2 * cs2 * mxxzz - one_minus_cs2 * cs2 * myyzz -
                             cs2 * mxyzz);

    // cix = 1; ciy = 0; ciz = 1;
    pop[9] = multiplyTerm * (pics2 + ux + uz + mxx + mzz + mxz +
                             // third-order
                             one_minus_cs2 * mxxz - cs2 * mxyy + one_minus_cs2 * mxzz - cs2 * myyz -
                             // fourth-order
                             one_minus_cs2 * cs2 * mxxyy + one_minus_cs2_square * mxxzz - one_minus_cs2 * cs2 * myyzz -
                             cs2 * mxyyz);

    // cix = -1; ciy = 0; ciz = -1;
    pop[10] = multiplyTerm * (pics2 - ux - uz + mxx + mzz + mxz -
                              // third-order
                              one_minus_cs2 * mxxz + cs2 * mxyy - one_minus_cs2 * mxzz + cs2 * myyz -
                              // fourth-order
                              one_minus_cs2 * cs2 * mxxyy + one_minus_cs2_square * mxxzz - one_minus_cs2 * cs2 * myyzz -
                              cs2 * mxyyz);

    // cix = 0; ciy = 1; ciz = 1;
    pop[11] = multiplyTerm * (pics2 + uy + uz + myy + mzz + myz -
                              // third-order
                              cs2 * mxxy - cs2 * mxxz + one_minus_cs2 * myzz + one_minus_cs2 * myyz -
                              // fourth-order
                              one_minus_cs2 * cs2 * mxxyy - one_minus_cs2 * cs2 * mxxzz + one_minus_cs2_square * myyzz -
                              cs2 * mxxyz);

    // cix = 0; ciy = -1; ciz = -1;
    pop[12] = multiplyTerm * (pics2 - uy - uz + myy + mzz + myz +
                              // third-order
                              cs2 * mxxy + cs2 * mxxz - one_minus_cs2 * myzz - one_minus_cs2 * myyz -
                              // fourth-order
                              one_minus_cs2 * cs2 * mxxyy - one_minus_cs2 * cs2 * mxxzz + one_minus_cs2_square * myyzz -
                              cs2 * mxxyz);

    // cix = 1; ciy = -1; ciz = 0;
    pop[13] = multiplyTerm * (pics2 + ux - uy + mxx + myy - mxy -
                              // third-order
                              one_minus_cs2 * mxxy + one_minus_cs2 * mxyy - cs2 * mxzz + cs2 * myzz +
                              // fourth-order
                              one_minus_cs2_square * mxxyy - one_minus_cs2 * cs2 * mxxzz - one_minus_cs2 * cs2 * myyzz +
                              cs2 * mxyzz);

    // cix = -1; ciy = 1; ciz = 0;
    pop[14] = multiplyTerm * (pics2 - ux + uy + mxx + myy - mxy +
                              // third-order
                              one_minus_cs2 * mxxy - one_minus_cs2 * mxyy + cs2 * mxzz - cs2 * myzz +
                              // fourth-order
                              one_minus_cs2_square * mxxyy - one_minus_cs2 * cs2 * mxxzz - one_minus_cs2 * cs2 * myyzz +
                              cs2 * mxyzz);

    // cix = 1; ciy = 0; ciz = -1;
    pop[15] = multiplyTerm * (pics2 + ux - uz + mxx + mzz - mxz -
                              // third-order
                              one_minus_cs2 * mxxz - cs2 * mxyy + one_minus_cs2 * mxzz + cs2 * myyz -
                              // fourth-order
                              one_minus_cs2 * cs2 * mxxyy + one_minus_cs2_square * mxxzz - one_minus_cs2 * cs2 * myyzz +
                              cs2 * mxyyz);

    // cix = -1; ciy = 0; ciz = 1;
    pop[16] = multiplyTerm * (pics2 - ux + uz + mxx + mzz - mxz +
                              // third-order
                              one_minus_cs2 * mxxz + cs2 * mxyy - one_minus_cs2 * mxzz - cs2 * myyz -
                              // fourth-order
                              one_minus_cs2 * cs2 * mxxyy + one_minus_cs2_square * mxxzz - one_minus_cs2 * cs2 * myyzz +
                              cs2 * mxyyz);

    // cix = 0; ciy = 1; ciz = -1;
    pop[17] = multiplyTerm * (pics2 + uy - uz + myy + mzz - myz -
                              // third-order
                              cs2 * mxxy + cs2 * mxxz + one_minus_cs2 * myzz - one_minus_cs2 * myyz -
                              // fourth-order
                              one_minus_cs2 * cs2 * mxxyy - one_minus_cs2 * cs2 * mxxzz + one_minus_cs2_square * myyzz +
                              cs2 * mxxyz);

    // cix = 0; ciy = -1; ciz = 1;
    pop[18] = multiplyTerm * (pics2 - uy + uz + myy + mzz - myz +
                              // third-order
                              cs2 * mxxy - cs2 * mxxz - one_minus_cs2 * myzz + one_minus_cs2 * myyz -
                              // fourth-order
                              one_minus_cs2 * cs2 * mxxyy - one_minus_cs2 * cs2 * mxxzz + one_minus_cs2_square * myyzz +
                              cs2 * mxxyz);
    #ifdef D3Q27
    multiplyTerm = rhoVar * W3;
    multiplyTerm = rho * W3;
    // cix = 1; ciy = 1; ciz = 1;
    pop[19] = multiplyTerm * (pics2 + ux + uy + uz + mxx + myy + mzz + (mxy + mxz + myz) +
                              // third-order
                              one_minus_cs2 * mxxy + one_minus_cs2 * mxxz + one_minus_cs2 * mxyy + one_minus_cs2 * mxzz +
                              one_minus_cs2 * myzz + one_minus_cs2 * myyz + one_minus_cs2 * mxyz +
                              // fourth-order
                              one_minus_cs2_square * mxxyy + one_minus_cs2_square * mxxzz + one_minus_cs2_square * myyzz +
                              one_minus_cs2 * mxyzz + one_minus_cs2 * mxyyz + one_minus_cs2 * mxxyz);

    // cix -1; ciy = -1; ciz = -1;
    pop[20] = multiplyTerm * (pics2 - ux - uy - uz + mxx + myy + mzz + (mxy + mxz + myz) -
                              // third-order
                              one_minus_cs2 * mxxy - one_minus_cs2 * mxxz - one_minus_cs2 * mxyy - one_minus_cs2 * mxzz -
                              one_minus_cs2 * myzz - one_minus_cs2 * myyz - one_minus_cs2 * mxyz +
                              // fourth-order
                              one_minus_cs2_square * mxxyy + one_minus_cs2_square * mxxzz + one_minus_cs2_square * myyzz +
                              one_minus_cs2 * mxyzz + one_minus_cs2 * mxyyz + one_minus_cs2 * mxxyz);

    // cix = 1; ciy = 1; ciz = -1;
    pop[21] = multiplyTerm * (pics2 + ux + uy - uz + mxx + myy + mzz + (mxy - mxz - myz) +
                              // third-order
                              one_minus_cs2 * mxxy - one_minus_cs2 * mxxz + one_minus_cs2 * mxyy + one_minus_cs2 * mxzz +
                              one_minus_cs2 * myzz - one_minus_cs2 * myyz - one_minus_cs2 * mxyz +
                              // fourth-order
                              one_minus_cs2_square * mxxyy + one_minus_cs2_square * mxxzz + one_minus_cs2_square * myyzz +
                              one_minus_cs2 * mxyzz - one_minus_cs2 * mxyyz - one_minus_cs2 * mxxyz);

    // cix = -1; ciy = -1; ciz = 1;
    pop[22] = multiplyTerm * (pics2 - ux - uy + uz + mxx + myy + mzz + (mxy - mxz - myz) -
                              // third-order
                              one_minus_cs2 * mxxy + one_minus_cs2 * mxxz - one_minus_cs2 * mxyy - one_minus_cs2 * mxzz -
                              one_minus_cs2 * myzz + one_minus_cs2 * myyz + one_minus_cs2 * mxyz +
                              // fourth-order
                              one_minus_cs2_square * mxxyy + one_minus_cs2_square * mxxzz + one_minus_cs2_square * myyzz +
                              one_minus_cs2 * mxyzz - one_minus_cs2 * mxyyz - one_minus_cs2 * mxxyz);

    // cix = 1; ciy = -1; ciz = 1;
    pop[23] = multiplyTerm * (pics2 + ux - uy + uz + mxx + myy + mzz - (mxy - mxz + myz) -
                              // third-order
                              one_minus_cs2 * mxxy + one_minus_cs2 * mxxz + one_minus_cs2 * mxyy + one_minus_cs2 * mxzz -
                              one_minus_cs2 * myzz + one_minus_cs2 * myyz - one_minus_cs2 * mxyz +
                              // fourth-order
                              one_minus_cs2_square * mxxyy + one_minus_cs2_square * mxxzz + one_minus_cs2_square * myyzz -
                              one_minus_cs2 * mxyzz + one_minus_cs2 * mxyyz - one_minus_cs2 * mxxyz);

    // cix = -1; ciy = 1; ciz = -1;
    pop[24] = multiplyTerm * (pics2 - ux + uy - uz + mxx + myy + mzz - (mxy - mxz + myz) +
                              // third-order
                              one_minus_cs2 * mxxy - one_minus_cs2 * mxxz - one_minus_cs2 * mxyy - one_minus_cs2 * mxzz +
                              one_minus_cs2 * myzz - one_minus_cs2 * myyz + one_minus_cs2 * mxyz +
                              // fourth-order
                              one_minus_cs2_square * mxxyy + one_minus_cs2_square * mxxzz + one_minus_cs2_square * myyzz -
                              one_minus_cs2 * mxyzz + one_minus_cs2 * mxyyz - one_minus_cs2 * mxxyz);

    // cix = -1; ciy = 1; ciz = 1;
    pop[25] = multiplyTerm * (pics2 - ux + uy + uz + mxx + myy + mzz - (mxy + mxz - myz) +
                              // third-order
                              one_minus_cs2 * mxxy + one_minus_cs2 * mxxz - one_minus_cs2 * mxyy - one_minus_cs2 * mxzz +
                              one_minus_cs2 * myzz + one_minus_cs2 * myyz - one_minus_cs2 * mxyz +
                              // fourth-order
                              one_minus_cs2_square * mxxyy + one_minus_cs2_square * mxxzz + one_minus_cs2_square * myyzz -
                              one_minus_cs2 * mxyzz - one_minus_cs2 * mxyyz + one_minus_cs2 * mxxyz);

    // cix = 1; ciy = -1; ciz = -1;
    pop[26] = multiplyTerm * (pics2 + ux - uy - uz + mxx + myy + mzz - (mxy + mxz - myz) -
                              // third-order
                              one_minus_cs2 * mxxy - one_minus_cs2 * mxxz + one_minus_cs2 * mxyy + one_minus_cs2 * mxzz -
                              one_minus_cs2 * myzz - one_minus_cs2 * myyz + one_minus_cs2 * mxyz +
                              // fourth-order
                              one_minus_cs2_square * mxxyy + one_minus_cs2_square * mxxzz + one_minus_cs2_square * myyzz -
                              one_minus_cs2 * mxyzz - one_minus_cs2 * mxyyz + one_minus_cs2 * mxxyz);
    #endif //D3Q27