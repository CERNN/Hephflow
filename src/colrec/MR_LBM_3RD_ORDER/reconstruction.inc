    multiplyTerm = rho * W0;
    pics2 = 1.0 - cs2 * (mxx + myy + mzz);

    const dfloat mxxy = ux * mxy + uy * mxx - ux * ux * uy;
    const dfloat mxxz = ux * mxz + uz * mxx - ux * ux * uz;
    const dfloat mxyy = ux * myy + uy * mxy - ux * uy * uy;
    const dfloat mxzz = ux * mzz + uz * mxz - ux * uz * uz;
    const dfloat myzz = uy * mzz + uz * myz - uy * uz * uz;
    const dfloat myyz = uy * myz + uz * myy - uy * uy * uz;
    const dfloat mxyz = ux * myz + uy * mxz + uz * mxy - 2 * ux * uy * uz;

    const dfloat one_minus_cs2 = 1.0 - cs2;

    pop[ 0] = multiplyTerm * (pics2);
    multiplyTerm = rho * W1;
    pop[ 1] = multiplyTerm * (pics2 + ux + mxx - cs2 * mxyy - cs2 * mxzz);
    pop[ 2] = multiplyTerm * (pics2 - ux + mxx + cs2 * mxyy + cs2 * mxzz);
    pop[ 3] = multiplyTerm * (pics2 + uy + myy - cs2 * mxxy - cs2 * myzz);
    pop[ 4] = multiplyTerm * (pics2 - uy + myy + cs2 * mxxy + cs2 * myzz);
    pop[ 5] = multiplyTerm * (pics2 + uz + mzz - cs2 * mxxz - cs2 * myyz);
    pop[ 6] = multiplyTerm * (pics2 - uz + mzz + cs2 * mxxz + cs2 * myyz);
    multiplyTerm = rho * W2;
    pop[ 7] = multiplyTerm * (pics2 + ux + uy + mxx + myy + mxy + one_minus_cs2 * mxxy + one_minus_cs2 * mxyy - cs2 * mxzz - cs2 * myzz);
    pop[ 8] = multiplyTerm * (pics2 - ux - uy + mxx + myy + mxy - one_minus_cs2 * mxxy - one_minus_cs2 * mxyy + cs2 * mxzz + cs2 * myzz);
    pop[ 9] = multiplyTerm * (pics2 + ux + uz + mxx + mzz + mxz + one_minus_cs2 * mxxz - cs2 * mxyy + one_minus_cs2 * mxzz - cs2 * myyz);
    pop[10] = multiplyTerm * (pics2 - ux - uz + mxx + mzz + mxz - one_minus_cs2 * mxxz + cs2 * mxyy - one_minus_cs2 * mxzz + cs2 * myyz);
    pop[11] = multiplyTerm * (pics2 + uy + uz + myy + mzz + myz - cs2 * mxxy - cs2 * mxxz + one_minus_cs2 * myzz + one_minus_cs2 * myyz);
    pop[12] = multiplyTerm * (pics2 - uy - uz + myy + mzz + myz + cs2 * mxxy + cs2 * mxxz - one_minus_cs2 * myzz - one_minus_cs2 * myyz);
    pop[13] = multiplyTerm * (pics2 + ux - uy + mxx + myy - mxy - one_minus_cs2 * mxxy + one_minus_cs2 * mxyy - cs2 * mxzz + cs2 * myzz);
    pop[14] = multiplyTerm * (pics2 - ux + uy + mxx + myy - mxy + one_minus_cs2 * mxxy - one_minus_cs2 * mxyy + cs2 * mxzz - cs2 * myzz);
    pop[15] = multiplyTerm * (pics2 + ux - uz + mxx + mzz - mxz - one_minus_cs2 * mxxz - cs2 * mxyy + one_minus_cs2 * mxzz + cs2 * myyz);
    pop[16] = multiplyTerm * (pics2 - ux + uz + mxx + mzz - mxz + one_minus_cs2 * mxxz + cs2 * mxyy - one_minus_cs2 * mxzz - cs2 * myyz);
    pop[17] = multiplyTerm * (pics2 + uy - uz + myy + mzz - myz - cs2 * mxxy + cs2 * mxxz + one_minus_cs2 * myzz - one_minus_cs2 * myyz);
    pop[18] = multiplyTerm * (pics2 - uy + uz + myy + mzz - myz + cs2 * mxxy - cs2 * mxxz - one_minus_cs2 * myzz + one_minus_cs2 * myyz);
    #ifdef D3Q27
    multiplyTerm = rho * W3;
    pop[19] = multiplyTerm * (pics2 + ux + uy + uz + mxx + myy + mzz + (mxy + mxz + myz) + 
                              one_minus_cs2 * mxxy + one_minus_cs2 * mxxz + one_minus_cs2 * mxyy + one_minus_cs2 * mxzz +
                              one_minus_cs2 * myzz + one_minus_cs2 * myyz + one_minus_cs2 * mxyz);
    pop[20] = multiplyTerm * (pics2 - ux - uy - uz + mxx + myy + mzz + (mxy + mxz + myz) -
                              one_minus_cs2 * mxxy - one_minus_cs2 * mxxz - one_minus_cs2 * mxyy - one_minus_cs2 * mxzz -
                              one_minus_cs2 * myzz - one_minus_cs2 * myyz - one_minus_cs2 * mxyz);
    pop[21] = multiplyTerm * (pics2 + ux + uy - uz + mxx + myy + mzz + (mxy - mxz - myz) +
                              one_minus_cs2 * mxxy - one_minus_cs2 * mxxz + one_minus_cs2 * mxyy + one_minus_cs2 * mxzz +
                              one_minus_cs2 * myzz - one_minus_cs2 * myyz - one_minus_cs2 * mxyz);
    pop[22] = multiplyTerm * (pics2 - ux - uy + uz + mxx + myy + mzz + (mxy - mxz - myz) -
                              one_minus_cs2 * mxxy + one_minus_cs2 * mxxz - one_minus_cs2 * mxyy - one_minus_cs2 * mxzz -
                              one_minus_cs2 * myzz + one_minus_cs2 * myyz + one_minus_cs2 * mxyz);
    pop[23] = multiplyTerm * (pics2 + ux - uy + uz + mxx + myy + mzz - (mxy - mxz + myz) -
                              one_minus_cs2 * mxxy + one_minus_cs2 * mxxz + one_minus_cs2 * mxyy + one_minus_cs2 * mxzz -
                              one_minus_cs2 * myzz + one_minus_cs2 * myyz - one_minus_cs2 * mxyz);
    pop[24] = multiplyTerm * (pics2 - ux + uy - uz + mxx + myy + mzz - (mxy - mxz + myz) +
                              one_minus_cs2 * mxxy - one_minus_cs2 * mxxz - one_minus_cs2 * mxyy - one_minus_cs2 * mxzz +
                              one_minus_cs2 * myzz - one_minus_cs2 * myyz + one_minus_cs2 * mxyz);
    pop[25] = multiplyTerm * (pics2 - ux + uy + uz + mxx + myy + mzz - (mxy + mxz - myz) +
                              one_minus_cs2 * mxxy + one_minus_cs2 * mxxz - one_minus_cs2 * mxyy - one_minus_cs2 * mxzz +
                              one_minus_cs2 * myzz + one_minus_cs2 * myyz - one_minus_cs2 * mxyz);
    pop[26] = multiplyTerm * (pics2 + ux - uy - uz + mxx + myy + mzz - (mxy + mxz - myz) -
                              one_minus_cs2 * mxxy - one_minus_cs2 * mxxz + one_minus_cs2 * mxyy + one_minus_cs2 * mxzz -
                              one_minus_cs2 * myzz - one_minus_cs2 * myyz + one_minus_cs2 * mxyz);
    #endif //D3Q27