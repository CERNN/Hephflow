multiplyTerm = rho * W0;
pics2 = 1.0_df - cs2 * (mxx + myy + mzz);

const dfloat mxxy = ux * mxy + uy * mxx - ux * ux * uy;
const dfloat mxxz = ux * mxz + uz * mxx - ux * ux * uz;
const dfloat mxyy = ux * myy + uy * mxy - ux * uy * uy;
const dfloat mxzz = ux * mzz + uz * mxz - ux * uz * uz;
const dfloat myzz = uy * mzz + uz * myz - uy * uz * uz;
const dfloat myyz = uy * myz + uz * myy - uy * uy * uz;
const dfloat mxyz = ux * myz + uy * mxz + uz * mxy - 2 * ux * uy * uz;

const dfloat mxxyy = (ux * ux * myy + ux * uy * mxy + uy * mxxy - 3.0_df * ux * ux * uy * uy/2)/2;
const dfloat mxxzz = (ux * ux * mzz + ux * uz * mxz + uz * mxxz - 3.0_df * ux * ux * uz * uz/2)/2;
const dfloat myyzz = (uy * uy * mzz + uy * uz * myz + uz * myyz - 3.0_df * uy * uy * uz * uz/2)/2;

const dfloat mxyzz = (ux * uy * mzz) + (ux * uz * myz) + (uy * uz * mxz) / 2 + (uz * mxyz) - 3.0_df * (ux * uy * uz * uz) / 2;
const dfloat mxyyz = (ux * uy * myz) + (uy * uy * mxz) + (    uz * mxyy) / 2               - 3.0_df * (ux * uy * uy * uz) / 2;
const dfloat mxxyz = (ux * uy * mxz) + (    uz * mxxy) + (ux * ux * myz) / 2               - 3.0_df * (ux * ux * uy * uz) / 2;

const dfloat cs4 = cs2 * cs2;

const dfloat one_minus_cs2 = 1.0_df - cs2;
const dfloat one_minus_cs2_square = one_minus_cs2 * one_minus_cs2;

pop[0] = multiplyTerm * (pics2 + cs4 * mxxyy + cs4 * mxxzz + cs4 * myyzz);
multiplyTerm = rho * W1;
pop[1] = multiplyTerm * (pics2 + ux + mxx - cs2 * mxyy - cs2 * mxzz - one_minus_cs2 * cs2 * mxxyy - one_minus_cs2 * cs2 * mxxzz + cs4 * myyzz);
pop[2] = multiplyTerm * (pics2 - ux + mxx + cs2 * mxyy + cs2 * mxzz - one_minus_cs2 * cs2 * mxxyy - one_minus_cs2 * cs2 * mxxzz + cs4 * myyzz);
pop[3] = multiplyTerm * (pics2 + uy + myy - cs2 * mxxy - cs2 * myzz - one_minus_cs2 * cs2 * mxxyy + cs4 * mxxzz - one_minus_cs2 * cs2 * myyzz);
pop[4] = multiplyTerm * (pics2 - uy + myy + cs2 * mxxy + cs2 * myzz - one_minus_cs2 * cs2 * mxxyy + cs4 * mxxzz - one_minus_cs2 * cs2 * myyzz);
pop[5] = multiplyTerm * (pics2 + uz + mzz - cs2 * mxxz - cs2 * myyz + cs4 * mxxyy - one_minus_cs2 * cs2 * mxxzz - one_minus_cs2 * cs2 * myyzz);
pop[6] = multiplyTerm * (pics2 - uz + mzz + cs2 * mxxz + cs2 * myyz + cs4 * mxxyy - one_minus_cs2 * cs2 * mxxzz - one_minus_cs2 * cs2 * myyzz);
multiplyTerm = rho * W2;
pop[ 7] = multiplyTerm * (pics2 + ux + uy + mxx + myy + mxy + one_minus_cs2 * mxxy + one_minus_cs2 * mxyy - cs2 * mxzz - cs2 * myzz +
                          one_minus_cs2_square * mxxyy - one_minus_cs2 * cs2 * mxxzz - one_minus_cs2 * cs2 * myyzz - cs2 * mxyzz);
pop[ 8] = multiplyTerm * (pics2 - ux - uy + mxx + myy + mxy - one_minus_cs2 * mxxy - one_minus_cs2 * mxyy + cs2 * mxzz + cs2 * myzz +
                          one_minus_cs2_square * mxxyy - one_minus_cs2 * cs2 * mxxzz - one_minus_cs2 * cs2 * myyzz - cs2 * mxyzz);
pop[ 9] = multiplyTerm * (pics2 + ux + uz + mxx + mzz + mxz +one_minus_cs2 * mxxz - cs2 * mxyy + one_minus_cs2 * mxzz - cs2 * myyz -
                          one_minus_cs2 * cs2 * mxxyy + one_minus_cs2_square * mxxzz - one_minus_cs2 * cs2 * myyzz - cs2 * mxyyz);
pop[10] = multiplyTerm * (pics2 - ux - uz + mxx + mzz + mxz - one_minus_cs2 * mxxz + cs2 * mxyy - one_minus_cs2 * mxzz + cs2 * myyz -
                          one_minus_cs2 * cs2 * mxxyy + one_minus_cs2_square * mxxzz - one_minus_cs2 * cs2 * myyzz - cs2 * mxyyz);
pop[11] = multiplyTerm * (pics2 + uy + uz + myy + mzz + myz - cs2 * mxxy - cs2 * mxxz + one_minus_cs2 * myzz + one_minus_cs2 * myyz -
                          one_minus_cs2 * cs2 * mxxyy - one_minus_cs2 * cs2 * mxxzz + one_minus_cs2_square * myyzz - cs2 * mxxyz);
pop[12] = multiplyTerm * (pics2 - uy - uz + myy + mzz + myz + cs2 * mxxy + cs2 * mxxz - one_minus_cs2 * myzz - one_minus_cs2 * myyz -
                          one_minus_cs2 * cs2 * mxxyy - one_minus_cs2 * cs2 * mxxzz + one_minus_cs2_square * myyzz - cs2 * mxxyz);
pop[13] = multiplyTerm * (pics2 + ux - uy + mxx + myy - mxy - one_minus_cs2 * mxxy + one_minus_cs2 * mxyy - cs2 * mxzz + cs2 * myzz +
                          one_minus_cs2_square * mxxyy - one_minus_cs2 * cs2 * mxxzz - one_minus_cs2 * cs2 * myyzz + cs2 * mxyzz);
pop[14] = multiplyTerm * (pics2 - ux + uy + mxx + myy - mxy + one_minus_cs2 * mxxy - one_minus_cs2 * mxyy + cs2 * mxzz - cs2 * myzz +
                          one_minus_cs2_square * mxxyy - one_minus_cs2 * cs2 * mxxzz - one_minus_cs2 * cs2 * myyzz + cs2 * mxyzz);
pop[15] = multiplyTerm * (pics2 + ux - uz + mxx + mzz - mxz - one_minus_cs2 * mxxz - cs2 * mxyy + one_minus_cs2 * mxzz + cs2 * myyz -
                          one_minus_cs2 * cs2 * mxxyy + one_minus_cs2_square * mxxzz - one_minus_cs2 * cs2 * myyzz + cs2 * mxyyz);
pop[16] = multiplyTerm * (pics2 - ux + uz + mxx + mzz - mxz + one_minus_cs2 * mxxz + cs2 * mxyy - one_minus_cs2 * mxzz - cs2 * myyz -
                          one_minus_cs2 * cs2 * mxxyy + one_minus_cs2_square * mxxzz - one_minus_cs2 * cs2 * myyzz + cs2 * mxyyz);
pop[17] = multiplyTerm * (pics2 + uy - uz + myy + mzz - myz - cs2 * mxxy + cs2 * mxxz + one_minus_cs2 * myzz - one_minus_cs2 * myyz -
                          one_minus_cs2 * cs2 * mxxyy - one_minus_cs2 * cs2 * mxxzz + one_minus_cs2_square * myyzz + cs2 * mxxyz);
pop[18] = multiplyTerm * (pics2 - uy + uz + myy + mzz - myz + cs2 * mxxy - cs2 * mxxz - one_minus_cs2 * myzz + one_minus_cs2 * myyz -
                          one_minus_cs2 * cs2 * mxxyy - one_minus_cs2 * cs2 * mxxzz + one_minus_cs2_square * myyzz + cs2 * mxxyz);
#ifdef D3Q27
multiplyTerm = rhoVar * W3;
multiplyTerm = rho * W3;
pop[19] = multiplyTerm * (pics2 + ux + uy + uz + mxx + myy + mzz + (mxy + mxz + myz) +
                            one_minus_cs2 * mxxy + one_minus_cs2 * mxxz + one_minus_cs2 * mxyy + one_minus_cs2 * mxzz +
                            one_minus_cs2 * myzz + one_minus_cs2 * myyz + one_minus_cs2 * mxyz +
                            one_minus_cs2_square * mxxyy + one_minus_cs2_square * mxxzz + one_minus_cs2_square * myyzz +
                            one_minus_cs2 * mxyzz + one_minus_cs2 * mxyyz + one_minus_cs2 * mxxyz);
pop[20] = multiplyTerm * (pics2 - ux - uy - uz + mxx + myy + mzz + (mxy + mxz + myz) -
                            one_minus_cs2 * mxxy - one_minus_cs2 * mxxz - one_minus_cs2 * mxyy - one_minus_cs2 * mxzz -
                            one_minus_cs2 * myzz - one_minus_cs2 * myyz - one_minus_cs2 * mxyz +
                            one_minus_cs2_square * mxxyy + one_minus_cs2_square * mxxzz + one_minus_cs2_square * myyzz +
                            one_minus_cs2 * mxyzz + one_minus_cs2 * mxyyz + one_minus_cs2 * mxxyz);
pop[21] = multiplyTerm * (pics2 + ux + uy - uz + mxx + myy + mzz + (mxy - mxz - myz) +
                            one_minus_cs2 * mxxy - one_minus_cs2 * mxxz + one_minus_cs2 * mxyy + one_minus_cs2 * mxzz +
                            one_minus_cs2 * myzz - one_minus_cs2 * myyz - one_minus_cs2 * mxyz +
                            one_minus_cs2_square * mxxyy + one_minus_cs2_square * mxxzz + one_minus_cs2_square * myyzz +
                            one_minus_cs2 * mxyzz - one_minus_cs2 * mxyyz - one_minus_cs2 * mxxyz);
pop[22] = multiplyTerm * (pics2 - ux - uy + uz + mxx + myy + mzz + (mxy - mxz - myz) -
                            one_minus_cs2 * mxxy + one_minus_cs2 * mxxz - one_minus_cs2 * mxyy - one_minus_cs2 * mxzz -
                            one_minus_cs2 * myzz + one_minus_cs2 * myyz + one_minus_cs2 * mxyz +
                            one_minus_cs2_square * mxxyy + one_minus_cs2_square * mxxzz + one_minus_cs2_square * myyzz +
                            one_minus_cs2 * mxyzz - one_minus_cs2 * mxyyz - one_minus_cs2 * mxxyz);
pop[23] = multiplyTerm * (pics2 + ux - uy + uz + mxx + myy + mzz - (mxy - mxz + myz) -
                            one_minus_cs2 * mxxy + one_minus_cs2 * mxxz + one_minus_cs2 * mxyy + one_minus_cs2 * mxzz -
                            one_minus_cs2 * myzz + one_minus_cs2 * myyz - one_minus_cs2 * mxyz +
                            one_minus_cs2_square * mxxyy + one_minus_cs2_square * mxxzz + one_minus_cs2_square * myyzz -
                            one_minus_cs2 * mxyzz + one_minus_cs2 * mxyyz - one_minus_cs2 * mxxyz);
pop[24] = multiplyTerm * (pics2 - ux + uy - uz + mxx + myy + mzz - (mxy - mxz + myz) +
                            one_minus_cs2 * mxxy - one_minus_cs2 * mxxz - one_minus_cs2 * mxyy - one_minus_cs2 * mxzz +
                            one_minus_cs2 * myzz - one_minus_cs2 * myyz + one_minus_cs2 * mxyz +
                            one_minus_cs2_square * mxxyy + one_minus_cs2_square * mxxzz + one_minus_cs2_square * myyzz -
                            one_minus_cs2 * mxyzz + one_minus_cs2 * mxyyz - one_minus_cs2 * mxxyz);
pop[25] = multiplyTerm * (pics2 - ux + uy + uz + mxx + myy + mzz - (mxy + mxz - myz) +
                            one_minus_cs2 * mxxy + one_minus_cs2 * mxxz - one_minus_cs2 * mxyy - one_minus_cs2 * mxzz +
                            one_minus_cs2 * myzz + one_minus_cs2 * myyz - one_minus_cs2 * mxyz +
                            one_minus_cs2_square * mxxyy + one_minus_cs2_square * mxxzz + one_minus_cs2_square * myyzz -
                            one_minus_cs2 * mxyzz - one_minus_cs2 * mxyyz + one_minus_cs2 * mxxyz);
pop[26] = multiplyTerm * (pics2 + ux - uy - uz + mxx + myy + mzz - (mxy + mxz - myz) -
                            one_minus_cs2 * mxxy - one_minus_cs2 * mxxz + one_minus_cs2 * mxyy + one_minus_cs2 * mxzz -
                            one_minus_cs2 * myzz - one_minus_cs2 * myyz + one_minus_cs2 * mxyz +
                            one_minus_cs2_square * mxxyy + one_minus_cs2_square * mxxzz + one_minus_cs2_square * myyzz -
                            one_minus_cs2 * mxyzz - one_minus_cs2 * mxyyz + one_minus_cs2 * mxxyz);
#endif //D3Q27