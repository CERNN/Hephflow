{
const dfloat rho = rhoVar;
const dfloat ux = ux_t30;
const dfloat uy = uy_t30;
const dfloat uz = uz_t30;
const dfloat mxx = m_xx_t45;
const dfloat mxy = m_xy_t90;
const dfloat mxz = m_xz_t90;
const dfloat myy = m_yy_t45;
const dfloat myz = m_yz_t90;
const dfloat mzz = m_zz_t45;

multiplyTerm = rho * W0;
pics2 = 1.0_df - cs2 * (mxx + myy + mzz);

const dfloat mxxy = mrlbmRecursiveCoefficient<2,1,0>(ux,uy,uz,mxx,mxy,mxz,myy,myz,mzz);
const dfloat mxxz = mrlbmRecursiveCoefficient<2,0,1>(ux,uy,uz,mxx,mxy,mxz,myy,myz,mzz);
const dfloat mxyy = mrlbmRecursiveCoefficient<1,2,0>(ux,uy,uz,mxx,mxy,mxz,myy,myz,mzz);
const dfloat mxzz = mrlbmRecursiveCoefficient<1,0,2>(ux,uy,uz,mxx,mxy,mxz,myy,myz,mzz);
const dfloat myzz = mrlbmRecursiveCoefficient<0,1,2>(ux,uy,uz,mxx,mxy,mxz,myy,myz,mzz);
const dfloat myyz = mrlbmRecursiveCoefficient<0,2,1>(ux,uy,uz,mxx,mxy,mxz,myy,myz,mzz);
const dfloat mxyz = mrlbmRecursiveCoefficient<1,1,1>(ux,uy,uz,mxx,mxy,mxz,myy,myz,mzz);

const dfloat mxxyy = mrlbmRecursiveCoefficient<2,2,0>(ux,uy,uz,mxx,mxy,mxz,myy,myz,mzz);
const dfloat mxxzz = mrlbmRecursiveCoefficient<2,0,2>(ux,uy,uz,mxx,mxy,mxz,myy,myz,mzz);
const dfloat myyzz = mrlbmRecursiveCoefficient<0,2,2>(ux,uy,uz,mxx,mxy,mxz,myy,myz,mzz);
const dfloat mxyzz = mrlbmRecursiveCoefficient<1,1,2>(ux,uy,uz,mxx,mxy,mxz,myy,myz,mzz);
const dfloat mxyyz = mrlbmRecursiveCoefficient<1,2,1>(ux,uy,uz,mxx,mxy,mxz,myy,myz,mzz);
const dfloat mxxyz = mrlbmRecursiveCoefficient<2,1,1>(ux,uy,uz,mxx,mxy,mxz,myy,myz,mzz);

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
#ifdef D3Q27
#define MR_CORNER_RECONSTRUCTION(SX,SY,SZ) (pics2 + \
    (SX)*ux + (SY)*uy + (SZ)*uz + mxx + myy + mzz + \
    (SX)*(SY)*mxy + (SX)*(SZ)*mxz + (SY)*(SZ)*myz + \
    one_minus_cs2*((SY)*mxxy + (SZ)*mxxz + (SX)*mxyy + (SX)*mxzz + (SY)*myzz + (SZ)*myyz) + \
    (SX)*(SY)*(SZ)*mxyz + one_minus_cs2_square*(mxxyy+mxxzz+myyzz) + \
    one_minus_cs2*((SX)*(SY)*mxyzz + (SX)*(SZ)*mxyyz + (SY)*(SZ)*mxxyz))
multiplyTerm = rho * W3;
pop[19] = multiplyTerm * MR_CORNER_RECONSTRUCTION( 1, 1, 1);
pop[20] = multiplyTerm * MR_CORNER_RECONSTRUCTION(-1,-1,-1);
pop[21] = multiplyTerm * MR_CORNER_RECONSTRUCTION( 1, 1,-1);
pop[22] = multiplyTerm * MR_CORNER_RECONSTRUCTION(-1,-1, 1);
pop[23] = multiplyTerm * MR_CORNER_RECONSTRUCTION( 1,-1, 1);
pop[24] = multiplyTerm * MR_CORNER_RECONSTRUCTION(-1, 1,-1);
pop[25] = multiplyTerm * MR_CORNER_RECONSTRUCTION(-1, 1, 1);
pop[26] = multiplyTerm * MR_CORNER_RECONSTRUCTION( 1,-1,-1);
#undef MR_CORNER_RECONSTRUCTION
#endif
#ifndef D3Q27
mrlbmConservativeD3Q19Projection(pop, rhoVar, ux_t30, uy_t30, uz_t30,
    m_xx_t45, m_xy_t90, m_xz_t90, m_yy_t45, m_yz_t90, m_zz_t45);
#endif
}
