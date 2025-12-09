//copy full macroscopic field
checkCudaErrors(cudaDeviceSynchronize());
checkCudaErrors(cudaMemcpy(h_fMom, fMom, sizeof(dfloat) * NUMBER_LBM_NODES*NUMBER_MOMENTS, cudaMemcpyDeviceToHost));
checkCudaErrors(cudaDeviceSynchronize());

int y_wall;
dfloat t_ux0, t_uy0,t_uz0;
dfloat t_rho0;
dfloat t_Mxx, t_Mxy, t_Mxz, t_Myy, t_Myz, t_Mzz;
dfloat t_omega0, t_eta0;
dfloat inv_eta, count_prob, count_visc;


dfloat rho_0[NY/2];

dfloat ux_1[NY/2];
dfloat uy_1[NY/2];
dfloat uz_1[NY/2];

dfloat ux_ux[NY/2];
dfloat uy_uy[NY/2];
dfloat uz_uz[NY/2];
dfloat ux_uy[NY/2];
dfloat ux_uz[NY/2];
dfloat uy_uz[NY/2];

dfloat ux_3[NY/2];
dfloat uy_3[NY/2];
dfloat uz_3[NY/2];

dfloat ux_4[NY/2];
dfloat uy_4[NY/2];
dfloat uz_4[NY/2];

dfloat SGxx[NY/2];
dfloat SGyy[NY/2];
dfloat SGzz[NY/2];
dfloat SGxy[NY/2];
dfloat SGxz[NY/2];
dfloat SGyz[NY/2];

dfloat yield_prob[NY/2];
dfloat inv_omega[NY/2];



for(int y = 0; y< NY;y++){
    //distance to any wall
    y_wall = (y <= (NY-1)/2) ? y : (NY-1 - y);

    rho_0[y_wall] = 0;

    ux_1[y_wall] = 0;
    uy_1[y_wall] = 0;
    uz_1[y_wall] = 0;

    ux_ux[y_wall] = 0;
    uy_uy[y_wall] = 0;
    uz_uz[y_wall] = 0;
    ux_uy[y_wall] = 0;
    ux_uz[y_wall] = 0;
    uy_uz[y_wall] = 0;

    ux_3[y_wall] = 0;
    uy_3[y_wall] = 0;
    uz_3[y_wall] = 0;

    ux_4[y_wall] = 0;
    uy_4[y_wall] = 0;
    uz_4[y_wall] = 0;

    SGxx[y_wall] = 0;
    SGyy[y_wall] = 0;
    SGzz[y_wall] = 0;
    SGxy[y_wall] = 0;
    SGxz[y_wall] = 0;
    SGyz[y_wall] = 0;

    yield_prob[y_wall] = 0;
    inv_omega[y_wall] = 0;
    count_prob = 0;
    count_visc = 0;

    for (int z = 0 ; z <NZ_TOTAL; z++){
        for(int x = 0; x< NX;x++){

            //current lattice value
            t_rho0 = h_fMom[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M_RHO_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)] + RHO_0;
            t_ux0 =  h_fMom[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M_UX_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)]/F_M_I_SCALE;
            t_uy0 =  h_fMom[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M_UY_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)]/F_M_I_SCALE;
            t_uz0 =  h_fMom[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M_UZ_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)]/F_M_I_SCALE;

            rho_0[y_wall] += t_rho0;

            ux_1[y_wall] += t_ux0;
            uy_1[y_wall] += t_uy0;
            uz_1[y_wall] += t_uz0;

            ux_ux[y_wall] += t_ux0 * t_ux0;
            uy_uy[y_wall] += t_uy0 * t_uy0;
            uz_uz[y_wall] += t_uz0 * t_uz0;
            ux_uy[y_wall] += t_ux0 * t_uy0;
            ux_uz[y_wall] += t_ux0 * t_uz0;
            uy_uz[y_wall] += t_uy0 * t_uz0;

            ux_3[y_wall] += t_ux0*t_ux0*t_ux0;
            uy_3[y_wall] += t_uy0*t_uy0*t_uy0;
            uz_3[y_wall] += t_uz0*t_uz0*t_uz0;

            ux_4[y_wall] += t_ux0*t_ux0*t_ux0*t_ux0;
            uy_4[y_wall] += t_uy0*t_uy0*t_uy0*t_uy0;
            uz_4[y_wall] += t_uz0*t_uz0*t_uz0*t_uz0;

            t_Mxx = h_fMom[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M_MXX_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)]/F_M_II_SCALE;
            t_Mxy = h_fMom[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M_MXY_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)]/F_M_IJ_SCALE;
            t_Mxz = h_fMom[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M_MXZ_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)]/F_M_IJ_SCALE;
            t_Myy = h_fMom[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M_MYY_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)]/F_M_II_SCALE;
            t_Myz = h_fMom[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M_MYZ_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)]/F_M_IJ_SCALE;
            t_Mzz = h_fMom[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M_MZZ_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)]/F_M_II_SCALE;

            #ifdef OMEGA_FIELD
                t_omega0 = h_fMom[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M_OMEGA_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)];
            #else
                t_omega0 = OMEGA;
            #endif //OMEGA_FIELD

            t_Mxx = t_Mxx - t_ux0 * t_ux0 + (FX * t_ux0 + t_ux0 * FX)/2;
            t_Mxy = t_Mxy - t_ux0 * t_uy0 + (FX * t_uy0 + t_ux0 * FY)/2;
            t_Mxz = t_Mxz - t_ux0 * t_uz0 + (FX * t_uz0 + t_ux0 * FZ)/2;
            t_Myy = t_Myy - t_uy0 * t_uy0 + (FY * t_uy0 + t_uy0 * FY)/2;
            t_Myz = t_Myz - t_uy0 * t_uz0 + (FY * t_uz0 + t_uy0 * FZ)/2;
            t_Mzz = t_Mzz - t_uz0 * t_uz0 + (FZ * t_uz0 + t_uz0 * FZ)/2;


            t_Mxx = (1-t_omega0/2)*t_Mxx;
            t_Mxy = (1-t_omega0/2)*t_Mxy;
            t_Mxz = (1-t_omega0/2)*t_Mxz;
            t_Myy = (1-t_omega0/2)*t_Myy;
            t_Myz = (1-t_omega0/2)*t_Myz;
            t_Mzz = (1-t_omega0/2)*t_Mzz;

            t_Mxx *= t_Mxx;
            t_Mxy *= t_Mxy;
            t_Mxz *= t_Mxz;
            t_Myy *= t_Myy;
            t_Myz *= t_Myz;
            t_Mzz *= t_Mzz;

            inv_eta = 6*t_omega0/(2-t_omega0);

            SGxx[y_wall] += t_Mxx * inv_eta;
            SGyy[y_wall] += t_Mxy * inv_eta;
            SGzz[y_wall] += t_Mxz * inv_eta;
            SGxy[y_wall] += t_Myy * inv_eta;
            SGxz[y_wall] += t_Myz * inv_eta;
            SGyz[y_wall] += t_Mzz * inv_eta;


            if(t_omega0 < OMEGA_CUTOFF){
                yield_prob[y_wall] += 1.0;
                count_prob++;
            }else{
                inv_omega[y_wall] += 1.0/t_omega0;
                count_visc++;
            }

        }
    }

    rho_0[y_wall] /= (NX*NZ_TOTAL);

    ux_1[y_wall] /= (NX*NZ_TOTAL);
    uy_1[y_wall] /= (NX*NZ_TOTAL);
    uz_1[y_wall] /= (NX*NZ_TOTAL);

    ux_ux[y_wall] /= (NX*NZ_TOTAL);
    uy_uy[y_wall] /= (NX*NZ_TOTAL);
    uz_uz[y_wall] /= (NX*NZ_TOTAL);
    ux_uy[y_wall] /= (NX*NZ_TOTAL);
    ux_uz[y_wall] /= (NX*NZ_TOTAL);
    uy_uz[y_wall] /= (NX*NZ_TOTAL);

    ux_3[y_wall] /= (NX*NZ_TOTAL);
    uy_3[y_wall] /= (NX*NZ_TOTAL);
    uz_3[y_wall] /= (NX*NZ_TOTAL);

    ux_4[y_wall] /= (NX*NZ_TOTAL);
    uy_4[y_wall] /= (NX*NZ_TOTAL);
    uz_4[y_wall] /= (NX*NZ_TOTAL);

    SGxx[y_wall] /= (NX*NZ_TOTAL);
    SGyy[y_wall] /= (NX*NZ_TOTAL);
    SGzz[y_wall] /= (NX*NZ_TOTAL);
    SGxy[y_wall] /= (NX*NZ_TOTAL);
    SGxz[y_wall] /= (NX*NZ_TOTAL);
    SGyz[y_wall] /= (NX*NZ_TOTAL);

    yield_prob[y_wall] /= count_prob;
    inv_omega[y_wall] /= count_visc;
}

std::ostringstream strDataInfo_rho_0("");

std::ostringstream strDataInfo_ux_1("");
std::ostringstream strDataInfo_uy_1("");
std::ostringstream strDataInfo_uz_1("");

std::ostringstream strDataInfo_ux_ux("");
std::ostringstream strDataInfo_uy_uy("");
std::ostringstream strDataInfo_uz_uz("");
std::ostringstream strDataInfo_ux_uy("");
std::ostringstream strDataInfo_ux_uz("");
std::ostringstream strDataInfo_uy_uz("");

std::ostringstream strDataInfo_ux_3("");
std::ostringstream strDataInfo_uy_3("");
std::ostringstream strDataInfo_uz_3("");

std::ostringstream strDataInfo_ux_4("");
std::ostringstream strDataInfo_uy_4("");
std::ostringstream strDataInfo_uz_4("");

std::ostringstream strDataInfo_SGxx("");
std::ostringstream strDataInfo_SGyy("");
std::ostringstream strDataInfo_SGzz("");
std::ostringstream strDataInfo_SGxy("");
std::ostringstream strDataInfo_SGxz("");
std::ostringstream strDataInfo_SGyz("");

std::ostringstream strDataInfo_yield_prob("");
std::ostringstream strDataInfo_inv_omega("");

strDataInfo_rho_0 <<"step,"<< step;

strDataInfo_ux_1 <<"step,"<< step;
strDataInfo_uy_1 <<"step,"<< step;
strDataInfo_uz_1 <<"step,"<< step;

strDataInfo_ux_ux <<"step,"<< step;
strDataInfo_uy_uy <<"step,"<< step;
strDataInfo_uz_uz <<"step,"<< step;
strDataInfo_ux_uy <<"step,"<< step;
strDataInfo_ux_uz <<"step,"<< step;
strDataInfo_uy_uz <<"step,"<< step;

strDataInfo_ux_3 <<"step,"<< step;
strDataInfo_uy_3 <<"step,"<< step;
strDataInfo_uz_3 <<"step,"<< step;

strDataInfo_ux_4 <<"step,"<< step;
strDataInfo_uy_4 <<"step,"<< step;
strDataInfo_uz_4 <<"step,"<< step;

strDataInfo_SGxx <<"step,"<< step;
strDataInfo_SGyy <<"step,"<< step;
strDataInfo_SGzz <<"step,"<< step;
strDataInfo_SGxy <<"step,"<< step;
strDataInfo_SGxz <<"step,"<< step;
strDataInfo_SGyz <<"step,"<< step;

strDataInfo_yield_prob <<"step,"<< step;
strDataInfo_inv_omega <<"step,"<< step;


for(int y = 0; y< NY/2;y++){

    strDataInfo_rho_0 << "," << rho_0[y];

    strDataInfo_ux_1 << "," << ux_1[y];
    strDataInfo_uy_1 << "," << uy_1[y];
    strDataInfo_uz_1 << "," << uz_1[y];

    strDataInfo_ux_ux  << "," << ux_ux[y];
    strDataInfo_uy_uy  << "," << uy_uy[y];
    strDataInfo_uz_uz  << "," << uz_uz[y];
    strDataInfo_ux_uy  << "," << ux_uy[y];
    strDataInfo_ux_uz  << "," << ux_uz[y];
    strDataInfo_uy_uz  << "," << uy_uz[y];

    strDataInfo_ux_3  << "," << ux_3[y];
    strDataInfo_uy_3  << "," << uy_3[y];
    strDataInfo_uz_3  << "," << uz_3[y];

    strDataInfo_ux_4  << "," << ux_4[y];
    strDataInfo_uy_4  << "," << uy_4[y];
    strDataInfo_uz_4  << "," << uz_4[y];

    strDataInfo_SGxx  << "," << SGxx[y];
    strDataInfo_SGyy  << "," << SGyy[y];
    strDataInfo_SGzz  << "," << SGzz[y];
    strDataInfo_SGxy  << "," << SGxy[y];
    strDataInfo_SGxz  << "," << SGxz[y];
    strDataInfo_SGyz  << "," << SGyz[y];

    strDataInfo_yield_prob << "," << yield_prob[y_wall];
    strDataInfo_inv_omega  << "," << inv_omega[y_wall];

}

saveTreatData("_turbulent_rho",strDataInfo_rho_0.str(),step);

saveTreatData("_turbulent_ux1",strDataInfo_ux_1.str(),step);
saveTreatData("_turbulent_uy1",strDataInfo_uy_1.str(),step);
saveTreatData("_turbulent_uz1",strDataInfo_uz_1.str(),step);

saveTreatData("_turbulent_ux_ux",strDataInfo_ux_ux.str(),step);
saveTreatData("_turbulent_uy_uy",strDataInfo_uy_uy.str(),step);
saveTreatData("_turbulent_uz_uz",strDataInfo_uz_uz.str(),step);
saveTreatData("_turbulent_ux_uy",strDataInfo_ux_uy.str(),step);
saveTreatData("_turbulent_ux_uz",strDataInfo_ux_uz.str(),step);
saveTreatData("_turbulent_uy_uz",strDataInfo_uy_uz.str(),step);

saveTreatData("_turbulent_ux3",strDataInfo_ux_3.str(),step);
saveTreatData("_turbulent_uy3",strDataInfo_uy_3.str(),step);
saveTreatData("_turbulent_uz3",strDataInfo_uz_3.str(),step);

saveTreatData("_turbulent_ux4",strDataInfo_ux_4.str(),step);
saveTreatData("_turbulent_uy4",strDataInfo_uy_4.str(),step);
saveTreatData("_turbulent_uz4",strDataInfo_uz_4.str(),step);

saveTreatData("_turbulent_SGxx",strDataInfo_SGxx.str(),step);
saveTreatData("_turbulent_SGyy",strDataInfo_SGyy.str(),step);
saveTreatData("_turbulent_SGzz",strDataInfo_SGzz.str(),step);
saveTreatData("_turbulent_SGxy",strDataInfo_SGxy.str(),step);
saveTreatData("_turbulent_SGxz",strDataInfo_SGxz.str(),step);
saveTreatData("_turbulent_SGyz",strDataInfo_SGyz.str(),step);

saveTreatData("_turbulent_yield_prob",strDataInfo_yield_prob.str(),step);
saveTreatData("_turbulent_inv_omega",strDataInfo_inv_omega.str(),step);



// FOR CORRELATIONS
//x direction
velocityProfile(fMom,4,step); //ux
velocityProfile(fMom,5,step); //uy
velocityProfile(fMom,6,step); //uz

//Z DIRECITON
velocityProfile(fMom,7,step); //ux
velocityProfile(fMom,8,step); //uy
velocityProfile(fMom,9,step); //uz


/*

// strain tensor

dfloat Sxx[NY/2];
dfloat Sxy[NY/2];
dfloat Sxz[NY/2];
dfloat Syy[NY/2];
dfloat Syz[NY/2];
dfloat Szz[NY/2];


//kinetic energy and invariants
dfloat turb_kinetic[NY/2];
dfloat invariant2[NY/2];
dfloat invariant3[NY/2];
dfloat R[3][3];
dfloat aij[3][3];



//Update the mean field values
dfloat mean_counter = 1.0/((dfloat)(step/MACR_SAVE)+1.0);




// mean velocity values

ux_mean[y_wall] /= (NX*NZ_TOTAL);;
uy_mean[y_wall] /= (NX*NZ_TOTAL);;
uz_mean[y_wall] /= (NX*NZ_TOTAL);;


for(int y = 0; y< NY;y++){
    //distance to any wall
    y_wall = (y <= (NY-1)/2) ? y : (NY-1 - y);
    

    for (int z = 0 ; z <NZ_TOTAL; z++){
        for(int x = 0; x< NX;x++){
            //current lattice value
        }
    }

    // Build Reynolds stress tensor R_ij
    R[0][0] = ux_fluct[y_wall];      // <u'x u'x>
    R[1][1] = uy_fluct[y_wall];      // <u'y u'y>
    R[2][2] = uz_fluct[y_wall];      // <u'z u'z>
    R[0][1] = R[1][0] = uxuy_fluct[y_wall];
    R[0][2] = R[2][0] = uxuz_fluct[y_wall];
    R[1][2] = R[2][1] = uyuz_fluct[y_wall];

    // Turbulent kinetic energy
    turb_kinetic[y_wall] = 0.5 * (R[0][0] + R[1][1] + R[2][2]);

    // Anisotropy tensor a_ij = R_ij/(2k) - delta_ij/3
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            aij[i][j] = R[i][j] / (2.0 * turb_kinetic[y_wall]);
            if (i == j) aij[i][j] -= 1.0/3.0;
        }
    }

    // Invariant II = -0.5 * a_ij a_ji
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            invariant2[y_wall] += aij[i][j] * aij[j][i];
        }
    }
    invariant2[y_wall] *= -0.5;

    invariant3[y_wall] = aij[0][0]*(aij[1][1]*aij[2][2] - aij[1][2]*aij[2][1]) -
                         aij[0][1]*(aij[1][0]*aij[2][2] - aij[1][2]*aij[2][0]) +
                         aij[0][2]*(aij[1][0]*aij[2][1] - aij[1][1]*aij[2][0]);


}

std::ostringstream strDataInfo_k("");
std::ostringstream strDataInfo_II("");
std::ostringstream strDataInfo_III("");

strDataInfo_k  << "step," << step;
strDataInfo_II << "step," << step;
strDataInfo_III<< "step," << step;



for(int y = 0; y< NY/2;y++){
    strDataInfo_k  << "," << turb_kinetic[y];
    strDataInfo_II << "," << invariant2[y];
    strDataInfo_III<< "," << invariant3[y];
}

saveTreatData("_turbulent_k",   strDataInfo_k.str(), step);
saveTreatData("_turbulent_II",  strDataInfo_II.str(), step);
saveTreatData("_turbulent_III", strDataInfo_III.str(), step);



*/