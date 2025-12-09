//copy full macroscopic field
checkCudaErrors(cudaDeviceSynchronize());
checkCudaErrors(cudaMemcpy(h_fMom, fMom, sizeof(dfloat) * NUMBER_LBM_NODES*NUMBER_MOMENTS, cudaMemcpyDeviceToHost));
checkCudaErrors(cudaDeviceSynchronize());

dfloat t_ux0, t_uy0,t_uz0;
dfloat t_rho0;
dfloat t_Mxx, t_Mxy, t_Mxz, t_Myy, t_Myz, t_Mzz;
dfloat t_omega0, t_eta0;


dfloat rho_0[NY];

dfloat ux_1[NY];
dfloat uy_1[NY];
dfloat uz_1[NY];


dfloat Sxx[NY];
dfloat Syy[NY];
dfloat Szz[NY];
dfloat Sxy[NY];
dfloat Sxz[NY];
dfloat Syz[NY];


for(int y = 0; y< NY;y++){
    //distance to any wall

    rho_0[y] = 0;

    ux_1[y] = 0;
    uy_1[y] = 0;
    uz_1[y] = 0;

    Sxx[y] = 0;
    Syy[y] = 0;
    Szz[y] = 0;
    Sxy[y] = 0;
    Sxz[y] = 0;
    Syz[y] = 0;


    for (int z = 0 ; z <NZ_TOTAL; z++){
        for(int x = 0; x< NX;x++){

            //current lattice value
            t_rho0 = h_fMom[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M_RHO_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)] + RHO_0;
            t_ux0 =  h_fMom[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M_UX_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)]/F_M_I_SCALE;
            t_uy0 =  h_fMom[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M_UY_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)]/F_M_I_SCALE;
            t_uz0 =  h_fMom[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M_UZ_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)]/F_M_I_SCALE;

            rho_0[y] += t_rho0;

            ux_1[y] += t_ux0;
            uy_1[y] += t_uy0;
            uz_1[y] += t_uz0;

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

            Sxx[y] += t_Mxx;
            Syy[y] += t_Mxy;
            Szz[y] += t_Mxz;
            Sxy[y] += t_Myy;
            Sxz[y] += t_Myz;
            Syz[y] += t_Mzz;

        }
    }

    rho_0[y] /= (NX*NZ_TOTAL);

    ux_1[y] /= (NX*NZ_TOTAL);
    uy_1[y] /= (NX*NZ_TOTAL);
    uz_1[y] /= (NX*NZ_TOTAL);

    Sxx[y] /= (NX*NZ_TOTAL);
    Syy[y] /= (NX*NZ_TOTAL);
    Szz[y] /= (NX*NZ_TOTAL);
    Sxy[y] /= (NX*NZ_TOTAL);
    Sxz[y] /= (NX*NZ_TOTAL);
    Syz[y] /= (NX*NZ_TOTAL);
}

std::ostringstream strDataInfo_rho_0("");

std::ostringstream strDataInfo_ux_1("");
std::ostringstream strDataInfo_uy_1("");
std::ostringstream strDataInfo_uz_1("");

std::ostringstream strDataInfo_Sxx("");
std::ostringstream strDataInfo_Syy("");
std::ostringstream strDataInfo_Szz("");
std::ostringstream strDataInfo_Sxy("");
std::ostringstream strDataInfo_Sxz("");
std::ostringstream strDataInfo_Syz("");


strDataInfo_rho_0 <<"step,"<< step;

strDataInfo_ux_1 <<"step,"<< step;
strDataInfo_uy_1 <<"step,"<< step;
strDataInfo_uz_1 <<"step,"<< step;

strDataInfo_Sxx <<"step,"<< step;
strDataInfo_Syy <<"step,"<< step;
strDataInfo_Szz <<"step,"<< step;
strDataInfo_Sxy <<"step,"<< step;
strDataInfo_Sxz <<"step,"<< step;
strDataInfo_Syz <<"step,"<< step;


for(int y = 0; y< NY;y++){

    strDataInfo_rho_0 << "," << rho_0[y];

    strDataInfo_ux_1 << "," << ux_1[y];
    strDataInfo_uy_1 << "," << uy_1[y];
    strDataInfo_uz_1 << "," << uz_1[y];

    strDataInfo_Sxx  << "," << Sxx[y];
    strDataInfo_Syy  << "," << Syy[y];
    strDataInfo_Szz  << "," << Szz[y];
    strDataInfo_Sxy  << "," << Sxy[y];
    strDataInfo_Sxz  << "," << Sxz[y];
    strDataInfo_Syz  << "," << Syz[y];

}

saveTreatData("_turbulent_rho",strDataInfo_rho_0.str(),step);

saveTreatData("_turbulent_ux1",strDataInfo_ux_1.str(),step);
saveTreatData("_turbulent_uy1",strDataInfo_uy_1.str(),step);
saveTreatData("_turbulent_uz1",strDataInfo_uz_1.str(),step);

saveTreatData("_turbulent_Sxx",strDataInfo_Sxx.str(),step);
saveTreatData("_turbulent_Syy",strDataInfo_Syy.str(),step);
saveTreatData("_turbulent_Szz",strDataInfo_Szz.str(),step);
saveTreatData("_turbulent_Sxy",strDataInfo_Sxy.str(),step);
saveTreatData("_turbulent_Sxz",strDataInfo_Sxz.str(),step);
saveTreatData("_turbulent_Syz",strDataInfo_Syz.str(),step);