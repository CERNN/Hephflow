//copy full macroscopic field
int y_wall;
double t_ux0, t_uy0,t_uz0;
double t_rho0;
double t_Mxx, t_Mxy, t_Mxz, t_Myy, t_Myz, t_Mzz;
double inv_eta;


double rho_0[NY];

double ux_1[NY];
double uy_1[NY];
double uz_1[NY];

double ux_ux[NY];
double uy_uy[NY];
double uz_uz[NY];
double ux_uy[NY];
double ux_uz[NY];
double uy_uz[NY];

double ux_3[NY];
double uy_3[NY];
double uz_3[NY];

double ux_4[NY];
double uy_4[NY];
double uz_4[NY];

double Sxx[NY];
double Sxy[NY];
double Sxz[NY];
double Syy[NY];
double Syz[NY];
double Szz[NY];

double SGxx[NY];
double SGxy[NY];
double SGxz[NY];
double SGyy[NY];
double SGyz[NY];
double SGzz[NY];

for (int y_wall = 0; y_wall <NY;y_wall++){
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

    Sxx[y_wall] = 0;
    Sxy[y_wall] = 0;
    Sxz[y_wall] = 0;
    Syy[y_wall] = 0;
    Syz[y_wall] = 0;
    Szz[y_wall] = 0;

    SGxx[y_wall] = 0;
    SGxy[y_wall] = 0;
    SGxz[y_wall] = 0;
    SGyy[y_wall] = 0;
    SGyz[y_wall] = 0;
    SGzz[y_wall] = 0;

}


for(int y = 0; y< NY;y++){

    y_wall = y;

    for (int z = 0 ; z <NZ_TOTAL; z++){
        for(int x = 0; x< NX;x++){

            //current lattice value
            t_rho0 = static_cast<double>(h_fMom[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M_RHO_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)]) + static_cast<double>(RHO_0);
            t_ux0 = static_cast<double>(h_fMom[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M_UX_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)]) / static_cast<double>(F_M_I_SCALE);
            t_uy0 = static_cast<double>(h_fMom[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M_UY_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)]) / static_cast<double>(F_M_I_SCALE);
            t_uz0 = static_cast<double>(h_fMom[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M_UZ_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)]) / static_cast<double>(F_M_I_SCALE);

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

            t_Mxx = static_cast<double>(h_fMom[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M_MXX_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)]) / static_cast<double>(F_M_II_SCALE);
            t_Mxy = static_cast<double>(h_fMom[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M_MXY_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)]) / static_cast<double>(F_M_IJ_SCALE);
            t_Mxz = static_cast<double>(h_fMom[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M_MXZ_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)]) / static_cast<double>(F_M_IJ_SCALE);
            t_Myy = static_cast<double>(h_fMom[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M_MYY_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)]) / static_cast<double>(F_M_II_SCALE);
            t_Myz = static_cast<double>(h_fMom[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M_MYZ_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)]) / static_cast<double>(F_M_IJ_SCALE);
            t_Mzz = static_cast<double>(h_fMom[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M_MZZ_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)]) / static_cast<double>(F_M_II_SCALE);

            //correction for pre-collsion values  
            t_Mxx = (t_Mxx - OMEGA * t_ux0 * t_ux0 - (1.0 - OMEGA/2)*0)/(1.0 - OMEGA);
            t_Mxy = (t_Mxy - OMEGA * t_ux0 * t_uy0 - (1.0 - OMEGA/2)*0)/(1.0 - OMEGA);
            t_Mxz = (t_Mxz - OMEGA * t_ux0 * t_uz0 - (1.0 - OMEGA/2)*(t_ux0*FZ))/(1.0 - OMEGA);
            t_Myy = (t_Myy - OMEGA * t_uy0 * t_uy0 - (1.0 - OMEGA/2)*0)/(1.0 - OMEGA);
            t_Myz = (t_Myz - OMEGA * t_uy0 * t_uz0 - (1.0 - OMEGA/2)*(t_uy0*FZ))/(1.0 - OMEGA);
            t_Mzz = (t_Mzz - OMEGA * t_uz0 * t_uz0 - (1.0 - OMEGA/2)*(t_uz0*FZ*2))/(1.0 - OMEGA);

            //computing stress
            t_Mxx = t_Mxx - t_ux0 * t_ux0 + (FX * t_ux0 + t_ux0 * FX)/2;
            t_Mxy = t_Mxy - t_ux0 * t_uy0 + (FX * t_uy0 + t_ux0 * FY)/2;
            t_Mxz = t_Mxz - t_ux0 * t_uz0 + (FX * t_uz0 + t_ux0 * FZ)/2;
            t_Myy = t_Myy - t_uy0 * t_uy0 + (FY * t_uy0 + t_uy0 * FY)/2;
            t_Myz = t_Myz - t_uy0 * t_uz0 + (FY * t_uz0 + t_uy0 * FZ)/2;
            t_Mzz = t_Mzz - t_uz0 * t_uz0 + (FZ * t_uz0 + t_uz0 * FZ)/2;

            t_Mxx = (1.0-OMEGA/2)*t_Mxx;
            t_Mxy = (1.0-OMEGA/2)*t_Mxy;
            t_Mxz = (1.0-OMEGA/2)*t_Mxz;
            t_Myy = (1.0-OMEGA/2)*t_Myy;
            t_Myz = (1.0-OMEGA/2)*t_Myz;
            t_Mzz = (1.0-OMEGA/2)*t_Mzz;

            Sxx[y_wall] += t_Mxx;
            Sxy[y_wall] += t_Mxy;
            Sxz[y_wall] += t_Mxz;
            Syy[y_wall] += t_Myy;
            Syz[y_wall] += t_Myz;
            Szz[y_wall] += t_Mzz;

            //computing stress times gamma
            t_Mxx *= t_Mxx;
            t_Mxy *= t_Mxy;
            t_Mxz *= t_Mxz;
            t_Myy *= t_Myy;
            t_Myz *= t_Myz;
            t_Mzz *= t_Mzz;

            inv_eta = 6*OMEGA/(2-OMEGA);

            SGxx[y_wall] += t_Mxx * inv_eta;
            SGxy[y_wall] += t_Mxy * inv_eta;
            SGxz[y_wall] += t_Mxz * inv_eta;
            SGyy[y_wall] += t_Myy * inv_eta;
            SGyz[y_wall] += t_Myz * inv_eta;
            SGzz[y_wall] += t_Mzz * inv_eta;

        }
    }
}

for (int y_wall = 0; y_wall <NY;y_wall++){
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

    Sxx[y_wall] /= (NX*NZ_TOTAL);
    Sxy[y_wall] /= (NX*NZ_TOTAL);
    Sxz[y_wall] /= (NX*NZ_TOTAL);
    Syy[y_wall] /= (NX*NZ_TOTAL);
    Syz[y_wall] /= (NX*NZ_TOTAL);
    Szz[y_wall] /= (NX*NZ_TOTAL);

    SGxx[y_wall] /= (NX*NZ_TOTAL);
    SGxy[y_wall] /= (NX*NZ_TOTAL);
    SGxz[y_wall] /= (NX*NZ_TOTAL);
    SGyy[y_wall] /= (NX*NZ_TOTAL);
    SGyz[y_wall] /= (NX*NZ_TOTAL);
    SGzz[y_wall] /= (NX*NZ_TOTAL);

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

std::ostringstream strDataInfo_Sxx("");
std::ostringstream strDataInfo_Sxy("");
std::ostringstream strDataInfo_Sxz("");
std::ostringstream strDataInfo_Syy("");
std::ostringstream strDataInfo_Syz("");
std::ostringstream strDataInfo_Szz("");

std::ostringstream strDataInfo_SGxx("");
std::ostringstream strDataInfo_SGxy("");
std::ostringstream strDataInfo_SGxz("");
std::ostringstream strDataInfo_SGyy("");
std::ostringstream strDataInfo_SGyz("");
std::ostringstream strDataInfo_SGzz("");

std::ostringstream* statistic_streams[] = {
    &strDataInfo_rho_0,
    &strDataInfo_ux_1, &strDataInfo_uy_1, &strDataInfo_uz_1,
    &strDataInfo_ux_ux, &strDataInfo_uy_uy, &strDataInfo_uz_uz,
    &strDataInfo_ux_uy, &strDataInfo_ux_uz, &strDataInfo_uy_uz,
    &strDataInfo_ux_3, &strDataInfo_uy_3, &strDataInfo_uz_3,
    &strDataInfo_ux_4, &strDataInfo_uy_4, &strDataInfo_uz_4,
    &strDataInfo_Sxx, &strDataInfo_Sxy, &strDataInfo_Sxz,
    &strDataInfo_Syy, &strDataInfo_Syz, &strDataInfo_Szz,
    &strDataInfo_SGxx, &strDataInfo_SGxy, &strDataInfo_SGxz,
    &strDataInfo_SGyy, &strDataInfo_SGyz, &strDataInfo_SGzz
};
for (std::ostringstream* stream : statistic_streams) {
    *stream << std::scientific << std::setprecision(17);
}

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

strDataInfo_Sxx <<"step,"<< step;
strDataInfo_Sxy <<"step,"<< step;
strDataInfo_Sxz <<"step,"<< step;
strDataInfo_Syy <<"step,"<< step;
strDataInfo_Syz <<"step,"<< step;
strDataInfo_Szz <<"step,"<< step;

strDataInfo_SGxx <<"step,"<< step;
strDataInfo_SGxy <<"step,"<< step;
strDataInfo_SGxz <<"step,"<< step;
strDataInfo_SGyy <<"step,"<< step;
strDataInfo_SGyz <<"step,"<< step;
strDataInfo_SGzz <<"step,"<< step;

for(int y = 0; y< NY;y++){

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

    strDataInfo_Sxx  << "," << Sxx[y];
    strDataInfo_Sxy  << "," << Sxy[y];
    strDataInfo_Sxz  << "," << Sxz[y];
    strDataInfo_Syy  << "," << Syy[y];
    strDataInfo_Syz  << "," << Syz[y];
    strDataInfo_Szz  << "," << Szz[y];

    strDataInfo_SGxx  << "," << SGxx[y];
    strDataInfo_SGxy  << "," << SGxy[y];
    strDataInfo_SGxz  << "," << SGxz[y];
    strDataInfo_SGyy  << "," << SGyy[y];
    strDataInfo_SGyz  << "," << SGyz[y];
    strDataInfo_SGzz  << "," << SGzz[y];

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

saveTreatData("_turbulent_Sxx",strDataInfo_Sxx.str(),step);
saveTreatData("_turbulent_Syy",strDataInfo_Syy.str(),step);
saveTreatData("_turbulent_Szz",strDataInfo_Szz.str(),step);
saveTreatData("_turbulent_Sxy",strDataInfo_Sxy.str(),step);
saveTreatData("_turbulent_Sxz",strDataInfo_Sxz.str(),step);
saveTreatData("_turbulent_Syz",strDataInfo_Syz.str(),step);


saveTreatData("_turbulent_SGxx",strDataInfo_SGxx.str(),step);
saveTreatData("_turbulent_SGyy",strDataInfo_SGyy.str(),step);
saveTreatData("_turbulent_SGzz",strDataInfo_SGzz.str(),step);
saveTreatData("_turbulent_SGxy",strDataInfo_SGxy.str(),step);
saveTreatData("_turbulent_SGxz",strDataInfo_SGxz.str(),step);
saveTreatData("_turbulent_SGyz",strDataInfo_SGyz.str(),step);

// FOR CORRELATIONS

int x0, y0, z0;

x0 = NX/2;
z0 = NZ_TOTAL/2;
//x direction
velocityProfile(h_fMom,4,x0,(2*del)/96,z0,step); //ux
velocityProfile(h_fMom,5,x0,(2*del)/96,z0,step); //uy
velocityProfile(h_fMom,6,x0,(2*del)/96,z0,step); //uz

velocityProfile(h_fMom,4,x0,(6*del)/96,z0,step); //ux
velocityProfile(h_fMom,5,x0,(6*del)/96,z0,step); //uy
velocityProfile(h_fMom,6,x0,(6*del)/96,z0,step); //uz

velocityProfile(h_fMom,4,x0,(52*del)/96,z0,step); //ux
velocityProfile(h_fMom,5,x0,(52*del)/96,z0,step); //uy
velocityProfile(h_fMom,6,x0,(52*del)/96,z0,step); //uz

velocityProfile(h_fMom,4,x0,(96*del)/96,z0,step); //ux
velocityProfile(h_fMom,5,x0,(96*del)/96,z0,step); //uy
velocityProfile(h_fMom,6,x0,(96*del)/96,z0,step); //uz

//Z DIRECITON
velocityProfile(h_fMom,7,x0,(2*del)/96,z0,step); //ux
velocityProfile(h_fMom,8,x0,(2*del)/96,z0,step); //uy
velocityProfile(h_fMom,9,x0,(2*del)/96,z0,step); //uz

velocityProfile(h_fMom,7,x0,(6*del)/96,z0,step); //ux
velocityProfile(h_fMom,8,x0,(6*del)/96,z0,step); //uy
velocityProfile(h_fMom,9,x0,(6*del)/96,z0,step); //uz

velocityProfile(h_fMom,7,x0,(52*del)/96,z0,step); //ux
velocityProfile(h_fMom,8,x0,(52*del)/96,z0,step); //uy
velocityProfile(h_fMom,9,x0,(52*del)/96,z0,step); //uz

velocityProfile(h_fMom,7,x0,(96*del)/96,z0,step); //ux
velocityProfile(h_fMom,8,x0,(96*del)/96,z0,step); //uy
velocityProfile(h_fMom,9,x0,(96*del)/96,z0,step); //uz


rhoProfile(h_fMom,3,x0,(2*del)/96,z0,step);  //rho
rhoProfile(h_fMom,3,x0,(6*del)/96,z0,step);  //rho
rhoProfile(h_fMom,3,x0,(52*del)/96,z0,step); //rho
rhoProfile(h_fMom,3,x0,(96*del)/96,z0,step); //rho

// Reynolds-stress statistics for the Lumley anisotropy-invariant map.
// These running means combine the homogeneous x-z plane average calculated
// above with an equally weighted average over calls to this treatment routine.
static unsigned long long turbulence_samples = 0;
static double turb_mean_ux[NY] = {};
static double turb_mean_uy[NY] = {};
static double turb_mean_uz[NY] = {};
static double turb_mean_uxux[NY] = {};
static double turb_mean_uyuy[NY] = {};
static double turb_mean_uzuz[NY] = {};
static double turb_mean_uxuy[NY] = {};
static double turb_mean_uxuz[NY] = {};
static double turb_mean_uyuz[NY] = {};

++turbulence_samples;
const double turbulence_sample_weight =
    1.0 / static_cast<double>(turbulence_samples);

double turb_kinetic[NY] = {};
double invariant2[NY] = {};
double invariant3[NY] = {};

for (int y = 0; y < NY; ++y) {
    turb_mean_ux[y] += (ux_1[y] - turb_mean_ux[y]) * turbulence_sample_weight;
    turb_mean_uy[y] += (uy_1[y] - turb_mean_uy[y]) * turbulence_sample_weight;
    turb_mean_uz[y] += (uz_1[y] - turb_mean_uz[y]) * turbulence_sample_weight;

    turb_mean_uxux[y] += (ux_ux[y] - turb_mean_uxux[y]) * turbulence_sample_weight;
    turb_mean_uyuy[y] += (uy_uy[y] - turb_mean_uyuy[y]) * turbulence_sample_weight;
    turb_mean_uzuz[y] += (uz_uz[y] - turb_mean_uzuz[y]) * turbulence_sample_weight;
    turb_mean_uxuy[y] += (ux_uy[y] - turb_mean_uxuy[y]) * turbulence_sample_weight;
    turb_mean_uxuz[y] += (ux_uz[y] - turb_mean_uxuz[y]) * turbulence_sample_weight;
    turb_mean_uyuz[y] += (uy_uz[y] - turb_mean_uyuz[y]) * turbulence_sample_weight;

    // R_ij = <u_i u_j> - <u_i><u_j>.
    double R[3][3] = {};
    R[0][0] = turb_mean_uxux[y] - turb_mean_ux[y] * turb_mean_ux[y];
    R[1][1] = turb_mean_uyuy[y] - turb_mean_uy[y] * turb_mean_uy[y];
    R[2][2] = turb_mean_uzuz[y] - turb_mean_uz[y] * turb_mean_uz[y];
    R[0][1] = R[1][0] = turb_mean_uxuy[y] - turb_mean_ux[y] * turb_mean_uy[y];
    R[0][2] = R[2][0] = turb_mean_uxuz[y] - turb_mean_ux[y] * turb_mean_uz[y];
    R[1][2] = R[2][1] = turb_mean_uyuz[y] - turb_mean_uy[y] * turb_mean_uz[y];

    turb_kinetic[y] = 0.5 * (R[0][0] + R[1][1] + R[2][2]);

    // b_ij = R_ij/(2k) - delta_ij/3. At negligible k the
    // anisotropy direction is undefined, so leave both invariants at zero.
    constexpr double K_TOLERANCE = 1.0e-20;
    if (turb_kinetic[y] <= K_TOLERANCE) {
        continue;
    }

    double bij[3][3] = {};
    const double inverse_two_k = 1.0 / (2.0 * turb_kinetic[y]);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            bij[i][j] = R[i][j] * inverse_two_k;
            if (i == j) {
                bij[i][j] -= 1.0 / 3.0;
            }
        }
    }

    // Principal invariants used by the Lumley map:
    // II_b = -0.5 tr(b^2), III_b = det(b) = tr(b^3)/3.
    double trace_b_squared = 0.0;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            trace_b_squared += bij[i][j] * bij[j][i];
        }
    }
    invariant2[y] = -0.5 * trace_b_squared;
    invariant3[y] =
        bij[0][0] * (bij[1][1] * bij[2][2] - bij[1][2] * bij[2][1]) -
        bij[0][1] * (bij[1][0] * bij[2][2] - bij[1][2] * bij[2][0]) +
        bij[0][2] * (bij[1][0] * bij[2][1] - bij[1][1] * bij[2][0]);
}

std::ostringstream strDataInfo_k("");
std::ostringstream strDataInfo_II("");
std::ostringstream strDataInfo_III("");

strDataInfo_k << std::scientific << std::setprecision(17);
strDataInfo_II << std::scientific << std::setprecision(17);
strDataInfo_III << std::scientific << std::setprecision(17);

strDataInfo_k << "step," << step;
strDataInfo_II << "step," << step;
strDataInfo_III << "step," << step;

// Compute each channel side independently before folding by wall distance.
// This avoids sign errors in cross-correlations containing wall-normal velocity.
constexpr int N_WALL_DISTANCES = (NY + 1) / 2;
for (int y_wall = 0; y_wall < N_WALL_DISTANCES; ++y_wall) {
    const int opposite_y = NY - 1 - y_wall;
    const double fold_weight = (opposite_y == y_wall) ? 1.0 : 0.5;

    strDataInfo_k << "," << fold_weight *
        (turb_kinetic[y_wall] +
         ((opposite_y == y_wall) ? 0.0 : turb_kinetic[opposite_y]));
    strDataInfo_II << "," << fold_weight *
        (invariant2[y_wall] +
         ((opposite_y == y_wall) ? 0.0 : invariant2[opposite_y]));
    strDataInfo_III << "," << fold_weight *
        (invariant3[y_wall] +
         ((opposite_y == y_wall) ? 0.0 : invariant3[opposite_y]));
}

saveTreatData("_turbulent_k", strDataInfo_k.str(), step);
saveTreatData("_turbulent_II", strDataInfo_II.str(), step);
saveTreatData("_turbulent_III", strDataInfo_III.str(), step);
