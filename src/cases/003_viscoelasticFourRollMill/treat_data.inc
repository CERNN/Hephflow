int ix = NX/2 - 1;   // lower x index (for midpoint between ix and ix+1)
int iy = NY/2 - 1;   // lower y index
int iz = NZ/2 - 1;   // lower z index

// helper lambda for periodic mom indexing to keep code shorter
auto getMom = [&](int x, int y, int z, int momIndex){
    int bx = (x % BLOCK_NX + BLOCK_NX) % BLOCK_NX;
    int by = (y % BLOCK_NY + BLOCK_NY) % BLOCK_NY;
    int bz = (z % BLOCK_NZ + BLOCK_NZ) % BLOCK_NZ;
    return h_fMom[idxMom(bx, by, bz, momIndex, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)];
};

// helper to fetch and subtract CONF_ZERO
auto getConf = [&](int x, int y, int z, int momIndex){
    return h_fMom[idxMom(x % BLOCK_NX, y % BLOCK_NY, z % BLOCK_NZ, momIndex, x / BLOCK_NX, y / BLOCK_NY, z / BLOCK_NZ)] - CONF_ZERO;
};


/// SYMETRY
dfloat minVal = 1e30;
dfloat xmin;
dfloat ymin;
dfloat zmin;

int ccx = (NX-1)/2;
int ccy = (NY-1)/2;
int ccz = (NZ-1)/2;
int halfWindow = 15;

for (int zz = 0; zz < NZ; ++zz) {
  for (int yy = 0; yy < NY; ++yy) {
    for (int xx = 0; xx < NX; ++xx) {

      // --- get Ayy at this node
      dfloat Ayy_val = getConf(xx,yy,zz, A_YY_C_INDEX);

      // --- only search in neighborhood of center
      if (abs(xx - ccx) <= halfWindow &&
          abs(yy - ccy) <= halfWindow &&
          abs(zz - ccz) <= halfWindow) {
          
          if (Ayy_val < minVal) {
              minVal = Ayy_val;
              xmin   = xx;
              ymin   = yy;
              zmin   = zz;
          }
      }
    }
  }
}
// neighbor indices
int xm = xmin - 1;
int xp = xmin + 1;
int ym = ymin - 1;
int yp = ymin + 1;


// --- get neighbor velocities
dfloat ux_left  = getMom(xm,ymin,zmin, M_UX_INDEX);
dfloat ux_right = getMom(xp,ymin,zmin, M_UX_INDEX);
dfloat ux_bot   = getMom(xmin,ym,zmin, M_UX_INDEX);
dfloat ux_top   = getMom(xmin,yp,zmin, M_UX_INDEX);

dfloat uy_left  = getMom(xm,ymin,zmin, M_UY_INDEX);
dfloat uy_right = getMom(xp,ymin,zmin, M_UY_INDEX);
dfloat uy_bot   = getMom(xmin,ym,zmin, M_UY_INDEX);
dfloat uy_top   = getMom(xmin,yp,zmin, M_UY_INDEX);

// now derivatives at the cell center (second-order accurate)
dfloat duxdx_t30 = (ux_right - ux_left) / 2;
dfloat duydx_t30 = (uy_right - uy_left) / 2;
dfloat duxdy_t30 = (ux_top   - ux_bot ) / 2;
dfloat duydy_t30 = (uy_top   - uy_bot ) / 2;


// 3 because the stored field is 3 timex
// U_max because need to normality velcoty
// N/(2*M_PI) because the domain is from 0 to 2*M_PI


// average over 8 corners
dfloat Axx = getConf(xmin,ymin,zmin, A_XX_C_INDEX);
dfloat Axy = getConf(xmin,ymin,zmin, A_XY_C_INDEX);
dfloat Ayy = getConf(xmin,ymin,zmin, A_YY_C_INDEX);


const dfloat epsi_c = 3.0*U_MAX/(N/(2*M_PI));

std::ostringstream strDataInfo2("");
strDataInfo2 << std::scientific;
strDataInfo2 << std::setprecision(6);
                        //step           // epsilon                         /Wi_eff                                // Axx     //Axy          //Ayy;
strDataInfo2 <<"step,"<< step/t_c << "," << (duxdx_t30)/epsi_c << "," << Weissenberg_number*duxdx_t30/epsi_c << "," << Axx << "," << Axy  << "," << Ayy << "," <<xmin<<"," <<ymin<<"," <<zmin;
saveTreatData("_data_min_000",strDataInfo2.str(),step);




//%%%%%%%%%%%%%%%%%%%%

int xx = NX/2;
int yy = NY/2;
int zz = NZ/2;

int x_m1 = xx - 1;
int x_p1 = xx + 1;

int y_m1 = yy - 1;
int y_p1 = yy + 1;


dfloat ux_t30 = h_fMom[idxMom(xx%BLOCK_NX, yy%BLOCK_NY, zz%BLOCK_NZ, M_UX_INDEX, xx/BLOCK_NX, yy/BLOCK_NY, zz/BLOCK_NZ)];
dfloat uy_t30 = h_fMom[idxMom(xx%BLOCK_NX, yy%BLOCK_NY, zz%BLOCK_NZ, M_UY_INDEX, xx/BLOCK_NX, yy/BLOCK_NY, zz/BLOCK_NZ)];

dfloat ux_xm1 = h_fMom[idxMom(x_m1%BLOCK_NX, yy%BLOCK_NY, zz%BLOCK_NZ, M_UX_INDEX, x_m1/BLOCK_NX, yy/BLOCK_NY, zz/BLOCK_NZ)];
dfloat ux_xp1 = h_fMom[idxMom(x_p1%BLOCK_NX, yy%BLOCK_NY, zz%BLOCK_NZ, M_UX_INDEX, x_p1/BLOCK_NX, yy/BLOCK_NY, zz/BLOCK_NZ)];

dfloat ux_ym1 = h_fMom[idxMom(xx%BLOCK_NX, y_m1%BLOCK_NY, zz%BLOCK_NZ, M_UX_INDEX, xx/BLOCK_NX, y_m1/BLOCK_NY, zz/BLOCK_NZ)];
dfloat ux_yp1 = h_fMom[idxMom(xx%BLOCK_NX, y_p1%BLOCK_NY, zz%BLOCK_NZ, M_UX_INDEX, xx/BLOCK_NX, y_p1/BLOCK_NY, zz/BLOCK_NZ)];

dfloat uy_xm1 = h_fMom[idxMom(x_m1%BLOCK_NX, yy%BLOCK_NY, zz%BLOCK_NZ, M_UY_INDEX, x_m1/BLOCK_NX, yy/BLOCK_NY, zz/BLOCK_NZ)];
dfloat uy_xp1 = h_fMom[idxMom(x_p1%BLOCK_NX, yy%BLOCK_NY, zz%BLOCK_NZ, M_UY_INDEX, x_p1/BLOCK_NX, yy/BLOCK_NY, zz/BLOCK_NZ)];

dfloat uy_ym1 = h_fMom[idxMom(xx%BLOCK_NX, y_m1%BLOCK_NY, zz%BLOCK_NZ, M_UY_INDEX, xx/BLOCK_NX, y_m1/BLOCK_NY, zz/BLOCK_NZ)];
dfloat uy_yp1 = h_fMom[idxMom(xx%BLOCK_NX, y_p1%BLOCK_NY, zz%BLOCK_NZ, M_UY_INDEX, xx/BLOCK_NX, y_p1/BLOCK_NY, zz/BLOCK_NZ)];

/*
dfloat duxdx_t30 = (ux_xp1 - ux_xm1) / 2;
dfloat duydx_t30 = (uy_xp1 - uy_xm1) / 2;

dfloat duxdy_t30 = (ux_yp1 - ux_ym1) / 2;
dfloat duydy_t30 = (uy_yp1 - uy_ym1) / 2;
*/
/* second order
dfloat duxdx_t30 = (ux_xp1 + ux_xm1 - 2*ux_t30);
dfloat duydx_t30 = (uy_xp1 + uy_xm1 - 2*uy_t30);

dfloat duxdy_t30 = (ux_yp1 + ux_ym1 + 2*ux_t30);
dfloat duydy_t30 = (uy_yp1 + uy_ym1 + 2*uy_t30);
*/

// When doing the central point at (N-1)/2  if the even number will cause the central do move down
// ie. N = 8 , center at 3.5, but doesnt exist a lattice, so derivative is between 3 and  4
duxdx_t30 = (ux_xp1 - ux_t30);
duydx_t30 = (uy_xp1 - uy_t30);

duxdy_t30 = (ux_yp1 - ux_t30);
duydy_t30 = (uy_yp1 - uy_t30);

// 3 because the stored field is 3 timex
// U_max because need to normality velcoty
// N/(2*M_PI) because the domain is from 0 to 2*M_PI

Axx = h_fMom[idxMom(xx%BLOCK_NX, yy%BLOCK_NY, zz%BLOCK_NZ, A_XX_C_INDEX, xx/BLOCK_NX, yy/BLOCK_NY, zz/BLOCK_NZ)] - CONF_ZERO;
Axy = h_fMom[idxMom(xx%BLOCK_NX, yy%BLOCK_NY, zz%BLOCK_NZ, A_XY_C_INDEX, xx/BLOCK_NX, yy/BLOCK_NY, zz/BLOCK_NZ)] - CONF_ZERO;
Ayy = h_fMom[idxMom(xx%BLOCK_NX, yy%BLOCK_NY, zz%BLOCK_NZ, A_YY_C_INDEX, xx/BLOCK_NX, yy/BLOCK_NY, zz/BLOCK_NZ)] - CONF_ZERO;

std::ostringstream strDataInfo("");
strDataInfo << std::scientific;
strDataInfo << std::setprecision(6);
                        //step           // epsilon                         /Wi_eff                                // Axx     //Axy          //Ayy;
strDataInfo <<"step,"<< step/t_c << "," << duxdx_t30/epsi_c << "," << Weissenberg_number*duxdx_t30/epsi_c << "," << Axx << "," << Axy  << "," << Ayy;
saveTreatData("_data_000",strDataInfo.str(),step);


{
    constexpr int N_PROBE   = 16;          // probes per direction (must divide 32)
    const     int stride_x  = NX / N_PROBE;
    const     int stride_y  = NY / N_PROBE;
    const     int iz_probe  = NZ / 2;

    // Build one CSV string per probe point and write as a single file.
    // Format: step_norm, ix, iy, x_star, y_star, ux, uy, Axx, Axy, Ayy
    std::ostringstream ssProbe;
    ssProbe << std::scientific << std::setprecision(8);

    // Header (written every snapshot; easy to strip in post-processing)
    ssProbe << "step_norm,ix,iy,x_star,y_star,ux,uy,Axx,Axy,Ayy\n";

    const dfloat step_norm = static_cast<dfloat>(step) / t_c;

    for (int pi = 0; pi < N_PROBE; ++pi) {
        const int probe_x = pi * stride_x;          // lattice index, exact on all grids
        for (int pj = 0; pj < N_PROBE; ++pj) {
            const int probe_y = pj * stride_y;

            // Physical coordinates in [0,1)
            const dfloat x_star = static_cast<dfloat>(probe_x) / NX;
            const dfloat y_star = static_cast<dfloat>(probe_y) / NY;

            // Velocity moments
            const dfloat ux_p = getMom(probe_x, probe_y, iz_probe, M_UX_INDEX);
            const dfloat uy_p = getMom(probe_x, probe_y, iz_probe, M_UY_INDEX);

            // Conformation tensor (CONF_ZERO subtracted via getConf)
            const dfloat Axx_p = getConf(probe_x, probe_y, iz_probe, A_XX_C_INDEX);
            const dfloat Axy_p = getConf(probe_x, probe_y, iz_probe, A_XY_C_INDEX);
            const dfloat Ayy_p = getConf(probe_x, probe_y, iz_probe, A_YY_C_INDEX);

            ssProbe << step_norm   << ","
                    << probe_x     << ","
                    << probe_y     << ","
                    << x_star      << ","
                    << y_star      << ","
                    << ux_p        << ","
                    << uy_p        << ","
                    << Axx_p       << ","
                    << Axy_p       << ","
                    << Ayy_p       << "\n";
        }
    }

saveTreatData("_probes", ssProbe.str(), step);
}
