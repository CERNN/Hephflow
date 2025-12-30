// Helper lambda to get stored gradients at neighbor positions
auto getGrad = [&](int dx, int dy, int dz, int comp) -> dfloat {
    int nx = (x + dx + NX) % NX;
    int ny = (y + dy + NY) % NY;
    int nz = (z + dz + NZ) % NZ;
    int ntx = nx % BLOCK_NX;
    int nty = ny % BLOCK_NY;
    int ntz = nz % BLOCK_NZ;
    int nbx = nx / BLOCK_NX;
    int nby = ny / BLOCK_NY;
    int nbz = nz / BLOCK_NZ;
    return fMom[idxMom(ntx, nty, ntz, comp, nbx, nby, nbz)];
};

// Load gradients at neighbor nodes
dfloat dphidx_xp1 = getGrad(+1, 0, 0, M3_NX_INDEX);
dfloat dphidx_xm1 = getGrad(-1, 0, 0, M3_NX_INDEX);
dfloat dphidy_yp1 = getGrad(0, +1, 0, M3_NY_INDEX);
dfloat dphidy_ym1 = getGrad(0, -1, 0, M3_NY_INDEX);
dfloat dphidz_zp1 = getGrad(0, 0, +1, M3_NZ_INDEX);
dfloat dphidz_zm1 = getGrad(0, 0, -1, M3_NZ_INDEX);

// Diagonal second derivatives
dfloat d2phidx2 = (dphidx_xp1 - dphidx_xm1) * 0.5_df;
dfloat d2phidy2 = (dphidy_yp1 - dphidy_ym1) * 0.5_df;
dfloat d2phidz2 = (dphidz_zp1 - dphidz_zm1) * 0.5_df;

// Mixed derivatives
dfloat dphidx_yp1 = getGrad(0, +1, 0, M3_NX_INDEX);
dfloat dphidx_ym1 = getGrad(0, -1, 0, M3_NX_INDEX);
dfloat dphidx_zp1 = getGrad(0, 0, +1, M3_NX_INDEX);
dfloat dphidx_zm1 = getGrad(0, 0, -1, M3_NX_INDEX);
//dfloat dphidy_xp1 = getGrad(+1, 0, 0, M3_NY_INDEX);
//dfloat dphidy_xm1 = getGrad(-1, 0, 0, M3_NY_INDEX);
dfloat dphidy_zp1 = getGrad(0, 0, +1, M3_NY_INDEX);
dfloat dphidy_zm1 = getGrad(0, 0, -1, M3_NY_INDEX);
//dfloat dphidz_xp1 = getGrad(+1, 0, 0, M3_NZ_INDEX);
//dfloat dphidz_xm1 = getGrad(-1, 0, 0, M3_NZ_INDEX);
//dfloat dphidz_yp1 = getGrad(0, +1, 0, M3_NZ_INDEX);
//dfloat dphidz_ym1 = getGrad(0, -1, 0, M3_NZ_INDEX);

dfloat d2phidxdy = (dphidx_yp1 - dphidx_ym1) * 0.5_df;
dfloat d2phidxdz = (dphidx_zp1 - dphidx_zm1) * 0.5_df;
dfloat d2phidydz = (dphidy_zp1 - dphidy_zm1) * 0.5_df;


const dfloat EPS_CURV = 1e-12_df;   // regularization
const dfloat PHI_MIN  = 0.05_df;    // interface band
const dfloat PHI_MAX  = 0.95_df;

// nabla^2 phi = d^2phi/dx^2 + d^2phi/dy^2 + d^2phi/dz^2
dfloat laplacian_phi = d2phidx2 + d2phidy2 + d2phidz2;

// Regularized gradient magnitude
dfloat grad2 = dphidx*dphidx + dphidy*dphidy + dphidz*dphidz;
dfloat mod_grad_reg = sqrt(grad2 + EPS_CURV);

// Default
dfloat curvature = 0.0_df;

if (phiVar > PHI_MIN && phiVar < PHI_MAX) {

    // Normalized normal (regularized)
    dfloat inv_mod_grad = 1.0_df / mod_grad_reg;
    dfloat nx = dphidx * inv_mod_grad;
    dfloat ny = dphidy * inv_mod_grad;
    dfloat nz = dphidz * inv_mod_grad;

    // n · H · n
    dfloat normal_hessian = nx*nx*d2phidx2 + ny*ny*d2phidy2 + nz*nz*d2phidz2 + 2.0_df * (nx*ny*d2phidxdy + nx*nz*d2phidxdz + ny*nz*d2phidydz);
    curvature = (laplacian_phi - normal_hessian) * inv_mod_grad;
}