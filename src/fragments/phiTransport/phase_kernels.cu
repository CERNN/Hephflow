#include "phase_kernels.cuh"

#ifdef PHI_DIST
__global__ void gpuComputePhaseNormals(
    dfloat *fMom,
    unsigned int *dNodeType, 
    macroInterfaceGPUData macroInterfaceGPU,
    size_t localNZ, 
    int zStart
)
{
    const int x = threadIdx.x + blockDim.x * blockIdx.x;
    const int y = threadIdx.y + blockDim.y * blockIdx.y;
    const int z = threadIdx.z + blockDim.z * blockIdx.z;
    
    if (x >= NX || y >= NY || z >= localNZ)
        return;
    
    const int tx = threadIdx.x, ty = threadIdx.y, tz = threadIdx.z;
    const int bx = blockIdx.x,  by = blockIdx.y,  bz = blockIdx.z;

    unsigned int nodeType = dNodeType[idxScalarBlock(tx, ty, tz, bx, by, bz)];
    if (nodeType == 0b11111111) return;  // only skip fully solid

    dfloat phi_c = fMom[idxMom(tx, ty, tz, M3_PHI_INDEX, bx, by, bz)];

    auto getPhi = [&](int dx, int dy, int dz) -> dfloat {
        #ifdef BC_X_WALL
        int nx = min(NX - 1, max(0, x + dx));
        #else
        int nx = (x + dx + NX) % NX;
        #endif
        #ifdef BC_Y_WALL
        int ny = min(NY - 1, max(0, y + dy));
        #else
        int ny = (y + dy + NY) % NY;
        #endif
        // #ifdef BC_Z_WALL
        // int nz = min((NZ/N_GPUS) - 1, max(0, z + dz));
        // #else
        // // int nz = (z + dz + (NZ/N_GPUS)) % (NZ/N_GPUS);
        // #endif
        int nz = z + dz;
        const int planeIdx = nx + ny * NX;

        if (nz < 0) { 
                #ifdef BC_Z_WALL
                if (zStart == 0) return phi_c;              
                #endif
                return  macroInterfaceGPU.phi.auxZ_1[planeIdx]; 
            
        }

        if (nz >= (int)localNZ) {
                #ifdef BC_Z_WALL
                if (zStart + (int)localNZ >= NZ) return phi_c; 
                #endif
                return macroInterfaceGPU.phi.auxZ_0[planeIdx]; 
        }

        int ntx = nx % BLOCK_NX, nty = ny % BLOCK_NY, ntz = nz % BLOCK_NZ;
        int nbx = nx / BLOCK_NX, nby = ny / BLOCK_NY, nbz = nz / BLOCK_NZ;
        
        return fMom[idxMom(ntx, nty, ntz, M3_PHI_INDEX, nbx, nby, nbz)];
    };

    // load all neighbors with standard clamping
    dfloat phi_xm1 = getPhi(-1, 0, 0);  dfloat phi_xp1 = getPhi(+1, 0, 0);
    dfloat phi_ym1 = getPhi( 0,-1, 0);  dfloat phi_yp1 = getPhi( 0,+1, 0);
    dfloat phi_zm1 = getPhi( 0, 0,-1);  dfloat phi_zp1 = getPhi( 0, 0,+1);

    dfloat phi_xm1_ym1 = getPhi(-1,-1, 0);  dfloat phi_xp1_ym1 = getPhi(+1,-1, 0);
    dfloat phi_xm1_yp1 = getPhi(-1,+1, 0);  dfloat phi_xp1_yp1 = getPhi(+1,+1, 0);
    dfloat phi_xm1_zm1 = getPhi(-1, 0,-1);  dfloat phi_xp1_zm1 = getPhi(+1, 0,-1);
    dfloat phi_xm1_zp1 = getPhi(-1, 0,+1);  dfloat phi_xp1_zp1 = getPhi(+1, 0,+1);
    dfloat phi_ym1_zm1 = getPhi( 0,-1,-1);  dfloat phi_yp1_zm1 = getPhi( 0,+1,-1);
    dfloat phi_ym1_zp1 = getPhi( 0,-1,+1);  dfloat phi_yp1_zp1 = getPhi( 0,+1,+1);

    // ---- bitmask wall detection (fires at the wall node itself) ----
    const bool wall_xm = (nodeType & 0b01010101) == 0b01010101;  // WEST  wall in -x
    const bool wall_xp = (nodeType & 0b10101010) == 0b10101010;  // EAST  wall in +x
    const bool wall_ym = (nodeType & 0b00110011) == 0b00110011;  // SOUTH wall in -y
    const bool wall_yp = (nodeType & 0b11001100) == 0b11001100;  // NORTH wall in +y
    const bool wall_zm = (nodeType & 0b00001111) == 0b00001111;  // BACK  wall in -z
    const bool wall_zp = (nodeType & 0b11110000) == 0b11110000;  // FRONT wall in +z

    // Neumann BC: ghost = phi_c for each wall direction.
    // Edges at corners use phi_c if BOTH directions are walls, else the
    // single-direction Neumann value.
    if (wall_ym) {
        phi_ym1     = phi_c;
        phi_xp1_ym1 = wall_xp ? phi_c : phi_xp1;
        phi_xm1_ym1 = wall_xm ? phi_c : phi_xm1;
        phi_ym1_zp1 = wall_zp ? phi_c : phi_zp1;
        phi_ym1_zm1 = wall_zm ? phi_c : phi_zm1;
    }
    if (wall_yp) {
        phi_yp1     = phi_c;
        phi_xp1_yp1 = wall_xp ? phi_c : phi_xp1;
        phi_xm1_yp1 = wall_xm ? phi_c : phi_xm1;
        phi_yp1_zp1 = wall_zp ? phi_c : phi_zp1;
        phi_yp1_zm1 = wall_zm ? phi_c : phi_zm1;
    }

    auto wettingGhostOnNode =[](dfloat phi_wall, dfloat phi_interior) -> dfloat{
        const dfloat dphi = PHI_TWO - PHI_ONE;

        dfloat Cw = (phi_wall     - PHI_ONE) / dphi;
        dfloat Cp = (phi_interior - PHI_ONE) / dphi;

        Cw = fmax(0.0_df, fmin(1.0_df, Cw));
        Cp = fmax(0.0_df, fmin(1.0_df, Cp));

        const dfloat q = -(4.0_df / phi_del) * cos(contact_angle);

        const dfloat Cg = Cp - 2.0_df * q * Cw * (1.0_df - Cw);

        return PHI_ONE + dphi * Cg;
    };

    if (wall_xm) {
        phi_xm1 = wettingGhostOnNode(phi_c,phi_xp1);
        phi_xm1_yp1 = wettingGhostOnNode(phi_yp1,phi_xp1_yp1);
        phi_xm1_ym1 = wettingGhostOnNode(phi_ym1,phi_xp1_ym1);
        phi_xm1_zp1 = wettingGhostOnNode(phi_zp1,phi_xp1_zp1);
        phi_xm1_zm1 = wettingGhostOnNode(phi_zm1,phi_xp1_zm1);
    }
    if (wall_xp) {
        phi_xp1 = wettingGhostOnNode(phi_c,phi_xm1);
        phi_xp1_yp1 = wettingGhostOnNode(phi_yp1,phi_xm1_yp1);
        phi_xp1_ym1 = wettingGhostOnNode(phi_ym1,phi_xm1_ym1);
        phi_xp1_zp1 = wettingGhostOnNode(phi_zp1,phi_xm1_zp1);
        phi_xp1_zm1 = wettingGhostOnNode(phi_zm1,phi_xm1_zm1);
    }

    if (wall_ym) {
        phi_ym1 = wettingGhostOnNode(phi_c,phi_yp1);
        phi_xp1_ym1 = wettingGhostOnNode(phi_xp1,phi_xp1_yp1);
        phi_xm1_ym1 = wettingGhostOnNode(phi_xm1,phi_xm1_yp1);
        phi_ym1_zp1 = wettingGhostOnNode(phi_zp1,phi_yp1_zp1);
        phi_ym1_zm1 = wettingGhostOnNode(phi_zm1,phi_yp1_zm1);
    }
    if (wall_yp) {
        phi_yp1 = wettingGhostOnNode(phi_c,phi_ym1);
        phi_xp1_yp1 = wettingGhostOnNode(phi_xp1,phi_xp1_ym1);
        phi_xm1_yp1 = wettingGhostOnNode(phi_xm1,phi_xm1_ym1);
        phi_yp1_zp1 = wettingGhostOnNode(phi_zp1,phi_ym1_zp1);
        phi_yp1_zm1 = wettingGhostOnNode(phi_zm1,phi_ym1_zm1);
    }

    if (wall_zm) {
        phi_zm1 = wettingGhostOnNode(phi_c,phi_zp1);
        phi_xp1_zm1 = wettingGhostOnNode(phi_xp1,phi_xp1_zp1);
        phi_xm1_zm1 = wettingGhostOnNode(phi_xm1,phi_xm1_zp1);
        phi_yp1_zm1 = wettingGhostOnNode(phi_yp1,phi_yp1_zp1);
        phi_ym1_zm1 = wettingGhostOnNode(phi_ym1,phi_ym1_zp1);
    }
    if (wall_zp) {
        phi_zp1 = wettingGhostOnNode(phi_c,phi_zm1);
        phi_xp1_zp1 = wettingGhostOnNode(phi_xp1,phi_xp1_zm1);
        phi_xm1_zp1 = wettingGhostOnNode(phi_xm1,phi_xm1_zm1);
        phi_yp1_zp1 = wettingGhostOnNode(phi_yp1,phi_yp1_zm1);
        phi_ym1_zp1 = wettingGhostOnNode(phi_ym1,phi_ym1_zm1);
    }

    // ---- isotropic gradient ----
    constexpr dfloat w_axis = 1.0_df / 6.0_df;
    constexpr dfloat w_edge = 1.0_df / 12.0_df;

    dfloat dphidx = w_axis * (phi_xp1 - phi_xm1)
                  + w_edge * (phi_xp1_ym1 - phi_xm1_ym1 + phi_xp1_yp1 - phi_xm1_yp1
                            + phi_xp1_zm1 - phi_xm1_zm1 + phi_xp1_zp1 - phi_xm1_zp1);

    dfloat dphidy = w_axis * (phi_yp1 - phi_ym1)
                  + w_edge * (phi_xm1_yp1 - phi_xm1_ym1 + phi_xp1_yp1 - phi_xp1_ym1
                            + phi_yp1_zm1 - phi_ym1_zm1 + phi_yp1_zp1 - phi_ym1_zp1);

    dfloat dphidz = w_axis * (phi_zp1 - phi_zm1)
                  + w_edge * (phi_xm1_zp1 - phi_xm1_zm1 + phi_xp1_zp1 - phi_xp1_zm1
                            + phi_ym1_zp1 - phi_ym1_zm1 + phi_yp1_zp1 - phi_yp1_zm1);

    // ---- isotropic Laplacian ----
    constexpr dfloat w_lap_zero = -4.0_df;
    constexpr dfloat w_lap_axis =  1.0_df / 3.0_df;
    constexpr dfloat w_lap_edge =  1.0_df / 6.0_df;

    dfloat laplacian_phi =
        w_lap_zero * phi_c
        + w_lap_axis * (phi_xp1 + phi_xm1 + phi_yp1 + phi_ym1 + phi_zp1 + phi_zm1)
        + w_lap_edge * (phi_xp1_yp1 + phi_xp1_ym1 + phi_xm1_yp1 + phi_xm1_ym1
                      + phi_xp1_zp1 + phi_xp1_zm1 + phi_xm1_zp1 + phi_xm1_zm1
                      + phi_yp1_zp1 + phi_yp1_zm1 + phi_ym1_zp1 + phi_ym1_zm1);

    fMom[idxMom(tx, ty, tz, M3_NX_INDEX, bx, by, bz)] = dphidx;
    fMom[idxMom(tx, ty, tz, M3_NY_INDEX, bx, by, bz)] = dphidy;
    fMom[idxMom(tx, ty, tz, M3_NZ_INDEX, bx, by, bz)] = dphidz;
    fMom[idxMom(tx, ty, tz, M3_LP_INDEX, bx, by, bz)] = laplacian_phi;
}

#ifdef PHASE_NEEDS_CHEMICAL_POTENTIAL
__global__ void gpuComputeChemicalPotential(
    dfloat *fMom,
    unsigned int *dNodeType, 
    size_t localNZ, 
    int zStart
)
{
    const int x = threadIdx.x + blockDim.x * blockIdx.x;
    const int y = threadIdx.y + blockDim.y * blockIdx.y;
    const int z = threadIdx.z + blockDim.z * blockIdx.z;
    
    if (x >= NX || y >= NY || z >= localNZ)
        return;
    
    const int tx = threadIdx.x, ty = threadIdx.y, tz = threadIdx.z;
    const int bx = blockIdx.x,  by = blockIdx.y,  bz = blockIdx.z;

    unsigned int nodeType =
        dNodeType[idxScalarBlock(tx, ty, tz, bx, by, bz)];
    if (nodeType == 0b11111111) return;  // only skip fully solid

    dfloat phi     = fMom[idxMom(tx, ty, tz, M3_PHI_INDEX, bx, by, bz)];
    dfloat lap_phi = fMom[idxMom(tx, ty, tz, M3_LP_INDEX,  bx, by, bz)];

    dfloat dfdphi = A_CH * (phi*phi*phi - phi);
    dfloat mu     = dfdphi - kappa_CH * lap_phi;

    // ---- bitmask wall detection ----
    const bool wall_xm = (nodeType & 0b01010101) == 0b01010101;  // WEST
    const bool wall_xp = (nodeType & 0b10101010) == 0b10101010;  // EAST
    const bool wall_ym = (nodeType & 0b00110011) == 0b00110011;  // SOUTH
    const bool wall_yp = (nodeType & 0b11001100) == 0b11001100;  // NORTH
    const bool wall_zm = (nodeType & 0b00001111) == 0b00001111;  // BACK
    const bool wall_zp = (nodeType & 0b11110000) == 0b11110000;  // FRONT

    const bool is_wall = wall_xm || wall_xp || wall_ym || wall_yp
                      || wall_zm || wall_zp;
                      
    fMom[idxMom(tx, ty, tz, M3_MU_INDEX, bx, by, bz)] = mu;
}
#endif // PHASE_NEEDS_CHEMICAL_POTENTIAL



#ifdef PHASE_TRANSPORT_CAHN_HILLIARD
__global__ void gpuComputeLaplacianMu(
    dfloat *fMom,
    unsigned int *dNodeType,
    macroInterfaceGPUData macroInterfaceGPU, 
    size_t localNZ, 
    int zStart
)
{
    const int x = threadIdx.x + blockDim.x * blockIdx.x;
    const int y = threadIdx.y + blockDim.y * blockIdx.y;
    const int z = threadIdx.z + blockDim.z * blockIdx.z;
    
    if (x >= NX || y >= NY || z >= localNZ)
        return;

    const int tx = threadIdx.x, ty = threadIdx.y, tz = threadIdx.z;
    const int bx = blockIdx.x,  by = blockIdx.y,  bz = blockIdx.z;

    unsigned int nodeType =
        dNodeType[idxScalarBlock(tx, ty, tz, bx, by, bz)];
    if (nodeType == 0b11111111) return;  // only skip fully solid

    dfloat mu0 = fMom[idxMom(tx, ty, tz, M3_MU_INDEX, bx, by, bz)];


    auto getMu = [&](int dx, int dy, int dz) -> dfloat {
        #ifdef BC_X_WALL
        int nx = min(NX - 1, max(0, x + dx));
        #else
        int nx = (x + dx + NX) % NX;
        #endif
        #ifdef BC_Y_WALL
        int ny = min(NY - 1, max(0, y + dy));
        #else
        int ny = (y + dy + NY) % NY;
        #endif
        // #ifdef BC_Z_WALL
        // int nz = min((NZ/N_GPUS) - 1, max(0, z + dz));
        // #else
        // int nz = (z + dz + (NZ/N_GPUS)) % (NZ/N_GPUS);
        // #endif
        int nz = z + dz;
        const int planeIdx = nx + ny * NX;
        

        if (nz < 0) {
                #ifdef BC_Z_WALL
                if (zStart == 0) return mu0;
                #endif
                return macroInterfaceGPU.mu.auxZ_1[planeIdx];
        }
        if (nz >= (int)localNZ) {
                #ifdef BC_Z_WALL
                if (zStart + (int)localNZ >= NZ) return mu0;
                #endif
                return macroInterfaceGPU.mu.auxZ_0[planeIdx];
        }

        int ntx = nx % BLOCK_NX, nty = ny % BLOCK_NY, ntz = nz % BLOCK_NZ;
        int nbx = nx / BLOCK_NX, nby = ny / BLOCK_NY, nbz = nz / BLOCK_NZ;
        return fMom[idxMom(ntx, nty, ntz, M3_MU_INDEX, nbx, nby, nbz)];
    };

    dfloat mu_xp1 = getMu(+1, 0, 0);  dfloat mu_xm1 = getMu(-1, 0, 0);
    dfloat mu_yp1 = getMu( 0,+1, 0);  dfloat mu_ym1 = getMu( 0,-1, 0);
    dfloat mu_zp1 = getMu( 0, 0,+1);  dfloat mu_zm1 = getMu( 0, 0,-1);

    dfloat mu_xp1_yp1 = getMu(+1,+1, 0);  dfloat mu_xp1_ym1 = getMu(+1,-1, 0);
    dfloat mu_xm1_yp1 = getMu(-1,+1, 0);  dfloat mu_xm1_ym1 = getMu(-1,-1, 0);
    dfloat mu_xp1_zp1 = getMu(+1, 0,+1);  dfloat mu_xp1_zm1 = getMu(+1, 0,-1);
    dfloat mu_xm1_zp1 = getMu(-1, 0,+1);  dfloat mu_xm1_zm1 = getMu(-1, 0,-1);
    dfloat mu_yp1_zp1 = getMu( 0,+1,+1);  dfloat mu_yp1_zm1 = getMu( 0,+1,-1);
    dfloat mu_ym1_zp1 = getMu( 0,-1,+1);  dfloat mu_ym1_zm1 = getMu( 0,-1,-1);


    const bool mu_wall_xm = (nodeType & 0b01010101) == 0b01010101;  // WEST
    const bool mu_wall_xp = (nodeType & 0b10101010) == 0b10101010;  // EAST
    const bool mu_wall_ym = (nodeType & 0b00110011) == 0b00110011;  // SOUTH
    const bool mu_wall_yp = (nodeType & 0b11001100) == 0b11001100;  // NORTH
    const bool mu_wall_zm = (nodeType & 0b00001111) == 0b00001111;  // BACK
    const bool mu_wall_zp = (nodeType & 0b11110000) == 0b11110000;  // FRONT

    if (mu_wall_xm) {
        mu_xm1 = mu_xp1;
        mu_xm1_yp1 = mu_xp1_yp1;
        mu_xm1_ym1 = mu_xp1_ym1;
        mu_xm1_zp1 = mu_xp1_zp1;
        mu_xm1_zm1 = mu_xp1_zm1;
    }
    if (mu_wall_xp) {
        mu_xp1 = mu_xm1;
        mu_xp1_yp1 = mu_xm1_yp1;
        mu_xp1_ym1 = mu_xm1_ym1;
        mu_xp1_zp1 = mu_xm1_zp1;
        mu_xp1_zm1 = mu_xm1_zm1;
    }

    if (mu_wall_ym) {
        mu_ym1 = mu_yp1;
        mu_xp1_ym1 = mu_xp1_yp1;
        mu_xm1_ym1 = mu_xm1_yp1;
        mu_ym1_zp1 = mu_yp1_zp1;
        mu_ym1_zm1 = mu_yp1_zm1;
    }
    if (mu_wall_yp) {
        mu_yp1 = mu_ym1;
        mu_xp1_yp1 = mu_xp1_ym1;
        mu_xm1_yp1 = mu_xm1_ym1;
        mu_yp1_zp1 = mu_ym1_zp1;
        mu_yp1_zm1 = mu_ym1_zm1;
    }

    if (mu_wall_zm) {
        mu_zm1 = mu_zp1;
        mu_xp1_zm1 = mu_xp1_zp1;
        mu_xm1_zm1 = mu_xm1_zp1;
        mu_yp1_zm1 = mu_yp1_zp1;
        mu_ym1_zm1 = mu_ym1_zp1;
    }
    if (mu_wall_zp) {
        mu_zp1 = mu_zm1;
        mu_xp1_zp1 = mu_xp1_zm1;
        mu_xm1_zp1 = mu_xm1_zm1;
        mu_yp1_zp1 = mu_yp1_zm1;
        mu_ym1_zp1 = mu_ym1_zm1;
    }

    constexpr dfloat w0 = -4.0_df;
    constexpr dfloat w1 =  1.0_df / 3.0_df;
    constexpr dfloat w2 =  1.0_df / 6.0_df;

    dfloat laplacian_mu =
        w0 * mu0
      + w1 * (mu_xp1 + mu_xm1 + mu_yp1 + mu_ym1 + mu_zp1 + mu_zm1)
      + w2 * (mu_xp1_yp1 + mu_xp1_ym1 + mu_xm1_yp1 + mu_xm1_ym1
            + mu_xp1_zp1 + mu_xp1_zm1 + mu_xm1_zp1 + mu_xm1_zm1
            + mu_yp1_zp1 + mu_yp1_zm1 + mu_ym1_zp1 + mu_ym1_zm1);

    fMom[idxMom(tx, ty, tz, M3_LAPLACIAN_MU_INDEX, bx, by, bz)] = laplacian_mu;
}
#endif // PHASE_TRANSPORT_CAHN_HILLIARD

#ifdef PHASE_NEEDS_NORMAL_HALOS
__global__ void gpuComputePhaseFluxDivergences(
    dfloat *fMom,
    unsigned int *dNodeType,
    macroInterfaceGPUData macroInterfaceGPU,
    size_t localNZ,
    int zStart)
{
    const int x = threadIdx.x + blockDim.x * blockIdx.x;
    const int y = threadIdx.y + blockDim.y * blockIdx.y;
    const int z = threadIdx.z + blockDim.z * blockIdx.z;
    if (x >= NX || y >= NY || z >= (int)localNZ) return;

    const int tx = threadIdx.x, ty = threadIdx.y, tz = threadIdx.z;
    const int bx = blockIdx.x, by = blockIdx.y, bz = blockIdx.z;
    const unsigned int nodeType =
        dNodeType[idxScalarBlock(tx, ty, tz, bx, by, bz)];
    if (nodeType == 0b11111111) return;

    struct PhaseAuxVector {
        dfloat nx, ny, nz;
        dfloat lx, ly, lz;
    };

    auto zeroVector = []() -> PhaseAuxVector {
        return {0.0_df, 0.0_df, 0.0_df, 0.0_df, 0.0_df, 0.0_df};
    };

    auto getVector = [&](int dx, int dy, int dz) -> PhaseAuxVector {
        int nxg = x + dx;
        int nyg = y + dy;
        int nzg = z + dz;

        #ifdef BC_X_WALL
        if (nxg < 0 || nxg >= NX) return zeroVector();
        #else
        nxg = (nxg + NX) % NX;
        #endif
        #ifdef BC_Y_WALL
        if (nyg < 0 || nyg >= NY) return zeroVector();
        #else
        nyg = (nyg + NY) % NY;
        #endif

        const int planeIdx = nxg + nyg * NX;
        dfloat phi, gx, gy, gz;
        if (nzg < 0) {
            #ifdef BC_Z_WALL
            if (zStart == 0) return zeroVector();
            #endif
            phi = macroInterfaceGPU.phi.auxZ_1[planeIdx];
            gx = macroInterfaceGPU.nx.auxZ_1[planeIdx];
            gy = macroInterfaceGPU.ny.auxZ_1[planeIdx];
            gz = macroInterfaceGPU.nz.auxZ_1[planeIdx];
        } else if (nzg >= (int)localNZ) {
            #ifdef BC_Z_WALL
            if (zStart + (int)localNZ >= NZ) return zeroVector();
            #endif
            phi = macroInterfaceGPU.phi.auxZ_0[planeIdx];
            gx = macroInterfaceGPU.nx.auxZ_0[planeIdx];
            gy = macroInterfaceGPU.ny.auxZ_0[planeIdx];
            gz = macroInterfaceGPU.nz.auxZ_0[planeIdx];
        } else {
            const int ntx = nxg % BLOCK_NX, nty = nyg % BLOCK_NY;
            const int ntz = nzg % BLOCK_NZ;
            const int nbx = nxg / BLOCK_NX, nby = nyg / BLOCK_NY;
            const int nbz = nzg / BLOCK_NZ;
            phi = fMom[idxMom(ntx, nty, ntz, M3_PHI_INDEX, nbx, nby, nbz)];
            gx = fMom[idxMom(ntx, nty, ntz, M3_NX_INDEX, nbx, nby, nbz)];
            gy = fMom[idxMom(ntx, nty, ntz, M3_NY_INDEX, nbx, nby, nbz)];
            gz = fMom[idxMom(ntx, nty, ntz, M3_NZ_INDEX, nbx, nby, nbz)];
        }

        const dfloat invNorm = rsqrt(gx*gx + gy*gy + gz*gz
                                   + PHI_NORMAL_EPS*PHI_NORMAL_EPS);
        const dfloat normalX = gx * invNorm;
        const dfloat normalY = gy * invNorm;
        const dfloat normalZ = gz * invNorm;
        const dfloat lambdaPhi =
            4.0_df * (phi - PHI_ONE) * (PHI_TWO - phi)
            / (PHI_INTERFACE_WIDTH * (PHI_TWO - PHI_ONE));
        return {normalX, normalY, normalZ,
                lambdaPhi * normalX,
                lambdaPhi * normalY,
                lambdaPhi * normalZ};
    };

    const PhaseAuxVector xp = getVector( 1, 0, 0), xm = getVector(-1, 0, 0);
    const PhaseAuxVector yp = getVector( 0, 1, 0), ym = getVector( 0,-1, 0);
    const PhaseAuxVector zp = getVector( 0, 0, 1), zm = getVector( 0, 0,-1);
    const PhaseAuxVector xpyp = getVector( 1, 1, 0), xpym = getVector( 1,-1, 0);
    const PhaseAuxVector xmyp = getVector(-1, 1, 0), xmym = getVector(-1,-1, 0);
    const PhaseAuxVector xpzp = getVector( 1, 0, 1), xpzm = getVector( 1, 0,-1);
    const PhaseAuxVector xmzp = getVector(-1, 0, 1), xmzm = getVector(-1, 0,-1);
    const PhaseAuxVector ypzp = getVector( 0, 1, 1), ypzm = getVector( 0, 1,-1);
    const PhaseAuxVector ymzp = getVector( 0,-1, 1), ymzm = getVector( 0,-1,-1);

    constexpr dfloat wa = 1.0_df / 6.0_df;
    constexpr dfloat we = 1.0_df / 12.0_df;
    auto divergence = [&](bool lambdaWeighted) -> dfloat {
        const dfloat axp = lambdaWeighted ? xp.lx : xp.nx;
        const dfloat axm = lambdaWeighted ? xm.lx : xm.nx;
        const dfloat ayp = lambdaWeighted ? yp.ly : yp.ny;
        const dfloat aym = lambdaWeighted ? ym.ly : ym.ny;
        const dfloat azp = lambdaWeighted ? zp.lz : zp.nz;
        const dfloat azm = lambdaWeighted ? zm.lz : zm.nz;
        dfloat result = wa * (axp - axm + ayp - aym + azp - azm);
        #define PHASE_COMPONENT(v, c) (lambdaWeighted ? (v).l##c : (v).n##c)
        result += we * (
            PHASE_COMPONENT(xpyp,x) + PHASE_COMPONENT(xpyp,y)
          + PHASE_COMPONENT(xpym,x) - PHASE_COMPONENT(xpym,y)
          - PHASE_COMPONENT(xmyp,x) + PHASE_COMPONENT(xmyp,y)
          - PHASE_COMPONENT(xmym,x) - PHASE_COMPONENT(xmym,y)
          + PHASE_COMPONENT(xpzp,x) + PHASE_COMPONENT(xpzp,z)
          + PHASE_COMPONENT(xpzm,x) - PHASE_COMPONENT(xpzm,z)
          - PHASE_COMPONENT(xmzp,x) + PHASE_COMPONENT(xmzp,z)
          - PHASE_COMPONENT(xmzm,x) - PHASE_COMPONENT(xmzm,z)
          + PHASE_COMPONENT(ypzp,y) + PHASE_COMPONENT(ypzp,z)
          + PHASE_COMPONENT(ypzm,y) - PHASE_COMPONENT(ypzm,z)
          - PHASE_COMPONENT(ymzp,y) + PHASE_COMPONENT(ymzp,z)
          - PHASE_COMPONENT(ymzm,y) - PHASE_COMPONENT(ymzm,z));
        #undef PHASE_COMPONENT
        return result;
    };

    #ifdef PHASE_TRANSPORT_CONSERVATIVE_ALLEN_CAHN
    fMom[idxMom(tx, ty, tz, M3_DIV_LAMBDA_NORMAL_INDEX, bx, by, bz)] =
        divergence(true);
    #endif
    #ifdef PHASE_FORCE_CSF
    fMom[idxMom(tx, ty, tz, M3_DIV_NORMAL_INDEX, bx, by, bz)] =
        divergence(false);
    #endif
}
#endif // PHASE_NEEDS_NORMAL_HALOS

#endif // PHI_DIST

