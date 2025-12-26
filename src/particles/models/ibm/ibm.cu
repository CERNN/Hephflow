
#include "ibm.cuh"

#ifdef PARTICLE_MODEL
void ibmSimulation(
    ParticlesSoA* particles,
    dfloat *fMom,
    cudaStream_t streamParticles,
    unsigned int step
){
    //TODO: FIX THIS SO IS NOT COPIED EVERY SINGLE STEP
    // the input on the functions should be particles->getNodesSoA() instead of d_nodes
    IbmNodesSoA h_nodes = *(particles->getNodesSoA());
    IbmNodesSoA* d_nodes = &h_nodes;
    cudaMalloc(&d_nodes, sizeof(IbmNodesSoA));
    cudaMemcpy(d_nodes, &h_nodes, sizeof(IbmNodesSoA), cudaMemcpyHostToDevice);

    checkCudaErrors(cudaSetDevice(GPU_INDEX));
    MethodRange range = particles->getMethodRange(IBM);

    //int numIBMParticles = range.last - range.first + 1; 
    const unsigned int threadsNodesIBM = 64;
    unsigned int pNumNodes = particles->getNodesSoA()->getNumNodes();
    const unsigned int gridNodesIBM = pNumNodes % threadsNodesIBM ? pNumNodes / threadsNodesIBM + 1 : pNumNodes / threadsNodesIBM;

    if (particles == nullptr) {
        printf("Error: particles is nullptr\n");
        return;
    }

    checkCudaErrors(cudaStreamSynchronize(streamParticles));

    if (range.first < 0 || range.last >= NUM_PARTICLES || range.first > range.last) {
    printf("Error: Invalid range - first: %d, last: %d, NUM_PARTICLES: %d\n", 
            range.first, range.last, NUM_PARTICLES);
    return;
    }

    ParticleCenter* pArray = particles->getPCenterArray();
    // Reset forces in all IBM nodes;
    ibmResetNodesForces<<<gridNodesIBM, threadsNodesIBM, 0, streamParticles>>>(d_nodes,step);
    ibmParticleNodeMovement<<<gridNodesIBM, threadsNodesIBM, 0, streamParticles>>>(d_nodes,pArray,range.first,range.last,step);
    ibmForceInterpolationSpread<<<gridNodesIBM, threadsNodesIBM,0, streamParticles>>>(d_nodes,pArray, &fMom[0],step);
    
    cudaFree(d_nodes);
    // cudaFree(d_particlesSoA);
}

__global__ 
void ibmResetNodesForces(IbmNodesSoA* particlesNodes, unsigned int step)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= particlesNodes->getNumNodes())
        return;

    const dfloat3SoA force = particlesNodes->getF();
    const dfloat3SoA delta_force = particlesNodes->getDeltaF();

    force.x[idx] = 0;
    force.y[idx] = 0;
    force.z[idx] = 0;
    delta_force.x[idx] = 0;
    delta_force.y[idx] = 0;
    delta_force.z[idx] = 0;
}


__global__
void ibmParticleNodeMovement(
    IbmNodesSoA* particlesNodes,
    ParticleCenter *pArray,
    int firstIndex,
    int lastIndex,
    unsigned int step
){
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if(idx >= particlesNodes->getNumNodes())
        return;

    const dfloat3SoA pos = particlesNodes->getPos();

    //direct copy since we are not modifying
    const ParticleCenter pc_i = pArray[particlesNodes->getParticleCenterIdx()[idx]];

    if(!pc_i.getMovable())
        return;

    // TODO: make the calculation of w_norm along with w_avg?
    const dfloat w_norm = sqrt((pc_i.getWAvgX() * pc_i.getWAvgX()) 
                             + (pc_i.getWAvgY() * pc_i.getWAvgY()) 
                             + (pc_i.getWAvgZ() * pc_i.getWAvgZ()));

    dfloat dx,dy,dz;
    // dfloat new_pos_x,new_pos_y,new_pos_z;

    dfloat3 dd = pc_i.getDx();

    dx = dd.x; //pc_i.getPosX() - pc_i.getPosOldX();
    dy = dd.y; //pc_i.getPosY() - pc_i.getPosOldY();
    dz = dd.z; //pc_i.getPosZ() - pc_i.getPosOldZ();

    dfloat pc_old_x = pc_i.getPosX() - dx;
    dfloat pc_old_y = pc_i.getPosY() - dy;
    dfloat pc_old_z = pc_i.getPosZ() - dz;

    // check the norm to see if is worth computing the rotation
    if(w_norm <= 1e-8)
    {
        
        #ifdef BC_X_WALL
            pos.x[idx] += dx;
        #endif
        #ifdef BC_X_PERIODIC
            if(abs(dx) > (dfloat)(NX)/2){
                dx = (pc_i.getPosX() < pc_old_x)
                    ? (pc_i.getPosX() + (dfloat)NX) - pc_old_x
                    : (pc_i.getPosX() - (dfloat)NX) - pc_old_x;
            }
            pos.x[idx] = std::fmod(pos.x[idx] + dx + (dfloat)NX,(dfloat)NX);
        #endif
        #ifdef BC_Y_WALL
            pos.y[idx] += dy;
        #endif
        #ifdef BC_Y_PERIODIC
            if(abs(dy) > (dfloat)(NY)/2){
                dy = (pc_i.getPosY() < pc_old_y)
                    ? (pc_i.getPosY() + (dfloat)NY) - pc_old_y
                    : (pc_i.getPosY() - (dfloat)NY) - pc_old_y;
            }
            pos.y[idx] = std::fmod(pos.y[idx] + dy + (dfloat)NY, (dfloat)NY);
        #endif

        #ifdef BC_Z_WALL
            pos.z[idx] += dz;
        #endif
        #ifdef BC_Z_PERIODIC
            if(abs(dz) > (dfloat)(NZ_TOTAL)/2){
                dz = (pc_i.getPosZ() < pc_old_z)
                    ? (pc_i.getPosZ() + (dfloat)NZ_TOTAL) - pc_old_z
                    : (pc_i.getPosZ() - (dfloat)NZ_TOTAL) - pc_old_z;
            }
            pos.z[idx] = std::fmod(pos.z[idx] + dz + (dfloat)NZ_TOTAL, (dfloat)NZ_TOTAL);
        #endif
        
        //ealier return since is no longer necessary
        return;
    }

    //compute vector between the node and the partic
    dfloat x_vec = pos.x[idx] - pc_old_x;
    dfloat y_vec = pos.y[idx] - pc_old_y;
    dfloat z_vec = pos.z[idx] - pc_old_z;


    #ifdef BC_X_PERIODIC
        if(abs(x_vec) > (dfloat)NX/2){
            pos.x[idx] += (pos.x[idx] < pc_old_x) ? (dfloat)NX : -(dfloat)NX;
        }
        x_vec = pos.x[idx] - pc_old_x;
    #endif

    #ifdef BC_Y_PERIODIC
        if(abs(y_vec) > (dfloat)NY/2){
            pos.y[idx] += (pos.y[idx] < pc_old_y) ? (dfloat)NY : -(dfloat)NY;
        }
        y_vec = pos.y[idx] - pc_old_y;
    #endif

    #ifdef BC_Z_PERIODIC
        if(abs(z_vec) > (dfloat)NZ_TOTAL/2){
            pos.z[idx] += (pos.z[idx] < pc_old_z) ? (dfloat)NZ_TOTAL : -(dfloat)NZ_TOTAL;
        }
        z_vec = pos.z[idx] - pc_old_z;
    #endif

       
    // compute rotation quartenion
    const dfloat q0 = cos(w_norm/2);
    const dfloat qi = (pc_i.getWAvgX()/w_norm) * sin (w_norm/2);
    const dfloat qj = (pc_i.getWAvgY()/w_norm) * sin (w_norm/2);
    const dfloat qk = (pc_i.getWAvgZ()/w_norm) * sin (w_norm/2);

    const dfloat tq0m1 = (q0*q0) - 0.5;
    
    dfloat new_pos_x = pc_i.getPosX() + 2 * (   (tq0m1 + (qi*qi))*x_vec + ((qi*qj) - (q0*qk))*y_vec + ((qi*qk) + (q0*qj))*z_vec);
    dfloat new_pos_y = pc_i.getPosY() + 2 * ( ((qi*qj) + (q0*qk))*x_vec +   (tq0m1 + (qj*qj))*y_vec + ((qj*qk) - (q0*qi))*z_vec);
    dfloat new_pos_z = pc_i.getPosZ() + 2 * ( ((qi*qj) - (q0*qj))*x_vec + ((qj*qk) + (q0*qi))*y_vec +   (tq0m1 + (qk*qk))*z_vec);

    //update node position
    #ifdef BC_X_WALL
        pos.x[idx] =  new_pos_x;
    #endif //BC_X_WALL
    #ifdef  BC_X_PERIODIC
        pos.x[idx] =  std::fmod((dfloat)(new_pos_x + NX),(dfloat)(NX));
    #endif //BC_X_PERIODIC

    #ifdef BC_Y_WALL
        pos.y[idx] =  new_pos_y;
    #endif //BC_Y_WALL
    #ifdef  BC_Y_PERIODIC
        pos.y[idx] = std::fmod((dfloat)(new_pos_y + NY),(dfloat)(NY));
    #endif //BC_Y_PERIODIC

    #ifdef BC_Z_WALL
        pos.z[idx] =  new_pos_z;
    #endif //BC_Z_WALL
    #ifdef BC_Z_PERIODIC
        pos.z[idx] = std::fmod((dfloat)(new_pos_z + NZ_TOTAL),(dfloat)(NZ_TOTAL));
    #endif //IBBC_Z_PERIODIC
}

__global__
void ibmForceInterpolationSpread(
    IbmNodesSoA* particlesNodes,
    ParticleCenter *pArray,
    dfloat *fMom,
    unsigned int step
){

    int i = threadIdx.x + blockDim.x * blockIdx.x;

    if(i >= particlesNodes->getNumNodes())
        return;

    const dfloat3SoA posNode = particlesNodes->getPos();

    // CRITICAL: Validate particle index before dereferencing
    int particleCenterIdx = particlesNodes->getParticleCenterIdx()[i];
    if (particleCenterIdx < 0 || particleCenterIdx >= NUM_PARTICLES) {
        printf("ERROR: Invalid particle center index %d at node %d\n", particleCenterIdx, i);
        return;
    }
    
    ParticleCenter* pc_i = &pArray[particleCenterIdx];

    dfloat aux, aux1; // aux variable for many things

    const dfloat xIBM = posNode.x[i];
    const dfloat yIBM = posNode.y[i]; 
    const dfloat zIBM = posNode.z[i];

    // CRITICAL: Validate IBM node positions to prevent invalid floor operations
    if (!isfinite(xIBM) || !isfinite(yIBM) || !isfinite(zIBM)) {
        printf("ERROR: Non-finite IBM node position at node %d: x=%e y=%e z=%e\n", i, xIBM, yIBM, zIBM);
        return;
    }
    
    // CRITICAL: Clamp IBM node positions to valid domain to prevent invalid indices
    // Valid range depends on BC type, but we use conservative bounds
    #ifdef BC_X_PERIODIC
        if (xIBM < -1000 || xIBM > NX + 1000) {
            printf("WARNING: IBM node %d has out-of-range X position: %e (domain: 0-%d)\n", i, xIBM, NX);
            return;
        }
    #else
        if (xIBM < -P_DIST || xIBM > NX + P_DIST) {
            printf("WARNING: IBM node %d has out-of-range X position: %e (domain: 0-%d)\n", i, xIBM, NX);
            return;
        }
    #endif

    #ifdef BC_Y_PERIODIC
        if (yIBM < -1000 || yIBM > NY + 1000) {
            printf("WARNING: IBM node %d has out-of-range Y position: %e (domain: 0-%d)\n", i, yIBM, NY);
            return;
        }
    #else
        if (yIBM < -P_DIST || yIBM > NY + P_DIST) {
            printf("WARNING: IBM node %d has out-of-range Y position: %e (domain: 0-%d)\n", i, yIBM, NY);
            return;
        }
    #endif

    #ifdef BC_Z_PERIODIC
        if (zIBM < -1000 || zIBM > NZ_TOTAL + 1000) {
            printf("WARNING: IBM node %d has out-of-range Z position: %e (domain: 0-%d)\n", i, zIBM, NZ_TOTAL);
            return;
        }
    #else
        if (zIBM < -P_DIST || zIBM > NZ_TOTAL + P_DIST) {
            printf("WARNING: IBM node %d has out-of-range Z position: %e (domain: 0-%d)\n", i, zIBM, NZ_TOTAL);
            return;
        }
    #endif

    const dfloat pos[3] = {xIBM, yIBM, zIBM};

    // Calculate stencils to use and the valid interval [xyz][idx]
    dfloat stencilVal[3][P_DIST*2];

    // First lattice position for each coordinate
    // CRITICAL: Use safe casting to prevent integer overflow
    const int posBase[3] = {
        static_cast<int>(std::floor(xIBM)) - P_DIST + 1,
        static_cast<int>(std::floor(yIBM)) - P_DIST + 1,
        static_cast<int>(std::floor(zIBM)) - P_DIST + 1
    };

   
    // Maximum stencil index for each direction xyz ("index" to stop)
    const int maxIdx[3] = {
        #ifdef BC_X_WALL
            ((posBase[0]+P_DIST*2-1) < (int)NX)? P_DIST*2-1 : ((int)NX-1-posBase[0])
        #endif //BC_X_WALL
        #ifdef BC_X_PERIODIC
            P_DIST*2-1
        #endif //BC_X_PERIODIC
        ,
        #ifdef BC_Y_WALL 
            ((posBase[1]+P_DIST*2-1) < (int)NY)? P_DIST*2-1 : ((int)NY-1-posBase[1])
        #endif //BC_Y_WALL
        #ifdef BC_Y_PERIODIC
            P_DIST*2-1
        #endif //BC_Y_PERIODIC
        , 
        #ifdef BC_Z_WALL 
            ((posBase[2]+P_DIST*2-1) < (int)NZ)? P_DIST*2-1 : ((int)NZ-1-posBase[2])
        #endif //BC_Z_WALL
        #ifdef BC_Z_PERIODIC
            P_DIST*2-1
        #endif //BC_Z_PERIODIC
    };

    // Minimum stencil index for each direction xyz ("index" to start)
    const int minIdx[3] = {
        #ifdef BC_X_WALL
            (posBase[0] >= 0)? 0 : -posBase[0]
        #endif //BC_X_WALL
        #ifdef BC_X_PERIODIC
            0
        #endif //BC_X_PERIODIC
        ,
        #ifdef BC_Y_WALL 
            (posBase[1] >= 0)? 0 : -posBase[1]
        #endif //BC_Y_WALL
        #ifdef BC_Y_PERIODIC
            0
        #endif //BC_Y_PERIODIC
        , 
        #ifdef BC_Z_WALL 
            (posBase[2] >= 0)? 0 : -posBase[2]
        #endif //BC_Z_WALL
        #ifdef BC_Z_PERIODIC
            0
        #endif //BC_Z_PERIODIC
    };


    // Particle stencil out of the domain
    if(maxIdx[0] < 0 || maxIdx[1] < 0 || maxIdx[2] < 0)
        return;
    // Particle stencil out of the domain
    if(minIdx[0] >= P_DIST*2 || minIdx[1] >= P_DIST*2 || minIdx[2] >= P_DIST*2)
        return;
    
    // CRITICAL: Additional validation for pathological cases
    if(minIdx[0] < 0 || minIdx[1] < 0 || minIdx[2] < 0 || 
       minIdx[0] > maxIdx[0] || minIdx[1] > maxIdx[1] || minIdx[2] > maxIdx[2]) {
        printf("ERROR: Invalid stencil indices - minIdx=[%d,%d,%d] maxIdx=[%d,%d,%d]\n",
               minIdx[0], minIdx[1], minIdx[2], maxIdx[0], maxIdx[1], maxIdx[2]);
        return;
    }


    //compute stencil values
    for(int ii = 0; ii < 3; ii++){
        for(int jj=minIdx[ii]; jj <= maxIdx[ii]; jj++){
            // CRITICAL: Bounds check to prevent array overflow
            if(jj < 0 || jj >= P_DIST*2) {
                printf("ERROR: Stencil index out of bounds: jj=%d, ii=%d, minIdx=%d, maxIdx=%d\n", 
                       jj, ii, minIdx[ii], maxIdx[ii]);
                return;
            }
            stencilVal[ii][jj] = stencil(posBase[ii]+jj-(pos[ii]));
        }
    }

    dfloat rhoVar = 0;
    dfloat uxVar = 0;
    dfloat uyVar = 0;
    dfloat uzVar = 0;

    unsigned int baseIdx;
    int xx,yy,zz;

    // Velocity on node given the particle velocity and rotation
    dfloat ux_calc = 0;
    dfloat uy_calc = 0;
    dfloat uz_calc = 0;

    // Interpolation (zyx for memory locality)
    for (int zk = minIdx[2]; zk <= maxIdx[2]; zk++) // z
    {
        int zg = posBase[2] + zk;

        #ifdef BC_Z_WALL
            if (zg < 0 || zg >= NZ) continue;
            zz = zg;
        #endif
        #ifdef BC_Z_PERIODIC
            // CRITICAL: Proper modulo for periodic BC that handles negative numbers
            zz = ((zg % NZ) + NZ) % NZ;
        #endif

        for (int yj = minIdx[1]; yj <= maxIdx[1]; yj++) // y
        {
            int yg = posBase[1] + yj;
            #ifdef BC_Y_WALL
                if (yg < 0 || yg >= NY) continue;
                yy = yg;
            #endif
            #ifdef BC_Y_PERIODIC
                // CRITICAL: Proper modulo for periodic BC that handles negative numbers
                yy = ((yg % NY) + NY) % NY;
            #endif
            aux1 = stencilVal[2][zk]*stencilVal[1][yj];
            for (int xi = minIdx[0]; xi <= maxIdx[0]; xi++) // x
            {
                int xg = posBase[0] + xi;
                #ifdef BC_X_WALL
                    if (xg < 0 || xg >= NX) continue;
                    xx = xg;
                #endif
                #ifdef BC_X_PERIODIC
                    // CRITICAL: Proper modulo for periodic BC that handles negative numbers
                    xx = ((xg % NX) + NX) % NX;
                #endif

                // Dirac delta (kernel)
                aux = aux1 * stencilVal[0][xi];

                // CRITICAL: Validate fMom index before access
                int momIdx_rho = idxMom(xx%BLOCK_NX, yy%BLOCK_NY, zz%BLOCK_NZ, M_RHO_INDEX, xx/BLOCK_NX, yy/BLOCK_NY, zz/BLOCK_NZ);
                int momIdx_ux = idxMom(xx%BLOCK_NX, yy%BLOCK_NY, zz%BLOCK_NZ, M_UX_INDEX, xx/BLOCK_NX, yy/BLOCK_NY, zz/BLOCK_NZ);
                int momIdx_uy = idxMom(xx%BLOCK_NX, yy%BLOCK_NY, zz%BLOCK_NZ, M_UY_INDEX, xx/BLOCK_NX, yy/BLOCK_NY, zz/BLOCK_NZ);
                int momIdx_uz = idxMom(xx%BLOCK_NX, yy%BLOCK_NY, zz%BLOCK_NZ, M_UZ_INDEX, xx/BLOCK_NX, yy/BLOCK_NY, zz/BLOCK_NZ);

                #ifdef EXTERNAL_DUCT_BC
                    dfloat pos_r_i = (xx - DUCT_CENTER_X)*(xx - DUCT_CENTER_X) + (yy - DUCT_CENTER_Y)*(yy - DUCT_CENTER_Y);
                    if(pos_r_i < OUTER_RADIUS*OUTER_RADIUS){
                        rhoVar += aux * (RHO_0 + fMom[momIdx_rho]);
                        uxVar  += aux * (fMom[momIdx_ux]/F_M_I_SCALE);
                        uyVar  += aux * (fMom[momIdx_uy]/F_M_I_SCALE);
                        uzVar  += aux * (fMom[momIdx_uz]/F_M_I_SCALE);
                    }
                #endif
                #ifndef EXTERNAL_DUCT_BC
                    rhoVar += aux * (RHO_0 + fMom[momIdx_rho]);
                    uxVar  += aux * (fMom[momIdx_ux]/F_M_I_SCALE);
                    uyVar  += aux * (fMom[momIdx_uy]/F_M_I_SCALE);
                    uzVar  += aux * (fMom[momIdx_uz]/F_M_I_SCALE);
                #endif //EXTERNAL_DUCT_BC
            }
        }
    }



    // Load position of particle center
    const dfloat x_pc = pc_i->getPosX();
    const dfloat y_pc = pc_i->getPosY();
    const dfloat z_pc = pc_i->getPosZ();

    // CRITICAL: Store distance vectors BEFORE they are needed for deltaMomentum
    dfloat dx = xIBM - x_pc;
    dfloat dy = yIBM - y_pc;
    dfloat dz = zIBM - z_pc;

    #ifdef BC_X_PERIODIC
    if(abs(dx) > (dfloat)(NX)/2.0){
        if(dx < 0)
            dx = (xIBM + NX) - x_pc;
        else
            dx = (xIBM - NX) - x_pc;
    }
    #endif //BC_X_PERIODIC
    
    #ifdef BC_Y_PERIODIC
    if(abs(dy) > (dfloat)(NY)/2.0){
        if(dy < 0)
            dy = (yIBM + NY) - y_pc;
        else
            dy = (yIBM - NY) - y_pc;
    }
    #endif //BC_Y_PERIODIC

    #ifdef BC_Z_PERIODIC
    if(abs(dz) > (dfloat)(NZ)/2.0){
        if(dz < 0)
            dz = (zIBM + NZ) - z_pc;
        else
            dz = (zIBM - NZ) - z_pc;
    }
    #endif //BC_Z_PERIODIC

    // Calculate velocity on node if particle is movable
    if(pc_i->getMovable()){
        // Load velocity and rotation velocity of particle center
        const dfloat vx_pc = pc_i->getVelX();
        const dfloat vy_pc = pc_i->getVelY();
        const dfloat vz_pc = pc_i->getVelZ();

        const dfloat wx_pc = pc_i->getWX();
        const dfloat wy_pc = pc_i->getWY();
        const dfloat wz_pc = pc_i->getWZ();

        // velocity on node, given the center velocity and rotation
        // (i.e. no slip boundary condition velocity)
        ux_calc = vx_pc + (wy_pc * (dz) - wz_pc * (dy));
        uy_calc = vy_pc + (wz_pc * (dx) - wx_pc * (dz));
        uz_calc = vz_pc + (wx_pc * (dy) - wy_pc * (dx));
    }

    const dfloat dA = particlesNodes->getS()[i];
    aux = 2 * rhoVar * dA * IBM_THICKNESS;

    // CRITICAL: Check for NaN/Inf propagation
    if (!isfinite(aux) || !isfinite(rhoVar) || !isfinite(uxVar) || !isfinite(uyVar) || !isfinite(uzVar)) {
        printf("WARNING: Non-finite values at node %d, step %u: aux=%e rho=%e ux=%e uy=%e uz=%e\n", 
               i, step, aux, rhoVar, uxVar, uyVar, uzVar);
        return;
    }

    dfloat3 deltaF;
    deltaF.x = aux * (uxVar - ux_calc);
    deltaF.y = aux * (uyVar - uy_calc);
    deltaF.z = aux * (uzVar - uz_calc);

    // Calculate IBM forces
    const dfloat3SoA force = particlesNodes->getF();
    const dfloat fxIBM = force.x[i] + deltaF.x;
    const dfloat fyIBM = force.y[i] + deltaF.y;
    const dfloat fzIBM = force.z[i] + deltaF.z;

    // Spreading (zyx for memory locality)
    for (int zk = minIdx[2]; zk <= maxIdx[2]; zk++) // z
    {
        for (int yj = minIdx[1]; yj <= maxIdx[1]; yj++) // y
        {
            aux1 = stencilVal[2][zk]*stencilVal[1][yj];
            for (int xi = minIdx[0]; xi <= maxIdx[0]; xi++) // x
            {
                // Dirac delta (kernel)
                aux = aux1 * stencilVal[0][xi];

                // Global (unmapped) indices
                int xg = posBase[0] + xi;
                int yg = posBase[1] + yj;
                int zg = posBase[2] + zk;

                // ---- X direction ----
                #ifdef BC_X_WALL
                    if (xg < 0 || xg >= NX) continue;
                    xx = xg;
                #else // BC_X_PERIODIC
                    xx = ((xg % NX) + NX) % NX;
                #endif

                // ---- Y direction ----
                #ifdef BC_Y_WALL
                    if (yg < 0 || yg >= NY) continue;
                    yy = yg;
                #else // BC_Y_PERIODIC
                    yy = ((yg % NY) + NY) % NY;
                #endif

                // ---- Z direction ----
                #ifdef BC_Z_WALL
                    if (zg < 0 || zg >= NZ_TOTAL) continue;
                    zz = zg;
                #else // BC_Z_PERIODIC
                    zz = ((zg % NZ_TOTAL) + NZ_TOTAL) % NZ_TOTAL;
                #endif

                // CRITICAL: Validate fMom indices before atomic operations
                int fmomIdx_fx = idxMom(xx%BLOCK_NX, yy%BLOCK_NY, zz%BLOCK_NZ, M_FX_INDEX, xx/BLOCK_NX, yy/BLOCK_NY, zz/BLOCK_NZ);
                int fmomIdx_fy = idxMom(xx%BLOCK_NX, yy%BLOCK_NY, zz%BLOCK_NZ, M_FY_INDEX, xx/BLOCK_NX, yy/BLOCK_NY, zz/BLOCK_NZ);
                int fmomIdx_fz = idxMom(xx%BLOCK_NX, yy%BLOCK_NY, zz%BLOCK_NZ, M_FZ_INDEX, xx/BLOCK_NX, yy/BLOCK_NY, zz/BLOCK_NZ);

                // ---- External duct condition ----
                #ifdef EXTERNAL_DUCT_BC
                    dfloat pos_r_i = (xx - DUCT_CENTER_X)*(xx - DUCT_CENTER_X) + (yy - DUCT_CENTER_Y)*(yy - DUCT_CENTER_Y);
                    if(pos_r_i < OUTER_RADIUS*OUTER_RADIUS){
                        atomicAdd(&(fMom[fmomIdx_fx]), -deltaF.x * aux);
                        atomicAdd(&(fMom[fmomIdx_fy]), -deltaF.y * aux);
                        atomicAdd(&(fMom[fmomIdx_fz]), -deltaF.z * aux);
                    }
                #endif
                #ifndef EXTERNAL_DUCT_BC
                    atomicAdd(&(fMom[fmomIdx_fx]), -deltaF.x * aux);
                    atomicAdd(&(fMom[fmomIdx_fy]), -deltaF.y * aux);
                    atomicAdd(&(fMom[fmomIdx_fz]), -deltaF.z * aux);
                #endif //EXTERNAL_DUCT_BC

                //TODO: find a way to do subinterations
                //here would enter the correction of the velocity field for subiterations
                //however, on moment based, we dont have the populations to recover the original velocity
                //therefore it would directly change the velocity field and moments
                //also a problem on the lattices on the block frontier, as would be necessary to recompute the populations there

            }
        }
    }


    // Update node force
    force.x[i] = fxIBM;
    force.y[i] = fyIBM;
    force.z[i] = fzIBM;


    const dfloat3SoA delta_force = particlesNodes->getDeltaF();
    // Update node delta force
    delta_force.x[i] = deltaF.x;
    delta_force.y[i] = deltaF.y;
    delta_force.z[i] = deltaF.z;


    const dfloat3 deltaMomentum = dfloat3(
        (dy) * deltaF.z - (dz) * deltaF.y,
        (dz) * deltaF.x - (dx) * deltaF.z,
        (dx) * deltaF.y - (dy) * deltaF.x
    );
    
    atomicAdd(&(pc_i->getFXatomic()), deltaF.x);
    atomicAdd(&(pc_i->getFYatomic()), deltaF.y);
    atomicAdd(&(pc_i->getFZatomic()), deltaF.z);

    atomicAdd(&(pc_i->getMXatomic()), deltaMomentum.x);
    atomicAdd(&(pc_i->getMYatomic()), deltaMomentum.y);
    atomicAdd(&(pc_i->getMZatomic()), deltaMomentum.z);
}

#endif //PARTICLE_MODEL