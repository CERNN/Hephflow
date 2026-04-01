
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
    const dfloat3SoA originalRelativePos = particlesNodes->getOriginalRelativePos();

    //direct copy since we are not modifying
    const ParticleCenter pc_i = pArray[particlesNodes->getParticleCenterIdx()[idx]];

    if(!pc_i.getMovable())
        return;

    // Get the original relative position (immutable reference set at initialization)
    dfloat3 original_offset = dfloat3(
        originalRelativePos.x[idx],
        originalRelativePos.y[idx],
        originalRelativePos.z[idx]
    );
    
    // Fetch the cumulative rotation quaternion from particle center
    dfloat4 q_cumulative = pc_i.getQ_cumulative_rot();
    
    // Rotate the original offset using the accumulated rotation
    dfloat3 rotated_offset = rotate_vector_by_quart_R(original_offset, q_cumulative);
    
    // Reconstruct node position from first principles
    dfloat new_pos_x = pc_i.getPosX() + rotated_offset.x;
    dfloat new_pos_y = pc_i.getPosY() + rotated_offset.y;
    dfloat new_pos_z = pc_i.getPosZ() + rotated_offset.z;
    
    // Apply boundary conditions AFTER rotation to final position
    #ifdef BC_X_WALL
        pos.x[idx] = new_pos_x;
    #endif
    #ifdef BC_X_PERIODIC
        pos.x[idx] = std::fmod((dfloat)(new_pos_x + NX), (dfloat)(NX));
        if (pos.x[idx] < 0) pos.x[idx] += (dfloat)NX;
    #endif

    #ifdef BC_Y_WALL
        pos.y[idx] = new_pos_y;
    #endif
    #ifdef BC_Y_PERIODIC
        pos.y[idx] = std::fmod((dfloat)(new_pos_y + NY), (dfloat)(NY));
        if (pos.y[idx] < 0) pos.y[idx] += (dfloat)NY;
    #endif

    #ifdef BC_Z_WALL
        pos.z[idx] = new_pos_z;
    #endif
    #ifdef BC_Z_PERIODIC
        pos.z[idx] = std::fmod((dfloat)(new_pos_z + NZ_TOTAL), (dfloat)(NZ_TOTAL));
        if (pos.z[idx] < 0) pos.z[idx] += (dfloat)NZ_TOTAL;
    #endif
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

    int particleCenterIdx = particlesNodes->getParticleCenterIdx()[i];
    ParticleCenter* pc_i = &pArray[particleCenterIdx];

    dfloat aux, aux1; // aux variable for many things

    const dfloat xIBM = posNode.x[i];
    const dfloat yIBM = posNode.y[i]; 
    const dfloat zIBM = posNode.z[i];

    const dfloat pos[3] = {xIBM, yIBM, zIBM};

    // Calculate stencils to use and the valid interval [xyz][idx]
    dfloat stencilVal[3][P_DIST*2];

    // First lattice position for each coordinate
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
                    xx = ((xg % NX) + NX) % NX;
                #endif

                // Dirac delta (kernel)
                aux = aux1 * stencilVal[0][xi];

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