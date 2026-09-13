
#include "ibm.cuh"

#ifdef PARTICLE_FORCE_DEBUG
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <vector>
#endif

#ifdef PARTICLE_MODEL

namespace {
static_assert(P_DIST * 2 <= IBM_CACHED_STENCIL_WIDTH,
    "IBM stencil cache is too small for the selected stencil");

__device__ __forceinline__ size_t ibmStencilWeightIndex(
    int axis, int offset, unsigned int node, unsigned int numNodes)
{
    return static_cast<size_t>(axis * IBM_CACHED_STENCIL_WIDTH + offset) *
        numNodes + node;
}
} // namespace

#ifdef PARTICLE_FORCE_DEBUG
namespace {
bool ibmDebugStep(unsigned int step) {
    return step - PARTICLE_FORCE_DEBUG_START_STEP <=
           PARTICLE_FORCE_DEBUG_END_STEP - PARTICLE_FORCE_DEBUG_START_STEP;
}

double debugMagnitude(const dfloat3& value) {
    return std::sqrt(
        static_cast<double>(value.x) * value.x +
        static_cast<double>(value.y) * value.y +
        static_cast<double>(value.z) * value.z);
}

void writeDebugVec3(std::ofstream& out, const dfloat3& value) {
    out << ',' << value.x << ',' << value.y << ',' << value.z;
}

void exportIbmNodeDebug(
    const std::vector<IbmNodeDebugRecord>& records,
    unsigned int step,
    int iteration
) {
    static bool initialized = false;
    const std::ios::openmode mode = initialized ? std::ios::app : std::ios::trunc;
    std::ofstream out("particle_ibm_node_debug.csv", mode);
    if (!out) {
        std::fprintf(stderr, "ERROR: Could not open particle_ibm_node_debug.csv\n");
        return;
    }

    if (!initialized) {
        out << "step,iteration,particle,node_count,valid_nodes,invalid_nodes"
            << ",nonfinite_nodes,clipped_nodes,min_weight_sum,max_weight_sum"
            << ",worst_node,worst_stencil_status,worst_clipped_points"
            << ",worst_min_x,worst_min_y,worst_min_z"
            << ",worst_max_x,worst_max_y,worst_max_z"
            << ",pos_x,pos_y,pos_z,rho,force_scale,weight_sum"
            << ",fluid_ux,fluid_uy,fluid_uz"
            << ",rigid_ux,rigid_uy,rigid_uz"
            << ",slip_x,slip_y,slip_z"
            << ",delta_fx,delta_fy,delta_fz,delta_f_mag\n";
        initialized = true;
    }
    out << std::scientific << std::setprecision(9);

    for (int particle = 0; particle < NUM_PARTICLES; ++particle) {
        int nodeCount = 0;
        int validNodes = 0;
        int invalidStencilNodes = 0;
        int nonfiniteNodes = 0;
        int clippedNodes = 0;
        dfloat minWeight = std::numeric_limits<dfloat>::infinity();
        dfloat maxWeight = -std::numeric_limits<dfloat>::infinity();
        const IbmNodeDebugRecord* worst = nullptr;
        double worstMagnitude = -1.0;

        for (const IbmNodeDebugRecord& record : records) {
            if (record.particleIndex != particle) continue;
            ++nodeCount;
            if (record.stencilStatus != 0) {
                ++invalidStencilNodes;
                if (worst == nullptr) worst = &record;
                continue;
            }

            ++validNodes;
            if (record.clippedPoints > 0) ++clippedNodes;
            if (std::isfinite(static_cast<double>(record.stencilWeightSum))) {
                if (record.stencilWeightSum < minWeight) minWeight = record.stencilWeightSum;
                if (record.stencilWeightSum > maxWeight) maxWeight = record.stencilWeightSum;
            }

            const double magnitude = debugMagnitude(record.deltaForce);
            const bool finite = std::isfinite(magnitude) &&
                std::isfinite(static_cast<double>(record.rho)) &&
                std::isfinite(static_cast<double>(record.forceScale)) &&
                std::isfinite(static_cast<double>(record.stencilWeightSum)) &&
                std::isfinite(debugMagnitude(record.fluidVelocity)) &&
                std::isfinite(debugMagnitude(record.rigidVelocity));
            if (!finite) ++nonfiniteNodes;
            if ((!finite && nonfiniteNodes == 1) ||
                (finite && nonfiniteNodes == 0 && magnitude > worstMagnitude)) {
                worst = &record;
                worstMagnitude = magnitude;
            }
        }

        if (validNodes == 0) {
            minWeight = 0;
            maxWeight = 0;
        }
        if (worst == nullptr) continue;

        const dfloat3 slip = worst->fluidVelocity - worst->rigidVelocity;
        out << step << ',' << iteration << ',' << particle << ',' << nodeCount << ',' << validNodes
            << ',' << invalidStencilNodes << ',' << nonfiniteNodes << ',' << clippedNodes
            << ',' << minWeight << ',' << maxWeight
            << ',' << worst->nodeIndex << ',' << worst->stencilStatus
            << ',' << worst->clippedPoints;
        for (int axis = 0; axis < 3; ++axis) out << ',' << worst->minIdx[axis];
        for (int axis = 0; axis < 3; ++axis) out << ',' << worst->maxIdx[axis];
        writeDebugVec3(out, worst->position);
        out << ',' << worst->rho << ',' << worst->forceScale
            << ',' << worst->stencilWeightSum;
        writeDebugVec3(out, worst->fluidVelocity);
        writeDebugVec3(out, worst->rigidVelocity);
        writeDebugVec3(out, slip);
        writeDebugVec3(out, worst->deltaForce);
        out << ',' << debugMagnitude(worst->deltaForce) << '\n';
    }
}
} // namespace
#endif

#ifndef IBM_UNIT_TEST
void ibmSimulation(
    ParticlesSoA* particles,
    dfloat *fMom,
    cudaStream_t streamParticles,
    unsigned int step
){
    if (particles == nullptr) {
        printf("Error: particles is nullptr\n");
        return;
    }

    checkCudaErrors(cudaSetDevice(GPU_INDEX));
    const unsigned int threadsNodesIBM = 64;
    IbmNodesSoA* nodesSoA = particles->getNodesSoA();
    const unsigned int pNumNodes = nodesSoA->getNumNodes();
    if (pNumNodes == 0) return;
    const unsigned int gridNodesIBM =
        (pNumNodes + threadsNodesIBM - 1) / threadsNodesIBM;
    const IbmNodesView nodes = nodesSoA->getView();

    ParticleCenter* pArray = particles->getPCenterArray();
    ibmPrepareNodes<<<gridNodesIBM, threadsNodesIBM, 0, streamParticles>>>(nodes, pArray);
    IbmNodeDebugRecord* d_debugRecords = nullptr;
    #ifdef PARTICLE_FORCE_DEBUG
    std::vector<IbmNodeDebugRecord> debugRecords;
    if (ibmDebugStep(step)) {
        debugRecords.resize(pNumNodes);
        checkCudaErrors(cudaMalloc(&d_debugRecords, pNumNodes * sizeof(IbmNodeDebugRecord)));
    }
    #endif

    for (int iteration = 0; iteration < IBM_MAX_ITERATION; ++iteration) {
        ibmForceInterpolationSpread<<<gridNodesIBM, threadsNodesIBM, 0, streamParticles>>>(
            nodes, pArray, fMom, step, d_debugRecords);

        #ifdef PARTICLE_FORCE_DEBUG
        if (d_debugRecords != nullptr) {
            checkCudaErrors(cudaMemcpyAsync(
                debugRecords.data(), d_debugRecords,
                pNumNodes * sizeof(IbmNodeDebugRecord),
                cudaMemcpyDeviceToHost, streamParticles));
            checkCudaErrors(cudaStreamSynchronize(streamParticles));
            exportIbmNodeDebug(debugRecords, step, iteration);
        }
        #endif

        // Same-stream launch ordering guarantees that spreading is complete
        // before the next iteration reinterpolates the predicted velocity.
        ibmSpreadForceCorrection<<<gridNodesIBM, threadsNodesIBM, 0, streamParticles>>>(
            nodes, fMom);
    }

    #ifdef PARTICLE_FORCE_DEBUG
    if (d_debugRecords != nullptr) checkCudaErrors(cudaFree(d_debugRecords));
    #endif
}
#endif

__global__ 
void ibmPrepareNodes(IbmNodesView particlesNodes, ParticleCenter* pArray)
{
    const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= particlesNodes.numNodes) return;

    const IbmVectorView force = particlesNodes.f;
    const IbmVectorView delta_force = particlesNodes.deltaF;
    force.x[idx] = 0;
    force.y[idx] = 0;
    force.z[idx] = 0;
    delta_force.x[idx] = 0;
    delta_force.y[idx] = 0;
    delta_force.z[idx] = 0;

    const IbmVectorView pos = particlesNodes.pos;
    const ParticleCenter pc_i = pArray[particlesNodes.particleCenterIdx[idx]];
    if (pc_i.getMovable()) {
        const IbmVectorView originalRelativePos = particlesNodes.originalRelativePos;
        const dfloat3 originalOffset(
            originalRelativePos.x[idx],
            originalRelativePos.y[idx],
            originalRelativePos.z[idx]);
        const dfloat3 rotatedOffset = rotate_vector_by_quart_R(
            originalOffset, pc_i.getQ_cumulative_rot());
        const dfloat newPosX = pc_i.getPosX() + rotatedOffset.x;
        const dfloat newPosY = pc_i.getPosY() + rotatedOffset.y;
        const dfloat newPosZ = pc_i.getPosZ() + rotatedOffset.z;

        #ifdef BC_X_WALL
            pos.x[idx] = newPosX;
        #endif
        #ifdef BC_X_PERIODIC
            pos.x[idx] = std::fmod((dfloat)(newPosX + NX), (dfloat)(NX));
            if (pos.x[idx] < 0) pos.x[idx] += (dfloat)NX;
        #endif
        #ifdef BC_Y_WALL
            pos.y[idx] = newPosY;
        #endif
        #ifdef BC_Y_PERIODIC
            pos.y[idx] = std::fmod((dfloat)(newPosY + NY), (dfloat)(NY));
            if (pos.y[idx] < 0) pos.y[idx] += (dfloat)NY;
        #endif
        #ifdef BC_Z_WALL
            pos.z[idx] = newPosZ;
        #endif
        #ifdef BC_Z_PERIODIC
            pos.z[idx] = std::fmod((dfloat)(newPosZ + NZ_TOTAL), (dfloat)(NZ_TOTAL));
            if (pos.z[idx] < 0) pos.z[idx] += (dfloat)NZ_TOTAL;
        #endif
    }

    const dfloat markerPosition[3] = {pos.x[idx], pos.y[idx], pos.z[idx]};
    if (!isfinite(markerPosition[0]) || !isfinite(markerPosition[1]) ||
        !isfinite(markerPosition[2])) return;

    const int3 base = make_int3(
        static_cast<int>(floor(markerPosition[0])) - P_DIST + 1,
        static_cast<int>(floor(markerPosition[1])) - P_DIST + 1,
        static_cast<int>(floor(markerPosition[2])) - P_DIST + 1);
    particlesNodes.stencilBase[idx] = base;
    const int baseAxis[3] = {base.x, base.y, base.z};
    for (int axis = 0; axis < 3; ++axis) {
        for (int offset = 0; offset < P_DIST * 2; ++offset) {
            particlesNodes.stencilWeights[ibmStencilWeightIndex(
                axis, offset, idx, particlesNodes.numNodes)] =
                stencil(baseAxis[axis] + offset - markerPosition[axis]);
        }
    }
}

__global__
void ibmForceInterpolationSpread(
    IbmNodesView particlesNodes,
    ParticleCenter *pArray,
    dfloat *fMom,
    unsigned int step,
    IbmNodeDebugRecord* debugRecords
){

    int i = threadIdx.x + blockDim.x * blockIdx.x;

    if(i >= particlesNodes.numNodes)
        return;

    const IbmVectorView posNode = particlesNodes.pos;

    int particleCenterIdx = particlesNodes.particleCenterIdx[i];
    ParticleCenter* pc_i = &pArray[particleCenterIdx];

    dfloat aux, aux1; // aux variable for many things

    const dfloat xIBM = posNode.x[i];
    const dfloat yIBM = posNode.y[i]; 
    const dfloat zIBM = posNode.z[i];

    #ifdef PARTICLE_FORCE_DEBUG
    IbmNodeDebugRecord debugRecord = {};
    debugRecord.nodeIndex = i;
    debugRecord.particleIndex = particleCenterIdx;
    debugRecord.stencilStatus = IBM_DEBUG_NONFINITE_POSITION;
    debugRecord.position = dfloat3(xIBM, yIBM, zIBM);
    #endif
    if (!isfinite(xIBM) || !isfinite(yIBM) || !isfinite(zIBM)) {
        #ifdef PARTICLE_FORCE_DEBUG
        if (debugRecords != nullptr) debugRecords[i] = debugRecord;
        #endif
        return;
    }

    // Calculate stencils to use and the valid interval [xyz][idx]
    dfloat stencilVal[3][P_DIST*2];

    // Prepared once per time step and shared by interpolation and spreading.
    const int3 cachedBase = particlesNodes.stencilBase[i];
    const int posBase[3] = {
        cachedBase.x, cachedBase.y, cachedBase.z
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

    #ifdef PARTICLE_FORCE_DEBUG
        for (int axis = 0; axis < 3; ++axis) {
            debugRecord.minIdx[axis] = minIdx[axis];
            debugRecord.maxIdx[axis] = maxIdx[axis];
        }
    #endif


    // Particle stencil out of the domain
    if(maxIdx[0] < 0 || maxIdx[1] < 0 || maxIdx[2] < 0) {
        #ifdef PARTICLE_FORCE_DEBUG
        debugRecord.stencilStatus = IBM_DEBUG_OUTSIDE_DOMAIN;
        if (debugRecords != nullptr) debugRecords[i] = debugRecord;
        #endif
        return;
    }
    // Particle stencil out of the domain
    if(minIdx[0] >= P_DIST*2 || minIdx[1] >= P_DIST*2 || minIdx[2] >= P_DIST*2) {
        #ifdef PARTICLE_FORCE_DEBUG
        debugRecord.stencilStatus = IBM_DEBUG_OUTSIDE_DOMAIN;
        if (debugRecords != nullptr) debugRecords[i] = debugRecord;
        #endif
        return;
    }
    
    // CRITICAL: Additional validation for pathological cases
    if(minIdx[0] < 0 || minIdx[1] < 0 || minIdx[2] < 0 || 
       minIdx[0] > maxIdx[0] || minIdx[1] > maxIdx[1] || minIdx[2] > maxIdx[2]) {
       // printf("ERROR: Invalid stencil indices - minIdx=[%d,%d,%d] maxIdx=[%d,%d,%d]\n",
       //        minIdx[0], minIdx[1], minIdx[2], maxIdx[0], maxIdx[1], maxIdx[2]);
        #ifdef PARTICLE_FORCE_DEBUG
            debugRecord.stencilStatus = IBM_DEBUG_INVALID_STENCIL;
            if (debugRecords != nullptr) debugRecords[i] = debugRecord;
        #endif
        return;
    }

    #ifdef PARTICLE_FORCE_DEBUG
        debugRecord.stencilStatus = IBM_DEBUG_VALID;
        const int retainedPoints =
            (maxIdx[0] - minIdx[0] + 1) *
            (maxIdx[1] - minIdx[1] + 1) *
            (maxIdx[2] - minIdx[2] + 1);
        const int stencilWidth = P_DIST * 2;
        debugRecord.clippedPoints =
            stencilWidth * stencilWidth * stencilWidth - retainedPoints;
    #endif


    // Load cached separable stencil values.
    for(int ii = 0; ii < 3; ii++){
        for(int jj=minIdx[ii]; jj <= maxIdx[ii]; jj++){
            stencilVal[ii][jj] = particlesNodes.stencilWeights[
                ibmStencilWeightIndex(ii, jj, i, particlesNodes.numNodes)];
        }
    }

    dfloat rhoVar = 0;
    dfloat uxVar = 0;
    dfloat uyVar = 0;
    dfloat uzVar = 0;
    bool invalidEulerianState = false;
    #ifdef PARTICLE_FORCE_DEBUG
        dfloat stencilWeightSum = 0;
    #endif

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

                const int tx = xx % BLOCK_NX;
                const int ty = yy % BLOCK_NY;
                const int tz = zz % BLOCK_NZ;
                const int bx = xx / BLOCK_NX;
                const int by = yy / BLOCK_NY;
                const int bz = zz / BLOCK_NZ;
                const size_t blockMomentBase = idxMom(
                    tx, ty, tz, M_RHO_INDEX, bx, by, bz);
                const size_t momIdx_rho = blockMomentBase;
                const size_t momIdx_ux = blockMomentBase + M_UX_INDEX * BLOCK_LBM_SIZE;
                const size_t momIdx_uy = blockMomentBase + M_UY_INDEX * BLOCK_LBM_SIZE;
                const size_t momIdx_uz = blockMomentBase + M_UZ_INDEX * BLOCK_LBM_SIZE;
                const size_t momIdx_fx = blockMomentBase + M_FX_INDEX * BLOCK_LBM_SIZE;
                const size_t momIdx_fy = blockMomentBase + M_FY_INDEX * BLOCK_LBM_SIZE;
                const size_t momIdx_fz = blockMomentBase + M_FZ_INDEX * BLOCK_LBM_SIZE;

                const dfloat rhoNode = RHO_0 + fMom[momIdx_rho];
                if (!isfinite(rhoNode) || rhoNode <= 1.0e-6_df) {
                    invalidEulerianState = true;
                    continue;
                }

                // M_F contains the force accumulated by previous MDF stages.
                // The collision operator changes physical velocity by F/rho,
                // so use that response to predict the velocity seen by the
                // next interpolation without modifying M_U in place.
                const dfloat uxPredicted = fMom[momIdx_ux] / F_M_I_SCALE +
                    (fMom[momIdx_fx] - FX) / rhoNode;
                const dfloat uyPredicted = fMom[momIdx_uy] / F_M_I_SCALE +
                    (fMom[momIdx_fy] - FY) / rhoNode;
                const dfloat uzPredicted = fMom[momIdx_uz] / F_M_I_SCALE +
                    (fMom[momIdx_fz] - FZ) / rhoNode;

                #ifdef EXTERNAL_DUCT_BC
                    dfloat pos_r_i = (xx - DUCT_CENTER_X)*(xx - DUCT_CENTER_X) + (yy - DUCT_CENTER_Y)*(yy - DUCT_CENTER_Y);
                    if(pos_r_i < OUTER_RADIUS*OUTER_RADIUS){
                        #ifdef PARTICLE_FORCE_DEBUG
                        stencilWeightSum += aux;
                        #endif
                        rhoVar += aux * rhoNode;
                        uxVar  += aux * uxPredicted;
                        uyVar  += aux * uyPredicted;
                        uzVar  += aux * uzPredicted;
                    }
                #endif
                #ifndef EXTERNAL_DUCT_BC
                    #ifdef PARTICLE_FORCE_DEBUG
                    stencilWeightSum += aux;
                    #endif
                    rhoVar += aux * rhoNode;
                    uxVar  += aux * uxPredicted;
                    uyVar  += aux * uyPredicted;
                    uzVar  += aux * uzPredicted;
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

    const dfloat dA = particlesNodes.S[i];
    aux = IBM_FORCE_RELAXATION * 2 * rhoVar * dA * IBM_THICKNESS;

    const dfloat3 velocityResidual(
        uxVar - ux_calc,
        uyVar - uy_calc,
        uzVar - uz_calc);
    const dfloat residualSquared = dot_product(velocityResidual, velocityResidual);
    const bool validCorrection = !invalidEulerianState &&
        isfinite(rhoVar) && rhoVar > 1.0e-6_df &&
        isfinite(aux) && isfinite(residualSquared);
    dfloat3 deltaF(0, 0, 0);
    if (validCorrection && residualSquared > IBM_VELOCITY_TOL * IBM_VELOCITY_TOL) {
        deltaF = aux * velocityResidual;
    }

    #ifdef PARTICLE_FORCE_DEBUG
        if (!validCorrection) {
            debugRecord.stencilStatus = IBM_DEBUG_INVALID_EULERIAN_STATE;
        }
    #endif

    #ifdef PARTICLE_FORCE_DEBUG
        if (debugRecords != nullptr) {
            debugRecord.fluidVelocity = dfloat3(uxVar, uyVar, uzVar);
            debugRecord.rigidVelocity = dfloat3(ux_calc, uy_calc, uz_calc);
            debugRecord.deltaForce = deltaF;
            debugRecord.rho = rhoVar;
            debugRecord.forceScale = aux;
            debugRecord.stencilWeightSum = stencilWeightSum;
            debugRecords[i] = debugRecord;
        }
    #endif

    // Calculate IBM forces
    const IbmVectorView force = particlesNodes.f;
    const dfloat fxIBM = force.x[i] + deltaF.x;
    const dfloat fyIBM = force.y[i] + deltaF.y;
    const dfloat fzIBM = force.z[i] + deltaF.z;

    // Update node force
    force.x[i] = fxIBM;
    force.y[i] = fyIBM;
    force.z[i] = fzIBM;


    const IbmVectorView delta_force = particlesNodes.deltaF;
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

__global__
void ibmSpreadForceCorrection(
    IbmNodesView particlesNodes,
    dfloat *fMom
) {
    const int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i >= particlesNodes.numNodes) return;

    const IbmVectorView posNode = particlesNodes.pos;
    const IbmVectorView deltaForce = particlesNodes.deltaF;
    const dfloat xIBM = posNode.x[i];
    const dfloat yIBM = posNode.y[i];
    const dfloat zIBM = posNode.z[i];
    if (!isfinite(xIBM) || !isfinite(yIBM) || !isfinite(zIBM)) return;

    const dfloat3 correction(
        deltaForce.x[i], deltaForce.y[i], deltaForce.z[i]);
    if (!isfinite(correction.x) || !isfinite(correction.y) ||
        !isfinite(correction.z)) return;
    if (correction.x == 0 && correction.y == 0 && correction.z == 0) return;

    constexpr int stencilWidth = P_DIST * 2;
    const int3 cachedBase = particlesNodes.stencilBase[i];
    const int posBase[3] = {cachedBase.x, cachedBase.y, cachedBase.z};
    dfloat stencilVal[3][stencilWidth];
    for (int axis = 0; axis < 3; ++axis) {
        for (int offset = 0; offset < stencilWidth; ++offset) {
            stencilVal[axis][offset] = particlesNodes.stencilWeights[
                ibmStencilWeightIndex(axis, offset, i, particlesNodes.numNodes)];
        }
    }

    for (int zk = 0; zk < stencilWidth; ++zk) {
        const int zg = posBase[2] + zk;
        int zz;
        #ifdef BC_Z_WALL
            if (zg < 0 || zg >= NZ_TOTAL) continue;
            zz = zg;
        #else
            zz = ((zg % NZ_TOTAL) + NZ_TOTAL) % NZ_TOTAL;
        #endif

        for (int yj = 0; yj < stencilWidth; ++yj) {
            const int yg = posBase[1] + yj;
            int yy;
            #ifdef BC_Y_WALL
                if (yg < 0 || yg >= NY) continue;
                yy = yg;
            #else
                yy = ((yg % NY) + NY) % NY;
            #endif

            const dfloat yzWeight = stencilVal[2][zk] * stencilVal[1][yj];
            for (int xi = 0; xi < stencilWidth; ++xi) {
                const int xg = posBase[0] + xi;
                int xx;
                #ifdef BC_X_WALL
                    if (xg < 0 || xg >= NX) continue;
                    xx = xg;
                #else
                    xx = ((xg % NX) + NX) % NX;
                #endif

                const dfloat weight = yzWeight * stencilVal[0][xi];
                #ifdef EXTERNAL_DUCT_BC
                    const dfloat radialSquared =
                        (xx - DUCT_CENTER_X) * (xx - DUCT_CENTER_X) +
                        (yy - DUCT_CENTER_Y) * (yy - DUCT_CENTER_Y);
                    if (radialSquared >= OUTER_RADIUS * OUTER_RADIUS) continue;
                #endif

                const size_t blockMomentBase = idxMom(
                    xx % BLOCK_NX, yy % BLOCK_NY, zz % BLOCK_NZ,
                    M_RHO_INDEX, xx / BLOCK_NX, yy / BLOCK_NY, zz / BLOCK_NZ);
                const size_t fxIndex = blockMomentBase + M_FX_INDEX * BLOCK_LBM_SIZE;
                const size_t fyIndex = blockMomentBase + M_FY_INDEX * BLOCK_LBM_SIZE;
                const size_t fzIndex = blockMomentBase + M_FZ_INDEX * BLOCK_LBM_SIZE;

                // deltaForce is the force on the particle. Apply the equal
                // and opposite correction to the Eulerian fluid.
                atomicAdd(&fMom[fxIndex], -correction.x * weight);
                atomicAdd(&fMom[fyIndex], -correction.y * weight);
                atomicAdd(&fMom[fzIndex], -correction.z * weight);
            }
        }
    }
}

#endif //PARTICLE_MODEL
