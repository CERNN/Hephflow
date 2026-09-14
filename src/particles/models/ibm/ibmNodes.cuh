/**
 *  @file main.cuh
 *  Contributors history:
 *  @author Waine Jr. (waine@alunos.utfpr.edu.br)
 *  @author Marco Aurelio Ferrari (e.marcoferrari@utfpr.edu.br)
 *  @author Ricardo de Souza
 *  @brief Struct for IBM particle node
 *  @version 0.4.0
 *  @date 01/09/2025
 */

#ifndef __IBM_NODES_H
#define __IBM_NODES_H

#include "../../../globalStructs.h"
#include "../../models/ibm/ibmVar.h"
// #include "../../class/Particle.cuh"
#include "../../class/ParticleCenter.cuh"
#include <type_traits>
#pragma once

#ifdef PARTICLE_MODEL
class Particle;
class ParticleCenter;

constexpr int IBM_CACHED_STENCIL_WIDTH = 4;

struct IbmVectorView
{
    dfloat* x;
    dfloat* y;
    dfloat* z;
};

/**
 * Lightweight kernel-facing view of the IBM marker arrays.  The view owns no
 * memory and is safe to pass to CUDA kernels by value.
 */
struct IbmNodesView
{
    unsigned int numNodes;
    unsigned int* particleCenterIdx;
    IbmVectorView pos;
    IbmVectorView f;
    IbmVectorView deltaF;
    IbmVectorView originalRelativePos;
    dfloat* S;
    int3* stencilBase;
    dfloat* stencilWeights;
};

static_assert(std::is_trivially_copyable<IbmNodesView>::value,
    "IbmNodesView must remain safe to pass to CUDA kernels by value");

/*
*   Class describe the IBM node properties
*/
class IbmNodes
{

public:
    __host__ __device__ IbmNodes();

    __host__ __device__ dfloat3 getPos() const;
    __host__ __device__ dfloat getPosX() const;
    __host__ __device__ dfloat getPosY() const;
    __host__ __device__ dfloat getPosZ() const;
    __host__ __device__ void setPos(const dfloat3& pos);
    __host__ __device__ void setPosX(const dfloat& Pos_x);
    __host__ __device__ void setPosY(const dfloat& Pos_y);
    __host__ __device__ void setPosZ(const dfloat& Pos_z);
   
    __host__ __device__ dfloat3 getVel() const;
    __host__ __device__ void setVel(const dfloat3& vel);
    
    __host__ __device__ dfloat3 getVelOld() const;
    __host__ __device__ void setVelOld(const dfloat3& vel_old);
   
    __host__ __device__ dfloat3 getF() const;
    __host__ __device__ dfloat getFX() const;
    __host__ __device__ dfloat getFY() const;
    __host__ __device__ dfloat getFZ() const;
    __host__ __device__ void setF(const dfloat3& f);
    __host__ __device__ void setFX(const dfloat& f_x);
    __host__ __device__ void setFY(const dfloat& f_y);
    __host__ __device__ void setFZ(const dfloat& f_z);
    
    __host__ __device__ dfloat3 getDeltaF() const;
    __host__ __device__ void setDeltaF(const dfloat3& deltaF);
    
    __host__ __device__ float getS() const;
    __host__ __device__ void setS(const dfloat& S);  

protected:
    dfloat3 pos; // node coordinate
    dfloat3 vel; // node velocity
    dfloat3 vel_old; // node last step velocity
    dfloat3 f;  // node force
    dfloat3 deltaF;  // node force variation
    dfloat S; // node surface area
};

/*
*   Class to represent the particle nodes as a Structure of Arrays, 
*   instead of a Array of Structures
*/
class IbmNodesSoA
{
protected:
    unsigned int numNodes; // number of nodes
    unsigned int* particleCenterIdx; // index of particle center for each node
    dfloat3SoA pos; // vectors with nodes coordinates
    dfloat3SoA vel; // vectors with nodes velocities
    dfloat3SoA vel_old; // vectors with nodes old velocities
    dfloat3SoA f;  // vectors with nodes forces
    dfloat3SoA deltaF;  // vectors with nodes forces variations
    dfloat3SoA originalRelativePos; // vectors with original relative positions (node_pos - particle_center_pos) - immutable reference for precision
    dfloat* S; // vector node surface area
    int3* stencilBase; // first lattice coordinate in each marker stencil
    dfloat* stencilWeights; // axis-major cached 1D weights, [axis][offset][node]

public:
    __host__ __device__
    IbmNodesSoA();
    __host__ __device__
    ~IbmNodesSoA();

    /**
     *  @brief Allocate memory for given maximum number of nodes 
     *  @param numMaxNodes: maximum number of nodes
     */
   __host__ void allocateMemory(unsigned int numMaxNodes);

    /**
     *  @brief Free allocated memory
     */
   __host__ void freeMemory();

    /**
     *  @brief Copy nodes values from particle
     *  @param p: particle with nodes to copy
     *  @param pCenterIdx: index of particle center for given particle nodes
     *  @param baseIdx: base index to use while copying
     */
    __host__ void copyNodesFromParticle(Particle *particle, unsigned int pCenterIdx, ParticleCenter* pArray, unsigned int n_gpu);
 
    __host__ void leftShiftNodesSoA(int idx, int left_shit);

    __host__ IbmNodesView getView() const;

    __host__ __device__  unsigned int getNumNodes() const;
    __host__ __device__ void setNumNodes(const int numNodes); 
    
    __host__ __device__ const unsigned int* getParticleCenterIdx() const;
    __host__ __device__ unsigned int* getParticleCenterIdx();
    __host__ __device__ void setParticleCenterIdx(unsigned int* particleCenterIdx);
    
    __host__ __device__ dfloat3SoA getPos() const;
    __host__ __device__ void setPos(const dfloat3SoA& pos);
    
    __host__ __device__ dfloat3SoA getVel() const;
    __host__ __device__ void setVel(const dfloat3SoA& vel);
    
    __host__ __device__ dfloat3SoA getVelOld() const;
    __host__ __device__ void setVelOld(const dfloat3SoA& vel_old);
   
    __host__ __device__ dfloat3SoA getF() const;
    __host__ __device__ dfloat getFX() const;
    __host__ __device__ dfloat getFY() const;
    __host__ __device__ dfloat getFZ() const;
    __host__ __device__ void setF(const dfloat3SoA& f);
    __host__ __device__ void setFX(const dfloat& fx);
    __host__ __device__ void setFY(const dfloat& fy);
    __host__ __device__ void setFZ(const dfloat& fz);
    
    __host__ __device__ dfloat3SoA getDeltaF() const;
    __host__ __device__ void setDeltaF(const dfloat3SoA& deltaF);
    
    __host__ __device__ dfloat3SoA getOriginalRelativePos() const;
    __host__ __device__ void setOriginalRelativePos(const dfloat3SoA& originalRelativePos);
   
    __host__ __device__ dfloat* getS() const;
    __host__ __device__ void setS(dfloat* S);
};

#endif //PARTICLE_MODEL
#endif //!__IBM_NODES_H


