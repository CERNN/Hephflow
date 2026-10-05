/*
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:
 * Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.
 * Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.
 * Neither the name of NVIDIA CORPORATION nor the names of its
   contributors may be used to endorse or promote products derived
   from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

For additional information on the license terms, see the CUDA EULA at
https://docs.nvidia.com/cuda/eula/index.html

*/

/**
 *  @file reduction.cu
 *  @brief Modified version of NVIDIA's parallel reduction sample
 *  Contributors history:
 *  @author Marco Aurelio Ferrari (e.marcoferrari@utfpr.edu.br)
 *  @date 2025-08-31
 *
 * This file is based on NVIDIA sample code (parallel reduction),
 * with modifications allow compute different flow metrics
 */



#include "reduction.cuh"

__global__ 
void sumReductionThread(dfloat* g_idata, dfloat* g_odata, int m_index)
{

    #include "fragments/shared_reduction.inc"


    //global index in the array
    unsigned int i =  idxMom(threadIdx.x, threadIdx.y, threadIdx.z, m_index, blockIdx.x, blockIdx.y, blockIdx.z);
    //thread index in the array
    unsigned int tid = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * (threadIdx.z));
    //block index
    unsigned int bid = blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * (blockIdx.z));
    //total number of threads in block
    unsigned int blockSize = blockDim.x * blockDim.y * blockDim.z;

    sdata[tid] = g_idata[i];
    __syncthreads();
    
    // Reduction that handles non-power-of-2 block sizes
    for (unsigned int s = blockSize; s > 1; ) {
        unsigned int half = s / 2;
        if (tid < half) {
            // If s is odd, thread (half-1) also needs to add the middle element
            sdata[tid] += sdata[tid + half];
            if ((s & 1) && tid == 0) {
                sdata[0] += sdata[s - 1];
            }
        }
        __syncthreads();
        s = half;
    }
    
    if (tid == 0) {
        g_odata[bid] = sdata[0];
    }
}

__global__ 
void sumReductionThread_TKE(dfloat* g_idata, dfloat* g_odata, dfloat *meanMom)
{
    #include "fragments/shared_reduction.inc"


    //global index in the array
    unsigned int ix =  idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_UX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z);
    unsigned int iy =  idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_UY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z);
    unsigned int iz =  idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_UZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z);
    //thread index in the array
    unsigned int tid = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * (threadIdx.z));
    //block index
    unsigned int bid = blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * (blockIdx.z));
    //total number of threads in block
    unsigned int blockSize = blockDim.x * blockDim.y * blockDim.z;

    dfloat fluc_ux = (g_idata[ix] - meanMom[ix])*(g_idata[ix] - meanMom[ix]);
    dfloat fluc_uy = (g_idata[iy] - meanMom[iy])*(g_idata[iy] - meanMom[iy]);
    dfloat fluc_uz = (g_idata[iz] - meanMom[iz])*(g_idata[iz] - meanMom[iz]);

    sdata[tid] = (fluc_ux*fluc_ux + fluc_uy*fluc_uy + fluc_uz*fluc_uz)/2;
    __syncthreads();
    
    // Reduction that handles non-power-of-2 block sizes
    for (unsigned int s = blockSize; s > 1; ) {
        unsigned int half = s / 2;
        if (tid < half) {
            sdata[tid] += sdata[tid + half];
            if ((s & 1) && tid == 0) {
                sdata[0] += sdata[s - 1];
            }
        }
        __syncthreads();
        s = half;
    }
    
    if (tid == 0) {
        g_odata[bid] = sdata[0];
    }
}


__global__
void densityReductionBlocks(
    const dfloat* fMom,
    const unsigned int* nodeType,
    dfloat* blockSums,
    unsigned long long* blockCounts,
    size_t localNZ)
{
    __shared__ dfloat sums[BLOCK_LBM_SIZE];
    __shared__ unsigned long long counts[BLOCK_LBM_SIZE];

    const unsigned int tid = threadIdx.x
        + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
    const unsigned int bid = blockIdx.x
        + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z);
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;

    dfloat rhoFluctuation = 0.0_df;
    unsigned long long active = 0;
    if (x < NX && y < NY && z < localNZ) {
        const size_t scalarIndex = idxScalarBlock(
            threadIdx.x, threadIdx.y, threadIdx.z,
            blockIdx.x, blockIdx.y, blockIdx.z);
        const unsigned int type = nodeType[scalarIndex];
        if ((type & SOLID_NODE) != SOLID_NODE) {
            const size_t momentIndex = idxMom(
                threadIdx.x, threadIdx.y, threadIdx.z, M_RHO_INDEX,
                blockIdx.x, blockIdx.y, blockIdx.z);
            rhoFluctuation = loadMoment(fMom, momentIndex, M_RHO_INDEX);
            active = 1;
        }
    }

    sums[tid] = rhoFluctuation;
    counts[tid] = active;
    __syncthreads();

    for (unsigned int stride = BLOCK_LBM_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sums[tid] += sums[tid + stride];
            counts[tid] += counts[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        blockSums[bid] = sums[0];
        blockCounts[bid] = counts[0];
    }
}

__global__
void densityReductionPass(
    const dfloat* inputSums,
    const unsigned long long* inputCounts,
    dfloat* outputSums,
    unsigned long long* outputCounts,
    size_t elementCount)
{
    constexpr unsigned int REDUCTION_THREADS = 256;
    __shared__ dfloat sums[REDUCTION_THREADS];
    __shared__ unsigned long long counts[REDUCTION_THREADS];

    const unsigned int tid = threadIdx.x;
    const size_t index = blockIdx.x * blockDim.x + tid;
    sums[tid] = index < elementCount ? inputSums[index] : 0.0_df;
    counts[tid] = index < elementCount ? inputCounts[index] : 0;
    __syncthreads();

    for (unsigned int stride = REDUCTION_THREADS / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sums[tid] += sums[tid + stride];
            counts[tid] += counts[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        outputSums[blockIdx.x] = sums[0];
        outputCounts[blockIdx.x] = counts[0];
    }
}

__global__ 
void sumReductionThread_KE(dfloat* g_idata, dfloat* g_odata)
{

    #include "fragments/shared_reduction.inc"

    //global index in the array
    unsigned int ip =  idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_RHO_INDEX, blockIdx.x, blockIdx.y, blockIdx.z);
    unsigned int ix =  idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_UX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z);
    unsigned int iy =  idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_UY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z);
    unsigned int iz =  idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_UZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z);
    //thread index in the array
    unsigned int tid = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * (threadIdx.z));
    //block index
    unsigned int bid = blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * (blockIdx.z));
    //total number of threads in block
    unsigned int blockSize = blockDim.x * blockDim.y * blockDim.z;

    const dfloat rho = RHO_0 + loadMoment(g_idata, ip, M_RHO_INDEX);
    const dfloat ux = loadMoment(g_idata, ix, M_UX_INDEX);
    const dfloat uy = loadMoment(g_idata, iy, M_UY_INDEX);
    const dfloat uz = loadMoment(g_idata, iz, M_UZ_INDEX);
    sdata[tid] = rho * (ux*ux + uy*uy + uz*uz)
        / (2*F_M_I_SCALE*F_M_I_SCALE);
    __syncthreads();
    
    // Reduction that handles non-power-of-2 block sizes
    for (unsigned int s = blockSize; s > 1; ) {
        unsigned int half = s / 2;
        if (tid < half) {
            sdata[tid] += sdata[tid + half];
            if ((s & 1) && tid == 0) {
                sdata[0] += sdata[s - 1];
            }
        }
        __syncthreads();
        s = half;
    }
    
    if (tid == 0) {
        g_odata[bid] = sdata[0];
    }
}

#ifdef CONVECTION_DIFFUSION_TRANSPORT
#ifdef CONFORMATION_TENSOR
__global__ 
void sumReductionThread_SE(dfloat* g_idata, dfloat* g_odata)
{

    #include "fragments/shared_reduction.inc"

    //global index in the array
    unsigned int ixx =  idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_XX_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z);
    unsigned int iyy =  idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_YY_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z);
    unsigned int izz =  idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_ZZ_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z);
    //thread index in the array
    unsigned int tid = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * (threadIdx.z));
    //block index
    unsigned int bid = blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * (blockIdx.z));
    //total number of threads in block
    unsigned int blockSize = blockDim.x * blockDim.y * blockDim.z;
    
    //sdata[tid] = ((g_idata[ixx] - 1.0_df - CONF_ZERO)*(g_idata[ixx] - 1.0_df - CONF_ZERO) + (g_idata[iyy] - 1.0_df - CONF_ZERO)*(g_idata[iyy] - 1.0_df - CONF_ZERO) + (g_idata[izz] - 1.0_df - CONF_ZERO)*(g_idata[izz] - 1.0_df - CONF_ZERO));
    sdata[tid] = (g_idata[ixx] + g_idata[iyy]+ g_idata[izz] - 3.0_df - 3.0_df*CONF_ZERO);
    __syncthreads();
    
    // Reduction that handles non-power-of-2 block sizes
    for (unsigned int s = blockSize; s > 1; ) {
        unsigned int half = s / 2;
        if (tid < half) {
            sdata[tid] += sdata[tid + half];
            if ((s & 1) && tid == 0) {
                sdata[0] += sdata[s - 1];
            }
        }
        __syncthreads();
        s = half;
    }
    
    if (tid == 0) {
        g_odata[bid] = sdata[0];
    }
}
#endif //CONFORMATION_TENSOR
#endif //CONVECTION_DIFFUSION_TRANSPORT

__global__ 
void sumReductionScalar(dfloat* g_idata, dfloat* g_odata)
{

    #include "fragments/shared_reduction.inc"

    //global index in the array
    unsigned int i =  idxScalarBlock(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z);
    //thread index in the array
    unsigned int tid = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * (threadIdx.z));
    //block index
    unsigned int bid = blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * (blockIdx.z));
    //total number of threads in block
    unsigned int blockSize = blockDim.x * blockDim.y * blockDim.z;

    sdata[tid] = g_idata[i];
    __syncthreads();
    
    // Reduction that handles non-power-of-2 block sizes
    for (unsigned int s = blockSize; s > 1; ) {
        unsigned int half = s / 2;
        if (tid < half) {
            sdata[tid] += sdata[tid + half];
            if ((s & 1) && tid == 0) {
                sdata[0] += sdata[s - 1];
            }
        }
        __syncthreads();
        s = half;
    }
    
    if (tid == 0) {
        g_odata[bid] = sdata[0];
    }
}



__global__ 
void sumReductionBlock(dfloat* g_idata, dfloat* g_odata)
{

    #include "fragments/shared_reduction.inc"

    //global index in the array
    unsigned int i = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * (threadIdx.z + blockDim.z * ((blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * (blockIdx.z))))));
    //thread index in the array
    unsigned int tid = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * (threadIdx.z));
    //block index
    unsigned int bid = blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * (blockIdx.z));
    //total number of threads in block
    unsigned int blockSize = blockDim.x * blockDim.y * blockDim.z;

    sdata[tid] = g_idata[i];
    __syncthreads();
    
    // Reduction that handles non-power-of-2 block sizes
    for (unsigned int s = blockSize; s > 1; ) {
        unsigned int half = s / 2;
        if (tid < half) {
            sdata[tid] += sdata[tid + half];
            if ((s & 1) && tid == 0) {
                sdata[0] += sdata[s - 1];
            }
        }
        __syncthreads();
        s = half;
    }
    
    if (tid == 0) {
        g_odata[bid] = sdata[0];
    }
}
