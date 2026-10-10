
#include "auxFunctions.cuh"

__host__
void reduceDensityPartial(
    const dfloat* fMom,
    const unsigned int* nodeType,
    dfloat* sumBuffer0,
    dfloat* sumBuffer1,
    unsigned long long* countBuffer0,
    unsigned long long* countBuffer1,
    size_t localNZ,
    dfloat* hostSum,
    unsigned long long* hostCount,
    cudaStream_t stream
){
    densityReductionBlocks<<<dim3(NUM_BLOCK_X, NUM_BLOCK_Y,
        static_cast<unsigned int>(ceilDivSize(localNZ, BLOCK_NZ))),
        dim3(BLOCK_NX, BLOCK_NY, BLOCK_NZ), 0, stream>>>
        ( fMom, nodeType, sumBuffer0, countBuffer0, localNZ);

    const size_t numBlocksZ = ceilDivSize(localNZ, BLOCK_NZ);
    size_t elementCount = NUM_BLOCK_X * NUM_BLOCK_Y * numBlocksZ;
    dfloat* inputSums = sumBuffer0;
    dfloat* outputSums = sumBuffer1;
    unsigned long long* inputCounts = countBuffer0;
    unsigned long long* outputCounts = countBuffer1;
    constexpr size_t REDUCTION_THREADS = 256;

    while (elementCount > 1) {
        const size_t outputCount = (elementCount + REDUCTION_THREADS - 1) / REDUCTION_THREADS;
        densityReductionPass<<<outputCount, REDUCTION_THREADS, 0, stream>>>(inputSums, inputCounts, outputSums, outputCounts, elementCount);
        elementCount = outputCount;
        dfloat* oldInputSums = inputSums;
        inputSums = outputSums;
        outputSums = oldInputSums;
        unsigned long long* oldInputCounts = inputCounts;
        inputCounts = outputCounts;
        outputCounts = oldInputCounts;
    }

    checkCudaErrors(cudaMemcpyAsync( hostSum, inputSums, sizeof(dfloat), cudaMemcpyDeviceToHost, stream));
    checkCudaErrors(cudaMemcpyAsync( hostCount, inputCounts, sizeof(unsigned long long), cudaMemcpyDeviceToHost, stream));
}


__host__
void meanFlowComputation(
    dfloat* h_fMom,
    dfloat* fMom,
    dfloat* fMom_mean,
    unsigned int step
){

    //current values
    dfloat t_ux0 = 0.0_df;
    dfloat t_uy0 = 0.0_df;
    dfloat t_uz0 = 0.0_df;

    #ifdef THERMAL_MODEL
    dfloat t_cc0 = 0.0_df;
    #endif //THERMAL_MODEL

    //old mean values
    dfloat m_ux = 0.0_df;
    dfloat m_uy = 0.0_df;
    dfloat m_uz = 0.0_df;

    #ifdef THERMAL_MODEL
    dfloat m_cc = 0.0_df;
    #endif //THERMAL_MODEL


    #pragma warning(push)
    #pragma warning(disable: 4804)
    dfloat mean_counter = 1.0_df/((dfloat)(step/MACR_SAVE)+1.0_df);
    #pragma warning(pop)
    
    int count = 0;

    for (int z = 0 ; z <NZ_TOTAL; z++){
        for(int y = 0; y< NY;y++){
            for(int x = 0; x< NX;x++){


                t_ux0 = h_fMom[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M_UX_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)];
                t_uy0 = h_fMom[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M_UY_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)];
                t_uz0 = h_fMom[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M_UZ_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)];

                #ifdef THERMAL_MODEL
                t_cc0 = h_fMom[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M2_C_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)];
                #endif //THERMAL_MODEL

                //STORE AND UPDATE MEANS

                //retrive mean values
                m_ux = fMom_mean[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M_UX_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)];
                m_uy = fMom_mean[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M_UY_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)];
                m_uz = fMom_mean[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M_UZ_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)];

                #ifdef THERMAL_MODEL
                m_cc = fMom_mean[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M2_C_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)];
                #endif //THERMAL_MODEL
                
                //update and store mean values
                fMom_mean[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M_UX_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)] = m_ux + (t_ux0 - m_ux)*(mean_counter);
                fMom_mean[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M_UY_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)] = m_uy + (t_uy0 - m_uy)*(mean_counter);
                fMom_mean[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M_UZ_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)] = m_uz + (t_uz0 - m_uz)*(mean_counter);

                #ifdef THERMAL_MODEL
                fMom_mean[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M2_C_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)] = m_cc + (t_cc0 - m_cc)*(mean_counter);
                #endif //THERMAL_MODEL

                count++;

            }
        }
    }
}
