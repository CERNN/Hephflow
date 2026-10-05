/**
 *  @file auxFunctions.h
 *  Contributors history:
 *  @author Marco Aurelio Ferrari (e.marcoferrari@utfpr.edu.br)
 *  @brief Auxiliary Functions
 *  @version 0.1.0
 *  @date 01/09/2025
 */


#ifndef __AUX_FUNCTION_CUH
#define __AUX_FUNCTION_CUH

#include <string>
#include <math.h>
#include <cuda.h>
#include <curand.h>
#include <cuda_runtime.h>
#include <builtin_types.h>
#include "globalFunctions.h"
#include "include/errorDef.h"
#include "var.h"
#include "nodeTypeMap.h"
#include "reduction.cuh"
#include "saveData.cuh"

#include <fstream>
#include <sstream>
#include <iostream>     // std::cout, std::fixed
#include <iomanip>      // std::setprecision


/** @brief Queue a local density sum/count reduction and host copy. */
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
    cudaStream_t stream = 0);

/** 
 *  @brief Compute and update the mean flow values (velocity and temperature if applicable) over time.
 *  @param h_fMom: Pointer to the host array containing the current macroscopic moments.
 *  @param fMom: Pointer to the device array containing the current macroscopic moments.
*/
__host__
void meanFlowComputation(
    dfloat* h_fMom,
    dfloat* fMom,
    dfloat* fMom_mean,
    unsigned int step
);


#endif // !__AUX_FUNCTION_CUH
