/**
 *  @file var.h
 *  Contributors history:
 *  @author Waine Jr. (waine@alunos.utfpr.edu.br)
 *  @author Marco Aurelio Ferrari (e.marcoferrari@utfpr.edu.br)
 *  @brief Global variables
 *  @version 0.4.0
 *  @date 01/09/2025
 */

#ifndef __VAR_H
#define __VAR_H

/* ================================ INCLUDES ================================ */

#define _USE_MATH_DEFINES
#include <math.h>

// Project type definitions and utilities
#include "include/var_types.h"
#include "non_newtonian/nnf_types.h"
#include "include/utils.h"
#include "include/constexpr_math.h"
#include "include/cuda_utils.cuh"
#include "include/endian_utils.h"

/* ======================== SIMULATION CONFIGURATION ======================= */

#define CELLDATA_SAVE true
#define GPU_INDEX 0

constexpr bool console_flush = false;
constexpr unsigned int N_GPUS = 1;                // Number of GPUS to use
constexpr unsigned int GPUS_TO_USE[N_GPUS] = {0}; // Which GPUs to use

/* ============================ PROBLEM SETUP ============================= */

#define BC_PROBLEM 006_singleParticleSettlingPIBM

/* ======================= CASE CONFIGURATION INCLUDES ===================== */

#include "include/case_definitions.h"

/* ============================= CASE INCLUDES ============================= */

#include CASE_MODEL
#include CASE_CONSTANTS
#include CASE_OUTPUTS

/* ======================== PROJECT HEADER INCLUDES ======================== */

#include "definitions.h"

/* ======================== COMPILE-TIME ASSERTIONS ======================== */

// Ensure domain dimensions are compatible with block sizes
static_assert(NX >= BLOCK_NX, "NX must be >= BLOCK_NX, Update block size in memory_layout.h or increase domain in constants.inc");
static_assert(NY >= BLOCK_NY, "NY must be >= BLOCK_NY, Update block size in memory_layout.h or increase domain in constants.inc");
static_assert(NZ >= BLOCK_NZ, "NZ must be >= BLOCK_NZ, Update block size in memory_layout.h or increase domain in constants.inc");

#endif //__VAR_H