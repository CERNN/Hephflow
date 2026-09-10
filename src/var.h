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

#define CELLDATA_SAVE false
#define GPU_INDEX 0

constexpr bool console_flush = false;
constexpr unsigned int N_GPUS = 2;                      // Number of GPUS to use
constexpr unsigned int GPUS_TO_USE[N_GPUS] = {0,1};       // Which GPUs to use

/* ============================ PROBLEM SETUP ============================= */

#define BC_PROBLEM 007_RayleighTaylor_3D

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


constexpr auto err = constexprPow(2.0_df, 0.5_df) - sqrtt(2.0_df);
constexpr dfloat tol = 100 * std::numeric_limits<dfloat>::epsilon();
static_assert(err < tol && err > -tol);

/* ----------------------------------------------------------------------------
 * 1. Square Root Equivalency (Exponent = 0.5)
 * ------------------------------------------------------------------------- */
static_assert(isClose(constexprPow(2.0_df, 0.5_df), sqrtt(2.0_df)),
              "Failed: constexprPow(2.0, 0.5) mismatch with sqrtt(2.0)");

static_assert(isClose(constexprPow(0.0091552734375_df, 0.5_df), sqrtt(0.0091552734375_df)),
              "Failed: Small base square root mismatch");

/* ----------------------------------------------------------------------------
 * 2. Integer Power Fast-Paths & Boundary Exponents
 * ------------------------------------------------------------------------- */
static_assert(constexprPow(5.5_df, 0.0_df) == 1.0_df,
              "Failed: Zero exponent identity (x^0 = 1)");

static_assert(constexprPow(1.0_df, 7.8_df) == 1.0_df,
              "Failed: Unit base identity (1^x = 1)");

static_assert(isClose(constexprPow(3.0_df, 3.0_df), 27.0_df),
              "Failed: Integer cube (3^3 = 27)");

static_assert(isClose(constexprPow(2.0_df, -2.0_df), 0.25_df),
              "Failed: Negative integer power (2^-2 = 0.25)");

/* ----------------------------------------------------------------------------
 * 3. Exact Roots Across Non-Newtonian Range (n_index in [0.1, 10.0])
 * ------------------------------------------------------------------------- */
// Extreme shear-thinning index (n = 0.1): 1024^0.1 = 2
static_assert(isClose(constexprPow(1024.0_df, 0.1_df), 2.0_df),
              "Failed: Flow index n = 0.1 (1024^0.1 = 2)");

// Moderate shear-thinning index (n = 0.25): 16^0.25 = 2
static_assert(isClose(constexprPow(16.0_df, 0.25_df), 2.0_df),
              "Failed: Flow index n = 0.25 (16^0.25 = 2)");

// Dilatant / Shear-thickening index (n = 3.0): 8^(1/3) = 2
static_assert(isClose(constexprPow(8.0_df, 1.0_df / 3.0_df), 2.0_df),
              "Failed: Flow index n = 3.0 (8^(1/3) = 2)");

// High flow behavior index (n = 10.0): 2^10 = 1024
static_assert(isClose(constexprPow(2.0_df, 10.0_df), 1024.0_df),
              "Failed: Flow index n = 10.0 (2^10 = 1024)");

/* ----------------------------------------------------------------------------
 * 4. Algebraic Identity Verification
 * ------------------------------------------------------------------------- */
// Product rule: x^a * x^b = x^(a + b)
constexpr dfloat base_val = 0.05_df;
constexpr dfloat exp_a = 0.3_df;
constexpr dfloat exp_b = 0.45_df;

static_assert(isClose(constexprPow(base_val, exp_a) * constexprPow(base_val, exp_b),
                      constexprPow(base_val, exp_a + exp_b)),
              "Failed: Product rule x^a * x^b = x^(a+b)");

// Inverse relation: x^a * x^(-a) = 1.0
static_assert(isClose(constexprPow(base_val, 1.7_df) * constexprPow(base_val, -1.7_df), 1.0_df),
              "Failed: Multiplicative inverse x^a * x^-a = 1");

// Power of a power: (x^a)^b = x^(a * b)
static_assert(isClose(constexprPow(constexprPow(3.0_df, 0.4_df), 2.5_df), 3.0_df),
              "Failed: Power rule (x^a)^b = x^(a*b)");


#endif //__VAR_H