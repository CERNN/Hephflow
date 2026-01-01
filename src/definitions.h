/**
 *  @file definitions.h
 *  Contributors history:
 *  @author Waine Jr. (waine@alunos.utfpr.edu.br)
 *  @author Marco Aurelio Ferrari (e.marcoferrari@utfpr.edu.br)
 *  @brief Global definitions
 *  @version 0.4.0
 *  @date 01/09/2025
 */

#ifndef __DEFINITIONS_H
#define __DEFINITIONS_H

#include "var.h"

/* ============================= MODULAR INCLUDES ============================= */

#include "include/math_constants.h"    // Mathematical constants
#include "include/feature_config.h"     // Feature flags and model macros
#include "include/memory_layout.h"      // Memory calculations and block layout
#include "include/cuda_config.h"        // CUDA-specific settings

/* ========================== AUXILIARY DEFINITIONS ======================== */

#define IN_HOST 1       // variable accessible only for host
#define IN_VIRTUAL 2    // variable accessible for device and host

/* ============================= VELOCITY SETS ============================= */
// Velocity sets are now included in memory_layout.h

#endif //!__DEFINITIONS_H