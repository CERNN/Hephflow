/**
 * @file pibmVar.h
 * @brief Compile-time controls for point-particle immersed-boundary coupling.
 */

#ifndef __PIBM_VAR_H
#define __PIBM_VAR_H

#include "../../../var.h"

#ifdef PARTICLE_MODEL

#define PIBM_COUPLING_TRILINEAR 0
#define PIBM_COUPLING_STENCIL 1

#ifndef PIBM_COUPLING_MODE
#define PIBM_COUPLING_MODE PIBM_COUPLING_STENCIL
#endif

#ifndef PIBM_FORCE_RELAXATION
#define PIBM_FORCE_RELAXATION 1.0_df
#endif

#ifndef PIBM_MAX_ITERATION
#define PIBM_MAX_ITERATION 1
#endif

static_assert(PIBM_COUPLING_MODE == PIBM_COUPLING_TRILINEAR || PIBM_COUPLING_MODE == PIBM_COUPLING_STENCIL,
    "PIBM_COUPLING_MODE must be PIBM_COUPLING_TRILINEAR or PIBM_COUPLING_STENCIL");
static_assert(PIBM_FORCE_RELAXATION > 0.0_df && PIBM_FORCE_RELAXATION <= 1.0_df,
    "PIBM_FORCE_RELAXATION must be in (0, 1]");
static_assert(PIBM_MAX_ITERATION > 0,
    "PIBM_MAX_ITERATION must be positive");

#endif // PARTICLE_MODEL
#endif // __PIBM_VAR_H
