/**
 *  @file ibmVar.h
 *  Contributors history:
 *  @author Waine Jr. (waine@alunos.utfpr.edu.br)
 *  @author Marco Aurelio Ferrari (e.marcoferrari@utfpr.edu.br)
 *  @author Ricardo de Souza
 *  @brief variables for IBM
 *  @version 0.4.0
 *  @date 01/09/2025
 */


#ifndef __IBM_VAR_H
#define __IBM_VAR_H

#include "../../../var.h"
#include <stdio.h>
#include <math.h>

#ifdef PARTICLE_MODEL

// Multi-direct-forcing controls. Cases may override these before including
// the particle model headers.
#ifndef IBM_MAX_ITERATION
#define IBM_MAX_ITERATION 1
#endif

#ifndef IBM_FORCE_RELAXATION
#define IBM_FORCE_RELAXATION 1.0_df
#endif

#ifndef IBM_VELOCITY_TOL
#define IBM_VELOCITY_TOL 1.0e-5_df
#endif
//#define IBM_DEBUG


/* ------------------------ THREADS AND GRIDS FOR IBM ----------------------- */


/* -------------------------------------------------------------------------- */

#endif //PARTICLE_MODEL
#endif // !__IBM_VAR_H


