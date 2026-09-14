/**
 *  @file collisionVar.h
 *  Contributors history:
 *  @author Marco Aurelio Ferrari (e.marcoferrari@utfpr.edu.br)
 *  @brief collision variables
 *  @version 0.4.0
 *  @date 01/09/2025
 */

#ifdef PARTICLE_MODEL
/* -------------------------- COLLISION PARAMETERS -------------------------- */

//collision schemes
#define SOFT_SPHERE 
#define FIRST_PARTICLE_COLLISION_SLOT 7
#ifndef MAX_ACTIVE_PARTICLE_COLLISIONS
#define MAX_ACTIVE_PARTICLE_COLLISIONS 16
#endif
#define MAX_ACTIVE_COLLISIONS (FIRST_PARTICLE_COLLISION_SLOT + MAX_ACTIVE_PARTICLE_COLLISIONS)

// One thread per unique particle pair. Wall checks use the ordinary
// one-thread-per-particle launch and are intentionally kept separate.
constexpr unsigned long long TOTAL_PCOLLISION_THREADS =
    (static_cast<unsigned long long>(NUM_PARTICLES) * (NUM_PARTICLES - 1ULL)) / 2ULL;
constexpr unsigned int TOTAL_PCOLLISION =
    TOTAL_PCOLLISION_THREADS > 64ULL ? 64U : static_cast<unsigned int>(TOTAL_PCOLLISION_THREADS);
constexpr unsigned int GRID_PCOLLISION = TOTAL_PCOLLISION_THREADS == 0ULL ? 0U :
    static_cast<unsigned int>((TOTAL_PCOLLISION_THREADS + TOTAL_PCOLLISION - 1ULL) /
                              TOTAL_PCOLLISION);



/* -------------------------- COLLISION PARAMETERS -------------------------- */
constexpr dfloat WALL_SHEAR_MODULUS = WALL_YOUNG_MODULUS / (2.0+2.0*WALL_POISSON_RATIO);
constexpr dfloat PARTICLE_SHEAR_MODULUS = PARTICLE_YOUNG_MODULUS / (2.0+2.0*PARTICLE_POISSON_RATIO);

//Hertzian contact theory -  Johnson 1985
constexpr dfloat SPHERE_SPHERE_STIFFNESS_NORMAL_CONST = (4.0/3.0) / ((1-PARTICLE_POISSON_RATIO*PARTICLE_POISSON_RATIO)/PARTICLE_YOUNG_MODULUS + (1-PARTICLE_POISSON_RATIO*PARTICLE_POISSON_RATIO)/PARTICLE_YOUNG_MODULUS);
constexpr dfloat SPHERE_WALL_STIFFNESS_NORMAL_CONST   = (4.0/3.0) / ((1-PARTICLE_POISSON_RATIO*PARTICLE_POISSON_RATIO)/PARTICLE_YOUNG_MODULUS + (1-WALL_POISSON_RATIO*WALL_POISSON_RATIO)/WALL_YOUNG_MODULUS);
//Mindlin theory 1949
constexpr dfloat SPHERE_SPHERE_STIFFNESS_TANGENTIAL_CONST =  4.0 * SQRT_2 / ((2-PARTICLE_POISSON_RATIO)/PARTICLE_SHEAR_MODULUS + (2-PARTICLE_POISSON_RATIO)/PARTICLE_SHEAR_MODULUS);
constexpr dfloat SPHERE_WALL_STIFFNESS_TANGENTIAL_CONST =  4.0 * SQRT_2 / ((2-PARTICLE_POISSON_RATIO)/PARTICLE_SHEAR_MODULUS + (2-WALL_POISSON_RATIO)/WALL_SHEAR_MODULUS);


#endif //PARTICLE_MODEL
