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
#define MAX_ACTIVE_COLLISIONS 18

// For IBM particles collision, the total of threads must be 
// totalThreads = NUM_PARTICLES*(NUM_PARTICLES+1)/2
constexpr unsigned int TOTAL_PCOLLISION_THREADS = (NUM_PARTICLES*(NUM_PARTICLES+1))/2;
// Threads for IBM particles collision 
constexpr unsigned int TOTAL_PCOLLISION = (TOTAL_PCOLLISION_THREADS > 64) ? 
    64 : TOTAL_PCOLLISION_THREADS;
// Grid for IBM particles collision
constexpr unsigned int GRID_PCOLLISION = 
    (TOTAL_PCOLLISION_THREADS % TOTAL_PCOLLISION ? 
        (TOTAL_PCOLLISION_THREADS / TOTAL_PCOLLISION + 1)
        : (TOTAL_PCOLLISION_THREADS / TOTAL_PCOLLISION));



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