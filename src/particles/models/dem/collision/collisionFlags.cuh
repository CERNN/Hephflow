#ifndef __PARTICLE_COLLISION_FLAGS_CUH
#define __PARTICLE_COLLISION_FLAGS_CUH



enum ParticleCollisionFlag : unsigned char {
    PARTICLE_COLLISION_DISABLED = 0,
    PARTICLE_COLLISION_WITH_PARTICLES = 1u << 0,
    PARTICLE_COLLISION_WITH_WALLS = 1u << 1
};

// Particle-particle collision is mutual opt-in: disabling it on either member
// suppresses the pair. Wall collision is an independent per-particle choice.
__host__ __device__ constexpr bool particlePairCollisionEnabled(unsigned char firstFlags,unsigned char secondFlags){
    return (firstFlags & PARTICLE_COLLISION_WITH_PARTICLES) != 0 && (secondFlags & PARTICLE_COLLISION_WITH_PARTICLES) != 0;
}

 __host__ __device__ constexpr bool particleWallCollisionEnabled(unsigned char flags){
    return (flags & PARTICLE_COLLISION_WITH_WALLS) != 0;
}



#endif
