/**
 *  @file particleField.cu
 *  Contributors history:
 *  @author Marco Aurelio Ferrari (e.marcoferrari@utfpr.edu.br)
 *  @brief Particle field management class implementation
 *  @version 0.1.0
 *  @date 31/12/2025
 */

#include "particleField.cuh"

#ifdef PARTICLE_MODEL

ParticleField::ParticleField() 
    : particles(nullptr)
    , d_pwForces(nullptr)
    , memoryAllocated(false)
    , streamsCreated(false) {
    // Constructor - initialize pointers to null
}

ParticleField::~ParticleField() {
    // Destructor - ensure cleanup
    if (memoryAllocated) {
        freeMemory();
    }
    if (streamsCreated) {
        destroyStreams();
    }
}

void ParticleField::allocateMemory() {
    if (memoryAllocated) {
        printf("Warning: ParticleField memory already allocated\n");
        return;
    }

    // Allocate host memory for particles
    particles = (Particle*)malloc(sizeof(Particle) * NUM_PARTICLES);
    if (particles == nullptr) {
        printf("Error: Failed to allocate memory for particles\n");
        exit(EXIT_FAILURE);
    }

    // Allocate device memory for particle-wall forces
    cudaError_t err = cudaMalloc((void**)&d_pwForces, sizeof(ParticleWallForces));
    if (err != cudaSuccess) {
        printf("Error: Failed to allocate device memory for wall forces: %s\n", 
               cudaGetErrorString(err));
        free(particles);
        exit(EXIT_FAILURE);
    }

    memoryAllocated = true;
    printf("ParticleField memory allocated successfully\n");
    if (console_flush) fflush(stdout);
}

void ParticleField::setupStreams() {
    if (streamsCreated) {
        printf("Warning: ParticleField streams already created\n");
        return;
    }

    checkCudaErrors(cudaStreamCreate(&streamsPart[0]));
    streamsCreated = true;
    
    printf("ParticleField streams created successfully\n");
    if (console_flush) fflush(stdout);
}

void ParticleField::initialize(int* step, dim3 gridBlock, dim3 threadBlock) {
    if (!memoryAllocated) {
        printf("Error: ParticleField memory must be allocated before initialization\n");
        exit(EXIT_FAILURE);
    }

    printf("Creating particles...\t");
    if (console_flush) fflush(stdout);
    
    particlesSoA.createParticles(particles);
    
    printf("Particles created!\n");
    if (console_flush) fflush(stdout);

    particlesSoA.updateParticlesAsSoA(particles);
    
    printf("Update ParticlesAsSoA!\n");
    if (console_flush) fflush(stdout);

    int checkpoint_state = 0;
    
    // Check if checkpoint exists and should be loaded
    if (LOAD_CHECKPOINT) {
        checkpoint_state = loadSimCheckpointParticle(particlesSoA, step);
    } else {
        if (checkpoint_state != 0) {
            *step = INI_STEP;
            dim3 gridInit = gridBlock;
            // Initialize ghost nodes
            gridInit.z += 1;
            
            checkCudaErrors(cudaSetDevice(GPU_INDEX));
            checkCudaErrors(cudaDeviceSynchronize());

            getLastCudaError("Particle initialization error");
        }
    }

    printf("ParticleField initialized successfully\n");
    if (console_flush) fflush(stdout);
}

void ParticleField::simulationStep(dfloat* d_fMom, unsigned int step) {
    if (!memoryAllocated || !streamsCreated) {
        printf("Error: ParticleField not properly initialized for simulation\n");
        return;
    }

    particleSimulation(&particlesSoA, d_fMom, streamsPart, d_pwForces, step);
}

void ParticleField::saveInfo(unsigned int step, std::atomic<bool>& savingFlag) {
    // Wait for any previous saving operation to complete
    while (savingFlag) std::this_thread::yield();
    
    saveParticlesInfo(&particlesSoA, step, savingFlag);
}

void ParticleField::saveCheckpoint(unsigned int step) {
    printf("Starting saveSimCheckpointParticle...\t");
    if (console_flush) fflush(stdout);
    
    int step_int = static_cast<int>(step);
    saveSimCheckpointParticle(particlesSoA, &step_int);
    
    printf("done\n");
    if (console_flush) fflush(stdout);
}

void ParticleField::exportWallForces(unsigned int step) {
    collectAndExportWallForces(d_pwForces, step);
}

void ParticleField::waitForSaving(std::atomic<bool>& savingFlag) {
    while (savingFlag) std::this_thread::yield();
}

void ParticleField::freeMemory() {
    if (!memoryAllocated) {
        return;
    }

    // Free particle structures
    if (particles != nullptr) {
        free(particles);
        particles = nullptr;
    }

    // Free device memory
    if (d_pwForces != nullptr) {
        cudaFree(d_pwForces);
        d_pwForces = nullptr;
    }

    // Free SoA structures
    particlesSoA.freeNodesAndCenters();

    memoryAllocated = false;
    
    printf("ParticleField memory freed\n");
    if (console_flush) fflush(stdout);
}

void ParticleField::destroyStreams() {
    if (!streamsCreated) {
        return;
    }

    cudaStreamDestroy(streamsPart[0]);
    streamsCreated = false;
    
    printf("ParticleField streams destroyed\n");
    if (console_flush) fflush(stdout);
}

ParticlesSoA* ParticleField::getParticlesSoA() {
    return &particlesSoA;
}

ParticleWallForces* ParticleField::getWallForcesDevice() {
    return d_pwForces;
}

cudaStream_t* ParticleField::getStream() {
    return streamsPart;
}

#endif // PARTICLE_MODEL
