/**
 *  @file particleField.cuh
 *  Contributors history:
 *  @author Marco Aurelio Ferrari (e.marcoferrari@utfpr.edu.br)
 *  @brief Particle field management class for LBM simulations
 *  @version 0.1.0
 *  @date 31/12/2025
 */

#ifndef __PARTICLE_FIELD_CUH
#define __PARTICLE_FIELD_CUH

#include <atomic>
#include "var.h"

#ifdef PARTICLE_MODEL

#include "particles/class/Particle.cuh"
#include "particles/utils/particlesReport.cuh"
#include "particles/models/particleSim.cuh"
#include "particles/models/dem/collision/collisionDetection.cuh"
#include "particles/models/dem/particleMovement.cuh"
#include "checkpoint.cuh"

/**
 * @brief Manages all particle-related data structures and operations for LBM-DEM/IBM simulations
 * 
 * This class encapsulates particle initialization, memory management, simulation stepping,
 * checkpointing, and data saving operations. It provides a clean interface to integrate
 * particle simulations with the main LBM solver.
 */
class ParticleField {
public:
    /**
     * @brief Default constructor
     */
    ParticleField();

    /**
     * @brief Destructor - cleans up all allocated memory
     */
    ~ParticleField();

    /**
     * @brief Allocate all particle-related memory (host and device)
     */
    void allocateMemory();

    /**
     * @brief Initialize particles with positions, velocities, and solver method
     * @param step Pointer to the current simulation step
     * @param gridBlock CUDA grid dimensions for kernel launches
     * @param threadBlock CUDA thread block dimensions for kernel launches
     */
    void initialize(int* step, dim3 gridBlock, dim3 threadBlock);

    /**
     * @brief Setup CUDA streams for asynchronous particle operations
     */
    void setupStreams();

    /**
     * @brief Execute one particle simulation step
     * @param d_fMom Pointer to the device macroscopic moments array
     * @param step Current simulation time step
     */
    void simulationStep(dfloat* d_fMom, unsigned int step);

    /**
     * @brief Save particle information to file
     * @param step Current simulation step
     * @param savingFlag Reference to atomic flag for thread-safe saving
     */
    void saveInfo(unsigned int step, std::atomic<bool>& savingFlag);

    /**
     * @brief Save checkpoint for particle state
     * @param step Current simulation step
     */
    void saveCheckpoint(unsigned int step);

    /**
     * @brief Collect and export wall forces to file
     * @param step Current simulation step
     */
    void exportWallForces(unsigned int step);

    /**
     * @brief Wait for all particle saving operations to complete
     * @param savingFlag Reference to atomic flag for thread-safe saving
     */
    void waitForSaving(std::atomic<bool>& savingFlag);

    /**
     * @brief Free all particle-related memory
     */
    void freeMemory();

    /**
     * @brief Destroy CUDA streams
     */
    void destroyStreams();

    /**
     * @brief Get pointer to ParticlesSoA structure
     * @return Pointer to the particles SoA data
     */
    ParticlesSoA* getParticlesSoA();

    /**
     * @brief Get pointer to wall forces on device
     * @return Pointer to device wall forces structure
     */
    ParticleWallForces* getWallForcesDevice();

    /**
     * @brief Get pointer to particle stream
     * @return Pointer to the CUDA stream for particles
     */
    cudaStream_t* getStream();

private:
    // Particle data structures
    ParticlesSoA particlesSoA;          ///< Structure of Arrays for particle data
    Particle* particles;                 ///< Array of individual particles (host)
    
    // Device memory for particle-wall forces
    ParticleWallForces* d_pwForces;     ///< Device pointer to wall forces
    
    // CUDA stream for asynchronous operations
    cudaStream_t streamsPart[1];        ///< CUDA streams for particle operations
    
    // Memory allocation flags
    bool memoryAllocated;               ///< Track if memory has been allocated
    bool streamsCreated;                ///< Track if streams have been created
};

#endif // PARTICLE_MODEL
#endif // __PARTICLE_FIELD_CUH
