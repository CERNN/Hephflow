/**
 * @file simulation.cuh
 * @brief Host-side orchestration state and procedural simulation phases.
 */
#ifndef SIMULATION_CUH
#define SIMULATION_CUH

#include <atomic>
#include <chrono>
#include <future>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include <cuda_runtime.h>

#include "var.h"
#include "auxFunctions.cuh"
#include "checkpoint.cuh"
#include "lbmInitialization.cuh"
#include "mlbm.cuh"
#include "deviceField.cuh"
#include "hostField.cuh"
#include "saveField.cuh"
#include "saveData.cuh"
#include "include/memory_estimate.h"

#ifdef PARTICLE_MODEL
    #include "particleField.cuh"
#endif // PARTICLE_MODEL

#if defined(OVERLAP_HALO_TRANSFERS) && !defined(SPLIT_Z_COLLISION_KERNEL)
    #error "OVERLAP_HALO_TRANSFERS requires SPLIT_Z_COLLISION_KERNEL"
#endif // defined(OVERLAP_HALO_TRANSFERS) && !defined(SPLIT_Z_COLLISION_KERNEL)

/**
 * @brief Owns the host-side resources and state shared by simulation phases.
 *
 * The procedural helpers below operate on this context so main.cu can expose
 * the lifecycle without carrying CUDA scheduling and output implementation.
 */
struct SimulationContext {
    SimulationContext();

    // hostField is the single shared host snapshot. deviceField provides
    // host-wide coordination methods; devices owns one partition per GPU.
    HostField hostField{};
    DeviceField deviceField{};
    std::vector<DeviceField> devices;
    DomainDecomposition decomposition;
    std::vector<std::thread> threads;

    // streamsLBM carries each partition's normal dependency chain.
    dim3 threadBlock;
    dim3 gridBlock;
    cudaStream_t streamsLBM[N_GPUS]{};

    #ifdef OVERLAP_HALO_TRANSFERS
        // Boundary work publishes the next halo generation while the current
        // interior runs. Events make the generation handoff explicit.
        cudaStream_t streamsBoundary[N_GPUS]{};
        cudaStream_t streamsHalo[N_GPUS]{};
        cudaEvent_t stepReady[N_GPUS]{};
        cudaEvent_t boundaryReady[N_GPUS]{};
        cudaEvent_t haloReady[N_GPUS]{};
    #endif // OVERLAP_HALO_TRANSFERS

    #ifdef PARTICLE_MODEL
        // Particle coupling currently uses the configured particle GPU and a
        // CUDA event to wait for the fluid force field without a device sync.
        ParticleField particleField;
        cudaEvent_t particleForceFieldReady{};
    #endif // PARTICLE_MODEL

    // Every producer and consumer of hostField.h_fMom shares this lock. The
    // future owns the asynchronous checkpoint/macro/report worker.
    std::mutex hostSnapshotMutex;
    std::future<void> hostSnapshotFuture;
    std::atomic<bool> savingMacrVtk{false};
    std::atomic<bool> savingMacrParticle{false};
    std::vector<std::atomic<bool>> savingMacrBin;

    // stepParity identifies the population generation active at step.
    int step = 0;
    int initialStep = 0;
    bool stepParity = false;
    bool densityHistoryStarted = false;
    std::chrono::high_resolution_clock::time_point simulationStartTime;
};

/** @brief Allocates resources, initializes/restores the domain, and seeds derived state. */
bool initializeSimulation(SimulationContext& context);

/** @brief Waits for the current asynchronous host-snapshot consumer, if any. */
void finishHostSnapshotTask(SimulationContext& context);

/** @brief Synchronizes the primary LBM stream on every partition. */
void synchronizeSimulationStreams(SimulationContext& context);

/** @brief Rebuilds phase derivatives and their required multi-GPU halos. */
void updatePhaseDerivatives(SimulationContext& context);

/** @brief Exchanges population halos and launches the collision/streaming kernels. */
void collideAndExchangeHalos(SimulationContext& context, const SaveField& saveField);

/** @brief Updates phase and density-correction fields produced by a timestep. */
void updateAuxiliaryPhysics(SimulationContext& context);

/** @brief Runs half-step physics, particles, and next-step dependency recording. */
void runHalfStepAndParticles(SimulationContext& context);

/** @brief Captures and dispatches checkpoint, report, and macro outputs. */
void processStepOutputs(SimulationContext& context, const SaveField& saveField);

/** @brief Executes the active case's compile-time test metric. */
void runConfiguredTestMetric(SimulationContext& context);

/** @brief Dispatches particle trajectory output after case test metrics. */
void processParticleOutputs(SimulationContext& context, const SaveField& saveField);

/** @brief Synchronizes the completed run and reports its throughput. */
dfloat finishSimulationTiming(SimulationContext& context);

/** @brief Writes final macros, checkpoints, checksum, and simulation information. */
void writeFinalOutputs(SimulationContext& context, dfloat mlups);

/** @brief Waits for output workers and releases all host and device resources. */
void destroySimulation(SimulationContext& context);

#endif // SIMULATION_CUH
