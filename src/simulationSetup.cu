#include "simulation.cuh"

#include <cstdlib>
#include <exception>
#include <iomanip>
#include <sstream>
#include <stdexcept>

#include "include/errorDef.h"

SimulationContext::SimulationContext()
    : devices(N_GPUS), decomposition(N_GPUS),
      threadBlock(BLOCK_NX, BLOCK_NY, BLOCK_NZ),
      gridBlock(decomposition[0].fullGrid) {
    // Reuse one host thread container for allocation, initialization, stepping,
    // and teardown without reallocating its backing storage.
    threads.reserve(N_GPUS);
}

void finishHostSnapshotTask(SimulationContext& context) {
    if (context.hostSnapshotFuture.valid()) {
        // get(), rather than wait(), also propagates writer exceptions.
        context.hostSnapshotFuture.get();
    }
}

void synchronizeSimulationStreams(SimulationContext& context) {
    // This is the host-wide barrier used before one partition consumes data
    // produced independently by every other partition.
    for (int g = 0; g < N_GPUS; ++g) {
        checkCudaErrors(cudaSetDevice(GPUS_TO_USE[g]));
        checkCudaErrors(cudaStreamSynchronize(context.streamsLBM[g]));
    }
}

namespace {

void updateDensityCorrection(SimulationContext& context, size_t densityStep) {
    #ifdef DENSITY_CORRECTION
        // Each device reduces only its local physical slab and copies the two
        // partial values into pinned host storage on its normal LBM stream.
        for (int g = 0; g < N_GPUS; ++g) {
            const size_t localNZ = context.decomposition[g].localNZ;
            checkCudaErrors(cudaSetDevice(GPUS_TO_USE[g]));
            context.devices[g].reduceDensityPartialDeviceField(
                g, localNZ,
                &context.hostField.h_density_partial_sum[g],
                &context.hostField.h_density_partial_count[g], context.streamsLBM[g]);
        }

        synchronizeSimulationStreams(context);

        // The host combines partition results so all devices receive exactly
        // the same global correction for the next collision.
        dfloat globalSum = 0.0_df;
        unsigned long long globalCount = 0;
        for (int g = 0; g < N_GPUS; ++g) {
            globalSum += context.hostField.h_density_partial_sum[g];
            globalCount += context.hostField.h_density_partial_count[g];
        }
        if (globalCount == 0) {
            throw std::runtime_error("Density correction found no physical active fluid nodes");
        }

        context.hostField.h_mean_rho[0] = globalSum / static_cast<dfloat>(globalCount);
        for (int g = 0; g < N_GPUS; ++g) {
            checkCudaErrors(cudaSetDevice(GPUS_TO_USE[g]));
            checkCudaErrors(cudaMemcpyAsync(
                context.devices[g].d_mean_rho[g], context.hostField.h_mean_rho,
                sizeof(dfloat), cudaMemcpyHostToDevice, context.streamsLBM[g]));
        }

        std::ostringstream history;
        history << std::scientific << std::setprecision(9)
            << "step," << densityStep
            << ",mean_rho," << (RHO_0 + context.hostField.h_mean_rho[0])
            << ",correction," << context.hostField.h_mean_rho[0]
            << ",active_nodes," << globalCount;
        saveTreatData("_densityHistory", history.str(), static_cast<int>(densityStep),
            context.densityHistoryStarted);
        context.densityHistoryStarted = true;
    #else
        (void)context;
        (void)densityStep;
    #endif // DENSITY_CORRECTION
}

void publishMacroHalos(SimulationContext& context) {
    // Pack all send faces first, then cross the host barrier. This prevents an
    // early receiver from reading a neighbor whose packing kernel is pending.
    for (int g = 0; g < N_GPUS; ++g) {
        context.devices[g].packMacroHalosDeviceField(g, context.streamsLBM[g]);
    }
    synchronizeSimulationStreams(context);
}

} // namespace

void updatePhaseDerivatives(SimulationContext& context) {
    #ifdef PHI_DIST
        // Phase derivatives must use phi and neighboring phi values from the
        // same population generation.
        publishMacroHalos(context);
        // Normals depend on the exchanged phi halo.
        for (int g = 0; g < N_GPUS; ++g) {
            context.devices[g].exchangeMacroField(g, context.devices.data(),
                &macroInterfaceGPUData::phi, context.streamsLBM[g], context.stepParity);
        }
        synchronizeSimulationStreams(context);

        for (int g = 0; g < N_GPUS; ++g) {
            context.devices[g].computePhaseNormalsDeviceField(context.threadBlock,
                g, context.streamsLBM[g], context.stepParity);
        }
        publishMacroHalos(context);

        // Some transport/force models additionally consume neighbor normals
        // or chemical potential before computing their final auxiliaries.
        #ifdef PHASE_NEEDS_NORMAL_HALOS
            for (int g = 0; g < N_GPUS; ++g) {
                context.devices[g].exchangeMacroField(g, context.devices.data(),
                    &macroInterfaceGPUData::nx, context.streamsLBM[g], context.stepParity);
                context.devices[g].exchangeMacroField(g, context.devices.data(),
                    &macroInterfaceGPUData::ny, context.streamsLBM[g], context.stepParity);
                context.devices[g].exchangeMacroField(g, context.devices.data(),
                    &macroInterfaceGPUData::nz, context.streamsLBM[g], context.stepParity);
            }
        #endif // PHASE_NEEDS_NORMAL_HALOS
        #ifdef PHASE_TRANSPORT_CAHN_HILLIARD
            for (int g = 0; g < N_GPUS; ++g) {
                context.devices[g].exchangeMacroField(g, context.devices.data(),
                    &macroInterfaceGPUData::mu, context.streamsLBM[g], context.stepParity);
            }
        #endif // PHASE_TRANSPORT_CAHN_HILLIARD
        synchronizeSimulationStreams(context);

        for (int g = 0; g < N_GPUS; ++g) {
            context.devices[g].computePhaseAuxiliariesDeviceField(context.threadBlock,
                g, context.streamsLBM[g], context.stepParity);
        }
        synchronizeSimulationStreams(context);
        CHECK_KERNEL_ERR("Phase derivatives kernels");
    #else
        (void)context;
    #endif // PHI_DIST
}

bool initializeSimulation(SimulationContext& context) {
    // P2P availability is established before any partition allocates or tries
    // to exchange peer-resident halo buffers.
    context.deviceField.enablePeerAccessDeviceField();

    // The outer pointer array is required even when RANDOM_NUMBERS is disabled;
    // initialization still receives a null pointer entry for every partition.
    dfloat** randomNumbers = static_cast<dfloat**>(calloc(N_GPUS, sizeof(dfloat*)));
    if (randomNumbers == nullptr) {
        fprintf(stderr, "Failed to allocate random-number pointer array\n");
        return false;
    }

    context.hostField.allocateHostMemoryHostField();
    #ifdef DENSITY_CORRECTION
        context.hostField.allocateDensityCorrectionMemory();
    #endif // DENSITY_CORRECTION

    const size_t peakPinnedHostFieldBytes = context.hostField.getPinnedBytesAllocated();
    printf("Peak persistent pinned host-field memory: %.2f MiB (%.3f GiB)\n",
        static_cast<double>(peakPinnedHostFieldBytes) / (1024.0 * 1024.0),
        static_cast<double>(peakPinnedHostFieldBytes) / (1024.0 * 1024.0 * 1024.0));
    if (console_flush) fflush(stdout);

    // Allocate each device partition concurrently. Every worker selects its
    // CUDA device before touching allocations, streams, or events.
    for (int g = 0; g < N_GPUS; ++g) {
        context.threads.emplace_back([&context, g]() {
            checkCudaErrors(cudaSetDevice(GPUS_TO_USE[g]));
            context.devices[g].allocateDeviceMemoryDeviceField(g, context.decomposition[g]);

            #if defined(NON_NEWTONIAN_FLUID) || defined(CONFORMATION_TENSOR)
                #if defined(CASE_PHASE_PROPS_PHASE1)
                    context.devices[g].phasePropsA = CASE_PHASE_PROPS_PHASE1;
                #elif defined(CASE_PHASE_PROPS)
                    context.devices[g].phasePropsA = CASE_PHASE_PROPS;
                #endif // defined(CASE_PHASE_PROPS_PHASE1)
                #ifdef PHI_DIST
                    #if defined(CASE_PHASE_PROPS_PHASE2)
                        context.devices[g].phasePropsB = CASE_PHASE_PROPS_PHASE2;
                    #elif defined(CASE_PHASE_PROPS)
                        context.devices[g].phasePropsB = CASE_PHASE_PROPS;
                    #endif // defined(CASE_PHASE_PROPS_PHASE2)
                #endif // PHI_DIST
            #endif // defined(NON_NEWTONIAN_FLUID) || defined(CONFORMATION_TENSOR)
            #ifdef CONFORMATION_TENSOR
                if (!validateVeFluidProps(context.devices[g].phasePropsA.ve)
                    #ifdef PHI_DIST
                        || !validateVeFluidProps(context.devices[g].phasePropsB.ve)
                        || (context.devices[g].phasePropsA.ve.type != VE_NEWTONIAN
                            && context.devices[g].phasePropsB.ve.type != VE_NEWTONIAN
                            && context.devices[g].phasePropsA.ve.type != context.devices[g].phasePropsB.ve.type)
                    #endif // PHI_DIST
                ) {
                    fprintf(stderr, "Invalid or unsupported viscoelastic phase properties\n");
                    abort();
                }
            #endif // CONFORMATION_TENSOR

            #ifdef DENSITY_CORRECTION
                context.devices[g].allocateDensityCorrectionMemory(g);
            #endif // DENSITY_CORRECTION

            checkCudaErrors(cudaStreamCreateWithFlags(&context.streamsLBM[g], cudaStreamNonBlocking));
            #ifdef OVERLAP_HALO_TRANSFERS
                int leastPriority = 0;
                int greatestPriority = 0;
                checkCudaErrors(cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority));
                checkCudaErrors(cudaStreamCreateWithPriority(&context.streamsBoundary[g], cudaStreamNonBlocking, greatestPriority));
                checkCudaErrors(cudaStreamCreateWithFlags(&context.streamsHalo[g], cudaStreamNonBlocking));
                checkCudaErrors(cudaEventCreateWithFlags(&context.stepReady[g], cudaEventDisableTiming));
                checkCudaErrors(cudaEventCreateWithFlags(&context.boundaryReady[g], cudaEventDisableTiming));
                checkCudaErrors(cudaEventCreateWithFlags(&context.haloReady[g], cudaEventDisableTiming));
            #endif // OVERLAP_HALO_TRANSFERS
            checkCudaErrors(cudaDeviceSynchronize());
        });
    }
    for (auto& thread : context.threads) thread.join();
    context.threads.clear();

    #ifdef PARTICLE_MODEL
        // Particle coupling retains its existing single-device ownership.
        checkCudaErrors(cudaSetDevice(GPUS_TO_USE[GPU_INDEX]));
        context.particleField.allocateMemory();
        context.particleField.setupStreams();
        checkCudaErrors(cudaEventCreateWithFlags(&context.particleForceFieldReady, cudaEventDisableTiming));
    #endif // PARTICLE_MODEL

    context.step = INI_STEP;
    context.savingMacrBin = std::vector<std::atomic<bool>>(context.hostField.NThread);
    for (int i = 0; i < context.hostField.NThread; ++i) {
        context.savingMacrBin[i].store(false);
    }

    // Domain initialization may restore a checkpoint. Keep each returned step
    // separate until every partition has completed and consistency is checked.
    std::vector<int> initializedSteps(N_GPUS, INI_STEP);
    std::vector<std::exception_ptr> initializationErrors(N_GPUS);
    for (int g = 0; g < N_GPUS; ++g) {
        context.threads.emplace_back([&context, &randomNumbers, &initializedSteps, &initializationErrors, g]() {
            try {
                checkCudaErrors(cudaSetDevice(GPUS_TO_USE[g]));
                context.devices[g].initializeDomainDeviceField(context.hostField, randomNumbers,
                    initializedSteps[g], context.threadBlock, g);
                const cudaError_t error = cudaGetLastError();
                if (error != cudaSuccess) {
                    printf("Error on device %d after initialization: %s\n",
                        GPUS_TO_USE[g], cudaGetErrorString(error));
                }
                checkCudaErrors(cudaDeviceSynchronize());
            } catch (...) {
                initializationErrors[g] = std::current_exception();
            }
        });
    }
    for (auto& thread : context.threads) thread.join();
    context.threads.clear();

    for (int g = 0; g < N_GPUS; ++g) {
        if (!initializationErrors[g]) continue;
        try {
            std::rethrow_exception(initializationErrors[g]);
        } catch (const std::exception& error) {
            fprintf(stderr, "Domain initialization failed on partition %d: %s\n", g, error.what());
        } catch (...) {
            fprintf(stderr, "Domain initialization failed on partition %d\n", g);
        }
        free(randomNumbers);
        return false;
    }

    context.step = initializedSteps[0];
    for (int g = 1; g < N_GPUS; ++g) {
        if (initializedSteps[g] != context.step) {
            fprintf(stderr, "Checkpoint step mismatch: partition 0 resumes at %d, partition %d at %d\n",
                context.step, g, initializedSteps[g]);
            free(randomNumbers);
            return false;
        }
    }
    context.stepParity = context.devices[0].effectiveStreamingParity(context.step);

    #ifdef RANDOM_NUMBERS
        for (int g = 0; g < N_GPUS; ++g) {
            checkCudaErrors(cudaSetDevice(GPUS_TO_USE[g]));
            checkCudaErrors(cudaFree(randomNumbers[g]));
            randomNumbers[g] = nullptr;
        }
    #endif // RANDOM_NUMBERS
    free(randomNumbers);

    // Checkpoints contain primary moments/populations, not every derived field.
    // Rebuild all derived state before the first collision consumes it.
    #ifdef DENSITY_CORRECTION
        updateDensityCorrection(context, static_cast<size_t>(context.step));
    #endif // DENSITY_CORRECTION

    #ifdef PHI_DIST
        updatePhaseDerivatives(context);
    #else
        if constexpr (MACRO_HALOS_REQUIRED) publishMacroHalos(context);
    #endif // PHI_DIST

    #ifdef CONFORMATION_PHYSICAL_A_HALO
        // Seed polymer-stress generation n from initialized/restored moments.
        // Later timesteps reconstruct generation n+1 in runHalfStepAndParticles().
        for (int g = 0; g < N_GPUS; ++g) {
            context.devices[g].fillPolymerStressHaloDeviceField(context.decomposition[g].fullGrid,
                context.threadBlock, context.stepParity, g, context.streamsLBM[g]);
        }
        synchronizeSimulationStreams(context);
        for (int g = 0; g < N_GPUS; ++g) {
            context.devices[g].exchangePolymerStressHalosDeviceField(
                g, context.devices.data(), context.stepParity, context.streamsLBM[g]);
        }
        synchronizeSimulationStreams(context);
    #endif // CONFORMATION_PHYSICAL_A_HALO

    #ifdef OVERLAP_HALO_TRANSFERS
        // Bootstrap the first incoming generation. Subsequent generations are
        // published immediately after their boundary collision completes.
        for (int g = 0; g < N_GPUS; ++g) {
            checkCudaErrors(cudaSetDevice(GPUS_TO_USE[g]));
            checkCudaErrors(cudaEventRecord(context.stepReady[g], context.streamsLBM[g]));
        }
        for (int g = 0; g < N_GPUS; ++g) {
            checkCudaErrors(cudaSetDevice(GPUS_TO_USE[g]));
            context.devices[g].sendTopToNext(g, context.devices.data(), context.streamsHalo[g], context.stepParity);
            context.devices[g].sendBottomToPrev(g, context.devices.data(), context.streamsHalo[g], context.stepParity);
            if constexpr (MACRO_HALOS_REQUIRED) {
                context.devices[g].sendMacroTopToNext(g, context.devices.data(), context.streamsHalo[g], context.stepParity);
                context.devices[g].sendMacroBottomToPrev(g, context.devices.data(), context.streamsHalo[g], context.stepParity);
            }
            checkCudaErrors(cudaEventRecord(context.haloReady[g], context.streamsHalo[g]));
        }
    #endif // OVERLAP_HALO_TRANSFERS

    context.initialStep = context.step;
    printf("Domain Initialized. Starting simulation\n");
    if (console_flush) fflush(stdout);

    #ifdef PARTICLE_MODEL
        context.particleField.initialize(&context.step, context.gridBlock, context.threadBlock);
        context.particleField.saveInfo(context.step, context.savingMacrParticle);
    #endif // PARTICLE_MODEL

    #ifdef DYNAMIC_SHARED_MEMORY
        for (int g = 0; g < N_GPUS; ++g) {
            checkCudaErrors(cudaSetDevice(GPUS_TO_USE[g]));
            if (configureDynamicSharedMemory(GPUS_TO_USE[g])) return false;
        }
    #endif // DYNAMIC_SHARED_MEMORY

    context.simulationStartTime = std::chrono::high_resolution_clock::now();
    return true;
}

void updateAuxiliaryPhysics(SimulationContext& context) {
    // Collision has produced step n+1. Recompute the derived fields that the
    // remaining half-step or the next collision will consume.
    #ifdef PHI_DIST
        updatePhaseDerivatives(context);
    #endif // PHI_DIST

    #ifdef DENSITY_CORRECTION
        updateDensityCorrection(context, static_cast<size_t>(context.step) + 1);
        CHECK_KERNEL_ERR("Density correction reduction");
    #endif // DENSITY_CORRECTION

    #if !defined(PHI_DIST) && !defined(DENSITY_CORRECTION)
        (void)context;
    #endif // !defined(PHI_DIST) && !defined(DENSITY_CORRECTION)
}
