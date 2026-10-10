#include "simulation.cuh"

#include <climits>
#include <memory>

#include "include/errorDef.h"

void processStepOutputs(SimulationContext& context, const SaveField& saveField) {
    // Same-step consumers share one device-to-host macro copy. This avoids a
    // full pinned field per checkpoint/report/macro request.
    const bool checkpointRequested = saveField.checkpoint;
    const bool reportRequested = saveField.reportSave;
    const bool macroRequested = saveField.macrSave;
    const bool snapshotRequested = checkpointRequested || reportRequested || macroRequested;

    if (checkpointRequested) {
        printf("\n--------------------------- Saving checkpoint %06d ---------------------------\n", context.step + 1);
    }
    if (reportRequested) {
        printf("\n--------------------------- Saving report %06d ---------------------------\n", context.step);
    }
    if (macroRequested) {
        printf("\n--------------------------- Saving macro %06d ---------------------------\n", context.step);
    }

    if (snapshotRequested) {
        // h_fMom has one owner at a time. Finish the previous worker, acquire
        // ownership, and overwrite it with a coherent multi-GPU snapshot.
        finishHostSnapshotTask(context);
        std::unique_lock<std::mutex> snapshotLock(context.hostSnapshotMutex);
        context.deviceField.copyMacroscopicDeviceField(context.hostField, context.devices.data());

        std::shared_ptr<CanonicalCheckpointHostState> checkpointPopulations;
        const int checkpointStep = context.step + 1;
        if (checkpointRequested) {
            // Checkpoints store decomposition-independent logical populations.
            // Refresh periodic partition faces before canonical host capture.
            const bool checkpointParity = context.devices[0].effectiveStreamingParity(checkpointStep);
            for (int g = 0; g < N_GPUS; ++g) {
                context.devices[g].sendTopToNext(g, context.devices.data(),
                    context.streamsLBM[g], checkpointParity);
                context.devices[g].sendBottomToPrev(g, context.devices.data(),
                    context.streamsLBM[g], checkpointParity);
            }
            for (int g = 0; g < N_GPUS; ++g) {
                checkCudaErrors(cudaSetDevice(GPUS_TO_USE[g]));
                checkCudaErrors(cudaDeviceSynchronize());
            }
            checkpointPopulations = std::make_shared<CanonicalCheckpointHostState>();
            checkpointPopulations->resizeForGlobalDomain();
            for (int g = 0; g < N_GPUS; ++g) {
                context.devices[g].captureCanonicalCheckpointDeviceField(
                    *checkpointPopulations, context.threadBlock, checkpointStep, g);
            }
        }

        const int snapshotStep = context.step;
        std::promise<void> snapshotOwnedPromise;
        std::future<void> snapshotOwned = snapshotOwnedPromise.get_future();
        // The worker holds hostSnapshotMutex while every CPU consumer reads
        // h_fMom. Disk serialization may continue while the next GPU step runs.
        context.hostSnapshotFuture = std::async(std::launch::async,
            [&context, snapshotStep, checkpointStep, checkpointRequested,
                reportRequested, macroRequested, checkpointPopulations,
                owned = std::move(snapshotOwnedPromise)]() mutable {
                std::lock_guard<std::mutex> workerLock(context.hostSnapshotMutex);

                #ifdef PARTICLE_MODEL
                    // Particle checkpoint state is device-resident. Capture it
                    // before releasing the simulation through the handshake.
                    if (checkpointRequested) {
                        context.particleField.saveCheckpoint(snapshotStep);
                    }
                #endif // PARTICLE_MODEL

                // Signal only after the worker owns the host snapshot and all
                // device-resident checkpoint components have been captured.
                owned.set_value();

                if (checkpointRequested) {
                    saveGlobalSimCheckpoint(context.hostField.h_fMom,
                        *checkpointPopulations, checkpointStep);
                    if (console_flush) fflush(stdout);
                }
                if (macroRequested && !ONLY_FINAL_MACRO) {
                    context.hostField.saveMacrHostField(snapshotStep,
                        context.savingMacrVtk, context.savingMacrBin, false);
                }
                if (reportRequested) {
                    context.deviceField.treatDataDeviceField(context.hostField, snapshotStep);
                }
            });

        // Release our lock, then wait until the worker has acquired ownership.
        // After this handshake, GPU work can safely continue asynchronously.
        snapshotLock.unlock();
        snapshotOwned.get();
    }

    #ifdef PARTICLE_MODEL
        if (reportRequested) {
            context.particleField.exportWallForces(context.step);
        }
    #endif // PARTICLE_MODEL
    if (console_flush && snapshotRequested) fflush(stdout);
}

void runConfiguredTestMetric(SimulationContext& context) {
    #ifdef TESTS
        // Case fragments historically refer to these local names. Preserve
        // that small compile-time interface while keeping it out of main.cu.
        finishHostSnapshotTask(context);
        std::lock_guard<std::mutex> snapshotLock(context.hostSnapshotMutex);
        int step = context.step;
        HostField& hostField = context.hostField;
        DeviceField& deviceField = context.deviceField;
        std::vector<DeviceField>& devices = context.devices;
        #ifdef PARTICLE_MODEL
            ParticleField& particleField = context.particleField;
        #endif // PARTICLE_MODEL
        #include CASE_TEST_METRIC
    #else
        (void)context;
    #endif // TESTS
}

void processParticleOutputs(SimulationContext& context, const SaveField& saveField) {
    #ifdef PARTICLE_MODEL
        if (saveField.particleSave) {
            printf("\n------------------------- Saving particles %06d -------------------------\n", context.step);
            context.particleField.saveInfo(context.step, context.savingMacrParticle);
        }
        if (console_flush) fflush(stdout);
    #else
        (void)context;
        (void)saveField;
    #endif // PARTICLE_MODEL
}

dfloat finishSimulationTiming(SimulationContext& context) {
    // Wall-clock timing is used because CUDA events cannot span all devices.
    for (int g = 0; g < N_GPUS; ++g) {
        checkCudaErrors(cudaSetDevice(GPUS_TO_USE[g]));
        checkCudaErrors(cudaDeviceSynchronize());
    }

    const auto simulationEndTime = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double> elapsed = simulationEndTime - context.simulationStartTime;
    const size_t physicalNodeCount = static_cast<size_t>(NX) * NY * NZ;
    const dfloat mlups = (static_cast<dfloat>((context.step - context.initialStep) * physicalNodeCount) / 1e6) / elapsed.count();
    printf("MLUPS: %f  (steps=%d, nodes=%zu, time=%.3fs)\n",
        mlups, context.step - context.initialStep, physicalNodeCount, elapsed.count());
    if (console_flush) fflush(stdout);
    return mlups;
}

void writeFinalOutputs(SimulationContext& context, dfloat mlups) {
    // Final consumers require a stable, complete host field, so no further GPU
    // progress is allowed to overlap this locked snapshot.
    finishHostSnapshotTask(context);
    std::lock_guard<std::mutex> finalSnapshotLock(context.hostSnapshotMutex);
    context.deviceField.copyMacroscopicDeviceField(context.hostField, context.devices.data());

    // FNV-1a over logical global moment order provides a decomposition-neutral
    // regression fingerprint for otherwise identical runs.
    unsigned long long checksum = 1469598103934665603ULL;
    for (size_t z = 0; z < static_cast<size_t>(NZ); ++z) {
        for (size_t y = 0; y < static_cast<size_t>(NY); ++y) {
            for (size_t x = 0; x < static_cast<size_t>(NX); ++x) {
                for (int moment = 0; moment < NUMBER_MOMENTS; ++moment) {
                    const size_t index = idxMom(x % BLOCK_NX, y % BLOCK_NY, z % BLOCK_NZ,
                        moment, x / BLOCK_NX, y / BLOCK_NY, z / BLOCK_NZ);
                    const unsigned char* valueBytes = reinterpret_cast<const unsigned char*>(
                        context.hostField.h_fMom + index);
                    for (size_t byte = 0; byte < sizeof(dfloat); ++byte) {
                        checksum ^= static_cast<unsigned long long>(valueBytes[byte]);
                        checksum *= 1099511628211ULL;
                    }
                }
            }
        }
    }
    printf("MOMENT_CHECKSUM: %016llx\n", checksum);

    context.hostField.saveMacrHostField(context.step,
        context.savingMacrVtk, context.savingMacrBin, false);

    if (CHECKPOINT_SAVE) {
        // As with periodic checkpoints, canonical populations are captured
        // only after both partition faces represent the selected generation.
        printf("\n--------------------------- Saving checkpoint %06d ---------------------------\n", context.step);
        CanonicalCheckpointHostState checkpointPopulations;
        checkpointPopulations.resizeForGlobalDomain();
        const bool checkpointParity = context.devices[0].effectiveStreamingParity(context.step);
        for (int g = 0; g < N_GPUS; ++g) {
            context.devices[g].sendTopToNext(g, context.devices.data(),
                context.streamsLBM[g], checkpointParity);
            context.devices[g].sendBottomToPrev(g, context.devices.data(),
                context.streamsLBM[g], checkpointParity);
        }
        for (int g = 0; g < N_GPUS; ++g) {
            checkCudaErrors(cudaSetDevice(GPUS_TO_USE[g]));
            checkCudaErrors(cudaDeviceSynchronize());
        }
        for (int g = 0; g < N_GPUS; ++g) {
            context.devices[g].captureCanonicalCheckpointDeviceField(
                checkpointPopulations, context.threadBlock, context.step, g);
        }
        saveGlobalSimCheckpoint(context.hostField.h_fMom, checkpointPopulations, context.step);
        if (console_flush) fflush(stdout);
    }

    for (int g = 0; g < N_GPUS; ++g) {
        checkCudaErrors(cudaSetDevice(GPUS_TO_USE[g]));
        checkCudaErrors(cudaDeviceSynchronize());
    }

    #if MEAN_FLOW
        context.hostField.saveMacrHostField(INT_MAX,
            context.savingMacrVtk, context.savingMacrBin, true);
    #endif // MEAN_FLOW

    #if defined(NON_NEWTONIAN_FLUID) || defined(CONFORMATION_TENSOR)
        #ifdef PHI_DIST
            saveSimInfo(context.step, mlups, context.deviceField.phasePropsA,
                context.deviceField.phasePropsB, true);
        #else
            saveSimInfo(context.step, mlups, context.deviceField.phasePropsA);
        #endif // PHI_DIST
    #else
        saveSimInfo(context.step, mlups, fluidPhaseProps{});
    #endif // defined(NON_NEWTONIAN_FLUID) || defined(CONFORMATION_TENSOR)
}

void destroySimulation(SimulationContext& context) {
    // saveMacrHostField may own independent VTK/BIN writers. Do not release
    // pinned host memory until every writer reports completion.
    while (context.savingMacrVtk) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    #ifdef PARTICLE_MODEL
        context.particleField.waitForSaving(context.savingMacrParticle);
    #endif // PARTICLE_MODEL
    for (size_t i = 0; i < context.savingMacrBin.size(); ++i) {
        while (context.savingMacrBin[i]) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

    context.hostField.freeHostField();
    // CUDA objects must be destroyed on the device that created them.
    for (int g = 0; g < N_GPUS; ++g) {
        context.threads.emplace_back([&context, g]() {
            checkCudaErrors(cudaSetDevice(GPUS_TO_USE[g]));
            #ifdef OVERLAP_HALO_TRANSFERS
                checkCudaErrors(cudaEventDestroy(context.haloReady[g]));
                checkCudaErrors(cudaEventDestroy(context.boundaryReady[g]));
                checkCudaErrors(cudaEventDestroy(context.stepReady[g]));
                checkCudaErrors(cudaStreamDestroy(context.streamsHalo[g]));
                checkCudaErrors(cudaStreamDestroy(context.streamsBoundary[g]));
            #endif // OVERLAP_HALO_TRANSFERS
            checkCudaErrors(cudaStreamDestroy(context.streamsLBM[g]));
            context.devices[g].freeDeviceField(g);
        });
    }
    for (auto& thread : context.threads) thread.join();
    context.threads.clear();

    #ifdef PARTICLE_MODEL
        checkCudaErrors(cudaEventDestroy(context.particleForceFieldReady));
        context.particleField.freeMemory();
        context.particleField.destroyStreams();
    #endif // PARTICLE_MODEL
}
