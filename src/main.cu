#include "main.cuh"
#include "hostField.cuh"
#include "deviceField.cuh"
#include "saveField.cuh"
#include "include/memory_estimate.h"
#ifdef PARTICLE_MODEL
#include "particleField.cuh"
#endif //PARTICLE_MODEL

#include <chrono>
#include <cstring>
#include <future>
#include <mutex>

#if defined(OVERLAP_HALO_TRANSFERS) && !defined(SPLIT_Z_COLLISION_KERNEL)
#error "OVERLAP_HALO_TRANSFERS requires SPLIT_Z_COLLISION_KERNEL"
#endif

using namespace std;

int main(int argc, char** argv) {
    if (argc == 2 && std::strcmp(argv[1], "--memory-estimate") == 0) {
        printCompileTimeMemoryEstimate();
        return 0;
    }

    // Setup saving folder
    folderSetup();

    // Set cuda device
    // checkCudaErrors(cudaSetDevice(GPU_INDEX));

    // Field Variables
    HostField hostField;
    DeviceField deviceField;
    // One ownership lock covers every reader and writer of the shared host
    // snapshot: GPU copies, checkpointing, macro extraction and treat-data.
    std::mutex hostSnapshotMutex;
    std::future<void> hostSnapshotFuture;
    auto finishHostSnapshotTask = [&]() {
        if (hostSnapshotFuture.valid()) {
            hostSnapshotFuture.get();
        }
    };

    // Multi-GPU Variables
    std::vector<std::thread> threads;
    std::vector<DeviceField> devices(N_GPUS);

    int slice = NZ / N_GPUS;

    static_assert(NZ % N_GPUS == 0,
        "NZ must be exactly divisible by N_GPUS for uniform Z-slab decomposition.");

    threads.reserve(N_GPUS);

    /* ----------------- GRID AND THREADS DEFINITION FOR LBM ---------------- */
    dim3 threadBlock(BLOCK_NX, BLOCK_NY, BLOCK_NZ);
    dim3 gridBlock(NUM_BLOCK_X, NUM_BLOCK_Y, NUM_BLOCK_Z_LOCAL);

    // Enable peer-to-peer (P2P) access between GPUs
    deviceField.enablePeerAccessDeviceField();

    /* ------------------------- ALLOCATION FOR CPU ------------------------- */
    int step = 0;

    // The outer array is required even when RANDOM_NUMBERS is disabled because
    // gpuInitialization_mom() is still passed randomNumbers[g] (a null device
    // pointer in that configuration).
    dfloat** randomNumbers = (dfloat**)calloc(N_GPUS, sizeof(dfloat*));
    if (randomNumbers == nullptr) {
        fprintf(stderr, "Failed to allocate random-number pointer array\n");
        return EXIT_FAILURE;
    }

    hostField.allocateHostMemoryHostField();

    #ifdef DENSITY_CORRECTION
    // This is one global host value, not one allocation per GPU.
    hostField.allocateDensityCorrectionMemory();
    #endif

    // Macro saving, checkpointing and treat-data share h_fMom.  All three take
    // hostSnapshotMutex while reading it, and every GPU-to-host copy takes the
    // same mutex while overwriting it.  This keeps one full moment snapshot in
    // host RAM instead of one per consumer. When multiple outputs trigger on
    // the same step, one combined worker lets all three reuse one device copy.
    // This is not total process RSS: CUDA runtime state, particle containers,
    // and transient serialization buffers are outside this accounting.
    size_t peakPinnedHostFieldBytes = hostField.getPinnedBytesAllocated();
    printf("Peak persistent pinned host-field memory: %.2f MiB (%.3f GiB)\n",
        static_cast<double>(peakPinnedHostFieldBytes) / (1024.0 * 1024.0),
        static_cast<double>(peakPinnedHostFieldBytes) / (1024.0 * 1024.0 * 1024.0));
    if(console_flush) fflush(stdout);
    
    /* -------------- ALLOCATION FOR GPUs ------------- */
    cudaStream_t streamsLBM[N_GPUS];
    #ifdef PARTICLE_MODEL
    // Particle coupling currently owns one single-GPU field/stream.
    ParticleField particleField;
    cudaEvent_t particleForceFieldReady;
    #endif
    #ifdef OVERLAP_HALO_TRANSFERS
    cudaStream_t streamsBoundary[N_GPUS];
    cudaStream_t streamsHalo[N_GPUS];
    cudaEvent_t stepReady[N_GPUS];
    cudaEvent_t boundaryReady[N_GPUS];
    cudaEvent_t haloReady[N_GPUS];
    #endif
    threads.reserve(N_GPUS);
    for(int g = 0; g < N_GPUS; g++){

        threads.emplace_back([&, g]() {
            checkCudaErrors(cudaSetDevice(GPUS_TO_USE[g]));
            devices[g].allocateDeviceMemoryDeviceField(g);
    
            #if defined(NON_NEWTONIAN_FLUID) || defined(CONFORMATION_TENSOR)
            #if defined(CASE_PHASE_PROPS_PHASE1)
            devices[g].phasePropsA = CASE_PHASE_PROPS_PHASE1;
            #elif defined(CASE_PHASE_PROPS)
            devices[g].phasePropsA = CASE_PHASE_PROPS;
            #endif
            #ifdef PHI_DIST
            #if defined(CASE_PHASE_PROPS_PHASE2)
            devices[g].phasePropsB = CASE_PHASE_PROPS_PHASE2;
            #elif defined(CASE_PHASE_PROPS)
            devices[g].phasePropsB = CASE_PHASE_PROPS;
            #endif
            #endif
            #endif //NON_NEWTONIAN_FLUID || CONFORMATION_TENSOR
            
            #ifdef DENSITY_CORRECTION
                devices[g].allocateDensityCorrectionMemory(g);
            #endif //DENSITY_CORRECTION

            /* -------------- Setup Streams ------------- */
            checkCudaErrors(cudaSetDevice(GPUS_TO_USE[g]));
            checkCudaErrors(cudaStreamCreateWithFlags(&streamsLBM[g], cudaStreamNonBlocking));
            #ifdef OVERLAP_HALO_TRANSFERS
            int leastPriority = 0;
            int greatestPriority = 0;
            checkCudaErrors(cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority));
            checkCudaErrors(cudaStreamCreateWithPriority(&streamsBoundary[g], cudaStreamNonBlocking, greatestPriority));
            checkCudaErrors(cudaStreamCreateWithFlags(&streamsHalo[g], cudaStreamNonBlocking));
            checkCudaErrors(cudaEventCreateWithFlags(&stepReady[g], cudaEventDisableTiming));
            checkCudaErrors(cudaEventCreateWithFlags(&boundaryReady[g], cudaEventDisableTiming));
            checkCudaErrors(cudaEventCreateWithFlags(&haloReady[g], cudaEventDisableTiming));
            #endif
            checkCudaErrors(cudaDeviceSynchronize());
        });
    }

    for (auto &t : threads) {
        t.join();
    }

    threads.clear();

    #ifdef PARTICLE_MODEL
    checkCudaErrors(cudaSetDevice(GPUS_TO_USE[GPU_INDEX]));
    particleField.allocateMemory();
    particleField.setupStreams();
    checkCudaErrors(cudaEventCreateWithFlags(
        &particleForceFieldReady, cudaEventDisableTiming));
    #endif

    step = INI_STEP;

    //Declaration of atomic flags to safely control the state of data saving in multiple threads.
    std::atomic<bool> savingMacrVtk(false);
    std::atomic<bool> savingMacrParticle(false);
    std::vector<std::atomic<bool>> savingMacrBin(hostField.NThread);

    for (int i = 0; i < hostField.NThread; i++){
        savingMacrBin[i].store(false);
    }

    /* -------------- Initialize the domain on the devices ------------- */
    std::vector<int> initializedSteps(N_GPUS, INI_STEP);
    for(int g = 0; g < N_GPUS; g++){

        threads.emplace_back([&, g, slice]() {
            checkCudaErrors(cudaSetDevice(GPUS_TO_USE[g]));
            devices[g].initializeDomainDeviceField(hostField, randomNumbers,
                initializedSteps[g], gridBlock, threadBlock, g, slice);
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                printf("Erro no device %d após init: %s\n", GPUS_TO_USE[g], cudaGetErrorString(err));
            }
            checkCudaErrors(cudaDeviceSynchronize());
        });
    }

    for (auto &t : threads) {
        t.join();
    }

    threads.clear();

    step = initializedSteps[0];
    for (int g = 1; g < N_GPUS; ++g) {
        if (initializedSteps[g] != step) {
            fprintf(stderr,
                "Checkpoint step mismatch: partition 0 resumes at %d, partition %d at %d\n",
                step, g, initializedSteps[g]);
            return EXIT_FAILURE;
        }
    }
    bool stepParity = (step & 1u);

    #ifdef RANDOM_NUMBERS
        for (int g = 0; g < N_GPUS; g++) {
            checkCudaErrors(cudaSetDevice(GPUS_TO_USE[g]));
            checkCudaErrors(cudaFree(randomNumbers[g]));
            randomNumbers[g] = nullptr;
        }
    #endif
    free(randomNumbers);
    randomNumbers = nullptr;

    // A host-wide barrier is needed before reading another GPU's send buffers.
    auto synchronizeLBMStreams = [&]() {
        for (int g = 0; g < N_GPUS; ++g) {
            checkCudaErrors(cudaSetDevice(GPUS_TO_USE[g]));
            checkCudaErrors(cudaStreamSynchronize(streamsLBM[g]));
        }
    };

    auto publishMacroHalos = [&]() {
        for (int g = 0; g < N_GPUS; ++g)
            devices[g].packMacroHalosDeviceField(g, slice, streamsLBM[g]);
        synchronizeLBMStreams();
    };

    #ifdef PHI_DIST
    auto updatePhaseDerivatives = [&]() {
        // PHI and its halo neighbors must belong to the same time level.
        publishMacroHalos();
        for (int g = 0; g < N_GPUS; ++g)
            devices[g].exchangeMacroField(g, devices.data(),
                &macroInterfaceGPUData::phi, streamsLBM[g], stepParity);
        synchronizeLBMStreams();

        for (int g = 0; g < N_GPUS; ++g)
            devices[g].computePhaseNormalsDeviceField(gridBlock, threadBlock,
                g, slice, streamsLBM[g], stepParity);
        // Packing follows the phase kernels on each stream; wait for all senders.
        publishMacroHalos();
        #ifdef PHASE_NEEDS_NORMAL_HALOS
        for (int g = 0; g < N_GPUS; ++g) {
            devices[g].exchangeMacroField(g, devices.data(),
                &macroInterfaceGPUData::nx, streamsLBM[g], stepParity);
            devices[g].exchangeMacroField(g, devices.data(),
                &macroInterfaceGPUData::ny, streamsLBM[g], stepParity);
            devices[g].exchangeMacroField(g, devices.data(),
                &macroInterfaceGPUData::nz, streamsLBM[g], stepParity);
        }
        #endif
        #ifdef PHASE_TRANSPORT_CAHN_HILLIARD
        for (int g = 0; g < N_GPUS; ++g)
            devices[g].exchangeMacroField(g, devices.data(),
                &macroInterfaceGPUData::mu, streamsLBM[g], stepParity);
        #endif
        synchronizeLBMStreams();

        for (int g = 0; g < N_GPUS; ++g)
            devices[g].computePhaseAuxiliariesDeviceField(gridBlock, threadBlock,
                g, slice, streamsLBM[g], stepParity);
        synchronizeLBMStreams();
        CHECK_KERNEL_ERR("Phase derivatives kernels");
    };

    // Also rebuild derived phase fields from initialized or restored moments.
    updatePhaseDerivatives();
    #else
    if constexpr (MACRO_HALOS_REQUIRED) {
        publishMacroHalos();
    }
    #endif

    #ifdef OVERLAP_HALO_TRANSFERS
    // Seed the first timestep's compute dependency.
    for (int g = 0; g < N_GPUS; ++g) {
        checkCudaErrors(cudaSetDevice(GPUS_TO_USE[g]));
        checkCudaErrors(cudaEventRecord(stepReady[g], streamsLBM[g]));
    }

    // Bootstrap the incoming generation from the initialized outgoing faces.
    // Subsequent transfers are launched immediately after boundary computation.
    for (int g = 0; g < N_GPUS; ++g) {
        checkCudaErrors(cudaSetDevice(GPUS_TO_USE[g]));
        devices[g].sendTopToNext(g, devices.data(), streamsHalo[g], stepParity);
        devices[g].sendBottomToPrev(g, devices.data(), streamsHalo[g], stepParity);
        if constexpr (MACRO_HALOS_REQUIRED) {
            devices[g].sendMacroTopToNext(g, devices.data(), streamsHalo[g], stepParity);
            devices[g].sendMacroBottomToPrev(g, devices.data(), streamsHalo[g], stepParity);
        }
        checkCudaErrors(cudaEventRecord(haloReady[g], streamsHalo[g]));
    }
    #endif

    int ini_step = step;

    printf("Domain Initialized. Starting simulation\n"); if(console_flush) fflush(stdout);
    
    #ifdef PARTICLE_MODEL
        // Initialize particle field with position, velocity, and solver method
        particleField.initialize(&step, gridBlock, threadBlock);
        particleField.saveInfo(step, savingMacrParticle);
    #endif //PARTICLE_MODEL

    /* ------------------------------ TIMER EVENTS  ------------------------------ */
    // Wall-clock timing for MLUPS (replaces CUDA event timing for multi-GPU compatibility)
    auto simStartTime = std::chrono::high_resolution_clock::now();
    
    /* ------------------------------ LBM LOOP ------------------------------ */

    #ifdef DYNAMIC_SHARED_MEMORY
        for (int g = 0; g < N_GPUS; g++) {
            checkCudaErrors(cudaSetDevice(GPUS_TO_USE[g]));
            if (configureDynamicSharedMemory(GPUS_TO_USE[g])) return 1;
        }
    #endif //DYNAMIC_SHARED_MEMORY
   
    /* --------------------------------------------------------------------- */
    /* ---------------------------- BEGIN LOOP ----------------------------- */
    /* --------------------------------------------------------------------- */
        
    for (;step<N_STEPS;step++){ // step is already initialized

        SaveField saveField;

        // update saving flags
        saveField.flagsUpdate(step);

        stepParity = (step & 1u);

        /* -------------- Exchanging halos between neighboring GPUs using P2P ------------- */
        #ifdef OVERLAP_HALO_TRANSFERS
        // Consume the incoming generation, then produce the next outgoing faces
        // first on a high-priority stream. Boundary and interior kernels operate
        // on disjoint block planes and were already unordered in the full grid.
        for (int g = 0; g < N_GPUS; ++g) {
            checkCudaErrors(cudaSetDevice(GPUS_TO_USE[g]));
            const int previous = (g + N_GPUS - 1) % N_GPUS;
            checkCudaErrors(cudaStreamWaitEvent(streamsBoundary[g], stepReady[g], 0));
            checkCudaErrors(cudaStreamWaitEvent(streamsBoundary[g], haloReady[g], 0));
            if (previous != g) {
                checkCudaErrors(cudaStreamWaitEvent(streamsBoundary[g], haloReady[previous], 0));
            }
            devices[g].gpuMomCollisionStreamDeviceField(gridBlock, threadBlock, step,
                saveField.save, g, slice, streamsBoundary[g], ZKernelLaunchRegion::Boundaries);
            checkCudaErrors(cudaEventRecord(boundaryReady[g], streamsBoundary[g]));
        }

        // Submit the long interior launch only after every boundary launch has
        // been submitted, so priority does not have to preempt resident blocks.
        for (int g = 0; g < N_GPUS; ++g) {
            checkCudaErrors(cudaSetDevice(GPUS_TO_USE[g]));
            devices[g].gpuMomCollisionStreamDeviceField(gridBlock, threadBlock, step,
                saveField.save, g, slice, streamsLBM[g], ZKernelLaunchRegion::Interior);
        }

        // The just-produced faces belong to the next AA generation. Copies can
        // overlap the current interior, and are omitted after the final step.
        if (step + 1 < N_STEPS) {
            const bool nextStepParity = ((step + 1) & 1u);
            for (int g = 0; g < N_GPUS; ++g) {
                checkCudaErrors(cudaSetDevice(GPUS_TO_USE[g]));
                const int next = (g + 1) % N_GPUS;
                checkCudaErrors(cudaStreamWaitEvent(streamsHalo[g], boundaryReady[g], 0));
                if (next != g) {
                    checkCudaErrors(cudaStreamWaitEvent(streamsHalo[g], boundaryReady[next], 0));
                }
                devices[g].sendTopToNext(g, devices.data(), streamsHalo[g], nextStepParity);
                devices[g].sendBottomToPrev(g, devices.data(), streamsHalo[g], nextStepParity);
                if constexpr (MACRO_HALOS_REQUIRED) {
                    devices[g].sendMacroTopToNext(g, devices.data(), streamsHalo[g], nextStepParity);
                    devices[g].sendMacroBottomToPrev(g, devices.data(), streamsHalo[g], nextStepParity);
                }
                checkCudaErrors(cudaEventRecord(haloReady[g], streamsHalo[g]));
            }
        }

        // Join the boundary stream only where downstream work consumes the
        // complete timestep. The interior is already ordered on streamsLBM.
        for (int g = 0; g < N_GPUS; ++g) {
            checkCudaErrors(cudaSetDevice(GPUS_TO_USE[g]));
            checkCudaErrors(cudaStreamWaitEvent(streamsLBM[g], boundaryReady[g], 0));
            checkCudaErrors(cudaGetLastError());
        }
        #else
        // sendTopToNext and sendBottomToPrev access different ghost buffers
        // (pop.Z_1/popAux.Z_1 vs pop.Z_0/popAux.Z_0) so they can launch together.
        for (int g = 0; g < N_GPUS; g++) {
            checkCudaErrors(cudaSetDevice(GPUS_TO_USE[g]));
            devices[g].sendTopToNext(g, devices.data(), streamsLBM[g], stepParity);
            devices[g].sendBottomToPrev(g, devices.data(), streamsLBM[g], stepParity);
            if constexpr (MACRO_HALOS_REQUIRED) {
                devices[g].sendMacroTopToNext(g, devices.data(), streamsLBM[g], stepParity);
                devices[g].sendMacroBottomToPrev(g, devices.data(), streamsLBM[g], stepParity);
            }
        }
    
        for (int g = 0; g < N_GPUS; g++) {
            checkCudaErrors(cudaSetDevice(GPUS_TO_USE[g]));
            checkCudaErrors(cudaDeviceSynchronize());
        }

        //------------------------- Main LBM Kernels -------------------------
        if (N_GPUS == 1) {
            // Direct call — zero thread overhead for single-GPU builds
            checkCudaErrors(cudaSetDevice(GPUS_TO_USE[0]));
            devices[0].gpuMomCollisionStreamDeviceField(gridBlock, threadBlock, step, saveField.save, 0, slice, streamsLBM[0]);
        } else {
            for(int g = 0; g < N_GPUS; g++){
                threads.emplace_back([&, g, slice]() {
                    checkCudaErrors(cudaSetDevice(GPUS_TO_USE[g]));
                    devices[g].gpuMomCollisionStreamDeviceField(gridBlock, threadBlock, step, saveField.save, g, slice, streamsLBM[g]);
                });
            }
            for (auto &t : threads) { t.join(); }
            threads.clear();
        }

       
        for (int g = 0; g < N_GPUS; g++) {
            checkCudaErrors(cudaSetDevice(GPUS_TO_USE[g]));
            checkCudaErrors(cudaDeviceSynchronize());
        }
        
        // AB writes the alternate population generation; make it current only
        // after every device has completed the collision--stream launch.
        #ifdef STREAMING_AB
        for (int g = 0; g < N_GPUS; ++g) {
            devices[g].swapGhostInterfacesDeviceField(g);
        }
        #endif
            
        CHECK_KERNEL_ERR("Stream Collision kernel");
        #endif

        //------------------------- Auxiliary Kernels -------------------------
        #ifdef PHI_DIST
        updatePhaseDerivatives();
        #endif

        if (N_GPUS == 1) {
            devices[0].halfStepKernels(gridBlock, threadBlock, step, 0, slice, streamsLBM[0]);
            #ifdef PARTICLE_MODEL
                checkCudaErrors(cudaEventRecord(
                    particleForceFieldReady, streamsLBM[0]));
                checkCudaErrors(cudaStreamWaitEvent(
                    particleField.getStream()[0], particleForceFieldReady, 0));
                particleField.simulationStep(devices[0].d_fMom[0], step);
            #endif //PARTICLE_MODEL
        } else {
            for(int g = 0; g < N_GPUS; g++){
                threads.emplace_back([&, g, slice]() {
                    checkCudaErrors(cudaSetDevice(GPUS_TO_USE[g]));
                    devices[g].halfStepKernels(gridBlock, threadBlock, step, g, slice, streamsLBM[g]);
                    #ifdef PARTICLE_MODEL
                        particleField.simulationStep(devices[g].d_fMom[g], step);
                    #endif //PARTICLE_MODEL
                });
            }
            for (auto &t : threads) { t.join(); }
            threads.clear();
        }

        #ifdef OVERLAP_HALO_TRANSFERS
        for (int g = 0; g < N_GPUS; ++g) {
            checkCudaErrors(cudaSetDevice(GPUS_TO_USE[g]));
            checkCudaErrors(cudaEventRecord(stepReady[g], streamsLBM[g]));
        }
        #endif

        //------------------------- Saving Data -------------------------
        const bool checkpointRequested = saveField.checkpoint;
        const bool reportRequested = saveField.reportSave;
        const bool macroRequested = saveField.macrSave;
        const bool snapshotRequested = checkpointRequested || reportRequested || macroRequested;

        if (checkpointRequested) {
            printf("\n--------------------------- Saving checkpoint %06d ---------------------------\n", step + 1);
        }
        if (reportRequested) {
            printf("\n--------------------------- Saving report %06d ---------------------------\n", step);
        }
        if (macroRequested) {
            printf("\n--------------------------- Saving macro %06d ---------------------------\n", step);
        }

        if (snapshotRequested) {
            // A single worker owns all same-step consumers. This gives every
            // checkpoint/macro/treatment request a place in the sequence and
            // prevents a later timestep from overtaking any of them.
            finishHostSnapshotTask();
            std::unique_lock<std::mutex> snapshotLock(hostSnapshotMutex);
            deviceField.copyMacroscopicDeviceField(hostField, devices.data(), slice);

            // Checkpoint interface populations must be captured before GPU
            // simulation resumes; disk serialization happens in the worker.
            if (checkpointRequested) {
                for (int g = 0; g < N_GPUS; ++g) {
                    threads.emplace_back([&, g]() {
                        checkCudaErrors(cudaSetDevice(GPUS_TO_USE[g]));
                        devices[g].interfaceCudaMemcpyDeviceField(true, g);
                    });
                }
                for (auto &t : threads) t.join();
                threads.clear();
            }

            const int snapshotStep = step;
            const int checkpointStep = step + 1;
            std::promise<void> snapshotOwnedPromise;
            std::future<void> snapshotOwned = snapshotOwnedPromise.get_future();
            hostSnapshotFuture = std::async(std::launch::async,
                [&, snapshotStep, checkpointStep, checkpointRequested,
                    reportRequested, macroRequested, slice,
                    owned = std::move(snapshotOwnedPromise)]() mutable {
                    std::lock_guard<std::mutex> workerLock(hostSnapshotMutex);

                    #ifdef PARTICLE_MODEL
                    // Particle state is device-resident, so capture it before
                    // telling the main thread that GPU simulation may resume.
                    if (checkpointRequested) {
                        particleField.saveCheckpoint(snapshotStep);
                    }
                    #endif

                    // Ownership is now established and every device-resident
                    // checkpoint component has been captured.
                    owned.set_value();

                    if (checkpointRequested) {
                        int checkpointStepForSave = checkpointStep;
                        std::vector<std::thread> checkpointThreads;
                        checkpointThreads.reserve(N_GPUS);
                        for (int g = 0; g < N_GPUS; ++g) {
                            checkpointThreads.emplace_back([&, g]() {
                                checkCudaErrors(cudaSetDevice(GPUS_TO_USE[g]));
                                devices[g].saveSimCheckpointHostDeviceField(
                                    hostField, checkpointStepForSave, g, slice);
                                if(console_flush) fflush(stdout);
                            });
                        }
                        for (auto &t : checkpointThreads) t.join();
                    }

                    if (macroRequested && !ONLY_FINAL_MACRO) {
                        hostField.saveMacrHostField(snapshotStep,
                            savingMacrVtk, savingMacrBin, false);
                    }

                    if (reportRequested) {
                        deviceField.treatDataDeviceField(
                            hostField, snapshotStep, slice);
                    }
                });

            // Release the capture lock, then wait only until the worker has
            // acquired it. The GPU may continue while CPU consumers run.
            snapshotLock.unlock();
            snapshotOwned.get();
        }

        #ifdef PARTICLE_MODEL
        if (reportRequested) {
            particleField.exportWallForces(step);
        }
        #endif
        if(console_flush && snapshotRequested) fflush(stdout);

        #ifdef TESTS
        finishHostSnapshotTask();
        {
            std::lock_guard<std::mutex> snapshotLock(hostSnapshotMutex);
            #include CASE_TEST_METRIC
        }
        #endif //TESTS

        #ifdef PARTICLE_MODEL
            if (saveField.particleSave){
                printf("\n------------------------- Saving particles %06d -------------------------\n", step);
                particleField.saveInfo(step, savingMacrParticle);
            }
            if(console_flush){fflush(stdout);}
        #endif //PARTICLE_MODEL

    } 

    /* --------------------------------------------------------------------- */
    /* ------------------------------ END LOOP ----------------------------- */
    /* --------------------------------------------------------------------- */

    for(int g = 0; g < N_GPUS; g++){
        checkCudaErrors(cudaSetDevice(GPUS_TO_USE[g]));
        cudaDeviceSynchronize();
    }


    //Calculate MLUPS using wall-clock time (works for multi-GPU and virtual multi-GPU)
    auto simEndTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> simElapsed = simEndTime - simStartTime;
    dfloat MLUPS = (static_cast<dfloat>((step - ini_step) * NUMBER_LBM_NODES) / 1e6) / simElapsed.count();
    printf("MLUPS: %f  (steps=%d, nodes=%zu, time=%.3fs)\n", MLUPS, step - ini_step, NUMBER_LBM_NODES, simElapsed.count());
    if(console_flush) fflush(stdout);
    
    /* ------------------------------ POST ------------------------------ */
    finishHostSnapshotTask();
    std::lock_guard<std::mutex> finalSnapshotLock(hostSnapshotMutex);
    deviceField.copyMacroscopicDeviceField(hostField, devices.data(), slice);

    {
        const unsigned char* checksumBytes =
            reinterpret_cast<const unsigned char*>(hostField.h_fMom);
        unsigned long long checksum = 1469598103934665603ULL;
        for (size_t byte = 0; byte < MEM_SIZE_MOM; ++byte) {
            checksum ^= static_cast<unsigned long long>(checksumBytes[byte]);
            checksum *= 1099511628211ULL;
        }
        printf("MOMENT_CHECKSUM: %016llx\n", checksum);
    }

    hostField.saveMacrHostField(step, savingMacrVtk, savingMacrBin, false);

    if(CHECKPOINT_SAVE){
        printf("\n--------------------------- Saving checkpoint %06d ---------------------------\n", step);
        for(int g = 0; g < N_GPUS; g++){
            threads.emplace_back([&, g, slice]() {
                checkCudaErrors(cudaSetDevice(GPUS_TO_USE[g]));
                devices[g].interfaceCudaMemcpyDeviceField(false, g); 
                devices[g].saveSimCheckpointHostDeviceField(hostField, step, g, slice);
                if(console_flush){fflush(stdout);}
            });
        }

        for (auto &t : threads) {
            t.join();
        }                  
        threads.clear();

        
    }

    for(int g = 0; g < N_GPUS; g++){
        checkCudaErrors(cudaSetDevice(GPUS_TO_USE[g]));
        cudaDeviceSynchronize();
    }

    #if MEAN_FLOW
            hostField.saveMacrHostField(INT_MAX, savingMacrVtk, savingMacrBin, true);
    #endif //MEAN_FLOW
    
    //Save info file (always saved, regardless of fluid model)
    #if defined(NON_NEWTONIAN_FLUID) || defined(CONFORMATION_TENSOR)
        #ifdef PHI_DIST
        saveSimInfo(step, MLUPS, deviceField.phasePropsA, deviceField.phasePropsB, true);
        #else
        saveSimInfo(step, MLUPS, deviceField.phasePropsA);
        #endif
    #else
        saveSimInfo(step, MLUPS, fluidPhaseProps{});
    #endif

    while (savingMacrVtk) std::this_thread::sleep_for(std::chrono::milliseconds(1));
    #ifdef PARTICLE_MODEL
    particleField.waitForSaving(savingMacrParticle);
    #endif
    for (size_t i = 0; i < savingMacrBin.size(); ++i) {
        while (savingMacrBin[i]) std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    /* ------------------------------ FREE ------------------------------ */

    hostField.freeHostField();
    for(int g = 0; g < N_GPUS; g++){
        threads.emplace_back([&, g]() {
            checkCudaErrors(cudaSetDevice(GPUS_TO_USE[g]));
            #ifdef OVERLAP_HALO_TRANSFERS
            checkCudaErrors(cudaEventDestroy(haloReady[g]));
            checkCudaErrors(cudaEventDestroy(boundaryReady[g]));
            checkCudaErrors(cudaEventDestroy(stepReady[g]));
            checkCudaErrors(cudaStreamDestroy(streamsHalo[g]));
            checkCudaErrors(cudaStreamDestroy(streamsBoundary[g]));
            #endif
            checkCudaErrors(cudaStreamDestroy(streamsLBM[g]));
            devices[g].freeDeviceField(g);
        });
    }

    for (auto &t : threads) {
        t.join();
    }

    threads.clear();

    // Free particle field
    #ifdef PARTICLE_MODEL
        checkCudaErrors(cudaEventDestroy(particleForceFieldReady));
        particleField.freeMemory();
        particleField.destroyStreams();
    #endif //PARTICLE_MODEL

    return 0;
}
