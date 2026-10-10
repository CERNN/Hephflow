#include "main.cuh"

#if defined(OVERLAP_HALO_TRANSFERS) && !defined(SPLIT_Z_COLLISION_KERNEL)
#error "OVERLAP_HALO_TRANSFERS requires SPLIT_Z_COLLISION_KERNEL"
#endif // defined(OVERLAP_HALO_TRANSFERS) && !defined(SPLIT_Z_COLLISION_KERNEL)

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

    const DomainDecomposition decomposition(N_GPUS);

    threads.reserve(N_GPUS);

    /* ----------------- GRID AND THREADS DEFINITION FOR LBM ---------------- */
    dim3 threadBlock(BLOCK_NX, BLOCK_NY, BLOCK_NZ);
    // Kept for the single-device particle API; fluid launches use each device layout.
    dim3 gridBlock = decomposition[0].fullGrid;

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
    #endif // DENSITY_CORRECTION

    // Macro saving, checkpointing and treat-data share h_fMom.  All three take
    // hostSnapshotMutex while reading it, and every GPU-to-host copy takes the
    // same mutex while overwriting it.  This keeps one full moment snapshot in
    // host RAM instead of one per consumer. When multiple outputs trigger on
    // the same step, one combined worker lets all three reuse one device copy.
    // This is not total process RSS: CUDA runtime state, particle containers,
    // and transient serialization buffers are outside this accounting.
    size_t peakPinnedHostFieldBytes = hostField.getPinnedBytesAllocated();
    printf("Peak persistent pinned host-field memory: %.2f MiB (%.3f GiB)\n",
        static_cast<double>(peakPinnedHostFieldBytes) / (1024.0 * 1024.0), static_cast<double>(peakPinnedHostFieldBytes) / (1024.0 * 1024.0 * 1024.0));
    if(console_flush) fflush(stdout);
    
    /* -------------- ALLOCATION FOR GPUs ------------- */
    cudaStream_t streamsLBM[N_GPUS];
    #ifdef PARTICLE_MODEL
        // Particle coupling currently owns one single-GPU field/stream.
        ParticleField particleField;
        cudaEvent_t particleForceFieldReady;
    #endif // PARTICLE_MODEL
    #ifdef OVERLAP_HALO_TRANSFERS
        cudaStream_t streamsBoundary[N_GPUS];
        cudaStream_t streamsHalo[N_GPUS];
        cudaEvent_t stepReady[N_GPUS];
        cudaEvent_t boundaryReady[N_GPUS];
        cudaEvent_t haloReady[N_GPUS];
    #endif // OVERLAP_HALO_TRANSFERS
    threads.reserve(N_GPUS);
    for(int g = 0; g < N_GPUS; g++){

        threads.emplace_back([&, g]() {
            checkCudaErrors(cudaSetDevice(GPUS_TO_USE[g]));
            devices[g].allocateDeviceMemoryDeviceField(g, decomposition[g]);
    
            #if defined(NON_NEWTONIAN_FLUID) || defined(CONFORMATION_TENSOR)
            #if defined(CASE_PHASE_PROPS_PHASE1)
                    devices[g].phasePropsA = CASE_PHASE_PROPS_PHASE1;
            #elif defined(CASE_PHASE_PROPS)
                    devices[g].phasePropsA = CASE_PHASE_PROPS;
            #endif // defined(CASE_PHASE_PROPS_PHASE1)
            #ifdef PHI_DIST
            #if defined(CASE_PHASE_PROPS_PHASE2)
                        devices[g].phasePropsB = CASE_PHASE_PROPS_PHASE2;
            #elif defined(CASE_PHASE_PROPS)
                        devices[g].phasePropsB = CASE_PHASE_PROPS;
            #endif // defined(CASE_PHASE_PROPS_PHASE2)
            #endif // PHI_DIST
            #endif //NON_NEWTONIAN_FLUID || CONFORMATION_TENSOR
            #ifdef CONFORMATION_TENSOR
                if (!validateVeFluidProps(devices[g].phasePropsA.ve)
            #ifdef PHI_DIST
                    || !validateVeFluidProps(devices[g].phasePropsB.ve)
                    || (devices[g].phasePropsA.ve.type != VE_NEWTONIAN &&
                    devices[g].phasePropsB.ve.type != VE_NEWTONIAN &&
                    devices[g].phasePropsA.ve.type != devices[g].phasePropsB.ve.type)
            #endif // PHI_DIST
                ) {
                    fprintf(stderr, "Invalid or unsupported viscoelastic phase properties\n");
                    abort();
                    }
            #endif // CONFORMATION_TENSOR
            
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
            #endif // OVERLAP_HALO_TRANSFERS
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
        checkCudaErrors(cudaEventCreateWithFlags( &particleForceFieldReady, cudaEventDisableTiming));
    #endif // PARTICLE_MODEL

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
    std::vector<std::exception_ptr> initializationErrors(N_GPUS);
    for(int g = 0; g < N_GPUS; g++){

        threads.emplace_back([&, g]() {
            try {
            checkCudaErrors(cudaSetDevice(GPUS_TO_USE[g]));
            devices[g].initializeDomainDeviceField(hostField, randomNumbers,
                initializedSteps[g], threadBlock, g);
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                printf("Erro no device %d apÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â³s init: %s\n", GPUS_TO_USE[g], cudaGetErrorString(err));
            }
            checkCudaErrors(cudaDeviceSynchronize());
            } catch (...) {
                initializationErrors[g] = std::current_exception();
            }
        });
    }

    for (auto &t : threads) {
        t.join();
    }

    threads.clear();

    for (int g = 0; g < N_GPUS; ++g) {
        if (!initializationErrors[g]) continue;
        try {
            std::rethrow_exception(initializationErrors[g]);
        } catch (const std::exception& error) {
            fprintf(stderr, "Domain initialization failed on partition %d: %s\n",
                g, error.what());
        } catch (...) {
            fprintf(stderr, "Domain initialization failed on partition %d\n", g);
        }
        return EXIT_FAILURE;
    }

    step = initializedSteps[0];
    for (int g = 1; g < N_GPUS; ++g) {
        if (initializedSteps[g] != step) {
            fprintf(stderr,
                "Checkpoint step mismatch: partition 0 resumes at %d, partition %d at %d\n",
                step, g, initializedSteps[g]);
            return EXIT_FAILURE;
        }
    }
    bool stepParity = devices[0].effectiveStreamingParity(step);

    #ifdef RANDOM_NUMBERS
        for (int g = 0; g < N_GPUS; g++) {
            checkCudaErrors(cudaSetDevice(GPUS_TO_USE[g]));
            checkCudaErrors(cudaFree(randomNumbers[g]));
            randomNumbers[g] = nullptr;
            }
    #endif // RANDOM_NUMBERS
    free(randomNumbers);
    randomNumbers = nullptr;

    // A host-wide barrier is needed before reading another GPU's send buffers.
    auto synchronizeLBMStreams = [&]() {
        for (int g = 0; g < N_GPUS; ++g) {
            checkCudaErrors(cudaSetDevice(GPUS_TO_USE[g]));
            checkCudaErrors(cudaStreamSynchronize(streamsLBM[g]));
        }
    };

    #ifdef DENSITY_CORRECTION
        bool densityHistoryStarted = false;
        auto updateDensityCorrection = [&](size_t densityStep) {
            for (int g = 0; g < N_GPUS; ++g) {
                const size_t localNZ = decomposition[g].localNZ;
                checkCudaErrors(cudaSetDevice(GPUS_TO_USE[g]));
                devices[g].reduceDensityPartialDeviceField(
                g, localNZ,
                &hostField.h_density_partial_sum[g],
                &hostField.h_density_partial_count[g], streamsLBM[g]);
                }

            // The host is the global reduction coordinator. Each stream only
            // synchronizes after its asynchronous local reduction/copy is queued.
            synchronizeLBMStreams();

            dfloat globalSum = 0.0_df;
            unsigned long long globalCount = 0;
            for (int g = 0; g < N_GPUS; ++g) {
                globalSum += hostField.h_density_partial_sum[g];
                globalCount += hostField.h_density_partial_count[g];
                }
            if (globalCount == 0) {
                throw std::runtime_error( "Density correction found no physical active fluid nodes");
                }

            hostField.h_mean_rho[0] =
            globalSum / static_cast<dfloat>(globalCount);
            for (int g = 0; g < N_GPUS; ++g) {
                checkCudaErrors(cudaSetDevice(GPUS_TO_USE[g]));
                checkCudaErrors(cudaMemcpyAsync(
                devices[g].d_mean_rho[g], hostField.h_mean_rho,
                sizeof(dfloat), cudaMemcpyHostToDevice, streamsLBM[g]));
                }

            std::ostringstream history;
            history << std::scientific << std::setprecision(9)
            << "step," << densityStep
            << ",mean_rho," << (RHO_0 + hostField.h_mean_rho[0])
            << ",correction," << hostField.h_mean_rho[0]
            << ",active_nodes," << globalCount;
            saveTreatData( "_densityHistory", history.str(), static_cast<int>(densityStep), densityHistoryStarted);
            densityHistoryStarted = true;
            };

        // Establish a valid correction from the initialized/restored field before
        // the first collision reads d_mean_rho.
        updateDensityCorrection(static_cast<size_t>(step));
    #endif // DENSITY_CORRECTION

    auto publishMacroHalos = [&]() {
        for (int g = 0; g < N_GPUS; ++g)
            devices[g].packMacroHalosDeviceField(g, streamsLBM[g]);
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
            devices[g].computePhaseNormalsDeviceField(threadBlock,
                g, streamsLBM[g], stepParity);
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
        #endif // PHASE_NEEDS_NORMAL_HALOS
        #ifdef PHASE_TRANSPORT_CAHN_HILLIARD
                for (int g = 0; g < N_GPUS; ++g)
                devices[g].exchangeMacroField(g, devices.data(),
                &macroInterfaceGPUData::mu, streamsLBM[g], stepParity);
        #endif // PHASE_TRANSPORT_CAHN_HILLIARD
            synchronizeLBMStreams();

            for (int g = 0; g < N_GPUS; ++g)
            devices[g].computePhaseAuxiliariesDeviceField(threadBlock,
                g, streamsLBM[g], stepParity);
            synchronizeLBMStreams();
            CHECK_KERNEL_ERR("Phase derivatives kernels");
            };

        // Also rebuild derived phase fields from initialized or restored moments.
        updatePhaseDerivatives();
    #else
        if constexpr (MACRO_HALOS_REQUIRED) {
            publishMacroHalos();
            }
    #endif // PHI_DIST

    #ifdef CONFORMATION_PHYSICAL_A_HALO
        // AB time-level contract: collision reads generation n, transport produces n+1, and the post-kernel
        // reconstruction below writes the polymer-stress halo for generation n+1. Seed generation n here from
        // initialized or checkpoint-restored moments; this derived halo is never checkpointed.
        for (int g = 0; g < N_GPUS; ++g)
        devices[g].fillPolymerStressHaloDeviceField(decomposition[g].fullGrid, threadBlock, stepParity, g, streamsLBM[g]);
        synchronizeLBMStreams();
        for (int g = 0; g < N_GPUS; ++g)
        devices[g].exchangePolymerStressHalosDeviceField(
            g, devices.data(), stepParity, streamsLBM[g]);
        synchronizeLBMStreams();
    #endif // CONFORMATION_PHYSICAL_A_HALO

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
    #endif // OVERLAP_HALO_TRANSFERS

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

        stepParity = devices[0].effectiveStreamingParity(step);

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
                devices[g].gpuMomCollisionStreamDeviceField(threadBlock, step,
                saveField.save, g, streamsBoundary[g], ZKernelLaunchRegion::Boundaries);
                checkCudaErrors(cudaEventRecord(boundaryReady[g], streamsBoundary[g]));
                }

            // Submit the long interior launch only after every boundary launch has
            // been submitted, so priority does not have to preempt resident blocks.
            for (int g = 0; g < N_GPUS; ++g) {
                checkCudaErrors(cudaSetDevice(GPUS_TO_USE[g]));
                devices[g].gpuMomCollisionStreamDeviceField(threadBlock, step,
                saveField.save, g, streamsLBM[g], ZKernelLaunchRegion::Interior);
                }

            // The just-produced faces belong to the next AA generation. Copies can
            // overlap the current interior, and are omitted after the final step.
            if (step + 1 < N_STEPS) {
                const bool nextStepParity = devices[0].effectiveStreamingParity(step + 1);
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
                // Direct call zero thread overhead for single-GPU builds
                checkCudaErrors(cudaSetDevice(GPUS_TO_USE[0]));
                devices[0].gpuMomCollisionStreamDeviceField(threadBlock, step, saveField.save, 0, streamsLBM[0]);
                } else {
                for(int g = 0; g < N_GPUS; g++){
                    threads.emplace_back([&, g]() {
                        checkCudaErrors(cudaSetDevice(GPUS_TO_USE[g]));
                        devices[g].gpuMomCollisionStreamDeviceField(threadBlock, step, saveField.save, g, streamsLBM[g]);
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
        #endif // STREAMING_AB
            
            CHECK_KERNEL_ERR("Stream Collision kernel");
        #endif // OVERLAP_HALO_TRANSFERS

        //------------------------- Auxiliary Kernels -------------------------
        #ifdef PHI_DIST
            updatePhaseDerivatives();
        #endif // PHI_DIST

        #ifdef DENSITY_CORRECTION
            // This global value corrects the field state produced by the collision
            // above and is consumed by every partition on the next collision.
            updateDensityCorrection(static_cast<size_t>(step) + 1);
            CHECK_KERNEL_ERR("Density correction reduction");
        #endif // DENSITY_CORRECTION

        if (N_GPUS == 1) {
            devices[0].halfStepKernels(threadBlock, step, 0, streamsLBM[0]);
            #ifdef PARTICLE_MODEL
                checkCudaErrors(cudaEventRecord(
                    particleForceFieldReady, streamsLBM[0]));
                checkCudaErrors(cudaStreamWaitEvent(
                    particleField.getStream()[0], particleForceFieldReady, 0));
                particleField.simulationStep(devices[0].d_fMom[0], step);
            #endif //PARTICLE_MODEL
        } else {
            for(int g = 0; g < N_GPUS; g++){
                threads.emplace_back([&, g]() {
                    checkCudaErrors(cudaSetDevice(GPUS_TO_USE[g]));
                    devices[g].halfStepKernels(threadBlock, step, g, streamsLBM[g]);
                    #ifdef PARTICLE_MODEL
                        particleField.simulationStep(devices[g].d_fMom[g], step);
                    #endif //PARTICLE_MODEL
                });
            }
            for (auto &t : threads) { t.join(); }
            threads.clear();
        }

        #ifdef CONFORMATION_PHYSICAL_A_HALO
            // All conformation-mutating work for this step is ordered before this reconstruction on the same stream.
            // The next collision reads the alternate polymer-stress halo generation corresponding to A^(n+1).
            for (int g = 0; g < N_GPUS; ++g)
            devices[g].fillPolymerStressHaloDeviceField(decomposition[g].fullGrid, threadBlock,
                devices[g].effectiveStreamingParity(step + 1u), g, streamsLBM[g]);
            synchronizeLBMStreams();
            for (int g = 0; g < N_GPUS; ++g)
            devices[g].exchangePolymerStressHalosDeviceField(
                g, devices.data(), devices[g].effectiveStreamingParity(step + 1u),
                streamsLBM[g]);
            synchronizeLBMStreams();
        #endif // CONFORMATION_PHYSICAL_A_HALO

        #ifdef OVERLAP_HALO_TRANSFERS
            for (int g = 0; g < N_GPUS; ++g) {
                checkCudaErrors(cudaSetDevice(GPUS_TO_USE[g]));
                checkCudaErrors(cudaEventRecord(stepReady[g], streamsLBM[g]));
                }
        #endif // OVERLAP_HALO_TRANSFERS

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
            deviceField.copyMacroscopicDeviceField(hostField, devices.data());

            // Capture decomposition-independent logical populations before
            // GPU simulation resumes; disk serialization happens in the worker.
            std::shared_ptr<CanonicalCheckpointHostState> checkpointPopulations;
            const int checkpointStep = step + 1;
            if (checkpointRequested) {
                const bool checkpointParity =
                    devices[0].effectiveStreamingParity(checkpointStep);
                for (int g = 0; g < N_GPUS; ++g) {
                    devices[g].sendTopToNext(
                        g, devices.data(), streamsLBM[g], checkpointParity);
                    devices[g].sendBottomToPrev(
                        g, devices.data(), streamsLBM[g], checkpointParity);
                }
                for (int g = 0; g < N_GPUS; ++g) {
                    checkCudaErrors(cudaSetDevice(GPUS_TO_USE[g]));
                    checkCudaErrors(cudaDeviceSynchronize());
                }
                checkpointPopulations = std::make_shared<CanonicalCheckpointHostState>();
                checkpointPopulations->resizeForGlobalDomain();
                for (int g = 0; g < N_GPUS; ++g) {
                    devices[g].captureCanonicalCheckpointDeviceField(
                        *checkpointPopulations, threadBlock, checkpointStep, g);
                }
            }

            const int snapshotStep = step;
            std::promise<void> snapshotOwnedPromise;
            std::future<void> snapshotOwned = snapshotOwnedPromise.get_future();
            hostSnapshotFuture = std::async(std::launch::async,
                [&, snapshotStep, checkpointStep, checkpointRequested,
                    reportRequested, macroRequested, checkpointPopulations,
                    owned = std::move(snapshotOwnedPromise)]() mutable {
                    std::lock_guard<std::mutex> workerLock(hostSnapshotMutex);

                #ifdef PARTICLE_MODEL
                    // Particle state is device-resident, so capture it before
                    // telling the main thread that GPU simulation may resume.
                    if (checkpointRequested) {
                        particleField.saveCheckpoint(snapshotStep);
                        }
                #endif // PARTICLE_MODEL

                    // Ownership is now established and every device-resident
                    // checkpoint component has been captured.
                    owned.set_value();

                    if (checkpointRequested) {
                        saveGlobalSimCheckpoint(hostField.h_fMom, *checkpointPopulations, checkpointStep);
                        if(console_flush) fflush(stdout);
                    }

                    if (macroRequested && !ONLY_FINAL_MACRO) {
                        hostField.saveMacrHostField(snapshotStep, savingMacrVtk, savingMacrBin, false);
                    }

                    if (reportRequested) {
                        deviceField.treatDataDeviceField( hostField, snapshotStep);
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
        #endif // PARTICLE_MODEL
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
    const size_t physicalNodeCount = static_cast<size_t>(NX) * NY * NZ;
    dfloat MLUPS = (static_cast<dfloat>((step - ini_step) * physicalNodeCount) / 1e6) / simElapsed.count();
    printf("MLUPS: %f  (steps=%d, nodes=%zu, time=%.3fs)\n", MLUPS, step - ini_step, physicalNodeCount, simElapsed.count());
    if(console_flush) fflush(stdout);
    
    /* ------------------------------ POST ------------------------------ */
    finishHostSnapshotTask();
    std::lock_guard<std::mutex> finalSnapshotLock(hostSnapshotMutex);
    deviceField.copyMacroscopicDeviceField(hostField, devices.data());

    {
        unsigned long long checksum = 1469598103934665603ULL;
        for (size_t z = 0; z < static_cast<size_t>(NZ); ++z)
            for (size_t y = 0; y < static_cast<size_t>(NY); ++y)
                for (size_t x = 0; x < static_cast<size_t>(NX); ++x)
                    for (int moment = 0; moment < NUMBER_MOMENTS; ++moment) {
                        const size_t index = idxMom(x % BLOCK_NX, y % BLOCK_NY, z % BLOCK_NZ, moment, x / BLOCK_NX, y / BLOCK_NY, z / BLOCK_NZ);
                        const unsigned char* valueBytes = reinterpret_cast<const unsigned char*>(hostField.h_fMom + index);
                        for (size_t byte = 0; byte < sizeof(dfloat); ++byte) {
                            checksum ^= static_cast<unsigned long long>(valueBytes[byte]);
                            checksum *= 1099511628211ULL;
                        }
                    }
        printf("MOMENT_CHECKSUM: %016llx\n", checksum);

    }

    hostField.saveMacrHostField(step, savingMacrVtk, savingMacrBin, false);

    if(CHECKPOINT_SAVE){
        printf("\n--------------------------- Saving checkpoint %06d ---------------------------\n", step);
        CanonicalCheckpointHostState checkpointPopulations;
        checkpointPopulations.resizeForGlobalDomain();
        const bool checkpointParity = devices[0].effectiveStreamingParity(step);
        for (int g = 0; g < N_GPUS; ++g) {
            devices[g].sendTopToNext(
                g, devices.data(), streamsLBM[g], checkpointParity);
            devices[g].sendBottomToPrev(
                g, devices.data(), streamsLBM[g], checkpointParity);
        }
        for (int g = 0; g < N_GPUS; ++g) {
            checkCudaErrors(cudaSetDevice(GPUS_TO_USE[g]));
            checkCudaErrors(cudaDeviceSynchronize());
        }
        for(int g = 0; g < N_GPUS; g++){
            devices[g].captureCanonicalCheckpointDeviceField(
                checkpointPopulations, threadBlock, step, g);
        }
        saveGlobalSimCheckpoint( hostField.h_fMom, checkpointPopulations, step);
        if(console_flush){fflush(stdout);}
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
    #endif // PHI_DIST
    #else
        saveSimInfo(step, MLUPS, fluidPhaseProps{});
    #endif // defined(NON_NEWTONIAN_FLUID) || defined(CONFORMATION_TENSOR)

    while (savingMacrVtk) std::this_thread::sleep_for(std::chrono::milliseconds(1));
    #ifdef PARTICLE_MODEL
        particleField.waitForSaving(savingMacrParticle);
    #endif // PARTICLE_MODEL
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
            #endif // OVERLAP_HALO_TRANSFERS
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
