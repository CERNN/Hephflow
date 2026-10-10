#include "simulation.cuh"

#include "include/errorDef.h"

void collideAndExchangeHalos(SimulationContext& context, const SaveField& saveField) {
    auto& devices = context.devices;
    auto& threads = context.threads;
    const int step = context.step;
    const bool stepParity = context.stepParity;

    #ifdef OVERLAP_HALO_TRANSFERS
        // Boundary kernels wait for both the local step dependency and the
        // incoming halo generation. They run on the high-priority stream so
        // the next generation can be transferred while interiors are active.
        for (int g = 0; g < N_GPUS; ++g) {
            checkCudaErrors(cudaSetDevice(GPUS_TO_USE[g]));
            const int previous = (g + N_GPUS - 1) % N_GPUS;
            checkCudaErrors(cudaStreamWaitEvent(context.streamsBoundary[g], context.stepReady[g], 0));
            checkCudaErrors(cudaStreamWaitEvent(context.streamsBoundary[g], context.haloReady[g], 0));
            if (previous != g) {
                checkCudaErrors(cudaStreamWaitEvent(context.streamsBoundary[g], context.haloReady[previous], 0));
            }
            devices[g].gpuMomCollisionStreamDeviceField(context.threadBlock, step,
                saveField.save, g, context.streamsBoundary[g], ZKernelLaunchRegion::Boundaries);
            checkCudaErrors(cudaEventRecord(context.boundaryReady[g], context.streamsBoundary[g]));
        }

        // Submit every boundary before any long interior launch. This avoids
        // relying on priority preemption of already-resident interior blocks.
        for (int g = 0; g < N_GPUS; ++g) {
            checkCudaErrors(cudaSetDevice(GPUS_TO_USE[g]));
            devices[g].gpuMomCollisionStreamDeviceField(context.threadBlock, step,
                saveField.save, g, context.streamsLBM[g], ZKernelLaunchRegion::Interior);
        }

        if (step + 1 < N_STEPS) {
            // Boundary outputs belong to the next streaming generation. A
            // separate halo stream transfers them while the interior runs.
            const bool nextStepParity = devices[0].effectiveStreamingParity(step + 1);
            for (int g = 0; g < N_GPUS; ++g) {
                checkCudaErrors(cudaSetDevice(GPUS_TO_USE[g]));
                const int next = (g + 1) % N_GPUS;
                checkCudaErrors(cudaStreamWaitEvent(context.streamsHalo[g], context.boundaryReady[g], 0));
                if (next != g) {
                    checkCudaErrors(cudaStreamWaitEvent(context.streamsHalo[g], context.boundaryReady[next], 0));
                }
                devices[g].sendTopToNext(g, devices.data(), context.streamsHalo[g], nextStepParity);
                devices[g].sendBottomToPrev(g, devices.data(), context.streamsHalo[g], nextStepParity);
                if constexpr (MACRO_HALOS_REQUIRED) {
                    devices[g].sendMacroTopToNext(g, devices.data(), context.streamsHalo[g], nextStepParity);
                    devices[g].sendMacroBottomToPrev(g, devices.data(), context.streamsHalo[g], nextStepParity);
                }
                checkCudaErrors(cudaEventRecord(context.haloReady[g], context.streamsHalo[g]));
            }
        }

        // Downstream kernels use the complete timestep, so join boundary work
        // into the normal LBM stream at this single dependency point.
        for (int g = 0; g < N_GPUS; ++g) {
            checkCudaErrors(cudaSetDevice(GPUS_TO_USE[g]));
            checkCudaErrors(cudaStreamWaitEvent(context.streamsLBM[g], context.boundaryReady[g], 0));
            checkCudaErrors(cudaGetLastError());
        }
    #else
        // The non-overlapped path publishes both Z faces, waits globally, and
        // then launches the full collision grid on each partition.
        for (int g = 0; g < N_GPUS; ++g) {
            checkCudaErrors(cudaSetDevice(GPUS_TO_USE[g]));
            devices[g].sendTopToNext(g, devices.data(), context.streamsLBM[g], stepParity);
            devices[g].sendBottomToPrev(g, devices.data(), context.streamsLBM[g], stepParity);
            if constexpr (MACRO_HALOS_REQUIRED) {
                devices[g].sendMacroTopToNext(g, devices.data(), context.streamsLBM[g], stepParity);
                devices[g].sendMacroBottomToPrev(g, devices.data(), context.streamsLBM[g], stepParity);
            }
        }

        for (int g = 0; g < N_GPUS; ++g) {
            checkCudaErrors(cudaSetDevice(GPUS_TO_USE[g]));
            checkCudaErrors(cudaDeviceSynchronize());
        }

        if (N_GPUS == 1) {
            // Avoid host-thread creation when one direct launch is sufficient.
            checkCudaErrors(cudaSetDevice(GPUS_TO_USE[0]));
            devices[0].gpuMomCollisionStreamDeviceField(context.threadBlock, step,
                saveField.save, 0, context.streamsLBM[0]);
        } else {
            for (int g = 0; g < N_GPUS; ++g) {
                threads.emplace_back([&context, &saveField, g]() {
                    checkCudaErrors(cudaSetDevice(GPUS_TO_USE[g]));
                    context.devices[g].gpuMomCollisionStreamDeviceField(context.threadBlock,
                        context.step, saveField.save, g, context.streamsLBM[g]);
                });
            }
            for (auto& thread : threads) thread.join();
            threads.clear();
        }

        for (int g = 0; g < N_GPUS; ++g) {
            checkCudaErrors(cudaSetDevice(GPUS_TO_USE[g]));
            checkCudaErrors(cudaDeviceSynchronize());
        }

        #ifdef STREAMING_AB
            // AB collision writes the alternate interface generation. Swap it
            // only after every partition has finished the timestep.
            for (int g = 0; g < N_GPUS; ++g) {
                devices[g].swapGhostInterfacesDeviceField(g);
            }
        #endif // STREAMING_AB

        CHECK_KERNEL_ERR("Stream Collision kernel");
    #endif // OVERLAP_HALO_TRANSFERS
}

void runHalfStepAndParticles(SimulationContext& context) {
    auto& devices = context.devices;

    if (N_GPUS == 1) {
        devices[0].halfStepKernels(context.threadBlock, context.step, 0, context.streamsLBM[0]);
        #ifdef PARTICLE_MODEL
            // The particle stream waits only for the fluid force field event;
            // unrelated queued fluid work remains asynchronous.
            checkCudaErrors(cudaEventRecord(context.particleForceFieldReady, context.streamsLBM[0]));
            checkCudaErrors(cudaStreamWaitEvent(context.particleField.getStream()[0],
                context.particleForceFieldReady, 0));
            context.particleField.simulationStep(devices[0].d_fMom[0], context.step);
        #endif // PARTICLE_MODEL
    } else {
        for (int g = 0; g < N_GPUS; ++g) {
            context.threads.emplace_back([&context, g]() {
                checkCudaErrors(cudaSetDevice(GPUS_TO_USE[g]));
                context.devices[g].halfStepKernels(context.threadBlock, context.step,
                    g, context.streamsLBM[g]);
                #ifdef PARTICLE_MODEL
                    context.particleField.simulationStep(context.devices[g].d_fMom[g], context.step);
                #endif // PARTICLE_MODEL
            });
        }
        for (auto& thread : context.threads) thread.join();
        context.threads.clear();
    }

    #ifdef CONFORMATION_PHYSICAL_A_HALO
        // All conformation mutations for n+1 precede this reconstruction on
        // streamsLBM. Exchange the reconstructed physical-A halo before n+1.
        for (int g = 0; g < N_GPUS; ++g) {
            devices[g].fillPolymerStressHaloDeviceField(context.decomposition[g].fullGrid,
                context.threadBlock, devices[g].effectiveStreamingParity(context.step + 1u),
                g, context.streamsLBM[g]);
        }
        synchronizeSimulationStreams(context);
        for (int g = 0; g < N_GPUS; ++g) {
            devices[g].exchangePolymerStressHalosDeviceField(g, devices.data(),
                devices[g].effectiveStreamingParity(context.step + 1u), context.streamsLBM[g]);
        }
        synchronizeSimulationStreams(context);
    #endif // CONFORMATION_PHYSICAL_A_HALO

    #ifdef OVERLAP_HALO_TRANSFERS
        // This event is the local dependency consumed by the next boundary
        // launch; it marks completion of all current-step LBM-stream work.
        for (int g = 0; g < N_GPUS; ++g) {
            checkCudaErrors(cudaSetDevice(GPUS_TO_USE[g]));
            checkCudaErrors(cudaEventRecord(context.stepReady[g], context.streamsLBM[g]));
        }
    #endif // OVERLAP_HALO_TRANSFERS
}
