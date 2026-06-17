#include "main.cuh"
#include "hostField.cuh"
#include "deviceField.cuh"
#include "saveField.cuh"
#ifdef PARTICLE_MODEL
#include "particleField.cuh"
#endif //PARTICLE_MODEL

#include <chrono>

using namespace std;

int main() {
    // Setup saving folder
    folderSetup();

    // Set cuda device
    // checkCudaErrors(cudaSetDevice(GPU_INDEX));

    // Field Variables
    HostField hostField;
    DeviceField deviceField;

    // Multi-GPU Variables
    std::vector<std::thread> threads;
    std::vector<DeviceField> devices(N_GPUS);

    int slice = NZ / N_GPUS;

    threads.reserve(N_GPUS);

    /* ----------------- GRID AND THREADS DEFINITION FOR LBM ---------------- */
    dim3 threadBlock(BLOCK_NX, BLOCK_NY, BLOCK_NZ);
    dim3 gridBlock(NUM_BLOCK_X, NUM_BLOCK_Y, NUM_BLOCK_Z_LOCAL);

    // Enable peer-to-peer (P2P) access between GPUs
    deviceField.enablePeerAccessDeviceField();

    /* ------------------------- ALLOCATION FOR CPU ------------------------- */
    int step = 0;

    dfloat** randomNumbers = nullptr;
    randomNumbers = (dfloat**)malloc(sizeof(dfloat*) * N_GPUS);

    hostField.allocateHostMemoryHostField();
    
    /* -------------- ALLOCATION FOR GPUs ------------- */
    cudaStream_t streamsLBM[N_GPUS];
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
            
            #ifdef PARTICLE_MODEL
            // Particle field initialization and allocation
            ParticleField particleField;
            particleField.allocateMemory();
            #endif //PARTICLE_MODEL
            
            #ifdef DENSITY_CORRECTION
                // Allocate density correction memory in both host and device fields
                hostField.allocateDensityCorrectionMemory();
                devices[g].allocateDensityCorrectionMemory();
            #endif //DENSITY_CORRECTION

            /* -------------- Setup Streams ------------- */
            checkCudaErrors(cudaSetDevice(GPUS_TO_USE[g]));
            checkCudaErrors(cudaStreamCreate(&streamsLBM[g]));
            checkCudaErrors(cudaDeviceSynchronize());
            #ifdef PARTICLE_MODEL
            particleField.setupStreams();
            #endif //PARTICLE_MODEL
        });
    }

    for (auto &t : threads) {
        t.join();
    }

    threads.clear();

    step = INI_STEP;


    //Declaration of atomic flags to safely control the state of data saving in multiple threads.
    std::atomic<bool> savingMacrVtk(false);
    std::atomic<bool> savingMacrParticle(false);
    std::vector<std::atomic<bool>> savingMacrBin(hostField.NThread);

    for (int i = 0; i < hostField.NThread; i++){
        savingMacrBin[i].store(false);
    }

    /* -------------- Initialize the domain on the devices ------------- */
    for(int g = 0; g < N_GPUS; g++){

        threads.emplace_back([&, g, slice]() {
            devices[g].initializeDomainDeviceField(hostField, randomNumbers,  step, gridBlock, threadBlock, g, slice);
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

    int ini_step = step;

    printf("Domain Initialized. Starting simulation\n"); if(console_flush) fflush(stdout);
    
    #ifdef PARTICLE_MODEL
        // Initialize particle field with position, velocity, and solver method
        particleField.initialize(&step, gridBlock, threadBlock);
        particleField.saveInfo(step, savingMacrParticle);
    #endif //PARTICLE_MODEL

    /* ------------------------------ TIMER EVENTS  ------------------------------ */
    // checkCudaErrors(cudaSetDevice(GPU_INDEX));
    // cudaEvent_t start, stop, start_step, stop_step;
    // initializeCudaEvents(start, stop, start_step, stop_step);
    
    /* ------------------------------ LBM LOOP ------------------------------ */

    #ifdef DYNAMIC_SHARED_MEMORY
        if (configureDynamicSharedMemory(GPU_INDEX)) return 1;
    #endif //DYNAMIC_SHARED_MEMORY
   
    /* --------------------------------------------------------------------- */
    /* ---------------------------- BEGIN LOOP ----------------------------- */
    /* --------------------------------------------------------------------- */

    for (;step<N_STEPS;step++){ // step is already initialized

        SaveField saveField;

        // update saving flags
        saveField.flagsUpdate(step);

        /* -------------- Exchanging halos between neighboring GPUs using P2P ------------- */
        for (int g = 0; g < N_GPUS; g++) {
            checkCudaErrors(cudaSetDevice(GPUS_TO_USE[g]));
                
            devices[g].sendTopToNext(g, devices.data(), streamsLBM[g]);
            // devices[g].sendBottomToPrev(g, devices.data(), streamsLBM[g]);     
        }
    
        for (int g = 0; g < N_GPUS; g++) {
            checkCudaErrors(cudaSetDevice(GPUS_TO_USE[g]));
            checkCudaErrors(cudaDeviceSynchronize());
        }
    
        for (int g = 0; g < N_GPUS; g++) {
            checkCudaErrors(cudaSetDevice(GPUS_TO_USE[g]));
            devices[g].sendBottomToPrev(g, devices.data(), streamsLBM[g]);
        }
    
        for (int g = 0; g < N_GPUS; g++) {
            checkCudaErrors(cudaSetDevice(GPUS_TO_USE[g]));
            checkCudaErrors(cudaDeviceSynchronize());
        }

        //------------------------- Main LBM Kernels -------------------------
        for(int g = 0; g < N_GPUS; g++){
            threads.emplace_back([&, g, slice]() {
                checkCudaErrors(cudaSetDevice(GPUS_TO_USE[g]));
                devices[g].gpuMomCollisionStreamDeviceField(gridBlock, threadBlock, step, saveField.save, g, slice);
            });
        }
        for (auto &t : threads) {
            t.join();
        }                  
        threads.clear();

       
        for (int g = 0; g < N_GPUS; g++) {
            checkCudaErrors(cudaSetDevice(GPUS_TO_USE[g]));
            checkCudaErrors(cudaDeviceSynchronize());
        }
        

        // swap interface pointers
        //deviceField.swapGhostInterfacesDeviceField();
            
        CHECK_KERNEL_ERR("Stream Collision kernel");

        //------------------------- Auxiliary Kernels -------------------------
        for(int g = 0; g < N_GPUS; g++){
            threads.emplace_back([&, g, slice]() {
                devices[g].halfStepKernels(gridBlock, threadBlock, step);
                #ifdef PARTICLE_MODEL
                    particleField.simulationStep(deviceField.d_fMom, step);
                #endif //PARTICLE_MODEL
            });
        }

        for (auto &t : threads) {
            t.join();
        }                  
        threads.clear();

        //------------------------- Saving Data -------------------------
        // Saving checkpoint     
        if(saveField.checkpoint){
            for(int g = 0; g < N_GPUS; g++){
                threads.emplace_back([&, g, slice]() {
                    checkCudaErrors(cudaSetDevice(GPUS_TO_USE[g]));
                    printf("\n--------------------------- Saving checkpoint %06d ---------------------------\n", step);
                    devices[g].cudaMemcpyDeviceField(hostField, g, slice);
                    devices[g].interfaceCudaMemcpyDeviceField(true, g);       
                    devices[g].saveSimCheckpointHostDeviceField(hostField, step, g);
                    #ifdef PARTICLE_MODEL
                        particleField.saveCheckpoint(step);
                    #endif //PARTICLE_MODEL    
                    if(console_flush){fflush(stdout);} 
                });
            }

            for (auto &t : threads) {
                t.join();
            }                  
            threads.clear();
        }
       
        // Saving treat data  checks
        if(saveField.reportSave){
            for(int g = 0; g < N_GPUS; g++){
                threads.emplace_back([&, g, slice]() {
                    checkCudaErrors(cudaSetDevice(GPUS_TO_USE[g]));
                    printf("\n--------------------------- Saving report %06d ---------------------------\n", step);
                    devices[g].treatDataDeviceField(hostField, step, g);
                    #ifdef PARTICLE_MODEL
                    particleField.exportWallForces(step);
                    #endif
                    if(console_flush){fflush(stdout);}
                });
            }

            for (auto &t : threads) {
                t.join();
            }                  
            threads.clear();
        }

        #ifdef TESTS
            #include CASE_TEST_METRIC
        #endif //TESTS
        
        if(saveField.macrSave){
            //copy data from device to host
            for(int g = 0; g < N_GPUS; g++){
                threads.emplace_back([&, g, slice]() {
                    checkCudaErrors(cudaSetDevice(GPUS_TO_USE[g]));
                    checkCudaErrors(cudaDeviceSynchronize()); 
                    devices[g].cudaMemcpyDeviceField(hostField, g, slice);
                });
            }

            for (auto &t : threads) {
                t.join();
            }                  
            threads.clear();
            printf("\n--------------------------- Saving macro %06d ---------------------------\n", step);
            if(!ONLY_FINAL_MACRO){ hostField.saveMacrHostField(step, savingMacrVtk, savingMacrBin, false);}
            if(console_flush){fflush(stdout);}
        }

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


    //Calculate MLUPS

    // dfloat MLUPS = recordElapsedTime(start_step, stop_step, step, ini_step, NUMBER_LBM_NODES);
    // printf("MLUPS: %f\n",MLUPS); if(console_flush){fflush(stdout);}     
    
    /* ------------------------------ POST ------------------------------ */
    for(int g = 0; g < N_GPUS; g++){

        threads.emplace_back([&, g, slice]() {
            checkCudaErrors(cudaSetDevice(GPUS_TO_USE[g]));
            devices[g].cudaMemcpyDeviceField(hostField, g, slice);
        });
    }

    for (auto &t : threads) {
        t.join();
    }                  
    threads.clear();

    hostField.saveMacrHostField(step, savingMacrVtk, savingMacrBin, false);

    if(CHECKPOINT_SAVE){
        for(int g = 0; g < N_GPUS; g++){
            threads.emplace_back([&, g, slice]() {
            checkCudaErrors(cudaSetDevice(GPUS_TO_USE[g]));
            printf("\n--------------------------- Saving checkpoint %06d ---------------------------\n", step);
            devices[g].cudaMemcpyDeviceField(hostField, g, slice);
            devices[g].interfaceCudaMemcpyDeviceField(false, g); 
            devices[g].saveSimCheckpointDeviceField(step, g);
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
    
    //Save info file
    //TODO: fix this later so it doesnt have defines
    #if defined(NON_NEWTONIAN_FLUID) || defined(CONFORMATION_TENSOR)
    #ifdef PHI_DIST
    saveSimInfo(step, MLUPS, deviceField.phasePropsA, deviceField.phasePropsB, true);
    #else
    saveSimInfo(step, MLUPS, deviceField.phasePropsA);
    #endif
    #else
    // saveSimInfo(step, MLUPS, {});
    #endif //NON_NEWTONIAN_FLUID

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
            devices[g].freeDeviceField(g);
        });
    }

    for (auto &t : threads) {
        t.join();
    }

    threads.clear();

    // Free particle field
    #ifdef PARTICLE_MODEL
        particleField.freeMemory();
        particleField.destroyStreams();
    #endif //PARTICLE_MODEL

    return 0;
}