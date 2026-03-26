#include "main.cuh"
#include "hostField.cuh"
#include "deviceField.cuh"
#include "saveField.cuh"

using namespace std;

int main() {
    // Setup saving folder
    folderSetup();

    // Field Variables
    HostField hostField;
    DeviceField deviceField;

    /* ----------------- GRID AND THREADS DEFINITION FOR LBM ---------------- */
    dim3 threadBlock(BLOCK_NX, BLOCK_NY, BLOCK_NZ);
    dim3 gridBlock(NUM_BLOCK_X, NUM_BLOCK_Y, NUM_BLOCK_Z_LOCAL);

    /* ------------------------- ALLOCATION FOR CPU ------------------------- */
    int step = 0;

    dfloat** randomNumbers = nullptr;
    randomNumbers = (dfloat**)malloc(sizeof(dfloat*) * N_GPUS);

    //TODO : move these malocs to inside teh corresponding mallocs
    
    #ifdef DENSITY_CORRECTION
        checkCudaErrors(cudaMallocHost((void**)&(hostField.h_mean_rho), sizeof(dfloat)));
    #endif //DENSITY_CORRECTION

    /* -------------- Setup Streams ------------- */
    cudaStream_t streamsLBM[N_GPUS];

    deviceField.enablePeerAccessDeviceField();

    #ifdef PARTICLE_MODEL
    cudaStream_t streamsPart[N_GPUS];
    #endif //PARTICLE_MODEL

    auto start_wall = std::chrono::high_resolution_clock::now();
    step = INI_STEP;

    //Declaration of atomic flags to safely control the state of data saving in multiple threads.
    std::atomic<bool> savingMacrVtk(false);
    std::atomic<bool> savingMacrParticle(false);
    std::vector<std::atomic<bool>> savingMacrBin(hostField.NThread);

    for (int i = 0; i < hostField.NThread; i++){
        savingMacrBin[i].store(false);
    }

    hostField.allocateHostMemoryHostField();
    
    std::vector<std::thread> threads;
    std::vector<DeviceField> devices(N_GPUS);

    int slice = NZ / N_GPUS;

    threads.reserve(N_GPUS);
    for(int g = 0; g < N_GPUS; g++){

        int zStart = g * slice;
        int zEnd   = (g == N_GPUS - 1) ? NZ : zStart + slice;

        size_t localNZ_physical = zEnd - zStart;
        size_t localNZ = localNZ_physical;

        printf("Start initialize %d\n", zStart);
        printf("End initialize %d\n", zEnd);
        printf("localNZ initialize %d\n", localNZ);

        threads.emplace_back([&, g, zStart, zEnd, localNZ]() {
            checkCudaErrors(cudaSetDevice(GPUS_TO_USE[g]));

            /* -------------- ALLOCATION FOR GPU ------------- */
            devices[g].allocateDeviceMemoryDeviceField(g);

            //TODO : move these malocs to inside teh corresponding mallocs
            #ifdef DENSITY_CORRECTION
                cudaMalloc((void**)&devices[g].d_mean_rho[g], sizeof(dfloat));  
            #endif //DENSITY_CORRECTION
            checkCudaErrors(cudaSetDevice(GPUS_TO_USE[g]));
            checkCudaErrors(cudaStreamCreate(&streamsLBM[g]));
            checkCudaErrors(cudaDeviceSynchronize());

            #ifdef PARTICLE_MODEL
            checkCudaErrors(cudaStreamCreate(&streamsPart[g]));
            #endif //PARTICLE_MODEL

            /* -------------- Initialize domain in the device ------------- */
            devices[g].initializeDomainDeviceField(hostField, randomNumbers, step, gridBlock, threadBlock, g, zStart, zEnd, localNZ);
            
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
        //memory allocation for particles in host and device
        ParticlesSoA particlesSoA;
        Particle *particles;
        particles = (Particle*) malloc(sizeof(Particle)*NUM_PARTICLES);
        
        // particle initialization with position, velocity, and solver method
        initializeParticle(particlesSoA, particles, &step, gridBlock, threadBlock);
        while (savingMacrParticle) std::this_thread::yield();
        saveParticlesInfo(&particlesSoA, step, savingMacrParticle);

    #endif //PARTICLE_MODEL

    #ifdef CURVED_BOUNDARY_CONDITION
        //Get number of curved boundary nodes
        unsigned int numberCurvedBoundaryNodes = getNumberCurvedBoundaryNodes(hostField.hNodeType);
    #endif //CURVE

    /* ------------------------------ TIMER EVENTS  ------------------------------ */
    // checkCudaErrors(cudaSetDevice(GPUS_TO_USE[0]));
    // cudaEvent_t start, stop, start_step, stop_step;
    // initializeCudaEvents(start, stop, start_step, stop_step);
    
    /* ------------------------------ LBM LOOP ------------------------------ */

    #ifdef DYNAMIC_SHARED_MEMORY
        int maxShared;
        cudaDeviceGetAttribute(&maxShared, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0);
        if (MAX_SHARED_MEMORY_SIZE > maxShared) {
            printf("Requested %d bytes exceeds device max %d bytes\n", MAX_SHARED_MEMORY_SIZE, maxShared);
        }else{
            printf("Using %d bytes of dynamic shared memory of a max of %d bytes\n", MAX_SHARED_MEMORY_SIZE, maxShared);
            cudaFuncSetAttribute(&gpuMomCollisionStream, cudaFuncAttributeMaxDynamicSharedMemorySize DYNAMIC_SHARED_MEMORY_PARAMS); // DOESNT WORK: DYNAMICALLY SHARED MEMORY HAS WORSE PERFORMANCE
        }
    #endif //DYNAMIC_SHARED_MEMORY
   
    /* --------------------------------------------------------------------- */
    /* ---------------------------- BEGIN LOOP ----------------------------- */
    /* --------------------------------------------------------------------- */

    for (;step<N_STEPS;step++){ // step is already initialized

        SaveField saveField;

        saveField.flagsUpdate(step);
       
        for(int g = 0; g < N_GPUS; g++){

            int zStart = g * slice;
            int zEnd   = (g == N_GPUS - 1) ? NZ : zStart + slice;
            size_t localNZ = zEnd - zStart;
           
            threads.emplace_back([&, g, zStart, zEnd, localNZ]() {

                checkCudaErrors(cudaSetDevice(GPUS_TO_USE[g]));

                // ghost interface should be inside the deviceField struct
                devices[g].gpuMomCollisionStreamDeviceField(gridBlock, threadBlock, step, saveField.save, g, localNZ, zStart, zEnd);
                cudaError_t err = cudaGetLastError();
                if (err != cudaSuccess) {
                    printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
                }
                #ifdef DENSITY_CORRECTION
                    devices[g].mean_rhoDeviceField(step, g)
                #endif //DENSITY_CORRECTION
    
                #ifdef CURVED_BOUNDARY_CONDITION
                    devices[g].updateCurvedBoundaryVelocitiesDeviceField(numberCurvedBoundaryNodes, g);
                #endif            
            });
        }
    
        for (auto &t : threads) {
            t.join();
        }                  
        threads.clear();

        for(int g = 0; g < N_GPUS; g++){
            checkCudaErrors(cudaSetDevice(GPUS_TO_USE[g]));
            checkCudaErrors(cudaDeviceSynchronize());
        }

        for (int g = 0; g < N_GPUS; g++) {
            devices[g].sendTopToNext(g, devices.data(), streamsLBM[g]);
        }
        for (int g = 0; g < N_GPUS; g++) {
            checkCudaErrors(cudaSetDevice(GPUS_TO_USE[g]));
            checkCudaErrors(cudaDeviceSynchronize());
        }
        
        for (int g = 0; g < N_GPUS; g++) {
            devices[g].recvTopFromNext(g, devices.data(), streamsLBM[g]);
        }

        for(int g = 0; g < N_GPUS; g++){
            checkCudaErrors(cudaSetDevice(GPUS_TO_USE[g]));
            checkCudaErrors(cudaDeviceSynchronize());
        }
        
        for(int g = 0; g < N_GPUS; g++){
            devices[g].swapGhostInterfacesDeviceField(g);
        }        
        
        for(int g = 0; g < N_GPUS; g++){

            int zStart = g * slice;
            int zEnd   = (g == N_GPUS - 1) ? NZ : zStart + slice;

            size_t localNZ_physical = zEnd - zStart;
            size_t localNZ = localNZ_physical;

            threads.emplace_back([&, g, zStart, zEnd, localNZ]() {
                #ifdef LOCAL_FORCES
                    devices[g].gpuResetMacroForcesDeviceField(gridBlock, threadBlock, g);
                #endif //LOCAL_FORCES

                #ifdef PARTICLE_MODEL
                    deviceField.particleSimulationDeviceField(particlesSoA,streamsPart,step);
                #endif //PARTICLE_MODEL
                
                if(saveField.checkpoint){
                    printf("\n--------------------------- Saving checkpoint %06d ---------------------------\n", step);if(console_flush){fflush(stdout);}
                    // throwing a warning for being used without being initialized. But does not matter since we are overwriting it;
                    devices[g].cudaMemcpyDeviceField(hostField, g, zStart);
                    devices[g].interfaceCudaMemcpyDeviceField(true, g);       
                    devices[g].saveSimCheckpointHostDeviceField(hostField, step, g);
                    
                    #ifdef PARTICLE_MODEL
                        printf("Starting saveSimCheckpointParticle...\t"); fflush(stdout);
                        saveSimCheckpointParticle(particlesSoA, &step);
                    #endif //PARTICLE_MODEL
                    
                }
            
                // Saving data checks
                if(saveField.reportSave){
                    printf("\n--------------------------- Saving report %06d ---------------------------\n", step);if(console_flush){fflush(stdout);}
                    devices[g].treatDataDeviceField(hostField, step, g);
                }
            });
        }
    
        for (auto &t : threads) {
            t.join();
        }                  
        threads.clear();

        if(saveField.macrSave){
            for(int g = 0; g < N_GPUS; g++){
                int zStart = g * slice;
                int zEnd   = (g == N_GPUS - 1) ? NZ : zStart + slice;
        
                int gpu = GPUS_TO_USE[g];
                threads.emplace_back([&, gpu, g, zStart, zEnd]() {
                #if defined BC_FORCES && defined SAVE_BC_FORCES
                    devices[g].saveBcForces(hostField, g);
                #endif //BC_FORCES && SAVE_BC_FORCES

                checkCudaErrors(cudaDeviceSynchronize()); 
                    
                checkCudaErrors(cudaSetDevice(gpu));
                devices[g].cudaMemcpyDeviceField(hostField, g, zStart);
                checkCudaErrors(cudaDeviceSynchronize());

                });
            }
        
            for (auto &t : threads) {
                t.join();
            }                  
            threads.clear();

            printf("\n--------------------------- Saving macro %06d ---------------------------\n", step); if(console_flush){fflush(stdout);}

            if(!ONLY_FINAL_MACRO){
                hostField.saveMacrHostField(step, savingMacrVtk, savingMacrBin, false);
            }
            for(int g = 0; g < N_GPUS; g++){
                int gpu = GPUS_TO_USE[g];
                threads.emplace_back([&, gpu, g]() {
                    #ifdef BC_FORCES
                        devices[g].totalBcDragDeviceField(step, g);
                    #endif //BC_FORCES
                    
                });
            }
            checkCudaErrors(cudaDeviceSynchronize());
            for (auto &t : threads) {
                t.join();
            }
            threads.clear();
        }

            #ifdef PARTICLE_MODEL
                if (saveField.particleSave){
                    printf("\n------------------------- Saving particles %06d -------------------------\n", step);
                    if(console_flush){fflush(stdout);}
                    while (savingMacrParticle) std::this_thread::yield();
                    saveParticlesInfo(&particlesSoA, step, savingMacrParticle);
                }
            #endif //PARTICLE_MODEL
        
   
    } 
    
    // /* --------------------------------------------------------------------- */
    // /* ------------------------------ END LOOP ----------------------------- */
    // /* --------------------------------------------------------------------- */

    for(int g = 0; g < N_GPUS; g++){
        checkCudaErrors(cudaSetDevice(GPUS_TO_USE[g]));
        cudaDeviceSynchronize();
    }

    //Calculate MLUPS

    // dfloat MLUPS = recordElapsedTime(start_step, stop_step, step, ini_step);
    // printf("MLUPS: %f\n",MLUPS);

    auto end_wall = std::chrono::high_resolution_clock::now();
    double elapsedSeconds = std::chrono::duration<double>(end_wall - start_wall).count();
    dfloat MLUPS = recordElapsedTime(elapsedSeconds, step, ini_step);
    printf("MLUPS: %f\n",MLUPS);

    
    /* ------------------------------ POST ------------------------------ */
    for(int g = 0; g < N_GPUS; g++){
        int gpu = GPUS_TO_USE[g];
        int zStart = g * slice;
        int zEnd   = (g == N_GPUS - 1) ? NZ : zStart + slice;

        threads.emplace_back([&, gpu, g, zStart, zEnd]() {
 
            checkCudaErrors(cudaSetDevice(gpu));
            devices[g].cudaMemcpyDeviceField(hostField, g, zStart);

            #if defined BC_FORCES && defined SAVE_BC_FORCES
            devices[g].saveBcForces(hostField, g);
            #endif //BC_FORCES && SAVE_BC_FORCES
        
        });
    }
   
    for (auto &t : threads) {
        t.join();
    }

    threads.clear();

    for(int g = 0; g < N_GPUS; g++){
        checkCudaErrors(cudaSetDevice(GPUS_TO_USE[g]));
        cudaDeviceSynchronize();
    }

    if(console_flush){fflush(stdout);}
    hostField.saveMacrHostField(step, savingMacrVtk, savingMacrBin, false);
    for(int g = 0; g < N_GPUS; g++){
        int zStart = g * slice;
        int zEnd   = (g == N_GPUS - 1) ? NZ : zStart + slice;
    
        int gpu = GPUS_TO_USE[g];
        threads.emplace_back([&, gpu, g, zStart, zEnd]() {
            checkCudaErrors(cudaSetDevice(gpu));
            if(CHECKPOINT_SAVE){
                printf("\n--------------------------- Saving checkpoint %06d ---------------------------\n", step);if(console_flush){fflush(stdout);}
                devices[g].cudaMemcpyDeviceField(hostField, g, zStart);
                devices[g].interfaceCudaMemcpyDeviceField(false, g); 
                devices[g].saveSimCheckpointDeviceField(step, g);
            }
            checkCudaErrors(cudaDeviceSynchronize());
        });
    }

    for (auto &t : threads) {
         t.join();
    }

    threads.clear();

    for(int g = 0; g < N_GPUS; g++){
        checkCudaErrors(cudaSetDevice(GPUS_TO_USE[g]));
        cudaDeviceSynchronize();
    }

    #if MEAN_FLOW
            hostField.saveMacrHostField(INT_MAX, savingMacrVtk, savingMacrBin, true);
    #endif //MEAN_FLOW
    
    //Save info file
    saveSimInfo(step,MLUPS);

    while (savingMacrVtk) std::this_thread::yield();
    #ifdef PARTICLE_MODEL
    while (savingMacrParticle) std::this_thread::yield();
    #endif
    for (size_t i = 0; i < savingMacrBin.size(); ++i) {
        while (savingMacrBin[i]) std::this_thread::yield();
    }

    /* ------------------------------ FREE ------------------------------ */

    hostField.freeHostField();
    for(int g = 0; g < N_GPUS; g++){
        int gpu = GPUS_TO_USE[g];
        threads.emplace_back([&, gpu, g]() {
            devices[g].freeDeviceField(g);
            checkCudaErrors(cudaDeviceSynchronize());
        });
    }

    for (auto &t : threads) {
        t.join();
    }

    threads.clear();

    // Free particle
    #ifdef PARTICLE_MODEL
        free(particles);
        particlesSoA.freeNodesAndCenters();
    #endif //PARTICLE_MODEL

    return 0;
}