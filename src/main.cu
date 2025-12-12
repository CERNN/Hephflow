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
    dim3 gridBlock(NUM_BLOCK_X, NUM_BLOCK_Y, NUM_BLOCK_Z);

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
   
    #ifdef PARTICLE_MODEL
    cudaStream_t streamsPart[N_GPUS];
    #endif //PARTICLE_MODEL

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

    threads.reserve(N_GPUS);
    for(int g = 0; g < N_GPUS; g++){

        int gpu = GPUS_TO_USE[g];
        threads.emplace_back([&, gpu, g]() {
            checkCudaErrors(cudaSetDevice(gpu));

            /* -------------- ALLOCATION FOR GPU ------------- */
            devices[gpu].allocateDeviceMemoryDeviceField(g);

            //TODO : move these malocs to inside teh corresponding mallocs
            #ifdef DENSITY_CORRECTION
                cudaMalloc((void**)&deviceField.d_mean_rho[g], sizeof(dfloat));  
            #endif //DENSITY_CORRECTION
            checkCudaErrors(cudaSetDevice(gpu));
            checkCudaErrors(cudaStreamCreate(&streamsLBM[g]));
            checkCudaErrors(cudaDeviceSynchronize());

            #ifdef PARTICLE_MODEL
            checkCudaErrors(cudaStreamCreate(&streamsPart[g]));
            #endif //PARTICLE_MODEL

            /* -------------- Initialize domain in the device ------------- */
            // threads.emplace_back([&devices, &hostField, &randomNumbers, &step, gridBlock, threadBlock, gpu, g]() {
            //     cudaSetDevice(gpu);
            devices[gpu].initializeDomainDeviceField(hostField, randomNumbers, step, gridBlock, threadBlock, g);
            
            checkCudaErrors(cudaDeviceSynchronize());
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                printf("Erro no device %d após init: %s\n", gpu, cudaGetErrorString(err));
            }
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
    checkCudaErrors(cudaSetDevice(GPUS_TO_USE[0]));
    cudaEvent_t start, stop, start_step, stop_step;
    initializeCudaEvents(start, stop, start_step, stop_step);
    
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
        // printf("Laço step\n");

        SaveField saveField;

        saveField.flagsUpdate(step);

        for(int g = 0; g < N_GPUS; g++){
            int gpu = GPUS_TO_USE[g];
            threads.emplace_back([&, gpu, g]() {
                checkCudaErrors(cudaSetDevice(gpu));

                // ghost interface should be inside the deviceField struct
                devices[gpu].gpuMomCollisionStreamDeviceField(gridBlock, threadBlock, step, saveField.save, g);
                cudaError_t err = cudaGetLastError();
                if (err != cudaSuccess) {
                    printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
                }

                #ifdef DENSITY_CORRECTION
                    devices[gpu].mean_rhoDeviceField(step, g)
                #endif //DENSITY_CORRECTION

                #ifdef CURVED_BOUNDARY_CONDITION
                    devices[gpu].updateCurvedBoundaryVelocitiesDeviceField(numberCurvedBoundaryNodes, g);
                #endif

                //swap interface pointers
                devices[gpu].swapGhostInterfacesDeviceField(g);
                
                #ifdef LOCAL_FORCES
                    devices[gpu].gpuResetMacroForcesDeviceField(gridBlock, threadBlock, g);
                #endif //LOCAL_FORCES

                #ifdef PARTICLE_MODEL
                    deviceField.particleSimulationDeviceField(particlesSoA,streamsPart,step);
                #endif //PARTICLE_MODEL
                
                // printf("Saiu Laço gpu denttro laço step\n");
                if(saveField.checkpoint){
                    printf("\n--------------------------- Saving checkpoint %06d ---------------------------\n", step);if(console_flush){fflush(stdout);}
                    // throwing a warning for being used without being initialized. But does not matter since we are overwriting it;
                    devices[gpu].cudaMemcpyDeviceField(hostField, g);
                    devices[gpu].interfaceCudaMemcpyDeviceField(true, g);       
                    devices[gpu].saveSimCheckpointHostDeviceField(hostField, step, g);
                    
                    #ifdef PARTICLE_MODEL
                        printf("Starting saveSimCheckpointParticle...\t"); fflush(stdout);
                        saveSimCheckpointParticle(particlesSoA, &step);
                    #endif //PARTICLE_MODEL
                    
                }
            
                // Saving data checks
                if(saveField.reportSave){
                    printf("\n--------------------------- Saving report %06d ---------------------------\n", step);if(console_flush){fflush(stdout);}
                    devices[gpu].treatDataDeviceField(hostField, step, g);
                }
                    checkCudaErrors(cudaDeviceSynchronize());
                    });
                }

                for (auto &t : threads) {
                    t.join();
                }
                
                threads.clear();
                
                if(saveField.macrSave){
                    for(int g = 0; g < N_GPUS; g++){
                        int gpu = GPUS_TO_USE[g];
                        threads.emplace_back([&, gpu, g]() {
                        #if defined BC_FORCES && defined SAVE_BC_FORCES
                            devices[gpu].saveBcForces(hostField, g);
                        #endif //BC_FORCES && SAVE_BC_FORCES

                        checkCudaErrors(cudaDeviceSynchronize()); 
                    
                        checkCudaErrors(cudaSetDevice(gpu));
                        devices[gpu].cudaMemcpyDeviceField(hostField, g);
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
                            devices[gpu].totalBcDragDeviceField(step, g);
                        #endif //BC_FORCES
                        checkCudaErrors(cudaDeviceSynchronize());
                        });
                    }
        
                    for (auto &t : threads) {
                        t.join();
                    }
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
    

    /* --------------------------------------------------------------------- */
    /* ------------------------------ END LOOP ----------------------------- */
    /* --------------------------------------------------------------------- */

    checkCudaErrors(cudaDeviceSynchronize());

    //Calculate MLUPS

    dfloat MLUPS = recordElapsedTime(start_step, stop_step, step, ini_step);
    printf("MLUPS: %f\n",MLUPS);
    
    /* ------------------------------ POST ------------------------------ */
    for(int g = 0; g < N_GPUS; g++){
        int gpu = GPUS_TO_USE[g];
        threads.emplace_back([&, gpu, g]() {
            checkCudaErrors(cudaSetDevice(gpu));
            devices[gpu].cudaMemcpyDeviceField(hostField, g);

            #if defined BC_FORCES && defined SAVE_BC_FORCES
            devices[gpu].saveBcForces(hostField, g);
            #endif //BC_FORCES && SAVE_BC_FORCES
        checkCudaErrors(cudaDeviceSynchronize());
        });
    }

    for (auto &t : threads) {
        t.join();
    }

    threads.clear();

            if(console_flush){fflush(stdout);}
            hostField.saveMacrHostField(step, savingMacrVtk, savingMacrBin, false);
            for(int g = 0; g < N_GPUS; g++){
                int gpu = GPUS_TO_USE[g];
                threads.emplace_back([&, gpu, g]() {
                    checkCudaErrors(cudaSetDevice(gpu));
                    if(CHECKPOINT_SAVE){
                        printf("\n--------------------------- Saving checkpoint %06d ---------------------------\n", step);if(console_flush){fflush(stdout);}
                        devices[gpu].cudaMemcpyDeviceField(hostField, g);
                        devices[gpu].interfaceCudaMemcpyDeviceField(false, g); 
                        devices[gpu].saveSimCheckpointDeviceField(step, g);
                    }
                    checkCudaErrors(cudaDeviceSynchronize());
                });
            }

            for (auto &t : threads) {
                t.join();
            }

            threads.clear();

    checkCudaErrors(cudaDeviceSynchronize());
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