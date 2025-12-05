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
        
    for(int g = 0; g < N_GPUS; g++){
        checkCudaErrors(cudaSetDevice(GPUS_TO_USE[g]));

        /* -------------- ALLOCATION FOR GPU ------------- */
        deviceField.allocateDeviceMemoryDeviceField(g);

        //TODO : move these malocs to inside teh corresponding mallocs
        #ifdef DENSITY_CORRECTION
            cudaMalloc((void**)&deviceField.d_mean_rho[g], sizeof(dfloat));  
        #endif //DENSITY_CORRECTION
        checkCudaErrors(cudaSetDevice(GPUS_TO_USE[g]));
        checkCudaErrors(cudaStreamCreate(&streamsLBM[g]));
        checkCudaErrors(cudaDeviceSynchronize());

        #ifdef PARTICLE_MODEL
        checkCudaErrors(cudaStreamCreate(&streamsPart[g]));
        #endif //PARTICLE_MODEL

        /* -------------- Initialize domain in the device ------------- */
        deviceField.initializeDomainDeviceField(hostField, randomNumbers,  step, gridBlock, threadBlock, g);
    }
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

        SaveField saveField;

        #ifdef DENSITY_CORRECTION
        deviceField.mean_rhoDeviceField(step)
        #endif //DENSITY_CORRECTION

        saveField.flagsUpdate(step);

        // ghost interface should be inside the deviceField struct
        deviceField.gpuMomCollisionStreamDeviceField(gridBlock, threadBlock, step, saveField.save);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
        }
        #ifdef CURVED_BOUNDARY_CONDITION
            deviceField.updateCurvedBoundaryVelocitiesDeviceField(numberCurvedBoundaryNodes);
        #endif

        //swap interface pointers
        deviceField.swapGhostInterfacesDeviceField();
        
        #ifdef LOCAL_FORCES
            deviceField.gpuResetMacroForcesDeviceField(gridBlock, threadBlock);
        #endif //LOCAL_FORCES

        #ifdef PARTICLE_MODEL
            deviceField.particleSimulationDeviceField(particlesSoA,streamsPart,step);
        #endif //PARTICLE_MODEL

        if(saveField.checkpoint){
            printf("\n--------------------------- Saving checkpoint %06d ---------------------------\n", step);if(console_flush){fflush(stdout);}
            // throwing a warning for being used without being initialized. But does not matter since we are overwriting it;
            deviceField.cudaMemcpyDeviceField(hostField);
            deviceField.interfaceCudaMemcpyDeviceField(true);       
            deviceField.saveSimCheckpointHostDeviceField(hostField, step);
            
            #ifdef PARTICLE_MODEL
                printf("Starting saveSimCheckpointParticle...\t"); fflush(stdout);
                saveSimCheckpointParticle(particlesSoA, &step);
            #endif //PARTICLE_MODEL
            
        }
       
        // Saving data checks
        if(saveField.reportSave){
            printf("\n--------------------------- Saving report %06d ---------------------------\n", step);if(console_flush){fflush(stdout);}
            deviceField.treatDataDeviceField(hostField, step);
        }
        
        if(saveField.macrSave){
            #if defined BC_FORCES && defined SAVE_BC_FORCES
                deviceField.saveBcForces(hostField);
            #endif //BC_FORCES && SAVE_BC_FORCES

            checkCudaErrors(cudaDeviceSynchronize()); 
            deviceField.cudaMemcpyDeviceField(hostField);

            printf("\n--------------------------- Saving macro %06d ---------------------------\n", step); if(console_flush){fflush(stdout);}

            if(!ONLY_FINAL_MACRO){
                hostField.saveMacrHostField(step, savingMacrVtk, savingMacrBin, false);
            }

            #ifdef BC_FORCES
                deviceField.totalBcDragDeviceField(step);
            #endif //BC_FORCES
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
    deviceField.cudaMemcpyDeviceField(hostField);

    #if defined BC_FORCES && defined SAVE_BC_FORCES
    deviceField.saveBcForces(hostField);
    #endif //BC_FORCES && SAVE_BC_FORCES

    if(console_flush){fflush(stdout);}
    hostField.saveMacrHostField(step, savingMacrVtk, savingMacrBin, false);

    if(CHECKPOINT_SAVE){
        printf("\n--------------------------- Saving checkpoint %06d ---------------------------\n", step);if(console_flush){fflush(stdout);}
        deviceField.cudaMemcpyDeviceField(hostField);
        deviceField.interfaceCudaMemcpyDeviceField(false); 
        deviceField.saveSimCheckpointDeviceField(step);
    }

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
    deviceField.freeDeviceField();

    // Free particle
    #ifdef PARTICLE_MODEL
        free(particles);
        particlesSoA.freeNodesAndCenters();
    #endif //PARTICLE_MODEL

    return 0;
}