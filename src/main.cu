#include "main.cuh"
#include "hostField.cuh"
#include "deviceField.cuh"
#include "saveField.cuh"
#ifdef PARTICLE_MODEL
#include "particleField.cuh"
#endif //PARTICLE_MODEL

using namespace std;

int main() {
    // Setup saving folder
    folderSetup();

    // Set cuda device
    checkCudaErrors(cudaSetDevice(GPU_INDEX));

    // Field Variables
    HostField hostField;
    DeviceField deviceField;

    /* ----------------- GRID AND THREADS DEFINITION FOR LBM ---------------- */
    dim3 threadBlock(BLOCK_NX, BLOCK_NY, BLOCK_NZ);
    dim3 gridBlock(NUM_BLOCK_X, NUM_BLOCK_Y, NUM_BLOCK_Z);

    /* ------------------------- ALLOCATION FOR CPU ------------------------- */
    int step = 0;

    dfloat** randomNumbers = nullptr;
    randomNumbers = (dfloat**)malloc(sizeof(dfloat*));

    hostField.allocateHostMemoryHostField();
    
    /* -------------- ALLOCATION FOR GPU ------------- */
    deviceField.allocateDeviceMemoryDeviceField();
    
    #ifdef PARTICLE_MODEL
    // Particle field initialization and allocation
    ParticleField particleField;
    particleField.allocateMemory();
    #endif //PARTICLE_MODEL
    
    #ifdef DENSITY_CORRECTION
        // Allocate density correction memory in both host and device fields
        hostField.allocateDensityCorrectionMemory();
        deviceField.allocateDensityCorrectionMemory();
    #endif //DENSITY_CORRECTION

    /* -------------- Setup Streams ------------- */
    cudaStream_t streamsLBM[1];
    checkCudaErrors(cudaSetDevice(GPU_INDEX));
    checkCudaErrors(cudaStreamCreate(&streamsLBM[0]));
    checkCudaErrors(cudaDeviceSynchronize());
    #ifdef PARTICLE_MODEL
    particleField.setupStreams();
    #endif //PARTICLE_MODEL

    step = INI_STEP;


    //Declaration of atomic flags to safely control the state of data saving in multiple threads.
    std::atomic<bool> savingMacrVtk(false);
    std::atomic<bool> savingMacrParticle(false);
    std::vector<std::atomic<bool>> savingMacrBin(hostField.NThread);

    for (int i = 0; i < hostField.NThread; i++){
        savingMacrBin[i].store(false);
    }

    /* -------------- Initialize domain in the device ------------- */
    deviceField.initializeDomainDeviceField(hostField, randomNumbers,  step, gridBlock, threadBlock);

    int ini_step = step;

    printf("Domain Initialized. Starting simulation\n"); if(console_flush) fflush(stdout);
    
    #ifdef PARTICLE_MODEL
        // Initialize particle field with position, velocity, and solver method
        particleField.initialize(&step, gridBlock, threadBlock);
        particleField.saveInfo(step, savingMacrParticle);
    #endif //PARTICLE_MODEL

    /* ------------------------------ TIMER EVENTS  ------------------------------ */
    checkCudaErrors(cudaSetDevice(GPU_INDEX));
    cudaEvent_t start, stop, start_step, stop_step;
    initializeCudaEvents(start, stop, start_step, stop_step);
    
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
        //------------------------- Main LBM Kernels -------------------------
        deviceField.gpuMomCollisionStreamDeviceField(gridBlock, threadBlock, step, saveField.save);
        // swap interface pointers
        deviceField.swapGhostInterfacesDeviceField();
        CHECK_KERNEL_ERR("Stream Collision kernel");

        //------------------------- Auxiliary Kernels -------------------------
        deviceField.halfStepKernels(gridBlock, threadBlock, step);
        #ifdef PARTICLE_MODEL
            particleField.simulationStep(deviceField.d_fMom, step);
        #endif //PARTICLE_MODEL

        //------------------------- Saving Data -------------------------
        // Saving checkpoint     
        if(saveField.checkpoint){
            printf("\n--------------------------- Saving checkpoint %06d ---------------------------\n", step);
            deviceField.cudaMemcpyDeviceField(hostField);
            deviceField.interfaceCudaMemcpyDeviceField(true);       
            deviceField.saveSimCheckpointHostDeviceField(hostField, step);
            #ifdef PARTICLE_MODEL
                particleField.saveCheckpoint(step);
            #endif //PARTICLE_MODEL    
            if(console_flush){fflush(stdout);} 
        }
       
        // Saving treat data  checks
        if(saveField.reportSave){
            printf("\n--------------------------- Saving report %06d ---------------------------\n", step);
            deviceField.treatDataDeviceField(hostField, step);
            #ifdef PARTICLE_MODEL
            particleField.exportWallForces(step);
            #endif
            if(console_flush){fflush(stdout);}
        }
        
        if(saveField.macrSave){
            //copy data from device to host
            checkCudaErrors(cudaDeviceSynchronize()); 
            deviceField.cudaMemcpyDeviceField(hostField);
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

    checkCudaErrors(cudaDeviceSynchronize());

    //Calculate MLUPS

    dfloat MLUPS = recordElapsedTime(start_step, stop_step, step, ini_step);
    printf("MLUPS: %f\n",MLUPS); if(console_flush){fflush(stdout);}     
    
    /* ------------------------------ POST ------------------------------ */
    deviceField.cudaMemcpyDeviceField(hostField);

    hostField.saveMacrHostField(step, savingMacrVtk, savingMacrBin, false);

    if(CHECKPOINT_SAVE){
        printf("\n--------------------------- Saving checkpoint %06d ---------------------------\n", step);
        deviceField.cudaMemcpyDeviceField(hostField);
        deviceField.interfaceCudaMemcpyDeviceField(false); 
        deviceField.saveSimCheckpointDeviceField(step);
        if(console_flush){fflush(stdout);}
    }

    checkCudaErrors(cudaDeviceSynchronize());
    #if MEAN_FLOW
            hostField.saveMacrHostField(INT_MAX, savingMacrVtk, savingMacrBin, true);
    #endif //MEAN_FLOW
    
    //Save info file
    saveSimInfo(step,MLUPS);

    while (savingMacrVtk) std::this_thread::yield();
    #ifdef PARTICLE_MODEL
    particleField.waitForSaving(savingMacrParticle);
    #endif
    for (size_t i = 0; i < savingMacrBin.size(); ++i) {
        while (savingMacrBin[i]) std::this_thread::yield();
    }

    /* ------------------------------ FREE ------------------------------ */

    hostField.freeHostField();
    deviceField.freeDeviceField();

    // Free particle field
    #ifdef PARTICLE_MODEL
        particleField.freeMemory();
        particleField.destroyStreams();
    #endif //PARTICLE_MODEL

    return 0;
}