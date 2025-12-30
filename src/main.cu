#include "main.cuh"
#include "hostField.cuh"
#include "deviceField.cuh"
#include "saveField.cuh"

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
    //TODO: should be inside particleField
    //forces from the particles into the wall
    ParticleWallForces* d_pwForces;
    cudaMalloc((void**)&d_pwForces, sizeof(ParticleWallForces));
    #endif //PARTICLE_MODEL
    
    #ifdef DENSITY_CORRECTION
        //TODO: move the functions to inside deviceField and hostField
        checkCudaErrors(cudaMallocHost((void**)&(hostField.h_mean_rho), sizeof(dfloat)));
        cudaMalloc((void**)&deviceField.d_mean_rho, sizeof(dfloat));  
    #endif //DENSITY_CORRECTION

    /* -------------- Setup Streams ------------- */
    cudaStream_t streamsLBM[1];
    checkCudaErrors(cudaSetDevice(GPU_INDEX));
    checkCudaErrors(cudaStreamCreate(&streamsLBM[0]));
    checkCudaErrors(cudaDeviceSynchronize());
    #ifdef PARTICLE_MODEL
    cudaStream_t streamsPart[1];
    checkCudaErrors(cudaStreamCreate(&streamsPart[0]));
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
        //TODO: create a particleField data structure, to store the particle functions and structs.
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
        //TODO: this should be a property of deviceField
        unsigned int numberCurvedBoundaryNodes = getNumberCurvedBoundaryNodes(hostField.hNodeType);
    #endif //CURVED_BOUNDARY_CONDITION

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
        #ifdef LOCAL_FORCES
            deviceField.gpuResetMacroForcesDeviceField(gridBlock, threadBlock);
            CHECK_KERNEL_ERR("Force Reset kernel");
        #endif //LOCAL_FORCES
        #ifdef CURVED_BOUNDARY_CONDITION
            //kernel used for adding curved boundary information
            //TODO: move this function to inside of deviceField
           updateCurvedBoundaryVelocities<<<curvedBCGridSize, curvedBCBlockSize>>>(deviceField.d_curvedBC_array, deviceField.d_fMom, numberCurvedBoundaryNodes);
           CHECK_KERNEL_ERR("Curved BC kernel");
        #endif //CURVED_BOUNDARY_CONDITION
        #ifdef PHI_DIST
            //kernel used to compute the gradients of phi
            //TODO: move this function to inside of deviceField
            gpuComputePhaseNormals<<<gridBlock, threadBlock>>>(deviceField.d_fMom, deviceField.dNodeType);
            CHECK_KERNEL_ERR("Phi gradients kernel");
            cudaDeviceSynchronize();
        #endif //PHI_DIST
        #ifdef DENSITY_CORRECTION
            deviceField.mean_rhoDeviceField(step);
            CHECK_KERNEL_ERR("Density correction kernel");
        #endif //DENSITY_CORRECTION
        #ifdef PARTICLE_MODEL
            deviceField.particleSimulationDeviceField(particlesSoA,streamsPart,d_pwForces,step);
        #endif //PARTICLE_MODEL

        //------------------------- Saving Data -------------------------
        // Saving checkpoint     
        if(saveField.checkpoint){
            printf("\n--------------------------- Saving checkpoint %06d ---------------------------\n", step);
            deviceField.cudaMemcpyDeviceField(hostField);
            deviceField.interfaceCudaMemcpyDeviceField(true);       
            deviceField.saveSimCheckpointHostDeviceField(hostField, step);
            #ifdef PARTICLE_MODEL
                printf("Starting saveSimCheckpointParticle...\t"); fflush(stdout);
                saveSimCheckpointParticle(particlesSoA, &step);
            #endif //PARTICLE_MODEL    
            if(console_flush){fflush(stdout);} 
        }
       
        // Saving treat data  checks
        if(saveField.reportSave){
            printf("\n--------------------------- Saving report %06d ---------------------------\n", step);
            deviceField.treatDataDeviceField(hostField, step);
            #ifdef PARTICLE_MODEL
            collectAndExportWallForces(d_pwForces,step);
            #endif
            if(console_flush){fflush(stdout);}
        }
        
        if(saveField.macrSave){
            // TODO: do something with this, it just shouldnt be on main.cu
            #if defined BC_FORCES && defined SAVE_BC_FORCES
                deviceField.saveBcForces(hostField);
            #endif //BC_FORCES && SAVE_BC_FORCES

            //copy data from device to host
            checkCudaErrors(cudaDeviceSynchronize()); 
            deviceField.cudaMemcpyDeviceField(hostField);

            printf("\n--------------------------- Saving macro %06d ---------------------------\n", step);

            if(!ONLY_FINAL_MACRO){ hostField.saveMacrHostField(step, savingMacrVtk, savingMacrBin, false);}

            //TODO: move this to treat data, no reason to stay on main as separate identity
            #ifdef BC_FORCES
                deviceField.totalBcDragDeviceField(step);
            #endif //BC_FORCES
            if(console_flush){fflush(stdout);}
        }

        #ifdef PARTICLE_MODEL
            if (saveField.particleSave){
                printf("\n------------------------- Saving particles %06d -------------------------\n", step);
                while (savingMacrParticle) std::this_thread::yield();
                saveParticlesInfo(&particlesSoA, step, savingMacrParticle);
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

    // TODO: do something with this, it just shouldnt be on main.cu
    #if defined BC_FORCES && defined SAVE_BC_FORCES
    deviceField.saveBcForces(hostField);
    #endif //BC_FORCES && SAVE_BC_FORCES

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