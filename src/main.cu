#include "main.cuh"
#include "hostField.cuh"
#include "deviceField.cuh"

using namespace std;

int main() {
    // Setup saving folder
    folderSetup();

    //set cuda device
    checkCudaErrors(cudaSetDevice(GPU_INDEX));

    //variable declaration        
    ghostInterfaceData ghostInterface;
    hostField hostField;
    deviceField deviceField;

    /* ----------------- GRID AND THREADS DEFINITION FOR LBM ---------------- */
    dim3 threadBlock(BLOCK_NX, BLOCK_NY, BLOCK_NZ);
    dim3 gridBlock(NUM_BLOCK_X, NUM_BLOCK_Y, NUM_BLOCK_Z);

    /* ------------------------- ALLOCATION FOR CPU ------------------------- */
    int step = 0;

    dfloat** randomNumbers = nullptr; // useful for turbulence
    randomNumbers = (dfloat**)malloc(sizeof(dfloat*));

   // Populations* pop;
   // Macroscopics* macr;

    hostField.allocateHostMemoryHostField();
    
    /* -------------- ALLOCATION FOR GPU ------------- */
    deviceField.allocateDeviceMemoryDeviceField(ghostInterface);
    #ifdef DENSITY_CORRECTION
        checkCudaErrors(cudaMallocHost((void**)&(hostField.h_mean_rho), sizeof(dfloat)));
        cudaMalloc((void**)&deviceField.d_mean_rho, sizeof(dfloat));  
    #endif //DENSITY_CORRECTION

    // Setup Streams
    cudaStream_t streamsLBM[1];
    checkCudaErrors(cudaSetDevice(GPU_INDEX));
    checkCudaErrors(cudaStreamCreate(&streamsLBM[0]));
    checkCudaErrors(cudaDeviceSynchronize());
    #ifdef PARTICLE_MODEL
    cudaStream_t streamsPart[1];
    checkCudaErrors(cudaStreamCreate(&streamsPart[0]));
    #endif //PARTICLE_MODEL

    step=INI_STEP;
    //declaration of atomic flags to safely control the state of data saving in multiple threads.
    std::atomic<bool> savingMacrVtk(false);
    std::atomic<bool> savingMacrParticle(false);
    std::vector<std::atomic<bool>> savingMacrBin(hostField.NThread);

    for (int i = 0; i < hostField.NThread; i++)
        savingMacrBin[i].store(false);

    deviceField.initializeDomain(ghostInterface, hostField, randomNumbers, &step, gridBlock, threadBlock);

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

    /* ------------------------------ TIMER EVENTS  ------------------------------ */
    checkCudaErrors(cudaSetDevice(GPU_INDEX));
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

        int aux = step-INI_STEP;
        bool checkpoint = false;

        #ifdef DENSITY_CORRECTION
        //TODO: mean_rho(deviceField);
        mean_rho(deviceField.d_fMom,step,deviceField.d_mean_rho);
        #endif //DENSITY_CORRECTION

        bool save =false;
        bool reportSave = false;
        bool macrSave = false;
        bool particleSave = false;

#pragma warning(push)
#pragma warning(disable: 4804)
        if(aux != 0){
            if(REPORT_SAVE){ reportSave = !(step % REPORT_SAVE);}                
            if(MACR_SAVE){ macrSave   = !(step % MACR_SAVE);}
            if(MACR_SAVE || REPORT_SAVE){ save = (reportSave || macrSave);}
            if(CHECKPOINT_SAVE){ checkpoint = !(aux % CHECKPOINT_SAVE);}
            #ifdef PARTICLE_MODEL
                if(PARTICLES_SAVE){ particleSave = !(aux % PARTICLES_SAVE);}
            #endif //PARTICLE MODEL
        }
#pragma warning(pop)

        //TODO: gpuMomCollisionStream << <gridBlock, threadBlock DYNAMIC_SHARED_MEMORY_PARAMS>> >(deviceField, step, save);
        // ghost interface should be inside the deviceField struct
        gpuMomCollisionStream << <gridBlock, threadBlock DYNAMIC_SHARED_MEMORY_PARAMS>> >(deviceField.d_fMom, deviceField.dNodeType,ghostInterface, DENSITY_CORRECTION_PARAMS(d_) BC_FORCES_PARAMS(d_) step, save);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
        }
        //swap interface pointers
        //TODO: swapGhostInterfaces(deviceField.ghostInterface);
        swapGhostInterfaces(ghostInterface);
        
        #ifdef LOCAL_FORCES
            //TODO: gpuResetMacroForces<<<gridBlock, threadBlock>>>(deviceField);
            gpuResetMacroForces<<<gridBlock, threadBlock>>>(deviceField.d_fMom);
        #endif //LOCAL_FORCES

        #ifdef PARTICLE_MODEL
            //TODO: particleSimulation(&particlesSoA, deviceField, streamsPart, step);
            particleSimulation(&particlesSoA,deviceField.d_fMom,streamsPart,step);
        #endif //PARTICLE_MODEL

        if(checkpoint){
            printf("\n--------------------------- Saving checkpoint %06d ---------------------------\n", step);fflush(stdout);
            // throwing a warning for being used without being initialized. But does not matter since we are overwriting it;
            //TODO: the line below should be a function call, something like hostField.deviceMemCpy(deviceField);
            checkCudaErrors(cudaMemcpy(hostField.h_fMom, deviceField.d_fMom, sizeof(dfloat) * NUMBER_LBM_NODES*NUMBER_MOMENTS, cudaMemcpyDeviceToHost));
            hostField.interfaceCudaMemcpyLoop(ghostInterface);       
            //TODO: saveSimCheckpoint(hostField, deviceField, &step);
            saveSimCheckpoint(hostField.h_fMom, ghostInterface, &step);
            
            #ifdef PARTICLE_MODEL
                printf("Starting saveSimCheckpointParticle...\t"); fflush(stdout);
                saveSimCheckpointParticle(particlesSoA, &step);
            #endif //PARTICLE_MODEL
            
        }
       
        //save macroscopics

        //if (N_STEPS - step < 4*((int)turn_over_time)){
        if(reportSave){
            printf("\n--------------------------- Saving report %06d ---------------------------\n", step);
            //TODO: treatData(hostField, deviceField,step);
            treatData(hostField.h_fMom,deviceField.d_fMom,
            #if MEAN_FLOW
            hostField.m_fMom,
            #endif //MEAN_FLOW
            step); 
        }
        if(macrSave){
            deviceField.saveBcForces(hostField);
            //if (!(step%((int)turn_over_time/10))){
            //if((step>N_STEPS-80*(int)(MACR_SAVE))){ 
            //if((step%((int)(turn_over_time/2))) == 0){
                checkCudaErrors(cudaDeviceSynchronize()); 
                //TODO: the line below should be a function call, something like hostField.deviceMemCpy(deviceField);
                checkCudaErrors(cudaMemcpy(hostField.h_fMom, deviceField.d_fMom, sizeof(dfloat) * NUMBER_LBM_NODES*NUMBER_MOMENTS, cudaMemcpyDeviceToHost));

                printf("\n--------------------------- Saving macro %06d ---------------------------\n", step);
                if(console_flush){fflush(stdout);}
                //if(step > N_STEPS - 14000){
                if(!ONLY_FINAL_MACRO){
                    hostField.saveMacrHostField(step, savingMacrVtk, savingMacrBin, false);
                }
            //}

            #ifdef BC_FORCES
                //TODO: totalBcDrag(deviceField, step);
                totalBcDrag(deviceField.d_BC_Fx, deviceField.d_BC_Fy, deviceField.d_BC_Fz, step);
            #endif //BC_FORCES
        }

        #ifdef PARTICLE_MODEL
            if (particleSave){
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
    //TODO: the line below should be a function call, something like hostField.deviceMemCpy(deviceField);
    checkCudaErrors(cudaMemcpy(hostField.h_fMom, deviceField.d_fMom, sizeof(dfloat) * NUMBER_LBM_NODES*NUMBER_MOMENTS, cudaMemcpyDeviceToHost));

    deviceField.saveBcForces(hostField);

    if(console_flush){fflush(stdout);}
    hostField.saveMacrHostField(step, savingMacrVtk, savingMacrBin, false);

    /*if(CHECKPOINT_SAVE){
        printf("\n--------------------------- Saving checkpoint %06d ---------------------------\n", step);fflush(stdout);
        checkCudaErrors(cudaMemcpy(hostField.h_fMom, deviceField.d_fMom, sizeof(dfloat) * NUMBER_LBM_NODES*NUMBER_MOMENTS, cudaMemcpyDeviceToHost));
        hostField.interfaceCudaMemcpyEnd(ghostInterface);  
        saveSimCheckpoint(deviceField.d_fMom,ghostInterface,&step);
    }*/
    checkCudaErrors(cudaDeviceSynchronize());
    #if MEAN_FLOW
            hostField.saveMacrHostField(INT_MAX, savingMacrVtk, savingMacrBin, true);
    #endif //MEAN_FLOW
    
    //save info file
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
    interfaceFree(ghostInterface);

    // Free particle
    #ifdef PARTICLE_MODEL
    free(particles);
    particlesSoA.freeNodesAndCenters();
    #endif //PARTICLE_MODEL

    return 0;
}