/*
*   LBM-CERNN
*   Copyright (C) 2018-2019 Waine Barbosa de Oliveira Junior
*
*   This program is free software; you can redistribute it and/or modify
*   it under the terms of the GNU General Public License as published by
*   the Free Software Foundation; either version 2 of the License, or
*   (at your option) any later version.
*
*   This program is distributed in the hope that it will be useful,
*   but WITHOUT ANY WARRANTY; without even the implied warranty of
*   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*   GNU General Public License for more details.
*
*   You should have received a copy of the GNU General Public License along
*   with this program; if not, write to the Free Software Foundation, Inc.,
*   51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
*
*   Contact: cernn-ct@utfpr.edu.br
*/

#include "checkpoint.cuh"


void createFolder(std::string foldername)
{
    // Check if folder exists
    struct stat buffer;
    if (stat(foldername.c_str(), &buffer) == 0)
        return;

    #ifdef _WIN32
    // Windows-specific code
    std::string cmd = "md ";
    cmd += foldername;
    system(cmd.c_str());
    #else
    // Linux/macOS-specific code
    if (mkdir(foldername.c_str(), 0777) == -1)  // Convert foldername to C-string
    {
        std::cout << "Error creating folder '" << foldername << "'.\n";
    }
    #endif //_WIN32
}


std::filesystem::path getExecutablePathCheckpoint() {
    #if defined(_WIN32)
        char result[MAX_PATH];
        DWORD count = GetModuleFileNameA(NULL, result, MAX_PATH);
        if (count == 0) throw std::runtime_error("Error obtaining path to executable (Windows).");
        return std::filesystem::path(std::string(result, count));
    #elif defined(__linux__)
        char result[1024];
        ssize_t count = readlink("/proc/self/exe", result, sizeof(result));
        if (count == -1) throw std::runtime_error("Error obtaining path to executable (Linux).");
        return std::filesystem::path(std::string(result, count));
    #elif defined(__APPLE__)
        char result[1024];
        uint32_t size = sizeof(result);
        if (_NSGetExecutablePath(result, &size) != 0)
            throw std::runtime_error("Error obtaining path to executable  (macOS).");
        return std::filesystem::path(result);
    #else
        #error "Platform not supported"
    #endif
}

std::filesystem::path folderCheckpoint()
{
    std::filesystem::path exePath = getExecutablePathCheckpoint();   
    std::filesystem::path binDir = exePath.parent_path();

    std::filesystem::path foldername = binDir / PATH_FILES / ID_SIM / "checkpoint";

    std::filesystem::create_directories(foldername);

    return foldername;
}

size_t getFileSize(
    std::string filename
){
    std::streampos fsize = 0;
    std::ifstream file( filename, std::ios::binary );

    fsize = file.tellg();
    file.seekg( 0, std::ios::end);
    fsize = file.tellg() - fsize;
    file.close();

    return fsize;
}

__host__
void readFileIntoArray(
    void* arr, 
    std::string filename, 
    size_t arr_size_bytes, 
    void* tmp
){
    FILE* file = fopen((filename+".bin").c_str(), "rb");
    // Check if file exists
    if(file == nullptr){
        std::cout << "Error reading file '" << filename << ".bin'. Exiting\n";
    }
    // load file size into array, if it is zero
    if(arr_size_bytes == 0){
        arr_size_bytes = getFileSize(filename);
    }

    // Read file into temporary array
    fread(tmp, arr_size_bytes, 1, file);
    // Copy file content in tmp to GPU array
    checkCudaErrors(cudaMemcpy(arr, tmp, arr_size_bytes, cudaMemcpyDefault));

    fclose(file);
}

__host__
void writeFileIntoArray(void* arr, const std::string filename, const size_t arr_size_bytes, void* tmp){
    FILE* file = fopen((filename+".bin").c_str(), "wb");
    // Check if file exists
    if(file == nullptr){
        std::cout << "Error opening file '" << filename << ".bin' to write. Exiting\n";
    }

    // Copy file content from GPU array to tmp
    checkCudaErrors(cudaMemcpy(tmp, arr, arr_size_bytes, cudaMemcpyDefault));

    // Write temporary array into file
    fwrite(tmp, arr_size_bytes, 1, file);

    fclose(file);
}

__host__ 
void writeFilesIntoDfloat3SoA(dfloat3SoA arr, const std::string foldername, const size_t arr_size_bytes, void* tmp){
    // Write x, y and z to files (portable)
    std::filesystem::path folder(foldername);
    createFolder(folder.string());
    writeFileIntoArray(arr.x, (folder / "x").string(), arr_size_bytes, tmp);
    writeFileIntoArray(arr.y, (folder / "y").string(), arr_size_bytes, tmp);
    writeFileIntoArray(arr.z, (folder / "z").string(), arr_size_bytes, tmp);
} 

__host__
std::string getCheckpointFilenameRead(std::string name, int gpu_index){
    std::filesystem::path p = std::filesystem::path(SIMULATION_FOLDER_LOAD_CHECKPOINT) / ID_SIM / "checkpoint" / (std::to_string(gpu_index) + "_" + name);
    return p.string();
} 

__host__
void readFilesIntoDfloat3SoA(dfloat3SoA arr, const std::string foldername, const size_t arr_size_bytes, void* tmp){
    // Read to x, y and z in dfloat3SoA (portable)
    std::filesystem::path folder(foldername);
    readFileIntoArray(arr.x, (folder / "x").string(), arr_size_bytes, tmp);
    readFileIntoArray(arr.y, (folder / "y").string(), arr_size_bytes, tmp);
    readFileIntoArray(arr.z, (folder / "z").string(), arr_size_bytes, tmp);
} 

__host__
std::string getCheckpointFilenameWrite(std::string name, int gpu_index){
    std::filesystem::path exePath = getExecutablePathCheckpoint();
    std::filesystem::path binDir = exePath.parent_path();
    std::filesystem::path p = binDir / PATH_FILES / ID_SIM / "checkpoint" / (std::to_string(gpu_index) + "_" + name);
    return p.string();
} 

__host__
void operateSimCheckpoint( 
    int oper,
    dfloat* fMom,
    ghostInterfaceData ghostInterface,
    int* step,
    int gpu_index
    )
{
    // Defining what functions to use (read or write to files)
    checkCudaErrors(cudaSetDevice(GPUS_TO_USE[gpu_index]));
    void (*f_arr)(void*, const std::string, size_t, void*);
    std::string (*f_filename)(std::string , int);

    if(oper == __LOAD_CHECKPOINT){
        f_arr = &readFileIntoArray;
        f_filename = &getCheckpointFilenameRead;
    }else if(oper == __SAVE_CHECKPOINT){
        f_arr = &writeFileIntoArray;
        f_filename = &getCheckpointFilenameWrite;
    }else{
        std::cout << "Invalid operation. Exiting\n";
        exit(-1);
    }

    // Everything will fit in this array
    dfloat* tmp = (dfloat*)malloc(MEM_SIZE_MOM_LOCAL);

    // Load/save current step
    f_arr(step, f_filename("curr_step", gpu_index), sizeof(int), tmp);

    if(oper == __LOAD_CHECKPOINT){
        printf("Loaded checkpoint: step %d | N_GPU: %d \n",step[0], gpu_index);
    }else if(oper == __SAVE_CHECKPOINT){
        printf("Saved checkpoint: step %d | N_GPU: %d \n",step[0], gpu_index);
    }

    // Load/save pop
    f_arr(fMom, f_filename("fMom", gpu_index), MEM_SIZE_MOM_LOCAL, tmp);
    
    if(oper == __LOAD_CHECKPOINT){
        printf("Loaded checkpoint: moments | N_GPU: %d \n", gpu_index);
    }else if(oper == __SAVE_CHECKPOINT){
        printf("Saved checkpoint: moments | N_GPU: %d \n", gpu_index);
    }

    // Load/save auxilary populations
    f_arr(ghostInterface.h_pop.X_0, f_filename("ghost.X_0", gpu_index), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * QF, tmp);
    f_arr(ghostInterface.h_pop.X_1, f_filename("ghost.X_1", gpu_index), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * QF, tmp);
    f_arr(ghostInterface.h_pop.Y_0, f_filename("ghost.Y_0", gpu_index), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * QF, tmp);
    f_arr(ghostInterface.h_pop.Y_1, f_filename("ghost.Y_1", gpu_index), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * QF, tmp);
    f_arr(ghostInterface.h_pop.Z_0, f_filename("ghost.Z_0", gpu_index), sizeof(dfloat) * NUMBER_GHOST_FACE_XY_LOCAL * QF, tmp);
    f_arr(ghostInterface.h_pop.Z_1, f_filename("ghost.Z_1", gpu_index), sizeof(dfloat) * NUMBER_GHOST_FACE_XY_LOCAL * QF, tmp);
    if(oper == __LOAD_CHECKPOINT){
        printf("Loaded checkpoint: f_pops | N_GPU: %d \n", gpu_index);
    }else if(oper == __SAVE_CHECKPOINT){
        printf("Saved checkpoint: f_pops | N_GPU: %d \n", gpu_index);
    }

    #ifdef SECOND_DIST 
    f_arr(ghostInterface.h_g.X_0, f_filename("g.X_0", gpu_index), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF, tmp);
    f_arr(ghostInterface.h_g.X_1, f_filename("g.X_1", gpu_index), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF, tmp);
    f_arr(ghostInterface.h_g.Y_0, f_filename("g.Y_0", gpu_index), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF, tmp);
    f_arr(ghostInterface.h_g.Y_1, f_filename("g.Y_1", gpu_index), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF, tmp);
    f_arr(ghostInterface.h_g.Z_0, f_filename("g.Z_0", gpu_index), sizeof(dfloat) * NUMBER_GHOST_FACE_XY_LOCAL * GF, tmp);
    f_arr(ghostInterface.h_g.Z_1, f_filename("g.Z_1", gpu_index), sizeof(dfloat) * NUMBER_GHOST_FACE_XY_LOCAL * GF, tmp);
    if(oper == __LOAD_CHECKPOINT){
        printf("Loaded checkpoint: g_pop | N_GPU: %d \n", gpu_index);
    }else if(oper == __SAVE_CHECKPOINT){
        printf("Saved checkpoint: g_pop | N_GPU: %d \n", gpu_index);
    }

    #endif //SECOND_DIST

    #ifdef PHI_DIST 
    f_arr(ghostInterface.h_phi.X_0, f_filename("phi.X_0", gpu_index), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF, tmp);
    f_arr(ghostInterface.h_phi.X_1, f_filename("phi.X_1", gpu_index), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF, tmp);
    f_arr(ghostInterface.h_phi.Y_0, f_filename("phi.Y_0", gpu_index), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF, tmp);
    f_arr(ghostInterface.h_phi.Y_1, f_filename("phi.Y_1", gpu_index), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF, tmp);
    f_arr(ghostInterface.h_phi.Z_0, f_filename("phi.Z_0", gpu_index), sizeof(dfloat) * NUMBER_GHOST_FACE_XY_LOCAL * GF, tmp);
    f_arr(ghostInterface.h_phi.Z_1, f_filename("phi.Z_1", gpu_index), sizeof(dfloat) * NUMBER_GHOST_FACE_XY_LOCAL * GF, tmp);
    if(oper == __LOAD_CHECKPOINT){
        printf("Loaded checkpoint: phi_pop | N_GPU: %d \n", gpu_index);
    }else if(oper == __SAVE_CHECKPOINT){
        printf("Saved checkpoint: phi_pop | N_GPU: %d \n", gpu_index);
    }

    #endif //PHI_DIST

    #ifdef A_XX_DIST 
    f_arr(ghostInterface.h_Axx.X_0, f_filename("Axx.X_0", gpu_index), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF, tmp);
    f_arr(ghostInterface.h_Axx.X_1, f_filename("Axx.X_1", gpu_index), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF, tmp);
    f_arr(ghostInterface.h_Axx.Y_0, f_filename("Axx.Y_0", gpu_index), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF, tmp);
    f_arr(ghostInterface.h_Axx.Y_1, f_filename("Axx.Y_1", gpu_index), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF, tmp);
    f_arr(ghostInterface.h_Axx.Z_0, f_filename("Axx.Z_0", gpu_index), sizeof(dfloat) * NUMBER_GHOST_FACE_XY_LOCAL * GF, tmp);
    f_arr(ghostInterface.h_Axx.Z_1, f_filename("Axx.Z_1", gpu_index), sizeof(dfloat) * NUMBER_GHOST_FACE_XY_LOCAL * GF, tmp);
    if(oper == __LOAD_CHECKPOINT){
        printf("Loaded checkpoint: Axx_pop | N_GPU: %d \n", gpu_index);
    }else if(oper == __SAVE_CHECKPOINT){
        printf("Saved checkpoint: Axx_pop | N_GPU: %d \n", gpu_index);
    }

    #endif //A_XX_DIST

    #ifdef A_XY_DIST 
    f_arr(ghostInterface.h_Axy.X_0, f_filename("Axy.X_0", gpu_index), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF, tmp);
    f_arr(ghostInterface.h_Axy.X_1, f_filename("Axy.X_1", gpu_index), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF, tmp);
    f_arr(ghostInterface.h_Axy.Y_0, f_filename("Axy.Y_0", gpu_index), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF, tmp);
    f_arr(ghostInterface.h_Axy.Y_1, f_filename("Axy.Y_1", gpu_index), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF, tmp);
    f_arr(ghostInterface.h_Axy.Z_0, f_filename("Axy.Z_0", gpu_index), sizeof(dfloat) * NUMBER_GHOST_FACE_XY_LOCAL * GF, tmp);
    f_arr(ghostInterface.h_Axy.Z_1, f_filename("Axy.Z_1", gpu_index), sizeof(dfloat) * NUMBER_GHOST_FACE_XY_LOCAL * GF, tmp);
    if(oper == __LOAD_CHECKPOINT){
        printf("Loaded checkpoint: Axy_pop | N_GPU: %d \n", gpu_index);
    }else if(oper == __SAVE_CHECKPOINT){
        printf("Saved checkpoint: Axy_pop | N_GPU: %d \n", gpu_index);
    }

    #endif //A_XY_DIST

    #ifdef A_XZ_DIST 
    f_arr(ghostInterface.h_Axz.X_0, f_filename("Axz.X_0", gpu_index), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF, tmp);
    f_arr(ghostInterface.h_Axz.X_1, f_filename("Axz.X_1", gpu_index), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF, tmp);
    f_arr(ghostInterface.h_Axz.Y_0, f_filename("Axz.Y_0", gpu_index), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF, tmp);
    f_arr(ghostInterface.h_Axz.Y_1, f_filename("Axz.Y_1", gpu_index), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF, tmp);
    f_arr(ghostInterface.h_Axz.Z_0, f_filename("Axz.Z_0", gpu_index), sizeof(dfloat) * NUMBER_GHOST_FACE_XY_LOCAL * GF, tmp);
    f_arr(ghostInterface.h_Axz.Z_1, f_filename("Axz.Z_1", gpu_index), sizeof(dfloat) * NUMBER_GHOST_FACE_XY_LOCAL * GF, tmp);
    if(oper == __LOAD_CHECKPOINT){
        printf("Loaded checkpoint: Axz_pop | N_GPU: %d \n", gpu_index);
    }else if(oper == __SAVE_CHECKPOINT){
        printf("Saved checkpoint: Axz_pop | N_GPU: %d \n", gpu_index);
    }

    #endif //A_XZ_DIST

    #ifdef A_YY_DIST 
    f_arr(ghostInterface.h_Ayy.X_0, f_filename("Ayy.X_0", gpu_index), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF, tmp);
    f_arr(ghostInterface.h_Ayy.X_1, f_filename("Ayy.X_1", gpu_index), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF, tmp);
    f_arr(ghostInterface.h_Ayy.Y_0, f_filename("Ayy.Y_0", gpu_index), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF, tmp);
    f_arr(ghostInterface.h_Ayy.Y_1, f_filename("Ayy.Y_1", gpu_index), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF, tmp);
    f_arr(ghostInterface.h_Ayy.Z_0, f_filename("Ayy.Z_0", gpu_index), sizeof(dfloat) * NUMBER_GHOST_FACE_XY_LOCAL * GF, tmp);
    f_arr(ghostInterface.h_Ayy.Z_1, f_filename("Ayy.Z_1", gpu_index), sizeof(dfloat) * NUMBER_GHOST_FACE_XY_LOCAL * GF, tmp);
    if(oper == __LOAD_CHECKPOINT){
        printf("Loaded checkpoint: Ayy_pop | N_GPU: %d \n", gpu_index);
    }else if(oper == __SAVE_CHECKPOINT){
        printf("Saved checkpoint: Ayy_pop | N_GPU: %d \n", gpu_index);
    }

    #endif //A_YY_DIST

    #ifdef A_YZ_DIST 
    f_arr(ghostInterface.h_Ayz.X_0, f_filename("Ayz.X_0", gpu_index), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF, tmp);
    f_arr(ghostInterface.h_Ayz.X_1, f_filename("Ayz.X_1", gpu_index), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF, tmp);
    f_arr(ghostInterface.h_Ayz.Y_0, f_filename("Ayz.Y_0", gpu_index), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF, tmp);
    f_arr(ghostInterface.h_Ayz.Y_1, f_filename("Ayz.Y_1", gpu_index), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF, tmp);
    f_arr(ghostInterface.h_Ayz.Z_0, f_filename("Ayz.Z_0", gpu_index), sizeof(dfloat) * NUMBER_GHOST_FACE_XY_LOCAL * GF, tmp);
    f_arr(ghostInterface.h_Ayz.Z_1, f_filename("Ayz.Z_1", gpu_index), sizeof(dfloat) * NUMBER_GHOST_FACE_XY_LOCAL * GF, tmp);
    if(oper == __LOAD_CHECKPOINT){
        printf("Loaded checkpoint: Ayz_pop | N_GPU: %d \n", gpu_index);
    }else if(oper == __SAVE_CHECKPOINT){
        printf("Saved checkpoint: Ayz_pop | N_GPU: %d \n", gpu_index);
    }

    #endif //A_YZ_DIST

    #ifdef A_ZZ_DIST 
    f_arr(ghostInterface.h_Azz.X_0, f_filename("Azz.X_0", gpu_index), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF, tmp);
    f_arr(ghostInterface.h_Azz.X_1, f_filename("Azz.X_1", gpu_index), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF, tmp);
    f_arr(ghostInterface.h_Azz.Y_0, f_filename("Azz.Y_0", gpu_index), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF, tmp);
    f_arr(ghostInterface.h_Azz.Y_1, f_filename("Azz.Y_1", gpu_index), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF, tmp);
    f_arr(ghostInterface.h_Azz.Z_0, f_filename("Azz.Z_0", gpu_index), sizeof(dfloat) * NUMBER_GHOST_FACE_XY * GF, tmp);
    f_arr(ghostInterface.h_Azz.Z_1, f_filename("Azz.Z_1", gpu_index), sizeof(dfloat) * NUMBER_GHOST_FACE_XY * GF, tmp);
    if(oper == __LOAD_CHECKPOINT){
        printf("Loaded checkpoint: Azz_pop | N_GPU: %d \n", gpu_index);
    }else if(oper == __SAVE_CHECKPOINT){
        printf("Saved checkpoint: Azz_pop | N_GPU: %d \n", gpu_index);
    }

    #endif //A_ZZ_DIST

    free(tmp);
}



__host__
int getStep(int gpu_index,  int* step){
    std::string filename = SIMULATION_FOLDER_LOAD_CHECKPOINT;

    // Build a portable path to the checkpoint file using std::filesystem
    std::filesystem::path dir = std::filesystem::path("..") / "bin" / filename / ID_SIM / "checkpoint";
    std::filesystem::path filePath = dir /(std::to_string(gpu_index) + "_curr_step.bin");

    std::ifstream fileread(filePath.string(), std::ios::binary);

    if (!fileread) {
        std::cerr << "Error opening file: " << filePath << std::endl;
        return -1;
    }

    fileread.seekg(0, std::ios::end);
    std::streampos filesize = fileread.tellg();

    if (filesize < sizeof(int)) {
        std::cerr << "Error: File smaller than expected!" << std::endl;
        return -2;
    }

    fileread.seekg(-sizeof(int), std::ios::end);

    int laststep = 0;
    fileread.read(reinterpret_cast<char*>(&laststep), sizeof(int));

    if (!fileread.good()) {
        std::cerr << "Error reading data from file!" << std::endl;
        return -3;
    }
    
    fileread.close();

    return laststep+1;
}

__host__
int loadSimCheckpoint( 
    dfloat* fMom,
    ghostInterfaceData ghostInterface,
    int *step,
    int gpu_index
    ){
    checkCudaErrors(cudaSetDevice(GPUS_TO_USE[gpu_index]));
    step[0] = getStep(gpu_index, step);

    if(step[0] < INI_STEP)
        step[0]=INI_STEP;

    if (step[0]<=0){
        std::cerr << "Starting from step " << step[0] << std::endl;
        return 0;
    }
    operateSimCheckpoint(__LOAD_CHECKPOINT, fMom, ghostInterface,step, gpu_index);
    return 1;
}


__host__
void saveSimCheckpoint( 
    dfloat* fMom,
    ghostInterfaceData ghostInterface,
    int *step,
    int gpu_index
    ){
    folderCheckpoint();

    operateSimCheckpoint(__SAVE_CHECKPOINT, fMom,ghostInterface, step, gpu_index);
}

#ifdef PARTICLE_MODEL

__host__
void operateSimCheckpointParticle( 
    int oper,
    ParticlesSoA& particlesSoA,
    int* step
    )
{
    // Defining what functions to use (read or write to files)
    void (*f_arr)(void*, const std::string, size_t, void*);
    void (*f_dfloat3SoA)(dfloat3SoA, const std::string, size_t, void*);
    std::string (*f_filename)(std::string);

    if(oper == __LOAD_CHECKPOINT){
        f_arr = &readFileIntoArray;
        f_dfloat3SoA = &readFilesIntoDfloat3SoA;
        f_filename = &getCheckpointFilenameRead;
    }else if(oper == __SAVE_CHECKPOINT){
        f_arr = &writeFileIntoArray;
        f_dfloat3SoA = &writeFilesIntoDfloat3SoA;
        f_filename = &getCheckpointFilenameWrite;
    }else{
        std::cout << "Invalid operation. Exiting\n";
        exit(-1);
    }

    // Everything will fit in this array
    dfloat* tmp = (dfloat*)malloc(MEM_SIZE_POP);

    // Load/save current step
    f_arr(step, f_filename("curr_step_particle"), sizeof(int), tmp);
    
    ParticleMethod* methodArray = particlesSoA.getPMethod();
    ParticleMethod& method = methodArray[GPU_INDEX];

    if(method == IBM){
        // Load particles centers positions
        checkCudaErrors(cudaSetDevice(GPU_INDEX));
        f_arr(particlesSoA.getPCenterArray(), f_filename("IBM_particles_centers"), 
            NUM_PARTICLES*sizeof(ParticleCenter), tmp);
    }
    
    checkCudaErrors(cudaSetDevice(GPU_INDEX));

    if(method == IBM){

        IbmNodesSoA* nodesArray = particlesSoA.getNodesSoA();
        IbmNodesSoA& nSoA = nodesArray[GPU_INDEX];

        // IBM nodes bytes size
        if(oper == __LOAD_CHECKPOINT){
            size_t filesize = getFileSize(f_filename("IBM_nodes_centers_idx.bin"));
            nSoA.setNumNodes(filesize / sizeof(unsigned int));
        }
        size_t ibm_nodes_arr_size = nSoA.getNumNodes() * sizeof(dfloat);
        size_t ibm_nodes_arr_size_uint = nSoA.getNumNodes() * sizeof(unsigned int);
        // Load/save IBM nodes values
        f_arr(nSoA.getParticleCenterIdx(), f_filename("IBM_nodes_centers_idx"), ibm_nodes_arr_size_uint, tmp);
        f_dfloat3SoA(nSoA.getPos(), f_filename("IBM_nodes_pos"), ibm_nodes_arr_size, tmp);
        f_dfloat3SoA(nSoA.getVel(), f_filename("IBM_nodes_vel"), ibm_nodes_arr_size, tmp);
        f_dfloat3SoA(nSoA.getVelOld(), f_filename("IBM_nodes_vel_old"), ibm_nodes_arr_size, tmp);
        f_dfloat3SoA(nSoA.getF(), f_filename("IBM_nodes_f"), ibm_nodes_arr_size, tmp);
        f_dfloat3SoA(nSoA.getDeltaF(), f_filename("IBM_nodes_deltaF"), ibm_nodes_arr_size, tmp);
        f_arr(nSoA.getS(), f_filename("IBM_nodes_S"), ibm_nodes_arr_size, tmp);
    }
    
    free(tmp);
    
}
__host__
int loadSimCheckpointParticle( 
    ParticlesSoA& particlesSoA,
    int *step
    ){
    step[0] = getStep();

    if(step[0] < INI_STEP)
        step[0]=INI_STEP;

    if (step[0]<=0){
        std::cerr << "Starting from step " << step[0] << std::endl;
        return 0;
    }
    operateSimCheckpointParticle(__LOAD_CHECKPOINT, particlesSoA, step);
    return 1;
}
__host__
void saveSimCheckpointParticle( 
    ParticlesSoA& particlesSoA,
    int *step
    ){
    folderCheckpoint();

    operateSimCheckpointParticle(__SAVE_CHECKPOINT, particlesSoA, step);
}
#endif //PARTICLE_MODEL