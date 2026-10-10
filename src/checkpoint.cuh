/**
 *  @file checkpoint.cuh
 *  Contributors history:
 *  @author Waine Jr. (waine@alunos.utfpr.edu.br)
 *  @author Marco Aurelio Ferrari (e.marcoferrari@utfpr.edu.br)
 *  @author Ricardo de Souza
 *  @brief Loading and saving of variables checkpoints
 *  @version 0.4.0
 *  @date 01/09/2025
 */


#ifndef __CHECKPOINT_H
#define __CHECKPOINT_H

#include <string>
#include <fstream>
#include <iostream>     // std::cout, std::fixed
#include <cuda.h>
#include <cuda_runtime.h>
#include <builtin_types.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <filesystem>
#include <array>
#include <cstdint>
#include <cstring>
#include <vector>

#if defined(_WIN32)
    #include <windows.h>
#elif defined(__linux__)
    #include <unistd.h>
#elif defined(__APPLE__)
    #include <mach-o/dyld.h>
#endif // __APPLE__

#include "include/errorDef.h"
#include "globalFunctions.h"
#include "var.h"
#include "checkpoint_codec.cuh"
#include "non_newtonian/conformation_representation.cuh"
#ifdef PARTICLE_MODEL
#include "particles/class/Particle.cuh"
#endif //PARTICLE_MODEL
#include <sys/stat.h>
#include <sys/types.h>



#define __LOAD_CHECKPOINT 1
#define __SAVE_CHECKPOINT 2

/**
 * @brief Fixed-size metadata stored at the beginning of a global checkpoint.
 *
 * The packed layout is part of the on-disk checkpoint format.
 */
#pragma pack(push, 1)
struct GlobalCheckpointHeader {
    char magic[16];
    std::uint32_t version;
    std::uint32_t endianTag;
    std::uint32_t scalarBytes;
    std::uint32_t q;
    std::uint32_t gq;
    std::uint32_t nx;
    std::uint32_t ny;
    std::uint32_t nz;
    std::uint32_t momentCount;
    std::uint32_t sectionCount;
    std::uint32_t populationParity;
    std::uint64_t step;
    std::uint64_t featureMask;
    std::uint64_t conformationHash;
};

/**
 * @brief Packed descriptor for one payload section in a global checkpoint.
 */
struct GlobalCheckpointSection {
    std::uint32_t id;
    std::uint32_t elementBytes;
    std::uint64_t elementCount;
    std::uint64_t byteOffset;
    std::uint64_t byteCount;
    std::uint64_t checksum;
};
#pragma pack(pop)

/**
 * @brief Read-only in-memory payload section used while saving a checkpoint.
 */
struct CheckpointSectionView {
    std::uint32_t id;
    const dfloat* data;
    size_t count;
};

/**
 * @brief Writable in-memory payload section used while loading a checkpoint.
 */
struct MutableCheckpointSectionView {
    std::uint32_t id;
    dfloat* data;
    size_t count;
};

/**
 * @brief Loads a decomposition-independent fluid checkpoint.
 *
 * Moments and canonical logical populations are read in dense physical
 * XYZ-major order, encoded into the selected streaming layout, and supplied
 * with fresh transient exchange halos.
 *
 * @param fMom Device moment storage for this GPU slab.
 * @param ghostInterface Runtime population-interface storage.
 * @param step Receives the saved simulation step.
 * @param layout This GPU's physical slab layout.
 * @param threadBlock CUDA block dimensions used by the codec kernels.
 * @return Nonzero when the checkpoint was loaded successfully.
 */
__host__ int loadGlobalSimCheckpoint( dfloat* fMom, ghostInterfaceData ghostInterface, int* step, const GpuDomainLayout& layout, dim3 threadBlock);

/**
 * @brief Writes a decomposition-independent fluid checkpoint.
 *
 * @param globalBlockedMoments Global moment snapshot in blocked storage order.
 * @param populations Canonical population snapshot in physical XYZ-major order.
 * @param step Simulation step represented by the snapshot.
 */
__host__ void saveGlobalSimCheckpoint( const dfloat* globalBlockedMoments, const CanonicalCheckpointHostState& populations, int step);

/**
 *  @brief Create a new folder
 *  @param foldername Folder name
 *  @return size_t Filesize
 */
void createFolder( std::string foldername );

__host__
std::filesystem::path getExecutablePathCheckpoint();

__host__
std::filesystem::path folderCheckpoint();


/**
 *  @brief Get the filesize of file  
 *  @param filename Filename to get size
 *  @return size_t Filesize
 */
size_t getFileSize( std::string filename );

/**
 *  @brief Reads file content into GPU array
 *  @param arr GPU array to write file content to
 *  @param filename Filename to read from
 *  @param arr_size_bytes Size in bytes to read from file. If zero, reads whole file
 *  @param tmp Temporary array used to read file (already allocated, make sure that the file content fits in it)
 */
__host__
void readFileIntoArray( void* arr, std::string filename, size_t arr_size_bytes, void* tmp );

/**
 *  @brief Writes GPU array content into file
 *  @param arr GPU array to read content from
 *  @param filename Filename to write to
 *  @param arr_size_bytes Size in bytes to write to file
 *  @param tmp Optional staging array for device data; nullptr writes host data directly
 */
__host__
void writeFileIntoArray( void* arr, const std::string filename, const size_t arr_size_bytes, void* tmp );

/**
 *  @brief Writes dfloat3SoA GPU arrays content into files
 *  @param arr GPU arrays to read content from
 *  @param foldername Foldername to save files to
 *  @param arr_size_bytes Size in bytes to write for each file. If zero, reads whole file
 *  @param tmp Temporary array used to read from file (already allocated, make sure that the file content fits in it)
 */
__host__ 
void writeFilesIntoDfloat3SoA( dfloat3SoA arr, const std::string foldername, const size_t arr_size_bytes, void* tmp );

/**
 *  @brief Get the checkpoint filename to read from 
 *  @param name Field name (such as "rho", "u", etc.)
 *  @param gpu_index Index for reading the files of the GPU being initialized.
 *  @return std::string string checkpoint filename
 */
__host__
std::string getCheckpointFilenameRead( std::string name, int gpu_index );

/**
 *  @brief Reads files contents into dfloat3SoA GPU arrays
 *  @param arr GPU arrays to write content to
 *  @param foldername Foldername to read files from
 *  @param arr_size_bytes Size in bytes to read for each file. 
 *  @param tmp Temporary array used to write to file (already allocated, make sure that the file content fits in it)
 */
__host__
void readFilesIntoDfloat3SoA( dfloat3SoA arr, const std::string foldername, const size_t arr_size_bytes, void* tmp );
  
/**
 *  @brief Get the checkpoint filename to write to
 *  @param name Field name (such as "rho", "u", etc.)
 *  @param gpu_index GPU index to obtain the checkpoint filename for writing.
 *  @return std::string string checkpoint filename
 */
__host__
std::string getCheckpointFilenameWrite( std::string name, int gpu_index );

#ifdef PARTICLE_MODEL
__host__
void operateSimCheckpointParticle( int oper, ParticlesSoA& particlesSoA, int* step );
#endif //PARTICLE_MODEL

#ifdef PARTICLE_MODEL
/**
 *  @brief Load simulation checkpoint IBM
 *  @param particlesSoA Particles
 *  @param step Pointer to current step value in main
 *  @return 0 = fail to load checkpoint, 1 = load success;
 */
__host__
int loadSimCheckpointParticle( ParticlesSoA& particlesSoA, int *step );
#endif //PARTICLE_MODEL


#ifdef PARTICLE_MODEL

/**
 *  @brief Save simulation checkpoint IMB
 *  @param particlesSoA particle
 *  @param step Pointer to current step value in main
 */
__host__
void saveSimCheckpointParticle( ParticlesSoA& particlesSoA, int *step );
#endif //PARTICLE_MODEL



#endif // __CHECKPOINT_H
