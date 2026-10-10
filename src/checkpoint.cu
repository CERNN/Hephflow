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
    if (stat(foldername.c_str(), &buffer) == 0) return;

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
        if (_NSGetExecutablePath(result, &size) != 0) throw std::runtime_error("Error obtaining path to executable  (macOS).");
        return std::filesystem::path(result);
    #else
        #error "Platform not supported"
    #endif // defined(_WIN32)
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
    const std::string path = filename + ".bin";
    FILE* file = fopen(path.c_str(), "rb");
    if(file == nullptr) throw std::runtime_error("Cannot open checkpoint section: " + path);
    // load file size into array, if it is zero
    if(arr_size_bytes == 0){
        arr_size_bytes = getFileSize(path);
    }

    const size_t fileSize = getFileSize(path);
    if (fileSize != arr_size_bytes) {
        fclose(file);
        throw std::runtime_error("Checkpoint section size mismatch: " + path);
    }

    // Read file into temporary array
    if (fread(tmp, arr_size_bytes, 1, file) != 1) {
        fclose(file);
        throw std::runtime_error("Truncated checkpoint section: " + path);
    }
    // Copy file content in tmp to GPU array
    checkCudaErrors(cudaMemcpy(arr, tmp, arr_size_bytes, cudaMemcpyDefault));

    fclose(file);
}

__host__
void writeFileIntoArray(void* arr, const std::string filename, const size_t arr_size_bytes, void* tmp){
    const std::filesystem::path path = filename + ".bin";
    const std::filesystem::path temporary = path.string() + ".tmp";
    FILE* file = fopen(temporary.string().c_str(), "wb");
    if(file == nullptr) throw std::runtime_error("Cannot open checkpoint section for writing: " + temporary.string());

    // Host-backed checkpoints can be written directly from the shared moment
    // snapshot. Device-backed checkpoints still need the staging allocation.
    const void* fileData = arr;
    if (tmp != nullptr) {
        checkCudaErrors(cudaMemcpy(tmp, arr, arr_size_bytes, cudaMemcpyDefault));
        fileData = tmp;
    }
    if (fwrite(fileData, arr_size_bytes, 1, file) != 1 || fflush(file) != 0) {
        fclose(file);
        throw std::runtime_error("Cannot write checkpoint section: " + temporary.string());
    }
    if (fclose(file) != 0) throw std::runtime_error("Cannot close checkpoint section: " + temporary.string());

    std::error_code error;
    std::filesystem::remove(path, error);
    std::filesystem::rename(temporary, path);
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
    std::filesystem::path checkpointRoot(SIMULATION_FOLDER_LOAD_CHECKPOINT);
    if (checkpointRoot.is_relative()) {
        checkpointRoot = getExecutablePathCheckpoint().parent_path() / checkpointRoot;
    }
    std::filesystem::path p = checkpointRoot / ID_SIM / "checkpoint" / (std::to_string(gpu_index) + "_" + name);
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

#ifdef CONFORMATION_TENSOR
/**
 * Save or validate the metadata that identifies how the six conformation fields are represented.
 * The tensor payload itself is handled by the normal moment checkpoint; this sidecar prevents loading identical
 * binary fields as A, log(A), or sqrt(A) under a different build configuration.
 */
static void operateConformationCheckpointMetadata(int oper, int gpu_index)
{
    // Read and write use their respective checkpoint roots so restart input may differ from the current output path.
    const std::string path = (oper == __LOAD_CHECKPOINT) ? getCheckpointFilenameRead("conformation.meta", gpu_index) : getCheckpointFilenameWrite("conformation.meta", gpu_index);

    // Saving always rewrites the complete versioned sidecar. Representation names are persisted API values.
    if (oper == __SAVE_CHECKPOINT) {
        std::ofstream out(path, std::ios::trunc);
        if (!out) throw std::runtime_error("Cannot write conformation checkpoint metadata: "+path);
        out << "version=1\nrepresentation=" << veConformationRepresentationName() << "\n";
        return;
    }

    // Loading rejects missing metadata rather than guessing how an untagged tensor should be interpreted.
    std::ifstream in(path);
    if (!in) throw std::runtime_error( "Conformation checkpoint metadata is missing; refusing an untagged restart: "+path);

    // Parse keys independently so future metadata can add fields without making version 1 readers order-dependent.
    std::string line, representation, version;
    while (std::getline(in,line)) {
        const std::string versionKey="version=";
        const std::string key="representation=";
        if (line.rfind(versionKey,0)==0) version=line.substr(versionKey.size());
        if (line.rfind(key,0)==0) representation=line.substr(key.size());
    }

    // The format version protects the parser; the representation protects the physical meaning of the payload.
    if (version != "1") throw std::runtime_error("Unsupported conformation checkpoint metadata version '"+ version+"' in "+path);
    if (representation != veConformationRepresentationName()) throw std::runtime_error("Checkpoint conformation representation is '"+ representation+"' but this build requires '"+ veConformationRepresentationName()+"'");
}
#endif // CONFORMATION_TENSOR

namespace {
    #ifdef CONVECTION_DIFFUSION_TRANSPORT
        constexpr unsigned int CHECKPOINT_GQ = GQ;
    #else
        constexpr unsigned int CHECKPOINT_GQ = 0;
    #endif // CONVECTION_DIFFUSION_TRANSPORT

unsigned long long checkpointFeatureMask()
{
    unsigned long long mask = 0;
        #ifdef SECOND_DIST
            mask |= 1ull << 0;
        #endif // SECOND_DIST
        #ifdef PHI_DIST
            mask |= 1ull << 1;
        #endif // PHI_DIST
        #ifdef LAMBDA_DIST
            mask |= 1ull << 2;
        #endif // LAMBDA_DIST
        #ifdef A_XX_DIST
            mask |= 1ull << 3;
        #endif // A_XX_DIST
        #ifdef A_XY_DIST
            mask |= 1ull << 4;
        #endif // A_XY_DIST
        #ifdef A_XZ_DIST
            mask |= 1ull << 5;
        #endif // A_XZ_DIST
        #ifdef A_YY_DIST
            mask |= 1ull << 6;
        #endif // A_YY_DIST
        #ifdef A_YZ_DIST
            mask |= 1ull << 7;
        #endif // A_YZ_DIST
        #ifdef A_ZZ_DIST
            mask |= 1ull << 8;
        #endif // A_ZZ_DIST
    return mask;
}

constexpr std::uint32_t GLOBAL_CHECKPOINT_VERSION = 4;
constexpr std::uint32_t CHECKPOINT_ENDIAN_TAG = 0x01020304u;

enum GlobalCheckpointSectionId : std::uint32_t {
    SECTION_MOMENTS = 1, SECTION_HYDRO = 2, SECTION_G = 3,
    SECTION_PHI = 4, SECTION_LAMBDA = 5, SECTION_AXX = 6,
    SECTION_AXY = 7, SECTION_AXZ = 8, SECTION_AYY = 9,
    SECTION_AYZ = 10, SECTION_AZZ = 11
};

std::uint64_t checkpointHash(const void* data, size_t bytes)
{
    const auto* values = static_cast<const unsigned char*>(data);
    std::uint64_t hash = 1469598103934665603ull;
    for (size_t i = 0; i < bytes; ++i) {
        hash ^= static_cast<std::uint64_t>(values[i]);
        hash *= 1099511628211ull;
    }
    return hash;
}

std::uint64_t checkpointConformationHash()
{
        #ifdef CONFORMATION_TENSOR
            const char* name = veConformationRepresentationName();
            return checkpointHash(name, std::strlen(name));
        #else
            return 0;
        #endif // CONFORMATION_TENSOR
}

std::filesystem::path globalCheckpointReadPath(const char* name)
{
    std::filesystem::path root(SIMULATION_FOLDER_LOAD_CHECKPOINT);
    if (root.is_relative()) root = getExecutablePathCheckpoint().parent_path() / root;
    return root / ID_SIM / "checkpoint" / name;
}

std::filesystem::path globalCheckpointWritePath(const char* name)
{
    return folderCheckpoint() / name;
}

GlobalCheckpointHeader makeGlobalCheckpointHeader(
    int step, size_t sectionCount)
{
    GlobalCheckpointHeader header{};
    constexpr char magic[] = "HEPHFLOW_GLOBAL";
    static_assert(sizeof(magic) <= sizeof(header.magic));
    std::memcpy(header.magic, magic, sizeof(magic));
    header.version = GLOBAL_CHECKPOINT_VERSION;
    header.endianTag = CHECKPOINT_ENDIAN_TAG;
    header.scalarBytes = sizeof(dfloat);
    header.q = Q;
    header.gq = CHECKPOINT_GQ;
    header.nx = NX;
    header.ny = NY;
    header.nz = NZ;
    header.momentCount = NUMBER_MOMENTS;
    header.sectionCount = static_cast<std::uint32_t>(sectionCount);
    header.populationParity = static_cast<std::uint32_t>(step) & 1u;
    header.step = static_cast<std::uint64_t>(step);
    header.featureMask = checkpointFeatureMask();
    header.conformationHash = checkpointConformationHash();
    return header;
}

void validateGlobalCheckpointHeader(const GlobalCheckpointHeader& header)
{
    constexpr char magic[] = "HEPHFLOW_GLOBAL";
    if (std::memcmp(header.magic, magic, sizeof(magic)) != 0 || header.version != GLOBAL_CHECKPOINT_VERSION) throw std::runtime_error( "Legacy or unsupported checkpoint format; expected HEPHFLOW_GLOBAL version 4");
    if (header.endianTag != CHECKPOINT_ENDIAN_TAG) throw std::runtime_error("Checkpoint endianness does not match this build");
    if (header.scalarBytes != sizeof(dfloat) || header.q != Q || header.gq != CHECKPOINT_GQ || header.nx != NX || header.ny != NY || header.nz != NZ || header.momentCount != NUMBER_MOMENTS || header.featureMask != checkpointFeatureMask() || header.conformationHash != checkpointConformationHash() || header.populationParity != (header.step & 1u)) throw std::runtime_error( "Checkpoint physical/build metadata does not match this run");
}

} // namespace

__host__
void saveGlobalSimCheckpoint(
    const dfloat* globalBlockedMoments,
    const CanonicalCheckpointHostState& populations,
    int step)
{
    const size_t physicalNodes = static_cast<size_t>(NX) * NY * NZ;
    std::vector<dfloat> dense(physicalNodes * NUMBER_MOMENTS);
    for (size_t z = 0; z < static_cast<size_t>(NZ); ++z)
        for (size_t y = 0; y < static_cast<size_t>(NY); ++y)
            for (size_t x = 0; x < static_cast<size_t>(NX); ++x) {
                const size_t node = x + static_cast<size_t>(NX) * (y + static_cast<size_t>(NY) * z);
                for (int moment = 0; moment < NUMBER_MOMENTS; ++moment) {
                    const size_t blocked = idxMom(x % BLOCK_NX, y % BLOCK_NY, z % BLOCK_NZ, moment, x / BLOCK_NX, y / BLOCK_NY, z / BLOCK_NZ);
                    dense[node * NUMBER_MOMENTS + moment] =
                        globalBlockedMoments[blocked];
                }
            }

    std::vector<CheckpointSectionView> views;
    views.push_back({SECTION_MOMENTS, dense.data(), dense.size()});
    views.push_back({SECTION_HYDRO, populations.hydro.data(), populations.hydro.size()});

    #ifdef SECOND_DIST
        views.push_back({SECTION_G, populations.g.data(), populations.g.size()});
    #endif // SECOND_DIST
    #ifdef PHI_DIST
        views.push_back({SECTION_PHI, populations.phi.data(), populations.phi.size()});
    #endif // PHI_DIST
    #ifdef LAMBDA_DIST
        views.push_back({SECTION_LAMBDA, populations.lambda.data(), populations.lambda.size()});
    #endif // LAMBDA_DIST
    #ifdef A_XX_DIST
        views.push_back({SECTION_AXX, populations.Axx.data(), populations.Axx.size()});
    #endif // A_XX_DIST
    #ifdef A_XY_DIST
        views.push_back({SECTION_AXY, populations.Axy.data(), populations.Axy.size()});
    #endif // A_XY_DIST
    #ifdef A_XZ_DIST
        views.push_back({SECTION_AXZ, populations.Axz.data(), populations.Axz.size()});
    #endif // A_XZ_DIST
    #ifdef A_YY_DIST
        views.push_back({SECTION_AYY, populations.Ayy.data(), populations.Ayy.size()});
    #endif // A_YY_DIST
    #ifdef A_YZ_DIST
        views.push_back({SECTION_AYZ, populations.Ayz.data(), populations.Ayz.size()});
    #endif // A_YZ_DIST
    #ifdef A_ZZ_DIST
        views.push_back({SECTION_AZZ, populations.Azz.data(), populations.Azz.size()});
    #endif // A_ZZ_DIST

    if (populations.hydro.size() != physicalNodes * Q) throw std::runtime_error("Canonical checkpoint population capture is incomplete");

    const GlobalCheckpointHeader header = makeGlobalCheckpointHeader(step, views.size());
    std::vector<GlobalCheckpointSection> sections(views.size());
    std::uint64_t offset = sizeof(header) + sections.size() * sizeof(GlobalCheckpointSection);

    for (size_t i = 0; i < views.size(); ++i) {
        sections[i].id = views[i].id;
        sections[i].elementBytes = sizeof(dfloat);
        sections[i].elementCount = views[i].count;
        sections[i].byteOffset = offset;
        sections[i].byteCount = views[i].count * sizeof(dfloat);
        sections[i].checksum = checkpointHash(views[i].data, sections[i].byteCount);
        offset += sections[i].byteCount;
    }

    const std::uint64_t descriptorChecksum = checkpointHash( sections.data(), sections.size() * sizeof(GlobalCheckpointSection));
    const std::filesystem::path state = globalCheckpointWritePath("global_state.bin");
    const std::filesystem::path stateTemporary = state.string() + ".tmp";
    const std::filesystem::path manifest = globalCheckpointWritePath("global.meta");
    const std::filesystem::path manifestTemporary = manifest.string() + ".tmp";

    std::error_code error;
    std::filesystem::remove(manifest, error);
    {
        std::ofstream out(stateTemporary, std::ios::binary | std::ios::trunc);
        if (!out) throw std::runtime_error( "Cannot create global checkpoint payload: " + stateTemporary.string());
        out.write(reinterpret_cast<const char*>(&header), sizeof(header));
        out.write(reinterpret_cast<const char*>(sections.data()), sections.size() * sizeof(GlobalCheckpointSection));
        for (const auto& view : views) out.write(reinterpret_cast<const char*>(view.data), view.count * sizeof(dfloat));
        out.flush();
        if (!out) throw std::runtime_error( "Cannot write global checkpoint payload: " + stateTemporary.string());
    }
    std::filesystem::remove(state, error);
    std::filesystem::rename(stateTemporary, state);

    {
        std::ofstream out(manifestTemporary, std::ios::trunc);
        if (!out) throw std::runtime_error( "Cannot create global checkpoint manifest: " + manifestTemporary.string());
        out << "HEPHFLOW_GLOBAL " << GLOBAL_CHECKPOINT_VERSION << ' '
            << offset << ' ' << descriptorChecksum << '\n';
        out.flush();
        if (!out) throw std::runtime_error( "Cannot write global checkpoint manifest: " + manifestTemporary.string());
    }
    std::filesystem::rename(manifestTemporary, manifest);
    printf("Saved global checkpoint: step %d | physical nodes: %zu\n",
        step, physicalNodes);
}

__host__
int loadGlobalSimCheckpoint(
    dfloat* fMom, ghostInterfaceData ghostInterface, int* step,
    const GpuDomainLayout& layout, dim3 threadBlock)
{
    const std::filesystem::path manifest = globalCheckpointReadPath("global.meta");
    const std::filesystem::path state = globalCheckpointReadPath("global_state.bin");
    std::ifstream metadata(manifest);
    if (!metadata) {
        throw std::runtime_error( "Legacy or incomplete checkpoint rejected: missing " + manifest.string());
    }
    std::string magic;
    std::uint32_t version = 0;
    std::uint64_t expectedFileBytes = 0, descriptorChecksum = 0;
    metadata >> magic >> version >> expectedFileBytes >> descriptorChecksum;
    if (!metadata || magic != "HEPHFLOW_GLOBAL" || version != GLOBAL_CHECKPOINT_VERSION) throw std::runtime_error( "Unsupported or truncated global checkpoint manifest: " + manifest.string());

    std::ifstream in(state, std::ios::binary);
    if (!in) throw std::runtime_error( "Cannot open global checkpoint payload: " + state.string());
    in.seekg(0, std::ios::end);
    const auto fileBytes = static_cast<std::uint64_t>(in.tellg());
    in.seekg(0, std::ios::beg);
    if (fileBytes != expectedFileBytes) throw std::runtime_error("Global checkpoint payload size mismatch");

    GlobalCheckpointHeader header{};
    in.read(reinterpret_cast<char*>(&header), sizeof(header));
    validateGlobalCheckpointHeader(header);

    std::vector<GlobalCheckpointSection> sections(header.sectionCount);
    in.read(reinterpret_cast<char*>(sections.data()), sections.size() * sizeof(GlobalCheckpointSection));
    if (!in || checkpointHash(sections.data(), sections.size() * sizeof(GlobalCheckpointSection)) != descriptorChecksum) throw std::runtime_error("Global checkpoint section table integrity check failed");

    const size_t physicalNodes = static_cast<size_t>(NX) * NY * NZ;
    std::vector<dfloat> dense(physicalNodes * NUMBER_MOMENTS);
    CanonicalCheckpointHostState populations;
    populations.resizeForGlobalDomain();
    std::vector<MutableCheckpointSectionView> expected;
    expected.push_back({SECTION_MOMENTS, dense.data(), dense.size()});
    expected.push_back({SECTION_HYDRO, populations.hydro.data(), populations.hydro.size()});

    #ifdef SECOND_DIST
        expected.push_back({SECTION_G, populations.g.data(), populations.g.size()});
    #endif // SECOND_DIST
    #ifdef PHI_DIST
        expected.push_back({SECTION_PHI, populations.phi.data(), populations.phi.size()});
    #endif // PHI_DIST
    #ifdef LAMBDA_DIST
        expected.push_back({SECTION_LAMBDA, populations.lambda.data(), populations.lambda.size()});
    #endif // LAMBDA_DIST
    #ifdef A_XX_DIST
        expected.push_back({SECTION_AXX, populations.Axx.data(), populations.Axx.size()});
    #endif // A_XX_DIST
    #ifdef A_XY_DIST
        expected.push_back({SECTION_AXY, populations.Axy.data(), populations.Axy.size()});
    #endif // A_XY_DIST
    #ifdef A_XZ_DIST
        expected.push_back({SECTION_AXZ, populations.Axz.data(), populations.Axz.size()});
    #endif // A_XZ_DIST
    #ifdef A_YY_DIST
        expected.push_back({SECTION_AYY, populations.Ayy.data(), populations.Ayy.size()});
    #endif // A_YY_DIST
    #ifdef A_YZ_DIST
        expected.push_back({SECTION_AYZ, populations.Ayz.data(), populations.Ayz.size()});
    #endif // A_YZ_DIST
    #ifdef A_ZZ_DIST
        expected.push_back({SECTION_AZZ, populations.Azz.data(), populations.Azz.size()});
    #endif // A_ZZ_DIST

    if (sections.size() != expected.size()) throw std::runtime_error("Checkpoint section count does not match enabled fields");
    std::uint64_t nextOffset = sizeof(header) + sections.size() * sizeof(GlobalCheckpointSection);

    for (size_t i = 0; i < sections.size(); ++i) {
        const auto& section = sections[i];
        const auto& destination = expected[i];
        const std::uint64_t requiredBytes = destination.count * sizeof(dfloat);
        if (section.id != destination.id || section.elementBytes != sizeof(dfloat) || section.elementCount != destination.count || section.byteCount != requiredBytes || section.byteOffset != nextOffset || section.byteOffset + section.byteCount > fileBytes) throw std::runtime_error("Checkpoint section metadata is invalid or incompatible");
        in.read(reinterpret_cast<char*>(destination.data), section.byteCount);
        if (!in || checkpointHash(destination.data, section.byteCount) != section.checksum) throw std::runtime_error("Global checkpoint section integrity check failed");
        nextOffset += section.byteCount;
    }

    if (nextOffset != fileBytes) throw std::runtime_error("Global checkpoint contains trailing or missing payload data");

    std::vector<dfloat> local(layout.paddedNodeCount * NUMBER_MOMENTS, 0.0_df);

    for (size_t zLocal = 0; zLocal < layout.localNZ; ++zLocal) {
        const size_t z = layout.zStart + zLocal;
        for (size_t y = 0; y < static_cast<size_t>(NY); ++y)
            for (size_t x = 0; x < static_cast<size_t>(NX); ++x) {
                const size_t node = x + static_cast<size_t>(NX) * (y + static_cast<size_t>(NY) * z);
                for (int moment = 0; moment < NUMBER_MOMENTS; ++moment) {
                    const size_t blocked = idxMom(x % BLOCK_NX, y % BLOCK_NY, zLocal % BLOCK_NZ, moment, x / BLOCK_NX, y / BLOCK_NY, zLocal / BLOCK_NZ);
                    local[blocked] = dense[node * NUMBER_MOMENTS + moment];
                }
            }
    }

    checkCudaErrors(cudaMemcpy(fMom, local.data(), layout.momentStorageBytes, cudaMemcpyHostToDevice));

    restoreCanonicalCheckpointPopulations( populations, ghostInterface, layout, threadBlock, header.step == 0 ? false : ((header.step - 1u) & 1u) != 0);
    *step = static_cast<int>(header.step);
    printf("Loaded global checkpoint: next step %d | z=[%zu,%zu)\n",
        *step, layout.zStart, layout.zStart + layout.localNZ);
    return 1;
}

__host__
int getStep(int gpu_index, int* /*step*/){
    std::filesystem::path filePath = getCheckpointFilenameRead("curr_step", gpu_index) + ".bin";

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

    return laststep;
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
    std::string (*f_filename)(std::string, int);

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
    f_arr(step, f_filename("curr_step_particle", GPU_INDEX), sizeof(int), tmp);
    
    ParticleMethod* methodArray = particlesSoA.getPMethod();
    ParticleMethod& method = methodArray[GPU_INDEX];

    if(method == IBM){
        // Load particles centers positions
        checkCudaErrors(cudaSetDevice(GPU_INDEX));
        f_arr(particlesSoA.getPCenterArray(), f_filename("IBM_particles_centers", GPU_INDEX), NUM_PARTICLES*sizeof(ParticleCenter), tmp);
        f_arr(particlesSoA.getCollisionDataArray(), f_filename("IBM_particles_collision_history", GPU_INDEX), NUM_PARTICLES*sizeof(CollisionData), tmp);
        if (oper == __LOAD_CHECKPOINT) particlesSoA.bindCollisionData();
    }
    
    checkCudaErrors(cudaSetDevice(GPU_INDEX));

    if(method == IBM){

        IbmNodesSoA* nodesArray = particlesSoA.getNodesSoA();
        IbmNodesSoA& nSoA = nodesArray[GPU_INDEX];

        // IBM nodes bytes size
        if(oper == __LOAD_CHECKPOINT){
            size_t filesize = getFileSize(f_filename("IBM_nodes_centers_idx.bin", GPU_INDEX));
            nSoA.setNumNodes(filesize / sizeof(unsigned int));
        }
        size_t ibm_nodes_arr_size = nSoA.getNumNodes() * sizeof(dfloat);
        size_t ibm_nodes_arr_size_uint = nSoA.getNumNodes() * sizeof(unsigned int);
        // Load/save IBM nodes values
        f_arr(nSoA.getParticleCenterIdx(), f_filename("IBM_nodes_centers_idx", GPU_INDEX), ibm_nodes_arr_size_uint, tmp);
        f_dfloat3SoA(nSoA.getPos(), f_filename("IBM_nodes_pos", GPU_INDEX), ibm_nodes_arr_size, tmp);
        f_dfloat3SoA(nSoA.getVel(), f_filename("IBM_nodes_vel", GPU_INDEX), ibm_nodes_arr_size, tmp);
        f_dfloat3SoA(nSoA.getVelOld(), f_filename("IBM_nodes_vel_old", GPU_INDEX), ibm_nodes_arr_size, tmp);
        f_dfloat3SoA(nSoA.getF(), f_filename("IBM_nodes_f", GPU_INDEX), ibm_nodes_arr_size, tmp);
        f_dfloat3SoA(nSoA.getDeltaF(), f_filename("IBM_nodes_deltaF", GPU_INDEX), ibm_nodes_arr_size, tmp);
        f_arr(nSoA.getS(), f_filename("IBM_nodes_S", GPU_INDEX), ibm_nodes_arr_size, tmp);
    }
    
    free(tmp);
    
}
__host__
int loadSimCheckpointParticle( 
    ParticlesSoA& particlesSoA,
    int *step
    ){
    step[0] = getStep(GPU_INDEX, step);

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
