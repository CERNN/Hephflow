#include "saveData.cuh"
#include <queue>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <chrono>
#include <algorithm>
#include <cstring>
#include <initializer_list>


std::filesystem::path getExecutablePath() {
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

namespace {
    const std::vector<size_t>& vtkToBlockedScalarIndex()
    {
        // The simulation stores scalars block-by-block, while VTK requires
        // x-fastest global ordering. Build this geometry-only permutation
        // once and reuse it for every output step.
        static const std::vector<size_t> indices = [] {
            const size_t vtkPointCount = static_cast<size_t>(NX) * NY * NZ;
            std::vector<size_t> result(vtkPointCount);

            for (int z = 0; z < NZ; ++z) {
                for (int y = 0; y < NY; ++y) {
                    for (int x = 0; x < NX; ++x) {
                        result[idxScalarGlobal(x, y, z)] = idxScalarBlock(
                            x % BLOCK_NX, y % BLOCK_NY, z % BLOCK_NZ,
                            x / BLOCK_NX, y / BLOCK_NY, z / BLOCK_NZ);
                    }
                }
            }
            return result;
        }();
        return indices;
    }

    size_t momentBaseFromBlockedScalar(size_t blockedScalarIndex)
    {
        const size_t block = blockedScalarIndex / BLOCK_LBM_SIZE;
        const size_t localNode = blockedScalarIndex - block * BLOCK_LBM_SIZE;
        return localNode + BLOCK_LBM_SIZE * NUMBER_MOMENTS * block;
    }

    void linearizeBlockedScalars(
        const std::vector<size_t>& vtkToBlocked,
        std::initializer_list<dfloat*> fields)
    {
        if (fields.size() == 0) return;

        const size_t pointCount = vtkToBlocked.size();
        const size_t fieldBytes = pointCount * sizeof(dfloat);
        dfloat* scratch = nullptr;
        checkCudaErrors(cudaMallocHost(
            reinterpret_cast<void**>(&scratch), fieldBytes * fields.size()));

        for (size_t vtkIndex = 0; vtkIndex < pointCount; ++vtkIndex) {
            size_t fieldIndex = 0;
            for (dfloat* field : fields) {
                scratch[fieldIndex * pointCount + vtkIndex] =
                    field[vtkToBlocked[vtkIndex]];
                ++fieldIndex;
            }
        }

        size_t fieldIndex = 0;
        for (dfloat* field : fields) {
            std::memcpy(field, scratch + fieldIndex * pointCount, fieldBytes);
            ++fieldIndex;
        }
        cudaFreeHost(scratch);
    }

    std::mutex vtkSeriesMutex;
    std::vector<std::pair<std::string, double>> vtkSeriesEntries;

    void updateVtkSeries(const std::string& vtkFilePath, unsigned int nSteps)
    {
#if defined(HAS_VTK_TIME)
        const double vtkPhysicalTime = static_cast<double>(nSteps) * static_cast<double>(vtk_time);
        const std::filesystem::path vtkPath(vtkFilePath);
        const std::filesystem::path seriesPath = vtkPath.parent_path() / (std::string(ID_SIM) + "_vtk.vtk.series");
        const std::string fileName = vtkPath.filename().string();

        std::lock_guard<std::mutex> lock(vtkSeriesMutex);
        auto it = std::find_if(
            vtkSeriesEntries.begin(),
            vtkSeriesEntries.end(),
            [&](const std::pair<std::string, double>& entry) {
                return entry.first == fileName;
            });

        if (it == vtkSeriesEntries.end()) {
            vtkSeriesEntries.emplace_back(fileName, vtkPhysicalTime);
        } else {
            it->second = vtkPhysicalTime;
        }

        std::ofstream series(seriesPath);
        if (!series) {
            std::cerr << "[updateVtkSeries] ERROR: cannot open " << seriesPath << "\n";
            return;
        }

        series << "{\n";
        series << "  \"file-series-version\" : \"1.0\",\n";
        series << "  \"files\" : [\n";
        for (size_t i = 0; i < vtkSeriesEntries.size(); ++i) {
            series << "    { \"name\" : \"" << vtkSeriesEntries[i].first
                   << "\", \"time\" : " << vtkSeriesEntries[i].second << " }";
            if (i + 1 < vtkSeriesEntries.size()) series << ",";
            series << "\n";
        }
        series << "  ]\n";
        series << "}\n";
#else
        (void)vtkFilePath;
        (void)nSteps;
#endif
    }
}
std::filesystem::path folderSetup()
{
    std::filesystem::path exePath = getExecutablePath();
    std::filesystem::path binDir = exePath.parent_path();

    std::filesystem::path baseDir = binDir / PATH_FILES / ID_SIM;
    std::filesystem::create_directories(baseDir);

    return baseDir;
}   

// choose correct swap based on sizeof(dfloat)
template<typename T>
void writeBigEndian(std::ofstream& ofs, const T* data, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        if constexpr (sizeof(T) == 4) {
            uint32_t tmp;
            memcpy(&tmp, &data[i], 4);
            tmp = swap32(tmp);
            ofs.write(reinterpret_cast<char*>(&tmp), 4);
        }
        else if constexpr (sizeof(T) == 8) {
            uint64_t tmp;
            memcpy(&tmp, &data[i], 8);
            tmp = swap64(tmp);
            ofs.write(reinterpret_cast<char*>(&tmp), 8);
        }
    }
}

// Simple single-worker queue to serialize file saves and avoid spawning unbounded threads
namespace {
    struct SaveTask {
        std::function<void()> run;
        std::atomic<bool>* flag; // flag to clear on completion
    };

    std::mutex saveQueueMutex;
    std::condition_variable saveQueueCv;
    std::queue<SaveTask> saveQueue;
    std::thread saveWorker;
    std::atomic<bool> workerStarted{false};

    void ensureSaveWorker()
    {
        if (workerStarted.load(std::memory_order_acquire)) return;
        bool expected = false;
        if (!workerStarted.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) return;

        saveWorker = std::thread([] {
            for (;;) {
                SaveTask task;
                {
                    std::unique_lock<std::mutex> lk(saveQueueMutex);
                    saveQueueCv.wait(lk, [] { return !saveQueue.empty(); });
                    task = std::move(saveQueue.front());
                    saveQueue.pop();
                }
                task.run();
                if (task.flag) task.flag->store(false, std::memory_order_release);
                saveQueueCv.notify_all();
            }
        });
        saveWorker.detach();
    }

    void enqueueSaveTask(SaveTask task)
    {
        ensureSaveWorker();
        {
            std::lock_guard<std::mutex> lk(saveQueueMutex);
            saveQueue.push(std::move(task));
        }
        saveQueueCv.notify_one();
    }

    void waitAllSaveTasks()
    {
        std::unique_lock<std::mutex> lk(saveQueueMutex);
        saveQueueCv.wait(lk, [] { return saveQueue.empty(); });
    }
}

__host__
void saveMacr(const SaveDataParams* params)
{
    // Unpack parameters from struct
    dfloat* h_fMom = params->h_fMom;
    dfloat* rho = params->h_rho;
    dfloat* ux = params->h_ux;
    dfloat* uy = params->h_uy;
    dfloat* uz = params->h_uz;
    unsigned int* hNodeType = params->h_nodeType;
    #ifdef OMEGA_FIELD
    dfloat* omega = params->h_omega;
    #endif
    #ifdef SECOND_DIST
    dfloat* C = params->h_C;
    #endif
    #ifdef PHI_DIST
    dfloat* phi = params->h_phi;
    #endif
    #ifdef LAMBDA_DIST
    dfloat* lambda = params->h_lambda;
    #endif
    #ifdef A_XX_DIST
    dfloat* Axx = params->h_Axx;
    #endif
    #ifdef A_XY_DIST
    dfloat* Axy = params->h_Axy;
    #endif
    #ifdef A_XZ_DIST
    dfloat* Axz = params->h_Axz;
    #endif
    #ifdef A_YY_DIST
    dfloat* Ayy = params->h_Ayy;
    #endif
    #ifdef A_YZ_DIST
    dfloat* Ayz = params->h_Ayz;
    #endif
    #ifdef A_ZZ_DIST
    dfloat* Azz = params->h_Azz;
    #endif
    #if NODE_TYPE_SAVE
    unsigned int* nodeTypeData = params->h_nodeTypeSave;
    #endif
    #if defined(BC_FORCES) && defined(SAVE_BC_FORCES)
    dfloat* h_BC_Fx = params->h_BC_Fx;
    dfloat* h_BC_Fy = params->h_BC_Fy;
    dfloat* h_BC_Fz = params->h_BC_Fz;
    #endif
    #ifdef SAVE_LOCAL_FORCES
    dfloat* h_Local_Fx = params->h_Local_Fx;
    dfloat* h_Local_Fy = params->h_Local_Fy;
    dfloat* h_Local_Fz = params->h_Local_Fz;
        #ifdef SECOND_DIST
    dfloat* h_Source_C = params->h_Source_C;
        #endif
        #ifdef PHI_DIST
    dfloat* h_Source_Phi = params->h_Source_Phi;
        #endif
        #ifdef LAMBDA_DIST
    dfloat* h_Source_Lambda = params->h_Source_Lambda;
        #endif
        #ifdef CONFORMATION_TENSOR
            #ifdef A_XX_DIST
    dfloat* h_Source_Gxx = params->h_Source_Gxx;
            #endif
            #ifdef A_XY_DIST
    dfloat* h_Source_Gxy = params->h_Source_Gxy;
            #endif
            #ifdef A_XZ_DIST
    dfloat* h_Source_Gxz = params->h_Source_Gxz;
            #endif
            #ifdef A_YY_DIST
    dfloat* h_Source_Gyy = params->h_Source_Gyy;
            #endif
            #ifdef A_YZ_DIST
    dfloat* h_Source_Gyz = params->h_Source_Gyz;
            #endif
            #ifdef A_ZZ_DIST
    dfloat* h_Source_Gzz = params->h_Source_Gzz;
            #endif
        #endif //CONFORMATION_TENSOR
    #endif
    unsigned int nSteps = params->nSteps;
    std::atomic<bool>& savingMacrVtk = *params->savingMacrVtk;
    std::vector<std::atomic<bool>>& savingMacrBin = *params->savingMacrBin;

    // Reuse of rho/ux/uy/uz buffers across saves requires waiting for previous
    // asynchronous file writes to finish before linearizing new data into them.
    while (savingMacrVtk.load(std::memory_order_acquire)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    for (size_t i = 0; i < savingMacrBin.size(); ++i) {
        while (savingMacrBin[i].load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
    }

    const std::vector<size_t>& vtkToBlocked = vtkToBlockedScalarIndex();
    for (size_t indexMacr = 0; indexMacr < vtkToBlocked.size(); ++indexMacr) {
        const size_t blockedScalarIndex = vtkToBlocked[indexMacr];
        const size_t momentBase = momentBaseFromBlockedScalar(blockedScalarIndex);

        rho[indexMacr] = RHO_0 + h_fMom[momentBase + BLOCK_LBM_SIZE * M_RHO_INDEX];
        ux[indexMacr]  = h_fMom[momentBase + BLOCK_LBM_SIZE * M_UX_INDEX];
        uy[indexMacr]  = h_fMom[momentBase + BLOCK_LBM_SIZE * M_UY_INDEX];
        uz[indexMacr]  = h_fMom[momentBase + BLOCK_LBM_SIZE * M_UZ_INDEX];

        #ifdef OMEGA_FIELD
        omega[indexMacr] = h_fMom[momentBase + BLOCK_LBM_SIZE * M_OMEGA_INDEX];
        #endif //OMEGA_FIELD

        #ifdef SECOND_DIST
        C[indexMacr] = h_fMom[momentBase + BLOCK_LBM_SIZE * M2_C_INDEX];
        #endif //SECOND_DIST
        #ifdef PHI_DIST
        phi[indexMacr] = h_fMom[momentBase + BLOCK_LBM_SIZE * M3_PHI_INDEX];
        #endif //PHI_DIST
        #ifdef LAMBDA_DIST
        lambda[indexMacr] = h_fMom[momentBase + BLOCK_LBM_SIZE * M4_LAMBDA_INDEX] - LAMBDA_ZERO;
        #endif //LAMBDA_DIST
        #ifdef A_XX_DIST
        Axx[indexMacr] = h_fMom[momentBase + BLOCK_LBM_SIZE * A_XX_C_INDEX] - CONF_ZERO;
        #endif //A_XX_DIST
        #ifdef A_XY_DIST
        Axy[indexMacr] = h_fMom[momentBase + BLOCK_LBM_SIZE * A_XY_C_INDEX] - CONF_ZERO;
        #endif //A_XY_DIST
        #ifdef A_XZ_DIST
        Axz[indexMacr] = h_fMom[momentBase + BLOCK_LBM_SIZE * A_XZ_C_INDEX] - CONF_ZERO;
        #endif //A_XZ_DIST
        #ifdef A_YY_DIST
        Ayy[indexMacr] = h_fMom[momentBase + BLOCK_LBM_SIZE * A_YY_C_INDEX] - CONF_ZERO;
        #endif //A_YY_DIST
        #ifdef A_YZ_DIST
        Ayz[indexMacr] = h_fMom[momentBase + BLOCK_LBM_SIZE * A_YZ_C_INDEX] - CONF_ZERO;
        #endif //A_YZ_DIST
        #ifdef A_ZZ_DIST
        Azz[indexMacr] = h_fMom[momentBase + BLOCK_LBM_SIZE * A_ZZ_C_INDEX] - CONF_ZERO;
        #endif //A_ZZ_DIST

        #if NODE_TYPE_SAVE
        nodeTypeData[indexMacr] = hNodeType[blockedScalarIndex];
        #endif //NODE_TYPE_SAVE
    }


    #if defined BC_FORCES && defined SAVE_BC_FORCES
        linearizeBlockedScalars(vtkToBlocked, {h_BC_Fx, h_BC_Fy, h_BC_Fz});
    #endif // BC_FORCES && SAVE_BC_FORCES


    #ifdef SAVE_LOCAL_FORCES
        linearizeBlockedScalars(vtkToBlocked, {h_Local_Fx, h_Local_Fy, h_Local_Fz});
    #endif // SAVE_LOCAL_FORCES


    // Linearize source term arrays (same block→global pattern)
    #ifdef SAVE_LOCAL_FORCES
        // Reuse the precomputed global-to-blocked permutation for each source.
        #define LINEARIZE_SOURCE_SCALAR(hSrc) \
            linearizeBlockedScalars(vtkToBlocked, {hSrc})

        #ifdef SECOND_DIST
        LINEARIZE_SOURCE_SCALAR(h_Source_C);
        #endif
        #ifdef PHI_DIST
        LINEARIZE_SOURCE_SCALAR(h_Source_Phi);
        #endif
        #ifdef LAMBDA_DIST
        LINEARIZE_SOURCE_SCALAR(h_Source_Lambda);
        #endif
        #ifdef CONFORMATION_TENSOR
            #ifdef A_XX_DIST
        LINEARIZE_SOURCE_SCALAR(h_Source_Gxx);
            #endif
            #ifdef A_XY_DIST
        LINEARIZE_SOURCE_SCALAR(h_Source_Gxy);
            #endif
            #ifdef A_XZ_DIST
        LINEARIZE_SOURCE_SCALAR(h_Source_Gxz);
            #endif
            #ifdef A_YY_DIST
        LINEARIZE_SOURCE_SCALAR(h_Source_Gyy);
            #endif
            #ifdef A_YZ_DIST
        LINEARIZE_SOURCE_SCALAR(h_Source_Gyz);
            #endif
            #ifdef A_ZZ_DIST
        LINEARIZE_SOURCE_SCALAR(h_Source_Gzz);
            #endif
        #endif //CONFORMATION_TENSOR

        #undef LINEARIZE_SOURCE_SCALAR
    #endif //SAVE_LOCAL_FORCES


    // Names of files
    std::string strFileRho, strFileUx, strFileUy, strFileUz; 
    std::string strFileOmega;
    std::string strFileC;
    std::string strFilePhi;
    std::string strFileLambda;
    std::string strFileBc; 
    std::string strFileFx, strFileFy, strFileFz;
    std::string strFileLocalFx, strFileLocalFy, strFileLocalFz;
    std::string strFileSourceC, strFileSourcePhi, strFileSourceLambda;
    std::string strFileSourceGxx, strFileSourceGxy, strFileSourceGxz, strFileSourceGyy, strFileSourceGyz, strFileSourceGzz;
    std::string strFileAxx, strFileAxy, strFileAxz, strFileAyy, strFileAyz, strFileAzz;


    if (VTK_SAVE){
        std::string strFileVtk, strFileVtr;
        strFileVtk = getVarFilename("vtk", nSteps, ".vtk");
        while (savingMacrVtk) std::this_thread::sleep_for(std::chrono::milliseconds(1));
        updateVtkSeries(strFileVtk, nSteps);
        
        SaveDataParams saveVarVtkParams;
        saveVarVtkParams.vtkFilename = strFileVtk.c_str();
        saveVarVtkParams.h_rho = rho;
        saveVarVtkParams.h_ux = ux;
        saveVarVtkParams.h_uy = uy;
        saveVarVtkParams.h_uz = uz;
        #ifdef OMEGA_FIELD
        saveVarVtkParams.h_omega = omega;
        #endif
        #ifdef SECOND_DIST
        saveVarVtkParams.h_C = C;
        #endif
        #ifdef PHI_DIST
        saveVarVtkParams.h_phi = phi;
        #endif
        #ifdef LAMBDA_DIST
        saveVarVtkParams.h_lambda = lambda;
        #endif
        #ifdef A_XX_DIST
        saveVarVtkParams.h_Axx = Axx;
        #endif
        #ifdef A_XY_DIST
        saveVarVtkParams.h_Axy = Axy;
        #endif
        #ifdef A_XZ_DIST
        saveVarVtkParams.h_Axz = Axz;
        #endif
        #ifdef A_YY_DIST
        saveVarVtkParams.h_Ayy = Ayy;
        #endif
        #ifdef A_YZ_DIST
        saveVarVtkParams.h_Ayz = Ayz;
        #endif
        #ifdef A_ZZ_DIST
        saveVarVtkParams.h_Azz = Azz;
        #endif
        #if NODE_TYPE_SAVE
        saveVarVtkParams.h_nodeTypeSave = nodeTypeData;
        #endif
        #if defined(BC_FORCES) && defined(SAVE_BC_FORCES)
        saveVarVtkParams.h_BC_Fx = h_BC_Fx;
        saveVarVtkParams.h_BC_Fy = h_BC_Fy;
        saveVarVtkParams.h_BC_Fz = h_BC_Fz;
        #endif
        #ifdef SAVE_LOCAL_FORCES
        saveVarVtkParams.h_Local_Fx = h_Local_Fx;
        saveVarVtkParams.h_Local_Fy = h_Local_Fy;
        saveVarVtkParams.h_Local_Fz = h_Local_Fz;
            #ifdef SECOND_DIST
        saveVarVtkParams.h_Source_C = h_Source_C;
            #endif
            #ifdef PHI_DIST
        saveVarVtkParams.h_Source_Phi = h_Source_Phi;
            #endif
            #ifdef LAMBDA_DIST
        saveVarVtkParams.h_Source_Lambda = h_Source_Lambda;
            #endif
            #ifdef CONFORMATION_TENSOR
                #ifdef A_XX_DIST
        saveVarVtkParams.h_Source_Gxx = h_Source_Gxx;
                #endif
                #ifdef A_XY_DIST
        saveVarVtkParams.h_Source_Gxy = h_Source_Gxy;
                #endif
                #ifdef A_XZ_DIST
        saveVarVtkParams.h_Source_Gxz = h_Source_Gxz;
                #endif
                #ifdef A_YY_DIST
        saveVarVtkParams.h_Source_Gyy = h_Source_Gyy;
                #endif
                #ifdef A_YZ_DIST
        saveVarVtkParams.h_Source_Gyz = h_Source_Gyz;
                #endif
                #ifdef A_ZZ_DIST
        saveVarVtkParams.h_Source_Gzz = h_Source_Gzz;
                #endif
            #endif //CONFORMATION_TENSOR
        #endif
        saveVarVtkParams.nSteps = nSteps;
        saveVarVtkParams.savingMacrVtk = &savingMacrVtk;
        
        saveVarVTK(&saveVarVtkParams);
    }
    if (BIN_SAVE){
        strFileRho = getVarFilename("rho", nSteps, ".bin");
        strFileUx = getVarFilename("ux", nSteps, ".bin");
        strFileUy = getVarFilename("uy", nSteps, ".bin");
        strFileUz = getVarFilename("uz", nSteps, ".bin");

        #ifdef OMEGA_FIELD
        strFileOmega = getVarFilename("omega", nSteps, ".bin");
        #endif //OMEGA_FIELD
        #ifdef SECOND_DIST 
        strFileC = getVarFilename("C", nSteps, ".bin");
        #endif //SECOND_DIST
        #ifdef PHI_DIST 
        strFilePhi = getVarFilename("phi", nSteps, ".bin");
        #endif //PHI_DIST
        #ifdef LAMBDA_DIST
        strFileLambda = getVarFilename("lambda", nSteps, ".bin");
        #endif //LAMBDA_DIST
        #ifdef A_XX_DIST 
        strFileAxx = getVarFilename("Axx", nSteps, ".bin");
        #endif //A_XX_DIST
        #ifdef A_XY_DIST 
        strFileAxy = getVarFilename("Axy", nSteps, ".bin");
        #endif //A_XY_DIST
        #ifdef A_XZ_DIST 
        strFileAxz = getVarFilename("Axz", nSteps, ".bin");
        #endif //A_XZ_DIST
        #ifdef A_YY_DIST 
        strFileAyy = getVarFilename("Ayy", nSteps, ".bin");
        #endif //A_YY_DIST
        #ifdef A_YZ_DIST 
        strFileAyz = getVarFilename("Ayz", nSteps, ".bin");
        #endif //A_YZ_DIST
        #ifdef A_ZZ_DIST 
        strFileAzz = getVarFilename("Azz", nSteps, ".bin");
        #endif //A_ZZ_DIST
        #if NODE_TYPE_SAVE
        strFileBc = getVarFilename("bc", nSteps, ".bin");
        #endif //NODE_TYPE_SAVE
        #if defined BC_FORCES && defined SAVE_BC_FORCES
        strFileFx = getVarFilename("fx", nSteps, ".bin");
        strFileFy = getVarFilename("fy", nSteps, ".bin");
        strFileFz = getVarFilename("fz", nSteps, ".bin");
        #endif //BC_FORCES &&  SAVE_BC_FORCES
        #ifdef SAVE_LOCAL_FORCES
        strFileLocalFx = getVarFilename("local_fx", nSteps, ".bin");
        strFileLocalFy = getVarFilename("local_fy", nSteps, ".bin");
        strFileLocalFz = getVarFilename("local_fz", nSteps, ".bin");
            #ifdef SECOND_DIST
        strFileSourceC = getVarFilename("source_C", nSteps, ".bin");
            #endif
            #ifdef PHI_DIST
        strFileSourcePhi = getVarFilename("source_Phi", nSteps, ".bin");
            #endif
            #ifdef LAMBDA_DIST
        strFileSourceLambda = getVarFilename("source_Lambda", nSteps, ".bin");
            #endif
            #ifdef CONFORMATION_TENSOR
                #ifdef A_XX_DIST
        strFileSourceGxx = getVarFilename("source_Gxx", nSteps, ".bin");
                #endif
                #ifdef A_XY_DIST
        strFileSourceGxy = getVarFilename("source_Gxy", nSteps, ".bin");
                #endif
                #ifdef A_XZ_DIST
        strFileSourceGxz = getVarFilename("source_Gxz", nSteps, ".bin");
                #endif
                #ifdef A_YY_DIST
        strFileSourceGyy = getVarFilename("source_Gyy", nSteps, ".bin");
                #endif
                #ifdef A_YZ_DIST
        strFileSourceGyz = getVarFilename("source_Gyz", nSteps, ".bin");
                #endif
                #ifdef A_ZZ_DIST
        strFileSourceGzz = getVarFilename("source_Gzz", nSteps, ".bin");
                #endif
            #endif //CONFORMATION_TENSOR
        #endif //SAVE_LOCAL_FORCES
        // saving files
        std::vector<const void*> varArray;
        std::vector<std::string> fileArray;
        std::vector<size_t> sizeArray;

        const auto addScalar = [&](const void* data, const std::string& file, size_t bytes = MEM_SIZE_SCALAR) {
            varArray.push_back(data);
            fileArray.push_back(file);
            sizeArray.push_back(bytes);
        };

        addScalar(rho, strFileRho);
        addScalar(ux, strFileUx);
        addScalar(uy, strFileUy);
        addScalar(uz, strFileUz);
        #ifdef OMEGA_FIELD
        addScalar(omega, strFileOmega);
        #endif
        #ifdef SECOND_DIST
        addScalar(C, strFileC);
        #endif //SECOND_DIST
        #ifdef PHI_DIST
        addScalar(phi, strFilePhi);
        #endif //PHI_DIST
        #ifdef LAMBDA_DIST
        addScalar(lambda, strFileLambda);
        #endif //LAMBDA_DIST
        #ifdef A_XX_DIST
        addScalar(Axx, strFileAxx);
        #endif
        #ifdef A_XY_DIST
        addScalar(Axy, strFileAxy);
        #endif
        #ifdef A_XZ_DIST
        addScalar(Axz, strFileAxz);
        #endif
        #ifdef A_YY_DIST
        addScalar(Ayy, strFileAyy);
        #endif
        #ifdef A_YZ_DIST
        addScalar(Ayz, strFileAyz);
        #endif
        #ifdef A_ZZ_DIST
        addScalar(Azz, strFileAzz);
        #endif

        #if NODE_TYPE_SAVE
        addScalar(nodeTypeData, strFileBc, sizeof(unsigned int) * NUMBER_LBM_NODES);
        #endif
        #if defined(BC_FORCES) && defined(SAVE_BC_FORCES)
        addScalar(h_BC_Fx, strFileFx);
        addScalar(h_BC_Fy, strFileFy);
        addScalar(h_BC_Fz, strFileFz);
        #endif
        #ifdef SAVE_LOCAL_FORCES
        addScalar(h_Local_Fx, strFileLocalFx);
        addScalar(h_Local_Fy, strFileLocalFy);
        addScalar(h_Local_Fz, strFileLocalFz);
            #ifdef SECOND_DIST
        addScalar(h_Source_C, strFileSourceC);
            #endif
            #ifdef PHI_DIST
        addScalar(h_Source_Phi, strFileSourcePhi);
            #endif
            #ifdef LAMBDA_DIST
        addScalar(h_Source_Lambda, strFileSourceLambda);
            #endif
            #ifdef CONFORMATION_TENSOR
                #ifdef A_XX_DIST
        addScalar(h_Source_Gxx, strFileSourceGxx);
                #endif
                #ifdef A_XY_DIST
        addScalar(h_Source_Gxy, strFileSourceGxy);
                #endif
                #ifdef A_XZ_DIST
        addScalar(h_Source_Gxz, strFileSourceGxz);
                #endif
                #ifdef A_YY_DIST
        addScalar(h_Source_Gyy, strFileSourceGyy);
                #endif
                #ifdef A_YZ_DIST
        addScalar(h_Source_Gyz, strFileSourceGyz);
                #endif
                #ifdef A_ZZ_DIST
        addScalar(h_Source_Gzz, strFileSourceGzz);
                #endif
            #endif //CONFORMATION_TENSOR
        #endif
        for(size_t i = 0; i < varArray.size(); ++i){
            while (savingMacrBin[i]) std::this_thread::yield();
            saveVarBin(fileArray[i], varArray[i], sizeArray[i], false, savingMacrBin[i]);
        }
    }
}

void saveVarBin(
    std::string strFile, 
    const void* var,
    size_t memSize,
    bool append,
    std::atomic<bool>& savingMacrBin)
{
    savingMacrBin = true;
    enqueueSaveTask({[=]() {
        FILE* outFile = nullptr;
        if(append)
            outFile = fopen(strFile.c_str(), "ab");
        else
            outFile = fopen(strFile.c_str(), "wb");
        if(outFile != nullptr)
        {
            fwrite(var, memSize, 1, outFile);
            fclose(outFile);
        }
        else
        {
            printf("Error saving \"%s\" \nProbably wrong path!\n", strFile.c_str());
        }
    }, &savingMacrBin});
}


std::vector<dfloat> convertPointToCellScalar(
    const dfloat* pointField, size_t NX, size_t NY, size_t NZ)
{
    size_t Ncells = (NX-1)*(NY-1)*(NZ-1);
    std::vector<dfloat> cellField(Ncells, 0.0f);

    for (size_t z=0; z<NZ-1; z++)
    for (size_t y=0; y<NY-1; y++)
    for (size_t x=0; x<NX-1; x++) {
        size_t cidx = x + y*(NX-1) + z*(NX-1)*(NY-1);
        dfloat sum=0.0f;
        for(int dz=0; dz<=1; dz++)
        for(int dy=0; dy<=1; dy++)
        for(int dx=0; dx<=1; dx++)
            sum += pointField[idxScalarGlobal(x+dx, y+dy, z+dz)];
        cellField[cidx] = sum/8.0f;
    }
    return cellField;
}

std::vector<dfloat3> convertPointToCellVector(
    const dfloat* ux, const dfloat* uy, const dfloat* uz,
    size_t NX, size_t NY, size_t NZ)
{
    size_t Ncells = (NX-1)*(NY-1)*(NZ-1);
    std::vector<dfloat3> cellField(Ncells);

    for (size_t z=0; z<NZ-1; z++)
    for (size_t y=0; y<NY-1; y++)
    for (size_t x=0; x<NX-1; x++) {
        size_t cidx = x + y*(NX-1) + z*(NX-1)*(NY-1);
        dfloat sumx=0.0f, sumy=0.0f, sumz=0.0f;
        for(int dz=0; dz<=1; dz++)
        for(int dy=0; dy<=1; dy++)
        for(int dx=0; dx<=1; dx++) {
            size_t pidx = idxScalarGlobal(x+dx, y+dy, z+dz);
            sumx += ux[pidx]; sumy += uy[pidx]; sumz += uz[pidx];
        }
        cellField[cidx] = { sumx/8.0f, sumy/8.0f, sumz/8.0f };
    }
    return cellField;
}

std::vector<dfloat6> convertPointToCellTensor6(
    const dfloat* Axx, const dfloat* Ayy, const dfloat* Azz,
    const dfloat* Axy, const dfloat* Axz, const dfloat* Ayz,
    size_t NX, size_t NY, size_t NZ)
{
    size_t Ncells = (NX-1)*(NY-1)*(NZ-1);
    std::vector<dfloat6> cellField(Ncells);

    for (size_t z=0; z<NZ-1; z++)
    for (size_t y=0; y<NY-1; y++)
    for (size_t x=0; x<NX-1; x++) {
        size_t cidx = x + y*(NX-1) + z*(NX-1)*(NY-1);
        dfloat sumxx=0,sumyy=0,sumzz=0,sumxy=0,sumxz=0,sumyz=0;
        for(int dz=0; dz<=1; dz++)
        for(int dy=0; dy<=1; dy++)
        for(int dx=0; dx<=1; dx++) {
            size_t pidx = idxScalarGlobal(x+dx, y+dy, z+dz);
            sumxx += Axx[pidx]; sumyy += Ayy[pidx]; sumzz += Azz[pidx];
            sumxy += Axy[pidx]; sumxz += Axz[pidx]; sumyz += Ayz[pidx];
        }
        cellField[cidx] = { sumxx/8.0f, sumyy/8.0f, sumzz/8.0f,
                            sumxy/8.0f, sumxz/8.0f, sumyz/8.0f };
    }
    return cellField;
}

std::vector<unsigned int> convertPointToCellIntMode(
    const unsigned int* pointField, size_t NX, size_t NY, size_t NZ)
{
    size_t Ncells = (NX-1)*(NY-1)*(NZ-1);
    std::vector<unsigned int> cellField(Ncells, 0);

    for (size_t z=0; z<NZ-1; z++)
    for (size_t y=0; y<NY-1; y++)
    for (size_t x=0; x<NX-1; x++) {
        size_t cidx = x + y*(NX-1) + z*(NX-1)*(NY-1);
        std::map<unsigned int, int> counts;

        for(int dz=0; dz<=1; dz++)
        for(int dy=0; dy<=1; dy++)
        for(int dx=0; dx<=1; dx++)
            counts[pointField[idxScalarGlobal(x+dx, y+dy, z+dz)]]++;

        unsigned int mode = 0;
        int maxCount = 0;
        for(auto &kv : counts)
            if(kv.second>maxCount) { maxCount=kv.second; mode=kv.first; }

        cellField[cidx] = mode;
    }
    return cellField;
}

void saveVarVTK(const SaveDataParams* params)
{
    // Unpack parameters from struct
    std::string filename = params->vtkFilename;
    dfloat* rho = params->h_rho;
    dfloat* ux = params->h_ux;
    dfloat* uy = params->h_uy;
    dfloat* uz = params->h_uz;
    #ifdef OMEGA_FIELD
    dfloat* omega = params->h_omega;
    #endif
    #ifdef SECOND_DIST
    dfloat* C = params->h_C;
    #endif
    #ifdef PHI_DIST
    dfloat* phi = params->h_phi;
    #endif
    #ifdef LAMBDA_DIST
    dfloat* lambda = params->h_lambda;
    #endif
    #ifdef A_XX_DIST
    dfloat* Axx = params->h_Axx;
    #endif
    #ifdef A_XY_DIST
    dfloat* Axy = params->h_Axy;
    #endif
    #ifdef A_XZ_DIST
    dfloat* Axz = params->h_Axz;
    #endif
    #ifdef A_YY_DIST
    dfloat* Ayy = params->h_Ayy;
    #endif
    #ifdef A_YZ_DIST
    dfloat* Ayz = params->h_Ayz;
    #endif
    #ifdef A_ZZ_DIST
    dfloat* Azz = params->h_Azz;
    #endif
    #if NODE_TYPE_SAVE
    unsigned int* nodeTypeData = params->h_nodeTypeSave;
    #endif
    #if defined(BC_FORCES) && defined(SAVE_BC_FORCES)
    dfloat* h_BC_Fx = params->h_BC_Fx;
    dfloat* h_BC_Fy = params->h_BC_Fy;
    dfloat* h_BC_Fz = params->h_BC_Fz;
    #endif
    #ifdef SAVE_LOCAL_FORCES
    dfloat* h_Local_Fx = params->h_Local_Fx;
    dfloat* h_Local_Fy = params->h_Local_Fy;
    dfloat* h_Local_Fz = params->h_Local_Fz;
        #ifdef SECOND_DIST
    dfloat* h_Source_C = params->h_Source_C;
        #endif
        #ifdef PHI_DIST
    dfloat* h_Source_Phi = params->h_Source_Phi;
        #endif
        #ifdef LAMBDA_DIST
    dfloat* h_Source_Lambda = params->h_Source_Lambda;
        #endif
        #ifdef CONFORMATION_TENSOR
            #ifdef A_XX_DIST
    dfloat* h_Source_Gxx = params->h_Source_Gxx;
            #endif
            #ifdef A_XY_DIST
    dfloat* h_Source_Gxy = params->h_Source_Gxy;
            #endif
            #ifdef A_XZ_DIST
    dfloat* h_Source_Gxz = params->h_Source_Gxz;
            #endif
            #ifdef A_YY_DIST
    dfloat* h_Source_Gyy = params->h_Source_Gyy;
            #endif
            #ifdef A_YZ_DIST
    dfloat* h_Source_Gyz = params->h_Source_Gyz;
            #endif
            #ifdef A_ZZ_DIST
    dfloat* h_Source_Gzz = params->h_Source_Gzz;
            #endif
        #endif //CONFORMATION_TENSOR
    #endif
    unsigned int nSteps = params->nSteps;
    std::atomic<bool>& savingMacrVtk = *params->savingMacrVtk;

    // Function body starts here
    const char* VTK_TYPE = nullptr;

    if (std::is_same<dfloat, float>::value) {
        VTK_TYPE = "float";
    } else if (std::is_same<dfloat, double>::value) {
        VTK_TYPE = "double";
    }

    if(!CELLDATA_SAVE){
        //printf("Saving VTK in POINT_DATA format");
        savingMacrVtk = true;
        enqueueSaveTask({[=]() {
            const size_t N = NX*NY*NZ;
            std::ofstream ofs(filename, std::ios::binary);
            if (!ofs) throw std::runtime_error("Cannot open " + filename);

            //Header 
            ofs << "# vtk DataFile Version 3.0\n"
                << "LBM output (binary)\n"
                << "BINARY\n"
                << "DATASET STRUCTURED_POINTS\n"
                << "DIMENSIONS " << NX << " " << NY << " " << NZ << "\n"
                << "ORIGIN 0 0 0\n"
                << "SPACING 1 1 1\n";
            ofs << "POINT_DATA " << N << "\n";
            ofs << "SCALARS rho " << VTK_TYPE << " 1\n"
                << "LOOKUP_TABLE default\n";
            writeBigEndian(ofs, rho, N);

            ofs << "VECTORS velocity " << VTK_TYPE << "\n";
            for (size_t i = 0; i < N; ++i) {
                dfloat v[3] = { ux[i]/F_M_I_SCALE, uy[i]/F_M_I_SCALE, uz[i]/F_M_I_SCALE};
                writeBigEndian(ofs, v, 3);
            }

            #ifdef OMEGA_FIELD
                ofs << "SCALARS omega " << VTK_TYPE << " 1\n"
                    << "LOOKUP_TABLE default\n";
                writeBigEndian(ofs, omega, N);
            #endif //OMEGA_FIELD

            #ifdef SECOND_DIST
                ofs << "SCALARS C " << VTK_TYPE << " 1\n"
                    << "LOOKUP_TABLE default\n";
                writeBigEndian(ofs, C, N);
            #endif //SECOND_DIST
            
            #ifdef PHI_DIST
                ofs << "SCALARS PHI " << VTK_TYPE << " 1\n"
                    << "LOOKUP_TABLE default\n";
                writeBigEndian(ofs, phi, N);
            #endif //PHI_DIST

            #ifdef LAMBDA_DIST
                ofs << "SCALARS lambda " << VTK_TYPE << " 1\n"
                    << "LOOKUP_TABLE default\n";
                writeBigEndian(ofs, lambda, N);
            #endif //LAMBDA_DIST

            #ifdef CONFORMATION_TENSOR
                ofs << "TENSORS6 Aij " << VTK_TYPE << "\n";
                for (size_t i = 0; i < N; ++i) {
                    dfloat tensor[6] = {
                        Axx[i], Ayy[i], Azz[i],
                        Axy[i], Axz[i], Ayz[i]
                    };
                    writeBigEndian(ofs, tensor, 6);
                }
            #endif //CONFORMATION_TENSOR

            #ifdef SAVE_BC_FORCES
                ofs << "VECTORS forces " << VTK_TYPE << "\n";
                for (size_t i = 0; i < N; ++i) {
                    dfloat f[3] = { h_BC_Fx[i], h_BC_Fy[i], h_BC_Fz[i] };
                    writeBigEndian(ofs, f, 3);
                }
            #endif //SAVE_BC_FORCES

            #ifdef SAVE_LOCAL_FORCES
                ofs << "VECTORS local_forces " << VTK_TYPE << "\n";
                for (size_t i = 0; i < N; ++i) {
                    dfloat f[3] = { h_Local_Fx[i], h_Local_Fy[i], h_Local_Fz[i] };
                    writeBigEndian(ofs, f, 3);
                }
                #ifdef SECOND_DIST
                ofs << "SCALARS source_C " << VTK_TYPE << " 1\n"
                    << "LOOKUP_TABLE default\n";
                writeBigEndian(ofs, h_Source_C, N);
                #endif
                #ifdef PHI_DIST
                ofs << "SCALARS source_Phi " << VTK_TYPE << " 1\n"
                    << "LOOKUP_TABLE default\n";
                writeBigEndian(ofs, h_Source_Phi, N);
                #endif
                #ifdef LAMBDA_DIST
                ofs << "SCALARS source_Lambda " << VTK_TYPE << " 1\n"
                    << "LOOKUP_TABLE default\n";
                writeBigEndian(ofs, h_Source_Lambda, N);
                #endif
                #ifdef CONFORMATION_TENSOR
                    #if defined(A_XX_DIST) && defined(A_YY_DIST) && defined(A_ZZ_DIST) && defined(A_XY_DIST) && defined(A_XZ_DIST) && defined(A_YZ_DIST)
                ofs << "TENSORS6 source_Gij " << VTK_TYPE << "\n";
                for (size_t i = 0; i < N; ++i) {
                    dfloat tensor[6] = {
                        h_Source_Gxx[i], h_Source_Gyy[i], h_Source_Gzz[i],
                        h_Source_Gxy[i], h_Source_Gxz[i], h_Source_Gyz[i]
                    };
                    writeBigEndian(ofs, tensor, 6);
                }
                    #else
                        #ifdef A_XX_DIST
                ofs << "SCALARS source_Gxx " << VTK_TYPE << " 1\n"
                    << "LOOKUP_TABLE default\n";
                writeBigEndian(ofs, h_Source_Gxx, N);
                        #endif
                        #ifdef A_XY_DIST
                ofs << "SCALARS source_Gxy " << VTK_TYPE << " 1\n"
                    << "LOOKUP_TABLE default\n";
                writeBigEndian(ofs, h_Source_Gxy, N);
                        #endif
                        #ifdef A_XZ_DIST
                ofs << "SCALARS source_Gxz " << VTK_TYPE << " 1\n"
                    << "LOOKUP_TABLE default\n";
                writeBigEndian(ofs, h_Source_Gxz, N);
                        #endif
                        #ifdef A_YY_DIST
                ofs << "SCALARS source_Gyy " << VTK_TYPE << " 1\n"
                    << "LOOKUP_TABLE default\n";
                writeBigEndian(ofs, h_Source_Gyy, N);
                        #endif
                        #ifdef A_YZ_DIST
                ofs << "SCALARS source_Gyz " << VTK_TYPE << " 1\n"
                    << "LOOKUP_TABLE default\n";
                writeBigEndian(ofs, h_Source_Gyz, N);
                        #endif
                        #ifdef A_ZZ_DIST
                ofs << "SCALARS source_Gzz " << VTK_TYPE << " 1\n"
                    << "LOOKUP_TABLE default\n";
                writeBigEndian(ofs, h_Source_Gzz, N);
                        #endif
                    #endif //all six components
                #endif //CONFORMATION_TENSOR
            #endif //SAVE_LOCAL_FORCES

            #if NODE_TYPE_SAVE
                ofs << "SCALARS bc unsigned_int 1\n"
                    << "LOOKUP_TABLE default\n";
                writeBigEndian(ofs, nodeTypeData, N);
            #endif //NODE_TYPE_SAVE
        }, &savingMacrVtk});
    }else{ 
        //printf("Saving VTK in CELL_DATA format");
        savingMacrVtk = true;
        enqueueSaveTask({[=]() {
            const size_t Ncells = (NX-1)*(NY-1)*(NZ-1);
            std::ofstream ofs(filename, std::ios::binary);
            if (!ofs) throw std::runtime_error("Cannot open " + filename);

            //Header 
            ofs << "# vtk DataFile Version 3.0\n"
                << "LBM output (binary)\n"
                << "BINARY\n"
                << "DATASET STRUCTURED_POINTS\n"
                << "DIMENSIONS " << NX << " " << NY << " " << NZ << "\n"
                << "ORIGIN 0 0 0\n"
                << "SPACING 1 1 1\n";
            ofs << "CELL_DATA " << Ncells << "\n";
            auto rho_cell = convertPointToCellScalar(rho,NX,NY,NZ);
            ofs << "SCALARS rho  " << VTK_TYPE << " 1\n"
                << "LOOKUP_TABLE default\n";
            writeBigEndian(ofs, rho_cell.data(), rho_cell.size());

            auto vel_cell = convertPointToCellVector(ux,uy,uz,NX,NY,NZ);
            ofs << "VECTORS velocity  " << VTK_TYPE << "\n";
            for(size_t i=0;i<Ncells;i++){
                dfloat v[3] = { vel_cell[i].x/F_M_I_SCALE,
                            vel_cell[i].y/F_M_I_SCALE,
                            vel_cell[i].z/F_M_I_SCALE };
                writeBigEndian(ofs,v,3);
            }

            #ifdef OMEGA_FIELD
                auto omega_cell = convertPointToCellScalar(omega,NX,NY,NZ);
                ofs << "SCALARS omega  " << VTK_TYPE << " 1\n"
                    << "LOOKUP_TABLE default\n";
                writeBigEndian(ofs, omega_cell.data(), omega_cell.size());
            #endif //OMEGA_FIELD

            #ifdef SECOND_DIST
                auto C_cell = convertPointToCellScalar(C,NX,NY,NZ);
                ofs << "SCALARS C  " << VTK_TYPE << " 1\n"
                    << "LOOKUP_TABLE default\n";
                writeBigEndian(ofs, C_cell.data(), Ncells);
            #endif //SECOND_DIST

            #ifdef PHI_DIST
                auto PHI_cell = convertPointToCellScalar(phi,NX,NY,NZ);
                ofs << "SCALARS PHI  " << VTK_TYPE << " 1\n"
                    << "LOOKUP_TABLE default\n";
                writeBigEndian(ofs, PHI_cell.data(), Ncells);
            #endif //PHI_DIST

            #ifdef LAMBDA_DIST
                auto lambda_cell = convertPointToCellScalar(lambda,NX,NY,NZ);
                ofs << "SCALARS lambda  " << VTK_TYPE << " 1\n"
                    << "LOOKUP_TABLE default\n";
                writeBigEndian(ofs, lambda_cell.data(), Ncells);
            #endif //LAMBDA_DIST

            #ifdef CONFORMATION_TENSOR
                auto A_cell = convertPointToCellTensor6(Axx,Ayy,Azz,Axy,Axz,Ayz,NX,NY,NZ);
                ofs << "TENSORS6 Aij  " << VTK_TYPE << "\n";
                for (size_t i = 0; i < Ncells; ++i) {
                    dfloat tensor[6] = {
                        A_cell[i].xx,A_cell[i].yy,A_cell[i].zz,
                        A_cell[i].xy,A_cell[i].xz,A_cell[i].yz
                    };
                    writeBigEndian(ofs, tensor, 6);
                }
            #endif //CONFORMATION_TENSOR

            #ifdef SAVE_BC_FORCES
                auto f_cell = convertPointToCellVector(h_BC_Fx, h_BC_Fy, h_BC_Fz, NX, NY, NZ);
                ofs << "VECTORS forces  " << VTK_TYPE << "\n";
                for (size_t i = 0; i < Ncells; ++i) {
                    dfloat f[3] = { f_cell[i].x, f_cell[i].y, f_cell[i].z };
                    writeBigEndian(ofs, f, 3);
                }
            #endif //SAVE_BC_FORCES

            #ifdef SAVE_LOCAL_FORCES
                auto local_f_cell = convertPointToCellVector(h_Local_Fx, h_Local_Fy, h_Local_Fz, NX, NY, NZ);
                ofs << "VECTORS local_forces  " << VTK_TYPE << "\n";
                for (size_t i = 0; i < Ncells; ++i) {
                    dfloat f[3] = { local_f_cell[i].x, local_f_cell[i].y, local_f_cell[i].z };
                    writeBigEndian(ofs, f, 3);
                }
                #ifdef SECOND_DIST
                {
                    auto src_cell = convertPointToCellScalar(h_Source_C, NX, NY, NZ);
                    ofs << "SCALARS source_C  " << VTK_TYPE << " 1\n"
                        << "LOOKUP_TABLE default\n";
                    writeBigEndian(ofs, src_cell.data(), Ncells);
                }
                #endif
                #ifdef PHI_DIST
                {
                    auto src_cell = convertPointToCellScalar(h_Source_Phi, NX, NY, NZ);
                    ofs << "SCALARS source_Phi  " << VTK_TYPE << " 1\n"
                        << "LOOKUP_TABLE default\n";
                    writeBigEndian(ofs, src_cell.data(), Ncells);
                }
                #endif
                #ifdef LAMBDA_DIST
                {
                    auto src_cell = convertPointToCellScalar(h_Source_Lambda, NX, NY, NZ);
                    ofs << "SCALARS source_Lambda  " << VTK_TYPE << " 1\n"
                        << "LOOKUP_TABLE default\n";
                    writeBigEndian(ofs, src_cell.data(), Ncells);
                }
                #endif
                #ifdef CONFORMATION_TENSOR
                    #if defined(A_XX_DIST) && defined(A_YY_DIST) && defined(A_ZZ_DIST) && defined(A_XY_DIST) && defined(A_XZ_DIST) && defined(A_YZ_DIST)
                {
                    auto G_cell = convertPointToCellTensor6(h_Source_Gxx, h_Source_Gyy, h_Source_Gzz,
                                                             h_Source_Gxy, h_Source_Gxz, h_Source_Gyz, NX, NY, NZ);
                    ofs << "TENSORS6 source_Gij  " << VTK_TYPE << "\n";
                    for (size_t i = 0; i < Ncells; ++i) {
                        dfloat tensor[6] = {
                            G_cell[i].xx, G_cell[i].yy, G_cell[i].zz,
                            G_cell[i].xy, G_cell[i].xz, G_cell[i].yz
                        };
                        writeBigEndian(ofs, tensor, 6);
                    }
                }
                    #else
                        #ifdef A_XX_DIST
                {
                    auto src_cell = convertPointToCellScalar(h_Source_Gxx, NX, NY, NZ);
                    ofs << "SCALARS source_Gxx  " << VTK_TYPE << " 1\n"
                        << "LOOKUP_TABLE default\n";
                    writeBigEndian(ofs, src_cell.data(), Ncells);
                }
                        #endif
                        #ifdef A_XY_DIST
                {
                    auto src_cell = convertPointToCellScalar(h_Source_Gxy, NX, NY, NZ);
                    ofs << "SCALARS source_Gxy  " << VTK_TYPE << " 1\n"
                        << "LOOKUP_TABLE default\n";
                    writeBigEndian(ofs, src_cell.data(), Ncells);
                }
                        #endif
                        #ifdef A_XZ_DIST
                {
                    auto src_cell = convertPointToCellScalar(h_Source_Gxz, NX, NY, NZ);
                    ofs << "SCALARS source_Gxz  " << VTK_TYPE << " 1\n"
                        << "LOOKUP_TABLE default\n";
                    writeBigEndian(ofs, src_cell.data(), Ncells);
                }
                        #endif
                        #ifdef A_YY_DIST
                {
                    auto src_cell = convertPointToCellScalar(h_Source_Gyy, NX, NY, NZ);
                    ofs << "SCALARS source_Gyy  " << VTK_TYPE << " 1\n"
                        << "LOOKUP_TABLE default\n";
                    writeBigEndian(ofs, src_cell.data(), Ncells);
                }
                        #endif
                        #ifdef A_YZ_DIST
                {
                    auto src_cell = convertPointToCellScalar(h_Source_Gyz, NX, NY, NZ);
                    ofs << "SCALARS source_Gyz  " << VTK_TYPE << " 1\n"
                        << "LOOKUP_TABLE default\n";
                    writeBigEndian(ofs, src_cell.data(), Ncells);
                }
                        #endif
                        #ifdef A_ZZ_DIST
                {
                    auto src_cell = convertPointToCellScalar(h_Source_Gzz, NX, NY, NZ);
                    ofs << "SCALARS source_Gzz  " << VTK_TYPE << " 1\n"
                        << "LOOKUP_TABLE default\n";
                    writeBigEndian(ofs, src_cell.data(), Ncells);
                }
                        #endif
                    #endif //all six
                #endif //CONFORMATION_TENSOR
            #endif //SAVE_LOCAL_FORCES

            #if NODE_TYPE_SAVE
                auto bc_cell = convertPointToCellIntMode(nodeTypeData, NX, NY, NZ);
                ofs << "SCALARS bc unsigned_int 1\n"
                    << "LOOKUP_TABLE default\n";
                writeBigEndian(ofs, bc_cell.data(), Ncells);
            #endif //NODE_TYPE_SAVE
        }, &savingMacrVtk});
    }  
}

std::string getVarFilename(
    const std::string varName, 
    unsigned int step,
    const std::string ext)
{
    unsigned int n_zeros = 0, pot_10 = 10;
    unsigned int aux1 = 1000000;  // 6 numbers on step
    // calculate number of zeros
    if (step != 0)
        for (n_zeros = 0; step * pot_10 < aux1; pot_10 *= 10)
            n_zeros++;
    else
        n_zeros = 6;

    // generates the file name as "PATH_FILES/id/id_varName000000.bin"
    
    std::filesystem::path baseDir = folderSetup();

    std::string baseName = ID_SIM + std::string("_") + varName;

    std::string strFile = (baseDir / baseName).string();

    for (unsigned int i = 0; i < n_zeros; i++)
        strFile += "0";
    strFile += std::to_string(step);
    strFile += ext;

    return strFile;
}

static void appendFluidProps(std::ostringstream& strSimInfo, const fluidProps& fp, const char* label)
{
    strSimInfo << label << "\n";
    switch (fp.type) {
        case FLUID_POWERLAW:
            strSimInfo << "              Model: Power-Law\n";
            strSimInfo << "        Power index: " << fp.u.powerlaw.n_index << "\n";
            strSimInfo << " Consistency factor: " << fp.u.powerlaw.k_consistency << "\n";
            strSimInfo << "            Gamma 0: " << fp.u.powerlaw.gamma_0 << "\n";
            break;
        case FLUID_BINGHAM: {
            strSimInfo << "              Model: Bingham (Viscoplastic/Newtonian if s_y=0)\n";
            strSimInfo << "       Yield stress: " << fp.u.bingham.s_y << "\n";
            strSimInfo << "      Plastic omega: " << fp.u.bingham.omega_p << "\n";
            dfloat tau_local = 1.0_df / fp.u.bingham.omega_p;
            dfloat visc_local = (tau_local - 0.5_df) / 3.0_df;
            strSimInfo << "  Apparent viscosity: " << visc_local << "\n";
            if (fp.u.bingham.s_y == 0.0_df)
                strSimInfo << "      Note: behaves Newtonian (s_y=0).\n";
            break;
        }
        case FLUID_HERSCHEL_BULKLEY:
            strSimInfo << "              Model: Herschel-Bulkley\n";
            strSimInfo << "       Yield stress: " << fp.u.hb.s_y << "\n";
            strSimInfo << "        Power index: " << fp.u.hb.n_index << "\n";
            strSimInfo << " Consistency factor: " << fp.u.hb.k_consistency << "\n";
            strSimInfo << "            Gamma 0: " << fp.u.hb.gamma_0 << "\n";
            break;
        case FLUID_BI_VISCOSITY:
            strSimInfo << "              Model: Bi-viscosity\n";
            strSimInfo << "       Yield stress: " << fp.u.bi.s_y << "\n";
            strSimInfo << "     Viscosity ratio: " << fp.u.bi.visc_ratio << "\n";
            strSimInfo << "        Yield omega: " << fp.u.bi.omega_y << "\n";
            strSimInfo << "      Plastic omega: " << fp.u.bi.omega_p << "\n";
            strSimInfo << "    Critical gamma: " << fp.u.bi.gamma_c << "\n";
            break;
        case FLUID_KEE_TURCOTEE:
            strSimInfo << "              Model: Kee-Turcotte\n";
            strSimInfo << "       Yield stress: " << fp.u.kee.s_y << "\n";
            strSimInfo << "          Time param: " << fp.u.kee.t1 << "\n";
            strSimInfo << "   Zero-shear visc.: " << fp.u.kee.eta_0 << "\n";
            break;
        case FLUID_THIXO:
            strSimInfo << "              Model: Thixotropic\n";
            strSimInfo << "         Has lambda: " << (fp.hasLambda ? "Yes" : "No") << "\n";
            switch (fp.u.thixo.model) {
                case THIXO_MOORE1959:
                    strSimInfo << "      Thixo submodel: Moore (1959)\n";
                    strSimInfo << "         Build rate: " << fp.u.thixo.u.moore1959.k1 << "\n";
                    strSimInfo << "         Break rate: " << fp.u.thixo.u.moore1959.k2 << "\n";
                    strSimInfo << "   Initial lambda: " << fp.u.thixo.u.moore1959.lambda_0 << "\n";
                    strSimInfo << "   Zero-shear visc: " << fp.u.thixo.u.moore1959.eta_0 << "\n";
                    break;
                case THIXO_WORRALL1964:
                    strSimInfo << "      Thixo submodel: Worrall (1964)\n";
                    strSimInfo << "         Break rate: " << fp.u.thixo.u.worrall1964.k1 << "\n";
                    strSimInfo << "  Initial yield str: " << fp.u.thixo.u.worrall1964.s_y_0 << "\n";
                    strSimInfo << "   Zero-shear visc: " << fp.u.thixo.u.worrall1964.eta_0 << "\n";
                    break;
                case THIXO_HOUSKA1980:
                    strSimInfo << "      Thixo submodel: Houska (1980)\n";
                    strSimInfo << "         Build rate: " << fp.u.thixo.u.houska1980.k1 << "\n";
                    strSimInfo << "         Break rate: " << fp.u.thixo.u.houska1980.k2 << "\n";
                    strSimInfo << "      Power exponent: " << fp.u.thixo.u.houska1980.m_exponent << "\n";
                    strSimInfo << "   Initial yield str: " << fp.u.thixo.u.houska1980.s_y_0 << "\n";
                    strSimInfo << "      Eq. yield str: " << fp.u.thixo.u.houska1980.s_y_inf << "\n";
                    strSimInfo << "  Consistency factor: " << fp.u.thixo.u.houska1980.k_consistency << "\n";
                    strSimInfo << "         Power index: " << fp.u.thixo.u.houska1980.n_index << "\n";
                    break;
                case THIXO_TOORMAN1997:
                    strSimInfo << "      Thixo submodel: Toorman (1997)\n";
                    strSimInfo << "         Build rate: " << fp.u.thixo.u.toorman1997.k1 << "\n";
                    strSimInfo << "         Break rate: " << fp.u.thixo.u.toorman1997.k2 << "\n";
                    strSimInfo << "              a exp: " << fp.u.thixo.u.toorman1997.a_exponent << "\n";
                    strSimInfo << "              b exp: " << fp.u.thixo.u.toorman1997.b_exponent << "\n";
                    strSimInfo << "   Initial yield str: " << fp.u.thixo.u.toorman1997.s_y_0 << "\n";
                    strSimInfo << "   Zero-shear visc: " << fp.u.thixo.u.toorman1997.eta_0 << "\n";
                    break;
            }
            break;
        default:
            strSimInfo << "              Model: Unknown\n";
            break;
    }
    strSimInfo << "--------------------------------------------------------------------------------\n";
}

static void appendVeProps(std::ostringstream& strSimInfo, const veFluidProps& vp)
{
    switch (vp.type) {
        case VE_NEWTONIAN:
            strSimInfo << "         VE model: None (Newtonian solvent)\n";
            break;
        case VE_OLDROYD_B:
            strSimInfo << "         VE model: Oldroyd-B\n";
            strSimInfo << "  Polymer viscosity: " << vp.eta_p << "\n";
            strSimInfo << "    Relaxation time: " << vp.lambda << "\n";
            break;
        case VE_FENE_P:
            strSimInfo << "         VE model: FENE-P\n";
            strSimInfo << "  Polymer viscosity: " << vp.eta_p << "\n";
            strSimInfo << "    Relaxation time: " << vp.lambda << "\n";
            strSimInfo << "              L_sq: " << vp.u.fenep.L_sq << "\n";
            strSimInfo << "                 L: " << std::sqrt(vp.u.fenep.L_sq) << "\n";
            break;
        case VE_GIESEKUS:
            strSimInfo << "         VE model: Giesekus\n";
            strSimInfo << "  Polymer viscosity: " << vp.eta_p << "\n";
            strSimInfo << "    Relaxation time: " << vp.lambda << "\n";
            strSimInfo << "             Alpha: " << vp.u.giesekus.alpha << "\n";
            break;
        case VE_PTT_LINEAR:
            strSimInfo << "         VE model: PTT (linear)\n";
            strSimInfo << "  Polymer viscosity: " << vp.eta_p << "\n";
            strSimInfo << "    Relaxation time: " << vp.lambda << "\n";
            strSimInfo << "           Epsilon: " << vp.u.ptt.epsilon << "\n";
            break;
        case VE_PTT_EXPONENTIAL:
            strSimInfo << "         VE model: PTT (exponential)\n";
            strSimInfo << "  Polymer viscosity: " << vp.eta_p << "\n";
            strSimInfo << "    Relaxation time: " << vp.lambda << "\n";
            strSimInfo << "           Epsilon: " << vp.u.ptt.epsilon << "\n";
            break;
        default:
            strSimInfo << "         VE model: Unknown\n";
            break;
    }
}

std::string getSimInfoString(int step, dfloat MLUPS, const fluidPhaseProps& phasePropsA, const fluidPhaseProps& phasePropsB, bool hasSecond)
{
    std::ostringstream strSimInfo("");
    
    strSimInfo << std::scientific;
    strSimInfo << std::setprecision(6);
    
    strSimInfo << "---------------------------- SIMULATION INFORMATION ----------------------------\n";
    strSimInfo << "      Simulation ID: " << ID_SIM << "\n";
    #ifdef D3Q19
    strSimInfo << "       Velocity set: D3Q19\n";
    #endif // !D3Q19
    #ifdef D3Q27
    strSimInfo << "       Velocity set: D3Q27\n";
    #endif // !D3Q27
    #ifdef SINGLE_PRECISION
        strSimInfo << "          Precision: float\n";
    #else
        strSimInfo << "          Precision: double\n";
    #endif //SINGLE_PRECISION
    strSimInfo << "                 NX: " << NX << "\n";
    strSimInfo << "                 NY: " << NY << "\n";
    strSimInfo << "                 NZ: " << NZ << "\n";
    strSimInfo << "           NZ_TOTAL: " << NZ_TOTAL << "\n";
    strSimInfo << std::scientific << std::setprecision(6);
    strSimInfo << "                Tau: " << TAU << "\n";
    strSimInfo << "               Umax: " << U_MAX << "\n";
    strSimInfo << "                 FX: " << FX << "\n";
    strSimInfo << "                 FY: " << FY << "\n";
    strSimInfo << "                 FZ: " << FZ << "\n";
    strSimInfo << "         Save steps: " << MACR_SAVE << "\n";
    strSimInfo << "       Report steps: " << REPORT_SAVE << "\n";
    strSimInfo << "             Nsteps: " << step << "\n";
    strSimInfo << "              MLUPS: " << MLUPS << "\n";
        strSimInfo << std::scientific << std::setprecision(0);
    strSimInfo << "                 BX: " << BLOCK_NX << "\n";
    strSimInfo << "                 BY: " << BLOCK_NY << "\n";
    strSimInfo << "                 BZ: " << BLOCK_NZ << "\n";
    strSimInfo << "--------------------------------------------------------------------------------\n";

    strSimInfo << "\n------------------------------ BOUNDARY CONDITIONS -----------------------------\n";
    #ifdef BC_MOMENT_BASED
    strSimInfo << "            BC mode: Moment Based \n";
    #endif //BC_MOMENT_BASED
    strSimInfo << "            BC type: " << STR(BC_PROBLEM) << "\n";
    #ifdef BC_X_WALL
    strSimInfo << "          BC. X-Dir: Wall \n";
    #endif
    #ifdef BC_X_PERIODIC
    strSimInfo << "          BC. X-Dir: Periodic \n";
    #endif
    #ifdef BC_X_WALL
    strSimInfo << "          BC. Y-Dir: Wall \n";
    #endif
    #ifdef BC_Y_PERIODIC
    strSimInfo << "          BC. Y-Dir: Periodic \n";
    #endif
    #ifdef BC_Z_WALL
    strSimInfo << "          BC. Z-Dir: Wall \n";
    #endif
    #ifdef BC_Z_PERIODIC
    strSimInfo << "          BC. Z-Dir: Periodic \n";
    #endif
    strSimInfo << "--------------------------------------------------------------------------------\n";
    #ifdef OMEGA_FIELD
    strSimInfo << "\n------------------------------ NON NEWTONIAN FLUID -----------------------------\n";
    strSimInfo << std::scientific << std::setprecision(6);
    
    #ifdef NON_NEWTONIAN_FLUID
    appendFluidProps(strSimInfo, phasePropsA.nnf, "Phase A properties:");
    if (hasSecond) {
        appendFluidProps(strSimInfo, phasePropsB.nnf, "Phase B properties:");
    }
    #endif // NON_NEWTONIAN_FLUID
    #endif // OMEGA_FIELD
    #ifdef CONFORMATION_TENSOR
    strSimInfo << "\n------------------------------ VISCOELASTIC FLUID ------------------------------\n";
    strSimInfo << std::scientific << std::setprecision(6);
    strSimInfo << "Phase A:\n";
    appendVeProps(strSimInfo, phasePropsA.ve);
    if (hasSecond) {
        strSimInfo << "Phase B:\n";
        appendVeProps(strSimInfo, phasePropsB.ve);
    }
    strSimInfo << "\n  --- Conformation transport ---\n";
    strSimInfo << std::scientific << std::setprecision(4);
    strSimInfo << "  Diffusivity ratio: " << CONF_DIFFUSIVITY_RATIO << "\n";
    strSimInfo << "  Diffusivity Coef.: " << CONF_DIFFUSIVITY << "\n";
    strSimInfo << "Conformation Offset: " << CONF_ZERO << "\n";
    strSimInfo << "           CONF_TAU: " << CONF_TAU << "\n";
    strSimInfo << "         CONF_OMEGA: " << CONF_OMEGA << "\n";
    strSimInfo << "     CONF_DIFF_FLUC: " << CONF_DIFF_FLUC << "\n";
    strSimInfo << "CONF_DIFF_FLUC_COEF: " << CONF_DIFF_FLUC_COEF << "\n";
    strSimInfo << "--------------------------------------------------------------------------------\n";
    #endif // CONFORMATION_TENSOR
    #ifdef PARTICLE_MODEL
    strSimInfo << "\n---------------------------------- PARTICLES -----------------------------------\n";
    strSimInfo << std::scientific << std::setprecision(6);
    strSimInfo << "   Number of particles: " << NUM_PARTICLES << "\n";
    strSimInfo << "         Fluid density: " << FLUID_DENSITY << "\n";
    strSimInfo << "                    GX: " << GX << "\n";
    strSimInfo << "                    GY: " << GY << "\n";
    strSimInfo << "                    GZ: " << GZ << "\n";
    strSimInfo << "        Particles save: " << PARTICLES_SAVE << "\n";
    #ifdef IBM_METHOD
        strSimInfo << "\n------------------------------------- IBM --------------------------------------\n";
        strSimInfo << "  Particles nodes save: " << IBM_PARTICLES_NODES_SAVE << "\n";
        strSimInfo << "            Mesh scale: " << MESH_SCALE << "\n";
        strSimInfo << "          Mesh coulomb: " << MESH_COULOMB << "\n";
        strSimInfo << "         IBM thickness: " << IBM_THICKNESS << "\n";

        strSimInfo << "          Stencil size: ";
        #if defined STENCIL_2
        strSimInfo << "2" << "\n";
        #elif defined STENCIL_4
        strSimInfo << "4" << "\n";
        #else
        strSimInfo << "Invalid" << "\n";
        #endif
    #endif //IBM_METHOD
    #ifdef DEM_METHOD
        strSimInfo << "\n------------------------------------- DEM --------------------------------------\n";
        strSimInfo << " Part-Part Frict Coef.: " << PP_FRICTION_COEF << "\n";
        strSimInfo << " Part-Wall Frict Coef.: " << PW_FRICTION_COEF << "\n";
        strSimInfo << " Part-Part Rest. Coef.: " << PP_REST_COEF << "\n";
        strSimInfo << " Part-Wall Rest. Coef.: " << PW_REST_COEF << "\n";
        strSimInfo << " Particle Young's Mod.: " << PARTICLE_YOUNG_MODULUS << "\n";
        strSimInfo << " Particle Poisson Rat.: " << PARTICLE_POISSON_RATIO << "\n";
        strSimInfo << "     Wall Young's Mod.: " << WALL_YOUNG_MODULUS << "\n";
        strSimInfo << "     Wall Poisson Rat.: " << WALL_POISSON_RATIO << "\n";
        #endif //DEM_METHOD
    strSimInfo << "--------------------------------------------------------------------------------\n";
    #endif //PARTICLE_MODEL
    #ifdef LES_MODEL
    strSimInfo << "\n------------------------------------- LES --------------------------------------\n";
    strSimInfo << "\t Smagorisky Constant:" << CONST_SMAGORINSKY <<"\n";
    strSimInfo << "--------------------------------------------------------------------------------\n";
    #endif //LES
    #ifdef THERMAL_MODEL 
    strSimInfo << "\n------------------------------ THERMAL -----------------------------\n";
        strSimInfo << std::scientific << std::setprecision(2);
    strSimInfo << "     Prandtl Number: " << T_PR_NUMBER << "\n";
        strSimInfo << std::scientific << std::setprecision(4);
    strSimInfo << "    Rayleigh Number: " << T_RA_NUMBER << "\n";
    strSimInfo << "     Grashof Number: " << T_GR_NUMBER << "\n";
       strSimInfo << std::scientific << std::setprecision(3);
    strSimInfo << "            Delta T: " << T_DELTA_T << "\n";
    strSimInfo << "        Reference T: " << T_REFERENCE << "\n";
    strSimInfo << "             Cold T: " << T_COLD << "\n";
    strSimInfo << "              Hot T: " << T_HOT << "\n";
    strSimInfo << std::scientific << std::setprecision(6);
    strSimInfo << "       Thermal Diff: " << T_DIFFUSIVITY << "\n";
    strSimInfo << "   Grav_t_Exp.Coeff: " << T_gravity_t_beta << "\n";
       strSimInfo << std::scientific << std::setprecision(2);
    strSimInfo << "          Gravity_x: " << gravity_vector[0] << "\n";
    strSimInfo << "          Gravity_y: " << gravity_vector[1] << "\n";
    strSimInfo << "          Gravity_z: " << gravity_vector[2] << "\n";
       strSimInfo << std::scientific << std::setprecision(6);
    strSimInfo << "              G_TAU: " << G_TAU << "\n";
    strSimInfo << "            G_OMEGA: " << G_OMEGA << "\n";

    strSimInfo << "--------------------------------------------------------------------------------\n";
    #endif// THERMAL_MODEL
    #ifdef PHASE_MODEL 
    strSimInfo << "\n------------------------------ PHASE -----------------------------\n";
    strSimInfo << std::scientific << std::setprecision(4);
    strSimInfo << "          Delta Phi: " << PHI_DELTA_PHI << "\n";
    strSimInfo << "      Reference Phi: " << PHI_REFERENCE << "\n";
    strSimInfo << "            Phi One: " << PHI_ONE << "\n";
    strSimInfo << "            Phi Two: " << PHI_TWO << "\n";
    strSimInfo << "  Diffusivity ratio: " << (VISC/PHI_LBM_DIFF) << "\n";
    strSimInfo << "  Diffusivity Coef.: " << PHI_LBM_DIFF << "\n";
    strSimInfo << "         Phi Offset: " << PHI_ZERO << "\n";
    strSimInfo << "            PHI_TAU: " << PHI_TAU << "\n";
    strSimInfo << "          PHI_OMEGA: " << PHI_OMEGA << "\n";
    strSimInfo << "--------------------------------------------------------------------------------\n";
    #endif// PHASE_MODEL
    #if defined(FENE_P) || defined(OLDROYD_B)
    // Note: model details already covered by the CONFORMATION_TENSOR block above (runtime dispatch).
    // Legacy compile-time constants printed here for reference.
    strSimInfo << "\n------------------------------ VISCOELASTIC (compile-time) -------------------\n";
        strSimInfo << std::scientific << std::setprecision(4);
    strSimInfo << " Weissenberg Number: " << Weissenberg_number << "\n";
    strSimInfo << "    Sum Viscosities: " << SUM_VISC << "\n";
    strSimInfo << "    Viscosity Ratio: " << BETA << "\n";
    strSimInfo << "  Solvent Viscosity: " << VISC << "\n";
    strSimInfo << "--------------------------------------------------------------------------------\n";
    #endif// FENE_P
    return strSimInfo.str();
}

void saveSimInfo(int step, dfloat MLUPS, const fluidPhaseProps& phasePropsA, const fluidPhaseProps& phasePropsB, bool hasSecond)
{
    std::filesystem::path baseDir = folderSetup();

    // Use a fixed-length substring of ID_SIM for the info file name (e.g., first 16 chars)
    std::string idSimShort = std::string(ID_SIM).substr(0, 16);
    std::string baseName = idSimShort + std::string("_info.txt");
    std::filesystem::path strInf = baseDir / baseName;

    // On Windows, prepend the extended-path prefix to bypass the 260-char MAX_PATH limit.
    #if defined(_WIN32)
    std::string pathStr = "\\\\?\\" + strInf.string();
    std::replace(pathStr.begin(), pathStr.end(), '/', '\\');
    #else
    std::string pathStr = strInf.string();
    #endif

    FILE* outFile = fopen(pathStr.c_str(), "w");
    if(outFile != nullptr)
    {
        std::string strSimInfo = getSimInfoString(step, MLUPS, phasePropsA, phasePropsB, hasSecond);
        fprintf(outFile, "%s", strSimInfo.c_str());
        fclose(outFile);
    }
    else
    {
        printf("Error saving \"%s\" \nProbably wrong path!\n", pathStr.c_str());
    }
    
}
/**/


void saveTreatData(std::string fileName, std::string dataString, int step, bool headerExist)
{
    #if SAVEDATA
    std::filesystem::path baseDir = folderSetup();;

    std::string baseName = std::string(ID_SIM) + fileName;
    std::filesystem::path strInf = baseDir / (baseName + ".txt");

    // On Windows, prepend the extended-path prefix to bypass the 260-char MAX_PATH limit.
    #if defined(_WIN32)
    std::string pathStr = "\\\\?\\" + strInf.string();
    std::replace(pathStr.begin(), pathStr.end(), '/', '\\');
    #else
    std::string pathStr = strInf.string();
    #endif

    std::ifstream file(pathStr);
    std::ofstream outfile;

    if(step == REPORT_SAVE  && !headerExist){ //check if first time step to save data
        outfile.open(pathStr);
    }else{
        if (file.good()) {
            outfile.open(pathStr, std::ios::app);
        }else{ 
            outfile.open(pathStr);
        }
    }

    if (!outfile.is_open()) {
        std::cerr << "[saveTreatData] ERROR: failed to open output file: "
                  << strInf << " (path length: " << strInf.string().size() << ")" << std::endl;
        return;
    }

    outfile << dataString.c_str() << std::endl; 
    outfile.close(); 
    #endif //SAVEDATA
    #if CONSOLEPRINT
    printf("%s \n",dataString.c_str());
    #endif //CONSOLEPRINT
}

void saveTreatDataHeader(std::string fileName, std::string headerString)
{
    #if SAVEDATA
    std::filesystem::path baseDir = folderSetup();
    std::string baseName = std::string(ID_SIM) + fileName;
    std::filesystem::path strInf = baseDir / (baseName + ".txt");

    #if defined(_WIN32)
    std::string pathStr = "\\\\?\\" + strInf.string();
    std::replace(pathStr.begin(), pathStr.end(), '/', '\\');
    #else
    std::string pathStr = strInf.string();
    #endif

    std::ofstream outfile(pathStr); // overwrite file
    if (!outfile.is_open()) {
        std::cerr << "[saveTreatDataHeader] ERROR: failed to open output file: "
                  << strInf << " (path length: " << strInf.string().size() << ")" << std::endl;
        return;
    }
    outfile << headerString << std::endl;
    outfile.close();
    #endif //SAVEDATA
}
