#ifndef __DOMAIN_DECOMPOSITION_H
#define __DOMAIN_DECOMPOSITION_H

#include <cstddef>
#include <stdexcept>
#include <vector>

/** @brief Returns the number of @p divisor-sized chunks needed for @p value. */
constexpr size_t ceilDivSize(size_t value, size_t divisor) {
    return (value + divisor - 1) / divisor;
}

/** @brief Returns the physical Z-plane count assigned to one GPU. */
constexpr size_t balancedLocalNZ(size_t nz, size_t gpuCount, size_t gpu) {
    return nz / gpuCount + (gpu < nz % gpuCount ? 1 : 0);
}

/** @brief Returns the global starting Z plane assigned to one GPU. */
constexpr size_t balancedZStart(size_t nz, size_t gpuCount, size_t gpu) {
    return gpu * (nz / gpuCount) + (gpu < nz % gpuCount ? gpu : nz % gpuCount);
}

struct GpuDomainLayout {
    unsigned int gpuIndex{};
    size_t zStart{};
    size_t localNZ{};
    size_t numBlocksX{};
    size_t numBlocksY{};
    size_t numBlocksZ{};

    size_t paddedNodeCount{};
    size_t physicalNodeCount{};
    size_t momentBytes{};
    size_t momentStorageBytes{};
    size_t scalarBytes{};
    size_t ghostFaceXYCount{};
    size_t ghostFaceXZCount{};
    size_t ghostFaceYZCount{};
    size_t exchangePlaneCount{};
    size_t hydroFaceXYBytes{};
    size_t hydroFaceXZBytes{};
    size_t hydroFaceYZBytes{};
    size_t hydroExchangePlaneBytes{};
    size_t scalarFaceXYBytes{};
    size_t scalarFaceXZBytes{};
    size_t scalarFaceYZBytes{};
    size_t scalarExchangePlaneBytes{};
    dim3 fullGrid{};
    dim3 boundaryGrid{};
    dim3 interiorGrid{};

    /** @brief Constructs an empty layout for deferred initialization. */
    GpuDomainLayout() = default;

    /**
     * @brief Builds geometry and allocation sizes for one GPU Z slab.
     * @param index GPU index.
     * @param start Global first Z plane owned by the GPU.
     * @param depth Number of physical Z planes owned by the GPU.
     */
    GpuDomainLayout(unsigned int index, size_t start, size_t depth)
        : gpuIndex(index), zStart(start), localNZ(depth),
          numBlocksX(ceilDivSize(NX, BLOCK_NX)),
          numBlocksY(ceilDivSize(NY, BLOCK_NY)),
          numBlocksZ(ceilDivSize(depth, BLOCK_NZ)) {
            
        paddedNodeCount = numBlocksX * numBlocksY * numBlocksZ * BLOCK_LBM_SIZE;
        
        physicalNodeCount = static_cast<size_t>(NX) * NY * localNZ;

        momentBytes = sizeof(dfloat) * paddedNodeCount * NUMBER_MOMENTS;

        momentStorageBytes = sizeof(dfloat) * NUMBER_MOMENTS * paddedNodeCount;

        scalarBytes = sizeof(dfloat) * paddedNodeCount;

        ghostFaceXYCount = BLOCK_NX * BLOCK_NY * numBlocksX * numBlocksY * numBlocksZ;
        ghostFaceXZCount = BLOCK_NX * BLOCK_NZ * numBlocksX * numBlocksY * numBlocksZ;
        ghostFaceYZCount = BLOCK_NY * BLOCK_NZ * numBlocksX * numBlocksY * numBlocksZ;

        exchangePlaneCount = BLOCK_NX * BLOCK_NY * numBlocksX * numBlocksY;

        hydroFaceXYBytes = sizeof(dfloat) * QF * ghostFaceXYCount;
        hydroFaceXZBytes = sizeof(dfloat) * QF * ghostFaceXZCount;
        hydroFaceYZBytes = sizeof(dfloat) * QF * ghostFaceYZCount;

        hydroExchangePlaneBytes = sizeof(dfloat) * QF * exchangePlaneCount;

        #ifdef CONVECTION_DIFFUSION_TRANSPORT
        scalarFaceXYBytes = sizeof(dfloat) * GF * ghostFaceXYCount;
        scalarFaceXZBytes = sizeof(dfloat) * GF * ghostFaceXZCount;
        scalarFaceYZBytes = sizeof(dfloat) * GF * ghostFaceYZCount;

        scalarExchangePlaneBytes = sizeof(dfloat) * GF * exchangePlaneCount;
        #endif // CONVECTION_DIFFUSION_TRANSPORT
        fullGrid = dim3(static_cast<unsigned int>(numBlocksX), static_cast<unsigned int>(numBlocksY), static_cast<unsigned int>(numBlocksZ));
        boundaryGrid = dim3(static_cast<unsigned int>(numBlocksX), static_cast<unsigned int>(numBlocksY), static_cast<unsigned int>(numBlocksZ > 1 ? 2 : 1));
        interiorGrid = dim3(static_cast<unsigned int>(numBlocksX), static_cast<unsigned int>(numBlocksY), static_cast<unsigned int>(numBlocksZ > 2 ? numBlocksZ - 2 : 0));
    }
};

class DomainDecomposition {
public:
    /**
     * @brief Partitions the global Z domain over @p gpuCount GPUs.
     * @throws std::invalid_argument if the domain or GPU count is invalid.
     */
    explicit DomainDecomposition(size_t gpuCount) {
        if (NX <= 0 || NY <= 0 || NZ <= 0) throw std::invalid_argument("Domain dimensions must be positive");
        if (gpuCount == 0 || gpuCount > static_cast<size_t>(NZ)) throw std::invalid_argument("GPU count must be between one and NZ");

        layouts_.reserve(gpuCount);
        for (size_t g = 0; g < gpuCount; ++g) {
            layouts_.emplace_back(static_cast<unsigned int>(g), balancedZStart(NZ, gpuCount, g), balancedLocalNZ(NZ, gpuCount, g));
        }
    }

    /** @brief Returns the layout assigned to @p gpu. */
    const GpuDomainLayout& operator[](size_t gpu) const { return layouts_.at(gpu); }
    /** @brief Returns the number of decomposed GPU slabs. */
    size_t size() const { return layouts_.size(); }
    /** @brief Returns all GPU slab layouts in index order. */
    const std::vector<GpuDomainLayout>& layouts() const { return layouts_; }

private:
    std::vector<GpuDomainLayout> layouts_;
};

#endif // __DOMAIN_DECOMPOSITION_H
