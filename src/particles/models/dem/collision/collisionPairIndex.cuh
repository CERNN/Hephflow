#ifndef __COLLISION_PAIR_INDEX_CUH
#define __COLLISION_PAIR_INDEX_CUH

#include <cmath>
#include <cstdint>

struct CollisionPairIndex {
    unsigned int column;
    unsigned int row;
};

// Convert a linear strict-lower-triangle index to the unique pair
// (column,row), where column < row. Double precision provides a close initial
// estimate; exact 64-bit triangular-number checks remove boundary ambiguity.
__host__ __device__ __forceinline__
CollisionPairIndex collisionPairFromLinearIndex(std::uint64_t index)
{
    unsigned int row = static_cast<unsigned int>(
        (1.0 + sqrt(1.0 + 8.0 * static_cast<double>(index))) * 0.5);
    std::uint64_t rowStart =
        (static_cast<std::uint64_t>(row) * (row - 1ULL)) / 2ULL;

    if (rowStart > index) {
        --row;
        rowStart = (static_cast<std::uint64_t>(row) * (row - 1ULL)) / 2ULL;
    } else {
        const std::uint64_t nextRowStart =
            (static_cast<std::uint64_t>(row) * (row + 1ULL)) / 2ULL;
        if (nextRowStart <= index) {
            ++row;
            rowStart = nextRowStart;
        }
    }

    return {
        static_cast<unsigned int>(index - rowStart),
        row
    };
}

#endif
