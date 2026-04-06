# CC=86

# Prevent runtime cudaErrorInsufficientDriver by validating CUDA compatibility
# before compiling: toolkit version (nvcc) must be <= driver supported version.
get_nvcc_version() {
    nvcc --version | sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -n 1
}

get_driver_cuda_version() {
    nvidia-smi | sed -n 's/.*CUDA Version: \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -n 1
}

version_gt() {
    [ "$(printf '%s\n' "$1" "$2" | sort -V | tail -n 1)" != "$2" ]
}

NVCC_VERSION=$(get_nvcc_version)
DRIVER_CUDA_VERSION=$(get_driver_cuda_version)

if [ -z "$NVCC_VERSION" ] || [ -z "$DRIVER_CUDA_VERSION" ]; then
    echo "Warning: Could not determine CUDA versions (nvcc='$NVCC_VERSION', driver='$DRIVER_CUDA_VERSION')."
    echo "Skipping compatibility check."
elif version_gt "$NVCC_VERSION" "$DRIVER_CUDA_VERSION"; then
    echo "Warning: CUDA toolkit/driver mismatch detected."
    echo "  nvcc toolkit version:           $NVCC_VERSION"
    echo "  Driver supported CUDA version:  $DRIVER_CUDA_VERSION"
    echo ""
    echo "This build may compile but fail at runtime with cudaErrorInsufficientDriver (error 35)."
    echo "Fix options:"
    echo "  1) Install/use CUDA toolkit <= $DRIVER_CUDA_VERSION"
    echo "  2) Update NVIDIA driver to support CUDA $NVCC_VERSION"
    read -r -p "Continue compilation anyway? [y/N] " CONTINUE_ANYWAY
    case "$CONTINUE_ANYWAY" in
        [yY]|[yY][eE][sS])
            echo "Proceeding with compilation despite mismatch."
            ;;
        *)
            echo "Compilation canceled."
            exit 1
            ;;
    esac
fi

# Set CC if it's not already defined
if [ -z "$CC" ]; then
    CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1 | tr -d '.')
    if [ -z "$CC" ]; then
        echo "Error: Unable to determine compute capability."
        exit 1
    fi
fi

# Read block dimensions from memory_layout.h and calculate max registers
MEMORY_LAYOUT="./include/memory_layout.h"
BLOCK_NX=$(grep -E "^#define BLOCK_NX" "$MEMORY_LAYOUT" | awk '{print $3}')
BLOCK_NY=$(grep -E "^#define BLOCK_NY" "$MEMORY_LAYOUT" | awk '{print $3}')
BLOCK_NZ=$(grep -E "^#define BLOCK_NZ" "$MEMORY_LAYOUT" | awk '{print $3}')
BLOCK_SIZE=$((BLOCK_NX * BLOCK_NY * BLOCK_NZ))

# SM_86 has 65536 registers per SM. Calculate max registers per thread.
MAX_REGS=$((65536 / BLOCK_SIZE))

# Cap at 255 (CUDA max per thread)
if [ "$MAX_REGS" -gt 255 ]; then
    MAX_REGS=255
fi

# Warn if registers are very low (will cause register spilling)
if [ "$MAX_REGS" -lt 48 ]; then
    echo "Warning: Block size $BLOCK_SIZE limits registers to $MAX_REGS (may cause spilling)"
fi

echo "Block size: ${BLOCK_NX}x${BLOCK_NY}x${BLOCK_NZ} = $BLOCK_SIZE threads"
echo "Max registers per thread: $MAX_REGS"

# Auto-detect velocity set (D3Q19/D3Q27) from the active case's model.inc
BC_PROBLEM=$(grep -E "^#define BC_PROBLEM" "./var.h" | awk '{print $3}')
if [ -z "$BC_PROBLEM" ]; then
    echo "Error: Could not read BC_PROBLEM from var.h"
    exit 1
fi
MODEL_INC="./cases/${BC_PROBLEM}/model.inc"
if [ ! -f "$MODEL_INC" ]; then
    echo "Error: model.inc not found at $MODEL_INC"
    exit 1
fi
VELOCITY_SET=$(grep -E "^#define (D3Q19|D3Q27)" "$MODEL_INC" | awk '{print $2}' | head -n 1)
if [ -z "$VELOCITY_SET" ]; then
    echo "Error: D3Q19 or D3Q27 not defined in $MODEL_INC"
    exit 1
fi
echo "Velocity set: $VELOCITY_SET (from cases/${BC_PROBLEM}/model.inc)"

nvcc --std=c++17 -gencode arch=compute_${CC},code=sm_${CC} -rdc=true -O3 --restrict -DSM_${CC}  \
    --maxrregcount=$MAX_REGS \
    $(find . -name '*.cu') \
    -diag-suppress 39 \
    -diag-suppress 179 \
    -lcudadevrt -lcurand -o ./../bin/$1sim_${VELOCITY_SET}_sm${CC} 2>&1 | tee compile_log.txt

rm -f ./../bin/*.exp ./../bin/*.lib

#--ptxas-options=-v
# 39,179 suppress division by false in the mods

        # -diag-suppress 550 \
        # -diag-suppress 549 \
        # -diag-suppress 177 \
        # -lineinfo \ #usefull for nsight compute debug