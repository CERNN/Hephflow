#!/bin/bash
set -o pipefail

# CC=86

SHOW_WARNINGS=0
OUTPUT_PREFIX=""

usage() {
    echo "Usage: $0 [--show-warnings|--hide-warnings] <output_prefix>"
    echo "  --show-warnings  Show nvcc warnings (warnings are hidden by default)."
    echo "  --hide-warnings  Hide nvcc warnings explicitly."
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --show-warnings|-W)
            SHOW_WARNINGS=1
            ;;
        --hide-warnings)
            SHOW_WARNINGS=0
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        --)
            shift
            break
            ;;
        -*)
            echo "Error: Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
        *)
            if [ -n "$OUTPUT_PREFIX" ]; then
                echo "Error: Only one output prefix may be specified." >&2
                usage >&2
                exit 2
            fi
            OUTPUT_PREFIX=$1
            ;;
    esac
    shift
done

if [ "$#" -gt 0 ]; then
    if [ -n "$OUTPUT_PREFIX" ] || [ "$#" -gt 1 ]; then
        echo "Error: Only one output prefix may be specified." >&2
        usage >&2
        exit 2
    fi
    OUTPUT_PREFIX=$1
fi

if [ -z "$OUTPUT_PREFIX" ]; then
    echo "Error: Missing output prefix." >&2
    usage >&2
    exit 2
fi

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

mapfile -t CUDA_SOURCES < <(find . -name '*.cu' -print | sort)
if [ "${#CUDA_SOURCES[@]}" -eq 0 ]; then
    echo "Error: No CUDA source files found." >&2
    exit 1
fi

WARNING_FLAGS=()
if [ "$SHOW_WARNINGS" -eq 0 ]; then
    WARNING_FLAGS+=(--disable-warnings)
fi

BUILD_DIR=$(mktemp -d "${TMPDIR:-/tmp}/hephflow-build.XXXXXX") || {
    echo "Error: Could not create temporary build directory." >&2
    exit 1
}
trap 'rm -rf -- "$BUILD_DIR"' EXIT

OBJECT_EXTENSION=o
if [ "${OS:-}" = "Windows_NT" ]; then
    OBJECT_EXTENSION=obj
fi

: > compile_log.txt
OBJECTS=()
SOURCE_INDEX=0

for SOURCE in "${CUDA_SOURCES[@]}"; do
    SOURCE_INDEX=$((SOURCE_INDEX + 1))
    OBJECT="$BUILD_DIR/${SOURCE_INDEX}.${OBJECT_EXTENSION}"
    OBJECTS+=("$OBJECT")
    echo "${SOURCE#./}"

    nvcc --std=c++17 -gencode arch=compute_${CC},code=sm_${CC} -rdc=true -dc -O3 --restrict -DSM_${CC} \
        --maxrregcount="$MAX_REGS" \
        "${WARNING_FLAGS[@]}" \
        -diag-suppress 39 \
        -diag-suppress 179 \
        "$SOURCE" -o "$OBJECT" \
        2>&1 | sed '/^[[:space:]]*tmpxft_[^[:space:]]*[[:space:]]*$/d' | tee -a compile_log.txt
    BUILD_STATUS=${PIPESTATUS[0]}

    if [ "$BUILD_STATUS" -ne 0 ]; then
        echo "Compilation failed (exit code $BUILD_STATUS)." >&2
        exit "$BUILD_STATUS"
    fi
done

echo "Linking executable"
nvcc --std=c++17 -gencode arch=compute_${CC},code=sm_${CC} -rdc=true \
    "${WARNING_FLAGS[@]}" \
    "${OBJECTS[@]}" \
    -lcudadevrt -lcurand -o "./../bin/${OUTPUT_PREFIX}sim_${VELOCITY_SET}_sm${CC}" \
    2>&1 | sed \
        -e '/^[[:space:]]*tmpxft_[^[:space:]]*[[:space:]]*$/d' \
        -e '/^[[:space:]]*[0-9][0-9]*\.\(o\|obj\)[[:space:]]*$/d' \
        | tee -a compile_log.txt
BUILD_STATUS=${PIPESTATUS[0]}

if [ "$BUILD_STATUS" -ne 0 ]; then
    echo "Linking failed (exit code $BUILD_STATUS)." >&2
    exit "$BUILD_STATUS"
fi

echo "Built: ../bin/${OUTPUT_PREFIX}sim_${VELOCITY_SET}_sm${CC}"

rm -f ./../bin/*.exp ./../bin/*.lib

#--ptxas-options=-v
# 39,179 suppress division by false in the mods

        # -diag-suppress 550 \
        # -diag-suppress 549 \
        # -diag-suppress 177 \
        # -lineinfo \ #usefull for nsight compute debug
