# CC=86

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

if [[ "$1" = "D3Q19" || "$1" = "D3Q27" ]]; then
    nvcc --std=c++17 -gencode arch=compute_${CC},code=sm_${CC} -rdc=true -O3 --restrict -DSM_${CC}  \
        --maxrregcount=$MAX_REGS \
        $(find . -name '*.cu') \
        -diag-suppress 39 \
        -diag-suppress 179 \
        -lcudadevrt -lcurand -o ./../bin/$2sim_$1_sm${CC} 2>&1 | tee compile_log.txt
else
    echo "Input error, example of usage is:"
    echo "sh compile.sh D3Q19 011"
    echo "sh compile.sh D3Q27 202"
fi

rm -f ./../bin/*.exp ./../bin/*.lib

#--ptxas-options=-v
# 39,179 suppress division by false in the mods

        # -diag-suppress 550 \
        # -diag-suppress 549 \
        # -diag-suppress 177 \
        # -lineinfo \ #usefull for nsight compute debug