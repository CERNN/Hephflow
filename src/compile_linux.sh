# To compile in linux:
# chmod +x compile_linux.sh
# dos2unix compile_linux.sh
#   
# ../bin/000sim_D3Q19_sm80
# edit where necessary

CC=80


nvcc -gencode arch=compute_${CC},code=sm_${CC} -rdc=true -O3 --restrict  -DSM_${CC} \
    -DTARGET_LINUX \
    *.cu \
    -diag-suppress=39 \
    -diag-suppress=179 \
    -lcudadevrt -lcurand -o ./../bin/"$1"sim_D3Q19_sm${CC}



#--ptxas-options=-v
# 39,179 suppress division by false in the mods

        # -diag-suppress 550 \
        # -diag-suppress 549 \
        # -diag-suppress 177 \
 #               -diag-suppress 39 \
#        -diag-suppress 179 \
        # -lineinfo \ #usefull for nsight compute debug