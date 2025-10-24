#ifndef __DEVICEFIELD_STRUCTS_H
#define __DEVICEFIELD_STRUCTS_H

#include "var.h"
#include "main.cuh"

typedef struct deviceField{
    dfloat* d_fMom;
    unsigned int* dNodeType;

    #if NODE_TYPE_SAVE
    unsigned int* nodeTypeSave; 
    #endif //NODE_TYPE_SAVE

    void allocateHostMemoryDeviceField(ghostInterfaceData ghostInterface) {
        allocateDeviceMemory(
            &d_fMom, &dNodeType, &ghostInterface
            BC_FORCES_PARAMS_PTR(d_)
        );
    }

    void freeDeviceField() {
        cudaFree(d_fMom);
        cudaFree(dNodeType);

        #if NODE_TYPE_SAVE
        cudaFree(nodeTypeSave);
        #endif //NODE_TYPE_SAVE
    }

} DeviceField;
#endif //__DEVICEFIELD_STRUCTS_H