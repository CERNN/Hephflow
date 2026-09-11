 const int planeIdx = x + y * NX;  

    if (tz == 0)
    { //b
            if  (bz == 0) {
                if constexpr (MACRO_HALOS_REQUIRED) {
                    macroInterfaceGPUrho.Z_0[planeIdx] = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_RHO_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
                    macroInterfaceGPUux.Z_0[planeIdx] = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_UX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
                    macroInterfaceGPUuy.Z_0[planeIdx] = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_UY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
                    macroInterfaceGPUuz.Z_0[planeIdx] = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_UZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
                }

                #ifdef SECOND_DIST
                macroInterfaceGPUg.Z_0[planeIdx] = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M2_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
                #endif //SECOND_DIST

                #ifdef LAMBDA_DIST
                macroInterfaceGPUlambda.Z_0[planeIdx] = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M4_LAMBDA_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
                #endif //LAMBDA_DIST

                #ifdef A_XX_DIST
                macroInterfaceGPUAxx.Z_0[planeIdx] = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_XX_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
                #endif //A_XX_DIST
                
                #ifdef A_XY_DIST
                macroInterfaceGPUAxy.Z_0[planeIdx] = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_XY_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
                #endif //A_XY_DIST
        
                #ifdef A_XZ_DIST
                macroInterfaceGPUAxz.Z_0[planeIdx] = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_XZ_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
                #endif //A_XZ_DIST
        
                #ifdef A_YY_DIST
                macroInterfaceGPUAyy.Z_0[planeIdx] = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_YY_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
                #endif //A_YY_DIST
       
                #ifdef A_YZ_DIST
                macroInterfaceGPUAyz.Z_0[planeIdx] = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_YZ_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
                #endif //A_YZ_DIST
        
                #ifdef A_ZZ_DIST
                macroInterfaceGPUAzz.Z_0[planeIdx] = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_ZZ_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
                #endif //A_ZZ_DIST

                #ifdef PHI_DIST
                macroInterfaceGPUphi.Z_0[planeIdx] = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M3_PHI_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
                macroInterfaceGPUnx.Z_0[planeIdx] = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M3_NX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
                macroInterfaceGPUny.Z_0[planeIdx] = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M3_NY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
                macroInterfaceGPUnz.Z_0[planeIdx] = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M3_NZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
                macroInterfaceGPUmu.Z_0[planeIdx] = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M3_MU_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
                #endif

            }
    }

    if (tz == (BLOCK_NZ - 1))
    { //b
            if  (bz == NUM_BLOCK_Z_LOCAL - 1) {
                if constexpr (MACRO_HALOS_REQUIRED) {
                    macroInterfaceGPUrho.Z_1[planeIdx] = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_RHO_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
                    macroInterfaceGPUux.Z_1[planeIdx] = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_UX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
                    macroInterfaceGPUuy.Z_1[planeIdx] = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_UY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
                    macroInterfaceGPUuz.Z_1[planeIdx] = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_UZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
                }

                #ifdef SECOND_DIST
                macroInterfaceGPUg.Z_1[planeIdx] = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M2_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
                #endif //SECOND_DIST

                #ifdef LAMBDA_DIST
                macroInterfaceGPUlambda.Z_1[planeIdx] = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M4_LAMBDA_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
                #endif //LAMBDA_DIST

                #ifdef A_XX_DIST
                macroInterfaceGPUAxx.Z_1[planeIdx] = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_XX_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
                #endif //A_XX_DIST
                
                #ifdef A_XY_DIST
                macroInterfaceGPUAxy.Z_1[planeIdx] = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_XY_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
                #endif //A_XY_DIST
        
                #ifdef A_XZ_DIST
                macroInterfaceGPUAxz.Z_1[planeIdx] = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_XZ_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
                #endif //A_XZ_DIST
        
                #ifdef A_YY_DIST
                macroInterfaceGPUAyy.Z_1[planeIdx] = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_YY_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
                #endif //A_YY_DIST
       
                #ifdef A_YZ_DIST
                macroInterfaceGPUAyz.Z_1[planeIdx] = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_YZ_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
                #endif //A_YZ_DIST
        
                #ifdef A_ZZ_DIST
                macroInterfaceGPUAzz.Z_1[planeIdx] = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_ZZ_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
                #endif //A_ZZ_DIST

                #ifdef PHI_DIST
                macroInterfaceGPUphi.Z_1[planeIdx] = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M3_PHI_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
                macroInterfaceGPUnx.Z_1[planeIdx] = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M3_NX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
                macroInterfaceGPUny.Z_1[planeIdx] = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M3_NY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
                macroInterfaceGPUnz.Z_1[planeIdx] = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M3_NZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
                macroInterfaceGPUmu.Z_1[planeIdx] = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M3_MU_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
                #endif
            }
    }
