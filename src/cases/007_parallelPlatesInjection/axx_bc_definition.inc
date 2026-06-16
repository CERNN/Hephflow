switch (nodeType) {
    case BULK:
        break;
    case NORTH:
        AxxVar     = (5 * fMom[idxMom(threadIdx.x, threadIdx.y - 1, threadIdx.z, A_XX_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]
                    - 4 * fMom[idxMom(threadIdx.x, threadIdx.y - 2, threadIdx.z, A_XX_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]
                        + fMom[idxMom(threadIdx.x, threadIdx.y - 3, threadIdx.z, A_XX_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]) *0.5;
        
        Axx_qx_t30 = (5 * fMom[idxMom(threadIdx.x, threadIdx.y - 1, threadIdx.z, A_XX_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]
                    - 4 * fMom[idxMom(threadIdx.x, threadIdx.y - 2, threadIdx.z, A_XX_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]
                        + fMom[idxMom(threadIdx.x, threadIdx.y - 3, threadIdx.z, A_XX_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]) *0.5;

        Axx_qy_t30 = (5 * fMom[idxMom(threadIdx.x, threadIdx.y - 1, threadIdx.z, A_XX_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]
                    - 4 * fMom[idxMom(threadIdx.x, threadIdx.y - 2, threadIdx.z, A_XX_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]
                        + fMom[idxMom(threadIdx.x, threadIdx.y - 3, threadIdx.z, A_XX_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]) *0.5;

        Axx_qz_t30 = (5 * fMom[idxMom(threadIdx.x, threadIdx.y - 1, threadIdx.z, A_XX_CZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]
                    - 4 * fMom[idxMom(threadIdx.x, threadIdx.y - 2, threadIdx.z, A_XX_CZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]
                        + fMom[idxMom(threadIdx.x, threadIdx.y - 3, threadIdx.z, A_XX_CZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]) *0.5;
        break;

    case SOUTH:
        AxxVar     = (5 * fMom[idxMom(threadIdx.x, threadIdx.y + 1, threadIdx.z, A_XX_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]
                    - 4 * fMom[idxMom(threadIdx.x, threadIdx.y + 2, threadIdx.z, A_XX_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]
                        + fMom[idxMom(threadIdx.x, threadIdx.y + 3, threadIdx.z, A_XX_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]) *0.5;

        Axx_qx_t30 = (5 * fMom[idxMom(threadIdx.x, threadIdx.y + 1, threadIdx.z, A_XX_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]
                    - 4 * fMom[idxMom(threadIdx.x, threadIdx.y + 2, threadIdx.z, A_XX_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]
                        + fMom[idxMom(threadIdx.x, threadIdx.y + 3, threadIdx.z, A_XX_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]) *0.5;

        Axx_qy_t30 = (5 * fMom[idxMom(threadIdx.x, threadIdx.y + 1, threadIdx.z, A_XX_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]
                    - 4 * fMom[idxMom(threadIdx.x, threadIdx.y + 2, threadIdx.z, A_XX_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]
                        + fMom[idxMom(threadIdx.x, threadIdx.y + 3, threadIdx.z, A_XX_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]) *0.5;

        Axx_qz_t30 = (5 * fMom[idxMom(threadIdx.x, threadIdx.y + 1, threadIdx.z, A_XX_CZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]
                    - 4 * fMom[idxMom(threadIdx.x, threadIdx.y + 2, threadIdx.z, A_XX_CZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]
                        + fMom[idxMom(threadIdx.x, threadIdx.y + 3, threadIdx.z, A_XX_CZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]) *0.5;
        break;
    case WEST + BC_VELOCITY_WALL:
        AxxVar = 1.0;
        Axx_qx_t30 = 3.0*AxxVar*(ux_t30 + 0.0);
        Axx_qy_t30 = 3.0*AxxVar*(uy_t30 + 0.0);
        Axx_qz_t30 = 3.0*AxxVar*(uz_t30 + 0.0);
    case EAST + BC_OUTFLOW: //copy values from last row - 1
        AxxVar = fMom[idxMom(threadIdx.x - 1, threadIdx.y, threadIdx.z, A_XX_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
        Axx_qx_t30 = fMom[idxMom(threadIdx.x - 1, threadIdx.y, threadIdx.z, A_XX_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
        Axx_qy_t30 = fMom[idxMom(threadIdx.x - 1, threadIdx.y, threadIdx.z, A_XX_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
        Axx_qz_t30 = fMom[idxMom(threadIdx.x - 1, threadIdx.y, threadIdx.z, A_XX_CZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
        break;
    default:
        break;
}
