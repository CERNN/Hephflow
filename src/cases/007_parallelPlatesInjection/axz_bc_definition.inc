switch (nodeType) {
    case BULK:
        break;
    case NORTH:
        AxzVar     = (5 * fMom[idxMom(threadIdx.x, threadIdx.y - 1, threadIdx.z, A_XZ_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]
                      - 4 * fMom[idxMom(threadIdx.x, threadIdx.y - 2, threadIdx.z, A_XZ_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]
                      + fMom[idxMom(threadIdx.x, threadIdx.y - 3, threadIdx.z, A_XZ_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]) / 2.0;

        Axz_qx_t30 = (5 * fMom[idxMom(threadIdx.x, threadIdx.y - 1, threadIdx.z, A_XZ_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]
                      - 4 * fMom[idxMom(threadIdx.x, threadIdx.y - 2, threadIdx.z, A_XZ_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]
                      + fMom[idxMom(threadIdx.x, threadIdx.y - 3, threadIdx.z, A_XZ_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]) / 2.0;

        Axz_qy_t30 = (5 * fMom[idxMom(threadIdx.x, threadIdx.y - 1, threadIdx.z, A_XZ_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]
                      - 4 * fMom[idxMom(threadIdx.x, threadIdx.y - 2, threadIdx.z, A_XZ_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]
                      + fMom[idxMom(threadIdx.x, threadIdx.y - 3, threadIdx.z, A_XZ_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]) / 2.0;

        Axz_qz_t30 = (5 * fMom[idxMom(threadIdx.x, threadIdx.y - 1, threadIdx.z, A_XZ_CZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]
                      - 4 * fMom[idxMom(threadIdx.x, threadIdx.y - 2, threadIdx.z, A_XZ_CZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]
                      + fMom[idxMom(threadIdx.x, threadIdx.y - 3, threadIdx.z, A_XZ_CZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]) / 2.0;
        break;

    case SOUTH:
        AxzVar     = (5 * fMom[idxMom(threadIdx.x, threadIdx.y + 1, threadIdx.z, A_XZ_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]
                      - 4 * fMom[idxMom(threadIdx.x, threadIdx.y + 2, threadIdx.z, A_XZ_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]
                      + fMom[idxMom(threadIdx.x, threadIdx.y + 3, threadIdx.z, A_XZ_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]) / 2.0;

        Axz_qx_t30 = (5 * fMom[idxMom(threadIdx.x, threadIdx.y + 1, threadIdx.z, A_XZ_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]
                      - 4 * fMom[idxMom(threadIdx.x, threadIdx.y + 2, threadIdx.z, A_XZ_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]
                      + fMom[idxMom(threadIdx.x, threadIdx.y + 3, threadIdx.z, A_XZ_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]) / 2.0;

        Axz_qy_t30 = (5 * fMom[idxMom(threadIdx.x, threadIdx.y + 1, threadIdx.z, A_XZ_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]
                      - 4 * fMom[idxMom(threadIdx.x, threadIdx.y + 2, threadIdx.z, A_XZ_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]
                      + fMom[idxMom(threadIdx.x, threadIdx.y + 3, threadIdx.z, A_XZ_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]) / 2.0;

        Axz_qz_t30 = (5 * fMom[idxMom(threadIdx.x, threadIdx.y + 1, threadIdx.z, A_XZ_CZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]
                      - 4 * fMom[idxMom(threadIdx.x, threadIdx.y + 2, threadIdx.z, A_XZ_CZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]
                      + fMom[idxMom(threadIdx.x, threadIdx.y + 3, threadIdx.z, A_XZ_CZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]) / 2.0;
        break;
    case WEST + BC_VELOCITY_WALL:
        AxzVar = 0.0;
        Axz_qx_t30 = 3.0*AxzVar*(ux_t30 + 0.0);
        Axz_qy_t30 = 3.0*AxzVar*(uy_t30 + 0.0);
        Axz_qz_t30 = 3.0*AxzVar*(uz_t30 + 0.0);
    case EAST + BC_OUTFLOW: //copy values from last row - 1
        AxzVar = fMom[idxMom(threadIdx.x - 1, threadIdx.y, threadIdx.z, A_XZ_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
        Axz_qx_t30 = fMom[idxMom(threadIdx.x - 1, threadIdx.y, threadIdx.z, A_XZ_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
        Axz_qy_t30 = fMom[idxMom(threadIdx.x - 1, threadIdx.y, threadIdx.z, A_XZ_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
        Axz_qz_t30 = fMom[idxMom(threadIdx.x - 1, threadIdx.y, threadIdx.z, A_XZ_CZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
        break;
    default:
        break;
}
