switch (nodeType) {
    case BULK:
        break;
    case NORTH:
        AzzVar     = (5 * fMom[idxMom(threadIdx.x, threadIdx.y - 1, threadIdx.z, A_ZZ_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]
                      - 4 * fMom[idxMom(threadIdx.x, threadIdx.y - 2, threadIdx.z, A_ZZ_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]
                      + fMom[idxMom(threadIdx.x, threadIdx.y - 3, threadIdx.z, A_ZZ_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]) / 2.0;

        Azz_qx_t30 = (5 * fMom[idxMom(threadIdx.x, threadIdx.y - 1, threadIdx.z, A_ZZ_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]
                      - 4 * fMom[idxMom(threadIdx.x, threadIdx.y - 2, threadIdx.z, A_ZZ_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]
                      + fMom[idxMom(threadIdx.x, threadIdx.y - 3, threadIdx.z, A_ZZ_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]) / 2.0;

        Azz_qy_t30 = (5 * fMom[idxMom(threadIdx.x, threadIdx.y - 1, threadIdx.z, A_ZZ_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]
                      - 4 * fMom[idxMom(threadIdx.x, threadIdx.y - 2, threadIdx.z, A_ZZ_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]
                      + fMom[idxMom(threadIdx.x, threadIdx.y - 3, threadIdx.z, A_ZZ_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]) / 2.0;

        Azz_qz_t30 = (5 * fMom[idxMom(threadIdx.x, threadIdx.y - 1, threadIdx.z, A_ZZ_CZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]
                      - 4 * fMom[idxMom(threadIdx.x, threadIdx.y - 2, threadIdx.z, A_ZZ_CZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]
                      + fMom[idxMom(threadIdx.x, threadIdx.y - 3, threadIdx.z, A_ZZ_CZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]) / 2.0;
        break;

    case SOUTH:
        AzzVar     = (5 * fMom[idxMom(threadIdx.x, threadIdx.y + 1, threadIdx.z, A_ZZ_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]
                      - 4 * fMom[idxMom(threadIdx.x, threadIdx.y + 2, threadIdx.z, A_ZZ_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]
                      + fMom[idxMom(threadIdx.x, threadIdx.y + 3, threadIdx.z, A_ZZ_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]) / 2.0;

        Azz_qx_t30 = (5 * fMom[idxMom(threadIdx.x, threadIdx.y + 1, threadIdx.z, A_ZZ_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]
                      - 4 * fMom[idxMom(threadIdx.x, threadIdx.y + 2, threadIdx.z, A_ZZ_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]
                      + fMom[idxMom(threadIdx.x, threadIdx.y + 3, threadIdx.z, A_ZZ_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]) / 2.0;

        Azz_qy_t30 = (5 * fMom[idxMom(threadIdx.x, threadIdx.y + 1, threadIdx.z, A_ZZ_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]
                      - 4 * fMom[idxMom(threadIdx.x, threadIdx.y + 2, threadIdx.z, A_ZZ_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]
                      + fMom[idxMom(threadIdx.x, threadIdx.y + 3, threadIdx.z, A_ZZ_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]) / 2.0;

        Azz_qz_t30 = (5 * fMom[idxMom(threadIdx.x, threadIdx.y + 1, threadIdx.z, A_ZZ_CZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]
                      - 4 * fMom[idxMom(threadIdx.x, threadIdx.y + 2, threadIdx.z, A_ZZ_CZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]
                      + fMom[idxMom(threadIdx.x, threadIdx.y + 3, threadIdx.z, A_ZZ_CZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]) / 2.0;
        break;
    case WEST + BC_VELOCITY_WALL:
        AzzVar = 1.0;
        Azz_qx_t30 = 3.0*AzzVar*(ux_t30 + 0.0);
        Azz_qy_t30 = 3.0*AzzVar*(uy_t30 + 0.0);
        Azz_qz_t30 = 3.0*AzzVar*(uz_t30 + 0.0);
    case EAST + BC_OUTFLOW: //copy values from last row - 1
        AzzVar = fMom[idxMom(threadIdx.x - 1, threadIdx.y, threadIdx.z, A_ZZ_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
        Azz_qx_t30 = fMom[idxMom(threadIdx.x - 1, threadIdx.y, threadIdx.z, A_ZZ_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
        Azz_qy_t30 = fMom[idxMom(threadIdx.x - 1, threadIdx.y, threadIdx.z, A_ZZ_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
        Azz_qz_t30 = fMom[idxMom(threadIdx.x - 1, threadIdx.y, threadIdx.z, A_ZZ_CZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
        break;
    default:
        break;
}
