switch (nodeType) {
    case BULK:
        break;
    case NORTH:
        AyzVar     = (5 * fMom[idxMom(threadIdx.x, threadIdx.y - 1, threadIdx.z, A_YZ_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]
                      - 4 * fMom[idxMom(threadIdx.x, threadIdx.y - 2, threadIdx.z, A_YZ_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]
                      + fMom[idxMom(threadIdx.x, threadIdx.y - 3, threadIdx.z, A_YZ_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]) / 2.0;

        Ayz_qx_t30 = (5 * fMom[idxMom(threadIdx.x, threadIdx.y - 1, threadIdx.z, A_YZ_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]
                      - 4 * fMom[idxMom(threadIdx.x, threadIdx.y - 2, threadIdx.z, A_YZ_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]
                      + fMom[idxMom(threadIdx.x, threadIdx.y - 3, threadIdx.z, A_YZ_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]) / 2.0;

        Ayz_qy_t30 = (5 * fMom[idxMom(threadIdx.x, threadIdx.y - 1, threadIdx.z, A_YZ_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]
                      - 4 * fMom[idxMom(threadIdx.x, threadIdx.y - 2, threadIdx.z, A_YZ_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]
                      + fMom[idxMom(threadIdx.x, threadIdx.y - 3, threadIdx.z, A_YZ_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]) / 2.0;

        Ayz_qz_t30 = (5 * fMom[idxMom(threadIdx.x, threadIdx.y - 1, threadIdx.z, A_YZ_CZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]
                      - 4 * fMom[idxMom(threadIdx.x, threadIdx.y - 2, threadIdx.z, A_YZ_CZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]
                      + fMom[idxMom(threadIdx.x, threadIdx.y - 3, threadIdx.z, A_YZ_CZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]) / 2.0;
        break;

    case SOUTH:
        AyzVar     = (5 * fMom[idxMom(threadIdx.x, threadIdx.y + 1, threadIdx.z, A_YZ_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]
                      - 4 * fMom[idxMom(threadIdx.x, threadIdx.y + 2, threadIdx.z, A_YZ_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]
                      + fMom[idxMom(threadIdx.x, threadIdx.y + 3, threadIdx.z, A_YZ_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]) / 2.0;

        Ayz_qx_t30 = (5 * fMom[idxMom(threadIdx.x, threadIdx.y + 1, threadIdx.z, A_YZ_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]
                      - 4 * fMom[idxMom(threadIdx.x, threadIdx.y + 2, threadIdx.z, A_YZ_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]
                      + fMom[idxMom(threadIdx.x, threadIdx.y + 3, threadIdx.z, A_YZ_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]) / 2.0;

        Ayz_qy_t30 = (5 * fMom[idxMom(threadIdx.x, threadIdx.y + 1, threadIdx.z, A_YZ_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]
                      - 4 * fMom[idxMom(threadIdx.x, threadIdx.y + 2, threadIdx.z, A_YZ_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]
                      + fMom[idxMom(threadIdx.x, threadIdx.y + 3, threadIdx.z, A_YZ_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]) / 2.0;

        Ayz_qz_t30 = (5 * fMom[idxMom(threadIdx.x, threadIdx.y + 1, threadIdx.z, A_YZ_CZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]
                      - 4 * fMom[idxMom(threadIdx.x, threadIdx.y + 2, threadIdx.z, A_YZ_CZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]
                      + fMom[idxMom(threadIdx.x, threadIdx.y + 3, threadIdx.z, A_YZ_CZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]) / 2.0;
        break;
    case WEST + BC_VELOCITY_WALL:
        AyzVar = 0.0;
        Ayz_qx_t30 = 3.0*AyzVar*(ux_t30 + 0.0);
        Ayz_qy_t30 = 3.0*AyzVar*(uy_t30 + 0.0);
        Ayz_qz_t30 = 3.0*AyzVar*(uz_t30 + 0.0);
    case EAST + BC_OUTFLOW: //copy values from last row - 1
        AyzVar = fMom[idxMom(threadIdx.x - 1, threadIdx.y, threadIdx.z, A_YZ_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
        Ayz_qx_t30 = fMom[idxMom(threadIdx.x - 1, threadIdx.y, threadIdx.z, A_YZ_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
        Ayz_qy_t30 = fMom[idxMom(threadIdx.x - 1, threadIdx.y, threadIdx.z, A_YZ_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
        Ayz_qz_t30 = fMom[idxMom(threadIdx.x - 1, threadIdx.y, threadIdx.z, A_YZ_CZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
        break;
    default:
        break;
}
