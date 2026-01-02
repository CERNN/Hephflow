// Lambda boundary condition definition mirroring Axx BC stencil from case 003
switch (nodeType) {
    case BULK:
        break;
    case NORTH:
        lambdaVar     = (5 * fMom[idxMom(threadIdx.x, threadIdx.y - 1, threadIdx.z, M4_LAMBDA_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]
                       - 4 * fMom[idxMom(threadIdx.x, threadIdx.y - 2, threadIdx.z, M4_LAMBDA_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]
                           + fMom[idxMom(threadIdx.x, threadIdx.y - 3, threadIdx.z, M4_LAMBDA_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]) * 0.5;

        lambda_qx_t30 = (5 * fMom[idxMom(threadIdx.x, threadIdx.y - 1, threadIdx.z, M4_LX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]
                       - 4 * fMom[idxMom(threadIdx.x, threadIdx.y - 2, threadIdx.z, M4_LX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]
                           + fMom[idxMom(threadIdx.x, threadIdx.y - 3, threadIdx.z, M4_LX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]) * 0.5;

        lambda_qy_t30 = (5 * fMom[idxMom(threadIdx.x, threadIdx.y - 1, threadIdx.z, M4_LY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]
                       - 4 * fMom[idxMom(threadIdx.x, threadIdx.y - 2, threadIdx.z, M4_LY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]
                           + fMom[idxMom(threadIdx.x, threadIdx.y - 3, threadIdx.z, M4_LY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]) * 0.5;

        lambda_qz_t30 = (5 * fMom[idxMom(threadIdx.x, threadIdx.y - 1, threadIdx.z, M4_LZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]
                       - 4 * fMom[idxMom(threadIdx.x, threadIdx.y - 2, threadIdx.z, M4_LZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]
                           + fMom[idxMom(threadIdx.x, threadIdx.y - 3, threadIdx.z, M4_LZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]) * 0.5;
        break;

    case SOUTH:
        lambdaVar     = (5 * fMom[idxMom(threadIdx.x, threadIdx.y + 1, threadIdx.z, M4_LAMBDA_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]
                       - 4 * fMom[idxMom(threadIdx.x, threadIdx.y + 2, threadIdx.z, M4_LAMBDA_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]
                           + fMom[idxMom(threadIdx.x, threadIdx.y + 3, threadIdx.z, M4_LAMBDA_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]) * 0.5;

        lambda_qx_t30 = (5 * fMom[idxMom(threadIdx.x, threadIdx.y + 1, threadIdx.z, M4_LX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]
                       - 4 * fMom[idxMom(threadIdx.x, threadIdx.y + 2, threadIdx.z, M4_LX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]
                           + fMom[idxMom(threadIdx.x, threadIdx.y + 3, threadIdx.z, M4_LX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]) * 0.5;

        lambda_qy_t30 = (5 * fMom[idxMom(threadIdx.x, threadIdx.y + 1, threadIdx.z, M4_LY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]
                       - 4 * fMom[idxMom(threadIdx.x, threadIdx.y + 2, threadIdx.z, M4_LY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]
                           + fMom[idxMom(threadIdx.x, threadIdx.y + 3, threadIdx.z, M4_LY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]) * 0.5;

        lambda_qz_t30 = (5 * fMom[idxMom(threadIdx.x, threadIdx.y + 1, threadIdx.z, M4_LZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]
                       - 4 * fMom[idxMom(threadIdx.x, threadIdx.y + 2, threadIdx.z, M4_LZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]
                           + fMom[idxMom(threadIdx.x, threadIdx.y + 3, threadIdx.z, M4_LZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]) * 0.5;
        break;

    default:
        break;
}

// Keep lambda within physical bounds at boundaries
lambdaVar = fmaxf(LAMBDA_ZERO, fminf(LAMBDA_ZERO + 1.0_df, lambdaVar));