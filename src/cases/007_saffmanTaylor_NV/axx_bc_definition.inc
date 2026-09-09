switch (nodeType) {
    case BULK:
        break;
    case NORTH:
        AxxVar     = (5 * fMom[idxMom(threadIdx.x, threadIdx.y - 1, threadIdx.z, A_XX_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]
                    - 4 * fMom[idxMom(threadIdx.x, threadIdx.y - 2, threadIdx.z, A_XX_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]
                        + fMom[idxMom(threadIdx.x, threadIdx.y - 3, threadIdx.z, A_XX_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]) *0.5;
        
        Axx_qx_t30 = (5 * fMom[idxMom(threadIdx.x, threadIdx.y - 1, threadIdx.z, A_XX_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]
                    - 4 * fMom[idxMom(threadIdx.x, threadIdx.y - 2, threadIdx.z, A_XX_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]
                        + fMom[idxMom(threadIdx.x, threadIdx.y - 3, threadIdx.z, A_XX_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]) * 0.5_df / F_M_I_SCALE;

        Axx_qy_t30 = (5 * fMom[idxMom(threadIdx.x, threadIdx.y - 1, threadIdx.z, A_XX_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]
                    - 4 * fMom[idxMom(threadIdx.x, threadIdx.y - 2, threadIdx.z, A_XX_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]
                        + fMom[idxMom(threadIdx.x, threadIdx.y - 3, threadIdx.z, A_XX_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]) * 0.5_df / F_M_I_SCALE;

        Axx_qz_t30 = (5 * fMom[idxMom(threadIdx.x, threadIdx.y - 1, threadIdx.z, A_XX_CZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]
                    - 4 * fMom[idxMom(threadIdx.x, threadIdx.y - 2, threadIdx.z, A_XX_CZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]
                        + fMom[idxMom(threadIdx.x, threadIdx.y - 3, threadIdx.z, A_XX_CZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]) * 0.5_df / F_M_I_SCALE;
        break;

    case SOUTH:
        AxxVar     = (5 * fMom[idxMom(threadIdx.x, threadIdx.y + 1, threadIdx.z, A_XX_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]
                    - 4 * fMom[idxMom(threadIdx.x, threadIdx.y + 2, threadIdx.z, A_XX_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]
                        + fMom[idxMom(threadIdx.x, threadIdx.y + 3, threadIdx.z, A_XX_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]) *0.5;

        Axx_qx_t30 = (5 * fMom[idxMom(threadIdx.x, threadIdx.y + 1, threadIdx.z, A_XX_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]
                    - 4 * fMom[idxMom(threadIdx.x, threadIdx.y + 2, threadIdx.z, A_XX_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]
                        + fMom[idxMom(threadIdx.x, threadIdx.y + 3, threadIdx.z, A_XX_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]) * 0.5_df / F_M_I_SCALE;

        Axx_qy_t30 = (5 * fMom[idxMom(threadIdx.x, threadIdx.y + 1, threadIdx.z, A_XX_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]
                    - 4 * fMom[idxMom(threadIdx.x, threadIdx.y + 2, threadIdx.z, A_XX_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]
                        + fMom[idxMom(threadIdx.x, threadIdx.y + 3, threadIdx.z, A_XX_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]) * 0.5_df / F_M_I_SCALE;

        Axx_qz_t30 = (5 * fMom[idxMom(threadIdx.x, threadIdx.y + 1, threadIdx.z, A_XX_CZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]
                    - 4 * fMom[idxMom(threadIdx.x, threadIdx.y + 2, threadIdx.z, A_XX_CZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]
                        + fMom[idxMom(threadIdx.x, threadIdx.y + 3, threadIdx.z, A_XX_CZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]) * 0.5_df / F_M_I_SCALE;
        break;

    case SOUTH_BACK:
    case NORTH_BACK:
    case BACK: {
        // Prescribed fully-developed Oldroyd-B inlet.
        // u_z = 6 U_IN eta (1-eta), eta = y/(NY-1)
        const dfloat H_bc   = (dfloat)(NY - 1);
        const dfloat eta_bc = ((dfloat)y) / H_bc;
        const dfloat uz_bc  = 6.0_df * U_IN * eta_bc * (1.0_df - eta_bc);

        // Physical fully-developed simple-shear value: A_xx = 1.
        AxxVar = CONF_ZERO + 1.0_df;

        Axx_qx_t30 = 0.0_df;
        Axx_qy_t30 = 0.0_df;
        Axx_qz_t30 = AxxVar * uz_bc;
        break;
    }

    case SOUTH_FRONT:
    case NORTH_FRONT:
    case FRONT: {
        const dfloat uz_out = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z - 1, M_UZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] / F_M_I_SCALE;
        if (uz_out < 0.0_df) {
            AxxVar = CONF_ZERO + 1.0_df;
            Axx_qx_t30 = Axx_qy_t30 = Axx_qz_t30 = 0.0_df;
            break;
        }
        // Homogeneous Neumann outlet: copy all A_xx moments from z-1.
        AxxVar = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z - 1, A_XX_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
        Axx_qx_t30 = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z - 1, A_XX_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] / F_M_I_SCALE;
        Axx_qy_t30 = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z - 1, A_XX_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] / F_M_I_SCALE;
        Axx_qz_t30 = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z - 1, A_XX_CZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] / F_M_I_SCALE;
        break;
    }
    default:
        break;
}
