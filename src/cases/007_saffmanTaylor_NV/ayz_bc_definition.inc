switch (nodeType) {
    case BULK:
        break;
    case NORTH:
        AyzVar     = (5 * fMom[idxMom(threadIdx.x, threadIdx.y - 1, threadIdx.z, A_YZ_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]
                      - 4 * fMom[idxMom(threadIdx.x, threadIdx.y - 2, threadIdx.z, A_YZ_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]
                      + fMom[idxMom(threadIdx.x, threadIdx.y - 3, threadIdx.z, A_YZ_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]) / 2.0;

        Ayz_qx_t30 = (5 * fMom[idxMom(threadIdx.x, threadIdx.y - 1, threadIdx.z, A_YZ_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]
                      - 4 * fMom[idxMom(threadIdx.x, threadIdx.y - 2, threadIdx.z, A_YZ_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]
                      + fMom[idxMom(threadIdx.x, threadIdx.y - 3, threadIdx.z, A_YZ_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]) * 0.5_df / F_M_I_SCALE;

        Ayz_qy_t30 = (5 * fMom[idxMom(threadIdx.x, threadIdx.y - 1, threadIdx.z, A_YZ_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]
                      - 4 * fMom[idxMom(threadIdx.x, threadIdx.y - 2, threadIdx.z, A_YZ_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]
                      + fMom[idxMom(threadIdx.x, threadIdx.y - 3, threadIdx.z, A_YZ_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]) * 0.5_df / F_M_I_SCALE;

        Ayz_qz_t30 = (5 * fMom[idxMom(threadIdx.x, threadIdx.y - 1, threadIdx.z, A_YZ_CZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]
                      - 4 * fMom[idxMom(threadIdx.x, threadIdx.y - 2, threadIdx.z, A_YZ_CZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]
                      + fMom[idxMom(threadIdx.x, threadIdx.y - 3, threadIdx.z, A_YZ_CZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]) * 0.5_df / F_M_I_SCALE;
        break;

    case SOUTH:
        AyzVar     = (5 * fMom[idxMom(threadIdx.x, threadIdx.y + 1, threadIdx.z, A_YZ_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]
                      - 4 * fMom[idxMom(threadIdx.x, threadIdx.y + 2, threadIdx.z, A_YZ_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]
                      + fMom[idxMom(threadIdx.x, threadIdx.y + 3, threadIdx.z, A_YZ_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]) / 2.0;

        Ayz_qx_t30 = (5 * fMom[idxMom(threadIdx.x, threadIdx.y + 1, threadIdx.z, A_YZ_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]
                      - 4 * fMom[idxMom(threadIdx.x, threadIdx.y + 2, threadIdx.z, A_YZ_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]
                      + fMom[idxMom(threadIdx.x, threadIdx.y + 3, threadIdx.z, A_YZ_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]) * 0.5_df / F_M_I_SCALE;

        Ayz_qy_t30 = (5 * fMom[idxMom(threadIdx.x, threadIdx.y + 1, threadIdx.z, A_YZ_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]
                      - 4 * fMom[idxMom(threadIdx.x, threadIdx.y + 2, threadIdx.z, A_YZ_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]
                      + fMom[idxMom(threadIdx.x, threadIdx.y + 3, threadIdx.z, A_YZ_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]) * 0.5_df / F_M_I_SCALE;

        Ayz_qz_t30 = (5 * fMom[idxMom(threadIdx.x, threadIdx.y + 1, threadIdx.z, A_YZ_CZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]
                      - 4 * fMom[idxMom(threadIdx.x, threadIdx.y + 2, threadIdx.z, A_YZ_CZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]
                      + fMom[idxMom(threadIdx.x, threadIdx.y + 3, threadIdx.z, A_YZ_CZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]) * 0.5_df / F_M_I_SCALE;
        break;

    case WEST: {
        // Prescribed fully-developed Oldroyd-B inlet.
        // u_x = 6 U_IN eta (1-eta), eta = y/(NY-1)
        // gamma_dot = du_x/dy = 6 U_IN/(NY-1) * (1-2 eta)
        const dfloat H_bc   = (dfloat)(NY - 1);
        const dfloat eta_bc = ((dfloat)y) / H_bc;
        const dfloat ux_bc  = 6.0_df * U_IN * eta_bc * (1.0_df - eta_bc);
        const dfloat gamma_bc = (6.0_df * U_IN / H_bc) * (1.0_df - 2.0_df * eta_bc);
        const dfloat Wi_local = LAMBDA * gamma_bc;

        // Physical Oldroyd-B fully-developed simple-shear value:
        // A_yz = 0
        AyzVar = CONF_ZERO;

        // Follow the same first-moment convention used by flow initialization.
        Ayz_qx_t30 = AyzVar * ux_bc;
        Ayz_qy_t30 = 0.0_df;
        Ayz_qz_t30 = 0.0_df;
        break;
    }

    case EAST: {
        const dfloat ux_out = fMom[idxMom(threadIdx.x - 1, threadIdx.y, threadIdx.z, M_UX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] / F_M_I_SCALE;
        if (ux_out < 0.0_df) {
            AyzVar = CONF_ZERO;
            Ayz_qx_t30 = Ayz_qy_t30 = Ayz_qz_t30 = 0.0_df;
            break;
        }
        // Homogeneous Neumann outlet: copy all A_yz moments from x-1.
        AyzVar = fMom[idxMom(threadIdx.x - 1, threadIdx.y, threadIdx.z, A_YZ_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
        Ayz_qx_t30 = fMom[idxMom(threadIdx.x - 1, threadIdx.y, threadIdx.z, A_YZ_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] / F_M_I_SCALE;
        Ayz_qy_t30 = fMom[idxMom(threadIdx.x - 1, threadIdx.y, threadIdx.z, A_YZ_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] / F_M_I_SCALE;
        Ayz_qz_t30 = fMom[idxMom(threadIdx.x - 1, threadIdx.y, threadIdx.z, A_YZ_CZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] / F_M_I_SCALE;
        break;
    }
    default:
        break;
}
