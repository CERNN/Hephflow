switch (nodeType) {
    case BULK:
        break;
    case NORTH:
        AxyVar     = (5 * fMom[idxMom(threadIdx.x, threadIdx.y - 1, threadIdx.z, A_XY_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]
                      - 4 * fMom[idxMom(threadIdx.x, threadIdx.y - 2, threadIdx.z, A_XY_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]
                      + fMom[idxMom(threadIdx.x, threadIdx.y - 3, threadIdx.z, A_XY_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]) *0.5;
        
        Axy_qx_t30 = (5 * fMom[idxMom(threadIdx.x, threadIdx.y - 1, threadIdx.z, A_XY_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]
                      - 4 * fMom[idxMom(threadIdx.x, threadIdx.y - 2, threadIdx.z, A_XY_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]
                      + fMom[idxMom(threadIdx.x, threadIdx.y - 3, threadIdx.z, A_XY_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]) * 0.5_df / F_M_I_SCALE;

        Axy_qy_t30 = (5 * fMom[idxMom(threadIdx.x, threadIdx.y - 1, threadIdx.z, A_XY_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]
                      - 4 * fMom[idxMom(threadIdx.x, threadIdx.y - 2, threadIdx.z, A_XY_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]
                      + fMom[idxMom(threadIdx.x, threadIdx.y - 3, threadIdx.z, A_XY_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]) * 0.5_df / F_M_I_SCALE;

        Axy_qz_t30 = (5 * fMom[idxMom(threadIdx.x, threadIdx.y - 1, threadIdx.z, A_XY_CZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]
                      - 4 * fMom[idxMom(threadIdx.x, threadIdx.y - 2, threadIdx.z, A_XY_CZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]
                      + fMom[idxMom(threadIdx.x, threadIdx.y - 3, threadIdx.z, A_XY_CZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]) * 0.5_df / F_M_I_SCALE;
        break;

    case SOUTH:
        AxyVar     = (5 * fMom[idxMom(threadIdx.x, threadIdx.y + 1, threadIdx.z, A_XY_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]
                      - 4 * fMom[idxMom(threadIdx.x, threadIdx.y + 2, threadIdx.z, A_XY_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]
                      + fMom[idxMom(threadIdx.x, threadIdx.y + 3, threadIdx.z, A_XY_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]) *0.5;

        Axy_qx_t30 = (5 * fMom[idxMom(threadIdx.x, threadIdx.y + 1, threadIdx.z, A_XY_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]
                      - 4 * fMom[idxMom(threadIdx.x, threadIdx.y + 2, threadIdx.z, A_XY_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]
                      + fMom[idxMom(threadIdx.x, threadIdx.y + 3, threadIdx.z, A_XY_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]) * 0.5_df / F_M_I_SCALE;

        Axy_qy_t30 = (5 * fMom[idxMom(threadIdx.x, threadIdx.y + 1, threadIdx.z, A_XY_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]
                      - 4 * fMom[idxMom(threadIdx.x, threadIdx.y + 2, threadIdx.z, A_XY_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]
                      + fMom[idxMom(threadIdx.x, threadIdx.y + 3, threadIdx.z, A_XY_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]) * 0.5_df / F_M_I_SCALE;

        Axy_qz_t30 = (5 * fMom[idxMom(threadIdx.x, threadIdx.y + 1, threadIdx.z, A_XY_CZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]
                      - 4 * fMom[idxMom(threadIdx.x, threadIdx.y + 2, threadIdx.z, A_XY_CZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]
                      + fMom[idxMom(threadIdx.x, threadIdx.y + 3, threadIdx.z, A_XY_CZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]) * 0.5_df / F_M_I_SCALE;
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
        // A_xy = lambda du_x/dy
        AxyVar = CONF_ZERO + Wi_local;

        // Follow the same first-moment convention used by flow initialization.
        Axy_qx_t30 = AxyVar * ux_bc;
        Axy_qy_t30 = 0.0_df;
        Axy_qz_t30 = 0.0_df;
        break;
    }

    case EAST: {
        const dfloat ux_out = fMom[idxMom(threadIdx.x - 1, threadIdx.y, threadIdx.z, M_UX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] / F_M_I_SCALE;
        if (ux_out < 0.0_df) {
            AxyVar = CONF_ZERO;
            Axy_qx_t30 = Axy_qy_t30 = Axy_qz_t30 = 0.0_df;
            break;
        }
        // Homogeneous Neumann outlet: copy all A_xy moments from x-1.
        AxyVar = fMom[idxMom(threadIdx.x - 1, threadIdx.y, threadIdx.z, A_XY_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
        Axy_qx_t30 = fMom[idxMom(threadIdx.x - 1, threadIdx.y, threadIdx.z, A_XY_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] / F_M_I_SCALE;
        Axy_qy_t30 = fMom[idxMom(threadIdx.x - 1, threadIdx.y, threadIdx.z, A_XY_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] / F_M_I_SCALE;
        Axy_qz_t30 = fMom[idxMom(threadIdx.x - 1, threadIdx.y, threadIdx.z, A_XY_CZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] / F_M_I_SCALE;
        break;
    }
    default:
        break;
}
