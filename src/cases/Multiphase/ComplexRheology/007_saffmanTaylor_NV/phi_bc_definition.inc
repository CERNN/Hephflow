//dfloat cc;
//dfloat sgw = (gW1 + 4.0*gW2);
switch (nodeType){
    case BULK:
        break;
    case NORTH:
            // bc_initialization.inc gives the wall precedence at the inlet
            // corners, so explicitly retain the injected phase there.
            if (z == 0) {
                phiVar = PHI_TWO;
                phi_qx_t30 = 0.0_df;
                phi_qy_t30 = 0.0_df;
                phi_qz_t30 = 0.0_df;
                break;
            }
            gNode[ 4] = gNode[ 3];
            gNode[ 8] = gNode[14];
            gNode[12] = gNode[17];
            gNode[13] = gNode[ 7];
            gNode[18] = gNode[11];
            phiVar = gNode[0] + gNode[1] + gNode[2] + gNode[3] + gNode[4] + gNode[5] + gNode[6] + gNode[7] + gNode[8] + gNode[9] + gNode[10] + gNode[11] + gNode[12] + gNode[13] + gNode[14] + gNode[15] + gNode[16] + gNode[17] + gNode[18];
            phiVar = myMin(myMax(phiVar, PHI_ONE), PHI_TWO);

            phi_qx_t30 = (gNode[1] - gNode[2] + gNode[7] - gNode[ 8] + gNode[ 9] - gNode[10] + gNode[13] - gNode[14] + gNode[15] - gNode[16]);
            phi_qy_t30 = (gNode[3] - gNode[4] + gNode[7] - gNode[ 8] + gNode[11] - gNode[12] + gNode[14] - gNode[13] + gNode[17] - gNode[18]);
            phi_qz_t30 = (gNode[5] - gNode[6] + gNode[9] - gNode[10] + gNode[11] - gNode[12] + gNode[16] - gNode[15] + gNode[18] - gNode[17]);
        break;
    case SOUTH:
            // Same inlet-corner treatment as the NORTH wall.
            if (z == 0) {
                phiVar = PHI_TWO;
                phi_qx_t30 = 0.0_df;
                phi_qy_t30 = 0.0_df;
                phi_qz_t30 = 0.0_df;
                break;
            }
            gNode[ 3] = gNode[ 4];
            gNode[ 7] = gNode[13];
            gNode[11] = gNode[18];
            gNode[14] = gNode[ 8];
            gNode[17] = gNode[12];
            phiVar = gNode[0] + gNode[1] + gNode[2] + gNode[3] + gNode[4] + gNode[5] + gNode[6] + gNode[7] + gNode[8] + gNode[9] + gNode[10] + gNode[11] + gNode[12] + gNode[13] + gNode[14] + gNode[15] + gNode[16] + gNode[17] + gNode[18];
            phiVar = myMin(myMax(phiVar, PHI_ONE), PHI_TWO);

            phi_qx_t30 = (gNode[1] - gNode[2] + gNode[7] - gNode[ 8] + gNode[ 9] - gNode[10] + gNode[13] - gNode[14] + gNode[15] - gNode[16]);
            phi_qy_t30 = (gNode[3] - gNode[4] + gNode[7] - gNode[ 8] + gNode[11] - gNode[12] + gNode[14] - gNode[13] + gNode[17] - gNode[18]);
            phi_qz_t30 = (gNode[5] - gNode[6] + gNode[9] - gNode[10] + gNode[11] - gNode[12] + gNode[16] - gNode[15] + gNode[18] - gNode[17]);
        break;
    case BACK:
    case SOUTH_BACK:
    case NORTH_BACK:
    {
        phiVar = PHI_TWO;

        phi_qx_t30 = 0.0_df;
        phi_qy_t30 = 0.0_df;
        phi_qz_t30 = PHI_TWO * (6.0_df * U_IN * ((dfloat)y) / ((dfloat)(NY - 1)) * (1.0_df - ((dfloat)y) / ((dfloat)(NY - 1))));

        break;
    }
    case FRONT:
    case SOUTH_FRONT:
    case NORTH_FRONT:
    {
        const dfloat uz_out = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z - 1,
                                          M_UZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] / F_M_I_SCALE;
        if (uz_out < 0.0_df) {
            // Backflow must have a prescribed incoming state. The fluid beyond
            // the outlet is the initially resident Newtonian phase.
            phiVar = PHI_ONE;
            phi_qx_t30 = 0.0_df;
            phi_qy_t30 = 0.0_df;
            phi_qz_t30 = 0.0_df;
            break;
        }

        phiVar = myMin(myMax(
            fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z - 1,
                        M3_PHI_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)],
            PHI_ONE), PHI_TWO);

        // Rebuild an equilibrium advective flux instead of copying the
        // nonequilibrium transport moments. Copying them lets the weakly
        // damped outlet mode reflect and propagate upstream.
        phi_qx_t30 = 0.0_df;
        phi_qy_t30 = 0.0_df;
        phi_qz_t30 = phiVar * uz_out;

        break;
    }
    default:
        break;
}
