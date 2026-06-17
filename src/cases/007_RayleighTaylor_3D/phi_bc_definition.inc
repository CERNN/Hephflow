//dfloat cc;
//dfloat sgw = (gW1 + 4.0*gW2);
dfloat invPhi;
switch (nodeType){
    case BULK:
        break;
    case NORTH:
            gNode[ 4] = gNode[ 3];
            gNode[13] = gNode[ 7];
            gNode[18] = gNode[11];
            gNode[ 8] = gNode[14];
            gNode[12] = gNode[17];
            phiVar = gNode[0] + gNode[1] + gNode[2] + gNode[3] + gNode[4] + gNode[5] + gNode[6] + gNode[7] + gNode[8] + gNode[9] + gNode[10] + gNode[11] + gNode[12] + gNode[13] + gNode[14] + gNode[15] + gNode[16] + gNode[17] + gNode[18];
            invPhi= 1.0/phiVar;

            phi_qx_t30 = ((gNode[1] - gNode[2] + gNode[7] - gNode[ 8] + gNode[ 9] - gNode[10] + gNode[13] - gNode[14] + gNode[15] - gNode[16]));
            phi_qy_t30 = ((gNode[3] - gNode[4] + gNode[7] - gNode[ 8] + gNode[11] - gNode[12] + gNode[14] - gNode[13] + gNode[17] - gNode[18]));
            phi_qz_t30 = ((gNode[5] - gNode[6] + gNode[9] - gNode[10] + gNode[11] - gNode[12] + gNode[16] - gNode[15] + gNode[18] - gNode[17]));
        break;
    case SOUTH:
            gNode[ 3] = gNode[ 4];
            gNode[ 7] = gNode[13];
            gNode[11] = gNode[18];
            gNode[14] = gNode[ 8];
            gNode[17] = gNode[12];
            phiVar = gNode[0] + gNode[1] + gNode[2] + gNode[3] + gNode[4] + gNode[5] + gNode[6] + gNode[7] + gNode[8] + gNode[9] + gNode[10] + gNode[11] + gNode[12] + gNode[13] + gNode[14] + gNode[15] + gNode[16] + gNode[17] + gNode[18];
            invPhi= 1.0/phiVar;

            phi_qx_t30 = ((gNode[1] - gNode[2] + gNode[7] - gNode[ 8] + gNode[ 9] - gNode[10] + gNode[13] - gNode[14] + gNode[15] - gNode[16]));
            phi_qy_t30 = ((gNode[3] - gNode[4] + gNode[7] - gNode[ 8] + gNode[11] - gNode[12] + gNode[14] - gNode[13] + gNode[17] - gNode[18]));
            phi_qz_t30 = ((gNode[5] - gNode[6] + gNode[9] - gNode[10] + gNode[11] - gNode[12] + gNode[16] - gNode[15] + gNode[18] - gNode[17]));
        break;
    default:
        break;
}
