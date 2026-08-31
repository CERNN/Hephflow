    /* write to global pop - face neighbors only for D3G7, all for D3G19 */
    if (INTERFACE_BC_WEST){//w
         ghostInterface.lambda_gGhost.X_0[g_idxPopX(ty, tz, 0, bx, by, bz)] = gNode[ 2]; 
#ifdef D3G19
         ghostInterface.lambda_gGhost.X_0[g_idxPopX(ty, tz, 1, bx, by, bz)] = gNode[ 8];
         ghostInterface.lambda_gGhost.X_0[g_idxPopX(ty, tz, 2, bx, by, bz)] = gNode[10];
         ghostInterface.lambda_gGhost.X_0[g_idxPopX(ty, tz, 3, bx, by, bz)] = gNode[14];
         ghostInterface.lambda_gGhost.X_0[g_idxPopX(ty, tz, 4, bx, by, bz)] = gNode[16];
#endif
    }if (INTERFACE_BC_EAST){//e
         ghostInterface.lambda_gGhost.X_1[g_idxPopX(ty, tz, 0, bx, by, bz)] = gNode[ 1];
#ifdef D3G19
         ghostInterface.lambda_gGhost.X_1[g_idxPopX(ty, tz, 1, bx, by, bz)] = gNode[ 7];
         ghostInterface.lambda_gGhost.X_1[g_idxPopX(ty, tz, 2, bx, by, bz)] = gNode[ 9];
         ghostInterface.lambda_gGhost.X_1[g_idxPopX(ty, tz, 3, bx, by, bz)] = gNode[13];
         ghostInterface.lambda_gGhost.X_1[g_idxPopX(ty, tz, 4, bx, by, bz)] = gNode[15];   
#endif
    }

    if (INTERFACE_BC_SOUTH){//s          
         ghostInterface.lambda_gGhost.Y_0[g_idxPopY(tx, tz, 0, bx, by, bz)] = gNode[ 4];
#ifdef D3G19
         ghostInterface.lambda_gGhost.Y_0[g_idxPopY(tx, tz, 1, bx, by, bz)] = gNode[ 8];
         ghostInterface.lambda_gGhost.Y_0[g_idxPopY(tx, tz, 2, bx, by, bz)] = gNode[12];
         ghostInterface.lambda_gGhost.Y_0[g_idxPopY(tx, tz, 3, bx, by, bz)] = gNode[13];
         ghostInterface.lambda_gGhost.Y_0[g_idxPopY(tx, tz, 4, bx, by, bz)] = gNode[18];
#endif
    }if (INTERFACE_BC_NORTH){//n
         ghostInterface.lambda_gGhost.Y_1[g_idxPopY(tx, tz, 0, bx, by, bz)] = gNode[ 3];
#ifdef D3G19
         ghostInterface.lambda_gGhost.Y_1[g_idxPopY(tx, tz, 1, bx, by, bz)] = gNode[ 7];
         ghostInterface.lambda_gGhost.Y_1[g_idxPopY(tx, tz, 2, bx, by, bz)] = gNode[11];
         ghostInterface.lambda_gGhost.Y_1[g_idxPopY(tx, tz, 3, bx, by, bz)] = gNode[14];
         ghostInterface.lambda_gGhost.Y_1[g_idxPopY(tx, tz, 4, bx, by, bz)] = gNode[17];
#endif
    }

    if (INTERFACE_BC_BACK){//b
         ghostInterface.lambda_gGhost.Z_0[g_idxPopZ(tx, ty, 0, bx, by, bz)] = gNode[ 6];
#ifdef D3G19
         ghostInterface.lambda_gGhost.Z_0[g_idxPopZ(tx, ty, 1, bx, by, bz)] = gNode[10];
         ghostInterface.lambda_gGhost.Z_0[g_idxPopZ(tx, ty, 2, bx, by, bz)] = gNode[12];
         ghostInterface.lambda_gGhost.Z_0[g_idxPopZ(tx, ty, 3, bx, by, bz)] = gNode[15];
         ghostInterface.lambda_gGhost.Z_0[g_idxPopZ(tx, ty, 4, bx, by, bz)] = gNode[17];
#endif
    }if (INTERFACE_BC_FRONT){//f
         ghostInterface.lambda_gGhost.Z_1[g_idxPopZ(tx, ty, 0, bx, by, bz)] = gNode[ 5];
#ifdef D3G19
         ghostInterface.lambda_gGhost.Z_1[g_idxPopZ(tx, ty, 1, bx, by, bz)] = gNode[ 9];
         ghostInterface.lambda_gGhost.Z_1[g_idxPopZ(tx, ty, 2, bx, by, bz)] = gNode[11];
         ghostInterface.lambda_gGhost.Z_1[g_idxPopZ(tx, ty, 3, bx, by, bz)] = gNode[16];
         ghostInterface.lambda_gGhost.Z_1[g_idxPopZ(tx, ty, 4, bx, by, bz)] = gNode[18];
#endif
    }
