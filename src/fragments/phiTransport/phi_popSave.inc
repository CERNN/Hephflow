    /* write to global pop - AA scheme */
    dfloat* storeGhostXWest  = ghostInterfacePhi.X_0;
    dfloat* storeGhostXEast  = ghostInterfacePhi.X_1;
    dfloat* storeGhostYSouth = ghostInterfacePhi.Y_0;
    dfloat* storeGhostYNorth = ghostInterfacePhi.Y_1;
    dfloat* storeGhostZBack  = ghostInterfacePhi.Z_0;
    dfloat* storeGhostZFront = ghostInterfacePhi.Z_1;

    const int storeBxWest  = (stepParity ? bx : bxm1);
    const int storeBxEast  = (stepParity ? bx : bxp1);
    const int storeBySouth = (stepParity ? by : bym1);
    const int storeByNorth = (stepParity ? by : byp1);
    const int storeBzBack  = (stepParity ? bz : bzm1);
    const int storeBzFront = (stepParity ? bz : bzp1);

    if(stepParity){
        if (INTERFACE_BC_WEST){//w
             storeGhostXWest[g_idxPopX(ty, tz, 0, storeBxWest, by, bz)] = gNode[ 2]; 
#ifdef D3G19
             storeGhostXWest[g_idxPopX(ty, tz, 1, storeBxWest, by, bz)] = gNode[ 8];
             storeGhostXWest[g_idxPopX(ty, tz, 2, storeBxWest, by, bz)] = gNode[10];
             storeGhostXWest[g_idxPopX(ty, tz, 3, storeBxWest, by, bz)] = gNode[14];
             storeGhostXWest[g_idxPopX(ty, tz, 4, storeBxWest, by, bz)] = gNode[16];
#endif
#ifdef D3G27
             storeGhostXWest[g_idxPopX(ty, tz, 1, storeBxWest, by, bz)] = gNode[ 8];
             storeGhostXWest[g_idxPopX(ty, tz, 2, storeBxWest, by, bz)] = gNode[10];
             storeGhostXWest[g_idxPopX(ty, tz, 3, storeBxWest, by, bz)] = gNode[14];
             storeGhostXWest[g_idxPopX(ty, tz, 4, storeBxWest, by, bz)] = gNode[16];
             storeGhostXWest[g_idxPopX(ty, tz, 5, storeBxWest, by, bz)] = gNode[20];
             storeGhostXWest[g_idxPopX(ty, tz, 6, storeBxWest, by, bz)] = gNode[22];
             storeGhostXWest[g_idxPopX(ty, tz, 7, storeBxWest, by, bz)] = gNode[24];
             storeGhostXWest[g_idxPopX(ty, tz, 8, storeBxWest, by, bz)] = gNode[25];
#endif
        }
        if (INTERFACE_BC_EAST){//e
             storeGhostXEast[g_idxPopX(ty, tz, 0, storeBxEast, by, bz)] = gNode[ 1]; 
#ifdef D3G19
             storeGhostXEast[g_idxPopX(ty, tz, 1, storeBxEast, by, bz)] = gNode[ 7];
             storeGhostXEast[g_idxPopX(ty, tz, 2, storeBxEast, by, bz)] = gNode[ 9];
             storeGhostXEast[g_idxPopX(ty, tz, 3, storeBxEast, by, bz)] = gNode[13];
             storeGhostXEast[g_idxPopX(ty, tz, 4, storeBxEast, by, bz)] = gNode[15];
/* DEV_REF_START (from dev during merge)
    write to global pop - face neighbors only for D3G7, all for D3G19
    if (INTERFACE_BC_WEST){//w
        ghostInterface.phi_gGhost.X_0[g_idxPopX(ty, tz, 0, bx, by, bz)] = gNode[ 2]; 
#ifdef D3G19
        ghostInterface.phi_gGhost.X_0[g_idxPopX(ty, tz, 1, bx, by, bz)] = gNode[ 8];
        ghostInterface.phi_gGhost.X_0[g_idxPopX(ty, tz, 2, bx, by, bz)] = gNode[10];
        ghostInterface.phi_gGhost.X_0[g_idxPopX(ty, tz, 3, bx, by, bz)] = gNode[14];
        ghostInterface.phi_gGhost.X_0[g_idxPopX(ty, tz, 4, bx, by, bz)] = gNode[16];
#endif
#ifdef D3G27
        ghostInterface.phi_gGhost.X_0[g_idxPopX(ty, tz, 1, bx, by, bz)] = gNode[ 8];
        ghostInterface.phi_gGhost.X_0[g_idxPopX(ty, tz, 2, bx, by, bz)] = gNode[10];
        ghostInterface.phi_gGhost.X_0[g_idxPopX(ty, tz, 3, bx, by, bz)] = gNode[14];
        ghostInterface.phi_gGhost.X_0[g_idxPopX(ty, tz, 4, bx, by, bz)] = gNode[16];
        ghostInterface.phi_gGhost.X_0[g_idxPopX(ty, tz, 5, bx, by, bz)] = gNode[20];
#ifdef D3G27
             storeGhostXEast[g_idxPopX(ty, tz, 1, storeBxEast, by, bz)] = gNode[ 7];
             storeGhostXEast[g_idxPopX(ty, tz, 2, storeBxEast, by, bz)] = gNode[ 9];
             storeGhostXEast[g_idxPopX(ty, tz, 3, storeBxEast, by, bz)] = gNode[13];
             storeGhostXEast[g_idxPopX(ty, tz, 4, storeBxEast, by, bz)] = gNode[15];
             storeGhostXEast[g_idxPopX(ty, tz, 5, storeBxEast, by, bz)] = gNode[19];
             storeGhostXEast[g_idxPopX(ty, tz, 6, storeBxEast, by, bz)] = gNode[21];
             storeGhostXEast[g_idxPopX(ty, tz, 7, storeBxEast, by, bz)] = gNode[23];
             storeGhostXEast[g_idxPopX(ty, tz, 8, storeBxEast, by, bz)] = gNode[26];
#endif
        ghostInterface.phi_gGhost.X_0[g_idxPopX(ty, tz, 6, bx, by, bz)] = gNode[22];
        ghostInterface.phi_gGhost.X_0[g_idxPopX(ty, tz, 7, bx, by, bz)] = gNode[24];
        ghostInterface.phi_gGhost.X_0[g_idxPopX(ty, tz, 8, bx, by, bz)] = gNode[25];
#endif
    }if (INTERFACE_BC_EAST){//e
        ghostInterface.phi_gGhost.X_1[g_idxPopX(ty, tz, 0, bx, by, bz)] = gNode[ 1];
#ifdef D3G19
        ghostInterface.phi_gGhost.X_1[g_idxPopX(ty, tz, 1, bx, by, bz)] = gNode[ 7];
        ghostInterface.phi_gGhost.X_1[g_idxPopX(ty, tz, 2, bx, by, bz)] = gNode[ 9];
        ghostInterface.phi_gGhost.X_1[g_idxPopX(ty, tz, 3, bx, by, bz)] = gNode[13];
        ghostInterface.phi_gGhost.X_1[g_idxPopX(ty, tz, 4, bx, by, bz)] = gNode[15];   
#endif
#ifdef D3G27
        ghostInterface.phi_gGhost.X_1[g_idxPopX(ty, tz, 1, bx, by, bz)] = gNode[ 7];
        ghostInterface.phi_gGhost.X_1[g_idxPopX(ty, tz, 2, bx, by, bz)] = gNode[ 9];
        ghostInterface.phi_gGhost.X_1[g_idxPopX(ty, tz, 3, bx, by, bz)] = gNode[13];
        ghostInterface.phi_gGhost.X_1[g_idxPopX(ty, tz, 4, bx, by, bz)] = gNode[15]; 
        ghostInterface.phi_gGhost.X_1[g_idxPopX(ty, tz, 5, bx, by, bz)] = gNode[19];
        ghostInterface.phi_gGhost.X_1[g_idxPopX(ty, tz, 6, bx, by, bz)] = gNode[21];
        ghostInterface.phi_gGhost.X_1[g_idxPopX(ty, tz, 7, bx, by, bz)] = gNode[23];
        ghostInterface.phi_gGhost.X_1[g_idxPopX(ty, tz, 8, bx, by, bz)] = gNode[26];  
#endif
    }

    if (INTERFACE_BC_SOUTH){//s          
        ghostInterface.phi_gGhost.Y_0[g_idxPopY(tx, tz, 0, bx, by, bz)] = gNode[ 4];
#ifdef D3G19
        ghostInterface.phi_gGhost.Y_0[g_idxPopY(tx, tz, 1, bx, by, bz)] = gNode[ 8];
        ghostInterface.phi_gGhost.Y_0[g_idxPopY(tx, tz, 2, bx, by, bz)] = gNode[12];
        ghostInterface.phi_gGhost.Y_0[g_idxPopY(tx, tz, 3, bx, by, bz)] = gNode[13];
        ghostInterface.phi_gGhost.Y_0[g_idxPopY(tx, tz, 4, bx, by, bz)] = gNode[18];
#endif
#ifdef D3G27
        ghostInterface.phi_gGhost.Y_0[g_idxPopY(tx, tz, 1, bx, by, bz)] = gNode[ 8];
        ghostInterface.phi_gGhost.Y_0[g_idxPopY(tx, tz, 2, bx, by, bz)] = gNode[12];
        ghostInterface.phi_gGhost.Y_0[g_idxPopY(tx, tz, 3, bx, by, bz)] = gNode[13];
        ghostInterface.phi_gGhost.Y_0[g_idxPopY(tx, tz, 4, bx, by, bz)] = gNode[18];
        ghostInterface.phi_gGhost.Y_0[g_idxPopY(tx, tz, 5, bx, by, bz)] = gNode[20];
        ghostInterface.phi_gGhost.Y_0[g_idxPopY(tx, tz, 6, bx, by, bz)] = gNode[22];
        ghostInterface.phi_gGhost.Y_0[g_idxPopY(tx, tz, 7, bx, by, bz)] = gNode[23];
        ghostInterface.phi_gGhost.Y_0[g_idxPopY(tx, tz, 8, bx, by, bz)] = gNode[26];
#endif
    }if (INTERFACE_BC_NORTH){//n
        ghostInterface.phi_gGhost.Y_1[g_idxPopY(tx, tz, 0, bx, by, bz)] = gNode[ 3];
#ifdef D3G19
        ghostInterface.phi_gGhost.Y_1[g_idxPopY(tx, tz, 1, bx, by, bz)] = gNode[ 7];
        ghostInterface.phi_gGhost.Y_1[g_idxPopY(tx, tz, 2, bx, by, bz)] = gNode[11];
        ghostInterface.phi_gGhost.Y_1[g_idxPopY(tx, tz, 3, bx, by, bz)] = gNode[14];
        ghostInterface.phi_gGhost.Y_1[g_idxPopY(tx, tz, 4, bx, by, bz)] = gNode[17];
#endif
#ifdef D3G27
        ghostInterface.phi_gGhost.Y_1[g_idxPopY(tx, tz, 1, bx, by, bz)] = gNode[ 7];
        ghostInterface.phi_gGhost.Y_1[g_idxPopY(tx, tz, 2, bx, by, bz)] = gNode[11];
        ghostInterface.phi_gGhost.Y_1[g_idxPopY(tx, tz, 3, bx, by, bz)] = gNode[14];
        ghostInterface.phi_gGhost.Y_1[g_idxPopY(tx, tz, 4, bx, by, bz)] = gNode[17];
        ghostInterface.phi_gGhost.Y_1[g_idxPopY(tx, tz, 5, bx, by, bz)] = gNode[19];
        ghostInterface.phi_gGhost.Y_1[g_idxPopY(tx, tz, 6, bx, by, bz)] = gNode[21];
        ghostInterface.phi_gGhost.Y_1[g_idxPopY(tx, tz, 7, bx, by, bz)] = gNode[24];
        ghostInterface.phi_gGhost.Y_1[g_idxPopY(tx, tz, 8, bx, by, bz)] = gNode[25];

DEV_REF_END */
#endif
        }

        if (INTERFACE_BC_SOUTH){//s
             storeGhostYSouth[g_idxPopY(tx, tz, 0, bx, storeBySouth, bz)] = gNode[ 4]; 
#ifdef D3G19
             storeGhostYSouth[g_idxPopY(tx, tz, 1, bx, storeBySouth, bz)] = gNode[ 8];
             storeGhostYSouth[g_idxPopY(tx, tz, 2, bx, storeBySouth, bz)] = gNode[12];
             storeGhostYSouth[g_idxPopY(tx, tz, 3, bx, storeBySouth, bz)] = gNode[13];
             storeGhostYSouth[g_idxPopY(tx, tz, 4, bx, storeBySouth, bz)] = gNode[18];
#endif
#ifdef D3G27
             storeGhostYSouth[g_idxPopY(tx, tz, 1, bx, storeBySouth, bz)] = gNode[ 8];
             storeGhostYSouth[g_idxPopY(tx, tz, 2, bx, storeBySouth, bz)] = gNode[12];
             storeGhostYSouth[g_idxPopY(tx, tz, 3, bx, storeBySouth, bz)] = gNode[13];
             storeGhostYSouth[g_idxPopY(tx, tz, 4, bx, storeBySouth, bz)] = gNode[18];
             storeGhostYSouth[g_idxPopY(tx, tz, 5, bx, storeBySouth, bz)] = gNode[20];
             storeGhostYSouth[g_idxPopY(tx, tz, 6, bx, storeBySouth, bz)] = gNode[22];
             storeGhostYSouth[g_idxPopY(tx, tz, 7, bx, storeBySouth, bz)] = gNode[23];
             storeGhostYSouth[g_idxPopY(tx, tz, 8, bx, storeBySouth, bz)] = gNode[26];
#endif
        }
        if (INTERFACE_BC_NORTH){//n
             storeGhostYNorth[g_idxPopY(tx, tz, 0, bx, storeByNorth, bz)] = gNode[ 3]; 
#ifdef D3G19
             storeGhostYNorth[g_idxPopY(tx, tz, 1, bx, storeByNorth, bz)] = gNode[ 7];
             storeGhostYNorth[g_idxPopY(tx, tz, 2, bx, storeByNorth, bz)] = gNode[11];
             storeGhostYNorth[g_idxPopY(tx, tz, 3, bx, storeByNorth, bz)] = gNode[14];
             storeGhostYNorth[g_idxPopY(tx, tz, 4, bx, storeByNorth, bz)] = gNode[17];
#endif
#ifdef D3G27
             storeGhostYNorth[g_idxPopY(tx, tz, 1, bx, storeByNorth, bz)] = gNode[ 7];
             storeGhostYNorth[g_idxPopY(tx, tz, 2, bx, storeByNorth, bz)] = gNode[11];
             storeGhostYNorth[g_idxPopY(tx, tz, 3, bx, storeByNorth, bz)] = gNode[14];
             storeGhostYNorth[g_idxPopY(tx, tz, 4, bx, storeByNorth, bz)] = gNode[17];
             storeGhostYNorth[g_idxPopY(tx, tz, 5, bx, storeByNorth, bz)] = gNode[19];
             storeGhostYNorth[g_idxPopY(tx, tz, 6, bx, storeByNorth, bz)] = gNode[21];
             storeGhostYNorth[g_idxPopY(tx, tz, 7, bx, storeByNorth, bz)] = gNode[24];
             storeGhostYNorth[g_idxPopY(tx, tz, 8, bx, storeByNorth, bz)] = gNode[25];
#endif
        }

        if (INTERFACE_BC_BACK){//b
             storeGhostZBack[g_idxPopZ(tx, ty, 0, bx, by, storeBzBack)] = gNode[ 6]; 
#ifdef D3G19
             storeGhostZBack[g_idxPopZ(tx, ty, 1, bx, by, storeBzBack)] = gNode[10];
             storeGhostZBack[g_idxPopZ(tx, ty, 2, bx, by, storeBzBack)] = gNode[12];
             storeGhostZBack[g_idxPopZ(tx, ty, 3, bx, by, storeBzBack)] = gNode[15];
             storeGhostZBack[g_idxPopZ(tx, ty, 4, bx, by, storeBzBack)] = gNode[17];
#endif
#ifdef D3G27
             storeGhostZBack[g_idxPopZ(tx, ty, 1, bx, by, storeBzBack)] = gNode[10];
             storeGhostZBack[g_idxPopZ(tx, ty, 2, bx, by, storeBzBack)] = gNode[12];
             storeGhostZBack[g_idxPopZ(tx, ty, 3, bx, by, storeBzBack)] = gNode[15];
             storeGhostZBack[g_idxPopZ(tx, ty, 4, bx, by, storeBzBack)] = gNode[17];
             storeGhostZBack[g_idxPopZ(tx, ty, 5, bx, by, storeBzBack)] = gNode[20];
             storeGhostZBack[g_idxPopZ(tx, ty, 6, bx, by, storeBzBack)] = gNode[21];
             storeGhostZBack[g_idxPopZ(tx, ty, 7, bx, by, storeBzBack)] = gNode[24];
             storeGhostZBack[g_idxPopZ(tx, ty, 8, bx, by, storeBzBack)] = gNode[26];
#endif
        }
        if (INTERFACE_BC_FRONT){//f
             storeGhostZFront[g_idxPopZ(tx, ty, 0, bx, by, storeBzFront)] = gNode[ 5]; 
#ifdef D3G19
             storeGhostZFront[g_idxPopZ(tx, ty, 1, bx, by, storeBzFront)] = gNode[ 9];
             storeGhostZFront[g_idxPopZ(tx, ty, 2, bx, by, storeBzFront)] = gNode[11];
             storeGhostZFront[g_idxPopZ(tx, ty, 3, bx, by, storeBzFront)] = gNode[16];
             storeGhostZFront[g_idxPopZ(tx, ty, 4, bx, by, storeBzFront)] = gNode[18];
#endif
#ifdef D3G27
             storeGhostZFront[g_idxPopZ(tx, ty, 1, bx, by, storeBzFront)] = gNode[ 9];
             storeGhostZFront[g_idxPopZ(tx, ty, 2, bx, by, storeBzFront)] = gNode[11];
             storeGhostZFront[g_idxPopZ(tx, ty, 3, bx, by, storeBzFront)] = gNode[16];
             storeGhostZFront[g_idxPopZ(tx, ty, 4, bx, by, storeBzFront)] = gNode[18];
             storeGhostZFront[g_idxPopZ(tx, ty, 5, bx, by, storeBzFront)] = gNode[19];
             storeGhostZFront[g_idxPopZ(tx, ty, 6, bx, by, storeBzFront)] = gNode[22];
             storeGhostZFront[g_idxPopZ(tx, ty, 7, bx, by, storeBzFront)] = gNode[23];
             storeGhostZFront[g_idxPopZ(tx, ty, 8, bx, by, storeBzFront)] = gNode[25];
#endif
        }

    }else{
        if (INTERFACE_BC_WEST){//w
             storeGhostXEast[g_idxPopX(ty, tz, 0, storeBxWest, by, bz)] = gNode[ 2]; 
#ifdef D3G19
             storeGhostXEast[g_idxPopX(tym1, tz, 1, storeBxWest, byAtSouth, bz)] = gNode[ 8];
             storeGhostXEast[g_idxPopX(ty, tzm1, 2, storeBxWest, by, bzAtBack)] = gNode[10];
             storeGhostXEast[g_idxPopX(typ1, tz, 3, storeBxWest, byAtNorth, bz)] = gNode[14];
             storeGhostXEast[g_idxPopX(ty, tzp1, 4, storeBxWest, by, bzAtFront)] = gNode[16];
#endif
#ifdef D3G27
             storeGhostXEast[g_idxPopX(tym1, tz, 1, storeBxWest, byAtSouth, bz)] = gNode[ 8];
             storeGhostXEast[g_idxPopX(ty, tzm1, 2, storeBxWest, by, bzAtBack)] = gNode[10];
             storeGhostXEast[g_idxPopX(typ1, tz, 3, storeBxWest, byAtNorth, bz)] = gNode[14];
             storeGhostXEast[g_idxPopX(ty, tzp1, 4, storeBxWest, by, bzAtFront)] = gNode[16];
             storeGhostXEast[g_idxPopX(tym1, tzm1, 5, storeBxWest, byAtSouth, bzAtBack)] = gNode[19];
             storeGhostXEast[g_idxPopX(tym1, tzp1, 6, storeBxWest, byAtSouth, bzAtFront)] = gNode[21];
             storeGhostXEast[g_idxPopX(typ1, tzm1, 7, storeBxWest, byAtNorth, bzAtBack)] = gNode[23];
             storeGhostXEast[g_idxPopX(typ1, tzp1, 8, storeBxWest, byAtNorth, bzAtFront)] = gNode[26];
#endif
        }
        if (INTERFACE_BC_EAST){//e
             storeGhostXWest[g_idxPopX(ty, tz, 0, storeBxEast, by, bz)] = gNode[ 1]; 
#ifdef D3G19
             storeGhostXWest[g_idxPopX(typ1, tz, 1, storeBxEast, byAtNorth, bz)] = gNode[ 7];
             storeGhostXWest[g_idxPopX(ty, tzp1, 2, storeBxEast, by, bzAtFront)] = gNode[ 9];
             storeGhostXWest[g_idxPopX(tym1, tz, 3, storeBxEast, byAtSouth, bz)] = gNode[13];
             storeGhostXWest[g_idxPopX(ty, tzm1, 4, storeBxEast, by, bzAtBack)] = gNode[15];
#endif
#ifdef D3G27
             storeGhostXWest[g_idxPopX(typ1, tz, 1, storeBxEast, byAtNorth, bz)] = gNode[ 7];
             storeGhostXWest[g_idxPopX(ty, tzp1, 2, storeBxEast, by, bzAtFront)] = gNode[ 9];
             storeGhostXWest[g_idxPopX(tym1, tz, 3, storeBxEast, byAtSouth, bz)] = gNode[13];
             storeGhostXWest[g_idxPopX(ty, tzm1, 4, storeBxEast, by, bzAtBack)] = gNode[15];
             storeGhostXWest[g_idxPopX(typ1, tzp1, 5, storeBxEast, byAtNorth, bzAtFront)] = gNode[20];
             storeGhostXWest[g_idxPopX(typ1, tzm1, 6, storeBxEast, byAtNorth, bzAtBack)] = gNode[22];
             storeGhostXWest[g_idxPopX(tym1, tzp1, 7, storeBxEast, byAtSouth, bzAtFront)] = gNode[24];
             storeGhostXWest[g_idxPopX(tym1, tzm1, 8, storeBxEast, byAtSouth, bzAtBack)] = gNode[25];
#endif
        }

        if (INTERFACE_BC_SOUTH){//s
             storeGhostYNorth[g_idxPopY(tx, tz, 0, bx, storeBySouth, bz)] = gNode[ 4]; 
#ifdef D3G19
             storeGhostYNorth[g_idxPopY(txm1, tz, 1, bxAtWest, storeBySouth, bz)] = gNode[ 8];
             storeGhostYNorth[g_idxPopY(tx, tzm1, 2, bx, storeBySouth, bzAtBack)] = gNode[12];
             storeGhostYNorth[g_idxPopY(txp1, tz, 3, bxAtEast, storeBySouth, bz)] = gNode[13];
             storeGhostYNorth[g_idxPopY(tx, tzp1, 4, bx, storeBySouth, bzAtFront)] = gNode[18];
#endif
#ifdef D3G27
             storeGhostYNorth[g_idxPopY(txm1, tz, 1, bxAtWest, storeBySouth, bz)] = gNode[ 8];
             storeGhostYNorth[g_idxPopY(tx, tzm1, 2, bx, storeBySouth, bzAtBack)] = gNode[12];
             storeGhostYNorth[g_idxPopY(txp1, tz, 3, bxAtEast, storeBySouth, bz)] = gNode[13];
             storeGhostYNorth[g_idxPopY(tx, tzp1, 4, bx, storeBySouth, bzAtFront)] = gNode[18];
             storeGhostYNorth[g_idxPopY(txm1, tzm1, 5, bxAtWest, storeBySouth, bzAtBack)] = gNode[19];
             storeGhostYNorth[g_idxPopY(txm1, tzp1, 6, bxAtWest, storeBySouth, bzAtFront)] = gNode[21];
             storeGhostYNorth[g_idxPopY(txp1, tzp1, 7, bxAtEast, storeBySouth, bzAtFront)] = gNode[24];
             storeGhostYNorth[g_idxPopY(txp1, tzm1, 8, bxAtEast, storeBySouth, bzAtBack)] = gNode[25];
#endif
        }
        if (INTERFACE_BC_NORTH){//n
             storeGhostYSouth[g_idxPopY(tx, tz, 0, bx, storeByNorth, bz)] = gNode[ 3]; 
#ifdef D3G19
             storeGhostYSouth[g_idxPopY(txp1, tz, 1, bxAtEast, storeByNorth, bz)] = gNode[ 7];
             storeGhostYSouth[g_idxPopY(tx, tzp1, 2, bx, storeByNorth, bzAtFront)] = gNode[11];
             storeGhostYSouth[g_idxPopY(txm1, tz, 3, bxAtWest, storeByNorth, bz)] = gNode[14];
             storeGhostYSouth[g_idxPopY(tx, tzm1, 4, bx, storeByNorth, bzAtBack)] = gNode[17];
#endif
#ifdef D3G27
             storeGhostYSouth[g_idxPopY(txp1, tz, 1, bxAtEast, storeByNorth, bz)] = gNode[ 7];
             storeGhostYSouth[g_idxPopY(tx, tzp1, 2, bx, storeByNorth, bzAtFront)] = gNode[11];
             storeGhostYSouth[g_idxPopY(txm1, tz, 3, bxAtWest, storeByNorth, bz)] = gNode[14];
             storeGhostYSouth[g_idxPopY(tx, tzm1, 4, bx, storeByNorth, bzAtBack)] = gNode[17];
             storeGhostYSouth[g_idxPopY(txp1, tzp1, 5, bxAtEast, storeByNorth, bzAtFront)] = gNode[20];
             storeGhostYSouth[g_idxPopY(txp1, tzm1, 6, bxAtEast, storeByNorth, bzAtBack)] = gNode[22];
             storeGhostYSouth[g_idxPopY(txm1, tzm1, 7, bxAtWest, storeByNorth, bzAtBack)] = gNode[23];
             storeGhostYSouth[g_idxPopY(txm1, tzp1, 8, bxAtWest, storeByNorth, bzAtFront)] = gNode[26];
#endif
        }

        dfloat* storeZBack_edge  = (bz == 0)  ? ghostInterfacePhi.Z_0 : storeGhostZFront;
        const int     storeBzBack_edge  = (bz == 0)  ? 0   : storeBzBack;
        dfloat* storeZFront_edge = (bz == top) ? ghostInterfacePhi.Z_1 : storeGhostZBack;
        const int     storeBzFront_edge = (bz == top) ? top : storeBzFront;

        if (INTERFACE_BC_BACK){//b
             storeZBack_edge[g_idxPopZ(tx, ty, 0, bx, by, storeBzBack_edge)] = gNode[ 6]; 
#ifdef D3G19
             storeZBack_edge[g_idxPopZ(txm1, ty, 1, bxAtWest, by, storeBzBack_edge)] = gNode[10];
             storeZBack_edge[g_idxPopZ(tx, tym1, 2, bx, byAtSouth, storeBzBack_edge)] = gNode[12];
             storeZBack_edge[g_idxPopZ(txp1, ty, 3, bxAtEast, by, storeBzBack_edge)] = gNode[15];
             storeZBack_edge[g_idxPopZ(tx, typ1, 4, bx, byAtNorth, storeBzBack_edge)] = gNode[17];
#endif
#ifdef D3G27
             storeZBack_edge[g_idxPopZ(txm1, ty, 1, bxAtWest, by, storeBzBack_edge)] = gNode[10];
             storeZBack_edge[g_idxPopZ(tx, tym1, 2, bx, byAtSouth, storeBzBack_edge)] = gNode[12];
             storeZBack_edge[g_idxPopZ(txp1, ty, 3, bxAtEast, by, storeBzBack_edge)] = gNode[15];
             storeZBack_edge[g_idxPopZ(tx, typ1, 4, bx, byAtNorth, storeBzBack_edge)] = gNode[17];
             storeZBack_edge[g_idxPopZ(txm1, tym1, 5, bxAtWest, byAtSouth, storeBzBack_edge)] = gNode[19];
             storeZBack_edge[g_idxPopZ(txp1, typ1, 6, bxAtEast, byAtNorth, storeBzBack_edge)] = gNode[22];
             storeZBack_edge[g_idxPopZ(txm1, typ1, 7, bxAtWest, byAtNorth, storeBzBack_edge)] = gNode[23];
             storeZBack_edge[g_idxPopZ(txp1, tym1, 8, bxAtEast, byAtSouth, storeBzBack_edge)] = gNode[25];
#endif
        }
        if (INTERFACE_BC_FRONT){//f
             storeZFront_edge[g_idxPopZ(tx, ty, 0, bx, by, storeBzFront_edge)] = gNode[ 5]; 
#ifdef D3G19
             storeZFront_edge[g_idxPopZ(txp1, ty, 1, bxAtEast, by, storeBzFront_edge)] = gNode[ 9];
             storeZFront_edge[g_idxPopZ(tx, typ1, 2, bx, byAtNorth, storeBzFront_edge)] = gNode[11];
             storeZFront_edge[g_idxPopZ(txm1, ty, 3, bxAtWest, by, storeBzFront_edge)] = gNode[16];
             storeZFront_edge[g_idxPopZ(tx, tym1, 4, bx, byAtSouth, storeBzFront_edge)] = gNode[18];
#endif
#ifdef D3G27
             storeZFront_edge[g_idxPopZ(txp1, ty, 1, bxAtEast, by, storeBzFront_edge)] = gNode[ 9];
             storeZFront_edge[g_idxPopZ(tx, typ1, 2, bx, byAtNorth, storeBzFront_edge)] = gNode[11];
             storeZFront_edge[g_idxPopZ(txm1, ty, 3, bxAtWest, by, storeBzFront_edge)] = gNode[16];
             storeZFront_edge[g_idxPopZ(tx, tym1, 4, bx, byAtSouth, storeBzFront_edge)] = gNode[18];
             storeZFront_edge[g_idxPopZ(txp1, typ1, 5, bxAtEast, byAtNorth, storeBzFront_edge)] = gNode[20];
             storeZFront_edge[g_idxPopZ(txm1, tym1, 6, bxAtWest, byAtSouth, storeBzFront_edge)] = gNode[21];
             storeZFront_edge[g_idxPopZ(txp1, tym1, 7, bxAtEast, byAtSouth, storeBzFront_edge)] = gNode[24];
             storeZFront_edge[g_idxPopZ(txm1, typ1, 8, bxAtWest, byAtNorth, storeBzFront_edge)] = gNode[26];
#endif
        }
    }
/* DEV_REF_START (from dev during merge)
    
    if (INTERFACE_BC_BACK){ //b
        ghostInterface.phi_gGhost.Z_0[g_idxPopZ(tx, ty, 0, bx, by, bz)] = gNode[ 6];
#ifdef D3G19
        ghostInterface.phi_gGhost.Z_0[g_idxPopZ(tx, ty, 1, bx, by, bz)] = gNode[10];
        ghostInterface.phi_gGhost.Z_0[g_idxPopZ(tx, ty, 2, bx, by, bz)] = gNode[12];
        ghostInterface.phi_gGhost.Z_0[g_idxPopZ(tx, ty, 3, bx, by, bz)] = gNode[15];
        ghostInterface.phi_gGhost.Z_0[g_idxPopZ(tx, ty, 4, bx, by, bz)] = gNode[17];
#endif
#ifdef D3G27
        ghostInterface.phi_gGhost.Z_0[g_idxPopZ(tx, ty, 1, bx, by, bz)] = gNode[10];
        ghostInterface.phi_gGhost.Z_0[g_idxPopZ(tx, ty, 2, bx, by, bz)] = gNode[12];
        ghostInterface.phi_gGhost.Z_0[g_idxPopZ(tx, ty, 3, bx, by, bz)] = gNode[15];
        ghostInterface.phi_gGhost.Z_0[g_idxPopZ(tx, ty, 4, bx, by, bz)] = gNode[17];
        ghostInterface.phi_gGhost.Z_0[g_idxPopZ(tx, ty, 5, bx, by, bz)] = gNode[20];
        ghostInterface.phi_gGhost.Z_0[g_idxPopZ(tx, ty, 6, bx, by, bz)] = gNode[21];
        ghostInterface.phi_gGhost.Z_0[g_idxPopZ(tx, ty, 7, bx, by, bz)] = gNode[24];
        ghostInterface.phi_gGhost.Z_0[g_idxPopZ(tx, ty, 8, bx, by, bz)] = gNode[26];
#endif
    }if (INTERFACE_BC_FRONT){
        ghostInterface.phi_gGhost.Z_1[g_idxPopZ(tx, ty, 0, bx, by, bz)] = gNode[ 5];
#ifdef D3G19
        ghostInterface.phi_gGhost.Z_1[g_idxPopZ(tx, ty, 1, bx, by, bz)] = gNode[ 9];
        ghostInterface.phi_gGhost.Z_1[g_idxPopZ(tx, ty, 2, bx, by, bz)] = gNode[11];
        ghostInterface.phi_gGhost.Z_1[g_idxPopZ(tx, ty, 3, bx, by, bz)] = gNode[16];
        ghostInterface.phi_gGhost.Z_1[g_idxPopZ(tx, ty, 4, bx, by, bz)] = gNode[18];
#endif
#ifdef D3G27
        ghostInterface.phi_gGhost.Z_1[g_idxPopZ(tx, ty, 1, bx, by, bz)] = gNode[ 9];
        ghostInterface.phi_gGhost.Z_1[g_idxPopZ(tx, ty, 2, bx, by, bz)] = gNode[11];
        ghostInterface.phi_gGhost.Z_1[g_idxPopZ(tx, ty, 3, bx, by, bz)] = gNode[16];
        ghostInterface.phi_gGhost.Z_1[g_idxPopZ(tx, ty, 4, bx, by, bz)] = gNode[18];
        ghostInterface.phi_gGhost.Z_1[g_idxPopZ(tx, ty, 5, bx, by, bz)] = gNode[19];
        ghostInterface.phi_gGhost.Z_1[g_idxPopZ(tx, ty, 6, bx, by, bz)] = gNode[22];
        ghostInterface.phi_gGhost.Z_1[g_idxPopZ(tx, ty, 7, bx, by, bz)] = gNode[23];
        ghostInterface.phi_gGhost.Z_1[g_idxPopZ(tx, ty, 8, bx, by, bz)] = gNode[25];
#endif
    }

DEV_REF_END */
