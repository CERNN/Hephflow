    /* write to global pop - AA scheme */
    dfloat* storeGhostXWest  = ghostInterface.Azz.X_0;
    dfloat* storeGhostXEast  = ghostInterface.Azz.X_1;
    dfloat* storeGhostYSouth = ghostInterface.Azz.Y_0;
    dfloat* storeGhostYNorth = ghostInterface.Azz.Y_1;
    dfloat* storeGhostZBack  = ghostInterface.Azz.Z_0;
    dfloat* storeGhostZFront = ghostInterface.Azz.Z_1;

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
        }
        if (INTERFACE_BC_EAST){//e
             storeGhostXEast[g_idxPopX(ty, tz, 0, storeBxEast, by, bz)] = gNode[ 1]; 
#ifdef D3G19
             storeGhostXEast[g_idxPopX(ty, tz, 1, storeBxEast, by, bz)] = gNode[ 7];
             storeGhostXEast[g_idxPopX(ty, tz, 2, storeBxEast, by, bz)] = gNode[ 9];
             storeGhostXEast[g_idxPopX(ty, tz, 3, storeBxEast, by, bz)] = gNode[13];
             storeGhostXEast[g_idxPopX(ty, tz, 4, storeBxEast, by, bz)] = gNode[15];
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
        }
        if (INTERFACE_BC_NORTH){//n
             storeGhostYNorth[g_idxPopY(tx, tz, 0, bx, storeByNorth, bz)] = gNode[ 3]; 
#ifdef D3G19
             storeGhostYNorth[g_idxPopY(tx, tz, 1, bx, storeByNorth, bz)] = gNode[ 7];
             storeGhostYNorth[g_idxPopY(tx, tz, 2, bx, storeByNorth, bz)] = gNode[11];
             storeGhostYNorth[g_idxPopY(tx, tz, 3, bx, storeByNorth, bz)] = gNode[14];
             storeGhostYNorth[g_idxPopY(tx, tz, 4, bx, storeByNorth, bz)] = gNode[17];
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
        }
        if (INTERFACE_BC_FRONT){//f
             storeGhostZFront[g_idxPopZ(tx, ty, 0, bx, by, storeBzFront)] = gNode[ 5]; 
#ifdef D3G19
             storeGhostZFront[g_idxPopZ(tx, ty, 1, bx, by, storeBzFront)] = gNode[ 9];
             storeGhostZFront[g_idxPopZ(tx, ty, 2, bx, by, storeBzFront)] = gNode[11];
             storeGhostZFront[g_idxPopZ(tx, ty, 3, bx, by, storeBzFront)] = gNode[16];
             storeGhostZFront[g_idxPopZ(tx, ty, 4, bx, by, storeBzFront)] = gNode[18];
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
        }
        if (INTERFACE_BC_EAST){//e
             storeGhostXWest[g_idxPopX(ty, tz, 0, storeBxEast, by, bz)] = gNode[ 1]; 
#ifdef D3G19
             storeGhostXWest[g_idxPopX(typ1, tz, 1, storeBxEast, byAtNorth, bz)] = gNode[ 7];
             storeGhostXWest[g_idxPopX(ty, tzp1, 2, storeBxEast, by, bzAtFront)] = gNode[ 9];
             storeGhostXWest[g_idxPopX(tym1, tz, 3, storeBxEast, byAtSouth, bz)] = gNode[13];
             storeGhostXWest[g_idxPopX(ty, tzm1, 4, storeBxEast, by, bzAtBack)] = gNode[15];
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
        }
        if (INTERFACE_BC_NORTH){//n
             storeGhostYSouth[g_idxPopY(tx, tz, 0, bx, storeByNorth, bz)] = gNode[ 3]; 
#ifdef D3G19
             storeGhostYSouth[g_idxPopY(txp1, tz, 1, bxAtEast, storeByNorth, bz)] = gNode[ 7];
             storeGhostYSouth[g_idxPopY(tx, tzp1, 2, bx, storeByNorth, bzAtFront)] = gNode[11];
             storeGhostYSouth[g_idxPopY(txm1, tz, 3, bxAtWest, storeByNorth, bz)] = gNode[14];
             storeGhostYSouth[g_idxPopY(tx, tzm1, 4, bx, storeByNorth, bzAtBack)] = gNode[17];
#endif
        }

        if (INTERFACE_BC_BACK){//b
             storeGhostZFront[g_idxPopZ(tx, ty, 0, bx, by, storeBzBack)] = gNode[ 6]; 
#ifdef D3G19
             storeGhostZFront[g_idxPopZ(txm1, ty, 1, bxAtWest, by, storeBzBack)] = gNode[10];
             storeGhostZFront[g_idxPopZ(tx, tym1, 2, bx, byAtSouth, storeBzBack)] = gNode[12];
             storeGhostZFront[g_idxPopZ(txp1, ty, 3, bxAtEast, by, storeBzBack)] = gNode[15];
             storeGhostZFront[g_idxPopZ(tx, typ1, 4, bx, byAtNorth, storeBzBack)] = gNode[17];
#endif
        }
        if (INTERFACE_BC_FRONT){//f
             storeGhostZBack[g_idxPopZ(tx, ty, 0, bx, by, storeBzFront)] = gNode[ 5]; 
#ifdef D3G19
             storeGhostZBack[g_idxPopZ(txp1, ty, 1, bxAtEast, by, storeBzFront)] = gNode[ 9];
             storeGhostZBack[g_idxPopZ(tx, typ1, 2, bx, byAtNorth, storeBzFront)] = gNode[11];
             storeGhostZBack[g_idxPopZ(txm1, ty, 3, bxAtWest, by, storeBzFront)] = gNode[16];
             storeGhostZBack[g_idxPopZ(tx, tym1, 4, bx, byAtSouth, storeBzFront)] = gNode[18];
#endif
        }
    }
