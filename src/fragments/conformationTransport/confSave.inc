    /* write to global conf - AA scheme */
    dfloat* storeGhostXWest  = ghostInterface.conf.X_0;
    dfloat* storeGhostXEast  = ghostInterface.conf.X_1;
    dfloat* storeGhostYSouth = ghostInterface.conf.Y_0;
    dfloat* storeGhostYNorth = ghostInterface.conf.Y_1;
    dfloat* storeGhostZBack  = ghostInterface.conf.Z_0;
    dfloat* storeGhostZFront = ghostInterface.conf.Z_1;

    const int storeBxWest  = (stepParity ? bx : bxm1);
    const int storeBxEast  = (stepParity ? bx : bxp1);
    const int storeBySouth = (stepParity ? by : bym1);
    const int storeByNorth = (stepParity ? by : byp1);
    const int storeBzBack  = (stepParity ? bz : bzm1);
    const int storeBzFront = (stepParity ? bz : bzp1);

    if(stepParity){
        if (INTERFACE_BC_WEST){
            storeGhostXWest[g_idxConfX(ty, tz, 0, storeBxWest, by, bz)] = AxxVar;
            storeGhostXWest[g_idxConfX(ty, tz, 1, storeBxWest, by, bz)] = AxyVar;
            storeGhostXWest[g_idxConfX(ty, tz, 2, storeBxWest, by, bz)] = AxzVar;
            storeGhostXWest[g_idxConfX(ty, tz, 3, storeBxWest, by, bz)] = AyyVar;
            storeGhostXWest[g_idxConfX(ty, tz, 4, storeBxWest, by, bz)] = AyzVar;
            storeGhostXWest[g_idxConfX(ty, tz, 5, storeBxWest, by, bz)] = AzzVar;
        }
        if (INTERFACE_BC_EAST){
            storeGhostXEast[g_idxConfX(ty, tz, 0, storeBxEast, by, bz)] = AxxVar;
            storeGhostXEast[g_idxConfX(ty, tz, 1, storeBxEast, by, bz)] = AxyVar;
            storeGhostXEast[g_idxConfX(ty, tz, 2, storeBxEast, by, bz)] = AxzVar;
            storeGhostXEast[g_idxConfX(ty, tz, 3, storeBxEast, by, bz)] = AyyVar;
            storeGhostXEast[g_idxConfX(ty, tz, 4, storeBxEast, by, bz)] = AyzVar;
            storeGhostXEast[g_idxConfX(ty, tz, 5, storeBxEast, by, bz)] = AzzVar;
        }
        if (INTERFACE_BC_SOUTH){
            storeGhostYSouth[g_idxConfY(tx, tz, 0, bx, storeBySouth, bz)] = AxxVar;
            storeGhostYSouth[g_idxConfY(tx, tz, 1, bx, storeBySouth, bz)] = AxyVar;
            storeGhostYSouth[g_idxConfY(tx, tz, 2, bx, storeBySouth, bz)] = AxzVar;
            storeGhostYSouth[g_idxConfY(tx, tz, 3, bx, storeBySouth, bz)] = AyyVar;
            storeGhostYSouth[g_idxConfY(tx, tz, 4, bx, storeBySouth, bz)] = AyzVar;
            storeGhostYSouth[g_idxConfY(tx, tz, 5, bx, storeBySouth, bz)] = AzzVar;
        }
        if (INTERFACE_BC_NORTH){
            storeGhostYNorth[g_idxConfY(tx, tz, 0, bx, storeByNorth, bz)] = AxxVar;
            storeGhostYNorth[g_idxConfY(tx, tz, 1, bx, storeByNorth, bz)] = AxyVar;
            storeGhostYNorth[g_idxConfY(tx, tz, 2, bx, storeByNorth, bz)] = AxzVar;
            storeGhostYNorth[g_idxConfY(tx, tz, 3, bx, storeByNorth, bz)] = AyyVar;
            storeGhostYNorth[g_idxConfY(tx, tz, 4, bx, storeByNorth, bz)] = AyzVar;
            storeGhostYNorth[g_idxConfY(tx, tz, 5, bx, storeByNorth, bz)] = AzzVar;
        }
        if (INTERFACE_BC_BACK){
            storeGhostZBack[g_idxConfZ(tx, ty, 0, bx, by, storeBzBack)] = AxxVar;
            storeGhostZBack[g_idxConfZ(tx, ty, 1, bx, by, storeBzBack)] = AxyVar;
            storeGhostZBack[g_idxConfZ(tx, ty, 2, bx, by, storeBzBack)] = AxzVar;
            storeGhostZBack[g_idxConfZ(tx, ty, 3, bx, by, storeBzBack)] = AyyVar;
            storeGhostZBack[g_idxConfZ(tx, ty, 4, bx, by, storeBzBack)] = AyzVar;
            storeGhostZBack[g_idxConfZ(tx, ty, 5, bx, by, storeBzBack)] = AzzVar;
        }
        if (INTERFACE_BC_FRONT){
            storeGhostZFront[g_idxConfZ(tx, ty, 0, bx, by, storeBzFront)] = AxxVar;
            storeGhostZFront[g_idxConfZ(tx, ty, 1, bx, by, storeBzFront)] = AxyVar;
            storeGhostZFront[g_idxConfZ(tx, ty, 2, bx, by, storeBzFront)] = AxzVar;
            storeGhostZFront[g_idxConfZ(tx, ty, 3, bx, by, storeBzFront)] = AyyVar;
            storeGhostZFront[g_idxConfZ(tx, ty, 4, bx, by, storeBzFront)] = AyzVar;
            storeGhostZFront[g_idxConfZ(tx, ty, 5, bx, by, storeBzFront)] = AzzVar;
        }
    }else{
        if (INTERFACE_BC_WEST){
            storeGhostXEast[g_idxConfX(ty, tz, 0, storeBxWest, by, bz)] = AxxVar;
            storeGhostXEast[g_idxConfX(ty, tz, 1, storeBxWest, by, bz)] = AxyVar;
            storeGhostXEast[g_idxConfX(ty, tz, 2, storeBxWest, by, bz)] = AxzVar;
            storeGhostXEast[g_idxConfX(ty, tz, 3, storeBxWest, by, bz)] = AyyVar;
            storeGhostXEast[g_idxConfX(ty, tz, 4, storeBxWest, by, bz)] = AyzVar;
            storeGhostXEast[g_idxConfX(ty, tz, 5, storeBxWest, by, bz)] = AzzVar;
        }
        if (INTERFACE_BC_EAST){
            storeGhostXWest[g_idxConfX(ty, tz, 0, storeBxEast, by, bz)] = AxxVar;
            storeGhostXWest[g_idxConfX(ty, tz, 1, storeBxEast, by, bz)] = AxyVar;
            storeGhostXWest[g_idxConfX(ty, tz, 2, storeBxEast, by, bz)] = AxzVar;
            storeGhostXWest[g_idxConfX(ty, tz, 3, storeBxEast, by, bz)] = AyyVar;
            storeGhostXWest[g_idxConfX(ty, tz, 4, storeBxEast, by, bz)] = AyzVar;
            storeGhostXWest[g_idxConfX(ty, tz, 5, storeBxEast, by, bz)] = AzzVar;
        }
        if (INTERFACE_BC_SOUTH){
            storeGhostYNorth[g_idxConfY(tx, tz, 0, bx, storeBySouth, bz)] = AxxVar;
            storeGhostYNorth[g_idxConfY(tx, tz, 1, bx, storeBySouth, bz)] = AxyVar;
            storeGhostYNorth[g_idxConfY(tx, tz, 2, bx, storeBySouth, bz)] = AxzVar;
            storeGhostYNorth[g_idxConfY(tx, tz, 3, bx, storeBySouth, bz)] = AyyVar;
            storeGhostYNorth[g_idxConfY(tx, tz, 4, bx, storeBySouth, bz)] = AyzVar;
            storeGhostYNorth[g_idxConfY(tx, tz, 5, bx, storeBySouth, bz)] = AzzVar;
        }
        if (INTERFACE_BC_NORTH){
            storeGhostYSouth[g_idxConfY(tx, tz, 0, bx, storeByNorth, bz)] = AxxVar;
            storeGhostYSouth[g_idxConfY(tx, tz, 1, bx, storeByNorth, bz)] = AxyVar;
            storeGhostYSouth[g_idxConfY(tx, tz, 2, bx, storeByNorth, bz)] = AxzVar;
            storeGhostYSouth[g_idxConfY(tx, tz, 3, bx, storeByNorth, bz)] = AyyVar;
            storeGhostYSouth[g_idxConfY(tx, tz, 4, bx, storeByNorth, bz)] = AyzVar;
            storeGhostYSouth[g_idxConfY(tx, tz, 5, bx, storeByNorth, bz)] = AzzVar;
        }
        if (INTERFACE_BC_BACK){
            storeGhostZFront[g_idxConfZ(tx, ty, 0, bx, by, storeBzBack)] = AxxVar;
            storeGhostZFront[g_idxConfZ(tx, ty, 1, bx, by, storeBzBack)] = AxyVar;
            storeGhostZFront[g_idxConfZ(tx, ty, 2, bx, by, storeBzBack)] = AxzVar;
            storeGhostZFront[g_idxConfZ(tx, ty, 3, bx, by, storeBzBack)] = AyyVar;
            storeGhostZFront[g_idxConfZ(tx, ty, 4, bx, by, storeBzBack)] = AyzVar;
            storeGhostZFront[g_idxConfZ(tx, ty, 5, bx, by, storeBzBack)] = AzzVar;
        }
        if (INTERFACE_BC_FRONT){
            storeGhostZBack[g_idxConfZ(tx, ty, 0, bx, by, storeBzFront)] = AxxVar;
            storeGhostZBack[g_idxConfZ(tx, ty, 1, bx, by, storeBzFront)] = AxyVar;
            storeGhostZBack[g_idxConfZ(tx, ty, 2, bx, by, storeBzFront)] = AxzVar;
            storeGhostZBack[g_idxConfZ(tx, ty, 3, bx, by, storeBzFront)] = AyyVar;
            storeGhostZBack[g_idxConfZ(tx, ty, 4, bx, by, storeBzFront)] = AyzVar;
            storeGhostZBack[g_idxConfZ(tx, ty, 5, bx, by, storeBzFront)] = AzzVar;
        }
    }
