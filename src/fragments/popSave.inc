dfloat* storeGhostXWest  = ghostInterface.pop.X_0; // Write own west face
dfloat* storeGhostXEast  = ghostInterface.pop.X_1; // Write own east face
dfloat* storeGhostYSouth = ghostInterface.pop.Y_0; // Write own south face
dfloat* storeGhostYNorth = ghostInterface.pop.Y_1; // Write own north face
dfloat* storeGhostZBack  = ghostInterface.pop.Z_0; // Write own back face
dfloat* storeGhostZFront = ghostInterface.pop.Z_1; // Write own front face

// Parity-aware block indices for ghost writes (AA scheme)
const int storeBxWest  =  (stepParity ? bx : bxm1);
const int storeBxEast  =  (stepParity ? bx : bxp1);
const int storeBySouth =  (stepParity ? by : bym1);
const int storeByNorth =  (stepParity ? by : byp1);
const int storeBzBack  =  (stepParity ? bz : bzm1);
const int storeBzFront =  (stepParity ? bz : bzp1);


    if(stepParity){
        if (INTERFACE_BC_WEST){//w
            storeGhostXWest[idxPopX(ty, tz, 0, storeBxWest, by, bz)] = pop[ 2]; 
            storeGhostXWest[idxPopX(ty, tz, 1, storeBxWest, by, bz)] = pop[ 8];
            storeGhostXWest[idxPopX(ty, tz, 2, storeBxWest, by, bz)] = pop[10];
            storeGhostXWest[idxPopX(ty, tz, 3, storeBxWest, by, bz)] = pop[14];
            storeGhostXWest[idxPopX(ty, tz, 4, storeBxWest, by, bz)] = pop[16];
            #ifdef D3Q27
            storeGhostXWest[idxPopX(ty, tz, 5, storeBxWest, by, bz)] = pop[20];
            storeGhostXWest[idxPopX(ty, tz, 6, storeBxWest, by, bz)] = pop[22];
            storeGhostXWest[idxPopX(ty, tz, 7, storeBxWest, by, bz)] = pop[24];
            storeGhostXWest[idxPopX(ty, tz, 8, storeBxWest, by, bz)] = pop[25];
            #endif //D3Q27
        }if (INTERFACE_BC_EAST){//e
            storeGhostXEast[idxPopX(ty, tz, 0, storeBxEast, by, bz)] = pop[ 1];
            storeGhostXEast[idxPopX(ty, tz, 1, storeBxEast, by, bz)] = pop[ 7];
            storeGhostXEast[idxPopX(ty, tz, 2, storeBxEast, by, bz)] = pop[ 9];
            storeGhostXEast[idxPopX(ty, tz, 3, storeBxEast, by, bz)] = pop[13];
            storeGhostXEast[idxPopX(ty, tz, 4, storeBxEast, by, bz)] = pop[15];
            #ifdef D3Q27
            storeGhostXEast[idxPopX(ty, tz, 5, storeBxEast, by, bz)] = pop[19];
            storeGhostXEast[idxPopX(ty, tz, 6, storeBxEast, by, bz)] = pop[21];
            storeGhostXEast[idxPopX(ty, tz, 7, storeBxEast, by, bz)] = pop[23];
            storeGhostXEast[idxPopX(ty, tz, 8, storeBxEast, by, bz)] = pop[26];
            #endif //D3Q27       
        }

        if (INTERFACE_BC_SOUTH){//s          
            storeGhostYSouth[idxPopY(tx, tz, 0, bx, storeBySouth, bz)] = pop[ 4];
            storeGhostYSouth[idxPopY(tx, tz, 1, bx, storeBySouth, bz)] = pop[ 8];
            storeGhostYSouth[idxPopY(tx, tz, 2, bx, storeBySouth, bz)] = pop[12];
            storeGhostYSouth[idxPopY(tx, tz, 3, bx, storeBySouth, bz)] = pop[13];
            storeGhostYSouth[idxPopY(tx, tz, 4, bx, storeBySouth, bz)] = pop[18];
            #ifdef D3Q27
            storeGhostYSouth[idxPopY(tx, tz, 5, bx, storeBySouth, bz)] = pop[20];
            storeGhostYSouth[idxPopY(tx, tz, 6, bx, storeBySouth, bz)] = pop[22];
            storeGhostYSouth[idxPopY(tx, tz, 7, bx, storeBySouth, bz)] = pop[23];
            storeGhostYSouth[idxPopY(tx, tz, 8, bx, storeBySouth, bz)] = pop[26];
            #endif //D3Q27
        }if (INTERFACE_BC_NORTH){//n
            storeGhostYNorth[idxPopY(tx, tz, 0, bx, storeByNorth, bz)] = pop[ 3];
            storeGhostYNorth[idxPopY(tx, tz, 1, bx, storeByNorth, bz)] = pop[ 7];
            storeGhostYNorth[idxPopY(tx, tz, 2, bx, storeByNorth, bz)] = pop[11];
            storeGhostYNorth[idxPopY(tx, tz, 3, bx, storeByNorth, bz)] = pop[14];
            storeGhostYNorth[idxPopY(tx, tz, 4, bx, storeByNorth, bz)] = pop[17];
            #ifdef D3Q27
            storeGhostYNorth[idxPopY(tx, tz, 5, bx, storeByNorth, bz)] = pop[19];
            storeGhostYNorth[idxPopY(tx, tz, 6, bx, storeByNorth, bz)] = pop[21];
            storeGhostYNorth[idxPopY(tx, tz, 7, bx, storeByNorth, bz)] = pop[24];
            storeGhostYNorth[idxPopY(tx, tz, 8, bx, storeByNorth, bz)] = pop[25];
            #endif //D3Q27
        }
        
        if (INTERFACE_BC_BACK){ //b
            storeGhostZBack[idxPopZ(tx, ty, 0, bx, by, storeBzBack)] = pop[ 6];
            storeGhostZBack[idxPopZ(tx, ty, 1, bx, by, storeBzBack)] = pop[10];
            storeGhostZBack[idxPopZ(tx, ty, 2, bx, by, storeBzBack)] = pop[12];
            storeGhostZBack[idxPopZ(tx, ty, 3, bx, by, storeBzBack)] = pop[15];
            storeGhostZBack[idxPopZ(tx, ty, 4, bx, by, storeBzBack)] = pop[17];
            #ifdef D3Q27
            storeGhostZBack[idxPopZ(tx, ty, 5, bx, by, storeBzBack)] = pop[20];
            storeGhostZBack[idxPopZ(tx, ty, 6, bx, by, storeBzBack)] = pop[21];
            storeGhostZBack[idxPopZ(tx, ty, 7, bx, by, storeBzBack)] = pop[24];
            storeGhostZBack[idxPopZ(tx, ty, 8, bx, by, storeBzBack)] = pop[26];
            #endif //D3Q27
        }if (INTERFACE_BC_FRONT){
            storeGhostZFront[idxPopZ(tx, ty, 0, bx, by, storeBzFront)] = pop[ 5];
            storeGhostZFront[idxPopZ(tx, ty, 1, bx, by, storeBzFront)] = pop[ 9];
            storeGhostZFront[idxPopZ(tx, ty, 2, bx, by, storeBzFront)] = pop[11];
            storeGhostZFront[idxPopZ(tx, ty, 3, bx, by, storeBzFront)] = pop[16];
            storeGhostZFront[idxPopZ(tx, ty, 4, bx, by, storeBzFront)] = pop[18];
            #ifdef D3Q27
            storeGhostZFront[idxPopZ(tx, ty, 5, bx, by, storeBzFront)] = pop[19];
            storeGhostZFront[idxPopZ(tx, ty, 6, bx, by, storeBzFront)] = pop[22];
            storeGhostZFront[idxPopZ(tx, ty, 7, bx, by, storeBzFront)] = pop[23];
            storeGhostZFront[idxPopZ(tx, ty, 8, bx, by, storeBzFront)] = pop[25];
            #endif //D3Q27
        }
    
    }else{
        if (INTERFACE_BC_WEST){//w
            storeGhostXEast[idxPopX(ty, tz, 0, storeBxWest, by, bz)] = pop[ 2]; 
            storeGhostXEast[idxPopX(tym1, tz, 1, storeBxWest, byAtSouth, bz)] = pop[ 8];
            storeGhostXEast[idxPopX(ty, tzm1, 2, storeBxWest, by, bzAtBack)] = pop[10];
            storeGhostXEast[idxPopX(typ1, tz, 3, storeBxWest, byAtNorth, bz)] = pop[14];
            storeGhostXEast[idxPopX(ty, tzp1, 4, storeBxWest, by, bzAtFront)] = pop[16];
            #ifdef D3Q27
            storeGhostXEast[idxPopX(tym1, tzm1, 5, storeBxWest, byAtSouth, bzAtBack)] = pop[20];
            storeGhostXEast[idxPopX(tym1, tzp1, 6, storeBxWest, byAtSouth, bzAtFront)] = pop[22];
            storeGhostXEast[idxPopX(typ1, tzm1, 7, storeBxWest, byAtNorth, bzAtBack)] = pop[24];
            storeGhostXEast[idxPopX(typ1, tzp1, 8, storeBxWest, byAtNorth, bzAtFront)] = pop[25];
            #endif //D3Q27
        }if (INTERFACE_BC_EAST){//e
            storeGhostXWest[idxPopX(ty, tz, 0, storeBxEast, by, bz)] = pop[ 1];
            storeGhostXWest[idxPopX(typ1, tz, 1, storeBxEast, byAtNorth, bz)] = pop[ 7];
            storeGhostXWest[idxPopX(ty, tzp1, 2, storeBxEast, by, bzAtFront)] = pop[ 9];
            storeGhostXWest[idxPopX(tym1, tz, 3, storeBxEast, byAtSouth, bz)] = pop[13];
            storeGhostXWest[idxPopX(ty, tzm1, 4, storeBxEast, by, bzAtBack)] = pop[15];
            #ifdef D3Q27
            storeGhostXWest[idxPopX(typ1, tzp1, 5, storeBxEast, byAtNorth, bzAtFront)] = pop[19];
            storeGhostXWest[idxPopX(typ1, tzm1, 6, storeBxEast, byAtNorth, bzAtBack)] = pop[21];
            storeGhostXWest[idxPopX(tym1, tzp1, 7, storeBxEast, byAtSouth, bzAtFront)] = pop[23];
            storeGhostXWest[idxPopX(tym1, tzm1, 8, storeBxEast, byAtSouth, bzAtBack)] = pop[26];
            #endif //D3Q27       
        }

        if (INTERFACE_BC_SOUTH){//s          
            storeGhostYNorth[idxPopY(tx, tz, 0, bx, storeBySouth, bz)] = pop[ 4];
            storeGhostYNorth[idxPopY(txm1, tz, 1, bxAtWest, storeBySouth, bz)] = pop[ 8];
            storeGhostYNorth[idxPopY(tx, tzm1, 2, bx, storeBySouth, bzAtBack)] = pop[12];
            storeGhostYNorth[idxPopY(txp1, tz, 3, bxAtEast, storeBySouth, bz)] = pop[13];
            storeGhostYNorth[idxPopY(tx, tzp1, 4, bx, storeBySouth, bzAtFront)] = pop[18];
            #ifdef D3Q27
            storeGhostYNorth[idxPopY(txm1, tzm1, 5, bxAtWest, storeBySouth, bzAtBack)] = pop[20];
            storeGhostYNorth[idxPopY(txm1, tzp1, 6, bxAtWest, storeBySouth, bzAtFront)] = pop[22];
            storeGhostYNorth[idxPopY(txp1, tzp1, 7, bxAtEast, storeBySouth, bzAtFront)] = pop[23];
            storeGhostYNorth[idxPopY(txp1, tzm1, 8, bxAtEast, storeBySouth, bzAtBack)] = pop[26];
            #endif //D3Q27
        }if (INTERFACE_BC_NORTH){//n
            storeGhostYSouth[idxPopY(tx, tz, 0, bx, storeByNorth, bz)] = pop[ 3];
            storeGhostYSouth[idxPopY(txp1, tz, 1, bxAtEast, storeByNorth, bz)] = pop[ 7];
            storeGhostYSouth[idxPopY(tx, tzp1, 2, bx, storeByNorth, bzAtFront)] = pop[11];
            storeGhostYSouth[idxPopY(txm1, tz, 3, bxAtWest, storeByNorth, bz)] = pop[14];
            storeGhostYSouth[idxPopY(tx, tzm1, 4, bx, storeByNorth, bzAtBack)] = pop[17];
            #ifdef D3Q27
            storeGhostYSouth[idxPopY(txp1, tzp1, 5, bxAtEast, storeByNorth, bzAtFront)] = pop[19];
            storeGhostYSouth[idxPopY(txp1, tzm1, 6, bxAtEast, storeByNorth, bzAtBack)] = pop[21];
            storeGhostYSouth[idxPopY(txm1, tzm1, 7, bxAtWest, storeByNorth, bzAtBack)] = pop[24];
            storeGhostYSouth[idxPopY(txm1, tzp1, 8, bxAtWest, storeByNorth, bzAtFront)] = pop[25];
            #endif //D3Q27
        }
        
        if (INTERFACE_BC_BACK){ //b
            storeGhostZFront[idxPopZ(tx, ty, 0, bx, by, storeBzBack)] = pop[ 6];
            storeGhostZFront[idxPopZ(txm1, ty, 1, bxAtWest, by, storeBzBack)] = pop[10];
            storeGhostZFront[idxPopZ(tx, tym1, 2, bx, byAtSouth, storeBzBack)] = pop[12];
            storeGhostZFront[idxPopZ(txp1, ty, 3, bxAtEast, by, storeBzBack)] = pop[15];
            storeGhostZFront[idxPopZ(tx, typ1, 4, bx, byAtNorth, storeBzBack)] = pop[17];
            #ifdef D3Q27
            storeGhostZFront[idxPopZ(txm1, tym1, 5, bxAtWest, byAtSouth, storeBzBack)] = pop[20];
            storeGhostZFront[idxPopZ(txp1, typ1, 6, bxAtEast, byAtNorth, storeBzBack)] = pop[21];
            storeGhostZFront[idxPopZ(txm1, typ1, 7, bxAtWest, byAtNorth, storeBzBack)] = pop[24];
            storeGhostZFront[idxPopZ(txp1, tym1, 8, bxAtEast, byAtSouth, storeBzBack)] = pop[26];
            #endif //D3Q27
        }if (INTERFACE_BC_FRONT){
            storeGhostZBack[idxPopZ(tx, ty, 0, bx, by, storeBzFront)] = pop[ 5];
            storeGhostZBack[idxPopZ(txp1, ty, 1, bxAtEast, by, storeBzFront)] = pop[ 9];
            storeGhostZBack[idxPopZ(tx, typ1, 2, bx, byAtNorth, storeBzFront)] = pop[11];
            storeGhostZBack[idxPopZ(txm1, ty, 3, bxAtWest, by, storeBzFront)] = pop[16];
            storeGhostZBack[idxPopZ(tx, tym1, 4, bx, byAtSouth, storeBzFront)] = pop[18];
            #ifdef D3Q27
            storeGhostZBack[idxPopZ(txp1, typ1, 5, bxAtEast, byAtNorth, storeBzFront)] = pop[19];
            storeGhostZBack[idxPopZ(txm1, tym1, 6, bxAtWest, byAtSouth, storeBzFront)] = pop[22];
            storeGhostZBack[idxPopZ(txp1, tym1, 7, bxAtEast, byAtSouth, storeBzFront)] = pop[23];
            storeGhostZBack[idxPopZ(txm1, typ1, 8, bxAtWest, byAtNorth, storeBzFront)] = pop[25];
            #endif //D3Q27
        }
    }
