// AA scheme: face pointers fixed, parity selects block index
const dfloat* loadGhostXWest  = ghostInterface.pop.X_1; // Even: neighbor -x; Odd: own slot (swapped by non-shared path)
const dfloat* loadGhostXEast  = ghostInterface.pop.X_0; // Even: neighbor +x; Odd: own slot (swapped by non-shared path)
const dfloat* loadGhostYSouth = ghostInterface.pop.Y_1;
const dfloat* loadGhostYNorth = ghostInterface.pop.Y_0;
const dfloat* loadGhostZBack  = ghostInterface.pop.Z_1;
const dfloat* loadGhostZFront = ghostInterface.pop.Z_0;

// Parity-aware block indices for ghost reads (AA scheme)
const int loadBxWest  = (stepParity ? bx : bxm1);
const int loadBxEast  = (stepParity ? bx : bxp1);
const int loadBySouth = (stepParity ? by : bym1);
const int loadByNorth = (stepParity ? by : byp1);
const int loadBzBack  = (stepParity ? bz : bzm1);
const int loadBzFront = (stepParity ? bz : bzp1);

// Diagonal block indices — when a surface thread also sits at another block edge,
// these give the correct neighbor block index in that transverse direction.
// Single point of change for all diagonal ghost accesses.
const int bxAtWest  = (tx == 0)              ? bxm1 : bx;
const int bxAtEast  = (tx == (BLOCK_NX - 1)) ? bxp1 : bx;
const int byAtSouth = (ty == 0)              ? bym1 : by;
const int byAtNorth = (ty == (BLOCK_NY - 1)) ? byp1 : by;
const int bzAtBack  = (tz == 0)              ? bzm1 : bz;
const int bzAtFront = (tz == (BLOCK_NZ - 1)) ? bzp1 : bz;

    if (stepParity){
        if (tx == 0)
        { // w
            pop[ 1] = loadGhostXEast[idxPopX(ty, tz, 0, loadBxWest, by, bz)];
            pop[ 7] = loadGhostXEast[idxPopX(ty, tz, 1, loadBxWest, by, bz)];
            pop[ 9] = loadGhostXEast[idxPopX(ty, tz, 2, loadBxWest, by, bz)];
            pop[13] = loadGhostXEast[idxPopX(ty, tz, 3, loadBxWest, by, bz)];
            pop[15] = loadGhostXEast[idxPopX(ty, tz, 4, loadBxWest, by, bz)];
            #ifdef D3Q27
            pop[19] = loadGhostXEast[idxPopX(ty, tz, 5, loadBxWest, by, bz)];
            pop[21] = loadGhostXEast[idxPopX(ty, tz, 6, loadBxWest, by, bz)];
            pop[23] = loadGhostXEast[idxPopX(ty, tz, 7, loadBxWest, by, bz)];
            pop[26] = loadGhostXEast[idxPopX(ty, tz, 8, loadBxWest, by, bz)];
            #endif // D3Q27
        }
        else if (tx == (BLOCK_NX - 1))
        { // e
            pop[ 2] = loadGhostXWest[idxPopX(ty, tz, 0, loadBxEast, by, bz)];
            pop[ 8] = loadGhostXWest[idxPopX(ty, tz, 1, loadBxEast, by, bz)];
            pop[10] = loadGhostXWest[idxPopX(ty, tz, 2, loadBxEast, by, bz)];
            pop[14] = loadGhostXWest[idxPopX(ty, tz, 3, loadBxEast, by, bz)];
            pop[16] = loadGhostXWest[idxPopX(ty, tz, 4, loadBxEast, by, bz)];
            #ifdef D3Q27
            pop[20] = loadGhostXWest[idxPopX(ty, tz, 5, loadBxEast, by, bz)];
            pop[22] = loadGhostXWest[idxPopX(ty, tz, 6, loadBxEast, by, bz)];
            pop[24] = loadGhostXWest[idxPopX(ty, tz, 7, loadBxEast, by, bz)];
            pop[25] = loadGhostXWest[idxPopX(ty, tz, 8, loadBxEast, by, bz)];
            #endif // D3Q27
        }

        if (ty == 0)
        { // s
            pop[ 3] = loadGhostYNorth[idxPopY(tx, tz, 0, bx, loadBySouth, bz)];
            pop[ 7] = loadGhostYNorth[idxPopY(tx, tz, 1, bx, loadBySouth, bz)];
            pop[11] = loadGhostYNorth[idxPopY(tx, tz, 2, bx, loadBySouth, bz)];
            pop[14] = loadGhostYNorth[idxPopY(tx, tz, 3, bx, loadBySouth, bz)];
            pop[17] = loadGhostYNorth[idxPopY(tx, tz, 4, bx, loadBySouth, bz)];
            #ifdef D3Q27
            pop[19] = loadGhostYNorth[idxPopY(tx, tz, 5, bx, loadBySouth, bz)];
            pop[21] = loadGhostYNorth[idxPopY(tx, tz, 6, bx, loadBySouth, bz)];
            pop[24] = loadGhostYNorth[idxPopY(tx, tz, 7, bx, loadBySouth, bz)];
            pop[25] = loadGhostYNorth[idxPopY(tx, tz, 8, bx, loadBySouth, bz)];
            #endif // D3Q27
        }
        else if (ty == (BLOCK_NY - 1))
        { // n
            pop[ 4] = loadGhostYSouth[idxPopY(tx, tz, 0, bx, loadByNorth, bz)];
            pop[ 8] = loadGhostYSouth[idxPopY(tx, tz, 1, bx, loadByNorth, bz)];
            pop[12] = loadGhostYSouth[idxPopY(tx, tz, 2, bx, loadByNorth, bz)];
            pop[13] = loadGhostYSouth[idxPopY(tx, tz, 3, bx, loadByNorth, bz)];
            pop[18] = loadGhostYSouth[idxPopY(tx, tz, 4, bx, loadByNorth, bz)];
            #ifdef D3Q27
            pop[20] = loadGhostYSouth[idxPopY(tx, tz, 5, bx, loadByNorth, bz)];
            pop[22] = loadGhostYSouth[idxPopY(tx, tz, 6, bx, loadByNorth, bz)];
            pop[23] = loadGhostYSouth[idxPopY(tx, tz, 7, bx, loadByNorth, bz)];
            pop[26] = loadGhostYSouth[idxPopY(tx, tz, 8, bx, loadByNorth, bz)];
            #endif // D3Q27
        }

        if (tz == 0)
        { // b
            pop[ 5] = loadGhostZFront[idxPopZ(tx, ty, 0, bx, by, loadBzBack)];
            pop[ 9] = loadGhostZFront[idxPopZ(tx, ty, 1, bx, by, loadBzBack)];
            pop[11] = loadGhostZFront[idxPopZ(tx, ty, 2, bx, by, loadBzBack)];
            pop[16] = loadGhostZFront[idxPopZ(tx, ty, 3, bx, by, loadBzBack)];
            pop[18] = loadGhostZFront[idxPopZ(tx, ty, 4, bx, by, loadBzBack)];
            #ifdef D3Q27
            pop[19] = loadGhostZFront[idxPopZ(tx, ty, 5, bx, by, loadBzBack)];
            pop[22] = loadGhostZFront[idxPopZ(tx, ty, 6, bx, by, loadBzBack)];
            pop[23] = loadGhostZFront[idxPopZ(tx, ty, 7, bx, by, loadBzBack)];
            pop[25] = loadGhostZFront[idxPopZ(tx, ty, 8, bx, by, loadBzBack)];
            #endif // D3Q27
        }
        else if (tz == (BLOCK_NZ - 1))
        { // f
            pop[ 6] = loadGhostZBack[idxPopZ(tx, ty, 0, bx, by, loadBzFront)];
            pop[10] = loadGhostZBack[idxPopZ(tx, ty, 1, bx, by, loadBzFront)];
            pop[12] = loadGhostZBack[idxPopZ(tx, ty, 2, bx, by, loadBzFront)];
            pop[15] = loadGhostZBack[idxPopZ(tx, ty, 3, bx, by, loadBzFront)];
            pop[17] = loadGhostZBack[idxPopZ(tx, ty, 4, bx, by, loadBzFront)];
            #ifdef D3Q27
            pop[20] = loadGhostZBack[idxPopZ(tx, ty, 5, bx, by, loadBzFront)];
            pop[21] = loadGhostZBack[idxPopZ(tx, ty, 6, bx, by, loadBzFront)];
            pop[24] = loadGhostZBack[idxPopZ(tx, ty, 7, bx, by, loadBzFront)];
            pop[26] = loadGhostZBack[idxPopZ(tx, ty, 8, bx, by, loadBzFront)];
            #endif // D3Q27
        }
    }else{
        if (tx == 0) { //w
            pop[ 1] = loadGhostXWest[idxPopX(ty    , tz   ,   0, loadBxWest , by        , bz        )];
            pop[ 7] = loadGhostXWest[idxPopX(tym1  , tz   ,   1, loadBxWest , byAtSouth , bz        )];
            pop[ 9] = loadGhostXWest[idxPopX(ty    , tzm1 ,   2, loadBxWest , by        , bzAtBack  )];
            pop[13] = loadGhostXWest[idxPopX(typ1  , tz   ,   3, loadBxWest , byAtNorth , bz        )];
            pop[15] = loadGhostXWest[idxPopX(ty    , tzp1 ,   4, loadBxWest , by        , bzAtFront )];
            #ifdef D3Q27
            pop[19] = loadGhostXWest[idxPopX(tym1  , tzm1 ,   5, loadBxWest , byAtSouth , bzAtBack  )];
            pop[21] = loadGhostXWest[idxPopX(tym1  , tzp1 ,   6, loadBxWest , byAtSouth , bzAtFront )];
            pop[23] = loadGhostXWest[idxPopX(typ1  , tzm1 ,   7, loadBxWest , byAtNorth , bzAtBack  )];
            pop[26] = loadGhostXWest[idxPopX(typ1  , tzp1 ,   8, loadBxWest , byAtNorth , bzAtFront )];
            #endif //D3Q27
        }else if (tx == (BLOCK_NX - 1)){ //e
            pop[ 2] = loadGhostXEast[idxPopX(ty    , tz   ,   0, loadBxEast , by        , bz        )];
            pop[ 8] = loadGhostXEast[idxPopX(typ1  , tz   ,   1, loadBxEast , byAtNorth , bz        )];
            pop[10] = loadGhostXEast[idxPopX(ty    , tzp1 ,   2, loadBxEast , by        , bzAtFront )];
            pop[14] = loadGhostXEast[idxPopX(tym1  , tz   ,   3, loadBxEast , byAtSouth , bz        )];
            pop[16] = loadGhostXEast[idxPopX(ty    , tzm1 ,   4, loadBxEast , by        , bzAtBack  )];
            #ifdef D3Q27
            pop[20] = loadGhostXEast[idxPopX(typ1  , tzp1 ,   5, loadBxEast , byAtNorth , bzAtFront )];
            pop[22] = loadGhostXEast[idxPopX(typ1  , tzm1 ,   6, loadBxEast , byAtNorth , bzAtBack  )];
            pop[24] = loadGhostXEast[idxPopX(tym1  , tzp1 ,   7, loadBxEast , byAtSouth , bzAtFront )];
            pop[25] = loadGhostXEast[idxPopX(tym1  , tzm1 ,   8, loadBxEast , byAtSouth , bzAtBack  )];
            #endif //D3Q27
        }

        if (ty == 0){ //s
            pop[ 3] = loadGhostYSouth[idxPopY(tx    , tz   ,   0, bx         , loadBySouth, bz        )];
            pop[ 7] = loadGhostYSouth[idxPopY(txm1  , tz   ,   1, bxAtWest   , loadBySouth, bz        )];
            pop[11] = loadGhostYSouth[idxPopY(tx    , tzm1 ,   2, bx         , loadBySouth, bzAtBack  )];
            pop[14] = loadGhostYSouth[idxPopY(txp1  , tz   ,   3, bxAtEast   , loadBySouth, bz        )];
            pop[17] = loadGhostYSouth[idxPopY(tx    , tzp1 ,   4, bx         , loadBySouth, bzAtFront )];
            #ifdef D3Q27
            pop[19] = loadGhostYSouth[idxPopY(txm1  , tzm1 ,   5, bxAtWest   , loadBySouth, bzAtBack  )];
            pop[21] = loadGhostYSouth[idxPopY(txm1  , tzp1 ,   6, bxAtWest   , loadBySouth, bzAtFront )];
            pop[24] = loadGhostYSouth[idxPopY(txp1  , tzp1 ,   7, bxAtEast   , loadBySouth, bzAtFront )];
            pop[25] = loadGhostYSouth[idxPopY(txp1  , tzm1 ,   8, bxAtEast   , loadBySouth, bzAtBack  )];
            #endif // D3Q27
        }else if (ty == (BLOCK_NY - 1)){ //n
            pop[ 4] = loadGhostYNorth[idxPopY(tx    , tz   ,   0, bx         , loadByNorth, bz        )];
            pop[ 8] = loadGhostYNorth[idxPopY(txp1  , tz   ,   1, bxAtEast   , loadByNorth, bz        )];
            pop[12] = loadGhostYNorth[idxPopY(tx    , tzp1 ,   2, bx         , loadByNorth, bzAtFront )];
            pop[13] = loadGhostYNorth[idxPopY(txm1  , tz   ,   3, bxAtWest   , loadByNorth, bz        )];
            pop[18] = loadGhostYNorth[idxPopY(tx    , tzm1 ,   4, bx         , loadByNorth, bzAtBack  )];
            #ifdef D3Q27
            pop[20] = loadGhostYNorth[idxPopY(txp1  , tzp1 ,   5, bxAtEast   , loadByNorth, bzAtFront )];
            pop[22] = loadGhostYNorth[idxPopY(txp1  , tzm1 ,   6, bxAtEast   , loadByNorth, bzAtBack  )];
            pop[23] = loadGhostYNorth[idxPopY(txm1  , tzm1 ,   7, bxAtWest   , loadByNorth, bzAtBack  )];
            pop[26] = loadGhostYNorth[idxPopY(txm1  , tzp1 ,   8, bxAtWest   , loadByNorth, bzAtFront )];
            #endif //D3Q27
        }

        if (tz == 0){ //b
            pop[ 5] = loadGhostZBack[idxPopZ(tx    , ty   ,   0, bx         , by        , loadBzBack)];
            pop[ 9] = loadGhostZBack[idxPopZ(txm1  , ty   ,   1, bxAtWest   , by        , loadBzBack)];
            pop[11] = loadGhostZBack[idxPopZ(tx    , tym1 ,   2, bx         , byAtSouth , loadBzBack)];
            pop[16] = loadGhostZBack[idxPopZ(txp1  , ty   ,   3, bxAtEast   , by        , loadBzBack)];
            pop[18] = loadGhostZBack[idxPopZ(tx    , typ1 ,   4, bx         , byAtNorth , loadBzBack)];
            #ifdef D3Q27
            pop[19] = loadGhostZBack[idxPopZ(txm1  , tym1 ,   5, bxAtWest   , byAtSouth , loadBzBack)];
            pop[22] = loadGhostZBack[idxPopZ(txp1  , typ1 ,   6, bxAtEast   , byAtNorth , loadBzBack)];
            pop[23] = loadGhostZBack[idxPopZ(txm1  , typ1 ,   7, bxAtWest   , byAtNorth , loadBzBack)];
            pop[25] = loadGhostZBack[idxPopZ(txp1  , tym1 ,   8, bxAtEast   , byAtSouth , loadBzBack)];
            #endif //D3Q27
        }else if (tz == (BLOCK_NZ - 1)){ //f
            pop[ 6] = loadGhostZFront[idxPopZ(tx    , ty   ,   0, bx         , by        , loadBzFront)];
            pop[10] = loadGhostZFront[idxPopZ(txp1  , ty   ,   1, bxAtEast   , by        , loadBzFront)];
            pop[12] = loadGhostZFront[idxPopZ(tx    , typ1 ,   2, bx         , byAtNorth , loadBzFront)];
            pop[15] = loadGhostZFront[idxPopZ(txm1  , ty   ,   3, bxAtWest   , by        , loadBzFront)];
            pop[17] = loadGhostZFront[idxPopZ(tx    , tym1 ,   4, bx         , byAtSouth , loadBzFront)];
            #ifdef D3Q27
            pop[20] = loadGhostZFront[idxPopZ(txp1  , typ1 ,   5, bxAtEast   , byAtNorth , loadBzFront)];
            pop[21] = loadGhostZFront[idxPopZ(txm1  , tym1 ,   6, bxAtWest   , byAtSouth , loadBzFront)];
            pop[24] = loadGhostZFront[idxPopZ(txp1  , tym1 ,   7, bxAtEast   , byAtSouth , loadBzFront)];
            pop[26] = loadGhostZFront[idxPopZ(txm1  , typ1 ,   8, bxAtWest   , byAtNorth , loadBzFront)];
            #endif //D3Q27
        }
    }

