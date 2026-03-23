// AA scheme: read from ghost, parity-aware block indices
const dfloat* loadGhostXWest  = ghostInterface.Azz.X_1;
const dfloat* loadGhostXEast  = ghostInterface.Azz.X_0;
const dfloat* loadGhostYSouth = ghostInterface.Azz.Y_1;
const dfloat* loadGhostYNorth = ghostInterface.Azz.Y_0;
const dfloat* loadGhostZBack  = ghostInterface.Azz.Z_1;
const dfloat* loadGhostZFront = ghostInterface.Azz.Z_0;

const int loadBxWest  = (stepParity ? bx : bxm1);
const int loadBxEast  = (stepParity ? bx : bxp1);
const int loadBySouth = (stepParity ? by : bym1);
const int loadByNorth = (stepParity ? by : byp1);
const int loadBzBack  = (stepParity ? bz : bzm1);
const int loadBzFront = (stepParity ? bz : bzp1);

const int bxAtWest  = (tx == 0)              ? bxm1 : bx;
const int bxAtEast  = (tx == (BLOCK_NX - 1)) ? bxp1 : bx;
const int byAtSouth = (ty == 0)              ? bym1 : by;
const int byAtNorth = (ty == (BLOCK_NY - 1)) ? byp1 : by;
const int bzAtBack  = (tz == 0)              ? bzm1 : bz;
const int bzAtFront = (tz == (BLOCK_NZ - 1)) ? bzp1 : bz;

    if (stepParity){
        if (tx == 0)
        { //w
            gNode[ 1] = loadGhostXEast[g_idxPopX(ty, tz, 0, loadBxWest, by, bz)];
            gNode[ 7] = loadGhostXEast[g_idxPopX(ty, tz, 1, loadBxWest, by, bz)];
            gNode[ 9] = loadGhostXEast[g_idxPopX(ty, tz, 2, loadBxWest, by, bz)];
            gNode[13] = loadGhostXEast[g_idxPopX(ty, tz, 3, loadBxWest, by, bz)];
            gNode[15] = loadGhostXEast[g_idxPopX(ty, tz, 4, loadBxWest, by, bz)];
        }
        else if (tx == (BLOCK_NX - 1))
        { //e
            gNode[ 2] = loadGhostXWest[g_idxPopX(ty, tz, 0, loadBxEast, by, bz)];
            gNode[ 8] = loadGhostXWest[g_idxPopX(ty, tz, 1, loadBxEast, by, bz)];
            gNode[10] = loadGhostXWest[g_idxPopX(ty, tz, 2, loadBxEast, by, bz)];
            gNode[14] = loadGhostXWest[g_idxPopX(ty, tz, 3, loadBxEast, by, bz)];
            gNode[16] = loadGhostXWest[g_idxPopX(ty, tz, 4, loadBxEast, by, bz)];
        }

        if (ty == 0)
        { //s
            gNode[ 3] = loadGhostYNorth[g_idxPopY(tx, tz, 0, bx, loadBySouth, bz)];
            gNode[ 7] = loadGhostYNorth[g_idxPopY(tx, tz, 1, bx, loadBySouth, bz)];
            gNode[11] = loadGhostYNorth[g_idxPopY(tx, tz, 2, bx, loadBySouth, bz)];
            gNode[14] = loadGhostYNorth[g_idxPopY(tx, tz, 3, bx, loadBySouth, bz)];
            gNode[17] = loadGhostYNorth[g_idxPopY(tx, tz, 4, bx, loadBySouth, bz)];
        }
        else if (ty == (BLOCK_NY - 1))
        { //n
            gNode[ 4] = loadGhostYSouth[g_idxPopY(tx, tz, 0, bx, loadByNorth, bz)];
            gNode[ 8] = loadGhostYSouth[g_idxPopY(tx, tz, 1, bx, loadByNorth, bz)];
            gNode[12] = loadGhostYSouth[g_idxPopY(tx, tz, 2, bx, loadByNorth, bz)];
            gNode[13] = loadGhostYSouth[g_idxPopY(tx, tz, 3, bx, loadByNorth, bz)];
            gNode[18] = loadGhostYSouth[g_idxPopY(tx, tz, 4, bx, loadByNorth, bz)];
        }

        if (tz == 0)
        { //b
            gNode[ 5] = loadGhostZFront[g_idxPopZ(tx, ty, 0, bx, by, loadBzBack)];
            gNode[ 9] = loadGhostZFront[g_idxPopZ(tx, ty, 1, bx, by, loadBzBack)];
            gNode[11] = loadGhostZFront[g_idxPopZ(tx, ty, 2, bx, by, loadBzBack)];
            gNode[16] = loadGhostZFront[g_idxPopZ(tx, ty, 3, bx, by, loadBzBack)];
            gNode[18] = loadGhostZFront[g_idxPopZ(tx, ty, 4, bx, by, loadBzBack)];
        }
        else if (tz == (BLOCK_NZ - 1))
        { //f
            gNode[ 6] = loadGhostZBack[g_idxPopZ(tx, ty, 0, bx, by, loadBzFront)];
            gNode[10] = loadGhostZBack[g_idxPopZ(tx, ty, 1, bx, by, loadBzFront)];
            gNode[12] = loadGhostZBack[g_idxPopZ(tx, ty, 2, bx, by, loadBzFront)];
            gNode[15] = loadGhostZBack[g_idxPopZ(tx, ty, 3, bx, by, loadBzFront)];
            gNode[17] = loadGhostZBack[g_idxPopZ(tx, ty, 4, bx, by, loadBzFront)];
        }

    }else{
        if (tx == 0){ //w
            gNode[ 1] = loadGhostXWest[g_idxPopX(ty    , tz   ,   0, loadBxWest , by        , bz        )];
            gNode[ 7] = loadGhostXWest[g_idxPopX(tym1  , tz   ,   1, loadBxWest , byAtSouth , bz        )];
            gNode[ 9] = loadGhostXWest[g_idxPopX(ty    , tzm1 ,   2, loadBxWest , by        , bzAtBack  )];
            gNode[13] = loadGhostXWest[g_idxPopX(typ1  , tz   ,   3, loadBxWest , byAtNorth , bz        )];
            gNode[15] = loadGhostXWest[g_idxPopX(ty    , tzp1 ,   4, loadBxWest , by        , bzAtFront )];
        }else if (tx == (BLOCK_NX - 1)){ //e
            gNode[ 2] = loadGhostXEast[g_idxPopX(ty    , tz   ,   0, loadBxEast , by        , bz        )];
            gNode[ 8] = loadGhostXEast[g_idxPopX(typ1  , tz   ,   1, loadBxEast , byAtNorth , bz        )];
            gNode[10] = loadGhostXEast[g_idxPopX(ty    , tzp1 ,   2, loadBxEast , by        , bzAtFront )];
            gNode[14] = loadGhostXEast[g_idxPopX(tym1  , tz   ,   3, loadBxEast , byAtSouth , bz        )];
            gNode[16] = loadGhostXEast[g_idxPopX(ty    , tzm1 ,   4, loadBxEast , by        , bzAtBack  )];
        }

        if (ty == 0){ //s
            gNode[ 3] = loadGhostYSouth[g_idxPopY(tx    , tz   ,   0, bx         , loadBySouth, bz        )];
            gNode[ 7] = loadGhostYSouth[g_idxPopY(txm1  , tz   ,   1, bxAtWest   , loadBySouth, bz        )];
            gNode[11] = loadGhostYSouth[g_idxPopY(tx    , tzm1 ,   2, bx         , loadBySouth, bzAtBack  )];
            gNode[14] = loadGhostYSouth[g_idxPopY(txp1  , tz   ,   3, bxAtEast   , loadBySouth, bz        )];
            gNode[17] = loadGhostYSouth[g_idxPopY(tx    , tzp1 ,   4, bx         , loadBySouth, bzAtFront )];
        }else if (ty == (BLOCK_NY - 1)){ //n
            gNode[ 4] = loadGhostYNorth[g_idxPopY(tx    , tz   ,   0, bx         , loadByNorth, bz        )];
            gNode[ 8] = loadGhostYNorth[g_idxPopY(txp1  , tz   ,   1, bxAtEast   , loadByNorth, bz        )];
            gNode[12] = loadGhostYNorth[g_idxPopY(tx    , tzp1 ,   2, bx         , loadByNorth, bzAtFront )];
            gNode[13] = loadGhostYNorth[g_idxPopY(txm1  , tz   ,   3, bxAtWest   , loadByNorth, bz        )];
            gNode[18] = loadGhostYNorth[g_idxPopY(tx    , tzm1 ,   4, bx         , loadByNorth, bzAtBack  )];
        }

        if (tz == 0){ //b
            gNode[ 5] = loadGhostZBack[g_idxPopZ(tx    , ty   ,   0, bx         , by        , loadBzBack)];
            gNode[ 9] = loadGhostZBack[g_idxPopZ(txm1  , ty   ,   1, bxAtWest   , by        , loadBzBack)];
            gNode[11] = loadGhostZBack[g_idxPopZ(tx    , tym1 ,   2, bx         , byAtSouth , loadBzBack)];
            gNode[16] = loadGhostZBack[g_idxPopZ(txp1  , ty   ,   3, bxAtEast   , by        , loadBzBack)];
            gNode[18] = loadGhostZBack[g_idxPopZ(tx    , typ1 ,   4, bx         , byAtNorth , loadBzBack)];
        }else if (tz == (BLOCK_NZ - 1)){ //f
            gNode[ 6] = loadGhostZFront[g_idxPopZ(tx    , ty   ,   0, bx         , by        , loadBzFront)];
            gNode[10] = loadGhostZFront[g_idxPopZ(txp1  , ty   ,   1, bxAtEast   , by        , loadBzFront)];
            gNode[12] = loadGhostZFront[g_idxPopZ(tx    , typ1 ,   2, bx         , byAtNorth , loadBzFront)];
            gNode[15] = loadGhostZFront[g_idxPopZ(txm1  , ty   ,   3, bxAtWest   , by        , loadBzFront)];
            gNode[17] = loadGhostZFront[g_idxPopZ(tx    , tym1 ,   4, bx         , byAtSouth , loadBzFront)];
        }

    }
