// AA scheme: read from ghost, parity-aware block indices
const dfloat* loadGhostXWest  = ghostInterface.phi.X_1;
const dfloat* loadGhostXEast  = ghostInterface.phi.X_0;
const dfloat* loadGhostYSouth = ghostInterface.phi.Y_1;
const dfloat* loadGhostYNorth = ghostInterface.phi.Y_0;
const dfloat* loadGhostZBack  = ghostInterface.phi.Z_1;
const dfloat* loadGhostZFront = ghostInterface.phi.Z_0;

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
// DEV_REF_START (from dev during merge)
// DEV_REF: // Load from global memory - face neighbors for D3G7/D3G19, edge neighbors only for D3G19
// DEV_REF: if (tx == 0) { //w
// DEV_REF:         gNode[ 1] = ghostInterface.phi_fGhost.X_1[g_idxPopX(ty  , tz, 0, bxm1, by                                       , bz)];
// DEV_REF: #ifdef D3G19
// DEV_REF:         gNode[ 7] = ghostInterface.phi_fGhost.X_1[g_idxPopX(tym1, tz, 1, bxm1, ((ty == 0) ? bym1 : by)                  , bz)];
// DEV_REF:         gNode[ 9] = ghostInterface.phi_fGhost.X_1[g_idxPopX(ty, tzm1, 2, bxm1, by                                       , ((tz == 0) ? bzm1 : bz))];
// DEV_REF:         gNode[13] = ghostInterface.phi_fGhost.X_1[g_idxPopX(typ1, tz, 3, bxm1, ((ty == (BLOCK_NY - 1)) ? byp1 : by)     , bz)];
// DEV_REF:         gNode[15] = ghostInterface.phi_fGhost.X_1[g_idxPopX(ty, tzp1, 4, bxm1, by                                       , ((tz == (BLOCK_NZ - 1)) ? bzp1 : bz))];
// DEV_REF: #endif
// DEV_REF: #ifdef D3G27
// DEV_REF:         gNode[ 7] = ghostInterface.phi_fGhost.X_1[g_idxPopX(tym1, tz, 1, bxm1, ((ty == 0) ? bym1 : by)                  , bz)];
// DEV_REF:         gNode[ 9] = ghostInterface.phi_fGhost.X_1[g_idxPopX(ty, tzm1, 2, bxm1, by                                       , ((tz == 0) ? bzm1 : bz))];
// DEV_REF:         gNode[13] = ghostInterface.phi_fGhost.X_1[g_idxPopX(typ1, tz, 3, bxm1, ((ty == (BLOCK_NY - 1)) ? byp1 : by)     , bz)];
// DEV_REF:         gNode[15] = ghostInterface.phi_fGhost.X_1[g_idxPopX(ty, tzp1, 4, bxm1, by                                       , ((tz == (BLOCK_NZ - 1)) ? bzp1 : bz))];
// DEV_REF:         gNode[19] = ghostInterface.phi_fGhost.X_1[g_idxPopX(tym1, tzm1, 5, bxm1, ((ty == 0) ? bym1 : by)                , ((tz == 0) ? bzm1 : bz))];
// DEV_REF:         gNode[21] = ghostInterface.phi_fGhost.X_1[g_idxPopX(tym1, tzp1, 6, bxm1, ((ty == 0) ? bym1 : by)                , ((tz == (BLOCK_NZ - 1)) ? bzp1 : bz))];
// DEV_REF:         gNode[23] = ghostInterface.phi_fGhost.X_1[g_idxPopX(typ1, tzm1, 7, bxm1, ((ty == (BLOCK_NY - 1)) ? byp1 : by)   , ((tz == 0) ? bzm1 : bz))];
// DEV_REF:         gNode[26] = ghostInterface.phi_fGhost.X_1[g_idxPopX(typ1, tzp1, 8, bxm1, ((ty == (BLOCK_NY - 1)) ? byp1 : by)   , ((tz == (BLOCK_NZ - 1)) ? bzp1 : bz))];
// DEV_REF: #endif
// DEV_REF: }
// DEV_REF: else if (tx == (BLOCK_NX - 1))
// DEV_REF:     { //e
// DEV_REF:         gNode[ 2] = ghostInterface.phi_fGhost.X_0[g_idxPopX(ty  , tz, 0, bxp1, by                                       , bz)];
// DEV_REF: #ifdef D3G19
// DEV_REF:         gNode[ 8] = ghostInterface.phi_fGhost.X_0[g_idxPopX(typ1, tz, 1, bxp1, ((ty == (BLOCK_NY - 1)) ? byp1 : by)     , bz)];
// DEV_REF:         gNode[10] = ghostInterface.phi_fGhost.X_0[g_idxPopX(ty, tzp1, 2, bxp1, by                                       , ((tz == (BLOCK_NZ - 1)) ? bzp1 : bz))];
// DEV_REF:         gNode[14] = ghostInterface.phi_fGhost.X_0[g_idxPopX(tym1, tz, 3, bxp1, ((ty == 0) ? bym1 : by)                  , bz)];
// DEV_REF:         gNode[16] = ghostInterface.phi_fGhost.X_0[g_idxPopX(ty, tzm1, 4, bxp1, by                                       , ((tz == 0) ? bzm1 : bz))];
// DEV_REF: #endif
// DEV_REF: #ifdef D3G27
// DEV_REF:         gNode[ 8] = ghostInterface.phi_fGhost.X_0[g_idxPopX(typ1, tz, 1, bxp1, ((ty == (BLOCK_NY - 1)) ? byp1 : by)     , bz)];
// DEV_REF:         gNode[10] = ghostInterface.phi_fGhost.X_0[g_idxPopX(ty, tzp1, 2, bxp1, by                                       , ((tz == (BLOCK_NZ - 1)) ? bzp1 : bz))];
// DEV_REF:         gNode[14] = ghostInterface.phi_fGhost.X_0[g_idxPopX(tym1, tz, 3, bxp1, ((ty == 0) ? bym1 : by)                  , bz)];
// DEV_REF:         gNode[16] = ghostInterface.phi_fGhost.X_0[g_idxPopX(ty, tzm1, 4, bxp1, by                                       , ((tz == 0) ? bzm1 : bz))];
// DEV_REF:         gNode[20] = ghostInterface.phi_fGhost.X_0[g_idxPopX(typ1, tzp1, 5, bxp1, ((ty == (BLOCK_NY - 1)) ? byp1 : by)   , ((tz == (BLOCK_NZ - 1)) ? bzp1 : bz))];
// DEV_REF:         gNode[22] = ghostInterface.phi_fGhost.X_0[g_idxPopX(typ1, tzm1, 6, bxp1, ((ty == (BLOCK_NY - 1)) ? byp1 : by)   , ((tz == 0) ? bzm1 : bz))];
// DEV_REF:         gNode[24] = ghostInterface.phi_fGhost.X_0[g_idxPopX(tym1, tzp1, 7, bxp1, ((ty == 0) ? bym1 : by)                , ((tz == (BLOCK_NZ - 1)) ? bzp1 : bz))];
// DEV_REF:         gNode[25] = ghostInterface.phi_fGhost.X_0[g_idxPopX(tym1, tzm1, 8, bxp1, ((ty == 0) ? bym1 : by)                , ((tz == 0) ? bzm1 : bz))];
// DEV_REF: #endif
// DEV_REF:     }
// DEV_REF:
// DEV_REF: if (ty == 0)
// DEV_REF:     { //s
// DEV_REF:         gNode[ 3] = ghostInterface.phi_fGhost.Y_1[g_idxPopY(tx  , tz, 0, bx                                     , bym1, bz)];
// DEV_REF: #ifdef D3G19
// DEV_REF:         gNode[ 7] = ghostInterface.phi_fGhost.Y_1[g_idxPopY(txm1, tz, 1, ((tx == 0) ? bxm1 : bx)                , bym1, bz)];
// DEV_REF:         gNode[11] = ghostInterface.phi_fGhost.Y_1[g_idxPopY(tx, tzm1, 2, bx                                     , bym1, ((tz == 0) ? bzm1 : bz))];
// DEV_REF:         gNode[14] = ghostInterface.phi_fGhost.Y_1[g_idxPopY(txp1, tz, 3, ((tx == (BLOCK_NX - 1)) ? bxp1 : bx)   , bym1, bz)];
// DEV_REF:         gNode[17] = ghostInterface.phi_fGhost.Y_1[g_idxPopY(tx, tzp1, 4, bx                                     , bym1, ((tz == (BLOCK_NZ - 1)) ? bzp1 : bz))];
// DEV_REF: #endif
// DEV_REF: #ifdef D3G27
// DEV_REF:         gNode[ 7] = ghostInterface.phi_fGhost.Y_1[g_idxPopY(txm1, tz, 1, ((tx == 0) ? bxm1 : bx)                , bym1, bz)];
// DEV_REF:         gNode[11] = ghostInterface.phi_fGhost.Y_1[g_idxPopY(tx, tzm1, 2, bx                                     , bym1, ((tz == 0) ? bzm1 : bz))];
// DEV_REF:         gNode[14] = ghostInterface.phi_fGhost.Y_1[g_idxPopY(txp1, tz, 3, ((tx == (BLOCK_NX - 1)) ? bxp1 : bx)   , bym1, bz)];
// DEV_REF:         gNode[17] = ghostInterface.phi_fGhost.Y_1[g_idxPopY(tx, tzp1, 4, bx                                     , bym1, ((tz == (BLOCK_NZ - 1)) ? bzp1 : bz))];
// DEV_REF:         gNode[19] = ghostInterface.phi_fGhost.Y_1[g_idxPopY(txm1, tzm1, 5, ((tx == 0) ? bxm1 : bx)              , bym1, ((tz == 0) ? bzm1 : bz))];
// DEV_REF:         gNode[21] = ghostInterface.phi_fGhost.Y_1[g_idxPopY(txm1, tzp1, 6, ((tx == 0) ? bxm1 : bx)              , bym1, ((tz == (BLOCK_NZ - 1)) ? bzp1 : bz))];
// DEV_REF:         gNode[24] = ghostInterface.phi_fGhost.Y_1[g_idxPopY(txp1, tzp1, 7, ((tx == (BLOCK_NX - 1)) ? bxp1 : bx) , bym1, ((tz == (BLOCK_NZ - 1)) ? bzp1 : bz))];
// DEV_REF:         gNode[25] = ghostInterface.phi_fGhost.Y_1[g_idxPopY(txp1, tzm1, 8, ((tx == (BLOCK_NX - 1)) ? bxp1 : bx) , bym1, ((tz == 0) ? bzm1 : bz))];
// DEV_REF: #endif
// DEV_REF:     }
// DEV_REF: else if (ty == (BLOCK_NY - 1))
// DEV_REF:     { //n
// DEV_REF:         gNode[ 4] = ghostInterface.phi_fGhost.Y_0[g_idxPopY(tx  , tz, 0, bx                                     , byp1, bz)];
// DEV_REF: #ifdef D3G19
// DEV_REF:         gNode[ 8] = ghostInterface.phi_fGhost.Y_0[g_idxPopY(txp1, tz, 1, ((tx == (BLOCK_NX - 1)) ? bxp1 : bx)   , byp1, bz)];
// DEV_REF:         gNode[12] = ghostInterface.phi_fGhost.Y_0[g_idxPopY(tx, tzp1, 2, bx                                     , byp1, ((tz == (BLOCK_NZ - 1)) ? bzp1 : bz))];
// DEV_REF:         gNode[13] = ghostInterface.phi_fGhost.Y_0[g_idxPopY(txm1, tz, 3, ((tx == 0) ? bxm1 : bx)                , byp1, bz)];
// DEV_REF:         gNode[18] = ghostInterface.phi_fGhost.Y_0[g_idxPopY(tx, tzm1, 4, bx                                     , byp1, ((tz == 0) ? bzm1 : bz))];
// DEV_REF: #endif
// DEV_REF: #ifdef D3G27
// DEV_REF:         gNode[ 8] = ghostInterface.phi_fGhost.Y_0[g_idxPopY(txp1, tz, 1, ((tx == (BLOCK_NX - 1)) ? bxp1 : bx)   , byp1, bz)];
// DEV_REF:         gNode[12] = ghostInterface.phi_fGhost.Y_0[g_idxPopY(tx, tzp1, 2, bx                                     , byp1, ((tz == (BLOCK_NZ - 1)) ? bzp1 : bz))];
// DEV_REF:         gNode[13] = ghostInterface.phi_fGhost.Y_0[g_idxPopY(txm1, tz, 3, ((tx == 0) ? bxm1 : bx)                , byp1, bz)];
// DEV_REF:         gNode[18] = ghostInterface.phi_fGhost.Y_0[g_idxPopY(tx, tzm1, 4, bx                                     , byp1, ((tz == 0) ? bzm1 : bz))];
// DEV_REF:         gNode[20] = ghostInterface.phi_fGhost.Y_0[g_idxPopY(txp1, tzp1, 5, ((tx == (BLOCK_NX - 1)) ? bxp1 : bx) , byp1, ((tz == (BLOCK_NZ - 1)) ? bzp1 : bz))];
// DEV_REF:         gNode[22] = ghostInterface.phi_fGhost.Y_0[g_idxPopY(txp1, tzm1, 6, ((tx == (BLOCK_NX - 1)) ? bxp1 : bx) , byp1, ((tz == 0) ? bzm1 : bz))];
// DEV_REF:         gNode[23] = ghostInterface.phi_fGhost.Y_0[g_idxPopY(txm1, tzm1, 7, ((tx == 0) ? bxm1 : bx)              , byp1, ((tz == 0) ? bzm1 : bz))];
// DEV_REF:         gNode[26] = ghostInterface.phi_fGhost.Y_0[g_idxPopY(txm1, tzp1, 8, ((tx == 0) ? bxm1 : bx)              , byp1, ((tz == (BLOCK_NZ - 1)) ? bzp1 : bz))];
// DEV_REF: #endif
// DEV_REF:     }
// DEV_REF:
// DEV_REF_END

    if (stepParity){
        if (tx == 0)
        { //w
            gNode[ 1] = loadGhostXEast[g_idxPopX(ty, tz, 0, loadBxWest, by, bz)];
#ifdef D3G19
            gNode[ 7] = loadGhostXEast[g_idxPopX(ty, tz, 1, loadBxWest, by, bz)];
            gNode[ 9] = loadGhostXEast[g_idxPopX(ty, tz, 2, loadBxWest, by, bz)];
            gNode[13] = loadGhostXEast[g_idxPopX(ty, tz, 3, loadBxWest, by, bz)];
            gNode[15] = loadGhostXEast[g_idxPopX(ty, tz, 4, loadBxWest, by, bz)];
#endif
        }
        else if (tx == (BLOCK_NX - 1))
        { //e
            gNode[ 2] = loadGhostXWest[g_idxPopX(ty, tz, 0, loadBxEast, by, bz)];
#ifdef D3G19
            gNode[ 8] = loadGhostXWest[g_idxPopX(ty, tz, 1, loadBxEast, by, bz)];
            gNode[10] = loadGhostXWest[g_idxPopX(ty, tz, 2, loadBxEast, by, bz)];
            gNode[14] = loadGhostXWest[g_idxPopX(ty, tz, 3, loadBxEast, by, bz)];
            gNode[16] = loadGhostXWest[g_idxPopX(ty, tz, 4, loadBxEast, by, bz)];
#endif
        }

        if (ty == 0)
        { //s
            gNode[ 3] = loadGhostYNorth[g_idxPopY(tx, tz, 0, bx, loadBySouth, bz)];
#ifdef D3G19
            gNode[ 7] = loadGhostYNorth[g_idxPopY(tx, tz, 1, bx, loadBySouth, bz)];
            gNode[11] = loadGhostYNorth[g_idxPopY(tx, tz, 2, bx, loadBySouth, bz)];
            gNode[14] = loadGhostYNorth[g_idxPopY(tx, tz, 3, bx, loadBySouth, bz)];
            gNode[17] = loadGhostYNorth[g_idxPopY(tx, tz, 4, bx, loadBySouth, bz)];
#endif
        }
        else if (ty == (BLOCK_NY - 1))
        { //n
            gNode[ 4] = loadGhostYSouth[g_idxPopY(tx, tz, 0, bx, loadByNorth, bz)];
#ifdef D3G19
            gNode[ 8] = loadGhostYSouth[g_idxPopY(tx, tz, 1, bx, loadByNorth, bz)];
            gNode[12] = loadGhostYSouth[g_idxPopY(tx, tz, 2, bx, loadByNorth, bz)];
            gNode[13] = loadGhostYSouth[g_idxPopY(tx, tz, 3, bx, loadByNorth, bz)];
            gNode[18] = loadGhostYSouth[g_idxPopY(tx, tz, 4, bx, loadByNorth, bz)];
#endif
        }

        if (tz == 0)
        { //b
            gNode[ 5] = loadGhostZFront[g_idxPopZ(tx, ty, 0, bx, by, loadBzBack)];
#ifdef D3G19
            gNode[ 9] = loadGhostZFront[g_idxPopZ(tx, ty, 1, bx, by, loadBzBack)];
            gNode[11] = loadGhostZFront[g_idxPopZ(tx, ty, 2, bx, by, loadBzBack)];
            gNode[16] = loadGhostZFront[g_idxPopZ(tx, ty, 3, bx, by, loadBzBack)];
            gNode[18] = loadGhostZFront[g_idxPopZ(tx, ty, 4, bx, by, loadBzBack)];
#endif
        }
        else if (tz == (BLOCK_NZ - 1))
        { //f
            gNode[ 6] = loadGhostZBack[g_idxPopZ(tx, ty, 0, bx, by, loadBzFront)];
#ifdef D3G19
            gNode[10] = loadGhostZBack[g_idxPopZ(tx, ty, 1, bx, by, loadBzFront)];
            gNode[12] = loadGhostZBack[g_idxPopZ(tx, ty, 2, bx, by, loadBzFront)];
            gNode[15] = loadGhostZBack[g_idxPopZ(tx, ty, 3, bx, by, loadBzFront)];
            gNode[17] = loadGhostZBack[g_idxPopZ(tx, ty, 4, bx, by, loadBzFront)];
#endif
        }

    }else{
        if (tx == 0){ //w
            gNode[ 1] = loadGhostXWest[g_idxPopX(ty    , tz   ,   0, loadBxWest , by        , bz        )];
#ifdef D3G19
            gNode[ 7] = loadGhostXWest[g_idxPopX(tym1  , tz   ,   1, loadBxWest , byAtSouth , bz        )];
            gNode[ 9] = loadGhostXWest[g_idxPopX(ty    , tzm1 ,   2, loadBxWest , by        , bzAtBack  )];
            gNode[13] = loadGhostXWest[g_idxPopX(typ1  , tz   ,   3, loadBxWest , byAtNorth , bz        )];
            gNode[15] = loadGhostXWest[g_idxPopX(ty    , tzp1 ,   4, loadBxWest , by        , bzAtFront )];
#endif
        }else if (tx == (BLOCK_NX - 1)){ //e
            gNode[ 2] = loadGhostXEast[g_idxPopX(ty    , tz   ,   0, loadBxEast , by        , bz        )];
#ifdef D3G19
            gNode[ 8] = loadGhostXEast[g_idxPopX(typ1  , tz   ,   1, loadBxEast , byAtNorth , bz        )];
            gNode[10] = loadGhostXEast[g_idxPopX(ty    , tzp1 ,   2, loadBxEast , by        , bzAtFront )];
            gNode[14] = loadGhostXEast[g_idxPopX(tym1  , tz   ,   3, loadBxEast , byAtSouth , bz        )];
            gNode[16] = loadGhostXEast[g_idxPopX(ty    , tzm1 ,   4, loadBxEast , by        , bzAtBack  )];
#endif
        }

        if (ty == 0){ //s
            gNode[ 3] = loadGhostYSouth[g_idxPopY(tx    , tz   ,   0, bx         , loadBySouth, bz        )];
#ifdef D3G19
            gNode[ 7] = loadGhostYSouth[g_idxPopY(txm1  , tz   ,   1, bxAtWest   , loadBySouth, bz        )];
            gNode[11] = loadGhostYSouth[g_idxPopY(tx    , tzm1 ,   2, bx         , loadBySouth, bzAtBack  )];
            gNode[14] = loadGhostYSouth[g_idxPopY(txp1  , tz   ,   3, bxAtEast   , loadBySouth, bz        )];
            gNode[17] = loadGhostYSouth[g_idxPopY(tx    , tzp1 ,   4, bx         , loadBySouth, bzAtFront )];
#endif
        }else if (ty == (BLOCK_NY - 1)){ //n
            gNode[ 4] = loadGhostYNorth[g_idxPopY(tx    , tz   ,   0, bx         , loadByNorth, bz        )];
#ifdef D3G19
            gNode[ 8] = loadGhostYNorth[g_idxPopY(txp1  , tz   ,   1, bxAtEast   , loadByNorth, bz        )];
            gNode[12] = loadGhostYNorth[g_idxPopY(tx    , tzp1 ,   2, bx         , loadByNorth, bzAtFront )];
            gNode[13] = loadGhostYNorth[g_idxPopY(txm1  , tz   ,   3, bxAtWest   , loadByNorth, bz        )];
            gNode[18] = loadGhostYNorth[g_idxPopY(tx    , tzm1 ,   4, bx         , loadByNorth, bzAtBack  )];
#endif
        }

        if (tz == 0){ //b
            gNode[ 5] = loadGhostZBack[g_idxPopZ(tx    , ty   ,   0, bx         , by        , loadBzBack)];
#ifdef D3G19
            gNode[ 9] = loadGhostZBack[g_idxPopZ(txm1  , ty   ,   1, bxAtWest   , by        , loadBzBack)];
            gNode[11] = loadGhostZBack[g_idxPopZ(tx    , tym1 ,   2, bx         , byAtSouth , loadBzBack)];
            gNode[16] = loadGhostZBack[g_idxPopZ(txp1  , ty   ,   3, bxAtEast   , by        , loadBzBack)];
            gNode[18] = loadGhostZBack[g_idxPopZ(tx    , typ1 ,   4, bx         , byAtNorth , loadBzBack)];
#endif
        }else if (tz == (BLOCK_NZ - 1)){ //f
            gNode[ 6] = loadGhostZFront[g_idxPopZ(tx    , ty   ,   0, bx         , by        , loadBzFront)];
#ifdef D3G19
            gNode[10] = loadGhostZFront[g_idxPopZ(txp1  , ty   ,   1, bxAtEast   , by        , loadBzFront)];
            gNode[12] = loadGhostZFront[g_idxPopZ(tx    , typ1 ,   2, bx         , byAtNorth , loadBzFront)];
            gNode[15] = loadGhostZFront[g_idxPopZ(txm1  , ty   ,   3, bxAtWest   , by        , loadBzFront)];
            gNode[17] = loadGhostZFront[g_idxPopZ(tx    , tym1 ,   4, bx         , byAtSouth , loadBzFront)];
#endif
        }

    }
// DEV_REF_START (from dev during merge)
// DEV_REF: #ifdef D3G27
// DEV_REF:         gNode[ 9] = ghostInterface.phi_fGhost.Z_1[g_idxPopZ(txm1, ty, 1, ((tx == 0) ? bxm1 : bx)                , by                                    , bzm1)];
// DEV_REF:         gNode[11] = ghostInterface.phi_fGhost.Z_1[g_idxPopZ(tx, tym1, 2, bx                                     , ((ty == 0) ? bym1 : by)               , bzm1)];
// DEV_REF:         gNode[16] = ghostInterface.phi_fGhost.Z_1[g_idxPopZ(txp1, ty, 3, ((tx == (BLOCK_NX - 1)) ? bxp1 : bx)   , by                                    , bzm1)];
// DEV_REF:         gNode[18] = ghostInterface.phi_fGhost.Z_1[g_idxPopZ(tx, typ1, 4, bx                                     , ((ty == (BLOCK_NY - 1)) ? byp1 : by)  , bzm1)];
// DEV_REF:         gNode[19] = ghostInterface.phi_fGhost.Z_1[g_idxPopZ(txm1, tym1, 5, ((tx == 0) ? bxm1 : bx)              , ((ty == 0) ? bym1 : by)               , bzm1)];
// DEV_REF:         gNode[22] = ghostInterface.phi_fGhost.Z_1[g_idxPopZ(txp1, typ1, 6, ((tx == (BLOCK_NX - 1)) ? bxp1 : bx) , ((ty == (BLOCK_NY - 1)) ? byp1 : by)  , bzm1)];
// DEV_REF:         gNode[23] = ghostInterface.phi_fGhost.Z_1[g_idxPopZ(txm1, typ1, 7, ((tx == 0) ? bxm1 : bx)              , ((ty == (BLOCK_NY - 1)) ? byp1 : by)  , bzm1)];
// DEV_REF:         gNode[25] = ghostInterface.phi_fGhost.Z_1[g_idxPopZ(txp1, tym1, 8, ((tx == (BLOCK_NX - 1)) ? bxp1 : bx) , ((ty == 0) ? bym1 : by)               , bzm1)];
// DEV_REF: #endif
// DEV_REF:     }
// DEV_REF: else if (tz == (BLOCK_NZ - 1))
// DEV_REF:     { //f
// DEV_REF:         gNode[ 6] = ghostInterface.phi_fGhost.Z_0[g_idxPopZ(tx  , ty, 0, bx                                     , by                                    , bzp1)];
// DEV_REF: #ifdef D3G19
// DEV_REF:         gNode[10] = ghostInterface.phi_fGhost.Z_0[g_idxPopZ(txp1, ty, 1, ((tx == (BLOCK_NX - 1)) ? bxp1 : bx)   , by                                    , bzp1)];
// DEV_REF:         gNode[12] = ghostInterface.phi_fGhost.Z_0[g_idxPopZ(tx, typ1, 2, bx                                     , ((ty == (BLOCK_NY - 1)) ? byp1 : by)  , bzp1)];
// DEV_REF:         gNode[15] = ghostInterface.phi_fGhost.Z_0[g_idxPopZ(txm1, ty, 3, ((tx == 0) ? bxm1 : bx)                , by                                    , bzp1)];
// DEV_REF:         gNode[17] = ghostInterface.phi_fGhost.Z_0[g_idxPopZ(tx, tym1, 4, bx                                     , ((ty == 0) ? bym1 : by)               , bzp1)];
// DEV_REF: #endif
// DEV_REF: #ifdef D3G27
// DEV_REF:         gNode[10] = ghostInterface.phi_fGhost.Z_0[g_idxPopZ(txp1, ty, 1, ((tx == (BLOCK_NX - 1)) ? bxp1 : bx)   , by                                    , bzp1)];
// DEV_REF:         gNode[12] = ghostInterface.phi_fGhost.Z_0[g_idxPopZ(tx, typ1, 2, bx                                     , ((ty == (BLOCK_NY - 1)) ? byp1 : by)  , bzp1)];
// DEV_REF:         gNode[15] = ghostInterface.phi_fGhost.Z_0[g_idxPopZ(txm1, ty, 3, ((tx == 0) ? bxm1 : bx)                , by                                    , bzp1)];
// DEV_REF:         gNode[17] = ghostInterface.phi_fGhost.Z_0[g_idxPopZ(tx, tym1, 4, bx                                     , ((ty == 0) ? bym1 : by)               , bzp1)];
// DEV_REF:         gNode[20] = ghostInterface.phi_fGhost.Z_0[g_idxPopZ(txp1, typ1, 5, ((tx == (BLOCK_NX - 1)) ? bxp1 : bx) , ((ty == (BLOCK_NY - 1)) ? byp1 : by)  , bzp1)];
// DEV_REF:         gNode[21] = ghostInterface.phi_fGhost.Z_0[g_idxPopZ(txm1, tym1, 6, ((tx == 0) ? bxm1 : bx)              , ((ty == 0) ? bym1 : by)               , bzp1)];
// DEV_REF:         gNode[24] = ghostInterface.phi_fGhost.Z_0[g_idxPopZ(txp1, tym1, 7, ((tx == (BLOCK_NX - 1)) ? bxp1 : bx) , ((ty == 0) ? bym1 : by)               , bzp1)];
// DEV_REF:         gNode[26] = ghostInterface.phi_fGhost.Z_0[g_idxPopZ(txm1, typ1, 8, ((tx == 0) ? bxm1 : bx)              , ((ty == (BLOCK_NY - 1)) ? byp1 : by)  , bzp1)];
// DEV_REF: #endif
// DEV_REF:     }
// DEV_REF:
// DEV_REF_END
