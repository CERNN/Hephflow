// AA scheme: read from ghost, parity-aware block indices
const dfloat* loadGhostXWest  = ghostInterfacePhi.X_1;
const dfloat* loadGhostXEast  = ghostInterfacePhi.X_0;
const dfloat* loadGhostYSouth = ghostInterfacePhi.Y_1;
const dfloat* loadGhostYNorth = ghostInterfacePhi.Y_0;
const dfloat* loadGhostZBack  = ghostInterfacePhi.Z_1;
const dfloat* loadGhostZFront = ghostInterfacePhi.Z_0;

const int loadBxWest  = (stepParity ? bx : bxm1);
const int loadBxEast  = (stepParity ? bx : bxp1);
const int loadBySouth = (stepParity ? by : bym1);
const int loadByNorth = (stepParity ? by : byp1);
const int loadBzBack  = (stepParity ? bz : bzm1);
const int loadBzFront = (stepParity ? bz : bzp1);

const int top = NUM_BLOCK_Z_LOCAL - 1;

const int bxAtWest  = (tx == 0)              ? bxm1 : bx;
const int bxAtEast  = (tx == (BLOCK_NX - 1)) ? bxp1 : bx;
const int byAtSouth = (ty == 0)              ? bym1 : by;
const int byAtNorth = (ty == (BLOCK_NY - 1)) ? byp1 : by;
const int bzAtBack  = (tz == 0)              ? bzm1 : bz;
const int bzAtFront = (tz == (BLOCK_NZ - 1)) ? bzp1 : bz;
/* DEV_REF_START (from dev during merge)
// Load from global memory - face neighbors for D3G7/D3G19, edge neighbors only for D3G19
if (tx == 0) { //w
        gNode[ 1] = ghostInterface.phi_fGhost.X_1[g_idxPopX(ty  , tz, 0, bxm1, by                                       , bz)];
#ifdef D3G19
        gNode[ 7] = ghostInterface.phi_fGhost.X_1[g_idxPopX(tym1, tz, 1, bxm1, ((ty == 0) ? bym1 : by)                  , bz)];
        gNode[ 9] = ghostInterface.phi_fGhost.X_1[g_idxPopX(ty, tzm1, 2, bxm1, by                                       , ((tz == 0) ? bzm1 : bz))];
        gNode[13] = ghostInterface.phi_fGhost.X_1[g_idxPopX(typ1, tz, 3, bxm1, ((ty == (BLOCK_NY - 1)) ? byp1 : by)     , bz)];
        gNode[15] = ghostInterface.phi_fGhost.X_1[g_idxPopX(ty, tzp1, 4, bxm1, by                                       , ((tz == (BLOCK_NZ - 1)) ? bzp1 : bz))];
#endif
#ifdef D3G27
        gNode[ 7] = ghostInterface.phi_fGhost.X_1[g_idxPopX(tym1, tz, 1, bxm1, ((ty == 0) ? bym1 : by)                  , bz)];
        gNode[ 9] = ghostInterface.phi_fGhost.X_1[g_idxPopX(ty, tzm1, 2, bxm1, by                                       , ((tz == 0) ? bzm1 : bz))];
        gNode[13] = ghostInterface.phi_fGhost.X_1[g_idxPopX(typ1, tz, 3, bxm1, ((ty == (BLOCK_NY - 1)) ? byp1 : by)     , bz)];
        gNode[15] = ghostInterface.phi_fGhost.X_1[g_idxPopX(ty, tzp1, 4, bxm1, by                                       , ((tz == (BLOCK_NZ - 1)) ? bzp1 : bz))];
        gNode[19] = ghostInterface.phi_fGhost.X_1[g_idxPopX(tym1, tzm1, 5, bxm1, ((ty == 0) ? bym1 : by)                , ((tz == 0) ? bzm1 : bz))];
        gNode[21] = ghostInterface.phi_fGhost.X_1[g_idxPopX(tym1, tzp1, 6, bxm1, ((ty == 0) ? bym1 : by)                , ((tz == (BLOCK_NZ - 1)) ? bzp1 : bz))];
        gNode[23] = ghostInterface.phi_fGhost.X_1[g_idxPopX(typ1, tzm1, 7, bxm1, ((ty == (BLOCK_NY - 1)) ? byp1 : by)   , ((tz == 0) ? bzm1 : bz))];
        gNode[26] = ghostInterface.phi_fGhost.X_1[g_idxPopX(typ1, tzp1, 8, bxm1, ((ty == (BLOCK_NY - 1)) ? byp1 : by)   , ((tz == (BLOCK_NZ - 1)) ? bzp1 : bz))];
#endif
}
else if (tx == (BLOCK_NX - 1))
    { //e
        gNode[ 2] = ghostInterface.phi_fGhost.X_0[g_idxPopX(ty  , tz, 0, bxp1, by                                       , bz)];
#ifdef D3G19
        gNode[ 8] = ghostInterface.phi_fGhost.X_0[g_idxPopX(typ1, tz, 1, bxp1, ((ty == (BLOCK_NY - 1)) ? byp1 : by)     , bz)];
        gNode[10] = ghostInterface.phi_fGhost.X_0[g_idxPopX(ty, tzp1, 2, bxp1, by                                       , ((tz == (BLOCK_NZ - 1)) ? bzp1 : bz))];
        gNode[14] = ghostInterface.phi_fGhost.X_0[g_idxPopX(tym1, tz, 3, bxp1, ((ty == 0) ? bym1 : by)                  , bz)];
        gNode[16] = ghostInterface.phi_fGhost.X_0[g_idxPopX(ty, tzm1, 4, bxp1, by                                       , ((tz == 0) ? bzm1 : bz))];
#endif
#ifdef D3G27
        gNode[ 8] = ghostInterface.phi_fGhost.X_0[g_idxPopX(typ1, tz, 1, bxp1, ((ty == (BLOCK_NY - 1)) ? byp1 : by)     , bz)];
        gNode[10] = ghostInterface.phi_fGhost.X_0[g_idxPopX(ty, tzp1, 2, bxp1, by                                       , ((tz == (BLOCK_NZ - 1)) ? bzp1 : bz))];
        gNode[14] = ghostInterface.phi_fGhost.X_0[g_idxPopX(tym1, tz, 3, bxp1, ((ty == 0) ? bym1 : by)                  , bz)];
        gNode[16] = ghostInterface.phi_fGhost.X_0[g_idxPopX(ty, tzm1, 4, bxp1, by                                       , ((tz == 0) ? bzm1 : bz))];
        gNode[20] = ghostInterface.phi_fGhost.X_0[g_idxPopX(typ1, tzp1, 5, bxp1, ((ty == (BLOCK_NY - 1)) ? byp1 : by)   , ((tz == (BLOCK_NZ - 1)) ? bzp1 : bz))];
        gNode[22] = ghostInterface.phi_fGhost.X_0[g_idxPopX(typ1, tzm1, 6, bxp1, ((ty == (BLOCK_NY - 1)) ? byp1 : by)   , ((tz == 0) ? bzm1 : bz))];
        gNode[24] = ghostInterface.phi_fGhost.X_0[g_idxPopX(tym1, tzp1, 7, bxp1, ((ty == 0) ? bym1 : by)                , ((tz == (BLOCK_NZ - 1)) ? bzp1 : bz))];
        gNode[25] = ghostInterface.phi_fGhost.X_0[g_idxPopX(tym1, tzm1, 8, bxp1, ((ty == 0) ? bym1 : by)                , ((tz == 0) ? bzm1 : bz))];
#endif
    }

if (ty == 0)
    { //s
        gNode[ 3] = ghostInterface.phi_fGhost.Y_1[g_idxPopY(tx  , tz, 0, bx                                     , bym1, bz)];
#ifdef D3G19
        gNode[ 7] = ghostInterface.phi_fGhost.Y_1[g_idxPopY(txm1, tz, 1, ((tx == 0) ? bxm1 : bx)                , bym1, bz)];
        gNode[11] = ghostInterface.phi_fGhost.Y_1[g_idxPopY(tx, tzm1, 2, bx                                     , bym1, ((tz == 0) ? bzm1 : bz))];
        gNode[14] = ghostInterface.phi_fGhost.Y_1[g_idxPopY(txp1, tz, 3, ((tx == (BLOCK_NX - 1)) ? bxp1 : bx)   , bym1, bz)];
        gNode[17] = ghostInterface.phi_fGhost.Y_1[g_idxPopY(tx, tzp1, 4, bx                                     , bym1, ((tz == (BLOCK_NZ - 1)) ? bzp1 : bz))];
#endif
#ifdef D3G27
        gNode[ 7] = ghostInterface.phi_fGhost.Y_1[g_idxPopY(txm1, tz, 1, ((tx == 0) ? bxm1 : bx)                , bym1, bz)];
        gNode[11] = ghostInterface.phi_fGhost.Y_1[g_idxPopY(tx, tzm1, 2, bx                                     , bym1, ((tz == 0) ? bzm1 : bz))];
        gNode[14] = ghostInterface.phi_fGhost.Y_1[g_idxPopY(txp1, tz, 3, ((tx == (BLOCK_NX - 1)) ? bxp1 : bx)   , bym1, bz)];
        gNode[17] = ghostInterface.phi_fGhost.Y_1[g_idxPopY(tx, tzp1, 4, bx                                     , bym1, ((tz == (BLOCK_NZ - 1)) ? bzp1 : bz))];
        gNode[19] = ghostInterface.phi_fGhost.Y_1[g_idxPopY(txm1, tzm1, 5, ((tx == 0) ? bxm1 : bx)              , bym1, ((tz == 0) ? bzm1 : bz))];
        gNode[21] = ghostInterface.phi_fGhost.Y_1[g_idxPopY(txm1, tzp1, 6, ((tx == 0) ? bxm1 : bx)              , bym1, ((tz == (BLOCK_NZ - 1)) ? bzp1 : bz))];
        gNode[24] = ghostInterface.phi_fGhost.Y_1[g_idxPopY(txp1, tzp1, 7, ((tx == (BLOCK_NX - 1)) ? bxp1 : bx) , bym1, ((tz == (BLOCK_NZ - 1)) ? bzp1 : bz))];
        gNode[25] = ghostInterface.phi_fGhost.Y_1[g_idxPopY(txp1, tzm1, 8, ((tx == (BLOCK_NX - 1)) ? bxp1 : bx) , bym1, ((tz == 0) ? bzm1 : bz))];
#endif
    }
else if (ty == (BLOCK_NY - 1))
    { //n
        gNode[ 4] = ghostInterface.phi_fGhost.Y_0[g_idxPopY(tx  , tz, 0, bx                                     , byp1, bz)];
#ifdef D3G19
        gNode[ 8] = ghostInterface.phi_fGhost.Y_0[g_idxPopY(txp1, tz, 1, ((tx == (BLOCK_NX - 1)) ? bxp1 : bx)   , byp1, bz)];
        gNode[12] = ghostInterface.phi_fGhost.Y_0[g_idxPopY(tx, tzp1, 2, bx                                     , byp1, ((tz == (BLOCK_NZ - 1)) ? bzp1 : bz))];
        gNode[13] = ghostInterface.phi_fGhost.Y_0[g_idxPopY(txm1, tz, 3, ((tx == 0) ? bxm1 : bx)                , byp1, bz)];
        gNode[18] = ghostInterface.phi_fGhost.Y_0[g_idxPopY(tx, tzm1, 4, bx                                     , byp1, ((tz == 0) ? bzm1 : bz))];
#endif
#ifdef D3G27
        gNode[ 8] = ghostInterface.phi_fGhost.Y_0[g_idxPopY(txp1, tz, 1, ((tx == (BLOCK_NX - 1)) ? bxp1 : bx)   , byp1, bz)];
        gNode[12] = ghostInterface.phi_fGhost.Y_0[g_idxPopY(tx, tzp1, 2, bx                                     , byp1, ((tz == (BLOCK_NZ - 1)) ? bzp1 : bz))];
        gNode[13] = ghostInterface.phi_fGhost.Y_0[g_idxPopY(txm1, tz, 3, ((tx == 0) ? bxm1 : bx)                , byp1, bz)];
        gNode[18] = ghostInterface.phi_fGhost.Y_0[g_idxPopY(tx, tzm1, 4, bx                                     , byp1, ((tz == 0) ? bzm1 : bz))];
        gNode[20] = ghostInterface.phi_fGhost.Y_0[g_idxPopY(txp1, tzp1, 5, ((tx == (BLOCK_NX - 1)) ? bxp1 : bx) , byp1, ((tz == (BLOCK_NZ - 1)) ? bzp1 : bz))];
        gNode[22] = ghostInterface.phi_fGhost.Y_0[g_idxPopY(txp1, tzm1, 6, ((tx == (BLOCK_NX - 1)) ? bxp1 : bx) , byp1, ((tz == 0) ? bzm1 : bz))];
        gNode[23] = ghostInterface.phi_fGhost.Y_0[g_idxPopY(txm1, tzm1, 7, ((tx == 0) ? bxm1 : bx)              , byp1, ((tz == 0) ? bzm1 : bz))];
        gNode[26] = ghostInterface.phi_fGhost.Y_0[g_idxPopY(txm1, tzp1, 8, ((tx == 0) ? bxm1 : bx)              , byp1, ((tz == (BLOCK_NZ - 1)) ? bzp1 : bz))];
#endif
    }

DEV_REF_END */

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
#ifdef D3G27
            gNode[ 7] = loadGhostXEast[g_idxPopX(ty, tz, 1, loadBxWest, by, bz)];
            gNode[ 9] = loadGhostXEast[g_idxPopX(ty, tz, 2, loadBxWest, by, bz)];
            gNode[13] = loadGhostXEast[g_idxPopX(ty, tz, 3, loadBxWest, by, bz)];
            gNode[15] = loadGhostXEast[g_idxPopX(ty, tz, 4, loadBxWest, by, bz)];
            gNode[19] = loadGhostXEast[g_idxPopX(ty, tz, 5, loadBxWest, by, bz)];
            gNode[21] = loadGhostXEast[g_idxPopX(ty, tz, 6, loadBxWest, by, bz)];
            gNode[23] = loadGhostXEast[g_idxPopX(ty, tz, 7, loadBxWest, by, bz)];
            gNode[26] = loadGhostXEast[g_idxPopX(ty, tz, 8, loadBxWest, by, bz)];
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
#ifdef D3G27
            gNode[ 8] = loadGhostXWest[g_idxPopX(ty, tz, 1, loadBxEast, by, bz)];
            gNode[10] = loadGhostXWest[g_idxPopX(ty, tz, 2, loadBxEast, by, bz)];
            gNode[14] = loadGhostXWest[g_idxPopX(ty, tz, 3, loadBxEast, by, bz)];
            gNode[16] = loadGhostXWest[g_idxPopX(ty, tz, 4, loadBxEast, by, bz)];
            gNode[20] = loadGhostXWest[g_idxPopX(ty, tz, 5, loadBxEast, by, bz)];
            gNode[22] = loadGhostXWest[g_idxPopX(ty, tz, 6, loadBxEast, by, bz)];
            gNode[24] = loadGhostXWest[g_idxPopX(ty, tz, 7, loadBxEast, by, bz)];
            gNode[25] = loadGhostXWest[g_idxPopX(ty, tz, 8, loadBxEast, by, bz)];
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
#ifdef D3G27
            gNode[ 7] = loadGhostYNorth[g_idxPopY(tx, tz, 1, bx, loadBySouth, bz)];
            gNode[11] = loadGhostYNorth[g_idxPopY(tx, tz, 2, bx, loadBySouth, bz)];
            gNode[14] = loadGhostYNorth[g_idxPopY(tx, tz, 3, bx, loadBySouth, bz)];
            gNode[17] = loadGhostYNorth[g_idxPopY(tx, tz, 4, bx, loadBySouth, bz)];
            gNode[20] = loadGhostYNorth[g_idxPopY(tx, tz, 5, bx, loadBySouth, bz)];
            gNode[22] = loadGhostYNorth[g_idxPopY(tx, tz, 6, bx, loadBySouth, bz)];
            gNode[23] = loadGhostYNorth[g_idxPopY(tx, tz, 7, bx, loadBySouth, bz)];
            gNode[26] = loadGhostYNorth[g_idxPopY(tx, tz, 8, bx, loadBySouth, bz)];
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
#ifdef D3G27
            gNode[ 8] = loadGhostYSouth[g_idxPopY(tx, tz, 1, bx, loadByNorth, bz)];
            gNode[12] = loadGhostYSouth[g_idxPopY(tx, tz, 2, bx, loadByNorth, bz)];
            gNode[13] = loadGhostYSouth[g_idxPopY(tx, tz, 3, bx, loadByNorth, bz)];
            gNode[18] = loadGhostYSouth[g_idxPopY(tx, tz, 4, bx, loadByNorth, bz)];
            gNode[19] = loadGhostYSouth[g_idxPopY(tx, tz, 5, bx, loadByNorth, bz)];
            gNode[21] = loadGhostYSouth[g_idxPopY(tx, tz, 6, bx, loadByNorth, bz)];
            gNode[24] = loadGhostYSouth[g_idxPopY(tx, tz, 7, bx, loadByNorth, bz)];
            gNode[25] = loadGhostYSouth[g_idxPopY(tx, tz, 8, bx, loadByNorth, bz)];
#endif
        }

        if (tz == 0)
        { //b
                if  (bz == 0) {
                        gNode[ 5] = ghostInterfacePhi.auxZ_1[g_idxPopZ(tx, ty, 0, bx, by, 0)];
#ifdef D3G19
                        gNode[ 9] = ghostInterfacePhi.auxZ_1[g_idxPopZ(tx, ty, 1, bx, by, 0)];
                        gNode[11] = ghostInterfacePhi.auxZ_1[g_idxPopZ(tx, ty, 2, bx, by, 0)];
                        gNode[16] = ghostInterfacePhi.auxZ_1[g_idxPopZ(tx, ty, 3, bx, by, 0)];
                        gNode[18] = ghostInterfacePhi.auxZ_1[g_idxPopZ(tx, ty, 4, bx, by, 0)];
#endif
#ifdef D3G27
                        gNode[ 9] = ghostInterfacePhi.auxZ_1[g_idxPopZ(tx, ty, 1, bx, by, 0)];
                        gNode[11] = ghostInterfacePhi.auxZ_1[g_idxPopZ(tx, ty, 2, bx, by, 0)];
                        gNode[16] = ghostInterfacePhi.auxZ_1[g_idxPopZ(tx, ty, 3, bx, by, 0)];
                        gNode[18] = ghostInterfacePhi.auxZ_1[g_idxPopZ(tx, ty, 4, bx, by, 0)];
                        gNode[19] = ghostInterfacePhi.auxZ_1[g_idxPopZ(tx, ty, 5, bx, by, 0)];
                        gNode[22] = ghostInterfacePhi.auxZ_1[g_idxPopZ(tx, ty, 6, bx, by, 0)];
                        gNode[23] = ghostInterfacePhi.auxZ_1[g_idxPopZ(tx, ty, 7, bx, by, 0)];
                        gNode[25] = ghostInterfacePhi.auxZ_1[g_idxPopZ(tx, ty, 8, bx, by, 0)];
#endif
                }else{
                        gNode[ 5] = loadGhostZFront[g_idxPopZ(tx, ty, 0, bx, by, loadBzBack)];
#ifdef D3G19
                        gNode[ 9] = loadGhostZFront[g_idxPopZ(tx, ty, 1, bx, by, loadBzBack)];
                        gNode[11] = loadGhostZFront[g_idxPopZ(tx, ty, 2, bx, by, loadBzBack)];
                        gNode[16] = loadGhostZFront[g_idxPopZ(tx, ty, 3, bx, by, loadBzBack)];
                        gNode[18] = loadGhostZFront[g_idxPopZ(tx, ty, 4, bx, by, loadBzBack)];
#endif
#ifdef D3G27
                        gNode[ 9] = loadGhostZFront[g_idxPopZ(tx, ty, 1, bx, by, loadBzBack)];
                        gNode[11] = loadGhostZFront[g_idxPopZ(tx, ty, 2, bx, by, loadBzBack)];
                        gNode[16] = loadGhostZFront[g_idxPopZ(tx, ty, 3, bx, by, loadBzBack)];
                        gNode[18] = loadGhostZFront[g_idxPopZ(tx, ty, 4, bx, by, loadBzBack)];
                        gNode[19] = loadGhostZFront[g_idxPopZ(tx, ty, 5, bx, by, loadBzBack)];
                        gNode[22] = loadGhostZFront[g_idxPopZ(tx, ty, 6, bx, by, loadBzBack)];
                        gNode[23] = loadGhostZFront[g_idxPopZ(tx, ty, 7, bx, by, loadBzBack)];
                        gNode[25] = loadGhostZFront[g_idxPopZ(tx, ty, 8, bx, by, loadBzBack)];
#endif    
                }
        }
        else if (tz == (BLOCK_NZ - 1))
        { //f
                if  (bz == top) {
                        gNode[ 6] = ghostInterfacePhi.auxZ_0[g_idxPopZ(tx, ty, 0, bx, by, 0)];
#ifdef D3G19
                        gNode[10] = ghostInterfacePhi.auxZ_0[g_idxPopZ(tx, ty, 1, bx, by, 0)];
                        gNode[12] = ghostInterfacePhi.auxZ_0[g_idxPopZ(tx, ty, 2, bx, by, 0)];
                        gNode[15] = ghostInterfacePhi.auxZ_0[g_idxPopZ(tx, ty, 3, bx, by, 0)];
                        gNode[17] = ghostInterfacePhi.auxZ_0[g_idxPopZ(tx, ty, 4, bx, by, 0)];
#endif
#ifdef D3G27
                        gNode[10] = ghostInterfacePhi.auxZ_0[g_idxPopZ(tx, ty, 1, bx, by, 0)];
                        gNode[12] = ghostInterfacePhi.auxZ_0[g_idxPopZ(tx, ty, 2, bx, by, 0)];
                        gNode[15] = ghostInterfacePhi.auxZ_0[g_idxPopZ(tx, ty, 3, bx, by, 0)];
                        gNode[17] = ghostInterfacePhi.auxZ_0[g_idxPopZ(tx, ty, 4, bx, by, 0)];
                        gNode[20] = ghostInterfacePhi.auxZ_0[g_idxPopZ(tx, ty, 5, bx, by, 0)];
                        gNode[21] = ghostInterfacePhi.auxZ_0[g_idxPopZ(tx, ty, 6, bx, by, 0)];
                        gNode[24] = ghostInterfacePhi.auxZ_0[g_idxPopZ(tx, ty, 7, bx, by, 0)];
                        gNode[26] = ghostInterfacePhi.auxZ_0[g_idxPopZ(tx, ty, 8, bx, by, 0)];
#endif
                }else{
                        gNode[ 6] = loadGhostZBack[g_idxPopZ(tx, ty, 0, bx, by, loadBzFront)];
#ifdef D3G19
                        gNode[10] = loadGhostZBack[g_idxPopZ(tx, ty, 1, bx, by, loadBzFront)];
                        gNode[12] = loadGhostZBack[g_idxPopZ(tx, ty, 2, bx, by, loadBzFront)];
                        gNode[15] = loadGhostZBack[g_idxPopZ(tx, ty, 3, bx, by, loadBzFront)];
                        gNode[17] = loadGhostZBack[g_idxPopZ(tx, ty, 4, bx, by, loadBzFront)];
#endif
#ifdef D3G27
                        gNode[10] = loadGhostZBack[g_idxPopZ(tx, ty, 1, bx, by, loadBzFront)];
                        gNode[12] = loadGhostZBack[g_idxPopZ(tx, ty, 2, bx, by, loadBzFront)];
                        gNode[15] = loadGhostZBack[g_idxPopZ(tx, ty, 3, bx, by, loadBzFront)];
                        gNode[17] = loadGhostZBack[g_idxPopZ(tx, ty, 4, bx, by, loadBzFront)];
                        gNode[20] = loadGhostZBack[g_idxPopZ(tx, ty, 5, bx, by, loadBzFront)];
                        gNode[21] = loadGhostZBack[g_idxPopZ(tx, ty, 6, bx, by, loadBzFront)];
                        gNode[24] = loadGhostZBack[g_idxPopZ(tx, ty, 7, bx, by, loadBzFront)];
                        gNode[26] = loadGhostZBack[g_idxPopZ(tx, ty, 8, bx, by, loadBzFront)];
#endif           
                }
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
#ifdef D3G27
            gNode[ 7] = loadGhostXWest[g_idxPopX(tym1  , tz   ,   1, loadBxWest , byAtSouth , bz        )];
            gNode[ 9] = loadGhostXWest[g_idxPopX(ty    , tzm1 ,   2, loadBxWest , by        , bzAtBack  )];
            gNode[13] = loadGhostXWest[g_idxPopX(typ1  , tz   ,   3, loadBxWest , byAtNorth , bz        )];
            gNode[15] = loadGhostXWest[g_idxPopX(ty    , tzp1 ,   4, loadBxWest , by        , bzAtFront )];
            gNode[19] = loadGhostXWest[g_idxPopX(tym1  , tzm1 ,   5, loadBxWest , byAtSouth , bzAtBack  )];
            gNode[21] = loadGhostXWest[g_idxPopX(tym1  , tzp1 ,   6, loadBxWest , byAtSouth , bzAtFront )];
            gNode[23] = loadGhostXWest[g_idxPopX(typ1  , tzm1 ,   7, loadBxWest , byAtNorth , bzAtBack  )];
            gNode[26] = loadGhostXWest[g_idxPopX(typ1  , tzp1 ,   8, loadBxWest , byAtNorth , bzAtFront )];
#endif
        }else if (tx == (BLOCK_NX - 1)){ //e
            gNode[ 2] = loadGhostXEast[g_idxPopX(ty    , tz   ,   0, loadBxEast , by        , bz        )];
#ifdef D3G19
            gNode[ 8] = loadGhostXEast[g_idxPopX(typ1  , tz   ,   1, loadBxEast , byAtNorth , bz        )];
            gNode[10] = loadGhostXEast[g_idxPopX(ty    , tzp1 ,   2, loadBxEast , by        , bzAtFront )];
            gNode[14] = loadGhostXEast[g_idxPopX(tym1  , tz   ,   3, loadBxEast , byAtSouth , bz        )];
            gNode[16] = loadGhostXEast[g_idxPopX(ty    , tzm1 ,   4, loadBxEast , by        , bzAtBack  )];
#endif
#ifdef D3G27
            gNode[ 8] = loadGhostXEast[g_idxPopX(typ1  , tz   ,   1, loadBxEast , byAtNorth , bz        )];
            gNode[10] = loadGhostXEast[g_idxPopX(ty    , tzp1 ,   2, loadBxEast , by        , bzAtFront )];
            gNode[14] = loadGhostXEast[g_idxPopX(tym1  , tz   ,   3, loadBxEast , byAtSouth , bz        )];
            gNode[16] = loadGhostXEast[g_idxPopX(ty    , tzm1 ,   4, loadBxEast , by        , bzAtBack  )];
            gNode[20] = loadGhostXEast[g_idxPopX(typ1  , tzp1 ,   5, loadBxEast , byAtNorth , bzAtFront )];
            gNode[22] = loadGhostXEast[g_idxPopX(typ1  , tzm1 ,   6, loadBxEast , byAtNorth , bzAtBack  )];
            gNode[24] = loadGhostXEast[g_idxPopX(tym1  , tzp1 ,   7, loadBxEast , byAtSouth , bzAtFront )];
            gNode[25] = loadGhostXEast[g_idxPopX(tym1  , tzm1 ,   8, loadBxEast , byAtSouth , bzAtBack  )];
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
#ifdef D3G27
            gNode[ 7] = loadGhostYSouth[g_idxPopY(txm1  , tz   ,   1, bxAtWest   , loadBySouth, bz        )];
            gNode[11] = loadGhostYSouth[g_idxPopY(tx    , tzm1 ,   2, bx         , loadBySouth, bzAtBack  )];
            gNode[14] = loadGhostYSouth[g_idxPopY(txp1  , tz   ,   3, bxAtEast   , loadBySouth, bz        )];
            gNode[17] = loadGhostYSouth[g_idxPopY(tx    , tzp1 ,   4, bx         , loadBySouth, bzAtFront )];
            gNode[19] = loadGhostYSouth[g_idxPopY(txm1  , tzm1 ,   5, bxAtWest   , loadBySouth, bzAtBack  )];
            gNode[21] = loadGhostYSouth[g_idxPopY(txm1  , tzp1 ,   6, bxAtWest   , loadBySouth, bzAtFront )];
            gNode[24] = loadGhostYSouth[g_idxPopY(txp1  , tzp1 ,   7, bxAtEast   , loadBySouth, bzAtFront )];
            gNode[25] = loadGhostYSouth[g_idxPopY(txp1  , tzm1 ,   8, bxAtEast   , loadBySouth, bzAtBack  )];
#endif
        }else if (ty == (BLOCK_NY - 1)){ //n
            gNode[ 4] = loadGhostYNorth[g_idxPopY(tx    , tz   ,   0, bx         , loadByNorth, bz        )];
#ifdef D3G19
            gNode[ 8] = loadGhostYNorth[g_idxPopY(txp1  , tz   ,   1, bxAtEast   , loadByNorth, bz        )];
            gNode[12] = loadGhostYNorth[g_idxPopY(tx    , tzp1 ,   2, bx         , loadByNorth, bzAtFront )];
            gNode[13] = loadGhostYNorth[g_idxPopY(txm1  , tz   ,   3, bxAtWest   , loadByNorth, bz        )];
            gNode[18] = loadGhostYNorth[g_idxPopY(tx    , tzm1 ,   4, bx         , loadByNorth, bzAtBack  )];
#endif
#ifdef D3G27
            gNode[ 8] = loadGhostYNorth[g_idxPopY(txp1  , tz   ,   1, bxAtEast   , loadByNorth, bz        )];
            gNode[12] = loadGhostYNorth[g_idxPopY(tx    , tzp1 ,   2, bx         , loadByNorth, bzAtFront )];
            gNode[13] = loadGhostYNorth[g_idxPopY(txm1  , tz   ,   3, bxAtWest   , loadByNorth, bz        )];
            gNode[18] = loadGhostYNorth[g_idxPopY(tx    , tzm1 ,   4, bx         , loadByNorth, bzAtBack  )];
            gNode[20] = loadGhostYNorth[g_idxPopY(txp1  , tzp1 ,   5, bxAtEast   , loadByNorth, bzAtFront )];
            gNode[22] = loadGhostYNorth[g_idxPopY(txp1  , tzm1 ,   6, bxAtEast   , loadByNorth, bzAtBack  )];
            gNode[23] = loadGhostYNorth[g_idxPopY(txm1  , tzm1 ,   7, bxAtWest   , loadByNorth, bzAtBack  )];
            gNode[26] = loadGhostYNorth[g_idxPopY(txm1  , tzp1 ,   8, bxAtWest   , loadByNorth, bzAtFront )];
#endif
        }

        if (tz == 0){ //b
                if (bz == 0) {
                        gNode[ 5] = ghostInterfacePhi.auxZ_1[g_idxPopZ(tx    , ty   ,   0, bx         , by        , 0)];
#ifdef D3G19
                        gNode[ 9] = ghostInterfacePhi.auxZ_1[g_idxPopZ(txm1  , ty   ,   1, bxAtWest   , by        , 0)];
                        gNode[11] = ghostInterfacePhi.auxZ_1[g_idxPopZ(tx    , tym1 ,   2, bx         , byAtSouth , 0)];
                        gNode[16] = ghostInterfacePhi.auxZ_1[g_idxPopZ(txp1  , ty   ,   3, bxAtEast   , by        , 0)];
                        gNode[18] = ghostInterfacePhi.auxZ_1[g_idxPopZ(tx    , typ1 ,   4, bx         , byAtNorth , 0)];
#endif
#ifdef D3G27
                        gNode[ 9] = ghostInterfacePhi.auxZ_1[g_idxPopZ(txm1  , ty   ,   1, bxAtWest   , by        , 0)];
                        gNode[11] = ghostInterfacePhi.auxZ_1[g_idxPopZ(tx    , tym1 ,   2, bx         , byAtSouth , 0)];
                        gNode[16] = ghostInterfacePhi.auxZ_1[g_idxPopZ(txp1  , ty   ,   3, bxAtEast   , by        , 0)];
                        gNode[18] = ghostInterfacePhi.auxZ_1[g_idxPopZ(tx    , typ1 ,   4, bx         , byAtNorth , 0)];
                        gNode[19] = ghostInterfacePhi.auxZ_1[g_idxPopZ(txm1  , tym1 ,   5, bxAtWest   , byAtSouth , 0)];
                        gNode[22] = ghostInterfacePhi.auxZ_1[g_idxPopZ(txp1  , typ1 ,   6, bxAtEast   , byAtNorth , 0)];
                        gNode[23] = ghostInterfacePhi.auxZ_1[g_idxPopZ(txm1  , typ1 ,   7, bxAtWest   , byAtNorth , 0)];
                        gNode[25] = ghostInterfacePhi.auxZ_1[g_idxPopZ(txp1  , tym1 ,   8, bxAtEast   , byAtSouth , 0)];
#endif
                }else{
                        gNode[ 5] = loadGhostZBack[g_idxPopZ(tx    , ty   ,   0, bx         , by        , loadBzBack)];
#ifdef D3G19
                        gNode[ 9] = loadGhostZBack[g_idxPopZ(txm1  , ty   ,   1, bxAtWest   , by        , loadBzBack)];
                        gNode[11] = loadGhostZBack[g_idxPopZ(tx    , tym1 ,   2, bx         , byAtSouth , loadBzBack)];
                        gNode[16] = loadGhostZBack[g_idxPopZ(txp1  , ty   ,   3, bxAtEast   , by        , loadBzBack)];
                        gNode[18] = loadGhostZBack[g_idxPopZ(tx    , typ1 ,   4, bx         , byAtNorth , loadBzBack)];
#endif
#ifdef D3G27
                        gNode[ 9] = loadGhostZBack[g_idxPopZ(txm1  , ty   ,   1, bxAtWest   , by        , loadBzBack)];
                        gNode[11] = loadGhostZBack[g_idxPopZ(tx    , tym1 ,   2, bx         , byAtSouth , loadBzBack)];
                        gNode[16] = loadGhostZBack[g_idxPopZ(txp1  , ty   ,   3, bxAtEast   , by        , loadBzBack)];
                        gNode[18] = loadGhostZBack[g_idxPopZ(tx    , typ1 ,   4, bx         , byAtNorth , loadBzBack)];
                        gNode[19] = loadGhostZBack[g_idxPopZ(txm1  , tym1 ,   5, bxAtWest   , byAtSouth , loadBzBack)];
                        gNode[22] = loadGhostZBack[g_idxPopZ(txp1  , typ1 ,   6, bxAtEast   , byAtNorth , loadBzBack)];
                        gNode[23] = loadGhostZBack[g_idxPopZ(txm1  , typ1 ,   7, bxAtWest   , byAtNorth , loadBzBack)];
                        gNode[25] = loadGhostZBack[g_idxPopZ(txp1  , tym1 ,   8, bxAtEast   , byAtSouth , loadBzBack)];
#endif     
                }
        }else if (tz == (BLOCK_NZ - 1)){ //f
                if (bz == top) {
                        gNode[ 6] = ghostInterfacePhi.auxZ_0[g_idxPopZ(tx    , ty   ,   0, bx         , by        , 0)];
#ifdef D3G19
                        gNode[10] = ghostInterfacePhi.auxZ_0[g_idxPopZ(txp1  , ty   ,   1, bxAtEast   , by        , 0)];
                        gNode[12] = ghostInterfacePhi.auxZ_0[g_idxPopZ(tx    , typ1 ,   2, bx         , byAtNorth , 0)];
                        gNode[15] = ghostInterfacePhi.auxZ_0[g_idxPopZ(txm1  , ty   ,   3, bxAtWest   , by        , 0)];
                        gNode[17] = ghostInterfacePhi.auxZ_0[g_idxPopZ(tx    , tym1 ,   4, bx         , byAtSouth , 0)];
#endif
#ifdef D3G27
                        gNode[10] = ghostInterfacePhi.auxZ_0[g_idxPopZ(txp1  , ty   ,   1, bxAtEast   , by        , 0)];
                        gNode[12] = ghostInterfacePhi.auxZ_0[g_idxPopZ(tx    , typ1 ,   2, bx         , byAtNorth , 0)];
                        gNode[15] = ghostInterfacePhi.auxZ_0[g_idxPopZ(txm1  , ty   ,   3, bxAtWest   , by        , 0)];
                        gNode[17] = ghostInterfacePhi.auxZ_0[g_idxPopZ(tx    , tym1 ,   4, bx         , byAtSouth , 0)];
                        gNode[20] = ghostInterfacePhi.auxZ_0[g_idxPopZ(txp1  , typ1 ,   5, bxAtEast   , byAtNorth , 0)];
                        gNode[21] = ghostInterfacePhi.auxZ_0[g_idxPopZ(txm1  , tym1 ,   6, bxAtWest   , byAtSouth , 0)];
                        gNode[24] = ghostInterfacePhi.auxZ_0[g_idxPopZ(txp1  , tym1 ,   7, bxAtEast   , byAtSouth , 0)];
                        gNode[26] = ghostInterfacePhi.auxZ_0[g_idxPopZ(txm1  , typ1 ,   8, bxAtWest   , byAtNorth , 0)];
#endif
                }else{
                        gNode[ 6] = loadGhostZFront[g_idxPopZ(tx    , ty   ,   0, bx         , by        , loadBzFront)];
#ifdef D3G19
                        gNode[10] = loadGhostZFront[g_idxPopZ(txp1  , ty   ,   1, bxAtEast   , by        , loadBzFront)];
                        gNode[12] = loadGhostZFront[g_idxPopZ(tx    , typ1 ,   2, bx         , byAtNorth , loadBzFront)];
                        gNode[15] = loadGhostZFront[g_idxPopZ(txm1  , ty   ,   3, bxAtWest   , by        , loadBzFront)];
                        gNode[17] = loadGhostZFront[g_idxPopZ(tx    , tym1 ,   4, bx         , byAtSouth , loadBzFront)];
#endif
#ifdef D3G27
                        gNode[10] = loadGhostZFront[g_idxPopZ(txp1  , ty   ,   1, bxAtEast   , by        , loadBzFront)];
                        gNode[12] = loadGhostZFront[g_idxPopZ(tx    , typ1 ,   2, bx         , byAtNorth , loadBzFront)];
                        gNode[15] = loadGhostZFront[g_idxPopZ(txm1  , ty   ,   3, bxAtWest   , by        , loadBzFront)];
                        gNode[17] = loadGhostZFront[g_idxPopZ(tx    , tym1 ,   4, bx         , byAtSouth , loadBzFront)];
                        gNode[20] = loadGhostZFront[g_idxPopZ(txp1  , typ1 ,   5, bxAtEast   , byAtNorth , loadBzFront)];
                        gNode[21] = loadGhostZFront[g_idxPopZ(txm1  , tym1 ,   6, bxAtWest   , byAtSouth , loadBzFront)];
                        gNode[24] = loadGhostZFront[g_idxPopZ(txp1  , tym1 ,   7, bxAtEast   , byAtSouth , loadBzFront)];
                        gNode[26] = loadGhostZFront[g_idxPopZ(txm1  , typ1 ,   8, bxAtWest   , byAtNorth , loadBzFront)];
#endif           
                }
        }

    }
/* DEV_REF_START (from dev during merge)
#ifdef D3G27
        gNode[ 9] = ghostInterface.phi_fGhost.Z_1[g_idxPopZ(txm1, ty, 1, ((tx == 0) ? bxm1 : bx)                , by                                    , bzm1)];
        gNode[11] = ghostInterface.phi_fGhost.Z_1[g_idxPopZ(tx, tym1, 2, bx                                     , ((ty == 0) ? bym1 : by)               , bzm1)];
        gNode[16] = ghostInterface.phi_fGhost.Z_1[g_idxPopZ(txp1, ty, 3, ((tx == (BLOCK_NX - 1)) ? bxp1 : bx)   , by                                    , bzm1)];
        gNode[18] = ghostInterface.phi_fGhost.Z_1[g_idxPopZ(tx, typ1, 4, bx                                     , ((ty == (BLOCK_NY - 1)) ? byp1 : by)  , bzm1)];
        gNode[19] = ghostInterface.phi_fGhost.Z_1[g_idxPopZ(txm1, tym1, 5, ((tx == 0) ? bxm1 : bx)              , ((ty == 0) ? bym1 : by)               , bzm1)];
        gNode[22] = ghostInterface.phi_fGhost.Z_1[g_idxPopZ(txp1, typ1, 6, ((tx == (BLOCK_NX - 1)) ? bxp1 : bx) , ((ty == (BLOCK_NY - 1)) ? byp1 : by)  , bzm1)];
        gNode[23] = ghostInterface.phi_fGhost.Z_1[g_idxPopZ(txm1, typ1, 7, ((tx == 0) ? bxm1 : bx)              , ((ty == (BLOCK_NY - 1)) ? byp1 : by)  , bzm1)];
        gNode[25] = ghostInterface.phi_fGhost.Z_1[g_idxPopZ(txp1, tym1, 8, ((tx == (BLOCK_NX - 1)) ? bxp1 : bx) , ((ty == 0) ? bym1 : by)               , bzm1)];
#endif
    }
else if (tz == (BLOCK_NZ - 1))
    { //f
        gNode[ 6] = ghostInterface.phi_fGhost.Z_0[g_idxPopZ(tx  , ty, 0, bx                                     , by                                    , bzp1)];
#ifdef D3G19
        gNode[10] = ghostInterface.phi_fGhost.Z_0[g_idxPopZ(txp1, ty, 1, ((tx == (BLOCK_NX - 1)) ? bxp1 : bx)   , by                                    , bzp1)];
        gNode[12] = ghostInterface.phi_fGhost.Z_0[g_idxPopZ(tx, typ1, 2, bx                                     , ((ty == (BLOCK_NY - 1)) ? byp1 : by)  , bzp1)];
        gNode[15] = ghostInterface.phi_fGhost.Z_0[g_idxPopZ(txm1, ty, 3, ((tx == 0) ? bxm1 : bx)                , by                                    , bzp1)];
        gNode[17] = ghostInterface.phi_fGhost.Z_0[g_idxPopZ(tx, tym1, 4, bx                                     , ((ty == 0) ? bym1 : by)               , bzp1)];
#endif
#ifdef D3G27
        gNode[10] = ghostInterface.phi_fGhost.Z_0[g_idxPopZ(txp1, ty, 1, ((tx == (BLOCK_NX - 1)) ? bxp1 : bx)   , by                                    , bzp1)];
        gNode[12] = ghostInterface.phi_fGhost.Z_0[g_idxPopZ(tx, typ1, 2, bx                                     , ((ty == (BLOCK_NY - 1)) ? byp1 : by)  , bzp1)];
        gNode[15] = ghostInterface.phi_fGhost.Z_0[g_idxPopZ(txm1, ty, 3, ((tx == 0) ? bxm1 : bx)                , by                                    , bzp1)];
        gNode[17] = ghostInterface.phi_fGhost.Z_0[g_idxPopZ(tx, tym1, 4, bx                                     , ((ty == 0) ? bym1 : by)               , bzp1)];
        gNode[20] = ghostInterface.phi_fGhost.Z_0[g_idxPopZ(txp1, typ1, 5, ((tx == (BLOCK_NX - 1)) ? bxp1 : bx) , ((ty == (BLOCK_NY - 1)) ? byp1 : by)  , bzp1)];
        gNode[21] = ghostInterface.phi_fGhost.Z_0[g_idxPopZ(txm1, tym1, 6, ((tx == 0) ? bxm1 : bx)              , ((ty == 0) ? bym1 : by)               , bzp1)];
        gNode[24] = ghostInterface.phi_fGhost.Z_0[g_idxPopZ(txp1, tym1, 7, ((tx == (BLOCK_NX - 1)) ? bxp1 : bx) , ((ty == 0) ? bym1 : by)               , bzp1)];
        gNode[26] = ghostInterface.phi_fGhost.Z_0[g_idxPopZ(txm1, typ1, 8, ((tx == 0) ? bxm1 : bx)              , ((ty == (BLOCK_NY - 1)) ? byp1 : by)  , bzp1)];
#endif
    }

DEV_REF_END */
