
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

if (tz == 0)
    { //b
        gNode[ 5] = ghostInterface.phi_fGhost.Z_1[g_idxPopZ(tx  , ty, 0, bx                                     , by                                    , bzm1)];
#ifdef D3G19
        gNode[ 9] = ghostInterface.phi_fGhost.Z_1[g_idxPopZ(txm1, ty, 1, ((tx == 0) ? bxm1 : bx)                , by                                    , bzm1)];
        gNode[11] = ghostInterface.phi_fGhost.Z_1[g_idxPopZ(tx, tym1, 2, bx                                     , ((ty == 0) ? bym1 : by)               , bzm1)];
        gNode[16] = ghostInterface.phi_fGhost.Z_1[g_idxPopZ(txp1, ty, 3, ((tx == (BLOCK_NX - 1)) ? bxp1 : bx)   , by                                    , bzm1)];
        gNode[18] = ghostInterface.phi_fGhost.Z_1[g_idxPopZ(tx, typ1, 4, bx                                     , ((ty == (BLOCK_NY - 1)) ? byp1 : by)  , bzm1)];
#endif
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