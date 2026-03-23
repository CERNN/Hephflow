// AA scheme: read conf from ghost, parity-aware
const int loadBxWest  = (stepParity ? bx : bxm1);
const int loadBxEast  = (stepParity ? bx : bxp1);
const int loadBySouth = (stepParity ? by : bym1);
const int loadByNorth = (stepParity ? by : byp1);
const int loadBzBack  = (stepParity ? bz : bzm1);
const int loadBzFront = (stepParity ? bz : bzp1);

if(stepParity){
    if (tx == 0) {
        s_pop[idxConfBlock(tx - 1, ty, tz, 0)] = ghostInterface.conf.X_0[g_idxConfX(ty, tz, 0, loadBxWest, by, bz)];
        s_pop[idxConfBlock(tx - 1, ty, tz, 1)] = ghostInterface.conf.X_0[g_idxConfX(ty, tz, 1, loadBxWest, by, bz)];
        s_pop[idxConfBlock(tx - 1, ty, tz, 2)] = ghostInterface.conf.X_0[g_idxConfX(ty, tz, 2, loadBxWest, by, bz)];
        s_pop[idxConfBlock(tx - 1, ty, tz, 3)] = ghostInterface.conf.X_0[g_idxConfX(ty, tz, 3, loadBxWest, by, bz)];
        s_pop[idxConfBlock(tx - 1, ty, tz, 4)] = ghostInterface.conf.X_0[g_idxConfX(ty, tz, 4, loadBxWest, by, bz)];
        s_pop[idxConfBlock(tx - 1, ty, tz, 5)] = ghostInterface.conf.X_0[g_idxConfX(ty, tz, 5, loadBxWest, by, bz)];
    }
    if (tx == BLOCK_NX - 1) {
        s_pop[idxConfBlock(tx + 1, ty, tz, 0)] = ghostInterface.conf.X_1[g_idxConfX(ty, tz, 0, loadBxEast, by, bz)];
        s_pop[idxConfBlock(tx + 1, ty, tz, 1)] = ghostInterface.conf.X_1[g_idxConfX(ty, tz, 1, loadBxEast, by, bz)];
        s_pop[idxConfBlock(tx + 1, ty, tz, 2)] = ghostInterface.conf.X_1[g_idxConfX(ty, tz, 2, loadBxEast, by, bz)];
        s_pop[idxConfBlock(tx + 1, ty, tz, 3)] = ghostInterface.conf.X_1[g_idxConfX(ty, tz, 3, loadBxEast, by, bz)];
        s_pop[idxConfBlock(tx + 1, ty, tz, 4)] = ghostInterface.conf.X_1[g_idxConfX(ty, tz, 4, loadBxEast, by, bz)];
        s_pop[idxConfBlock(tx + 1, ty, tz, 5)] = ghostInterface.conf.X_1[g_idxConfX(ty, tz, 5, loadBxEast, by, bz)];
    }
    if (ty == 0) {
        s_pop[idxConfBlock(tx, ty - 1, tz, 0)] = ghostInterface.conf.Y_0[g_idxConfY(tx, tz, 0, bx, loadBySouth, bz)];
        s_pop[idxConfBlock(tx, ty - 1, tz, 1)] = ghostInterface.conf.Y_0[g_idxConfY(tx, tz, 1, bx, loadBySouth, bz)];
        s_pop[idxConfBlock(tx, ty - 1, tz, 2)] = ghostInterface.conf.Y_0[g_idxConfY(tx, tz, 2, bx, loadBySouth, bz)];
        s_pop[idxConfBlock(tx, ty - 1, tz, 3)] = ghostInterface.conf.Y_0[g_idxConfY(tx, tz, 3, bx, loadBySouth, bz)];
        s_pop[idxConfBlock(tx, ty - 1, tz, 4)] = ghostInterface.conf.Y_0[g_idxConfY(tx, tz, 4, bx, loadBySouth, bz)];
        s_pop[idxConfBlock(tx, ty - 1, tz, 5)] = ghostInterface.conf.Y_0[g_idxConfY(tx, tz, 5, bx, loadBySouth, bz)];
    }
    if (ty == BLOCK_NY - 1) {
        s_pop[idxConfBlock(tx, ty + 1, tz, 0)] = ghostInterface.conf.Y_1[g_idxConfY(tx, tz, 0, bx, loadByNorth, bz)];
        s_pop[idxConfBlock(tx, ty + 1, tz, 1)] = ghostInterface.conf.Y_1[g_idxConfY(tx, tz, 1, bx, loadByNorth, bz)];
        s_pop[idxConfBlock(tx, ty + 1, tz, 2)] = ghostInterface.conf.Y_1[g_idxConfY(tx, tz, 2, bx, loadByNorth, bz)];
        s_pop[idxConfBlock(tx, ty + 1, tz, 3)] = ghostInterface.conf.Y_1[g_idxConfY(tx, tz, 3, bx, loadByNorth, bz)];
        s_pop[idxConfBlock(tx, ty + 1, tz, 4)] = ghostInterface.conf.Y_1[g_idxConfY(tx, tz, 4, bx, loadByNorth, bz)];
        s_pop[idxConfBlock(tx, ty + 1, tz, 5)] = ghostInterface.conf.Y_1[g_idxConfY(tx, tz, 5, bx, loadByNorth, bz)];
    }
    if (tz == 0) {
        s_pop[idxConfBlock(tx, ty, tz - 1, 0)] = ghostInterface.conf.Z_0[g_idxConfZ(tx, ty, 0, bx, by, loadBzBack)];
        s_pop[idxConfBlock(tx, ty, tz - 1, 1)] = ghostInterface.conf.Z_0[g_idxConfZ(tx, ty, 1, bx, by, loadBzBack)];
        s_pop[idxConfBlock(tx, ty, tz - 1, 2)] = ghostInterface.conf.Z_0[g_idxConfZ(tx, ty, 2, bx, by, loadBzBack)];
        s_pop[idxConfBlock(tx, ty, tz - 1, 3)] = ghostInterface.conf.Z_0[g_idxConfZ(tx, ty, 3, bx, by, loadBzBack)];
        s_pop[idxConfBlock(tx, ty, tz - 1, 4)] = ghostInterface.conf.Z_0[g_idxConfZ(tx, ty, 4, bx, by, loadBzBack)];
        s_pop[idxConfBlock(tx, ty, tz - 1, 5)] = ghostInterface.conf.Z_0[g_idxConfZ(tx, ty, 5, bx, by, loadBzBack)];
    }
    if (tz == BLOCK_NZ - 1) {
        s_pop[idxConfBlock(tx, ty, tz + 1, 0)] = ghostInterface.conf.Z_1[g_idxConfZ(tx, ty, 0, bx, by, loadBzFront)];
        s_pop[idxConfBlock(tx, ty, tz + 1, 1)] = ghostInterface.conf.Z_1[g_idxConfZ(tx, ty, 1, bx, by, loadBzFront)];
        s_pop[idxConfBlock(tx, ty, tz + 1, 2)] = ghostInterface.conf.Z_1[g_idxConfZ(tx, ty, 2, bx, by, loadBzFront)];
        s_pop[idxConfBlock(tx, ty, tz + 1, 3)] = ghostInterface.conf.Z_1[g_idxConfZ(tx, ty, 3, bx, by, loadBzFront)];
        s_pop[idxConfBlock(tx, ty, tz + 1, 4)] = ghostInterface.conf.Z_1[g_idxConfZ(tx, ty, 4, bx, by, loadBzFront)];
        s_pop[idxConfBlock(tx, ty, tz + 1, 5)] = ghostInterface.conf.Z_1[g_idxConfZ(tx, ty, 5, bx, by, loadBzFront)];
    }
}else{
    if (tx == 0) {
        s_pop[idxConfBlock(tx - 1, ty, tz, 0)] = ghostInterface.conf.X_1[g_idxConfX(ty, tz, 0, loadBxWest, by, bz)];
        s_pop[idxConfBlock(tx - 1, ty, tz, 1)] = ghostInterface.conf.X_1[g_idxConfX(ty, tz, 1, loadBxWest, by, bz)];
        s_pop[idxConfBlock(tx - 1, ty, tz, 2)] = ghostInterface.conf.X_1[g_idxConfX(ty, tz, 2, loadBxWest, by, bz)];
        s_pop[idxConfBlock(tx - 1, ty, tz, 3)] = ghostInterface.conf.X_1[g_idxConfX(ty, tz, 3, loadBxWest, by, bz)];
        s_pop[idxConfBlock(tx - 1, ty, tz, 4)] = ghostInterface.conf.X_1[g_idxConfX(ty, tz, 4, loadBxWest, by, bz)];
        s_pop[idxConfBlock(tx - 1, ty, tz, 5)] = ghostInterface.conf.X_1[g_idxConfX(ty, tz, 5, loadBxWest, by, bz)];
    }
    if (tx == BLOCK_NX - 1) {
        s_pop[idxConfBlock(tx + 1, ty, tz, 0)] = ghostInterface.conf.X_0[g_idxConfX(ty, tz, 0, loadBxEast, by, bz)];
        s_pop[idxConfBlock(tx + 1, ty, tz, 1)] = ghostInterface.conf.X_0[g_idxConfX(ty, tz, 1, loadBxEast, by, bz)];
        s_pop[idxConfBlock(tx + 1, ty, tz, 2)] = ghostInterface.conf.X_0[g_idxConfX(ty, tz, 2, loadBxEast, by, bz)];
        s_pop[idxConfBlock(tx + 1, ty, tz, 3)] = ghostInterface.conf.X_0[g_idxConfX(ty, tz, 3, loadBxEast, by, bz)];
        s_pop[idxConfBlock(tx + 1, ty, tz, 4)] = ghostInterface.conf.X_0[g_idxConfX(ty, tz, 4, loadBxEast, by, bz)];
        s_pop[idxConfBlock(tx + 1, ty, tz, 5)] = ghostInterface.conf.X_0[g_idxConfX(ty, tz, 5, loadBxEast, by, bz)];
    }
    if (ty == 0) {
        s_pop[idxConfBlock(tx, ty - 1, tz, 0)] = ghostInterface.conf.Y_1[g_idxConfY(tx, tz, 0, bx, loadBySouth, bz)];
        s_pop[idxConfBlock(tx, ty - 1, tz, 1)] = ghostInterface.conf.Y_1[g_idxConfY(tx, tz, 1, bx, loadBySouth, bz)];
        s_pop[idxConfBlock(tx, ty - 1, tz, 2)] = ghostInterface.conf.Y_1[g_idxConfY(tx, tz, 2, bx, loadBySouth, bz)];
        s_pop[idxConfBlock(tx, ty - 1, tz, 3)] = ghostInterface.conf.Y_1[g_idxConfY(tx, tz, 3, bx, loadBySouth, bz)];
        s_pop[idxConfBlock(tx, ty - 1, tz, 4)] = ghostInterface.conf.Y_1[g_idxConfY(tx, tz, 4, bx, loadBySouth, bz)];
        s_pop[idxConfBlock(tx, ty - 1, tz, 5)] = ghostInterface.conf.Y_1[g_idxConfY(tx, tz, 5, bx, loadBySouth, bz)];
    }
    if (ty == BLOCK_NY - 1) {
        s_pop[idxConfBlock(tx, ty + 1, tz, 0)] = ghostInterface.conf.Y_0[g_idxConfY(tx, tz, 0, bx, loadByNorth, bz)];
        s_pop[idxConfBlock(tx, ty + 1, tz, 1)] = ghostInterface.conf.Y_0[g_idxConfY(tx, tz, 1, bx, loadByNorth, bz)];
        s_pop[idxConfBlock(tx, ty + 1, tz, 2)] = ghostInterface.conf.Y_0[g_idxConfY(tx, tz, 2, bx, loadByNorth, bz)];
        s_pop[idxConfBlock(tx, ty + 1, tz, 3)] = ghostInterface.conf.Y_0[g_idxConfY(tx, tz, 3, bx, loadByNorth, bz)];
        s_pop[idxConfBlock(tx, ty + 1, tz, 4)] = ghostInterface.conf.Y_0[g_idxConfY(tx, tz, 4, bx, loadByNorth, bz)];
        s_pop[idxConfBlock(tx, ty + 1, tz, 5)] = ghostInterface.conf.Y_0[g_idxConfY(tx, tz, 5, bx, loadByNorth, bz)];
    }
    if (tz == 0) {
        s_pop[idxConfBlock(tx, ty, tz - 1, 0)] = ghostInterface.conf.Z_1[g_idxConfZ(tx, ty, 0, bx, by, loadBzBack)];
        s_pop[idxConfBlock(tx, ty, tz - 1, 1)] = ghostInterface.conf.Z_1[g_idxConfZ(tx, ty, 1, bx, by, loadBzBack)];
        s_pop[idxConfBlock(tx, ty, tz - 1, 2)] = ghostInterface.conf.Z_1[g_idxConfZ(tx, ty, 2, bx, by, loadBzBack)];
        s_pop[idxConfBlock(tx, ty, tz - 1, 3)] = ghostInterface.conf.Z_1[g_idxConfZ(tx, ty, 3, bx, by, loadBzBack)];
        s_pop[idxConfBlock(tx, ty, tz - 1, 4)] = ghostInterface.conf.Z_1[g_idxConfZ(tx, ty, 4, bx, by, loadBzBack)];
        s_pop[idxConfBlock(tx, ty, tz - 1, 5)] = ghostInterface.conf.Z_1[g_idxConfZ(tx, ty, 5, bx, by, loadBzBack)];
    }
    if (tz == BLOCK_NZ - 1) {
        s_pop[idxConfBlock(tx, ty, tz + 1, 0)] = ghostInterface.conf.Z_0[g_idxConfZ(tx, ty, 0, bx, by, loadBzFront)];
        s_pop[idxConfBlock(tx, ty, tz + 1, 1)] = ghostInterface.conf.Z_0[g_idxConfZ(tx, ty, 1, bx, by, loadBzFront)];
        s_pop[idxConfBlock(tx, ty, tz + 1, 2)] = ghostInterface.conf.Z_0[g_idxConfZ(tx, ty, 2, bx, by, loadBzFront)];
        s_pop[idxConfBlock(tx, ty, tz + 1, 3)] = ghostInterface.conf.Z_0[g_idxConfZ(tx, ty, 3, bx, by, loadBzFront)];
        s_pop[idxConfBlock(tx, ty, tz + 1, 4)] = ghostInterface.conf.Z_0[g_idxConfZ(tx, ty, 4, bx, by, loadBzFront)];
        s_pop[idxConfBlock(tx, ty, tz + 1, 5)] = ghostInterface.conf.Z_0[g_idxConfZ(tx, ty, 5, bx, by, loadBzFront)];
    }
}
