// Thermal BC for a laterally heated lid-driven cavity.
// WEST/EAST: Dirichlet; SOUTH/NORTH/BACK/FRONT: adiabatic.
// This include runs after streaming. q*_t30 are raw moments; F_M_I_SCALE is
// applied by the caller.
#define G_UPDATE_MOMENTS() do { cVar = gNode[0] + gNode[1] + gNode[2] + gNode[3] + gNode[4] + gNode[5] + gNode[6] + gNode[7] + gNode[8] + gNode[9] + gNode[10] + gNode[11] + gNode[12] + gNode[13] + gNode[14] + gNode[15] + gNode[16] + gNode[17] + gNode[18]; invC = 1.0_df / cVar; qx_t30 = gNode[1] - gNode[2] + gNode[7] - gNode[8] + gNode[9] - gNode[10] + gNode[13] - gNode[14] + gNode[15] - gNode[16]; qy_t30 = gNode[3] - gNode[4] + gNode[7] - gNode[8] + gNode[11] - gNode[12] + gNode[14] - gNode[13] + gNode[17] - gNode[18]; qz_t30 = gNode[5] - gNode[6] + gNode[9] - gNode[10] + gNode[11] - gNode[12] + gNode[16] - gNode[15] + gNode[18] - gNode[17]; } while (0)


switch (nodeType)
{

// WEST: hot Dirichlet. Takes precedence at intersections.

case SOUTH_WEST_BACK:
case SOUTH_WEST_FRONT:
case NORTH_WEST_BACK:
case NORTH_WEST_FRONT:

case SOUTH_WEST:
case NORTH_WEST:

case WEST_BACK:
case WEST_FRONT:

case WEST:
{
    cVar = T_HOT;
    invC = 1.0_df / cVar;

    qx_t30 = 0.0_df;
    qy_t30 = 0.0_df;
    qz_t30 = 0.0_df;

    break;
}


// EAST: cold Dirichlet.

case SOUTH_EAST_BACK:
case SOUTH_EAST_FRONT:
case NORTH_EAST_BACK:
case NORTH_EAST_FRONT:
case SOUTH_EAST:
case NORTH_EAST:
case EAST_BACK:
case EAST_FRONT:
case EAST:
{
    cVar = T_COLD;
    invC = 1.0_df / cVar;

    qx_t30 = 0.0_df;
    qy_t30 = 0.0_df;
    qz_t30 = 0.0_df;

    break;
}


// SOUTH-BACK edge: interior is +y, +z.

case SOUTH_BACK:
{
    // South reflection
    gNode[3]  = gNode[4];
    gNode[7]  = gNode[13];
    gNode[14] = gNode[8];

    // Back reflection
    gNode[5]  = gNode[6];
    gNode[9]  = gNode[15];
    gNode[16] = gNode[10];

    // yz corner populations; only g12 is sourced from the interior.
    gNode[11] = gNode[12];
    gNode[17] = gNode[12];
    gNode[18] = gNode[12];


    G_UPDATE_MOMENTS();

    break;
}

case SOUTH_FRONT:
{
    // South
    gNode[3]  = gNode[4];
    gNode[7]  = gNode[13];
    gNode[14] = gNode[8];

    // Front
    gNode[6]  = gNode[5];
    gNode[15] = gNode[9];
    gNode[10] = gNode[16];

    // yz corner populations
    gNode[11] = gNode[18];
    gNode[12] = gNode[18];
    gNode[17] = gNode[18];


    G_UPDATE_MOMENTS();

    break;
}
case NORTH_BACK:
{
    // North
    gNode[4]  = gNode[3];
    gNode[13] = gNode[7];
    gNode[8]  = gNode[14];

    // Back
    gNode[5]  = gNode[6];
    gNode[9]  = gNode[15];
    gNode[16] = gNode[10];


    // yz corner populations
    gNode[11] = gNode[17];
    gNode[12] = gNode[17];
    gNode[18] = gNode[17];

    G_UPDATE_MOMENTS();

    break;
}
case NORTH_FRONT:
{
    // North
    gNode[4]  = gNode[3];
    gNode[13] = gNode[7];
    gNode[8]  = gNode[14];

    // Front
    gNode[6]  = gNode[5];
    gNode[15] = gNode[9];
    gNode[10] = gNode[16];


    // yz corner populations
    gNode[12] = gNode[11];
    gNode[17] = gNode[11];
    gNode[18] = gNode[11];


    G_UPDATE_MOMENTS();

    break;
}
case SOUTH:
{
    // normal
    gNode[3] = gNode[4];

    // xy
    gNode[7]  = gNode[13];
    gNode[14] = gNode[8];

    // yz
    gNode[11] = gNode[18];
    gNode[17] = gNode[12];

    G_UPDATE_MOMENTS();

    break;
}
case NORTH:
{
    // Normal
    gNode[4] = gNode[3];
    // xy diagonals
    gNode[13] = gNode[7];
    gNode[8]  = gNode[14];
    // yz diagonals
    gNode[18] = gNode[11];
    gNode[12] = gNode[17];

    G_UPDATE_MOMENTS();
#endif
    break;
}
case BACK:
{
    // Normal
    gNode[5] = gNode[6];

    // xz diagonals
    gNode[9]  = gNode[15];
    gNode[16] = gNode[10];

    // yz diagonals
    gNode[11] = gNode[17];
    gNode[18] = gNode[12];

    G_UPDATE_MOMENTS();

    break;
}
case FRONT:
{
    // Normal
    gNode[6] = gNode[5];

    // xz diagonals
    gNode[15] = gNode[9];
    gNode[10] = gNode[16];

    // yz diagonals
    gNode[17] = gNode[11];
    gNode[12] = gNode[18];

    G_UPDATE_MOMENTS();

    break;
}

} // switch(nodeType)


#undef G_UPDATE_MOMENTS
