// Geometry for the displacement case:
//   x = 0      : inlet  (WEST)
//   x = NX - 1 : outlet (EAST)
//   y = 0      : no-slip wall (SOUTH)
//   y = NY - 1 : no-slip wall (NORTH)
//   z           : periodic
if (y == 0) {                         // South wall
    nodeType = SOUTH;
} else if (y == (NY - 1)) {          // North wall
    nodeType = NORTH;
} else if (x == 0) {                 // Inlet
    nodeType = WEST;
} else if (x == (NX - 1)) {          // Outlet
    nodeType = EAST;
} else {
    nodeType = BULK;
}
