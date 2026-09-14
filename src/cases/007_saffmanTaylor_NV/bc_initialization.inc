// Geometry for the displacement case:
//   z = 0            : inlet  (BACK)
//   z = NZ_TOTAL - 1 : outlet (FRONT)
//   y = 0      : no-slip wall (SOUTH)
//   y = NY - 1 : no-slip wall (NORTH)
//   x           : periodic
if (y == 0 && z == 0) {               // South/inlet edge
    nodeType = SOUTH_BACK;
} else if (y == (NY - 1) && z == 0) { // North/inlet edge
    nodeType = NORTH_BACK;
} else if (y == 0 && z == (NZ_TOTAL - 1)) { // South/outlet edge
    nodeType = SOUTH_FRONT;
} else if (y == (NY - 1) && z == (NZ_TOTAL - 1)) { // North/outlet edge
    nodeType = NORTH_FRONT;
} else if (y == 0) {                  // South wall
    nodeType = SOUTH;
} else if (y == (NY - 1)) {          // North wall
    nodeType = NORTH;
} else if (z == 0) {                 // Inlet
    nodeType = BACK;
} else if (z == (NZ_TOTAL - 1)) {    // Outlet
    nodeType = FRONT;
} else {
    nodeType = BULK;
}
