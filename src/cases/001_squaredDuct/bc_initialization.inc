if (y == 0 && x == 0) {
    nodeType = SOUTH_WEST;
} else if (y == (NY - 1) && x == 0) {
    nodeType = NORTH_WEST;
} else if (y == 0 && x == (NX - 1)) {
    nodeType = SOUTH_EAST;
} else if (y == (NY - 1) && x == (NX - 1)) {
    nodeType = NORTH_EAST;
} else if (y == 0) {
    nodeType = SOUTH;
} else if (y == (NY - 1)) {
    nodeType = NORTH;
} else if (x == 0) {
    nodeType = WEST;
} else if (x == (NX - 1)) {
    nodeType = EAST;
} else {
    nodeType = BULK;
}
