if (y == 0){ //S
    nodeType = SOUTH;
}else if (y == (NY - 1)){ // N
    nodeType = NORTH;
}else if (x == 0){ // W
    nodeType = WEST + BC_VELOCITY_WALL;
}else if (x == (NX - 1)){ // E
    nodeType = EAST + BC_OUTFLOW;
}else{
    nodeType = BULK;
}
    