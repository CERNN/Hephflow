// definition of solid and fluid nodes for cylinder in the z direction

dfloat3 v = dfloat3(x - DUCT_CENTER_X, y - DUCT_CENTER_Y, z - z);  //direction
dfloat r = vector_length(v);

if(r >= EXTERNAL_DUCT_BC_RADIUS){ //solid
    nodeType = SOLID_NODE;
}else{
    nodeType = MISSING_DEFINITION;
}