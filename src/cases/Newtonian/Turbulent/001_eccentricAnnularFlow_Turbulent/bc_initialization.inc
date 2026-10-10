// Fluid occupies the region inside the outer circle and outside the offset
// inner circle. Both cylinders are extruded along z.
const dfloat outerDx = (dfloat)x - OUTER_CENTER_X;
const dfloat outerDy = (dfloat)y - OUTER_CENTER_Y;
const dfloat innerDx = (dfloat)x - INNER_CENTER_X;
const dfloat innerDy = (dfloat)y - INNER_CENTER_Y;
const dfloat outerRadiusSquared = outerDx * outerDx + outerDy * outerDy;
const dfloat innerRadiusSquared = innerDx * innerDx + innerDy * innerDy;

if (outerRadiusSquared >= OUTER_RADIUS * OUTER_RADIUS ||
    innerRadiusSquared <= INNER_RADIUS * INNER_RADIUS) {
    nodeType = SOLID_NODE;
} else {
    nodeType = MISSING_DEFINITION;
}

