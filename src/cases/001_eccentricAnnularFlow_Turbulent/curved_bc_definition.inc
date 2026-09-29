const dfloat3 b = dfloat3(x, y, z);
h_curvedBC_array[curvedBCCount].b = b;

const bool isConvex = isConvexCurvedBoundaryNodeType(nodeType);
const dfloat3 wallCenter = isConvex
    ? dfloat3(INNER_CENTER_X, INNER_CENTER_Y, z)
    : dfloat3(OUTER_CENTER_X, OUTER_CENTER_Y, z);
const dfloat wallRadius = isConvex ? INNER_RADIUS : OUTER_RADIUS;
const dfloat3 centerToBoundaryNode = b - wallCenter;
const dfloat3 radialNormal = vector_normalize(centerToBoundaryNode);
const dfloat3 wall = wallCenter + radialNormal * wallRadius;
h_curvedBC_array[curvedBCCount].w = wall;

const dfloat translationX = isConvex
    ? INNER_TRANSLATION_X : OUTER_TRANSLATION_X;
const dfloat translationY = isConvex
    ? INNER_TRANSLATION_Y : OUTER_TRANSLATION_Y;
const dfloat translationZ = isConvex
    ? INNER_TRANSLATION_Z : OUTER_TRANSLATION_Z;
const dfloat angularVelocity = isConvex
    ? INNER_ANGULAR_VELOCITY : OUTER_ANGULAR_VELOCITY;

// Rigid-wall velocity U + omega x (wall - center), with omega along z.
h_curvedBC_array[curvedBCCount].wallVel = dfloat3(
    translationX - angularVelocity * (wall.y - wallCenter.y),
    translationY + angularVelocity * (wall.x - wallCenter.x),
    translationZ);
h_curvedBC_array[curvedBCCount].vel =
    h_curvedBC_array[curvedBCCount].wallVel;

h_curvedBC_array[curvedBCCount].delta = vector_length(wall - b);
const dfloat latticeSpacingAlongNormal = sqrt(2.0);
h_curvedBC_array[curvedBCCount].delta_r = latticeSpacingAlongNormal;

// The fluid lies outward from the convex inner cylinder and inward from the
// concave outer cylinder.
const dfloat3 wallToFluidNormal = isConvex ? radialNormal : -radialNormal;
h_curvedBC_array[curvedBCCount].pf1 =
    wall + wallToFluidNormal * latticeSpacingAlongNormal;
h_curvedBC_array[curvedBCCount].pf2 =
    wall + wallToFluidNormal * (2.0 * latticeSpacingAlongNormal);
h_curvedBC_array[curvedBCCount].pf3 =
    wall + wallToFluidNormal * (3.0 * latticeSpacingAlongNormal);

dfloat theta = atan2f(centerToBoundaryNode.y, centerToBoundaryNode.x);
if (theta < 0.0) theta += 2.0 * M_PI;
h_curvedBC_array[curvedBCCount].theta = theta;

