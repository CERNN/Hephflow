const dfloat outerDx = (dfloat)x - OUTER_CENTER_X;
const dfloat outerDy = (dfloat)y - OUTER_CENTER_Y;
const dfloat innerDx = (dfloat)x - INNER_CENTER_X;
const dfloat innerDy = (dfloat)y - INNER_CENTER_Y;
const dfloat outerRadius = sqrt(outerDx * outerDx + outerDy * outerDy);
const dfloat innerRadius = sqrt(innerDx * innerDx + innerDy * innerDy);
const dfloat outerWallDistance = OUTER_RADIUS - outerRadius;
const dfloat innerWallDistance = innerRadius - INNER_RADIUS;

ux = 0.0;
uy = 0.0;
uz = 0.0;
rho = RHO_0;

if (outerWallDistance > 0.0 && innerWallDistance > 0.0) {
    const dfloat distanceSum = innerWallDistance + outerWallDistance;
    const dfloat innerWeight = outerWallDistance / distanceSum;
    const dfloat outerWeight = innerWallDistance / distanceSum;
    const dfloat profileShape =
        4.0 * innerWallDistance * outerWallDistance /
        (distanceSum * distanceSum);

    const dfloat innerWallUx =
        INNER_TRANSLATION_X - INNER_ANGULAR_VELOCITY * innerDy;
    const dfloat innerWallUy =
        INNER_TRANSLATION_Y + INNER_ANGULAR_VELOCITY * innerDx;
    const dfloat outerWallUx =
        OUTER_TRANSLATION_X - OUTER_ANGULAR_VELOCITY * outerDy;
    const dfloat outerWallUy =
        OUTER_TRANSLATION_Y + OUTER_ANGULAR_VELOCITY * outerDx;

    ux = innerWeight * innerWallUx + outerWeight * outerWallUx;
    uy = innerWeight * innerWallUy + outerWeight * outerWallUy;
    uz = innerWeight * INNER_TRANSLATION_Z +
         outerWeight * OUTER_TRANSLATION_Z +
         U_BULK * profileShape;

    // Taper perturbations to zero at both physical walls.
    const dfloat perturbation = 0.2 * U_BULK * profileShape;
    const int l = index;
    const int Nt = NUMBER_LBM_POP_NODES;
    uz += perturbation * randomNumbers[l + z - Nt * ((l + z) / Nt)];
    ux += perturbation * randomNumbers[l + x - Nt * ((l + x) / Nt)];
    uy += perturbation * randomNumbers[l + y - Nt * ((l + y) / Nt)];
}
