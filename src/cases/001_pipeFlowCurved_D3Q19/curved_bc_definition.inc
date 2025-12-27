const dfloat3 b = dfloat3(x, y, z);
h_curvedBC_array[curvedBCCount].b = b;

const dfloat3 duct_center = dfloat3(DUCT_CENTER_X, DUCT_CENTER_Y, z); // cylnder is on the z direction
const dfloat max_radius = OUTER_RADIUS; 
const dfloat lattice_dx = sqrt(2.0); // TODO: evaluate its effect

const dfloat3 center_to_b = b - duct_center;
const dfloat3 unit_normal = vector_normalize(center_to_b);

const dfloat3 wall = duct_center + unit_normal * max_radius;

h_curvedBC_array[curvedBCCount].w = wall;

h_curvedBC_array[curvedBCCount].delta = vector_length(wall - b);

const dfloat3 pf1 = wall + unit_normal * (1.0 * lattice_dx);
h_curvedBC_array[curvedBCCount].pf1 = pf1;

const dfloat3 pf2 = wall + unit_normal * (2.0 * lattice_dx);
h_curvedBC_array[curvedBCCount].pf2 = pf2;

const dfloat3 pf3 = wall + unit_normal * (3.0 * lattice_dx);
h_curvedBC_array[curvedBCCount].pf3 = pf3;


dfloat theta = atan2f(center_to_b.y, center_to_b.x);
if (theta < 0.0){theta += 2.0 * M_PI;}
h_curvedBC_array[curvedBCCount].theta = theta;

