/* ============================================================================
 * BOUNDARY CONDITION INITIALIZATION
 * Assigns node types to grid points based on geometry
 * 
 * NOTE: This file is included inside an existing loop over (x, y, z).
 *       Do NOT add for loops here.
 * 
 * Available node types are defined in nodeTypeMap.h using bitwise operations:
 * 
 * FACES:
 *   BULK          - Interior fluid node (0b00000000)
 *   SOUTH         - Y- boundary (y=0)
 *   NORTH         - Y+ boundary (y=NY-1)
 *   WEST          - X- boundary (x=0)
 *   EAST          - X+ boundary (x=NX-1)
 *   BACK          - Z- boundary (z=0)
 *   FRONT         - Z+ boundary (z=NZ-1)
 * 
 * EDGES (combine faces with bitwise OR):
 *   NORTH_WEST, NORTH_EAST, SOUTH_WEST, SOUTH_EAST
 *   NORTH_FRONT, NORTH_BACK, SOUTH_FRONT, SOUTH_BACK
 *   WEST_FRONT, WEST_BACK, EAST_FRONT, EAST_BACK
 * 
 * CORNERS (combine three faces with bitwise OR):
 *   NORTH_WEST_FRONT, NORTH_WEST_BACK, etc.
 * 
 * BOUNDARY CONDITIONS (combined with direction using bitwise OR):
 *   BC_ZERO_VELOCITY_WALL - No-slip wall (default)
 *   BC_VELOCITY_WALL      - Prescribed velocity wall
 *   BC_OUTFLOW            - Outflow/pressure boundary
 *   BC_SYMMETRY           - Symmetry plane
 *   BC_FREESLIP           - Free-slip wall
 *   BC_CURVED_BC_CONCAV   - Curved boundary (concave)
 *   BC_CURVED_BC_CONVEX   - Curved boundary (convex)
 * 
 * See nodeTypeMap.h for complete bitwise encoding details.
 * ========================================================================== */

// Default: bulk node
nodeType = BULK;

/* --------------------- Y BOUNDARIES --------------------- */
if (y == 0) {
    nodeType = SOUTH;
} else if (y == (NY - 1)) {
    nodeType = NORTH;
}

/* --------------------- X BOUNDARIES --------------------- */
// if (x == 0) {
//     nodeType = WEST;
// } else if (x == (NX - 1)) {
//     nodeType = EAST;
// }

/* --------------------- Z BOUNDARIES --------------------- */
// if (z == 0) {
//     nodeType = BACK;
// } else if (z == (NZ - 1)) {
//     nodeType = FRONT;
// }

/* --------------------- CUSTOM GEOMETRY --------------------- */
// Add case-specific geometry here
// Example: cylindrical obstacle
// dfloat cx = NX / 2.0, cy = NY / 2.0, r = 10.0;
// if ((x - cx) * (x - cx) + (y - cy) * (y - cy) < r * r) {
//     nodeType = WALL;
// }
