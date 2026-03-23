/**
 *  @file ghostStagingLayout.inc
 *  @brief Constants and macros for shared-memory ghost staging layout
 *  @details Defines the layout of ghost face data within s_pop when used as
 *           a staging buffer for cooperative ghost I/O. The staging buffer
 *           reuses s_pop after intra-block streaming is complete.
 *
 *           Layout in s_pop: [West][East][South][North][Back][Front]
 *           Each face section has FACE_NODES * QF elements.
 *
 *  @version 0.1.0
 *  @date 01/03/2026
 */

#ifndef __GHOST_STAGING_LAYOUT_INC
#define __GHOST_STAGING_LAYOUT_INC

/* ======================== FACE SIZES ======================== */

// Number of nodes on each face of the block
constexpr int GHOST_FACE_NODES_X = BLOCK_NY * BLOCK_NZ;
constexpr int GHOST_FACE_NODES_Y = BLOCK_NX * BLOCK_NZ;
constexpr int GHOST_FACE_NODES_Z = BLOCK_NX * BLOCK_NY;

// Total elements per face (nodes * populations per face)
constexpr int GHOST_FACE_SIZE_X = GHOST_FACE_NODES_X * QF;
constexpr int GHOST_FACE_SIZE_Y = GHOST_FACE_NODES_Y * QF;
constexpr int GHOST_FACE_SIZE_Z = GHOST_FACE_NODES_Z * QF;

/* ======================== STAGING OFFSETS ======================== */

// Offsets into s_pop for each face's staging region
constexpr int GHOST_OFF_W = 0;                                            // West  (X_1, -X)
constexpr int GHOST_OFF_E = GHOST_FACE_SIZE_X;                            // East  (X_0, +X)
constexpr int GHOST_OFF_S = GHOST_FACE_SIZE_X * 2;                        // South (Y_1, -Y)
constexpr int GHOST_OFF_N = GHOST_FACE_SIZE_X * 2 + GHOST_FACE_SIZE_Y;    // North (Y_0, +Y)
constexpr int GHOST_OFF_B = GHOST_FACE_SIZE_X * 2 + GHOST_FACE_SIZE_Y * 2;            // Back  (Z_1, -Z)
constexpr int GHOST_OFF_F = GHOST_FACE_SIZE_X * 2 + GHOST_FACE_SIZE_Y * 2 + GHOST_FACE_SIZE_Z;  // Front (Z_0, +Z)

// Total staging buffer size (must fit within s_pop)
constexpr int GHOST_STAGING_TOTAL = GHOST_FACE_SIZE_X * 2 + GHOST_FACE_SIZE_Y * 2 + GHOST_FACE_SIZE_Z * 2;

// Compile-time check: staging must fit in s_pop
// s_pop size = BLOCK_LBM_SIZE * (Q-1) = 512 * 18 (D3Q19) = 9216 dfloats
static_assert(GHOST_STAGING_TOTAL <= BLOCK_LBM_SIZE * (Q - 1),
    "Ghost staging buffer exceeds s_pop size");

/* ======================== INDEX MACROS ======================== */

// Face-local flat index matching the idxPopX/Y/Z global layout.
//   idxPopX(ty, tz, pop, bx, by, bz) = ty + BLOCK_NY * (tz + BLOCK_NZ * (pop + QF * ...))
//   The face-local part is:           ty + BLOCK_NY * (tz + BLOCK_NZ * pop)
// These macros compute that face-local index for use as s_pop offset within a face section.

#define GHOST_LOCAL_X(ty, tz, pop) ((ty) + BLOCK_NY * ((tz) + BLOCK_NZ * (pop)))
#define GHOST_LOCAL_Y(tx, tz, pop) ((tx) + BLOCK_NX * ((tz) + BLOCK_NZ * (pop)))
#define GHOST_LOCAL_Z(tx, ty, pop) ((tx) + BLOCK_NX * ((ty) + BLOCK_NY * (pop)))

/* ======================== BLOCK-LEVEL GHOST WRITE CONDITIONS ======================== */

#ifdef BC_X_PERIODIC
    #define GHOST_WRITE_WEST true
    #define GHOST_WRITE_EAST true
#elif defined(BC_X_WALL)
    #define GHOST_WRITE_WEST (bx > 0)
    #define GHOST_WRITE_EAST (bx < (int)(NUM_BLOCK_X) - 1)
#endif

#ifdef BC_Y_PERIODIC
    #define GHOST_WRITE_SOUTH true
    #define GHOST_WRITE_NORTH true
#elif defined(BC_Y_WALL)
    #define GHOST_WRITE_SOUTH (by > 0)
    #define GHOST_WRITE_NORTH (by < (int)(NUM_BLOCK_Y) - 1)
#endif

#ifdef BC_Z_PERIODIC
    #define GHOST_WRITE_BACK true
    #define GHOST_WRITE_FRONT true
#elif defined(BC_Z_WALL)
    #define GHOST_WRITE_BACK (bz > 0)
    #define GHOST_WRITE_FRONT (bz < (int)(NUM_BLOCK_Z) - 1)
#endif

#endif //__GHOST_STAGING_LAYOUT_INC
