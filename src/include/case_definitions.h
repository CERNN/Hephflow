/**
 *  @file case_definitions.h
 *  @brief Case file path definitions and configuration macros
 *  @version 0.4.0
 *  @date 27/12/2025
 */

#ifndef __CASE_DEFINITIONS_H
#define __CASE_DEFINITIONS_H

/* ======================== STRING CONVERSION MACROS ====================== */

#define STR_IMPL(A) #A
#define STR(A) STR_IMPL(A)

/* ========================= CASE FILE DEFINITIONS ========================= */

// Main case files
#define CASE_DIRECTORY cases
#define CASE_CONSTANTS STR(CASE_DIRECTORY/BC_PROBLEM/constants.inc)
#define CASE_OUTPUTS STR(CASE_DIRECTORY/BC_PROBLEM/output.inc)
#define CASE_MODEL STR(CASE_DIRECTORY/BC_PROBLEM/model.inc)
#define CASE_BC_INIT STR(CASE_DIRECTORY/BC_PROBLEM/bc_initialization.inc)
#define CASE_BC_DEF STR(CASE_DIRECTORY/BC_PROBLEM/bc_definition.inc)
#define CASE_FLOW_INITIALIZATION STR(CASE_DIRECTORY/BC_PROBLEM/flow_initialization.inc)
#define CASE_TREAT_DATA STR(CASE_DIRECTORY/BC_PROBLEM/treat_data.inc)
#define CASE_CURVED_BC_DEF STR(CASE_DIRECTORY/BC_PROBLEM/curved_bc_definition.inc)
#define VOXEL_BC_DEFINE STR(../../CASE_DIRECTORY/voxel/bc_definition.inc)

// Collision-reconstruction files
#define COLREC_DIRECTORY colrec
#define COLREC_COLLISION STR(COLREC_DIRECTORY/COLLISION_TYPE/collision.inc)
#define COLREC_RECONSTRUCTION STR(COLREC_DIRECTORY/COLLISION_TYPE/reconstruction.inc)

// G scalar field definitions
#define CASE_G_BC_DEF STR(CASE_DIRECTORY/BC_PROBLEM/g_bc_definition.inc)
#define COLREC_G_RECONSTRUCTION STR(COLREC_DIRECTORY/G_SCALAR/reconstruction.inc)
#define COLREC_G_COLLISION STR(COLREC_DIRECTORY/G_SCALAR/collision.inc)

// Phi scalar field definitions
#define CASE_PHI_BC_DEF STR(CASE_DIRECTORY/BC_PROBLEM/phi_bc_definition.inc)
#define COLREC_PHI_RECONSTRUCTION STR(COLREC_DIRECTORY/PHI_SCALAR/reconstruction.inc)
#define COLREC_PHI_COLLISION STR(COLREC_DIRECTORY/PHI_SCALAR/collision.inc)

// Lambda scalar field definitions
#define CASE_LAMBDA_BC_DEF STR(CASE_DIRECTORY/BC_PROBLEM/lambda_bc_definition.inc)
#define COLREC_LAMBDA_RECONSTRUCTION STR(COLREC_DIRECTORY/LAMBDA_SCALAR/reconstruction.inc)
#define COLREC_LAMBDA_COLLISION STR(COLREC_DIRECTORY/LAMBDA_SCALAR/collision.inc)

// Aij tensor field definitions (stress tensor components)
#define CASE_AXX_BC_DEF STR(CASE_DIRECTORY/BC_PROBLEM/axx_bc_definition.inc)
#define COLREC_AXX_RECONSTRUCTION STR(COLREC_DIRECTORY/AIJ_SCALAR/reconstruction_xx.inc)
#define COLREC_AXX_COLLISION STR(COLREC_DIRECTORY/AIJ_SCALAR/collision_xx.inc)

#define CASE_AXY_BC_DEF STR(CASE_DIRECTORY/BC_PROBLEM/axy_bc_definition.inc)
#define COLREC_AXY_RECONSTRUCTION STR(COLREC_DIRECTORY/AIJ_SCALAR/reconstruction_xy.inc)
#define COLREC_AXY_COLLISION STR(COLREC_DIRECTORY/AIJ_SCALAR/collision_xy.inc)

#define CASE_AXZ_BC_DEF STR(CASE_DIRECTORY/BC_PROBLEM/axz_bc_definition.inc)
#define COLREC_AXZ_RECONSTRUCTION STR(COLREC_DIRECTORY/AIJ_SCALAR/reconstruction_xz.inc)
#define COLREC_AXZ_COLLISION STR(COLREC_DIRECTORY/AIJ_SCALAR/collision_xz.inc)

#define CASE_AYY_BC_DEF STR(CASE_DIRECTORY/BC_PROBLEM/ayy_bc_definition.inc)
#define COLREC_AYY_RECONSTRUCTION STR(COLREC_DIRECTORY/AIJ_SCALAR/reconstruction_yy.inc)
#define COLREC_AYY_COLLISION STR(COLREC_DIRECTORY/AIJ_SCALAR/collision_yy.inc)

#define CASE_AYZ_BC_DEF STR(CASE_DIRECTORY/BC_PROBLEM/ayz_bc_definition.inc)
#define COLREC_AYZ_RECONSTRUCTION STR(COLREC_DIRECTORY/AIJ_SCALAR/reconstruction_yz.inc)
#define COLREC_AYZ_COLLISION STR(COLREC_DIRECTORY/AIJ_SCALAR/collision_yz.inc)

#define CASE_AZZ_BC_DEF STR(CASE_DIRECTORY/BC_PROBLEM/azz_bc_definition.inc)
#define COLREC_AZZ_RECONSTRUCTION STR(COLREC_DIRECTORY/AIJ_SCALAR/reconstruction_zz.inc)
#define COLREC_AZZ_COLLISION STR(COLREC_DIRECTORY/AIJ_SCALAR/collision_zz.inc)

// Particle creation
#define CASE_PARTICLE_CREATE STR(../../CASE_DIRECTORY/BC_PROBLEM/particleCreation.inc)

// Physics regression test metric
#define CASE_TEST_METRIC STR(CASE_DIRECTORY/BC_PROBLEM/_test/test_metric.inc)

#endif //__CASE_DEFINITIONS_H