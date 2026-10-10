/* Oldroyd-B circular duct using the same stationary IBM geometry as the baseline. */
#define PARTICLE_MODEL
#define VISCOELASTIC_MODEL
#define D3Q19
#define COLLISION_TYPE MR_LBM_2ND_ORDER

#define FORCE_FIELD_INCLUDE
#define TREAT_DATA_INCLUDE

#define BREUGEM_PARAMETER (0.0_df)
#define MESH_SCALE 1.0_df
#define MESH_COULOMB 0
#define IBM_THICKNESS 1.0_df
#define STENCIL_4
#define IBM_MAX_ITERATION 3
#define IBM_FORCE_RELAXATION 1.0_df
#define IBM_VELOCITY_TOL 1.0e-8_df
