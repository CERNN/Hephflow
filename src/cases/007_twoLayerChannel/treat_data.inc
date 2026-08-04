int x0 = NX/2;
int z0 = NZ_TOTAL/2;

velocityProfile(fMom, 3, x0, 0, z0, step);
phiProfile(fMom, 2, x0, 0, z0, step);

#ifdef SAVE_LOCAL_FORCES
forceProfile(d_Local_Fx, d_Local_Fy, d_Local_Fz, 1, x0, 0, z0, step);
#ifdef PHI_DIST
sourceProfile(d_Source_Phi, "source_Phi", 1, x0, 0, z0, step);
#endif
#ifdef LAMBDA_DIST
sourceProfile(d_Source_Lambda, "source_Lambda", 1, x0, 0, z0, step);
#endif
#ifdef CONFORMATION_TENSOR
#ifdef A_XX_DIST
sourceProfile(d_Source_Gxx, "source_Gxx", 1, x0, 0, z0, step);
#endif
#ifdef A_XY_DIST
sourceProfile(d_Source_Gxy, "source_Gxy", 1, x0, 0, z0, step);
#endif
#ifdef A_XZ_DIST
sourceProfile(d_Source_Gxz, "source_Gxz", 1, x0, 0, z0, step);
#endif
#ifdef A_YY_DIST
sourceProfile(d_Source_Gyy, "source_Gyy", 1, x0, 0, z0, step);
#endif
#ifdef A_YZ_DIST
sourceProfile(d_Source_Gyz, "source_Gyz", 1, x0, 0, z0, step);
#endif
#ifdef A_ZZ_DIST
sourceProfile(d_Source_Gzz, "source_Gzz", 1, x0, 0, z0, step);
#endif
#endif //CONFORMATION_TENSOR
#endif //SAVE_LOCAL_FORCES








