#ifdef VISCOELASTIC_STARTUP_RAMP
// Smoothstep startup over five polymer relaxation times. The same factor is
// applied to polymer feedback in conformation_evolution.inc.
const dfloat forceRampCoordinate = fmin(
    1.0_df, (dfloat)step / (dfloat)STARTUP_RAMP_STEPS);
const dfloat viscoelasticStartupRamp = forceRampCoordinate * forceRampCoordinate
                                     * (3.0_df - 2.0_df * forceRampCoordinate);
L_Fx *= viscoelasticStartupRamp;
L_Fy *= viscoelasticStartupRamp;
L_Fz *= viscoelasticStartupRamp;
#endif
