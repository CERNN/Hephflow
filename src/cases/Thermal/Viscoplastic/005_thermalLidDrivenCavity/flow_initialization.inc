
rho = RHO_0;
ux = U_0_X;
uy = U_0_Y;
uz = U_0_Z;

#ifdef OMEGA_FIELD
omega = OMEGA;
#endif

#ifdef THERMAL_MODEL
cVar = T_HOT + (dfloat(x)/dfloat(NX-1))*(T_COLD - T_HOT);
#endif