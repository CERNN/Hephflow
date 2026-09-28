{
    const dfloat ductDx = (dfloat)x - DUCT_CENTER_X;
    const dfloat ductDy = (dfloat)y - DUCT_CENTER_Y;
    const dfloat ellipticRadiusSquared =
        ductDx * ductDx / (DUCT_RADIUS_A * DUCT_RADIUS_A) +
        ductDy * ductDy / (DUCT_RADIUS_B * DUCT_RADIUS_B);
    if (ellipticRadiusSquared >= 1.0_df) {
        L_Fz -= DUCT_DRIVING_FORCE;
    }
}
