CurvedBoundary* conformationCurvedBC = d_curvedBC[
    idxScalarBlock(threadIdx.x, threadIdx.y, threadIdx.z,
                   blockIdx.x, blockIdx.y, blockIdx.z)];

if (conformationCurvedBC != nullptr) {
    const CurvedScalarBoundaryResult curvedResult =
        curvedCircularConformationBoundaryD3G19(
            gNode, conformationCurvedBC->conformation[CURVED_CONF_ZZ], Gzz,
            x, y, DUCT_CENTER_X, DUCT_CENTER_Y, OUTER_RADIUS,
            ux_t30, uy_t30, uz_t30,
            &conformationCurvedBC->conformationFluxT30[3 * CURVED_CONF_ZZ],
            conformationCurvedBC->conformationInteriorVelocityT30);

    AzzVar = curvedResult.value;
    Azz_qx_t30 = curvedResult.qxRaw;
    Azz_qy_t30 = curvedResult.qyRaw;
    Azz_qz_t30 = curvedResult.qzRaw;
}
