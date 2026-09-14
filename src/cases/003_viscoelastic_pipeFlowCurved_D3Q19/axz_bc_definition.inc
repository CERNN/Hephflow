CurvedBoundary* conformationCurvedBC = d_curvedBC[
    idxScalarBlock(threadIdx.x, threadIdx.y, threadIdx.z,
                   blockIdx.x, blockIdx.y, blockIdx.z)];

if (conformationCurvedBC != nullptr) {
    const CurvedScalarBoundaryResult curvedResult =
        curvedCircularConformationBoundaryD3G19(
            gNode, conformationCurvedBC->conformation[CURVED_CONF_XZ], Gxz,
            x, y, DUCT_CENTER_X, DUCT_CENTER_Y, OUTER_RADIUS,
            ux_t30, uy_t30, uz_t30,
            &conformationCurvedBC->conformationFluxT30[3 * CURVED_CONF_XZ],
            conformationCurvedBC->conformationInteriorVelocityT30);

    AxzVar = curvedResult.value;
    Axz_qx_t30 = curvedResult.qxRaw;
    Axz_qy_t30 = curvedResult.qyRaw;
    Axz_qz_t30 = curvedResult.qzRaw;
}
