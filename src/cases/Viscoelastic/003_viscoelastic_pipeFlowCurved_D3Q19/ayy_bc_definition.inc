CurvedBoundary* conformationCurvedBC = d_curvedBC[
    idxScalarBlock(threadIdx.x, threadIdx.y, threadIdx.z,
                   blockIdx.x, blockIdx.y, blockIdx.z)];

if (conformationCurvedBC != nullptr) {
    const CurvedScalarBoundaryResult curvedResult =
        curvedCircularConformationBoundaryD3G19(
            gNode, conformationCurvedBC->conformation[CURVED_CONF_YY], Gyy,
            x, y, DUCT_CENTER_X, DUCT_CENTER_Y, OUTER_RADIUS,
            ux_t30, uy_t30, uz_t30,
            &conformationCurvedBC->conformationFluxT30[3 * CURVED_CONF_YY],
            conformationCurvedBC->conformationInteriorVelocityT30);

    AyyVar = curvedResult.value;
    Ayy_qx_t30 = curvedResult.qxRaw;
    Ayy_qy_t30 = curvedResult.qyRaw;
    Ayy_qz_t30 = curvedResult.qzRaw;
}
