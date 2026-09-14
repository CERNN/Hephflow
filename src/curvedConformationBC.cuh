#ifndef __CURVED_CONFORMATION_BC_CUH
#define __CURVED_CONFORMATION_BC_CUH

#include "var.h"
#include "globalStructs.h"
#include "globalFunctions.h"

#if defined(CURVED_BOUNDARY_CONDITION) && defined(CONFORMATION_TENSOR) && defined(D3G19)

struct CurvedScalarBoundaryResult {
    dfloat value;
    // First moments computed directly from the populations. These are raw
    // lattice moments. The common conformation path in mlbm.cu converts them
    // to the stored t30 representation exactly once with F_M_I_SCALE.
    dfloat qxRaw;
    dfloat qyRaw;
    dfloat qzRaw;
};

enum CurvedConformationComponent {
    CURVED_CONF_XX = 0,
    CURVED_CONF_XY,
    CURVED_CONF_XZ,
    CURVED_CONF_YY,
    CURVED_CONF_YZ,
    CURVED_CONF_ZZ
};

/**
 * Reconstruct one D3G19 conformation component at a curved, z-aligned
 * circular-duct boundary. The coupled mode prescribes a consistent scalar
 * and flux state from a stable pre-collision snapshot. The fallback mode uses
 * the original population anti-bounce-back reconstruction.
 */
__device__ inline
CurvedScalarBoundaryResult curvedCircularConformationBoundaryD3G19(
    dfloat gNode[GQ],
    dfloat boundaryValue,
    dfloat sourceTerm,
    int x,
    int y,
    dfloat centerX,
    dfloat centerY,
    dfloat radius,
    dfloat velocityXT30,
    dfloat velocityYT30,
    dfloat velocityZT30,
    const dfloat interiorFluxT30[3],
    const dfloat interiorVelocityT30[3])
{
#ifdef CURVED_CONF_COUPLED_MOMENT_BC
    CurvedScalarBoundaryResult result{};
    result.value = boundaryValue + sourceTerm;

    // Work entirely in the stored t30 representation until the final handoff.
    // Preserve the nearest-interior tangential non-equilibrium moment, remove
    // its wall-normal part, and rebuild the equilibrium part from the boundary
    // scalar and local velocity. The shared mlbm path then converts qRaw back
    // to t30 exactly once before collision.
    dfloat qNonEqX = interiorFluxT30[0]
                   - boundaryValue * interiorVelocityT30[0];
    dfloat qNonEqY = interiorFluxT30[1]
                   - boundaryValue * interiorVelocityT30[1];
    const dfloat qNonEqZ = interiorFluxT30[2]
                         - boundaryValue * interiorVelocityT30[2];
#ifdef CURVED_CONF_ZERO_NORMAL_NON_EQ_FLUX
    const dfloat nodeDx = (dfloat)x - centerX;
    const dfloat nodeDy = (dfloat)y - centerY;
    const dfloat invNodeRadius = 1.0_df / sqrt(nodeDx * nodeDx + nodeDy * nodeDy);
    const dfloat normalX = nodeDx * invNodeRadius;
    const dfloat normalY = nodeDy * invNodeRadius;
    const dfloat qNonEqNormal = normalX * qNonEqX + normalY * qNonEqY;
    qNonEqX -= normalX * qNonEqNormal;
    qNonEqY -= normalY * qNonEqNormal;
#endif

    result.qxRaw = (result.value * velocityXT30 + qNonEqX) / F_M_I_SCALE;
    result.qyRaw = (result.value * velocityYT30 + qNonEqY) / F_M_I_SCALE;
    result.qzRaw = (result.value * velocityZT30 + qNonEqZ) / F_M_I_SCALE;
    return result;
#else
    const dfloat radiusSq = radius * radius;
    for (int i = 1; i < GQ; ++i) {
        // Pull streaming reads population i from x-c_i. The pipe is extruded
        // along z, so solidity depends only on the source x/y coordinates.
        const dfloat sourceX = (dfloat)x - (dfloat)gcx[i];
        const dfloat sourceY = (dfloat)y - (dfloat)gcy[i];
        const dfloat dx = sourceX - centerX;
        const dfloat dy = sourceY - centerY;
        if (dx * dx + dy * dy >= radiusSq) {
            const int opposite = (i & 1) ? i + 1 : i - 1;
            gNode[i] = -gNode[opposite] + 2.0_df * gw[i] * boundaryValue;
        }
    }

    CurvedScalarBoundaryResult result{};
    for (int i = 0; i < GQ; ++i) {
        result.value += gNode[i];
        result.qxRaw += (dfloat)gcx[i] * gNode[i];
        result.qyRaw += (dfloat)gcy[i] * gNode[i];
        result.qzRaw += (dfloat)gcz[i] * gNode[i];
    }
    result.value += sourceTerm;

#ifdef CURVED_CONF_ZERO_NORMAL_NON_EQ_FLUX
    // qRaw is converted below the case include to q_t30 = F_M_I_SCALE*qRaw.
    // Therefore its advective equilibrium is A*u (not F_M_I_SCALE*A*u).
    // Remove only the non-equilibrium normal part and retain the tangential
    // population moment. This is the impermeable-wall/no-diffusive-flux test.
    const dfloat nodeDx = (dfloat)x - centerX;
    const dfloat nodeDy = (dfloat)y - centerY;
    const dfloat invNodeRadius = 1.0_df / sqrt(nodeDx * nodeDx + nodeDy * nodeDy);
    const dfloat normalX = nodeDx * invNodeRadius;
    const dfloat normalY = nodeDy * invNodeRadius;
    const dfloat normalFluxRaw = normalX * result.qxRaw + normalY * result.qyRaw;
    const dfloat normalVelocity =
        (normalX * velocityXT30 + normalY * velocityYT30) / F_M_I_SCALE;
    const dfloat normalEquilibriumFluxRaw = result.value * normalVelocity;
    const dfloat normalCorrection = normalEquilibriumFluxRaw - normalFluxRaw;
    result.qxRaw += normalX * normalCorrection;
    result.qyRaw += normalY * normalCorrection;
#endif

    return result;
#endif
}

#endif // CURVED_BOUNDARY_CONDITION && CONFORMATION_TENSOR && D3G19
#endif // __CURVED_CONFORMATION_BC_CUH
