#include "curvedBC.cuh"

#ifdef CURVED_BOUNDARY_CONDITION

__host__ __device__
dfloat curvedBoundaryExtrapolation(dfloat delta, dfloat deltaR, dfloat wallValue, dfloat firstFluidValue, dfloat secondFluidValue)
{
    const dfloat inverseDeltaRSquared = 1.0_df / (deltaR * deltaR);
    const dfloat wallWeight = (2.0_df * deltaR * deltaR - delta * delta + 3.0_df * delta * deltaR) * 0.5_df * inverseDeltaRSquared;
    const dfloat firstFluidWeight = delta * (delta - 2.0_df * deltaR) * inverseDeltaRSquared;
    const dfloat secondFluidWeight = -delta * (delta - deltaR) * 0.5_df * inverseDeltaRSquared;
    return wallWeight * wallValue + firstFluidWeight * firstFluidValue + secondFluidWeight * secondFluidValue;
}

static __device__ __forceinline__
dfloat interpolateCurvedBoundaryMoment(
    dfloat3 point,
    int moment,
    dfloat* moments,
    bool hasConstantX,
    bool hasConstantY,
    bool hasConstantZ,
    int constantX,
    int constantY,
    int constantZ)
{
    if (hasConstantZ) {
        return mom_bilinear_interp_xy(point.x, point.y, constantZ, moment, moments);
    } else if (hasConstantY) {
        return mom_bilinear_interp_xz(point.x, constantY, point.z, moment, moments);
    } else if (hasConstantX) {
        return mom_bilinear_interp_yz(constantX, point.y, point.z, moment, moments);
    }
    return mom_trilinear_interp(point.x, point.y, point.z, moment, moments);
}

static __device__ __forceinline__
void updateCurvedBoundaryVelocity(dfloat delta, dfloat deltaR, dfloat3 firstFluidPoint, dfloat3 secondFluidPoint, dfloat* moments, CurvedBoundary* curvedBoundary)
{
    constexpr dfloat EPSILON = 1e-6_df;
    const bool hasConstantX = fabs(firstFluidPoint.x - secondFluidPoint.x) < EPSILON;
    const bool hasConstantY = fabs(firstFluidPoint.y - secondFluidPoint.y) < EPSILON;
    const bool hasConstantZ = fabs(firstFluidPoint.z - secondFluidPoint.z) < EPSILON;
    const int constantX = (int)floor(firstFluidPoint.x);
    const int constantY = (int)floor(firstFluidPoint.y);
    const int constantZ = (int)floor(firstFluidPoint.z);

    dfloat firstValue = interpolateCurvedBoundaryMoment(firstFluidPoint, M_UX_INDEX, moments, hasConstantX, hasConstantY, hasConstantZ, constantX, constantY, constantZ) / F_M_I_SCALE;
    dfloat secondValue = interpolateCurvedBoundaryMoment(secondFluidPoint, M_UX_INDEX, moments, hasConstantX, hasConstantY, hasConstantZ, constantX, constantY, constantZ) / F_M_I_SCALE;
    const dfloat velocityX = curvedBoundaryExtrapolation(delta, deltaR, curvedBoundary->wallVel.x, firstValue, secondValue);

    firstValue = interpolateCurvedBoundaryMoment(firstFluidPoint, M_UY_INDEX, moments, hasConstantX, hasConstantY, hasConstantZ, constantX, constantY, constantZ) / F_M_I_SCALE;
    secondValue = interpolateCurvedBoundaryMoment(secondFluidPoint, M_UY_INDEX, moments, hasConstantX, hasConstantY, hasConstantZ, constantX, constantY, constantZ) / F_M_I_SCALE;
    const dfloat velocityY = curvedBoundaryExtrapolation(delta, deltaR, curvedBoundary->wallVel.y, firstValue, secondValue);

    firstValue = interpolateCurvedBoundaryMoment(firstFluidPoint, M_UZ_INDEX, moments, hasConstantX, hasConstantY, hasConstantZ, constantX, constantY, constantZ) / F_M_I_SCALE;
    secondValue = interpolateCurvedBoundaryMoment(secondFluidPoint, M_UZ_INDEX, moments, hasConstantX, hasConstantY, hasConstantZ, constantX, constantY, constantZ) / F_M_I_SCALE;
    const dfloat velocityZ = curvedBoundaryExtrapolation(delta, deltaR, curvedBoundary->wallVel.z, firstValue, secondValue);

    curvedBoundary->vel = dfloat3(velocityX, velocityY, velocityZ);
}

__global__
void updateCurvedBoundaryVelocities(CurvedBoundary* curvedBoundaryArray, dfloat* moments, unsigned int numberCurvedBoundaryNodes)
{
    const int index = threadIdx.x + blockDim.x * blockIdx.x;
    if (index >= numberCurvedBoundaryNodes) {
        return;
    }

    CurvedBoundary* curvedBoundary = &curvedBoundaryArray[index];
    updateCurvedBoundaryVelocity(curvedBoundary->delta, curvedBoundary->delta_r, curvedBoundary->pf1, curvedBoundary->pf2, moments, curvedBoundary);
}

#if defined(CONFORMATION_TENSOR) && defined(D3G19)
__global__
void updateCurvedBoundaryConformation(CurvedBoundary* curvedBoundaryArray, dfloat* moments, unsigned int numberCurvedBoundaryNodes)
{
    const int index = threadIdx.x + blockDim.x * blockIdx.x;
    if (index >= numberCurvedBoundaryNodes) {
        return;
    }

    CurvedBoundary* curvedBoundary = &curvedBoundaryArray[index];
    #ifndef CURVED_CONF_COUPLED_MOMENT_BC
        const dfloat delta = curvedBoundary->delta;
        const dfloat deltaR = curvedBoundary->delta_r;
        const dfloat inverseTwoDeltaRSquared = 0.5_df / (deltaR * deltaR);
        const dfloat firstWeight = (delta - 2.0_df * deltaR) * (delta - 3.0_df * deltaR) * inverseTwoDeltaRSquared;
        const dfloat secondWeight = -(delta - deltaR) * (delta - 3.0_df * deltaR) / (deltaR * deltaR);
        const dfloat thirdWeight = (delta - deltaR) * (delta - 2.0_df * deltaR) * inverseTwoDeltaRSquared;
    #endif // CURVED_CONF_COUPLED_MOMENT_BC
    const int z = (int)floor(curvedBoundary->pf1.z);
    const int scalarMoments[6] = {
        A_XX_C_INDEX, A_XY_C_INDEX, A_XZ_C_INDEX,
        A_YY_C_INDEX, A_YZ_C_INDEX, A_ZZ_C_INDEX
    };
    #ifdef CURVED_CONF_COUPLED_MOMENT_BC
        const int fluxMoments[18] = {
            A_XX_CX_INDEX, A_XX_CY_INDEX, A_XX_CZ_INDEX,
            A_XY_CX_INDEX, A_XY_CY_INDEX, A_XY_CZ_INDEX,
            A_XZ_CX_INDEX, A_XZ_CY_INDEX, A_XZ_CZ_INDEX,
            A_YY_CX_INDEX, A_YY_CY_INDEX, A_YY_CZ_INDEX,
            A_YZ_CX_INDEX, A_YZ_CY_INDEX, A_YZ_CZ_INDEX,
            A_ZZ_CX_INDEX, A_ZZ_CY_INDEX, A_ZZ_CZ_INDEX
        };

        curvedBoundary->conformationInteriorVelocityT30[0] = mom_bilinear_interp_xy(curvedBoundary->pf1.x, curvedBoundary->pf1.y, z, M_UX_INDEX, moments);
        curvedBoundary->conformationInteriorVelocityT30[1] = mom_bilinear_interp_xy(curvedBoundary->pf1.x, curvedBoundary->pf1.y, z, M_UY_INDEX, moments);
        curvedBoundary->conformationInteriorVelocityT30[2] = mom_bilinear_interp_xy(curvedBoundary->pf1.x, curvedBoundary->pf1.y, z, M_UZ_INDEX, moments);
    #endif // CURVED_CONF_COUPLED_MOMENT_BC

    #pragma unroll
    for (int component = 0; component < 6; ++component) {
        const int moment = scalarMoments[component];
        const dfloat firstValue = mom_bilinear_interp_xy(curvedBoundary->pf1.x, curvedBoundary->pf1.y, z, moment, moments);
        #ifdef CURVED_CONF_COUPLED_MOMENT_BC
            // The nearest-interior interpolation is convex and avoids the
            // negative-weight overshoot of quadratic extrapolation.
            curvedBoundary->conformation[component] = firstValue;
        #else
            const dfloat secondValue = mom_bilinear_interp_xy(curvedBoundary->pf2.x, curvedBoundary->pf2.y, z, moment, moments);
            const dfloat thirdValue = mom_bilinear_interp_xy(curvedBoundary->pf3.x, curvedBoundary->pf3.y, z, moment, moments);
            curvedBoundary->conformation[component] = firstWeight * firstValue + secondWeight * secondValue + thirdWeight * thirdValue;
        #endif // CURVED_CONF_COUPLED_MOMENT_BC

        #ifdef CURVED_CONF_COUPLED_MOMENT_BC
            #pragma unroll
            for (int direction = 0; direction < 3; ++direction) {
                curvedBoundary->conformationFluxT30[3 * component + direction] = mom_bilinear_interp_xy(
                    curvedBoundary->pf1.x,
                    curvedBoundary->pf1.y,
                    z,
                    fluxMoments[3 * component + direction],
                    moments);
            }
        #endif // CURVED_CONF_COUPLED_MOMENT_BC
    }
}
#endif // defined(CONFORMATION_TENSOR) && defined(D3G19)

#endif // CURVED_BOUNDARY_CONDITION
