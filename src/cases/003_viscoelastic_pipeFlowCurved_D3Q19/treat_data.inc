// Steady Oldroyd-B circular-Poiseuille validation at one z plane.
auto getPipeMoment = [&](int px, int py, int pz, int moment) {
    return h_fMom[idxMom(px % BLOCK_NX, py % BLOCK_NY, pz % BLOCK_NZ,
                         moment, px / BLOCK_NX, py / BLOCK_NY, pz / BLOCK_NZ)];
};
auto getPipeConformation = [&](int px, int py, int pz, int moment) {
    return getPipeMoment(px, py, pz, moment) - CONF_ZERO;
};

const int validationZ = NZ_TOTAL / 2;
const dfloat radiusSq = OUTER_RADIUS * OUTER_RADIUS;
const dfloat invRadiusSq = 1.0_df / radiusSq;

dfloat velocityNumerator = 0.0_df;
dfloat velocityDenominator = 0.0_df;
dfloat axzNumerator = 0.0_df;
dfloat axzDenominator = 0.0_df;
dfloat ayzNumerator = 0.0_df;
dfloat ayzDenominator = 0.0_df;
dfloat azzNumerator = 0.0_df;
dfloat azzDenominator = 0.0_df;
dfloat identityErrorSq = 0.0_df;
dfloat azimuthNumerator = 0.0_df;
dfloat azimuthDenominator = 0.0_df;
dfloat minimumPrincipalMinor = 1.0e30_df;
size_t validationCount = 0;
size_t azimuthCount = 0;

for (int py = 0; py < NY; ++py) {
    for (int px = 0; px < NX; ++px) {
        const dfloat dx = (dfloat)px - DUCT_CENTER_X;
        const dfloat dy = (dfloat)py - DUCT_CENTER_Y;
        const dfloat rSq = dx * dx + dy * dy;
        if (rSq >= radiusSq) continue;

        const dfloat uz = getPipeMoment(px, py, validationZ, M_UZ_INDEX) / F_M_I_SCALE;
        const dfloat Axx = getPipeConformation(px, py, validationZ, A_XX_C_INDEX);
        const dfloat Axy = getPipeConformation(px, py, validationZ, A_XY_C_INDEX);
        const dfloat Axz = getPipeConformation(px, py, validationZ, A_XZ_C_INDEX);
        const dfloat Ayy = getPipeConformation(px, py, validationZ, A_YY_C_INDEX);
        const dfloat Ayz = getPipeConformation(px, py, validationZ, A_YZ_C_INDEX);
        const dfloat Azz = getPipeConformation(px, py, validationZ, A_ZZ_C_INDEX);

        const dfloat duzdx = -2.0_df * U_C * dx * invRadiusSq;
        const dfloat duzdy = -2.0_df * U_C * dy * invRadiusSq;
        const dfloat uzAnalytical = U_C * (1.0_df - rSq * invRadiusSq);
        const dfloat AxzAnalytical = LAMBDA * duzdx;
        const dfloat AyzAnalytical = LAMBDA * duzdy;
        const dfloat AzzAnalytical = 1.0_df + 2.0_df * LAMBDA * LAMBDA
            * (duzdx * duzdx + duzdy * duzdy);

        const dfloat duz = uz - uzAnalytical;
        const dfloat dAxz = Axz - AxzAnalytical;
        const dfloat dAyz = Ayz - AyzAnalytical;
        const dfloat dAzz = Azz - AzzAnalytical;
        velocityNumerator += duz * duz;
        velocityDenominator += uzAnalytical * uzAnalytical;
        axzNumerator += dAxz * dAxz;
        axzDenominator += AxzAnalytical * AxzAnalytical;
        ayzNumerator += dAyz * dAyz;
        ayzDenominator += AyzAnalytical * AyzAnalytical;
        azzNumerator += dAzz * dAzz;
        azzDenominator += AzzAnalytical * AzzAnalytical;

        identityErrorSq += (Axx - 1.0_df) * (Axx - 1.0_df)
                         + (Ayy - 1.0_df) * (Ayy - 1.0_df)
                         + Axy * Axy;

        const dfloat leadingMinor2 = Axx * Ayy - Axy * Axy;
        const dfloat determinant =
            Axx * (Ayy * Azz - Ayz * Ayz)
          - Axy * (Axy * Azz - Ayz * Axz)
          + Axz * (Axy * Ayz - Ayy * Axz);
        dfloat localMinimum = Axx;
        if (leadingMinor2 < localMinimum) localMinimum = leadingMinor2;
        if (determinant < localMinimum) localMinimum = determinant;
        if (localMinimum < minimumPrincipalMinor) minimumPrincipalMinor = localMinimum;

        // Compare scalar fields after a 90-degree rotation around the pipe axis.
        const int rotatedX = (int)(DUCT_CENTER_X - dy);
        const int rotatedY = (int)(DUCT_CENTER_Y + dx);
        if (rotatedX >= 0 && rotatedX < NX && rotatedY >= 0 && rotatedY < NY) {
            const dfloat uzRotated = getPipeMoment(
                rotatedX, rotatedY, validationZ, M_UZ_INDEX) / F_M_I_SCALE;
            const dfloat AzzRotated = getPipeConformation(
                rotatedX, rotatedY, validationZ, A_ZZ_C_INDEX);
            const dfloat velocityDifference = uz - uzRotated;
            const dfloat conformationDifference = Azz - AzzRotated;
            azimuthNumerator += velocityDifference * velocityDifference
                              + conformationDifference * conformationDifference;
            azimuthDenominator += uz * uz + Azz * Azz;
            ++azimuthCount;
        }

        ++validationCount;
    }
}

const dfloat velocityL2 = sqrt(velocityNumerator / velocityDenominator);
const dfloat axzL2 = sqrt(axzNumerator / axzDenominator);
const dfloat ayzL2 = sqrt(ayzNumerator / ayzDenominator);
const dfloat azzL2 = sqrt(azzNumerator / azzDenominator);
const dfloat identityRms = sqrt(identityErrorSq / (3.0_df * (dfloat)validationCount));
const dfloat azimuthL2 = azimuthCount > 0
    ? sqrt(azimuthNumerator / azimuthDenominator)
    : 0.0_df;

std::ostringstream validationOutput;
validationOutput << std::scientific << std::setprecision(8)
    << step << "," << (dfloat)step / t_c << ","
    << velocityL2 << "," << axzL2 << "," << ayzL2 << "," << azzL2 << ","
    << identityRms << "," << azimuthL2 << "," << minimumPrincipalMinor;
saveTreatData("_oldroyd_b_pipe_validation", validationOutput.str(), step);

// Boundary-only diagnostics. Conformation fluxes are stored in fMom in the
// t30 representation, so divide them by F_M_I_SCALE exactly once when
// reporting physical lattice fluxes. The collision equilibrium is
// q_t30 = F_M_I_SCALE * A_stored * u; A_stored includes CONF_ZERO.
const int conformationMoments[6] = {
    A_XX_C_INDEX, A_XY_C_INDEX, A_XZ_C_INDEX,
    A_YY_C_INDEX, A_YZ_C_INDEX, A_ZZ_C_INDEX
};
const int conformationFluxXMoments[6] = {
    A_XX_CX_INDEX, A_XY_CX_INDEX, A_XZ_CX_INDEX,
    A_YY_CX_INDEX, A_YZ_CX_INDEX, A_ZZ_CX_INDEX
};
const int conformationFluxYMoments[6] = {
    A_XX_CY_INDEX, A_XY_CY_INDEX, A_XZ_CY_INDEX,
    A_YY_CY_INDEX, A_YZ_CY_INDEX, A_ZZ_CY_INDEX
};

dfloat boundaryMismatchSq[6] = {};
dfloat boundaryMismatchMax[6] = {};
dfloat boundaryNormalFluxSq[6] = {};
dfloat boundaryNormalFluxMax[6] = {};
dfloat boundaryNormalNonEqFluxSq[6] = {};
dfloat boundaryNormalNonEqFluxMax[6] = {};
dfloat minimumBoundaryTargetMinor = 1.0e30_df;
size_t boundaryValidationCount = 0;

auto bilinearPipeMoment = [&](dfloat sampleX, dfloat sampleY, int moment) {
    const int x0 = (int)floor(sampleX);
    const int y0 = (int)floor(sampleY);
    const int x1 = x0 + 1;
    const int y1 = y0 + 1;
    const dfloat tx = sampleX - (dfloat)x0;
    const dfloat ty = sampleY - (dfloat)y0;
    const dfloat v00 = getPipeMoment(x0, y0, validationZ, moment);
    const dfloat v10 = getPipeMoment(x1, y0, validationZ, moment);
    const dfloat v01 = getPipeMoment(x0, y1, validationZ, moment);
    const dfloat v11 = getPipeMoment(x1, y1, validationZ, moment);
    return (1.0_df - tx) * (1.0_df - ty) * v00
         + tx * (1.0_df - ty) * v10
         + (1.0_df - tx) * ty * v01
         + tx * ty * v11;
};

for (int py = 0; py < NY; ++py) {
    for (int px = 0; px < NX; ++px) {
        const dfloat dx = (dfloat)px - DUCT_CENTER_X;
        const dfloat dy = (dfloat)py - DUCT_CENTER_Y;
        const dfloat rSq = dx * dx + dy * dy;
        if (rSq >= radiusSq) continue;

        bool isBoundary = false;
        for (int sy = -1; sy <= 1 && !isBoundary; ++sy) {
            for (int sx = -1; sx <= 1; ++sx) {
                if (sx == 0 && sy == 0) continue;
                const dfloat sourceDx = dx - (dfloat)sx;
                const dfloat sourceDy = dy - (dfloat)sy;
                if (sourceDx * sourceDx + sourceDy * sourceDy >= radiusSq) {
                    isBoundary = true;
                    break;
                }
            }
        }
        if (!isBoundary) continue;

        const dfloat radius = sqrt(rSq);
        const dfloat normalX = dx / radius;
        const dfloat normalY = dy / radius;
        const dfloat wallX = DUCT_CENTER_X + normalX * OUTER_RADIUS;
        const dfloat wallY = DUCT_CENTER_Y + normalY * OUTER_RADIUS;
        const dfloat delta = OUTER_RADIUS - radius;
        const dfloat dr = sqrt(2.0_df);
        const dfloat pf1x = wallX - normalX * dr;
        const dfloat pf1y = wallY - normalY * dr;
        const dfloat pf2x = wallX - normalX * (2.0_df * dr);
        const dfloat pf2y = wallY - normalY * (2.0_df * dr);
        const dfloat pf3x = wallX - normalX * (3.0_df * dr);
        const dfloat pf3y = wallY - normalY * (3.0_df * dr);
        const dfloat w1 = (delta - 2.0_df * dr) * (delta - 3.0_df * dr)
                        / (2.0_df * dr * dr);
        const dfloat w2 = -(delta - dr) * (delta - 3.0_df * dr)
                        / (dr * dr);
        const dfloat w3 = (delta - dr) * (delta - 2.0_df * dr)
                        / (2.0_df * dr * dr);

        const dfloat uxT30 = getPipeMoment(px, py, validationZ, M_UX_INDEX);
        const dfloat uyT30 = getPipeMoment(px, py, validationZ, M_UY_INDEX);
        const dfloat normalVelocityT30 = normalX * uxT30 + normalY * uyT30;
        dfloat targetPhysical[6];

        for (int component = 0; component < 6; ++component) {
            const int scalarMoment = conformationMoments[component];
#ifdef CURVED_CONF_COUPLED_MOMENT_BC
            const dfloat targetStored =
                bilinearPipeMoment(pf1x, pf1y, scalarMoment);
#else
            const dfloat targetStored =
                w1 * bilinearPipeMoment(pf1x, pf1y, scalarMoment)
              + w2 * bilinearPipeMoment(pf2x, pf2y, scalarMoment)
              + w3 * bilinearPipeMoment(pf3x, pf3y, scalarMoment);
#endif
            const dfloat actualStored = getPipeMoment(
                px, py, validationZ, scalarMoment);
            const dfloat mismatch = actualStored - targetStored;
            const dfloat normalFluxT30 =
                normalX * getPipeMoment(px, py, validationZ,
                                        conformationFluxXMoments[component])
              + normalY * getPipeMoment(px, py, validationZ,
                                        conformationFluxYMoments[component]);
            const dfloat normalFlux = normalFluxT30 / F_M_I_SCALE;
            const dfloat normalNonEqFlux =
                (normalFluxT30 - actualStored * normalVelocityT30)
                / F_M_I_SCALE;

            boundaryMismatchSq[component] += mismatch * mismatch;
            boundaryNormalFluxSq[component] += normalFlux * normalFlux;
            boundaryNormalNonEqFluxSq[component] +=
                normalNonEqFlux * normalNonEqFlux;
            if (fabs(mismatch) > boundaryMismatchMax[component])
                boundaryMismatchMax[component] = fabs(mismatch);
            if (fabs(normalFlux) > boundaryNormalFluxMax[component])
                boundaryNormalFluxMax[component] = fabs(normalFlux);
            if (fabs(normalNonEqFlux) > boundaryNormalNonEqFluxMax[component])
                boundaryNormalNonEqFluxMax[component] = fabs(normalNonEqFlux);
            targetPhysical[component] = targetStored - CONF_ZERO;
        }

        const dfloat targetMinor2 = targetPhysical[0] * targetPhysical[3]
                                  - targetPhysical[1] * targetPhysical[1];
        const dfloat targetDeterminant =
            targetPhysical[0] * (targetPhysical[3] * targetPhysical[5]
                               - targetPhysical[4] * targetPhysical[4])
          - targetPhysical[1] * (targetPhysical[1] * targetPhysical[5]
                               - targetPhysical[4] * targetPhysical[2])
          + targetPhysical[2] * (targetPhysical[1] * targetPhysical[4]
                               - targetPhysical[3] * targetPhysical[2]);
        dfloat targetMinimum = targetPhysical[0];
        if (targetMinor2 < targetMinimum) targetMinimum = targetMinor2;
        if (targetDeterminant < targetMinimum) targetMinimum = targetDeterminant;
        if (targetMinimum < minimumBoundaryTargetMinor)
            minimumBoundaryTargetMinor = targetMinimum;
        ++boundaryValidationCount;
    }
}

std::ostringstream boundaryOutput;
boundaryOutput << std::scientific << std::setprecision(8)
    << step << "," << (dfloat)step / t_c << "," << boundaryValidationCount;
for (int component = 0; component < 6; ++component)
    boundaryOutput << "," << sqrt(boundaryMismatchSq[component]
                                      / (dfloat)boundaryValidationCount)
                   << "," << boundaryMismatchMax[component];
for (int component = 0; component < 6; ++component)
    boundaryOutput << "," << sqrt(boundaryNormalFluxSq[component]
                                      / (dfloat)boundaryValidationCount)
                   << "," << boundaryNormalFluxMax[component];
for (int component = 0; component < 6; ++component)
    boundaryOutput << "," << sqrt(boundaryNormalNonEqFluxSq[component]
                                      / (dfloat)boundaryValidationCount)
                   << "," << boundaryNormalNonEqFluxMax[component];
boundaryOutput << "," << minimumBoundaryTargetMinor;
saveTreatData("_curved_conformation_boundary", boundaryOutput.str(), step);

const int centerX = (int)DUCT_CENTER_X;
const int centerY = (int)DUCT_CENTER_Y;
velocityProfile(h_fMom, 3, centerX, 0, validationZ, step);
velocityProfile(h_fMom, 6, 0, centerY, validationZ, step);
conformationProfile(h_fMom, 1, centerX, 0, validationZ, step);
conformationProfile(h_fMom, 2, 0, centerY, validationZ, step);
