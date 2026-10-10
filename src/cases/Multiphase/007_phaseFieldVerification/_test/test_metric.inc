#ifdef TESTS
if (step == N_STEPS - 1) {
    deviceField.copyMacroscopicDeviceField(hostField, devices.data());

    double phiSum = 0.0;
    double initialPhiSum = 0.0;
    double phaseVolume = 0.0;
    double initialPhaseVolume = 0.0;
    double profileError2 = 0.0;
    double profileReference2 = 0.0;
    double sourceSum = 0.0;
    double sourceAbsSum = 0.0;
    double sourceMaxAbs = 0.0;
    double compressionSum = 0.0;
    double compressionAbsSum = 0.0;
    double curvatureWeightedSum = 0.0;
    double curvatureWeight = 0.0;
    double phiMin = (double)PHI_TWO;
    double phiMax = (double)PHI_ONE;
    double maxVelocity = 0.0;
    double clipSigned = 0.0;
    double clipAbsolute = 0.0;
    double rhoInsideSum = 0.0;
    double rhoOutsideSum = 0.0;
    unsigned long long rhoInsideCount = 0;
    unsigned long long rhoOutsideCount = 0;
    unsigned long long nonFiniteCount = 0;

    const double deltaPhi = (double)(PHI_TWO - PHI_ONE);
    for (int z = 0; z < NZ; ++z) {
        for (int y = 0; y < NY; ++y) {
            for (int x = 0; x < NX; ++x) {
                const size_t phiIndex = idxMom(
                    x % BLOCK_NX, y % BLOCK_NY, z % BLOCK_NZ,
                    M3_PHI_INDEX, x / BLOCK_NX, y / BLOCK_NY,
                    z / BLOCK_NZ);
                const double phi = (double)loadMoment(
                    hostField.h_fMom, phiIndex, M3_PHI_INDEX);
                const double initialPhi = (double)phaseTestInitialPhi(
                    (dfloat)x, (dfloat)y, (dfloat)z);
                if (!std::isfinite(phi)) {
                    ++nonFiniteCount;
                    continue;
                }

                phiSum += phi;
                initialPhiSum += initialPhi;
                phaseVolume += (phi - (double)PHI_ONE) / deltaPhi;
                initialPhaseVolume +=
                    (initialPhi - (double)PHI_ONE) / deltaPhi;
                const double profileDifference = phi - initialPhi;
                profileError2 += profileDifference * profileDifference;
                profileReference2 += initialPhi * initialPhi;
                phiMin = std::min(phiMin, phi);
                phiMax = std::max(phiMax, phi);

                const auto moment = [&](int index) -> double {
                    const size_t address = idxMom(
                        x % BLOCK_NX, y % BLOCK_NY, z % BLOCK_NZ,
                        index, x / BLOCK_NX, y / BLOCK_NY,
                        z / BLOCK_NZ);
                    return (double)loadMoment(hostField.h_fMom, address, index);
                };
                const double uxValue = moment(M_UX_INDEX) / (double)F_M_I_SCALE;
                const double uyValue = moment(M_UY_INDEX) / (double)F_M_I_SCALE;
                const double uzValue = moment(M_UZ_INDEX) / (double)F_M_I_SCALE;
                maxVelocity = std::max(maxVelocity,
                    std::sqrt(uxValue*uxValue + uyValue*uyValue + uzValue*uzValue));

#if PHASE_TEST_SCENARIO == PHASE_TEST_CAPILLARY_DROPLET
                const double cx = 0.5 * (NX - 1);
                const double cy = 0.5 * (NY - 1);
                const double cz = 0.5 * (NZ - 1);
                const double dx = std::min(std::abs(x - cx), NX - std::abs(x - cx));
                const double dy = std::min(std::abs(y - cy), NY - std::abs(y - cy));
                const double dz = std::min(std::abs(z - cz), NZ - std::abs(z - cz));
                const double radius = std::sqrt(dx*dx + dy*dy + dz*dz);
                const double localRho = (double)RHO_0 + moment(M_RHO_INDEX);
                if (radius < (double)PHASE_TEST_RADIUS - 0.5*(double)phi_del) {
                    rhoInsideSum += localRho;
                    ++rhoInsideCount;
                } else if (radius > (double)PHASE_TEST_RADIUS + 0.5*(double)phi_del) {
                    rhoOutsideSum += localRho;
                    ++rhoOutsideCount;
                }
#endif

                const double lapPhi = moment(M3_LP_INDEX);
                double source = 0.0;
#if defined(PHASE_TRANSPORT_CONSERVATIVE_ALLEN_CAHN)
                const double compression = moment(M3_DIV_LAMBDA_NORMAL_INDEX);
                compressionSum += compression;
                compressionAbsSum += std::abs(compression);
                source = ((double)M_AC - (double)PHI_LBM_DIFF) * lapPhi
                       - (double)M_AC * compression;
#elif defined(PHASE_TRANSPORT_CAHN_HILLIARD)
                source = (double)M_CH * moment(M3_LAPLACIAN_MU_INDEX)
                       - (double)PHI_LBM_DIFF * lapPhi;
#endif
                sourceSum += source;
                sourceAbsSum += std::abs(source);
                sourceMaxAbs = std::max(sourceMaxAbs, std::abs(source));

#ifdef PHASE_DIAGNOSTICS
                clipSigned += moment(M3_CLIP_SIGNED_INDEX);
                clipAbsolute += moment(M3_CLIP_ABSOLUTE_INDEX);
#endif

#ifdef PHASE_FORCE_CSF
                const double interfaceWeight = std::max(0.0,
                    1.0 - std::abs(phi - 0.5*((double)PHI_ONE + (double)PHI_TWO))
                          / (0.5 * deltaPhi));
                curvatureWeightedSum += interfaceWeight * moment(M3_DIV_NORMAL_INDEX);
                curvatureWeight += interfaceWeight;
#endif
            }
        }
    }

    const double nodeCount = (double)NX * (double)NY * (double)NZ;
    const double phaseDrift = std::abs(phiSum - initialPhiSum)
        / (nodeCount * deltaPhi);
    const double phaseOvershoot = std::max(0.0, std::max(
        (double)PHI_ONE - phiMin, phiMax - (double)PHI_TWO));
    const double profileL2 = std::sqrt(profileError2
        / std::max(profileReference2, 1.0e-300));
    const double sourceNormalized = std::abs(sourceSum)
        / std::max(sourceAbsSum, 1.0e-300);
    const double compressionNormalized = std::abs(compressionSum)
        / std::max(compressionAbsSum, 1.0e-300);
    const double volumeDrift = std::abs(phaseVolume - initialPhaseVolume)
        / std::max(initialPhaseVolume, 1.0e-300);
    const double equivalentRadius = std::cbrt(
        3.0 * phaseVolume / (4.0 * (double)M_PI));
    const double initialEquivalentRadius = std::cbrt(
        3.0 * initialPhaseVolume / (4.0 * (double)M_PI));
    const double radiusDrift = std::abs(equivalentRadius - initialEquivalentRadius)
        / std::max(initialEquivalentRadius, 1.0e-300);
    const double meanCurvature = curvatureWeight > 0.0
        ? curvatureWeightedSum / curvatureWeight : 0.0;
    const double expectedCurvature = -2.0 / (double)PHASE_TEST_RADIUS;
    const double curvatureRelativeError = curvatureWeight > 0.0
        ? std::abs(meanCurvature - expectedCurvature) / std::abs(expectedCurvature)
        : 0.0;
    const double clipAccountingError = std::abs(
        (phiSum - initialPhiSum) - clipSigned) / (nodeCount * deltaPhi);
    const double pressureInside = rhoInsideCount > 0
        ? (1.0/3.0) * rhoInsideSum / (double)rhoInsideCount : 0.0;
    const double pressureOutside = rhoOutsideCount > 0
        ? (1.0/3.0) * rhoOutsideSum / (double)rhoOutsideCount : 0.0;
    const double pressureJump = pressureInside - pressureOutside;
    const double expectedPressureJump = 2.0 * (double)phi_sigma
        / (double)PHASE_TEST_RADIUS;
    const double laplaceRelativeError = std::abs(
        pressureJump - expectedPressureJump) / expectedPressureJump;

    FILE *testFile = fopen("test_results.txt", "w");
    if (testFile != nullptr) {
        fprintf(testFile, "STEP,%d\n", step);
        fprintf(testFile, "FINITE_COUNT,%llu\n", nonFiniteCount);
        fprintf(testFile, "PHI_SUM,%.17e\n", phiSum);
        fprintf(testFile, "INITIAL_PHI_SUM,%.17e\n", initialPhiSum);
        fprintf(testFile, "PHASE_DRIFT,%.17e\n", phaseDrift);
        fprintf(testFile, "PHASE_VOLUME_DRIFT,%.17e\n", volumeDrift);
        fprintf(testFile, "PHI_MIN,%.17e\n", phiMin);
        fprintf(testFile, "PHI_MAX,%.17e\n", phiMax);
        fprintf(testFile, "PHI_OVERSHOOT,%.17e\n", phaseOvershoot);
        fprintf(testFile, "PROFILE_L2,%.17e\n", profileL2);
        fprintf(testFile, "SOURCE_SUM,%.17e\n", sourceSum);
        fprintf(testFile, "SOURCE_MAX_ABS,%.17e\n", sourceMaxAbs);
        fprintf(testFile, "SOURCE_SUM_NORMALIZED,%.17e\n", sourceNormalized);
        fprintf(testFile, "COMPRESSION_SUM_NORMALIZED,%.17e\n", compressionNormalized);
        fprintf(testFile, "EQUIVALENT_RADIUS,%.17e\n", equivalentRadius);
        fprintf(testFile, "RADIUS_REL_DRIFT,%.17e\n", radiusDrift);
        fprintf(testFile, "MEAN_CURVATURE,%.17e\n", meanCurvature);
        fprintf(testFile, "EXPECTED_CURVATURE,%.17e\n", expectedCurvature);
        fprintf(testFile, "CURVATURE_REL_ERROR,%.17e\n", curvatureRelativeError);
        fprintf(testFile, "MAX_VELOCITY,%.17e\n", maxVelocity);
        fprintf(testFile, "CLIP_SIGNED_CORRECTION,%.17e\n", clipSigned);
        fprintf(testFile, "CLIP_ABSOLUTE_CORRECTION,%.17e\n", clipAbsolute);
        fprintf(testFile, "CLIP_ACCOUNTING_ERROR,%.17e\n", clipAccountingError);
        fprintf(testFile, "PRESSURE_JUMP,%.17e\n", pressureJump);
        fprintf(testFile, "EXPECTED_PRESSURE_JUMP,%.17e\n", expectedPressureJump);
        fprintf(testFile, "LAPLACE_REL_ERROR,%.17e\n", laplaceRelativeError);
        fclose(testFile);
    }
    printf("[TEST] phase drift %.6e, profile L2 %.6e, source balance %.6e\n",
        phaseDrift, profileL2, sourceNormalized);
}
#endif // TESTS
