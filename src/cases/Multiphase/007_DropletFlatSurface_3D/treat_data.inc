{
        // =========================================================
    // Contact-angle measurement for a static droplet on SOUTH wall
    // =========================================================

    constexpr dfloat WALL_Y = 0.0_df;
    constexpr int FIT_WALL_EXCLUDE_LAYERS = 2;
    constexpr int FIT_TOP_EXCLUDE_LAYERS = 1;

    // ---------------------------------------------------------
    // Sample phi from the host macroscopic moment field
    // ---------------------------------------------------------
    auto samplePhiAt = [&](int x, int y, int z) -> dfloat {
        const size_t idx = idxMom(x % BLOCK_NX, y % BLOCK_NY, z % BLOCK_NZ, M3_PHI_INDEX, x / BLOCK_NX, y / BLOCK_NY, z / BLOCK_NZ);
        return h_fMom[idx];
    };

    // ---------------------------------------------------------
    // Linear phi=PHI_REFERENCE crossing
    // ---------------------------------------------------------
    auto interpolateCrossing = [&](dfloat p0, dfloat p1, dfloat x0) -> dfloat {
        const dfloat denom = p1 - p0;
        if (fabs(denom) < 1.0e-14_df) return x0;
        return x0 + (PHI_REFERENCE - p0) / denom;
    };

    // ---------------------------------------------------------
    // Find phi=PHI_REFERENCE crossing along +y
    // ---------------------------------------------------------
    auto findInterfaceY = [&](int xProbe, int zProbe, int yStart) -> dfloat {
        if (yStart < 0 || yStart >= NY - 1) return -1.0_df;
        dfloat prev = samplePhiAt(xProbe, yStart, zProbe);
        for (int y = yStart + 1; y < NY; ++y) {
            const dfloat curr = samplePhiAt(xProbe, y, zProbe);
            const dfloat s0 = prev - PHI_REFERENCE;
            const dfloat s1 = curr - PHI_REFERENCE;
            if (s0 == 0.0_df || s0 * s1 <= 0.0_df) return interpolateCrossing(prev, curr, static_cast<dfloat>(y - 1));
            prev = curr;
        }
        return -1.0_df;
    };

    // ---------------------------------------------------------
    // Find leftmost/rightmost phi=PHI_REFERENCE crossing along x
    // ---------------------------------------------------------
    auto findXCrossing = [&](int yProbe, int zProbe, bool findFirst) -> dfloat {
        if (yProbe < 0 || yProbe >= NY) return -1.0_df;
        dfloat prev = samplePhiAt(0, yProbe, zProbe);
        dfloat result = -1.0_df;
        for (int x = 1; x < NX; ++x) {
            const dfloat curr = samplePhiAt(x, yProbe, zProbe);
            const dfloat s0 = prev - PHI_REFERENCE;
            const dfloat s1 = curr - PHI_REFERENCE;
            if (s0 == 0.0_df || s0 * s1 <= 0.0_df) {
                const dfloat xCross = interpolateCrossing(prev, curr, static_cast<dfloat>(x - 1));
                if (findFirst) return xCross;
                result = xCross;
            }
            prev = curr;
        }
        return result;
    };

    // ---------------------------------------------------------
    // Extrapolate contact line from y=1 and y=2 to physical wall
    // ---------------------------------------------------------
    auto findBaseDiameterExtrapolated = [&](int zProbe, dfloat& xLeft1, dfloat& xRight1, dfloat& xLeft2, dfloat& xRight2, dfloat& xLeftWall, dfloat& xRightWall) -> dfloat {
        xLeft1 = findXCrossing(1, zProbe, true);
        xRight1 = findXCrossing(1, zProbe, false);
        xLeft2 = findXCrossing(2, zProbe, true);
        xRight2 = findXCrossing(2, zProbe, false);
        if (xLeft1 < 0.0_df || xRight1 < 0.0_df || xLeft2 < 0.0_df || xRight2 < 0.0_df) return 0.0_df;
        xLeftWall = xLeft1 + (WALL_Y - 1.0_df) * (xLeft2 - xLeft1);
        xRightWall = xRight1 + (WALL_Y - 1.0_df) * (xRight2 - xRight1);
        return fabs(xRightWall - xLeftWall);
    };

    // =========================================================
    // METHOD 1: Spherical-cap geometry
    // =========================================================

    const int xCenter = NX / 2;
    const int zCenter = NZ / 2;
    const dfloat yTop = findInterfaceY(xCenter, zCenter, 1);

    dfloat xLeft1 = -1.0_df;
    dfloat xRight1 = -1.0_df;
    dfloat xLeft2 = -1.0_df;
    dfloat xRight2 = -1.0_df;
    dfloat xLeftWall = -1.0_df;
    dfloat xRightWall = -1.0_df;

    const dfloat b = findBaseDiameterExtrapolated(zCenter, xLeft1, xRight1, xLeft2, xRight2, xLeftWall, xRightWall);
    const dfloat h = (yTop > WALL_Y) ? (yTop - WALL_Y) : 0.0_df;
    const dfloat r = 0.5_df * b;

    dfloat thetaCapDeg = 0.0_df;

    if (h > 1.0e-12_df && r > 1.0e-12_df) {
        thetaCapDeg = 2.0_df * atan2(h, r) * (180.0_df / M_PI);
    }

    // =========================================================
    // METHOD 2: Circle fit of phi=PHI_REFERENCE
    // =========================================================

    int fitYStart = static_cast<int>(ceil(WALL_Y)) + FIT_WALL_EXCLUDE_LAYERS;
    if (fitYStart < 1) fitYStart = 1;

    int fitYEnd = (yTop > 0.0_df) ? static_cast<int>(floor(yTop)) - FIT_TOP_EXCLUDE_LAYERS : -1;
    if (fitYEnd > NY - 2) fitYEnd = NY - 2;

    int fitNPoints = 0;
    dfloat sumFitX = 0.0_df;
    dfloat sumFitY = 0.0_df;

    if (fitYEnd >= fitYStart) {
        for (int y = fitYStart; y <= fitYEnd; ++y) {
            const dfloat xl = findXCrossing(y, zCenter, true);
            const dfloat xr = findXCrossing(y, zCenter, false);
            if (xl < 0.0_df || xr < 0.0_df) continue;
            if (fabs(xr - xl) < 1.0e-8_df) continue;
            sumFitX += xl + xr;
            sumFitY += 2.0_df * static_cast<dfloat>(y);
            fitNPoints += 2;
        }
    }

    dfloat fitMeanX = 0.0_df;
    dfloat fitMeanY = 0.0_df;

    if (fitNPoints > 0) {
        fitMeanX = sumFitX / static_cast<dfloat>(fitNPoints);
        fitMeanY = sumFitY / static_cast<dfloat>(fitNPoints);
    }

    dfloat Sxx = 0.0_df;
    dfloat Sxy = 0.0_df;
    dfloat Syy = 0.0_df;
    dfloat Sx = 0.0_df;
    dfloat Sy = 0.0_df;
    dfloat bxFit = 0.0_df;
    dfloat byFit = 0.0_df;
    dfloat bcFit = 0.0_df;

    if (fitNPoints >= 6) {
        for (int y = fitYStart; y <= fitYEnd; ++y) {
            const dfloat xl = findXCrossing(y, zCenter, true);
            const dfloat xr = findXCrossing(y, zCenter, false);
            if (xl < 0.0_df || xr < 0.0_df) continue;
            if (fabs(xr - xl) < 1.0e-8_df) continue;
            const dfloat yy = static_cast<dfloat>(y) - fitMeanY;
            const dfloat xValues[2] = {xl, xr};
            for (int p = 0; p < 2; ++p) {
                const dfloat xx = xValues[p] - fitMeanX;
                const dfloat rr2 = xx * xx + yy * yy;
                Sxx += xx * xx;
                Sxy += xx * yy;
                Syy += yy * yy;
                Sx += xx;
                Sy += yy;
                bxFit -= xx * rr2;
                byFit -= yy * rr2;
                bcFit -= rr2;
            }
        }
    }

    dfloat mat[3][4] = {{Sxx, Sxy, Sx, bxFit}, {Sxy, Syy, Sy, byFit}, {Sx, Sy, static_cast<dfloat>(fitNPoints), bcFit}};

    auto solve3x3 = [&](dfloat M[3][4], dfloat& A, dfloat& B, dfloat& C) -> bool {
        for (int col = 0; col < 3; ++col) {
            int pivot = col;
            dfloat pivotAbs = fabs(M[col][col]);
            for (int row = col + 1; row < 3; ++row) {
                const dfloat val = fabs(M[row][col]);
                if (val > pivotAbs) {
                    pivotAbs = val;
                    pivot = row;
                }
            }
            if (pivotAbs < 1.0e-14_df) return false;
            if (pivot != col) {
                for (int j = col; j < 4; ++j) {
                    const dfloat tmp = M[col][j];
                    M[col][j] = M[pivot][j];
                    M[pivot][j] = tmp;
                }
            }
            const dfloat invPivot = 1.0_df / M[col][col];
            for (int j = col; j < 4; ++j) M[col][j] *= invPivot;
            for (int row = 0; row < 3; ++row) {
                if (row == col) continue;
                const dfloat factor = M[row][col];
                for (int j = col; j < 4; ++j) M[row][j] -= factor * M[col][j];
            }
        }
        A = M[0][3];
        B = M[1][3];
        C = M[2][3];
        return true;
    };

    dfloat fitA = 0.0_df;
    dfloat fitB = 0.0_df;
    dfloat fitC = 0.0_df;
    dfloat xFitCenter = 0.0_df;
    dfloat yFitCenter = 0.0_df;
    dfloat R_fit = 0.0_df;
    dfloat thetaFitDeg = 0.0_df;
    dfloat fitRms = 0.0_df;
    bool fitValid = false;

    if (fitNPoints >= 6) {
        fitValid = solve3x3(mat, fitA, fitB, fitC);
        if (fitValid) {
            const dfloat xcCentered = -0.5_df * fitA;
            const dfloat ycCentered = -0.5_df * fitB;
            const dfloat R2 = xcCentered * xcCentered + ycCentered * ycCentered - fitC;
            if (R2 > 0.0_df) {
                R_fit = sqrt(R2);
                xFitCenter = fitMeanX + xcCentered;
                yFitCenter = fitMeanY + ycCentered;
                const dfloat cosFit = (WALL_Y - yFitCenter) / R_fit;
                const dfloat cosFitClamped = fmax(-1.0_df, fmin(1.0_df, cosFit));
                thetaFitDeg = acos(cosFitClamped) * (180.0_df / M_PI);
            } else {
                fitValid = false;
            }
        }
    }

    if (fitValid && fitNPoints > 0) {
        dfloat sumResidual2 = 0.0_df;
        int residualCount = 0;
        for (int y = fitYStart; y <= fitYEnd; ++y) {
            const dfloat xl = findXCrossing(y, zCenter, true);
            const dfloat xr = findXCrossing(y, zCenter, false);
            if (xl < 0.0_df || xr < 0.0_df) continue;
            if (fabs(xr - xl) < 1.0e-8_df) continue;
            const dfloat xValues[2] = {xl, xr};
            for (int p = 0; p < 2; ++p) {
                const dfloat dx = xValues[p] - xFitCenter;
                const dfloat dy = static_cast<dfloat>(y) - yFitCenter;
                const dfloat residual = sqrt(dx * dx + dy * dy) - R_fit;
                sumResidual2 += residual * residual;
                ++residualCount;
            }
        }
        if (residualCount > 0) fitRms = sqrt(sumResidual2 / static_cast<dfloat>(residualCount));
    }

    // =========================================================
    // Contact-angle errors
    // =========================================================

    const dfloat absErrCapDeg = fabs(CONTACT_ANGLE_DEG - thetaCapDeg);
    const dfloat absErrFitDeg = fitValid ? fabs(CONTACT_ANGLE_DEG - thetaFitDeg) : -1.0_df;

    // =========================================================
    // CSV output
    // =========================================================

    const std::filesystem::path baseDir = folderSetup();
    const std::filesystem::path csvPath = baseDir / (std::string("contact_angle_summary_ID") + ID_SIM + ".csv");

    static bool csvInitialized = false;

    if (!csvInitialized) {
        std::ofstream header(csvPath, std::ios::trunc);
        if (header.is_open()) {
            header << "step," << "theta_imposed_deg," << "theta_cap_deg," << "abs_error_cap_deg," << "height_h," << "base_diameter_b," << "contact_radius_r," << "yTop," << "xLeft_y1," << "xRight_y1," << "xLeft_y2," << "xRight_y2," << "xLeft_wall," << "xRight_wall," << "theta_circle_fit_deg," << "abs_error_circle_fit_deg," << "circle_fit_valid," << "circle_fit_npoints," << "circle_fit_rms," << "fit_x_center," << "fit_y_center," << "R_fit" << "\n";
        }
        csvInitialized = true;
    }

    std::ofstream out(csvPath, std::ios::app);

    if (out.is_open()) {
        out << std::fixed << std::setprecision(8) << step << "," << CONTACT_ANGLE_DEG << "," << thetaCapDeg << "," << absErrCapDeg << "," << h << "," << b << "," << r << "," << yTop << "," << xLeft1 << "," << xRight1 << "," << xLeft2 << "," << xRight2 << "," << xLeftWall << "," << xRightWall << "," << thetaFitDeg << "," << absErrFitDeg << "," << (fitValid ? 1 : 0) << "," << fitNPoints << "," << fitRms << "," << xFitCenter << "," << yFitCenter << "," << R_fit << "\n";
    }
}