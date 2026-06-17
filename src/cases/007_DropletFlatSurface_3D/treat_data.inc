{
    // =========================================================
    // Contact angle measurement via spherical-cap geometry
    // Paper formula:
    //   theta <= 90: theta = 2 * atan(2h / b)        [O below wall]
    //   theta >  90: theta = arcsin((h-r)/r) + 90    [O above wall, NOTE: r here is R_sphere]
    //
    // Unified robust form (no branching needed):
    //   R_sphere = (r^2 + h^2) / (2h),  r = b/2
    //   theta = acos((h - R_sphere) / R_sphere)
    // =========================================================

    // ---------------------------------------------------------
    // 1. Per-node phi sampler (device -> host)
    // ---------------------------------------------------------
    auto samplePhiAt = [&](int x, int y, int z) -> dfloat {
        dfloat phi = 0.0_df;
        const size_t idx = idxMom(
            x % BLOCK_NX, y % BLOCK_NY, z % BLOCK_NZ,
            M3_PHI_INDEX,
            x / BLOCK_NX, y / BLOCK_NY, z / BLOCK_NZ
        );
        checkCudaErrors(cudaMemcpy(&phi, fMom + idx,
                                   sizeof(dfloat), cudaMemcpyDeviceToHost));
        return phi;
    };

    // ---------------------------------------------------------
    // 2. Find the phi=PHI_REFERENCE crossing along y (vertical)
    //    scanning upward from yStart.
    //    Returns the interpolated y-coordinate, or -1 if not found.
    // ---------------------------------------------------------
    auto findInterfaceY = [&](int xProbe, int zProbe, int yStart) -> dfloat {
        dfloat prev = samplePhiAt(xProbe, yStart, zProbe);
        for (int y = yStart + 1; y < NY; ++y) {
            const dfloat curr = samplePhiAt(xProbe, y, zProbe);
            const dfloat s0   = prev - PHI_REFERENCE;
            const dfloat s1   = curr - PHI_REFERENCE;
            if (s0 == 0.0_df || s0 * s1 <= 0.0_df) {
                dfloat yCross = static_cast<dfloat>(y - 1);
                const dfloat denom = curr - prev;
                if (fabs(denom) > 1.0e-12_df)
                    yCross += (PHI_REFERENCE - prev) / denom;
                return yCross;
            }
            prev = curr;
        }
        return -1.0_df;
    };

    // ---------------------------------------------------------
    // 3. Find ONE phi=PHI_REFERENCE crossing along x (horizontal)
    //    at a given y,z slice.
    //    findFirst=true  -> leftmost crossing (xLeft)
    //    findFirst=false -> rightmost crossing (xRight, keeps scanning)
    //    Returns -1 if not found.
    // ---------------------------------------------------------
    auto findXCrossing = [&](int yProbe, int zProbe, bool findFirst) -> dfloat {
        dfloat prev   = samplePhiAt(0, yProbe, zProbe);
        dfloat result = -1.0_df;
        for (int x = 1; x < NX; ++x) {
            const dfloat curr = samplePhiAt(x, yProbe, zProbe);
            const dfloat s0   = prev - PHI_REFERENCE;
            const dfloat s1   = curr - PHI_REFERENCE;
            if (s0 == 0.0_df || s0 * s1 <= 0.0_df) {
                dfloat xCross = static_cast<dfloat>(x - 1);
                const dfloat denom = curr - prev;
                if (fabs(denom) > 1.0e-12_df)
                    xCross += (PHI_REFERENCE - prev) / denom;
                if (findFirst) return xCross;   // left contact: return immediately
                result = xCross;                // right contact: keep overwriting
            }
            prev = curr;
        }
        return result;
    };

    // ---------------------------------------------------------
    // 4. Extrapolate contact-line position to the wall (y=0)
    //    from the two lowest fluid layers (y=1 and y=2).
    //    Linear extrapolation: x_wall = 2*x(y=1) - x(y=2)
    //    This removes the ~1/tan(theta) bias from sampling at y=1.
    //    Returns base diameter b, and fills xLeft0/xRight0 for debug.
    // ---------------------------------------------------------
    auto findBaseDiameterExtrapolated = [&](
            int zProbe,
            dfloat& xLeft1,  dfloat& xRight1,   // crossings at y=1
            dfloat& xLeft2,  dfloat& xRight2,   // crossings at y=2
            dfloat& xLeft0,  dfloat& xRight0    // extrapolated to y=0
        ) -> dfloat
    {
        xLeft1  = findXCrossing(1, zProbe, true);
        xRight1 = findXCrossing(1, zProbe, false);
        xLeft2  = findXCrossing(2, zProbe, true);
        xRight2 = findXCrossing(2, zProbe, false);

        if (xLeft1 < 0.0_df || xRight1 < 0.0_df ||
            xLeft2 < 0.0_df || xRight2 < 0.0_df)
            return 0.0_df;

        // Linear extrapolation to y=0
        xLeft0  = 2.0_df * xLeft1  - xLeft2;
        xRight0 = 2.0_df * xRight1 - xRight2;

        return fabs(xRight0 - xLeft0);
    };

    // ---------------------------------------------------------
    // 5. Measure geometry
    // ---------------------------------------------------------
    const int xCenter = NX / 2;
    const int zCenter = NZ / 2;

    // Cap height h: top interface crossing, scanning from y=1 upward.
    // yTop is in lattice units measured from y=0 (the wall plane).
    const dfloat yTop = findInterfaceY(xCenter, zCenter, /*yStart=*/1);

    // Base diameter b and all intermediate/extrapolated coordinates
    dfloat xLeft1 = 0.0_df, xRight1 = 0.0_df;
    dfloat xLeft2 = 0.0_df, xRight2 = 0.0_df;
    dfloat xLeft0 = 0.0_df, xRight0 = 0.0_df;
    const dfloat b = findBaseDiameterExtrapolated(
        zCenter, xLeft1, xRight1, xLeft2, xRight2, xLeft0, xRight0);

    const dfloat h = yTop;          // cap height above wall
    const dfloat r = b / 2.0_df;   // contact radius

    // ---------------------------------------------------------
    // 6. Contact angle - unified formula (valid for all theta)
    //
    //   R_sphere = (r^2 + h^2) / (2*h)
    //   theta    = acos( (h - R_sphere) / R_sphere )
    //
    // Derivation:
    //   The sphere centre O sits at height (R_sphere - h) below the wall
    //   for theta<90, or above the wall for theta>90 (signed distance d = R-h).
    //   The wall intersects the sphere where the perpendicular from O has
    //   length r, so by Pythagoras: r^2 + d^2 = R^2, d = R - h  =>
    //   r^2 + (R-h)^2 = R^2  =>  R = (r^2 + h^2)/(2h).
    //   The contact angle satisfies cos(pi-theta) = d/R = (R-h)/R,
    //   i.e. theta = acos((h-R)/R).
    //
    // Cross-check against paper's two-branch formula:
    //   theta<=90: 2*atan(2h/b) = 2*atan(h/r) [cot(theta/2)=h/r] - NOTE
    //              this is the SUPPLEMENT of our derivation; verify with
    //              your convention (gas-side vs liquid-side angle).
    //   theta>90:  arcsin((h-r)/R_sphere) + 90  [R_sphere, not r]
    // ---------------------------------------------------------
    dfloat thetaMesDeg = 0.0_df;

    if (b > 1.0e-12_df && h > 1.0e-12_df) {
        const dfloat R_sphere = (r * r + h * h) / (2.0_df * h);
        const dfloat cosArg   = (h - R_sphere) / R_sphere;

        // clamp to [-1,1] to guard against floating-point noise
        const dfloat cosArgClamped = fmax(-1.0_df, fmin(1.0_df, cosArg));
        const dfloat thetaMesRad   = acos(cosArgClamped);
        thetaMesDeg = thetaMesRad * (180.0_df / M_PI);

        // Also compute paper's two-branch result for comparison in CSV
    }

    // Paper branch 1: 2*atan(2h/b)  - valid for theta<=90
    const dfloat thetaPaper1Deg = (b > 1.0e-12_df && h > 1.0e-12_df)
        ? 2.0_df * atan(2.0_df * h / b) * (180.0_df / M_PI)
        : 0.0_df;

    // Paper branch 2: arcsin((h-r)/R_sphere) + 90  - valid for theta>90
    const dfloat R_sphere = (r > 1.0e-12_df && h > 1.0e-12_df)
        ? (r * r + h * h) / (2.0_df * h)
        : 0.0_df;
    const dfloat thetaPaper2Deg = (R_sphere > 1.0e-12_df)
        ? (asin(fmax(-1.0_df, fmin(1.0_df, (h - r) / R_sphere)))
           * (180.0_df / M_PI) + 90.0_df)
        : 0.0_df;

    const dfloat absErrDeg = fabs(CONTACT_ANGLE_DEG - thetaMesDeg);

    // ---------------------------------------------------------
    // 7. CSV output
    // ---------------------------------------------------------
    const std::filesystem::path baseDir = folderSetup();
    const std::filesystem::path csvPath = baseDir / "contact_angle_summary.csv";

    if (step == REPORT_SAVE) {
        std::ofstream header(csvPath, std::ios::trunc);
        if (header.is_open()) {
            header
                // --- core results ---
                << "step,"
                << "theta_imposed_deg,"
                << "theta_measured_deg,"
                << "abs_error_deg,"
                // --- geometry ---
                << "height_h,"
                << "base_diameter_b,"
                << "contact_radius_r,"
                << "R_sphere,"
                // --- paper branches (debug) ---
                << "theta_paper_branch1_deg,"   // 2*atan(2h/b), theta<=90
                << "theta_paper_branch2_deg,"   // arcsin((h-r)/R)+90, theta>90
                // --- probe location ---
                << "x_center,"
                << "z_center,"
                // --- y=1 crossings ---
                << "xLeft_y1,"
                << "xRight_y1,"
                // --- y=2 crossings ---
                << "xLeft_y2,"
                << "xRight_y2,"
                // --- extrapolated to y=0 ---
                << "xLeft_y0,"
                << "xRight_y0,"
                // --- top interface ---
                << "yTop"
                << "\n";
        }
    }

    std::ofstream out(csvPath, std::ios::app);
    if (out.is_open()) {
        out << std::fixed << std::setprecision(6)
            << step              << ","
            << CONTACT_ANGLE_DEG << ","
            << thetaMesDeg       << ","
            << absErrDeg         << ","
            << h                 << ","
            << b                 << ","
            << r                 << ","
            << R_sphere          << ","
            << thetaPaper1Deg    << ","
            << thetaPaper2Deg    << ","
            << xCenter           << ","
            << zCenter           << ","
            << xLeft1            << ","
            << xRight1           << ","
            << xLeft2            << ","
            << xRight2           << ","
            << xLeft0            << ","
            << xRight0           << ","
            << yTop              << "\n";
    }
}