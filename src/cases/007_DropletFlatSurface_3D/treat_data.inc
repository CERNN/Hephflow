{
    auto samplePhiAt = [&](int x, int y, int z) -> dfloat {
        dfloat phi = 0.0_df;
        const size_t idx = idxMom(
            x % BLOCK_NX,
            y % BLOCK_NY,
            z % BLOCK_NZ,
            M3_PHI_INDEX,
            x / BLOCK_NX,
            y / BLOCK_NY,
            z / BLOCK_NZ
        );
        checkCudaErrors(cudaMemcpy(&phi, fMom + idx, sizeof(dfloat), cudaMemcpyDeviceToHost));
        return phi;
    };

    auto findInterfaceYAtXZ = [&](int xProbe, int zProbe) -> dfloat {
        dfloat prev = samplePhiAt(xProbe, 0, zProbe);
        for (int y = 1; y < NY; ++y) {
            const dfloat curr = samplePhiAt(xProbe, y, zProbe);
            const dfloat s0 = prev - PHI_REFERENCE;
            const dfloat s1 = curr - PHI_REFERENCE;
            if ((s0 == 0.0_df) || (s0 * s1 <= 0.0_df)) {
                dfloat yCross = (dfloat)(y - 1);
                const dfloat denom = curr - prev;
                if (fabs(denom) > 1.0e-12_df) {
                    yCross += (PHI_REFERENCE - prev) / denom;
                }
                return yCross;
            }
            prev = curr;
        }
        return 0.0_df;
    };

    auto findBaseDiameterAtY = [&](int yProbe, int zProbe) -> dfloat {
        bool foundLeft = false;
        bool foundRight = false;
        dfloat xLeft = 0.0_df;
        dfloat xRight = 0.0_df;

        dfloat prev = samplePhiAt(0, yProbe, zProbe);
        for (int x = 1; x < NX; ++x) {
            const dfloat curr = samplePhiAt(x, yProbe, zProbe);
            const dfloat s0 = prev - PHI_REFERENCE;
            const dfloat s1 = curr - PHI_REFERENCE;
            if ((s0 == 0.0_df) || (s0 * s1 <= 0.0_df)) {
                dfloat xCross = (dfloat)(x - 1);
                const dfloat denom = curr - prev;
                if (fabs(denom) > 1.0e-12_df) {
                    xCross += (PHI_REFERENCE - prev) / denom;
                }

                if (!foundLeft) {
                    xLeft = xCross;
                    foundLeft = true;
                }
                xRight = xCross;
                foundRight = true;
            }
            prev = curr;
        }

        if (!foundLeft || !foundRight) {
            return 0.0_df;
        }
        return fabs(xRight - xLeft);
    };

    const int xCenter = NX / 2;
    const int zCenter = NZ / 2;

    const dfloat yTop = findInterfaceYAtXZ(xCenter, zCenter);
    const dfloat h = yTop;

    const int yNearWall = 1;
    const dfloat b = findBaseDiameterAtY(yNearWall, zCenter);

    dfloat thetaMesDeg = 0.0_df;
    if (b > 1.0e-12_df) {
        const dfloat thetaMesRad = 2.0_df * atan(2.0_df * h / b);
        thetaMesDeg = thetaMesRad * (180.0_df / M_PI);
    }

    const dfloat absErrDeg = fabs(CONTACT_ANGLE_DEG - thetaMesDeg);

    const std::filesystem::path baseDir = folderSetup();
    const std::filesystem::path csvPath = baseDir / "contact_angle_summary.csv";

    if (step == REPORT_SAVE) {
        std::ofstream header(csvPath, std::ios::trunc);
        if (header.is_open()) {
            header << "step,theta_imposed_deg,theta_measured_deg,abs_error_deg,height_h,base_diameter_b" << std::endl;
        }
    }

    std::ofstream out(csvPath, std::ios::app);
    if (out.is_open()) {
        out << step << ","
            << std::fixed << std::setprecision(6)
            << CONTACT_ANGLE_DEG << ","
            << thetaMesDeg << ","
            << absErrDeg << ","
            << h << ","
            << b << std::endl;
    }
}
