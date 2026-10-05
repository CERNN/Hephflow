#ifdef TESTS
if (((step + 1) % TGV_SAMPLE_EVERY == 0) || step == N_STEPS - 1) {
    checkCudaErrors(cudaDeviceSynchronize());
    for (int g = 0; g < N_GPUS; ++g)
        devices[g].cudaMemcpyDeviceField(hostField, g, slice);
    checkCudaErrors(cudaDeviceSynchronize());

    const dfloat elapsed = static_cast<dfloat>(step + 1);
    const dfloat amplitude = exp(-3.0_df * VISC * elapsed / (L * L));
    dfloat velocityError2 = 0.0_df, velocityReference2 = 0.0_df;
    dfloat kinetic = 0.0_df, kineticInitial = 0.0_df, mass = 0.0_df;
    dfloat momentumX = 0.0_df, momentumY = 0.0_df, momentumZ = 0.0_df;
    dfloat symmetryError = 0.0_df;

    for (int z = 0; z < NZ_TOTAL; ++z) {
        for (int y = 0; y < NY; ++y) {
            for (int x = 0; x < NX; ++x) {
                const size_t rhoIndex = idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M_RHO_INDEX,
                                               x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ);
                const dfloat rhoNode = RHO_0 + hostField.h_fMom[rhoIndex];
                const dfloat uxNode = hostField.h_fMom[rhoIndex + M_UX_INDEX*BLOCK_LBM_SIZE] / F_M_I_SCALE;
                const dfloat uyNode = hostField.h_fMom[rhoIndex + M_UY_INDEX*BLOCK_LBM_SIZE] / F_M_I_SCALE;
                const dfloat uzNode = hostField.h_fMom[rhoIndex + M_UZ_INDEX*BLOCK_LBM_SIZE] / F_M_I_SCALE;
                const dfloat ux0 = U_MAX * sin(static_cast<dfloat>(x)/L)
                    * cos(static_cast<dfloat>(y)/L) * cos(static_cast<dfloat>(z)/L);
                const dfloat uy0 = -U_MAX * cos(static_cast<dfloat>(x)/L)
                    * sin(static_cast<dfloat>(y)/L) * cos(static_cast<dfloat>(z)/L);
                const dfloat uxExact = amplitude * ux0;
                const dfloat uyExact = amplitude * uy0;
                const dfloat dx = uxNode - uxExact, dy = uyNode - uyExact;
                velocityError2 += dx*dx + dy*dy + uzNode*uzNode;
                velocityReference2 += uxExact*uxExact + uyExact*uyExact;
                kinetic += 0.5_df * (uxNode*uxNode + uyNode*uyNode + uzNode*uzNode);
                kineticInitial += 0.5_df * (ux0*ux0 + uy0*uy0);
                mass += rhoNode;
                momentumX += rhoNode * uxNode;
                momentumY += rhoNode * uyNode;
                momentumZ += rhoNode * uzNode;

                const size_t swapped = idxMom(y%BLOCK_NX, x%BLOCK_NY, z%BLOCK_NZ, M_UY_INDEX,
                                              y/BLOCK_NX, x/BLOCK_NY, z/BLOCK_NZ);
                symmetryError = fmax(symmetryError,
                    fabs(uxNode + hostField.h_fMom[swapped] / F_M_I_SCALE));
            }
        }
    }

    const dfloat nodeCount = static_cast<dfloat>(NUMBER_LBM_NODES);
    const dfloat velocityL2 = sqrt(velocityError2 / velocityReference2);
    const dfloat numericalEnergyRatio = kinetic / kineticInitial;
    const dfloat exactEnergyRatio = amplitude * amplitude;
    const dfloat energyRelativeError = fabs(numericalEnergyRatio-exactEnergyRatio)/exactEnergyRatio;
    const dfloat massDrift = fabs(mass/(RHO_0*nodeCount)-1.0_df);
    const dfloat momentumMagnitude = sqrt(momentumX*momentumX+momentumY*momentumY+momentumZ*momentumZ)
        /(RHO_0*nodeCount);

    FILE* historyFile = fopen("tgv_history.csv", step + 1 == TGV_SAMPLE_EVERY ? "w" : "a");
    if (historyFile != nullptr) {
        if (step + 1 == TGV_SAMPLE_EVERY)
            fprintf(historyFile, "step,viscous_time,velocity_l2,energy_numerical,energy_analytical,mass_drift,momentum,symmetry\n");
        fprintf(historyFile, "%d,%.15e,%.15e,%.15e,%.15e,%.15e,%.15e,%.15e\n",
                step + 1, static_cast<double>(VISC * elapsed / (L * L)),
                static_cast<double>(velocityL2), static_cast<double>(numericalEnergyRatio),
                static_cast<double>(exactEnergyRatio), static_cast<double>(massDrift),
                static_cast<double>(momentumMagnitude), static_cast<double>(symmetryError));
        fclose(historyFile);
    }

    if (step == N_STEPS - 1) {
        FILE* testFile = fopen("test_results.txt", "w");
        if (testFile != nullptr) {
            fprintf(testFile, "STEP,%d\n", step + 1);
            fprintf(testFile, "L2_ERROR_ABSOLUTE,%.15e\n", static_cast<double>(velocityL2));
            fprintf(testFile, "ENERGY_RATIO_NUMERICAL,%.15e\n", static_cast<double>(numericalEnergyRatio));
            fprintf(testFile, "ENERGY_RATIO_ANALYTICAL,%.15e\n", static_cast<double>(exactEnergyRatio));
            fprintf(testFile, "ENERGY_RELATIVE_ERROR,%.15e\n", static_cast<double>(energyRelativeError));
            fprintf(testFile, "MASS_RELATIVE_DRIFT,%.15e\n", static_cast<double>(massDrift));
            fprintf(testFile, "MOMENTUM_MAGNITUDE,%.15e\n", static_cast<double>(momentumMagnitude));
            fprintf(testFile, "SYMMETRY_MAX_ERROR,%.15e\n", static_cast<double>(symmetryError));
            fclose(testFile);
        }
        printf("[TEST] Taylor-Green velocity L2: %.8e, energy error: %.8e\n",
               static_cast<double>(velocityL2), static_cast<double>(energyRelativeError));
    }
}
#endif
