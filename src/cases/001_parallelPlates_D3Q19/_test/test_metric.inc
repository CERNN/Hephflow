/* ============================================================================
 * TEST METRIC CONFIGURATION
 * Case: 001_parallelPlates_D3Q19
 * Description: L2 error norm computation for Poiseuille flow (analytical solution)
 * ========================================================================== */

#ifdef TESTS

// Only evaluate at the last time step
if (step == N_STEPS - 1) {
    
    // Copy full macroscopic field from device to host
    checkCudaErrors(cudaDeviceSynchronize());
    deviceField.cudaMemcpyDeviceField(hostField);
    checkCudaErrors(cudaDeviceSynchronize());
    
    // Analytical solution parameters for Poiseuille flow
    // Analytical velocity profile: u_z(y) = (FZ / (2*VISC)) * y * (NY - y)
    // Where FZ is the driving force and VISC is kinematic viscosity
    
    const dfloat analytical_factor = FZ / (2.0 * VISC);
    dfloat L2_error = 0.0;
    dfloat count = 0.0;
    
    // Extract velocity profile in Z direction over Y distance from wall
    dfloat uz_profile[NY];
    dfloat uz_analytical[NY];
    
    // Average UZ over X and Z directions at each Y position
    for (int y = 0; y < NY; y++) {
        uz_profile[y] = 0.0;
        int num_samples = 0;
        
        for (int z = 0; z < NZ_TOTAL; z++) {
            for (int x = 0; x < NX; x++) {
                dfloat t_uz = hostField.h_fMom[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M_UZ_INDEX, 
                                           x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)] / F_M_I_SCALE;
                uz_profile[y] += t_uz;
                num_samples++;
            }
        }
        
        if (num_samples > 0) {
            uz_profile[y] /= (dfloat)num_samples;
        }
    }
    
    // Compute analytical solution and L2 error
    // Poiseuille profile: u_z(y) = (dp/dz)/(2*mu) * y * (H - y)
    // In lattice units with force FZ: u_z(y) = (FZ/(2*VISC)) * y * (NY - y)
    dfloat max_uz_numerical = 0.0;
    dfloat max_uz_analytical = 0.0;
    
    for (int y = 0; y < NY; y++) {
        // Analytical solution (parabolic profile)
        uz_analytical[y] = analytical_factor * (dfloat)y * (dfloat)(NY - y);
        
        // Compute L2 error at this point
        dfloat diff = uz_profile[y] - uz_analytical[y];
        L2_error += diff * diff;
        
        max_uz_numerical = fmax(max_uz_numerical, fabs(uz_profile[y]));
        max_uz_analytical = fmax(max_uz_analytical, fabs(uz_analytical[y]));
        
        count += uz_analytical[y]*uz_analytical[y];
    }
    
    // Normalize L2 error by number of points and max analytical velocity
    L2_error = sqrt(L2_error / count);
    

    // Write test results to file (in current working directory)
    FILE* test_file = fopen("test_results.txt", "w");
    if (test_file != NULL) {
        fprintf(test_file, "# Test Results for 001_parallelPlates_D3Q19\n");
        fprintf(test_file, "# Analytical solution: Poiseuille flow (parabolic profile)\n");
        fprintf(test_file, "# ================================================================\n");
        fprintf(test_file, "STEP,%d\n", step);
        fprintf(test_file, "MAX_UZ_NUMERICAL,%.15e\n", max_uz_numerical);
        fprintf(test_file, "MAX_UZ_ANALYTICAL,%.15e\n", max_uz_analytical);
        fprintf(test_file, "L2_ERROR_ABSOLUTE,%.15e\n", L2_error);
        fprintf(test_file, "\n# Detailed velocity profile comparison\n");
        fprintf(test_file, "# Y_INDEX,UZ_NUMERICAL,UZ_ANALYTICAL,DIFFERENCE\n");
        
        for (int y = 0; y < NY; y++) {
            dfloat diff = uz_profile[y] - uz_analytical[y];
            fprintf(test_file, "%d,%.15e,%.15e,%.15e\n", y, uz_profile[y], uz_analytical[y], diff);
        }
        
        fclose(test_file);
        printf("\n[TEST] Results written to test_results.txt\n");
        printf("[TEST] L2 Error (absolute): %.15e\n", L2_error);
    }
}

#endif // TESTS
