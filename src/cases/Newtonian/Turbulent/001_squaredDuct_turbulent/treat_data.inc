// The streamwise direction is statistically homogeneous.  Each call first
// averages over z, then updates a time average at every (x,y) location.  Raw
// first and second moments are retained so the Reynolds stresses are formed as
// R_ij = <u_i u_j> - <u_i><u_j> without discarding secondary duct motion.
if (step >= TURBULENCE_STATS_START_STEP) {
    constexpr int CROSS_SECTION_SIZE = NX * NY;
    static_assert(
        TURBULENCE_STATS_SAVE_INTERVAL % REPORT_SAVE == 0,
        "Turbulence save interval must be a multiple of REPORT_SAVE");
    static_assert(
        TURBULENCE_STATS_START_STEP % REPORT_SAVE == 0,
        "Turbulence statistics must start on a report step");

    static unsigned long long turbulence_samples = 0;
    static bool first_turbulence_export = true;
    static double mean_rho[CROSS_SECTION_SIZE] = {};
    static double mean_ux[CROSS_SECTION_SIZE] = {};
    static double mean_uy[CROSS_SECTION_SIZE] = {};
    static double mean_uz[CROSS_SECTION_SIZE] = {};

    static double mean_uxux[CROSS_SECTION_SIZE] = {};
    static double mean_uyuy[CROSS_SECTION_SIZE] = {};
    static double mean_uzuz[CROSS_SECTION_SIZE] = {};
    static double mean_uxuy[CROSS_SECTION_SIZE] = {};
    static double mean_uxuz[CROSS_SECTION_SIZE] = {};
    static double mean_uyuz[CROSS_SECTION_SIZE] = {};

    ++turbulence_samples;
    const double time_weight = 1.0 / static_cast<double>(turbulence_samples);
    const double inverse_nz = 1.0 / static_cast<double>(NZ_TOTAL);

    for (int y = 0; y < NY; ++y) {
        for (int x = 0; x < NX; ++x) {
            const int cross_index = x + NX * y;

            double rho_z = 0.0;
            double ux_z = 0.0;
            double uy_z = 0.0;
            double uz_z = 0.0;
            double uxux_z = 0.0;
            double uyuy_z = 0.0;
            double uzuz_z = 0.0;
            double uxuy_z = 0.0;
            double uxuz_z = 0.0;
            double uyuz_z = 0.0;

            for (int z = 0; z < NZ_TOTAL; ++z) {
                const int block_x = x / BLOCK_NX;
                const int block_y = y / BLOCK_NY;
                const int block_z = z / BLOCK_NZ;
                const int thread_x = x % BLOCK_NX;
                const int thread_y = y % BLOCK_NY;
                const int thread_z = z % BLOCK_NZ;

                const double rho_node = static_cast<double>(h_fMom[idxMom(thread_x, thread_y, thread_z, M_RHO_INDEX, block_x, block_y, block_z)]) + static_cast<double>(RHO_0);
                const double ux_node = static_cast<double>(h_fMom[idxMom(thread_x, thread_y, thread_z, M_UX_INDEX, block_x, block_y, block_z)]) / static_cast<double>(F_M_I_SCALE);
                const double uy_node = static_cast<double>(h_fMom[idxMom(thread_x, thread_y, thread_z, M_UY_INDEX, block_x, block_y, block_z)]) / static_cast<double>(F_M_I_SCALE);
                const double uz_node = static_cast<double>(h_fMom[idxMom(thread_x, thread_y, thread_z, M_UZ_INDEX, block_x, block_y, block_z)]) / static_cast<double>(F_M_I_SCALE);

                rho_z += rho_node;
                ux_z += ux_node;
                uy_z += uy_node;
                uz_z += uz_node;
                uxux_z += ux_node * ux_node;
                uyuy_z += uy_node * uy_node;
                uzuz_z += uz_node * uz_node;
                uxuy_z += ux_node * uy_node;
                uxuz_z += ux_node * uz_node;
                uyuz_z += uy_node * uz_node;
            }

            rho_z *= inverse_nz;
            ux_z *= inverse_nz;
            uy_z *= inverse_nz;
            uz_z *= inverse_nz;
            uxux_z *= inverse_nz;
            uyuy_z *= inverse_nz;
            uzuz_z *= inverse_nz;
            uxuy_z *= inverse_nz;
            uxuz_z *= inverse_nz;
            uyuz_z *= inverse_nz;

            mean_rho[cross_index] += (rho_z - mean_rho[cross_index]) * time_weight;
            mean_ux[cross_index] += (ux_z - mean_ux[cross_index]) * time_weight;
            mean_uy[cross_index] += (uy_z - mean_uy[cross_index]) * time_weight;
            mean_uz[cross_index] += (uz_z - mean_uz[cross_index]) * time_weight;
            mean_uxux[cross_index] += (uxux_z - mean_uxux[cross_index]) * time_weight;
            mean_uyuy[cross_index] += (uyuy_z - mean_uyuy[cross_index]) * time_weight;
            mean_uzuz[cross_index] += (uzuz_z - mean_uzuz[cross_index]) * time_weight;
            mean_uxuy[cross_index] += (uxuy_z - mean_uxuy[cross_index]) * time_weight;
            mean_uxuz[cross_index] += (uxuz_z - mean_uxuz[cross_index]) * time_weight;
            mean_uyuz[cross_index] += (uyuz_z - mean_uyuz[cross_index]) * time_weight;
        }
    }

    const bool save_statistics =
        ((step - TURBULENCE_STATS_START_STEP) %
         TURBULENCE_STATS_SAVE_INTERVAL) == 0;

    if (save_statistics) {
        static double reynolds_xx[CROSS_SECTION_SIZE] = {};
        static double reynolds_yy[CROSS_SECTION_SIZE] = {};
        static double reynolds_zz[CROSS_SECTION_SIZE] = {};
        static double reynolds_xy[CROSS_SECTION_SIZE] = {};
        static double reynolds_xz[CROSS_SECTION_SIZE] = {};
        static double reynolds_yz[CROSS_SECTION_SIZE] = {};
        static double kinetic_energy[CROSS_SECTION_SIZE] = {};
        static double invariant_2[CROSS_SECTION_SIZE] = {};
        static double invariant_3[CROSS_SECTION_SIZE] = {};

        constexpr double K_TOLERANCE = 1.0e-20;
        for (int cross_index = 0; cross_index < CROSS_SECTION_SIZE; ++cross_index) {
            reynolds_xx[cross_index] = mean_uxux[cross_index] -
                mean_ux[cross_index] * mean_ux[cross_index];
            reynolds_yy[cross_index] = mean_uyuy[cross_index] -
                mean_uy[cross_index] * mean_uy[cross_index];
            reynolds_zz[cross_index] = mean_uzuz[cross_index] -
                mean_uz[cross_index] * mean_uz[cross_index];
            reynolds_xy[cross_index] = mean_uxuy[cross_index] -
                mean_ux[cross_index] * mean_uy[cross_index];
            reynolds_xz[cross_index] = mean_uxuz[cross_index] -
                mean_ux[cross_index] * mean_uz[cross_index];
            reynolds_yz[cross_index] = mean_uyuz[cross_index] -
                mean_uy[cross_index] * mean_uz[cross_index];

            kinetic_energy[cross_index] = 0.5 *
                (reynolds_xx[cross_index] + reynolds_yy[cross_index] +
                 reynolds_zz[cross_index]);
            invariant_2[cross_index] = 0.0;
            invariant_3[cross_index] = 0.0;

            if (kinetic_energy[cross_index] > K_TOLERANCE) {
                double anisotropy[3][3] = {};
                const double inverse_two_k =
                    1.0 / (2.0 * kinetic_energy[cross_index]);
                const double reynolds[3][3] = {
                    {reynolds_xx[cross_index], reynolds_xy[cross_index], reynolds_xz[cross_index]},
                    {reynolds_xy[cross_index], reynolds_yy[cross_index], reynolds_yz[cross_index]},
                    {reynolds_xz[cross_index], reynolds_yz[cross_index], reynolds_zz[cross_index]}
                };

                for (int i = 0; i < 3; ++i) {
                    for (int j = 0; j < 3; ++j) {
                        anisotropy[i][j] = reynolds[i][j] * inverse_two_k;
                        if (i == j) {
                            anisotropy[i][j] -= 1.0 / 3.0;
                        }
                    }
                }

                double trace_b_squared = 0.0;
                for (int i = 0; i < 3; ++i) {
                    for (int j = 0; j < 3; ++j) {
                        trace_b_squared += anisotropy[i][j] * anisotropy[j][i];
                    }
                }
                invariant_2[cross_index] = -0.5 * trace_b_squared;
                invariant_3[cross_index] =
                    anisotropy[0][0] *
                        (anisotropy[1][1] * anisotropy[2][2] -
                         anisotropy[1][2] * anisotropy[2][1]) -
                    anisotropy[0][1] *
                        (anisotropy[1][0] * anisotropy[2][2] -
                         anisotropy[1][2] * anisotropy[2][0]) +
                    anisotropy[0][2] *
                        (anisotropy[1][0] * anisotropy[2][1] -
                         anisotropy[1][1] * anisotropy[2][0]);
            }
        }

        const auto save_matrix = [&](const char* statistic_name,
                                     const double* values) {
            std::ostringstream matrix;
            matrix << std::scientific << std::setprecision(17);

            for (int y = 0; y < NY; ++y) {
                for (int x = 0; x < NX; ++x) {
                    if (x > 0 || y > 0) {
                        matrix << ',';
                    }
                    matrix << values[x + NX * y];
                }
            }

            std::ostringstream file_name;
            file_name << "_turbulent_" << statistic_name;

            // A fresh process also starts fresh running averages. Truncate
            // old files on its first export, then append one field per line.
            const int file_step = first_turbulence_export
                ? REPORT_SAVE
                : static_cast<int>(step);
            saveTreatData(file_name.str(), matrix.str(), file_step);
        };

        if (first_turbulence_export) {
            std::ostringstream metadata;
            metadata
                << "format=square_duct_turbulence_fields_v1\n"
                << "nx=" << NX << '\n'
                << "ny=" << NY << '\n'
                << "values_per_line=" << CROSS_SECTION_SIZE << '\n'
                << "ordering=x_fastest_then_y\n"
                << "delimiter=comma\n"
                << "numeric_format=scientific_17_digits\n"
                << "first_step=" << TURBULENCE_STATS_START_STEP << '\n'
                << "step_interval=" << TURBULENCE_STATS_SAVE_INTERVAL << '\n'
                << "sampling_interval=" << REPORT_SAVE << '\n'
                << "line_step_formula=first_step+line_index*step_interval\n"
                << "files=rho,mean_ux,mean_uy,mean_uz,Rxx,Ryy,Rzz,"
                << "Rxy,Rxz,Ryz,k,II_b,III_b";
            saveTreatData(
                "_turbulent_metadata", metadata.str(), REPORT_SAVE);
        }

        save_matrix("rho", mean_rho);
        save_matrix("mean_ux", mean_ux);
        save_matrix("mean_uy", mean_uy);
        save_matrix("mean_uz", mean_uz);
        save_matrix("Rxx", reynolds_xx);
        save_matrix("Ryy", reynolds_yy);
        save_matrix("Rzz", reynolds_zz);
        save_matrix("Rxy", reynolds_xy);
        save_matrix("Rxz", reynolds_xz);
        save_matrix("Ryz", reynolds_yz);
        save_matrix("k", kinetic_energy);
        save_matrix("II_b", invariant_2);
        save_matrix("III_b", invariant_3);
        first_turbulence_export = false;
    }
}
