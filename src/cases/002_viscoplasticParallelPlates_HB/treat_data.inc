/* ============================================================================
 * DATA TREATMENT: VELOCITY PROFILE & L2 ERROR VALIDATION
 * Domain height H = (NY - 1) / 2 with boundary walls at y = 0 and y = NY - 1
 * ========================================================================== */

dfloat uz_sim[NY];
dfloat uz_exact[NY];

// 1. Reset profile accumulator
for (int y = 0; y < NY; y++) {
    uz_sim[y] = 0.0_df;
}

// 2. Compute X-Z plane averaged velocity profile
for (int y = 0; y < NY; y++) {
    for (int z = 0; z < NZ_TOTAL; z++) {
        for (int x = 0; x < NX; x++) {
            dfloat t_uz = h_fMom[idxMom(x % BLOCK_NX, y % BLOCK_NY, z % BLOCK_NZ, M_UZ_INDEX, x / BLOCK_NX, y / BLOCK_NY, z / BLOCK_NZ)] / F_M_I_SCALE;
            uz_sim[y] += t_uz;
        }
    }
    uz_sim[y] /= static_cast<dfloat>(NX * NZ_TOTAL);
}

// 3. Compute exact Herschel-Bulkley analytical solution using H = (NY - 1) / 2
const dfloat H_chan = (static_cast<dfloat>(NY) - 1.0_df) / 2.0_df;
const dfloat y_plug = H_chan - (S_Y / FZ);
const dfloat inv_n  = 1.0_df / N_INDEX;
const dfloat exp_hb = (N_INDEX + 1.0_df) / N_INDEX;
const dfloat prefactor = POW_FUNCTION(FZ / K_CONSISTENCY, inv_n) * (N_INDEX / (N_INDEX + 1.0_df));

for (int y = 0; y < NY; y++) {
    dfloat y_pos = static_cast<dfloat>(y);
    dfloat d = (y_pos < H_chan) ? y_pos : (2.0_df * H_chan - y_pos); // Distance from nearest wall node

    if (d >= y_plug) {
        uz_exact[y] = prefactor * POW_FUNCTION(y_plug, exp_hb);
    } else {
        uz_exact[y] = prefactor * (POW_FUNCTION(y_plug, exp_hb) - POW_FUNCTION(y_plug - d, exp_hb));
    }
}

// 4. Compute L2 Error Norms
dfloat l2_diff_sq = 0.0_df;
dfloat l2_exact_sq = 0.0_df;

for (int y = 0; y < NY; y++) {
    dfloat diff = uz_sim[y] - uz_exact[y];
    l2_diff_sq  += diff * diff;
    l2_exact_sq += uz_exact[y] * uz_exact[y];
}

dfloat l2_relative_error = std::sqrt(l2_diff_sq / (l2_exact_sq + 1e-30_df));
dfloat l2_absolute_error = std::sqrt(l2_diff_sq / static_cast<dfloat>(NY));

// 5. Export minimal dataset
std::ostringstream str_uz_sim("");
std::ostringstream str_uz_exact("");
std::ostringstream str_l2("");

str_uz_sim   << "step," << step;
str_uz_exact << "step," << step;
str_l2       << "step," << step << "," << l2_relative_error << "," << l2_absolute_error;

for (int y = 0; y < NY; y++) {
    str_uz_sim   << "," << uz_sim[y];
    str_uz_exact << "," << uz_exact[y];
}

saveTreatData("_hb_uz_sim",   str_uz_sim.str(),   step);
saveTreatData("_hb_uz_exact", str_uz_exact.str(), step);
saveTreatData("_hb_l2_error", str_l2.str(),       step);