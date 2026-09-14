{
    // Sample at the channel mid-plane in X and Z
    const int cx = NX / 2;
    const int cz = NZ / 2;

    // Current oscillation phase [rad]
    const dfloat phase = std::fmod((dfloat)OMEGA_OSC * (dfloat)step,
                                   2.0 * M_PI);

    std::ostringstream oss;
    oss << std::scientific << std::setprecision(8);
    oss << step << "," << phase;

    for (int iy = 0; iy < NY; ++iy) {
        dfloat ux_val = h_fMom[idxMom(
            cx % BLOCK_NX, iy % BLOCK_NY, cz % BLOCK_NZ,
            M_UX_INDEX,
            cx / BLOCK_NX, iy / BLOCK_NY, cz / BLOCK_NZ)];
        oss << "," << ux_val;
    }

    saveTreatData("_ux_profile", oss.str(), step);
}
