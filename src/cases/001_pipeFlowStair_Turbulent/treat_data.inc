// Cylindrical turbulence statistics for a pipe aligned with the z axis.
// Velocity components are rotated at every node before averaging so that the
// radial and azimuthal normal stresses are not mixed by the annular average.
double shell_radius[N_RADIAL_BINS] = {};
double shell_rho[N_RADIAL_BINS] = {};
double shell_ur[N_RADIAL_BINS] = {};
double shell_ut[N_RADIAL_BINS] = {};
double shell_uz[N_RADIAL_BINS] = {};
double shell_urur[N_RADIAL_BINS] = {};
double shell_utut[N_RADIAL_BINS] = {};
double shell_uzuz[N_RADIAL_BINS] = {};
double shell_urut[N_RADIAL_BINS] = {};
double shell_uruz[N_RADIAL_BINS] = {};
double shell_utuz[N_RADIAL_BINS] = {};
double shell_ur3[N_RADIAL_BINS] = {};
double shell_ut3[N_RADIAL_BINS] = {};
double shell_uz3[N_RADIAL_BINS] = {};
double shell_ur4[N_RADIAL_BINS] = {};
double shell_ut4[N_RADIAL_BINS] = {};
double shell_uz4[N_RADIAL_BINS] = {};
double shell_Srr[N_RADIAL_BINS] = {};
double shell_Stt[N_RADIAL_BINS] = {};
double shell_Szz[N_RADIAL_BINS] = {};
double shell_Srt[N_RADIAL_BINS] = {};
double shell_Srz[N_RADIAL_BINS] = {};
double shell_Stz[N_RADIAL_BINS] = {};
double shell_SGrr[N_RADIAL_BINS] = {};
double shell_SGtt[N_RADIAL_BINS] = {};
double shell_SGzz[N_RADIAL_BINS] = {};
double shell_SGrt[N_RADIAL_BINS] = {};
double shell_SGrz[N_RADIAL_BINS] = {};
double shell_SGtz[N_RADIAL_BINS] = {};
unsigned long long shell_count[N_RADIAL_BINS] = {};

const double omega = static_cast<double>(OMEGA);
const double force_x = static_cast<double>(FX);
const double force_y = static_cast<double>(FY);
const double force_z = static_cast<double>(FZ);
const double one_minus_half_omega = 1.0 - 0.5*omega;
const double inverse_viscosity = 6.0*omega/(2.0-omega);

for (int z = 0; z < NZ_TOTAL; ++z) {
    for (int y = 0; y < NY; ++y) {
        for (int x = 0; x < NX; ++x) {
            const double dx = static_cast<double>(x) - static_cast<double>(DUCT_CENTER_X);
            const double dy = static_cast<double>(y) - static_cast<double>(DUCT_CENTER_Y);
            const double radius = sqrt(dx*dx + dy*dy);
            if (radius >= PIPE_RADIUS) {
                continue;
            }

            const int radial_bin = static_cast<int>(radius);
            if (radial_bin < 0 || radial_bin >= N_RADIAL_BINS) {
                continue;
            }

            double cos_theta = 1.0;
            double sin_theta = 0.0;
            if (radius > 0.0) {
                cos_theta = dx/radius;
                sin_theta = dy/radius;
            }

            const double ux = static_cast<double>(h_fMom[idxMom(
                x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M_UX_INDEX,
                x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)]) /
                static_cast<double>(F_M_I_SCALE);
            const double uy = static_cast<double>(h_fMom[idxMom(
                x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M_UY_INDEX,
                x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)]) /
                static_cast<double>(F_M_I_SCALE);
            const double uz = static_cast<double>(h_fMom[idxMom(
                x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M_UZ_INDEX,
                x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)]) /
                static_cast<double>(F_M_I_SCALE);
            const double rho = static_cast<double>(h_fMom[idxMom(
                x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M_RHO_INDEX,
                x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)]) +
                static_cast<double>(RHO_0);

            const double ur = cos_theta*ux + sin_theta*uy;
            const double ut = -sin_theta*ux + cos_theta*uy;

            double Sxx = static_cast<double>(h_fMom[idxMom(
                x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M_MXX_INDEX,
                x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)]) /
                static_cast<double>(F_M_II_SCALE);
            double Sxy = static_cast<double>(h_fMom[idxMom(
                x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M_MXY_INDEX,
                x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)]) /
                static_cast<double>(F_M_IJ_SCALE);
            double Sxz = static_cast<double>(h_fMom[idxMom(
                x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M_MXZ_INDEX,
                x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)]) /
                static_cast<double>(F_M_IJ_SCALE);
            double Syy = static_cast<double>(h_fMom[idxMom(
                x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M_MYY_INDEX,
                x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)]) /
                static_cast<double>(F_M_II_SCALE);
            double Syz = static_cast<double>(h_fMom[idxMom(
                x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M_MYZ_INDEX,
                x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)]) /
                static_cast<double>(F_M_IJ_SCALE);
            double Szz = static_cast<double>(h_fMom[idxMom(
                x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M_MZZ_INDEX,
                x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)]) /
                static_cast<double>(F_M_II_SCALE);

            // Recover the pre-collision second moments and form the same
            // stress measure used by the parallel-plates treatment.
            Sxx = (Sxx - omega*ux*ux - one_minus_half_omega*(2.0*ux*force_x)) /
                (1.0-omega);
            Sxy = (Sxy - omega*ux*uy - one_minus_half_omega*(ux*force_y + uy*force_x)) /
                (1.0-omega);
            Sxz = (Sxz - omega*ux*uz - one_minus_half_omega*(ux*force_z + uz*force_x)) /
                (1.0-omega);
            Syy = (Syy - omega*uy*uy - one_minus_half_omega*(2.0*uy*force_y)) /
                (1.0-omega);
            Syz = (Syz - omega*uy*uz - one_minus_half_omega*(uy*force_z + uz*force_y)) /
                (1.0-omega);
            Szz = (Szz - omega*uz*uz - one_minus_half_omega*(2.0*uz*force_z)) /
                (1.0-omega);

            Sxx = one_minus_half_omega*(
                Sxx - ux*ux + 0.5*(force_x*ux + ux*force_x));
            Sxy = one_minus_half_omega*(
                Sxy - ux*uy + 0.5*(force_x*uy + ux*force_y));
            Sxz = one_minus_half_omega*(
                Sxz - ux*uz + 0.5*(force_x*uz + ux*force_z));
            Syy = one_minus_half_omega*(
                Syy - uy*uy + 0.5*(force_y*uy + uy*force_y));
            Syz = one_minus_half_omega*(
                Syz - uy*uz + 0.5*(force_y*uz + uy*force_z));
            Szz = one_minus_half_omega*(
                Szz - uz*uz + 0.5*(force_z*uz + uz*force_z));

            // Rotate S_cyl = Q*S_cart*Q^T at the local azimuth.
            const double cos_squared = cos_theta*cos_theta;
            const double sin_squared = sin_theta*sin_theta;
            const double sin_cos = sin_theta*cos_theta;
            const double Srr = cos_squared*Sxx + 2.0*sin_cos*Sxy + sin_squared*Syy;
            const double Stt = sin_squared*Sxx - 2.0*sin_cos*Sxy + cos_squared*Syy;
            const double Srt = sin_cos*(Syy-Sxx) +
                (cos_squared-sin_squared)*Sxy;
            const double Srz = cos_theta*Sxz + sin_theta*Syz;
            const double Stz = -sin_theta*Sxz + cos_theta*Syz;

            shell_radius[radial_bin] += radius;
            shell_rho[radial_bin] += rho;
            shell_ur[radial_bin] += ur;
            shell_ut[radial_bin] += ut;
            shell_uz[radial_bin] += uz;
            shell_urur[radial_bin] += ur*ur;
            shell_utut[radial_bin] += ut*ut;
            shell_uzuz[radial_bin] += uz*uz;
            shell_urut[radial_bin] += ur*ut;
            shell_uruz[radial_bin] += ur*uz;
            shell_utuz[radial_bin] += ut*uz;
            shell_ur3[radial_bin] += ur*ur*ur;
            shell_ut3[radial_bin] += ut*ut*ut;
            shell_uz3[radial_bin] += uz*uz*uz;
            shell_ur4[radial_bin] += ur*ur*ur*ur;
            shell_ut4[radial_bin] += ut*ut*ut*ut;
            shell_uz4[radial_bin] += uz*uz*uz*uz;
            shell_Srr[radial_bin] += Srr;
            shell_Stt[radial_bin] += Stt;
            shell_Szz[radial_bin] += Szz;
            shell_Srt[radial_bin] += Srt;
            shell_Srz[radial_bin] += Srz;
            shell_Stz[radial_bin] += Stz;
            shell_SGrr[radial_bin] += Srr*Srr*inverse_viscosity;
            shell_SGtt[radial_bin] += Stt*Stt*inverse_viscosity;
            shell_SGzz[radial_bin] += Szz*Szz*inverse_viscosity;
            shell_SGrt[radial_bin] += Srt*Srt*inverse_viscosity;
            shell_SGrz[radial_bin] += Srz*Srz*inverse_viscosity;
            shell_SGtz[radial_bin] += Stz*Stz*inverse_viscosity;
            ++shell_count[radial_bin];
        }
    }
}

for (int radial_bin = 0; radial_bin < N_RADIAL_BINS; ++radial_bin) {
    if (shell_count[radial_bin] == 0) {
        continue;
    }
    const double inverse_count =
        1.0/static_cast<double>(shell_count[radial_bin]);
    shell_radius[radial_bin] *= inverse_count;
    shell_rho[radial_bin] *= inverse_count;
    shell_ur[radial_bin] *= inverse_count;
    shell_ut[radial_bin] *= inverse_count;
    shell_uz[radial_bin] *= inverse_count;
    shell_urur[radial_bin] *= inverse_count;
    shell_utut[radial_bin] *= inverse_count;
    shell_uzuz[radial_bin] *= inverse_count;
    shell_urut[radial_bin] *= inverse_count;
    shell_uruz[radial_bin] *= inverse_count;
    shell_utuz[radial_bin] *= inverse_count;
    shell_ur3[radial_bin] *= inverse_count;
    shell_ut3[radial_bin] *= inverse_count;
    shell_uz3[radial_bin] *= inverse_count;
    shell_ur4[radial_bin] *= inverse_count;
    shell_ut4[radial_bin] *= inverse_count;
    shell_uz4[radial_bin] *= inverse_count;
    shell_Srr[radial_bin] *= inverse_count;
    shell_Stt[radial_bin] *= inverse_count;
    shell_Szz[radial_bin] *= inverse_count;
    shell_Srt[radial_bin] *= inverse_count;
    shell_Srz[radial_bin] *= inverse_count;
    shell_Stz[radial_bin] *= inverse_count;
    shell_SGrr[radial_bin] *= inverse_count;
    shell_SGtt[radial_bin] *= inverse_count;
    shell_SGzz[radial_bin] *= inverse_count;
    shell_SGrt[radial_bin] *= inverse_count;
    shell_SGrz[radial_bin] *= inverse_count;
    shell_SGtz[radial_bin] *= inverse_count;
}

// These are instantaneous theta-z spatial statistics. Raw moments are saved
// at every report so temporal averaging and central-moment construction can be
// performed later by post-processing, matching the parallel-plates workflow.
double reynolds_rr[N_RADIAL_BINS] = {};
double reynolds_tt[N_RADIAL_BINS] = {};
double reynolds_zz[N_RADIAL_BINS] = {};
double reynolds_rt[N_RADIAL_BINS] = {};
double reynolds_rz[N_RADIAL_BINS] = {};
double reynolds_tz[N_RADIAL_BINS] = {};
double turb_kinetic[N_RADIAL_BINS] = {};
double invariant2[N_RADIAL_BINS] = {};
double invariant3[N_RADIAL_BINS] = {};

for (int radial_bin = 0; radial_bin < N_RADIAL_BINS; ++radial_bin) {
    reynolds_rr[radial_bin] =
        shell_urur[radial_bin] - shell_ur[radial_bin]*shell_ur[radial_bin];
    reynolds_tt[radial_bin] =
        shell_utut[radial_bin] - shell_ut[radial_bin]*shell_ut[radial_bin];
    reynolds_zz[radial_bin] =
        shell_uzuz[radial_bin] - shell_uz[radial_bin]*shell_uz[radial_bin];
    reynolds_rt[radial_bin] =
        shell_urut[radial_bin] - shell_ur[radial_bin]*shell_ut[radial_bin];
    reynolds_rz[radial_bin] =
        shell_uruz[radial_bin] - shell_ur[radial_bin]*shell_uz[radial_bin];
    reynolds_tz[radial_bin] =
        shell_utuz[radial_bin] - shell_ut[radial_bin]*shell_uz[radial_bin];

    // Negative normal stresses can only be roundoff here; clamp before using k.
    if (reynolds_rr[radial_bin] < 0.0) reynolds_rr[radial_bin] = 0.0;
    if (reynolds_tt[radial_bin] < 0.0) reynolds_tt[radial_bin] = 0.0;
    if (reynolds_zz[radial_bin] < 0.0) reynolds_zz[radial_bin] = 0.0;

    double R[3][3] = {};
    R[0][0] = reynolds_rr[radial_bin];
    R[1][1] = reynolds_tt[radial_bin];
    R[2][2] = reynolds_zz[radial_bin];
    R[0][1] = R[1][0] = reynolds_rt[radial_bin];
    R[0][2] = R[2][0] = reynolds_rz[radial_bin];
    R[1][2] = R[2][1] = reynolds_tz[radial_bin];

    turb_kinetic[radial_bin] =
        0.5*(R[0][0] + R[1][1] + R[2][2]);
    constexpr double K_TOLERANCE = 1.0e-20;
    if (turb_kinetic[radial_bin] <= K_TOLERANCE) {
        continue;
    }

    double bij[3][3] = {};
    const double inverse_two_k = 1.0/(2.0*turb_kinetic[radial_bin]);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            bij[i][j] = R[i][j]*inverse_two_k;
            if (i == j) {
                bij[i][j] -= 1.0/3.0;
            }
        }
    }

    double trace_b_squared = 0.0;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            trace_b_squared += bij[i][j]*bij[j][i];
        }
    }
    invariant2[radial_bin] = -0.5*trace_b_squared;
    invariant3[radial_bin] =
        bij[0][0]*(bij[1][1]*bij[2][2] - bij[1][2]*bij[2][1]) -
        bij[0][1]*(bij[1][0]*bij[2][2] - bij[1][2]*bij[2][0]) +
        bij[0][2]*(bij[1][0]*bij[2][1] - bij[1][1]*bij[2][0]);
}

auto save_radial_profile = [step](const char* suffix, const double* values) {
    std::ostringstream data;
    data << std::scientific << std::setprecision(17) << "step," << step;
    for (int radial_bin = 0; radial_bin < N_RADIAL_BINS; ++radial_bin) {
        data << "," << values[radial_bin];
    }
    saveTreatData(suffix, data.str(), step);
};

save_radial_profile("_turbulent_radius", shell_radius);
save_radial_profile("_turbulent_rho", shell_rho);
save_radial_profile("_turbulent_ur1", shell_ur);
save_radial_profile("_turbulent_ut1", shell_ut);
save_radial_profile("_turbulent_uz1", shell_uz);
save_radial_profile("_turbulent_ur_ur", shell_urur);
save_radial_profile("_turbulent_ut_ut", shell_utut);
save_radial_profile("_turbulent_uz_uz", shell_uzuz);
save_radial_profile("_turbulent_ur_ut", shell_urut);
save_radial_profile("_turbulent_ur_uz", shell_uruz);
save_radial_profile("_turbulent_ut_uz", shell_utuz);
save_radial_profile("_turbulent_ur3", shell_ur3);
save_radial_profile("_turbulent_ut3", shell_ut3);
save_radial_profile("_turbulent_uz3", shell_uz3);
save_radial_profile("_turbulent_ur4", shell_ur4);
save_radial_profile("_turbulent_ut4", shell_ut4);
save_radial_profile("_turbulent_uz4", shell_uz4);
save_radial_profile("_turbulent_Srr", shell_Srr);
save_radial_profile("_turbulent_Stt", shell_Stt);
save_radial_profile("_turbulent_Szz", shell_Szz);
save_radial_profile("_turbulent_Srt", shell_Srt);
save_radial_profile("_turbulent_Srz", shell_Srz);
save_radial_profile("_turbulent_Stz", shell_Stz);
save_radial_profile("_turbulent_SGrr", shell_SGrr);
save_radial_profile("_turbulent_SGtt", shell_SGtt);
save_radial_profile("_turbulent_SGzz", shell_SGzz);
save_radial_profile("_turbulent_SGrt", shell_SGrt);
save_radial_profile("_turbulent_SGrz", shell_SGrz);
save_radial_profile("_turbulent_SGtz", shell_SGtz);
save_radial_profile("_turbulent_Rrr", reynolds_rr);
save_radial_profile("_turbulent_Rtt", reynolds_tt);
save_radial_profile("_turbulent_Rzz", reynolds_zz);
save_radial_profile("_turbulent_Rrt", reynolds_rt);
save_radial_profile("_turbulent_Rrz", reynolds_rz);
save_radial_profile("_turbulent_Rtz", reynolds_tz);
save_radial_profile("_turbulent_k", turb_kinetic);
save_radial_profile("_turbulent_II", invariant2);
save_radial_profile("_turbulent_III", invariant3);

// Axial lines at theta = 0 and four center-to-wall radii. At theta = 0,
// ux = ur and uy = u_theta, so the three files are cylindrical components.
const double profile_radius_fraction[4] = {0.0, 0.25, 0.50, 0.90};
const int profile_y = static_cast<int>(DUCT_CENTER_Y + 0.5);
for (int profile = 0; profile < 4; ++profile) {
    const int profile_x = static_cast<int>(
        DUCT_CENTER_X + profile_radius_fraction[profile]*PIPE_RADIUS + 0.5);
    velocityProfile(h_fMom, 7, profile_x, profile_y, 0, step); // ur along z
    velocityProfile(h_fMom, 8, profile_x, profile_y, 0, step); // u_theta along z
    velocityProfile(h_fMom, 9, profile_x, profile_y, 0, step); // uz along z
}
