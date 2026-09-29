// Turbulence statistics on four rays starting at the outer-duct center.
// Each fluid point is averaged only along periodic z; no radial/azimuthal
// averaging is used because it would erase the eccentric-annulus asymmetry.
// Solid points at the start of a ray (inside the inner cylinder) are skipped.

constexpr int RAY_COUNT = 4;
constexpr int MAX_RAY_SAMPLES = HALF_WIDTH;
const int ray_dx[RAY_COUNT] = {1, -1, 0, 0};
const int ray_dy[RAY_COUNT] = {0, 0, 1, -1};
const char* const ray_name[RAY_COUNT] = {"xp", "xm", "yp", "ym"};

enum RayField {
    RF_RHO, RF_RHO2, RF_RHO3, RF_RHO4,
    RF_UR1, RF_UT1, RF_UZ1,
    RF_UR_UR, RF_UT_UT, RF_UZ_UZ, RF_UR_UT, RF_UR_UZ, RF_UT_UZ,
    RF_UR3, RF_UT3, RF_UZ3, RF_UR4, RF_UT4, RF_UZ4,
    RF_UR_UT_UT, RF_UR_UZ_UZ, RF_UR_UR_UZ, RF_UT_UT_UZ,
    RF_UR_PGR, RF_UT_PGT, RF_UZ_PGZ, RF_UR_PGZ, RF_UZ_PGR,
    RF_GRAD_RR, RF_GRAD_TT, RF_GRAD_ZZ, RF_GRAD_RZ,
    RF_SRR, RF_STT, RF_SZZ, RF_SRT, RF_SRZ, RF_STZ,
    RF_SGRR, RF_SGTT, RF_SGZZ, RF_SGRT, RF_SGRZ, RF_SGTZ,
    RF_COUNT
};

const char* const ray_field_name[RF_COUNT] = {
    "rho", "rho2", "rho3", "rho4", "ur1", "ut1", "uz1",
    "ur_ur", "ut_ut", "uz_uz", "ur_ut", "ur_uz", "ut_uz",
    "ur3", "ut3", "uz3", "ur4", "ut4", "uz4",
    "ur_ut_ut", "ur_uz_uz", "ur_ur_uz", "ut_ut_uz",
    "ur_pgr", "ut_pgt", "uz_pgz", "ur_pgz", "uz_pgr",
    "grad_rr", "grad_tt", "grad_zz", "grad_rz",
    "Srr", "Stt", "Szz", "Srt", "Srz", "Stz",
    "SGrr", "SGtt", "SGzz", "SGrt", "SGrz", "SGtz"
};

enum RayDerivedField {
    RDF_RRR, RDF_RTT, RDF_RZZ, RDF_RRT, RDF_RRZ, RDF_RTZ,
    RDF_K, RDF_II, RDF_III, RDF_COUNT
};
const char* const ray_derived_name[RDF_COUNT] = {
    "Rrr", "Rtt", "Rzz", "Rrt", "Rrz", "Rtz", "k", "II", "III"
};

double ray_sum[RAY_COUNT][RF_COUNT][MAX_RAY_SAMPLES] = {};
double ray_derived[RAY_COUNT][RDF_COUNT][MAX_RAY_SAMPLES] = {};
double ray_distance[RAY_COUNT][MAX_RAY_SAMPLES] = {};
double ray_x_coordinate[RAY_COUNT][MAX_RAY_SAMPLES] = {};
double ray_y_coordinate[RAY_COUNT][MAX_RAY_SAMPLES] = {};
double ray_outer_wall_distance[RAY_COUNT][MAX_RAY_SAMPLES] = {};
double ray_inner_wall_distance[RAY_COUNT][MAX_RAY_SAMPLES] = {};
unsigned long long ray_sample_count[RAY_COUNT][MAX_RAY_SAMPLES] = {};
int ray_point_count[RAY_COUNT] = {};
int ray_grid_x[RAY_COUNT][MAX_RAY_SAMPLES] = {};
int ray_grid_y[RAY_COUNT][MAX_RAY_SAMPLES] = {};

#ifdef NON_NEWTONIAN_FLUID
double ray_yield_probability[RAY_COUNT][MAX_RAY_SAMPLES] = {};
double ray_inv_omega[RAY_COUNT][MAX_RAY_SAMPLES] = {};
double ray_inv_omega2[RAY_COUNT][MAX_RAY_SAMPLES] = {};
unsigned long long ray_yielded_count[RAY_COUNT][MAX_RAY_SAMPLES] = {};
#ifdef OMEGA_CUTOFF
const double non_newtonian_omega_cutoff = static_cast<double>(OMEGA_CUTOFF);
#else
const double non_newtonian_omega_cutoff = 1.0e-3;
#endif
#endif

const double base_omega = static_cast<double>(OMEGA);
const double force_x = static_cast<double>(FX);
const double force_y = static_cast<double>(FY);
const double force_z = static_cast<double>(FZ);
const int duct_center_x = static_cast<int>(OUTER_CENTER_X + 0.5);
const int duct_center_y = static_cast<int>(OUTER_CENTER_Y + 0.5);

auto is_fluid = [](int x, int y) {
    if (x < 0 || x >= NX || y < 0 || y >= NY) return false;
    const double odx = static_cast<double>(x) - static_cast<double>(OUTER_CENTER_X);
    const double ody = static_cast<double>(y) - static_cast<double>(OUTER_CENTER_Y);
    const double idx = static_cast<double>(x) - static_cast<double>(INNER_CENTER_X);
    const double idy = static_cast<double>(y) - static_cast<double>(INNER_CENTER_Y);
    return odx*odx + ody*ody < static_cast<double>(OUTER_RADIUS)*OUTER_RADIUS &&
        idx*idx + idy*idy > static_cast<double>(INNER_RADIUS)*INNER_RADIUS;
};

auto density_at = [&](int x, int y, int z) {
    const int zp = (z + NZ_TOTAL)%NZ_TOTAL;
    return static_cast<double>(h_fMom[idxMom(
        x%BLOCK_NX, y%BLOCK_NY, zp%BLOCK_NZ, M_RHO_INDEX,
        x/BLOCK_NX, y/BLOCK_NY, zp/BLOCK_NZ)]) + static_cast<double>(RHO_0);
};
auto velocity_at = [&](int x, int y, int z, int component) {
    const int zp = (z + NZ_TOTAL)%NZ_TOTAL;
    const int moment = component == 0 ? M_UX_INDEX :
        (component == 1 ? M_UY_INDEX : M_UZ_INDEX);
    return static_cast<double>(h_fMom[idxMom(
        x%BLOCK_NX, y%BLOCK_NY, zp%BLOCK_NZ, moment,
        x/BLOCK_NX, y/BLOCK_NY, zp/BLOCK_NZ)]) /
        static_cast<double>(F_M_I_SCALE);
};

for (int ray = 0; ray < RAY_COUNT; ++ray) {
    // These are the outer-cylinder cylindrical basis vectors on each axis.
    const double cos_theta = static_cast<double>(ray_dx[ray]);
    const double sin_theta = static_cast<double>(ray_dy[ray]);

    for (int offset = 0; offset < MAX_RAY_SAMPLES; ++offset) {
        const int x = duct_center_x + ray_dx[ray]*offset;
        const int y = duct_center_y + ray_dy[ray]*offset;
        if (!is_fluid(x, y)) continue;

        const int point = ray_point_count[ray]++;
        ray_grid_x[ray][point] = x;
        ray_grid_y[ray][point] = y;
        const double odx = static_cast<double>(x) - static_cast<double>(OUTER_CENTER_X);
        const double ody = static_cast<double>(y) - static_cast<double>(OUTER_CENTER_Y);
        const double idx = static_cast<double>(x) - static_cast<double>(INNER_CENTER_X);
        const double idy = static_cast<double>(y) - static_cast<double>(INNER_CENTER_Y);
        const double distance = sqrt(odx*odx + ody*ody);
        ray_distance[ray][point] = distance;
        ray_x_coordinate[ray][point] = static_cast<double>(x);
        ray_y_coordinate[ray][point] = static_cast<double>(y);
        ray_outer_wall_distance[ray][point] =
            static_cast<double>(OUTER_RADIUS) - distance;
        ray_inner_wall_distance[ray][point] =
            sqrt(idx*idx + idy*idy) - static_cast<double>(INNER_RADIUS);

        for (int z = 0; z < NZ_TOTAL; ++z) {
            const double ux = velocity_at(x, y, z, 0);
            const double uy = velocity_at(x, y, z, 1);
            const double uz = velocity_at(x, y, z, 2);
            const double rho = density_at(x, y, z);
#ifdef OMEGA_FIELD
            const double local_omega = static_cast<double>(h_fMom[idxMom(
                x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M_OMEGA_INDEX,
                x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)]);
#else
            const double local_omega = base_omega;
#endif
            const double one_minus_half_omega = 1.0 - 0.5*local_omega;
            const double viscosity_denominator = 2.0-local_omega;
            const double inverse_viscosity = local_omega > 0.0 &&
                fabs(viscosity_denominator) > 1.0e-14 ?
                6.0*local_omega/viscosity_denominator : 0.0;
            const double ur = cos_theta*ux + sin_theta*uy;
            const double ut = -sin_theta*ux + cos_theta*uy;

            auto scalar_derivative = [&](int axis) {
                if (axis == 2)
                    return 0.5*(density_at(x, y, z+1) - density_at(x, y, z-1));
                const int ddx = axis == 0 ? 1 : 0;
                const int ddy = axis == 1 ? 1 : 0;
                const bool minus = is_fluid(x-ddx, y-ddy);
                const bool plus = is_fluid(x+ddx, y+ddy);
                if (minus && plus)
                    return 0.5*(density_at(x+ddx, y+ddy, z) -
                        density_at(x-ddx, y-ddy, z));
                if (plus) return density_at(x+ddx, y+ddy, z) - rho;
                if (minus) return rho - density_at(x-ddx, y-ddy, z);
                return 0.0;
            };
            auto velocity_derivative = [&](int component, int axis, double center) {
                if (axis == 2)
                    return 0.5*(velocity_at(x, y, z+1, component) -
                        velocity_at(x, y, z-1, component));
                const int ddx = axis == 0 ? 1 : 0;
                const int ddy = axis == 1 ? 1 : 0;
                const bool minus = is_fluid(x-ddx, y-ddy);
                const bool plus = is_fluid(x+ddx, y+ddy);
                if (minus && plus)
                    return 0.5*(velocity_at(x+ddx, y+ddy, z, component) -
                        velocity_at(x-ddx, y-ddy, z, component));
                if (plus) return velocity_at(x+ddx, y+ddy, z, component) - center;
                if (minus) return center - velocity_at(x-ddx, y-ddy, z, component);
                return 0.0;
            };

            const double pgx = scalar_derivative(0)/3.0;
            const double pgy = scalar_derivative(1)/3.0;
            const double pgz = scalar_derivative(2)/3.0;
            const double pgr = cos_theta*pgx + sin_theta*pgy;
            const double pgt = -sin_theta*pgx + cos_theta*pgy;

            double grad_cart[3][3] = {};
            const double velocity[3] = {ux, uy, uz};
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j)
                    grad_cart[i][j] = velocity_derivative(i, j, velocity[i]);
            const double rotation[3][3] = {
                {cos_theta, sin_theta, 0.0},
                {-sin_theta, cos_theta, 0.0},
                {0.0, 0.0, 1.0}
            };
            double grad_cyl[3][3] = {};
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j)
                    for (int a = 0; a < 3; ++a)
                        for (int b = 0; b < 3; ++b)
                            grad_cyl[i][j] += rotation[i][a]*grad_cart[a][b]*rotation[j][b];

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

            // Recover the same pre-collision stress measure used by the pipe
            // and parallel-plate turbulence treatments.
            const double collision_denominator = 1.0-local_omega;
            if (fabs(collision_denominator) > 1.0e-14) {
                Sxx = (Sxx - local_omega*ux*ux -
                    one_minus_half_omega*(2.0*ux*force_x))/collision_denominator;
                Sxy = (Sxy - local_omega*ux*uy -
                    one_minus_half_omega*(ux*force_y + uy*force_x))/collision_denominator;
                Sxz = (Sxz - local_omega*ux*uz -
                    one_minus_half_omega*(ux*force_z + uz*force_x))/collision_denominator;
                Syy = (Syy - local_omega*uy*uy -
                    one_minus_half_omega*(2.0*uy*force_y))/collision_denominator;
                Syz = (Syz - local_omega*uy*uz -
                    one_minus_half_omega*(uy*force_z + uz*force_y))/collision_denominator;
                Szz = (Szz - local_omega*uz*uz -
                    one_minus_half_omega*(2.0*uz*force_z))/collision_denominator;
            } else {
                Sxx = Sxy = Sxz = Syy = Syz = Szz = 0.0;
            }
            Sxx = one_minus_half_omega*(Sxx - ux*ux + force_x*ux);
            Sxy = one_minus_half_omega*(Sxy - ux*uy +
                0.5*(force_x*uy + ux*force_y));
            Sxz = one_minus_half_omega*(Sxz - ux*uz +
                0.5*(force_x*uz + ux*force_z));
            Syy = one_minus_half_omega*(Syy - uy*uy + force_y*uy);
            Syz = one_minus_half_omega*(Syz - uy*uz +
                0.5*(force_y*uz + uy*force_z));
            Szz = one_minus_half_omega*(Szz - uz*uz + force_z*uz);

            const double c2 = cos_theta*cos_theta;
            const double s2 = sin_theta*sin_theta;
            const double sc = sin_theta*cos_theta;
            const double Srr = c2*Sxx + 2.0*sc*Sxy + s2*Syy;
            const double Stt = s2*Sxx - 2.0*sc*Sxy + c2*Syy;
            const double Srt = sc*(Syy-Sxx) + (c2-s2)*Sxy;
            const double Srz = cos_theta*Sxz + sin_theta*Syz;
            const double Stz = -sin_theta*Sxz + cos_theta*Syz;

            const double drho = rho - static_cast<double>(RHO_0);
            ray_sum[ray][RF_RHO][point] += rho;
            ray_sum[ray][RF_RHO2][point] += drho*drho;
            ray_sum[ray][RF_RHO3][point] += drho*drho*drho;
            ray_sum[ray][RF_RHO4][point] += drho*drho*drho*drho;
            ray_sum[ray][RF_UR1][point] += ur;
            ray_sum[ray][RF_UT1][point] += ut;
            ray_sum[ray][RF_UZ1][point] += uz;
            ray_sum[ray][RF_UR_UR][point] += ur*ur;
            ray_sum[ray][RF_UT_UT][point] += ut*ut;
            ray_sum[ray][RF_UZ_UZ][point] += uz*uz;
            ray_sum[ray][RF_UR_UT][point] += ur*ut;
            ray_sum[ray][RF_UR_UZ][point] += ur*uz;
            ray_sum[ray][RF_UT_UZ][point] += ut*uz;
            ray_sum[ray][RF_UR3][point] += ur*ur*ur;
            ray_sum[ray][RF_UT3][point] += ut*ut*ut;
            ray_sum[ray][RF_UZ3][point] += uz*uz*uz;
            ray_sum[ray][RF_UR4][point] += ur*ur*ur*ur;
            ray_sum[ray][RF_UT4][point] += ut*ut*ut*ut;
            ray_sum[ray][RF_UZ4][point] += uz*uz*uz*uz;
            ray_sum[ray][RF_UR_UT_UT][point] += ur*ut*ut;
            ray_sum[ray][RF_UR_UZ_UZ][point] += ur*uz*uz;
            ray_sum[ray][RF_UR_UR_UZ][point] += ur*ur*uz;
            ray_sum[ray][RF_UT_UT_UZ][point] += ut*ut*uz;
            ray_sum[ray][RF_UR_PGR][point] += ur*pgr;
            ray_sum[ray][RF_UT_PGT][point] += ut*pgt;
            ray_sum[ray][RF_UZ_PGZ][point] += uz*pgz;
            ray_sum[ray][RF_UR_PGZ][point] += ur*pgz;
            ray_sum[ray][RF_UZ_PGR][point] += uz*pgr;
            for (int direction = 0; direction < 3; ++direction) {
                ray_sum[ray][RF_GRAD_RR][point] += grad_cyl[0][direction]*grad_cyl[0][direction];
                ray_sum[ray][RF_GRAD_TT][point] += grad_cyl[1][direction]*grad_cyl[1][direction];
                ray_sum[ray][RF_GRAD_ZZ][point] += grad_cyl[2][direction]*grad_cyl[2][direction];
                ray_sum[ray][RF_GRAD_RZ][point] += grad_cyl[0][direction]*grad_cyl[2][direction];
            }
            ray_sum[ray][RF_SRR][point] += Srr;
            ray_sum[ray][RF_STT][point] += Stt;
            ray_sum[ray][RF_SZZ][point] += Szz;
            ray_sum[ray][RF_SRT][point] += Srt;
            ray_sum[ray][RF_SRZ][point] += Srz;
            ray_sum[ray][RF_STZ][point] += Stz;
            ray_sum[ray][RF_SGRR][point] += Srr*Srr*inverse_viscosity;
            ray_sum[ray][RF_SGTT][point] += Stt*Stt*inverse_viscosity;
            ray_sum[ray][RF_SGZZ][point] += Szz*Szz*inverse_viscosity;
            ray_sum[ray][RF_SGRT][point] += Srt*Srt*inverse_viscosity;
            ray_sum[ray][RF_SGRZ][point] += Srz*Srz*inverse_viscosity;
            ray_sum[ray][RF_SGTZ][point] += Stz*Stz*inverse_viscosity;
#ifdef NON_NEWTONIAN_FLUID
            if (local_omega < non_newtonian_omega_cutoff) {
                ray_yield_probability[ray][point] += 1.0;
            } else {
                const double inv_omega = 1.0/local_omega;
                ray_inv_omega[ray][point] += inv_omega;
                ray_inv_omega2[ray][point] += inv_omega*inv_omega;
                ++ray_yielded_count[ray][point];
            }
#endif
            ++ray_sample_count[ray][point];
        }
    }
}

// Normalize z sums, then form the instantaneous Reynolds tensor and its
// anisotropy invariants. Temporal averaging is deferred to post-processing.
for (int ray = 0; ray < RAY_COUNT; ++ray) {
    for (int point = 0; point < ray_point_count[ray]; ++point) {
        if (ray_sample_count[ray][point] == 0) continue;
        const double inv_count =
            1.0/static_cast<double>(ray_sample_count[ray][point]);
        for (int field = 0; field < RF_COUNT; ++field)
            ray_sum[ray][field][point] *= inv_count;
#ifdef NON_NEWTONIAN_FLUID
        ray_yield_probability[ray][point] *= inv_count;
        if (ray_yielded_count[ray][point] > 0) {
            const double inv_yielded =
                1.0/static_cast<double>(ray_yielded_count[ray][point]);
            ray_inv_omega[ray][point] *= inv_yielded;
            ray_inv_omega2[ray][point] *= inv_yielded;
        }
#endif

        double R[3][3] = {};
        R[0][0] = ray_sum[ray][RF_UR_UR][point] -
            ray_sum[ray][RF_UR1][point]*ray_sum[ray][RF_UR1][point];
        R[1][1] = ray_sum[ray][RF_UT_UT][point] -
            ray_sum[ray][RF_UT1][point]*ray_sum[ray][RF_UT1][point];
        R[2][2] = ray_sum[ray][RF_UZ_UZ][point] -
            ray_sum[ray][RF_UZ1][point]*ray_sum[ray][RF_UZ1][point];
        R[0][1] = R[1][0] = ray_sum[ray][RF_UR_UT][point] -
            ray_sum[ray][RF_UR1][point]*ray_sum[ray][RF_UT1][point];
        R[0][2] = R[2][0] = ray_sum[ray][RF_UR_UZ][point] -
            ray_sum[ray][RF_UR1][point]*ray_sum[ray][RF_UZ1][point];
        R[1][2] = R[2][1] = ray_sum[ray][RF_UT_UZ][point] -
            ray_sum[ray][RF_UT1][point]*ray_sum[ray][RF_UZ1][point];
        if (R[0][0] < 0.0) R[0][0] = 0.0;
        if (R[1][1] < 0.0) R[1][1] = 0.0;
        if (R[2][2] < 0.0) R[2][2] = 0.0;

        ray_derived[ray][RDF_RRR][point] = R[0][0];
        ray_derived[ray][RDF_RTT][point] = R[1][1];
        ray_derived[ray][RDF_RZZ][point] = R[2][2];
        ray_derived[ray][RDF_RRT][point] = R[0][1];
        ray_derived[ray][RDF_RRZ][point] = R[0][2];
        ray_derived[ray][RDF_RTZ][point] = R[1][2];
        const double kinetic_energy = 0.5*(R[0][0] + R[1][1] + R[2][2]);
        ray_derived[ray][RDF_K][point] = kinetic_energy;
        if (kinetic_energy <= 1.0e-20) continue;

        double bij[3][3] = {};
        const double inv_two_k = 1.0/(2.0*kinetic_energy);
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                bij[i][j] = R[i][j]*inv_two_k;
                if (i == j) bij[i][j] -= 1.0/3.0;
            }
        }
        double trace_b2 = 0.0;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                trace_b2 += bij[i][j]*bij[j][i];
        ray_derived[ray][RDF_II][point] = -0.5*trace_b2;
        ray_derived[ray][RDF_III][point] =
            bij[0][0]*(bij[1][1]*bij[2][2] - bij[1][2]*bij[2][1]) -
            bij[0][1]*(bij[1][0]*bij[2][2] - bij[1][2]*bij[2][0]) +
            bij[0][2]*(bij[1][0]*bij[2][1] - bij[1][1]*bij[2][0]);
    }
}

auto save_ray_profile = [step](const char* direction, const char* quantity,
    const double* values, int count) {
    std::ostringstream data;
    data << std::scientific << std::setprecision(17) << "step," << step;
    for (int point = 0; point < count; ++point) data << "," << values[point];
    std::string suffix = "_turbulent_";
    suffix += direction;
    suffix += "_";
    suffix += quantity;
    saveTreatData(suffix, data.str(), step);
};

for (int ray = 0; ray < RAY_COUNT; ++ray) {
    const int count = ray_point_count[ray];
    save_ray_profile(ray_name[ray], "distance", ray_distance[ray], count);
    save_ray_profile(ray_name[ray], "x", ray_x_coordinate[ray], count);
    save_ray_profile(ray_name[ray], "y", ray_y_coordinate[ray], count);
    save_ray_profile(ray_name[ray], "outer_wall_distance",
        ray_outer_wall_distance[ray], count);
    save_ray_profile(ray_name[ray], "inner_wall_distance",
        ray_inner_wall_distance[ray], count);
    for (int field = 0; field < RF_COUNT; ++field)
        save_ray_profile(ray_name[ray], ray_field_name[field],
            ray_sum[ray][field], count);
    for (int field = 0; field < RDF_COUNT; ++field)
        save_ray_profile(ray_name[ray], ray_derived_name[field],
            ray_derived[ray][field], count);
#ifdef NON_NEWTONIAN_FLUID
    save_ray_profile(ray_name[ray], "yield_prob",
        ray_yield_probability[ray], count);
    save_ray_profile(ray_name[ray], "inv_omega", ray_inv_omega[ray], count);
    save_ray_profile(ray_name[ray], "inv_omega_2", ray_inv_omega2[ray], count);
#endif

    // Mid-gap axial lines retain instantaneous z structure for spectra and
    // autocorrelations. Cartesian components can be rotated using the ray.
    if (count > 0) {
        const int middle = count/2;
        const int profile_x = ray_grid_x[ray][middle];
        const int profile_y = ray_grid_y[ray][middle];
        velocityProfile(h_fMom, 7, profile_x, profile_y, 0, step);
        velocityProfile(h_fMom, 8, profile_x, profile_y, 0, step);
        velocityProfile(h_fMom, 9, profile_x, profile_y, 0, step);
    }
}
