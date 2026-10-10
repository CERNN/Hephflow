// Cylindrical turbulence statistics for a pipe aligned with the z axis.
// Velocity components are rotated at every node before averaging so that the
// radial and azimuthal normal stresses are not mixed by the annular average.
double shell_radius[N_RADIAL_BINS] = {};
double shell_min_radius[N_RADIAL_BINS] = {};
double shell_max_radius[N_RADIAL_BINS] = {};
double shell_wall_distance[N_RADIAL_BINS] = {};
double shell_min_wall_distance[N_RADIAL_BINS] = {};
double shell_max_wall_distance[N_RADIAL_BINS] = {};
double shell_rho[N_RADIAL_BINS] = {};
// Raw moments of rho-RHO_0 avoid catastrophic cancellation in p' moments.
double shell_rho2[N_RADIAL_BINS] = {};
double shell_rho3[N_RADIAL_BINS] = {};
double shell_rho4[N_RADIAL_BINS] = {};
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
// Mixed third moments needed by the cylindrical Reynolds-stress budgets.
double shell_ur_ut_ut[N_RADIAL_BINS] = {};
double shell_ur_uz_uz[N_RADIAL_BINS] = {};
double shell_ur_ur_uz[N_RADIAL_BINS] = {};
double shell_ut_ut_uz[N_RADIAL_BINS] = {};
// Raw velocity/pressure-gradient products. In isothermal LBM p = rho/3.
double shell_ur_pgr[N_RADIAL_BINS] = {};
double shell_ut_pgt[N_RADIAL_BINS] = {};
double shell_uz_pgz[N_RADIAL_BINS] = {};
double shell_ur_pgz[N_RADIAL_BINS] = {};
double shell_uz_pgr[N_RADIAL_BINS] = {};
// Sums over derivative directions of cylindrical velocity gradients.
double shell_grad_rr[N_RADIAL_BINS] = {};
double shell_grad_tt[N_RADIAL_BINS] = {};
double shell_grad_zz[N_RADIAL_BINS] = {};
double shell_grad_rz[N_RADIAL_BINS] = {};
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
#ifdef NON_NEWTONIAN_FLUID
// Viscoplastic/rheological statistics. inv(omega) moments are conditioned on
// yielded nodes so apparent-viscosity moments can be reconstructed safely.
double shell_yield_probability[N_RADIAL_BINS] = {};
double shell_inv_omega[N_RADIAL_BINS] = {};
double shell_inv_omega2[N_RADIAL_BINS] = {};
unsigned long long shell_yielded_count[N_RADIAL_BINS] = {};
#endif
unsigned long long shell_count[N_RADIAL_BINS] = {};
// Geometry chooses the radial bin; the solver's node type decides whether the
// node is a valid statistics sample. These raw counters diagnose the gap.
unsigned long long shell_candidate_count[N_RADIAL_BINS] = {};
unsigned long long shell_fluid_count[N_RADIAL_BINS] = {};
unsigned long long shell_rejected_count[N_RADIAL_BINS] = {};
unsigned long long shell_curved_bc_count[N_RADIAL_BINS] = {};
unsigned long long shell_solid_count[N_RADIAL_BINS] = {};
unsigned long long shell_other_node_count[N_RADIAL_BINS] = {};

for (int radial_bin = 0; radial_bin < N_RADIAL_BINS; ++radial_bin) {
    shell_min_radius[radial_bin] = static_cast<double>(PIPE_RADIUS);
    shell_min_wall_distance[radial_bin] = static_cast<double>(PIPE_RADIUS);
}

const double base_omega = static_cast<double>(OMEGA);
const double force_x = static_cast<double>(FX);
const double force_y = static_cast<double>(FY);
const double force_z = static_cast<double>(FZ);
#ifdef NON_NEWTONIAN_FLUID
#ifdef OMEGA_CUTOFF
const double non_newtonian_omega_cutoff = static_cast<double>(OMEGA_CUTOFF);
#else
const double non_newtonian_omega_cutoff = 1.0e-3;
#endif
#endif

auto node_type_at = [&](int x, int y, int z) {
    if (x < 0 || x >= NX || y < 0 || y >= NY) {
        return static_cast<unsigned int>(SOLID_NODE);
    }
    const int periodic_z = (z + NZ_TOTAL)%NZ_TOTAL;
    return h_nodeType[idxScalarBlock(
        x%BLOCK_NX, y%BLOCK_NY, periodic_z%BLOCK_NZ,
        x/BLOCK_NX, y/BLOCK_NY, periodic_z/BLOCK_NZ)];
};
auto is_fluid = [&](int x, int y, int z) {
    // BULK is the solver's canonical collision/streaming fluid-node type.
    // Curved-BC nodes have their own nonzero type and are deliberately not
    // sampled as bulk turbulence nodes.
    return node_type_at(x, y, z) == BULK;
};
auto density_at = [&](int x, int y, int z) {
    const int periodic_z = (z + NZ_TOTAL)%NZ_TOTAL;
    return static_cast<double>(h_fMom[idxMom(
        x%BLOCK_NX, y%BLOCK_NY, periodic_z%BLOCK_NZ, M_RHO_INDEX,
        x/BLOCK_NX, y/BLOCK_NY, periodic_z/BLOCK_NZ)]) +
        static_cast<double>(RHO_0);
};
auto velocity_at = [&](int x, int y, int z, int component) {
    const int periodic_z = (z + NZ_TOTAL)%NZ_TOTAL;
    const int moment_index = component == 0 ? M_UX_INDEX :
        (component == 1 ? M_UY_INDEX : M_UZ_INDEX);
    return static_cast<double>(h_fMom[idxMom(
        x%BLOCK_NX, y%BLOCK_NY, periodic_z%BLOCK_NZ, moment_index,
        x/BLOCK_NX, y/BLOCK_NY, periodic_z/BLOCK_NZ)]) /
        static_cast<double>(F_M_I_SCALE);
};

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

            ++shell_candidate_count[radial_bin];
            const unsigned int node_type = node_type_at(x, y, z);
            if (node_type != BULK) {
                ++shell_rejected_count[radial_bin];
                if (isCurvedBoundaryNodeType(node_type)) {
                    ++shell_curved_bc_count[radial_bin];
                } else if (node_type == SOLID_NODE) {
                    ++shell_solid_count[radial_bin];
                } else {
                    ++shell_other_node_count[radial_bin];
                }
#ifdef TURBULENCE_NODE_FILTER_DEBUG
                static int rejected_debug_count = 0;
                if (rejected_debug_count < 16) {
                    printf("Rejected turbulence node: x=%d y=%d z=%d r=%.8f bin=%d node_type=0x%08x\n",
                        x, y, z, radius, radial_bin, node_type);
                    ++rejected_debug_count;
                }
#endif
                continue;
            }
            ++shell_fluid_count[radial_bin];

            const double wall_distance = static_cast<double>(PIPE_RADIUS) - radius;
            if (radius < shell_min_radius[radial_bin]) {
                shell_min_radius[radial_bin] = radius;
            }
            if (radius > shell_max_radius[radial_bin]) {
                shell_max_radius[radial_bin] = radius;
            }
            if (wall_distance < shell_min_wall_distance[radial_bin]) {
                shell_min_wall_distance[radial_bin] = wall_distance;
            }
            if (wall_distance > shell_max_wall_distance[radial_bin]) {
                shell_max_wall_distance[radial_bin] = wall_distance;
            }
            shell_wall_distance[radial_bin] += wall_distance;

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

#ifdef OMEGA_FIELD
            const double local_omega = static_cast<double>(h_fMom[idxMom(
                x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M_OMEGA_INDEX,
                x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)]);
#else
            const double local_omega = base_omega;
#endif
            const double local_one_minus_half_omega = 1.0 - 0.5*local_omega;
            const double local_inverse_viscosity =
                local_omega > 0.0 ? 6.0*local_omega/(2.0-local_omega) : 0.0;

            const double ur = cos_theta*ux + sin_theta*uy;
            const double ut = -sin_theta*ux + cos_theta*uy;

            auto scalar_derivative = [&](int axis) {
                if (axis == 2) {
                    return 0.5*(density_at(x, y, z+1) - density_at(x, y, z-1));
                }
                const int dx_axis = axis == 0 ? 1 : 0;
                const int dy_axis = axis == 1 ? 1 : 0;
                const bool minus_fluid = is_fluid(x-dx_axis, y-dy_axis, z);
                const bool plus_fluid = is_fluid(x+dx_axis, y+dy_axis, z);
                if (minus_fluid && plus_fluid) {
                    return 0.5*(density_at(x+dx_axis, y+dy_axis, z) -
                        density_at(x-dx_axis, y-dy_axis, z));
                }
                if (plus_fluid) {
                    return density_at(x+dx_axis, y+dy_axis, z) - rho;
                }
                if (minus_fluid) {
                    return rho - density_at(x-dx_axis, y-dy_axis, z);
                }
                return 0.0;
            };
            auto velocity_derivative = [&](int component, int axis, double center) {
                if (axis == 2) {
                    return 0.5*(velocity_at(x, y, z+1, component) -
                        velocity_at(x, y, z-1, component));
                }
                const int dx_axis = axis == 0 ? 1 : 0;
                const int dy_axis = axis == 1 ? 1 : 0;
                const bool minus_fluid = is_fluid(x-dx_axis, y-dy_axis, z);
                const bool plus_fluid = is_fluid(x+dx_axis, y+dy_axis, z);
                if (minus_fluid && plus_fluid) {
                    return 0.5*(velocity_at(x+dx_axis, y+dy_axis, z, component) -
                        velocity_at(x-dx_axis, y-dy_axis, z, component));
                }
                if (plus_fluid) {
                    return velocity_at(x+dx_axis, y+dy_axis, z, component) - center;
                }
                if (minus_fluid) {
                    return center - velocity_at(x-dx_axis, y-dy_axis, z, component);
                }
                return 0.0;
            };

            const double pressure_gradient_x = scalar_derivative(0)/3.0;
            const double pressure_gradient_y = scalar_derivative(1)/3.0;
            const double pressure_gradient_z = scalar_derivative(2)/3.0;
            const double pressure_gradient_r =
                cos_theta*pressure_gradient_x + sin_theta*pressure_gradient_y;
            const double pressure_gradient_t =
                -sin_theta*pressure_gradient_x + cos_theta*pressure_gradient_y;

            double gradient_cartesian[3][3] = {};
            const double velocity_cartesian[3] = {ux, uy, uz};
            for (int component = 0; component < 3; ++component) {
                for (int direction = 0; direction < 3; ++direction) {
                    gradient_cartesian[component][direction] = velocity_derivative(
                        component, direction, velocity_cartesian[component]);
                }
            }
            const double rotation[3][3] = {
                {cos_theta, sin_theta, 0.0},
                {-sin_theta, cos_theta, 0.0},
                {0.0, 0.0, 1.0}
            };
            double gradient_cylindrical[3][3] = {};
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    for (int a = 0; a < 3; ++a) {
                        for (int b = 0; b < 3; ++b) {
                            gradient_cylindrical[i][j] += rotation[i][a]*
                                gradient_cartesian[a][b]*rotation[j][b];
                        }
                    }
                }
            }

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
            Sxx = (Sxx - local_omega*ux*ux -
                local_one_minus_half_omega*(2.0*ux*force_x)) / (1.0-local_omega);
            Sxy = (Sxy - local_omega*ux*uy -
                local_one_minus_half_omega*(ux*force_y + uy*force_x)) /
                (1.0-local_omega);
            Sxz = (Sxz - local_omega*ux*uz -
                local_one_minus_half_omega*(ux*force_z + uz*force_x)) /
                (1.0-local_omega);
            Syy = (Syy - local_omega*uy*uy -
                local_one_minus_half_omega*(2.0*uy*force_y)) / (1.0-local_omega);
            Syz = (Syz - local_omega*uy*uz -
                local_one_minus_half_omega*(uy*force_z + uz*force_y)) /
                (1.0-local_omega);
            Szz = (Szz - local_omega*uz*uz -
                local_one_minus_half_omega*(2.0*uz*force_z)) / (1.0-local_omega);

            Sxx = local_one_minus_half_omega*(
                Sxx - ux*ux + 0.5*(force_x*ux + ux*force_x));
            Sxy = local_one_minus_half_omega*(
                Sxy - ux*uy + 0.5*(force_x*uy + ux*force_y));
            Sxz = local_one_minus_half_omega*(
                Sxz - ux*uz + 0.5*(force_x*uz + ux*force_z));
            Syy = local_one_minus_half_omega*(
                Syy - uy*uy + 0.5*(force_y*uy + uy*force_y));
            Syz = local_one_minus_half_omega*(
                Syz - uy*uz + 0.5*(force_y*uz + uy*force_z));
            Szz = local_one_minus_half_omega*(
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
            const double density_delta = rho - static_cast<double>(RHO_0);
            shell_rho2[radial_bin] += density_delta*density_delta;
            shell_rho3[radial_bin] += density_delta*density_delta*density_delta;
            shell_rho4[radial_bin] +=
                density_delta*density_delta*density_delta*density_delta;
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
            shell_ur_ut_ut[radial_bin] += ur*ut*ut;
            shell_ur_uz_uz[radial_bin] += ur*uz*uz;
            shell_ur_ur_uz[radial_bin] += ur*ur*uz;
            shell_ut_ut_uz[radial_bin] += ut*ut*uz;
            shell_ur_pgr[radial_bin] += ur*pressure_gradient_r;
            shell_ut_pgt[radial_bin] += ut*pressure_gradient_t;
            shell_uz_pgz[radial_bin] += uz*pressure_gradient_z;
            shell_ur_pgz[radial_bin] += ur*pressure_gradient_z;
            shell_uz_pgr[radial_bin] += uz*pressure_gradient_r;
            for (int direction = 0; direction < 3; ++direction) {
                shell_grad_rr[radial_bin] +=
                    gradient_cylindrical[0][direction]*gradient_cylindrical[0][direction];
                shell_grad_tt[radial_bin] +=
                    gradient_cylindrical[1][direction]*gradient_cylindrical[1][direction];
                shell_grad_zz[radial_bin] +=
                    gradient_cylindrical[2][direction]*gradient_cylindrical[2][direction];
                shell_grad_rz[radial_bin] +=
                    gradient_cylindrical[0][direction]*gradient_cylindrical[2][direction];
            }
            shell_Srr[radial_bin] += Srr;
            shell_Stt[radial_bin] += Stt;
            shell_Szz[radial_bin] += Szz;
            shell_Srt[radial_bin] += Srt;
            shell_Srz[radial_bin] += Srz;
            shell_Stz[radial_bin] += Stz;
            shell_SGrr[radial_bin] += Srr*Srr*local_inverse_viscosity;
            shell_SGtt[radial_bin] += Stt*Stt*local_inverse_viscosity;
            shell_SGzz[radial_bin] += Szz*Szz*local_inverse_viscosity;
            shell_SGrt[radial_bin] += Srt*Srt*local_inverse_viscosity;
            shell_SGrz[radial_bin] += Srz*Srz*local_inverse_viscosity;
            shell_SGtz[radial_bin] += Stz*Stz*local_inverse_viscosity;
#ifdef NON_NEWTONIAN_FLUID
            if (local_omega < non_newtonian_omega_cutoff) {
                shell_yield_probability[radial_bin] += 1.0;
            } else {
                const double inverse_omega = 1.0/local_omega;
                shell_inv_omega[radial_bin] += inverse_omega;
                shell_inv_omega2[radial_bin] += inverse_omega*inverse_omega;
                ++shell_yielded_count[radial_bin];
            }
#endif
            ++shell_count[radial_bin];
        }
    }
}

for (int radial_bin = 0; radial_bin < N_RADIAL_BINS; ++radial_bin) {
    if (shell_candidate_count[radial_bin] !=
        shell_fluid_count[radial_bin] + shell_rejected_count[radial_bin]) {
        printf("ERROR: turbulence node-count mismatch in bin %d: candidate=%llu fluid=%llu rejected=%llu\n",
            radial_bin, shell_candidate_count[radial_bin],
            shell_fluid_count[radial_bin], shell_rejected_count[radial_bin]);
    }
    if (shell_count[radial_bin] != shell_fluid_count[radial_bin]) {
        printf("ERROR: turbulence normalization-count mismatch in bin %d: shell=%llu fluid=%llu\n",
            radial_bin, shell_count[radial_bin], shell_fluid_count[radial_bin]);
    }
    if (shell_rejected_count[radial_bin] !=
        shell_curved_bc_count[radial_bin] + shell_solid_count[radial_bin] +
        shell_other_node_count[radial_bin]) {
        printf("ERROR: turbulence rejection-category mismatch in bin %d\n", radial_bin);
    }
}

for (int radial_bin = 0; radial_bin < N_RADIAL_BINS; ++radial_bin) {
    if (shell_count[radial_bin] == 0) {
        continue;
    }
    const double inverse_count =
        1.0/static_cast<double>(shell_count[radial_bin]);
    shell_radius[radial_bin] *= inverse_count;
    shell_wall_distance[radial_bin] *= inverse_count;
    shell_rho[radial_bin] *= inverse_count;
    shell_rho2[radial_bin] *= inverse_count;
    shell_rho3[radial_bin] *= inverse_count;
    shell_rho4[radial_bin] *= inverse_count;
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
    shell_ur_ut_ut[radial_bin] *= inverse_count;
    shell_ur_uz_uz[radial_bin] *= inverse_count;
    shell_ur_ur_uz[radial_bin] *= inverse_count;
    shell_ut_ut_uz[radial_bin] *= inverse_count;
    shell_ur_pgr[radial_bin] *= inverse_count;
    shell_ut_pgt[radial_bin] *= inverse_count;
    shell_uz_pgz[radial_bin] *= inverse_count;
    shell_ur_pgz[radial_bin] *= inverse_count;
    shell_uz_pgr[radial_bin] *= inverse_count;
    shell_grad_rr[radial_bin] *= inverse_count;
    shell_grad_tt[radial_bin] *= inverse_count;
    shell_grad_zz[radial_bin] *= inverse_count;
    shell_grad_rz[radial_bin] *= inverse_count;
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
#ifdef NON_NEWTONIAN_FLUID
    shell_yield_probability[radial_bin] *= inverse_count;
    if (shell_yielded_count[radial_bin] > 0) {
        const double inverse_yielded_count =
            1.0/static_cast<double>(shell_yielded_count[radial_bin]);
        shell_inv_omega[radial_bin] *= inverse_yielded_count;
        shell_inv_omega2[radial_bin] *= inverse_yielded_count;
    }
#endif
}

printf("Turbulence fluid-node diagnostics (outermost 5 candidate bins):\n");
printf("  bin       mean_r      candidate       fluid    rejected  fluid_fraction          mean_Uz\n");
int candidate_diagnostic_bins_printed = 0;
for (int radial_bin = N_RADIAL_BINS - 1;
     radial_bin >= 0 && candidate_diagnostic_bins_printed < 5; --radial_bin) {
    if (shell_candidate_count[radial_bin] == 0) {
        continue;
    }
    const double fluid_fraction = static_cast<double>(shell_fluid_count[radial_bin]) /
        static_cast<double>(shell_candidate_count[radial_bin]);
    printf("%5d %12.6f %14llu %11llu %11llu %15.8f %16.8e\n",
        radial_bin, shell_radius[radial_bin], shell_candidate_count[radial_bin],
        shell_fluid_count[radial_bin], shell_rejected_count[radial_bin],
        fluid_fraction, shell_uz[radial_bin]);
    ++candidate_diagnostic_bins_printed;
}

printf("Turbulence fluid-node radial-range diagnostics (outermost 5 populated fluid bins):\n");
printf("  bin       fluid        r_min       r_mean        r_max        y_min       y_mean        y_max          mean_Uz\n");
int diagnostic_bins_printed = 0;
for (int radial_bin = N_RADIAL_BINS - 1;
     radial_bin >= 0 && diagnostic_bins_printed < 5; --radial_bin) {
    if (shell_fluid_count[radial_bin] == 0) {
        continue;
    }
    printf("%5d %11llu %12.6f %12.6f %12.6f %12.6f %12.6f %12.6f %16.8e\n",
        radial_bin, shell_fluid_count[radial_bin],
        shell_min_radius[radial_bin], shell_radius[radial_bin],
        shell_max_radius[radial_bin], shell_min_wall_distance[radial_bin],
        shell_wall_distance[radial_bin], shell_max_wall_distance[radial_bin],
        shell_uz[radial_bin]);
    ++diagnostic_bins_printed;
}

for (int radial_bin = N_RADIAL_BINS - 1; radial_bin >= 0; --radial_bin) {
    if (shell_candidate_count[radial_bin] > 0 && shell_fluid_count[radial_bin] == 0) {
        printf("  bin %d has %llu candidates but no populated fluid samples\n",
            radial_bin, shell_candidate_count[radial_bin]);
    }
}

bool mean_uz_monotonic_inward = true;
bool have_outer_mean_uz = false;
double previous_outer_mean_uz = 0.0;
int monotonic_bins_checked = 0;
for (int radial_bin = N_RADIAL_BINS - 1;
     radial_bin >= 0 && monotonic_bins_checked < 5; --radial_bin) {
    if (shell_fluid_count[radial_bin] == 0) {
        continue;
    }
    if (have_outer_mean_uz && shell_uz[radial_bin] < previous_outer_mean_uz) {
        mean_uz_monotonic_inward = false;
    }
    previous_outer_mean_uz = shell_uz[radial_bin];
    have_outer_mean_uz = true;
    ++monotonic_bins_checked;
}
printf("Mean Uz monotonic moving inward across these %d populated bins: %s\n",
    monotonic_bins_checked, mean_uz_monotonic_inward ? "yes" : "no");

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

auto save_radial_count = [step](const char* suffix,
    const unsigned long long* values) {
    std::ostringstream data;
    data << "step," << step;
    for (int radial_bin = 0; radial_bin < N_RADIAL_BINS; ++radial_bin) {
        data << "," << values[radial_bin];
    }
    saveTreatData(suffix, data.str(), step);
};

double shell_fluid_fraction[N_RADIAL_BINS] = {};
for (int radial_bin = 0; radial_bin < N_RADIAL_BINS; ++radial_bin) {
    if (shell_candidate_count[radial_bin] > 0) {
        shell_fluid_fraction[radial_bin] =
            static_cast<double>(shell_fluid_count[radial_bin]) /
            static_cast<double>(shell_candidate_count[radial_bin]);
    }
}

save_radial_count("_turbulent_candidate_count", shell_candidate_count);
save_radial_count("_turbulent_fluid_count", shell_fluid_count);
save_radial_count("_turbulent_rejected_count", shell_rejected_count);
save_radial_count("_turbulent_curved_bc_count", shell_curved_bc_count);
save_radial_count("_turbulent_solid_count", shell_solid_count);
save_radial_count("_turbulent_other_node_count", shell_other_node_count);
save_radial_profile("_turbulent_fluid_fraction", shell_fluid_fraction);

save_radial_profile("_turbulent_radius", shell_radius);
save_radial_profile("_turbulent_rho", shell_rho);
save_radial_profile("_turbulent_rho2", shell_rho2);
save_radial_profile("_turbulent_rho3", shell_rho3);
save_radial_profile("_turbulent_rho4", shell_rho4);
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
save_radial_profile("_turbulent_ur_ut_ut", shell_ur_ut_ut);
save_radial_profile("_turbulent_ur_uz_uz", shell_ur_uz_uz);
save_radial_profile("_turbulent_ur_ur_uz", shell_ur_ur_uz);
save_radial_profile("_turbulent_ut_ut_uz", shell_ut_ut_uz);
save_radial_profile("_turbulent_ur_pgr", shell_ur_pgr);
save_radial_profile("_turbulent_ut_pgt", shell_ut_pgt);
save_radial_profile("_turbulent_uz_pgz", shell_uz_pgz);
save_radial_profile("_turbulent_ur_pgz", shell_ur_pgz);
save_radial_profile("_turbulent_uz_pgr", shell_uz_pgr);
save_radial_profile("_turbulent_grad_rr", shell_grad_rr);
save_radial_profile("_turbulent_grad_tt", shell_grad_tt);
save_radial_profile("_turbulent_grad_zz", shell_grad_zz);
save_radial_profile("_turbulent_grad_rz", shell_grad_rz);
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
#ifdef NON_NEWTONIAN_FLUID
save_radial_profile("_turbulent_yield_prob", shell_yield_probability);
save_radial_profile("_turbulent_inv_omega", shell_inv_omega);
save_radial_profile("_turbulent_inv_omega_2", shell_inv_omega2);
#endif
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
