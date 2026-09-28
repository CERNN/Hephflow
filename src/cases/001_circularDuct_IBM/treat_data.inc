// Circular Poiseuille validation over the complete physical IBM duct.
static_assert(DUCT_RADIUS_A == DUCT_RADIUS_B,
    "The analytical validation below is for a circular duct.");

auto ductMoment = [&](int px, int py, int pz, int moment) -> double {
    return static_cast<double>(h_fMom[idxMom(
        px % BLOCK_NX, py % BLOCK_NY, pz % BLOCK_NZ, moment,
        px / BLOCK_NX, py / BLOCK_NY, pz / BLOCK_NZ)]);
};
auto ductUz = [&](int px, int py, int pz) -> double {
    return ductMoment(px, py, pz, M_UZ_INDEX) /
        static_cast<double>(F_M_I_SCALE);
};

constexpr int ductBins = static_cast<int>(DUCT_RADIUS) + 1;
double radiusSum[ductBins] = {};
double uzSum[ductBins] = {};
double uzExactSum[ductBins] = {};
double urSum[ductBins] = {};
unsigned long long binCount[ductBins] = {};

double fullErrorSq = 0.0, fullExactSq = 0.0, fullLinf = 0.0;
double bulkErrorSq = 0.0, bulkExactSq = 0.0, bulkLinf = 0.0;
double wallErrorSq = 0.0, wallExactSq = 0.0, wallLinf = 0.0;
double flowRateSum = 0.0;
double maxRadialVelocity = 0.0;
double maxNearWallShearError = 0.0;
unsigned long long physicalCount = 0, bulkCount = 0, wallCount = 0;

const double radius = static_cast<double>(DUCT_RADIUS);
const double radiusSquared = radius * radius;
const double viscosity = static_cast<double>(RHO_0 * VISC);
const double driving = static_cast<double>(DUCT_DRIVING_FORCE);
const double centerlineExact = driving * radiusSquared / (4.0 * viscosity);

for (int pz = 0; pz < NZ_TOTAL; ++pz) {
    for (int py = 0; py < NY; ++py) {
        for (int px = 0; px < NX; ++px) {
            const double dx = static_cast<double>(px) - DUCT_CENTER_X;
            const double dy = static_cast<double>(py) - DUCT_CENTER_Y;
            const double rSquared = dx * dx + dy * dy;
            if (rSquared >= radiusSquared) continue;

            const double r = sqrt(rSquared);
            const double ux = ductMoment(px, py, pz, M_UX_INDEX) / F_M_I_SCALE;
            const double uy = ductMoment(px, py, pz, M_UY_INDEX) / F_M_I_SCALE;
            const double uz = ductUz(px, py, pz);
            const double uzExact = driving * (radiusSquared - rSquared) /
                (4.0 * viscosity);
            const double error = uz - uzExact;
            const double absError = fabs(error);
            const double ur = r > 0.0 ? (dx * ux + dy * uy) / r : 0.0;

            fullErrorSq += error * error;
            fullExactSq += uzExact * uzExact;
            fullLinf = fmax(fullLinf, absError);
            flowRateSum += uz;
            maxRadialVelocity = fmax(maxRadialVelocity, fabs(ur));
            ++physicalCount;

            if (r < radius - 2.0) {
                bulkErrorSq += error * error;
                bulkExactSq += uzExact * uzExact;
                bulkLinf = fmax(bulkLinf, absError);
                ++bulkCount;
            } else {
                wallErrorSq += error * error;
                wallExactSq += uzExact * uzExact;
                wallLinf = fmax(wallLinf, absError);
                ++wallCount;

                // This deliberately measures the solver's present centered
                // stencil across the immersed wall; it exposes the need for
                // the later IBM-aware velocity reconstruction.
                const int xm = (px - 1 + NX) % NX;
                const int xp = (px + 1) % NX;
                const int ym = (py - 1 + NY) % NY;
                const int yp = (py + 1) % NY;
                const double duzdx = 0.5 * (ductUz(xp, py, pz) - ductUz(xm, py, pz));
                const double duzdy = 0.5 * (ductUz(px, yp, pz) - ductUz(px, ym, pz));
                const double radialShear = (dx * duzdx + dy * duzdy) / r;
                const double exactShear = -driving * r / (2.0 * viscosity);
                maxNearWallShearError = fmax(
                    maxNearWallShearError, fabs(radialShear - exactShear));
            }

            const int bin = std::min(ductBins - 1, static_cast<int>(r));
            radiusSum[bin] += r;
            uzSum[bin] += uz;
            uzExactSum[bin] += uzExact;
            urSum[bin] += ur;
            ++binCount[bin];
        }
    }
}

double centerlineNumerical = 0.0;
const int cx0 = static_cast<int>(floor(DUCT_CENTER_X));
const int cy0 = static_cast<int>(floor(DUCT_CENTER_Y));
for (int pz = 0; pz < NZ_TOTAL; ++pz) {
    centerlineNumerical += 0.25 * (
        ductUz(cx0, cy0, pz) + ductUz(cx0 + 1, cy0, pz) +
        ductUz(cx0, cy0 + 1, pz) + ductUz(cx0 + 1, cy0 + 1, pz));
}
centerlineNumerical /= static_cast<double>(NZ_TOTAL);

const double numericalQ = flowRateSum / static_cast<double>(NZ_TOTAL);
const double exactQ = M_PI * driving * radiusSquared * radiusSquared /
    (8.0 * viscosity);
const double epsilon = 1.0e-30;
std::ostringstream summary;
summary << std::scientific << std::setprecision(12)
    << "newtonian_ibm_validation," << step
    << ",full_l2," << sqrt(fullErrorSq / fmax(fullExactSq, epsilon))
    << ",full_linf," << fullLinf
    << ",bulk_l2," << sqrt(bulkErrorSq / fmax(bulkExactSq, epsilon))
    << ",bulk_linf," << bulkLinf
    << ",wall_l2," << sqrt(wallErrorSq / fmax(wallExactSq, epsilon))
    << ",wall_linf," << wallLinf
    << ",centerline," << centerlineNumerical
    << ",centerline_exact," << centerlineExact
    << ",q," << numericalQ << ",q_exact," << exactQ
    << ",q_relative_error," << fabs(numericalQ - exactQ) / exactQ
    << ",max_abs_ur," << maxRadialVelocity
    << ",max_near_wall_shear_error," << maxNearWallShearError
    << ",physical_nodes," << physicalCount
    << ",bulk_nodes," << bulkCount << ",wall_nodes," << wallCount;
saveTreatData("_newtonian_ibm_validation", summary.str(), step);

std::ostringstream profile;
profile << std::scientific << std::setprecision(12) << step;
for (int bin = 0; bin < ductBins; ++bin) {
    if (binCount[bin] == 0) continue;
    const double inverseCount = 1.0 / static_cast<double>(binCount[bin]);
    profile << "," << radiusSum[bin] * inverseCount / radius
            << "," << uzSum[bin] * inverseCount
            << "," << uzExactSum[bin] * inverseCount
            << "," << urSum[bin] * inverseCount;
}
saveTreatData("_newtonian_ibm_radial_profile", profile.str(), step);
